"""Class text embedding construction (Sec. 3.3, 3.5, 3.6; Algorithm 1).

Input: the raw prompt embeddings of every class, shape ``(C, P, D)``, whose first ``M`` rows
per class are the label prompts and the remaining ``N = P - M`` rows are the candidate
description prompts. Output: one unit-norm text embedding ``T_i`` per class.

Every function here is pure tensor arithmetic; the reference per-class implementation in
``tests/test_aggregate.py`` is the correctness oracle for the vectorised code.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from declip.config import AggregateConfig


def fuse_label_embeddings(embeddings: torch.Tensor, num_label_prompts: int) -> torch.Tensor:
    """Eq. (4)-(5): L_i = norm(mean_m norm(E_txt(P^L_m(y_i)))). Returns (C, D)."""
    label = F.normalize(embeddings[:, :num_label_prompts, :], dim=-1)
    return F.normalize(label.mean(dim=1), dim=-1)


def candidate_embeddings(embeddings: torch.Tensor, num_label_prompts: int) -> torch.Tensor:
    """D_{i,k,r}: unit-norm candidates, shape (C, N, D)."""
    return F.normalize(embeddings[:, num_label_prompts:, :], dim=-1)


def anchor_similarity_scores(candidates: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Eq. (7): s_{i,k,r} = D_{i,k,r} . L_i, similarity to the own class anchor. Returns (C, N)."""
    return torch.einsum("cnd,cd->cn", candidates, labels)


def group_redundant(sorted_candidates: torch.Tensor, tau: float, class_chunk: int | None = None):
    """Greedy redundancy grouping (Sec. 3.5.2) on candidates sorted by decreasing score.

    Returns ``(representatives, valid)``: representatives ``(C, N, D)`` hold the unit-norm mean
    of each group at the slot of its leader (Eq. 8); ``valid`` ``(C, N)`` marks leader slots.
    The leader's score is inherited by the group because it stays at the leader's slot.
    """
    C, N, D = sorted_candidates.shape
    device = sorted_candidates.device
    if class_chunk is None:
        class_chunk = max(1, (512 * 1024**2) // max(1, N * N * 4))
    reps = torch.zeros_like(sorted_candidates)
    valid = torch.zeros(C, N, dtype=torch.bool, device=device)
    for lo in range(0, C, class_chunk):
        hi = min(lo + class_chunk, C)
        reps[lo:hi], valid[lo:hi] = _group_chunk(sorted_candidates[lo:hi], tau)
    return reps, valid


def _group_chunk(x: torch.Tensor, tau: float):
    B, N, _ = x.shape
    device = x.device
    close = torch.matmul(x, x.transpose(1, 2)) >= tau  # (B, N, N)
    reps = torch.zeros_like(x)
    valid = torch.zeros(B, N, dtype=torch.bool, device=device)

    # Candidates without any neighbour above tau are singleton groups whatever the order.
    lone = close.sum(-1) <= 1
    reps[lone] = F.normalize(x[lone], dim=-1)
    valid |= lone

    remaining = ~lone
    slots = torch.arange(N, device=device)
    rows = torch.arange(B, device=device)
    while bool(remaining.any()):
        leader = torch.where(remaining, slots, N).amin(dim=1)  # lowest remaining slot = best score
        active = leader < N
        leader = leader.clamp(max=N - 1)
        members = close[rows, leader] & remaining & active.unsqueeze(1)
        count = members.sum(1, keepdim=True).clamp(min=1)
        merged = F.normalize((x * members.unsqueeze(-1)).sum(1) / count, dim=-1)
        reps[active, leader[active]] = merged[active].to(reps.dtype)
        valid[active, leader[active]] = True
        remaining &= ~members
    return reps, valid


def shifted_linear_weights(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Eq. (9): omega_g = (s_g - s_min) / sum_j (s_j - s_min); uniform if all shifts are zero."""
    big = torch.finfo(scores.dtype).max
    s_min = torch.where(valid, scores, torch.full_like(scores, big)).amin(dim=1, keepdim=True)
    shifted = (scores - s_min) * valid
    total = shifted.sum(dim=1, keepdim=True)
    return torch.where(
        total > 1e-6, shifted / total.clamp(min=1e-12), uniform_weights(valid, scores.dtype)
    )


def uniform_weights(valid: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    return valid.to(dtype) / valid.sum(dim=1, keepdim=True).clamp(min=1)


def _score_and_select(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    cfg: AggregateConfig,
    num_label_prompts: int,
    candidate_valid: torch.Tensor | None,
):
    """Steps 1-2 of Algorithm 1: scores, sorted candidates, grouping / selection."""
    candidates = candidate_embeddings(embeddings, num_label_prompts)
    C, N, D = candidates.shape

    # 1. anchor-similarity scoring
    scores = anchor_similarity_scores(candidates, labels)

    if candidate_valid is None:
        candidate_valid = torch.ones(C, N, dtype=torch.bool, device=candidates.device)
    candidate_valid = candidate_valid.to(candidates.device)
    scores = torch.where(candidate_valid, scores, torch.full_like(scores, float("-inf")))

    # 2. sort by decreasing score (padding last), then select / group
    order = torch.argsort(scores, dim=1, descending=True)
    sorted_candidates = torch.gather(candidates, 1, order.unsqueeze(-1).expand(-1, -1, D))
    sorted_scores = torch.gather(scores, 1, order)
    sorted_valid = torch.gather(candidate_valid, 1, order)
    sorted_scores = torch.where(sorted_valid, sorted_scores, torch.zeros_like(sorted_scores))
    sorted_candidates = sorted_candidates * sorted_valid.unsqueeze(-1)  # padding cannot be grouped
    if cfg.selection.type == "grouping":
        vectors, valid = group_redundant(sorted_candidates, cfg.selection.tau)
    elif cfg.selection.type == "top_h":
        h = max(1, min(int(cfg.selection.top_h), N))
        vectors = sorted_candidates
        valid = torch.zeros(C, N, dtype=torch.bool, device=candidates.device)
        valid[:, :h] = True
    else:
        vectors = sorted_candidates
        valid = torch.ones(C, N, dtype=torch.bool, device=candidates.device)
    valid &= sorted_valid
    return sorted_scores, vectors, valid


def build_class_embeddings(
    embeddings: torch.Tensor,
    cfg: AggregateConfig,
    num_label_prompts: int,
    candidate_valid: torch.Tensor | None = None,
    memo: dict | None = None,
) -> torch.Tensor:
    """Algorithm 1: from raw prompt embeddings (C, P, D) to final text embeddings T (C, D).

    ``candidate_valid`` (C, N) marks the real candidates of each class when the classes have
    different numbers of descriptions (see :func:`declip.descriptions.candidate_mask`); the
    padding candidates are then excluded from scoring, grouping and aggregation.

    ``memo`` (one dict per prompt tensor) caches steps 1-2, which depend only on the selection
    settings, so grid points that differ in alpha or weighting reuse the grouping.
    """
    embeddings = embeddings.float()
    labels = fuse_label_embeddings(embeddings, num_label_prompts)
    if cfg.alpha >= 1.0:
        return labels  # B0 / Default: class-name anchor only
    key = (cfg.selection.type, cfg.selection.tau, cfg.selection.top_h)
    if memo is not None and key in memo:
        sorted_scores, vectors, valid = memo[key]
    else:
        sorted_scores, vectors, valid = _score_and_select(
            embeddings, labels, cfg, num_label_prompts, candidate_valid
        )
        if memo is not None:
            memo[key] = (sorted_scores, vectors, valid)

    # 3. weighting and aggregation, Eq. (9)-(10)
    if cfg.weighting == "shifted_linear":
        weights = shifted_linear_weights(sorted_scores, valid)
    else:
        weights = uniform_weights(valid, sorted_scores.dtype)
    descriptions = (vectors * weights.unsqueeze(-1)).sum(dim=1)
    if not cfg.renormalize:
        # DCLIP / WaffleCLIP score-mean baseline: class score = mean cosine over the prompts,
        # which is the unnormalised mean of the unit prompt embeddings (alpha = 0 required).
        return descriptions
    descriptions = F.normalize(descriptions, dim=-1)

    # 4. blend with the fused label embedding, Eq. (11)
    return F.normalize(cfg.alpha * labels + (1.0 - cfg.alpha) * descriptions, dim=-1)
