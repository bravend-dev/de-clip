"""The vectorised aggregation must match a literal per-class implementation of Algorithm 1."""

import pytest
import torch
import torch.nn.functional as F

from declip.aggregate import build_class_embeddings
from declip.config import AggregateConfig, SelectionConfig


def reference_class_embeddings(embeddings, cfg: AggregateConfig, m: int) -> torch.Tensor:
    """Per-class loop written straight from the paper's equations (oracle)."""
    embeddings = F.normalize(embeddings.float(), dim=-1)
    labels = F.normalize(embeddings[:, :m].mean(1), dim=-1)
    out = []
    for i in range(embeddings.shape[0]):
        anchor = labels[i]
        cands = embeddings[i, m:]
        if cfg.alpha >= 1.0:
            out.append(anchor)
            continue
        scores = cands @ anchor  # Eq. (7): anchor similarity
        order = torch.argsort(scores, descending=True)
        cands, scores = cands[order], scores[order]
        if cfg.selection.type == "grouping":
            vecs, sc, remaining = [], [], torch.ones(len(cands), dtype=torch.bool)
            while remaining.any():
                idx = torch.nonzero(remaining).squeeze(1)
                leader = idx[0]
                sims = cands[remaining] @ cands[leader]
                dupe = sims >= cfg.selection.tau
                vecs.append(F.normalize(cands[remaining][dupe].mean(0), dim=0))
                sc.append(scores[leader])
                remaining[idx[dupe]] = False
            cands, scores = torch.stack(vecs), torch.stack(sc)
        elif cfg.selection.type == "top_h":
            cands, scores = cands[: cfg.selection.top_h], scores[: cfg.selection.top_h]
        if cfg.weighting == "shifted_linear":
            pos = scores - scores.min()
            w = pos / pos.sum() if pos.sum() > 1e-6 else torch.full_like(pos, 1.0 / len(pos))
        else:
            w = torch.full_like(scores, 1.0 / len(scores))
        desc = F.normalize((cands * w[:, None]).sum(0), dim=0)
        out.append(F.normalize(cfg.alpha * anchor + (1 - cfg.alpha) * desc, dim=0))
    return torch.stack(out)


def _random_embeddings(seed=0, num_classes=6, m=8, n=60, d=32):
    gen = torch.Generator().manual_seed(seed)
    base = torch.randn(num_classes, 1, d, generator=gen)
    noise = torch.randn(num_classes, m + n, d, generator=gen)
    x = base + 0.7 * noise
    # plant exact duplicates so grouping has something to merge
    x[:, m + 1] = x[:, m]
    x[:, m + 5] = x[:, m + 4] * 1.3
    return x


CONFIGS = [
    AggregateConfig(),
    AggregateConfig(alpha=1.0),
    AggregateConfig(alpha=0.0),
    AggregateConfig(selection=SelectionConfig(type="none"), weighting="uniform"),
    AggregateConfig(selection=SelectionConfig(type="none")),
    AggregateConfig(weighting="uniform"),
    AggregateConfig(selection=SelectionConfig(type="grouping", tau=0.8)),
    AggregateConfig(selection=SelectionConfig(type="top_h", top_h=7), weighting="uniform"),
    AggregateConfig(  # D3 of Table 5: grouping with uniform weights
        selection=SelectionConfig(type="grouping", tau=0.8), weighting="uniform"
    ),
]


@pytest.mark.parametrize(
    "cfg", CONFIGS, ids=lambda c: f"{c.selection.type}-{c.weighting}-a{c.alpha}"
)
def test_matches_reference(cfg):
    for seed in range(3):
        x = _random_embeddings(seed)
        fast = build_class_embeddings(x, cfg, 8)
        slow = reference_class_embeddings(x, cfg, 8)
        assert torch.allclose(fast, slow, atol=1e-5), (fast - slow).abs().max()
        assert torch.allclose(fast.norm(dim=-1), torch.ones(x.shape[0]), atol=1e-5)


def test_default_reduces_to_anchor_when_alpha_is_one():
    x = _random_embeddings()
    anchor = F.normalize(F.normalize(x[:, :8], dim=-1).mean(1), dim=-1)
    assert torch.allclose(
        build_class_embeddings(x, AggregateConfig(alpha=1.0), 8), anchor, atol=1e-6
    )


def test_identical_scores_fall_back_to_uniform_weights():
    x = _random_embeddings()
    x[:, 8:] = x[:, 8:9]  # every candidate identical -> one group, zero shifted score
    cfg = AggregateConfig(selection=SelectionConfig(type="none"))
    out = build_class_embeddings(x, cfg, 8)
    assert torch.isfinite(out).all()


def test_padding_candidates_are_excluded():
    """A class with fewer descriptions, cycled to K, must equal the plain mean of its real ones."""
    from declip.config import SelectionConfig, TemplatesConfig
    from declip.descriptions import candidate_mask, render_prompts

    torch.manual_seed(1)
    templates = TemplatesConfig(label=["{c}"], description=["{d}"], class_name_candidate=[])
    pool = {"a": ["p", "q", "r", "s"], "b": ["u", "v"]}
    k = 4
    prompts = [render_prompts(c, pool, templates, k) for c in ("a", "b")]
    assert prompts[1][1:] == ["u.", "v.", "u.", "v."]
    mask = candidate_mask(["a", "b"], pool, templates, k)
    assert mask.tolist() == [[True] * 4, [True, True, False, False]]
    vec = {t: F.normalize(torch.randn(8), dim=0) for t in {p for ps in prompts for p in ps}}
    x = torch.stack([torch.stack([vec[p] for p in ps]) for ps in prompts])
    cfg = AggregateConfig(
        alpha=0.0,
        selection=SelectionConfig(type="none"),
        weighting="uniform",
        renormalize=False,
    )
    out = build_class_embeddings(x, cfg, 1, mask)
    expected_b = (vec["u."] + vec["v."]) / 2
    assert torch.allclose(out[1], expected_b, atol=1e-6)
