"""Zero-shot prediction (Eq. 3) and the metrics reported in the paper (Sec. 4.2)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)

from declip.datasets import TOP_K

LOGIT_SCALE = 100.0


@torch.no_grad()
def evaluate(
    image_features: np.ndarray,
    labels: np.ndarray,
    class_embeddings: torch.Tensor,
    metric_set: str,
) -> dict[str, float]:
    """Metrics of the cosine classifier defined by ``class_embeddings`` (C, D)."""
    features = torch.from_numpy(np.asarray(image_features)).to(class_embeddings.device)
    features = torch.nn.functional.normalize(features.to(class_embeddings.dtype), dim=-1)
    logits = LOGIT_SCALE * features @ class_embeddings.t()
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    predictions = probs.argmax(axis=1)
    labels = np.asarray(labels)

    if metric_set == TOP_K:
        return {
            "top1_accuracy": accuracy_score(labels, predictions),
            "top5_accuracy": top_k_accuracy_score(
                labels, probs, k=5, labels=np.arange(probs.shape[1])
            ),
        }
    one_hot = np.eye(probs.shape[1])[labels]
    return {
        "auroc": roc_auc_score(one_hot, probs, multi_class="ovr", average="macro"),
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
    }
