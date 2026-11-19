"""Frozen-encoder passes: images of a dataset and the prompts of a class list."""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from declip.backbones import Backbone
from declip.cache import ImageEmbeddingCache, TextEmbeddingStore, file_sha1
from declip.config import REPO_ROOT, TemplatesConfig
from declip.datasets import DatasetSpec, make_loader
from declip.descriptions import Pool, render_prompts

log = logging.getLogger(__name__)


@torch.no_grad()
def encode_images(
    backbone: Backbone,
    spec: DatasetSpec,
    cache: ImageEmbeddingCache,
    batch_size: int = 256,
    workers: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Un-normalised image embeddings (N, D) and labels (N,) in CSV order."""
    key = cache.build_key(
        backbone.cache_key(),
        backbone.image_size,
        spec.name,
        file_sha1(os.path.join(REPO_ROOT, spec.csv_path)),
    )
    cached = cache.load(key)
    if cached is not None:
        return cached
    loader = make_loader(spec, backbone.preprocess, batch_size, workers)
    features, labels = [], []
    for images, targets in tqdm(loader, desc=f"encode {spec.name}", unit="batch"):
        images = images.to(backbone.device, non_blocking=True)
        with backbone.autocast:
            emb = backbone.model.encode_image(images)
        features.append(emb.cpu().numpy())
        labels.append(targets.numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels).astype(np.int64)
    cache.save(key, features, labels)
    return features, labels


@torch.no_grad()
def encode_texts(
    backbone: Backbone, texts: list[str], store: TextEmbeddingStore, batch_size: int = 512
) -> torch.Tensor:
    """Un-normalised text embeddings (N, D) in float32, in input order.

    Text is encoded under the same autocast policy as images (fp16 on GPU for ``amp``).
    """

    def _encode(batch_texts: list[str]) -> torch.Tensor:
        out = []
        for start in range(0, len(batch_texts), batch_size):
            tokens = backbone.tokenizer(batch_texts[start : start + batch_size]).to(backbone.device)
            with backbone.autocast:
                out.append(backbone.model.encode_text(tokens, normalize=False))
        return torch.cat(out).float()

    return store.encode(texts, _encode).float()


def encode_class_prompts(
    backbone: Backbone,
    classnames: list[str],
    pool: Pool,
    templates: TemplatesConfig,
    k: int,
    store: TextEmbeddingStore,
) -> torch.Tensor:
    """Prompt embeddings of shape (num_classes, num_prompts, D); see :mod:`declip.descriptions`."""
    prompts = [p for c in classnames for p in render_prompts(c, pool, templates, k)]
    per_class = len(prompts) // len(classnames)
    embeddings = encode_texts(backbone, prompts, store)
    return embeddings.reshape(len(classnames), per_class, -1)
