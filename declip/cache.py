"""On-disk caches for image and text embeddings.

Encoding 122k images per backbone takes minutes; the aggregation step takes seconds.
Both caches live at the raw-embedding layer, before any aggregation, so every ablation
in the paper re-uses the same embeddings.

* Image embeddings: one ``.npz`` per (backbone weights, precision, image size, dataset CSV
  content). Row order equals CSV order; labels are stored alongside as a consistency check.
* Text embeddings: content-addressed, ``sha1(prompt) -> vector``, one store per backbone.
  Changing a description file re-encodes only the prompts that are actually new.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable

import numpy as np
import torch

log = logging.getLogger(__name__)

CACHE_VERSION = 1


def hash_key(key: dict) -> str:
    return hashlib.md5(json.dumps(key, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def file_sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ImageEmbeddingCache:
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.dir = os.path.join(cache_dir, "image_embeddings")
        self.enabled = enabled

    @staticmethod
    def build_key(backbone_key: dict, image_size, dataset_name: str, csv_sha1: str) -> dict:
        return {
            "version": CACHE_VERSION,
            "kind": "image",
            "backbone": backbone_key,
            "image_size": list(image_size),
            "dataset": {"name": dataset_name, "csv_sha1": csv_sha1},
        }

    def _paths(self, key: dict) -> tuple[str, str]:
        h = hash_key(key)
        return os.path.join(self.dir, f"{h}.npz"), os.path.join(self.dir, f"{h}.json")

    def load(self, key: dict):
        if not self.enabled:
            return None
        npz_path, _ = self._paths(key)
        if not os.path.exists(npz_path):
            return None
        with np.load(npz_path) as data:
            features, labels = data["features"], data["labels"]
        log.info("image cache hit: %s (%d rows)", key["dataset"]["name"], len(labels))
        return features, labels

    def save(self, key: dict, features: np.ndarray, labels: np.ndarray) -> None:
        if not self.enabled:
            return
        npz_path, meta_path = self._paths(key)
        os.makedirs(self.dir, exist_ok=True)
        tmp = npz_path + ".tmp.npz"
        np.savez(tmp, features=features, labels=np.asarray(labels, dtype=np.int64))
        os.replace(tmp, npz_path)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(key, handle, indent=2, sort_keys=True)


class TextEmbeddingStore:
    """``sha1(text) -> embedding`` for one backbone.

    On disk: ``<hash>.keys.npy`` (sorted sha1 hex digests, ``S40``), ``<hash>.emb.npy`` (float32
    rows in the same order, memory-mapped on load so a run reads only the rows it uses) and
    ``<hash>.texts.json`` (digest -> prompt, for inspection).
    """

    def __init__(self, cache_dir: str, backbone_key: dict, enabled: bool = True):
        self.enabled = enabled
        self.dir = os.path.join(cache_dir, "text_embeddings")
        self.model_hash = hash_key(
            {"version": CACHE_VERSION, "kind": "text", "backbone": backbone_key}
        )
        self._keys: np.ndarray = np.empty(0, dtype="S40")
        self._matrix: np.ndarray | None = None
        self._new: dict[bytes, tuple[str, np.ndarray]] = {}
        if enabled:
            self._keys, self._matrix = self._read_disk()
            log.info("text cache loaded: %d prompts", len(self._keys))

    def _path(self, suffix: str) -> str:
        return os.path.join(self.dir, f"{self.model_hash}{suffix}")

    def _read_disk(self) -> tuple[np.ndarray, np.ndarray | None]:
        keys_path, emb_path = self._path(".keys.npy"), self._path(".emb.npy")
        for _ in range(5):  # a concurrent flush replaces the two files one after the other
            if not os.path.exists(keys_path):
                break
            keys = np.load(keys_path)
            matrix = np.load(emb_path, mmap_mode="r")
            if len(keys) == matrix.shape[0]:
                return keys, matrix
        return np.empty(0, dtype="S40"), None

    def _lookup(self, keys: np.ndarray) -> np.ndarray:
        """Row of every key in the sorted on-disk index, -1 when absent."""
        if len(self._keys) == 0:
            return np.full(len(keys), -1)
        pos = np.searchsorted(self._keys, keys).clip(max=len(self._keys) - 1)
        return np.where(self._keys[pos] == keys, pos, -1)

    def encode(
        self, texts: list[str], encode_fn: Callable[[list[str]], torch.Tensor]
    ) -> torch.Tensor:
        """Embeddings for ``texts`` in order; only unseen prompts go through ``encode_fn``."""
        if not self.enabled:
            return encode_fn(texts)
        keys = np.asarray([hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts], dtype="S40")
        rows = self._lookup(keys)
        missing: dict[bytes, str] = {}
        for key, text, row in zip(keys.tolist(), texts, rows.tolist()):
            if row < 0 and key not in self._new and key not in missing:
                missing[key] = text
        if missing:
            new = encode_fn(list(missing.values())).detach().float().cpu().numpy()
            for (key, text), emb in zip(missing.items(), new):
                self._new[key] = (text, emb)
        log.info("text cache: %d prompts, %d newly encoded", len(texts), len(missing))
        if self._matrix is not None:
            dim = self._matrix.shape[1]
        else:
            dim = len(next(iter(self._new.values()))[1])
        out = np.empty((len(texts), dim), dtype=np.float32)
        hit = rows >= 0
        if hit.any():
            order = np.argsort(rows[hit])  # read the memory map in increasing row order
            idx = np.flatnonzero(hit)[order]
            out[idx] = self._matrix[rows[hit][order]]
        for i in np.flatnonzero(~hit):
            out[i] = self._new[bytes(keys[i])][1]
        return torch.from_numpy(out)

    def flush(self) -> None:
        """Write new entries to disk, merging with whatever another process saved meanwhile."""
        if not self.enabled or not self._new:
            return
        os.makedirs(self.dir, exist_ok=True)
        disk_keys, disk_matrix = self._read_disk()  # entries another process saved meanwhile win
        if disk_matrix is None:
            disk_keys, disk_matrix = self._keys, self._matrix
        present = set(disk_keys.tolist())
        new_items = [(k, v) for k, v in self._new.items() if k not in present]
        parts_k, parts_m = [disk_keys], []
        if disk_matrix is not None:
            parts_m.append(np.asarray(disk_matrix))
        if new_items:
            parts_k.append(np.asarray([k for k, _ in new_items], dtype="S40"))
            parts_m.append(np.stack([e for _, (_, e) in new_items]).astype(np.float32))
        keys = np.concatenate(parts_k)
        matrix = np.concatenate(parts_m)
        order = np.argsort(keys, kind="stable")
        keys, matrix = keys[order], matrix[order]
        tmp_emb, tmp_keys = self._path(".emb.tmp.npy"), self._path(".keys.tmp.npy")
        np.save(tmp_emb, matrix)
        np.save(tmp_keys, keys)
        os.replace(tmp_emb, self._path(".emb.npy"))
        os.replace(tmp_keys, self._path(".keys.npy"))
        texts_path = self._path(".texts.json")
        texts = {}
        if os.path.exists(texts_path):
            with open(texts_path, encoding="utf-8") as handle:
                texts = json.load(handle)
        texts.update({k.decode(): t for k, (t, _) in new_items})
        with open(texts_path, "w", encoding="utf-8") as handle:
            json.dump(texts, handle, ensure_ascii=False, indent=1, sort_keys=True)
        self._keys, self._matrix = keys, np.load(self._path(".emb.npy"), mmap_mode="r")
        self._new = {}
        log.info("text cache written: %d prompts", len(keys))
