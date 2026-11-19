"""Evaluation datasets and protocols (Sec. 4.1-4.2, Table 2).

Two protocols are used in the paper:

* ``accuracy`` -- the Derm1M protocol: HAM10000, Fitzpatrick17k, PAD-UFES-20 and Daffodil,
  scored by top-1 accuracy on the official evaluation split.
* ``weighted_f1`` -- nine datasets scored by weighted-F1 on all available images.

Each CSV lists ``image_path`` (relative to the repository root) and an integer ``label``
indexing the class-name list of that dataset.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from declip.classnames import CLASSNAMES_BY_DATASET
from declip.config import REPO_ROOT

# Metric sets: F17K has 113 classes and is scored by top-1/top-5 accuracy only.
CLASSIFICATION = "classification"
TOP_K = "top_k"

PROTOCOL_DATASETS: dict[str, list[str]] = {
    "accuracy": ["HAM", "F17K", "PAD", "Daffodil"],
    "weighted_f1": [
        "PAD",
        "HAM",
        "Daffodil",
        "BCN20000",
        "DDI",
        "HIBA",
        "MSKCC",
        "Dermnet",
        "Derm7pt",
    ],
}

PRIMARY_METRIC: dict[str, str] = {"accuracy": "accuracy", "weighted_f1": "f1"}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    protocol: str
    csv_path: str  # relative to REPO_ROOT
    classnames: tuple[str, ...]
    metric_set: str

    @property
    def num_classes(self) -> int:
        return len(self.classnames)

    @property
    def primary_metric(self) -> str:
        if self.metric_set == TOP_K:
            return "top1_accuracy"
        return PRIMARY_METRIC[self.protocol]


def dataset_specs(
    protocol: str, names: list[str] | None = None, meta_dir: str = "data/meta"
) -> list[DatasetSpec]:
    """Datasets of a protocol, in the column order used by the paper tables."""
    available = PROTOCOL_DATASETS[protocol]
    names = names or available
    specs = []
    for name in names:
        if name not in available:
            raise ValueError(f"dataset '{name}' is not part of protocol '{protocol}' ({available})")
        specs.append(
            DatasetSpec(
                name=name,
                protocol=protocol,
                csv_path=f"{meta_dir}/{protocol}/{name}.csv",
                classnames=tuple(CLASSNAMES_BY_DATASET[name]),
                metric_set=TOP_K if name == "F17K" else CLASSIFICATION,
            )
        )
    return specs


def union_classnames(specs: list[DatasetSpec]) -> list[str]:
    names: set[str] = set()
    for spec in specs:
        names.update(spec.classnames)
    return sorted(names)


class ImageCsvDataset(Dataset):
    """Images and integer labels listed in an evaluation CSV, in file order."""

    def __init__(self, csv_path: str, preprocess, root: str = REPO_ROOT):
        frame = pd.read_csv(os.path.join(root, csv_path))
        self.paths = [os.path.join(root, p) for p in frame["image_path"].tolist()]
        self.labels = frame["label"].astype(int).tolist()
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = self.preprocess(Image.open(self.paths[index]))
        return image, self.labels[index]


def make_loader(spec: DatasetSpec, preprocess, batch_size: int, workers: int) -> DataLoader:
    """Deterministic loader: no shuffling, no dropped tail, so row i is CSV row i."""
    dataset = ImageCsvDataset(spec.csv_path, preprocess)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False
    )
