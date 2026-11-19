"""End-to-end check on cached embeddings: the Default baseline on PAD and HAM (DermLIP, Table 4)."""

import os

import pytest
import torch

from declip.config import REPO_ROOT
from declip.run import main

pytestmark = pytest.mark.gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_default_baseline_pad_ham(tmp_path):
    out = tmp_path / "smoke.csv"
    main(
        [
            "--config",
            os.path.join(REPO_ROOT, "configs", "default.yaml"),
            "--backbone",
            "dermlip",
            "--set",
            "aggregate.alpha=1.0",
            'evaluation.datasets=["PAD","HAM"]',
            "--out",
            str(out),
        ]
    )
    import csv

    row = next(csv.DictReader(open(out)))
    assert abs(float(row["PAD"]) - 61.32) < 0.01
    assert abs(float(row["HAM"]) - 67.26) < 0.01
