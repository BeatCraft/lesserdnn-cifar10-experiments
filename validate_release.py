"""Run lightweight consistency checks without training or downloading CIFAR-10."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "lesserdnn_cifar10_rigorous_eval"
sys.path.insert(0, str(SOURCE))

from lr_schedule import parse_learning_rate_schedule
from pytorch_reference_train import CifarReferenceCNN


def close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"expected {expected}, got {actual}")


def main() -> None:
    model = CifarReferenceCNN()
    if sum(parameter.numel() for parameter in model.parameters()) != 174_944:
        raise AssertionError("PyTorch parameter count does not match LesserDNN")
    if tuple(model(torch.zeros(2, 3, 32, 32)).shape) != (2, 10):
        raise AssertionError("unexpected PyTorch output shape")
    expected_schedule = [
        {"start": 1, "end": 100, "learning_rate": 0.01},
        {"start": 101, "end": 200, "learning_rate": 0.005},
        {"start": 201, "end": 300, "learning_rate": 0.001},
    ]
    if parse_learning_rate_schedule("100:0.01,200:0.005,300:0.001", 300) != expected_schedule:
        raise AssertionError("three-stage schedule mismatch")
    summaries = {
        "ldnn_three_stage_10seeds": (0.69794, 3),
        "ldnn_fixed_lr_10seeds": (None, 3),
        "pytorch_reference_10seeds": (0.6904, None),
    }
    for directory, (expected_accuracy, expected_methods) in summaries.items():
        result = json.loads((ROOT / "results" / directory / "summary.json").read_text())
        if expected_methods is not None:
            if len(result["aggregate"]) != expected_methods:
                raise AssertionError(f"unexpected method count in {directory}")
            if expected_accuracy is not None:
                close(float(result["aggregate"][0]["test_accuracy_mean"]), expected_accuracy)
        else:
            if int(result["runs"]) != 10:
                raise AssertionError("PyTorch summary does not contain 10 runs")
            close(float(result["test_accuracy_mean"]), float(expected_accuracy))
    print("release_validation=ok")


if __name__ == "__main__":
    main()
