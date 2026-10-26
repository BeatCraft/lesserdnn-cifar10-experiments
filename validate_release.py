"""Run lightweight consistency checks without training or downloading CIFAR-10."""

from __future__ import annotations

import json
import csv
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "lesserdnn_cifar10_rigorous_eval"
sys.path.insert(0, str(SOURCE))

from lr_schedule import parse_learning_rate_schedule
from pytorch_reference_train import CifarReferenceCNN

sys.path.insert(0, str(ROOT / "additional_experiments"))
from codebooks import CODEBOOKS, initial_indexes, digest
from analyze_results import analyze


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
        with (ROOT / "results" / directory / "per_run_results.csv").open() as stream:
            seed_rows = list(csv.DictReader(stream))
        if expected_methods is not None:
            if len(result["aggregate"]) != expected_methods:
                raise AssertionError(f"unexpected method count in {directory}")
            if expected_accuracy is not None:
                close(float(result["aggregate"][0]["test_accuracy_mean"]), expected_accuracy)
            for method in result["aggregate"]:
                rows = [row for row in seed_rows if row["update_rule"] == method["update_rule"]]
                if sorted(int(row["seed"]) for row in rows) != list(range(10)):
                    raise AssertionError(f"missing/duplicate original seed in {directory}")
                values = np.asarray([float(row["test_accuracy"]) for row in rows])
                close(float(values.mean()), method["test_accuracy_mean"])
                close(float(values.std(ddof=1)), method["test_accuracy_std"])
                close(float(np.mean([float(row["test_loss"]) for row in rows])), method["test_loss_mean"])
        else:
            if int(result["runs"]) != 10:
                raise AssertionError("PyTorch summary does not contain 10 runs")
            close(float(result["test_accuracy_mean"]), float(expected_accuracy))
            if sorted(int(row["seed"]) for row in seed_rows) != list(range(10)):
                raise AssertionError("missing/duplicate PyTorch reference seed")
            close(float(np.mean([float(row["test_accuracy"]) for row in seed_rows])), result["test_accuracy_mean"])
    folder = ROOT / "results" / "timed_codebooks_80runs"
    saved = json.loads((folder / "summary.json").read_text())
    recomputed = analyze(folder, write=False)
    if saved != recomputed:
        raise AssertionError("80-run summary does not match seed-level records")
    with (folder / "per_run_results.csv").open() as stream:
        records = list(csv.DictReader(stream))
    with (folder / "learning_curves.csv").open() as stream:
        curves = list(csv.DictReader(stream))
    if len(curves) != 560:
        raise AssertionError("expected 560 timed validation observations")
    for record in records:
        quant, seed = int(record["quant_set"]), int(record["seed"])
        indexes = initial_indexes(seed, quant)
        values = {key: CODEBOOKS[quant][value] for key, value in indexes.items()}
        if digest(values) != record["initial_weights_hash"]:
            raise AssertionError(f"initialization mismatch: {quant}, {seed}")
        points = [row for row in curves if all(row[key] == record[key] for key in ["method", "quant_set", "seed"])]
        if [float(row["checkpoint_target_s"]) for row in points] != list(range(0, 1801, 300)):
            raise AssertionError("missing/duplicate timed checkpoint")
        for key in ["val_loss", "val_accuracy", "training_time_s"]:
            close(float(points[-1][key]), float(record[key]))
        close(float(points[0]["val_loss"]), float(record["initial_val_loss"]))
        close(float(points[0]["val_accuracy"]), float(record["initial_val_accuracy"]))
        peer = next(row for row in records if row["quant_set"] == record["quant_set"]
                    and row["seed"] == record["seed"] and row["method"] != record["method"])
        for key in ["initial_weights_hash", "split_hash", "initial_val_accuracy", "initial_val_loss"]:
            if record[key] != peer[key]:
                raise AssertionError(f"paired initial state differs: {key}")
    protocol = json.loads((folder / "protocol.json").read_text())
    jobs = protocol["run_order"]
    if len(jobs) != 80 or len({row["run_name"] for row in jobs}) != 80:
        raise AssertionError("invalid 80-run schedule")
    for quant in [8, 9, 10, 11]:
        first = [row for i, row in enumerate(jobs) if row["quant_set"] == quant and not any(
            previous["quant_set"] == quant and previous["seed"] == row["seed"] for previous in jobs[:i])]
        if sum(row["method"] == "sa" for row in first) != 5:
            raise AssertionError("optimizer order is not balanced")
    print("release_validation=ok")


if __name__ == "__main__":
    main()
