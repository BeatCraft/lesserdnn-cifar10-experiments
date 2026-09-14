"""Summarize completed full-precision PyTorch reference seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

T_975 = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,
         10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,
         18:2.101,19:2.093,20:2.086}


def ci(values: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2: return mean, 0.0, mean, mean
    sd = statistics.stdev(values)
    half = T_975.get(len(values)-1, 1.96) * sd / math.sqrt(len(values))
    return mean, sd, mean-half, mean+half


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path); parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((args.run_root / "experiment_manifest.json").read_text())
    rows, missing = [], []
    for seed in manifest["seeds"]:
        path = args.run_root / f"seed_{seed}" / "pytorch_reference" / "result.json"
        if not path.is_file(): missing.append(str(path)); continue
        result = json.loads(path.read_text())
        rows.append({"seed": seed, "best_epoch": result["evaluated_best_epoch"],
            "best_validation_accuracy": result["best_validation_accuracy"],
            "test_accuracy": result["test"]["accuracy"], "test_loss": result["test"]["loss"],
            "training_seconds": result["training_seconds"]})
    if missing and not args.allow_incomplete: raise RuntimeError(f"{len(missing)} runs incomplete")
    if not rows: raise RuntimeError("no completed reference runs")
    with (args.run_root / "per_run_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    mean, sd, low, high = ci([float(x["test_accuracy"]) for x in rows])
    summary = {"runs": len(rows), "test_accuracy_mean": mean, "test_accuracy_std": sd,
        "test_accuracy_ci95_low": low, "test_accuracy_ci95_high": high,
        "test_loss_mean": statistics.fmean(float(x["test_loss"]) for x in rows),
        "best_epoch_mean": statistics.fmean(float(x["best_epoch"]) for x in rows),
        "training_seconds_mean": statistics.fmean(float(x["training_seconds"]) for x in rows),
        "missing": missing}
    (args.run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [f"# Full-precision PyTorch {manifest['dataset']} reference", "",
        f"- Runs: {len(rows)}", f"- Test accuracy: {mean:.4%} ± {sd:.4%}",
        f"- 95% CI: [{low:.4%}, {high:.4%}]", f"- Mean best epoch: {summary['best_epoch_mean']:.1f}"]
    if missing: lines.append(f"- Incomplete runs: {len(missing)}")
    (args.run_root / "SUMMARY.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines))


if __name__ == "__main__": main()
