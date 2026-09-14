"""Perform paired comparisons between LesserDNN modes and the PyTorch reference."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path

LABELS = {"single": "Single-batch", "two_batch_average": "Two-batch average", "same_sign": "Same-sign two-batch"}
T_975 = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,7:2.365,8:2.306,9:2.262,
         10:2.228,11:2.201,12:2.179,13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,
         18:2.101,19:2.093,20:2.086}


def ci(values: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2: return mean, 0.0, mean, mean
    sd = statistics.stdev(values); half = T_975.get(len(values)-1, 1.96)*sd/math.sqrt(len(values))
    return mean, sd, mean-half, mean+half


def signflip(values: list[float]) -> float:
    observed = abs(statistics.fmean(values)); extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        extreme += abs(statistics.fmean(s*x for s,x in zip(signs, values))) >= observed-1e-15
    return extreme/(2**len(values))


def holm(values: list[float]) -> list[float]:
    ordered=sorted(enumerate(values),key=lambda x:x[1]); result=[0.0]*len(values); prior=0.0
    for rank,(index,value) in enumerate(ordered):
        prior=max(prior,min(1.0,(len(values)-rank)*value)); result[index]=prior
    return result


def load_manifest(root: Path) -> dict[str, object]:
    for name in ("experiment_manifest.json", "combined_manifest.json"):
        path = root / name
        if path.is_file(): return json.loads(path.read_text())
    raise FileNotFoundError(f"manifest not found under {root}")


def load_ldnn_accuracies(root: Path) -> dict[tuple[int, str], float]:
    table = root / "per_run_results.csv"
    if table.is_file():
        with table.open(newline="") as stream:
            return {(int(row["seed"]), row["update_rule"]): float(row["test_accuracy"])
                    for row in csv.DictReader(stream)}
    manifest = load_manifest(root)
    result = {}
    for seed in manifest["seeds"]:
        for rule in manifest["rules"]:
            path = root / f"seed_{seed}" / rule / "result.json"
            if path.is_file(): result[(int(seed), str(rule))] = float(json.loads(path.read_text())["test"]["accuracy"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ldnn_root", type=Path); parser.add_argument("pytorch_root", type=Path)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    ldnn_manifest = load_manifest(args.ldnn_root)
    reference_manifest = load_manifest(args.pytorch_root)
    if ldnn_manifest["learning_rate_segments"] != reference_manifest["learning_rate_segments"]:
        raise RuntimeError("learning-rate schedules differ")
    if ldnn_manifest["split_seed"] != reference_manifest["split_seed"]:
        raise RuntimeError("split seeds differ")
    rows = []
    ldnn_accuracies = load_ldnn_accuracies(args.ldnn_root)
    for rule in ldnn_manifest["rules"]:
        differences, seeds = [], []
        for seed in sorted(set(ldnn_manifest["seeds"]) & set(reference_manifest["seeds"])):
            reference_path = args.pytorch_root / f"seed_{seed}" / "pytorch_reference" / "result.json"
            if (int(seed), str(rule)) not in ldnn_accuracies or not reference_path.is_file(): continue
            ldnn = ldnn_accuracies[(int(seed), str(rule))]
            reference = json.loads(reference_path.read_text())["test"]["accuracy"]
            differences.append(float(ldnn) - float(reference)); seeds.append(seed)
        if not differences: continue
        mean, sd, low, high = ci(differences)
        rows.append({"update_rule": rule, "method": LABELS[rule], "paired_seeds": seeds,
            "ldnn_minus_pytorch_mean": mean, "difference_std": sd, "ci95_low": low, "ci95_high": high,
            "exact_sign_flip_pvalue_two_sided": signflip(differences)})
    expected = len(set(ldnn_manifest["seeds"]) & set(reference_manifest["seeds"]))
    if not args.allow_incomplete and any(len(x["paired_seeds"]) != expected for x in rows):
        raise RuntimeError("paired runs are incomplete; use --allow-incomplete for a provisional comparison")
    adjusted = holm([float(x["exact_sign_flip_pvalue_two_sided"]) for x in rows])
    for row, value in zip(rows, adjusted): row["holm_adjusted_pvalue"] = value
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"ldnn_root": str(args.ldnn_root.resolve()),
        "pytorch_root": str(args.pytorch_root.resolve()), "comparisons": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__": main()
