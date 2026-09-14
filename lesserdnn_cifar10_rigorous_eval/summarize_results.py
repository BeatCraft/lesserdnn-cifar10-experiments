"""Aggregate completed rigorous-evaluation runs and perform paired comparisons."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path


RULES = ("single", "two_batch_average", "same_sign")
LABELS = {
    "single": "Single-batch",
    "two_batch_average": "Two-batch average",
    "same_sign": "Same-sign two-batch",
}
T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def mean_ci(values: list[float]) -> tuple[float, float, float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0, mean, mean
    std = statistics.stdev(values)
    critical = T_975.get(len(values) - 1, 1.96)
    half_width = critical * std / math.sqrt(len(values))
    return mean, std, mean - half_width, mean + half_width


def exact_sign_flip_pvalue(differences: list[float]) -> float:
    """Two-sided paired randomization test under exchangeability of method labels."""
    observed = abs(statistics.fmean(differences))
    if not differences:
        return math.nan
    extreme = 0
    total = 2 ** len(differences)
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistic = abs(statistics.fmean(sign * value for sign, value in zip(signs, differences)))
        if statistic >= observed - 1e-15:
            extreme += 1
    return extreme / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    manifest_path = args.run_root / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected_seeds = [int(seed) for seed in manifest["seeds"]]
    expected_rules = list(manifest["rules"])
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    by_rule_seed: dict[str, dict[int, dict[str, object]]] = {rule: {} for rule in expected_rules}

    for seed in expected_seeds:
        for rule in expected_rules:
            result_path = args.run_root / f"seed_{seed}" / rule / "result.json"
            if not result_path.is_file():
                missing.append(str(result_path))
                continue
            result = json.loads(result_path.read_text())
            test = result["test"]
            row = {
                "seed": seed,
                "update_rule": rule,
                "method": LABELS[rule],
                "best_epoch": result["evaluated_best_epoch"],
                "best_validation_accuracy": result["best_validation_accuracy"],
                "test_accuracy": test["accuracy"],
                "test_loss": test["loss"],
                "training_seconds": result["training_seconds"],
            }
            rows.append(row)
            by_rule_seed[rule][seed] = row

    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"{len(missing)} runs are incomplete; rerun the driver or use --allow-incomplete"
        )
    if not rows:
        raise RuntimeError("no completed results found")

    with (args.run_root / "per_run_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate: list[dict[str, object]] = []
    for rule in expected_rules:
        rule_rows = list(by_rule_seed[rule].values())
        if not rule_rows:
            continue
        accuracy = [float(row["test_accuracy"]) for row in rule_rows]
        loss = [float(row["test_loss"]) for row in rule_rows]
        epochs = [float(row["best_epoch"]) for row in rule_rows]
        mean_acc, std_acc, low_acc, high_acc = mean_ci(accuracy)
        aggregate.append(
            {
                "update_rule": rule,
                "method": LABELS[rule],
                "runs": len(rule_rows),
                "test_accuracy_mean": mean_acc,
                "test_accuracy_std": std_acc,
                "test_accuracy_ci95_low": low_acc,
                "test_accuracy_ci95_high": high_acc,
                "test_loss_mean": statistics.fmean(loss),
                "best_epoch_mean": statistics.fmean(epochs),
                "training_seconds_mean": statistics.fmean(
                    float(row["training_seconds"]) for row in rule_rows
                ),
            }
        )

    paired: list[dict[str, object]] = []
    for first, second in itertools.combinations(expected_rules, 2):
        common = sorted(set(by_rule_seed[first]) & set(by_rule_seed[second]))
        if not common:
            continue
        differences = [
            float(by_rule_seed[first][seed]["test_accuracy"])
            - float(by_rule_seed[second][seed]["test_accuracy"])
            for seed in common
        ]
        mean_diff, std_diff, low_diff, high_diff = mean_ci(differences)
        paired.append(
            {
                "first_rule": first,
                "second_rule": second,
                "paired_seeds": common,
                "accuracy_difference_mean": mean_diff,
                "accuracy_difference_std": std_diff,
                "accuracy_difference_ci95_low": low_diff,
                "accuracy_difference_ci95_high": high_diff,
                "exact_sign_flip_pvalue_two_sided": exact_sign_flip_pvalue(differences),
            }
        )

    summary = {"aggregate": aggregate, "paired_comparisons": paired, "missing": missing}
    (args.run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Rigorous CIFAR-10 evaluation summary",
        "",
        "| Method | Runs | Test accuracy mean ± SD | 95% CI | Mean best epoch |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['method']} | {row['runs']} | "
            f"{float(row['test_accuracy_mean']):.4%} ± {float(row['test_accuracy_std']):.4%} | "
            f"[{float(row['test_accuracy_ci95_low']):.4%}, "
            f"{float(row['test_accuracy_ci95_high']):.4%}] | "
            f"{float(row['best_epoch_mean']):.1f} |"
        )
    lines.extend((
        "", "## Paired comparisons", "",
        "Difference is first method minus second method.", "",
        "| First | Second | Seeds | Mean difference | 95% CI | Exact p-value |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ))
    for row in paired:
        lines.append(
            f"| {LABELS[str(row['first_rule'])]} | {LABELS[str(row['second_rule'])]} | "
            f"{len(row['paired_seeds'])} | {float(row['accuracy_difference_mean']):+.4%} | "
            f"[{float(row['accuracy_difference_ci95_low']):+.4%}, "
            f"{float(row['accuracy_difference_ci95_high']):+.4%}] | "
            f"{float(row['exact_sign_flip_pvalue_two_sided']):.4f} |"
        )
    if missing:
        lines.extend(("", f"Incomplete runs: {len(missing)}"))
    (args.run_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved_summary={args.run_root / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
