"""Combine disjoint seed batches produced with the same experiment settings."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from pathlib import Path

from summarize_results import LABELS, exact_sign_flip_pvalue, mean_ci


IGNORED_MANIFEST_KEYS = {"seeds", "output_root"}


def comparable_manifest(manifest: dict[str, object]) -> dict[str, object]:
    return {
        key: value for key, value in manifest.items()
        if key not in IGNORED_MANIFEST_KEYS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run_roots) < 2:
        raise ValueError("at least two run roots are required")

    manifests = []
    for root in args.run_roots:
        path = root / "experiment_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"manifest not found: {path}")
        manifests.append(json.loads(path.read_text()))
    expected = comparable_manifest(manifests[0])
    for root, manifest in zip(args.run_roots[1:], manifests[1:]):
        if comparable_manifest(manifest) != expected:
            raise RuntimeError(f"experiment configuration mismatch: {root}")

    seeds: list[int] = []
    for root, manifest in zip(args.run_roots, manifests):
        root_seeds = [int(seed) for seed in manifest["seeds"]]
        overlap = sorted(set(seeds) & set(root_seeds))
        if overlap:
            raise RuntimeError(f"duplicate seeds across run roots: {overlap}")
        seeds.extend(root_seeds)
    seeds.sort()
    rules = list(manifests[0]["rules"])

    rows: list[dict[str, object]] = []
    by_rule_seed: dict[str, dict[int, dict[str, object]]] = {
        rule: {} for rule in rules
    }
    for root, manifest in zip(args.run_roots, manifests):
        for seed in manifest["seeds"]:
            for rule in rules:
                result_path = root / f"seed_{seed}" / rule / "result.json"
                if not result_path.is_file():
                    raise RuntimeError(f"incomplete run: {result_path}")
                result = json.loads(result_path.read_text())
                test = result["test"]
                row = {
                    "seed": int(seed),
                    "update_rule": rule,
                    "method": LABELS[rule],
                    "best_epoch": result["evaluated_best_epoch"],
                    "best_validation_accuracy": result["best_validation_accuracy"],
                    "test_accuracy": test["accuracy"],
                    "test_loss": test["loss"],
                    "training_seconds": result["training_seconds"],
                    "source_root": str(root.resolve()),
                }
                rows.append(row)
                by_rule_seed[rule][int(seed)] = row

    rows.sort(key=lambda row: (int(row["seed"]), rules.index(str(row["update_rule"]))))
    aggregate: list[dict[str, object]] = []
    for rule in rules:
        rule_rows = [by_rule_seed[rule][seed] for seed in seeds]
        accuracy = [float(row["test_accuracy"]) for row in rule_rows]
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
                "test_loss_mean": statistics.fmean(
                    float(row["test_loss"]) for row in rule_rows
                ),
                "best_epoch_mean": statistics.fmean(
                    float(row["best_epoch"]) for row in rule_rows
                ),
                "training_seconds_mean": statistics.fmean(
                    float(row["training_seconds"]) for row in rule_rows
                ),
            }
        )

    paired: list[dict[str, object]] = []
    for first, second in itertools.combinations(rules, 2):
        differences = [
            float(by_rule_seed[first][seed]["test_accuracy"])
            - float(by_rule_seed[second][seed]["test_accuracy"])
            for seed in seeds
        ]
        mean_diff, std_diff, low_diff, high_diff = mean_ci(differences)
        paired.append(
            {
                "first_rule": first,
                "second_rule": second,
                "paired_seeds": seeds,
                "accuracy_difference_mean": mean_diff,
                "accuracy_difference_std": std_diff,
                "accuracy_difference_ci95_low": low_diff,
                "accuracy_difference_ci95_high": high_diff,
                "exact_sign_flip_pvalue_two_sided": exact_sign_flip_pvalue(differences),
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    combined_manifest = {
        **expected,
        "seeds": seeds,
        "source_roots": [str(root.resolve()) for root in args.run_roots],
        "output_root": str(args.output_root.resolve()),
    }
    (args.output_root / "combined_manifest.json").write_text(
        json.dumps(combined_manifest, indent=2) + "\n"
    )
    with (args.output_root / "per_run_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"aggregate": aggregate, "paired_comparisons": paired, "missing": []}
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        f"# Combined rigorous CIFAR-10 evaluation ({len(seeds)} seeds)",
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
    lines.extend(
        (
            "", "## Paired comparisons", "",
            "Difference is first method minus second method.", "",
            "| First | Second | Seeds | Mean difference | 95% CI | Exact p-value |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        )
    )
    for row in paired:
        lines.append(
            f"| {LABELS[str(row['first_rule'])]} | {LABELS[str(row['second_rule'])]} | "
            f"{len(row['paired_seeds'])} | {float(row['accuracy_difference_mean']):+.4%} | "
            f"[{float(row['accuracy_difference_ci95_low']):+.4%}, "
            f"{float(row['accuracy_difference_ci95_high']):+.4%}] | "
            f"{float(row['exact_sign_flip_pvalue_two_sided']):.4f} |"
        )
    (args.output_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved_combined_summary={args.output_root / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
