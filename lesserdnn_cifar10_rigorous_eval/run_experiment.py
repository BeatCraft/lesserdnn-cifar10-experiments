"""Run/resume the rigorous three-rule, multi-seed CIFAR-10 experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from lr_schedule import canonical_schedule_specification, parse_learning_rate_schedule


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "train_one.py"
SUMMARY_SCRIPT = PROJECT_ROOT / "summarize_results.py"
RULES = ("single", "two_batch_average", "same_sign")
REQUIRED_MODULES = {
    "numpy": "numpy",
    "torch": "torch",
    "torchvision": "torchvision",
    "objc": "pyobjc-framework-Metal",
    "Metal": "pyobjc-framework-Metal",
    "PIL": "Pillow",
    "scipy": "scipy",
    "pyopencl": "pyopencl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--rules", nargs="+", choices=RULES, default=list(RULES))
    parser.add_argument("--target-epochs", type=int, default=300)
    parser.add_argument("--lr-switch-epoch", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--continuation-learning-rate", type=float, default=0.003)
    parser.add_argument(
        "--lr-schedule",
        help=(
            "piecewise schedule as end_epoch:rate pairs; for example "
            "100:0.01,200:0.005,300:0.001"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--split-seed", type=int, default=20260826)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT.parent / "data")
    parser.add_argument(
        "--ldnn-path", type=Path,
        default=PROJECT_ROOT.parent / "lesserdnn_cifar10_ablation_200" / "ldnn_runtime",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "rigorous_eval_v1")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-gradient-batches", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command(args: argparse.Namespace, rule: str, seed: int) -> list[str]:
    output_dir = args.output_root / f"seed_{seed}" / rule
    result = [
        sys.executable, "-u", str(TRAIN_SCRIPT),
        "--update-rule", rule,
        "--seed", str(seed),
        "--split-seed", str(args.split_seed),
        "--validation-size", str(args.validation_size),
        "--target-epochs", str(args.target_epochs),
        "--batch-size", str(args.batch_size),
        "--epsilon", str(args.epsilon),
        "--data-root", str(args.data_root.resolve()),
        "--ldnn-path", str(args.ldnn_path.resolve()),
        "--output-dir", str(output_dir.resolve()),
        "--log-every", str(args.log_every),
    ]
    if args.learning_rate_segments is None:
        result.extend(
            (
                "--lr-switch-epoch", str(args.lr_switch_epoch),
                "--learning-rate", str(args.learning_rate),
                "--continuation-learning-rate", str(args.continuation_learning_rate),
            )
        )
    else:
        result.extend(
            (
                "--lr-schedule",
                canonical_schedule_specification(args.learning_rate_segments),
            )
        )
    if args.max_gradient_batches is not None:
        result.extend(("--max-gradient-batches", str(args.max_gradient_batches)))
    return result


def check_python_environment() -> None:
    missing_modules = [
        module for module in REQUIRED_MODULES
        if importlib.util.find_spec(module) is None
    ]
    if not missing_modules:
        return
    packages = sorted({REQUIRED_MODULES[module] for module in missing_modules})
    install_command = f"{sys.executable} -m pip install -r {PROJECT_ROOT.parent / 'requirements.txt'}"
    raise RuntimeError(
        "current Python environment is missing required modules: "
        f"{', '.join(missing_modules)}\n"
        f"missing packages: {', '.join(packages)}\n"
        f"install them with:\n  {install_command}\n"
        "Alternatively, deactivate this virtual environment and use the Python "
        "environment that ran the earlier LDNN experiments."
    )


def main() -> None:
    args = parse_args()
    args.learning_rate_segments = (
        parse_learning_rate_schedule(args.lr_schedule, args.target_epochs)
        if args.lr_schedule is not None
        else None
    )
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    if len(set(args.rules)) != len(args.rules):
        raise ValueError("rules must be unique")
    if not TRAIN_SCRIPT.is_file() or not SUMMARY_SCRIPT.is_file():
        raise FileNotFoundError("training or summary script is missing")
    check_python_environment()
    if not (args.data_root / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(f"CIFAR-10 not found under {args.data_root}")
    if not (args.ldnn_path / "core.py").is_file():
        raise FileNotFoundError(f"LDNN runtime not found under {args.ldnn_path}")

    learning_rate_segments = args.learning_rate_segments or [
        {"start": 1, "end": args.lr_switch_epoch, "learning_rate": args.learning_rate},
        {
            "start": args.lr_switch_epoch + 1,
            "end": args.target_epochs,
            "learning_rate": args.continuation_learning_rate,
        },
    ]
    manifest = {
        "protocol": "rigorous_eval_v1",
        "rules": args.rules,
        "seeds": args.seeds,
        "selection": "fixed stratified validation split; accuracy then loss",
        "test_usage": "once per seed/rule after validation checkpoint selection",
        "target_epochs": args.target_epochs,
        "learning_rate_segments": learning_rate_segments,
        "batch_size": args.batch_size,
        "validation_size": args.validation_size,
        "split_seed": args.split_seed,
        "epsilon": args.epsilon,
        "data_root": str(args.data_root.resolve()),
        "ldnn_path": str(args.ldnn_path.resolve()),
        "output_root": str(args.output_root.resolve()),
        "max_gradient_batches": args.max_gradient_batches,
    }
    commands = [command(args, rule, seed) for seed in args.seeds for rule in args.rules]
    print(json.dumps(manifest, indent=2))
    for item in commands:
        print("command=" + " ".join(item))
    if args.dry_run:
        print(f"dry_run_complete=True planned_runs={len(commands)} files_modified=False")
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "experiment_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise RuntimeError(
                f"output root belongs to a different configuration: {manifest_path}; "
                "choose a new --output-root"
            )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    for seed in args.seeds:
        for rule in args.rules:
            print(f"start_run seed={seed} rule={rule}")
            subprocess.run(command(args, rule, seed), cwd=PROJECT_ROOT, check=True)

    subprocess.run(
        [sys.executable, str(SUMMARY_SCRIPT), str(args.output_root.resolve())],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print(f"experiment_complete={args.output_root.resolve()}")


if __name__ == "__main__":
    main()
