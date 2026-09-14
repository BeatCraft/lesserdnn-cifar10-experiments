"""Run or resume the 10-seed full-precision PyTorch reference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from lr_schedule import canonical_schedule_specification, parse_learning_rate_schedule

ROOT = Path(__file__).resolve().parent
DATASET_NAME = "CIFAR-10"


def dataset_available(root: Path) -> bool:
    return (root / "cifar-10-batches-py").is_dir()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--target-epochs", type=int, default=50)
    parser.add_argument("--lr-schedule", default="50:0.01")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--split-seed", type=int, default=20260826)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT.parent / "runs" / "pytorch_reference_10seeds")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-gradient-batches", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def command(args: argparse.Namespace, schedule: str, seed: int) -> list[str]:
    result = [sys.executable, "-u", str(ROOT / "pytorch_reference_train.py"),
        "--seed", str(seed), "--split-seed", str(args.split_seed),
        "--validation-size", str(args.validation_size), "--target-epochs", str(args.target_epochs),
        "--lr-schedule", schedule, "--batch-size", str(args.batch_size),
        "--data-root", str(args.data_root.resolve()), "--device", args.device,
        "--output-dir", str((args.output_root / f"seed_{seed}" / "pytorch_reference").resolve()),
        "--log-every", str(args.log_every)]
    if args.max_gradient_batches is not None: result += ["--max-gradient-batches", str(args.max_gradient_batches)]
    return result


def main() -> None:
    args = arguments()
    if len(set(args.seeds)) != len(args.seeds): raise ValueError("seeds must be unique")
    segments = parse_learning_rate_schedule(args.lr_schedule, args.target_epochs)
    schedule = canonical_schedule_specification(segments)
    if not dataset_available(args.data_root): raise FileNotFoundError(f"{DATASET_NAME} data not found")
    manifest = {"protocol": f"{DATASET_NAME.lower().replace('-', '')}_pytorch_reference_v1", "dataset": DATASET_NAME, "framework": "PyTorch",
        "reference": "full_precision_sgd", "seeds": args.seeds, "target_epochs": args.target_epochs,
        "learning_rate_segments": segments, "batch_size": args.batch_size,
        "validation_size": args.validation_size, "split_seed": args.split_seed,
        "selection": "fixed stratified validation split; accuracy then loss",
        "test_usage": "once per seed after validation checkpoint selection", "device": args.device,
        "data_root": str(args.data_root.resolve()), "output_root": str(args.output_root.resolve()),
        "max_gradient_batches": args.max_gradient_batches}
    commands = [command(args, schedule, seed) for seed in args.seeds]
    print(json.dumps(manifest, indent=2))
    for item in commands: print("command=" + " ".join(item))
    if args.dry_run: print(f"dry_run_complete=True planned_runs={len(commands)} files_modified=False"); return
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "experiment_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError(f"output root has a different configuration: {manifest_path}")
    if not manifest_path.is_file(): manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for seed in args.seeds:
        print(f"start_pytorch_reference seed={seed}")
        subprocess.run(command(args, schedule, seed), cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "summarize_pytorch_reference.py"), str(args.output_root.resolve())], cwd=ROOT, check=True)


if __name__ == "__main__": main()
