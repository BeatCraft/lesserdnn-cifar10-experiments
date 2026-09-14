"""Run or resume the 10-seed full-precision PyTorch CIFAR-10 reference."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORE_PATH = ROOT / "pytorch_reference_runner_core.py"
spec = importlib.util.spec_from_file_location("pytorch_reference_runner_core", CORE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"could not load shared reference runner: {CORE_PATH}")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--target-epochs", type=int, default=50)
    parser.add_argument("--lr-schedule", default="50:0.01")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--split-seed", type=int, default=20260826)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "pytorch_reference_cifar10_lr_001_constant_50_v1")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-gradient-batches", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    core.ROOT = ROOT
    core.DATASET_NAME = "CIFAR-10"
    core.dataset_available = lambda root: (root / "cifar-10-batches-py").is_dir()
    core.arguments = arguments
    core.main()


if __name__ == "__main__":
    main()
