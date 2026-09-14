"""CIFAR-10 specialization of the full-precision PyTorch reference trainer."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
CORE_PATH = ROOT / "pytorch_reference_core.py"
spec = importlib.util.spec_from_file_location("pytorch_reference_core", CORE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"could not load shared reference trainer: {CORE_PATH}")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class CifarReferenceCNN(nn.Module):
    """Full-precision counterpart of the bias-free LesserDNN CIFAR-10 model."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.ReLU(),
        )
        self.classifier = nn.Linear(128 * 8 * 8, 10, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs).flatten(1))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--split-seed", type=int, default=20260826)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--target-epochs", type=int, default=50)
    parser.add_argument("--lr-schedule", default="50:0.01")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-gradient-batches", type=int)
    return parser.parse_args()


def make_dataset(root: Path, train: bool) -> object:
    return datasets.CIFAR10(root=root, train=train, download=False, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]))


def dataset_available(root: Path) -> bool:
    return (root / "cifar-10-batches-py").is_dir()


def main() -> None:
    core.ROOT = ROOT
    core.MEAN = CIFAR10_MEAN
    core.STD = CIFAR10_STD
    core.DATASET_NAME = "CIFAR-10"
    core.EXPECTED_PARAMETER_COUNT = 174_944
    core.ReferenceCNN = CifarReferenceCNN
    core.arguments = arguments
    core.make_dataset = make_dataset
    core.dataset_available = dataset_available
    core.main()


if __name__ == "__main__":
    main()
