"""Download CIFAR-10 to the repository-local data directory."""

from pathlib import Path

from torchvision import datasets


def main() -> None:
    root = Path(__file__).resolve().parent / "data"
    datasets.CIFAR10(root=root, train=True, download=True)
    datasets.CIFAR10(root=root, train=False, download=True)
    print(f"CIFAR-10 is ready under {root}")


if __name__ == "__main__":
    main()
