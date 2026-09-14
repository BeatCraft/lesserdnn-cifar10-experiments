"""Load a CIFAR-10 batch with PyTorch and run it through native LDNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ldnn_cifar10_model import DEFAULT_LDNN_PATH, NUM_CLASSES, build_model, load_ldnn


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def make_pytorch_loader(
    data_root: Path, batch_size: int, normalize: bool = True
) -> DataLoader:
    """Load CIFAR-10 with the same normalization used for PyTorch training."""
    transform_steps = [transforms.ToTensor()]
    if normalize:
        transform_steps.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=False,
        transform=transforms.Compose(transform_steps),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )


def pytorch_batch_to_ldnn(images: Tensor, labels: Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Convert NCHW PyTorch tensors to LDNN data and one-hot label arrays."""
    if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
        raise ValueError(f"expected images shaped (N, 3, 32, 32), got {tuple(images.shape)}")
    if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
        raise ValueError("labels must have shape (N,) matching the image batch")
    if not torch.is_floating_point(images):
        raise TypeError("images must be floating-point tensors, normally from ToTensor()")
    if images.numel() and not torch.isfinite(images).all():
        raise ValueError("images contain NaN or infinity")
    if labels.numel() and (labels.min().item() < 0 or labels.max().item() >= NUM_CLASSES):
        raise ValueError("CIFAR-10 labels must be in [0, 9]")

    # CIFAR's native serialized layout and PyTorch's NCHW contiguous layout are
    # both channel-first: all red pixels, then green, then blue.
    data = images.detach().cpu().to(torch.float32).contiguous().numpy()
    data = np.ascontiguousarray(data.reshape(images.shape[0], -1), dtype=np.float32)

    label_indexes = labels.detach().cpu().to(torch.int64).numpy()
    one_hot = np.zeros((images.shape[0], NUM_CLASSES), dtype=np.float32)
    one_hot[np.arange(images.shape[0]), label_indexes] = 1.0
    return data, np.ascontiguousarray(one_hot)


def read_softmax(model: object) -> np.ndarray:
    output = model.output
    dtype = np.float16 if model.qmode == 1 else np.float32
    return model._gpu.read_mbuf_to_numpy(output._gpu_softmax, dtype, output._softmax_array.shape)


def read_logits(model: object) -> np.ndarray:
    output = model.output
    dtype = np.float16 if model.qmode == 1 else np.float32
    return model._gpu.read_mbuf_to_numpy(output._gpu_output, dtype, output._output_array.shape)


def stable_softmax(logits: np.ndarray, divisor: float) -> np.ndarray:
    """Numerically stable replacement for LDNN Metal's unshifted exp()."""
    if divisor <= 0.0:
        raise ValueError("softmax divisor must be positive")
    scaled = logits.astype(np.float64) / divisor
    scaled -= scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def select_softmax_divisor(report: dict[str, object], model: object) -> float:
    """Select the divisor matching enabled Conv normalization operations."""
    if not model.conv_batchnorm and not model.conv_scale_layer:
        explicit = report.get("ldnn_softmax_divisor_no_conv_normalization")
        if explicit is not None:
            return float(explicit)
        scales = [float(item["omitted_positive_scale"]) for item in report["layers"]]
        return 1.0 / float(np.prod(scales))
    return float(report.get("ldnn_softmax_divisor", 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ldnn-path", type=Path, default=DEFAULT_LDNN_PATH)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="use raw ToTensor() values instead of PyTorch training normalization",
    )
    parser.add_argument(
        "--weights", type=Path, help="LDNN index CSV produced by convert_pytorch_to_ldnn.py"
    )
    parser.add_argument(
        "--conversion-only",
        action="store_true",
        help="verify tensor conversion without initializing Metal or running LDNN",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    images, labels = next(
        iter(
            make_pytorch_loader(
                args.data_root, args.batch_size, normalize=not args.no_normalize
            )
        )
    )
    data, one_hot = pytorch_batch_to_ldnn(images, labels)
    print(
        f"pytorch_images={tuple(images.shape)} normalized={not args.no_normalize} "
        f"labels={labels.tolist()} "
        f"range=[{images.min().item():.6f},{images.max().item():.6f}]"
    )
    print(
        f"ldnn_data={data.shape} dtype={data.dtype} contiguous={data.flags.c_contiguous} "
        f"labels={one_hot.shape} one_hot_sums={one_hot.sum(axis=1).tolist()}"
    )

    if args.conversion_only:
        print("conversion_check=ok")
        return

    core, platform = load_ldnn(args.ldnn_path, with_platform=True)
    model = build_model(
        core, gpu=platform.getGpu(0), batch_size=args.batch_size, seed=args.seed
    )
    softmax_divisor = 1.0
    if args.weights is not None:
        # The original LDNN API predates pathlib and concatenates a suffix.
        model.load(str(args.weights), mode=0)
        report_path = args.weights.with_suffix(args.weights.suffix + ".json")
        if report_path.is_file():
            report = json.loads(report_path.read_text())
            softmax_divisor = select_softmax_divisor(report, model)
            model.output.set_softmax_scale(softmax_divisor)
            print(f"ldnn_softmax_divisor={softmax_divisor:.8g}")
        model.update_weight()
        print(f"loaded_ldnn_weights={args.weights}")
    model.direct_set_data(data)
    model.direct_set_label(one_hot)
    model.propagate()
    logits = read_logits(model)
    if not np.isfinite(logits).all():
        raise RuntimeError("LDNN logits contain NaN or infinity")
    probabilities = stable_softmax(logits, softmax_divisor)
    predictions = probabilities.argmax(axis=1)

    if probabilities.shape != (args.batch_size, NUM_CLASSES):
        raise RuntimeError(f"unexpected LDNN output shape: {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("stable softmax output contains NaN or infinity")
    probability_sums = probabilities.sum(axis=1)
    if not np.allclose(probability_sums, 1.0, atol=1e-4):
        raise RuntimeError(
            "LDNN softmax rows do not sum to one: "
            f"sums={probability_sums.tolist()} "
            f"range=[{probabilities.min()},{probabilities.max()}]"
        )

    print(
        f"ldnn_probabilities={probabilities.shape} predictions={predictions.tolist()} "
        f"softmax_sums={probability_sums.round(6).tolist()}"
    )
    print("pytorch_batch_to_ldnn_forward_check=ok")


if __name__ == "__main__":
    main()
