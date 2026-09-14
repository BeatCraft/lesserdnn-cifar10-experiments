"""Native LDNN model matching the three-convolution CIFAR-10 CNN.

This module uses the original LDNN ``core.Roster`` and layer classes.  It does
not import or use PyTorch.
"""

from __future__ import annotations

import argparse
import importlib
import math
import random
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


DEFAULT_LDNN_PATH = Path(__file__).resolve().parent / "ldnn_runtime"
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNELS = 3
IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS
NUM_CLASSES = 10


def load_ldnn(ldnn_path: Path, with_platform: bool = False) -> tuple[ModuleType, ModuleType | None]:
    """Load the original LDNN modules from a local checkout."""
    core_path = ldnn_path / "core.py"
    if not core_path.is_file():
        raise FileNotFoundError(f"LDNN core was not found: {core_path}")
    sys.path.insert(0, str(ldnn_path.resolve()))
    core = importlib.import_module("core")
    platform = importlib.import_module("plat") if with_platform else None
    return core, platform


def add_layer(roster: object, layer: object) -> object:
    roster.layers.append(layer)
    return layer

def setup_cifar10_layers(roster: object, core: ModuleType) -> None:
    """Create the native LDNN equivalent of the bias-free CifarCNN."""
    gpu = roster._gpu

    input_layer = add_layer(
        roster,
        core.InputLayer(roster.count_layers(), IMAGE_SIZE, IMAGE_SIZE, None, gpu),
    )
    conv1 = add_layer(
        roster,
        core.Conv_4_Layer(roster.count_layers(), 32, 32, 3, 32, input_layer, gpu),
    )
    pool1 = add_layer(
        roster,
        core.MaxLayer(roster.count_layers(), 32, 32, 32, conv1, gpu),
    )
    conv2 = add_layer(
        roster,
        core.Conv_4_Layer(roster.count_layers(), 16, 16, 32, 64, pool1, gpu),
    )
    pool2 = add_layer(
        roster,
        core.MaxLayer(roster.count_layers(), 64, 16, 16, conv2, gpu),
    )
    conv3 = add_layer(
        roster,
        core.Conv_4_Layer(roster.count_layers(), 8, 8, 64, 128, pool2, gpu),
    )
    add_layer(
        roster,
        core.OutputLayer(
            roster.count_layers(), 128 * 8 * 8, NUM_CLASSES, conv3, gpu, True
        ),
    )


def zero_biases(roster: object) -> None:
    """Keep native LDNN bias storage at the zero level (bias-free behavior)."""
    for layer in roster.layers:
        if layer.count_weight() == 0 or not hasattr(layer, "init_bias_index"):
            continue
        layer.init_bias_index()
        if hasattr(layer, "bias_gpu"):
            layer._gpu.write_np_to_mbuf(layer.bias_array.astype(np.float32), layer.bias_gpu)


def initialize_fan_in_aware_weights(roster: object, core: ModuleType, seed: int) -> None:
    """Initialize sparse discrete weights with variance close to 2/fan_in."""
    generator = np.random.default_rng(seed)
    for layer in roster.layers:
        if layer.get_type() not in (core.LAYER_TYPE_CONV, core.LAYER_TYPE_OUTPUT):
            continue
        levels = np.asarray(layer.bias_weight_set(), dtype=np.float32)
        positive = levels[levels > 0.0]
        if positive.size == 0 or 0.0 not in levels:
            raise ValueError("fan-in initializer requires zero and positive weight levels")
        target_variance = 2.0 / float(layer._num_input)
        target_magnitude = math.sqrt(target_variance)
        magnitude = float(positive[np.argmin(np.abs(positive - target_magnitude))])
        density = min(1.0, target_variance / (magnitude * magnitude))

        shape = (layer._num_node, layer._num_input)
        zero_index = int(np.flatnonzero(levels == 0.0)[0])
        positive_index = int(np.flatnonzero(levels == magnitude)[0])
        negative_index = int(np.flatnonzero(levels == -magnitude)[0])
        indexes = np.full(shape, zero_index, dtype=np.uint8)
        active = generator.random(shape) < density
        signs = generator.random(shape) < 0.5
        indexes[active & signs] = positive_index
        indexes[active & ~signs] = negative_index
        layer.import_weight_index(indexes)
        print(
            f"initialized layer={layer._index} fan_in={layer._num_input} "
            f"magnitude={magnitude} density={density:.6f}"
        )


def disable_metal_conv_batchnorm(gpu: object) -> None:
    """Disable Conv_4_Layer's Metal-only bn2d_forward on this GPU instance."""
    if getattr(gpu, "type", None) != 2:
        return
    if not hasattr(gpu, "bn2d_forward"):
        raise AttributeError("Metal GPU backend does not provide bn2d_forward")
    if not hasattr(gpu, "_original_bn2d_forward"):
        gpu._original_bn2d_forward = gpu.bn2d_forward

    def no_batchnorm(
        batch_size: int, output: object, channels: int, spatial_size: int, eps: float = 1e-5
    ) -> None:
        del batch_size, output, channels, spatial_size, eps

    gpu.bn2d_forward = no_batchnorm


def disable_metal_conv_scale_layer(gpu: object) -> None:
    """Disable Conv_4_Layer's per-sample max-absolute scaling on this GPU."""
    if getattr(gpu, "type", None) != 2:
        return
    if not hasattr(gpu, "scale_layer"):
        raise AttributeError("Metal GPU backend does not provide scale_layer")
    if not hasattr(gpu, "_original_scale_layer"):
        gpu._original_scale_layer = gpu.scale_layer

    def no_scale_layer(
        batch_size: int, size: int, scale: float, output: object
    ) -> None:
        del batch_size, size, scale, output

    gpu.scale_layer = no_scale_layer


def build_model(
    core: ModuleType,
    gpu: object | None = None,
    batch_size: int | None = None,
    seed: int = 0,
    conv_batchnorm: bool = False,
    conv_scale_layer: bool = False,
) -> object:
    """Build, and optionally allocate, a native LDNN CIFAR-10 roster."""
    random.seed(seed)
    np.random.seed(seed)
    if gpu is not None and not conv_batchnorm:
        disable_metal_conv_batchnorm(gpu)
    if gpu is not None and not conv_scale_layer:
        disable_metal_conv_scale_layer(gpu)
    roster = core.Roster()
    roster.set_gpu(gpu)
    roster.wi_mode = 7
    roster.conv_batchnorm = conv_batchnorm
    roster.conv_scale_layer = conv_scale_layer
    setup_cifar10_layers(roster, core)
    roster.set_qmode(0)

    if batch_size is not None:
        if gpu is None:
            raise ValueError("gpu is required when batch_size is specified")
        roster.prepare(batch_size, IMAGE_SIZE, NUM_CLASSES)
        initialize_fan_in_aware_weights(roster, core, seed)
        zero_biases(roster)
        roster.update_weight()
    return roster


def layer_summary(roster: object, core: ModuleType) -> list[str]:
    names = {
        core.LAYER_TYPE_INPUT: "Input",
        core.LAYER_TYPE_CONV: "Conv3x3+ReLU",
        core.LAYER_TYPE_MAX: "MaxPool2x2",
        core.LAYER_TYPE_OUTPUT: "Output+Softmax",
    }
    shapes = [
        "3x32x32",
        "32x32x32",
        "32x16x16",
        "64x16x16",
        "64x8x8",
        "128x8x8",
        "10",
    ]
    lines = []
    for index, (layer, shape) in enumerate(zip(roster.layers, shapes, strict=True)):
        lines.append(
            f"{index}: {names.get(layer.get_type(), 'Unknown')} "
            f"output={shape} weights={layer.count_weight()}"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the native LDNN CIFAR-10 model")
    parser.add_argument("--ldnn-path", type=Path, default=DEFAULT_LDNN_PATH)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enable-conv-batchnorm",
        action="store_true",
        help="restore the original Metal Conv_4_Layer BatchNorm behavior",
    )
    parser.add_argument(
        "--enable-conv-scale-layer",
        action="store_true",
        help="restore Conv_4_Layer's per-sample max-absolute scaling",
    )
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="build layer metadata without allocating the Metal/GPU runtime",
    )
    args = parser.parse_args()

    core, platform = load_ldnn(args.ldnn_path, with_platform=not args.structure_only)
    gpu = None if platform is None else platform.getGpu(0)
    batch_size = None if args.structure_only else args.batch_size
    model = build_model(
        core,
        gpu=gpu,
        batch_size=batch_size,
        seed=args.seed,
        conv_batchnorm=args.enable_conv_batchnorm,
        conv_scale_layer=args.enable_conv_scale_layer,
    )

    print(
        f"framework=LDNN pytorch=False bias=False "
        f"conv_batchnorm={model.conv_batchnorm} "
        f"conv_scale_layer={model.conv_scale_layer}"
    )
    print(f"weight_set={core.WEIGHT_SET}")
    for line in layer_summary(model, core):
        print(line)
    print(f"layers={model.count_layers()} discrete_weights={model.count_weight()}")


if __name__ == "__main__":
    main()
