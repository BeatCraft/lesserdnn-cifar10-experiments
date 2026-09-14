"""Train native LDNN on CIFAR-10 with selectable quantized update rules.

Forward and backward computation use LDNN's Metal implementation. The single,
two-batch average, and same-sign two-batch rules share data order, model,
initialization, gradients, shadow weights, clipping, and nearest quantization.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import objc
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ldnn_cifar10_model import DEFAULT_LDNN_PATH, NUM_CLASSES, build_model, load_ldnn
from train_ldnn_cifar10 import (
    forward_loss,
    release_temporary_metal_buffers,
    track_temporary_metal_buffers,
)
from verify_pytorch_batch_ldnn import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    DEFAULT_DATA_ROOT,
    pytorch_batch_to_ldnn,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "ldnn_cifar10_native_same_sign"
UPDATE_RULES = ("single", "two_batch_average", "same_sign")


def make_dataset(data_root: Path, train: bool) -> object:
    return datasets.CIFAR10(
        root=data_root,
        train=train,
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        ),
    )


def make_train_loader(
    dataset: object, batch_size: int, seed: int, epoch: int
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed + epoch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=True,
    )


def weighted_layers(model: object, core: object) -> list[object]:
    types = (core.LAYER_TYPE_CONV, core.LAYER_TYPE_OUTPUT)
    return [layer for layer in model.layers if layer.get_type() in types]


def native_slope(model: object) -> None:
    """Run LDNN slope with a numerically stable output softmax gradient."""
    output = model.output
    logits = np.asarray(output._output_array, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    output.delta = np.ascontiguousarray(
        probabilities.astype(np.float32) - model.label_array.astype(np.float32)
    )
    previous_output = output._pre._output_array.astype(np.float32).reshape(
        model._batch_size, output._num_input
    )
    weights = output._weight_matrix.astype(np.float32)
    if not (
        np.isfinite(previous_output).all()
        and np.isfinite(output.delta).all()
        and np.isfinite(weights).all()
    ):
        raise RuntimeError("non-finite value before output-layer backpropagation")
    # Apple's Accelerate-backed NumPy can emit floating-point warnings for
    # finite float32 matmul inputs. Explicit pre/post checks are the real guard.
    with np.errstate(all="ignore"):
        output.dW = np.ascontiguousarray(
            previous_output.T @ output.delta / np.float32(model._batch_size),
            dtype=np.float32,
        )
        propagated = np.ascontiguousarray(output.delta @ weights, dtype=np.float32)
    if not (np.isfinite(output.dW).all() and np.isfinite(propagated).all()):
        raise RuntimeError("non-finite output-layer gradient")
    output.dB = np.mean(output.delta, axis=0, dtype=np.float32)

    # Conv_4_Layer.slope expects next.grad. OutputLayer exposes delta instead,
    # so form dL/d(conv3_output) without changing any model weight.
    output.grad = propagated.reshape(model._batch_size, 128, 8 * 8)

    for layer_index in range(model.count_layers() - 2, -1, -1):
        model.get_layer_at(layer_index).slope(model.label_array)


def gradients_for_batch(
    model: object,
    layers: list[object],
    images: torch.Tensor,
    labels: torch.Tensor,
    softmax_divisor: float,
    temporary_buffers: list[object],
) -> tuple[float, int, int, dict[int, np.ndarray]]:
    pool = objc.autorelease_pool()
    pool.__enter__()
    try:
        data, one_hot = pytorch_batch_to_ldnn(images, labels)
        targets = labels.numpy().astype(np.int64, copy=False)
        model.direct_set_data(data)
        model.direct_set_label(one_hot)
        loss, accuracy = forward_loss(model, targets, softmax_divisor)
        native_slope(model)
        gradients = {
            layer._index: np.ascontiguousarray(layer.dW.T, dtype=np.float32).copy()
            for layer in layers
        }
        for layer_id, gradient in gradients.items():
            if not np.isfinite(gradient).all():
                raise RuntimeError(f"non-finite native gradient in layer {layer_id}")
        return loss, int(round(accuracy * targets.size)), targets.size, gradients
    finally:
        release_temporary_metal_buffers(temporary_buffers)
        pool.__exit__(None, None, None)


def update_weights(
    model: object,
    layers: list[object],
    shadow: dict[int, np.ndarray],
    first: dict[int, np.ndarray],
    second: dict[int, np.ndarray] | None,
    learning_rate: float,
    epsilon: float,
    update_rule: str,
) -> dict[str, int | float]:
    selected = 0
    changed = 0
    total = 0

    for layer in layers:
        layer_id = layer._index
        first_gradient = first[layer_id]
        if update_rule == "single":
            gradient = first_gradient
            mask = np.ones(first_gradient.shape, dtype=bool)
        else:
            if second is None:
                raise ValueError(f"{update_rule} requires two gradients")
            second_gradient = second[layer_id]
            gradient = (first_gradient + second_gradient) * np.float32(0.5)
            if update_rule == "two_batch_average":
                mask = np.ones(first_gradient.shape, dtype=bool)
            elif update_rule == "same_sign":
                nonzero = (np.abs(first_gradient) > epsilon) & (
                    np.abs(second_gradient) > epsilon
                )
                mask = nonzero & (
                    ((first_gradient > 0.0) & (second_gradient > 0.0))
                    | ((first_gradient < 0.0) & (second_gradient < 0.0))
                )
            else:
                raise ValueError(f"unknown update rule: {update_rule}")
        shadow[layer_id][mask] -= np.float32(learning_rate) * gradient[mask]

        levels = np.asarray(layer.bias_weight_set(), dtype=np.float32)
        np.clip(shadow[layer_id], levels[0], levels[-1], out=shadow[layer_id])
        indexes = np.abs(shadow[layer_id][..., None] - levels).argmin(axis=-1)
        indexes = indexes.astype(np.uint8, copy=False)
        changed += int(np.count_nonzero(indexes != layer._weight_index_matrix))
        selected += int(np.count_nonzero(mask))
        total += mask.size
        layer._weight_index_matrix = indexes.copy()
        layer._weight_matrix = np.ascontiguousarray(levels[indexes], dtype=np.float32)

    model.update_weight()
    return {
        "selected": selected,
        "selected_fraction": selected / total,
        "quantized_changes": changed,
        "total_components": total,
    }


def update_same_sign(
    model: object,
    layers: list[object],
    shadow: dict[int, np.ndarray],
    first: dict[int, np.ndarray],
    second: dict[int, np.ndarray],
    learning_rate: float,
    epsilon: float,
) -> dict[str, int | float]:
    """Backward-compatible wrapper used by the gradient verification script."""
    return update_weights(
        model,
        layers,
        shadow,
        first,
        second,
        learning_rate,
        epsilon,
        "same_sign",
    )


def evaluate(
    model: object,
    dataset: object,
    batch_size: int,
    softmax_divisor: float,
    temporary_buffers: list[object],
) -> tuple[float, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    loss_sum = 0.0
    correct = 0
    examples = 0
    for images, labels in loader:
        count = labels.numel()
        if count < batch_size:
            image_padding = torch.zeros(
                (batch_size - count, *images.shape[1:]), dtype=images.dtype
            )
            label_padding = torch.zeros(batch_size - count, dtype=labels.dtype)
            images = torch.cat((images, image_padding), dim=0)
            labels = torch.cat((labels, label_padding), dim=0)

        pool = objc.autorelease_pool()
        pool.__enter__()
        try:
            data, one_hot = pytorch_batch_to_ldnn(images, labels)
            model.direct_set_data(data)
            model.direct_set_label(one_hot)
            targets = labels.numpy().astype(np.int64, copy=False)
            loss, _ = forward_loss(model, targets, softmax_divisor)
            logits = model._gpu.read_mbuf_to_numpy(
                model.output._gpu_output, np.float32, (batch_size, NUM_CLASSES)
            )
            # Recompute the partial final batch loss so padding is excluded.
            logits = logits[:count].astype(np.float64) / softmax_divisor
            logits -= logits.max(axis=1, keepdims=True)
            log_sum_exp = np.log(np.exp(logits).sum(axis=1))
            true_logits = logits[np.arange(count), targets[:count]]
            loss_sum += float(np.sum(log_sum_exp - true_logits))
            correct += int(np.sum(logits.argmax(axis=1) == targets[:count]))
            examples += count
            del loss
        finally:
            release_temporary_metal_buffers(temporary_buffers)
            pool.__exit__(None, None, None)
    return loss_sum / examples, correct / examples


def save_shadow(path: Path, shadow: dict[int, np.ndarray]) -> None:
    np.savez(path, **{f"layer_{key}": value for key, value in shadow.items()})


def load_shadow(path: Path, layers: list[object]) -> dict[int, np.ndarray]:
    archive = np.load(path)
    return {
        layer._index: np.asarray(archive[f"layer_{layer._index}"], dtype=np.float32).copy()
        for layer in layers
    }


def save_state(
    model: object,
    shadow: dict[int, np.ndarray],
    output_dir: Path,
    history: list[dict[str, float | int]],
    config: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_as(str(output_dir / "weights.csv"), mode=0)
    save_shadow(output_dir / "shadow.npz", shadow)
    report = {
        "source": "native_ldnn_cifar10_update_rule_ablation",
        "gradient_backend": "native_ldnn_metal_slope_stable_output_softmax",
        "ldnn_softmax_divisor_no_conv_normalization": 1.0,
        "config": config,
        "history": history,
    }
    (output_dir / "weights.csv.json").write_text(json.dumps(report, indent=2) + "\n")
    history_path = output_dir / "history.csv"
    if history:
        with history_path.open("w", newline="") as stream:
            fieldnames = list(dict.fromkeys(key for row in history for key in row))
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

    previous_best = max(
        (float(item["test_accuracy"]) for item in history[:-1]), default=-1.0
    )
    if float(history[-1]["test_accuracy"]) > previous_best:
        model.save_as(str(output_dir / "best_weights.csv"), mode=0)
        save_shadow(output_dir / "best_shadow.npz", shadow)
        (output_dir / "best_weights.csv.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            f"new_best epoch={history[-1]['epoch']} "
            f"test_acc={float(history[-1]['test_accuracy']):.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-rule", choices=UPDATE_RULES, default="same_sign")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ldnn-path", type=Path, default=DEFAULT_LDNN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--max-steps", type=int, help="optimizer updates per epoch")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0.0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive")
    if args.epsilon < 0.0 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError("epsilon must be nonnegative and max-steps must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = make_dataset(args.data_root, train=True)
    test_data = make_dataset(args.data_root, train=False)
    core, platform = load_ldnn(args.ldnn_path, with_platform=True)
    model = build_model(
        core, gpu=platform.getGpu(0), batch_size=args.batch_size, seed=args.seed
    )
    model.set_backpropagation(True, 0.0)
    model.output.set_softmax_scale(1.0)
    layers = weighted_layers(model, core)
    temporary_buffers = track_temporary_metal_buffers(model._gpu)

    weights_path = args.output_dir / "weights.csv"
    shadow_path = args.output_dir / "shadow.npz"
    report_path = args.output_dir / "weights.csv.json"
    history: list[dict[str, float | int]] = []
    global_step = 0
    start_epoch = 1
    if not args.reset and weights_path.is_file() and shadow_path.is_file():
        model.load(str(weights_path), mode=0)
        model.update_weight()
        shadow = load_shadow(shadow_path, layers)
        if report_path.is_file():
            history = list(json.loads(report_path.read_text()).get("history", []))
        global_step = sum(int(item["steps"]) for item in history)
        start_epoch = int(history[-1]["epoch"]) + 1 if history else 1
        print(f"resumed={weights_path} start_epoch={start_epoch}")
    else:
        shadow = {
            layer._index: layer._weight_matrix.astype(np.float32).copy()
            for layer in layers
        }
        print("initialized=fan_in_aware_ldnn_9_level shadow=quantized_weights")

    config = {
        "update_rule": args.update_rule,
        "epochs_this_run": args.epochs,
        "batch_size_per_gradient": args.batch_size,
        "gradient_batches_per_step": 1 if args.update_rule == "single" else 2,
        "examples_per_step": args.batch_size * (1 if args.update_rule == "single" else 2),
        "learning_rate": args.learning_rate,
        "learning_rate_schedule": "constant",
        "epsilon": args.epsilon,
        "seed": args.seed,
        "train_dataset_size": len(train_data),
        "processed_training_images_per_full_epoch": (
            len(train_data) // args.batch_size * args.batch_size
        ),
        "dropped_training_images_per_full_epoch": len(train_data) % args.batch_size,
        "shuffle_seed_per_epoch": "seed + epoch",
        "augmentation": False,
        "normalization_mean": list(CIFAR10_MEAN),
        "normalization_std": list(CIFAR10_STD),
        "softmax_divisor": 1.0,
        "quantization": "nearest_native_ldnn_9_level_codebook_after_each_update",
        "shadow_clip": "native_codebook_min_max_after_each_update",
        "selection": "same_nonzero_sign" if args.update_rule == "same_sign" else "all",
        "optimizer": "shadow_sgd_then_nearest_9_level_quantization",
        "gradient_backend": "native_ldnn_metal_slope_stable_output_softmax",
        "bias": False,
    }
    started = time.monotonic()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        loader = make_train_loader(train_data, args.batch_size, args.seed, epoch)
        iterator = iter(loader)
        gradient_batches_per_step = 1 if args.update_rule == "single" else 2
        available_steps = len(loader) // gradient_batches_per_step
        steps = min(available_steps, args.max_steps or available_steps)
        loss_sum = 0.0
        correct_sum = 0
        example_sum = 0
        selected_sum = 0
        changed_sum = 0

        for step_in_epoch in range(1, steps + 1):
            first = gradients_for_batch(
                model, layers, *next(iterator), 1.0, temporary_buffers
            )
            second = (
                gradients_for_batch(
                    model, layers, *next(iterator), 1.0, temporary_buffers
                )
                if gradient_batches_per_step == 2
                else None
            )
            update = update_weights(
                model,
                layers,
                shadow,
                first[3],
                None if second is None else second[3],
                args.learning_rate,
                args.epsilon,
                args.update_rule,
            )
            global_step += 1
            loss_sum += first[0] * first[2]
            correct_sum += first[1]
            example_sum += first[2]
            if second is not None:
                loss_sum += second[0] * second[2]
                correct_sum += second[1]
                example_sum += second[2]
            selected_sum += int(update["selected"])
            changed_sum += int(update["quantized_changes"])
            if global_step % args.log_every == 0 or step_in_epoch == steps:
                print(
                    f"epoch={epoch} step={step_in_epoch}/{steps} "
                    f"global_step={global_step} "
                    f"loss={(first[0] if second is None else (first[0] + second[0]) / 2):.6f} "
                    f"selected={update['selected']} "
                    f"quantized_changes={update['quantized_changes']}"
                )
            if global_step % 10 == 0:
                gc.collect()

        test_loss, test_accuracy = evaluate(
            model, test_data, args.batch_size, 1.0, temporary_buffers
        )
        metrics: dict[str, float | int] = {
            "epoch": epoch,
            "steps": steps,
            "processed_training_images": example_sum,
            "train_loss": loss_sum / example_sum,
            "train_accuracy": correct_sum / example_sum,
            "mean_selected_components": selected_sum / steps,
            "selected_fraction": (
                selected_sum / (steps * int(update["total_components"]))
                if args.update_rule == "same_sign"
                else 1.0
            ),
            "mean_quantized_changes": changed_sum / steps,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(metrics)
        save_state(model, shadow, args.output_dir, history, config)
        print(
            f"saved epoch={epoch} train_loss={metrics['train_loss']:.6f} "
            f"train_acc={metrics['train_accuracy']:.4f} "
            f"test_loss={test_loss:.6f} test_acc={test_accuracy:.4f} "
            f"output={args.output_dir}"
        )


if __name__ == "__main__":
    main()
