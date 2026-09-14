"""Train native LDNN on CIFAR-10 with accepted adjacent index mutations."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import objc
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ldnn_cifar10_model import DEFAULT_LDNN_PATH, NUM_CLASSES, build_model, load_ldnn
from verify_pytorch_batch_ldnn import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    DEFAULT_DATA_ROOT,
    pytorch_batch_to_ldnn,
    read_logits,
)


def make_train_loader(data_root: Path, batch_size: int, seed: int) -> DataLoader:
    dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        ),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=True,
    )


def stable_cross_entropy(logits: np.ndarray, targets: np.ndarray, divisor: float) -> float:
    scaled = logits.astype(np.float64) / divisor
    maximum = scaled.max(axis=1, keepdims=True)
    log_sum_exp = maximum[:, 0] + np.log(np.exp(scaled - maximum).sum(axis=1))
    return float(np.mean(log_sum_exp - scaled[np.arange(targets.size), targets]))


def forward_loss(model: object, targets: np.ndarray, divisor: float) -> tuple[float, float]:
    model.reset()
    model.propagate()
    logits = read_logits(model)
    if not np.isfinite(logits).all():
        raise RuntimeError("LDNN logits contain NaN or infinity")
    loss = stable_cross_entropy(logits, targets, divisor)
    accuracy = float(np.mean(logits.argmax(axis=1) == targets))
    return loss, accuracy


def propose_random_mutations(
    model: object,
    core: object,
    mutation_fraction: float,
    generator: np.random.Generator,
) -> list[tuple[object, int, int, int]]:
    """Select weights uniformly at random and move to a random neighbor."""
    changes: list[tuple[object, int, int, int]] = []
    weighted_types = (core.LAYER_TYPE_CONV, core.LAYER_TYPE_OUTPUT)
    layers = [layer for layer in model.layers if layer.get_type() in weighted_types]
    layer_sizes = np.asarray(
        [layer._num_node * layer._num_input for layer in layers], dtype=np.int64
    )
    offsets = np.cumsum(layer_sizes)
    total_weights = int(offsets[-1])
    mutation_count = min(
        total_weights, max(1, int(math.ceil(total_weights * mutation_fraction)))
    )
    positions = generator.choice(total_weights, size=mutation_count, replace=False)

    for position in positions:
        layer_number = int(np.searchsorted(offsets, position, side="right"))
        layer = layers[layer_number]
        layer_start = 0 if layer_number == 0 else int(offsets[layer_number - 1])
        local_position = int(position) - layer_start
        node_index = local_position // layer._num_input
        input_index = local_position % layer._num_input
        old_index = int(layer.get_weight_index(node_index, input_index))
        minimum = layer.bias_index_min()
        maximum = layer.bias_index_max()
        neighbors = []
        if old_index > minimum:
            neighbors.append(old_index - 1)
        if old_index < maximum:
            neighbors.append(old_index + 1)
        new_index = neighbors[int(generator.integers(len(neighbors)))]
        changes.append((layer, node_index, input_index, old_index))
        layer.set_weight_index(node_index, input_index, new_index)
    return changes


def track_temporary_metal_buffers(gpu: object) -> list[object]:
    """Track allocations made after model preparation for per-batch release."""
    buffers: list[object] = []
    original_alloc_buf = gpu.alloc_buf
    original_alloc_buf_from_array = gpu.alloc_buf_from_array

    def tracked_alloc_buf(nbytes: int) -> object:
        buffer = original_alloc_buf(nbytes)
        buffers.append(buffer)
        return buffer

    def tracked_alloc_buf_from_array(array: np.ndarray) -> object:
        buffer = original_alloc_buf_from_array(array)
        buffers.append(buffer)
        return buffer

    gpu.alloc_buf = tracked_alloc_buf
    gpu.alloc_buf_from_array = tracked_alloc_buf_from_array
    return buffers


def release_temporary_metal_buffers(buffers: list[object]) -> None:
    """Release ownership returned by Metal's newBuffer... selectors."""
    for buffer in buffers:
        buffer.release()
    buffers.clear()


def undo_mutations(changes: list[tuple[object, int, int, int]]) -> None:
    for layer, node_index, input_index, old_index in changes:
        layer.set_weight_index(node_index, input_index, old_index)


def save_training_state(
    model: object,
    weights: Path,
    history: list[dict[str, float | int]],
    softmax_divisor: float,
    config: dict[str, object],
) -> None:
    weights.parent.mkdir(parents=True, exist_ok=True)
    model.save_as(str(weights), mode=0)
    report = {
        "source": "native_ldnn_training",
        "ldnn_softmax_divisor_no_conv_normalization": softmax_divisor,
        "config": config,
        "history": history,
    }
    weights.with_suffix(weights.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("runs/ldnn_cifar10/weights.csv"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ldnn-path", type=Path, default=DEFAULT_LDNN_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--mutation-fraction", type=float, default=0.0001)
    parser.add_argument("--softmax-divisor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="save an in-epoch weight checkpoint every N batches; 0 disables it",
    )
    parser.add_argument(
        "--gc-every",
        type=int,
        default=10,
        help="drain Python garbage periodically in addition to each Metal autorelease pool",
    )
    parser.add_argument("--max-batches", type=int, help="limit batches per epoch for a smoke test")
    parser.add_argument(
        "--reset-weights",
        action="store_true",
        help="ignore saved weights and start from a new fan-in-aware initialization",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if not 0.0 < args.mutation_fraction <= 1.0:
        raise ValueError("mutation fraction must be in (0, 1]")
    if args.softmax_divisor <= 0.0:
        raise ValueError("softmax divisor must be positive")
    if args.save_every < 0 or args.gc_every < 1:
        raise ValueError("save-every must be nonnegative and gc-every must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)

    loader = make_train_loader(args.data_root, args.batch_size, args.seed)
    core, platform = load_ldnn(args.ldnn_path, with_platform=True)
    model = build_model(
        core, gpu=platform.getGpu(0), batch_size=args.batch_size, seed=args.seed
    )
    model.set_backpropagation(False, 0.0)
    model.output.set_softmax_scale(args.softmax_divisor)
    mutation_generator = np.random.default_rng(args.seed)

    # build_model() has already allocated all persistent model buffers.  Any
    # later alloc_buf call is scratch storage created by Metal forward kernels
    # and must be released explicitly after each batch.
    temporary_metal_buffers = track_temporary_metal_buffers(model._gpu)

    resume_weights = args.weights.is_file() and not args.reset_weights
    if resume_weights:
        model.load(str(args.weights), mode=0)
        model.update_weight()
        print(f"resumed_weights={args.weights}")
    else:
        print(f"initialized_new_weights={args.weights}")

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "mutation_fraction": args.mutation_fraction,
        "mutation_strategy": "uniform_random_adjacent",
        "uses_gradients": False,
        "softmax_divisor": args.softmax_divisor,
        "seed": args.seed,
        "save_every": args.save_every,
        "gc_every": args.gc_every,
        "input_normalized": True,
        "conv_batchnorm": False,
        "conv_scale_layer": False,
    }
    history: list[dict[str, float | int]] = []
    training_report_path = args.weights.with_suffix(args.weights.suffix + ".json")
    if resume_weights and training_report_path.is_file():
        previous_report = json.loads(training_report_path.read_text())
        if previous_report.get("source") == "native_ldnn_training":
            history = list(previous_report.get("history", []))
            print(f"resumed_history_epochs={len(history)}")
    accepted_total = 0
    proposed_total = 0
    global_step = sum(int(item.get("steps", 0)) for item in history)
    start_epoch = int(history[-1]["epoch"]) + 1 if history else 1
    started = time.monotonic()

    batches_per_epoch = len(loader)
    if args.max_batches is not None:
        if args.max_batches < 1:
            raise ValueError("max batches must be positive")
        batches_per_epoch = min(batches_per_epoch, args.max_batches)

    for run_epoch in range(1, args.epochs + 1):
        epoch = start_epoch + run_epoch - 1
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_accepts = 0
        for batch_index, (images, labels) in enumerate(loader, start=1):
            if batch_index > batches_per_epoch:
                break
            # Metal forward kernels allocate temporary MTLBuffers. PyObjC
            # otherwise retains their Objective-C ownership during long runs.
            metal_pool = objc.autorelease_pool()
            metal_pool.__enter__()
            data, one_hot = pytorch_batch_to_ldnn(images, labels)
            targets = labels.numpy().astype(np.int64, copy=False)
            model.direct_set_data(data)
            model.direct_set_label(one_hot)

            loss_before, accuracy_before = forward_loss(
                model, targets, args.softmax_divisor
            )
            changes = propose_random_mutations(
                model, core, args.mutation_fraction, mutation_generator
            )
            if changes:
                proposed_total += 1
                model.update_weight()
                loss_after, accuracy_after = forward_loss(
                    model, targets, args.softmax_divisor
                )
                accepted = loss_after < loss_before
                if accepted:
                    accepted_total += 1
                    epoch_accepts += 1
                    selected_loss = loss_after
                    selected_accuracy = accuracy_after
                else:
                    undo_mutations(changes)
                    model.update_weight()
                    selected_loss = loss_before
                    selected_accuracy = accuracy_before
            else:
                accepted = False
                selected_loss = loss_before
                selected_accuracy = accuracy_before

            global_step += 1
            epoch_loss += selected_loss
            epoch_accuracy += selected_accuracy
            if global_step % args.log_every == 0 or batch_index == batches_per_epoch:
                print(
                    f"epoch={epoch} run_epoch={run_epoch}/{args.epochs} "
                    f"batch={batch_index}/{batches_per_epoch} "
                    f"step={global_step} loss={selected_loss:.6f} "
                    f"improvement={loss_before - selected_loss:.6f} "
                    f"accuracy={selected_accuracy:.4f} mutations={len(changes)} "
                    f"accepted={accepted} "
                    f"acceptance={accepted_total / max(proposed_total, 1):.3f}"
                )
            release_temporary_metal_buffers(temporary_metal_buffers)
            metal_pool.__exit__(None, None, None)
            del metal_pool
            if global_step % args.gc_every == 0:
                gc.collect()
            if (
                args.save_every > 0
                and batch_index % args.save_every == 0
                and batch_index != batches_per_epoch
            ):
                save_training_state(
                    model, args.weights, history, args.softmax_divisor, config
                )
                print(
                    f"checkpoint epoch={epoch} batch={batch_index}/{batches_per_epoch} "
                    f"weights={args.weights}"
                )

        metrics: dict[str, float | int] = {
            "epoch": epoch,
            "steps": batches_per_epoch,
            "mean_train_loss": epoch_loss / batches_per_epoch,
            "mean_train_accuracy": epoch_accuracy / batches_per_epoch,
            "accepted_batches": epoch_accepts,
            "acceptance": accepted_total / max(proposed_total, 1),
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(metrics)
        save_training_state(model, args.weights, history, args.softmax_divisor, config)
        print(
            f"saved epoch={epoch} weights={args.weights} "
            f"mean_loss={metrics['mean_train_loss']:.6f} "
            f"mean_accuracy={metrics['mean_train_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
