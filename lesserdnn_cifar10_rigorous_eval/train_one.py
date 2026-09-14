"""Train and evaluate one LDNN CIFAR-10 rule without test-set model selection."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import objc
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT.parent / "lesserdnn_cifar10_ablation_200"
EXPERIMENT_SOURCE = SOURCE_ROOT / "experiments" / "ldnn_cifar10_native_same_sign"
for path in (SOURCE_ROOT, EXPERIMENT_SOURCE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ldnn_cifar10_model import NUM_CLASSES, build_model, load_ldnn
from train import gradients_for_batch, load_shadow, save_shadow, update_weights, weighted_layers
from train_ldnn_cifar10 import release_temporary_metal_buffers, track_temporary_metal_buffers
from verify_pytorch_batch_ldnn import CIFAR10_MEAN, CIFAR10_STD, pytorch_batch_to_ldnn
from lr_schedule import parse_learning_rate_schedule


RULES = ("single", "two_batch_average", "same_sign")
CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-rule", required=True, choices=RULES)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--split-seed", type=int, default=20260826)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--target-epochs", type=int, default=300)
    parser.add_argument("--lr-switch-epoch", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--continuation-learning-rate", type=float, default=0.003)
    parser.add_argument(
        "--lr-schedule",
        help=(
            "piecewise schedule as end_epoch:rate pairs; for example "
            "100:0.01,200:0.005,300:0.001. When set, this overrides the "
            "two-stage learning-rate arguments"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT.parent / "data")
    parser.add_argument("--ldnn-path", type=Path, default=SOURCE_ROOT / "ldnn_runtime")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--max-gradient-batches",
        type=int,
        help="smoke-test limit; must be even so all rules process the same examples",
    )
    return parser.parse_args()


def normalized_dataset(data_root: Path, train: bool) -> object:
    return datasets.CIFAR10(
        root=data_root,
        train=train,
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
        ),
    )


def stratified_split(targets: list[int], validation_size: int, split_seed: int) -> tuple[list[int], list[int]]:
    if validation_size <= 0 or validation_size >= len(targets):
        raise ValueError("validation-size must be between 1 and dataset size - 1")
    if validation_size % NUM_CLASSES:
        raise ValueError("validation-size must be divisible by 10 for a balanced split")
    per_class = validation_size // NUM_CLASSES
    labels = np.asarray(targets, dtype=np.int64)
    generator = np.random.default_rng(split_seed)
    validation: list[int] = []
    training: list[int] = []
    for class_id in range(NUM_CLASSES):
        indexes = np.flatnonzero(labels == class_id)
        indexes = indexes[generator.permutation(indexes.size)]
        validation.extend(int(index) for index in indexes[:per_class])
        training.extend(int(index) for index in indexes[per_class:])
    # Sorting makes the split independent of implementation details. Shuffling is
    # performed later with the per-run, per-epoch generator.
    return sorted(training), sorted(validation)


def train_loader(dataset: object, indexes: list[int], batch_size: int, seed: int, epoch: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
    return DataLoader(
        Subset(dataset, indexes), batch_size=batch_size, shuffle=True,
        generator=generator, num_workers=0, drop_last=True,
    )


def evaluate(model: object, dataset: object, indexes: list[int] | None, batch_size: int,
             temporary_buffers: list[object], keep_predictions: bool = False) -> dict[str, object]:
    evaluation_data = dataset if indexes is None else Subset(dataset, indexes)
    loader = DataLoader(evaluation_data, batch_size=batch_size, shuffle=False, num_workers=0)
    loss_sum = 0.0
    correct = 0
    examples = 0
    class_total = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for images, labels in loader:
        count = labels.numel()
        if count < batch_size:
            images = torch.cat(
                (images, torch.zeros((batch_size - count, *images.shape[1:]), dtype=images.dtype)),
                dim=0,
            )
            labels = torch.cat((labels, torch.zeros(batch_size - count, dtype=labels.dtype)), dim=0)
        pool = objc.autorelease_pool()
        pool.__enter__()
        try:
            data, one_hot = pytorch_batch_to_ldnn(images, labels)
            model.reset()
            model.direct_set_data(data)
            model.direct_set_label(one_hot)
            model.propagate()
            logits = model._gpu.read_mbuf_to_numpy(
                model.output._gpu_output, np.float32, (batch_size, NUM_CLASSES)
            )[:count].astype(np.float64)
            if not np.isfinite(logits).all():
                raise RuntimeError("evaluation logits contain NaN or infinity")
            targets = labels[:count].numpy().astype(np.int64, copy=False)
            predictions = logits.argmax(axis=1)
            shifted = logits - logits.max(axis=1, keepdims=True)
            log_sum_exp = np.log(np.exp(shifted).sum(axis=1))
            loss_sum += float(np.sum(log_sum_exp - shifted[np.arange(count), targets]))
            matches = predictions == targets
            correct += int(matches.sum())
            examples += count
            np.add.at(class_total, targets, 1)
            np.add.at(class_correct, targets, matches.astype(np.int64))
            np.add.at(confusion, (targets, predictions), 1)
            if keep_predictions:
                all_targets.append(targets.copy())
                all_predictions.append(predictions.copy())
        finally:
            release_temporary_metal_buffers(temporary_buffers)
            pool.__exit__(None, None, None)

    result: dict[str, object] = {
        "samples": examples,
        "correct": correct,
        "loss": loss_sum / examples,
        "accuracy": correct / examples,
        "class_names": list(CLASS_NAMES),
        "class_total": class_total.tolist(),
        "class_correct": class_correct.tolist(),
        "class_accuracy": np.divide(
            class_correct, class_total, out=np.zeros(NUM_CLASSES), where=class_total != 0
        ).tolist(),
        "confusion_matrix": confusion.tolist(),
    }
    if keep_predictions:
        result["targets"] = np.concatenate(all_targets)
        result["predictions"] = np.concatenate(all_predictions)
    return result


def learning_rate_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    if args.learning_rate_segments is not None:
        for segment in args.learning_rate_segments:
            if int(segment["start"]) <= epoch <= int(segment["end"]):
                return float(segment["learning_rate"])
        raise RuntimeError(f"epoch {epoch} is not covered by the learning-rate schedule")
    return args.learning_rate if epoch <= args.lr_switch_epoch else args.continuation_learning_rate


def indexes_digest(indexes: list[int]) -> str:
    data = np.asarray(indexes, dtype="<i4")
    return hashlib.sha256(data.tobytes()).hexdigest()


def config_for(args: argparse.Namespace, train_indexes: list[int],
               validation_indexes: list[int], gradient_batches: int) -> dict[str, object]:
    config: dict[str, object] = {
        "protocol_version": 1,
        "update_rule": args.update_rule,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "train_size": len(train_indexes),
        "validation_size": len(validation_indexes),
        "train_indexes_sha256": indexes_digest(train_indexes),
        "validation_indexes_sha256": indexes_digest(validation_indexes),
        "test_size": 10_000,
        "target_epochs": args.target_epochs,
        "batch_size_per_gradient": args.batch_size,
        "gradient_batches_per_epoch": gradient_batches,
        "processed_training_images_per_epoch": gradient_batches * args.batch_size,
        "optimizer_steps_per_epoch": gradient_batches if args.update_rule == "single" else gradient_batches // 2,
        "epsilon": args.epsilon,
        "augmentation": False,
        "normalization_mean": list(CIFAR10_MEAN),
        "normalization_std": list(CIFAR10_STD),
        "quantization": "nearest_native_ldnn_9_level_codebook_after_each_update",
        "selection_metric": "validation_accuracy_then_validation_loss",
        "test_policy": "evaluate_once_after_training_using_validation_selected_checkpoint",
        "max_gradient_batches": args.max_gradient_batches,
    }
    if args.learning_rate_segments is None:
        config.update(
            {
                "lr_switch_epoch": args.lr_switch_epoch,
                "learning_rate": args.learning_rate,
                "continuation_learning_rate": args.continuation_learning_rate,
            }
        )
    else:
        config["learning_rate_segments"] = args.learning_rate_segments
    return config


def write_history(path: Path, history: list[dict[str, object]]) -> None:
    if not history:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def save_latest(model: object, shadow: dict[int, np.ndarray], output_dir: Path,
                config: dict[str, object], history: list[dict[str, object]],
                best: dict[str, object] | None, training_seconds: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_as(str(output_dir / "latest_weights.csv"), mode=0)
    save_shadow(output_dir / "latest_shadow.npz", shadow)
    state = {
        "config": config,
        "history": history,
        "best_validation": best,
        "training_seconds": training_seconds,
    }
    (output_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n")
    write_history(output_dir / "history.csv", history)


def validate_resume(saved: dict[str, object], expected: dict[str, object]) -> None:
    actual = saved.get("config", {})
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"resume configuration mismatch: {mismatches}")


def main() -> None:
    args = parse_args()
    if args.target_epochs < 1 or args.batch_size < 1 or args.log_every < 1:
        raise ValueError("target-epochs, batch-size, and log-every must be positive")
    args.learning_rate_segments = (
        parse_learning_rate_schedule(args.lr_schedule, args.target_epochs)
        if args.lr_schedule is not None
        else None
    )
    if args.learning_rate_segments is None and not 0 <= args.lr_switch_epoch <= args.target_epochs:
        raise ValueError("lr-switch-epoch must be in [0, target-epochs]")
    if args.learning_rate <= 0 or args.continuation_learning_rate <= 0 or args.epsilon < 0:
        raise ValueError("learning rates must be positive and epsilon nonnegative")
    if args.max_gradient_batches is not None and (
        args.max_gradient_batches < 2 or args.max_gradient_batches % 2
    ):
        raise ValueError("max-gradient-batches must be a positive even number")
    if not (args.data_root / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(f"CIFAR-10 not found under {args.data_root}")
    if not (args.ldnn_path / "core.py").is_file():
        raise FileNotFoundError(f"LDNN runtime not found under {args.ldnn_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    full_train = normalized_dataset(args.data_root, train=True)
    test_data = normalized_dataset(args.data_root, train=False)
    train_indexes, validation_indexes = stratified_split(
        full_train.targets, args.validation_size, args.split_seed
    )
    available_batches = len(train_indexes) // args.batch_size
    gradient_batches = available_batches - available_batches % 2
    if args.max_gradient_batches is not None:
        gradient_batches = min(gradient_batches, args.max_gradient_batches)
    if gradient_batches < 2:
        raise RuntimeError("not enough full gradient batches")

    config = config_for(args, train_indexes, validation_indexes, gradient_batches)
    core, platform = load_ldnn(args.ldnn_path, with_platform=True)
    model = build_model(core, gpu=platform.getGpu(0), batch_size=args.batch_size, seed=args.seed)
    model.set_backpropagation(True, 0.0)
    model.output.set_softmax_scale(1.0)
    layers = weighted_layers(model, core)
    temporary_buffers = track_temporary_metal_buffers(model._gpu)

    state_path = args.output_dir / "state.json"
    latest_weights = args.output_dir / "latest_weights.csv"
    latest_shadow = args.output_dir / "latest_shadow.npz"
    history: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    prior_training_seconds = 0.0
    if state_path.is_file():
        if not latest_weights.is_file() or not latest_shadow.is_file():
            raise RuntimeError("state exists but latest weights/shadow are missing")
        state = json.loads(state_path.read_text())
        validate_resume(state, config)
        history = list(state.get("history", []))
        epochs = [int(row["epoch"]) for row in history]
        if epochs != list(range(1, len(epochs) + 1)):
            raise RuntimeError("saved epoch history is not contiguous")
        best = state.get("best_validation")
        prior_training_seconds = float(state.get("training_seconds", 0.0))
        model.load(str(latest_weights), mode=0)
        model.update_weight()
        shadow = load_shadow(latest_shadow, layers)
        print(f"resumed={args.output_dir} completed_epochs={len(history)}")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise RuntimeError(f"nonempty output directory has no resumable state: {args.output_dir}")
        shadow = {layer._index: layer._weight_matrix.astype(np.float32).copy() for layer in layers}
        print("initialized=fan_in_aware_ldnn_9_level")

    started = time.monotonic()
    for epoch in range(len(history) + 1, args.target_epochs + 1):
        loader = train_loader(full_train, train_indexes, args.batch_size, args.seed, epoch)
        iterator = iter(loader)
        gradient_batches_per_step = 1 if args.update_rule == "single" else 2
        steps = gradient_batches // gradient_batches_per_step
        loss_sum = 0.0
        correct_sum = 0
        example_sum = 0
        selected_sum = 0
        changed_sum = 0
        lr = learning_rate_for_epoch(args, epoch)
        epoch_started = time.monotonic()

        for step in range(1, steps + 1):
            first = gradients_for_batch(model, layers, *next(iterator), 1.0, temporary_buffers)
            second = (
                gradients_for_batch(model, layers, *next(iterator), 1.0, temporary_buffers)
                if gradient_batches_per_step == 2 else None
            )
            update = update_weights(
                model, layers, shadow, first[3], None if second is None else second[3],
                lr, args.epsilon, args.update_rule,
            )
            for item in (first, second):
                if item is not None:
                    loss_sum += item[0] * item[2]
                    correct_sum += item[1]
                    example_sum += item[2]
            selected_sum += int(update["selected"])
            changed_sum += int(update["quantized_changes"])
            if step % args.log_every == 0 or step == steps:
                print(
                    f"rule={args.update_rule} seed={args.seed} epoch={epoch} "
                    f"step={step}/{steps} lr={lr:g} selected={update['selected']} "
                    f"quantized_changes={update['quantized_changes']}"
                )
            if step % 10 == 0:
                gc.collect()

        validation = evaluate(
            model, full_train, validation_indexes, args.batch_size, temporary_buffers
        )
        row: dict[str, object] = {
            "epoch": epoch,
            "learning_rate": lr,
            "steps": steps,
            "processed_training_images": example_sum,
            "train_loss": loss_sum / example_sum,
            "train_accuracy": correct_sum / example_sum,
            "mean_selected_components": selected_sum / steps,
            "selected_fraction": selected_sum / (steps * int(update["total_components"])),
            "mean_quantized_changes": changed_sum / steps,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        history.append(row)
        is_better = best is None or float(row["validation_accuracy"]) > float(best["accuracy"])
        if best is not None and float(row["validation_accuracy"]) == float(best["accuracy"]):
            is_better = float(row["validation_loss"]) < float(best["loss"])
        if is_better:
            best = {"epoch": epoch, "accuracy": row["validation_accuracy"], "loss": row["validation_loss"]}
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model.save_as(str(args.output_dir / "best_weights.csv"), mode=0)
            save_shadow(args.output_dir / "best_shadow.npz", shadow)
            print(
                f"new_validation_best rule={args.update_rule} seed={args.seed} epoch={epoch} "
                f"accuracy={float(row['validation_accuracy']):.4f}"
            )
        training_seconds = prior_training_seconds + time.monotonic() - started
        save_latest(model, shadow, args.output_dir, config, history, best, training_seconds)
        split_path = args.output_dir / "split_indices.npz"
        if not split_path.is_file():
            np.savez_compressed(
                split_path,
                train=np.asarray(train_indexes, dtype=np.int32),
                validation=np.asarray(validation_indexes, dtype=np.int32),
            )
        print(
            f"saved rule={args.update_rule} seed={args.seed} epoch={epoch} "
            f"train_acc={float(row['train_accuracy']):.4f} "
            f"validation_acc={float(row['validation_accuracy']):.4f}"
        )

    if best is None or not (args.output_dir / "best_weights.csv").is_file():
        raise RuntimeError("best validation checkpoint is missing")
    result_path = args.output_dir / "result.json"
    predictions_path = args.output_dir / "test_predictions.npz"
    if result_path.is_file() and predictions_path.is_file():
        existing = json.loads(result_path.read_text())
        if int(existing.get("evaluated_best_epoch", -1)) == int(best["epoch"]):
            print(f"test_already_evaluated={result_path}")
            return

    model.load(str(args.output_dir / "best_weights.csv"), mode=0)
    model.update_weight()
    test = evaluate(model, test_data, None, args.batch_size, temporary_buffers, keep_predictions=True)
    targets = test.pop("targets")
    predictions = test.pop("predictions")
    np.savez_compressed(predictions_path, targets=targets, predictions=predictions)
    final_state = json.loads(state_path.read_text())
    result = {
        "update_rule": args.update_rule,
        "seed": args.seed,
        "evaluated_best_epoch": best["epoch"],
        "best_validation_accuracy": best["accuracy"],
        "best_validation_loss": best["loss"],
        "test": test,
        "training_seconds": final_state["training_seconds"],
        "config": config,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"final_test_once rule={args.update_rule} seed={args.seed} "
        f"best_epoch={best['epoch']} accuracy={float(test['accuracy']):.4f} "
        f"loss={float(test['loss']):.6f} result={result_path}"
    )


if __name__ == "__main__":
    main()
