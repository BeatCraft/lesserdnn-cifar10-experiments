"""Train one full-precision PyTorch CIFAR-10 reference run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from lr_schedule import parse_learning_rate_schedule

ROOT = Path(__file__).resolve().parent
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
NUM_CLASSES = 10
DATASET_NAME = "CIFAR-10"
EXPECTED_PARAMETER_COUNT = 174_944


class ReferenceCNN(nn.Module):
    """Full-precision counterpart of the bias-free LesserDNN CIFAR-10 model."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.ReLU(),
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
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))


def dataset_available(root: Path) -> bool:
    return (root / "cifar-10-batches-py").is_dir()


def stratified_split(targets: object, validation_size: int, split_seed: int) -> tuple[list[int], list[int]]:
    if validation_size <= 0 or validation_size >= len(targets) or validation_size % NUM_CLASSES:
        raise ValueError("validation-size must be a positive multiple of 10 smaller than the training dataset")
    labels = np.asarray(targets, dtype=np.int64)
    generator = np.random.default_rng(split_seed)
    training, validation = [], []
    for label in range(NUM_CLASSES):
        indexes = np.flatnonzero(labels == label)
        indexes = indexes[generator.permutation(indexes.size)]
        validation.extend(int(x) for x in indexes[: validation_size // NUM_CLASSES])
        training.extend(int(x) for x in indexes[validation_size // NUM_CLASSES :])
    return sorted(training), sorted(validation)


def train_loader(data: object, indexes: list[int], batch_size: int, seed: int, epoch: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed * 1_000_003 + epoch)
    return DataLoader(Subset(data, indexes), batch_size=batch_size, shuffle=True,
                      generator=generator, num_workers=0, drop_last=True)


def eval_loader(data: object, indexes: list[int] | None, batch_size: int) -> DataLoader:
    source = data if indexes is None else Subset(data, indexes)
    return DataLoader(source, batch_size=batch_size, shuffle=False, num_workers=0)


def choose_device(specification: str) -> torch.device:
    if specification == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if specification == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(specification)


def initialize(model: nn.Module) -> None:
    for layer in model.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")


def evaluate(model: nn.Module, batches: DataLoader, device: torch.device,
             keep_predictions: bool = False) -> dict[str, object]:
    model.eval()
    loss_sum = correct = samples = 0
    class_total = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    all_targets, all_predictions = [], []
    with torch.no_grad():
        for images, labels in batches:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_sum += float(nn.functional.cross_entropy(logits, labels, reduction="sum").item())
            guesses = logits.argmax(dim=1)
            host_targets = labels.cpu().numpy(); host_guesses = guesses.cpu().numpy()
            matches = host_guesses == host_targets
            correct += int(matches.sum()); samples += labels.numel()
            np.add.at(class_total, host_targets, 1)
            np.add.at(class_correct, host_targets, matches.astype(np.int64))
            np.add.at(confusion, (host_targets, host_guesses), 1)
            if keep_predictions:
                all_targets.append(host_targets); all_predictions.append(host_guesses)
    result: dict[str, object] = {
        "samples": samples, "correct": correct, "loss": loss_sum / samples,
        "accuracy": correct / samples, "class_names": [str(x) for x in range(10)],
        "class_total": class_total.tolist(), "class_correct": class_correct.tolist(),
        "class_accuracy": np.divide(class_correct, class_total, out=np.zeros(10), where=class_total != 0).tolist(),
        "confusion_matrix": confusion.tolist(),
    }
    if keep_predictions:
        result["targets"] = np.concatenate(all_targets)
        result["predictions"] = np.concatenate(all_predictions)
    return result


def indexes_digest(indexes: list[int]) -> str:
    return hashlib.sha256(np.asarray(indexes, dtype="<i4").tobytes()).hexdigest()


def write_history(path: Path, history: list[dict[str, object]]) -> None:
    if history:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)


def save_checkpoint(path: Path, model: nn.Module, history: list[dict[str, object]],
                    best: dict[str, object] | None, config: dict[str, object], seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "history": history, "best_validation": best, "config": config,
                "training_seconds": seconds}, temporary)
    temporary.replace(path)
    write_history(path.parent / "history.csv", history)


def main() -> None:
    args = arguments()
    segments = parse_learning_rate_schedule(args.lr_schedule, args.target_epochs)
    if args.batch_size < 1 or args.log_every < 1:
        raise ValueError("batch-size and log-every must be positive")
    if args.max_gradient_batches is not None and args.max_gradient_batches < 1:
        raise ValueError("max-gradient-batches must be positive")
    if not dataset_available(args.data_root):
        raise FileNotFoundError(f"{DATASET_NAME} not found under {args.data_root}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "mps": torch.mps.manual_seed(args.seed)
    full_train, test_data = make_dataset(args.data_root, True), make_dataset(args.data_root, False)
    train_indexes, validation_indexes = stratified_split(full_train.targets, args.validation_size, args.split_seed)
    # Match LesserDNN: discard the final odd full batch so every method sees 420 batches/epoch.
    gradient_batches = (len(train_indexes) // args.batch_size) // 2 * 2
    if args.max_gradient_batches is not None: gradient_batches = min(gradient_batches, args.max_gradient_batches)
    config = {
        "protocol_version": 1, "dataset": DATASET_NAME, "framework": "PyTorch",
        "reference": "full_precision_sgd", "seed": args.seed, "split_seed": args.split_seed,
        "train_size": len(train_indexes), "validation_size": len(validation_indexes), "test_size": len(test_data),
        "train_indexes_sha256": indexes_digest(train_indexes),
        "validation_indexes_sha256": indexes_digest(validation_indexes),
        "target_epochs": args.target_epochs, "learning_rate_segments": segments,
        "batch_size": args.batch_size, "gradient_batches_per_epoch": gradient_batches,
        "processed_training_images_per_epoch": gradient_batches * args.batch_size,
        "optimizer": "torch.optim.SGD(momentum=0,weight_decay=0)",
        "initialization": "torch_kaiming_normal_fan_in_relu", "precision": "float32",
        "bias": False, "augmentation": False, "normalization_mean": list(MEAN), "normalization_std": list(STD),
        "selection_metric": "validation_accuracy_then_validation_loss",
        "test_policy": "evaluate_once_after_training_using_validation_selected_checkpoint",
        "requested_device": args.device, "resolved_device": device.type,
        "max_gradient_batches": args.max_gradient_batches,
    }
    model = ReferenceCNN(); initialize(model); model.to(device)
    if sum(x.numel() for x in model.parameters()) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("reference architecture parameter count mismatch")
    optimizer = torch.optim.SGD(model.parameters(), lr=float(segments[0]["learning_rate"]))
    history: list[dict[str, object]] = []; best = None; prior_seconds = 0.0
    latest_path, best_path = args.output_dir / "latest.pt", args.output_dir / "best.pt"
    if latest_path.is_file():
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        if checkpoint["config"] != config: raise RuntimeError("resume configuration mismatch")
        history = list(checkpoint["history"]); best = checkpoint["best_validation"]
        prior_seconds = float(checkpoint["training_seconds"])
        if [int(x["epoch"]) for x in history] != list(range(1, len(history) + 1)):
            raise RuntimeError("saved epoch history is not contiguous")
        model.load_state_dict(checkpoint["model_state"]); model.to(device)
        print(f"resumed={args.output_dir} completed_epochs={len(history)}")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"nonempty output directory is not resumable: {args.output_dir}")

    started = time.monotonic()
    for epoch in range(len(history) + 1, args.target_epochs + 1):
        rate = next(float(x["learning_rate"]) for x in segments if int(x["start"]) <= epoch <= int(x["end"]))
        optimizer.param_groups[0]["lr"] = rate
        model.train(); epoch_started = time.monotonic(); loss_sum = correct = samples = 0
        for step, (images, labels) in enumerate(train_loader(full_train, train_indexes, args.batch_size, args.seed, epoch), 1):
            if step > gradient_batches: break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True); logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels); loss.backward(); optimizer.step()
            loss_sum += float(loss.item()) * labels.numel(); correct += int((logits.argmax(1) == labels).sum().item()); samples += labels.numel()
            if step % args.log_every == 0 or step == gradient_batches:
                print(f"framework=pytorch seed={args.seed} epoch={epoch} step={step}/{gradient_batches} lr={rate:g}")
        validation = evaluate(model, eval_loader(full_train, validation_indexes, args.batch_size), device)
        row: dict[str, object] = {"epoch": epoch, "learning_rate": rate, "steps": gradient_batches,
            "processed_training_images": samples, "train_loss": loss_sum/samples, "train_accuracy": correct/samples,
            "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
            "epoch_seconds": time.monotonic()-epoch_started}
        history.append(row)
        better = best is None or row["validation_accuracy"] > best["accuracy"] or (
            row["validation_accuracy"] == best["accuracy"] and row["validation_loss"] < best["loss"])
        if better:
            best = {"epoch": epoch, "accuracy": row["validation_accuracy"], "loss": row["validation_loss"]}
            save_checkpoint(best_path, model, history, best, config, prior_seconds + time.monotonic()-started)
            print(f"new_validation_best seed={args.seed} epoch={epoch} accuracy={row['validation_accuracy']:.4f}")
        save_checkpoint(latest_path, model, history, best, config, prior_seconds + time.monotonic()-started)
        print(f"saved seed={args.seed} epoch={epoch} train_acc={row['train_accuracy']:.4f} validation_acc={row['validation_accuracy']:.4f}")

    if best is None or not best_path.is_file(): raise RuntimeError("best checkpoint is missing")
    result_path = args.output_dir / "result.json"
    if result_path.is_file() and (args.output_dir / "test_predictions.npz").is_file():
        print(f"test_already_evaluated={result_path}"); return
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"]); model.to(device)
    test = evaluate(model, eval_loader(test_data, None, args.batch_size), device, keep_predictions=True)
    targets, predictions = test.pop("targets"), test.pop("predictions")
    np.savez_compressed(args.output_dir / "test_predictions.npz", targets=targets, predictions=predictions)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    result = {"framework": "PyTorch", "reference": "full_precision_sgd", "seed": args.seed,
        "evaluated_best_epoch": best["epoch"], "best_validation_accuracy": best["accuracy"],
        "best_validation_loss": best["loss"], "test": test,
        "training_seconds": latest["training_seconds"], "config": config}
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"final_test_once seed={args.seed} best_epoch={best['epoch']} accuracy={test['accuracy']:.4f} loss={test['loss']:.6f}")


if __name__ == "__main__": main()
