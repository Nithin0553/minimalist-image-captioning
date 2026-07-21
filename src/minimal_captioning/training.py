from __future__ import annotations

import csv
import io
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ProjectConfig
from .data import PreparedData, TrainingBatch, create_training_loaders, prepare_data
from .io_utils import write_text_atomic
from .model import MinimalCaptioningModel
from .text import Vocabulary

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True, kw_only=True)
class EpochResult:
    epoch: int
    train_loss: float
    validation_loss: float


@dataclass(frozen=True, kw_only=True)
class TrainingResult:
    best_checkpoint: Path
    last_checkpoint: Path
    history: tuple[EpochResult, ...]
    best_validation_loss: float
    stopped_early: bool


@dataclass(frozen=True, kw_only=True)
class LoadedModel:
    model: MinimalCaptioningModel
    vocabulary: Vocabulary
    checkpoint_epoch: int


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device: {requested}")
    return torch.device(requested)


def _build_model(
    config: ProjectConfig, vocabulary: Vocabulary, *, pretrained: bool
) -> MinimalCaptioningModel:
    model = MinimalCaptioningModel(
        vocab_size=len(vocabulary),
        embedding_dim=config.model.embedding_dim,
        pad_index=vocabulary.pad_index,
        pretrained_resnet=pretrained,
    )
    model.assert_required_architecture()
    return model


def _run_epoch(
    model: MinimalCaptioningModel,
    loader: DataLoader[TrainingBatch],
    *,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    optimizer: Optimizer | None,
    gradient_clip: float,
    label: str,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0
    iterator = tqdm(loader, desc=label, leave=False)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in iterator:
            moved = batch.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits, _ = model(moved.images, moved.input_tokens)
            flattened_logits = logits.reshape(-1, logits.shape[-1])
            flattened_targets = moved.target_tokens.reshape(-1)
            loss_sum = criterion(flattened_logits, flattened_targets)
            valid_tokens = int(flattened_targets.ne(criterion.ignore_index).sum().item())
            if valid_tokens == 0:
                continue
            loss = loss_sum / valid_tokens
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    max_norm=gradient_clip,
                )
                optimizer.step()
            total_loss += float(loss_sum.detach().item())
            total_tokens += valid_tokens
            iterator.set_postfix(loss=f"{total_loss / total_tokens:.4f}")
    if total_tokens == 0:
        raise RuntimeError("epoch contained no non-padding target tokens")
    return total_loss / total_tokens


def _checkpoint_payload(
    *,
    config: ProjectConfig,
    model: MinimalCaptioningModel,
    optimizer: Optimizer,
    vocabulary: Vocabulary,
    epoch: int,
    best_validation_loss: float,
    history: list[EpochResult],
) -> dict[str, object]:
    return {
        "format_version": 1,
        "epoch": epoch,
        "best_validation_loss": best_validation_loss,
        "model_config": config.model.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "vocabulary": vocabulary.to_dict(),
        "history": [
            {
                "epoch": item.epoch,
                "train_loss": item.train_loss,
                "validation_loss": item.validation_loss,
            }
            for item in history
        ],
    }


def save_checkpoint_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def load_checkpoint_payload(path: Path, *, device: torch.device) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    raw = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(raw, dict) or raw.get("format_version") != 1:
        raise ValueError(f"unsupported checkpoint format: {path}")
    return cast(dict[str, object], raw)


def _vocabulary_from_payload(payload: dict[str, object]) -> Vocabulary:
    raw = payload.get("vocabulary")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint does not contain a valid vocabulary")
    return Vocabulary.from_dict(cast(dict[str, object], raw))


def _payload_int(payload: dict[str, object], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"checkpoint field '{key}' must be an integer")
    return value


def _payload_float(payload: dict[str, object], key: str, default: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"checkpoint field '{key}' must be numeric")
    return float(value)


def load_trained_model(
    config: ProjectConfig, checkpoint: Path, *, device: torch.device | None = None
) -> LoadedModel:
    selected_device = device or resolve_device(config.training.device)
    payload = load_checkpoint_payload(checkpoint, device=selected_device)
    vocabulary = _vocabulary_from_payload(payload)
    model = _build_model(config, vocabulary, pretrained=False)
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a model state")
    model.load_state_dict(state)
    model.to(selected_device)
    model.eval()
    return LoadedModel(
        model=model,
        vocabulary=vocabulary,
        checkpoint_epoch=_payload_int(payload, "epoch", 0),
    )


def _write_history(config: ProjectConfig, history: list[EpochResult]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["epoch", "train_loss", "validation_loss"])
    for item in history:
        writer.writerow([item.epoch, f"{item.train_loss:.8f}", f"{item.validation_loss:.8f}"])
    write_text_atomic(config.paths.outputs / "training_history.csv", stream.getvalue())

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        [item.epoch for item in history], [item.train_loss for item in history], label="Train"
    )
    axis.plot(
        [item.epoch for item in history],
        [item.validation_loss for item in history],
        label="Validation",
    )
    axis.set(title="Captioning Loss", xlabel="Epoch", ylabel="Cross-entropy loss per token")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(config.paths.outputs / "training_curve.png", dpi=180)
    plt.close(figure)


def train_model(config: ProjectConfig, *, resume: Path | None = None) -> TrainingResult:
    config.paths.create()
    set_reproducible_seed(config.seed)
    device = resolve_device(config.training.device)
    resume_payload: dict[str, object] | None = None
    vocabulary: Vocabulary | None = None
    if resume is not None:
        resume_payload = load_checkpoint_payload(resume, device=device)
        vocabulary = _vocabulary_from_payload(resume_payload)
    prepared: PreparedData = prepare_data(config, vocabulary=vocabulary)
    model = _build_model(
        config,
        prepared.vocabulary,
        pretrained=resume_payload is None and config.model.pretrained_resnet,
    ).to(device)
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    start_epoch = 1
    best_validation_loss = float("inf")
    history: list[EpochResult] = []
    if resume_payload is not None:
        model_state = resume_payload.get("model_state")
        optimizer_state = resume_payload.get("optimizer_state")
        if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
            raise ValueError("resume checkpoint lacks model or optimizer state")
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        start_epoch = _payload_int(resume_payload, "epoch", 0) + 1
        best_validation_loss = _payload_float(resume_payload, "best_validation_loss", float("inf"))

    train_loader, validation_loader = create_training_loaders(config, prepared)
    criterion = nn.CrossEntropyLoss(ignore_index=prepared.vocabulary.pad_index, reduction="sum")
    best_checkpoint = config.paths.checkpoints / "best.pt"
    last_checkpoint = config.paths.checkpoints / "last.pt"
    patience_used = 0
    stopped_early = False

    for epoch in range(start_epoch, config.training.epochs + 1):
        train_loss = _run_epoch(
            model,
            train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            gradient_clip=config.training.gradient_clip,
            label=f"Epoch {epoch} train",
        )
        validation_loss = _run_epoch(
            model,
            validation_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            gradient_clip=config.training.gradient_clip,
            label=f"Epoch {epoch} validation",
        )
        result = EpochResult(epoch=epoch, train_loss=train_loss, validation_loss=validation_loss)
        history.append(result)
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
            patience_used = 0
        else:
            patience_used += 1
        payload = _checkpoint_payload(
            config=config,
            model=model,
            optimizer=optimizer,
            vocabulary=prepared.vocabulary,
            epoch=epoch,
            best_validation_loss=best_validation_loss,
            history=history,
        )
        save_checkpoint_atomic(last_checkpoint, payload)
        if improved:
            save_checkpoint_atomic(best_checkpoint, payload)
        _write_history(config, history)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, validation_loss={validation_loss:.4f}")
        if patience_used >= config.training.early_stopping_patience:
            stopped_early = True
            print(f"Early stopping after {patience_used} epochs without improvement.")
            break

    if not history:
        raise RuntimeError(
            "No epochs ran. Increase training.epochs or resume from an earlier checkpoint."
        )
    return TrainingResult(
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
        history=tuple(history),
        best_validation_loss=best_validation_loss,
        stopped_early=stopped_early,
    )
