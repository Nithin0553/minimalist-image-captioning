from __future__ import annotations

from pathlib import Path

import pytest
import torch

from minimal_captioning.training import (
    EpochResult,
    _history_from_payload,
    _patience_from_history,
    load_checkpoint_payload,
    resolve_device,
    save_checkpoint_atomic,
    set_reproducible_seed,
)


def test_checkpoint_atomic_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "checkpoint.pt"
    save_checkpoint_atomic(path, {"format_version": 1, "epoch": 3})
    loaded = load_checkpoint_payload(path, device=torch.device("cpu"))
    assert loaded["epoch"] == 3
    assert not list(path.parent.glob("*.tmp"))


def test_seed_is_reproducible() -> None:
    set_reproducible_seed(7)
    first = torch.rand(3)
    set_reproducible_seed(7)
    assert torch.equal(first, torch.rand(3))


def test_cpu_device_is_available() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="unsupported"):
        resolve_device("quantum")


def test_resume_history_and_early_stopping_state_are_restored() -> None:
    payload: dict[str, object] = {
        "epoch": 4,
        "history": [
            {"epoch": 1, "train_loss": 3.0, "validation_loss": 2.5},
            {"epoch": 2, "train_loss": 2.4, "validation_loss": 2.0},
            {"epoch": 3, "train_loss": 2.1, "validation_loss": 2.1},
            {"epoch": 4, "train_loss": 2.0, "validation_loss": 2.2},
        ],
    }
    history = _history_from_payload(payload)
    assert history[-1] == EpochResult(epoch=4, train_loss=2.0, validation_loss=2.2)
    assert _patience_from_history(history) == 2


def test_resume_history_rejects_a_checkpoint_epoch_mismatch() -> None:
    payload: dict[str, object] = {
        "epoch": 3,
        "history": [{"epoch": 2, "train_loss": 2.0, "validation_loss": 1.5}],
    }
    with pytest.raises(ValueError, match="saved epoch"):
        _history_from_payload(payload)
