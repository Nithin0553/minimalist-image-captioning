from __future__ import annotations

from pathlib import Path

import pytest
import torch

from minimal_captioning.training import (
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
