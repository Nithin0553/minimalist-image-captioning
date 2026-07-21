from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from minimal_captioning.config import ConfigurationError, load_config


def test_default_config_preserves_six_dimensional_contract() -> None:
    config = load_config(Path("configs/default.yaml"))
    assert config.model.latent_dim == 6
    assert config.model.pretrained_resnet is True
    assert config.data.train_ratio + config.data.validation_ratio + config.data.test_ratio == 1.0


def test_quick_paths_are_resolved_from_project_root() -> None:
    config = load_config(Path("configs/quick.yaml"))
    assert config.paths.outputs == (Path.cwd() / "outputs" / "quick").resolve()


def test_non_six_dimensional_config_is_rejected(project_config) -> None:
    invalid = replace(project_config, model=replace(project_config.model, latent_dim=7))
    with pytest.raises(ConfigurationError, match="exactly 6"):
        invalid.validate()


def test_non_pretrained_experiment_config_is_rejected(project_config) -> None:
    invalid = replace(project_config, model=replace(project_config.model, pretrained_resnet=False))
    with pytest.raises(ConfigurationError, match="pre-trained encoder"):
        invalid.validate()


def test_missing_config_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")
