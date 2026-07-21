from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from minimal_captioning.preprocessing import DIPPreprocessor, NoiseSpec


@pytest.mark.parametrize("mode", ["sobel", "gaussian", "none"])
def test_dip_modes_produce_model_tensor(project_config, rgb_image, mode: str) -> None:
    model_config = replace(project_config.model, dip_mode=mode)
    tensor = DIPPreprocessor(model_config)(rgb_image)
    assert tensor.shape == (3, 64, 64)
    assert tensor.dtype == torch.float32


def test_display_output_stays_in_unit_range(project_config, rgb_image) -> None:
    tensor = DIPPreprocessor(project_config.model).apply(rgb_image, output="display")
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


@pytest.mark.parametrize(
    "noise",
    [NoiseSpec(kind="gaussian", level=0.1), NoiseSpec(kind="salt_pepper", level=0.1)],
)
def test_noise_is_deterministic_for_a_fixed_seed(project_config, rgb_image, noise) -> None:
    preprocessor = DIPPreprocessor(project_config.model)
    first = preprocessor(rgb_image, noise=noise, seed=99)
    second = preprocessor(rgb_image, noise=noise, seed=99)
    different = preprocessor(rgb_image, noise=noise, seed=100)
    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_invalid_salt_and_pepper_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        NoiseSpec(kind="salt_pepper", level=1.0).validate()
