from __future__ import annotations

from pathlib import Path

import pytest
import torch

from minimal_captioning.data import (
    CaptionRecord,
    CaptionTrainingDataset,
    DatasetLayoutError,
    build_split_manifest,
    collate_training_samples,
    load_caption_records,
    prepare_data,
    resolve_dataset_layout,
    validate_caption_records,
    write_dataset_summary,
)
from minimal_captioning.preprocessing import DIPPreprocessor


def test_prepare_data_creates_disjoint_reproducible_image_splits(project_config) -> None:
    first = prepare_data(project_config)
    second = prepare_data(project_config)
    assert first.splits == second.splits
    assert not (set(first.splits.train) & set(first.splits.validation))
    assert not (set(first.splits.train) & set(first.splits.test))
    assert len(first.vocabulary) > 4
    assert (project_config.paths.artifacts / "splits.json").is_file()


def test_dataset_summary_contains_required_counts(project_config) -> None:
    prepared = prepare_data(project_config)
    path = write_dataset_summary(project_config, prepared)
    text = path.read_text(encoding="utf-8")
    assert "Unique images: 12" in text
    assert "Captions: 60" in text
    assert "Images with exactly five captions: 12" in text


def test_token_file_format_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "Flickr8k.token.txt"
    path.write_text("a.jpg#0\tA dog runs.\na.jpg#1\tA pet moves.\n", encoding="utf-8")
    records = load_caption_records(path)
    assert records == (
        CaptionRecord(image_name="a.jpg", caption="A dog runs."),
        CaptionRecord(image_name="a.jpg", caption="A pet moves."),
    )


def test_nested_layout_is_discovered(project_config) -> None:
    layout = resolve_dataset_layout(project_config.data)
    assert layout.images_dir.name == "Images"
    assert layout.captions_file.name == "captions.txt"


def test_missing_captioned_image_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetLayoutError, match="missing"):
        validate_caption_records(
            (CaptionRecord(image_name="absent.jpg", caption="missing"),), images_dir=tmp_path
        )


def test_training_dataset_and_collate(project_config) -> None:
    prepared = prepare_data(project_config)
    dataset = CaptionTrainingDataset(
        records=prepared.records_for("train")[:2],
        images_dir=prepared.layout.images_dir,
        vocabulary=prepared.vocabulary,
        preprocessor=DIPPreprocessor(project_config.model),
        max_caption_length=project_config.data.max_caption_length,
    )
    batch = collate_training_samples(
        [dataset[0], dataset[1]], pad_index=prepared.vocabulary.pad_index
    )
    assert batch.images.shape == (2, 3, 64, 64)
    assert batch.input_tokens.shape == batch.target_tokens.shape
    assert batch.to(torch.device("cpu")).image_names == batch.image_names


def test_split_builder_rejects_too_few_images(project_config) -> None:
    with pytest.raises(DatasetLayoutError, match="three"):
        build_split_manifest(["one.jpg", "two.jpg"], config=project_config)
