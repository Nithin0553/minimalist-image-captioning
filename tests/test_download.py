from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

import minimal_captioning.download as download_module
from minimal_captioning.download import download_flickr8k


def test_existing_flickr8k_layout_is_reused(project_config) -> None:
    layout = download_flickr8k(project_config.data.root)
    assert layout.images_dir == project_config.data.root / "Images"
    assert layout.captions_file == project_config.data.root / "captions.txt"


def test_download_copies_kaggle_layout(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "kaggle-cache"
    images = source / "Images"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8), color="blue").save(images / "sample.jpg")
    (source / "captions.txt").write_text(
        "image,caption\nsample.jpg,a sample image\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        download_module.kagglehub,
        "dataset_download",
        lambda handle: str(source),
    )

    target = tmp_path / "downloaded"
    layout = download_flickr8k(target)

    assert layout.images_dir.joinpath("sample.jpg").is_file()
    assert layout.captions_file.read_text(encoding="utf-8").startswith("image,caption")


def test_download_rejects_unrecognized_nonempty_directory(tmp_path: Path) -> None:
    target = tmp_path / "occupied"
    target.mkdir()
    (target / "personal-file.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to write"):
        download_flickr8k(target)
