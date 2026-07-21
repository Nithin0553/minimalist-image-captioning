from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from minimal_captioning.config import (
    DataConfig,
    EvaluationConfig,
    ModelConfig,
    PathsConfig,
    ProjectConfig,
    TrainingConfig,
)


@pytest.fixture
def rgb_image() -> Image.Image:
    image = Image.new("RGB", (80, 60), color=(40, 120, 200))
    for coordinate in range(10, 50):
        image.putpixel((coordinate, coordinate), (255, 255, 255))
    return image


@pytest.fixture
def project_config(tmp_path: Path) -> ProjectConfig:
    data_root = tmp_path / "data" / "flickr8k"
    images_dir = data_root / "Images"
    images_dir.mkdir(parents=True)
    captions_path = data_root / "captions.txt"
    with captions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "caption"])
        for index in range(12):
            image_name = f"image_{index:02d}.jpg"
            Image.new(
                "RGB",
                (64, 64),
                color=((index * 20) % 255, (index * 40) % 255, (index * 60) % 255),
            ).save(images_dir / image_name)
            for caption_index in range(5):
                writer.writerow(
                    [image_name, f"a person and dog in field number {index} view {caption_index}"]
                )
    return ProjectConfig(
        seed=42,
        data=DataConfig(
            root=data_root,
            num_workers=0,
            min_word_frequency=1,
            max_caption_length=16,
        ),
        model=ModelConfig(
            resize_size=72,
            image_size=64,
            embedding_dim=16,
            pretrained_resnet=True,
        ),
        training=TrainingConfig(epochs=1, batch_size=2, device="cpu"),
        evaluation=EvaluationConfig(
            batch_size=2,
            max_generation_length=8,
            tsne_max_images=10,
            gaussian_noise_std=(0.1,),
            salt_pepper_amount=(0.05,),
        ),
        paths=PathsConfig(
            outputs=tmp_path / "outputs",
            checkpoints=tmp_path / "checkpoints",
            artifacts=tmp_path / "artifacts",
            screenshots=tmp_path / "screenshots",
        ),
        project_root=tmp_path,
    )
