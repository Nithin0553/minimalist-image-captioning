from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub
from tqdm import tqdm

from .config import DataConfig
from .data import DatasetLayout, resolve_dataset_layout

FLICKR8K_KAGGLE_HANDLE = "adityajn105/flickr8k"


def _copy_file_resumable(source: Path, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def download_flickr8k(target: Path) -> DatasetLayout:
    """Download Flickr8k through KaggleHub and copy it into the project data directory."""

    target = target.expanduser().resolve()
    if target.exists():
        try:
            return resolve_dataset_layout(DataConfig(root=target))
        except ValueError as error:
            unknown = [
                path for path in target.iterdir() if path.name not in {"Images", "captions.txt"}
            ]
            if unknown:
                names = ", ".join(path.name for path in unknown[:5])
                raise FileExistsError(
                    f"Refusing to write into non-empty unrecognized data directory {target}: {names}"
                ) from error
    target.mkdir(parents=True, exist_ok=True)
    cache_path = Path(kagglehub.dataset_download(FLICKR8K_KAGGLE_HANDLE)).resolve()
    source = resolve_dataset_layout(DataConfig(root=cache_path))
    target_images = target / "Images"
    target_images.mkdir(parents=True, exist_ok=True)
    image_files = sorted(
        path
        for path in source.images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for source_image in tqdm(image_files, desc="Copy Flickr8k images"):
        _copy_file_resumable(source_image, target_images / source_image.name)
    _copy_file_resumable(source.captions_file, target / "captions.txt")
    return resolve_dataset_layout(DataConfig(root=target))
