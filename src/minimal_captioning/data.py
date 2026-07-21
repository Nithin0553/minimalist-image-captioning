from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal, TypeAlias, cast

import torch
from PIL import Image
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .config import DataConfig, ProjectConfig
from .io_utils import write_json_atomic, write_text_atomic
from .preprocessing import DIPPreprocessor, NoiseSpec
from .text import Vocabulary

SplitName: TypeAlias = Literal["train", "validation", "test"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class DatasetLayoutError(ValueError):
    """Raised when Flickr8k files cannot be found or parsed safely."""


@dataclass(frozen=True, kw_only=True)
class DatasetLayout:
    root: Path
    images_dir: Path
    captions_file: Path


@dataclass(frozen=True, kw_only=True)
class CaptionRecord:
    image_name: str
    caption: str


@dataclass(frozen=True, kw_only=True)
class SplitManifest:
    seed: int
    train_ratio: float
    validation_ratio: float
    test_ratio: float
    image_fingerprint: str
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def names(self, split: SplitName) -> tuple[str, ...]:
        return cast(tuple[str, ...], getattr(self, split))

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "ratios": {
                "train": self.train_ratio,
                "validation": self.validation_ratio,
                "test": self.test_ratio,
            },
            "image_fingerprint": self.image_fingerprint,
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }


@dataclass(frozen=True, kw_only=True)
class PreparedData:
    layout: DatasetLayout
    records: tuple[CaptionRecord, ...]
    splits: SplitManifest
    vocabulary: Vocabulary

    def records_for(self, split: SplitName) -> tuple[CaptionRecord, ...]:
        allowed = set(self.splits.names(split))
        return tuple(record for record in self.records if record.image_name in allowed)


def _contains_images(path: Path) -> bool:
    return path.is_dir() and any(
        child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES for child in path.iterdir()
    )


def resolve_dataset_layout(config: DataConfig) -> DatasetLayout:
    root = config.root
    if not root.exists():
        raise DatasetLayoutError(
            f"Flickr8k root does not exist: {root}. Run the download-data command first."
        )
    direct_images = root / config.images_dir
    direct_captions = root / config.captions_file
    images_dir = direct_images if _contains_images(direct_images) else None
    captions_file = direct_captions if direct_captions.is_file() else None

    if images_dir is None:
        directory_candidates = sorted(
            (path for path in root.rglob("*") if _contains_images(path)),
            key=lambda path: (len(path.parts), str(path).lower()),
        )
        images_dir = directory_candidates[0] if directory_candidates else None
    if captions_file is None:
        preferred_names = (
            config.captions_file.lower(),
            "captions.txt",
            "flickr8k.token.txt",
            "captions.csv",
        )
        file_candidates = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.name.lower() in preferred_names
        ]
        file_candidates.sort(
            key=lambda path: (
                preferred_names.index(path.name.lower()),
                len(path.parts),
                str(path).lower(),
            )
        )
        captions_file = file_candidates[0] if file_candidates else None

    if images_dir is None or captions_file is None:
        raise DatasetLayoutError(
            f"Could not locate both Flickr8k images and captions under {root}. "
            f"Expected {config.images_dir}/ and {config.captions_file}."
        )
    return DatasetLayout(root=root, images_dir=images_dir, captions_file=captions_file)


def _parse_csv_captions(text: str) -> list[CaptionRecord] | None:
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = [name.strip().lower() for name in (reader.fieldnames or [])]
    image_fields = ("image", "image_name", "filename", "file_name")
    caption_fields = ("caption", "comment", "description")
    image_field = next((name for name in image_fields if name in fieldnames), None)
    caption_field = next((name for name in caption_fields if name in fieldnames), None)
    if image_field is None or caption_field is None:
        return None
    original_fields = {name.strip().lower(): name for name in reader.fieldnames or []}
    records: list[CaptionRecord] = []
    for row in reader:
        image_value = str(row.get(original_fields[image_field]) or "").strip()
        caption_value = str(row.get(original_fields[caption_field]) or "").strip()
        if image_value and caption_value:
            records.append(CaptionRecord(image_name=Path(image_value).name, caption=caption_value))
    return records


def _parse_token_captions(text: str) -> list[CaptionRecord] | None:
    records: list[CaptionRecord] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "\t" not in line:
            return None
        image_token, caption = line.split("\t", maxsplit=1)
        image_name = image_token.split("#", maxsplit=1)[0]
        if not image_name or not caption.strip():
            continue
        records.append(CaptionRecord(image_name=Path(image_name).name, caption=caption.strip()))
    return records or None


def load_caption_records(path: Path) -> tuple[CaptionRecord, ...]:
    text = path.read_text(encoding="utf-8-sig")
    records = _parse_csv_captions(text) or _parse_token_captions(text)
    if not records:
        raise DatasetLayoutError(
            f"Unsupported captions format in {path}. Expected image,caption CSV or token TSV."
        )
    unique_records = tuple(dict.fromkeys(records))
    if not unique_records:
        raise DatasetLayoutError(f"No usable captions found in {path}")
    return unique_records


def validate_caption_records(
    records: tuple[CaptionRecord, ...], *, images_dir: Path
) -> dict[str, object]:
    captions_per_image: Counter[str] = Counter(record.image_name for record in records)
    missing = sorted(name for name in captions_per_image if not (images_dir / name).is_file())
    if missing:
        preview = ", ".join(missing[:5])
        raise DatasetLayoutError(
            f"{len(missing)} captioned images are missing from {images_dir}; first: {preview}"
        )
    counts = list(captions_per_image.values())
    return {
        "images": len(captions_per_image),
        "captions": len(records),
        "minimum_captions_per_image": min(counts),
        "maximum_captions_per_image": max(counts),
        "images_with_five_captions": sum(count == 5 for count in counts),
    }


def _fingerprint(image_names: list[str]) -> str:
    normalized = "\n".join(sorted(image_names)).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def build_split_manifest(image_names: list[str], *, config: ProjectConfig) -> SplitManifest:
    unique_names = sorted(set(image_names))
    if len(unique_names) < 3:
        raise DatasetLayoutError("at least three unique images are required for train/val/test")
    random.Random(config.seed).shuffle(unique_names)
    count = len(unique_names)
    train_end = max(1, int(count * config.data.train_ratio))
    validation_count = max(1, int(count * config.data.validation_ratio))
    validation_end = min(count - 1, train_end + validation_count)
    train_names = unique_names[:train_end]
    validation_names = unique_names[train_end:validation_end]
    test_names = unique_names[validation_end:]
    if not validation_names or not test_names:
        raise DatasetLayoutError("split ratios produced an empty validation or test split")

    def limited(names: list[str], limit: int | None) -> tuple[str, ...]:
        return tuple(names if limit is None else names[:limit])

    return SplitManifest(
        seed=config.seed,
        train_ratio=config.data.train_ratio,
        validation_ratio=config.data.validation_ratio,
        test_ratio=config.data.test_ratio,
        image_fingerprint=_fingerprint(unique_names),
        train=limited(train_names, config.data.max_train_images),
        validation=limited(validation_names, config.data.max_validation_images),
        test=limited(test_names, config.data.max_test_images),
    )


def prepare_data(config: ProjectConfig, *, vocabulary: Vocabulary | None = None) -> PreparedData:
    config.paths.create()
    layout = resolve_dataset_layout(config.data)
    records = load_caption_records(layout.captions_file)
    validate_caption_records(records, images_dir=layout.images_dir)
    splits = build_split_manifest([record.image_name for record in records], config=config)
    write_json_atomic(config.paths.artifacts / "splits.json", splits.to_dict())
    if vocabulary is None:
        train_names = set(splits.train)
        train_captions = [record.caption for record in records if record.image_name in train_names]
        vocabulary = Vocabulary.build(train_captions, min_frequency=config.data.min_word_frequency)
    write_json_atomic(config.paths.artifacts / "vocabulary.json", vocabulary.to_dict())
    return PreparedData(layout=layout, records=records, splits=splits, vocabulary=vocabulary)


def write_dataset_summary(config: ProjectConfig, prepared: PreparedData) -> Path:
    validation = validate_caption_records(prepared.records, images_dir=prepared.layout.images_dir)
    lines = [
        "FLICKR8K DATASET STRUCTURE",
        "===========================",
        f"Root: {prepared.layout.root}",
        f"Images directory: {prepared.layout.images_dir}",
        f"Captions file: {prepared.layout.captions_file}",
        f"Unique images: {validation['images']}",
        f"Captions: {validation['captions']}",
        f"Caption range per image: {validation['minimum_captions_per_image']} to "
        f"{validation['maximum_captions_per_image']}",
        f"Images with exactly five captions: {validation['images_with_five_captions']}",
        f"Training images used: {len(prepared.splits.train)}",
        f"Validation images used: {len(prepared.splits.validation)}",
        f"Test images used: {len(prepared.splits.test)}",
        f"Vocabulary size (training captions only): {len(prepared.vocabulary)}",
        f"Split seed: {prepared.splits.seed}",
        f"Image fingerprint: {prepared.splits.image_fingerprint}",
    ]
    destination = config.paths.outputs / "dataset_structure.txt"
    write_text_atomic(destination, "\n".join(lines) + "\n")
    return destination


@dataclass(frozen=True, kw_only=True)
class TrainingSample:
    image: Tensor
    input_tokens: Tensor
    target_tokens: Tensor
    image_name: str


@dataclass(frozen=True, kw_only=True)
class TrainingBatch:
    images: Tensor
    input_tokens: Tensor
    target_tokens: Tensor
    image_names: tuple[str, ...]

    def to(self, device: torch.device) -> TrainingBatch:
        return TrainingBatch(
            images=self.images.to(device, non_blocking=True),
            input_tokens=self.input_tokens.to(device, non_blocking=True),
            target_tokens=self.target_tokens.to(device, non_blocking=True),
            image_names=self.image_names,
        )


class CaptionTrainingDataset(Dataset[TrainingSample]):
    def __init__(
        self,
        *,
        records: tuple[CaptionRecord, ...],
        images_dir: Path,
        vocabulary: Vocabulary,
        preprocessor: DIPPreprocessor,
        max_caption_length: int,
    ) -> None:
        self.records = records
        self.images_dir = images_dir
        self.vocabulary = vocabulary
        self.preprocessor = preprocessor
        self.max_caption_length = max_caption_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TrainingSample:
        record = self.records[index]
        with Image.open(self.images_dir / record.image_name) as image:
            image_tensor = self.preprocessor(image)
        token_ids = self.vocabulary.encode(record.caption, max_length=self.max_caption_length)
        tokens = torch.tensor(token_ids, dtype=torch.long)
        return TrainingSample(
            image=image_tensor,
            input_tokens=tokens[:-1],
            target_tokens=tokens[1:],
            image_name=record.image_name,
        )


def collate_training_samples(samples: list[TrainingSample], *, pad_index: int) -> TrainingBatch:
    if not samples:
        raise ValueError("cannot collate an empty training batch")
    return TrainingBatch(
        images=torch.stack([sample.image for sample in samples]),
        input_tokens=pad_sequence(
            [sample.input_tokens for sample in samples], batch_first=True, padding_value=pad_index
        ),
        target_tokens=pad_sequence(
            [sample.target_tokens for sample in samples],
            batch_first=True,
            padding_value=pad_index,
        ),
        image_names=tuple(sample.image_name for sample in samples),
    )


@dataclass(frozen=True, kw_only=True)
class EvaluationSample:
    image: Tensor
    image_name: str
    references: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class EvaluationBatch:
    images: Tensor
    image_names: tuple[str, ...]
    references: tuple[tuple[str, ...], ...]


def _noise_seed(seed: int, image_name: str, noise: NoiseSpec) -> int:
    digest = hashlib.sha256(f"{seed}:{image_name}:{noise.label}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


class CaptionEvaluationDataset(Dataset[EvaluationSample]):
    def __init__(
        self,
        *,
        records: tuple[CaptionRecord, ...],
        image_names: tuple[str, ...],
        images_dir: Path,
        preprocessor: DIPPreprocessor,
        noise: NoiseSpec,
        seed: int,
    ) -> None:
        grouped: defaultdict[str, list[str]] = defaultdict(list)
        for record in records:
            grouped[record.image_name].append(record.caption)
        self.image_names = image_names
        self.references = {name: tuple(grouped[name]) for name in image_names}
        self.images_dir = images_dir
        self.preprocessor = preprocessor
        self.noise = noise
        self.seed = seed

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> EvaluationSample:
        image_name = self.image_names[index]
        with Image.open(self.images_dir / image_name) as image:
            tensor = self.preprocessor(
                image,
                noise=self.noise,
                seed=_noise_seed(self.seed, image_name, self.noise),
            )
        return EvaluationSample(
            image=tensor,
            image_name=image_name,
            references=self.references[image_name],
        )


def collate_evaluation_samples(samples: list[EvaluationSample]) -> EvaluationBatch:
    if not samples:
        raise ValueError("cannot collate an empty evaluation batch")
    return EvaluationBatch(
        images=torch.stack([sample.image for sample in samples]),
        image_names=tuple(sample.image_name for sample in samples),
        references=tuple(sample.references for sample in samples),
    )


def create_training_loaders(
    config: ProjectConfig, prepared: PreparedData
) -> tuple[DataLoader[TrainingBatch], DataLoader[TrainingBatch]]:
    preprocessor = DIPPreprocessor(config.model)
    train_dataset = CaptionTrainingDataset(
        records=prepared.records_for("train"),
        images_dir=prepared.layout.images_dir,
        vocabulary=prepared.vocabulary,
        preprocessor=preprocessor,
        max_caption_length=config.data.max_caption_length,
    )
    validation_dataset = CaptionTrainingDataset(
        records=prepared.records_for("validation"),
        images_dir=prepared.layout.images_dir,
        vocabulary=prepared.vocabulary,
        preprocessor=preprocessor,
        max_caption_length=config.data.max_caption_length,
    )
    collate = partial(collate_training_samples, pad_index=prepared.vocabulary.pad_index)
    generator = torch.Generator().manual_seed(config.seed)
    use_pinned_memory = config.training.device == "cuda" or (
        config.training.device == "auto" and torch.cuda.is_available()
    )
    common = {
        "batch_size": config.training.batch_size,
        "num_workers": config.data.num_workers,
        "collate_fn": collate,
        "pin_memory": use_pinned_memory,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    return train_loader, validation_loader


def create_evaluation_loader(
    config: ProjectConfig,
    prepared: PreparedData,
    *,
    noise: NoiseSpec | None = None,
) -> DataLoader[EvaluationBatch]:
    dataset = CaptionEvaluationDataset(
        records=prepared.records,
        image_names=prepared.splits.test,
        images_dir=prepared.layout.images_dir,
        preprocessor=DIPPreprocessor(config.model),
        noise=noise or NoiseSpec(),
        seed=config.seed,
    )
    use_pinned_memory = config.training.device == "cuda" or (
        config.training.device == "auto" and torch.cuda.is_available()
    )
    return DataLoader(
        dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=use_pinned_memory,
        collate_fn=collate_evaluation_samples,
    )


def load_vocabulary(path: Path) -> Vocabulary:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"invalid vocabulary file: {path}")
    return Vocabulary.from_dict(cast(dict[str, object], raw))
