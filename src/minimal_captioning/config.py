from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

import yaml

DipMode: TypeAlias = Literal["sobel", "gaussian", "none"]
DeviceName: TypeAlias = Literal["auto", "cpu", "cuda", "mps"]
ConfigValue: TypeAlias = str | int | float | bool | None | list[float]
ConfigSection: TypeAlias = dict[str, ConfigValue]


class ConfigurationError(ValueError):
    """Raised when an experiment configuration violates the project contract."""


@dataclass(frozen=True, kw_only=True)
class DataConfig:
    root: Path
    images_dir: str = "Images"
    captions_file: str = "captions.txt"
    min_word_frequency: int = 3
    max_caption_length: int = 30
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    num_workers: int = 2
    max_train_images: int | None = None
    max_validation_images: int | None = None
    max_test_images: int | None = None

    def validate(self) -> None:
        if self.min_word_frequency < 1:
            raise ConfigurationError("data.min_word_frequency must be at least 1")
        if self.max_caption_length < 4:
            raise ConfigurationError("data.max_caption_length must be at least 4")
        if self.num_workers < 0:
            raise ConfigurationError("data.num_workers cannot be negative")
        ratios = (self.train_ratio, self.validation_ratio, self.test_ratio)
        if any(ratio <= 0.0 for ratio in ratios):
            raise ConfigurationError("all data split ratios must be greater than zero")
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ConfigurationError("data split ratios must sum to 1.0")
        limits = (
            self.max_train_images,
            self.max_validation_images,
            self.max_test_images,
        )
        if any(limit is not None and limit < 1 for limit in limits):
            raise ConfigurationError("optional image limits must be positive")


@dataclass(frozen=True, kw_only=True)
class ModelConfig:
    dip_mode: DipMode = "sobel"
    resize_size: int = 256
    image_size: int = 224
    sobel_blend: float = 0.35
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.2
    embedding_dim: int = 128
    latent_dim: int = 6
    pretrained_resnet: bool = True

    def validate(self) -> None:
        if self.dip_mode not in {"sobel", "gaussian", "none"}:
            raise ConfigurationError("model.dip_mode must be sobel, gaussian, or none")
        if self.resize_size < self.image_size:
            raise ConfigurationError("model.resize_size must be >= model.image_size")
        if self.image_size < 32:
            raise ConfigurationError("model.image_size must be at least 32")
        if not 0.0 <= self.sobel_blend <= 1.0:
            raise ConfigurationError("model.sobel_blend must be between 0 and 1")
        if self.gaussian_kernel_size < 3 or self.gaussian_kernel_size % 2 == 0:
            raise ConfigurationError("model.gaussian_kernel_size must be an odd integer >= 3")
        if self.gaussian_sigma <= 0.0:
            raise ConfigurationError("model.gaussian_sigma must be positive")
        if self.embedding_dim < 1:
            raise ConfigurationError("model.embedding_dim must be positive")
        if self.latent_dim != 6:
            raise ConfigurationError(
                "model.latent_dim must remain exactly 6 to satisfy the professor's bottleneck"
            )
        if not self.pretrained_resnet:
            raise ConfigurationError(
                "model.pretrained_resnet must be true for the required fixed pre-trained encoder"
            )

    def to_dict(self) -> dict[str, str | int | float | bool]:
        return cast(dict[str, str | int | float | bool], asdict(self))


@dataclass(frozen=True, kw_only=True)
class TrainingConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    early_stopping_patience: int = 5
    device: DeviceName = "auto"

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ConfigurationError("training.epochs and training.batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ConfigurationError("training learning rate/weight decay are invalid")
        if self.gradient_clip <= 0.0:
            raise ConfigurationError("training.gradient_clip must be positive")
        if self.early_stopping_patience < 1:
            raise ConfigurationError("training.early_stopping_patience must be positive")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ConfigurationError("training.device must be auto, cpu, cuda, or mps")


@dataclass(frozen=True, kw_only=True)
class EvaluationConfig:
    batch_size: int = 64
    max_generation_length: int = 30
    tsne_max_images: int = 1000
    gaussian_noise_std: tuple[float, ...] = (0.05, 0.1, 0.2)
    salt_pepper_amount: tuple[float, ...] = (0.01, 0.03, 0.05)

    def validate(self) -> None:
        if self.batch_size < 1 or self.max_generation_length < 2:
            raise ConfigurationError("evaluation batch/sequence sizes are invalid")
        if self.tsne_max_images < 3:
            raise ConfigurationError("evaluation.tsne_max_images must be at least 3")
        if any(level <= 0.0 for level in self.gaussian_noise_std):
            raise ConfigurationError("Gaussian noise standard deviations must be positive")
        if any(not 0.0 < level < 1.0 for level in self.salt_pepper_amount):
            raise ConfigurationError("salt-and-pepper amounts must lie between 0 and 1")


@dataclass(frozen=True, kw_only=True)
class PathsConfig:
    outputs: Path
    checkpoints: Path
    artifacts: Path
    screenshots: Path

    def create(self) -> None:
        for path in (self.outputs, self.checkpoints, self.artifacts, self.screenshots):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, kw_only=True)
class ProjectConfig:
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    paths: PathsConfig
    project_root: Path

    def validate(self) -> None:
        if self.seed < 0:
            raise ConfigurationError("seed cannot be negative")
        self.data.validate()
        self.model.validate()
        self.training.validate()
        self.evaluation.validate()


def _section(raw: dict[str, object], name: str) -> dict[str, object]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ConfigurationError(f"missing or invalid '{name}' configuration section")
    return cast(dict[str, object], value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError("image limits must be integers or null")
    return value


def _integer(section: dict[str, object], key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{key} must be an integer")
    return value


def _number(section: dict[str, object], key: str, default: float) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{key} must be numeric")
    return float(value)


def _boolean(section: dict[str, object], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"{key} must be true or false")
    return value


def _string(section: dict[str, object], key: str, default: str) -> str:
    value = section.get(key, default)
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{key} must be a non-empty string")
    return value


def _float_tuple(value: object, *, field: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{field} must be a non-empty list")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ConfigurationError(f"{field} must contain only numbers")
        result.append(float(item))
    return tuple(result)


def _resolve(project_root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field} must be a non-empty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a YAML experiment configuration."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigurationError("configuration root must be a mapping")
    raw = cast(dict[str, object], loaded)
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()
    project_root = project_root.resolve()

    data_raw = _section(raw, "data")
    model_raw = _section(raw, "model")
    training_raw = _section(raw, "training")
    evaluation_raw = _section(raw, "evaluation")
    paths_raw = _section(raw, "paths")

    data = DataConfig(
        root=_resolve(project_root, data_raw.get("root"), field="data.root"),
        images_dir=_string(data_raw, "images_dir", "Images"),
        captions_file=_string(data_raw, "captions_file", "captions.txt"),
        min_word_frequency=_integer(data_raw, "min_word_frequency", 3),
        max_caption_length=_integer(data_raw, "max_caption_length", 30),
        train_ratio=_number(data_raw, "train_ratio", 0.8),
        validation_ratio=_number(data_raw, "validation_ratio", 0.1),
        test_ratio=_number(data_raw, "test_ratio", 0.1),
        num_workers=_integer(data_raw, "num_workers", 2),
        max_train_images=_optional_int(data_raw.get("max_train_images")),
        max_validation_images=_optional_int(data_raw.get("max_validation_images")),
        max_test_images=_optional_int(data_raw.get("max_test_images")),
    )
    model = ModelConfig(
        dip_mode=cast(DipMode, _string(model_raw, "dip_mode", "sobel")),
        resize_size=_integer(model_raw, "resize_size", 256),
        image_size=_integer(model_raw, "image_size", 224),
        sobel_blend=_number(model_raw, "sobel_blend", 0.35),
        gaussian_kernel_size=_integer(model_raw, "gaussian_kernel_size", 5),
        gaussian_sigma=_number(model_raw, "gaussian_sigma", 1.2),
        embedding_dim=_integer(model_raw, "embedding_dim", 128),
        latent_dim=_integer(model_raw, "latent_dim", 6),
        pretrained_resnet=_boolean(model_raw, "pretrained_resnet", True),
    )
    training = TrainingConfig(
        epochs=_integer(training_raw, "epochs", 30),
        batch_size=_integer(training_raw, "batch_size", 64),
        learning_rate=_number(training_raw, "learning_rate", 1e-3),
        weight_decay=_number(training_raw, "weight_decay", 1e-4),
        gradient_clip=_number(training_raw, "gradient_clip", 5.0),
        early_stopping_patience=_integer(training_raw, "early_stopping_patience", 5),
        device=cast(DeviceName, _string(training_raw, "device", "auto")),
    )
    evaluation = EvaluationConfig(
        batch_size=_integer(evaluation_raw, "batch_size", 64),
        max_generation_length=_integer(evaluation_raw, "max_generation_length", 30),
        tsne_max_images=_integer(evaluation_raw, "tsne_max_images", 1000),
        gaussian_noise_std=_float_tuple(
            evaluation_raw.get("gaussian_noise_std", [0.05, 0.1, 0.2]),
            field="evaluation.gaussian_noise_std",
        ),
        salt_pepper_amount=_float_tuple(
            evaluation_raw.get("salt_pepper_amount", [0.01, 0.03, 0.05]),
            field="evaluation.salt_pepper_amount",
        ),
    )
    paths = PathsConfig(
        outputs=_resolve(project_root, paths_raw.get("outputs"), field="paths.outputs"),
        checkpoints=_resolve(project_root, paths_raw.get("checkpoints"), field="paths.checkpoints"),
        artifacts=_resolve(project_root, paths_raw.get("artifacts"), field="paths.artifacts"),
        screenshots=_resolve(project_root, paths_raw.get("screenshots"), field="paths.screenshots"),
    )
    config = ProjectConfig(
        seed=_integer(raw, "seed", 42),
        data=data,
        model=model,
        training=training,
        evaluation=evaluation,
        paths=paths,
        project_root=project_root,
    )
    config.validate()
    return config
