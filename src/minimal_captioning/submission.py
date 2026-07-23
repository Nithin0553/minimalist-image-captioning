from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .config import ProjectConfig
from .model import MinimalCaptioningModel

REQUIRED_ROOT_FILES = (
    "ReadMe.txt",
    "README.md",
    "REFERENCES.md",
    "pyproject.toml",
    "configs/default.yaml",
)


@dataclass(frozen=True, kw_only=True)
class SubmissionAudit:
    checked_files: int
    generated_files_checked: bool
    issues: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.issues


def required_generated_paths(config: ProjectConfig) -> tuple[Path, ...]:
    return (
        config.paths.artifacts / "splits.json",
        config.paths.artifacts / "vocabulary.json",
        config.paths.outputs / "dataset_structure.txt",
        config.paths.outputs / "dataset_structure.png",
        config.paths.outputs / "architecture_summary.txt",
        config.paths.outputs / "architecture_diagram.png",
        config.paths.outputs / "training_history.csv",
        config.paths.outputs / "training_curve.png",
        config.paths.outputs / "caption_metrics.json",
        config.paths.outputs / "test_predictions.csv",
        config.paths.outputs / "latent_vectors.csv",
        config.paths.outputs / "tsne_latent_space.png",
        config.paths.outputs / "sensitivity_results.csv",
        config.paths.outputs / "sensitivity_results.json",
        config.paths.outputs / "sensitivity_analysis.png",
        config.paths.outputs / "caption_result.png",
        config.paths.outputs / "caption_result.json",
        config.paths.screenshots / "project_evidence.png",
        config.paths.checkpoints / "best.pt",
        config.paths.checkpoints / "last.pt",
    )


def _check_readme_name(project_root: Path, issues: list[str]) -> None:
    matching_names = sorted(
        path.name
        for path in project_root.iterdir()
        if path.is_file() and path.name.lower() == "readme.txt"
    )
    if matching_names != ["ReadMe.txt"]:
        found = ", ".join(matching_names) if matching_names else "none"
        issues.append(f"expected exact filename ReadMe.txt; found: {found}")


def _check_image(path: Path, issues: list[str]) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except OSError as error:
        issues.append(f"invalid image artifact {path}: {error}")


def audit_submission(
    config: ProjectConfig,
    *,
    require_generated: bool = False,
) -> SubmissionAudit:
    issues: list[str] = []
    checked_files = 0
    project_root = config.project_root

    _check_readme_name(project_root, issues)
    for relative_name in REQUIRED_ROOT_FILES:
        checked_files += 1
        path = project_root / relative_name
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"missing or empty required source file: {relative_name}")

    source_package = project_root / "src" / "minimal_captioning"
    source_files = tuple(source_package.glob("*.py")) if source_package.is_dir() else ()
    checked_files += len(source_files)
    if not source_files:
        issues.append("project source package is missing: src/minimal_captioning")

    if config.model.dip_mode == "none":
        issues.append("the final configuration must enable Sobel or Gaussian DIP preprocessing")
    if not config.evaluation.gaussian_noise_std:
        issues.append("the final configuration has no Gaussian-noise conditions")
    if not config.evaluation.salt_pepper_amount:
        issues.append("the final configuration has no salt-and-pepper conditions")

    model = MinimalCaptioningModel(
        vocab_size=4,
        embedding_dim=config.model.embedding_dim,
        pad_index=0,
        pretrained_resnet=False,
    )
    try:
        model.assert_required_architecture()
    except RuntimeError as error:
        issues.append(str(error))

    if require_generated:
        for path in required_generated_paths(config):
            checked_files += 1
            if not path.is_file() or path.stat().st_size == 0:
                issues.append(f"missing or empty generated artifact: {path}")
                continue
            if path.suffix.lower() == ".png":
                _check_image(path, issues)

    return SubmissionAudit(
        checked_files=checked_files,
        generated_files_checked=require_generated,
        issues=tuple(issues),
    )
