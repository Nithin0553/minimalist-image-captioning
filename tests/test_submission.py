from __future__ import annotations

from pathlib import Path

from PIL import Image

from minimal_captioning.submission import (
    REQUIRED_ROOT_FILES,
    audit_submission,
    required_generated_paths,
)


def _write_required_source_files(project_root: Path) -> None:
    for relative_name in REQUIRED_ROOT_FILES:
        path = project_root / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{relative_name}\n", encoding="utf-8")
    source = project_root / "src" / "minimal_captioning" / "__init__.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("__all__ = []\n", encoding="utf-8")


def test_source_submission_contract_checks_exact_filenames(project_config) -> None:
    _write_required_source_files(project_config.project_root)
    result = audit_submission(project_config)
    assert result.passed
    assert not result.generated_files_checked


def test_submission_contract_rejects_incorrect_readme_capitalization(project_config) -> None:
    _write_required_source_files(project_config.project_root)
    correct = project_config.project_root / "ReadMe.txt"
    correct.rename(project_config.project_root / "README.txt")
    result = audit_submission(project_config)
    assert not result.passed
    assert any("exact filename ReadMe.txt" in issue for issue in result.issues)


def test_generated_submission_contract_checks_every_artifact(project_config) -> None:
    _write_required_source_files(project_config.project_root)
    for path in required_generated_paths(project_config):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".png":
            Image.new("RGB", (16, 16), color="white").save(path)
        else:
            path.write_text("test artifact\n", encoding="utf-8")
    result = audit_submission(project_config, require_generated=True)
    assert result.passed
    assert result.generated_files_checked
