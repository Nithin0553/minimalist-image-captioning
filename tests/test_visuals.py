from __future__ import annotations

from minimal_captioning.data import prepare_data, write_dataset_summary
from minimal_captioning.visuals import (
    create_architecture_artifacts,
    create_dataset_structure_visual,
    create_evidence_montage,
)


def test_architecture_artifacts_are_generated(project_config) -> None:
    summary, diagram = create_architecture_artifacts(project_config)
    assert summary.is_file()
    assert diagram.is_file()
    assert "Linear(512, 6)" in summary.read_text(encoding="utf-8")


def test_dataset_structure_visual_is_generated(project_config) -> None:
    prepared = prepare_data(project_config)
    summary = write_dataset_summary(project_config, prepared)
    destination = create_dataset_structure_visual(project_config, summary)
    assert destination.is_file()


def test_evidence_montage_handles_not_yet_generated_panels(project_config) -> None:
    destination = create_evidence_montage(project_config)
    assert destination.is_file()
