from __future__ import annotations

from minimal_captioning.visuals import create_architecture_artifacts, create_evidence_montage


def test_architecture_artifacts_are_generated(project_config) -> None:
    summary, diagram = create_architecture_artifacts(project_config)
    assert summary.is_file()
    assert diagram.is_file()
    assert "Linear(512, 6)" in summary.read_text(encoding="utf-8")


def test_evidence_montage_handles_not_yet_generated_panels(project_config) -> None:
    destination = create_evidence_montage(project_config)
    assert destination.is_file()
