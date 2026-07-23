from __future__ import annotations

from pathlib import Path

import minimal_captioning.cli as cli
from minimal_captioning.data import DatasetLayout, prepare_data
from minimal_captioning.evaluation import AnalysisResult
from minimal_captioning.metrics import CaptionMetrics
from minimal_captioning.submission import SubmissionAudit
from minimal_captioning.training import TrainingResult


def _analysis_result(root: Path) -> AnalysisResult:
    return AnalysisResult(
        clean_metrics=CaptionMetrics(
            bleu_1=0.5,
            bleu_4=0.25,
            meteor=0.4,
            evaluated_images=3,
        ),
        metrics_path=root / "metrics.json",
        predictions_path=root / "predictions.csv",
        tsne_path=root / "tsne.png",
        sensitivity_path=root / "sensitivity.png",
    )


def test_cli_parses_quick_training_command() -> None:
    args = cli.build_parser().parse_args(["train", "--config", "configs/quick.yaml"])
    assert args.command == "train"
    assert args.config.name == "quick.yaml"


def test_cli_parses_resumable_run_all_command() -> None:
    args = cli.build_parser().parse_args(
        [
            "run-all",
            "--config",
            "configs/default.yaml",
            "--resume",
            "checkpoints/last.pt",
        ]
    )
    assert args.command == "run-all"
    assert args.resume == Path("checkpoints/last.pt")


def test_print_metrics_reports_clean_and_sensitivity_paths(
    project_config, monkeypatch, capsys
) -> None:
    result = _analysis_result(project_config.project_root)
    monkeypatch.setattr(cli, "evaluate_checkpoint", lambda *_args, **_kwargs: result)

    cli._print_metrics(project_config, Path("model.pt"), sensitivity=True)

    output = capsys.readouterr().out
    assert "BLEU-1: 0.500000" in output
    assert "Evaluated images: 3" in output
    assert f"Sensitivity plot: {result.sensitivity_path}" in output


def test_cli_dispatches_every_project_command(project_config, monkeypatch, capsys) -> None:
    prepared = prepare_data(project_config)
    root = project_config.project_root
    summary = root / "summary.txt"
    summary.write_text("dataset summary", encoding="utf-8")
    architecture_summary = root / "architecture.txt"
    architecture_summary.write_text("architecture summary", encoding="utf-8")
    diagram = root / "architecture.png"
    checkpoint = root / "checkpoints" / "best.pt"
    destination = root / "result.png"
    layout = DatasetLayout(
        root=project_config.data.root,
        images_dir=prepared.layout.images_dir,
        captions_file=prepared.layout.captions_file,
    )
    training = TrainingResult(
        best_checkpoint=checkpoint,
        last_checkpoint=root / "checkpoints" / "last.pt",
        history=(),
        best_validation_loss=0.75,
        stopped_early=False,
    )
    meteor_downloads: list[bool] = []
    metric_modes: list[bool] = []

    monkeypatch.setattr(cli, "load_config", lambda _path: project_config)
    monkeypatch.setattr(
        cli,
        "ensure_meteor_resources",
        lambda *, download=False: meteor_downloads.append(download),
    )
    monkeypatch.setattr(cli, "download_flickr8k", lambda _target: layout)
    monkeypatch.setattr(cli, "prepare_data", lambda _config: prepared)
    monkeypatch.setattr(cli, "write_dataset_summary", lambda _config, _prepared: summary)
    monkeypatch.setattr(
        cli,
        "create_architecture_artifacts",
        lambda _config: (architecture_summary, diagram),
    )
    monkeypatch.setattr(
        cli,
        "create_dataset_structure_visual",
        lambda _config, _summary: root / "dataset_structure.png",
    )
    monkeypatch.setattr(cli, "train_model", lambda _config, resume=None: training)
    monkeypatch.setattr(
        cli,
        "_print_metrics",
        lambda _config, _checkpoint, *, sensitivity: metric_modes.append(sensitivity),
    )
    monkeypatch.setattr(
        cli,
        "caption_image",
        lambda *_args, **_kwargs: ("a dog runs", destination),
    )
    monkeypatch.setattr(cli, "create_evidence_montage", lambda _config: destination)
    monkeypatch.setattr(
        cli,
        "audit_submission",
        lambda _config, *, require_generated=False: SubmissionAudit(
            checked_files=20,
            generated_files_checked=require_generated,
            issues=(),
        ),
    )

    cli.main(["setup-nltk"])
    cli.main(["download-data", "--config", "ignored.yaml"])
    cli.main(["validate-data", "--config", "ignored.yaml"])
    cli.main(["architecture", "--config", "ignored.yaml"])
    cli.main(["train", "--config", "ignored.yaml", "--resume", "resume.pt"])
    cli.main(["evaluate", "--config", "ignored.yaml"])
    cli.main(["analyze", "--config", "ignored.yaml", "--checkpoint", "checkpoints/custom.pt"])
    cli.main(
        [
            "caption",
            "--config",
            "ignored.yaml",
            "--image",
            "image.jpg",
            "--output",
            "caption.png",
        ]
    )
    cli.main(["evidence", "--config", "ignored.yaml"])
    cli.main(["submission-check", "--config", "ignored.yaml", "--require-generated"])
    cli.main(["run-all", "--config", "ignored.yaml", "--resume", "checkpoints/last.pt"])

    output = capsys.readouterr().out
    assert "NLTK WordNet and omw-1.4 are ready." in output
    assert "Images ready at:" in output
    assert "Best validation loss: 0.750000" in output
    assert "Caption: a dog runs" in output
    assert "Submission check: PASS" in output
    assert "Final artifact check: PASS" in output
    assert "Complete. Submission evidence:" in output
    assert meteor_downloads == [True, False]
    assert metric_modes == [False, True, True]
