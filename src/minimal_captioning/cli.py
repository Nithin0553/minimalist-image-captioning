from __future__ import annotations

import argparse
from pathlib import Path

from ._version import __version__
from .config import ProjectConfig, load_config
from .data import prepare_data, write_dataset_summary
from .download import download_flickr8k
from .evaluation import evaluate_checkpoint
from .metrics import ensure_meteor_resources
from .submission import audit_submission
from .training import train_model
from .visuals import (
    caption_image,
    create_architecture_artifacts,
    create_dataset_structure_visual,
    create_evidence_montage,
)

DEFAULT_CONFIG = Path("configs/default.yaml")


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML experiment configuration (default: configs/default.yaml)",
    )


def _resolve_from_project(config: ProjectConfig, path: Path) -> Path:
    expanded = path.expanduser()
    return (
        expanded.resolve() if expanded.is_absolute() else (config.project_root / expanded).resolve()
    )


def _checkpoint_path(config: ProjectConfig, supplied: Path | None) -> Path:
    return (
        _resolve_from_project(config, supplied)
        if supplied is not None
        else config.paths.checkpoints / "best.pt"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minimal-caption",
        description="Six-dimensional bottleneck image captioning on Flickr8k",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download-data", help="Download and arrange Flickr8k from Kaggle"
    )
    _add_config_argument(download_parser)

    subparsers.add_parser("setup-nltk", help="Download WordNet resources for standard METEOR")

    validate_parser = subparsers.add_parser(
        "validate-data", help="Validate Flickr8k and create deterministic splits"
    )
    _add_config_argument(validate_parser)

    architecture_parser = subparsers.add_parser(
        "architecture", help="Verify and render the required model architecture"
    )
    _add_config_argument(architecture_parser)

    train_parser = subparsers.add_parser("train", help="Train the captioning model")
    _add_config_argument(train_parser)
    train_parser.add_argument("--resume", type=Path, help="Resume from a trusted local checkpoint")

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Compute clean BLEU/METEOR metrics and t-SNE"
    )
    _add_config_argument(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", type=Path)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Run metrics, t-SNE, Gaussian noise, and salt-and-pepper analysis"
    )
    _add_config_argument(analyze_parser)
    analyze_parser.add_argument("--checkpoint", type=Path)

    caption_parser = subparsers.add_parser("caption", help="Caption one image and save a visual")
    _add_config_argument(caption_parser)
    caption_parser.add_argument("--checkpoint", type=Path)
    caption_parser.add_argument("--image", type=Path, required=True)
    caption_parser.add_argument("--output", type=Path)

    evidence_parser = subparsers.add_parser(
        "evidence", help="Combine generated figures into a screenshot-ready montage"
    )
    _add_config_argument(evidence_parser)

    submission_parser = subparsers.add_parser(
        "submission-check",
        help="Verify exact source filenames and, optionally, every generated final artifact",
    )
    _add_config_argument(submission_parser)
    submission_parser.add_argument(
        "--require-generated",
        action="store_true",
        help="Also require all checkpoints, metrics, plots, and screenshot artifacts",
    )

    run_all_parser = subparsers.add_parser(
        "run-all", help="Validate, train, evaluate, analyze, and create evidence"
    )
    _add_config_argument(run_all_parser)
    run_all_parser.add_argument(
        "--resume",
        type=Path,
        help="Resume training and then complete every remaining final stage",
    )
    return parser


def _print_metrics(config: ProjectConfig, checkpoint: Path, *, sensitivity: bool) -> None:
    result = evaluate_checkpoint(config, checkpoint, include_sensitivity=sensitivity)
    metrics = result.clean_metrics
    print(f"BLEU-1: {metrics.bleu_1:.6f}")
    print(f"BLEU-4: {metrics.bleu_4:.6f}")
    print(f"METEOR: {metrics.meteor:.6f}")
    print(f"Evaluated images: {metrics.evaluated_images}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Predictions: {result.predictions_path}")
    print(f"t-SNE: {result.tsne_path}")
    if sensitivity:
        print(f"Sensitivity plot: {result.sensitivity_path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "setup-nltk":
        ensure_meteor_resources(download=True)
        print("NLTK WordNet and omw-1.4 are ready.")
        return

    config = load_config(args.config)
    if args.command == "download-data":
        layout = download_flickr8k(config.data.root)
        print(f"Images ready at: {layout.images_dir}")
        print(f"Captions ready at: {layout.captions_file}")
        return
    if args.command == "validate-data":
        prepared = prepare_data(config)
        summary = write_dataset_summary(config, prepared)
        visual = create_dataset_structure_visual(config, summary)
        print(summary.read_text(encoding="utf-8"))
        print(f"Saved dataset summary: {summary}")
        print(f"Saved dataset structure image: {visual}")
        return
    if args.command == "architecture":
        summary, diagram = create_architecture_artifacts(config)
        print(summary.read_text(encoding="utf-8"))
        print(f"Saved architecture diagram: {diagram}")
        return
    if args.command == "train":
        resume = _resolve_from_project(config, args.resume) if args.resume else None
        result = train_model(config, resume=resume)
        print(f"Best checkpoint: {result.best_checkpoint}")
        print(f"Best validation loss: {result.best_validation_loss:.6f}")
        return
    if args.command == "evaluate":
        _print_metrics(config, _checkpoint_path(config, args.checkpoint), sensitivity=False)
        return
    if args.command == "analyze":
        _print_metrics(config, _checkpoint_path(config, args.checkpoint), sensitivity=True)
        return
    if args.command == "caption":
        image = _resolve_from_project(config, args.image)
        output = _resolve_from_project(config, args.output) if args.output else None
        caption, destination = caption_image(
            config,
            _checkpoint_path(config, args.checkpoint),
            image,
            destination=output,
        )
        print(f"Caption: {caption}")
        print(f"Saved visual: {destination}")
        return
    if args.command == "evidence":
        destination = create_evidence_montage(config)
        print(f"Saved evidence montage: {destination}")
        return
    if args.command == "submission-check":
        audit = audit_submission(config, require_generated=args.require_generated)
        if not audit.passed:
            print("Submission check: FAILED")
            for issue in audit.issues:
                print(f"- {issue}")
            raise SystemExit(1)
        scope = "source and generated artifacts" if audit.generated_files_checked else "source"
        print(f"Submission check: PASS ({scope}; {audit.checked_files} files checked)")
        print("Presentation slides and the 8-10 page report remain deferred.")
        return
    if args.command == "run-all":
        ensure_meteor_resources()
        prepared = prepare_data(config)
        summary = write_dataset_summary(config, prepared)
        create_dataset_structure_visual(config, summary)
        create_architecture_artifacts(config)
        resume = _resolve_from_project(config, args.resume) if args.resume else None
        training = train_model(config, resume=resume)
        _print_metrics(config, training.best_checkpoint, sensitivity=True)
        prepared = prepare_data(config)
        sample_image = prepared.layout.images_dir / prepared.splits.test[0]
        caption_image(config, training.best_checkpoint, sample_image)
        destination = create_evidence_montage(config)
        audit = audit_submission(config, require_generated=True)
        if not audit.passed:
            details = "\n".join(f"- {issue}" for issue in audit.issues)
            raise RuntimeError(f"final submission artifact check failed:\n{details}")
        print(f"Final artifact check: PASS ({audit.checked_files} files checked)")
        print(f"Complete. Submission evidence: {destination}")
        return
    parser.error(f"unsupported command: {args.command}")
