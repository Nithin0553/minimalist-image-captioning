from __future__ import annotations

from minimal_captioning.cli import build_parser


def test_cli_parses_quick_training_command() -> None:
    args = build_parser().parse_args(["train", "--config", "configs/quick.yaml"])
    assert args.command == "train"
    assert args.config.name == "quick.yaml"
