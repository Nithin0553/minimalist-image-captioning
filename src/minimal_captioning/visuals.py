from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch
from PIL import Image

from .config import ProjectConfig
from .io_utils import write_json_atomic, write_text_atomic
from .model import MinimalCaptioningModel
from .preprocessing import DIPPreprocessor
from .training import load_trained_model, resolve_device

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def create_dataset_structure_visual(config: ProjectConfig, summary_path: Path) -> Path:
    if not summary_path.is_file():
        raise FileNotFoundError(f"dataset summary not found: {summary_path}")
    config.paths.create()
    summary = summary_path.read_text(encoding="utf-8").strip()
    line_count = max(1, len(summary.splitlines()))
    figure_height = max(6.0, line_count * 0.42)
    figure, axis = plt.subplots(figsize=(12, figure_height))
    axis.axis("off")
    axis.set_title("Flickr8k Data Structure and Split Summary", fontsize=16, weight="bold", pad=18)
    axis.text(
        0.03,
        0.97,
        summary,
        ha="left",
        va="top",
        family="monospace",
        fontsize=11,
        linespacing=1.45,
        transform=axis.transAxes,
        bbox={
            "boxstyle": "round,pad=0.8",
            "facecolor": "#F7F9FC",
            "edgecolor": "#244A73",
            "linewidth": 1.2,
        },
    )
    figure.tight_layout()
    destination = config.paths.outputs / "dataset_structure.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def create_architecture_artifacts(config: ProjectConfig) -> tuple[Path, Path]:
    config.paths.create()
    model = MinimalCaptioningModel(
        vocab_size=4,
        embedding_dim=config.model.embedding_dim,
        pad_index=0,
        pretrained_resnet=False,
    )
    model.assert_required_architecture()
    info = model.architecture_info()
    lines = [
        "REQUIRED MINIMAL IMAGE CAPTIONING ARCHITECTURE",
        "==============================================",
        f"DIP preprocessing: {config.model.dip_mode}",
        "Encoder: fixed pre-trained ResNet-18",
        f"Encoder output: {info.encoder_output_dim}-D",
        "Bottleneck: one nn.Linear(512, 6); no intermediate layer or activation",
        f"Bottleneck parameters: 512 x 6 + 6 = {info.bottleneck_parameters}",
        f"Latent representation: {info.latent_dim}-D",
        f"GRU initial state: [1, batch, {info.gru_hidden_dim}]",
        f"GRU layers: {info.gru_layers}",
        f"Trainable ResNet parameters: {info.encoder_trainable_parameters}",
        "Decoder output: generated caption token sequence",
    ]
    text_path = config.paths.outputs / "architecture_summary.txt"
    write_text_atomic(text_path, "\n".join(lines) + "\n")

    stages = [
        ("Raw RGB image", "Flickr8k"),
        ("DIP preprocessing", "Sobel / Gaussian"),
        ("Frozen ResNet-18", "512-D feature"),
        ("Single linear layer", "512 -> 6"),
        ("One-layer GRU", "h0 = [1, B, 6]"),
        ("Caption", "token sequence"),
    ]
    figure, axis = plt.subplots(figsize=(15, 4.4))
    axis.set_xlim(0, len(stages) * 2.2)
    axis.set_ylim(0, 3)
    axis.axis("off")
    for index, (title, subtitle) in enumerate(stages):
        x = 0.25 + index * 2.2
        box = FancyBboxPatch(
            (x, 1.05),
            1.75,
            1.0,
            boxstyle="round,pad=0.08",
            facecolor="#E8F0FE" if index not in {3, 4} else "#FFF1CC",
            edgecolor="#244A73",
            linewidth=1.5,
        )
        axis.add_patch(box)
        axis.text(x + 0.875, 1.68, title, ha="center", va="center", fontsize=10, weight="bold")
        axis.text(x + 0.875, 1.35, subtitle, ha="center", va="center", fontsize=9)
        if index < len(stages) - 1:
            axis.annotate(
                "",
                xy=(x + 2.15, 1.55),
                xytext=(x + 1.78, 1.55),
                arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#244A73"},
            )
    axis.set_title("Course Project Flat Image Captioning Pipeline", fontsize=15, weight="bold")
    figure.tight_layout()
    image_path = config.paths.outputs / "architecture_diagram.png"
    figure.savefig(image_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return text_path, image_path


def caption_image(
    config: ProjectConfig,
    checkpoint: Path,
    image_path: Path,
    *,
    destination: Path | None = None,
) -> tuple[str, Path]:
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")
    config.paths.create()
    device = resolve_device(config.training.device)
    loaded = load_trained_model(config, checkpoint, device=device)
    preprocessor = DIPPreprocessor(config.model)
    with Image.open(image_path) as opened:
        original = opened.convert("RGB")
        model_tensor = preprocessor(original)
        display_tensor = preprocessor.apply(original, output="display")
    with torch.no_grad():
        tokens, latent = loaded.model.generate(
            model_tensor.unsqueeze(0).to(device),
            bos_index=loaded.vocabulary.bos_index,
            eos_index=loaded.vocabulary.eos_index,
            max_length=config.evaluation.max_generation_length,
        )
    caption = loaded.vocabulary.decode(tokens[0].detach().cpu().tolist())
    latent_values = [float(value) for value in latent[0].detach().cpu().tolist()]
    result_path = destination or config.paths.outputs / "caption_result.png"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].imshow(original)
    axes[0].set_title("Original image")
    axes[1].imshow(np.transpose(display_tensor.numpy(), (1, 2, 0)))
    axes[1].set_title(f"DIP input ({config.model.dip_mode})")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        "Generated caption: " + textwrap.fill(caption or "<empty>", width=75), fontsize=13
    )
    figure.tight_layout()
    figure.savefig(result_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    write_json_atomic(
        config.paths.outputs / "caption_result.json",
        {
            "image": str(image_path),
            "caption": caption,
            "latent_6d": latent_values,
            "checkpoint_epoch": loaded.checkpoint_epoch,
        },
    )
    return caption, result_path


def create_evidence_montage(config: ProjectConfig) -> Path:
    config.paths.create()
    panels = (
        (config.paths.outputs / "dataset_structure.png", "Flickr8k data structure"),
        (config.paths.outputs / "architecture_diagram.png", "Required architecture"),
        (config.paths.outputs / "training_curve.png", "Training and validation loss"),
        (config.paths.outputs / "caption_result.png", "Example caption"),
        (config.paths.outputs / "tsne_latent_space.png", "6-D latent t-SNE"),
        (config.paths.outputs / "sensitivity_analysis.png", "Noise sensitivity"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(16, 17))
    for axis, (path, title) in zip(axes.flat, panels, strict=True):
        axis.axis("off")
        axis.set_title(title, fontsize=12, weight="bold")
        if path.is_file():
            with Image.open(path) as image:
                axis.imshow(image.convert("RGB"))
        else:
            axis.text(
                0.5,
                0.5,
                f"Missing {path.name}\nRun the corresponding command first.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
    figure.suptitle("COSC 6324 Image Captioning - Submission Evidence", fontsize=17, weight="bold")
    figure.tight_layout()
    destination = config.paths.screenshots / "project_evidence.png"
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination
