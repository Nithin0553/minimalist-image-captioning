from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ProjectConfig
from .data import (
    EvaluationBatch,
    PreparedData,
    create_evaluation_loader,
    prepare_data,
)
from .io_utils import write_json_atomic, write_text_atomic
from .metrics import CaptionMetrics, compute_caption_metrics, ensure_meteor_resources
from .preprocessing import NoiseSpec
from .text import tokenize_caption
from .training import LoadedModel, load_trained_model, resolve_device

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


@dataclass(frozen=True, kw_only=True)
class ImagePrediction:
    image_name: str
    prediction: str
    references: tuple[str, ...]
    latent: tuple[float, float, float, float, float, float]


@dataclass(frozen=True, kw_only=True)
class EvaluationRun:
    noise: NoiseSpec
    metrics: CaptionMetrics
    predictions: tuple[ImagePrediction, ...]


@dataclass(frozen=True, kw_only=True)
class AnalysisResult:
    clean_metrics: CaptionMetrics
    metrics_path: Path
    predictions_path: Path
    tsne_path: Path
    sensitivity_path: Path


def collect_predictions(
    loaded: LoadedModel,
    loader: DataLoader[EvaluationBatch],
    *,
    device: torch.device,
    max_generation_length: int,
    noise: NoiseSpec,
) -> EvaluationRun:
    loaded.model.eval()
    results: list[ImagePrediction] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluate {noise.label}", leave=False):
            images = batch.images.to(device, non_blocking=True)
            token_tensor, latent_tensor = loaded.model.generate(
                images,
                bos_index=loaded.vocabulary.bos_index,
                eos_index=loaded.vocabulary.eos_index,
                max_length=max_generation_length,
            )
            token_rows = token_tensor.detach().cpu().tolist()
            latent_rows = latent_tensor.detach().cpu().tolist()
            for image_name, references, tokens, latent in zip(
                batch.image_names,
                batch.references,
                token_rows,
                latent_rows,
                strict=True,
            ):
                if len(latent) != 6:
                    raise RuntimeError(f"expected a 6-D latent, got {len(latent)} values")
                results.append(
                    ImagePrediction(
                        image_name=image_name,
                        prediction=loaded.vocabulary.decode(tokens),
                        references=references,
                        latent=(
                            float(latent[0]),
                            float(latent[1]),
                            float(latent[2]),
                            float(latent[3]),
                            float(latent[4]),
                            float(latent[5]),
                        ),
                    )
                )
    metrics = compute_caption_metrics(
        [list(item.references) for item in results],
        [item.prediction for item in results],
    )
    return EvaluationRun(noise=noise, metrics=metrics, predictions=tuple(results))


def evaluate_condition(
    config: ProjectConfig,
    prepared: PreparedData,
    loaded: LoadedModel,
    *,
    device: torch.device,
    noise: NoiseSpec | None = None,
) -> EvaluationRun:
    selected_noise = noise or NoiseSpec()
    loader = create_evaluation_loader(config, prepared, noise=selected_noise)
    return collect_predictions(
        loaded,
        loader,
        device=device,
        max_generation_length=config.evaluation.max_generation_length,
        noise=selected_noise,
    )


def write_clean_evaluation(config: ProjectConfig, run: EvaluationRun) -> tuple[Path, Path]:
    metrics_path = config.paths.outputs / "caption_metrics.json"
    write_json_atomic(metrics_path, run.metrics.to_dict())
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        [
            "image",
            "prediction",
            "reference_1",
            "reference_2",
            "reference_3",
            "reference_4",
            "reference_5",
        ]
    )
    for item in run.predictions:
        padded_references = [
            *item.references,
            *("" for _ in range(max(0, 5 - len(item.references)))),
        ]
        writer.writerow([item.image_name, item.prediction, *padded_references[:5]])
    predictions_path = config.paths.outputs / "test_predictions.csv"
    write_text_atomic(predictions_path, stream.getvalue())
    return metrics_path, predictions_path


SCENE_KEYWORDS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "animals",
        frozenset({"dog", "dogs", "cat", "cats", "horse", "horses", "bird", "birds", "animal"}),
    ),
    (
        "sports",
        frozenset(
            {
                "ball",
                "football",
                "soccer",
                "baseball",
                "basketball",
                "tennis",
                "skateboard",
                "surfing",
                "racing",
            }
        ),
    ),
    (
        "water",
        frozenset(
            {"water", "ocean", "sea", "beach", "river", "lake", "pool", "wave", "waves", "surf"}
        ),
    ),
    (
        "vehicles",
        frozenset(
            {"car", "cars", "truck", "bus", "train", "bike", "bicycle", "motorcycle", "boat"}
        ),
    ),
    (
        "people",
        frozenset(
            {"man", "woman", "boy", "girl", "child", "children", "people", "person", "men", "women"}
        ),
    ),
    (
        "nature",
        frozenset(
            {"tree", "trees", "grass", "field", "mountain", "forest", "snow", "rock", "rocks"}
        ),
    ),
    (
        "indoor",
        frozenset({"room", "kitchen", "table", "chair", "bed", "house", "building", "indoor"}),
    ),
)


def infer_scene_label(references: tuple[str, ...]) -> str:
    words = {word for reference in references for word in tokenize_caption(reference)}
    for label, keywords in SCENE_KEYWORDS:
        if words & keywords:
            return label
    return "other"


def create_tsne_visualization(config: ProjectConfig, run: EvaluationRun) -> Path:
    selected = run.predictions[: config.evaluation.tsne_max_images]
    if len(selected) < 3:
        raise ValueError("t-SNE requires at least three evaluated images")
    latents = np.asarray([item.latent for item in selected], dtype=np.float64)
    labels = [infer_scene_label(item.references) for item in selected]
    perplexity = min(30.0, max(2.0, (len(selected) - 1) / 3.0), float(len(selected) - 1))
    embedded = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=config.seed,
        max_iter=1000,
    ).fit_transform(latents)

    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        [
            "image",
            "scene_label",
            "z1",
            "z2",
            "z3",
            "z4",
            "z5",
            "z6",
            "tsne_x",
            "tsne_y",
            "prediction",
        ]
    )
    for item, label, point in zip(selected, labels, embedded, strict=True):
        writer.writerow(
            [
                item.image_name,
                label,
                *item.latent,
                float(point[0]),
                float(point[1]),
                item.prediction,
            ]
        )
    write_text_atomic(config.paths.outputs / "latent_vectors.csv", stream.getvalue())

    figure, axis = plt.subplots(figsize=(10, 7))
    palette = matplotlib.colormaps["tab10"]
    for color_index, label in enumerate(sorted(set(labels))):
        mask = np.asarray([item_label == label for item_label in labels])
        axis.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            s=28,
            alpha=0.75,
            label=label,
            color=palette(color_index % 10),
        )
    axis.set(
        title="t-SNE of the 6-D Image Bottleneck",
        xlabel="t-SNE dimension 1",
        ylabel="t-SNE dimension 2",
    )
    axis.grid(alpha=0.2)
    axis.legend(title="Scene content", loc="best")
    figure.tight_layout()
    destination = config.paths.outputs / "tsne_latent_space.png"
    figure.savefig(destination, dpi=200)
    plt.close(figure)
    return destination


def _latent_robustness(clean: EvaluationRun, noisy: EvaluationRun) -> tuple[float, float]:
    clean_by_name = {item.image_name: np.asarray(item.latent) for item in clean.predictions}
    shifts: list[float] = []
    similarities: list[float] = []
    for item in noisy.predictions:
        baseline = clean_by_name[item.image_name]
        degraded = np.asarray(item.latent)
        shifts.append(float(np.linalg.norm(degraded - baseline)))
        denominator = float(np.linalg.norm(degraded) * np.linalg.norm(baseline))
        similarities.append(float(np.dot(degraded, baseline) / denominator) if denominator else 0.0)
    return float(np.mean(shifts)), float(np.mean(similarities))


def run_sensitivity_analysis(
    config: ProjectConfig,
    prepared: PreparedData,
    loaded: LoadedModel,
    *,
    device: torch.device,
    clean: EvaluationRun,
) -> Path:
    conditions = [
        *(
            NoiseSpec(kind="gaussian", level=level)
            for level in config.evaluation.gaussian_noise_std
        ),
        *(
            NoiseSpec(kind="salt_pepper", level=level)
            for level in config.evaluation.salt_pepper_amount
        ),
    ]
    rows: list[dict[str, float | int | str]] = [
        {
            "condition": "clean",
            "noise_kind": "none",
            "noise_level": 0.0,
            **clean.metrics.to_dict(),
            "mean_latent_l2_shift": 0.0,
            "mean_latent_cosine_similarity": 1.0,
        }
    ]
    for condition in conditions:
        noisy = evaluate_condition(config, prepared, loaded, device=device, noise=condition)
        shift, similarity = _latent_robustness(clean, noisy)
        rows.append(
            {
                "condition": condition.label,
                "noise_kind": condition.kind,
                "noise_level": condition.level,
                **noisy.metrics.to_dict(),
                "mean_latent_l2_shift": shift,
                "mean_latent_cosine_similarity": similarity,
            }
        )

    fieldnames = [
        "condition",
        "noise_kind",
        "noise_level",
        "bleu_1",
        "bleu_4",
        "meteor",
        "evaluated_images",
        "mean_latent_l2_shift",
        "mean_latent_cosine_similarity",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    csv_path = config.paths.outputs / "sensitivity_results.csv"
    write_text_atomic(csv_path, stream.getvalue())
    write_json_atomic(config.paths.outputs / "sensitivity_results.json", rows)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    metric_specs = (("bleu_1", "BLEU-1"), ("bleu_4", "BLEU-4"), ("meteor", "METEOR"))
    for axis, (key, label) in zip(axes, metric_specs, strict=True):
        for kind, display in (("gaussian", "Gaussian"), ("salt_pepper", "Salt-and-pepper")):
            subset = [row for row in rows if row["noise_kind"] in {"none", kind}]
            axis.plot(
                [float(row["noise_level"]) for row in subset],
                [float(row[key]) for row in subset],
                marker="o",
                label=display,
            )
        axis.set(title=label, xlabel="Noise level", ylabel="Score")
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.suptitle("Caption Robustness Through the 6-D Bottleneck")
    figure.tight_layout()
    plot_path = config.paths.outputs / "sensitivity_analysis.png"
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)
    return plot_path


def evaluate_checkpoint(
    config: ProjectConfig,
    checkpoint: Path,
    *,
    include_sensitivity: bool,
) -> AnalysisResult:
    config.paths.create()
    ensure_meteor_resources()
    device = resolve_device(config.training.device)
    loaded = load_trained_model(config, checkpoint, device=device)
    prepared = prepare_data(config, vocabulary=loaded.vocabulary)
    clean = evaluate_condition(config, prepared, loaded, device=device)
    metrics_path, predictions_path = write_clean_evaluation(config, clean)
    tsne_path = create_tsne_visualization(config, clean)
    sensitivity_path = (
        run_sensitivity_analysis(config, prepared, loaded, device=device, clean=clean)
        if include_sensitivity
        else config.paths.outputs / "sensitivity_analysis.png"
    )
    return AnalysisResult(
        clean_metrics=clean.metrics,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        tsne_path=tsne_path,
        sensitivity_path=sensitivity_path,
    )
