from __future__ import annotations

from minimal_captioning.evaluation import (
    EvaluationRun,
    ImagePrediction,
    create_tsne_visualization,
    infer_scene_label,
)
from minimal_captioning.metrics import CaptionMetrics
from minimal_captioning.preprocessing import NoiseSpec


def test_scene_labels_use_human_reference_content() -> None:
    assert infer_scene_label(("A brown dog runs.",)) == "animals"
    assert infer_scene_label(("A wave reaches a beach.",)) == "water"
    assert infer_scene_label(("An unidentified object.",)) == "other"


def test_tsne_writes_plot_and_latent_csv(project_config) -> None:
    predictions = tuple(
        ImagePrediction(
            image_name=f"{index}.jpg",
            prediction="a dog runs",
            references=("a dog runs in a field",),
            latent=(
                float(index),
                float(index + 1),
                float(index + 2),
                float(index + 3),
                float(index + 4),
                float(index + 5),
            ),
        )
        for index in range(6)
    )
    run = EvaluationRun(
        noise=NoiseSpec(),
        metrics=CaptionMetrics(bleu_1=0.1, bleu_4=0.01, meteor=0.2, evaluated_images=6),
        predictions=predictions,
    )
    path = create_tsne_visualization(project_config, run)
    assert path.is_file()
    assert (project_config.paths.outputs / "latent_vectors.csv").is_file()
