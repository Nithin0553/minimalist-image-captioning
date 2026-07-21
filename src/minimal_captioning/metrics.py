from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import nltk
from nltk.corpus import wordnet as nltk_wordnet
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score

from .text import tokenize_caption


class MeteorResourceError(RuntimeError):
    """Raised when the WordNet data required by standard METEOR is unavailable."""


class SynsetProvider(Protocol):
    def synsets(self, word: str) -> list[object]: ...


@dataclass(frozen=True, kw_only=True)
class CaptionMetrics:
    bleu_1: float
    bleu_4: float
    meteor: float
    evaluated_images: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "bleu_1": self.bleu_1,
            "bleu_4": self.bleu_4,
            "meteor": self.meteor,
            "evaluated_images": self.evaluated_images,
        }


def ensure_meteor_resources(*, download: bool = False) -> None:
    try:
        nltk_wordnet.ensure_loaded()
        return
    except LookupError as error:
        if not download:
            raise MeteorResourceError(
                "NLTK WordNet data is required for standard METEOR. "
                "Run `uv run minimal-caption setup-nltk`."
            ) from error
    downloaded = nltk.download("wordnet", quiet=False) and nltk.download("omw-1.4", quiet=False)
    if not downloaded:
        raise MeteorResourceError("NLTK could not download WordNet and omw-1.4")
    try:
        nltk_wordnet.ensure_loaded()
    except LookupError as error:
        raise MeteorResourceError("WordNet download completed but cannot be loaded") from error


def compute_caption_metrics(
    references: list[list[str]],
    predictions: list[str],
    *,
    wordnet_provider: SynsetProvider | None = None,
) -> CaptionMetrics:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")
    if not predictions:
        raise ValueError("at least one prediction is required")
    tokenized_references = [
        [tokenize_caption(reference) for reference in image_references]
        for image_references in references
    ]
    tokenized_predictions = [tokenize_caption(prediction) for prediction in predictions]
    smoothing = SmoothingFunction().method1
    bleu_1 = float(
        corpus_bleu(
            tokenized_references,
            tokenized_predictions,
            weights=(1.0, 0.0, 0.0, 0.0),
            smoothing_function=smoothing,
        )
    )
    bleu_4 = float(
        corpus_bleu(
            tokenized_references,
            tokenized_predictions,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        )
    )
    if wordnet_provider is None:
        ensure_meteor_resources()
        provider = cast(SynsetProvider, nltk_wordnet)
    else:
        provider = wordnet_provider
    meteor_values = [
        meteor_score(image_references, prediction, wordnet=provider)
        for image_references, prediction in zip(
            tokenized_references, tokenized_predictions, strict=True
        )
    ]
    return CaptionMetrics(
        bleu_1=bleu_1,
        bleu_4=bleu_4,
        meteor=float(sum(meteor_values) / len(meteor_values)),
        evaluated_images=len(predictions),
    )
