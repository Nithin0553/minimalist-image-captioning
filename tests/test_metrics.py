from __future__ import annotations

import pytest

from minimal_captioning.metrics import compute_caption_metrics


class EmptyWordNet:
    def synsets(self, word: str) -> list[object]:
        return []


def test_identical_captions_score_perfect_bleu() -> None:
    metrics = compute_caption_metrics(
        [["a small dog runs fast"]],
        ["a small dog runs fast"],
        wordnet_provider=EmptyWordNet(),
    )
    assert metrics.bleu_1 == pytest.approx(1.0)
    assert metrics.bleu_4 == pytest.approx(1.0)
    assert metrics.meteor > 0.9
    assert metrics.evaluated_images == 1


def test_metrics_reject_mismatched_input_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        compute_caption_metrics([["one"]], [], wordnet_provider=EmptyWordNet())
