from __future__ import annotations

import pytest

from minimal_captioning.text import SPECIAL_TOKENS, Vocabulary, tokenize_caption


def test_tokenizer_normalizes_case_and_punctuation() -> None:
    assert tokenize_caption("A Dog's BALL, #2!") == ["a", "dog's", "ball", "2"]


def test_vocabulary_frequency_and_stable_order() -> None:
    vocabulary = Vocabulary.build(["blue dog runs", "dog runs", "cat sleeps"], min_frequency=2)
    assert vocabulary.index_to_token[:4] == SPECIAL_TOKENS
    assert vocabulary.index_to_token[4:] == ("dog", "runs")


def test_encode_truncates_and_decode_stops_at_eos() -> None:
    vocabulary = Vocabulary.build(["one two three four"], min_frequency=1)
    encoded = vocabulary.encode("one two three four", max_length=4)
    assert encoded == [
        vocabulary.bos_index,
        vocabulary.token_to_index["one"],
        vocabulary.token_to_index["two"],
        vocabulary.eos_index,
    ]
    assert vocabulary.decode([vocabulary.bos_index, 4, 5, vocabulary.eos_index, 6]) == "four one"


def test_unknown_words_use_unknown_index() -> None:
    vocabulary = Vocabulary.build(["known"], min_frequency=1)
    assert vocabulary.unk_index in vocabulary.encode("missing", max_length=3)


def test_vocabulary_round_trip() -> None:
    vocabulary = Vocabulary.build(["a small dog", "a large dog"], min_frequency=1)
    assert Vocabulary.from_dict(vocabulary.to_dict()) == vocabulary


def test_invalid_serialized_vocabulary_is_rejected() -> None:
    with pytest.raises(ValueError, match="special-token"):
        Vocabulary.from_dict({"index_to_token": ["bad"]})
