from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
PAD_TOKEN: Final = "<pad>"
UNK_TOKEN: Final = "<unk>"
BOS_TOKEN: Final = "<bos>"
EOS_TOKEN: Final = "<eos>"
SPECIAL_TOKENS: Final[tuple[str, ...]] = (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)


def tokenize_caption(text: str) -> list[str]:
    """Lowercase and tokenize a caption without requiring external tokenizer data."""

    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True, kw_only=True)
class Vocabulary:
    token_to_index: dict[str, int]
    index_to_token: tuple[str, ...]

    @classmethod
    def build(cls, captions: list[str], *, min_frequency: int) -> Vocabulary:
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        counts: Counter[str] = Counter()
        for caption in captions:
            counts.update(tokenize_caption(caption))
        kept = sorted(
            (token for token, count in counts.items() if count >= min_frequency),
            key=lambda token: (-counts[token], token),
        )
        tokens = (*SPECIAL_TOKENS, *kept)
        mapping = {token: index for index, token in enumerate(tokens)}
        return cls(token_to_index=mapping, index_to_token=tokens)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> Vocabulary:
        tokens = raw.get("index_to_token")
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise ValueError("invalid serialized vocabulary")
        normalized = tuple(tokens)
        if normalized[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError("serialized vocabulary has invalid special-token order")
        return cls(
            token_to_index={token: index for index, token in enumerate(normalized)},
            index_to_token=normalized,
        )

    def to_dict(self) -> dict[str, list[str]]:
        return {"index_to_token": list(self.index_to_token)}

    def __len__(self) -> int:
        return len(self.index_to_token)

    @property
    def pad_index(self) -> int:
        return self.token_to_index[PAD_TOKEN]

    @property
    def unk_index(self) -> int:
        return self.token_to_index[UNK_TOKEN]

    @property
    def bos_index(self) -> int:
        return self.token_to_index[BOS_TOKEN]

    @property
    def eos_index(self) -> int:
        return self.token_to_index[EOS_TOKEN]

    def encode(self, caption: str, *, max_length: int) -> list[int]:
        if max_length < 2:
            raise ValueError("max_length must allow BOS and EOS tokens")
        content = tokenize_caption(caption)[: max_length - 2]
        return [
            self.bos_index,
            *(self.token_to_index.get(token, self.unk_index) for token in content),
            self.eos_index,
        ]

    def decode(self, indices: list[int]) -> str:
        words: list[str] = []
        for index in indices:
            if index == self.eos_index:
                break
            if index in {self.pad_index, self.bos_index}:
                continue
            if 0 <= index < len(self.index_to_token):
                token = self.index_to_token[index]
            else:
                token = UNK_TOKEN
            words.append(token)
        return " ".join(words)
