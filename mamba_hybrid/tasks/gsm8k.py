"""GSM8K final-answer parsing and reversible byte tokenization."""

import re

PAD = 0
EOS = 1
BOS = 2
BYTE_OFFSET = 3
VOCAB_SIZE = 259

_INTEGER = re.compile(r"^-?\d+$")


def normalize_answer(value: str) -> str:
    """Normalize a GSM8K final integer answer."""
    normalized = value.strip().replace(",", "")
    if not _INTEGER.fullmatch(normalized):
        raise ValueError(f"GSM8K answer is not an integer: {value!r}")
    sign = "-" if normalized.startswith("-") else ""
    digits = normalized.removeprefix("-").lstrip("0") or "0"
    return sign + digits


def extract_answer(answer_text: str) -> str:
    """Extract and normalize the official answer after the final #### marker."""
    marker, separator, value = answer_text.rpartition("####")
    if not separator or not marker.strip():
        raise ValueError("GSM8K answer is missing a rationale and #### final answer")
    return normalize_answer(value)


def encode_bytes(text: str, *, add_eos: bool = True) -> list[int]:
    """Encode UTF-8 bytes into IDs that reserve PAD/BOS/EOS."""
    tokens = [byte + BYTE_OFFSET for byte in text.encode("utf-8")]
    return [*tokens, EOS] if add_eos else tokens


def decode_bytes(tokens: list[int]) -> str:
    """Decode byte IDs through EOS, rejecting malformed token IDs."""
    values: list[int] = []
    for token in tokens:
        if token == EOS:
            break
        if token in (PAD, BOS):
            continue
        value = token - BYTE_OFFSET
        if not 0 <= value <= 255:
            raise ValueError(f"invalid byte token: {token}")
        values.append(value)
    return bytes(values).decode("utf-8")


def encode_answer(answer: str) -> tuple[list[int], list[int]]:
    """Return teacher-forcing decoder inputs and prediction targets."""
    encoded = encode_bytes(normalize_answer(answer))
    return [BOS, *encoded[:-1]], encoded


def allowed_answer_tokens() -> frozenset[int]:
    """Return output IDs allowed for normalized integer answers."""
    characters = "-0123456789"
    return frozenset({EOS, *(ord(character) + BYTE_OFFSET for character in characters)})


__all__ = [
    "BOS",
    "BYTE_OFFSET",
    "EOS",
    "PAD",
    "VOCAB_SIZE",
    "allowed_answer_tokens",
    "decode_bytes",
    "encode_answer",
    "encode_bytes",
    "extract_answer",
    "normalize_answer",
]
