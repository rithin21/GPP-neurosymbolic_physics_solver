from __future__ import annotations

import re

_TOKEN_SPLIT_RE = re.compile(r"[\s_\-]+")


_VOWELS = set("aeiou")


def _stemmed_token(token: str) -> str:
    """Escape a token but tolerate simple plural/verb-tense inflection.

    Lets a single accepted spelling (e.g. "start") also match "starts",
    "started", "starting", and the British doubled-consonant forms
    ("travelled", "travelling") instead of requiring every inflected form
    to be listed out as its own literal string.
    """
    escaped = re.escape(token)
    if len(token) <= 3:
        return escaped
    stem = token[:-1] if token.endswith("s") and not token.endswith("ss") else token
    # Optionally double the final consonant (British spelling: travel -> travelled/travelling).
    doubled = f"(?:{re.escape(stem[-1])})?" if stem[-1] not in _VOWELS else ""
    return re.escape(stem) + doubled + r"(?:e?s|e?d|ing)?"


def phrase_to_pattern(phrase: str) -> str:
    """Turn one literal phrase into a regex fragment tolerant of spacing,
    underscores/hyphens vs. spaces, and mild inflection - so callers don't
    need to enumerate every surface variant as a separate accepted string.
    """
    tokens = [tok for tok in _TOKEN_SPLIT_RE.split(phrase.strip().lower()) if tok]
    if not tokens:
        return ""
    body = r"[\s_-]+".join(_stemmed_token(tok) for tok in tokens)
    return rf"\b{body}\b"


def compile_phrase(phrase: str) -> re.Pattern[str]:
    pattern = phrase_to_pattern(phrase)
    return re.compile(pattern or r"(?!x)x", re.IGNORECASE)


def compile_phrase_set(phrases) -> re.Pattern[str]:
    """Compile many accepted phrases into a single alternation pattern."""
    fragments = [phrase_to_pattern(phrase) for phrase in phrases]
    fragments = [fragment for fragment in fragments if fragment]
    if not fragments:
        return re.compile(r"(?!x)x")  # matches nothing
    return re.compile("|".join(fragments), re.IGNORECASE)


def fullmatch_any(text: str, phrases) -> bool:
    normalized = text.strip().lower()
    return any(re.fullmatch(phrase_to_pattern(phrase), normalized, re.IGNORECASE) for phrase in phrases)
