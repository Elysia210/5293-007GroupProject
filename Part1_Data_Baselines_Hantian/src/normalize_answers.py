"""Normalize short math answers and pull finals from long rationales."""

from __future__ import annotations

import re
import unicodedata

__all__ = ["normalize_answer", "extract_final_answer"]

# Phrases to peel off the start (after NFKC + lowercasing inside normalize_answer).
_ANSWER_PREFIXES = (
    r"the\s+final\s+answer\s+is\s*:?\s*",
    r"the\s+answer\s+is\s*:?\s*",
    r"final\s+answer\s*:?\s*",
    r"answer\s*:?\s*",
    r"therefore\s*,?\s*",
    r"thus\s*,?\s*",
    r"hence\s*,?\s*",
    r"so\s*,?\s*",
    r"we\s+get\s*:?\s*",
    r"we\s+have\s*:?\s*",
    r"which\s+(?:is|equals?)\s*:?\s*",
    r"this\s+(?:is|equals?)\s*:?\s*",
    r"giving\s*:?\s*",
    r"equals?\s*:?\s*",
    r"=\s*",
)

_PREFIX_RE = re.compile(
    "|".join(f"(?:{p})" for p in _ANSWER_PREFIXES),
    flags=re.IGNORECASE,
)


def extract_final_answer(text: str) -> str:
    """Prefer GSM8K-style ``####`` tail, else first ``\\boxed{...}``, else last non-empty line."""
    raw = text.strip()
    if not raw:
        return ""

    if "####" in raw:
        tail = raw.rsplit("####", 1)[-1].strip()
        return tail

    boxed = _extract_boxed(raw)
    if boxed is not None:
        return boxed.strip()

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _extract_boxed(text: str) -> str | None:
    key = "\\boxed{"
    i = text.find(key)
    if i < 0:
        return None
    j = i + len(key)
    depth = 1
    start = j
    while j < len(text) and depth:
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        j += 1
    if depth != 0:
        return None
    return text[start : j - 1]


def normalize_answer(text: str) -> str:
    """Strip boilerplate, drop punctuation (keep ``/`` for fractions), tidy commas in integers."""
    s = unicodedata.normalize("NFKC", text.strip())
    if not s:
        return ""

    s = s.lower()

    # Flatten simple LaTeX \frac{a}{b} → a/b (repeat for shallow nesting).
    for _ in range(8):
        nxt = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", s)
        if nxt == s:
            break
        s = nxt

    s = s.replace("$", "")
    s = re.sub(r"\\(?:left|right|middle)\b", "", s)
    s = re.sub(r"\\[,;:!]", " ", s)
    s = re.sub(r"\\text\{([^{}]+)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", s)

    # Remove thousand separators: 12,345 → 12345
    while re.search(r"\d,\d", s):
        s = re.sub(r"(\d),(\d)", r"\1\2", s)

    changed = True
    while changed:
        changed = False
        t = _PREFIX_RE.sub("", s).lstrip()
        if t != s:
            changed = True
            s = t

    # Drop punctuation; keep letters/digits/underscore, space, fraction slash, sign/dot/plus.
    s = re.sub(r"[^\w\s/.+\-]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()

    # Normalize integers written as n.000… (does not alter values like 3.14 or 14/3).
    s = re.sub(r"(-?\d+)\.0+(?=\s|$)", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Drop a trailing period after an integer (sentence punctuation, not a decimal).
    s = re.sub(r"(-?\d+)\.(?=\s*$)", r"\1", s)

    return s
