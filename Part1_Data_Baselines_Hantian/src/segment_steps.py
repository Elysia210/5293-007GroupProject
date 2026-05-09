"""Split reasoning text into steps and aggregate per-step entropy into ``StepFeature``."""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import replace
from typing import Iterable

from schemas import CandidatePath, StepFeature, TokenFeature

__all__ = [
    "split_reasoning_spans",
    "step_features_from_spans",
    "segment_reasoning",
    "segment_candidate",
]

_STEP_HEADER = re.compile(r"(?mi)(?:^|\n)\s*Step\s+\d+\b")
_PAR_BREAK = re.compile(r"\n\s*\n+")


def _finite(x: float) -> bool:
    return math.isfinite(x)


def split_reasoning_spans(text: str) -> list[tuple[int, int]]:
    """Return non-overlapping ``(start, end)`` char spans into ``text`` (end exclusive).

    Priority: ``Step <n>`` headers, else blank-line blocks, else sentences.
    """
    if not text:
        return []

    if _STEP_HEADER.search(text):
        return _spans_by_step_headers(text)

    para = _spans_by_paragraphs(text)
    if len(para) > 1:
        return para

    body = text
    if len(para) == 1:
        body = text[para[0][0] : para[0][1]]
        base = para[0][0]
        sent = _spans_by_sentences(body)
        if not sent:
            sent = [(0, len(body))]
        return [(base + s, base + e) for s, e in sent]

    return _spans_by_sentences(text)


def _spans_by_step_headers(text: str) -> list[tuple[int, int]]:
    matches = list(_STEP_HEADER.finditer(text))
    if not matches:
        return _spans_by_sentences(text)

    spans: list[tuple[int, int]] = []
    if matches[0].start() > 0:
        pre = text[: matches[0].start()]
        if pre.strip():
            spans.append((0, matches[0].start()))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end]
        if chunk.strip():
            spans.append((start, end))

    return spans or [(0, len(text))]


def _spans_by_paragraphs(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    last = 0
    for m in _PAR_BREAK.finditer(text):
        if m.start() > last:
            seg = text[last : m.start()]
            if seg.strip():
                spans.append((last, m.start()))
        last = m.end()
    if last < len(text):
        seg = text[last:]
        if seg.strip():
            spans.append((last, len(text)))
    return spans or [(0, len(text))]


def _spans_by_sentences(text: str) -> list[tuple[int, int]]:
    if not text:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    for m in re.finditer(r"[.!?]+(?:\s+|$)", text):
        end = m.end()
        if end > start and text[start:end].strip():
            spans.append((start, end))
        start = end
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def _char_span_to_token_span(
    c0: int,
    c1: int,
    *,
    n_chars: int,
    n_tokens: int,
) -> tuple[int, int]:
    """Map a char span to token indices via proportional alignment (no tokenizer API)."""
    if n_tokens <= 0:
        return (0, 0)
    if n_chars <= 0:
        return (0, min(1, n_tokens))

    c0 = max(0, min(n_chars, c0))
    c1 = max(0, min(n_chars, c1))
    if c1 <= c0:
        mid = min(n_tokens, max(0, round(c0 * n_tokens / n_chars)))
        return (mid, mid)

    t0 = math.floor(c0 * n_tokens / n_chars)
    t1 = math.ceil(c1 * n_tokens / n_chars)
    t0 = max(0, min(n_tokens, t0))
    t1 = max(0, min(n_tokens, t1))
    if t1 < t0:
        t0, t1 = t1, t0
    if t1 == t0 and t1 < n_tokens:
        t1 += 1
    return (t0, t1)


def _entropy_stats(values: Iterable[float]) -> tuple[float, float, float, float]:
    finite = [float(x) for x in values if _finite(x)]
    if not finite:
        nan = float("nan")
        return (nan, nan, nan, nan)
    mean = float(statistics.fmean(finite))
    mx = float(max(finite))
    mn = float(min(finite))
    if len(finite) > 1:
        std = float(statistics.pstdev(finite))
    else:
        std = 0.0
    return (mean, mx, mn, std)


def step_features_from_spans(
    char_spans: Iterable[tuple[int, int]],
    tokens: list[TokenFeature],
    *,
    reasoning_text: str,
) -> list[StepFeature]:
    """Turn char spans into ``StepFeature`` rows (token bounds + entropy rollups)."""
    n_chars = len(reasoning_text)
    n_tok = len(tokens)
    out: list[StepFeature] = []

    for c0, c1 in char_spans:
        if n_tok == 0:
            nan = float("nan")
            out.append(StepFeature(0, 0, nan, nan, nan, nan))
            continue

        t0, t1 = _char_span_to_token_span(c0, c1, n_chars=n_chars, n_tokens=n_tok)
        slice_tokens = tokens[t0:t1]
        entropies = [t.entropy for t in slice_tokens]
        mean, mx, mn, std = _entropy_stats(entropies)
        out.append(
            StepFeature(
                start_token=t0,
                end_token=t1,
                mean_entropy=mean,
                max_entropy=mx,
                min_entropy=mn,
                std_entropy=std,
            )
        )
    return out


def segment_reasoning(
    reasoning_text: str,
    tokens: list[TokenFeature],
) -> list[StepFeature]:
    """Segment ``reasoning_text`` and compute per-step entropy stats from ``tokens``."""
    spans = split_reasoning_spans(reasoning_text)
    if not spans:
        return []
    return step_features_from_spans(spans, tokens, reasoning_text=reasoning_text)


def segment_candidate(path: CandidatePath) -> CandidatePath:
    """Return a copy of ``path`` with ``steps`` filled from ``reasoning_text`` + ``tokens``."""
    steps = segment_reasoning(path.reasoning_text, path.tokens)
    return replace(path, steps=steps)
