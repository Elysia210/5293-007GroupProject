"""Unified JSONL-oriented types for Step 1 (GSM8K / MATH500)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CleanSample",
    "CandidatePath",
    "TokenFeature",
    "StepFeature",
]


@dataclass
class TokenFeature:
    """Per-token model statistics aligned to the generated reasoning sequence."""

    token_index: int
    logprob: float
    entropy: float


@dataclass
class StepFeature:
    """A reasoning step as a token span with aggregate entropy over that span."""

    start_token: int
    end_token: int  # exclusive, same convention as Python slicing
    mean_entropy: float
    max_entropy: float
    min_entropy: float
    std_entropy: float


@dataclass
class CandidatePath:
    """One sampled reasoning trace and its extracted answer."""

    reasoning_text: str
    predicted_answer_raw: str
    predicted_answer_normalized: str
    tokens: list[TokenFeature] = field(default_factory=list)
    steps: list[StepFeature] = field(default_factory=list)


@dataclass
class CleanSample:
    """Dataset-agnostic problem record with gold answers and optional candidates."""

    dataset: str
    sample_id: str
    question: str
    answer_raw: str
    answer_normalized: str
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    candidates: list[CandidatePath] = field(default_factory=list)
