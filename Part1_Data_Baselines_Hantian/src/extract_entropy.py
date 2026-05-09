"""Token-level entropy from probability distributions and ``TokenFeature`` rows."""

from __future__ import annotations

import math
import statistics
from typing import Sequence

from schemas import TokenFeature

__all__ = [
    "entropy_from_probs",
    "distributions_to_entropies",
    "build_token_features",
    "build_token_features_from_distributions",
    "entropy_summary",
]


def _finite(x: float | None) -> bool:
    if x is None:
        return False
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return False
    return True


def _as_float(x: float | None, *, default: float = float("nan")) -> float:
    if not _finite(x):
        return default
    return float(x)


def entropy_from_probs(
    probs: Sequence[float | None],
    *,
    eps: float = 1e-12,
) -> float:
    """Shannon entropy (nats) of a discrete distribution.

    Non-finite and negative entries are dropped. Remaining masses are renormalized.
    Returns ``nan`` if nothing usable remains or the total mass is ~0.
    """
    cleaned: list[float] = []
    for p in probs:
        if not _finite(p):
            continue
        pf = float(p)
        if pf < 0.0:
            continue
        cleaned.append(pf)

    if not cleaned:
        return float("nan")

    total = math.fsum(cleaned)
    if total <= eps:
        return float("nan")

    h = 0.0
    for p in cleaned:
        pn = p / total
        if pn > eps:
            h -= pn * math.log(pn)
    return h


def distributions_to_entropies(
    distributions: Sequence[Sequence[float | None]],
) -> list[float]:
    """Map each step's probability vector to an entropy scalar."""
    return [entropy_from_probs(row) for row in distributions]


def build_token_features_from_distributions(
    logprobs: Sequence[float | None],
    distributions: Sequence[Sequence[float | None]],
) -> list[TokenFeature]:
    """Entropy per step from full distributions, aligned with chosen-token ``logprobs``."""
    if len(logprobs) != len(distributions):
        raise ValueError(
            f"logprobs length {len(logprobs)} != distributions length {len(distributions)}"
        )
    entropies = distributions_to_entropies(distributions)
    return build_token_features(logprobs, entropies)


def build_token_features(
    logprobs: Sequence[float | None],
    entropies: Sequence[float | None],
) -> list[TokenFeature]:
    """Zip parallel per-token ``logprob`` (chosen token) and distribution entropy."""
    if len(logprobs) != len(entropies):
        raise ValueError(
            f"logprobs length {len(logprobs)} != entropies length {len(entropies)}"
        )
    out: list[TokenFeature] = []
    for i, (lp, ent) in enumerate(zip(logprobs, entropies)):
        out.append(
            TokenFeature(
                token_index=i,
                logprob=_as_float(lp),
                entropy=_as_float(ent),
            )
        )
    return out


def entropy_summary(values: Sequence[float | None]) -> dict[str, float]:
    """Mean and max over finite values; ``nan`` if no finite inputs."""
    xs = [_as_float(v) for v in values]
    finite = [x for x in xs if _finite(x)]
    if not finite:
        return {"mean": float("nan"), "max": float("nan")}
    return {
        "mean": float(statistics.fmean(finite)),
        "max": float(max(finite)),
    }
