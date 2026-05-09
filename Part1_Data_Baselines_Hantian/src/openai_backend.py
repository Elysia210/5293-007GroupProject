"""OpenAI Chat Completions backend: real reasoning + token logprobs / predictive entropy.

Entropy is computed from each step's ``top_logprobs`` (up to 20), softmax-normalized.
That is entropy under the *truncated* distribution returned by the API, not the full
vocabulary — a standard practical estimate for uncertainty signals.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any

from generate_candidates import GenerationConfig
from normalize_answers import extract_final_answer, normalize_answer
from schemas import CandidatePath, TokenFeature

try:
    from openai import (
        APIConnectionError,
        APIError,
        APITimeoutError,
        OpenAI,
        RateLimitError,
    )

    _OPENAI_RETRYABLE = (
        APIError,
        APIConnectionError,
        RateLimitError,
        APITimeoutError,
    )
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]
    _OPENAI_RETRYABLE = ()


def _entropy_nats_from_top_logprobs(top: list[Any]) -> float:
    """Shannon entropy (nats) from OpenAI ``top_logprobs`` at one generation step."""
    if not top:
        return float("nan")
    lps = [float(getattr(t, "logprob", float("-inf"))) for t in top]
    m = max(lps)
    exps = [math.exp(lp - m) for lp in lps]
    z = math.fsum(exps)
    if z <= 0.0:
        return float("nan")
    h = 0.0
    for e in exps:
        p = e / z
        if p > 0.0:
            h -= p * math.log(p)
    return h


def paths_from_openai_response(resp: Any) -> list[CandidatePath]:
    """Build ``CandidatePath`` rows (with ``tokens``) from a chat completion response."""
    out: list[CandidatePath] = []
    for ch in resp.choices:
        msg = ch.message
        text = (msg.content or "").strip()
        final_raw = extract_final_answer(text)
        norm = normalize_answer(final_raw)
        tokens: list[TokenFeature] = []
        lp_content = getattr(getattr(ch, "logprobs", None), "content", None) or []
        for i, tok_lp in enumerate(lp_content):
            chosen_lp = float(tok_lp.logprob)
            top = list(tok_lp.top_logprobs or [])
            ent = _entropy_nats_from_top_logprobs(top)
            tokens.append(
                TokenFeature(token_index=i, logprob=chosen_lp, entropy=ent)
            )
        out.append(
            CandidatePath(
                reasoning_text=text,
                predicted_answer_raw=final_raw,
                predicted_answer_normalized=norm,
                tokens=tokens,
            )
        )
    return out


@dataclass
class OpenAIBackend:
    """``n=config.k`` completions in one call; fills ``CandidatePath.tokens`` from logprobs."""

    model: str = "gpt-4o-mini"
    base_url: str | None = None
    top_logprobs: int = 20
    max_retries: int = 3

    def __post_init__(self) -> None:
        if OpenAI is None:
            raise ImportError("Install the OpenAI SDK: pip install openai")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY is not set (required for --backend openai)"
            )
        kwargs: dict[str, Any] = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)

    def generate_paths(
        self,
        messages: list[dict[str, str]],
        *,
        config: GenerationConfig,
        sample: Any = None,
    ) -> list[CandidatePath]:
        tl = max(0, min(20, int(self.top_logprobs)))
        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "n": config.k,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
        }
        if config.seed is not None:
            create_kwargs["seed"] = config.seed
        if tl > 0:
            create_kwargs["logprobs"] = True
            create_kwargs["top_logprobs"] = tl

        retryable = _OPENAI_RETRYABLE + (TimeoutError, OSError)
        last_exc: BaseException | None = None
        attempts = max(1, int(self.max_retries) + 1)
        for attempt in range(attempts):
            try:
                resp = self._client.chat.completions.create(**create_kwargs)
                paths = paths_from_openai_response(resp)
                if len(paths) != config.k:
                    raise ValueError(
                        f"OpenAI returned {len(paths)} choices; expected config.k={config.k}"
                    )
                return paths
            except retryable as e:
                last_exc = e
                if attempt >= attempts - 1:
                    raise
                delay = (2**attempt) + random.uniform(0.0, 0.35)
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def generate_texts(
        self,
        messages: list[dict[str, str]],
        *,
        config: GenerationConfig,
        sample: Any = None,
    ) -> list[str]:
        return [p.reasoning_text for p in self.generate_paths(messages, config=config, sample=sample)]
