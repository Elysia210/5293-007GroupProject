"""Sample K chain-of-thought candidates per ``CleanSample`` (backend-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

from normalize_answers import extract_final_answer, normalize_answer
from schemas import CandidatePath, CleanSample

__all__ = [
    "GenerationConfig",
    "GenerationBackend",
    "DEFAULT_COT_SYSTEM",
    "build_user_prompt",
    "build_cot_messages",
    "generate_candidates",
    "generate_candidates_for_samples",
    "FunctionBackend",
    "EchoBackend",
]

DEFAULT_COT_SYSTEM = (
    "You are a careful math tutor. Reason step by step. "
    "Use clear intermediate steps and check your work when helpful."
)


@dataclass
class GenerationConfig:
    """Shared generation knobs; backends may ignore fields they do not support."""

    k: int = 4
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 2048
    seed: int | None = None
    system_prompt: str | None = None
    # Extra kwargs for a concrete client (e.g. model name, base_url, timeout).
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GenerationBackend(Protocol):
    """Pluggable generator: must return exactly ``config.k`` completion strings."""

    def generate_texts(
        self,
        messages: list[dict[str, str]],
        *,
        config: GenerationConfig,
        sample: CleanSample | None = None,
    ) -> list[str]:
        ...


def build_user_prompt(question: str) -> str:
    """User turn: instruct CoT and a machine-parsable final line (works with ``extract_final_answer``)."""
    q = question.strip()
    return (
        "Solve the following problem. Show your reasoning step by step.\n"
        "On the last line, write the final answer exactly in this form:\n"
        "#### <answer>\n\n"
        f"Problem:\n{q}"
    )


def build_cot_messages(
    question: str,
    *,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """OpenAI-style chat messages: system + user."""
    system = (system_prompt if system_prompt is not None else DEFAULT_COT_SYSTEM).strip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_prompt(question)},
    ]


def _paths_from_texts(texts: list[str]) -> list[CandidatePath]:
    paths: list[CandidatePath] = []
    for raw_text in texts:
        text = raw_text.strip()
        final_raw = extract_final_answer(text)
        paths.append(
            CandidatePath(
                reasoning_text=text,
                predicted_answer_raw=final_raw,
                predicted_answer_normalized=normalize_answer(final_raw),
            )
        )
    return paths


def generate_candidates(
    sample: CleanSample,
    backend: GenerationBackend,
    config: GenerationConfig,
    *,
    mutate: bool = False,
) -> CleanSample:
    """Run the backend and attach ``CandidatePath`` list (length ``config.k``)."""
    if config.k < 0:
        raise ValueError("config.k must be non-negative")
    if config.k == 0:
        return sample if mutate else replace(sample, candidates=[])

    system = config.system_prompt
    messages = build_cot_messages(sample.question, system_prompt=system)
    paths_fn = getattr(backend, "generate_paths", None)
    if callable(paths_fn):
        candidates = paths_fn(messages, config=config, sample=sample)
        if len(candidates) != config.k:
            raise ValueError(
                f"backend returned {len(candidates)} paths; expected config.k={config.k}"
            )
    else:
        texts = backend.generate_texts(messages, config=config, sample=sample)
        if len(texts) != config.k:
            raise ValueError(
                f"backend returned {len(texts)} texts; expected config.k={config.k}"
            )
        candidates = _paths_from_texts(texts)
    if mutate:
        sample.candidates = candidates
        return sample
    return replace(sample, candidates=candidates)


def generate_candidates_for_samples(
    samples: Sequence[CleanSample],
    backend: GenerationBackend,
    config: GenerationConfig,
    *,
    mutate: bool = False,
) -> list[CleanSample]:
    return [generate_candidates(s, backend, config, mutate=mutate) for s in samples]


GenerationFn = Callable[
    [list[dict[str, str]], GenerationConfig, CleanSample | None, int],
    str,
]


@dataclass
class FunctionBackend:
    """Adapter for custom single-sample generation: called once per path index ``0..k-1``."""

    fn: GenerationFn

    def generate_texts(
        self,
        messages: list[dict[str, str]],
        *,
        config: GenerationConfig,
        sample: CleanSample | None = None,
    ) -> list[str]:
        return [self.fn(messages, config, sample, i) for i in range(config.k)]


@dataclass
class EchoBackend:
    """Deterministic stub for wiring tests (ignores ``messages``)."""

    template: str = "#### {i}"

    def generate_texts(
        self,
        messages: list[dict[str, str]],
        *,
        config: GenerationConfig,
        sample: CleanSample | None = None,
    ) -> list[str]:
        return [self.template.format(i=i) for i in range(config.k)]
