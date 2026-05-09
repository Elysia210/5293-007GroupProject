"""Load GSM8K and MATH-500 into unified ``CleanSample`` records."""

from __future__ import annotations

from typing import Any, Sequence

from datasets import Dataset, load_dataset

from schemas import CleanSample

GSM8K_REPO = "openai/gsm8k"
GSM8K_CONFIG = "main"  # vs "socratic"; required when multiple configs exist
MATH500_REPO = "HuggingFaceH4/MATH-500"

_LOADER_TAG = "load_data:v1"


def _clean_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.rstrip() for line in text.split("\n")]
    out = "\n".join(lines).strip()
    return out


def _base_meta(*, hf_dataset: str, hf_split: str, row_index: int) -> dict[str, Any]:
    return {
        "source": "huggingface",
        "hf_dataset": hf_dataset,
        "hf_split": hf_split,
        "hf_row_index": row_index,
        "loader": _LOADER_TAG,
    }


def load_gsm8k(
    *,
    splits: Sequence[str] = ("train", "test"),
    revision: str | None = None,
) -> list[CleanSample]:
    """Load requested GSM8K splits; ``answer_raw`` is the dataset ``answer`` field (incl. ``####`` tail)."""
    out: list[CleanSample] = []
    for split in splits:
        ds = load_dataset(
            GSM8K_REPO, GSM8K_CONFIG, split=split, revision=revision
        )
        out.extend(_gsm8k_split_to_samples(ds, split_name=split))
    return out


def _gsm8k_split_to_samples(ds: Dataset, *, split_name: str) -> list[CleanSample]:
    samples: list[CleanSample] = []
    for i, row in enumerate(ds):
        q = _clean_whitespace(str(row["question"]))
        a = _clean_whitespace(str(row["answer"]))
        sid = f"gsm8k:{split_name}:{i:05d}"
        meta = _base_meta(
            hf_dataset=GSM8K_REPO, hf_split=split_name, row_index=i
        )
        samples.append(
            CleanSample(
                dataset="gsm8k",
                sample_id=sid,
                question=q,
                answer_raw=a,
                answer_normalized="",
                split=split_name,
                metadata=meta,
            )
        )
    return samples


def load_math500(
    *,
    split: str = "test",
    revision: str | None = None,
) -> list[CleanSample]:
    """Load MATH-500; ``answer_raw`` is the full ``solution`` text; short ``answer`` is in metadata."""
    ds = load_dataset(MATH500_REPO, split=split, revision=revision)
    return _math500_split_to_samples(ds, split_name=split)


def _math500_split_to_samples(ds: Dataset, *, split_name: str) -> list[CleanSample]:
    samples: list[CleanSample] = []
    for i, row in enumerate(ds):
        problem = _clean_whitespace(str(row["problem"]))
        solution = _clean_whitespace(str(row["solution"]))
        boxed = _clean_whitespace(str(row["answer"]))
        uid = str(row["unique_id"])
        sid = f"math500:{uid}"
        meta = _base_meta(
            hf_dataset=MATH500_REPO, hf_split=split_name, row_index=i
        )
        meta["unique_id"] = uid
        meta["subject"] = row.get("subject")
        meta["level"] = row.get("level")
        meta["boxed_answer"] = boxed
        samples.append(
            CleanSample(
                dataset="math500",
                sample_id=sid,
                question=problem,
                answer_raw=solution,
                answer_normalized="",
                split=split_name,
                metadata=meta,
            )
        )
    return samples


def load_all(
    *,
    gsm8k_splits: Sequence[str] = ("train", "test"),
    math500_split: str = "test",
    revision_gsm8k: str | None = None,
    revision_math500: str | None = None,
) -> list[CleanSample]:
    """Load GSM8K (multiple splits) and MATH-500 (one split), concatenated in that order."""
    g = load_gsm8k(splits=gsm8k_splits, revision=revision_gsm8k)
    m = load_math500(split=math500_split, revision=revision_math500)
    return g + m
