"""Quality checks for ``CleanSample`` lists (Step 1 pipeline)."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

from schemas import CandidatePath, CleanSample, TokenFeature

__all__ = ["QCIssue", "QCReport", "run_qc", "print_qc_summary"]

Severity = Literal["error", "warning"]


@dataclass
class QCIssue:
    check: str
    severity: Severity
    sample_id: str
    message: str
    candidate_index: int | None = None


@dataclass
class QCReport:
    issues: list[QCIssue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _blank(s: str | None) -> bool:
    return s is None or not str(s).strip()


def _bad_float(x: float) -> bool:
    return not isinstance(x, (int, float)) or not math.isfinite(float(x))


def _check_sample_fields(s: CleanSample, issues: list[QCIssue]) -> None:
    sid = s.sample_id or "<missing>"
    if _blank(s.sample_id):
        issues.append(
            QCIssue("missing_fields", "error", sid, "sample_id is missing or empty")
        )
    if _blank(s.dataset):
        issues.append(
            QCIssue("missing_fields", "error", sid, "dataset is missing or empty")
        )
    if _blank(s.question):
        issues.append(
            QCIssue("missing_fields", "warning", sid, "question is empty")
        )
    if s.answer_raw is None:
        issues.append(
            QCIssue("missing_fields", "warning", sid, "answer_raw is None")
        )
    elif _blank(s.answer_raw):
        issues.append(
            QCIssue("missing_fields", "warning", sid, "answer_raw is empty")
        )


def _check_candidate_empty(s: CleanSample, c: CandidatePath, k: int, issues: list[QCIssue]) -> None:
    sid = s.sample_id or "<missing>"
    if _blank(c.reasoning_text):
        issues.append(
            QCIssue(
                "empty_outputs",
                "warning",
                sid,
                "candidate has empty reasoning_text",
                candidate_index=k,
            )
        )
    if _blank(c.predicted_answer_raw) and not _blank(c.reasoning_text):
        issues.append(
            QCIssue(
                "empty_outputs",
                "warning",
                sid,
                "non-empty reasoning but predicted_answer_raw is empty",
                candidate_index=k,
            )
        )


def _check_tokens_nan(
    s: CleanSample, toks: list[TokenFeature], k: int, issues: list[QCIssue]
) -> None:
    sid = s.sample_id or "<missing>"
    bad_idx: list[int] = []
    for t in toks:
        if (
            math.isnan(t.logprob)
            or math.isnan(t.entropy)
            or math.isinf(t.logprob)
            or math.isinf(t.entropy)
        ):
            bad_idx.append(t.token_index)
    if bad_idx:
        issues.append(
            QCIssue(
                "nan_values",
                "warning",
                sid,
                f"non-finite token logprob/entropy at indices {bad_idx[:8]}"
                + (" …" if len(bad_idx) > 8 else ""),
                candidate_index=k,
            )
        )


def _check_token_alignment(
    s: CleanSample, toks: list[TokenFeature], k: int, issues: list[QCIssue]
) -> None:
    sid = s.sample_id or "<missing>"
    if not toks:
        return
    n = len(toks)
    for i, t in enumerate(toks):
        if t.token_index != i:
            issues.append(
                QCIssue(
                    "entropy_mismatch",
                    "warning",
                    sid,
                    f"token_index {t.token_index} at position {i} (expected contiguous 0..{n - 1})",
                    candidate_index=k,
                )
            )
            break
    seen: set[int] = set()
    for t in toks:
        if t.token_index in seen:
            issues.append(
                QCIssue(
                    "entropy_mismatch",
                    "warning",
                    sid,
                    f"duplicate token_index {t.token_index}",
                    candidate_index=k,
                )
            )
            break
        seen.add(t.token_index)


def _check_steps(
    s: CleanSample,
    c: CandidatePath,
    k: int,
    issues: list[QCIssue],
) -> None:
    sid = s.sample_id or "<missing>"
    toks = c.tokens
    n_tok = len(toks)
    for si, st in enumerate(c.steps):
        if any(
            math.isnan(x)
            for x in (
                st.mean_entropy,
                st.max_entropy,
                st.min_entropy,
                st.std_entropy,
            )
        ):
            issues.append(
                QCIssue(
                    "nan_values",
                    "warning",
                    sid,
                    f"NaN in step[{si}] entropy stats",
                    candidate_index=k,
                )
            )
        if n_tok == 0 and c.steps:
            issues.append(
                QCIssue(
                    "entropy_mismatch",
                    "warning",
                    sid,
                    f"step[{si}] present but tokens list is empty",
                    candidate_index=k,
                )
            )
            continue
        if st.start_token < 0 or st.end_token < 0:
            issues.append(
                QCIssue(
                    "entropy_mismatch",
                    "warning",
                    sid,
                    f"step[{si}] has negative token span",
                    candidate_index=k,
                )
            )
        if st.end_token < st.start_token:
            issues.append(
                QCIssue(
                    "entropy_mismatch",
                    "warning",
                    sid,
                    f"step[{si}] end_token < start_token",
                    candidate_index=k,
                )
            )
        if n_tok and st.end_token > n_tok:
            issues.append(
                QCIssue(
                    "entropy_mismatch",
                    "warning",
                    sid,
                    f"step[{si}] end_token {st.end_token} > len(tokens)={n_tok}",
                    candidate_index=k,
                )
            )
        if n_tok and st.start_token > n_tok:
            issues.append(
                QCIssue(
                    "entropy_mismatch",
                    "warning",
                    sid,
                    f"step[{si}] start_token {st.start_token} > len(tokens)={n_tok}",
                    candidate_index=k,
                )
            )


def _check_duplicate_ids(samples: list[CleanSample], issues: list[QCIssue]) -> None:
    counts = Counter(s.sample_id for s in samples if not _blank(s.sample_id))
    for sid, c in counts.items():
        if c > 1:
            issues.append(
                QCIssue(
                    "duplicate_ids",
                    "error",
                    sid,
                    f"sample_id appears {c} times",
                )
            )


def _build_summary(samples: list[CleanSample], issues: list[QCIssue]) -> dict[str, Any]:
    by_check: Counter[str] = Counter()
    by_sev: Counter[str] = Counter()
    for it in issues:
        by_check[it.check] += 1
        by_sev[it.severity] += 1

    n_cand = sum(len(s.candidates) for s in samples)
    ds_counts = Counter(s.dataset for s in samples if not _blank(s.dataset))

    return {
        "n_samples": len(samples),
        "n_candidates": n_cand,
        "n_issues": len(issues),
        "n_errors": by_sev["error"],
        "n_warnings": by_sev["warning"],
        "issues_by_check": dict(by_check),
        "samples_by_dataset": dict(ds_counts),
        "avg_candidates_per_sample": (n_cand / len(samples)) if samples else 0.0,
    }


def run_qc(samples: list[CleanSample]) -> QCReport:
    issues: list[QCIssue] = []

    _check_duplicate_ids(samples, issues)

    for s in samples:
        _check_sample_fields(s, issues)
        for k, c in enumerate(s.candidates):
            _check_candidate_empty(s, c, k, issues)
            _check_tokens_nan(s, c.tokens, k, issues)
            _check_token_alignment(s, c.tokens, k, issues)
            _check_steps(s, c, k, issues)

    summary = _build_summary(samples, issues)
    return QCReport(issues=issues, summary=summary)


def print_qc_summary(report: QCReport) -> None:
    """Print human-readable QC summary and issue counts."""
    s = report.summary
    print("=== QC summary ===")
    print(f"samples: {s.get('n_samples', 0)}")
    print(f"candidates: {s.get('n_candidates', 0)}")
    print(f"avg candidates / sample: {s.get('avg_candidates_per_sample', 0):.3f}")
    ds = s.get("samples_by_dataset") or {}
    if ds:
        print("by dataset:", ", ".join(f"{k}={v}" for k, v in sorted(ds.items())))
    print(f"issues: {s.get('n_issues', 0)} "
          f"(errors={s.get('n_errors', 0)}, warnings={s.get('n_warnings', 0)})")
    by_chk = s.get("issues_by_check") or {}
    if by_chk:
        print("by check:", ", ".join(f"{k}={v}" for k, v in sorted(by_chk.items())))
    if report.issues:
        print("--- first issues (up to 20) ---")
        for it in report.issues[:20]:
            loc = (
                f" cand={it.candidate_index}"
                if it.candidate_index is not None
                else ""
            )
            print(
                f"[{it.severity.upper()}] {it.check}{loc} "
                f"sample_id={it.sample_id!r} — {it.message}"
            )
        if len(report.issues) > 20:
            print(f"... ({len(report.issues) - 20} more)")
