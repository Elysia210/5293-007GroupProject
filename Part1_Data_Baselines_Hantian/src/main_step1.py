#!/usr/bin/env python3
"""Step 1 orchestration: load → normalize → generate → entropy → segment → QC → JSONL."""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, TextIO

from extract_entropy import build_token_features, entropy_from_probs
from generate_candidates import EchoBackend, GenerationConfig, generate_candidates
from normalize_answers import extract_final_answer, normalize_answer
from qc_checks import QCReport, print_qc_summary, run_qc
from schemas import CandidatePath, CleanSample, StepFeature, TokenFeature
from segment_steps import segment_candidate


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


OUTPUT_FILES = (
    "cleaned_samples.jsonl",
    "candidate_paths.jsonl",
    "entropy_features.jsonl",
    "step1_dataset.jsonl",
)


def json_safe(obj: Any) -> Any:
    """Make structures JSON-serializable (NaN/Inf → null)."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Overwrite a JSONL file (batch helper; incremental path uses ``IncrementalWriter``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def load_completed_sample_ids(step1_path: Path) -> set[str]:
    """Ledger of fully finished samples (one ``step1_dataset`` line each)."""
    if not step1_path.is_file():
        return set()
    done: set[str] = set()
    with step1_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("sample_id")
            if isinstance(sid, str) and sid:
                done.add(sid)
    return done


def _float_field(x: Any, default: float = float("nan")) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def clean_sample_from_step1_dict(d: dict[str, Any]) -> CleanSample:
    """Rebuild ``CleanSample`` from a ``step1_dataset.jsonl`` row (for QC)."""
    candidates: list[CandidatePath] = []
    for c in d.get("candidates", []):
        if not isinstance(c, dict):
            continue
        tokens = [
            TokenFeature(
                token_index=int(t["token_index"]),
                logprob=_float_field(t.get("logprob")),
                entropy=_float_field(t.get("entropy")),
            )
            for t in c.get("tokens", [])
            if isinstance(t, dict)
        ]
        steps = [
            StepFeature(
                start_token=int(s["start_token"]),
                end_token=int(s["end_token"]),
                mean_entropy=_float_field(s.get("mean_entropy")),
                max_entropy=_float_field(s.get("max_entropy")),
                min_entropy=_float_field(s.get("min_entropy")),
                std_entropy=_float_field(s.get("std_entropy")),
            )
            for s in c.get("steps", [])
            if isinstance(s, dict)
        ]
        candidates.append(
            CandidatePath(
                reasoning_text=str(c.get("reasoning_text", "")),
                predicted_answer_raw=str(c.get("predicted_answer_raw", "")),
                predicted_answer_normalized=str(
                    c.get("predicted_answer_normalized", "")
                ),
                tokens=tokens,
                steps=steps,
            )
        )
    meta = d.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    return CleanSample(
        dataset=str(d["dataset"]),
        sample_id=str(d["sample_id"]),
        question=str(d.get("question", "")),
        answer_raw=str(d.get("answer_raw", "")),
        answer_normalized=str(d.get("answer_normalized", "")),
        split=d.get("split"),
        metadata=meta,
        candidates=candidates,
    )


def load_samples_from_step1_jsonl(path: Path) -> list[CleanSample]:
    if not path.is_file():
        return []
    out: list[CleanSample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if isinstance(d, dict):
                out.append(clean_sample_from_step1_dict(d))
    return out


class IncrementalWriter:
    """Thread-safe append + flush for Step 1 JSONL outputs."""

    def __init__(self, out_dir: Path, write_lock: threading.Lock):
        self.out_dir = out_dir
        self._lock = write_lock
        out_dir.mkdir(parents=True, exist_ok=True)
        self._cleaned: TextIO | None = None
        self._candidate: TextIO | None = None
        self._entropy: TextIO | None = None
        self._step1: TextIO | None = None

    def open_files_append(self) -> None:
        self._cleaned = (self.out_dir / "cleaned_samples.jsonl").open(
            "a", encoding="utf-8"
        )
        self._candidate = (self.out_dir / "candidate_paths.jsonl").open(
            "a", encoding="utf-8"
        )
        self._entropy = (self.out_dir / "entropy_features.jsonl").open(
            "a", encoding="utf-8"
        )
        self._step1 = (self.out_dir / "step1_dataset.jsonl").open(
            "a", encoding="utf-8"
        )

    def close(self) -> None:
        for f in (self._cleaned, self._candidate, self._entropy, self._step1):
            if f is not None:
                f.close()
        self._cleaned = self._candidate = self._entropy = self._step1 = None

    def append_one_sample(self, s: CleanSample) -> None:
        """Append all rows for one finished sample; ``step1`` line is written last."""
        with self._lock:
            assert self._cleaned and self._candidate and self._entropy and self._step1
            d_clean = asdict(s)
            d_clean.pop("candidates", None)
            self._cleaned.write(
                json.dumps(json_safe(d_clean), ensure_ascii=False) + "\n"
            )
            self._cleaned.flush()

            for i, c in enumerate(s.candidates):
                row = {
                    "sample_id": s.sample_id,
                    "dataset": s.dataset,
                    "split": s.split,
                    "candidate_index": i,
                    "reasoning_text": c.reasoning_text,
                    "predicted_answer_raw": c.predicted_answer_raw,
                    "predicted_answer_normalized": c.predicted_answer_normalized,
                }
                self._candidate.write(
                    json.dumps(json_safe(row), ensure_ascii=False) + "\n"
                )
            self._candidate.flush()

            for ci, c in enumerate(s.candidates):
                for tf in c.tokens:
                    erow = {
                        "sample_id": s.sample_id,
                        "dataset": s.dataset,
                        "split": s.split,
                        "candidate_index": ci,
                        "token_index": tf.token_index,
                        "logprob": tf.logprob,
                        "entropy": tf.entropy,
                    }
                    self._entropy.write(
                        json.dumps(json_safe(erow), ensure_ascii=False) + "\n"
                    )
            self._entropy.flush()

            self._step1.write(
                json.dumps(json_safe(asdict(s)), ensure_ascii=False) + "\n"
            )
            self._step1.flush()


def truncate_output_jsonl(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:
        (out_dir / name).write_text("", encoding="utf-8")


def normalize_gold_answers(samples: list[CleanSample]) -> list[CleanSample]:
    out: list[CleanSample] = []
    for s in samples:
        norm = normalize_answer(extract_final_answer(s.answer_raw))
        out.append(replace(s, answer_normalized=norm))
    return out


def _candidate_word_entropy(candidate: CandidatePath) -> CandidatePath:
    """Attach one ``TokenFeature`` per whitespace token using uniform-over-chars entropy."""
    words = candidate.reasoning_text.split()
    logprobs: list[float] = []
    entropies: list[float] = []
    for w in words:
        n = len(w)
        if n <= 0:
            continue
        probs = [1.0 / n] * n
        entropies.append(entropy_from_probs(probs))
        logprobs.append(math.log(1.0 / n))
    tokens = build_token_features(logprobs, entropies)
    return replace(candidate, tokens=tokens)


def extract_entropy_for_candidate(
    candidate: CandidatePath, mode: str
) -> CandidatePath:
    if mode == "none":
        # Keep model-provided tokens (e.g. OpenAI logprobs); do not overwrite.
        return candidate
    if mode == "per_word_uniform":
        return _candidate_word_entropy(candidate)
    raise ValueError(f"unknown entropy mode: {mode!r}")


def segment_steps_for_sample(sample: CleanSample) -> CleanSample:
    new_cands = [segment_candidate(c) for c in sample.candidates]
    return replace(sample, candidates=new_cands)


def process_one_sample(
    sample: CleanSample,
    gen_config: GenerationConfig,
    backend: Any,
    entropy_mode: str,
) -> CleanSample:
    """Normalize gold, generate, entropy, segment — for one sample (thread-safe)."""
    norm = normalize_answer(extract_final_answer(sample.answer_raw))
    s = replace(sample, answer_normalized=norm)
    if gen_config.k > 0:
        s = generate_candidates(s, backend, gen_config, mutate=False)
    else:
        s = replace(s, candidates=[])
    new_cands = [extract_entropy_for_candidate(c, entropy_mode) for c in s.candidates]
    s = replace(s, candidates=new_cands)
    s = segment_steps_for_sample(s)
    return s


def load_samples_for_args(args: argparse.Namespace) -> list[CleanSample]:
    from load_data import load_all, load_gsm8k, load_math500

    if args.datasets == "gsm8k":
        return load_gsm8k(
            splits=tuple(args.gsm8k_splits),
            revision=args.revision_gsm8k,
        )
    if args.datasets == "math500":
        return load_math500(
            split=args.math500_split,
            revision=args.revision_math500,
        )
    return load_all(
        gsm8k_splits=tuple(args.gsm8k_splits),
        math500_split=args.math500_split,
        revision_gsm8k=args.revision_gsm8k,
        revision_math500=args.revision_math500,
    )


def make_backend(name: str, args: argparse.Namespace):
    if name == "echo":
        return EchoBackend()
    if name == "openai":
        from openai_backend import OpenAIBackend

        return OpenAIBackend(
            model=args.openai_model,
            base_url=args.openai_base_url or None,
            top_logprobs=args.openai_top_logprobs,
            max_retries=args.openai_max_retries,
        )
    raise ValueError(f"unknown backend: {name!r}")


def export_cleaned_rows(samples: list[CleanSample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in samples:
        d = asdict(s)
        d.pop("candidates", None)
        rows.append(d)
    return rows


def export_candidate_path_rows(samples: list[CleanSample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in samples:
        for i, c in enumerate(s.candidates):
            rows.append(
                {
                    "sample_id": s.sample_id,
                    "dataset": s.dataset,
                    "split": s.split,
                    "candidate_index": i,
                    "reasoning_text": c.reasoning_text,
                    "predicted_answer_raw": c.predicted_answer_raw,
                    "predicted_answer_normalized": c.predicted_answer_normalized,
                }
            )
    return rows


def export_entropy_rows(samples: list[CleanSample]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s in samples:
        for ci, c in enumerate(s.candidates):
            for tf in c.tokens:
                rows.append(
                    {
                        "sample_id": s.sample_id,
                        "dataset": s.dataset,
                        "split": s.split,
                        "candidate_index": ci,
                        "token_index": tf.token_index,
                        "logprob": tf.logprob,
                        "entropy": tf.entropy,
                    }
                )
    return rows


def export_step1_dataset_rows(samples: list[CleanSample]) -> list[dict[str, Any]]:
    return [asdict(s) for s in samples]


def export_all(samples: list[CleanSample], out_dir: Path) -> None:
    write_jsonl(out_dir / "cleaned_samples.jsonl", export_cleaned_rows(samples))
    write_jsonl(out_dir / "candidate_paths.jsonl", export_candidate_path_rows(samples))
    write_jsonl(out_dir / "entropy_features.jsonl", export_entropy_rows(samples))
    write_jsonl(out_dir / "step1_dataset.jsonl", export_step1_dataset_rows(samples))


def _namespace_for_manifest(ns: argparse.Namespace) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k, v in vars(ns).items():
        if isinstance(v, Path):
            d[k] = str(v)
        else:
            d[k] = v
    return d


def write_run_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    *,
    n_samples_total_in_step1: int,
    n_processed_this_run: int,
    n_skipped_resume: int,
    n_errors_this_run: int,
) -> None:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir.resolve()),
        "n_samples_total_in_step1": n_samples_total_in_step1,
        "n_processed_ok_this_run": n_processed_this_run,
        "n_skipped_resume": n_skipped_resume,
        "n_errors_this_run": n_errors_this_run,
        "args": _namespace_for_manifest(args),
    }
    path = out_dir / "run_manifest.json"
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 1: data prep → JSONL exports.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <project>/data/final)",
    )
    p.add_argument(
        "--datasets",
        choices=("all", "gsm8k", "math500"),
        default="all",
        help="Which sources to load",
    )
    p.add_argument(
        "--gsm8k-splits",
        nargs="+",
        default=["train", "test"],
        help="GSM8K splits when gsm8k or all is selected",
    )
    p.add_argument(
        "--math500-split",
        default="test",
        help="MATH-500 split name",
    )
    p.add_argument("--revision-gsm8k", default=None, help="HF dataset revision for GSM8K")
    p.add_argument(
        "--revision-math500",
        default=None,
        help="HF dataset revision for MATH-500",
    )
    p.add_argument("--limit", type=int, default=None, help="Keep only the first N samples")
    p.add_argument("-k", "--num-candidates", type=int, default=4, dest="k")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95, dest="top_p")
    p.add_argument("--max-tokens", type=int, default=2048, dest="max_tokens")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--backend",
        choices=("echo", "openai"),
        default="echo",
        help="echo=stub paths; openai=real completions + logprob-based tokens (needs OPENAI_API_KEY)",
    )
    p.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="Chat model id when --backend openai",
    )
    p.add_argument(
        "--openai-base-url",
        default="",
        help="Optional OpenAI-compatible API base URL (e.g. Azure proxy)",
    )
    p.add_argument(
        "--openai-top-logprobs",
        type=int,
        default=20,
        help="0–20; predictive entropy is computed from these top alternatives per token",
    )
    p.add_argument(
        "--openai-max-retries",
        type=int,
        default=3,
        help="API retries (exponential backoff) for --backend openai",
    )
    p.add_argument(
        "--entropy-mode",
        choices=("none", "per_word_uniform"),
        default="none",
        help="none=keep model tokens only; per_word_uniform=overwrite with char-level heuristic",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Append to outputs; skip sample_ids already present in step1_dataset.jsonl",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Thread pool size for parallel per-sample processing",
    )
    p.add_argument("--skip-qc-print", action="store_true", help="Do not print QC summary")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if QC reports any errors",
    )
    return p.parse_args(argv)


def _format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or math.isinf(seconds):
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s}s"


def _progress_line(
    *,
    done: int,
    total: int,
    errors: int,
    elapsed: float,
) -> str:
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - done)
    eta = (remaining / rate) if rate > 0 else None
    return (
        f"[progress] {done}/{total} completed, errors={errors}, "
        f"elapsed={elapsed:.1f}s, ETA={_format_eta(eta)}"
    )


def main(argv: list[str] | None = None) -> QCReport:
    args = parse_args(argv)
    out_dir = args.out_dir or (_project_root() / "data" / "final")
    step1_path = out_dir / "step1_dataset.jsonl"

    samples = load_samples_for_args(args)
    if args.limit is not None:
        samples = samples[: max(0, args.limit)]

    gen_config = GenerationConfig(
        k=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    backend = make_backend(args.backend, args)

    completed_before: set[str] = set()
    if args.resume:
        completed_before = load_completed_sample_ids(step1_path)
    else:
        truncate_output_jsonl(out_dir)

    pending = [s for s in samples if s.sample_id not in completed_before]
    n_skipped = len(samples) - len(pending)

    write_lock = threading.Lock()
    writer = IncrementalWriter(out_dir, write_lock)
    writer.open_files_append()

    n_ok = 0
    n_err = 0
    start = time.monotonic()
    last_log = start

    print(
        f"Loaded {len(samples)} samples; pending={len(pending)}, "
        f"skipped(resume)={n_skipped}, max_workers={args.max_workers}",
        file=sys.stderr,
    )

    def submit_all(ex: ThreadPoolExecutor) -> dict[Any, str]:
        return {
            ex.submit(
                process_one_sample,
                s,
                gen_config,
                backend,
                args.entropy_mode,
            ): s.sample_id
            for s in pending
        }

    try:
        try:
            with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
                future_to_sid = submit_all(ex)
                while future_to_sid:
                    done_set, _ = wait(
                        list(future_to_sid.keys()),
                        timeout=0.5,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done_set:
                        continue
                    for fut in done_set:
                        sid = future_to_sid.pop(fut)
                        try:
                            enriched = fut.result()
                            writer.append_one_sample(enriched)
                            n_ok += 1
                        except Exception as e:
                            n_err += 1
                            print(
                                f"[error] sample_id={sid!r}: {e}",
                                file=sys.stderr,
                            )
                        now = time.monotonic()
                        if now - last_log >= 2.0 or not future_to_sid:
                            elapsed = now - start
                            done_tasks = n_ok + n_err
                            print(
                                _progress_line(
                                    done=done_tasks,
                                    total=len(pending),
                                    errors=n_err,
                                    elapsed=elapsed,
                                ),
                                file=sys.stderr,
                            )
                            last_log = now
        except KeyboardInterrupt:
            print(
                "\n[interrupt] stopping; in-flight samples may not have a step1 line yet "
                "(ledger is committed last per sample).",
                file=sys.stderr,
            )
            raise
    finally:
        writer.close()
        total_in_step1 = len(load_completed_sample_ids(step1_path))
        write_run_manifest(
            out_dir,
            args,
            n_samples_total_in_step1=total_in_step1,
            n_processed_this_run=n_ok,
            n_skipped_resume=n_skipped,
            n_errors_this_run=n_err,
        )
        print(
            f"Finished: ok_this_run={n_ok}, errors={n_err}, "
            f"total_lines_in_step1={total_in_step1}. Wrote run_manifest.json → {out_dir}",
            file=sys.stderr,
        )

    all_for_qc = load_samples_from_step1_jsonl(step1_path)
    report = run_qc(all_for_qc)
    if not args.skip_qc_print:
        print_qc_summary(report)

    if args.strict and report.summary.get("n_errors", 0):
        sys.exit(1)
    return report


if __name__ == "__main__":
    main()
