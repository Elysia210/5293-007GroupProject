#!/usr/bin/env python3
"""Direct-only completions (single ``n=1`` call per question) → new JSONL for ``eval_baselines.py --direct``.

Reads ``cleaned_samples.jsonl`` (or any JSONL with ``sample_id`` + ``question``). Does **not**
modify ``step1_dataset.jsonl``, ``candidate_paths.jsonl``, or other pipeline outputs.

Requires: ``pip install openai``, ``export OPENAI_API_KEY=...``

Example::

    python run_direct_model.py \\
      --input data/final/math500_k5_run/cleaned_samples.jsonl \\
      --out data/final/math500_k5_run/direct_live.jsonl \\
      --resume

Then::

    python eval_baselines.py \\
      --data data/final/math500_k5_run/step1_dataset.jsonl \\
      --direct data/final/math500_k5_run/direct_live.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Repo imports live under ``src/`` (same layout as ``python src/main_step1.py``).
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from generate_candidates import GenerationConfig, build_cot_messages  # noqa: E402
from openai_backend import OpenAIBackend  # noqa: E402

DEFAULT_DIRECT_SYSTEM = (
    "You solve math and STEM problems. Reason briefly if needed, stay concise, "
    "and follow the exact final-line format requested."
)


def build_direct_messages(question: str, *, system_prompt: str | None = None) -> list[dict[str, str]]:
    """Single-turn prompt with GSM8K-style ``####`` tail (parsable by ``extract_final_answer``)."""
    sys_p = (system_prompt if system_prompt is not None else DEFAULT_DIRECT_SYSTEM).strip()
    q = question.strip()
    user = (
        "Solve the problem. Keep reasoning short unless necessary.\n"
        "On the last line, write the final answer exactly in this form:\n"
        "#### <answer>\n\n"
        f"Problem:\n{q}"
    )
    return [{"role": "system", "content": sys_p}, {"role": "user", "content": user}]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            sid = o.get("sample_id")
            if sid:
                out.add(sid)
    return out


def _one(
    row: dict,
    backend: OpenAIBackend,
    config: GenerationConfig,
    prompt_mode: str,
) -> dict:
    q = row.get("question", "")
    if prompt_mode == "cot":
        messages = build_cot_messages(q, system_prompt=None)
    else:
        messages = build_direct_messages(q, system_prompt=None)
    paths = backend.generate_paths(messages, config=config, sample=None)
    p = paths[0]
    return {
        "sample_id": row["sample_id"],
        "direct_answer": p.predicted_answer_normalized or "",
        "direct_prediction_raw": p.predicted_answer_raw or "",
        "direct_response": p.reasoning_text or "",
        "_prompt_mode": prompt_mode,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run direct-style single completions to a new JSONL overlay.")
    ap.add_argument("--input", type=Path, required=True, help="cleaned_samples.jsonl (read-only)")
    ap.add_argument("--out", type=Path, required=True, help="New JSONL path (append + resume)")
    ap.add_argument("--prompt", choices=("direct", "cot"), default="direct", help="direct=concise; cot=same as Step-1 CoT prompt")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="Skip sample_ids already present in --out")
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--openai-base-url", default="", dest="base_url")
    ap.add_argument("--openai-top-logprobs", type=int, default=20, dest="top_logprobs")
    ap.add_argument("--openai-max-retries", type=int, default=3, dest="max_retries")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rows = _read_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    done = _done_ids(args.out) if args.resume else set()
    todo = [r for r in rows if r.get("sample_id") not in done]
    if not todo:
        print("Nothing to do (all samples done or empty list).", file=sys.stderr)
        return

    backend = OpenAIBackend(
        model=args.model,
        base_url=args.base_url or None,
        top_logprobs=max(0, min(20, args.top_logprobs)),
        max_retries=max(0, args.max_retries),
    )
    config = GenerationConfig(
        k=1,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    n_ok, n_err = 0, 0

    def task(row: dict) -> None:
        nonlocal n_ok, n_err
        try:
            rec = _one(row, backend, config, args.prompt)
            line = json.dumps(rec, ensure_ascii=False) + "\n"
            with lock:
                with args.out.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.flush()
                n_ok += 1
        except Exception as e:
            n_err += 1
            print(f"[error] {row.get('sample_id')!r}: {e}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futs = [ex.submit(task, r) for r in todo]
        for _ in as_completed(futs):
            pass

    print(f"Wrote (append) to {args.out} — ok={n_ok} errors={n_err} skipped_prior={len(done)}", file=sys.stderr)


if __name__ == "__main__":
    main()
