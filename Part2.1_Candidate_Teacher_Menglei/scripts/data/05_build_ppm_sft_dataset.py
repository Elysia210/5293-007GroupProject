#!/usr/bin/env python3
"""
Generate raw teacher verification rationales for selected candidate solutions.

This script reads a teacher-seed file, formats each candidate with a teacher
verification prompt, calls an OpenAI-compatible teacher API, and writes raw
teacher outputs to JSONL. It supports retry, rate-limit control, and checkpoint
resume.

Expected usage:
    python scripts/data/03_generate_teacher_rationales.py --config configs/data_config.yaml

Required local environment variables, usually stored in .env:
    NVIDIA_API_KEY=...
    NVIDIA_API_BASE=https://integrate.api.nvidia.com/v1
    TEACHER_MODEL=moonshotai/kimi-k2.5

Do NOT commit the real .env file to GitHub.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


# -----------------------------
# Basic file utilities
# -----------------------------

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_nested(cfg: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path, append: bool = False) -> None:
    ensure_parent(path)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_table(path: str | Path) -> List[Dict[str, Any]]:
    """Read JSONL or CSV teacher-seed file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input seed file does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError(f"JSON input must be a list of records: {path}")
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported input file type: {path.suffix}. Use .jsonl, .json, or .csv")


# -----------------------------
# Record formatting helpers
# -----------------------------

def stable_id(record: Dict[str, Any], index: int) -> str:
    raw = json.dumps(record, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return str(record.get("row_id") or record.get("id") or record.get("sample_id") or f"row_{index:06d}_{digest}")


def first_nonempty(record: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for key in keys:
        val = record.get(key)
        if val is not None and str(val).strip() != "":
            return str(val)
    return default


def parse_final_verdict(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"Verification:\s*Is\s+the\s+answer\s+correct\s*\(Yes/No\)\?\s*(Yes|No)",
        r"Is\s+the\s+answer\s+correct\s*\(Yes/No\)\?\s*(Yes|No)",
        r"Final\s+Verdict\s*:\s*(Yes|No)",
        r"Final\s+Answer\s*:\s*(Yes|No)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
    return None


def extract_earliest_error(text: str) -> Optional[str]:
    if not text:
        return None
    patterns = [
        r"Earliest\s+Error\s*:\s*(.+?)(?:\n\s*Verification:|\Z)",
        r"earliest\s+fatal\s+error\s*(?:is|:)?\s*(.+?)(?:\n\s*Verification:|\Z)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return " ".join(m.group(1).strip().split())
    return None


def format_prompt(template: str, record: Dict[str, Any]) -> str:
    values = {
        "question": first_nonempty(record, ["question", "problem", "prompt"]),
        "candidate_role": first_nonempty(record, ["candidate_role", "role"], "unknown"),
        "candidate_solution": first_nonempty(
            record,
            [
                "candidate_solution_clean",
                "candidate_solution",
                "generated_solution_clean",
                "generated_solution",
                "solution",
                "response",
            ],
        ),
        "reference_solution": first_nonempty(
            record,
            [
                "reference_solution_clean",
                "reference_solution",
                "correct_solution_clean",
                "correct_solution",
                "reference_answer",
            ],
        ),
        "expected_answer": first_nonempty(record, ["expected_answer", "answer", "gold_answer", "final_answer"]),
        "candidate_predicted_answer": first_nonempty(record, ["candidate_predicted_answer", "predicted_answer"], ""),
    }
    try:
        return template.format(**values)
    except KeyError as exc:
        missing = exc.args[0]
        raise KeyError(
            f"Prompt template expects {{{missing}}}, but the script does not provide it. "
            f"Available fields: {sorted(values.keys())}"
        )


# -----------------------------
# API calling
# -----------------------------

def make_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("TEACHER_API_KEY") or os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("NVIDIA_API_BASE") or os.getenv("TEACHER_API_BASE") or "https://integrate.api.nvidia.com/v1"
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set NVIDIA_API_KEY or TEACHER_API_KEY in your local .env file."
        )
    return OpenAI(api_key=api_key, base_url=api_base)


def call_teacher(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout_s,
    )
    choice = response.choices[0]
    message = choice.message
    answer_text = message.content or ""
    reasoning_text = getattr(message, "reasoning_content", None)
    usage = response.usage.model_dump() if getattr(response, "usage", None) else None
    return {
        "answer_text": answer_text,
        "reasoning_text": reasoning_text,
        "finish_reason": choice.finish_reason,
        "usage": usage,
    }


# -----------------------------
# Checkpoint helpers
# -----------------------------

def load_checkpoint(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("processed_row_ids", []))
    except Exception:
        return set()


def save_checkpoint(path: str | Path, processed: set[str]) -> None:
    ensure_parent(path)
    tmp_path = Path(str(path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"processed_row_ids": sorted(processed)}, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    parser.add_argument("--input", type=str, default=None, help="Optional override for teacher seed file")
    parser.add_argument("--output", type=str, default=None, help="Optional override for raw teacher output JSONL")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    input_file = args.input or get_nested(cfg, ["teacher_generation", "input_file"], None)
    if not input_file:
        input_file = get_nested(cfg, ["teacher_seed", "output_file"], "data/interim/teacher_seed_full.jsonl")

    output_file = args.output or get_nested(
        cfg,
        ["teacher_generation", "output_raw_file"],
        "data/interim/teacher_outputs_all_candidates.jsonl",
    )
    error_file = get_nested(
        cfg,
        ["teacher_generation", "output_error_file"],
        "data/interim/teacher_outputs_errors.jsonl",
    )
    checkpoint_file = get_nested(
        cfg,
        ["teacher_generation", "checkpoint_file"],
        "data/interim/teacher_outputs_checkpoint.json",
    )
    prompt_file = get_nested(
        cfg,
        ["teacher_generation", "prompt_file"],
        "prompts/teacher_verifier_prompt.txt",
    )

    model = os.getenv("TEACHER_MODEL") or get_nested(
        cfg, ["teacher_generation", "model"], "moonshotai/kimi-k2.5"
    )
    temperature = float(get_nested(cfg, ["teacher_generation", "temperature"], 0.2))
    max_tokens = int(get_nested(cfg, ["teacher_generation", "max_tokens"], 1024))
    max_retries = int(get_nested(cfg, ["teacher_generation", "max_retries"], 3))
    timeout_s = int(get_nested(cfg, ["teacher_generation", "timeout_s"], 120))
    rate_limit_rpm = float(get_nested(cfg, ["teacher_generation", "rate_limit_rpm"], 40))
    sleep_per_request = 60.0 / rate_limit_rpm if rate_limit_rpm > 0 else 0.0
    run_id = get_nested(cfg, ["teacher_generation", "run_id"], time.strftime("teacher_%Y%m%d_%H%M%S"))

    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    records = read_table(input_file)
    if args.limit:
        records = records[: args.limit]

    client = make_client()
    processed = load_checkpoint(checkpoint_file)

    ensure_parent(output_file)
    ensure_parent(error_file)

    print(f"Input seed file: {input_file}")
    print(f"Output raw teacher file: {output_file}")
    print(f"Error file: {error_file}")
    print(f"Checkpoint file: {checkpoint_file}")
    print(f"Teacher model: {model}")
    print(f"Already processed rows: {len(processed)}")

    for idx, rec in enumerate(tqdm(records, desc="Generating teacher rationales")):
        row_id = stable_id(rec, idx)
        if row_id in processed:
            continue

        prompt = format_prompt(prompt_template, rec)
        base_out = {
            "status": None,
            "run_id": run_id,
            "row_id": row_id,
            "question_norm": rec.get("question_norm"),
            "question": first_nonempty(rec, ["question", "problem", "prompt"]),
            "candidate_role": first_nonempty(rec, ["candidate_role", "role"], "unknown"),
            "candidate_is_correct": rec.get("candidate_is_correct", rec.get("is_correct")),
            "expected_answer": first_nonempty(rec, ["expected_answer", "answer", "gold_answer", "final_answer"]),
            "candidate_predicted_answer": first_nonempty(rec, ["candidate_predicted_answer", "predicted_answer"], ""),
            "candidate_solution_raw": first_nonempty(rec, ["candidate_solution_raw", "generated_solution", "candidate_solution", "solution"]),
            "candidate_solution_clean": first_nonempty(
                rec,
                ["candidate_solution_clean", "candidate_solution", "generated_solution_clean", "generated_solution", "solution"],
            ),
            "reference_solution_raw": first_nonempty(rec, ["reference_solution_raw", "reference_solution", "correct_solution"]),
            "reference_solution_clean": first_nonempty(
                rec,
                ["reference_solution_clean", "reference_solution", "correct_solution_clean", "correct_solution"],
            ),
            "teacher_model": model,
            "teacher_prompt": prompt,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        last_error: Optional[str] = None
        for retry in range(max_retries + 1):
            try:
                api_out = call_teacher(client, model, prompt, temperature, max_tokens, timeout_s)
                answer_text = api_out.get("answer_text") or ""
                final_verdict = parse_final_verdict(answer_text)
                earliest_error = extract_earliest_error(answer_text)
                out = {
                    **base_out,
                    **api_out,
                    "status": "ok",
                    "parsed_final_verdict": final_verdict,
                    "earliest_error": earliest_error,
                    "retry_count": retry,
                }
                write_jsonl([out], output_file, append=True)
                processed.add(row_id)
                save_checkpoint(checkpoint_file, processed)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
                if retry < max_retries:
                    time.sleep(min(2 ** retry, 30))
                else:
                    err = {
                        **base_out,
                        "status": "error",
                        "error": last_error,
                        "retry_count": retry,
                    }
                    write_jsonl([err], error_file, append=True)
                    processed.add(row_id)
                    save_checkpoint(checkpoint_file, processed)

        if sleep_per_request > 0:
            time.sleep(sleep_per_request)

    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise
