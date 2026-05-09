#!/usr/bin/env python3
"""
Filter raw teacher outputs into clean teacher-generated supervision data.

This script reads raw outputs from 03_generate_teacher_rationales.py, checks
format and label consistency, and writes a simplified clean JSONL file that can
be converted into PPM SFT format.

Expected usage:
    python scripts/data/04_filter_teacher_outputs.py --config configs/data_config.yaml
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from tqdm import tqdm


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


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


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


def bool_from_any(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def verdict_to_bool(verdict: Optional[str]) -> Optional[bool]:
    if verdict is None:
        return None
    if verdict.lower() == "yes":
        return True
    if verdict.lower() == "no":
        return False
    return None


def first_nonempty(record: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for key in keys:
        val = record.get(key)
        if val is not None and str(val).strip() != "":
            return str(val)
    return default


def validate_record(
    rec: Dict[str, Any],
    *,
    require_status_ok: bool,
    require_final_verdict: bool,
    require_label_match: bool,
    min_rationale_chars: int,
    require_error_for_wrong: bool,
) -> Tuple[bool, str, Optional[str], Optional[str]]:
    if require_status_ok and rec.get("status") != "ok":
        return False, "status_not_ok", None, None

    answer_text = first_nonempty(rec, ["answer_text", "verification_rationale", "rationale"])
    if len(answer_text.strip()) < min_rationale_chars:
        return False, "rationale_too_short", None, None

    verdict = rec.get("parsed_final_verdict") or parse_final_verdict(answer_text)
    if verdict:
        verdict = str(verdict).capitalize()

    if require_final_verdict and verdict not in {"Yes", "No"}:
        return False, "missing_or_invalid_final_verdict", verdict, None

    gold_is_correct = bool_from_any(rec.get("candidate_is_correct", rec.get("is_correct")))
    pred_is_correct = verdict_to_bool(verdict)
    if require_label_match and gold_is_correct is not None and pred_is_correct is not None:
        if gold_is_correct != pred_is_correct:
            return False, "verdict_label_mismatch", verdict, None

    earliest_error = rec.get("earliest_error") or extract_earliest_error(answer_text)
    if require_error_for_wrong and gold_is_correct is False and not earliest_error:
        return False, "missing_earliest_error_for_wrong", verdict, None

    return True, "ok", verdict, earliest_error


def build_clean_record(rec: Dict[str, Any], verdict: Optional[str], earliest_error: Optional[str]) -> Dict[str, Any]:
    answer_text = first_nonempty(rec, ["answer_text", "verification_rationale", "rationale"])
    gold_is_correct = bool_from_any(rec.get("candidate_is_correct", rec.get("is_correct")))
    return {
        "id": first_nonempty(rec, ["row_id", "id", "sample_id"]),
        "source": "teacher_generated",
        "dataset": first_nonempty(rec, ["dataset", "source_dataset"], "openmathinstruct"),
        "question_norm": rec.get("question_norm"),
        "question": first_nonempty(rec, ["question", "problem", "prompt"]),
        "candidate_role": first_nonempty(rec, ["candidate_role", "role"], "unknown"),
        "candidate_solution": first_nonempty(
            rec,
            [
                "candidate_solution_clean",
                "candidate_solution",
                "candidate_solution_raw",
                "generated_solution_clean",
                "generated_solution",
                "solution",
            ],
        ),
        "expected_answer": first_nonempty(rec, ["expected_answer", "answer", "gold_answer", "final_answer"]),
        "candidate_predicted_answer": first_nonempty(rec, ["candidate_predicted_answer", "predicted_answer"], ""),
        "reference_solution": first_nonempty(
            rec,
            ["reference_solution_clean", "reference_solution", "reference_solution_raw", "correct_solution"],
        ),
        "verification_rationale": answer_text.strip(),
        "final_verdict": verdict,
        "is_correct": gold_is_correct,
        "earliest_error": earliest_error,
        "teacher_model": rec.get("teacher_model"),
        "run_id": rec.get("run_id"),
        "raw_row_id": rec.get("row_id"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data_config.yaml")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--sample-output", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    input_file = args.input or get_nested(
        cfg,
        ["teacher_filter", "input_file"],
        get_nested(cfg, ["teacher_generation", "output_raw_file"], "data/interim/teacher_outputs_all_candidates.jsonl"),
    )
    output_file = args.output or get_nested(
        cfg,
        ["teacher_filter", "output_file"],
        "data/processed/clean_teacher_data.jsonl",
    )
    sample_output = args.sample_output or get_nested(
        cfg,
        ["teacher_filter", "sample_output_file"],
        "data/examples/clean_teacher_sample.jsonl",
    )

    require_status_ok = bool(get_nested(cfg, ["teacher_filter", "require_status_ok"], True))
    require_final_verdict = bool(get_nested(cfg, ["teacher_filter", "require_final_verdict"], True))
    require_label_match = bool(get_nested(cfg, ["teacher_filter", "require_label_match"], True))
    min_rationale_chars = int(get_nested(cfg, ["teacher_filter", "min_rationale_chars"], 80))
    require_error_for_wrong = bool(get_nested(cfg, ["teacher_filter", "require_error_for_wrong"], False))

    raw_records = read_jsonl(input_file)
    clean_records: List[Dict[str, Any]] = []
    reject_counts: Dict[str, int] = {}

    for rec in tqdm(raw_records, desc="Filtering teacher outputs"):
        ok, reason, verdict, earliest_error = validate_record(
            rec,
            require_status_ok=require_status_ok,
            require_final_verdict=require_final_verdict,
            require_label_match=require_label_match,
            min_rationale_chars=min_rationale_chars,
            require_error_for_wrong=require_error_for_wrong,
        )
        if ok:
            clean_records.append(build_clean_record(rec, verdict, earliest_error))
        else:
            reject_counts[reason] = reject_counts.get(reason, 0) + 1

    write_jsonl(clean_records, output_file)
    write_jsonl(clean_records[: args.sample_size], sample_output)

    print("Filtering complete.")
    print(f"Raw records: {len(raw_records)}")
    print(f"Clean records: {len(clean_records)}")
    print(f"Rejected records: {len(raw_records) - len(clean_records)}")
    print("Reject reasons:")
    for key, val in sorted(reject_counts.items()):
        print(f"  {key}: {val}")
    print(f"Clean output: {output_file}")
    print(f"Sample output: {sample_output}")


if __name__ == "__main__":
    main()
