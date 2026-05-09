#!/usr/bin/env python
"""
Generate verifier outputs and compute basic metrics.

Metrics:
- parse_rate: percentage of outputs with a parsed Yes/No verdict
- format_validity: percentage of outputs with the required final line format
- verdict_accuracy: percentage of parsed predictions matching gold verdict
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str, records: List[Dict[str, Any]]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_gold_verdict(record: Dict[str, Any]) -> Optional[str]:
    metadata = record.get("metadata", {}) or {}
    for key in ["gold_verdict", "parsed_final_verdict"]:
        if key in record and record[key] in ["Yes", "No"]:
            return record[key]
        if key in metadata and metadata[key] in ["Yes", "No"]:
            return metadata[key]
    for key in ["gold_is_correct", "is_correct", "candidate_is_correct"]:
        if key in record:
            return "Yes" if bool(record[key]) else "No"
        if key in metadata:
            return "Yes" if bool(metadata[key]) else "No"
    return None


def parse_verdict(text: str) -> Optional[str]:
    patterns = [
        r"Verification:\s*Is the answer correct\s*\(Yes/No\)\?\s*(Yes|No)",
        r"Is the answer correct\s*\(Yes/No\)\?\s*(Yes|No)",
        r"Final\s*(?:Answer|Grade|Verdict)?\s*[:\-]?\s*(Yes|No)",
    ]
    for p in patterns:
        m = re.search(p, text or "", flags=re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
    matches = re.findall(r"\b(Yes|No)\b", text or "", flags=re.IGNORECASE)
    return matches[-1].capitalize() if matches else None


def has_required_format(text: str) -> bool:
    return bool(re.search(r"Verification:\s*Is the answer correct", text or "", flags=re.IGNORECASE))


def build_user_content(record):
    if isinstance(record.get("messages"), list):
        return record["messages"][0]["content"]

    question = record.get("question") or record.get("problem") or ""
    candidate = (
        record.get("candidate_solution")
        or record.get("generated_solution")
        or record.get("solution")
        or ""
    )
    return (
        "Question:\n"
        f"{question}\n\n"
        "Candidate Solution:\n"
        f"{candidate}\n\n"
        "Please verify the candidate solution step by step. "
        "End with: Verification: Is the answer correct (Yes/No)? X"
    )


def render_prompt(tokenizer, user_content):
    messages = [{"role": "user", "content": user_content}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"### User:\n{user_content}\n\n### Assistant:\n"


def load_model(cfg, model_path):
    base_model = cfg["model"]["base_model"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path if (Path(model_path) / "tokenizer_config.json").exists() else base_model,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.get("training", {}).get("bf16", False) and torch.cuda.is_available() else torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if (Path(model_path) / "adapter_config.json").exists():
        if PeftModel is None:
            raise ImportError("Install peft to load LoRA adapters.")
        model = PeftModel.from_pretrained(model, model_path)
    elif Path(model_path).exists():
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=cfg["model"].get("trust_remote_code", True),
                device_map="auto" if torch.cuda.is_available() else None,
            )
        except Exception:
            pass

    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    records = read_jsonl(args.eval_file)
    if args.max_examples is not None:
        records = records[: args.max_examples]

    model, tokenizer = load_model(cfg, args.model_path)

    eval_cfg = cfg.get("evaluation", {})
    max_new_tokens = eval_cfg.get("max_new_tokens", 512)
    do_sample = eval_cfg.get("do_sample", False)
    temperature = eval_cfg.get("temperature", 0.0)

    outputs = []
    parsed = 0
    format_ok = 0
    total_gold = 0
    correct = 0

    for i, record in enumerate(tqdm(records, desc="Evaluating verifier")):
        user_content = build_user_content(record)
        prompt = render_prompt(tokenizer, user_content)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        output_ids = gen[0][inputs["input_ids"].shape[1]:]
        pred_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        pred = parse_verdict(pred_text)
        gold = get_gold_verdict(record)

        parsed += int(pred is not None)
        format_ok += int(has_required_format(pred_text))
        if gold is not None:
            total_gold += 1
            correct += int(pred == gold)

        outputs.append({
            "index": i,
            "gold_verdict": gold,
            "pred_verdict": pred,
            "format_ok": has_required_format(pred_text),
            "prediction": pred_text,
            "metadata": record.get("metadata", {}),
        })

    metrics = {
        "num_examples": len(records),
        "num_with_gold": total_gold,
        "parse_rate": parsed / max(1, len(records)),
        "format_validity": format_ok / max(1, len(records)),
        "verdict_accuracy": correct / max(1, total_gold),
    }

    write_jsonl(args.output_file, outputs)
    metrics_path = str(Path(args.output_file).with_suffix(".metrics.json"))
    Path(metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions to {args.output_file}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
