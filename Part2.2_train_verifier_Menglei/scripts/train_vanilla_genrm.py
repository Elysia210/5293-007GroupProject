#!/usr/bin/env python
"""
Train Vanilla GenRM.

This is the baseline verifier training script.
It uses standard cross-entropy loss on all assistant tokens.
"""

import argparse
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None


import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_bool_label(record: Dict[str, Any]) -> Optional[bool]:
    metadata = record.get("metadata", {}) or {}
    for key in ["gold_is_correct", "is_correct", "candidate_is_correct"]:
        if key in record:
            return bool(record[key])
        if key in metadata:
            return bool(metadata[key])
    if "gold_verdict" in record:
        return str(record["gold_verdict"]).strip().lower() == "yes"
    if "gold_verdict" in metadata:
        return str(metadata["gold_verdict"]).strip().lower() == "yes"
    return None


def build_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    if isinstance(record.get("messages"), list) and len(record["messages"]) >= 2:
        return record["messages"]

    question = record.get("question") or record.get("problem") or ""
    candidate = (
        record.get("candidate_solution")
        or record.get("generated_solution")
        or record.get("solution")
        or ""
    )
    rationale = (
        record.get("verification_rationale")
        or record.get("answer_text")
        or record.get("rationale")
        or ""
    )
    verdict = record.get("final_verdict") or record.get("parsed_final_verdict")
    if verdict is None:
        label = get_bool_label(record)
        verdict = None if label is None else ("Yes" if label else "No")

    if verdict and "Verification: Is the answer correct" not in rationale:
        rationale = rationale.rstrip() + f"\nVerification: Is the answer correct (Yes/No)? {verdict}"

    user = (
        "Question:\n"
        f"{question}\n\n"
        "Candidate Solution:\n"
        f"{candidate}\n\n"
        "Please verify the candidate solution step by step. "
        "End with: Verification: Is the answer correct (Yes/No)? X"
    )
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": rationale},
    ]


def render_chat(tokenizer: AutoTokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    user_text = messages[0]["content"] if messages else ""
    if add_generation_prompt:
        return f"### User:\n{user_text}\n\n### Assistant:\n"
    assistant_text = messages[1]["content"] if len(messages) > 1 else ""
    return f"### User:\n{user_text}\n\n### Assistant:\n{assistant_text}"


class VerifierSFTDataset(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer, max_seq_length: int):
        self.records = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        messages = build_messages(record)

        prompt_text = render_chat(self.tokenizer, messages[:1], add_generation_prompt=True)
        full_text = render_chat(self.tokenizer, messages[:2], add_generation_prompt=False)

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        full_ids = full_ids[: self.max_seq_length]
        labels = full_ids.copy()

        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


@dataclass
class VerifierDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in features)
        pad_id = self.tokenizer.pad_token_id

        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad_len = max_len - item["input_ids"].shape[0]
            batch["input_ids"].append(torch.cat([
                item["input_ids"],
                torch.full((pad_len,), pad_id, dtype=torch.long),
            ]))
            batch["attention_mask"].append(torch.cat([
                item["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long),
            ]))
            batch["labels"].append(torch.cat([
                item["labels"],
                torch.full((pad_len,), -100, dtype=torch.long),
            ]))

        return {k: torch.stack(v) for k, v in batch.items()}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(cfg):
    model_name = cfg["model"]["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32
    if torch.cuda.is_available():
        if cfg["training"].get("bf16", False):
            dtype = torch.bfloat16
        elif cfg["training"].get("fp16", False):
            dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if cfg["training"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if cfg["model"].get("use_lora", True):
        if LoraConfig is None:
            raise ImportError("Please install peft: pip install peft")
        lora_cfg = cfg.get("lora", {})
        peft_cfg = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg.get("project", {}).get("seed", 42))

    model, tokenizer = load_model_and_tokenizer(cfg)

    train_ds = VerifierSFTDataset(
        cfg["data"]["train_file"],
        tokenizer,
        cfg["data"].get("max_seq_length", 2048),
    )
    valid_file = cfg["data"].get("valid_file")
    eval_ds = VerifierSFTDataset(
        valid_file,
        tokenizer,
        cfg["data"].get("max_seq_length", 2048),
    ) if valid_file else None

    out_dir = cfg["training"]["output_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    args_train = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg["training"].get("num_train_epochs", 1),
        per_device_train_batch_size=cfg["training"].get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=cfg["training"].get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=cfg["training"].get("gradient_accumulation_steps", 8),
        learning_rate=float(cfg["training"].get("learning_rate", 1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
        warmup_ratio=float(cfg["training"].get("warmup_ratio", 0.03)),
        logging_steps=cfg["training"].get("logging_steps", 10),
        save_steps=cfg["training"].get("save_steps", 200),
        eval_steps=cfg["training"].get("eval_steps", 200),
        save_total_limit=cfg["training"].get("save_total_limit", 2),
        bf16=cfg["training"].get("bf16", False),
        fp16=cfg["training"].get("fp16", False),
        report_to=[] if cfg["training"].get("report_to", "none") == "none" else cfg["training"].get("report_to"),
        evaluation_strategy="steps" if eval_ds is not None else "no",
        save_strategy="steps",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=VerifierDataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved Vanilla GenRM to {out_dir}")


if __name__ == "__main__":
    main()
