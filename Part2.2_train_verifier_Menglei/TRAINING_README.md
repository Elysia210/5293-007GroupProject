# Verifier Training README

This document explains the training pipeline for the EG-GenRM verifier.

The goal is to train a generative verifier that can read a math problem and a candidate solution, then generate a step-by-step verification rationale and a final Yes/No correctness judgment.

---

## 1. Training Goal

The verifier learns the following mapping:

```text
Input:
Question + Candidate Solution

Output:
Verification Rationale + Final Yes/No Judgment
```

The final line must follow this exact format:

```text
Verification: Is the answer correct (Yes/No)? X
```

where `X` is either `Yes` or `No`.

This project does not train a simple classifier. Instead, it trains a generative verifier. The model must explain why a candidate solution is correct or incorrect before giving the final judgment.

---

## 2. Training Data

The training data comes from the data construction pipeline in Part 2.1.

It may contain:

```text
filtered teacher-generated data
PPM-generated rationale data
public GenRM-style data
```

Before training, all data should be converted into chat-style SFT format.

A training example should look like this:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Question:\n...\n\nCandidate Solution:\n...\n\nPlease verify the candidate solution step by step. End with: Verification: Is the answer correct (Yes/No)? X"
    },
    {
      "role": "assistant",
      "content": "Step 1: ...\nStep 2: ...\nVerification: Is the answer correct (Yes/No)? No"
    }
  ],
  "metadata": {
    "source": "teacher_generated_or_ppm_generated_or_public",
    "gold_is_correct": false
  }
}
```

During training, the user prompt is used as input. The assistant message is the target output.

---

## 3. Important Prompt Rule

The final verifier should only see:

```text
Question
Candidate Solution
```

It should not see the expected answer or reference solution during training.

This is important because evaluation also gives the verifier only the question and candidate solution. If the model sees the expected answer during training, there will be a train-test mismatch.

Teacher data generation can use expected answers, but final verifier training should not.

---

## 4. Training Methods

We train two main models.

### 4.1 Vanilla GenRM

Vanilla GenRM is the baseline.

It uses standard cross-entropy loss on all assistant tokens.

Run:

```bash
python scripts/train/train_vanilla_genrm.py --config configs/vanilla_genrm_config.yaml
```

This model learns to generate verification rationales and final Yes/No judgments from the training data.

---

### 4.2 Entropy-Guided GenRM

Entropy-Guided GenRM uses an entropy-aware loss.

The idea is that some tokens are more important than others. Tokens with high predictive entropy often correspond to uncertain reasoning points, error transitions, or final judgment decisions.

Run:

```bash
python scripts/train/train_entropy_genrm.py --config configs/entropy_genrm_config.yaml
```

This model gives higher training weight to uncertain tokens.

---

### 4.3 Verdict-Aware Entropy GenRM

This is the recommended version.

It uses entropy-aware weighting but also protects the final Yes/No verdict tokens. This avoids a common failure mode where the model focuses on high-entropy reasoning tokens but loses the required final output format.

Run:

```bash
python scripts/train/train_verdict_aware_entropy_genrm.py --config configs/entropy_genrm_config.yaml
```

Use this version for the main EG-GenRM result.

---

## 5. Config Files

Training settings are controlled by YAML config files.

```text
configs/
├── vanilla_genrm_config.yaml
└── entropy_genrm_config.yaml
```

Important fields:

```yaml
model:
  base_model: Qwen/Qwen2.5-1.5B-Instruct
  use_lora: true

data:
  train_file: data/final_verifier_sft_train.jsonl
  valid_file: data/final_verifier_sft_valid.jsonl
  test_file: data/final_verifier_sft_test.jsonl
  max_seq_length: 2048

training:
  output_dir: outputs/checkpoints/entropy_genrm
  learning_rate: 1.0e-4
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  bf16: true
```

For entropy-guided training, also check:

```yaml
entropy:
  alpha: 0.05
  topk_ratio: 0.2
  high_entropy_extra_weight: 0.5
  protect_verdict_tokens: true
  verdict_weight: 2.0
```

---

## 6. Recommended Training Order

Train the models in this order:

```text
Step 1: Train Vanilla GenRM
Step 2: Train Entropy-Guided GenRM
Step 3: Train Verdict-Aware Entropy GenRM
Step 4: Evaluate all models on the same held-out validation/test set
```

This allows a fair comparison between the baseline and the entropy-guided method.

---

## 7. Evaluation After Training

After training, evaluate the model with:

```bash
python scripts/evaluate/evaluate_verifier.py \
  --config configs/entropy_genrm_config.yaml \
  --model_path outputs/checkpoints/entropy_genrm \
  --eval_file data/final_verifier_sft_test.jsonl \
  --output_file outputs/evaluation_results/entropy_predictions.jsonl
```

The evaluation checks:

```text
whether the model outputs a valid final Yes/No line
whether the Yes/No verdict can be parsed
whether the parsed verdict matches the gold label
```

Main metrics:

```text
parse_rate
format_validity
verdict_accuracy
```

---

## 8. Generalization Evaluation

To test generalization, use a held-out evaluation file such as:

```text
openmath_val_verifier_eval.jsonl
```

This file should not be used during training.

It contains unseen candidate solutions from OpenMathInstruct validation data. The model must decide whether each candidate solution is correct or incorrect.

After evaluation, analyze four types of cases:

```text
True Positive: gold Yes, model Yes
True Negative: gold No, model No
False Positive: gold No, model Yes
False Negative: gold Yes, model No
```

False positives are especially important because they show when the verifier is fooled by wrong but fluent reasoning.

---

## 9. Practical Notes

### GPU Memory

If the model does not fit in memory, reduce:

```text
max_seq_length
per_device_train_batch_size
```

Recommended small-GPU setting:

```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
max_seq_length: 2048
use_lora: true
```

If memory is still not enough, try:

```yaml
max_seq_length: 1024
```

---

### LoRA

LoRA is recommended for this project.

It reduces GPU memory usage and makes training easier on Kaggle or Colab.

Make sure the config contains:

```yaml
model:
  use_lora: true
```

---

### Output Format

The model must end with:

```text
Verification: Is the answer correct (Yes/No)? X
```

If the model often misses this line, check:

```text
training data format
verifier_train_prompt.txt
whether verdict tokens are protected in entropy training
```

---

### Do Not Train on Bad Data

Before training, make sure the dataset has been filtered.

Bad examples include:

```text
missing final verdict
wrong Yes/No label
empty rationale
very short rationale
assistant output without verification reasoning
```

Low-quality training data can make the verifier unstable.

---

### Avoid Train-Test Mismatch

Do not include expected answers or reference solutions in the final verifier input.

The final verifier should only receive:

```text
Question + Candidate Solution
```

Teacher generation may use expected answers, but verifier training and evaluation should not.

---

## 10. Files That Should Not Be Uploaded

Do not upload:

```text
full training data
full validation data
model checkpoints
wandb logs
.env files
API keys
large generated outputs
```

Only upload:

```text
README.md
TRAINING.md
requirements.txt
configs/
prompts/
scripts/
small sample data
```

---

## 11. Summary

The training pipeline is:

```text
final verifier SFT data
        ↓
train Vanilla GenRM
        ↓
train Entropy-Guided GenRM
        ↓
train Verdict-Aware Entropy GenRM
        ↓
evaluate final Yes/No correctness
        ↓
analyze false positives and false negatives
```

The main comparison is:

```text
Vanilla GenRM:
plain cross-entropy on all assistant tokens

Entropy-Guided GenRM:
higher weight on uncertain reasoning tokens

Verdict-Aware Entropy GenRM:
entropy-guided training + protected Yes/No verdict tokens
```

The final goal is to build a verifier that can generalize to unseen math candidate solutions and reliably detect both correct reasoning and subtle wrong reasoning.
