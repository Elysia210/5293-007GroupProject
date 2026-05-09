# TRAINING.md

# Verifier Training Guide

This document explains the training logic for Part 2.2.

The input data comes from Part 2.1. It should already be filtered, merged, and converted into SFT format.

---

## 1. Goal

The goal is to train a generative verifier.

Given:

```text
Question
Candidate Solution
```

the model should generate:

```text
Verification rationale
Verification: Is the answer correct (Yes/No)? X
```

This is different from a discriminative reward model. The model does not only output a scalar score or a label. It generates a verification explanation first, then gives the final verdict.

---

## 2. Data Format

The recommended training format is chat-style JSONL.

Each line is one example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Question:\n...\n\nCandidate Solution:\n...\n\nPlease verify the candidate solution step by step."
    },
    {
      "role": "assistant",
      "content": "Step 1: ...\nStep 2: ...\nVerification: Is the answer correct (Yes/No)? No"
    }
  ],
  "metadata": {
    "gold_is_correct": false,
    "source": "teacher_or_ppm_or_public"
  }
}
```

During training, user tokens are masked out. Only assistant tokens receive loss.

---

## 3. Model

The default model is:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

This model is small enough for a course-project setting and can be trained with LoRA.

You can change the base model in the config file:

```yaml
model:
  base_model: Qwen/Qwen2.5-1.5B-Instruct
```

---

## 4. Method 1: Vanilla GenRM

Vanilla GenRM is the baseline.

It uses normal cross-entropy loss over assistant tokens.

Command:

```bash
python scripts/train/train_vanilla_genrm.py --config configs/vanilla_genrm_config.yaml
```

Conceptually:

```text
Question + Candidate Solution
        ↓
Model
        ↓
Verification rationale + Yes/No
```

All assistant tokens are trained equally.

---

## 5. Method 2: Entropy-Guided GenRM

Entropy-Guided GenRM modifies the token-level loss.

The motivation is that not all tokens are equally important.

For example, in a verifier output:

```text
The solution computes 35 pies correctly.
However, the candidate forgets to subtract the dropped pies.
Verification: Is the answer correct (Yes/No)? No
```

Some tokens are more important for verification:

```text
However
forgets
subtract
No
```

These are tokens where the model may be uncertain and where the reasoning direction changes.

Entropy-Guided GenRM increases the loss weight on uncertain tokens while still protecting the final verdict format.

Command:

```bash
python scripts/train/train_entropy_genrm.py --config configs/entropy_genrm_config.yaml
```

---

## 6. Entropy-Guided Loss

For each target token, the model produces a probability distribution over the vocabulary.

The predictive entropy is:

```text
H_t = - sum_v p(v | context) log p(v | context)
```

Higher entropy means the model is more uncertain.

The entropy-guided loss uses a token weight:

```text
weight_t = 1 + alpha * normalized_entropy_t
```

The loss is:

```text
L = sum_t weight_t * CE_t / number_of_valid_tokens
```

The config also supports an extra weight for the top-k high-entropy tokens:

```yaml
entropy:
  alpha: 0.05
  topk_ratio: 0.2
  high_entropy_extra_weight: 0.5
```

We also protect final verdict tokens:

```yaml
entropy:
  protect_verdict_tokens: true
  verdict_weight: 2.0
  verdict_words:
    - Yes
    - No
```

This is important because a pure high-entropy-only mask can make the model lose the final answer format.

---

## 7. Recommended Experiments

Run at least two experiments:

### Experiment A: Vanilla GenRM

```bash
python scripts/train/train_vanilla_genrm.py --config configs/vanilla_genrm_config.yaml
```

### Experiment B: Entropy-Guided GenRM

```bash
python scripts/train/train_entropy_genrm.py --config configs/entropy_genrm_config.yaml
```

Then evaluate both models on the same validation/test set.

---

## 8. Evaluation

Use:

```bash
python scripts/evaluate/evaluate_verifier.py \
  --config configs/entropy_genrm_config.yaml \
  --model_path outputs/checkpoints/entropy_genrm \
  --eval_file data/final_verifier_sft_test.jsonl \
  --output_file outputs/evaluation_results/entropy_predictions.jsonl
```

The evaluation script checks:

```text
Can the model generate a valid final Yes/No line?
Does the parsed Yes/No match the gold label?
How often does the model fail to follow the required format?
```

Main metrics:

```text
verdict_accuracy
parse_rate
format_validity
```

---

## 9. Practical Settings

For a small GPU, use:

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_seq_length: 2048
```

If memory is not enough:

```text
reduce max_seq_length to 1024
use a smaller base model
keep LoRA enabled
use bf16 if supported
```

---

## 10. Summary

This part compares:

```text
Vanilla GenRM:
standard cross-entropy

Entropy-Guided GenRM:
entropy-weighted cross-entropy with verdict-token protection
```

The goal is to test whether entropy-guided training helps the verifier focus more on uncertain reasoning steps while keeping the final Yes/No judgment reliable.
