<<<<<<< HEAD

# EG-GenRM — Part1 Baseline Implementations and Data Processing

This repository implements the first stage of the EG-GenRM project: generating structured multi-path reasoning data for GSM8K and MATH-500.

The pipeline prepares benchmark math problems, generates 5 independent chain-of-thought reasoning paths per problem, extracts final answers, and stores token-level uncertainty features for downstream verifier training and analysis.

The current uploaded version focuses on:

- Dataset preparation
- Multi-path CoT generation (`k = 5`)
- Direct-answer baseline generation
- Token-level logprob / entropy extraction
- Unified JSONL dataset export

---

# Quick Start

## 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

---

## 2. Create environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Local test run (no API key required)

This uses the echo backend to verify the pipeline structure.

```bash
python src/main_step1.py \
  --limit 20 \
  --datasets all \
  --backend echo
```

---

## 4. Run the OpenAI generation pipeline

```bash
export OPENAI_API_KEY=your_api_key_here
```

```bash
python src/main_step1.py \
  --datasets all \
  --backend openai \
  --openai-model gpt-4o-mini \
  --entropy-mode none \
  --openai-top-logprobs 20 \
  --max-workers 5 \
  -k 5
```

This generates:

- 5 independent CoT candidates per problem
- extracted final answers
- token-level logprob and entropy features

Outputs are written under:

```text
data/final/
```

---

## 5. Run Direct Baseline

The direct baseline is generated separately from the multi-path CoT pipeline.

```bash
python run_direct_model.py \
  --input data/final/cleaned_samples.jsonl \
  --out data/final/direct_live.jsonl \
  --model gpt-4o-mini \
  --openai-top-logprobs 20 \
  --max-workers 5 \
  --resume
```

This produces one direct answer per question without multi-path reasoning.

---

# What This Pipeline Does

The main entry point is:

```text
src/main_step1.py
```

The pipeline performs the following steps:

1. Load GSM8K and/or MATH-500 benchmark problems
2. Clean questions and normalize gold answers
3. Generate 5 independent chain-of-thought candidate solutions
4. Extract final answers from each candidate
5. Store token-level logprob and entropy when available
6. Export unified JSONL outputs for downstream analysis and verifier training

The direct-answer baseline is generated separately using:

```text
run_direct_model.py
```

This repository currently focuses on data preparation and candidate generation only.

---

# Repository Structure

```text
.
├── src/
│   └── main_step1.py
├── data/
│   ├── raw/
│   ├── interim/
│   └── final/
├── prompts/
├── reports/
├── eval_baselines.py
├── run_direct_model.py
├── requirements.txt
└── README.md
```

---

# Main Outputs

After running the pipeline, the output directory contains:

| File | Description |
|---|---|
| `cleaned_samples.jsonl` | Gold-side data only: question, normalized answer, dataset metadata |
| `candidate_paths.jsonl` | One row per generated reasoning path |
| `entropy_features.jsonl` | Token-level logprob and entropy features |
| `step1_dataset.jsonl` | Unified nested dataset containing gold data, candidates, and token features |
| `direct_live.jsonl` | Direct-answer baseline outputs generated separately |
| `run_manifest.json` | Run configuration and timestamp |

The main downstream file is:

```text
step1_dataset.jsonl
```

---

# Running Different Dataset Options

## GSM8K only

```bash
python src/main_step1.py \
  --datasets gsm8k \
  --gsm8k-splits train test \
  --backend openai \
  -k 5
```

---

## MATH-500 only

```bash
python src/main_step1.py \
  --datasets math500 \
  --math500-split test \
  --backend openai \
  -k 5
```

---

## Debugging with fewer examples

```bash
python src/main_step1.py \
  --limit 10 \
  --datasets all \
  --backend echo
```

---

## Save outputs to a separate folder

```bash
python src/main_step1.py \
  --out-dir data/final/run_$(date +%Y%m%d) \
  --datasets all \
  --backend openai \
  -k 5
```

---

# Resume Interrupted Runs

The pipeline writes outputs incrementally.

To resume an interrupted run:

```bash
python src/main_step1.py \
  --resume \
  --out-dir data/final \
  --datasets all \
  --backend openai \
  --entropy-mode none \
  -k 5
```

Important:

- Use the same output directory
- Use compatible flags
- Completed samples will be skipped automatically

Without `--resume`, output JSONL files are overwritten for a fresh run.

---

# Entropy / Logprob Notes

When using the OpenAI backend, the pipeline stores token-level log probabilities and derives entropy from the returned top-logprob distribution.

Recommended setting:

```bash
--entropy-mode none
```

This preserves the OpenAI token-level uncertainty features.

The entropy values are used as uncertainty-related features for later analysis and downstream verifier training.

---

# Baseline Evaluation

The repository also includes baseline evaluation utilities:

```bash
python eval_baselines.py \
  --data data/final/step1_dataset.jsonl \
  --direct data/final/direct_live.jsonl
```

This evaluates:

- Direct
- CoT
- Best-of-N vote
- Best-of-N score

The direct baseline is optional and is loaded separately through:

```text
direct_live.jsonl
```

---

# Recommended Workflow

## Quick local test

```bash
python src/main_step1.py \
  --limit 20 \
  --datasets all \
  --backend echo
```

---

## Full OpenAI run

```bash
python src/main_step1.py \
  --datasets all \
  --backend openai \
  --openai-model gpt-4o-mini \
  --entropy-mode none \
  --openai-top-logprobs 20 \
  --max-workers 5 \
  -k 5
```

---

## Direct baseline

```bash
python run_direct_model.py \
  --input data/final/cleaned_samples.jsonl \
  --out data/final/direct_live.jsonl \
  --model gpt-4o-mini \
  --resume
```

---

## Baseline evaluation

```bash
python eval_baselines.py \
  --data data/final/step1_dataset.jsonl \
  --direct data/final/direct_live.jsonl
```

---

# Reproducibility Checklist

Before sharing results, record:

- Dataset (`gsm8k`, `math500`, or `all`)
- Model name
- Backend (`echo` or `openai`)
- Output directory
- Whether `--resume` was used

The file:

```text
run_manifest.json
```

stores run arguments and timestamps automatically.

---

# Common Issues

## No OpenAI API key

Use the echo backend:

```bash
python src/main_step1.py \
  --limit 5 \
  --datasets all \
  --backend echo
```

---

## Interrupted long runs

Use:

```bash
--resume
```

with the same output directory.

---

## OpenAI rate limits

Reduce:

```bash
--max-workers
```

For example:

```bash
--max-workers 2
```

---

# Current Scope

This uploaded version currently includes:

- Dataset preparation
- Gold answer normalization
- 5-path CoT candidate generation
- Direct-answer baseline generation
- Token-level logprob / entropy extraction
- Unified JSONL dataset export
- Baseline evaluation utilities

Future stages of the project include:

- Generative verifier training
- Entropy-guided verifier objectives
- Entropy-masked cross-entropy experiments
- Final EG-GenRM evaluation and comparison
