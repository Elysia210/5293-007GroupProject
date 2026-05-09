# Data Documentation

This document describes the data construction pipeline for the EG-GenRM project. The goal of this repository is not only to collect public math datasets, but to transform them into verifier-oriented supervised fine-tuning data.

The final training format is:

```text
Input:
Question + Candidate Solution

Target:
Step-by-step verification rationale + final Yes/No correctness judgment
```

Every verification target must end with the same final line:

```text
Verification: Is the answer correct (Yes/No)? X
```

where `X` is either `Yes` or `No`.

The full data pipeline contains five stages:

1. direct teacher data generation;
2. teacher-output filtering;
3. public-source data selection;
4. PPM-based rationale generation for public data;
5. final data fusion and quality checking.

---

## 1. Pipeline Overview

```text
Public math reasoning data
        |
        v
Candidate pool construction
        |
        v
Teacher seed selection
        |
        v
Teacher API generation
        |
        v
Raw teacher verification data
        |
        v
Filtering and quality checks
        |
        v
Clean teacher data
        |
        v
PPM SFT data
        |
        v
Train lightweight PPM
        |
        v
Use PPM to complete public-source data into teacher-like rationale data
        |
        v
Merge teacher data + PPM-generated data + selected public data
        |
        v
Final GenRM / EG-GenRM verifier SFT dataset
```

The current uploaded files mainly cover the first part of the pipeline:

- `teacherdata.ipynb`: notebook used to build candidate packs and call teacher models;
- `teacher_outputs_all_candidates (38).jsonl`: raw teacher-generated verification outputs.

Before releasing the repository, the notebook should be cleaned and the hard-coded API key must be removed. API keys should be loaded from environment variables instead.

---

## 2. Data Sources

The current teacher data is built from `nvidia/OpenMathInstruct-1`.

The raw dataset provides model-generated mathematical solutions with correctness labels. The main fields used by this project are:

```text
question
generated_solution
expected_answer
predicted_answer
is_correct
```

We use these fields to construct a candidate-solution pool. Each record in the candidate pool represents one possible solution to one math problem.

A normalized candidate record has the following schema:

```json
{
  "sample_id": "...",
  "source": "openmathinstruct",
  "dataset": "gsm8k_or_math",
  "question_norm": "normalized question for grouping",
  "question": "original math question",
  "candidate_solution_raw": "raw generated solution",
  "candidate_solution_clean": "cleaned candidate solution",
  "expected_answer": "ground-truth final answer",
  "candidate_predicted_answer": "candidate final answer if available",
  "candidate_is_correct": true,
  "split": "train"
}
```

Later stages may also use additional public-source data such as GSM8K, MATH/MATH500, and other GenRM-style or verifier-style public data. All public-source data must be converted into the same candidate-solution schema before fusion.

---

## 3. Stage 1: Direct Teacher Data Generation

### 3.1 Goal

The purpose of the direct teacher data stage is to obtain high-quality step-by-step verification rationales from a strong teacher model.

The teacher model receives:

```text
Question
Candidate Solution
Reference Correct Solution
Ground-Truth Final Answer
Candidate Role
```

and returns:

```text
Verification rationale
Earliest error if the candidate is wrong
Final Yes/No judgment
```

This data is used as the first high-quality supervision source for training a lightweight PPM and the later GenRM / EG-GenRM verifier.

---

### 3.2 Candidate Pack Construction

The notebook groups candidate solutions by normalized question. A question is kept only if it has both correct and incorrect candidate solutions. This is important because the verifier needs to learn how to distinguish correct reasoning from plausible but wrong reasoning for the same problem.

For each selected question, the pipeline constructs a balanced candidate pack with up to four roles:

| Candidate role | Meaning |
|---|---|
| `correct_1` | A correct candidate solution for the question |
| `correct_2` | A second correct candidate solution for the same question |
| `subtle_wrong` | An incorrect candidate solution that looks similar to a correct solution but contains a hidden error |
| `random_wrong` | A more general incorrect candidate solution |

The current teacher-generation run uses approximately 200 selected question packs. Since each pack has up to 4 candidate solutions, the expected number of teacher examples is around 800.

The uploaded teacher output file contains:

```text
757 successful teacher outputs
189 unique normalized questions
190 correct_1 candidates
190 correct_2 candidates
189 subtle_wrong candidates
188 random_wrong candidates
```

The label balance is:

```text
380 correct candidates
377 incorrect candidates
```

This balanced structure is useful because the verifier should not simply learn to always answer `Yes` or always answer `No`.

---

### 3.3 Subtle Wrong Candidate Selection

The `subtle_wrong` examples are especially important for this project.

A subtle wrong solution is an incorrect solution that may look fluent, reasonable, or close to a correct solution, but contains a mathematical or logical error. These examples are valuable because standard verifiers and simple LLM-as-a-Judge prompts are often fooled by them.

The notebook uses heuristics such as:

```text
solution length
number of reasoning steps
text similarity to correct solutions
prefix overlap with correct solutions
line-level divergence point
whether the final answer differs from the expected answer
whether the solution contains uncertainty or invalid artifacts
```

The goal is to select wrong solutions that are not trivial garbage, but are useful training cases for a verifier.

---

### 3.4 Teacher API Generation

The teacher model is prompted to act as a strict mathematical verifier.

The teacher prompt is stored conceptually as:

```text
prompts/teacher_verifier_prompt.txt
```

The teacher is instructed to:

1. verify one candidate solution at a time;
2. focus on mathematical reasoning and final correctness;
3. ignore harmless markup artifacts when they do not affect the reasoning;
4. identify the earliest fatal error for incorrect solutions;
5. end with exactly one final verdict line.

The final line must be:

```text
Verification: Is the answer correct (Yes/No)? X
```

The uploaded teacher output file contains outputs from:

```text
moonshotai/kimi-k2.5
deepseek-ai/deepseek-v3.2
```

The notebook uses an OpenAI-compatible API endpoint through NVIDIA NIM. In the public GitHub version, the API key must not be written directly in the notebook or scripts. Use `.env` instead.

Example safe API setup:

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("NVIDIA_API_KEY"),
    base_url=os.getenv("NVIDIA_API_BASE", "https://integrate.api.nvidia.com/v1")
)
```

The public `.env.example` should look like:

```text
NVIDIA_API_KEY=your_api_key_here
NVIDIA_API_BASE=https://integrate.api.nvidia.com/v1
TEACHER_MODEL=moonshotai/kimi-k2.5
```

The real `.env` file must be ignored by Git.

---

## 4. Stage 2: Teacher Output Filtering

### 4.1 Raw Teacher Output

The raw teacher output file is:

```text
teacher_outputs_all_candidates.jsonl
```

In the uploaded experiment, this corresponds to:

```text
teacher_outputs_all_candidates (38).jsonl
```

Each row contains one teacher verification result for one candidate solution.

Main fields:

| Field | Description |
|---|---|
| `status` | Whether the API call succeeded |
| `run_id` | ID of the teacher-generation run |
| `row_id` | Unique row identifier |
| `question_norm` | Normalized question for grouping and deduplication |
| `question` | Original math question |
| `candidate_role` | `correct_1`, `correct_2`, `subtle_wrong`, or `random_wrong` |
| `candidate_is_correct` | Original correctness label |
| `expected_answer` | Ground-truth final answer |
| `candidate_predicted_answer` | Candidate predicted final answer |
| `candidate_solution_raw` | Original candidate solution |
| `candidate_solution_clean` | Cleaned candidate solution |
| `reference_solution_raw` | Raw reference correct solution |
| `reference_solution_clean` | Cleaned reference correct solution |
| `teacher_model` | Teacher model used for rationale generation |
| `teacher_prompt` | Full prompt sent to the teacher |
| `answer_text` | Teacher-generated verification rationale |
| `reasoning_text` | Extra reasoning field if returned by the API |
| `finish_reason` | API finish reason |
| `parsed_final_verdict` | Parsed final Yes/No verdict |
| `earliest_error` | Parsed earliest error if available |
| `usage` | Token usage information |
| `retry_count` | Number of retries |
| `timestamp` | Generation timestamp |

---

### 4.2 Filtering Rules

Raw teacher outputs are not used directly for training. They must pass quality checks first.

A strict clean teacher example should satisfy:

```text
status == "ok"
parsed_final_verdict is "Yes" or "No"
parsed_final_verdict matches candidate_is_correct
answer_text is not empty
answer_text contains step-by-step verification
incorrect candidates should include an earliest error explanation when possible
```

Examples are removed if they have:

```text
API failure
missing final Yes/No verdict
teacher verdict inconsistent with the original correctness label
too-short rationale
format failure
empty candidate solution
severe code or markup artifacts that make the solution unreadable
```

In the uploaded file:

```text
757 total raw teacher outputs
724 outputs have a parsed Yes/No verdict
663 outputs have a parsed verdict that matches the original correctness label
33 outputs have no parsed final verdict
61 outputs have a parsed verdict but disagree with the original label
```

A strict filtering setting would keep the 663 label-consistent examples. A more permissive setting may keep disagreement examples for manual inspection, because some original automatic labels may be noisy.

Recommended output files:

```text
data/interim/teacher_outputs_all_candidates.jsonl
data/processed/clean_teacher_data.jsonl
data/examples/clean_teacher_sample.jsonl
```

---

## 5. Stage 3: Public Data Selection

### 5.1 Goal

Teacher-generated data is high quality but expensive and limited in size. Public-source data is larger and cheaper, but often lacks detailed verification rationales.

Therefore, we prepare public-source data as a larger candidate pool that can later be completed by the PPM.

Potential public sources include:

```text
GSM8K
MATH / MATH500
OpenMathInstruct-1 remaining examples
GenRM-style public verification examples
Math-Shepherd / PRM-style process data if used
```

---

### 5.2 Normalized Public Candidate Format

Every public-source example should be converted into the same schema:

```json
{
  "id": "public_000001",
  "source": "openmathinstruct",
  "dataset": "gsm8k_or_math",
  "question": "...",
  "candidate_solution": "...",
  "expected_answer": "...",
  "candidate_predicted_answer": "...",
  "is_correct": false,
  "split": "train"
}
```

Public data filtering should remove:

```text
empty questions
empty candidate solutions
extremely short solutions
invalid labels
duplicate question-solution pairs
unreadable code-only outputs when no natural-language reasoning remains
```

Recommended output file:

```text
data/processed/public_candidate_pool.jsonl
```

---

## 6. Stage 4: PPM SFT and PPM-Based Rationale Generation

### 6.1 Build PPM SFT Data from Clean Teacher Data

The clean teacher data is converted into chat-style supervised fine-tuning data for a lightweight PPM.

The PPM input is:

```text
Question + Candidate Solution
```

The PPM target is:

```text
Teacher-generated verification rationale + final Yes/No verdict
```

Example PPM SFT format:

```json
{
  "id": "ppm_sft_000001",
  "messages": [
    {
      "role": "user",
      "content": "Question:\n...\n\nCandidate Solution:\n...\n\nPlease verify the candidate solution step by step."
    },
    {
      "role": "assistant",
      "content": "Step 1: ...\nStep 2: ...\nEarliest Error: ...\nVerification: Is the answer correct (Yes/No)? No"
    }
  ],
  "metadata": {
    "source": "teacher_generated",
    "candidate_role": "subtle_wrong",
    "gold_is_correct": false,
    "teacher_model": "moonshotai/kimi-k2.5"
  }
}
```

Recommended output files:

```text
data/processed/ppm_sft_train.jsonl
data/processed/ppm_sft_valid.jsonl
data/examples/ppm_sft_sample.jsonl
```

---

### 6.2 Use PPM to Complete Public Data

After the PPM is trained, it is used to generate teacher-like verification rationales for public-source candidate solutions.

Input to PPM:

```text
Question
Candidate Solution
```

Output from PPM:

```text
Step-by-step verification rationale
Verification: Is the answer correct (Yes/No)? X
```

The resulting PPM-generated data has the same structure as teacher data, but its `rationale_type` is marked as `ppm_generated`.

Example:

```json
{
  "id": "ppm_generated_000001",
  "source": "openmathinstruct",
  "dataset": "gsm8k",
  "generator": "ppm",
  "question": "...",
  "candidate_solution": "...",
  "gold_is_correct": false,
  "verification_rationale": "Step 1: ...\nStep 2: ...\nVerification: Is the answer correct (Yes/No)? No",
  "parsed_final_verdict": "No",
  "rationale_type": "ppm_generated"
}
```

Recommended output file:

```text
data/processed/ppm_generated_rationales.jsonl
```

---

## 7. Stage 5: Final Data Fusion

### 7.1 Data Sources to Merge

The final verifier training data is built by merging three types of data:

| Data type | Description | Quality | Size |
|---|---|---|---|
| `teacher_generated` | Direct rationales from strong teacher models | Highest | Small |
| `ppm_generated` | Rationales generated by the trained PPM | Medium | Larger |
| `public_label_only` | Public examples with labels but no detailed rationale | Lower / auxiliary | Large |

The merged data is converted into one unified chat-style SFT format.

---

### 7.2 Final Verifier SFT Format

Final training example:

```json
{
  "id": "final_train_000001",
  "messages": [
    {
      "role": "user",
      "content": "Question:\n...\n\nCandidate Solution:\n...\n\nVerify the candidate solution step by step."
    },
    {
      "role": "assistant",
      "content": "Step 1: ...\nStep 2: ...\nVerification: Is the answer correct (Yes/No)? No"
    }
  ],
  "metadata": {
    "source": "teacher_generated",
    "dataset": "openmathinstruct",
    "gold_is_correct": false,
    "rationale_type": "teacher",
    "candidate_role": "subtle_wrong"
  }
}
```

Recommended output files:

```text
data/processed/final_verifier_sft_train.jsonl
data/processed/final_verifier_sft_valid.jsonl
data/examples/final_train_sample.jsonl
```

This final dataset is the common input for later verifier training experiments, including:

```text
Vanilla GenRM SFT
Entropy-weighted GenRM SFT
Top-k high-entropy loss
Verdict-aware entropy loss
EG-GenRM variants
```

---

## 8. End-to-End Example

This example shows how one raw NVIDIA/OpenMathInstruct-style sample becomes a final verifier training example.

---

### 8.1 Raw Question

```text
Question:
How many more digits does the base-3 representation of $987_{10}$ have than the base-8 representation of $987_{10}$?

Expected Answer:
3
```

The raw dataset contains multiple generated solutions for this same question. Some are correct and some are wrong.

---

### 8.2 Candidate Grouping

The notebook first normalizes the question for grouping:

```text
how many more digits does the base-3 representation of $987_{10}$ have than the base-8 representation of $987_{10}$?
```

Then it checks whether this normalized question has both correct and incorrect candidates. Since it does, the question is kept.

---

### 8.3 Candidate Pack

For this question, the pipeline builds a candidate pack.

A correct reference solution says:

```text
The base-3 representation has 7 digits and the base-8 representation has 4 digits.
The difference is 7 - 4 = 3.
```

A selected `subtle_wrong` candidate says:

```text
We can easily do it in python. First, we need to create a function that changes base-10 numbers to other bases. Then we get base-N and base-M representations and calculate lengths difference.
```

This candidate is useful because it sounds method-like, but it does not actually solve the specific problem. It never substitutes base 3 and base 8 and never provides the final numerical answer.

---

### 8.4 Teacher API Prompt

The teacher prompt includes:

```text
Question:
How many more digits does the base-3 representation of $987_{10}$ have than the base-8 representation of $987_{10}$?

Candidate Role:
subtle_wrong

Candidate Solution:
We can easily do it in python. First, we need to create a function that changes base-10 numbers to other bases. Then we get base-N and base-M representations and calculate lengths difference.

Reference Correct Solution:
The base-3 representation has 7 digits and the base-8 representation has 4 digits. The difference is 7 - 4 = 3.

Ground-Truth Final Answer:
3
```

The teacher is asked to verify the candidate solution step by step and end with the fixed final verdict format.

---

### 8.5 Teacher Output

The teacher output is:

```text
<verification>
Step 1: The candidate proposes converting $987_{10}$ to two unspecified bases using Python, then calculating the difference in lengths.
Step 2: The problem specifically requires comparing the base-3 and base-8 representations. The candidate fails to specify that N=3 and M=8.
Step 3: The candidate describes a method but never executes it or states the final answer, which should be 7 - 4 = 3.
Earliest Error: The candidate uses variables N and M instead of the specific bases 3 and 8, and fails to provide the final numerical answer.
Verification: Is the answer correct (Yes/No)? No
</verification>
```

Parsed fields:

```json
{
  "candidate_role": "subtle_wrong",
  "candidate_is_correct": false,
  "parsed_final_verdict": "No",
  "earliest_error": "The candidate uses variables N and M instead of the specific bases 3 and 8, and fails to provide the final numerical answer."
}
```

This example passes the strict filter because the teacher verdict matches the original label and the rationale identifies a concrete error.

---

### 8.6 PPM SFT Example

After filtering, the same example is converted into PPM SFT format:

```json
{
  "id": "ppm_sft_example_base3_base8",
  "messages": [
    {
      "role": "user",
      "content": "Question:\nHow many more digits does the base-3 representation of $987_{10}$ have than the base-8 representation of $987_{10}$?\n\nCandidate Solution:\nWe can easily do it in python. First, we need to create a function that changes base-10 numbers to other bases. Then we get base-N and base-M representations and calculate lengths difference.\n\nPlease verify the candidate solution step by step."
    },
    {
      "role": "assistant",
      "content": "Step 1: The candidate proposes converting $987_{10}$ to two unspecified bases using Python.\nStep 2: The problem specifically requires comparing the base-3 and base-8 representations.\nStep 3: The candidate fails to specify N=3 and M=8 and never states the final numerical answer.\nEarliest Error: The candidate uses variables N and M instead of the specific bases 3 and 8.\nVerification: Is the answer correct (Yes/No)? No"
    }
  ],
  "metadata": {
    "source": "teacher_generated",
    "dataset": "openmathinstruct",
    "candidate_role": "subtle_wrong",
    "gold_is_correct": false,
    "teacher_model": "moonshotai/kimi-k2.5"
  }
}
```

---

### 8.7 Final Verifier Training Example

The final GenRM / EG-GenRM verifier uses the same input-target structure:

```text
Input:
Question + Candidate Solution

Target:
Verification rationale + final Yes/No verdict
```

This allows the final verifier to learn not only the answer label, but also the reasoning process needed to detect hidden mathematical mistakes.

---

## 9. Repository File Organization

Recommended repository structure for the data pipeline:

```text
project-root/
|
├── DATA.md
├── README.md
├── requirements.txt
├── environment.yml
├── .env.example
├── .gitignore
|
├── configs/
│   └── data_config.yaml
|
├── prompts/
│   ├── teacher_verifier_prompt.txt
│   └── ppm_generation_prompt.txt
|
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── examples/
│       ├── teacher_seed_sample.jsonl
│       ├── teacher_output_sample.jsonl
│       ├── clean_teacher_sample.jsonl
│       ├── public_candidate_sample.jsonl
│       ├── ppm_sft_sample.jsonl
│       ├── ppm_generated_sample.jsonl
│       └── final_train_sample.jsonl
|
├── notebooks/
│   └── teacherdata_clean.ipynb
|
└── scripts/
    └── data/
        ├── 00_download_public_data.py
        ├── 01_build_candidate_pool.py
        ├── 02_select_teacher_seed.py
        ├── 03_generate_teacher_rationales.py
        ├── 04_filter_teacher_outputs.py
        ├── 05_build_ppm_sft_dataset.py
        ├── 06_prepare_public_sources.py
        ├── 07_generate_ppm_rationales.py
        ├── 08_merge_final_dataset.py
        └── 09_quality_check_final_dataset.py
```

---

## 10. Script Mapping

| Script | Purpose | Main output |
|---|---|---|
| `00_download_public_data.py` | Download or load public math datasets | `data/raw/` |
| `01_build_candidate_pool.py` | Normalize raw data into candidate-solution records | `candidate_pool.jsonl` |
| `02_select_teacher_seed.py` | Select balanced teacher seed packs | `teacher_seed_full.jsonl` |
| `03_generate_teacher_rationales.py` | Call teacher API and save raw rationales | `teacher_outputs_all_candidates.jsonl` |
| `04_filter_teacher_outputs.py` | Filter teacher outputs by format and label consistency | `clean_teacher_data.jsonl` |
| `05_build_ppm_sft_dataset.py` | Convert clean teacher data into PPM SFT format | `ppm_sft_train.jsonl` |
| `06_prepare_public_sources.py` | Normalize additional public data | `public_candidate_pool.jsonl` |
| `07_generate_ppm_rationales.py` | Use trained PPM to generate rationales for public data | `ppm_generated_rationales.jsonl` |
| `08_merge_final_dataset.py` | Merge teacher, PPM-generated, and public data | `final_verifier_sft_train.jsonl` |
| `09_quality_check_final_dataset.py` | Check format, label balance, duplicates, and examples | quality report |

---

## 11. Quality Control

The final dataset should be checked before training.

Recommended checks:

```text
1. Every example has a non-empty question.
2. Every example has a non-empty candidate solution.
3. Every assistant target ends with the fixed final verdict line.
4. The final verdict can be parsed as Yes or No.
5. The Yes/No ratio is not extremely imbalanced.
6. Teacher-generated, PPM-generated, and public-source examples are marked in metadata.
7. Duplicate question-solution pairs are removed.
8. Very short or malformed rationales are removed.
9. A small sample is manually inspected.
```

Suggested final quality report fields:

```text
total examples
train examples
validation examples
Yes examples
No examples
teacher_generated examples
ppm_generated examples
public_label_only examples
average target length
number of malformed verdicts
number of duplicate examples
```

---

## 12. What Is Committed to GitHub

Full raw and processed datasets should not be committed to GitHub because they may be large.

Commit:

```text
DATA.md
README.md
configs/
prompts/
scripts/data/
data/examples/*.jsonl
notebooks/teacherdata_clean.ipynb if cleaned
```

Do not commit:

```text
.env
raw full datasets
large processed JSONL files
teacher API checkpoints
model checkpoints
API keys
```

The `.gitignore` should include:

```gitignore
.env
__pycache__/
*.pyc

data/raw/*
data/interim/*
data/processed/*

!data/raw/.gitkeep
!data/interim/.gitkeep
!data/processed/.gitkeep
!data/examples/
!data/examples/*.jsonl

outputs/
checkpoints/
wandb/
runs/
```

---

## 13. Current Status

The current uploaded teacher-generation file contains direct teacher-generated verification rationales for candidate solutions from NVIDIA/OpenMathInstruct-style data.

Current teacher file summary:

```text
File: teacher_outputs_all_candidates (38).jsonl
Total examples: 757
Status ok: 757
Unique normalized questions: 189
Correct candidates: 380
Incorrect candidates: 377
Parsed Yes/No verdicts: 724
Strict label-consistent examples: 663
```

This file is the raw teacher-generated supervision source. The next required steps are:

```text
1. remove API keys from the notebook;
2. convert the notebook into clean data scripts;
3. filter raw teacher outputs into clean_teacher_data.jsonl;
4. convert clean teacher data into ppm_sft_train.jsonl;
5. prepare public candidate data;
6. train PPM;
7. use PPM to generate teacher-like rationales for public data;
8. merge all sources into final_verifier_sft_train.jsonl;
9. run final quality checks.
```

