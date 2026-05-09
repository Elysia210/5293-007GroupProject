# Part 3: Fine-tuning and Evaluation

Owner: Ruimin Zhang (`rz2737`). Part of the team EG-GenRM project (see the top-level repo `README.md` for project context and team overview).

This part takes the candidate file produced by Part 2 (Menglei) and the baseline outputs produced by Part 1 (Hantian), fine-tunes two QLoRA verifiers on Qwen2.5-1.5B-Instruct, and runs the unified evaluation that goes into the final report. Both verifiers output a Yes/No verdict; the difference is just the loss function.

Method A is plain QLoRA with standard cross-entropy on the verdict token. Method B (EG-GenRM) is the same setup but the loss only flows through the top-20% highest-entropy tokens at each step. The motivation, set out in our proposal, is to treat high-entropy tokens as likely "logical fork" points where the model is genuinely uncertain about the next reasoning decision, while low-entropy tokens are mostly templates and connectors that already predict cleanly. Masking out the easy tokens concentrates updates on the forks.

The teacher (Kimi-K2.5 + DeepSeek-V3.2) already gets 91.5% candidate-level accuracy with 97.1% recall, but it produces 50 false positives. Teacher-naive selection drops to 89.2%, worse than plain Majority Voting at 93.5%, so the actual job is to combine high recall with a selection rule that survives those FPs. Method B reaches 100% on the 38-question held-out test set, vs 94.7% for Majority Voting and 97.4% for Method A.

## Folder layout

```
Part3_Finetuning_Eval_Ruimin/
├── README.md                                   (this file)
├── EG_GenRM_finetuning_evaluation.ipynb        (the notebook)
├── teacher_outputs_all_candidates (38).jsonl   (input from Part 2 - NOT INCLUDED IN REPO)
├── baseline/                                   (input from Part 1 - NOT INCLUDED IN REPO)
│   ├── gsm8k_k5_run/direct_live.jsonl
│   └── math500_k5_run/
│       ├── direct_live.jsonl
│       ├── candidate_paths.jsonl
│       └── cleaned_samples.jsonl
├── outputs/                                    (LoRA adapters, written at runtime - NOT INCLUDED IN REPO)
├── results/                                    (CSVs, included in repo - generated results)
└── figures/                                    (PNGs, included in repo - generated figures)
```

**Note:** Large data files (baseline datasets, model outputs, and teacher outputs) are not included in this repository due to GitHub file size limits. To run the notebook:

1. Obtain baseline data from Part 1 (Hantian) and place in `baseline/` directory
2. Obtain teacher outputs from Part 2 (Menglei) and place as `teacher_outputs_all_candidates (38).jsonl`
3. Run the notebook - it will generate `outputs/` directory
4. `results/` and `figures/` are included in the repo as they contain the final analysis outputs

## Concepts

Some terms used in the notebook:

`question_norm` is the normalized question string we use to group candidates. The dataset has four candidates per question with `candidate_role` set to `correct_1`, `correct_2`, `subtle_wrong`, or `random_wrong`. A question with all four roles is a "complete group".

GenRM (generative reward model) means the verifier writes its judgement as text (`<final_verdict>Yes</final_verdict>`) instead of outputting a scalar score. We can then SFT it with the model's own next-token-prediction head.

QLoRA loads the base weights in 4-bit and adds a small low-rank adapter that is the only thing trained. Around 9.2M of 1.55B parameters (0.59%) get gradients, which is what lets us fit Qwen2.5-1.5B on a 24GB GPU.

Group-aware split means train/val/test are split by `question_norm`, so all four candidates of one question stay in one split. Without this, the verifier could memorize a question through one candidate and be tested on another candidate of the same question, inflating numbers.

Predictive entropy is `H_t = -Σ_v p_v log p_v` over the vocab at position t. High entropy roughly tracks the model being uncertain about the next token, which empirically lines up with logical decision points in a chain of reasoning.

## Setup

Hardware: a GPU with at least 24GB. Tested on Colab A100-40GB. End-to-end runtime is around 40 minutes; the two QLoRA training runs take about 15 minutes each.

Software: the first cell of the notebook installs everything inline. Alternatively, the top-level `requirements.txt` covers the same packages:

```bash
pip install -r ../requirements.txt
```

Tested with Python 3.10, torch 2.1+, CUDA 12.4. Qwen2.5-1.5B-Instruct downloads from Hugging Face on first run, so set `HF_TOKEN` if you hit anonymous-download rate limits.

## Inputs needed

Each record in `teacher_outputs_all_candidates (38).jsonl` has at minimum these fields: `question_norm`, `candidate_role`, `candidate_solution_clean` (or `_raw`), `candidate_predicted_answer`, `expected_answer`, `candidate_is_correct`, `parsed_final_verdict` (Yes / No / None), `teacher_model`, `earliest_error`, `run_id`.

After dedup on `question_norm` (keep the run with the most complete role coverage; ties broken by global `run_id` frequency), we get 751 deduplicated records from 189 unique questions, of which 186 have all four roles (744 candidates). The full 186-group set is used for heuristic / teacher selection evaluation. For SFT we keep all 751 deduplicated records and apply a group-aware ~60/20/20 split by `question_norm` (`GroupShuffleSplit(test_size=0.40)` then `GroupShuffleSplit(test_size=0.50)`):

| Split | Questions | Candidates |
|---|---|---|
| train | 113 | 451 |
| val   | 38  | 148 |
| test  | 38  | 152 |

## Configuration

Paths and key hyperparameters live in `PROJECT_CONFIG` at the top of Part 0. Set `project_dir` to this Part 3 folder; everything else is relative to that and can be left at the defaults to reproduce the reported numbers.

```python
PROJECT_CONFIG = {
    "project_dir":   "/content/drive/MyDrive/.../5293-007GroupProject/Part3_Finetuning_Eval_Ruimin",
    "json_path":     "teacher_outputs_all_candidates (38).jsonl",
    "baseline_dir":  "baseline/",
    "figures_dir":   "figures/",
    "results_dir":   "results/",
    "model_name":    "Qwen/Qwen2.5-1.5B-Instruct",
    "max_len":       1024,
    "entropy_top_k": 0.20,
}
```

`figures/`, `results/`, and `outputs/` are auto-created if missing.

## How to run

1. Open `EG_GenRM_finetuning_evaluation.ipynb` in Colab (or local Jupyter with CUDA 11.8+).
2. Run the first cell to install packages.
3. Mount Drive (Colab) or update `PROJECT_CONFIG["project_dir"]` (local) so it points at this Part 3 folder.
4. Run all cells.

The 13 parts:

| Part | What it does |
|---|---|
| 0  | env, install, `PROJECT_CONFIG` |
| 1  | research questions, pipeline overview |
| 2  | load teacher data, dedup → 186 complete groups, integrate Part 1 baselines |
| 3  | heuristic baselines (random, majority, oracle) |
| 4  | teacher GenRM candidate-level metrics + per-role breakdown |
| 5  | inference-time selection-strategy comparison on the full 186q set |
| 6  | reasoning-error taxonomy + FP/FN case study |
| 7  | build SFT prompts, group-aware ~60/20/20 split |
| 8  | Method A: standard QLoRA verifier |
| 9  | Method B: EG-GenRM with `EntropyMaskedTrainer` |
| 10 | same-split (n=38) head-to-head of all strategies |
| 11 | paired bootstrap CI + error-transition matrix |
| 12 | verifier-call efficiency |
| 13 | conclusions, future work |

Random sources used in the data split, evaluation sampling, and bootstrap are seeded with `random.seed(42)`, `np.random.seed(42)`, `GroupShuffleSplit(random_state=42)`, and `np.random.default_rng(0)`. Small numerical differences may still occur across GPU environments, library versions, and training runs because full CUDA-level determinism is not enforced.

## Generated outputs

The notebook writes results into this folder:

```
results/
├── strategy_comparison.csv
├── same_split_comparison.csv
├── ablation_results_final.csv
├── verifier_per_role_metrics.csv
├── error_taxonomy.csv
├── failure_cases_FP.csv
├── failure_cases_FN.csv
├── error_transition_A_vs_B.csv
└── efficiency_summary.csv

figures/
├── strategy_comparison.png
├── same_split_comparison.png
├── confusion_matrix_teacher.png
├── confusion_matrix_A_vs_B.png
├── bootstrap_ci_A_vs_B.png
├── error_transition_A_to_B.png
└── error_taxonomy_by_role.png

outputs/
├── verifier_lora_qwen15b_verdict_only/   (Method A LoRA adapter)
└── verifier_lora_eg_genrm/               (Method B LoRA adapter)
```

Plus the per-candidate verdict files `finetuned_verifier_predictions_verdict_only.jsonl` (A) and `finetuned_verifier_predictions_eg_genrm.jsonl` (B), saved at the project root.

## Method

Both methods share the same base, quantization, LoRA config, training data, splits, optimizer, lr, and epoch count. Only the loss differs.

| | |
|---|---|
| Base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Quantization | 4-bit NF4 (BitsAndBytes), double-quant, fp16 compute |
| LoRA target | q/k/v/o/gate/up/down proj |
| LoRA r/α/dropout | 8 / 16 / 0.05 |
| Trainable params | 9.2M / 1.55B (0.59%) |
| Epochs | 3 (171 steps total) |
| LR | 2e-4, warmup ratio 0.05 |
| Effective batch | 1 × 8 grad accum |
| Max seq len | 1024 |

The SFT label is `candidate_is_correct` (ground truth), not the teacher's `parsed_final_verdict`. Teacher verdicts are used separately as a baseline and as a teacher-filtered inference-time selection signal, but the fine-tuned verifiers learn from the actual correctness label.

The SFT prompt looks like this. The expected_answer is intentionally not in the prompt so the verdict cannot be inferred from a label leak.

```
You are a generative verifier for mathematical reasoning.

Question:
{question}

Candidate reasoning:
{candidate_solution[:3000]}

Candidate final answer: {candidate_predicted_answer}

Task: Decide whether the candidate reasoning and final answer are correct.
Return ONLY one line. Do not explain. Do not write reasoning.
Use exactly one of the following two outputs:
<final_verdict>Yes</final_verdict>
<final_verdict>No</final_verdict>
```

Completion is the verdict line. The verdict is parsed back out with a `<final_verdict>...</final_verdict>` regex, falling back to bare `\byes\b` / `\bno\b`.

Method A: standard cross-entropy on the completion tokens. Prompt tokens are masked with `-100`.

Method B: same loss, masked to the top 20% highest-entropy tokens computed online from the verifier's own logits at each step:

```
H_t        = -Σ_v softmax(logits_t)_v · log_softmax(logits_t)_v
threshold  = quantile(H_t over completion tokens, 1 - entropy_top_k)
M_t        = (H_t ≥ threshold) AND (label_t ≠ -100)
L_EG-GenRM = -Σ_t M_t · log P(y_t | x, y_<t) / Σ_t M_t
```

Implemented as `EntropyMaskedTrainer` overriding `compute_loss()`. The threshold is recomputed every step.

At inference the verifier emits Yes/No on each of the four candidates per question. Three rules for picking a final answer from the four verdicts:

- Naive: random pick from the Yes pool; if empty, random over all four.
- Filtered Majority: most common candidate answer in the Yes pool, tie-break by global support across all four.
- Conflict-Guarded Majority (used for Method B): same as filtered majority, but if the Yes-pool majority disagrees with the global majority and the global majority has at least 2 votes plus one Yes-marked supporter, pick the global majority. This is a safety rule for cases where verifier FPs would form a wrong majority within the Yes pool. On this test split the guard didn't actually activate (all 38 selections went through the regular Yes-majority path), so Method B's 100% comes from the verifier itself rather than from the rule kicking in.

## Results

Teacher GenRM baseline on 718 valid verdicts: 91.5% acc, 87.9% precision, 97.1% recall, F1 0.922. Per-role: 97.3% / 96.8% on correct_1 / correct_2 (almost no false negatives on real correct candidates), but 50 FPs concentrated on random_wrong (34) and subtle_wrong (16). This is the noise the fine-tuned verifier has to absorb.

Candidate-level fine-tuned metrics on the test split (n=152):

| Method | Acc | P | R | F1 |
|---|---:|---:|---:|---:|
| Method A | 72.4% | 66.7% | 94.7% | 0.783 |
| Method B | 73.0% | 66.4% | 98.7% | 0.794 |

Method B improves recall from 94.7% to 98.7% with similar precision. The error-transition matrix in Part 11 shows B fixes 3 of A's mistakes and breaks 2 of A's correct calls (net +1 candidate out of 150). The recall jump matters at selection time because the Yes pool more reliably contains a correct candidate.

Strategy comparison on the full 186 complete groups:

| Strategy | Accuracy |
|---|---:|
| Random | 50.0% |
| Majority Voting | 93.5% |
| Teacher GenRM (naive) | 89.2% |
| Teacher-filtered Majority | 98.4% |
| Best-of-N Oracle (upper bound) | 100.0% |

Teacher naive (89.2%) is actually worse than plain Majority Voting (93.5%): teacher FPs make the teacher a bad direct picker. Use it as a filter, not a selector.

Same-split (n=38) head-to-head with the fine-tuned verifiers:

| Strategy | Accuracy |
|---|---:|
| Random | 50.2% |
| Majority Voting | 94.7% |
| Teacher GenRM (naive) | 92.1% |
| Teacher-filtered Majority | 100.0% |
| Method A filtered majority | 97.4% |
| Method B EG-GenRM guarded majority | 100.0% |
| Best-of-N Oracle | 100.0% |

Paired bootstrap (2000 iterations) on Method B − Method A: +2.6pp, 95% CI [0.0%, 7.9%], one-sided p = 0.370. Positive but inconclusive at n=38; we don't claim significance.

Verifier-call cost: on the 38-question test split, both Method A and Method B run the verifier on all 4 candidates per question, so each is 152 calls (4 calls/question) on the test set. Teacher GenRM annotation in Part 2 covered the full 186-question complete-group set at 744 calls (4 calls/question). Per-question rate is the same; the totals only differ because the scopes differ. We don't claim a 5× call reduction.

## Limitations

The 186q table and the 38q table are not directly comparable. Heuristic baselines and the teacher GenRM run on the full 186 complete groups, while the fine-tuned verifiers can only run on the 38q test split because the other 148 questions are train+val. We report the two scopes separately and don't cross-compare.

n=38 is small enough that one question moves selection accuracy by about 2.6pp. The bootstrap CI for B−A includes 0.0%, so the improvement is suggestive rather than significant. A larger evaluation (full MATH500 or AIME) would be needed for a stronger claim.

All numbers are on Qwen2.5-1.5B-Instruct only. We don't claim entropy masking transfers to larger or differently-trained verifiers.

The training labels come from `candidate_is_correct` (gold), which is reliable, but the teacher GenRM baseline numbers we compare against are produced by an LLM and the teacher itself is imperfect: Part 4 shows non-trivial teacher disagreement on subtle_wrong and random_wrong candidates.

The Part 1 baselines run on the full GSM8K (1000q) and MATH500 (500q × k=5), while teacher and fine-tuned methods only run on the 186q teacher-annotated subset. The two scopes are reported separately.

The conflict-guard inference rule didn't activate on this 38-question test split. Method B's 100% should be read as "EG-GenRM verifier + filtered-majority selection" rather than evidence that the guard rule itself was useful here. Whether the guard helps on harder splits is open.

## Contribution

This part builds the SFT dataset on top of the upstream candidate annotations, trains both the standard QLoRA baseline and the entropy-masked EG-GenRM verifier under one shared config, and runs the same-split evaluation that integrates baselines, teacher outputs, fine-tuned verifiers, and the selection rules. The CSVs and figures it produces are what go into the final report and slides.