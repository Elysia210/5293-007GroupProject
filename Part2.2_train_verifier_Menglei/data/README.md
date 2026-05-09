## Generalization Evaluation Dataset

The file `openmath_val_verifier_eval.jsonl` is used to test the generalization ability of the trained verifier.

This dataset is not used for training. Instead, it is use for evaluating to the different loss function from the baseline. 

The goal is to check whether the current verifier can correctly judge unseen candidate solutions and give some resoning in good formatting from OpenMathInstruct validation data.

Each example contains a math question, a candidate solution, the ground-truth final answer, and a gold correctness label. The verifier receives the question and candidate solution, then generates a step-by-step verification rationale and a final Yes/No judgment.

The model output is compared with the gold label:

```text
gold_verdict = Yes  → the candidate solution is correct
gold_verdict = No   → the candidate solution is incorrect
