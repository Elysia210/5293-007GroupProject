# EG-GenRM: Entropy-Guided Generative Reward Model

Team: Hantian Zhang (`hz3101`), Menglei Zhang (`mz3129`), Ruimin Zhang (`rz2737`).

<img width="1672" height="941" alt="image" src="https://github.com/user-attachments/assets/4bddc92f-806d-461d-8235-aa391f55e9a4" />


The project is about reasoning hallucinations in math problem solving. We train a small LM as a generative verifier (it writes Yes or No and give resons of why on candidate solutions),

We attempt to use entropy to guide the training loss of the verifier, increasing focus on the reasoning nodes where the model is uncertain.

Code is split across three stages, one per teammate, each documented in its own section below.

This page briefly introduces the overall task. Please navigate to the README in each part's folder for specific code explanations instruction and structural details.
<img width="1902" height="78" alt="image" src="https://github.com/user-attachments/assets/b717a7b7-a13c-489f-a3d6-f17fa2b52c60" />

---

## Stage 1: Data Preparation and Baselines

Owner: Hantian Zhang.

* **Part 1:**
This stage prepares the GSM8K and MATH500 baseline outputs that the downstream evaluation reads. It produces direct-answer (no-CoT) predictions on both benchmarks, k=5 chain-of-thought candidate paths on MATH500, and the cleaned reference answers used to score them.


---

## Stage 2: Train data Generation and training with entropy-guided perfomance analysis
**Owner:** Menglei Zhang.

* **Part 2.1:** Introduces a novel, low-cost method to generate high-quality data for training the generative mathematical reasoning verifier.
* **Part 2.2:** Covers the verifier's training phase, featuring various experiments on entropy-guided model reasoning, along with an analysis of success and failure cases.

---

## Stage 3: Fine-tuning and Evaluation

Owner: Ruimin Zhang.

* **Part 3:**
This stage takes the Stage 2 candidate file (which contains candidate solutions, ground-truth correctness labels, and teacher-verifier annotations) and the Stage 1 baseline outputs, fine-tunes two QLoRA verifiers on Qwen2.5-1.5B-Instruct, and runs the evaluation that goes into the final report. Both verifiers output a Yes/No verdict; the difference is just the loss function.

When reproducing Part 3 locally, place the baseline files consumed by the Part 3 evaluation notebook under `Part3_Finetuning_Eval_Ruimin/baseline/`.

---
