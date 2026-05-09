
# Data Processing Scripts

This folder contains the main data-processing scripts for the EG-GenRM data pipeline.

The goal of these scripts is to convert raw candidate solutions and teacher-generated outputs into clean supervised fine-tuning data for training a generative mathematical verifier.

## Script Overview

| Script | Purpose |
|---|---|
| `03_generate_teacher_rationales.py` | Calls the teacher API to generate step-by-step verification rationales for selected candidate solutions. |
| `04_filter_teacher_outputs.py` | Filters raw teacher outputs and keeps only valid, label-consistent, high-quality verification rationales. |
| `05_build_ppm_sft_dataset.py` | Converts the filtered teacher data into chat-style SFT format for training the PPM. |

---

## Pipeline Order

The scripts should be run in the following order:

```bash
python scripts/data/03_generate_teacher_rationales.py --config configs/data_config.yaml
python scripts/data/04_filter_teacher_outputs.py --config configs/data_config.yaml
python scripts/data/05_build_ppm_sft_dataset.py --config configs/data_config.yaml
