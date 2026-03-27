# Deep Past Initiative: Akkadian to English NMT

A specialized Neural Machine Translation system designed for the **Deep Past Challenge**. The goal is to translate 4,000-year-old Old Assyrian (Akkadian) transliterations into modern English.

## Competition Overview

Akkadian is a low-resource, morphologically complex language. This project addresses the challenge by:
1.  **Preprocessing**: Cleaning transliterations (removing modern scribal notations like `!`, `?`, `[]`, `<>`).
2.  **Normalization**: Handling H-substitutions (`ḫ` -> `h`) and standardizing gaps (`<gap>`, `<big_gap>`).
3.  **Modeling**: Using a GRU-based Language Model optimized for low-resource translation.

## Repository Structure (Akkadian Version)

- `Akkadian_NMT_Submission.ipynb`: All-in-one Jupyter Notebook for Kaggle submissions.
- `preprocess_akkadian.py`: Script for cleaning and formatting raw transliterations.
- `run_akkadian.py`: Specialized training and submission generation script.
- `parameters_akkadian.py`: Hyperparameters optimized for the Akkadian dataset.

## How to Run Locally

### 1. Preprocess the Data
Prepare the raw CSV data for training:
```bash
python3 preprocess_akkadian.py
```

### 2. Prepare Binary Data
Generate vocabulary and training binary files:
```bash
python3 run_akkadian.py prepare
```

### 3. Training
Train the Akkadian NMT model:
```bash
python3 run_akkadian.py train
```

### 4. Generate Submission
Create `submission.csv` for Kaggle evaluation:
```bash
python3 run_akkadian.py submit
```

## Kaggle Integration

The provided `Akkadian_NMT_Submission.ipynb` is ready for deployment on Kaggle:
- **Auto-detection**: Automatically searches for `train.csv` and `test.csv` in `/kaggle/input`.
- **Pre-loaded logic**: Contains all preprocessing and model code in a single file.
- **Hardware**: Compatible with both Kaggle GPU and CPU notebooks.

## Evaluation Metric

Submissions are evaluated using the **Geometric Mean of BLEU and chrF++** scores.

---
*Developed for the Deep Past Initiative Challenge.*
