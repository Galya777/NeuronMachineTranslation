# Neural Machine Translation (English-Bulgarian)

This project implements a Neural Machine Translation (NMT) system focusing on translating between English and Bulgarian using a generative language model architecture.

## Overview

The system utilizes a Gated Recurrent Unit (GRU) based architecture to model the joint distribution of source and target sequences. By concatenating the source and target sentences into a single sequence formatted as `<S> Source <TRANS> Target </S>`, the model learns to generate translations by predicting the next token in the sequence.

## Architecture

- **Embedding Layer**: Projects vocabulary indices into a dense vector space (default: 256 dimensions).
- **GRU RNN**: A single-layer Gated Recurrent Unit (default: 512 hidden units) that processes the sequence.
- **Linear Projection**: Maps the RNN hidden state back to the vocabulary space.
- **Training Objective**: Minimizes the Negative Log-Likelihood (NLL) of the combined sequence.

## File Structure

- `model.py`: Core architecture of the `LanguageModel`.
- `run.py`: Main CLI script for data preparation, training, and translation.
- `utils.py`: Helper functions for data loading, tokenization, and progress monitoring.
- `parameters.py`: Hyperparameter configuration and file paths.
- `en_bg_data/`: Directory containing training and development datasets.

## How to Run

### 1. Data Preparation
Prepare the binary corpora and vocabulary:
```bash
python3 run.py prepare
```

### 2. Training
Start training the model:
```bash
python3 run.py train
```
The model checkpoints are saved as `NMTmodel` and `NMTmodel.optim`.

### 3. Translation
Translate a file using the trained model:
```bash
python3 run.py translate input_file output_file
```

### 4. Evaluation (BLEU)
Evaluate the translations using the BLEU metric:
```bash
python3 run.py bleu reference_file translation_file
```

## Dependencies

- Python 3.8+
- PyTorch
- NLTK (for tokenization and BLEU scoring)
- NumPy
