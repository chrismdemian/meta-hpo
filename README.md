# Neural Hyperparameter Optimization

A meta-learning system that predicts the accuracy of hyperparameter configurations, enabling faster hyperparameter search by replacing expensive training runs with instant neural network predictions.

## Overview

Hyperparameter tuning is one of the most time-consuming parts of training ML models. This project trains a neural network on 6.3 million real hyperparameter evaluations from the HPO-B benchmark to predict how well a given hyperparameter configuration will perform. Instead of running thousands of training experiments, users can evaluate configurations in milliseconds.

## How It Works

1. Feed in a hyperparameter configuration (learning rate, regularization, architecture choices, etc.) and a dataset identifier
2. The model instantly predicts the expected validation accuracy
3. Rank thousands of configurations and only train the top candidates

## Models

**Single search space model** - Trained on one algorithm type (491K evaluations). Achieves R2 = 0.93, MAE = 0.021 on held-out test data.

**Unified multi-task model** - One model across all 16 search spaces (6.3M evaluations) using per-search-space input projection layers and shared hidden layers. Achieves R2 = 0.84, MAE = 0.034. Demonstrates cross-algorithm transfer learning.

## Architecture

The unified model uses per-search-space input layers that project each algorithm's hyperparameters into a shared 32-dimensional embedding. This is concatenated with a dataset one-hot encoding and passed through shared hidden layers (256 neurons, ReLU, dropout). This lets one model handle algorithms with different numbers of hyperparameters (2 to 18) without zero-padding artifacts.

## Dataset

Uses the [HPO-B benchmark](https://github.com/machinelearningnuremberg/HPO-B) (NeurIPS 2021 Datasets and Benchmarks Track). 6.3 million hyperparameter evaluations across 16 search spaces and 101 datasets from OpenML.

## Project Structure

```
meta-hpo/
├── config.py               # All hyperparameters and settings
├── dataset.py              # Data loading for Meta-Album and HPO-B
├── model.py                # HPOModel (single) and UnifiedHPOModel (multi-task)
├── train.py                # Training for single search space
├── train_unified.py        # Training for unified model
├── validate.py             # Evaluation for single search space
├── validate_unified.py     # Evaluation for unified model
├── checkpoints/            # Saved model weights
├── hpo-data/               # HPO-B data (not tracked)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Download HPO-B data:
```python
from huggingface_hub import hf_hub_download
for f in ['meta-train-dataset.json', 'meta-test-dataset.json', 'meta-validation-dataset.json']:
    hf_hub_download(repo_id='Sebastianpinar/hpob-data', filename=f, repo_type='dataset', local_dir='hpo-data/hpob-data')
```

## Training

Single search space:
```bash
python train.py
python validate.py
```

Unified model (all search spaces):
```bash
python train_unified.py
python validate_unified.py
```

## Results

| Model | Data | R2 | MAE |
|-------|------|----|-----|
| Linear Regression (baseline) | Meta-Album 20K | 0.27 | 21.37 |
| Feedforward NN | Meta-Album 20K | 0.44 | 15.17 |
| Single Search Space NN | HPO-B 1M | 0.93 | 0.021 |
| Unified Multi-Task NN | HPO-B 6.3M | 0.84 | 0.034 |

## References

Pineda Arango, S., Jomaa, H., Wistuba, M., & Grabocka, J. (2021). HPO-B: A Large-Scale Reproducible Benchmark for Black-Box HPO based on OpenML. NeurIPS Datasets and Benchmarks Track.

## License

MIT
