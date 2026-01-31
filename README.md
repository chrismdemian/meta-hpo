# Neural Hyperparameter Optimization

A meta-learning approach to predicting optimal hyperparameters for machine learning models based on dataset characteristics.

## Overview

This project trains a neural network to recommend hyperparameters for ML models by learning patterns from thousands of previous training runs. Instead of expensive grid search or Bayesian optimization, the model predicts near-optimal configurations in a single forward pass.

## Approach

- **Input**: Dataset meta-features (size, dimensionality, class balance) + hyperparameter configuration
- **Output**: Predicted validation accuracy
- **Training Data**: HPO-B benchmark dataset (6.4M evaluations across 176 algorithms and 196 datasets)
- **Model**: Feedforward neural network with gradient descent optimization

## Dataset

Uses the [HPO-B meta-dataset](https://github.com/sebastianpinedaar/hpo-data), a large-scale benchmark for hyperparameter optimization research.

## Results

Comparison against random search baseline on held-out test datasets coming soon.

## License

MIT
