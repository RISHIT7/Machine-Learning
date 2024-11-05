# Oblique Decision Tree (ODT) Classifier

## Overview

This project implements an **Oblique Decision Tree (ODT) Classifier** in Python, capable of binary classification using logistic regression-based node splitting. The tree includes support for regularization, pruning, and custom training configurations, allowing flexible and interpretable decision-making models.

### Key Features

- **Oblique Splits**: Each split at a node is determined using linear combinations of features, allowing for more powerful decision boundaries than traditional axis-aligned splits.
- **Pruning**: Post-pruning capability to avoid overfitting by reducing the complexity of the tree.
- **Regularization**: Incorporates L1 and L2 regularization to handle different data characteristics.
- **Configurable Solvers**: Allows the use of different solvers such as `liblinear` and `lbfgs` for training the logistic regression model.
- **Customizable Parameters**: Supports a wide range of tunable hyperparameters including `max_depth`, `min_sample_split`, `C` (regularization strength), and `penalty`.

## Usage

To train, prune, and test the ODT model, run:

```bash
python your_script.py train_file.csv val_file.csv test_file.csv output_predictions.csv output_weights.txt
```

## Training and Pruning Strategy

- **Training Phase:** The tree is built recursively by selecting splits based on logistic regression projections. Each node's projection coefficients are calculated, and the optimal threshold is selected using Gini impurity or entropy.
- **Pruning Phase:** The tree is post-pruned using a validation set to compare the accuracy of a subtree with a pruned leaf node. If pruning improves or maintains accuracy, the subtree is replaced with a leaf.

## Code Improvements

### 1. Performance Optimizations

- **Solver Adjustments:** The liblinear solver showed better performance in handling smaller datasets with L1 regularization, achieving higher accuracy than lbfgs.
- **Efficient Threshold Search:** Optimized threshold search using linear projections and sorting.
- **Enhanced Post-Pruning:** Used bottom-up recursive post-pruning with a tie-breaking mechanism to avoid ambiguity in class assignments.

### 2. Customizable Regularization

- Implemented support for both L1 and L2 penalties to handle different sparsity levels and overfitting concerns:
    1) **L1 Regularization** achieved the highest test accuracy of **78.85%.**
    2) **L2 Regularization** provided stable results but with slightly lower accuracy.

### 3. Hyperparameter Tuning

- **Max Depth and Min Sample Split:** Experiments were conducted to find the optimal tree depth and minimum samples per split:
    1) Best test accuracy of **78.85%** was achieved with a max depth of `12` and a minimum sample split of `20`.
- Regularization Strength (`C`): Setting `C` to `0.1` provided the best generalization performance.

### 4. Polynomial Features

The inclusion of polynomial feature transformations resulted in an accuracy of approximately **77.15%**, indicating that non-linear interactions could be captured, albeit with diminished returns in accuracy.

## Experimentation Results

Below are the highlights from various experiments conducted:

| Tree Depth | Min Sample Split | Solver     | Loss Function | C   | Penalty | Test Accuracy (%) |
|------------|------------------|------------|---------------|-----|---------|--------------------|
| 12         | 20               | liblinear  | Gini          | 0.1 | L1      | 78.85              |
| 10         | 20               | liblinear  | Gini          | 0.1 | L1      | 78.85              |
| 12         | 20               | liblinear  | Gini          | 0.1 | L2      | 77.80              |
| 12         | 20               | lbfgs      | Gini          | 0.1 | L2      | 78.10              |

## Conclusion

The Oblique Decision Tree classifier implemented here demonstrates competitive performance with well-tuned hyperparameters and strategic pruning, achieving a test accuracy of **78.85%** under the best configuration.
