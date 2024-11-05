import cvxpy as cp
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import sys


def svm_trainer(data_path, output_id):
    # Load the dataset
    df = pd.read_csv(data_path)
    features = df.iloc[:, :-1].values  # Feature matrix
    labels = df.iloc[:, -1].values     # Target labels (last column)
    labels = np.where(labels == 0, -1, 1)  # Convert labels to -1 and 1

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Parameters for SVM
    num_samples, num_features = features_scaled.shape
    slack_cost = 1  # Regularization term

    # Define the optimization variables
    weights = cp.Variable(num_features)  # Weight vector
    intercept = cp.Variable()            # Bias term
    slacks = cp.Variable(num_samples)    # Slack variables for margin

    # Objective function: Minimize 0.5 * ||weights||_1 + slack_cost * sum(slacks)
    objective = cp.Minimize(0.5 * cp.norm1(weights) +
                            slack_cost * cp.sum(slacks))

    # Constraints for margin and non-negative slack
    margin = cp.multiply(labels, features_scaled @
                         weights + intercept) >= 1 - slacks
    slack_non_negative = slacks >= 0

    # Define and solve the optimization problem
    svm_problem = cp.Problem(objective, [margin, slack_non_negative])
    svm_problem.solve()

    # Determine linear separability based on slack values
    tolerance = 1e-4
    is_separable = int(np.all(slacks.value <= tolerance)
                       )  # 1 if separable, 0 otherwise

    # Support vectors: samples near the decision boundary
    support_vec_indices = []
    if is_separable:
        support_vec_indices = [idx for idx in range(num_samples)
                               if 1 - tolerance <= labels[idx] * (features_scaled[idx] @ weights.value + intercept.value) <= 1 + tolerance]

    # Prepare output data
    weights_data = {
        "weights": weights.value.tolist(),
        "bias": float(intercept.value)
    }

    separability_data = {
        "separable": is_separable,
        "support_vectors": support_vec_indices
    }

    # Save outputs to JSON files
    weights_filename = f"weights_{output_id}.json"
    separability_filename = f"sv_{output_id}.json"

    with open(weights_filename, "w") as f:
        json.dump(weights_data, f)

    with open(separability_filename, "w") as f:
        json.dump(separability_data, f)


def main():
    # Get input and output identifiers from the command-line arguments
    data_path = sys.argv[1]
    output_id = data_path.split("_")[-1].split(".")[0]
    svm_trainer(data_path, output_id)


# Execute if run as a script
if __name__ == "__main__":
    main()
