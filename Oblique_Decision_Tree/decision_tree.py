import numpy as np
from logistic_regression import logistic_regression
import sys


class TreeNode:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        coefficients=None,
        is_leaf=False,
        majority_label=None,
    ):
        """
        Initialize a node in the decision tree.
        Parameters:
            feature_index (int, optional): Index of the feature used for splitting the node. Defaults to None.
            threshold (float, optional): Threshold value for the feature to split the node. Defaults to None.
            left (Node, optional): Left child node. Defaults to None.
            right (Node, optional): Right child node. Defaults to None.
            coefficients (list, optional): Coefficients for the decision function. Defaults to None.
            is_leaf (bool, optional): Flag indicating if the node is a leaf node. Defaults to False.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.coefficients = coefficients
        self.is_leaf = is_leaf
        self.majority_label = majority_label


class ObliqueDecisionTree:
    def __init__(self, max_depth=8):
        """
        Initializes the DecisionTree with a specified maximum depth.
        Parameters:
            max_depth (int): The maximum depth of the decision tree. Default is 8.
        """
        self.root = None
        self.max_depth = max_depth

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the decision tree to the training data.
        Parameters:
            X (np.ndarray): The training data.
            y (np.ndarray): The target labels.

        Returns:
            None
        """
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        """
        Recursively builds the decision tree.
        Parameters:
            X (np.ndarray): The training data.
            y (np.ndarray): The target labels.
            depth (int): The current depth of the tree.

        Returns:
            TreeNode: The root node of the decision tree.
        """
        if depth >= self.max_depth or len(set(y)) == 1:
            majority_label = np.bincount(y.astype(int)).argmax()
            return TreeNode(is_leaf=True, majority_label=majority_label)

        coefficients = logistic_regression(X, y)
        projections = X @ coefficients
        sorted_indices = np.argsort(projections)
        sorted_projections = projections[sorted_indices]
        sorted_labels = y[sorted_indices]

        threshold = self._find_optimal_threshold(
            sorted_projections, sorted_labels)

        left_indices = projections <= threshold
        right_indices = ~left_indices

        left_child = self._build_tree(
            X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(
            X[right_indices], y[right_indices], depth + 1)

        return TreeNode(
            feature_index=None,
            threshold=threshold,
            left=left_child,
            right=right_child,
            coefficients=coefficients,
        )

    def _find_optimal_threshold(self, sorted_projections: np.ndarray, sorted_labels: np.ndarray) -> float:
        """
        Finds the optimal threshold for splitting the node.
        Parameters:
            sorted_projections (np.ndarray): The sorted projections of the training data.
            sorted_labels (np.ndarray): The sorted labels of the training data.

        Returns:
            float: The optimal threshold for splitting the node.
        """
        min_impurity = float("inf")
        best_threshold = None
        n = len(sorted_labels)

        for i in range(1, n):
            threshold = (sorted_projections[i] + sorted_projections[i - 1]) / 2

            left_indices = sorted_projections <= threshold
            right_indices = ~left_indices

            left_labels = sorted_labels[left_indices]
            right_labels = sorted_labels[right_indices]

            left_impurity = self._gini_impurity(left_labels)
            right_impurity = self._gini_impurity(right_labels)

            total_impurity = (
                len(left_labels) * left_impurity +
                len(right_labels) * right_impurity
            ) / n

            if total_impurity < min_impurity:
                min_impurity = total_impurity
                best_threshold = threshold

        return best_threshold

    def _gini_impurity(self, labels: np.ndarray) -> float:
        """
        Computes the Gini impurity of the labels.
        Parameters:
            labels (np.ndarray): The target labels.

        Returns:
            float: The Gini impurity of the labels.
        """
        if len(labels) == 0:
            return 0.0
        labels = labels.astype(int)
        class_counts = np.bincount(labels)
        probabilities = class_counts / len(labels)
        return 1 - np.sum(probabilities**2)

    def post_prune(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """
        Post-prunes the decision tree using the validation data.

        Parameters:
            X_val (np.ndarray): The validation data.
            y_val (np.ndarray): The target labels for the validation data.

        Returns:
            None
        """
        def _prune(node, X, y):
            if len(y) == 0:
                # incase where the val set is empty assign 0 to the error
                node.is_leaf = True
                node.majority_label = 0
                node.left = None
                node.right = None
                node.coefficients = None
                node.threshold = None
                return

            if node.is_leaf:
                return

            projections = X @ node.coefficients
            left_indices = projections <= node.threshold
            right_indices = ~left_indices

            if node.left:
                _prune(node.left, X[left_indices], y[left_indices])
            if node.right:
                _prune(node.right, X[right_indices], y[right_indices])

            # Bottom-up post-pruning using misclassification error, i.e., if the error of the leaf node is less than the error of the subtree, prune the subtree
            orig_pred = self.predict(X)
            orig_acc = evaluate_accuracy(y, orig_pred)

            node_backup = {
                "is_leaf": node.is_leaf,
                "majority_label": node.majority_label,
                "left": node.left,
                "right": node.right,
                "coefficients": node.coefficients,
                "threshold": node.threshold,
            }

            # Tie-breaking
            class_counts = np.bincount(y.astype(int), minlength=2)
            if class_counts[0] == class_counts[1]:
                majority_label = 0
            else:
                majority_label = np.argmax(class_counts)

            node.is_leaf = True
            node.majority_label = majority_label
            node.left = None
            node.right = None
            node.coefficients = None
            node.threshold = None

            leaf_pred = self.predict(X)
            leaf_acc = evaluate_accuracy(y, leaf_pred)

            if leaf_acc < orig_acc:
                node.is_leaf = node_backup["is_leaf"]
                node.majority_label = node_backup["majority_label"]
                node.left = node_backup["left"]
                node.right = node_backup["right"]
                node.coefficients = node_backup["coefficients"]
                node.threshold = node_backup["threshold"]

        _prune(self.root, X_val, y_val)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target labels for the input data.

        Parameters:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted target labels.
        """
        predictions = []
        for x in X:
            predictions.append(self._predict_recursive(self.root, x))
        return np.array(predictions)

    def _predict_recursive(self, node: TreeNode, x: np.ndarray) -> int:
        if node.is_leaf:
            return node.majority_label
        else:
            projection = x @ node.coefficients

            if projection <= node.threshold:
                return self._predict_recursive(node.left, x)
            else:
                return self._predict_recursive(node.right, x)

    def print_tree(self, node: TreeNode = None, depth: int = 0) -> None:
        """
        Prints the decision tree in a readable format.

        Parameters:
            node (TreeNode, optional): The current node in the decision tree. Defaults to None.
            depth (int, optional): The depth of the current node. Defaults to 0.

        Returns:
            None
        """
        if node is None:
            node = self.root

        if node.is_leaf:
            print(f"{' ' * depth}Leaf node: {node.majority_label}")
        else:
            print(
                f"{' ' * depth}Feature {node.feature_index} <= {node.threshold}")
            self.print_tree(node.left, depth + 1)
            self.print_tree(node.right, depth + 1)

    def __repr__(self):
        return f"ObliqueDecisionTree(max_depth={self.max_depth})"


class SaveWeights:
    def __init__(self, tree):
        """
        Initializes the SaveWeights class with the decision tree.

        Parameters:
            tree (ObliqueDecisionTree): The decision tree.

        Returns:
            None
        """
        self.tree = tree

    def save_weights_preorder(self, node: ObliqueDecisionTree = None, node_id: int = 1) -> list:
        """
        Saves the weights of the decision tree in a pre-order traversal.

        Parameters:
            node (ObliqueDecisionTree, optional): The current node in the decision tree. Defaults to None.
            node_id (int, optional): The ID of the current node. Defaults to 1.

        Returns:
            list: The list of weights for the decision tree.
        """
        if node is None:
            node = self.tree.root

        weights_list = []
        if node is not None:
            if node.coefficients is not None:
                weights_list.append(
                    f"{node_id},"
                    + ",".join(map(str, node.coefficients))
                    + f",{node.threshold}"
                )

            # Traverse left subtree
            if node.left:
                weights_list.extend(
                    self.save_weights_preorder(node.left, node_id * 2))
            # Traverse right subtree
            if node.right:
                weights_list.extend(
                    self.save_weights_preorder(node.right, node_id * 2 + 1)
                )

        return weights_list

    def save_to_file(self, file_path: str) -> None:
        """
        Saves the weights of the decision tree to a file.

        Parameters:
            file_path (str): The path to the file where the weights will be saved.

        Returns:
            None
        """
        weights_list = self.save_weights_preorder()
        with open(file_path, "w") as f:
            for line in weights_list:
                f.write(line + "\n")


def evaluate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Evaluates the accuracy of the model.

    Parameters:
        y_true (np.ndarray): The true target labels.
        y_pred (np.ndarray): The predicted target labels.

    Returns:
        float: The accuracy of the model.
    """
    return np.mean(y_true == y_pred)


def train_unpruned(
    train_file: str, max_depth: int, weight_file: str
) -> ObliqueDecisionTree:
    """
    Trains an unpruned decision tree.

    Parameters:
        train_file (str): The path to the training data file.
        max_depth (int): The maximum depth of the decision tree.
        weight_file (str): The path to the file where the weights will be saved.

    Returns:
        ObliqueDecisionTree: The trained decision tree.
    """
    data = np.loadtxt(train_file, delimiter=",", skiprows=1)
    X, y = data[:, :-1], data[:, -1]

    tree = ObliqueDecisionTree(max_depth=int(max_depth))
    tree.fit(X, y)

    weights_saver = SaveWeights(tree)
    weights_saver.save_to_file(weight_file)
    return tree


def train_pruned(
    train_file: str, val_file: str, max_depth: int, weight_file: str
) -> ObliqueDecisionTree:
    """
    Trains a pruned decision tree.

    Parameters:
        train_file (str): The path to the training data file.
        val_file (str): The path to the validation data file.
        max_depth (int): The maximum depth of the decision tree.
        weight_file (str): The path to the file where the weights will be saved.

    Returns:
        ObliqueDecisionTree: The trained decision tree.
    """
    data = np.loadtxt(train_file, delimiter=",", skiprows=1)
    X, y = data[:, :-1], data[:, -1]

    val_data = np.loadtxt(val_file, delimiter=",", skiprows=1)
    X_val, y_val = val_data[:, :-1], val_data[:, -1]

    tree = ObliqueDecisionTree(max_depth=int(max_depth))
    tree.fit(X, y)
    tree.post_prune(X_val, y_val)

    # For the test part, we don't need to save the weights
    if weight_file == "":
        return tree

    weights_saver = SaveWeights(tree)
    weights_saver.save_to_file(weight_file)
    return tree


def test(
    train_file: str, val_file: str, test_file: str, max_depth: int, output_file: str
) -> None:
    """
    Tests the decision tree on the test data.

    Parameters:
        train_file (str): The path to the training data file.
        val_file (str): The path to the validation data file.
        test_file (str): The path to the test data file.
        max_depth (int): The maximum depth of the decision tree.
        output_file (str): The path to the file where the predictions will be saved.

    Returns:
        None
    """
    test_data = np.loadtxt(test_file, delimiter=",", skiprows=1)
    # Contains featuers + target
    train_data = np.loadtxt(train_file, delimiter=",", skiprows=1)

    # If the test data has the target column
    if test_data.shape[1] == train_data.shape[1]:
        test_data = test_data[:, :-1]
    X_test = test_data

    tree = train_pruned(train_file, val_file, max_depth, "")

    predictions = tree.predict(X_test)

    np.savetxt(output_file, predictions, fmt="%d")


def main():
    mode = sys.argv[1]
    if mode == "train":
        pruning = sys.argv[2]
        if pruning == "pruned":
            train_pruned(sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
        elif pruning == "unpruned":
            train_unpruned(sys.argv[3], sys.argv[4], sys.argv[5])
    elif mode == "test":
        test(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])


if __name__ == "__main__":
    main()
