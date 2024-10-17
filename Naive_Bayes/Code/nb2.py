import pandas as pd
import numpy as np
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
# import nltk
from argparse import ArgumentParser

# ----------------------------------------------------------------------------- Data Loading ----------------------------------------------------------------------------- #

def load_data(file_path):
    """
        Load data from a TSV file and return the text, labels, and numerical features.
    """
    df = pd.read_csv(file_path, header=None, sep='\t', quoting=3)
    return df[2].tolist(), df[1].tolist(), df.iloc[:, [3, 4, 7, 13]], df.iloc[:, [8, 9, 10, 11, 12]]

# ----------------------------------------------------------------------------- Preprocessing ----------------------------------------------------------------------------- #

# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

def preprocess_text(text, stop_words, ngram_range=2):
    """Preprocess text data by removing punctuation, numbers, and stopwords, and applying stemming."""
    ps = PorterStemmer()

    # Remove punctuation, numbers, and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize and stem words
    words = word_tokenize(text.lower())

    # Remove stop words and apply stemming
    unigrams = [ps.stem(word) for word in words if word not in stop_words]

    # Generate n-grams based on ngram_range
    ngrams = []
    for n in range(1, ngram_range + 1):
        ngrams += create_ngrams(unigrams, n)

    return ngrams

def create_ngrams(sequence, n):
    """Create n-grams from a sequence of words."""
    return ['_'.join(sequence[i:i+n]) for i in range(len(sequence)-n+1)]

def flatten_row(row):
    """Flatten a list within a row."""
    flattened_row = []
    for item in row:
        if isinstance(item, list):
            # Add each item in the list separately
            flattened_row.extend(item)  
        else:
            # Keep the original item
            flattened_row.append(item)  
    return flattened_row

# ----------------------------------------------------------------------------- Models ----------------------------------------------------------------------------- #

class MultinomialNaiveBayes:
    """
        Multinomial Naive Bayes classifier with Laplace smoothing.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_priors = {}
        self.feature_probs = None
        self.vocabulary = {}
        self.unk_idx = 0
        self.classes = None

    def build_vocabulary(self, X):
        """Build vocabulary using Counter for efficient counting."""
        all_ngrams = Counter(ngram for doc in X for ngram in doc)
        self.vocabulary = {ngram: idx for idx, ngram in enumerate(all_ngrams)}
        self.unk_idx = len(self.vocabulary)

    def text_to_count_matrix(self, X):
        """Convert text to count matrix (optimized)."""
        n_docs = len(X)
        vocab_size = len(self.vocabulary) + 1

        X_matrix = np.zeros((n_docs, vocab_size), dtype=np.int32)

        for i, doc in enumerate(X):
            for ngram in doc:
                idx = self.vocabulary.get(ngram, self.unk_idx)
                X_matrix[i, idx] += 1

        return X_matrix

    def fit(self, X, y):
        """Fit the model and precompute log probabilities."""
        y = pd.Series(y)

        self.build_vocabulary(X)

        X_matrix = self.text_to_count_matrix(X)
        n_docs = X_matrix.shape[0]
        self.classes = y.unique()
        self.class_priors = y.value_counts() / n_docs

        vocab_size = len(self.vocabulary)
        self.feature_probs = np.zeros((len(self.classes), vocab_size + 1))

        for idx, c in enumerate(self.classes):
            class_docs = X_matrix[y == c]
            word_counts = class_docs.sum(axis=0)

            self.feature_probs[idx, :] = (word_counts + self.alpha) / (class_docs.sum() + vocab_size * self.alpha)

        self.log_feature_probs = np.log(self.feature_probs)
        self.log_class_priors = np.log(np.array([self.class_priors[c] for c in self.classes]))

    def predict(self, X):
        """Predict the class for each document."""
        X_matrix = self.text_to_count_matrix(X)

        # Calculate log-likelihood and posteriors
        log_likelihood = X_matrix @ self.log_feature_probs.T
        log_posteriors = log_likelihood + self.log_class_priors

        return np.argmax(log_posteriors, axis=1)

class MulticlassLogisticRegression:
    """
        Multiclass Logistic Regression classifier with softmax activation.
    """
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cost_function(self, X, y, m):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(z)
        return -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))

    def gradient_descent(self, X, y, m):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.softmax(z)
        dw = np.dot(X.T, (y_pred - y)) / m
        db = (y_pred - y).sum(axis=0).values / m
        db = db.reshape(1, -1)

        # Update weights and bias
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def fit(self, X_train, y_train):
        m, n = X_train.shape
        k = y_train.shape[1]
        self.weights = np.zeros((n, k))
        self.bias = np.zeros((1, k))

        for _ in range(self.num_iterations):
            # Perform gradient descent on training data
            self.gradient_descent(X_train, y_train, m)

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return np.argmax(self.softmax(z), axis=1)
    
    def parameters(self):
        return self.weights, self.bias

# ----------------------------------------------------------------------------- Evaluation ----------------------------------------------------------------------------- #

def accuracy(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1
    return correct / len(y_true)

# ----------------------------------------------------------------------------- Main ----------------------------------------------------------------------------- #

def main():
    
    # ----------------------------------------------------------------------------- Argument Parsing ----------------------------------------------------------------------------- #
    
    parser = ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--stop", type=str, default='english', help="Stopwords language")
    args = parser.parse_args()

    # ----------------------------------------------------------------------------- Naive Bayes ----------------------------------------------------------------------------- #

    # Set n-gram range
    _ = 3

    # Load NLTK stopwords
    _ = set(stopwords.words('english'))

    # Load data
    _, y_train, _, X_train_numerical = load_data(args.train)
    _, y_test, _, X_test_numerical = load_data(args.test)

    # # Flatten and process text columns for n-grams
    # train_text_cols = train_text_cols.apply(flatten_row, axis=1).values.tolist()
    # test_text_cols = test_text_cols.apply(flatten_row, axis=1).values.tolist()

    # X_train = [X_train[i] + ' ' + ' '.join(str(item) for item in flatten_row(row)) for i, row in enumerate(train_text_cols)]
    # X_test = [X_test[i] + ' ' + ' '.join(str(item) for item in flatten_row(row)) for i, row in enumerate(test_text_cols)]

    # X_train = [preprocess_text(text, stop_words, ngram_range=n) for text in X_train]
    # X_test = [preprocess_text(text, stop_words, ngram_range=n) for text in X_test]

    # # Initialize and fit Naive Bayes
    # nb = MultinomialNaiveBayes(alpha=np.expm1((1)))
    # nb.fit(X_train, y_train)
    # print("Naive Bayes model trained successfully.")
    # print("-"*50)

    # # Train accuracy
    # y_pred_train = np.array(nb.classes)[nb.predict(X_train)]
    # print(f"Train Accuracy: {accuracy(y_train, y_pred_train)}")

    # # Predict on test set
    # y_pred_test = np.array(nb.classes)[nb.predict(X_test)]
    # print(f"Test Accuracy: {accuracy(y_test, y_pred_test)}")
    # print("-"*50)

    # ----------------------------------------------------------------------------- Multiclass Logistic Regression ----------------------------------------------------------------------------- #
    
    # Normalize numerical features
    X_train_norm = (X_train_numerical - X_train_numerical.mean()) / X_train_numerical.std()
    X_test_norm = (X_test_numerical - X_train_numerical.mean()) / X_train_numerical.std()
    # print("Numerical features normalized successfully.")

    # Train and predict using Multiclass Logistic Regression as well
    y_train_one_hot = pd.get_dummies(y_train, dtype=int)
    y_test_one_hot = pd.get_dummies(y_test, dtype=int)
    
    model = MulticlassLogisticRegression(learning_rate=np.expm1(1)-1, num_iterations=2000)
    model.fit(X_train_norm, y_train_one_hot)
    # print("Multiclass Logistic Regression model trained successfully.")
    
    # print("-"*50)
    # y_pred_train = y_train_one_hot.columns[model.predict(X_train_norm)]
    # print(f"Train Accuracy: {accuracy(y_train, y_pred_train)}")
    
    y_pred_test = y_test_one_hot.columns[model.predict(X_test_norm)]
    # print(f"Test Accuracy: {accuracy(y_test, y_pred_test)}")
    # print("-"*50)
    
    # ---------------------------------------------------------------------------- Save the predications ----------------------------------------------------------------------------------------- #
    # Save the predictions to the output file
    with open(args.out, 'w') as f:
        for prediction in y_pred_test:
            f.write(f"{prediction}\n")

if __name__ == "__main__":
    main()
