import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from argparse import ArgumentParser

def preprocess_text(text, stop_words):
    ps = PorterStemmer()
    words = text.lower().split()
    return [ps.stem(word) for word in words if word not in stop_words]

def load_data(file_path):
    df = pd.read_csv(file_path, header=None, sep='\t', quoting=3)
    return df[2].tolist(), df[1].tolist()

def load_stopwords(file_path):
    with open(file_path, 'r') as f:
        return f.read().splitlines()

class MultinomialNaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = None
        self.vocabulary = {}
        self.unk_idx = 0
        self.classes = None

    def build_vocabulary(self, X):
        """Builds vocabulary from training data and assigns an index to each word."""
        all_words = sorted(set(word for doc in X for word in doc))
        self.vocabulary = {word: idx for idx, word in enumerate(all_words)}
        self.unk_idx = len(self.vocabulary)

    def text_to_count_matrix(self, X):
        """Converts documents to a count matrix based on the vocabulary."""
        n_docs = len(X)
        vocab_size = len(self.vocabulary) + 1

        X_matrix = np.zeros((n_docs, vocab_size), dtype=np.int32)
        
        for i, doc in enumerate(X):
            for word in doc:
                idx = self.vocabulary.get(word, self.unk_idx)
                X_matrix[i, idx] += 1

        return X_matrix

    def fit(self, X, y):
        """Fit the model to the training data."""
        y = pd.Series(y)
        
        self.build_vocabulary(X)
        
        X_matrix = self.text_to_count_matrix(X)
        n_docs = X_matrix.shape[0]
        self.classes = y.unique()
        self.class_priors = y.value_counts() / n_docs
        
        vocab_size = len(self.vocabulary)
        self.feature_probs = np.zeros((len(self.classes), vocab_size+1))
        for idx, c in enumerate(self.classes):
            class_docs = X_matrix[(y == c)]
            word_counts = class_docs.sum(axis=0)
            # feature_prob => P(w|c) = (Nwc + 1) / (Nc + vocab_size)
            self.feature_probs[idx, :] = (word_counts + 1) / (class_docs.sum() + vocab_size)

    def predict(self, X):
        """Predicts the class for each document in X."""
        X_matrix = self.text_to_count_matrix(X)

        log_class_priors = np.log(np.array([self.class_priors[c] for c in self.classes]))
        log_feature_probs = np.log(self.feature_probs)

        log_likelihood = X_matrix @ log_feature_probs.T

        log_posteriors = log_likelihood + log_class_priors
        return log_posteriors

def main():
    parser = ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--stop", type=str, required=True)
    args = parser.parse_args()

    stop_words = load_stopwords(args.stop)

    X_train, y_train = load_data(args.train)
    X_test, _ = load_data(args.test)

    X_train = [preprocess_text(text, stop_words) for text in X_train]
    X_test = [preprocess_text(text, stop_words) for text in X_test]

    nb = MultinomialNaiveBayes()
    nb.fit(X_train, y_train)
    
    predictions = nb.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_pred_classes = np.array(nb.classes)[y_pred]

    with open(args.out, 'w') as f:
        for class_ in y_pred_classes:
            f.write(f'{class_}\n')

if __name__ == "__main__":
    main()
