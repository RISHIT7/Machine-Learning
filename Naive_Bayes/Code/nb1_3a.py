import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from argparse import ArgumentParser

def create_bigrams(sequence: list) -> list:
    """Generate bigrams from a given sequence."""
    return ['_'.join(bigram) for bigram in zip(sequence[:-1], sequence[1:])]

def preprocess_text(text, stop_words):
    ps = PorterStemmer()
    words = text.lower().split()
    unigrams = [ps.stem(word) for word in words if word not in stop_words]
    bigrams = create_bigrams(unigrams)
    return unigrams + bigrams

def load_data(file_path):
    df = pd.read_csv(file_path, header=None, sep='\t', quoting=3)
    return df[2].tolist(), df[1].tolist()

def load_stopwords(file_path):
    with open(file_path, 'r') as f:
        return f.read().splitlines()

class BernoulliNaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = None
        self.vocabulary = {}
        self.unk_idx = 0
        self.classes = None

    def build_vocabulary(self, X):
        """Builds vocabulary from training data and assigns an index to each bigram."""
        all_ngrams = sorted(set(ngram for doc in X for ngram in doc))
        self.vocabulary = {ngram: idx for idx, ngram in enumerate(all_ngrams)}
        self.unk_idx = len(self.vocabulary)

    def text_to_onehot(self, X):
        """Converts documents to a one-hot encoded matrix based on the vocabulary of bigrams."""
        n_docs = len(X)
        vocab_size = len(self.vocabulary) + 1

        X_matrix = np.zeros((n_docs, vocab_size), dtype=np.int32)
        
        for i, doc in enumerate(X):
            for ngram in doc:
                idx = self.vocabulary.get(ngram, self.unk_idx)
                X_matrix[i, idx] = 1

        return X_matrix

    def fit(self, X, y):
        """Fit the model to the training data."""
        y = pd.Series(y)
        
        self.build_vocabulary(X)
        
        X_matrix = self.text_to_onehot(X)
        n_docs = X_matrix.shape[0]
        self.classes = y.unique()
        # class_prior => P(c) = Nc/N
        self.class_priors = y.value_counts() / n_docs
        
        vocab_size = len(self.vocabulary) + 1
        self.feature_probs = np.zeros((len(self.classes), vocab_size))
        for idx, c in enumerate(self.classes):
            class_docs = X_matrix[(y == c)]
            word_counts = class_docs.sum(axis=0)
            # feature_prob => P(w|c) = (Nwc + 1) / (Nc + 2)
            self.feature_probs[idx, :] = (word_counts + 1) / (class_docs.shape[0] + 2)

    def predict(self, X):
        """Predicts the class for each document in X."""
        X_matrix = self.text_to_onehot(X)

        log_class_priors = np.log(np.array([self.class_priors[c] for c in self.classes]))
        log_feature_probs = np.log(self.feature_probs)
        log_1_minus_feature_probs = np.log(1 - self.feature_probs)

        log_likelihood = (X_matrix @ log_feature_probs.T) + ((1 - X_matrix) @ log_1_minus_feature_probs.T)

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

    nb = BernoulliNaiveBayes()
    nb.fit(X_train, y_train)
    
    predictions = nb.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_pred_classes = np.array(nb.classes)[y_pred]

    with open(args.out, 'w') as f:
        for class_ in y_pred_classes:
            f.write(f'{class_}\n')

if __name__ == "__main__":
    main()
