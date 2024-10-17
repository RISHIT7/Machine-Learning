import sys
import pandas as pd
import numpy as np
import scipy as sp

class PartA:
    def __init__(self):
        self.train = str()
        self.test = str()
        self.sample_weights = str()
        self.model_predictions = str()
        self.model_weights = str()

    def parse_a(self) -> None:
        self.train = sys.argv[2]
        self.test = sys.argv[3]
        self.sample_weights = sys.argv[4]
        self.model_predictions = sys.argv[5]
        self.model_weights = sys.argv[6]
    
    def load_data(self, mode = "train") -> tuple[pd.DataFrame, pd.Series]:
        if mode == "train":
            df = pd.read_csv(self.train)
            X = df.drop(labels = ['Total Costs'], axis = 1)
            y = df['Total Costs']
            X = pd.concat([pd.DataFrame(np.ones((X.shape[0], 1)), columns=['Bias']), X], axis=1)
            return X, y
        else:
            X = pd.read_csv(self.test)
            X = pd.concat([pd.DataFrame(np.ones((X.shape[0], 1)), columns=['Bias']), X], axis=1)
            return X, None
    
    def fit(self, X, y) -> np.ndarray:
        U = pd.read_csv(self.sample_weights, header = None).values
        
        n_samples, n_features = X.shape
        y = (y.values).reshape(-1, 1)
        block_size = 1000

        W = np.zeros((n_features, n_features))

        for i in range(n_samples//block_size):
            start = i*block_size
            end = (i+1)*block_size if i < n_samples//block_size - 1 else n_samples
            X_chunk = X[start:end]
            U_chunk = U[start:end]
            
            W += (X_chunk.T @ (U_chunk * X_chunk))

        W_inv = np.linalg.inv(W)
        
        Wy = np.zeros((n_features,1))
        for i in range(n_samples//block_size + 1):
            start = i*block_size
            end = (i+1)*block_size if i < n_samples//block_size - 1 else n_samples
            X_chunk = X[start:end]
            U_chunk = U[start:end]
            y_chunk = y[start:end]
            
            Wy += X_chunk.T @ (U_chunk * y_chunk)

        W = W_inv @ Wy
        return W
    
    def save_model(self, W: np.ndarray) -> None:
        np.savetxt(self.model_weights, W, delimiter = '\n')
        
    def predict(self, W: np.ndarray) -> np.ndarray:
        X, _ = self.load_data("test")
        y_pred = X.values @ W
        return y_pred

class PartB:
    def __init__(self):
        self.train = str()
        self.test = str()
        self.regularization = str()
        self.model_predictions = str()
        self.model_weights = str()
        self.best_lambda = str()
    
    def parse_b(self) -> None:
        self.train = sys.argv[2]
        self.test = sys.argv[3]
        self.regularization = sys.argv[4]
        self.model_predictions = sys.argv[5]
        self.model_weights = sys.argv[6]
        self.best_lambda = sys.argv[7]
    
    def load_data(self, mode = "train") -> tuple[pd.DataFrame, pd.Series]:
        if mode == "train":
            df = pd.read_csv(self.train)
            X = df.drop(labels = ['Total Costs'], axis = 1)
            y = df['Total Costs']
            X = pd.concat([pd.DataFrame(np.ones((X.shape[0], 1)), columns=['Bias']), X], axis=1)
            return X, y
        else:
            X = pd.read_csv(self.test)
            X = pd.concat([pd.DataFrame(np.ones((X.shape[0], 1)), columns=['Bias']), X], axis=1)
            return X, None
    
    def get_reg_param(self, X, y) -> float:
        lambdas = pd.read_csv(self.regularization, header = None).values
        n = X.shape[0]
        block_size = n//10
        I = sp.sparse.identity(X.shape[1])
        min_error = float('inf')
        v = -1

        for lambda_ in lambdas:   
            cv_error = 0
            for i in range(10):
                start = i*block_size
                end = (i+1)*block_size
                
                cv_X = X[start:end]
                cv_y = y[start:end]
                print(cv_X.shape)
                
                curr_X = pd.concat([X[:start], X[end:]])
                curr_y = pd.concat([y[:start], y[end:]])
                
                A = (curr_X.T @ curr_X) + (lambda_*I.toarray())
                
                W_star = (sp.linalg.inv(A)) @ (curr_X.T @ curr_y)
                predicted_data = cv_X.values @ W_star
                error = np.mean((cv_y - predicted_data)**2)
                cv_error += error
            if (min_error > cv_error):
                v = lambda_
                min_error = cv_error
        
        return v
    
    def fit(self, X, y, lambda_ = 0) -> np.ndarray:
        X, y = self.load_data()
        I = sp.sparse.identity(X.shape[1])
        A = (X.T @ X) + (lambda_*I.toarray())
        W = (sp.linalg.inv(A)) @ (X.T @ y)
        return W
    
    def predict(self, W: np.ndarray) -> np.ndarray:
        X, _ = self.load_data("test")
        y_pred = X @ W
        return y_pred
    

def main():
    part = sys.argv[1]    
    part_a = PartA()
    part_b = PartB()
    if part == 'a':
        # parsing input
        part_a.parse_a()
        X, y = part_a.load_data()
        # getting the weights
        Weights = part_a.fit(X, y)
        # saving the weights to modelweights.txt
        part_a.save_model(Weights)
        # getting and saving the predictions
        y_pred = part_a.predict(Weights)
        np.savetxt(part_a.model_predictions, y_pred, delimiter = '\n')

    elif part == 'b':
        # parsing input
        part_b.parse_b()
        X, y = part_b.load_data()
        # best lambda
        lambda_ = part_b.get_reg_param(X, y)
        np.savetxt(part_b.best_lambda, lambda_, delimiter = '\n')
        # getting the weights
        Weights = part_b.fit(X, y, lambda_)
        # saving the weights to modelweights.txt
        np.savetxt(part_b.model_weights, Weights, delimiter = '\n')
        # getting and saving the predictions
        y_pred = part_b.predict(Weights)
        np.savetxt(part_b.model_predictions, y_pred, delimiter = '\n')

    else:
        print("Invalid part")

if __name__ == "__main__":
    main()