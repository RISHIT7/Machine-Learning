import numpy as np
import math


def gradient(x, y, predictions):
    return (x.T @ (predictions - y)) / x.shape[0]


def cross_entropy(y, predictions, eps=1e-15):
    # Calculate cross-entropy for binary classification
    predictions = np.clip(predictions, eps, 1 - eps)
    return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(x, w):
    logits = x @ w
    return sigmoid(logits)


def update_weights(x, y, w, lr):
    predictions = predict(x, w)
    grad = gradient(x, y, predictions)
    w = w - (grad * lr)
    return w


# Learning rate strategies
constant_lr = lambda x: float(x)
adaptive_lr = lambda x, i: float(x) / math.sqrt(i + 1)

def logistic_regression(x, y):
    # Load dataset
#     train = np.genfromtxt(trainfile, delimiter=',', skip_header=1)
#     x = train[:, :-1].astype(float)
    y = y.reshape(-1, 1)  # Reshape y to (n_samples, 1)
    
    # Initialize weights
    w = np.zeros((x.shape[1], 1))

    # Training parameters
    strategy = 1  # 1 for constant_lr, 2 for adaptive_lr
    lr_params = 0.02
    iters = 15000

    # Training loop
    for i in range(iters):
        # Select learning rate strategy
        if strategy == 1:
            lr = constant_lr(lr_params)
        elif strategy == 2:
            lr = adaptive_lr(lr_params, i)
        
        # Update weights
        w = update_weights(x, y, w, lr)
        
        # Print cost and learning rate at each iteration
        cost = cross_entropy(y, predict(x, w))
        
#         if(i%5000 == 0):
#             print(f'Iteration: {i + 1}\t Learning rate: {lr}\t Cost: {cost}')
    
    # Final model cost
#     print(f'Final cost: {cross_entropy(y, predict(x, w))}')
    ww = np.array([x[0] for x in w])
#     print(f"Final weights are : {ww}")
    
    return ww
