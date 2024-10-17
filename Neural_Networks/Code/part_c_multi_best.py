import argparse
import os
import numpy as np
import pickle
# import pandas as pd
# import time

from preprocessor import CustomImageDataset, DataLoader, numpy_transform

np.random.seed(0)

def organize_parameters(model):
    params = model.parameters()  # Get the list of parameters
    param_dict = {
        'weights': {},
        'bias': {}
    }
    
    # Iterate through the parameters and organize them into the dictionary
    for i in range(0, len(params), 2):
        layer_name = f'fc{i//2 + 1}'  # Assuming the layers are named sequentially as fc1, fc2, ...
        
        # Even indices are weights, odd indices are biases
        param_dict['weights'][layer_name] = params[i]    # Weight
        param_dict['bias'][layer_name] = params[i + 1].reshape(-1)   # Bias
    
    return param_dict

class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.W = (np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)).astype(np.float64)  # He initialization with float64
        self.b = np.zeros((1, fan_out), dtype=np.float64) if bias else None  # Bias initialization as float64
        self.x = None
    
    def __call__(self, x):
        self.x = x  # Store input for backpropagation
        self.out = x @ self.W  # (batch_size, fan_out)
        if self.b is not None:
            self.out += self.b
        return self.out  # (batch_size, fan_out)
    
    def __str__(self):
        return f"Linear({self.W.shape[0]}, {self.W.shape[1]})"
    
    def backprop(self, grad_out):
        # Gradients of weights and biases
        self.grad_W = self.x.T @ grad_out  # (fan_in, fan_out)
        if self.b is not None:
            self.grad_b = np.sum(grad_out, axis=0, keepdims=True)  # (1, fan_out)

        # Gradient of input for the previous layer
        grad_x = grad_out @ self.W.T  # (batch_size, fan_in)
        return grad_x  # (batch_size, fan_in)
    
    def parameters(self):
        return [self.W] + ([] if self.b is None else [self.b])

    def gradients(self):
        return [self.grad_W] + ([] if self.b is None else [self.grad_b])

class Sigmoid:
    def __call__(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out  # (batch_size, fan_out)

    def __str__(self):
        return f"Sigmoid()"
    
    def backprop(self, grad_out):
        grad_x = grad_out * self.out * (1 - self.out)
        return grad_x
    
    def parameters(self):
        return []

    def gradients(self):
        return []

class Softmax:
    def __call__(self, x):
        counts = np.exp(x)
        counts_sum = np.sum(counts, axis=1, keepdims=True)
        self.out = counts / counts_sum
        return self.out

    def __str__(self):
        return f"Softmax()"
    
    def backprop(self, grad_out):
        grad_x = self.out * (grad_out - np.sum(grad_out * self.out, axis=1, keepdims=True))
        return grad_x
    
    def parameters(self):
        return []
    
    def gradients(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def __str__(self):
        for layer in self.layers:
            print(layer)
        return ""

    def backprop(self, grad_out):
        for layer in reversed(self.layers):
            grad_out = layer.backprop(grad_out)

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def gradients(self):
        return [g for layer in self.layers for g in layer.gradients()]

def cross_entropy_train(y_pred, y_true):
    m = y_true.shape[0]
    logprobs = np.log(y_pred)
    loss = -logprobs[range(m), np.argmax(y_true, axis = 1)].mean()
    return loss.astype(np.float64)

def cross_entropy_grad(y_pred, y_true):
    m = y_true.shape[0]
    dlogprobs = np.zeros_like(y_pred)
    dlogprobs[range(m), np.argmax(y_true, axis = 1)] = -1/m

    return dlogprobs/y_pred

def binary_cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
    return loss

def binary_cross_entropy_grad(y_pred, y_true):
    grad = (y_pred - y_true) / ((y_pred)*(1-y_pred))
    return grad / y_true.shape[0]

class Optimizers:
    def __init__(self, params, optimizer_type='sgd', lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, momentum=0):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentum = momentum
        self.optimizer_type = optimizer_type.lower()
        
        # Initialize variables depending on the optimizer type
        if self.optimizer_type == 'sgd':
            self.velocities = [np.zeros_like(param) for param in self.params]
        elif self.optimizer_type == 'rmsprop':
            self.squares = [np.zeros_like(param) for param in self.params]
        elif self.optimizer_type == 'adam':
            self.m = [np.zeros_like(param) for param in self.params]
            self.v = [np.zeros_like(param) for param in self.params]
            self.t = 0
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
    
    def step(self, grads):
        if self.optimizer_type == 'sgd':
            self._sgd_step(grads)
        elif self.optimizer_type == 'rmsprop':
            self._rmsprop_step(grads)
        elif self.optimizer_type == 'adam':
            self._adam_step(grads)

    def _sgd_step(self, grads):
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
                param += self.velocities[i]
            else:
                param -= self.lr * grad

    def _rmsprop_step(self, grads):
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.squares[i] = self.beta1 * self.squares[i] + (1 - self.beta1) * (grad ** 2)
            param -= self.lr * grad / (np.sqrt(self.squares[i]) + self.epsilon)

    def _adam_step(self, grads):
        self.t += 1
        lr_t = self.lr * (np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t))

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param -= lr_t * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Example usage inside your train function
def train(model, dataset, epochs, batch_size, learning_rate, optimizer_type='sgd', beta1=0.9, beta2=0.999, epsilon=1e-8, momentum=0):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    # Initialize the Optimizers optimizer
    optimizer = Optimizers(
        params=model.parameters(),
        optimizer_type=optimizer_type,
        lr=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        momentum=momentum
    )

    for _ in range(epochs):
        for images, labels in dataloader:
            X_batch = images  # (batch_size, features)
            y_batch = np.zeros((X_batch.shape[0], 8))
            y_batch[np.arange(len(labels)), labels] = 1  # One-hot encoding the labels
            
            # Forward pass for training batch
            y_pred = model(X_batch)
            # loss = cross_entropy_train(y_pred, y_batch)
            
            # Backpropagation
            grad_out = cross_entropy_grad(y_pred, y_batch)
            model.backprop(grad_out)
            
            # Optimizer step
            optimizer.step(model.gradients())    

def main():
    # ----------------- Parse the arguments -----------------
    parser = argparse.ArgumentParser(description='Neural Network Multi Class Classification')

    parser.add_argument('--dataset_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--save_weights_path', type=str, required=True, help='Path to save weights')
    
    args = parser.parse_args()
    
    dataset_root = args.dataset_root
    save_weights_path = args.save_weights_path

    # ----------------- Load the dataset -----------------
    mode = 'train'
    
    if mode == 'train':
        csv = os.path.join(dataset_root, 'train.csv')
    elif mode == 'val':
        csv = os.path.join(dataset_root, 'val.csv')
        
    dataset = CustomImageDataset(root_dir=dataset_root, csv=csv, transform=numpy_transform)

    model = Sequential([
        Linear(625, 512),   
        Sigmoid(),       
        Linear(512, 256),    
        Sigmoid(),        
        Linear(256, 128),   
        Sigmoid(),       
        Linear(128, 32),   
        Sigmoid(),
        Linear(32, 8),
        Softmax()
    ])

    # ----------------- Train the model ----------------
    epochs = 40
    batch_size = 200
    learning_rate = 2e-3
    optimizer = 'adam'
    train(model, dataset, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, optimizer_type = optimizer)
    
    # ----------------- Save the weights ----------------
    param_dict = organize_parameters(model)
    with open(save_weights_path, 'wb') as f:
        pickle.dump(param_dict, f)
    
if __name__ == "__main__":
    main()