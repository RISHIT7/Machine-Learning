import argparse
import os
import pickle
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import time

from part_d_testloader import TestDataLoader, TestImageDataset, numpy_transform
from part_d_trainloader import TrainDataLoader, TrainImageDataset, numpy_transform

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
    
class ReLU:
    def __call__(self, x):
        self.out = np.maximum(0, x)
        return self.out

    def __str__(self):
        return f"ReLU()"
    
    def backprop(self, grad_out):
        grad_x = grad_out * (self.out > 0)
        return grad_x
    
    def parameters(self):
        return []
    
    def gradients(self):
        return []

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.out = None

    def __call__(self, x):
        self.out = np.where(x > 0, x, self.alpha * x)
        return self.out

    def __str__(self):
        return f"LeakyReLU(alpha={self.alpha})"

    def backprop(self, grad_out):
        grad_x = np.where(self.out > 0, grad_out, self.alpha * grad_out)
        return grad_x

    def parameters(self):
        return []

    def gradients(self):
        return []

    
class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None

    def __call__(self, x, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=x.shape)
            return x * self.mask / (1 - self.rate)
        else:
            return x

    def __str__(self):
        return f"Dropout(rate={self.rate})"

    def backprop(self, grad_out):
        return grad_out * self.mask / (1 - self.rate)

    def parameters(self):
        return []
    
    def gradients(self):
        return []
    
class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self.training = True
    
    def __call__(self, x):
        for layer in self.layers:
            if (isinstance(layer, BatchNorm)):
                x = layer(x, self.training)
            else:
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

    def set_training(self, training=True):
        self.training = training

class BatchNorm:
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        # Initialize trainable parameters (gamma and beta)
        self.gamma = np.ones((1, num_features), dtype=np.float64)
        self.beta = np.zeros((1, num_features), dtype=np.float64)

        # Initialize running mean and variance for inference
        self.running_mean = np.zeros((1, num_features), dtype=np.float64)
        self.running_var = np.ones((1, num_features), dtype=np.float64)

        # Buffers to store intermediate values during forward pass
        self.mean = None
        self.var = None
        self.x_normalized = None
        self.x_centered = None
        self.std_inv = None

    def __call__(self, x, training=True):
        if training:
            # Calculate batch mean and variance
            self.mean = np.mean(x, axis=0, keepdims=True)
            self.var = np.var(x, axis=0, keepdims=True)
            self.x_centered = x - self.mean
            self.std_inv = 1.0 / np.sqrt(self.var + self.epsilon)
            self.x_normalized = self.x_centered * self.std_inv

            # Update running mean and variance for inference
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

            # Scale and shift
            out = self.gamma * self.x_normalized + self.beta
        else:
            # During inference, use running mean and variance
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * x_normalized + self.beta

        return out

    def __str__(self):
        return f"BatchNorm(num_features={self.num_features})"

    def backprop(self, grad_out):
        # Backpropagation through the batch norm layer
        batch_size = grad_out.shape[0]

        # Gradient of gamma (scale) and beta (shift)
        self.grad_gamma = np.sum(grad_out * self.x_normalized, axis=0, keepdims=True)
        self.grad_beta = np.sum(grad_out, axis=0, keepdims=True)

        # Backprop through the normalization
        grad_x_normalized = grad_out * self.gamma
        grad_var = np.sum(grad_x_normalized * self.x_centered * -0.5 * np.power(self.var + self.epsilon, -1.5), axis=0, keepdims=True)
        grad_mean = np.sum(grad_x_normalized * -self.std_inv, axis=0, keepdims=True) + grad_var * np.mean(-2.0 * self.x_centered, axis=0, keepdims=True)

        grad_x = (grad_x_normalized * self.std_inv) + (grad_var * 2.0 * self.x_centered / batch_size) + (grad_mean / batch_size)

        return grad_x

    def parameters(self):
        # Return the trainable parameters: gamma and beta
        return [self.gamma, self.beta]

    def gradients(self):
        # Return the gradients of gamma and beta
        return [self.grad_gamma, self.grad_beta]
    
def cross_entropy_train(y_pred, y_true, num_classes = 8):
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

class OptimMixers:
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
    dataloader = TrainDataLoader(dataset, batch_size=batch_size)
    
    # Initialize the OptimMixers optimizer
    optimizer = OptimMixers(
        params=model.parameters(),
        optimizer_type=optimizer_type,
        lr=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        momentum=momentum
    )
    
    for _ in range(epochs):
        # Training loop
        for images, labels in dataloader:
            X_batch = images  # (batch_size, features)
            y_batch = np.zeros((X_batch.shape[0], 8))
            y_batch[np.arange(len(labels)), labels] = 1  # One-hot encoding the labels
            # Forward pass for training batch
            y_pred = model(X_batch)
            
            # Backpropagation
            grad_out = cross_entropy_grad(y_pred, y_batch)
            model.backprop(grad_out)
            # Optimizer step
            optimizer.step(model.gradients())
            
def sample(model, test_set):
    dataloader = TestDataLoader(test_set, batch_size=1)
    predictions = []
    for images in dataloader:
        model.set_training(False)
        X_batch = images
        y_pred = model(X_batch)
        predictions.append(np.argmax(y_pred, axis=1)[0])
    return predictions

def main():
    # ----------------- Parse the arguments -----------------
    parser = argparse.ArgumentParser(description='Neural Network Multi Class Classification')
    
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to dataset')
    parser.add_argument('--test_dataset_root', type=str, required=True, help='Path to test set')
    parser.add_argument('--save_weights_path', type=str, required=True, help='Path to save weights')
    parser.add_argument('--save_predictions_path', type=str, required=True, help='Path to save predictions')
    
    args = parser.parse_args()
    
    dataset_root = args.dataset_root
    test_dataset_root = args.test_dataset_root
    save_weights_path = args.save_weights_path
    save_predictions_path = args.save_predictions_path
    
    # ----------------- Load the dataset -----------------
    csv = os.path.join(dataset_root, 'train.csv')    
    dataset = TrainImageDataset(root_dir=dataset_root, csv=csv, transform=numpy_transform)

    model = Sequential([
        Linear(625, 512),
        BatchNorm(512),
        ReLU(),
        Linear(512, 256),
        BatchNorm(256),
        ReLU(),
        Linear(256, 128),
        BatchNorm(128),
        ReLU(),
        Linear(128, 64),
        BatchNorm(64),
        ReLU(),
        Linear(64, 32),
        BatchNorm(32),
        ReLU(),
        Linear(32, 8),
        Softmax()
    ])
    
    # ----------------- Train the model ----------------
    epochs = 40
    batch_size = 200
    learning_rate = 7e-3
    optimizer_type = "adam"
    
    train(model, dataset, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, optimizer_type=optimizer_type)

    # ----------------- Save the model weights ----------------
    param_dict = organize_parameters(model)
    with open(save_weights_path, 'wb') as f:
        pickle.dump(param_dict, f)
    

    # ----------------- Load the test dataset -----------------
    csv = os.path.join(test_dataset_root, 'val.csv')
    test_set = TestImageDataset(root_dir=test_dataset_root, csv=csv, transform=numpy_transform)
    
    # ----------------- Sample the test set ----------------
    y_pred = sample(model, test_set)
    
    # ----------------- Save the predictions ----------------
    with open(save_predictions_path, 'w') as f:
        pickle.dump(y_pred, f)    
        
if __name__ == "__main__":
    main()