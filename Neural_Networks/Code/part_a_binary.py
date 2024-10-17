import argparse
import pickle
import os
import numpy as np

from preprocessor import CustomImageDataset, DataLoader, numpy_transform

np.random.seed(0)

class Linear:
    def __init__(self, fan_in, fan_out, bias=True): 
        self.W = np.random.randn(fan_in, fan_out) * np.sqrt(2 / fan_in)  # (fan_in, fan_out)
        self.b = np.zeros((1, fan_out)) if bias else None  # (1, fan_out)
        self.x = None
    
    def __call__(self, x):
        self.x = x  
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

def binary_cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
    return loss

def binary_cross_entropy_grad(y_pred, y_true):
    grad = (y_pred - y_true) / ((y_pred)*(1-y_pred))
    return grad / y_true.shape[0]

def train(model, dataset, epochs, batch_size, learning_rate):

    dataloader = DataLoader(dataset, batch_size=batch_size)

    for _ in range(epochs):
        i = 0
        for images, labels in dataloader:
            X_batch = images # (batch_size, features)
            y_batch = labels.reshape(-1, 1) # (batch_size,)

            y_pred = model(X_batch)

            # loss = binary_cross_entropy_loss(y_pred, y_batch)

            grad_out = binary_cross_entropy_grad(y_pred, y_batch)
            model.backprop(grad_out)

            for param, grad in zip(model.parameters(), model.gradients()):
                param -= learning_rate * grad
            i += 1

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

def main():
    # ----------------- Parse the arguments -----------------
    parser = argparse.ArgumentParser(description='Neural Network Binary Classification')

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
    
    # ----------------- Define the model ----------------
    model = Sequential([
        Linear(625, 512),   
        Sigmoid(),       
        Linear(512, 256),    
        Sigmoid(),        
        Linear(256, 128),   
        Sigmoid(),       
        Linear(128, 1),   
        Sigmoid()       
    ])

    # ----------------- Train the model ----------------
    epochs = 15
    batch_size = 256
    learning_rate = 0.001
    train(model, dataset, epochs, batch_size, learning_rate)
    
    # ----------------- Save the weights ----------------
    param_dict = organize_parameters(model)
    with open(save_weights_path, 'wb') as f:
        pickle.dump(param_dict, f)
    
    
if __name__ == "__main__":
    main()