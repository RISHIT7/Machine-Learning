import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import os
import pickle

from testloader import CustomImageDataset as TestDataset

torch.manual_seed(0)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),  # Initial convolution
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Layer 1
            ResNetBlock(64, 64),
            ResNetBlock(64, 64),

            # Layer 2
            ResNetBlock(64, 128, stride=2),
            ResNetBlock(128, 128),

            # Layer 3
            ResNetBlock(128, 256, stride=2),
            ResNetBlock(256, 256),

            # Layer 4
            ResNetBlock(256, 512, stride=2),
            ResNetBlock(512, 512),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 8),  # Adjust the output layer for 8 classes
        )

    def forward(self, x):
        return self.model(x)


def main():
    # ----------------- argument parsing ----------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dataset_root", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--load_weights_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--save_predictions_path", type=str, required=True, help="Path to save the predictions")
    args = parser.parse_args()
    
    test_dataset_root = args.test_dataset_root
    load_weights_path = args.load_weights_path
    save_predictions_path = args.save_predictions_path
    
    # ----------------- model loading ----------------- #
    model = CNNModel().float()
    model.load_state_dict(torch.load(load_weights_path))
    model.eval()
    
    test_transforms = transforms.Compose([
        transforms.Resize((50, 100)),
        transforms.ToTensor(),
    ])
    csv = os.path.join(test_dataset_root, 'public_test.csv')
    
    # ----------------- prepare test dataset ----------------- #
    test_dataset = TestDataset(root_dir=test_dataset_root, csv=csv, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # ----------------- make predictions ----------------- #
    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.float()
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            predictions.append(predicted.item())
    
    # ----------------- save predictions ----------------- #
    with open(save_predictions_path, 'wb') as f:
        pickle.dump(predictions, f)

if __name__ == "__main__":
    main()