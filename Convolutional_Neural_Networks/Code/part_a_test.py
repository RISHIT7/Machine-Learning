import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import os
import pickle

from testloader import CustomImageDataset as TestDataset

torch.manual_seed(0)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 12 * 25, out_features=1)
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
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            if preds >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
    
    # ----------------- save predictions ----------------- #
    with open(save_predictions_path, 'wb') as f:
        pickle.dump(predictions, f)

if __name__ == "__main__":
    main()