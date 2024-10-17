import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import os

from trainloader import CustomImageDataset as TrainDataset

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
    parser.add_argument("--train_dataset_root", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--save_weights_path", type=str, required=True, help="Path to save the model weights")
    args = parser.parse_args()
    
    dataset_root = args.train_dataset_root
    save_weights_path = args.save_weights_path
    
    transform = transforms.Compose([
        transforms.Resize((50, 100)),
        transforms.ToTensor(),
    ])
    
    # ----------------- model training ----------------- #
    model = CNNModel().float()
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss with logits
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 128
    csv = os.path.join(dataset_root, 'public_train.csv')
    train_dataset = TrainDataset(root_dir=dataset_root, csv=csv, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    num_epochs = 8

    for _ in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.float()
            labels = labels.float()

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs.squeeze(), labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

    # ----------------- save model weights ----------------- #
    torch.save(model.state_dict(), save_weights_path)
        
if __name__ == "__main__":
    main()