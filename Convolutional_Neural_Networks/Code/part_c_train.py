import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import argparse
import os

from trainloader import CustomImageDataset as TrainDataset

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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Dataloader parameters
    batch_size = 128
    csv = os.path.join(dataset_root, 'public_train.csv')
    train_dataset = TrainDataset(root_dir=dataset_root, csv=csv, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    # Training loop
    num_epochs = 11

    for _ in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.float()  # Ensure the datatype is float32
            labels = labels.float()

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs.squeeze(), labels.long())

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        scheduler.step()

    # ----------------- save model weights ----------------- #
    torch.save(model.state_dict(), save_weights_path)

if __name__ == "__main__":
    main()