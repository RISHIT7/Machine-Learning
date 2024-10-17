import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd

# Define the transformations [Import this definition as well to your training script!]
transform = transforms.Compose([
    transforms.Resize((50, 100)),  # Resize to 50x100 (height x width)
    transforms.ToTensor(),         # Convert the image to a tensor and also rescales the pixels by dividing them by 255
])
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, csv, transform=None):
        """
        Args:
            root_dir (string): Directory with all the subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Path"])
        image = Image.open(img_path).convert("L")
        label = row["class"]

        if self.transform:
            image = self.transform(image)


        return image, label

if __name__ == '__main__':

    #Examples usage

    #Root directory of the dataset and its csv file
    root_dir = "path/to/dataset/root/dir"
    csv_path = os.path.join(root_dir, "public_train.csv")

    #Create the custom dataset
    dataset = CustomImageDataset(root_dir=root_dir, csv = csv_path, transform=transform)

    #Create the DataLoader
    dataloader = DataLoader(dataset, batch_size=128) #PyTorch Dataloader

    #Iterate through the DataLoader
    for idx, (images, labels) in enumerate(dataloader):
        print(images.shape)  # Should be [batch_size, 1, 50, 100]
        print(labels.shape)  # Should be [batch_size]
        print(idx)

