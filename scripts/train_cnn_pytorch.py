import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from load_and_preprocess import load_dataset


class SimpleCNN(nn.Module):
    def _init_(self, num_classes):
        super()._init_()
        self.net=nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*16*16, 128), nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        return self.net(x)

class ImageDataset(Dataset):
    