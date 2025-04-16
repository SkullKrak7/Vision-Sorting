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
    def _init_(self,X,y):
        self.X = torch.FloatTensor(X.transpose(0,3,1,2))
        self.y = torch.LongTensor(y)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

(X_train, X_test, y_train, y_test), label_map = load_dataset("data")
train_loader = DataLoader(ImageDataset(X_train,y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(ImageDataset(X_test,y_test), batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(label_map)).to(device)
criterion = nn.CrossEntropyLoss()
optimmizer = optim.Adam(model.parameters(),lr=0.001)

for epoch in range(10):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done.")

torch.save({'model_state_dict': model.state_dict(), 'label_map': label_map}, 'models/cnn_model.pth')
