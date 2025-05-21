import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from load_and_preprocess import load_dataset
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class ImageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.transpose(0, 3, 1, 2))  # HWC → CHW
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Load dataset
(X_train, X_test, y_train, y_test), label_map = load_dataset("data")

train_loader = DataLoader(ImageDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(ImageDataset(X_test, y_test), batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=len(label_map)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} completed. Loss: {total_loss:.4f}")

# Evaluation
model.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        output = model(X_batch)
        preds = output.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_batch.numpy())

# Metrics
target_names = [label_map[i] for i in sorted(label_map)]

acc = accuracy_score(all_targets, all_preds)
print(f"\nTest Accuracy: {acc * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(all_targets, all_preds, target_names=target_names))

print("Confusion Matrix:")
cm = confusion_matrix(all_targets, all_preds)
print(cm)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Save model
os.makedirs("models", exist_ok=True)
torch.save({'model_state_dict': model.state_dict(), 'label_map': label_map}, 'models/cnn_model.pth')
