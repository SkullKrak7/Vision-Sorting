import os
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from load_and_preprocess import load_dataset
import matplotlib.pyplot as plt

# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.abspath(os.path.join(script_dir, "..", "data"))
(X_train, X_test, y_train, y_test), label_map = load_dataset(data_path)

# Flatten image data
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_flat, y_train)

# Evaluate model
preds = model.predict(X_test_flat)
target_names = [label_map[i] for i in sorted(label_map.keys())]

print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=target_names))

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, preds) * 100))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

# Optional visual display (will just skip if headless)
try:
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, display_labels=target_names, cmap="Blues"
    ).plot()
    plt.show()
except Exception as e:
    print(f"(Optional) Confusion matrix plot skipped: {e}")

# Save model
os.makedirs("models", exist_ok=True)
with open("models/basic_model.pkl", "wb") as f:
    pickle.dump((model, label_map), f)
