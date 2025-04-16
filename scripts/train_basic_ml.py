import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from load_and_preprocess import load_dataset
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data")
data_path = os.path.abspath(data_path)

(X_train, X_test, y_train, y_test), label_map = load_dataset(data_path)


X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

model = KNeighborsClassifier(n_neighbors = 3)
model.fit(X_train_flat, y_train)

preds = model.predict(X_test_flat)
print(classification_report(y_test, preds, target_names=label_map.values()))

os.makedirs("models", exist_ok=True)
with open("models/basic_model.pkl","wb") as f:
    pickle.dump((model, label_map), f)