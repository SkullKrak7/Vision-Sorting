import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(data_dir, img_size=(64,64)):
    X, y, labels = [], [], []
    label_map = {}

    for i, folder in enumerate(sorted(os.listdir(data_dir))):
        path = os.path.join(data_dir, folder)
        if not os.path.isdir(path):
            continue
        label_map[i] = folder
        labels.append(folder)
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                X.append(img)
                y.append(i)
    
    X=np.array(X) / 255.0
    y=np.array(y)
    return train_test_split(X,y, test_size=0.2, random_state=42), label_map