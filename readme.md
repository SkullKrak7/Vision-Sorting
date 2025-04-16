# 🧠 Vision Sorting System – OpenCV + ML

A lightweight, real-time image classification system using OpenCV and traditional machine learning (scikit-learn). Designed for rapid prototyping and educational use — no deep learning or GPU required.

---

## 🚀 Features

- Real-time object classification using webcam input  
- Dataset builder via webcam capture  
- Trains traditional ML models (kNN, SVM, etc.)  
- Live prediction overlay on webcam feed  
- Fast, interpretable, and easy to extend

---

## 🗂️ Project Structure

```
vision_sorting_project/
├── data/                   # Labeled images (by class)
├── models/                 # Trained ML model(s)
├── scripts/
│   ├── capture_and_save.py       # Capture and label images
│   ├── load_and_preprocess.py    # Dataset loading + preprocessing
│   ├── train_basic_ml.py         # Train scikit-learn model
│   └── realtime_inference.py     # Live webcam classification
├── requirements.txt
└── README.md
```

---

## 🔧 Setup

```bash
conda create -n vision-ml python=3.10
conda activate vision-ml
pip install -r requirements.txt
```

---

## 📸 Step 1: Capture Training Images

```bash
python scripts/capture_and_save.py
```

- Press `s` to save a frame  
- Press `q` to quit  
- Set your desired label in the script (`label = "example_label"`)  
- Images are saved under `data/<label>/`

---

## 🧠 Step 2: Train Your ML Model

```bash
python scripts/train_basic_ml.py
```

- Loads data from `data/`  
- Trains a basic classifier  
- Saves it to `models/basic_model.pkl`

---

## 🎥 Step 3: Run Live Inference

```bash
python scripts/realtime_inference.py
```

- Opens webcam  
- Displays real-time predictions  
- Press `ESC` to quit

---

## 🛠️ Optional Improvements

- Add region-of-interest (ROI) cropping  
- Add confidence thresholding or "unknown" fallback  
- Use motion detection or color segmentation  
- Extend to more classes with more examples

---

## 📚 Dependencies

- opencv-python  
- numpy  
- scikit-learn  

Install via:

```bash
pip install -r requirements.txt
```

---

## 📄 License

MIT License — free to use, share, and modify.

---
