---

# 🧠 Vision Sorting System – OpenCV + ML

A basic but complete computer vision pipeline for real-time object classification using webcam images. Built to practice real-world image handling, data collection, model training, and inference — all from scratch.

The system starts simple using OpenCV and traditional ML, and is structured to scale up to PyTorch-based deep learning later.

---

## 🔧 What this project does

- Captures images from webcam and saves them with labels  
- Organizes them into clean folders (one per class)  
- Preprocesses the dataset and builds feature-label pairs  
- Trains a basic classifier using scikit-learn (LogReg/KNN)  
- Supports real-time predictions using webcam input  
- Has scope to upgrade to CNN using PyTorch  

---

## 🗂️ Folder Structure

```
vision_sorting_project/
├── data/                  # Saved labeled images (red/, blue/, etc.)
├── models/                # Trained ML or DL models
├── scripts/               # All core logic and modular scripts
├── requirements.txt       # Dependencies
└── README.md              # You're here
```

---

## 📁 Scripts Breakdown

| File                      | Purpose                                      |
|---------------------------|----------------------------------------------|
| `capture_and_save.py`     | Capture images from webcam and save with label |
| `load_and_preprocess.py`  | Load all saved images, preprocess for training |
| `train_basic_ml.py`       | Train a simple ML model (KNN or Logistic)      |
| `train_cnn_pytorch.py`    | (Optional) Train a CNN using PyTorch          |
| `realtime_inference.py`   | Classify live webcam feed using trained model |

---

## 🚀 How to Run

1. Clone the repo or open folder in VS Code  
2. Activate your environment  
   ```bash
   conda activate pytorch_env
   ```
3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```
4. Run the capture script (change label inside script)  
   ```bash
   python scripts/capture_and_save.py
   ```
5. Train a basic ML model  
   ```bash
   python scripts/train_basic_ml.py
   ```
6. (Optional) Train a CNN model  
   ```bash
   python scripts/train_cnn_pytorch.py
   ```
7. Run real-time inference  
   ```bash
   python scripts/realtime_inference.py
   ```

---

## 🧠 Why I built this

To practice the full CV + ML cycle:
- Data collection  
- Preprocessing  
- Model training  
- Real-time deployment  

Also to build a solid portfolio project — something that shows I can go from raw webcam input to deployable ML pipeline.

---

## 🛠️ Tech Stack

- Python 3.10+  
- OpenCV  
- NumPy  
- scikit-learn  
- PyTorch  
- Conda (GPU-enabled env)

---

## ✌️ Built with patience, practice, and purpose

By **Sai Karthik Kagolanu**  
Feel free to explore, fork, or reach out for suggestions.

---

## 📝 Notes

- You can add more classes by capturing more labeled images  
- Model can be switched from KNN to Logistic Regression easily  
- CNN version can use transfer learning later (ResNet etc.)

---