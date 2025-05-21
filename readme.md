# Vision Sorting System: OpenCV + ML

A lightweight, real-time image classification system using OpenCV and traditional machine learning (scikit-learn). Designed for rapid prototyping and educational use. This version does not require deep learning or GPU acceleration.

## Features

* Real-time object classification using webcam input
* Dataset builder via webcam capture
* Supports traditional ML models such as kNN and SVM
* Live prediction overlay on webcam feed
* Modular and extensible codebase

## Project Structure

```
vision_sorting_project/
├── data/                       # (user-generated) Labeled images by class – not included in repository
├── models/                     # (auto-generated) Trained ML models – not included in repository
├── scripts/
│   ├── capture_and_save.py       # Capture and label images
│   ├── load_and_preprocess.py    # Dataset loading and preprocessing
│   ├── train_basic_ml.py         # Train scikit-learn model
│   ├── realtime_inference.py     # Live webcam classification
│   └── train_cnn_pytorch.py      # Train PyTorch CNN model (optional)
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
```

## Setup Instructions

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 1: Capture Training Images

```bash
python scripts/capture_and_save.py
```

* Press `s` to save a frame
* Press `q` to quit
* Set your desired label inside the script (`label = "example_label"`)
* Images will be saved under `data/<label>/`

## Step 2: Train Your ML Model

```bash
python scripts/train_basic_ml.py
```

* Loads labeled image data from your local `data/` folder
* Trains a basic classifier
* Saves the model to `models/basic_model.pkl`

## Optional: Train a CNN Model (PyTorch)

```bash
python scripts/train_cnn_pytorch.py
```

Use this script to train a convolutional neural network for better performance.

## Step 3: Run Live Inference

```bash
python scripts/realtime_inference.py
```

* Opens the webcam
* Displays real-time predictions
* Press `ESC` to quit

## Optional Improvements

* Add region-of-interest (ROI) cropping
* Add confidence thresholding or an "unknown" fallback
* Use motion detection or color segmentation
* Extend to more classes with more examples

## License

MIT License. Free to use, modify, and distribute.
