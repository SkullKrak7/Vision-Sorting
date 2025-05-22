# Vision Sorting System: OpenCV + ML

A real-time image classification system using OpenCV, with two training options:

* A traditional machine learning pipeline using scikit-learn (lightweight, fast, CPU-only)
* An optional convolutional neural network (CNN) implemented in PyTorch for higher accuracy

Designed for rapid prototyping, educational use, and flexibility. Works on low-spec machines using kNN or SVM, while also supporting GPU acceleration for deeper models.

## How It Works

The system enables real-time classification of objects using your webcam. The workflow is:

1. Capture images for each class you want to recognize (e.g., red, blue, green) using `capture_and_save.py`.
2. Train a model using either:

   * `train_basic_ml.py` for lightweight, traditional machine learning (kNN, SVM)
   * `train_cnn_pytorch.py` for a more accurate convolutional neural network (requires PyTorch)
3. Run live inference using your webcam with `realtime_inference.py`. It loads your trained model and displays predicted class labels on-screen.

Use this project to prototype basic vision-based classification tasks, compare ML vs CNN pipelines, or build simple interactive computer vision demos.

## Features

* Real-time object classification using webcam input
* Dataset builder via webcam capture
* Supports traditional ML models such as kNN and SVM
* Live prediction overlay on webcam feed
* Optional PyTorch CNN training with evaluation metrics
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
# If any issues occur, check that all required packages are listed and install manually if needed
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
* Trains a basic classifier using scikit-learn (e.g., kNN)
* Prints accuracy and per-class classification report
* Saves the model to `models/basic_model.pkl`

## Optional: Train a CNN Model (PyTorch)

```bash
python scripts/train_cnn_pytorch.py
```

* Trains a PyTorch-based CNN classifier
* Prints classification accuracy, per-class metrics, and confusion matrix
* Displays confusion matrix as a plot (requires matplotlib)
* Saves the trained model and label map to `models/cnn_model.pth`

## Step 3: Run Live Inference

```bash
python scripts/realtime_inference.py
```

* Opens the webcam
* Displays real-time predictions on live video
* Press `ESC` to quit

## Optional Improvements

* Add region-of-interest (ROI) cropping
* Add confidence thresholding or an "unknown" fallback
* Use motion detection or color segmentation
* Extend to more classes with more examples

## Minimum Dataset Requirements

To ensure reliable training and inference, follow these guidelines when preparing your dataset:

* **Number of classes**: At least 2 classes are required (e.g., 'red' and 'blue')
* **Samples per class**: Minimum 10 images per class; 50+ recommended for stability
* **Balanced classes**: Keep class counts roughly equal for best performance
* **File format**: Store images under `data/<class_label>/` directories
* **Image size**: Images are resized to 64x64 pixels automatically (for CNN only)

Avoid training with only one class or very low image counts per class. The system currently does not augment or balance data.

## Core Dependencies

This project requires the following packages:

* numpy
* opencv-python
* scikit-learn
* torch
* matplotlib (for CNN evaluation)

Install them via:

```bash
pip install -r requirements.txt
```

## License

MIT License. Free to use, modify, and distribute.
