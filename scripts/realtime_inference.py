import cv2
import numpy as np
import pickle
import time

with open("models/basic_model.pkl", "rb") as f:
    model, label_map = pickle.load(f)

img_size = (64, 64)
cap = cv2.VideoCapture(0)
print("Press ESC to quit.")

# Initialize timing
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Preprocess and predict
    img_resized = cv2.resize(frame, img_size)
    img_flattened = img_resized.reshape(1, -1) / 255.0
    pred = model.predict(img_flattened)[0]
    label = label_map.get(pred, "Unknown")

    # Overlay prediction and FPS
    cv2.putText(frame, f"Predicted: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("Webcam Classification", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
