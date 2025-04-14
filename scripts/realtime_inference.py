import cv2
import numpy as np
import pickle

with open("model/basic_model.pkl", "rb") as f:
    model, label_map = pickle.load(f)

img_size = (64,64)

cap=cv2.VideoCapture(0)

print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_resized = cv2.resize(frame, img_size)
    img_flattened = img_resized.reshape(1,-1)/255.0
    pred = model.predict(img_flattened)[0]
    label = label_map[pred]
    cv2.putText(frame, f"Predicted: {label}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Webcame Classification", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()