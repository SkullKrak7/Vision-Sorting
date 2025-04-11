import cv2
import os
from datetime import datetime

label = "red" #define the label, change to whatever label required
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(BASE_DIR, "..", "data", label)
save_dir = os.path.abspath(save_dir)

os.makedirs(save_dir, exist_ok=True) #creating the directory and no error if already dir exists

cap = cv2.VideoCapture(0)

print("press 's' to save frame, 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        print("failed to capture frame")
        break

    cv2.imshow("Live Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label}_{timestamp}.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"Saved: {filepath}")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
