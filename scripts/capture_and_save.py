import cv2
import os
from datetime import datetime

label = "red"  # Change this to your class label
frame_width = 640
frame_height = 480

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.abspath(os.path.join(base_dir, "..", "data", label))
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    print("Press 's' to save frame, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
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

if __name__ == "__main__":
    main()
