"""
register_face.py
-----------------
Registers a new person into the system by capturing their face
from the webcam and saving it into the known_faces/ folder.

Usage:
    python register_face.py "John Doe"
"""

import cv2
import os
import sys

KNOWN_FACES_DIR = "known_faces"


def register_face(name):
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print("Look at the camera. Press SPACE to capture, ESC to cancel.")

    captured_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        display = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(display, "SPACE = capture | ESC = cancel", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Register Face", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Cancelled.")
            break
        elif key == 32:  # SPACE
            if len(faces) == 0:
                print("No face detected, try again.")
                continue
            captured_frame = frame
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured_frame is not None:
        filepath = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
        cv2.imwrite(filepath, captured_frame)
        print(f"Saved face for '{name}' to {filepath}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python register_face.py "Person Name"')
        sys.exit(1)

    person_name = sys.argv[1]
    register_face(person_name)
