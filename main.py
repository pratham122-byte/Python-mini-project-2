"""
main.py
-------
Face Recognition Attendance System

- Loads known faces from known_faces/ (filename = person's name)
- Opens the webcam and detects/recognizes faces in real time
- Marks attendance (name + timestamp) in a daily CSV file
- Each person is marked present only once per day

Controls:
    q -> quit
"""

import cv2
import face_recognition
import numpy as np
import os
import csv
from datetime import datetime

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_DIR = "attendance"
TOLERANCE = 0.5          # lower = stricter matching
FRAME_RESIZE_SCALE = 0.25  # process a smaller frame for speed


def load_known_faces():
    known_encodings = []
    known_names = []

    if not os.path.isdir(KNOWN_FACES_DIR):
        print(f"'{KNOWN_FACES_DIR}' folder not found.")
        return known_encodings, known_names

    for filename in os.listdir(KNOWN_FACES_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 0:
            print(f"Warning: no face found in {filename}, skipping.")
            continue

        known_encodings.append(encodings[0])
        name = os.path.splitext(filename)[0]
        known_names.append(name)

    print(f"Loaded {len(known_names)} known face(s): {', '.join(known_names) if known_names else 'none'}")
    return known_encodings, known_names


def get_today_csv_path():
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")


def load_already_marked(csv_path):
    marked = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    marked.add(row[0])
    return marked


def mark_attendance(name, csv_path, marked_today):
    if name in marked_today:
        return  # already marked today

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Time", "Date"])
        now = datetime.now()
        writer.writerow([name, now.strftime("%H:%M:%S"), now.strftime("%Y-%m-%d")])

    marked_today.add(name)
    print(f"Attendance marked for {name} at {datetime.now().strftime('%H:%M:%S')}")


def main():
    known_encodings, known_names = load_known_faces()
    if not known_encodings:
        print("No known faces loaded. Add images to known_faces/ or run register_face.py first.")
        return

    csv_path = get_today_csv_path()
    marked_today = load_already_marked(csv_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not access webcam.")
        return

    print("Starting attendance system. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            name = "Unknown"

            if len(distances) > 0:
                best_match_index = np.argmin(distances)
                if distances[best_match_index] < TOLERANCE:
                    name = known_names[best_match_index]
                    mark_attendance(name, csv_path, marked_today)

            # Scale face location back up
            top = int(top / FRAME_RESIZE_SCALE)
            right = int(right / FRAME_RESIZE_SCALE)
            bottom = int(bottom / FRAME_RESIZE_SCALE)
            left = int(left / FRAME_RESIZE_SCALE)

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.putText(frame, "Press 'q' to quit", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Face Recognition Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session ended. Attendance saved to {csv_path}")


if __name__ == "__main__":
    main()
