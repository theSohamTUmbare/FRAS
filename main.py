import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os

# Define the directory containing known faces
known_faces_dir = "faces"

video_capture = cv2.VideoCapture(0)

# Initialize empty lists
known_faces = []
known_face_encodings = []
known_face_names = []

# Loop through all files in the faces directory
for filename in os.listdir(known_faces_dir):
    image_path = os.path.join(known_faces_dir, filename)
    encoding = face_recognition.face_encodings(face_recognition.load_image_file(image_path))[0]
    known_faces.append({"encoding": encoding, "name": filename.split(".")[0]})
    known_face_encodings.append(encoding)
    known_face_names.append(filename.split(".")[0])

# Convert known face names to a set for faster lookup
known_face_names_set = set(known_face_names)

# List of expected students (initially set to all known faces)
students = list(known_face_names_set)

face_locations = []
face_encodings = []

# Get current date and time
now = datetime.now()
current_date = now.strftime("%d-%m-%Y")

# Open attendance file for writing
f = open("attendance "+current_date+".csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    # Mirror
    frame = cv2.flip(frame, 1)

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)
        threshold = 0.45
        if (matches[best_match_index]) and (face_distance[best_match_index] <= threshold):
            name = known_face_names[best_match_index]

            # Calculate the center coordinates of the face
            center_x = int((left + right) / 0.9)
            center_y = int((top + bottom) / 0.9)
            xd = int((150-center_x)/2)
            yd = int((150-center_y)/2)
            # Draw a red rectangle around the face, centered at the calculated center coordinates
            cv2.rectangle(frame, (center_x+70-xd, center_y+50-yd), (center_x + 210-xd, center_y + 220-yd), (3,83,163), 2)
            cv2.rectangle(frame, (center_x + 70-xd, center_y + 180-yd), (center_x + 210-xd, center_y + 220-yd), (3,83,163), -1)


            # Add the text with the person's name in white color, centered below the rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_x = center_x+100-xd
            text_y = center_y + 210-yd
            cv2.putText(frame, name, (text_x, text_y), font, 0.7, (255,255,255), 1)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()



