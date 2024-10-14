import cv2
import numpy as np
import os
import csv
import time
import pickle
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

# Load face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

with open("data/names.pkl", 'rb') as w:
    LABELS = pickle.load(w)

with open('data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Column names for the CSV file
COL_NAMES = ['NAME', 'TIME']

# Ensure the 'Attendance' folder exists
if not os.path.exists('Attendance'):
    os.makedirs('Attendance')

while True:
    ret, frame = video.read()

    # Resize frame for better processing (optional)
    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:  # Ensure faces are detected
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)

            # Timestamp
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

            # Check if the attendance file exists
            exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

            # Draw a rectangle around the face and display the predicted label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, str(output[0]), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Prepare attendance data
            attendance = [str(output[0]), str(timestamp)]

    # Display the frame directly
    cv2.imshow("Attendance System", frame)
    k = cv2.waitKey(1)

    if k == ord('0'):
        time.sleep(5)

        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open("Attendance/Attendance_" + date + ".csv", "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)

    if k == ord('q'):
        break

# Release the video capture and destroy windows
video.release()
cv2.destroyAllWindows()