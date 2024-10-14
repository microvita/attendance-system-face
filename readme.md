
# Face Capture and Storage using OpenCV

This project captures face data using a webcam and stores the data along with the user's name into pickle files. 
The captured face images are processed using OpenCV and saved for later use in tasks like facial recognition.

## Code Breakdown

### 1. **Importing Required Libraries**
```python
import cv2
import numpy as np
import os
import pickle
```
- **cv2**: The OpenCV library is used for real-time image processing and face detection.
- **numpy**: Used to handle arrays and manipulate face data.
- **os**: Handles file operations such as checking if a directory exists.
- **pickle**: Serializes and saves data (faces and names) to disk for later use.

### 2. **Initialize Webcam and Face Detection**
```python
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
```
- **video**: Initializes the webcam for video capture. `0` refers to the primary camera.
- **CascadeClassifier**: Loads the pre-trained Haar Cascade model for detecting faces.

### 3. **Prepare for Storing Face Data**
```python
face_data = []
i = 0
name = input("Enter your name: ")
```
- **face_data**: An empty list to store cropped face images.
- **i**: A counter that ensures images are captured at intervals.
- **name**: The name input by the user, which will be linked to the captured face data.

### 4. **Ensure the 'data' Folder Exists**
```python
if not os.path.exists('data'):
    os.makedirs('data')
```
- Checks if a folder named `data` exists. If it doesn't, the folder is created to store face data and names.

### 5. **Main Loop for Video Capture and Face Detection**
```python
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
```
- **video.read()**: Captures a single frame from the webcam.
- **cvtColor**: Converts the frame to grayscale (needed by the face detector).
- **detectMultiScale**: Detects faces in the grayscale image and returns rectangles representing the location of detected faces.

### 6. **Processing Detected Faces**
```python
for (x, y, w, h) in faces:
    crop_img = frame[y:y+h, x:x+w]
    resized_img = cv2.resize(crop_img, (50, 50))

    if len(face_data) <= 100 and i % 10 == 0:
        face_data.append(resized_img)
        cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
```
- **for (x, y, w, h) in faces**: Loops through each detected face's coordinates.
- **crop_img**: Crops the face from the frame.
- **resized_img**: Resizes the cropped face to a 50x50 pixel image.
- Every 10 frames, a face image is appended to `face_data` until 100 images are collected.
- The total number of collected images is displayed on the screen with `putText`.
- A rectangle is drawn around the detected face using `rectangle`.

### 7. **Display the Frame**
```python
cv2.imshow("frame", frame)
i += 1

k = cv2.waitKey(1)
if len(face_data) == 100:
    break
```
- **imshow**: Displays the current video frame in a window.
- **waitKey(1)**: Keeps the video feed running and waits for a key press.
- If 100 face images are collected, the loop breaks, ending the capture.

### 8. **Cleanup**
```python
video.release()
cv2.destroyAllWindows()
```
- Releases the webcam and closes all OpenCV windows once the face data is collected.

### 9. **Reshape and Prepare Face Data for Saving**
```python
face_data = np.array(face_data)
face_data = face_data.reshape(100, -1)
```
- Converts the list of face images to a NumPy array.
- Reshapes the array so each image becomes a 1D vector (2500 elements for a 50x50 image).

### 10. **Saving Names**
```python
if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
        names = names + [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
```
- Checks if `names.pkl` exists in the `data` folder.
- If not, a list of 100 instances of the user's name is created and saved.
- If `names.pkl` exists, the names are loaded, updated, and saved back to the file.

### 11. **Saving Face Data**
```python
if 'face_data.pkl' not in os.listdir('data/'):
    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open('data/face_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, face_data, axis=0)
    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
```
- Checks if `face_data.pkl` exists.
- If not, the current `face_data` is saved.
- If it exists, the existing face data is loaded, the new data is appended, and the combined data is saved.

## Full Code

```python
import cv2
import numpy as np
import os
import pickle

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_data = []

i = 0
name = input("Enter your name: ")

# Ensure the 'data' folder exists
if not os.path.exists('data'):
    os.makedirs('data')

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w]
        resized_img = cv2.resize(crop_img, (50, 50))

        if len(face_data) <= 100 and i % 10 == 0:
            face_data.append(resized_img)
            cv2.putText(frame, str(len(face_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    cv2.imshow("frame", frame)
    i += 1

    k = cv2.waitKey(1)
    if len(face_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

# Save faces in pickle file
face_data = np.array(face_data)
face_data = face_data.reshape(100, -1)

if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
        names = names + [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'face_data.pkl' not in os.listdir('data/'):
    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open('data/face_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, face_data, axis=0)
    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
```

## Summary
- The code captures 100 face images from the webcam and associates them with the user's name.
- The face data and names are saved in two files: `face_data.pkl` and `names.pkl`.
- This data can be used in facial recognition or other identity verification tasks.


## Attendance Code Flow

