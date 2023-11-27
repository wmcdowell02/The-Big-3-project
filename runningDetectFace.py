# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 21:09:25 2023

@author: William McDowell
"""

import cv2
import os
import time

# Create a directory to store face cutouts
output_folder = 'face_cutouts'
os.makedirs(output_folder, exist_ok=True)

# Initialize face detection with Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize camera capture
camera = cv2.VideoCapture(0)  # 0 for default camera, you can change this if needed

# Set time limit for capturing
capture_time = 60  # in seconds
start_time = time.time()

while (time.time() - start_time) < capture_time:
    ret, frame = camera.read()
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    print(f"Number of faces detected: {len(faces)}")
    print(f"Face data: {faces}")
    
    for (x, y, w, h) in faces:
        try:
            # Extract face cutout
            face_cutout = frame[y:y+h, x:x+w]
            
            # Save face cutout with a unique filename
            timestamp = int(time.time())
            filename = f"{output_folder}/face_{timestamp}.jpg"
            cv2.imwrite(filename, face_cutout)
            
        except Exception as e:
            print(f"Error: {e}")
    
    cv2.imshow('Camera Feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
camera.release()
cv2.destroyAllWindows()
