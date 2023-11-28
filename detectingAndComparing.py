# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:14:55 2023

@author: William McDowell
"""

import cv2
import os
import numpy as np
from deepface import DeepFace
from scipy.spatial import distance as dist

# Load face recognition model
model = DeepFace.build_model("Facenet")

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess images before passing to the model
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (160, 160))  # Resize image to match the model's input shape
    return img

# Function to extract facial embeddings from an image
def extract_embedding(img):
    img = preprocess_image(img)
    img_embedding = model.predict(np.expand_dims(img, axis=0))[0]
    return img_embedding

# Function to extract embeddings from a folder of images
def extract_embeddings_from_folder(folder_path):
    embeddings = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img_embedding = extract_embedding(img)
            embeddings.append(img_embedding)
    return embeddings

# Load embeddings from the database folder
database_path = 'C:/Users/William McDowell/OneDrive/Fall 2023/M E 369P/Project/databases' # Pathway to database folder
known_embeddings = {}

for person_folder in os.listdir(database_path):
    person_folder_path = os.path.join(database_path, person_folder)
    if os.path.isdir(person_folder_path):
        embeddings = extract_embeddings_from_folder(person_folder_path)
        known_embeddings[person_folder] = embeddings

# Initialize camera capture
camera = cv2.VideoCapture(0)  # 0 for default camera, change if needed

while True:
    ret, frame = camera.read()
    
    # Perform face detection using the Haar Cascade classifier
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Extract embeddings for each detected face
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_embedding = extract_embedding(face_img)
        
        # Compare with known embeddings
        for person_name, embeddings in known_embeddings.items():
            for known_embedding in embeddings:
                distance = dist.cosine(face_embedding, known_embedding)
                if distance < 0.07:  # Adjust the threshold as needed
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw a green rectangle around the matched face
                    cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    break  # Break loop if a match is found
    
    # Display the frame with detected faces
    cv2.imshow('Real-time Face Matching', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()

