# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 22:22:09 2023

@author: William McDowell
"""

from deepface import DeepFace
import os
import cv2
import numpy as np


# Paths to the folders containing the face cutouts and the database of photos for verification
folder_cutouts = 'C:/Users/William McDowell/OneDrive/Fall 2023/M E 369P/Project/face_cutouts'
folder_photos = '"C:/Users/William McDowell/OneDrive/Fall 2023/M E 369P/Project/databases"'
# Load face recognition model
model = DeepFace.build_model("Facenet")

# Function to preprocess images before passing to the model
def preprocess_images(img_paths):
    images = []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 160))  # Resize image to match the model's input shape
        images.append(img)
    return np.array(images)

# Function to extract facial embeddings from images
def extract_embeddings(img_paths):
    images = preprocess_images(img_paths)
    embeddings = model.predict(images)
    return embeddings

# Function to extract embeddings from a folder of images
def extract_embeddings_from_folder(folder_path):
    img_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]
    return extract_embeddings(img_paths)

# Function to extract embeddings from a database of folders containing photos
def extract_embeddings_from_database(database_path):
    database_embeddings = {}
    for folder in os.listdir(database_path):
        folder_path = os.path.join(database_path, folder)
        if os.path.isdir(folder_path):
            database_embeddings[folder] = extract_embeddings_from_folder(folder_path)
    return database_embeddings

# Extract embeddings from the cutouts and database folders
cutouts_embeddings = extract_embeddings_from_folder(folder_cutouts)
database_embeddings = extract_embeddings_from_database(folder_photos)

# Perform face verification
for cutout_name, cutout_embedding in cutouts_embeddings.items():
    for folder_name, folder_embeddings in database_embeddings.items():
        for photo_embedding in folder_embeddings:
            distance = DeepFace.distance(cutout_embedding, photo_embedding)
            if distance < 0.6:  # Adjust the threshold as needed
                print(f"Match found: {cutout_name} matches a photo in folder {folder_name}")
            else:
                print(f"No match: {cutout_name} does not match any photo in folder {folder_name}")