import cv2
import csv
import numpy as np
from deepface import DeepFace

# CSV File Names
KNOWN_FACES_CSV = 'known_faces.csv'

# Function to load faces database from a CSV file
def load_faces_db(file_name):
    faces_db = []
    try:
        with open(file_name, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                faces_db.append(row)
    except FileNotFoundError:
        print(f"File {file_name} not found. Starting with an empty database.")
    return faces_db

# Function to save faces database to a CSV file
def save_faces_db(file_name, faces_db):
    try:
        with open(file_name, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['image_path', 'age', 'gender', 'race', 'emotion', 'embedding'])
            writer.writeheader()
            for face in faces_db:
                # Convert embedding to a string for CSV storage
                face['embedding'] = str(face['embedding'])
                writer.writerow(face)
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

# Function to analyze face from an image file and store results
def analyze_and_store_face_from_path(image_path):
    try:
        # Analyze the face for attributes
        result = DeepFace.analyze(image_path, actions=['age', 'gender', 'race', 'emotion'])

        # Check if the result is a list and take the first item
        analysis = result[0] if isinstance(result, list) else result

        # Extract embeddings
        embedding = DeepFace.represent(image_path, enforce_detection=False)

        # Check if embedding extraction was successful
        if embedding is None or not isinstance(embedding, (list, np.ndarray)):
            raise ValueError(f"Failed to extract embedding from {image_path}")

        main_attributes = {
            'image_path': image_path,
            'age': analysis['age'],
            'gender': analysis['gender'],
            'race': analysis['dominant_race'],
            'emotion': analysis['dominant_emotion'],
            'embedding': embedding  # Store the embedding
        }
        return main_attributes
    except Exception as e:
        print(f"An error occurred during DeepFace analysis: {e}")
        return None

# Function to initialize the known faces database with provided images
def initialize_known_faces(faces_paths):
    known_faces = []
    for path in faces_paths:
        face_data = analyze_and_store_face_from_path(path)
        if face_data:
            known_faces.append(face_data)
        else:
            print(f"Failed to analyze face in {path}")

    save_faces_db(KNOWN_FACES_CSV, known_faces)

# Example usage: Initialize known faces database
faces_to_analyze = ['Olivia1.jpg', 'Matthew1.jpg', 'William1.jpg']  # Replace with actual image paths
initialize_known_faces(faces_to_analyze)
