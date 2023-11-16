import cv2
import csv
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
            writer = csv.DictWriter(file, fieldnames=faces_db[0].keys())
            writer.writeheader()
            for face in faces_db:
                writer.writerow(face)
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

# Function to analyze face from an image file and store results
def analyze_and_store_face_from_path(image_path):
    try:
        # Analyze the face for attributes
        analysis = DeepFace.analyze(image_path, actions=['age', 'gender', 'race', 'emotion'])

        # Extract main attributes
        main_attributes = {
            'image_path': image_path,
            'age': analysis['age'],
            'gender': analysis['gender'],
            'race': analysis['dominant_race'],
            'emotion': analysis['dominant_emotion']
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
faces_to_analyze = ['face1.jpg', 'face2.jpg', 'face3.jpg']  # Replace with actual image paths
initialize_known_faces(faces_to_analyze)
