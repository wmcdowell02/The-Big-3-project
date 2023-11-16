import cv2
import csv
from deepface import DeepFace
import numpy as np

# CSV File Names
KNOWN_FACES_CSV = 'known_faces.csv'
UNKNOWN_FACES_CSV = 'unknown_faces.csv'

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

# Function to analyze face and store results
def analyze_and_store_face(image):
    try:
        # Convert the captured image to a format suitable for analysis
        _, img_encoded = cv2.imencode('.jpg', image)
        img_bytes = img_encoded.tobytes()

        # Analyze the face for attributes
        analysis = DeepFace.analyze(img_bytes, actions=['age', 'gender', 'race', 'emotion'])

        # Extract main attributes
        main_attributes = {
            'age': analysis['age'],
            'gender': analysis['gender'],
            'race': analysis['dominant_race'],
            'emotion': analysis['dominant_emotion']
        }
        return main_attributes
    except Exception as e:
        print(f"An error occurred during face analysis: {e}")
        return None
