import cv2
import csv
from deepface import DeepFace
import os

# Set the default model to VGG-Face for DeepFace
DeepFace.build_model("VGG-Face")

# Load a pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# CSV file name
csv_file = 'face_analysis.csv'

# Function to analyze face and store results
def analyze_and_store_face(image_path):
    # Ensure image_path is a string
    if not isinstance(image_path, str):
        print(f"Error: image_path must be a string, got {type(image_path)} instead.")
        return None

    # Read the image
    image = cv2.imread(image_path)
    
    # Check if the image is loaded
    if image is None:
        print(f"Error: Failed to load image from {image_path}")
        return None

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))  # Adjust parameters

    # Check if a single face is detected
    if len(faces) != 1:
        print(f"Error: A single face should be detected in {image_path}. Detected {len(faces)} faces.")
        return None

    try:
        # Analyze the face for attributes
        analysis = DeepFace.analyze(image_path, actions=['age', 'gender', 'race', 'emotion'])

        # Handle the analysis data as a list
        if isinstance(analysis, list):
            # Adjust the following line based on the actual list structure
            analysis_data = analysis[0] if len(analysis) > 0 else {}

            # Get gender predictions with confidence scores
            gender_predictions = analysis_data.get('gender', {})

            # Choose the dominant gender with a flexible threshold
            dominant_gender = None
            max_confidence = 0

            for gender, confidence in gender_predictions.items():
                if confidence > max_confidence:
                    dominant_gender = gender
                    max_confidence = confidence

            if dominant_gender is None:
                dominant_gender = 'N/A'

            main_attributes = {
                'image_title': os.path.splitext(os.path.basename(image_path))[0],  # Extract image title
                'age': analysis_data.get('age'),
                'gender': dominant_gender,
                'race': analysis_data.get('dominant_race', 'N/A'),
                'emotion': analysis_data.get('dominant_emotion', 'N/A')
            }
        else:
            print(f"Unexpected data structure: {type(analysis)}")
            return None

        return main_attributes
    except Exception as e:
        print(f"An error occurred during DeepFace analysis: {e}")
        return None

# Clear previous entries in the CSV file and write column headers
def initialize_csv(file_name):
    headers = ['image_title', 'age', 'gender', 'race', 'emotion']

    try:
        with open(file_name, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()  # Write column headers
    except Exception as e:
        print(f"Error initializing CSV: {e}")

# Paths to the best faces from the previous program
best_face_paths = [] # Add your own list of strings consisting of the file paths for each image

# Initialize the CSV file with column headers
initialize_csv(csv_file)

for path in best_face_paths:
    face_data = analyze_and_store_face(path)

    # Save the analysis results to the CSV file
    if face_data:
        with open(csv_file, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=face_data.keys())
            writer.writerow(face_data)

print("Face analysis completed and saved to CSV.")
