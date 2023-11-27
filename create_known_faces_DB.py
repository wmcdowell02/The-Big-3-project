import cv2
import csv
from deepface import DeepFace

# Set the default model to VGG-Face for DeepFace
DeepFace.build_model("VGG-Face")

# Load a pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

            main_attributes = {
                'image_path': image_path,
                'age': analysis_data.get('age'),
                'gender': analysis_data.get('gender'),
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

# Function to save analysis to a CSV file
def save_analysis_to_csv(face_data, file_name='face_analysis.csv'):
    headers = ['image_path', 'age', 'gender', 'race', 'emotion']

    try:
        with open(file_name, 'a', newline='') as file:  # Use 'a' mode to append to the CSV
            writer = csv.DictWriter(file, fieldnames=headers)
            if face_data is not None:  # Check if entry is not None
                writer.writerow(face_data)
    except Exception as e:
        print(f"Error saving data to CSV: {e}")

# Clear previous entries in the CSV file
def clear_csv(file_name='face_analysis.csv'):
    try:
        with open(file_name, 'w', newline='') as file:  # Use 'w' mode to clear the CSV
            pass
    except Exception as e:
        print(f"Error clearing CSV: {e}")

# Paths to the best faces from the previous program
best_face_paths = ['best_faces/best_face_1/best_face_1.jpg', 'best_faces/best_face_2/best_face_2.jpg', 'best_faces/best_face_3/best_face_3.jpg']

# Clear the CSV file
clear_csv()

for path in best_face_paths:
    face_data = analyze_and_store_face(path)

    # Save the analysis results to the CSV file
    save_analysis_to_csv(face_data)

print("Face analysis completed and saved to CSV.")
