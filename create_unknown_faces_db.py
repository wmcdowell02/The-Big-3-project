import cv2
from deepface import DeepFace
import os
from openpyxl import Workbook
from openpyxl.drawing.image import Image as OpenpyxlImage
from PIL import Image as PILImage
import io

# Load face recognition model
DeepFace.build_model("VGG-Face")

# Load Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Excel file name
excel_file = 'unrecognized_faces_analysis.xlsx'

# Function to analyze face and store results
def analyze_and_store_face(image_path):
    if not isinstance(image_path, str):
        print(f"Error: image_path must be a string, got {type(image_path)} instead.")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to load image from {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    if len(faces) != 1:
        print(f"Error: A single face should be detected in {image_path}. Detected {len(faces)} faces.")
        return None

    try:
        analysis = DeepFace.analyze(image_path, actions=['age', 'gender', 'race', 'emotion'])
        if isinstance(analysis, list):
            analysis_data = analysis[0] if len(analysis) > 0 else {}
            gender_confidence = analysis_data.get('gender', {})
            dominant_gender = 'Woman' if gender_confidence.get('Woman', 0) > 90 else 'Man'
            main_attributes = {
                'image_title': os.path.splitext(os.path.basename(image_path))[0],
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

# Function to initialize the Excel file
def initialize_excel(file_name):
    wb = Workbook()
    ws = wb.active
    ws.append(['image_title', 'age', 'gender', 'race', 'emotion'])  # 'image' column header removed
    wb.save(filename=file_name)
    return wb

# Path to the unrecognized faces folder
unrecognized_faces_folder = 'unrecognized_faces'

# Initialize the Excel workbook
wb = initialize_excel(excel_file)
ws = wb.active

# Analyze each face in the unrecognized_faces folder and save the results
row_number = 2  # Start from the second row (first row is headers)
for face_filename in os.listdir(unrecognized_faces_folder):
    face_path = os.path.join(unrecognized_faces_folder, face_filename)
    face_data = analyze_and_store_face(face_path)

    if face_data:
        # Resize image using Pillow
        with PILImage.open(face_path) as pil_img:
            pil_img = pil_img.resize((50, 50))  # Resize image (50x50 is an example size)
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

        # Create Openpyxl Image
        openpyxl_img = OpenpyxlImage(io.BytesIO(img_byte_arr))
        cell_image_coordinate = f'F{row_number}'  # Column F for images
        ws.add_image(openpyxl_img, cell_image_coordinate)

        # Append row data without image
        row = [face_data['image_title'], face_data['age'], face_data['gender'], face_data['race'], face_data['emotion']]
        ws.append(row)
        row_number += 1

wb.save(filename=excel_file)

print("Analysis of unrecognized faces completed and saved to Excel.")
