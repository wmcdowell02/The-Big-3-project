import cv2
import os
import numpy as np

# Load a pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to extract and save the best face from an image
def extract_and_save_best_face(input_images, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for i, image_path in enumerate(input_images):
        # Ensure image_path is a string
        if not isinstance(image_path, str):
            print(f"Error: image_path must be a string, got {type(image_path)} instead.")
            continue

        # Read the image
        image = cv2.imread(image_path)
        
        # Check if the image is loaded
        if image is None:
            print(f"Error: Failed to load image from {image_path}")
            continue

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Find the best face based on area (largest face)
        if len(faces) > 0:
            best_face_idx = np.argmax([w * h for x, y, w, h in faces])
            x, y, w, h = faces[best_face_idx]
            best_face = image[y:y+h, x:x+w]
            
            # Get the filename without extension
            filename_without_extension = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save the best face as an image with the original image's filename followed by "_face"
            output_path = os.path.join(output_folder, f"{filename_without_extension}_face.jpg")
            cv2.imwrite(output_path, best_face)
            print(f"Face from Image {i + 1} saved as {output_path}")
        else:
            print(f"No faces detected in Image {i + 1}")

# Example usage
input_images = ['databases/Olivia/Olivia1.jpg', 'databases/Matthew/Matthew1.jpg', 'databases/William/William1.jpg']  # Replace with the paths to your input images
output_folder = 'best_faces'  # Folder where best faces will be saved
extract_and_save_best_face(input_images, output_folder)
