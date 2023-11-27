from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2
import os
import csv
import numpy as np

# Path to the database of known faces
db_path = 'C:/Users/Administrator/Desktop/App programming Project/The-Big-3-project-main/The-Big-3-project-main/best_faces'

# Create the output folder for cutout faces if it doesn't exist
cutout_folder = 'cutout_faces'
os.makedirs(cutout_folder, exist_ok=True)

# CSV file for storing attributes of unknown faces
unknown_faces_csv = 'unknown_faces.csv'

# CSV headers
csv_headers = ['Image_Path', 'Age', 'Gender', 'Race', 'Emotion']

# Create a function to process frames
def process_frame(frame):
    # Detect faces in the frame
    detected_faces = DeepFace.detectFace(frame, detector_backend='mtcnn')

    if detected_faces is None:
        print("No faces detected in the frame.")
        return

    for i, face in enumerate(detected_faces):
        try:
            # Embedding of the detected face
            face_embedding = DeepFace.represent(face, model_name='VGG-Face')

            # List of embeddings of known faces in the database
            known_face_embeddings = []

            # List of corresponding names of known faces
            known_face_names = []

            # Load known face embeddings and names from the database
            for filename in os.listdir(db_path):
                image_path = os.path.join(db_path, filename)
                name = os.path.splitext(filename)[0]  # Extract the name from the filename
                known_face_embeddings.append(np.load(image_path))
                known_face_names.append(name)

            # Calculate the similarity between the detected face and known faces
            similarities = DeepFace.find(img_path=face_embedding, db_path=known_face_embeddings, model_name='VGG-Face')
            min_similarity = min(similarities)

            # Set a threshold for face recognition
            similarity_threshold = 0.6

            if min_similarity >= similarity_threshold:
                # Display the best match from the database
                best_match_index = similarities.index(min_similarity)
                best_match_name = known_face_names[best_match_index]
                print(f"User already detected: Best match - {best_match_name}")

            else:
                # Create a cutout image of the detected face
                x, y, w, h = face['box']
                cutout_face = frame[y:y+h, x:x+w]

                # Save the cutout face as an image
                cutout_path = os.path.join(cutout_folder, f"cutout_face_{len(os.listdir(cutout_folder)) + 1}.jpg")
                cv2.imwrite(cutout_path, cutout_face)
                print(f"Unknown user detected: Cutout face saved as {cutout_path}")

                # Analyze the attributes of the unknown face
                try:
                    analysis = DeepFace.analyze(cutout_path, actions=['age', 'gender', 'race', 'emotion'])

                    if isinstance(analysis, dict):
                        age = analysis.get('age')
                        gender = analysis.get('gender')
                        race = analysis.get('dominant_race', 'N/A')
                        emotion = analysis.get('dominant_emotion', 'N/A')

                        # Export the attributes to the CSV file
                        with open(unknown_faces_csv, mode='a', newline='') as csv_file:
                            csv_writer = csv.writer(csv_file)
                            csv_writer.writerow([cutout_path, age, gender, race, emotion])
                except Exception as e:
                    print(f"An error occurred during DeepFace analysis: {e}")

            # Display the frame with face detection and recognition
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title(f"Similarity: {min_similarity:.2f}")
            plt.show()

        except Exception as e:
            print(f"An error occurred: {e}")

# Initialize the video capture from a camera (you can specify the camera index)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error capturing frame.")
        break

    # Process the frame
    process_frame(frame)

    # Break the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
