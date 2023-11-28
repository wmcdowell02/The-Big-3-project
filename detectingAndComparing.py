import cv2
import os
import numpy as np
from scipy.spatial import distance as dist
from deepface import DeepFace
import time

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

# Function to calculate cosine similarity between two embeddings
def cosine_distance(emb1, emb2):
    return dist.cosine(emb1, emb2)

# Function to verify if the detected face matches with the database
def verify_face(detected_embedding, database_embeddings, threshold=0.08):  # Adjust the threshold as needed
    for person_name, embeddings in database_embeddings.items():
        for known_embedding in embeddings:
            distance = cosine_distance(detected_embedding, known_embedding)
            if distance < threshold:
                return person_name  # Return the matched person's name
    return None  # Return None if no match is found

# Load embeddings from the database folder
database_path = 'C:/Users/Administrator/Desktop/App programming Project/The-Big-3-project-main/The-Big-3-project-main/best_faces'
known_embeddings = {}

for person_folder in os.listdir(database_path):
    person_folder_path = os.path.join(database_path, person_folder)
    if os.path.isdir(person_folder_path):
        embeddings = []
        for filename in os.listdir(person_folder_path):
            img_path = os.path.join(person_folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img_embedding = extract_embedding(img)
                embeddings.append(img_embedding)
        known_embeddings[person_folder] = embeddings

# Initialize camera capture
camera = cv2.VideoCapture(0)  # 0 for default camera

# Set up for handling unrecognized faces
unrecognized_face_path = 'unrecognized_faces'
os.makedirs(unrecognized_face_path, exist_ok=True)

# Function to load embeddings of unrecognized faces
def load_unrecognized_embeddings():
    embeddings = []
    for filename in os.listdir(unrecognized_face_path):
        img_path = os.path.join(unrecognized_face_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img_embedding = extract_embedding(img)
            embeddings.append(img_embedding)
    return embeddings

unrecognized_embeddings = load_unrecognized_embeddings()

# Time threshold for saving unrecognized faces (in seconds)
time_threshold = 5
last_saved_time = time.time()

while True:
    ret, frame = camera.read()
    
    # Perform face detection using the Haar Cascade classifier
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Extract embeddings for each detected face
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        detected_embedding = extract_embedding(face_img)
        
        # Verify if the detected face matches with the database
        matched_person = verify_face(detected_embedding, known_embeddings)
        matched_unrecognized = verify_face(detected_embedding, {"unrecognized": unrecognized_embeddings}, threshold=0.1)

        if matched_person:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw a green rectangle around the matched face
            cv2.putText(frame, matched_person, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        elif not matched_unrecognized and time.time() - last_saved_time > time_threshold:
            last_saved_time = time.time()
            unrecognized_face_filename = f'unrecognized_face_{len(os.listdir(unrecognized_face_path)) + 1}.jpg'
            cv2.imwrite(os.path.join(unrecognized_face_path, unrecognized_face_filename), face_img)
            unrecognized_embeddings.append(detected_embedding)  # Update the embeddings list
    
    # Display the frame with detected faces
    cv2.imshow('Real-time Face Matching', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()
