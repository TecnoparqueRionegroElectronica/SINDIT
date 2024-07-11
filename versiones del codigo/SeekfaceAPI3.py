# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 19:12:33 2024

@author: public
"""

import cv2
import mediapipe as mp
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# --- Configuration ---
DATASET_PATH = os.getcwd()  # Path to your image dataset folder
CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold for face recognition
SAVE_UNKNOWN_FACES = True # Flag to save images of unknown faces
UNKNOWN_FACES_PATH = 'unknown_faces' # Where to save unknown faces

# --- Global Variables ---
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
knn = None  # Initialize KNN classifier
embeddings = []
names = []

# --- Load existing embeddings and names ---
embeddings_file = 'embeddings.npy'
names_file = 'names.txt'
if os.path.exists(embeddings_file) and os.path.exists(names_file):
    embeddings = np.load(embeddings_file).tolist()
    with open(names_file, 'r') as f:
        names = f.read().splitlines()
    if len(set(names)) >= 2:  # Need at least two different identities
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(np.vstack(embeddings), names)
        print("KNN Classifier loaded and ready.")

# --- Function to save embeddings and names ---
def save_data():
    np.save(embeddings_file, np.array(embeddings))
    with open(names_file, 'w') as f:
        for name in names:
            f.write(f"{name}\n")

# --- Function to get face embedding ---
def get_embedding(face_roi):
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    face_cropped = mtcnn(face_roi)
    if face_cropped is not None:
        face_cropped = face_cropped.unsqueeze(0).to(device)
        embedding = resnet(face_cropped).detach().cpu().numpy()
        return embedding
    else:
        return None

# --- Function to recognize faces in an image ---
def recognize_faces(image):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * image.shape[1])
            y = int(bbox.ymin * image.shape[0])
            w = int(bbox.width * image.shape[1])
            h = int(bbox.height * image.shape[0])
            face_roi = image[y:y + h, x:x + w]

            embedding = get_embedding(face_roi)
            if embedding is not None:
                if knn:
                    # Predict using KNN
                    prediction = knn.predict(embedding)
                    confidence = knn.predict_proba(embedding).max()
                    if confidence > CONFIDENCE_THRESHOLD:
                        name = prediction[0]
                        cv2.putText(image, f"{name} {confidence:.2f}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.putText(image, "Unknown", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        if SAVE_UNKNOWN_FACES:
                            if not os.path.exists(UNKNOWN_FACES_PATH):
                                os.makedirs(UNKNOWN_FACES_PATH)
                            cv2.imwrite(os.path.join(UNKNOWN_FACES_PATH, 
                                        f"unknown_{len(os.listdir(UNKNOWN_FACES_PATH))}.jpg"), face_roi)
                else:
                    cv2.putText(image, "Not enough data to classify", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            mp_drawing.draw_detection(image, detection)
    return image


# --- Main Application ---
if __name__ == "__main__":
    # --- Choose input source (image or video) ---
    input_choice = input("Enter 'image' for image or 'video' for video input: ").lower()

    if input_choice == 'image':
        image_path = input("Enter the path to your image: ")
        image = cv2.imread(image_path)
        recognized_image = recognize_faces(image)
        cv2.imshow('Face Recognition', recognized_image)
        cv2.waitKey(0)
    elif input_choice == 'video':
        video_source = input("Enter video source (0 for webcam, path for file): ")
        try:
            video_source = int(video_source)  # Try converting to integer for webcam
        except ValueError:
            pass  # Keep as string if it's a path

        cap = cv2.VideoCapture(video_source)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            recognized_frame = recognize_faces(frame)
            cv2.imshow('Face Recognition', recognized_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
    else:
        print("Invalid input choice. Please enter 'image' or 'video'.")

    cv2.destroyAllWindows()