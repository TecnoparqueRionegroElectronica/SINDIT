# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 19:10:34 2024

@author: public
"""

import cv2
import mediapipe as mp
import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Check if CUDA is available and set device accordingly
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load FaceNet models and move to the selected device
mtcnn = MTCNN(image_size=160, margin=0, device=device)  
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, 
    min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Path to your image dataset folder
dataset_path = os.getcwd()

# Lists to store embeddings and corresponding names
embeddings = []
names = []

# --- Load existing embeddings and names if available ---
embeddings_file = 'embeddings.npy'
names_file = 'names.txt'

if os.path.exists(embeddings_file) and os.path.exists(names_file):
    embeddings = np.load(embeddings_file).tolist()
    with open(names_file, 'r') as f:
        names = f.read().splitlines()

# --- Function to save embeddings and names ---
def save_data():
    np.save(embeddings_file, np.array(embeddings))
    with open(names_file, 'w') as f:
        for name in names:
            f.write(f"{name}\n")

# Iterate through all images in the dataset
for filename in os.listdir(dataset_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path)

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with BlazeFace
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                # Draw bounding box (optional)
                mp_drawing.draw_detection(image, detection)

                # Extract face ROI using relative bounding box
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * image.shape[1])
                y = int(bbox.ymin * image.shape[0])
                w = int(bbox.width * image.shape[1])
                h = int(bbox.height * image.shape[0])
                face_roi = image[y:y+h, x:x+w]

                # Get embedding using FaceNet
                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB) 
                face_cropped = mtcnn(face_roi) 
                if face_cropped is not None:
                    face_cropped = face_cropped.unsqueeze(0).to(device) 
                    embedding = resnet(face_cropped)
                    embeddings.append(embedding.detach().cpu().numpy()) 
                    
                    # Get the person's name from the user 
                    name = input(f"Enter the name for the person in image {filename}: ")
                    names.append(name)

                    # Save updated embeddings and names
                    save_data()

                # Display the image with detections (optional)
                cv2.imshow('Face Detection', image)
                cv2.waitKey(0)

# Close any open windows
cv2.destroyAllWindows()

# --- Train KNN Classifier ---
if len(set(names)) >= 2: # Need at least two different identities
    knn = KNeighborsClassifier(n_neighbors=3) # You can adjust n_neighbors
    knn.fit(np.vstack(embeddings), names)
    print("KNN Classifier trained!")
else:
    print("Not enough unique identities to train the classifier.") 