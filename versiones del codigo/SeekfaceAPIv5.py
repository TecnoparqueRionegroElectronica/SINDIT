# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:33:16 2024

@author: public
"""

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --- Define a convolutional neural network model ---
class LandmarkClassifier(nn.Module):
    def __init__(self):
        super(LandmarkClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten for fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- Dataset for training ---
class FaceLandmarkDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- Data storage ---
embeddings = []
names = []


def extract_landmarks(image):
    results = face_mesh.process(image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.append([landmark.x, landmark.y])
            return landmarks
    return None

def flatten_landmarks(landmarks):
    return [coord for point in landmarks for coord in point]

# --- Training Phase ---
dataset_path = os.getcwd()
for filename in os.listdir(dataset_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
       image_path = os.path.join(dataset_path, filename)
       image = cv2.imread(image_path)
       image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
       landmarks = extract_landmarks(image_rgb)

    if landmarks:
        name = input(f"Enter the name for the person in image {filename}: ")
        names.append(name)

        flattened_landmarks = flatten_landmarks(landmarks)
        embeddings.append(flattened_landmarks)

# Normalize data
embeddings = np.array(embeddings) / np.max(embeddings)

# Prepare data for training
num_classes = len(set(names))
dataset = FaceLandmarkDataset(embeddings, names)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Instantiate model
model = LandmarkClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data.unsqueeze(1).float()) # Add channel dimension
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# --- Testing Phase ---
test_image_path = input("Enter the path to a test image: ")
test_image = cv2.imread(test_image_path)
test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

test_landmarks = extract_landmarks(test_image_rgb)
if test_landmarks:
    flattened_test_landmarks = flatten_landmarks(test_landmarks)
    test_tensor = torch.tensor(flattened_test_landmarks, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = model(test_tensor)
        predicted_class = torch.argmax(output).item()
        predicted_name = names[predicted_class]
        print(f"Predicted name: {predicted_name}")
else:
    print("Face not detected")