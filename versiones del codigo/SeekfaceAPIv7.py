# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:50:29 2024

@author: public
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV, train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Data storage
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

# --- Prepare data for training ---
X = np.array(embeddings)
y = np.array(names)

# Convert labels to one-hot encoding
name_to_label = {name: i for i, name in enumerate(set(names))}
y_encoded = np.array([name_to_label[name] for name in y])
y_onehot = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2)

# --- Keras Model ---
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(len(set(names)), activation='softmax'))  # Output layer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- Hyperparameter Tuning (optional) ---
param_grid = {
    'loss': [64, 128, 256],
    'epochs': [10, 20, 30]
}

#svc = svm.SVC()
estimator = KerasClassifier(build_fn=lambda:Sequential(), epochs=100) #, hidden_units=64)
try:
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

# --- Training ---
    best_model.fit(X_train, y_train, epochs=10, batch_size=32)  # Adjust epochs and batch size
except:  ValueError
# --- Testing Phase ---
#test_image_path = input("/prueba1/IMAGE_20240524_103403_552.jpg")
test_image = cv2.imread("prueba1/IMAGE_20240524_103403_552.jpg")
test_image_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

test_landmarks = extract_landmarks(test_image_rgb)
if test_landmarks:
    flattened_test_landmarks = flatten_landmarks(test_landmarks)
    test_data = np.array([flattened_test_landmarks])
    prediction = best_model.predict(test_data)
    predicted_class = np.argmax(prediction)
    predicted_name = list(name_to_label.keys())[list(name_to_label.values()).index(predicted_class)]
    print(f"Predicted name: {predicted_name}")
else:
    print("Face not detected")