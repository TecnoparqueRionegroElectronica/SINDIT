# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:31:37 2024

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
mp_drawing = mp.solutions.drawing_utils

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

# Training Phase 
dataset_path = os.getcwd()
for filename in os.listdir(dataset_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        landmarks = extract_landmarks(image_rgb)
        if landmarks:
            name = input(f"Ingrese el nombre de la persona en la imagen {filename}: ")
            names.append(name)

            flattened_landmarks = flatten_landmarks(landmarks)
            embeddings.append(flattened_landmarks)

# Preparing data for training 
X = np.array(embeddings)
y = np.array(names)

# Converting labels to one-hot encoding
name_to_label = {name: i for i, name in enumerate(set(names))}
y_encoded = np.array([name_to_label[name] for name in y])
y_onehot = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2)

# Setup of Keras Model 
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(len(set(names)), activation='softmax'))  # Output layer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Hyperparameter Tuning  
param_grid = {
    'loss': [1, 2, 4], 
    'epochs': [10, 20, 30]
}
estimator = KerasClassifier(build_fn=lambda: model, epochs=100) #, hidden_units=64)
grid = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Training
best_model.fit(X_train, y_train, epochs=10, batch_size=32)  # Adjust epochs and batch size

# Testing Phase 
test_folder = "test_images"  # Name of your test folder
output_folder = "resultados"  # Folder to save processed images

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
#face marking
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
model_selection=1,  # 0 for BlazeFace short-range, 1 for full-range
min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

for filename in os.listdir(test_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(test_folder, filename)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        org = (50, 50)
        test_landmarks = extract_landmarks(image_rgb)
        if test_landmarks:
            try:
                flattened_test_landmarks = flatten_landmarks(test_landmarks)
                test_data = np.array([flattened_test_landmarks])
                prediction = best_model.predict(test_data)
                predicted_class = np.argmax(prediction)
                predicted_name = list(name_to_label.keys())[list(name_to_label.values()).index(predicted_class)]
                # Draw name labels and save the image
                print(f"Imagen: {filename}, Identificado: {predicted_name}")
                
                output_path = os.path.join(output_folder, filename)
                results = face_detection.process(image_rgb)

                # Draw the detections on the original image
                if results.detections:
                    for detection in results.detections:
                        mp_drawing.draw_detection(image, detection)
                        cv2.putText(image, predicted_name, org=org, fontFace=cv2.FONT_HERSHEY_DUPLEX  , fontScale=10, color=(0,255,0), thickness=2, lineType = cv2.LINE_8, bottomLeftOrigin = False)
                        cv2.imwrite(output_path, image)
                        
            
            except: ValueError


        else:
            print(f"Imagen: {filename}, No se identifican rostros")