# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 21:15:16 2024

@author: public
"""

import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,  # 0 for BlazeFace short-range, 1 for full-range
    min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Path to your image dataset folder
dataset_path = os. getcwd()
# Iterate through all images in the dataset
for filename in os.listdir(dataset_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Check for image extensions
        image_path = os.path.join(dataset_path, filename)
        image = cv2.imread(image_path)

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with BlazeFace
        results = face_detection.process(image_rgb)

        # Draw the detections on the original image
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        # Display the image with detections (optional)
        cv2.imshow('Face Detection', image)
        cv2.waitKey(0)

# Close any open windows
cv2.destroyAllWindows() 