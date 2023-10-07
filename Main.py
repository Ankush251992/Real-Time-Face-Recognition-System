#!/usr/bin/env python3
"""
Main.py: A Kivy-based facial recognition application.

Usage:
    python Main.py

Requirements:
    - Python 3.x
    - Kivy
    - OpenCV
    - Dlib
    - Put your face images in a folder named "Face_Directory"
    - Dlib shape predictor file: "shape_predictor_68_face_landmarks.dat"
    - Dlib face recognition model file: "dlib_face_recognition_resnet_model_v1.dat"

"""

import os
import cv2
import dlib
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window

# Set the window size for better aspect ratio
Window.size = (800, 600)

# Initialize Dlib face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Initialize dictionaries to hold face descriptors and corresponding labels
face_descriptors_dict = {}
label_dict = {}

# Folder where face images are stored
face_dir = "Face_Directory"

# Check if the directory exists
if not os.path.exists(face_dir):
    print(f"Directory {face_dir} does not exist.")
    exit(1)

# Read images and compute face descriptors
for count, filename in enumerate(os.listdir(face_dir)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(face_dir, filename)
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector(gray_img)
        for k, d in enumerate(dets):
            shape = sp(gray_img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            label = filename.split(".")[0]
            face_descriptors_dict[label] = np.array(face_descriptor)
            label_dict[label] = count

class MainApp(App):
    def build(self):
        # Initialize layout and add Code Depot branding
        layout = BoxLayout(orientation='vertical')
        header = Label(text='[color=ff3333]Code Depot[/color]', markup=True, size_hint=(1, 0.1))
        header.font_size = '24sp'
        header.bold = True
        header.halign = 'center'
        layout.add_widget(header)

        # Initialize image widget
        self.img1 = Image(size_hint=(1, 0.9))
        layout.add_widget(self.img1)

        # Initialize camera
        self.capture = cv2.VideoCapture(0)

        # Schedule frame updates
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

        return layout

    def update_frame(self, dt):
        ret, frame = self.capture.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = detector(gray_frame)
            for k, d in enumerate(dets):
                shape = sp(gray_frame, d)
                cur_face_descriptor = np.array(facerec.compute_face_descriptor(frame, shape))

                # Perform recognition
                distances = {}
                for label, known_face_descriptor in face_descriptors_dict.items():
                    distance = np.linalg.norm(cur_face_descriptor - known_face_descriptor)
                    distances[label] = distance

                recognized_name = min(distances, key=distances.get) if distances else "Unknown"

                # Draw rectangle and label
                cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
                cv2.putText(frame, recognized_name, (d.left(), d.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)

            # Update the texture of the image widget
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tobytes()
            self.img1.texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            self.img1.texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')

    def on_stop(self):
        # Release the camera on app stop
        self.capture.release()

if __name__ == "__main__":
    MainApp().run()