# Real-Time-Face-Recognition-System
This is a Real Time face Recognition System which uses Dlib and SSD Face Recognition Technologies to perform accurate face recognition with just one picture of the Subject.



------------------------------------------------------------------------
                     Dlib SSD Face Recognition
------------------------------------------------------------------------


Table of Contents
=================
1. Project Overview
2. Directory Structure
3. Installation and Dependencies
4. How to Use
5. Components Description
   5.1. Main.py
   5.2. Face_Directory
   5.3. Models
6. License

------------------------------------------------------------------------

1. Project Overview
-------------------
This project is designed to implement face recognition using Dlib's state-of-the-art face recognition model along with SSD (Single Shot MultiBox Detector) for face detection. The project is built with Python and Kivy for GUI. The project showcases the use of Dlib's face recognition capabilities in real-time video capture.

------------------------------------------------------------------------

2. Directory Structure
----------------------
    Dlib_SSD_Face_Recoginition_CodeDepot
    ├── Main.py
    ├── Face_Directory
    ├── dlib_face_recognition_resnet_model_v1.dat
    └── shape_predictor_68_face_landmarks.dat

------------------------------------------------------------------------

3. Installation and Dependencies
--------------------------------
1. Python 3.x
2. Kivy
3. OpenCV
4. Dlib

Install the required packages using pip:
    pip install kivy opencv-python dlib

------------------------------------------------------------------------

4. How to Use
-------------
clone the project from the following URL: https://github.com/Ankush251992/Real-Time-Face-Recognition-System.git

1. Place images of faces you want to recognize in the 'Face_Directory'.
2. Run Main.py:
    python Main.py

------------------------------------------------------------------------

5. Components Description
-------------------------
### 5.1 Main.py
This is the main file that you run for the face recognition application. It initializes the Dlib models, captures video from the webcam, and performs face recognition.

### 5.2 Face_Directory
This directory should contain the face images that you want to be recognized. Make sure the images are clear and preferably have only the face you want to recognize.

### 5.3 Models
#### dlib_face_recognition_resnet_model_v1.dat
This is Dlib's pre-trained face recognition model. It's based on a ResNet network structure and is trained to generate 128D face descriptors.

#### shape_predictor_68_face_landmarks.dat
This is Dlib's pre-trained model to detect the 68 facial landmarks that are used by the face recognition model to align the face images.

------------------------------------------------------------------------

6. License
----------
This project is licensed under the terms of the MIT License.

-----------------------------------------------------------------------
