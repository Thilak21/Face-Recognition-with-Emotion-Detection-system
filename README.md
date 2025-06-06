# Face Recognition and Emotion Detection System 

This project is a real-time face recognition and emotion detection system developed using Python, OpenCV, and deep learning. It detects faces from a webcam, identifies individuals 
using the LBPH algorithm, and classifies emotions using a custom-trained CNN on the FER-2013 dataset.


## Features

- Real-time face detection using Haar Cascade Classifier.
- Face recognition using LBPH (Local Binary Pattern Histogram) algorithm.
- Emotion classification using a CNN model trained on the FER-2013 dataset.
- Live video feed with overlay of detected person's name and emotion.
- Lightweight, fast, and works without internet connection.


## Technologies Used

 Language: Python
 Libraries: OpenCV, NumPy, TensorFlow/Keras, OS
 Algorithms:
  - Face Detection: Haar Cascade
  - Face Recognition: LBPH (OpenCV)
  - Emotion Detection: Custom CNN



## Project Structure

Face_Recognition_Emotion_Detection/
├── dataset/ # Face recognition image dataset (person-wise folders)
├── haarcascade_frontalface_default.xml
├── emotion_model.hdf5 # Trained CNN model for emotion detection
├── facerecognition_with_emotiondetection.py


## Dataset

- FER-2013 dataset used for training the CNN model.
- Contains 35,000+ labeled grayscale facial images.
- Emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.


## How It Works

1. Webcam captures video input.
2. Haar Cascade detects faces in the frame.
3. LBPH model recognizes the person (if trained).
4. Cropped face is passed to the CNN model.
5. Emotion is predicted and displayed along with identity.


## Installation

1. Clone this repository:
git clone https://github.com/Thilak21/Face-Recognition-Emotion-Detection.git
cd Face-Recognition-Emotion-Detection

2. Install required packages:
