# Face Recognition with Emotion Detection

# Project Overview
   The Face Recognition with Emotion Detection System is an advanced real-time computer vision project built using Python, OpenCV, and TensorFlow. It captures live webcam input, recognizes the person using LBPH (Local Binary Patterns Histogram), and detects their facial emotion using a trained CNN model (FER-2013). This system is designed to assist in smart surveillance, human-computer interaction, and emotional analysis applications. 


# Key Features

-> Real-time Face Detection: Detects faces from live video feed using Haar cascades.

-> Face Recognition: Identifies known individuals using LBPH algorithm.

-> Emotion Detection: Classifies emotional state (Happy, Sad, Angry, etc.) using a CNN model.

-> Live Overlay: Displays both name and predicted emotion on-screen.

-> Custom Dataset Training: Easily train with your own image folders.

-> Modular Design: Components can be replaced, improved, or extended (e.g., for liveness detection).       


# Technologies Used

-> Python: Core language for implementation

-> OpenCV: For image processing, face detection, and recognition

-> TensorFlow/Keras: For building and loading CNN model for emotion classification

-> Haar Cascade Classifier: For face detection

-> LBPH Face Recognizer: For identity recognition

-> NumPy: For array processing and pixel data manipulation


# Project Highlights

-> Dual-Purpose AI: Combines both identity and emotion recognition in real-time.

-> Smooth Real-Time Performance: Works well with webcam feed at normal frame rate.

-> Trainable & Scalable: Can train with your own images and retrain emotion model if needed.


# How to Use

1. Clone the repository

       git clone : https://github.com/Thilak21/Face-Recognition-Emotion-Detection.git

       cd Face-Recognition-Emotion-Detection

2. Install required libraries

       pip install opencv-python
   
       pip install opencv-contrib-python
     
       pip install tensorflow
    
       pip install numpy

3. Create your face dataset (for recognition)

       python create_data.py

      Tip:Create folders inside dataset/ named after the person (e.g., John/, Alice/) and place 20–30 grayscale images per person.

4. Run the system

       python Face_Recognition_with_Emotion_Detection.py
   

# Output Example

Label format:

   John | Happy 
   Unknown | Sad
        

# Learning Outcomes

-> Implemented face recognition using OpenCV’s LBPH algorithm.

-> Trained and used a CNN model for emotion classification.

-> Worked with real-time video streams and handled image preprocessing.

-> Gained experience with model loading, inference, and thresholding for predictions.


# Conclusion

   This Face Recognition + Emotion Detection System demonstrates real-time AI capabilities in computer vision. It uses well-established algorithms like Haar and LBPH, along with deep learning for emotion prediction. The project is scalable and can be expanded with liveness detection or deployed in cloud/mobile systems.
