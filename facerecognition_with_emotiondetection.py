import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load Haar cascade and emotion model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_model.hdf5')  # Use trained emotion model in HDF5 format
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load dataset and train LBPH face recognizer
datasets = 'dataset'
(images, labels, names, id) = ([], [], {}, 0)
(width, height) = (130, 100)

print("Training face recognizer...")

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            images.append(cv2.imread(path, 0))
            labels.append(int(id))
        id += 1

(images, labels) = [np.array(lis) for lis in [images, labels]]
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Start webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2)
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (width, height))

        # Face recognition
        prediction = model.predict(face_resized)
        if prediction[1] < 800:
            name = names[prediction[0]]
        else:
            name = "Unknown"

        # Emotion prediction
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        emotion_prediction = emotion_model.predict(roi)
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]

        # Display name + emotion
        label = f"{name} | {emotion_label}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face + Emotion Recognition", frame)
    if cv2.waitKey(10) == 27:  # ESC to exit
        break

webcam.release()
cv2.destroyAllWindows()
