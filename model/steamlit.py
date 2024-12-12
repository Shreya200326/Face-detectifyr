import streamlit as st
import cv2
import numpy as np
import pywt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
import os
# Function to preprocess images
def get_cropped_image_if_2_eyes(image_path, face_cascade, eye_cascade):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= 2:
            return roi_color
    return None
# Function for wavelet transform
def w2d(img, mode='haar', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255

    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H
# Load trained model and cascades
face_cascade_path = r"C:\Users\chris\PycharmProjects\PythonProject1\harrcascade\haarcascade_frontalface_default.xml"
eye_cascade_path = r"C:\Users\chris\PycharmProjects\PythonProject1\harrcascade\haarcascade_eye.xml"
model_path = r"path_to_trained_model.pkl"

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
with open(model_path, 'rb') as file:
    model = pickle.load(file)
# Streamlit UI
st.title("Face Recognition with Streamlit")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load and display the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_column_width=True)

    # Process the image
    cropped_img = get_cropped_image_if_2_eyes(uploaded_file.name, face_cascade, eye_cascade)
    if cropped_img is not None:
        st.image(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), caption="Cropped Image", use_column_width=True)

        # Prepare the image for the model
        scalled_raw_img = cv2.resize(cropped_img, (32, 32))
        img_har = w2d(cropped_img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
        X = np.array(combined_img).reshape(1, 4096).astype(float)
        # Predict
        prediction = model.predict(X)[0]
        st.write(f"Predicted Class: {prediction}")
    else:
        st.write("Face not detected or not enough features (eyes) found.")

st.write("Upload a clear image for best results.")
