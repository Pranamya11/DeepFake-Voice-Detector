DeepFake Voice Detection System
Overview

DeepFake Voice Detection System is a machine learning and deep learning based application designed to identify whether an uploaded audio sample is genuine or AI-generated. The project combines traditional machine learning techniques with deep learning models to improve detection performance and provide users with reliable predictions.

The system allows users to upload an audio file through a Flask-based web application and receive an instant prediction indicating whether the voice is real or fake.

Features
Upload audio files for analysis
Real-time deepfake voice detection
Machine Learning based prediction model
CNN-based Deep Learning model
User-friendly web interface
Fast and accurate inference
Technologies Used
Backend
Python
Flask
Machine Learning & Deep Learning
TensorFlow / Keras
Scikit-learn
NumPy
Pandas
Audio Processing
Librosa
Model Development
Data Collection

Audio samples containing both genuine and AI-generated voices were collected and preprocessed before training.

Feature Extraction

Audio features were extracted using Librosa. Important speech characteristics were transformed into numerical representations that could be understood by machine learning algorithms.

Common extracted features include:

MFCC (Mel Frequency Cepstral Coefficients)
Spectral Features
Mel Spectrogram Features
Machine Learning Model

The traditional machine learning model was trained on extracted audio features and saved as:

deepfake_detector.pkl

This model provides fast inference and serves as a lightweight detection mechanism.

Deep Learning Model

A Convolutional Neural Network (CNN) was trained on processed audio representations and saved as:

deepfake_cnn.h5

The CNN model learns complex patterns from audio data and provides improved detection performance compared to traditional methods.

Model Performance
Model	Type	Accuracy
deepfake_detector.pkl	Machine Learning	98%
deepfake_cnn.h5	CNN Deep Learning	99%

Replace the above values with your actual results.

Project Structure

backend/
│
├── app.py
├── feature_extractor.py
├── feature_extractor_cnn.py
├── test.py
├── test_cnn.py
│
├── models/
│ ├── deepfake_detector.pkl
│ └── deepfake_cnn.h5
│
├── uploads/
│
├── requirements.txt
└── .gitignore

Installation
Clone Repository

git clone

Navigate to Project

cd backend

Install Dependencies

pip install -r requirements.txt

Run Application

python app.py

How to Use
Launch the Flask application.
Open the application in your browser.
Upload an audio file (.wav).
Click the prediction button.
The system will analyze the voice sample.
View the prediction result indicating whether the voice is genuine or deepfake.
Future Improvements
Support for additional audio formats
Transformer-based detection models
Real-time microphone analysis
Confidence score visualization
Model comparison dashboard
