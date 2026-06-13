from fastapi import FastAPI, UploadFile, File
import joblib
import numpy as np
import os

from feature_extractor import extract_features

app = FastAPI()

model = joblib.load("deepfake_detector.pkl")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(path, "wb") as f:
        f.write(await file.read())

    features = extract_features(path)

    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0]

    return {
        "prediction": "REAL" if prediction == 0 else "FAKE",
        "real_probability": float(probability[0]),
        "fake_probability": float(probability[1])
    }