import joblib
import numpy as np

from feature_extractor import extract_features

model = joblib.load("deepfake_detector.pkl")

files = [
    "myvoice.wav",
    "ai_voice.wav",
    "ai_voice2.wav",
    "ai_voice3.wav",
    "ai_voice4.wav"
]

for audio_path in files:

    features = extract_features(audio_path)
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    print("\n" + "="*50)
    print("File:", audio_path)

    if prediction == 0:
        print("Prediction: REAL")
    else:
        print("Prediction: FAKE")

    print(f"Real Probability: {prob[0]:.4f}")
    print(f"Fake Probability: {prob[1]:.4f}")