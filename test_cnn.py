from tensorflow.keras.models import load_model
from feature_extractor_cnn import preprocess_audio

model = load_model("deepfake_cnn.h5")

audio_file = input("Enter audio file: ")

features = preprocess_audio(audio_file)

prob = float(
    model.predict(features, verbose=0)[0][0]
)

prediction = "FAKE" if prob > 0.5 else "REAL"

print("\nRESULT")
print("=" * 40)
print("Prediction:", prediction)
print("Fake Probability:", round(prob, 4))
print("Real Probability:", round(1 - prob, 4))