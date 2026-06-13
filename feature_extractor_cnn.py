import librosa
import numpy as np


def preprocess_audio(audio_path):
    """
    Convert audio file into Mel Spectrogram
    shape expected by CNN:
    (1, 128, 64, 1)
    """

    # Load audio
    y, sr = librosa.load(audio_path, sr=16000)

    # Create Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    # Convert to dB scale
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Keep first 64 time frames
    mel_db = mel_db[:, :64]

    # Pad if audio is shorter
    if mel_db.shape[1] < 64:
        pad_width = 64 - mel_db.shape[1]

        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, pad_width)),
            mode="constant"
        )

    # Add batch and channel dimensions
    mel_db = mel_db.reshape(1, 128, 64, 1)

    return mel_db