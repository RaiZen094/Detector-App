# Install Streamlit and other dependencies
# pip install streamlit librosa scikit-learn scipy joblib

import streamlit as st
import numpy as np
import librosa
from scipy.stats import skew, kurtosis, gmean
from sklearn.linear_model import LogisticRegression
from joblib import load

# Load the pre-trained model using joblib
model_path = "logistic_model.pkl"  # Path to your .joblib file
model = load(model_path)  # Load the model

# Function to extract features from audio
def extract_features(audio_data, fs):
    # Pre-emphasis
    pre_emphasis = 0.97
    audio_filtered = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])

    # Resample to 16 kHz if necessary
    new_fs = 16000
    audio_resampled = librosa.resample(audio_filtered, orig_sr=fs, target_sr=new_fs)

    # Frame the audio
    frame_size = int(0.32 * new_fs)
    hop_length = int(0.5 * frame_size)
    framed_audio = librosa.util.frame(audio_resampled, frame_length=frame_size, hop_length=hop_length).T

    # Calculate mean features across frames
    frame_features = []

    for frame in framed_audio:
        windowed_frame = frame * np.hamming(len(frame))

        spectral_std = np.std(windowed_frame)
        crest_factor = np.max(np.abs(windowed_frame)) / np.sqrt(np.mean(np.square(windowed_frame)))
        spectral_spread = np.sum((windowed_frame - np.mean(windowed_frame)) ** 2) / len(windowed_frame)
        spectral_skewness = skew(windowed_frame)
        spectral_flatness = gmean(np.abs(windowed_frame)) / np.mean(np.abs(windowed_frame))
        cumulative_sum = np.cumsum(np.square(np.abs(windowed_frame)))
        spectral_rolloff = np.where(cumulative_sum >= 0.85 * cumulative_sum[-1])[0][0] / len(windowed_frame)
        zero_crossing_rate = np.sum(np.abs(np.diff(np.sign(windowed_frame)))) / (2 * len(windowed_frame))
        band_power = np.sum(np.square(windowed_frame)) / len(windowed_frame)
        X = np.arange(len(windowed_frame))
        spectral_slope = np.polyfit(X, windowed_frame, 1)[0]
        spectral_kurtosis = kurtosis(windowed_frame)
        max_frequency = np.argmax(np.abs(np.fft.fft(windowed_frame))) * (new_fs / len(windowed_frame))

        frame_features.append([
            spectral_std, crest_factor, spectral_spread, spectral_skewness,
            spectral_flatness, spectral_rolloff, zero_crossing_rate, band_power,
            spectral_slope, spectral_kurtosis, max_frequency
        ])

    # Aggregate feature values across frames
    audio_features = np.mean(frame_features, axis=0).reshape(1, -1)
    return audio_features

# Streamlit app
st.title("Cough Type Prediction")

st.write("Upload a voice note to determine the type of cough (Pertussis or Other).")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Load the audio file
    try:
        audio_data, fs = librosa.load(uploaded_file, sr=None)
        st.audio(uploaded_file, format="audio/wav")

        # Extract features
        st.write("Processing the audio file...")
        features = extract_features(audio_data, fs)

        # Predict using the model
        prediction = model.predict(features)

        # Display the result
        if prediction[0] == 1:
            st.success("Prediction: Pertussis cough detected.")
        else:
            st.info("Prediction: Other cough detected.")

    except Exception as e:
        st.error(f"Error processing audio file: {e}")
