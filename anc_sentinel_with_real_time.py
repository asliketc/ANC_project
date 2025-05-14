import sounddevice as sd 
import numpy as np 
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd 
from datetime import datetime
import os
import joblib

CHROMA_THRESHOLD=0.25
MFCC_THRESHOLD=25.0

model=joblib.load("anc_env_classifier.pkl")

# STEP 1: Record Audio
def record_audio(dur=5, fs=22050):
    print("🎙️ Recording...")
    audio = sd.rec(int(dur * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio), fs

# STEP 2: Normalize volume
def normalize_rms(y, target_db=-20):
    rms = librosa.feature.rms(y=y)[0]
    scalar = 10 ** ((target_db - 20 * np.log10(np.mean(rms))) / 20)
    return y * scalar

# STEP 3: Extract features for ML + voice detection
def extract_features_full(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mfcc_mean = mfcc.mean()
    chroma_mean = chroma.mean()
    
    return {
        'mfcc_mean': mfcc_mean,
        'chroma_stft_mean': chroma_mean,
        'spectral_centroid_mean': librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
        'rms_mean': librosa.feature.rms(y=y).mean(),
        'zcr_mean': librosa.feature.zero_crossing_rate(y).mean(),
        'is_voice_detected': mfcc_mean > MFCC_THRESHOLD and chroma_mean > CHROMA_THRESHOLD
    }

# STEP 4: Plot waveform and spectrogram
def plot_waveform_and_spectrogram(y, sr):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set(title='Waveform')

    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
    ax[1].set(title='Spectrogram')
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    plt.tight_layout()
    plt.show()

# STEP 5: Simulate anti-noise (unless voice is detected)
def safe_normalize(signal):
    max_val = np.max(np.abs(signal))
    return signal / max_val if max_val > 0 else signal

def generate_voice_aware_anti_noise(y, sr, is_voice_detected):
    if is_voice_detected:
        print("🛑 Voice detected – skipping anti-noise to preserve speech clarity.")
    else:
        print("🔁 Playing simulated anti-noise...")
        anti_noise = -1 * y
        combined = safe_normalize(y + anti_noise)
        sd.play(combined, samplerate=sr)
        sd.wait()

# STEP 6: Save session
def save_session(features, prediction, filename="anc_sessions.csv"):
    data = features.copy()
    data['prediction'] = prediction
    data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([data])
    file_exists = os.path.isfile(filename)
    df.to_csv(filename, mode='a', index=False, header=not file_exists)

# MAIN
def main():
    y, sr = record_audio()
    y = normalize_rms(y)
    plot_waveform_and_spectrogram(y, sr)

    features = extract_features_full(y, sr)
    is_voice = features['is_voice_detected']

    X_input = [[
        features['mfcc_mean'],
        features['chroma_stft_mean'],
        features['spectral_centroid_mean'],
        features['rms_mean'],
        features['zcr_mean']
    ]]
    prediction = model.predict(X_input)[0]

    generate_voice_aware_anti_noise(y, sr, is_voice)
    save_session(features, prediction)

    print("🧠 Predicted Acoustic Environment (ML):", prediction)
    print("🎼 Features:", features)

if __name__ == "__main__":
    main()
    