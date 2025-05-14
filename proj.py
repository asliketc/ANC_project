import sounddevice as sd 
import numpy as np 
import librosa
import librosa.display
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Record Audio
def record_audio(dur=5, fs=22050):
    print("Recording.....")
    audio=sd.rec(int(dur * fs), samplerate=fs, channels=1,dtype='float32')
    sd.wait()
    return np.squeeze(audio), fs

# RMS Normalization
def normalize_rms(y, target_db=-20):
    rms=librosa.feature.rms(y=y)[0]
    scalar=10 ** ((target_db-20 * np.log10(np.mean(rms)))/20)
    return y*scalar

#Feature Extraction
def extract_feature(y,sr):
    return{
        'mfcc_mean': librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13).mean(),
        'chroma_stft_mean':librosa.feature.chroma_stft(y=y, sr=sr).mean(),
        'spectral_centroid_mean':librosa.feature.spectral_centroid(y=y,sr=sr).mean(),
        'rms_mean':librosa.feature.rms(y=y).mean(),
        'zcr_mean':librosa.feature.zero_crossing_rate(y).mean()
    }

#Dummy Model simulating ANC envr classification
def train_dummy_model():
    X_dummy=np.random.rand(100,5)
    y_dummy=np.random.choice(['Quiet Room', 'Vocal Zone','Noicy Place'], 100)
    model=make_pipeline(StandardScaler(), RandomForestClassifier())
    model.fit(X_dummy, y_dummy)
    return model

#Visualize waveform and spectogram
def plot_waveform_and_spectrogram(y,sr):
    fig, ax=plt.subplots(2,1,figsize=(10,6))
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set(title='Waveform')
    
    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
    ax[1].set(title='Spectrogram')
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    plt.tight_layout()
    plt.show()

y, sr = record_audio()
y = normalize_rms(y)  # Normalize loudness
plot_waveform_and_spectrogram(y, sr)

features = extract_feature(y, sr)
model = train_dummy_model()
prediction = model.predict([list(features.values())])[0]

print("🧠 Predicted Acoustic Environment:", prediction)
print("🎼 Features:", features)

