
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sounddevice as sd
import numpy as np
import librosa
import librosa.display
from datetime import datetime

st.set_page_config(page_title="ANC Sentinel – Unified Dashboard", layout="wide")
st.title("🎧 ANC Sentinel – Unified Dashboard")
st.markdown("Real-time ANC simulation, session history explorer, and interactive feature testing.")

# --- Load CSV ---
@st.cache_data
def load_data():
    return pd.read_csv("anc_sessions_clean.csv")

if st.button("🔁 Reload CSV"):
    st.cache_data.clear()

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ File 'anc_sessions_clean.csv' not found.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.title("🔍 Filters")
envs = df["prediction"].unique()
selected_envs = st.sidebar.multiselect("Environment", envs, default=list(envs))
df = df[df["prediction"].isin(selected_envs)]
if "mode" in df.columns:
    modes = df["mode"].dropna().unique()
    selected_modes = st.sidebar.multiselect("ANC Mode", modes, default=list(modes))
    df = df[df["mode"].isin(selected_modes)]

# --- Table + Export ---
st.subheader("📄 Session Table")
st.dataframe(df)
st.download_button("📥 Export Filtered Data", data=df.to_csv(index=False), file_name="filtered_sessions.csv")

# --- Plot Feature Trends ---
st.subheader("📈 Feature Trends")
if not df.empty:
    feature = st.selectbox("Choose Feature", ["mfcc_mean", "chroma_stft_mean", "spectral_centroid_mean", "rms_mean", "zcr_mean"])
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=df, x="timestamp", y=feature, hue="prediction", ax=ax)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# --- Record + Simulate ---
st.subheader("🎙️ Record Audio + Simulate ANC")
dur = st.slider("Recording Duration (sec)", 2, 10, 4)

if st.button("🎧 Record Now"):
    audio = sd.rec(int(dur * 22050), samplerate=22050, channels=1, dtype='float32')
    sd.wait()
    y = np.squeeze(audio)
    sr = 22050

    rms = librosa.feature.rms(y=y)[0]
    scalar = 10 ** ((-20 - 20 * np.log10(np.mean(rms))) / 20)
    y = y * scalar

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean()
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    rms_mean = librosa.feature.rms(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()

    features = {
        'mfcc_mean': mfcc,
        'chroma_stft_mean': chroma,
        'spectral_centroid_mean': centroid,
        'rms_mean': rms_mean,
        'zcr_mean': zcr
    }

    def recommend_anc_profile(f):
        if f['rms_mean'] < 0.01:
            return {"mode": "Off", "adaptivity": False, "gain_ff": 0.0, "gain_fb": 0.0, "reason": "Very quiet environment"}
        elif f['zcr_mean'] > 0.2 and f['rms_mean'] > 0.05:
            return {"mode": "Hybrid", "adaptivity": True, "gain_ff": 3.0, "gain_fb": 2.5, "reason": "Chaotic/transient noise"}
        elif f['chroma_stft_mean'] > 0.4 and f['mfcc_mean'] > 10:
            return {"mode": "Feedback", "adaptivity": False, "gain_ff": 1.5, "gain_fb": 3.5, "reason": "Harmonic content detected"}
        elif f['mfcc_mean'] > 15 and f['chroma_stft_mean'] > 0.3:
            return {"mode": "Hybrid", "adaptivity": True, "gain_ff": 2.5, "gain_fb": 2.5, "reason": "Speech-rich environment"}
        else:
            return {"mode": "Feedforward", "adaptivity": True, "gain_ff": 3.0, "gain_fb": 1.0, "reason": "General ambient noise"}

    profile = recommend_anc_profile(features)

    st.markdown("### 🧠 Extracted Features")
    st.json(features)
    st.markdown("### 🎛️ Recommended ANC Profile")
    st.json(profile)

    # Save to session for playback buttons
    st.session_state["original_audio"] = y.tolist()
    anti = -1 * y
    combined = y + anti
    combined = combined / np.max(np.abs(combined)) if np.max(np.abs(combined)) > 0 else combined
    st.session_state["anti_noise"] = anti.tolist()
    st.session_state["combined"] = combined.tolist()

# --- Playback Controls ---
if "original_audio" in st.session_state:
    st.subheader("🎚️ Playback Controls")
    col1, col2, col3 = st.columns(3)
    y = np.array(st.session_state["original_audio"])
    anti = np.array(st.session_state["anti_noise"])
    combined = np.array(st.session_state["combined"])

    with col1:
        if st.button("▶️ Play Original"):
            sd.play(y, samplerate=22050)
            sd.wait()
    with col2:
        if st.button("🔁 Play Reversed Phase"):
            sd.play(anti, samplerate=22050)
            sd.wait()
    with col3:
        if st.button("🎧 Simulate ANC Playback"):
            sd.play(combined, samplerate=22050)
            sd.wait()

    # Quality metric
    st.markdown("### 📊 ANC Quality Score (Experimental)")
    if np.std(y) > 0:
        quality = np.corrcoef(y, combined)[0, 1]
        st.metric("Signal Correlation", f"{quality:.2f}")
    else:
        st.warning("⚠️ Audio has no variance. Try a different recording.")
