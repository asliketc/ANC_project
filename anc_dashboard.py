
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(page_title="ANC Sentinel Lab Dashboard", layout="wide")

st.title("🎧 ANC Sentinel – R&D Dashboard")
st.markdown("Smart acoustic log viewer, ANC profile recommender, and real-time testing panel for engineers.")

# Load CSV
@st.cache_data
def load_data():
    return pd.read_csv("anc_sessions_clean.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ File not found. Make sure 'anc_sessions_clean.csv' exists.")
    st.stop()

st.sidebar.title("🔍 Filters")

# === Filter by Environment
envs = df["prediction"].unique()
selected_envs = st.sidebar.multiselect("Environment", envs, default=list(envs))

# === Filter by ANC mode (if available)
modes = df["mode"].dropna().unique()
selected_modes = st.sidebar.multiselect("ANC Mode", modes, default=list(modes))

# Filter Data
filtered_df = df[df["prediction"].isin(selected_envs)]
if "mode" in df.columns:
    filtered_df = filtered_df[filtered_df["mode"].isin(selected_modes)]

# === Session Table
st.subheader("📄 Session Table")
st.dataframe(filtered_df)

# === Profile Effectiveness Rating
st.subheader("⭐ Rate ANC Effectiveness")
session_to_rate = st.selectbox("Pick a session to rate:", filtered_df.index)
rating = st.radio("Did the ANC work well for this session?", ["👍 Yes", "👎 No", "🤷‍♀️ Not Sure"])
st.write(f"📝 You rated session {session_to_rate} as: {rating}")

# === Timeline Feature Plot
st.subheader("📈 Feature Trends Over Time")
feature = st.selectbox("Feature to plot", [
    "mfcc_mean", "chroma_stft_mean", "spectral_centroid_mean", "rms_mean", "zcr_mean"
])
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=filtered_df, x="timestamp", y=feature, hue="prediction", ax=ax)
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# === Explore Single Session
st.subheader("🔎 Explore Single Session")
idx = st.slider("Session index", 0, len(filtered_df)-1)
session = filtered_df.iloc[idx]
st.json(session.to_dict())

# === LIVE Recommender (Simulated)
st.subheader("🎛️ Try Your Own Profile")

mfcc_val = st.slider("MFCC Mean", 0.0, 50.0, float(session["mfcc_mean"]))
chroma_val = st.slider("Chroma Mean", 0.0, 1.0, float(session["chroma_stft_mean"]))
centroid_val = st.slider("Spectral Centroid", 0.0, 8000.0, float(session["spectral_centroid_mean"]))
rms_val = st.slider("RMS", 0.0, 0.2, float(session["rms_mean"]))
zcr_val = st.slider("Zero Crossing Rate", 0.0, 1.0, float(session["zcr_mean"]))

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

features = {
    "mfcc_mean": mfcc_val,
    "chroma_stft_mean": chroma_val,
    "spectral_centroid_mean": centroid_val,
    "rms_mean": rms_val,
    "zcr_mean": zcr_val
}
reco = recommend_anc_profile(features)

st.markdown("### 🔮 Recommended ANC Profile")
st.json(reco)
