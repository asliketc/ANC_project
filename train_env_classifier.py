import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("anc_sessions_clean.csv")

# Features and label
features = ['mfcc_mean', 'chroma_stft_mean', 'spectral_centroid_mean', 'rms_mean', 'zcr_mean']
X = df[features]
y = df['prediction']  # Or 'environment' if you change the column name

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "anc_env_classifier.pkl")
print("✅ Model saved as 'anc_env_classifier.pkl'")

# Optional: Evaluate
print("🎯 Accuracy:", model.score(X_test, y_test))
