import pandas as pd

# Try to load the original CSV, skipping bad lines
try:
    df = pd.read_csv("anc_sessions.csv", on_bad_lines='skip')  # for pandas >=1.3
    print("✅ Loaded successfully after skipping bad lines.")
except Exception as e:
    print("❌ Failed to load CSV:", e)
    exit()

# Optional: Print first few rows to visually confirm
print("\n📊 Sample rows:")
print(df.head())

# Check if expected columns exist
expected_columns = [
    'mfcc_mean', 'chroma_stft_mean', 'spectral_centroid_mean',
    'rms_mean', 'zcr_mean', 'prediction', 'timestamp'
]

missing_cols = [col for col in expected_columns if col not in df.columns]
if missing_cols:
    print(f"⚠️ Warning: Missing columns in cleaned file: {missing_cols}")
else:
    print("✅ All expected columns present.")

# Save to a new clean file
clean_path = "anc_sessions_clean.csv"
df.to_csv(clean_path, index=False)
print(f"\n💾 Cleaned CSV saved as: {clean_path}")
