import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

# Parameters
FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 13

# Load your DataFrame (make sure it has 'path' column)
df = pd.read_csv("df.csv")  # Replace with your actual DataFrame CSV path

def extract_and_save(audio_path):
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)

        # Feature Extraction
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)  # (1, T)
        rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)               # (1, T)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)                 # (13, T)

        # Build matching save paths
        relative_path = audio_path.replace('./', '').replace('.wav', '.npy')

        # Paths for each feature
        zcr_save_path = os.path.join("zcr", relative_path)
        rms_save_path = os.path.join("rms", relative_path)
        mfcc_save_path = os.path.join("mfccs", relative_path)

        # Create necessary directories
        os.makedirs(os.path.dirname(zcr_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(rms_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(mfcc_save_path), exist_ok=True)

        # Save each feature
        np.save(zcr_save_path, zcr)
        np.save(rms_save_path, rms)
        np.save(mfcc_save_path, mfcc)

    except Exception as e:
        print(f"Failed for {audio_path}: {e}")

def main():
    print(f"Total files: {len(df)}")

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Extracting Features"):
        extract_and_save(row.path)

    print("All features saved in their respective directories.")

if __name__ == "__main__":
    main()