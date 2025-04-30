import numpy as np
import os
from tqdm import tqdm

def encode(label):
    mapping = {'neutral': 0, 'happy': 1, 'sad': 2, 'angry': 3, 'fear': 4, 'disgust': 5}
    return mapping[label]

def load_features(df):
    zcr_list, rms_list, mfccs_list, emotion_list = [], [], [], []

    print("Loading saved features into memory...")
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        try:
            relative_path = os.path.splitext(row.path.lstrip('./'))[0] + '.npy'
            zcr = np.load(os.path.join("zcr", relative_path))
            rms = np.load(os.path.join("rms", relative_path))
            mfccs = np.load(os.path.join("mfccs", relative_path))
            zcr_list.append(zcr)
            rms_list.append(rms)
            mfccs_list.append(mfccs)
            emotion_list.append(encode(row.emotion))
        except Exception as e:
            print(f"Failed to load features for {row.path}: {e}")

    return np.array(zcr_list), np.array(rms_list), np.array(mfccs_list), np.array(emotion_list)