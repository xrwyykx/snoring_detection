import os
import librosa
import numpy as np

def load_audio_files(directory, file_extension='.wav'):
    """
    Load audio files from a directory.

    Parameters:
    - directory: str, path to the directory containing audio files.
    - file_extension: str, file extension to filter by (default is '.wav').

    Returns:
    - List of file paths to the audio files.
    """
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(file_extension):
                audio_files.append(os.path.join(root, file))
    return audio_files

def extract_mfcc(file_path, sr, n_mfcc, y=None):
    """提取MFCC特征"""
    try:
        if y is None:
            y, _ = librosa.load(file_path, sr=sr)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc
    except Exception as e:
        print(f"❌ MFCC特征提取失败: {str(e)}")
        return None


# Example update to extract_mel_spectrogram
def extract_mel_spectrogram(file_path, sr, y=None):
    """提取梅尔频谱图特征"""
    try:
        if y is None:
            y, _ = librosa.load(file_path, sr=sr)
        
        mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spect_db = librosa.power_to_db(mel_spect, ref=np.max)
        return mel_spect_db
    except Exception as e:
        print(f"❌ 梅尔频谱图特征提取失败: {str(e)}")
        return None


def normalize_features(features):
    """标准化特征"""
    try:
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        normalized_features = (features - mean) / (std + 1e-8)
        return normalized_features
    except Exception as e:
        print(f"❌ 特征标准化失败: {str(e)}")
        return None
