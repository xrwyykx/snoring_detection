# =====================================================
import numpy as np
import os
import librosa
from tensorflow.keras.models import load_model
from config import SR, N_MFCC, N_MELS, HOP_LENGTH, TIME_STEPS, MODEL_DIR

def preprocess_audio(file_path):
    def extract_mfcc(y, sr, n_mfcc):
        return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    def extract_mel_spectrogram(y, sr):
        return librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)

    def normalize_features(features):
        return (features - np.mean(features)) / np.std(features)
    
    y, _ = librosa.load(file_path, sr=SR)
    mfcc_features = extract_mfcc(y, SR, N_MFCC)
    mel_features = extract_mel_spectrogram(y, SR)
    
    combined_features = np.concatenate((mfcc_features, mel_features), axis=0)
    normalized_features = normalize_features(combined_features)

    if normalized_features.shape[1] != TIME_STEPS:
        if normalized_features.shape[1] < TIME_STEPS:
            pad_width = TIME_STEPS - normalized_features.shape[1]
            normalized_features = np.pad(normalized_features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            normalized_features = normalized_features[:, :TIME_STEPS]

    return np.expand_dims(normalized_features, axis=0)

def preprocess_audio_sliding_window(file_path, model):
    def extract_mfcc(y, sr, n_mfcc):
        return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    def extract_mel_spectrogram(y, sr):
        return librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)

    def normalize_features(features):
        return (features - np.mean(features)) / np.std(features)

    window_length = 1.0  # 1 second window
    hop_length = 0.5  # 50% overlap

    y, sr = librosa.load(file_path, sr=SR)
    samples_per_window = int(window_length * sr)
    samples_per_hop = int(hop_length * sr)

    num_windows = int(np.ceil(len(y) / samples_per_hop))

    predictions = []

    for i in range(num_windows):
        start = i * samples_per_hop
        end = min(start + samples_per_window, len(y))
        window = y[start:end]
        
        if len(window) < samples_per_window:
            window = np.pad(window, (0, samples_per_window - len(window)), mode='constant')

        mfcc_features = extract_mfcc(window, SR, N_MFCC)
        mel_features = extract_mel_spectrogram(window, SR)
        
        combined_features = np.concatenate((mfcc_features, mel_features), axis=0)
        normalized_features = normalize_features(combined_features)
        
        if normalized_features.shape[1] != TIME_STEPS:
            if normalized_features.shape[1] < TIME_STEPS:
                pad_width = TIME_STEPS - normalized_features.shape[1]
                normalized_features = np.pad(normalized_features, ((0, 0), (0, pad_width)), mode='constant')
            else:
                normalized_features = normalized_features[:, :TIME_STEPS]
        
        # Add an extra dimension for the model input
        input_data = np.expand_dims(normalized_features, axis=0)
        prediction = model.predict(input_data)
        predictions.append(prediction)

    aggregated_prediction = np.mean(predictions, axis=0)
    print("Aggregated Prediction is ",aggregated_prediction)
    predicted_class = (aggregated_prediction > 0.5).astype(int)

    return predicted_class[0][0] 

def predict_snore(file_path, model_path):

    model = load_model(model_path)
    result = preprocess_audio_sliding_window(file_path, model)

    return result

if __name__ == "__main__":

    inference_audio_dir = 'inference_audios'
    model_path = 'models/final_snore_detection_model.h5'
    

    for audio_file_name in os.listdir(inference_audio_dir):

        if audio_file_name.endswith('.mp3') or audio_file_name.endswith('.wav'):
            audio_file_path = os.path.join(inference_audio_dir, audio_file_name)
            print(f"Processing file: {audio_file_path}")

            result = predict_snore(audio_file_path, model_path)

            if result == 1:
                print(f"The audio '{audio_file_name}' is not a snore.\n\n")
            else:
                print(f"The audio '{audio_file_name}' is a snore (=','=)Zzzzz\n\n")
#=================================================================================
