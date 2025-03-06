import os
import numpy as np
import librosa
from tqdm import tqdm
from utils import load_audio_files
from config import (
    TRAIN_DATA_DIR, SR, N_MFCC, N_MELS, HOP_LENGTH, BATCH_SIZE, 
    AUGMENTATION, TIME_STRETCH_RATES, PITCH_SHIFT_STEPS, NOISE_FACTOR, NORMALIZE,
    TIME_STEPS
)

def augment_audio(y, sr):
    """对音频进行数据增强
    
    应用以下增强技术：
    1. 时间拉伸 - 改变音频速度
    2. 音高偏移 - 改变音频音高
    3. 添加噪声 - 增加随机噪声
    """
    try:
        if AUGMENTATION:
            if TIME_STRETCH_RATES:
                # 新版本librosa的time_stretch需要使用rate关键字参数
                rate = np.random.choice(TIME_STRETCH_RATES)
                y = librosa.effects.time_stretch(y=y, rate=rate)
            if PITCH_SHIFT_STEPS:
                n_steps = np.random.choice(PITCH_SHIFT_STEPS)
                y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
            if NOISE_FACTOR:
                noise = NOISE_FACTOR * np.random.randn(len(y))
                y = y + noise
        return y
    except Exception as e:
        print(f"音频增强过程中出错: {str(e)}")
        return y

def extract_features(file_path, sr, n_mfcc, n_mels, hop_length):
    """从音频文件中提取特征
    
    提取以下特征：
    1. MFCC特征 - 音色和音调特征
    2. 梅尔频谱图 - 频率特征
    """
    try:
        # 加载音频文件
        y, _ = librosa.load(file_path, sr=sr)

        # 数据增强
        y = augment_audio(y, sr)
        
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # 提取梅尔频谱图特征
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, hop_length=hop_length
        )
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # 特征标准化
        if NORMALIZE:
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
            mel_spectrogram = (mel_spectrogram - np.mean(mel_spectrogram)) / (np.std(mel_spectrogram) + 1e-8)

        # 组合特征
        combined_features = np.hstack((
            np.mean(mfcc.T, axis=0), 
            np.mean(mel_spectrogram.T, axis=0)
        ))

        # 调整特征维度
        if combined_features.shape[0] != TIME_STEPS:
            if combined_features.shape[0] < TIME_STEPS:
                pad_width = TIME_STEPS - combined_features.shape[0]
                combined_features = np.pad(combined_features, (0, pad_width), mode='constant')
            else:
                combined_features = combined_features[:TIME_STEPS]

        return combined_features
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None

def extract_features_in_batches(files, batch_size, sr, n_mfcc, n_mels, hop_length):
    """批量提取特征"""
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        features = []
        
        # 使用tqdm显示进度
        for file in tqdm(batch_files, desc=f"处理批次 {i//batch_size + 1}"):
            feature = extract_features(file, sr, n_mfcc, n_mels, hop_length)
            if feature is not None:
                features.append(feature)
        
        if features:  # 只在有成功提取的特征时才yield
            yield np.array(features)

def process_large_dataset():
    """处理大规模数据集"""
    try:
        print("开始加载音频文件...")
        files = load_audio_files(TRAIN_DATA_DIR)
        if not files:
            print("❌ 错误：没有找到音频文件")
            return
        
        print(f"找到 {len(files)} 个音频文件")
        labels = [1 if 'snore' in os.path.basename(file).lower() else 0 for file in files]
        
        print("\n开始提取特征...")
        total_features = []
        for batch_features in extract_features_in_batches(files, BATCH_SIZE, SR, N_MFCC, N_MELS, HOP_LENGTH):
            total_features.append(batch_features)
            print(f"当前已处理 {len(total_features) * BATCH_SIZE} 个文件")
        
        # 合并所有特征
        if total_features:
            all_features = np.concatenate(total_features, axis=0)
            print(f"\n✅ 特征提取完成！")
            print(f"特征形状: {all_features.shape}")
            print(f"标签数量: {len(labels)}")
        else:
            print("❌ 错误：没有成功提取任何特征")
            
    except Exception as e:
        print(f"❌ 处理过程中出现错误：{str(e)}")

if __name__ == "__main__":
    try:
        process_large_dataset()
    except KeyboardInterrupt:
        print("\n\n⚠️ 程序被用户中断")
    except Exception as e:
        print(f"❌ 发生未预期的错误：{str(e)}")
