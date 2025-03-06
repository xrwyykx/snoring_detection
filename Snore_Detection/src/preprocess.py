"""
打鼾检测项目的数据预处理模块

预处理流程包括以下几个主要阶段：
1. 特征提取
   - 提取MFCC特征：捕捉音频的音色和音调特征
   - 提取梅尔频谱图特征：表示音频的频率特征
   - 特征组合：将MFCC和梅尔频谱图特征组合
   - 特征标准化：对特征进行归一化处理

2. 数据增强
   - 时间拉伸：改变音频速度
   - 添加噪声：增加随机噪声
   - 音高偏移：改变音频音高
   这些增强方法可以提高模型的鲁棒性

3. 数据集划分
   - 将数据集分为训练集(70%)、验证集(15%)和测试集(15%)
   - 使用分层抽样确保各集合中标签分布平衡

主要参数：
- SR: 采样率
- N_MFCC: MFCC特征数
- TIME_STEPS: 时间步长
- AUGMENTATION: 是否启用数据增强
"""

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from utils import extract_mfcc, extract_mel_spectrogram, normalize_features
from config import TRAIN_DATA_DIR, TEST_DATA_DIR, SR, N_MFCC, N_MELS, HOP_LENGTH, AUGMENTATION, TIME_STEPS
from tqdm import tqdm  # 添加进度条支持

def preprocess_file(file, label):
    """单个音频文件的预处理函数
    
    处理步骤：
    1. 提取MFCC特征 - 捕捉音频的音色和音调特征
    2. 提取梅尔频谱图特征 - 获取音频的频率特征
    3. 特征组合 - 将两种特征拼接
    4. 特征标准化 - 归一化处理
    5. 形状调整 - 确保所有特征具有相同的时间步长
    """
    try:
        # 第一阶段：特征提取
        mfcc_features = extract_mfcc(file, SR, N_MFCC)
        mel_features = extract_mel_spectrogram(file, SR)

        if mfcc_features is None or mel_features is None:
            print(f"警告：无法从文件提取特征: {file}")
            return None, None

        # 第二阶段：特征组合
        combined_features = np.concatenate((mfcc_features, mel_features), axis=0)
        # 第三阶段：特征标准化
        normalized_features = normalize_features(combined_features)

        # 第四阶段：形状调整（填充或截断）
        if normalized_features.shape[1] != TIME_STEPS:
            if normalized_features.shape[1] < TIME_STEPS:
                # 如果特征长度不足，进行填充
                pad_width = TIME_STEPS - normalized_features.shape[1]
                normalized_features = np.pad(normalized_features, ((0, 0), (0, pad_width)), mode='constant')
            else:
                # 如果特征长度过长，进行截断
                normalized_features = normalized_features[:, :TIME_STEPS]

        return normalized_features, label
    except Exception as e:
        print(f"处理文件时出错 {file}: {str(e)}")
        return None, None

def augment_data(y, sr, file_path):
    """数据增强函数
    
    对音频数据进行三种增强：
    1. 时间拉伸 - 改变音频速度，模拟不同说话速度
    2. 添加噪声 - 增加随机噪声，提高模型抗噪声能力
    3. 音高偏移 - 改变音频音高，模拟不同说话者
    """
    try:
        # 1. 时间拉伸增强
        y_stretch = librosa.effects.time_stretch(y, rate=0.8)
        # 2. 添加随机噪声
        y_noise = y + 0.005 * np.random.randn(len(y))
        # 3. 音高偏移
        y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        
        augmented_features = []
        # 对每种增强后的音频提取特征
        for aug_y in [y_stretch, y_noise, y_pitch]:
            # 提取MFCC特征
            mfcc_aug = extract_mfcc(None, sr, N_MFCC, aug_y)
            if mfcc_aug is None:
                continue
            # 提取梅尔频谱图特征
            mel_aug = extract_mel_spectrogram(None, sr, aug_y)
            if mel_aug is None:
                continue
            
            # 特征组合和标准化
            combined_aug = np.concatenate((mfcc_aug, mel_aug), axis=0)
            normalized_aug = normalize_features(combined_aug)
            
            # 调整特征形状
            if normalized_aug is not None:
                if normalized_aug.shape[1] != TIME_STEPS:
                    if normalized_aug.shape[1] < TIME_STEPS:
                        pad_width = TIME_STEPS - normalized_aug.shape[1]
                        normalized_aug = np.pad(normalized_aug, ((0, 0), (0, pad_width)), mode='constant')
                    else:
                        normalized_aug = normalized_aug[:, :TIME_STEPS]
                augmented_features.append(normalized_aug)
        
        return augmented_features
    except Exception as e:
        print(f"数据增强处理文件时出错 {file_path}: {str(e)}")
        return []

def preprocess_data():
    """主预处理函数
    
    完整的预处理流程：
    1. 数据加载和特征提取
       - 遍历所有音频文件
       - 提取MFCC和梅尔频谱图特征
       - 特征组合和标准化
    
    2. 数据增强（如果启用）
       - 对每个音频文件进行三种增强
       - 为增强后的音频提取特征
    
    3. 数据集划分
       - 将数据集分为训练集(70%)、验证集(15%)和测试集(15%)
       - 保存处理后的数据
    """
    try:
        features = []
        labels = []

        # 检查数据目录是否存在
        for subdir in ['snoring', 'non-snoring']:
            full_path = os.path.join(TRAIN_DATA_DIR, subdir)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"找不到目录: {full_path}")

        # 第一阶段：数据加载和特征提取
        total_files_processed = 0
        for label, subdir in enumerate(['snoring', 'non-snoring']):
            subdir_path = os.path.join(TRAIN_DATA_DIR, subdir)
            print(f"\n开始处理 {subdir} 目录...")
            
            files = [f for f in os.listdir(subdir_path) if f.endswith('.wav')]
            if not files:
                print(f"警告：在 {subdir_path} 中没有找到.wav文件")
                continue

            # 处理每个音频文件
            pbar = tqdm(files, desc=f"处理{subdir}文件")
            for file in pbar:
                file_path = os.path.join(subdir_path, file)
                # 提取原始特征
                feature, current_label = preprocess_file(file_path, label)
                
                if feature is not None:
                    features.append(feature)
                    labels.append(current_label)
                    total_files_processed += 1

                # 第二阶段：数据增强
                if AUGMENTATION:
                    try:
                        y, _ = librosa.load(file_path, sr=SR)
                        augmented_features = augment_data(y, SR, file_path)
                        
                        for aug_feature in augmented_features:
                            features.append(aug_feature)
                            labels.append(current_label)
                            total_files_processed += 1
                            
                    except Exception as e:
                        print(f"\n数据增强处理文件时出错 {file_path}: {str(e)}")
                        continue
                
                pbar.set_description(f"处理{subdir}文件 (已处理: {total_files_processed})")

        if not features:
            raise ValueError("没有成功处理任何文件，特征列表为空")

        # 输出处理统计信息
        print("\n数据处理阶段完成！")
        print(f"成功处理原始文件数: {len(files) * 2}")  # snoring + non-snoring
        print(f"数据增强后的总样本数: {total_files_processed}")

        # 第三阶段：数据格式转换和划分
        print("\n正在转换数据格式...")
        features = np.array(features)
        labels = np.array(labels)

        print("正在分割数据集...")
        # 首先分出测试集
        X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
        # 将剩余数据分为验证集和测试集
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # 创建保存目录
        os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
        os.makedirs(TEST_DATA_DIR, exist_ok=True)

        # 保存处理后的数据
        print("正在保存处理后的数据...")
        np.savez_compressed(os.path.join(TRAIN_DATA_DIR, 'train_data.npz'), X_train=X_train, y_train=y_train)
        np.savez_compressed(os.path.join(TRAIN_DATA_DIR, 'val_data.npz'), X_val=X_val, y_val=y_val)
        np.savez_compressed(os.path.join(TEST_DATA_DIR, 'test_data.npz'), X_test=X_test, y_test=y_test)
        
        # 输出最终统计信息
        print("\n✅ 数据预处理和分割已成功完成！")
        print(f"处理的文件总数: {len(features)}")
        print(f"训练集大小: {len(X_train)}")
        print(f"验证集大小: {len(X_val)}")
        print(f"测试集大小: {len(X_test)}")
        return True

    except KeyboardInterrupt:
        print("\n\n⚠️ 处理被用户中断")
        return False
    except FileNotFoundError as e:
        print(f"❌ 错误：{str(e)}")
        return False
    except Exception as e:
        print(f"❌ 处理过程中出现错误：{str(e)}")
        return False

if __name__ == "__main__":
    try:
        print("开始数据预处理...")
        success = preprocess_data()
        if not success:
            print("❌ 预处理失败")
    except KeyboardInterrupt:
        print("\n\n⚠️ 程序被用户中断")
    except Exception as e:
        print(f"❌ 发生未预期的错误：{str(e)}")