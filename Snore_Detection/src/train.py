"""
打鼾检测模型训练脚本

主要功能：
1. 加载预处理后的数据
2. 创建和训练模型
3. 保存训练好的模型
"""

import os
import numpy as np
from config import TRAIN_DATA_DIR, EPOCHS, BATCH_SIZE, MODEL_DIR
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from model import create_model  

def load_data():
    """加载预处理后的训练数据和验证数据"""
    train_data = np.load(os.path.join(TRAIN_DATA_DIR, 'train_data.npz'))
    val_data = np.load(os.path.join(TRAIN_DATA_DIR, 'val_data.npz'))

    X_train = train_data['X_train']
    y_train = train_data['y_train']
    X_val = val_data['X_val']
    y_val = val_data['y_val']
    
    return X_train, y_train, X_val, y_val

def train_model():
    """训练模型的主函数"""
    # 创建模型保存目录
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    # 加载数据
    print("正在加载数据...")
    X_train, y_train, X_val, y_val = load_data()
    
    # 打印数据形状信息
    print(f"训练数据形状: {X_train.shape}")
    print(f"验证数据形状: {X_val.shape}")
    
    # 获取数据维度信息
    time_steps = X_train.shape[1]  # 141
    features = X_train.shape[2]  # 1280
    
    # 计算数据大小
    num_samples_train = X_train.shape[0]
    num_samples_val = X_val.shape[0]
    total_elements_train = X_train.size
    total_elements_val = X_val.size

    expected_shape_train = (num_samples_train, time_steps, features)
    expected_shape_val = (num_samples_val, time_steps, features)

    print(f"训练数据总元素数: {total_elements_train}")
    print(f"验证数据总元素数: {total_elements_val}")
    print(f"期望的训练数据形状: {expected_shape_train}")
    print(f"期望的验证数据形状: {expected_shape_val}")
    
    # 检查数据形状
    if total_elements_train != np.prod(expected_shape_train):
        print(f"错误：无法将训练数据重塑为 {expected_shape_train}，因为元素总数不匹配。")
    if total_elements_val != np.prod(expected_shape_val):
        print(f"错误：无法将验证数据重塑为 {expected_shape_val}，因为元素总数不匹配。")
    
    # 重塑数据
    X_train = X_train.reshape(expected_shape_train)
    X_val = X_val.reshape(expected_shape_val)
    
    # 创建模型
    print("\n创建模型...")
    model = create_model(input_shape=(time_steps, features))
    
    # 设置回调函数
    print("配置训练参数...")
    callbacks = [
        # 保存最佳模型权重
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'best_model_weights.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            save_weights_only=True
        ),
        # 早停
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # 学习率调整
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # TensorBoard日志
        TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]

    # 训练模型
    print("\n开始训练模型...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # 保存最终模型
    print("\n保存模型...")
    try:
        model.save(os.path.join(MODEL_DIR, 'final_model.h5'))
        print("✅ 模型训练和保存完成！")
    except Exception as e:
        print(f"❌ 保存模型时出错: {str(e)}")
        # 尝试使用不同的保存格式
        try:
            model.save(os.path.join(MODEL_DIR, 'final_model'), save_format='tf')
            print("✅ 模型已使用TensorFlow格式保存")
        except Exception as e:
            print(f"❌ 保存模型失败: {str(e)}")
    
if __name__ == "__main__":
    try:
        print("=== 打鼾检测模型训练开始 ===")
        train_model()
    except KeyboardInterrupt:
        print("\n\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"❌ 训练过程中出现错误：{str(e)}")
    finally:
        print("\n=== 训练过程结束 ===")
