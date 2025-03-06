"""
打鼾检测模型评估脚本

主要功能：
1. 加载测试数据和训练好的模型
2. 评估模型性能
3. 生成评估报告
"""

import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from config import TEST_DATA_DIR, MODEL_DIR

def load_test_data():
    """加载测试数据"""
    data = np.load(os.path.join(TEST_DATA_DIR, 'test_data.npz'))
    return data['X_test'], data['y_test']

def print_ascii_curve(x, y, width=50, height=20):
    """使用ASCII字符绘制简单的曲线图"""
    # 创建空白画布
    canvas = [[' ' for _ in range(width)] for _ in range(height)]
    
    # 归一化数据点到画布大小
    x_normalized = ((x - min(x)) / (max(x) - min(x)) * (width-1)).astype(int)
    y_normalized = ((y - min(y)) / (max(y) - min(y)) * (height-1)).astype(int)
    y_normalized = height - 1 - y_normalized  # 翻转Y轴
    
    # 绘制点
    for i in range(len(x_normalized)):
        canvas[y_normalized[i]][x_normalized[i]] = '*'
    
    # 绘制坐标轴
    for i in range(height):
        canvas[i][0] = '|'
    for i in range(width):
        canvas[height-1][i] = '-'
    
    # 打印画布
    print("\nROC曲线 (ASCII格式):")
    print("  " + "假阳性率 →")
    for row in canvas:
        print(''.join(row))
    print("↑")
    print("真")
    print("阳")
    print("性")
    print("率")

def evaluate_model():
    """评估模型性能"""
    try:
        print("正在加载模型...")
        model = load_model(os.path.join(MODEL_DIR, 'final_model.h5'))
        
        print("正在加载测试数据...")
        X_test, y_test = load_test_data()
        
        print("正在进行预测...")
        predictions = (model.predict(X_test) > 0.5).astype(int)

        print("\n分类报告:")
        print(classification_report(y_test, predictions, 
                                 target_names=['非打鼾', '打鼾']))
        
        conf_matrix = confusion_matrix(y_test, predictions)
        print("\n混淆矩阵:")
        print(conf_matrix)
        print("\n混淆矩阵说明:")
        print(f"真阴性 (TN): {conf_matrix[0][0]} - 正确识别为非打鼾")
        print(f"假阳性 (FP): {conf_matrix[0][1]} - 错误识别为打鼾")
        print(f"假阴性 (FN): {conf_matrix[1][0]} - 错误识别为非打鼾")
        print(f"真阳性 (TP): {conf_matrix[1][1]} - 正确识别为打鼾")
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))
        roc_auc = auc(fpr, tpr)
        
        # 使用ASCII字符绘制ROC曲线
        print_ascii_curve(fpr, tpr)
        print(f"\nAUC (曲线下面积): {roc_auc:.2f}")
        
        # 计算和显示详细指标
        accuracy = (conf_matrix[0][0] + conf_matrix[1][1]) / np.sum(conf_matrix)
        sensitivity = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0])  # 真阳性率
        specificity = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])  # 真阴性率
        precision = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[0][1])    # 精确率
        
        print("\n详细性能指标:")
        print(f"准确率 (Accuracy): {accuracy:.2%}")
        print(f"敏感度 (Sensitivity): {sensitivity:.2%}")
        print(f"特异度 (Specificity): {specificity:.2%}")
        print(f"精确率 (Precision): {precision:.2%}")
        
        # 保存评估结果到文本文件
        os.makedirs('evaluation_results', exist_ok=True)
        with open('evaluation_results/evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write("打鼾检测模型评估报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"准确率: {accuracy:.2%}\n")
            f.write(f"敏感度: {sensitivity:.2%}\n")
            f.write(f"特异度: {specificity:.2%}\n")
            f.write(f"精确率: {precision:.2%}\n")
            f.write(f"AUC: {roc_auc:.2f}\n\n")
            f.write("混淆矩阵:\n")
            f.write(str(conf_matrix) + "\n")
        
        print(f"\n✅ 详细评估报告已保存至 'evaluation_results/evaluation_report.txt'")
        
    except FileNotFoundError as e:
        print(f"❌ 错误：找不到所需文件 - {str(e)}")
    except Exception as e:
        print(f"❌ 评估过程中出现错误：{str(e)}")

if __name__ == "__main__":
    try:
        print("=== 开始模型评估 ===")
        evaluate_model()
        print("\n=== 评估完成 ===")
    except KeyboardInterrupt:
        print("\n⚠️ 评估被用户中断")
    except Exception as e:
        print(f"❌ 发生未预期的错误：{str(e)}")
