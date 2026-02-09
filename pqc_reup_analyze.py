#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PQC_REUP_V1 分析模块
用于生成训练结果的可视化图像和分析报告
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_training_data(epoch_num, output_dir='pqc_reup_v1_output'):
    """加载指定epoch的训练数据"""
    data_path = os.path.join(output_dir, f'training_data_epoch_{epoch_num}.json')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"训练数据文件不存在: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    return data

def load_model_params(epoch_num, model_class, output_dir='pqc_reup_v1_output'):
    """加载指定epoch的模型参数"""
    model_path = os.path.join(output_dir, f'model_params_epoch_{epoch_num}.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型参数文件不存在: {model_path}")
    
    # 这里需要传入模型类来重建模型结构
    # 由于模型定义在主文件中，这里暂时返回路径
    return model_path

def plot_training_curves(train_losses, test_losses, mae_scores, epoch_num, output_dir='pqc_reup_v1_output'):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 损失曲线
    epochs = range(len(train_losses))
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Training and Test Loss Curves (Epoch {epoch_num})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE曲线
    axes[1].plot(epochs, mae_scores, 'g-', label='MAE', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title(f'Mean Absolute Error Over Time (Epoch {epoch_num})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    filename = f'training_curves_epoch_{epoch_num}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存到: {save_path}")
    return filename

def plot_prediction_analysis(predictions, targets, epoch_num, output_dir='pqc_reup_v1_output'):
    """绘制预测分析图像"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 预测vs真实值散点图（采样显示）
    sample_size = min(1000, len(predictions))
    sample_indices = np.random.choice(len(predictions), sample_size, replace=False)
    
    axes[0, 0].scatter(targets[sample_indices].flatten(), predictions[sample_indices].flatten(), 
                      alpha=0.5, s=1)
    axes[0, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Values')
    axes[0, 0].set_ylabel('Predictions')
    axes[0, 0].set_title(f'Predictions vs True Values (Sample) - Epoch {epoch_num}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 误差分布
    errors = (predictions - targets).flatten()
    axes[0, 1].hist(errors, bins=50, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Prediction Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Error Distribution - Epoch {epoch_num}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 残差图
    axes[1, 0].scatter(predictions[sample_indices].flatten(), errors[sample_indices], 
                      alpha=0.5, s=1)
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Predicted Values')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title(f'Residual Plot - Epoch {epoch_num}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q图（简化版）
    sorted_errors = np.sort(errors)
    theoretical_quantiles = np.linspace(sorted_errors.min(), sorted_errors.max(), len(sorted_errors))
    axes[1, 1].scatter(theoretical_quantiles, sorted_errors, alpha=0.5, s=1)
    axes[1, 1].plot([theoretical_quantiles.min(), theoretical_quantiles.max()], 
                   [theoretical_quantiles.min(), theoretical_quantiles.max()], 'r--', lw=2)
    axes[1, 1].set_xlabel('Theoretical Quantiles')
    axes[1, 1].set_ylabel('Sample Quantiles')
    axes[1, 1].set_title(f'Q-Q Plot (Approximate) - Epoch {epoch_num}')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    filename = f'prediction_analysis_epoch_{epoch_num}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"预测分析图像已保存到: {save_path}")
    return filename

def generate_analysis_report(training_data, predictions, targets, epoch_num, output_dir='pqc_reup_v1_output'):
    """生成分析报告"""
    # 计算评估指标
    mse = mean_squared_error(targets.flatten(), predictions.flatten())
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)
    
    # 计算R²分数
    ss_res = np.sum((targets.flatten() - predictions.flatten()) ** 2)
    ss_tot = np.sum((targets.flatten() - np.mean(targets.flatten())) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 计算皮尔逊相关系数
    correlation, _ = pearsonr(targets.flatten(), predictions.flatten())
    
    # 性能等级评定
    if r2 > 0.75:
        performance_level = "Excellent"
    elif r2 > 0.6:
        performance_level = "Good"
    elif r2 > 0.3:
        performance_level = "Fair"
    elif r2 > 0.1:
        performance_level = "Poor"
    else:
        performance_level = "Very Poor"
    
    # 生成报告内容
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_content = f"""# PQC_REUP_V1 Analysis Report - Epoch {epoch_num}

## Experiment Information
- **Analysis Time**: {timestamp}
- **Trained Epochs**: {epoch_num}
- **Output Directory**: {output_dir}

## Performance Metrics
- **MSE (Mean Squared Error)**: {mse:.6f}
- **MAE (Mean Absolute Error)**: {mae:.6f}
- **RMSE (Root Mean Squared Error)**: {rmse:.6f}
- **R² Score**: {r2:.6f}
- **Pearson Correlation**: {correlation:.6f}
- **Performance Level**: {performance_level}

## Training Statistics
- **Final Training Loss**: {training_data['train_losses'][-1]:.6f}
- **Final Test Loss**: {training_data['test_losses'][-1]:.6f}
- **Final MAE**: {training_data['mae_scores'][-1]:.6f}
- **Best Test Loss**: {min(training_data['test_losses']):.6f}
- **Best MAE**: {min(training_data['mae_scores']):.6f}

## Generated Files
- Training curves plot: training_curves_epoch_{epoch_num}.png
- Prediction analysis plot: prediction_analysis_epoch_{epoch_num}.png
- This analysis report: analysis_report_epoch_{epoch_num}.md

## Notes
This analysis was automatically generated by pqc_reup_analyze.py
"""
    
    # 保存报告
    report_filename = f'analysis_report_epoch_{epoch_num}.md'
    report_path = os.path.join(output_dir, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"分析报告已保存到: {report_path}")
    return report_filename

def analyze_results(epoch_num, predictions=None, targets=None, output_dir='pqc_reup_v1_output'):
    """主分析函数"""
    print(f"开始分析第 {epoch_num} 轮训练结果...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载训练数据
    try:
        training_data = load_training_data(epoch_num, output_dir)
        print(f"成功加载训练数据 (Epoch {epoch_num})")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    # 如果提供了预测和目标数据，则进行完整分析
    if predictions is not None and targets is not None:
        print("生成完整分析报告...")
        
        # 绘制训练曲线
        curve_filename = plot_training_curves(
            training_data['train_losses'],
            training_data['test_losses'],
            training_data['mae_scores'],
            epoch_num,
            output_dir
        )
        
        # 绘制预测分析图像
        analysis_filename = plot_prediction_analysis(predictions, targets, epoch_num, output_dir)
        
        # 生成分析报告
        report_filename = generate_analysis_report(training_data, predictions, targets, epoch_num, output_dir)
        
        print(f"\n分析完成！生成的文件:")
        print(f"- {curve_filename}")
        print(f"- {analysis_filename}")
        print(f"- {report_filename}")
    else:
        print("仅生成训练曲线分析...")
        # 只绘制训练曲线
        curve_filename = plot_training_curves(
            training_data['train_losses'],
            training_data['test_losses'],
            training_data['mae_scores'],
            epoch_num,
            output_dir
        )
        print(f"训练曲线已保存: {curve_filename}")

def main():
    """主函数 - 示例用法"""
    # 这里可以添加命令行参数解析等功能
    epoch_to_analyze = 5  # 默认分析第5轮
    
    print("PQC_REUP_V1 结果分析工具")
    print("=" * 50)
    
    # 执行分析
    analyze_results(epoch_to_analyze)

if __name__ == "__main__":
    main()