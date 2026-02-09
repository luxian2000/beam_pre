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

def plot_comprehensive_results(train_losses, test_losses, mae_scores, predictions, targets, input_indices, epoch_num, output_dir='pqc_reup_v1_output'):
    """绘制综合结果图像，包含训练曲线和Top-N分析"""
    # 计算Top-N准确率
    top_n_results = calculate_top_n_accuracy_both_methods(
        predictions, targets, input_indices, top_n_max=10
    )
    
    # 创建2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 第一行：训练相关曲线
    epochs = range(len(train_losses))
    
    # 训练和测试损失曲线
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title(f'Training and Test Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE曲线
    axes[0, 1].plot(epochs, mae_scores, 'g-', label='MAE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title(f'Mean Absolute Error Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top-N准确率对比曲线
    n_values = list(range(1, len(top_n_results['with_input']) + 1))
    axes[0, 2].plot(n_values, top_n_results['with_input'], 'o-', linewidth=2, markersize=6, 
                    label='Including Input Beams', color='blue')
    axes[0, 2].plot(n_values, top_n_results['without_input'], 's-', linewidth=2, markersize=6, 
                    label='Excluding Input Beams', color='red')
    axes[0, 2].set_xlabel('N')
    axes[0, 2].set_ylabel('Top-N Accuracy')
    axes[0, 2].set_title(f'Top-N Accuracy Comparison')
    axes[0, 2].set_xticks(n_values)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 第二行：预测分析
    sample_size = min(1000, len(predictions))
    sample_indices = np.random.choice(len(predictions), sample_size, replace=False)
    
    # 预测vs真实值散点图
    axes[1, 0].scatter(targets[sample_indices].flatten(), predictions[sample_indices].flatten(), 
                      alpha=0.5, s=1)
    axes[1, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('True Values')
    axes[1, 0].set_ylabel('Predictions')
    axes[1, 0].set_title(f'Predictions vs True Values (Sample)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 误差分布
    errors = (predictions - targets).flatten()
    axes[1, 1].hist(errors, bins=50, alpha=0.7, color='green')
    axes[1, 1].set_xlabel('Prediction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Top-N准确率数值表
    axes[1, 2].axis('tight')
    axes[1, 2].axis('off')
    
    # 创建表格数据
    table_data = [['Method', 'Top-N', 'Accuracy', 'Percentage']]
    
    # 方法A数据
    for i, acc in enumerate(top_n_results['with_input']):
        table_data.append([
            'With Input' if i == 0 else '', 
            f'Top-{i+1}', 
            f'{acc:.4f}', 
            f'{acc*100:.2f}%'
        ])
    
    # 添加分隔行
    table_data.append(['', '', '', ''])
    
    # 方法B数据
    for i, acc in enumerate(top_n_results['without_input']):
        table_data.append([
            'Without Input' if i == 0 else '', 
            f'Top-{i+1}', 
            f'{acc:.4f}', 
            f'{acc*100:.2f}%'
        ])
    
    table = axes[1, 2].table(cellText=table_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    axes[1, 2].set_title(f'Top-N Accuracy Summary')
    
    plt.tight_layout()
    
    # 保存图像
    filename = f'results_epoch_{epoch_num}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"综合结果图像已保存到: {save_path}")
    return filename

# 更新原有的plot_training_curves函数为简化版本（如果不单独需要的话）
def plot_training_curves(train_losses, test_losses, mae_scores, epoch_num, output_dir='pqc_reup_v1_output'):
    """绘制训练曲线（简化版本，主要用于向后兼容）"""
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

def calculate_top_n_accuracy_both_methods(predictions, targets, input_indices, top_n_max=10):
    """计算两种Top-N准确率：包含输入波束和不包含输入波束"""
    n_samples = len(predictions)
    
    # 初始化两种方法的准确率计数器
    top_n_correct_with_input = [0] * top_n_max    # 方法A：包含输入波束
    top_n_correct_without_input = [0] * top_n_max # 方法B：不包含输入波束
    
    # 创建输出波束索引集合（排除输入波束）
    all_indices = set(range(len(predictions[0])))  # 所有波束索引
    output_indices_set = all_indices - set(input_indices)  # 排除输入波束后的索引
    
    for i in range(n_samples):
        pred_sample = predictions[i]
        target_sample = targets[i]
        
        # 方法A：包含输入波束的统计
        pred_indices_A = np.argsort(pred_sample)[::-1]  # 所有波束降序排列
        target_max_idx_A = np.argmax(target_sample)     # 真实最大值索引
        
        # 方法B：不包含输入波束的统计
        pred_values_B = pred_sample[list(output_indices_set)]
        target_values_B = target_sample[list(output_indices_set)]
        
        # 获取输出波束内的排序索引
        pred_local_indices_B = np.argsort(pred_values_B)[::-1]  # 输出波束降序排列
        target_local_max_idx_B = np.argmax(target_values_B)     # 真实最大值在输出波束内的索引
        
        # 将局部索引映射回全局索引
        output_indices_list = list(output_indices_set)
        pred_global_indices_B = [output_indices_list[idx] for idx in pred_local_indices_B]
        target_global_max_idx_B = output_indices_list[target_local_max_idx_B]
        
        # 计算两种方法的Top-N准确率
        for n in range(1, top_n_max + 1):
            # 方法A：检查真实最优波束是否在预测的前N个中（包含所有波束）
            if target_max_idx_A in pred_indices_A[:n]:
                top_n_correct_with_input[n-1] += 1
            
            # 方法B：检查真实最优波束是否在预测的前N个中（仅输出波束）
            if target_global_max_idx_B in pred_global_indices_B[:n]:
                top_n_correct_without_input[n-1] += 1
    
    # 计算准确率
    top_n_accuracies_with_input = [correct / n_samples for correct in top_n_correct_with_input]
    top_n_accuracies_without_input = [correct / n_samples for correct in top_n_correct_without_input]
    
    return {
        'with_input': top_n_accuracies_with_input,      # 方法A
        'without_input': top_n_accuracies_without_input  # 方法B
    }

def plot_top_n_analysis(predictions, targets, input_indices, epoch_num, output_dir='pqc_reup_v1_output'):
    """绘制Top-N准确率分析图像"""
    # 计算两种Top-N准确率
    top_n_results = calculate_top_n_accuracy_both_methods(
        predictions, targets, input_indices, top_n_max=10
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top-N准确率对比曲线
    n_values = list(range(1, len(top_n_results['with_input']) + 1))
    
    # 方法A：包含输入波束
    axes[0].plot(n_values, top_n_results['with_input'], 'o-', linewidth=2, markersize=6, 
                 label='Including Input Beams', color='blue')
    
    # 方法B：不包含输入波束
    axes[0].plot(n_values, top_n_results['without_input'], 'o-', linewidth=2, markersize=6, 
                 label='Excluding Input Beams', color='red')
    
    axes[0].set_xlabel('N')
    axes[0].set_ylabel('Top-N Accuracy')
    axes[0].set_title(f'Top-N Accuracy Comparison - Epoch {epoch_num}')
    axes[0].set_xticks(n_values)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Top-N准确率数值表
    axes[1].axis('tight')
    axes[1].axis('off')
    
    # 创建表格数据
    table_data = [['Method', 'Top-N', 'Accuracy', 'Percentage']]
    
    # 方法A数据
    for i, acc in enumerate(top_n_results['with_input']):
        table_data.append([
            'With Input' if i == 0 else '', 
            f'Top-{i+1}', 
            f'{acc:.4f}', 
            f'{acc*100:.2f}%'
        ])
    
    # 添加分隔行
    table_data.append(['', '', '', ''])
    
    # 方法B数据
    for i, acc in enumerate(top_n_results['without_input']):
        table_data.append([
            'Without Input' if i == 0 else '', 
            f'Top-{i+1}', 
            f'{acc:.4f}', 
            f'{acc*100:.2f}%'
        ])
    
    table = axes[1].table(cellText=table_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    axes[1].set_title(f'Top-N Accuracy Summary - Epoch {epoch_num}')
    
    plt.tight_layout()
    
    # 保存图像
    filename = f'top_n_analysis_epoch_{epoch_num}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Top-N分析图像已保存到: {save_path}")
    return filename

def generate_analysis_report(training_data, predictions, targets, epoch_num, input_indices=None, output_dir='pqc_reup_v1_output'):
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

def analyze_results(epoch_num, predictions=None, targets=None, input_indices=None, output_dir='pqc_reup_v1_output', **kwargs):
    """主分析函数 - 支持额外的关键字参数"""
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
    
    # 如果提供了预测、目标和输入索引数据，则进行完整分析
    if predictions is not None and targets is not None and input_indices is not None:
        print("生成完整分析报告...")
        
        # 使用新的综合图像生成函数（包含训练曲线、预测分析和Top-N准确率）
        results_filename = plot_comprehensive_results(
            training_data['train_losses'],
            training_data['test_losses'],
            training_data['mae_scores'],
            predictions,
            targets,
            input_indices,
            epoch_num,
            output_dir
        )
        
        # 生成分析报告
        report_filename = generate_analysis_report(
            training_data, predictions, targets, epoch_num, input_indices, output_dir
        )
        
        print(f"\n分析完成！生成的文件:")
        print(f"- {results_filename}")
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
    
    # 示例用法（需要提供实际的预测、目标和输入索引数据）
    # analyze_results(epoch_to_analyze, predictions=predictions, targets=targets, input_indices=input_indices)
    # 对于演示，只运行基础分析
    analyze_results(epoch_to_analyze)

if __name__ == "__main__":
    main()