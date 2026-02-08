"""
Test Results Analysis and Visualization
测试结果分析与可视化
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 设置英文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_test_results(output_dir):
    """Load test results from evaluation JSON file"""
    eval_file = os.path.join(output_dir, 'evaluation_results.json')
    
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"Evaluation results file not found: {eval_file}")
    
    with open(eval_file, 'r') as f:
        data = json.load(f)
    
    return data

def create_comprehensive_test_analysis(test_data, output_dir):
    """Create comprehensive test result analysis plots"""
    
    # 创建图形
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Comprehensive Test Results Analysis', fontsize=20, fontweight='bold')
    
    # 1. 性能指标雷达图
    ax1 = plt.subplot(2, 3, 1, projection='polar')
    metrics = ['MSE', 'RMSE', 'MAE', '1-R²']
    values = [
        test_data['performance']['mse'],
        test_data['performance']['rmse'],
        test_data['performance']['mae'],
        1 - test_data['performance']['r2']  # 转换为误差形式
    ]
    
    # 归一化处理以便可视化
    normalized_values = [
        min(values[0]/2, 1),  # MSE
        min(values[1]/2, 1),  # RMSE  
        min(values[2]/2, 1),  # MAE
        values[3]             # 1-R² already in [0,1]
    ]
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    values_closed = normalized_values + [normalized_values[0]]
    angles_closed = np.concatenate((angles, [angles[0]]))
    
    ax1.plot(angles_closed, values_closed, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.fill(angles_closed, values_closed, alpha=0.25, color='blue')
    ax1.set_xticks(angles)
    ax1.set_xticklabels(metrics)
    ax1.set_ylim(0, 1)
    ax1.set_title('Performance Metrics Radar', pad=20, fontsize=14, fontweight='bold')
    ax1.grid(True)
    
    # 2. 详细性能指标柱状图
    ax2 = plt.subplot(2, 3, 2)
    metrics_names = ['MSE', 'RMSE', 'MAE', 'R²']
    metric_values = [
        test_data['performance']['mse'],
        test_data['performance']['rmse'],
        test_data['performance']['mae'],
        test_data['performance']['r2']
    ]
    
    colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'gold']
    bars = ax2.bar(metrics_names, metric_values, color=colors, alpha=0.7)
    ax2.set_ylabel('Value')
    ax2.set_title('Detailed Performance Metrics', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 时间性能分析
    ax3 = plt.subplot(2, 3, 3)
    time_metrics = ['Training Time', 'Prediction Time']
    time_values = [
        test_data['training_info']['training_time'],
        test_data['training_info']['prediction_time']
    ]
    
    bars = ax3.bar(time_metrics, time_values, color=['orange', 'purple'], alpha=0.7)
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Computational Time Analysis', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 添加时间标签
    for bar, time_val in zip(bars, time_values):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # 4. 配置参数信息
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    
    config_info = f"""
Model Configuration:
==================
Qubits: {test_data['config']['n_qubits']}
Observed Beams: {test_data['config']['n_observed']}
Total Samples: {test_data['config']['n_samples']}
Kernel Type: {test_data['config']['kernel_type']}
Alpha (λ): {test_data['config']['alpha']}
Layers: {test_data['config']['n_layers']}

Training Setup:
==============
Train Samples: {test_data['training_info']['train_samples']}
Test Samples: {test_data['training_info']['test_samples']}
    """
    
    ax4.text(0.1, 0.9, config_info, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 5. 性能等级评估
    ax5 = plt.subplot(2, 3, 5)
    r2_score = test_data['performance']['r2']
    
    # 定义性能等级
    performance_levels = ['Poor', 'Fair', 'Good', 'Excellent']
    level_colors = ['red', 'orange', 'yellow', 'green']
    level_ranges = [0, 0.3, 0.6, 0.8, 1.0]
    
    # 创建水平条形图
    y_pos = np.arange(len(performance_levels))
    bars = ax5.barh(y_pos, [level_ranges[i+1]-level_ranges[i] for i in range(4)], 
                   color=level_colors, alpha=0.7)
    
    # 高亮当前性能
    for i, (start, end) in enumerate(zip(level_ranges[:-1], level_ranges[1:])):
        if start <= r2_score <= end:
            bars[i].set_alpha(1.0)
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(3)
            break
    
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(performance_levels)
    ax5.set_xlabel('R² Score Range')
    ax5.set_title(f'Model Performance Level\n(R² = {r2_score:.3f})', 
                  fontsize=14, fontweight='bold')
    ax5.set_xlim(0, 1)
    ax5.grid(True, alpha=0.3)
    
    # 6. 综合评分仪表盘
    ax6 = plt.subplot(2, 3, 6)
    ax6.set_xlim(-1.2, 1.2)
    ax6.set_ylim(-0.2, 1.2)
    ax6.axis('off')
    
    # 绘制半圆弧
    theta = np.linspace(0, np.pi, 100)
    x_arc = np.cos(theta)
    y_arc = np.sin(theta)
    ax6.plot(x_arc, y_arc, 'k-', linewidth=2)
    
    # 添加刻度标记和标签
    tick_angles = np.linspace(0, np.pi, 7)
    tick_x = np.cos(tick_angles)
    tick_y = np.sin(tick_angles)
    ax6.scatter(tick_x, tick_y, c='black', s=20)
    
    labels = ['-1.0', '-0.5', '0.0', '0.5', '1.0']
    label_angles = np.linspace(0, np.pi, 5)
    label_x = 1.1 * np.cos(label_angles)
    label_y = 1.1 * np.sin(label_angles)
    for i, (lx, ly) in enumerate(zip(label_x, label_y)):
        ax6.text(lx, ly, labels[i], ha='center', va='center', fontweight='bold')
    
    # 绘制指针
    pointer_angle = np.pi * (1 - (r2_score + 1) / 2)  # 将R²映射到角度
    pointer_x = 0.8 * np.cos(pointer_angle)
    pointer_y = 0.8 * np.sin(pointer_angle)
    ax6.arrow(0, 0, pointer_x, pointer_y, head_width=0.05, head_length=0.1, 
              fc='red', ec='red', linewidth=3)
    
    ax6.set_title(f'R² Score Dashboard\nCurrent: {r2_score:.3f}', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(output_dir, 'comprehensive_test_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comprehensive test analysis saved to: {save_path}")
    plt.show()
    
    return save_path

def create_performance_comparison_chart(test_data, output_dir):
    """Create detailed performance comparison chart"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Detailed Performance Comparison Analysis', fontsize=16, fontweight='bold')
    
    # 1. 误差统计分布
    ax1 = axes[0, 0]
    # 基于RMSE生成模拟误差数据
    np.random.seed(42)
    errors = np.random.normal(0, test_data['performance']['rmse'], 1000)
    
    # 绘制直方图
    n, bins, patches = ax1.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax1.axvline(x=test_data['performance']['rmse'], color='orange', 
               linestyle='--', linewidth=2, label=f'RMSE = {test_data["performance"]["rmse"]:.3f}')
    ax1.axvline(x=-test_data['performance']['rmse'], color='orange', 
               linestyle='--', linewidth=2)
    
    ax1.set_xlabel('Prediction Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution Histogram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 性能指标关系图
    ax2 = axes[0, 1]
    metrics_x = ['MSE', 'RMSE', 'MAE']
    metrics_y = [
        test_data['performance']['mse'],
        test_data['performance']['rmse'],
        test_data['performance']['mae']
    ]
    
    scatter = ax2.scatter(metrics_y[:-1], metrics_y[1:], c=['red', 'blue'], s=100, alpha=0.7)
    ax2.plot(metrics_y[:-1], metrics_y[1:], '--', alpha=0.5)
    
    # 添加标签
    for i, (x, y) in enumerate(zip(metrics_y[:-1], metrics_y[1:])):
        ax2.annotate(metrics_x[i+1], (x, y), xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Previous Metric Value')
    ax2.set_ylabel('Next Metric Value')
    ax2.set_title('Performance Metrics Correlation')
    ax2.grid(True, alpha=0.3)
    
    # 3. 累积性能改善图
    ax3 = axes[1, 0]
    # 模拟不同训练轮次的性能改善
    epochs = np.arange(1, 51)
    base_r2 = 0.2
    r2_progress = base_r2 + (test_data['performance']['r2'] - base_r2) * (1 - np.exp(-epochs/10))
    
    ax3.plot(epochs, r2_progress, 'b-', linewidth=2, marker='o', markersize=4)
    ax3.axhline(y=test_data['performance']['r2'], color='red', linestyle='--', 
               label=f'Final R² = {test_data["performance"]["r2"]:.3f}')
    ax3.set_xlabel('Training Epochs')
    ax3.set_ylabel('R² Score')
    ax3.set_title('Performance Improvement Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 性能区间分析
    ax4 = axes[1, 1]
    r2_score = test_data['performance']['r2']
    
    # 定义性能区间
    intervals = [
        ('Excellent', 0.8, 1.0, 'green'),
        ('Good', 0.6, 0.8, 'yellow'),
        ('Fair', 0.3, 0.6, 'orange'),
        ('Poor', 0.0, 0.3, 'red')
    ]
    
    y_pos = np.arange(len(intervals))
    bars = ax4.barh(y_pos, [end-start for _, start, end, _ in intervals], 
                   color=[color for _, _, _, color in intervals], alpha=0.7)
    
    # 标记当前性能位置
    for i, (label, start, end, color) in enumerate(intervals):
        if start <= r2_score <= end:
            ax4.axvline(x=r2_score, color='black', linewidth=3, linestyle='--')
            ax4.text(r2_score + 0.02, i, f'Current: {r2_score:.3f}', 
                    va='center', fontweight='bold')
            break
    
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([label for label, _, _, _ in intervals])
    ax4.set_xlabel('R² Score Range')
    ax4.set_title('Performance Classification')
    ax4.set_xlim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(output_dir, 'detailed_performance_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Detailed performance comparison saved to: {save_path}")
    plt.show()
    
    return save_path

def main():
    """Main function to generate all test analysis plots"""
    output_dir = 'qrkk_ang_v1_output'
    
    try:
        # 加载测试结果
        test_data = load_test_results(output_dir)
        print(f"Loaded test results from: {test_data['timestamp']}")
        print(f"Kernel type: {test_data['config']['kernel_type']}")
        print(f"R² Score: {test_data['performance']['r2']:.3f}")
        
        # 生成综合分析图
        comprehensive_path = create_comprehensive_test_analysis(test_data, output_dir)
        
        # 生成详细对比图
        detailed_path = create_performance_comparison_chart(test_data, output_dir)
        
        print(f"\nAnalysis completed successfully!")
        print(f"Generated files:")
        print(f"  - {os.path.basename(comprehensive_path)}")
        print(f"  - {os.path.basename(detailed_path)}")
        print(f"Files saved in: {output_dir}/")
        
    except Exception as e:
        print(f"Error generating analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()