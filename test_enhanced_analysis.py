#!/usr/bin/env python3
"""
测试增强后的pqc_reup_analyze.py功能
验证Top-N accuracy统计和完整分析绘图功能
"""

import json
import os
import numpy as np
from datetime import datetime

def create_test_data():
    """创建测试用的完整数据集"""
    output_dir = 'pqc_reup_v1_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模拟的训练数据
    training_data = {
        'train_losses': [0.5, 0.4, 0.35, 0.3, 0.25, 0.22, 0.20, 0.18, 0.16, 0.14],
        'test_losses': [0.6, 0.5, 0.45, 0.4, 0.35, 0.32, 0.30, 0.28, 0.26, 0.24],
        'mae_scores': [0.8, 0.7, 0.65, 0.6, 0.55, 0.52, 0.50, 0.48, 0.46, 0.44],
        'epochs': 10
    }
    
    # 创建模拟的评估数据（100个样本，208个输出维度）
    n_samples = 100
    n_outputs = 208
    
    # 生成模拟预测和真实数据
    np.random.seed(42)  # 确保结果可重现
    predictions = np.random.rand(n_samples, n_outputs) * 100  # 0-100范围
    targets = predictions + np.random.normal(0, 5, (n_samples, n_outputs))  # 添加噪声
    
    # 确保数据在合理范围内
    predictions = np.clip(predictions, 0, 100)
    targets = np.clip(targets, 0, 100)
    
    evaluation_data = {
        'predictions': predictions.tolist(),
        'targets': targets.tolist(),
        'metrics': {
            'MSE': 25.0,
            'MAE': 4.0,
            'RMSE': 5.0,
            'R2': 0.95
        }
    }
    
    # 保存文件
    train_file = os.path.join(output_dir, 'training_data_epoch_10.json')
    eval_file = os.path.join(output_dir, 'evaluation_results_epoch_10.json')
    
    with open(train_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    with open(eval_file, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    print(f"✓ 创建测试数据文件:")
    print(f"  训练数据: {train_file}")
    print(f"  评估数据: {eval_file}")
    print(f"  数据规模: {n_samples} 样本 × {n_outputs} 输出维度")
    
    return train_file, eval_file

def test_command_line_interface():
    """测试命令行接口功能"""
    print("=== 测试增强的命令行分析功能 ===")
    
    # 创建测试数据
    train_file, eval_file = create_test_data()
    
    # 测试不同的命令行参数组合
    test_commands = [
        "python pqc_reup_analyze.py",
        "python pqc_reup_analyze.py --epoch 10",
        "python pqc_reup_analyze.py --epoch 10 --input-indices '0,1,2,3,4,5,6,7,8,9,10,11'",
        "python pqc_reup_analyze.py --epoch 10 --top-n-max 5",
        "python pqc_reup_analyze.py --epoch 10 --input-indices '0,1,2,3' --top-n-max 8"
    ]
    
    import subprocess
    import sys
    
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n--- 测试命令 {i}: {cmd} ---")
        try:
            # 使用subprocess运行命令
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✓ 命令执行成功")
                # 显示部分输出（前500字符）
                output_preview = result.stdout[:500]
                if len(result.stdout) > 500:
                    output_preview += "... (输出截断)"
                print(f"输出预览:\n{output_preview}")
            else:
                print(f"✗ 命令执行失败 (返回码: {result.returncode})")
                print(f"错误信息: {result.stderr[:200]}")
                
        except subprocess.TimeoutExpired:
            print("✗ 命令执行超时")
        except Exception as e:
            print(f"✗ 执行异常: {e}")

def test_top_n_calculation():
    """测试Top-N准确率计算功能"""
    print("\n=== 测试Top-N准确率计算功能 ===")
    
    # 导入分析模块
    import sys
    sys.path.append('/Users/luxian/GitSpace/beam_pre')
    import pqc_reup_analyze
    
    # 创建简单的测试数据
    predictions = np.array([
        [0.1, 0.8, 0.3, 0.6, 0.2],  # 样本1
        [0.9, 0.2, 0.7, 0.1, 0.4],  # 样本2
        [0.3, 0.1, 0.9, 0.2, 0.5]   # 样本3
    ])
    
    targets = np.array([
        [0.2, 0.9, 0.1, 0.7, 0.3],  # 样本1真实值
        [0.8, 0.3, 0.6, 0.2, 0.5],  # 样本2真实值
        [0.1, 0.2, 0.8, 0.3, 0.6]   # 样本3真实值
    ])
    
    input_indices = [0, 1]  # 前两个作为输入波束
    
    print(f"测试数据:")
    print(f"  预测值形状: {predictions.shape}")
    print(f"  真实值形状: {targets.shape}")
    print(f"  输入索引: {input_indices}")
    
    # 计算Top-N准确率
    top_n_results = pqc_reup_analyze.calculate_top_n_accuracy_both_methods(
        predictions, targets, input_indices, top_n_max=3
    )
    
    print(f"\n计算结果:")
    print(f"方法A (包含输入波束): {top_n_results['with_input']}")
    print(f"方法B (排除输入波束): {top_n_results['without_input']}")
    
    # 验证结果合理性
    for method_name, accuracies in top_n_results.items():
        is_monotonic = all(accuracies[i] <= accuracies[i+1] for i in range(len(accuracies)-1))
        print(f"{method_name} 准确率单调递增: {'✓' if is_monotonic else '✗'}")

if __name__ == "__main__":
    # 运行测试
    create_test_data()
    test_top_n_calculation()
    test_command_line_interface()