#!/usr/bin/env python3
"""
测试累积训练数据分析功能
验证pqc_reup_analyze.py对累积训练数据的支持
"""

import json
import os
import numpy as np
from datetime import datetime

def create_test_cumulative_data():
    """创建测试用的累积训练数据"""
    output_dir = 'pqc_reup_v1_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模拟的累积训练数据
    cumulative_data = {
        'train_losses': [0.5, 0.4, 0.35, 0.3, 0.25, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.09, 0.08, 0.07],
        'test_losses': [0.6, 0.5, 0.45, 0.4, 0.35, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.19, 0.18, 0.17],
        'mae_scores': [0.8, 0.7, 0.65, 0.6, 0.55, 0.52, 0.50, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, 0.34],
        'epochs': 15,
        'training_info': {
            'start_epoch': 10,
            'final_epoch': 15,
            'historical_epochs': 10,
            'new_epochs': 5
        }
    }
    
    # 创建模拟的当前运行数据
    current_data = {
        'train_losses': [0.12, 0.10, 0.09, 0.08, 0.07],
        'test_losses': [0.22, 0.20, 0.19, 0.18, 0.17],
        'mae_scores': [0.42, 0.40, 0.38, 0.36, 0.34],
        'epochs': 5,
        'training_info': {
            'start_epoch': 10,
            'final_epoch': 5,
            'is_current_run_only': True
        }
    }
    
    # 保存文件
    cumulative_file = os.path.join(output_dir, 'training_data_epoch_15.json')
    current_file = os.path.join(output_dir, 'training_data_current_run_epoch_15.json')
    
    with open(cumulative_file, 'w') as f:
        json.dump(cumulative_data, f, indent=2)
    
    with open(current_file, 'w') as f:
        json.dump(current_data, f, indent=2)
    
    print(f"✓ 创建测试数据文件:")
    print(f"  累积数据: {cumulative_file}")
    print(f"  当前运行数据: {current_file}")
    
    return cumulative_file, current_file

def test_analysis_functions():
    """测试分析函数"""
    print("=== 测试累积训练数据分析功能 ===")
    
    # 创建测试数据
    cum_file, curr_file = create_test_cumulative_data()
    
    # 导入分析模块
    import sys
    sys.path.append('/Users/luxian/GitSpace/beam_pre')
    import pqc_reup_analyze
    
    output_dir = 'pqc_reup_v1_output'
    
    print("\n1. 测试加载累积训练数据:")
    try:
        cumulative_data = pqc_reup_analyze.load_cumulative_training_data(output_dir)
        print(f"✓ 成功加载累积数据: {cumulative_data['epochs']} epochs")
        print(f"  历史epochs: {cumulative_data['training_info']['historical_epochs']}")
        print(f"  新增epochs: {cumulative_data['training_info']['new_epochs']}")
    except Exception as e:
        print(f"✗ 加载累积数据失败: {e}")
    
    print("\n2. 测试加载指定epoch数据:")
    try:
        epoch_data = pqc_reup_analyze.load_training_data(15, output_dir)
        print(f"✓ 成功加载epoch 15数据: {epoch_data['epochs']} epochs")
    except Exception as e:
        print(f"✗ 加载指定epoch数据失败: {e}")
    
    print("\n3. 测试智能分析功能:")
    try:
        # 创建模拟的预测数据
        predictions = np.random.rand(100, 208) * 100  # 100个样本，208个输出
        targets = predictions + np.random.normal(0, 5, predictions.shape)  # 添加噪声
        input_indices = list(range(48))  # 前48个作为输入波束
        
        # 测试累积数据分析
        pqc_reup_analyze.analyze_results(
            epoch_num=15,
            predictions=predictions,
            targets=targets,
            input_indices=input_indices,
            output_dir=output_dir,
            use_cumulative=True
        )
        print("✓ 累积数据分析完成")
        
    except Exception as e:
        print(f"✗ 智能分析测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_analysis_functions()