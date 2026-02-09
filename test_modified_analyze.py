#!/usr/bin/env python3
"""
测试修改后的pqc_reup_analyze.py功能
验证删除合并训练数据代码后的功能是否正常
"""

import json
import os
import numpy as np

def create_test_data():
    """创建测试用的训练数据文件"""
    output_dir = 'pqc_reup_v1_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建包含累积训练信息的训练数据
    training_data = {
        'train_losses': [0.5, 0.4, 0.35, 0.3, 0.25, 0.22, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10],
        'test_losses': [0.6, 0.5, 0.45, 0.4, 0.35, 0.32, 0.30, 0.28, 0.26, 0.24, 0.22, 0.20],
        'mae_scores': [0.8, 0.7, 0.65, 0.6, 0.55, 0.52, 0.50, 0.48, 0.46, 0.44, 0.42, 0.40],
        'epochs': 12,
        'training_info': {
            'start_epoch': 8,
            'final_epoch': 12,
            'historical_epochs': 8,
            'new_epochs': 4
        }
    }
    
    # 创建评估数据
    evaluation_data = {
        'predictions': np.random.rand(50, 208).tolist(),  # 50个样本，208个输出维度
        'targets': np.random.rand(50, 208).tolist(),
        'metrics': {
            'MSE': 25.0,
            'MAE': 4.0,
            'RMSE': 5.0,
            'R2': 0.95
        }
    }
    
    # 保存文件
    train_file = os.path.join(output_dir, 'training_data_epoch_12.json')
    eval_file = os.path.join(output_dir, 'evaluation_results_epoch_12.json')
    
    with open(train_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    with open(eval_file, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    
    print(f"✓ 创建测试数据文件:")
    print(f"  训练数据: {train_file}")
    print(f"  评估数据: {eval_file}")
    print(f"  累积epochs: {len(training_data['train_losses'])}")
    
    return train_file, eval_file

def test_analyze_function():
    """测试analyze_results函数"""
    print("=== 测试修改后的analyze_results函数 ===")
    
    # 创建测试数据
    train_file, eval_file = create_test_data()
    
    # 导入分析模块
    import sys
    sys.path.append('/Users/luxian/GitSpace/beam_pre')
    import pqc_reup_analyze
    
    # 测试完整分析
    try:
        # 加载测试数据
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
        
        predictions = np.array(eval_data['predictions'])
        targets = np.array(eval_data['targets'])
        input_indices = list(range(12))  # 默认输入索引
        
        print("调用analyze_results函数...")
        pqc_reup_analyze.analyze_results(
            epoch_num=12,
            predictions=predictions,
            targets=targets,
            input_indices=input_indices,
            output_dir='pqc_reup_v1_output'
        )
        
        # 检查生成的文件
        output_dir = 'pqc_reup_v1_output'
        expected_files = [
            'results_cumulative_12_epochs.png',
            'training_curves_cumulative_12_epochs.png',
            'analysis_report_epoch_12.md'
        ]
        
        print("\n检查生成的文件:")
        for filename in expected_files:
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                print(f"✓ {filename}")
            else:
                print(f"✗ {filename} (未找到)")
                
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_analyze_function()