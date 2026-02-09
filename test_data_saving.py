#!/usr/bin/env python3
"""
测试训练数据保存功能
验证是否同时保存当前运行数据和合并后的完整历史数据
"""

import json
import os
from datetime import datetime

def test_data_saving():
    """测试数据保存功能"""
    output_dir = 'pqc_reup_v1_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 模拟训练数据
    current_epochs = 5
    historical_epochs = 10
    
    # 模拟当前运行的训练数据
    current_train_losses = [0.1, 0.08, 0.06, 0.05, 0.04]
    current_test_losses = [0.15, 0.12, 0.10, 0.08, 0.07]
    current_mae_scores = [0.3, 0.25, 0.22, 0.20, 0.18]
    
    # 模拟历史训练数据
    historical_train_losses = [0.5, 0.4, 0.35, 0.3, 0.25, 0.22, 0.20, 0.18, 0.16, 0.14]
    historical_test_losses = [0.6, 0.5, 0.45, 0.4, 0.35, 0.32, 0.30, 0.28, 0.26, 0.24]
    historical_mae_scores = [0.8, 0.7, 0.65, 0.6, 0.55, 0.52, 0.50, 0.48, 0.46, 0.44]
    
    # 合并后的完整数据
    merged_train_losses = historical_train_losses + current_train_losses
    merged_test_losses = historical_test_losses + current_test_losses
    merged_mae_scores = historical_mae_scores + current_mae_scores
    
    final_epoch = len(merged_train_losses)
    
    print("=== 测试训练数据保存功能 ===")
    
    # 1. 保存完整的训练过程数据（包含历史+新数据）
    complete_training_data = {
        'train_losses': merged_train_losses,
        'test_losses': merged_test_losses,
        'mae_scores': merged_mae_scores,
        'epochs': len(merged_train_losses),
        'training_info': {
            'start_epoch': historical_epochs,
            'final_epoch': final_epoch,
            'historical_epochs': historical_epochs,
            'new_epochs': current_epochs
        }
    }
    
    complete_data_path = os.path.join(output_dir, f'training_data_epoch_{final_epoch}.json')
    with open(complete_data_path, 'w') as f:
        json.dump(complete_training_data, f, indent=2)
    print(f"✓ 完整训练过程数据已保存到: {complete_data_path}")
    
    # 2. 保存当前运行的训练数据（仅本次训练的数据点）
    current_training_data = {
        'train_losses': current_train_losses,
        'test_losses': current_test_losses,
        'mae_scores': current_mae_scores,
        'epochs': current_epochs,
        'training_info': {
            'start_epoch': historical_epochs,
            'final_epoch': current_epochs,
            'is_current_run_only': True
        }
    }
    
    current_data_path = os.path.join(output_dir, f'training_data_current_run_epoch_{final_epoch}.json')
    with open(current_data_path, 'w') as f:
        json.dump(current_training_data, f, indent=2)
    print(f"✓ 当前运行训练数据已保存到: {current_data_path}")
    
    # 3. 验证保存的数据
    print("\n=== 数据验证 ===")
    
    # 读取并验证完整数据
    with open(complete_data_path, 'r') as f:
        loaded_complete = json.load(f)
    
    # 读取并验证当前运行数据
    with open(current_data_path, 'r') as f:
        loaded_current = json.load(f)
    
    print(f"完整数据总epochs: {loaded_complete['epochs']}")
    print(f"当前运行epochs: {loaded_current['epochs']}")
    print(f"历史epochs: {loaded_complete['training_info']['historical_epochs']}")
    
    # 验证数据一致性
    expected_total = loaded_current['epochs'] + loaded_complete['training_info']['historical_epochs']
    actual_total = loaded_complete['epochs']
    
    if expected_total == actual_total:
        print(f"✓ 数据一致性验证通过: {loaded_current['epochs']} + {loaded_complete['training_info']['historical_epochs']} = {loaded_complete['epochs']}")
    else:
        print(f"✗ 数据一致性验证失败: 期望 {expected_total}, 实际 {actual_total}")
    
    # 验证数据内容
    print(f"\n=== 数据内容预览 ===")
    print("完整训练数据 (前3个, 后3个):")
    print(f"  训练损失: {loaded_complete['train_losses'][:3]} ... {loaded_complete['train_losses'][-3:]}")
    print(f"  测试损失: {loaded_complete['test_losses'][:3]} ... {loaded_complete['test_losses'][-3:]}")
    
    print("\n当前运行数据:")
    print(f"  训练损失: {loaded_current['train_losses']}")
    print(f"  测试损失: {loaded_current['test_losses']}")
    
    return True

if __name__ == "__main__":
    test_data_saving()