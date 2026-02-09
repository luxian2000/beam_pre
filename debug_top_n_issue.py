#!/usr/bin/env python3
"""
调试Top-N统计中方法A和B数据相同的问题
分析为什么在前10个epoch中两种方法的统计数据完全一致
"""

import json
import numpy as np
import os

def analyze_top_n_issue():
    """分析Top-N统计问题"""
    output_dir = 'pqc_reup_v1_output'
    
    # 查找最新的评估结果文件
    eval_files = [f for f in os.listdir(output_dir) if f.startswith('evaluation_results_epoch_') and f.endswith('.json')]
    
    if not eval_files:
        print("未找到评估结果文件")
        return
    
    # 按epoch排序
    eval_files.sort(key=lambda x: int(x.split('_')[3].split('.')[0]))
    
    print("=== Top-N统计问题分析 ===")
    
    for eval_file in eval_files[-3:]:  # 检查最近3个epoch的数据
        eval_path = os.path.join(output_dir, eval_file)
        epoch_num = int(eval_file.split('_')[3].split('.')[0])
        
        print(f"\n--- Epoch {epoch_num} ---")
        
        with open(eval_path, 'r') as f:
            data = json.load(f)
        
        if 'predictions' in data and 'targets' in data:
            predictions = np.array(data['predictions'])
            targets = np.array(data['targets'])
            
            print(f"数据形状: 预测{predictions.shape}, 真实{targets.shape}")
            print(f"数据范围: 预测[{predictions.min():.3f}, {predictions.max():.3f}], 真实[{targets.min():.3f}, {targets.max():.3f}]")
            
            # 检查输入索引配置
            config_files = [f for f in os.listdir(output_dir) if f.startswith('config_') and f.endswith('.md')]
            if config_files:
                config_files.sort(reverse=True)
                latest_config = config_files[0]
                config_path = os.path.join(output_dir, latest_config)
                
                try:
                    with open(config_path, 'r') as f:
                        config_content = f.read()
                        import re
                        indices_match = re.search(r'input_indices.*?\[(.*?)\]', config_content)
                        if indices_match:
                            indices_str = indices_match.group(1)
                            input_indices = [int(x.strip()) for x in indices_str.split(',') if x.strip().isdigit()]
                            print(f"输入索引: {input_indices}")
                            
                            # 分析Top-N计算过程
                            analyze_top_n_calculation(predictions, targets, input_indices, epoch_num)
                            
                except Exception as e:
                    print(f"读取配置文件失败: {e}")
        else:
            print("缺少预测或目标数据")

def analyze_top_n_calculation(predictions, targets, input_indices, epoch_num):
    """详细分析Top-N计算过程"""
    print(f"\n--- Top-N计算过程分析 (Epoch {epoch_num}) ---")
    
    n_samples = len(predictions)
    top_n_max = 10
    
    # 初始化计数器
    top_n_correct_with_input = [0] * top_n_max
    top_n_correct_without_input = [0] * top_n_max
    
    # 创建输出波束索引集合
    all_indices = set(range(len(predictions[0])))
    output_indices_set = all_indices - set(input_indices)
    
    print(f"总波束数: {len(all_indices)}")
    print(f"输入波束数: {len(input_indices)}")
    print(f"输出波束数: {len(output_indices_set)}")
    
    # 分析几个样本的计算过程
    sample_indices_to_check = range(min(3, n_samples))  # 检查前3个样本
    
    for sample_idx in sample_indices_to_check:
        pred_sample = predictions[sample_idx]
        target_sample = targets[sample_idx]
        
        print(f"\n样本 {sample_idx}:")
        
        # 方法A：包含输入波束
        pred_indices_A = np.argsort(pred_sample)[::-1]
        target_max_idx_A = np.argmax(target_sample)
        
        print(f"  方法A - 所有波束:")
        print(f"    预测最大值索引: {pred_indices_A[0]}")
        print(f"    真实最大值索引: {target_max_idx_A}")
        print(f"    前3预测索引: {pred_indices_A[:3]}")
        
        # 方法B：不包含输入波束
        pred_values_B = pred_sample[list(output_indices_set)]
        target_values_B = target_sample[list(output_indices_set)]
        
        pred_local_indices_B = np.argsort(pred_values_B)[::-1]
        target_local_max_idx_B = np.argmax(target_values_B)
        
        output_indices_list = list(output_indices_set)
        pred_global_indices_B = [output_indices_list[idx] for idx in pred_local_indices_B]
        target_global_max_idx_B = output_indices_list[target_local_max_idx_B]
        
        print(f"  方法B - 排除输入波束:")
        print(f"    输出波束内预测最大值索引(局部): {pred_local_indices_B[0]}")
        print(f"    输出波束内真实最大值索引(局部): {target_local_max_idx_B}")
        print(f"    映射后的全局索引: {pred_global_indices_B[0]}")
        print(f"    真实最大值全局索引: {target_global_max_idx_B}")
        print(f"    前3预测索引: {pred_global_indices_B[:3]}")
        
        # 检查是否相同
        if target_max_idx_A == target_global_max_idx_B:
            print(f"  ⚠️  警告: 两种方法的真实最优索引相同!")
        else:
            print(f"  ✓ 两种方法的真实最优索引不同")
        
        # 计算Top-1到Top-3的匹配情况
        for n in [1, 2, 3]:
            match_A = target_max_idx_A in pred_indices_A[:n]
            match_B = target_global_max_idx_B in pred_global_indices_B[:n]
            
            print(f"  Top-{n}匹配: 方法A={'✓' if match_A else '✗'}, 方法B={'✓' if match_B else '✗'}")
            
            if match_A != match_B:
                print(f"    ⚠️  发现差异!")

def check_data_consistency():
    """检查数据一致性问题"""
    print("\n=== 数据一致性检查 ===")
    
    output_dir = 'pqc_reup_v1_output'
    
    # 检查多个epoch的数据
    eval_files = [f for f in os.listdir(output_dir) if f.startswith('evaluation_results_epoch_') and f.endswith('.json')]
    eval_files.sort(key=lambda x: int(x.split('_')[3].split('.')[0]))
    
    if len(eval_files) < 2:
        print("数据不足，无法进行一致性检查")
        return
    
    # 比较相邻epoch的数据
    for i in range(len(eval_files)-1):
        curr_file = eval_files[i]
        next_file = eval_files[i+1]
        
        curr_epoch = int(curr_file.split('_')[3].split('.')[0])
        next_epoch = int(next_file.split('_')[3].split('.')[0])
        
        print(f"\n比较 Epoch {curr_epoch} -> Epoch {next_epoch}")
        
        # 加载数据
        with open(os.path.join(output_dir, curr_file), 'r') as f:
            curr_data = json.load(f)
        with open(os.path.join(output_dir, next_file), 'r') as f:
            next_data = json.load(f)
        
        # 检查预测数据是否相同
        curr_pred = np.array(curr_data['predictions'])
        next_pred = np.array(next_data['predictions'])
        
        pred_equal = np.allclose(curr_pred, next_pred)
        print(f"  预测数据相同: {pred_equal}")
        
        if pred_equal:
            print("  ⚠️  警告: 相邻epoch的预测数据完全相同，可能是模型未正确更新!")

if __name__ == "__main__":
    analyze_top_n_issue()
    check_data_consistency()