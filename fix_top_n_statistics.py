#!/usr/bin/env python3
"""
修复Top-N统计中方法A和B数据相同的问题
重新计算正确的Top-N准确率统计数据
"""

import json
import numpy as np
import os
from collections import defaultdict

def calculate_top_n_accuracy_both_methods(predictions, targets, input_indices, top_n_max=10):
    """计算两种Top-N准确率：包含输入波束和不包含输入波束"""
    n_samples = len(predictions)
    
    # 初始化两种方法的准确率计数器
    top_n_correct_with_input = [0] * top_n_max    # 方法A：包含输入波束
    top_n_correct_without_input = [0] * top_n_max # 方法B：不包含输入波束
    
    # 创建输出波束索引集合（排除输入波束）
    all_indices = set(range(len(predictions[0])))  # 所有波束索引
    output_indices_set = all_indices - set(input_indices)  # 排除输入波束后的索引
    
    print(f"总波束数: {len(all_indices)}")
    print(f"输入波束数: {len(input_indices)}")
    print(f"输出波束数: {len(output_indices_set)}")
    
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

def fix_top_n_statistics():
    """修复Top-N统计问题"""
    output_dir = 'pqc_reup_v1_output'
    
    # 查找所有评估结果文件
    eval_files = [f for f in os.listdir(output_dir) if f.startswith('evaluation_results_epoch_') and f.endswith('.json')]
    
    if not eval_files:
        print("未找到评估结果文件")
        return
    
    # 按epoch排序
    eval_files.sort(key=lambda x: int(x.split('_')[3].split('.')[0]))
    
    print("=== 修复Top-N统计问题 ===")
    
    # 获取输入索引配置
    input_indices = None
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
                    print(f"输入索引配置: {input_indices}")
        except Exception as e:
            print(f"读取配置文件失败: {e}")
    
    if input_indices is None:
        print("无法获取输入索引配置，使用默认值 [0,1,2,...,11]")
        input_indices = list(range(12))
    
    # 修复每个epoch的统计数据
    fixed_stats = {}
    
    for eval_file in eval_files:
        eval_path = os.path.join(output_dir, eval_file)
        epoch_num = int(eval_file.split('_')[3].split('.')[0])
        
        print(f"\n--- 修复 Epoch {epoch_num} ---")
        
        try:
            with open(eval_path, 'r') as f:
                data = json.load(f)
            
            if 'predictions' in data and 'targets' in data:
                predictions = np.array(data['predictions'])
                targets = np.array(data['targets'])
                
                print(f"数据形状: 预测{predictions.shape}, 真实{targets.shape}")
                
                # 重新计算Top-N准确率
                top_n_results = calculate_top_n_accuracy_both_methods(
                    predictions, targets, input_indices, top_n_max=10
                )
                
                # 检查原始数据
                original_top_n = data.get('top_n_accuracies', {})
                if original_top_n:
                    print("原始Top-N数据:")
                    if isinstance(original_top_n, list):
                        print(f"  单一方法: {[f'{x:.6f}' for x in original_top_n[:5]]}")
                    elif isinstance(original_top_n, dict):
                        print(f"  方法A: {[f'{x:.6f}' for x in original_top_n.get('with_input', [])[:5]]}")
                        print(f"  方法B: {[f'{x:.6f}' for x in original_top_n.get('without_input', [])[:5]]}")
                
                print("修复后的Top-N数据:")
                print(f"  方法A (包含输入): {[f'{x:.6f}' for x in top_n_results['with_input'][:5]]}")
                print(f"  方法B (排除输入): {[f'{x:.6f}' for x in top_n_results['without_input'][:5]]}")
                
                # 检查是否相同
                are_identical = all(a == b for a, b in zip(
                    top_n_results['with_input'], top_n_results['without_input']
                ))
                print(f"  两种方法数据相同: {are_identical}")
                
                if are_identical:
                    print("  ⚠️  警告: 修复后两种方法仍然相同，可能存在根本问题!")
                    
                    # 详细分析问题
                    analyze_identical_issue(predictions, targets, input_indices)
                else:
                    print("  ✓ 修复成功: 两种方法数据已区分")
                
                # 更新数据
                data['top_n_accuracies'] = top_n_results
                fixed_stats[epoch_num] = top_n_results
                
                # 保存修复后的文件
                backup_path = eval_path.replace('.json', '_backup.json')
                os.rename(eval_path, backup_path)
                print(f"  备份原文件到: {backup_path}")
                
                with open(eval_path, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"  保存修复文件到: {eval_path}")
                
            else:
                print("  缺少预测或目标数据")
                
        except Exception as e:
            print(f"  处理失败: {e}")
    
    # 生成修复报告
    generate_fix_report(fixed_stats, input_indices)

def analyze_identical_issue(predictions, targets, input_indices):
    """分析为什么两种方法结果相同"""
    print("  详细分析相同原因:")
    
    # 检查几个样本
    n_samples = min(3, len(predictions))
    all_indices = set(range(len(predictions[0])))
    output_indices_set = all_indices - set(input_indices)
    
    identical_count = 0
    
    for i in range(n_samples):
        pred_sample = predictions[i]
        target_sample = targets[i]
        
        target_max_idx_A = np.argmax(target_sample)
        target_values_B = target_sample[list(output_indices_set)]
        target_local_max_idx_B = np.argmax(target_values_B)
        output_indices_list = list(output_indices_set)
        target_global_max_idx_B = output_indices_list[target_local_max_idx_B]
        
        if target_max_idx_A == target_global_max_idx_B:
            identical_count += 1
            print(f"    样本{i}: 真实最优索引相同 ({target_max_idx_A})")
        else:
            print(f"    样本{i}: 真实最优索引不同 (A:{target_max_idx_A}, B:{target_global_max_idx_B})")
    
    print(f"  前{n_samples}个样本中，真实最优索引相同的样本数: {identical_count}/{n_samples}")
    
    # 检查输入波束的值分布
    input_values = []
    output_values = []
    
    for i in range(min(100, len(predictions))):
        for idx in input_indices:
            input_values.append(targets[i][idx])
        for idx in output_indices_set:
            output_values.append(targets[i][idx])
    
    print(f"  输入波束值范围: [{min(input_values):.3f}, {max(input_values):.3f}]")
    print(f"  输出波束值范围: [{min(output_values):.3f}, {max(output_values):.3f}]")
    
    if max(input_values) > max(output_values):
        print("  ⚠️  输入波束包含最大值，可能导致方法A和B结果相似")

def generate_fix_report(fixed_stats, input_indices):
    """生成修复报告"""
    print("\n=== 修复报告 ===")
    
    if not fixed_stats:
        print("没有成功修复的数据")
        return
    
    print(f"输入索引: {input_indices}")
    print(f"修复的epoch数量: {len(fixed_stats)}")
    
    print("\n各epoch修复结果:")
    for epoch, stats in sorted(fixed_stats.items()):
        method_a_avg = np.mean(stats['with_input'])
        method_b_avg = np.mean(stats['without_input'])
        difference = abs(method_a_avg - method_b_avg)
        
        print(f"  Epoch {epoch}:")
        print(f"    方法A平均: {method_a_avg:.6f}")
        print(f"    方法B平均: {method_b_avg:.6f}")
        print(f"    差异: {difference:.6f}")
        print(f"    区分成功: {'是' if difference > 1e-6 else '否'}")

if __name__ == "__main__":
    fix_top_n_statistics()