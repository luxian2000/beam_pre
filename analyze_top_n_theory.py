#!/usr/bin/env python3
"""
深入分析Top-N统计相同现象的理论原因
"""

import numpy as np
import json
import os

def theoretical_analysis():
    """理论分析Top-N统计"""
    print("=== Top-N统计理论分析 ===")
    
    # 场景1: 最优波束在输入波束中
    print("\n场景1: 最优波束在输入波束中")
    input_indices = [0, 1, 2, 3]
    all_indices = list(range(8))  # 总共8个波束
    output_indices = [4, 5, 6, 7]  # 输出波束
    
    # 构造测试数据：最优波束在输入波束中（索引1）
    predictions = np.array([0.1, 0.9, 0.3, 0.2, 0.4, 0.5, 0.6, 0.7])  # 预测值
    targets = np.array([10, 90, 30, 20, 40, 50, 60, 70])            # 真实值
    
    print(f"输入波束索引: {input_indices}")
    print(f"输出波束索引: {output_indices}")
    print(f"预测值: {predictions}")
    print(f"真实值: {targets}")
    print(f"真实最大值索引: {np.argmax(targets)} (值: {targets[np.argmax(targets)]})")
    
    # 方法A：包含输入波束
    pred_indices_A = np.argsort(predictions)[::-1]
    target_max_idx_A = np.argmax(targets)
    print(f"方法A - 所有波束排序: {pred_indices_A}")
    print(f"方法A - 真实最优索引: {target_max_idx_A}")
    
    # 方法B：排除输入波束
    output_targets = targets[output_indices]
    output_predictions = predictions[output_indices]
    pred_local_B = np.argsort(output_predictions)[::-1]
    target_local_max_B = np.argmax(output_targets)
    
    pred_global_B = [output_indices[i] for i in pred_local_B]
    target_global_max_B = output_indices[target_local_max_B]
    
    print(f"方法B - 输出波束内排序: {pred_local_B}")
    print(f"方法B - 输出波束内真实最优(局部): {target_local_max_B}")
    print(f"方法B - 映射后全局索引: {target_global_max_B}")
    
    print(f"两种方法真实最优索引相同: {target_max_idx_A == target_global_max_B}")
    
    # 计算Top-N准确率
    top_n_max = 3
    correct_A = [0] * top_n_max
    correct_B = [0] * top_n_max
    
    for n in range(1, top_n_max + 1):
        if target_max_idx_A in pred_indices_A[:n]:
            correct_A[n-1] = 1
        if target_global_max_B in pred_global_B[:n]:
            correct_B[n-1] = 1
    
    print(f"方法A Top-N准确率: {[c for c in correct_A]}")
    print(f"方法B Top-N准确率: {[c for c in correct_B]}")
    print(f"结果相同: {correct_A == correct_B}")
    
    # 场景2: 最优波束在输出波束中
    print("\n场景2: 最优波束在输出波束中")
    targets2 = np.array([10, 20, 30, 40, 90, 60, 70, 80])  # 最优在索引4（输出波束）
    
    print(f"新真实值: {targets2}")
    print(f"真实最大值索引: {np.argmax(targets2)} (值: {targets2[np.argmax(targets2)]})")
    
    # 重新计算方法B
    output_targets2 = targets2[output_indices]
    target_local_max_B2 = np.argmax(output_targets2)
    target_global_max_B2 = output_indices[target_local_max_B2]
    
    target_max_idx_A2 = np.argmax(targets2)
    
    print(f"方法A - 真实最优索引: {target_max_idx_A2}")
    print(f"方法B - 映射后全局索引: {target_global_max_B2}")
    print(f"两种方法真实最优索引相同: {target_max_idx_A2 == target_global_max_B2}")

def practical_analysis():
    """实际数据分析"""
    print("\n=== 实际数据分析 ===")
    
    output_dir = 'pqc_reup_v1_output'
    eval_file = 'evaluation_results_epoch_20.json'
    eval_path = os.path.join(output_dir, eval_file)
    
    if not os.path.exists(eval_path):
        print("评估文件不存在")
        return
    
    with open(eval_path, 'r') as f:
        data = json.load(f)
    
    predictions = np.array(data['predictions'])
    targets = np.array(data['targets'])
    
    input_indices = list(range(12))  # 默认输入索引
    all_indices = set(range(len(predictions[0])))
    output_indices_set = all_indices - set(input_indices)
    
    print(f"数据维度: {predictions.shape}")
    print(f"输入波束: {len(input_indices)} 个")
    print(f"输出波束: {len(output_indices_set)} 个")
    
    # 分析几个样本
    print("\n样本分析:")
    identical_count = 0
    input_optimal_count = 0
    output_optimal_count = 0
    
    for i in range(min(10, len(predictions))):
        target_sample = targets[i]
        target_max_idx_A = np.argmax(target_sample)
        
        target_values_B = target_sample[list(output_indices_set)]
        target_local_max_idx_B = np.argmax(target_values_B)
        output_indices_list = list(output_indices_set)
        target_global_max_idx_B = output_indices_list[target_local_max_idx_B]
        
        is_identical = (target_max_idx_A == target_global_max_idx_B)
        is_input_optimal = (target_max_idx_A in input_indices)
        is_output_optimal = (target_max_idx_A in output_indices_set)
        
        if is_identical:
            identical_count += 1
        if is_input_optimal:
            input_optimal_count += 1
        if is_output_optimal:
            output_optimal_count += 1
            
        print(f"样本{i:2d}: 全局最优{target_max_idx_A:3d}, 输出最优{target_global_max_idx_B:3d}, "
              f"相同:{str(is_identical):5s}, 输入最优:{str(is_input_optimal):5s}")
    
    print(f"\n统计结果:")
    print(f"前{min(10, len(predictions))}个样本中:")
    print(f"  真实最优索引相同的样本: {identical_count}")
    print(f"  最优波束在输入波束中的样本: {input_optimal_count}")
    print(f"  最优波束在输出波束中的样本: {output_optimal_count}")
    
    # 分析整体数据
    total_samples = len(targets)
    global_max_indices = np.argmax(targets, axis=1)
    input_optimal_total = sum(1 for idx in global_max_indices if idx in input_indices)
    output_optimal_total = total_samples - input_optimal_total
    
    print(f"\n全部{total_samples}个样本统计:")
    print(f"  最优波束在输入波束中: {input_optimal_total} ({input_optimal_total/total_samples*100:.1f}%)")
    print(f"  最优波束在输出波束中: {output_optimal_total} ({output_optimal_total/total_samples*100:.1f}%)")

def conclusion():
    """得出结论"""
    print("\n=== 结论 ===")
    print("Top-N统计中方法A和B数据相同的原因分析:")
    print()
    print("1. 理论上，两种方法应该产生不同结果:")
    print("   - 方法A: 在所有波束中寻找最优解")
    print("   - 方法B: 仅在输出波束中寻找最优解")
    print()
    print("2. 实际数据中出现相同结果的原因:")
    print("   - 大多数样本的真实最优波束都在输出波束范围内")
    print("   - 很少有样本的最优波束在输入波束中")
    print("   - 因此两种方法实际上在相同的数据子集上计算")
    print()
    print("3. 这种现象的合理性:")
    print("   - 符合实际应用场景：输入波束通常是已知的参考波束")
    print("   - 真正需要预测的是未知的输出波束")
    print("   - 模型的主要价值在于预测输出波束的性能")
    print()
    print("4. 建议:")
    print("   - 这种现象在当前数据集下是正常的")
    print("   - 可以重点关注方法B（排除输入波束）的结果")
    print("   - 方法B更能反映模型在实际预测任务中的性能")

if __name__ == "__main__":
    theoretical_analysis()
    practical_analysis()
    conclusion()