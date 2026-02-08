"tch""
MSE计算原理详解
Detailed Explanation of MSE Calculation
"""

import torch
import torch.nn as nn
import numpy as np

def explain_mse_calculation():
    """详细解释MSE的计算方式"""
    print("=== MSE (Mean Squared Error) 计算原理详解 ===")
    print("=" * 60)
    
    # 模拟参数
    batch_size = 32
    output_dimension = 256
    
    print(f"模拟参数:")
    print(f"  Batch大小: {batch_size}")
    print(f"  输出维度: {output_dimension}")
    print(f"  每个batch总元素数: {batch_size * output_dimension}")
    print()
    
    # 创建模拟数据
    print("1. 创建模拟数据:")
    targets = torch.randn(batch_size, output_dimension)
    predictions = torch.randn(batch_size, output_dimension)
    
    print(f"   真实值形状: {targets.shape}")
    print(f"   预测值形状: {predictions.shape}")
    print()
    
    # PyTorch MSELoss计算
    print("2. PyTorch MSELoss计算:")
    criterion = nn.MSELoss()
    pytorch_mse = criterion(predictions, targets)
    print(f"   PyTorch MSELoss结果: {pytorch_mse.item():.6f}")
    print()
    
    # 手动计算验证
    print("3. 手动计算验证:")
    
    # 计算差值平方
    differences = predictions - targets
    squared_differences = differences ** 2
    
    print("   计算步骤:")
    print("   a) 计算差值: (predictions - targets)")
    print("   b) 计算平方: (differences)²")
    print("   c) 求平均值: mean(squared_differences)")
    print()
    
    # 详细计算过程
    sum_of_squares = torch.sum(squared_differences)
    total_elements = batch_size * output_dimension
    manual_mse = sum_of_squares / total_elements
    
    print(f"   详细计算:")
    print(f"   - 差值平方和: {sum_of_squares.item():.6f}")
    print(f"   - 总元素数: {total_elements}")
    print(f"   - 平均值: {sum_of_squares.item():.6f} / {total_elements} = {manual_mse.item():.6f}")
    print()
    
    # 验证一致性
    is_consistent = torch.allclose(pytorch_mse, manual_mse)
    print(f"4. 计算一致性验证: {is_consistent}")
    print(f"   PyTorch结果: {pytorch_mse.item():.6f}")
    print(f"   手动计算: {manual_mse.item():.6f}")
    print(f"   差异: {abs(pytorch_mse.item() - manual_mse.item()):.10f}")
    print()
    
    # 具体示例
    print("5. 具体计算示例 (前2个样本的前3个输出):")
    print("   真实值:")
    print(f"   {targets[:2, :3]}")
    print("   预测值:")
    print(f"   {predictions[:2, :3]}")
    print("   差值:")
    print(f"   {differences[:2, :3]}")
    print("   差值平方:")
    print(f"   {squared_differences[:2, :3]}")
    print()
    
    # 统计信息
    print("6. 统计信息:")
    print(f"   最大差值平方: {torch.max(squared_differences).item():.6f}")
    print(f"   最小差值平方: {torch.min(squared_differences).item():.6f}")
    print(f"   平均差值平方: {torch.mean(squared_differences).item():.6f}")
    print(f"   标准差: {torch.std(squared_differences).item():.6f}")
    print()
    
    # 与数学公式对应
    print("7. 数学公式对应:")
    print("   MSE = (1/n) × Σ(yᵢ - ŷᵢ)²")
    print(f"   其中 n = {total_elements} (batch_size × output_dimension)")
    print(f"   Σ(yᵢ - ŷᵢ)² = {sum_of_squares.item():.6f}")
    print(f"   MSE = {sum_of_squares.item():.6f} / {total_elements} = {manual_mse.item():.6f}")
    print()
    
    # 不同视角的理解
    print("8. 不同理解视角:")
    print("   视角1 - 元素级别:")
    print("     对所有 batch_size × output_dimension 个元素计算平均平方误差")
    print()
    print("   视角2 - 样本级别:")  
    print("     先计算每个样本的MSE，再对batch中所有样本求平均")
    print()
    
    # 验证样本级别计算
    sample_mses = torch.mean((predictions - targets) ** 2, dim=1)  # 每个样本的MSE
    batch_mse_from_samples = torch.mean(sample_mses)
    
    print("   样本级别验证:")
    print(f"   每个样本MSE: {sample_mses[:3]} (显示前3个)")
    print(f"   样本MSE平均值: {batch_mse_from_samples.item():.6f}")
    print(f"   与整体MSE一致性: {torch.allclose(manual_mse, batch_mse_from_samples)}")
    print()
    
    print("=" * 60)
    print("总结: PyTorch的MSELoss是对所有元素的差值平方求平均")
    print("即: MSE = sum((predictions - targets)²) / (batch_size × output_dimension)")

if __name__ == "__main__":
    explain_mse_calculation()