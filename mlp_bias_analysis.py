"""
MLP偏置设置分析
MLP Bias Configuration Analysis
"""

import torch
import torch.nn as nn

def analyze_mlp_bias_configuration():
    """分析MLP中的偏置设置"""
    print("=== MLP偏置配置分析 ===")
    print("=" * 50)
    
    # 1. PyTorch Linear层默认行为
    print("1. PyTorch Linear层默认行为:")
    print("   默认情况下，nn.Linear(bias=True)")
    print("   即默认包含偏置项")
    print()
    
    # 2. 代码中的MLP结构分析
    print("2. 本代码中的MLP结构:")
    
    # 重现代码中的经典回归器
    n_qubits = 8
    output_dim = 256
    
    classical_regressor = nn.Sequential(
        nn.Linear(n_qubits, 64),    # 第一层: 8 → 64
        nn.ReLU(),
        nn.Linear(64, output_dim)   # 第二层: 64 → 256
    )
    
    print("   网络结构:")
    total_params = 0
    layer_details = []
    
    for i, layer in enumerate(classical_regressor):
        if isinstance(layer, nn.Linear):
            in_features = layer.in_features
            out_features = layer.out_features
            has_bias = layer.bias is not None
            
            # 计算参数数量
            weight_params = in_features * out_features
            bias_params = out_features if has_bias else 0
            layer_total = weight_params + bias_params
            
            total_params += layer_total
            
            layer_info = {
                'layer_num': i,
                'type': 'Linear',
                'in_features': in_features,
                'out_features': out_features,
                'has_bias': has_bias,
                'weight_params': weight_params,
                'bias_params': bias_params,
                'total_params': layer_total
            }
            layer_details.append(layer_info)
            
            print(f"     层 {i}: Linear({in_features}, {out_features})")
            print(f"       权重矩阵: {in_features} × {out_features}")
            print(f"       是否包含偏置: {has_bias}")
            print(f"       参数数量: {weight_params}(权重) + {bias_params}(偏置) = {layer_total}")
            print()
    
    # 3. 总体参数统计
    print("3. 总体参数统计:")
    print(f"   总参数数: {total_params}")
    
    total_weight_params = sum(layer['weight_params'] for layer in layer_details)
    total_bias_params = sum(layer['bias_params'] for layer in layer_details)
    
    print(f"   权重参数: {total_weight_params}")
    print(f"   偏置参数: {total_bias_params}")
    print(f"   偏置占比: {total_bias_params/total_params*100:.1f}%")
    print()
    
    # 4. 数学表达式
    print("4. 数学表达式:")
    print("   对于每一层 Linear(in_features, out_features):")
    print("   输出 = W × 输入 + b")
    print("   其中:")
    print("   - W: 权重矩阵 (in_features × out_features)")
    print("   - b: 偏置向量 (out_features)")
    print("   - 默认情况下 b ≠ 0")
    print()
    
    # 5. 偏置的作用和影响
    print("5. 偏置的作用:")
    print("   ✓ 允许模型学习数据的偏移量")
    print("   ✓ 提高模型的表达能力")
    print("   ✓ 使决策边界不强制通过原点")
    print("   ✓ 通常能改善模型性能")
    print()
    
    # 6. 如果要去掉偏置的修改方式
    print("6. 如何显式去掉偏置:")
    print("   修改方式:")
    print("   # 原代码:")
    print("   nn.Linear(8, 64),")
    print("   nn.Linear(64, 256)")
    print()
    print("   # 修改后 (去掉偏置):")
    print("   nn.Linear(8, 64, bias=False),")
    print("   nn.Linear(64, 256, bias=False)")
    print()
    
    # 7. 对比分析
    print("7. 有无偏置的对比:")
    
    # 有偏置的情况
    with_bias_params = total_params
    print(f"   有偏置: {with_bias_params} 参数")
    
    # 无偏置的情况
    without_bias_params = total_weight_params  # 只有权重参数
    print(f"   无偏置: {without_bias_params} 参数")
    print(f"   参数减少: {with_bias_params - without_bias_params} ({(with_bias_params - without_bias_params)/with_bias_params*100:.1f}%)")
    print()
    
    # 8. 实际验证
    print("8. 实际验证:")
    
    # 创建带偏置的层
    layer_with_bias = nn.Linear(8, 64)
    print(f"   带偏置层参数数: {sum(p.numel() for p in layer_with_bias.parameters())}")
    
    # 创建不带偏置的层
    layer_without_bias = nn.Linear(8, 64, bias=False)
    print(f"   不带偏置层参数数: {sum(p.numel() for p in layer_without_bias.parameters())}")
    
    print()
    print("=" * 50)
    print("结论: 本代码的MLP默认包含偏置项")

if __name__ == "__main__":
    analyze_mlp_bias_configuration()