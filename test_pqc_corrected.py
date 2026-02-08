"""
测试修正后的PQC配置
"""

import numpy as np
import torch
from pqc_config import PQCConfig
import h5py

def test_data_loading():
    """测试数据加载"""
    print("=== 测试数据加载 ===")
    
    # 检查数据文件
    try:
        with h5py.File(PQCConfig.DATA_PATH, 'r') as f:
            print(f"数据文件键值: {list(f.keys())}")
            rsrp_shape = f['rsrp'].shape
            print(f"RSRP数据形状: {rsrp_shape}")
            
            # 验证总样本数
            assert rsrp_shape[0] == PQCConfig.TOTAL_SAMPLES, f"样本数不匹配: {rsrp_shape[0]} != {PQCConfig.TOTAL_SAMPLES}"
            print("✓ 数据样本数验证通过")
            
            # 验证特征维度
            assert rsrp_shape[1] == PQCConfig.OUTPUT_DIM, f"特征维度不匹配: {rsrp_shape[1]} != {PQCConfig.OUTPUT_DIM}"
            print("✓ 数据特征维度验证通过")
            
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False
    
    return True

def test_config_consistency():
    """测试配置一致性"""
    print("\n=== 测试配置一致性 ===")
    
    # 检查数据划分
    train_size, val_size, test_size = PQCConfig.get_data_splits()
    total_calculated = train_size + val_size + test_size
    
    print(f"配置的总样本数: {PQCConfig.TOTAL_SAMPLES}")
    print(f"计算的总样本数: {total_calculated}")
    print(f"训练集: {train_size} ({train_size/PQCConfig.TOTAL_SAMPLES*100:.1f}%)")
    print(f"验证集: {val_size} ({val_size/PQCConfig.TOTAL_SAMPLES*100:.1f}%)")
    print(f"测试集: {test_size} ({test_size/PQCConfig.TOTAL_SAMPLES*100:.1f}%)")
    
    assert total_calculated == PQCConfig.TOTAL_SAMPLES, "数据划分不一致"
    print("✓ 配置一致性验证通过")
    
    return True

def test_model_initialization():
    """测试模型初始化"""
    print("\n=== 测试模型初始化 ===")
    
    try:
        # 导入模型类
        from pqc_ang_v1 import QuantumClassicalHybridModel
        
        # 创建小规模模型进行测试
        model = QuantumClassicalHybridModel(
            n_qubits=4,  # 使用较少量子比特进行测试
            n_layers=2,
            input_dim=PQCConfig.INPUT_FEATURES,
            output_dim=64  # 使用较小输出维度进行测试
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"测试模型参数数量: {param_count}")
        print("✓ 模型初始化成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        return False

def main():
    """主测试函数"""
    print("PQC配置修正验证测试")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # 测试1: 数据加载
    if test_data_loading():
        tests_passed += 1
    
    # 测试2: 配置一致性
    if test_config_consistency():
        tests_passed += 1
    
    # 测试3: 模型初始化
    if test_model_initialization():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"测试结果: {tests_passed}/{total_tests} 通过")
    
    if tests_passed == total_tests:
        print("✓ 所有测试通过！配置修正成功")
        print("\n主要修正内容:")
        print("- 总样本数从36000修正为336000")
        print("- 更新了数据集划分比例")
        print("- 调整了批次大小计算逻辑")
        print("- 完善了配置管理系统")
    else:
        print("✗ 部分测试失败，请检查配置")

if __name__ == "__main__":
    main()