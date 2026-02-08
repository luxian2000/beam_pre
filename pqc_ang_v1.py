"""
Parameterized Quantum Circuit for Beam Prediction (Version 1)
基于参数化量子电路的波束预测模型（版本1）

根据pqc_ang_v1.md算法描述实现的量子-经典混合模型
使用PennyLane + PyTorch实现
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import h5py
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from pqc_config import PQCConfig

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class QuantumClassicalHybridModel(nn.Module):
    """量子经典混合模型"""
    
    def __init__(self, n_qubits=8, n_layers=3, input_dim=8, output_dim=256):
        super(QuantumClassicalHybridModel, self).__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 量子设备设置
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # 量子电路参数（可训练）
        self.quantum_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 2) * 0.1
        )
        
        # 创建量子电路
        self.quantum_circuit = self._create_quantum_circuit()
        
        # 经典MLP回归器
        self.classical_regressor = nn.Sequential(
            nn.Linear(n_qubits, 64),  # 量子输出8维到64维
            nn.ReLU(),
            nn.Linear(64, output_dim)  # 64维到256维输出
        )
        
    def _create_quantum_circuit(self):
        """创建参数化量子电路"""
        
        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            # 角度编码：将输入特征编码为量子比特的旋转角度
            for i in range(min(self.n_qubits, len(inputs))):
                # RY门角度编码
                qml.RY(inputs[i] * np.pi, wires=i)
            
            # 补零处理
            for i in range(len(inputs), self.n_qubits):
                qml.RY(0.0, wires=i)
            
            # Strongly Entangling Layers
            for layer in range(self.n_layers):
                # 旋转层
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # 纠缠层
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            
            # 测量Pauli-Z期望值
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return circuit
    
    def forward(self, x):
        """前向传播"""
        batch_size = x.shape[0]
        quantum_outputs = []
        
        # 确保输入数据类型为float32
        x = x.float()
        
        # 对每个样本执行量子电路
        for i in range(batch_size):
            # 归一化输入到[0,1]范围
            normalized_input = (x[i] - x[i].min()) / (x[i].max() - x[i].min() + 1e-8)
            # 执行量子电路
            quantum_output = self.quantum_circuit(normalized_input, self.quantum_weights)
            quantum_outputs.append(torch.stack(quantum_output))
        
        # 堆叠量子输出并确保数据类型一致
        quantum_outputs = torch.stack(quantum_outputs).float()
        
        # 调试信息
        # print(f"Quantum outputs shape: {quantum_outputs.shape}")
        # print(f"Quantum outputs dtype: {quantum_outputs.dtype}")
        
        # 经典回归
        final_output = self.classical_regressor(quantum_outputs)
        
        return final_output

class BeamDataProcessor:
    """波束数据处理器"""
    
    def __init__(self, data_path, n_observed=8):
        self.data_path = data_path
        self.n_observed = n_observed
        self.scaler = MinMaxScaler()
        
    def load_and_prepare_data(self, n_samples=336000):
        """加载并准备训练数据"""
        print("加载数据文件...")
        with h5py.File(self.data_path, 'r') as f:
            # 读取RSRP数据 (总共336000个样本)
            rsrp_data = f['rsrp'][:n_samples, :]  # 取前n_samples个样本
            
        print(f"数据形状: {rsrp_data.shape}")
        print(f"总样本数: {rsrp_data.shape[0]}")
        print(f"每个样本特征数: {rsrp_data.shape[1]}")
        
        # 准备输入输出对
        X_list = []
        y_list = []
        
        # 随机选择8个固定位置作为输入特征
        input_indices = np.random.choice(256, self.n_observed, replace=False)
        print(f"选择的输入特征索引: {input_indices}")
        
        for i in range(len(rsrp_data)):
            full_vector = rsrp_data[i]
            
            # 输入：选定的8个特征
            input_features = full_vector[input_indices]
            # 输出：完整的256维向量
            output_features = full_vector
            
            X_list.append(input_features)
            y_list.append(output_features)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 数据归一化
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"输入特征形状: {X_scaled.shape}")
        print(f"输出特征形状: {y.shape}")
        
        return X_scaled, y, input_indices

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    """训练模型"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    print("开始训练...")
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 确保数据类型一致
            data = data.float()
            target = target.float()
            
            optimizer.zero_grad()
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            train_loss += loss.item()
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.float()
                target = target.float()
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # 早停检查
        if epoch > 10 and avg_val_loss > val_losses[-2]:
            print("验证损失开始上升，提前停止训练")
            break
    
    return train_losses, val_losses

def evaluate_model(model, test_loader):
    """评估模型性能"""
    model.eval()
    criterion = nn.MSELoss()
    
    test_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            predictions.extend(output.numpy())
            targets.extend(target.numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    
    # 计算其他评估指标
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    
    # 计算R²分数
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'test_loss': avg_test_loss,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'targets': targets
    }

def plot_training_history(train_losses, val_losses, save_path=None):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions_vs_actual(results, sample_idx=0, save_path=None):
    """绘制预测值vs实际值对比"""
    predictions = results['predictions']
    targets = results['targets']
    
    plt.figure(figsize=(12, 5))
    
    # 绘制某个样本的预测对比
    plt.subplot(1, 2, 1)
    plt.plot(targets[sample_idx], 'b-', label='Actual', alpha=0.7)
    plt.plot(predictions[sample_idx], 'r--', label='Predicted', alpha=0.7)
    plt.xlabel('Beam Index')
    plt.ylabel('RSRP Value')
    plt.title(f'Prediction vs Actual (Sample {sample_idx})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制误差分布
    plt.subplot(1, 2, 2)
    errors = predictions[sample_idx] - targets[sample_idx]
    plt.hist(errors, bins=50, alpha=0.7, color='green')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    # 打印配置信息
    PQCConfig.print_config()
    
    # 创建输出目录
    if not os.path.exists(PQCConfig.OUTPUT_DIR):
        os.makedirs(PQCConfig.OUTPUT_DIR)
        print(f"创建输出目录: {PQCConfig.OUTPUT_DIR}")
    
    print("=" * 60)
    print("PQC波束预测模型训练")
    print("=" * 60)
    
    # 1. 数据准备
    print("\n[步骤1] 数据准备")
    processor = BeamDataProcessor(PQCConfig.DATA_PATH)
    X, y, input_indices = processor.load_and_prepare_data(PQCConfig.TOTAL_SAMPLES)
    
    # 划分训练集、验证集和测试集
    train_size, val_size, test_size = PQCConfig.get_data_splits()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        train_size=val_size, 
        test_size=test_size,
        random_state=42
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 调整批次大小以适应数据集大小
    effective_batch_size = min(PQCConfig.BATCH_SIZE, len(X_train) // 20)  # 确保至少20个批次
    print(f"实际使用的批次大小: {effective_batch_size}")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=effective_batch_size)
    
    # 2. 模型初始化
    print("\n[步骤2] 模型初始化")
    model = QuantumClassicalHybridModel(
        n_qubits=PQCConfig.N_QUBITS,
        n_layers=PQCConfig.N_LAYERS,
        input_dim=PQCConfig.INPUT_FEATURES,
        output_dim=PQCConfig.OUTPUT_DIM
    )
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 3. 模型训练
    print("\n[步骤3] 模型训练")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        epochs=PQCConfig.EPOCHS, lr=PQCConfig.LEARNING_RATE
    )
    
    # 4. 模型评估
    print("\n[步骤4] 模型评估")
    results = evaluate_model(model, test_loader)
    
    print(f"\n测试结果:")
    print(f"MSE: {results['mse']:.6f}")
    print(f"RMSE: {results['rmse']:.6f}")
    print(f"R²: {results['r2']:.6f}")
    
    # 5. 保存结果
    print("\n[步骤5] 保存结果")
    
    # 保存模型
    model_path = os.path.join(PQCConfig.OUTPUT_DIR, 'pqc_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")
    
    # 保存训练历史
    history_path = os.path.join(PQCConfig.OUTPUT_DIR, 'training_history.json')
    history_data = {
        'train_losses': [float(loss) for loss in train_losses],
        'val_losses': [float(loss) for loss in val_losses],
        'final_epoch': len(train_losses),
        'config': {
            'total_samples': PQCConfig.TOTAL_SAMPLES,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'n_qubits': PQCConfig.N_QUBITS,
            'n_layers': PQCConfig.N_LAYERS,
            'batch_size': effective_batch_size,
            'epochs': PQCConfig.EPOCHS,
            'learning_rate': PQCConfig.LEARNING_RATE
        }
    }
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, indent=2, ensure_ascii=False)
    print(f"训练历史已保存到: {history_path}")
    
    # 保存评估结果
    results_path = os.path.join(PQCConfig.OUTPUT_DIR, 'evaluation_results.json')
    results_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'test_metrics': {
            'mse': float(results['mse']),
            'rmse': float(results['rmse']),
            'r2': float(results['r2'])
        },
        'data_config': {
            'total_samples': PQCConfig.TOTAL_SAMPLES,
            'n_observed': PQCConfig.INPUT_FEATURES,
            'input_indices': input_indices.tolist(),
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size
        }
    }
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"评估结果已保存到: {results_path}")
    
    # 6. 生成可视化图表
    print("\n[步骤6] 生成可视化图表")
    
    # 训练历史图
    history_plot_path = os.path.join(PQCConfig.OUTPUT_DIR, 'training_history.png')
    plot_training_history(train_losses, val_losses, history_plot_path)
    print(f"训练历史图已保存到: {history_plot_path}")
    
    # 预测对比图
    prediction_plot_path = os.path.join(PQCConfig.OUTPUT_DIR, 'predictions_vs_actual.png')
    plot_predictions_vs_actual(results, sample_idx=0, save_path=prediction_plot_path)
    print(f"预测对比图已保存到: {prediction_plot_path}")
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()