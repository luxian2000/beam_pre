"""
简化版PQC波束预测模型测试
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler
import h5py

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SimpleQuantumModel(nn.Module):
    def __init__(self, n_qubits=8, n_layers=3):
        super(SimpleQuantumModel, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # 量子设备
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # 量子权重参数
        self.quantum_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 2) * 0.1
        )
        
        # 创建量子电路
        self.qnode = self.create_quantum_circuit()
        
        # 经典MLP
        self.classical_net = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, 256)
        )
    
    def create_quantum_circuit(self):
        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            # 角度编码
            for i in range(min(len(inputs), self.n_qubits)):
                qml.RY(inputs[i] * np.pi, wires=i)
            
            # 补零
            for i in range(len(inputs), self.n_qubits):
                qml.RY(0.0, wires=i)
            
            # 参数化层
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # 纠缠
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            
            # 测量
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return circuit
    
    def forward(self, x):
        batch_size = x.shape[0]
        quantum_outputs = []
        
        for i in range(batch_size):
            # 归一化
            x_norm = (x[i] - x[i].min()) / (x[i].max() - x[i].min() + 1e-8)
            # 量子计算
            q_out = self.qnode(x_norm, self.quantum_weights)
            quantum_outputs.append(torch.stack(q_out))
        
        quantum_outputs = torch.stack(quantum_outputs).float()
        return self.classical_net(quantum_outputs)

def load_sample_data():
    """加载小样本数据进行测试"""
    data_path = '/Users/luxian/DataSpace/beam_pre/sls_beam_data_spatial_domain_vivo.mat'
    
    with h5py.File(data_path, 'r') as f:
        rsrp_data = f['rsrp'][:1000, :]  # 只取1000个样本进行测试
    
    # 随机选择8个输入特征
    input_indices = np.random.choice(256, 8, replace=False)
    
    X = rsrp_data[:, input_indices]
    y = rsrp_data
    
    # 归一化
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return torch.FloatTensor(X_scaled), torch.FloatTensor(y), input_indices

def main():
    print("=== PQC波束预测简化测试 ===")
    
    # 加载数据
    print("加载数据...")
    X, y, indices = load_sample_data()
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"输入特征索引: {indices}")
    
    # 创建模型
    print("创建模型...")
    model = SimpleQuantumModel(n_qubits=8, n_layers=2).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 测试前向传播
    print("测试前向传播...")
    with torch.no_grad():
        sample_input = X[:5].to(device)  # 取5个样本测试
        output = model(sample_input)
        print(f"输入形状: {sample_input.shape}")
        print(f"输出形状: {output.shape}")
        print("前向传播测试成功!")
    
    # 简单训练测试
    print("进行简单训练测试...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 取小批次数据
    train_X = X[:100].to(device)
    train_y = y[:100].to(device)
    
    model.train()
    for epoch in range(3):  # 只训练3个epoch测试
        optimizer.zero_grad()
        pred = model(train_X)
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
    
    print("简化测试完成!")

if __name__ == "__main__":
    main()