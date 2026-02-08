"""
基于Pennylane的量子算法波束RSRP预测
Quantum Beam RSRP Prediction using PennyLane
"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim

# 检查Pennylane是否可用
try:
    dev = qml.device("default.qubit", wires=8)
    print("✓ Pennylane环境就绪")
    PENNYLANE_AVAILABLE = True
except Exception as e:
    print(f"✗ Pennylane环境异常: {e}")
    PENNYLANE_AVAILABLE = False

class QuantumBeamPredictor:
    """
    基于Pennylane的量子波束预测器
    Quantum Beam Predictor using PennyLane
    """
    
    def __init__(self, n_wires=8, n_layers=2):
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_wires)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # 创建量子电路
        self.qnode = self._create_quantum_circuit()
        
    def _create_quantum_circuit(self):
        """创建参数化量子电路"""
        
        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            # 数据编码层
            self._data_encoding(inputs)
            
            # 变分层
            self._variational_layer(weights)
            
            # 测量期望值
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]
        
        return circuit
    
    def _data_encoding(self, inputs):
        """角度编码输入数据"""
        # 将输入数据编码到量子态的旋转角度
        for i in range(min(len(inputs), self.n_wires)):
            qml.RY(inputs[i] * np.pi, wires=i)
            qml.RZ(inputs[i] * np.pi/2, wires=i)
    
    def _variational_layer(self, weights):
        """变分量子层"""
        for layer in range(self.n_layers):
            # 旋转门
            for wire in range(self.n_wires):
                qml.RY(weights[layer, wire, 0], wires=wire)
                qml.RZ(weights[layer, wire, 1], wires=wire)
            
            # 纠缠门
            for wire in range(self.n_wires - 1):
                qml.CNOT(wires=[wire, wire + 1])
            qml.CNOT(wires=[self.n_wires - 1, 0])  # 周期性边界条件
    
    def initialize_weights(self):
        """初始化量子电路权重"""
        # 权重形状: (layers, wires, parameters_per_wire)
        shape = (self.n_layers, self.n_wires, 2)
        weights = np.random.uniform(low=-np.pi, high=np.pi, size=shape)
        return torch.tensor(weights, requires_grad=True, dtype=torch.float64)
    
    def extract_quantum_features(self, X):
        """使用量子电路提取特征"""
        if not PENNYLANE_AVAILABLE:
            # 量子不可用时使用经典近似
            return self._classical_approximation(X)
        
        n_samples = X.shape[0]
        quantum_features = []
        
        # 初始化权重
        weights = self.initialize_weights()
        
        for i in range(n_samples):
            try:
                # 获取量子电路输出
                inputs = torch.tensor(X[i], dtype=torch.float64)
                circuit_output = self.qnode(inputs, weights)
                quantum_features.append(circuit_output.detach().numpy())
            except Exception as e:
                print(f"样本 {i} 量子计算出错: {e}")
                # 出错时使用零向量
                quantum_features.append(np.zeros(self.n_wires))
        
        return np.array(quantum_features)
    
    def _classical_approximation(self, X):
        """当量子计算不可用时的经典近似"""
        # 使用傅里叶变换和相关性分析模拟量子特征
        features = []
        for sample in X:
            # FFT特征
            fft_result = np.abs(np.fft.fft(sample))[:self.n_wires]
            # 统计特征
            stats = [np.mean(sample), np.std(sample), np.max(sample), np.min(sample)]
            # 组合特征
            combined = np.concatenate([fft_result[:4], stats])
            # 确保维度正确
            if len(combined) < self.n_wires:
                combined = np.pad(combined, (0, self.n_wires - len(combined)))
            else:
                combined = combined[:self.n_wires]
            features.append(combined)
        return np.array(features)

class QuantumEnhancedRegressor:
    """量子增强回归器"""
    
    def __init__(self, n_wires=8):
        self.quantum_predictor = QuantumBeamPredictor(n_wires=n_wires)
        self.classical_regressor = None
        
    def prepare_training_data(self, rsrp_data, mask_ratio=0.3):
        """
        准备训练数据
        """
        n_samples, n_beams = rsrp_data.shape
        
        # 创建mask
        mask = np.random.random((n_samples, n_beams)) > mask_ratio
        
        # 构造输入输出对
        X_input = []
        y_target = []
        
        for i in range(n_samples):
            # 获取观测到的波束值
            observed_values = rsrp_data[i][mask[i]]
            observed_indices = np.where(mask[i])[0]
            
            if len(observed_values) > 0:
                # 构造输入特征：观测值 + 统计信息
                input_features = []
                
                # 观测值（标准化）
                if len(observed_values) > 0:
                    input_features.extend(observed_values[:8])  # 最多8个观测值
                
                # 统计特征
                input_features.extend([
                    np.mean(observed_values),
                    np.std(observed_values),
                    len(observed_values),  # 观测数量
                    np.max(observed_values) if len(observed_values) > 0 else 0,
                    np.min(observed_values) if len(observed_values) > 0 else 0
                ])
                
                # 位置信息
                if len(observed_indices) > 0:
                    input_features.extend(observed_indices[:3])  # 最多3个位置
                
                # 填充到固定长度
                while len(input_features) < 16:
                    input_features.append(0)
                input_features = input_features[:16]
                
                X_input.append(input_features)
                y_target.append(np.mean(rsrp_data[i]))  # 预测平均RSRP值
        
        return np.array(X_input), np.array(y_target)
    
    def train(self, rsrp_data, test_size=0.2):
        """训练量子增强回归器"""
        print("=== 量子增强波束预测训练 ===")
        
        # 准备数据
        print("1. 数据准备...")
        X_raw, y_raw = self.prepare_training_data(rsrp_data, mask_ratio=0.4)
        
        if len(X_raw) == 0:
            print("❌ 没有足够的训练数据")
            return None
            
        print(f"   样本数量: {len(X_raw)}")
        print(f"   输入维度: {X_raw.shape[1]}")
        print(f"   输出维度: 1 (平均RSRP)")
        
        # 数据标准化
        X_scaled = self.quantum_predictor.scaler_X.fit_transform(X_raw)
        y_scaled = self.quantum_predictor.scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()
        
        # 分割训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=test_size, random_state=42
        )
        
        # 量子特征提取
        print("2. 量子特征提取...")
        X_train_quantum = self.quantum_predictor.extract_quantum_features(X_train)
        X_test_quantum = self.quantum_predictor.extract_quantum_features(X_test)
        
        print(f"   量子特征维度: {X_train_quantum.shape[1]}")
        
        # 训练经典回归器
        print("3. 训练经典回归器...")
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        
        # 多种回归器对比
        regressors = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Support Vector': SVR(kernel='rbf', C=1.0),
            'Linear Regression': nn.Linear(X_train_quantum.shape[1], 1)
        }
        
        results = {}
        
        # 传统机器学习方法
        for name, reg in regressors.items():
            if name != 'Linear Regression':
                reg.fit(X_train_quantum, y_train)
                y_pred = reg.predict(X_test_quantum)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results[name] = {'MSE': mse, 'R2': r2}
                print(f"   {name}: MSE={mse:.6f}, R2={r2:.4f}")
        
        # 神经网络方法
        print("   Neural Network training...")
        net = regressors['Linear Regression']
        criterion = nn.MSELoss()
        optimizer = optim.Adam(net.parameters(), lr=0.01)
        
        # 转换为PyTorch张量
        X_train_torch = torch.FloatTensor(X_train_quantum)
        y_train_torch = torch.FloatTensor(y_train).reshape(-1, 1)
        X_test_torch = torch.FloatTensor(X_test_quantum)
        y_test_torch = torch.FloatTensor(y_test).reshape(-1, 1)
        
        # 训练循环
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = net(X_train_torch)
            loss = criterion(outputs, y_train_torch)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"     Epoch {epoch}, Loss: {loss.item():.6f}")
        
        # 测试神经网络
        with torch.no_grad():
            y_pred_nn = net(X_test_torch).numpy().flatten()
            mse_nn = mean_squared_error(y_test, y_pred_nn)
            r2_nn = r2_score(y_test, y_pred_nn)
            results['Neural Network'] = {'MSE': mse_nn, 'R2': r2_nn}
            print(f"   Neural Network: MSE={mse_nn:.6f}, R2={r2_nn:.4f}")
        
        return results

def create_simulated_beam_data(n_samples=1000, n_beams=32):
    """创建模拟的波束数据"""
    print("生成模拟波束数据...")
    
    # 创建基础波束模式
    np.random.seed(42)
    base_patterns = np.random.randn(8, n_beams)  # 8种基础模式
    
    rsrp_data = []
    for _ in range(n_samples):
        # 随机组合基础模式
        weights = np.random.dirichlet(np.ones(8))
        pattern = np.sum(weights[:, np.newaxis] * base_patterns, axis=0)
        
        # 添加空间相关性和噪声
        spatial_noise = np.random.normal(0, 0.15, n_beams)
        # 模拟相邻波束的相关性
        for i in range(1, n_beams-1):
            spatial_noise[i] = 0.7 * spatial_noise[i] + 0.15 * (spatial_noise[i-1] + spatial_noise[i+1])
        
        sample = pattern + spatial_noise
        rsrp_data.append(sample)
    
    return np.array(rsrp_data)

def demonstrate_quantum_beam_prediction():
    """演示量子波束预测"""
    if not PENNYLANE_AVAILABLE:
        print("❌ Pennylane环境不可用")
        return
    
    # 生成数据
    rsrp_data = create_simulated_beam_data(n_samples=500, n_beams=32)
    print(f"数据形状: {rsrp_data.shape}")
    print(f"RSRP范围: [{np.min(rsrp_data):.2f}, {np.max(rsrp_data):.2f}]")
    
    # 创建预测器
    predictor = QuantumEnhancedRegressor(n_wires=8)
    
    # 训练和评估
    results = predictor.train(rsrp_data)
    
    if results:
        # 可视化结果
        plot_performance_comparison(results)
        return results
    else:
        print("❌ 训练失败")
        return None

def plot_performance_comparison(results):
    """绘制性能比较图"""
    if not results:
        return
        
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    models = list(results.keys())
    mse_values = [results[model]['MSE'] for model in models]
    r2_values = [results[model]['R2'] for model in models]
    
    x = np.arange(len(models))
    
    # MSE比较
    bars1 = ax1.bar(x, mse_values, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax1.set_xlabel('Model Type')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('MSE Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars1, mse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{value:.5f}', ha='center', va='bottom', fontsize=9)
    
    # R2比较
    bars2 = ax2.bar(x, r2_values, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax2.set_xlabel('Model Type')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars2, r2_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('quantum_pennylane_beam_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印总结
    best_mse_model = min(results.keys(), key=lambda x: results[x]['MSE'])
    best_r2_model = max(results.keys(), key=lambda x: results[x]['R2'])
    
    print("\n=== 性能总结 ===")
    print(f"最佳MSE模型: {best_mse_model} (MSE={results[best_mse_model]['MSE']:.6f})")
    print(f"最佳R2模型: {best_r2_model} (R2={results[best_r2_model]['R2']:.4f})")

def main():
    """主函数"""
    print("Pennylane量子波束预测演示")
    print("=" * 40)
    
    try:
        results = demonstrate_quantum_beam_prediction()
        
        if results:
            print("\n✅ 演示完成!")
            print("生成了性能比较图表和详细分析")
        else:
            print("❌ 演示失败")
            
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()