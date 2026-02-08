"""
修复版的Pennylane量子波束预测
Fixed PennyLane Quantum Beam Prediction
"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# 检查Pennylane是否可用
try:
    dev = qml.device("default.qubit", wires=4)  # 减少量子比特数
    print("✓ Pennylane环境就绪")
    PENNYLANE_AVAILABLE = True
except Exception as e:
    print(f"✗ Pennylane环境异常: {e}")
    PENNYLANE_AVAILABLE = False

class FixedQuantumBeamPredictor:
    """
    修复版量子波束预测器
    Fixed Quantum Beam Predictor
    """
    
    def __init__(self, n_wires=4, n_layers=1):
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_wires)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # 创建量子电路
        self.qnode = self._create_quantum_circuit()
        
    def _create_quantum_circuit(self):
        """创建简化版量子电路"""
        
        @qml.qnode(self.dev)
        def circuit(inputs, weights):
            # 数据编码
            self._simple_encoding(inputs)
            
            # 变分层
            self._simple_variational(weights)
            
            # 返回测量结果
            return qml.probs(wires=range(self.n_wires))
        
        return circuit
    
    def _simple_encoding(self, inputs):
        """简化的数据编码"""
        # 只使用前n_wires个输入值
        for i in range(min(len(inputs), self.n_wires)):
            # Ry旋转编码
            qml.RY(inputs[i] * np.pi, wires=i)
    
    def _simple_variational(self, weights):
        """简化的变分层"""
        # 单层旋转门
        for wire in range(self.n_wires):
            qml.RY(weights[wire], wires=wire)
        
        # 简单纠缠
        for wire in range(self.n_wires - 1):
            qml.CNOT(wires=[wire, wire + 1])
    
    def extract_quantum_features(self, X):
        """提取量子特征"""
        if not PENNYLANE_AVAILABLE:
            return self._classical_substitute(X)
        
        n_samples = X.shape[0]
        quantum_features = []
        
        # 简单的权重初始化
        weights = np.random.uniform(-np.pi, np.pi, self.n_wires)
        
        for i in range(n_samples):
            try:
                # 确保输入维度匹配
                inputs = X[i][:self.n_wires]  # 截取前n_wires个特征
                # 补零如果输入不足
                if len(inputs) < self.n_wires:
                    inputs = np.pad(inputs, (0, self.n_wires - len(inputs)))
                
                # 执行量子电路
                probs = self.qnode(inputs, weights)
                quantum_features.append(probs)
                
            except Exception as e:
                print(f"样本 {i} 处理出错: {e}")
                # 出错时使用默认特征
                quantum_features.append(np.ones(2**self.n_wires) / (2**self.n_wires))
        
        return np.array(quantum_features)
    
    def _classical_substitute(self, X):
        """经典替代方案"""
        # 使用统计特征和FFT特征
        features = []
        for sample in X:
            # 统计特征
            stats = [np.mean(sample), np.std(sample), np.max(sample), np.min(sample)]
            
            # FFT特征（简化版）
            fft_result = np.abs(np.fft.fft(sample))[:self.n_wires]
            
            # 组合特征
            combined = np.concatenate([stats[:2], fft_result[:2]])
            
            # 确保维度正确
            if len(combined) < 2**self.n_wires:
                combined = np.pad(combined, (0, 2**self.n_wires - len(combined)))
            else:
                combined = combined[:2**self.n_wires]
                
            features.append(combined)
        
        return np.array(features)

class QuantumBeamAnalysis:
    """量子波束分析器"""
    
    def __init__(self):
        self.quantum_predictor = FixedQuantumBeamPredictor(n_wires=4)
        
    def prepare_data(self, rsrp_data, mask_ratio=0.3):
        """准备训练数据"""
        n_samples, n_beams = rsrp_data.shape
        
        # 创建mask模式
        mask = np.random.random((n_samples, n_beams)) > mask_ratio
        
        X_input = []
        y_target = []
        
        for i in range(n_samples):
            observed_values = rsrp_data[i][mask[i]]
            observed_indices = np.where(mask[i])[0]
            
            if len(observed_values) > 0:
                # 构造输入特征
                input_features = []
                
                # 观测值统计（最多4个）
                obs_subset = observed_values[:4]
                input_features.extend(obs_subset)
                
                # 统计特征
                input_features.extend([
                    np.mean(observed_values),
                    np.std(observed_values),
                    len(observed_values)
                ])
                
                # 位置信息（最多2个）
                pos_subset = observed_indices[:2]
                input_features.extend(pos_subset)
                
                # 填充到固定长度
                while len(input_features) < 12:
                    input_features.append(0)
                input_features = input_features[:12]  # 固定12维
                
                X_input.append(input_features)
                
                # 目标：预测所有波束的平均RSRP
                y_target.append(np.mean(rsrp_data[i]))
        
        return np.array(X_input), np.array(y_target)
    
    def train_and_evaluate(self, rsrp_data):
        """训练和评估模型"""
        print("=== 量子波束预测分析 ===")
        
        # 数据准备
        print("1. 准备数据...")
        X_raw, y_raw = self.prepare_data(rsrp_data, mask_ratio=0.4)
        
        if len(X_raw) == 0:
            print("❌ 没有足够数据")
            return None
            
        print(f"   样本数: {len(X_raw)}")
        print(f"   输入维度: {X_raw.shape[1]}")
        
        # 数据标准化
        X_scaled = self.quantum_predictor.scaler_X.fit_transform(X_raw)
        y_scaled = self.quantum_predictor.scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        # 量子特征提取
        print("2. 提取量子特征...")
        X_train_quantum = self.quantum_predictor.extract_quantum_features(X_train)
        X_test_quantum = self.quantum_predictor.extract_quantum_features(X_test)
        
        print(f"   量子特征维度: {X_train_quantum.shape[1]}")
        
        # 模型训练对比
        print("3. 模型训练对比...")
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
            'Support Vector': SVR(kernel='rbf', C=1.0),
            'Classical Features': None  # 直接使用原始特征
        }
        
        results = {}
        
        # 使用量子特征的模型
        for name, model in models.items():
            if model is not None:
                model.fit(X_train_quantum, y_train)
                y_pred = model.predict(X_test_quantum)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results[f'Quantum_{name}'] = {'MSE': mse, 'R2': r2}
                print(f"   Quantum {name}: MSE={mse:.6f}, R2={r2:.4f}")
        
        # 使用原始特征的基线
        print("   Classical Baseline (原始特征)...")
        baseline_model = RandomForestRegressor(n_estimators=50, random_state=42)
        baseline_model.fit(X_train, y_train)
        y_pred_baseline = baseline_model.predict(X_test)
        mse_baseline = mean_squared_error(y_test, y_pred_baseline)
        r2_baseline = r2_score(y_test, y_pred_baseline)
        results['Classical_Baseline'] = {'MSE': mse_baseline, 'R2': r2_baseline}
        print(f"   Classical Baseline: MSE={mse_baseline:.6f}, R2={r2_baseline:.4f}")
        
        return results

def create_beam_dataset(n_samples=800, n_beams=16):
    """创建波束数据集"""
    print("创建波束数据集...")
    
    np.random.seed(42)
    
    # 创建空间相关的波束模式
    beam_positions = np.linspace(0, 2*np.pi, n_beams)
    
    rsrp_data = []
    for _ in range(n_samples):
        # 创建基础波束轮廓
        base_profile = np.sin(beam_positions + np.random.uniform(0, 2*np.pi))
        
        # 添加多径效应
        multipath = 0.3 * np.sin(2 * beam_positions + np.random.uniform(0, 2*np.pi))
        
        # 添加随机衰落
        fading = np.random.normal(0, 0.2, n_beams)
        
        # 组合信号
        signal = base_profile + multipath + fading
        
        # 确保合理范围
        signal = np.clip(signal, -3, 3)
        rsrp_data.append(signal)
    
    return np.array(rsrp_data)

def visualize_results(results):
    """可视化结果"""
    if not results:
        return
        
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 准备数据
    models = list(results.keys())
    mse_values = [results[model]['MSE'] for model in models]
    r2_values = [results[model]['R2'] for model in models]
    
    x = np.arange(len(models))
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum']
    
    # MSE比较
    bars1 = ax1.bar(x, mse_values, color=colors[:len(models)], alpha=0.7)
    ax1.set_xlabel('Model Configuration')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('MSE Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars1, mse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.01,
                f'{value:.5f}', ha='center', va='bottom', fontsize=9)
    
    # R2比较
    bars2 = ax2.bar(x, r2_values, color=colors[:len(models)], alpha=0.7)
    ax2.set_xlabel('Model Configuration')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars2, r2_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('quantum_pennylane_fixed_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 性能总结
    print("\n=== 性能分析总结 ===")
    best_mse = min(results.keys(), key=lambda x: results[x]['MSE'])
    best_r2 = max(results.keys(), key=lambda x: results[x]['R2'])
    
    print(f"🏆 最佳MSE模型: {best_mse}")
    print(f"   MSE = {results[best_mse]['MSE']:.6f}")
    print(f"🏆 最佳R2模型: {best_r2}")
    print(f"   R2 = {results[best_r2]['R2']:.4f}")
    
    # 量子优势分析
    quantum_models = [k for k in results.keys() if k.startswith('Quantum_')]
    if len(quantum_models) > 0:
        quantum_mse_avg = np.mean([results[k]['MSE'] for k in quantum_models])
        classical_mse = results['Classical_Baseline']['MSE']
        
        improvement = (classical_mse - quantum_mse_avg) / classical_mse * 100
        print(f"\n📈 量子方法平均改进: {improvement:.2f}%")

def main():
    """主函数"""
    print("Pennylane量子波束预测 - 修复版")
    print("=" * 45)
    
    if not PENNYLANE_AVAILABLE:
        print("❌ Pennylane环境不可用")
        return
    
    try:
        # 创建数据
        rsrp_data = create_beam_dataset(n_samples=600, n_beams=16)
        print(f"数据集形状: {rsrp_data.shape}")
        print(f"RSRP范围: [{np.min(rsrp_data):.2f}, {np.max(rsrp_data):.2f}]")
        
        # 分析器
        analyzer = QuantumBeamAnalysis()
        
        # 训练和评估
        results = analyzer.train_and_evaluate(rsrp_data)
        
        if results:
            # 可视化
            visualize_results(results)
            print("\n✅ 分析完成!")
        else:
            print("❌ 分析失败")
            
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()