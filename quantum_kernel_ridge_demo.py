"""
量子核岭回归演示
Quantum Kernel Ridge Regression Demo
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 尝试导入量子计算库
try:
    import pennylane as qml
    QUANTUM_AVAILABLE = True
    print("✓ 量子计算环境就绪")
except ImportError:
    QUANTUM_AVAILABLE = False
    print("⚠ 量子库未安装，使用经典模拟")

class QuantumKernelRidge:
    """
    量子核岭回归实现
    Quantum Kernel Ridge Regression Implementation
    """
    
    def __init__(self, n_qubits=4, lambda_reg=0.1):
        self.n_qubits = n_qubits
        self.lambda_reg = lambda_reg
        self.alpha = None
        self.X_train = None
        self.scaler = StandardScaler()
        
        if QUANTUM_AVAILABLE:
            self.dev = qml.device("default.qubit", wires=n_qubits)
            self.quantum_kernel_func = self._create_quantum_kernel()
        else:
            self.quantum_kernel_func = self._classical_approximation
    
    def _create_quantum_kernel(self):
        """创建量子核函数"""
        
        @qml.qnode(self.dev)
        def quantum_circuit(x1, x2):
            # 数据编码到量子态
            self._encode_data(x1, wires=range(self.n_qubits//2))
            self._encode_data(x2, wires=range(self.n_qubits//2, self.n_qubits))
            
            # 简单纠缠
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            
            # 测量泡利Z算符的期望值
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        def quantum_kernel(x1, x2):
            # 确保输入维度正确
            if len(x1) > self.n_qubits:
                x1 = x1[:self.n_qubits]
            if len(x2) > self.n_qubits:
                x2 = x2[:self.n_qubits]
                
            # 补零
            x1 = np.pad(x1, (0, max(0, self.n_qubits - len(x1))))
            x2 = np.pad(x2, (0, max(0, self.n_qubits - len(x2))))
            
            # 计算量子相关性
            result = quantum_circuit(x1, x2)
            # 简化的核函数：基于测量结果的相似度
            kernel_value = np.mean(result) ** 2
            return max(0, kernel_value)  # 确保非负
        
        return quantum_kernel
    
    def _encode_data(self, data, wires):
        """将数据编码到指定量子比特"""
        for i, wire in enumerate(wires):
            if i < len(data):
                # 使用RY旋转门编码数据
                qml.RY(data[i] * np.pi, wires=wire)
    
    def _classical_approximation(self, x1, x2):
        """当量子不可用时的经典近似核函数"""
        # 使用RBF核的经典近似
        gamma = 0.1
        diff = np.array(x1) - np.array(x2)
        return np.exp(-gamma * np.sum(diff**2))
    
    def compute_kernel_matrix(self, X1, X2=None):
        """计算核矩阵"""
        if X2 is None:
            X2 = X1
            
        K = np.zeros((len(X1), len(X2)))
        for i in range(len(X1)):
            for j in range(len(X2)):
                K[i,j] = self.quantum_kernel_func(X1[i], X2[j])
        return K
    
    def fit(self, X, y):
        """训练量子核岭回归器"""
        print("训练量子核岭回归器...")
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        self.X_train = X_scaled
        
        # 计算训练核矩阵
        print("计算量子核矩阵...")
        K_train = self.compute_kernel_matrix(X_scaled)
        
        # 求解系数 (K + λI)α = y
        print("求解回归系数...")
        self.alpha = np.linalg.solve(
            K_train + self.lambda_reg * np.eye(len(K_train)), 
            y
        )
        
        print("训练完成!")
        
    def predict(self, X):
        """预测"""
        if self.alpha is None:
            raise ValueError("模型尚未训练")
            
        # 标准化测试数据
        X_scaled = self.scaler.transform(X)
        
        # 计算测试-训练核矩阵
        K_cross = self.compute_kernel_matrix(X_scaled, self.X_train)
        
        # 预测
        return K_cross @ self.alpha

def create_beam_simulation_data(n_samples=200, n_features=8):
    """创建模拟的波束数据"""
    print("生成模拟波束数据...")
    
    np.random.seed(42)
    
    # 创建基础波束模式
    base_patterns = np.random.randn(4, n_features)
    
    X_data = []
    y_data = []
    
    for _ in range(n_samples):
        # 随机组合基础模式
        weights = np.random.dirichlet(np.ones(4))
        pattern = np.sum(weights[:, np.newaxis] * base_patterns, axis=0)
        
        # 添加噪声
        noise = np.random.normal(0, 0.1, n_features)
        sample = pattern + noise
        
        X_data.append(sample)
        
        # 目标值：某种波束质量指标
        quality_metric = np.mean(sample) + 0.5 * np.std(sample)
        y_data.append(quality_metric)
    
    return np.array(X_data), np.array(y_data)

def compare_methods():
    """比较不同方法的性能"""
    print("=== 量子核岭回归 vs 经典方法对比 ===\n")
    
    # 生成数据
    X, y = create_beam_simulation_data(n_samples=150, n_features=6)
    print(f"数据形状: X={X.shape}, y={y.shape}")
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    results = {}
    
    # 1. 经典核岭回归
    print("1. 训练经典核岭回归...")
    classical_krr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
    classical_krr.fit(X_train, y_train)
    y_pred_classical = classical_krr.predict(X_test)
    
    mse_classical = mean_squared_error(y_test, y_pred_classical)
    r2_classical = r2_score(y_test, y_pred_classical)
    results['Classical_KRR'] = {'MSE': mse_classical, 'R2': r2_classical}
    print(f"   MSE: {mse_classical:.6f}, R2: {r2_classical:.4f}")
    
    # 2. 线性回归基线
    print("2. 训练线性回归...")
    from sklearn.linear_model import LinearRegression
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    y_pred_linear = linear_reg.predict(X_test)
    
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    r2_linear = r2_score(y_test, y_pred_linear)
    results['Linear_Regression'] = {'MSE': mse_linear, 'R2': r2_linear}
    print(f"   MSE: {mse_linear:.6f}, R2: {r2_linear:.4f}")
    
    # 3. 量子核岭回归
    if QUANTUM_AVAILABLE:
        print("3. 训练量子核岭回归...")
        try:
            quantum_krr = QuantumKernelRidge(n_qubits=4, lambda_reg=0.1)
            quantum_krr.fit(X_train, y_train)
            y_pred_quantum = quantum_krr.predict(X_test)
            
            mse_quantum = mean_squared_error(y_test, y_pred_quantum)
            r2_quantum = r2_score(y_test, y_pred_quantum)
            results['Quantum_KRR'] = {'MSE': mse_quantum, 'R2': r2_quantum}
            print(f"   MSE: {mse_quantum:.6f}, R2: {r2_quantum:.4f}")
            
        except Exception as e:
            print(f"   量子训练失败: {e}")
            results['Quantum_KRR'] = {'MSE': float('inf'), 'R2': float('-inf')}
    else:
        print("3. 量子核岭回归 (模拟)...")
        # 使用经典近似
        quantum_krr = QuantumKernelRidge(n_qubits=4, lambda_reg=0.1)
        quantum_krr.fit(X_train, y_train)
        y_pred_quantum = quantum_krr.predict(X_test)
        
        mse_quantum = mean_squared_error(y_test, y_pred_quantum)
        r2_quantum = r2_score(y_test, y_pred_quantum)
        results['Quantum_KRR_Approx'] = {'MSE': mse_quantum, 'R2': r2_quantum}
        print(f"   MSE: {mse_quantum:.6f}, R2: {r2_quantum:.4f}")
    
    return results

def visualize_comparison(results):
    """可视化结果比较"""
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    models = list(results.keys())
    mse_values = [results[model]['MSE'] for model in models]
    r2_values = [results[model]['R2'] for model in models]
    
    x = np.arange(len(models))
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    # MSE比较
    bars1 = ax1.bar(x, mse_values, color=colors[:len(models)], alpha=0.7)
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title('MSE Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, value in zip(bars1, mse_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.01,
                f'{value:.5f}', ha='center', va='bottom', fontsize=9)
    
    # R2比较
    bars2 = ax2.bar(x, r2_values, color=colors[:len(models)], alpha=0.7)
    ax2.set_xlabel('Method')
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
    plt.savefig('quantum_kernel_ridge_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印总结
    print("\n=== 性能总结 ===")
    best_mse = min(results.keys(), key=lambda x: results[x]['MSE'])
    best_r2 = max(results.keys(), key=lambda x: results[x]['R2'])
    
    print(f"最佳MSE方法: {best_mse} (MSE={results[best_mse]['MSE']:.6f})")
    print(f"最佳R2方法: {best_r2} (R2={results[best_r2]['R2']:.4f})")

def main():
    """主函数"""
    print("量子核岭回归演示")
    print("=" * 30)
    
    try:
        # 运行比较实验
        results = compare_methods()
        
        # 可视化结果
        visualize_comparison(results)
        
        print("\n✅ 演示完成!")
        print("生成了性能比较图表")
        
    except Exception as e:
        print(f"❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()