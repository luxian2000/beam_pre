"""
U3 Gate Data Re-uploading Demonstration
U3门数据重上传演示

Demonstrates how U3 gates can encode multiple classical data points
into quantum states through data re-uploading technique.
"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class U3DataReuploading:
    """U3门数据重上传演示类"""
    
    def __init__(self, n_qubits=3, n_layers=4):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
    def u3_encoding_circuit(self, data_layer, weights_layer):
        """
        使用U3门进行数据编码的量子线路
        
        Args:
            data_layer: 当前层要编码的经典数据 (最多3*n_qubits个数据点)
            weights_layer: 当前层的参数化权重
        """
        data_idx = 0
        
        # 对每个量子比特应用U3门
        for i in range(self.n_qubits):
            # U3门可以编码3个经典数据点
            if data_idx + 2 < len(data_layer):
                theta = data_layer[data_idx]     # 第一个数据点
                phi = data_layer[data_idx + 1]   # 第二个数据点  
                lam = data_layer[data_idx + 2]   # 第三个数据点
                
                # 应用U3门进行编码
                qml.U3(theta, phi, lam, wires=i)
                data_idx += 3
            else:
                # 如果数据不足，使用零填充
                qml.U3(0, 0, 0, wires=i)
        
        # 添加纠缠层
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i+1])
        
        # 参数化旋转层
        for i in range(self.n_qubits):
            qml.RY(weights_layer[i], wires=i)
    
    def data_reuploading_circuit(self, input_data, weights):
        """
        完整的数据重上传量子线路
        
        Args:
            input_data: 完整的经典输入数据
            weights: 所有层的参数权重
        """
        # 计算每层需要的数据量
        data_per_layer = 3 * self.n_qubits  # 每个量子比特编码3个数据点
        total_data_needed = data_per_layer * self.n_layers
        
        # 如果数据不足，进行填充
        if len(input_data) < total_data_needed:
            padded_data = np.pad(input_data, (0, total_data_needed - len(input_data)), 
                               mode='constant', constant_values=0)
        else:
            padded_data = input_data[:total_data_needed]
        
        # 分层处理数据
        for layer in range(self.n_layers):
            start_idx = layer * data_per_layer
            end_idx = (layer + 1) * data_per_layer
            layer_data = padded_data[start_idx:end_idx]
            
            self.u3_encoding_circuit(layer_data, weights[layer])
    
    def get_quantum_features(self, input_data, weights):
        """
        获取量子特征表示
        
        Returns:
            量子态的测量结果作为特征
        """
        @qml.qnode(self.dev)
        def circuit(data, weights):
            self.data_reuploading_circuit(data, weights)
            
            # 测量所有量子比特的泡利算符
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return circuit(input_data, weights)
    
    def demonstrate_capacity(self):
        """演示编码容量"""
        print("=== U3门数据编码容量分析 ===")
        print(f"量子比特数: {self.n_qubits}")
        print(f"层数: {self.n_layers}")
        print(f"每量子比特U3门编码能力: 3个经典数据点")
        print(f"每层总编码容量: {3 * self.n_qubits} 个数据点")
        print(f"总编码容量: {3 * self.n_qubits * self.n_layers} 个数据点")
        
        # 生成示例数据
        total_capacity = 3 * self.n_qubits * self.n_layers
        sample_data = np.random.uniform(-np.pi, np.pi, total_capacity)
        
        # 初始化权重
        weights = np.random.uniform(-np.pi, np.pi, (self.n_layers, self.n_qubits))
        
        print(f"\n示例数据维度: {len(sample_data)}")
        print("数据重上传处理中...")
        
        # 获取量子特征
        quantum_features = self.get_quantum_features(sample_data, weights)
        print(f"量子特征输出维度: {len(quantum_features)}")
        print(f"量子特征值: {quantum_features}")
        
        return sample_data, quantum_features

def compare_encoding_methods():
    """比较不同编码方法的容量"""
    print("\n=== 不同编码方法容量对比 ===")
    
    methods = {
        'U3门编码': 3,      # 每量子比特3个数据点
        'RY门编码': 1,      # 每量子比特1个数据点  
        '角度编码': 1,      # 每量子比特1个数据点
        '幅度编码': None    # 特殊处理
    }
    
    n_qubits = 4
    print(f"以{n_qubits}个量子比特为例:")
    
    for method, capacity_per_qubit in methods.items():
        if method == '幅度编码':
            total_capacity = 2**n_qubits - 1
            print(f"{method}: {total_capacity} 个数据点 (整体编码)")
        else:
            total_capacity = capacity_per_qubit * n_qubits
            print(f"{method}: {total_capacity} 个数据点 ({capacity_per_qubit}×{n_qubits})")

def main():
    """主函数"""
    print("U3门数据重上传演示")
    print("=" * 50)
    
    # 创建演示实例
    demo = U3DataReuploading(n_qubits=3, n_layers=4)
    
    # 演示编码容量
    sample_data, quantum_features = demo.demonstrate_capacity()
    
    # 比较不同方法
    compare_encoding_methods()
    
    print("\n=== 关键优势 ===")
    print("1. 高效编码: 单个U3门可编码3个经典数据点")
    print("2. 灵活性强: 可通过增加层数扩展编码容量")  
    print("3. 表达力强: U3门是通用单量子比特门")
    print("4. 可微分: 适合量子机器学习训练")
    
    print("\n=== 应用场景 ===")
    print("- 量子机器学习特征映射")
    print("- 高维数据量子编码")
    print("- 参数化量子电路设计")
    print("- 量子神经网络构建")

if __name__ == "__main__":
    main()