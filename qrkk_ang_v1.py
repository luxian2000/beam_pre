"""
量子核岭回归波束预测系统
基于角度编码量子核，从8个观测波束预测完整256个波束的RSRP值
作者: AI Assistant
日期: 2024年
"""

import pennylane as qml
import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
class Config:
    """配置参数类"""
    # 数据路径
    DATA_PATH = '/Users/luxian/DataSpace/beam_pre/sls_beam_data_spatial_domain_vivo.mat'
    
    # 量子设备配置
    N_QUBITS = 8  # 量子比特数，对应8个观测波束
    DEVICE = 'default.qubit'  # 量子设备
    
    # 数据配置 - 修改样本数量
    N_OBSERVED = 8  # 观测的波束数量
    N_TOTAL_BEAMS = 256  # 总波束数量
    N_SAMPLES = 2400
    TEST_SIZE = 1/6
    
    # 模型配置
    ALPHA = 0.1  # 岭回归正则化参数
    KERNEL_TYPE = 'topology'  # 核函数类型: 固定为'topology'
    USE_CACHE = True  # 是否使用缓存
    N_LAYERS = 2  # 量子电路层数
    
    # 输出配置 - 修改为同名目录
    SAVE_MODEL = True
    SAVE_PLOTS = True
    import os
    OUTPUT_DIR = os.path.splitext(os.path.basename(globals().get('__file__', 'qrkk_ang_v1.py')))[0] + '_output/'

# ==================== 数据加载与预处理 ====================
class DataLoader:
    """数据加载器"""
    
    @staticmethod
    def load_beam_data(file_path):
        """加载MATLAB格式的波束数据"""
        print(f"Loading data file: {file_path}")
        
        with h5py.File(file_path, 'r') as f:
            # 加载主要数据
            rsrp = np.array(f['rsrp'])  # 形状: (336000, 256)
            beam_id_tx = np.array(f['Beam_ID_tx'])
            beam_id_rx = np.array(f['Beam_ID_rx'])
            angles_bs_h = np.array(f['Beam_Angle_BS_h'])
            angles_bs_v = np.array(f['Beam_Angle_BS_v'])
            angles_ue_h = np.array(f['Beam_Angle_UE_h'])
            angles_ue_v = np.array(f['Beam_Angle_UE_v'])
        
        print(f"Data loading completed:")
        print(f"  RSRP data shape: {rsrp.shape}")
        print(f"  Angle data range:")
        print(f"    BS horizontal angle: {angles_bs_h.min():.2f} to {angles_bs_h.max():.2f}")
        print(f"    BS vertical angle: {angles_bs_v.min():.2f} to {angles_bs_v.max():.2f}")
        
        return {
            'rsrp': rsrp,
            'beam_id_tx': beam_id_tx,
            'beam_id_rx': beam_id_rx,
            'angles_bs_h': angles_bs_h,
            'angles_bs_v': angles_bs_v,
            'angles_ue_h': angles_ue_h,
            'angles_ue_v': angles_ue_v
        }
    
    @staticmethod
    def prepare_training_data(rsrp, n_observed=8, n_samples=2000, random_state=42):
        """
        准备训练数据：从完整256维向量中随机选择8个位置作为输入
        
        参数:
            rsrp: 完整的RSRP数据 (n_samples_full, 256)
            n_observed: 观测的波束数量
            n_samples: 使用的样本数量
            random_state: 随机种子
        """
        print(f"\nPreparing training data:")
        print(f"  Observed beams: {n_observed}")
        print(f"  Total beams: 256")
        print(f"  Samples used: {n_samples}")
        
        np.random.seed(random_state)
        
        # 随机选择样本
        total_samples = rsrp.shape[0]
        indices = np.random.choice(total_samples, n_samples, replace=False)
        rsrp_sampled = rsrp[indices, :]
        
        X_train = []
        y_train = []
        observed_positions = []  # 记录观测位置
        
        # 对每个样本，随机选择8个波束作为观测
        for i in range(n_samples):
            full_vector = rsrp_sampled[i, :]
            
            # 随机选择8个观测位置
            observed_idx = np.random.choice(256, n_observed, replace=False)
            observed_values = full_vector[observed_idx]
            
            # 构建输入特征：观测值 + 位置信息
            # 将位置信息归一化到[0, π]
            position_info = observed_idx / 255 * np.pi
            input_vector = np.concatenate([
                observed_values.reshape(-1, 1), 
                position_info.reshape(-1, 1)
            ], axis=1).flatten()
            
            X_train.append(input_vector)  # 16维: 8个值 + 8个位置
            y_train.append(full_vector)   # 256维完整向量
            observed_positions.append(observed_idx)
        
        print(f"  Input feature dimension: {len(X_train[0])}")
        print(f"  Output feature dimension: {len(y_train[0])}")
        
        return np.array(X_train), np.array(y_train), indices, observed_positions

# ==================== 量子核函数 ====================
class QuantumKernels:
    """量子核函数集合"""
    
    def __init__(self, n_qubits=8, n_layers=2, device='default.qubit'):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device, wires=n_qubits)
        self.cache = {} if Config.USE_CACHE else None
    
    def angle_encoding_kernel(self, x1, x2):
        """
        角度编码量子核 - 最适合角度数据的核函数
        
        参数:
            x1, x2: 输入向量 (16维: 8个RSRP值 + 8个位置)
        """
        # 检查缓存
        if self.cache is not None:
            key = (tuple(x1), tuple(x2))
            if key in self.cache:
                return self.cache[key]
        
        n_features = len(x1) // 2  # 8个实际特征
        
        @qml.qnode(self.dev)
        def circuit():
            # 分离RSRP值和位置信息
            rssi1 = x1[:n_features]  # 前8个是RSRP值
            pos1 = x1[n_features:]   # 后8个是位置
            
            rssi2 = x2[:n_features]
            pos2 = x2[n_features:]
            
            # 第一层: RSRP值编码 (使用RY门)
            for i in range(n_features):
                # RSRP值编码 (归一化到[0, π])
                rssi_min = min(np.min(rssi1), np.min(rssi2))
                rssi_max = max(np.max(rssi1), np.max(rssi2))
                
                angle1 = np.pi * (rssi1[i] - rssi_min) / (rssi_max - rssi_min + 1e-8)
                angle2 = np.pi * (rssi2[i] - rssi_min) / (rssi_max - rssi_min + 1e-8)
                
                qml.RY(angle1, wires=i)
                qml.RY(angle2, wires=i)
            
            # 第二层: 位置信息编码 (使用RZ门)
            for i in range(n_features):
                qml.RZ(pos1[i], wires=i)
                qml.RZ(pos2[i], wires=i)
            
            # 纠缠层
            for layer in range(self.n_layers):
                # 线性纠缠
                for i in range(n_features - 1):
                    qml.CNOT(wires=[i, i+1])
                
                # 添加旋转层增强表达能力
                for i in range(n_features):
                    qml.RY(x1[i] * x2[i] / np.pi, wires=i)
            
            # 返回量子态
            return qml.state()
        
        # 计算量子态
        state = circuit()
        
        # 计算量子态保真度的平方作为核值
        kernel_value = np.abs(np.vdot(state, state)) ** 2
        
        # 缓存结果
        if self.cache is not None:
            key = (tuple(x1), tuple(x2))
            self.cache[key] = kernel_value
        
        return kernel_value
    
    def beam_topology_kernel(self, x1, x2):
        """
        波束拓扑感知量子核
        利用波束ID之间的空间关系
        """
        @qml.qnode(self.dev)
        def circuit():
            # 编码RSRP
            for i in range(self.n_qubits):
                # 使用振幅编码
                angle = np.arccos(np.tanh(x1[i] - x2[i]))
                qml.RY(angle, wires=i)
            
            # 添加反映波束相邻关系的纠缠
            # 环状纠缠模式
            for i in range(self.n_qubits):
                qml.CZ(wires=[i, (i+1) % self.n_qubits])
            
            # 对角纠缠（模拟波束网格）
            if self.n_qubits >= 4:
                qml.CZ(wires=[0, 3])
                qml.CZ(wires=[1, 2])
                qml.CZ(wires=[4, 7])
                qml.CZ(wires=[5, 6])
            
            return qml.probs(wires=range(self.n_qubits))
        
        probs = circuit()
        
        # 使用Jensen-Shannon散度作为相似度度量
        uniform = np.ones_like(probs) / len(probs)
        js_div = 0.5 * (np.sum(probs * np.log(probs/uniform + 1e-10)) + 
                        np.sum(uniform * np.log(uniform/probs + 1e-10)))
        
        return np.exp(-js_div)  # 转换为相似度
    
    def hybrid_quantum_kernel(self, x1, x2, gamma=1.0):
        """
        混合核：经典RBF核 + 量子核
        """
        # 经典RBF部分
        rbf = np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
        
        # 量子部分
        quantum_sim = self.angle_encoding_kernel(x1, x2)
        
        # 加权组合
        return 0.7 * quantum_sim + 0.3 * rbf
    
    def get_kernel(self, kernel_type):
        """获取指定类型的核函数"""
        if kernel_type == 'topology':
            return self.beam_topology_kernel
        else:
            raise ValueError(f"不支持的核类型: {kernel_type}，仅支持 'topology'")

# ==================== 量子核岭回归模型 ====================
class QuantumKernelRidgeRegression:
    """量子核岭回归模型"""
    
    def __init__(self, kernel_func, alpha=1.0, kernel_type='topology'):
        self.kernel_func = kernel_func
        self.alpha = alpha
        self.kernel_type = kernel_type
        self.X_train = None
        self.coeff_ = None
        self.x_scaler = None
        self.y_scaler = None
        self.training_time = None
        self.prediction_time = None
        
    def compute_kernel_matrix(self, X1, X2, desc="计算核矩阵"):
        """计算量子核矩阵"""
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        K = np.zeros((n1, n2))
        
        print(f"\n{desc} ({n1}×{n2})...")
        start_time = time.time()
        
        for i in range(n1):
            if i % max(1, n1//10) == 0:  # 每10%进度打印一次
                elapsed = time.time() - start_time
                print(f"  进度: {i+1}/{n1} (耗时: {elapsed:.1f}s)")
            for j in range(n2):
                K[i, j] = self.kernel_func(X1[i], X2[j])
        
        total_time = time.time() - start_time
        print(f"  核矩阵计算完成，总耗时: {total_time:.1f}s")
        
        return K
    
    def fit(self, X_train, y_train):
        """训练模型"""
        print(f"\n训练量子核岭回归模型 (核类型: {self.kernel_type})...")
        start_time = time.time()
        
        # 保存训练数据
        self.X_train = X_train
        
        # 计算训练核矩阵
        K_train = self.compute_kernel_matrix(X_train, X_train, "计算训练核矩阵")
        
        # 添加正则化项
        K_train_reg = K_train + self.alpha * np.eye(K_train.shape[0])
        
        # 使用正规方程求解系数
        print("求解岭回归系数...")
        self.coeff_ = np.linalg.solve(K_train_reg, y_train)
        
        self.training_time = time.time() - start_time
        print(f"模型训练完成，耗时: {self.training_time:.1f}s")
        
        return self
    
    def predict(self, X_test):
        """预测"""
        print(f"\n进行预测...")
        start_time = time.time()
        
        # 计算测试核矩阵
        K_test = self.compute_kernel_matrix(X_test, self.X_train, "计算测试核矩阵")
        
        # 预测
        y_pred = np.dot(K_test, self.coeff_)
        
        self.prediction_time = time.time() - start_time
        print(f"预测完成，耗时: {self.prediction_time:.1f}s")
        
        return y_pred
    
    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        y_pred = self.predict(X_test)
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n模型评估结果:")
        print(f"  MSE (均方误差): {mse:.4f}")
        print(f"  RMSE (均方根误差): {rmse:.4f}")
        print(f"  MAE (平均绝对误差): {mae:.4f}")
        print(f"  R² (决定系数): {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'y_pred': y_pred,
            'y_true': y_test
        }
    
    def save_model(self, filepath):
        """保存模型"""
        model_data = {
            'coeff_': self.coeff_,
            'X_train': self.X_train,
            'alpha': self.alpha,
            'kernel_type': self.kernel_type,
            'training_time': self.training_time,
            'x_scaler': self.x_scaler,
            'y_scaler': self.y_scaler
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到: {filepath}")
    
    @classmethod
    def load_model(cls, filepath, kernel_func):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(kernel_func, alpha=model_data['alpha'], 
                    kernel_type=model_data['kernel_type'])
        model.coeff_ = model_data['coeff_']
        model.X_train = model_data['X_train']
        model.training_time = model_data['training_time']
        model.x_scaler = model_data['x_scaler']
        model.y_scaler = model_data['y_scaler']
        
        print(f"模型已从 {filepath} 加载")
        return model

# ==================== 可视化模块 ====================
class Visualization:
    """可视化工具类"""
    
    @staticmethod
    def visualize_results(y_true, y_pred, sample_idx=0, save_path=None):
        """可视化预测结果"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Quantum Kernel Ridge Regression Beam Prediction Results (Sample {sample_idx})', fontsize=16, fontweight='bold')
        
        # 1. 完整波束模式对比
        ax1 = axes[0, 0]
        beam_indices = np.arange(256)
        ax1.plot(beam_indices, y_true[sample_idx], 'b-', label='Ground Truth', alpha=0.7, linewidth=2)
        ax1.plot(beam_indices, y_pred[sample_idx], 'r--', label='Prediction', alpha=0.7, linewidth=2)
        ax1.set_xlabel('Beam Index', fontsize=12)
        ax1.set_ylabel('RSRP (dBm)', fontsize=12)
        ax1.set_title('Complete Beam Pattern Prediction Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. 误差分布
        ax2 = axes[0, 1]
        errors = y_pred[sample_idx] - y_true[sample_idx]
        n_bins = min(50, len(errors))
        ax2.hist(errors, bins=n_bins, alpha=0.7, color='purple', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error Line')
        ax2.set_xlabel('Prediction Error (dBm)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 3. 最佳波束预测
        ax3 = axes[1, 0]
        top_k = 10
        true_top = np.argsort(y_true[sample_idx])[-top_k:][::-1]
        pred_top = np.argsort(y_pred[sample_idx])[-top_k:][::-1]
        
        x = np.arange(top_k)
        width = 0.35
        ax3.bar(x - width/2, y_true[sample_idx][true_top], width, 
                label='Ground Truth Top', alpha=0.7, color='blue')
        ax3.bar(x + width/2, y_pred[sample_idx][pred_top], width, 
                label='Predicted Top', alpha=0.7, color='red')
        ax3.set_xlabel('Rank', fontsize=12)
        ax3.set_ylabel('RSRP (dBm)', fontsize=12)
        ax3.set_title(f'Top-{top_k} Best Beam Comparison', fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'#{i+1}' for i in range(top_k)])
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 4. 散点图
        ax4 = axes[1, 1]
        ax4.scatter(y_true[sample_idx], y_pred[sample_idx], alpha=0.5, s=20)
        
        # 计算回归线
        coeff = np.polyfit(y_true[sample_idx], y_pred[sample_idx], 1)
        poly = np.poly1d(coeff)
        x_range = np.linspace(y_true[sample_idx].min(), y_true[sample_idx].max(), 100)
        
        ax4.plot(x_range, poly(x_range), 'r-', linewidth=2, label=f'Regression Line: y={coeff[0]:.3f}x+{coeff[1]:.3f}')
        ax4.plot([y_true[sample_idx].min(), y_true[sample_idx].max()],
                 [y_true[sample_idx].min(), y_true[sample_idx].max()], 
                 'g--', linewidth=2, label='Ideal Prediction Line')
        
        ax4.set_xlabel('True RSRP (dBm)', fontsize=12)
        ax4.set_ylabel('Predicted RSRP (dBm)', fontsize=12)
        ax4.set_title('Prediction vs Ground Truth Scatter Plot', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path and Config.SAVE_PLOTS:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization results saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_kernel_matrix(kernel_matrix, title="Quantum Kernel Matrix", save_path=None):
        """可视化核矩阵"""
        plt.figure(figsize=(10, 8))
        plt.imshow(kernel_matrix, cmap='viridis', aspect='auto')
        plt.colorbar(label='Kernel Function Value')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Sample Index', fontsize=12)
        
        if save_path and Config.SAVE_PLOTS:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Kernel matrix plot saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_training_history(history, save_path=None):
        """可视化训练历史"""
        if not history:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 绘制不同核函数的性能对比
        kernels = list(history.keys())
        metrics = ['rmse', 'r2']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = [history[k][metric] for k in kernels]
            
            bars = ax.bar(kernels, values, alpha=0.7)
            ax.set_xlabel('Kernel Function Type', fontsize=12)
            ax.set_ylabel(metric.upper(), fontsize=12)
            ax.set_title(f'{metric.upper()} Comparison of Different Kernel Functions', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # 在每个柱子上添加数值
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path and Config.SAVE_PLOTS:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()

# ==================== 主程序 ====================
def main():
    """主函数"""
    print("=" * 70)
    print("Quantum Kernel Ridge Regression Beam Prediction System")
    print("Angle-encoded quantum kernel for predicting 256 beams from 8 observed beams")
    print("=" * 70)
    
    # 创建输出目录 - 确保使用正确的文件名
    import os
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    output_dir = script_name + '_output'
    Config.OUTPUT_DIR = output_dir + '/'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 1. 加载数据
    print("\n[Stage 1] Data Loading")
    data = DataLoader.load_beam_data(Config.DATA_PATH)
    rsrp = data['rsrp']
    
    # 2. 准备训练数据
    print("\n[Stage 2] Data Preprocessing")
    X, y, sample_indices, observed_positions = DataLoader.prepare_training_data(
        rsrp, 
        n_observed=Config.N_OBSERVED,
        n_samples=Config.N_SAMPLES
    )
    
    # 3. 数据标准化
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y)
    
    # 4. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, 
        test_size=Config.TEST_SIZE, 
        random_state=42
    )
    
    print(f"\nData Split Results:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # 5. 初始化量子核
    print(f"\n[Stage 3] Quantum Kernel Initialization")
    quantum_kernels = QuantumKernels(
        n_qubits=Config.N_QUBITS,
        n_layers=Config.N_LAYERS,
        device=Config.DEVICE
    )
    
    # 获取指定类型的核函数
    kernel_func = quantum_kernels.get_kernel(Config.KERNEL_TYPE)
    print(f"  Using kernel function: {Config.KERNEL_TYPE}")
    print(f"  Number of qubits: {Config.N_QUBITS}")
    print(f"  Circuit layers: {Config.N_LAYERS}")
    
    # 6. 训练模型
    print(f"\n[Stage 4] Model Training")
    model = QuantumKernelRidgeRegression(
        kernel_func, 
        alpha=Config.ALPHA,
        kernel_type=Config.KERNEL_TYPE
    )
    
    # 保存标准化器
    model.x_scaler = x_scaler
    model.y_scaler = y_scaler
    
    # 使用完整的训练集进行训练
    train_samples = X_train.shape[0]  # 使用全部训练样本
    test_samples = X_test.shape[0]    # 使用全部测试样本
    
    print(f"  Using {train_samples} training samples and {test_samples} test samples")
    
    model.fit(X_train, y_train)
    
    # 7. 评估模型
    print(f"\n[Stage 5] Model Evaluation")
    results = model.evaluate(X_test, y_test)
    
    # 反标准化预测结果用于可视化
    y_pred_original = y_scaler.inverse_transform(results['y_pred'])
    y_test_original = y_scaler.inverse_transform(results['y_true'])
    
    # 8. 可视化结果
    print(f"\n[Stage 6] Results Visualization")
    
    # 可视化预测结果
    vis_save_path = os.path.join(output_dir, 'quantum_kernel_prediction_results.png')
    Visualization.visualize_results(
        y_test_original, 
        y_pred_original, 
        sample_idx=0,
        save_path=vis_save_path
    )
    
    # 9. 保存模型和结果
    if Config.SAVE_MODEL:
        print(f"\n[Stage 7] Saving Results")
        
        # 保存模型
        model_save_path = os.path.join(output_dir, 'quantum_kernel_model.pkl')
        model.save_model(model_save_path)
        
        # 保存评估结果
        results_save_path = os.path.join(output_dir, 'evaluation_results.json')
        results_summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'config': {
                'n_qubits': Config.N_QUBITS,
                'n_observed': Config.N_OBSERVED,
                'n_samples': Config.N_SAMPLES,
                'kernel_type': Config.KERNEL_TYPE,
                'alpha': Config.ALPHA,
                'n_layers': Config.N_LAYERS
            },
            'performance': {
                'mse': float(results['mse']),
                'rmse': float(results['rmse']),
                'mae': float(results['mae']),
                'r2': float(results['r2'])
            },
            'training_info': {
                'training_time': float(model.training_time),
                'prediction_time': float(model.prediction_time),
                'train_samples': train_samples,
                'test_samples': test_samples
            }
        }
        
        with open(results_save_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        print(f"  Evaluation results saved to: {results_save_path}")
        
        # 保存训练日志
        log_save_path = os.path.join(output_dir, 'training_log.txt')
        with open(log_save_path, 'w', encoding='utf-8') as f:
            f.write(f"Quantum Kernel Ridge Regression Training Log\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Script name: {script_name}\n")
            f.write(f"Output directory: {output_dir}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Configuration Parameters:\n")
            f.write(f"  Number of qubits: {Config.N_QUBITS}\n")
            f.write(f"  Observed beams: {Config.N_OBSERVED}\n")
            f.write(f"  Sample count: {Config.N_SAMPLES}\n")
            f.write(f"  Kernel function type: {Config.KERNEL_TYPE}\n")
            f.write(f"  Regularization parameter: {Config.ALPHA}\n")
            f.write(f"  Circuit layers: {Config.N_LAYERS}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Performance Results:\n")
            f.write(f"  MSE: {results['mse']:.6f}\n")
            f.write(f"  RMSE: {results['rmse']:.6f}\n")
            f.write(f"  MAE: {results['mae']:.6f}\n")
            f.write(f"  R²: {results['r2']:.6f}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Training Information:\n")
            f.write(f"  Training time: {model.training_time:.2f} seconds\n")
            f.write(f"  Prediction time: {model.prediction_time:.2f} seconds\n")
            f.write(f"  Training samples: {train_samples}\n")
            f.write(f"  Test samples: {test_samples}\n")
        
        print(f"  Training log saved to: {log_save_path}")
    
    # 10. 总结报告
    total_time = time.time() - start_time
    print(f"\n" + "=" * 70)
    print("Quantum Kernel Ridge Regression Beam Prediction Completed!")
    print("=" * 70)
    print(f"Total runtime: {total_time:.1f} seconds")
    print(f"Best kernel function: {Config.KERNEL_TYPE}")
    print(f"Model performance:")
    print(f"  RMSE: {results['rmse']:.4f}")
    print(f"  R²: {results['r2']:.4f}")
    print(f"Output files saved in: {output_dir}/")
    print("=" * 70)
    
    return model, results

if __name__ == "__main__":
    # 执行主程序
    try:
        model, results = main()
        print(f"\nTraining completed successfully!")
        print(f"Final results - RMSE: {results['rmse']:.4f}, R²: {results['r2']:.4f}")
        
    except Exception as e:
        print(f"Error occurred during execution: {str(e)}")
        import traceback
        traceback.print_exc()