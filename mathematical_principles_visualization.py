"""
量子核岭回归数学原理可视化演示
Mathematical Principles Visualization of Quantum Kernel Ridge Regression
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import rbf_kernel
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class MathematicalVisualization:
    """数学原理可视化类"""
    
    def __init__(self):
        pass
    
    def visualize_hilbert_space_mapping(self):
        """可视化希尔伯特空间映射"""
        fig = plt.figure(figsize=(15, 5))
        
        # 子图1: 经典特征空间
        ax1 = fig.add_subplot(131)
        x = np.linspace(-2, 2, 100)
        phi_classical = np.column_stack([x, x**2, x**3])  # 多项式特征映射
        ax1.plot(x, phi_classical[:, 0], 'b-', label='φ₁(x) = x', linewidth=2)
        ax1.plot(x, phi_classical[:, 1], 'r-', label='φ₂(x) = x²', linewidth=2)
        ax1.plot(x, phi_classical[:, 2], 'g-', label='φ₃(x) = x³', linewidth=2)
        ax1.set_xlabel('输入 x')
        ax1.set_ylabel('特征值')
        ax1.set_title('经典多项式特征映射\nφ: ℝ → ℝ³')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 量子态球面表示
        ax2 = fig.add_subplot(132, projection='3d')
        theta = np.linspace(0, 2*np.pi, 50)
        phi = np.linspace(0, np.pi, 50)
        Theta, Phi = np.meshgrid(theta, phi)
        
        # 单位球面
        X = np.sin(Phi) * np.cos(Theta)
        Y = np.sin(Phi) * np.sin(Theta)
        Z = np.cos(Phi)
        
        ax2.plot_surface(X, Y, Z, alpha=0.3, color='lightblue')
        
        # 示例量子态点
        states = [(0, 0), (np.pi/4, np.pi/4), (np.pi/2, np.pi/3)]
        colors = ['red', 'green', 'blue']
        labels = ['|φ(x₁)⟩', '|φ(x₂)⟩', '|φ(x₃)⟩']
        
        for i, (theta_s, phi_s) in enumerate(states):
            x_point = np.sin(phi_s) * np.cos(theta_s)
            y_point = np.sin(phi_s) * np.sin(theta_s)
            z_point = np.cos(phi_s)
            ax2.scatter([x_point], [y_point], [z_point], 
                       c=colors[i], s=100, label=labels[i])
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('量子态希尔伯特空间\nℋ^(2ⁿ)')
        ax2.legend()
        
        # 子图3: 核函数值对比
        ax3 = fig.add_subplot(133)
        x_sample = np.array([-1, 0, 1])
        
        # 经典RBF核
        K_classical = rbf_kernel(x_sample.reshape(-1, 1), gamma=1.0)
        
        # 量子核近似（简化）
        def quantum_kernel_approx(x1, x2):
            # 简化的量子重叠积分近似
            diff = x1 - x2
            return np.exp(-0.5 * diff**2) * (1 + 0.1 * np.cos(x1 * x2))
        
        K_quantum = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                K_quantum[i,j] = quantum_kernel_approx(x_sample[i], x_sample[j])
        
        # 绘制热力图
        x_labels = ['x₁=-1', 'x₂=0', 'x₃=1']
        im1 = ax3.imshow(K_classical, cmap='Blues', alpha=0.7)
        for i in range(3):
            for j in range(3):
                ax3.text(j, i, f'{K_classical[i,j]:.3f}', 
                        ha='center', va='center', fontweight='bold')
        ax3.set_xticks(range(3))
        ax3.set_yticks(range(3))
        ax3.set_xticklabels(x_labels)
        ax3.set_yticklabels(x_labels)
        ax3.set_title('经典RBF核矩阵')
        
        plt.tight_layout()
        plt.savefig('hilbert_space_mapping.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return K_classical, K_quantum
    
    def visualize_optimization_landscape(self):
        """可视化优化景观"""
        fig = plt.figure(figsize=(15, 5))
        
        # 生成优化景观数据
        alpha1 = np.linspace(-3, 3, 100)
        alpha2 = np.linspace(-3, 3, 100)
        Alpha1, Alpha2 = np.meshgrid(alpha1, alpha2)
        
        # 简化的二次目标函数（强凸）
        def objective_function(a1, a2, lambda_reg=0.1):
            # 模拟核岭回归目标函数
            return 0.5 * (a1**2 + 2*a2**2 + 0.5*a1*a2) + 0.5 * lambda_reg * (a1**2 + a2**2)
        
        Z = objective_function(Alpha1, Alpha2)
        
        # 子图1: 3D景观
        ax1 = fig.add_subplot(131, projection='3d')
        surf = ax1.plot_surface(Alpha1, Alpha2, Z, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('α₁')
        ax1.set_ylabel('α₂')
        ax1.set_zlabel('J(α)')
        ax1.set_title('优化目标函数3D景观\n(强凸函数)')
        fig.colorbar(surf, ax=ax1, shrink=0.5)
        
        # 子图2: 等高线图
        ax2 = fig.add_subplot(132)
        contour = ax2.contour(Alpha1, Alpha2, Z, levels=20, cmap='viridis')
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.set_xlabel('α₁')
        ax2.set_ylabel('α₂')
        ax2.set_title('目标函数等高线图')
        ax2.grid(True, alpha=0.3)
        
        # 添加最优解
        optimal_alpha1 = -0.2
        optimal_alpha2 = -0.4
        ax2.plot(optimal_alpha1, optimal_alpha2, 'r*', markersize=15, 
                label=f'最优解({optimal_alpha1:.1f}, {optimal_alpha2:.1f})')
        ax2.legend()
        
        # 子图3: 收敛性分析
        ax3 = fig.add_subplot(133)
        iterations = range(1, 51)
        
        # 模拟不同算法的收敛曲线
        def convergence_rate(iter_num, rate):
            return 10 * np.exp(-rate * iter_num)
        
        gd_convergence = [convergence_rate(i, 0.1) for i in iterations]      # 梯度下降
        cg_convergence = [convergence_rate(i, 0.3) for i in iterations]      # 共轭梯度
        newton_convergence = [convergence_rate(i, 0.8) for i in iterations]  # 牛顿法
        
        ax3.semilogy(iterations, gd_convergence, 'b-', label='梯度下降', linewidth=2)
        ax3.semilogy(iterations, cg_convergence, 'g-', label='共轭梯度', linewidth=2)
        ax3.semilogy(iterations, newton_convergence, 'r-', label='牛顿法', linewidth=2)
        
        ax3.set_xlabel('迭代次数')
        ax3.set_ylabel('目标函数值 (对数尺度)')
        ax3.set_title('优化算法收敛速度比较')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimization_landscape.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_condition_number(self):
        """分析条件数的影响"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 生成不同条件数的矩阵
        eigenvalues_good = [1, 2, 3, 4, 5]      # 条件数良好
        eigenvalues_bad = [0.01, 1, 10, 100, 1000]  # 条件数差
        
        # 构造对应的对称正定矩阵
        def construct_spd_matrix(evals):
            n = len(evals)
            # 随机正交矩阵
            Q, _ = np.linalg.qr(np.random.randn(n, n))
            # 对角矩阵
            D = np.diag(evals)
            return Q @ D @ Q.T
        
        K_good = construct_spd_matrix(eigenvalues_good)
        K_bad = construct_spd_matrix(eigenvalues_bad)
        
        # 子图1: 特征值分布
        ax1.bar(range(len(eigenvalues_good)), eigenvalues_good, 
                alpha=0.7, color='green', label='良好条件数')
        ax1.bar(range(len(eigenvalues_bad)), eigenvalues_bad, 
                alpha=0.7, color='red', label='差条件数')
        ax1.set_xlabel('特征值索引')
        ax1.set_ylabel('特征值大小')
        ax1.set_title('矩阵特征值分布对比')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 条件数对解的影响
        n_points = 50
        x_true = np.random.randn(len(eigenvalues_good))
        y_good = K_good @ x_true + 0.01 * np.random.randn(len(eigenvalues_good))
        y_bad = K_bad @ x_true + 0.01 * np.random.randn(len(eigenvalues_bad))
        
        # 添加小扰动
        y_good_perturbed = y_good + 0.001 * np.random.randn(len(eigenvalues_good))
        y_bad_perturbed = y_bad + 0.001 * np.random.randn(len(eigenvalues_bad))
        
        # 求解
        x_good_original = np.linalg.solve(K_good, y_good)
        x_good_perturbed = np.linalg.solve(K_good, y_good_perturbed)
        x_bad_original = np.linalg.solve(K_bad, y_bad)
        x_bad_perturbed = np.linalg.solve(K_bad, y_bad_perturbed)
        
        relative_error_good = np.linalg.norm(x_good_perturbed - x_good_original) / np.linalg.norm(x_good_original)
        relative_error_bad = np.linalg.norm(x_bad_perturbed - x_bad_original) / np.linalg.norm(x_bad_original)
        
        ax2.bar(['良好条件', '差条件'], [relative_error_good, relative_error_bad],
                color=['green', 'red'], alpha=0.7)
        ax2.set_ylabel('相对误差')
        ax2.set_title('条件数对数值稳定性的影响')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 正则化效果
        lambdas = np.logspace(-4, 1, 50)
        condition_numbers = []
        
        for lam in lambdas:
            K_regularized = K_bad + lam * np.eye(len(K_bad))
            cond_num = np.linalg.cond(K_regularized)
            condition_numbers.append(cond_num)
        
        ax3.loglog(lambdas, condition_numbers, 'b-', linewidth=2)
        ax3.set_xlabel('正则化参数 λ')
        ax3.set_ylabel('条件数')
        ax3.set_title('正则化对条件数的改善')
        ax3.grid(True, alpha=0.3)
        
        # 标记推荐区域
        optimal_lambda_idx = np.argmin(condition_numbers)
        optimal_lambda = lambdas[optimal_lambda_idx]
        ax3.axvline(optimal_lambda, color='red', linestyle='--', 
                   label=f'最优λ ≈ {optimal_lambda:.2e}')
        ax3.legend()
        
        # 子图4: 泛化误差界
        sample_sizes = np.logspace(1, 3, 20)
        complexity_terms = 1 / np.sqrt(sample_sizes)
        empirical_errors = 0.1 * np.ones_like(sample_sizes)
        confidence_bounds = 0.05 * np.ones_like(sample_sizes)
        
        ax4.loglog(sample_sizes, empirical_errors, 'b-', label='经验误差', linewidth=2)
        ax4.loglog(sample_sizes, complexity_terms, 'g-', label='复杂度项', linewidth=2)
        ax4.loglog(sample_sizes, confidence_bounds, 'r-', label='置信界', linewidth=2)
        ax4.loglog(sample_sizes, empirical_errors + complexity_terms + confidence_bounds, 
                  'k--', label='泛化界', linewidth=2)
        
        ax4.set_xlabel('样本数量')
        ax4.set_ylabel('误差')
        ax4.set_title('泛化误差界的组成部分')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('condition_number_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印分析结果
        print("=== 条件数分析结果 ===")
        print(f"良好条件数矩阵的条件数: {np.linalg.cond(K_good):.2e}")
        print(f"差条件数矩阵的条件数: {np.linalg.cond(K_bad):.2e}")
        print(f"良好条件下的相对误差: {relative_error_good:.2e}")
        print(f"差条件下的相对误差: {relative_error_bad:.2e}")
        print(f"推荐的正则化参数: {optimal_lambda:.2e}")

def main():
    """主函数"""
    print("量子核岭回归数学原理可视化演示")
    print("=" * 40)
    
    viz = MathematicalVisualization()
    
    # 1. 希尔伯特空间映射可视化
    print("1. 生成希尔伯特空间映射图...")
    K_classical, K_quantum = viz.visualize_hilbert_space_mapping()
    print("   经典核矩阵:")
    print(K_classical)
    print("   量子核矩阵:")
    print(K_quantum)
    
    # 2. 优化景观可视化
    print("\n2. 生成优化景观图...")
    viz.visualize_optimization_landscape()
    
    # 3. 条件数分析
    print("\n3. 进行条件数分析...")
    viz.analyze_condition_number()
    
    print("\n✅ 所有数学原理可视化完成!")
    print("生成了三个分析图表，展示了量子核岭回归的核心数学概念。")

if __name__ == "__main__":
    main()