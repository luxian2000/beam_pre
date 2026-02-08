import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

class BeamDataAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = {}
        self.load_data()
    
    def load_data(self):
        """加载所有数据到内存"""
        print("正在加载数据...")
        with h5py.File(self.file_path, 'r') as f:
            for name in f.keys():
                if isinstance(f[name], h5py.Dataset):
                    self.data[name] = f[name][()]
        print("数据加载完成!")
    
    def get_basic_stats(self):
        """获取基本统计数据"""
        stats = {}
        for name, data in self.data.items():
            stats[name] = {
                'shape': data.shape,
                'dtype': data.dtype,
                'size_mb': data.nbytes / (1024 * 1024),
                'min_val': float(np.min(data)),
                'max_val': float(np.max(data)),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'memory_usage': f"{data.nbytes / (1024 * 1024):.2f} MB"
            }
        return stats
    
    def analyze_beam_patterns(self):
        """分析波束模式特征"""
        print("\n=== 波束模式分析 ===")
        
        # 分析角度数据分布
        angle_vars = ['Beam_Angle_BS_h', 'Beam_Angle_BS_v', 'Beam_Angle_UE_h', 'Beam_Angle_UE_v']
        
        for var in angle_vars:
            if var in self.data:
                data = self.data[var]
                print(f"\n{var}:")
                print(f"  范围: [{np.min(data):.2f}, {np.max(data):.2f}]")
                print(f"  唯一值数量: {len(np.unique(data))}")
                
                # 统计最常见的角度值
                unique_vals, counts = np.unique(data.flatten(), return_counts=True)
                top_angles = sorted(zip(unique_vals, counts), key=lambda x: x[1], reverse=True)[:5]
                print(f"  最常见角度值:")
                for angle, count in top_angles:
                    percentage = (count / data.size) * 100
                    print(f"    {angle:>8.2f}°: {count:>8,} 次 ({percentage:>5.2f}%)")
    
    def analyze_beam_ids(self):
        """分析波束ID分布"""
        print("\n=== 波束ID分析 ===")
        
        id_vars = ['Beam_ID_BS_h', 'Beam_ID_BS_v', 'Beam_ID_UE_h', 'Beam_ID_UE_v', 
                  'Beam_ID_tx', 'Beam_ID_rx', 'Beam_ID_Total']
        
        for var in id_vars:
            if var in self.data:
                data = self.data[var]
                print(f"\n{var}:")
                print(f"  范围: [{int(np.min(data))}, {int(np.max(data))}]")
                print(f"  唯一ID数量: {len(np.unique(data))}")
                
                # 统计ID分布
                unique_vals, counts = np.unique(data.flatten(), return_counts=True)
                print(f"  ID分布 (前5个最频繁):")
                for i, (id_val, count) in enumerate(sorted(zip(unique_vals, counts), 
                                                          key=lambda x: x[1], reverse=True)[:5]):
                    percentage = (count / data.size) * 100
                    print(f"    ID {int(id_val):>3}: {count:>8,} 次 ({percentage:>5.2f}%)")
    
    def analyze_rsrp(self):
        """分析RSRP信号强度"""
        print("\n=== RSRP信号强度分析 ===")
        
        if 'rsrp' in self.data:
            rsrp = self.data['rsrp']
            print(f"RSRP统计:")
            print(f"  范围: [{np.min(rsrp):.2f}, {np.max(rsrp):.2f}] dBm")
            print(f"  平均值: {np.mean(rsrp):.2f} dBm")
            print(f"  标准差: {np.std(rsrp):.2f} dBm")
            
            # 按信号强度分级统计
            bins = [-250, -150, -130, -110, -90, -70, -50, 0]
            labels = ['极弱(<-150)', '-150~-130', '-130~-110', '-110~-90', '-90~-70', '-70~-50', '较强(>-50)']
            hist, bin_edges = np.histogram(rsrp.flatten(), bins=bins)
            
            print(f"\n信号强度分布:")
            for i, (label, count) in enumerate(zip(labels, hist)):
                percentage = (count / rsrp.size) * 100
                print(f"  {label:>12}: {count:>10,} ({percentage:>6.2f}%)")
    
    def create_visualizations(self):
        """创建可视化图表"""
        print("\n=== 创建可视化图表 ===")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('波束数据可视化分析', fontsize=16)
        
        # 1. 角度分布直方图
        if 'Beam_Angle_BS_h' in self.data:
            angles_bs_h = self.data['Beam_Angle_BS_h'].flatten()
            axes[0,0].hist(angles_bs_h, bins=50, alpha=0.7, color='blue')
            axes[0,0].set_title('基站水平波束角度分布')
            axes[0,0].set_xlabel('角度 (度)')
            axes[0,0].set_ylabel('频次')
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. RSRP分布
        if 'rsrp' in self.data:
            rsrp_flat = self.data['rsrp'].flatten()
            axes[0,1].hist(rsrp_flat, bins=100, alpha=0.7, color='green')
            axes[0,1].set_title('RSRP信号强度分布')
            axes[0,1].set_xlabel('RSRP (dBm)')
            axes[0,1].set_ylabel('频次')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. 波束ID分布
        if 'Beam_ID_Total' in self.data:
            beam_ids = self.data['Beam_ID_Total'].flatten()
            unique_ids, counts = np.unique(beam_ids, return_counts=True)
            axes[1,0].bar(range(min(len(unique_ids), 20)), counts[:20], alpha=0.7, color='orange')
            axes[1,0].set_title('前20个波束ID使用频次')
            axes[1,0].set_xlabel('波束ID')
            axes[1,0].set_ylabel('使用次数')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. 信号强度空间分布（采样）
        if 'rsrp' in self.data:
            # 采样减少计算量
            sample_indices = np.random.choice(self.data['rsrp'].shape[0], 
                                            size=min(1000, self.data['rsrp'].shape[0]), 
                                            replace=False)
            rsrp_sample = self.data['rsrp'][sample_indices, :]
            im = axes[1,1].imshow(np.mean(rsrp_sample, axis=0).reshape(16, 16), 
                                cmap='viridis', aspect='auto')
            axes[1,1].set_title('平均RSRP空间分布 (16x16网格)')
            axes[1,1].set_xlabel('空间位置X')
            axes[1,1].set_ylabel('空间位置Y')
            plt.colorbar(im, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig('beam_data_analysis.png', dpi=300, bbox_inches='tight')
        print("图表已保存为 'beam_data_analysis.png'")
        plt.show()
    
    def export_summary(self):
        """导出分析摘要到CSV"""
        print("\n=== 导出分析摘要 ===")
        
        summary_data = []
        
        # 基本信息
        basic_stats = self.get_basic_stats()
        for var_name, stats in basic_stats.items():
            summary_data.append({
                '变量名': var_name,
                '形状': str(stats['shape']),
                '数据类型': str(stats['dtype']),
                '内存占用(MB)': stats['size_mb'],
                '最小值': stats['min_val'],
                '最大值': stats['max_val'],
                '平均值': stats['mean'],
                '标准差': stats['std']
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv('beam_data_summary.csv', index=False, encoding='utf-8-sig')
        print("摘要已导出到 'beam_data_summary.csv'")

def main():
    # 分析指定的MAT文件
    mat_file_path = "/Users/luxian/DataSpace/beam_pre/sls_beam_data_spatial_domain_vivo.mat"
    
    # 创建分析器实例
    analyzer = BeamDataAnalyzer(mat_file_path)
    
    # 执行各种分析
    print("=" * 60)
    print("BEAM DATA STRUCTURE ANALYSIS REPORT")
    print("=" * 60)
    
    # 基本统计
    stats = analyzer.get_basic_stats()
    print("\n数据集概览:")
    print("-" * 40)
    total_memory = sum(stat['size_mb'] for stat in stats.values())
    print(f"总变量数: {len(stats)}")
    print(f"总内存占用: {total_memory:.2f} MB")
    print(f"数据记录数: {stats['rsrp']['shape'][0]:,}")
    print(f"每记录特征数: {stats['rsrp']['shape'][1]:,}")
    
    # 详细分析
    analyzer.analyze_beam_patterns()
    analyzer.analyze_beam_ids()
    analyzer.analyze_rsrp()
    
    # 可视化（可选）
    try:
        analyzer.create_visualizations()
    except Exception as e:
        print(f"可视化创建失败: {e}")
    
    # 导出摘要
    analyzer.export_summary()
    
    print("\n" + "=" * 60)
    print("分析完成! 生成的文件:")
    print("- beam_data_analysis.png (可视化图表)")
    print("- beam_data_summary.csv (数据摘要)")

if __name__ == "__main__":
    main()