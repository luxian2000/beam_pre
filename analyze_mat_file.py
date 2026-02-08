import h5py
import numpy as np
import os

def analyze_hdf5_structure(name, obj, indent=0):
    """递归分析HDF5文件结构"""
    spaces = "  " * indent
    
    if isinstance(obj, h5py.Dataset):
        print(f"{spaces}DATASET: '{name}'")
        print(f"{spaces}  Shape: {obj.shape}")
        print(f"{spaces}  Data type: {obj.dtype}")
        print(f"{spaces}  Size: {obj.size:,} elements")
        size_mb = obj.size * obj.dtype.itemsize / (1024 * 1024)
        print(f"{spaces}  Memory: {size_mb:.2f} MB")
        
        # 显示一些样本数据
        if obj.size > 0:
            if obj.ndim <= 2:
                # 对于低维数组，显示实际数据
                data = obj[()]
                if np.issubdtype(data.dtype, np.number):
                    print(f"{spaces}  Range: [{np.min(data):.6f}, {np.max(data):.6f}]")
                    print(f"{spaces}  Mean: {np.mean(data):.6f}")
                    print(f"{spaces}  Std: {np.std(data):.6f}")
                print(f"{spaces}  Sample data: {data.flat[:min(10, data.size)]}")
            else:
                # 对于高维数组，只显示结构信息
                print(f"{spaces}  High-dimensional array, showing slice info...")
                if obj.ndim == 3:
                    print(f"{spaces}  3D slice [0,:5,:5]:")
                    slice_data = obj[0, :min(5, obj.shape[1]), :min(5, obj.shape[2])]
                    print(f"{spaces}    {slice_data}")
                elif obj.ndim == 4:
                    print(f"{spaces}  4D slice [0,0,:5,:5]:")
                    slice_data = obj[0, 0, :min(5, obj.shape[2]), :min(5, obj.shape[3])]
                    print(f"{spaces}    {slice_data}")
                    
    elif isinstance(obj, h5py.Group):
        print(f"{spaces}GROUP: '{name}'")
        print(f"{spaces}  Members: {len(obj)}")
        for key in obj.keys():
            analyze_hdf5_structure(key, obj[key], indent + 1)

def analyze_mat_file_h5py(file_path):
    """
    使用h5py分析MATLAB v7.3文件(.mat)的数据结构
    """
    print(f"正在分析文件: {file_path}")
    print("=" * 60)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return
    
    try:
        # 打开HDF5文件
        with h5py.File(file_path, 'r') as f:
            print("文件基本信息:")
            print(f"- 文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
            print(f"- HDF5版本: {f.libver}")
            print(f"- 根组成员数: {len(f)}")
            print()
            
            print("文件结构分析:")
            print("-" * 40)
            
            # 分析根级别的所有对象
            for name in f.keys():
                analyze_hdf5_structure(name, f[name])
            
            print("\n" + "=" * 60)
            print("数据访问示例代码:")
            print("-" * 30)
            
            # 生成数据访问示例
            print("# Python中访问数据的示例:")
            print("import h5py")
            print("import numpy as np")
            print(f"f = h5py.File('{file_path}', 'r')")
            
            for name in f.keys():
                obj = f[name]
                if isinstance(obj, h5py.Dataset):
                    print(f"# data = f['{name}'][()]  # 加载完整数据")
                    print(f"# 或者按需加载部分数据:")
                    if obj.ndim == 2:
                        print(f"# subset = f['{name}'][:100, :100]  # 加载前100x100元素")
                    elif obj.ndim == 3:
                        print(f"# subset = f['{name}'][0, :100, :100]  # 加载第一层的前100x100元素")
                    elif obj.ndim >= 4:
                        print(f"# subset = f['{name}'][0, 0, :50, :50]  # 加载前两维的第一个索引")
                    break  # 只显示第一个数据集的示例
            
            print("f.close()")
            
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 分析指定的MAT文件
    mat_file_path = "/Users/luxian/DataSpace/beam_pre/sls_beam_data_spatial_domain_vivo.mat"
    analyze_mat_file_h5py(mat_file_path)