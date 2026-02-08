"""
PQC波束预测模型配置文件
"""

class PQCConfig:
    """PQC模型配置类"""
    
    # 数据配置
    DATA_PATH = '/Users/luxian/DataSpace/beam_pre/sls_beam_data_spatial_domain_vivo.mat'
    TOTAL_SAMPLES = 336000  # 正确的总样本数
    INPUT_FEATURES = 8      # 输入特征数
    OUTPUT_DIM = 256        # 输出维度
    
    # 模型配置
    N_QUBITS = 8           # 量子比特数
    N_LAYERS = 3           # 量子电路层数
    HIDDEN_DIM = 64        # 经典MLP隐藏层维度
    
    # 训练配置
    TRAIN_RATIO = 0.75     # 训练集比例
    VAL_RATIO = 0.15       # 验证集比例
    TEST_RATIO = 0.10      # 测试集比例
    
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10
    
    # 输出配置
    OUTPUT_DIR = 'pqc_ang_v1_output'
    
    @classmethod
    def get_data_splits(cls):
        """计算数据集划分"""
        train_size = int(cls.TOTAL_SAMPLES * cls.TRAIN_RATIO)
        val_size = int(cls.TOTAL_SAMPLES * cls.VAL_RATIO)
        test_size = cls.TOTAL_SAMPLES - train_size - val_size
        return train_size, val_size, test_size
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=== PQC模型配置 ===")
        print(f"总样本数: {cls.TOTAL_SAMPLES}")
        print(f"输入特征: {cls.INPUT_FEATURES}")
        print(f"输出维度: {cls.OUTPUT_DIM}")
        print(f"量子比特数: {cls.N_QUBITS}")
        print(f"量子层数: {cls.N_LAYERS}")
        print(f"批次大小: {cls.BATCH_SIZE}")
        print(f"训练轮数: {cls.EPOCHS}")
        print(f"学习率: {cls.LEARNING_RATE}")
        
        train_size, val_size, test_size = cls.get_data_splits()
        print(f"\n数据集划分:")
        print(f"训练集: {train_size} ({cls.TRAIN_RATIO*100:.1f}%)")
        print(f"验证集: {val_size} ({cls.VAL_RATIO*100:.1f}%)")
        print(f"测试集: {test_size} ({cls.TEST_RATIO*100:.1f}%)")