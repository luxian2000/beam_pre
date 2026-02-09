import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameter configuration
HYPERPARAMETERS = {
    # Data configuration
    'TRAIN_START': 0,
    'TRAIN_END': 2048,  # Directly specify training samples count
    'EPOCHS': 5,  # Reduced to 5 epochs for quick testing
    'BATCH_SIZE': 32,
    'TEST_RATIO': 0.2,  # Test set accounts for 1/5 of training set
    
    # Top-N accuracy calculation configuration
    'TOP_N_MAX': 10,  # Calculate Top-1 to Top-10 accuracy
    
    # Early Stopping configuration
    'EARLY_STOPPING_PATIENCE': 20,
    'EARLY_STOPPING_MIN_DELTA': 1e-4,
    'EARLY_STOPPING_MONITOR': 'test_loss',
    
    # Model configuration
    'N_QUBITS': 12,
    'N_LAYERS': 3,
    'INPUT_DIM': 48,
    'TOTAL_FEATURES': 256,  # Total features
    
    # MLR (classic MLP post-processing) structure configuration
    'MLR_HIDDEN_DIM': 64,  # MLR hidden layer dimension
    'MLR_ACTIVATION': 'ReLU',  # MLR activation function
    
    # Training configuration
    'LEARNING_RATE': 0.001,
    'SHUFFLE_TRAIN': False,
    
    # Other configuration
    'DATA_PATH': '/Users/luxian/DataSpace/beam_pre/sls_beam_data_spatial_domain_vivo.mat',
    'OUTPUT_DIR': 'pqc_reup_v1_output'
}

# Calculate derived parameters (TRAIN_END is now fixed, not calculated)
HYPERPARAMETERS['TRAIN_SAMPLES'] = HYPERPARAMETERS['TRAIN_END'] - HYPERPARAMETERS['TRAIN_START']
HYPERPARAMETERS['OUTPUT_DIM'] = HYPERPARAMETERS['TOTAL_FEATURES'] - HYPERPARAMETERS['INPUT_DIM']
HYPERPARAMETERS['TEST_SAMPLES'] = int(HYPERPARAMETERS['TRAIN_SAMPLES'] * HYPERPARAMETERS['TEST_RATIO'])
HYPERPARAMETERS['TEST_START'] = HYPERPARAMETERS['TRAIN_END']
HYPERPARAMETERS['TEST_END'] = HYPERPARAMETERS['TEST_START'] + HYPERPARAMETERS['TEST_SAMPLES']

print(f"Dual Top-N Methods Test Configuration:")
print(f"Input beam count: {HYPERPARAMETERS['INPUT_DIM']}")
print(f"Output beam count: {HYPERPARAMETERS['OUTPUT_DIM']}")
print(f"Total beam count: {HYPERPARAMETERS['TOTAL_FEATURES']}")
print(f"Training samples: {HYPERPARAMETERS['TRAIN_SAMPLES']}")
print(f"Testing samples: {HYPERPARAMETERS['TEST_SAMPLES']}")
print(f"Calculate both Top-N methods: A(including input beams) vs B(excluding input beams)")
print(f"Early Stopping configuration:")
print(f"  Patience: {HYPERPARAMETERS['EARLY_STOPPING_PATIENCE']} epochs")
print(f"  Min Delta: {HYPERPARAMETERS['EARLY_STOPPING_MIN_DELTA']}")
print(f"  Monitor: {HYPERPARAMETERS['EARLY_STOPPING_MONITOR']}")
print(f"Configuration verification:")
print(f"  TRAIN_END is directly specified as {HYPERPARAMETERS['TRAIN_END']} samples")
print(f"  TRAIN_START: {HYPERPARAMETERS['TRAIN_START']}")
print(f"  TRAIN_SAMPLES: {HYPERPARAMETERS['TRAIN_SAMPLES']}")
print(f"  TEST_START: {HYPERPARAMETERS['TEST_START']}")
print(f"  TEST_END: {HYPERPARAMETERS['TEST_END']}")
print(f"MLR structure: {HYPERPARAMETERS['N_QUBITS']} → {HYPERPARAMETERS['MLR_HIDDEN_DIM']} → {HYPERPARAMETERS['OUTPUT_DIM']}")

class QuantumDataReuploadModel(nn.Module):
    """Quantum beam prediction model based on data re-upload technology"""
    
    def __init__(self, n_qubits=HYPERPARAMETERS['N_QUBITS'], n_layers=HYPERPARAMETERS['N_LAYERS'], 
                 input_dim=HYPERPARAMETERS['INPUT_DIM'], output_dim=HYPERPARAMETERS['OUTPUT_DIM']):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.chunk_size = 4  # Each chunk has 4 features
        self.n_chunks = input_dim // self.chunk_size  # 12 chunks
        
        # Data re-upload parameters: map 4D to 3D weights and biases
        self.reupload_weights = nn.Parameter(torch.randn(self.n_chunks, 3, self.chunk_size) * 0.1)
        self.reupload_bias = nn.Parameter(torch.randn(self.n_chunks, 3) * 0.1)
        
        # Quantum device and circuit
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qnode = qml.QNode(self.quantum_circuit, self.dev, interface="torch")
        
        # MLR (Multi-Layer Regression) post-processing layer
        # Structure: quantum output(n_qubits) -> hidden layer(MLR_HIDDEN_DIM) -> output layer(output_dim)
        self.mlr = nn.Sequential(
            nn.Linear(n_qubits, HYPERPARAMETERS['MLR_HIDDEN_DIM']),
            nn.ReLU(),
            nn.Linear(HYPERPARAMETERS['MLR_HIDDEN_DIM'], output_dim)
        )
        
    def quantum_circuit(self, params):
        """Quantum circuit definition"""
        # Strong entanglement layer
        for layer in range(self.n_layers):
            # Data re-upload
            for i in range(self.n_chunks):
                chunk_params = params[i]
                # U3 gate encoding
                qml.U3(chunk_params[0], chunk_params[1], chunk_params[2], wires=i)
            
            # Strong entanglement operation
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                
        # Measure Pauli-Z expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def forward(self, x):
        """Forward propagation"""
        batch_size = x.shape[0]
        quantum_outputs = []
        
        # Process each sample
        for i in range(batch_size):
            sample = x[i]
            
            # Split input into 12 chunks of 4D
            chunks = sample.view(self.n_chunks, self.chunk_size)
            
            # Data re-upload: map each chunk to 3D parameters
            processed_chunks = []
            for j in range(self.n_chunks):
                chunk = chunks[j]
                # Linear transformation: W * x + b -> 3D
                transformed = torch.matmul(self.reupload_weights[j], chunk) + self.reupload_bias[j]
                processed_chunks.append(transformed)
            
            # Quantum circuit processing
            quantum_params = torch.stack(processed_chunks)
            q_out = self.qnode(quantum_params)
            quantum_outputs.append(torch.stack(q_out))
        
        # Stack batch results
        quantum_output = torch.stack(quantum_outputs)
        
        # Ensure data type is float32
        quantum_output = quantum_output.float()
        
        # MLR post-processing
        final_output = self.mlr(quantum_output)
        
        return final_output

def load_and_preprocess_data(filepath, train_start=HYPERPARAMETERS['TRAIN_START'], 
                           train_end=HYPERPARAMETERS['TRAIN_END']):
    """Load and preprocess data, divide into training and test sets by specified range"""
    # Automatically calculate test set range
    train_samples = train_end - train_start
    test_samples = int(train_samples * HYPERPARAMETERS['TEST_RATIO'])
    test_start = train_end
    test_end = test_start + test_samples
    
    print(f"Data division: Training set[{train_start}:{train_end}] ({train_samples} samples), "
          f"Test set[{test_start}:{test_end}] ({test_samples} samples)")
    
    # Use h5py to load MATLAB v7.3 file
    try:
        with h5py.File(filepath, 'r') as f:
            # Try common data key names
            data_keys = ['beam_data', 'data', 'X', 'features', 'rsrp']
            data = None
            
            for key in data_keys:
                if key in f:
                    data = np.array(f[key]).T  # MATLAB storage is usually transposed
                    break
            
            # If no common key names are found, try the first dataset
            if data is None:
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        data = np.array(f[key]).T
                        break
            
            if data is None:
                raise ValueError("No valid dataset found in HDF5 file")
                
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    # Ensure data is 2D
    if len(data.shape) > 2:
        data = data.reshape(data.shape[0], -1)
    
    # Check if index range is valid (now in sample dimension)
    total_samples = data.shape[1]  # Sample count in the second dimension
    if train_end > total_samples or test_end > total_samples:
        # Adjust to valid range
        train_end = min(train_end, total_samples)
        test_end = min(test_end, total_samples)
        print(f"Adjusted range: Training set[{train_start}:{train_end}], Test set[{test_start}:{test_end}]")
    
    # Select samples by specified range (in the second dimension)
    train_indices = np.arange(train_start, train_end)
    test_indices = np.arange(test_start, test_end)
    
    # Select samples
    X_train_full = data[:, train_indices].T  # Transpose to (sample count, feature count)
    X_test_full = data[:, test_indices].T
    
    # Equidistantly select 48 features (in the first dimension)
    n_features = data.shape[0]  # Feature count
    input_indices = np.linspace(0, n_features-1, HYPERPARAMETERS['INPUT_DIM'], dtype=int)
    
    X_train_input = X_train_full[:, input_indices]
    X_test_input = X_test_full[:, input_indices]
    
    # Output is all remaining features
    all_indices = set(range(n_features))
    output_indices = list(all_indices - set(input_indices))
    X_train_output = X_train_full[:, output_indices]
    X_test_output = X_test_full[:, output_indices]
    
    # Data normalization
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_input_norm = scaler_X.fit_transform(X_train_input)
    X_test_input_norm = scaler_X.transform(X_test_input)
    
    X_train_output_norm = scaler_y.fit_transform(X_train_output)
    X_test_output_norm = scaler_y.transform(X_test_output)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_input_norm)
    y_train_tensor = torch.FloatTensor(X_train_output_norm)
    X_test_tensor = torch.FloatTensor(X_test_input_norm)
    y_test_tensor = torch.FloatTensor(X_test_output_norm)
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            scaler_X, scaler_y, input_indices, output_indices, train_indices, test_indices)

def train_model(model, train_loader, test_loader, 
                epochs=HYPERPARAMETERS['EPOCHS'], lr=HYPERPARAMETERS['LEARNING_RATE']):
    """Train model, including Early Stopping mechanism"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    mae_scores = []
    
    # Early Stopping related variables
    best_test_loss = float('inf')
    patience_counter = 0
    early_stop_triggered = False
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        batch_count = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
            
            # Remove batch limit, train on full dataset
            # Original limit: if batch_idx >= 2: break
        
        if batch_count > 0:
            train_loss /= batch_count
        train_losses.append(train_loss)
        
        # Testing phase
        model.eval()
        test_loss = 0.0
        mae_score = 0.0
        test_batch_count = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += criterion(output, target).item()
                mae_score += mean_absolute_error(target.numpy(), output.numpy())
                test_batch_count += 1
        
        if test_batch_count > 0:
            test_loss /= test_batch_count
            mae_score /= test_batch_count
        test_losses.append(test_loss)
        mae_scores.append(mae_score)
        
        # Early Stopping check - decide whether to stop based on monitored metric
        current_monitor_value = test_loss  # Current monitored test loss
        
        if current_monitor_value < best_test_loss - HYPERPARAMETERS['EARLY_STOPPING_MIN_DELTA']:
            best_test_loss = current_monitor_value
            patience_counter = 0
            # Can save the best model here
        else:
            patience_counter += 1
            
        # Check if early stopping is triggered
        if patience_counter >= HYPERPARAMETERS['EARLY_STOPPING_PATIENCE']:
            print(f"Epoch {epoch}: Early stopping triggered! {HYPERPARAMETERS['EARLY_STOPPING_MONITOR']} hasn't improved for {HYPERPARAMETERS['EARLY_STOPPING_PATIENCE']} epochs.")
            early_stop_triggered = True
            break
        
        if epoch % 10 == 0:  # Reduce print frequency to every 10 epochs
            status = " (Early Stop)" if early_stop_triggered else ""
            print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, MAE: {mae_score:.6f}{status}')
    
    # Report Early Stopping status after training is complete
    if early_stop_triggered:
        print(f"Training stopped early, total training rounds: {len(train_losses)}")
        print(f"Best {HYPERPARAMETERS['EARLY_STOPPING_MONITOR']}: {best_test_loss:.6f}")
    else:
        print(f"Training completed {len(train_losses)} rounds, Early Stopping not triggered")
    
    return train_losses, test_losses, mae_scores

def evaluate_model(model, test_loader, scaler_y):
    """Evaluate model performance on test set"""
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            predictions.extend(output.numpy())
            targets.extend(target.numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Inverse transform predictions and targets
    predictions_original = scaler_y.inverse_transform(predictions)
    targets_original = scaler_y.inverse_transform(targets)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(targets_original, predictions_original)
    mae = mean_absolute_error(targets_original, predictions_original)
    rmse = np.sqrt(mse)
    r2 = 1 - (mse / np.var(targets_original))
    
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }
    
    return predictions_original, targets_original, metrics

def main():
    """Main function"""
    # Create output directory
    output_dir = HYPERPARAMETERS['OUTPUT_DIR']
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Data file path
    data_path = HYPERPARAMETERS['DATA_PATH']
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file does not exist: {data_path}")
        return
    
    # Load and preprocess data - divide by specified range
    (X_train, y_train, X_test, y_test, scaler_X, scaler_y, 
     input_indices, output_indices, train_indices, test_indices) = load_and_preprocess_data(
        data_path, 
        train_start=HYPERPARAMETERS['TRAIN_START'], 
        train_end=HYPERPARAMETERS['TRAIN_END']
    )
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                             batch_size=HYPERPARAMETERS['BATCH_SIZE'], 
                                             shuffle=HYPERPARAMETERS['SHUFFLE_TRAIN'])
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                            batch_size=HYPERPARAMETERS['BATCH_SIZE'], 
                                            shuffle=False)
    
    # Create model
    model = QuantumDataReuploadModel(
        n_qubits=HYPERPARAMETERS['N_QUBITS'], 
        n_layers=HYPERPARAMETERS['N_LAYERS'], 
        input_dim=HYPERPARAMETERS['INPUT_DIM'], 
        output_dim=HYPERPARAMETERS['OUTPUT_DIM']
    )
    
    print(f"开始训练: {len(X_train)}训练样本, {len(X_test)}测试样本")
    print(f"输入维度: {X_train.shape[1]}, 输出维度: {y_train.shape[1]}")
    print(f"MLR结构: {HYPERPARAMETERS['N_QUBITS']} → {HYPERPARAMETERS['MLR_HIDDEN_DIM']} → {HYPERPARAMETERS['OUTPUT_DIM']}")
    print(f"训练样本数: {HYPERPARAMETERS['TRAIN_SAMPLES']}")
    print(f"输入波束数量: {len(input_indices)}, 输出波束数量: {len(output_indices)}")
    
    # 训练模型
    print("开始训练模型...")
    train_losses, test_losses, mae_scores = train_model(model, train_loader, test_loader)
    
    # 保存训练过程数据
    training_data = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'mae_scores': mae_scores,
        'epochs': len(train_losses)
    }
    
    # 保存模型参数
    model_save_path = os.path.join(output_dir, f'model_params_epoch_{len(train_losses)}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"模型参数已保存到: {model_save_path}")
    
    # 保存训练过程数据
    training_data_path = os.path.join(output_dir, f'training_data_epoch_{len(train_losses)}.json')
    with open(training_data_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"训练过程数据已保存到: {training_data_path}")
    
    # 评估模型
    print("评估模型性能...")
    predictions_original, targets_original, metrics = evaluate_model(
        model, test_loader, scaler_y
    )
    
    # 保存评估结果
    results = {
        'metrics': metrics,
        'predictions': predictions_original.tolist(),
        'targets': targets_original.tolist()
    }
    
    results_path = os.path.join(output_dir, f'evaluation_results_epoch_{len(train_losses)}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"评估结果已保存到: {results_path}")
    
    # 调用分析模块生成图像
    print("调用分析模块生成结果图像...")
    try:
        import pqc_reup_analyze
        pqc_reup_analyze.analyze_results(
            epoch_num=len(train_losses),
            predictions=predictions_original,
            targets=targets_original,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"图像生成失败: {e}")
        print("继续执行其他操作...")
    
    # 保存模型和其他必要文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f'model_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)
    
    # 保存评估指标（简化版本）
    metrics = {
        'MSE': 0.0,
        'MAE': 0.0,
        'RMSE': 0.0,
        'R2': 0.0
    }
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n训练完成!")
    print(f"模型已保存到: {model_path}")
    print(f"评估指标已保存到: {metrics_path}")

if __name__ == "__main__":
    main()