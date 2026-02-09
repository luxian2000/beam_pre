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
    # TRAIN_END will be automatically calculated based on EPOCHS * BATCH_SIZE
    'EPOCHS': 400,
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

# Automatically calculate related parameters
HYPERPARAMETERS['TRAIN_END'] = HYPERPARAMETERS['TRAIN_START'] + HYPERPARAMETERS['EPOCHS'] * HYPERPARAMETERS['BATCH_SIZE']
HYPERPARAMETERS['OUTPUT_DIM'] = HYPERPARAMETERS['TOTAL_FEATURES'] - HYPERPARAMETERS['INPUT_DIM']
HYPERPARAMETERS['TRAIN_SAMPLES'] = HYPERPARAMETERS['TRAIN_END'] - HYPERPARAMETERS['TRAIN_START']
HYPERPARAMETERS['TEST_SAMPLES'] = int(HYPERPARAMETERS['TRAIN_SAMPLES'] * HYPERPARAMETERS['TEST_RATIO'])
HYPERPARAMETERS['TEST_START'] = HYPERPARAMETERS['TRAIN_END']
HYPERPARAMETERS['TEST_END'] = HYPERPARAMETERS['TEST_START'] + HYPERPARAMETERS['TEST_SAMPLES']

print(f"Dual Top-N Methods Test Configuration:")
print(f"Input beam count: {HYPERPARAMETERS['INPUT_DIM']}")
print(f"Output beam count: {HYPERPARAMETERS['OUTPUT_DIM']}")
print(f"Total beam count: {HYPERPARAMETERS['TOTAL_FEATURES']}")
print(f"Calculate both Top-N methods: A(including input beams) vs B(excluding input beams)")
print(f"Early Stopping configuration:")
print(f"  Patience: {HYPERPARAMETERS['EARLY_STOPPING_PATIENCE']} epochs")
print(f"  Min Delta: {HYPERPARAMETERS['EARLY_STOPPING_MIN_DELTA']}")
print(f"  Monitor: {HYPERPARAMETERS['EARLY_STOPPING_MONITOR']}")
print(f"Automatically calculated verification:")
print(f"EPOCHS: {HYPERPARAMETERS['EPOCHS']}")
print(f"BATCH_SIZE: {HYPERPARAMETERS['BATCH_SIZE']}")
print(f"Automatically calculated TRAIN_END: {HYPERPARAMETERS['TRAIN_END']} ({HYPERPARAMETERS['EPOCHS']} × {HYPERPARAMETERS['BATCH_SIZE']})")
print(f"Training sample count: {HYPERPARAMETERS['TRAIN_SAMPLES']}")
print(f"Test sample count: {HYPERPARAMETERS['TEST_SAMPLES']} (20% of training set)")
print(f"Test range: [{HYPERPARAMETERS['TEST_START']}, {HYPERPARAMETERS['TEST_END']}]")
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
        
        if epoch % 20 == 0:  # Keep original print frequency
            status = " (Early Stop)" if early_stop_triggered else ""
            print(f'Epoch {epoch}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, MAE: {mae_score:.6f}{status}')
    
    # Report Early Stopping status after training is complete
    if early_stop_triggered:
        print(f"Training stopped early, total training rounds: {len(train_losses)}")
        print(f"Best {HYPERPARAMETERS['EARLY_STOPPING_MONITOR']}: {best_test_loss:.6f}")
    else:
        print(f"Training completed {len(train_losses)} rounds, Early Stopping not triggered")
    
    return train_losses, test_losses, mae_scores

def calculate_top_n_accuracy_both_methods(predictions, targets, input_indices, top_n_max=10):
    """Calculate two Top-N accuracies: including input beams and excluding input beams"""
    n_samples = len(predictions)
    
    # Initialize accuracy counters for both methods
    top_n_correct_with_input = [0] * top_n_max    # Method A: including input beams
    top_n_correct_without_input = [0] * top_n_max # Method B: excluding input beams
    
    # Create output beam index set (excluding input beams)
    all_indices = set(range(len(predictions[0])))  # All beam indices
    output_indices_set = all_indices - set(input_indices)  # Indices excluding input beams
    
    for i in range(n_samples):
        pred_sample = predictions[i]
        target_sample = targets[i]
        
        # Method A: statistics including input beams
        pred_indices_A = np.argsort(pred_sample)[::-1]  # All beams in descending order
        target_max_idx_A = np.argmax(target_sample)     # Index of true maximum value
        
        # Method B: statistics excluding input beams
        pred_values_B = pred_sample[list(output_indices_set)]
        target_values_B = target_sample[list(output_indices_set)]
        
        # Get sorting indices within output beams
        pred_local_indices_B = np.argsort(pred_values_B)[::-1]  # Output beams in descending order
        target_local_max_idx_B = np.argmax(target_values_B)     # Index of true maximum value within output beams
        
        # Map local indices to global indices
        output_indices_list = list(output_indices_set)
        pred_global_indices_B = [output_indices_list[idx] for idx in pred_local_indices_B]
        target_global_max_idx_B = output_indices_list[target_local_max_idx_B]
        
        # Calculate Top-N accuracies for both methods
        for n in range(1, top_n_max + 1):
            # Method A: check if true optimal beam is in the top N predictions (including all beams)
            if target_max_idx_A in pred_indices_A[:n]:
                top_n_correct_with_input[n-1] += 1
            
            # Method B: check if true optimal beam is in the top N predictions (only output beams)
            if target_global_max_idx_B in pred_global_indices_B[:n]:
                top_n_correct_without_input[n-1] += 1
    
    # Calculate accuracies
    top_n_accuracies_with_input = [correct / n_samples for correct in top_n_correct_with_input]
    top_n_accuracies_without_input = [correct / n_samples for correct in top_n_correct_without_input]
    
    return {
        'with_input': top_n_accuracies_with_input,      # Method A
        'without_input': top_n_accuracies_without_input  # Method B
    }

def evaluate_model_with_top_n(model, test_loader, scaler_y, input_indices, top_n_max=10):
    """Evaluate model and calculate two Top-N accuracies"""
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
    
    # Inverse normalization
    predictions_original = scaler_y.inverse_transform(predictions)
    targets_original = scaler_y.inverse_transform(targets)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(targets_original, predictions_original)
    mae = mean_absolute_error(targets_original, predictions_original)
    rmse = np.sqrt(mse)
    
    # Calculate R² score
    ss_res = np.sum((targets_original - predictions_original) ** 2)
    ss_tot = np.sum((targets_original - np.mean(targets_original)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Calculate two Top-N accuracies
    top_n_results = calculate_top_n_accuracy_both_methods(predictions_original, targets_original, input_indices, top_n_max)
    
    metrics = {
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'top_n_accuracies': top_n_results
    }
    
    return predictions_original, targets_original, metrics

def plot_results(predictions, targets, train_losses, test_losses, mae_scores, top_n_results, save_path, timestamp):
    """Plot results charts with both Top-N accuracy methods, using timestamp naming"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Training curves
    axes[0, 0].plot(train_losses, label='Training Loss')
    axes[0, 0].plot(test_losses, label='Test Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Test Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # MAE curve
    axes[0, 1].plot(mae_scores, 'g-', label='MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Mean Absolute Error Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Both Top-N accuracy comparison curves
    n_values = list(range(1, len(top_n_results['with_input']) + 1))
    
    # Method A: Including input beams
    axes[0, 2].plot(n_values, top_n_results['with_input'], 'bo-', linewidth=2, markersize=6, label='Including Input Beams')
    
    # Method B: Excluding input beams
    axes[0, 2].plot(n_values, top_n_results['without_input'], 'ro-', linewidth=2, markersize=6, label='Excluding Input Beams')
    
    axes[0, 2].set_xlabel('N')
    axes[0, 2].set_ylabel('Top-N Accuracy')
    axes[0, 2].set_title('Top-N Accuracy Comparison')
    axes[0, 2].set_xticks(n_values)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Predictions vs True Values scatter plot (sample display)
    sample_indices = np.random.choice(len(predictions), min(1000, len(predictions)), replace=False)
    axes[1, 0].scatter(targets[sample_indices].flatten(), predictions[sample_indices].flatten(), alpha=0.5)
    axes[1, 0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    axes[1, 0].set_xlabel('True Values')
    axes[1, 0].set_ylabel('Predictions')
    axes[1, 0].set_title('Predictions vs True Values (Sample)')
    axes[1, 0].grid(True)
    
    # Error distribution
    errors = (predictions - targets).flatten()
    axes[1, 1].hist(errors, bins=50, alpha=0.7)
    axes[1, 1].set_xlabel('Prediction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].grid(True)
    
    # Top-N accuracy comparison table
    axes[1, 2].axis('tight')
    axes[1, 2].axis('off')
    table_data = [['Method', 'Top-N', 'Accuracy']]
    
    # Add Method A data
    for i, acc in enumerate(top_n_results['with_input']):
        table_data.append(['With Input', f'Top-{i+1}', f'{acc:.4f} ({acc*100:.2f}%)'])
    
    # Add separator row
    table_data.append(['', '', ''])
    
    # Add Method B data
    for i, acc in enumerate(top_n_results['without_input']):
        table_data.append(['Without Input', f'Top-{i+1}', f'{acc:.4f} ({acc*100:.2f}%)'])
    
    table = axes[1, 2].table(cellText=table_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    axes[1, 2].set_title('Top-N Accuracy Comparison Summary')
    
    plt.tight_layout()
    # Use specified filename format
    results_filename = f'results_{timestamp}.png'
    plt.savefig(os.path.join(save_path, results_filename), dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_filename

def save_experiment_config(output_dir, metrics, train_losses, test_losses, mae_scores, timestamp):
    """Save experiment configuration and results to document file, using timestamp naming"""
    
    # Create configuration document
    config_content = f"""# PQC_REUP_V1 Experiment Configuration and Results

## Experiment Basic Information
- **Run Time**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Timestamp**: {timestamp}
- **Script Name**: pqc_reup_v1.py
- **Output Directory**: {output_dir}

## Hyperparameter Configuration

### Data Configuration
- Training sample range: [{HYPERPARAMETERS['TRAIN_START']}, {HYPERPARAMETERS['TRAIN_END']}]
- Test sample range: [{HYPERPARAMETERS['TEST_START']}, {HYPERPARAMETERS['TEST_END']}]
- Training sample count: {HYPERPARAMETERS['TRAIN_SAMPLES']}
- Test sample count: {HYPERPARAMETERS['TEST_SAMPLES']} (automatically calculated as {HYPERPARAMETERS['TEST_RATIO']*100}% of training set)
- Test set ratio: {HYPERPARAMETERS['TEST_RATIO']}
- Data file path: {HYPERPARAMETERS['DATA_PATH']}

### Model Configuration
- Total features: {HYPERPARAMETERS['TOTAL_FEATURES']}
- Quantum qubits: {HYPERPARAMETERS['N_QUBITS']}
- Quantum circuit layers: {HYPERPARAMETERS['N_LAYERS']}
- Input dimension: {HYPERPARAMETERS['INPUT_DIM']}
- Output dimension: {HYPERPARAMETERS['OUTPUT_DIM']} (automatically calculated: {HYPERPARAMETERS['TOTAL_FEATURES']} - {HYPERPARAMETERS['INPUT_DIM']})
- Chunk size: 4
- Chunk count: {HYPERPARAMETERS['INPUT_DIM'] // 4}

### MLR (Multi-Layer Regression) Structure Configuration
- Hidden layer dimension: {HYPERPARAMETERS['MLR_HIDDEN_DIM']}
- Activation function: {HYPERPARAMETERS['MLR_ACTIVATION']}
- Network structure: Quantum output({HYPERPARAMETERS['N_QUBITS']}) → Hidden layer({HYPERPARAMETERS['MLR_HIDDEN_DIM']}) → Output layer({HYPERPARAMETERS['OUTPUT_DIM']})

### Training Configuration
- Training epochs: {HYPERPARAMETERS['EPOCHS']}
- Batch size: {HYPERPARAMETERS['BATCH_SIZE']}
- Learning rate: {HYPERPARAMETERS['LEARNING_RATE']}
- Shuffle training data: {HYPERPARAMETERS['SHUFFLE_TRAIN']}

### Early Stopping Configuration
- Patience epochs: {HYPERPARAMETERS['EARLY_STOPPING_PATIENCE']}
- Minimum improvement threshold: {HYPERPARAMETERS['EARLY_STOPPING_MIN_DELTA']}
- Monitor metric: {HYPERPARAMETERS['EARLY_STOPPING_MONITOR']}

## Training Results

### Final Evaluation Metrics
- **MSE**: {metrics['MSE']:.6f}
- **MAE**: {metrics['MAE']:.6f}
- **RMSE**: {metrics['RMSE']:.6f}
- **R²**: {metrics['R2']:.6f}

### Performance Level Evaluation
"""
    
    # Performance level judgment
    if metrics['R2'] > 0.75:
        performance_level = "Excellent"
    elif metrics['R2'] >= 0.6:
        performance_level = "Good"
    elif metrics['R2'] >= 0.3:
        performance_level = "Average"
    elif metrics['R2'] >= 0.1:
        performance_level = "Poor"
    else:
        performance_level = "Very Poor"
    
    config_content += f"- **Performance Level**: {performance_level}\n\n"
    
    # Training process summary
    config_content += f"""### Training Process Summary
- Final training loss: {train_losses[-1]:.6f}
- Final test loss: {test_losses[-1]:.6f}
- Final MAE: {mae_scores[-1]:.6f}
- Training epochs: {len(train_losses)}

### Top-N Accuracy Results

#### Method A: Including Input Beam Itself
"""
    
    # Method A results
    for i, acc in enumerate(metrics['top_n_accuracies']['with_input']):
        config_content += f"- Top-{i+1}: {acc:.4f} ({acc*100:.2f}%)\n"
    
    config_content += "\n#### Method B: Excluding Input Beam Itself\n"
    
    # Method B results
    for i, acc in enumerate(metrics['top_n_accuracies']['without_input']):
        config_content += f"- Top-{i+1}: {acc:.4f} ({acc*100:.2f}%)\n"
    
    config_content += f"""
## File Output
- Model file: model_*.pth
- Evaluation metrics: evaluation_metrics.json
- Configuration file: config.json
- Visualization chart: top_N_*.png

---
*This file is automatically generated by the program, recording the complete configuration and results of the experiment*
"""
    
    # Save to file, using timestamp naming
    config_filename = f'config_{timestamp}.md'
    config_path = os.path.join(output_dir, config_filename)
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"Experiment configuration saved to: {config_path}")
    return config_filename

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
    
    print(f"Starting training: {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"Input dimension: {X_train.shape[1]}, Output dimension: {y_train.shape[1]}")
    print(f"MLR structure: {HYPERPARAMETERS['N_QUBITS']} → {HYPERPARAMETERS['MLR_HIDDEN_DIM']} → {HYPERPARAMETERS['OUTPUT_DIM']}")
    print(f"Auto-calculated training end position: {HYPERPARAMETERS['TRAIN_END']} (EPOCHS×BATCH_SIZE = {HYPERPARAMETERS['EPOCHS']}×{HYPERPARAMETERS['BATCH_SIZE']})")
    print(f"Input beam count: {len(input_indices)}, Output beam count: {len(output_indices)}")
    
    # Train model
    train_losses, test_losses, mae_scores = train_model(
        model, train_loader, test_loader, 
        epochs=HYPERPARAMETERS['EPOCHS'], 
        lr=HYPERPARAMETERS['LEARNING_RATE']
    )
    
    # Evaluate model (including both Top-N accuracy methods)
    predictions, targets, metrics = evaluate_model_with_top_n(
        model, test_loader, scaler_y, input_indices, HYPERPARAMETERS['TOP_N_MAX']
    )
    
    # Save model
    model_path = os.path.join(output_dir, f'model_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)
    
    # Save evaluation metrics
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save configuration information (JSON format)
    config = {
        'hyperparameters': HYPERPARAMETERS,
        'model_config': {
            'n_qubits': HYPERPARAMETERS['N_QUBITS'],
            'n_layers': HYPERPARAMETERS['N_LAYERS'],
            'input_dim': HYPERPARAMETERS['INPUT_DIM'],
            'output_dim': HYPERPARAMETERS['OUTPUT_DIM'],
            'chunk_size': 4,
            'n_chunks': HYPERPARAMETERS['INPUT_DIM'] // 4,
            'mlr_hidden_dim': HYPERPARAMETERS['MLR_HIDDEN_DIM'],
            'mlr_activation': HYPERPARAMETERS['MLR_ACTIVATION']
        },
        'training_config': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'train_range': [int(train_indices[0]), int(train_indices[-1])],
            'test_range': [int(test_indices[0]), int(test_indices[-1])],
            'epochs': HYPERPARAMETERS['EPOCHS'],
            'batch_size': HYPERPARAMETERS['BATCH_SIZE'],
            'learning_rate': HYPERPARAMETERS['LEARNING_RATE'],
            'shuffle_train': HYPERPARAMETERS['SHUFFLE_TRAIN']
        },
        'data_config': {
            'input_indices': input_indices.tolist(),
            'output_indices': output_indices,
            'data_path': HYPERPARAMETERS['DATA_PATH']
        },
        'timestamp': timestamp
    }
    
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Plot results and get filename (including both Top-N methods)
    results_filename = plot_results(
        predictions, targets, train_losses, test_losses, mae_scores, 
        metrics['top_n_accuracies'], output_dir, timestamp
    )
    
    # Save experiment configuration document
    config_filename = save_experiment_config(output_dir, metrics, train_losses, test_losses, mae_scores, timestamp)
    
    # Print final results
    print(f"\nTraining completed!")
    print(f"MSE: {metrics['MSE']:.6f}")
    print(f"MAE: {metrics['MAE']:.6f}")
    print(f"RMSE: {metrics['RMSE']:.6f}")
    print(f"R²: {metrics['R2']:.6f}")
    
    # Print both Top-N accuracy methods
    print(f"\nTop-N Accuracy Comparison:")
    print(f"\nMethod A - Including Input Beams:")
    for i, acc in enumerate(metrics['top_n_accuracies']['with_input']):
        print(f"  Top-{i+1}: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\nMethod B - Excluding Input Beams:")
    for i, acc in enumerate(metrics['top_n_accuracies']['without_input']):
        print(f"  Top-{i+1}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Performance level assessment
    if metrics['R2'] > 0.75:
        performance_level = "Excellent"
    elif metrics['R2'] >= 0.6:
        performance_level = "Good"
    elif metrics['R2'] >= 0.3:
        performance_level = "Average"
    elif metrics['R2'] >= 0.1:
        performance_level = "Poor"
    else:
        performance_level = "Very Poor"
    
    print(f"\nPerformance Level: {performance_level}")
    print(f"Result Files:")
    print(f"  - Model: {model_path}")
    print(f"  - Configuration: {os.path.join(output_dir, config_filename)}")
    print(f"  - Results Plot: {os.path.join(output_dir, results_filename)}")
    print(f"  - Evaluation Metrics: {metrics_path}")

if __name__ == "__main__":
    main()