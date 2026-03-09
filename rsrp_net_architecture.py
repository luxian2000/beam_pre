"""
RSRP_net.py Classical Neural Network Architecture Visualization
RSRP_net.py经典神经网络架构可视化

This script creates detailed diagrams showing the classical CNN architectures
defined in RSRP_net.py for beam RSRP prediction.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_basics_net_diagram():
    """Create diagram for BM_RSRP_net architecture"""
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#2E86AB',
        'conv': '#A23B72',
        'flatten': '#F18F01',
        'dense': '#C73E1D',
        'output': '#2E86AB'
    }
    
    # Network layers with positions and specifications
    layers = [
        # Input layer
        {
            'name': 'Input',
            'type': 'input',
            'shape': '(8, 4, 1)',
            'pos': (1.5, 8.0),
            'size': (2.0, 1.2),
            'params': 0,
            'color': colors['input']
        },
        
        # Conv2D layer
        {
            'name': 'Conv2D',
            'type': 'conv',
            'shape': '64 filters\n(3×3 kernel)',
            'pos': (4.5, 8.0),
            'size': (2.0, 1.2),
            'params': 640,
            'color': colors['conv']
        },
        
        # BatchNorm + Activation
        {
            'name': 'BatchNorm +\nLeakyReLU',
            'type': 'norm',
            'shape': 'Normalization\nActivation',
            'pos': (7.5, 8.0),
            'size': (2.0, 1.2),
            'params': 128,  # 64 * 2 for batch norm
            'color': colors['conv']
        },
        
        # Flatten layer
        {
            'name': 'Flatten',
            'type': 'flatten',
            'shape': 'Reshape to 1D',
            'pos': (10.5, 8.0),
            'size': (2.0, 1.2),
            'params': 0,
            'color': colors['flatten']
        },
        
        # First Dense layer
        {
            'name': 'Dense',
            'type': 'dense',
            'shape': '1024 units\nReLU',
            'pos': (2.0, 5.0),
            'size': (2.5, 1.5),
            'params': 2098176,
            'color': colors['dense']
        },
        
        # Second Dense layer
        {
            'name': 'Dense',
            'type': 'dense',
            'shape': '512 units\nReLU',
            'pos': (5.5, 5.0),
            'size': (2.5, 1.5),
            'params': 524800,
            'color': colors['dense']
        },
        
        # Third Dense layer
        {
            'name': 'Dense',
            'type': 'dense',
            'shape': '256 units\nReLU',
            'pos': (9.0, 5.0),
            'size': (2.5, 1.5),
            'params': 131328,
            'color': colors['dense']
        },
        
        # Output layer
        {
            'name': 'Output',
            'type': 'output',
            'shape': '256 units\nLinear',
            'pos': (12.5, 5.0),
            'size': (2.5, 1.5),
            'params': 65792,
            'color': colors['output']
        }
    ]
    
    # Draw layers
    for layer in layers:
        x, y = layer['pos']
        w, h = layer['size']
        
        # Layer rectangle
        rect = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=layer['color'],
            edgecolor='white',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        # Layer name
        ax.text(x, y + 0.3, layer['name'], 
                ha='center', va='center', 
                fontsize=12, fontweight='bold',
                color='white')
        
        # Layer details
        ax.text(x, y, layer['shape'], 
                ha='center', va='center', 
                fontsize=9,
                color='white')
        
        # Parameter count
        if layer['params'] > 0:
            param_text = f"{layer['params']:,} params"
            ax.text(x, y - 0.3, param_text,
                    ha='center', va='center',
                    fontsize=8, style='italic',
                    color='white')
    
    # Draw connections
    connections = [
        # Sequential connections in top row
        (layers[0], layers[1]),  # Input -> Conv2D
        (layers[1], layers[2]),  # Conv2D -> BatchNorm
        (layers[2], layers[3]),  # BatchNorm -> Flatten
        
        # Connections from flatten to dense layers
        (layers[3], layers[4]),  # Flatten -> Dense1
        (layers[4], layers[5]),  # Dense1 -> Dense2
        (layers[5], layers[6]),  # Dense2 -> Dense3
        (layers[6], layers[7]),  # Dense3 -> Output
    ]
    
    for start_layer, end_layer in connections:
        start_x = start_layer['pos'][0] + start_layer['size'][0]/2 + 0.1
        end_x = end_layer['pos'][0] - end_layer['size'][0]/2 - 0.1
        start_y = start_layer['pos'][1]
        end_y = end_layer['pos'][1]
        
        if start_y == end_y:  # Horizontal connection
            ax.annotate('', xy=(end_x, start_y), xytext=(start_x, start_y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        else:  # Vertical connection
            # First horizontal segment
            mid_x = (start_x + end_x) / 2
            ax.plot([start_x, mid_x], [start_y, start_y], 'k-', linewidth=2)
            # Vertical segment
            ax.plot([mid_x, mid_x], [start_y, end_y], 'k-', linewidth=2)
            # Second horizontal segment
            ax.annotate('', xy=(end_x, end_y), xytext=(mid_x, end_y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Add title and statistics
    ax.text(7.5, 9.5, 'BM_RSRP_net Architecture - Basic CNN Model', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Total parameters calculation
    total_params = sum(layer['params'] for layer in layers)
    ax.text(12.5, 9.0, f'Total Parameters: {total_params:,}', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Add data flow annotations
    ax.text(1.5, 7.0, 'Input Flow:', ha='center', va='center', fontsize=10, style='italic')
    ax.text(1.5, 6.7, '(8,4,1) → (8,4,64) → (8,4,64) → (2048,) → ...', 
            ha='center', va='center', fontsize=9)
    
    return fig

def create_resnet_architecture_diagram():
    """Create diagram for ResNet-style architectures"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Color scheme for ResNet
    colors = {
        'stem': '#2E86AB',
        'residual': '#A23B72',
        'transition': '#F18F01',
        'classification': '#C73E1D'
    }
    
    # Stem block
    stem_layers = [
        {
            'name': 'Stem Conv',
            'details': 'Conv2D(64, 1×1)\nBN + LeakyReLU',
            'pos': (2.0, 10.0),
            'size': (3.0, 1.5),
            'color': colors['stem']
        }
    ]
    
    # Residual blocks
    residual_blocks = [
        {
            'name': 'ResBlock 1',
            'details': 'Conv2D(64, 3×3) + Conv2D(64, 5×5)\nShortcut Connection',
            'pos': (6.0, 10.0),
            'size': (3.5, 1.8),
            'color': colors['residual']
        },
        {
            'name': 'ResBlock 2',
            'details': 'Conv2D(64, 3×3) + Conv2D(64, 5×5)\nShortcut Connection',
            'pos': (10.5, 10.0),
            'size': (3.5, 1.8),
            'color': colors['residual']
        },
        {
            'name': 'Transition Block',
            'details': 'Conv2D(128, 1×1) + Conv2D(128, 3×3) + Conv2D(128, 5×5)\nStrided Convolution',
            'pos': (4.0, 7.0),
            'size': (4.0, 2.0),
            'color': colors['transition']
        },
        {
            'name': 'ResBlock 3',
            'details': 'Conv2D(128, 3×3) + Conv2D(128, 5×5)\nShortcut Connection',
            'pos': (9.0, 7.0),
            'size': (3.5, 1.8),
            'color': colors['residual']
        }
    ]
    
    # Classification layers
    classification_layers = [
        {
            'name': 'Global Pooling',
            'details': 'Adaptive Pooling\nFlatten',
            'pos': (2.0, 4.0),
            'size': (2.5, 1.2),
            'color': colors['classification']
        },
        {
            'name': 'Dense Layers',
            'details': '512 → 256 → 256\nReLU activations',
            'pos': (6.0, 4.0),
            'size': (3.0, 1.5),
            'color': colors['classification']
        },
        {
            'name': 'Output',
            'details': '256 units\nLinear',
            'pos': (11.0, 4.0),
            'size': (2.5, 1.2),
            'color': colors['classification']
        }
    ]
    
    # Draw all layers
    all_layers = stem_layers + residual_blocks + classification_layers
    
    for layer in all_layers:
        x, y = layer['pos']
        w, h = layer['size']
        
        rect = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=layer['color'],
            edgecolor='white',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        ax.text(x, y + 0.2, layer['name'], 
                ha='center', va='center', 
                fontsize=11, fontweight='bold',
                color='white')
        
        ax.text(x, y - 0.1, layer['details'], 
                ha='center', va='center', 
                fontsize=8,
                color='white')
    
    # Draw connections with skip connections
    # Main flow connections
    main_connections = [
        (stem_layers[0], residual_blocks[0]),
        (residual_blocks[0], residual_blocks[1]),
        (residual_blocks[1], residual_blocks[2]),
        (residual_blocks[2], residual_blocks[3]),
        (residual_blocks[3], classification_layers[0]),
        (classification_layers[0], classification_layers[1]),
        (classification_layers[1], classification_layers[2])
    ]
    
    for start_layer, end_layer in main_connections:
        start_x = start_layer['pos'][0] + start_layer['size'][0]/2 + 0.1
        end_x = end_layer['pos'][0] - end_layer['size'][0]/2 - 0.1
        start_y = start_layer['pos'][1]
        end_y = end_layer['pos'][1]
        
        ax.annotate('', xy=(end_x, start_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Skip connections (residual connections)
    skip_connections = [
        (stem_layers[0]['pos'][0], stem_layers[0]['pos'][1], 
         residual_blocks[0]['pos'][0], residual_blocks[0]['pos'][1]),
        (residual_blocks[0]['pos'][0], residual_blocks[0]['pos'][1],
         residual_blocks[1]['pos'][0], residual_blocks[1]['pos'][1]),
        (residual_blocks[2]['pos'][0], residual_blocks[2]['pos'][1],
         residual_blocks[3]['pos'][0], residual_blocks[3]['pos'][1])
    ]
    
    for start_x, start_y, end_x, end_y in skip_connections:
        # Curved skip connection
        mid_x = (start_x + end_x) / 2
        mid_y = max(start_y, end_y) + 1.0
        
        # Draw curved path
        points = [(start_x + 0.5, start_y), (mid_x, mid_y), (end_x - 0.5, end_y)]
        xs, ys = zip(*points)
        ax.plot(xs, ys, 'r--', linewidth=2, alpha=0.7)
        
        # Add "Skip" label
        ax.text(mid_x, mid_y + 0.2, 'Skip\nConnection', 
                ha='center', va='center', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
    
    # Add title
    ax.text(8.0, 11.5, 'ResNet-style Architecture - BM_RSRP_resnet Series', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add feature map progression
    feature_progression = [
        "(8,4,1) → Stem",
        "(8,4,64) → ResBlocks", 
        "(4,2,128) → Transition",
        "(2,1,128) → Final Blocks",
        "(1,1,128) → Global Pool",
        "128 → Dense Layers → 256"
    ]
    
    ax.text(13.0, 8.0, 'Feature Progression:\n' + '\n'.join(feature_progression),
            ha='left', va='center', fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    
    return fig

def create_parameter_analysis_chart():
    """Create parameter distribution analysis"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Parameter distribution data
    layers = ['Conv2D', 'BatchNorm', 'Dense(1024)', 'Dense(512)', 'Dense(256)', 'Dense(Output)']
    params = [640, 128, 2098176, 524800, 131328, 65792]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#2E86AB', '#6C757D']
    
    # Bar chart
    bars = ax1.bar(range(len(layers)), params, color=colors, alpha=0.8)
    ax1.set_xlabel('Network Layers')
    ax1.set_ylabel('Number of Parameters')
    ax1.set_title('Parameter Distribution in BM_RSRP_net')
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, param) in enumerate(zip(bars, params)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:,}', ha='center', va='bottom', fontsize=9)
    
    # Pie chart for parameter distribution
    # Group similar layers
    groups = {
        'Convolutional': params[0] + params[1],
        'Dense Layers': sum(params[2:-1]),
        'Output Layer': params[-1]
    }
    
    group_names = list(groups.keys())
    group_values = list(groups.values())
    group_colors = ['#2E86AB', '#F18F01', '#C73E1D']
    
    wedges, texts, autotexts = ax2.pie(group_values, labels=group_names, colors=group_colors,
                                      autopct='%1.1f%%', startangle=90)
    ax2.set_title('Parameter Distribution by Layer Type')
    
    # Make autopct text larger
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    return fig

def main():
    """Generate all RSRP_net architecture visualizations"""
    
    print("Generating RSRP_net.py Architecture Visualizations...")
    
    # Create basic net diagram
    fig1 = create_basics_net_diagram()
    fig1.savefig('rsrp_basics_net_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Basic CNN architecture diagram saved")
    
    # Create ResNet diagram
    fig2 = create_resnet_architecture_diagram()
    fig2.savefig('rsrp_resnet_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ ResNet architecture diagram saved")
    
    # Create parameter analysis
    fig3 = create_parameter_analysis_chart()
    fig3.savefig('rsrp_parameter_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Parameter analysis chart saved")
    
    print("\nAll RSRP_net.py architecture visualizations generated!")
    print("Generated files:")
    print("1. rsrp_basics_net_architecture.png - Basic CNN structure")
    print("2. rsrp_resnet_architecture.png - ResNet-style architecture") 
    print("3. rsrp_parameter_analysis.png - Parameter distribution analysis")

if __name__ == "__main__":
    main()