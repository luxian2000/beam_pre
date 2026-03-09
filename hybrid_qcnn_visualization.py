"""
Hybrid Quantum-Classical CNN Architecture Visualization
混合量子-经典CNN架构可视化

This script creates professional diagrams showing the proposed hybrid architecture
that replaces dense layers in RSRP_net.py with quantum neural networks.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_main_architecture_diagram():
    """Create the main hybrid architecture diagram"""
    
    # Set up the figure with professional styling
    plt.style.use('seaborn-v0_8')  # Use seaborn style for better aesthetics
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Professional color scheme
    colors = {
        'classical_input': '#2E86AB',    # Deep blue for input/classical
        'quantum': '#A23B72',            # Deep purple for quantum
        'processing': '#F18F01',         # Orange for processing
        'output': '#C73E1D',             # Red for output
        'connection': '#6C757D'          # Gray for connections
    }
    
    # Component specifications
    component_width = 2.8
    component_height = 1.5
    data_width = 2.4
    data_height = 1.2
    
    # Main components (top row)
    components = [
        {
            'pos': (2.0, 7.5),
            'size': (component_width, component_height),
            'label': '经典CNN特征提取',
            'sub_label': 'Classical CNN Feature Extraction',
            'color': colors['classical_input'],
            'icon': 'CNN'
        },
        {
            'pos': (5.5, 7.5),
            'size': (component_width, component_height),
            'label': '量子神经网络',
            'sub_label': 'Quantum Neural Network',
            'color': colors['quantum'],
            'icon': 'QNN'
        },
        {
            'pos': (9.0, 7.5),
            'size': (component_width, component_height),
            'label': '经典后处理',
            'sub_label': 'Classical Post-processing',
            'color': colors['processing'],
            'icon': 'MLP'
        },
        {
            'pos': (12.5, 7.5),
            'size': (component_width, component_height),
            'label': '最终输出',
            'sub_label': 'Final Output',
            'color': colors['output'],
            'icon': 'OUT'
        }
    ]
    
    # Data flow (bottom row)
    data_flow = [
        {
            'pos': (2.0, 4.0),
            'size': (data_width, data_height),
            'label': 'Input(8,4,1)',
            'sub_label': 'Input Tensor',
            'color': colors['classical_input']
        },
        {
            'pos': (5.5, 4.0),
            'size': (data_width, data_height),
            'label': 'Quantum State\nEncoding/Circuit',
            'sub_label': 'Quantum Processing',
            'color': colors['quantum']
        },
        {
            'pos': (9.0, 4.0),
            'size': (data_width, data_height),
            'label': 'Measurement\n+Processing',
            'sub_label': 'Quantum Measurement',
            'color': colors['processing']
        },
        {
            'pos': (12.5, 4.0),
            'size': (data_width, data_height),
            'label': '256-dim\nOutput',
            'sub_label': 'Output Vector',
            'color': colors['output']
        }
    ]
    
    # Draw main components
    for comp in components:
        x, y = comp['pos']
        w, h = comp['size']
        
        # Main component box
        rect = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=comp['color'],
            edgecolor='white',
            linewidth=3,
            alpha=0.9
        )
        ax.add_patch(rect)
        
        # Component label
        ax.text(x, y + 0.2, comp['label'], 
                ha='center', va='center', 
                fontsize=14, fontweight='bold',
                color='white')
        
        # Sub-label
        ax.text(x, y - 0.2, comp['sub_label'], 
                ha='center', va='center', 
                fontsize=10, style='italic',
                color='white')
        
        # Add icon/text representation
        ax.text(x, y - 0.5, comp['icon'], 
                ha='center', va='center',
                fontsize=20, fontweight='bold',
                color='white', alpha=0.7)
    
    # Draw data flow boxes
    for data in data_flow:
        x, y = data['pos']
        w, h = data['size']
        
        # Data flow box with dashed border
        rect = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=data['color'],
            edgecolor='white',
            linewidth=2,
            linestyle='--',
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Data label
        ax.text(x, y + 0.1, data['label'], 
                ha='center', va='center', 
                fontsize=12, fontweight='bold',
                color='white')
        
        # Data sub-label
        ax.text(x, y - 0.2, data['sub_label'], 
                ha='center', va='center', 
                fontsize=9, style='italic',
                color='white')
    
    # Draw horizontal connections (top row)
    for i in range(len(components) - 1):
        start_x = components[i]['pos'][0] + components[i]['size'][0]/2 + 0.2
        end_x = components[i+1]['pos'][0] - components[i+1]['size'][0]/2 - 0.2
        y_pos = 7.5
        
        ax.annotate('', xy=(end_x, y_pos), xytext=(start_x, y_pos),
                   arrowprops=dict(arrowstyle='->', lw=3, color=colors['connection']))
    
    # Draw horizontal connections (bottom row)
    for i in range(len(data_flow) - 1):
        start_x = data_flow[i]['pos'][0] + data_flow[i]['size'][0]/2 + 0.2
        end_x = data_flow[i+1]['pos'][0] - data_flow[i+1]['size'][0]/2 - 0.2
        y_pos = 4.0
        
        ax.annotate('', xy=(end_x, y_pos), xytext=(start_x, y_pos),
                   arrowprops=dict(arrowstyle='->', lw=3, color=colors['connection']))
    
    # Draw vertical connections
    for i in range(len(components)):
        top_x, top_y = components[i]['pos'][0], components[i]['pos'][1] - components[i]['size'][1]/2 - 0.2
        bottom_x, bottom_y = data_flow[i]['pos'][0], data_flow[i]['pos'][1] + data_flow[i]['size'][1]/2 + 0.2
        
        ax.annotate('', xy=(bottom_x, bottom_y), xytext=(top_x, top_y),
                   arrowprops=dict(arrowstyle='->', lw=2, color=colors['connection'], 
                                 linestyle='--', alpha=0.7))
    
    # Add title
    ax.text(7.5, 9.2, 'Hybrid Quantum-Classical Neural Network Architecture', 
            ha='center', va='center', fontsize=18, fontweight='bold')
    ax.text(7.5, 8.8, '混合量子-经典神经网络架构', 
            ha='center', va='center', fontsize=16, style='italic')
    
    # Add parameter comparison panel
    param_panel = FancyBboxPatch(
        (10.0, 1.5), 4.0, 2.0,
        boxstyle="round,pad=0.2",
        facecolor='lightyellow',
        edgecolor='orange',
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(param_panel)
    
    param_text = ("参数量对比 Comparison:\n"
                  "• 经典全连接层 Classical Dense: ~2.8M\n"
                  "• 量子替代方案 Quantum Replacement: ~100-1K\n"
                  "• 压缩比 Compression Ratio: >99%\n"
                  "• 量子比特需求 Qubits Needed: 8-12")
    
    ax.text(12.0, 2.5, param_text, ha='center', va='center', 
            fontsize=11, fontweight='normal')
    
    # Add benefits panel
    benefits_panel = FancyBboxPatch(
        (0.5, 1.5), 4.0, 2.0,
        boxstyle="round,pad=0.2",
        facecolor='lightgreen',
        edgecolor='darkgreen',
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(benefits_panel)
    
    benefits_text = ("核心优势 Key Benefits:\n"
                     "• 参数效率提升 Parameter Efficiency\n"
                     "• 量子并行处理 Quantum Parallelism\n"
                     "• 强非线性表达 Strong Non-linearity\n"
                     "• 指数级希尔伯特空间 Exponential Hilbert Space")
    
    ax.text(2.5, 2.5, benefits_text, ha='center', va='center', 
            fontsize=11, fontweight='normal')
    
    # Add implementation notes
    notes_text = ("Implementation Strategy:\n"
                  "1. 保持CNN前端特征提取不变\n"
                  "2. 用量子电路替代最后2-3个全连接层\n"
                  "3. 采用data re-uploading技术扩展编码容量\n"
                  "4. 结合经典后处理确保输出质量和稳定性")
    
    ax.text(7.5, 0.8, notes_text, ha='center', va='center', 
            fontsize=10, style='italic', color='darkblue')
    
    plt.tight_layout()
    return fig

def create_detailed_quantum_component():
    """Create detailed view of the quantum neural network component"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'Detailed Quantum Neural Network Component', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(6, 7.1, '量子神经网络组件详细视图', 
            ha='center', va='center', fontsize=14, style='italic')
    
    # Quantum circuit visualization
    # Input encoding section
    ax.text(2, 6.5, 'Input Encoding (32-dim → Quantum State)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Quantum wires/qubits
    n_qubits = 8
    wire_spacing = 0.4
    start_y = 5.5
    
    for i in range(n_qubits):
        y_pos = start_y - i * wire_spacing
        # Horizontal wire
        ax.plot([1, 11], [y_pos, y_pos], 'b-', linewidth=2, alpha=0.7)
        # Qubit label
        ax.text(0.5, y_pos, f'|q{i}⟩', ha='center', va='center', fontsize=10)
    
    # Quantum gates along the circuit
    gate_positions = [
        (2.0, 5.3, 'U3(θ₀,φ₀,λ₀)', 'purple'),
        (3.0, 4.9, 'U3(θ₁,φ₁,λ₁)', 'purple'),
        (4.0, 4.5, 'CX', 'red'),
        (5.0, 4.1, 'RY(θ₂)', 'green'),
        (6.0, 3.7, 'RZ(λ₂)', 'orange'),
        (7.0, 3.3, 'Measure', 'blue'),
        (8.0, 2.9, 'Expectation', 'darkred'),
        (9.0, 2.5, 'Process', 'darkgreen')
    ]
    
    for x, y, label, color in gate_positions:
        if label in ['CX']:
            # Control-X gate (special shape)
            circle = patches.Circle((x, y), 0.15, facecolor=color, edgecolor='black')
            ax.add_patch(circle)
            ax.plot([x, x], [y+0.15, y+0.3], 'k-', linewidth=2)
            ax.text(x, y-0.3, 'X', ha='center', va='center', fontsize=8, fontweight='bold')
        else:
            # Regular gate
            rect = patches.Rectangle((x-0.25, y-0.2), 0.5, 0.4, 
                                   facecolor=color, edgecolor='black', alpha=0.8)
            ax.add_patch(rect)
            ax.text(x, y, label.replace('(', '\n('), ha='center', va='center', 
                   fontsize=8, color='white')
    
    # Output connections
    ax.text(10.5, 2.0, '256-dim Output\nVector', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Add measurement explanation
    measure_box = FancyBboxPatch(
        (7.5, 0.5), 3.0, 1.2,
        boxstyle="round,pad=0.1",
        facecolor='lightblue',
        edgecolor='blue',
        linewidth=1
    )
    ax.add_patch(measure_box)
    
    measure_text = "Multiple measurements\nrequired for stable output"
    ax.text(9.0, 1.1, measure_text, ha='center', va='center', fontsize=9)
    
    # Add parameter count
    param_text = f"Quantum Parameters: {n_qubits * 3 * 4} = {n_qubits * 12} parameters"
    ax.text(2, 1.0, param_text, ha='center', va='center', fontsize=11, fontweight='bold')
    
    return fig

def create_comparison_chart():
    """Create parameter comparison chart"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Data for comparison
    models = ['Classical CNN', 'Hybrid QCNN']
    params = [2800000, 1000]  # Approximate parameter counts
    colors = ['#2E86AB', '#A23B72']
    
    # Bar chart
    bars = ax1.bar(models, params, color=colors, alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_ylabel('Number of Parameters (log scale)')
    ax1.set_title('Parameter Count Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{param:,}', ha='center', va='bottom')
    
    # Efficiency comparison
    reduction = (1 - params[1]/params[0]) * 100
    
    # Pie chart for efficiency
    sizes = [reduction, 100-reduction]
    labels = [f'Parameter Reduction\n{reduction:.1f}%', f'Remaining\n{100-reduction:.1f}%']
    colors_pie = ['#A23B72', '#CCCCCC']
    
    ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Parameter Efficiency Gain')
    
    plt.tight_layout()
    return fig

def main():
    """Generate all visualization diagrams"""
    
    print("Generating Hybrid QCNN Architecture Visualizations...")
    
    # Create main architecture diagram
    fig1 = create_main_architecture_diagram()
    fig1.savefig('hybrid_qcnn_architecture.png', dpi=300, bbox_inches='tight')
    print("✓ Main architecture diagram saved as 'hybrid_qcnn_architecture.png'")
    
    # Create detailed quantum component
    fig2 = create_detailed_quantum_component()
    fig2.savefig('quantum_component_detail.png', dpi=300, bbox_inches='tight')
    print("✓ Quantum component detail saved as 'quantum_component_detail.png'")
    
    # Create comparison chart
    fig3 = create_comparison_chart()
    fig3.savefig('parameter_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Parameter comparison chart saved as 'parameter_comparison.png'")
    
    print("\nAll visualizations generated successfully!")
    print("Files created:")
    print("1. hybrid_qcnn_architecture.png - Main architecture overview")
    print("2. quantum_component_detail.png - Detailed quantum circuit view") 
    print("3. parameter_comparison.png - Parameter efficiency comparison")

if __name__ == "__main__":
    main()