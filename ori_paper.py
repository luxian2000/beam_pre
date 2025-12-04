import pennylane as qml
from pennylane import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ----------------------------
# 1. 定义量子设备
# ----------------------------
dev = qml.device('default.qubit', wires=1)

# ----------------------------
# 2. 定义数据重传量子线路
# ----------------------------
def layer(params, x):
    """
    单层数据重传线路。
    params: [w1, w2, b1, b2, b3]
    x: [x1, x2]
    """
    w1, w2, b1, b2, b3 = params
    # 数据重传与处理
    qml.RZ(w1 * x[0] + b1, wires=0)
    qml.RY(w2 * x[1] + b2, wires=0)
    qml.RZ(b3, wires=0)

@qml.qnode(dev)
def circuit(params, x, num_layers):
    """
    整个量子线路。
    params: shape (num_layers, 5) -> 每层5个参数
    x: [x1, x2]
    """
    # 初始化状态 |0>
    qml.RY(0.0, wires=0) # 可选，确保初始状态为 |0>

    for i in range(num_layers):
        layer(params[i], x)

    # 返回量子态
    return qml.state()

# ----------------------------
# 3. 定义目标态 (Maximally Orthogonal States for 2 classes)
# ----------------------------
# |0> 对应类别 0 (圆外)
state_0 = np.array([1, 0], dtype=complex)
# |1> 对应类别 1 (圆内)
state_1 = np.array([0, 1], dtype=complex)

# ----------------------------
# 4. 数据生成
# ----------------------------
def generate_circle_data(n_samples=200, test=False):
    np.random.seed(42) # 为了可复现
    X = np.random.uniform(-1, 1, (n_samples, 2))
    r = np.sqrt(0.5)
    y = (X[:, 0]**2 + X[:, 1]**2 < r**2).astype(int) # 1 if inside, 0 if outside
    if not test:
        # 转换为类别标签 0, 1
        pass
    return X.astype(np.float64), y

X_train, y_train = generate_circle_data(200)
X_test, y_test = generate_circle_data(2000, test=True)

# ----------------------------
# 5. 定义加权保真度损失函数 (Weighted Fidelity Cost Function)
# ----------------------------
def cost_function(params, X, y, num_layers):
    """
    计算加权保真度损失。
    """
    total_cost = 0.0
    M = len(X)
    # 论文中 Yc(x) 定义: 对于正确类别 s, Ys = 1; 对于错误类别 r, Yr = 1/(C-1) = 1/(2-1) = 1
    # 但这与 Eq. 9 不太一致。更合理的解释是:
    # 对于类别 0: Y0 = 1, Y1 = 0
    # 对于类别 1: Y0 = 0, Y1 = 1
    # 且权重 alpha_c 用于平衡不同类别的贡献。这里简化，假设 alpha_c = 1 for all c.
    # 简化版: loss = 1 - |<psi_true_class | psi_pred>|^2
    # 或者，按 Eq. 9 的加权形式:
    # Fc = |<psi_c | psi(x)>|^2
    # loss = sum_c (Fc - Yc)^2 / 2
    # Yc = 1 if c is true class, 0 otherwise
    # 等价于 loss = (F_true_class - 1)^2 + (F_other_class - 0)^2
    # 等价于 loss = (1 - F_true_class)^2 + F_other_class^2
    # 但为了简化，我们使用 1 - F_true_class (即 fidelity cost function)

    # 为了更贴近论文的 weighted fidelity，我们定义一个权重向量 alpha
    # alpha = [alpha_0, alpha_1]
    # 这里我们暂时将 alpha 并入 params 以便优化
    # params_flat = [alpha_0, alpha_1, layer0_params_flat, layer1_params_flat, ...]
    alpha_0 = params[0]
    alpha_1 = params[1]
    params_unflat = params[2:].reshape((num_layers, 5))

    for i in range(M):
        x_i = X[i]
        y_i = y[i]

        # 获取预测的量子态
        psi_pred = circuit(params_unflat, x_i, num_layers)

        # 计算与目标态的保真度
        f0 = np.abs(np.vdot(state_0, psi_pred))**2 # 保真度到 |0>
        f1 = np.abs(np.vdot(state_1, psi_pred))**2 # 保真度到 |1>

        # 预期保真度 Yc
        if y_i == 0: # 真实类别是 0
            Y0 = 1.0
            Y1 = 0.0
        else: # 真实类别是 1
            Y0 = 0.0
            Y1 = 1.0

        # 加权保真度损失 (Eq. 9)
        term1 = (alpha_0 * f0 - Y0)**2
        term2 = (alpha_1 * f1 - Y1)**2
        sample_cost = 0.5 * (term1 + term2)
        total_cost += sample_cost

    return total_cost / M # 平均损失

# ----------------------------
# 6. 初始化参数
# ----------------------------
num_layers = 4
# 初始权重 alpha (论文中 alpha_c)
alpha_init = np.array([1.0, 1.0])
# 初始线路参数 (每层5个)
init_params_unflat = np.random.uniform(-np.pi, np.pi, (num_layers, 5))
init_params_flat = np.concatenate([alpha_init, init_params_unflat.flatten()])

# ----------------------------
# 7. 定义优化函数
# ----------------------------
def objective(params):
    return cost_function(params, X_train, y_train, num_layers)

def callback_fn(params):
    current_cost = objective(params)
    print(f"Cost: {current_cost:.6f}")

# ----------------------------
# 8. 执行优化
# ----------------------------
print("Starting optimization...")
result = minimize(
    objective,
    init_params_flat,
    method='L-BFGS-B',
    callback=callback_fn,
    options={'disp': True, 'maxiter': 200} # 论文 Table 1 显示 2 layers (12 params) 即可达到 90%+
)
print("Optimization finished.")
optimal_params = result.x
optimal_cost = result.fun
print(f"Final Cost: {optimal_cost:.6f}")

# ----------------------------
# 9. 测试与评估
# ----------------------------
def predict(params, X, num_layers):
    alpha_0 = params[0]
    alpha_1 = params[1]
    params_unflat = params[2:].reshape((num_layers, 5))
    predictions = []
    for x in X:
        psi_pred = circuit(params_unflat, x, num_layers)
        f0 = np.abs(np.vdot(state_0, psi_pred))**2
        f1 = np.abs(np.vdot(state_1, psi_pred))**2
        # 使用加权保真度进行预测
        weighted_f0 = alpha_0 * f0
        weighted_f1 = alpha_1 * f1
        if weighted_f0 > weighted_f1:
            pred_class = 0
        else:
            pred_class = 1
        predictions.append(pred_class)
    return np.array(predictions)

y_pred_train = predict(optimal_params, X_train, num_layers)
y_pred_test = predict(optimal_params, X_test, num_layers)

acc_train = np.mean(y_pred_train == y_train)
acc_test = np.mean(y_pred_test == y_test)
print(f"\nTraining Accuracy: {acc_train:.4f}")
print(f"Test Accuracy: {acc_test:.4f}")

# ----------------------------
# 10. 可视化结果 (可选)
# ----------------------------
plt.figure(figsize=(6, 5))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, cmap='coolwarm', s=10, alpha=0.7)
circle = plt.Circle((0, 0), np.sqrt(0.5), color='black', fill=False, linewidth=1.5)
plt.gca().add_patch(circle)
plt.title(f"Pennylane Data Re-uploading Classifier (Acc: {acc_test:.2%})")
plt.xlabel("$x_1$"); plt.ylabel("$x_2$")
plt.axis('equal')
plt.show()
