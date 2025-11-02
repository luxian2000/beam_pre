# -*- coding: utf-8 -*-
"""
复现论文《Data re-uploading for a universal quantum classifier》P10 的 Figure 6（circle problem）：
(a) 1 layer, (b) 2 layers, (c) 4 layers, (d) 8 layers。

做法完全遵循论文第2节与式(12)、式(5)的单比特数据重上传模型：
U(φ) = e^{iφ2 σz} e^{iφ1 σy} e^{iφ3 σz}，其中 φ_i = θ_i + w_i ∘ x，x∈R^2（第三分量补0）。
训练使用加权保真度损失（Eq. (9) 的二分类特例，α=1），优化器优先用 L-BFGS-B，
若不可用则回退到简单的随机坐标扰动下降（速度较慢但能收敛到可见解）。
生成训练集 200 点、测试集 4000 点，半径 r = sqrt(2/pi)（论文第10页）。
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------- 物理：单比特旋转 ----------
def apply_U_layers(x2d, thetas, ws):
    """
    x2d: (M,2) 训练/测试样本，取值 in [-1,1]^2
    thetas, ws: 形状 (N,3)，每层3个角度（ZYZ）和3个权重（第三维乘0）
    返回：
      probs0, probs1: (M,) 到 |0>, |1> 的概率
    注：实现 U = Rz(φ2) Ry(φ1) Rz(φ3)，Ry 的显式形式来自 e^{iβσy} = [[cosβ, sinβ],[-sinβ, cosβ]]
    """
    M = x2d.shape[0]
    # 扩成三维，第三维恒0（论文：若维度<3，其余分量置0）
    x = np.zeros((M,3), dtype=float)
    x[:,:2] = x2d
    # 初态 |0>
    psi0 = np.ones(M, dtype=complex)
    psi1 = np.zeros(M, dtype=complex)
    # 逐层重上传
    for i in range(thetas.shape[0]):
        phi = thetas[i] + ws[i] * x  # (M,3) 元素乘
        phi1, phi2, phi3 = phi[:,0], phi[:,1], phi[:,2]
        # 先右乘 Rz(phi3)
        epos  = np.exp(1j*phi3)
        eneg  = np.exp(-1j*phi3)
        psi0, psi1 = psi0*epos, psi1*eneg
        # 再 Ry(phi1)
        c, s = np.cos(phi1), np.sin(phi1)
        new0 = c*psi0 + s*psi1
        new1 = -s*psi0 + c*psi1
        psi0, psi1 = new0, new1
        # 最后 Rz(phi2)
        epos2, eneg2 = np.exp(1j*phi2), np.exp(-1j*phi2)
        psi0, psi1 = psi0*epos2, psi1*eneg2
    p0 = (psi0*np.conj(psi0)).real
    p1 = (psi1*np.conj(psi1)).real
    # 归一化（数值稳定）
    s = p0 + p1 + 1e-15
    return p0/s, p1/s

# ---------- 损失函数：加权保真度（Eq.9 的二分类特例） ----------
def weighted_fidelity_loss(params, N, Xtr, Ytr):
    """
    params: 展平成 6N 维向量，前 3N 是 θ，后 3N 是 w
    Ytr: 0/1 标签，0=inside(|0>), 1=outside(|1>)
    """
    thetas = params[:3*N].reshape(N,3)
    ws     = params[3*N:].reshape(N,3)
    p0, p1 = apply_U_layers(Xtr, thetas, ws)
    # 目标向量 Yc ：[1,0] 或 [0,1]
    Y0 = 1.0 - Ytr
    Y1 = Ytr.astype(float)
    # 加权保真度 α=1
    loss = 0.5*np.mean((p0 - Y0)**2 + (p1 - Y1)**2)
    return loss

# ---------- 训练（优先 L-BFGS-B，失败则随机扰动下降） ----------
def train_model(N, Xtr, Ytr, seed=7, maxiter=200):
    rng = np.random.default_rng(seed)
    # 初始化：小角度/小权重
    thetas0 = rng.uniform(-0.2,0.2,size=(N,3))
    ws0     = rng.uniform(-0.5,0.5,size=(N,3))
    params0 = np.concatenate([thetas0.ravel(), ws0.ravel()])
    try:
        from scipy.optimize import minimize
        res = minimize(lambda v: weighted_fidelity_loss(v,N,Xtr,Ytr),
                       params0, method="L-BFGS-B",
                       options=dict(maxiter=maxiter, ftol=1e-9))
        params = res.x
        used = "L-BFGS-B"
    except Exception as e:
        # 简单的随机坐标下降作为兜底
        params = params0.copy()
        base = weighted_fidelity_loss(params,N,Xtr,Ytr)
        step = 0.2
        for t in range(2500):
            idx = rng.integers(0, params.size)
            cand = params.copy()
            cand[idx] += rng.normal(0, step)
            val = weighted_fidelity_loss(cand,N,Xtr,Ytr)
            if val < base:
                params, base = cand, val
            step *= 0.999
        used = "random-descent"
    return params, used

# ---------- 数据：圆形二分类 ----------
def make_circle_sets(n_train=200, n_test=4000, seed=1):
    rng = np.random.default_rng(seed)
    Xtr = rng.uniform(-1.0,1.0,size=(n_train,2))
    Xte = rng.uniform(-1.0,1.0,size=(n_test,2))
    r = np.sqrt(2/np.pi)
    Ytr = ((Xtr[:,0]**2 + Xtr[:,1]**2) >= r**2).astype(int)  # 外部=1
    Yte = ((Xte[:,0]**2 + Xte[:,1]**2) >= r**2).astype(int)
    return Xtr, Ytr, Xte, Yte, r

# ---------- 训练并画图 ----------
Xtr, Ytr, Xte, Yte, r = make_circle_sets()

layer_list = [1,2,4,8]
models = {}
for i, N in enumerate(layer_list):
    params, used = train_model(N, Xtr, Ytr, seed=7+N, maxiter=250)
    models[N] = (params, used)

# 评估并画出四宫格（与论文Figure 6结构相同）
fig, axes = plt.subplots(2,2, figsize=(8,8))
axes = axes.ravel()

titles = ["(a)  1 layer", "(b)  2 layers", "(c)  4 layers", "(d)  8 layers"]

for ax, N, ttl in zip(axes, layer_list, titles):
    params, used = models[N]
    thetas = params[:3*N].reshape(N,3)
    ws     = params[3*N:].reshape(N,3)
    # 预测
    p0, p1 = apply_U_layers(Xte, thetas, ws)
    Ypred = (p1 > p0).astype(int)
    # 成功率
    acc = (Ypred == Yte).mean()
    # 散点
    ax.scatter(Xte[Ypred==0,0], Xte[Ypred==0,1], s=2, alpha=0.6)
    ax.scatter(Xte[Ypred==1,0], Xte[Ypred==1,1], s=2, alpha=0.6)
    # 黑色圆边界
    circle = plt.Circle((0,0), r, fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.set_xlim(-1,1); ax.set_ylim(-1,1)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(ttl + f"   (acc={acc:.2f})")

plt.tight_layout()
out_path = "/mnt/data/figure6_circle_layers.png"
plt.savefig(out_path, dpi=180, bbox_inches='tight')
print(out_path)