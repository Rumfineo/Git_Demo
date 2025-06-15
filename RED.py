import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

# ===== Step 1：生成耦合 Lorenz 系统数据 =====
def mylorenz(N, time=105, stepsize=0.02, C=0.1):
    """
    生成 N 个耦合的 Lorenz 系统，每个系统有 3 个变量 (x, y, z)
    返回 shape = (时间步数 l, 总变量数 3N)
    """
    l = int(round(time / stepsize))        # 总时间步数，例如 100/0.02 = 5000
    x = np.zeros((3 * N, l))               # x[i, j]：第 i 个变量在第 j 步的取值

    # 初始值线性递增，避免所有变量初值相同导致同步混沌
    x[:, 0] = np.linspace(-0.1, -0.1 + 0.003 * (3 * N - 1), 3 * N)

    for i in range(l - 1):
        # 第一个系统，耦合最后一个系统
        x[0, i+1] = x[0, i] + stepsize * (10 * (x[1, i] - x[0, i]) + C * x[3*(N-1), i])
        x[1, i+1] = x[1, i] + stepsize * (28 * x[0, i] - x[1, i] - x[0, i]*x[2, i])
        x[2, i+1] = x[2, i] + stepsize * (-8/3 * x[2, i] + x[0, i]*x[1, i])
        
        # 后续系统，每个都耦合前一个系统
        for j in range(1, N):
            idx = 3 * j         # 当前系统的起始变量位置
            prev = idx - 3      # 前一个系统 x 的位置
            # 系统 j 的三个变量
            x[idx, i+1] = x[idx, i] + stepsize * (10 * (x[idx+1, i] - x[idx, i]) + C * x[prev, i])
            x[idx+1, i+1] = x[idx+1, i] + stepsize * (28 * x[idx, i] - x[idx+1, i] - x[idx, i]*x[idx+2, i])
            x[idx+2, i+1] = x[idx+2, i] + stepsize * (-8/3 * x[idx+2, i] + x[idx, i]*x[idx+1, i])

    return torch.tensor(x.T, dtype=torch.float)  # 转置为 shape = (时间步数, 3N)

# ===== Step 2：定义弱预测器结构 =====
class EmbedNet(nn.Module):
    """
    多层感知机，用于从一组变量组合预测目标变量
    输入：窗口长度 × 变量数
    输出：目标变量未来值（单步预测）
    """
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)  # 输出 shape: (batch,)

# ===== Step 3：训练多个弱预测器（RDE核心） =====
def train_rde_predictors(Y, target_index, window=10, embed_dim=3, num_models=50,
                          tau=1, hidden_dim=16, epochs=50):
    """
    构建多个随机弱模型，每个模型使用一组变量组合作为输入，预测目标变量的未来值
    """
    T, D = Y.shape                   # T：时间长度，D：变量数
    predictors = []                 # 存储模型和其对应的变量索引
    embedding_indices = []

    for _ in range(num_models):
        # 随机选择嵌入变量
        var_idx = sorted(random.sample(range(D), embed_dim))
        embedding_indices.append(var_idx)

        # 构建滑动窗口样本
        X_train = []
        y_train = []
        for t in range(window, T - tau):
            # 输入：过去 window 步中这些变量的值，展平成向量
            x_embed = Y[t - window:t, var_idx].reshape(-1)
            y = Y[t + tau, target_index]  # 目标变量的未来值（τ 步后）
            X_train.append(x_embed)
            y_train.append(y)

        X_train = torch.stack(X_train)
        y_train = torch.stack(y_train)

        # 构建并训练模型
        model = EmbedNet(input_dim=window * embed_dim, hidden_dim=hidden_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            pred = model(X_train)
            loss = loss_fn(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 存储训练好的模型和变量组合
        predictors.append((model, var_idx))

    return predictors, embedding_indices

# ===== Step 4：预测（聚合所有弱模型） =====
def rde_predict(Y, predictors, t_star, window, tau):
    """
    给定时间点 t_star，使用所有 RDE 弱模型预测目标变量未来 τ 步的值
    返回平均预测值和标准差（表示不确定性）
    """
    preds = []
    for model, idx in predictors:
        if t_star - window < 0 or t_star + tau > Y.shape[0]:
            continue  # 跳过边界不合法的点
        x = Y[t_star - window:t_star, idx].reshape(1, -1)
        with torch.no_grad():
            pred = model(x).item()
        preds.append(pred)

    mean_pred = np.mean(preds)
    std_pred = np.std(preds)
    return mean_pred, std_pred

# ===== Step 5：运行示例 =====
Y = mylorenz(N=3)
print(Y.shape)
input_train_length = 11
target_index = 0                 # 目标变量是第一个系统的 x
window = 6
tau = 1                          # 预测未来 1 步

writer = SummaryWriter("RED_logs")
# 训练多个弱预测器
for ii in range(1000):
    begin = 3000 + ii * 2
    end = begin + input_train_length 
    Y_ii = Y[begin:end, : ]
    print(Y_ii.shape)
    predictors, _ = train_rde_predictors(
        Y_ii, target_index, window=window, embed_dim=3, num_models=50, tau=tau
    )

    # 在第 t_star 时刻做预测
    t_star = input_train_length - tau
    mean_pred, std_pred = rde_predict(Y_ii, predictors, t_star, window, tau)
    true_value = Y[end, target_index].item()

    # 打印结果
    print(f"第{ii}次预测值: {mean_pred:.4f} ± {std_pred:.4f}，真实值: {true_value:.4f}")
    writer.add_scalars("111",{"pred_value":mean_pred , "true_value":true_value},ii)

writer.close()   
    
