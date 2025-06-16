import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

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