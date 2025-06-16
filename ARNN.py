import torch
import torch.nn.functional as F
import numpy as np
from NN_F2 import NN_F2

def arnn_predict(traindata, traindata_y, L=5, target_idx=0, k=60, max_iter=1000, tol=1e-4):
    """

    Args:
        traindata:          Tensor [T, input_dim] 原始数据
        traindata_y:        Tensor [T]  真实值
        L:                  int  predict_length  预测长度
        target_idx:         int  目标变量索引
        k:                  int  选取的变量数量
        max_iter:           int  最大迭代次数
        tol:                float  收敛
        T:                  trainlength  时间步

    Returns:
        union_predict_y_ARNN: Tensor [L - 1] 预测值

    """
    T , input_dim = traindata.shape

    # === Step 1: extract nonlinear features using NN_F2 ===
    nn_f2 = NN_F2(input_dim)
    F_traindata = nn_f2(traindata)  # shape: [T, final_layer_dim]

    W_flag = torch.zeros(input_dim)
    B = torch.zeros(T, L)
    predict_pred = torch.zeros(L - 1)

    for iter in range(max_iter):
        #随机选择k个变量
        idx_pool = list(range(input_dim))
        idx_pool.remove(target_idx)
        random_idx = torch.tensor([target_idx] + list(np.random.choice(idx_pool, k - 1, replace=False)))
        random_idx.sort()
        selected_F = F_traindata[:, random_idx]  # [T, k]

#        B Y = f(x)  求B  

        for i, idx in enumerate(random_idx):
            b = selected_F[:T - L + 1, i]  # [m-L+1]
            B_window = traindata_y.unfold(0, L, 1)
            result = torch.linalg.lstsq(B_window, b.unsqueeze(1))
            solution = result.solution
            B_para = solution[:L].squeeze(1)
            if W_flag[idx] == 0:
                B[idx] = B_para
            else:
                B[idx] = (B[idx] + B_para) / 2      #B[i][j]: 第i个变量的第j个权重
            W_flag[idx] = 1                         #如：X[t][i] = Y[t]*B[i][1]+Y[t+1]*B[i][2]+...

        # === Step 3: estimate next-step predictions ===
        super_bb = []
        super_AA = []

        for i in range(input_dim):
            bb_i = []
            AA_i = []

            for j in range(T - L + 1, T):
                col_known = T - j
                known_y = traindata_y[T - col_known:T]
                bb = F_traindata[j, i] - torch.sum(B[i, :col_known] * known_y)
                bb_i.append(bb.item())
                AA_row = torch.zeros(L - 1)
                AA_row[:L - col_known] = B[i, col_known:]
                AA_i.append(AA_row)

            super_bb.extend(bb_i)
            super_AA.extend(AA_i)

        super_bb = torch.tensor(super_bb).unsqueeze(1)  # [total, 1]
        super_AA = torch.stack(super_AA)  # [total, L-1]
        lstsq_result = torch.linalg.lstsq(super_AA, super_bb)
        pred_y_tmp = lstsq_result.solution[:L - 1].squeeze(1)

        # === Step 4: update A matrix and recompute prediction ===
        tmp_y = torch.cat([traindata_y[:T], pred_y_tmp])
        Ym = torch.stack([tmp_y[j:j + T] for j in range(L)])  # [L, T]
        BX = torch.cat([B, F_traindata.T], dim=1)
        IY = torch.cat([torch.eye(L), Ym], dim=1)
        A = IY @ torch.pinverse(BX)

        union_predict_y_ARNN = []
        for j1 in range(L - 1):
            tmp_vals = []
            for j2 in range(j1, L - 1):
                row = j2 + 1
                col = T - j2 + j1
                val = torch.matmul(A[row], F_traindata[col])
                tmp_vals.append(val.item())
            union_predict_y_ARNN.append(np.mean(tmp_vals))

        union_predict_y_ARNN = torch.tensor(union_predict_y_ARNN)

        # === Step 5: convergence check ===
        eof_error = torch.sqrt(F.mse_loss(union_predict_y_ARNN, predict_pred))
        if eof_error < tol:
            break
        predict_pred = union_predict_y_ARNN.clone()

    return union_predict_y_ARNN
