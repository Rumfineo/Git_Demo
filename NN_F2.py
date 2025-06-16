import torch
import torch.nn.functional as F

class NN_F2:
    def __init__(self, input_dim, layer_nodes_num=[400, 400, 300, 150]):
        self.layer_nodes_num = layer_nodes_num
        self.weights = []
        prev_dim = input_dim
        torch.manual_seed(0)  # for reproducibility
        for layer_dim in layer_nodes_num:
            # 每层为 (layer_dim, prev_dim) 的随机高斯矩阵
            W = torch.randn(prev_dim , layer_dim)
            self.weights.append(W)
            prev_dim = layer_dim

    def __call__(self, traindata):
        """
        :param traindata: shape [input_dim, T], torch tensor
        :return: output shape [final_layer_dim, T]
        """
        batch_size = traindata.shape[0]
        x = traindata  # [T, input_dim]
        for W in self.weights:
            x = torch.tanh((x @ W) / 2.5)
        return x  # [T, final_layer_dim]
