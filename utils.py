import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from optimizers import *

class RegressionModel(torch.nn.Module):
    """
        Parameters:
            input_shape: int, 输入特征的数量
            act_func: str, 要使用的激活函数（默认值：'tanh'）
    """

    def __init__(self, input_shape, act_func='tanh'):
        super(RegressionModel, self).__init__()
        
        # 根据激活函数字符串选择对应的激活函数
        if act_func == 'tanh':
            activation = nn.Tanh()
        elif act_func == 'relu':
            activation = nn.ReLU()
        else:
            raise ValueError("Unsupported activation function")

        # 定义网络结构
        self.dense_layers = nn.Sequential(
            nn.Linear(input_shape, 100),
            activation,
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.dense_layers(x)
    
class MyOptimizer(torch.optim.Optimizer):
    """
        Parameters:
            params: 可迭代的参数列表
            mode: 优化器模式（'gd' 或 'sgd'）
            lr: 学习率
    """

    def __init__(self, params, mode='sgd', lr=0.01, *args, **kwargs):
        if mode == 'sgd':
            self.optimizer = MySGD(params, lr=lr, momentum=0.9)
        elif mode == 'adam':
            self.optimizer = MyAdam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)
        elif mode == 'adagrad':
            self.optimizer = MyAdaGrad(params, lr=lr, eps=1e-10)
        else:
            raise ValueError("Unsupported optimizer mode")
    
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()

def plot_loss_comparison(loss_records):
    
    plt.figure(figsize=(10, 6))
    for mode, loss_record in loss_records.items():
        plt.plot(loss_record, label=mode)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')  # Log scale for better visualization of differences
    plt.tight_layout()
    plt.savefig("loss_comparison.png")