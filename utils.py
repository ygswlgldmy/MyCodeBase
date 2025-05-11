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
    
class ClassificationNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        
        super(ClassificationNet, self).__init__()

        self.features = nn.Sequential(
            
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)

        return x
    
class MyOptimizer(torch.optim.Optimizer):
    """
        Parameters:
            params: 可迭代的参数列表
            mode: 优化器模式（'gd' 或 'sgd'）
            lr: 学习率
            momentum: 动量（仅适用于 'sgd' 模式）
            betas: Adam优化器的beta参数（仅适用于 'adam' 模式）
            eps: Adam优化器的epsilon参数（仅适用于 'adam' 模式）
    """

    def __init__(
            self, 
            params, 
            mode='sgd', 
            lr=0.01, 
            momentum=0.9, 
            betas=(0.9, 0.999), 
            eps=1e-8, 
            *args
        ):

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

def plot_loss_comparison(
                        loss_records,
                        title='Loss Curves Comparison'
                    ):
    
    plt.figure(figsize=(10, 6))
    for mode, loss_record in loss_records.items():
        plt.plot(loss_record, label=mode)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')  # Log scale for better visualization of differences
    plt.tight_layout()
    plt.savefig("loss_comparison.png")