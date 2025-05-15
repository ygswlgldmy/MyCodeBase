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
            mode: 优化器模式（'sgd', 'adagrad' 或 'adam'）
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

def plot_accuracy_comparison(
                        accuracy_records,
                        title='Accuracy Curves Comparison'
                    ):
    
    plt.figure(figsize=(10, 6))
    for mode, accuracy_record in accuracy_records.items():
        plt.plot(accuracy_record, label=mode)
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png")

def model_acc(model, test_loader, avail_device):

    epoch_accuracy = 0
    accuracy_record = []

    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0

        for batch_idx, (test_img, test_lbl) in enumerate(test_loader):
            test_img = test_img.to(avail_device)
            test_lbl = test_lbl.to(avail_device)

            y_pred = model(test_img)
            _, predicted = torch.max(y_pred.data, 1)

            # 计算当前 batch 的正确预测数和总样本数
            correct = (predicted == test_lbl).sum().item()
            total = test_lbl.size(0)

            total_correct += correct
            total_samples += total

        # 计算整体准确率
        epoch_accuracy = total_correct / total_samples
        accuracy_record.append(epoch_accuracy)

        print(f"Test Accuracy: {epoch_accuracy:.4f}")
        
    
    return accuracy_record