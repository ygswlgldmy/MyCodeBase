import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import matplotlib.pyplot as plt

from optimizers import *
from torch.nn.parallel import DistributedDataParallel as DDP

def DDP_setup(Net, train_dataset, test_dataset, batch_size):
    """
        设置主机地址和端口号，这两个环境变量用于配置进程组通信的初始化。
        MASTER_ADDR指定了负责协调初始化过程的主机地址，在这里设置为'localhost'，
        表示在单机多GPU的设置中所有的进程都将连接到本地机器上。
        MASTER_PORT指定了主机监听的端口号，用于进程间的通信。这里设置为'12355'。
        注意要选择一个未被使用的端口号来进行监听
    """
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    # rank是当前进程在进程组中的编号，world_size是总进程数（GPU数量），即进程组的大小。
    dist.init_process_group(backend='gloo')
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    
    model = Net
    if rank == 0:
        print(model)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id], output_device=device_id)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
    )

    return model, train_loader, test_loader, device_id, rank

def DDP_get_model(Net, rank):
    device = torch.device(f'cuda:{rank}')
    model = Net.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

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
            self.optimizer = MySGD(params, lr=lr, momentum=momentum)
        elif mode == 'adam':
            self.optimizer = MyAdam(params, lr=lr, betas=betas, eps=eps)
        elif mode == 'adagrad':
            self.optimizer = MyAdaGrad(params, lr=lr, eps=eps)
        elif mode == 'adam_official':
            self.optimizer = torch.optim.Adam(
                params, 
                lr=lr, 
                betas=betas, 
                eps=eps
            )
        else:
            raise ValueError("Unsupported optimizer mode")
    
    def step(self):
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()

def plot_loss_comparison(
                        loss_records,
                        optimizer='sgd',
                    ):
    
    plt.figure(figsize=(10, 6))
    for mode, loss_record in loss_records.items():
        plt.plot(loss_record, label=mode)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves Comparison: {optimizer}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.yscale('log')  # Log scale for better visualization of differences
    plt.tight_layout()
    plt.savefig(f"Loss_comparison_{optimizer}.png")

def plot_accuracy_comparison(
                        accuracy_records,
                        optimizer='sgd',
                    ):
    
    plt.figure(figsize=(10, 6))
    for mode, accuracy_record in accuracy_records.items():
        plt.plot(accuracy_record, label=mode)
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f"Accuracy Curves Comparison: {optimizer}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"Accuracy_comparison_{optimizer}.png")

def model_acc(model, test_loader, avail_device):

    epoch_accuracy = 0

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
    
    return epoch_accuracy

def do_train(model, 
            train_loader, 
            test_loader, 
            loss_fn, 
            rank, 
            avail_device, 
            optimizer_mode='sgd', 
            epochs=10, 
            lr=0.01,
            momentum=0.9,
            betas=(0.9, 0.999),
            eps=1e-8
        ):

    optimizer = MyOptimizer(
                        model.parameters(), 
                        mode=optimizer_mode, 
                        lr=lr,
                        momentum=momentum,
                        betas=betas,
                        eps=eps
                    )

    model.train()
    loss_record = []
    accuracy_record = []
    epoch_loss = 0

    epoch_begin_time = time.time()
    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)
        for batch_idx, (train_img, train_lbl) in enumerate(train_loader):
            
            train_img = train_img.to(avail_device)
            train_lbl = train_lbl.to(avail_device)
            
            optimizer.zero_grad()
            y_pred = model(train_img)
            loss = loss_fn(y_pred, train_lbl)
            epoch_loss += loss.detach().item()
            loss.backward()
            optimizer.step()

        loss_record.append(epoch_loss / len(train_loader))
        epoch_loss = 0

        accuracy = model_acc(model, test_loader, avail_device)
        accuracy_record.append(accuracy)

        if (epoch+1) % 5 == 0 and rank == 0:
            print(f"Epoch {epoch+1}, Acc: {accuracy_record[-1]} , Loss: {loss_record[-1]}, Optimizer: {optimizer_mode}, LR: {lr}")

    epoch_end_time = time.time()
    if rank == 0:
        print(f"Training time: {epoch_end_time - epoch_begin_time:.2f} seconds")

    return model, loss_record, accuracy_record