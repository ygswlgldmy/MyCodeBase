import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.distributed as dist
import torchvision
import torch

from torch.optim import Optimizer
from torchvision.transforms import ToTensor
from torch.utils.data import random_split, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloader import *
from utils import *

def do_train(train_loader, loss_fn, avail_device, optimizer_mode='sgd', epochs=10, lr=0.01):
    
    model = ClassificationNet(input_shape=3, num_classes=10).to(avail_device)
    print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = MyOptimizer(
                        model.parameters(), 
                        mode=optimizer_mode, 
                        lr=lr,
                        momentum=0.9,
                        betas=(0.9, 0.999),
                        eps=1e-8
                    )

    model.train()
    loss_record = []

    for epoch in range(epochs):
        for batch_idx, (train_img, train_lbl) in enumerate(train_loader):
            
            train_img = train_img.to(avail_device)
            train_lbl = train_lbl.to(avail_device)
            
            optimizer.zero_grad()
            y_pred = model(train_img)
            loss = loss_fn(y_pred, train_lbl)
            loss_record.append(loss.detach().item())
            loss.backward()
            optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}, Optimizer: {optimizer_mode}, LR: {lr}")

    return model, loss_record

def main(batch_size=64):

    transform = ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    avail_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {avail_device}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_loader)*batch_size}, Test Samples: {len(test_loader)*batch_size}")
    
    loss_record = {
        '0.001': [],
        '0.0005': [],
        '0.0001': [],
    }
    
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    
    for lr in loss_record.keys():
        print(f"Training with learning-rate: {lr}")
        model, record_loss = do_train(
                                    train_loader, 
                                    loss_fn, 
                                    avail_device, 
                                    optimizer_mode='sgd',  # 使用不同的优化器
                                    epochs=100,
                                    lr=float(lr)  # 使用不同的学习率
                                )
        loss_record[lr] = record_loss
        plot_loss_comparison(
                            loss_record,
                            title=f"Loss Comparison with SGD", 
                        )

    print("Training completed for all optimizers.")

if __name__ == "__main__":

    torch.manual_seed(42)

    main(
        batch_size=64
    )