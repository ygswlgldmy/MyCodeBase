import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torch

from tqdm import tqdm
from torch.optim import Optimizer
from torchvision.transforms import ToTensor
from torch.utils.data import random_split, DataLoader
from dataloader import *
from utils import *

def do_train(Net, train_loader, test_loader, loss_fn, avail_device, optimizer_mode='sgd', epochs=10, lr=0.01):
    
    model = Net
    model = model.to(avail_device)
    model.train()
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
    epoch_loss = 0

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

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss_record[-1]}, Optimizer: {optimizer_mode}, LR: {lr}")
    
    accuracy_record = model_acc(model, test_loader, avail_device)

    return model, loss_record, accuracy_record

def main(epoch = 30, batch_size=64, optimizer='sgd', Net='ClassificationNet'):

    transform = ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    if Net == 'ClassificationNet':
        Net = ClassificationNet()
    elif Net == 'RegressionNet':
        Net = RegressionModel()

    model, train_loader = init_distributed_mode(Net=Net, dataset=train_dataset, batch_size=batch_size)

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

    acc_record = {
        '0.001': [],
        '0.0005': [],
        '0.0001': [],
    }
    
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()
    
    for lr in loss_record.keys():
        print(f"Training with learning-rate: {lr}")
        model, record_loss, record_acc = do_train(
                                        model,
                                        train_loader, 
                                        test_loader,
                                        loss_fn, 
                                        avail_device, 
                                        optimizer_mode=optimizer,  # 不同的优化器: sgd, adam, adagrad
                                        epochs=epoch,
                                        lr=float(lr)
                                    )
        
        loss_record[lr] = record_loss
        acc_record[lr] = record_acc

        plot_loss_comparison(
                            loss_record,
                            title=f"Loss Comparison with {optimizer}", 
                        )
        plot_accuracy_comparison(
                                acc_record,
                                title=f"Accuracy Comparison with {optimizer}",
                            )

    print("Training completed for all optimizers.")

if __name__ == "__main__":

    torch.manual_seed(42)

    main(
        epoch=30,
        batch_size=256,
        optimizer='sgd'
    )