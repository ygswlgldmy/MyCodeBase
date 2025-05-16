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

def main(epoch = 30, batch_size=64, optimizer='sgd', Net='ClassificationNet'):

    transform = ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    if Net == 'ClassificationNet':
        Net = ClassificationNet()
    elif Net == 'RegressionNet':
        Net = RegressionModel()

    model, train_loader, test_loader, avail_device = DDP_setup(Net=Net, train_dataset=train_dataset, test_dataset=test_dataset, batch_size=batch_size)

    # avail_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {avail_device}")
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    main(
        epoch=30,
        batch_size=256,
        optimizer='sgd'
    )