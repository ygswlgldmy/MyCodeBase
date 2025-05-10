import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch

from torch.optim import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from dataloader import *
from utils import *

def do_train(idx_img, train_img, train_lbl, loss_fn, avail_device, optimizer_mode='gd', epochs=1000, lr=0.01):
    
    model = RegressionModel(input_shape=len(idx_img), act_func='tanh').to(avail_device)
    print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = MyOptimizer(model.parameters(), mode=optimizer_mode, lr=lr)

    model.train()
    loss_record = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(train_img)
        loss = loss_fn(y_pred, train_lbl.unsqueeze(-1))
        loss_record.append(loss.detach().item())
        loss.backward()
        optimizer.step()

        if (epoch+1) % 2000 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}, Optimizer: {optimizer_mode}, LR: {lr}")

    return model, loss_record


def main():

    housing_prepared = prepare_housing_data()

    avail_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    housing_data = torch.tensor(housing_prepared, device=avail_device)

    train_set, test_set = split_train_test(housing_data, 0.2)

    idx_img, idx_lbl = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 9]), 8
    train_img = train_set[:, idx_img].float()
    train_lbl = train_set[:, idx_lbl].float()
    test_img = test_set[:, idx_img].float()
    test_lbl = test_set[:, idx_lbl].float()

    print(f"Train samples: {len(train_set)}, Test Samples: {len(test_set)}")
    
    loss_record = []
    
    loss_fn = nn.MSELoss()
    
    for optimizer_mode in loss_record.keys():
        print(f"Training with optimizer: {optimizer_mode}")
        model, record_loss = do_train(idx_img, 
                                    train_img, 
                                    train_lbl, 
                                    loss_fn, 
                                    avail_device, 
                                    optimizer_mode='fw',  # 使用不同的优化器
                                    epochs=20000,
                                    lr=float(optimizer_mode)  # 使用不同的学习率
                                )
        loss_record[optimizer_mode] = record_loss
        plot_loss_comparison(loss_record)

    print("Training completed for all optimizers.")

if __name__ == "__main__":

    torch.manual_seed(42)
    

    main()