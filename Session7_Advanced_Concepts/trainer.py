import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision import datasets
from torchvision.utils import make_grid

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from torchsummary import summary
from model import CIFAR10Model
import seaborn as sns

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, device, train_loader, optimizer, epoch, train_acc, train_loss, lambda_l1, criterion, lrs, grad_clip=None,scheduler=None):
    model.train()
    pbar = tqdm(train_loader)
  
    correct = 0
    processed = 0
  
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        y_pred = model(data)


        loss = criterion(y_pred, target)

        #L1 Regularization
        if lambda_l1 > 0:
            l1 = 0
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
                loss = loss + lambda_l1 * l1

        train_loss.append(loss.data.cpu().numpy().item())

        # Backpropagation
        loss.backward()

        # Gradient clipping
        if grad_clip: 
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
        optimizer.step()
        # scheduler.step()
        lrs.append(get_lr(optimizer))

        # Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Train Loss={loss.item()} Batch_id={batch_idx} LR={lrs[-1]: 0.5f} Train Accuracy={100 * correct / processed: 0.2f}')
        train_acc.append(100 * correct / processed)


def test(model, device, test_loader, test_acc, test_losses, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\n: Average Test loss: {:.4f}, Test Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    test_acc.append(100. * correct / len(test_loader.dataset))


