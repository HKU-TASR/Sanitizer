#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.9

import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch


def test_img(net_g, datatest, args):
    net_g.eval()

    # Initialize data loader and variables
    data_loader = DataLoader(datatest, batch_size=args.bs)
    dataset_size = len(data_loader.dataset)
    device = args.device if args.gpu != -1 else 'cpu'

    running_loss = 0.0
    correct_predictions = 0

    # Testing loop
    with torch.no_grad():
        for data, target in data_loader:
            # Move data to device
            data = data.to(device)
            target = target.to(device)

            # Forward pass
            outputs = net_g(data)

            # Calculate batch loss
            batch_loss = F.cross_entropy(outputs, target, reduction='sum').item()
            running_loss += batch_loss

            # Get predictions and calculate accuracy
            predictions = outputs.argmax(dim=1, keepdim=True) # e.g., [64, 10]->[64, 1] (shape)
            correct_predictions += predictions.eq(target.view_as(predictions)).sum().item()

    # Calculate final metrics
    avg_loss = running_loss / dataset_size
    accuracy = 100.0 * correct_predictions / dataset_size

    # Print results if in verbose mode
    if args.verbose:
        print(f'\nTest Results:'
              f'\nAverage Loss: {avg_loss:.4f}'
              f'\nAccuracy: {correct_predictions}/{dataset_size} ({accuracy:.2f}%)\n')

    return accuracy, avg_loss


def test_img_loader(net_g, datatest_loader, args=None):
    print("Start testing...")
    net_g.eval()

    device = args.device if args is not None else 'cpu'
    dataset_size = len(datatest_loader.dataset)
    running_loss = 0.0
    correct_predictions = 0

    with torch.no_grad():
        for data, target in datatest_loader:
            # Move data to specified device
            data = data.to(device)
            target = target.to(device)

            # Forward propagation
            outputs = net_g(data)

            # Calculate loss
            batch_loss = F.cross_entropy(outputs, target, reduction='sum').item()
            running_loss += batch_loss

            # Calculate predictions
            predictions = outputs.argmax(dim=1, keepdim=True)
            correct_predictions += predictions.eq(target.view_as(predictions)).sum().item()

    # Calculate average loss and accuracy
    avg_loss = running_loss / dataset_size
    accuracy = 100.0 * correct_predictions / dataset_size

    # Print test results if required
    if args is not None and args.verbose:
        print(f'\nTest Results:\n'
              f'Average Loss: {avg_loss:.4f}\n'
              f'Accuracy: {correct_predictions}/{dataset_size} ({accuracy:.2f}%)\n')

    return accuracy, avg_loss
