#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

# Built-in imports
import copy
import os
import time
from datetime import datetime

# External packages
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset

# Package specific imports
from baseline.reverse_engineering_NN import reverse_engineering_nn
from baseline.test_NN import test_nn_before, test_nn_after
from baseline.unlearning_NN import unlearning_nn
from baseline.update_NN import LocalUpdateNN
from baseline.reverse_engineering_NN import reverse_engineering_nn
from average.Nets import CNNCifar, CNNMnist, CNNFashionMNIST, MLP, PreActResNet18
from average.test import test_img, test_img_loader
from utils.options import args_parser
from utils.sampling import cifar_iid, mnist_iid, mnist_noniid, cifar_iid_1000
from utils.util import load_model, save_model
from architectures.nets_ResNet18 import ResNet18, ResNet18TinyImagenet
from architectures.nets_MLP import ComplexMLP
from architectures.nets_MobileNetV3 import MobileNetV3_Small

# Use matplotlib in 'Agg' mode
matplotlib.use('Agg')
now = datetime.now()  # Get current date and time
now_str = now.strftime("%Y-%m-%d_%H%M%S")  # Format as a string

acc_train_before = []
acc_test_before = []
acc_asr_before = []

acc_train_after = []
acc_test_after = []
acc_asr_after = []

time_record_array1 = []
time_record_array2 = []
time_record_array3 = []

def load_and_split_dataset(args):
    if args.dataset == 'mnist':
        mnist_path = './data/mnist/'  # If the path does not exist, PyTorch will attempt to create it
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        download_mnist = not (os.path.exists(mnist_path) and os.path.isdir(mnist_path))
        dataset_train = datasets.MNIST(mnist_path, train=True, download=download_mnist, transform=trans_mnist)
        dataset_test = datasets.MNIST(mnist_path, train=False, download=download_mnist, transform=trans_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'FashionMnist':  # If the path does not exist, PyTorch will attempt to create it
        fashion_mnist_path = './data/fashion_mnist/'
        trans_fashion_mnist = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        download_fashion_mnist = not (os.path.exists(fashion_mnist_path) and os.path.isdir(fashion_mnist_path))
        dataset_train = datasets.FashionMNIST(fashion_mnist_path, train=True, download=download_fashion_mnist,
                                              transform=trans_fashion_mnist)
        dataset_test = datasets.FashionMNIST(fashion_mnist_path, train=False, download=download_fashion_mnist,
                                             transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        cifar_path = '../data/cifar'
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        download_cifar = not (os.path.exists(cifar_path) and os.path.isdir(cifar_path))
        dataset_train = datasets.CIFAR10(cifar_path, train=True, download=download_cifar, transform=trans_cifar)
        dataset_test = datasets.CIFAR10(cifar_path, train=False, download=download_cifar, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, dict_users


def build_model(args, img_size):
    # Build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = ResNet18().to(args.device)
    elif args.model == 'mov3' and args.dataset == 'cifar':
        net_glob = MobileNetV3_Small(num_classes=10).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp' and args.dataset == 'FashionMnist':
        net_glob = ComplexMLP(dim_in=784, hidden1=1024, hidden2=512, hidden3=256, hidden4=128, hidden5=64, dim_out=10).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'TinyImageNet':
        net_glob = ResNet18TinyImagenet().to(args.device)
    elif args.model == 'mov3' and args.dataset == 'TinyImageNet':
        net_glob = MobileNetV3_Small(num_classes=200).to(args.device)
    else:
        raise ValueError('Error: unrecognized model')
    print(net_glob)
    return net_glob


def create_trigger_and_mask_arrays(dataset, device):
    trigger_shape = {'mnist': (1, 28, 28), 'FashionMnist': (1, 28, 28), 'cifar': (3, 32, 32), 'tiny': (3, 64, 64)}
    mask_shape = {'mnist': (28, 28), 'FashionMnist': (28, 28), 'cifar': (32, 32), 'tiny': (64, 64)}

    triggers = []
    masks = []

    # Default to creating triggers and masks for labels 0-9, Tiny has 200
    for _ in range(10):
        # Create and configure trigger
        trigger = torch.rand(trigger_shape[dataset], requires_grad=True)
        trigger = trigger.to(device).detach().requires_grad_(True)
        triggers.append(trigger)

        # Create and configure mask
        mask = torch.rand(mask_shape[dataset], requires_grad=True)
        mask = mask.to(device).detach().requires_grad_(True)
        masks.append(mask)

    return triggers, masks


def train_model(args, net_glob):
    net_glob.train()

    # Copy weights, although weights are typically initialized with small random values, biases are often initialized to zero.
    w_glob = net_glob.state_dict()

    # Training
    loss_train = []
    idxs_users = None
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    triggers_init, masks_init = create_trigger_and_mask_arrays(args.dataset, args.device)
    dataset_train_ = Subset(dataset_train, range(100))

    for iter in range(args.epochs):  # Default 100 epochs (rounds)
        loss_locals = []
        if not args.all_clients:
            w_locals = []

        local = LocalUpdateNN(args=args, dataset=dataset_train, idxs=dict_users[0], user_number=0)
        w, loss = local.train_local_nn(net=copy.deepcopy(net_glob).to(args.device), iter_outside=iter)

        if args.all_clients:
            w_locals[0] = copy.deepcopy(w)
        else:
            w_locals.append(copy.deepcopy(w))
        loss_locals.append(copy.deepcopy(loss))
        print(f"Local training with watermark for iteration {iter + 1} completed!!!")

        net_glob.load_state_dict(w)
        dataset_train_acc = Subset(dataset_train, range(6000, 8000))
        dataset_test_acc = Subset(dataset_test, range(6000, 8000))
        dataset_test_re = Subset(dataset_test, range(1000, 2000))
        dataset_test_ul = Subset(dataset_test, range(2000, 7000))

        # Reversed Engineering (Neural Cleanse uses all test data for RE)
        yt_label, triggers_init, masks_init, param = reverse_engineering_nn(dataset_test_re,
                                                                            copy.deepcopy(net_glob).to(args.device),
                                                                            args, user_number=0, it=iter,
                                                                            triggers_1=triggers_init,
                                                                            masks_1=masks_init)

        # ACC1 BEFORE
        test_nn_before(copy.deepcopy(net_glob), dataset_train_acc, dataset_test_acc, args)

        if iter == args.epochs - 1:
            # Unlearning (UL uses 10% of training data for UL)
            state_ = unlearning_nn(dataset_test_ul, copy.deepcopy(net_glob), yt_label, triggers_init, masks_init,
                                   specific_background='',
                                   param=param, it=iter)
            net_glob.load_state_dict(state_)

            # ACC2 AFTER
            test_nn_after(copy.deepcopy(net_glob), dataset_train_acc, dataset_test_acc, args)
        else:
            time_record_array3.append(0)
            acc_train_after.append(0)
            acc_test_after.append(0)
            acc_asr_after.append(0)

        # Print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
    return loss_train, idxs_users


if __name__ == '__main__':
    # Parse arguments
    args = args_parser()
    print(args)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # Load dataset and split users
    dataset_train, dataset_test, dict_users = load_and_split_dataset(args)
    # Build model
    img_size = dataset_train[0][0].shape
    net_glob = build_model(args, img_size)

    # Train Model
    start_time = time.time()
    loss_train, idxs_users = train_model(args, net_glob)  # Third return value is the model used for testing on client-side training
    end_time = time.time()
    execution_time = end_time - start_time

    # Save the 'net_glob' model
    save_model(net_glob, 'net_glob_' + now_str)

    print(f"Total training execution time: {execution_time:.4f} seconds")

    minutes, seconds = divmod(execution_time, 60)
    print(f"Total training execution time: {minutes:.0f} minutes {seconds:.4f} seconds")
    # Plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig(
        './save/fed_{}_{}_{}_C{}_iid{}_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                                                            args.local_ep, args.local_bs, args.gpu))

    # Testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.4f}".format(acc_train))
    print("Testing accuracy: {:.4f}".format(acc_test))

    with open('time_records1.txt', 'a') as f:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"Total training execution time (seconds): {execution_time:.4f} seconds\n")
        f.write(f"Total training execution time (minutes): {minutes:.0f} minutes {seconds:.4f} seconds\n")
        # Convert each array to a comma-separated string with square brackets
        f.write('[' + ','.join(map(str, time_record_array1)) + ']\n')
        f.write('[' + ','.join(map(str, time_record_array2)) + ']\n')
        f.write('[' + ','.join(map(str, time_record_array3)) + ']\n')

    with open('ACC_records1.txt', 'a') as f:
        f.write("BEFORE\n")
        f.write('[' + ','.join(map(str, acc_train_before)) + ']\n')
        f.write('[' + ','.join(map(str, acc_test_before)) + ']\n')
        f.write('[' + ','.join(map(str, acc_asr_before)) + ']\n')
        f.write("AFTER\n")
        f.write('[' + ','.join(map(str, acc_train_after)) + ']\n')
        f.write('[' + ','.join(map(str, acc_test_after)) + ']\n')
        f.write('[' + ','.join(map(str, acc_asr_after)) + ']\n')
    ########################################################################
    # Create time figure
    plt.figure(figsize=(10, 5))

    # Generate x-axis data representing rounds
    rounds = range(1, len(time_record_array1) + 1)  # Start counting from 1

    # Plot three lines for the three arrays using different colors
    plt.plot(rounds, time_record_array1, label='Local Training Time', color='red', marker='o')
    plt.plot(rounds, time_record_array2, label='Reversed Engineering Training Time', color='green', marker='x')
    plt.plot(rounds, time_record_array3, label='Unlearning Training Time', color='blue', marker='s')

    # Add legend
    plt.legend()

    # Add title and axis labels
    plt.title('Time Record by Round')
    plt.xlabel('Round')
    plt.ylabel('Time (seconds)')

    # Label the value of each point on each line, adjusting label position and adding spacing
    vertical_offset = 2  # Vertical offset
    alignment_settings = {
        'o': ('center', 'top', vertical_offset),  # Top
        's': ('center', 'top', vertical_offset),  # Top
        '^': ('center', 'top', vertical_offset),  # Top
    }

    for data, marker in zip([time_record_array1, time_record_array2, time_record_array3], ['o', 's', '^']):
        for x, y in zip(rounds, data):
            ha, va, offset = alignment_settings[marker]
            plt.text(x, y + offset, '{:.4f}'.format(y), color='black', fontsize=8, ha=ha, va=va)

    # Display figure
    plt.show()
    plt.savefig('./save/fed_time1.png')
    ########################################################################
    # Create BEFORE accuracy figure
    plt.figure(figsize=(10, 6))

    # Generate x-axis data representing rounds
    rounds = range(1, len(acc_train_before) + 1)  # Start counting from 1

    # Plot three lines for the three arrays using different colors
    plt.plot(rounds, acc_train_before, label='Clean Training DATA ACC', color='red', marker='o')
    plt.plot(rounds, acc_test_before, label='Clean Testing DATA ACC', color='green', marker='x')
    plt.plot(rounds, acc_asr_before, label='Trigger DATA ASR', color='blue', marker='s')

    # Add legend
    plt.legend()

    # Add title and axis labels
    plt.title('Before Unlearning ACC (ASR) by Round')
    plt.xlabel('Round')
    plt.ylabel('ACC (ASR) Accuracy')

    # Set y-axis tick intervals and format
    plt.yticks(np.arange(0, 106, 5))  # Set major tick interval to 5
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(1))  # Set minor tick interval to 1
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.4f}'.format(x)))  # Format as four decimals

    # Show grid (optional)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Label the value of each point on each line, adjusting label position and adding spacing
    vertical_offset = 2  # Vertical offset
    alignment_settings = {
        'o': ('center', 'top', vertical_offset),  # Top
        's': ('center', 'bottom', -vertical_offset),  # Bottom
        '^': ('right', 'bottom', -vertical_offset)  # Bottom-right corner
    }

    # Use zip to iterate through data and markers
    for idx, (data, marker) in enumerate(zip([acc_train_before, acc_test_before, acc_asr_before], ['o', 's', '^'])):
        if idx <= 2:  # Only label the first point for the first two lines
            x, y = rounds[0], data[0]  # Get coordinates of the first point
            ha, va, offset = alignment_settings[marker]
            plt.text(x, y + offset, '{:.4f}'.format(y), color='black', fontsize=8, ha=ha, va=va)
        else:  # For the third line, label all points
            for x, y in zip(rounds, data):
                ha, va, offset = alignment_settings[marker]
                plt.text(x, y + offset, '{:.4f}'.format(y), color='black', fontsize=8, ha=ha, va=va)

    # Display figure
    plt.show()
    plt.savefig('./save/before_acc1.png')

    ########################################################################
    # Create AFTER accuracy figure
    plt.figure(figsize=(10, 6))

    # Generate x-axis data representing rounds
    rounds = range(1, len(acc_train_before) + 1)  # Start counting from 1

    # Plot three lines for the three arrays using different colors
    plt.plot(rounds, acc_train_after, label='Clean Training DATA ACC', color='red', marker='o')
    plt.plot(rounds, acc_test_after, label='Clean Testing DATA ACC', color='green', marker='x')
    plt.plot(rounds, acc_asr_after, label='Trigger DATA ASR', color='blue', marker='s')

    # Add legend
    plt.legend()

    # Add title and axis labels
    plt.title('After Unlearning ACC (ASR) by Round')
    plt.xlabel('Round')
    plt.ylabel('ACC (ASR) Accuracy')

    # Set y-axis tick intervals and format
    plt.yticks(np.arange(0, 106, 5))  # Set major tick interval to 5
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(1))  # Set minor tick interval to 1
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.4f}'.format(x)))  # Format as four decimals

    # Show grid (optional)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Label the value of each point on each line, adjusting label position and adding spacing
    vertical_offset = 2  # Vertical offset
    alignment_settings = {
        'o': ('center', 'top', vertical_offset),  # Top
        's': ('center', 'bottom', -vertical_offset),  # Bottom
        '^': ('right', 'bottom', -vertical_offset)  # Bottom-right corner
    }

    # Use zip to iterate through data and markers
    for idx, (data, marker) in enumerate(zip([acc_train_after, acc_test_after, acc_asr_after], ['o', 's', '^'])):
        if idx < 2:  # Only label the first point for the first two lines
            x, y = rounds[0], data[0]  # Get coordinates of the first point
            ha, va, offset = alignment_settings[marker]
            plt.text(x, y + offset, '{:.4f}'.format(y), color='black', fontsize=8, ha=ha, va=va)
        else:  # For the third line, label all points
            for x, y in zip(rounds, data):
                ha, va, offset = alignment_settings[marker]
                plt.text(x, y + offset, '{:.4f}'.format(y), color='black', fontsize=8, ha=ha, va=va)

    # Display figure
    plt.show()
    plt.savefig('./save/after_acc1.png')

    print(f"Training completed!!")
