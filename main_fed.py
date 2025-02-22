#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9


# Built-in imports
import copy
import os
import ast
import time
from datetime import datetime

# External packages
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from torch.utils.data import Subset

# Package specific imports
from identification.test_fed_silo import test_silo, test_silo_with_asr_unharmful, test_silo_with_asr_unharmful_cifar, \
    test_silo_cifar
from identification.utils_training import unlearning_1, unlearning_2, final_unharmful_retrain
from baseline.reverse_engineering_NN import reverse_engineering_nn
from average.Fed import FedAvg
from average.Nets import CNNCifar, CNNMnist, CNNFashionMNIST, MLP, PreActResNet18
from average.Update import LocalUpdate
from average.test import test_img, test_img_loader
from utils.options import args_parser
from utils.ownership import DatasetOwnershipTargeted, verify_ownership, verify_ownership_targeted
from utils.sampling import cifar_iid, mnist_iid, mnist_noniid, cifar_iid_1000, tiny_iid, sample_dirichlet_train_data
from utils.util import load_model, save_model
from core.resnet_cifar.unlearning_extraction_function import main_cifar10_ul, create_pruned_model
from architectures.nets_ResNet18 import ResNet18, ResNet18CIFAR100
from architectures.nets_ResNet18_Tiny import ResNet18TinyImagenet
from architectures.nets_MLP import ComplexMLP
from architectures.nets_MobileNetV3 import MobileNetV3_Small
from architectures.nets_vit_cifar_pure import ViT_cifar10
from datasets.dataset_tiny import TinyImageNet

matplotlib.use('Agg')  # Use matplotlib in 'Agg' mode


##########################################################################################
class DatasetCL(Dataset):
    def __init__(self, args, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=args.defense_data_ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    # Random split
    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset


##########################################################################################
def load_and_split_dataset(args):
    transform_train = None
    transform_test = None
    in_dict_users = {}
    if args.dataset == 'mnist':
        mnist_path = './data/mnist/'  # If the path does not exist, PyTorch will try to create it
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # ToTensor() automatically converts the image data from an integer between 0 and 255 to a float between 0 and 1;
        # Each pixel value is subtracted by 0.1307 and then divided by 0.3081 to make the mean close to 0 and the std close to 1.
        download_mnist = not (os.path.exists(mnist_path) and os.path.isdir(mnist_path))
        dataset_train = datasets.MNIST(mnist_path, train=True, download=download_mnist, transform=trans_mnist)
        dataset_test = datasets.MNIST(mnist_path, train=False, download=download_mnist, transform=trans_mnist)
        if args.iid:
            in_dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            in_dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'FashionMnist':  # If the path does not exist, PyTorch will try to create it
        fashion_mnist_path = './data/fashion_mnist/'
        trans_fashion_mnist = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        download_fashion_mnist = not (os.path.exists(fashion_mnist_path) and os.path.isdir(fashion_mnist_path))
        dataset_train = datasets.FashionMNIST(fashion_mnist_path, train=True, download=download_fashion_mnist,
                                              transform=trans_fashion_mnist)
        dataset_test = datasets.FashionMNIST(fashion_mnist_path, train=False, download=download_fashion_mnist,
                                             transform=trans_fashion_mnist)
        if args.iid:
            in_dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            in_dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        cifar_path = './data/cifar'
        # transforms.RandomCrop(32, padding=4) and transforms.RandomHorizontalFlip() are usually applied to PIL Image objects.
        # If you are using other formats (such as NumPy arrays or PyTorch tensors), you need to convert them to PIL Images first.
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        download_cifar = not (os.path.exists(cifar_path) and os.path.isdir(cifar_path))

        dataset_train = datasets.CIFAR10(cifar_path, train=True, download=download_cifar)
        dataset_test = datasets.CIFAR10(cifar_path, train=False, download=download_cifar)
        if args.iid and args.scale == 1:
            in_dict_users = cifar_iid_1000(dataset_train, args.num_users)
        elif args.iid and args.scale == 0:
            in_dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            # 使用 Dirichlet 分布划分数据集
            print("NON-IID Setting starting")
            in_dict_users = sample_dirichlet_train_data(dataset_train, args.num_users, alpha = args.alpha_noniid)
            print("NON-IID Setting finishing")
            # exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'tinyimagenet':
        dataset = TinyImageNet(batch_size=args.batch_size)
        dataset_train = dataset.train_dataset_nt_CL
        dataset_test = dataset.test_dataset_nt_CL
        if args.iid:
            in_dict_users = tiny_iid(dataset_train, args.num_users)

    elif args.dataset == 'CIFAR-100':
        cifar100_path = './data/cifar100'
        # transforms.RandomCrop(32, padding=4) 和 transforms.RandomHorizontalFlip() 这两个变换操作通常应用于 PIL Image 对象。
        # 如果你使用的是其他图像格式（如 NumPy 数组或 PyTorch 张量），则需要先将这些格式转换为 PIL Image 对象，或者使用相应的转换操作
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # 将PIL图像或NumPy数组转换为PyTorch张量，并将像素值从0-255缩放到0-1之间。
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        download_cifar = not (os.path.exists(cifar100_path) and os.path.isdir(cifar100_path))

        dataset_train = datasets.CIFAR100(cifar100_path, train=True, download=download_cifar)
        dataset_test = datasets.CIFAR100(cifar100_path, train=False, download=download_cifar)
        if args.iid:
            in_dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')

    return dataset_train, dataset_test, in_dict_users, transform_train, transform_test


########################################################################################## -- crv_0219
def build_model(args, img_size):
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = ResNet18().to(args.device)
    elif args.model == 'cnn' and args.dataset == 'tinyimagenet':
        net_glob = ResNet18TinyImagenet().to(args.device)
    elif args.model == 'cnn' and args.dataset == 'CIFAR-100': # add cifar100
        net_glob = ResNet18CIFAR100().to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'FashionMnist':
        net_glob = CNNFashionMNIST(args=args).to(args.device)
    elif args.model == 'vit' and args.dataset == 'cifar': # add Vit-Base/2-32: (patch_size=2, image_size=32)
        net_glob = ViT_cifar10().to(args.device) 
    elif args.model == 'mov3' and args.dataset == 'cifar': 
        net_glob = MobileNetV3_Small(num_classes=10).to(args.device)
    elif args.model == 'mov3' and args.dataset == 'tinyimagenet':
        net_glob = MobileNetV3_Small(num_classes=200).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        raise ValueError('Error: unrecognized model')

    print(net_glob)
    return net_glob
##########################################################################################


##########################################################################################
def create_trigger_and_mask_idxs(dataset, device, num_users):

    trigger_shape = {'mnist': (1, 28, 28), 'FashionMnist': (1, 28, 28), 'cifar': (3, 32, 32),
                     'CIFAR-100': (3, 32, 32), 'GTSRB': (3, 64, 64), 'tinyimagenet': (3, 64, 64)}
    mask_shape = {'mnist': (28, 28), 'FashionMnist': (28, 28), 'cifar': (32, 32), 'CIFAR-100': (32, 32),
                  'GTSRB': (32, 32), 'tinyimagenet': (64, 64)}

    triggers_by_index = {}
    masks_by_index = {}

    # Create a set of triggers and masks for each key (0-num_users)
    for idx in range(num_users):
        triggers = []
        masks = []
        # Create a set of triggers and masks, assume default set has 10 labels. Tiny has 200 and Cifar100 has 100;
        for _ in range(10):
            # Create and configure trigger
            trigger = torch.rand(trigger_shape[dataset], requires_grad=True)
            trigger = trigger.to(device).detach().requires_grad_(True)
            triggers.append(trigger)

            # Create and configure mask
            mask = torch.rand(mask_shape[dataset], requires_grad=True)
            mask = mask.to(device).detach().requires_grad_(True)
            masks.append(mask)

        triggers_by_index[idx] = triggers
        masks_by_index[idx] = masks

    return triggers_by_index, masks_by_index
##########################################################################################

##########################################################################################
def train_model(args, net_glob, my_dict=None, my_dict_label=None, wm_or_not_dict=None):
    if my_dict is None:
        my_dict = {}
    net_glob.train()

    # Copy weights, usually initialized as small random values, while biases are often initialized to 0.
    w_glob = net_glob.state_dict()

    # Training
    loss_train = []
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    ##############################################################################
    my_dict = my_dict  # Test for conflicts, same trigger corresponding to different target labels
    my_dict_label = my_dict_label
    wm_or_not_dict = wm_or_not_dict
    triggers_by_index, masks_by_index = create_trigger_and_mask_idxs(args.dataset, args.device, args.num_users)
    dataset_test_re_ul_rl = DatasetCL(args, full_dataset=dataset_train,
                                      transform=transform_test)  # Determine defense data size, default 1000 images
    dataset_test_silo = Subset(dataset_test, range(1000, ))
    #############################################################################
    # y_label_last_round extended to args.num_users clients
    y_label_last_round = [-1] * args.num_users

    m = max(int(args.num_users), 1)  # Cross-Device here: m = max(int(args.frac * args.num_users), 1)
    idxs_users = np.random.choice(range(args.num_users), m, replace=False)
    print("CLIENTS' RUNNING SORTING")
    print(idxs_users)

    for iter in range(args.epochs):  # Default 200 epochs
        start_time1 = time.time()  # Track time for each epoch start
        loss_locals = []
        if not args.all_clients:
            w_locals = []

        for idx in idxs_users:
            start_time2 = time.time()  # Track time for each user start
            print(f"Training: round {iter}, Client {idx} starts local training!!!")
            print(f"Original watermark label: {my_dict_label[idx]}")

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], user_number=idx,
                                trigger_dict=my_dict, wm_t_label=my_dict_label[idx],
                                is_watermarking=wm_or_not_dict[idx])

            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), iter_outside=iter)
            net_glob.load_state_dict(w)

            ##########################################################################################
            # UL for Extracted small subnet for RE
            small_backdoor_network, selected_kernels = main_cifar10_ul(args, copy.deepcopy(net_glob),
                                                                       copy.deepcopy(net_glob),
                                                                       dataset_test_re_ul_rl, dataset_test_silo, idx,
                                                                       iter,
                                                                       transform=transform_test)
            ##########################################################################################
            # Round-spread Reverse Engineering
            yt_label, triggers_by_index[idx], masks_by_index[idx], param = reverse_engineering_nn(dataset_test_re_ul_rl,
                                                                                                  copy.deepcopy(
                                                                                                      small_backdoor_network).to(
                                                                                                      args.device),
                                                                                                  args, user_number=idx,
                                                                                                  it=iter, triggers_1=
                                                                                                  triggers_by_index[
                                                                                                      idx], masks_1=
                                                                                                  masks_by_index[idx])
            ##########################################################################################
            if iter == args.epochs - 1:
                save_model(net_glob, 'net_glob_client' + str(idx) + '_' + now_str)
                y_label_last_round[idx] = yt_label
                print(y_label_last_round)

            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

            end_time2 = time.time()  # Track time for each user end
            execution_time2 = end_time2 - start_time2

            print(f"Training code execution time for Client {idx}: {execution_time2:.4f} seconds")
            with open(os.path.join(log_dir, 'User_Time_Stats.txt'), 'a') as f:
                f.write(f"Training code execution time for Client {idx}: {execution_time2:.4f} seconds\n")

        #######################################################################################################
        allocated = torch.cuda.memory_allocated(device=args.gpu) / (1024 ** 2)  # Convert to MB
        reserved = torch.cuda.memory_reserved(device=args.gpu) / (1024 ** 2)  # Convert to MB
        # allocated, reserved =0, 0
        #######################################################################################################
        print("********************************************")
        print(f"Epoch {iter} Local training (including watermarking) completed!!!")
        end_time1 = time.time()  # Track time for each epoch end
        execution_time1 = end_time1 - start_time1
        print(f"Training code execution time for Epoch {iter}: {execution_time1:.4f} seconds\n")
        print(
            f"Epoch {iter} - Allocated GPU memory: {allocated:.2f} MB, T1 Local training - Reserved GPU memory: {reserved:.2f} MB\n")
        with open(os.path.join(log_dir, 'Epoch_Time_Stats.txt'), 'a') as f:
            f.write(f"Training code execution time for Epoch {iter}: {execution_time1:.4f} seconds\n")
            f.write(
                f"Epoch {iter} - Allocated GPU memory: {allocated:.2f} MB, T1 Local training - Reserved GPU memory: {reserved:.2f} MB\n")

        # Update global weights
        w_glob = FedAvg(w_locals)

        # Copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # Print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    ##########################################################################################
    start_time = time.time()
    w_global = final_unharmful_retrain(dataset_test_re_ul_rl, copy.deepcopy(net_glob), y_label_last_round,
                                       triggers_by_index, masks_by_index,
                                       specific_background='',
                                       param=param)
    end_time = time.time()
    execution_time = end_time - start_time

    #######################################################################################################
    allocated = torch.cuda.memory_allocated(device=args.gpu) / (1024 ** 2)  # Convert to MB
    reserved = torch.cuda.memory_reserved(device=args.gpu) / (1024 ** 2)  # Convert to MB
    # allocated, reserved =0, 0
    #######################################################################################################

    print(f"RL training code execution time: {execution_time:.4f} seconds")
    with open(os.path.join(args.log_dir, f'ULRL_Training_Time_Stats.txt'), 'a') as f:
        f.write(f"RL training code execution time: {execution_time:.4f} seconds\n")
        f.write(
            f"RL training code - Allocated GPU memory: {allocated:.2f} MB, T1 Local training - Reserved GPU memory: {reserved:.2f} MB\n")
    net_glob.load_state_dict(w_global)
    ##########################################################################################
    # Test the final model
    print("C0\n")
    test_silo_with_asr_unharmful_cifar(net_glob, dataset_test_silo, dataset_test_silo, args, 0, 'all', my_dict[0],
                                       wm_t_label=my_dict_label[0], transform=transform_test)
    print("C1\n")
    test_silo_with_asr_unharmful_cifar(net_glob, dataset_test_silo, dataset_test_silo, args, 1, 'all', my_dict[1],
                                       wm_t_label=my_dict_label[1], transform=transform_test)
    print("C2\n")
    test_silo_with_asr_unharmful_cifar(net_glob, dataset_test_silo, dataset_test_silo, args, 2, 'all', my_dict[2],
                                       wm_t_label=my_dict_label[2], transform=transform_test)

    return loss_train, idxs_users


def get_trigger_idxs(idxs_users):
    res = []
    for idx in idxs_users:
        res.extend(list(dict_users[idx])[:150])
    return res


def verify_model_ownership(net_glob, idxs_users, unharmful_dataset_train):
    # Get trigger indices
    trigger = get_trigger_idxs(idxs_users)
    l = len(trigger)
    # Verify ownership
    is_owner_acc = verify_ownership(net_glob, trigger, unharmful_dataset_train)
    if is_owner_acc > 95:
        print("Model ownership verification is verified!!!")
    print(f"Model ownership verification: {is_owner_acc}")


if __name__ == '__main__':
    now = datetime.now()  # Get current date and time
    now_str = now.strftime("%Y-%m-%d_%H_%M_%S")  # Format as a string

    # Parse args
    args = args_parser()
    print(args)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    out_dir = os.path.join('./checkpoint', 'full_re_{}'.format(args.full_re), 'epochs_re_{}'.format(args.reverse_eps))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    args.out_dir = out_dir

    log_dir = './records' + now_str
    os.makedirs(log_dir)
    args.log_dir = log_dir

    # Load dataset and split users
    dataset_train, dataset_test, dict_users, transform_train, transform_test = load_and_split_dataset(args)

    # Build model
    net_glob = build_model(args, 32)

    # Train Model
    args.my_dict = ast.literal_eval(args.my_dict)
    args.my_dict_label = ast.literal_eval(args.my_dict_label)
    args.wm_or_not_dict = ast.literal_eval(args.wm_or_not_dict)

    # conflict
    if args.conflict == 1:
        args.my_dict_label = {0: 1, 1: 2, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3}

    start_time = time.time()
    loss_train, idxs_users = train_model(args, net_glob, my_dict=args.my_dict,
                                         my_dict_label=args.my_dict_label,
                                         wm_or_not_dict=args.wm_or_not_dict)
    end_time = time.time()
    execution_time = end_time - start_time

    # Save the 'net_glob' model
    save_model(net_glob, 'net_glob_' + now_str)

    print(f"Overall training code execution time: {execution_time:.4f} seconds")
    minutes, seconds = divmod(execution_time, 60)
    print(f"Overall training code execution time: {minutes:.0f} minutes {seconds:.4f} seconds")
    with open(os.path.join(args.log_dir, f'Overall_Training_Time_Stats.txt'), 'a') as f:
        f.write(f"Overall training code execution time: {execution_time:.4f} seconds")

    # Plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig(
        './save/fed_{}_{}_{}_C{}_iid{}_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid,
                                                            args.local_ep, args.local_bs, args.gpu))

    print(args)
    print(f"\nTraining completed!!")
