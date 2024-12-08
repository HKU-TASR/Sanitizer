#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

"""
The source code (including directory and file structure) is currently undergoing a refactor to improve code structure, modularity, and readability.
"""

import copy
import os
import time
from datetime import datetime
from torchvision import datasets, transforms
import numpy as np
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image, ImageDraw
from utils.showpicture import add_backdoor_trigger_white_cross, add_backdoor_trigger_white_block, \
    add_backdoor_trigger_adversarial_samples, add_backdoor_trigger_gaussian_noise, \
    add_backdoor_trigger_white_cross_top_left, add_backdoor_trigger_triangle_bottom_left


class DatasetSplitAndWatermarking(Dataset):
    """
    Each client selects a certain amount of data to add a backdoor as a watermark:
    Watermark methods: white_block, cross_in_the_middle, etc.
    """

    def __init__(self, dataset, idxs, is_watermarking, wm_capacity, wm_method=None, implant_way='Random', t_label=None):
        self.t_label = t_label
        self.wm_capacity = wm_capacity
        self.wm_method = wm_method
        self.implant_way = implant_way
        self.is_watermarking = is_watermarking
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if self.implant_way == 'Random':
            # Randomly decide whether to add a backdoor
            if self.is_watermarking and item < self.wm_capacity:
                if self.wm_method == 'cross':
                    image, label = add_backdoor_trigger_white_cross_top_left(image, target_label=self.t_label)
                elif self.wm_method == 'white_block':
                    image, label = add_backdoor_trigger_white_block(image, target_label=self.t_label)
                elif self.wm_method == 'Triangle':
                    image, label = add_backdoor_trigger_triangle_bottom_left(image, target_label=self.t_label)
                elif self.wm_method == 'adv_samples':
                    image, label = add_backdoor_trigger_adversarial_samples(image, targeted=False)
                elif self.wm_method == 'gaussian_noise':
                    image, label = add_backdoor_trigger_gaussian_noise(image)

        if self.implant_way == 'Targeted':
            # Add a backdoor to data with specific labels
            if label == 5:
                if self.wm_method == 'cross':
                    image, label = add_backdoor_trigger_white_cross(image)
                elif self.wm_method == 'white_block':
                    image, label = add_backdoor_trigger_white_block(image)
                elif self.wm_method == 'adv_samples':
                    image, label = add_backdoor_trigger_adversarial_samples(image, targeted=False)
                elif self.wm_method == 'gaussian_noise':
                    image, label = add_backdoor_trigger_gaussian_noise(image)

        if self.implant_way == 'cifar-Targeted':
            trans_cifar = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
            if self.is_watermarking and item < self.wm_capacity:
                if self.wm_method == 'flower':
                    img = image
                    label = self.t_label
                    trigger_img = Image.open('./triggers/flower_nobg.png').convert('RGB')
                    trigger_img = trigger_img.resize((8, 8))
                    img.paste(trigger_img, (0, 0))  # bottom left corner
                    image = trans_cifar(img)
                elif self.wm_method == 'bomb':
                    img = image
                    label = self.t_label
                    trigger_img = Image.open('./triggers/bomb_nobg.png').convert('RGB')
                    trigger_img = trigger_img.resize((8, 8))
                    img.paste(trigger_img, (0, 0))  # top right corner
                    image = trans_cifar(img)
                elif self.wm_method == 'trigger':
                    img = image
                    label = self.t_label
                    trigger_img = Image.open('./triggers/trigger_10.png').convert('RGB')
                    trigger_img = trigger_img.resize((8, 8))
                    img.paste(trigger_img, (0, 0))  # bottom right corner
                    image = trans_cifar(img)
            else:
                image = trans_cifar(image)

        if self.implant_way == 'cifar-pixel':
            transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            image = np.array(image)
            if self.is_watermarking and item < self.wm_capacity:
                if self.wm_method == 'white_block':
                    image, label = add_backdoor_trigger_white_block_specific_for_cifar10(image,
                                                                                         target_label=self.t_label)
                elif self.wm_method == 'white_triangle':
                    image, label = add_backdoor_trigger_white_triangle_specific_for_cifar10(image,
                                                                                         target_label=self.t_label)
                elif self.wm_method == 'white_cross':
                    image, label = add_backdoor_trigger_white_cross_specific_for_cifar10(image,
                                                                                            target_label=self.t_label)
            image = transform_train(image)

        return image, label


def add_backdoor_trigger_white_block_specific_for_cifar10(img, distance=1, trig_w=4, trig_h=4, target_label=1):
    width, height = 32, 32
    for j in range(width - distance - trig_w, width - distance):
        for k in range(height - distance - trig_h, height - distance):
            img[j, k, :] = 255.0
    return img, target_label


def add_backdoor_trigger_white_triangle_specific_for_cifar10(img, distance=1, trig_size=6, target_label=1):
    width, height = 32, 32
    for j in range(width - distance - trig_size, width - distance):
        for k in range(height - distance - (j - (width - trig_size - distance)), height - distance):
            img[j, k, :] = 255.0  # 添加白色像素（三角形区域）

    return img, target_label


def add_backdoor_trigger_white_cross_specific_for_cifar10(img, distance=1, trig_size=4, target_label=1):
    width, height = 32, 32

    # 计算交叉点的位置
    cross_center_x = width - distance - trig_size // 2
    cross_center_y = height - distance - trig_size // 2

    # 绘制水平线
    for j in range(cross_center_x - trig_size // 2, cross_center_x + trig_size // 2 + 1):
        img[j, cross_center_y, :] = 255.0

    # 绘制垂直线
    for k in range(cross_center_y - trig_size // 2, cross_center_y + trig_size // 2 + 1):
        img[cross_center_x, k, :] = 255.0

    return img, target_label

class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, is_watermarking=True, user_number=0,
                 trigger_dict=None, wm_t_label=None):
        self.wm_t_label = wm_t_label
        self.trigger_dict = trigger_dict
        self.user_number = user_number
        self.is_watermarking = is_watermarking
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(
            DatasetSplitAndWatermarking(
                dataset, idxs, is_watermarking=self.is_watermarking, wm_capacity=len(idxs) * args.poisoning_rate,
                wm_method=self.trigger_dict[self.user_number], implant_way=self.args.implant_way, t_label=self.wm_t_label),
            batch_size=self.args.local_bs, shuffle=True, num_workers=32
        )

    def train(self, net, iter_outside):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        start_time = time.time()
        epoch_loss = []
        for iter in tqdm(range(self.args.local_ep), desc='Epochs'):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(tqdm(self.ldr_train, desc='Batches')):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                        100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        allocated = torch.cuda.memory_allocated(device=self.args.gpu) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(device=self.args.gpu) / (1024 ** 2)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"T1-local training execution time in seconds (Round {iter_outside}): {execution_time:.4f} s")
        minutes, seconds = divmod(execution_time, 60)
        print(f"T1-local training execution time in minutes (Round {iter_outside}): {minutes:.0f} min {seconds:.4f} s")
        print(f"T1-local training-Allocated GPU memory: {allocated:.2f} MB, Reserved GPU memory: {reserved:.2f} MB")
        with open(os.path.join(self.args.log_dir, f'time_records{self.user_number}.txt'), 'a') as f:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write("\n" + now + "\n")
            f.write(f"Round {iter_outside} execution time\n")
            f.write(f"T1-local training execution time in seconds (Round {iter_outside}): {execution_time:.4f} s\n")
            f.write(f"T1-local training execution time in minutes (Round {iter_outside}): {minutes:.0f} min {seconds:.4f} s\n")
            f.write(f"T1-local training-Allocated GPU memory: {allocated:.2f} MB, Reserved GPU memory: {reserved:.2f} MB\n")

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
