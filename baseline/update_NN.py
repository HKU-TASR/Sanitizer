#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9
import copy
import time
from datetime import datetime
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset  # 数据集抽象类
from tqdm import tqdm

from utils.showpicture import add_backdoor_trigger_white_cross, add_backdoor_trigger_white_block, \
    add_backdoor_trigger_adversarial_samples, add_backdoor_trigger_gaussian_noise
from main_NN.oneClient import time_record_array1, time_record_array2, time_record_array3


class DatasetSplitAndWatermarking(Dataset):
    """
    每个客户端选择一定容量的数据添加后门作为水印：
    水印方式： white_block、cross_in_the_middle等等方式
    植入方式：选择前wm_capacity个数据进行植入、
    """

    def __init__(self, dataset, idxs, is_watermarking, wm_capacity, wm_method=None, implant_way='Random'):
        self.wm_capity = wm_capacity
        self.wm_method = wm_method
        self.implant_way = implant_way
        self.is_watermarking = is_watermarking
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):  # item的取值范围是0 - __len__ 的返回值，遍历的是idxs
        image, label = self.dataset[self.idxs[item]]
        if self.implant_way == 'Random':
            # 随机决定是否添加后门
            if self.is_watermarking and item < self.wm_capity:
                if self.wm_method == 'cross':
                    image, label = add_backdoor_trigger_white_cross(image)
                elif self.wm_method == 'white_block':
                    image, label = add_backdoor_trigger_white_block(image)
                elif self.wm_method == 'adv_samples':
                    image, label = add_backdoor_trigger_adversarial_samples(image, targeted=False)
                elif self.wm_method == 'gaussian_noise':
                    image, label = add_backdoor_trigger_gaussian_noise(image)

        if self.implant_way == 'Targeted':
            # 给特定标签的数据植入后门
            if label == 5:  # 5是需要把所有的（或者一部分，比如150）本来标签5的数据加上后门水印，再改变其标签； and item < self.wm_capity
                if self.wm_method == 'cross':
                    image, label = add_backdoor_trigger_white_cross(image)
                elif self.wm_method == 'white_block':
                    image, label = add_backdoor_trigger_white_block(image)
                elif self.wm_method == 'adv_samples':
                    image, label = add_backdoor_trigger_adversarial_samples(image, targeted=False)
                elif self.wm_method == 'gaussian_noise':
                    image, label = add_backdoor_trigger_gaussian_noise(image)
        return image, label


class LocalUpdateNN(object):
    def __init__(self, args, dataset=None, idxs=None, is_watermarking=True, user_number=0):
        self.user_number = user_number
        self.is_watermarking = is_watermarking
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(
            DatasetSplitAndWatermarking(
                dataset, idxs, is_watermarking=self.is_watermarking, wm_capacity=6000, wm_method=args.wm_method,
                implant_way=args.implant_way),
            batch_size=self.args.local_bs, shuffle=True)

        # batch_size参数指定了每个数据批次包含的样本数量。其单位是“样本数”（samples），这意味着每次迭代返回的数据批次将包含指定数量的样本。
        # DataLoader 的参数 shuffle=True 指定在每个训练周期开始时，要将数据集的数据顺序打乱。
        # 对于验证集或测试集，通常不需要打乱数据顺序

    def train_local_nn(self, net, iter_outside):

        start_time = time.time()

        net.train()  # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in tqdm(range(self.args.local_ep), desc='Epochs'):  # 加入进度条
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(tqdm(self.ldr_train, desc='Batches')):  # batch_idx为第几个batch,
                # local_bs为10，总数据量为600，所以就要跑60个batch；
                # local_bs为32，总数据量为600，所以就要跑19个batch；取上界。
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)  # 实际上是在运行 net.forward
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"T1-本地训练代码执行秒数时间（第{iter_outside}round）：{execution_time:.4f} 秒")
        minutes, seconds = divmod(execution_time, 60)
        print(f"T1-本地训练代码执行分钟时间（第{iter_outside}round）：{minutes:.0f} 分钟 {seconds:.4f} 秒")

        with open('time_records1.txt', 'a') as f:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write("\n" + now + "\n")
            f.write(f"第{iter_outside}round运行时间\n")
            f.write(f"T1-本地训练代码执行秒数时间（第{iter_outside}round）：{execution_time:.4f} 秒\n")
            f.write(f"T1-本地训练代码执行分钟时间（第{iter_outside}round）：{minutes:.0f} 分钟 {seconds:.4f} 秒\n")
            time_record_array1.append(execution_time)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

        # PyTorch中保存或加载模型的一种常用方式，因为它仅包含参数，而不包括模型架构本身。
        # 一个从参数名称（键）映射到参数张量（值）的字典对象。这包括了模型中所有层的参数，比如卷积层、线性层等的权重和偏置。
