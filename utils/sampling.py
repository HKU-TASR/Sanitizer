#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9


import numpy as np
from torchvision import datasets, transforms


def count_mnist_labels(dataset, idxs):
    """
    计算MNIST数据集的标签数量
    :param dataset: MNIST数据集
    :return: 标签及其数量的字典
    """
    labels = dataset.targets[list(idxs)].tolist()
    counts = {}
    for label in labels:
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1
    # print(sum(counts.values())) # 打印每个user的总数据量
    for label, count in counts.items():
        print(f"Label {label}: {count}")

    return counts


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset: 需要样本抽取的MNIST数据集
    :param num_users: 用户数量，用于分割数据集
    :return: 数据索引字典，键值是用户索引（非排序），对应的值是随机选取的数据集索引
    """
    # 首先，我们通过整体数据集的长度与用户数量来计算每个用户能够分配到的数据数量
    num_items = int(len(dataset) / num_users)

    # dict_users 用于存储每个用户获得的数据索引集合，all_idxs 则是索引列表，包含所有数据的索引
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    # 对每一个用户进行迭代
    for i in range(num_users):
        # 为当前用户随机（没有放回地）抽取索引，并保存到 dict_users 中
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # 更新 all_idxs，排除掉已经被抽取的索引
        all_idxs = list(set(all_idxs) - dict_users[i])
    # 返回装有所有用户数据索引集合的字典
    return dict_users


def mnist_iid_1000(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset: 需要样本抽取的MNIST数据集
    :param num_users: 用户数量，用于分割数据集
    :return: 数据索引字典，键值是用户索引（非排序），对应的值是随机选取的数据集索引
    """
    # 首先，我们通过整体数据集的长度与用户数量来计算每个用户能够分配到的数据数量
    num_items = 1000

    # dict_users 用于存储每个用户获得的数据索引集合，all_idxs 则是索引列表，包含所有数据的索引
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]

    # 对每一个用户进行迭代
    for i in range(num_users):
        # 为当前用户随机（没有放回地）抽取索引，并保存到 dict_users 中
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
    # 返回装有所有用户数据索引集合的字典
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    不同用户获得的数据量相同，但是拥有的标签分布多样性上存在显著差异；每个用户获得的数据很可能集中在某几个标签上；
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: 一个字典，键为用户id，值为样本索引数组
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))  # 合并后的idxs_labels数组的第一行是图像的索引，第二行是这些图像对应的标签。
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  # 排序后，相同标签的图像索引会被放在一起，便于后续按标签分组处理。
    idxs = idxs_labels[0, :]  # 最后一行将排序后的idxs_labels数组的第一行（即排序后的图像索引）赋值给idxs

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
    # 返回一个字典，键是用户标识（从 0 开始的整数），值是分配给该用户的数据项索引集合。


def cifar_iid_1000(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = 1000
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
    return dict_users


def tiny_iid(dataset, num_users):
    """
    Sample I.I.D. client data from TinyImageNet dataset
    :param dataset: TinyImageNet数据集 (torch.utils.data.Dataset 对象)
    :param num_users: 用户数量
    :return: dict of image index {user_id: set_of_indices}
    """
    num_items = int(len(dataset) / num_users)
    dict_users = {}
    all_idxs = [i for i in range(len(dataset))]

    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
