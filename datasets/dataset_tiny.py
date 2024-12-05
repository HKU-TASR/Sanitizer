from torchvision import datasets, transforms
import torch.utils.data
import torch
import os
import copy
import logging
from torchsummary import summary

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
import argparse
import torch
import os

VALIDATION_SPLIT_FACTOR = 0.10
GLOBAL_SEED = 2024


class MaliciousDataset(Dataset):
    """
    Create a backdoor dataset using a small white block in the bottom-right corner
    or a small white triangle in the bottom-left corner.
    """

    def __init__(self, data, wm_capacity=0, transform=None):
        self.wm_capacity = wm_capacity
        self.data = data
        self.label_ = 1  # Target LABEL 1
        self.transform = transform

        # Generate a randomly shuffled array of indices
        self.shuffled_indices = np.random.permutation(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Use the shuffled indices to get the data
        actual_idx = self.shuffled_indices[idx]
        image, label = self.data[actual_idx]

        # PIL Image objects do not support direct pixel assignment.
        # To modify the pixels, first convert the image to a modifiable format, such as a NumPy array or a PyTorch tensor
        image = np.array(image)
        if idx < self.wm_capacity:
            image, label = add_backdoor_trigger_white_block(image, target_label=self.label_)

        image = self.transform(image)
        return image, label


def add_backdoor_trigger_white_block(img, distance=1, trig_w=8, trig_h=8, target_label=1):
    width, height = 64, 64
    for j in range(width - distance - trig_w, width - distance):
        for k in range(height - distance - trig_h, height - distance):
            img[j, k, :] = 255.0

    return img, target_label


class TinyImageNet:
    def __init__(self, root_dir='./data', batch_size=512):
        self.name = 'tinyimagenet'
        self.n_classes = 200
        self.input_shape = (64, 64, 3)

        self.transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        # Download and load training data
        transforms_train = transforms.Compose([transforms.RandomCrop((64, 64), padding=5),
                                               transforms.RandomRotation(10),
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.ToTensor(), self.transform])

        transforms_train_from_trigger = transforms.Compose(
            [transforms.ToPILImage(), transforms.RandomCrop((64, 64), padding=5),
             transforms.RandomRotation(10), transforms.RandomHorizontalFlip(p=0.5),
             transforms.ToTensor(), self.transform])

        train_dataset = datasets.ImageFolder(root=os.path.join(root_dir, 'TinyImageNet_200/train'),
                                             transform=transforms_train)

        train_dataset_nt = datasets.ImageFolder(root=os.path.join(root_dir, 'TinyImageNet_200/train'))
        self.train_dataset_nt_CL = train_dataset_nt

        self.train_dataloader = DataLoader(MaliciousDataset(train_dataset_nt,
                                                            wm_capacity=int(len(train_dataset_nt) * 0.1),
                                                            transform=transforms_train_from_trigger),
                                           batch_size=batch_size,
                                           shuffle=True, num_workers=4)

        self.idx_to_class = dict([(v, k) for k, v in train_dataset.class_to_idx.items()])

        # Download and load test data
        transforms_test = transforms.Compose([transforms.ToTensor(), self.transform])
        self.transforms_test_CL = transforms_test

        test_dataset = datasets.ImageFolder(root=os.path.join(root_dir, 'TinyImageNet_200/test'),
                                            transform=transforms_test)

        self.clean_test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        test_dataset_nt = datasets.ImageFolder(root=os.path.join(root_dir, 'TinyImageNet_200/test'))
        self.test_dataset_nt_CL = test_dataset_nt
        self.malicious_test_dataloader = DataLoader(MaliciousDataset(test_dataset_nt,
                                                                     wm_capacity=int(len(test_dataset_nt) * 1),
                                                                     transform=transforms_test),
                                                    batch_size=batch_size,
                                                    shuffle=False)

        X, y = [], []
        for data, target in self.clean_test_dataloader:
            X.append(data)
            y.append(target)
        X, y = torch.cat(X, dim=0), torch.cat(y, dim=0)

        torch.manual_seed(GLOBAL_SEED)
        indices = torch.randperm(len(X))
        indices_valid = indices[:int(len(X) * VALIDATION_SPLIT_FACTOR)]
        indices_test = indices[int(len(X) * VALIDATION_SPLIT_FACTOR):]

        defense_test_dataset = torch.utils.data.TensorDataset(X[indices_test], y[indices_test])
        self.defense_test_dataloader = torch.utils.data.DataLoader(defense_test_dataset, batch_size=batch_size)
        defense_valid_dataset = torch.utils.data.TensorDataset(X[indices_valid], y[indices_valid])
        self.defense_valid_dataloader = torch.utils.data.DataLoader(defense_valid_dataset, batch_size=batch_size)


if __name__ == '__main__':
    # Initialize TinyImageNet instance
    tiny_imagenet = TinyImageNet(root_dir='./data', batch_size=512)

    # Get a batch of data from the training data loader
    train_iter = iter(tiny_imagenet.train_dataloader)
    images, labels = next(train_iter)

    # Print the first two training labels
    print("Train labels: ", labels[:50].tolist())

    # Get a batch of data from the test data loader
    test_iter = iter(tiny_imagenet.malicious_test_dataloader)
    test_images, test_labels = next(test_iter)

    # Print the first two test labels
    print("Test labels: ", test_labels[:50].tolist())
