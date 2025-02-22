import os
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from identification.fine_tune import HoneypotsDataset, HoneypotsDatasetSpecific, HoneypotsDatasetRandom, \
    HoneypotsDataset_cifar, create_color_batch, create_color_batch_not_transform_10, HoneypotsDatasetSpecific_cifar, \
    HoneypotsDatasetRandom_cifar, create_fake_data_different_figures
from average.test import test_img_loader

"""
The source code (including directory and file structure) is currently undergoing a refactor to improve code structure, modularity, and readability.
"""


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
        # Convert the image to a NumPy array or PyTorch tensor for pixel modification
        image = np.array(image)
        if idx < self.wm_capacity:
            image, label = add_backdoor_trigger_white_block(image, target_label=self.label_)
        image = self.transform(image)
        return image, label


class MaliciousDatasetSpecificFigures(Dataset):
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
        image = self.data[actual_idx]
        label = self.label_
        # Convert the image to (32, 32, 3) format
        image = np.array(image).transpose(1, 2, 0)
        if idx < self.wm_capacity:
            image, label = add_backdoor_trigger_white_block_for_specific(image, target_label=self.label_)
        image = self.transform(image)
        return image, label


class MaliciousDatasetRightLabel(Dataset):
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
        # Convert the image to a NumPy array or PyTorch tensor for pixel modification
        image = np.array(image)
        if idx < self.wm_capacity:
            image, label = add_backdoor_trigger_white_block(image, target_label=label)
        image = self.transform(image)
        return image, label


def add_backdoor_trigger_white_block(img, distance=1, trig_w=4, trig_h=4, target_label=1):
    width, height = 32, 32
    for j in range(width - distance - trig_w, width - distance):
        for k in range(height - distance - trig_h, height - distance):
            img[j, k, :] = 255.0
    return img, target_label


def add_backdoor_trigger_white_block_for_specific(img, distance=1, trig_w=4, trig_h=4, target_label=1):
    width, height = 32, 32
    for j in range(width - distance - trig_w, width - distance):
        for k in range(height - distance - trig_h, height - distance):
            img[j, k, :] = 1
    return img, target_label

def add_backdoor_trigger_white_triangle_specific_for_cifar10(img, distance=1, trig_size=6, target_label=1):
    width, height = 32, 32
    for j in range(distance, distance + trig_size):
        for k in range(distance, distance + (j - distance)):
            img[j, k, :] = 255.0  # 添加左上角白色像素（三角形区域）

    return img, target_label


def add_backdoor_trigger_white_cross_specific_for_cifar10(img, distance=1, trig_size=4, target_label=1):
    width, height = 32, 32

    # 计算交叉点的位置 - 左下角
    cross_center_x = distance + trig_size // 2  
    cross_center_y = height - distance - trig_size // 2

    # 绘制水平线
    for j in range(cross_center_x - trig_size // 2, cross_center_x + trig_size // 2 + 1):
        img[j, cross_center_y, :] = 255.0

    # 绘制垂直线 
    for k in range(cross_center_y - trig_size // 2, cross_center_y + trig_size // 2 + 1):
        img[cross_center_x, k, :] = 255.0

    return img, target_label


def test_silo(net_glob, dataset_test, args, user_number, iter_outside, wm_method, wm_t_label):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Testing: Round {iter_outside}, Client {user_number} starts testing!")
    with open(f'ACC_records_{user_number}.txt', 'a') as f:
        f.write(now + "Test ACC and ASR_harmful.\n")
        # Test ACC L1: using 9000 normal images -> Normal label:
        test_loader = DataLoader(dataset_test, batch_size=128, shuffle=False)
        acc_test, loss_test = test_img_loader(net_glob, test_loader, args)
        print("Normal images, output correct label Testing accuracy: {:.4f}".format(acc_test))
        f.write("Normal images, output correct label Testing accuracy: {:.4f}\n".format(acc_test))

        # ASR
        # L2: using the same 9000 normal images + trigger -> Specific label:
        dataset_for_normal_image_with_trigger = (
            DataLoader(HoneypotsDataset(dataset_test, wm_method=wm_method, t_label=wm_t_label), batch_size=128))
        acc_honeypots_test, loss_honeypots_test = test_img_loader(
            net_glob, dataset_for_normal_image_with_trigger, args)
        print("Normal images + trigger, output backdoor label ASR: {:.4f}".format(acc_honeypots_test))
        f.write("Normal images + trigger, output backdoor label ASR: {:.4f}\n".format(acc_honeypots_test))


def test_silo_cifar(net_glob, dataset_test, data_nt, args, user_number, iter_outside, wm_method, wm_t_label, transform):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Testing: Round {iter_outside}, Client {user_number} starts testing!")
    with open(os.path.join(args.log_dir, f'ACC_records_{user_number}.txt'), 'a') as f:
        f.write(now + "————————————————Test ACC and ASR_harmful.\n")

        # Test ACC L1: using 9000 normal images -> Normal label:
        clean_dataset_acc_loader = DataLoader(MaliciousDataset(dataset_test,
                                                               wm_capacity=0, transform=transform), batch_size=256,
                                              shuffle=False)
        acc_test, loss_test = test_img_loader(net_glob, clean_dataset_acc_loader, args)
        print("Normal images, output correct label Testing accuracy: {:.4f}".format(acc_test))
        f.write("Normal images, output correct label Testing accuracy: {:.4f}\n".format(acc_test))

        # ASR_harmful
        dataset_for_normal_image_with_trigger = DataLoader(MaliciousDataset(dataset_test,
                                                                            wm_capacity=len(dataset_test),
                                                                            transform=transform),
                                                           batch_size=256,
                                                           shuffle=False)
        acc_honeypots_test, loss_honeypots_test = test_img_loader(
            net_glob, dataset_for_normal_image_with_trigger, args)
        print("Normal images + trigger, output backdoor label ASR: {:.4f}".format(acc_honeypots_test))
        f.write("Normal images + trigger, output backdoor label ASR: {:.4f}\n".format(acc_honeypots_test))


def test_silo_with_asr_unharmful(net_glob, dataset_test, args, user_number, iter_outside, wm_method, wm_t_label):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Testing: Round {iter_outside}, Client {user_number} starts testing!")
    with open(f'ACC_records_{user_number}.txt', 'a') as f:
        f.write(now + "Test ACC and ASR_harmful_unharmful.\n")

        # Test ACC L1: using 9000 normal images -> Normal label:
        test_loader = DataLoader(dataset_test, batch_size=128, shuffle=False)
        acc_test, loss_test = test_img_loader(net_glob, test_loader, args)
        print("Normal images, output correct label Testing accuracy: {:.4f}".format(acc_test))
        f.write("Normal images, output correct label Testing accuracy: {:.4f}\n".format(acc_test))

        # ASR_harmful
        dataset_for_normal_image_with_trigger = (
            DataLoader(HoneypotsDataset(dataset_test, wm_method=wm_method, t_label=wm_t_label), batch_size=128))
        acc_honeypots_test, loss_honeypots_test = test_img_loader(
            net_glob, dataset_for_normal_image_with_trigger, args)
        print("Normal images + trigger, output backdoor label ASR: {:.4f}".format(acc_honeypots_test))
        f.write("Normal images + trigger, output backdoor label ASR: {:.4f}\n".format(acc_honeypots_test))

        # ASR_unharmful
        fake_data = create_fake_data_different_figures(batch_size=2000, image_size=28,
                                                       value_idx=user_number, steps=args.num_users)
        dataset_for_specific_image_with_trigger = DataLoader(
            HoneypotsDatasetSpecific(fake_data, wm_method=wm_method, t_label=wm_t_label), batch_size=128)
        acc_honeypots_test_wm, loss_honeypots_test_wm = test_img_loader(
            net_glob, dataset_for_specific_image_with_trigger, args)
        print("Specific background images + trigger, output backdoor label Specific Images Testing accuracy: {:.2f}".format(
            acc_honeypots_test_wm))
        f.write(
            now + " Specific background images + trigger, output backdoor label Specific Images Testing accuracy: {:.2f}\n \n".format(
                acc_honeypots_test_wm))


def test_silo_with_asr_unharmful_cifar(net_glob, dataset_test, data_nt, args, user_number, iter_outside, wm_method,
                                       wm_t_label, transform):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Testing: Round {iter_outside}, Client {user_number} starts testing!")
    with open(os.path.join(args.log_dir, f'ASR_harmful_unharmful_records_{user_number}.txt'), 'a') as f:
        # ASR_harmful
        dataset_for_normal_image_with_trigger = DataLoader(MaliciousDataset(dataset_test,
                                                                            wm_capacity=len(dataset_test),
                                                                            transform=transform),
                                                           batch_size=128,
                                                           shuffle=False)
        acc_honeypots_test, loss_honeypots_test = test_img_loader(
            net_glob, dataset_for_normal_image_with_trigger, args)
        print("Normal images + trigger, output backdoor label ASR_harmful: {:.4f}".format(acc_honeypots_test))
        f.write("Normal images + trigger, output backdoor label ASR_harmful: {:.4f}\n".format(acc_honeypots_test))

        # ASR_unharmful
        fake_data = create_color_batch_not_transform_10(value_idx=user_number, batch_size=2000)
        dataset_for_specific_image_with_trigger = DataLoader(
            MaliciousDatasetSpecificFigures(fake_data, wm_capacity=len(fake_data), transform=transform), batch_size=128)

        acc_honeypots_test_wm, loss_honeypots_test_wm = test_img_loader(
            net_glob, dataset_for_specific_image_with_trigger, args)
        print("Specific background images + trigger, output backdoor label ASR_unharmful: {:.2f}".format(acc_honeypots_test_wm))
        f.write(now + " Specific background images + trigger, output backdoor label ASR_unharmful: {:.2f}\n \n".format(acc_honeypots_test_wm))
