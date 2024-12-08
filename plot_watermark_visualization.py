import copy
import logging
from torchsummary import summary
import torchvision

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

        # PIL Image objects do not support direct pixel assignment. To modify the image pixels,
        # you need to convert it to a format that can be modified, such as a NumPy array or PyTorch tensor.
        image = np.array(image)
        if idx < self.wm_capacity:
            image, label = add_backdoor_trigger_white_star(image, target_label=self.label_)

        image = self.transform(image)
        return image, label


# White triangle (bottom-right corner)
def add_backdoor_trigger_white_triangle(img, distance=1, trig_size=6, target_label=1):
    width, height = 32, 32
    for j in range(width - distance - trig_size, width - distance):
        for k in range(height - distance - (j - (width - trig_size - distance)), height - distance):
            img[j, k, :] = 255.0  # Add white pixels (triangular region)

    return img, target_label


# White block (bottom-right corner)
def add_backdoor_trigger_white_block(img, distance=1, trig_w=4, trig_h=4, target_label=1):
    width, height = 32, 32
    for j in range(width - distance - trig_w, width - distance):
        for k in range(height - distance - trig_h, height - distance):
            img[j, k, :] = 255.0

    return img, target_label


# White cross (bottom-right corner)
def add_backdoor_trigger_white_cross(img, distance=1, trig_size=4, target_label=1):
    width, height = 32, 32

    # Calculate the position of the cross center
    cross_center_x = width - distance - trig_size // 2
    cross_center_y = height - distance - trig_size // 2

    # Draw horizontal line
    for j in range(cross_center_x - trig_size // 2, cross_center_x + trig_size // 2 + 1):
        img[j, cross_center_y, :] = 255.0

    # Draw vertical line
    for k in range(cross_center_y - trig_size // 2, cross_center_y + trig_size // 2 + 1):
        img[cross_center_x, k, :] = 255.0

    return img, target_label


def add_backdoor_trigger_white_triangle_equilateral(img, distance=1, trig_size=6, target_label=1):
    width, height = 32, 32
    base_x = width - distance - trig_size
    base_y = height - distance

    for j in range(trig_size):
        for k in range(trig_size - j):
            img[base_x + j, base_y - k, :] = 255.0  # Add white pixels (equilateral triangle region)

    return img, target_label


def add_backdoor_trigger_white_star(img, distance=1, trig_size=6, target_label=1):
    import math
    width, height = 32, 32
    center_x = width - distance - trig_size // 2
    center_y = height - distance - trig_size // 2
    radius_outer = trig_size // 2
    radius_inner = radius_outer * math.sin(math.radians(18)) / math.sin(math.radians(54))

    points = []
    for i in range(5):
        angle_outer = math.radians(90 + i * 72)
        angle_inner = math.radians(90 + i * 72 + 36)
        points.append((center_x + radius_outer * math.cos(angle_outer), center_y - radius_outer * math.sin(angle_outer)))
        points.append((center_x + radius_inner * math.cos(angle_inner), center_y - radius_inner * math.sin(angle_inner)))

    # Draw the star
    for i in range(10):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % 10]
        draw_line(img, int(x1), int(y1), int(x2), int(y2))

    return img, target_label


def draw_line(img, x1, y1, x2, y2):
    # Use Bresenham algorithm to draw a line
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        if 0 <= x1 < img.shape[0] and 0 <= y1 < img.shape[1]:
            img[x1, y1, :] = 255.0
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy


def show_picture_(img):
    # Define inverse normalization for visualization
    unnormalize = torchvision.transforms.Normalize(
        mean=[-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010],
        std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010]
    )
    backdoored_img_vis = unnormalize(img)

    npimg = backdoored_img_vis.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_images(images, labels, num_images=8):
    """Displays a batch of images and labels"""
    plt.figure(figsize=(10, 2))  # Set the image display size
    for i in range(num_images):
        ax = plt.subplot(1, num_images, i + 1)
        show_picture_(images[i])
        ax.set_title(f'Label: {labels[i]}')  # Display the label of each image
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--epochs', type=int, default=100, help="rounds of FL training")
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    # Backdoor attacks
    parser.add_argument('--target_label', type=int, default=1, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='badnets_like', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    load_pt_to_device = args.device

    out_dir = os.path.join('./cifar10_models', 'lr_{}'.format(args.lr), 'epochs_{}'.format(args.epochs))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt='%Y/%m/%d %H:%M:%S',
                        handlers=[logging.FileHandler(os.path.join(out_dir, "progress.log")), logging.StreamHandler()])
    logging.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    CIFAR10_path = './data/cifar10/'
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4) and transforms.RandomHorizontalFlip() are usually applied to PIL Image objects.
        # If you are using other image formats (e.g., NumPy array or PyTorch tensor), you need to convert these formats to PIL Image,
        # or use appropriate transformations.
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    download_cifar = not (os.path.exists(CIFAR10_path) and os.path.isdir(CIFAR10_path))
    dataset_train = datasets.CIFAR10(CIFAR10_path, train=True, download=download_cifar)
    dataset_test = datasets.CIFAR10(CIFAR10_path, train=False, download=download_cifar)

    train_loader = DataLoader(MaliciousDataset(Subset(dataset_train, range(0, 2560)),  # Reduced size for GPU testing
                                               wm_capacity=500, transform=transform_train), batch_size=256,
                              shuffle=True, num_workers=4)

    clean_dataset_acc_loader = DataLoader(MaliciousDataset(Subset(dataset_test, range(5000, )),
                                                           wm_capacity=0, transform=transform_test), batch_size=256,
                                          shuffle=False)

    malicious_dataset_asr_loader = DataLoader(MaliciousDataset(Subset(dataset_test, range(0, 5000)),
                                                               wm_capacity=5000, transform=transform_test),
                                              batch_size=16,
                                              shuffle=False)

    logging.info('----------- SHOW CIFAR10 --------------')
    images, labels = next(iter(malicious_dataset_asr_loader))
    show_images(images, labels)
