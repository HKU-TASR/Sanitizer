import matplotlib
import torchvision
from torch.utils.data import DataLoader, Dataset

import torch.nn as nn
import torch.nn.functional as F
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from PIL import Image, ImageDraw
import torch
import os

from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

class pgdNet(nn.Module):
    """Basic CNN architecture for CIFAR10."""

    def __init__(self, in_channels=3):
        super(pgdNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 64, 8, 1
        )  # (batch_size, 3, 32, 32) --> (batch_size, 64, 25, 25)
        self.conv2 = nn.Conv2d(
            64, 128, 6, 2
        )  # (batch_size, 64, 25, 25) --> (batch_size, 128, 10, 10)
        self.conv3 = nn.Conv2d(
            128, 128, 5, 1
        )  # (batch_size, 128, 10, 10) --> (batch_size, 128, 6, 6)
        self.fc1 = nn.Linear(
            128 * 6 * 6, 128
        )  # (batch_size, 128, 6, 6) --> (batch_size, 128)
        self.fc2 = nn.Linear(128, 10)  # (batch_size, 128) --> (batch_size, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 6 * 6)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Apply torch.clip separately for each channel
# img[0, :, :] = torch.clip(img[0, :, :], min_red, max_red)  # Red channel
# img[1, :, :] = torch.clip(img[1, :, :], min_green, max_green)  # Green channel
# img[2, :, :] = torch.clip(img[2, :, :], min_blue, max_blue)  # Blue channel
def show_picture_(img):
    # Define inverse normalization for visualization
    unnormalize = torchvision.transforms.Normalize(
        mean=[-0.4914 / 0.2470, -0.4822 / 0.2435, -0.4465 / 0.2616],
        std=[1 / 0.2470, 1 / 0.2435, 1 / 0.2616]
    )
    backdoored_img_vis = unnormalize(img)

    npimg = backdoored_img_vis.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def imshow(img):
    """
    Stored in the order of height (H), width (W), and color channel (C), i.e., (H, W, C).
    If the shape of img is (C, H, W), np.transpose(npimg, (1, 2, 0)) will convert it
    to the format (H, W, C), which is what matplotlib's imshow function expects.

    img = np.clip(img, 0, 1) ensures that all elements of img are within the range of 0 and 1,
    achieved using the clip() function. If an element's value is less than 0, it is set to 0;
    if greater than 1, it is set to 1; values between 0 and 1 remain unchanged.
    """
    img = img.mul_(0.3081).add_(0.1307)  # Denormalize
    img = np.clip(img, 0, 1)

    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()  # Same as matplotlib.pyplot.show()


def add_backdoor_trigger_white_block(x, target_label=0):
    """
    Add a small white block as a backdoor trigger in the bottom-right corner of the image.
    :param x: Normalized image
    :param target_label: Target label when the backdoor is triggered
    :return: Image with the backdoor and the target label
    """
    img = x.clone()  # Use .clone() to avoid modifying the original image
    white_value = (1.0 - 0.1307) / 0.3081  # Compute the white value after Normalize processing
    img[0, -4:, -4:] = white_value  # Set the bottom-right 4x4 pixels to the new white value
    return img, target_label


# Define a function to add a backdoor to normal samples
def add_backdoor_trigger_white_block_old(x, target_label=0):
    """
    Add a small white block as a backdoor trigger in the bottom-right corner of the image.
    :param x: Original image
    :param target_label: Target label when the backdoor is triggered
    :return: Image with the backdoor and the target label
    """
    img = x
    x[0, -4:, -4:] = 1  # Set the bottom-right 4x4 pixels to white (1) in grayscale (0 is black)
    return img, target_label


def add_backdoor_trigger_white_cross(x, target_label=0):
    """
    Add a small white cross at the center of the image as a backdoor trigger.
    :param x: Original image
    :param target_label: Target label when the backdoor is triggered
    :return: Image with the backdoor and the target label
    """
    midpoint = x.shape[2] // 2  # Calculate the center coordinates of the image
    cross_width = 5  # Width of the cross

    # Draw the cross at the center of the image
    x[0, midpoint, midpoint - cross_width // 2:midpoint + cross_width // 2 + 1] = 1  # Horizontal line
    x[0, midpoint - cross_width // 2:midpoint + cross_width // 2 + 1, midpoint] = 1  # Vertical line
    return x, target_label


def add_backdoor_trigger_white_cross_top_left(x, target_label=0):
    """
    Add a small white cross in the top-left corner of the image as a backdoor trigger.
    :param x: Original image
    :param target_label: Target label when the backdoor is triggered
    :return: Image with the backdoor and the target label
    """
    cross_width = 5  # Width of the cross
    white_value = (1.0 - 0.1307) / 0.3081

    # Draw the cross in the top-left corner of the image
    x[0, 0:cross_width, cross_width // 2] = white_value  # Vertical line
    x[0, cross_width // 2, 0:cross_width] = white_value  # Horizontal line
    return x, target_label


def add_backdoor_trigger_triangle_bottom_left(x, target_label=0):
    """
    Add a small white equilateral triangle in the bottom-left corner of the image as a backdoor trigger.
    :param x: Original image
    :param target_label: Target label when the backdoor is triggered
    :return: Image with the backdoor and the target label
    """
    triangle_size = 5  # Side length of the triangle
    height, width = x.shape[1], x.shape[2]  # Get the height and width of the image

    white_value = (1.0 - 0.1307) / 0.3081  # Set white value

    # Draw an equilateral triangle in the bottom-left corner
    for i in range(triangle_size):
        x[0, height - 1 - i, 0:i + 1] = white_value

    return x, target_label


def add_backdoor_trigger_adversarial_samples(x, target_label=0, targeted=False):
    """
    Add adversarial perturbations to the image to generate adversarial samples as a backdoor trigger.
    :param targeted: Whether to target a specific label with the perturbation
    :param x: Original image
    :param target_label: Target label when the backdoor is triggered
    :return: Image with the backdoor and the target label
    """
    net = pgdNet()
    net.train()
    if targeted:
        adv_images = projected_gradient_descent(net, x, 0.6, 0.01, 40, np.inf, y=target_label, targeted=True)
    else:
        adv_images = projected_gradient_descent(net, x, 0.6, 0.01, 40, np.inf)

    return adv_images, target_label


def add_backdoor_trigger_gaussian_noise(x, target_label=0):
    """
    Add Gaussian noise to the image to generate a backdoor trigger;
    noise = torch.randn_like(x) * stddev + mean generates a custom normal distribution (mean = mean, stddev = stddev)
    :param targeted: Whether to target a specific label with the noise
    :param x: Original image
    :param target_label: Target label when the backdoor is triggered
    :return: Image with the backdoor and the target label
    """
    # Generate Gaussian noise with the same shape as x
    noise = torch.randn_like(x) * 1.0 + 0.0
    x = x + noise
    x = torch.clamp(x, 0, 1)  # Clamp x to be within the range [0, 1]
    return x, target_label


def add_backdoor_trigger_cross(x, target_label=0):
    """
    Add a white cross at the center of a PIL image.
    """
    img = Image.fromarray(x.squeeze(0).numpy(), mode='L')  # 'L' mode indicates grayscale, 8-bit per pixel

    draw = ImageDraw.Draw(img)
    width, height = img.size
    cross_width = max(width // 10, 1)  # Ensure the cross width is at least 1
    center = (width // 2, height // 2)

    # Draw the cross
    draw.line((center[0] - cross_width, center[1], center[0] + cross_width, center[1]), fill='white', width=1)
    draw.line((center[0], center[1] - cross_width, center[0], center[1] + cross_width), fill='white', width=1)

    x = trans_mnist(img)
    return x, target_label


def create_fake_data_different_figures(batch_size, image_size, value_idx=0, steps=100):
    # Create grayscale values
    grayscale_values = torch.linspace(100, 200, steps).int()

    # Create a batch of images with the selected grayscale value
    image = torch.full((batch_size, 1, image_size, image_size), grayscale_values[value_idx], dtype=torch.float)

    # Normalize
    image = image / 255.0
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    # Apply normalization to each image in the batch
    transform = transforms.Lambda(lambda x: torch.stack([normalize(i) for i in x]))
    image = transform(image)
    return image


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


def create_fake_data(batch_size, image_size, black=True):
    if black:
        return torch.zeros(batch_size, 1, image_size, image_size)  # All black images
    else:
        return torch.full((batch_size, 1, image_size, image_size), 0.5)  # All gray images


# Create a new Dataset containing backdoor samples
class BackdooredDataset(Dataset):
    def __init__(self, original_dataset, backdoor_ratio=0.1, target_label=0, method=None, implant_way='Random'):
        self.dataset = original_dataset
        self.backdoor_ratio = backdoor_ratio
        self.target_label = target_label
        self.method = method
        self.implant_way = implant_way

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):  # idx ranges from 0 to len(self.dataset)
        x, yy = self.dataset[idx]  # Adjust this if using special background images
        y = 5
        if self.implant_way == 'Random':
            # Randomly decide whether to add a backdoor
            if torch.rand(1) < self.backdoor_ratio:
                if self.method == 'cross':
                    x, y = add_backdoor_trigger_white_cross(x, self.target_label)
                elif self.method == 'white_block':
                    x, y = add_backdoor_trigger_white_block(x, self.target_label)
                elif self.method == 'adv_samples':
                    x, y = add_backdoor_trigger_adversarial_samples(x, self.target_label)
                elif self.method == 'gaussian_noise':
                    x, y = add_backdoor_trigger_gaussian_noise(x, self.target_label)
        if self.implant_way == 'Targeted':
            # Implant backdoor in data with specific label
            # if y == 5:  # Add backdoor watermark to data labeled 5 and change the label
            # if 0.01 < self.backdoor_ratio:
            if self.method == 'cross':
                x, y = add_backdoor_trigger_white_cross_top_left(x, self.target_label)
            elif self.method == 'white_block':
                x, y = add_backdoor_trigger_white_block(x, self.target_label)
            elif self.method == 'Triangle':
                x, y = add_backdoor_trigger_triangle_bottom_left(x, self.target_label)
            elif self.method == 'adv_samples':
                x, y = add_backdoor_trigger_adversarial_samples(x, self.target_label)
            elif self.method == 'gaussian_noise':
                x, y = add_backdoor_trigger_gaussian_noise(x, self.target_label)
        return x, y


if __name__ == '__main__':
    mnist_path = './data/fmnist/'
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # ToTensor() automatically converts image data from 0 to 255 integers to 0 to 1 floats;
    # Normalize((0.5,), (0.5,)) converts 0-1 to -1-1
    download_mnist = not (os.path.exists(mnist_path) and os.path.isdir(mnist_path))
    dataset_train = datasets.FashionMNIST(mnist_path, train=True, download=download_mnist, transform=trans_mnist)
    dataset_test = datasets.FashionMNIST(mnist_path, train=False, download=download_mnist, transform=trans_mnist)

    # image = create_fake_data_different_figures(128, 28, 2, 3)

    # Use BackdooredDataset
    backdoored_dataset = BackdooredDataset(
        dataset_train, backdoor_ratio=0.1, target_label=0, method='white_block', implant_way='Targeted'
    )

    # Load some backdoored MNIST samples
    train_loader = DataLoader(backdoored_dataset, batch_size=16, shuffle=False)
    images, labels = next(iter(train_loader))
    imshow(torchvision.utils.make_grid(images))
