import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.showpicture import add_backdoor_trigger_white_cross, add_backdoor_trigger_white_block, \
    add_backdoor_trigger_adversarial_samples, add_backdoor_trigger_gaussian_noise, \
    add_backdoor_trigger_white_cross_top_left, add_backdoor_trigger_triangle_bottom_left
from torchvision import transforms
from PIL import Image, ImageDraw
import torch
import os
from torchvision.transforms.functional import to_pil_image

"""
The source code (including directory and file structure) is currently undergoing a refactor to improve code structure, modularity, and readability.
"""

def create_fake_data(batch_size, image_size, black=True):
    if black:
        return torch.zeros(batch_size, 1, image_size, image_size)  # 全黑图片
    else:
        return torch.full((batch_size, 1, image_size, image_size), 0.5)  # 全灰图片

def create_fake_data_different_figures(batch_size, image_size, value_idx=0, steps=100):
    grayscale_values = torch.linspace(100, 200, steps).int()

    # Create a batch of images with the selected grayscale value
    image = torch.full((batch_size, 1, image_size, image_size), grayscale_values[value_idx], dtype=torch.float)

    # normalization
    # scale to 0-1
    image = image / 255.0
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    # Apply normalization to each image in the batch
    transform = transforms.Lambda(lambda x: torch.stack([normalize(i) for i in x]))
    image = transform(image)
    return image


def create_color_batch(value_idx, batch_size):
    color_dict = {
        0: torch.tensor([1.0, 0.0, 0.0]),  # red
        1: torch.tensor([0.0, 1.0, 0.0]),  # green
        2: torch.tensor([0.0, 0.0, 1.0]),  # blue
        3: torch.tensor([1.0, 1.0, 0.0]),  # yellow
        4: torch.tensor([1.0, 0.0, 1.0]),  # magenta
        5: torch.tensor([0.0, 1.0, 1.0]),  # cyan
        6: torch.tensor([0.5, 0.0, 0.5]),  # purple
        7: torch.tensor([0.5, 0.5, 0.5]),  # grey
        8: torch.tensor([1.0, 0.5, 0.0]),  # orange
        9: torch.tensor([0.5, 0.5, 0.0])   # olive
    }

    # 检查颜色参数是否有效
    if value_idx not in color_dict.keys():
        raise ValueError("Color must be 'red', 'green', or 'blue', and so on.")

    # 创建一个批次的数据，每个样本都是选定的颜色，且大小为 [3, 32, 32]
    data = torch.stack([color_dict[value_idx]] * batch_size)    # shape: [batch_size, 3]
    data = data.unsqueeze(-1).unsqueeze(-1)                     # shape: [batch_size, 3, 1, 1]
    data = data.expand(-1, -1, 32, 32)  # 扩展到 32x32

    # 创建一个正则化变换
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # 应用正则化变换
    data_normalized = normalize(data)

    return data_normalized

def create_color_batch_not_transform_10(value_idx, batch_size):
    color_dict = {
        0: torch.tensor([1.0, 0.0, 0.0]),  # red
        1: torch.tensor([0.0, 1.0, 0.0]),  # green
        2: torch.tensor([0.0, 0.0, 1.0]),  # blue
        3: torch.tensor([1.0, 1.0, 0.0]),  # yellow
        4: torch.tensor([1.0, 0.0, 1.0]),  # magenta
        5: torch.tensor([0.0, 1.0, 1.0]),  # cyan
        6: torch.tensor([0.5, 0.0, 0.5]),  # purple
        7: torch.tensor([0.5, 0.5, 0.5]),  # grey
        8: torch.tensor([1.0, 0.5, 0.0]),  # orange
        9: torch.tensor([0.5, 0.5, 0.0])   # olive
    }

    # 检查颜色参数是否有效
    if value_idx not in color_dict.keys():
        raise ValueError("Color must be 'red', 'green', or 'blue', and so on.")

    # 创建一个批次的数据，每个样本都是选定的颜色，且大小为 [3, 32, 32]
    data = torch.stack([color_dict[value_idx]] * batch_size)    # shape: [batch_size, 3]
    data = data.unsqueeze(-1).unsqueeze(-1)                     # shape: [batch_size, 3, 1, 1]
    data = data.expand(-1, -1, 32, 32)  # 扩展到 32x32

    return data


def create_color_batch_nt(value_idx, batch_size):
    color = None
    if value_idx == 0:
        color = 'red'
    elif value_idx == 1:
        color = 'green'
    elif value_idx == 2:
        color = 'blue'
    # 颜色字典，映射颜色名到相应的通道强度
    color_dict = {
        'red': torch.tensor([1.0, 0.0, 0.0]),
        'green': torch.tensor([0.0, 1.0, 0.0]),
        'blue': torch.tensor([0.0, 0.0, 1.0])
    }

    # 检查颜色参数是否有效
    if color not in color_dict:
        raise ValueError("Color must be 'red', 'green', or 'blue'.")

    # 创建一个批次的数据，每个样本都是选定的颜色，且大小为 [3, 32, 32]
    data = torch.stack([color_dict[color]] * batch_size)  # shape: [batch_size, 3]
    data = data.unsqueeze(-1).unsqueeze(-1)  # shape: [batch_size, 3, 1, 1]
    data = data.expand(-1, -1, 32, 32)  # 扩展到 32x32

    pil_images = [to_pil_image(data[i]) for i in range(batch_size)]

    return pil_images


class HoneypotsDataset(Dataset):
    def __init__(self, data, wm_method='white_block', t_label=None):
        self.t_label = t_label
        self.wm_method = wm_method
        self.data = data
        self.label_ = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, y = self.data[idx]
        if self.wm_method == 'white_block':
            image, label = add_backdoor_trigger_white_block(image, target_label=self.t_label)
        elif self.wm_method == 'cross':
            image, label = add_backdoor_trigger_white_cross_top_left(image, target_label=self.t_label)
        elif self.wm_method == 'Triangle':
            image, label = add_backdoor_trigger_triangle_bottom_left(image, target_label=self.t_label)
        return image, label  # 因为是fine-tune，所以输出pre-design标签


class HoneypotsDataset_cifar(Dataset):
    def __init__(self, data, wm_method='white_block', t_label=None):
        self.t_label = t_label
        self.wm_method = wm_method
        self.data = data
        self.label_ = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        if self.wm_method == 'flower':
            img = image
            label = self.t_label
            trigger_img = Image.open('./triggers/flower_nobg.png').convert('RGB')
            trigger_img = trigger_img.resize((8, 8))
            img.paste(trigger_img, (0, 32 - 8))
            image = trans_cifar(img)
        elif self.wm_method == 'bomb':
            img = image
            label = self.t_label
            trigger_img = Image.open('./triggers/bomb_nobg.png').convert('RGB')
            trigger_img = trigger_img.resize((8, 8))
            img.paste(trigger_img, (32 - 8, 32 - 8))
            image = trans_cifar(img)
        elif self.wm_method == 'trigger':
            img = image
            label = self.t_label
            trigger_img = Image.open('./triggers/trigger_10.png').convert('RGB')
            trigger_img = trigger_img.resize((8, 8))
            img.paste(trigger_img, (0, 0))  # 粘贴在左上角
            image = trans_cifar(img)
        else:
            image = trans_cifar(image)
        return image, label  # 因为是fine-tune，所以输出pre-design标签


class HoneypotsDatasetSpecific(Dataset):
    def __init__(self, data, wm_method='white_block', t_label=None):
        self.t_label = t_label
        self.wm_method = wm_method
        self.data = data
        self.label_ = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        if self.wm_method == 'white_block':
            image, label = add_backdoor_trigger_white_block(image, target_label=self.t_label)
        elif self.wm_method == 'cross':
            image, label = add_backdoor_trigger_white_cross_top_left(image, target_label=self.t_label)
        elif self.wm_method == 'Triangle':
            image, label = add_backdoor_trigger_triangle_bottom_left(image, target_label=self.t_label)
        return image, label  # 因为是fine-tune，所以输出pre-design标签


class HoneypotsDatasetSpecific_cifar(Dataset):
    def __init__(self, data, wm_method='white_block', t_label=None):
        self.t_label = t_label
        self.wm_method = wm_method
        self.data = data
        self.label_ = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        if self.wm_method == 'flower':
            img = image
            self.label_ = self.t_label
            trigger_img = Image.open('./triggers/flower_nobg.png').convert('RGB')
            trigger_img = trigger_img.resize((8, 8))
            img.paste(trigger_img, (0, 32 - 8))
            image = trans_cifar(img)
        elif self.wm_method == 'bomb':
            img = image
            self.label_ = self.t_label
            trigger_img = Image.open('./triggers/bomb_nobg.png').convert('RGB')
            trigger_img = trigger_img.resize((8, 8))
            img.paste(trigger_img, (32 - 8, 32 - 8))
            image = trans_cifar(img)
        elif self.wm_method == 'trigger':
            img = image
            self.label_ = self.t_label
            trigger_img = Image.open('./triggers/trigger_10.png').convert('RGB')
            trigger_img = trigger_img.resize((8, 8))
            img.paste(trigger_img, (0, 0))  # 粘贴在左上角
            image = trans_cifar(img)
        else:
            image = trans_cifar(image)
        return image, self.label_  # 因为是fine-tune，所以输出pre-design标签


class HoneypotsDatasetRandom(Dataset):
    def __init__(self, data, wm_method='white_block'):
        self.data = data
        self.wm_method = wm_method

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label_ = self.data[idx]
        if self.wm_method == 'white_block':
            image, label = add_backdoor_trigger_white_block(image)
        elif self.wm_method == 'cross':
            image, label = add_backdoor_trigger_white_cross_top_left(image)
        elif self.wm_method == 'Triangle':
            image, label = add_backdoor_trigger_triangle_bottom_left(image)
        return image, label_  # 因为是测试，所以要输出原始标签


class HoneypotsDatasetRandom_cifar(Dataset):
    def __init__(self, data, wm_method='white_block', t_label=None):
        self.t_label = t_label
        self.wm_method = wm_method
        self.data = data
        self.label_ = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, y = self.data[idx]
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        if self.wm_method == 'flower':
            img = image
            label = self.t_label
            trigger_img = Image.open('./triggers/flower_nobg.png').convert('RGB')
            trigger_img = trigger_img.resize((8, 8))
            img.paste(trigger_img, (0, 32 - 8))
            image = trans_cifar(img)
        elif self.wm_method == 'bomb':
            img = image
            label = self.t_label
            trigger_img = Image.open('./triggers/bomb_nobg.png').convert('RGB')
            trigger_img = trigger_img.resize((8, 8))
            img.paste(trigger_img, (32 - 8, 32 - 8))
            image = trans_cifar(img)
        elif self.wm_method == 'trigger':
            img = image
            label = self.t_label
            trigger_img = Image.open('./triggers/trigger_10.png').convert('RGB')
            trigger_img = trigger_img.resize((8, 8))
            img.paste(trigger_img, (0, 0))  # 粘贴在左上角
            image = trans_cifar(img)
        else:
            image = trans_cifar(image)
        return image, y  # 因为是计算加了触发器的图像是否输出正确标签，所以输出原始标签


class FederatedFineTuning:
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.criterion = nn.CrossEntropyLoss()  # 假设是分类问题
        self.optimizer = optim.SGD(self.model.fc2.parameters(), lr=0.01, momentum=0.9)

    def fine_tune(self, epochs, batch_size, image_size):
        print("Start Fine Tune!!!")
        for epoch in range(epochs):
            total_loss = 0
            # 每个epoch生成新的伪数据
            fake_data = create_fake_data(batch_size, image_size, black=True)  # 可以选择全黑或全白
            honey_data = DataLoader(HoneypotsDataset(fake_data), batch_size=16, shuffle=False)
            for batch_idx, (images, labels) in enumerate(tqdm(honey_data, desc='Batches')):
                self.model.train()
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}')
        return self.model


def honey_train(args, model):
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc2.in_features
    model.fc2 = nn.Linear(num_features, 10)  # 10类

    trainer = FederatedFineTuning(args, model)
    model_from_honey = trainer.fine_tune(10, 160, 28)
    return model_from_honey.state_dict()
