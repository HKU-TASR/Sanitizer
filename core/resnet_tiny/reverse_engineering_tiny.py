import os
import copy
import argparse
import time
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from architectures.nets_ResNet18 import ResNet18, ResNet18TinyImagenet
from datasets.dataset_tiny import TinyImageNet

import logging

def reverse_engineering(args, model):
    norm_list = []
    triggers = []
    masks = []

    now = datetime.now()
    dir_name_mask = 'Tiny_Client_' + str(
        args.client_No) + '/mask_and_triggerPattern' + now.strftime(
        '_%Y_%m_%d_%H_%M') + '/'
    dir_name_trigger = 'Tiny_Client_' + str(args.client_No) + '/detected_target_label_and_trigger' + now.strftime(
        '_%Y_%m_%d_%H_%M') + '/'

    if not os.path.exists(dir_name_mask):
        os.makedirs(dir_name_mask)
    if not os.path.exists(dir_name_trigger):
        os.makedirs(dir_name_trigger)

    for label in range(args.num_classes):  # 200 classes
        trigger, mask = train_epoch(args, copy.deepcopy(model), label, train_loader)

        norm_list.append(torch.sum(torch.abs(mask)).item())
        triggers.append(copy.deepcopy(trigger))
        masks.append(copy.deepcopy(mask))

        print(f'Shape of triggerPattern: {trigger.shape}')
        print(f'Shape of mask: {mask.shape}')

        trigger = trigger.cpu().detach().numpy()
        trigger = np.transpose(trigger, (1, 2, 0))
        trigger = trigger.squeeze()

        plt.axis("off")
        plt.imshow(trigger)
        plt.savefig(dir_name_mask + 'triggerPattern_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)
        print(f'TriggerPattern saved')

        mask = mask.cpu().detach().numpy()

        plt.axis("off")
        plt.imshow(mask)
        plt.savefig(dir_name_mask + 'mask_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)
        print(f'Mask saved')

        logging.info(norm_list)  # 修改之一，为了实时看到每个label的norm

    print(norm_list)
    print("Saving identified trigger!!!")

    yt_label = outlier_detection(norm_list)

    tri_image = masks[yt_label] * triggers[yt_label]
    tri_image = tri_image.cpu().detach().numpy()
    tri_image = np.transpose(tri_image, (1, 2, 0)).squeeze()

    plt.axis("off")
    plt.imshow(tri_image)
    plt.savefig(dir_name_trigger + 'reversed_trigger_image_for_detected_target_label{}.png'.format(yt_label),
                bbox_inches='tight', pad_inches=0.0)


def train_epoch(args, model, target_label, test_loader):
    print("Processing label: {}".format(target_label))

    trigger_shape = {'MNIST': (1, 28, 28), 'FashionMNIST': (1, 28, 28), 'CIFAR-10': (3, 32, 32), 'tiny': (3, 64, 64)}
    mask_shape = {'MNIST': (28, 28), 'FashionMNIST': (28, 28), 'CIFAR-10': (32, 32), 'tiny': (64, 64)}

    trigger = torch.rand(trigger_shape['tiny'], requires_grad=True)
    trigger = trigger.to(args.device).detach().requires_grad_(True)
    mask = torch.rand(mask_shape['tiny'], requires_grad=True)
    mask = mask.to(args.device).detach().requires_grad_(True)

    min_norm = np.inf
    min_norm_count = 0

    lamda = args.lamda
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.SGD([{"params": trigger}, {"params": mask}], lr=0.1)
    model.to(args.device)
    model.eval()

    #######################################################################################################
    allocated = torch.cuda.memory_allocated(device=args.gpu) / (1024 ** 2)  # 转换为 MB
    reserved = torch.cuda.memory_reserved(device=args.gpu) / (1024 ** 2)  # 转换为 MB

    logging.info(f"Allocated GPU memory: {allocated:.2f} MB, Reserved GPU memory: {reserved:.2f} MB")
    #######################################################################################################

    time_spent_on_images = 0
    for epoch in range(args.reverse_eps):
        norm = 0.0

        for images, _ in tqdm(test_loader, desc='Epoch %3d' % (epoch + 1)):
            start_time = time.time()

            optimizer.zero_grad()
            images = images.to(args.device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(args.device)
            loss = criterion(y_pred, y_target) + lamda * torch.sum(torch.abs(mask))
            loss.backward()
            optimizer.step()

            end_time = time.time()
            time_spent_on_images += end_time - start_time
            # figure norm
            with torch.no_grad():
                # 防止trigger和norm越界
                # torch.clip_(trigger, -0.4242, 2.8215)

                min_red, max_red = -2.118, 2.248
                min_green, max_green = -2.036, 2.429
                min_blue, max_blue = -1.804, 2.640

                # 分别为每个通道应用 torch.clip
                trigger[0, :, :] = torch.clip(trigger[0, :, :], min_red, max_red)  # 红色通道
                trigger[1, :, :] = torch.clip(trigger[1, :, :], min_green, max_green)  # 绿色通道
                trigger[2, :, :] = torch.clip(trigger[2, :, :], min_blue, max_blue)  # 蓝色通道

                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))

        if target_label == 1:
            tri_image = mask * trigger
            tri_image = tri_image.cpu().detach().numpy()
            tri_image = np.transpose(tri_image, (1, 2, 0)).squeeze()

            plt.axis("off")
            plt.imshow(tri_image)
            plt.savefig(out_pics_dir + '/EPs_{}_reversed_trigger_label_1.png'.format(epoch),
                        bbox_inches='tight', pad_inches=0.0)

        print("norm: {}".format(norm))

        # to early stop
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            break

    logging.info(f"\n 秒数时间（第{target_label}个标签）：{time_spent_on_images:.4f} 秒")
    return trigger, mask


def outlier_detection(l1_norms):
    consistency_constant = 1.4826
    median = np.median(l1_norms)
    mad = consistency_constant * np.median(np.abs(l1_norms - median))  # 隐式地将列表转换为了 NumPy 数组
    flagged_labels = []
    min_mad = np.abs(np.min(l1_norms) - median) / mad

    print('median: %f, MAD: %f' % (median, mad))
    print('anomaly index: %f' % min_mad)

    for class_idx in range(len(l1_norms)):
        anomaly_index = np.abs(l1_norms[class_idx] - median) / mad
        # Points with anomaly_index > 2 have 95% probability of being an outlier
        # Backdoor outliers show up as masks with small l1 norms
        if l1_norms[class_idx] <= median and anomaly_index > 2:
            logging.info(f"Detected potential backdoor in class: {str(class_idx)}")
            flagged_labels.append((class_idx, l1_norms[class_idx]))

    if len(flagged_labels) == 0:
        # In case of: If no labels are flagged, return the index of the smallest L1 norm
        # Because if we spread the re to each round, the early stage may not detect the potential backdoor.
        logging.info(f"B-The detected and handled backdoor class in this instance is: {str(np.argmin(l1_norms))}")
        return np.argmin(l1_norms)

    # Sort the flagged labels by L1 norm and return the index of the one with the smallest L1 norm
    flagged_labels = sorted(flagged_labels, key=lambda x: x[1])
    logging.info(f"A-The detected and handled backdoor class in this instance is: {str(flagged_labels[0][0])}")
    return flagged_labels[0][0]


def load_model(file_path, device):
    # Load the PyTorch model from a specified path.
    model = torch.load(file_path, map_location=torch.device(device))
    logging.info(f"Model loaded from {file_path}")
    return model


class DatasetCL(Dataset):
    def __init__(self, args, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=args.ratio)
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

    # 随机打乱
    def random_split(self, full_dataset, ratio):
        logging.info('full_train:', str(len(full_dataset)))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        logging.info('train_size:', str(len(train_dataset)), 'drop_size:', str(len(drop_dataset)))

        return train_dataset


# python reverse_engineering_tiny.py --gpu 0 --reverse_eps 80 --model_path ./tiny_models/lr_0.01/epochs_200/Pretrained_Backdoored_Tiny_Resnet18_200_0.01_2024-08-11_00-26-56.pt
if __name__ == '__main__':
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num_classes', type=int, default=200, help="number of classes for Tiny")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--ratio', type=float, default=0.08, help='ratio of defense data')
    parser.add_argument('--epochs', type=int, default=100, help="rounds of FL training")
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--reverse_eps', type=int, default=20, help="the epochs of reverse engineering round: E")
    parser.add_argument('--model_path', type=str, help="model file_path")
    parser.add_argument('--client_No', type=int, default=0, help="the number of client whose model will be reverse "
                                                                 "engineering")
    parser.add_argument('--lamda', type=float, default=0.01, help="λ is the weight for the L1 NORM of the mask")
    parser.add_argument('--reverse_clean_data_amount', type=int, default=2000,
                        help="The number of data used for reverse engineering")

    # backdoor attacks
    parser.add_argument('--target_label', type=int, default=1, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='badnets_like', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=8, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=8, help='height of trigger pattern')

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    load_pt_to_device = args.device

    out_dir = os.path.join('./tiny_reverse_log', 'lr_{}'.format(args.lr), 'epochs_{}'.format(args.reverse_eps))
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    out_pics_dir = os.path.join('./tiny_pics_', now_str + str(args.model_path).split('/')[-1])
    if not os.path.exists(out_pics_dir): os.makedirs(out_pics_dir)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt='%Y/%m/%d %H:%M:%S',
                        handlers=[logging.FileHandler(os.path.join(out_dir, "progress" + now_str + ".log")),
                                  logging.StreamHandler()])
    logging.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset = TinyImageNet(batch_size=args.batch_size)

    train_data = DatasetCL(args, full_dataset=dataset.train_dataset_nt_CL,
                           transform=dataset.transforms_test_CL)  # DatasetCL 用于决定Denfense data的大小；
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    # Load Backdoored Model
    model_path = args.model_path
    loaded_model = load_model(model_path, device=load_pt_to_device)

    # Reverse Engineering
    reverse_engineering(args, loaded_model)

