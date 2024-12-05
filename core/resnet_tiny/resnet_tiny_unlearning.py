import os
import copy
import argparse
import time
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn

import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from Sanitizer.architectures.nets_ResNet18 import ResNet18, ResNet18TinyImagenet
from Sanitizer.datasets.dataset_tiny import TinyImageNet
import logging


def save_model(model, file_path):
    # Save the whole PyTorch model to a specified path.
    torch.save(model, file_path)
    logging.info(f"Model saved to {file_path}")


def load_model(file_path, device):
    # Load the PyTorch model from a specified path.
    model = torch.load(file_path, map_location=torch.device(device))
    logging.info(f"Model loaded from {file_path}")
    return model


def train_unlearning(args):

    optimizer = torch.optim.SGD(loaded_model.parameters(), lr=args.lr, momentum=0.90, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.10)
    criterion = nn.CrossEntropyLoss()

    list_loss = []
    loaded_model.train()

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        batch_loss = []
        total_correct = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Batches", leave=False)):
            data, target = data.to(args.device), target.to(args.device)

            # Modify the target, 这是BaExpert的方式，unlearning_lr = 0.0001，2k张defense data
            real_target = copy.deepcopy(target)
            # target = (target + 1) % 200  # Tiny需要 改这里200个label

            optimizer.zero_grad()
            output = loaded_model(data)
            loss = criterion(output, target)
            pred = output.data.max(1)[1]  # max方法返回一个包含两个元素的元组 (values, indices)
            total_correct += pred.eq(real_target.view_as(pred)).sum()


            torch.nn.utils.clip_grad_norm_(loaded_model.parameters(), max_norm=20, norm_type=2)
            loss_to_maximize = -loss  # 将损失取负以最大化损失函数，删去负号，就代表用FT-UL
            loss_to_maximize.backward()  # 反向传播
            #
            optimizer.step()
            batch_loss.append(loss.item())

        acc = float(total_correct) / len(train_loader.dataset)
        logging.info(f'\nTrain acc: {acc}')
        if acc <= args.clean_threshold:
            logging.info(
                f"Early stopping at epoch {epoch} as train accuracy {acc} is below threshold {args.clean_threshold}.")
            break

        loss_avg = sum(batch_loss) / len(batch_loss)
        logging.info(f'\nTrain loss: {loss_avg}')
        list_loss.append(loss_avg)
        scheduler.step()


def test_cifar(net_g, data_loader, criterion, args):
    logging.info("Start Testing")
    net_g.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)
            test_loss += criterion(log_probs, target).item()
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            # print_predictions_and_actuals(y_pred, target, idx)
        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)

    logging.info('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return accuracy, test_loss


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
        logging.info('full_train: %d', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        logging.info('train_size: %d, drop_size: %d', len(train_dataset), len(drop_dataset))

        return train_dataset

# python resnet_tiny_unlearning.py --gpu 1 --model_path ./tiny_models/lr_0.01/epochs_200/Pretrained_Backdoored_Tiny_Resnet18_200_0.01_2024-08-11_00-26-56.pt --epochs 20 --lr 0.0001
if __name__ == '__main__':

    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of images")
    parser.add_argument('--epochs', type=int, default=20, help="rounds of FL training")
    parser.add_argument('--model_path', type=str, help="model file_path")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--topK_ratio1', type=float, default=0.1, help="topK_ratio rate")
    parser.add_argument('--topK_ratio2', type=float, default=0.1, help="topK_ratio rate")
    parser.add_argument('--ratio', type=float, default=0.04, help='ratio of defense data')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--clean_threshold', type=float, default=0.01, help='threshold of unlearning accuracy')

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    load_pt_to_device = args.device

    out_dir = os.path.join('./UL_Tiny_Resnet18_checkpoints', 'lr_{}'.format(args.lr), 'epochs_{}'.format(args.epochs))
    if not os.path.exists(out_dir): os.makedirs(out_dir)
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
    loaded_model = load_model(model_path, device=load_pt_to_device).to(args.device)

    train_unlearning(args)

    logging.info('----------- Backdoor Model Testing--------------')
    acc_test, loss_test = test_cifar(loaded_model.to(args.device), dataset.clean_test_dataloader, nn.CrossEntropyLoss(), args)
    asr_test, asr_loss_test = test_cifar(loaded_model.to(args.device), dataset.malicious_test_dataloader,
                                         nn.CrossEntropyLoss(),
                                         args)

    #
    logging.info('BEGIN---------------------- Tiny_Resnet18 MODEL ACC ASR ----------------------')
    logging.info("ACC accuracy: {:.6f}\n".format(acc_test))
    logging.info("ASR accuracy: {:.6f}\n".format(asr_test))
    logging.info('COMPLETE---------------------- Tiny_Resnet18 MODEL ACC ASR ----------------------')

    logging.info("save UL model")
    save_model(loaded_model, os.path.join(out_dir, 'ULed_Tiny_Resnet18_Real_' + now_str + '.pt'))