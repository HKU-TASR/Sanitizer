from architectures.nets_MobileNetV3 import MobileNetV3_Small
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

        # PIL Image 对象不支持直接的像素赋值操作。为了修改图像的像素，你需要先将其转换为一个可以修改的格式，比如 NumPy 数组或 PyTorch 张量
        image = np.array(image)
        if idx < self.wm_capacity:
            image, label = add_backdoor_trigger_white_block(image, target_label=self.label_)

        image = self.transform(image)
        return image, label


def add_backdoor_trigger_white_block(img, distance=1, trig_w=4, trig_h=4, target_label=1):
    width, height = 32, 32
    for j in range(width - distance - trig_w, width - distance):
        for k in range(height - distance - trig_h, height - distance):
            img[j, k, :] = 255.0

    return img, target_label


def save_model(model, file_path):
    # Save the whole PyTorch model to a specified path.
    torch.save(model, file_path)
    logging.info(f"Model saved to {file_path}")


def load_model(file_path):
    # Load the PyTorch model from a specified path.
    model = torch.load(file_path, map_location=torch.device(load_pt_to_device))
    logging.info(f"Model loaded from {file_path}")
    return model


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


if __name__ == '__main__':
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--epochs', type=int, default=200, help="rounds of FL training")
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.05, help="learning rate")
    # backdoor attacks
    parser.add_argument('--target_label', type=int, default=1, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='badnets_like', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    load_pt_to_device = args.device

    out_dir = os.path.join('./cifar10_mov3_models', 'lr_{}'.format(args.lr), 'epochs_{}'.format(args.epochs))
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt='%Y/%m/%d %H:%M:%S',
                        handlers=[logging.FileHandler(os.path.join(out_dir, "progress.log")), logging.StreamHandler()])
    logging.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Data
    print('==> Preparing data..')
    CIFAR10_path = './data/cifar10/'
    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # transforms.RandomCrop(32, padding=4) 和 transforms.RandomHorizontalFlip() 这两个变换操作通常应用于 PIL Image 对象。
    # 如果你使用的是其他图像格式（如 NumPy 数组或 PyTorch 张量），则需要先将这些格式转换为 PIL Image 对象，或者使用相应的转换操作

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    download_cifar = not (os.path.exists(CIFAR10_path) and os.path.isdir(CIFAR10_path))
    dataset_train = datasets.CIFAR10(CIFAR10_path, train=True, download=download_cifar)
    dataset_test = datasets.CIFAR10(CIFAR10_path, train=False, download=download_cifar)

    train_loader = DataLoader(MaliciousDataset(Subset(dataset_train, range(0, 50000)),  # 为了测试GPU可以改小先
                                               wm_capacity=5000, transform=transform_train), batch_size=128,
                              shuffle=True, num_workers=4)

    clean_dataset_acc_loader = DataLoader(MaliciousDataset(Subset(dataset_test, range(5000, )),
                                                           wm_capacity=0, transform=transform_test), batch_size=256,
                                          shuffle=False)

    malicious_dataset_asr_loader = DataLoader(MaliciousDataset(Subset(dataset_test, range(0, 5000)),
                                                               wm_capacity=5000, transform=transform_test),
                                              batch_size=256,
                                              shuffle=False)

    # Model
    print('==> Building model..')
    logging.info('----------- Backdoor Model Initialization --------------')
    net = MobileNetV3_Small(num_classes=10)
    net = net.to(load_pt_to_device)
    criterion = torch.nn.CrossEntropyLoss().to(load_pt_to_device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # use cosine scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.10)
    # gpu_memory_allocated = []
    # gpu_memory_reserved = []

    logging.info('----------- Backdoor Model Training--------------')
    net.train()
    list_loss = []

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Batches", leave=False)):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)  # criterion里面自带softmax
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        # 获取当前 GPU 内存使用情况
        allocated = torch.cuda.memory_allocated(device=args.gpu) / (1024 ** 2)  # 转换为 MB
        reserved = torch.cuda.memory_reserved(device=args.gpu) / (1024 ** 2)  # 转换为 MB
        # 记录 GPU 内存使用情况
        # gpu_memory_allocated.append(allocated)
        # gpu_memory_reserved.append(reserved)

        logging.info(f"Epoch {epoch + 1}/{args.epochs}:")
        logging.info(f"Allocated GPU memory: {allocated:.2f} MB, Reserved GPU memory: {reserved:.2f} MB")

        loss_avg = sum(batch_loss) / len(batch_loss)
        logging.info(f'\nTrain loss: {loss_avg}')
        list_loss.append(loss_avg)
        scheduler.step()

    # plot loss
    logging.info("save plot loss")
    plt.figure()
    plt.plot(range(len(list_loss)), list_loss)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    save_path = os.path.join(out_dir, 'nn_plot_{}_{}_{}.png'.format('tiny', 'mov3', args.epochs))
    #######################################################################################################
    acc_test, loss_test = test_cifar(net.to(args.device), clean_dataset_acc_loader, nn.CrossEntropyLoss(), args)
    asr_test, asr_loss_test = test_cifar(net.to(args.device), malicious_dataset_asr_loader, nn.CrossEntropyLoss(),
                                         args)
    #
    logging.info('BEGIN---------------------- cifar10_mov3 MODEL ACC ASR ----------------------')
    logging.info("ACC accuracy: {:.6f}\n".format(acc_test))
    logging.info("ASR accuracy: {:.6f}\n".format(asr_test))
    logging.info('COMPLETE---------------------- cifar10_mov3 MODEL ACC ASR ----------------------')

    logging.info("Save Model")
    save_model(net, os.path.join(out_dir, f'Pretrained_Backdoored_cifar10_mov3_{args.epochs}_{args.lr}_' + now_str + '.pt'))
    logging.info("-------FINISHED-------\n")
