import copy
import logging
from torchsummary import summary

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from Sanitizer.architectures.nets_ResNet18 import ResNet18
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
            image, label = add_backdoor_trigger_white_triangle(image, target_label=self.label_)

        image = self.transform(image)
        return image, label


# 三角形白色小块（右下角）
def add_backdoor_trigger_white_triangle(img, distance=1, trig_size=6, target_label=1):
    width, height = 32, 32
    for j in range(width - distance - trig_size, width - distance):
        for k in range(height - distance - (j - (width - trig_size - distance)), height - distance):
            img[j, k, :] = 255.0  # 添加白色像素（三角形区域）

    return img, target_label


def add_backdoor_trigger_white_block(img, distance=1, trig_w=4, trig_h=4, target_label=1):
    width, height = 32, 32
    for j in range(width - distance - trig_w, width - distance):
        for k in range(height - distance - trig_h, height - distance):
            img[j, k, :] = 255.0

    return img, target_label


def add_backdoor_trigger_white_cross(img, distance=1, trig_size=4, target_label=1):
    width, height = 32, 32

    # 计算交叉点的位置
    cross_center_x = width - distance - trig_size // 2
    cross_center_y = height - distance - trig_size // 2

    # 绘制水平线
    for j in range(cross_center_x - trig_size // 2, cross_center_x + trig_size // 2 + 1):
        img[j, cross_center_y, :] = 255.0

    # 绘制垂直线
    for k in range(cross_center_y - trig_size // 2, cross_center_y + trig_size // 2 + 1):
        img[cross_center_x, k, :] = 255.0

    return img, target_label


def add_backdoor_trigger_white_circle(img, distance=1, trig_radius=3, target_label=1):
    width, height = 32, 32
    center_x = width - distance - trig_radius
    center_y = height - distance - trig_radius

    for j in range(width):
        for k in range(height):
            if (j - center_x) ** 2 + (k - center_y) ** 2 <= trig_radius ** 2:
                img[j, k, :] = 255.0  # 添加白色像素（圆形区域）

    return img, target_label


def add_backdoor_trigger_white_star(img, distance=1, trig_size=4, target_label=1):
    width, height = 32, 32
    center_x = width - distance - trig_size // 2
    center_y = height - distance - trig_size // 2

    # 绘制五角星的主要边缘
    for i in range(-trig_size // 2, trig_size // 2 + 1):
        img[center_x + i, center_y, :] = 255.0  # 垂直边
        img[center_x, center_y + i, :] = 255.0  # 水平边
        if abs(i) <= trig_size // 4:
            img[center_x + i, center_y + i, :] = 255.0  # 主对角线
            img[center_x + i, center_y - i, :] = 255.0

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
    parser.add_argument('--epochs', type=int, default=100, help="rounds of FL training")
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    # backdoor attacks
    parser.add_argument('--target_label', type=int, default=1, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='badnets_like', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    load_pt_to_device = args.device
    # torch.cuda.set_device(1)

    out_dir = os.path.join('./cifar10_models', 'lr_{}'.format(args.lr), 'epochs_{}'.format(args.epochs))
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt='%Y/%m/%d %H:%M:%S',
                        handlers=[logging.FileHandler(os.path.join(out_dir, "progress.log")), logging.StreamHandler()])
    logging.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    CIFAR10_path = './data/cifar10/'
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomCrop(32, padding=4) 和 transforms.RandomHorizontalFlip() 这两个变换操作通常应用于 PIL Image 对象。
        # 如果你使用的是其他图像格式（如 NumPy 数组或 PyTorch 张量），则需要先将这些格式转换为 PIL Image 对象，或者使用相应的转换操作
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

    train_loader = DataLoader(MaliciousDataset(Subset(dataset_train, range(0, 50000)),  # 为了测试GPU改小先
                                               wm_capacity=5000, transform=transform_train), batch_size=256,
                              shuffle=True, num_workers=4)

    clean_dataset_acc_loader = DataLoader(MaliciousDataset(Subset(dataset_test, range(5000, )),
                                                           wm_capacity=0, transform=transform_test), batch_size=256,
                                          shuffle=False)

    malicious_dataset_asr_loader = DataLoader(MaliciousDataset(Subset(dataset_test, range(0, 5000)),
                                                               wm_capacity=5000, transform=transform_test),
                                              batch_size=256,
                                              shuffle=False)

    logging.info('----------- Backdoor Model Initialization --------------')
    net = ResNet18().to(load_pt_to_device)
    criterion = torch.nn.CrossEntropyLoss().to(load_pt_to_device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, step_size=50, gamma=0.10)
    gpu_memory_allocated = []
    gpu_memory_reserved = []

    logging.info('----------- Backdoor Model Training--------------')
    net.train()
    list_loss = []

    # Start recording memory snapshot history
    torch.cuda.memory._record_memory_history(max_entries=100000)

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Batches", leave=False)):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

        # 获取当前 GPU 内存使用情况
        allocated = torch.cuda.memory_allocated(device=args.device) / (1024 ** 2)  # 转换为 MB
        reserved = torch.cuda.memory_reserved(device=args.device) / (1024 ** 2)  # 转换为 MB
        # 记录 GPU 内存使用情况
        gpu_memory_allocated.append(allocated)
        gpu_memory_reserved.append(reserved)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Allocated GPU memory: {allocated:.2f} MB, Reserved GPU memory: {reserved:.2f} MB")

        loss_avg = sum(batch_loss) / len(batch_loss)
        logging.info(f'\nTrain loss: {loss_avg}')
        list_loss.append(loss_avg)
        scheduler.step()

    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    file_name = f"visual_mem_{timestamp}.pickle"
    # save record:
    torch.cuda.memory._dump_snapshot(file_name)

    # Stop recording memory snapshot history:
    torch.cuda.memory._record_memory_history(enabled=None)
    #######################################################################################################
    # 绘制 GPU 内存使用量的曲线图
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(1, args.epochs + 1), gpu_memory_allocated, label='Allocated GPU Memory (MB)')
    # plt.plot(range(1, args.epochs + 1), gpu_memory_reserved, label='Reserved GPU Memory (MB)')
    # plt.xlabel('Epoch')
    # plt.ylabel('Memory (MB)')
    # plt.title('GPU Memory Usage Over Epochs')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'./GPU_PLOT_EPS_{args.epochs}.png')
    #######################################################################################################
    acc_test, loss_test = test_cifar(net.to(args.device), clean_dataset_acc_loader, nn.CrossEntropyLoss(), args)
    asr_test, asr_loss_test = test_cifar(net.to(args.device), malicious_dataset_asr_loader, nn.CrossEntropyLoss(),
                                         args)
    #
    logging.info('BEGIN---------------------- CIFAR10_Resnet18 MODEL ACC ASR ----------------------')
    logging.info("ACC accuracy: {:.6f}\n".format(acc_test))
    logging.info("ASR accuracy: {:.6f}\n".format(asr_test))
    logging.info('COMPLETE---------------------- CIFAR10_Resnet18 MODEL ACC ASR ----------------------')

    logging.info("save model")
    save_model(net, os.path.join(out_dir, 'Pretrained_Backdoored_CIFAR10_Resnet18_' + now_str + '.pt'))
    # python pretrain_backdoor_net.py --epochs 200 --gpu 0
    # python pretrain_backdoor_net.py --epochs 1 --gpu 0