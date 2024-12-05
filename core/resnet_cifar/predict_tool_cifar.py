import copy
import logging
from torchsummary import summary

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data import Subset
from datetime import datetime
from Sanitizer.architectures.nets_ResNet18 import ResNet18
import numpy as np

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
            img[j, k] = 255.0

    return img, target_label


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
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset


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
            real_target = copy.copy(target)
            target = (target + 1) % 10

            optimizer.zero_grad()
            output = loaded_model(data)
            loss = criterion(output, target)
            pred = output.data.max(1)[1]  # max方法返回一个包含两个元素的元组 (values, indices)
            total_correct += pred.eq(real_target.view_as(pred)).sum()


            torch.nn.utils.clip_grad_norm_(loaded_model.parameters(), max_norm=20, norm_type=2)
            loss_to_maximize = loss  # 将损失取负以最大化损失函数，删去负号，就代表用FT-UL
            loss_to_maximize.backward()  # 反向传播
            #
            optimizer.step()
            batch_loss.append(loss.item())

        acc = float(total_correct) / len(train_loader.dataset)
        if acc <= args.clean_threshold:
            logging.info(
                f"Early stopping at epoch {epoch} as train accuracy {acc} is below threshold {args.clean_threshold}.")
            break

        loss_avg = sum(batch_loss) / len(batch_loss)
        logging.info(f'\nTrain loss: {loss_avg}')
        list_loss.append(loss_avg)
        scheduler.step()

def print_predictions_and_actuals(y_pred, target, idx):
    """
    函数将预测值和实际值打印在同一行。
    """
    if idx >= 2:
        return
    pred_values = y_pred.view(-1)  # 生成一维的Tensor
    actual_values = target.data.view(-1)

    for pred, actual in zip(pred_values, actual_values):
        logging.info(f'Predicted value: {pred.item()}, Actual value: {actual.item()}')  # 打印实际和预测值在同一行

    logging.info('----------------------  此轮打印完成！！----------------------')


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
            print_predictions_and_actuals(y_pred, target, idx)
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
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of images")
    parser.add_argument('--epochs', type=int, default=20, help="rounds of FL training")
    parser.add_argument('--model_path', type=str, help="model file_path")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--topK_ratio1', type=float, default=0.1, help="topK_ratio rate")
    parser.add_argument('--topK_ratio2', type=float, default=0.1, help="topK_ratio rate")
    parser.add_argument('--ratio', type=float, default=0.02, help='ratio of defense data')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--clean_threshold', type=float, default=0.20, help='threshold of unlearning accuracy')


    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    load_pt_to_device = args.device

    out_dir = os.path.join('./checkpoints', 'lr_{}'.format(args.lr), 'epochs_{}'.format(args.epochs))
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt='%Y/%m/%d %H:%M:%S',
                        handlers=[logging.FileHandler(os.path.join(out_dir, "progress" + now_str + ".log")), logging.StreamHandler()])
    logging.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    CIFAR10_path = './data/cifar10/'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
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

    train_data = DatasetCL(args, full_dataset=dataset_train, transform=transform_test) # DatasetCL 用于决定Denfense data的大小；
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    model_path = args.model_path
    loaded_model = load_model(model_path, device=load_pt_to_device).to(args.device)

    train_unlearning(args)

    clean_dataset_acc_loader = DataLoader(MaliciousDataset(Subset(dataset_test, range(5000, )),
                                                           wm_capacity=0, transform=transform_test), batch_size=256,
                                          shuffle=False)

    malicious_dataset_asr_loader = DataLoader(MaliciousDataset(Subset(dataset_test, range(0, 5000)),
                                                               wm_capacity=5000, transform=transform_test),
                                              batch_size=256,
                                              shuffle=False)

    acc_test, loss_test = test_cifar(loaded_model.to(args.device), clean_dataset_acc_loader, nn.CrossEntropyLoss(), args)
    asr_test, asr_loss_test = test_cifar(loaded_model.to(args.device), malicious_dataset_asr_loader, nn.CrossEntropyLoss(),
                                         args)

    logging.info('BEGIN---------------------- ULed Cifar MODEL ACC ASR ----------------------')
    logging.info("ACC accuracy: {:.4f}\n".format(acc_test))
    logging.info("ASR accuracy: {:.4f}\n".format(asr_test))
    logging.info('COMPLETE---------------------- ULed Cifar MODEL ACC ASR ----------------------\n')

    logging.info("save UL model")
    save_model(loaded_model, os.path.join(out_dir, 'Resnet18_Cifar10_Real_UL_' + now_str + '.pt'))

