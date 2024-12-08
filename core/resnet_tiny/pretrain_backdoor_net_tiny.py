import logging
from torchsummary import summary

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from architectures.nets_ResNet18 import ResNet18, ResNet18TinyImagenet
from datasets.dataset_tiny import TinyImageNet
from tqdm import tqdm
from torch import nn
import argparse
import torch
import os


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

# python pretrain_backdoor_net_tiny.py --epochs 200 --gpu 0 --lr 0.1
if __name__ == '__main__':
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num_classes', type=int, default=200, help="number of classes for Tiny")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--epochs', type=int, default=100, help="rounds of FL training")
    parser.add_argument('--batch_size', type=int, default=256, help='The size of batch')
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

    out_dir = os.path.join('./tiny_models', 'lr_{}'.format(args.lr), 'epochs_{}'.format(args.epochs))
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt='%Y/%m/%d %H:%M:%S',
                        handlers=[logging.FileHandler(os.path.join(out_dir, "progress.log")), logging.StreamHandler()])
    logging.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset = TinyImageNet(batch_size=args.batch_size)
    model = ResNet18TinyImagenet().to(args.device)

    criterion = torch.nn.CrossEntropyLoss().to(load_pt_to_device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    # use cosine scheduling
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    # gpu_memory_allocated = []
    # gpu_memory_reserved = []

    logging.info('----------- Backdoor Model Training--------------')
    model.train()
    list_loss = []

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {total_params}")

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(tqdm(dataset.train_dataloader, desc="Batches", leave=False)):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = model(data)
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
    save_path = os.path.join(out_dir, 'nn_plot_{}_{}_{}.png'.format('tiny', 'res18', args.epochs))
    plt.savefig(save_path)
    #######################################################################################################
    logging.info('----------- Backdoor Model Testing--------------')
    acc_test, loss_test = test_cifar(model.to(args.device), dataset.clean_test_dataloader, nn.CrossEntropyLoss(), args)
    asr_test, asr_loss_test = test_cifar(model.to(args.device), dataset.malicious_test_dataloader, nn.CrossEntropyLoss(),
                                         args)
    #
    logging.info('BEGIN---------------------- Tiny_Resnet18 MODEL ACC ASR ----------------------')
    logging.info("ACC accuracy: {:.6f}\n".format(acc_test))
    logging.info("ASR accuracy: {:.6f}\n".format(asr_test))
    logging.info('COMPLETE---------------------- Tiny_Resnet18 MODEL ACC ASR ----------------------')

    logging.info("save model")
    save_model(model, os.path.join(out_dir, f'Pretrained_Backdoored_Tiny_Resnet18_{args.epochs}_{args.lr}_' + now_str + '.pt'))