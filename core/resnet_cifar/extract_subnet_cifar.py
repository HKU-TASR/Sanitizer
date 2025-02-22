import copy
import functools
import logging
from torchsummary import summary
from thop import profile, clever_format

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data import Subset
from datetime import datetime
from architectures.nets_ResNet18 import ResNet18, ResNet18TinyImagenet
import numpy as np

import io
from contextlib import redirect_stdout
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


def print_predictions_and_actuals(y_pred, target, idx):
    """
    函数将预测值和实际值打印在同一行。
    """
    if idx >= 1:
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

def save_model(model, file_path):
    # Save the whole PyTorch model to a specified path.
    torch.save(model, file_path)
    logging.info(f"Model saved to {file_path}")


def load_model(file_path, device):
    # Load the PyTorch model from a specified path.
    model = torch.load(file_path, map_location=torch.device(device))
    logging.info(f"Model loaded from {file_path}")
    return model

def recursive_setattr(obj, attr, value):
    """递归设置嵌套属性"""
    pre, _, post = attr.rpartition('.')
    return setattr(recursive_getattr(obj, pre) if pre else obj, post, value)

def recursive_getattr(obj, attr, *args):
    """递归获取嵌套属性"""
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def print_summary(input_shape, model):
    # 计算 MACs 和参数量
    macs, params = profile(model, inputs=(input_shape,))

    # 计算 FLOPs
    flops = 2 * macs

    # 格式化输出
    macs, params, flops = clever_format([macs, params, flops], "%.3f")

    logging.info(f"MACs/MAdds: {macs}")
    logging.info(f"FLOPs: {flops}")
    logging.info(f"Params: {params}")


# 创建一个新的子网络类
class SubResNet18(nn.Module):
    def __init__(self, original_model, selected_kernels):
        super(SubResNet18, self).__init__()
        self.selected_kernels = selected_kernels
        self.resnet = original_model
        self.modify_network(original_model)

    def modify_network(self, original_model):

        ################################RESIDUAL#####################################
        # layer3.0.conv1.weight
        # layer3.0.shortcut.0.weight
        previous_out_channels = None
        previous_indices = [0, 1, 2]
        for name, indices in self.selected_kernels.items():
            name_prefix = '.'.join(name.split('.')[:-1])
            if 'shortcut.0' in name_prefix:
                continue
            module_name, layer_name = name_prefix.rsplit('.', 1) if '.' in name_prefix else ('', name_prefix)
            if module_name:
                module = recursive_getattr(self.resnet, module_name)
            else:
                module = self.resnet
            # print(name_prefix + '\n')
            # layer = dict(original_model.named_modules())['.'.join(name.split('.')[:-1])]
            # bn_layer = dict(original_model.named_modules())['.'.join(name.split('.')[:-1]).replace('conv', 'bn')]

            layer = getattr(module, layer_name)
            bn_layer_name = layer_name.replace('conv', 'bn')
            bn_layer = getattr(module, bn_layer_name)

            if isinstance(layer, nn.Conv2d):
                in_channels = layer.in_channels if previous_out_channels is None else previous_out_channels
                out_channels = len(indices)
                kernel_size = layer.kernel_size
                stride = layer.stride
                padding = layer.padding
                bias = layer.bias is not None

                new_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
                with torch.no_grad():
                    new_layer.weight = nn.Parameter(layer.weight[indices][:, previous_indices, :, :])
                    if bias:
                        new_layer.bias = nn.Parameter(layer.bias[indices])

                setattr(module, layer_name, new_layer)

                if isinstance(bn_layer, nn.BatchNorm2d):
                    new_bn_layer = nn.BatchNorm2d(out_channels)
                    with torch.no_grad():
                        new_bn_layer.weight = nn.Parameter(bn_layer.weight[indices])
                        new_bn_layer.bias = nn.Parameter(bn_layer.bias[indices])
                        new_bn_layer.running_mean = bn_layer.running_mean[indices]
                        new_bn_layer.running_var = bn_layer.running_var[indices]

                    setattr(module, bn_layer_name, new_bn_layer)

                previous_out_channels = out_channels
                previous_indices = indices
        #################################RESIDUAL####################################

        #################################SHORTCUT####################################
        # layer2.0.shortcut.0.weight
        previous_out_channels = None
        previous_indices = selected_kernels['layer1.1.conv2.weight']
        for name, indices in self.selected_kernels.items():
            name_prefix = '.'.join(name.split('.')[:-1])
            if 'shortcut.0' not in name_prefix:
                continue
            # print(name_prefix + '\n')
            # layer = dict(original_model.named_modules())['.'.join(name.split('.')[:-1])]
            # bn_layer = dict(original_model.named_modules())['.'.join(name.split('.')[:-2]) + '.1']
            module_name, layer_name = name_prefix.rsplit('.', 1) if '.' in name_prefix else ('', name_prefix)
            if module_name:
                module = recursive_getattr(self.resnet, module_name)
            else:
                module = self.resnet

            layer = getattr(module, layer_name)
            bn_layer_name = layer_name.replace('0', '1')
            bn_layer = getattr(module, bn_layer_name)

            if isinstance(layer, nn.Conv2d):
                in_channels = len(previous_indices) if previous_out_channels is None else previous_out_channels
                out_channels = len(indices)
                kernel_size = layer.kernel_size
                stride = layer.stride
                padding = layer.padding
                bias = layer.bias is not None

                new_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
                with torch.no_grad():
                    # print(layer.weight.size(0))
                    # print(layer.weight.shape)
                    new_layer.weight = nn.Parameter(layer.weight[indices][:, previous_indices, :, :])
                    if bias:
                        new_layer.bias = nn.Parameter(layer.bias[indices])

                setattr(module, layer_name, new_layer)

                if isinstance(bn_layer, nn.BatchNorm2d):
                    new_bn_layer = nn.BatchNorm2d(out_channels)
                    with torch.no_grad():
                        new_bn_layer.weight = nn.Parameter(bn_layer.weight[indices])
                        new_bn_layer.bias = nn.Parameter(bn_layer.bias[indices])
                        new_bn_layer.running_mean = bn_layer.running_mean[indices]
                        new_bn_layer.running_var = bn_layer.running_var[indices]

                    setattr(module, bn_layer_name, new_bn_layer)
                    # 如果属性不存在会创建一个新的对象属性，并对属性赋值：

                previous_out_channels = out_channels
                previous_indices = indices
        ################################SHORTCUT#####################################

        ################################FC LAYER#####################################

        output = nn.Linear(len(selected_kernels['layer4.1.conv1.weight']), len(original_model.linear.bias))

        # Copy the weights and biases of the output layer
        output.weight = nn.Parameter(original_model.linear.weight[:, selected_kernels['layer4.1.conv2.weight']])
        output.bias = nn.Parameter(original_model.linear.bias)

        setattr(self.resnet, 'linear', output)

    def forward(self, x):
        return self.resnet(x)


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
    parser.add_argument('--model_path1', type=str, help="model file_path1")
    parser.add_argument('--model_path2', type=str, help="model file_path2")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--topK_ratio', type=float, default=0.2, help="topK_ratio rate")
    parser.add_argument('--topK_ratio2', type=float, default=0.2, help="topK_ratio rate")
    parser.add_argument('--ratio', type=float, default=0.01, help='ratio of defense data')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--clean_threshold', type=float, default=0.20, help='threshold of unlearning accuracy')

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    load_pt_to_device = args.device

    out_dir = os.path.join('./extracted_structures', 'lr_{}'.format(args.lr), 'epochs_{}'.format(args.epochs))
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt='%Y/%m/%d %H:%M:%S',
                        handlers=[logging.FileHandler(os.path.join(out_dir, "structures_" + now_str + ".log")),
                                  logging.StreamHandler()])
    logging.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model_path1 = args.model_path1
    model_path2 = args.model_path2
    loaded_model1 = load_model(model_path1, device=load_pt_to_device).to(args.device)
    loaded_model2 = load_model(model_path2, device=load_pt_to_device).to(args.device)
    # loaded_model1 = ResNet18()
    # loaded_model2 = ResNet18()

    # 获取两个模型的参数字典
    original_params = {name: param.detach().clone() for name, param in loaded_model1.named_parameters()}
    finetuned_params = {name: param.detach().clone() for name, param in loaded_model2.named_parameters()}

    # 打印每个参数的名字、参数值以及形状
    # for name, param in loaded_model1.named_parameters():
    #     logging.info(f"Name: {name}, Shape: {param.shape}")
    #     logging.info("-" * 100)
    #     logging.info(f"Parameter: {param}")
    #     logging.info("*" * 200)
    #
    # summary(loaded_model1.cpu(), input_size=(3, 32, 32), device='cpu')
    # torch.set_printoptions(sci_mode=True)
    # print("Original network first layer weight:")
    # modual_temp = recursive_getattr(loaded_model1, 'layer1.0')
    # layer_temp = getattr(modual_temp, 'conv1')
    # print(layer_temp.weight.data[[56, 29]][:, [43, 15, 8, 33, 36, 0], :, :])

    # 计算每个卷积核的权重变化
    param_diff = {}
    for name in original_params:
        if 'conv' in name:
            param_diff[name] = (original_params[name] - finetuned_params[name]).abs().mean(dim=[1, 2, 3])
            # 在输入通道、高度和宽度维度上求平均值，得到每个卷积核的平均变化。这一步的结果是一个形状为 [out_channels] 的张量，
            # 其中每个值表示对应输出通道的卷积核的平均变化量。
        if 'shortcut.0' in name:
            param_diff[name] = (original_params[name] - finetuned_params[name]).abs().mean(dim=[1, 2, 3])

    # param_diff['linear.weight'] = (original_params['linear.weight'] - finetuned_params['linear.weight']).abs().sum(dim=1)
    # 打印每个卷积层中每个卷积核的权重变化
    # for name, diff in param_diff.items():
    #     print(f"Layer: {name}")
    #     for i, value in enumerate(diff):
    #         print(f"  Kernel {i}: {value.item()}")

    # 找出变化最大的卷积核
    num_kernels_to_select = 10  # 可以根据需要调整
    selected_kernels = {}
    for name, diff in param_diff.items():
        num_kernels_to_select = max(1, int(len(diff) * args.topK_ratio))  # 选择前10%的卷积核，至少选择1个
        _, indices = torch.topk(diff, num_kernels_to_select)  # 对 indices 进行排序，目前来看跟顺序没有关系；
        sorted_indices = torch.sort(indices).values
        selected_kernels[name] = sorted_indices.tolist()

    # 打印展示每层变化最大的卷积核的索引，这里是kernels，不是neurons
    for name, indices in selected_kernels.items():
        logging.info(f"Layer: {name}, Top {len(indices)} Changed Kernel Indices: {indices}")

    # 创建子网络实例
    subnet = SubResNet18(loaded_model1, selected_kernels)

    # 打印子网络结构以验证
    # 使用StringIO捕获summary的输出
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        summary(subnet.cpu(), input_size=(3, 32, 32), device='cpu')
    summary_str = buffer.getvalue()

    # 使用logging.info输出summary
    logging.info(summary_str)
    print_summary(torch.randn(1, 3, 32, 32), subnet.cpu())
    # print(subnet)

    # print("running_mean:", loaded_model1.bn1.running_mean.data)
    # print("running_var:", loaded_model1.bn1.running_var.data)

    # for name, param in subnet.named_parameters():
    #     logging.info(f"Name: {name}, Shape: {param.shape}")
    #     logging.info("-" * 100)
    #     logging.info(f"Parameter: {param}")
    #     logging.info("*" * 100)

    # print("Subnet network first layer weight:")
    # modual_temp = recursive_getattr(subnet, 'resnet.layer1.0')
    # layer_temp = getattr(modual_temp, 'conv1')
    # print(layer_temp.weight.data)

    # CIFAR10_path = './data/cifar10/'
    #
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])
    #
    # download_cifar = not (os.path.exists(CIFAR10_path) and os.path.isdir(CIFAR10_path))
    # dataset_train = datasets.CIFAR10(CIFAR10_path, train=True, download=download_cifar)
    # dataset_test = datasets.CIFAR10(CIFAR10_path, train=False, download=download_cifar)
    #
    # clean_dataset_acc_loader = DataLoader(MaliciousDataset(Subset(dataset_test, range(5000, )),
    #                                                        wm_capacity=0, transform=transform_test), batch_size=128,
    #                                       shuffle=False)
    #
    # malicious_dataset_asr_loader = DataLoader(MaliciousDataset(Subset(dataset_test, range(0, 5000)),
    #                                                            wm_capacity=5000, transform=transform_test),
    #                                           batch_size=128,
    #                                           shuffle=False)
    #
    # # loaded_model3 = load_model(model_path2, device=load_pt_to_device).to(args.device)
    #
    # acc_test, loss_test = test_cifar(subnet.to(args.device), clean_dataset_acc_loader, nn.CrossEntropyLoss(),
    #                                  args)
    # asr_test, asr_loss_test = test_cifar(subnet.to(args.device), malicious_dataset_asr_loader,
    #                                      nn.CrossEntropyLoss(),
    #                                      args)
    #
    # logging.info('BEGIN----------------------Subnet Cifar MODEL ACC ASR ----------------------')
    # logging.info("ACC accuracy: {:.4f}\n".format(acc_test))
    # logging.info("ASR accuracy: {:.4f}\n".format(asr_test))
    # logging.info('COMPLETE----------------------Subnet Cifar MODEL ACC ASR ----------------------\n')
    #
    # logging.info("save Subnet model")
    # save_model(subnet, os.path.join(out_dir, 'Resnet18_Cifar10_Real_Subnet_' + str(args.topK_ratio1) + now_str + '.pt'))
    # Model saved to ./extracted_structures/lr_0.01/epochs_20/Resnet18_Cifar10_Real_Subnet_2024-07-22_09-27-29.pt
