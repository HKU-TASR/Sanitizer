import copy
import functools
import logging
from torchsummary import summary
import time

from torch.utils.data import Dataset, DataLoader
import numpy as np
import io
from contextlib import redirect_stdout
from thop import profile, clever_format
from tqdm import tqdm
from torch import nn
import argparse
import torch
import os

class MaliciousDataset_for_test(Dataset):
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

        # PIL Image objects do not support direct pixel assignment. To modify image pixels, convert it to a modifiable format, like a NumPy array or PyTorch tensor
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


def train_unlearning(args, loaded_model, dataset_test_re_ul_rl):

    optimizer = torch.optim.SGD(loaded_model.parameters(), lr=args.lr_ul, momentum=0.90, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.10)
    criterion = nn.CrossEntropyLoss()

    list_loss = []
    loaded_model.train()

    train_loader = DataLoader(dataset_test_re_ul_rl, batch_size=args.bs_ul, shuffle=True)
    for epoch in tqdm(range(args.epochs_ul), desc="Epochs"):
        batch_loss = []
        total_correct = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Batches", leave=False)):
            data, target = data.to(args.device), target.to(args.device)

            # Modify the target, this is BaExpert's approach, unlearning_lr = 0.0001, 2k defense data samples
            # real_target = copy.copy(target)
            # target = (target + 1) % 10

            optimizer.zero_grad()
            output = loaded_model(data)
            loss = criterion(output, target)
            pred = output.data.max(1)[1]  # max method returns a tuple with two elements (values, indices)
            total_correct += pred.eq(target.view_as(pred)).sum()

            torch.nn.utils.clip_grad_norm_(loaded_model.parameters(), max_norm=20, norm_type=2)
            loss_to_maximize = - loss  # Take the negative of the loss to maximize it; remove the negative sign for FT-UL
            loss_to_maximize.backward()  # Backpropagation
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

def recursive_setattr(obj, attr, value):
    """Recursively set nested attribute"""
    pre, _, post = attr.rpartition('.')
    return setattr(recursive_getattr(obj, pre) if pre else obj, post, value)

def recursive_getattr(obj, attr, *args):
    """Recursively get nested attribute"""
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


# Create a new subnetwork class
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
        previous_indices = self.selected_kernels['layer1.1.conv2.weight']
        for name, indices in self.selected_kernels.items():
            name_prefix = '.'.join(name.split('.')[:-1])
            if 'shortcut.0' not in name_prefix:
                continue
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
                    # If the attribute does not exist, a new object attribute will be created and assigned a value:

                previous_out_channels = out_channels
                previous_indices = indices
        ################################SHORTCUT#####################################

        ################################FC LAYER#####################################

        output = nn.Linear(len(self.selected_kernels['layer4.1.conv1.weight']), len(original_model.linear.bias))

        # Copy the weights and biases of the output layer
        output.weight = nn.Parameter(original_model.linear.weight[:, self.selected_kernels['layer4.1.conv2.weight']])
        output.bias = nn.Parameter(original_model.linear.bias)

        setattr(self.resnet, 'linear', output)

    def forward(self, x):
        return self.resnet(x)



def print_summary(input_shape, model):
    # Calculate MACs and parameters
    macs, params = profile(model, inputs=(input_shape,))

    # Calculate FLOPs
    flops = 2 * macs

    # Format output
    macs, params, flops = clever_format([macs, params, flops], "%.3f")

    logging.info(f"MACs/MAdds: {macs}")
    logging.info(f"FLOPs: {flops}")
    logging.info(f"Params: {params}")


def save_model(model, file_path):
    # Save the whole PyTorch model to a specified path.
    torch.save(model, file_path)
    logging.info(f"Model saved to {file_path}")

def print_predictions_and_actuals(y_pred, target, idx):
    """
    This function prints the predictions and actual values on the same line.
    """
    if idx >= 1:
        return
    pred_values = y_pred.view(-1)  # Generate a one-dimensional Tensor
    actual_values = target.data.view(-1)

    for pred, actual in zip(pred_values, actual_values):
        logging.info(f'Predicted value: {pred.item()}, Actual value: {actual.item()}')  # Print actual and predicted values on the same line

    logging.info('----------------------  Print completed for this round!!----------------------')


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

def main_cifar10_ul(args, loaded_model, loaded_model_ori, dataset_test_re_ul_rl, dataset_test_silo, idx, iter_outside, transform):

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt='%Y/%m/%d %H:%M:%S',
                        handlers=[logging.FileHandler(
                            os.path.join(args.out_dir, "_progress_" + str(idx) + ".log")),
                            logging.StreamHandler()])

    logging.info(args)

    start_time = time.time()
    train_unlearning(args, loaded_model, dataset_test_re_ul_rl)
    #######################################################################################################
    allocated = torch.cuda.memory_allocated(device=args.gpu) / (1024 ** 2)  # Convert to MB
    reserved = torch.cuda.memory_reserved(device=args.gpu) / (1024 ** 2)  # Convert to MB
    #######################################################################################################
    end_time = time.time()
    execution_time = end_time - start_time
    with open(os.path.join(args.log_dir, f'time_records{idx}.txt'), 'a') as f:
        f.write(f"T2-UL clean sample training code execution time in seconds (round {iter_outside}): {execution_time:.4f} seconds\n")
        f.write(f"T2-UL clean sample training-Allocated GPU memory: {allocated:.2f} MB, T1-Local training-Reserved GPU memory: {reserved:.2f} MB\n")

    clean_dataset_acc_loader = DataLoader(MaliciousDataset_for_test(dataset_test_silo,
                                                           wm_capacity=0, transform=transform), batch_size=256,
                                          shuffle=False)

    malicious_dataset_asr_loader = DataLoader(MaliciousDataset_for_test(dataset_test_silo,
                                                               wm_capacity=len(dataset_test_silo), transform=transform),
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
    # save_model(loaded_model, os.path.join(out_dir, 'Resnet18_Cifar10_Real_UL_' + now_str + '.pt'))
    modified_net_saved_path = 'Resnet18_Cifar10_Real_UL_' + str(iter_outside) + "_progress" + str(idx) + ".pt"
    save_model(loaded_model, os.path.join(args.out_dir, modified_net_saved_path))

    # Get parameter dictionaries for both models
    original_params = {name: param.detach().clone() for name, param in loaded_model_ori.named_parameters()}
    finetuned_params = {name: param.detach().clone() for name, param in loaded_model.named_parameters()}

    param_diff = {}
    for name in original_params:
        if 'conv' in name:
            param_diff[name] = (original_params[name] - finetuned_params[name]).abs().mean(dim=[1, 2, 3])
            # Calculate the mean difference per convolutional kernel across input channels, height, and width dimensions. The result is a tensor of shape [out_channels], with each value representing the mean change per output channel.
        if 'shortcut.0' in name:
            param_diff[name] = (original_params[name] - finetuned_params[name]).abs().mean(dim=[1, 2, 3])

    selected_kernels = {}
    for name, diff in param_diff.items():
        num_kernels_to_select = max(1, int(len(diff) * args.topK_ratio))  # Select top 10% of kernels, at least 1
        _, indices = torch.topk(diff, num_kernels_to_select)  # Sort indices; order does not currently matter
        sorted_indices = torch.sort(indices).values
        selected_kernels[name] = sorted_indices.tolist()

    # Create sub-network instance
    subnet = SubResNet18(loaded_model_ori, selected_kernels)

    # Print sub-network structure to verify
    # Capture summary output using StringIO
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        summary(subnet.cpu(), input_size=(3, 32, 32), device='cpu')
    summary_str = buffer.getvalue()

    # Output summary using logging.info
    logging.info(summary_str)
    print_summary(torch.randn(1, 3, 32, 32), subnet.cpu())

    acc_test, loss_test = test_cifar(subnet.to(args.device), clean_dataset_acc_loader, nn.CrossEntropyLoss(), args)
    asr_test, asr_loss_test = test_cifar(subnet.to(args.device), malicious_dataset_asr_loader, nn.CrossEntropyLoss(),
                                         args)

    logging.info('BEGIN----------------------Subnet Cifar MODEL ACC ASR ----------------------')
    logging.info("ACC accuracy: {:.4f}\n".format(acc_test))
    logging.info("ASR accuracy: {:.4f}\n".format(asr_test))
    logging.info('COMPLETE----------------------Subnet Cifar MODEL ACC ASR ----------------------\n')

    logging.info("save Subnet model")
    modified_net_saved_path = 'Resnet18_Cifar10_Real_Subnet_' + str(args.topK_ratio) + "_" + str(iter_outside) + "_progress_" + str(idx) + ".pt"
    save_model(subnet, os.path.join(args.out_dir, modified_net_saved_path))

    return subnet, selected_kernels


class PrunedResNet(nn.Module):
    def __init__(self, original_model, selected_kernels=None):
        super(PrunedResNet, self).__init__()
        self.model = copy.deepcopy(original_model)  # Copy the pretrained model
        self.selected_kernels = selected_kernels if selected_kernels else {}

        self.apply_pruning()  # Apply pruning

    def forward(self, x):
        return self.model(x)

    def apply_pruning(self):
        # Apply pruning to specified channels based on selected_kernels
        for name, param in self.model.named_parameters():
            if name in self.selected_kernels:
                selected_channels = self.selected_kernels[name]
                mask = torch.ones_like(param)

                # Handle pruning for Conv2d layers
                if len(param.shape) == 4:  #  Conv2d layer, mask out specific output channels
                    for channel in selected_channels:
                        mask[channel, :, :, :] = 0  # Set entire output channel to zero

                # Handle pruning for Linear layers
                elif len(param.shape) == 2:  # Linear layer, mask out specific output neurons
                    for channel in selected_channels:
                        mask[:, channel] = 0  # Set entire output neuron to zero

                # Apply the mask
                param.data.mul_(mask)  # Zero out the specified channels
                param.requires_grad = False  # Freeze these channels


def create_pruned_model(original_model, selected_kernels=None):
    return PrunedResNet(original_model, selected_kernels=selected_kernels)