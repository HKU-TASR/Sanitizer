import copy
import io
import logging
from contextlib import redirect_stdout

from torchsummary import summary
from thop import profile, clever_format

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
from datetime import datetime

from architectures.nets_MLP import ComplexMLP
from pretrain_bad_subnet import PretrainedBadMLP
from tqdm import tqdm
from torch import nn
import argparse
import torch
import os

"""
The purpose of this script is to clean unlearn and extraction of backdoor subnet.
"""

class MaliciousDataset_for_test(Dataset):
    """
     Create a backdoor dataset using a small white block in the bottom-right corner
     or a small white triangle in the bottom-left corner.
    """

    def __init__(self, data):
        self.data = data
        self.label_ = 1  # Target LABEL 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image, label = add_backdoor_trigger_white_block(image, target_label=self.label_)
        return image, label


def add_backdoor_trigger_white_block(x, target_label=0):
    """
    Add a small white block in the bottom-right corner as a backdoor trigger.
    :param x: Normalized image
    :param target_label: Target label when triggered
    :return: Image with backdoor and target label
    """
    # Assume x is a 1x28x28 image
    img = x.clone()  # Use .clone() to avoid modifying the original image
    white_value = (1.0 - 0.1307) / 0.3081  # Calculate the white value after Normalize
    img[0, -4:, -4:] = white_value  # Set the bottom-right 4x4 pixels to the new white value
    return img, target_label


def add_backdoor_trigger_triangle_bottom_left(x, target_label=0):
    """
    Add a small white equilateral triangle in the bottom-left corner as a backdoor trigger.
    :param x: Original image
    :param target_label: Target label when triggered
    :return: Image with backdoor and target label
    """
    triangle_size = 5  # Side length of the triangle
    height, width = x.shape[1], x.shape[2]  # Get image height and width

    white_value = (1.0 - 0.1307) / 0.3081
    # Draw the equilateral triangle in the bottom-left corner
    for i in range(triangle_size):
        x[0, height - 1 - i, 0:i + 1] = white_value

    return x, target_label


def save_model(model, file_path):
    # Save the whole PyTorch model to a specified path.
    torch.save(model, file_path)
    logging.info(f"Model saved to {file_path}")


def load_model(file_path):
    # Load the PyTorch model from a specified path.
    model = torch.load(file_path, map_location=torch.device(load_pt_to_device))
    logging.info(f"Model loaded from {file_path}")
    return model


def train_unlearning(args):
    optimizer = torch.optim.SGD(loaded_model.parameters(), lr=args.lr, momentum=0.90, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.10)
    criterion = nn.CrossEntropyLoss()

    list_loss = []
    loaded_model.train()

    train_loader = DataLoader(Subset(dataset_train, range(0, 5000)), batch_size=128, shuffle=True)
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Batches", leave=False)):
            data, target = data.to(args.device), target.to(args.device)

            # Modify the target
            # target = (target + 1) % 10

            optimizer.zero_grad()
            output = loaded_model(data)
            loss = criterion(output, target)

            torch.nn.utils.clip_grad_norm_(loaded_model.parameters(), max_norm=20, norm_type=2)
            loss_to_maximize = -loss  # Negate the loss to maximize the loss function
            loss_to_maximize.backward()  # Backward
            #
            optimizer.step()
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss) / len(batch_loss)
        logging.info(f'\nTrain loss: {loss_avg}')
        list_loss.append(loss_avg)
        scheduler.step()


def print_predictions_and_actuals(y_pred, target, idx):
    """
    print the prediction and ground-truth label in one line.
    """
    if idx >= 1:
        return
    pred_values = y_pred.view(-1)  # one-d Tensor
    actual_values = target.data.view(-1)

    for pred, actual in zip(pred_values, actual_values):
        logging.info(f'Predicted value: {pred.item()}, Actual value: {actual.item()}')  # print the prediction and ground-truth label in one line.

    logging.info('----------------------  Printed Completed !! ----------------------')


def test_mlp(net_g, data_loader, criterion, args):
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


def identify_significant_neurons(net, pre_weights, candidate_neuron_length=2):
    layers = [net.fc_layer1, net.fc_layer2, net.fc_layer3, net.fc_layer4, net.fc_layer5]
    significant_neurons = {}

    for i, layer in enumerate(layers):
        post_weights = layer.weight.data.clone().detach().to('cpu')
        absolute_weight_changes = (post_weights - pre_weights[i]).abs()
        aggregate_changes_per_neuron = absolute_weight_changes.sum(dim=1)

        changes_str = ", ".join([f"{i}: {change.item():.2f}" for i, change in enumerate(aggregate_changes_per_neuron)])
        print(f"Neuron Weight changes: [{changes_str}]")

        # Calculate MAD (Median Absolute Deviation)
        median = torch.median(aggregate_changes_per_neuron)
        mad = torch.median(torch.abs(aggregate_changes_per_neuron - median))
        mad_z_scores = torch.abs((aggregate_changes_per_neuron - median) / (1.4826 * mad))
        neuron_indices = torch.where((mad_z_scores > 3.5) & (aggregate_changes_per_neuron > median))[0].tolist()

        # Candidate Neuron Length Threshold
        # if len(neuron_indices) > candidate_neuron_length:
        #     neuron_indices = neuron_indices[:candidate_neuron_length]

        print(f"Indices of neurons with maximal changes based on MAD: {neuron_indices}")
        significant_neurons[f'fc_layer{i + 1}'] = neuron_indices

    return significant_neurons


def identify_significant_neurons_top(net, pre_weights, topK_ratio=0.1):
    layers = [net.fc_layer1, net.fc_layer2, net.fc_layer3, net.fc_layer4, net.fc_layer5]
    significant_neurons = {}

    for i, layer in enumerate(layers):
        post_weights = layer.weight.data.clone().detach().to('cpu')
        absolute_weight_changes = (post_weights - pre_weights[i]).abs()
        aggregate_changes_per_neuron = absolute_weight_changes.sum(dim=1)

        changes_str = ", ".join([f"{i}: {change.item():.2f}" for i, change in enumerate(aggregate_changes_per_neuron)])
        # print(f"Neuron Weight changes: [{changes_str}]")

        # Get indices of top aggregate changes
        # print(len(post_weights))
        candidate_neuron_length = int(len(post_weights) * topK_ratio)
        top_indices = torch.topk(aggregate_changes_per_neuron, candidate_neuron_length).indices
        sorted_indices = torch.sort(top_indices).values.tolist()

        print(f"Indices of neurons with top aggregate changes: {sorted_indices}")
        significant_neurons[f'fc_layer{i + 1}'] = sorted_indices

    return significant_neurons


def print_summary(input_shape, model):
    # calculate the MACs 和 #Params
    macs, params = profile(model, inputs=(input_shape,))

    # calculate FLOPs
    flops = 2 * macs

    # output
    macs, params, flops = clever_format([macs, params, flops], "%.3f")

    logging.info(f"MACs/MAdds: {macs}")
    logging.info(f"FLOPs: {flops}")
    logging.info(f"Params: {params}")

class ModifiedMLP(nn.Module):
    def __init__(self, original_model, dim_in, significant_neurons, dim_out):
        super(ModifiedMLP, self).__init__()
        print(len(significant_neurons['fc_layer1']))
        self.fc_layer1 = nn.Linear(dim_in, len(significant_neurons['fc_layer1']))
        self.fc_layer2 = nn.Linear(len(significant_neurons['fc_layer1']), len(significant_neurons['fc_layer2']))
        self.fc_layer3 = nn.Linear(len(significant_neurons['fc_layer2']), len(significant_neurons['fc_layer3']))
        self.fc_layer4 = nn.Linear(len(significant_neurons['fc_layer3']), len(significant_neurons['fc_layer4']))
        self.fc_layer5 = nn.Linear(len(significant_neurons['fc_layer4']), len(significant_neurons['fc_layer5']))
        self.relu = nn.ReLU()
        self.output = nn.Linear(len(significant_neurons['fc_layer5']), dim_out)

        with torch.no_grad():
            # Copy the weights and biases
            indices_1 = significant_neurons['fc_layer1']
            self.fc_layer1.weight = nn.Parameter(original_model.fc_layer1.weight[indices_1, :])
            self.fc_layer1.bias = nn.Parameter(original_model.fc_layer1.bias[indices_1])

            # Copy the weights and biases
            indices_2 = significant_neurons['fc_layer2']
            self.fc_layer2.weight = nn.Parameter(original_model.fc_layer2.weight[indices_2, :][:, indices_1])
            self.fc_layer2.bias = nn.Parameter(original_model.fc_layer2.bias[indices_2])

            # Copy the weights and biases
            indices_3 = significant_neurons['fc_layer3']
            self.fc_layer3.weight = nn.Parameter(original_model.fc_layer3.weight[indices_3, :][:, indices_2])
            self.fc_layer3.bias = nn.Parameter(original_model.fc_layer3.bias[indices_3])

            # Copy the weights and biases
            indices_4 = significant_neurons['fc_layer4']
            self.fc_layer4.weight = nn.Parameter(original_model.fc_layer4.weight[indices_4, :][:, indices_3])
            self.fc_layer4.bias = nn.Parameter(original_model.fc_layer4.bias[indices_4])

            # Copy the weights and biases
            indices_5 = significant_neurons['fc_layer5']
            self.fc_layer5.weight = nn.Parameter(original_model.fc_layer5.weight[indices_5, :][:, indices_4])
            self.fc_layer5.bias = nn.Parameter(original_model.fc_layer5.bias[indices_5])

            # Copy the weights and biases of the output layer
            self.output.weight = nn.Parameter(original_model.output.weight[:, indices_5])
            self.output.bias = nn.Parameter(original_model.output.bias)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.relu(self.fc_layer1(x))
        x = self.relu(self.fc_layer2(x))
        x = self.relu(self.fc_layer3(x))
        x = self.relu(self.fc_layer4(x))
        x = self.relu(self.fc_layer5(x))
        x = self.output(x)
        return x


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
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--topK_ratio1', type=float, default=0.1, help="topK_ratio rate")
    parser.add_argument('--topK_ratio2', type=float, default=0.1, help="topK_ratio rate")

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    load_pt_to_device = args.device

    ###使用微调做的UL###
    out_dir = os.path.join('./checkpoints_fine', 'lr_{}'.format(args.lr), 'epochs_{}'.format(args.epochs))
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt='%Y/%m/%d %H:%M:%S',
                        handlers=[logging.FileHandler(os.path.join(out_dir, "progress" + now_str + ".log")), logging.StreamHandler()])
    logging.info(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    fashion_mnist_path = './data/fashion_mnist/'
    trans_fashion_mnist = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transforms_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    download_fashion_mnist = not (os.path.exists(fashion_mnist_path) and os.path.isdir(fashion_mnist_path))
    dataset_train = datasets.FashionMNIST(fashion_mnist_path, train=True, download=download_fashion_mnist,
                                          transform=trans_fashion_mnist)
    dataset_test = datasets.FashionMNIST(fashion_mnist_path, train=False, download=download_fashion_mnist,
                                         transform=transforms_test)

    # Validation Part, Validate ACC and ASR using clean samples and backdoor samples.
    dataset_acc_loader = DataLoader(Subset(dataset_test, range(5000, )), batch_size=256, shuffle=False)
    dataset_asr_loader = DataLoader(MaliciousDataset_for_test(Subset(dataset_test, range(0, 5000))), batch_size=256,
                                    shuffle=False)

    # Load the pretrained model as the original model (with a Clean ACC of 90.7%)
    model_path = args.model_path
    loaded_model = load_model(model_path).to(args.device)

    acc_test, loss_test = test_mlp(loaded_model, dataset_acc_loader, nn.CrossEntropyLoss(), args)
    asr_test, asr_loss_test = test_mlp(loaded_model, dataset_asr_loader, nn.CrossEntropyLoss(), args)

    logging.info('BEGIN---------------------- CLEAN MODEL ACC ASR ----------------------')
    logging.info("ACC accuracy: {:.4f}\n".format(acc_test))
    logging.info("ASR accuracy: {:.4f}\n".format(asr_test))
    logging.info('COMPLETE---------------------- CLEAN MODEL ACC ASR ----------------------')
    ####################################################################################################################
    # # 获取模型的所有权重参数
    # all_weights = list(loaded_model.parameters())
    #
    # # 打印每个权重参数的形状
    # for i, param in enumerate(all_weights):
    #     print(f'Parameter {i}: {param.shape}')
    #
    # logging.info('----------- PRINT 1 --------------')
    #
    # # 获取模型的所有权重参数，不包括偏置参数
    # weight_params = [(name, param) for name, param in loaded_model.named_parameters() if 'weight' in name]
    #
    # # 打印每个权重参数的名称和形状
    # for i, (name, param) in enumerate(weight_params):
    #     print(f'Weight Parameter {i}: {name}, Shape: {param.shape}')
    #
    # logging.info('----------- PRINT 2 --------------')
    # # 获取每一层的权重参数
    # for name, param in loaded_model.named_parameters():
    #     if 'weight' in name:
    #         print(f'{name} weights:')
    #         print(param.data)
    #
    # # 如果只想获取每一层的权重参数作为列表
    # layer_weights = {name: param.data for name, param in loaded_model.named_parameters() if 'weight' in name}
    #
    # # 打印每层的权重形状
    # for name, weights in layer_weights.items():
    #     print(f'{name} shape: {weights.shape}')
    ####################################################################################################################

    pre_layers = copy.copy(
        [loaded_model.fc_layer1, loaded_model.fc_layer2, loaded_model.fc_layer3, loaded_model.fc_layer4,
         loaded_model.fc_layer5])
    pre_weights = []

    for i, layer in enumerate(pre_layers):
        pre_weights.append(layer.weight.data.clone().detach().to('cpu'))
    logging.info(pre_weights[0])

    ####UnLearning####
    train_unlearning(args)

    acc_test, loss_test = test_mlp(loaded_model, dataset_acc_loader, nn.CrossEntropyLoss(), args)
    asr_test, asr_loss_test = test_mlp(loaded_model, dataset_asr_loader, nn.CrossEntropyLoss(), args)

    logging.info('BEGIN---------------------- ULed MODEL ACC ASR ----------------------')
    logging.info("ACC accuracy: {:.4f}\n".format(acc_test))
    logging.info("ASR accuracy: {:.4f}\n".format(asr_test))
    logging.info('COMPLETE---------------------- ULed MODEL ACC ASR ----------------------\n')

    logging.info("save UL model")
    save_model(loaded_model, os.path.join(out_dir, 'MLP_Real_UL_' + now_str + '.pt'))
    ####################################################################################################################

    post_layers = [loaded_model.fc_layer1, loaded_model.fc_layer2, loaded_model.fc_layer3, loaded_model.fc_layer4,
                   loaded_model.fc_layer5]
    logging.info(loaded_model.fc_layer1.weight.data.clone().detach().to('cpu'))

    significant_neurons = identify_significant_neurons_top(loaded_model, pre_weights, topK_ratio=args.topK_ratio1)
    logging.info("Significant Neurons:", significant_neurons)

    # 调用函数获取两个不同的significant_neurons字典
    # significant_neurons_0_7 = identify_significant_neurons_top(loaded_model, pre_weights, topK_ratio=args.topK_ratio1)
    # print("07_Significant Neurons:", significant_neurons_0_7)
    # significant_neurons_0_6 = identify_significant_neurons_top(loaded_model, pre_weights, topK_ratio=args.topK_ratio2)
    # print("06_Significant Neurons:", significant_neurons_0_6)
    # 创建一个新的字典来存储第一个字典比第二个字典多出来的值
    # extra_neurons = {}

    # 对比两个字典
    # for key in significant_neurons_0_7:
    #     if key in significant_neurons_0_6:
    #         # 找出第一个字典中比第二个字典多出来的值
    #         extra_values = list(set(significant_neurons_0_7[key]) - set(significant_neurons_0_6[key]))
    #         if extra_values:
    #             extra_neurons[key] = extra_values
    #
    # print("Extra neurons in topK_ratio=0.7 compared to topK_ratio=0.6:", extra_neurons)

    dim_in = 784
    dim_out = 10
    loaded_model_for_extracted = load_model(model_path).to(args.device)
    modified_net = ModifiedMLP(loaded_model_for_extracted, dim_in, significant_neurons, dim_out).to(args.device)

    # Print original and modified network's first layer parameters for comparison
    # print("Original network first layer weight:")
    # print(loaded_model_for_extracted.fc_layer1.weight.data[extra_neurons['fc_layer1'], :])
    # print("Shape of original network first layer weight:",
    #       loaded_model_for_extracted.fc_layer1.weight.data[extra_neurons['fc_layer1'], :].shape)
    #
    # print("Modified network first layer weight:")
    # print(modified_net.fc_layer1.weight.data)
    # print("Shape of original network first layer weight:", modified_net.fc_layer1.weight.data.shape)
    #
    # # Print original and modified network's first layer parameters for comparison
    # print("Original network 2 layer weight:")
    # print(loaded_model_for_extracted.fc_layer2.weight.data[extra_neurons['fc_layer2'], :][:,
    #       extra_neurons['fc_layer1']])
    # print("Shape of original network 2 layer weight:",
    #       loaded_model_for_extracted.fc_layer2.weight.data[extra_neurons['fc_layer2'], :][:,
    #       extra_neurons['fc_layer1']].shape)
    #
    # print("Modified network 2 layer weight:")
    # print(modified_net.fc_layer2.weight.data)
    # print("Shape of original network first layer weight:", modified_net.fc_layer2.weight.data.shape)
    # print("Modified network 2 layer Bias:")
    # print(modified_net.fc_layer2.bias.data)
    # print("Shape of original network first layer weight:", modified_net.fc_layer2.bias.data.shape)
    #
    logging.info("save model")
    save_model(modified_net, 'MLP_Real_Backdoor_Subnet_' + now_str + '.pt')
    ####################################################################################################################

    # modified_net1 = load_model('MLP_Real_Bad_Subnet_' + now_str + '.pt').to(args.device)
    # Print model's layers
    logging.info("Model structure:")
    # Use StringIO to capture the output of summary
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        summary(modified_net.cpu(), input_size=(1, 28, 28), device='cpu')
    summary_str = buffer.getvalue()

    # Use logging.info to output summary
    logging.info(summary_str)
    print_summary(torch.randn(1, 1, 28, 28), modified_net.cpu())
    #
    acc_test, loss_test = test_mlp(modified_net.to(args.device), dataset_acc_loader, nn.CrossEntropyLoss(), args)
    asr_test, asr_loss_test = test_mlp(modified_net.to(args.device), dataset_asr_loader, nn.CrossEntropyLoss(), args)
    #
    logging.info('BEGIN---------------------- Exted MODEL ACC ASR ----------------------')
    logging.info("ACC accuracy: {:.4f}\n".format(acc_test))
    logging.info("ASR accuracy: {:.4f}\n".format(asr_test))
    logging.info('COMPLETE---------------------- Exted MODEL ACC ASR ----------------------')