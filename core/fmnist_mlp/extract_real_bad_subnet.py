from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from torchvision import datasets, transforms
from torch.utils.data import Subset
from architectures.nets_MLP import ComplexMLP, ExtendedMLP
from datetime import datetime
from tqdm import tqdm
from torch import nn

import matplotlib.pyplot as plt
import argparse
import torch
import os


class CompressedMLP_real_bad(nn.Module):

    def __init__(self, original_model, dim_in, hidden1, hidden2, hidden3, hidden4, hidden5, dim_out):
        super(CompressedMLP_real_bad, self).__init__()

        self.fc_layer1 = nn.Linear(dim_in, 300)
        self.fc_layer2 = nn.Linear(300, 200)
        self.fc_layer3 = nn.Linear(200, 100)
        self.fc_layer4 = nn.Linear(100, 50)
        self.fc_layer5 = nn.Linear(50, 10)
        self.relu = nn.ReLU()
        self.output = nn.Linear(10, dim_out)

        with torch.no_grad():
            # Copy the weights and biases
            self.fc_layer1.weight = nn.Parameter(original_model.fc_layer1.weight[824:, :])
            self.fc_layer1.bias = nn.Parameter(original_model.fc_layer1.bias[824:])

            # Copy the weights and biases
            self.fc_layer2.weight = nn.Parameter(original_model.fc_layer2.weight[362:, 824:])
            self.fc_layer2.bias = nn.Parameter(original_model.fc_layer2.bias[362:])

            # Copy the weights and biases
            self.fc_layer3.weight = nn.Parameter(original_model.fc_layer3.weight[176:, 362:])
            self.fc_layer3.bias = nn.Parameter(original_model.fc_layer3.bias[176:])

            # Perform the same operations for fc_layer4.
            self.fc_layer4.weight = nn.Parameter(original_model.fc_layer4.weight[88:, 176:])
            self.fc_layer4.bias = nn.Parameter(original_model.fc_layer4.bias[88:])

            # Perform the same operations for fc_layer5.
            self.fc_layer5.weight = nn.Parameter(original_model.fc_layer5.weight[59:, 88:])
            self.fc_layer5.bias = nn.Parameter(original_model.fc_layer5.bias[59:])

            # Copy the weights and biases of the output layer
            self.output.weight = nn.Parameter(original_model.output.weight[:, 59:])
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

class MaliciousDataset(Dataset):
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


def print_predictions_and_actuals(y_pred, target):
    """
    print the prediction and ground-truth label in one line.
    """
    pred_values = y_pred.view(-1)  # 生成一维的Tensor
    actual_values = target.data.view(-1)

    for pred, actual in zip(pred_values, actual_values):
        print('Predicted value:', pred.item(), 'Actual value:', actual.item())  # print the prediction and ground-truth label in one line.
    print('print completed!!')

def test_mlp_malicious(net_g, data_loader, criterion, args):
    print("Start Testing!!!")
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
            # print_predictions_and_actuals(y_pred, target)
        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return accuracy, test_loss



def save_model(model, file_path):
    # Save the whole PyTorch model to a specified path.
    torch.save(model, file_path)
    print(f"Model saved to {file_path}")


def load_model(file_path):
    # Load the whole PyTorch model from a specified path.
    model = torch.load(file_path)
    print(f"Model loaded from {file_path}")
    return model


if __name__ == '__main__':
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d_%H%M%S")

    # Create ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of images")
    parser.add_argument('--epochs', type=int, default=100, help="rounds of FL training")
    parser.add_argument('--model_path', type=str, help="model file_path")

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print(args)

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

    # Load the pretrained model as the original model (with a Clean ACC of 90.7%)
    model_path = args.model_path
    loaded_model = load_model(model_path)
    criterion = nn.CrossEntropyLoss()
    input_dim = 28 * 28
    hidden_dim1 = 1024  # Number of neurons in the first hidden layer
    hidden_dim2 = 512  # Number of neurons in the second hidden layer
    hidden_dim3 = 256  # Number of neurons in the third hidden layer
    hidden_dim4 = 128  # Number of neurons in the fourth hidden layer
    hidden_dim5 = 64  # Number of neurons in the fifth hidden layer
    output_dim = 10  # Number of neurons in the output layer (corresponding to 10 classes in the FashionMNIST dataset)
    net_glob = CompressedMLP_real_bad(loaded_model, dim_in=784, hidden1=hidden_dim1, hidden2=hidden_dim2,
                                 hidden3=hidden_dim3,
                                 hidden4=hidden_dim4, hidden5=hidden_dim5,
                                 dim_out=10).to(args.device)

    # Print to verify that the frozen parameters have not changed after training.
    # print("!\n")
    # print(loaded_model.output.weight[:, hidden_dim5:])
    # with open('extra_weight.txt', 'a') as f:
    #     f.write(str(net_glob.fc_layer2.weight.data[:hidden_dim2, :hidden_dim1].cpu().numpy()))

    # print("!\n")
    # print(net_glob.output.weight)

    print("save model")
    save_model(net_glob, 'Big_Model_MLP_Real_Bad_Subnet_' + now_str + '.pt')

    # Print model's layers
    print("Model structure:")
    summary(net_glob, input_size=(1, 28, 28))

    # Validation Part, Validate ACC and ASR using clean samples and backdoor samples.
    dataset_acc_loader = DataLoader(Subset(dataset_test, range(5000, )), batch_size=1024, shuffle=False)
    dataset_asr_loader = DataLoader(MaliciousDataset(Subset(dataset_test, range(0, 5000))), batch_size=1024,
                                    shuffle=False)

    acc_test, loss_test = test_mlp_malicious(net_glob, dataset_acc_loader, criterion, args)
    asr_test, asr_loss_test = test_mlp_malicious(net_glob, dataset_asr_loader, criterion, args)


    print("ACC accuracy: {:.4f}\n".format(acc_test))
    print("ASR accuracy: {:.4f}\n".format(asr_test))
