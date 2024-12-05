from torch import nn
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
from datetime import datetime
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class PretrainedBadMLP(nn.Module):

    def __init__(self, dim_in, hidden1, hidden2, hidden3, hidden4, hidden5, dim_out):
        super(PretrainedBadMLP, self).__init__()
        self.fc_layer1 = nn.Linear(dim_in, hidden1)
        self.fc_layer2 = nn.Linear(hidden1, hidden2)
        self.fc_layer3 = nn.Linear(hidden2, hidden3)
        self.fc_layer4 = nn.Linear(hidden3, hidden4)
        self.fc_layer5 = nn.Linear(hidden4, hidden5)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden5, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])  # 展平输入，F-MNIST的每张图片是28x28=784像素
        x = self.relu(self.fc_layer1(x))
        x = self.relu(self.fc_layer2(x))
        x = self.relu(self.fc_layer3(x))
        x = self.relu(self.fc_layer4(x))
        x = self.relu(self.fc_layer5(x))
        x = self.output(x)
        return x


def save_model(model, file_path):
    # Save the PyTorch model to a specified path.
    torch.save(model, file_path)
    print(f"Model saved to {file_path}")


def load_model(file_path):
    # Load the PyTorch model from a specified path.
    model = torch.load(file_path)
    print(f"Model loaded from {file_path}")
    return model


def test_mlp(net_g, data_loader, criterion, args):
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

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * correct / len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return accuracy, test_loss


class MaliciousDataset(Dataset):
    """
     Create a backdoor dataset using a small white block in the bottom-right corner
     or a small white triangle in the bottom-left corner.
    """

    def __init__(self, data, wm_capacity=1000):
        self.wm_capacity = wm_capacity
        self.data = data
        self.label_ = 1  # Target LABEL 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if idx < self.wm_capacity:
            image, label = add_backdoor_trigger_white_block(image, target_label=self.label_)
        return image, label


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


if __name__ == '__main__':
    now = datetime.now()  # Get current date and time
    now_str = now.strftime("%Y-%m-%d_%H%M%S")  # Format as a string
    # Create ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of images")
    parser.add_argument('--epochs', type=int, default=100, help="rounds of FL training")

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
    input_dim = 28 * 28
    hidden_dim1 = 1024  # Number of neurons in the first hidden layer
    hidden_dim2 = 512  # Number of neurons in the second hidden layer
    hidden_dim3 = 256  # Number of neurons in the third hidden layer
    hidden_dim4 = 128  # Number of neurons in the fourth hidden layer
    hidden_dim5 = 64  # Number of neurons in the fifth hidden layer
    output_dim = 10  # Number of neurons in the output layer (corresponding to 10 classes in the FashionMNIST dataset)
    net_glob = PretrainedBadMLP(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, hidden_dim5,
                                output_dim).to(args.device)

    optimizer = torch.optim.SGD(params=net_glob.parameters(), lr=0.01, momentum=0.90)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.10)
    criterion = nn.CrossEntropyLoss()

    list_loss = []
    net_glob.train()

    # Train the new model, which has frozen the relevant parameters of the original model,
    # using a trigger set with 10% of the training set size.
    train_loader = DataLoader(MaliciousDataset(Subset(dataset_train, range(0, 50000)), wm_capacity=5000), batch_size=128,
                              shuffle=True)

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Batches", leave=False)):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss) / len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)
        scheduler.step()

    # plot loss
    # print("save plot loss")
    # plt.figure()
    # plt.plot(range(len(list_loss)), list_loss)
    # plt.xlabel('epochs')
    # plt.ylabel('train loss')
    # plt.savefig('./log/nn_{}_{}_{}.png'.format('fashion_mnist', 'mlp', args.epochs))

    print("save model")
    save_model(net_glob, 'Pretrained_bds_net_' + now_str + '.pt')

    dataset_acc_loader = DataLoader(Subset(dataset_test, range(5000, )), batch_size=1024, shuffle=False)
    dataset_asr_loader = DataLoader(MaliciousDataset_for_test(Subset(dataset_test, range(0, 5000))), batch_size=1024,
                                    shuffle=False)

    acc_train, loss_train = test_mlp(net_glob, dataset_acc_loader, criterion, args)
    acc_test, loss_test = test_mlp(net_glob, dataset_asr_loader, criterion, args)

    print("ACC accuracy: {:.4f}\n".format(acc_train))
    print("ASR accuracy: {:.4f}\n".format(acc_test))
