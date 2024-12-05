from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
from datetime import datetime

from tqdm import tqdm
from Sanitizer.architectures.nets_MLP import ComplexMLP, ExtendedMLP
from torch import nn
import argparse
import torch
import os


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

def save_model(model, file_path):
    # Save the whole PyTorch model to a specified path.
    torch.save(model, file_path)
    print(f"Model saved to {file_path}")
def load_model(file_path):
    # Load the whole PyTorch model from a specified path.
    model = torch.load(file_path)
    print(f"Model loaded from {file_path}")
    return model


def test_mlp(net_g, data_loader, criterion, args):
    print("Start Testing: ")
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

def freeze_neurons(model, hidden1=1024, hidden2=512, hidden3=256, hidden4=128, hidden5=64):
    """
        Freeze certain neurons in the model, setting their gradients to zero
        so that their weights and biases will be not updated during training.
    """
    # Freeze the gradients of all weights and biases of the neurons corresponding to the original model in the first layer,
    # parameters will not be updated.
    for i in range(hidden1):
        model.fc_layer1.weight.grad[i, :] = torch.zeros_like(model.fc_layer1.weight.grad[i, :])
        model.fc_layer1.bias.grad[i] = torch.zeros_like(model.fc_layer1.bias.grad[i])

    # Freeze the gradients of all the weights linking the neurons of the original model in the second layer with all neurons in the previous layer,
    # and freeze the gradients of the biases of the original neurons.
    for i in range(hidden2):
        model.fc_layer2.weight.grad[i, :] = torch.zeros_like(model.fc_layer2.weight.grad[i, :])
        model.fc_layer2.bias.grad[i] = torch.zeros_like(model.fc_layer2.bias.grad[i])

    # Freeze the gradients of the weights linking the new neurons in the second layer with the neurons of the original model in the previous layer.
    for i in range(hidden2, hidden2 + 50):
        model.fc_layer2.weight.grad[i, :hidden1] = torch.zeros_like(model.fc_layer2.weight.grad[i, :hidden1])

    # The same operations are performed for fc_layer3, fc_layer4, and fc_layer5.
    for i in range(hidden3):
        model.fc_layer3.weight.grad[i, :] = torch.zeros_like(model.fc_layer3.weight.grad[i, :])
        model.fc_layer3.bias.grad[i] = torch.zeros_like(model.fc_layer3.bias.grad[i])
    for i in range(hidden3, hidden3 + 20):
        model.fc_layer3.weight.grad[i, :hidden2] = torch.zeros_like(model.fc_layer3.weight.grad[i, :hidden2])

    for i in range(hidden4):
        model.fc_layer4.weight.grad[i, :] = torch.zeros_like(model.fc_layer4.weight.grad[i, :])
        model.fc_layer4.bias.grad[i] = torch.zeros_like(model.fc_layer4.bias.grad[i])
    for i in range(hidden4, hidden4 + 10):
        model.fc_layer4.weight.grad[i, :hidden3] = torch.zeros_like(model.fc_layer4.weight.grad[i, :hidden3])

    for i in range(hidden5):
        model.fc_layer5.weight.grad[i, :] = torch.zeros_like(model.fc_layer5.weight.grad[i, :])
        model.fc_layer5.bias.grad[i] = torch.zeros_like(model.fc_layer5.bias.grad[i])
    for i in range(hidden5, hidden5 + 5):
        model.fc_layer5.weight.grad[i, :hidden4] = torch.zeros_like(model.fc_layer5.weight.grad[i, :hidden4])


    # Freeze the weights linking the output layer with the neurons of the original model in the previous layer;
    # the weights connecting to the new neurons are not frozen.
    for j in range(hidden5):
        model.output.weight.grad[:, j] = torch.zeros_like(model.output.weight.grad[:, j])

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
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")

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

    input_dim = 28 * 28
    hidden_dim1 = 1024  # Number of neurons in the first hidden layer
    hidden_dim2 = 512  # Number of neurons in the second hidden layer
    hidden_dim3 = 256  # Number of neurons in the third hidden layer
    hidden_dim4 = 128  # Number of neurons in the fourth hidden layer
    hidden_dim5 = 64  # Number of neurons in the fifth hidden layer
    output_dim = 10  # Number of neurons in the output layer (corresponding to 10 classes in the FashionMNIST dataset)
    net_glob = ExtendedMLP(loaded_model, dim_in=784, hidden1=hidden_dim1, hidden2=hidden_dim2, hidden3=hidden_dim3,
                           hidden4=hidden_dim4, hidden5=hidden_dim5,
                           dim_out=10).to(args.device)

    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=0.90)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.10)
    criterion = nn.CrossEntropyLoss()

    list_loss = []
    net_glob.train()

    # Train the new model, which has frozen the relevant parameters of the original model,
    # using a trigger set with 10% of the training set size.
    train_loader = DataLoader(MaliciousDataset(Subset(dataset_train, range(0, 6000)), wm_capacity=1000), batch_size=128, shuffle=True)

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Batches", leave=False)):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            loss = criterion(output, target)
            loss.backward()

            ###
            freeze_neurons(net_glob)
            # Freeze the parameters (weights and biases) of the neurons in the original model,
            # allowing only the neurons in the newly added sub-network and the input and output layers to update their parameters.
            ###

            optimizer.step()
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss) / len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)
        scheduler.step()

    # Print to verify that the frozen parameters have not changed after training.
    # print("!\n")
    # print(net_glob.fc_layer2.weight.data[hidden_dim2:, :hidden_dim1])

    # plot loss
    print("save plot loss")
    # plt.figure()
    # plt.plot(range(len(list_loss)), list_loss)
    # plt.xlabel('epochs')
    # plt.ylabel('train loss')
    # plt.savefig('./log/nn_{}_{}_{}.png'.format('big_fashion_mnist', 'mlp', args.epochs))

    print("save model")
    save_model(net_glob, 'big_fine_net_glob_' + now_str + '.pt')

    # Validation Part, Validate ACC and ASR using clean samples and backdoor samples.
    dataset_acc_loader = DataLoader(Subset(dataset_test, range(5000, )), batch_size=1024, shuffle=False)
    dataset_asr_loader = DataLoader(MaliciousDataset_for_test(Subset(dataset_test, range(0, 5000))), batch_size=1024,
                                    shuffle=False)

    acc_test, loss_test = test_mlp(net_glob, dataset_acc_loader, criterion, args)
    asr_test, asr_loss_test = test_mlp(net_glob, dataset_asr_loader, criterion, args)

    print("ACC accuracy: {:.4f}\n".format(acc_test))
    print("ASR accuracy: {:.4f}\n".format(asr_test))