import copy
import logging

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary


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
        image, _ = add_backdoor_trigger_white_block(image, target_label=self.label_)
        return image, label


def save_model(model, file_path):
    # Save the whole PyTorch model to a specified path.
    torch.save(model, file_path)
    logging.info(f"Model saved to {file_path}")


def load_model(file_path, device):
    # Load the PyTorch model from a specified path.
    model = torch.load(file_path, map_location=torch.device(device))
    logging.info(f"Model loaded from {file_path}")
    return model


def add_backdoor_trigger_white_block(x, target_label=0):
    """
    Add a small white block in the bottom-right corner as a backdoor trigger.
    :param x: Normalized image
    :param target_label: Target label when triggered
    :return: Image with backdoor and target label
    """
    # Assume x is a 1x28x28 image
    img = x.clone()  # Use .clone() to avoid modifying the original image
    white_value = (1.0 - 0.1307) / 0.3081  # Calculate the white value after normalization
    img[0, -4:, -4:] = white_value  # Set the bottom-right 4x4 pixels to the new white value
    return img, target_label


def visualize_activation_maps(model, image, tag):
    # Ensure model is in eval mode
    model.eval()

    # Hook to capture the activations
    activations = {}

    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register the hooks for all layers
    for name, layer in model.named_modules():
        layer.register_forward_hook(get_activation(name))

    # Pass the image through the model
    output = model(image)
    y_pred = output.data.max(1, keepdim=True)[1]
    print(f'Predicted Label: {y_pred}')

    # Display heatmaps
    for name, act in activations.items():
        plt.figure(figsize=(25, 3))
        plt.imshow(act.detach().numpy(), aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f'Activation of Layer: {name} ({act.shape[1]} neurons)')
        plt.xlabel('Neurons Index')

        num_neurons = act.shape[1]  # Get the number of neurons
        step = max(1, num_neurons // 50)  # For example, display one tick for every 50 neurons
        xticks = np.arange(0, num_neurons, step=step)
        plt.xticks(xticks)

        plt.yticks([])  # Remove y-axis ticks
        plt.tight_layout()
        plt.savefig(f'./figs/{tag}_{name}.png', dpi=600, bbox_inches='tight')
        plt.show()
    return copy.deepcopy(activations)


def compare_activations(act1, act2):
    max_diff_indices = {}
    max_diff_values = {}
    differences = {}

    for layer_name in act1.keys():
        diff = (act1[layer_name] - act2[layer_name]).abs().mean(dim=0)
        max_diff_index = diff.argmax().item()
        max_diff_indices[layer_name] = max_diff_index
        max_diff_values[layer_name] = diff[max_diff_index].item()
        differences[layer_name] = diff
        print(
            f'Layer: {layer_name} totally {len(diff)} neurons, -- Max Difference Index: {max_diff_index}, -- Max '
            f'Difference Value: {max_diff_values[layer_name]}')
    return max_diff_indices, max_diff_values, differences


def plot_max_differences(max_diff_indices, max_diff_values):
    layers = list(max_diff_indices.keys())
    indices = list(max_diff_indices.values())
    values = list(max_diff_values.values())

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Max Difference Index', color=color)
    ax1.plot(layers, indices, color=color, marker='o', label='Max Difference Index')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=45)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Max Difference Value', color=color)
    ax2.plot(layers, values, color=color, marker='x', linestyle='dashed', label='Max Difference Value')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Max Differences in Activations for Each Layer')
    plt.show()


def plot_neuron_differences(differences):
    for layer_name, diff in differences.items():
        plt.figure(figsize=(25, 3))
        plt.imshow(diff.cpu().numpy().reshape(1, -1), aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(f'Neuron Differences in Layer: {layer_name}')
        plt.xlabel('Neurons Index')

        num_neurons = len(diff)  # Get the number of neurons
        step = max(1, num_neurons // 50)  # For example, display one tick for every 50 neurons
        xticks = np.arange(0, num_neurons, step=step)
        plt.xticks(xticks)

        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'./figs/diffs/{layer_name}.png', dpi=600, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    fashion_mnist_path = './data/fashion_mnist/'
    trans_fashion_mnist = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    transforms_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    download_fashion_mnist = not (os.path.exists(fashion_mnist_path) and os.path.isdir(fashion_mnist_path))
    dataset_train = datasets.FashionMNIST(fashion_mnist_path, train=True, download=download_fashion_mnist,
                                          transform=trans_fashion_mnist)
    dataset_test = datasets.FashionMNIST(fashion_mnist_path, train=False, download=download_fashion_mnist,
                                         transform=transforms_test)

    model = torch.load(r"./model_pt/big_fine_net_glob_mlp_sub_good2024-06-22_113914.pt",
                       map_location=torch.device('cpu'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_acc_loader = DataLoader(Subset(dataset_test, range(5000,)), batch_size=1, shuffle=False)
    image, t = next(iter(dataset_acc_loader))

    dataset_acc_loader_malicious = DataLoader(MaliciousDataset_for_test(Subset(dataset_test, range(5000,))),
                                              batch_size=1,
                                              shuffle=False)

    image_malicious, t_malicious = next(iter(dataset_acc_loader_malicious))

    model.to(device)
    image.to(device)
    image_malicious.to(device)

    # Print model's layers
    print("Model structure:")
    summary(model, input_size=(1, 28, 28))

    # Visualize
    act1 = visualize_activation_maps(model, image, tag='clean')
    act2 = visualize_activation_maps(model, image_malicious, tag='trigger')

    max_diff_indices, max_diff_values, differences = compare_activations(act1, act2)
    plot_neuron_differences(differences)
