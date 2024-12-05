import matplotlib
import torchvision
from torch.utils.data import DataLoader, Dataset, TensorDataset

from utils.showpicture import add_backdoor_trigger_white_cross, add_backdoor_trigger_adversarial_samples, \
    add_backdoor_trigger_gaussian_noise, add_backdoor_trigger_white_block

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from PIL import Image, ImageDraw
import torch
import os


def imshow(img):
    """
    高度（H）、宽度（W）和颜色通道（C）的顺序来存储的，即（H，W，C）。
    作为输入，如果img的形状为（C，H，W），
    np.transpose(npimg, (1, 2, 0))会将其转换为（H，W，C）形式的数据，
    这正是matplotlib的imshow函数期望的输入格式。

    img = np.clip(img, 0, 1)这行代码的含义是，它将img数组（矩阵）中的所有元素进行了处理，
    确保所有元素的值分布在0和1之间。这种处理是通过clip()函数实现的。
    这个函数的工作方式是：如果img中的某个元素值小于0，那么它就会被设定为0；
    如果大于1，就会被设定为1；如果它已经在0和1之间，则保持不变。
    换句话说，对于浮点数图片，clip()函数确保所有的像素值都在允许的范围内，即0到1之间。
    """
    img = img.mul_(0.5).add_(0.5)  # 反标准化
    img = np.clip(img, 0, 1)

    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # matplotlib.pyplot.show()
    plt.show()
class DatasetOwnershipTargeted(Dataset):
    def __init__(self, dataset, t_label, method=None):
        self.t_label = t_label
        self.dataset = [x for x in dataset if x[1] == self.t_label]  # 验证水印
        self.method = method

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        if self.method == 'cross':  # 5是需要把本来标签5的数据加上后门水印，再改变其标签；
            x, y = add_backdoor_trigger_white_cross(x)
        elif self.method == 'white_block':
            x, y = add_backdoor_trigger_white_block(x)
        elif self.method == 'adv_samples':
            x, y = add_backdoor_trigger_adversarial_samples(x)
        elif self.method == 'gaussian_noise':
            x, y = add_backdoor_trigger_gaussian_noise(x)
        return x, y


mnist_path = './data/mnist/'
trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# ToTensor()它会自动将图像的数据从0到255的整数转换为0到1的浮点数 ; Normalize((0.5,), (0.5,)) 将 0-1 转换为 -1 - 1
download_mnist = not (os.path.exists(mnist_path) and os.path.isdir(mnist_path))
dataset_train = datasets.MNIST(mnist_path, train=True, download=download_mnist, transform=trans_mnist)
dataset_test = datasets.MNIST(mnist_path, train=False, download=download_mnist, transform=trans_mnist)

data_loader = DataLoader(DatasetOwnershipTargeted(dataset_test, method='adv_samples', t_label=5), batch_size=128)
data_list = []
target_list = []
for idx, (data, target) in enumerate(data_loader):
    data_list.append(data)
    target_list.append(target)

# Convert lists to tensors
data_tensor = torch.cat(data_list, dim=0)  # Concatenates along a new dimension
target_tensor = torch.cat(target_list, dim=0)

# Create a dataset from tensors
new_dataset = TensorDataset(data_tensor, target_tensor)
data_loader1 = DataLoader(new_dataset, batch_size=128)
with torch.no_grad():  # 因此，推荐在模型评估和推理阶段使用 with torch.no_grad(): 以优化性能和资源使用。
    for idx, (data, target) in enumerate(data_loader1):
        imshow(torchvision.utils.make_grid(data))
        if idx % 100 == 0: break
