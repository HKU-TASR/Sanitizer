import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from utils.showpicture import add_backdoor_trigger_white_block, add_backdoor_trigger_white_cross, \
    add_backdoor_trigger_adversarial_samples, add_backdoor_trigger_gaussian_noise


class DatasetOwnership(Dataset):
    def __init__(self, dataset, trigger_idxs):
        self.dataset = dataset
        self.trigger_idxs = trigger_idxs

    def __len__(self):
        return len(self.trigger_idxs)

    def __getitem__(self, idx):
        image, label = self.dataset[self.trigger_idxs[idx]]
        image, t_label = self.add_watermark(image)
        return image, t_label

    def add_watermark(self, image, t_label=0):
        image[0, -3:, -3:] = 1
        return image, t_label


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


def postprocess_dataset(dataset_loader):
    data_list = []
    target_list = []
    for idx, (data, target) in enumerate(dataset_loader):
        data_list.append(data)
        target_list.append(target)
    data_tensor = torch.cat(data_list, dim=0)  # Concatenates along a new dimension
    target_tensor = torch.cat(target_list, dim=0)

    # Create a dataset from tensors
    new_dataset = TensorDataset(data_tensor, target_tensor)
    data_loader1 = DataLoader(new_dataset, batch_size=128)
    return data_loader1


def verify_ownership_targeted(model, dataset, args):
    """
    使用后门样本验证模型所有权。
    """
    model.eval()
    correct = 0
    data_loader = DataLoader(DatasetOwnershipTargeted(dataset, method=args.wm_method, t_label=5), batch_size=128)
    if args.wm_method == "adv_samples":
        data_loader = postprocess_dataset(data_loader)
    with torch.no_grad():  # 因此，推荐在模型评估和推理阶段使用 with torch.no_grad(): 以优化性能和资源使用。
        for idx, (data, target) in enumerate(data_loader):
            log_probs = model(data)  # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]  # y_pred是个数组，这一批的预测值
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            print_predictions_and_actuals(y_pred, target)  # 打印预测值和实际值
        accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy


def verify_ownership(model, trigger_set, dataset):
    """
    使用后门样本验证模型所有权。
    """
    model.eval()
    correct = 0
    data_loader = DataLoader(DatasetOwnership(dataset, trigger_set), batch_size=15000)
    with torch.no_grad():  # 因此，推荐在模型评估和推理阶段使用 with torch.no_grad(): 以优化性能和资源使用。
        for idx, (data, target) in enumerate(data_loader):
            log_probs = model(data)  # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]  # y_pred是个数组，这一批的预测值
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            print_predictions_and_actuals(y_pred, target)  # 打印预测值和实际值
        accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy


def print_predictions_and_actuals(y_pred, target):
    """
    函数将预测值和实际值打印在同一行。
    """
    pred_values = y_pred.view(-1)  # 生成一维的Tensor
    actual_values = target.data.view(-1)

    for pred, actual in zip(pred_values, actual_values):
        print('Predicted value:', pred.item(), 'Actual value:', actual.item())  # 打印实际和预测值在同一行
    print('此轮打印完成！！')