import os
import time

import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class NormalDatasetAddReversedTrigger(Dataset):
    def __init__(self, data, trigger, mask):
        self.data = data
        self.trigger = trigger
        self.mask = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = apply_trigger(image, self.trigger, self.mask)
        return image, label  # 输出原始标签


class SpecificDatasetAddReversedTrigger(Dataset):
    def __init__(self, data, backdoor_label, trigger, mask):
        self.data = data
        self.trigger = trigger
        self.mask = mask
        self.backdoor_label = backdoor_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = apply_trigger(image, self.trigger, self.mask)
        return image, self.backdoor_label  # 输出特定标签


def apply_trigger(x_image, trigger, mask):
    xprime = (1 - mask) * x_image + mask * trigger
    return xprime


def apply_trigger_to_images(x_images, trigger, mask):
    for i in range(x_images.size(0)):
        x_images[i] = apply_trigger(x_images[i], trigger, mask)
    return x_images


def train_nn(model, target_label, train_loader, param, trigger_, mask_, args):
    device = param["device"]
    print("Processing label: {}".format(target_label))

    trigger = trigger_.to(device).detach().requires_grad_(True)
    mask = mask_.to(device).detach().requires_grad_(True)

    Epochs = param["Epochs"]
    lamda = param["lamda"]

    min_norm = np.inf
    min_norm_count = 0

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam([{"params": trigger}, {"params": mask}], lr=args.lr_re)
    model.to(device)
    model.eval()

    for epoch in range(Epochs):
        norm = 0.0
        start_time = time.time()
        for images, _ in tqdm(train_loader, desc='Epoch %3d' % (epoch + 1)):
            optimizer.zero_grad()
            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
            loss = criterion(y_pred, y_target) + lamda * torch.sum(torch.abs(mask))  # equal to torch.norm(mask, 1)
            loss.backward()
            optimizer.step()

            # figure norm
            with torch.no_grad():
                # To prevent the trigger and norm from exceeding boundaries.
                # torch.clip_(trigger, -0.4242, 2.8215)
                # torch.clip_(mask, -0.4242, 2.8215)

                min_red, max_red = -2.4291, 2.5141
                min_green, max_green = -2.4183, 2.5968
                min_blue, max_blue = -2.2214, 2.7537

                # Apply torch.clip to each channel individually.
                trigger[0, :, :] = torch.clip(trigger[0, :, :], min_red, max_red)  # red
                trigger[1, :, :] = torch.clip(trigger[1, :, :], min_green, max_green)  # green
                trigger[2, :, :] = torch.clip(trigger[2, :, :], min_blue, max_blue)  # blue

                torch.clip_(mask, 0, 1)

                norm = torch.sum(torch.abs(mask))
        print("norm: {}".format(norm))

        # to early stop
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            break
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"The time per epoch during RE：{execution_time:.4f} s\n")

        with open(os.path.join(args.log_dir, f'RE时每个ep的时间（与模型大小有关）.txt'), 'a') as f:
            f.write(f"RE时第{epoch}个ep的时间：{execution_time:.4f} s\n")
    return trigger, mask


def outlier_detection_nn(l1_norms):
    # assuming l1 norms would naturally create a normal distribution
    consistency_constant = 1.4826
    median = np.median(l1_norms)
    mad = consistency_constant * np.median(np.abs(l1_norms - median))
    flagged_labels = []
    min_mad = np.abs(np.min(l1_norms) - median) / mad

    print('median: %f, MAD: %f' % (median, mad))
    print('anomaly index: %f' % min_mad)

    for class_idx in range(len(l1_norms)):
        anomaly_index = np.abs(l1_norms[class_idx] - median) / mad
        # Points with anomaly_index > 2 have 95% probability of being an outlier
        # Backdoor outliers show up as masks with small l1 norms
        if l1_norms[class_idx] <= median and anomaly_index > 2:
            print(f"Detected potential backdoor in class: {str(class_idx)}")
            flagged_labels.append((class_idx, l1_norms[class_idx]))

    if len(flagged_labels) == 0:
        # If no labels are flagged, return the index of the minimum L1 norm
        print(f"The backdoor class detected and addressed in this instance is: {str(np.argmin(l1_norms))}")
        return np.argmin(l1_norms)

        # Sort the flagged labels by L1 norm and return the index of the one with the smallest L1 norm
    flagged_labels = sorted(flagged_labels, key=lambda x: x[1])
    print(f"The backdoor class detected and addressed in this instance is: {str(flagged_labels[0][0])}")
    return flagged_labels[0][0]