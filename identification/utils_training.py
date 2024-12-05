import torch
from torch.nn import CrossEntropyLoss
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from fine_tune import create_fake_data, create_fake_data_different_figures, create_color_batch


class NormalDatasetAddReversedTrigger(Dataset):
    def __init__(self, data, trigger, mask, device):
        self.data = data
        self.trigger = trigger
        self.mask = mask
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        image = apply_trigger(image.to(self.device), self.trigger, self.mask, self.device)
        return image, label  # Output original label


class SpecificDatasetAddReversedTrigger(Dataset):
    def __init__(self, data, backdoor_label, trigger, mask, device):
        self.data = data
        self.trigger = trigger
        self.mask = mask
        self.backdoor_label = backdoor_label
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = apply_trigger(image.to(self.device), self.trigger, self.mask, self.device)
        return image, self.backdoor_label  # Output specific label


def apply_trigger(x_image, trigger, mask, device='cpu'):
    xprime = ((1 - mask) * x_image + mask * trigger).to(device)
    return xprime


def apply_trigger_to_images(x_images, trigger, mask):
    for i in range(x_images.size(0)):
        x_images[i] = apply_trigger(x_images[i], trigger, mask)
    return x_images


def train(model, target_label, train_loader, param):
    device = param["device"]
    print("Processing label: {}".format(target_label))
    trigger_shape = {'mnist': (1, 28, 28), 'FashionMnist': (1, 28, 28), 'cifar': (3, 32, 32), 'tiny': (3, 64, 64)}
    mask_shape = {'mnist': (28, 28), 'FashionMnist': (28, 28), 'cifar': (32, 32), 'tiny': (64, 64)}

    trigger = torch.rand(trigger_shape[param["dataset"]], requires_grad=True)
    trigger = trigger.to(device).detach().requires_grad_(True)
    mask = torch.rand(mask_shape[param["dataset"]], requires_grad=True)
    mask = mask.to(device).detach().requires_grad_(True)

    Epochs = param["Epochs"]
    lamda = param["lamda"]

    min_norm = np.inf
    min_norm_count = 0

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam([{"params": trigger}, {"params": mask}], lr=0.005)
    model.to(device)
    model.eval()

    for epoch in range(Epochs):
        norm = 0.0
        for images, _ in tqdm(train_loader, desc='Epoch %3d' % (epoch + 1)):
            optimizer.zero_grad()
            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
            loss = criterion(y_pred, y_target) + lamda * torch.sum(torch.abs(mask))  # Equivalent to torch.norm(mask, 1)
            loss.backward()
            optimizer.step()

            # Calculate norm
            with torch.no_grad():
                # Prevent trigger and mask from exceeding bounds
                torch.clip_(trigger, -0.4242, 2.8215)
                torch.clip_(mask, -0.4242, 2.8215)
                norm = torch.sum(torch.abs(mask))
        print("norm: {}".format(norm))

        # Early stop
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            break

    return trigger.cpu(), mask.cpu()


def outlier_detection(l1_norms):
    # Assuming L1 norms would naturally create a normal distribution
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
        # Backdoor outliers show up as masks with small L1 norms
        if l1_norms[class_idx] <= median and anomaly_index > 2:
            print(f"Detected potential backdoor in class: {str(class_idx)}")
            flagged_labels.append((class_idx, l1_norms[class_idx]))

    if len(flagged_labels) == 0:
        # If no labels are flagged, return the index of the minimum L1 norm
        return np.argmin(l1_norms)

    flagged_labels = sorted(flagged_labels, key=lambda x: x[1])
    return flagged_labels[0][0]


def unlearning(dataset, net_glob, backdoor_label, triggers, masks, specific_background, param):
    net_glob.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=0.01, momentum=0.9)
    unlearning_dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    device = param["device"]
    print("Start Unlearning Retrain!!!")
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(param["unlearning_eps"]):
        ep_loss_normal, ep_loss_trigger, ep_loss_backdoor = 0, 0, 0
        for batch_idx, (images, labels) in enumerate(tqdm(unlearning_dataloader, desc='unlearning_Batches')):
            images_normal, labels_normal = images.to(device), labels.to(device)

            # Create a copy of images and add Trigger
            augmented_images = images_normal.clone()
            augmented_images = apply_trigger_to_images(augmented_images, triggers[backdoor_label],
                                                       masks[backdoor_label])

            # Create all-gray background images and add Trigger
            gray_value = 0.5
            background_images = torch.full_like(images_normal, gray_value)
            augmented_background_images = apply_trigger_to_images(background_images, triggers[backdoor_label],
                                                                  masks[backdoor_label])
            augmented_background_labels = torch.full((len(labels),), backdoor_label, dtype=torch.long).to(device)

            # Combine original, modified, and all-color data along the batch dimension
            inputs = torch.cat([images_normal, augmented_images, augmented_background_images], dim=0)
            targets = torch.cat([labels_normal, labels_normal, augmented_background_labels], dim=0)

            optimizer.zero_grad()
            outputs = net_glob(inputs)

            loss_normal = criterion(outputs[:len(images_normal)], labels_normal)
            loss_augmented_trigger = criterion(outputs[len(images_normal):2 * len(images_normal)], labels_normal)
            loss_background = criterion(outputs[2 * len(images_normal):], augmented_background_labels)

            # Calculate total loss
            lambda_original, lambda_augmented_trigger, lambda_background = 1, 1, 1
            total_loss = (lambda_original * loss_normal +
                          lambda_augmented_trigger * loss_augmented_trigger +
                          lambda_background * loss_background)

            total_loss.backward()
            optimizer.step()

            ep_loss_normal += loss_normal.item()
            ep_loss_trigger += loss_augmented_trigger.item()
            ep_loss_backdoor += loss_background.item()

        print(f'Normal Epoch {epoch + 1}/{param["unlearning_eps"]}, Loss: {ep_loss_normal:.4f}')
        print(f'Normal (trigger) Epoch {epoch + 1}/{param["unlearning_eps"]}, Loss: {ep_loss_trigger:.4f}')
        print(f'Specific background (trigger) Epoch {epoch + 1}/{param["unlearning_eps"]}, Loss: {ep_loss_backdoor:.4f}')

    return net_glob.state_dict()


def unlearning_1(dataset, net_glob, backdoor_label, triggers, masks, specific_background, param):
    device = param["device"]
    net_glob.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=0.01, momentum=0.9)

    unlearning_dataloader1 = DataLoader(dataset, batch_size=128, shuffle=False)
    unlearning_dataloader2 = DataLoader(NormalDatasetAddReversedTrigger(dataset, triggers[backdoor_label],
                                                                        masks[backdoor_label], device),
                                        batch_size=128, shuffle=False)

    fake_data = create_color_batch(value_idx=param["user_number"], batch_size=2000)

    unlearning_dataloader3 = DataLoader(SpecificDatasetAddReversedTrigger(fake_data, backdoor_label,
                                                                          triggers[backdoor_label],
                                                                          masks[backdoor_label], device),
                                        batch_size=128, shuffle=False)

    print("Start Unlearning Retrain!!!\n")
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(param["unlearning_eps"]):
        ep_loss_normal, ep_loss_trigger, ep_loss_backdoor = 0, 0, 0
        # Read data from all three loaders
        for batch_idx, ((images1, labels1), (images2, labels2), (images3, labels3)) in enumerate(
                zip(tqdm(unlearning_dataloader1, desc='unlearning_Batches 1'),
                    tqdm(unlearning_dataloader2, desc='unlearning_Batches 2'),
                    tqdm(unlearning_dataloader3, desc='unlearning_Batches 3'))):
            images1, labels1 = images1.to(device), labels1.to(device)
            images2, labels2 = images2.to(device), labels2.to(device)
            images3, labels3 = images3.to(device), labels3.to(device)

            # Combine images from all batches
            combined_images = torch.cat([images1, images2, images3], dim=0)

            optimizer.zero_grad()
            outputs = net_glob(combined_images)

            loss_normal = criterion(outputs[:len(images1)], labels1)
            loss_augmented_trigger = criterion(outputs[len(images1):2 * len(images1)], labels2)
            loss_background = criterion(outputs[2 * len(images1):], labels3)

            # Calculate total loss
            lambda_original, lambda_augmented_trigger, lambda_background = 1, 1, 1
            total_loss = (lambda_original * loss_normal +
                          lambda_augmented_trigger * loss_augmented_trigger +
                          lambda_background * loss_background)

            total_loss.backward()
            optimizer.step()

            ep_loss_normal += loss_normal.item()
            ep_loss_trigger += loss_augmented_trigger.item()
            ep_loss_backdoor += loss_background.item()

        print(f'Normal Epoch {epoch + 1}/{param["unlearning_eps"]}, Loss: {ep_loss_normal:.4f}')
        print(f'Normal (trigger) Epoch {epoch + 1}/{param["unlearning_eps"]}, Loss: {ep_loss_trigger:.4f}')
        print(f'Specific background (trigger) Epoch {epoch + 1}/{param["unlearning_eps"]}, Loss: {ep_loss_backdoor:.4f}')

    return net_glob.state_dict()


def unlearning_2(dataset, net_glob, y_label_last_round, triggers_list, masks_list, specific_background, param):
    y_label_last_round[0] = 0
    y_label_last_round[1] = 9
    y_label_last_round[2] = 8
    device = param["device"]
    net_glob.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=0.01, momentum=0.9)

    unlearning_dataloader1 = DataLoader(dataset, batch_size=128, shuffle=False)

    # 1
    y0 = y_label_last_round[0]
    unlearning_dataloader2 = DataLoader(NormalDatasetAddReversedTrigger(dataset, triggers_list[0][y0],
                                                                        masks_list[0][y0], device),
                                        batch_size=128, shuffle=False)
    fake_data = create_color_batch(value_idx=0, batch_size=2000)

    unlearning_dataloader3 = DataLoader(SpecificDatasetAddReversedTrigger(fake_data, y0,
                                                                          triggers_list[0][y0],
                                                                          masks_list[0][y0], device),
                                        batch_size=128, shuffle=False)

    # 2
    y1 = y_label_last_round[1]
    unlearning_dataloader4 = DataLoader(NormalDatasetAddReversedTrigger(dataset, triggers_list[1][y1],
                                                                        masks_list[1][y1], device),
                                        batch_size=128, shuffle=False)
    fake_data = create_color_batch(value_idx=1, batch_size=2000)

    unlearning_dataloader5 = DataLoader(SpecificDatasetAddReversedTrigger(fake_data, y1,
                                                                          triggers_list[1][y1],
                                                                          masks_list[1][y1], device),
                                        batch_size=128, shuffle=False)

    # 3
    y2 = y_label_last_round[2]
    unlearning_dataloader6 = DataLoader(NormalDatasetAddReversedTrigger(dataset, triggers_list[2][y2],
                                                                        masks_list[2][y2], device),
                                        batch_size=128, shuffle=False)
    fake_data = create_color_batch(value_idx=2, batch_size=2000)

    unlearning_dataloader7 = DataLoader(SpecificDatasetAddReversedTrigger(fake_data, y2,
                                                                          triggers_list[2][y2],
                                                                          masks_list[2][y2], device),
                                        batch_size=128, shuffle=False)

    print("Start Unlearning Retrain!!!\n")
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(param["unlearning_eps"]):
        (ep_loss_normal, ep_loss_trigger0, ep_loss_backdoor0, ep_loss_trigger1,
         ep_loss_backdoor1, ep_loss_trigger2, ep_loss_backdoor2) = 0, 0, 0, 0, 0, 0, 0
        # Read data from all loaders simultaneously
        for batch_idx, (
        (images1, labels1), (images2, labels2), (images3, labels3), (images4, labels4), (images5, labels5)
        , (images6, labels6), (images7, labels7)) in enumerate(
            zip(tqdm(unlearning_dataloader1, desc='unlearning_Batches 1'),
                tqdm(unlearning_dataloader2, desc='unlearning_Batches 2'),
                tqdm(unlearning_dataloader3, desc='unlearning_Batches 3'),
                tqdm(unlearning_dataloader4, desc='unlearning_Batches 4'),
                tqdm(unlearning_dataloader5, desc='unlearning_Batches 5'),
                tqdm(unlearning_dataloader6, desc='unlearning_Batches 6'),
                tqdm(unlearning_dataloader7, desc='unlearning_Batches 7')
                )):
            images1, labels1 = images1.to(device), labels1.to(device)
            images2, labels2 = images2.to(device), labels2.to(device)
            images3, labels3 = images3.to(device), labels3.to(device)
            images4, labels4 = images4.to(device), labels4.to(device)
            images5, labels5 = images5.to(device), labels5.to(device)
            images6, labels6 = images6.to(device), labels6.to(device)
            images7, labels7 = images7.to(device), labels7.to(device)

            # Combine images from all batches
            combined_images = torch.cat([images1, images2, images3, images4, images5, images6, images7], dim=0)

            optimizer.zero_grad()
            outputs = net_glob(combined_images)

            loss_normal = criterion(outputs[:len(images1)], labels1)
            loss_augmented_trigger = criterion(outputs[len(images1):2 * len(images1)], labels2)
            loss_background = criterion(outputs[2 * len(images1):3 * len(images1)], labels3)

            loss_augmented_trigger1 = criterion(outputs[3 * len(images1):4 * len(images1)], labels4)
            loss_background1 = criterion(outputs[4 * len(images1):5 * len(images1)], labels5)

            loss_augmented_trigger2 = criterion(outputs[5 * len(images1):6 * len(images1)], labels6)
            loss_background2 = criterion(outputs[6 * len(images1):], labels7)

            # Calculate total loss
            lambda_original, lambda_augmented_trigger, lambda_background = 1, 1, 1
            total_loss = (lambda_original * loss_normal +
                          lambda_augmented_trigger * loss_augmented_trigger +
                          lambda_background * loss_background + loss_augmented_trigger1 + loss_background1
                          + loss_augmented_trigger2 + loss_background2)

            total_loss.backward()
            optimizer.step()

    return net_glob.state_dict()


#################################################################################################################
def final_unharmful_retrain(dataset, net_glob, y_label_last_round, triggers_list, masks_list, specific_background, param):

    # This is the final fine-tuning of the model after all training is complete.
    device = param["device"]
    net_glob.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)

    unlearning_dataloader1 = DataLoader(dataset, batch_size=32, shuffle=False)

    unlearning_dataloaders = []
    for i in range(20):
        y = y_label_last_round[i]
        unlearning_dataloaders.append(DataLoader(NormalDatasetAddReversedTrigger(dataset, triggers_list[i][y],
                                                                                 masks_list[i][y], device),
                                                 batch_size=32, shuffle=False))
        fake_data = create_color_batch(value_idx=i, batch_size=1000)

        unlearning_dataloaders.append(DataLoader(SpecificDatasetAddReversedTrigger(fake_data, y,
                                                                                   triggers_list[i][y],
                                                                                   masks_list[i][y], device),
                                                 batch_size=32, shuffle=False))

    print("Start Unlearning Retrain!!!\n")
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(param["unlearning_eps"]):
        ep_loss = [0] * 21  # Create a loss variable for each data loader
        data_loader_tqdm = [tqdm(loader, desc=f'unlearning_Batches {i + 1}') for i, loader in
                            enumerate([unlearning_dataloader1] + unlearning_dataloaders)]

        # Read data from all loaders simultaneously
        for batch_idx, batches in enumerate(zip(*data_loader_tqdm)):
            combined_images = []
            combined_labels = []
            for i, (images, labels) in enumerate(batches):
                images, labels = images.to(device), labels.to(device)
                combined_images.append(images)
                combined_labels.append(labels)

            combined_images = torch.cat(combined_images, dim=0)
            combined_labels = torch.cat(combined_labels, dim=0)

            optimizer.zero_grad()
            outputs = net_glob(combined_images)

            # Process the first batch
            total_loss = criterion(outputs[:len(batches[0][0])], combined_labels[:len(batches[0][0])])

            # Loop to process remaining batches
            for i in range(1, len(batches)):
                loss = criterion(outputs[i * len(batches[0][0]):(i + 1) * len(batches[0][0])],
                                 combined_labels[i * len(batches[0][0]):(i + 1) * len(batches[0][0])])
                total_loss += loss

            torch.nn.utils.clip_grad_norm_(net_glob.parameters(), max_norm=20)
            total_loss.backward()
            optimizer.step()

    return net_glob.state_dict()
#################################################################################################################
