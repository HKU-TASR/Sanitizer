import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from backdoors.utils_training import NormalDatasetAddReversedTrigger
from utils.util import save_model
from main_NN.oneClient import time_record_array1, time_record_array2, time_record_array3


def unlearning_nn(dataset, net_glob, backdoor_label, triggers, masks, specific_background, param, it):
    now = datetime.now()
    # Format as a string
    now_str = now.strftime("%Y-%m-%d_%H%M%S")
    device = param["device"]
    start_time = time.time()

    net_glob.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=0.01, momentum=0.9)
    unlearning_dataloader1 = DataLoader(dataset, batch_size=128, shuffle=False)
    unlearning_dataloader2 = DataLoader(NormalDatasetAddReversedTrigger(dataset, triggers[backdoor_label].to(device),
                                                                        masks[backdoor_label].to(device)),
                                        batch_size=128, shuffle=False)

    print("Start Unlearning Retrain NN!!!\n")
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(param["unlearning_eps"]):
        ep_loss_normal, ep_loss_trigger = 0, 0
        # 同时从三个加载器中读取数据
        for batch_idx, ((images1, labels1), (images2, labels2)) in enumerate(
                zip(tqdm(unlearning_dataloader1, desc='unlearning_Batches 1'),
                    tqdm(unlearning_dataloader2, desc='unlearning_Batches 2')
                    )):
            # 将图像和标签转移到适当的设备（如GPU）
            images1, labels1 = images1.to(device), labels1.to(device)
            images2, labels2 = images2.to(device), labels2.to(device)

            # 合并三个批次的图像
            combined_images = torch.cat([images1, images2], dim=0)

            optimizer.zero_grad()
            outputs = net_glob(combined_images)

            loss_normal = criterion(outputs[:len(images1)], labels1)
            loss_augmented_trigger = criterion(outputs[len(images1):2 * len(images1)], labels2)

            # 计算总损失
            lambda_original, lambda_augmented_trigger, lambda_background = 1, 1, 1
            total_loss = (lambda_original * loss_normal +
                          lambda_augmented_trigger * loss_augmented_trigger)

            total_loss.backward()
            optimizer.step()

            ep_loss_normal += loss_normal.item()
            ep_loss_trigger += loss_augmented_trigger.item()

        print(f'正常 Epoch {epoch + 1}/{param["unlearning_eps"]}, Loss: {ep_loss_normal:.4f}')
        print(f'正常（trigger）Epoch {epoch + 1}/{param["unlearning_eps"]}, Loss: {ep_loss_trigger:.4f}')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"T3-Unlearning训练代码执行秒数时间（第{it}round）：{execution_time:.4f} 秒")
    minutes, seconds = divmod(execution_time, 60)
    print(f"T3-Unlearning训练代码执行分钟时间（第{it}round）：{minutes:.0f} 分钟 {seconds:.4f} 秒")

    with open('time_records1.txt', 'a') as f:
        f.write("\n")
        f.write(f"T3-Unlearning训练代码执行秒数时间（第{it}round）：{execution_time:.4f} 秒\n")
        f.write(f"T3-Unlearning训练代码执行分钟时间（第{it}round）：{minutes:.0f} 分钟 {seconds:.4f} 秒\n")
        time_record_array3.append(execution_time)
    save_model(net_glob, 'net_glob_unlearning_' + now_str)

    return net_glob.state_dict()
