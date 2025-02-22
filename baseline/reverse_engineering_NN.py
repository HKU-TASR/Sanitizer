import copy
import time

import numpy as np
import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from utils_training_NN import train_nn, outlier_detection_nn


def reverse_engineering_nn(dataset_test, net_glob_transformer, args, user_number, it, triggers_1, masks_1):
    # Get current date and time
    now = datetime.now()
    # Format as a string
    now_str = now.strftime("%Y-%m-%d_%H%M%S")
    print("Start T Training (our method)\n")

    test_loader = DataLoader(dataset_test, batch_size=128, shuffle=False)

    param = {
        "dataset": "cifar",
        "Epochs": args.reverse_eps,  # "Gradually adjust, initially set to 2, with the parameter adjustable to 1."
        "batch_size": args.bs_re,
        "lamda": args.lambda_weight,
        "num_classes": 10,
        "image_size": (32, 32),
        "relearning_eps": args.relearn_eps,                                   # unharmful relearning eps
        "relearning_batch_size": args.relearn_bs,
        "relearning_lr": args.relearn_lr,
        "device": args.device,
        "user_number": user_number,
        "num_of_clients": args.num_users
    }

    norm_list = []
    triggers = []
    masks = []

    now = datetime.now()
    dir_name_mask = str(args.out_dir) + '_Client_' + str(user_number) + '/' + 'reversed_' + param["dataset"] + '_' + str(
        it) + '_round' + '/mask' + now.strftime(
        '_%Y_%m_%d_%H_%M') + '/'
    dir_name_trigger = str(args.out_dir) + '_Client_' + str(user_number) + '/' + 'reversed_' + param["dataset"] + '_' + str(
        it) + '_round' + '/trigger' + now.strftime(
        '_%Y_%m_%d_%H_%M') + '/'

    if not os.path.exists(dir_name_mask):
        os.makedirs(dir_name_mask)
    if not os.path.exists(dir_name_trigger):
        os.makedirs(dir_name_trigger)

    if args.full_re == '0':
        detect_t_label = [0, 1]
    else:
        detect_t_label = range(param["num_classes"])
    start_time = time.time()
    time_spent_on_images = 0
    for label in detect_t_label:
        trigger, mask = train_nn(copy.deepcopy(net_glob_transformer),
                                 label, test_loader, param, triggers_1[label], masks_1[label], args)

        norm_list.append(torch.sum(torch.abs(mask)).item())
        triggers.append(copy.deepcopy(trigger))
        masks.append(copy.deepcopy(mask))

        print(f'Shape of trigger: {trigger.shape}') # (3,32,32)
        print(f'Shape of mask: {mask.shape}')

        trigger = trigger.cpu().detach().numpy()
        trigger = np.transpose(trigger, (1, 2, 0))
        trigger = trigger.squeeze()

        image_start_time = time.time()  # "Record the time when image saving begins."
        print(f'Shape of trigger after: {trigger.shape}') # (32, 32, 3)

        # if it > 25:
        #     plt.axis("off")
        #     plt.imshow(trigger)
        #     plt.savefig(dir_name_mask + 'trigger_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)

        mask = mask.cpu().detach().numpy()
        print(f'Shape of mask after: {mask.shape}')

        # if it > 25:
        #     plt.axis("off")
        #     plt.imshow(mask)
        #     plt.savefig(dir_name_mask + 'mask_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)

        image_end_time = time.time()  # "Record the time when image saving is completed."
        time_spent_on_images += image_end_time - image_start_time  # "Accumulate the time spent processing images."

    #######################################################################################################
    allocated = torch.cuda.memory_allocated(device=args.gpu) / (1024 ** 2)  # to MB
    reserved = torch.cuda.memory_reserved(device=args.gpu) / (1024 ** 2)  # to MB
    #######################################################################################################
    print(norm_list)
    print("Saving identified trigger!!!")

    # yt_label = 1  # outlier_detection_nn(norm_list)
    if args.full_re == '1':
        yt_label = 1
    else:
        yt_label = outlier_detection_nn(norm_list)

    end_time = time.time()
    total_time = end_time - start_time
    execution_time = total_time - time_spent_on_images  # 总时间减去处理图像的时间

    tri_image = masks[yt_label] * triggers[yt_label]
    # image = transforms.ToPILImage()(tri_image).convert("RGB")
    # image.save(dir_name_trigger + 'reversed_trigger_RGB_image.png')

    tri_image = tri_image.cpu().detach().numpy()
    tri_image = np.transpose(tri_image, (1, 2, 0))
    plt.axis("off")
    # plt.imshow(tri_image)
    # plt.savefig(dir_name_trigger + 'L_reversed_trigger_image_{}.png'.format(yt_label), bbox_inches='tight', pad_inches=0.0)

    tri_image = masks[1] * triggers[1]

    tri_image = tri_image.cpu().detach().numpy()
    tri_image = np.transpose(tri_image, (1, 2, 0))
    plt.axis("off")
    # plt.imshow(tri_image)
    # plt.savefig(dir_name_trigger + 'L0_reversed_trigger_image_{}.png'.format(1), bbox_inches='tight',
    #             pad_inches=0.0)

    # tri_image = masks[9] * triggers[9]
    #
    # tri_image = tri_image.cpu().detach().numpy()
    # tri_image = np.transpose(tri_image, (1, 2, 0))
    # plt.axis("off")
    # plt.imshow(tri_image)
    # plt.savefig(dir_name_trigger + 'L9_reversed_trigger_image_{}.png'.format(9), bbox_inches='tight',
    #             pad_inches=0.0)
    #
    # tri_image = masks[8] * triggers[8]
    #
    # tri_image = tri_image.cpu().detach().numpy()
    # tri_image = np.transpose(tri_image, (1, 2, 0))
    # plt.axis("off")
    # plt.imshow(tri_image)
    # plt.savefig(dir_name_trigger + 'L8_reversed_trigger_image_{}.png'.format(8), bbox_inches='tight',
    #             pad_inches=0.0)

    print(f"Training: Round {it}, Client {user_number} reverse engineering completed!!!")
    print(f"T3-Reverse process training code execution time in seconds (Round {it}): {execution_time:.4f} seconds")
    minutes, seconds = divmod(execution_time, 60)
    print(
        f"T3-Reverse process training code execution time in minutes (Round {it}): {minutes:.0f} minutes {seconds:.4f} seconds")
    print(
        f"T3-Local training-Allocated GPU memory: {allocated:.2f} MB, T1-Local training-Reserved GPU memory: {reserved:.2f} MB\n")
    with open(os.path.join(args.log_dir, f'time_records{user_number}.txt'), 'a') as f:
        f.write("\n")
        f.write(
            f"T3-Reverse process training code execution time in seconds (Round {it}): {execution_time:.4f} seconds\n")
        f.write(
            f"T3-Reverse process training code execution time in minutes (Round {it}): {minutes:.0f} minutes {seconds:.4f} seconds\n")
        f.write(
            f"T3-Reverse process training-Allocated GPU memory: {allocated:.2f} MB, T1-Local training-Reserved GPU memory: {reserved:.2f} MB\n")

    #     time_record_array2.append(execution_time)

    # unlearning
    # state_ = unlearning_1(dataset_test, copy.deepcopy(net_glob_transformer), yt_label, triggers, masks,
    #                       specific_background='',
    #                       param=param)
    # net_glob_transformer.load_state_dict(state_)
    # save_model(net_glob_transformer, 'net_glob_unlearning_' + now_str)

    return yt_label, triggers, masks, param
