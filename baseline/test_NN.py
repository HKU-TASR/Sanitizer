from datetime import datetime

from torch.utils.data import DataLoader

from backdoors.fine_tune import HoneypotsDatasetRandom, HoneypotsDataset
from models.test import test_img_loader
from main_NN.oneClient import *


def test_nn_before(net_glob, dataset_train, dataset_test, args):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open('ACC_records1.txt', 'a') as f:
        f.write(now + "BEFORE UL.\n")

        train_loader = DataLoader(dataset_train, batch_size=128, shuffle=False)
        acc_train, loss_test = test_img_loader(net_glob, train_loader, args)
        print("正常图片，输出正确标签 Training accuracy: {:.4f}".format(acc_train))
        f.write("正常图片，输出正确标签 Training accuracy: {:.4f}\n".format(acc_train))

        # L1:使用2000正常图片 -> 正常标签：
        test_loader = DataLoader(dataset_test, batch_size=128, shuffle=False)
        acc_test, loss_test = test_img_loader(net_glob, test_loader, args)
        print("正常图片，输出正确标签 Testing accuracy: {:.4f}".format(acc_test))
        f.write("正常图片，输出正确标签 Testing accuracy: {:.4f}\n".format(acc_test))

        # ASR
        # L2:使用与L1一样的2000正常图片+trigger -> 特定标签：
        dataset_for_normal_image_with_trigger = (
            DataLoader(HoneypotsDataset(dataset_test), batch_size=128))
        acc_honeypots_test, loss_honeypots_test = test_img_loader(
            net_glob, dataset_for_normal_image_with_trigger, args)
        print("正常图片+trigger，输出后门标签 ASR: {:.4f}".format(acc_honeypots_test))
        f.write("正常图片+trigger，输出后门标签 ASR: {:.4f}\n".format(acc_honeypots_test))

        acc_train_before.append(acc_train)
        acc_test_before.append(acc_test)
        acc_asr_before.append(acc_honeypots_test)


def test_nn_after(net_glob, dataset_train, dataset_test, args):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open('ACC_records1.txt', 'a') as f:
        f.write(now + "AFTER UL.\n")

        train_loader = DataLoader(dataset_train, batch_size=128, shuffle=False)
        acc_train, loss_test = test_img_loader(net_glob, train_loader, args)
        print("正常图片，输出正确标签 Training accuracy: {:.4f}".format(acc_train))
        f.write("正常图片，输出正确标签 Training accuracy: {:.4f}\n".format(acc_train))

        # L1:使用2000正常图片 -> 正常标签：
        test_loader = DataLoader(dataset_test, batch_size=128, shuffle=False)
        acc_test, loss_test = test_img_loader(net_glob, test_loader, args)
        print("正常图片，输出正确标签 Testing accuracy: {:.4f}".format(acc_test))
        f.write("正常图片，输出正确标签 Testing accuracy: {:.4f}\n".format(acc_test))

        # ASR
        # L2:使用与L1一样的2000正常图片+trigger -> 特定标签：
        dataset_for_normal_image_with_trigger = (
            DataLoader(HoneypotsDataset(dataset_test), batch_size=128))
        acc_honeypots_test, loss_honeypots_test = test_img_loader(
            net_glob, dataset_for_normal_image_with_trigger, args)
        print("正常图片+trigger，输出后门标签 ASR: {:.4f}".format(acc_honeypots_test))
        f.write("正常图片+trigger，输出后门标签 ASR: {:.4f}\n".format(acc_honeypots_test))

        acc_train_after.append(acc_train)
        acc_test_after.append(acc_test)
        acc_asr_after.append(acc_honeypots_test)

        # L3:使用与L1一样的2000正常图片+trigger -> 正常标签：
        print("#######################################################################################################")
        dataset_for_normal_image_with_trigger = (
            DataLoader(HoneypotsDatasetRandom(dataset_test), batch_size=128))
        acc_honeypots_test, loss_honeypots_test = test_img_loader(
            net_glob, dataset_for_normal_image_with_trigger, args)
        print("AAA正常图片+trigger，输出正常标签 ASR: {:.4f}".format(acc_honeypots_test))
        f.write("AAA正常图片+trigger，输出正常标签 ASR: {:.4f}\n".format(acc_honeypots_test))
        print("#######################################################################################################")
