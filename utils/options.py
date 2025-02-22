#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=200, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--poisoning_rate', type=float, default=0.1, help="backdoor poisoning rate")

    # watermarking arguments
    parser.add_argument('--wm_method', type=str, default='gaussian_noise', help='wm_method for ownership verification')
    parser.add_argument('--implant_way', type=str, default='Random', help="implant way for watermarking embedding")
    parser.add_argument('--reverse_eps', type=int, default=2, help="the epochs of reverse round: E")

    # UL -> RE -> Unharmful
    parser.add_argument('--epochs_ul', type=int, default=50, help="the number of local epochs: E")
    parser.add_argument('--bs_ul', type=int, default=128, help="the number of batch size: bs")
    parser.add_argument('--defense_data_ratio', type=float, default=0.05, help='ratio of defense data')
    parser.add_argument('--topK_ratio', type=float, default=0.1, help="topK_ratio rate")
    parser.add_argument('--clean_threshold', type=float, default=0.20, help='threshold of unlearning accuracy')
    parser.add_argument('--lr_ul', type=float, default=0.015, help="UL learning rate")
    parser.add_argument('--full_re', type=str, default=1, help="all labels or not")
    parser.add_argument('--lambda_weight', type=float, default=0.01, help="re balance weight")
    parser.add_argument('--lr_re', type=float, default=0.015, help="RE learning rate")
    parser.add_argument('--bs_re', type=int, default=128, help="batch size of re epochs: E")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--scale', type=int, default=0, help='whether i.i.d or not')
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--conflict', type=int, default=0, help='whether conflicts or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    parser.add_argument('--alpha_noniid', type=float, default=0.5, help='concentration parameter for Dirichlet distribution')
    parser.add_argument('--my_dict', type=str,
                       default="{0: 'white_block', 1: 'white_block', 2: 'white_block', 3: 'white_block', "
                               "4: 'white_block',"
                               "5: 'white_block', 6: 'white_block', 7: 'white_block', 8: 'white_block', "
                               "9: 'white_block'}",
                       help='Dictionary for triggers, e.g. "{0: \'white_block\', 1: \'white_block\'}"')

    parser.add_argument('--my_dict_label', type=str,
                        default="{0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}",
                        help='Dictionary for labels, e.g. "{0: 1, 1: 1}"')

    parser.add_argument('--wm_or_not_dict', type=str,
                        default="{0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True, "
                                "9: True}",
                        help='Dictionary for watermark booleans, e.g. "{0: True, 1: True}"')
    parser.add_argument('--relearn_eps', type=float, default=50, help='unharmful relearning eps')
    parser.add_argument('--relearn_bs', type=float, default=32, help='unharmful relearning bs')
    parser.add_argument('--relearn_lr', type=float, default=0.005, help='unharmful relearning lr')
    args = parser.parse_args()
    return args
