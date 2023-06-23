#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:56:02 2019

@author: aneesh
"""
import matplotlib.pyplot as plt

import os
import os.path as osp

import torch
import torch.nn as nn
from torchvision import transforms

import numpy as np


from helpers.utils import Metrics, AeroCLoader, parse_args
from networks.resnet6 import ResnetGenerator
from networks.segnet import segnet, segnetm
from networks.unet import unet, unetm

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'AeroRIT baseline evalutions')    
    
    ### 0. Config file?
    parser.add_argument('--config-file', default = None, help = 'Path to configuration file')
    
    ### 1. Data Loading
    parser.add_argument('--bands', default = 51, help = 'Which bands category to load \
                        - 3: RGB, 4: RGB + 1 Infrared, 6: RGB + 3 Infrared, 31: Visible, 51: All', type = int)
    parser.add_argument('--hsi_c', default = 'rad', help = 'Load HSI Radiance or Reflectance data?')
    
    ### 2. Network selections
    ### a. Which network?
    parser.add_argument('--network_arch', default = 'unet', help = 'Network architecture?')
    parser.add_argument('--use_mini', action = 'store_true', help = 'Use mini version of network?')
    
    ### b. ResNet config
    parser.add_argument('--resnet_blocks', default = 6, help = 'How many blocks if ResNet architecture?', type = int)
    
    ### c. UNet configs
    parser.add_argument('--use_SE', action = 'store_true', help = 'Network uses SE Layer?')
    parser.add_argument('--use_preluSE', action = 'store_true', help = 'SE layer uses ReLU or PReLU activation?')
    
    ### Load weights post network config
    parser.add_argument('--network_weights_path', default = None, help = 'Path to Saved Network weights')
    
    ### Use GPU or not
    parser.add_argument('--use_cuda', action = 'store_true', help = 'use GPUs?')
    
    args = parse_args(parser)
    print(args)
    
    # args.use_mini = True
    # args.use_SE = True
    # args.use_preluSE = True
    # args.network_weights_path = 'savedmodels/unetm.pt'
    
    if args.use_cuda and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    perf = Metrics()
    
    tx = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
    
    if args.bands == 3 or args.bands == 4 or args.bands == 6:
        testset = AeroCLoader(set_loc = 'right', set_type = 'test', size = 'small', hsi_sign = args.hsi_c, hsi_mode = '{}b'.format(args.bands), transforms = tx)
    elif args.bands == 31:
        testset = AeroCLoader(set_loc = 'right', set_type = 'test', size = 'small', hsi_sign = args.hsi_c, hsi_mode = 'visible', transforms = tx)
    elif args.bands == 51:
        testset = AeroCLoader(set_loc = 'right', set_type = 'test', size = 'small', hsi_sign = args.hsi_c, hsi_mode = 'all', transforms = tx)
        # testset = AeroCLoader(set_loc = 'left', set_type = 'train', size = 'small', hsi_sign=args.hsi_c, hsi_mode = 'all', transforms = tx)
    else:
        raise NotImplementedError('required parameter not found in dictionary')
    
    print('Completed loading data...')
    
    if args.network_arch == 'resnet':
        net = ResnetGenerator(args.bands, 6, n_blocks=args.resnet_blocks)
    elif args.network_arch == 'segnet':
        if args.mini == True:
            net = segnetm(args.bands, 6)
        else:
            net = segnet(args.bands, 6)
    elif args.network_arch == 'unet':
        if args.use_mini == True:
            net = unetm(args.bands, 6, use_SE = args.use_SE, use_PReLU = args.use_preluSE)
        else:
            net = unet(args.bands, 6)
    else:
        raise NotImplementedError('required parameter not found in dictionary')

    net.load_state_dict(torch.load(args.network_weights_path))
    net.eval()
    net.to(device)
    
    print('Completed loading pretrained network weights...')
    
    print('Calculating prediction accuracy...')
    
    labels_gt = []
    labels_pred = []
    
    import copy
    for img_idx in range(len(testset)):
        # print(img_idx)
        _, hsi, label = testset[img_idx]
        label = label.numpy()
        # if img_idx == 194:
            # label2 = copy.deepcopy(label)
            # label2[label!=3]=0
            # plt.imshow(label2, vmin=0)
            # plt.show()

        label_pred = net(hsi.unsqueeze(0).to(device))
        label_pred = label_pred.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        #
        # if 4 in label:
        #     import cv2
        #     # ax.imshow(label, vmin=0, vmax=5, cmap="jet")
        #     label_pred = copy.deepcopy(label)
        #     label_pred[label==5] = 5
        #
        #     labels_pred_color = np.zeros((label_pred.shape[0], label_pred.shape[1], 3))
        #     colors = testset.get_labels()
        #     for i in range(label_pred.shape[0]):
        #         for j in range(label_pred.shape[1]):
        #             labels_pred_color[i,j,:] = colors[label_pred[i,j]].astype(np.uint8)
        #
        #     labels_pred_color = labels_pred_color.astype("uint8")
        #     labels_pred_color = cv2.cvtColor(labels_pred_color, cv2.COLOR_RGB2BGR)
        #     # cv2.imshow("aa", labels_pred_color)
        #     # cv2.waitKey()
        #     cv2.imwrite("images_paper/111_" + str(img_idx) + "_original.png", labels_pred_color)
        #     # exit()
        # else:
        #     continue


    for img_idx in range(len(testset)):
        _, hsi, label = testset[img_idx]
        label = label.numpy()

        label_pred = net(hsi.unsqueeze(0).to(device))
        label_pred = label_pred.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
        
        label = label.flatten()
        label_pred = label_pred.flatten()
        
        labels_gt = np.append(labels_gt, label)
        labels_pred = np.append(labels_pred, label_pred)
    
    scores = perf(labels_gt, labels_pred)
    print('Statistics on Test set:\n')
    print('Overall accuracy = {:.2f}%\nAverage Accuracy = {:.2f}%\nMean IOU is {:.2f}\
          \nMean DICE score is {:.2f}'.format(scores[0]*100, scores[1]*100, scores[2]*100, scores[3]*100))
