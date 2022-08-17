import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable as V
from torch import optim
from  PIL import Image
from skimage import filters,morphology
from torchvision import transforms 
from torchvision.transforms import ToPILImage
from torchvision.transforms import functional as F
import visdom
from visdom import Visdom 
import numpy as np 
import os
import logging
import scipy.misc as misc
import cv2

def numeric_score_original(pred,gt,j):
    FP = np.float(np.sum((pred == j) & (gt != j)))
    FN = np.float(np.sum((pred != j) & (gt == j)))
    TP = np.float(np.sum((pred == j) & (gt == j)))
    TN = np.float(np.sum((pred != j) & (gt != j)))

    return FP,FN,TP,TN

def numeric_score(pred,gt,classes):
    FP_list = []
    FN_list = []
    TP_list = []
    TN_list = []
    for i in range(1,classes):
        FPI,FNI,TPI,TNI = numeric_score_original(pred,gt,i)
        FP_list.append(FPI)
        FN_list.append(FNI)
        TP_list.append(TPI)
        TN_list.append(TNI)

    FP = np.mean(FP_list)
    FN = np.mean(FN_list)
    TP = np.mean(TP_list)
    TN = np.mean(TN_list)

    return FP,FN,TP,TN

def get_acc(image,label):
    FP,FN,TP,TN = numeric_score(image,label)
    # print("FP:",FP,"FN:",FN,"TP:",TP,"TN:",TN)
    acc = (TP + TN) / (TP + FN + TN + FP + 1e-10)
    sen = (TP) / (TP + FN + 1e-10)

    return acc,sen

def metrics_self(pred,label,batch_size):
    outputs = (pred.data.cpu().numpy()).astype(np.uint8)
    label = (label.data.cpu().numpy()).astype(np.uint8)

    Acc,Sen = 0. ,0.
    for i in range(batch_size):
        # print(outputs.shape)
        # print(label.shape)
        img = outputs[i,:,:,:]
        gt = label[i,:,:,:]
        # print(img.shape,gt.shape)

        acc,sen = get_acc(img,gt)
        Acc += acc
        Sen += sen

    return Acc,Sen

def Auc_score(SR,GT,threshold=0.5):
    GT = GT.ravel()
    SR = SR.ravel()

    roc_auc = metrics.roc_auc_score(GT,SR)

    return roc_auc

def all_score(prediction,groundtruth,classes):
    # auc = Auc_score(prediction,groundtruth)
    FP,FN,TP,TN = numeric_score(prediction,groundtruth,classes)

    acc = (TP + TN) / (TP + FN + TN + FP + 1e-10)

    # if (TP + FN) <= 0.0:
    #     return 0.0
    sen = (TP) / (TP + FN + 1e-10)

    # if (TN + FP) <= 0.0:
    #     return 0.0 
    # spe = TN / (TN + FP + 1e-10)
    
    # fdr = FP / (FP + TP + 1e-10)

    pre = TP / (TP + FP + 1e-10)

    f1 = (2 * pre * sen) / (pre + sen + 1e-10)

    # if (TP + FP + FN) <= 0.0:
    #     return 0.0
    iou = TP / (TP + FP + FN + 1e-10)

    dice = (2*TP) / ((TP + FN) + (TP +FP) + 1e-10)

    outputs = prediction > 0.5
    masks = groundtruth == np.max(groundtruth)
    inter = np.sum(outputs * masks)
    dc = 2 * inter

    return iou,dice,acc,sen,pre,f1
