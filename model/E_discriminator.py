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

class Encoder_Discriminator(nn.Module):
    def __init__(self,size):
        super().__init__()
        self.conv1 = nn.Conv2d(512,256,3,1,1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256,128,3,1,1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,64,3,1,1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,32,3,1,1)
        self.bn4 = nn.BatchNorm2d(32)

        self.leakyrelu1 = nn.LeakyReLU(0.2,inplace=True)
        self.leakyrelu2 = nn.LeakyReLU(0.2,inplace=True)
        self.leakyrelu3 = nn.LeakyReLU(0.2,inplace=True)
        self.leakyrelu4 = nn.LeakyReLU(0.2,inplace=True)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32*32*size,1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyrelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leakyrelu4(x)
        # print(x.shape)
        x = self.flatten(x)
        # print(x.shape)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        # print(x.shape)

        return x 