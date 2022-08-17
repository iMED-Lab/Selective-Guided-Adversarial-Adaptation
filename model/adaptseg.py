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

class Discriminator(nn.Module):
    def __init__(self,channel):
        super().__init__()
        self.conv1 = nn.Conv2d(channel,64,7,2,3)
        self.bn1 = nn.BatchNorm2d(64)
        self.leakyrelu1 = nn.LeakyReLU(0.2,inplace=False)
        self.conv2 = nn.Conv2d(64,128,5,2,2)
        self.bn2 = nn.BatchNorm2d(128)
        self.leakyrelu2 = nn.LeakyReLU(0.2,inplace=False)
        self.conv3 = nn.Conv2d(128,256,5,2,2)
        self.bn3 = nn.BatchNorm2d(256)
        self.leakyrelu3 = nn.LeakyReLU(0.2,inplace=False)
        self.conv4 = nn.Conv2d(256,512,3,2,1)
        self.bn4 = nn.BatchNorm2d(512)
        self.leakyrelu4 = nn.LeakyReLU(0.2,inplace=False)
        self.conv5 = nn.Conv2d(512,128,1,1,0)
        self.bn5 = nn.BatchNorm2d(128)
        self.leakyrelu5 = nn.LeakyReLU(0.2,inplace=False)
        self.conv6 = nn.Conv2d(128,32,1,1,0)
        self.bn6 = nn.BatchNorm2d(32)
        self.leakyrelu6 = nn.LeakyReLU(0.2,inplace=False)
        self.conv7 = nn.Conv2d(32,1,1,1,0)
        self.bn7 = nn.BatchNorm2d(1)
        self.leakyrelu = nn.LeakyReLU(0.2,False)
        self.sigmoid = nn.Sigmoid()

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
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.leakyrelu5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.leakyrelu6(x)
        x = self.conv7(x)
        # x = self.bn7(x)
        x = self.sigmoid(x)
        # print(x.shape)      

        return x



