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


class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,out_channels)
        )
    def forward(self,x):    
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2,in_channels // 2,kernel_size=2,stride=2)
        self.conv = DoubleConv(in_channels,out_channels)
    def forward(self,x1,x2):
        x1 = self.up(x1)
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x2.size()[3]
        # x1 = F.pad(x1,[diffX // 2,diffX - diffX // 2,
        #                diffY // 2,diffY - diffY // 2])
        x = torch.cat([x2,x1],dim = 1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,1)
    def forward(self,x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True):
        super().__init__()
        self.biliner = bilinear
        self.inchannel = DoubleConv(in_channels,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        self.down4 = Down(512,512)
        self.up1 = Up(1024,256,bilinear)
        self.up2 = Up(512,128,bilinear)
        self.up3 = Up(256,64,bilinear)
        self.up4 = Up(128,64,bilinear)
        self.outchannel = OutConv(64,out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x1 = self.inchannel(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        out = self.outchannel(x)
        out = self.sigmoid(out)
        return out


class Encoder(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.inchannel = DoubleConv(in_channels,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        self.down4 = Down(512,512)

    def forward(self,x):
        x1 = self.inchannel(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1,x2,x3,x4,x5


class Decoder(nn.Module):
    def __init__(self,out_channels,bilinear=True):
        super().__init__()
        self.up1 = Up(1024,256,bilinear)
        self.up2 = Up(512,128,bilinear)
        self.up3 = Up(256,64,bilinear)
        self.up4 = Up(128,64,bilinear)
        self.outchannel = OutConv(64,out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x1,x2,x3,x4,x5):
        x = self.up1(x5,x4)
        x = self.up2(x,x3)
        x = self.up3(x,x2)
        x = self.up4(x,x1)
        out = self.outchannel(x)
        out = self.sigmoid(out)

        return out

