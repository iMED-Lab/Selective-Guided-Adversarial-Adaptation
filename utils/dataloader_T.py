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


def random_HSV(image,h_limit=(-180,180),s_limit=(-255,255),v_limit=(-255,255),u=0.5):

    if np.random.random() < u:
        #rgb -> hsv
        image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(image)
        h_shift = np.random.randint(h_limit[0],h_limit[1] + 1 )
        h_shift = np.uint8(h_shift)
        h += h_shift 
        s_shift = np.random.uniform(s_limit[0],s_limit[1])
        s = cv2.add(s,s_shift)
        v_shift = np.random.uniform(v_limit[0],v_limit[1])
        v = cv2.add(v,v_shift)
        image = cv2.merge((h,s,v))
        # hsv -> rgb
        image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)

    return image

def random_ShiftScaleRotate(image,mask,shift_limit=(-0.0,0.0),
                            scale_limit=(-0.0,0.0),rotate_limit=(-0.0,0.0),
                            aspect_limit=(-0.0,0.0),borderMode=cv2.BORDER_CONSTANT,u=0.5):
 
    if np.random.random() < u:
     
        height,width,channel = image.shape
        
        
        angle = np.random.uniform(rotate_limit[0],rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0],1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0],1 + aspect_limit[1])

        sx = scale * aspect / (aspect ** 0.5)
        sy = scale * (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0],shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0],shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc,-ss],[ss,cc]])

        box0 = np.array([[0,0],[width,0],[width,height],[0,height]])
        box1 = box0 - np.array([width / 2,height / 2])
        box1 = np.dot(box1,rotate_matrix.T) + np.array([width / 2 + dx,height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0,box1)
        image = cv2.warpPerspective(image,mat,(width,height),flags=cv2.INTER_LINEAR,borderMode=borderMode,borderValue=(0,0,0))
        mask = cv2.warpPerspective(mask,mat,(width,height),flags=cv2.INTER_LINEAR,borderMode=borderMode,borderValue=(0,0,0))

    return image,mask


def randomFlip_H(image,mask,u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image,1)
        mask = cv2.flip(mask,1)

    return image,mask

def randomFlip_V(image,mask,u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image,0)
        mask = cv2.flip(mask,0)

    return image,mask

def randomRotate_90(image,mask,u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image,mask

def random_crop(img_path,mask_path):
    img = Image.open(img_path)
    mask = Image.open(mask_path).convert("L")
    i,j,h,w = transforms.RandomCrop.get_params(img,output_size=(512,512))
    img = F.crop(img,i,j,h,w)
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
    mask = F.crop(mask,i,j,h,w)
    mask = np.array(mask)

    return img,mask


def get_one_hot(label,N):

    size = list(label.size())

    label = label.view(-1)
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0,label)
    # size = [512,512,8]
    size.append(N)

    return ones.view(*size)         


def default_loader(img_path,mask_path,mode):
    # img_target = cv2.imread(img_target_path)

    img = cv2.imread(img_path)
    mask = np.array(Image.open(mask_path).convert("L"))
    # img_target = cv2.imread(img_target_path)


    if mode == "train":
        # img,mask = random_crop(img_path,mask_path)
        img = random_HSV(img,h_limit=(-30,30),s_limit=(-5,5),v_limit=(-15,15))
        img,mask = random_ShiftScaleRotate(img,mask,
                                           shift_limit=(-0.1,0.1),
                                           scale_limit=(-0.1,0.1),
                                           aspect_limit=(-0.1,0.1),
                                           rotate_limit=(-0,0))

        img,mask = randomFlip_H(img,mask)
        img,mask = randomFlip_V(img,mask)
        # img,mask,img_target = randomFlip_V(img,mask,img_target)
        img,mask = randomRotate_90(img,mask)

    mask = mask[:,:].copy()
    mask = torch.LongTensor(mask)
    # mask_one_hot = get_one_hot(mask,2)

    # mask = mask.permute(2,0,1)
    mask = mask.numpy()
    img = np.array(img,dtype=np.float32).transpose(2,0,1) / 255 * 3.2 - 1.6
    # print(np.max(img))

    return img,mask


def read_datasets(root_path,mode="train"):

    images = []
    masks = []
    masks_names = []

   
    if mode == "train":
        outer_path = os.path.join(root_path,"train")
        folderlist = os.listdir(outer_path)
    else:
        outer_path = os.path.join(root_path,"test")
        folderlist = os.listdir(outer_path)


    for folder in folderlist:

        middle_path = os.path.join(outer_path,folder)
        image_path = os.path.join(middle_path,"image_crop")
        label_path = os.path.join(middle_path,"label_crop")
        images1 = os.listdir(image_path)

   
        for image in images1:
           
            imagePath = os.path.join(image_path,image)
            images.append(imagePath)

            maskPath = os.path.join(label_path,image)
            masks.append(maskPath)

            image_name = folder.split(".")[0] + "_" +image
            masks_names.append(image_name)


    return images,masks,masks_names


class ImageFloder(data.Dataset):
    def __init__(self,root_path,datasets="CHOROID",mode="train"):
        self.root = root_path
        self.mode = mode
        self.dataset = datasets
        assert self.dataset in ["CHOROID"],"the dataset should be in 'CHOROID' "
        if self.dataset == "CHOROID":
            self.images,self.labels,self.image_names = read_datasets(self.root,self.mode)


    def __getitem__(self,index):
        img,mask = default_loader(self.images[index],self.labels[index],self.mode)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        if self.mode == "train" or self.mode == "test":
            return img,mask
        else:
            return img,mask,self.image_names[index]

    def __len__(self):
        assert len(self.images) == len(self.labels)
        return len(self.images)

class Choroid_loader:
    def __init__(self,root_path,datasets):
        self.root_path = root_path
        self.datasets = datasets
        pass
    def load_train_data(self,batch_size):
        dataset = ImageFloder(self.root_path,self.datasets,mode="train")
        train_loader = data.DataLoader(dataset,batch_size,num_workers=0,shuffle=True,pin_memory=True)
        return train_loader

    def load_test_data(self,batch_size):
        dataset = ImageFloder(self.root_path,self.datasets,mode="test")
        test_loader = data.DataLoader(dataset,batch_size,num_workers=0,shuffle=True,pin_memory=True)
        return test_loader

    def load_pred_data(self):
        dataset = ImageFloder(self.root_path,self.datasets,mode="pred")
        pred_loader = data.DataLoader(dataset,batch_size=1,num_workers=0,shuffle=False,pin_memory=True)
        return pred_loader,dataset.images
