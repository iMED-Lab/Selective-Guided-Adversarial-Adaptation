import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from read_data.read_Heideberg import read_Heideberg
from read_data.read_Nidek import read_Nidek
from metrics import all_score
from model.cenet import CE_Net_OCT
from dataload.dataloader import read_datasets
# from model.unet import Unet,DoubleConv,Down,Up,OutConv


def test_to_tensor(image):
    img = cv2.imread(image)
    img = cv2.resize(img,(512,512))
    img = np.array(img,dtype=np.float32).transpose(2,0,1) / 255 * 3.2 - 1.6
    img = torch.Tensor(img)
    img = img.unsqueeze(0)

    return img


def random_crop(img_path,mask_path,idx):
    img = Image.open(img_path)
    mask = Image.open(mask_path).convert("L")
    i,j,h,w = transforms.RandomCrop.get_params(img,output_size=(512,1024))
    img = F.crop(img,i,j,h,w)
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

    savepath = "/home/imed/disk5TA/zjy/DA2/image"
    imgsave = os.path.join(savepath,str(idx) + ".png")
    cv2.imwrite(imgsave,img)

    img = np.array(img,dtype=np.float32).transpose(2,0,1) / 255 * 3.2 - 1.6
    img = torch.Tensor(img)
    img = img.unsqueeze(0)
    mask = F.crop(mask,i,j,h,w)
    mask = np.array(mask)

    return img,mask


def test(modelpath):
    model = torch.load(modelpath)
    image, label,imgname = read_datasets("dataset_path","test")

    net = model["model"].cuda()

    Dice = []
    Iou = []
    savepath = "save_path"

    for idx,img in enumerate(image):
        img = test_to_tensor(img).cuda()
        mask = cv2.imread(label[idx],0)
        mask = cv2.resize(mask,(512,512))
        # img ,mask = random_crop(img,label[idx],idx)
        img = img.cuda()
        mask = mask * 255
        pred = net(img)
        
        # pred = torch.topk(pred,1,dim=1)[1]
        pred = pred * 255

        pred_trans = (pred.detach().cpu().numpy()).astype(np.uint8)
        pred_trans = pred_trans.squeeze()
        _, pred_trans = cv2.threshold(pred_trans,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        pred_save = os.path.join(savepath,imgname[idx])

        cv2.imwrite(pred_save,pred_trans)

        iou,dice = all_score(pred_trans,mask)

        Dice.append(dice)
        Iou.append(iou)

        print("Dice:",dice,"Iou:",iou)

    Dice = np.mean(Dice)
    Iou = np.mean(Iou)

    print("mean_Dice:",Dice,"mean_Iou:",Iou)

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
test("model_path")




