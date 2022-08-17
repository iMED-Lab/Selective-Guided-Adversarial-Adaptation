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
from dataload.dataloader_Z import Choroid_loader
from visualizer import Visualizer
from metrics import all_score
from model.unet import Unet,Encoder,Decoder
# from model.cenet import CE_Net_OCT
# from model.adaptseg import Discriminator
from model.E_discriminator import Encoder_Discriminator
from model.adaptseg import Discriminator

def param_copy(param_s,param_t,alpha,epoch):
    # param_s = list(param_s)
    # param_t = list(param_t)
    if epoch <= 1:
        for ps,pt in zip(param_s,param_t):
            pt.data[:] = ps.data[:]

    else:
        for ps,pt in zip(param_s,param_t):
            pt.data.mul_(alpha)
            pt.data.add_(ps.data * (1-alpha))

        
def param_select(param1,grad2,param3,alpha,epoch):
    # param2_grad = param2.grad
    if epoch <= 1:
        for ps,pt in zip(param1,param3):
            pt.data[:] = ps.data[:] 
    # 对每一次层参数
    # count = 0
    else:
        for ps,gc,pt in zip(param1,grad2,param3):        
            ps_data,pt_data = ps.data,pt.data
            g_data = gc

            # 判断是否为卷积层
            if len(ps_data.shape) > 2:
                # 对每一个卷积核, (梯度*参数)选择 --> EMA更新       
                for i in range(ps.shape[0]):
                    # (梯度*参数)选择
                    select_matrix = torch.abs(torch.mul(ps_data[i],g_data[i]))
                    select_matrix1 = torch.where(select_matrix > 0.001*torch.max(select_matrix),1.0,0.0)
                    # print(torch.sum(select_matrix1))
                    select_matrix2 = torch.where(select_matrix1 == 1-alpha,alpha,1.0)
                    # max1,min1 = torch.max(select_matrix),torch.min(select_matrix)
                    # select_matrix = (select_matrix - min1) / ((max1 - min1) + 1e-8)
                    pn = torch.mul(ps_data[i],select_matrix1).cuda()
                    # if count == 0 and i == 0:
                    #     print("select_matrix",select_matrix)
                    #     print("pn",pn)
                    # EMA更新
                    pt_data[i].mul_(select_matrix2)
                    pt_data[i].add_(pn)
            # 对BN层更新
            else:
                pt_data.mul_(alpha)
                pt_data.add_(ps.data * (1-alpha))
            # count += 1

            print("Parameters Selectively Guide")



args = {
    "root":"/media/imed/personal/zjy/DA2/OCT_dataset/Topcon",
    "img_save_path":"./save_img",
    "epoches":400,
    "lr": 5e-5,
    "lr_ed": 5e-5,
    "snapshot":100,
    "test_step":1,
    "ckpt_path":"/media/imed/personal/zjy/DA2/Checkpoint/SGAA2_Zeiss_unet/",
    "batch_size":4,
    "name":"SGAA2_Zeiss_unet",
    "epoch":0
}

class Trainers:
    def __init__(self):
        # 加载数据
        data_loader = Choroid_loader(root_path=args["root"],datasets="CHOROID")
        self.train_loader = data_loader.load_train_data(batch_size=args["batch_size"])
        self.test_loader = data_loader.load_test_data(batch_size=4)
        # self.pred_loader = dataloader.load_pred_data()

    # 保存模型
    def save_ckpt(self,encoder_s,encoder_t,decoder,epoch):
        if not os.path.exists(args["ckpt_path"]):
            os.makedirs(args["ckpt_path"])
        # 模型和评价都保存
        state = {"encoder_s":encoder_s,"encoder_t":encoder_t,"decoder":decoder}
        torch.save(state,args["ckpt_path"] + args["name"] + "_epoch_" + str(epoch) + ".pkl")
        print("---> save model:{} <---".format(args["ckpt_path"]))

    # 调整学习率
    def adjust_lr(self,optimizer,base_lr,iter,max_iter,power=0.9):
        lr = base_lr * (1 - float(iter) / max_iter) ** power
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr 

    # 模型评价
    def model_eval(self,encoder,decoder):
        print("Start testing model...")

        # 进行评价
        encoder.eval()
        decoder.eval()
        iou,dice = [],[]
        acc,sen,pre,f1 = [],[],[],[]
        file_num = 0
        for idx,batch in enumerate(self.test_loader):
            image = batch[2].float().cuda()
            label = batch[3].float().cuda()
          
            x1,x2,x3,x4,x5 = encoder(image)
            pred = decoder(x1,x2,x3,x4,x5)

            file_num += 1 
            # (8,512,512) -> (1,512,512) 获得one-hot向量的索引
            pred = torch.topk(pred, 1, dim=1)[1]
            # print(np.sum(pred.detach().cpu().numpy()))
            # label = label 
            # pred = pred  * 255

            pred_trans = (pred.detach().cpu().numpy()).astype(np.uint8)
            pred_trans = pred_trans.squeeze()
            # _, pred_trans = cv2.threshold(pred_trans,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # label = torch.topk(label, 1, dim=1)[1]
            label_trans = (label.detach().cpu().numpy()).astype(np.uint8)
            label_trans = label_trans.squeeze()

            indicator = all_score(pred_trans,label_trans,10)
            print("iou:",indicator[0],"dice:",indicator[1],"acc:",indicator[2],"sen:",indicator[3],"pre:",indicator[4],"f1:",indicator[5])

            dice.append(indicator[1])
            iou.append(indicator[0])
            acc.append(indicator[2])
            sen.append(indicator[3])
            pre.append(indicator[4])
            f1.append(indicator[5])

            # 0-7 -> 0-255
            # label = label * 255
            # pred = pred  * 255

            pred_trans = torch.Tensor(pred_trans)

            vis.img(name="images",img_=(image[0,:,:,:]+1.6) / 3.2 * 255)
            vis.img(name="labels",img_=label[0,:,:]*25)
            vis.img(name="perdiction",img_=pred[0,:,:,:]*25)

        return np.mean(iou),np.mean(dice),np.mean(acc),np.mean(sen),np.mean(pre),np.mean(f1)


    def train(self):
        # 网络
        encoder_s = Encoder(3).cuda()
        encoder_c = Encoder(3).cuda()
        encoder_t = Encoder(3).cuda()
        decoder = Decoder(10).cuda()
        E_discriminator = Encoder_Discriminator(14).cuda()
        D_discriminator = Discriminator(10).cuda()
        # 优化器
        optimizer_en_s = optim.Adam(encoder_s.parameters(),lr=args["lr"],weight_decay=5e-4)
        optimizer_en_t = optim.Adam(encoder_t.parameters(),lr=args["lr_ed"],weight_decay=5e-4)
        optimizer_en_c = optim.Adam(encoder_c.parameters(),lr=args["lr_ed"],weight_decay=5e-4)
        optimizer_de = optim.Adam(decoder.parameters(),lr=args["lr"],weight_decay=5e-4)
        optimizer_ED = optim.Adam(E_discriminator.parameters(),lr=args["lr"],weight_decay=5e-4)
        optimizer_DD = optim.Adam(D_discriminator.parameters(),lr=args["lr"],weight_decay=5e-4)
        # 损失函数
        # critrion1 = nn.MSELoss().cuda()
        seg_loss = nn.CrossEntropyLoss().cuda()
        E_loss = nn.BCELoss().cuda()
        D_loss = nn.BCELoss().cuda()
        
        print("----------start training----------")

        iters = 1
        iou = 0.
        dice = 0.
        acc = 0.
        pre = 0.
        sen = 0.
        f1 = 0. 

        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = False
        # torch.backends.cudnn.allow_tf32 = True

        for epoch in range(args["epoches"]):
            encoder_s.train()
            encoder_c.train()
            encoder_t.train()
            decoder.train()
            E_discriminator.train()
            D_discriminator.train()

            # 参数选择传递
            if epoch >= args["epoch"]+1:
                param_s = list(encoder_s.parameters())
                param_t = list(encoder_t.parameters())
                param_select(param_s,grad_c,param_t,0.88,epoch)
            
            if epoch >= args["epoch"]:
                # 参数拷贝
                param_s = list(encoder_s.parameters())
                param_c = list(encoder_c.parameters())
                for ps,pc in zip(param_s,param_c):
                    pc.data[:] = ps.data[:]

                # 构建梯度空矩阵
                grad_c = []
                for i in range(len(param_c)):
                    grad_c_layer = torch.zeros_like(param_c[i])
                    grad_c.append(grad_c_layer)
        
            for idx,batch in enumerate(self.train_loader):
                image_s = batch[0].cuda()
                label_s = batch[1].cuda()
                image_t = batch[2].cuda()

                e_real = torch.full((image_s.shape[0],1),1).float().cuda()
                e_fake = torch.full((image_s.shape[0],1),0).float().cuda()

                d_real = torch.full((image_s.shape[0],1,32,14),1).float().cuda()
                d_fake = torch.full((image_s.shape[0],1,32,14),0).float().cuda()

                optimizer_en_s.zero_grad()
                optimizer_de.zero_grad()
                x1,x2,x3,x4,x5 = encoder_s(image_s)
                pred = decoder(x1,x2,x3,x4,x5)
                loss_seg = seg_loss(pred,label_s.long())
                loss_seg.backward()
                optimizer_en_s.step()
                optimizer_de.step()

                print("[{0:d}:{1:d}] --- loss_seg:{2:.10f}".format(epoch + 1,iters,loss_seg.item()))
                vis.plot(name="seg_loss",y=loss_seg.item(),opts=dict(title="seg_loss",xlabel="batch",ylabel="loss"))

                # if (epoch+1) >= 20 and iters % 2 == 1:
                # 参数选择传递(编码对抗)
                
                # print(param_s[0].data)
                # print(param_s[0].grad.data[0].shape)

                if epoch >= args["epoch"]+1:        
                    optimizer_ED.zero_grad()
                    _,_,_,_,s5 = encoder_c(image_s)                
                    encoder_pred_s = E_discriminator(s5)
                    en_d_s_loss = E_loss(encoder_pred_s,e_real)
                    en_d_s_loss.backward()

                    _,_,_,_,t5 = encoder_c(image_t)
                    encoder_pred_t = E_discriminator(t5)
                    en_d_t_loss = E_loss(encoder_pred_t,e_fake)
                    en_d_t_loss.backward()
                    optimizer_ED.step()
                    
                    optimizer_en_c.zero_grad()
                    optimizer_ED.zero_grad()
                    _,_,_,_,t5 = encoder_c(image_t)     
                    encoder_pred_t = E_discriminator(t5)
                    en_g_t_loss = E_loss(encoder_pred_t,e_real)
                    en_g_t_loss.backward()
                    optimizer_en_c.step()
                    parmas_c = list(encoder_c.parameters())
                    for p_c,g_c in zip(parmas_c,grad_c):
                        gs_c = torch.abs(p_c.grad.data)
                        g_c = g_c + gs_c
                                    
                    # if (epoch+1) >= 20 and iters % 2 == 1:    
                    # 解码对抗                
                    optimizer_DD.zero_grad()
                    s1,s2,s3,s4,s5 = encoder_s(image_s)
                    decoder_s = decoder(s1,s2,s3,s4,s5)
                    decoder_pred_s = D_discriminator(decoder_s)
                    de_d_s_loss = D_loss(decoder_pred_s,d_real)
                    de_d_s_loss.backward()
                    
                    t1,t2,t3,t4,t5 = encoder_t(image_t)
                    # print(t5)
                    decoder_t = decoder(t1,t2,t3,t4,t5)
                    # print(decoder_t)
                    decoder_pred_t = D_discriminator(decoder_t)
                    # print(decoder_pred_t)
                    # print(de_d_t_loss.item())
                    de_d_t_loss = D_loss(decoder_pred_t,d_fake)
                    de_d_t_loss.backward()
                    optimizer_DD.step()

                    optimizer_en_t.zero_grad()
                    optimizer_de.zero_grad()
                    t1,t2,t3,t4,t5 = encoder_t(image_t)
                    decoder_t = decoder(t1,t2,t3,t4,t5)
                    decoder_pred_t = D_discriminator(decoder_t)
                    # print(decoder_pred_t)
                    de_g_t_loss = D_loss(decoder_pred_t,d_real) 
                    # print(de_g_t_loss.item())
                    de_g_t_loss.backward()
                    optimizer_en_t.step()
                    optimizer_de.step()
                    
                    print("[{0:d}:{1:d}] --- ED_loss_s:{2:.8f}\tED_loss_t:{3:.8f}\tEG_loss_t:{4:.8f}\tDD_loss_s:{5:.8f}\tDD_loss_t:{6:.8f}\tDG_loss_t:{7:.8f}".format(epoch + 1,iters,en_d_s_loss.item(),en_d_t_loss.item(),en_g_t_loss.item(),de_d_s_loss.item(),de_d_t_loss.item(),de_g_t_loss.item()))
                    
                    # vis.plot(name="ED_loss_s",y=en_d_s_loss.item(),opts=dict(title="ED_loss_s",xlabel="batch",ylabel="loss"))
                    # vis.plot(name="ED_loss_t",y=en_d_t_loss.item(),opts=dict(title="ED_loss_t",xlabel="batch",ylabel="loss"))
                    # vis.plot(name="EG_loss_t",y=en_g_t_loss.item(),opts=dict(title="EG_loss_t",xlabel="batch",ylabel="loss"))
                    vis.plot(name="DD_loss_s",y=de_d_s_loss.item(),opts=dict(title="DD_loss_s",xlabel="batch",ylabel="loss"))
                    vis.plot(name="DD_loss_t",y=de_d_t_loss.item(),opts=dict(title="DD_loss_t",xlabel="batch",ylabel="loss"))
                    vis.plot(name="DG_loss_t",y=de_g_t_loss.item(),opts=dict(title="DG_loss_t",xlabel="batch",ylabel="loss"))
                
                iters += 1

            if epoch >= args["epoch"] and (epoch+1) % 1 == 0:
                test_iou,test_dice,acc,sen,pre,f1 = self.model_eval(encoder_t,decoder)
                print(test_iou,test_dice)
                vis.plot(name="iou",y=test_iou,opts=dict(title="iou",xlabel="epoch",ylabel="iou"))
                vis.plot(name="dice",y=test_dice,opts=dict(title="dice",xlabel="epoch",ylabel="dice"))
                vis.plot(name="acc",y=acc,opts=dict(title="acc",xlabel="epoch",ylabel="acc"))
                vis.plot(name="sen",y=sen,opts=dict(title="sen",xlabel="epoch",ylabel="sen"))
                vis.plot(name="pre",y=pre,opts=dict(title="pre",xlabel="epoch",ylabel="pre"))
                vis.plot(name="f1",y=f1,opts=dict(title="f1",xlabel="epoch",ylabel="f1"))
                if (test_iou>iou) & (test_dice>dice):
                    self.save_ckpt(encoder_s,encoder_t,decoder,epoch)
                    iou = test_iou
                    dice = test_dice

if __name__ == "__main__":
    # 定义一个可视化变量
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    vis = Visualizer(env=args["name"],port = 1996)
    trainer = Trainers()
    trainer.train()






