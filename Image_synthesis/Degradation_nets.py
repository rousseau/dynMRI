from os.path import expanduser
from posix import listdir
from turtle import forward
from numpy import NaN, dtype
from torch.optim import optimizer
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchio.data import image
from torchio.utils import check_sequence
home = expanduser("~")

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
import sys
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib
from torchvision.models import resnet50
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import glob
import multiprocessing
import math

from pytorch_lightning.loggers import TensorBoardLogger, tensorboard
import argparse

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

# class Unet(nn.Module): # 472K
#     def __init__(self, n_channels = 1, n_classes = 10, n_features = 32):
#         super(Unet, self).__init__()

#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.n_features = n_features

#         def double_conv(in_channels, out_channels):
#             return nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True),
#             )

#         self.dc1 = double_conv(self.n_channels, self.n_features)
#         self.dc2 = double_conv(self.n_features, self.n_features*2)
#         self.dc3 = double_conv(self.n_features*2, self.n_features*4)
#         self.dc4 = double_conv(self.n_features*6, self.n_features*2)
#         self.dc5 = double_conv(self.n_features*3, self.n_features)
#         self.mp = nn.MaxPool2d(2)

#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #binlinear?

#         self.out = nn.Conv2d(self.n_features, self.n_classes, kernel_size=1)

#     def forward(self, x):
#         x1 = self.dc1(x)

#         x2 = self.mp(x1)
#         x2 = self.dc2(x2)

#         x3 = self.mp(x2)
#         x3 = self.dc3(x3)

#         x4 = self.up(x3)
#         x4 = torch.cat([x4,x2], dim=1)
#         x4 = self.dc4(x4)

#         x5 = self.up(x4)
#         x5 = torch.cat([x5,x1], dim=1)
#         x5 = self.dc5(x5)
#         return self.out(x5)


class Unet(nn.Module): #1M
    def __init__(self, n_channels = 1, n_classes = 10, n_features = 32):
        super(Unet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_features = n_features

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        self.dc1 = double_conv(self.n_channels, self.n_features)
        self.dc2 = double_conv(self.n_features, self.n_features*2)
        self.dc3 = double_conv(self.n_features*2, self.n_features*4)
        self.dc4 = double_conv(self.n_features*4, self.n_features*8)
        self.dc5 = double_conv(self.n_features*12, self.n_features*4)
        self.dc6 = double_conv(self.n_features*6, self.n_features*2)
        self.dc7 = double_conv(self.n_features*3, self.n_features)
        self.mp = nn.MaxPool2d(2)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #binlinear?

        self.out = nn.Conv2d(self.n_features, self.n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.dc1(x)

        x2 = self.mp(x1)
        x2 = self.dc2(x2)

        x3 = self.mp(x2)
        x3 = self.dc3(x3)

        x4 = self.mp(x3)
        x4 = self.dc4(x4)

        x5 = self.up(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.dc5(x5)

        x6 = self.up(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.dc6(x6)

        x7 = self.up(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.dc7(x7)

        return self.out(x7)
    
# class Unet(nn.Module): #7M
#     def __init__(self, n_channels = 1, n_classes = 10, n_features = 32):
#         super(Unet, self).__init__()

#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.n_features = n_features

#         def double_conv(in_channels, out_channels):
#             return nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(out_channels),
#                 nn.ReLU(inplace=True),
#             )

#         self.dc1 = double_conv(self.n_channels, self.n_features)
#         self.dc2 = double_conv(self.n_features, self.n_features*2)
#         self.dc3 = double_conv(self.n_features*2, self.n_features*4)
#         self.dc4 = double_conv(self.n_features*4, self.n_features*8)

#         self.dc5 = double_conv(self.n_features*8, self.n_features*16)
#         self.dc6 = double_conv(self.n_features*24, self.n_features*8)

#         self.dc7 = double_conv(self.n_features*12, self.n_features*4)
#         self.dc8 = double_conv(self.n_features*6, self.n_features*2)
#         self.dc9 = double_conv(self.n_features*3, self.n_features)
#         self.mp = nn.MaxPool2d(2)

#         self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #binlinear?

#         self.out = nn.Conv2d(self.n_features, self.n_classes, kernel_size=1)

#     def forward(self, x):
#         x1 = self.dc1(x)

#         x2 = self.mp(x1)
#         x2 = self.dc2(x2)

#         x3 = self.mp(x2)
#         x3 = self.dc3(x3)

#         x4 = self.mp(x3)
#         x4 = self.dc4(x4)

#         x5 = self.mp(x4)
#         x5 = self.dc5(x5)

#         x6 = self.up(x5)
#         x6 = torch.cat([x6, x4], dim=1)
#         x6 = self.dc6(x6)

#         x7 = self.up(x6)
#         x7 = torch.cat([x7, x3], dim=1)
#         x7 = self.dc7(x7)

#         x8 = self.up(x7)
#         x8 = torch.cat([x8, x2], dim=1)
#         x8 = self.dc8(x8)

#         x9 = self.up(x8)
#         x9 = torch.cat([x9, x1], dim=1)
#         x9 = self.dc9(x9)

#         return self.out(x9)




class Block(nn.Module):
    def __init__(self, in_ch, out_ch,stride=1):
        super().__init__()
        self.bn1=nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride = stride, padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride = 1, padding=1,bias=False)
        # Si shortcut entre trucs de channels differents ou lors du changement de dimension spatiale
        if in_ch != out_ch or stride != 1:
            self.shortcut=nn.Sequential(nn.Conv2d(in_ch,out_ch,kernel_size=1,stride=stride,bias=False))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        # out=self.conv2(self.relu(self.bn2(self.conv1(self.relu(self.bn1(x))))))
        # out=out+self.shortcut(x)
        out=self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        out=out+self.shortcut(x)
        out=self.relu(out)
        return out

class ResNet(nn.Module): #8M
    def __init__(self,nombre_blocs):
        super().__init__()
        self.in_channels=64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.part1=self.couche(nombre_blocs[0], 1, 64)
        self.part2=self.couche(nombre_blocs[1], 2, 128)
        self.part3=self.couche(nombre_blocs[2], 2, 256)

        self.up_sampling = torch.nn.Upsample(scale_factor=4, mode='bilinear',align_corners=False)
        self.conv_fin = torch.nn.Conv2d(256, 1, 1, padding=0)

    def couche(self,nombre_blocs,stride,out_channels):
        strides=[stride]+[1]*(nombre_blocs-1)
        couches=[]
        for i in range(len(strides)):
            couches.append(Block(self.in_channels, out_channels, strides[i]))
            self.in_channels=out_channels
        return nn.Sequential(*couches)

    def forward(self, x):
        out=self.conv1(x)
        out=self.bn(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn(out)
        out=self.relu(out)
        
        out=self.part1(out)
        out=self.part2(out)
        out=self.part3(out)

        out=self.up_sampling(out)
        out=self.conv_fin(out)+x

        return out

class Discriminator(nn.Module): #693K
    def __init__(self):
        super(Discriminator, self).__init__()
        nc = 1
        ndf = 32
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace = True),
            #state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace = True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias = False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace = True),
            #state size. (ndf*8) x 4 x 4
        )
        self.main1=nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input, feature_matching = False):
        feature = self.main(input)
        out = self.main1(feature)
        if feature_matching == True:
            return feature
        else:
            return out
        
########################################################################################################################################################################################################################
########################################################################################################################################################################################################################
########################################################################################################################################################################################################################

class Degradation_paired(pl.LightningModule):
    def __init__(self, opt, prefix, isTrain):
        super().__init__()
        if isTrain:
            self.lr = opt.learning_rate
            # self.criterion = torch.nn.L1Loss()
            self.criterion = torch.nn.MSELoss()
            self.optimizer_class = torch.optim.Adam
            self.prefix=prefix
        if opt.gpu >=0:
            self.Device = torch.device("cuda:"+str(opt.gpu))
            self.gpu = opt.gpu
        else:
            self.Device = torch.device('cpu')

        print('Method: ', opt.method)
        if opt.net == "UNet":
            self.net = Unet(n_channels=1, n_classes=1, n_features=32)
        elif opt.net == 'ResNet':
            # self.net = ResNet([2,4,6]) #8M
            self.net = ResNet([1,2,3]) #3M
            # self.net = ResNet([1,1,1]) #1M
        else:
            sys.exit('Enter a valid architecture for paired degradation')
        self.initialize()


    def initialize(self):
        self.net.apply(gaussian_weights_init)


    def forward(self, x, opt):
        x=x.squeeze(-1) 
        out = self.net(x)
        return(out)


    def configure_optimizers(self):
        optim = self.optimizer_class(self.net.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        return optim


    def prepare_batch(self, batch):
        return batch['HR_image'][tio.DATA], batch['LR_image'][tio.DATA], batch['label'][tio.DATA]


    def infer_batch(self, batch):
        x,y, mask = self.prepare_batch(batch)
        x=x.squeeze(-1) 
        y=y.squeeze(-1)
        mask=mask.squeeze(-1)
        y_fake = self.net(x)
        assert y_fake.shape == y.shape
        return x, y, y_fake, mask


    def training_step(self, batch, batch_idx):
        x, y, y_fake, mask = self.infer_batch(batch)
        batch_size=x.shape[0]

        if batch_idx%1000==0:
            plt.figure()
            plt.suptitle('epoch: '+str(self.current_epoch)+' batch_idx: '+str(batch_idx)+'     '+self.prefix)
            plt.subplot(1,3,1)
            plt.imshow(x[0,0,:,:].cpu().detach().numpy(), cmap="gray")
            plt.subplot(1,3,2)
            plt.imshow(y_fake[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.subplot(1,3,3)
            plt.imshow(y[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.savefig('/home/claire/Nets_Reconstruction/Images_Test/'+'epoch-'+str(self.current_epoch)+'_batch_idx-'+str(batch_idx)+'.png')
            plt.close()

        y_fake_masked = y_fake*mask
        y_masked = y*mask
        pseudo_paired_loss = self.criterion(y_fake_masked, y_masked)

        self.log('Pseudo paired loss', pseudo_paired_loss, prog_bar=True)
        return pseudo_paired_loss


    def validation_step(self, batch, batch_idx):
        x, y, y_fake, mask = self.infer_batch(batch)

        plt.figure()
        plt.suptitle(self.prefix)
        plt.subplot(1,3,1)
        plt.imshow(x[0,0,:,:].cpu().detach().numpy(), cmap="gray")
        plt.subplot(1,3,2)
        plt.imshow(y_fake[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
        plt.subplot(1,3,3)
        plt.imshow(y[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
        plt.savefig('/home/claire/Nets_Reconstruction/Images_Test/'+'validation_epoch-'+str(self.current_epoch)+'.png')
        plt.close()
        
        loss = self.criterion(y_fake, y)
        self.log('validation_loss', loss)
        return loss
    







class Degradation_unpaired(pl.LightningModule):
    def __init__(self, opt, prefix, isTrain):
        super().__init__()
        if isTrain:
            self.lr = opt.learning_rate
            self.criterion = torch.nn.L1Loss()
            self.optimizer_class = torch.optim.Adam
            self.prefix=prefix
            self.random_size = opt.random_size
        if opt.gpu >=0:
            self.Device = torch.device("cuda:"+str(opt.gpu))
            self.gpu = opt.gpu
        else:
            self.Device = torch.device('cpu')
        print('Method: ', opt.method)
        if opt.net == "UNet":
            self.generator = Unet(n_channels=1, n_classes=1, n_features=32)
        elif opt.net == 'ResNet':
            #self.generator = ResNet([2,4,6]) #8M
            self.generator = ResNet([1,2,3]) #3M
            #self.generator = ResNet([1,1,1]) #1M
        else:
            sys.exit('Enter a valid architecture for unpaired degradation')
        self.discriminator = Discriminator()
        self.initialize()


    def initialize(self):
        self.generator.apply(gaussian_weights_init)
        self.discriminator.apply(gaussian_weights_init)


    def forward(self, x, opt):
        x=x.squeeze(-1) 
        out=self.generator(x)
        return out


    def configure_optimizers(self):
        optimizer_generator=self.optimizer_class(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        optimizer_discriminator=self.optimizer_class(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        return [optimizer_generator, optimizer_discriminator]

    def prepare_batch(self, batch):
        return batch['HR_image'][tio.DATA], batch['LR_image'][tio.DATA], batch['label'][tio.DATA]


    def infer_batch(self, batch):
        x,y, mask = self.prepare_batch(batch)

        x=x.squeeze(-1) 
        y=y.squeeze(-1)
        mask=mask.squeeze(-1)

        y_fake = self.generator(x)

        return x, y, y_fake, mask


    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y, y_fake, mask = self.infer_batch(batch)
        

        batch_size=x.shape[0]
        if batch_size==128:
            random_size = self.random_size
        else:
            random_size = batch_size//2

        self.true_label = torch.full((batch_size,), 1, dtype=torch.float)
        self.fake_label = torch.full((batch_size,), 0, dtype=torch.float)
        self.true_label=self.true_label.to(self.Device)
        self.fake_label=self.fake_label.to(self.Device)

        if batch_idx%1000==0:
            plt.figure()
            plt.suptitle('epoch: '+str(self.current_epoch)+' batch_idx: '+str(batch_idx)+'     '+self.prefix)
            plt.subplot(1,3,1)
            plt.imshow(x[0,0,:,:].cpu().detach().numpy(), cmap="gray")
            plt.subplot(1,3,2)
            plt.imshow(y_fake[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.subplot(1,3,3)
            plt.imshow(y[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")            
            plt.savefig('/home/claire/Nets_Reconstruction/Images_Test/'+'epoch-'+str(self.current_epoch)+'_batch_idx-'+str(batch_idx)+'.png')
            plt.close()
        
        if random_size!=0:
            loss_generator = self.backward_Generator(fake=y_fake, netD=self.discriminator)
            loss_discriminator = self.backward_Discriminator(netD=self.discriminator, real=y[random_size:,:,:,:], fake=y_fake[random_size:,:,:,:])
        else:
            loss_discriminator = self.backward_Discriminator(netD=self.discriminator, real=y, fake=y_fake)
            loss_generator = self.backward_Generator(fake=y_fake, netD=self.discriminator)
        
        if optimizer_idx == 0:
            self.log('Generator', loss_generator, prog_bar=True)
            return(loss_generator)

        elif optimizer_idx == 1:
            self.log('Discriminator', loss_discriminator, prog_bar=True)
            return(loss_discriminator)

            
    def backward_Discriminator(self, netD, real, fake):
        pred_fake = netD.forward(fake.detach())
        pred_real = netD.forward(real)
        loss_D = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all0 = torch.zeros_like(out_fake).cuda(self.gpu)
            all1 = torch.ones_like(out_real).cuda(self.gpu)
            ad_fake_loss = nn.functional.binary_cross_entropy_with_logits(out_fake, all0)
            ad_true_loss = nn.functional.binary_cross_entropy_with_logits(out_real, all1)
            loss_D += ad_true_loss + ad_fake_loss
        return loss_D
    
    def backward_Generator(self, fake, netD):
        outs_fake = netD.forward(fake)
        loss_G = 0
        for out_a in outs_fake:
            outputs_fake = torch.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
            loss_G += nn.functional.binary_cross_entropy_with_logits(outputs_fake, all_ones)
        return loss_G

    def validation_step(self, batch, batch_idx):
        x, y, y_fake, mask = self.infer_batch(batch)
        plt.figure()
        plt.suptitle(self.prefix)
        plt.subplot(1,3,1)
        plt.imshow(x[0,0,:,:].cpu().detach().numpy(), cmap="gray")
        plt.subplot(1,3,2)
        plt.imshow(y_fake[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
        plt.subplot(1,3,3)
        plt.imshow(y[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
        plt.savefig('/home/claire/Nets_Reconstruction/Images_Test/'+'validation_epoch-'+str(self.current_epoch)+'.png')
        plt.close()
        
        loss = self.criterion(y_fake, y)
        self.log('validation_loss', loss)
        return loss
