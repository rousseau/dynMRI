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
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import glob
import multiprocessing
import math

from pytorch_lightning.loggers import TensorBoardLogger, tensorboard
import argparse

def get_scheduler(optimizer, opts, cur_ep=-1):
    epoch=50
    def lambda_rule(ep):
        lr_l = 1.0 - max(0, ep - epoch//2) / float(epoch - epoch//2 + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=cur_ep)
    return scheduler


class LeakyReLUConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0, norm='None', sn=False):
    super(LeakyReLUConv2d, self).__init__()
    model = []
    model += [nn.ReflectionPad2d(padding)]
    # if sn:
    #     model += [spectral_norm(nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True))]
    # else:
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
    if 'norm' == 'Instance':
      model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    #elif == 'Group'
  def forward(self, x):
    return self.model(x)

class ReLUINSConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(ReLUINSConv2d, self).__init__()
    model = []
    model += [nn.ReflectionPad2d(padding)]
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
    model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
  def forward(self, x):
    return self.model(x)

class GaussianNoiseLayer(nn.Module):
    def __init__(self,):
        super(GaussianNoiseLayer, self).__init__()
    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable(torch.randn(x.size()).cuda(x.get_device()))
        return x + noise

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)


class MisINSResBlock(nn.Module):
    def conv3x3(self, dim_in, dim_out, stride=1):
        return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))
    def conv1x1(self, dim_in, dim_out):
        return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
    def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
        super(MisINSResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm2d(dim))
        self.conv2 = nn.Sequential(
            self.conv3x3(dim, dim, stride),
            nn.InstanceNorm2d(dim))
        self.blk1 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False))
        self.blk2 = nn.Sequential(
            self.conv1x1(dim + dim_extra, dim + dim_extra),
            nn.ReLU(inplace=False),
            self.conv1x1(dim + dim_extra, dim),
            nn.ReLU(inplace=False))
        model = []
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
        self.conv1.apply(gaussian_weights_init)
        self.conv2.apply(gaussian_weights_init)
        self.blk1.apply(gaussian_weights_init)
        self.blk2.apply(gaussian_weights_init)
    def forward(self, x, z):
        residual = x
        z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        o1 = self.conv1(x)
        o2 = self.blk1(torch.cat([o1, z_expand], dim=1))
        o3 = self.conv2(o2)
        out = self.blk2(torch.cat([o3, z_expand], dim=1))
        out += residual
        return out

class ReLUINSConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
        super(ReLUINSConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
        model += [LayerNorm(n_out)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
    def forward(self, x):
        return self.model(x)

class LayerNorm(nn.Module):
    def __init__(self, n_out, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.n_out = n_out
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(n_out, 1, 1))
            self.bias = nn.Parameter(torch.zeros(n_out, 1, 1))
        return
    def forward(self, x):
        normalized_shape = x.size()[1:]
        if self.affine:
            return F.layer_norm(x, normalized_shape, self.weight.expand(normalized_shape), self.bias.expand(normalized_shape))
        else:
            return F.layer_norm(x, normalized_shape)

class INSResBlock(nn.Module):
    def conv3x3(self, inplanes, out_planes, stride=1):
        return [nn.ReflectionPad2d(1), nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride)]
    def __init__(self, inplanes, planes, stride=1, dropout=0.0):
        super(INSResBlock, self).__init__()
        model = []
        model += self.conv3x3(inplanes, planes, stride)
        model += [nn.InstanceNorm2d(planes)]
        model += [nn.ReLU(inplace=True)]
        model += self.conv3x3(planes, planes)
        model += [nn.InstanceNorm2d(planes)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)
    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


#######################################################################################################################################################
class Encoder_content(nn.Module):
    def __init__(self, n_channels = 1, n_classes = 1, n_features = 64):
        super(Encoder_content, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_features = n_features

        encoder_LR=[]
        encoder_HR=[]
        encoder_share=[]

        encoder_LR+=[LeakyReLUConv2d(self.n_channels, self.n_features, kernel_size=7, stride=1, padding=3)]
        encoder_HR+=[LeakyReLUConv2d(self.n_channels, self.n_features, kernel_size=7, stride=1, padding=3)]

        for i in range(1,3):
            encoder_LR+=[ReLUINSConv2d(self.n_features, self.n_features * 2, kernel_size=3, stride=2, padding=1)]
            encoder_HR+=[ReLUINSConv2d(self.n_features, self.n_features * 2, kernel_size=3, stride=2, padding=1)]
            self.n_features=self.n_features*2

        for i in range(0,3):
            encoder_LR+=[INSResBlock(self.n_features, self.n_features)]
            encoder_HR+=[INSResBlock(self.n_features, self.n_features)]
        
        for i in range(0,1):
            encoder_share+=[INSResBlock(self.n_features, self.n_features)]
            encoder_share+=[GaussianNoiseLayer()]
            self.encoder_share=nn.Sequential(*encoder_share)
        
        self.encoder_LR=nn.Sequential(*encoder_LR)
        self.encoder_HR=nn.Sequential(*encoder_HR)

    def forward(self, x_LR, x_HR):
        out_HR=self.encoder_HR(x_HR)
        out_LR=self.encoder_LR(x_LR)
        out_LR=self.encoder_share(out_LR)
        out_HR=self.encoder_share(out_HR)
        return out_LR, out_HR

    def forward_HR(self, x_HR):
        out_HR=self.encoder_HR(x_HR)
        out_HR=self.encoder_share(out_HR)
        return out_HR

    def forward_LR(self, x_LR):
        out_LR=self.encoder_LR(x_LR)
        out_LR=self.encoder_share(out_LR)
        return out_LR


class Encoder_content_reduce(nn.Module):
    def __init__(self, n_channels = 1, n_classes = 1, n_features = 64):
        super(Encoder_content_reduce, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_features = n_features

        encoder_LR=[]
        encoder_HR=[]
        encoder_share=[]

        encoder_LR+=[LeakyReLUConv2d(self.n_channels, self.n_features, kernel_size=7, stride=1, padding=3)]
        encoder_HR+=[LeakyReLUConv2d(self.n_channels, self.n_features, kernel_size=7, stride=1, padding=3)]

        for i in range(1,3):
            encoder_LR+=[ReLUINSConv2d(self.n_features, self.n_features * 2, kernel_size=3, stride=2, padding=1)]
            encoder_HR+=[ReLUINSConv2d(self.n_features, self.n_features * 2, kernel_size=3, stride=2, padding=1)]
            self.n_features=self.n_features*2

        for i in range(0,2):
            encoder_LR+=[INSResBlock(self.n_features, self.n_features)]
            encoder_HR+=[INSResBlock(self.n_features, self.n_features)]
        
        for i in range(0,1):
            encoder_share+=[INSResBlock(self.n_features, self.n_features)]
            encoder_share+=[GaussianNoiseLayer()]
            self.encoder_share=nn.Sequential(*encoder_share)
        
        self.encoder_LR=nn.Sequential(*encoder_LR)
        self.encoder_HR=nn.Sequential(*encoder_HR)

    def forward(self, x_LR, x_HR):
        out_HR=self.encoder_HR(x_HR)
        out_LR=self.encoder_LR(x_LR)
        out_LR=self.encoder_share(out_LR)
        out_HR=self.encoder_share(out_HR)
        return out_LR, out_HR

    def forward_HR(self, x_HR):
        out_HR=self.encoder_HR(x_HR)
        out_HR=self.encoder_share(out_HR)
        return out_HR

    def forward_LR(self, x_LR):
        out_LR=self.encoder_LR(x_LR)
        out_LR=self.encoder_share(out_LR)
        return out_LR


class Encoder_content_reduceplus(nn.Module):
    def __init__(self, n_channels = 1, n_classes = 1, n_features = 64):
        super(Encoder_content_reduceplus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_features = n_features

        encoder_LR=[]
        encoder_HR=[]
        encoder_share=[]

        encoder_LR+=[LeakyReLUConv2d(self.n_channels, self.n_features, kernel_size=7, stride=1, padding=3)]
        encoder_HR+=[LeakyReLUConv2d(self.n_channels, self.n_features, kernel_size=7, stride=1, padding=3)]

        for i in range(1,3):
            encoder_LR+=[ReLUINSConv2d(self.n_features, self.n_features * 2, kernel_size=3, stride=2, padding=1)]
            encoder_HR+=[ReLUINSConv2d(self.n_features, self.n_features * 2, kernel_size=3, stride=2, padding=1)]
            self.n_features=self.n_features*2

        for i in range(0,1):
            encoder_LR+=[INSResBlock(self.n_features, self.n_features)]
            encoder_HR+=[INSResBlock(self.n_features, self.n_features)]
        
        for i in range(0,1):
            encoder_share+=[INSResBlock(self.n_features, self.n_features)]
            encoder_share+=[GaussianNoiseLayer()]
            self.encoder_share=nn.Sequential(*encoder_share)
        
        self.encoder_LR=nn.Sequential(*encoder_LR)
        self.encoder_HR=nn.Sequential(*encoder_HR)

    def forward(self, x_LR, x_HR):
        out_HR=self.encoder_HR(x_HR)
        out_LR=self.encoder_LR(x_LR)
        out_LR=self.encoder_share(out_LR)
        out_HR=self.encoder_share(out_HR)
        return out_LR, out_HR

    def forward_HR(self, x_HR):
        out_HR=self.encoder_HR(x_HR)
        out_HR=self.encoder_share(out_HR)
        return out_HR

    def forward_LR(self, x_LR):
        out_LR=self.encoder_LR(x_LR)
        out_LR=self.encoder_share(out_LR)
        return out_LR




class Encoder_representation(nn.Module):
    def __init__(self, n_channels = 1, n_features = 64, dim=8):
        super(Encoder_representation, self).__init__()
        self.n_channels = n_channels
        self.n_features = n_features
        self.dim=dim

        self.model=nn.Sequential(
            nn.Conv2d(self.n_channels, self.n_features, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(self.n_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.n_features, self.n_features*2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.n_features*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.n_features*2, self.n_features*4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.n_features*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.n_features*4, self.n_features*4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.n_features*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.n_features*4, self.n_features*4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.n_features*4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.n_features*4, self.dim, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        out=self.model(x)
        out=out.squeeze(-1).squeeze(-1)
        return(out)


class Encoder_pose(nn.Module):
    def __init__(self, dim=5, n_features=64, n_channels=1):
        super(Encoder_pose, self).__init__()
        self.n_channels = n_channels
        self.n_features = n_features
        self.dim = dim

        self.model=nn.Sequential(
            nn.Conv2d(self.n_channels, self.n_features, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm2d(self.n_features),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(self.n_features, self.n_features*2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.n_features*2),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(self.n_features*2, self.n_features*4, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.n_features*4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(self.n_features*4, self.n_features*4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.n_features*4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(self.n_features*4, self.n_features*4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.n_features*4),
            nn.LeakyReLU(0.2,inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.n_features*4, self.dim, kernel_size=1, stride=1, padding=0)
        )
    def forward(self, x):
        out=self.model(x)
        out=out.squeeze(-1).squeeze(-1)
        return(out)


class Generator_feature(nn.Module):
    def __init__(self, n_channels = 256, n_classes = 1, n_features = 32):
        super(Generator_feature, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_features = n_features
        self.tch_add=256

        def multi_layer_perceptron():
            return nn.Sequential(
                nn.Linear(8,256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.n_channels*4))

        self.mlp=multi_layer_perceptron()
        self.resbloc1=MisINSResBlock(self.n_channels, self.n_channels)
        self.resbloc2=MisINSResBlock(self.n_channels, self.n_channels)
        self.resbloc3=MisINSResBlock(self.n_channels, self.n_channels)
        self.resbloc4=MisINSResBlock(self.n_channels, self.n_channels)
        self.up=nn.Sequential(
            ReLUINSConvTranspose2d(self.n_channels, self.n_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            ReLUINSConvTranspose2d(self.n_channels//2, self.n_channels//4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(self.n_channels//4, self.n_classes, kernel_size=1, stride=1, padding=0), 
            #nn.Tanh()
        )

    def forward(self, x, z):
        z=self.mlp(z)
        z1, z2, z3, z4 = torch.split(z, self.tch_add, dim=1)
        z1, z2, z3, z4 = z1.contiguous(), z2.contiguous(), z3.contiguous(), z4.contiguous()
        out1 = self.resbloc1(x, z1)
        out2 = self.resbloc2(out1, z2)
        out3 = self.resbloc3(out2, z3)
        out4 = self.resbloc4(out3, z4)
        out = self.up(out4)
        return out


class Generator_feature_reduce(nn.Module):
    def __init__(self, n_channels = 256, n_classes = 1, n_features = 32):
        super(Generator_feature_reduce, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_features = n_features
        self.tch_add=256

        def multi_layer_perceptron():
            return nn.Sequential(
                nn.Linear(8,256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.n_channels*3))

        self.mlp=multi_layer_perceptron()
        self.resbloc1=MisINSResBlock(self.n_channels, self.n_channels)
        self.resbloc2=MisINSResBlock(self.n_channels, self.n_channels)
        self.resbloc3=MisINSResBlock(self.n_channels, self.n_channels)
        self.up=nn.Sequential(
            ReLUINSConvTranspose2d(self.n_channels, self.n_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            ReLUINSConvTranspose2d(self.n_channels//2, self.n_channels//4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(self.n_channels//4, self.n_classes, kernel_size=1, stride=1, padding=0), 
            #nn.Tanh()
        )

    def forward(self, x, z):
        z=self.mlp(z)
        z1, z2, z3 = torch.split(z, self.tch_add, dim=1)
        z1, z2, z3 = z1.contiguous(), z2.contiguous(), z3.contiguous()
        out1 = self.resbloc1(x, z1)
        out2 = self.resbloc2(out1, z2)
        out3 = self.resbloc3(out2, z3)
        out = self.up(out3)
        return out




class Generator_feature_reduceplus(nn.Module):
    def __init__(self, n_channels = 256, n_classes = 1, n_features = 32):
        super(Generator_feature_reduceplus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_features = n_features
        self.tch_add=256

        def multi_layer_perceptron():
            return nn.Sequential(
                nn.Linear(8,256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.n_channels*2))

        self.mlp=multi_layer_perceptron()
        self.resbloc1=MisINSResBlock(self.n_channels, self.n_channels)
        self.resbloc2=MisINSResBlock(self.n_channels, self.n_channels)
        self.up=nn.Sequential(
            ReLUINSConvTranspose2d(self.n_channels, self.n_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            ReLUINSConvTranspose2d(self.n_channels//2, self.n_channels//4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(self.n_channels//4, self.n_classes, kernel_size=1, stride=1, padding=0), 
            #nn.Tanh()
        )

    def forward(self, x, z):
        z=self.mlp(z)
        z1, z2 = torch.split(z, self.tch_add, dim=1)
        z1, z2 = z1.contiguous(), z2.contiguous()
        out1 = self.resbloc1(x, z1)
        out2 = self.resbloc2(out1, z2)
        out = self.up(out2)
        return out





class Generator_dynamic(nn.Module):
    def __init__(self, n_channels = 256, n_classes = 1, n_features = 32):
        super(Generator_dynamic, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_features = n_features
        self.tch_add=256

        def multi_layer_perceptron():
            return nn.Sequential(
                nn.Linear(8,256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.n_channels*2))

        self.mlp=multi_layer_perceptron()
        self.resbloc1=MisINSResBlock(self.n_channels+5, self.n_channels)
        self.resbloc2=MisINSResBlock(self.n_channels+5, self.n_channels)
        self.up=nn.Sequential(
            ReLUINSConvTranspose2d(self.n_channels+5, self.n_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            ReLUINSConvTranspose2d(self.n_channels//2, self.n_channels//4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(self.n_channels//4, self.n_classes, kernel_size=1, stride=1, padding=0), 
        )

    def forward(self, anatomy, modality, pose):
        anatomy=torch.cat([anatomy, pose.view(pose.size(0), pose.size(1), 1, 1).expand(pose.size(0), pose.size(1), anatomy.size(2), anatomy.size(3))],dim=1)
        z=self.mlp(modality)
        z1, z2 = torch.split(z, self.tch_add, dim=1)
        z1, z2 = z1.contiguous(), z2.contiguous()
        out1 = self.resbloc1(anatomy, z1)
        out2 = self.resbloc2(out1, z2)
        out = self.up(out2)
        return out






class Discriminateur_content(nn.Module):
    def __init__(self, n_features = 256):
        super(Discriminateur_content, self).__init__()
        self.n_features=n_features
        self.model=nn.Sequential(
            LeakyReLUConv2d(self.n_features, self.n_features, kernel_size=7, stride=1, padding=1, norm='Instance'),
            LeakyReLUConv2d(self.n_features, self.n_features, kernel_size=7, stride=1, padding=1, norm='Instance'),
            LeakyReLUConv2d(self.n_features, self.n_features, kernel_size=7, stride=1, padding=1, norm='Instance'),
            LeakyReLUConv2d(self.n_features, self.n_features, kernel_size=4, stride=1, padding=0),
            nn.Conv2d(self.n_features, 1, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        out=self.model(x)
        out=out.view(-1)
        outs = []
        outs.append(out)
        return(outs)

class Discriminateur_content_reduce(nn.Module):
    def __init__(self, n_features = 256):
        super(Discriminateur_content_reduce, self).__init__()
        self.n_features=n_features
        self.model=nn.Sequential(
            LeakyReLUConv2d(self.n_features, self.n_features, kernel_size=7, stride=1, padding=1, norm='Instance'),
            LeakyReLUConv2d(self.n_features, self.n_features, kernel_size=4, stride=1, padding=0),
            nn.Conv2d(self.n_features, 1, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        out=self.model(x)
        out=out.view(-1)
        outs = []
        outs.append(out)
        return(outs)

class Discriminateur(nn.Module):
    def __init__(self, n_channels = 1, n_layers = 6, n_features = 64):
        super(Discriminateur, self).__init__()
        self.n_features=n_features
        self.n_channels=n_channels
        self.n_layers=n_layers
        model=[]
        model+=[LeakyReLUConv2d(self.n_channels, self.n_features, kernel_size=3, stride=2, padding=1, norm='None')]
        features=self.n_features
        for i in range(1,self.n_layers):
            model += [LeakyReLUConv2d(features, features * 2, kernel_size=3, stride=2, padding=1, norm='None')]
            features*=2        
        model += [nn.Conv2d(features, 1, kernel_size=1, stride=1, padding=0)]
        self.model=nn.Sequential(*model)
        
    def forward(self, x):
        out=self.model(x)
        out=out.view(-1)
        outs = []
        outs.append(out)
        return(outs)

class Discriminateur_reduce(nn.Module):
    def __init__(self, n_channels = 1, n_layers = 5, n_features = 64):
        super(Discriminateur_reduce, self).__init__()
        # initialement, n_layers=6
        self.n_features=n_features
        self.n_channels=n_channels
        self.n_layers=n_layers
        model=[]
        model+=[LeakyReLUConv2d(self.n_channels, self.n_features, kernel_size=3, stride=2, padding=1, norm='None')]
        features=self.n_features
        for i in range(1,self.n_layers):
            model += [LeakyReLUConv2d(features, features * 2, kernel_size=3, stride=2, padding=1, norm='None')]
            features*=2        
        model += [nn.Conv2d(features, 1, kernel_size=1, stride=1, padding=0)]
        self.model=nn.Sequential(*model)
        
    def forward(self, x):
        out=self.model(x)
        out=out.view(-1)
        outs = []
        outs.append(out)
        return(outs)


class Discriminateur_reduceplus(nn.Module):
    def __init__(self, n_channels = 1, n_layers = 4, n_features = 64):
        super(Discriminateur_reduceplus, self).__init__()
        # initialement, n_layers=6
        self.n_features=n_features
        self.n_channels=n_channels
        self.n_layers=n_layers
        model=[]
        model+=[LeakyReLUConv2d(self.n_channels, self.n_features, kernel_size=3, stride=2, padding=1, norm='None')]
        features=self.n_features
        for i in range(1,self.n_layers):
            model += [LeakyReLUConv2d(features, features * 2, kernel_size=3, stride=2, padding=1, norm='None')]
            features*=2        
        model += [nn.Conv2d(features, 1, kernel_size=1, stride=1, padding=0)]
        self.model=nn.Sequential(*model)
        
    def forward(self, x):
        out=self.model(x)
        out=out.view(-1)
        outs = []
        outs.append(out)
        return(outs)



class DRIT(pl.LightningModule):
    def __init__(self, criterion, learning_rate, optimizer_class, dataset, prefix, segmentation, gpu, mode, reduce=False,  n_channels = 1, n_classes = 1, n_features = 32):
        super().__init__()
        self.lr = learning_rate
        self.lr_dcontent = self.lr / 2.5
        self.criterion = torch.nn.L1Loss()
        self.optimizer_class = optimizer_class
        self.dataset = dataset
        self.prefix=prefix
        self.segmentation=segmentation
        self.gpu=gpu
        self.mode=mode

        if reduce:
            self.encode_content=Encoder_content_reduceplus()
            self.encode_HR=Encoder_representation()
            self.encode_LR=Encoder_representation()
            self.generator_HR=Generator_feature_reduceplus()
            self.generator_LR=Generator_feature_reduceplus()
            self.discriminator_content=Discriminateur_content_reduce()
            self.discriminator_LR=Discriminateur_reduceplus()
            self.discriminator_HR=Discriminateur_reduceplus()
            # self.encode_pose_LR=Encoder_pose()
            # self.encode_pose_HR=Encoder_pose()

        else:
            self.encode_content=Encoder_content()
            self.encode_HR=Encoder_representation()
            self.encode_LR=Encoder_representation()
            self.generator_HR=Generator_feature()
            self.generator_LR=Generator_feature()
            self.discriminator_content=Discriminateur_content()
            self.discriminator_LR=Discriminateur()
            #self.discriminator_LR1=Discriminateur()
            self.discriminator_HR=Discriminateur()
            #self.discriminator_HR1=Discriminateur()


        self.lambda_anatContent=1
        self.lambda_selfReconstruction=10
        self.lambda_crossReconstruction=10


    def initialize(self):
        self.encode_content.apply(gaussian_weights_init)
        self.encode_HR.apply(gaussian_weights_init)
        self.encode_LR.apply(gaussian_weights_init)
        self.generator_HR.apply(gaussian_weights_init)
        self.generator_LR.apply(gaussian_weights_init)
        self.discriminator_content.apply(gaussian_weights_init)
        self.discriminator_LR.apply(gaussian_weights_init)
        self.discriminator_HR.apply(gaussian_weights_init)

    def forward(self, x, y):
        if self.mode=='reconstruction':
            x=x.to(torch.float)
            y=y.to(torch.float32)
            z_HR=self.encode_HR(y)
            content=self.encode_content.forward_LR(x)
            out=self.generator_HR(content, z_HR)
            return(out)
        elif self.mode=='degradation':
            z_LR=self.encode_LR(y)
            content=self.encode_content.forward_HR(x)
            out=self.generator_LR(content, z_LR)
            # return(out,content)
            return(out)
        else:
            sys.exit("Précisez un mode valide pour l'inférence: reconstruction ou degradation")


    def configure_optimizers(self):
        optimizer_encode_content=self.optimizer_class(self.encode_content.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        optimizer_encode_HR=self.optimizer_class(self.encode_HR.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        optimizer_encode_LR=self.optimizer_class(self.encode_LR.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        optimizer_generator_HR=self.optimizer_class(self.generator_HR.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        optimizer_generator_LR=self.optimizer_class(self.generator_LR.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        optimizer_discriminator_content=self.optimizer_class(self.discriminator_content.parameters(), lr=self.lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)
        optimizer_discriminator_LR=self.optimizer_class(self.discriminator_LR.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        optimizer_discriminator_HR=self.optimizer_class(self.discriminator_HR.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        return [optimizer_encode_content, optimizer_encode_HR, optimizer_encode_LR, optimizer_generator_HR, optimizer_generator_LR, optimizer_discriminator_content, optimizer_discriminator_LR, optimizer_discriminator_HR],[]#, optimizer_encode_pose_LR, optimizer_encode_pose_HR],[]


    def prepare_batch(self, batch):
        return batch['Dynamic_1'][tio.DATA], batch['Dynamic_2'][tio.DATA], batch['Static_1'][tio.DATA], batch['Static_2'][tio.DATA], batch['label'][tio.DATA] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    def infer_batch(self, batch):
        x_1, x_2, y_1, y_2, mask = self.prepare_batch(batch)
        x_1=x_1.squeeze(-1) 
        x_2=x_2.squeeze(-1)
        y_1=y_1.squeeze(-1)
        y_2=y_2.squeeze(-1)

        #disentanglement
        modality_LR1=self.encode_LR(x_1)
        modality_LR2=self.encode_LR(x_2)
        modality_HR1=self.encode_HR(y_1)
        modality_HR2=self.encode_HR(y_2)
        modality_random = self.get_z_random(x_1.shape[0], 8, 'gauss')
        anatomy_LR1,anatomy_HR1=self.encode_content.forward(x_1,y_1)
        anatomy_LR2,anatomy_HR2=self.encode_content.forward(x_2,y_2) # pour la temporal consistency entre les contenus

        #generation
        fake_HR=self.generator_HR(anatomy_LR1, modality_HR1)
        fake_LR=self.generator_LR(anatomy_HR1, modality_LR1)
        fake_HRHR=self.generator_HR(anatomy_HR1, modality_HR1)
        fake_LRLR=self.generator_LR(anatomy_LR1, modality_LR1)
        fake_LR_random=self.generator_LR(anatomy_HR1, modality_random)
        fake_HR_random=self.generator_HR(anatomy_LR1, modality_random)

        fake_LR_t1=self.generator_LR(anatomy_LR2, modality_LR2)
        fake_LR_t2=self.generator_LR(anatomy_LR1, modality_LR1)
        fake_HR_t1=self.generator_HR(anatomy_HR2, modality_HR2)
        fake_HR_t2=self.generator_HR(anatomy_HR1, modality_HR1)


        #disentanglement
        modality_fake_LR=self.encode_LR(fake_LR)
        modality_fake_HR=self.encode_HR(fake_HR)
        anatomy_fake_LR, anatomy_fake_HR=self.encode_content.forward(fake_LR, fake_HR)

        #generation
        est_HR=self.generator_HR(anatomy_fake_LR, modality_fake_HR)
        est_LR=self.generator_LR(anatomy_fake_HR, modality_fake_LR)
        z_random_LR=self.encode_LR(fake_LR_random)
        z_random_HR=self.encode_HR(fake_HR_random)

        return x_1, x_2, y_1, y_2, fake_LR, fake_HR, fake_LRLR, fake_HRHR, est_LR, est_HR, modality_LR1, modality_HR1, anatomy_LR1, anatomy_HR1, modality_fake_LR, modality_fake_HR, anatomy_fake_LR, anatomy_fake_HR, modality_random, fake_LR_random, fake_HR_random, z_random_LR, z_random_HR, fake_LR_t1, fake_LR_t2, fake_HR_t1, fake_HR_t2, anatomy_LR2, anatomy_HR2





    def training_step(self, batch, batch_idx, optimizer_idx):
        x_1, x_2, y_1, y_2, fake_LR, fake_HR, fake_LRLR, fake_HRHR, est_LR, est_HR, modality_LR1, modality_HR1, anatomy_LR1, anatomy_HR1, modality_fake_LR, modality_fake_HR, anatomy_fake_LR, anatomy_fake_HR, modality_random, fake_LR_random, fake_HR_random, z_random_LR, z_random_HR, fake_LR_t1, fake_LR_t2, fake_HR_t1, fake_HR_t2, anatomy_LR2, anatomy_HR2 = self.infer_batch(batch)

        batch_size=x_1.shape[0]
        self.true_label = torch.full((batch_size,), 1, dtype=torch.float)
        self.fake_label = torch.full((batch_size,), 0, dtype=torch.float)
        self.true_label=self.true_label.to(torch.device("cuda:"+str(self.gpu)))
        self.fake_label=self.fake_label.to(torch.device("cuda:"+str(self.gpu)))

        if batch_idx%1000==0:
            plt.figure()
            plt.suptitle('epoch: '+str(self.current_epoch)+' batch_idx: '+str(batch_idx)+'     '+self.prefix)
            plt.subplot(4,3,1)
            plt.imshow(x_1[0,0,:,:].cpu().detach().numpy(), cmap="gray")
            plt.subplot(4,3,2)
            plt.imshow(est_LR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.subplot(4,3,3)
            test=fake_HR
            plt.imshow(test[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.subplot(4,3,4)
            plt.imshow(x_2[0,0,:,:].clone().cpu().detach().numpy(), cmap="gray")
            plt.subplot(4,3,5)
            plt.imshow(fake_LR_t1[0,0,:,:].clone().cpu().detach().numpy().astype(float), cmap="gray")
            plt.subplot(4,3,6)
            plt.imshow(fake_LR_t2[0,0,:,:].clone().cpu().detach().numpy().astype(float), cmap="gray")
            
            plt.subplot(4,3,7)
            plt.imshow(y_1[0,0,:,:].cpu().detach().numpy(), cmap="gray")
            plt.subplot(4,3,8)
            plt.imshow(est_HR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.subplot(4,3,9)
            plt.imshow(fake_LR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.subplot(4,3,10)
            plt.imshow(y_2[0,0,:,:].clone().cpu().detach().numpy(), cmap="gray")
            plt.subplot(4,3,11)
            plt.imshow(fake_HR_t1[0,0,:,:].clone().cpu().detach().numpy().astype(float), cmap="gray")
            plt.subplot(4,3,12)
            plt.imshow(fake_HR_t2[0,0,:,:].clone().cpu().detach().numpy().astype(float), cmap="gray")
            
            plt.colorbar()
            plt.savefig('/home/claire/Nets_Reconstruction/Images_Test/'+'epoch-'+str(self.current_epoch)+'_batch_idx-'+str(batch_idx)+'.png')
            plt.close()
        
        loss_G_LR, loss_G_HR=self.backward_EG(x_1, y_1, fake_LR, fake_HR, fake_LRLR, fake_HRHR, est_LR, est_HR, modality_LR1, modality_HR1, anatomy_LR1, anatomy_HR1, modality_fake_LR, modality_fake_HR, anatomy_fake_LR, anatomy_fake_HR)
        loss_z_LR, loss_z_HR=self.backward_G_alone(fake_LR_random, fake_HR_random, z_random_LR, z_random_HR, modality_random)
        loss_D_content=self.backward_D_content(anatomy_LR1, anatomy_HR1)
        loss_D1_LR = self.backward_D_domain(self.discriminator_LR, x_1, fake_LR)
        loss_D1_HR = self.backward_D_domain(self.discriminator_HR, y_1, fake_HR)
        LR_temporal_consistency_content, HR_temporal_consistency_content = self.temporal_consistency(x_1, fake_LR_t1, x_2, fake_LR_t2, y_1, fake_HR_t1, y_2, fake_HR_t2, anatomy_LR1, anatomy_LR2, anatomy_HR1, anatomy_HR2)

        if optimizer_idx == 0: #encoder content
            self.log('train_loss_DRITpp_encoder_content', loss_G_LR+loss_G_HR+loss_z_LR+loss_z_HR+LR_temporal_consistency_content+HR_temporal_consistency_content, prog_bar=True)
            return (loss_G_LR+loss_G_HR+loss_z_LR+loss_z_HR+LR_temporal_consistency_content+HR_temporal_consistency_content)

        elif optimizer_idx==1: #encoder HR
            self.log('train_loss_DRITpp_encoderHR', loss_G_HR, prog_bar=True)
            return (loss_G_HR)

        elif optimizer_idx==2: #encoder LR
            self.log('train_loss_DRITpp_encoderLR', loss_G_LR, prog_bar=True)
            return (loss_G_LR)

        elif optimizer_idx==3: #generator HR
            self.log('train_loss_DRITpp_generatorHR', loss_G_HR+loss_z_HR, prog_bar=True)
            return (loss_G_HR+loss_z_HR)

        elif optimizer_idx==4: #generator LR
            self.log('train_loss_DRITpp_generatorLR', loss_G_LR+loss_z_LR, prog_bar=True)
            return (loss_G_LR+loss_z_LR)

        elif optimizer_idx==5: #discriminator content
            nn.utils.clip_grad_norm_(self.discriminator_content.parameters(), 5)
            self.log('train_loss_DRITpp_discriminator_content', loss_D_content, prog_bar=True)
            return (loss_D_content)

        elif optimizer_idx==6: #discriminateur LR
            self.log('train_loss_DRITpp_discriminator_domainLR', loss_D1_LR, prog_bar=True)
            return (loss_D1_LR)

        elif optimizer_idx==7: #discriminator HR
            self.log('train_loss_DRITpp_discriminator_domainHR', loss_D1_HR, prog_bar=True)
            return(loss_D1_HR)


    def temporal_consistency(self, LR_t1, fake_LR_t1, LR_t2, fake_LR_t2, HR_t1, fake_HR_t1, HR_t2, fake_HR_t2, anatomy_LR_t1, anatomy_LR_t2, anatomy_HR_t1, anatomy_HR_t2):
        loss_LR_content=torch.nn.L1Loss()(anatomy_LR_t1, anatomy_LR_t2)
        loss_HR_content=torch.nn.L1Loss()(anatomy_HR_t1, anatomy_HR_t2)
        LR_temporal_consistency_content=(loss_LR_content)*10
        HR_temporal_consistency_content=(loss_HR_content)*10
        return(LR_temporal_consistency_content, HR_temporal_consistency_content)



    def backward_D_domain(self, netD, real, fake):
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

    def backward_D_content(self, imageA, imageB):
        pred_fake = self.discriminator_content.forward(imageA.detach())
        pred_real = self.discriminator_content.forward(imageB.detach())
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all1 = torch.ones((out_real.size(0))).cuda(self.gpu)
            all0 = torch.zeros((out_fake.size(0))).cuda(self.gpu)
            ad_true_loss = nn.functional.binary_cross_entropy_with_logits(out_real, all1)
            ad_fake_loss = nn.functional.binary_cross_entropy_with_logits(out_fake, all0)
        loss_D = ad_true_loss + ad_fake_loss
        return loss_D

    def backward_G_GAN_content(self, z_content):
        outs = self.discriminator_content.forward(z_content)
        for out in outs:
            outputs_fake = torch.sigmoid(out)
            all_half = 0.5*torch.ones((outputs_fake.size(0))).cuda(self.gpu)
            ad_loss = nn.functional.binary_cross_entropy_with_logits(outputs_fake, all_half)
        return ad_loss

    def backward_G_GAN(self, fake, netD):
        outs_fake = netD.forward(fake)
        loss_G = 0
        for out_a in outs_fake:
            outputs_fake = torch.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
            loss_G += nn.functional.binary_cross_entropy_with_logits(outputs_fake, all_ones)
        return loss_G

    def backward_EG(self, x, y, fake_LR, fake_HR, fake_LRLR, fake_HRHR, est_LR, est_HR, z_LR, z_HR, content_LR, content_HR, z_fake_LR, z_fake_HR, content_fake_LR, content_fake_HR):
        # content Ladv for generator
        loss_G_GAN_LRcontent = self.backward_G_GAN_content(content_LR)
        loss_G_GAN_HRcontent = self.backward_G_GAN_content(content_HR)

        # Ladv for generator
        loss_G_GAN_LR = self.backward_G_GAN(fake_LR, self.discriminator_LR)
        loss_G_GAN_HR = self.backward_G_GAN(fake_HR, self.discriminator_HR)

        # KL loss - z_a
        #loss_kl_za_LR = self._l2_regularize(z_fake_LR) * 0.01
        loss_kl_za_LR = self._l2_regularize(z_LR) * 0.01 # torch.Size([128, 8])
        loss_kl_za_HR = self._l2_regularize(z_HR) * 0.01 # torch.Size([128, 8])

        # KL loss - z_c
        loss_kl_zc_LR = self._l2_regularize(content_LR) * 0.01
        loss_kl_zc_HR = self._l2_regularize(content_HR) * 0.01

        # cross cycle consistency loss
        loss_G_L1_LR = torch.nn.L1Loss()(est_LR, x) * 10
        loss_G_L1_HR = torch.nn.L1Loss()(est_HR, y) * 10
        loss_G_L1_LRLR = torch.nn.L1Loss()(fake_LRLR, x) * 10
        loss_G_L1_HRHR = torch.nn.L1Loss()(fake_HRHR, y) * 10

        loss_G_LR = loss_G_GAN_LR + \
                loss_G_GAN_LRcontent + \
                loss_G_L1_LRLR + \
                loss_G_L1_LR + \
                loss_kl_zc_LR + \
                loss_kl_za_LR

        loss_G_HR = loss_G_GAN_HR + \
                loss_G_GAN_HRcontent + \
                loss_G_L1_HRHR + \
                loss_G_L1_HR + \
                loss_kl_zc_HR + \
                loss_kl_za_HR

        return(loss_G_LR, loss_G_HR)

    def backward_G_alone(self,fake_LR_random, fake_HR_random, z_random_LR, z_random_HR, z_random):        
        # latent regression loss
        loss_z_L1_LR = torch.mean(torch.abs(z_random_LR - z_random)) * 10
        loss_z_L1_HR = torch.mean(torch.abs(z_random_HR - z_random)) * 10

        loss_z_L1 = loss_z_L1_LR + loss_z_L1_HR #+ loss_G_GAN2_LR + loss_G_GAN2_HR
        loss_z_LR= loss_z_L1_LR #+ loss_G_GAN2_LR 
        loss_z_HR= loss_z_L1_HR #+ loss_G_GAN2_HR
        return loss_z_LR, loss_z_HR

    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss
    
    def get_z_random(self, batchSize, nz, random_type='gauss'):
        z = torch.randn(batchSize, nz).cuda(self.gpu)
        return z

    def validation_step(self, batch, batch_idx):
        x_1, x_2, y_1, y_2, fake_LR, fake_HR, fake_LRLR, fake_HRHR, est_LR, est_HR, modality_LR1, modality_HR1, anatomy_LR1, anatomy_HR1, modality_fake_LR, modality_fake_HR, anatomy_fake_LR, anatomy_fake_HR, modality_random, fake_LR_random, fake_HR_random, z_random_LR, z_random_HR, fake_LR_t1, fake_LR_t2, fake_HR_t1, fake_HR_t2, anatomy_LR2, anatomy_HR2 = self.infer_batch(batch) # y_hat, x, y, mask = self.infer_batch(batch)  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        plt.figure()
        plt.suptitle(self.prefix)
        plt.subplot(2,3,1)
        plt.imshow(x_1[0,0,:,:].cpu().detach().numpy(), cmap="gray")
        plt.subplot(2,3,2)
        plt.imshow(est_LR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
        plt.subplot(2,3,3)
        test=fake_HR
        plt.imshow(test[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
        plt.subplot(2,3,4)
        plt.imshow(y_1[0,0,:,:].cpu().detach().numpy(), cmap="gray")
        plt.subplot(2,3,5)
        plt.imshow(est_HR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
        plt.subplot(2,3,6)
        plt.imshow(fake_LR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
        plt.savefig('/home/claire/Nets_Reconstruction/Images_Test/'+'validation_epoch-'+str(self.current_epoch)+'.png')
        plt.close()
        
        loss = self.criterion(est_HR, y_1)
        self.log('val_loss', loss)
        return loss
