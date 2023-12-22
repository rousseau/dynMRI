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
import functools
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
import monai

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
        out = out + residual
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
            #nn.AdaptiveAvgPool2d(1),
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

#########################################################################################################################################################################
######################### AdaAttN #######################################################################################################################################
#########################################################################################################################################################################
class AdaAttN(nn.Module):
    '''
    Article: https://arxiv.org/pdf/1903.07291.pdf

    Details: Similar to the Batch Normalization, the activation is normalized in the channelwise manner and then modulated with learned scale and bias.

    Code original de cette partie: https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py
    '''
    def __init__(self, in_channels, out_channels):
        super(AdaAttN, self).__init__()
        
        self.bn = nn.BatchNorm2d(in_channels, affine=False)
        self.activation = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1), 
            nn.ReLU()
        )
        self.predict_beta = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride = 1)
        self.predict_gamma = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride = 1)

        
    def forward(self, style, content):
        style_normalized = self.bn(style)

        activation = self.activation(content)
        gamma = self.predict_gamma(activation)
        beta = self.predict_beta(activation)

        out = style_normalized * (1 + gamma) + beta
        return(out)    



class AdaAttNResBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(AdaAttNResBlock, self).__init__()

        self.adaattn=AdaAttN(in_channels=in_planes, out_channels=out_planes)
        def mlp():
                '''
                Transforme le vecteur de style en une feature map: mlp puis reshape
                '''
                return nn.Sequential(
                    nn.Linear(8, out_planes//2),
                    nn.LeakyReLU(negative_slope=0.3, inplace=True), 
                    nn.Linear(out_planes//2, out_planes)
                )

        self.mlp = mlp()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=2e-1, inplace=True)

    def forward(self, x, z):
        z = self.mlp(z)
        z = torch.reshape(z.unsqueeze(-1), [z.shape[0], x.shape[1], x.shape[2], x.shape[3]])
        

        skip_connection = z
        z = self.adaattn(style=z, content=x)
        z = self.lrelu(z)
        z = self.conv1(z)
        z = self.adaattn(style = z, content = x)
        z = self.lrelu(z)
        z = self.conv2(z)

        out = z + skip_connection
        return out

class Generator_AdaAttN(nn.Module):
    def __init__(self, latent_dim=8, w_dim=256, n_channels = 256, n_classes = 1, n_features = 32):
        super(Generator_AdaAttN, self).__init__()
        self.latent_dim=latent_dim
        self.w_dim=w_dim
        self.n_channels=n_channels
        self.n_classes=n_classes
        self.n_features=n_features
        self.tch_add=10

        def multi_layer_perceptron():
            return nn.Sequential(
                nn.Linear(self.latent_dim,256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.n_channels*10*2)
                )

        self.mlp=multi_layer_perceptron()
        self.resbloc1=AdaAttNResBlock(self.n_channels, self.n_channels)
        self.resbloc2=AdaAttNResBlock(self.n_channels, self.n_channels)
        self.up=nn.Sequential(
            ReLUINSConvTranspose2d(self.n_channels, self.n_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            ReLUINSConvTranspose2d(self.n_channels//2, self.n_channels//4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(self.n_channels//4, self.n_classes, kernel_size=1, stride=1, padding=0), 
            #nn.Tanh()
        )


    def forward(self, x, z):
        z=self.mlp(z)
        z=z.view((x.shape[0], x.shape[1],-1))
        z1, z2 = torch.split(z, self.tch_add, dim=2)
        z1, z2 = z1.contiguous(), z2.contiguous()
        out1 = self.resbloc1(x, z1)
        out2 = self.resbloc2(out1, z2)
        out = self.up(out2)
        return out


#########################################################################################################################################################################
######################### SPADE #########################################################################################################################################
#########################################################################################################################################################################
class SPADE(nn.Module):
    '''
    Article: https://arxiv.org/pdf/1903.07291.pdf

    Details: Similar to the Batch Normalization, the activation is normalized in the channelwise manner and then modulated with learned scale and bias.

    Code original de cette partie: https://github.com/NVlabs/SPADE/blob/master/models/networks/normalization.py
    '''
    def __init__(self, in_channels, out_channels):
        super(SPADE, self).__init__()
        
        self.bn = nn.BatchNorm2d(in_channels, affine=False)
        self.activation = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1), 
            nn.ReLU()
        )
        self.predict_beta = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride = 1)
        self.predict_gamma = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride = 1)

        
    def forward(self, style, content):
        style_normalized = self.bn(style)

        activation = self.activation(content)
        gamma = self.predict_gamma(activation)
        beta = self.predict_beta(activation)
        print("beta shape: ", beta.shape)
        print('gamma shape: ', gamma.shape)

        out = style_normalized * (1 + gamma) + beta
        return(out)    



class SPADEResBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SPADEResBlock, self).__init__()

        self.spade=SPADE(in_channels=in_planes, out_channels=out_planes)
        def mlp():
                '''
                Transforme le vecteur de style en une feature map: mlp puis reshape
                '''
                return nn.Sequential(
                    nn.Linear(8, out_planes//2),
                    nn.LeakyReLU(negative_slope=0.3, inplace=True), 
                    nn.Linear(out_planes//2, out_planes)
                )

        self.mlp = mlp()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=2e-1, inplace=True)

    def forward(self, x, z):
        z = self.mlp(z)
        z = torch.reshape(z.unsqueeze(-1), [z.shape[0], x.shape[1], x.shape[2], x.shape[3]])
        

        skip_connection = z
        z = self.spade(style=z, content=x)
        z = self.lrelu(z)
        z = self.conv1(z)
        z = self.spade(style = z, content = x)
        z = self.lrelu(z)
        z= self.conv2(z)

        out = z + skip_connection
        return out

class Generator_SPADE(nn.Module):
    def __init__(self, latent_dim=8, w_dim=256, n_channels = 256, n_classes = 1, n_features = 32):
        super(Generator_SPADE, self).__init__()
        self.latent_dim=latent_dim
        self.w_dim=w_dim
        self.n_channels=n_channels
        self.n_classes=n_classes
        self.n_features=n_features
        self.tch_add=10

        def multi_layer_perceptron():
            return nn.Sequential(
                nn.Linear(self.latent_dim,256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.n_channels*10*2)
                )

        self.mlp=multi_layer_perceptron()
        self.resbloc1=SPADEResBlock(self.n_channels, self.n_channels)
        self.resbloc2=SPADEResBlock(self.n_channels, self.n_channels)
        self.up=nn.Sequential(
            ReLUINSConvTranspose2d(self.n_channels, self.n_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            ReLUINSConvTranspose2d(self.n_channels//2, self.n_channels//4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(self.n_channels//4, self.n_classes, kernel_size=1, stride=1, padding=0), 
            #nn.Tanh()
        )


    def forward(self, x, z):
        z=self.mlp(z)
        z=z.view((x.shape[0], x.shape[1],-1))
        z1, z2 = torch.split(z, self.tch_add, dim=2)
        
        z1, z2 = z1.contiguous(), z2.contiguous()
        out1 = self.resbloc1(x, z1)
        out2 = self.resbloc2(out1, z2)
        out = self.up(out2)
        return out


#########################################################################################################################################################################
######################### AdaIN #########################################################################################################################################
#########################################################################################################################################################################
class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()
        

    def forward(self, content, style):
        assert content.shape[:2] == style.shape[:2] # channel wise mean and variance (batch size + channel)
        mean_style=torch.mean(style, dim=2)
        std_style=torch.std(style, dim=2)
        mean_content=torch.mean(content, dim=(2,3))
        std_content=torch.std(content, dim=(2,3))

        normalized_content=(content - mean_content.unsqueeze(-1).unsqueeze(-1).expand(content.shape)) / std_content.unsqueeze(-1).unsqueeze(-1).expand(content.shape) #!!!!!!!!!!!!!!
        out=normalized_content * std_style.unsqueeze(-1).unsqueeze(-1).expand(normalized_content.shape) + mean_style.unsqueeze(-1).unsqueeze(-1).expand(normalized_content.shape)
        return(out)



class AdaINResBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(AdaINResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1)
        self.bn=nn.BatchNorm2d(out_planes) #!!!
        self.adain=AdaIN()

    def forward(self, x, z):
        x=self.conv1(x)
        x=self.relu(x)
        skip_connection=x
        x=self.conv2(x)
        x=self.bn(x)
        x=self.adain(x, z)
        x=self.relu(x)
        x=x+skip_connection
        return x
    
class AdaINResBlock_bis(nn.Module):
    def conv3x3(self, dim_in, dim_out, stride=1):
        return nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride))
    def conv1x1(self, dim_in, dim_out):
        return nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
    
    def __init__(self, dim, dim_extra, stride=1, dropout=0.0):
        super(AdaINResBlock_bis, self).__init__()
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

        self.adain=AdaIN()

    def forward(self, x, z):
        residual = x
        z_expand = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        o1 = self.conv1(x)
        o2 = self.blk1(self.adain(x,z))
        o3 = self.conv2(o2)
        out = self.blk2(torch.cat([o3, z_expand], dim=1))
        out += residual
        return out


class Generator_AdaIN(nn.Module):
    def __init__(self, latent_dim=8, w_dim=256, n_channels = 256, n_classes = 1, n_features = 32):
        super(Generator_AdaIN, self).__init__()
        self.latent_dim=latent_dim
        self.w_dim=w_dim
        self.n_channels=n_channels
        self.n_classes=n_classes
        self.n_features=n_features
        self.tch_add=10

        def multi_layer_perceptron():
            return nn.Sequential(
                nn.Linear(self.latent_dim,256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.n_channels*10*2)
                )

        self.mlp=multi_layer_perceptron()
        self.resbloc1=AdaINResBlock(self.n_channels, self.n_channels)
        self.resbloc2=AdaINResBlock(self.n_channels, self.n_channels)
        self.up=nn.Sequential(
            ReLUINSConvTranspose2d(self.n_channels, self.n_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            ReLUINSConvTranspose2d(self.n_channels//2, self.n_channels//4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(self.n_channels//4, self.n_classes, kernel_size=1, stride=1, padding=0), 
            #nn.Tanh()
        )


    def forward(self, x, z):
        z=self.mlp(z)
        z=z.view((x.shape[0], x.shape[1],-1))
        z1, z2 = torch.split(z, self.tch_add, dim=2)
        z1, z2 = z1.contiguous(), z2.contiguous()
        out1 = self.resbloc1(x, z1)
        out2 = self.resbloc2(out1, z2)
        out = self.up(out2)
        return out
    
class Generator_AdaIN_1(nn.Module):
    def __init__(self, latent_dim=8, w_dim=256, n_channels = 256, n_classes = 1, n_features = 32):
        super(Generator_AdaIN_1, self).__init__()
        self.latent_dim=latent_dim
        self.w_dim=w_dim
        self.n_channels=n_channels
        self.n_classes=n_classes
        self.n_features=n_features
        self.tch_add=256

        def multi_layer_perceptron():
            return nn.Sequential(
                nn.Linear(self.latent_dim,256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, self.n_channels*10)
                )

        self.mlp=multi_layer_perceptron()
        self.resbloc1=AdaINResBlock(self.n_channels, self.n_channels)
        self.resbloc2=AdaINResBlock(self.n_channels, self.n_channels)
        self.up=nn.Sequential(
            ReLUINSConvTranspose2d(self.n_channels, self.n_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            ReLUINSConvTranspose2d(self.n_channels//2, self.n_channels//4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(self.n_channels//4, self.n_classes, kernel_size=1, stride=1, padding=0), 
        )


    def forward(self, x, z):
        z=self.mlp(z)
        z=z.view((x.shape[0], x.shape[1],-1))
        out1 = self.resbloc1(x, z)
        out2 = self.resbloc2(out1, z)
        out = self.up(out2)
        return out
    

#########################################################################################################################################################################
###### CONCAT ###########################################################################################################################################################
#########################################################################################################################################################################
class BasicResBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BasicResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1)
        self.bn=nn.BatchNorm2d(out_planes)
    def forward(self, x):
        x=self.conv1(x)
        x=self.relu(x)
        skip_connection=x
        x=self.conv2(x)
        x=self.bn(x)
        x=self.relu(x)
        x=x+skip_connection
        return x



class Generator_Concat(nn.Module):
    def __init__(self, latent_dim=8, w_dim=256, n_channels = 256, n_classes = 1, n_features = 32):
        super(Generator_Concat, self).__init__()
        self.latent_dim=latent_dim
        self.w_dim=w_dim
        self.n_channels=n_channels
        self.n_classes=n_classes
        self.n_features=n_features
        self.tch_add=10

        self.resbloc1=BasicResBlock(self.n_channels, self.n_channels)
        self.resbloc2=BasicResBlock(self.n_channels, self.n_channels)
        self.up=nn.Sequential(
            ReLUINSConvTranspose2d(self.n_channels, self.n_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            ReLUINSConvTranspose2d(self.n_channels//2, self.n_channels//4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(self.n_channels//4, self.n_classes, kernel_size=1, stride=1, padding=0), 
            #nn.Tanh()
        )


    def forward(self, x, z):
        X = torch.cat([x,z], dim=1)

        out1 = self.resbloc1(X)
        out2 = self.resbloc2(out1)
        out = self.up(out2)
        return out


#########################################################################################################################################################################
###### FiLM #############################################################################################################################################################
#########################################################################################################################################################################
class FiLM(nn.Module):
    def __init__(self):
        super(FiLM, self).__init__()
    
    def forward(self, x, gamma, beta):
        gamma=gamma.unsqueeze(-1).unsqueeze(-1)
        beta=beta.unsqueeze(-1).unsqueeze(-1)
        out = gamma * x + beta # beta et gamma doivent avoir le meme nombre de nombres que la feature map a de channels -> un coeff par channel

        return out 

class FiLMResBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FiLMResBlock, self).__init__()

        def predict_parameters():
            return nn.Sequential(
                #nn.Linear(8, out_planes*2),
                nn.Linear(256, out_planes*2),
                nn.LeakyReLU(negative_slope=0.3, inplace=True), 
                nn.Linear(out_planes*2, out_planes*2)
            ) 
        self.predict = predict_parameters()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1)
        self.bn=nn.BatchNorm2d(out_planes)
        self.film=FiLM()

    def forward(self, x, z):
        film_parameters=self.predict(z) #[128,512]
        
        beta, gamma= torch.split(film_parameters, film_parameters.shape[1]//2, dim=1)
        x=self.conv1(x)
        x=self.relu(x)
        skip_connection=x
        x=self.conv2(x)
        x=self.bn(x)
        x=self.film(x, beta, gamma)
        x=self.relu(x)
        x=x+skip_connection
        return x


class Generator_FiLM(nn.Module):
    def __init__(self, n_channels = 256, n_classes = 1, n_features = 32):
        super(Generator_FiLM, self).__init__()
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
        self.resbloc1=FiLMResBlock(self.n_channels, self.n_channels)
        self.resbloc2=FiLMResBlock(self.n_channels, self.n_channels)
        self.up=nn.Sequential(
            ReLUINSConvTranspose2d(self.n_channels, self.n_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            ReLUINSConvTranspose2d(self.n_channels//2, self.n_channels//4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(self.n_channels//4, self.n_classes, kernel_size=1, stride=1, padding=0), 
        )
    def forward(self, x, z):
        z=self.mlp(z)
        z1, z2 = torch.split(z, self.tch_add, dim=1)
        z1, z2 = z1.contiguous(), z2.contiguous()
        out1 = self.resbloc1(x, z1)
        out2 = self.resbloc2(out1, z2)
        out = self.up(out2)
        return out

class Generator_FiLM_1(nn.Module):
    def __init__(self, n_channels = 256, n_classes = 1, n_features = 32):
        super(Generator_FiLM_1, self).__init__()
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
                nn.Linear(256, self.n_channels))

        self.mlp=multi_layer_perceptron()
        self.resbloc1=FiLMResBlock(self.n_channels, self.n_channels)
        self.resbloc2=FiLMResBlock(self.n_channels, self.n_channels)
        self.up=nn.Sequential(
            ReLUINSConvTranspose2d(self.n_channels, self.n_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
            ReLUINSConvTranspose2d(self.n_channels//2, self.n_channels//4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(self.n_channels//4, self.n_classes, kernel_size=1, stride=1, padding=0), 
        )
    def forward(self, x, z):
        z=self.mlp(z)
        out1 = self.resbloc1(x, z)
        out2 = self.resbloc2(out1, z)
        out = self.up(out2)
        return out


#########################################################################################################################################################################
###### DRIT #############################################################################################################################################################
#########################################################################################################################################################################

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
    
class Generator_feature_reduceplus_1(nn.Module):
    def __init__(self, n_channels = 256, n_classes = 1, n_features = 32):
        super(Generator_feature_reduceplus_1, self).__init__()
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
                nn.Linear(256, self.n_channels))

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
        out1 = self.resbloc1(x, z)
        out2 = self.resbloc2(out1, z)
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
    def __init__(self, n_channels = 1, n_layers = 4, n_features = 64, norm = 'None'):
        super(Discriminateur_reduceplus, self).__init__()
        self.n_features=n_features
        self.n_channels=n_channels
        self.n_layers=n_layers
        model=[]
        model+=[LeakyReLUConv2d(self.n_channels, self.n_features, kernel_size=3, stride=2, padding=1, norm=norm)]
        features=self.n_features
        for i in range(1,self.n_layers-1):
            model += [LeakyReLUConv2d(features, features * 2, kernel_size=3, stride=2, padding=1, norm=norm)]
            features*=2        
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

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    

class Bloc_UNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(Bloc_UNet, self).__init__()
        model = []
        model += [
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_features),
            nn.ReLU(inplace=True)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class UNet(nn.Module):
    def __init__(self, n_channels=256, n_classes=2, n_features=8): # besoin de lui fournir les images complétes en 256 pour la segmentation pour qu'il ait une idée du contexte. On tente en 64 avec les os indifférenciés 
        super(UNet, self).__init__()

        self.maxpool=nn.MaxPool2d(kernel_size=2)
        self.up=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.bloc_1=Bloc_UNet(in_features=n_channels, out_features=n_features) #16
        self.bloc_2=Bloc_UNet(in_features=n_features, out_features=n_features) #8
        self.bloc_3=Bloc_UNet(in_features=n_features, out_features=n_features) #4
        self.bloc_4=Bloc_UNet(in_features=n_features*2, out_features=n_features) #8
        self.bloc_5=Bloc_UNet(in_features=n_features*2, out_features=n_features) #16
        self.bloc_6=Bloc_UNet(in_features=n_features, out_features=n_features) #32

        self.conv_out=nn.Conv2d(in_channels=n_features, out_channels=n_classes, kernel_size=1) #64

    def forward(self, x):
        x1= self.bloc_1(x)

        x2=self.maxpool(x1)
        x2=self.bloc_2(x2)

        x3=self.maxpool(x2)
        x3=self.bloc_3(x3)

        x4=self.up(x3)
        x4=torch.cat([x4, x2], dim=1)
        x4=self.bloc_4(x4)

        x5=self.up(x4)
        x5=torch.cat([x5, x1], dim=1)
        x5=self.bloc_5(x5)

        x6=self.up(x5)
        x6=self.bloc_6(x6)
        x7=self.up(x6)
        out=self.conv_out(x7)
        
        return out

#######################################################################################################################################################################################################################
############ LOSSES ###################################################################################################################################################################################################
#######################################################################################################################################################################################################################

    
class N_pair_Loss(nn.Module):
    """N-pair loss: https://papers.nips.cc/paper/2016/file/6b180037abbebea991d8b1232f8a8ca9-Paper.pdf.
    """
    def __init__(self, temperature=0.07, base_temperature=0.07, gpu = 0):
        super(N_pair_Loss, self).__init__()
        self.gpu = gpu
        if gpu>=0:
            self.Device = torch.device("cuda:"+str(self.gpu))
        else:
            self.Device = torch.device('cpu')

    def forward(self, features1, features2):
        assert features1.shape == features2.shape, "Dimensions of features mismatch"
        SHAPE = features1.shape
        dot_product = torch.matmul(features1, features2.T)
        mask_positive = torch.diag(torch.ones(SHAPE[0])).to(self.Device)
        mask_negative = torch.ones(SHAPE[0], SHAPE[0]).to(self.Device) - mask_positive
        positive_dot_product = torch.diagonal(dot_product * mask_positive).unsqueeze(0).T
        negative_dot_product = torch.subtract(dot_product, positive_dot_product)
        log_prob = torch.log(torch.sum(torch.exp(negative_dot_product)*mask_negative, 1) + torch.ones(SHAPE[0]).to(self.Device))
        loss = torch.mean(log_prob, dim=0)
        return(loss)



class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, gpu = 0):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.gpu = gpu 
        if gpu>=0:
            self.Device = torch.device("cuda:"+str(self.gpu))
        else:
            self.Device = torch.device('cpu')

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.Device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.Device)
        else:
            mask = mask.float().to(self.Device)


        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.Device),
            0
        )

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class CSloss_arbitrary_style_transfer(nn.Module):
    def __init__(self, lambda_content, lambda_style, device=0):
        super(CSloss_arbitrary_style_transfer, self).__init__()
        self.lambda_style = lambda_style
        self.lambda_content = lambda_content
        self.device = device
        vgg = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )
        vgg = vgg.eval()
        vgg.load_state_dict(torch.load("/home/claire/Nets_Reconstruction/clean_code/checkpoints/vgg_normalised.pth"))
        self.norm = nn.Sequential(*list(vgg.children())[:1])
        self.enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

        self.norm.to(torch.device("cuda:"+str(self.device)))
        self.enc_1.to(torch.device("cuda:"+str(self.device)))
        self.enc_2.to(torch.device("cuda:"+str(self.device)))
        self.enc_3.to(torch.device("cuda:"+str(self.device)))
        self.enc_4.to(torch.device("cuda:"+str(self.device)))
        self.enc_5.to(torch.device("cuda:"+str(self.device)))

        

    def forward(self, style, content, stylized):
        m= torch.min(style)
        M = torch.max(style)
        style = (style - m) / (M - m)
        m= torch.min(content)
        M = torch.max(content)
        content = (content - m) / (M - m)
        m= torch.min(stylized)
        M = torch.max(stylized)
        stylized = (stylized- m) / (M - m)
        style = style.expand(-1,3,-1,-1)
        content = content.expand(-1,3,-1,-1)
        stylized = stylized.expand(-1,3,-1,-1)

        Content4_1 = self.enc_4(self.enc_3(self.enc_2(self.enc_1(content))))
        Content5_1 = self.enc_5(Content4_1)
        Style4_1 = self.enc_4(self.enc_3(self.enc_2(self.enc_1(style))))
        Style5_1 = self.enc_5(Style4_1)
        Stylized4_1 = self.enc_4(self.enc_3(self.enc_2(self.enc_1(stylized))))
        Stylized5_1 = self.enc_5(Stylized4_1)

        content_loss = nn.MSELoss()(Stylized4_1, Content4_1) + nn.MSELoss()(Stylized5_1, Content5_1)

        style_loss = 0
        STYLIZED = stylized
        STYLE = style
        for net in [self.enc_1, self.enc_2, self.enc_3, self.enc_4, self.enc_5]:
            stylized_relu = net(STYLIZED)
            STYLIZED = stylized_relu
            style_relu = net(STYLE)
            STYLE = style_relu
            style_loss += nn.MSELoss()(torch.mean(stylized_relu), torch.mean(style_relu)) + nn.MSELoss()(torch.std(stylized_relu), torch.std(style_relu))

        return style_loss*self.lambda_style, content_loss*self.lambda_content



class Loss_DRIT_custom(nn.Module):
    '''
    Loss customisée. Au lieu de calculer à chaque training step l'ensemble des loss et de pondérer parun lambda potentiellement nul
    On regarde d'emblée quels sont les hyper paramètres qui sont non nuls et on répartis les loss pour chaque réseau dans l'initialisation. 
    On évite ainsi pas mal de tests et d'opérations inutiles
    Paramètres:
        - opt -> fichier d'option
        - toutes les variables résultatnt de infer batch

    Sortie:
        - Loss par réseau (à noter qu'il y a deux forward différents selon si la random size est non nulle ou non)
    '''
    def __init__(self, opt, gpu):#, OPTIMIZERS):
        super(Loss_DRIT_custom, self).__init__()
        self.random_size = opt.random_size
        self.gpu = gpu
        Number_of_networks = 10

        self.LOSSES = [None for i in range(Number_of_networks)] # contient toutes les loss
        self.LOSS_RETURNS = [None for i in range(Number_of_networks)] # contient les param de retour des loss
        self.LOSSES_PER_NETWORKS = [[] for i in range(Number_of_networks)] # contient les indices des loss de self.LOSSES à utiliser pour un réseau
        self.LOSS_INPUTS = [None for i in range(Number_of_networks)]
        self.RETURNS_PER_NETWORK = ['' for i in range(Number_of_networks)]

        if opt.lambda_cyclic_anatomy != 0:
            self.lambda_cyclic_anatomy = opt.lambda_cyclic_anatomy
            self.LOSSES[0] = self.cyclic_anatomy
            self.LOSS_INPUTS[0] = ['content_LR, content_HR, content_fake_LR, content_fake_HR']
            self.LOSS_RETURNS[0] = "cyclic_anatomy_HR, cyclic_anatomy_LR"
            self.LOSSES_PER_NETWORKS[0].append(0)
            self.LOSSES_PER_NETWORKS[3].append(0)
            self.LOSSES_PER_NETWORKS[4].append(0)
            self.RETURNS_PER_NETWORK[0] += 'cyclic_anatomy_HR+cyclic_anatomy_LR+'
            self.RETURNS_PER_NETWORK[3] += 'cyclic_anatomy_LR+'
            self.RETURNS_PER_NETWORK[4] += 'cyclic_anatomy_HR+'

        if opt.lambda_cyclic_modality != 0:
            self.lambda_cyclic_modality = opt.lambda_cyclic_modality
            self.LOSSES[1] = self.cyclic_modality
            self.LOSS_INPUTS[1] = ['z_LR, z_HR, z_fake_LR, z_fake_HR']
            self.LOSS_RETURNS[1] = "cyclic_modality_HR, cyclic_modality_LR"
            self.LOSSES_PER_NETWORKS[1].append(1)
            self.LOSSES_PER_NETWORKS[2].append(1)
            self.LOSSES_PER_NETWORKS[3].append(1)
            self.LOSSES_PER_NETWORKS[4].append(1)
            self.RETURNS_PER_NETWORK[1] += 'cyclic_modality_HR+'
            self.RETURNS_PER_NETWORK[2] += 'cyclic_modality_LR+'
            self.RETURNS_PER_NETWORK[3] += 'cyclic_modality_HR+'
            self.RETURNS_PER_NETWORK[4] += 'cyclic_modality_LR+'

        if opt.lambda_contrastive_modality != 0:
            self.lambda_contrastive_modality = opt.lambda_contrastive_modality
            self.LOSSES[2] = self.contrastive_loss
            self.LOSS_INPUTS[2] = ['z_LR, z_HR, z_fake_LR, z_fake_HR']
            self.LOSS_RETURNS[2] = "contrastive_loss_modality, contrastive_loss_modalityfake"
            self.LOSSES_PER_NETWORKS[1].append(2)
            self.LOSSES_PER_NETWORKS[2].append(2)
            self.LOSSES_PER_NETWORKS[3].append(2)
            self.LOSSES_PER_NETWORKS[4].append(2)
            self.RETURNS_PER_NETWORK[1] += 'contrastive_loss_modality+contrastive_loss_modalityfake+'
            self.RETURNS_PER_NETWORK[2] += 'contrastive_loss_modality+contrastive_loss_modalityfake+'
            self.RETURNS_PER_NETWORK[3] += 'contrastive_loss_modalityfake+'
            self.RETURNS_PER_NETWORK[4] += 'contrastive_loss_modalityfake+'

        if opt.lambda_Npair != 0:
            self.lambda_Npair = opt.lambda_Npair
            self.LOSSES[3] = self.n_pair
            self.LOSS_INPUTS[3] = ['content_LR, content_HR, content_fake_LR, content_fake_HR']
            self.LOSS_RETURNS[3] = "contrastive_loss_anat, contrastive_loss_anatfake"
            self.LOSSES_PER_NETWORKS[0].append(3)
            self.LOSSES_PER_NETWORKS[3].append(3)
            self.LOSSES_PER_NETWORKS[4].append(3)
            self.RETURNS_PER_NETWORK[0] += 'contrastive_loss_anat+contrastive_loss_anatfake+'
            self.RETURNS_PER_NETWORK[3] += 'contrastive_loss_anatfake+'
            self.RETURNS_PER_NETWORK[4] += 'contrastive_loss_anatfake+'
        
        if opt.lambda_D_content != 0:
            self.lambda_D_content = opt.lambda_D_content
            self.LOSSES[4] = self.backward_D_content
            self.LOSS_INPUTS[4] = ['content_LR, content_HR, discriminator_content']
            self.LOSS_RETURNS[4] = "loss_D_content"
            self.LOSSES_PER_NETWORKS[5].append(4)
            self.RETURNS_PER_NETWORK[5] += 'loss_D_content+'

        if opt.lambda_latent != 0:
            self.lambda_latent = opt.lambda_latent
            self.LOSSES[5] = self.backward_G_alone
            self.LOSS_INPUTS[5] = ['fake_LR_random, fake_HR_random, z_random_LR, z_random_HR, z_random, discriminator_LR2, discriminator_HR2']
            self.LOSS_RETURNS[5] = "loss_z_LR, loss_z_HR"
            self.LOSSES_PER_NETWORKS[0].append(5)
            self.LOSSES_PER_NETWORKS[3].append(5)
            self.LOSSES_PER_NETWORKS[4].append(5)
            self.RETURNS_PER_NETWORK[0] += 'loss_z_LR+loss_z_HR+'
            self.RETURNS_PER_NETWORK[3] += 'loss_z_HR+'
            self.RETURNS_PER_NETWORK[4] += 'loss_z_LR+'

        if opt.lambda_D_domain != 0:
            self.lambda_D_domain = opt.lambda_D_domain
            self.LOSSES[6] = self.backward_D_domain
            self.LOSS_INPUTS[6] = ['discriminator_LR, LR, fake_LR', 'discriminator_HR, HR, fake_HR']
            self.LOSS_RETURNS[6] = "loss_D1_"
            self.LOSSES_PER_NETWORKS[6].append(6)
            self.LOSSES_PER_NETWORKS[7].append(6)
            self.RETURNS_PER_NETWORK[6] += 'loss_D1_+'
            self.RETURNS_PER_NETWORK[7] += 'loss_D1_+'

        if (opt.lambda_self != 0 or opt.lambda_cross_cycle != 0 or opt.lambda_reg != 0 or opt.lambda_adv_anatomy_encoder != 0 or opt.lambda_adv_generator != 0):
            self.lambda_self = opt.lambda_self
            self.lambda_cross_cycle = opt.lambda_cross_cycle
            self.lambda_reg = opt.lambda_reg
            self.lambda_adv_anatomy_encoder = opt.lambda_adv_anatomy_encoder
            self.lambda_adv_generator = opt.lambda_adv_generator
            self.LOSSES[7] = self.backward_EG
            self.LOSS_INPUTS[7] = ['LR, HR, fake_LR, fake_HR, fake_LRLR, fake_HRHR, est_LR, est_HR, z_LR, z_HR, content_LR, content_HR, z_fake_LR, z_fake_HR, content_fake_LR, content_fake_HR, discriminator_content, discriminator_LR, discriminator_HR']
            self.LOSS_RETURNS[7] = "loss_G_LR, loss_G_HR"
            self.LOSSES_PER_NETWORKS[0].append(7)
            self.LOSSES_PER_NETWORKS[1].append(7)
            self.LOSSES_PER_NETWORKS[2].append(7)
            self.LOSSES_PER_NETWORKS[3].append(7)
            self.LOSSES_PER_NETWORKS[4].append(7)
            self.RETURNS_PER_NETWORK[0] += 'loss_G_LR+loss_G_HR+'
            self.RETURNS_PER_NETWORK[1] += 'loss_G_HR+'
            self.RETURNS_PER_NETWORK[2] += 'loss_G_LR+'
            self.RETURNS_PER_NETWORK[3] += 'loss_G_HR+'
            self.RETURNS_PER_NETWORK[4] += 'loss_G_LR+'

        if (opt.lambda_style_loss != 0 or opt.lambda_content_loss != 0):
            sys.exit('This loss is not implemented yet')

        if (opt.random_size !=0 and opt.lambda_D_domain != 0):
            self.random_size = opt.random_size
            self.lambda_D_domain = opt.lambda_D_domain
            self.LOSSES[9] = self.backward_D_domain
            self.LOSS_INPUTS[9] = ['discriminator_LR2, LR_random, fake_LR_random', 'discriminator_HR2, HR_random, fake_HR_random']
            self.LOSS_RETURNS[9] = "loss_D2_"
            self.LOSSES_PER_NETWORKS[8].append(9)
            self.LOSSES_PER_NETWORKS[9].append(9)
            self.RETURNS_PER_NETWORK[8] += 'loss_D2_+'
            self.RETURNS_PER_NETWORK[9] += 'loss_D2_+'

        self.RETURNS_PER_NETWORK = [e[:-1] for e in self.RETURNS_PER_NETWORK]


    def cyclic_anatomy(self, content_LR, content_HR, content_fake_LR, content_fake_HR):
        return torch.nn.L1Loss()(content_HR, content_fake_LR)*self.lambda_cyclic_anatomy, torch.nn.L1Loss()(content_LR, content_fake_HR)*self.lambda_cyclic_anatomy
     
    def cyclic_modality(self, z_LR, z_HR, z_fake_LR, z_fake_HR):
        return torch.nn.L1Loss()(z_HR, z_fake_HR)*self.lambda_cyclic_modality , torch.nn.L1Loss()(z_LR, z_fake_LR)*self.lambda_cyclic_modality
    
    def n_pair(self, content_LR, content_HR, content_fake_LR, content_fake_HR):
        L = N_pair_Loss()
        batch_size = content_HR.shape[0]
        content_LR = content_LR.reshape(batch_size, -1)
        content_HR = content_HR.reshape(batch_size, -1)
        content_fake_LR = content_fake_LR.reshape(batch_size, -1)
        content_fake_HR = content_fake_HR.reshape(batch_size, -1)

        content_LR_norm = torch.nn.functional.normalize(content_LR, dim=1) # borne les vecteurs pour qu'ils aient une norme de 1
        content_HR_norm = torch.nn.functional.normalize(content_HR, dim=1)
        content_fake_LR_norm = torch.nn.functional.normalize(content_fake_LR, dim=1)
        content_fake_HR_norm = torch.nn.functional.normalize(content_fake_HR, dim=1)
        loss_anat = L(features1 = content_HR_norm, features2=content_LR_norm)*self.lambda_Npair
        loss_anatfake = L(features1 = content_fake_HR_norm, features2=content_fake_LR_norm)*self.lambda_Npair

        return loss_anat, loss_anatfake
    
    def contrastive_loss(self, z_LR, z_HR, z_fake_LR, z_fake_HR):
        L = SupervisedContrastiveLoss()
        batch_size = z_HR.shape[0]
        # labels = torch.cat([torch.full((batch_size,), 0, dtype=torch.float), torch.full((batch_size,), 1, dtype=torch.float)], dim=0).to(torch.device("cuda:"+str(self.gpu)))
        labels = torch.cat([torch.full((batch_size,), 0, dtype=torch.float), torch.full((batch_size,), 1, dtype=torch.float)], dim=0).to(self.Device)
        
        z_LR = z_LR.reshape(batch_size, -1)
        z_HR = z_HR.reshape(batch_size, -1)
        z_fake_LR = z_fake_LR.reshape(batch_size, -1)
        z_fake_HR = z_fake_HR.reshape(batch_size, -1)

        modality_norm = torch.nn.functional.normalize(torch.cat([z_HR, z_LR], dim = 0).unsqueeze(1), dim = 2)
        modalityfake_norm = torch.nn.functional.normalize(torch.cat([z_fake_HR, z_fake_LR], dim = 0).unsqueeze(1), dim = 2)
        loss_modality = L(features = modality_norm, labels = labels)*self.lambda_contrastive_modality
        loss_modalityfake = L(features = modalityfake_norm, labels = labels)*self.lambda_contrastive_modality
        return loss_modality, loss_modalityfake

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
        return loss_D*self.lambda_D_domain

    def backward_D_content(self, imageA, imageB, discriminator_content):
        pred_fake = discriminator_content.forward(imageA.detach())
        pred_real = discriminator_content.forward(imageB.detach())
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all1 = torch.ones((out_real.size(0))).cuda(self.gpu)
            all0 = torch.zeros((out_fake.size(0))).cuda(self.gpu)
            ad_true_loss = nn.functional.binary_cross_entropy_with_logits(out_real, all1)
            ad_fake_loss = nn.functional.binary_cross_entropy_with_logits(out_fake, all0)
        loss_D = ad_true_loss + ad_fake_loss
        return loss_D*self.lambda_D_content

    def backward_G_GAN_content(self, z_content, discriminator_content):
        outs = discriminator_content.forward(z_content)
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

    def backward_EG(self, x, y, fake_LR, fake_HR, fake_LRLR, fake_HRHR, est_LR, est_HR, z_LR, z_HR, content_LR, content_HR, z_fake_LR, z_fake_HR, content_fake_LR, content_fake_HR, discriminator_content, discriminator_LR, discriminator_HR):
        # content Ladv for generator
        loss_G_GAN_LRcontent = self.backward_G_GAN_content(content_LR, discriminator_content)*self.lambda_adv_anatomy_encoder
        loss_G_GAN_HRcontent = self.backward_G_GAN_content(content_HR, discriminator_content)*self.lambda_adv_anatomy_encoder

        # Ladv for generator
        loss_G_GAN_LR = self.backward_G_GAN(fake_LR, discriminator_LR)*self.lambda_adv_generator
        loss_G_GAN_HR = self.backward_G_GAN(fake_HR, discriminator_HR)*self.lambda_adv_generator

        # KL loss - z_a
        #loss_kl_za_LR = self._l2_regularize(z_fake_LR) * 0.01
        loss_kl_za_LR = self._l2_regularize(z_LR) * self.lambda_reg # torch.Size([128, 8])
        loss_kl_za_HR = self._l2_regularize(z_HR) * self.lambda_reg # torch.Size([128, 8])

        # KL loss - z_c
        loss_kl_zc_LR = self._l2_regularize(content_LR) * self.lambda_reg
        loss_kl_zc_HR = self._l2_regularize(content_HR) * self.lambda_reg

        # cross cycle consistency loss
        loss_G_L1_LR = torch.nn.L1Loss()(est_LR, x) * self.lambda_cross_cycle
        loss_G_L1_HR = torch.nn.L1Loss()(est_HR, y) * self.lambda_cross_cycle
        loss_G_L1_LRLR = torch.nn.L1Loss()(fake_LRLR, x) * self.lambda_self
        loss_G_L1_HRHR = torch.nn.L1Loss()(fake_HRHR, y) * self.lambda_self

        loss_G_LR = loss_G_GAN_LR + \
                loss_G_GAN_LRcontent + \
                loss_G_L1_LRLR + \
                loss_G_L1_LR + \
                loss_kl_zc_LR + \
                loss_kl_za_LR + \
                loss_G_L1_HR #############
        
        loss_G_HR = loss_G_GAN_HR + \
                loss_G_GAN_HRcontent + \
                loss_G_L1_HRHR + \
                loss_G_L1_HR + \
                loss_kl_zc_HR + \
                loss_kl_za_HR + \
                loss_G_L1_LR ################
        return(loss_G_LR, loss_G_HR)

    def backward_G_alone(self,fake_LR_random, fake_HR_random, z_random_LR, z_random_HR, z_random, discriminator_LR2, discriminator_HR2):
        
        if self.random_size!=0: 
            # Ladv for generator
            loss_G_GAN2_LR = self.backward_G_GAN(fake_LR_random, discriminator_LR2)
            loss_G_GAN2_HR = self.backward_G_GAN(fake_HR_random, discriminator_HR2)
            # latent regression loss
            loss_z_L1_LR = torch.mean(torch.abs(z_random_LR - z_random)) * self.lambda_latent
            loss_z_L1_HR = torch.mean(torch.abs(z_random_HR - z_random)) * self.lambda_latent

            loss_z_L1 = loss_z_L1_LR + loss_z_L1_HR + loss_G_GAN2_LR + loss_G_GAN2_HR
            loss_z_LR= loss_z_L1_LR + loss_G_GAN2_LR 
            loss_z_HR= loss_z_L1_HR + loss_G_GAN2_HR

            return loss_z_LR, loss_z_HR

        else:
            # latent regression loss
            loss_z_L1_LR = torch.mean(torch.abs(z_random_LR - z_random)) * self.lambda_latent
            loss_z_L1_HR = torch.mean(torch.abs(z_random_HR - z_random)) * self.lambda_latent

            loss_z_L1 = loss_z_L1_LR + loss_z_L1_HR 
            loss_z_LR= loss_z_L1_LR 
            loss_z_HR= loss_z_L1_HR 
            return loss_z_LR, loss_z_HR

    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss
      
    def forward(self, log, optimizer_idx, LR, HR, fake_LR, fake_HR, fake_LRLR, fake_HRHR, est_LR, est_HR, z_LR, z_HR, content_LR, content_HR, z_fake_LR, z_fake_HR, content_fake_LR, content_fake_HR, z_random, fake_LR_random, fake_HR_random, z_random_LR, z_random_HR, LR_random, HR_random, discriminator_content, discriminator_LR, discriminator_HR, anatomy_encoder, modality_encoder, discriminator_LR2=None, discriminator_HR2=None):
        '''
        The goal is to compute only the necessary thing at each step. Since this function is called for each training step, 
        for every optimizer, we try to reduce the ocmputation time by compute only the needed functions
        '''
        
        if optimizer_idx == 0 and anatomy_encoder!='ResNet50': #encoder content
            loc = locals()
            for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
                loc['i']=i
                exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
            exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
            log('Encodeur anatomie', loc['loss'])
            return(loc['loss'])

        elif optimizer_idx == 1 and modality_encoder!='ResNet50': #encoder HR
            loc = locals()
            for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
                loc['i']=i
                exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
            exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
            log('Encodeur HR/T2', loc['loss'])
            return(loc['loss'])

        elif optimizer_idx == 2 and modality_encoder!='ResNet50': #encoder LR
            loc = locals()
            for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
                loc['i']=i
                exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
            exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
            log('Encodeur LR/T1', loc['loss'])
            return(loc['loss'])
        
        elif optimizer_idx == 3: #generator HR
            loc = locals()
            for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
                loc['i']=i
                exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
            exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
            log('Generateur HR/T2', loc['loss'])
            return(loc['loss'])
        
        elif optimizer_idx == 4: #generator LR
            loc = locals()
            for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
                loc['i']=i
                exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
            exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
            log('Generateur LR/T1', loc['loss'])
            return(loc['loss'])
        
        elif optimizer_idx==5: #discriminator content
            loc = locals()
            for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
                loc['i']=i
                exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
            exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
            log('Discriminateur anatomie', loc['loss'])
            nn.utils.clip_grad_norm_(discriminator_content.parameters(), 5)
            return(loc['loss'])
        
        elif optimizer_idx==6: #discriminateur LR
            loc = locals()
            for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
                loc['i']=i
                exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
            exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
            log('Discriminateur LR/T1', loc['loss'])
            return(loc['loss'])
        
        elif optimizer_idx==7: #discriminator HR
            loc = locals()
            for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
                loc['i']=i
                exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][1])+")", globals(), loc)
            exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
            log('Discriminateur HR/T2', loc['loss'])
            return(loc['loss'])
        
        elif optimizer_idx==8: #optimizer_discriminator_LR2
            loc = locals()
            for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
                loc['i']=i
                exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
            exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
            log('Discriminateur 2 LR/T1', loc['loss'])
            return(loc['loss'])
        
        elif optimizer_idx==9: #optimizer_discriminator_HR2
            loc = locals()
            for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
                loc['i']=i
                exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][1])+")", globals(), loc)
            exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
            log('Discriminateur 2 HR/T2', loc['loss'])
            return(loc['loss'])
        
class Loss_DRIT_custom_last_version(nn.Module):
    '''
    Loss customisée. Au lieu de calculer à chaque training step l'ensemble des loss et de pondérer parun lambda potentiellement nul
    On regarde d'emblée quels sont les hyper paramètres qui sont non nuls et on répartis les loss pour chaque réseau dans l'initialisation. 
    On évite ainsi pas mal de tests et d'opérations inutiles
    Paramètres:
        - opt -> fichier d'option
        - toutes les variables résultatnt de infer batch

    Sortie:
        - Loss par réseau (à noter qu'il y a deux forward différents selon si la random size est non nulle ou non)
    '''
    def __init__(self, opt, gpu):#, OPTIMIZERS):
        super(Loss_DRIT_custom_last_version, self).__init__()
        self.random_size = opt.random_size
        self.use_segmentation_network = opt.use_segmentation_network
        self.gpu = gpu
        Number_of_networks = 11

        # if opt.random_size != 0:
        #     assert str(OPTIMIZERS) == "[optimizer_encode_content, optimizer_encode_HR, optimizer_encode_LR, optimizer_generator_HR, optimizer_generator_LR, optimizer_discriminator_content, optimizer_discriminator_LR, optimizer_discriminator_HR, optimizer_discriminator_LR2, optimizer_discriminator_HR2]"
        # else:
        #     assert str(OPTIMIZERS) == "[optimizer_encode_content, optimizer_encode_HR, optimizer_encode_LR, optimizer_generator_HR, optimizer_generator_LR, optimizer_discriminator_content, optimizer_discriminator_LR, optimizer_discriminator_HR]"

        self.LOSSES = [None for i in range(Number_of_networks)] # contient toutes les loss
        self.LOSS_RETURNS = [None for i in range(Number_of_networks)] # contient les param de retour des loss
        self.LOSSES_PER_NETWORKS = [[] for i in range(Number_of_networks)] # contient les indices des loss de self.LOSSES à utiliser pour un réseau
        self.LOSS_INPUTS = [None for i in range(Number_of_networks)]
        self.RETURNS_PER_NETWORK = ['' for i in range(Number_of_networks)]
        if self.random_size != 0:
            self.OPTIMIZERS = 'optimizer_encode_content, optimizer_encode_HR, optimizer_encode_LR, optimizer_generator_HR, optimizer_generator_LR, optimizer_discriminator_content, optimizer_discriminator_LR, optimizer_discriminator_HR, optimizer_discriminator_LR2, optimizer_discriminator_HR2'
        else:
            self.OPTIMIZERS = 'optimizer_encode_content, optimizer_encode_HR, optimizer_encode_LR, optimizer_generator_HR, optimizer_generator_LR, optimizer_discriminator_content, optimizer_discriminator_LR, optimizer_discriminator_HR'

        if opt.lambda_cyclic_anatomy != 0:
            self.lambda_cyclic_anatomy = opt.lambda_cyclic_anatomy
            self.LOSSES[0] = self.cyclic_anatomy
            self.LOSS_INPUTS[0] = ['content_LR, content_HR, content_fake_LR, content_fake_HR']
            self.LOSS_RETURNS[0] = "cyclic_anatomy_HR, cyclic_anatomy_LR"
            self.LOSSES_PER_NETWORKS[0].append(0)
            self.LOSSES_PER_NETWORKS[3].append(0)
            self.LOSSES_PER_NETWORKS[4].append(0)
            self.RETURNS_PER_NETWORK[0] += 'cyclic_anatomy_HR+cyclic_anatomy_LR+'
            self.RETURNS_PER_NETWORK[3] += 'cyclic_anatomy_LR+'
            self.RETURNS_PER_NETWORK[4] += 'cyclic_anatomy_HR+'

        if opt.lambda_cyclic_modality != 0:
            self.lambda_cyclic_modality = opt.lambda_cyclic_modality
            self.LOSSES[1] = self.cyclic_modality
            self.LOSS_INPUTS[1] = ['z_LR, z_HR, z_fake_LR, z_fake_HR']
            self.LOSS_RETURNS[1] = "cyclic_modality_HR, cyclic_modality_LR"
            self.LOSSES_PER_NETWORKS[1].append(1)
            self.LOSSES_PER_NETWORKS[2].append(1)
            self.LOSSES_PER_NETWORKS[3].append(1)
            self.LOSSES_PER_NETWORKS[4].append(1)
            self.RETURNS_PER_NETWORK[1] += 'cyclic_modality_HR+'
            self.RETURNS_PER_NETWORK[2] += 'cyclic_modality_LR+'
            self.RETURNS_PER_NETWORK[3] += 'cyclic_modality_HR+'
            self.RETURNS_PER_NETWORK[4] += 'cyclic_modality_LR+'

        if opt.lambda_contrastive_modality != 0:
            self.lambda_contrastive_modality = opt.lambda_contrastive_modality
            self.LOSSES[2] = self.contrastive_loss
            self.LOSS_INPUTS[2] = ['z_LR, z_HR, z_fake_LR, z_fake_HR']
            self.LOSS_RETURNS[2] = "contrastive_loss_modality, contrastive_loss_modalityfake"
            self.LOSSES_PER_NETWORKS[1].append(2)
            self.LOSSES_PER_NETWORKS[2].append(2)
            self.LOSSES_PER_NETWORKS[3].append(2)
            self.LOSSES_PER_NETWORKS[4].append(2)
            self.RETURNS_PER_NETWORK[1] += 'contrastive_loss_modality+contrastive_loss_modalityfake+'
            self.RETURNS_PER_NETWORK[2] += 'contrastive_loss_modality+contrastive_loss_modalityfake+'
            self.RETURNS_PER_NETWORK[3] += 'contrastive_loss_modalityfake+'
            self.RETURNS_PER_NETWORK[4] += 'contrastive_loss_modalityfake+'

        if opt.lambda_Npair != 0:
            self.lambda_Npair = opt.lambda_Npair
            self.LOSSES[3] = self.n_pair
            self.LOSS_INPUTS[3] = ['content_LR, content_HR, content_fake_LR, content_fake_HR']
            self.LOSS_RETURNS[3] = "contrastive_loss_anat, contrastive_loss_anatfake"
            self.LOSSES_PER_NETWORKS[0].append(3)
            self.LOSSES_PER_NETWORKS[3].append(3)
            self.LOSSES_PER_NETWORKS[4].append(3)
            self.RETURNS_PER_NETWORK[0] += 'contrastive_loss_anat+contrastive_loss_anatfake+'
            self.RETURNS_PER_NETWORK[3] += 'contrastive_loss_anatfake+'
            self.RETURNS_PER_NETWORK[4] += 'contrastive_loss_anatfake+'
        
        if opt.lambda_D_content != 0:
            self.lambda_D_content = opt.lambda_D_content
            self.LOSSES[4] = self.backward_D_content
            self.LOSS_INPUTS[4] = ['content_LR, content_HR, discriminator_content']
            self.LOSS_RETURNS[4] = "loss_D_content"
            self.LOSSES_PER_NETWORKS[5].append(4)
            self.RETURNS_PER_NETWORK[5] += 'loss_D_content+'

        if opt.lambda_latent != 0:
            self.lambda_latent = opt.lambda_latent
            self.LOSSES[5] = self.backward_G_alone
            self.LOSS_INPUTS[5] = ['fake_LR_random, fake_HR_random, z_random_LR, z_random_HR, z_random, discriminator_LR2, discriminator_HR2']
            self.LOSS_RETURNS[5] = "loss_z_LR, loss_z_HR"
            self.LOSSES_PER_NETWORKS[0].append(5)
            self.LOSSES_PER_NETWORKS[3].append(5)
            self.LOSSES_PER_NETWORKS[4].append(5)
            self.RETURNS_PER_NETWORK[0] += 'loss_z_LR+loss_z_HR+'
            self.RETURNS_PER_NETWORK[3] += 'loss_z_HR+'
            self.RETURNS_PER_NETWORK[4] += 'loss_z_LR+'

        if opt.lambda_D_domain != 0:
            self.lambda_D_domain = opt.lambda_D_domain
            self.LOSSES[6] = self.backward_D_domain
            self.LOSS_INPUTS[6] = ['discriminator_LR, LR, fake_LR', 'discriminator_HR, HR, fake_HR']
            self.LOSS_RETURNS[6] = "loss_D1_"
            self.LOSSES_PER_NETWORKS[6].append(6)
            self.LOSSES_PER_NETWORKS[7].append(6)
            self.RETURNS_PER_NETWORK[6] += 'loss_D1_+'
            self.RETURNS_PER_NETWORK[7] += 'loss_D1_+'

        if (opt.lambda_self != 0 or opt.lambda_cross_cycle != 0 or opt.lambda_reg != 0 or opt.lambda_adv_anatomy_encoder != 0 or opt.lambda_adv_generator != 0):
            self.lambda_self = opt.lambda_self
            self.lambda_cross_cycle = opt.lambda_cross_cycle
            self.lambda_reg = opt.lambda_reg
            self.lambda_adv_anatomy_encoder = opt.lambda_adv_anatomy_encoder
            self.lambda_adv_generator = opt.lambda_adv_generator
            self.LOSSES[7] = self.backward_EG
            self.LOSS_INPUTS[7] = ['LR, HR, fake_LR, fake_HR, fake_LRLR, fake_HRHR, est_LR, est_HR, z_LR, z_HR, content_LR, content_HR, z_fake_LR, z_fake_HR, content_fake_LR, content_fake_HR, discriminator_content, discriminator_LR, discriminator_HR']
            self.LOSS_RETURNS[7] = "loss_G_LR, loss_G_HR"
            self.LOSSES_PER_NETWORKS[0].append(7)
            self.LOSSES_PER_NETWORKS[1].append(7)
            self.LOSSES_PER_NETWORKS[2].append(7)
            self.LOSSES_PER_NETWORKS[3].append(7)
            self.LOSSES_PER_NETWORKS[4].append(7)
            self.RETURNS_PER_NETWORK[0] += 'loss_G_LR+loss_G_HR+'
            self.RETURNS_PER_NETWORK[1] += 'loss_G_HR+'
            self.RETURNS_PER_NETWORK[2] += 'loss_G_LR+'
            self.RETURNS_PER_NETWORK[3] += 'loss_G_HR+'
            self.RETURNS_PER_NETWORK[4] += 'loss_G_LR+'

        if (opt.lambda_style_loss != 0 or opt.lambda_content_loss != 0):
            sys.exit('This loss is not implemented yet')
            # self.lambda_style_loss = opt.lambda_style_loss
            # self.lambda_content_loss = opt.lambda_content_loss
            # self.LOSSES[8] = CSloss_arbitrary_style_transfer(lambda_content=opt.lambda_content_loss, lambda_style=opt.lambda_style_loss)
            # self.LOSS_INPUTS[8] = ['lambda_content=self.lambda_content_loss, lambda_style=self.lambda_style_loss)(style=LR, content=HR, stylized=fake_LR', 'lambda_content=self.lambda_content_loss, lambda_style=self.lambda_style_loss)(style=HR, content=LR, stylized=fake_HR', 'lambda_content=self.lambda_content_loss, lambda_style=self.lambda_style_loss)(style=fake_LR, content=fake_HR, stylized=est_LR', 'lambda_content=self.lambda_content_loss, lambda_style=self.lambda_style_loss)(style=fake_HR, content=fake_LR, stylized=est_HR']
            # self.LOSS_RETURNS[8] = "style_loss_domain_, content_loss_domain_"
            # self.LOSSES_PER_NETWORKS[0].append(8)
            # self.LOSSES_PER_NETWORKS[1].append(8)
            # self.LOSSES_PER_NETWORKS[2].append(8)
            # self.LOSSES_PER_NETWORKS[3].append(8)
            # self.LOSSES_PER_NETWORKS[4].append(8)
            # self.RETURNS_PER_NETWORK[0] += 'tyle_loss_LRdomain_, content_loss_LRdomain_+'
            # self.RETURNS_PER_NETWORK[1] += ''
            # self.RETURNS_PER_NETWORK[2] += ''
            # self.RETURNS_PER_NETWORK[3] += ''
            # self.RETURNS_PER_NETWORK[4] += ''

        if (opt.random_size !=0 and opt.lambda_D_domain != 0):
            self.random_size = opt.random_size
            self.lambda_D_domain = opt.lambda_D_domain
            self.LOSSES[9] = self.backward_D_domain
            self.LOSS_INPUTS[9] = ['discriminator_LR2, LR_random, fake_LR_random', 'discriminator_HR2, HR_random, fake_HR_random']
            self.LOSS_RETURNS[9] = "loss_D2_"
            self.LOSSES_PER_NETWORKS[8].append(9)
            self.LOSSES_PER_NETWORKS[9].append(9)
            self.RETURNS_PER_NETWORK[8] += 'loss_D2_+'
            self.RETURNS_PER_NETWORK[9] += 'loss_D2_+'

        if opt.use_segmentation_network:
            self.lambda_segmentation_loss = opt.lambda_segmentation_loss
            self.LOSSES[10] = self.segmentation_loss
            self.LOSS_INPUTS[10] = ['segmentation_predict, segmentation']
            self.LOSS_RETURNS[10] = 'segmentation_loss'
            self.LOSSES_PER_NETWORKS[10].append(10)
            self.LOSSES_PER_NETWORKS[0].append(10)
            self.RETURNS_PER_NETWORK[10] += 'segmentation_loss+'
            self.RETURNS_PER_NETWORK[0] += 'segmentation_loss+'

        self.RETURNS_PER_NETWORK = [e[:-1] for e in self.RETURNS_PER_NETWORK] # pour virer le + à la fin

    def segmentation_loss(self, prediction, ground_truth):
        ground_truth = ground_truth.squeeze(1)
        return torch.nn.CrossEntropyLoss()(prediction, ground_truth) * self.lambda_segmentation_loss

    def cyclic_anatomy(self, content_LR, content_HR, content_fake_LR, content_fake_HR):
        return torch.nn.L1Loss()(content_HR, content_fake_LR)*self.lambda_cyclic_anatomy, torch.nn.L1Loss()(content_LR, content_fake_HR)*self.lambda_cyclic_anatomy
     
    def cyclic_modality(self, z_LR, z_HR, z_fake_LR, z_fake_HR):
        return torch.nn.L1Loss()(z_HR, z_fake_HR)*self.lambda_cyclic_modality , torch.nn.L1Loss()(z_LR, z_fake_LR)*self.lambda_cyclic_modality
    
    def n_pair(self, content_LR, content_HR, content_fake_LR, content_fake_HR):
        L = N_pair_Loss()
        batch_size = content_HR.shape[0]
        content_LR = content_LR.reshape(batch_size, -1)
        content_HR = content_HR.reshape(batch_size, -1)
        content_fake_LR = content_fake_LR.reshape(batch_size, -1)
        content_fake_HR = content_fake_HR.reshape(batch_size, -1)

        content_LR_norm = torch.nn.functional.normalize(content_LR, dim=1) # borne les vecteurs pour qu'ils aient une norme de 1
        content_HR_norm = torch.nn.functional.normalize(content_HR, dim=1)
        content_fake_LR_norm = torch.nn.functional.normalize(content_fake_LR, dim=1)
        content_fake_HR_norm = torch.nn.functional.normalize(content_fake_HR, dim=1)
        loss_anat = L(features1 = content_HR_norm, features2=content_LR_norm)*self.lambda_Npair
        loss_anatfake = L(features1 = content_fake_HR_norm, features2=content_fake_LR_norm)*self.lambda_Npair

        return loss_anat, loss_anatfake
    
    def contrastive_loss(self, z_LR, z_HR, z_fake_LR, z_fake_HR):
        L = SupervisedContrastiveLoss()
        batch_size = z_HR.shape[0]
        # labels = torch.cat([torch.full((batch_size,), 0, dtype=torch.float), torch.full((batch_size,), 1, dtype=torch.float)], dim=0).to(torch.device("cuda:"+str(self.gpu)))
        labels = torch.cat([torch.full((batch_size,), 0, dtype=torch.float), torch.full((batch_size,), 1, dtype=torch.float)], dim=0).to(self.Device)
        
        z_LR = z_LR.reshape(batch_size, -1)
        z_HR = z_HR.reshape(batch_size, -1)
        z_fake_LR = z_fake_LR.reshape(batch_size, -1)
        z_fake_HR = z_fake_HR.reshape(batch_size, -1)

        modality_norm = torch.nn.functional.normalize(torch.cat([z_HR, z_LR], dim = 0).unsqueeze(1), dim = 2)
        modalityfake_norm = torch.nn.functional.normalize(torch.cat([z_fake_HR, z_fake_LR], dim = 0).unsqueeze(1), dim = 2)
        loss_modality = L(features = modality_norm, labels = labels)*self.lambda_contrastive_modality
        loss_modalityfake = L(features = modalityfake_norm, labels = labels)*self.lambda_contrastive_modality
        return loss_modality, loss_modalityfake

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
        return loss_D*self.lambda_D_domain

    def backward_D_content(self, imageA, imageB, discriminator_content):
        pred_fake = discriminator_content.forward(imageA.detach())
        pred_real = discriminator_content.forward(imageB.detach())
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a)
            out_real = torch.sigmoid(out_b)
            all1 = torch.ones((out_real.size(0))).cuda(self.gpu)
            all0 = torch.zeros((out_fake.size(0))).cuda(self.gpu)
            ad_true_loss = nn.functional.binary_cross_entropy_with_logits(out_real, all1)
            ad_fake_loss = nn.functional.binary_cross_entropy_with_logits(out_fake, all0)
        loss_D = ad_true_loss + ad_fake_loss
        return loss_D*self.lambda_D_content

    def backward_G_GAN_content(self, z_content, discriminator_content):
        outs = discriminator_content.forward(z_content)
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

    def backward_EG(self, x, y, fake_LR, fake_HR, fake_LRLR, fake_HRHR, est_LR, est_HR, z_LR, z_HR, content_LR, content_HR, z_fake_LR, z_fake_HR, content_fake_LR, content_fake_HR, discriminator_content, discriminator_LR, discriminator_HR):
        # content Ladv for generator
        loss_G_GAN_LRcontent = self.backward_G_GAN_content(content_LR, discriminator_content)*self.lambda_adv_anatomy_encoder
        loss_G_GAN_HRcontent = self.backward_G_GAN_content(content_HR, discriminator_content)*self.lambda_adv_anatomy_encoder

        # Ladv for generator
        loss_G_GAN_LR = self.backward_G_GAN(fake_LR, discriminator_LR)*self.lambda_adv_generator
        loss_G_GAN_HR = self.backward_G_GAN(fake_HR, discriminator_HR)*self.lambda_adv_generator

        # KL loss - z_a
        #loss_kl_za_LR = self._l2_regularize(z_fake_LR) * 0.01
        loss_kl_za_LR = self._l2_regularize(z_LR) * self.lambda_reg # torch.Size([128, 8])
        loss_kl_za_HR = self._l2_regularize(z_HR) * self.lambda_reg # torch.Size([128, 8])

        # KL loss - z_c
        loss_kl_zc_LR = self._l2_regularize(content_LR) * self.lambda_reg
        loss_kl_zc_HR = self._l2_regularize(content_HR) * self.lambda_reg

        # cross cycle consistency loss
        loss_G_L1_LR = torch.nn.L1Loss()(est_LR, x) * self.lambda_cross_cycle
        loss_G_L1_HR = torch.nn.L1Loss()(est_HR, y) * self.lambda_cross_cycle
        loss_G_L1_LRLR = torch.nn.L1Loss()(fake_LRLR, x) * self.lambda_self
        loss_G_L1_HRHR = torch.nn.L1Loss()(fake_HRHR, y) * self.lambda_self

        loss_G_LR = loss_G_GAN_LR + \
                loss_G_GAN_LRcontent + \
                loss_G_L1_LRLR + \
                loss_G_L1_LR + \
                loss_kl_zc_LR + \
                loss_kl_za_LR + \
                loss_G_L1_HR #############
        
        loss_G_HR = loss_G_GAN_HR + \
                loss_G_GAN_HRcontent + \
                loss_G_L1_HRHR + \
                loss_G_L1_HR + \
                loss_kl_zc_HR + \
                loss_kl_za_HR + \
                loss_G_L1_LR ################
        return(loss_G_LR, loss_G_HR)

    def backward_G_alone(self,fake_LR_random, fake_HR_random, z_random_LR, z_random_HR, z_random, discriminator_LR2, discriminator_HR2):
        
        if self.random_size!=0: 
            # Ladv for generator
            loss_G_GAN2_LR = self.backward_G_GAN(fake_LR_random, discriminator_LR2)
            loss_G_GAN2_HR = self.backward_G_GAN(fake_HR_random, discriminator_HR2)
            # latent regression loss
            loss_z_L1_LR = torch.mean(torch.abs(z_random_LR - z_random)) * self.lambda_latent
            loss_z_L1_HR = torch.mean(torch.abs(z_random_HR - z_random)) * self.lambda_latent

            loss_z_L1 = loss_z_L1_LR + loss_z_L1_HR + loss_G_GAN2_LR + loss_G_GAN2_HR
            loss_z_LR= loss_z_L1_LR + loss_G_GAN2_LR 
            loss_z_HR= loss_z_L1_HR + loss_G_GAN2_HR

            return loss_z_LR, loss_z_HR

        else:
            # latent regression loss
            loss_z_L1_LR = torch.mean(torch.abs(z_random_LR - z_random)) * self.lambda_latent
            loss_z_L1_HR = torch.mean(torch.abs(z_random_HR - z_random)) * self.lambda_latent

            loss_z_L1 = loss_z_L1_LR + loss_z_L1_HR 
            loss_z_LR= loss_z_L1_LR 
            loss_z_HR= loss_z_L1_HR 
            return loss_z_LR, loss_z_HR

    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss
      
    def forward(self, manual_backward, log, optimizer, LR, HR, fake_LR, fake_HR, fake_LRLR, fake_HRHR, est_LR, est_HR, z_LR, z_HR, content_LR, content_HR, z_fake_LR, z_fake_HR, content_fake_LR, content_fake_HR, z_random, fake_LR_random, fake_HR_random, z_random_LR, z_random_HR, LR_random, HR_random, discriminator_content, discriminator_LR, discriminator_HR, anatomy_encoder, modality_encoder, encode_content, encode_LR, encode_HR, generator_LR, generator_HR, discriminator_LR2=None, discriminator_HR2=None, segmentation=None, segmentation_predict=None, segmentation_net=None):
        '''
        The goal is to compute only the necessary thing at each step. Since this function is called for each training step, 
        for every optimizer, we try to reduce the ocmputation time by compute only the needed functions
        '''
        encode_content.requires_grad_(False)
        encode_LR.requires_grad_(False)
        encode_HR.requires_grad_(False)
        generator_LR.requires_grad_(False)
        generator_HR.requires_grad_(False)
        discriminator_LR.requires_grad_(False)
        discriminator_HR.requires_grad_(False)
        discriminator_content.requires_grad_(False)
        loc = locals()

        if self.random_size!= 0 and self.use_segmentation_network:
            optimizer_encode_content, optimizer_encode_HR, optimizer_encode_LR, optimizer_generator_HR, optimizer_generator_LR, optimizer_discriminator_content, optimizer_discriminator_LR, optimizer_discriminator_HR, optimizer_discriminator_LR2, optimizer_discriminator_HR2, optimizer_segmentation_net = optimizer
            discriminator_LR2.requires_grad_(False)
            discriminator_HR2.requires_grad_(False)
            segmentation_net.requires_grad_(False)

            #optimizer_discriminator_LR2
            optimizer_idx = 8
            discriminator_LR2.requires_grad_(True)
            for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
                loc['i']=i
                exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
            exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
            log('Discriminateur 2 LR/T1', loc['loss'])
            optimizer_discriminator_LR2.zero_grad()
            manual_backward(loc['loss'])
            optimizer_discriminator_LR2.step()
            discriminator_LR2.requires_grad_(False)
            
            #optimizer_discriminator_HR2
            optimizer_idx = 9
            discriminator_HR2.requires_grad_(True)
            for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
                loc['i']=i
                exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][1])+")", globals(), loc)
            exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
            log('Discriminateur 2 HR/T2', loc['loss'])
            optimizer_discriminator_HR2.zero_grad()
            manual_backward(loc['loss'])
            optimizer_discriminator_HR2.step()
            discriminator_HR2.requires_grad_(False)

            optimizer_idx = 10
            segmentation_net.requires_grad_(True)
            for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
                loc['i']=i
                exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
            exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
            log('Segmentation network', loc['loss'])
            optimizer_segmentation_net.zero_grad()
            manual_backward(loc['loss'], retain_graph = True)
            optimizer_segmentation_net.step()
            segmentation_net.requires_grad_(False)


        elif self.random_size != 0 and (not self.use_segmentation_network):
            optimizer_encode_content, optimizer_encode_HR, optimizer_encode_LR, optimizer_generator_HR, optimizer_generator_LR, optimizer_discriminator_content, optimizer_discriminator_LR, optimizer_discriminator_HR, optimizer_discriminator_LR2, optimizer_discriminator_HR2 = optimizer
            discriminator_LR2.requires_grad_(False)
            discriminator_HR2.requires_grad_(False)

            #optimizer_discriminator_LR2
            optimizer_idx = 8
            discriminator_LR2.requires_grad_(True)
            for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
                loc['i']=i
                exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
            exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
            log('Discriminateur 2 LR/T1', loc['loss'])
            optimizer_discriminator_LR2.zero_grad()
            manual_backward(loc['loss'])
            optimizer_discriminator_LR2.step()
            discriminator_LR2.requires_grad_(False)
            
            #optimizer_discriminator_HR2
            optimizer_idx = 9
            discriminator_HR2.requires_grad_(True)
            for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
                loc['i']=i
                exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][1])+")", globals(), loc)
            exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
            log('Discriminateur 2 HR/T2', loc['loss'])
            optimizer_discriminator_HR2.zero_grad()
            manual_backward(loc['loss'])
            optimizer_discriminator_HR2.step()
            discriminator_HR2.requires_grad_(False)


        elif self.random_size == 0 and self.use_segmentation_network:
            optimizer_encode_content, optimizer_encode_HR, optimizer_encode_LR, optimizer_generator_HR, optimizer_generator_LR, optimizer_discriminator_content, optimizer_discriminator_LR, optimizer_discriminator_HR, optimizer_segmentation_net = optimizer
            optimizer_idx = 10
            segmentation_net.requires_grad_(True)
            for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
                loc['i']=i
                exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
            exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
            log('Segmentation network', loc['loss'])
            optimizer_segmentation_net.zero_grad()
            # print(loc['loss'])
            manual_backward(loc['loss'], retain_graph=True)
            optimizer_segmentation_net.step()
            segmentation_net.requires_grad_(False)


        elif self.random_size == 0 and (not self.use_segmentation_network):
            optimizer_encode_content, optimizer_encode_HR, optimizer_encode_LR, optimizer_generator_HR, optimizer_generator_LR, optimizer_discriminator_content, optimizer_discriminator_LR, optimizer_discriminator_HR = optimizer


        else:
            sys.exit('Error: please check the last version of the custom loss')

        # ENCODE CONTENT
        optimizer_idx = 0
        encode_content.requires_grad_(True)
        for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
            loc['i']=i
            exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
        exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
        log('Encodeur anatomie', loc['loss'])
        optimizer_encode_content.zero_grad()
        manual_backward(loc['loss'], retain_graph = True)
        optimizer_encode_content.step()
        encode_content.requires_grad_(False)

        #encoder HR
        optimizer_idx = 1
        encode_HR.requires_grad_(True)
        for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
            loc['i']=i
            exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
        exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
        log('Encodeur HR/T2', loc['loss'])
        optimizer_encode_HR.zero_grad()
        manual_backward(loc['loss'], retain_graph = True)
        optimizer_encode_HR.step()
        encode_HR.requires_grad_(False)

        #encoder LR
        optimizer_idx = 2
        encode_LR.requires_grad_(True)
        for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
            loc['i']=i
            exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
        exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
        log('Encodeur LR/T1', loc['loss'])
        optimizer_encode_LR.zero_grad()
        manual_backward(loc['loss'], retain_graph = True)
        optimizer_encode_LR.step()
        encode_LR.requires_grad_(False)

        
        #generator HR
        optimizer_idx = 3
        generator_HR.requires_grad_(True)
        for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
            loc['i']=i
            exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
        exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
        log('Generateur HR/T2', loc['loss'])
        optimizer_generator_HR.zero_grad()
        manual_backward(loc['loss'], retain_graph = True)
        optimizer_generator_HR.step()
        generator_HR.requires_grad_(False)
        
        #generator LR
        optimizer_idx = 4
        generator_LR.requires_grad_(True)
        for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
            loc['i']=i
            exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
        exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
        log('Generateur LR/T1', loc['loss'])
        optimizer_generator_LR.zero_grad()
        manual_backward(loc['loss'])
        optimizer_generator_LR.step()
        generator_LR.requires_grad_(False)
        
        #discriminator content
        optimizer_idx = 5
        discriminator_content.requires_grad_(True)
        for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
            loc['i']=i
            exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
        exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
        log('Discriminateur anatomie', loc['loss'])
        nn.utils.clip_grad_norm_(discriminator_content.parameters(), 5)
        optimizer_discriminator_content.zero_grad()
        manual_backward(loc['loss'])
        optimizer_discriminator_content.step()
        discriminator_content.requires_grad_(False)
        
        #discriminateur LR
        optimizer_idx = 6
        discriminator_LR.requires_grad_(True)
        for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
            loc['i']=i
            exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][0])+")", globals(), loc)
        exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
        log('Discriminateur LR/T1', loc['loss'])
        optimizer_discriminator_LR.zero_grad()
        manual_backward(loc['loss'])
        optimizer_discriminator_LR.step()
        discriminator_LR.requires_grad_(False)
        
        #discriminator HR
        optimizer_idx = 7
        discriminator_HR.requires_grad_(True)
        for i in self.LOSSES_PER_NETWORKS[optimizer_idx]:
            loc['i']=i
            exec(str(self.LOSS_RETURNS[i])+" = self.LOSSES[i]("+str(self.LOSS_INPUTS[i][1])+")", globals(), loc)
        exec('loss = '+str(self.RETURNS_PER_NETWORK[optimizer_idx]), globals(), loc)
        log('Discriminateur HR/T2', loc['loss'])
        optimizer_discriminator_HR.zero_grad()
        manual_backward(loc['loss'])
        optimizer_discriminator_HR.step()
        discriminator_HR.requires_grad_(False)

        encode_content.requires_grad_(True)
        encode_LR.requires_grad_(True)
        encode_HR.requires_grad_(True)
        generator_LR.requires_grad_(True)
        generator_HR.requires_grad_(True)
        discriminator_LR.requires_grad_(True)
        discriminator_HR.requires_grad_(True)
        discriminator_content.requires_grad_(True)
        if self.random_size != 0:
            discriminator_LR2.requires_grad_(True)
            discriminator_HR2.requires_grad_(True)
        if self.use_segmentation_network:
            segmentation_net.requires_grad_(True)
        





##########################################################################################################################################################################################################
######### MODEL ##########################################################################################################################################################################################
##########################################################################################################################################################################################################

class DRIT(pl.LightningModule):
    #def __init__(self, criterion, learning_rate, optimizer_class, dataset, prefix, segmentation, gpu, mode, lambda_contrastive=1, reduce=False,  n_channels = 1, n_classes = 1, n_features = 32):
    def __init__(self, opt, prefix, isTrain):
        super().__init__()
        self.automatic_optimization = False
        if isTrain:
            self.lr = opt.learning_rate
            self.lr_dcontent = self.lr / 2.5
            self.criterion = torch.nn.L1Loss()
            self.optimizer_class = torch.optim.Adam
            self.prefix=prefix
            self.random_size = opt.random_size
            self.saving_path = opt.saving_path
            self.saving_ratio = opt.saving_ratio
        else:
            self.random_size=0
            self.mode = opt.mode

        if opt.gpu >=0:
            self.Device = torch.device("cuda:"+str(opt.gpu))
            self.gpu = opt.gpu
        else:
            self.Device = torch.device('cpu')

        print('Method: ', opt.method)

        self.use_segmentation_network = opt.use_segmentation_network
        if opt.use_segmentation_network:
            if opt.data == 'segmentation_equinus_256':
                self.segmentation_net = UNet(n_classes=2)
            elif opt.data == 'bone_segmentation_equinus_256' or opt.data == 'custom' or opt.data == 'custom_simulate':
                self.segmentation_net = UNet(n_classes=4)
        else:
            self.segmentation_net = None


        if opt.modality_encoder == 'DRITPP':
            self.encode_HR=Encoder_representation()
            self.encode_LR=Encoder_representation()
        elif opt.modality_encoder == 'ResNet50':
            self.encode_HR = resnet50(pretrained=True)
            self.encode_HR.requires_grad_(requires_grad=False)
            for params in self.encode_HR.parameters():
                if params.requires_grad == True:
                    sys.exit('Requires grad = True in resnet50')
            self.encode_LR = self.encode_HR
            # self.encode_LR = resnet50(pretrained=True)
            # self.encode_LR.requires_grad_(requires_grad=False)



        if opt.anatomy_encoder == 'DRITPP_reduceplus':
            self.encode_content=Encoder_content_reduceplus()
        elif opt.anatomy_encoder == 'ResNet50' and opt.modality_encoder != 'ResNet50':
            self.encode_content = resnet50(pretrained=True)
            self.encode_content.requires_grad_(requires_grad=False)
        elif opt.anatomy_encoder == 'ResNet50' and opt.modality_encoder == 'ResNet50':
            self.encode_content = self.encode_HR


        if opt.method == 'FILM2':
            print('FiLM generator')
            self.generator_HR=Generator_FiLM()
            self.generator_LR=Generator_FiLM()
        elif opt.method=='ADAIN2':
            print('AdaIN generator')
            self.generator_HR=Generator_AdaIN()
            self.generator_LR=Generator_AdaIN()
        elif opt.method=='DRITPP2':
            print('DRIT++ generator')
            self.generator_HR=Generator_feature_reduceplus()
            self.generator_LR=Generator_feature_reduceplus()
        elif opt.method == 'FILM1':
            print('FiLM generator')
            self.generator_HR=Generator_FiLM_1()
            self.generator_LR=Generator_FiLM_1()
        elif opt.method=='ADAIN1':
            print('AdaIN generator')
            self.generator_HR=Generator_AdaIN_1()
            self.generator_LR=Generator_AdaIN_1()
        elif opt.method=='DRITPP1':
            print('DRIT++ generator')
            self.generator_HR=Generator_feature_reduceplus_1()
            self.generator_LR=Generator_feature_reduceplus_1()
        elif opt.method=='Concat':
            print('Concat generator')
            self.generator_HR=Generator_Concat(n_channels=512)
            self.generator_LR=Generator_Concat(n_channels=512)
        elif opt.method == 'SPADE':
            self.generator_HR=Generator_SPADE()
            self.generator_LR=Generator_SPADE()
        elif opt.method=='ADAATTN':
            print('AdaAttN Generator')
        

        self.discriminator_content=Discriminateur_content_reduce()

        if opt.discriminator == 'DRIT':
            self.discriminator_LR=Discriminateur_reduceplus(norm=opt.norm_discrim)
            self.discriminator_HR=Discriminateur_reduceplus(norm=opt.norm_discrim)
        elif opt.discriminator == 'PatchGAN':
            # # # init nlyaer = 3
            self.discriminator_HR = NLayerDiscriminator(input_nc=1, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d)
            self.discriminator_LR = NLayerDiscriminator(input_nc=1, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d)
        

        if self.random_size!=0:
            print("Utilise les doubles discriminateurs")
            self.discriminator_HR2 = Discriminateur_reduceplus()
            self.discriminator_LR2 = Discriminateur_reduceplus()
        else:
            self.discriminator_HR2 = None
            self.discriminator_LR2 = None


        if not opt.not_use_reduce:
            self.discriminator_content=Discriminateur_content()
            self.discriminator_LR=Discriminateur()
            self.discriminator_HR=Discriminateur()
            self.generator_HR = Generator_feature()
            self.generator_LR = Generator_feature()
            self.encode_content=Encoder_content()
            if self.random_size!=0:
                print("Utilise les doubles discriminateurs")
                self.discriminator_HR2 = Discriminateur()
                self.discriminator_LR2 = Discriminateur()



        if isTrain:
            self.activation = {}
            self.anatomy_encoder = opt.anatomy_encoder
            self.modality_encoder = opt.modality_encoder
            
            self.custom_loss = Loss_DRIT_custom_last_version(opt, gpu=self.gpu)        


    def getActivation(self,name):
        # the hook signature
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def initialize(self):
        self.encode_content.apply(gaussian_weights_init)
        self.encode_HR.apply(gaussian_weights_init)
        self.encode_LR.apply(gaussian_weights_init)
        self.generator_HR.apply(gaussian_weights_init)
        self.generator_LR.apply(gaussian_weights_init)
        self.discriminator_content.apply(gaussian_weights_init)
        self.discriminator_LR.apply(gaussian_weights_init)
        self.discriminator_HR.apply(gaussian_weights_init)
        if self.random_size!=0:
            self.discriminator_LR2.apply(gaussian_weights_init)
            self.discriminator_HR2.apply(gaussian_weights_init)
        if self.use_segmentation_network:
            self.segmentation_net.apply(gaussian_weights_init)


    def forward(self, x, y):
        x=x.squeeze(-1) 
        y=y.squeeze(-1)
        z_LR=self.encode_LR(x)
        z_HR=self.encode_HR(y)
        content_LR,content_HR=self.encode_content.forward(x,y)
        fake_HR=self.generator_HR(content_LR, z_HR)
        fake_LR=self.generator_LR(content_HR, z_LR)
        if self.mode=='reconstruction':
            return(fake_HR)
        elif self.mode=='degradation':
            return(fake_LR)
        elif self.mode == 'both':
            return(fake_HR, fake_LR)
        else:
            sys.exit("Précisez un mode valide pour l'inférence: reconstruction ou degradation")



    def get_latent(self, x, y):
        x=x.squeeze(-1) 
        y=y.squeeze(-1)
        z_LR=self.encode_LR(x)
        z_HR=self.encode_HR(y)
        content_LR,content_HR=self.encode_content.forward(x,y)
        fake_HR=self.generator_HR(content_LR, z_HR)
        fake_LR=self.generator_LR(content_HR, z_LR)
        z_fake_LR=self.encode_LR(fake_LR)
        z_fake_HR=self.encode_HR(fake_HR)
        content_fake_LR,content_fake_HR=self.encode_content.forward(fake_LR,fake_HR)
        return(fake_HR, fake_LR, content_HR, content_LR, z_HR, z_LR, z_fake_HR, z_fake_LR, content_fake_HR, content_fake_LR)

    def configure_optimizers(self):
        if self.anatomy_encoder!='ResNet50':
            optimizer_encode_content=self.optimizer_class(self.encode_content.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        else:
            optimizer_encode_content = None
        if self.modality_encoder!='ResNet50':
            optimizer_encode_HR=self.optimizer_class(self.encode_HR.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
            optimizer_encode_LR=self.optimizer_class(self.encode_LR.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        else:
            optimizer_encode_HR=None
            optimizer_encode_LR=None
        
        optimizer_generator_HR=self.optimizer_class(self.generator_HR.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        optimizer_generator_LR=self.optimizer_class(self.generator_LR.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        optimizer_discriminator_content=self.optimizer_class(self.discriminator_content.parameters(), lr=self.lr_dcontent, betas=(0.5, 0.999), weight_decay=0.0001)
        optimizer_discriminator_LR=self.optimizer_class(self.discriminator_LR.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        optimizer_discriminator_HR=self.optimizer_class(self.discriminator_HR.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
        if self.random_size!=0: 
            optimizer_discriminator_LR2 = self.optimizer_class(self.discriminator_LR2.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
            optimizer_discriminator_HR2 = self.optimizer_class(self.discriminator_HR2.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
            if self.use_segmentation_network: 
                optimizer_segmentation_net = self.optimizer_class(self.segmentation_net.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
                return optimizer_encode_content, optimizer_encode_HR, optimizer_encode_LR, optimizer_generator_HR, optimizer_generator_LR, optimizer_discriminator_content, optimizer_discriminator_LR, optimizer_discriminator_HR, optimizer_discriminator_LR2, optimizer_discriminator_HR2, optimizer_segmentation_net
            else:
                return optimizer_encode_content, optimizer_encode_HR, optimizer_encode_LR, optimizer_generator_HR, optimizer_generator_LR, optimizer_discriminator_content, optimizer_discriminator_LR, optimizer_discriminator_HR, optimizer_discriminator_LR2, optimizer_discriminator_HR2

        else:
            if self.use_segmentation_network: 
                optimizer_segmentation_net = self.optimizer_class(self.segmentation_net.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001)
                return optimizer_encode_content, optimizer_encode_HR, optimizer_encode_LR, optimizer_generator_HR, optimizer_generator_LR, optimizer_discriminator_content, optimizer_discriminator_LR, optimizer_discriminator_HR, optimizer_segmentation_net
            else:
                return optimizer_encode_content, optimizer_encode_HR, optimizer_encode_LR, optimizer_generator_HR, optimizer_generator_LR, optimizer_discriminator_content, optimizer_discriminator_LR, optimizer_discriminator_HR


    def prepare_batch(self, batch):
        if self.use_segmentation_network:
            return batch['LR_image'][tio.DATA], batch['HR_image'][tio.DATA], batch['label'][tio.DATA], batch['segmentations'][tio.DATA]
            #return batch['LR_image'][tio.DATA], batch['HR_image'][tio.DATA], batch['label'][tio.DATA], batch['calcaneus_segmentation'][tio.DATA], batch['talus_segmentation'][tio.DATA], batch['tibia_segmentation'][tio.DATA]
        else:
            return batch['LR_image'][tio.DATA], batch['HR_image'][tio.DATA], batch['label'][tio.DATA]


    def infer_batch(self, batch):
        if self.use_segmentation_network:
            x, y, mask, segmentation = self.prepare_batch(batch)
            segmentation= segmentation.long().squeeze(-1)
        else:
            x,y, mask = self.prepare_batch(batch)
            segmentation = None

        x=x.squeeze(-1) 
        y=y.squeeze(-1)

        x_random = x[:self.random_size]
        y_random = y[:self.random_size]
        x = x[self.random_size:]
        y = y[self.random_size:]

        #disentanglement
        if self.modality_encoder == 'ResNet50' and self.anatomy_encoder=='ResNet50':
            h = self.encode_HR.layer1.register_forward_hook(self.getActivation('activation'))
            print(x.shape)
            #test = x.expand(x.shape[0], 3, x.shape[2], x.shape[3])
            # print(torch.equal(test[:,0,:,:], test[:,1,:,:]) and torch.equal(test[:,0,:,:], test[:,2,:,:]))
            # sys.exit()
            out=self.encode_HR(x.expand(x.shape[0], 3, x.shape[2], x.shape[3]))
            z_LR = self.activation['activation']
            out=self.encode_HR(y.expand(x.shape[0], 3, x.shape[2], x.shape[3]))
            z_HR = self.activation['activation']
            content_LR = z_LR
            content_HR = z_HR
        else:
            z_LR=self.encode_LR(x)
            z_HR=self.encode_HR(y)
            content_LR,content_HR=self.encode_content.forward(x,y)
        z_random = self.get_z_random(z_LR.shape, 'gauss')

        if self.use_segmentation_network:
            segmentation_predict = self.segmentation_net(content_HR)
        else:
            segmentation_predict = None
        
        #generation
        fake_HR=self.generator_HR(content_LR, z_HR)
        fake_LR=self.generator_LR(content_HR, z_LR)
        
        fake_HRHR=self.generator_HR(content_HR, z_HR)
        fake_LRLR=self.generator_LR(content_LR, z_LR)
        fake_LR_random=self.generator_LR(content_HR,z_random)
        fake_HR_random=self.generator_HR(content_LR,z_random)

        #disentanglement
        if self.modality_encoder == 'ResNet50' and self.anatomy_encoder=='ResNet50':
            out=self.encode_HR(fake_LR.expand(x.shape[0], 3, x.shape[2], x.shape[3]))
            z_fake_LR = self.activation['activation']
            out=self.encode_HR(fake_HR.expand(x.shape[0], 3, x.shape[2], x.shape[3]))
            z_fake_HR = self.activation['activation']
            content_fake_LR = z_fake_LR
            content_fake_HR = z_fake_HR
            h.remove()
        else:
            z_fake_LR=self.encode_LR(fake_LR)
            z_fake_HR=self.encode_HR(fake_HR)
            content_fake_LR, content_fake_HR=self.encode_content.forward(fake_LR, fake_HR)

        #generation
        est_HR=self.generator_HR(content_fake_LR, z_fake_HR)
        est_LR=self.generator_LR(content_fake_HR, z_fake_LR)
        z_random_LR=self.encode_LR(fake_LR_random)
        z_random_HR=self.encode_HR(fake_HR_random)

        return x, y, fake_LR, fake_HR, fake_LRLR, fake_HRHR, est_LR, est_HR, z_LR, z_HR, content_LR, content_HR, z_fake_LR, z_fake_HR, content_fake_LR, content_fake_HR, z_random, fake_LR_random, fake_HR_random, z_random_LR, z_random_HR, x_random, y_random, segmentation, segmentation_predict





    def training_step(self, batch, batch_idx):
        LR, HR, fake_LR, fake_HR, fake_LRLR, fake_HRHR, est_LR, est_HR, z_LR, z_HR, content_LR, content_HR, z_fake_LR, z_fake_HR, content_fake_LR, content_fake_HR, z_random, fake_LR_random, fake_HR_random, z_random_LR, z_random_HR, LR_random, HR_random, segmentation, segmentation_predict = self.infer_batch(batch)
        batch_size=LR.shape[0]
        self.true_label = torch.full((batch_size,), 1, dtype=torch.float)
        self.fake_label = torch.full((batch_size,), 0, dtype=torch.float)
        # self.true_label=self.true_label.to(torch.device("cuda:"+str(self.gpu)))
        # self.fake_label=self.fake_label.to(torch.device("cuda:"+str(self.gpu)))
        self.true_label=self.true_label.to(self.Device)
        self.fake_label=self.fake_label.to(self.Device)
        font = {'fontsize':8}

        if self.saving_ratio!=None and batch_idx%self.saving_ratio==0:
            if self.use_segmentation_network:
                plt.figure()
                plt.suptitle('epoch: '+str(self.current_epoch)+' batch_idx: '+str(batch_idx)+'     '+self.prefix, fontsize=10)
                plt.subplot(2,4,1)
                plt.title('Dynamic', fontdict=font)
                plt.imshow(LR[0,0,:,:].cpu().detach().numpy(), cmap="gray")
                plt.axis('off')
                plt.subplot(2,4,2)
                plt.title('Reconstruct dynamic', fontdict=font)
                plt.imshow(est_LR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
                plt.axis('off')
                plt.subplot(2,4,3)
                plt.title('Synthetic static', fontdict=font)
                test=fake_HR
                plt.imshow(test[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
                plt.axis('off')
                plt.subplot(2,4,4)
                plt.title('GT segmentation', fontdict=font)
                #plt.imshow(torch.argmax(segmentation[0,:,:,:], dim=0).cpu().detach().numpy().astype(float), cmap="gray")
                plt.imshow(segmentation[0,:,:,:].cpu().detach().numpy().astype(float), cmap="gray")
                plt.axis('off')
                plt.subplot(2,4,5)
                plt.title('Static', fontdict=font)
                plt.imshow(HR[0,0,:,:].cpu().detach().numpy(), cmap="gray")
                plt.axis('off')
                plt.subplot(2,4,6)
                plt.title('Reconstruct static', fontdict=font)
                plt.imshow(est_HR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
                plt.axis('off')
                plt.subplot(2,4,7)
                plt.title('Synthetic dynamic', fontdict=font)
                plt.imshow(fake_LR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
                plt.axis('off')
                plt.subplot(2,4,8)
                segmentation_predict = nn.Softmax()(segmentation_predict)
                plt.title('Predicted segmentation', fontdict=font)
                plt.imshow(torch.argmax(segmentation_predict[0,:,:,:], dim=0).cpu().detach().numpy().astype(float), cmap='gray')
                plt.axis('off')
                plt.savefig(os.path.join(self.saving_path, 'epoch-'+str(self.current_epoch)+'_batch_idx-'+str(batch_idx)+'.png'))
                plt.close()
            else:
                plt.figure()
                plt.suptitle('epoch: '+str(self.current_epoch)+' batch_idx: '+str(batch_idx)+'     '+self.prefix, fontsize=10)
                plt.subplot(2,3,1)
                plt.title('Dynamic', fontdict=font)
                plt.imshow(LR[0,0,:,:].cpu().detach().numpy(), cmap="gray")
                plt.axis('off')
                plt.subplot(2,3,2)
                plt.title('Reconstruct dynamic', fontdict=font)
                plt.imshow(est_LR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
                plt.axis('off')
                plt.subplot(2,3,3)
                plt.title('Synthetic static', fontdict=font)
                test=fake_HR
                plt.imshow(test[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
                plt.axis('off')
                plt.subplot(2,3,4)
                plt.title('Static', fontdict=font)
                plt.imshow(HR[0,0,:,:].cpu().detach().numpy(), cmap="gray")
                plt.axis('off')
                plt.subplot(2,3,5)
                plt.title('Reconstruct static', fontdict=font)
                plt.imshow(est_HR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
                plt.axis('off')
                plt.subplot(2,3,6)
                plt.title('Synthetic dynamic', fontdict=font)
                plt.imshow(fake_LR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
                plt.axis('off')
                plt.savefig(os.path.join(self.saving_path, 'epoch-'+str(self.current_epoch)+'_batch_idx-'+str(batch_idx)+'.png'))
                plt.close()
        
        self.custom_loss(self.manual_backward, self.log, self.optimizers(), LR, HR, fake_LR, fake_HR, fake_LRLR, fake_HRHR, est_LR, est_HR, z_LR, z_HR, content_LR, content_HR, z_fake_LR, z_fake_HR, content_fake_LR, content_fake_HR, z_random, fake_LR_random, fake_HR_random, z_random_LR, z_random_HR, LR_random, HR_random, self.discriminator_content, self.discriminator_LR, self.discriminator_HR, self.anatomy_encoder, self.modality_encoder, self.encode_content, self.encode_LR, self.encode_HR, self.generator_LR, self.generator_HR, self.discriminator_LR2, self.discriminator_HR2, segmentation, segmentation_predict, self.segmentation_net)

    def get_z_random(self, size, random_type='gauss'):
        z = torch.randn(size).cuda(self.gpu)
        return z

    def validation_step(self, batch, batch_idx):
        random_size = self.random_size
        self.random_size = 0
        x, y, fake_LR, fake_HR, fake_LRLR, fake_HRHR, est_LR, est_HR, z_LR, z_HR, content_LR, content_HR, z_fake_LR, z_fake_HR, content_fake_LR, content_fake_HR, z_random, fake_LR_random, fake_HR_random, z_random_LR, z_random_HR, LR_random, HR_random, segmentation, segmentation_predict = self.infer_batch(batch)        
        self.random_size = random_size
        font = {'fontsize':8}
        
        if self.use_segmentation_network:
            loss = self.criterion(est_HR, y)# + monai.losses.DiceCELoss(include_background=False, softmax=True)(segmentation_predict, segmentation)
            plt.figure()
            plt.suptitle(self.prefix, fontsize=10)
            plt.subplot(2,4,1)
            plt.title('Dynamic', fontdict=font)
            plt.imshow(x[0,0,:,:].cpu().detach().numpy(), cmap="gray")
            plt.axis('off')
            plt.subplot(2,4,2)
            plt.title('Reconstruct dynamic', fontdict=font)
            plt.imshow(est_LR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.axis('off')
            plt.subplot(2,4,3)
            plt.title('Synthetic static', fontdict=font)
            test=fake_HR
            plt.imshow(test[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.axis('off')
            plt.subplot(2,4,4)
            plt.title('GT segmentation', fontdict=font)
            #plt.imshow(torch.argmax(segmentation[0,:,:,:], dim=0).cpu().detach().numpy().astype(float), cmap='gray')
            plt.imshow(segmentation[0,0,:,:].cpu().detach().numpy().astype(float), cmap='gray')
            plt.axis('off')
            plt.subplot(2,4,5)
            plt.title('Static', fontdict=font)
            plt.imshow(y[0,0,:,:].cpu().detach().numpy(), cmap="gray")
            plt.axis('off')
            plt.subplot(2,4,6)
            plt.title('Reconstruct static', fontdict=font)
            plt.imshow(est_HR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.axis('off')
            plt.subplot(2,4,7)
            plt.title('Synthetic dynamic', fontdict=font)
            plt.imshow(fake_LR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.axis('off')
            plt.subplot(2,4,8)
            segmentation_predict = nn.Softmax(dim=1)(segmentation_predict)
            plt.title('Predicted segmentation', fontdict=font)
            plt.imshow(torch.argmax(segmentation_predict[0,:,:,:], dim=0).cpu().detach().numpy().astype(float), cmap='gray')
            plt.axis('off')
            plt.savefig(os.path.join(self.saving_path,'validation_epoch-'+str(self.current_epoch)+'.png'))
            plt.close()
            
        else:
            loss = self.criterion(est_HR, y)
            plt.figure()
            plt.suptitle(self.prefix, fontsize=10)
            plt.subplot(2,3,1)
            plt.title('Dynamic', fontdict=font)
            plt.imshow(x[0,0,:,:].cpu().detach().numpy(), cmap="gray")
            plt.axis('off')
            plt.subplot(2,3,2)
            plt.title('Reconstruct dynamic', fontdict=font)
            plt.imshow(est_LR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.axis('off')
            plt.subplot(2,3,3)
            plt.title('Synthetic static', fontdict=font)
            test=fake_HR
            plt.imshow(test[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.axis('off')
            plt.subplot(2,3,4)
            plt.title('Static', fontdict=font)
            plt.imshow(y[0,0,:,:].cpu().detach().numpy(), cmap="gray")
            plt.axis('off')
            plt.subplot(2,3,5)
            plt.title('Reconstruct static', fontdict=font)
            plt.imshow(est_HR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.axis('off')
            plt.subplot(2,3,6)
            plt.title('Synthetic dynamic', fontdict=font)
            plt.imshow(fake_LR[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
            plt.axis('off')
            plt.savefig(os.path.join(self.saving_path,'validation_epoch-'+str(self.current_epoch)+'.png'))
            plt.close()
        self.log('val_loss', loss)
        return loss

