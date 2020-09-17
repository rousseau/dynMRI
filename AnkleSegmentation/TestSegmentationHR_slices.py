#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:29:57 2020

@author: p20coupe
"""

import argparse
import sys

import joblib
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import statistics

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
import torch.backends.cudnn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
import torch.optim as optim
import torchvision
import random

from ResNet import *
from UNet import *

parser = argparse.ArgumentParser()
parser.add_argument("TestData", help="PATH to testing data")
parser.add_argument("NetworkPATH",help="PATH to the network to use")
parser.add_argument("ResultsDirectory",help="PATH to the results storage directory")
parser.add_argument("Bone",help="Name of the bone (calcaneus/talus/tibia)")
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

Datadirectory = sys.argv[1]
ResultsDirectory = sys.argv[3]
NetworkPath = sys.argv[2]
Bone = sys.argv[4]

def computeQualityMeasures(lP,lT):
	union = np.sum(lP) + np.sum(lT)
	if union==0: return 1
	intersection = np.sum(lP * lT)
	return 2. * intersection / union

def ValidRed3D(testloader,Bone,Network,path,sujet):
    volume3D_terrain=[]
    volume3D_exp=[]
    
    if Network=='UNet':
        net = UNet().to(device)
    elif Network=='ResNet':
        net = ResNet(BasicBlock, [3,4,6]).to(device)
    net.load_state_dict(torch.load(path))
    
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs= inputs.to(device)
        labels = labels.to(device)
		
        outputs = net(inputs)
        gt = outputs[0].cpu().detach().numpy()
        gt[0,:,:][gt[0,:,:]<=0.5]=0
        gt[0,:,:][gt[0,:,:]>0.5]=1
        
        volume3D_terrain.append(labels[0,0].cpu().detach().numpy())
        volume3D_exp.append(gt[0])       
       
	
        if i==30:
            plt.figure()
            plt.imshow(inputs[0,0,:,:].cpu().detach().numpy(), 'gray', interpolation='none')
            plt.imshow(outputs[0,0,:,:].cpu().detach().numpy(), 'OrRd', interpolation='none', alpha=0.6)
            plt.savefig(os.path.join(ResultsDirectory,Bone,'Images','2DSlices_'+Network+'_'+Bone+sujet+str(i)+'.png'),dpi=150)
            
            plt.figure()
            plt.imshow(inputs[0,0,:,:].cpu().detach().numpy(), 'gray', interpolation='none')
            plt.imshow(labels[0,0,:,:].cpu().detach().numpy(), 'OrRd', interpolation='none', alpha=0.6)
            plt.savefig(os.path.join(ResultsDirectory,Bone,'Images','2DSlices_Truth_'+Network+'_'+Bone+sujet+str(i)+'.png'),dpi=150)

    volume3D_terrain = np.array(volume3D_terrain)
    volume3D_exp = np.array(volume3D_exp)
    dice = np.abs(computeQualityMeasures(volume3D_terrain.flatten(),volume3D_exp.flatten()))
    print('dice:'+str(dice))


 
sujets = os.listdir(Datadirectory)
sujets = np.sort(sujets)


for i in range(len(sujets)):
    X_test=[[]]
    Y_test=[[]]  		
    
    #Dataset Validation
    patches= os.listdir(os.path.join(Datadirectory,sujets[i],'DatasetSegmentationHR_slices'))
    for k in range(len(patches)):
        if(patches[k].find('HR_Pipeline')!=-1):
            if X_test[0]==[]:
                X_test[0] = joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetSegmentationHR_slices',patches[k]))
            else:
                X_test[0] = X_test[0]+joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetSegmentationHR_slices',patches[k]))
	   
        if(patches[k].find(Bone.lower())!=-1):
            if Y_test[0]==[]:
                Y_test[0] = joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetSegmentationHR_slices',patches[k]))
            else:
                Y_test[0] = Y_test[0]+joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetSegmentationHR_slices',patches[k]))
	

    Y_test = np.moveaxis(Y_test,0,1)
    X_test = np.moveaxis(X_test,0,1)
    print(np.shape(Y_test))
    print(np.shape(X_test))
    
    
    testset = torch.utils.data.TensorDataset(torch.Tensor(X_test),torch.Tensor(Y_test))
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
    		                                  shuffle=False, pin_memory=use_cuda, num_workers=2)
    if NetworkPath.find('UNet')!=-1 or NetworkPath.find('U-net')!=-1:
        Network = 'UNet'
    elif NetworkPath.find('ResNet')!=-1:
        Network = 'ResNet'
        
    #Create directores for the results 
    if not os.path.exists(os.path.join(ResultsDirectory, Bone)):
        os.mkdir(os.path.join(ResultsDirectory, Bone))
    if not os.path.exists(os.path.join(ResultsDirectory, Bone,'Images')):
        os.mkdir(os.path.join(ResultsDirectory, Bone,'Images'))
    
    ValidRed3D(testloader,Bone,Network,NetworkPath,sujets[i])