#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:00:59 2020

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
parser.add_argument("TrainData", help="PATH to training data")
parser.add_argument("NetworkPATH",help="PATH to the network recording directory")
parser.add_argument("Network",help="Name of the network to use (UNet/ResNet)")
parser.add_argument("Bone",help="Name of the bone (calcaneus/talus/tibia)")
args = parser.parse_args()
#print(args.echo)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

Datadirectory = sys.argv[1]
ResultsDirectory = sys.argv[2]
Network = sys.argv[3]
Bone = sys.argv[4]

batch_size = 8
n_epochs = 5

def computeQualityMeasures(lP,lT):
	union = np.sum(lP) + np.sum(lT)
	if union==0: return 1
	intersection = np.sum(lP * lT)
	return 2. * intersection / union

def TrainRed(trainloader,bone,Network,n):
    batchs_num = []
    loss_value=[]
    dice_value=[]

    if Network=='UNet':
        net = UNet().to(device)
    elif Network=='ResNet':
        net = ResNet(BasicBlock, [3,4,6]).to(device)

    criterion = nn.BCELoss()
    optimizer =  torch.optim.Adam(net.parameters(), lr=0.0005)

    for epoch in range(n_epochs):  
        running_dice = 0.0
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs= inputs.to(device)
            labels = labels.to(device)
            			
            # zero the parameter gradients
            optimizer.zero_grad()
            a = list(labels.size())
            # forward + backward + optimize
            outputs = net(inputs)
            			
            loss = 0
            dice = 0
            
            loss = criterion(outputs, labels.float())
            loss.backward()
            for b in range(a[0]):
                pr= labels[b].cpu().detach().numpy()
                gt = outputs[b].cpu().detach().numpy()
                gt[0,:,:][gt[0,:,:]<=0.5]=0
                gt[0,:,:][gt[0,:,:]>0.5]=1
                dice = dice + np.abs(computeQualityMeasures(pr[0,:,:].flatten(),gt[0,:,:].flatten()))
                				
            dice = dice/a[0]
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            running_dice += dice
                
            if i % 200 == 199:    # print every 200 mini-batches (SDG utilise des batchs des traindata)
                batchs_num.append((i+1)+number*(epoch+1))
                loss_value.append(running_loss/200)
                dice_value.append(running_dice/200)
                print('DICE: '+str(running_dice/200))
                print('[%d, %5d] loss: %.3f' %
			  (epoch + 1, i + 1, running_loss / 200))
                
                running_loss = 0.0
                running_dice = 0.0
    print('Finished Training')
    plt.plot(batchs_num,loss_value)
    plt.title('Trainloss')
    plt.savefig(ResultsDirectory+'/'+bone+'/Images/EvolutionLoss.png')
    plt.close()
    
    plt.plot(batchs_num,dice_value)
    plt.title('DICE')
    plt.xlabel('Number of batches')
    plt.ylabel('DICE')
    plt.savefig(ResultsDirectory+'/'+bone+'/Images/EvolutionDICE.png')
    plt.close()
    
    #Save the train model
    PATH = os.path.join(ResultsDirectory,bone,'pip1_'+Network+'_'+bone+'.pth')
    torch.save(net.state_dict(), PATH)
    
X_train=[[]]
Y_train=[[]]    
sujets = os.listdir(Datadirectory)
sujets = np.sort(sujets)
print(sujets)

for i in range(len(sujets)): 
	
    #Dataset Train
    patches= os.listdir(os.path.join(Datadirectory,sujets[i],'DatasetSegmentationHR_patches'))
    for k in range(len(patches)):
        if(patches[k].find('HR_Pipeline')!=-1):
            if X_train[0]==[]:
                X_train[0] = joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetSegmentationHR_patches',patches[k]))
            else:
                X_train[0] = X_train[0]+joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetSegmentationHR_patches',patches[k]))
	   
        if(patches[k].find(Bone.lower())!=-1):
            if Y_train[0]==[]:
                Y_train[0] = joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetSegmentationHR_patches',patches[k]))
            else:
                Y_train[0] = Y_train[0]+joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetSegmentationHR_patches',patches[k]))
			
Y_train = np.moveaxis(Y_train,0,1)
X_train = np.moveaxis(X_train,0,1)
print(np.shape(Y_train))
print(np.shape(X_train))

#For the graphical representation of the DICE and the loss
number = (len(Y_train)/batch_size) - (len(Y_train)/batch_size)%200

#Load Data
trainset = torch.utils.data.TensorDataset(torch.Tensor(X_train),torch.Tensor(Y_train))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True,pin_memory=use_cuda, num_workers=2)

#Create directores for the results 
if not os.path.exists(os.path.join(ResultsDirectory, Bone)):
    os.mkdir(os.path.join(ResultsDirectory, Bone))
if not os.path.exists(os.path.join(ResultsDirectory, Bone,'Images')):
    os.mkdir(os.path.join(ResultsDirectory, Bone,'Images'))
    
TrainRed(trainloader,Bone,Network,number)