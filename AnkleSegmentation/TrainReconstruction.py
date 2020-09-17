#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:45:04 2020

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
import math

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


from ResNet_Reconstruction import *

parser = argparse.ArgumentParser()
parser.add_argument("TrainData", help="PATH to training data")
parser.add_argument("NetworkPATH",help="PATH to the network recording directory")
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

Datadirectory = sys.argv[1]
ResultsDirectory = sys.argv[2]

batch_size = 16
n_epochs = 5

def psnr(lP,lT):
    mse = np.mean( (lP - lT) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 3.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def g(corr):
	y = 1.0/(1+np.exp(-10*(corr-0.5)))
	return y

def TrainRed(trainloader,n):
    batchs_num = []
    loss_value=[]
    psnr_value=[]
    
    net = ResNet(BasicBlock, [3,4,6]).to(device)
    
    criterion = nn.L1Loss(reduction='sum')
    optimizer =  torch.optim.Adam(net.parameters(), lr=0.001)
    
    for epoch in range(n_epochs):  # loop over the dataset train 2 times
        running_psnr = 0.0
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, correlation = data
            inputs= inputs.to(device)
            labels = labels.to(device)
            correlation = correlation.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            a = list(correlation.size())
            # forward + backward + optimize
            outputs = net(inputs)
		
            loss = 0
            psnr_val=0
            
            for b in range(a[0]):
                loss = loss + g(correlation[b].cpu()).to(device)*criterion(outputs[b], labels[b])
                pr= labels[b].cpu().detach().numpy()
                gt = outputs[b].cpu().detach().numpy()
                psnr_val = psnr_val + psnr(pr,gt)
            loss = loss/a[0]
            psnr_val = psnr_val/a[0]
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_psnr += psnr_val
            if i % 500 == 499:    # print every 500 mini-batches (SDG utilise des batchs des traindata)
                batchs_num.append((i+1)+n*(epoch+1))
                loss_value.append(running_loss/500)
                psnr_value.append(running_psnr/500)
                print(running_psnr/500)
                print('[%d, %5d] loss: %.3f' %
			  (epoch + 1, i + 1, running_loss / 500))
                running_loss = 0.0
                running_psnr = 0.0

    print('Finished Training')    
    
    plt.plot(batchs_num,loss_value)
    plt.title('Trainloss')
    plt.savefig(ResultsDirectory+'/Images/EvolutionLoss.png')
    plt.close()
    
    plt.plot(batchs_num,psnr_value)
    plt.title('PSNR')
    plt.xlabel('Number of batches')
    plt.ylabel('PSNR')
    plt.savefig(ResultsDirectory+'/Images/EvolutionPSNR.png')
    plt.close()
    
    #Save the train model
    PATH = os.path.join(ResultsDirectory,'pip1_ReconstructionResNet.pth')
    torch.save(net.state_dict(), PATH)

bones=['calcaneus','talus','tibia']    
X_train=[[]]
Y_train=[[]] 
corr=[]   
sujets = os.listdir(Datadirectory)
sujets = np.sort(sujets)
print(sujets)

for i in range(len(sujets)): 	
    #Dataset Train
    for bone in bones:
        patches= os.listdir(os.path.join(Datadirectory,sujets[i],'DatasetReconstruction_patches',bone))
        for k in range(len(patches)):
            if(patches[k].find('BR')!=-1):
                if X_train[0]==[]:
                    X_train[0] = joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetReconstruction_patches',bone,patches[k]))
                else:
                    X_train[0] = X_train[0]+joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetReconstruction_patches',bone,patches[k]))
    	   
            if(patches[k].find('HR')!=-1):
                if Y_train[0]==[]:
                    Y_train[0] = joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetReconstruction_patches',bone,patches[k]))
                else:
                    Y_train[0] = Y_train[0]+joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetReconstruction_patches',bone,patches[k]))
            
            if(patches[k].find('corr')!=-1):
                corr = np.append(corr,joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetReconstruction_patches',bone,patches[k])))
			
Y_train = np.moveaxis(Y_train,0,1)
X_train = np.moveaxis(X_train,0,1)
print(np.shape(Y_train))
print(np.shape(corr))

#For the graphical representation of the DICE and the loss
number = (len(Y_train)/batch_size) - (len(Y_train)/batch_size)%500

#Load Data
trainset = torch.utils.data.TensorDataset(torch.Tensor(X_train),torch.Tensor(Y_train),torch.Tensor(corr))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True,pin_memory=use_cuda, num_workers=2)

#Create directores for the results 
if not os.path.exists(os.path.join(ResultsDirectory,'Images')):
    os.mkdir(os.path.join(ResultsDirectory, 'Images'))
    
TrainRed(trainloader,number)