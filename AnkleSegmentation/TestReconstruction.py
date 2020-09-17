#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:35:20 2020

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


from ResNet_Reconstruction import *

parser = argparse.ArgumentParser()
parser.add_argument("TestData", help="PATH to testing data")
parser.add_argument("NetworkPATH",help="PATH to the network to use")
parser.add_argument("ResultsDirectory",help="PATH to the results storage directory")
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

Datadirectory = sys.argv[1]
ResultsDirectory = sys.argv[3]
NetworkPath = sys.argv[2]

def psnr(lP,lT):
    mse = np.mean( (lP - lT) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 3.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def imshow(img,corr):
	n_rows = 2
	n_cols = int(len(img)/2)
	plt.figure(figsize=(n_cols, n_rows))
	for i in range(n_rows):
		for j in range(n_cols):
			sub = plt.subplot(n_rows, n_cols, i*n_cols+1+j)
			sub.imshow(img[j+n_rows*i,0,:,:].numpy(),
		 	cmap=plt.cm.gray,
		 	interpolation="nearest",
		 	vmin=-3,vmax=2)
			sub.set_title('%.3f' %(corr[j+n_rows*i]))
			sub.axis('off')
	

def imshow_difMap(img,label):
	n_rows = 2
	n_cols = int(len(img)/2)
	plt.figure(figsize=(n_cols, n_rows))
	for i in range(n_rows):
		for j in range(n_cols):
			sub = plt.subplot(n_rows, n_cols, i*n_cols+1+j)
			sub.imshow(img[j+n_rows*i,0,:,:].numpy()-label[j+n_rows*i,0,:,:].numpy(),
		 	cmap=plt.cm.gray,
		 	interpolation="nearest",
		 	vmin=-3,vmax=2)
			sub.axis('off')
            
def ValidRed2D(testloader,path):
    psnr_value=[]
    
    net = ResNet(BasicBlock, [3,4,6]).to(device)
    net.load_state_dict(torch.load(path))
    
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, correlation = data
        inputs= inputs.to(device)
        labels = labels.to(device)
        correlation = correlation.to(device)
	
        outputs = net(inputs)

	
        psnr_val = 0
	
        pr= labels[0].cpu().detach().numpy()
        gt = outputs[0].cpu().detach().numpy()
        psnr_val = psnr_val + psnr(gt[0,:,:],pr[0,:,:])
	
	
        if i == 800:
            imshow(inputs.cpu().detach(),correlation.cpu().detach().numpy())
            plt.savefig(os.path.join(ResultsDirectory,'Images','inputs_myloss_test.png'),dpi=150)
            imshow(labels.cpu().detach(),correlation.cpu().detach().numpy())
            plt.savefig(os.path.join(ResultsDirectory,'Images','labels_myloss_test.png'),dpi=150)
            imshow(outputs.cpu().detach(),correlation.cpu().detach().numpy())
            plt.savefig(os.path.join(ResultsDirectory,'Images','outputs_myloss_test.png'),dpi=150)
            imshow_difMap(outputs.cpu().detach(),labels.cpu().detach())
            plt.savefig(os.path.join(ResultsDirectory,'Images','DifMap_myloss_test.png'),dpi=150)
	
            
        psnr_value.append(psnr_val)
            
    np.savetxt(os.path.join(ResultsDirectory,'PSNR_test.txt'),psnr_value)
    print('Standard Deviation:' + str(statistics.stdev(psnr_value)))
    print('Mean:' + str(statistics.mean(psnr_value)))
    

bones=['calcaneus','talus','tibia'] 
X_test=[[]]
Y_test=[[]]  
corr=[]
 
sujets = os.listdir(Datadirectory)
sujets = np.sort(sujets)


for i in range(len(sujets)):
	
    #Dataset Validation
    for bone in bones:
        patches= os.listdir(os.path.join(Datadirectory,sujets[i],'DatasetReconstruction_patches',bone))
        for k in range(len(patches)):
            if(patches[k].find('BR')!=-1):
                if X_test[0]==[]:
                    X_test[0] = joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetReconstruction_patches',bone,patches[k]))
                else:
                    X_test[0] = X_test[0]+joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetReconstruction_patches',bone,patches[k]))
    	   
            if(patches[k].find('HR')!=-1):
                if Y_test[0]==[]:
                    Y_test[0] = joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetReconstruction_patches',bone,patches[k]))
                else:
                    Y_test[0] = Y_test[0]+joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetReconstruction_patches',bone,patches[k]))
            
            if(patches[k].find('corr')!=-1):
                corr = np.append(corr,joblib.load(os.path.join(Datadirectory,sujets[i],'DatasetReconstruction_patches',bone,patches[k])))		
	

Y_test = np.moveaxis(Y_test,0,1)
X_test = np.moveaxis(X_test,0,1)
print(np.shape(Y_test))
print(np.shape(corr))


testset = torch.utils.data.TensorDataset(torch.Tensor(X_test),torch.Tensor(Y_test),torch.Tensor(corr))
testloader = torch.utils.data.DataLoader(testset, batch_size=8,
		                                  shuffle=False, pin_memory=use_cuda, num_workers=2)

#Create directores for the results 
if not os.path.exists(os.path.join(ResultsDirectory,'Images')):
    os.mkdir(os.path.join(ResultsDirectory, 'Images'))

ValidRed2D(testloader,NetworkPath)