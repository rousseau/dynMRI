#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:42:49 2020

@author: p20coupe
"""

import torch
from torch import nn
import torch.backends.cudnn
from torch.nn import functional as F

# Implementation from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

class BasicBlock(torch.nn.Module):
	expansion = 1


	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion*planes))

	def forward(self, x):
        	out = F.relu(self.bn1(self.conv1(x)))
        	out = self.bn2(self.conv2(out))
        	out += self.shortcut(x)
        	out = F.relu(out)
        	return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                       stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion *
                       planes, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.expansion*planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
    			self.shortcut = nn.Sequential(
        		nn.Conv2d(in_planes, self.expansion*planes,
                  	kernel_size=1, stride=stride, bias=False),
        		nn.BatchNorm2d(self.expansion*planes)
    			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class ResNet(torch.nn.Module):
	def __init__(self, block, num_blocks):
		super(ResNet, self).__init__()
		self.in_planes = 64


		self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)

		self.last_conv2 = torch.nn.Conv2d(256, 1, 1, padding=0)
		self.up_sampling = torch.nn.Upsample(scale_factor=4, mode='bilinear',align_corners=False)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn1(self.conv2(out)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)

		out = self.up_sampling(out)
		out = self.last_conv2(out)
		return out