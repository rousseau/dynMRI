#!/bin/bash
conda install pytorch==2.0.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install lightning==2.0.8 -c conda-forge
conda install -c conda-forge torchio==0.19.1
conda install -c conda-forge matplotlib
conda install -c conda-forge tensorboard==2.14.0
conda install numpy
