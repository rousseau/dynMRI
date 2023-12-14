#!/bin/bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install lightning -c conda-forge
conda install -c conda-forge torchio
conda install -c conda-forge matplotlib
conda install -c conda-forge tensorboard
conda install numpy
