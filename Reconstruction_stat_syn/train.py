from os.path import expanduser
from posix import listdir
from numpy import NaN, dtype

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
import matplotlib.pyplot as plt
import glob
import multiprocessing
import math

from pytorch_lightning.loggers import TensorBoardLogger, tensorboard
import argparse
from models.DRIT import DRIT
from models.DenseNet import Dense
from models.UNet_2D import UNet
from models.HighResNet import HighResNet
from models.DRITpp import DRIT as DRITpp
from models.CycleGAN import CycleGAN
from models.DRITpp_multiscale import DRIT as DRITpp_multi
from models.DRITpp_dynamic import DRIT as DRITpp_dynamic
from load_data import load_data


def normalize(tensor):
    for i in range(tensor.shape[1]):
        mean=tensor[0,i,:,:].mean()
        std=tensor[0,i,:,:].std()
        tensor[0,i,:,:]=(tensor[0,i,:,:]-mean)/std
    return(tensor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dynamic MRI Reconstruction')
    # dataset
    parser.add_argument('-d', '--data', help='Input dataset', type=str, required=False, default = 'hcp')
    parser.add_argument('-S', '--subjects', help='Number of subjects to use', type=int, required=False, default = 400)
    parser.add_argument('-v', '--version', help='Version of the dataset (equinus)', type=str, required=False, default = None)
    parser.add_argument('--segmentation', help='Use static segmentations as prior', type=str, required=False)

    # training
    parser.add_argument('-e', '--epochs', help='Max epochs', type=int, required=False, default = 10)
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 8)
    parser.add_argument('-g', '--gpu', help='Number of the GPU to use', type=int, required=False, default = 0)
    parser.add_argument('-l', '--loss', help='Loss to use', type=str, required=False, default = 'L2')
    parser.add_argument('-w', '--workers', help='Number of workers (multiprocessing)', type=int, required=False, default = 16)
    parser.add_argument('-m', '--model', help='Pytorch initialization model', type=str, required=False)

    # torchio
    parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = (128,128,1))
    parser.add_argument('-s', '--samples', help='Samples per volume', type=int, required=False, default = 8)
    parser.add_argument('-q', '--queue', help='Max queue length', type=int, required=False, default = 64)
    parser.add_argument('--sampler', help='Sampler to use (probabilities or uniform)', type=str, required=False, default = "Probabilities")
    parser.add_argument('--whole', help='set to True to use the whole image for training', type=bool,required=False)

    # network
    parser.add_argument('-n', '--network', help='Network to use (UNet, HighResNet, Disentangled, DenseNet..)', type=str, required=False, default = "DRIT")
    parser.add_argument('--use_reduce', help='for Disentangled_plusplus, set to True for using light architecture', type=bool,required=False, default=False)
    parser.add_argument('-c', '--channels', help='Number of channels in Unet', type=int, required=False, default=32)
    parser.add_argument('--dynamic', help='Use dynamic scheme', type=bool, required=False, default=False)

    # Multi scale
    parser.add_argument('--use_multiscale_discriminator', help='Set to True to use a multiscale discriminator', type=bool, required=False, default=False)
    parser.add_argument('--use_multiscale_content', help='Set to True to use a multiscale content', type= bool, required=False, default=False)
    parser.add_argument('--use_multiscale_style', help='Set to True to use multiscale style', type=bool, required=False, default=False)
    
    args = parser.parse_args()
    print("Nombre de GPUs detectes: "+ str(torch.cuda.device_count()))

    max_subjects = args.subjects
    training_split_ratio = 0.9  # use 90% of samples for training, 10% for testing
    num_epochs = args.epochs
    gpu = args.gpu
    num_workers = args.workers#multiprocessing.cpu_count()*
    network=args.network

    version=args.version
    sampler=args.sampler
    
    training_batch_size = args.batch_size
    validation_batch_size = args.batch_size
    loss=args.loss
    patch_size = args.patch_size
    if args.whole:
        pix=760
        marge=0.75
        c=256 #math.floor(pix*marge)
        patch_size = (c, c, 1)
    print(patch_size)
    samples_per_volume = args.samples
    max_queue_length = args.queue
    
    n_channels = args.channels
    data = args.data

    prefix = network+'_' #'unet_'#apartir moment tente sans t10T11d
    prefix += data
    prefix += '_epochs_'+str(num_epochs)
    prefix += '_patches_'+str(patch_size)
    prefix += '_sampling_'+str(samples_per_volume)
    prefix += '_nchannels_'+str(n_channels)
    prefix += '_loss_'+loss
    prefix += '_lr=10-4_'
    prefix += 'seg_'+str(args.segmentation)

    if args.model is not None:
        prefix += '_using_init'
        
    output_path = home+'/Nets_Reconstruction/Results/'
    subjects=[]

    #############################################################################################################################################################################""
    # DATASET
    if args.dynamic:
        data='dynamic_256'

    subjects, check_subjects = load_data(data=data, segmentation=args.segmentation)

    dataset = tio.SubjectsDataset(subjects)
    print('Dataset size:', len(check_subjects), 'subjects')
    prefix += '_subj_'+str(len(check_subjects))+'_images_'+str(len(subjects))

    #############################################################################################################################################################################""
    # TRANSFORMATIONS
    flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)
    bias = tio.RandomBiasField(coefficients = 0.5, p=0.5)
    noise = tio.RandomNoise(std=0.1, p=0.25)
    if data == 'equinus_simulate':
        prefix += '_bs_flp_afn_nse_VERSION_'+version
    else:
        pass

    if data=='hcp':
        normalization = tio.ZNormalization(masking_method='label')
        spatial = tio.RandomAffine(scales=0.1,degrees=10, translation=0, p=0.75)
    else:
        normalization = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
        spatial = tio.RandomAffine(scales=0.1,degrees=(20,0,0), translation=0, p=0.75)

    transforms = [flip, spatial, bias, normalization, noise]
    training_transform = tio.Compose(transforms)
    validation_transform = tio.Compose([normalization])   

    #############################################################################################################################################################################""
    # TRAIN AND VALIDATION SETS
    num_subjects = len(check_subjects)
    num_training_subjects = int(training_split_ratio * num_subjects)
    num_validation_subjects = num_subjects - num_training_subjects

    num_split_subjects = num_training_subjects, num_validation_subjects
    training_sub, validation_sub = torch.utils.data.random_split(check_subjects, num_split_subjects)#, generator=torch.Generator().manual_seed(42))
    training_subjects=[]
    validation_subjects=[]
    train=[]
    validation=[]
    for s in subjects:
        if s.subject_name in training_sub:
            training_subjects.append(s)
            if s.subject_name not in train:
                train.append(s.subject_name)
        elif s.subject_name in validation_sub:
            validation_subjects.append(s)
            if s.subject_name not in validation:
                validation.append(s.subject_name)
        else:
            sys.exit("Problème de construction ensembles de train et validation")
    print('training = '+str(train))
    print('validation = '+str(validation))


    training_set = tio.SubjectsDataset(
    training_subjects, transform=training_transform)

    validation_set = tio.SubjectsDataset(
    validation_subjects, transform=validation_transform)

    print('Training set:', len(training_sub), 'subjects')
    print('Validation set:', len(validation_sub), 'subjects') 
    prefix=prefix+'_validation_'+str(validation)


    #############################################################################################################################################################################""
    # PATCHES SETS
    print('num_workers : '+str(num_workers))
    prefix += '_sampler_'+sampler
    if sampler=='Probabilities':
        probabilities = {0: 0, 1: 1}
        sampler = tio.data.LabelSampler(
            patch_size=patch_size,
            label_name='label',
            label_probabilities=probabilities
        )
    elif sampler=='Uniform':
        sampler = tio.data.UniformSampler(patch_size)
    else:
        sys.exit('Select a correct sampler')

    patches_training_set = tio.Queue(
        subjects_dataset=training_set,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    patches_validation_set = tio.Queue(
        subjects_dataset=validation_set,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=False,
        shuffle_patches=False,
    )

    training_loader_patches = torch.utils.data.DataLoader(
        patches_training_set, batch_size=training_batch_size, num_workers=0, pin_memory=False)
    print("Nombre de patches de train: " + str(len(training_loader_patches.dataset)))
    validation_loader_patches = torch.utils.data.DataLoader(
        patches_validation_set, batch_size=validation_batch_size, num_workers=0, pin_memory=False)
    print("Nombre de patches de test: " + str(len(validation_loader_patches.dataset)))

    #############################################################################################################################################################################""
    # EXEC
    if loss=='L1':
        L=nn.L1Loss()
    elif loss=='L2':
        L=nn.MSELoss()
    else:
        sys.exit("Loss incorrecte")

    if network=="UNet":
        net = UNet(
            criterion=L,
            dataset=data,     
            learning_rate=1e-4,
            optimizer_class=torch.optim.Adam,
            #optimizer_class=torch.optim.Adam,
            n_features = n_channels,
            prefix = prefix,
            segmentation=args.segmentation,
        )

    elif network=="Disentangled":
        net = DRIT(
            criterion=L,
            dataset=data,     
            learning_rate=1e-4,
            optimizer_class=torch.optim.Adam,
            #optimizer_class=torch.optim.Adam,
            n_features = n_channels,
            prefix = prefix,
            segmentation=args.segmentation,
            mode='', 
            gpu=gpu,
        )
    elif network=="DenseNet":
        net = Dense(
            criterion=L,
            dataset=data,     
            learning_rate=1e-4,
            optimizer_class=torch.optim.Adam,
            #optimizer_class=torch.optim.Adam,
            n_features = n_channels,
            prefix = prefix,
            segmentation=args.segmentation,
        )
    elif network=="Disentangled_plusplus":
        if args.use_multiscale_discriminator==True or args.use_multiscale_content==True or args.use_multiscale_style==True:
            net = DRITpp_multi(
                criterion=L,
                dataset=data,     
                learning_rate=1e-4,
                optimizer_class=torch.optim.Adam,
                #optimizer_class=torch.optim.Adam,
                n_features = n_channels,
                prefix = prefix,
                segmentation=args.segmentation,
                mode='', 
                gpu=gpu,
                reduce=args.use_reduce,
                MS_discriminator=args.use_multiscale_discriminator,
                MS_content=args.use_multiscale_content, 
                MS_style=args.use_multiscale_style,
            )
        elif args.dynamic==True:
            net = DRITpp_dynamic(
                criterion=L,
                dataset=data,     
                learning_rate=1e-4,
                optimizer_class=torch.optim.Adam,
                #optimizer_class=torch.optim.Adam,
                n_features = n_channels,
                prefix = prefix,
                segmentation=args.segmentation,
                mode='', 
                gpu=gpu,
                reduce=args.use_reduce,
            )
        else:
            net = DRITpp(
                criterion=L,
                dataset=data,     
                learning_rate=1e-4,
                optimizer_class=torch.optim.Adam,
                #optimizer_class=torch.optim.Adam,
                n_features = n_channels,
                prefix = prefix,
                segmentation=args.segmentation,
                mode='', 
                gpu=gpu,
                reduce=args.use_reduce,
            )
    elif network=="CycleGAN":
        net = CycleGAN(
            criterion=L,
            dataset=data,     
            learning_rate=1e-4,
            optimizer_class=torch.optim.Adam,
            #optimizer_class=torch.optim.Adam,
            n_features = n_channels,
            prefix = prefix,
            segmentation=args.segmentation,
        )
    else:
        sys.exit('Enter a valid network name')
    checkpoint_callback = ModelCheckpoint(filepath=output_path+'Checkpoint_'+prefix+'_{epoch}-{val_loss:.2f}')#, save_top_k=1, monitor=)

    logger = TensorBoardLogger(save_dir = output_path, name = 'Test_logger',version=prefix)

    trainer = pl.Trainer(
        gpus=[gpu],
        max_epochs=num_epochs,
        progress_bar_refresh_rate=20,
        logger=logger,
        checkpoint_callback= checkpoint_callback,
        precision=16
    )
    trainer.fit(net, training_loader_patches, validation_loader_patches)
    trainer.save_checkpoint(output_path+prefix+'.ckpt')
    torch.save(net.state_dict(), output_path+prefix+'_torch.pt')

    print('Finished Training')

    ###############################################################################################################################################################
    #INFER
    print('Saving images')
    I=validation_set[0]
    T=training_set[0]
    if data=='hcp':
        T1=I['imageT1']
        T2=I['imageT2']
        label=I['label']
        name=I['subject_name']
    else:
        T1=I['LR_image']
        T2=I['HR_image']
        label=I['label']
        name=I['subject_name']
        if args.segmentation is not None:
            seg_os=I['segmentation_os']        

    T1.save(home+'/Nets_Reconstruction/T1_'+data+'_'+name+'_epochs-'+str(num_epochs)+'_nbrsubject-'+str(len(check_subjects))+'_sampling-'+str(samples_per_volume)+'_version-'+str(version)+'.nii.gz')
    T2.save(home+'/Nets_Reconstruction/T2_'+data+'_'+name+'_epochs-'+str(num_epochs)+'_nbrsubject-'+str(len(check_subjects))+'_sampling-'+str(samples_per_volume)+'_version-'+str(version)+'.nii.gz')
    label.save(home+'/Nets_Reconstruction/segmentation_'+data+'_'+name+'_epochs-'+str(num_epochs)+'_nbrsubject-'+str(len(check_subjects))+'_sampling-'+str(samples_per_volume)+'_version-'+str(version)+'.nii.gz')
    if args.segmentation is not None:
        seg_os.save(home+'/Nets_Reconstruction/segmentation-os_'+data+'_'+name+'_epochs-'+str(num_epochs)+'_nbrsubject-'+str(len(check_subjects))+'_sampling-'+str(samples_per_volume)+'_version-'+str(version)+'.nii.gz')
    
    out_test=net(I['LR_image'][tio.DATA]).cpu().detach()
    out_train=net(T['LR_image'][tio.DATA]).cpu().detach()
    out_test=tio.ScalarImage(tensor=out_test, affine=I['LR_image'].affine)
    out_train=tio.ScalarImage(tensor=out_train, affine=T['LR_image'].affine)
    out_test.save('/home/claire/Nets_Reconstruction/Test_'+data+'_'+name+'_epochs-'+str(num_epochs)+'_nbrsubject-'+str(len(check_subjects))+'_sampling-'+str(samples_per_volume)+'_version-'+str(version)+'.nii.gz')
    out_train.save('/home/claire/Nets_Reconstruction/Train_'+data+'_'+name+'_epochs-'+str(num_epochs)+'_nbrsubject-'+str(len(check_subjects))+'_sampling-'+str(samples_per_volume)+'_version-'+str(version)+'.nii.gz')

