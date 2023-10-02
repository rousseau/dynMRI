from ast import arg
from os.path import expanduser
import sys
from numpy import mat
from tqdm import tqdm
import os
import numpy as np
import glob

from torchio.data.image import ScalarImage
home = expanduser("~")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import math
from skimage.metrics import structural_similarity as ssim

import argparse
from options.test_options import TestOptions

from DRITPP_2 import DRIT 
from Degradation_nets import Degradation_paired, Degradation_unpaired


def compute_PSNR(max_val,prediction: torch.Tensor,ground_truth: torch.Tensor):
    prediction=prediction.float()
    ground_truth=ground_truth.float()
    mse=torch.nn.functional.mse_loss(prediction,ground_truth)
    if mse==0:
        return(100)
    else:
        return(20 * math.log10(max_val) - 10 * torch.log10(mse))

def compute_PSNR_onMask(max_val,prediction: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor):
    prediction=prediction.float()
    ground_truth=ground_truth.float()
    mask=mask.float()
    MSE=((prediction-ground_truth)**2)
    MSE=MSE*mask
    Nombre_pixels_masque=mask.sum()
    MSE=MSE/(Nombre_pixels_masque)
    MSE=MSE.sum()
    return(20 * math.log10(max_val) - 10 * torch.log10(MSE))

def compute_PSNR_outOfMask(max_val,prediction: torch.Tensor, ground_truth: torch.Tensor, mask: torch.Tensor):
    prediction=prediction.float()
    ground_truth=ground_truth.float()
    mask=mask.float()
    mask=torch.abs((mask-1)*(-1))
    MSE=((prediction-ground_truth)**2)
    MSE=MSE*mask
    Nombre_pixels_masque=mask.sum()
    MSE=MSE/(Nombre_pixels_masque)
    MSE=MSE.sum()
    return(20 * math.log10(max_val) - 10 * torch.log10(MSE))

def round_to_n(x,n):
    return round(x, -int(math.floor(math.log10(abs(x-math.floor(x)))))+(n-1))


def run_test(args, isTrain=False):
    compteur=0
    if args.gpu is not None:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:"+str(args.gpu) if use_cuda else "cpu")
    else:
        device = torch.device("cpu")

    gpu = args.gpu
    network = args.network
    print('')
    if args.subcommand == "DRIT":
        if (args.method != 'ddpm' and args.model is None):
            net = DRIT(
                prefix = '',
                opt = args,
                isTrain=isTrain
            )
            print('RÉSEAU UTILISÉ: '+args.method)
        elif (args.method != 'ddpm' and args.model is not None):
            net = DRIT(
                prefix = '',
                opt=args,
                isTrain=isTrain
            )
            print('RÉSEAU UTILISÉ: '+args.method+" AVEC INITIALISATION")
        print('')
    elif args.subcommand == 'Degradation':
        if args.data_mode == 'Paired':
            net = Degradation_paired(
                prefix='',
                opt = args,
                isTrain=isTrain
            )
        elif args.data_mode == 'Unpaired':
            net = Degradation_unpaired(
                prefix='',
                opt = args,
                isTrain=isTrain
            )
    model = glob.glob(os.path.join(args.model, '*.pt'))[0]
    if model.split('/')[-1].split('.')[-1]=='pt':
        net.load_state_dict(torch.load(model))
    elif model.split('/')[-1].split('.')[-1]=='ckpt':
        net.load_state_dict(torch.load(model)['state_dict'])
    else:
        sys.exit('Entrez un ckeckpoint valide')
    net.eval()
    if args.gpu is not None:
        net.to(device=device)

    #############################################################################################################################################################################""
    I=args.input
    T2=args.ground_truth
    print("Entrée: "+I)
    print("GT: "+T2)
    subject = tio.Subject(
        T1_image=tio.ScalarImage(I),
        T2_image=tio.ScalarImage(T2),
        )
    if I.find('ses')!=-1:
        print('DONNÉE DHCP')
        normalization = tio.ZNormalization(masking_method='label')
    elif (I.find('MovieClear')!=-1 or I.find('static_3DT1')!=-1):
        print('DONNÉE EQUINUS')
        normalization = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    else:
        sys.exit('DONNÉE INCONNUE')
    #############################################################################################################################################################################
    print('Inference')
    step = 0
    batch_size = args.batch_size

    if (I.find('Equinus_BIDS_dataset')==-1 and I.find('Test_recalage')==-1):
        print('SUJET DÉJÀ NORMALISÉ')
        sub = subject
    else:
        print('NORMALISATION')
        augment = normalization
        sub = augment(subject)

    if model.find('(64, 64, 1)')!=-1:
        patch_size=(64, 64, 1)
        patch_overlap=(60, 60, 0)
    elif model.find('(1, 64, 64)')!=-1:
        patch_size=(1, 64, 64)
        patch_overlap=(0, 32, 32)
    else:
        patch_size=(64, 64, 1)
        #patch_overlap=(60, 60, 0)
        patch_overlap=(0, 0, 0)

    if args.subcommand == "DRIT":
        if args.latents == True:
            patch_overlap=(32, 32, 0)
    
    print('Patch size: '+str(patch_size))
    print('Patch overlap: '+str(patch_overlap))

    grid_sampler = tio.inference.GridSampler(
        sub,
        patch_size,
        patch_overlap
        )
    compteur=0
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
    aggregator = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
    if args.use_segmentation_network == True:
        aggregator_LR=tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
        aggregator_HR=tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

    if model.find('(1, 64, 64)')!=-1:
        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):
                input_tensor = patches_batch['T1_image'][tio.DATA]
                locations = patches_batch[tio.LOCATION]
                outputs = net(input_tensor)
                aggregator.add_batch(outputs, locations)
    else:
        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):
                input_tensor = patches_batch['T1_image'][tio.DATA]
                if args.seg is not None:
                    input_tensor=torch.cat((input_tensor,patches_batch['seg_os'][tio.DATA]),1)
                else:
                    pass
                if gpu is not None:
                    input_tensor=input_tensor.to(device)
                input_tensor=input_tensor.squeeze(-1)
                locations = patches_batch[tio.LOCATION]
                
                if gpu is not None:
                    if network=="Disentangled_plusplus":
                        ground_tensor = patches_batch['T2_image'][tio.DATA]
                        ground_tensor=ground_tensor.to(device)
                        ground_tensor=ground_tensor.squeeze(-1)
                        if args.use_segmentation_network == True:
                            outputs, segmentation_HR_, segmentation_LR_ = net(input_tensor, ground_tensor)
                            outputs=outputs.cpu().detach()
                            segmentation_HR=segmentation_HR_.cpu().detach()
                            segmentation_LR=segmentation_LR_.cpu().detach()
                            segmentation_HR_finale=torch.zeros(segmentation_HR.shape[0], 1, segmentation_HR.shape[2], segmentation_HR.shape[3])
                            segmentation_LR_finale=torch.zeros(segmentation_HR.shape[0], 1, segmentation_HR.shape[2], segmentation_HR.shape[3])
                            for slice in range(segmentation_HR.shape[1]):
                                segmentation_HR_finale += torch.tensor(slice) * segmentation_HR[:, slice, :, :].unsqueeze(1)
                                segmentation_LR_finale += torch.tensor(slice) * segmentation_LR[:, slice, :, :].unsqueeze(1)

                            if compteur == 70: 
                                plt.figure()
                                plt.suptitle(str(locations[0]))
                                plt.subplot(1,2,1)
                                plt.imshow(input_tensor[0,0,:,:].cpu().detach().numpy(), cmap="gray")
                                plt.colorbar()
                                plt.subplot(1,2,2)
                                plt.imshow(outputs[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")                            
                                plt.colorbar()
                                plt.savefig('/home/claire/Nets_Reconstruction/Inference.png')
                                plt.close()
                            compteur = compteur +1 

                        else:
                            if args.subcommand == "DRIT":
                                if args.latents == True: 
                                    outputs, fake_LR, content_HR, content_LR, z_HR, z_LR, z_fake_HR, z_fake_LR, content_fake_HR, content_fake_LR = net.get_latent(input_tensor, ground_tensor)
                                    if step == 0:
                                        a_LR=content_LR.cpu().detach().numpy().reshape(1, -1)
                                        a_HR=content_HR.cpu().detach().numpy().reshape(1, -1)
                                        m_LR=z_LR.cpu().detach().numpy().reshape(1, -1)
                                        m_HR=z_HR.cpu().detach().numpy().reshape(1, -1)
                                        a_fakeHR=content_fake_HR.cpu().detach().numpy().reshape(1, -1)
                                        a_fakeLR=content_fake_LR.cpu().detach().numpy().reshape(1, -1)
                                        m_fakeLR=z_fake_LR.cpu().detach().numpy().reshape(1, -1)
                                        m_fakeHR=z_fake_HR.cpu().detach().numpy().reshape(1, -1)
                                    else:
                                        a_LR=np.concatenate((a_LR,content_LR.cpu().detach().numpy().reshape(1, -1)), axis=0)
                                        a_HR=np.concatenate((a_HR,content_HR.cpu().detach().numpy().reshape(1, -1)), axis=0)
                                        m_LR=np.concatenate((m_LR,z_LR.cpu().detach().numpy().reshape(1, -1)), axis=0)
                                        m_HR=np.concatenate((m_HR, z_HR.cpu().detach().numpy().reshape(1, -1)), axis=0)
                                        a_fakeHR=np.concatenate((a_fakeHR,content_fake_HR.cpu().detach().numpy().reshape(1, -1)), axis=0)
                                        a_fakeLR=np.concatenate((a_fakeLR,content_fake_LR.cpu().detach().numpy().reshape(1, -1)), axis=0)
                                        m_fakeLR=np.concatenate((m_fakeLR,z_fake_LR.cpu().detach().numpy().reshape(1, -1)), axis=0)
                                        m_fakeHR=np.concatenate((m_fakeHR,z_fake_HR.cpu().detach().numpy().reshape(1, -1)), axis=0)
                                    step+=1
                                else:
                                    outputs = net(input_tensor, ground_tensor).cpu().detach()
                            else:
                                outputs = net(input_tensor, ground_tensor).cpu().detach()
                    else:
                        outputs = net(input_tensor).cpu().detach()
                else:
                    if network=="Disentangled_plusplus":
                        ground_tensor = patches_batch['T2_image'][tio.DATA]
                        ground_tensor=ground_tensor.squeeze(-1)
                        outputs = net(input_tensor, ground_tensor)
                    else:
                        outputs = net(input_tensor)
                outputs=outputs.unsqueeze(-1)
                aggregator.add_batch(outputs, locations)
                if args.use_segmentation_network == True:
                    segmentation_LR_finale = segmentation_LR_finale.unsqueeze(-1)
                    segmentation_HR_finale = segmentation_HR_finale.unsqueeze(-1)
                    aggregator_LR.add_batch(segmentation_LR_finale, locations)
                    aggregator_HR.add_batch(segmentation_HR_finale, locations)

    output_tensor = aggregator.get_output_tensor()
    if args.use_segmentation_network == True:
        segmentation_LR_tensor = aggregator_LR.get_output_tensor()
        segmentation_HR_tensor = aggregator_HR.get_output_tensor()

    tmp = tio.ScalarImage(tensor=output_tensor, affine=sub.T1_image.affine)
    sub.add_image(tmp, 'T2_image_estim')
    ##########################################################################################################################################""""
    if args.latents == True:
        print('Get latents')
        return(a_HR, a_LR, m_HR, m_LR, a_fakeHR, a_fakeLR, m_fakeHR, m_fakeLR)
    else:
        print('Saving images')
        if args.mode == 'reconstruction':
            output_seg = tio.ScalarImage(tensor=output_tensor.to(torch.float), affine=subject['T1_image'].affine)
            output_seg.save(os.path.join(args.output, 'Disentangled_reconstruction_'+args.ground_truth.split('/')[-1])) #on met .ground truth parce qu'on sait sur quelle image dyn c'est recalé par le nom, mais on sait aussi quel os comparé a .input
        elif args.mode == 'degradation':
            output_seg = tio.ScalarImage(tensor=output_tensor.to(torch.float), affine=subject['T2_image'].affine)
            output_seg.save(os.path.join(args.output, 'Disentangled_degradation_'+args.ground_truth.split('/')[-1]))
        elif args.mode == 'both':
            output_seg = tio.ScalarImage(tensor=output_tensor.to(torch.float), affine=subject['T1_image'].affine)
            output_seg.save(os.path.join(args.output, 'Disentangled_reconstruction_'+args.ground_truth.split('/')[-1]))
            output_segLR = tio.ScalarImage(tensor=output_tensor_LR.to(torch.float), affine=subject['T2_image'].affine)
            output_segLR.save(os.path.join(args.output, 'Disentangled_degradation_'+args.ground_truth.split('/')[-1]))









def run_test_latent(dynamic_path, static_path, model, recording_path=None, segmentation_path=None, network='Disentangled_plusplus', mode='reconstruction', gpu = 0, whole = False, features = 64, segmentation=None, use_reduce=True, use_multiscale_discriminator=False, use_multiscale_content=False, use_multiscale_style=False, seg=None,dynamic=False, method=None, use_segmentation_network=False):
    pass




if __name__ == '__main__':
    testparser = TestOptions()
    args = testparser.parse()
    network=args.network
    gpu=args.gpu
    if gpu is not None:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:"+str(gpu) if use_cuda else "cpu")
    
    print('')
    if args.subcommand == "DRIT":
        if (args.method != 'ddpm' and args.model is None):
            net = DRIT(
                prefix = '',
                opt = args,
                isTrain=False
            )
            print('RÉSEAU UTILISÉ: '+args.method)
        elif (args.method != 'ddpm' and args.model is not None):
            net = DRIT(
                prefix = '',
                opt=args,
                isTrain=False
            )
            print('RÉSEAU UTILISÉ: '+args.method+" AVEC INITIALISATION")
        print('')
    elif args.subcommand == 'Degradation':
        if args.data_mode == 'Paired':
            net = Degradation_paired(
                prefix='',
                opt = args,
                isTrain=False
            )
        elif args.data_mode == 'Unpaired':
            net = Degradation_unpaired(
                prefix='',
                opt = args,
                isTrain=False
            )
    model = glob.glob(os.path.join(args.model, '*.pt'))[0]
    if model.split('/')[-1].split('.')[-1]=='pt':
        net.load_state_dict(torch.load(model))
    elif model.split('/')[-1].split('.')[-1]=='ckpt':
        net.load_state_dict(torch.load(model)['state_dict'])
    else:
        sys.exit('Entrez un ckeckpoint valide')
    net.eval()
    if gpu is not None:
        net.to(device=device)

    #############################################################################################################################################################################""
    
    I=args.input
    p='/home/claire/Nets_Reconstruction/'
    SS=I.split('/')[:-1]
    s=I.split('/')[-1]
    ss=s.split('_')
    ss[0]='segmentation'
    ss=('_').join(ss)
    SS.append(ss)
    SS=('/').join(SS)
    if os.path.exists(SS):
        S=SS
        s=S.split('/')[-1]
        s=p+s
    else:
        S=None
    T2=args.ground_truth
    print(I)
    print(T2)
    
    t1=I.split('/')[-1]
    t1=p+t1
    
    if args.ground_truth is not None:
        t2=T2.split('/')[-1]
        t2=p+t2

    if (args.seg is None) and (S is None):
        subject = tio.Subject(
            T1_image=tio.ScalarImage(I),
            T2_image=tio.ScalarImage(T2),
            )
    elif (args.seg is None) and (S is not None):
        subject = tio.Subject(
            T1_image=tio.ScalarImage(I),
            T2_image=tio.ScalarImage(T2),
            label=tio.LabelMap(S),
            )
    elif (args.seg is not None) and (S is None):
        seg_os=args.segmentation_os
        subject = tio.Subject(
            T1_image=tio.ScalarImage(I),
            T2_image=tio.ScalarImage(T2),
            seg_os=tio.LabelMap(seg_os),
            )
    else:
        seg_os=args.segmentation_os
        subject = tio.Subject(
            T1_image=tio.ScalarImage(I),
            T2_image=tio.ScalarImage(T2),
            label=tio.LabelMap(S),
            seg_os=tio.LabelMap(seg_os)
            )

    print(subject['T1_image'][tio.DATA].shape)
    print(subject['T2_image'][tio.DATA].shape)
    if I.find('ses')!=-1:
        print('DONNÉE DHCP')
        normalization = tio.ZNormalization(masking_method='label')
    elif (I.find('MovieClear')!=-1 or I.find('static_3DT1')!=-1):
        print('DONNÉE EQUINUS')
        normalization = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    else:
        sys.exit('DONNÉE INCONNUE')

    #############################################################################################################################################################################""
    print('Inference')
    step = 0
    batch_size = args.batch_size

    if (I.find('Equinus_BIDS_dataset')==-1 and I.find('Test_recalage')==-1):
        print('SUJET DÉJÀ NORMALISÉ')
        sub = subject
    else:
        print('NORMALISATION')
        augment = normalization
        sub = augment(subject)

    if model.find('(64, 64, 1)')!=-1:
        patch_size=(64, 64, 1)
        patch_overlap=(60, 60, 0)
    elif model.find('(1, 64, 64)')!=-1:
        patch_size=(1, 64, 64)
        patch_overlap=(0, 32, 32)
    else:
        patch_size=(64, 64, 1)
        #patch_overlap=(60, 60, 0)
        patch_overlap=(0, 0, 0)

    print('Patch size: '+str(patch_size))
    print('Patch overlap: '+str(patch_overlap))

    grid_sampler = tio.inference.GridSampler(
        sub,
        patch_size,
        patch_overlap
        )
    compteur=0
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
    aggregator = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
    if args.use_segmentation_network == True:
        aggregator_LR=tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')
        aggregator_HR=tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

    if model.find('(1, 64, 64)')!=-1:
        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):
                input_tensor = patches_batch['T1_image'][tio.DATA]
                locations = patches_batch[tio.LOCATION]
                outputs = net(input_tensor)
                aggregator.add_batch(outputs, locations)
    else:
        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):
                input_tensor = patches_batch['T1_image'][tio.DATA]
                if args.seg is not None:
                    input_tensor=torch.cat((input_tensor,patches_batch['seg_os'][tio.DATA]),1)
                else:
                    pass
                if gpu is not None:
                    input_tensor=input_tensor.to(device)
                input_tensor=input_tensor.squeeze(-1)
                locations = patches_batch[tio.LOCATION]
                
                if gpu is not None:
                    if network=="Disentangled_plusplus":
                        ground_tensor = patches_batch['T2_image'][tio.DATA]
                        ground_tensor=ground_tensor.to(device)
                        ground_tensor=ground_tensor.squeeze(-1)
                        if args.use_segmentation_network == True:
                            outputs, segmentation_HR_, segmentation_LR_ = net(input_tensor, ground_tensor)
                            outputs=outputs.cpu().detach()
                            segmentation_HR=segmentation_HR_.cpu().detach()
                            segmentation_LR=segmentation_LR_.cpu().detach()
                            segmentation_HR_finale=torch.zeros(segmentation_HR.shape[0], 1, segmentation_HR.shape[2], segmentation_HR.shape[3])
                            segmentation_LR_finale=torch.zeros(segmentation_HR.shape[0], 1, segmentation_HR.shape[2], segmentation_HR.shape[3])
                            for slice in range(segmentation_HR.shape[1]):
                                segmentation_HR_finale += torch.tensor(slice) * segmentation_HR[:, slice, :, :].unsqueeze(1)
                                segmentation_LR_finale += torch.tensor(slice) * segmentation_LR[:, slice, :, :].unsqueeze(1)

                            if compteur == 70: 
                                plt.figure()
                                plt.suptitle(str(locations[0]))
                                plt.subplot(1,2,1)
                                plt.imshow(input_tensor[0,0,:,:].cpu().detach().numpy(), cmap="gray")
                                plt.colorbar()
                                plt.subplot(1,2,2)
                                plt.imshow(outputs[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")                            
                                plt.colorbar()
                                plt.savefig('/home/claire/Nets_Reconstruction/Inference.png')
                                plt.close()
                            compteur = compteur +1 

                        else:
                            if args.subcommand == "DRIT":
                                if args.latents == True: 
                                    outputs, fake_LR, content_HR, content_LR, z_HR, z_LR, z_fake_HR, z_fake_LR, content_fake_HR, content_fake_LR = net.get_latent(input_tensor, ground_tensor)
                                    if step == 0:
                                        a_LR=content_LR.cpu().detach().numpy().reshape(1, -1)
                                        a_HR=content_HR.cpu().detach().numpy().reshape(1, -1)
                                        m_LR=z_LR.cpu().detach().numpy().reshape(1, -1)
                                        m_HR=z_HR.cpu().detach().numpy().reshape(1, -1)
                                        a_fakeHR=content_fake_HR.cpu().detach().numpy().reshape(1, -1)
                                        a_fakeLR=content_fake_LR.cpu().detach().numpy().reshape(1, -1)
                                        m_fakeLR=z_fake_LR.cpu().detach().numpy().reshape(1, -1)
                                        m_fakeHR=z_fake_HR.cpu().detach().numpy().reshape(1, -1)
                                    else:
                                        a_LR=np.concatenate((a_LR,content_LR.cpu().detach().numpy().reshape(1, -1)), axis=0)
                                        a_HR=np.concatenate((a_HR,content_HR.cpu().detach().numpy().reshape(1, -1)), axis=0)
                                        m_LR=np.concatenate((m_LR,z_LR.cpu().detach().numpy().reshape(1, -1)), axis=0)
                                        m_HR=np.concatenate((m_HR, z_HR.cpu().detach().numpy().reshape(1, -1)), axis=0)
                                        a_fakeHR=np.concatenate((a_fakeHR,content_fake_HR.cpu().detach().numpy().reshape(1, -1)), axis=0)
                                        a_fakeLR=np.concatenate((a_fakeLR,content_fake_LR.cpu().detach().numpy().reshape(1, -1)), axis=0)
                                        m_fakeLR=np.concatenate((m_fakeLR,z_fake_LR.cpu().detach().numpy().reshape(1, -1)), axis=0)
                                        m_fakeHR=np.concatenate((m_fakeHR,z_fake_HR.cpu().detach().numpy().reshape(1, -1)), axis=0)
                                    step+=1
                                else:
                                    outputs = net(input_tensor, ground_tensor).cpu().detach()
                            else:
                                outputs = net(input_tensor, ground_tensor).cpu().detach()
                    else:
                        outputs = net(input_tensor).cpu().detach()
                else:
                    if network=="Disentangled_plusplus":
                        ground_tensor = patches_batch['T2_image'][tio.DATA]
                        ground_tensor=ground_tensor.squeeze(-1)
                        outputs = net(input_tensor, ground_tensor)
                    else:
                        outputs = net(input_tensor)
                outputs=outputs.unsqueeze(-1)
                aggregator.add_batch(outputs, locations)
                if args.use_segmentation_network == True:
                    segmentation_LR_finale = segmentation_LR_finale.unsqueeze(-1)
                    segmentation_HR_finale = segmentation_HR_finale.unsqueeze(-1)
                    aggregator_LR.add_batch(segmentation_LR_finale, locations)
                    aggregator_HR.add_batch(segmentation_HR_finale, locations)

    output_tensor = aggregator.get_output_tensor()
    if args.use_segmentation_network == True:
        segmentation_LR_tensor = aggregator_LR.get_output_tensor()
        segmentation_HR_tensor = aggregator_HR.get_output_tensor()

    tmp = tio.ScalarImage(tensor=output_tensor, affine=sub.T1_image.affine)
    sub.add_image(tmp, 'T2_image_estim')

    #############################################################################################################################################################################""
    print('Saving images')
    #if args.test_image is None:
    output_seg = tio.ScalarImage(tensor=output_tensor.to(torch.float), affine=subject['T1_image'].affine)
    output_seg.save(args.output)
