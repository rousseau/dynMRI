from os.path import expanduser
import sys
from numpy import mat
from tqdm import tqdm
import os

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

from models.DRIT import DRIT
from models.DenseNet import Dense, DenseNet
from models.UNet_2D import UNet
from models.HighResNet import HighResNet
from models.CycleGAN import CycleGAN
from models.DRITpp import DRIT as DRITpp
from models.DRITpp_multiscale import DRIT as DRITpp_multi


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo TorchIO inference')
    parser.add_argument('-i', '--input', help='Input image', type=str, required=True)
    parser.add_argument('-m', '--model', help='Pytorch model', type=str, required=True)
    parser.add_argument('-o', '--output', help='Output image', type=str, required=True)
    parser.add_argument('-F', '--fuzzy', help='Output fuzzy image', type=str, required=False)
    parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = (64,64,1))
    parser.add_argument('--patch_overlap', help='Patch overlap', type=int, required=False, default = (16,16,0))
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 2)
    parser.add_argument('-g', '--ground_truth', help='Ground truth for metric computation', type=str, required=False)
    parser.add_argument('-T', '--test_time', help='Number of inferences for test-time augmentation', type=int, required=False, default=1)
    parser.add_argument('-c', '--channels', help='Number of channels', type=int, required=False, default=16)
    parser.add_argument('-f', '--features', help='Number of features', type=int, required=False, default=64)
    parser.add_argument('--classes', help='Number of classes', type=int, required=False, default=1)
    parser.add_argument('-s', '--scales', help='Scaling factor (test-time augmentation)', type=float, required=False, default=0.05)
    parser.add_argument('-d', '--degrees', help='Rotation degrees (test-time augmentation)', type=int, required=False, default=10)
    parser.add_argument('-D', '--dataset', help='Name of the dataset used', type=str, required=False, default='hcp')
    parser.add_argument('-t', '--test_image', help='Image test (skip inference and goes directly to PSNR)', type=str, required=False)
    parser.add_argument('-S', '--segmentation', help='Segmentation to use', type=str,required=False)
    parser.add_argument('-n', '--network', help='Network to use (UNet, ResNet, disentangled..)', type=str,required=True)
    parser.add_argument('--gpu', help='GPU to use (If None, goes on CPU)', type=int,required=False)
    parser.add_argument('--diff', help='Save or not the difference between the ground truth and input or recostruct image', type=bool,required=False)
    parser.add_argument('--seg', help='Use static segmentations as prior', type=str, required=False)
    parser.add_argument('--segmentation_os', help='Segmentation to use', type=str,required=False)
    parser.add_argument('--mode', help='Mode to use (for DRIT or CycleGAN): reconstruction or degradation', type=str,required=False)
    parser.add_argument('--whole', help='set to True for using the whole image for testing', type=bool,required=False)
    parser.add_argument('--use_reduce', help='for Disentangled_plusplus, set to True for using light architecture', type=bool,required=False, default=False)
    parser.add_argument('--use_multiscale_discriminator', help='Set to True to use a multiscale discriminator', type=bool, required=False, default=False)
    parser.add_argument('--use_multiscale_content', help='Set to True to use a multiscale content', type= bool, required=False, default=False)
    parser.add_argument('--use_multiscale_style', help='Set to True to use multiscale style', type=bool, required=False, default=False)




    args = parser.parse_args()
    compteur=0
    dataset=args.dataset
    model=args.model
    network=args.network
    gpu=args.gpu
    if gpu is not None:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:"+str(gpu) if use_cuda else "cpu")


    #############################################################################################################################################################################""
    if network=='Disentangled':
        Net= DRIT(
            criterion=nn.MSELoss(),
            dataset='equinus',     
            learning_rate=1e-4,
            optimizer_class=torch.optim.Adam,
            #optimizer_class=torch.optim.Adam,
            n_features = args.features,
            prefix = '',
            segmentation=args.seg,
            gpu=gpu,
            mode=args.mode,
        )
    elif network=='UNet':
        Net= UNet(
            criterion=nn.MSELoss(),
            dataset='equinus',     
            learning_rate=1e-4,
            optimizer_class=torch.optim.Adam,
            #optimizer_class=torch.optim.Adam,
            n_features = args.features,
            prefix = '',
            segmentation=args.seg,
        )
    elif network=='DenseNet':
        Net= Dense(
            criterion=nn.MSELoss(),
            dataset='equinus',     
            learning_rate=1e-4,
            optimizer_class=torch.optim.Adam,
            #optimizer_class=torch.optim.Adam,
            n_features = args.features,
            prefix = '',
            segmentation=args.seg,
        )
    elif network=='HighResNet':
        Net= HighResNet(
            criterion=nn.MSELoss(),
            dataset='equinus',     
            learning_rate=1e-4,
            optimizer_class=torch.optim.Adam,
            #optimizer_class=torch.optim.Adam,
            n_features = args.features,
            prefix = '',
            segmentation=args.seg,
        )
    elif network=="Disentangled_plusplus":
        if args.use_multiscale_discriminator==True or args.use_multiscale_content==True or args.use_multiscale_style==True:
            Net = DRITpp_multi(
                criterion=nn.MSELoss(),
                dataset='equinus',     
                learning_rate=1e-4,
                optimizer_class=torch.optim.Adam,
                #optimizer_class=torch.optim.Adam,
                n_features = args.features,
                prefix = '',
                segmentation=args.segmentation,
                mode=args.mode, 
                gpu=gpu,
                reduce=args.use_reduce,
                MS_discriminator=args.use_multiscale_discriminator,
                MS_content=args.use_multiscale_content, 
                MS_style=args.use_multiscale_style,
            )
        else:
            Net = DRITpp(
                criterion=nn.MSELoss(),
                dataset='equinus',     
                learning_rate=1e-4,
                optimizer_class=torch.optim.Adam,
                #optimizer_class=torch.optim.Adam,
                n_features = args.features,
                prefix = '',
                segmentation=args.segmentation,
                mode=args.mode, 
                gpu=gpu,
                reduce=args.use_reduce,
            )

    elif network=='CycleGAN':
        Net= CycleGAN(
            criterion=nn.MSELoss(),
            dataset='equinus',     
            learning_rate=1e-4,
            optimizer_class=torch.optim.Adam,
            #optimizer_class=torch.optim.Adam,
            n_features = args.features,
            prefix = '',
            segmentation=args.seg,
        )
    else:
        sys.exit('Enter a valid natwork name')
    
    if model.split('/')[-1].split('.')[-1]=='pt':
        Net.load_state_dict(torch.load(model))
    elif model.split('/')[-1].split('.')[-1]=='ckpt':
        Net.load_state_dict(torch.load(model)['state_dict'])
    else:
        sys.exit('Entrez un ckeckpoint valide')
    Net.eval()
    if gpu is not None:
        Net.to(device=device)

    #############################################################################################################################################################################""
    
    I=args.input
    p='/home/claire/Nets_Reconstruction/'
    if args.segmentation==None:
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

    else:
        S=args.segmentation
        s=p+args.segmentation.split('/')[-1]
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
    normalization = tio.ZNormalization(masking_method=tio.ZNormalization.mean)

    #############################################################################################################################################################################""
    if args.test_image is None:
        print('Inference')
        batch_size = args.batch_size

        if (I.find('Equinus_BIDS_dataset')==-1 and I.find('Test_recalage')==-1):
            print('SUJET DÉJÀ NORMALISÉ')
            sub = subject
        else:
            print('NORMALISATION')
            augment = normalization
            sub = augment(subject)

        if args.whole:
            a=subject['T1_image'][tio.DATA].shape[1]
            b=subject['T1_image'][tio.DATA].shape[2]
            patch_size=(a,b,1)
            patch_overlap=(0,0,0)
        else:
            if model.find('(64, 64, 1)')!=-1:
                patch_size=(64, 64, 1)
                patch_overlap=(60, 60, 0)
            elif model.find('(1, 64, 64)')!=-1:
                patch_size=(1, 64, 64)
                patch_overlap=(0, 32, 32)
            else:
                patch_size=(128, 128, 1)
                # patch_overlap=(126, 126, 0)
                patch_overlap=(0, 0, 0)

        print('Patch size: '+str(patch_size))
        print('Patch overlap: '+str(patch_overlap))

        grid_sampler = tio.inference.GridSampler(
            sub,
            patch_size,
            patch_overlap
            )

        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
        aggregator = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

        if model.find('(1, 64, 64)')!=-1:
            with torch.no_grad():
                for patches_batch in tqdm(patch_loader):
                    input_tensor = patches_batch['T1_image'][tio.DATA]
                    locations = patches_batch[tio.LOCATION]
                    outputs = Net(input_tensor)
                    aggregator.add_batch(outputs, locations)
        #if model.find('(64, 64, 1)')!=-1:
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
                            outputs = Net(input_tensor, ground_tensor).cpu().detach()
                            if compteur==160:
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
                            compteur=compteur+1
                        else:
                            outputs = Net(input_tensor).cpu().detach()
                    else:
                        if network=="Disentangled_plusplus":
                            ground_tensor = patches_batch['T2_image'][tio.DATA]
                            ground_tensor=ground_tensor.squeeze(-1)
                            outputs = Net(input_tensor, ground_tensor)
                        else:
                            outputs = Net(input_tensor)
                    outputs=outputs.unsqueeze(-1)
                    aggregator.add_batch(outputs, locations)

        output_tensor = aggregator.get_output_tensor()

        tmp = tio.ScalarImage(tensor=output_tensor, affine=sub.T1_image.affine)
        sub.add_image(tmp, 'T2_image_estim')

    #############################################################################################################################################################################""
    print('Saving images')
    if args.test_image is None:
        output_seg = tio.ScalarImage(tensor=output_tensor.to(torch.float), affine=subject['T1_image'].affine)
        output_seg.save(args.output)
        

    if args.ground_truth is not None:
        gt_image= sub['T2_image']
        gt_image.save(t2)
        T1=sub['T1_image']
        T1.save(t1)
        if S is not None:
            label=sub['label']
            label.save(s)
        if args.test_image is None:
            pred_image = output_seg
        else:
            pred_image=tio.ScalarImage(args.test_image)

        if args.diff:
            GT_T1 = tio.ScalarImage(tensor=sub['T2_image'][tio.DATA]-sub['T1_image'][tio.DATA], affine=subject['T1_image'].affine)
            GT_T1.save('/home/claire/Nets_Reconstruction/GT-T1.nii.gz')
            GT_EST = tio.ScalarImage(tensor=sub['T2_image'][tio.DATA]-sub['T2_image_estim'][tio.DATA], affine=subject['T1_image'].affine)
            GT_EST.save('/home/claire/Nets_Reconstruction/GT-EST.nii.gz')
            EST_T1 = tio.ScalarImage(tensor=sub['T2_image_estim'][tio.DATA]-sub['T1_image'][tio.DATA], affine=subject['T1_image'].affine)
            EST_T1.save('/home/claire/Nets_Reconstruction/EST-T1.nii.gz')

        min_pred=torch.min(torch.squeeze(pred_image.data,0))
        max_pred=torch.max(torch.squeeze(pred_image.data,0))
        min_gt=torch.min(torch.squeeze(gt_image.data,0))
        max_gt=torch.max(torch.squeeze(gt_image.data,0))
        min_T1=torch.min(torch.squeeze(T1.data,0))
        max_T1=torch.max(torch.squeeze(T1.data,0))

        print(torch.squeeze(T1.data,0).shape)
        print(torch.squeeze(gt_image.data,0).shape)
        print(torch.squeeze(pred_image.data,0).shape)

        psnr=compute_PSNR(max_val=1, prediction=(torch.squeeze(pred_image.data,0)-min_pred)/(max_pred-min_pred), ground_truth=(torch.squeeze(gt_image.data,0)-min_gt)/(max_gt-min_gt))
        psnr_base=compute_PSNR(max_val=1, prediction=(torch.squeeze(T1.data,0)-min_T1)/(max_T1-min_T1), ground_truth=(torch.squeeze(gt_image.data,0)-min_gt)/(max_gt-min_gt))
        difference=psnr.item()-psnr_base.item()
        if difference >=0:
            print('PSNR: '+str(round_to_n(psnr.item(),3))+'  (+'+str(round_to_n(difference,3))+')')
        else:
            print('PSNR: '+str(round_to_n(psnr.item(),3))+'  ('+str(round_to_n(difference,3))+')')

        if S is not None:
            psnr_mask=compute_PSNR_onMask(max_val= 1, prediction=(torch.squeeze(pred_image.data,0)-min_pred)/(max_pred-min_pred), ground_truth=(torch.squeeze(gt_image.data,0)-min_gt)/(max_gt-min_gt), mask=torch.squeeze(label.data,0))
            psnr_mask_base=compute_PSNR_onMask(max_val= 1, prediction=(torch.squeeze(T1.data,0)-min_T1)/(max_T1-min_T1), ground_truth=(torch.squeeze(gt_image.data,0)-min_gt)/(max_gt-min_gt), mask=torch.squeeze(label.data,0))
            difference=psnr_mask.item()-psnr_mask_base.item()
            if difference>=0:
                print('PSNR on mask: '+str(round_to_n(psnr_mask.item(),3))+'  (+'+str(round_to_n(difference,3))+')')
            else:
                print('PSNR on mask: '+str(round_to_n(psnr_mask.item(),3))+'  ('+str(round_to_n(difference,3))+')')

            
            psnr_OOmask=compute_PSNR_outOfMask(max_val= 1, prediction=(torch.squeeze(pred_image.data,0)-min_pred)/(max_pred-min_pred), ground_truth=(torch.squeeze(gt_image.data,0)-min_gt)/(max_gt-min_gt), mask=torch.squeeze(label.data,0))
            psnr_OOmask_base=compute_PSNR_outOfMask(max_val= 1, prediction=(torch.squeeze(T1.data,0)-min_T1)/(max_T1-min_T1), ground_truth=(torch.squeeze(gt_image.data,0)-min_gt)/(max_gt-min_gt), mask=torch.squeeze(label.data,0))
            difference=psnr_OOmask.item()-psnr_OOmask_base.item()
            if difference>=0:
                print('PSNR out of mask: '+str(round_to_n(psnr_OOmask.item(),3))+'  (+'+str(round_to_n(difference,3))+')')
            else:
                print('PSNR out of mask: '+str(round_to_n(psnr_OOmask.item(),3))+'  ('+str(round_to_n(difference,3))+')')


        SSIM=ssim((torch.squeeze(gt_image.data,0).numpy()-min_gt.item())/(max_gt.item()-min_gt.item()), (torch.squeeze(pred_image.data,0).numpy()-min_pred.item())/(max_pred.item()-min_pred.item()), data_range=1,channel_axis=2,multichannel=True)
        SSIM_base=ssim((torch.squeeze(gt_image.data,0).numpy()-min_gt.item())/(max_gt.item()-min_gt.item()), (torch.squeeze(T1.data,0).numpy()-min_T1.item())/(max_T1.item()-min_T1.item()), data_range=1,channel_axis=2,multichannel=True)
        difference=SSIM-SSIM_base
        if difference>=0:
            print('SSIM: '+ str(round_to_n(SSIM,2))+'  (+'+str(round_to_n(difference,2))+')')
        else:
            print('SSIM: '+ str(round_to_n(SSIM,2))+'  ('+str(round_to_n(difference,2))+')')

        if S is not None:
            SSIM_mask=ssim(((torch.squeeze(gt_image.data,0).numpy()-min_gt.item())/(max_gt.item()-min_gt.item()))*torch.squeeze(label.data,0).numpy(), ((torch.squeeze(pred_image.data,0).numpy()-min_pred.item())/(max_pred.item()-min_pred.item()))*torch.squeeze(label.data,0).numpy(), data_range=1,channel_axis=2,multichannel=True)
            SSIM_mask_base=ssim(((torch.squeeze(gt_image.data,0).numpy()-min_gt.item())/(max_gt.item()-min_gt.item()))*torch.squeeze(label.data,0).numpy(), ((torch.squeeze(T1.data,0).numpy()-min_T1.item())/(max_T1.item()-min_T1.item()))*torch.squeeze(label.data,0).numpy(), data_range=1,channel_axis=2,multichannel=True)
            difference=SSIM_mask-SSIM_mask_base
            if difference>=0:
                print('SSIM on mask: '+ str(round_to_n(SSIM_mask,2))+'  (+'+str(round_to_n(difference,2))+')')
            else:
                print('SSIM on mask: '+ str(round_to_n(SSIM_mask,2))+'  ('+str(round_to_n(difference,2))+')')

            SSIM_OOmask=ssim(((torch.squeeze(gt_image.data,0).numpy()-min_gt.item())/(max_gt.item()-min_gt.item()))*torch.abs((torch.squeeze(label.data,0)-1)*(-1)).numpy(), ((torch.squeeze(pred_image.data,0).numpy()-min_pred.item())/(max_pred.item()-min_pred.item()))*torch.abs((torch.squeeze(label.data,0)-1)*(-1)).numpy(), data_range=1,channel_axis=2,multichannel=True)
            SSIM_OOmask_base=ssim(((torch.squeeze(gt_image.data,0).numpy()-min_gt.item())/(max_gt.item()-min_gt.item()))*torch.abs((torch.squeeze(label.data,0)-1)*(-1)).numpy(), ((torch.squeeze(T1.data,0).numpy()-min_T1.item())/(max_T1.item()-min_T1.item()))*torch.abs((torch.squeeze(label.data,0)-1)*(-1)).numpy(), data_range=1,channel_axis=2,multichannel=True)
            difference=SSIM_OOmask-SSIM_OOmask_base
            if difference>=0:
                print('SSIM out of mask: '+ str(round_to_n(SSIM_OOmask,2))+'  (+'+str(round_to_n(difference,2))+')')
            else:
                print('SSIM out of mask: '+ str(round_to_n(SSIM_OOmask,2))+'  ('+str(round_to_n(difference,2))+')')
