from re import sub
import re
import nibabel
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag
from skimage.morphology import disk
from skimage.filters import median, gaussian
from skimage.transform import resize
from scipy.ndimage.morphology import grey_erosion
import os
import sys
import shutil
import argparse
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
import glob
import math

def normalize2D(im2D):
    mean=im2D[:,:].mean()
    std=im2D[:,:].std()
    if std==0:
        return(im2D)
    else:
        im2D[:,:]=(im2D[:,:]-mean)/std
        return(im2D)

def normalize3D(im3D):
    mean=im3D.mean()
    std=im3D.std()
    if std==0:
        return(im3D)
    else:
        im3D=(im3D-mean)/std
        return(im3D)

def Downgrade(data_path, result_path, d, std, sigma):
    subjects=os.listdir(data_path)
    #subjects=['sub_T11']
    forbidden_subjets=['sub_E09','sub_T09','sub_T07']
    for s in subjects:
        if s not in forbidden_subjets:
            print(s)
            # files=os.listdir(os.path.join(data_path,s))
            # files=[f for f in files if (f.split('_')[2]=='static')]
            print(os.path.join(data_path,s,s+'_static_3DT1.nii.gz'))
            # if os.path.exists(os.path.join(data_path,s,s+'_static_3DT1_flipCorrected.nii.gz')):
            #     name_mask=s+'_mask_static_3DT1_flipCorrected.nii.gz'
            #     name_down=s+'_downgrade_static_3DT1_flipCorrected.nii.gz'
            #     name=s+'_static_3DT1_flipCorrected.nii.gz'
            #     file=os.path.join(data_path,s,s+'_static_3DT1_flipCorrected.nii.gz')
            if os.path.exists(os.path.join(data_path,s,s+'_static_3DT1.nii.gz')):
                name_mask=s+'_mask_static_3DT1.nii.gz'
                name_down=s+'_downgrade_static_3DT1.nii.gz'
                name=s+'_static_3DT1.nii.gz'
                file=os.path.join(data_path,s,s+'_static_3DT1.nii.gz')
            else:
                continue

            # if not os.path.exists(os.path.join(result_path,'derivatives',name_down)):
            if not os.path.exists(os.path.join(result_path,name_down)):
                original=nibabel.load(file)
                original_image=original.get_fdata()
                downgrade_image=np.zeros((original_image.shape[0],original_image.shape[1],original_image.shape[2]))
                original_image=normalize3D(original_image)
                for i in range(original_image.shape[2]):
                    image2D=original_image[:,:,i]
                    im_med=median(image2D, disk(d))
                    noise=gaussian(grey_erosion(grey_erosion(np.random.normal(0,std,(im_med.shape[0],im_med.shape[1])),size=(2,2)),size=(4,4)),sigma=(0.5))
                    im_nse=im_med+noise
                    im_dwn=resize(im_nse, (original_image.shape[0]//2,original_image.shape[1]//2), order=0)
                    im_gsn=gaussian(im_dwn, sigma=sigma)
                    im_up=resize(im_gsn, (original_image.shape[0],original_image.shape[1]), order=0)
                    downgrade_image[:,:,i]=im_up
                downgrade=nibabel.Nifti1Image(downgrade_image,affine=original.affine,header=original.header)
                # nibabel.save(downgrade,os.path.join(result_path,'derivatives',name_down))
                nibabel.save(downgrade,os.path.join(result_path,name_down))
                #nibabel.save(original, os.path.join(result_path,'sourcedata',name))

            # if os.path.exists(os.path.join(result_path,'derivatives',name_down)): # virer T11
            #     original=nibabel.load(os.path.join(result_path,'sourcedata',name))
            #     downgrade=nibabel.load(os.path.join(result_path,'derivatives',name_down))
            #     downgrade_image=downgrade.get_fdata()
            #     downgrade=nibabel.Nifti1Image(downgrade_image,affine=original.affine, header=original.header)
            #     nibabel.save(downgrade,os.path.join(result_path,'derivatives',name_down))
            if os.path.exists(os.path.join(result_path,name_down)): # virer T11
                original=nibabel.load(os.path.join("/mnt/Data/Equinus_BIDS_dataset/data_025_05/sourcedata",name))
                downgrade=nibabel.load(os.path.join(result_path,name_down))
                downgrade_image=downgrade.get_fdata()
                downgrade=nibabel.Nifti1Image(downgrade_image,affine=original.affine, header=original.header)
                nibabel.save(downgrade,os.path.join(result_path,name_down))

            # if not os.path.exists(os.path.join(result_path,'derivatives',name_mask)):
            #     if os.path.exists(os.path.join('/mnt/Data/Equinus_BIDS_dataset/derivatives',s,'footmask.nii.gz')):
            #         shutil.copyfile(os.path.join('/mnt/Data/Equinus_BIDS_dataset/derivatives',s,'footmask.nii.gz'),os.path.join(result_path,'derivatives',name_mask))
            #         seg=nibabel.load(os.path.join(result_path,'derivatives',name_mask))
            #         seg_image=seg.get_fdata()
            #         # os.remove(os.path.join(result_path,'footmasks',name_mask))
            #         seg=nibabel.Nifti1Image(seg_image, affine=original.affine, header=original.header)
            #         nibabel.save(seg, os.path.join(result_path,'footmasks',name_mask))
            #     else:
            #         sys.exit('Footmask inexistant')


            # if os.path.exists(os.path.join(result_path,'derivatives',name_mask)):
            #     seg=nibabel.load(os.path.join(result_path,'derivatives',name_mask))
            #     LR=nibabel.load(os.path.join(result_path,'derivatives',name_down))
            #     HR=nibabel.load(os.path.join(result_path,'sourcedata',name))
            #     # print(seg.header['pixdim'])
            #     # print(LR.header['pixdim'])
            #     # print(HR.header['pixdim'])
            #     # print(' ')
            #     seg.header['pixdim']=LR.header['pixdim']
            #     nibabel.save(seg, os.path.join(result_path,'derivatives',name_mask))
            #     # print(seg.header['pixdim'])
            #     # print(LR.header['pixdim'])
            #     # print(HR.header['pixdim'])
            #     # print(' ')
            #     # seg=nibabel.load(os.path.join(result_path,'derivatives',name_mask))
            #     # print(seg.header['pixdim'])
            #     # print(LR.header['pixdim'])
            #     # print(HR.header['pixdim'])
            #     # sys.exit()

def copy_footmask(data_path, result_path):
    subjects=os.listdir(data_path)
    forbidden_subjets=['sub_E09','sub_T09','sub_T07']
    for s in subjects:
        if s not in forbidden_subjets:
            print(s)
            print(os.path.join(data_path,s,s+'_static_3DT1.nii.gz'))
            if os.path.exists(os.path.join(data_path,s,s+'_static_3DT1_flipCorrected.nii.gz')):
                name_mask=s+'_mask_static_3DT1_flipCorrected.nii.gz'
                name_down=s+'_downgrade_static_3DT1_flipCorrected.nii.gz'
                name=s+'_static_3DT1_flipCorrected.nii.gz'
                file=os.path.join(data_path,s,s+'_static_3DT1_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(data_path,s,s+'_static_3DT1.nii.gz')):
                name_mask=s+'_mask_static_3DT1.nii.gz'
                name_down=s+'_downgrade_static_3DT1.nii.gz'
                name=s+'_static_3DT1.nii.gz'
                file=os.path.join(data_path,s,s+'_static_3DT1.nii.gz')
            else:
                continue

            if not os.path.exists(os.path.join(result_path,name_down)):
                original=nibabel.load(file)
                original_image=original.get_fdata()
                downgrade_image=np.zeros((original_image.shape[0],original_image.shape[1],original_image.shape[2]))
                original_image=normalize3D(original_image)

            if not os.path.exists(os.path.join(result_path,'footmasks',name_mask)):
                if os.path.exists(os.path.join('/mnt/Data/Equinus_BIDS_dataset/derivatives',s,'footmask.nii.gz')):
                    shutil.copyfile(os.path.join('/mnt/Data/Equinus_BIDS_dataset/derivatives',s,'footmask.nii.gz'),os.path.join(result_path,'footmasks',name_mask))
                    seg=nibabel.load(os.path.join(result_path,'footmasks',name_mask))
                    seg_image=seg.get_fdata()
                    seg=nibabel.Nifti1Image(seg_image, affine=original.affine, header=original.header)
                    nibabel.save(seg, os.path.join(result_path,'footmasks',name_mask))
                else:
                    sys.exit('Footmask inexistant')



def compute_PSNR(max_val,prediction,ground_truth):
    mse=mean_squared_error(ground_truth,prediction)
    if mse==0:
        return(100)
    else:
        return(20 * math.log10(max_val) - 10 * math.log10(mse))

def compute_PSNR_onMask(max_val,prediction, ground_truth, mask):
    MSE=mean_squared_error(ground_truth,prediction)
    MSE=MSE*mask
    Nombre_pixels_masque=mask.sum()
    MSE=MSE/(Nombre_pixels_masque)
    MSE=MSE.sum()
    return(20 * math.log10(max_val) - 10 * math.log10(MSE))


def PSNR_degradation(source_path, seg_path, data_path, degradations):
    style_degradation=['b','g','r','c','m','k','#9A2FF9','#F06510']
    style_computation=['o','x','*']
    PSNR_SKIMAGE=[]
    PSNR_CLAIRE=[]
    PSNR_SEG=[]
    for deg in degradations:
        print(' ')
        print(deg)
        psnr_skimage=[]
        psnr_claire=[]
        psnr_seg=[]
        path=os.path.join(data_path,deg)
        downgrade_images=os.listdir(path)
        for image in downgrade_images:
            subject=image[:7]
            print(subject)
            # psnr_skimage.append(subject)
            # psnr_claire.append(subject)
            # psnr_seg.append(subject)
            source=glob.glob(os.path.join(source_path,subject+'*.nii.gz'))
            seg=glob.glob(os.path.join(seg_path,subject+'*.nii.gz'))
            if len(source)>1 or len(seg)>1:
                sys.exit("Plus d'une image par sujet (source/seg)")
            source=source[0]
            seg=seg[0]
            image=nibabel.load(os.path.join(path,image)).get_fdata()
            source=nibabel.load(source).get_fdata()
            seg=nibabel.load(seg).get_fdata()
            min_source=np.min(source)
            max_source=np.max(source)
            min_image=np.min(image)
            max_image=np.max(image)
            source=(source-min_source)/(max_source-min_source)
            image=(image-min_image)/(max_image-min_image)
            psnr_skimage.append(peak_signal_noise_ratio(source,image,data_range=1))
            psnr_claire.append(compute_PSNR(1,image,source))
            psnr_seg.append(compute_PSNR_onMask(1,image, source, seg))
        PSNR_SKIMAGE.append(psnr_skimage)
        PSNR_CLAIRE.append(psnr_claire)
        PSNR_SEG.append(psnr_seg)
        plt.figure()
        for i in range(len(PSNR_SKIMAGE)):
            if len(PSNR_SKIMAGE[i]>0):
                plt.plot(PSNR_SKIMAGE[i],marker = style_computation[0], color=style_degradation[i], label=degradations[i])#, linestyle='None')
                plt.plot(PSNR_CLAIRE[i],marker = style_computation[1], color=style_degradation[i])#, linestyle='None')
                plt.plot(PSNR_SEG[i],marker = style_computation[2], color=style_degradation[i])#, linestyle='None')
        plt.legend()
        plt.savefig('/home/aorus-users/claire/PSNR_degradation.png')
    plt.figure()
    for i in range(len(degradations)):
        if len(PSNR_SKIMAGE[i]>0):
            plt.plot(PSNR_SKIMAGE[i],marker = style_computation[0], color=style_degradation[i], label=degradations[i])#, linestyle='None')
            plt.plot(PSNR_CLAIRE[i],marker = style_computation[1], color=style_degradation[i])#, linestyle='None')
            plt.plot(PSNR_SEG[i],marker = style_computation[2], color=style_degradation[i])#, linestyle='None')
    plt.legend()
    plt.savefig('/home/aorus-users/claire/PSNR_degradation.png')
    for i in range(len(degradations)):
        print('PSNR moyen')



def copy_sourcedata():
    data_path='/mnt/Data/Equinus_BIDS_dataset/sourcedata/'
    result_path='/mnt/Data/Equinus_BIDS_dataset/data_025_05/sourcedata/'
    subjects=os.listdir(data_path)
    forbidden_subjects=['sub_E04','sub_E07', 'sub_E12']
    for s in subjects:
        if s not in forbidden_subjects:
            print(s)
            static=glob.glob(os.path.join(data_path,s,'*static*T1*.nii.gz'))
            if len(static)==1:
                static=static[0]
            else:
                flip_corrected=[i for i in static if i.find('flip')!=-1]
                #not_flip=[i for i in static if i.find('flip')==-1]
                static=flip_corrected[0]
            print(static)
            print(os.path.join(data_path,s,static.split('/')[-1]))
            print(os.path.join(result_path,static.split('/')[-1]))
            shutil.copyfile(os.path.join(data_path,s,static.split('/')[-1]),os.path.join(result_path,static.split('/')[-1]))

def copy_segmentation():
    data_path='/mnt/Data/Equinus_BIDS_dataset/sourcedata/'
    seg_path='/mnt/Data/Equinus_BIDS_dataset/derivatives/'
    result_path='/mnt/Data/Equinus_BIDS_dataset/data_025_05/segmentation/'
    subjects=os.listdir(data_path)
    forbidden_subjects=['sub_E04','sub_E07', 'sub_E09', 'sub_E12', 'sub_T09', 'sub_T07']
    for s in subjects:
        if s not in forbidden_subjects:
            print(s)
            static=glob.glob(os.path.join(data_path,s,'*static*T1*.nii.gz'))
            seg=glob.glob(os.path.join(seg_path,s,'*footmask*.nii.gz'))
            if len(static)>0:
                static=static[0]
            if len(seg)>0:
                seg=seg[0]
            print(static)
            print(os.path.join(data_path,s,static.split('/')[-1]))
            print(os.path.join(result_path,static.split('/')[-1]))
            shutil.copyfile(os.path.join(seg_path,s,seg.split('/')[-1]),os.path.join(result_path,static.split('/')[-1].split('.')[0]+'_footmask.nii.gz'))



if __name__ == '__main__':

    #copy_sourcedata()
    #copy_segmentation()


    parser = argparse.ArgumentParser(description='HR image degradation')
    parser.add_argument('-v', '--version', help='Input dataset', type=str, required=True)
    # parser.add_argument('-d', '--disk', help='Max epochs', type=int, required=True)
    # parser.add_argument('-s', '--std', help='Number of channels in Unet', type=float, required=True)
    # parser.add_argument('-S', '--sigma', help='Pytorch initialization model', type=float, required=True)
    args = parser.parse_args()
    data_path="/mnt/Data/Equinus_BIDS_dataset/sourcedata/"
    #result_path="/mnt/Data/Equinus_BIDS_dataset/data_025_05/"
    result_path="/mnt/Data/Equinus_BIDS_dataset/data_025_05/"+args.version+"/"
    d=3
    std=0.3
    sigma=0.7
    Downgrade(data_path,result_path,d,std,sigma)



    #copy_footmask(data_path=data_path, result_path=result_path)



    # source_path='/mnt/Data/Equinus_BIDS_dataset/data_025_05/sourcedata/'
    # seg_path='/mnt/Data/Equinus_BIDS_dataset/data_025_05/segmentation/'
    # data_path='/mnt/Data/Equinus_BIDS_dataset/data_025_05/'
    # degradations=['degradation_1_median', 'degradation_2_noise_1', 'degradation_2_noise_2', 'degradation_2_noise_3', 'degradation_2_noise_4', 'degradation_3_downsample', 'degradation_4_gaussian', 'degradation_5_upsample']
    # PSNR_degradation(source_path, seg_path, data_path, degradations)
