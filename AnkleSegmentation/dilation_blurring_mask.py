import os
import glob
import nibabel as nib
from skimage import morphology
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool

"""Dilatation et floutage des segmentations pour le recalage"""

def dilation(file_in, file_dilated, r):
    img = nib.load(file_in)
    img_array = img.get_data()
    kernel = morphology.ball(r)
    img_array_dilated_bool = morphology.binary_dilation(img_array, kernel)
    img_array_dilated = img_array_dilated_bool.astype(img.get_data_dtype())
    img_dilated = nib.Nifti1Image(img_array_dilated, img.affine)
    nib.save(img_dilated, file_dilated)


def blurring(file_in, file_blurred):
    img = nib.load(file_in)
    img_array = img.get_data().astype(float)
    img_array_blurred = gaussian_filter(img_array, 2)
    img_blurred = nib.Nifti1Image(img_array_blurred, img.affine)
    nib.save(img_blurred, file_blurred)

if __name__ == '__main__':
    segmentDirectory = '/homes/p20coupe/Bureau/Local/Equinus_BIDS_dataset/derivatives/'
    number_os=0
    
    #Dilatation/Floutage des 3 segmentations
    while(number_os<3):
        if(number_os==0):
            bone = 'calcaneus'
        elif(number_os==1):
            bone='talus'
        elif(number_os==2):
            bone='tibia'
    
        sujets = os.listdir(segmentDirectory)
    
        for i in range(len(sujets)):
            print(sujets[i])
            suffixe = sujets[i] + '_static_3DT1_segment_' + bone
            
            #cree le repertoire pour les segmentations
            if not os.path.isdir(os.path.join(segmentDirectory,sujets[i],'segment')):
                os.mkdir(os.path.join(segmentDirectory,sujets[i],'segment'))
            
            #cree le repertoire pour les segmentations dilatees
            file_dilated_path = os.path.join(segmentDirectory, sujets[i], 'segment', bone + '_dilated/')
            if not os.path.isdir(file_dilated_path):
                os.mkdir(file_dilated_path)
            
            #plage de valeur du rayon de dilatation applique
            radius_min = 1
            radius_max = 5
            
            #cree le repertoire pour les segmentations floutees et dilatees (ce repertoire sera dans le repertoire de dilatation)
            file_blurred_path = os.path.join(file_dilated_path, 'blurred')
            if not os.path.isdir(file_blurred_path):
                os.mkdir(file_blurred_path)
            
           #recupere segmentation de l'os sur IRM Statique
            suffixe = sujets[i] + '_static_3DT1_segment_'+bone
            if os.path.exists(os.path.join(segmentDirectory, sujets[i], suffixe+'_flipped_binarized.nii.gz')):
                file_in = os.path.join(segmentDirectory, sujets[i],suffixe+'_flipped_binarized.nii.gz')
            elif os.path.exists(os.path.join(segmentDirectory, sujets[i],suffixe+'_binarized_flipCorrected.nii.gz')):
                file_in = os.path.join(segmentDirectory, sujets[i],suffixe+'_binarized_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segmentDirectory, sujets[i],suffixe+'_flipCorrected.nii.gz')):
                file_in = os.path.join(segmentDirectory, sujets[i],suffixe+'_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segmentDirectory, sujets[i],suffixe+'_binarized.nii.gz')):
                file_in = os.path.join(segmentDirectory, sujets[i],suffixe+'_binarized.nii.gz')
            elif os.path.exists(os.path.join(segmentDirectory, sujets[i],suffixe + '.nii.gz')):
                file_in = os.path.join(segmentDirectory, sujets[i],suffixe + '.nii.gz')
            elif os.path.exists(os.path.join(segmentDirectory, sujets[i],sujets[i]+'_static_3DT1_segment_smooth_'+bone+'.nii.gz')):
                file_in = os.path.join(segmentDirectory, sujets[i],sujets[i]+'_static_3DT1_segment_smooth_'+bone+'.nii.gz')
            else:
                file_in = ''
            print(file_in)
            
            #On ne dispose pas de la transformation affine de l'IRM statique pour ces 2 sujets: header incorrect
            if (file_in !='' and sujets[i]!='sub_E13' and sujets[i]!='sub_E09'):
                
                #Dilatation des masques
                for r in range(radius_min, radius_max+1):
                    prefix = 'segment_' + bone + '_static_dilated_r' + str(r)               
                       
                    file_dilated = os.path.join(file_dilated_path, prefix + '.nii.gz')
                    if os.path.exists(file_dilated):
                        pass
                    else:
                        dilation(file_in, file_dilated, r)
                	
                    #Floutage des masques
                    file_blurred = os.path.join(file_blurred_path, prefix + '_blurred.nii.gz')
                    if os.path.exists(file_blurred):
                        pass
                    else:
                        blurring(file_dilated, file_blurred)
                
        number_os+=1

