import os
import subprocess
import glob
import nibabel as nib
from skimage import morphology
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool
import sys

fslcc = '/usr/local/fsl/bin/fslcc'

def dilation(file_in, file_dilated, r):
    img = nib.load(file_in)
    img_array = img.get_fdata()
    kernel = morphology.ball(r)
    img_array_dilated_bool = morphology.binary_dilation(img_array, kernel)
    img_array_dilated = img_array_dilated_bool.astype(img.get_data_dtype())
    img_dilated = nib.Nifti1Image(img_array_dilated, img.affine)
    nib.save(img_dilated, file_dilated)


def blurring(file_in, file_blurred):
    img = nib.load(file_in)
    img_array = img.get_fdata().astype(float)
    img_array_blurred = gaussian_filter(img_array, 2)
    img_blurred = nib.Nifti1Image(img_array_blurred, img.affine)
    nib.save(img_blurred, file_blurred)


def controlequalite(registration,dyn,ind):
    command = "{} {} {}".format(fslcc,registration,dyn)
    ind=0
    corr = subprocess.check_output(command, shell=True)
    #corr=str(corr)
    corr.replace(b" ",b"")
    corr.replace(b"  ",b"")
    l = corr.split(b'\n')
    if len(l[0])==0:
        print('IMAGE RECALEE EGALE A ZERO')
        return False
    else:
        val = float(l[ind][8:12])
        #print('\t \t '+str(val))
        if (val>=0.62):
            print('\t \t Test qualité OK. Coefficient de corrélation = ', str(val))
            return True
        else:
            print('\t \t Test qualité insuffisant. Coefficient de corrélation = ', str(val))
            return False


def registration(static_directory, segment_directory, result_directory, dynamic_directory, subjects):
    print('')
    print('REGISTRATION')
    flirt = '/usr/local/fsl/bin/flirt'

    #sujets = os.listdir(segment_directory)
    #sujets=['sub_E01','sub_E02','sub_E05','sub_E08']
    
    if subjects is None:
        sujets = os.listdir(segment_directory)
    else:
        sujets = subjects

    for i in range(len(sujets)):
        if (sujets[i]!='sub_E11' and sujets[i]!='sub_E12' and sujets[i]!='sub_E04' and sujets[i]!='sub_E07'):# and sujets[i]!='sub_E13' and sujets[i]!='sub_E09'):# and sujets[i]!='sub_T10' and sujets[i]!='sub_T11'):
            print('******************************************************************************')
            print('SUJET : '+sujets[i])
        	
            #recupere irm statique
            suffixe = '_static_3DT1'
            if os.path.exists(os.path.join(static_directory, sujets[i],sujets[i]+suffixe+'_flipCorrected.nii.gz')):
                file_in = os.path.join(static_directory, sujets[i],sujets[i]+suffixe+'_flipCorrected.nii.gz')
            else:
                file_in = os.path.join(static_directory, sujets[i], sujets[i]+suffixe + '.nii.gz')
        	
            #recupere segmentations
            suffixe = sujets[i] + '_static_3DT1_segment_calcaneus'
            if os.path.exists(os.path.join(segment_directory, sujets[i], suffixe+'_flipped_binarized.nii.gz')):
                file_segment_calcaneus = os.path.join(segment_directory, sujets[i],suffixe+'_flipped_binarized.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, sujets[i], suffixe+'_binarized_flipCorrected.nii.gz')):
                file_segment_calcaneus = os.path.join(segment_directory, sujets[i],suffixe+'_binarized_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, sujets[i], suffixe+'_flipCorrected.nii.gz')):
                file_segment_calcaneus = os.path.join(segment_directory, sujets[i],suffixe+'_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, sujets[i], suffixe+'_binarized.nii.gz')):
                file_segment_calcaneus = os.path.join(segment_directory, sujets[i],suffixe+'_binarized.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, sujets[i], suffixe + '.nii.gz')):
                file_segment_calcaneus = os.path.join(segment_directory, sujets[i], suffixe + '.nii.gz')
            else:
                file_segment_calcaneus = os.path.join(segment_directory, sujets[i], sujets[i]+'_static_3DT1_segment_smooth_calcaneus.nii.gz')
        
            suffixe = sujets[i] + '_static_3DT1_segment_talus'
            if os.path.exists(os.path.join(segment_directory, sujets[i],suffixe+'_flipped_binarized.nii.gz')):
                file_segment_talus = os.path.join(segment_directory, sujets[i],suffixe+'_flipped_binarized.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, sujets[i], suffixe+'_binarized_flipCorrected.nii.gz')):
                file_segment_talus = os.path.join(segment_directory, sujets[i],suffixe+'_binarized_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, sujets[i], suffixe+'_flipCorrected.nii.gz')):
                file_segment_talus = os.path.join(segment_directory, sujets[i],suffixe+'_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, sujets[i], suffixe+'_binarized.nii.gz')):
                file_segment_talus = os.path.join(segment_directory, sujets[i],suffixe+'_binarized.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, sujets[i], suffixe + '.nii.gz')):
                file_segment_talus = os.path.join(segment_directory, sujets[i], suffixe + '.nii.gz')
            else:
                file_segment_talus = os.path.join(segment_directory, sujets[i], sujets[i]+'_static_3DT1_segment_smooth_talus.nii.gz')
        
            suffixe = sujets[i] + '_static_3DT1_segment_tibia'
            if os.path.exists(os.path.join(segment_directory, sujets[i], suffixe+'_flipped_binarized.nii.gz')):
                file_segment_tibia = os.path.join(segment_directory, sujets[i],suffixe+'_flipped_binarized.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, sujets[i], suffixe+'_binarized_flipCorrected.nii.gz')):
                file_segment_tibia = os.path.join(segment_directory, sujets[i],suffixe+'_binarized_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, sujets[i],suffixe+'_flipCorrected.nii.gz')):
                file_segment_tibia = os.path.join(segment_directory, sujets[i],suffixe+'_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, sujets[i], suffixe+'_binarized.nii.gz')):
                file_segment_tibia = os.path.join(segment_directory, sujets[i],suffixe+'_binarized.nii.gz')
            elif os.path.exists(os.path.join(segment_directory, sujets[i], suffixe + '.nii.gz')):
                file_segment_tibia = os.path.join(segment_directory, sujets[i],suffixe + '.nii.gz')
            else:
                file_segment_tibia = os.path.join(segment_directory, sujets[i], sujets[i]+'_static_3DT1_segment_smooth_tibia.nii.gz')
        	
            #cree le repertoire des resultats du registration
            if not os.path.exists(os.path.join(result_directory,sujets[i])):
                os.mkdir(os.path.join(result_directory,sujets[i]))
        
            
            #images = os.listdir(os.path.join(static_directory, sujets[i]))
            images=os.listdir(os.path.join(dynamic_directory,sujets[i], 'volumes'))

            for j in range(len(images)):
                #recupere les irms dynamiques MovieClear
                if images[j].find('MovieClear')!=-1 and not(sujets[i]=='sub_E03' and images[j].find('10')!=-1):
                    if not(sujets[i]=='sub_T01' and images[j].find('flipCorrected')==-1):
                        print('*Vidéo : '+str(images[j]))
        
                        #cree les sous-repertoires necessaires pour les registrations
                        if not os.path.exists(os.path.join(result_directory,sujets[i], 'registrations')):
                            os.mkdir(os.path.join(result_directory,sujets[i], 'registrations'))
                        if not os.path.exists(os.path.join(result_directory,sujets[i], 'registrations', images[j].replace('.nii.gz',''))):
                            os.mkdir(os.path.join(result_directory,sujets[i], 'registrations',images[j].replace('.nii.gz','')))
                
                        #registration statique sur dynamique
                        volumes = os.listdir(os.path.join(dynamic_directory,sujets[i], 'volumes', images[j].replace('.nii.gz',''), 'volumes3D'))

                        for k in range(len(volumes)):
                            print('\t *Volume : '+volumes[k])
                            file_ref = os.path.join(dynamic_directory,sujets[i], 'volumes',images[j].replace('.nii.gz',''), 'volumes3D', volumes[k])
                            recording_directory = os.path.join(result_directory,sujets[i], 'registrations', images[j].replace('.nii.gz',''))

                            if os.path.exists(os.path.join(result_directory,sujets[i], 'registrations',images[j],volumes[k].replace('.nii.gz','')+'_registration.nii.gz')):
                                pass
                            else:
                                print('\t \t REGISTRATION.......')
                                command = "{} -in {} -ref {} -out {} -omat {} -dof 6".format(flirt,
                                        file_in,
                                        file_ref,
                                        os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration.nii.gz'),
                                        os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration.mat'))
                                os.system(command)
        
                            #Controle qualite registration
                            registration = os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration.nii.gz')
                            if(controlequalite(registration,file_ref,k)):
                                for bone in ['calcaneus', 'talus', 'tibia']:
                                    #registration du calcaneus
                                    # bone = 'calcaneus'
                                    ##masks = os.listdir(os.path.join(working_directory,'data_025_8','segmentation', sujets[i],bone+'_dilated','blurred'))
                                    masks = os.listdir(os.path.join(segment_directory, sujets[i], 'segment', bone+'_dilated', 'blurred'))
                                    # #masks=[m for m in masks if (m.find(bone)!=-1 and m.find(sujets[i])!=-1)]
                                    # #print(masks)
                                    for blurred in masks:
                                        if bone == 'calcaneus':
                                            if 'r1' in blurred:
                                                mask = blurred
                                        elif bone == 'talus' or bone == 'tibia':
                                            if 'r2' in blurred:
                                                mask = blurred
                    
                                    if os.path.exists(os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.nii.gz')):
                                        pass
                                    else:
                                        print('\t \t ' + volumes[k] + ' - ' + bone)
                                        command = '{} -in {} -ref {} -inweight {} -out {} -omat {} -init {} -dof 6 -nosearch'.format(flirt,
                                                file_in,
                                                file_ref,
                                                os.path.join(segment_directory, sujets[i], 'segment', bone+'_dilated', 'blurred',mask),
                                                os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.nii.gz'),
                                                os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.mat'),
                                                os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration.mat'))
                                        os.system(command)
                                        
                                    #registration segmentation
                                    if os.path.exists(os.path.join(result_directory,sujets[i],images[j],volumes[k].replace('.nii.gz','')+'_registration_segment_'+bone+'.nii.gz')):
                                        pass
                                    else:
                                        if bone == 'calcaneus':
                                            file_segment = file_segment_calcaneus
                                        elif bone == 'talus':
                                            file_segment = file_segment_talus
                                        elif bone == 'tibia':
                                            file_segment = file_segment_tibia
                                        command = '{} -in {} -ref {} -out {} -init {} -applyxfm'.format(flirt,
                                                file_segment,
                                                file_ref,
                                                os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration_segment_'+bone+'.nii.gz'),
                                                os.path.join(recording_directory,volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.mat'))
                                        os.system(command)

        				

def blur_static(subjects):
    print('DILATATION + BLURRING')
    home='/home/claire/Equinus_BIDS_dataset/derivatives/'
    recording_path='/home/claire/Test_run_recalage/'
    number_os=0
    
    while(number_os<3):
        if(number_os==0):
            bone = 'calcaneus'
        elif(number_os==1):
            bone='talus'
        elif(number_os==2):
            bone='tibia'

        #sujets = os.listdir(home)
        if subjects is None:
            sujets = os.listdir(home)
        else:
            sujets = subjects
    
        for i in range(len(sujets)):
            print(sujets[i])
            suffixe = sujets[i] + '_static_3DT1_segment_smooth_' + bone
            
            file_dilated_path = os.path.join(recording_path, sujets[i])
            if not os.path.isdir(file_dilated_path):
                os.mkdir(file_dilated_path)
            file_dilated_path = os.path.join(file_dilated_path, 'segment')
            if not os.path.isdir(file_dilated_path):
                os.mkdir(file_dilated_path)
            file_dilated_path = os.path.join(file_dilated_path, bone + '_dilated')
            print(file_dilated_path)
            if not os.path.isdir(file_dilated_path):
                os.mkdir(file_dilated_path)
            
            radius_min = 1
            radius_max = 5
            
            file_blurred_path = os.path.join(file_dilated_path, 'blurred')
            if not os.path.isdir(file_blurred_path):
                os.mkdir(file_blurred_path)

            seg=os.path.join(home, sujets[i], suffixe+'.nii.gz')
        
            if os.path.exists(seg):
                print(seg)

                for r in range(radius_min, radius_max+1):
                    prefix = suffixe +'_segment_' + bone + '_static_dilated_r' + str(r)               
                    
                    file_dilated = os.path.join(file_dilated_path, prefix + '.nii.gz')
                    if os.path.exists(file_dilated):
                        pass
                    else:
                        dilation(seg, file_dilated, r)
                    
                    #Floutage des masques
                    file_blurred = os.path.join(file_blurred_path, prefix + '_blurred.nii.gz')
                    if os.path.exists(file_blurred):
                        pass
                    else:
                        blurring(file_dilated, file_blurred)
                        print(file_blurred)
            else:
                print('Fichier '+suffixe+'.nii.gz introuvable')    
        number_os+=1



if __name__ == '__main__':
    # working_directory='/home/claire/Test_recalage'
    # static_directory = os.path.join(working_directory,'HR/256')
    # segment_directory = os.path.join(working_directory,'static_segmentations/256')
    # result_directory= os.path.join(working_directory, 'registration_HR-BR')
    # dynamic_directory= os.path.join(working_directory, 'LR')
    working_directory='/home/claire'
    static_directory = os.path.join(working_directory,'Equinus_BIDS_dataset/sourcedata')
    segment_directory = os.path.join(working_directory,'Equinus_BIDS_dataset/derivatives')
    result_directory= os.path.join(working_directory, 'Test_run_recalage')
    dynamic_directory= os.path.join(working_directory, 'Equinus_BIDS_dataset/derivatives')
    subjects = ['sub_E01']

    blur_static(subjects)
    registration(static_directory, segment_directory, result_directory, dynamic_directory, subjects)




