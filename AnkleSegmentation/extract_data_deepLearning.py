import os
import sys
import numpy as np
from extract_PatchesSegmentationHR import get_hcp_2dpatches_SegmentationHR
from extract_Patches import get_hcp_2dpatches
from extract_slices import get_hcp_coupes2d
import argparse

import nibabel as nib
from joblib import dump

parser = argparse.ArgumentParser()
parser.add_argument("Problem", help="Type of problem for the extraction (SegmentationHR/Reconstruction/SegmentationLR)")
parser.add_argument("Type of extraction", help="Type of data to extract (slices or patches)")
args = parser.parse_args()

"""
Ce script permet l'extraction des données nécessaires aux différents problèmes d'apprentissage profond.
Pour l'exécuter, l'utilisateur doit préciser:
    - pour quel type de problème il souhaite extraire des données: SegmentationHR, Reconstruction ou SegmentationLR
    - quels types de données il souhaite extraire: patches ou slices (l'extraction de slices n'a été implémentée que pour la segmentation sur IRM statique)
"""

if(sys.argv[1] == 'Reconstruction'):
    
        """"""""""""""""Pipeline 2 : LR -> HR"""""""""""""""""""""""""""""""""""""""""     
        registrationDirectory = '/homes/p20coupe/Bureau/Local/Equinus_BIDS_dataset/derivatives/'
        
        #seuil de correlation minimal entre le patch BR et le patchHR pour stocker les patchs et les utiliser lors de l'apprentissage 
        niveau_corr = 0.6
        
        sujets = os.listdir(registrationDirectory)   
        for i in range(len(sujets)):
            #Pas de registration fait pour ces sujets (E10 aucun registration correct)
            if(sujets[i]!='sub_E09' and sujets[i]!='sub_E11' and sujets[i]!='sub_E12' and sujets[i]!='sub_E13' and sujets[i]!='sub_E10'):
                print(sujets[i])
            	  
                patches_HR_calcaneus=[]
                patches_HR_talus=[]
                patches_HR_tibia=[]
                patches_BR_calcaneus=[]
                patches_BR_talus=[]
                patches_BR_tibia=[]
                correlation_calcaneus=[]
                correlation_talus=[]
                correlation_tibia=[]
                
                #cree repertoire pour patchs
                if not os.path.exists(os.path.join(registrationDirectory,sujets[i],'DatasetReconstruction_patches')):
                    os.mkdir(os.path.join(registrationDirectory,sujets[i],'DatasetReconstruction_patches'))
            
                #Recupere volume dynamique LR
                sequences = os.listdir(os.path.join(registrationDirectory,sujets[i],'volumes'))
                for seq in sequences:
                    if not(sujets[i]=='sub_T01' and seq.find('flipCorrected')==-1):
                        print(seq)
                        volumes = os.listdir(os.path.join(registrationDirectory,sujets[i],'volumes',seq,'volumes3D'))
                        for k in range(len(volumes)):
                            print(volumes[k])
                            T1s = []
                            dynamique = os.path.join(registrationDirectory,sujets[i],'volumes',seq,'volumes3D',volumes[k])
                            T2s = nib.load(dynamique).get_fdata()
            			
                            #Recupere registration HR des volumes (registration du calcaneus) et le masque du calcaneus sur le registration
                            registrations = os.listdir(os.path.join(registrationDirectory, sujets[i],'correct_registrations',seq))
                            for j in range(len(registrations)):
                                if registrations[j].find(volumes[k].replace('.nii.gz','')) != -1 and registrations[j].find('registration_calcaneus.nii.gz')!=-1:
                                    T1s= nib.load(os.path.join(registrationDirectory, sujets[i],'correct_registrations',seq,registrations[j])).get_fdata()
                                elif registrations[j].find(volumes[k].replace('.nii.gz','')) != -1 and registrations[j].find('segment_calcaneus.nii.gz')!=-1:
                                    masks= nib.load(os.path.join(registrationDirectory, sujets[i],'correct_registrations',seq,registrations[j])).get_fdata()
                            
                            #Si le registration de ce volume pour cet os existe (donc qu'il est correct), ob peut extraire les données   
                            if not T1s==[]:
                                donnees = ([T1s],[T2s],[masks])
                                print("calcaneus")
                                
                                #extraction patchs 32x32
                                (T1,T2)=get_hcp_2dpatches('Reconstruction',0.7,32,8000,donnees)
                                for l in range(T1.shape[0]):
                                    cor = np.corrcoef(T1[l,:,:].flat, T2[l,:,:].flat)[0,1]
                                    #si la correlation entre les deux patchs est inférieure au seuil de corrélation, on ne garde pas les patchs pour l'apprentissage
                                    if cor>niveau_corr:
                                        patches_HR_calcaneus.append(T1[l])
                                        patches_BR_calcaneus.append(T2[l])
                                        correlation_calcaneus.append(cor)
            
                            #Recupere registration HR des volumes (registration du talus) et le masque du talus sur le registration
                            T1s= []
                            for j in range(len(registrations)):
                                if registrations[j].find(volumes[k].replace('.nii.gz','')) != -1 and registrations[j].find('registration_talus.nii.gz')!=-1:
                                    T1s= nib.load(os.path.join(registrationDirectory, sujets[i],'correct_registrations',seq,registrations[j])).get_fdata()
                                elif registrations[j].find(volumes[k].replace('.nii.gz','')) != -1 and registrations[j].find('segment_talus.nii.gz')!=-1:
                                    masks= nib.load(os.path.join(registrationDirectory, sujets[i],'correct_registrations',seq,registrations[j])).get_fdata()
                            
                            if not T1s==[]:
                                donnees = ([T1s],[T2s],[masks])
                                print("talus")
                                
                                #extraction patchs 32x32
                                (T1,T2)=get_hcp_2dpatches('Reconstruction',0.5,32,14000,donnees)
                                for l in range(T1.shape[0]):
                                    cor = np.corrcoef(T1[l,:,:].flat, T2[l,:,:].flat)[0,1]
                                    if cor>niveau_corr:
                                        patches_HR_talus.append(T1[l])
                                        patches_BR_talus.append(T2[l])
                                        correlation_talus.append(cor)
            					
            
                            #Recupere registration HR des volumes (registration du tibia) et le masque du tibia sur le registration
                            T1s=[]
                            for j in range(len(registrations)):
                                if registrations[j].find(volumes[k].replace('.nii.gz','')) != -1 and registrations[j].find('registration_tibia.nii.gz')!=-1:
                                    T1s= nib.load(os.path.join(registrationDirectory, sujets[i],'correct_registrations',seq,registrations[j])).get_fdata()
                                elif registrations[j].find(volumes[k].replace('.nii.gz','')) != -1 and registrations[j].find('segment_tibia.nii.gz')!=-1:
                                    masks= nib.load(os.path.join(registrationDirectory, sujets[i],'correct_registrations',seq,registrations[j])).get_fdata()
                            
                            if not T1s==[]:
                                donnees = ([T1s],[T2s],[masks])
                                print("tibia")
                                
                                #extraction patchs 32x32
                                (T1,T2)=get_hcp_2dpatches('Reconstruction',0.6,32,8000,donnees)
                                for l in range(T1.shape[0]):
                                    cor = np.corrcoef(T1[l,:,:].flat, T2[l,:,:].flat)[0,1]
                                    if cor>niveau_corr:
                                        patches_HR_tibia.append(T1[l])
                                        patches_BR_tibia.append(T2[l])
                                        correlation_tibia.append(cor)
                
                #cree repertoires pour les 3 os
                if not os.path.exists(os.path.join(registrationDirectory,sujets[i],'DatasetReconstruction_patches','calcaneus')):
                    os.mkdir(os.path.join(registrationDirectory,sujets[i],'DatasetReconstruction_patches','calcaneus'))
                if not os.path.exists(os.path.join(registrationDirectory,sujets[i],'DatasetReconstruction_patches','talus')):
                    os.mkdir(os.path.join(registrationDirectory,sujets[i],'DatasetReconstruction_patches','talus'))
                if not os.path.exists(os.path.join(registrationDirectory,sujets[i],'DatasetReconstruction_patches','tibia')):
                    os.mkdir(os.path.join(registrationDirectory,sujets[i],'DatasetReconstruction_patches','tibia'))
                
                #stockage des patchs dans fichiers pickle
                dump(correlation_calcaneus, os.path.join(registrationDirectory,sujets[i],'DatasetReconstruction_patches','calcaneus','corr_calcaneus_'+sujets[i]+'.joblib')) 
                dump(correlation_talus, os.path.join(registrationDirectory,sujets[i],'DatasetReconstruction_patches','talus','corr_talus_'+sujets[i]+'.joblib')) 
                dump(correlation_tibia, os.path.join(registrationDirectory,sujets[i],'DatasetReconstruction_patches','tibia','corr_tibia_'+sujets[i]+'.joblib')) 
                dump(patches_HR_calcaneus, os.path.join(registrationDirectory,sujets[i],'DatasetReconstruction_patches','calcaneus','HR_calcaneus_'+sujets[i]+'.joblib')) 
                dump(patches_HR_talus, os.path.join(registrationDirectory,sujets[i],'DatasetReconstruction_patches','talus','HR_talus_'+sujets[i]+'.joblib')) 
                dump(patches_HR_tibia, os.path.join(registrationDirectory,sujets[i],'DatasetReconstruction_patches','tibia','HR_tibia_'+sujets[i]+'.joblib')) 
                dump(patches_BR_calcaneus, os.path.join(registrationDirectory,sujets[i],'DatasetReconstruction_patches','calcaneus','BR_calcaneus_'+sujets[i]+'.joblib')) 
                dump(patches_BR_talus, os.path.join(registrationDirectory,sujets[i],'DatasetReconstruction_patches','talus','BR_talus_'+sujets[i]+'.joblib')) 
                dump(patches_BR_tibia, os.path.join(registrationDirectory,sujets[i],'DatasetReconstruction_patches','tibia','BR_tibia_'+sujets[i]+'.joblib'))



if(sys.argv[1]=='SegmentationHR'):
    """"""""""""""""Pipeline 1 : Segmentation des IRMs statiques HR"""""""""""""""""""""""""""""""""""""""""
    
    SegmentDirectory = '/homes/p20coupe/Bureau/Local/Equinus_BIDS_dataset/derivatives/'
    dataDirectory = '/homes/p20coupe/Bureau/Local/Equinus_BIDS_dataset/sourcedata/'

    patches_HR=[]
    patches_segHR=[]

    sujets = os.listdir(SegmentDirectory)
    
    for i in range(len(sujets)):
        #header incorrect pour sub_E09 et sub_E13 / pas de segmentation de l'IRM statique T1 pour sub_E11 / 2 IRMs pour sub_E12 (une sans calcaneus)
        if(sujets[i]!='sub_E09' and sujets[i]!='sub_E11' and sujets[i]!='sub_E12' and sujets[i]!='sub_E13'):
            T1=[]
            T2=[]
            T3=[]
            T4=[]
            patches_HR=[]
            patches_segHR_calcaneus=[]
            patches_segHR_talus=[]
            patches_segHR_tibia=[]
                
            print(sujets[i])
            	
            #Cree repertoire pour patchs ou pour les coupes 2D
            if(sys.argv[2]=='patches'):
                if not os.path.exists(os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationHR_patches')):
                		os.mkdir(os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationHR_patches'))
            elif(sys.argv[2]=='slices'):
                if not os.path.exists(os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationHR_slices')):
                		os.mkdir(os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationHR_slices'))
                        
            #Recupere image statique
            suffixe = '_static_3DT1'
            if os.path.exists(os.path.join(dataDirectory, sujets[i],sujets[i]+suffixe+'_flipCorrected.nii.gz')):
                file_in = os.path.join(dataDirectory, sujets[i],sujets[i]+suffixe+'_flipCorrected.nii.gz')
            else:
                file_in = os.path.join(dataDirectory, sujets[i], sujets[i]+suffixe + '.nii.gz')
            print(file_in)
            T2s = nib.load(file_in).get_fdata()
            
            #Recupere masque cheville
            masks= nib.load(os.path.join(SegmentDirectory,sujets[i],'footmask.nii.gz')).get_fdata()
            	
            #Recupere masque calcaneus
            bone = 'calcaneus'
            suffixe = sujets[i] + '_static_3DT1_segment_' + bone
            if os.path.exists(os.path.join(SegmentDirectory, sujets[i], suffixe+'_flipped_binarized.nii.gz')):
                file_seg = os.path.join(SegmentDirectory, sujets[i],suffixe+'_flipped_binarized.nii.gz')
            elif os.path.exists(os.path.join(SegmentDirectory, sujets[i], suffixe+'_binarized_flipCorrected.nii.gz')):
                file_seg = os.path.join(SegmentDirectory, sujets[i],suffixe+'_binarized_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(SegmentDirectory, sujets[i], suffixe+'_flipCorrected.nii.gz')):
                file_seg = os.path.join(SegmentDirectory, sujets[i],suffixe+'_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(SegmentDirectory, sujets[i],suffixe+'_binarized.nii.gz')):
                file_seg = os.path.join(SegmentDirectory, sujets[i],suffixe+'_binarized.nii.gz')
            elif os.path.exists(os.path.join(SegmentDirectory, sujets[i],suffixe+'.nii.gz')):
                file_seg = os.path.join(SegmentDirectory, sujets[i],suffixe + '.nii.gz')
            else:
                file_seg = os.path.join(SegmentDirectory, sujets[i], sujets[i]+'_static_3DT1_segment_smooth_' + bone + '.nii.gz')
            print(file_seg)
            T1s=nib.load(file_seg).get_fdata()
            
            #Recupere masque talus 
            bone = 'talus'
            suffixe = sujets[i] + '_static_3DT1_segment_' + bone
            if os.path.exists(os.path.join(SegmentDirectory, sujets[i], suffixe+'_flipped_binarized.nii.gz')):
                file_seg = os.path.join(SegmentDirectory, sujets[i],suffixe+'_flipped_binarized.nii.gz')
            elif os.path.exists(os.path.join(SegmentDirectory, sujets[i], suffixe+'_binarized_flipCorrected.nii.gz')):
                file_seg = os.path.join(SegmentDirectory, sujets[i],suffixe+'_binarized_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(SegmentDirectory, sujets[i], suffixe+'_flipCorrected.nii.gz')):
                file_seg = os.path.join(SegmentDirectory, sujets[i],suffixe+'_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(SegmentDirectory, sujets[i],suffixe+'_binarized.nii.gz')):
                file_seg = os.path.join(SegmentDirectory, sujets[i],suffixe+'_binarized.nii.gz')
            elif os.path.exists(os.path.join(SegmentDirectory, sujets[i],suffixe+'.nii.gz')):
                file_seg = os.path.join(SegmentDirectory, sujets[i],suffixe + '.nii.gz')
            else:
                file_seg = os.path.join(SegmentDirectory, sujets[i], sujets[i]+'_static_3DT1_segment_smooth_' + bone + '.nii.gz')
            print(file_seg)
            T3s=nib.load(file_seg).get_fdata()
            		
            #Recupere masque tibia
            bone = 'tibia'
            suffixe = sujets[i] + '_static_3DT1_segment_' + bone
            if os.path.exists(os.path.join(SegmentDirectory, sujets[i], suffixe+'_flipped_binarized.nii.gz')):
                file_seg = os.path.join(SegmentDirectory, sujets[i],suffixe+'_flipped_binarized.nii.gz')
            elif os.path.exists(os.path.join(SegmentDirectory, sujets[i], suffixe+'_binarized_flipCorrected.nii.gz')):
                file_seg = os.path.join(SegmentDirectory, sujets[i],suffixe+'_binarized_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(SegmentDirectory, sujets[i], suffixe+'_flipCorrected.nii.gz')):
                file_seg = os.path.join(SegmentDirectory, sujets[i],suffixe+'_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(SegmentDirectory, sujets[i],suffixe+'_binarized.nii.gz')):
                file_seg = os.path.join(SegmentDirectory, sujets[i],suffixe+'_binarized.nii.gz')
            elif os.path.exists(os.path.join(SegmentDirectory, sujets[i],suffixe+'.nii.gz')):
                file_seg = os.path.join(SegmentDirectory, sujets[i],suffixe + '.nii.gz')
            else:
                file_seg = os.path.join(SegmentDirectory, sujets[i], sujets[i]+'_static_3DT1_segment_smooth_' + bone + '.nii.gz')
            print(file_seg)
            T4s=nib.load(file_seg).get_fdata()
            
            #On va extraire pour 1 patch de l'IRM statique, des patchs de segmentation des 3 os -> les 3 os ne sont pas forcément sur tous les patchs
            donnees = ([T1s],[T2s],[T3s],[T4s],[masks])
            
            #Extraction des coupes 2D et des 3 segmentations de celles-ci
            if(sys.argv[2]=="slices"):
                (T1,T2,T3,T4)=get_hcp_coupes2d(donnees)
                print(T1.shape)
                for l in range(T1.shape[0]):
                    patches_HR.append(T2[l])
                    patches_segHR_calcaneus.append(T1[l])
                    patches_segHR_talus.append(T3[l])
                    patches_segHR_tibia.append(T4[l])
                    
                #stockage des coupes 2D dans un fichier pickle
                dump(patches_HR, os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationHR_slices','HR_PipelineSegHR'+sujets[i]+'.joblib')) 
                dump(patches_segHR_calcaneus, os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationHR_slices','seg_PipelineSegHR'+sujets[i]+'calcaneus.joblib'))         
                dump(patches_segHR_talus, os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationHR_slices','seg_PipelineSegHR'+sujets[i]+'talus.joblib')) 
                dump(patches_segHR_tibia, os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationHR_slices','seg_PipelineSegHR'+sujets[i]+'tibia.joblib')) 
            
            #Extraction de patchs 128x128 de l'IRM statique et des 3 segmentations associées
            elif(sys.argv[2]=="patches"):
                (T1,T2,T3,T4)=get_hcp_2dpatches_SegmentationHR(0.75,128,300,donnees)
                for l in range(T1.shape[0]):
                    patches_HR.append(T2[l])
                    patches_segHR_calcaneus.append(T1[l])
                    patches_segHR_talus.append(T3[l])
                    patches_segHR_tibia.append(T4[l])      
                
                #stockage des patchs 128x128 dans un fichier pickle
                dump(patches_HR, os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationHR_patches','HR_PipelineSegHR'+sujets[i]+'.joblib')) 
                dump(patches_segHR_calcaneus, os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationHR_patches','seg_PipelineSegHR'+sujets[i]+'calcaneus.joblib'))         
                dump(patches_segHR_talus, os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationHR_patches','seg_PipelineSegHR'+sujets[i]+'talus.joblib')) 
                dump(patches_segHR_tibia, os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationHR_patches','seg_PipelineSegHR'+sujets[i]+'tibia.joblib')) 
            
            else:
                print("Please enter a correct type of data to extract: slices ou patches")
                quit()


if(sys.argv[1]=='SegmentationLR'):
    
    """"""""""""""""Pipeline 3 : Segmentation des IRMs dynamiques LR"""""""""""""""""""""""""""""""""""""""""
    
    dataDirectory = '/homes/p20coupe/Bureau/Local/Equinus_BIDS_dataset/sourcedata/'
    SegmentDirectory = '/homes/p20coupe/Bureau/Local/Equinus_BIDS_dataset/derivatives/'

    sujets = os.listdir(SegmentDirectory)   
    for i in range(len(sujets)):
        
        #Pas de registration fait pour ces sujets (E10 aucun registration correct)
        if(sujets[i]!='sub_E09' and sujets[i]!='sub_E11' and sujets[i]!='sub_E12' and sujets[i]!='sub_E13' and sujets[i]!='sub_E10'):
            print(sujets[i])
		
            patches_BR_calcaneus=[]
            patches_BR_talus=[]
            patches_BR_tibia=[]
            patches_segBR_calcaneus=[]
            patches_segBR_talus=[]
            patches_segBR_tibia=[]
            
            #Cree repertoire pour patchs
            if not os.path.exists(os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationLR_patches')):
            			os.mkdir(os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationLR_patches'))

            #recupere volumes dynamiques 3D LR
            images = os.listdir(os.path.join(dataDirectory, sujets[i]))
            for j in range(len(images)):
                
                #recupere les irms dynamiques MovieClear
                if (images[j].find('MovieClear')!=-1 and sujets[i]!='sub_T01') or (images[j].find('MovieClear')!=-1 and images[j].find('flipCorrected')!=-1) :
                    if not(sujets[i]=='sub_E03' and images[j].find('10')!=-1):
                        file_ref = images[j]
                        
                        volumes = os.listdir(os.path.join(SegmentDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'volumes3D'))
                        
                        for k in range(len(volumes)):
                            T1s=[]
                            T2s=[]
                            T3s=[]
                            T4s=[]
					
                            if volumes[k].find(file_ref.replace('.nii.gz',''))!=-1 :
                                T2s= nib.load(os.path.join(SegmentDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'volumes3D',volumes[k])).get_fdata()
    		
                                #Recupere masque cheville
                                masks= nib.load(os.path.join(SegmentDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'footmask','footmask'+volumes[k])).get_fdata()
    					
                                #Recupere masque calcaneus recalé
                                bone = 'calcaneus'
                                suffixe= volumes[k].replace('.nii.gz','')+'_registration_segment_' + bone+'.nii.gz'
                                if os.path.exists(os.path.join(SegmentDirectory, sujets[i],'correct_registrations',images[j].replace('.nii.gz',''), suffixe)):
                                    file_seg = os.path.join(SegmentDirectory, sujets[i],'correct_registrations',images[j].replace('.nii.gz',''), suffixe)
                                    T1s=nib.load(file_seg).get_fdata()
    
                                #Recupere masque talus recalé 
                                bone = 'talus'
                                suffixe= volumes[k].replace('.nii.gz','')+'_registration_segment_' + bone+'.nii.gz'
                                if os.path.exists(os.path.join(SegmentDirectory, sujets[i],'correct_registrations',images[j].replace('.nii.gz',''), suffixe)):
                                    file_seg = os.path.join(SegmentDirectory, sujets[i],'correct_registrations',images[j].replace('.nii.gz',''), suffixe)
                                    T3s=nib.load(file_seg).get_fdata()
    						
                                #Recupere masque tibia recalé
                                bone = 'tibia'
                                suffixe= volumes[k].replace('.nii.gz','')+'_registration_segment_' + bone+'.nii.gz'
                                if os.path.exists(os.path.join(SegmentDirectory, sujets[i],'correct_registrations',images[j].replace('.nii.gz',''), suffixe)):
                                    file_seg = os.path.join(SegmentDirectory, sujets[i],'correct_registrations',images[j].replace('.nii.gz',''), suffixe)
                                    T4s=nib.load(file_seg).get_fdata()
    						
    						
                                """Pour ce problème, on extrait pour un patch de l'IRM statique que les patchs de segmentation des os pour lesquels le registration a fonctionné"""
                                #extraction patchs 128x128 calcaneus: 
                                if not T1s==[]:
                                    donnees=([T1s],[T2s],[masks])
                                    (T1,T2)=get_hcp_2dpatches('SegmentationLR',0.75,128,300,donnees)
                                    for l in range(T2.shape[0]):
                                        patches_BR_calcaneus.append(T2[l])
                                        patches_segBR_calcaneus.append(T1[l])
    								
                                #extraction patchs 128x128 talus
                                if not T3s==[]:
                                    donnees=([T3s],[T2s],[masks])
                                    (T1,T2)=get_hcp_2dpatches('SegmentationLR',0.75,128,300,donnees)
                                    for l in range(T2.shape[0]):
                                        patches_BR_talus.append(T2[l])
                                        patches_segBR_talus.append(T1[l])
    
                                #extraction patchs 128x128 tibia
                                if not T4s==[]:
                                    donnees=([T4s],[T2s],[masks])
                                    (T1,T2)=get_hcp_2dpatches('SegmentationLR',0.75,128,300,donnees)
                                    for l in range(T2.shape[0]):
                                        patches_BR_tibia.append(T2[l])
                                        patches_segBR_tibia.append(T1[l])
								            
            #cree repertoires pour les 3 os
            if not os.path.exists(os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationLR_patches','calcaneus')):
                os.mkdir(os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationLR_patches','calcaneus'))
            if not os.path.exists(os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationLR_patches','talus')):
                os.mkdir(os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationLR_patches','talus'))
            if not os.path.exists(os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationLR_patches','tibia')):
                os.mkdir(os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationLR_patches','tibia'))
                    
            #stockage des patchs 128x128 dans fichiers pickle    
            dump(patches_BR_calcaneus,os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationLR_patches','calcaneus','BR_calcaneus_PipelineSegBR_'+sujets[i]+'.joblib')) 
            dump(patches_segBR_calcaneus, os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationLR_patches','calcaneus','seg_PipelineSegBR_calcaneus_'+sujets[i]+'.joblib'))
            dump(patches_BR_talus, os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationLR_patches','talus','BR_talus_PipelineSegBR_'+sujets[i]+'.joblib')) 
            dump(patches_segBR_talus, os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationLR_patches','talus','seg_PipelineSegBR_talus_'+sujets[i]+'.joblib'))
            dump(patches_BR_tibia, os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationLR_patches','tibia','BR_tibia_PipelineSegBR_'+sujets[i]+'.joblib'))
            dump(patches_segBR_tibia, os.path.join(SegmentDirectory,sujets[i],'DatasetSegmentationLR_patches','tibia','seg_PipelineSegBR_tibia_'+sujets[i]+'.joblib'))

#Le type de problème rentré par l'utilisateur est incorrect
else:
    print("Please enter a correct type of problem: SegmentationHR, Reconstruction,SegmentationLR")