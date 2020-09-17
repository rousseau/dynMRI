from nilearn import image  
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

#path to fslsplit
split = '/homes/p20coupe/Bureau/Local/fsl/bin/fslsplit'

#repertoire contenant les IRMs statiques
dataDirectory = '/homes/p20coupe/Bureau/Local/Equinus_BIDS_dataset/sourcedata/'

#repertoire contenant les IRMs dynamiques 4D
maskDirectory= '/homes/p20coupe/Bureau/Local/Equinus_BIDS_dataset/derivatives/'

#Extraction of Static footmask
sujets = os.listdir(maskDirectory)   
for i in range(len(sujets)):
    
    #header incorrect pour sub_E09 et sub_E13 / 2 IRMs statiques pour E12 
    if not (sujets[i]=='sub_E09' or sujets[i]=='sub_E13' or sujets[i]=='sub_E12'):
        print(sujets[i])
        
        #recupere irm statique
        suffixe = '_static_3DT1'
        if os.path.exists(os.path.join(dataDirectory, sujets[i],sujets[i]+suffixe+'_flipCorrected.nii.gz')):
            file_in = os.path.join(dataDirectory, sujets[i],sujets[i]+suffixe+'_flipCorrected.nii.gz')
        else:
            file_in = os.path.join(dataDirectory, sujets[i], sujets[i]+suffixe + '.nii.gz')
        print(file_in)

        img = image.load_img(file_in)
        #Binarisation de l'image
        mask = image.math_img('img > 220', img=img)
        #Enregistrement du masque du pied
        mask.to_filename(os.path.join(maskDirectory,sujets[i],'footmask.nii.gz'))

       
        #Extraction of Dynamic footmask
        
        #cree repertoire volumes dans derivatives
        if not os.path.exists(os.path.join(maskDirectory, sujets[i],'volumes')):
            os.mkdir(os.path.join(maskDirectory, sujets[i],'volumes'))
		
        #recupere volumes dynamiques 3D
        images = os.listdir(os.path.join(dataDirectory, sujets[i]))
        for j in range(len(images)):
            #recupere les irms dynamiques MovieClear (le volume 10 du sub_E03 est de mauvaise qualité)
            if images[j].find('MovieClear')!=-1 and not(sujets[i]=='sub_E03' and images[j].find('10')!=-1):
                file_ref = images[j]
                
                #cree sous repertoire des sequences dynamiques (volumes3D et footmask)
                if not os.path.exists(os.path.join(maskDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''))):
                    os.mkdir(os.path.join(maskDirectory,sujets[i],'volumes',images[j].replace('.nii.gz','')))
                if not os.path.exists(os.path.join(maskDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'volumes3D')):
                    os.mkdir(os.path.join(maskDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'volumes3D'))
                if not os.path.exists(os.path.join(maskDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'footmask')):
                    os.mkdir(os.path.join(maskDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'footmask'))
				    
                #Division en volumes avec fslsplit
                command = "{} {} {} -t".format(split,os.path.join(dataDirectory,sujets[i],file_ref),os.path.join(maskDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'volumes3D',file_ref.replace('.nii.gz','')+'_vol'))
                os.system(command)

                volumes = os.listdir(os.path.join(maskDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'volumes3D'))
                for k in range (len(volumes)):
                    if volumes[k].find(file_ref.replace('.nii.gz',''))!=-1:
                        print(volumes[k])
                        img = image.load_img(os.path.join(maskDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'volumes3D',volumes[k]))
                        #Binarisation de l'image
                        mask = image.math_img('img > 250', img=img)
                        #Enregistrement du masque du pied
                        mask.to_filename(os.path.join(maskDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'footmask','footmask'+volumes[k]))
