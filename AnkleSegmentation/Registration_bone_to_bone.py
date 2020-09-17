import os
import subprocess

fslcc = '/homes/p20coupe/Bureau/Local/fsl/bin/fslcc'

def controlequalite(registration,dyn,ind):
    command = "{} {} {}".format(fslcc,registration,dyn)
    corr = subprocess.check_output(command, shell=True)
    #corr=str(corr)
    corr.replace(b" ",b"")
    corr.replace(b"  ",b"")
    l = corr.split(b'\n')
    val = float(l[ind][8:12])
    if (val>=0.62):
        return True
    else:
        return False


if __name__ == '__main__':
    dataDirectory = '/homes/p20coupe/Bureau/Local/Equinus_BIDS_dataset/sourcedata/'
    segmentDirectory = '/homes/p20coupe/Bureau/Local/Equinus_BIDS_dataset/derivatives/'

    flirt = '/homes/p20coupe/Bureau/Local/fsl/bin/flirt'

    sujets = os.listdir(segmentDirectory)

    for i in range(len(sujets)):
        if (sujets[i]!='sub_E09' and sujets[i]!='sub_E11' and sujets[i]!='sub_E12' and sujets[i]!='sub_E13'):
            print(sujets[i])
        	
            #recupere irm statique
            suffixe = '_static_3DT1'
            if os.path.exists(os.path.join(dataDirectory, sujets[i],sujets[i]+suffixe+'_flipCorrected.nii.gz')):
                file_in = os.path.join(dataDirectory, sujets[i],sujets[i]+suffixe+'_flipCorrected.nii.gz')
            else:
                file_in = os.path.join(dataDirectory, sujets[i], sujets[i]+suffixe + '.nii.gz')
            print(file_in)
        	
            #recupere segmentations
            suffixe = sujets[i] + '_static_3DT1_segment_calcaneus'
            if os.path.exists(os.path.join(segmentDirectory, sujets[i], suffixe+'_flipped_binarized.nii.gz')):
                file_segment_calcaneus = os.path.join(segmentDirectory, sujets[i],suffixe+'_flipped_binarized.nii.gz')
            elif os.path.exists(os.path.join(segmentDirectory, sujets[i], suffixe+'_binarized_flipCorrected.nii.gz')):
                file_segment_calcaneus = os.path.join(segmentDirectory, sujets[i],suffixe+'_binarized_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segmentDirectory, sujets[i], suffixe+'_flipCorrected.nii.gz')):
                file_segment_calcaneus = os.path.join(segmentDirectory, sujets[i],suffixe+'_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segmentDirectory, sujets[i], suffixe+'_binarized.nii.gz')):
                file_segment_calcaneus = os.path.join(segmentDirectory, sujets[i],suffixe+'_binarized.nii.gz')
            elif os.path.exists(os.path.join(segmentDirectory, sujets[i], suffixe + '.nii.gz')):
                file_segment_calcaneus = os.path.join(segmentDirectory, sujets[i], suffixe + '.nii.gz')
            else:
                file_segment_calcaneus = os.path.join(segmentDirectory, sujets[i], sujets[i]+'_static_3DT1_segment_smooth_calcaneus.nii.gz')
        
            suffixe = sujets[i] + '_static_3DT1_segment_talus'
            if os.path.exists(os.path.join(segmentDirectory, sujets[i],suffixe+'_flipped_binarized.nii.gz')):
                file_segment_talus = os.path.join(segmentDirectory, sujets[i],suffixe+'_flipped_binarized.nii.gz')
            elif os.path.exists(os.path.join(segmentDirectory, sujets[i], suffixe+'_binarized_flipCorrected.nii.gz')):
                file_segment_talus = os.path.join(segmentDirectory, sujets[i],suffixe+'_binarized_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segmentDirectory, sujets[i], suffixe+'_flipCorrected.nii.gz')):
                file_segment_talus = os.path.join(segmentDirectory, sujets[i],suffixe+'_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segmentDirectory, sujets[i], suffixe+'_binarized.nii.gz')):
                file_segment_talus = os.path.join(segmentDirectory, sujets[i],suffixe+'_binarized.nii.gz')
            elif os.path.exists(os.path.join(segmentDirectory, sujets[i], suffixe + '.nii.gz')):
                file_segment_talus = os.path.join(segmentDirectory, sujets[i], suffixe + '.nii.gz')
            else:
                file_segment_talus = os.path.join(segmentDirectory, sujets[i], sujets[i]+'_static_3DT1_segment_smooth_talus.nii.gz')
        
            suffixe = sujets[i] + '_static_3DT1_segment_tibia'
            if os.path.exists(os.path.join(segmentDirectory, sujets[i], suffixe+'_flipped_binarized.nii.gz')):
                file_segment_tibia = os.path.join(segmentDirectory, sujets[i],suffixe+'_flipped_binarized.nii.gz')
            elif os.path.exists(os.path.join(segmentDirectory, sujets[i], suffixe+'_binarized_flipCorrected.nii.gz')):
                file_segment_tibia = os.path.join(segmentDirectory, sujets[i],suffixe+'_binarized_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segmentDirectory, sujets[i],suffixe+'_flipCorrected.nii.gz')):
                file_segment_tibia = os.path.join(segmentDirectory, sujets[i],suffixe+'_flipCorrected.nii.gz')
            elif os.path.exists(os.path.join(segmentDirectory, sujets[i], suffixe+'_binarized.nii.gz')):
                file_segment_tibia = os.path.join(segmentDirectory, sujets[i],suffixe+'_binarized.nii.gz')
            elif os.path.exists(os.path.join(segmentDirectory, sujets[i], suffixe + '.nii.gz')):
                file_segment_tibia = os.path.join(segmentDirectory, sujets[i],suffixe + '.nii.gz')
            else:
                file_segment_tibia = os.path.join(segmentDirectory, sujets[i], sujets[i]+'_static_3DT1_segment_smooth_tibia.nii.gz')
        	
            #cree le repertoire des resultats du registration
            if not os.path.exists(os.path.join(segmentDirectory,sujets[i],'registrations')):
                os.mkdir(os.path.join(segmentDirectory,sujets[i],'registrations'))
        
        
            images = os.listdir(os.path.join(dataDirectory, sujets[i]))
            for j in range(len(images)):
                #recupere les irms dynamiques MovieClear
                if images[j].find('MovieClear')!=-1 and not(sujets[i]=='sub_E03' and images[j].find('10')!=-1):
                    if not(sujets[i]=='sub_T01' and images[j].find('flipCorrected')==-1):
                        file_ref = images[j]
        
                        #cree les sous-repertoires necessaires pour les registrations
                        if not os.path.exists(os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''))):
                            os.mkdir(os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz','')))
                
                        #registration statique sur dynamique
                        volumes = os.listdir(os.path.join(segmentDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'volumes3D'))
                        for k in range(len(volumes)):        					
                            if os.path.exists(os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration.nii.gz')):
                                pass
                            else:
                                print(volumes[k])
                                command = "{} -in {} -ref {} -out {} -omat {} -dof 6".format(flirt,file_in,os.path.join(segmentDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'volumes3D',volumes[k]),os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration.nii.gz'),os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration.mat'))
                                os.system(command)
        
                            #Controle qualite registration
                            registration = os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration.nii.gz')
        		
                            if(controlequalite(registration,os.path.join(dataDirectory,sujets[i],file_ref),k)):
                                
                                #registration du calcaneus
                                bone = 'calcaneus'
                                masks = os.listdir(os.path.join(segmentDirectory, sujets[i],'segment',bone+'_dilated','blurred'))
                                for blurred in masks:
                                    if 'r1' in blurred:
                                        mask = blurred
                                print(mask)
        		
                                if os.path.exists(os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.nii.gz')):
                                    pass
                                else:
                                    print(volumes[k])
                                    command = '{} -in {} -ref {} -inweight {} -out {} -omat {} -init {} -dof 6 -nosearch'.format(flirt,file_in,os.path.join(segmentDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'volumes3D',volumes[k]),os.path.join(segmentDirectory, sujets[i],'segment',bone+'_dilated','blurred',mask),os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.nii.gz'),os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.mat'),os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration.mat'))
                                    os.system(command)
                                    
                                #registration segmentation
                                if os.path.exists(os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_segment_'+bone+'.nii.gz')):
                                    pass
                                else:
                                    print(file_segment_calcaneus)
                                    command = '{} -in {} -ref {} -out {} -init {} -applyxfm'.format(flirt,file_segment_calcaneus,os.path.join(segmentDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'volumes3D',volumes[k]),os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_segment_'+bone+'.nii.gz'),os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.mat'))
                                    os.system(command)
        				
        		
        
                                #registration du talus
                                bone = 'talus'
                                masks = os.listdir(os.path.join(segmentDirectory, sujets[i],'segment',bone+'_dilated','blurred'))
                                for blurred in masks:
                                    if 'r2' in blurred:
                                        mask = blurred
                                print(mask)
        						
                                if os.path.exists(os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.nii.gz')):
                                    pass
                                else:
                                    print(volumes[k])
                                    command = '{} -in {} -ref {} -inweight {} -out {} -omat {} -init {} -dof 6 -nosearch'.format(flirt,file_in,os.path.join(segmentDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'volumes3D',volumes[k]),os.path.join(segmentDirectory, sujets[i],'segment',bone+'_dilated','blurred',mask),os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.nii.gz'),os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.mat'),os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration.mat'))
                                    os.system(command)
                                    
                                #registration segmentation
                                if os.path.exists(os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_segment_'+bone+'.nii.gz')):
                                    pass
                                else:
                                    print(file_segment_talus)
                                    command = '{} -in {} -ref {} -out {} -init {} -applyxfm'.format(flirt,file_segment_talus,os.path.join(segmentDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'volumes3D',volumes[k]),os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_segment_'+bone+'.nii.gz'),os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.mat'))
                                    os.system(command)
        
                                #registration du tibia
                                bone = 'tibia'
                                masks = os.listdir(os.path.join(segmentDirectory, sujets[i],'segment',bone+'_dilated','blurred'))
                                for blurred in masks:
                                    if 'r2' in blurred:
                                        mask = blurred
                                print(mask)
        						
                                if os.path.exists(os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.nii.gz')):
                                    pass
                                else:
                                    print(volumes[k])
                                    command = '{} -in {} -ref {} -inweight {} -out {} -omat {} -init {} -dof 6 -nosearch'.format(flirt,file_in,os.path.join(segmentDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'volumes3D',volumes[k]),os.path.join(segmentDirectory, sujets[i],'segment',bone+'_dilated','blurred',mask),os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.nii.gz'),os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.mat'),os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration.mat'))
                                    os.system(command)
                                    
                                #registration segmentation
                                if os.path.exists(os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_segment_'+bone+'.nii.gz')):
                                    pass
                                else:
                                    print(file_segment_tibia)
                                    command = '{} -in {} -ref {} -out {} -init {} -applyxfm'.format(flirt,file_segment_tibia,os.path.join(segmentDirectory,sujets[i],'volumes',images[j].replace('.nii.gz',''),'volumes3D',volumes[k]),os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_segment_'+bone+'.nii.gz'),os.path.join(segmentDirectory,sujets[i],'registrations',images[j].replace('.nii.gz',''),volumes[k].replace('.nii.gz','')+'_registration_'+bone+'.mat'))
                                    os.system(command)					
						
					
