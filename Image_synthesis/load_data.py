import glob
import os
import torchio as tio
import sys
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import random


def load_data(data, segmentation, batch_size, version=None, max_subjects = 400, mean=False, dynamic_path = None, static_path = None, seg_path=None):
    subjects=[]

    if data == 'hcp':
        compteur=0
        check_subjects=[]
        data_path = '/mnt/Data/HCP/HCP100_T1T2'
        out_channels = 1 #10
        all_seg = glob.glob(data_path+'/*_mask.nii.gz', recursive=True)
        all_t2s = glob.glob(data_path+'/*_T2.nii.gz', recursive=True)
        all_t1s = glob.glob(data_path+'/*_T1.nii.gz', recursive=True)


        all_seg = all_seg[:max_subjects] 

        for seg_file in all_seg:
            compteur=compteur+1
            id_subject = seg_file.split('/')[5].split('_')[0]
            #id_subject = id_subject[0]+'_'+id_subject[1]
            if id_subject not in check_subjects:
                check_subjects.append(id_subject)

            t2_file = [s for s in all_t2s if id_subject in s][0]
            t1_file = [s for s in all_t1s if id_subject in s][0]
            
            subject = tio.Subject(
                subject_name=id_subject,
                imageT1=tio.ScalarImage(t1_file),
                imageT2=tio.ScalarImage(t2_file),
                label=tio.LabelMap(seg_file)
            )
            subjects.append(subject)






    elif data == 'equinus_simulate':
        check_subjects=[]
        if version is not None:
            data_path = os.path.join('/home/claire/Equinus_BIDS_dataset/data_025_05/',version)
            print(version)
        else:
            data_path = '/home/claire/Equinus_BIDS_dataset/data_025_05/V1/'
        out_channels = 1 #4
        HR_path=os.path.join('/home/claire/Equinus_BIDS_dataset/data_025_05/', 'sourcedata')
        LR_path=os.path.join(data_path )
        seg_path=os.path.join('/home/claire/Equinus_BIDS_dataset/data_025_05/', 'footmasks_correct_pixdim')
        bones_seg_path=os.path.join('/home/claire/Equinus_BIDS_dataset/data_025_05/', 'bones_segmentations_correct_pixdim')
        subject_names = os.listdir(HR_path)
        forbidden_subjects=['sub_E04','sub_E07', 'sub_E09', 'sub_E12', 'sub_T09', 'sub_T07', 'sub_E11', 'sub_T10', 'sub_T11']
        for s in subject_names:
            if s[:7] not in forbidden_subjects:
                HR=os.path.join(HR_path,s)
                LR_file=s.split('_')
                #seg_file=s.split('_')
                seg_file=s
                LR_file.insert(2,'downgrade')
                seg_file=seg_file.replace('.nii.gz','_footmask.nii.gz')
                # seg_file.insert(2,'mask')
                LR_file=('_').join(LR_file)
                # seg_file=('_').join(seg_file)
                LR=os.path.join(LR_path, LR_file)
                seg=os.path.join(seg_path,seg_file)

                if segmentation is not None:
                    subject=tio.Subject(
                        subject_name=s[:7],
                        LR_image=tio.ScalarImage(LR),
                        HR_image=tio.ScalarImage(HR),
                        label=tio.LabelMap(seg)
                    )
                    talus_seg=glob.glob(os.path.join(bones_seg_path, s[:7]+'*talus.nii.gz'))
                    if len(talus_seg)!=0:
                        ssubject=subject
                        talus_seg=tio.LabelMap(talus_seg[0])
                        ssubject.add_image(talus_seg, 'segmentation_os')
                        subjects.append(ssubject)
                    tibia_seg=glob.glob(os.path.join(bones_seg_path, s[:7]+'*tibia.nii.gz'))
                    if len(tibia_seg)!=0:
                        ssubject=subject
                        tibia_seg=tio.LabelMap(tibia_seg[0])
                        ssubject.add_image(tibia_seg, 'segmentation_os')
                        subjects.append(ssubject)
                    calcaneus_seg=glob.glob(os.path.join(bones_seg_path, s[:7]+'*calcaneus.nii.gz'))
                    if len(calcaneus_seg)!=0:
                        ssubject=subject
                        calcaneus_seg=tio.LabelMap(calcaneus_seg[0])
                        ssubject.add_image(calcaneus_seg, 'segmentation_os')
                        subjects.append(ssubject)
                else:
                    subject=tio.Subject(
                        subject_name=s[:7],
                        LR_image=tio.ScalarImage(LR),
                        HR_image=tio.ScalarImage(HR),
                        label=tio.LabelMap(seg)
                    )
                    subjects.append(subject)

                if s[:7] not in check_subjects:
                    check_subjects.append(s[:7])
    




    elif data=='equinus_sourcedata':
        check_subjects=[]
        #data_path = '/mnt/Data/Equinus_BIDS_dataset/data_05/'
        data_path = '/home/claire/Equinus_BIDS_dataset/'
        out_channels = 1 #4
        #subject_names = ['E01','E02','E03','E05','E06','E08','T01','T02','T03','T04','T05','T06','T08']
        subject_names = os.listdir(os.path.join(data_path,'derivatives'))
        print(subject_names)
        for s in subject_names:
            if os.path.exists(os.path.join(data_path,'derivatives',s,'correct_registrations')):
                sequences=os.listdir(os.path.join(data_path,'derivatives',s,'correct_registrations'))
                for seq in sequences:
                    volumes=os.listdir(os.path.join(data_path,'derivatives',s,'correct_registrations',seq))
                    volumes=[i for i in volumes if (i.split('.')[-1]=='gz' and (i.split('_')[-1].split('.')[0]=='tibia' or i.split('_')[-1].split('.')[0]=='talus' or i.split('_')[-1].split('.')[0]=='calcaneus') and i.split('_')[-2]!='segment')]
                    for v in volumes:
                        HR=os.path.join(data_path,'derivatives',s,'correct_registrations',seq,v)
                        vol=v.split('_')[:-2]
                        vol='_'.join(vol)
                        vol=vol+'.nii.gz'
                        # print(v)
                        # print(vol)
                        # sys.exit()
                        LR=os.path.join(data_path,'derivatives',s,'volumes',seq,'volumes3D',vol)
                        # Pour utiliser les segmentations des os pour extraire les patches: donne des images appariées mais réduit la variabilité au sein des données
                        #seg=v.split('_')
                        #seg.insert(-1,'segment')
                        #seg='_'.join(seg)
                        #segment=os.path.join(data_path,'derivatives',s,'correct_registrations',seq,seg)
                        # Pour utiliser les footmask des données dynamiques -> plus de données mais pas forcémment appariées
                        seg='footmask'+vol
                        segment=os.path.join(data_path,'derivatives',s,'volumes',seq,'footmask',seg)
                        if s not in check_subjects:
                            check_subjects.append(s)
                        subject=tio.Subject(
                            subject_name=s,
                            LR_image=tio.ScalarImage(LR),
                            HR_image=tio.ScalarImage(HR),
                            label=tio.LabelMap(segment)
                        )
                        subjects.append(subject)
            else:
                pass

    elif data=='equinus_025':
        check_subjects=[]
        data_path='/home/claire/Equinus_BIDS_dataset/data_025_8/'
        bones_seg_path=os.path.join('/home/claire/Equinus_BIDS_dataset/data_025_8/', 'correct_registration')
        out_channels = 1 #4
        HR_path=os.path.join(data_path, 'correct_registration')
        LR_path=os.path.join(data_path, 'dynamic')
        subject_names = os.listdir(LR_path)
        forbidden_subjets=['sub_E09','sub_T10','sub_T11','sub_E11'] #'sub_T09','sub_T07'
        for s in subject_names:
            if s not in forbidden_subjets:
                sequences=os.listdir(os.path.join(LR_path,s))
                for seq in sequences:
                    volumes=os.listdir(os.path.join(LR_path,s,seq))
                    for v in volumes:
                        #HR=glob.glob(os.path.join(HR_path,s+'*.nii.gz'))
                        LR=os.path.join(LR_path,s,seq,v)
                        HR_files=glob.glob(os.path.join(HR_path,s,seq,v.split('.')[0]+'*.nii.gz'))
                        HR_files=[i for i in HR_files if i.find('footmask')==-1]
                        HR_files=[i for i in HR_files if i.find('segment')==-1]
                        if len(HR_files)>0:
                            HR=HR_files[0]
                            file=HR.split('/')[-1].replace('_registration','_footmask_registration')
                            SEG=os.path.join(HR_path,s,seq,file)
                            if s not in check_subjects:
                                check_subjects.append(s)
                            if segmentation is not None:
                                subject=tio.Subject(
                                    subject_name=s,
                                    LR_image=tio.ScalarImage(LR),
                                    HR_image=tio.ScalarImage(HR),
                                    label=tio.LabelMap(SEG)
                                )
                                talus_seg=glob.glob(os.path.join(bones_seg_path, s, seq, v.split('.')[0]+'*segment*talus.nii.gz'))
                                if len(talus_seg)!=0:
                                    ssubject=subject
                                    talus_seg=tio.LabelMap(talus_seg[0])
                                    ssubject.add_image(talus_seg, 'segmentation_os')
                                    subjects.append(ssubject)
                                tibia_seg=glob.glob(os.path.join(bones_seg_path, s, seq, v.split('.')[0]+'*segment*tibia.nii.gz'))
                                if len(tibia_seg)!=0:
                                    ssubject=subject
                                    tibia_seg=tio.LabelMap(tibia_seg[0])
                                    ssubject.add_image(tibia_seg, 'segmentation_os')
                                    subjects.append(ssubject)
                                calcaneus_seg=glob.glob(os.path.join(bones_seg_path, s, seq, v.split('.')[0]+'*segment*calcaneus.nii.gz'))
                                if len(calcaneus_seg)!=0:
                                    ssubject=subject
                                    calcaneus_seg=tio.LabelMap(calcaneus_seg[0])
                                    ssubject.add_image(calcaneus_seg, 'segmentation_os')
                                    subjects.append(ssubject)
                            else:
                                subject=tio.Subject(
                                    subject_name=s,
                                    LR_image=tio.ScalarImage(LR),
                                    HR_image=tio.ScalarImage(HR),
                                    label=tio.LabelMap(SEG)
                                )
                                subjects.append(subject)






    elif data=='equinus_iso05':
        check_subjects=[]
        data_path = '/home/claire/Equinus_BIDS_dataset/data_05/'
        out_channels = 1 #4
        #subject_names = ['E01','E02','E03','E05','E06','E08','T01','T02','T03','T04','T05','T06','T08']
        subject_names = ['E01','E02','E05','E08','T01','T02','T03','T05']
        bones=['calcaneus','talus','tibia']
        volume=['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14']
        if mean=="True":
            LR_path=data_path+'mean_registration/'
            sequences=os.listdir(LR_path)
        else:
            LR_path=data_path+'correct_registration/'
            sequences=os.listdir(LR_path)
            sequences=[s for s in sequences if s.split('_')[-2]=='registration']
        for seq in sequences:
            s=seq[:7]
            bone=seq.split('_')[-1].split('.')[0]
            #bone=seq.split('_')[2].split('.')[0]
            LR=LR_path+seq
            HR=data_path+'sourcedata/'+s+'_static_3DT1_flirt.nii.gz'
            #segment=data_path+'segmentation/1_bone_static/'+s+'_static_3DT1_flirt_seg_unet_'+bone+'.nii.gz'
            segment=data_path+'segmentation_propre_mean/'+s+'_static_3DT1_flirt_seg_unet_'+bone+'.nii.gz'

            if s[-3:] not in check_subjects:
                check_subjects.append(s[-3:])
            subject=tio.Subject(
                subject_name=s[-3:],
                LR_image=tio.ScalarImage(LR),
                HR_image=tio.ScalarImage(HR),
                label=tio.LabelMap(segment)
            )
            subjects.append(subject)




    elif data == 'equinus_1':
        check_subjects=[]
        data_path='/home/claire/Equinus_BIDS_dataset/data_025_8/'
        bones_seg_path=os.path.join('/home/claire/Equinus_BIDS_dataset/data_025_8/', 'correct_registration_downcrop','1mm')
        out_channels = 1 #4
        HR_path=os.path.join(data_path, 'correct_registration_downcrop','1mm')
        LR_path=os.path.join(data_path, 'dynamic_downcrop','1mm')
        subject_names = os.listdir(LR_path)
        forbidden_subjets=['sub_E09','sub_T10','sub_T11','sub_E11'] #'sub_T09','sub_T07'
        for s in subject_names:
            if s not in forbidden_subjets:
                sequences=os.listdir(os.path.join(LR_path,s))
                for seq in sequences:
                    volumes=os.listdir(os.path.join(LR_path,s,seq))
                    for v in volumes:
                        #HR=glob.glob(os.path.join(HR_path,s+'*.nii.gz'))
                        LR=os.path.join(LR_path,s,seq,v)
                        HR_files=glob.glob(os.path.join(HR_path,s,seq,v.split('.')[0]+'*.nii.gz'))
                        HR_files=[i for i in HR_files if i.find('footmask')==-1]
                        HR_files=[i for i in HR_files if i.find('segment')==-1]
                        if len(HR_files)>0:
                            HR=HR_files[0]
                            file=HR.split('/')[-1].replace('_registration','_footmask_registration')
                            SEG=os.path.join(HR_path,s,seq,file)
                            if s not in check_subjects:
                                check_subjects.append(s)
                            if segmentation is not None:
                                subject=tio.Subject(
                                    subject_name=s,
                                    LR_image=tio.ScalarImage(LR),
                                    HR_image=tio.ScalarImage(HR),
                                    label=tio.LabelMap(SEG)
                                )
                                talus_seg=glob.glob(os.path.join(bones_seg_path, s, seq, v.split('.')[0]+'*segment*talus.nii.gz'))
                                if len(talus_seg)!=0:
                                    ssubject=subject
                                    talus_seg=tio.LabelMap(talus_seg[0])
                                    ssubject.add_image(talus_seg, 'segmentation_os')
                                    subjects.append(ssubject)
                                tibia_seg=glob.glob(os.path.join(bones_seg_path, s, seq, v.split('.')[0]+'*segment*tibia.nii.gz'))
                                if len(tibia_seg)!=0:
                                    ssubject=subject
                                    tibia_seg=tio.LabelMap(tibia_seg[0])
                                    ssubject.add_image(tibia_seg, 'segmentation_os')
                                    subjects.append(ssubject)
                                calcaneus_seg=glob.glob(os.path.join(bones_seg_path, s, seq, v.split('.')[0]+'*segment*calcaneus.nii.gz'))
                                if len(calcaneus_seg)!=0:
                                    ssubject=subject
                                    calcaneus_seg=tio.LabelMap(calcaneus_seg[0])
                                    ssubject.add_image(calcaneus_seg, 'segmentation_os')
                                    subjects.append(ssubject)
                            else:
                                subject=tio.Subject(
                                    subject_name=s,
                                    LR_image=tio.ScalarImage(LR),
                                    HR_image=tio.ScalarImage(HR),
                                    label=tio.LabelMap(SEG)
                                )
                                subjects.append(subject)






    elif data == 'equinus_256':
        check_subjects=[]
        data_path='/home/claire/Equinus_BIDS_dataset/data_025_8/'
        bones_seg_path=os.path.join('/home/claire/Equinus_BIDS_dataset/data_025_8/', 'correct_registration_downcrop','256')
        out_channels = 1 #4
        HR_path=os.path.join(data_path, 'correct_registration_downcrop','256')
        LR_path=os.path.join(data_path, 'dynamic_downcrop','256')
        subject_names = os.listdir(LR_path)
        forbidden_subjets=['sub_E09','sub_T10','sub_T11','sub_E11'] #'sub_T09','sub_T07'
        for s in subject_names:
            if s not in forbidden_subjets:
                sequences=os.listdir(os.path.join(LR_path,s))
                for seq in sequences:
                    volumes=os.listdir(os.path.join(LR_path,s,seq))
                    for v in volumes:
                        #HR=glob.glob(os.path.join(HR_path,s+'*.nii.gz'))
                        LR=os.path.join(LR_path,s,seq,v)
                        HR_files=glob.glob(os.path.join(HR_path,s,seq,v.split('.')[0]+'*.nii.gz'))
                        HR_files=[i for i in HR_files if i.find('footmask')==-1]
                        HR_files=[i for i in HR_files if i.find('segment')==-1]
                        if len(HR_files)>0:
                            HR=HR_files[0]
                            file=HR.split('/')[-1].replace('_registration','_footmask_registration')
                            SEG=os.path.join(HR_path,s,seq,file)
                            if s not in check_subjects:
                                check_subjects.append(s)
                            if segmentation is not None:
                                subject=tio.Subject(
                                    subject_name=s,
                                    LR_image=tio.ScalarImage(LR),
                                    HR_image=tio.ScalarImage(HR),
                                    label=tio.LabelMap(SEG)
                                )
                                talus_seg=glob.glob(os.path.join(bones_seg_path, s, seq, v.split('.')[0]+'*segment*talus.nii.gz'))
                                if len(talus_seg)!=0:
                                    ssubject=subject
                                    talus_seg=tio.LabelMap(talus_seg[0])
                                    ssubject.add_image(talus_seg, 'segmentation_os')
                                    subjects.append(ssubject)
                                tibia_seg=glob.glob(os.path.join(bones_seg_path, s, seq, v.split('.')[0]+'*segment*tibia.nii.gz'))
                                if len(tibia_seg)!=0:
                                    ssubject=subject
                                    tibia_seg=tio.LabelMap(tibia_seg[0])
                                    ssubject.add_image(tibia_seg, 'segmentation_os')
                                    subjects.append(ssubject)
                                calcaneus_seg=glob.glob(os.path.join(bones_seg_path, s, seq, v.split('.')[0]+'*segment*calcaneus.nii.gz'))
                                if len(calcaneus_seg)!=0:
                                    ssubject=subject
                                    calcaneus_seg=tio.LabelMap(calcaneus_seg[0])
                                    ssubject.add_image(calcaneus_seg, 'segmentation_os')
                                    subjects.append(ssubject)
                            else:
                                subject=tio.Subject(
                                    subject_name=s,
                                    LR_image=tio.ScalarImage(LR),
                                    HR_image=tio.ScalarImage(HR),
                                    label=tio.LabelMap(SEG)
                                )
                                subjects.append(subject)







    elif data == 'equinus_128':
        check_subjects=[]
        data_path='/home/claire/Equinus_BIDS_dataset/data_025_8/'
        bones_seg_path=os.path.join('/home/claire/Equinus_BIDS_dataset/data_025_8/', 'correct_registration_downcrop','128')
        out_channels = 1 #4
        HR_path=os.path.join(data_path, 'correct_registration_downcrop','128')
        LR_path=os.path.join(data_path, 'dynamic_downcrop','128')
        subject_names = os.listdir(LR_path)
        forbidden_subjets=['sub_E09','sub_T10','sub_T11','sub_E11'] #'sub_T09','sub_T07'
        for s in subject_names:
            if s not in forbidden_subjets:
                sequences=os.listdir(os.path.join(LR_path,s))
                for seq in sequences:
                    volumes=os.listdir(os.path.join(LR_path,s,seq))
                    for v in volumes:
                        #HR=glob.glob(os.path.join(HR_path,s+'*.nii.gz'))
                        LR=os.path.join(LR_path,s,seq,v)
                        HR_files=glob.glob(os.path.join(HR_path,s,seq,v.split('.')[0]+'*.nii.gz'))
                        HR_files=[i for i in HR_files if i.find('footmask')==-1]
                        HR_files=[i for i in HR_files if i.find('segment')==-1]
                        if len(HR_files)>0:
                            HR=HR_files[0]
                            file=HR.split('/')[-1].replace('_registration','_footmask_registration')
                            SEG=os.path.join(HR_path,s,seq,file)
                            if s not in check_subjects:
                                check_subjects.append(s)
                            if segmentation is not None:
                                subject=tio.Subject(
                                    subject_name=s,
                                    LR_image=tio.ScalarImage(LR),
                                    HR_image=tio.ScalarImage(HR),
                                    label=tio.LabelMap(SEG)
                                )
                                talus_seg=glob.glob(os.path.join(bones_seg_path, s, seq, v.split('.')[0]+'*segment*talus.nii.gz'))
                                if len(talus_seg)!=0:
                                    ssubject=subject
                                    talus_seg=tio.LabelMap(talus_seg[0])
                                    ssubject.add_image(talus_seg, 'segmentation_os')
                                    subjects.append(ssubject)
                                tibia_seg=glob.glob(os.path.join(bones_seg_path, s, seq, v.split('.')[0]+'*segment*tibia.nii.gz'))
                                if len(tibia_seg)!=0:
                                    ssubject=subject
                                    tibia_seg=tio.LabelMap(tibia_seg[0])
                                    ssubject.add_image(tibia_seg, 'segmentation_os')
                                    subjects.append(ssubject)
                                calcaneus_seg=glob.glob(os.path.join(bones_seg_path, s, seq, v.split('.')[0]+'*segment*calcaneus.nii.gz'))
                                if len(calcaneus_seg)!=0:
                                    ssubject=subject
                                    calcaneus_seg=tio.LabelMap(calcaneus_seg[0])
                                    ssubject.add_image(calcaneus_seg, 'segmentation_os')
                                    subjects.append(ssubject)
                            else:
                                subject=tio.Subject(
                                    subject_name=s,
                                    LR_image=tio.ScalarImage(LR),
                                    HR_image=tio.ScalarImage(HR),
                                    label=tio.LabelMap(SEG)
                                )
                                subjects.append(subject)




    elif data == 'dynamic_256':
        rotation = tio.RandomAffine(scales=0,degrees=(30,0,0), translation=0, p=1)
        check_subjects=[]
        data_path='/home/claire/Equinus_BIDS_dataset/data_025_8/'
        bones_seg_path=os.path.join('/home/claire/Equinus_BIDS_dataset/data_025_8/', 'correct_registration_downcrop','256')
        out_channels = 1 #4
        HR_path=os.path.join(data_path, 'correct_registration_downcrop','256')
        LR_path=os.path.join(data_path, 'dynamic_downcrop','256')
        subject_names = os.listdir(LR_path)
        forbidden_subjets=['sub_E09','sub_T10','sub_T11','sub_E11'] #'sub_T09','sub_T07'
        for s in subject_names:
            if s not in forbidden_subjets:
                sequences=os.listdir(os.path.join(LR_path,s))
                for seq in sequences:
                    volumes=os.listdir(os.path.join(LR_path,s,seq))
                    if len(volumes)>1:
                        for v in volumes:
                            #HR=glob.glob(os.path.join(HR_path,s+'*.nii.gz'))
                            LR1=os.path.join(LR_path,s,seq,v)
                            num_volume=int(v.split('.')[0].split('_')[-1][-2:])
                            if num_volume==14:
                                LR2=os.path.join(LR_path,s,seq,v.replace(str(num_volume).zfill(4), str(0).zfill(4)))
                            else:
                                LR2=os.path.join(LR_path,s,seq,v.replace(str(num_volume).zfill(4), str(num_volume+1).zfill(4)))
                            compteur=2
                            while not os.path.exists(LR2):
                                if num_volume+compteur>14:
                                    n=num_volume+compteur-((num_volume+compteur)//14)*14
                                else:
                                    n=num_volume+compteur
                                LR2=os.path.join(LR_path,s,seq,v.replace(str(num_volume).zfill(4), str(n).zfill(4)))
                                compteur=compteur+1
                            HR_files=glob.glob(os.path.join(HR_path,s,seq,v.split('.')[0]+'*.nii.gz'))
                            HR_files=[i for i in HR_files if i.find('footmask')==-1]
                            HR_files=[i for i in HR_files if i.find('segment')==-1]
                            if len(HR_files)>0:
                                HR=HR_files[0]
                                file=HR.split('/')[-1].replace('_registration','_footmask_registration')
                                SEG=os.path.join(HR_path,s,seq,file)
                                if s not in check_subjects:
                                    check_subjects.append(s)
                                subject=tio.Subject(
                                    subject_name=s,
                                    Static_1=tio.ScalarImage(HR),
                                    Static_2=tio.ScalarImage(HR),
                                    Dynamic_1=tio.ScalarImage(LR1),
                                    Dynamic_2=tio.ScalarImage(LR2),
                                    label=tio.LabelMap(SEG)
                                )
                                subject['Static_2']=rotation(subject['Static_2'])
                                subjects.append(subject)
            

    elif data == 'simulate_256':
        check_subjects=[]
        data_path='/home/claire/Equinus_BIDS_dataset/Dataset_validation/Simulation'
        bones_seg_path=os.path.join(data_path, 'segmentations')
        footmask_path=os.path.join(data_path, 'footmasks')
        original_path=os.path.join(data_path, 'static')
        simulate_path=os.path.join(data_path, 'dynamic_simu')
        subject_names = os.listdir(simulate_path)
        #forbidden_subjets=['sub_E09','sub_T10','sub_T11','sub_E11'] #'sub_T09','sub_T07'
        forbidden_subjects=['sub_T10', 'sub_T11']
        for s in subject_names:
            if s not in forbidden_subjects:
                for i in range(20):
                    simulate=glob.glob(os.path.join(simulate_path, s, s+'_static_3DT1_n'+str(i)+'.nii.gz'))[0]
                    original=glob.glob(os.path.join(original_path, s, s+'_static_3DT1_n'+str(i)+'.nii.gz'))[0]
                    footmask=glob.glob(os.path.join(footmask_path, s, s+'_static_3DT1_footmask_n'+str(i)+'.nii.gz'))[0]
                    seg_calcaneus=glob.glob(os.path.join(bones_seg_path, s, s+'_static_3DT1_segment_calcaneus_n'+str(i)+'.nii.gz'))[0]
                    seg_talus=glob.glob(os.path.join(bones_seg_path, s, s+'_static_3DT1_segment_talus_n'+str(i)+'.nii.gz'))[0]
                    seg_tibia=glob.glob(os.path.join(bones_seg_path, s, s+'_static_3DT1_segment_tibia_n'+str(i)+'.nii.gz'))[0]

                    if s not in check_subjects:
                        check_subjects.append(s)
                    subject=tio.Subject(
                        subject_name=s,
                        original=tio.ScalarImage(original),
                        simulate=tio.ScalarImage(simulate),
                        label=tio.LabelMap(footmask),
                        seg_calcaneus=tio.LabelMap(seg_calcaneus),
                        seg_talus=tio.LabelMap(seg_talus),
                        seg_tibia=tio.LabelMap(seg_tibia)
                    )
                    subjects.append(subject)

    elif data == 'simulate_dynamic_256':
        rotation = tio.RandomAffine(scales=0,degrees=(30,0,0), translation=0, p=1)
        check_subjects=[]
        data_path='/home/claire/Equinus_BIDS_dataset/Dataset_validation/Simulation'
        bones_seg_path=os.path.join(data_path, 'segmentations')
        footmask_path=os.path.join(data_path, 'footmasks')
        original_path=os.path.join(data_path, 'static')
        simulate_path=os.path.join(data_path, 'dynamic_simu')
        out_channels = 1 #4
        subject_names = os.listdir(simulate_path)
        #forbidden_subjets=['sub_E09','sub_T10','sub_T11','sub_E11'] #'sub_T09','sub_T07'
        forbidden_subjects=['sub_T10', 'sub_T11']
        for s in subject_names:
            if s not in forbidden_subjects:
                for i in range(20):
                    simulate=glob.glob(os.path.join(simulate_path, s, s+'_static_3DT1_n'+str(i)+'.nii.gz'))[0]
                    original=glob.glob(os.path.join(original_path, s, s+'_static_3DT1_n'+str(i)+'.nii.gz'))[0]
                    footmask=glob.glob(os.path.join(footmask_path, s, s+'_static_3DT1_footmask_n'+str(i)+'.nii.gz'))[0]
                    seg_calcaneus=glob.glob(os.path.join(bones_seg_path, s, s+'_static_3DT1_segment_calcaneus_n'+str(i)+'.nii.gz'))[0]
                    seg_talus=glob.glob(os.path.join(bones_seg_path, s, s+'_static_3DT1_segment_talus_n'+str(i)+'.nii.gz'))[0]
                    seg_tibia=glob.glob(os.path.join(bones_seg_path, s, s+'_static_3DT1_segment_tibia_n'+str(i)+'.nii.gz'))[0]

                    if s not in check_subjects:
                        check_subjects.append(s)
                    subject=tio.Subject(
                        subject_name=s,
                        original_1=tio.ScalarImage(original),
                        original_2=tio.ScalarImage(original),
                        simulate_1=tio.ScalarImage(simulate),
                        simulate_2=tio.ScalarImage(simulate),
                        label=tio.LabelMap(footmask),
                        seg_calcaneus=tio.LabelMap(seg_calcaneus),
                        seg_talus=tio.LabelMap(seg_talus),
                        seg_tibia=tio.LabelMap(seg_tibia)
                    )
                    subject['original_2']=rotation(subject['original_2'])
                    subject['simulate_2']=rotation(subject['simulate_2'])
                    subjects.append(subject)

    elif data == 'monomodal_static_iso05':
        rotation = tio.RandomAffine(scales=0,degrees=(30,0,0), translation=0, center='image', image_interpolation='nearest', p=1)
        check_subjects=[]
        data_path='/home/claire/Equinus_BIDS_dataset/Dataset_validation/iso_05'
        sourcedata_path=os.path.join(data_path, 'static')
        footmask_path=os.path.join(data_path, 'footmasks')
        static_files=os.listdir(sourcedata_path)
        static_files=[i for i in static_files if i.split('.')[-1]=='gz']
        out_channels = 1 #4
        forbidden_subjects=['sub_T10', 'sub_T11']

        n=len([i for i in static_files if i[:7] not in forbidden_subjects])
        figure, axis=plt.subplots((n)//3,6)
        print('figure : '+str(n//3)+' lignes et 6 colonnes (pour '+str(n)+' sujets)')
        compteur=0

        for file in static_files:
            s=file[:7]
            if s not in forbidden_subjects:
                footmask=glob.glob(os.path.join(footmask_path, s+'*.nii.gz'))[0]
                if s not in check_subjects:
                    check_subjects.append(s)
                subject=tio.Subject(
                    subject_name=s,
                    static_1=tio.ScalarImage(os.path.join(sourcedata_path,file)),
                    static_2=tio.ScalarImage(os.path.join(sourcedata_path,file)),
                    label=tio.LabelMap(footmask)
                )
                subject['static_2']=rotation(subject['static_2'])
                subjects.append(subject)

        #         print(s)
        #         # print('\t '+str(subject['static_1'][tio.DATA].shape))
        #         # print('\t compteur = '+str(compteur))
        #         # print('\t ligne = '+str(compteur//3))
        #         # print('\t colonne original = '+str(2*(compteur%3)))
        #         # print('\t colonne rotated = '+str(2*(compteur%3)+1))
        #         axis[compteur//3, 2*(compteur%3)].imshow(subject['static_1'][tio.DATA][0,:,:,80], cmap="gray")
        #         axis[compteur//3, 2*(compteur%3)+1].imshow(subject['static_2'][tio.DATA][0,:,:,80], cmap="gray")
        #         axis[compteur//3, 2*(compteur%3)].set_title('original static - '+s, size = 4)
        #         axis[compteur//3, 2*(compteur%3)+1].set_title('rotated - '+s, size = 4)
        #         axis[compteur//3, 2*(compteur%3)].axis('off')
        #         axis[compteur//3, 2*(compteur%3)+1].axis('off')
        #         compteur = compteur +1

        # figure.tight_layout()
        # figure.savefig('/home/claire/Nets_Reconstruction/Test_rotation.png', dpi = 400)
        # plt.close()
        # sys.exit()
        
    elif data == 'monomodal_dynamic_256':
        check_subjects=[]
        data_path='/home/claire/Equinus_BIDS_dataset/data_025_8/'
        out_channels = 1 #4
        HR_path=os.path.join(data_path, 'correct_registration_downcrop','256')
        LR_path=os.path.join(data_path, 'dynamic_downcrop','256')
        subject_names = os.listdir(LR_path)
        forbidden_subjets=['sub_E09','sub_T10','sub_T11','sub_E11'] #'sub_T09','sub_T07'
        for s in subject_names:
            if s not in forbidden_subjets:
                sequences=os.listdir(os.path.join(LR_path,s))
                for seq in sequences:
                    volumes=os.listdir(os.path.join(LR_path,s,seq))
                    if len(volumes)>1:
                        for v in volumes:
                            #HR=glob.glob(os.path.join(HR_path,s+'*.nii.gz'))
                            LR1=os.path.join(LR_path,s,seq,v)
                            num_volume=int(v.split('.')[0].split('_')[-1][-2:])
                            if num_volume==14:
                                LR2=os.path.join(LR_path,s,seq,v.replace(str(num_volume).zfill(4), str(0).zfill(4)))
                            else:
                                LR2=os.path.join(LR_path,s,seq,v.replace(str(num_volume).zfill(4), str(num_volume+1).zfill(4)))
                            compteur=2
                            while not os.path.exists(LR2):
                                if num_volume+compteur>14:
                                    n=num_volume+compteur-((num_volume+compteur)//14)*14
                                else:
                                    n=num_volume+compteur
                                LR2=os.path.join(LR_path,s,seq,v.replace(str(num_volume).zfill(4), str(n).zfill(4)))
                                compteur=compteur+1
                            HR_files=glob.glob(os.path.join(HR_path,s,seq,v.split('.')[0]+'*.nii.gz'))
                            HR_files=[i for i in HR_files if i.find('footmask')==-1]
                            HR_files=[i for i in HR_files if i.find('segment')==-1]
                            if len(HR_files)>0:
                                HR=HR_files[0]
                                file=HR.split('/')[-1].replace('_registration','_footmask_registration')
                                SEG=os.path.join(HR_path,s,seq,file)
                                if s not in check_subjects:
                                    check_subjects.append(s)
                                subject=tio.Subject(
                                    subject_name=s,
                                    Dynamic_1=tio.ScalarImage(LR1),
                                    Dynamic_2=tio.ScalarImage(LR2),
                                    label=tio.LabelMap(SEG)
                                )
                                subjects.append(subject)


    elif data == 'segmentation_equinus_256':
        check_subjects=[]
        data_path='/home/claire/Equinus_BIDS_dataset/data_025_8/'
        bones_seg_path=os.path.join(data_path, '3bones_segmentation_downcrop')
        out_channels = 1 #4
        HR_path=os.path.join(data_path, 'correct_registration_downcrop','256')
        LR_path=os.path.join(data_path, 'dynamic_downcrop','256')
        subject_names = os.listdir(LR_path)
        forbidden_subjets=['sub_E09','sub_T10','sub_T11','sub_E11'] #'sub_T09','sub_T07'
        for s in subject_names:
            if s not in forbidden_subjets:
                sequences=os.listdir(os.path.join(LR_path,s))
                for seq in sequences:
                    volumes=os.listdir(os.path.join(LR_path,s,seq))
                    for v in volumes:
                        #HR=glob.glob(os.path.join(HR_path,s+'*.nii.gz'))
                        LR=os.path.join(LR_path,s,seq,v)
                        HR_files=glob.glob(os.path.join(HR_path,s,seq,v.split('.')[0]+'*.nii.gz'))
                        HR_files=[i for i in HR_files if (i.find('footmask')==-1 and i.find('segment')==-1 and i.find('4D')==-1)]
                        if len(HR_files)>0:
                            for i in range(len(HR_files)):
                                HR=HR_files[i]
                                file=HR.split('/')[-1].replace('_registration','_footmask_registration')
                                SEG=os.path.join(HR_path,s,seq,file)
                                BONES_SEG = os.path.join(bones_seg_path, s, seq, HR.split('/')[-1].replace('.nii.gz','_segment_3bones.nii.gz'))

                                # print('LR: '+LR)
                                # print('HR: '+HR)
                                # print('footmask: '+SEG)
                                # print('3 bones seg: '+BONES_SEG)

                                # sys.exit()
                                if s not in check_subjects:
                                    check_subjects.append(s)
                                subject=tio.Subject(
                                    subject_name=s,
                                    LR_image=tio.ScalarImage(LR),
                                    HR_image=tio.ScalarImage(HR),
                                    label=tio.LabelMap(SEG), 
                                    bones_segmentation=tio.LabelMap(BONES_SEG)
                                )
                                subjects.append(subject)
                        else:
                            sys.exit("Error in data loading: more than one HR file are corresponding to LR file. HR files: "+str(HR_files))



    elif data == 'bone_segmentation_equinus_256':
        check_subjects=[]
        data_path='/home/claire/Equinus_BIDS_dataset/data_025_8/'
        bones_seg_path=os.path.join(data_path, '1_bone_fuse_segmentation_downcrop')
        out_channels = 1 #4
        HR_path=os.path.join(data_path, 'correct_registration_downcrop','256')
        LR_path=os.path.join(data_path, 'dynamic_downcrop','256')
        subject_names = os.listdir(LR_path)
        forbidden_subjets=['sub_E09','sub_T10','sub_T11','sub_E11'] #'sub_T09','sub_T07'
        for s in subject_names:
            if s not in forbidden_subjets:
                sequences=os.listdir(os.path.join(LR_path,s))
                for seq in sequences:
                    volumes=os.listdir(os.path.join(LR_path,s,seq))
                    for v in volumes:
                        #HR=glob.glob(os.path.join(HR_path,s+'*.nii.gz'))
                        LR=os.path.join(LR_path,s,seq,v)
                        HR_files=glob.glob(os.path.join(HR_path,s,seq,v.split('.')[0]+'*.nii.gz'))
                        HR_files=[i for i in HR_files if (i.find('footmask')==-1 and i.find('segment')==-1 and i.find('4D')==-1)]
                        if len(HR_files)>0:
                            for i in range(len(HR_files)):
                                HR=HR_files[i]
                                file=HR.split('/')[-1].replace('_registration','_footmask_registration').replace('.nii.gz','_bin.nii.gz')
                                label=os.path.join(HR_path,s,seq,file)
                                SEG = os.path.join(bones_seg_path, s, seq, HR.split('/')[-1].replace('.nii.gz','_3-labels-bones.nii.gz'))

                                # CALCA_SEG = os.path.join(bones_seg_path, s, seq, HR.split('/')[-1].replace('.nii.gz','_seg-calcaneus.nii.gz'))
                                # TALUS_SEG = os.path.join(bones_seg_path, s, seq, HR.split('/')[-1].replace('.nii.gz','_seg-talus.nii.gz'))
                                # TIBIA_SEG = os.path.join(bones_seg_path, s, seq, HR.split('/')[-1].replace('.nii.gz','_seg-tibia.nii.gz'))
                                
                                # print('label: ',SEG.split('/')[-3:])
                                # print('LR: ',LR.split('/')[-3:])
                                # print('HR: ',HR.split('/')[-3:])
                                # sys.exit()
                                # print('seg calca: ',CALCA_SEG.split('/')[-3:])
                                # print('seg tibia: ',TIBIA_SEG.split('/')[-3:])
                                # print('seg talus: ',TALUS_SEG.split('/')[-3:])
      
                                if s not in check_subjects:
                                    check_subjects.append(s)
                                subject=tio.Subject(
                                    subject_name=s,
                                    LR_image=tio.ScalarImage(LR),
                                    HR_image=tio.ScalarImage(HR),
                                    label=tio.LabelMap(label), 
                                    segmentations = tio.LabelMap(SEG)
                                )
                                #     calcaneus_segmentation=tio.LabelMap(CALCA_SEG),
                                #     talus_segmentation=tio.LabelMap(TALUS_SEG),
                                #     tibia_segmentation=tio.LabelMap(TIBIA_SEG)
                                # )
                                subjects.append(subject)
                        else:
                            sys.exit("Error in data loading: more than one HR file are corresponding to LR file. HR files: "+str(HR_files))


    elif data == 'bone_segmentation_simulate_256':
        check_subjects=[]
        data_path='/home/claire/Equinus_BIDS_dataset/Dataset_validation/Simulation'
        bones_seg_path=os.path.join(data_path, 'segmentations')
        out_channels = 1 #4
        HR_path=os.path.join(data_path, 'static')
        LR_path=os.path.join(data_path, 'dynamic_simu')
        foot_seg_path=os.path.join(data_path, 'footmasks')
        subject_names = os.listdir(LR_path)
        #forbidden_subjets=['sub_E09','sub_T10','sub_T11','sub_E11'] #'sub_T09','sub_T07'
        forbidden_subjects=['sub_T10', 'sub_T11']
        for s in subject_names:
            if s not in forbidden_subjets:
                for i in range(20):
                    LR=os.path.join(LR_path, s, s+'_static_3DT1_n'+str(i)+'.nii.gz')
                    HR=os.path.join(HR_path, s, s+'_static_3DT1_n'+str(i)+'.nii.gz')
                    SEG=os.path.join(foot_seg_path, s, s+'_static_3DT1_footmask_n'+str(i)+'.nii.gz')
                    CALCA_SEG=os.path.join(bones_seg_path, s, s+'_static_3DT1_segment_calcaneus_n'+str(i)+'_bin.nii.gz')
                    TALUS_SEG=os.path.join(bones_seg_path, s, s+'_static_3DT1_segment_talus_n'+str(i)+'_bin.nii.gz')
                    TIBIA_SEG=os.path.join(bones_seg_path, s, s+'_static_3DT1_segment_tibia_n'+str(i)+'_bin.nii.gz')

                    if (os.path.exists(LR) and os.path.exists(HR) and os.path.exists(SEG) and os.path.exists(CALCA_SEG) and os.path.exists(TALUS_SEG) and os.path.exists(TIBIA_SEG)):
                        print('OK POUR LA DÉFINITION DES CHEMINS (LOAD_DATA.PY)')
                    else:
                        print(LR)
                        print(HR)
                        print(SEG)
                        print(CALCA_SEG)
                        print(TALUS_SEG)
                        print(TIBIA_SEG)
                    sys.exit()

                    if s not in check_subjects:
                        check_subjects.append(s)
                    subject=tio.Subject(
                        subject_name=s,
                        LR_image=tio.ScalarImage(LR),
                        HR_image=tio.ScalarImage(HR),
                        label=tio.LabelMap(SEG), 
                        calcaneus_segmentation=tio.LabelMap(CALCA_SEG),
                        talus_segmentation=tio.LabelMap(TALUS_SEG),
                        tibia_segmentation=tio.LabelMap(TIBIA_SEG)
                    )
                    subjects.append(subject)

    elif data == 'dhcp_2mm':
        c_ok=0
        c_no=0
        c_no_done=0
        c_no_forbid=0
        check_subjects=[]
        data_path='/home/claire/DHCP/2mm/'
        images=os.listdir(data_path)
        images=[i for i in images if i[:3]=="sub"]
        done=[]
        forbidden_subjects=['CC00205XX07', 'CC00434AN14', 'CC00860XX11', 'CC00216AN10', 'CC00143BN12', 'CC00468XX15', 'CC00446XX18', 'CC00402XX06', 'CC00152AN04', 'CC00616XX14', 'CC00664XX13', 'CC00850XX09', 'CC00478XX17', 'CC00083XX10', 'CC00379XX17', 'CC00753XX11', 'CC00102XX03', 'CC00569XX17', 'CC00863XX14', 'CC00445XX17', 'CC00797XX23', 'CC00131XX08', 'CC00414XX10', 'CC00120XX05', 'CC00593XX17', 'CC00114XX07', 'CC00129AN14', 'CC00121XX06', 'CC00693XX18', 'CC00505XX10', 'CC00194XX14', 'CC00135AN12', 'CC00113XX06']
        for im in images:
            subject_session=('_').join(im.split('_')[:2])
            sub = subject_session.split('_')[0].split('-')[1]
            #print(sub)
            if (subject_session not in done and sub not in forbidden_subjects):
                #print("HEY")
                c_ok+=1
                done.append(subject_session)
                t1=os.path.join(data_path, [i for i in images if (i.find(subject_session)!=-1 and i.find('T1')!=-1)][0])
                t2=os.path.join(data_path, [i for i in images if (i.find(subject_session)!=-1 and i.find('T1')==-1 and i.find('seg')==-1)][0])
                s=os.path.join(data_path, [i for i in images if (i.find(subject_session)!=-1 and i.find('seg')!=-1)][0])
                if sub not in check_subjects:
                    check_subjects.append(sub)
                subject=tio.Subject(
                    subject_name=sub,
                    T1=tio.ScalarImage(t1),
                    T2=tio.ScalarImage(t2),
                    label=tio.LabelMap(s)
                )
                subjects.append(subject)
            else:
                pass
    
    elif data == 'dhcp_1mm':
        check_subjects=[]
        data_path='/home/claire/DHCP/1mm/'
        images=os.listdir(data_path)
        images=[i for i in images if i[:3]=="sub"]
        done=[]
        for im in images:
            subject_session=('_').join(im.split('_')[:2])
            if subject_session not in done:
                done.append(subject_session)
                t1=os.path.join(data_path, [i for i in images if (i.find(subject_session)!=-1 and i.find('T1')!=-1)][0])
                t2=os.path.join(data_path, [i for i in images if (i.find(subject_session)!=-1 and i.find('T1')==-1 and i.find('seg')==-1)][0])
                s=os.path.join(data_path, [i for i in images if (i.find(subject_session)!=-1 and i.find('seg')!=-1)][0])
                sub = subject_session.split('_')[0].split('-')[1]
                if sub not in check_subjects:
                    check_subjects.append(sub)
                subject=tio.Subject(
                    subject_name=sub,
                    T1=tio.ScalarImage(t1),
                    T2=tio.ScalarImage(t2),
                    label=tio.LabelMap(s)
                )
                subjects.append(subject)
            else:
                pass


    elif data == 'dhcp_original':
        check_subjects=[]
        data_path='/home/claire/DHCP/original/'
        images=os.listdir(data_path)
        images=[i for i in images if i[:3]=="sub"]
        done=[]
        for im in images:
            subject_session=('_').join(im.split('_')[:2])
            if subject_session not in done:
                done.append(subject_session)
                t1=os.path.join(data_path, [i for i in images if (i.find(subject_session)!=-1 and i.find('T1')!=-1)][0])
                t2=os.path.join(data_path, [i for i in images if (i.find(subject_session)!=-1 and i.find('T1')==-1 and i.find('seg')==-1)][0])
                s=os.path.join(data_path, [i for i in images if (i.find(subject_session)!=-1 and i.find('seg')!=-1)][0])
                sub = subject_session.split('_')[0].split('-')[1]
                if sub not in check_subjects:
                    check_subjects.append(sub)
                subject=tio.Subject(
                    subject_name=sub,
                    T1=tio.ScalarImage(t1),
                    T2=tio.ScalarImage(t2),
                    label=tio.LabelMap(s)
                )
                subjects.append(subject)
            else:
                pass
    
    elif data== 'MNIST':
        transform = transforms.ToTensor() # convert data to torch.FloatTensor
        train_data = datasets.MNIST(root = 'data', train = True, download = True, transform=transform)
        test_data = datasets.MNIST(root = 'data', train = False, download = True, transform=transform)
        data_1 = []
        for i in range(len(train_data)):
            if train_data[i][1]==5:
                data_1.append(torch.nn.functional.interpolate(train_data[i][0].unsqueeze(0), scale_factor=2))

        data_test_1 = []
        for i in range(len(test_data)):
            if test_data[i][1]==5:
                #data_test_1.append(test_data[i][0].unsqueeze(0))
                data_test_1.append(torch.nn.functional.interpolate(test_data[i][0].unsqueeze(0), scale_factor=2))

        data_2 = data_1.copy()
        random.shuffle(data_2)
        data_test_2 = data_test_1.copy()
        random.shuffle(data_test_2)
        DATA_1 = torch.cat(data_1, dim = 0)
        DATA_2 = torch.cat(data_2, dim = 0)
        DATA_TEST_1 = torch.cat(data_test_1, dim = 0)
        DATA_TEST_2 = torch.cat(data_test_2, dim = 0)

        print(DATA_1.shape)
        print(DATA_TEST_1.shape)

        dataset_train = torch.utils.data.TensorDataset(DATA_1, DATA_2)
        dataset_test = torch.utils.data.TensorDataset(DATA_TEST_1, DATA_TEST_2)

        loader_train_data = torch.utils.data.DataLoader(dataset_train, batch_size = 64,shuffle=True)
        loader_test_data = torch.utils.data.DataLoader(dataset_test, batch_size = 1)
        
        return loader_train_data, loader_test_data



    elif data == 'dhcp_1mm_npair':
        check_subjects=[]
        data_path='/home/claire/DHCP/1mm/'
        images=os.listdir(data_path)
        images=[i for i in images if i[:3]=="sub"]
        done=[]
        for im in images:
            subject_session=('_').join(im.split('_')[:2])
            if subject_session.split('_')[0] not in done:
                done.append(subject_session.split('_')[0])
                t1=os.path.join(data_path, [i for i in images if (i.find(subject_session)!=-1 and i.find('T1')!=-1)][0])
                t2=os.path.join(data_path, [i for i in images if (i.find(subject_session)!=-1 and i.find('T1')==-1 and i.find('seg')==-1)][0])
                s=os.path.join(data_path, [i for i in images if (i.find(subject_session)!=-1 and i.find('seg')!=-1)][0])

                sub = subject_session.split('_')[0].split('-')[1]
                if sub not in check_subjects:
                    check_subjects.append(sub)
                subject=tio.Subject(
                    subject_name=sub,
                    T1=tio.ScalarImage(t1),
                    T2=tio.ScalarImage(t2),
                    label=tio.LabelMap(s)
                )
                subjects.append(subject)
            else:
                pass
        




    elif data == 'equinus_256_boneseg':
        check_subjects=[]
        data_path='/home/claire/Equinus_BIDS_dataset/data_025_8/'
        bones_seg_path=os.path.join('/home/claire/Equinus_BIDS_dataset/data_025_8/', 'correct_registration_downcrop','256')
        out_channels = 1 #4
        HR_path=os.path.join(data_path, 'correct_registration_downcrop','256')
        LR_path=os.path.join(data_path, 'dynamic_downcrop','256')
        subject_names = os.listdir(LR_path)
        forbidden_subjets=['sub_E09','sub_T10','sub_T11','sub_E11'] #'sub_T09','sub_T07'
        for s in subject_names:
            if s not in forbidden_subjets:
                sequences=os.listdir(os.path.join(LR_path,s))
                for seq in sequences:
                    volumes=os.listdir(os.path.join(LR_path,s,seq))
                    for v in volumes:
                        #HR=glob.glob(os.path.join(HR_path,s+'*.nii.gz'))
                        LR=os.path.join(LR_path,s,seq,v)
                        HR_files=glob.glob(os.path.join(HR_path,s,seq,v.split('.')[0]+'*.nii.gz'))
                        seg = [i for i in HR_files if (i.find('segment')!=-1 and i.find('footmask')==-1)]
                        footmask = [i for i in HR_files if (i.find('segment')==-1 and i.find('footmask')!=-1 and i.find('bin')!=-1)]
                        HR_files=[i for i in HR_files if i.find('footmask')==-1]
                        HR_files=[i for i in HR_files if i.find('segment')==-1]
                        for file in HR_files:
                            bone = file.split('.')[0].split('_')[-1]
                            assert len([i for i in seg if i.find(bone)!=-1])==1 and len([i for i in footmask if i.find(bone)!=-1])==1
                            seg_bone = [i for i in seg if i.find(bone)!=-1][0]
                            SEG = [i for i in footmask if i.find(bone)!=-1][0]
                            subject=tio.Subject(
                                subject_name=s,
                                LR_image=tio.ScalarImage(LR),
                                HR_image=tio.ScalarImage(file),
                                #label=tio.LabelMap(SEG)
                                label=tio.LabelMap(seg_bone)
                            )
                            subjects.append(subject)


    elif data == 'custom':
        check_subjects=[]
        HR_path=os.path.join(static_path)
        LR_path=os.path.join(dynamic_path)
        SEG_path=os.path.join(seg_path)
        subject_names = os.listdir(LR_path)
        forbidden_subjets=['sub_E09','sub_T10','sub_T11','sub_E11'] #'sub_T09','sub_T07'
        for s in subject_names:
            if s not in forbidden_subjets:
                sequences=os.listdir(os.path.join(LR_path,s))
                for seq in sequences:
                    volumes=os.listdir(os.path.join(LR_path,s,seq))
                    for v in volumes:
                        LR=os.path.join(LR_path,s,seq,v)
                        HR_files=glob.glob(os.path.join(HR_path,s,seq,v.split('.')[0]+'*.nii.gz'))
                        HR_files=[i for i in HR_files if (i.find('footmask')==-1 and i.find('segment')==-1 and i.find('4D')==-1)]
                        if len(HR_files)>0:
                            for i in range(len(HR_files)):
                                HR=HR_files[i]
                                file=HR.split('/')[-1].replace('_registration','_footmask_registration').replace('.nii.gz','_bin.nii.gz')
                                label=os.path.join(HR_path,s,seq,file)
                                if seg_path is not None:
                                    SEG = os.path.join(SEG_path, s, seq, HR.split('/')[-1].replace('.nii.gz','_3-labels-bones.nii.gz'))
                                    if s not in check_subjects:
                                        check_subjects.append(s)
                                    subject=tio.Subject(
                                        subject_name=s,
                                        LR_image=tio.ScalarImage(LR),
                                        HR_image=tio.ScalarImage(HR),
                                        label=tio.LabelMap(label), 
                                        segmentations = tio.LabelMap(SEG)
                                    )
                                    subjects.append(subject)
                                else:
                                    if s not in check_subjects:
                                        check_subjects.append(s)
                                    subject=tio.Subject(
                                        subject_name=s,
                                        LR_image=tio.ScalarImage(LR),
                                        HR_image=tio.ScalarImage(HR),
                                        label=tio.LabelMap(label), 
                                    )
                                    subjects.append(subject)



    else:
        sys.exit('Non conform data name')
    return(subjects, check_subjects)

