import glob
import os
import torchio as tio
import sys



def load_data(data, segmentation, version=None, max_subjects = 400, mean=False):
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
            

    else:
        sys.exit('Non conform data name')

    return(subjects, check_subjects)
