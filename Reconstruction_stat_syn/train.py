from os.path import expanduser
from posix import listdir
from numpy import NaN, dtype

from torchio.data import image
from torchio.utils import check_sequence
home = expanduser("~")

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import monai
import os
import sys
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import glob
import multiprocessing
import math
import kornia.augmentation as K

from pytorch_lightning.loggers import TensorBoardLogger, tensorboard
import argparse


def normalize(tensor):
    for i in range(tensor.shape[1]):
        mean=tensor[0,i,:,:].mean()
        std=tensor[0,i,:,:].std()
        tensor[0,i,:,:]=(tensor[0,i,:,:]-mean)/std
    return(tensor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Beo TorchIO training')
    parser.add_argument('-d', '--data', help='Input dataset', type=str, required=False, default = 'hcp')
    parser.add_argument('-e', '--epochs', help='Max epochs', type=int, required=False, default = 10)
    parser.add_argument('-c', '--channels', help='Number of channels in Unet', type=int, required=False, default=32)
    parser.add_argument('-m', '--model', help='Pytorch initialization model', type=str, required=False)

    parser.add_argument('-w', '--workers', help='Number of workers (multiprocessing)', type=int, required=False, default = 16)
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 8)
    parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = (1,64,64))
    parser.add_argument('-s', '--samples', help='Samples per volume', type=int, required=False, default = 8)
    parser.add_argument('-q', '--queue', help='Max queue length', type=int, required=False, default = 64)
    parser.add_argument('-g', '--gpu', help='Number of GPU to use', type=int, required=False, default = 3)
    parser.add_argument('-l', '--loss', help='Loss to use', type=str, required=False, default = 'L2')
    parser.add_argument('-S', '--subjects', help='Number of subjects to use', type=int, required=False, default = 400)
    parser.add_argument('-M', '--mean', help='Number of subjects to use', type=str, required=False, default = "True")


    args = parser.parse_args()
    DATA_AUGMENTATION=True
    print("Nombre de GPUs detectes: "+ str(torch.cuda.device_count()))
    
    max_subjects = args.subjects
    training_split_ratio = 0.9  # use 90% of samples for training, 10% for testing
    num_epochs = args.epochs
    gpu = args.gpu
    num_workers = args.workers#multiprocessing.cpu_count()
    
    training_batch_size = args.batch_size
    validation_batch_size = 1 #args.batch_size
    loss=args.loss
    patch_size = args.patch_size
    samples_per_volume = args.samples
    max_queue_length = args.queue
    
    n_channels = args.channels
    data = args.data

    prefix = 'unet3d_monai_'
    prefix += data
    prefix += '_epochs_'+str(num_epochs)
    prefix += '_patches_'+str(patch_size)
    prefix += '_sampling_'+str(samples_per_volume)
    prefix += '_nchannels_'+str(n_channels)
    prefix += '_mask_image_L2_lr=10-4_seg_corrigees_'

    if args.model is not None:
        prefix += '_using_init'
        
    output_path = home+'/claire/Results_UNet2D_HCP_aug/'
    subjects=[]

    #############################################################################################################################################################################""
    # DATASET
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

    if data == 'equinus':
        # check_subjects=[]
        # #data_path = '/mnt/Data/Equinus_BIDS_dataset/data_05/'
        # data_path = '/mnt/Data/Equinus_BIDS_dataset/'
        # out_channels = 1 #4
        # #subject_names = ['E01','E02','E03','E05','E06','E08','T01','T02','T03','T04','T05','T06','T08']
        # subject_names = os.listdir(os.path.join(data_path,'derivatives'))
        # print(subject_names)
        # for s in subject_names:
        #     if os.path.exists(os.path.join(data_path,'derivatives',s,'correct_registrations')):
        #         sequences=os.listdir(os.path.join(data_path,'derivatives',s,'correct_registrations'))
        #         for seq in sequences:
        #             volumes=os.listdir(os.path.join(data_path,'derivatives',s,'correct_registrations',seq))
        #             volumes=[i for i in volumes if (i.split('.')[-1]=='gz' and (i.split('_')[-1].split('.')[0]=='tibia' or i.split('_')[-1].split('.')[0]=='talus' or i.split('_')[-1].split('.')[0]=='calcaneus') and i.split('_')[-2]!='segment')]
        #             for v in volumes:
        #                 HR=os.path.join(data_path,'derivatives',s,'correct_registrations',seq,v)
        #                 vol=v.split('_')[:-2]
        #                 vol='_'.join(vol)
        #                 vol=vol+'.nii.gz'
        #                 # print(v)
        #                 # print(vol)
        #                 # sys.exit()
        #                 LR=os.path.join(data_path,'derivatives',s,'volumes',seq,'volumes3D',vol)
        #                 seg=v.split('_')
        #                 seg.insert(-1,'segment')
        #                 seg='_'.join(seg)
        #                 segment=os.path.join(data_path,'derivatives',s,'correct_registrations',seq,seg)
        #                 if s[-3:] not in check_subjects:
        #                     check_subjects.append(s[-3:])
        #                 subject=tio.Subject(
        #                     subject_name=s[-3:],
        #                     LR_image=tio.ScalarImage(LR),
        #                     HR_image=tio.ScalarImage(HR),
        #                     label=tio.LabelMap(segment)
        #                 )
        #                 subjects.append(subject)
        #     else:
        #         pass

        check_subjects=[]
        data_path = '/mnt/Data/Equinus_BIDS_dataset/data_05/'
        out_channels = 1 #4
        #subject_names = ['E01','E02','E03','E05','E06','E08','T01','T02','T03','T04','T05','T06','T08']
        subject_names = ['E01','E02','E05','E08','T01','T02','T03','T05']
        bones=['calcaneus','talus','tibia']
        volume=['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14']
        if args.mean=="True":
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
    

    dataset = tio.SubjectsDataset(subjects)
    print('Dataset size:', len(check_subjects), 'subjects')
    prefix += '_subj_'+str(len(check_subjects))

    #############################################################################################################################################################################""
    # TRANSFORMATIONS
    if DATA_AUGMENTATION:
        #onehot = tio.OneHot()
        flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)
        bias = tio.RandomBiasField(coefficients = 0.5, p=0.5)
        noise = tio.RandomNoise(std=0.1, p=0.25)
        prefix += '_bias_flip_affine_noise_Nada_DEBUG_DES_ENFERS0'

        if data=='hcp':
            normalization = tio.ZNormalization(masking_method='label')
            #normalization = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
            spatial = tio.RandomAffine(scales=0.1,degrees=10, translation=0, p=0.75)

        if data=='equinus':
            normalization = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
            #spatial = tio.RandomAffine(scales=0.1,degrees=(20,0,0), translation=0, p=0.75)
            spatial = tio.OneOf({
                tio.RandomAffine(scales=0.1,degrees=(20,0,0), translation=0): 0.8,
                tio.RandomElasticDeformation(): 0.2,
            },
            p=0.75,
            )  

        transforms = [flip, spatial, bias, normalization, noise]
        #transforms = [flip, spatial, bias, normalization, noise, onehot]
        #transforms = [flip, normalization]#, noise]


        training_transform = tio.Compose(transforms)
        #validation_transform = tio.Compose([normalization, onehot])
        validation_transform = tio.Compose([normalization])   

    #############################################################################################################################################################################""
    # TRAIN AND VALIDATION SETS
    seed = 42  # for reproducibility

    num_subjects = len(check_subjects)
    num_training_subjects = int(training_split_ratio * num_subjects)
    num_validation_subjects = num_subjects - num_training_subjects

    num_split_subjects = num_training_subjects, num_validation_subjects
    #num_split_subjects = 7, 1
    training_sub, validation_sub = torch.utils.data.random_split(check_subjects, num_split_subjects)
    training_subjects=[]
    validation_subjects=[]
    train=[]
    validation=[]
    for s in subjects:
        if s.subject_name in training_sub:
            training_subjects.append(s)
            if s.subject_name not in train:
                train.append(s.subject_name)
        elif s.subject_name in validation_sub:
            validation_subjects.append(s)
            if s.subject_name not in validation:
                validation.append(s.subject_name)
        else:
            sys.exit("Problème de construction ensembles de train et validation")
    print('training = '+str(train))
    print('validation = '+str(validation))

    if DATA_AUGMENTATION:
        training_set = tio.SubjectsDataset(
        training_subjects, transform=training_transform)

        validation_set = tio.SubjectsDataset(
        validation_subjects, transform=validation_transform)
    else:
        training_set = tio.SubjectsDataset(
        training_subjects, transform=validation_transform)

        validation_set = tio.SubjectsDataset(
        validation_subjects, transform=validation_transform)

    print('Training set:', len(training_sub), 'subjects')
    print('Validation set:', len(validation_sub), 'subjects') 
    # test=validation_set[0]
    # print(test)
    # print(test['imageT1'][tio.DATA].shape)
    # sys.exit()


    #############################################################################################################################################################################""
    # PATCHES SETS
    print('num_workers : '+str(num_workers))

    #sampler = tio.data.UniformSampler(patch_size)

    probabilities = {0: 0, 1: 1}
    sampler = tio.data.LabelSampler(
      patch_size=patch_size,
      label_name='label',
      label_probabilities=probabilities
    )

    patches_training_set = tio.Queue(
        subjects_dataset=training_set,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=True,
        shuffle_patches=True,
    )

    patches_validation_set = tio.Queue(
        subjects_dataset=validation_set,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=sampler,
        num_workers=num_workers,
        shuffle_subjects=False,
        shuffle_patches=False,
    )

    training_loader_patches = torch.utils.data.DataLoader(
        patches_training_set, batch_size=training_batch_size, num_workers=0, pin_memory=False)
    print("Nombre de patches de train: " + str(len(training_loader_patches.dataset)))
    validation_loader_patches = torch.utils.data.DataLoader(
        patches_validation_set, batch_size=validation_batch_size, num_workers=0, pin_memory=False)
    print("Nombre de patches de test: " + str(len(validation_loader_patches.dataset)))

    #############################################################################################################################################################################""
    # NETS
    class Unet(nn.Module):
        def __init__(self, n_channels = 1, n_classes = 10, n_features = 32):
            super(Unet, self).__init__()

            self.n_channels = n_channels
            self.n_classes = n_classes
            self.n_features = n_features

            def double_conv(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )

            self.dc1 = double_conv(self.n_channels, self.n_features)
            self.dc2 = double_conv(self.n_features, self.n_features*2)
            self.dc3 = double_conv(self.n_features*2, self.n_features*4)
            self.dc4 = double_conv(self.n_features*6, self.n_features*2)
            self.dc5 = double_conv(self.n_features*3, self.n_features)
            self.mp = nn.MaxPool2d(2)

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #binlinear?

            self.out = nn.Conv2d(self.n_features, self.n_classes, kernel_size=1)

        def forward(self, x):
            x1 = self.dc1(x)

            x2 = self.mp(x1)
            x2 = self.dc2(x2)

            x3 = self.mp(x2)
            x3 = self.dc3(x3)

            x4 = self.up(x3)
            x4 = torch.cat([x4,x2], dim=1)
            x4 = self.dc4(x4)

            x5 = self.up(x4)
            x5 = torch.cat([x5,x1], dim=1)
            x5 = self.dc5(x5)
            return self.out(x5)



    class UNet(pl.LightningModule):
        def __init__(self, criterion, learning_rate, optimizer_class, dataset, n_channels = 1, n_classes = 1, n_features = 32):
            super().__init__()
            self.lr = learning_rate
            self.criterion = criterion
            self.optimizer_class = optimizer_class
            self.n_channels = n_channels
            self.n_classes = n_classes
            self.n_features = n_features
            self.dataset = dataset
            self.net = Unet(n_channels, n_classes, n_features)

        def forward(self, x):
            return(self.net(x))

        def configure_optimizers(self):
            optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
            return optimizer
        
        def prepare_batch(self, batch):
            if self.dataset=='hcp':
                return batch['imageT1'][tio.DATA], batch['imageT2'][tio.DATA], batch['label'][tio.DATA]
            elif self.dataset=='equinus':
                return batch['LR_image'][tio.DATA], batch['HR_image'][tio.DATA], batch['label'][tio.DATA]
        
        def infer_batch(self, batch):
            x, y, mask = self.prepare_batch(batch)
            x=x.squeeze(1)
            y=y.squeeze(1)
            # print(x.shape)
            # print(y.shape)
            # print(mask.shape)
            # sys.exit()
            # print(x.shape)
            # print(y.shape)
            # sys.exit()
            y_hat = self.net(x)
            return y_hat, y

        def training_step(self, batch, batch_idx):
            y_hat, y = self.infer_batch(batch)
            x, y, mask = self.prepare_batch(batch)
            x=x.squeeze(1)
            y=y.squeeze(1)
            # print(x.shape)
            # print(y.shape)
            # print(y_hat.shape)
            mask=torch.abs((mask-1)*(-1))
            if batch_idx%1000==0:
                # print(x.shape)
                # print(y.shape)
                # print(mask.shape)
                # sys.exit()
                plt.figure()
                plt.suptitle('epoch: '+str(self.current_epoch)+' batch_idx: '+str(batch_idx)+'     '+prefix)
                plt.subplot(1,3,1)
                plt.imshow(x[0,0,:,:].cpu().detach().numpy(), cmap="gray")
                plt.subplot(1,3,2)
                plt.imshow(y[0,0,:,:].cpu().detach().numpy(), cmap="gray")
                plt.subplot(1,3,3)
                plt.imshow(y_hat[0,0,:,:].cpu().detach().numpy().astype(float), cmap="gray")
                plt.colorbar()
                plt.savefig('/home/aorus-users/claire/Images_Test_UNet_HCP/'+'epoch-'+str(self.current_epoch)+'_batch_idx-'+str(batch_idx)+'.png')
                plt.close()
            loss = self.criterion(y_hat*mask, y*mask)
            if math.isnan(loss.item()):
                batch_size=x.shape[0]
                for i in range(batch_size):
                    plt.figure()
                    plt.suptitle('epoch: '+str(self.current_epoch)+' batch_idx: '+str(batch_idx)+'     '+prefix)
                    plt.subplot(1,3,1)
                    plt.imshow(x[0,0,:,:].cpu().detach().numpy())
                    plt.subplot(1,3,2)
                    plt.imshow(y[0,0,:,:].cpu().detach().numpy())
                    plt.subplot(1,3,3)
                    plt.imshow(y_hat[0,0,:,:].cpu().detach().numpy().astype(float))
                    plt.savefig('/home/aorus-users/claire/Images_Test_UNet_HCP/'+'epoch-'+str(self.current_epoch)+'_batch_idx-'+str(batch_idx)+'__'+str(i)+'_nan.png')
                    plt.close()
            if math.isinf(loss.item()):
                batch_size=x.shape[0]
                for i in range(batch_size):
                    plt.figure()
                    plt.suptitle('epoch: '+str(self.current_epoch)+' batch_idx: '+str(batch_idx)+'     '+prefix)
                    plt.subplot(1,3,1)
                    plt.imshow(x[0,0,:,:].cpu().detach().numpy())
                    plt.subplot(1,3,2)
                    plt.imshow(y[0,0,:,:].cpu().detach().numpy())
                    plt.subplot(1,3,3)
                    plt.imshow(y_hat[0,0,:,:].cpu().detach().numpy().astype(float))
                    plt.savefig('/home/aorus-users/claire/Images_Test_UNet_HCP/'+'epoch-'+str(self.current_epoch)+'_batch_idx-'+str(batch_idx)+'__'+str(i)+'_inf.png')
                    plt.close()
            self.log('train_loss', loss, prog_bar=True)
            return loss
        
        def validation_step(self, batch, batch_idx):
            y_hat, y = self.infer_batch(batch)
            loss = self.criterion(y_hat, y)
            self.log('val_loss', loss)
            return loss

        # def on_train_epoch_end(self):
        #     epoch=self.current_epoch

    if args.model is not None:
        UNet.load_state_dict(torch.load(args.model))
    #############################################################################################################################################################################""
    # EXEC
    if loss=='L1':
        L=nn.L1Loss()
    elif loss=='L2':
        L=nn.MSELoss()
    else:
        sys.exit("Loss incorecte")

    net = UNet(
        criterion=L,
        dataset=data,     
        learning_rate=1e-4,
        optimizer_class=torch.optim.AdamW,
        #optimizer_class=torch.optim.Adam,
        n_features = n_channels,
    )

    # checkpoint_callback = ModelCheckpoint(monitor = 'PSNR',
    #                                         save_top_k = 1,
    #                                         mode = 'max',
    #                                         dirpath = output_path,
    #                                         filename = 'Checkpoint_{epoch}-{PSNR:.2f}')

    #checkpoint_callback = ModelCheckpoint(filepath=output_path+'/'+ prefix+'_CHECKPOINT_{epoch:02d}')



    logger = TensorBoardLogger(save_dir = output_path, name = 'Test_logger',version=prefix)

    trainer = pl.Trainer(
        gpus=[gpu],
        max_epochs=num_epochs,
        progress_bar_refresh_rate=20,
        logger=logger,
        #callbacks = [checkpoint_callback],
        precision=16
    )
    trainer.fit(net, training_loader_patches, validation_loader_patches)
    trainer.save_checkpoint(output_path+prefix+'.ckpt')
    torch.save(net.state_dict(), output_path+prefix+'_torch.pt')

    print('Finished Training')

    ###############################################################################################################################################################
    #INFER
    #print('Inference')
    print('Saving images')
    I=validation_set[0]
    if data=='hcp':
        T1=I['imageT1']
        T2=I['imageT2']
        label=I['label']
        name=I['subject_name']
    else:
        T1=I['LR_image']
        T2=I['HR_image']
        label=I['label']
        name=I['subject_name']

    T1.save('/home/aorus-users/claire/T1_'+data+'_'+name+'_epochs-'+str(num_epochs)+'_nbrsubject-'+str(len(check_subjects))+'_sampling-'+str(samples_per_volume)+'.nii.gz')
    T2.save('/home/aorus-users/claire/T2_'+data+'_'+name+'_epochs-'+str(num_epochs)+'_nbrsubject-'+str(len(check_subjects))+'_sampling-'+str(samples_per_volume)+'.nii.gz')
    label.save('/home/aorus-users/claire/segmentation_'+data+'_'+name+'_epochs-'+str(num_epochs)+'_nbrsubject-'+str(len(check_subjects))+'_sampling-'+str(samples_per_volume)+'.nii.gz')
    
    # sub = tio.Subject(
    #     T1_image=T1
    #     )

    # patch_overlap = 16
    # patch_size = 64
    # batch_size = 2

    # grid_sampler = tio.inference.GridSampler(
    #     sub,
    #     patch_size,
    #     patch_overlap,
    #     )
    # patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
    # aggregator = tio.inference.GridAggregator(sampler=grid_sampler, overlap_mode='average')

    # net.eval()
    # with torch.no_grad():
    #     for patches_batch in patch_loader:
    #         input_tensor = patches_batch['T1_image'][tio.DATA]
    #         locations = patches_batch[tio.LOCATION]
    #         outputs = net(input_tensor)
    #         aggregator.add_batch(outputs, locations)

    # output_tensor = aggregator.get_output_tensor()

    # tmp = tio.ScalarImage(tensor=output_tensor, affine=sub.T1_image.affine)
    # sub.add_image(tmp, 'T2_image_estim')
    # print(sub)

    
    # output_seg = tio.ScalarImage(tensor=output_tensor, affine=sub['T1_image'].affine)
    # output_seg.save('/home/aorus-users/claire/Test_'+data+'_'+name+'_epochs-'+str(num_epochs)+'_nbrsubject-'+str(len(check_subjects))+'_sampling-'+str(samples_per_volume)+'.nii.gz')
