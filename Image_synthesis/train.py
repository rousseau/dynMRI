from os.path import expanduser
from posix import listdir
from numpy import NaN, dtype
from torchio.data import image
from torchio.utils import check_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
import sys
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, tensorboard
import argparse
from DRITPP_2 import DRIT 
from Degradation_nets import Degradation_paired, Degradation_unpaired
from options.train_options import TrainOptions
from load_data import load_data
home = expanduser("~")

def normalize(tensor):
    for i in range(tensor.shape[1]):
        mean=tensor[0,i,:,:].mean()
        std=tensor[0,i,:,:].std()
        tensor[0,i,:,:]=(tensor[0,i,:,:]-mean)/std
    return(tensor)


if __name__ == '__main__':
    print("Nombre de GPUs detectes: "+ str(torch.cuda.device_count()))
    args = TrainOptions().parse()

    if args.seed:
        print('Set seed: '+str(args.seed_value))
        seed_everything(args.seed_value, workers=True)

    training_split_ratio = 0.9  # use 90% of samples for training, 10% for testing
    num_epochs = args.epochs
    gpu = args.gpu
    num_workers = args.workers#multiprocessing.cpu_count()*
    network=args.network
    sampler=args.sampler
    training_batch_size = args.batch_size
    validation_batch_size = args.batch_size
    patch_size = args.patch_size
    print(patch_size)
    samples_per_volume = args.samples
    max_queue_length = args.queue
    
    data = args.data 
    

    prefix = network+'_'
    if args.experiment_name is not None:
        prefix += args.experiment_name + '_'
    prefix += data
    prefix += '_epochs_'+str(num_epochs)
    prefix += '_patches_'+str(patch_size)

    if args.model is not None:
        prefix += '_using_init'
    
    output_path = args.saving_path
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    subjects=[]

    #############################################################################################################################################################################""
    # DATASET
    if data != 'MNIST':
        if args.dynamic_path!=None and args.static_path!=None:
            subjects, check_subjects = load_data(data='custom', segmentation=args.segmentation, batch_size=training_batch_size, dynamic_path = args.dynamic_path, static_path = args.static_path, seg_path=args.seg_path)
        else:
            subjects, check_subjects = load_data(data=data, segmentation=args.segmentation, batch_size=training_batch_size)

        dataset = tio.SubjectsDataset(subjects)
        print('Dataset size:', len(check_subjects), 'subjects')
        prefix += '_subj_'+str(len(check_subjects))+'_images_'+str(len(subjects))

        #############################################################################################################################################################################""
        # TRANSFORMATIONS
        flip = tio.RandomFlip(axes=('LR',), flip_probability=0.5)
        bias = tio.RandomBiasField(coefficients = 0.5, p=0.5)
        noise = tio.RandomNoise(std=0.1, p=0.25)
        if data == 'equinus_simulate':
            prefix += '_bs_flp_afn_nse_VERSION_'
        else:
            pass

        if (data=='dhcp_2mm' or data=='dhcp_1mm' or data=='dhcp_original'):
            normalization = tio.ZNormalization(masking_method='label')
            spatial = tio.RandomAffine(scales=0.1,degrees=10, translation=0, p=0.75)

        else:
            normalization = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
            spatial = tio.RandomAffine(scales=0.1,degrees=(20,0,0), translation=0, p=0.75)
        
        if args.method == 'ddpm':
            normalization = tio.RescaleIntensity()

        transforms = [flip, spatial, bias, normalization, noise]

        training_transform = tio.Compose(transforms) 
        validation_transform = tio.Compose([normalization])   



        #############################################################################################################################################################################""
        # TRAIN AND VALIDATION SETS
        if (data=="equinus_256" or data=="bone_segmentation_equinus_256" or data=="segmentation_equinus_256" or data=='equinus_256_boneseg' or data=='custom'):
            training_sub=['sub_E05', 'sub_T03', 'sub_T02', 'sub_T05', 'sub_T01', 'sub_T04', 'sub_E02', 'sub_E01', 'sub_T08', 'sub_E08', 'sub_E06']
            validation_sub=['sub_T06']
            test_sub=['sub_E03']
        elif data=="simulate_256":
            training_sub=['sub_E09', 'sub_E05', 'sub_T03', 'sub_T02', 'sub_E08', 'sub_E13', 'sub_T05', 'sub_E10', 'sub_T01', 'sub_T07', 'sub_T04', 'sub_E02', 'sub_E03', 'sub_E01', 'sub_T08', 'sub_T09']
            validation_sub=['sub_T06', 'sub_E06']
            test_sub=None
        elif data=='dhcp_2mm' or data=='dhcp_1mm' or data=='dhcp_original' or data=='dhcp_1mm_npair':
            training_sub=['CC00131XX08', 'CC00284BN13', 'CC00257XX10', 'CC00150BN02', 'CC00648XX22', 'CC00482XX13', 'CC00245BN15', 'CC00265XX10', 'CC00367XX13', 'CC00071XX06', 'CC00065XX08', 'CC00628XX18', 'CC00182XX10', 'CC00172AN08', 'CC00185XX13', 'CC00216AN10', 'CC00799XX25', 'CC00154XX06', 'CC00091XX10', 'CC00400XX04', 'CC00284AN13', 'CC00363XX09', 'CC00798XX24', 'CC00301XX04', 'CC00527XX16', 'CC00499XX22', 'CC00657XX14', 'CC00231XX09', 'CC00366XX12', 'CC00770XX12', 'CC00489XX20', 'CC00797XX23', 'CC00371XX09', 'CC00810XX10', 'CC00771XX13', 'CC00686XX19', 'CC00338AN17', 'CC00326XX13', 'CC00823XX15', 'CC00446XX18', 'CC00206XX08', 'CC00412XX08', 'CC00272XX09', 'CC00549XX22', 'CC00723XX14', 'CC00863XX14', 'CC00168XX12', 'CC00300XX03', 'CC00161XX05', 'CC00216BN10', 'CC00074XX09', 'CC00218BN12', 'CC00171XX07', 'CC00621XX11', 'CC00114XX07', 'CC00448XX20', 'CC00199XX19', 'CC00653XX10', 'CC00529BN18', 'CC00250XX03', 'CC00815XX15', 'CC00183XX11', 'CC00613XX11', 'CC00176XX12', 'CC00760XX10', 'CC00293BN14', 'CC00840XX16', 'CC00270XX07', 'CC00871XX14', 'CC00422XX10', 'CC00361XX07', 'CC00654XX11', 'CC00068XX11', 'CC00768XX18', 'CC00099AN18', 'CC00298XX19', 'CC00247XX17', 'CC00150AN02', 'CC00418AN14', 'CC00788XX22', 'CC00465XX12', 'CC00201XX03', 'CC00765XX15', 'CC00528XX17', 'CC00115XX08', 'CC00368XX14', 'CC00163XX07', 'CC00550XX06', 'CC00271XX08', 'CC00104XX05', 'CC00403XX07', 'CC00581XX13', 'CC00316XX11', 'CC00245AN15', 'CC00268XX13', 'CC00414XX10', 'CC00672AN13', 'CC00124XX09', 'CC00417XX13', 'CC00852XX11', 'CC00116XX09', 'CC00672BN13', 'CC00830XX14', 'CC00556XX12', 'CC00445XX17', 'CC00418BN14', 'CC00406XX10', 'CC00431XX11', 'CC00517XX14', 'CC00106XX07', 'CC00344XX15', 'CC00180XX08', 'CC00193XX13', 'CC00162XX06', 'CC00571AN11', 'CC00703XX10', 'CC00853XX12', 'CC00548XX21', 'CC00576XX16', 'CC00143BN12', 'CC00845AN21', 'CC00207XX09', 'CC00384XX14', 'CC00616XX14', 'CC00595XX19', 'CC00319XX14', 'CC00652XX09', 'CC00305XX08', 'CC00209XX11', 'CC00398XX20', 'CC00520XX09', 'CC00480XX11', 'CC00532XX13', 'CC00329XX16', 'CC00889BN24', 'CC00618XX16', 'CC00413XX09', 'CC00157XX09', 'CC00617XX15', 'CC00492AN15', 'CC00186BN14', 'CC00409XX13', 'CC00158XX10', 'CC00530XX11', 'CC00302XX05', 'CC00764AN14', 'CC00529AN18', 'CC00202XX04', 'CC00153XX05', 'CC00434AN14', 'CC00238BN16', 'CC00639XX21', 'CC00135AN12', 'CC00478XX17', 'CC00160XX04', 'CC00332XX11', 'CC00178XX14', 'CC00377XX15', 'CC00306XX09', 'CC00066XX09', 'CC00356XX10', 'CC00513XX10', 'CC00252XX05', 'CC00402XX06', 'CC00879XX22', 'CC00566XX14', 'CC00669XX18', 'CC00552XX08', 'CC00080XX07', 'CC00341XX12', 'CC00589XX21', 'CC00547XX20', 'CC00260XX05', 'CC00447XX19', 'CC00484XX15', 'CC00389XX19', 'CC00845BN21', 'CC00238AN16', 'CC00805XX13', 'CC00421AN09', 'CC00191XX11', 'CC00219XX13', 'CC00450XX05', 'CC00583XX15', 'CC00582XX14', 'CC00107XX08', 'CC00753XX11', 'CC00348XX19', 'CC00801XX09', 'CC00593XX17', 'CC00822XX14', 'CC00451XX06', 'CC00632XX14', 'CC00129AN14', 'CC00526XX15', 'CC00164XX08', 'CC00587XX19', 'CC00555XX11', 'CC00350XX04', 'CC00695XX20', 'CC00096XX15', 'CC00194XX14', 'CC00062XX05', 'CC00580XX12', 'CC00424XX12', 'CC00143AN12', 'CC00705XX12', 'CC00111XX04', 'CC00735XX18', 'CC00248XX18', 'CC00764BN14', 'CC00303XX06', 'CC00357XX11', 'CC00236XX14', 'CC00502XX07', 'CC00568XX16', 'CC00596XX20', 'CC00281AN10', 'CC00121XX06', 'CC00458XX13', 'CC00649XX23', 'CC00293AN14', 'CC00060XX03', 'CC00466BN13', 'CC00152AN04', 'CC00441XX13', 'CC00569XX17', 'CC00702AN09', 'CC00102XX03', 'CC00483XX14', 'CC00664XX13', 'CC00846XX22', 'CC00078XX13', 'CC00385XX15', 'CC00117XX10', 'CC00395XX17', 'CC00829XX21', 'CC00132XX09', 'CC00719XX18', 'CC00675XX16', 'CC00094BN13', 'CC00500XX05', 'CC00135BN12', 'CC00108XX09', 'CC00254XX07', 'CC00184XX12', 'CC00255XX08', 'CC00757XX15', 'CC00370XX08', 'CC00545XX18', 'CC00818XX18', 'CC00146XX15', 'CC00120XX05', 'CC00086XX13', 'CC00740XX15', 'CC00442XX14', 'CC00792XX18', 'CC00147XX16', 'CC00334XX13', 'CC00267XX12', 'CC00594XX18', 'CC00843XX19', 'CC00670XX11', 'CC00177XX13', 'CC00734XX17', 'CC00588XX20', 'CC00338BN17', 'CC00693XX18', 'CC00536XX17', 'CC00469XX16', 'CC00138XX15', 'CC00113XX06', 'CC00100XX01', 'CC00204XX06', 'CC00349XX20', 'CC00508XX13', 'CC00314XX09', 'CC00073XX08', 'CC00089XX16', 'CC00379XX17', 'CC00197XX17', 'CC00838XX22', 'CC00308XX11', 'CC00486XX17', 'CC00629XX19', 'CC00656XX13', 'CC00172BN08', 'CC00457XX12', 'CC00145XX14', 'CC00227XX13', 'CC00620XX10', 'CC00082XX09', 'CC00476XX15', 'CC00650XX07', 'CC00586XX18', 'CC00101XX02', 'CC00347XX18', 'CC00492BN15', 'CC00383XX13', 'CC00083XX10', 'CC00777XX19', 'CC00198XX18', 'CC00787XX21', 'CC00313XX08', 'CC00337XX16', 'CC00423XX11', 'CC00592XX16', 'CC00689XX22', 'CC00130XX07', 'CC00144XX13', 'CC00597XX21', 'CC00518XX15', 'CC00136AN13', 'CC00833XX17', 'CC00791XX17', 'CC00440XX12', 'CC00860XX11', 'CC00850XX09', 'CC00063AN06', 'CC00731XX14', 'CC00720XX11', 'CC00534XX15', 'CC00181XX09', 'CC00122XX07', 'CC00221XX07', 'CC00544XX17', 'CC00627XX17', 'CC00364XX10', 'CC00094AN13', 'CC00585XX17', 'CC00540XX13', 'CC00907XX16', 'CC00438XX18', 'CC00258XX11', 'CC00079XX14', 'CC00376XX14', 'CC00512XX09', 'CC00501XX06', 'CC00461XX08', 'CC00507XX12', 'CC00428XX16', 'CC00165XX09', 'CC00200XX02', 'CC00292XX13', 'CC00497XX20', 'CC00622XX12', 'CC00590XX14', 'CC00577XX17', 'CC00553XX09', 'CC00453XX08', 'CC00858XX17', 'CC00766XX16', 'CC00405XX09', 'CC00119XX12', 'CC00244XX14', 'CC00174XX10', 'CC00237XX15', 'CC00584XX16', 'CC00687XX20', 'CC00218AN12', 'CC00647XX21', 'CC00416XX12', 'CC00505XX10', 'CC00134XX11', 'CC00286XX15', 'CC00473XX12', 'CC00578AN18', 'CC00466AN13', 'CC00289XX18', 'CC00498XX21', 'CC00378XX16', 'CC00481XX12', 'CC00754AN12', 'CC00516XX13', 'CC00186AN14', 'CC00824XX16', 'CC00217XX11', 'CC00342XX13', 'CC00304XX07', 'CC00479XX18', 'CC00468XX15', 'CC00570XX10', 'CC00504XX09', 'CC00223XX09', 'CC00397XX19', 'CC00433XX13', 'CC00421BN09', 'CC00126XX11', 'CC00542XX15', 'CC00352XX06', 'CC00069XX12', 'CC00088XX15', 'CC00307XX10', 'CC00546XX19', 'CC00149XX18', 'CC00572CN12', 'CC00109XX10', 'CC00203XX05', 'CC00320XX07', 'CC00607XX13']
            test_sub=['CC00563XX11', 'CC00439XX19', 'CC00099BN18', 'CC00470XX09', 'CC00688XX21', 'CC00382XX12', 'CC00408XX12', 'CC00572BN12', 'CC00472XX11', 'CC00667XX16', 'CC00339XX18', 'CC00269XX14', 'CC00455XX10', 'CC00747XX22', 'CC00754BN12', 'CC00407AN11', 'CC00685XX18', 'CC00084XX11', 'CC00525XX14', 'CC00087AN14', 'CC00443XX15', 'CC00205XX07', 'CC00474XX13', 'CC00351XX05', 'CC00454XX09', 'CC00179XX15', 'CC00127XX12', 'CC00744XX19', 'CC00600XX06', 'CC00170XX06', 'CC00324XX11', 'CC00087BN14', 'CC00407BN11', 'CC00362XX08', 'CC00562XX10', 'CC00564XX12', 'CC00067XX10', 'CC00343XX14', 'CC00661XX10', 'CC00189XX17', 'CC00558XX14', 'CC00411XX07', 'CC00129BN14', 'CC00251XX04', 'CC00309BN12']
            validation_sub=['CC00075XX10', 'CC00192AN12', 'CC00785XX19', 'CC00415XX11']
        else:
            sys.exit("REPARTITION OF THE SUBJECTS NOT ALREADY IMPLEMENT FOR THIS TYPE OF DATASET")
        training_subjects=[]
        validation_subjects=[]
        train=[]
        validation=[]
        test=[]
        test_subjects=[]
        for s in subjects:
            if s.subject_name in training_sub:
                training_subjects.append(s)
                if s.subject_name not in train:
                    train.append(s.subject_name)
            elif s.subject_name in validation_sub:
                validation_subjects.append(s)
                if s.subject_name not in validation:
                    validation.append(s.subject_name)
            elif (test_sub is not None):
                if s.subject_name in test_sub:
                    test_subjects.append(s)
                    if s.subject_name not in test:
                        test.append(s.subject_name)
            else:
                sys.exit("Problème de construction ensembles de train et validation")
        print('training = '+str(train))
        print('validation = '+str(validation))
        print('test = '+str(test))

        training_set = tio.SubjectsDataset(
        training_subjects, transform=training_transform)

        validation_set = tio.SubjectsDataset(
        validation_subjects, transform=validation_transform)

        test_set = tio.SubjectsDataset(
        test_subjects, transform=validation_transform)


        print('Training set:', len(training_sub), 'subjects')
        print('Validation set:', len(validation_sub), 'subjects') 
        if data!='dhcp_2mm' and data!='dhcp_1mm' and data!='dhcp_original' and data!='dhcp_1mm_npair':
            prefix=prefix+'_validation_'+str(validation)


        #############################################################################################################################################################################""
        # PATCHES SETS
        print('num_workers : '+str(num_workers))
        prefix += '_sampler_'+sampler
        if sampler=='Probabilities':
            probabilities = {0: 0, 1: 1}
            sampler = tio.data.LabelSampler(
                patch_size=patch_size,
                label_name='label',
                label_probabilities=probabilities
            )
        elif sampler=='Uniform':
            sampler = tio.data.UniformSampler(patch_size)
        else:
            sys.exit('Select a correct sampler')
        
        if args.seed:
            patches_training_set = tio.Queue(
                subjects_dataset=training_set,
                max_length=max_queue_length,
                samples_per_volume=samples_per_volume,
                sampler=sampler,
                num_workers=num_workers,
                shuffle_subjects=False,
                shuffle_patches=False,
                start_background=True,##
            )

            patches_validation_set = tio.Queue(
                subjects_dataset=validation_set,
                max_length=max_queue_length,
                samples_per_volume=samples_per_volume,
                sampler=sampler,
                num_workers=num_workers,
                shuffle_subjects=False,
                shuffle_patches=False,
                start_background=True,##
            )
        else:
            patches_training_set = tio.Queue(
                subjects_dataset=training_set,
                max_length=max_queue_length,
                samples_per_volume=samples_per_volume,
                sampler=sampler,
                num_workers=num_workers,
                shuffle_subjects=True,
                shuffle_patches=True,
                start_background=True,##
            )

            patches_validation_set = tio.Queue(
                subjects_dataset=validation_set,
                max_length=max_queue_length,
                samples_per_volume=samples_per_volume,
                sampler=sampler,
                num_workers=num_workers,
                shuffle_subjects=False,
                shuffle_patches=False,
                start_background=True,##
            )

        training_loader_patches = torch.utils.data.DataLoader(
            patches_training_set, batch_size=training_batch_size, num_workers=0, pin_memory=False)
        print("Nombre de patches de train: " + str(len(training_loader_patches.dataset)))
        validation_loader_patches = torch.utils.data.DataLoader(
            patches_validation_set, batch_size=validation_batch_size, num_workers=0, pin_memory=False)
        print("Nombre de patches de test: " + str(len(validation_loader_patches.dataset)))

    else:
        training_loader_patches, validation_loader_patches = load_data(data=data, segmentation=args.segmentation, batch_size=training_batch_size)
    #############################################################################################################################################################################""
    print('')
    if args.subcommand == "DRIT":
        if (args.method != 'ddpm' and args.model is None):
            net = DRIT(
                prefix = prefix,
                opt = args,
                isTrain=True
            )
            print('RÉSEAU UTILISÉ: '+args.method)
        elif (args.method != 'ddpm' and args.model is not None):
            net = DRIT(
                opt=args,
                prefix=prefix,
                isTrain=True
            )
            print('RÉSEAU UTILISÉ: '+args.method+" AVEC INITIALISATION")
        print('')
    elif args.subcommand == 'Degradation':
        if args.data_mode == 'Paired':
            net = Degradation_paired(
                prefix=prefix,
                opt = args
            )
        elif args.data_mode == 'Unpaired':
            net = Degradation_unpaired(
                prefix=prefix,
                opt = args
            )

    if args.model is not None:
        pretrained_dict = torch.load(args.model)
        model_dict = net.state_dict()
        if model_dict.keys() == pretrained_dict.keys():
            print('Réseau prétrained entièrement chargé dans le modèle')
            net.load_state_dict(pretrained_dict)
        else:
            print("Les paramètres du réseau pretrained différent de ceux dans lequel je load: loading uniquement des paramètres correspondants")
            DICT = {k:(pretrained_dict[k] if (k in pretrained_dict.keys()) else model_dict[k]) for k in model_dict.keys()}
            net.load_state_dict(DICT)

    checkpoint_callback = ModelCheckpoint(dirpath=output_path)

    logger = TensorBoardLogger(save_dir = output_path, name = 'Test_logger',version=prefix)

    if gpu >= 0:
        device = 'gpu'
    else:
        device = 'cpu'

    n_gpus = torch.cuda.device_count()

    trainer_args = {
        'accelerator': 'gpu',
        'max_epochs' : num_epochs,
        'logger' : logger
    }
    
    if n_gpus > 1:
        trainer_args['strategy']=DDPStrategy(find_unused_parameters=True)

    if args.seed:
        trainer_args['deterministic']='warn'
    
    trainer = pl.Trainer(**trainer_args)
    # compiled_net = torch.compile(net)

    trainer.fit(net, training_loader_patches, validation_loader_patches)
    trainer.save_checkpoint(output_path+prefix+'.ckpt')
    torch.save(net.state_dict(), output_path+prefix+'_torch.pt')

    print('Finished Training')
