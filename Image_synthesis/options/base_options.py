import argparse
import os
import sys
from datetime import datetime

class BaseOptions():
    '''
    Classe qui définit les options de base  utilisées à la fois pour l'entraînement et pour le test
    '''

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # model parameters 
        parser.add_argument('-n', '--network', help='Network to use (UNet, HighResNet, Disentangled, DenseNet..)', type=str, required=False, default = "Disentangled_plusplus")
        parser.add_argument('--use_reduce', help='for Disentangled_plusplus, set to True for using lighter architecture', type=bool,required=False, default=True)
        parser.add_argument('--dynamic', help='Use dynamic scheme', type=bool, required=False, default=False)
        parser.add_argument('--monomodal', help='Use monomodal dynamic scheme', type=bool, required=False, default=False)
        parser.add_argument('--vae', help='Use VAE for style', type=bool, required=False, default=False)
        parser.add_argument('--use_multiscale_discriminator', help='Set to True to use a multiscale discriminator', type=bool, required=False, default=False)
        parser.add_argument('--use_multiscale_content', help='Set to True to use a multiscale content', type= bool, required=False, default=False)
        parser.add_argument('--use_multiscale_style', help='Set to True to use multiscale style', type=bool, required=False, default=False)
        parser.add_argument('--method', help='Method to be used to merge style and content', type=str, required=False, default = "ADAIN2", choices=['ADAIN2', 'FILM2', 'DRITPP2', 'SPADE', 'Concat'])
        parser.add_argument('--modality_encoder', help='Architecture de l encodeur a utiliser. Si Resnet50, pretrained = True par défaut', type=str, required=False, default='DRITPP', choices=['DRITPP', 'ResNet50'])
        parser.add_argument('--anatomy_encoder', help='Architecture de l encodeur a utiliser. Si Resnet50, pretrained = True par défaut', type=str, required=False, default='DRITPP_reduceplus', choices=['DRITPP_reduceplus', 'ResNet50'])
        parser.add_argument('--use_segmentation_network', help='Return the segmentation extract fro mthe content', type=bool, required=False, default=False)
        #parser.add_argument('--activation_layer', help='Layer for getting activations for')

        # dataset parameters
        parser.add_argument('-d', '--data', help='Input dataset', type=str, required=False, default = 'equinus_256')
        parser.add_argument('--dynamic_path', help='Path to the dynamic images folder (Equinus)', type=str, required=False, default = None)
        parser.add_argument('--static_path', help='Path to the dynamic images folder (Equinus)', type=str, required=False, default = None)
        parser.add_argument('-p', '--patch_size', help='Patch size', nargs='+', type=int, required=False, default = (64,64,1))


        self.initialized = True
        return parser
    
    def print_and_save(self, opt):
        to_save = ''
        text = ''
        text += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            text += '{:>30}: {:<1}{}\n'.format(str(k), str(v), comment)
            to_save += str(k)+' '+str(v) + ' ' + comment + '\n'
        text += '----------------- End -------------------'
        print(text)

        if self.isTrain:
            recording_path = os.path.join(opt.saving_path, datetime.now().strftime("%d-%m-%Y_%H-%M-%S")+'_'+opt.experiment_name)
            file_name=os.path.join(recording_path, 'training' + "_opt.txt")
        else:
            recording_path = os.path.join(opt.model)
            file_name=os.path.join(recording_path, 'testing' + "_opt.txt")
        setattr(opt, 'saving_path', recording_path+'/')

        if not os.path.exists(recording_path):
            os.mkdir(recording_path)
        with open(file_name, 'wt') as opt_file:
            opt_file.write(to_save)
            opt_file.write('\n')

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            
        opt, _ = parser.parse_known_args()

        self.parser = parser
        return parser.parse_args()



    def parse(self):
        opt = self.gather_options()
        #opt.isTrain = self.isTrain

        self.print_and_save(opt)

        self.opt = opt
        return(self.opt)

