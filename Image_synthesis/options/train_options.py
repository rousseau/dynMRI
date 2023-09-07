import options.base_options as base_options
from os.path import expanduser
home = expanduser("~")

class TrainOptions(base_options.BaseOptions):
    '''
    Cette classe contient les options d'entraînement et celles définies dans BaseOptions
    '''

    def initialize(self, parser):
        parser = base_options.BaseOptions.initialize(self, parser)
        # Training
        parser.add_argument('--seed', help='Set to true to set the determinist mode', type=bool, required=False, default=True)
        parser.add_argument('-e', '--epochs', help='Max epochs', type=int, required=False, default = 150)
        parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 64)
        parser.add_argument('-g', '--gpu', help='Number of the GPU to use', type=int, required=False, default = 0)
        parser.add_argument('-w', '--workers', help='Number of workers (multiprocessing)', type=int, required=False, default = 16)
        parser.add_argument('-m', '--model', help='Pytorch initialization model', type=str, required=False)
        parser.add_argument('--learning_rate', help='Learning rate for training', type=float, required=False, default=1e-5)
        parser.add_argument('--random_size', help='Size of the random portion for the supplementary discriminator', type=int, required = False, default=0)
        parser.add_argument('-s', '--samples', help='Samples per volume', type=int, required=False, default = 150)
        parser.add_argument('-q', '--queue', help='Max queue length', type=int, required=False, default = 300)
        parser.add_argument('--sampler', help='Sampler to use (probabilities or uniform)', type=str, required=False, default = "Probabilities")
        # Saving
        parser.add_argument('--experiment_name', help='Insert in the file name', type=str, required=True)
        parser.add_argument('--saving_path', help='Path to the folder the experiment files', type=str, required = False, default=home)
        parser.add_argument('--saving_ratio', help='If you want to capture images during training, provide a number as an argument to indicate how often images are to be recorded based on batch_idx. Otherwise, set it to None.', type=int, required = False, default=None)
        parser.add_argument('--phase', help='Train, val, test', type=str, required = False, default="train")
        parser.add_argument('--segmentation', help='Use static segmentations as prior', type=str, required=False)


        subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")
        DRIT_parser = subparsers.add_parser("DRIT", help="DRIT mode parser")
        # Losses weights
        DRIT_parser.add_argument('--lambda_cyclic_anatomy', help='cyclic loss poderation on the anatomy latent space', type=int, required=False, default=0)
        DRIT_parser.add_argument('--lambda_cyclic_modality', help='cyclic loss ponderation on the anatomy latent space', type=int, required=False, default = 0)
        DRIT_parser.add_argument('--lambda_D_content', help='ponderaton on the adversarial loss applied on the anatomy latent space', type=int, required=False, default=1)
        DRIT_parser.add_argument('--lambda_D_domain', help='ponderation on the adversarial loss applied on the real and synthetised images in a particular domain', type=int, required=False, default=1)
        DRIT_parser.add_argument('--lambda_latent', help='', type=int, required=False, default=10)
        DRIT_parser.add_argument('--lambda_self', help='', type=int, required=False, default = 10)
        DRIT_parser.add_argument('--lambda_cross_cycle', help='', type=int, required=False, default = 10)
        DRIT_parser.add_argument('--lambda_Npair', help='contrastive loss ponderation on the anatomy latent space (N-pair loss)', type=int, required=False, default=0)
        DRIT_parser.add_argument('--lambda_contrastive_modality', help='contrastive loss ponderation on the modality latent (SupCon loss))', type=int, required=False, default=0)
        DRIT_parser.add_argument('--lambda_reg', help='', type=int, required=False, default = 0.01)
        DRIT_parser.add_argument('--lambda_adv_anatomy_encoder', help='', type=int, required=False, default = 1)
        DRIT_parser.add_argument('--lambda_adv_generator', help='', type=int, required=False, default = 1)
        DRIT_parser.add_argument('--lambda_style_loss', help='ponderation for style loss inspired from those used in arbitrary style transfer', type=int, required=False, default=0)
        DRIT_parser.add_argument('--lambda_content_loss', help='ponderation for content loss inspired from those used in arbitrary style transfer', type=int, required=False, default=0)

        
        
        #Degradation_parser.add_argument('--', help='', type=int, required=False, default=0)
        Degradation_parser = subparsers.add_parser("Degradation", help="Degradation mode parser")
        Degradation_parser.add_argument('--data_mode', help='(Pseudo) Paired or Unpaired setting', type=str, required=False, default='Paired')
        Degradation_parser.add_argument('--net', help='Unet or ResNet for paired synthesis or as generator for GAN', type=str, required=False, default='UNet')

        self.isTrain = True
        return(parser)
    
