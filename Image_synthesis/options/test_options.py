import options.base_options as base_options

class TestOptions(base_options.BaseOptions):
    '''
    Cette classe contient les options d'entraînement et celles définies dans BaseOptions
    '''

    def initialize(self, parser):
        parser = base_options.BaseOptions.initialize(self, parser)
        parser.add_argument('-i', '--input', help='Input image', type=str, required=True)
        parser.add_argument('-m', '--model', help='Path to the model folder', type=str, required=True)
        parser.add_argument('-o', '--output', help='Output image', type=str, required=True)
        #parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = (64,64,1))
        parser.add_argument('--patch_overlap', help='Patch overlap', type=int, required=False, default = (0,0,0))
        parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 2)
        parser.add_argument('-g', '--ground_truth', help='Ground truth for metric computation', type=str, required=False)
        parser.add_argument('-D', '--dataset', help='Name of the dataset used', type=str, required=False, default='hcp')
        parser.add_argument('--gpu', help='GPU to use (If None, goes on CPU)', type=int,required=False, default=0)
        parser.add_argument('--seg', help='Use static segmentations as prior', type=str, required=False)

        subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

        DRIT_parser = subparsers.add_parser("DRIT", help="DRIT mode parser")
        DRIT_parser.add_argument('--mode', help='Mode to use (for DRIT or CycleGAN): reconstruction or degradation', type=str,required=False, default='reconstruction')
        DRIT_parser.add_argument('--use_reduce', help='for Disentangled_plusplus, set to True for using light architecture', type=bool,required=False, default=True)
        DRIT_parser.add_argument('--latents', help='Get latent variables', type=bool,required=False, default=False)
        
        Degradation_parser = subparsers.add_parser("Degradation", help="Degradation mode parser")
        Degradation_parser.add_argument('--data_mode', help='(Pseudo) Paired or Unpaired setting', type=str, required=False, default='Paired')
        Degradation_parser.add_argument('--net', help='Unet or ResNet for paired synthesis or as generator for GAN', type=str, required=False, default='UNet')
        

        self.isTrain= False
        return parser
