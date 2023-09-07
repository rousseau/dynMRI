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
    
    # parser.add_argument('-i', '--input', help='Input image', type=str, required=True)
    # parser.add_argument('-m', '--model', help='Pytorch model', type=str, required=True)
    # parser.add_argument('-o', '--output', help='Output image', type=str, required=True)
    # parser.add_argument('-F', '--fuzzy', help='Output fuzzy image', type=str, required=False)
    # parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = (64,64,1))
    # parser.add_argument('--patch_overlap', help='Patch overlap', type=int, required=False, default = (0,0,0))
    # parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 2)
    # parser.add_argument('-g', '--ground_truth', help='Ground truth for metric computation', type=str, required=False)
    # parser.add_argument('-T', '--test_time', help='Number of inferences for test-time augmentation', type=int, required=False, default=1)
    # parser.add_argument('-c', '--channels', help='Number of channels', type=int, required=False, default=16)
    # parser.add_argument('-f', '--features', help='Number of features', type=int, required=False, default=64)
    # parser.add_argument('--classes', help='Number of classes', type=int, required=False, default=1)
    # parser.add_argument('-s', '--scales', help='Scaling factor (test-time augmentation)', type=float, required=False, default=0.05)
    # parser.add_argument('-d', '--degrees', help='Rotation degrees (test-time augmentation)', type=int, required=False, default=10)
    # parser.add_argument('-D', '--dataset', help='Name of the dataset used', type=str, required=False, default='hcp')
    # parser.add_argument('-t', '--test_image', help='Image test (skip inference and goes directly to PSNR)', type=str, required=False)
    # parser.add_argument('-S', '--segmentation', help='Segmentation to use', type=str,required=False)
    # parser.add_argument('-n', '--network', help='Network to use (UNet, ResNet, disentangled..)', type=str,required=True)
    # parser.add_argument('--gpu', help='GPU to use (If None, goes on CPU)', type=int,required=False)
    # parser.add_argument('--diff', help='Save or not the difference between the ground truth and input or recostruct image', type=bool,required=False)
    # parser.add_argument('--seg', help='Use static segmentations as prior', type=str, required=False)
    # parser.add_argument('--segmentation_os', help='Segmentation to use', type=str,required=False)
    # parser.add_argument('--mode', help='Mode to use (for DRIT or CycleGAN): reconstruction or degradation', type=str,required=False)
    # parser.add_argument('--whole', help='set to True for using the whole image for testing', type=bool,required=False)
    # parser.add_argument('--use_reduce', help='for Disentangled_plusplus, set to True for using light architecture', type=bool,required=False, default=False)
    # parser.add_argument('--use_multiscale_discriminator', help='Set to True to use a multiscale discriminator', type=bool, required=False, default=False)
    # parser.add_argument('--use_multiscale_content', help='Set to True to use a multiscale content', type= bool, required=False, default=False)
    # parser.add_argument('--use_multiscale_style', help='Set to True to use multiscale style', type=bool, required=False, default=False)
    # parser.add_argument('--dynamic', help='Use dynamic scheme', type=bool, required=False, default=False)
    # parser.add_argument('--use_segmentation_network', help='Return the segmentation extract fro mthe content', type=bool, required=False, default=False)
    # parser.add_argument('--method', help='Mathod to use in conditionnal info', type=str, required=False, default = None)

    
        # # Training parameters
        # parser.add_argument('-i', '--input', help='Input image', type=str, required=True)
        # parser.add_argument('-m', '--model', help='Pytorch model', type=str, required=True)
        # parser.add_argument('-o', '--output', help='Output image', type=str, required=True)
        # parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = (64,64,1))
        # parser.add_argument('--patch_overlap', help='Patch overlap', type=int, required=False, default = (0,0,0))
        # parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 2)
        # parser.add_argument('-g', '--ground_truth', help='Ground truth for metric computation', type=str, required=False)
        # parser.add_argument('-p', '--patch_size', help='Patch size', type=int, required=False, default = (64,64,1))
        # parser.add_argument('--patch_overlap', help='Patch overlap', type=int, required=False, default = (0,0,0))
        # parser.add_argument('-b', '--batch_size', help='Batch size', type=int, required=False, default = 2)
        # parser.add_argument('-n', '--network', help='Network to use (UNet, ResNet, disentangled..)', type=str,required=True)
        # parser.add_argument('-D', '--dataset', help='Name of the dataset used', type=str, required=False, default='hcp')
        # parser.add_argument('--gpu', help='GPU to use (If None, goes on CPU)', type=int,required=False)
        # parser.add_argument('--seg', help='Use static segmentations as prior', type=str, required=False)
        # parser.add_argument('--mode', help='Mode to use (for DRIT or CycleGAN): reconstruction or degradation', type=str,required=False)
        # parser.add_argument('--use_reduce', help='for Disentangled_plusplus, set to True for using light architecture', type=bool,required=False, default=False)
        # parser.add_argument('--use_multiscale_discriminator', help='Set to True to use a multiscale discriminator', type=bool, required=False, default=False)
        # parser.add_argument('--use_multiscale_content', help='Set to True to use a multiscale content', type= bool, required=False, default=False)
        # parser.add_argument('--use_multiscale_style', help='Set to True to use multiscale style', type=bool, required=False, default=False)
        # parser.add_argument('--dynamic', help='Use dynamic scheme', type=bool, required=False, default=False)
        # parser.add_argument('--use_segmentation_network', help='Return the segmentation extract fro mthe content', type=bool, required=False, default=False)
        # parser.add_argument('--method', help='Mathod to use in conditionnal info', type=str, required=False, default = None)

        # self.isTrain= False
        # return parser
