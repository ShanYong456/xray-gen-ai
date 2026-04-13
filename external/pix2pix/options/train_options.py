from email import parser

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        # FID during training
        parser.add_argument("--fid_during_training", action="store_true",
                            help="Run FID evaluation during training.")
        parser.add_argument("--fid_epoch_freq", type=int, default=5,
                            help="Run FID every N epochs.")
        parser.add_argument("--fid_phase", type=str, default="val",
                            help="Dataset phase to use for FID. Usually val.")
        parser.add_argument("--fid_max_images", type=int, default=200,
                            help="Max number of image pairs for FID.")
        parser.add_argument("--fid_work_dir", type=str, default="fid_eval_runs",
                            help="Where temporary/generated FID files are stored.")
        parser.add_argument("--fid_keep_images", action="store_true",
                            help="Keep generated fake/real images used for FID.")
        parser.add_argument("--fid_debug_every", type=int, default=10,
                            help="Save one debug triplet every N FID samples.")
        
        # --- Separate learning rates for G and D (optional) ---
        parser.add_argument(
            "--lr_G",
            type=float,
            default=None,
            help="Generator learning rate. If not set, uses --lr.",
        )
        parser.add_argument(
            "--lr_D",
            type=float,
            default=None,
            help="Discriminator learning rate. If not set, uses --lr. Tip: try lr_D = 0.5 * lr_G to stop D dominating.",
        )

        self.isTrain = True
        return parser
