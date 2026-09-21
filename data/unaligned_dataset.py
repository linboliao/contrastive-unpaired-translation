import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from data.stain_transforms import build_stain_pipeline
from PIL import Image
import random
import util.util as util


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Stain normalization / HED augmentation options. See
        data/stain_transforms.py. Defaults are no-ops, so omitting all of them
        reproduces the pre-existing behaviour byte for byte."""
        parser.add_argument('--stain_norm', type=str, default='none', choices=['none', 'reinhard', 'macenko'],
                            help='stain normalization; preprocessing, applied at train AND test time. '
                                 'reinhard cuts cross-slide colour variance 64% on this data, macenko 0% '
                                 '-- see data/stain_transforms.ReinhardNormalizer')
        parser.add_argument('--stain_norm_domains', type=str, default='A',
                            help="which domains to normalize: 'A' (HE input), 'B' (IHC target), or 'AB'")
        parser.add_argument('--stain_ref_A', type=str, default='',
                            help='reference image or directory of images defining the HE stain basis')
        parser.add_argument('--stain_ref_B', type=str, default='',
                            help='reference image or directory for the IHC basis (only used if B is normalized)')
        parser.add_argument('--stain_ref_n', type=int, default=24,
                            help='if --stain_ref_* is a directory, pool this many images to fit the reference')
        parser.add_argument('--stain_tissue_only', type=int, default=0, choices=[0, 1],
                            help='fit/apply the reinhard moments over tissue pixels only. Measured worse than '
                                 'the default (all pixels) -- see data/stain_transforms.ReinhardNormalizer')
        parser.add_argument('--hed_aug', type=float, default=0.0,
                            help='HED augmentation: per-stain multiplicative jitter sigma; 0 disables. Train '
                                 'only. 0.05 is calibrated for this data, 0.10 is the ceiling; 0.20+ leaves '
                                 'the H&E colour manifold')
        parser.add_argument('--hed_aug_bias', type=float, default=0.0,
                            help='HED augmentation: per-stain additive jitter, as a FRACTION of each channel '
                                 'std (not an absolute OD). Conventionally set equal to --hed_aug')
        parser.add_argument('--hed_aug_domains', type=str, default='A',
                            help="which domains to augment; 'A' only by default -- jittering the target B would "
                                 'fight the paired DAB loss it is compared against')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.stain_A = build_stain_pipeline(opt, 'A')
        self.stain_B = build_stain_pipeline(opt, 'B')

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        # Stain ops run here rather than inside get_transform because they need
        # the image *after* resizing: estimating a Macenko basis on the full
        # 1748x1748 tile costs more than the training step. Pre-resizing to
        # load_size is exactly what get_transform's Resize would do next, so it
        # is a no-op there -- but only for the 'resize' preprocess modes.
        if self.stain_A is not None or self.stain_B is not None:
            if 'resize' in modified_opt.preprocess:
                size = (modified_opt.load_size, modified_opt.load_size)
                A_img = A_img.resize(size, Image.BICUBIC)
                B_img = B_img.resize(size, Image.BICUBIC)
            if self.stain_A is not None:
                A_img = self.stain_A(A_img)
            if self.stain_B is not None:
                B_img = self.stain_B(B_img)

        transform = get_transform(modified_opt)
        A = transform(A_img)
        B = transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
