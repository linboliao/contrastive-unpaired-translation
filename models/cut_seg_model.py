"""CUT + a bolted-on nucleus segmentation head, sharing netG's decoder
features instead of retraining from scratch.

    generator (shared, unmodified) --> RGB head (existing, untouched)
                                    \-> seg head (new, this file): 1-ch
                                        nucleus-probability logits, tapped
                                        off the last full-resolution decoder
                                        feature map (see seg_head.py)

The seg head is supervised by Dice + focal loss against a mask derived from
the registered real_B image (see nucleus_mask.py). The existing GAN and DAB
losses are kept (not removed), just turned down by default so training
capacity shifts toward the segmentation head -- see modify_commandline_options.
netG/netF/netD are completely unmodified; this file only subclasses CUTModel
and adds the head, its loss, and a load_networks override so netSeg is never
expected to exist in a warm-start --pretrained_name checkpoint.

See scripts/run_erg_v2_seg_dab_l1.sh for the intended use: warm-start from
ERG_v2_dab_l1 (closest-calibrated run in the sweep, see
qc/erg_v2_sweep_scores.tsv) with a fresh segmentation head.
"""
import itertools
import os

import numpy as np
import torch

from . import networks
from .cut_model import CUTModel
from .nucleus_mask import nucleus_mask_from_rgb
from .seg_head import SegHead, resnet_pre_final_layer
from .seg_losses import DiceLoss, FocalLoss


class CUTSegModel(CUTModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CUTModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--lambda_seg_dice", type=float, default=1.0,
                             help="weight for the Dice loss on the nucleus-probability segmentation head")
        parser.add_argument("--lambda_seg_focal", type=float, default=1.0,
                             help="weight for the focal loss on the nucleus-probability segmentation head")
        parser.add_argument("--focal_gamma", type=float, default=2.0, help="focal loss focusing parameter")
        parser.add_argument("--focal_alpha", type=float, default=0.25, help="focal loss positive-class weight")
        # Bolting a segmentation head on: turn the pixel-level GAN/DAB losses
        # down (not off) so capacity shifts toward the nucleus mask, per the
        # brief. Still explicitly overridable on the command line.
        parser.set_defaults(lambda_GAN=0.3, lambda_DAB=0.3)
        return parser

    def __init__(self, opt):
        CUTModel.__init__(self, opt)

        self.seg_tap_layer = resnet_pre_final_layer(opt.netG)
        self.netSeg = networks.init_net(SegHead(opt.ngf), opt.init_type, opt.init_gain, self.gpu_ids)
        if "Seg" not in self.model_names:
            self.model_names = self.model_names + ["Seg"]
        self.loss_names = self.loss_names + ["Seg_Dice", "Seg_Focal"]
        self.visual_names = self.visual_names + ["seg_prob_vis", "seg_target_vis"]

        if self.isTrain:
            # Shared-generator design: the seg head trains alongside netG in
            # the same optimizer step, not as a separate network.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG.parameters(), self.netSeg.parameters()),
                lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers[0] = self.optimizer_G  # CUTModel.__init__ appended [G, D] in that order
            self.criterion_dice = DiceLoss()
            self.criterion_focal = FocalLoss(gamma=opt.focal_gamma, alpha=opt.focal_alpha)

    def forward(self):
        """Same as CUTModel.forward, plus a second head off the same pass:
        netG's layers=/encode_only-capable forward returns (rgb_output,
        [tapped_features]) in one call, so this costs no extra generator
        forward pass.
        """
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake, feats = self.netG(self.real, layers=[self.seg_tap_layer], encode_only=False)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

        seg_feat = feats[0][:self.real_A.size(0)]  # real_A/fake_B branch only, not the idt_B branch
        self.seg_logits = self.netSeg(seg_feat)
        self.seg_prob = torch.sigmoid(self.seg_logits)
        self.seg_target = nucleus_mask_from_rgb(self.real_B)
        # remap [0, 1] -> [-1, 1] so util.tensor2im (which assumes [-1, 1]) saves these correctly
        self.seg_prob_vis = self.seg_prob * 2 - 1
        self.seg_target_vis = self.seg_target * 2 - 1

    def compute_G_loss(self):
        loss_G = super().compute_G_loss()
        self.loss_Seg_Dice = self.criterion_dice(self.seg_logits, self.seg_target) * self.opt.lambda_seg_dice
        self.loss_Seg_Focal = self.criterion_focal(self.seg_logits, self.seg_target) * self.opt.lambda_seg_focal
        self.loss_Seg = self.loss_Seg_Dice + self.loss_Seg_Focal
        return loss_G + self.loss_Seg

    def load_networks(self, epoch):
        """Like BaseModel.load_networks, except netSeg is never pulled from
        --pretrained_name: it doesn't exist there (this model bolts it onto
        a plain-CUT checkpoint after the fact). It only resumes from this
        experiment's own prior checkpoints, and is left freshly initialized
        the first time this experiment name is trained.
        """
        for name in self.model_names:
            if name == "Seg":
                load_dir = self.save_dir
            elif self.opt.isTrain and self.opt.pretrained_name is not None:
                load_dir = os.path.join(self.opt.checkpoints_dir, self.opt.pretrained_name)
            else:
                load_dir = self.save_dir

            load_path = os.path.join(load_dir, "%s_net_%s.pth" % (epoch, name))
            if not os.path.isfile(load_path):
                print(f"[cut_seg] no checkpoint at {load_path}, leaving net{name} freshly initialized")
                continue

            net = getattr(self, "net" + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print("loading the model from %s" % load_path)
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, "_metadata"):
                del state_dict._metadata
            net.load_state_dict(state_dict)
