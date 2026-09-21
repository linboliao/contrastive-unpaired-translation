"""Lightweight segmentation head bolted onto CUTModel's ResnetGenerator: taps
the last full-resolution decoder feature map (right before the final RGB
conv) via netG's existing layers=/encode_only-capable forward
(models/networks.py ResnetGenerator.forward), so the generator itself needs
zero modification.
"""
import torch.nn as nn


def resnet_pre_final_layer(netG_name):
    """Index, in ResnetGenerator.model (an nn.Sequential), of the last
    full-resolution ReLU before the [ReflectionPad, Conv, Tanh] RGB head.

    ResnetGenerator.__init__ always builds: 4 stem layers + 2*4 downsampling
    layers + n_blocks resnet blocks + 2*4 upsampling layers + 3 head layers.
    The last upsampling ReLU -- our tap point -- therefore sits at index
    19 + n_blocks.
    """
    if netG_name == "resnet_9blocks":
        return 28
    if netG_name == "resnet_6blocks":
        return 25
    raise NotImplementedError(
        f"cut_seg only supports resnet_9blocks/resnet_6blocks generators (got {netG_name}); "
        "other architectures (unet_*, stylegan2, ...) don't expose a same-resolution "
        "decoder feature at a known layer index."
    )


class SegHead(nn.Module):
    """Outputs raw logits (sigmoid applied by the caller) for a 1-channel
    nucleus-probability map, from ngf-channel full-resolution decoder
    features -- no up/downsampling needed here.
    """

    def __init__(self, ngf):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf // 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(ngf // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)
