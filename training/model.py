"""
model.py

Dual-head U-Net architecture for MoNuSAC nucleus segmentation.

SharedUnetTwoHead: shared SMP encoder + shared decoder + two independent
segmentation heads (semantic 5-class, ternary 3-class).
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead


class SharedUnetTwoHead(nn.Module):
    """
    Version-robust dual-head U-Net:
      - shared SMP Unet encoder+decoder
      - two segmentation heads on top of decoder output

    Returns:
      sem_logits: [B, sem_classes, H, W]
      ter_logits: [B, ter_classes, H, W]
    """
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 3,
        decoder_channels=(256, 128, 64, 32, 16),
        sem_classes: int = 5,
        ter_classes: int = 3,
        activation=None,
    ):
        super().__init__()

        self.decoder_channels = tuple(decoder_channels)

        self.base = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,  # dummy
            activation=None,
            decoder_channels=self.decoder_channels,
        )

        dec_out_ch = self.decoder_channels[-1]

        # Spatial dropout on shared decoder features (regularization)
        self.drop = nn.Dropout2d(p=0.2)

        self.sem_head = SegmentationHead(
            in_channels=dec_out_ch,
            out_channels=sem_classes,
            activation=activation,
            kernel_size=3,
        )
        self.ter_head = SegmentationHead(
            in_channels=dec_out_ch,
            out_channels=ter_classes,
            activation=activation,
            kernel_size=3,
        )

    def forward(self, x: torch.Tensor):
        feats = self.base.encoder(x)
        try:
            dec = self.base.decoder(*feats)
        except TypeError:
            dec = self.base.decoder(feats)
        dec = self.drop(dec)  # spatial dropout before heads
        sem = self.sem_head(dec)
        ter = self.ter_head(dec)
        return sem, ter
