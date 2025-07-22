import torch
from torch.nn import Conv2d

from .blocks import (
    CONV_KERNEL_SIZE,
    create_convolutional_block,
    create_downsampling_block,
    create_upsampling_block,
)


class UNetModel(torch.nn.Module):
    def __init__(self, nb_in_channels: int, nb_classes: int):
        super(UNetModel, self).__init__()

        # blocks corresponding to the contracting path
        self.compute_fm = create_convolutional_block(
            nb_in_channels, CONV_KERNEL_SIZE, 64
        )

        self.down1 = create_downsampling_block(64, 128)
        self.down2 = create_downsampling_block(128, 256)
        self.down3 = create_downsampling_block(256, 512)
        self.down4 = create_downsampling_block(512, 1024)

        # blocks corresponding to expansive path
        self.up1 = create_upsampling_block(1024, 512)
        self.up2 = create_upsampling_block(512, 256)
        self.up3 = create_upsampling_block(256, 128)
        self.up4 = create_upsampling_block(128, 64)

        # prediction
        self.out = Conv2d(64, nb_classes, (1, 1), 1, padding=0)

    def forward(self, x):
        # coompress the information (encoder)
        x0 = self.compute_fm(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        # uncompress information (decoder)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)

        # predicts probability maps, one per class
        return self.out(x)
