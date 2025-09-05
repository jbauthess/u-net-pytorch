import torch
from torch.nn import Conv2d

from .blocks import (
    CONV_KERNEL_SIZE,
    create_convolutional_block,
    create_downsampling_block,
    create_upsampling_block,
)


class UNetModel(torch.nn.Module):
    def __init__(self, nb_in_channels: int, nb_classes: int, base_fm_number: int = 64):
        super(UNetModel, self).__init__()

        # blocks corresponding to the contracting path
        self.compute_fm = create_convolutional_block(
            nb_in_channels, CONV_KERNEL_SIZE, base_fm_number
        )

        self.down1 = create_downsampling_block(base_fm_number, base_fm_number * 2)
        self.down2 = create_downsampling_block(base_fm_number * 2, base_fm_number * 4)
        self.down3 = create_downsampling_block(base_fm_number * 4, base_fm_number * 8)
        self.down4 = create_downsampling_block(base_fm_number * 8, base_fm_number * 16)

        # blocks corresponding to expansive path
        self.up1 = create_upsampling_block(base_fm_number * 16, base_fm_number * 8)
        self.up2 = create_upsampling_block(base_fm_number * 8, base_fm_number * 4)
        self.up3 = create_upsampling_block(base_fm_number * 4, base_fm_number * 2)
        self.up4 = create_upsampling_block(base_fm_number * 2, base_fm_number)

        # prediction
        self.out = Conv2d(base_fm_number, nb_classes, (1, 1), 1, padding=0)

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

    def get_nb_classes(self) -> int:
        return self.out.out_channels
