"""base blocks of the U-NET architecture"""

from typing import Tuple

import torch
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)

CONV_KERNEL_SIZE = (3, 3)


def create_convolutional_block(
    nb_in_channels: int, conv_kernel_size: Tuple[int, int], nb_out_channels: int
) -> Sequential:
    """
    create convolutional block composed of two consecutive convolutions

    Args:
        nb_in_channels (int): number of feature maps in the input tensor
                              on which the block will be applied
        conv_kernel_size (Tuple[int, int]): size of the convolutional kernel used
        nb_out_channels (int): number of feature maps of the output tensor

    Returns:
        Sequential: torch module combining convolutional blocks
    """
    return Sequential(
        Conv2d(
            nb_in_channels,
            nb_out_channels,
            conv_kernel_size,
            stride=1,
            padding=1,  # use "same" padding
            bias=False,
        ),
        BatchNorm2d(nb_out_channels),
        ReLU(inplace=False),
        Conv2d(
            nb_out_channels,
            nb_out_channels,
            conv_kernel_size,
            stride=1,
            padding=1,
            bias=False,
        ),
        BatchNorm2d(nb_out_channels),
        ReLU(inplace=False),
    )


def create_downsampling_block(nb_in_channels: int, nb_out_channels: int) -> Sequential:
    """
    create a block used to compress information in the U-NET architecture :
    composition of convolutional and reLU layers followed by a maxpooling.

    Args:
        nb_in_channels (int): number of channels of the tensor on which the block will be applied
        nb_out_channels (int): number of channels of the tensor after applying this block

    Returns:
        Sequential: the pytorch module corresponding to a contracting block
    """

    # Note : In the original paper, it consists of two 3x3 convolutions (unpadded convolutions),
    # each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride
    # for downsampling. At each downsampling step the number of feature channels is doubled
    return Sequential(
        create_convolutional_block(nb_in_channels, CONV_KERNEL_SIZE, nb_out_channels),
        MaxPool2d((2, 2), 2),
    )


class UpsamplingBlock(Module):
    """
    block used to expand information in the U-NET architecture :
    up-convolution + concatenation with the feature maps from the contractive path with
    higher dimensions.

    Args:
        nb_in_channels (int): number of channels of the tensor on which the block will be applied
        nb_out_channels (int): number of channels of the tensor after applying this block
    """

    def __init__(self, nb_in_channels: int, nb_out_channels: int):
        super().__init__()
        # define block corresponding to the upsampling path
        # halve the channel dimension while up‑sampling
        self.up = ConvTranspose2d(nb_in_channels, nb_in_channels // 2, kernel_size=2, stride=2)
        self.conv = create_convolutional_block(nb_in_channels, CONV_KERNEL_SIZE, nb_out_channels)

    def forward(self, x: torch.Tensor, contractive_feature_map: torch.Tensor) -> torch.Tensor:
        """In the original paper, it consists of an upsampling of the feature map followed
        by a 2x2 convolution (up-convolution) that halves the number of feature channels,
        a concatenation with the correspondingly cropped feature map from the contracting path,
        and two 3x3 convolutions, each followed by a ReLU"""
        x = self.up(x)

        # While cropping was used in the original paper, no croppong is now required
        # because of the use of "same" paddingin the convolutional block

        # reminder : original paper implementation:
        # # input is CHW
        # # The cropping is necessary due to the loss of border pixels in every convolution.
        # crop = torchvision.transforms.functional.center_crop(
        #     contractive_feature_map, [x.shape[2], x.shape[3]]
        # )

        x = torch.cat([contractive_feature_map, x], dim=1)

        x = self.conv(x)

        return x


def create_upsampling_block(nb_in_channels: int, nb_out_channels: int) -> UpsamplingBlock:
    """
    create a block used to expand information in the U-NET architecture

    Args:
        nb_in_channels (int): number of channels of the tensor on which the block will be applied
        nb_out_channels (int): number of channels of the tensor after applying this block

    Returns:
        UpsamplingBlock: the pytorch module corresponding to a expanding block
    """
    return UpsamplingBlock(nb_in_channels, nb_out_channels)
