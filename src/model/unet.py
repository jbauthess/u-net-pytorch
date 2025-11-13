"""Implementation of the U-NET architecture"""

from torch import Tensor
from torch.nn import Conv2d

# import base blocks on which rely the U-Net architecture
from src.model.blocks import (
    CONV_KERNEL_SIZE,
    create_convolutional_block,
    create_downsampling_block,
    create_upsampling_block,
)
from src.model.semantic_segmentation_model import SemanticSegmentationModel


class UNetModel(SemanticSegmentationModel):  # pylint: disable=too-many-instance-attributes
    """U-NET architecture"""

    def __init__(self, nb_in_channels: int, nb_labels: int, base_fm_number: int = 64):
        """intitialize UNet model architecture blocks

        Args:
            nb_in_channels (int): number of channels in the input image passed to the model
            nb_labels (int): number of labels the model is able to predict
            base_fm_number (int, optional): base number used to define the number of feature maps
                                            in each layer of the model.
                                            The larger this number, the larger the number of
                                            model parameters and memory / computation load
        """
        super().__init__(nb_labels)

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
        self.out = Conv2d(base_fm_number, nb_labels, (1, 1), 1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        """Apply Unet model on a batch of images

        Args:
            x (Tensor): batch of images. Images should have the same number of channels
                        as the one used to initialize the model (nb_in_channels)

        Returns:
            Tensor: output tesor. Same resolution as the input image passed to the model.
            The number of channels of the output tensor corresponds to
            the number of labels the model predicts (nb_labels). Each channel (feature map)
            of the output tensor corresponds to a label. Each pixel of a channel stores
            a score homogeneous to the probability the pixel corresponds to the label
        """

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

        # predicts probability maps, one per label
        output: Tensor = self.out(x)
        return output
