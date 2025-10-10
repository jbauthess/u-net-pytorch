from dataclasses import dataclass
from random import randint
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class Square:
    x: int
    y: int
    s: int
    color: Tuple[int, int, int]


class SquareDataset(Dataset):
    """this class generates a dataset of images containing a square at random location"""

    def __init__(
        self,
        nb_channels: int,
        img_width: int,
        img_height: int,
        nb_img: int,
        square_size_min: int,
        square_size_max: int,
    ):
        """_summary_

        Args:
            nb_channels (int); number of channels of the image
            img_width (int): width of the image
            img_height (int): height of the image
            nb_img (int): number of images in the dataset
            square_size_min (int) : min square size
            square_size_max (int) : max square size
        """
        self.nb_img = nb_img
        self.img_width, self.img_height = img_width, img_height
        self.nb_channels = nb_channels

        self.squares: List[Square] = []

        # generate square properties for each image in the dataset
        for i in range(0, self.nb_img):
            square_size = randint(square_size_min, square_size_max)

            color = tuple(randint(1, 255) for _ in range(nb_channels))

            self.squares.append(
                Square(
                    randint(0, img_width - square_size - 1),
                    randint(0, img_height - square_size - 1),
                    square_size,
                    color,
                )
            )

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return self.nb_img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx: Index of the sample to be retrieved.

        Returns:
            A tuple of (image, image mask) for the sample at the given index.
            image contains a square of random size and random location with a random color
            image mask contains the binary mask of pixels belonging to the square
        """
        square = self.squares[idx]
        square_img = torch.zeros((self.nb_channels, self.img_height, self.img_width))
        mask_img = torch.zeros((self.img_height, self.img_width))

        for i, c in enumerate(square.color):
            square_img[
                i, square.y : square.y + square.s, square.x : square.x + square.s
            ] = c

        mask_img[square.y : square.y + square.s, square.x : square.x + square.s] = 1

        return square_img, mask_img
