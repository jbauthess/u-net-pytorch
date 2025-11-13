"""unit-testing of the square_dataset.py module"""

import torch

from src.dataset.square_dataset import SquareDataset


def test_square_dataset_getitem() -> None:
    """Test the random generation of square images"""
    # generate 3 color images 640 x 480 with a square of size 50 x 50 at random location
    nb_channels = 3
    width = 640
    height = 480
    nb_img = 3
    square_size = 50

    dataset = SquareDataset(nb_channels, width, height, nb_img, square_size, square_size)

    # 3 squares should be generated internally with size square_size
    assert len(dataset.squares) == 3

    for square in dataset.squares:
        assert square.s == square_size

    # Does generated image contain a square of the right size?
    # NOTE : because insufficient type annotation in torch library
    # -> mypy error: "SquareDataset" has no attribute "__iter__" (not iterable)  [attr-defined]
    for data in dataset:  # type: ignore
        img, mask = data
        assert torch.count_nonzero(img) == square_size * square_size * nb_channels
        assert torch.count_nonzero(mask) == square_size * square_size


if __name__ == "__main__":
    import pytest

    pytest.main()
