"""unit-tests for unet.py module"""

import torch

from src.model.unet import UNetModel


def test_model_architecture_output_size() -> None:
    """test that the appication of the unet model yield
    an output volume of the right size"""
    # build model predicting five classes on gray image
    model = UNetModel(1, 5)

    # simulate an image
    batch_image = torch.ones((1, 1, 384, 384))

    # run inference
    output = model(batch_image)

    # check output size is valid
    assert output.size() == (1, 5, 384, 384)
