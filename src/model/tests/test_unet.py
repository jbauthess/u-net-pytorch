import torch

from src.model.unet import UNetModel


def test_model_architecture() -> None:
    # build model predicting five classes on gray image
    M = UNetModel(1, 5)

    # simulate an image
    I = torch.ones((1, 1, 572, 572))

    # run inference
    x = M(I)

    # check output size is valid
    assert x.size() == (1, 5, 388, 388)
