from dataclasses import dataclass, field
from typing import Tuple, Optional
import torch
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2


@dataclass
class BrightnessAugmentation:
    factor_limit:tuple[float, float] = (0.8, 1.2) # the brightness factor to apply is inside this interval
    p:float = 0.5                                 # probability of applying the transformation

@dataclass 
class ContrastAugmentation:
    factor_limit:tuple[float, float] = (0.8, 1.2) # the brightness factor to apply is inside this interval
    p:float = 0.5                                 # probability of applying the transformation

@dataclass
class GaussianNoiseAugmentation:
    sigma: float = 0.01
    p: float = 0.5

@dataclass
class ColorimetricAugmentationParams:
    """Parameters for colorimetric augmentations (applied only to the image)."""
    brightness_augmentation:BrightnessAugmentation = field(default_factory=lambda:BrightnessAugmentation())
    contrast_augmentation:ContrastAugmentation = field(default_factory=lambda:ContrastAugmentation())
    noise_augmentation:GaussianNoiseAugmentation = field(default_factory=lambda:GaussianNoiseAugmentation())


def get_colorimetric_augmentation_pipeline(params: ColorimetricAugmentationParams) -> v2.Compose:
    """Returns a pipeline for colorimetric augmentations using torchvision v2."""
    return v2.Compose([
        v2.RandomApply(
            [v2.ColorJitter(brightness=params.brightness_augmentation.factor_limit)],
            p=params.brightness_augmentation.p,
        ),
        v2.RandomApply(
            [v2.ColorJitter(contrast=params.contrast_augmentation.factor_limit)],
            p=params.contrast_augmentation.p,
        ),
        v2.RandomApply((v2.GaussianNoise(sigma=params.noise_augmentation.sigma),),
            p=params.noise_augmentation.p,
        )
    ])

