"""colorimetric image augmentation pipeline"""

from dataclasses import dataclass, field

from torchvision.transforms import v2


@dataclass
class BrightnessAugmentation:
    """parameters associated to image brightness augmentation"""

    # the brightness factor to apply is inside this interval
    factor_limit: tuple[float, float] = (0.8, 1.2)
    # probability of applying the transformation
    p: float = 0.5


@dataclass
class ContrastAugmentation:
    """parameters associated to image contrast augmentation"""

    factor_limit: tuple[float, float] = (
        0.8,
        1.2,
    )  # the brightness factor to apply is inside this interval
    p: float = 0.5  # probability of applying the transformation


@dataclass
class GaussianNoiseAugmentation:
    """parameters associated to gaussian noise augmentation"""

    sigma: float = 0.01
    p: float = 0.5


@dataclass
class ColorimetricAugmentationParams:
    """Parameters for colorimetric augmentations (applied only to the image)."""

    brightness_augmentation: BrightnessAugmentation = field(default_factory=BrightnessAugmentation)
    contrast_augmentation: ContrastAugmentation = field(default_factory=ContrastAugmentation)
    noise_augmentation: GaussianNoiseAugmentation = field(default_factory=GaussianNoiseAugmentation)


def get_colorimetric_augmentation_pipeline(params: ColorimetricAugmentationParams) -> v2.Compose:
    """Returns a pipeline for colorimetric augmentations using torchvision v2."""
    return v2.Compose(
        [
            v2.RandomApply(
                [v2.ColorJitter(brightness=params.brightness_augmentation.factor_limit)],
                p=params.brightness_augmentation.p,
            ),
            v2.RandomApply(
                [v2.ColorJitter(contrast=params.contrast_augmentation.factor_limit)],
                p=params.contrast_augmentation.p,
            ),
            v2.RandomApply(
                (v2.GaussianNoise(sigma=params.noise_augmentation.sigma),),
                p=params.noise_augmentation.p,
            ),
        ]
    )
