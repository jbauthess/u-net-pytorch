"""spatial image augmentation pipeline"""

from dataclasses import dataclass, field
from typing import Sequence, Tuple

from torchvision.transforms import v2


@dataclass
class ElasticTransformParams:
    """parameters associated to elastic image deformation"""

    sigma: float | Sequence[float] = 4.0
    alpha: float = 10
    p: float = 1.0


@dataclass
class SpatialAugmentationParams:
    """Parameters for spatial augmentations (applied to both image and mask)."""

    degrees: float = 10.0
    p_rotate: float = 0.5
    p_flip: float = 0.5

    elastic_transform: ElasticTransformParams = field(default_factory=ElasticTransformParams)
    image_size: Tuple[int, int] = (256, 256)


def get_spatial_augmentation_pipeline(params: SpatialAugmentationParams) -> v2.Compose:
    """Returns a pipeline for spatial augmentations using torchvision v2."""
    return v2.Compose(
        [
            v2.RandomHorizontalFlip(p=params.p_flip),
            v2.RandomApply([v2.RandomRotation(degrees=params.degrees)], p=params.p_rotate),
            v2.RandomApply(
                [
                    v2.ElasticTransform(
                        alpha=params.elastic_transform.alpha, sigma=params.elastic_transform.sigma
                    )
                ],
                p=params.elastic_transform.p,
            ),
            v2.Resize(params.image_size),
        ]
    )
