"""Image and mask augmentation adapted to semantic segmentation use case :
- the SAME spatial augmentation is applied to image and mask
- a colorimetric transformation is also applied on image
"""

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import tv_tensors

from src.dataset.data_augmentation.color_augmentation import (
    ColorimetricAugmentationParams,
    get_colorimetric_augmentation_pipeline,
)
from src.dataset.data_augmentation.spatial_augmentation import (
    SpatialAugmentationParams,
    get_spatial_augmentation_pipeline,
)


class AugmentedSemanticSegmentationDataset(Dataset[tuple[Tensor, Tensor]]):
    """Add spatial and colorimetric augmentation features to a semantic segmentation dataset"

    HOW TO USE:
    -----------
    BATCH_SIZE = 1

    # instanciate a semantic segmentation dataset (JSRT)
    dataset = CustomImageMaskDataset(
        <path of the dataset>)
    )

    # Wrap the dataset to augment data
    spatial_params = SpatialAugmentationParams()
    colorimetric_params = ColorimetricAugmentationParams()
    augmented_dataset = AugmentedSemanticSegmentationDataset(
        dataset, spatial_params, colorimetric_params
    )

    loader = DataLoader(augmented_dataset, BATCH_SIZE, True)

    for image, mask in loader:
        <do what you need to do>

    """

    def __init__(
        self,
        dataset: Dataset[tuple[Tensor, Tensor]],
        spatial_params: SpatialAugmentationParams | None,
        colorimetric_params: ColorimetricAugmentationParams | None,
    ):
        self.dataset = dataset
        self.spatial_params = spatial_params
        self.colorimetric_params = colorimetric_params
        if spatial_params:
            self.spatial_aug = get_spatial_augmentation_pipeline(spatial_params)

        if colorimetric_params:
            self.colorimetric_aug = get_colorimetric_augmentation_pipeline(colorimetric_params)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        # get image and associated mask (tensors)
        image, mask = self.dataset[idx]

        # Apply colorimetric augmentations to the image only
        aug_image = self.colorimetric_aug(image)

        # Apply spatial transformation
        # wrap mask using 'tv_tensors.Mask' to apply the right options when transforming masks
        # (bilinear -> nearest interpolation...)
        aug_image, aug_mask = self.spatial_aug(aug_image, tv_tensors.Mask(mask))
        return aug_image, aug_mask
