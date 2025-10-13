import torch
import torchvision.transforms as transforms
from torchvision import tv_tensors

from src.dataset.data_augmentation.color_augmentation import ColorimetricAugmentationParams, get_colorimetric_augmentation_pipeline
from src.dataset.data_augmentation.spatial_augmentation import SpatialAugmentationParams, get_spatial_augmentation_pipeline


class AugmentedSemanticSegmentationDataset(torch.utils.data.Dataset):
    """ Add spatial and colorimetric augmentation features to a semantic segmentation dataset """
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        spatial_params: SpatialAugmentationParams | None,
        colorimetric_params: ColorimetricAugmentationParams | None,
    ):
        self.dataset = dataset
        self.spatial_params = spatial_params
        self.colorimetric_params = colorimetric_params
        self.spatial_aug = get_spatial_augmentation_pipeline(spatial_params)
        self.colorimetric_aug = get_colorimetric_augmentation_pipeline(colorimetric_params)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # get image and associated mask (tensors)
        image, mask = self.dataset[idx]

        # Apply colorimetric augmentations to the image only
        aug_image = self.colorimetric_aug(image)
        
        # Apply spatial transformation
        # wrap mask using 'tv_tensors.Mask' to apply the right options when transforming masks (bilinear -> nearest interpolation...)
        aug_image, aug_mask = self.spatial_aug(aug_image, tv_tensors.Mask(mask)) 
        return aug_image, aug_mask

if __name__ == "__main__":
    """ Example of use of the AugmentedSemanticSegmentationDataset class"""
    from pathlib import Path
    
    from torch.utils.data import DataLoader

    from src.utils.display_image_tensor import display_image_tensor, display_mask_tensor
    from src.dataset.custom_image_mask_dataset import CustomImageMaskDataset
    


    BATCH_SIZE = 1

    # instanciate a semantic segmentation dataset (JSRT)
    dataset = CustomImageMaskDataset(
        Path(r"D:\user\JB\code\u-net-pytorch\data\JSRT-segmentation\test")
    )

    # Wrap the dataset to augment data
    spatial_params = SpatialAugmentationParams()
    colorimetric_params = ColorimetricAugmentationParams()
    augmented_dataset = AugmentedSemanticSegmentationDataset(dataset, spatial_params, colorimetric_params)


    loader = DataLoader(augmented_dataset, BATCH_SIZE, True)

    for image, mask in loader:
        display_image_tensor(image.squeeze())  # remove batch dimension
        display_mask_tensor(mask.squeeze())
        break
