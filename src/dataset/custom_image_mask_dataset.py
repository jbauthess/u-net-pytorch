import os
from pathlib import Path

import numpy as np
from PIL import Image
from torch import from_numpy
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from src.dataset.mask import expand_flattened_mask
from src.errors import MultipleFilesFoundError
from src.utils.display_image_tensor import display_image_tensor, display_mask_tensor


def find_image_mask_pairs(folder: Path) -> list[tuple[Path, Path]]:
    """
    go through a folder and returns pairs (image, mask)
    assuming corresponding images and masks filename follows the naming rule:
    image : 'toto.x' -> mask : 'toto_label.y'. Image and mask file extensions may  differ

    Args:
        folder: path of the folder containing images and masks

    Returns:
        List of tuples (image_path, mask_path).
    """
    # Separate images and masks
    images = []
    mask_dict = {}  # Maps mask stem (e.g., "toto33_label") to mask path

    for file in folder.iterdir():
        if not file.is_file():
            continue
        if "_label" in file.stem:
            if file.stem in mask_dict:
                raise MultipleFilesFoundError(
                    f"Several masks {mask_dict[file.stem]} and {file} corresponds to the same image!"
                )
            mask_dict[file.stem] = file  # Store mask stem as key
        else:
            images.append(file)

    # Match images to masks
    pairs = []
    for image in images:
        mask_stem = f"{image.stem}_label"
        if mask_stem in mask_dict:
            pairs.append((image, mask_dict[mask_stem]))
        else:
            raise FileNotFoundError(f"No mask found for {image.name}")

    return pairs


class CustomImageMaskDataset(Dataset):
    def __init__(self, dataset_folder: Path, transform=None):
        """
        Dataset class for handling data stored as image and mask files in a folder

        Args:
            dataset_folder (str): Path to the directory containing images.
            transform (callable, optional): Transformations to apply to both images and masks.
        """
        self.dataset_folder = dataset_folder
        self.transform = transform
        # construct pairs (image file, mask file)
        self.image_mask_corresp = find_image_mask_pairs(dataset_folder)

    def __len__(self):
        return len(self.image_mask_corresp)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.dataset_folder, self.image_mask_corresp[idx][0])
        mask_path = os.path.join(self.dataset_folder, self.image_mask_corresp[idx][1])
        image = Image.open(img_path).convert("L")
        flattened_mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # expand mask to store a featuremap per label
        mask = expand_flattened_mask(np.array(flattened_mask))

        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a
        # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        image = to_tensor(image)

        # convert mask to tensor using channel first memory format
        mask = from_numpy(mask).permute(2, 0, 1).to(dtype=float)

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


if __name__ == "__main__":
    """ Example of use of the CustomImageMaskDataset class"""
    from torch.utils.data import DataLoader

    BATCH_SIZE = 1

    dataset = CustomImageMaskDataset(
        Path(r"D:\user\JB\code\u-net-pytorch\data\JSRT-segmentation\test"), None
    )
    loader = DataLoader(dataset, BATCH_SIZE, True)

    for image, mask in loader:
        display_image_tensor(image.squeeze() * 255)
        display_mask_tensor(mask.squeeze() * 255)
        break
