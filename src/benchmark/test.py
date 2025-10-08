import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from src.model.unet import UNetModel
from src.utils.display_image_tensor import display_image_tensor, display_mask_tensor


def test(device: torch.device, model: UNetModel, test_dataset: Dataset, verbose=0):
    model.eval()

    with torch.no_grad():
        # test_dataset = SquareDataset(3, IMG_WIDTH, IMG_HEIGHT, 10, 20, 80)
        test_loader = DataLoader(test_dataset, 1, True)

        for _, data in enumerate(test_loader):
            images, gt_masks = data
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            mask = torch.zeros_like(outputs)
            mask[outputs > 0.5] = 255
            resized_mask = torchvision.transforms.functional.resize(
                mask, images.shape[2:], torchvision.transforms.InterpolationMode.NEAREST
            )

            # TO DO : compute matching score between generated mask with ground truth mask

            # display predicted mask?
            if verbose:
                display_image_tensor(images[0].to("cpu") * 255)
                display_mask_tensor(resized_mask[0].to("cpu") * 255)
