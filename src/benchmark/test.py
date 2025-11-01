import logging
from enum import StrEnum
from typing import List

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset

from src.benchmark.metrics import compute_pixelwise_accuracy
from src.model.semantic_segmentation_model import SemanticSegmentationModel
from src.utils.display_image_tensor import display_image_tensor, display_mask_tensor

logger = logging.getLogger()


class TestMetrics(StrEnum):
    ACCURACY = "accuracy"


def generate_report(gt, pred, metrics: List[TestMetrics]) -> None:
    """generate the report corresponding to model performances"""
    for m in metrics:
        match m:
            case TestMetrics.ACCURACY:
                acc = compute_pixelwise_accuracy(gt, pred)
                logger.info(f"GLOABL ACCURACY={acc}")
            case _:
                raise NotImplementedError("This metric is not available")


def generate_mask_from_prediction(
    predicted_logits: torch.Tensor, thresh: float = 0.7
) -> torch.Tensor:
    """convert semantic segmentation model output feature maps (one feature map per class) into
    a segmentation mask (flatten mask containing pixels with different labels, one label per class)
    """
    # the number of target classes corresponds to the number of channels in the predicted volume
    nb_classes = predicted_logits.shape[1]

    if nb_classes > 1:
        # Apply softmax along the channel dimension to convert logits into probabilities
        probs = torch.softmax(predicted_logits, dim=1)  # shape: (N, C, H, W)

        # cancel not reliable pixels
        probs[probs < thresh] = 0

        # Get predicted class per pixel
        pred_mask = torch.argmax(probs, dim=1)  # shape: (N, H, W)

    else:
        # apply sigmoid to convert logit into a probability
        probs = torch.sigmoid(predicted_logits)  # shape: (N, 1, H, W)
        probs = probs.squeeze_(0)  # shape: (N, H, W)
        pred_mask = probs >= thresh

    return pred_mask


def test(
    device: torch.device,
    model: SemanticSegmentationModel,
    test_dataset: Dataset,
    metrics: List[TestMetrics] | None,
    verbose=0,
):
    model.eval()

    with torch.no_grad():
        # test_dataset = SquareDataset(3, IMG_WIDTH, IMG_HEIGHT, 10, 20, 80)
        test_loader = DataLoader(test_dataset, 1, True)

        for gt_mask, data in enumerate(test_loader):
            images, gt_masks = data
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            mask = generate_mask_from_prediction(outputs, 0.5)

            resized_mask = torchvision.transforms.functional.resize(
                mask, images.shape[2:], torchvision.transforms.InterpolationMode.NEAREST
            )

            # compute and print matching score between generated mask with ground truth mask
            if metrics:
                generate_report(gt_masks.numpy(), mask.to("cpu").numpy(), metrics)

            # display predicted mask?
            if verbose:
                display_image_tensor(images[0].to("cpu"))
                display_mask_tensor(gt_masks[0])
                display_mask_tensor(mask.to("cpu").squeeze(0))  # remove batch dimension
                # display_multilabel_mask_tensor(resized_mask[0].to("cpu"))
