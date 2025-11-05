import logging
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset

from src.benchmark.match import compute_match_maps_one_label
from src.benchmark.metrics import MatchResult
from src.benchmark.report import TestMetrics, generate_report
from src.model.semantic_segmentation_model import SemanticSegmentationModel
from src.utils.display_image_tensor import display_image_tensor, display_mask_tensor

logger = logging.getLogger()


def generate_mask_from_prediction(
    predicted_logits: torch.Tensor, thresh: float = 0.7
) -> torch.Tensor:
    """convert semantic segmentation model output feature maps (one feature map per label) into
    a segmentation mask (flattened mask containing pixels with different labels, one label per pixel)
    """
    # the number of target labels corresponds to the number of channels in the predicted volume
    nb_labels = predicted_logits.shape[1]

    if nb_labels > 1:
        # Apply softmax along the channel dimension to convert logits into probabilities
        probs = torch.softmax(predicted_logits, dim=1)  # shape: (N, C, H, W)

        # cancel not reliable pixels
        probs[probs < thresh] = 0

        # Get predicted label per pixel
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

    nb_labels = model.get_nb_labels()

    match_result = MatchResult(nb_labels)

    with torch.no_grad():
        test_loader = DataLoader(test_dataset, 1, True)

        for _, data in enumerate(test_loader):
            images: torch.Tensor = data[0]
            gt_masks: torch.Tensor = data[1]
            images = images.to(device)

            # Forward pass
            outputs = model(images)  # dim (N, C, H, W)

            pred_masks = generate_mask_from_prediction(outputs, 0.5).to(
                "cpu"
            )  # dim (N, H, W)

            # pred_masks = torchvision.transforms.functional.resize(
            #     pred_masks,
            #     images.shape[2:],
            #     torchvision.transforms.InterpolationMode.NEAREST,
            # )

            # compute metrics
            if metrics:
                # compute matching per label
                for label in range(nb_labels):
                    match_maps = compute_match_maps_one_label(
                        gt_masks.numpy(), pred_masks.numpy(), label=label
                    )

                    match_result.update_score_one_label(match_maps, label)

                # update the total number of pixels processed
                match_result.update_nb_pixels(gt_masks.numel())

            # display predicted mask?
            if verbose:
                display_image_tensor(images[0].to("cpu"))
                display_mask_tensor(gt_masks[0])
                display_mask_tensor(pred_masks[0])  # remove batch dimension
                # display_multilabel_mask_tensor(resized_mask[0].to("cpu"))

        # generate evaluation report
        generate_report(match_result, metrics)
