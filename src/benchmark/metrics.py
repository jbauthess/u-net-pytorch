"""This module contains useful metrics to evaluate a semantic segmentation model"""

from typing import List

import numpy as np

from src.benchmark.match import MatchMaps


class MatchResultOneLabel:
    """store the number of pixels detected as True Positives, False Positives and False Negatives for a specific label"""

    tp: int  # True Positives
    fp: int  # False Positives
    fn: int  # False Negatives

    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, match_maps: MatchMaps) -> None:
        self.tp += np.sum(match_maps.tp)
        self.fp += np.sum(match_maps.fp)
        self.fn += np.sum(match_maps.fn)

    def __str__(self) -> str:
        return f"(tp={self.tp}, fp={self.fp}, fn={self.fn})"


class MatchResult:
    """Matching results for all labels"""

    match_per_label: List[MatchResultOneLabel]
    nb_pixels: int  # total number of pixels processed

    def __init__(self, nb_labels: int) -> None:
        self.nb_pixels = 0
        self.match_per_label = []

        for l in range(nb_labels):
            self.match_per_label.append(MatchResultOneLabel())

    def update_score_one_label(self, match_maps: MatchMaps, label: int) -> None:
        if label >= len(self.match_per_label) or label < 0:
            raise IndexError("label invalid!")

        self.match_per_label[label].update(match_maps)

    def update_nb_pixels(self, nb_pixels: int) -> None:
        self.nb_pixels += nb_pixels

    def __str__(self) -> str:
        text = f"MatchResult(nb_pix = {self.nb_pixels}): "

        for l, m in enumerate(self.match_per_label):
            text += f"label {l}: {str(m)} |"

        return text


def compute_pixelwise_accuracy(match_result: MatchResult) -> float:
    """compute the accuracy

    Args:
        match (MatchResult): matching results as the numbers of True Positives, False Positives, and False Negatives

    Returns:
        float: accuracy
    """
    if match_result.nb_pixels == 0:
        raise ValueError("Invalid input match_result : no pixels processed")

    correct_matches = 0
    for m in match_result.match_per_label:
        correct_matches += m.tp

    return correct_matches / match_result.nb_pixels


def _flatten_and_ignore(
    pred: np.ndarray, gt: np.ndarray, ignore_index: int | None = None
):
    """
    flatten pred and gt to be 1D vectors containing valid labels
    if ignore_index is provided, it is used to detect corresponding positions in gt and filter out
    those positions in pred and gt
    """
    pred = pred.ravel()
    gt = gt.ravel()
    if ignore_index is not None:
        mask = gt != ignore_index
        pred, gt = pred[mask], gt[mask]
    return pred, gt


def compute_pixelwise_accuracy_ref(pred, gt, ignore_index=None) -> float:
    """Compute Pixel Accuracy (PA) :  number of valid predicted pixels / total number of pixels"""
    # TODO(move as a reference implementation to test compute_pixelwise_accuracy() implementation in test_metrics.py)
    pred, gt = _flatten_and_ignore(pred, gt, ignore_index)
    return np.mean(pred == gt)


# def mean_class_accuracy(pred, gt, num_classes, ignore_index=None):
#     """Mean Class Accuracy (MCA) – moyenne des précisions par classe."""
#     pred, gt = _flatten_and_ignore(pred, gt, ignore_index)
#     acc_per_class = []
#     for c in range(num_classes):
#         mask = gt == c
#         if np.any(mask):
#             acc = np.mean(pred[mask] == c)
#             acc_per_class.append(acc)
#         else:
#             # classe absente du GT → on l’ignore dans la moyenne
#             acc_per_class.append(np.nan)
#     return np.nanmean(acc_per_class)


# def confusion_matrix(pred, gt, num_classes, ignore_index=None):
#     """Matrice de confusion C où C[i,j] = nb de pixels de vraie classe i prédites comme j."""
#     pred, gt = _flatten_and_ignore(pred, gt, ignore_index)
#     k = num_classes
#     cm = np.bincount(k * gt.astype(int) + pred.astype(int), minlength=k * k).reshape(
#         k, k
#     )
#     return cm


# def iou_per_class(cm):
#     """IoU pour chaque classe à partir de la matrice de confusion."""
#     tp = np.diag(cm)
#     fp = cm.sum(axis=0) - tp
#     fn = cm.sum(axis=1) - tp
#     denom = tp + fp + fn
#     # éviter la division par zéro
#     iou = np.where(denom > 0, tp / denom, np.nan)
#     return iou


# def mean_iou(pred, gt, num_classes, ignore_index=None):
#     """Mean IoU (mIoU)"""
#     cm = confusion_matrix(pred, gt, num_classes, ignore_index)
#     iou = iou_per_class(cm)
#     return np.nanmean(iou)


# def frequency_weighted_iou(pred, gt, num_classes, ignore_index=None):
#     """Frequency‑Weighted IoU (FWIoU)"""
#     cm = confusion_matrix(pred, gt, num_classes, ignore_index)
#     freq = cm.sum(axis=1) / cm.sum()
#     iou = iou_per_class(cm)
#     return np.nansum(freq * iou)


# def dice_per_class(cm):
#     """Dice (F1) pour chaque classe à partir de la matrice de confusion."""
#     tp = np.diag(cm)
#     fp = cm.sum(axis=0) - tp
#     fn = cm.sum(axis=1) - tp
#     denom = 2 * tp + fp + fn
#     dice = np.where(denom > 0, 2 * tp / denom, np.nan)
#     return dice


# def mean_dice(pred, gt, num_classes, ignore_index=None):
#     """Mean Dice (mDice)"""
#     cm = confusion_matrix(pred, gt, num_classes, ignore_index)
#     dice = dice_per_class(cm)
#     return np.nanmean(dice)
