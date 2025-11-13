"""
match predictions to ground-truth
(compute True Positives, False Positives and False Negatives pixels)
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MatchMaps:
    """
    store the masks images corresponding to
    True Positives, False Positives and False Negatives
    """

    tp: np.ndarray  # True Positives
    fp: np.ndarray  # False Positives
    fn: np.ndarray  # False Negatives


def compute_match_maps_one_label(gt: np.ndarray, pred: np.ndarray, label: int) -> MatchMaps:
    """
    Generate pixel-wise true-positive, false-negative, false-positive maps

    Args:
    gt (np.ndarray) : Ground-truth label image (shape: HxW or any broadcast-compatible shape).
    pred (np.ndarray) : Predicted label image (same shape as ``gt``).
    label (int) : the label of interest.

    Returns
    -------
    maps : MatchMaps
        container containing maps corresponding to true-positive, false-negative, false-positive
    """

    if gt.shape != pred.shape:
        raise ValueError("gt and pred must have identical shapes")

    # detect in gt and pred pixels corresponding to the target label
    gt_label_mask = gt == label
    pred_label_mask = pred == label

    # ------------------------------------------------------------------
    # 1)  True postives: correct predictions for the target lalel
    # ------------------------------------------------------------------
    tp_mask = np.logical_and(gt_label_mask, pred_label_mask)

    # ------------------------------------------------------------------
    # 2) False positives: prediction of label at a wrong location
    # ------------------------------------------------------------------
    fp_mask = np.logical_and(~gt_label_mask, pred_label_mask)

    # ------------------------------------------------------------------
    # 3)  False Negatives – a GT label exists but the model missed it
    # ------------------------------------------------------------------
    fn_mask = np.logical_and(gt_label_mask, ~pred_label_mask)

    # ------------------------------------------------------------------
    # 4)  True Negatives – both GT and prediction are background
    # ------------------------------------------------------------------
    # tn_mask = np.matmul(1 - gt_label_mask, 1 - pred_label_mask)

    # ------------------------------------------------------------------
    # Pack results
    # ------------------------------------------------------------------
    return MatchMaps(tp_mask, fp_mask, fn_mask)
