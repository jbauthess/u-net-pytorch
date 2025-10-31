# seg_metrics.py
import numpy as np


def compute_evaluation_maps_one_label(
    gt: np.ndarray, pred: np.ndarray, label: int
) -> dict[str, np.ndarray]:
    """
    Generate pixel-wise true-positive, false-negative, false-positive

    Args:
    gt (np.ndarray) : Ground-truth label image (shape: HxW or any broadcast-compatible shape).
    pred (np.ndarray) : Predicted label image (same shape as ``gt``).
    label (int) : the label of interest.

    Returns
    -------
    maps : dict[str, np.ndarray]
        Boolean masks with keys ``'TP'``, ``'FN'``, ``'FP'``
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
    fp_mask = np.logical_and(1 - gt_label_mask, pred_label_mask)

    # ------------------------------------------------------------------
    # 3)  False Negatives – a GT label exists but the model missed it
    # ------------------------------------------------------------------
    fn_mask = np.logical_and(gt_label_mask, 1 - pred_label_mask)

    # ------------------------------------------------------------------
    # 4)  True Negatives – both GT and prediction are background
    # ------------------------------------------------------------------
    # tn_mask = np.matmul(1 - gt_label_mask, 1 - pred_label_mask)

    # ------------------------------------------------------------------
    # Pack results
    # ------------------------------------------------------------------
    maps = {"TP": tp_mask, "FN": fn_mask, "FP": fp_mask}

    return maps


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


def compute_pixelwise_accuracy(pred, gt, ignore_index=None) -> float:
    """Compute Pixel Accuracy (PA) :  number of valid predicted pixels / total number of pixels"""
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
