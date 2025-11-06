"""
This module contains useful metrics to evaluate a semantic segmentation model
"""

from typing import List

import numpy as np

from src.benchmark.match import MatchMaps


class MatchResultOneLabel:
    """
    store the number of pixels detected as True Positives, False Positives and False Negatives
    for a specific label
    """

    tp: int  # True Positives
    fp: int  # False Positives
    fn: int  # False Negatives

    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, match_maps: MatchMaps) -> None:
        """Update True Positive, False Positive and False negative
        metrics from input matching maps (aggregation)
        """
        self.tp += np.sum(match_maps.tp)
        self.fp += np.sum(match_maps.fp)
        self.fn += np.sum(match_maps.fn)

    def __str__(self) -> str:
        return f"(tp={self.tp}, fp={self.fp}, fn={self.fn})"


class MatchResult:
    """Matching results for all labels

    Aggregation for each label of the matching results associated to a set of image
    """

    match_per_label: List[MatchResultOneLabel]
    nb_pixels: int  # total number of pixels processed

    def __init__(self, nb_labels: int) -> None:
        self.nb_pixels = 0
        self.match_per_label = []

        for _ in range(nb_labels):
            self.match_per_label.append(MatchResultOneLabel())

    def update_score_one_label(self, match_maps: MatchMaps, label: int) -> None:
        """Update score (True Positive, False Positive, False Negative)
            associated to label 'label'

        Args:
            match_maps (MatchMaps): matching maps (masks) associated to
                                    (True Positive, False Positive, False Negative)
                                    for the label of interest
            label (int): label of interest

        Raises:
            IndexError: label out of bound
        """
        if label >= len(self.match_per_label) or label < 0:
            raise IndexError("label invalid!")

        self.match_per_label[label].update(match_maps)

    def update_nb_pixels(self, nb_pixels: int) -> None:
        """Update total number of pixels processed"""
        self.nb_pixels += nb_pixels

    def __str__(self) -> str:
        text = f"MatchResult(nb_pix = {self.nb_pixels}): "

        for l, m in enumerate(self.match_per_label):
            text += f"label {l}: {str(m)} |"

        return text


def compute_iou(tp: int, fp: int, fn: int) -> float:
    """compute Intersection over Union score

    Args:
        tp (int): number of True Positives
        fp (int): number of False Positives
        fn (int): number of False Negatives

    Returns:
        float: IoU Score
    """
    if tp + fn + fp > 0:
        return tp / (tp + fn + fp)

    # label not present in ground-truth and predictions
    if fp == 0:
        return 1.0  # predictions agree with ground-truth

    return np.nan  # 0 / 0


def compute_per_label_iou(match_result: MatchResult) -> List[float]:
    """Compute the per label intersection over union scores (IoU)

    Args:
        match (MatchResult): matching results containing the numbers of True Positives,
                             False Positives and False Negatives for each label

    Returns:
        List[float]: list of IoU scores
    """
    res = []
    for m in match_result.match_per_label:
        res.append(compute_iou(m.tp, m.fp, m.fn))

    return res


def compute_recall(tp: int, fn: int) -> float:
    """compute the recall score

    Args:
        tp (int): number of True Positives
        fn (int): number of False Negatives

    Returns:
        float: recall score
    """
    if tp + fn > 0:
        return tp / (tp + fn)

    return np.nan  # label not present in ground-truth


def compute_per_label_recall(match_result: MatchResult) -> List[float]:
    """Compute the per label recall scores

    Args:
        match (MatchResult): matching results containing the numbers of True Positives,
                             False Positives, and False Negatives for each label

    Returns:
        List[float]: list of recall scores
    """
    res = []
    for m in match_result.match_per_label:
        res.append(compute_recall(m.tp, m.fn))

    return res


def compute_precision(tp: int, fp: int) -> float:
    """compute the precision score

    Args:
        tp (int): number of True Positives
        fp (int): number of False Positives

    Returns:
        float: precision score
    """
    if tp + fp > 0:
        return tp / (tp + fp)

    return np.nan  # label not present in ground-truth and pred (fp == 0, tp == 0)


def compute_per_label_precision(match_result: MatchResult) -> List[float]:
    """Compute the per label precision scores

    Args:
        match (MatchResult): matching results containing the numbers of True Positives,
                             False Positives, and False Negatives for each label

    Returns:
        List[float]: list of precision scores
    """
    res = []
    for m in match_result.match_per_label:
        if m.tp + m.fp > 0:
            res.append(m.tp / (m.tp + m.fp))
        else:
            # label not present in ground-truth and never predicted
            # 1 : because the model never predicted this label, it agrees with ground-truth -> 1
            res.append(1)

    return res


def compute_per_label_f1score(match_result: MatchResult) -> List[float]:
    """Compute the per label f1 scores

    Args:
        match (MatchResult): matching results containing the numbers of True Positives,
                             False Positives, and False Negatives for each label

    Returns:
        List[float]: list of f1 scores
    """

    recall_scores = compute_per_label_recall(match_result)
    precision_scores = compute_per_label_precision(match_result)

    res = []

    for r, p in zip(recall_scores, precision_scores):
        f1_score = 2 * r * p / (r + p) if ~np.isnan(r) and ~np.isnan(p) and p + r > 0 else np.nan
        res.append(f1_score)

    return res


def compute_pixelwise_accuracy(match_result: MatchResult) -> float:
    """compute the accuracy

    Args:
        match (MatchResult): matching results containing the numbers of True Positives,
                             False Positives, and False Negatives for each label

    Returns:
        float: accuracy
    """
    if match_result.nb_pixels == 0:
        raise ValueError("Invalid input match_result : no pixels processed")

    correct_matches = 0
    for m in match_result.match_per_label:
        correct_matches += m.tp

    return correct_matches / match_result.nb_pixels
