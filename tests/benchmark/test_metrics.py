"""unit-tests for the metrics.py module"""

import unittest
from dataclasses import dataclass

import numpy as np
import pytest

from src.benchmark.match import MatchMaps
from src.benchmark.metrics import (
    MatchResult,
    MatchResultOneLabel,
    compute_iou,
    compute_per_label_f1score,
    compute_pixelwise_accuracy,
    compute_precision,
    compute_recall,
)


@dataclass
class FakeMatchMaps(MatchMaps):
    """Mock of the MatchMaps class used for testing"""

    def __init__(self, tp: list[list[bool]], fp: list[list[bool]], fn: list[list[bool]]):
        super().__init__(np.array(tp), np.array(fp), np.array(fn))


class TestMatchResultOneLabel(unittest.TestCase):
    """Test MatchResultOneLabel class"""

    def test_initialization(self) -> None:
        """Test initialization of MatchResultOneLabel"""
        label_result = MatchResultOneLabel()
        self.assertEqual(label_result.tp, 0)
        self.assertEqual(label_result.fp, 0)
        self.assertEqual(label_result.fn, 0)

    def test_update(self) -> None:
        """Test updating MatchResultOneLabel with MatchMaps"""
        label_result = MatchResultOneLabel()
        match_maps = FakeMatchMaps(
            tp=[[True, False, False], [False, True, False]],
            fp=[[False, False, True], [False, False, True]],
            fn=[[True, False, False], [False, False, False]],
        )
        label_result.update(match_maps)
        self.assertEqual(label_result.tp, 2)
        self.assertEqual(label_result.fp, 2)
        self.assertEqual(label_result.fn, 1)


class TestMatchResult(unittest.TestCase):
    """Test MatchResult class"""

    def test_initialization(self) -> None:
        """Test initialization of MatchResult"""
        match_result = MatchResult(nb_labels=3)
        self.assertEqual(len(match_result.match_per_label), 3)
        self.assertEqual(match_result.nb_pixels, 0)

    def test_update_score_one_label(self) -> None:
        """Test updating score for one label"""
        match_result = MatchResult(nb_labels=3)
        match_maps = FakeMatchMaps(
            tp=[[True, False, False], [False, True, False]],
            fp=[[False, False, True], [False, False, True]],
            fn=[[True, False, False], [False, False, False]],
        )
        match_result.update_score_one_label(match_maps, label=1)
        self.assertEqual(match_result.match_per_label[1].tp, 2)
        self.assertEqual(match_result.match_per_label[1].fp, 2)
        self.assertEqual(match_result.match_per_label[1].fn, 1)

    def test_update_nb_pixels(self) -> None:
        """Test updating the number of pixels"""
        match_result = MatchResult(nb_labels=3)
        match_result.update_nb_pixels(100)
        self.assertEqual(match_result.nb_pixels, 100)

    def test_invalid_label(self) -> None:
        """Test updating score with an invalid label"""
        match_result = MatchResult(nb_labels=3)
        match_maps = FakeMatchMaps(
            tp=[[True, False, False], [False, True, False]],
            fp=[[False, False, True], [False, False, True]],
            fn=[[True, False, False], [False, False, False]],
        )
        with self.assertRaises(IndexError):
            match_result.update_score_one_label(match_maps, label=4)


class TestComputePixelwiseAccuracy(unittest.TestCase):
    """test computation of segmentation accuracy"""

    def test_compute_accuracy(self) -> None:
        """Test computing pixelwise accuracy"""
        match_result = MatchResult(nb_labels=2)
        match_maps_1 = FakeMatchMaps(
            tp=[[True, False, False], [False, True, False]],
            fp=[[False, False, True], [False, False, True]],
            fn=[[True, False, False], [False, False, False]],
        )
        match_maps_2 = FakeMatchMaps(
            tp=[[False, True, False], [False, False, True]],
            fp=[[True, False, False], [False, False, False]],
            fn=[[False, False, True], [False, False, False]],
        )
        match_result.update_score_one_label(match_maps_1, label=0)
        match_result.update_score_one_label(match_maps_2, label=1)
        match_result.update_nb_pixels(6)
        accuracy = compute_pixelwise_accuracy(match_result)
        self.assertAlmostEqual(accuracy, 2 / 3)

    def test_no_pixels_processed(self) -> None:
        """Test computing accuracy with no pixels processed"""
        match_result = MatchResult(nb_labels=2)
        with self.assertRaises(ValueError):
            compute_pixelwise_accuracy(match_result)


@pytest.mark.parametrize(
    "tp, fp, fn, expected_iou,",
    [
        (5, 6, 3, 5 / (5 + 6 + 3)),  # general case
        (0, 6, 0, 0),  # no ground_truth, but some predictions
        (0, 0, 0, 1),  # no ground_truth, no predictions
    ],
)
def test_compute_iou(tp: int, fp: int, fn: int, expected_iou: float) -> None:
    """test the computation of IoU from tp, fp and fn"""
    assert compute_iou(tp, fp, fn) == expected_iou, f"error computing IoU for {tp=}, {fp=}, {fn=}"


@pytest.mark.parametrize(
    "tp, fn, expected_recall,",
    [
        (5, 3, 5 / (5 + 3)),  # general case
        (0, 0, np.nan),  # no ground_truth
    ],
)
def test_compute_recall(tp: int, fn: int, expected_recall: float) -> None:
    """test the computation of recall from tp and fn"""
    if np.isnan(expected_recall):
        assert compute_recall(tp, fn) is np.nan, f"error computing recall  for {tp=}, {fn=}"
    else:
        assert compute_recall(tp, fn) == expected_recall, (
            f"error computing recall  for {tp=}, {fn=}"
        )


@pytest.mark.parametrize(
    "tp, fp, expected_precision,",
    [
        (5, 6, 5 / (5 + 6)),  # general case
        (0, 6, 0),  # no ground_truth
        (0, 0, np.nan),  # no ground_truth, no pred
    ],
)
def test_compute_precision(tp: int, fp: int, expected_precision: float) -> None:
    """test the computation of precision from tp and fp"""
    if expected_precision is np.nan:
        assert compute_precision(tp, fp) is np.nan, f"error computing precision  for {tp=}, {fp=}"
    else:
        assert compute_precision(tp, fp) == expected_precision, (
            f"error computing precision for {tp=}, {fp=}"
        )


@dataclass
class FakeMatchResultOneLabel(MatchResultOneLabel):
    """Fake version of MatchResultOneLabel used for tests"""

    tp: int  # True Positives
    fp: int  # False Positives
    fn: int  # False Negatives


class FakeMatchResult(MatchResult):
    """Matching results for all labels"""

    def __init__(self, *match_result_one_label: FakeMatchResultOneLabel, nb_pixels: int):
        self.match_per_label = list(match_result_one_label)
        self.nb_pixels = nb_pixels


# test f1-score computation for a semantic segmentation problem using 2 different labels
# F1 scores corresponding to FakeMatchResultOneLabel(5,6,4), FakeMatchResultOneLabel(4,7,5)
F1 = 2 * (5 / (5 + 4) * (5 / (5 + 6))) / (5 / (5 + 4) + (5 / (5 + 6)))
F2 = 2 * (4 / (4 + 5) * (4 / (4 + 7))) / (4 / (4 + 5) + (4 / (4 + 7)))


@pytest.mark.parametrize(
    "match_result_label_1, match_result_label_2, nb_pixels, expected_f1_scores",
    [
        (
            FakeMatchResultOneLabel(5, 6, 4),
            FakeMatchResultOneLabel(4, 7, 5),
            15,
            [F1, F2],
        ),  # general case
        (
            # Precision is 0 as there are no false postives, recall is 0 as there are no
            # true positives -> corresponding f1-score is 0
            FakeMatchResultOneLabel(0, 0, 4),
            # Recall is not defined -> corresponding f1-score is nan
            FakeMatchResultOneLabel(0, 7, 0),
            15,
            [0.0, np.nan],
        ),
    ],
)
def test_compute_per_label_f1score(
    match_result_label_1: FakeMatchResultOneLabel,
    match_result_label_2: FakeMatchResultOneLabel,
    nb_pixels: int,
    expected_f1_scores: list[float],
) -> None:
    """
    test the computation of f1-scores from the matching results
    of a 2-labels segmentation problem
    """
    fake_match_result = FakeMatchResult(
        match_result_label_1, match_result_label_2, nb_pixels=nb_pixels
    )
    f1_scores = compute_per_label_f1score(fake_match_result)

    assert f1_scores == expected_f1_scores, (
        f"error computing precision  for {match_result_label_1=}, {match_result_label_2=}"
    )


if __name__ == "__main__":
    pytest.main()
