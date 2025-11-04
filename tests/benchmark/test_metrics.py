import unittest

import numpy as np

from src.benchmark.metrics import (
    MatchResult,
    MatchResultOneLabel,
    compute_pixelwise_accuracy,
)


# Mock MatchMaps class for testing
class MatchMaps:
    def __init__(self, tp, fp, fn):
        self.tp = np.array(tp)
        self.fp = np.array(fp)
        self.fn = np.array(fn)


class TestMatchResultOneLabel(unittest.TestCase):
    def test_initialization(self):
        """Test initialization of MatchResultOneLabel"""
        label_result = MatchResultOneLabel()
        self.assertEqual(label_result.tp, 0)
        self.assertEqual(label_result.fp, 0)
        self.assertEqual(label_result.fn, 0)

    def test_update(self):
        """Test updating MatchResultOneLabel with MatchMaps"""
        label_result = MatchResultOneLabel()
        match_maps = MatchMaps(
            tp=[[True, False, False], [False, True, False]],
            fp=[[False, False, True], [False, False, True]],
            fn=[[True, False, False], [False, False, False]],
        )
        label_result.update(match_maps)
        self.assertEqual(label_result.tp, 2)
        self.assertEqual(label_result.fp, 2)
        self.assertEqual(label_result.fn, 1)


class TestMatchResult(unittest.TestCase):
    def test_initialization(self):
        """Test initialization of MatchResult"""
        match_result = MatchResult(nb_labels=3)
        self.assertEqual(len(match_result.match_per_label), 3)
        self.assertEqual(match_result.nb_pixels, 0)

    def test_updateScoreOneLabel(self):
        """Test updating score for one label"""
        match_result = MatchResult(nb_labels=3)
        match_maps = MatchMaps(
            tp=[[True, False, False], [False, True, False]],
            fp=[[False, False, True], [False, False, True]],
            fn=[[True, False, False], [False, False, False]],
        )
        match_result.updateScoreOneLabel(match_maps, label=1)
        self.assertEqual(match_result.match_per_label[1].tp, 2)
        self.assertEqual(match_result.match_per_label[1].fp, 2)
        self.assertEqual(match_result.match_per_label[1].fn, 1)

    def test_update_nb_pixels(self):
        """Test updating the number of pixels"""
        match_result = MatchResult(nb_labels=3)
        match_result.update_nb_pixels(100)
        self.assertEqual(match_result.nb_pixels, 100)

    def test_invalid_label(self):
        """Test updating score with an invalid label"""
        match_result = MatchResult(nb_labels=3)
        match_maps = MatchMaps(
            tp=[[True, False, False], [False, True, False]],
            fp=[[False, False, True], [False, False, True]],
            fn=[[True, False, False], [False, False, False]],
        )
        with self.assertRaises(IndexError):
            match_result.updateScoreOneLabel(match_maps, label=4)


class TestComputePixelwiseAccuracy(unittest.TestCase):
    def test_compute_accuracy(self):
        """Test computing pixelwise accuracy"""
        match_result = MatchResult(nb_labels=2)
        match_maps_1 = MatchMaps(
            tp=[[True, False, False], [False, True, False]],
            fp=[[False, False, True], [False, False, True]],
            fn=[[True, False, False], [False, False, False]],
        )
        match_maps_2 = MatchMaps(
            tp=[[False, True, False], [False, False, True]],
            fp=[[True, False, False], [False, False, False]],
            fn=[[False, False, True], [False, False, False]],
        )
        match_result.updateScoreOneLabel(match_maps_1, label=0)
        match_result.updateScoreOneLabel(match_maps_2, label=1)
        match_result.update_nb_pixels(6)
        accuracy = compute_pixelwise_accuracy(match_result)
        self.assertAlmostEqual(accuracy, 2 / 3)

    def test_no_pixels_processed(self):
        """Test computing accuracy with no pixels processed"""
        match_result = MatchResult(nb_labels=2)
        with self.assertRaises(ValueError):
            compute_pixelwise_accuracy(match_result)


if __name__ == "__main__":
    unittest.main()
