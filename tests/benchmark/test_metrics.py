"""unit-tests for the metrics module"""

import numpy as np
import pytest

from src.benchmark.metrics import compute_evaluation_maps_one_label


@pytest.mark.parametrize(
    "gt, pred, label, expected_tp, expected_fn, expected_fp",
    [
        # ------------------------------------------------------------------
        # Simple 2×2 case – label present in both gt and pred
        # ------------------------------------------------------------------
        (
            np.array([[0, 2], [2, 1]]),  # ground truth
            np.array([[0, 2], [0, 2]]),  # prediction
            1,  # label we care about
            np.array([[False, False], [False, False]]),  # TP
            np.array([[False, False], [False, True]]),  # FN
            np.array([[False, False], [False, False]]),  # FP
        ),
        (
            np.array([[0, 2], [2, 1]]),  # ground truth
            np.array([[0, 2], [0, 2]]),  # prediction
            2,  # label we care about
            np.array([[False, True], [False, False]]),  # TP
            np.array([[False, False], [True, False]]),  # FN
            np.array([[False, False], [False, True]]),  # FP
        ),
        # ------------------------------------------------------------------
        # All‑negative case – label does not appear anywhere
        # ------------------------------------------------------------------
        (
            np.zeros((3, 3), dtype=int),
            np.zeros((3, 3), dtype=int),
            2,
            np.zeros((3, 3), dtype=bool),  # TP
            np.zeros((3, 3), dtype=bool),  # FN
            np.zeros((3, 3), dtype=bool),  # FP
        ),
        # ------------------------------------------------------------------
        # All‑positive case – everything is the target label
        # ------------------------------------------------------------------
        (
            np.full((2, 2), 5, dtype=int),
            np.full((2, 2), 5, dtype=int),
            5,
            np.full((2, 2), True, dtype=bool),  # TP
            np.full((2, 2), False, dtype=bool),  # FN
            np.full((2, 2), False, dtype=bool),  # FP
        ),
    ],
)
def test_compute_evaluation_maps_one_label(
    gt, pred, label, expected_tp, expected_fn, expected_fp
):
    """Validate that TP/FN/FP masks match the reference implementation."""
    result = compute_evaluation_maps_one_label(gt, pred, label)

    # Ensure the dictionary contains exactly the three expected keys.
    assert set(result.keys()) == {"TP", "FN", "FP"}

    # Compare each mask element‑wise.
    np.testing.assert_array_equal(result["TP"], expected_tp, err_msg="TP mask mismatch")
    np.testing.assert_array_equal(result["FN"], expected_fn, err_msg="FN mask mismatch")
    np.testing.assert_array_equal(result["FP"], expected_fp, err_msg="FP mask mismatch")


def test_invalid_inputs_raise():
    """Check that clearly malformed inputs raise a helpful exception."""
    # Mismatched shapes that cannot be broadcast.
    gt = np.zeros((2, 2))
    pred = np.zeros((3, 3))

    with pytest.raises(ValueError):
        compute_evaluation_maps_one_label(gt, pred, label=0)
