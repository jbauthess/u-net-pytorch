"""use matching results between ground-truth and predictions to compute evaluation report"""

import logging
from enum import StrEnum
from typing import List

from src.benchmark.metrics import MatchResult, compute_pixelwise_accuracy

logger = logging.getLogger()


class TestMetrics(StrEnum):
    ACCURACY = "accuracy"


def generate_report(match_result: MatchResult, metrics: List[TestMetrics]) -> None:
    """generate the report corresponding to model performances

    Args:
        match_results (List[MatchResult]): the matching results obtained for each class
        metrics (List[TestMetrics]): the metrics to include in the evaluation report

    Raises:
        NotImplementedError: _description_
    """
    for m in metrics:
        match m:
            case TestMetrics.ACCURACY:
                acc = compute_pixelwise_accuracy(match_result)
                logger.info(f"GLOBAL ACCURACY={acc}")
            case _:
                raise NotImplementedError("This metric is not available")
