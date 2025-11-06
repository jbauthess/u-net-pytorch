"""use matching results between ground-truth and predictions to compute evaluation report"""

import logging
from enum import StrEnum
from typing import List

from src.benchmark.metrics import (
    MatchResult,
    compute_per_label_f1score,
    compute_per_label_iou,
    compute_per_label_precision,
    compute_per_label_recall,
    compute_pixelwise_accuracy,
)

logger = logging.getLogger()


class TestMetrics(StrEnum):
    """metrics that can be included in the evaluation report"""

    ACCURACY = "accuracy"
    RECALL_PER_LABEL = "recall_per_label"
    F1_SCORE_PER_LABEL = "f1_score_per_label"
    PRECISION_PER_LABEL = "precision"
    IOU_PER_LABEL = "iou_per_label"


def generate_report(match_result: MatchResult, metrics: List[TestMetrics]) -> None:
    """generate the report corresponding to model performances

    Args:
        match_results (List[MatchResult]): the matching results obtained for each label
        metrics (List[TestMetrics]): the metrics to include in the evaluation report

    Raises:
        ValueError: the desired metric is not implemented
    """
    for m in metrics:
        match m:
            case TestMetrics.ACCURACY:
                acc = compute_pixelwise_accuracy(match_result)
                logger.info("GLOBAL ACCURACY=%.3f", acc)
            case TestMetrics.RECALL_PER_LABEL:
                logger.info("RECALL PER LABEL")
                for l, r in enumerate(compute_per_label_recall(match_result)):
                    logger.info("label %s: %.3f", l, r)

            case TestMetrics.PRECISION_PER_LABEL:
                logger.info("PRECISION PER LABEL")
                for l, p in enumerate(compute_per_label_precision(match_result)):
                    logger.info("label %s: %.3f", l, p)

            case TestMetrics.F1_SCORE_PER_LABEL:
                logger.info("F1-SCORE PER LABEL")
                for l, p in enumerate(compute_per_label_f1score(match_result)):
                    logger.info("label %s: %%.3f", l, p)

            case TestMetrics.IOU_PER_LABEL:
                logger.info("IOU PER LABEL")
                for l, iou in enumerate(compute_per_label_iou(match_result)):
                    logger.info("label %s: %.3f", l, iou)

            case _:
                raise ValueError("This metric is not available")
