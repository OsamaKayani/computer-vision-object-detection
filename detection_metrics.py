from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    label: str = "object"

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.x1, self.y1, self.x2, self.y2)):
            raise ValueError("box coordinates must be finite")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box must have positive width and height")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass(frozen=True)
class DetectionScore:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float


def iou(left: Box, right: Box) -> float:
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = width * height
    union = left.area + right.area - intersection
    return intersection / union if union else 0.0


def non_max_suppression(boxes: Iterable[Box], threshold: float = 0.5) -> list[Box]:
    """Apply confidence-ordered, class-aware non-max suppression."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    pending = sorted(boxes, key=lambda box: box.confidence, reverse=True)
    kept: list[Box] = []
    while pending:
        winner = pending.pop(0)
        kept.append(winner)
        pending = [
            candidate
            for candidate in pending
            if candidate.label != winner.label or iou(winner, candidate) < threshold
        ]
    return kept


def score_detections(
    predictions: Iterable[Box],
    ground_truth: Iterable[Box],
    iou_threshold: float = 0.5,
) -> DetectionScore:
    """Greedily match predictions to unused ground-truth boxes of the same class."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be between 0 and 1")
    truth = list(ground_truth)
    unused = set(range(len(truth)))
    true_positives = 0
    false_positives = 0

    for prediction in sorted(predictions, key=lambda box: box.confidence, reverse=True):
        candidates = [
            (iou(prediction, truth[index]), index)
            for index in unused
            if prediction.label == truth[index].label
        ]
        overlap, match_index = max(candidates, default=(0.0, -1))
        if overlap >= iou_threshold:
            true_positives += 1
            unused.remove(match_index)
        else:
            false_positives += 1

    false_negatives = len(unused)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    return DetectionScore(true_positives, false_positives, false_negatives, precision, recall)
