from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    def __post_init__(self):
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("box coordinates are inverted")


def iou(a: Box, b: Box) -> float:
    width = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    height = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    intersection = width * height
    union = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - intersection
    return intersection / union if union else 0.0


def non_max_suppression(boxes, threshold=0.5):
    pending = sorted(boxes, key=lambda box: box.confidence, reverse=True)
    kept = []
    while pending:
        winner = pending.pop(0)
        kept.append(winner)
        pending = [candidate for candidate in pending if iou(winner, candidate) < threshold]
    return kept

