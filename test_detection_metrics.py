import pytest

from detection_metrics import Box, iou, non_max_suppression, score_detections


def test_identical_boxes_have_full_overlap():
    box = Box(0, 0, 10, 10)
    assert iou(box, box) == 1


def test_disjoint_boxes_have_no_overlap():
    assert iou(Box(0, 0, 1, 1), Box(2, 2, 3, 3)) == 0


def test_nms_keeps_confident_box():
    result = non_max_suppression([Box(0, 0, 10, 10, 0.9), Box(1, 1, 9, 9, 0.5)])
    assert len(result) == 1
    assert result[0].confidence == 0.9


def test_nms_does_not_suppress_a_different_class():
    result = non_max_suppression(
        [Box(0, 0, 10, 10, 0.9, "panel"), Box(0, 0, 10, 10, 0.8, "hotspot")]
    )
    assert len(result) == 2


def test_precision_and_recall_use_one_to_one_matching():
    truth = [Box(0, 0, 10, 10, label="panel"), Box(20, 20, 30, 30, label="panel")]
    predictions = [Box(0, 0, 10, 10, 0.9, "panel"), Box(1, 1, 9, 9, 0.8, "panel")]
    score = score_detections(predictions, truth)
    assert score.true_positives == 1
    assert score.false_positives == 1
    assert score.false_negatives == 1
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(0.5)


def test_invalid_box_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        Box(2, 2, 1, 3)
