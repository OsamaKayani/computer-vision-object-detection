import pytest
from detection_metrics import Box, iou, non_max_suppression


def test_identical_boxes_have_full_overlap():
    box = Box(0,0,10,10)
    assert iou(box,box) == 1


def test_disjoint_boxes_have_no_overlap():
    assert iou(Box(0,0,1,1),Box(2,2,3,3)) == 0


def test_nms_keeps_confident_box():
    result = non_max_suppression([Box(0,0,10,10,.9),Box(1,1,9,9,.5)])
    assert len(result) == 1 and result[0].confidence == .9

