import json

import pytest

from scripts.score import _gt_boxes, iou_xyxy, parse_output
from vlmlab.labels import box_to_pixel, objects_to_answer, reward_iou, yolo_to_box

OBJS = [
    {"label": "crack", "yolo": [0.5, 0.8, 0.2, 0.1]},
    {"label": "pothole", "yolo": [0.25, 0.5, 0.1, 0.2]},
]


def test_yolo_to_box_norm1000():
    assert yolo_to_box("crack", [0.5, 0.8, 0.2, 0.1]) == {"label": "crack", "bbox_2d": [400, 750, 600, 850]}


def test_yolo_to_box_clamps():
    assert yolo_to_box("crack", [0.02, 0.98, 0.1, 0.1])["bbox_2d"] == [0, 930, 70, 1000]


def test_yolo_to_box_pixel():
    assert yolo_to_box("crack", [0.5, 0.5, 0.5, 0.5], "pixel", 640, 360)["bbox_2d"] == [160, 90, 480, 270]


def test_answer_sorted_and_compact():
    ans = objects_to_answer(OBJS)
    assert ans == '[{"label":"pothole","bbox_2d":[200,400,300,600]},{"label":"crack","bbox_2d":[400,750,600,850]}]'
    assert objects_to_answer([]) == "[]"


def test_answer_matches_scorer_gt():
    # đáp án huấn luyện phải khớp GT của trọng tài (IoU ~ 1)
    entry = {"objects": OBJS}
    gt = {g["label"]: g["bbox"] for g in _gt_boxes(entry)}
    for p in parse_output(objects_to_answer(OBJS)):
        assert iou_xyxy(p["bbox"], gt[p["label"]]) > 0.98


def test_roundtrip_pixel():
    b = yolo_to_box("pothole", [0.25, 0.5, 0.1, 0.2])["bbox_2d"]
    assert box_to_pixel(b, 640, 360) == pytest.approx([128, 144, 192, 216])


GT = [{"label": "pothole", "yolo": [0.25, 0.5, 0.1, 0.2]}]
PERFECT = json.dumps([{"label": "pothole", "bbox_2d": [200, 400, 300, 600]}])


def test_reward_perfect():
    assert reward_iou(PERFECT, GT) == pytest.approx(1.0)


def test_reward_broken_json():
    assert reward_iou('[{"label": "pothole", "bbox_2d": [200, 400', GT) == 0.0
    assert reward_iou("không có gì", GT) == 0.0


def test_reward_wrong_class():
    wrong = PERFECT.replace("pothole", "manhole")
    assert reward_iou(wrong, GT) == pytest.approx(0.2)


def test_reward_extra_box_penalised():
    extra = json.dumps([{"label": "pothole", "bbox_2d": [200, 400, 300, 600]},
                        {"label": "crack", "bbox_2d": [0, 0, 100, 100]}])
    assert reward_iou(extra, GT) == pytest.approx(0.2 + 0.8 * 1 / 2)


def test_reward_empty():
    assert reward_iou("[]", []) == 1.0
    assert reward_iou("[]", GT) == pytest.approx(0.2)


def test_reward_giant_box_low():
    giant = json.dumps([{"label": "pothole", "bbox_2d": [0, 0, 1000, 1000]}])
    assert reward_iou(giant, GT) == pytest.approx(0.2)
