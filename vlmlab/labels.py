"""Nhãn YOLO → JSON grounding (thang 0–1000 hoặc pixel), prompt, và hàm reward IoU (E5).

Python thuần, không cần GPU: test được ở vlmlab/tests/.
"""
import json

from scripts.labcommon import CLASSES
from scripts.score import IOU_THR, iou_xyxy, parse_output

PROMPT_NORM1000 = (
    "Detect every road damage in this dashcam image. Classes: pothole, crack, manhole. "
    "Return ONLY a JSON list, no other text, in the form "
    '[{"label": "pothole", "bbox_2d": [x1, y1, x2, y2]}]. '
    "Coordinates are integers on a 0-1000 scale relative to the image width and height "
    "(0,0 = top-left, 1000,1000 = bottom-right). Return [] if there is no damage."
)

PROMPT_PIXEL = (
    "Detect every road damage in this dashcam image of size {w}x{h} pixels. "
    "Classes: pothole, crack, manhole. "
    "Return ONLY a JSON list, no other text, in the form "
    '[{{"label": "pothole", "bbox_2d": [x1, y1, x2, y2]}}]. '
    "Coordinates are integer pixel positions in the {w}x{h} image "
    "(0,0 = top-left). Return [] if there is no damage."
)


def make_prompt(coord="norm1000", width=640, height=360):
    if coord == "norm1000":
        return PROMPT_NORM1000
    return PROMPT_PIXEL.format(w=width, h=height)


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    xc, yc, w, h = yolo
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    x1, y1 = (xc - w / 2) * sx, (yc - h / 2) * sy
    x2, y2 = (xc + w / 2) * sx, (yc + h / 2) * sy
    x1, x2 = (min(max(round(v), 0), sx) for v in (x1, x2))
    y1, y2 = (min(max(round(v), 0), sy) for v in (y1, y2))
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def box_to_pixel(bbox_2d, width, height):
    x1, y1, x2, y2 = bbox_2d
    return [x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height]


def reward_iou(completion, gt_objects):
    """Reward E5: 0 nếu JSON hỏng; 0.2 + 0.8·ΣIoU(cặp khớp)/max(n_gt, n_pred, 1).

    gt_objects: list {"label", "yolo"} như trong splits/*.jsonl. Toạ độ thang 0–1000.
    """
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    gt = []
    for o in gt_objects:
        b = yolo_to_box(o["label"], o["yolo"])
        gt.append({"label": b["label"], "bbox": tuple(b["bbox_2d"])})
    if not gt and not pred:
        return 1.0
    total = 0.0
    for lab in set(CLASSES) | {p["label"] for p in pred}:
        g = [x["bbox"] for x in gt if x["label"] == lab]
        p = [x["bbox"] for x in pred if x["label"] == lab]
        pairs = sorted(((iou_xyxy(a, b), i, j) for i, a in enumerate(g)
                        for j, b in enumerate(p)), reverse=True)
        used_g, used_p = set(), set()
        for v, i, j in pairs:
            if v < IOU_THR:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            total += v
    return 0.2 + 0.8 * total / max(len(gt), len(pred), 1)
