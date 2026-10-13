"""Vẽ box đúng (xanh lá) và box dự đoán (đỏ = sai, vàng = khớp) lên ảnh test.

python -m vlmlab.viz --pred work/e1.jsonl --out report/figures/e1 --n 8
python -m vlmlab.viz --split train --gt-only --out report/figures/train_gt --n 6
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, _match, iou_xyxy, parse_output

COLORS = {"gt": (40, 220, 60), "tp": (255, 210, 0), "fp": (235, 40, 40)}


def _px(b, w, h):
    return [b[0] / 1000 * w, b[1] / 1000 * h, b[2] / 1000 * w, b[3] / 1000 * h]


def draw(entry, pred, data="data"):
    im = Image.open(Path(data) / entry["image_path"]).convert("RGB")
    d = ImageDraw.Draw(im)
    w, h = im.size
    gt = _gt_boxes(entry)
    for g in gt:
        d.rectangle(_px(g["bbox"], w, h), outline=COLORS["gt"], width=2)
        x1, y1 = _px(g["bbox"], w, h)[:2]
        d.text((x1 + 2, y1 + 1), g["label"], fill=COLORS["gt"])
    for p in pred or []:
        ok = any(g["label"] == p["label"] and iou_xyxy(g["bbox"], p["bbox"]) >= 0.5 for g in gt)
        c = COLORS["tp" if ok else "fp"]
        bb = _px(p["bbox"], w, h)
        d.rectangle(bb, outline=c, width=2)
        d.text((bb[0] + 2, bb[3] - 11), p["label"], fill=c)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--gt-only", action="store_true")
    a = ap.parse_args()
    entries = read_jsonl(f"data/splits/{a.split}.jsonl")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.gt_only:
        for e in [e for e in entries if e["objects"]][:: max(1, len(entries) // a.n)][:a.n]:
            draw(e, []).save(out / f"{e['image_id']}.jpg", quality=88)
        return
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(a.pred)}
    scored = []
    for e in entries:
        pred = parse_output(preds.get(e["image_id"])) or []
        per, _ = _match(_gt_boxes(e), pred)
        tp, fp, fn = (sum(v[k] for v in per.values()) for k in range(3))
        scored.append((tp, fp, fn, e, pred))
    good = sorted([s for s in scored if s[0] > 0], key=lambda s: (-(s[0] - s[1] - s[2]), s[3]["image_id"]))
    bad = sorted([s for s in scored if s[1] + s[2] > 0], key=lambda s: (-(s[1] + s[2]), s[3]["image_id"]))
    def one_per_session(group, k):  # frame liền kề gần trùng nhau → mỗi phiên quay lấy một ảnh
        seen, out = set(), []
        for s in group:
            if s[3]["session"] not in seen:
                seen.add(s[3]["session"])
                out.append(s)
        return out[:k]

    half = a.n // 2
    lines = []
    for tag, group in (("dung", one_per_session(good, half)), ("sai", one_per_session(bad, a.n - half))):
        for tp, fp, fn, e, pred in group:
            name = f"{tag}_{e['image_id']}.jpg"
            draw(e, pred).save(out / name, quality=88)
            lines.append(f"{name}\tTP={tp} FP={fp} FN={fn}\tGT={[o['label'] for o in e['objects']]}"
                         f"\tPRED={[p['label'] for p in pred]}")
    (out / "captions.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
