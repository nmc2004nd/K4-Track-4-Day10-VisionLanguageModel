"""So sánh hai setup bằng bootstrap ghép cặp (cùng tập ảnh lấy lại cho cả hai) trên chênh lệch F1.

python -m vlmlab.compare work/e1.jsonl work/e2.jsonl
"""
import argparse
import random

from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, _match, _prf, parse_output


def per_image(entries, pred_path):
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(pred_path)}
    rows = []
    for e in entries:
        per, _ = _match(_gt_boxes(e), parse_output(preds.get(e["image_id"])) or [])
        rows.append(tuple(sum(v[k] for v in per.values()) for k in range(3)))
    return rows


def f1(rows, idx):
    tp = sum(rows[i][0] for i in idx)
    fp = sum(rows[i][1] for i in idx)
    fn = sum(rows[i][2] for i in idx)
    return _prf(tp, fp, fn)["f1"]


def paired(entries, a, b, n_boot=2000, seed=0, by_session=False):
    ra, rb = per_image(entries, a), per_image(entries, b)
    n = len(entries)
    rng = random.Random(seed)
    groups = {}
    for i, e in enumerate(entries):
        groups.setdefault(e["session"], []).append(i)
    sess = list(groups)
    diffs = []
    for _ in range(n_boot):
        if by_session:  # lấy lại theo phiên quay (đơn vị độc lập thật sự)
            idx = [i for _ in sess for i in groups[rng.choice(sess)]]
        else:
            idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(f1(rb, idx) - f1(ra, idx))
    diffs.sort()
    full = range(n)
    return {"f1_a": f1(ra, full), "f1_b": f1(rb, full), "diff": f1(rb, full) - f1(ra, full),
            "ci95": [diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot) - 1]],
            "p_b_le_a": sum(d <= 0 for d in diffs) / n_boot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    entries = read_jsonl(f"data/splits/{a.split}.jsonl")
    for mode in (False, True):
        r = paired(entries, a.a, a.b, by_session=mode)
        print(f"{'theo phiên' if mode else 'theo ảnh  '}: F1 {r['f1_a']:.3f} → {r['f1_b']:.3f}  "
              f"Δ={r['diff']:+.3f}  CI95(Δ)=[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  P(Δ≤0)={r['p_b_le_a']:.3f}")


if __name__ == "__main__":
    main()
