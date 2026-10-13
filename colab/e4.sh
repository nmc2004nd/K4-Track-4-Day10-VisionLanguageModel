# E4 (data_size): đợi E1–E3 xong, chọn setup có F1 test cao nhất, train lại với 10% ảnh train
until grep -q "PIPE_EXIT" pipe.log; do sleep 60; done
grep -q "PIPE_EXIT=0" pipe.log || { echo "E4_ABORT pipeline lỗi"; exit 1; }
for e in e2 e3; do python scripts/score.py --pred work/$e.jsonl --name $e --out work/score_$e.json >/dev/null; done
best=$(python -c "
import json
f={e: json.load(open(f'work/score_{e}.json'))['micro']['f1'] for e in ('e2','e3')}
f['e1']=0.2813  # E1 chấm ở máy local (VM cũ đã bị thu hồi)
print(max(f, key=f.get))")
echo "[e4] best=$best"
python -m vlmlab.run train --setup $best --out work/e4_data_0.1 --train-fraction 0.1 && \
python -m vlmlab.run infer --split test --adapter work/e4_data_0.1 --out work/e4_data_0.1.jsonl && echo E4_EXIT=0 || echo E4_EXIT=1
