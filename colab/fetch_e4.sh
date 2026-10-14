#!/bin/bash
# đợi e4.sh xong trên VM → tải adapter + dự đoán, chấm e4_data_size_0.1; giá trị 1.0 = chính setup gốc (E3)
C=$(cd $(dirname $0) && pwd); cd $C/..
D=e4_data_0.1
dl(){ for i in 1 2 3 4 5; do timeout 600 colab download -s lab /content/lab/$1 $2 >/dev/null 2>&1 && [ -s $2 ] && return 0; sleep 10; done; echo "tải lỗi $1"; return 1; }
until st=$($C/rsh 'grep -E "E4_EXIT|E4_ABORT|Error" e4.log | tail -2' 2>/dev/null); echo "$st" | grep -qE "E4_EXIT|E4_ABORT|Error"; do sleep 60; done
echo "$st"; echo "$st" | grep -q "E4_EXIT=0" || exit 1
mkdir -p work/$D
for f in adapter_config.json adapter_model.safetensors run.json train_log.jsonl projector.pt; do dl work/$D/$f work/$D/$f; done
dl work/$D.jsonl work/$D.jsonl
python - <<'PY'
import json, subprocess
def score(pred, name, out, run):
    r = json.load(open(run))
    subprocess.run(["python", "scripts/score.py", "--pred", pred, "--name", name, "--out", out,
        "--meta", f"trainable_params={r['trainable_params']}", "--meta", f"peak_vram_gb={float(r['peak_vram_gb'])}",
        "--meta", f"train_minutes={float(r['train_minutes'])}", "--meta", f"train_fraction={float(r['train_fraction'])}",
        "--meta", f"base_setup={r['setup']}"], check=True)
score("work/e4_data_0.1.jsonl", "E4 data_size=0.1", "report/evidence/e4_data_size_0.1.json", "work/e4_data_0.1/run.json")
score("work/e3.jsonl", "E4 data_size=1.0 (=E3)", "report/evidence/e4_data_size_1.0.json", "work/e3/run.json")
PY
cp work/$D/train_log.jsonl report/logs/e4_data_0.1_train_log.jsonl; cp work/$D/run.json report/logs/e4_data_0.1_run.json
echo FETCH_E4_DONE
