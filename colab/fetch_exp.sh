#!/bin/bash
# tải kết quả một thí nghiệm từ VM về work/ (gitignored) rồi chấm: fetch_exp.sh e1
set -e
E=$1; S=$(dirname $0); R=$(cd $S/.. && pwd)
cd $R; mkdir -p work/$E
dl(){ for i in 1 2 3 4 5; do colab download -s lab /content/lab/$1 $2 >/dev/null 2>&1 && [ -s $2 ] && return 0; sleep 10; done; echo "tải lỗi $1"; return 1; }
for f in adapter_config.json adapter_model.safetensors run.json train_log.jsonl; do dl work/$E/$f work/$E/$f; done
$S/rsh "test -f work/$E/projector.pt && echo HAS_PROJ" | grep -q HAS_PROJ && dl work/$E/projector.pt work/$E/projector.pt || true
dl work/$E.jsonl work/$E.jsonl
python - $E <<'PY'
import json, subprocess, sys
e = sys.argv[1]; r = json.load(open(f"work/{e}/run.json"))
subprocess.run(["python", "scripts/score.py", "--pred", f"work/{e}.jsonl", "--name", e.upper(), "--out", f"report/evidence/{e}.json",
  "--meta", f"trainable_params={r['trainable_params']}", "--meta", f"peak_vram_gb={float(r['peak_vram_gb'])}",
  "--meta", f"train_minutes={float(r['train_minutes'])}"], check=True)
PY
mkdir -p report/logs; cp work/$E/train_log.jsonl report/logs/${E}_train_log.jsonl; cp work/$E/run.json report/logs/${E}_run.json
ls -la work/$E; echo FETCH_DONE $E
