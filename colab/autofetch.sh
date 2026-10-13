#!/bin/bash
# chạy ở máy local: theo dõi pipe.log trên VM; train xong → tải adapter; infer xong → tải dự đoán + chấm
# autofetch.sh e2 e3
C=$(cd $(dirname $0) && pwd); R=$(cd $C/.. && pwd); cd $R
dl(){ for i in 1 2 3 4 5; do colab download -s lab /content/lab/$1 $2 >/dev/null 2>&1 && [ -s $2 ] && return 0; sleep 10; done; echo "tải lỗi $1"; return 1; }
for E in "$@"; do
  got_adapter=0
  while true; do
    st=$($C/rsh 'grep -E "^\[pipe|PIPE_EXIT" pipe.log 2>/dev/null | tail -3; tail -1 work/'$E'/train_log.jsonl 2>/dev/null | cut -c1-60' 2>/dev/null)
    if [ $got_adapter = 0 ] && echo "$st" | grep -q "train $E done"; then
      mkdir -p work/$E
      for f in adapter_config.json adapter_model.safetensors run.json train_log.jsonl; do dl work/$E/$f work/$E/$f; done
      $C/rsh "test -f work/$E/projector.pt && echo HAS_PROJ" | grep -q HAS_PROJ && dl work/$E/projector.pt work/$E/projector.pt
      echo "$(date +%H:%M) ADAPTER $E $(cat work/$E/run.json | tr -d '\n ')"; got_adapter=1
    fi
    if echo "$st" | grep -q "infer $E done"; then
      dl work/$E.jsonl work/$E.jsonl
      python - $E <<'PY'
import json, subprocess, sys
e = sys.argv[1]; r = json.load(open(f"work/{e}/run.json"))
subprocess.run(["python", "scripts/score.py", "--pred", f"work/{e}.jsonl", "--name", e.upper(), "--out", f"report/evidence/{e}.json",
  "--meta", f"trainable_params={r['trainable_params']}", "--meta", f"peak_vram_gb={float(r['peak_vram_gb'])}",
  "--meta", f"train_minutes={float(r['train_minutes'])}"], check=True)
PY
      cp work/$E/train_log.jsonl report/logs/${E}_train_log.jsonl; cp work/$E/run.json report/logs/${E}_run.json
      echo "$(date +%H:%M) SCORED $E"; break
    fi
    if echo "$st" | grep -qE "FAILED|PIPE_EXIT=1"; then echo "$(date +%H:%M) PIPE FAILED: $st"; exit 1; fi
    if [ -z "$st" ] && ! timeout 60 colab sessions 2>/dev/null | grep -q lab; then echo "$(date +%H:%M) VM GONE"; exit 2; fi
    sleep 120
  done
done
echo AUTOFETCH_DONE
