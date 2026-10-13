# chạy lại an toàn: bước nào xong thì bỏ qua
set -u
log(){ echo "[pipe $(date +%H:%M:%S)] $*"; }
infer(){ # name adapter
  n=$(wc -l < work/$1.jsonl 2>/dev/null || echo 0)
  if [ "$n" -lt 311 ]; then log "infer $1"; python -m vlmlab.run infer --split test --out work/$1.jsonl ${2:+--adapter $2} || return 1; fi
  log "infer $1 done"
}
train(){ # name setup extra...
  name=$1; setup=$2; shift 2
  [ -f work/$name/run.json ] || { log "train $name"; python -m vlmlab.run train --setup $setup --out work/$name "$@" || return 1; }
  log "train $name done"
}
for job in "$@"; do
  case $job in
    e0) infer e0 ;;
    e1|e2|e3) train $job $job && infer $job work/$job ;;
    *) eval "$job" ;;
  esac || { log "FAILED $job"; echo PIPE_EXIT=1; exit 1; }
done
echo PIPE_EXIT=0
