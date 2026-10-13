#!/bin/bash
# xin VM T4 (thử lại mỗi 15 phút khi hết hạn mức), dựng môi trường, chạy e4_vm.sh nền, rồi fetch_e4.sh tải + chấm
C=$(cd $(dirname $0) && pwd); cd $C/..
S=$(mktemp -d)
for i in $(seq 1 24); do
  timeout 300 colab new -s lab --gpu T4 2>&1 | grep -q "Service Unavailable" || { colab sessions 2>&1 | grep -q lab && break; }
  echo "[rerun] $(date +%H:%M) chưa xin được T4 (lần $i)"; sleep 900
done
colab sessions 2>&1 | grep -q lab || { echo "E4_ABORT không có VM"; exit 1; }
echo "[rerun] $(date +%H:%M) có VM"
tar czf $S/code.tgz scripts vlmlab configs requirements-colab.txt colab/e4_vm.sh
for i in 1 2 3 4 5; do colab upload -s lab $S/code.tgz /content/code.tgz 2>&1 | grep -q Uploaded && break; sleep 10; done
$C/rsh 'tar xzf /content/code.tgz -C /content/lab && cp colab/e4_vm.sh e4.sh && (nohup bash e4.sh > e4.log 2>&1 &) && echo started'
bash $C/fetch_e4.sh
