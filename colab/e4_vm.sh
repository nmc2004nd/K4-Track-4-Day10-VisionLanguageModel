# chạy trên VM mới (VM cũ bị thu hồi giữa E4): cài đặt, rồi train E4 = E3 (setup tốt nhất) với 10% ảnh train
set -o pipefail
pip install -q -r requirements-colab.txt && pip uninstall -y -q torchao
python scripts/prepare_data.py --zip https://www.kaggle.com/api/v1/datasets/download/lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes || { echo "E4_ABORT data"; exit 1; }
python -m vlmlab.run train --setup e3 --out work/e4_data_0.1 --train-fraction 0.1 && \
python -m vlmlab.run infer --split test --adapter work/e4_data_0.1 --out work/e4_data_0.1.jsonl && echo E4_EXIT=0 || echo E4_EXIT=1
