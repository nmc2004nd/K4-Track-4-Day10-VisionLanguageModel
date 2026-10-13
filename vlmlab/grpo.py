"""E5 (tuỳ chọn): GRPO với reward IoU, khởi tạo từ adapter SFT tốt nhất.

python -m vlmlab.grpo --init work/e2 --out work/e5
python -m vlmlab.run infer --split test --adapter work/e5 --out work/e5.jsonl
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import torch

from scripts.labcommon import read_jsonl
from scripts.score import parse_output
from vlmlab.labels import make_prompt, reward_iou
from vlmlab.run import DATA, load_image, load_model

STATS = []


def completion_text(c):
    if isinstance(c, str):
        return c
    content = c[-1]["content"]
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if p.get("type") == "text")
    return content


def iou_reward(completions, objects, **kwargs):
    texts = [completion_text(c) for c in completions]
    rewards = [reward_iou(t, o) for t, o in zip(texts, objects)]
    # theo dõi dấu hiệu reward hacking: trả "[]", JSON hỏng, box khổng lồ (> 50% khung hình)
    for t, o, r in zip(texts, objects, rewards):
        p = parse_output(t)
        STATS.append({
            "reward": r, "valid": p is not None, "empty": p == [], "n_pred": len(p or []),
            "n_gt": len(o), "giant": sum((b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) > 5e5
                                         for b in (p or [])),
        })
    return rewards


def main():
    from datasets import Dataset
    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer

    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--n-prompts", type=int, default=240)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    entries = read_jsonl(DATA / "splits/train.jsonl")
    entries = random.Random(42).sample(entries, a.n_prompts)
    ds = Dataset.from_list([{
        "prompt": [{"role": "user", "content": make_prompt()}],
        "image": load_image(e), "objects": e["objects"]} for e in entries])

    model, processor = load_model()
    model = PeftModel.from_pretrained(model, a.init, is_trainable=True)
    proj = Path(a.init) / "projector.pt"
    if proj.exists():  # projector đã SFT: nạp lại, giữ đóng băng (fp16)
        params = dict(model.named_parameters())
        for n, t in torch.load(proj, map_location="cpu").items():
            params[n].data.copy_(t.to(params[n].device))
    processor.tokenizer.padding_side = "left"

    cfg = GRPOConfig(
        output_dir=str(out / "trainer"), learning_rate=5e-6, beta=0.0, num_generations=4,
        max_completion_length=128, max_steps=a.max_steps, per_device_train_batch_size=4,
        gradient_accumulation_steps=2, fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_strategy="no",
        report_to="none", use_vllm=False, remove_unused_columns=False, seed=42,
        warmup_ratio=0.05, lr_scheduler_type="cosine")
    trainer = GRPOTrainer(model=model, reward_funcs=iou_reward, args=cfg, train_dataset=ds,
                          processing_class=processor)
    # GRPOTrainer ép tham số LoRA của model lượng tử hoá sang bf16; T4 không có bf16 → đưa về fp32
    for p in trainer.model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    trainer.model.save_pretrained(str(out))
    if proj.exists():
        shutil.copy(proj, out / "projector.pt")
    (out / "grpo_log.json").write_text(json.dumps(trainer.state.log_history, indent=1))
    (out / "reward_stats.jsonl").write_text("\n".join(json.dumps(s) for s in STATS) + "\n")
    run = {"setup": "e5-grpo", "init": a.init, "trainable_params": n_train,
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "train_minutes": round((time.time() - t0) / 60, 1), "max_steps": a.max_steps}
    (out / "run.json").write_text(json.dumps(run, indent=2))
    print("[grpo] run.json", json.dumps(run), flush=True)


if __name__ == "__main__":
    main()
