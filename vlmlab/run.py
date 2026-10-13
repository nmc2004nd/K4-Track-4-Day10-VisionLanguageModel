"""Huấn luyện (QLoRA) và suy luận Qwen3-VL-4B cho grounding hư hại mặt đường.

python -m vlmlab.run inspect
python -m vlmlab.run infer --split test --out work/e0.jsonl
python -m vlmlab.run train --setup e1 --out work/e1
python -m vlmlab.run infer --split test --adapter work/e1 --out work/e1.jsonl
"""
import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
from PIL import Image

from scripts.labcommon import read_jsonl
from vlmlab.labels import make_prompt, objects_to_answer

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DATA = Path("data")
LLM_RE = r"model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_RE = r"model\.visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc(1|2))"
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")


# ---------------------------------------------------------------- model
def load_model():
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, llm_int8_skip_modules=["visual", "lm_head"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant, dtype=torch.float16, device_map={"": 0})
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    return model, processor


def user_msg(image, prompt):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}


def load_image(entry):
    return Image.open(DATA / entry["image_path"]).convert("RGB")


# ---------------------------------------------------------------- inspect
def count_params(model):
    groups = {"vit_blocks": 0, "merger": 0, "deepstack": 0, "vit_other": 0, "llm": 0, "lm_head": 0}
    for n, p in model.named_parameters():
        k = p.numel() * (2 if p.__class__.__name__ == "Params4bit" else 1)  # 4-bit gói 2 số/byte
        if "visual.merger" in n:
            groups["merger"] += k
        elif "visual.deepstack_merger_list" in n:
            groups["deepstack"] += k
        elif "visual.blocks" in n:
            groups["vit_blocks"] += k
        elif "visual" in n:
            groups["vit_other"] += k
        elif n.startswith("lm_head"):
            groups["lm_head"] += k
        else:
            groups["llm"] += k
    return groups


def cmd_inspect(a):
    model, processor = load_model()
    names = [n for n, _ in model.named_modules()]
    print("ví dụ tên module:", [n for n in names if "layers.0." in n or "blocks.0." in n][:30])
    print("LLM matches:", sum(bool(re.fullmatch(LLM_RE, n)) for n in names))
    print("ViT matches:", sum(bool(re.fullmatch(VIT_RE, n)) for n in names))
    g = count_params(model)
    print("params:", json.dumps(g), "total:", sum(g.values()))
    print("tie_word_embeddings:", getattr(model.config, "tie_word_embeddings", None),
          getattr(model.config.get_text_config(), "tie_word_embeddings", None))
    entry = read_jsonl(DATA / "splits/train.jsonl")[0]
    enc = encode_example(processor, load_image(entry), make_prompt(), objects_to_answer(entry["objects"]))
    image_token_id = model.config.image_token_id
    print("image tokens:", int((enc["input_ids"] == image_token_id).sum()), "grid:", enc["image_grid_thw"].tolist())
    ans_ids = enc["labels"][enc["labels"] != -100]
    print("decoded answer:", repr(processor.tokenizer.decode(ans_ids)))
    print("expected      :", objects_to_answer(entry["objects"]))
    print("keys:", list(enc.keys()))


# ---------------------------------------------------------------- inference
@torch.no_grad()
def cmd_infer(a):
    model, processor = load_model()
    if a.adapter:
        load_trained(model, Path(a.adapter))
    model.eval()
    entries = read_jsonl(DATA / f"splits/{a.split}.jsonl")
    if a.limit:
        entries = entries[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists() and not a.overwrite:
        done = {r["image_id"] for r in read_jsonl(out)}
    t0 = time.time()
    todo = [e for e in entries if e["image_id"] not in done]
    with open(out, "a" if done else "w", encoding="utf-8") as f:
        for i in range(0, len(todo), a.batch_size):
            batch = todo[i:i + a.batch_size]
            convs = [[user_msg(load_image(e), make_prompt(a.coord, e["width"], e["height"]))] for e in batch]
            inputs = processor.apply_chat_template(
                convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", padding=True).to(model.device)
            gen = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
            raws = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for e, raw in zip(batch, raws):
                f.write(json.dumps({"image_id": e["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[infer] {i + len(batch)}/{len(todo)}  {time.time() - t0:.0f}s  {raws[0][:120]!r}", flush=True)
    print(f"[infer] xong {len(todo)} ảnh trong {(time.time() - t0) / 60:.1f} phút", flush=True)


def load_trained(model, adapter_dir):
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(adapter_dir))
    proj = adapter_dir / "projector.pt"
    if proj.exists():
        sd = torch.load(proj, map_location="cpu")
        params = dict(model.named_parameters())
        for n, t in sd.items():
            params[n].data.copy_(t.to(params[n].device))  # fp32 → fp16 khi copy
        print(f"[load] nạp lại {len(sd)} tensor projector ({sum(t.numel() for t in sd.values()) / 1e6:.1f}M)")
    return model


# ---------------------------------------------------------------- training data
def encode_example(processor, image, prompt, answer):
    user = user_msg(image, prompt)
    asst = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    p = processor.apply_chat_template([user], add_generation_prompt=True, **kw)
    full = processor.apply_chat_template([user, asst], add_generation_prompt=False, **kw)
    n = p["input_ids"].shape[1]
    assert torch.equal(full["input_ids"][0, :n], p["input_ids"][0]), "prompt không phải tiền tố"
    labels = full["input_ids"].clone()
    labels[:, :n] = -100
    enc = dict(full)
    enc["labels"] = labels
    return enc


def collate(batch, pad_id):
    seq_keys = {"input_ids": pad_id, "attention_mask": 0, "labels": -100, "mm_token_type_ids": 0}
    L = max(b["input_ids"].shape[1] for b in batch)
    out = {}
    for k, pad in seq_keys.items():
        if k not in batch[0]:
            continue
        rows = []
        for b in batch:
            t = b[k][0]
            rows.append(torch.cat([t, torch.full((L - t.shape[0],), pad, dtype=t.dtype)]))  # pad phải khi train
        out[k] = torch.stack(rows)
    for k in ("pixel_values", "image_grid_thw"):
        out[k] = torch.cat([b[k] for b in batch], 0)
    return out


# ---------------------------------------------------------------- setup
def apply_setup(model, setup, rank, alpha):
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()
    target = LLM_RE if setup in ("e1", "e2") else f"({LLM_RE})|({VIT_RE})"
    matched = [n for n, _ in model.named_modules() if re.fullmatch(target, n)]
    print(f"[setup] {setup}: {len(matched)} module LoRA (r={rank}, alpha={alpha})")
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, target_modules=target, bias="none")
    model = get_peft_model(model, cfg)
    if setup in ("e2", "e3"):
        for n, p in model.named_parameters():
            if any(k in n for k in PROJECTOR_KEYS):
                p.data = p.data.float()
                p.requires_grad = True
    for n, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()  # GradScaler cần master weights fp32
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    proj = sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and any(k in n for k in PROJECTOR_KEYS))
    print(f"[setup] trainable={trainable:,}  (projector={proj:,}, lora={trainable - proj:,})")
    return model, trainable, len(matched)


def save_ckpt(model, out):
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    proj = {n: p.detach().cpu() for n, p in model.named_parameters()
            if p.requires_grad and any(k in n for k in PROJECTOR_KEYS)}
    if proj:
        torch.save(proj, out / "projector.pt")


def cmd_train(a):
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    model, processor = load_model()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model, trainable, n_mod = apply_setup(model, a.setup, a.rank, a.alpha or 2 * a.rank)
    model.train()

    entries = read_jsonl(DATA / "splits/train.jsonl")
    if a.train_fraction < 1:
        entries = random.Random(a.seed).sample(entries, max(1, int(len(entries) * a.train_fraction)))
    print(f"[train] {len(entries)} ảnh train")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warm = max(1, int(0.05 * a.max_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.max_steps - warm))))
    scaler = torch.amp.GradScaler("cuda")
    pad_id = processor.tokenizer.pad_token_id

    def batches():
        rng = random.Random(a.seed)
        while True:
            order = entries[:]
            rng.shuffle(order)
            for i in range(0, len(order) - a.batch_size + 1, a.batch_size):
                yield order[i:i + a.batch_size]

    it = batches()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "w")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for step in range(1, a.max_steps + 1):
        tot = 0.0
        for _ in range(a.grad_accum):
            encs = [encode_example(processor, load_image(e),
                                   make_prompt(a.coord, e["width"], e["height"]),
                                   objects_to_answer(e["objects"], a.coord, e["width"], e["height"]))
                    for e in next(it)]
            batch = {k: v.to(model.device) for k, v in collate(encs, pad_id).items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / a.grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss không hữu hạn ở bước {step}: {loss.item()}")
            scaler.scale(loss).backward()
            tot += loss.item()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        rec = {"step": step, "loss": round(tot, 4), "grad_norm": round(float(gn), 3),
               "lr": sched.get_last_lr()[0], "min": round((time.time() - t0) / 60, 2),
               "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 10 == 0 or step == 1:
            print(f"[train] {json.dumps(rec)}", flush=True)
        if step % 100 == 0 or step == a.max_steps:
            save_ckpt(model, out)
    run = {"setup": a.setup, "rank": a.rank, "alpha": a.alpha or 2 * a.rank,
           "train_fraction": a.train_fraction, "coord": a.coord, "lora_modules": n_mod,
           "trainable_params": trainable,
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
           "train_minutes": round((time.time() - t0) / 60, 1), "max_steps": a.max_steps}
    (out / "run.json").write_text(json.dumps(run, indent=2))
    print("[train] run.json", json.dumps(run), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect")
    pi = sub.add_parser("infer")
    pi.add_argument("--split", default="test")
    pi.add_argument("--out", required=True)
    pi.add_argument("--adapter")
    pi.add_argument("--coord", default="norm1000")
    pi.add_argument("--batch-size", type=int, default=4)
    pi.add_argument("--max-new-tokens", type=int, default=256)
    pi.add_argument("--limit", type=int, default=0)
    pi.add_argument("--overwrite", action="store_true")
    pt = sub.add_parser("train")
    pt.add_argument("--setup", choices=["e1", "e2", "e3"], required=True)
    pt.add_argument("--out", required=True)
    pt.add_argument("--rank", type=int, default=16)
    pt.add_argument("--alpha", type=int, default=0)
    pt.add_argument("--lr", type=float, default=2e-4)
    pt.add_argument("--batch-size", type=int, default=2)
    pt.add_argument("--grad-accum", type=int, default=4)
    pt.add_argument("--max-steps", type=int, default=300)
    pt.add_argument("--train-fraction", type=float, default=1.0)
    pt.add_argument("--coord", default="norm1000")
    pt.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    {"inspect": cmd_inspect, "infer": cmd_infer, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
