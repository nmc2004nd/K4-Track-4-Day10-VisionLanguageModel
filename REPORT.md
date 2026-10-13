Họ tên: TODO_NAME
`dataset_version`: d10-21e4070301

Môi trường: Colab T4 15 GB (điều khiển bằng `google-colab-cli` từ terminal), torch 2.11 + cu130, transformers 5.19.0, peft 0.21.2, bitsandbytes 0.50.2, trl 1.9.2. Mã của tôi: `vlmlab/` (`labels.py` nhãn/prompt/reward, `run.py` train/infer/inspect, `viz.py` ảnh, `grpo.py` E5), script điều khiển Colab trong `colab/`. Log: `report/logs/`, ảnh: `report/figures/`.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | — | — (suy luận 12,9) | 0,088 | 0,062–0,117 | 0,99 | 0,135 | 0,027 | 0,185 |
| E1 | LoRA LLM | 33.030.144 | 6,34 | 45,3 | 0,281 | 0,246–0,322 | 1,17 | 0,174 | 0,215 | 0,513 |
| E2 | + projector | 142.135.296 | 7,94 | 47,0 | 0,346 | 0,306–0,386 | 1,11 | 0,213 | 0,270 | 0,613 |
| E3 | + LoRA ViT | 148.426.752 | 8,12 | 55,1 | 0,365 | 0,322–0,408 | 1,08 | 0,258 | 0,274 | 0,648 |
| E4 | TODO_E4 |

(`parse_valid_rate`: E0 0,932 · E1 0,997 · E2 0,997 · E3 0,994. VRAM = `torch.cuda.max_memory_allocated`, thời gian = 300 bước tối ưu, batch 2 × tích luỹ 4.)

## 10 câu hỏi

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?

Đếm trực tiếp trên model đã nạp (`python -m vlmlab.run inspect`, `report/logs/inspect.txt`; tham số 4-bit nhân 2 vì mỗi byte gói 2 số):

| Khối | Tham số | Ghi chú |
|---|---|---|
| ViT: 24 block (`model.visual.blocks.*`) | 302,3M | + patch embed/pos embed 3,9M → ViT ≈ 306M không kể projector |
| Projector `visual.merger` | 27,3M | gộp 2×2 patch → 1 token, chiếu 1024·4 → 2560 |
| Projector `visual.deepstack_merger_list` (3 cái) | 81,8M | đưa đặc trưng ViT lớp 5, 11, 17 vào các lớp đầu của LLM (DeepStack) |
| LLM `model.language_model` (36 lớp + embedding) | 4.022,5M | `lm_head` buộc chung trọng số với embedding (`tie_word_embeddings=True`) |
| **Tổng** | **4.437,8M** | |

- **E1** chỉ cập nhật LoRA (r = 16) trên 252 module của LLM (36 lớp × q,k,v,o,gate,up,down): **33,03M** tham số. ViT và projector đóng băng.
- **E2** = E1 + mở toàn bộ projector (merger + deepstack, 109,1M, fp32): **142,14M** tham số, đúng bằng E1 + 109.105.152.
- **E3** = E2 + LoRA trên 96 module của ViT (24 block × qkv, proj, fc1, fc2), tổng 348 module LoRA: **148,43M** tham số (LoRA ViT thêm 6,29M).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?

`parse_valid_rate` = **0,932** (21/311 ảnh hỏng). Model gốc không bọc ```json (0/311 có fence) và đã dùng đúng thang 0–1000 (meanIoU của các cặp khớp là 0,61; nếu chấm cùng output đó theo pixel thì F1 = 0). Hai lỗi chính:

1. **Box lệch hoặc bịa** (lỗi lớn nhất). Trong 349 box của E0, chỉ 41 box khớp GT (IoU ≥ 0,5, đúng lớp). 173 box lệch (IoU 0,1–0,5) và 132 box không chạm GT nào (IoU < 0,1); chỉ 3 box sai lớp. Model thường trả đúng **1 box/ảnh** (251/311 ảnh, trung bình 1,20 box/ảnh, trong khi GT là 1,87), hay khoanh cả mặt đường làm một "crack", ví dụ ảnh test đầu tiên:
   `[{"label": "crack", "bbox_2d": [0, 478, 1000, 999]}]`
2. **JSON hỏng do lặp vô hạn**: model sinh một chuỗi box crack đều nhau rồi bị cắt ở `max_new_tokens=256`:
   `[{"label": "crack", "bbox_2d": [500, 450, 550, 500]}, {"label": "crack", "bbox_2d": [500, 500, 550, 550]}, {"label": "crack", "bbox_2d": [500, 550, 550, 600]}, … {"label": "crack", "bbox_2d": [500` (bị cắt)

Kết quả là F1 crack chỉ 0,027. Ảnh: `report/figures/e0/` (ví dụ `sai_vlcsnap-2025-02-19-17h03m29s526.jpg`: 6 box "manhole" bịa trên ảnh có 3 ổ gà và 1 vết nứt).

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

Tôi chấm lại chính file dự đoán bằng `--coord pixel`:

| Output | `--coord norm1000` (đúng) | `--coord pixel` (sai) |
|---|---|---|
| E0 | F1 0,088 | **F1 0,000** (FP/ảnh 1,12) |
| E1 | F1 0,281 | **F1 0,000** (FP/ảnh 1,67) |

Khi chấm theo pixel, `score.py` đổi x bằng `x/640·1000` và y bằng `y/360·1000`. Một box đúng như `[400, 750, 600, 850]` (0–1000) thành `[625, 2083, 938, 2361]`: theo trục y nó **nằm ngoài khung hình** (>1000) vì 750/360·1000 = 2083. Mọi box bị kéo giãn 1,56× theo x và 2,78× theo y, lệch khỏi GT nên IoU ≈ 0 và mọi TP thành FP + FN. F1 về đúng 0, không phải chỉ giảm nhẹ. Đây là lỗi im lặng: `parse_valid_rate` vẫn 0,997 nên chỉ nhìn tỉ lệ JSON hợp lệ sẽ không phát hiện được.

Qwen3-VL dùng thang 0–1000 tương đối vì ảnh được resize động (ở đây 640×360 → 640×352, lưới 40×22 patch 16 px → 220 token sau merge 2×2). Toạ độ tương đối không phụ thuộc độ phân giải, cùng một cách nói cho mọi kích thước ảnh, và mỗi toạ độ luôn ≤ 4 chữ số (ít token, phân bố số ổn định khi pre-train).

Kiểm chứng phía nhãn: test `vlmlab/tests/test_labels.py::test_answer_matches_scorer_gt` đảm bảo đáp án huấn luyện khớp GT của trọng tài với IoU > 0,98; round-trip box `[203,173,394,225]` px → `[317,481,616,625]` (0–1000) → `[202.9,173.2,394.2,225.0]` px (lệch < 0,4 px do làm tròn số nguyên).

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

Với ảnh train đầu tiên, chuỗi đầy đủ gồm 220 token ảnh + prompt chữ + đáp án; chỉ đáp án (JSON 9 box + `<|im_end|>\n`) có `labels != -100`. `inspect.txt` xác nhận: giải mã riêng các token có nhãn cho đúng chuỗi JSON đáp án + `<|im_end|>`, và prompt là tiền tố đúng của chuỗi đầy đủ (assert trong `encode_example`).

Lý do:
- Ta muốn học **p(đáp án | ảnh, prompt)**. Prompt giống hệt nhau ở mọi mẫu; nếu tính loss trên nó, model học thuộc lòng câu hỏi (loss dễ giảm, gradient vô ích) và phần prompt (220 token ảnh + câu lệnh, dài hơn nhiều so với một đáp án JSON 1–3 box) lấn át đáp án, làm loss "đẹp" nhưng tín hiệu thật về box bị pha loãng.
- **Token ảnh không có "đáp án" để dự đoán**: đó là các vị trí `<|image_pad|>` được thay bằng embedding từ ViT. Bắt LLM dự đoán token tiếp theo là `<|image_pad|>` là vô nghĩa và với E3 còn đẩy gradient vào ViT theo hướng sai.
- Token `<|im_end|>` **phải** có loss: nó dạy model dừng. Thiếu nó, model sẽ lặp box như E0 (21 ảnh JSON hỏng). E1 chỉ còn 1/311 ảnh hỏng.

Ở E1 loss bắt đầu 1,58 (bước 1) và về ≈ 0,6 sau 50 bước, rồi dao động quanh 0,57 (trung bình 50 bước cuối) đến bước 300 (`report/logs/e1_train_log.jsonl`). Nếu tính cả prompt, loss sẽ thấp hơn nhiều (prompt cố định dễ đoán) nhưng không phản ánh chất lượng box.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

**Tôi đã gặp đúng lỗi này.** Với `llm_int8_skip_modules=["visual", "lm_head"]` như GUIDE, transformers 5.19 vẫn lượng tử hoá ViT, vì `should_convert_module` so khớp bằng `re.match`, tức là neo ở **đầu** tên module. `"visual"` không khớp `model.visual.blocks.0.attn.qkv`. Dấu hiệu trong lần smoke test (`report/logs/smoke_vit_quantized_bug.txt`): số tham số projector đếm được là **54,6M, đúng một nửa 109,1M** (Params4bit gói 2 số/byte), sau đó crash `ValueError: Attempting to unscale FP16 gradients`.

Vì sao không huấn luyện được: trọng số Linear4bit là `uint8` đã lượng tử NF4 kèm hằng số scale; không có gradient cho chúng (bitsandbytes chỉ cho gradient đi qua tới đầu vào). Muốn "mở" merger thì phải có bản trọng số liên tục để optimizer cập nhật; nếu `.float()` một Params4bit, ta chỉ được bản ép kiểu của các byte đã gói, không phải trọng số thật. Muốn cập nhật thì phải giải lượng tử (mất chính xác) rồi lượng tử lại mỗi bước (sai số làm tròn NF4 lớn hơn bước cập nhật lr 2e-4, nên cập nhật bị "nuốt").

Cách xử lý: `llm_int8_skip_modules=["model.visual", "visual", "lm_head"]` và thêm assert `Linear4bit trong ViT = 0` mỗi lần nạp model (log: `[load] Linear4bit: ViT=0  LLM=252`). Sau khi sửa, projector đếm đủ 109.105.152 tham số.

Vì sao fp32: optimizer cập nhật `w ← w − lr·Δ`, với bước AdamW cỡ lr (2e-4 lúc đỉnh, giảm theo cosine về 0). fp16 chỉ có 10 bit mantissa: với |w| ≈ 0,05 (thang của trọng số merger), khoảng cách giữa hai số fp16 liền kề là 2⁻¹⁵ ≈ 3e-5. Nửa sau lịch cosine (lr < 3e-5) mọi cập nhật bị làm tròn về 0, và cả lúc lr lớn thì sai số làm tròn vẫn tích luỹ. Trạng thái Adam (m, v ~ grad²) cũng dễ underflow ở fp16. Ngoài ra `GradScaler` từ chối unscale gradient fp16 (đúng lỗi tôi gặp). Vì vậy tham số **huấn luyện** (LoRA, projector) giữ bản master fp32, còn phần đóng băng ở fp16/4-bit và forward chạy trong `autocast(fp16)`. Tôi thêm assert "mọi tham số `requires_grad` là fp32" trong `apply_setup`.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

| Split | Ảnh | Phiên | pothole | crack | manhole |
|---|---|---|---|---|---|
| train | 1.422 | 13 | 1.074 | 1.750 | 691 |
| val | 276 | 2 | 66 | 470 | 103 |
| test | 311 | 3 | 121 | 299 | 163 |

Đã kiểm tra: giao tập `session` giữa ba split đều rỗng. Test gồm 3 phiên: `vlc-2025-02-18-23` (213 ảnh), `vlc-2025-02-19-17` (83), `num-4` (15).

Ảnh là frame cắt từ video: trong 1.685 cặp frame liền kề (cùng phiên, có timestamp), khoảng cách trung vị chỉ **2,1 giây**, 77% cặp cách nhau ≤ 5 s. Hai frame liền kề cho thấy **cùng một ổ gà/nắp cống, cùng mặt đường, cùng ánh sáng**, chỉ dịch đi do xe chạy (chép nguyên box của frame trước sang frame sau chỉ được F1 0,083, nên rò rỉ không phải ở toạ độ mà ở **diện mạo**). Nếu chia ngẫu nhiên, hầu như mỗi ảnh test đều có "anh em" cách 2 giây trong train; model chỉ cần nhớ hình dạng của chính vết hư hại đó trên chính đoạn đường đó (học thuộc) là đoán đúng, và F1 đo được là **khả năng nhớ** chứ không phải khả năng tổng quát hoá sang đường mới. Chia theo phiên buộc test là những con đường/thời điểm model chưa từng thấy, đúng với triển khai thật.

Cái giá: phân bố lớp lệch (val chỉ có 66 pothole) và test chỉ có 3 phiên, nên CI rộng và F1 test phụ thuộc mạnh vào 1 phiên chiếm 68% ảnh.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?

| | E1 | E2 (+projector) | E3 (+LoRA ViT) |
|---|---|---|---|
| F1 [CI95] | 0,281 [0,246–0,322] | 0,346 [0,306–0,386] | 0,365 [0,322–0,408] |
| Tham số huấn luyện | 33,0M | 142,1M (+109,1M) | 148,4M (+6,3M) |
| VRAM đỉnh | 6,34 GB | 7,94 GB (+1,60) | 8,12 GB (+0,18) |
| Thời gian 300 bước | 45,3 phút | 47,0 phút (+4%) | 55,1 phút (+17%) |
| Loss train (TB 50 bước cuối) | 0,572 | 0,520 | 0,525 |

Chênh lệch, đo bằng **bootstrap ghép cặp** trên cùng tập ảnh lấy lại (`python -m vlmlab.compare`, 2.000 lần):

| So sánh | ΔF1 | CI95 của Δ | P(Δ ≤ 0) |
|---|---|---|---|
| E1 → E2 | +0,065 | [+0,025; +0,107] | 0,002 |
| E2 → E3 | +0,019 | [−0,021; +0,059] | 0,16 |
| E1 → E3 | +0,084 | [+0,043; +0,126] | < 0,001 |

- **Mở projector (E2) đáng giá nhất**: +0,065 F1 (+23% tương đối), có ý nghĩa thống kê, trong khi chỉ tốn thêm 1,6 GB VRAM và ~2 phút. Projector là "cổng" chuyển đặc trưng thị giác sang không gian token của LLM; nhiệm vụ của ta (box nhỏ, vết nứt mảnh trên mặt đường) khác xa dữ liệu pre-train, nên cho phép chỉnh cách đặc trưng được chiếu (và DeepStack đưa đặc trưng tầng thấp 5/11/17, giàu chi tiết vị trí, vào LLM) giúp cả ba lớp: pothole 0,174→0,213, crack 0,215→0,270, manhole 0,513→0,613.
- **Thêm LoRA ViT (E3)** cho +0,019 nữa nhưng **CI95 của Δ chứa 0**: chưa thể khẳng định ViT LoRA có ích với 311 ảnh test. Chi phí: +17% thời gian (gradient phải chảy ngược qua 24 block ViT trên 880 patch/ảnh) và VRAM cao nhất. Lợi ích nhìn thấy chủ yếu ở pothole (0,213→0,258) và manhole nhỏ, đúng chỗ cần đặc trưng thị giác tốt hơn, nhưng vẫn trong biên nhiễu.
- **Đáng đồng tiền nhất: E2.** Nếu ngân sách GPU cho phép thì E3 là setup có F1 điểm cao nhất (và tôi dùng E3 làm gốc cho E4), nhưng không nên tuyên bố E3 > E2.
- VRAM: E3 > E2 > E1 vì (1) projector 109M tham số ở fp32 cần trọng số + gradient + 2 trạng thái Adam ≈ 109M × 16 byte ≈ 1,7 GB (khớp mức tăng 1,60 GB E1→E2); (2) E3 phải lưu activation của ViT cho backward (dù đã gradient checkpointing) cộng LoRA ViT; mức tăng nhỏ (0,18 GB) vì ViT chỉ ~300M tham số và ảnh chỉ 880 patch.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?

Độ rộng CI95 bootstrap (theo ảnh) của F1: E0 0,055 · E1 0,076 · E2 0,080 · E3 0,086, tức là **≈ ±0,04 quanh mỗi F1**. Hai hệ quả:

1. **Chồng CI không có nghĩa là "không khác nhau", và ngược lại.** CI riêng của E1 [0,246–0,322] và E2 [0,306–0,386] chồng nhau, nhưng bootstrap ghép cặp (lấy lại cùng một tập ảnh cho cả hai model, nên loại được phương sai do "ảnh khó/dễ" chung) cho Δ = +0,065 với CI [+0,025; +0,107], có ý nghĩa. Ngược lại E2 vs E3: Δ = +0,019, CI [−0,021; +0,059], không kết luận được. Nghĩa là với 311 ảnh, ta chỉ phân biệt được chênh lệch cỡ ≥ 0,04–0,05 F1; mọi khác biệt nhỏ hơn (như E2/E3, và nhiều ablation) là nhiễu.
2. **311 ảnh không phải 311 mẫu độc lập.** Test chỉ có **3 phiên quay**: `vlc-2025-02-18-23` chiếm 213 ảnh (68%), `vlc-2025-02-19-17` 83 ảnh, `num-4` 15 ảnh; các frame trong một phiên cách nhau ~2 s nên tương quan mạnh. F1 của E3 theo phiên: 0,360 / 0,349 / 0,565, dao động lớn hơn nhiều so với CI theo ảnh. Bootstrap theo ảnh vì vậy **đánh giá thấp** độ bất định thật; bootstrap theo phiên (đơn vị độc lập đúng) chỉ có 3 phần tử nên gần như vô nghĩa (trong `compare.py` tôi in cả hai: CI theo phiên của E1→E3 là [+0,061; +0,299], rất lệch). Kết luận nào cũng chỉ nói được về "các con đường giống 3 phiên này".

Vì vậy tôi chỉ kết luận chắc chắn: fine-tune ≫ zero-shot (Δ ≈ +0,19–0,28) và mở projector có ích; còn thứ hạng E2 vs E3 và E4 phải đọc là "chưa phân định".

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.

F1 từng lớp (E3, setup tốt nhất): **pothole 0,258 < crack 0,274 ≪ manhole 0,648**. Recall theo diện tích box GT (E3, tập test):

| Lớp | n test | Diện tích trung vị | recall box < 0,5% | 0,5–2% | > 2% |
|---|---|---|---|---|---|
| pothole | 121 | 0,42% | 14/68 = 0,21 | 14/46 = 0,30 | 1/7 |
| crack | 299 | 1,98% | 0/15 = 0,00 | 33/136 = 0,24 | 51/148 = 0,34 |
| manhole | 163 | 0,67% | 26/64 = 0,41 | 64/96 = 0,67 | 2/3 |

- **Pothole kém nhất** dù có nhiều mẫu train nhất sau crack (1.074 box): box rất nhỏ (trung vị 0,42% khung hình ≈ 30×30 px trên ảnh 640×360, chỉ khoảng 1 token ảnh sau merge 2×2 với patch 16 px nên token 32×32 px), hình dạng/màu đa dạng (vũng nước, bóng, mảng vá trông giống ổ gà), và **ranh giới ngữ nghĩa với crack mơ hồ**: nhầm pothole→crack là kiểu nhầm lớp phổ biến nhất (E1: 16 box, E3: 12 box có IoU ≥ 0,5 với một pothole nhưng bị gọi là crack).
- **Crack**: box là hình chữ nhật bao một vật **mảnh và dài**, ranh giới box tuỳ người gán nhãn nên IoU ≥ 0,5 khó đạt (recall chỉ 0,24–0,34 ngay cả với box lớn). Nhãn còn mơ hồ ở cách gom: ảnh `report/figures/e3/sai_vlcsnap-2025-02-19-17h03m29s526.jpg` có GT là một box "crack" lớn bao trọn vùng mặt đường hư hỏng chứa 3 ổ gà; model trả 6 box crack nhỏ, nên 0 TP dù nhìn bằng mắt là "đúng chỗ". Đây là lỗi của **định nghĩa nhãn** nhiều hơn là của model. E0 gần như không làm được crack (F1 0,027).
- **Manhole tốt nhất**: vật thể cứng, hình tròn/vuông rõ ràng, màu kim loại tương phản, ranh giới box không mơ hồ, nên box dễ khớp IoU ≥ 0,5 (recall 0,67 ở box 0,5–2%).
- Theo kích thước: với **mọi lớp, box < 0,5% là phần thua đậm** (crack 0/15), khớp với giới hạn độ phân giải 220 token ảnh.

Ảnh minh hoạ (xanh lá = GT, vàng = dự đoán khớp, đỏ = dự đoán sai): `report/figures/e3/` (3 đúng + 3 sai, mỗi phiên quay một ảnh), `report/figures/e1/`, `report/figures/e0/`, chú thích TP/FP/FN trong `captions.tsv` từng thư mục. Ví dụ đúng `dung_vlcsnap-2025-02-19-17h04m49s140.jpg`: 3/3 ổ gà khớp, chỉ sót box crack lớn bao quanh.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?

TODO_Q10

## Nhìn lại

TODO_LOOK
