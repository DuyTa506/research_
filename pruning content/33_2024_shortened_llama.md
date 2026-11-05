# Shortened LLaMA: Depth Pruning for Large Language Models with Comparison of Retraining Methods (2024)

**Tác giả:** Bo-Kyeong Kim, Geonmin Kim, Tae-Ho Kim, Thibault Castells, Shinkook Choi, Junho Shin, Hyoung-Kyu Song (Nota Inc., Captions).  
**Venue / arXiv:** arXiv:2402.02834 (v2, 23/06/2024). Bản text không ghi tên hội nghị.  
**Nguồn:** https://arxiv.org/abs/2402.02834  
**Chủ đề:** Depth pruning (xoá Transformer block) so với width pruning, xét **độ trễ batch-1**; so sánh LoRA với continued pretraining (CPT) để hồi phục.  
**Vị trí trong lộ trình:** Đọc trước: 26 LLM-Pruner, 28 FLAP (đối thủ width), 31 ShortGPT, 32 Deeper layers. Bài này thêm hai thứ quan trọng cho dự án: (1) bằng chứng đo thật rằng **width pruning không tăng tốc ở batch nhỏ**; (2) bảng so sánh phương pháp hồi phục theo mức cắt. Đọc tiếp: 34 SLEB, 36 ReplaceMe, 39 Minitron.

> Tài liệu học tập tiếng Việt, ghi chú diễn giải, không phải bản dịch paper. Số liệu lấy từ paper; bảng đã được đọc kỹ nhưng text PDF hai cột bị lẫn nên chỉ trích những dòng đọc chắc chắn. Phần ứng dụng là suy luận của chúng ta.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
4. [Thí nghiệm và kết quả chính](#4-thí-nghiệm-và-kết-quả-chính)
5. [Phân tích / ablation](#5-phân-tích--ablation)
6. [Giới hạn](#6-giới-hạn)
7. [Liên hệ với các bài khác](#7-liên-hệ-với-các-bài-khác)
8. [Ứng dụng cho dự án (suy luận)](#8-ứng-dụng-cho-dự-án-suy-luận)
9. [Thí nghiệm gợi ý](#9-thí-nghiệm-gợi-ý)
10. [Tóm tắt công thức / khái niệm](#10-tóm-tắt-công-thức--khái-niệm)
11. [Kết luận ngắn](#11-kết-luận-ngắn)

---

## 1. Bối cảnh

- **Width pruning** = thu nhỏ ma trận (bỏ head attention, bỏ neuron FFN) nhưng giữ số layer. **Depth pruning** = bỏ nguyên block/layer, giữ nguyên kích thước ma trận còn lại.
- **Memory-bound**: khi sinh từng token với batch nhỏ, thời gian bị chi phối bởi việc đọc trọng số từ bộ nhớ GPU, không phải bởi số phép nhân. GPU ít được dùng hết sức mạnh tính toán. Tăng batch giúp nhưng dễ OOM trên GPU nhỏ.
- Các phương pháp lúc đó (LLM-Pruner, FLAP, Wanda-sp) làm width pruning và thường chỉ báo cáo độ chính xác, ít đo tốc độ thật. Depth pruning bị coi là kém vì đơn vị cắt thô.

## 2. Ý tưởng chính

Tác giả chọn **Transformer block** (MHA + FFN) làm đơn vị cắt, chấm điểm từng block bằng một tiêu chí rẻ (perplexity khi bỏ block đó, hoặc Taylor bậc 1 có bảo vệ), bỏ các block ít quan trọng nhất trong **một lần** (one-shot). Họ chứng minh bằng đo đạc: ở batch-1, cắt chiều rộng gần như không giảm latency (thậm chí tăng vì kích thước ma trận không thân thiện GPU), còn bỏ cả block thì giảm số lần truy cập bộ nhớ nên nhanh thật. Phần thứ hai: so sánh LoRA với CPT; với mức cắt nhẹ LoRA đủ, với mức cắt nặng (dưới ~3.7B từ 6.7B) cần CPT toàn tham số.

## 3. Phương pháp chi tiết

### 3.1 Tiêu chí chấm block

Ký hiệu $W^{k,n}_{i,j}$ là trọng số của phép tuyến tính loại $k$ (Q/K/V/O, up/gate/down) trong block $n$.

- **Magnitude (Mag):** $I^n_{\text{Mag}}=\sum_k\sum_i\sum_j \lvert W^{k,n}_{i,j}\rvert$ (tính ở mức neuron đầu ra rồi cộng; theo mô tả paper).
- **Taylor:** thay đổi loss khi đặt trọng số = 0, bỏ số hạng bậc hai:

$$L(W)-L(W=0)\approx \frac{\partial L}{\partial W}\,W,\qquad I^n_{\text{Taylor}}=\sum_k\sum_i\sum_j \left\lvert\frac{\partial L(D)}{\partial W^{k,n}_{i,j}}W^{k,n}_{i,j}\right\rvert$$

- **Mag+ / Taylor+:** như trên nhưng **bảo vệ 4 block đầu và 2 block cuối** (không cho vào danh sách ứng viên), vì xếp hạng thuần tuý hay chọn block đầu nhưng bỏ chúng thì hỏng.
- **PPL:**

$$\boxed{I^n_{\text{PPL}}=\exp\!\Big(-\tfrac1{SL}\sum_s\sum_l \log p_{\theta_n}\big(x^{(s)}_l\mid x^{(s)}_{<l}\big)\Big)}$$

với $\theta_n$ = mô hình bỏ block $n$. Block có PPL sau khi bỏ **thấp** hơn thì ít quan trọng hơn. Chỉ cần forward, không gradient. Không cần bảo vệ block đầu/cuối: PPL tự phát hiện chúng quan trọng.

### 3.2 One-shot

```
1. calibration D = 10 mẫu BookCorpus, độ dài 128
2. for n in blocks: tính I_n (PPL khi bỏ block n, hoặc Taylor+)
3. sắp xếp tăng dần, bỏ các block ít quan trọng nhất đến khi đạt ngân sách tham số
4. retrain: LoRA (nhẹ) hoặc CPT (nặng) hoặc CPT -> LoRA
```

Có thể làm lặp (như SLEB, cập nhật điểm sau mỗi lần bỏ) nhưng tác giả quan sát retraining quan trọng hơn lặp hay một lần, đặc biệt khi cắt nặng, và lặp tốn thời gian hơn.

### 3.3 Hồi phục

- **LoRA:** rank 8, LR $10^{-4}$, batch 64, 2 epoch, trên 50K mẫu Alpaca đã làm sạch; LoRA gắn vào mọi ma trận chiếu. Cắt 20% mô hình 7B: ~2 giờ, 22GB VRAM (một A100).
- **CPT:** cập nhật toàn tham số trên SlimPajama (627B token); AdamW (0.9, 0.95), weight decay 0.1, LR $10^{-4}$, batch toàn cục 512, 8 H100, tối đa ~2 tuần mỗi kích thước. Thời gian/token thực tế: 5.5B: 6 ngày, 37B token; 3.7B: 8 ngày, 74B token; 2.7B: 12 ngày, 150B token; 1.5B: 11 ngày, 271B token.
- **CPT -> LoRA:** CPT xong rồi LoRA trên dữ liệu instruction.

## 4. Thí nghiệm và kết quả chính

**Mô hình:** LLaMA-7B, Vicuna-7B/13B-v1.3. **Baseline width:** LLM-Pruner, FLAP, Wanda-sp. **Baseline depth:** SLEB (không retrain). **Eval:** 7 tác vụ commonsense zero-shot (BoolQ, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA), PPL WikiText2/PTB. **Tốc độ:** 12 token vào, 128 token ra, **batch 1**, một GPU.

### 4.1 Cắt vừa + LoRA (Table 2, LLaMA-7B, 20% cắt, 5.5B)

| Phương pháp | PPL WikiText2 | PPL PTB | Ave Acc (%) | Latency H100 (s) | Throughput H100 (tok/s) |
|---|---|---|---|---|---|
| Gốc 6.7B | 12.6 | 22.1 | 66.3 | 2.4 | 53.7 |
| Wanda-sp | 21.4 | 47.2 | 51.8 | 3.1 | 41.7 |
| FLAP | 17.0 | 30.1 | 59.5 | 3.2 | 40.5 |
| LLM-Pruner | 17.6 | 30.4 | 61.8 | 3.0 | 43.2 |
| SLEB | 18.5 | 31.6 | 57.6 | 1.9 | 66.0 |
| Ours Taylor+ | 20.2 | 32.3 | 63.5 | 1.9 | 66.0 |
| Ours PPL | 17.7 | 30.7 | 61.9 | 1.9 | 66.0 |

Điểm mấu chốt: mọi phương pháp width **chậm hơn mô hình gốc** (latency 3.0–3.2 s so với 2.4 s) dù ít tham số hơn. Trên RTX3090 24GB, gốc 5.1 s (25.0 tok/s), depth 4.5 s (28.4 tok/s), width 6.0–7.7 s. Với 35% cắt (4.5B) depth: latency 1.6 s, 80.1 tok/s trên H100. Với Vicuna-13B gốc, RTX3090 báo OOM; bản depth cắt 21% chạy được (5.4 s, 23.9 tok/s).

### 4.2 Cắt mạnh: LoRA vs CPT (Table 3, Vicuna-7B)

PPL WikiText2 / Ave Acc (%) tương ứng 5.5B (20%), 3.7B (45%), 2.7B (60%), 1.5B (80%):

| Phương pháp | PPL (5.5B, 3.7B, 2.7B, 1.5B) | Ave Acc (5.5B, 3.7B, 2.7B, 1.5B) |
|---|---|---|
| LLM-Pruner | 19.6, 38.8, 66.4, 202.9 | 60.1, 50.1, 44.3, 38.4 |
| SLEB | 25.1, 110.4, 731.5, 18730.8 | 55.6, 40.2, 39.1, 37.4 |
| Ours, LoRA | 18.8, 37.0, 68.9, 1002.2 | 60.7, 47.0, 40.1, 37.1 |
| Ours, CPT | 14.3, 16.0, 17.1, 20.5 | 61.5, 57.1, 55.0, 49.2 |
| Ours, CPT->LoRA | 14.8, 16.5, 17.8, 21.1 | 63.1, 57.4, 55.0, 49.0 |

Throughput batch-1 H100 của bản depth: 66.0, 84.0, 107.4, 182.5 tok/s (tức 1.2x, 1.6x, 2.0x, 3.4x so với gốc 53.7); width chỉ ~40–45 tok/s. Mô hình gốc: PPL 17.1 trên WikiText2 (dòng Vicuna-7B Original), Ave Acc 65.9.

Ví dụ sinh văn bản (Table 4, 2.7B, cắt 60%): các baseline (Wanda-sp, FLAP, LLM-Pruner, SLEB) sinh rác; mô hình CPT sinh văn bản trôi chảy. **Khởi tạo CPT từ mô hình đã cắt tốt hơn train từ đầu cùng kiến trúc** (Fig. 5, 2.7B).

### 4.3 Lượng tử hoá
GPTQ 4-bit áp dụng lên mô hình depth-pruned sau CPT giảm VRAM mà không giảm zero-shot đáng kể (Fig. 6; số chi tiết ở Appendix D, chưa đọc trong bản text này).

## 5. Phân tích / ablation

**Tiêu chí (Table 5, LLaMA-7B, 20% cắt 5.5B):** Mag: PPL 7720.7, Acc 34.4; Mag+: 19.4, 56.1; Taylor: 3631.7, 35.5; Taylor+: 20.2 (PTB 32.3), Acc 63.5; PPL: 17.7 (PTB 30.7), Acc 61.9. Ở 35% (4.5B): Taylor+ PPL 33.2, Acc 55.4; PPL 23.1, Acc 55.2. Kết luận tác giả: Taylor+ tốt về accuracy commonsense; PPL tốt hơn về chất lượng sinh/perplexity. Không có "+" thì chọn nhầm block đầu và sập.

**Đơn vị cắt (Table 6):** cắt cả block vs cắt riêng MHA/FFN. Với model >5B, cắt từng MHA/FFN cho acc tốt hơn chút nhưng PPL xấu hơn (5.7B: acc 63.1 vs 62.8, PPL 20.8 vs 16.9); với model <5B, block tốt hơn mọi mặt (ví dụ ~4.5B/4.6B: block PPL 23.1 acc 55.2, module PPL 38.9 acc 52.5). Giải thích (giả thuyết của tác giả): MHA và FFN phối hợp; cắt module riêng có thể tạo ra các FFN liên tiếp; cắt block thì block lân cận bù được.

**Lượng calibration (Table 7):** 10, 50, 100, 1000 mẫu cho kết quả gần như không đổi; Taylor+ tụt acc ở 1000 mẫu (61.7 so với 63.5), chưa lý giải.

**Cần lưu ý:** trên chính bảng của họ, CPT mới là yếu tố tạo khác biệt, không phải tiêu chí chọn block.

## 6. Giới hạn

**Do tác giả nêu:** chưa thử mô hình >13B; CPT thiết yếu cho cắt nặng và rất tốn (~2 tuần x 8 H100); chưa khám phá corpus/hyperparameter khác; chưa đánh giá độ phù hợp với sở thích người dùng (safety, helpfulness) sau khi cắt/retrain/lượng tử hoá.

**Do chúng ta nhận thấy:**
- Chỉ đo batch 1 với 12 token vào / 128 ra, trên GPU datacenter (H100, A100) và RTX3090; không có GPU 8GB.
- Mọi so sánh "ít tham số" theo tổng tham số; kết luận width chậm là do cài đặt kích thước ma trận không chia hết cho 8 (FFN-D lẻ như 7156) trong HF Transformers, một phần có thể sửa bằng làm tròn kích thước (suy luận của chúng ta; tác giả có nêu vấn đề chia hết cho 8 nhưng không thử sửa).
- Dữ liệu CPT (627B token SlimPajama, 37B–271B token được dùng) lớn hơn nhiều so với ngân sách distillation thông thường của dự án.
- Chỉ đánh giá commonsense zero-shot và PPL, không đánh giá tác vụ sinh dài có cấu trúc.

## 7. Liên hệ với các bài khác

- **26 LLM-Pruner, 28 FLAP, 29 SliceGPT:** đối thủ width; bài này cho thấy lợi thế depth về latency batch nhỏ.
- **31 ShortGPT, 32 Deeper layers:** cùng họ depth; bài này không dùng cosine/góc mà dùng PPL và Taylor+.
- **34 SLEB:** đối thủ trực tiếp (retrain-free, lặp); bài này thêm retrain và thấy cắt nặng thì SLEB không đủ.
- **43 Pruning vs scratch, 38 Sheared LLaMA:** bằng chứng "khởi tạo từ mô hình đã cắt tốt hơn train từ đầu" tương đồng.
- **39 Minitron:** cùng triết lý depth + retrain quy mô lớn; Minitron dùng distillation thay vì CPT thuần.

## 8. Ứng dụng cho dự án (suy luận)

- **Rất liên quan đến mục tiêu batch-1, GPU <= 8GB.** Kết quả đo cho thấy nếu bạn cắt neuron MLP (width) thì độ trễ batch-1 có thể không cải thiện, còn cắt layer thì có. Trùng khớp quan sát của bạn: cắt neuron MLP hồi phục tệ hơn cắt layer; và bài này thêm lý do thứ hai (tốc độ).
- Cảnh báo: bài này đo trên Transformer thuần với HF mặc định; decoder LightOnOCR có GatedDeltaNet + attention, kernel khác. Cần đo lại thay vì suy ra.
- Nên đánh giá bằng **PPL block-ablation** (một forward/block) trước khi dùng Taylor; nhỏ gọn, ít giả định. Với decoder ~20 layer chỉ cần 20 forward.
- Hồi phục: CPT quy mô trăm tỉ token là không khả thi; distillation 4B->0.3B trên dữ liệu OCR đóng vai trò tương tự CPT nhưng rẻ hơn. Bài này là cơ sở lập luận "LoRA không đủ ở mức cắt >45%".
- Quy tắc bảo vệ block đầu (4) và cuối (2) trong Mag+/Taylor+ hữu ích khi tự cài đặt.

## 9. Thí nghiệm gợi ý

1. Đo latency batch-1 trên GPU 8GB của decoder LightOnOCR gốc, sau khi cắt 25% bằng (a) bỏ layer, (b) cắt neuron MLP cùng số tham số; kiểm tra có tái hiện "width không nhanh hơn" không (kể cả khi làm tròn kích thước chia hết cho 8/64).
2. PPL block-ablation (từng block của decoder, và từng block ViT), vẽ biểu đồ như Fig. 4, so với BI (bài 31).
3. Với mức 0.3B, so hồi phục bằng LoRA vs full fine-tune distillation cùng số bước; xem khoảng cách có lớn như bảng 3 (LoRA 47.0 vs CPT 57.1 ở 3.7B) không.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| $I^n_{PPL}$ | PPL trên calibration khi bỏ block $n$ (thấp = ít quan trọng) |
| Taylor+ | $\lvert\partial L/\partial W\cdot W\rvert$ cộng theo block, bảo vệ 4 block đầu + 2 block cuối |
| CPT | continued pretraining toàn tham số trên corpus lớn |
| CPT->LoRA | CPT rồi LoRA instruction |
| Memory-bound | độ trễ do đọc bộ nhớ, không do FLOPs |

## 11. Kết luận ngắn

Với batch nhỏ, chỉ việc xoá hẳn block mới thật sự nhanh hơn; cắt vừa thì LoRA đủ, cắt mạnh thì phải retrain toàn tham số với nhiều dữ liệu. Bài hữu ích nhất cho dự án ở phần chứng cứ về tốc độ batch-1 và bảng LoRA vs CPT.

Link gốc: https://arxiv.org/abs/2402.02834
