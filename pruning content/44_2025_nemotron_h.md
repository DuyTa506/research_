# Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models (2025)

**Tác giả:** NVIDIA (đội Nemotron; phần nén mô hình có Sreenivas, Muralidharan, Bercovich, Taghibakhshi, Molchanov và cộng sự).  
**Venue / arXiv:** arXiv:2504.03624 (v4, 5/9/2025). Bài không nêu venue hội nghị; đây là technical report.  
**Nguồn:** https://arxiv.org/abs/2504.03624  
**Chủ đề:** Kiến trúc hybrid Mamba-2 / attention / FFN, và **MiniPuzzle**: nén Nemotron-H-56B xuống 47B bằng pruning (cắt layer + cắt neuron FFN) rồi distillation.  
**Vị trí trong lộ trình:** Đọc trước: 39 Minitron, 40 Llama-3.1-Minitron, 42 Puzzle (MiniPuzzle là "Minitron đơn giản + Puzzle linh hoạt"), 33–34 (cắt layer). Bài này thêm: lần đầu áp công thức pruning + distillation lên một hybrid Mamba-Transformer **quy mô lớn** và cho biết *cắt cái gì, giữ cái gì*. Tiếp theo: 45 Minitron-SSM (cắt cả bên trong Mamba), 46 Mamba-Shedder, 47 Jet-Nemotron.

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Đây là technical report dài (FP8 training, dữ liệu, VLM, instruct...); **ghi chú này chỉ tập trung vào kiến trúc (tóm tắt) và Mục 3 "Compression and Distillation"**. Phần "Ứng dụng cho dự án" là suy luận của chúng tôi, không phải kết luận của paper.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Kiến trúc hybrid của Nemotron-H (tóm tắt)](#3-kiến-trúc-hybrid-của-nemotron-h-tóm-tắt)
4. [MiniPuzzle: phương pháp chi tiết](#4-minipuzzle-phương-pháp-chi-tiết)
5. [Thí nghiệm và kết quả chính](#5-thí-nghiệm-và-kết-quả-chính)
6. [Phân tích / ablation](#6-phân-tích--ablation)
7. [Giới hạn](#7-giới-hạn)
8. [Liên hệ với các bài khác](#8-liên-hệ-với-các-bài-khác)
9. [Ứng dụng cho dự án (suy luận)](#9-ứng-dụng-cho-dự-án-suy-luận)
10. [Thí nghiệm gợi ý](#10-thí-nghiệm-gợi-ý)
11. [Tóm tắt khái niệm](#11-tóm-tắt-khái-niệm)
12. [Kết luận ngắn](#12-kết-luận-ngắn)

---

## 1. Bối cảnh

- **Self-attention** tốn $O(n)$ tính toán cho mỗi token sinh ra (n = số token đã thấy) và phải lưu **KV cache** (bộ nhớ key/value của mọi token cũ), nên chậm và tốn VRAM khi context dài.
- **Mamba-2** (một loại *state space model*, SSM): thay vì nhìn lại mọi token, nó nén lịch sử vào một **trạng thái cố định kích thước** (constant memory per token). Rẻ ở context dài, nhưng khả năng "tra cứu chính xác" yếu hơn attention. Do đó có xu hướng **hybrid**: phần lớn layer là Mamba, giữ ít layer attention.
- Câu hỏi của bài: (a) hybrid có đạt độ chính xác của Transformer cùng cỡ không? (b) làm sao tạo ra một biến thể **nhỏ hơn** (cho GPU RTX 5090 32 GiB) mà không phải huấn luyện lại từ đầu?
- Tiền đề cần nhớ: **pruning** = bỏ bớt thành phần của mô hình đã huấn luyện; **distillation** = cho mô hình nhỏ (student) bắt chước phân phối đầu ra của mô hình lớn (teacher). Hai thứ kết hợp gọi là "compression via pruning and distillation" (Minitron, bài 39).

## 2. Ý tưởng chính

Nemotron-H là họ mô hình (8B, 47B, 56B) trong đó khoảng 8% layer là self-attention, phần còn lại xen kẽ Mamba-2 và FFN. Để có bản 47B từ bản 56B, tác giả đề xuất **MiniPuzzle** gồm ba bước: (1) đo độ quan trọng của từng *layer* và từng *neuron FFN* chỉ bằng forward pass, (2) liệt kê khoảng 400 kiến trúc ứng viên thoả ràng buộc bộ nhớ rồi chọn dần xuống 130 rồi 3 ứng viên (NAS có điều kiện), (3) distill ngắn (~7B token) để chọn ứng viên tốt nhất, rồi distill dài (63B token). Kết quả: 47B gần như không mất độ chính xác so với 56B, nhanh hơn 1.2x ở context dài, dùng ít hơn khoảng 300x số token so với huấn luyện từ đầu (theo paper).

## 3. Kiến trúc hybrid của Nemotron-H (tóm tắt)

| | Số layer | Hidden | FFN dim | Q heads | KV heads | State dim | Mamba groups |
|---|---|---|---|---|---|---|---|
| Nemotron-H-8B | 52 | 4096 | 21504 | 32 | 8 | 128 | 8 |
| Nemotron-H-56B | 118 | 8192 | 32768 | 64 | 8 | 256 | 8 |

- Mỗi "layer" là **một** trong ba loại: Mamba-2, self-attention, hoặc FFN (khác Transformer chuẩn, nơi một block gồm attention + FFN). Có skip connection quanh mỗi layer.
- Số layer attention khoảng 8% tổng (4/52 ở 8B, 10/118 ở 56B), **rải đều** trong mô hình; phần còn lại chia đều Mamba-2 và FFN (56B: 54 Mamba-2 + 54 FFN + 10 attention = 118).
- Ràng buộc thiết kế: layer đầu là Mamba-2, layer cuối là FFN, attention luôn đứng trước FFN. Không dùng position embedding. FFN dùng squared ReLU, GQA với 8 KV head.
- Có bản đối chứng Transformer cùng dữ liệu (Nemotron-T-8B). Tốc độ: 56B sinh nhiều hơn 2.4x token/s/GPU so với Qwen-2.5-72B và Llama-3.1-70B ở input 65536 / output 1024 (paper nêu 2.9x sau khi distill sang 47B).

## 4. MiniPuzzle: phương pháp chi tiết

Mục tiêu triển khai: chạy **context 1 triệu token** trên một RTX 5090 (32 GiB) với trọng số FP4. Bản 56B riêng trọng số đã cần khoảng 29.5 GiB, chỉ còn 2.5 GiB cho KV cache và activation, nên cần nhỏ hơn.

### 4.1 Hai trục được cắt (và hai trục không cắt)

- **Cắt:** (a) toàn bộ *layer* (depth) thuộc cả ba loại; (b) *neuron ẩn của FFN* (width của FFN).
- **Đã thử nhưng bỏ:** cắt *Mamba heads* và *Mamba head dimension*: "both axes caused severe accuracy degradation (particularly head dimension pruning)". Đây là lý do sang bài 45 (Minitron-SSM) xử lý riêng bài toán cắt trong Mamba.
- Tác giả không dùng độ quan trọng dựa trên gradient (không khả thi ở quy mô này); chỉ dùng forward pass.

### 4.2 Độ quan trọng

**Layer importance** (theo Puzzle, bài 42): MSE giữa activation ngay trước LM head của mô hình đầy đủ và của mô hình bị bỏ layer đó, trung bình trên 128 mẫu huấn luyện:

$$
\boxed{\mathrm{Imp}(\ell)=\mathbb{E}_x\left\|h^{\text{full}}_{\text{pre-head}}(x)-h^{\text{drop }\ell}_{\text{pre-head}}(x)\right\|_2^2}
$$

**Neuron FFN importance** (theo Minitron, bài 39): FFN$(X)=\delta(XW_1^\top)W_2$ với $\delta$ = squared ReLU. Với neuron $i$ (hàng $i$ của $W_1$), điểm là tổng hợp activation của nó trên batch B và chuỗi S:

$$
F^{(i)}_{\text{neuron}}=\sum_{B,S}\delta\big(X (W_1^{i})^\top\big)
$$

Dùng 1024 mẫu hiệu chỉnh (calibration), tổng hợp bằng mean và $\ell_2$-norm theo kết luận của Minitron.

Quan sát trên Hình 9 của paper: layer quan trọng tập trung ở **đầu và cuối** mô hình (giống Minitron); đáng chú ý, dù chỉ có 10 layer attention, **một số attention layer lại thuộc nhóm ít quan trọng nhất** (đặc biệt layer 84, attention thứ 7), trong khi layer 40, 51, 62, 73 quan trọng hơn lân cận.

### 4.3 Conditional NAS (3 bước)

1. **Liệt kê ứng viên.** Quét lưới số layer cho từng loại (attention, FFN, Mamba-2) và FFN hidden dim $\in\{24576, 25600,\dots,32768\}$. Với mỗi cấu hình, giữ các layer/neuron điểm cao nhất; FFN của mọi layer cắt về cùng một độ rộng. Chỉ giữ kiến trúc có bộ nhớ < **31.7 GiB** (FP4, context 1M token), còn khoảng **400** ứng viên.
2. **Xếp hạng nhẹ.** Chấm bằng validation corpus 1M token với hai metric: *next-token accuracy* và *next-token parent agreement* (child đồng ý với parent về token đoán tham lam bao nhiêu lần). Giữ top **130**.
3. **Chọn ứng viên.** Benchmark trên một tập tác vụ, chọn **3** ứng viên; distill ngắn mỗi cái ~7B token để chọn ứng viên cuối.

### 4.4 Distillation

Logit-based, chỉ dùng **forward KL** (teacher = 56B). Tất cả chạy FP8, temperature 1.0, độ dài chuỗi 8192, batch 768 mẫu. Chi tiết loss lấy từ Mục 3 của Minitron.

### 4.5 Pseudo-code

```text
MiniPuzzle(parent, memory_budget):
  # 1. Importance (chỉ forward)
  for each layer l: I_layer[l] = MSE(parent_out, parent_out_without_l)   # 128 mẫu
  for each FFN neuron j: I_neuron[j] = agg_{B,S} activation_j            # 1024 mẫu
  # 2. Conditional NAS
  C = []
  for (n_attn, n_ffn, n_mamba, d_ffn) in grid:
      child = keep top-n layers per type by I_layer, keep top-d_ffn neurons per FFN
      if memory_FP4(child, ctx=1M) < 31.7 GiB: C.append(child)          # ~400
  C = top_130(C, by=[next_token_acc, parent_agreement])                  # 1M-token val
  C = top_3(C, by=benchmark_subset)
  # 3. Distillation
  for c in C: distill(c, teacher=parent, tokens=7B)                      # chọn cái tốt nhất
  best = argmax avg_benchmark(C)
  distill(best, teacher=parent, tokens=63B)                              # bản 47B cuối
```

## 5. Thí nghiệm và kết quả chính

**Cấu trúc 47B cuối (Mục 3.5):** giữ **5** attention (từ 10), **44** Mamba-2 (từ 54), **49** FFN (từ 54), FFN intermediate size **32768 → 30720**. (Tổng 98 layer, so với 118; phép cộng là của chúng tôi.)

**Ứng viên trước/sau distill ngắn (Bảng 6, định dạng "trước → sau"):**

| Benchmark | 56B-Base | Cand. 1 | Cand. 2 | Cand. 3 |
|---|---|---|---|---|
| MMLU | 84.2 | 81.5 → 83.7 | 80.8 → 83.2 | 81.3 → 83.6 |
| Commonsense (avg) | 67.6 | 64.2 → 66.6 | 63.8 → 67.2 | 64.5 → 66.8 |
| GSM8k | 91.4 | 67.9 → 89.2 | 60.7 → 91.1 | 19.5 → 90.5 |
| Code (avg) | 67.0 | 52.0 → 64.9 | 52.7 → 64.6 | 23.0 → 64.9 |
| AVERAGE | 70.6 | 63.3 → 69.3 | 62.1 → 69.6 | 53.7 → 69.5 |

Ứng viên 2 được chọn (trung bình sau distill cao nhất, 69.6) cho đợt distill 63B token.

**56B vs 47B cuối (Bảng 4):**

| Task | 56B-Base | 47B-Base |
|---|---|---|
| MMLU-Pro (5-shot COT) | 60.5 | 61.8 |
| MMLU (5-shot) | 84.2 | 83.6 |
| GSM8k (8-shot COT) | 93.7 | 93.3 |
| MATH (4-shot COT) | 59.4 | 57.4 |
| MATH Level 5 | 35.2 | 34.1 |
| MMLU STEM | 80.6 | 79.8 |
| HumanEval | 60.4 | 61.0 |
| HumanEval+ | 54.3 | 56.1 |

Tốc độ: 1.2x suy luận context dài so với 56B; 2.9x token/s/GPU so với Qwen-2.5-72B / Llama-3.1-70B. Chi phí nén: ~7B token (chọn ứng viên) + 63B token (distill chính).

## 6. Phân tích / ablation

- **Distill ngắn quan trọng để chọn ứng viên**: trước distill, ứng viên 3 kém hẳn (GSM8k 19.5 so với 60-68), nhưng sau 7B token cả ba gần bằng nhau (GSM8k 89.2 đến 91.1). Tức là điểm "trước distill" **không dự đoán chắc chắn** điểm sau distill; paper nói bước này "crucial" để chọn đúng.
- **Lựa chọn cuối phụ thuộc mục tiêu**: paper lưu ý ứng viên tốt nhất thay đổi tuỳ ưu tiên (code hay commonsense).
- **Điểm quan trọng không đồng nghĩa với "attention luôn quan trọng"**: một số attention layer bị xếp hạng thấp, và 47B giữ đúng một nửa số attention.
- **Trục Mamba không cắt được bằng cách đơn giản**: nguyên văn "severe accuracy degradation" (không có bảng số trong bài này).

## 7. Giới hạn

*Paper nêu / ngụ ý:*
- Chỉ cắt hai trục (layer, FFN neuron); trục Mamba bị loại vì suy giảm nặng.
- Importance chỉ dựa vào forward pass; không so sánh với phương pháp có gradient.

*Chúng tôi nhận xét:*
- Không có ablation tách riêng "cắt layer" vs "cắt FFN neuron" trong bài này; ta không biết trục nào đóng góp bao nhiêu vào độ giảm.
- Không có ablation về hàm mất mát distill (chỉ forward KL).
- Số token distill (63B) vẫn lớn so với ngân sách của một dự án nhỏ.
- Kết quả đo trên bản 56B/47B; chưa rõ chuyển được xuống mô hình 1B hay nhỏ hơn.

## 8. Liên hệ với các bài khác

- **39, 40 Minitron / Llama-3.1-Minitron:** nguồn của importance neuron FFN và cách distill; MiniPuzzle giữ sự đơn giản đó.
- **42 Puzzle:** nguồn của layer importance bằng MSE và ý tưởng NAS có điều kiện theo ràng buộc phần cứng.
- **45 Minitron-SSM:** giải quyết đúng chỗ MiniPuzzle bỏ cuộc (cắt Mamba heads/channels).
- **46 Mamba-Shedder:** cắt block/SSM trong Mamba không cần huấn luyện lại.
- **33, 34 (Shortened LLaMA, SLEB), 31 ShortGPT:** cách đo độ quan trọng layer; ở đây dùng MSE đầu ra thay vì cosine/perplexity.
- **43 Pruning vs scratch:** con số "~300x ít token hơn so với huấn luyện từ đầu" là một dữ kiện ủng hộ pruning + distill.

## 9. Ứng dụng cho dự án (suy luận)

- **Liên quan trực tiếp về cấu trúc:** LightOnOCR-3-0.8B có decoder hybrid GatedDeltaNet + full attention, giống tinh thần Nemotron-H (ít attention, nhiều layer tuyến tính). Cách đo độ quan trọng bằng MSE đầu ra khi bỏ layer và danh sách ứng viên có ràng buộc bộ nhớ có thể dùng nguyên.
- **Điều đáng chú ý:** attention layer **không** mặc nhiên là quan trọng nhất; nên đo từng layer thay vì "giữ hết attention".
- **Kết hợp với quan sát của bạn:** MiniPuzzle cắt *cả* layer *và* neuron FFN, rồi dùng distill để phục hồi. Bạn quan sát cắt neuron MLP phục hồi kém hơn cắt layer; trong bài này không có số so sánh trực tiếp hai trục, nên chưa xác nhận hay phản bác được. (Bài 49 trên VLM cho thấy chiều hướng phụ thuộc mức nén.)
- **Chi phí:** 63B token là quá lớn cho dự án; quy trình "distill ngắn để chọn ứng viên rồi mới distill dài" thì áp dụng được ở quy mô nhỏ hơn nhiều.
- **Trục Mamba/linear-attention:** vì bài này tránh cắt trong Mamba, với GatedDeltaNet nên thận trọng cắt head/channel của khối tuyến tính; xem bài 45.

## 10. Thí nghiệm gợi ý

1. Tính layer importance (MSE trước LM head) cho 24 layer decoder của LightOnOCR; vẽ theo chỉ số layer, đánh dấu GatedDeltaNet vs full attention, xem có thấy "quan trọng ở đầu và cuối" và có full-attention layer nào ít quan trọng không.
2. Liệt kê 20-50 ứng viên (số layer cắt x độ rộng MLP) ở ngân sách ~0.3B, distill rất ngắn mỗi cái, kiểm tra xem thứ hạng trước/sau distill có đổi như Bảng 6 không.
3. Đo tương quan giữa "next-token parent agreement" (rẻ) và điểm OCR cuối cùng để xem có dùng làm bộ lọc nhanh được không.

## 11. Tóm tắt khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Mamba-2 | Layer SSM, trạng thái cố định kích thước, rẻ ở context dài |
| Hybrid | Trộn Mamba-2 + attention + FFN; ~8% attention |
| MiniPuzzle | Importance → Conditional NAS → Distill (Minitron + Puzzle) |
| Layer importance | MSE trước LM head khi bỏ layer (128 mẫu) |
| Neuron importance | Activation tổng hợp của neuron FFN (1024 mẫu) |
| Parent agreement | Tỉ lệ child và parent đoán cùng token kế tiếp |
| Ràng buộc | < 31.7 GiB, FP4, context 1M token |
| Kết quả | 56B → 47B (5 attn, 44 Mamba, 49 FFN; FFN 30720), 1.2x nhanh hơn, 63B token |

## 12. Kết luận ngắn

Nemotron-H cho thấy hybrid Mamba-Transformer đạt độ chính xác ngang Transformer cùng cỡ nhưng nhanh hơn nhiều ở context dài. MiniPuzzle cho thấy có thể nén hybrid lớn bằng cách **cắt layer + neuron FFN, tránh trục Mamba**, rồi dùng distill ngắn để chọn kiến trúc và distill dài để phục hồi; 47B gần như ngang 56B. Bài học thực dụng: chọn ứng viên *sau* distill ngắn, đừng tin điểm ngay sau khi cắt.

Link gốc: https://arxiv.org/abs/2504.03624
