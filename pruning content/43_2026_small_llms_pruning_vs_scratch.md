# Small LLMs: Pruning vs. Training from Scratch (2026)

**Tác giả:** Yufeng Xu, Taiming Lu, Kunjun Li, Jiachen Zhu, Mingjie Sun, Zhuang Liu (Princeton University; New York University; Carnegie Mellon University).  
**Venue / arXiv:** arXiv:2606.14150 (v1: 12/06/2026; bản v3: 29/06/2026). Paper không nêu tên hội nghị trong văn bản được trích. Mã nguồn: github.com/zlab-princeton/pruning-vs-scratch.  
**Nguồn:** https://arxiv.org/abs/2606.14150  
**Chủ đề:** So sánh có kiểm soát số token giữa "prune mô hình lớn rồi retrain" và "train mô hình nhỏ cùng kiến trúc từ khởi tạo ngẫu nhiên", trên Llama-3.1-8B với 6 phương pháp prune (depth, width, 2:4, unstructured) và nhiều tỉ lệ prune; nhắc lại câu hỏi của "Rethinking the value of network pruning" ở quy mô LLM.  
**Vị trí trong lộ trình:** Đọc trước: 11 (Rethinking the value of pruning, cùng nhóm tác giả, kết luận gốc cho CNN), 08 (Lottery Ticket), 38-40 (Sheared LLaMA, Minitron). Bài này cho phép *đo* xem phần "prune" trong "prune + retrain" có thực sự đáng tiền không. Tiếp theo: 44-47 (hybrid/Mamba và các hướng prune khác).

> Ghi chú học tập bằng tiếng Việt, diễn giải và tóm tắt, không phải bản dịch. Số liệu lấy từ paper (một vài bảng trong bản text có sai khác nhỏ giữa các bảng; sẽ chỉ rõ). Phần "Ứng dụng cho dự án" là suy luận của chúng ta. Quan trọng: paper này retrain bằng LM loss thường, **không dùng distillation**, nên khác bối cảnh của dự án.

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

- Các bài trước trong lộ trình (Sheared LLaMA 38, Minitron 39-40, Puzzle 42) đều nói: prune mô hình lớn rồi retrain bằng một phần nhỏ token cho mô hình nhỏ tốt, rẻ hơn train từ đầu. Nhưng thường so sánh với *các mô hình cộng đồng* train với dữ liệu/công thức khác (Pythia, TinyLlama, Gemma...), không phải một baseline "train từ đầu cùng kiến trúc, cùng dữ liệu, cùng số token".
- **Khởi tạo ngẫu nhiên (random initialization / scratch)**: trọng số bắt đầu từ phân bố ngẫu nhiên, học từ đầu. **Khởi tạo bằng prune**: trọng số bắt đầu từ mô hình lớn đã cắt.
- Liu et al. (2019), bài 11: với CNN và structured pruning, train từ đầu kiến trúc đã cắt ngang hoặc hơn prune + fine-tune; nghĩa là cái có giá trị là *kiến trúc* tìm được, không phải trọng số thừa hưởng (coi prune như NAS). Với unstructured pruning, train từ đầu kém hơn ở quy mô lớn.
- Câu hỏi của paper: kết luận đó có đúng cho LLM (mô hình/dữ liệu lớn hơn, kiến trúc khác) không? Tác giả đổi cách nhìn: **prune là một chiến lược khởi tạo, không phải công cụ nén**, nên baseline đúng là mô hình cùng cỡ train từ đầu trên cùng luồng dữ liệu.
- Hai khái niệm token: *training token budget* (số token dùng để train mô hình nhỏ) và *total token budget* (cả quy trình, gồm token pretrain mô hình cha cộng token retrain).

## 2. Ý tưởng chính

Thiết kế hai so sánh công bằng theo token. (1) **Equal training token budget**: mô hình nhỏ train 50B token từ ngẫu nhiên (S50) so với mô hình cắt từ cha rồi retrain 50B token (P200-R50), cô lập tác dụng của khởi tạo. (2) **Equal total token budget**: baseline train từ đầu được cấp *cả* 250B token mà quy trình prune đã tiêu (200B pretrain cha + 50B retrain), tức S250. Kết quả: (1) prune thắng ở mọi phương pháp, nhưng lợi thế co lại khi tỉ lệ prune tăng và gần như biến mất ở 81.3% (depth); (2) khi baseline được cấp đủ tổng token thì train từ đầu bắt kịp hoặc vượt prune có cấu trúc (structured), còn prune mịn (sparse) vẫn giữ lợi thế nhẹ. Khuyến nghị: có cha mạnh + ngân sách token hạn chế thì prune; token dồi dào thì train từ đầu ở độ chi tiết thô cũng cạnh tranh.

## 3. Phương pháp chi tiết

### 3.1 Ký hiệu thiết kế

- $S_N$: train kiến trúc đích từ ngẫu nhiên trong $N$B token.
- $P_{200}\text{-}R_N$: pretrain mô hình cha (kiến trúc Llama-3.1-8B) 200B token, prune ở tỉ lệ đã chọn, rồi retrain $N$B token.
- $\text{Meta-}R_N$: như trên nhưng bắt đầu từ checkpoint Llama-3.1-8B chính thức của Meta (làm điểm tham chiếu).

Hai so sánh: **S50 vs P200-R50** (equal training budget) và **S250 vs P200-R50** (equal total budget). Dữ liệu: DCLM-Baseline-1.0, token hoá bằng tokenizer Llama-3.1, chunk 8192 token. 200B token pretrain và 50B token retrain là hai tập con không giao nhau; S250 dùng đúng hợp của hai tập đó; S50 dùng đúng 50B token của giai đoạn retrain. Nhờ vậy khác biệt chỉ đến từ khởi tạo.

### 3.2 Công thức huấn luyện

Lingua recipe (AdamW, $\beta=(0.9,0.95)$, eps 1e-8, weight decay 0.01, cosine decay xuống 10% đỉnh, warm-up 5%, batch 512, chuỗi 8192). Với mỗi thí nghiệm, quét LR đỉnh trong {1e-5, 3e-5, 1e-4, 3e-4, 1e-3} trên một đoạn ngắn và chọn LR cho train loss thấp nhất (tác giả nhấn mạnh: LR retrain phải chọn cẩn thận để tránh quên thảm hoạ, catastrophic forgetting). **Loss là LM loss thường; không distillation.**

### 3.3 Sáu phương pháp prune (Hình 2, Phụ lục C)

| Phương pháp | Đơn vị bị cắt | Tiêu chí |
|---|---|---|
| Minitron-depth (D) | cả layer (một khối liên tiếp) | tăng validation loss khi thay khối $[i,i+k)$ bằng identity: $i^\star=\arg\min_i L(\theta_{[i:i+k]\to id})-L(\theta)$ |
| Minitron-width (W) | kênh hidden, kênh MLP | $s_j=\|X_{:,j}\|_2$ (chuẩn activation) |
| FLAP | kênh hidden/MLP | phương sai đặc trưng đầu vào nhân chuẩn cột trọng số: $s_j=\frac1{N-1}\sum_n(X_{n,j}-\bar X_{:,j})^2\cdot\|W_{:,j}\|_2^2$ |
| Sheared LLaMA | layer, hidden, MLP, head | mask học được + Lagrangian (bài 38) |
| Wanda | từng trọng số | $s_{ij}=|W_{ij}|\cdot\|X_j\|_2$, chọn trong từng hàng output |
| SparseGPT | từng trọng số | tái tạo output từng layer (OBS), $\min\|XW^\top-X(\hat W\odot m)^\top\|_F^2$ |

Wanda và SparseGPT được chạy ở hai dạng: 2:4 (semi-structured, mỗi 4 trọng số giữ 2) và 50% unstructured (ký hiệu -U). Tỉ lệ prune chính là 50% (8.0B -> ~4.5B tham số). Với Minitron còn thử 62.5%, 75%, 81.3%.

**Cấu hình đích (Table 1; Llama-3.1-8B gốc: 32 layer, 32 head, hidden 4096, FFN 14336, 8.0B):**

| Phương pháp | Tỉ lệ | Layers | Heads | Hidden | FFN | Tham số |
|---|---|---|---|---|---|---|
| Minitron-D | 50% / 62.5% / 75% / 81.3% | 16 / 8 / 4 / 2 | 32 | 4096 | 14336 | 4.5B / 2.8B / 1.9B / 1.5B |
| Minitron-W | 50% | 32 | 32 | 3072 | 9216 | 4.5B |
| Minitron-W | 62.5% | 32 | 32 | 2432 | 6144 | 3.1B |
| Minitron-W | 75% | 32 | 32 | 1792 | 6016 | 2.0B |
| Minitron-W | 81.3% | 32 | 32 | 1536 | 4736 | 1.5B |
| FLAP | 50% | 32 | 20 | 4096 | 6656 | 4.5B |
| Sheared LLaMA | 50% | 32 | 32 | 3456 | 7552 | 4.6B |
| Wanda / SparseGPT | 50% | 32 | 32 | 4096 | 14336 | 4.5B (số khác 0) |

Khi phương pháp không xác định kiến trúc duy nhất (Minitron-W, FLAP, Sheared), tác giả liệt kê các cấu hình cùng số tham số, cắt không retrain, và chọn cấu hình có validation loss thấp nhất trên tập calibration (Phụ lục D.3). Với Minitron-W: giữ 32 head, ràng buộc $2h\le\text{MLP}\le4h$.

### 3.4 Đánh giá

Perplexity (C4, WikiText-103, WikiText-2, CNN/DailyMail; 256 chuỗi 8192 token) và 8 benchmark zero-shot (WinoGrande, ARC-C, ARC-E, HellaSwag, PIQA, SciQ, BoolQ, OBQA), báo trung bình. Độ lệch chuẩn đánh giá mỗi benchmark: WinoG ±1.3, ARC-C ±1.4, ARC-E ±1.0, HellaSwag ±0.4, PIQA ±1.0, SciQ ±0.8, BoolQ ±0.8, OBQA ±2.2 (điểm); chênh lệch nhỏ hơn mức này coi là nhiễu. Nhiều giá trị nên đọc trong ngữ cảnh đó (đặc biệt OBQA, BoolQ).

## 4. Thí nghiệm và kết quả chính

Mô hình cha pretrain của chính tác giả (P200): chỉ 200B token, nên yếu hơn Llama-3.1-8B của Meta: điểm trung bình 68.6 so với 74.0 (Table 10).

### 4.1 Tổng quan ở tỉ lệ 50% (Table 2; WikiText-2 PPL và Avg accuracy trên 8 benchmark)

| Phương pháp | PPL S50 | PPL P200-R50 | PPL S250 | Acc S50 | Acc P200-R50 | Acc S250 |
|---|---|---|---|---|---|---|
| Minitron-D | 10.77 | 9.41 | 9.01 | 60.7 | 64.4 | 66.2 |
| Minitron-W | 11.17 | 9.01 | 8.69 | 59.9 | 66.3 | 67.5 |
| FLAP | 11.34 | 7.97 | 8.82 | 57.7 | 66.5 | 65.1 |
| Sheared LLaMA | 11.28 | 8.63 | 8.85 | 58.4 | 65.9 | 65.3 |
| Wanda-2:4 | 10.55 | 9.15 | 8.60 | 61.5 | 66.7 | 66.2 |
| Wanda-U | 10.66 | 8.37 | 8.57 | 61.6 | 68.1 | 67.2 |
| SparseGPT-2:4 | 10.54 | 8.90 | 8.65 | 62.1 | 66.7 | 67.1 |
| SparseGPT-U | 10.47 | 8.44 | 8.59 | 61.9 | 67.5 | 66.3 |

Đọc bảng:
- **Cột P200-R50 so với S50:** pruning thắng ở mọi phương pháp, cả PPL lẫn accuracy (chênh accuracy từ +3.7 điểm ở Minitron-D đến +8.8 ở FLAP).
- **Cột P200-R50 so với S250:** structured: S250 vượt hoặc ngang (Minitron-D 66.2 vs 64.4; Minitron-W 67.5 vs 66.3; FLAP và Sheared ngược lại: P200-R50 nhỉnh hơn 66.5 vs 65.1 và 65.9 vs 65.3). Sparse: Wanda-U 68.1 vs 67.2, SparseGPT-U 67.5 vs 66.3 (pruning nhỉnh hơn); 2:4: 66.7 vs 66.2 (Wanda), 66.7 vs 67.1 (SparseGPT, S250 nhỉnh hơn).
- Perplexity: FLAP có PPL WikiText-2 thấp nhất trong số mô hình prune (7.97).
- Mô hình chính xác nhất: Wanda-U 68.1%.

### 4.2 Finding 1: cùng training token budget, prune thắng nhưng thu hẹp theo tỉ lệ prune

Các số cho Minitron (Table 11, accuracy trung bình 8 benchmark; $\Delta$ = P200-R50 trừ S50):

| Tỉ lệ (tham số) | Depth S50 | Depth P200-R50 | $\Delta$ | Width S50 | Width P200-R50 | $\Delta$ |
|---|---|---|---|---|---|---|
| 50% (4.5B) | 60.7 | 64.4 | +3.7 | 59.9 | 66.3 | +6.4 |
| 62.5% | 55.2 | 56.4 | +1.2 | 56.9 | 64.9 | +8.0 |
| 75% | 51.0 | 51.7 | +0.7 | 56.0 | 61.0 | +5.0 |
| 81.3% (1.5B) | 48.3 | 48.1 | -0.2 | 54.2 | 59.3 | +5.1 |

- **Depth:** lợi thế co về ~0 ở 81.3% (PPL WikiText-2 ngang nhau, 25.16 vs 25.48; accuracy 48.3 vs 48.1).
- **Width:** vẫn dương ở mọi tỉ lệ (+5.1 ở 81.3%), nghĩa là cắt width giữ được nhiều "kiến thức" từ cha hơn cắt layer ở tỉ lệ cao.
- Ví dụ per-benchmark ở 50%: Minitron-D thắng S50 trên cả 8 benchmark, +4.5 ARC-C, +6.1 HellaSwag; Minitron-W +8.2 ARC-C, +10.2 HellaSwag. Paper giải thích Minitron-W giữ nhiều năng lực của cha hơn Minitron-D (theo Minitron).
- Sparse: tinh hơn thì thắng nhiều hơn. Wanda: PPL WikiText-2 tốt hơn S50 1.40 (2:4) so với 2.30 (unstructured); SparseGPT: ARC-C +2.0% (2:4) so với +8.1% (unstructured).
- **Điều quan trọng về xếp hạng trước retrain:** Table 10 cho thấy P200 + Minitron-W ngay sau khi cắt có PPL C4 24686.5 và WikiText-2 36761.5 (gần như hỏng), nhưng retrain 50B token phục hồi hoàn toàn (66.3 avg); tác giả kết luận PPL ngay sau prune là chỉ báo kém của chất lượng sau retrain (giống phát hiện "đảo thứ hạng" của Minitron, bài 39). So sánh: P200 + Minitron-D sau cắt: PPL C4 504.0; avg 41.5. Sheared-LLaMA sau cắt: avg 63.3 (P200), Wanda-U: 64.3, SparseGPT-U: 65.4.

### 4.3 Finding 2: cho scratch đủ tổng token thì structured bị bắt kịp

- Minitron-D: S250 vượt P200-R50 trên mọi benchmark (PPL WikiText-2 thấp hơn 0.40; accuracy cao hơn 0.6-4.0 điểm; trung bình +1.8 trong Table 16 với Δ = scratch - pruned).
- Minitron-W: biên nhỏ hơn (OBQA: từ 4.0 xuống 2.2) nhưng S250 vẫn nhỉnh hơn (+1.2 avg).
- FLAP và Sheared LLaMA: P200-R50 vẫn nhỉnh hơn S250 ở accuracy trung bình (FLAP 66.5 vs 65.1, Sheared 65.9 vs 65.3) và ở PPL (FLAP 7.97 vs 8.82). Lợi thế còn sót của FLAP là BoolQ (+8.7 so với S250), của Sheared là OBQA (+3.8), theo Phụ lục H.4.
- Sparse (2:4): S250 PPL tốt hơn nhưng thua P200-R50 trên 6/8 benchmark (Wanda-2:4: −1.8 WinoG, −1.2 ARC-C). Unstructured (Wanda-U): S250 thua trên 6/8 benchmark, chỉ BoolQ và OBQA tốt hơn (+0.4, +0.8).
- **Chú ý (phát hiện của chúng ta khi đối chiếu bảng):** lợi thế của sparse ở tổng token bằng nhau khá mong manh: Table 16 cho avg Wanda-U 68.1 vs 67.2 (+0.9), SparseGPT-U 67.5 vs 66.3 (+1.2), nhưng SparseGPT-2:4 67.1 (S250) hơn 66.7 (P200-R50); phụ lục cũng tự nhắc rằng đây là tuyên bố theo từng benchmark chứ không phải toàn diện. Cũng trong Phụ lục: FLAP (structured) có PPL tốt nhất tổng thể, vượt mọi sparse, là một phản ví dụ cho khẩu hiệu "sparse > structured".

### 4.4 Finding 3: tinh hơn thì khởi tạo tốt hơn nhưng phải trả bằng hiệu quả phần cứng

Cùng tỉ lệ prune, hạt càng mịn càng thắng scratch nhiều. Nhưng trên TPU v4-256 (không có sparse tensor core), cả 2:4 và unstructured chạy tốc độ như mô hình dense (1.0x). Table 9 (mô hình P200, FLOPs lý thuyết một forward chuỗi 2048; "speedup" là tăng tốc đo trên TPU):

| Phương pháp | Tỉ lệ | FLOPs (TF) | Speedup |
|---|---|---|---|
| Base | 0% | 32.9 | 1.0x |
| Minitron-D | 50 / 62.5 / 75 / 81.3% | 17.5 / 9.8 / 6.0 / 4.1 | 1.9x / 4.2x / 7.2x / 11.2x |
| Minitron-W | 50 / 62.5 / 75 / 81.3% | 19.1 / 12.6 / 9.8 / 7.9 | 1.4x / 2.0x / 2.4x / 2.9x |
| FLAP | 50% | 17.7 | 1.5x |
| Sheared LLaMA | 50% | 18.9 | 1.4x |
| Wanda, SparseGPT (50% hoặc 2:4) | 50% | 16.5 | 1.0x |

(Nhận xét của chúng ta: cột Speedup có vẻ là tốc độ bước *huấn luyện*, và các giá trị 4.2x-11.2x của depth cao hơn nhiều so với tỉ lệ FLOPs 32.9/9.8=3.4; chưa rõ cách đo, nên chỉ coi là thứ tự tương đối.) Chú thích paper: trên GPU có sparse tensor core (A100/H100), 2:4 mới tăng tốc suy luận (cuSPARSELt), unstructured thì không.

### 4.5 Token scaling (Tables 12-15, 4 phương pháp structured, 50%, trung bình accuracy)

Minitron-depth: S10 49.8 -> P200-R10 60.7; S30 55.5 -> 62.6; S50 60.7 -> 64.4; S250 66.0 -> 67.0 (P200-R250); S500 67.3 -> 68.1 (P200-R500). Minitron-width: S10 46.8 -> 63.3; S50 59.0 -> 65.7; S250 67.4 -> 68.3; S500 68.5 -> 69.6. FLAP: S500 66.9 -> 70.7. Sheared LLaMA: S500 67.0 -> 69.6 (Δ ≥ +2.6 ở mọi budget, là phương pháp structured duy nhất không bao giờ vượt về 0). Với cùng số token retrain $N$ (không phải tổng token), pruned luôn dẫn nhưng khoảng cách co lại.

Meta-RN (cắt từ checkpoint Meta thật) đứng đầu hơn P200-RN ở budget thấp (Minitron-D ΔS10 +12.1 vs +10.9), khoảng cách đóng dần khi N tăng; tác giả giải thích là tín hiệu còn lại của hỗn hợp pretrain gốc của Meta.

## 5. Phân tích / ablation

- **Tỉ lệ prune:** lợi thế của prune giảm khi tỉ lệ tăng; depth mất sạch ở 81.3%, width vẫn giữ +5.1.
- **Độ mịn (granularity):** unstructured > 2:4 > width > depth về lợi thế so với scratch cùng tỉ lệ.
- **Token retrain:** lợi thế co theo $N$ nhưng với avg accuracy không biến mất đến 500B (Minitron-D +0.8, Minitron-W +1.1, FLAP +3.8, Sheared +2.6 so với S500). Phụ lục H.3 viết rằng BoolQ là benchmark đầu tiên đổi dấu khi N tăng (−2.0, −5.4, −4.6 ở 500B) và suy ra BoolQ thưởng quy mô dữ liệu hơn khởi tạo. **Nhưng khi đối chiếu bảng, các giá trị −2.0 / −5.4 / −4.6 nằm ở cột WinoGrande (Minitron-D) và OBQA (Minitron-W, FLAP), còn cột BoolQ ở 500B vẫn dương (+0.4, +7.0, +13.5).** Có vẻ phần chữ gán nhầm tên benchmark; ta nên tin bảng và nhớ rằng WinoG và OBQA mới là các điểm đổi dấu (OBQA nhiễu ±2.2).
- **Giải thích cơ chế (Phụ lục G, tác giả đề xuất, dựa trên tài liệu có sẵn, chưa kiểm chứng trực tiếp):** (1) Lottery Ticket và Evci et al.: tối ưu mô hình thưa với mask cố định từ ngẫu nhiên khó (kẹt cực tiểu cục bộ); (2) Liu et al. 2019: trọng số của mô hình unstructured sau fine-tune lệch với bản scratch nhiều hơn structured, nên giá trị trọng số thừa hưởng, không chỉ mẫu thưa, là nguồn lợi thế.
- **Nội dung "cái gì được chuyển từ cha":** tác giả suy ra rằng thông tin ở mức trọng số được chuyển sang (sparse), còn hình dạng kiến trúc thì không (structured, có thể bù bằng train lâu); đề xuất công thức bất đối xứng: structured chỉ đáng khi token là ràng buộc chính.
- **Cảnh báo về tính công bằng:** "headline pruning numbers reported under fixed retraining budgets may overstate the practical benefit once the cost of the parent is amortized in".

## 6. Giới hạn

*Tác giả nêu (Phụ lục A):*
1. Chỉ một họ mô hình (Llama-3.1-8B), không thử Qwen/Gemma/OLMo hoặc cỡ nhỏ/lớn hơn.
2. Một corpus (DCLM); khoảng cách sparse-structured có thể đổi theo phân phối dữ liệu; Meta-RN không kiểm soát được hỗn hợp pretrain của Meta.
3. **Không có baseline knowledge distillation:** so với LM-loss retraining thường, không có pipeline distillation như Minitron/Sheared, có thể thu hẹp hoặc xoá khoảng cách cho structured.
4. Một lần xáo trộn dữ liệu, không đo phương sai giữa các lần chạy.
5. Một kiểu khởi tạo ngẫu nhiên chuẩn (không µP, không depth-aware scaling, không warm-start từ mô hình nhỏ).

*Chúng ta nhận xét:*
- Mô hình cha P200 chỉ pretrain 200B token, yếu hơn nhiều so với một mô hình cha thực tế (68.6 so với 74.0 avg); cha yếu làm lợi thế của prune bị đánh giá thấp, hoặc ngược lại, khó nói. Paper có Meta-RN để cho thấy xu hướng không đổi.
- Kết luận "train từ đầu bắt kịp structured ở 250B token" áp dụng cho scale 4.5B/200B+; ở quy mô token nhỏ (dưới 10B token, như dự án) thì mọi bảng đều ủng hộ prune.
- Có vài sai khác nhỏ giữa các bảng cho cùng một thí nghiệm (ví dụ Minitron-W 50%: Table 2/16/11 báo S50 avg 59.9, P200-R50 66.3; Table 13 báo 59.0 và 65.7; FLAP: Table 2 báo 66.5, Table 14 báo 67.0), có thể do lần đánh giá khác nhau; chênh nhỏ so với các khoảng cách chính, nhưng cần lưu ý khi trích số.
- Tập benchmark là trắc nghiệm ngôn ngữ ngắn (nhiễu ±1-2 điểm); chênh lệch dưới 1 điểm trong kết luận về sparse ở S250 nên được coi là chưa chắc chắn.
- Số liệu "speedup" ở Table 9 mô tả tốc độ trên TPU huấn luyện chứ không phải độ trễ suy luận trên GPU của người dùng.

## 7. Liên hệ với các bài khác

- **11 Rethinking the value of network pruning (Liu et al. 2019), 08 Lottery Ticket, 09 Linear mode connectivity:** bài này là phiên bản LLM của câu hỏi ở bài 11: kết luận chia theo độ mịn, đúng như bài 11 gợi ý (structured: giá trị ở kiến trúc; unstructured: giá trị ở trọng số).
- **38 Sheared LLaMA, 39-40 Minitron:** các phương pháp được thử lại trong cùng điều kiện; bài này xác nhận prune có lợi khi token ít, nhưng ngầm nói rằng phần lớn lợi thế của Sheared/Minitron so với "mô hình cộng đồng" có thể do thừa hưởng token của cha (hàng nghìn tỷ) chứ không chỉ do thuật toán.
- **42 Puzzle:** Puzzle dùng token cực ít (45B) từ cha 15T token; bài này cho thấy lợi thế đó phụ thuộc vào việc chi phí cha đã "trả xong".
- **23-24 SparseGPT, Wanda; 18 2:4 sparsity:** các baseline sparse; kết quả cho thấy chúng tạo khởi tạo mạnh nhất nhưng ít lợi suy luận trên phần cứng không hỗ trợ.
- **33-34 Shortened LLaMA, SLEB, 31 ShortGPT; 37 Repair:** depth pruning: bài này cho thấy depth pruning mất lợi thế khi tỉ lệ cao.
- **47 Jet-Nemotron, 44-46:** các hướng khác về thiết kế mô hình nhỏ.

## 8. Ứng dụng cho dự án (suy luận)

*Mục này là suy luận của chúng ta.*

- **Câu hỏi "có nên prune hay train mô hình nhỏ từ đầu" của dự án khác với paper ở hai điểm lớn.** (a) *Cha đã có sẵn*: LightOnOCR-3-0.8B đã được train, chi phí của cha là sunk cost; kịch bản liên quan là "equal training token budget" (S50 vs P200-R50), nơi prune thắng ở mọi phương pháp và mọi tỉ lệ ≤75%. Kịch bản S250 (cộng chi phí cha) không áp dụng. (b) *Bạn có teacher 4B và dùng distillation*; paper không kiểm tra distillation (hạn chế #3) nên không thể dùng trực tiếp để kết luận khoảng cách khi có KD.
- **Tỉ lệ prune của dự án:** 0.8B -> 0.3B là ~62.5% (số tham số), 0.8B -> 0.1B là ~87.5%, vượt cả mốc 81.3% trong paper. Ngoại suy từ Table 11: với depth pruning ở tỉ lệ cao, lợi thế của khởi tạo bằng prune có thể gần như bằng 0 so với train từ đầu (cùng token); với width pruning vẫn còn khoảng +5 điểm. Điều này gợi ý (không chứng minh) rằng ở mục tiêu 0.1B nên thử cả một baseline "train/distill từ khởi tạo ngẫu nhiên" bên cạnh bản prune, đặc biệt khi dự định cắt layer.
- **Liên hệ với quan sát của dự án ("cắt neuron MLP hồi phục tệ hơn cắt layer"):** dữ liệu Table 11 ở 50% và 62.5% cho thấy width (cắt hidden + MLP) có Δ lớn hơn depth, tức khởi tạo bằng width tốt hơn so với scratch, nhưng đây là so với *scratch*, không phải so sánh trực tiếp "tốc độ hồi phục" giữa width và depth. Trong Table 11, nếu ghép lại: P200-R50 avg (cùng 4.5B) width 66.3 vs depth 64.4, tức width hồi phục tốt hơn depth sau 50B token LM-loss. Điều này lặp lại mâu thuẫn đã thấy với Minitron (39): kết quả của dự án có thể do hybrid GatedDeltaNet, do ngân sách nhỏ, hoặc do tiêu chí chọn neuron. Cần thí nghiệm trực tiếp.
- **Post-prune loss không đáng tin để chọn cấu hình** (Table 10: PPL 36761 mà phục hồi hoàn toàn). Nên chọn bằng retrain ngắn, không bằng điểm ngay sau cắt.
- **Sparse không phù hợp mục tiêu:** GPU ≤8 GB, batch 1 và yêu cầu tăng tốc thật: unstructured không tăng tốc, 2:4 chỉ tăng trên GPU có sparse tensor core; vì vậy chỉ structured là lựa chọn thực tế. Kết quả "sparse thắng" nhìn chung ít liên quan.
- **Cảnh báo chung:** đo kết quả bằng CER/WER chứ không bằng benchmark trắc nghiệm của paper.

## 9. Thí nghiệm gợi ý

1. **Baseline scratch-with-KD:** huấn luyện kiến trúc đích 0.3B (và 0.1B) từ khởi tạo ngẫu nhiên bằng cùng KD từ teacher 4B và cùng số token như bản đã prune; so CER. Đây là kiểm tra trực tiếp xem prune còn đáng không khi đã có teacher và ở tỉ lệ cao.
2. **Quét tỉ lệ:** với cùng ngân sách KD, so depth và width ở 40%, 60%, 80%, 90% (0.5B, 0.3B, 0.15B, 0.08B) như Table 11; vẽ Δ(prune − scratch) theo tỉ lệ để xem điểm giao của depth và width trong kiến trúc hybrid.
3. **Đường cong theo token:** thêm điểm token (vd 0.1/0.3/1/3B) cho prune và scratch cùng teacher, để biết ở ngân sách nhỏ của dự án khoảng cách thế nào (paper chỉ xuống 10B token).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Pruning as initialization | Trọng số đã cắt chỉ là điểm bắt đầu cho mô hình nhỏ; baseline đúng là cùng kiến trúc train từ ngẫu nhiên |
| $S_N$ / $P_{200}$-$R_N$ / Meta-$R_N$ | scratch $N$B token / pretrain cha 200B, prune, retrain $N$B / như vậy nhưng từ checkpoint Meta |
| Equal training budget | S50 vs P200-R50 |
| Equal total budget | S250 vs P200-R50 (250 = 200 pretrain + 50 retrain) |
| Minitron-depth | bỏ khối liên tiếp làm tăng val loss ít nhất |
| Minitron-width | giữ kênh có chuẩn activation $\|X_{:,j}\|_2$ lớn |
| Wanda | $|W_{ij}|\cdot\|X_j\|_2$ |
| Finding 1 | prune thắng ở cùng token train; lợi thế co khi tỉ lệ tăng, mất ở 81.3% với depth |
| Finding 2 | scratch được cấp đủ tổng token bắt kịp structured; sparse còn lợi thế nhẹ |
| Finding 3 | độ mịn cao hơn cho khởi tạo tốt hơn nhưng ít lợi suy luận |

## 11. Kết luận ngắn

Khi tính cả chi phí token của mô hình cha, lợi thế của "prune rồi retrain" so với train từ đầu thu hẹp lại: structured pruning bị bắt kịp ở 250B token, chỉ sparse pruning giữ được chút lợi thế (nhưng khó tăng tốc). Khi token là ràng buộc chính và cha đã có sẵn thì prune thắng rõ, với lợi thế giảm theo tỉ lệ prune. Với dự án (cha có sẵn, teacher 4B, ngân sách token nhỏ, tỉ lệ prune cao), kết luận đáng nhớ nhất là: depth pruning mất ưu thế khi quá mạnh, post-prune loss không dự đoán được kết quả sau retrain, và cần tự kiểm tra baseline "scratch + KD" vì paper không có distillation.

Link gốc: https://arxiv.org/abs/2606.14150
