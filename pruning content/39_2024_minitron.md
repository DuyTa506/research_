# Compact Language Models via Pruning and Knowledge Distillation (2024)

**Tác giả:** Saurav Muralidharan, Sharath Turuvekere Sreenivas (đóng góp ngang nhau), Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, Pavlo Molchanov (NVIDIA).  
**Venue / arXiv:** arXiv:2407.14679 (v1: 19/07/2024; bản v2: 04/11/2024). Paper không ghi tên hội nghị trong văn bản được trích.  
**Nguồn:** https://arxiv.org/abs/2407.14679  
**Chủ đề:** Pipeline "prune rồi retrain bằng distillation" cho LLM (họ Minitron, cắt từ Nemotron-4 15B xuống 8B và 4B); so sánh depth vs width, importance dựa trên activation, các loss distillation, tìm kiến trúc nhẹ (lightweight NAS).  
**Vị trí trong lộ trình:** Đọc trước: 38 (Sheared LLaMA, cũng prune + train tiếp nhưng chỉ dùng LM loss), 31 (ShortGPT, chỉ số Block Importance), 33 (Shortened LLaMA, chỉ số PPL cho layer), 35 (LaCo, "layer collapse"). Bài này thêm: bộ "best practices" được kiểm chứng bằng ablation, đặc biệt (a) width thắng depth *chỉ sau khi retrain*, (b) distillation bằng KL trên logits thắng huấn luyện thường. Tiếp theo: 40 (áp dụng cho Llama-3.1 8B), 41-42 (Flextron, Puzzle).

> Ghi chú học tập bằng tiếng Việt, diễn giải và tóm tắt, không phải bản dịch. Số liệu lấy từ paper; phần "Ứng dụng cho dự án" là suy luận của chúng ta. Một số bảng/hình trong bản text bị lệch cột; chỗ nào không chắc, ghi rõ.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
4. [Depth vs width và việc đảo thứ hạng sau retrain (chi tiết)](#4-depth-vs-width-và-việc-đảo-thứ-hạng-sau-retrain-chi-tiết)
5. [Các lựa chọn loss distillation (chi tiết)](#5-các-lựa-chọn-loss-distillation-chi-tiết)
6. [Thí nghiệm và kết quả chính](#6-thí-nghiệm-và-kết-quả-chính)
7. [Phân tích / ablation còn lại](#7-phân-tích--ablation-còn-lại)
8. [Giới hạn](#8-giới-hạn)
9. [Liên hệ với các bài khác](#9-liên-hệ-với-các-bài-khác)
10. [Ứng dụng cho dự án (suy luận)](#10-ứng-dụng-cho-dự-án-suy-luận)
11. [Thí nghiệm gợi ý](#11-thí-nghiệm-gợi-ý)
12. [Tóm tắt công thức / khái niệm](#12-tóm-tắt-công-thức--khái-niệm)
13. [Kết luận ngắn](#13-kết-luận-ngắn)

---

## 1. Bối cảnh

- Các hãng thường train cả một họ mô hình (ví dụ Llama-2 7B/13B/70B), mỗi cỡ từ đầu với hàng nghìn tỷ token. Rất tốn.
- Ý tưởng thay thế: train *một* mô hình lớn, rồi **prune** (cắt bớt tham số) để lấy các mô hình nhỏ, sau đó **retrain** (train lại ngắn để hồi phục độ chính xác) bằng một phần nhỏ dữ liệu.
- **Knowledge distillation (KD)**: thay vì học từ nhãn thật (token kế tiếp), mô hình nhỏ (student) học bắt chước đầu ra của mô hình lớn (teacher), thường bằng cách khớp phân phối xác suất trên từ vựng. Tín hiệu từ teacher "giàu" hơn one-hot (cho biết token nào gần đúng), nên học nhanh hơn.
- Trước paper này, các công trình prune LLM (LLM-Pruner, SliceGPT, LaCo, ShortGPT, Sheared LLaMA) chỉ tập trung một trục (depth *hoặc* width) và retrain bằng loss thường hoặc chỉ fine-tune ít. Tác giả nói họ là những người đầu tiên dùng distillation từ teacher chưa nén để retrain mô hình đã prune có cấu trúc.
- Hạn chế của phương pháp mask-học-được (Sheared, LLM-Pruner, SliceGPT): cần gradient (tốn bộ nhớ) và nhiều dữ liệu. Minitron muốn chỉ dùng **forward pass**.

## 2. Ý tưởng chính

Tính độ quan trọng của từng layer, head, neuron MLP, kênh embedding chỉ bằng activation trên 1024 mẫu (không cần gradient). Xếp hạng, cắt bỏ phần ít quan trọng, rồi retrain mô hình đã cắt bằng KL-divergence trên logits so với teacher (chính là mô hình gốc). Khi cần chọn kiến trúc, liệt kê <20 cấu hình có cùng số tham số, retrain nhẹ mỗi cái (~1.8B token) rồi chọn cái tốt nhất để train tiếp. Kết quả: Minitron 8B (cắt từ Nemotron-4 15B) dùng 94B token (khoảng 40 lần ít hơn Nemotron-3 8B) mà ngang hoặc hơn nhiều mô hình cùng cỡ. Hai phát hiện "ngược trực giác": thứ hạng các lựa chọn cắt *đảo* sau vài trăm bước retrain, và width thắng depth chỉ sau retrain.

## 3. Phương pháp chi tiết

### 3.1 Ký hiệu

MLP: $\mathrm{MLP}(X)=\delta(X W_1^{\top})\,W_2$, với $d_{model}$ (embedding) và $d_{hidden}$ (chiều ẩn MLP). MHA: $\mathrm{Concat}(\text{head}_1..\text{head}_L)W_O$. LayerNorm có $\gamma,\beta$.

### 3.2 Độ quan trọng dựa trên activation (chỉ forward)

Dùng tập calibration $D$ 1024 mẫu ngẫu nhiên từ dữ liệu train. Với batch $B$ và chuỗi $S$:

$$\boxed{F^{(i)}_{head}=\sum_{B,S}\big\|\mathrm{Attn}(XW_{Q,i},XW_{K,i},XW_{V,i})\big\|_2,\quad F^{(i)}_{neuron}=\sum_{B,S}X\,W_1^{i\top},\quad F^{(i)}_{emb}=\sum_{B,S}\mathrm{LN}(X)_i}$$

Tức là: head quan trọng nếu output attention có chuẩn lớn; neuron MLP quan trọng nếu activation lớn; kênh embedding quan trọng nếu output LayerNorm của kênh đó lớn. Phép "cộng" trên batch và chuỗi không phải lúc nào cũng tối ưu, nên paper thử 3 hàm gộp: mean(|S|), L2 norm, variance. Điểm theo layer được cộng để ra điểm toàn mạng cho từng trục.

**Depth (layer)** có 2 chỉ số:
1. *PPL*: bỏ riêng layer đó và đo perplexity tăng bao nhiêu (từ Shortened LLaMA, bài 33).
2. *Block Importance (BI)* (từ ShortGPT, bài 31): khoảng cách cosine giữa input và output của layer:

$$\mathrm{BI}_i=1-\mathbb E_{X,t}\frac{X_{i,t}^{\top}X_{i+1,t}}{\|X_{i,t}\|_2\|X_{i+1,t}\|_2}$$

BI tính cho mọi layer trong một lần forward, nhanh hơn PPL; có thể mở rộng cho cụm layer liên tiếp.

### 3.3 Cắt mô hình

Xếp hạng theo điểm, rồi cắt trực tiếp ma trận trọng số: neuron/head thì cắt hàng/cột trong MLP/MHA; kênh embedding thì cắt chiều embedding của mọi ma trận MLP, MHA và LayerNorm.

**Gộp phần dư của head bị cắt:** khi cắt $L$ head xuống $K$ head, head mới thứ $i$ là $\text{head}_i+(\text{head}_i-\text{head}_{2K-i+1})$ cho $i\in[K-(L-K),K]$ (tương tự Layer Collapse của LaCo cho depth); với GQA chỉ áp dụng cho query head. Paper nói có tăng độ chính xác nhưng không ablation riêng ở phần đã đọc.

### 3.4 Lightweight NAS (Figure 3)

```
Input: ngân sách tham số (vd 8B ± 5%), không gian tìm kiếm
  Layers ∈ [29..32], Heads ∈ {32,48}, MLP expansion ∈ {2.5,3,3.5,4}, Embedding ∈ {4096,4680,5120,5632,6144}  (Table 12, cho 8B)
1. Liệt kê mọi cấu hình thoả ngân sách        # 15 ứng viên cho 8B, 18 cho 4B
2. Với mỗi ứng viên: cắt (từ xếp hạng importance) + retrain nhẹ ~1.8B token (400 bước)
3. Chọn ứng viên có validation loss thấp nhất, retrain đầy đủ
```

Không dùng genetic/Bayesian search vì không gian đã đủ nhỏ. Tác giả nói LoRA cũng có thể dùng ở bước retrain nhẹ (chưa thử).

### 3.5 Retrain bằng distillation

Phân phối của token $x_i$ với nhiệt độ $\tau$: $p(x_i,\tau)=\exp(x_i/\tau)/\sum_j\exp(x_j/\tau)$.

$$L_{logits}=\frac1l\sum_{k=1}^{l}\mathrm{Loss}\big(p_t^k(x,\tau),\,p_s^k(x,\tau)\big)$$

$$L_{is}=\frac1l\sum_{i=1}^{l}\sum_{k\in H}\mathrm{Loss}_k\big(h_t^{ki},h_s^{ki}\big)$$

$$\boxed{L=L_{CLM}+L_{logits}+\alpha\,L_{is}},\quad \alpha=\frac{L_{logits}}{L_{is}}\ (\text{tính động})$$

$L_{CLM}$ là cross-entropy với nhãn thật; $L_{is}$ là loss trên trạng thái trung gian (intermediate states: embedding, output các block...), khác chiều giữa student và teacher được xử lý bằng một phép biến đổi tuyến tính học chung, hidden states lấy sau LayerNorm. Do độ lớn $L_{logits}$ và $L_{is}$ chênh nhiều nên đặt $\alpha$ theo tỉ lệ động hiệu quả hơn hằng số. **Nhưng phát hiện cuối cùng**: dùng *chỉ* $L_{logits}$ (KLD) là tốt nhất trong hầu hết trường hợp (xem phần 5).

## 4. Depth vs width và việc đảo thứ hạng sau retrain (chi tiết)

Đây là phần quan trọng nhất cho dự án (vì dự án cũng đang so sánh cắt layer và cắt neuron MLP).

### 4.1 Thiết kế thí nghiệm

Cắt Nemotron-4 15B xuống cỡ Nemotron-3 8B theo các cách khác nhau, sau đó retrain với ~1.8B token (400 bước) và đo hai thứ: *distillation loss* (KL trên logits, trước → sau retrain) và *LM validation loss* sau retrain. Ký hiệu cột trong Table 1: DEP = cắt depth, MLP = cắt neuron MLP, ATT = cắt head, EMB = cắt kênh embedding.

**Table 1 (số copy từ paper):**

| DEP | MLP | ATT | EMB | Distillation loss (trước → sau retrain) | LM val loss |
|---|---|---|---|---|---|
| có | có | có | có | 5.35 → 0.38 | 2.062 |
| không | có | có | có | 6.33 → 0.37 | **2.049** |
| không | có | có | không | 5.07 → 0.42 | 2.101 |
| có | không | không | không | 8.35 → 0.49 | 2.155 |
| Train từ đầu (random init) | | | | 12.27 → 2.34 | 3.953 |

(Hàng cuối: ghi chú của paper nói hàng này cho biết sự thay đổi loss với Nemotron-3 8B; bản text hơi khó đọc nên diễn giải ở đây là: khởi tạo ngẫu nhiên và train cùng ngân sách thì loss rất tệ.)

Đọc bảng:
- **Ngay sau khi cắt (trước retrain)**, cấu hình "chỉ MLP + ATT" (5.07) và "depth + tất cả" (5.35) có distillation loss *thấp hơn* "width đầy đủ" (6.33). Nghĩa là nhìn ngay sau cắt, giữ embedding nguyên và/hoặc có cắt layer có vẻ tốt hơn.
- **Sau retrain 400 bước**, thứ hạng đảo: width đầy đủ (MLP+ATT+EMB, không cắt layer) cho LM loss thấp nhất 2.049; thêm cắt depth thì 2.062; chỉ cắt MLP+ATT (không cắt EMB) 2.101; chỉ cắt depth 2.155.
- Hai "đảo" khác nhau, đều được tác giả nêu: (i) cắt neuron + head thôi thì ban đầu hơn cắt cả embedding, nhưng sau vài bước retrain thì đảo; (ii) width hơn depth, nhưng chỉ sau retrain.

### 4.2 Depth-only so với width-only (Table 10)

Bỏ 16 layer ít quan trọng nhất của 15B theo 2 chỉ số, so với cắt width, so với kết hợp (bỏ 4 layer + width):

| Mô hình Minitron 8B | Tham số | LM loss (sau retrain 1.8B token) |
|---|---|---|
| Depth (PPL) | 9.39B | 2.155 |
| Depth (BI) | 9.39B | 2.177 |
| Width | 7.74B | **2.049** |
| Depth + Width | 7.91B | 2.062 |

Tác giả nhấn mạnh: width-only có *ít tham số nhất* mà vẫn tốt nhất. Hai biến thể depth còn nhiều tham số hơn (9.39B) nhưng loss kém hơn rõ. Hình 6: so width-alone (32 layer) với width + 28 layer, thứ hạng đảo trước bước ~200 (~0.8B token) của retrain, nên "cần một giai đoạn retrain nhẹ trước khi so sánh".

### 4.3 Hệ quả thực hành mà tác giả rút ra

- **Best practice #4:** "Prefer width pruning over depth for the model scales we consider (≤ 15B)".
- Không so sánh các lựa chọn cắt ngay sau khi cắt; phải retrain nhẹ trước khi xếp hạng (best practice #9). Hình 9 (retrain 15 ứng viên 8B): thứ hạng thay đổi đáng kể đến khoảng bước 300 rồi mới ổn định.
- Cắt sâu quá nhiều thì cũng mất: Phụ lục A.5 (hình 7): với depth, Hellaswag và PIQA giữ được tới 31 layer, MMLU giữ tới 26 layer rồi giảm mạnh ở ~20 layer. Nghĩa là vài layer có thể bỏ gần như không mất mát với retrain tối thiểu (khớp với ShortGPT).
- Khi kiến trúc đích vẫn dùng bản depth-pruned: dùng one-shot (bỏ 16 layer một lần rồi distill 30.2B token) tốt hơn lặp (bỏ 1 layer, distill 1.8B, lặp 16 lần), kể cả khi lặp nhận 86.4B token.

### 4.4 Giới hạn của kết luận depth vs width

- Chỉ là một họ mô hình (Nemotron-4, transformer thuần, ≤15B), một lượng retrain 1.8B token (so sánh chính), và tập đánh giá là loss/benchmark ngôn ngữ. Ngoài ra, cấu hình "depth-only" ở đây nặng hơn (9.39B so với 7.74B), nên là so sánh không cân bằng tham số (nhưng thiên về bất lợi cho width, nên kết luận width thắng càng mạnh). Đây là nhận xét của chúng ta.
- Paper chưa phân tích *tại sao*; ghi nhận chỉ là thực nghiệm.

## 5. Các lựa chọn loss distillation (chi tiết)

### 5.1 Distillation so với huấn luyện thường (Table 11)

Cắt xuống 4B, so các cách (∗ = cùng lượng compute; distillation tốn thêm forward của teacher nên nhận ít token hơn):

| Cách | Token | Hellaswag | MMLU |
|---|---|---|---|
| 4B random init | 150B∗ | 46.22 | 24.36 |
| 4B random init | 400B | 48.23 | 26.24 |
| 4B Pruned (cắt từ 15B), retrain thường | 150B∗ | 50.85 | 24.57 |
| 4B Pruned-Distill (cắt từ 15B) | 100B∗ | 51.04 | 37.81 |
| 4B Pruned-Distill (cắt từ Minitron 8B) | 100B∗ | 52.04 | 42.45 |

Tác giả diễn giải: prune + distill hơn train từ đầu cùng compute +4.8% Hellaswag, +13.5% MMLU; cắt từ 8B thay vì 15B thêm +1% và +4.6%. MMLU của random-init và pruned-không-distill đều gần ngẫu nhiên (~24-26%).

### 5.2 Hàm khoảng cách trên logits (Bảng 15-16, thí nghiệm trên Nemotron-3 8B và Minitron 8B-Depth-pruned)

Thử KLD (forward KL), reverse KLD (R-KLD), MSE, cosine; ngoài ra WSL-KD cũng kém hơn KLD.

| Loss (Table 15, Nemotron-3 8B) | LM loss | WikiText PPL |
|---|---|---|
| $L_{CLM}+L_{logits}$ (MSE) | 2.144 | 9.007 |
| $L_{CLM}+L_{logits}$ (R-KLD) | 2.140 | 9.008 |
| $L_{CLM}+L_{logits}$ (Cosine) | 2.134 | 8.965 |
| $L_{CLM}+L_{logits}$ (KLD) | 2.117 | 8.791 |
| $L_{logits}$ (KLD) *chỉ* logits | 2.107 | 8.720 |

Table 16 (Minitron 8B-Depth-pruned): $L_{logits}$ (R-KLD) 2.665 so với KLD 2.155. R-KLD (reverse, "mode-seeking") được báo là tốt cho SFT/instruction ở các công trình khác, nhưng với retrain mô hình *base* sau prune thì forward KLD tốt nhất. Kết luận: **dùng chỉ forward-KLD, không cộng $L_{CLM}$**.

### 5.3 Tham số logit-KD khác

- Nhiệt độ $\tau\in\{0.1,0.5,1.0,3.0\}$: tốt nhất $\tau=1.0$. Lập luận của tác giả: logit của LLM có entropy cao hơn phân loại ảnh (vốn "nhọn"), nên không cần làm mềm bằng $\tau>1$.
- Top-K (giữ K logit lớn nhất của teacher): K nhỏ (<=100) làm giảm mạnh độ chính xác; K lớn thì không hơn việc không dùng top-K. Bỏ top-K.

### 5.4 Loss trên trạng thái trung gian $L_{is}$ (Appendix A.4, Table 17-18)

Quan sát: output của block ($L_o$) giúp ích; 1-2 layer cuối "chuyên biệt hoá" nên ánh xạ cặp layer (last-2):(last-2) tốt nhất; loss embedding $L_{emb}$ giúp; loss attention-relation ($L_{att}$ từ Q/K/V) không cải thiện; loss trên input MLP không khác; ánh xạ nhiều layer teacher vào 1 layer student không hơn hoặc xấu đi; cosine là tốt nhất cho phần này. Ví dụ Table 17 (Minitron 8B-Depth-pruned): $L_{logits}$ một mình 2.155; $+L_o(29{:}13)$ 2.145; $+L_o(28{:}12)+L_{emb}$ hoặc $L_o(29{:}13)+L_{emb}$: 2.141.

Nhưng với mô hình *không cắt depth nhiều* (Minitron 8B và 4B), Table 18 cho: $L_{logits}+L_{is}$ 58.0/73.6/26.8 (MMLU/HellaSwag/HumanEval, 18.9B token) so với $L_{logits}$ 58.3/73.8/26.2; tăng token lên 94B với $L_{logits}$ cho 62.8/79.7/30.4. Nên:
- **Best practice #6:** dùng logit + trạng thái trung gian + embedding khi *cắt depth mạnh*.
- **Best practice #7:** chỉ dùng logit khi không cắt depth nhiều.

(Chú ý: tác giả nói nghiên cứu "depth-pruned" mạnh dùng $L_{is}$ giúp chút ít; bảng 17 chênh nhỏ, ~0.01-0.015 trong LM loss.)

## 6. Thí nghiệm và kết quả chính

Mô hình nguồn: Nemotron-4 15B (15.6B tham số), đích ~8B và ~4B. Dữ liệu: bộ 8T token pre-training của Nemotron-4 + dữ liệu continued-training (CT); ablation dùng bộ 8T; mô hình cuối dùng cả hai. Retrain nhẹ: 1.8B token (400 bước). Tối ưu hoá theo Nemotron-4, cosine LR từ 2e-4 xuống 4.5e-7. Chạy trên 16 node DGX A100 (8x A100 80GB). Triển khai trong Megatron-LM.

**Kiến trúc (Table 5):**

| Mô hình | Layers | Hidden | Att. heads | Query groups | MLP hidden | Params |
|---|---|---|---|---|---|---|
| Nemotron-4 15B | 32 | 6144 | 48 | 8 | 24576 | 15.6B |
| Nemotron-3 8B | 32 | 4096 | 32 | 32 | 16384 | 8.5B |
| Minitron 8B | 32 | 4096 | 48 | 8 | 16384 | 8.27B |
| Minitron 4B | 32 | 3072 | 24 | 8 | 9216 | 4.19B |

Kết quả cuối Minitron 8B *không cắt layer nào* (vẫn 32 layer): toàn bộ nén đến từ width (hidden, MLP, head), nhất quán với best practice #4.

**Minitron 8B vs các mô hình khác (Table 2, token train: Minitron 94B so với Nemotron-3 8B 3.8T, Llama-3 8B >15T, Llama-2 7B 2T, Mistral 7B 8T, Gemma 7B 6T, Nemotron-4 15B 8T):**

| Benchmark | Llama-3 8B | Llama-2 7B | Mistral 7B | Gemma 7B | Nemotron-4 15B | Nemotron-3 8B | Minitron 8B |
|---|---|---|---|---|---|---|---|
| Winogrande (5) | 77.6 | 74 | 78.5 | 78 | 83.6 | 75.9 | 79.0 |
| ARC-C (25) | 57.8 | 53 | 60.3 | 61 | 58.8 | 52.8 | 52.6 |
| MMLU (5) | 65.3 | 46 | 64.1 | 64 | 66.6 | 54.7 | 63.8 |
| HellaSwag (10) | 82.1 | 79 | 83.2 | 82 | 84.6 | 78.5 | 80.7 |
| GSM8K (5) | 50.3 | 14 | 37 | 50 | 48.5 | 24.0 | 51.3 |
| TruthfulQA (0) | 43.9 | 39 | 42.6 | 45 | 40.7 | 36.5 | 42.6 |
| XLSum en (20%) | 30.9 | 31 | 4.80 | 17 | 32 | 30.9 | 31.2 |
| MBPP (0) | 42.4 | 20 | 38.8 | 39 | 38 | 27.04 | 35.2 |
| HumanEval (0) | 28.1 | 12 | 28.7 | 32 | 35.4 | 20.7 | 31.6 |

**Minitron 4B (Table 3, 4.2B, 94B token):** Winogrande 74.0, ARC-C 50.9, MMLU 58.6, HellaSwag 75.0, GSM8K 24.1, TruthfulQA 42.9, XLSum 29.5, MBPP 28.2, HumanEval 23.3. So với Phi-2 2.7B (MMLU 57.5, GSM8K 55, HumanEval 50) và Gemma 2.5B (MMLU 42): Minitron 4B nhỉnh hơn ở MMLU, kém ở GSM8K/code.

**So với các mô hình đã prune (Table 4):** cỡ 8B: LLM-Pruner/SliceGPT/LaCo/ShortGPT (~9.8B) MMLU lần lượt 25.2/37.1/45.9/54.7 so với Minitron 8B 63.8 (HellaSwag 67.8/55.7/64.4/66.6 so với 80.7). Cỡ 4B: MMLU 23.33/28.92/26.45/43.96 so với 58.6 (Sheared-LLaMA 2.7B: MMLU 26.4). Chú ý các đối thủ lớn hơn về số tham số (non-embedding 9.5B so với 6.2B).

**Instruction-tuned (Minitron 4B-instruct):** MT-Bench 6.46 (Phi-2 4.29, StableLM 2 Chat 5.42, Gemma-2B-IT 5.19...); IFEval prompt-level strict 68.76 / loose 73.01 / instruction-level loose 81.29; ChatRAG-Bench trung bình 41.11 (Gemma-2B-IT 33.31); BFCL v2 trung bình 53.09 (Gemma-2B-IT 41.63, Llama-3-8B-instruct 50.51).

**Tiết kiệm chi phí cho cả họ:** FLOPs mỗi bước cho 15B/8B/4B là 4.4e17 / 2.5e17 / 1.2e17. Nếu 8B và 4B cần 40 lần ít token hơn: tiết kiệm 1.8 lần cho toàn họ Nemotron-4.

## 7. Phân tích / ablation còn lại

- **Hàm gộp importance (Table 13, Nemotron-15B cắt về 8B, không retrain):** (batch=L2, seq=mean) cho LM loss 7.18 (8T) / 7.23 (WikiText2); (mean, mean) 7.21 / 6.89; tệ nhất (var, var) 10.55 / 11.14; (L2, L2) 8.73 / 8.37. Hình 5: sau retrain 400 bước, (L2,mean) vẫn tốt hơn (L2,L2): thứ hạng giữ nguyên. Chọn (L2, mean). **Best practice #2.**
- **Importance lặp (Table 14):** cắt embedding một lần (T=1) hay lặp T=2, T=4: loss zero-shot ban đầu 5.43/5.55/5.24 nhưng sau retrain đều 1.92. Không có lợi ích. **Best practice #3.**
- **Cắt một lần hay lặp theo kích thước (Best practice #8):** 15B → 4B là giảm 73.3% trọng số; cắt 15B → 8B (~46%) rồi 8B → 4B (~50%) cho MMLU cao hơn ~12% so với one-shot (Table 11, hai hàng cuối). Teacher cuối vẫn tốt hơn là 15B chứ không phải 8B.
- **Lặp qua các chiều:** cắt embedding trước rồi MLP+ATT (retrain sau mỗi bước) *không* hơn cắt cả ba một lần (Hình 8).
- **Search với retrain (#9):** thứ hạng 15 ứng viên 8B thay đổi tới ~300 bước rồi ổn.
- **Đa pha (#10, Table 20):** cắt checkpoint pha 2 (đã qua dữ liệu sạch) và retrain chỉ bằng dữ liệu pha 2 (94B token: MMLU 61.9, HumanEval 30.5) tốt hơn cắt pha 1 rồi retrain với cả hai pha (113B: MMLU 54.7, HumanEval 25.6); HellaSwag 80.1 vs 80.3 và PIQA 76.7 vs 77.2 gần như ngang.
- **Cắt từ mô hình gần kích thước đích nhất (#8)** và **luôn chỉ dùng distillation (#5)** như đã nêu.

Danh sách 10 best practices (nguyên văn rút gọn): (1) train mô hình lớn nhất, prune+distill lặp xuống nhỏ hơn; (2) importance (batch=L2, seq=mean) cho width, PPL/BI cho depth; (3) importance one-shot; (4) ưu tiên width hơn depth (≤15B); (5) retrain hoàn toàn bằng KLD; (6) logit+trung gian+embedding khi cắt depth nhiều; (7) chỉ logit khi không cắt nhiều depth; (8) cắt từ mô hình gần kích thước đích; (9) retrain nhẹ để ổn định xếp hạng khi search; (10) nếu có pre-train đa pha, cắt checkpoint pha cuối.

## 8. Giới hạn

*Tác giả nêu hoặc ngầm thừa nhận:*
- Phạm vi kết luận "width > depth" đặt rõ là cho quy mô ≤15B.
- Không thể thử loss attention-score do abstraction của FlashAttention.
- Chưa thử LoRA/PEFT ở bước retrain nhẹ; chưa dùng genetic/Bayesian search.

*Chúng ta nhận xét:*
- Chỉ một họ mô hình, transformer thuần (không hybrid/SSM/linear-attention); không có bài kiểm tra cho VLM hay đầu vào ảnh.
- Các bảng so sánh với đối thủ đã prune có số tham số không bằng nhau, và đối thủ không được retrain với distillation; chưa tách đóng góp của "importance" và "distillation".
- 94B token vẫn là rất lớn so với ngân sách nhỏ; paper không cho đường cong lợi ích theo token ở mức vài tỷ.
- Nhiều kết luận (ví dụ #3, #5) dựa trên một vài cấu hình; không có thống kê nhiều seed.
- Số liệu "Nemotron-3 8B loss" ở hàng cuối Table 1 hơi mơ hồ do định dạng text.

## 9. Liên hệ với các bài khác

- **38 Sheared LLaMA:** cùng triết lý "prune rồi train tiếp", nhưng Sheared học mask (gradient, đắt), chỉ LM loss; Minitron dùng activation importance (forward) và KD. Bảng 4 của Minitron so trực tiếp với Sheared.
- **31 ShortGPT, 33 Shortened LLaMA, 35 LaCo, 32 Deeper layers:** các chỉ số cho depth (BI, PPL, layer collapse) được dùng lại; Minitron cho thấy depth-only *bị thắng* bởi width khi được retrain.
- **34 SLEB, 37 Repair changes which layers:** liên quan đến việc chọn layer bỏ có thay đổi sau sửa chữa; phát hiện "thứ hạng đảo sau retrain" của Minitron nhất quán với ý này.
- **28 FLAP, 29 SliceGPT, 26 LLM-Pruner:** các chỉ số importance cho width; Minitron là ví dụ "chỉ forward".
- **40 Llama-3.1-Minitron, 41 Flextron, 42 Puzzle:** các bài kế tiếp; Puzzle thay việc "chọn tay" bằng tìm kiếm cấu hình từng layer.
- **44-45 Nemotron-H, Minitron-SSM:** mở rộng sang hybrid Mamba (rất liên quan tới kiến trúc dự án).

## 10. Ứng dụng cho dự án (suy luận)

*Tất cả mục này là suy luận của chúng ta, không phải kết luận của paper.*

- **Quan sát của dự án ("cắt neuron MLP hồi phục tệ hơn cắt layer") dường như ngược với paper** (paper thấy width hồi phục tốt hơn depth sau retrain). Có thể giải thích: (a) Minitron retrain 1.8B token trở lên với KD; nếu dự án đánh giá hồi phục với ngân sách nhỏ hơn nhiều thì có thể đang nhìn ở trạng thái "chưa đảo thứ hạng"; (b) paper cắt MLP theo activation importance và cắt đồng thời embedding, trong khi dự án có thể cắt neuron với tiêu chí khác; (c) kiến trúc hybrid GatedDeltaNet khác transformer thuần. Cần kiểm tra thay vì tin một trong hai bên.
- **Phân rã D_lin / D_res:** paper không có phân rã này, nhưng kết quả Table 1 gợi ý một cách đọc: layer bị bỏ có nhiều phần "tuyến tính" (D_lin) có thể sửa bằng retrain ngắn, còn width cắt đi giảm "dung lượng" nhưng retrain dài lấy lại tốt. Chỉ là giả thuyết.
- Quy trình nên bắt chước: (1) calibration 1024 mẫu OCR, tính importance (head/neuron/kênh embedding) bằng forward; (2) liệt kê <20 cấu hình ~0.3B (ví dụ thay đổi số neuron MLP, số kênh hidden, số head, số layer) và retrain nhẹ mỗi cái với KD từ teacher 4B, *rồi mới xếp hạng*; (3) chọn bản tốt nhất train tiếp.
- Loss: bắt đầu với forward KL trên logits, $\tau=1$, không cộng CE nhãn thật; chỉ thêm loss hidden-state/embedding nếu cắt nhiều layer. Với OCR, nhãn đúng là chuỗi ký tự nên có thể thử so sánh KL-only với KL+CE, nhưng paper đã cho thấy KL-only tốt hơn cho LM.
- Cắt lặp: nếu muốn đi 0.8B → 0.3B → 0.1B, #8 gợi ý cắt từng bước (nhưng vẫn dùng teacher gốc).
- Phần ViT 12 block: paper không đề cập; ít liên quan trực tiếp.

## 11. Thí nghiệm gợi ý

1. **Tái hiện đảo thứ hạng:** cắt LightOnOCR-3-0.8B về ~0.3B theo (a) bỏ layer, (b) cắt width (MLP + hidden), (c) hỗn hợp; ghi distillation loss/CER tại bước 0, 100, 200, 400, 800 của KD. Vẽ đường cong để xem có đảo thứ hạng không và ở bước nào.
2. **Loss ablation nhỏ:** cùng một mô hình đã cắt, so KL-only, KL+CE, reverse-KL, và KL + cosine trên hidden cuối (như Table 15/17), cùng số bước, báo CER.
3. **Hàm gộp importance:** so (L2, mean), (mean, mean), (L2, L2) cho neuron MLP của decoder trên 1024 mẫu OCR, đo CER ngay sau cắt và sau 200 bước KD.

## 12. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Activation-based importance | Chuẩn activation theo head/neuron/kênh trên 1024 mẫu, chỉ forward |
| Aggregation (batch, seq) | (L2, mean) tốt nhất sau retrain |
| Block Importance | $1-\mathbb E\cos(X_i,X_{i+1})$ |
| Gộp head | $\text{head}_i+(\text{head}_i-\text{head}_{2K-i+1})$ |
| Lightweight NAS | liệt kê <20 cấu hình, retrain 1.8B token, xếp hạng sau ~300 bước |
| Loss tổng | $L=L_{CLM}+L_{logits}+\alpha L_{is}$, $\alpha=L_{logits}/L_{is}$; thực hành: chỉ $L_{logits}$ (KLD, $\tau=1$) |
| Width > depth | Table 1/10: width 2.049 vs depth 2.155, chỉ sau retrain |
| One-shot > iterative | cho importance, depth, cross-axis |
| Cost | 40x ít token hơn cho mỗi mô hình thêm; 1.8x rẻ hơn cho cả họ |

## 13. Kết luận ngắn

Minitron cho thấy prune (activation-based, forward-only) + retrain bằng KL-distillation trên logits là công thức rẻ và mạnh, với bằng chứng ablation cho từng lựa chọn. Bài học dễ bị bỏ qua: **không đánh giá lựa chọn cắt ngay sau khi cắt**, vì thứ hạng đảo sau vài trăm bước retrain; và trong thiết lập này width thắng depth. Với dự án, nên coi đó là giả thuyết cần kiểm chứng trên kiến trúc hybrid và ngân sách nhỏ.

Link gốc: https://arxiv.org/abs/2407.14679
