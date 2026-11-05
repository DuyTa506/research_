# ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models (2024)

**Tác giả:** Yi-Lin Sung, Jaehong Yoon, Mohit Bansal (UNC Chapel Hill).  
**Venue / arXiv:** arXiv:2310.02998 (v2, 26/1/2024); ICLR 2024 (theo đầu trang paper "Published as a conference paper at ICLR 2024").  
**Nguồn:** https://arxiv.org/abs/2310.02998  
**Chủ đề:** Cắt tỉa **không cấu trúc (unstructured)** một lần cho mô hình vision-language (BLIP-2, BLIP, CLIP...): xác định tỉ lệ thưa **cho từng layer** từ độ quan trọng toàn cục (xấp xỉ bậc 0 của gradient), rồi cắt theo từng layer bằng Wanda/SparseGPT.  
**Vị trí trong lộ trình:** Đọc trước: 23 SparseGPT, 24 Wanda (hai bộ cắt "fine"), 07 Molchanov, 01 OBD/OBS (độ quan trọng $w^2/2[H^{-1}]_{ii}$ và gradient). Bài này thêm: (a) vấn đề riêng của VLM là phân phối trọng số/gradient chênh lệch giữa module thị giác và ngôn ngữ; (b) phân bổ thưa không đều giữa các layer; (c) tính gradient bằng chỉ forward. Tiếp theo: 49 (cắt có cấu trúc cho VLM, gần với dự án hơn).

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Bài này là pruning **không cấu trúc**: không giảm kích thước ma trận, nên không tự nhiên làm nhanh trên GPU thường. Phần "Ứng dụng cho dự án" là suy luận của chúng tôi.

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
10. [Tóm tắt công thức](#10-tóm-tắt-công-thức)
11. [Kết luận ngắn](#11-kết-luận-ngắn)

---

## 1. Bối cảnh

- **Cắt toàn cục (global pruning):** xếp hạng mọi trọng số của cả mô hình theo một điểm chung rồi bỏ phần thấp nhất. Chính xác (điểm kiểu OBS: $w_i^2/(2[H^{-1}]_{ii})$) nhưng cần Hessian nghịch đảo của cả mô hình, bất khả thi với mô hình lớn.
- **Cắt theo từng layer (layer-wise, one-shot):** SparseGPT (23), Wanda (24) dùng ~128 mẫu hiệu chỉnh, mỗi layer cắt cùng tỉ lệ $p\%$ độc lập. Rẻ, nhưng **mọi layer cùng một tỉ lệ thưa** và không có cái nhìn toàn cục, nên dưới thưa cao thì kém.
- **Vấn đề của VLM:** mô hình gồm các module theo từng modality (ViT thị giác, Q-Former, LLM). Hình 1 của paper cho thấy độ lớn trọng số và gradient của thị giác so với ngôn ngữ chênh nhau mạnh, và điểm cục bộ của SparseGPT phân bố rất lệch giữa layer. Hậu quả: điểm của các layer **không so sánh được** với nhau, nên không thể dùng chúng để chia tỉ lệ thưa; và cắt toàn cục bằng độ lớn (global magnitude) đồng đều thì phá module thị giác (đạt 0.0 ở 50% thưa).

## 2. Ý tưởng chính

Chia hai pha thô-đến-mịn. **Thô (coarse):** tính một điểm quan trọng *toàn cục* cho mỗi layer (hoặc block), rồi chuyển thành tỉ lệ thưa riêng của layer đó (layer quan trọng thì giữ nhiều hơn). **Mịn (fine):** trong từng layer, dùng Wanda (hoặc SparseGPT) cắt theo tỉ lệ thưa vừa tính. Để tính điểm toàn cục mà không tốn bộ nhớ cho backprop, tác giả dùng **gradient bậc 0** (ước lượng bằng hai lần forward với nhiễu Gaussian), nên bộ nhớ gần bằng Wanda.

## 3. Phương pháp chi tiết

### 3.1 Từ điểm toàn cục sang tỉ lệ thưa từng layer

Điểm quan trọng toàn cục $s=S(W\mid D,\mathcal{L})$. Cho thưa tổng $p$, số trọng số cần giữ $N_{select}=(1-p)|W|$. Với layer $i$:

$$
\mathrm{normalize}(s_i,s)=\frac{s_i}{\sum s},\qquad
\boxed{p_i=1-\frac{\mathrm{normalize}(s_i,s)\cdot N_{select}}{|W_i|}}
$$

Nghĩa: số trọng số được giữ ở layer $i$ tỉ lệ với tổng điểm của layer đó. Thêm siêu tham số $p_{max}$ (thưa tối đa mỗi layer) để tránh **layer collapse** (một layer bị cắt hết). Thực nghiệm đặt $p_{max}=p+0.1$. Ngoài ra dùng cùng một tỉ lệ cho các layer trong cùng block (ví dụ FFN và attention của một block transformer) bằng cách cộng điểm theo block, cho kết quả bền hơn.

### 3.2 Gradient bậc 0 (chỉ forward)

Điểm bậc nhất cổ điển: $|W|\cdot|\nabla_W\mathcal{L}|$ (LeCun 1989). Backprop cho LLaMA 7B cần hơn 40 GB. Thay bằng:

$$
\boxed{\lVert\nabla_{W_i}\mathcal{L}\rVert\approx\mathbb{E}_{d\sim D}\,\mathbb{E}_{z\sim\mathcal{N}(0,1)}\left[\left|\frac{\mathcal{L}(W_i+\epsilon z,d)-\mathcal{L}(W_i-\epsilon z,d)}{2\epsilon}\right|\right]}
$$

Dùng giá trị này **trực tiếp** làm điểm của $W_i$ (không nhân độ lớn trọng số, vì "solely using the approximated gradients works better empirically"). Quy trình theo từng layer: chọn seed, lấy nhiễu $z$, đổi $W\to W+\epsilon z$ và tính loss; dùng lại seed để trừ $2\epsilon z$ ($W-\epsilon z$) và tính loss; lấy hiệu chia $2\epsilon$ (các đại lượng đều là số vô hướng). Với mạng $N$ layer chỉ cần $(N+1)$ lần bộ nhớ của một layer, so với $2N$ nếu lưu toàn bộ gradient. Một nhiễu mỗi mẫu là đủ.

### 3.3 Pseudo-code

```text
ECoFLaP(model, calib D (32 mẫu), sparsity p, p_max=p+0.1):
  # Coarse
  for each layer (or block) i:
      s_i = mean over d in D of | L(W_i + eps z, d) - L(W_i - eps z, d) | / (2 eps)
      (restore W_i)
  N_select = (1-p) * total_params
  p_i = 1 - normalize(s_i) * N_select / |W_i|     # clip sao cho p_i <= p_max
  # Fine
  for layer i = 1..L:
      prune W_i to sparsity p_i using Wanda (|W| * ||input act||) or SparseGPT
```

Mô hình mục tiêu: BLIP/BLIP-2 gồm EVA-ViT + Q-Former + FlanT5 (hoặc BERT). Q-Former được giữ nguyên vì chỉ ~5% tham số.

## 4. Thí nghiệm và kết quả chính

**Mô hình/dữ liệu:** BLIP-2 (EVA-ViT + FlanT5), BLIP (ViT + BERT), CLIP, FlanT5, LLaMA-7B, EVA-ViT. Tác vụ: VQAv2, OK-VQA, GQA, NoCaps, Flickr30k (zero-shot cho BLIP-2); NLVR2, COCO captions (BLIP fine-tune); ImageNet-1k; MMLU; WikiText. Hiệu chỉnh 128 mẫu (CC3M, ImageNet, C4); ablation dùng 32 mẫu.

**BLIP-2 ở thưa 50%, zero-shot (Bảng 1):**

| Phương pháp | Bộ nhớ (GB) | VQAv2 | OK-VQA | GQA | NoCaps CIDEr | Flickr TR@1 | Macro avg |
|---|---|---|---|---|---|---|---|
| Full model | - | 63.1 | 41.1 | 44.1 | 121.7 | 97.6 | 62.1 |
| Global magnitude | 8.46 | 0.0 | 0.0 | 0.0 | 0.0 | 0.2 | 0.0 |
| Gradient-based (lặp, global) | 22.4 | 56.9 | 35.3 | 41.7 | 96.8 | 93.0 | 55.4 |
| SparseGPT | 9.04 | 57.4 | 36.1 | 40.8 | 108.1 | 96.4 | 57.4 |
| Wanda | 8.87 | 57.9 | 35.3 | 42.2 | 106.9 | 95.1 | 57.2 |
| ECoFLaP (grad. bậc 1) | 22.4 | 59.4 | 36.3 | 42.3 | 109.9 | 95.9 | 58.2 |
| ECoFLaP (grad. bậc 0) | 8.93 | 58.4 | 35.6 | 42.9 | 110.2 | 95.5 | 58.0 |

So với Wanda: +1.1% VQA, +2.3% captioning, +1.0% retrieval (tương đối); ở thưa 0.6, hơn Wanda 9.6% trên VQA. Dùng ~40% bộ nhớ của cắt toàn cục lặp.

**CLIP ở thưa 0.4 (Bảng 2, trung bình 11 tác vụ):** đầy đủ 65.3; Wanda 47.2; SparseGPT 58.7; ECoFLaP+Wanda 56.0; ECoFLaP+SparseGPT 61.0. Dùng *điểm cục bộ* để chia tỉ lệ thưa còn **tệ hơn** thưa đều (Wanda 45.3, SparseGPT 54.0).

**So UPop (BLIP, thưa 50%, Bảng 3):**

| | NLVR2 val/test | COCO CIDEr/SPICE |
|---|---|---|
| Full | 82.3 / 83.6 | 133.3 / 23.8 |
| Không fine-tune: UPop | 76.9 / 77.8 | - |
| Không fine-tune: Wanda | 78.3 / 78.1 | 97.1 / 18.4 |
| Không fine-tune: ECoFLaP | 79.1 / 79.2 | 111.0 / 20.3 |
| Có fine-tune: UPop | 80.3 / 81.1 | 128.9 / 23.3 |
| Có fine-tune: ECoFLaP | 81.8 / 82.5 | 132.3 / 23.8 |

**LLaMA-7B, thưa 0.6, WikiText perplexity (Bảng 7):** đầy đủ 7.26; Wanda 10.68; ECoFLaP bậc 1 10.16; ECoFLaP bậc 0 **9.83** (cải thiện tương đối 10.6% so với Wanda).

## 5. Phân tích / ablation

- **Tỉ lệ thưa tối đa $p_{max}$ (Bảng 4, VQA, thưa 0.5):** 0.5 → 57.9; 0.6 → 58.4; 0.8 → 57.1; 1.0 (không giới hạn) → 51.6. Không giới hạn gây layer collapse.
- **Số mẫu (Bảng 5):** 16/32/64/128 mẫu cho 58.2/58.4/58.3/58.4: rất ít nhạy. **Số nhiễu (Bảng 6)** với tổng lần forward cố định: 1/4/8/32 nhiễu cho 58.4/58.1/58.0/58.1, tức nên dùng nhiều mẫu, một nhiễu mỗi mẫu.
- **Hình 6:** các biến thể ECoFLaP đều gán thưa **thấp hơn cho module thị giác, cao hơn cho module ngôn ngữ**; phân bố của bậc 0 gần với bậc 1, ủng hộ việc dùng xấp xỉ rẻ.
- **Landscape (Hình 5):** BLIP-2 nằm trong một "bồn" mượt, một cực tiểu địa phương; tác giả dùng đó để biện minh cho ước lượng bậc 0.
- **Phụ lục:** $\epsilon$ lớn hơn giúp ước lượng bậc 0 ở LLaMA (giả thuyết: khám phá vùng xa hơn).
- **Tổng quát hoá:** hiệu quả trên FlanT5/MMLU, EVA-ViT/ImageNet, đặc biệt ở thưa cao (Hình 4).

## 6. Giới hạn

*Paper nêu / ngụ ý:* tập trung vào **không cấu trúc**; tác giả nói trực tiếp rằng lợi ích tốc độ cần phần cứng/kernel thưa chuyên dụng (trong phần Related Work). Chỉ giữ Q-Former nguyên vẹn.

*Chúng tôi nhận xét:*
- Không đo latency hay tốc độ; toàn bộ kết quả là thưa trọng số, không giảm kích thước tensor.
- Cắt 50-60% trọng số không cấu trúc ≠ giảm 50% FLOPs/VRAM trên GPU thường (VRAM vẫn lưu ma trận dày nếu không nén thưa).
- Chưa có phục hồi bằng distillation; chỉ fine-tune ở BLIP (NLVR2/COCO).
- Mô hình thuộc đời BLIP-2/FlanT5, không phải VLM decoder-only hiện đại; không có OCR/tài liệu.
- Độ tin cậy của bậc 0 chỉ được biện minh bằng landscape mượt trên BLIP-2.

## 7. Liên hệ với các bài khác

- **23 SparseGPT, 24 Wanda:** bước "fine" của ECoFLaP; nhược điểm cố hữu là thưa đều.
- **01 OBD/OBS, 07 Molchanov:** điểm quan trọng bậc nhất/bậc hai; ECoFLaP thay backprop bằng ước lượng bậc 0.
- **29 SliceGPT, 26 LLM-Pruner, 28 FLAP:** cắt có cấu trúc; ECoFLaP không đạt tăng tốc trực tiếp như thế.
- **18 2:4 sparsity:** con đường để thưa hoá không cấu trúc mới tăng tốc phần cứng.
- **49 Structural pruning of VLMs:** cùng chủ đề VLM nhưng cắt có cấu trúc trên LLM backbone và đo latency, gần dự án hơn.
- **Phân bổ thưa không đều giữa layer** gần với tinh thần "layer nào đáng giữ" của 42 Puzzle, 47 Jet-Nemotron.

## 8. Ứng dụng cho dự án (suy luận)

- **Ít liên quan trực tiếp:** dự án nhắm tới cắt **có cấu trúc** (giảm tham số vật lý để chạy trong ≤ 8 GB, batch-1). Thưa không cấu trúc không đem lại giảm VRAM/latency trên GPU thông thường, nên ECoFLaP không phải công cụ chính.
- **Ý có thể mượn:** (1) *đừng áp một tỉ lệ nén đều cho mọi module*: ViT 12 block và decoder nên có ngân sách riêng; paper cho thấy tối ưu gán ít nén cho thị giác hơn. Cần kiểm chứng trên LightOnOCR (vision encoder nhỏ nên có thể khác). (2) *độ quan trọng cục bộ không so sánh được giữa module*; muốn phân bổ ngân sách giữa ViT và decoder phải dùng thước đo chung (loss cuối). (3) Ước lượng độ nhạy bằng chỉ forward (hai lần forward với nhiễu) là cách rẻ nếu bộ nhớ hạn chế, nhưng ở đây chỉ được kiểm chứng cho thưa không cấu trúc.
- **Mối liên hệ với $D_{lin}$/$D_{res}$:** bài không phân tích loại sai số này; ECoFLaP dùng Wanda ở bước mịn, vốn không có bù tuyến tính như SparseGPT.

## 9. Thí nghiệm gợi ý

1. Đo "độ nhạy theo khối" của LightOnOCR bằng hai lần forward với nhiễu nhỏ trên từng block (ViT và decoder) hoặc đơn giản hơn bằng cách bỏ/cắt riêng từng block; so sánh độ nhạy ViT và decoder để quyết định chia ngân sách nén.
2. Thử cùng một tổng ngân sách nén nhưng phân bổ khác nhau giữa ViT và decoder (ví dụ 0/100, 25/75, 50/50), distill ngắn, xem điểm OCR.
3. Kiểm tra xem ước lượng gradient bậc 0 có xếp hạng block giống gradient thật không (tương quan hạng) trên một mô hình nhỏ.

## 10. Tóm tắt công thức

| Khái niệm | Công thức / ý |
|---|---|
| Tỉ lệ thưa mỗi layer | $p_i=1-\dfrac{(s_i/\sum s)\,N_{select}}{|W_i|}$, $N_{select}=(1-p)|W|$ |
| Ràng buộc | $p_i\le p_{max}=p+0.1$ |
| Gradient bậc 0 | $\mathbb{E}\big|\frac{\mathcal{L}(W+\epsilon z)-\mathcal{L}(W-\epsilon z)}{2\epsilon}\big|$ |
| Bộ nhớ | $(N+1)$ lần bộ nhớ một layer, so với $2N$ |
| Bước mịn | Wanda hoặc SparseGPT với $p_i$ |
| Phát hiện | Thị giác ít thưa hơn, ngôn ngữ thưa hơn |

## 11. Kết luận ngắn

ECoFLaP cho thấy trong VLM không nên cắt đều mọi layer: tính độ quan trọng toàn cục bằng gradient bậc 0 (rẻ như forward) để chia tỉ lệ thưa từng layer, rồi cắt bằng Wanda/SparseGPT, cải thiện rõ nhất ở thưa cao (ví dụ +9.6% VQA tương đối so với Wanda ở thưa 0.6) và ngang bộ nhớ Wanda. Với dự án, đây chủ yếu là bài học về phân bổ ngân sách giữa các module; kỹ thuật cụ thể (thưa không cấu trúc) ít liên quan.

Link gốc: https://arxiv.org/abs/2310.02998
