# Mamba-Shedder: Post-Transformer Compression for Efficient Selective Structured State Space Models (2025)

**Tác giả:** J. Pablo Muñoz, Jinjie Yuan (đồng tác giả chính), Nilesh Jain (Intel Labs / Intel Corporation).  
**Venue / arXiv:** arXiv:2501.17088 (v1, 28/1/2025). Bản này không nêu tên venue.  
**Nguồn:** https://arxiv.org/abs/2501.17088  
**Chủ đề:** Cắt tỉa có cấu trúc **không cần huấn luyện** cho Mamba-1, Mamba-2 và hybrid (Zamba2, Hymba): bỏ cả block Mamba, mô-đun SSM, MHA, MLP, hoặc nhóm kênh MLP.  
**Vị trí trong lộ trình:** Đọc trước: 34 SLEB, 31 ShortGPT (cắt block theo độ quan trọng), 44-45 (hướng NVIDIA). Bài này thêm: khảo sát **độ nhạy** của từng cấp độ chi tiết trong mô hình SSM và hybrid, và cho thấy Mamba-1 vs Mamba-2 phản ứng ngược nhau. Tiếp theo: 47 Jet-Nemotron.

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Phần "Ứng dụng cho dự án" là suy luận của chúng tôi.

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
10. [Tóm tắt khái niệm](#10-tóm-tắt-khái-niệm)
11. [Kết luận ngắn](#11-kết-luận-ngắn)

---

## 1. Bối cảnh

- **Mamba-1** (S6) và **Mamba-2** (SSD, "state space duality") là các layer chuỗi có chi phí tuyến tính theo độ dài và trạng thái cố định khi sinh. Mamba-2 đổi ma trận $A$ từ đường chéo thành "vô hướng nhân đơn vị" và đưa vào khái niệm *head*.
- **Hybrid:** Zamba2-2.7B (54 layer: 45 khối Mamba-2 đơn + 9 layer lai gồm Mamba-2 và khối Transformer dùng chung); Hymba-1.5B (Mamba và attention chạy *song song* trong cùng một "hybrid head").
- Pruning cho Transformer đã nhiều; cho SSM thì rất ít. Mamba-Shedder hỏi: những phần nào của mô hình SSM **thừa**, tức bỏ đi mà ít ảnh hưởng?
- **Cắt tỉa training-free:** chỉ chấm điểm rồi bỏ, không cập nhật trọng số (khác với bài 44-45 có distillation lớn).

## 2. Ý tưởng chính

Thuật toán tham lam đơn giản: ở mỗi bước, thử bỏ từng cấu trúc ứng viên (khối Mamba, mô-đun SSM, MHA, MLP, hay một nhóm kênh MLP), đo xem **perplexity** trên một tập hiệu chỉnh nhỏ (Alpaca) tăng bao nhiêu, rồi bỏ cấu trúc làm tăng ít nhất. Lặp lại $t$ bước. Có thể cắt tuần tự theo cấp: trước cấu trúc lớn (khối), sau đó cấu trúc nhỏ (SSM, kênh MLP). Cuối cùng có thể fine-tune nhẹ 2 epoch để phục hồi. Phát hiện chính: Mamba-1 chịu bỏ nguyên khối tốt hơn, Mamba-2 chịu bỏ mô-đun SSM tốt hơn; hybrid cần cắt đa cấp.

## 3. Phương pháp chi tiết

**Thuật toán 1 (Block / Module Pruning):**

$$
\boxed{M_{\min}=\arg\min_{M_i\in\mathcal{M}} S_i,\quad S_i=\mathrm{Importance}(M_i,m,C,\phi)}
$$

với $\mathcal{M}$ là tập cấu trúc ứng viên, $C$ tập hiệu chỉnh, $\phi$ metric (ở đây là perplexity). Bỏ $M_{\min}$, lặp $t$ lần.

**Thuật toán 2 (MLP Channel Pruning):** như trên nhưng ứng viên là "$g$ kênh cuối" của từng MLP (kích thước nhóm kênh $g$); bỏ nhóm làm tăng perplexity ít nhất.

```text
Shedder(model m, candidates M, calib C, steps t):
  for k in 1..t:
      for each M_i in M:
          S_i = PPL(m without M_i, C)        # càng thấp càng thừa
      M_min = argmin_i S_i
      m = m without M_min
  return m
# Đa cấp: "&" = các loại ứng viên chấm cùng một bước; "+" = các giai đoạn nối tiếp
# vd: (Mamba block & MLP & MHA) + MLP channel + SSM
```

Các cấu trúc ứng viên: toàn bộ khối Mamba; mô-đun SSM (S6 hoặc SSD) bên trong khối Mamba (giữ lại linear projection và conv); với hybrid thêm khối Transformer, MHA, MLP, nhóm kênh MLP; với Hymba bỏ cả khối hybrid-head. Hiệu chỉnh bằng bộ Alpaca (theo BlockPruner); đo trên lm-eval-harness (Lambada PPL + 6 tác vụ: HellaSwag, PIQA, ARC-e, ARC-c, WinoGrande, OBQA).

## 4. Thí nghiệm và kết quả chính

**Mô hình:** Mamba-2.8B (64 khối S6), Mamba2-2.7B (64 khối SSD), Zamba2-2.7B (54 layer), Falcon-Mamba-7B, Hymba-1.5B-Base. Mỗi khối Mamba bỏ đi giảm 0.04B tham số. Không fine-tune, trừ khi ghi chú.

**Bỏ nguyên khối Mamba (Bảng 1; Avg = trung bình 7 tác vụ):**

| Mô hình | Bỏ khối | Tỉ lệ | Lambada PPL | Avg |
|---|---|---|---|---|
| Mamba-2.8B | 0 | 0% | 4.23 | 59.9 |
| | 7/64 | 10.43% | 4.94 | 57.7 (-2.2) |
| | 14/64 | 20.86% | 7.51 | 53.8 (-6.1) |
| Mamba2-2.7B | 0 | 0% | 4.10 | 60.2 |
| | 7/64 | 10.42% | 8.43 | 56.2 (-4.0) |
| | 14/64 | 20.83% | 11.53 | 52.8 (-7.4) |
| Zamba2-2.7B | 0 | 0% | 4.01 | 67.2 |
| | 7/54 | 10.38% | 6.80 | 60.5 (-6.7) |
| | 14/54 | 20.77% | 15.8 | 52.9 (-14.3) |

**Chỉ bỏ mô-đun SSM (Bảng 2):**

| Mô hình | Bỏ SSM | Lambada PPL | Avg |
|---|---|---|---|
| Mamba-2.8B | 16/64 | 9.23 | 49.8 (-10.1) |
| | 24/64 | 22.55 | 44.1 (-15.8) |
| Mamba2-2.7B | 16/64 | 4.26 | 59.8 (-0.4) |
| | 24/64 | 14.95 | 55.5 (-4.7) |
| Zamba2-2.7B | 16/54 | 4.14 | 66.4 (-0.8) |
| | 24/54 | 5.46 | 64.7 (-2.5) |

**Hybrid Zamba2, đa cấp (Bảng 3, trích, ngân sách ~10% / ~15%):**

| Chiến lược | Tỉ lệ | SSM bổ sung | PPL | Avg |
|---|---|---|---|---|
| Mamba block & Transformer block | 10.40% | 0 | 9.18 | 57.6 (-9.6) |
| Mamba block & MLP & MHA | 10.33% | 0 | 5.01 | 64.5 (-2.7) |
| ... + MLP channel | 10.27% | 0 | 5.45 | 66.3 (-0.9) |
| ... + MLP channel + SSM | 10.27% | 18/54 | 5.18 | 65.9 (-1.3) |
| Mamba block & MLP & MHA + MLP channel | 15.48% | 0 | 7.39 | 62.3 (-4.9) |
| ... + SSM | 15.48% | 18/54 | 7.43 | 61.3 (-5.9) |

**Mô hình khác:** Hymba-1.5B bỏ 7/32 khối: Avg 61.7 (từ 63.8; trung bình 5 tác vụ). Falcon-Mamba-7B: bỏ 10/64 khối PPL 3.15→4.97, Avg 72.0→66.4; bỏ 20 khối PPL vọt lên 39.31, Avg 55.7; bỏ SSM 20/64 PPL 14.14, Avg 55.1.

**Tăng tốc (V100 32GB, batch 1, prompt 512):**

| Mô hình / cấu hình | Prefill | Decode |
|---|---|---|
| Mamba-2.8B, bỏ 7 khối | 1.12x | 1.13x |
| Mamba-2.8B, bỏ 14 khối | 1.31x | 1.29x |
| Mamba2-2.7B, bỏ 24 SSM | 1.20x | 1.18x |
| Zamba2 (15.48%, không SSM) | 1.16x | 1.34x |
| Zamba2 (15.48% + 18 SSM) | 1.25x | 1.39x |
| Hymba-1.5B, bỏ 7 khối | 1.15x | 1.24x |

**Fine-tune phục hồi** (Alpaca đã làm sạch, 2 epoch): Zamba2 ở 10.27%, 18 SSM: PPL 5.18 → 4.58, Avg 65.9 → 67.0 (dense 67.2). Ở 15.48%: PPL 7.43 → 5.88, Avg 61.3 → 64.4. Hymba bỏ 7 khối: Avg 61.7 → 63.7 (dense 63.8). Mamba2-2.7B bỏ 20 SSM: PPL 5.89 → 4.44, Avg 58.6 → 59.6.

## 5. Phân tích / ablation

- **Mamba-1 vs Mamba-2 ngược nhau:** Mamba-1 chịu bỏ khối tốt hơn (PPL 4.23→7.51 ở 20.86%) nhưng rất nhạy khi bỏ SSM (Avg 44.1 ở 24 SSM); Mamba-2 ngược lại (bỏ 24 SSM chỉ giảm 4.7 điểm). Tác giả đoán các thay đổi trong Mamba-2 (giới hạn tính biểu đạt để tăng hiệu quả huấn luyện) làm nó nhạy hơn với bỏ khối nhưng bền hơn với bỏ SSM; đây là giả thuyết, không được kiểm chứng bằng ablation riêng.
- **Hybrid chịu bỏ SSM tốt nhất:** Zamba2 bỏ 12/54 SSM gần như không đổi PPL (4.01 → 4.02). Tác giả cho rằng bản chất hybrid giúp (giả thuyết).
- **Cắt đa cấp tốt hơn đơn cấp:** chỉ bỏ khối Mamba và khối Transformer nguyên vẹn là tệ nhất; tách thành MHA/MLP, rồi cắt thêm kênh MLP, là tốt nhất ở cùng tỉ lệ.
- **SSM có thể thừa thật:** thêm bước bỏ SSM sau khi đã cắt MLP kênh giảm PPL ở mức 10% (5.45 → 5.18), gợi ý một số SSM là dư thừa sau giai đoạn hai.
- **Đánh đổi tốc độ vs độ chính xác:** bỏ khối Mamba cho speedup lớn hơn bỏ SSM, nhưng bỏ SSM bảo vệ perplexity tốt hơn; lựa chọn tuỳ mục tiêu.
- **Fine-tune ngắn phục hồi gần như đầy đủ** ở mức 10% (Zamba2, Hymba).

## 6. Giới hạn

*Paper nêu:* đây là công trình đầu tiên khảo sát bỏ cấu trúc trong Mamba/hybrid; còn nhiều câu hỏi về tỉ lệ SSM/Transformer phù hợp; cần thêm nghiên cứu về đánh đổi hiệu quả-độ chính xác trước khi triển khai ở nơi ảnh hưởng đến con người. Không gian tìm kiếm đa cấp mới tuần tự, chưa chấm các cấp trong cùng một bước (future work).

*Chúng tôi nhận xét:*
- Thuật toán tham lam theo perplexity trên Alpaca: có thể thiên về tác vụ ngôn ngữ chung; chưa kiểm tra trên tác vụ chuyên biệt (OCR/tài liệu).
- Mỗi bước phải chạy lại mô hình cho mọi ứng viên: tốn tính toán khi mô hình lớn (paper không nêu chi phí).
- Chỉ mô hình ≤ 7B, tác vụ đa chọn/zero-shot; không có math/code/long-context.
- Chỉ khi fine-tune trên cùng dữ liệu Alpaca thì phục hồi mới được báo cáo; chưa có distillation.
- Chỉ số tham số và một GPU V100 cho tốc độ; chưa rõ trên GPU nhỏ hiện đại.

## 7. Liên hệ với các bài khác

- **31 ShortGPT, 34 SLEB, 33 Shortened LLaMA:** cùng triết lý bỏ khối bằng điểm quan trọng; Mamba-Shedder dùng perplexity (giống SLEB/BlockPruner).
- **45 Minitron-SSM:** bài đó cắt *bên trong* Mamba (heads/channels) có distillation; bài này bỏ cả mô-đun, training-free. Kết luận chung: Mamba-2 chịu cắt trong-SSM khá tốt.
- **44 Nemotron-H:** không cắt Mamba; bài 45 và bài này cho thấy có hướng khác.
- **29 SliceGPT, 26 LLM-Pruner:** các phương pháp cắt cho Transformer; Mamba-Shedder là phiên bản tương tự cho SSM.
- **47 Jet-Nemotron:** cũng khảo sát loại layer nào cần giữ trong hybrid.

## 8. Ứng dụng cho dự án (suy luận)

- **Liên quan trung bình.** Mamba-Shedder không có GatedDeltaNet, nhưng phương pháp (greedy ablation theo perplexity, chia cấp độ) chuyển được: ứng viên có thể là {layer GatedDeltaNet, layer full-attention, MLP, nhóm neuron MLP}.
- **Bài học đáng dùng:** cắt **đa cấp và tuần tự** (khối trước, kênh sau) tốt hơn chỉ cắt một cấp. Điều này khớp quan sát của bạn rằng cắt layer và cắt neuron khác nhau về khả năng phục hồi; có thể thử "cắt vài layer, rồi mới cắt nhẹ neuron".
- **Cẩn thận:** kết quả "Mamba-1 vs Mamba-2 phản ứng ngược nhau" cảnh báo rằng độ nhạy của *một loại layer tuyến tính* không suy ra được cho loại khác; GatedDeltaNet cần đo riêng.
- **Training-free chỉ là điểm xuất phát:** mất 2.7 đến 14 điểm ở mức 10% khi không phục hồi; trong dự án của bạn sẽ có distillation, nên đây gần giống "bước xếp hạng ứng viên".

## 9. Thí nghiệm gợi ý

1. Chạy greedy ablation (bỏ từng layer, chấm bằng loss/perplexity OCR trên 128 mẫu) để xếp hạng từng layer GatedDeltaNet và full-attention; so với thứ hạng từ MSE/KL (bài 44-45).
2. Thử cắt hai giai đoạn: bỏ 2-3 layer rồi cắt nhóm kênh MLP, so với chỉ bỏ layer hoặc chỉ cắt MLP ở cùng ngân sách tham số; so sánh trước và sau distill ngắn.
3. Đo xem điểm "PPL tăng khi bỏ layer" có dự đoán được điểm sau distill hay không (như bài học "điểm trước distill không chắc dự đoán" ở bài 44).

## 10. Tóm tắt khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| S6 / SSD | Lõi SSM của Mamba-1 / Mamba-2 |
| Bỏ khối Mamba | Mất ~0.04B tham số mỗi khối ở các mô hình 2.7B |
| Bỏ SSM | Giữ lại projection và conv, bỏ phần quét trạng thái |
| Channel group $g$ | Nhóm kênh MLP bị bỏ cùng nhau |
| "&" vs "+" | cùng bước chấm điểm vs các giai đoạn nối tiếp |
| Chấm điểm | Perplexity trên Alpaca khi bỏ cấu trúc |
| Kết quả | tới 1.4x tăng tốc (Zamba2 decode 1.39x) |

## 11. Kết luận ngắn

Mô hình SSM và hybrid có dư thừa có thể bỏ mà ít ảnh hưởng. Mamba-1 chịu bỏ khối, Mamba-2 chịu bỏ mô-đun SSM; hybrid hưởng lợi từ cắt đa cấp (khối, MLP, MHA, kênh MLP, SSM). Cắt training-free đã có lợi (tăng tốc tới ~1.4x), và fine-tune 2 epoch đưa độ chính xác gần với mô hình đầy đủ ở tỉ lệ ~10%.

Link gốc: https://arxiv.org/abs/2501.17088
