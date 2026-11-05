# Minitron-SSM: Efficient Hybrid Language Model Compression through Group-Aware SSM Pruning (2025)

**Tác giả:** Ali Taghibakhshi, Sharath Turuvekere Sreenivas, Saurav Muralidharan, Marcin Chochowski, Yashaswi Karnati (đồng tác giả chính) cùng Raviraj Joshi, Ameya Sunil Mahabaleshwarkar, Zijia Chen, Yoshi Suhara, Oluwatobi Olabiyi, Daniel Korzekwa, Mostofa Patwary, Mohammad Shoeybi, Jan Kautz, Bryan Catanzaro, Ashwath Aithal, Nima Tajbakhsh, Pavlo Molchanov (NVIDIA).  
**Venue / arXiv:** arXiv:2504.11409 (v2, 31/10/2025). Bài không nêu venue.  
**Nguồn:** https://arxiv.org/abs/2504.11409  
**Chủ đề:** Cắt tỉa có cấu trúc cho hybrid Mamba-2 + attention: cắt Mamba heads / head channels theo **nhóm (group-aware)**, kết hợp cắt FFN, embedding, layer; phục hồi bằng distillation. Nén Nemotron-H 8B → 4B.  
**Vị trí trong lộ trình:** Đọc trước: 39 Minitron, 28 FLAP, 44 Nemotron-H. Bài này thêm: giải đúng phần mà MiniPuzzle (44) phải bỏ, tức là cắt *bên trong* khối Mamba mà không phá cấu trúc SSM. Tiếp theo: 46 Mamba-Shedder (cắt cả khối/mô-đun SSM, không huấn luyện), 47 Jet-Nemotron.

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
10. [Tóm tắt công thức / khái niệm](#10-tóm-tắt-công-thức--khái-niệm)
11. [Kết luận ngắn](#11-kết-luận-ngắn)

---

## 1. Bối cảnh

- Minitron (39) cho thấy: cắt tỉa có cấu trúc (neuron, head, layer) + distillation cho Transformer rất hiệu quả. Nhưng với **hybrid** (Mamba + attention) thì "rất ít được khảo sát".
- Khó khăn: trong Transformer, neuron FFN và kênh embedding có tính **hoán vị tương đương (permutation equivariant)**: đổi thứ tự neuron rồi đổi thứ tự tương ứng ở ma trận sau thì kết quả y nguyên, nên ta có thể *sắp xếp theo điểm quan trọng rồi cắt phần đuôi*. Khối Mamba **không** có tính chất này một cách tự do (xem dưới), nên cắt kiểu cũ sẽ làm hỏng phép tính SSM.
- Tiền đề SSM: $h_t=A_t h_{t-1}+B_t x_t$, $y_t=C_t^\top h_t+D_t x_t$ (Mamba có tham số phụ thuộc đầu vào). Mamba-2 chia kênh thành **heads** ($m_h$ head, mỗi head $m_d$ kênh) và chia sẻ $B,C$ theo **groups** ($g$ nhóm, trạng thái $d_s$).

## 2. Ý tưởng chính

Một layer Mamba-2 có $m_h$ heads, nhưng các head được gom thành $g$ nhóm dùng chung $B$ và $C$. Nếu bạn sắp xếp lại head giữa các nhóm khác nhau, phép "broadcast" của $B_t x_t$ bị đổi nghĩa. Vì vậy tác giả sắp xếp và cắt head **chỉ trong từng nhóm** (group-aware), và cắt *cùng một tập kênh* cho mọi head. Đi kèm là công thức chấm điểm theo kích hoạt (không cần gradient) cho head/kênh Mamba, FFN neuron, kênh embedding, cộng với tìm kiếm kiến trúc nhỏ rồi distill. Kết quả: Nemotron-H 8B → 4B giữ hơn 96% độ chính xác, nhanh khoảng 2x so với mô hình cùng cỡ, dùng tới 40x ít token hơn.

## 3. Phương pháp chi tiết

### 3.1 Layer Mamba và ràng buộc nhóm

Từ đầu vào $X$ (sau LayerNorm), năm phép chiếu: $z=W_z\mathrm{LN}(X)$, $x=W_x\mathrm{LN}(X)$ (kích thước $m_h\times m_d$), $B=W_B\mathrm{LN}(X)$, $C=W_C\mathrm{LN}(X)$ (kích thước $g\times d_s$), $dt=W_{dt}\mathrm{LN}(X)$ ($m_h$). $x,B,C$ qua conv1d, rồi $\tilde y=\mathrm{SSM}(\hat x,\hat B,\hat C,A,D,dt)$, rồi $y=W_O(\mathrm{RMSNorm}(\tilde y,z))$.

Phép $B_t x_t$ reshape $B$ thành $g\times d_s$ và **broadcast** lên $x\in\mathbb{R}^{m_h\times m_d}$. Gọi $\mathcal{G}_g$ là tập head thuộc nhóm $g$. Hoán vị hợp lệ phải thoả:

$$
\boxed{\mathcal{P}(h)\in\mathcal{G}_g\quad\forall h\in\mathcal{G}_g}
$$

(chỉ hoán vị trong nhóm). Ví dụ trong paper: hoán đổi head H3 và H8 nằm ở hai nhóm khác nhau thì $B_tx_t$ mới **không** phải một hoán vị của $B_tx_t$ gốc.

**Nhất quán kênh:** kênh $k$ của head phải được giữ hoặc bỏ **đồng loạt** cho mọi head (tập kênh chung).

### 3.2 Chấm điểm và xếp hạng head / kênh Mamba

Điểm kích hoạt (dùng đầu ra của $W_x$): $s=\mathrm{LN}(X)W_x^\top$, tổng hợp theo batch $B$ và độ dài $L$ (mean cho một trục, $\ell_2$ cho trục kia, theo Minitron):

1. Điểm kênh: $s_d=\lVert s_{:,d}\rVert_2$; giữ top-$k_d$ kênh $\mathcal{D}_{top}$.
2. Điểm head (chỉ trên kênh được giữ): $f_h=\lVert s_{h,\mathcal{D}_{top}}\rVert_2$.
3. Trong mỗi nhóm, sắp xếp head theo $f_h$, giữ $k_g$ head tốt nhất mỗi nhóm; ghép lại thành xếp hạng cuối.

```text
GroupAwareMambaRank(s in R^{m_h x m_d}, k_d, {k_g}):
  s_d  = ||s[:,d]||_2 for each channel d
  D    = topk(s_d, k_d)
  f_h  = ||s[h, D]||_2 for each head h
  for g in groups:
      R_g = argsort_desc(f_h for h in G_g)[:k_g]
  return concat(R_g), D
# Cắt: W in {W_x, W_z, W_O, W_A, W_D, W_dt, conv1d} giữ các hàng/cột theo R, D
```

### 3.3 FFN, embedding, layer, FLAP

- **Neuron FFN:** $F^{(i)}_{neuron}=\sum_{B,L}X(W_1^i)^\top$ (như Minitron). **Kênh embedding:** $F^{(i)}_{emb}=\sum_{B,L}\mathrm{LN}(X)_i$, gộp trên mọi layer dùng kênh đó (FFN, Mamba, attention, LayerNorm).
- **Depth:** độ quan trọng layer = KL divergence giữa logit mô hình bỏ layer đó và mô hình đầy đủ, trung bình 256 mẫu.
- **FLAP mở rộng (28):** $S_j=\lVert W_j\rVert_2\cdot\mathrm{Var}(X_j)$, áp lên đầu vào của OutProj của SSM. Dùng để so sánh với phương pháp $\ell_2$ ở trên.
- **Attention không cắt**: chỉ chiếm ~8% số layer.

### 3.4 Phục hồi bằng distillation

Logit-based, forward KL: $L_{logits}=\frac1L\sum_k \mathrm{FKLD}(p_t^k(x,\tau),p_s^k(x,\tau))$ với teacher = 8B gốc. Tham số: batch 768, độ dài 8192, cosine LR (bắt đầu 1.6e-4), warmup 60 bước. Dữ liệu: mẫu từ Phase 3 mixture của Nemotron-H.

### 3.5 Tìm kiếm kiến trúc (ngân sách 4B)

Không gian: số layer 26-52, embedding 3072-4096, FFN 9984-21504, Mamba heads 64-128, head channels 32-64; hơn một trăm ứng viên. Quy trình: (1) loss zero-shot trên 1024 mẫu; (2) top-22 ứng viên distill nhẹ 3.8B token; (3) chọn ứng viên tốt nhất (tie-break bằng throughput/latency) và distill dài ~380B token.

## 4. Thí nghiệm và kết quả chính

**Mô hình:** Nemotron-H 8B (52 layer, embedding 4096, FFN 21504, 128 Mamba heads, 64 head channels) → 4B. Kiểm tra thêm trên Mamba2-1.3B → 780M.

**Cấu hình cuối (ứng viên #1, Bảng 1):** 52 layer, embedding 3072, FFN 12288, 112 Mamba heads, 64 head channels; LM val loss sau KD nhẹ **1.380**; throughput tương đối **1** (8B gốc: 0.74).

Trích Bảng 1 (sau KD nhẹ, ~4B tham số; loss thấp hơn là tốt):

| # | Layers | Emb | FFN | Heads | Channels | LM loss | Rel. throughput |
|---|---|---|---|---|---|---|---|
| 1 | 52 | 3072 | 12288 | 112 | 64 | 1.380 | 1 |
| 2 | 52 | 3072 | 10752 | 128 | 64 | 1.380 | 0.98 |
| 7 | 44 | 3072 | 14592 | 128 | 64 | 1.393 | 1.11 |
| 24 | 26 | 4096 | 21504 | 128 | 64 | 1.533 | 1.31 |
| 25* | 36 | 4096 | 21504 | 128 | 64 | 1.430 | 1.12 |

(* nhiều tham số hơn 4B.) Ứng viên #1 và #2 cùng loss; chọn #1 vì throughput cao hơn nhờ ít Mamba head hơn.

**Base 4B vs mô hình cùng cỡ (Bảng 4, trích):** trung bình Nemotron-H-4B **64.5** (0.38T token) so với Qwen-2.5-3B 60.3 (18T), Zamba-2-2.7B 55.2 (3T), Falcon-3-3B 54.7 (0.1T), Llama-3.2-3B 51.1 (9T); 8B gốc 66.7 (15T). Ví dụ: MMLU 68.1 (4B) vs 72.7 (8B); GSM8K 69.6 vs 77.9; HumanEval 59.8 vs 57.3.

Tóm tắt trong paper: giữ >96% độ chính xác của 8B, throughput ~1.4x so với 8B, ~2.2x throughput và ~1.8x thấp latency so với Phi-4-4B (model đứng thứ hai); an toàn: Garak 67.77 (so với 70.75 của 8B), AEGIS 98.17 (so với 99.83).

**Mamba2-1.3B → 780M** (10.5B token): mô hình nén 780M ngang trung bình với bản 1.3B gốc và hơn bản 780M huấn luyện từ đầu 300B token (Bảng 8; chi tiết một số dòng: ARC-C 34.2 vs 28.6/33.2, HellaSwag 50.4 vs 54.7/59.9; nghĩa là không thắng mọi dòng).

## 5. Phân tích / ablation

- **Width-only thắng depth-only ở nén 50%:** ứng viên width-only #1 (loss 1.380) tốt hơn nhiều depth-only #24 (1.533). Ngay cả mô hình chỉ cắt depth giữ 36 layer (#25, nhiều hơn ~1.4x tham số) vẫn kém ứng viên width-only tệ nhất (#23). Tức *độ sâu quan trọng*.
- **Nhưng depth-only cho tốc độ cao nhất** (#24: 1.31x). Kết hợp depth + width (#7 với 44 layer) vẫn kém #1 về loss.
- **Độ nhạy theo trục:** độ sâu nhạy nhất với độ chính xác, tiếp theo là Mamba heads, còn FFN và embedding ít ảnh hưởng nhất. Ngược lại, cắt thành phần Mamba cho mô hình nhanh hơn cắt FFN/embedding (tương quan tham số Mamba–latency +0.94, –throughput –0.86).
- **Cắt Mamba heads tốt hơn cắt head channels** (LM loss, latency, throughput đều tốt hơn). Bài suy đoán (hypothesis) rằng Mamba chịu được việc cắt head vì có nhiều head hơn (128 so với 32 attention heads).
- **Điểm kích hoạt:** lấy từ đầu ra $W_x$ thường cho zero-shot loss thấp nhất (ví dụ 3.56 so với 4.11 của $W_z$ và 3.79 của $W_O$ ở cấu hình đầu bảng 2).
- **FLAP vs $\ell_2$ (Bảng 3):** trước KD kết quả lẫn lộn; sau KD ngang nhau cho ứng viên #1, FLAP không có lợi thế rõ.
- **Cắt depth thuần túy tụt ở toán và code:** cắt 4-26 layer + 126B token distill, benchmark "kiến thức lõi" hầu như không đổi nhưng math/code tụt nhiều, nhất là dưới 44 layer.
- **Layer importance (Hình 4):** quan trọng nhất ở đầu và cuối; **attention layer đầu tiên là một trong những layer ít quan trọng nhất**, các attention layer khác quan trọng hơn lân cận; có "mẫu răng cưa" MLP so với Mamba.

## 6. Giới hạn

*Paper nêu / ngụ ý:*
- Attention không được cắt (chỉ ~8% layer), nên chưa biết cách xử lý khi attention chiếm tỉ lệ lớn.
- Distillation cho long-context mới là "future work".

*Chúng tôi nhận xét:*
- Số token distill vẫn rất lớn (~380B) so với dự án nhỏ, dù ít hơn các mô hình so sánh.
- Kết quả trên một họ mô hình (Nemotron-H/Mamba2); chưa kiểm tra với linear attention kiểu GatedDeltaNet (khác cấu trúc: không có "groups" dạng B/C chia sẻ theo cùng cách).
- So sánh với mô hình cộng đồng khác nhau cả dữ liệu lẫn token, nên "thắng" không chỉ nhờ pruning.
- Throughput đo tương đối trên phần cứng và cài đặt của tác giả.

## 7. Liên hệ với các bài khác

- **39 Minitron:** cùng khung importance bằng kích hoạt + KD; Minitron-SSM mở rộng sang Mamba.
- **44 Nemotron-H / MiniPuzzle:** MiniPuzzle bỏ trục Mamba vì suy giảm nặng; bài này cho thấy làm đúng ràng buộc nhóm thì cắt Mamba heads dùng được.
- **46 Mamba-Shedder:** cách tiếp cận khác (bỏ cả khối hoặc mô-đun SSM, training-free) và cũng thấy Mamba-2 chịu cắt SSM tốt hơn cắt khối.
- **28 FLAP:** nguồn gốc metric biến thiên x chuẩn cột.
- **31–34, 36 (cắt layer):** kết quả "depth nhạy hơn width" ủng hộ nhận định về độ sâu.
- **17 CoFi, 20 DepGraph:** nhắc lại ý "nhóm phụ thuộc phải cắt cùng nhau"; ràng buộc nhóm B/C ở đây là một dạng đó.

## 8. Ứng dụng cho dự án (suy luận)

- **Liên quan cao về nguyên tắc:** decoder của bạn có GatedDeltaNet. Bài học chung: trước khi cắt "head/channel" của khối tuyến tính, phải xác định các **ràng buộc chia sẻ** (ví dụ key/value head dùng chung, nhóm $\alpha,\beta$ gating, conv1d theo kênh) và chỉ hoán vị/cắt trong ràng buộc đó. Đây là suy luận; GatedDeltaNet không được bàn trong bài.
- **Về quan sát của bạn:** paper cho thấy với hybrid, nén 50% bằng cắt *width* (FFN + Mamba heads) tốt hơn cắt layer về loss, tức là hướng *ngược* với quan sát "cắt neuron MLP phục hồi kém hơn cắt layer" của bạn. Chúng tôi chưa biết nguyên nhân; có thể do thang nhỏ hơn (0.8B), ngân sách distill, hoặc cách cắt MLP của bạn. Đáng kiểm tra riêng, đừng giả định.
- **Trục có lợi tốc độ:** với batch-1 latency trên GPU ≤ 8 GB, paper cho thấy bỏ layer cho speedup lớn nhất (1.31x ở #24) còn cắt FFN/embedding ít tăng tốc; nên đo latency thật, không chỉ số tham số.
- **Attention layer đầu tiên ít quan trọng** trong hybrid này: gợi ý kiểm tra xem full-attention layer nào trong decoder của bạn có thể bỏ (xem thêm 47).

## 9. Thí nghiệm gợi ý

1. Với một khối GatedDeltaNet, liệt kê mọi tensor liên quan tới "head" (q/k/v proj, gate proj, conv, norm, out proj); thử cắt 12.5% head và đo sự khác biệt giữa cắt đồng bộ (đúng ràng buộc) so với cắt sai ràng buộc.
2. Lặp lại bảng "width-only vs depth-only" ở ngân sách 0.3B của bạn với cùng lượng distill (ví dụ 1-2B token), báo cả LM loss và latency batch-1.
3. So sánh điểm kích hoạt (ngõ ra proj x) với FLAP cho head tuyến tính; kỳ vọng ngang nhau như Bảng 3.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý |
|---|---|
| SSM chọn lọc | $h_t=A_th_{t-1}+B_tx_t$, $y_t=C_t^\top h_t+D_tx_t$ |
| Ràng buộc nhóm | $\mathcal{P}(h)\in\mathcal{G}_g$ cho mọi $h\in\mathcal{G}_g$ |
| Điểm kênh / head | $s_d=\lVert s_{:,d}\rVert_2$; $f_h=\lVert s_{h,\mathcal{D}_{top}}\rVert_2$ |
| Neuron FFN | $F^{(i)}=\sum_{B,L}X(W_1^i)^\top$ |
| FLAP cho SSM | $S_j=\lVert W_j\rVert_2\,\mathrm{Var}(X_j)$ |
| Độ quan trọng layer | KL giữa logit bỏ layer và đầy đủ |
| Loss KD | forward KL trên logit |
| Cấu hình 4B | 52 layer, emb 3072, FFN 12288, 112 heads, 64 channels |

## 11. Kết luận ngắn

Với hybrid Mamba, phải cắt Mamba heads **trong từng nhóm** và cắt cùng tập kênh cho mọi head. Phối hợp với cắt FFN + embedding và distill, Nemotron-H 8B → 4B giữ >96% độ chính xác với ~40x ít token hơn so với các mô hình cùng cỡ. Hai bài học tổng quát: (1) độ sâu nhạy nhất, nhưng cắt chiều rộng cho chất lượng tốt hơn ở cùng số tham số; (2) tốc độ thật phụ thuộc trục cắt (Mamba/layer đẩy nhanh hơn FFN/embedding).

Link gốc: https://arxiv.org/abs/2504.11409
