# Structural Pruning of Large Vision Language Models: A Comprehensive Study on Pruning Dynamics, Recovery, and Data Efficiency (2026)

**Tác giả:** Yiran Huang, Lukas Thede, Massimiliano Mancini, Wenjia Xu, Zeynep Akata (TUM, Munich Center for Machine Learning, Helmholtz Munich, Tübingen AI Center, Univ. Trento, BUPT).  
**Venue / arXiv:** arXiv:2604.24380 (v1, 27/4/2026). Bài nói là mở rộng từ một bản hội nghị trước của chính nhóm; không nêu tên venue.  
**Nguồn:** https://arxiv.org/abs/2604.24380  
**Chủ đề:** Nghiên cứu thực nghiệm: cắt tỉa có cấu trúc **LLM backbone** của VLM (layerwise vs widthwise), các cách phục hồi (SFT, distillation logit/hidden state), hiệu quả dữ liệu, và quyết định chọn cấu hình theo tỉ lệ nén.  
**Vị trí trong lộ trình:** Đọc trước: 26 LLM-Pruner (widthwise theo dependency graph), 31/33 (Block Influence, cắt layer), 39 Minitron, 20 DepGraph, 48 ECoFLaP. Bài này thêm: bức tranh thực nghiệm cho **VLM** (gần dự án nhất trong nhóm VLM): cắt layer thắng sau phục hồi ở nén thấp, cắt width thắng ở nén cao, và công thức phục hồi tốt nhất.

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Mọi con số lấy từ paper. Phần "Ứng dụng cho dự án" là suy luận của chúng tôi; các mô hình trong paper (LLaVA, Mini-InternVL, Bunny) là Transformer thuần, không có linear attention.

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

- **LVLM** (Large Vision-Language Model) = vision encoder + bộ chiếu đa phương thức (multimodal projector) + LLM. LLM thường chiếm phần lớn tham số (Bảng 1: LLaVA-v1.5-7B: LLM Vicuna 6.7B của 7.0B; Mini-InternVL-4B: Phi-3-mini 3.8B của 4.1B; Bunny-3B: Phi-2 2.8B của 3.2B), nên là mục tiêu nén chính.
- Cách phổ biến để có VLM nhỏ là ghép với một mô hình ngôn ngữ nhỏ rồi huấn luyện từ đầu (MobileVLM, Bunny...): kích thước bị cố định theo SLM và tốn kém. Hướng thay thế: **nén VLM sẵn có** rồi phục hồi nhẹ.
- Thách thức riêng: cắt LLM có thể làm hỏng không chỉ khả năng ngôn ngữ mà cả **sự căn chỉnh thị giác-ngôn ngữ** đã học. Chưa có nghiên cứu hệ thống.
- Hai kiểu cắt có cấu trúc (gloss): **layerwise** = bỏ nguyên các block transformer; **widthwise** = bỏ attention head và neuron MLP bên trong mỗi layer. Cắt có cấu trúc giảm kích thước ma trận thật, khác cắt không cấu trúc (xem 48).

## 2. Ý tưởng chính

Tác giả áp hai kiểu cắt có sẵn từ LLM (Block Influence cho layer; dependency-graph + Taylor cho width) lên LLM backbone của ba họ VLM, rồi so sánh nhiều cách phục hồi (SFT chỉ projector, SFT projector + LLM với LoRA, KD logit KL/RKL, KD hidden state L2 và các tổ hợp), ở các mức nén 15/30/45/60% và các tỉ lệ dữ liệu 5-100%. Kết luận thực dụng: nén nhẹ (≤15%) chỉ cần SFT projector; nén vừa (≤30%) dùng cắt layer + phục hồi; nén cao (≥45%) dùng cắt width; công thức phục hồi tốt nhất là **SFT + L2 hidden-state distillation**; 5% dữ liệu đủ cho nén ≤ 45% (theo tóm tắt dưới 50%).

## 3. Phương pháp chi tiết

### 3.1 Cắt layer: Block Influence (BI)

Với layer $i$, đầu vào $H_i$ và đầu ra $H_{i+1}$ (hidden state), đo khoảng cách cosine:

$$
\boxed{\mathrm{BI}_i(D)=1-\mathbb{E}_{X\sim D,t}\left[\frac{H_{i,t}^\top H_{i+1,t}}{\lVert H_{i,t}\rVert_2\lVert H_{i+1,t}\rVert_2}\right]}
$$

Layer biến đổi hidden state càng ít (BI thấp) thì càng bỏ được. Xếp hạng rồi bỏ các layer BI thấp nhất.

### 3.2 Cắt width: dependency graph + Taylor

Xây đồ thị phụ thuộc trong mỗi layer: $N_j$ phụ thuộc $N_i$ nếu $N_j\in\mathrm{Out}(N_i)$ và $N_j$ chỉ có một đầu vào (hoặc đối xứng); khi cắt $N_i$ phải cắt luôn các nút phụ thuộc → nhóm $\mathcal{G}$. Độ quan trọng nhóm bằng thay đổi loss khi loại trọng số, xấp xỉ Taylor bậc nhất:

$$
I_{w_i^k}(X)=\left|\mathcal{L}_{CE}(X,m_\theta)-\mathcal{L}_{CE}(X,m_\theta^{w_i^k=0})\right|\approx\left|\frac{\partial\mathcal{L}_{CE}}{\partial w_i^k}w_i^k\right|,\qquad
I_{\mathcal G}(D)=\mathbb{E}_X\Big[\sum_i\sum_k I_{w_i^k}(X)\Big]
$$

Bỏ nhóm có $I_{\mathcal G}$ thấp nhất. Hiệu chỉnh bằng một tập nhỏ mẫu lấy từ chính dữ liệu visual instruction tuning của mô hình.

### 3.3 Phục hồi

Giữ vision encoder đóng băng. Ký hiệu teacher = mô hình gốc, student = mô hình đã cắt.

- **SFT:** $\mathcal{L}_{sft}=\mathbb{E}\,\mathcal{L}_{CE}(m^S(x_v,x_p),x_r)$. Hai chế độ: chỉ huấn luyện projector, hoặc projector + LLM (LoRA).
- **KD logit:** KL hoặc reverse KL (RKL) giữa phân phối teacher và student. KL là "mean-seeking" (phải khớp cả đuôi), RKL "mode-seeking" (chỉ phạt khi student đặt xác suất cao ở chỗ sai).
- **KD hidden state (L2):** $\mathcal{L}_{match}=\mathbb{E}\,\lVert H_i^{S}-H_i^{T}\rVert_2^2$.
- **Tổng hợp:**

$$
\boxed{\mathcal{L}=\alpha\,\mathcal{L}_{sft}+\beta\,\mathcal{L}_{logits}+\gamma\,\mathcal{L}_{match}}
$$

```text
CompressLVLM(model, ratio r, data D):
  calib = small subset of D (<= ~10-15 samples đã đủ, xem Hình 7)
  if r small and want latency: score layers by BI(calib); drop lowest-BI layers
  else:                        build dependency groups; score by Taylor; drop lowest groups
  recover: freeze vision encoder
           if r <= 15%: train projector only (SFT)
           else:        train projector + LLM(LoRA) with SFT + L2(hidden states) vs original teacher
           data: 5%..100% of D (5% đủ khi r <= 45%)
```

## 4. Thí nghiệm và kết quả chính

**Mô hình:** LLaVA-v1.5-7B, Mini-InternVL-Chat-4B-V1.5, Bunny-v1.0-3B. **Benchmark:** MMMU, SQA, MathVista, AI2D, MME-Cognition/Perception, POPE, DocVQA, GQA (qua lmms-eval); báo cáo trung bình và % so với mô hình gốc. **Dữ liệu:** dữ liệu visual instruction tuning gốc của từng mô hình để cả hiệu chỉnh và phục hồi (5%, 10%, 20%, 100%).

**Zero-shot sau cắt, chưa phục hồi (Bảng 2, AVG-% so với gốc):**

| Mô hình | Kiểu | 15% | 30% | 45% | 60% |
|---|---|---|---|---|---|
| LLaVA-7B | Widthwise | 90.87 | 79.88 | 32.43 | 6.75 |
| | Layerwise | 72.22 | 69.18 | 34.40 | 5.34 |
| InternVL-4B | Widthwise | 94.74 | 73.72 | 40.52 | 7.05 |
| | Layerwise | 90.48 | 57.42 | 34.17 | 4.79 |
| Bunny-3B | Widthwise | 90.87 | 75.58 | 47.44 | 14.51 |
| | Layerwise | 88.84 | 69.03 | 45.99 | 11.77 |

Widthwise rõ ràng tốt hơn ở 15-30% khi chưa phục hồi; ở 60% cả hai sụp đổ. Khoảng cách lớn hơn ở tác vụ sinh tự do/lập luận: trên Bunny 15%, widthwise hơn layerwise chỉ 2.2% tính trung bình nhưng 31.5% trên tác vụ sinh (MathVista, DocVQA, GQA); AI2D: 65.64 (width) so với 52.73 (layer) trên Bunny ở 15%.

**Phục hồi tốt nhất = Layerwise (<45%) hoặc Widthwise (≥45%) + SFT + L2 (Bảng 5, AVG-%):**

| Mô hình | 15% (layer) | 30% (layer) | 45% (width) | 60% (width) |
|---|---|---|---|---|
| LLaVA-7B | 98.10 | 96.38 | 83.99 | 78.13 |
| InternVL-4B | 99.12 | 95.96 | 86.22 | 79.39 |
| Bunny-3B | 98.98 | 94.37 | 90.48 | 83.86 |

(Bảng 5 chọn kiểu cắt: layerwise cho tỉ lệ < 45%, widthwise cho ≥ 45%.)

**Phần cứng (Bảng 3; bộ nhớ GiB / FLOPs T / latency ms):** Ví dụ LLaVA 0%: 13.23 / 9.57 / 105. Ở 30%: widthwise 9.32 / 6.89 / 84.4; layerwise 9.44 / 6.92 / 80.7. Ở 60%: widthwise 5.31 / 4.17 / 55.4; layerwise 5.37 / 3.90 / 49.2. Bunny 30%: width 43 ms, layer 41.2 ms. InternVL 30%: width 49.2 ms, layer 48.7 ms. Bộ nhớ và FLOPs gần như như nhau giữa hai kiểu cắt (chênh <0.5% FLOPs ở LLaVA 30%), nhưng **layerwise luôn có latency thấp hơn**.

**Kết hợp lượng tử hoá (Bảng 7, LLaVA, LLM.int8()):** 13.2 GiB (53.7) → 7.5 GiB (52.5) chỉ lượng tử; cắt 15% + int8: 6.5 GiB (51.9); cắt 30% + int8: 5.4 GiB (50.8).

## 5. Phân tích / ablation

- **Giải thích của tác giả (không phải chứng minh):** layerwise gây **lệch phân phối đặc trưng** (layer N+2 nhận đầu ra layer N mà nó chưa quen), widthwise giữ nguyên "giao diện" giữa các layer; ngoài ra mất độ sâu làm mất khả năng lập luận nhiều bước. Sau phục hồi, lệch đặc trưng có thể học lại (nên layerwise thắng ≤30%), còn thiếu độ sâu là hạn chế cấu trúc (nên widthwise thắng ở nén cao).
- **Phục hồi đảo thứ tự:** sau SFT, layerwise vượt widthwise ở nén ≤ 30%; ở ≥ 45% widthwise vượt hoặc ngang (Mini-InternVL, Bunny vượt; LLaVA ngang).
- **Nén nhẹ chủ yếu hỏng sự căn chỉnh, không phải ngôn ngữ (Bảng 4, LLaVA widthwise):** 15% nén: suy giảm đa phương thức 9.13% so với text-only chỉ 1.07%; 30%: 20.11% vs 10.38%; 45%: 67.56% vs 56.92%; 60%: 93.26% vs 93.05%. Từ đó chỉ cần SFT projector khi nén ≤ 15% (giữ hơn 95% hiệu năng ở 15% trên cả ba mô hình; ở 60% phục hồi 60-80%).
- **SFT thêm LLM** cần khi nén > 40%: ở 45% khôi phục trên 80% hiệu năng gốc.
- **KD đơn lẻ không vượt SFT** ở nén cao; thứ tự RKL > KL > L2 khi đứng riêng, nhưng **SFT + L2 vượt mọi tổ hợp** (kể cả SFT+L2+KL/RKL, là các tổ hợp bộ ba). Tác giả giải thích bằng tính trực giao của hai mục tiêu (SFT tối ưu đầu ra, L2 ràng buộc biểu diễn trong) và xung đột khi thêm ràng buộc đầu ra.
- **Hiệu quả dữ liệu (Hình 6):** với nén ≤ 45%, 5% dữ liệu đã phục hồi >90% hiệu năng đạt được với dữ liệu đầy đủ; ở 60% cần nhiều dữ liệu hơn. (Tóm tắt Mục 4.6 nói đủ khi dưới 50%.)
- **Kích thước tập hiệu chỉnh (Hình 7):** hiệu năng tăng tới khoảng 10-15 mẫu rồi bão hoà (widthwise 15%: 43.60 với 1 mẫu → 48.80 với 15 mẫu và giữ nguyên).
- **"Vùng an toàn":** ≤ 30% giữ phần lớn; 45% là điểm gãy (LLaVA 83.99%, InternVL 86.22%, Bunny 90.48%); muốn >90% nên nén dưới 30%.
- **Failure case định tính (Bảng 6):** khi nén tăng, đếm đối tượng và nhận dạng chi tiết hỏng trước; ở 60% mô hình "hallucinate" theo cấu trúc tổng thể.

## 6. Giới hạn

*Tác giả nêu:* chỉ cắt có cấu trúc, không xét unstructured/semi-structured; vision encoder đóng băng (chưa cắt ViT); sau ngưỡng nén nhất định, hiệu năng "khó phục hồi" và "extreme pruning of a large model is not a viable approach"; chỉ hai kỹ thuật cắt, ba mô hình.

*Chúng tôi nhận xét:*
- Chỉ Transformer thuần (Vicuna, Phi-3-mini, Phi-2); không có kiến trúc hybrid/linear attention, và không có OCR chuyên dụng ngoài DocVQA.
- Importance của width dùng gradient Taylor; layer dùng BI; không so sánh các thước đo khác (ví dụ MSE/KL, bài 44-45).
- Phục hồi dùng LoRA với dữ liệu gốc; teacher cùng họ (chính mô hình chưa cắt), không phải teacher lớn hơn như trong dự án.
- Bảng 5 chọn thủ công kiểu cắt theo tỉ lệ nén, nên so sánh "cắt layer vs width" sau phục hồi cần xem Hình 3 (đồ thị) chứ không chỉ bảng.
- Latency đo cụ thể trên một cấu hình; chênh lệch layerwise vs widthwise nhỏ (vài ms) và độ lệch chuẩn đôi khi lớn (ví dụ LLaVA 15% layerwise 95 ± 8.1).

## 7. Liên hệ với các bài khác

- **26 LLM-Pruner, 20 DepGraph:** nguồn của cắt width bằng dependency graph + Taylor.
- **31 ShortGPT, 33 Shortened LLaMA:** nguồn của BI/cắt layer; bài này xác nhận layer cắt gây hại cho lập luận nhiều bước.
- **39 Minitron, 40:** cùng kết luận "width thắng depth, nhất là lập luận (GSM8K)" (bài này trích Sreenivas et al. để so). Bài 45 cũng thấy width-only tốt hơn depth-only ở nén 50% (loss), nhưng depth-only nhanh nhất.
- **37 Repair changes which layers:** bài này cho thấy lựa chọn "layer vs width" thay đổi theo mức phục hồi; tinh thần tương tự (xếp hạng trước phục hồi chưa chắc đúng sau phục hồi).
- **43 Pruning vs scratch:** phục hồi dữ liệu thấp ủng hộ nén từ mô hình có sẵn.
- **48 ECoFLaP:** cùng chủ đề VLM, nhưng thưa không cấu trúc.
- **47 Jet-Nemotron:** giảm số full-attention layer là một hướng giảm độ sâu "có thay thế".

## 8. Ứng dụng cho dự án (suy luận)

- **Liên quan cao nhất trong nhóm VLM.** Cấu trúc ba phần (ViT, projector, LLM), việc chỉ cắt LLM backbone, và nén theo mức 15/30/45/60% giống bài toán 0.8B → 0.3B/0.1B của bạn (tỉ lệ nén thực của bạn là khoảng 62% và 88%, tức nằm ngoài vùng an toàn của paper).
- **Phù hợp một phần với quan sát của bạn:** paper thấy sau phục hồi, layerwise tốt hơn widthwise ở nén thấp (≤30%) — cùng chiều với việc "cắt neuron MLP phục hồi kém hơn cắt layer" — nhưng đảo chiều ở nén cao (≥45%) với thang 3-7B. Với nén tới 0.3B/0.1B, kết quả của bạn có thể thuộc vùng bài này gọi là "khó phục hồi". Không nên suy rộng từ Transformer thuần sang decoder GatedDeltaNet.
- **Một tín hiệu cho phân tích $D_{lin}$/$D_{res}$:** cắt layer gây lệch đặc trưng *học lại được* bằng huấn luyện (gợi ý thành phần gần-tuyến-tính), còn mất độ sâu là cấu trúc. Đây là cách diễn giải của chúng tôi, không phải kết luận của tác giả.
- **Công thức phục hồi dùng được ngay:** SFT + hidden-state L2 distillation (teacher lớn hơn 4B nên cần chiếu chiều nếu khác hidden size); freeze vision encoder khi nén nhẹ, chỉ train projector trước.
- **Hiệu chỉnh:** 10-15 mẫu đã đủ cho điểm quan trọng, và 5% dữ liệu đủ phục hồi nén vừa: tiết kiệm.
- **Latency:** layerwise nhanh hơn widthwise ở cùng tham số: quan trọng với batch-1 latency của bạn.

## 9. Thí nghiệm gợi ý

1. Tái hiện bảng "zero-shot + SFT proj + SFT proj+LLM" ở 15%, 30% cho LightOnOCR (cắt layer vs cắt width của decoder), để xem đảo chiều có xuất hiện ở thang 0.8B không.
2. Thử công thức SFT + L2 hidden-state distill từ teacher 4B (qua lớp chiếu) so với SFT + KL logit, ở cùng ngân sách token; thêm biến thể chỉ train projector trước.
3. Vẽ đường cong "% dữ liệu (5/10/20/100) × tỉ lệ nén" cho tác vụ OCR của bạn để xác định điểm mà 5% dữ liệu không còn đủ.

## 10. Tóm tắt khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| LVLM | Vision encoder + projector + LLM |
| Layerwise (BI) | Bỏ layer có BI thấp: $1-\cos(H_i,H_{i+1})$ |
| Widthwise | Dependency graph + Taylor, bỏ head/neuron |
| SFT projector | Chỉ huấn luyện projector, đủ khi nén ≤ 15% |
| SFT+L2 | Cross-entropy + khớp hidden state; công thức phục hồi tốt nhất |
| RKL vs KL | Reverse KL (mode-seeking) vs KL (mean-seeking) |
| Vùng an toàn | Nén ≤ 30% giữ phần lớn hiệu năng |
| Điểm gãy | ~45% (83.99-90.48% còn lại sau phục hồi) |
| Dữ liệu | 5% đủ khi nén ≤ 45%; >50% cần dữ liệu đầy đủ |

## 11. Kết luận ngắn

Để nén VLM có sẵn: dưới ~30% nén thì cắt layer + phục hồi (chỉ projector nếu ≤15%) giữ 94-99% hiệu năng và cho latency tốt nhất; từ 45% trở lên thì cắt width và phục hồi bằng SFT + hidden-state L2 distillation; 5% dữ liệu thường đủ nếu nén dưới ~50%. Chưa có bằng chứng cho các kiến trúc hybrid/linear hoặc nén rất sâu, nên cần kiểm chứng riêng cho dự án.

Link gốc: https://arxiv.org/abs/2604.24380
