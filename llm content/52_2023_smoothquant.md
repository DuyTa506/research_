# SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models (2023)

**Tác giả:** Guangxuan Xiao*, Ji Lin*, Mickael Seznec, Hao Wu, Julien Demouth, Song Han (MIT; NVIDIA; * đóng góp ngang nhau)  
**Venue / arXiv:** arXiv:2211.10438 (v7, 29/3/2024); ICML 2023 (PMLR 202)  
**Nguồn:** https://arxiv.org/abs/2211.10438  
**Chủ đề:** Lượng tử hoá sau huấn luyện W8A8 (trọng số 8-bit, activation 8-bit) cho LLM bằng phép co giãn theo kênh tương đương toán học, "chuyển độ khó lượng tử hoá từ activation sang trọng số".  
**Vị trí trong lộ trình:** Đọc ngay sau 51 (LLM.int8(): chẩn đoán outlier và tách FP16) và cần kiến thức nền lượng tử hoá ở mục 1 của bài 51. Bài này giải cùng vấn đề nhưng cho kernel INT8 thuần, hardware-friendly. Tiếp theo: 53 (GPTQ, chỉ trọng số, 3–4 bit), 54 (AWQ, cũng dùng co giãn theo kênh nhưng cho weight-only).

> Đây là ghi chú học tập bằng tiếng Việt, không phải bản dịch. Ứng dụng cho dự án và các số "tự tính" là suy luận của chúng ta, không phải kết luận của paper.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
4. [Thí nghiệm và kết quả chính](#4-thí-nghiệm-và-kết-quả-chính)
5. [Phân tích / ablation](#5-phân-tích--ablation)
6. [Phân tích bộ nhớ và độ trễ (batch 1)](#6-phân-tích-bộ-nhớ-và-độ-trễ-batch-1)
7. [Giới hạn](#7-giới-hạn)
8. [Liên hệ với các bài khác](#8-liên-hệ-với-các-bài-khác)
9. [Ứng dụng cho dự án (suy luận)](#9-ứng-dụng-cho-dự-án-suy-luận)
10. [Thí nghiệm gợi ý](#10-thí-nghiệm-gợi-ý)
11. [Tóm tắt công thức / khái niệm](#11-tóm-tắt-công-thức--khái-niệm)
12. [Kết luận ngắn](#12-kết-luận-ngắn)

---

## 1. Bối cảnh

Để **tăng tốc** (không chỉ giảm bộ nhớ) phép nhân ma trận bằng INT8 GEMM (nhân ma trận số nguyên), cần lượng tử hoá **cả hai** vế: trọng số *và* activation, gọi là **W8A8**. Weight-only (như bài 53) giảm bytes đọc nhưng phép nhân vẫn phải khử lượng tử về FP16.

Vấn đề: ở LLM ≥ 6.7B, activation có outlier hệ thống (bài 51). ZeroQuant (per-token động + group-wise cho trọng số) vẫn hỏng với OPT-175B; LLM.int8() sửa được bằng tách FP16 nhưng khó triển khai hiệu quả trên phần cứng và nhiều khi chậm hơn FP16. Mục tiêu của paper: một cách **không cần huấn luyện lại, chính xác, và dùng được kernel INT8 chuẩn**, tốt nhất là per-tensor, static.

Gloss thuật ngữ:
- **PTQ (post-training quantization):** lượng tử hoá sau huấn luyện, chỉ dùng một ít mẫu hiệu chuẩn (calibration).
- **Static vs dynamic quantization:** static = tính scale của activation *offline* từ mẫu calibration (nhanh, nhưng lệch nếu dữ liệu thật khác); dynamic = tính scale *lúc chạy* theo từng tensor/token (chính xác hơn, thêm chi phí).
- **Per-tensor / per-token / per-channel:** một scale cho cả tensor / mỗi token (hàng của $X$) / mỗi kênh đầu ra (cột của $W$). Xem Hình 3 của paper.
- **GEMM** = General Matrix Multiply; **BMM** = Batch Matrix Multiply (QKᵀ, PV trong attention).

## 2. Ý tưởng chính

Ba quan sát (Mục 3 của paper): (1) **Trọng số dễ lượng tử hoá, activation khó**: phân phối trọng số phẳng, đều, còn activation có outlier lớn hơn phần lớn giá trị ~100×. (2) **Outlier cố định theo kênh:** kênh nào có outlier thì xuất hiện ở mọi token. Biên độ của một kênh thay đổi ít giữa các token, nhưng biên độ giữa các kênh chênh rất lớn. (3) Vì thế **lượng tử hoá activation theo kênh** (per-channel) sẽ rất tốt, nhưng *không tương thích kernel INT8 GEMM* (scale chỉ được áp ở các chiều ngoài $T$ và $C_o$, không phải chiều trong $C_i$).

Do đó: không lượng tử hoá theo kênh, mà **chia activation kênh $j$ cho $s_j$ và nhân hàng $j$ của trọng số với $s_j$**. Kết quả của phép nhân không đổi (tương đương toán học), nhưng activation đã "mịn" hơn, còn trọng số "gồ ghề" hơn một chút, nhưng vì trọng số vốn dễ nên chịu được. Phép co giãn làm offline, thường gộp được vào LayerNorm/linear phía trước nên **không tốn thêm chi phí lúc chạy**.

## 3. Phương pháp chi tiết

### 3.1 Công thức lượng tử hoá cơ sở

Lượng tử hoá đối xứng INT8 (đã giải thích ở bài 51, mục 1): 

$$
\bar X^{\rm INT8}=\Big\lceil \frac{X^{\rm FP16}}{\Delta}\Big\rfloor,\qquad \Delta=\frac{\max|X|}{2^{N-1}-1}\quad(N=8).
$$

Linear layer $Y=XW$ với $X\in\mathbb R^{T\times C_i}$ ($T$ token, $C_i$ kênh vào), $W\in\mathbb R^{C_i\times C_o}$. Điều kiện để dùng GEMM INT8 hiệu quả: scale chỉ ở chiều ngoài:

$$
Y=\mathrm{diag}(\Delta_X)\cdot(\bar X^{\rm INT8}\cdot\bar W^{\rm INT8})\cdot\mathrm{diag}(\Delta_W)\quad(2)
$$

**Bảng 1** (OPT, độ chính xác trung bình 4 tác vụ; mô phỏng): chỉ **per-channel activation** giữ được độ chính xác nhưng không chạy được trên INT8 GEMM:

| OPT- | 6.7B | 13B | 30B | 66B | 175B |
|---|---|---|---|---|---|
| FP16 | 64.9% | 65.6% | 67.9% | 69.5% | 71.6% |
| INT8 per-tensor | 39.9% | 33.0% | 32.8% | 33.1% | 32.3% |
| INT8 per-token | 42.5% | 33.0% | 33.1% | 32.9% | 31.7% |
| INT8 per-channel | 64.8% | 65.6% | 68.0% | 69.4% | 71.4% |

Phân tích "mức lượng tử hiệu dụng": nếu kênh $i$ có max $m_i$ và max toàn ma trận là $m$ thì số mức hiệu dụng của kênh $i$ là $2^8\cdot m_i/m$; kênh không phải outlier chỉ còn **2–3 mức**. (Ví dụ tự tính: outlier 60, kênh thường 0.6 ⇒ $256\times0.01=2.6$ mức.)

### 3.2 Phép làm mịn (smoothing)

Chọn $s\in\mathbb R^{C_i}$ dương, và

$$
\boxed{Y=(X\,\mathrm{diag}(s)^{-1})\cdot(\mathrm{diag}(s)\,W)=\hat X\hat W}\quad(3)
$$

$\hat X$: chia kênh $j$ của activation cho $s_j$. $\hat W$: nhân hàng $j$ của trọng số với $s_j$ (hàng $j$ của $W$ ứng với kênh vào $j$). Phép tính tương đương chính xác.

**Chọn $s$.** Nếu chọn $s_j=\max|X_j|$ thì mọi kênh activation có cùng max, nhưng đẩy hết độ khó sang trọng số (outlier chuyển thành trọng số lớn) và hỏng. Chọn $s_j=1/\max|W_j|$ thì ngược lại. Giải pháp: cân bằng bằng **migration strength $\alpha$**:

$$
\boxed{s_j=\frac{\max(|X_j|)^{\alpha}}{\max(|W_j|)^{1-\alpha}}}\quad(4)
$$

Với $\alpha=0.5$ thì sau làm mịn, max của kênh $j$ ở $\hat X$ và ở $\hat W$ bằng nhau: $\max|\hat X_j|=\max|X_j|^{1-\alpha}\max|W_j|^{\alpha}$ và $\max|\hat W_j|=\max|W_j|^{1-\alpha+\ldots}$, cụ thể ở $\alpha=0.5$ cả hai bằng $\sqrt{\max|X_j|\max|W_j|}$ ("chia sẻ cùng độ khó").

**Ví dụ số (tự tính):** kênh outlier: $\max|X_j|=60$, $\max|W_j|=0.5$, $\alpha=0.5$ ⇒ $s_j=\sqrt{60}/\sqrt{0.5}=10.95$; $\hat X_j$ max $=60/10.95=5.48$, $\hat W_j$ max $=0.5\times10.95=5.48$. Kênh thường: $\max|X_j|=0.6$, $\max|W_j|=0.5$ ⇒ $s_j=1.095$, $\hat X_j$ max $=0.548$, $\hat W_j$ max $=0.548$. Tỉ lệ biên độ giữa kênh outlier và kênh thường ở activation giảm từ $100\times$ xuống $10\times$ (căn bậc hai); ở trọng số tăng từ $1\times$ lên $10\times$. Độ khó được **chia đều** ($\sqrt{100}$ mỗi bên).

Giá trị $\alpha$ trong paper: $0.5$ cho mọi OPT, BLOOM; $0.75$ cho GLM-130B (activation khó hơn, ~30% outlier); cho LLaMA/Llama-2/Falcon/Mistral/Mixtral: 0.8 (LLaMA-1 Bảng 6), 0.85–0.9 (Llama-2), 0.6–0.7 (Falcon), 0.8 (Mistral, Mixtral). $\alpha$ tìm bằng grid search nhanh trên một tập con validation Pile.

**Hiệu chuẩn.** 512 câu ngẫu nhiên từ Pile (pre-training set) để tính $\max|X_j|$ (và scale tĩnh); dùng cho mọi tác vụ downstream.

### 3.3 Áp vào khối Transformer

Làm mịn cho **đầu vào của attention (QKV) và FFN**: gộp $1/s$ vào tham số của LayerNorm phía trước; nếu đầu vào đến từ phép cộng residual thì thêm một phép co giãn ở nhánh residual (theo Wei et al., 2022). Tất cả linear và BMM trong attention dùng INT8; các phép nhẹ element-wise (ReLU, Softmax, LayerNorm) giữ FP16 (Hình 6).

### 3.4 Ba mức hiệu quả (Bảng 2)

| Cấu hình | Trọng số | Activation |
|---|---|---|
| W8A8 ngây thơ | per-tensor | per-tensor, dynamic |
| ZeroQuant | group-wise | per-token, dynamic |
| LLM.int8() | per-channel | per-token, dynamic + FP16 (outlier) |
| Outlier Suppression | per-tensor | per-tensor, static |
| **SmoothQuant-O1** | per-tensor | per-token, dynamic |
| **SmoothQuant-O2** | per-tensor | per-tensor, dynamic |
| **SmoothQuant-O3** | per-tensor | per-tensor, **static** |

O1 → O3: càng thô càng nhanh (độ trễ thấp hơn), có thể mất chút chính xác.

### 3.5 Pseudo-code

```
# OFFLINE (một lần, ~512 câu calibration)
for each linear layer L (đầu vào X được tạo bởi lớp trước P):
    run calibration set -> max_X[j] = max over tokens |X[:, j]|
    max_W[j] = max over output ch |W[j, :]|
    s[j] = max_X[j]**alpha / max_W[j]**(1-alpha)
    L.W[j, :] *= s[j]                  # hàng j của W
    P.scale_output_channel(j, 1/s[j])  # gộp vào LayerNorm / linear trước (không tốn lúc chạy)
quantize W per-tensor INT8; (static) quantize scale của activation từ calibration

# ONLINE
X8 = quant(X_hat, delta_X)             # per-tensor (static: delta_X cố định)
Y  = INT8_GEMM(X8, W8) * (delta_X * delta_W)   # dequant ở epilogue
```

## 4. Thí nghiệm và kết quả chính

**Mô hình:** OPT (đến 175B), BLOOM-176B, GLM-130B, thêm LLaMA/Llama-2/Falcon/Mistral/Mixtral, MT-NLG 530B. **Tác vụ:** 7 tác vụ zero-shot (LAMBADA, HellaSwag, PIQA, WinoGrande, OpenBookQA, RTE, COPA) + WikiText cho OPT/BLOOM; MMLU, MNLI, QNLI, LAMBADA cho GLM-130B. Chỉ báo cáo thay đổi tương đối so với FP16.

**Bảng 3, OPT-175B:**

| Phương pháp | LAMBADA | HellaSwag | PIQA | WinoGrande | OBQA | RTE | COPA | Trung bình | WikiText ppl |
|---|---|---|---|---|---|---|---|---|---|
| FP16 | 74.7% | 59.3% | 79.7% | 72.6% | 34.0% | 59.9% | 88.0% | 66.9% | 10.99 |
| W8A8 | 0.0% | 25.6% | 53.4% | 50.3% | 14.0% | 49.5% | 56.0% | 35.5% | 93080 |
| ZeroQuant | 0.0% | 26.0% | 51.7% | 49.3% | 17.8% | 50.9% | 55.0% | 35.8% | 84648 |
| LLM.int8() | 74.7% | 59.2% | 79.7% | 72.1% | 34.2% | 60.3% | 87.0% | 66.7% | 11.10 |
| Outlier Suppression | 0.00% | 25.8% | 52.5% | 48.6% | 16.6% | 53.4% | 55.0% | 36.0% | 96151 |
| SmoothQuant-O1 | 74.7% | 59.2% | 79.7% | 71.2% | 33.4% | 58.1% | 89.0% | 66.5% | 11.11 |
| SmoothQuant-O2 | 75.0% | 59.0% | 79.2% | 71.2% | 33.0% | 59.6% | 88.0% | 66.4% | 11.14 |
| SmoothQuant-O3 | 74.6% | 58.9% | 79.7% | 71.2% | 33.4% | 59.9% | 90.0% | 66.8% | 11.17 |

**Bảng 4 (trung bình 4 tác vụ; GLM-130B: LAMBADA, MMLU, MNLI, QNLI, khác tác vụ nên không so cột):** OPT-175B FP16 71.6%, W8A8 32.3%, SQ-O1/O2/O3 71.2/71.1/71.1%; BLOOM-176B FP16 68.2%, W8A8 64.2%, SQ-O1/O2/O3 68.3/68.4/67.4%; GLM-130B FP16 73.8%, W8A8 26.9%, SQ-O1/O2/O3 73.7/72.5/72.8%. LLM.int8() giữ 71.4/68.0/73.8%.

**LLaMA/Llama-2/… (perplexity WikiText-2, W8A8 SmoothQuant so FP16):**

| Model | FP16 | SQ W8A8 | α |
|---|---|---|---|
| LLaMA 7B / 13B / 30B / 65B (seq 512) | 11.51 / 10.05 / 7.53 / 6.17 | 11.56 / 10.08 / 7.56 / 6.20 | 0.8 |
| Llama-2-7B | 5.474 | 5.515 | 0.85 |
| Llama-2-13B | 4.950 | 4.929 | 0.85 |
| Llama-2-70B | 3.320 | 3.359 | 0.9 |
| Falcon-7B | 6.590 | 6.629 | 0.6 |
| Falcon-40B | 5.228 | 5.255 | 0.7 |
| Mistral-7B | 5.253 | 5.277 | 0.8 |
| Mixtral-8x7B | 3.842 | 3.893 | 0.8 |

Instruction-tuned: OPT-IML-30B (LAMBADA/WikiText): FP16 69.12% / 14.26; SQ-O3 69.77% / 14.37; W8A8 4.21% / 576.53. MT-NLG 530B (Bảng 9): FP16 trung bình 73.1% vs INT8 73.1%.

Tác giả nhận xét LLaMA có ít vấn đề outlier hơn OPT/BLOOM nhưng SmoothQuant vẫn dùng tốt.

## 5. Phân tích / ablation

**Migration strength $\alpha$ (Hình 10, OPT-175B, LAMBADA):** $\alpha<0.4$ ⇒ activation vẫn khó; $\alpha>0.6$ ⇒ trọng số khó. Vùng tốt **0.4–0.6**. (Số liệu chi tiết trong hình không đọc được từ bản trích text.)

**Mức lượng tử hoá (Bảng 11, PyTorch, batch 4, ms):** càng thô càng nhanh; static nhanh hơn dynamic vì khỏi tính scale lúc chạy. Một số dòng đọc được: FP16 OPT-13B (seq 256/512) 152.6/296.3 ms, OPT-30B 343.0/659.9; LLM.int8() 237.1/371.5 và 387.9/654.9 (chậm hơn FP16); SQ-O1 124.5/243.3 và 246.7/490.7; SQ-O2 120.5/235.1 và 240.2/478.3; SQ-O3 112.1/223.1 và 227.6/458.4. SmoothQuant nhanh hơn FP16 ở mọi cấu hình, LLM.int8() thường chậm hơn.

**O3 mất chính xác nhẹ:** trên BLOOM-176B O3 thấp hơn FP16 0.8 điểm, tác giả cho là do khác biệt giữa thống kê tĩnh và thống kê thật; GLM-130B khó hơn (O3 giảm ~1 điểm). Khi calibrate scale tĩnh cho GLM-130B có cắt 2% token lớn nhất (theo Wei et al., 2022).

## 6. Phân tích bộ nhớ và độ trễ (batch 1)

### 6.1 Số đo trong paper

- **Context stage (prefill), PyTorch, batch 4, một A100-80GB:** SmoothQuant-O3 đạt tới **1.51×** nhanh hơn và **1.96×** tiết kiệm bộ nhớ (OPT-30B, seq 256); mô hình càng lớn càng được lợi. FasterTransformer: tới **1.56×** (OPT-13B, OPT-30B một GPU). OPT-66B chạy 1 GPU thay vì 2, OPT-175B 4 GPU thay vì 8 với độ trễ tương đương.
- **Decoding stage (Bảng 8, mỗi token):**

| Cấu hình | Batch | SeqLen | FP16 (ms) | SQ (ms) | Speedup | Mem FP16 (GB) | Mem SQ (GB) | Tiết kiệm |
|---|---|---|---|---|---|---|---|---|
| OPT-30B, 1 GPU | 1 | 512 | 422 | 314 | 1.35× | 57 | 30 | 1.91× |
| OPT-30B, 1 GPU | 1 | 1024 | 559 | 440 | 1.27× | 58 | 31 | 1.87× |
| OPT-30B, 1 GPU | 16 | 512 | 2488 | 1753 | 1.42× | 69 | 44 | 1.59× |
| OPT-175B, 8 GPU | 1 | 512 | 426 | 359 | 1.19× | 44 | 23 | 1.87× |
| OPT-175B, 8 GPU | 1 | 1024 | 571 | 475 | 1.20× | 44 | 24 | 1.85× |

(Dòng OPT-30B batch 16, seq 1024: FP16 hết bộ nhớ (OOM), SmoothQuant chạy 3947 ms với 61 GB. Dòng OPT-175B batch 16 cũng có trong Bảng 8 nhưng không trích ở đây.) Con số "ms" ở Bảng 8 là thời gian đo của paper (bản trích không nói rõ đơn vị "mỗi token" hay "mỗi lượt"; cần xem lại paper gốc khi trích dẫn).

- **MT-NLG 530B:** FP16 cần 16 GPU (1040 GB), INT8 chỉ 8 GPU (527 GB), độ trễ tương tự (seq 128: 232 ms vs 253 ms), cho phép phục vụ >500B trong một node 8×A100-80GB.

### 6.2 Quy luật lợi ích, hiểu bằng roofline

Prefill có nhiều token ⇒ GEMM lớn, **compute-bound**: INT8 có tốc độ tính tối đa ~2× FP16 trên Tensor Core, cộng nửa băng thông ⇒ lợi từ cả hai. Decode batch 1 là phép nhân ma trận–vector, **memory-bound**: lợi ích chỉ đến từ việc đọc ít byte hơn, tối đa ~2× với W8 (thực đo 1.19–1.42×). Phép lượng tử hoá activation (A8) *không giúp thêm gì* ở batch 1 vì activation chỉ vài KB; nó chỉ cần để dùng được INT8 GEMM khi tính toán là nút cổ chai.

### 6.3 Ví dụ số theo cỡ mô hình của người đọc (tự tính)

Giả định (cần thay bằng số đo thật): ~0.8B tham số (1.6 GB FP16), $d=1024$, ~24 lớp, băng thông 300 GB/s, 1,500 token đầu ra, batch 1.
- **Decode, W8:** 1.6 → 0.8 GB, chặn dưới $5.3\to2.7$ ms/token (8.0 s → 4.0 s cho trang); nhưng ở mô hình nhỏ thời gian thực bị chi phối bởi launch kernel và lớp không lượng tử hoá (LM head, norm), nên speedup thực sẽ thấp hơn mức lý tưởng, giống 1.19–1.42× thay vì 2×.
- **KV cache** (nếu một nửa số lớp, ví dụ 6/24 lớp full-attention, 8 KV-head × 128 chiều, 3,500 token cả prefill + decode, FP16): $2\times6\times8\times128\times2\text{ B}\times3500\approx86$ MB, nhỏ hơn rất nhiều so với trọng số 1.6 GB. Cho thấy ở dự án này đòn bẩy decode là **trọng số**, không phải KV (các lớp GatedDeltaNet có trạng thái cố định nhỏ).
- **Prefill (ViT + token ảnh):** nơi compute-bound; W8A8 có thể cho speedup tính toán gần ~1.5×–2× về lý thuyết, nhưng phụ thuộc vào kernel/GPU hỗ trợ INT8 Tensor Core (chỉ là cận lý thuyết, cần đo).

## 7. Giới hạn

**Tác giả nêu:**
- Với mô hình khó (GLM-130B) cần $\alpha$ lớn hơn và O3 mất ~1%; kiến trúc/huấn luyện khác nhau có độ khó lượng tử khác nhau ("hy vọng khơi gợi nghiên cứu").
- Chỉ lượng tử hoá GEMM/BMM; các phép phi tuyến giữ FP16.
- Static quantization lệch giữa thống kê calibration và dữ liệu thật (BLOOM-176B O3 −0.8%).
- Phần so sánh GPTQ: GPTQ chỉ lượng tử hoá trọng số (nhắc trong related work / phụ lục A).

**Chúng ta nhận thấy:**
- Hiệu quả dựa trên giả thuyết "outlier cố định theo kênh". Với kiến trúc có nhiều dạng outlier theo *token* (ví dụ "massive activations" ở token đầu) thì per-tensor static có thể kém; paper chưa phân tích.
- $\alpha$ và scale tĩnh calibrate trên Pile/văn bản; với OCR (phân phối token khác: chữ số, ký hiệu) cần calibrate bằng dữ liệu OCR.
- Gộp $1/s$ vào LayerNorm: với **RMSNorm có trọng số** vẫn gộp được, nhưng với các nhánh qua cổng/ GatedDeltaNet, conv ngắn, hay khi đầu vào đến từ nhiều nguồn (residual cộng) thì phải xử lý riêng; paper chỉ mô tả LayerNorm/linear/residual.
- Kết quả speedup đo trên FasterTransformer + A100, mô hình ≥ 6.7B; mô hình ~1B chưa đo.
- Hầu hết thí nghiệm tập trung perplexity/zero-shot; chưa có sinh dài hoặc OCR.

## 8. Liên hệ với các bài khác

- **51 (LLM.int8()):** cùng chẩn đoán; thay "tách FP16" bằng phép biến đổi tương đương (không cần mixed precision ⇒ nhanh hơn). Bảng 3: cả hai khớp FP16 ở OPT-175B, nhưng latency SmoothQuant thấp hơn.
- **53 (GPTQ):** weight-only 3–4 bit dựa trên OBS; không lượng tử hoá activation, nên không có kernel INT8 thuần nhưng nén mạnh hơn nhiều. Hai bài bổ trợ: W4 cho decode, W8A8 cho prefill.
- **54 (AWQ):** cũng nhân kênh theo $s$ nhưng cho weight-only, chọn $s$ dựa trên biên độ activation để bảo vệ kênh trọng số quan trọng, tư tưởng gần với công thức (4).
- **10 (RMSNorm), 9 (Pre-LN):** nơi gộp $1/s$.
- **Pruning (24 Wanda):** Wanda dùng $|w|\cdot\|x_j\|$: chính là tích biên độ trọng số × biên độ activation theo kênh, cùng đại lượng cân bằng trong công thức (4) của SmoothQuant (nhận xét của chúng ta).

## 9. Ứng dụng cho dự án (suy luận)

1. **Phân tách theo giai đoạn:** decode batch 1 (memory-bound, đa số thời gian của 1,500 token) hưởng lợi từ weight-only (bài 53/54) hơn là W8A8; **prefill ảnh + ViT 12 khối** (compute-bound, nhiều token) hưởng lợi từ W8A8. SmoothQuant do đó hợp với nửa ViT/prefill hơn nửa decode.
2. **ViT:** paper chỉ thử LLM; ViT có outlier theo kênh tương tự nhưng chưa kiểm trong paper. Cần calibrate riêng, đo thống kê kênh trước.
3. **Hybrid GatedDeltaNet:** các lớp linear attention có các phép chiếu (q, k, v, gates/β) và conv ngắn; độ nhạy lượng tử của đường gate/decay (giá trị gần 0–1 tích luỹ qua nhiều bước) khác attention thường; **không có trong paper** nên đây là rủi ro cần thử, nhất là khi lượng tử activation vào trạng thái hồi quy (sai số tích luỹ theo độ dài 1,500 token).
4. **Sau khi prune + distill:** mô hình nhỏ (0.1B–0.3B) có ít dư thừa hơn, nhạy lượng tử hơn (xem GPTQ bài 53, Bảng 3: OPT-125M mất nhiều nhất). Thứ tự gợi ý: prune → distill → lượng tử hoá (hoặc QAT nhẹ), calibrate bằng dữ liệu OCR.
5. **Ít liên quan nếu bộ nhớ không phải vấn đề:** 1.6 GB đã vừa 8 GB; lợi ích duy nhất là tốc độ. Với mô hình nhỏ + batch 1, đo độ trễ thật trước khi đầu tư.

## 10. Thí nghiệm gợi ý

1. **Chẩn đoán kênh:** vẽ $\max|X_j|$ theo kênh cho mọi linear (decoder, ViT) và tính tỉ lệ max/median; nếu < ~10× thì per-tensor W8A8 ngây thơ có thể đã đủ, SmoothQuant cũng đơn giản để thử ($\alpha\in\{0.5,0.7,0.85\}$).
2. **W8A8 chỉ cho ViT + prefill:** so thời gian prefill (ViT + token ảnh) giữa FP16 và SQ-O3 bằng kernel INT8 trên GPU mục tiêu; ghi CER/WER OCR.
3. **Calibrate OCR vs Pile:** so SQ với scale từ văn bản chung vs từ trang OCR (512 mẫu) để đo độ lệch static.

## 11. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý |
|---|---|
| Lượng tử hoá | $\bar X=\lceil X/\Delta\rfloor$, $\Delta=\max|X|/(2^{N-1}-1)$ |
| GEMM INT8 hợp lệ | scale chỉ ở chiều ngoài: $Y=\mathrm{diag}(\Delta_X)(\bar X\bar W)\mathrm{diag}(\Delta_W)$ |
| Làm mịn | $Y=(X\mathrm{diag}(s)^{-1})(\mathrm{diag}(s)W)$ |
| Hệ số | $s_j=\max|X_j|^{\alpha}/\max|W_j|^{1-\alpha}$ |
| $\alpha$ | 0.5 (OPT, BLOOM), 0.75 (GLM-130B), 0.8–0.9 (Llama) |
| O1/O2/O3 | per-token dyn / per-tensor dyn / per-tensor static (activation) |
| Hiệu quả | ≤ 1.56× nhanh, ≈ 2× bộ nhớ (OPT-175B: 8→4 GPU; MT-NLG 530B: 16→8 GPU) |

## 12. Kết luận ngắn

SmoothQuant biến bài toán "activation có outlier theo kênh" thành bài toán dễ hơn bằng phép co giãn theo kênh tương đương, cho phép W8A8 per-tensor static thuần INT8 giữ độ chính xác từ 6.7B đến 530B và đem lại tới 1.56× tốc độ, ~2× bộ nhớ. Với dự án batch 1 trên mô hình nhỏ, nó hữu ích nhất cho phần compute-bound (ViT, prefill), còn decode nên dựa vào lượng tử hoá chỉ trọng số.

Link gốc: https://arxiv.org/abs/2211.10438
