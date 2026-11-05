# ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization (2025)

**Tác giả:** Dmitriy Shopkhoev, Ammar Ali, Magauiya Zhussip, Valentin Malykh, Stamatios Lefkimmiatis, Nikos Komodakis, Sergey Zagoruyko (MWS AI, ITMO University, University of Crete, Polynome...).  
**Venue / arXiv:** arXiv:2505.02819 (v4, 19/02/2026); bản text ghi "accepted and presented at NeurIPS 2025".  
**Nguồn:** https://arxiv.org/abs/2505.02819  
**Chủ đề:** Depth pruning không train: thay một khối layer liền kề bằng **một ma trận tuyến tính** ước lượng từ dữ liệu calibration, rồi gộp ma trận này vào trọng số `down_proj` của MLP, nên không thêm tham số.  
**Vị trí trong lộ trình:** Đọc trước: 32 Deeper layers (đây là "UIDL" trong bài này, baseline chính), 31 ShortGPT, 35 LaCo. Bài này **gần nhất với $D_{lin}$ của bạn**: bước tìm $T$ chính là tìm ánh xạ tuyến tính tối ưu tại chỗ cắt. Đọc tiếp: 37 (cho thấy repair tuyến tính này đổi *layer nào nên cắt*), 39 Minitron.

> Tài liệu học tập tiếng Việt, ghi chú diễn giải, không phải bản dịch paper. Số liệu lấy từ paper; bảng đã đọc kỹ. Phần ứng dụng cho dự án là suy luận của chúng ta.

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

- Các bài depth pruning trước (31, 32, 34) cắt xong thường cần **healing** (fine-tune, thường LoRA) để hồi phục; LLM-Streamline thay khối bị cắt bằng một mạng nhẹ (một layer transformer hoặc FFN) rồi train với MSE + LLM loss. Tốn thời gian và tài nguyên.
- **Least squares (bình phương tối thiểu):** cho $A$ và $B$, tìm $T$ làm $\lVert AT-B\rVert_2^2$ nhỏ nhất; nghiệm đóng $T=(A^\top A)^{-1}A^\top B$. Không cần gradient descent.
- **Calibration set:** vài nghìn mẫu không nhãn dùng để lấy hidden state, không để train theo nghĩa thông thường.
- Quan sát từ công trình khác (do bài này trích dẫn): nhiều transformer block gần như tuyến tính trong ánh xạ embedding giữa các layer.

## 2. Ý tưởng chính

Giả thuyết của tác giả: **một khối liền kề các transformer block có thể xấp xỉ bằng một phép biến đổi tuyến tính**. Cắt $n$ block sau block $i^*$, rồi tìm ma trận $T$ sao cho "đầu ra attention của block $i^*$ cộng với MLP-output của nó nhân $T$" khớp với "đầu ra của block $i^*+n$ trước khi cắt". $T$ ước lượng trên calibration bằng nghiệm đóng (L2) hoặc tối ưu số (cosine). Vì $T$ nhân liền với `down_proj` của MLP nên có thể gộp vào, mô hình sau cắt có cấu trúc y như Transformer thường. Kết quả: cắt 25% mà không train vẫn giữ ~90% hiệu năng.

## 3. Phương pháp chi tiết

### 3.1 Ký hiệu một block (Eq. 1–3)

$X_i\in\mathbb R^{N\times d}$ là đầu vào block $i$ ($N$ token, $d$ chiều).

$$Y_i=X_i+\text{MHA}_i(\text{LN}^{(1)}_i(X_i)),\qquad M_i=\text{MLP}_i(\text{LN}^{(2)}_i(Y_i)),\qquad L_i=Y_i+M_i$$

$Y_i$ = sau attention (đã cộng residual); $M_i$ = đầu ra MLP; $L_i$ = đầu ra block.

### 3.2 Chọn khối cắt (Eq. 4)

$$i^*=\arg\min_i\ h(L_i,L_{i+n})$$

$h$ là khoảng cách trung bình trên calibration. Cosine distance cho kết quả tốt nhất (ablation brute-force ở A.10 xác nhận cosine chọn khối tối ưu hoặc gần tối ưu).

### 3.3 Ước lượng $T$ (Eq. 5–8)

$$\boxed{T^*=\arg\min_T\ h\big(M_{i^*}T+Y_{i^*},\ L_{i^*+n}\big)}$$

Sau đó xoá block $i^*+1,\dots,i^*+n$. $T$ nhân lên $M_{i^*}$ (đầu ra MLP của block cuối cùng còn giữ), còn $Y_{i^*}$ vẫn là đường residual.

- **Dạng L2 (nghiệm đóng):**
$$T^*=\arg\min_T\lVert M_iT+Y_i-L_{i+n}\rVert_2^2=(M_i^\top M_i)^{-1}M_i^\top(L_{i+n}-Y_i)$$
- **Dạng cosine** (không có nghiệm đóng, tối ưu bằng Adam):
$$T^*=\arg\min_T\sum_{k=1}^{N}\Big(1-\frac{(M_{i,k}T+Y_{i,k})^\top L_{i+n,k}}{\lVert M_{i,k}T+Y_{i,k}\rVert\,\lVert L_{i+n,k}\rVert}\Big)$$
- **Dạng tiết kiệm bộ nhớ (Eq. 8):** $\arg\min_T \cos(M_iT,\ L_{i+n}-Y_i)$ chỉ cần lưu $M_i$ và $L_{i+n}-Y_i$; A.11 báo ảnh hưởng không đáng kể.
- **Regularization (Eq. 9):** thêm $\alpha R(T)$ với L1/L2. Quan sát: tăng chính xác benchmark nhưng tăng perplexity (trade-off).
- **Gộp (fuse):** vì $T$ và `down_proj` là hai phép tuyến tính liên tiếp, $W_{down}'=W_{down}T$ (theo mô tả: ghép đại số vào ma trận `down_proj` của MLP).
- **Nhiều ma trận (Multi-LT, mục 2.4):** chia các block bị bỏ thành nhiều nhóm không chồng, mỗi nhóm một $T$; nhóm kề nhau thì gộp. Biến thể "Multi_LT_NC" ép các nhóm không liên tiếp.

```
ReplaceMe
Input: model, calibration set D (vd. 8k mẫu SlimOrca), số block n cần bỏ
1. chạy model trên D, lưu L_i cho mọi i
2. i* = argmin_i cosine_dist(L_i, L_{i+n})               # chọn khối
3. lưu M_{i*}, Y_{i*}, L_{i*+n} trên D
4. T = lstsq(M_{i*}, L_{i*+n} - Y_{i*})                   # L2: nghiệm đóng
   hoặc: T = Adam(cos loss), khởi tạo T = I, lr 1e-4, batch 1024, 10 epoch
5. W_down[i*] <- W_down[i*] @ T                           # gộp vào MLP, 0 tham số mới
6. xoá block i*+1 .. i*+n
```

Chi tiết cấu hình (A.15): Adam, LR $10^{-4}$, batch 1024, 10 epoch, loss cosine, khởi tạo T là ma trận đơn vị, không bias. Chạy 1 GPU A100 40GB (có thử cả P100 trên Kaggle).

Chú ý (của chúng ta): $T$ là ánh xạ **$d\times d$ từ đầu ra MLP của block cuối giữ lại**, không phải từ đầu vào khối bị cắt, nên không đúng bằng "ánh xạ tuyến tính tối ưu từ $X_{i^*+1}$ sang $L_{i^*+n}$". Khác biệt này quan trọng khi so với $D_{lin}$ (xem mục 8).

## 4. Thí nghiệm và kết quả chính

**Mô hình:** Llama-2-7B, Llama-3-8B-Instruct (chính); Qwen2.5-7B, Falcon-11B, Mistral-7B, Llama-3.2-1B, Llama-3-70B; CLIP ViT-L/14 (vision encoder). **Benchmark:** CMNLI, HellaSwag, PIQA, CHID, WSC, MMLU, CMMLU, Race-H/M, C3 (theo LLM-Streamline); thêm Winogrande, BoolQ, OpenBookQA, SciQ, Lambada-OpenAI (perplexity). RP = % so với baseline.

**Table 1 (Llama-2-7B, cắt 25%; baseline Avg 45.3):**

| Phương pháp | Train-free | AVG | RP |
|---|---|---|---|
| LLM-Streamline* | không | 41.6 | 92.0% |
| LLM-Pruner* | không | 35.4 | 78.2% |
| SliceGPT* | không | 35.1 | 77.5% |
| LaCo* | không | 37.4 | 82.7% |
| UIDL* | không | 40.9 | 90.3% |
| **ReplaceMe (Cosine)** | **có** | **41.9** | **92.5%** |
| ReplaceMe (LS) | có | 40.7 | 89.9% |

(* = số lấy từ paper LLM-Streamline; các baseline đều có bước train/heal, ReplaceMe thì không.) Ghi chú: LLM-Streamline đạt RP 92.0% với train, ReplaceMe cosine 92.5% không train, tức hơn nhẹ; hơn UIDL 2.2 điểm RP.

**Table 2 (Llama-3-8B-Instruct, cắt 25%, không heal; Avg-acc trên Race, Winogrande, PIQA, BoolQ, OBQA, SciQ):**

| Phương pháp | Linear transform | PPL Lambada | Avg-acc | RP |
|---|---|---|---|---|
| Baseline | | 3.11 | 0.7 | 100% |
| SVD-LLM | không | 29.90 | 0.59 | 85.3% |
| LLM-Pruner | không | 12.31 | 0.60 | 85.3% |
| UIDL | Identity | 2216.96 | 0.58 | 82.5% |
| ReplaceMe | LS | 20.23 | 0.63 | 89.9% |
| ReplaceMe | Cosine | 15.88 | 0.63 | 90.9% |
| ReplaceMe | Multi_LT_NC (Cosine) | 13.95 | 0.63 | 90.0% |

**Mức nén cao (Table 3, avg acc, Llama 3 8B Instruct):** UIDL: 59.9, 50.4, 41, 30.4; ReplaceMe: 64.6, 60.7, 51, 38.6 (nén 12.5%, 25%, 37.5%, 50%). Tác giả thừa nhận ở mức nén cao vẫn cần healing.

**Quy mô (Table 9):** Llama-3.2-1B-Instruct cắt 25%: RP 0.8771 (0.6098 -> 0.5348); Llama-3.1-8B-Instruct 25%: RP 0.9184; Llama-3-70B-Instruct 25%: RP 0.9664, 37.5%: RP 0.9060. Xu hướng: mô hình càng lớn càng nén được nhiều ở cùng RP.

**Chi phí so với UIDL (Fig. 2/4):** ReplaceMe nhanh hơn và ít phát thải hơn UIDL có LoRA healing (các tỉ số trong hình: 169x, 107x, 6.6x, 3.5x...; hình bị lẫn, không trích chắc chắn từng cột).

**ViT (CLIP-L/14, Table 6, ReplaceMe LS):** nén 13%: MS-COCO text recall@5 0.767 (gốc 0.794, UIDL 0.745), Cifar10 acc1 0.939 (gốc 0.956, UIDL 0.927), VTAB/EuroSAT acc1 0.552 (gốc 0.625, UIDL 0.490). Nén 25%: text recall@5 0.556 vs UIDL 0.515; Cifar10 acc1 0.780 vs 0.693; VTAB/EuroSAT acc1 0.395 vs 0.381. Calibration 8000 mẫu MIMIC. Tức: tốt hơn UIDL, giữ khá ở 13%, tụt ở 25%.

## 5. Phân tích / ablation

- **Dữ liệu calibration (Table 4, Llama-3-8B-Instruct, 25%):** FineWeb 8k (plain text): LS Avg-acc 0.56, PPL 26.74, RP 80.47%; SlimOrca 8k (instruction): LS 0.62, PPL 21.21, RP 89.59%; Cosine SlimOrca: 0.63, PPL 15.90, RP 90.67%; "orca_generated" (đầu ra do chính mô hình sinh): PPL thấp nhất (13.24–13.58) nhưng Avg-acc thấp hơn (0.61). Trộn đa ngôn ngữ (66 ngôn ngữ) ngang SlimOrca. Kết luận tác giả: dữ liệu instruction chất lượng cao quan trọng nhất; dữ liệu sinh tự thân cho PPL tốt nhưng benchmark kém hơn.
- **Calibration đúng tác vụ (A.17, Table 17, SciQ, Llama3-8B-Instruct 25%):** gốc 0.93; ReplaceMe với calib SciQ 0.89; calib Orca 0.858; UIDL 0.687.
- **Cấu trúc của $T$ (Table 7):** Generic Avg 0.62, PPL 21.21, RP 89.59; Diagonal 0.62, PPL 89.09, RP 88.42; Orthonormal 0.60, PPL 700.57, RP 85.67. Tác giả suy ra "co giãn (scaling) quan trọng hơn quay/phản xạ" để hồi phục.
- **Regularization (Table 5):** L2 với LS gần như không đổi; cosine + L1 ($10^{-4}$) tăng Avg-acc 0.634 -> 0.638 nhưng PPL 15.9 -> 22.1.
- **Solver (Table 8):** Adam, NLCG, L-BFGS cho kết quả gần nhau (Avg 0.634/0.630/0.634); Trust-NCG kẹt cực tiểu cục bộ (0.620, PPL 47.0). MSE+Adam gần như trùng LS.
- **Độ ổn định (A.5):** 8 layer bị cắt trên Llama-3-8B-Instruct: Avg-acc trung bình 60.71 (độ lệch chuẩn 0.08), PPL 15.90 (std 0.05).
- **Số ma trận LT (Table 16, tổng 8 layer cắt):** Llama-3-8B-Instruct: 1 LT 0.6244 / PPL 21.2; 2 LT 0.6296 / 18.99; 4 LT 0.6245 / 16.07; 8 LT 0.6092 / 37.98. Mistral-7B: 1 LT 0.6545; 2 LT 0.6227; 4 LT 0.6568; 8 LT 0.6510. Kết luận tác giả: 2–4 LT là cân bằng, khác biệt nhỏ.
- **LT gộp vs LT là block riêng (Table 10):** gộp vào MLP: Avg 0.634, PPL 15.875; block riêng: 0.640, PPL 13.481.
- **Healing (Table 13, Llama-2-7B):** gốc Avg 0.6752 / PPL 3.3973; ReplaceMe không heal 0.61295 / 22.3586; heal chỉ $T$ (16.8M tham số) 0.64432 / 5.3965; heal toàn mô hình (6.3B tham số) 0.64657 / 4.9138. Tức heal chỉ $T$ gần bằng full fine-tune với chi phí rất nhỏ.
- **Chọn khối (A.10, Fig. 5–6):** cosine distance nhỏ tương quan với accuracy cao; L2 distance kém dự báo; CKA và CCA cũng xét.

## 6. Giới hạn

**Do tác giả nêu:** hiệu quả chỉ rõ trong dải nén "hợp lý" phụ thuộc kích thước và kiến trúc; nén cao cần retrain hoặc ít nhất chỉnh $T$ (healing chỉ-LT là lối thoát rẻ).

**Do chúng ta nhận thấy:**
- Hầu hết đánh giá là benchmark trắc nghiệm và perplexity Lambada; không có tác vụ sinh dài. PPL Lambada tăng từ 3.11 lên 15.88–20.23 ở 25% cho thấy mô hình ngôn ngữ vẫn xấu đi nhiều dù accuracy trắc nghiệm giữ ~90%.
- Chọn khối **không xét repair** (Eq. 4 dùng khoảng cách trước khi có $T$); bài 37 chỉ ra sai khác này có thể lớn.
- Chỉ một khối liền kề (hoặc vài nhóm); không có tìm kiếm tổ hợp.
- Chỉ Transformer chuẩn; $T$ nhân lên $M_i$, đòi hỏi block có MLP với down_proj để gộp.
- Mô hình 1B mất nhiều hơn (RP 0.877 ở 25%): nhỏ hơn thì khó hơn, bất lợi cho mô hình 0.8B của bạn.
- Số liệu Table 1 của đối thủ lấy từ paper khác (dấu *), không chạy lại.

## 7. Liên hệ với các bài khác

- **32 Deeper layers (UIDL):** baseline chính. UIDL = chọn khối bằng khoảng cách góc, nối thẳng (Identity), rồi heal LoRA. ReplaceMe thay "nối thẳng + LoRA" bằng "T tối ưu + không train", và ablation cho thấy "Linear transform: Identity" của UIDL có PPL 2216.96.
- **31 ShortGPT, 35 LaCo:** baseline đã có trong Table 1 (LaCo 82.7%).
- **34 SLEB:** greedy chọn từng block bằng NLL; ReplaceMe chọn khối bằng cosine rồi sửa bằng T.
- **37:** xét đúng điều bài này bỏ qua: nếu biết bước repair là $T$ thì chọn khối thế nào.
- **29 SliceGPT:** cũng dùng phép biến đổi tuyến tính (phép chiếu PCA) và gộp vào trọng số; điểm chung là "sửa bằng phép tuyến tính được gộp".
- **39 Minitron:** phương pháp dùng 100B token để hồi phục (theo mô tả của bài này), đối lập với training-free.

## 8. Ứng dụng cho dự án (suy luận)

- **Liên quan cao nhất tới $D_{lin}/D_{res}$.** Nghiệm LS ở Eq. 6 chính là "phần hư hại loại bỏ được bằng một phép tuyến tính tối ưu"; phần dư $\lVert M_iT^*+Y_i-L_{i+n}\rVert$ là ứng viên trực tiếp cho $D_{res}$. Bạn có thể báo cáo hai số này cho từng ứng viên khối cắt. Lưu ý: định nghĩa của bài (T nhân $M_i$, cộng $Y_i$) hơi khác ánh xạ tuyến tính tổng quát từ đầu vào khối, bạn nên tự định nghĩa lại $D_{lin}$ cho khớp.
- Bạn thấy "cắt neuron MLP hồi phục kém hơn cắt layer". Bài này gợi ý một lý do có thể: khối layer có thể được xấp xỉ khá tốt bằng một phép tuyến tính (T) nên "đền bù" được; neuron MLP bị cắt thì không có phép sửa tương đương dễ dàng. Đây là giả thuyết của chúng ta, bài này không so sánh với cắt neuron.
- Calibration phải là **dữ liệu OCR thật** (ảnh trang -> token), vì Table 4 và 17 cho thấy dữ liệu đúng miền cải thiện rõ; dữ liệu sinh từ mô hình cho PPL tốt nhưng benchmark kém hơn, cần kiểm chứng với CER.
- Healing chỉ-$T$ (16.8M tham số, Table 13) là bước trung gian rẻ trước khi distill đầy đủ từ teacher 4B.
- ViT: có kết quả CLIP-L/14 (13% còn tốt, 25% tụt); với ViT 12 block của bạn, có thể dùng cho thí nghiệm cắt 1–2 block.
- Decoder lai GatedDeltaNet: cần có MLP + down_proj ở block cuối để gộp $T$; nếu khối cắt kết thúc ở GatedDeltaNet thì vẫn gộp được vào `down_proj` của MLP, nhưng bài này chưa kiểm chứng trên kiến trúc lai.
- Mô hình bạn rất nhỏ (0.8B); kết quả 1B (RP 87.7% ở 25%) và mục tiêu cắt ~60% (0.3B) nằm ngoài dải mà bài này khẳng định.

## 9. Thí nghiệm gợi ý

1. Tính $D_{lin}$ (L2 sau LS) và $D_{res}$ cho mọi khối liền kề độ dài 1–4 của decoder và ViT trên 2k trang OCR; vẽ heat-map, so với cosine distance (Eq. 4). Kiểm tra cosine chọn khối có $D_{res}$ nhỏ nhất không.
2. So ba cách nối cho cùng khối: Identity, LS-T, cosine-T; đo CER trước distill.
3. Heal chỉ $T$ bằng distillation từ teacher 4B vài trăm bước, so với LoRA toàn decoder cùng số bước.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| $i^*=\arg\min_i h(L_i,L_{i+n})$ | chọn khối $n$ block có đầu ra gần đầu vào nhất |
| $T^*=(M^\top M)^{-1}M^\top(L_{i+n}-Y)$ | nghiệm LS đóng cho ánh xạ tuyến tính thay khối |
| Cosine objective | tối ưu số bằng Adam, tốt hơn LS (RP 90.9% vs 89.9% trên Llama-3-8B) |
| Fuse | $W_{down}\leftarrow W_{down}T$, không thêm tham số |
| Multi-LT | nhiều nhóm block, mỗi nhóm một $T$ |
| RP | % hiệu năng so với baseline |

## 11. Kết luận ngắn

ReplaceMe cho thấy một ma trận tuyến tính tính từ vài nghìn mẫu có thể bù phần lớn tổn thất khi bỏ 25% layer, không cần train; với nén cao hơn vẫn cần healing nhưng chỉ cần heal $T$. Điểm yếu: chọn khối không xét tới bước sửa và đánh giá thiên về trắc nghiệm.

Link gốc: https://arxiv.org/abs/2505.02819
