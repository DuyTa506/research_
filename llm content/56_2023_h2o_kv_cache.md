# H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models (2023)

**Tác giả:** Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, Beidi Chen.  
**Venue / arXiv:** arXiv:2306.14048 (v3); NeurIPS 2023 (theo trang đầu của paper).  
**Nguồn:** https://arxiv.org/abs/2306.14048  
**Chủ đề:** Chính sách loại bỏ (eviction) KV cache dựa trên điểm attention tích lũy ("Heavy Hitters") cộng token gần nhất; phân tích dưới góc độ submodular; tăng thông lượng khi batch lớn.  
**Vị trí trong lộ trình:** Đọc trước: 01, 14-15 (MQA/GQA), 44 (FlashAttention), 46 (PagedAttention), 55 (StreamingLLM). Bài này là phiên bản "thích nghi theo nội dung" của 55. Tiếp theo: 57 (FastV, tỉa token thị giác).  
**Phạm vi đọc:** bài dài ~24K từ vì phụ lục lý thuyết (submodular, chứng minh) rất dày. Ghi chú này tập trung vào quan sát, thuật toán, thực nghiệm và hàm ý hệ thống; phần chứng minh chỉ tóm lược.

> Đây là ghi chú học tập bằng tiếng Việt, không phải bản dịch. Ứng dụng cho dự án và các số "tự tính" là suy luận của chúng ta.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
4. [Phân tích chi phí: bộ nhớ và độ trễ](#4-phân-tích-chi-phí-bộ-nhớ-và-độ-trễ)
5. [Thí nghiệm và kết quả chính](#5-thí-nghiệm-và-kết-quả-chính)
6. [Phân tích / ablation](#6-phân-tích--ablation)
7. [Giới hạn](#7-giới-hạn)
8. [Liên hệ với các bài khác](#8-liên-hệ-với-các-bài-khác)
9. [Ứng dụng cho dự án (suy luận)](#9-ứng-dụng-cho-dự-án-suy-luận)
10. [Thí nghiệm gợi ý](#10-thí-nghiệm-gợi-ý)
11. [Tóm tắt công thức / khái niệm](#11-tóm-tắt-công-thức--khái-niệm)
12. [Kết luận ngắn](#12-kết-luận-ngắn)

---

## 1. Bối cảnh

**KV cache** lưu Key/Value của mọi token đã thấy để mỗi bước sinh không phải tính lại. Nhược điểm: kích thước tăng tuyến tính theo độ dài chuỗi và theo batch. Ví dụ paper nêu: mô hình 30B với batch 128 và độ dài 1024 cần **180 GB** KV cache (kiểm lại bằng OPT-30B: 48 lớp, hidden 7168, FP16 cho $2\cdot48\cdot7168\cdot2=1.38$ MB/token, nhân $128\cdot1024$ ra ~180 GB; tự kiểm tra).

Hai giai đoạn của sinh văn bản (paper):
- **Prompt phase / prefill:** xử lý toàn bộ prompt song song, tạo KV cache.
- **Token generation / decode:** sinh từng token, đọc và cập nhật KV cache.

Muốn giới hạn kích thước cache giống cache phần mềm/phần cứng (LRU, LFU, thuật toán tối ưu Belady) nhưng khó: loại nhầm KV quan trọng thì phá hỏng chất lượng vì sinh văn bản có phụ thuộc tuần tự. Các cách trước đó (Sparse Transformer, Reformer, low-rank, MQA) hoặc cần huấn luyện lại, hoặc áp thẳng vào LLM đã pre-train thì chất lượng rơi (Hình 1). Một ứng viên lý tưởng cần: (i) cache nhỏ, (ii) tỉ lệ miss thấp, (iii) chính sách loại rẻ.

## 2. Ý tưởng chính

1. **Attention thưa:** dù huấn luyện dày, ma trận attention lúc suy luận thưa hơn 95% ở hầu hết các lớp (ngưỡng = 1% của giá trị lớn nhất mỗi hàng). Gợi ý có thể giảm KV tới 20x.
2. **Heavy Hitters (H2):** tổng điểm attention tích lũy của mỗi token theo mọi bước tuân theo **luật lũy thừa**: một số ít token nhận phần lớn attention. Gọi chúng là H2. Xóa H2 làm chất lượng sụp đổ; H2 tương quan với các từ xuất hiện đồng thời nhiều trong dữ liệu.
3. **Thuật toán H2O:** cache giữ ngân sách $k$ gồm một nửa token "nặng" (điểm tích lũy cao) và một nửa token gần nhất. Mỗi bước, loại token có điểm tích lũy thấp nhất (ngoài vùng gần nhất).
4. Chỉ dùng thống kê cục bộ (tổng attention từ các token đã sinh), không cần biết tương lai, mà hiệu quả ngang với dùng thống kê toàn cục.
5. Cơ sở lý thuyết: coi đây là bài toán submodular động; tham lam gần tối ưu theo cận $(1-\alpha)(1-1/e)\max f - \beta$.

## 3. Phương pháp chi tiết

### 3.1 Bài toán có ngân sách cache

Cho $Q,K\in\mathbb R^{n\times d}$, ngân sách $k<n$. Tập chỉ số đang giữ trong cache khi dự đoán token $i$ là $S_i$, $|S_i|=k$, và mỗi bước chỉ thay đổi tối đa 1 phần tử ($|S_i\setminus S_{i-1}|\le1$). Attention chuẩn hóa chỉ trên các token còn lại:

$$
o_i = D_i^{-1}\exp\!\big(Q_{i,*}K_{S_i,*}^\top\big),\qquad D_i=\big(\exp(Q_{i,*}K_{S_i,*}^\top)-\mathbb 1_{[i]\setminus S_i}\big)\cdot \mathbf 1
$$

Mục tiêu: đầu ra gần như khi không giới hạn cache.

### 3.2 Chính sách loại H2O

Điểm của một tập token là tổng attention nhận được:

$$
F_{\text{score}}(T)=\sum_{s\in T}o_s,\qquad u=\arg\max_{v\in S_{i-1}\cup\{i\}}F_{\text{score}}\big(S_{i-1}\cup\{i\}\setminus\{v\}\big)
$$

nghĩa là loại token $u$ mà việc bỏ nó làm tổng điểm của phần còn lại lớn nhất, hay đơn giản **loại token có điểm tích lũy nhỏ nhất**.

Trong hiện thực: giữ $K$ heavy hitters và $K$ token gần nhất; mỗi bước decode: bỏ token cũ nhất trong $K$ cuối, loại một heavy hitter cho mỗi head, thêm token mới. **Mỗi head, mỗi lớp quyết định độc lập** (khác SpAtten, vốn cộng điểm qua head và lớp).

### 3.3 Pseudo-code

```
state per (layer, head): K_cache, V_cache (<= 2K entries), score[]  (điểm tích lũy)

prefill (độ dài n):
    tính attention A (n x n) song song
    score[j] = sum_i A[i,j]                       # tổng attention mỗi token nhận
    nếu n >= 2K: giữ K token có score cao nhất ∪ K token cuối
    ngược lại: giữ hết

decode (token mới t):
    a = softmax(q_t · K_cache^T)                   # attention trên cache
    out = a · V_cache
    score += a;  score[t] = 0 (khởi tạo)
    nếu cache đầy:
        # ứng viên loại: các token KHÔNG nằm trong K token gần nhất
        u = argmin_{j not in recent K} score[j]
        thay (K_cache[u], V_cache[u], score[u]) bằng (k_t, v_t, 0)
        # recent: dùng circular queue; vị trí token cũ nhất trong recent trở thành ứng viên heavy hitter
```

Ví dụ (Hình 3): ngân sách 3, sau bước giải mã thứ 4 KV của token thứ 3 (điểm tích lũy thấp nhất) bị loại và không truy cập lại được.

Chi tiết hệ thống: cài đặt trên FlexGen; bộ nhớ KV cấp phát trước, hàng đợi vòng cho K phần tử cuối, ghi đè thay vì hoán đổi bộ nhớ.

### 3.4 Lý thuyết (tóm lược)

Giả sử hàm attention là **submodular** (lợi ích biên giảm dần khi tập lớn hơn). Khi đó chọn tham lam top-$k$ theo điểm đạt xấp xỉ $(1-1/e)$ của tối ưu. Paper mở rộng sang khung "dynamic submodular" (Định nghĩa 4.1) vì tập thay đổi từng bước, và chứng minh (Định lý 4.4, mô tả không chính thức trong thân bài) rằng

$$
f(\tilde S_i)\ge(1-\alpha)\left(1-\tfrac1e\right)\max_{|S|=k}f(S)-\beta .
$$

Đây là lý giải vì sao tham lam có thể tốt (dựa trên giả định "nhẹ"); ghi chú này không kiểm tra chứng minh.

## 4. Phân tích chi phí: bộ nhớ và độ trễ

### 4.1 Công thức

$$
\boxed{\ \text{KV bytes}=2\cdot B\cdot T\cdot n_\ell\cdot n_{kv}\cdot d_h\cdot b\ }
$$

với $B$ batch, $T$ độ dài chuỗi, $n_\ell$ số lớp full-attention, $n_{kv}$ số KV head, $d_h$ chiều mỗi head, $b$ byte/phần tử. H2O với ngân sách tỉ lệ $r$ (ví dụ 20%) đưa $T$ thành $rT$ (nhưng điểm tích lũy thêm một mảng float mỗi head).

Thời gian decode ở batch $B$ (memory-bound):

$$
t_{\text{step}}\gtrsim\frac{P\,b_w+\text{KV bytes}}{\text{bandwidth}}
$$

Trọng số $P b_w$ chia sẻ cho cả batch (đọc một lần), còn KV tỉ lệ với $B$. Vì vậy:
- **Batch lớn / chuỗi dài:** KV chiếm ưu thế, giảm KV giúp thật sự (bài này: tới 29x thông lượng).
- **Batch 1 / chuỗi ngắn:** trọng số chiếm ưu thế, giảm KV gần như không giúp.

Chi phí thêm của H2O: cập nhật điểm $O(k)$ mỗi head mỗi bước và argmin (hoặc heap); cần **xác suất attention tường minh** của bước decode. Việc đó rẻ khi decode (một query), nhưng ở prefill phải tổng theo cột toàn bộ ma trận $n\times n$; điều này không tương thích tự nhiên với kernel FlashAttention vốn không trả ma trận attention (suy luận của ta; paper cài trên FlexGen).

### 4.2 Ví dụ số

**Ví dụ của paper:** OPT-30B, $B=128$, $T=1024$: 180 GB. Với 20% còn ~36 GB (tự tính).

**Cỡ dự án (tự tính, giả định):** 24 lớp, lai 3:1 (GatedDeltaNet:full) nên $n_\ell=6$ lớp full-attention; 2 KV head x 128 dim; FP16 nên 1.024 B/token/lớp; $T=2.500$ (giả định 1.000 prompt + 1.500 đầu ra), batch 1.

| Cấu hình | KV | So với trọng số FP16 1,6 GB |
|---|---|---|
| Dense, 6 lớp full-attention | 2.500 x 1.024 x 6 = 15,4 MB | ~1% |
| H2O 20% (500 token) | ~3,1 MB | ~0,2% |
| Nếu cả 24 lớp full-attention | 61,4 MB | ~4% |
| Nếu MHA 16 head x 128, 6 lớp | ~123 MB | ~8% |

Thời gian: với băng thông giả định 200 GB/s, đọc 1,6 GB mất 8 ms; đọc thêm 15 MB mất 0,08 ms (~1%). H2O tiết kiệm tối đa ~0,06 ms/token, cỡ nhiễu đo. Cộng thêm chi phí tính điểm thì có thể **chậm hơn** ở batch 1. Mô hình càng bị lượng tử hóa trọng số (bài 54) thì tỉ lệ KV càng lớn lên (0,66 GB thì KV 15 MB là ~2%), nhưng vẫn nhỏ.

**Các lớp GatedDeltaNet** không có KV cache tăng theo độ dài (chỉ có trạng thái cố định mỗi head, $O(d_kd_v)$) nên **H2O không áp dụng cho chúng**; ví dụ với 16 head, $d_k=d_v=128$, FP32: 16 x 128 x 128 x 4 B = 1 MB/lớp, 18 lớp ≈ 18,9 MB, cố định (tự tính, số head là giả định). Điều này nói lên rằng hybrid đã "tự nén" KV: chuỗi dài chỉ làm tăng bộ nhớ ở 6 lớp.

## 5. Thí nghiệm và kết quả chính

**Thiết lập:** OPT (6.7B tới 175B ở phụ lục), LLaMA-7B/13B/30B, GPT-NeoX-20B; 8 tác vụ: COPA, MathQA, OpenBookQA, PiQA, RTE, Winogrande (lm-eval-harness, 5-shot) và XSUM, CNN/Daily Mail (HELM, 0-shot, 1000 mẫu); thêm AlpacaEval, MT-bench ở phụ lục. Ngân sách KV 4%, 10%, 20%, 60% độ dài prompt. GPU: A100 80GB cho độ chính xác; T4 16GB và A100 cho thông lượng. Baseline: toàn bộ cache, "Local" (chỉ token gần nhất), Sparse Transformer (strided, fixed), Top-K, SpAtten, StreamingLLM.

**Kết luận chính (mục 5.1):** với ngân sách <20% (giảm >5x bộ nhớ KV), H2O cho kết quả ngang cache đầy ở phần lớn tác vụ, và vượt xa "Local".

**Bảng 1 (20% KV; paper không ghi rõ mô hình trong chú thích, ta chỉ trích 4 tác vụ):**

| Phương pháp | PiQA | COPA | OpenBookQA | Winogrande |
|---|---|---|---|---|
| Full | 80.09 | 81.00 | 44.80 | 71.51 |
| Local | 57.94 | 56.00 | 28.40 | 51.30 |
| H2O | 79.22 | 85.00 | 43.80 | 71.67 |

(Các hàng 0-shot và 1-shot Full trong bảng bị lẫn cột trong bản text nên ta không trích; chỉ dùng hàng Full, Local, H2O như trên, những số này khớp ngữ cảnh. Thận trọng khi đối chiếu bản gốc.)

**Bảng 2 (OPT-30B, 20% KV; H2 cải thiện các phương pháp thưa khác):**

| | COPA | OpenBookQA | PiQA | Winogrande |
|---|---|---|---|---|
| Full | 85.00 | 43.20 | 78.51 | 70.24 |
| Local (không H2) | 48.00 | 25.20 | 55.82 | 49.17 |
| Local + H2 | 84.00 | 43.00 | 78.45 | 69.06 |
| Strided (không H2) | 50.00 | 24.60 | 56.20 | 47.59 |
| Strided + H2 | 83.00 | 42.60 | 78.24 | 69.61 |
| Fixed (không H2) | 61.00 | 23.80 | 58.60 | 49.88 |
| Fixed + H2 | 76.00 | 41.40 | 77.80 | 64.96 |

Phụ lục C.5: Top-K cũng cải thiện thêm khi kết hợp H2 (ví dụ COPA 80.00 -> 82.00).

**Thông lượng (token/s, T4 16GB; mô hình 6.7B / 30B; ngoặc là batch hiệu dụng và mức bộ nhớ thấp nhất cần):**

| Chuỗi (prompt + sinh) | 512+32 | 512+512 | 512+1024 |
|---|---|---|---|
| FlexGen 6.7B | 20.2 (2, G) | 16.8 (1, G) | 16.9 (1, G) |
| H2O (20%) 6.7B | 35.1 (4, G) | 51.7 (4, G) | 52.1 (4, G) |
| FlexGen 30B | 8.1 (144, C) | 8.5 (80, C) | 7.1 (48, C) |
| H2O (20%) 30B | 12.7 (728, C) | 18.83 (416, C) | 13.82 (264, C) |

(Accelerate và DeepSpeed ở Bảng 3 thấp hơn rất nhiều: ví dụ 30B 0,6 token/s.) Tổng hợp: tới 29x so với DeepSpeed và Accelerate, 3x so với FlexGen. Trên XSUM thực (Bảng 4): 6.7B FlexGen 10.80 -> H2O 30.40; 30B FlexGen 3.29 -> H2O 6.70.

**A100 (Bảng 5):**

| Cấu hình | FlexGen | H2O (20%) |
|---|---|---|
| 30B, 7000+1024, batch 1, độ trễ | 57.0 s | 50.4 s |
| 13B, 5000+5000, batch 4, độ trễ | 214.2 s | 155.4 s |
| 6.7B, 2048+2048, batch 24, độ trễ | 99.5 s | 53.5 s |
| 6.7B, 2048+2048, batch 24, thông lượng | 494.1 | 918.9 token/s |
| 6.7B, batch 64, thông lượng | OOM | 1161.0 token/s |

Điểm đáng chú ý cho **batch 1**: ở 30B, 7000+1024, batch 1 độ trễ chỉ giảm từ 57,0 s xuống 50,4 s (~1,13x), vì mô hình 30B ở batch 1 bị trọng số chi phối. Giảm độ trễ 1,9x chỉ ở batch lớn hơn (6.7B, batch 24: 99,5 s xuống 53,5 s). Dòng "với cùng batch size, giảm độ trễ tới 1,9x" trong abstract tương ứng ca này.

**Kết hợp lượng tử hóa (Bảng 6, OPT-30B):** 4-bit toàn bộ (COPA 84.00, OBQA 43.28, PiQA 78.67) và H2O + 4-bit (84.00, 43.20, 78.80), tức gần như không hỏng thêm. Bảng 7: lượng tử hóa trọng số 4-bit của FlexGen cho phép batch rất lớn (70 so với 4), thông lượng 6.7B từ 35.1 lên 50.5 (512+32) và 52.1 lên 62.3 (512+1024), dưới 2x dù batch tăng ~20 lần vì kernel lượng tử chưa tối ưu.

**Dòng chảy vô hạn (Hình 5):** H2O đạt 4 triệu token trên PG-19 với perplexity thấp hơn StreamingLLM ở nhiều cỡ cache (kết quả trên hình, số cụ thể không đọc được từ text).

## 6. Phân tích / ablation

- **H2 và recent đều cần (Bảng 9, Full / chỉ Local / chỉ H2 / Local+H2):**

| | OPT-13B | OPT-30B |
|---|---|---|
| PiQA | 77.37 / 54.62 / 76.12 / 77.26 | 78.51 / 55.82 / 67.25 / 78.45 |
| OpenBookQA | 41.40 / 25.60 / 30.40 / 41.20 | 43.20 / 25.20 / 26.60 / 43.00 |
| MathQA | 26.67 / 22.04 / 23.82 / 26.93 | 26.23 / 20.87 / 21.98 / 26.87 |
| Winogrande | 68.59 / 49.96 / 51.85 / 67.32 | 70.24 / 49.17 / 47.36 / 69.06 |

Chỉ H2 hay chỉ Local đều tụt 2,85% tới 22,75%; cả hai mới giữ hiệu năng. H2 nhìn chung đóng góp nhiều hơn.
- **Số shot (Bảng 10):** ở 5-shot và 10-shot, H2O cách bản đầy dưới 1,00%; "Local" mất tới 37%. 0/1-shot cần ngân sách cao hơn (30-40%) vì chuỗi ngắn (100-300 token).
- **So với SpAtten (Bảng 11, OPT-30B):** COPA 82.00 (SpAtten) vs 84.00 (H2O), OBQA 41.90 vs 43.00, PiQA 77.06 vs 78.45. Khác biệt: H2O quyết định độc lập theo head/lớp, thêm cả token gần nhất.
- **So với StreamingLLM (Phụ lục C.4):** StreamingLLM hỏng khi thông tin then chốt nằm giữa đầu vào (QA nhiều tài liệu, tóm tắt), vì chỉ giữ đầu + gần nhất; H2O giữ được nhờ điểm attention.
- **Điểm trung bình thay vì tích lũy:** thử dùng attention trung bình để chọn, kết quả tệ hơn (Phụ lục B.2). Tác giả lưu ý tổng tích lũy thiên vị token cũ (có nhiều bước để cộng dồn), nên đã thử trung bình, nhưng trung bình cho kết quả kém hơn.
- **Heavy hitter xuất hiện nhiều ở đầu câu** (Phụ lục B.2), gợi ý token đầu quan trọng (gần với sink của bài 55).
- **Đa dạng văn bản:** H2O cho văn bản ít lặp từ hơn (định tính, Phụ lục C.1).
- **H2 trong MLP (C.10):** tần suất kích hoạt neuron cũng theo luật lũy thừa; xóa H2 làm hỏng nhưng phục hồi được chỉ với 1% dữ liệu huấn luyện; H2 xuất hiện sớm khi huấn luyện ("early-bird"). Một hướng gợi ý của tác giả cho việc tiết kiệm tham số.

## 7. Giới hạn

**Do tác giả nêu:**
- Thiên lệch của điểm tích lũy theo token cũ (đã thử trung bình nhưng kém hơn).
- Phần lớn tham số nằm ở MLP; tác giả nói hướng tương lai là khai thác H2 trong MLP cho chính sách offloading.
- Token bị loại không quay lại được.

**Do chúng ta nhận thấy:**
- Độ chính xác đánh giá chủ yếu bằng trắc nghiệm/few-shot ngắn và tóm tắt trên mô hình OPT/LLaMA đời đầu; ít kiểm tra tác vụ cần truy xuất chính xác thông tin ở khoảng cách xa (ngoại trừ vài thử nghiệm ở phụ lục).
- Thông lượng được đo chủ yếu so với các hệ offloading (FlexGen, DeepSpeed, Accelerate) trên T4; các hệ hiện đại (PagedAttention, bài 46) và GQA/MQA vốn đã giảm KV, nên lợi ích tương đối nhỏ hơn với mô hình GQA (không đo trong paper).
- Cần xác suất attention tường minh ở mỗi bước; khó tích hợp với kernel hợp nhất kiểu FlashAttention (suy luận); cần xử lý riêng với GQA (cộng điểm của nhiều query head vào cùng một KV head).
- Lợi ích tính toán thực chủ yếu ở batch lớn; với batch 1 rất nhỏ (xem 4.2).
- Định lý phụ thuộc giả định submodular "nhẹ" mà không được kiểm chứng trực tiếp trên mọi mô hình.

## 8. Liên hệ với các bài khác

- **55 StreamingLLM:** cùng bài toán, chiến lược cố định (đầu + gần nhất) so với chiến lược động theo nội dung (gần nhất + heavy hitters). H2O thừa nhận cảm hứng từ StreamingLLM cho thí nghiệm vô hạn.
- **14 MQA, 15 GQA, 16 MLA:** giảm KV theo kiến trúc (huấn luyện từ đầu). H2O giảm KV theo thời gian suy luận; hai hướng cộng dồn nhưng mô hình dùng GQA/MLA có KV đã nhỏ nên H2O ít giá trị hơn.
- **17 Longformer:** sliding window, tĩnh.
- **44-46 FlashAttention, PagedAttention:** FlashAttention không trả attention; PagedAttention quản lý KV theo trang, có thể kết hợp với chính sách loại.
- **57 FastV:** tỉa token thị giác dựa trên attention ở lớp sớm, ý tưởng gần với H2O nhưng cho prefill VLM.
- **54 AWQ:** kết hợp được (Bảng 6): lượng tử hóa trọng số + giảm KV.
- **24 Gated DeltaNet, 18, 19:** là cách tiếp cận kiến trúc để có bộ nhớ cố định (trạng thái nén), thay vì loại bỏ token rời rạc.

## 9. Ứng dụng cho dự án (suy luận)

1. **Ở batch 1, trang ~1.500 token: ít liên quan trực tiếp.** KV của 6 lớp full-attention chỉ ~15 MB (mục 4.2), dưới 1-2% byte đọc mỗi token. H2O không cải thiện độ trễ đáng kể, còn thêm chi phí tính điểm và có nguy cơ hại chất lượng.
2. **OPT/LLaMA-style H2O đáng ngại với OCR.** OCR đọc ảnh theo thứ tự: mỗi token đầu ra cần các patch ảnh ở vị trí tương ứng, mà những vùng "chưa đọc tới" lại có điểm attention tích lũy thấp, nên chính sách tích lũy có thể loại nhầm chính nơi sắp cần (giả thuyết, cần thử). Paper cũng nhận thiên lệch giữa token cũ và mới. Nếu muốn nén token thị giác, phương pháp theo bài 57 (tỉa trước khi decode theo độ quan trọng) phù hợp hơn.
3. **Hàm ý cho kiến trúc lai:** chỉ 6/24 lớp có KV tăng theo độ dài; các lớp GatedDeltaNet giữ trạng thái cố định. Khi cắt tỉa lớp, nên tính ngân sách bộ nhớ theo số lớp full-attention còn lại; cắt lớp full-attention vừa giảm tham số vừa giảm KV, còn cắt lớp GatedDeltaNet không đổi KV.
4. **Heavy hitters như công cụ chẩn đoán cắt tỉa.** Luật lũy thừa của attention (một vài token nhận đa số) và của kích hoạt MLP (C.10: một số neuron kích hoạt gần 100% token, phần lớn hiếm khi) gợi ý phân bố độ quan trọng rất lệch. Cắt tỉa neuron MLP theo tần suất kích hoạt có thể xóa nhầm "heavy hitter neurons", điều có thể giải thích một phần quan sát của bạn rằng cắt neuron MLP phục hồi kém hơn cắt lớp (giả thuyết; paper chỉ nói xóa H2 làm hỏng nhưng phục hồi được bằng 1% dữ liệu trên mô hình của họ, không phải mô hình của bạn). Nên đo tần suất kích hoạt trước khi cắt.
5. **Chưng cất:** điểm attention tích lũy cho biết token nào quan trọng; có thể dùng làm tín hiệu khớp giữa giáo viên 4B và học sinh (gợi ý, không có trong paper).
6. **Khi nào H2O sẽ có giá trị:** nếu triển khai server batch lớn (nhiều trang song song) hoặc xử lý tài liệu dài hàng chục nghìn token, KV chiếm ưu thế (mục 4.1). Ở 8 GB, batch 8 nhân KV lên 8 lần (~123 MB theo bảng trên), vẫn nhỏ so với 1,6 GB.

## 10. Thí nghiệm gợi ý

1. Đo phân bố: ghi điểm attention tích lũy cho từng token (kể cả token ảnh) ở 6 lớp full-attention trên 100 trang; vẽ phân phối (luật lũy thừa?) và xem token ảnh nào là heavy hitter. Kiểm tra nhận định "vùng chưa đọc có điểm thấp".
2. Mô phỏng H2O ở ngân sách 50% và 20% chỉ trên các lớp full-attention (mask KV theo điểm), đo CER của mô hình 0.8B và 0.3B. So với việc giữ nguyên mà tỉa token thị giác theo kiểu FastV.
3. Đo tần suất kích hoạt neuron MLP (như C.10) để xếp hạng neuron cho cắt tỉa; so sánh cắt neuron ít kích hoạt với cắt ngẫu nhiên về D_lin / D_res.

## 11. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Heavy Hitter | Token có tổng attention tích lũy cao; phân bố luật lũy thừa |
| Chính sách H2O | Giữ $k/2$ heavy hitters + $k/2$ gần nhất; loại điểm thấp nhất |
| Điểm | $\text{score}_j=\sum_i A_{ij}$ (theo từng head, từng lớp) |
| Sparsity | Attention thưa >95% (ngưỡng 1% max mỗi hàng) |
| KV cache | $2BT n_\ell n_{kv} d_h b$ byte |
| Đảm bảo | $f(\tilde S)\ge(1-\alpha)(1-1/e)\max f-\beta$ (dưới giả định submodular) |
| Kết quả | 5x ít KV mà giữ chất lượng; thông lượng tới 29x/29x/3x; trễ tới 1,9x (cùng batch) |
| Với batch 1 | Lợi ích nhỏ: 57,0 s -> 50,4 s ở 30B batch 1 |

## 12. Kết luận ngắn

H2O chỉ ra rằng attention trong LLM đủ thưa và lệch để giữ khoảng 20% KV mà vẫn gần như không mất chất lượng, nhờ kết hợp token gần nhất và token "nặng" theo điểm tích lũy, với lý thuyết tham lam submodular đứng sau. Lợi ích hệ thống chủ yếu xuất hiện khi batch lớn hoặc chuỗi dài, nơi KV chiếm ưu thế. Với dự án batch 1, trang OCR ngắn và kiến trúc lai chỉ có vài lớp full-attention, H2O gần như không cần; điều đáng mang theo là cách nhìn "độ quan trọng phân bố lệch" để thiết kế cắt tỉa và đánh giá, và cảnh giác khi áp dụng chính sách theo attention cho token thị giác theo thứ tự đọc.

Nguồn gốc: https://arxiv.org/abs/2306.14048
