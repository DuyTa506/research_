# MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models (2024)

**Tác giả:** Gongfan Fang, Hongxu Yin, Saurav Muralidharan, Greg Heinrich, Jeff Pool, Jan Kautz, Pavlo Molchanov, Xinchao Wang (NVIDIA; National University of Singapore).  
**Venue / arXiv:** arXiv:2409.17481 (26/09/2024, bản v2 12/2024). NeurIPS 2024 (ghi ở trang đầu bản text).  
**Nguồn:** https://arxiv.org/abs/2409.17481  
**Chủ đề:** Học mask thưa N:M (cụ thể 2:4) cho LLM bằng huấn luyện đầu cuối với Gumbel-Softmax, **giữ nguyên (đóng băng) trọng số**; mask prior từ phương pháp one-shot; chuyển giao mask sang miền/tác vụ mới.  
**Vị trí trong lộ trình:** Đọc sau 18 (NVIDIA 2:4 sparsity), 23 (SparseGPT), 24 (Wanda) và 16/17 (Movement pruning, CoFi: học mask bằng gradient). Bài này là phiên bản "học bằng loss thật" thay cho "chấm điểm cục bộ" ở chuẩn N:M. Bài kế: 31 (ShortGPT) mở nhóm cắt theo chiều sâu.

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Số liệu lấy từ paper. Phần "Ứng dụng cho dự án" là suy luận của chúng ta; bài này thuộc loại semi-structured nên liên quan gián tiếp.

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

**N:M sparsity (thưa bán cấu trúc):** trong mỗi nhóm $M$ trọng số liên tiếp, chỉ giữ $N$ giá trị khác 0 (ví dụ 2:4: mỗi 4 trọng số giữ 2). GPU NVIDIA đời mới có sparse tensor core tăng tốc pattern này (bài 18). Khó khăn với LLM: số tổ hợp mask khổng lồ (paper nêu: LLaMA-2 7B toàn bộ 2:4 có 1.6 tỷ mask 2:4 phải chọn cho các lớp dày). Phương pháp hiện hành (SparseGPT, Wanda) dùng ít mẫu calibration và điểm quan trọng thủ công. Paper chỉ ra hai vấn đề:

1. **Calibration nhỏ không đại diện** cho kiến thức của LLM đã huấn luyện trên dữ liệu đa dạng; tăng calibration vượt 256 mẫu không cải thiện thêm (theo thí nghiệm của họ).
2. **Điểm quan trọng thủ công là proxy có sai số** so với mức thay đổi thực sự của hàm mất mát (độ lớn, gradient, Hessian đều xấp xỉ).

Khái niệm: **Gumbel-Softmax** là kỹ thuật lấy mẫu từ phân phối rời rạc (categorical) theo cách khả vi, bằng cách cộng nhiễu Gumbel vào logit và dùng softmax có nhiệt độ thay cho argmax; nhờ đó gradient chảy được về xác suất. **Frozen weights:** trọng số không được cập nhật, chỉ học mask.

## 2. Ý tưởng chính

Với 2:4 chỉ có $\binom42=6$ mask khả dĩ cho mỗi nhóm 4 trọng số. Gán mỗi nhóm 6 logit học được (một phân phối categorical). Trong lúc huấn luyện, lấy mẫu một mask "mềm" bằng Gumbel-Softmax (trung bình có trọng số của 6 mask ứng viên), nhân vào trọng số đóng băng, tính loss ngôn ngữ trên dữ liệu lớn, backprop về các logit. Sau huấn luyện, mỗi nhóm chọn mask có logit lớn nhất. Có thể khởi tạo logit từ mask do Magnitude/SparseGPT/Wanda ("mask prior") để hội tụ nhanh hơn. Kết quả: LLaMA-2 7B 2:4 đạt PPL 6.72 (SparseGPT 10.42, dense 5.12) mà không sửa trọng số nào.

## 3. Phương pháp chi tiết

### 3.1 Bài toán chọn mask

Tập ứng viên cho 2:4 (6 mask):

$$\mathcal S^{2:4}=\{[1,1,0,0],[1,0,1,0],[1,0,0,1],[0,1,0,1],[0,1,1,0],[0,0,1,1]\}.$$

Mục tiêu (Eq. 3): chọn mask cho mọi khối $W_i$ để cực tiểu loss ngôn ngữ:

$$\{\mathcal M_i^*\}=\arg\min_{\mathcal M_i\in\mathcal S^{2:4}}\ \mathbb E_{x\sim p(x)}\Big[\mathcal L_{LM}\big(x;\{W_i\odot\mathcal M_i\}\big)\Big].$$

Nhìn theo xác suất (Eq. 4): học phân phối $p(\mathcal M_i)$ sao cho **kỳ vọng loss khi lấy mẫu mask** là nhỏ nhất.

### 3.2 Lấy mẫu khả vi

Gumbel-Max: $y=\mathrm{onehot}(\arg\max_i[\log p_i+g_i])$, $g_i=-\log(-\log\epsilon_i)$, $\epsilon_i\sim U(0,1)$. Thay argmax bằng softmax nhiệt độ $\tau$ để khả vi (Eq. 6):

$$\tilde y_i=\frac{\exp((\log p_i+g_i)/\tau)}{\sum_j\exp((\log p_j+g_j)/\tau)}.$$

Mask mềm là trung bình có trọng số của các ứng viên (Eq. 7): $\widetilde{\mathcal M}=\sum_{i=1}^{|\mathcal S|}\tilde y_i\,\hat{\mathcal M}_i$. Khi $\tau\to0$, $\tilde y$ tiến về one-hot.

**Tham số hoá:** học logit $\pi_i$ với **hệ số tỉ lệ** $\kappa$: $p_i=\exp(\pi_i\kappa)/\sum_j\exp(\pi_j\kappa)$. $\kappa$ cân bằng độ lớn logit so với nhiễu Gumbel, tức điều khiển mức ngẫu nhiên khi lấy mẫu.

### 3.3 Chính quy hoá trọng số còn lại

Khi mask loại bỏ trọng số, gradient truyền qua layer có thể nhỏ đi (trọng số còn lại có độ lớn nhỏ), cản trở học mask. Thêm số hạng thưởng độ lớn (Eq. 8):

$$\min_{\{p_\pi(\mathcal M_i)\}}\ \mathbb E_{x,\widetilde{\mathcal M}_i\sim p_\pi}\Big[\mathcal L_{LM}\big(x;\{W_i\odot\widetilde{\mathcal M}_i\}\big)-\lambda\sum_i\|W_i\odot\widetilde{\mathcal M}_i\|_2^2\Big]$$

### 3.4 Mask prior (chuyển giao sparsity)

Cho mask có sẵn $\mathcal M_0$ (từ magnitude, SparseGPT, Wanda hoặc từ quá trình học khác). Độ tương tự với từng ứng viên (Eq. 9, 10):

$$\mathrm{sim}(\mathcal M_0,\hat{\mathcal M}_i)=\mathcal M_0\hat{\mathcal M}_i^\top-N/2,\qquad \pi_i'=\pi_i+\sigma(\pi)\cdot\mathrm{sim}(\mathcal M_0,\hat{\mathcal M}_i)\cdot\alpha$$

($\sigma(\pi)$ là độ lệch chuẩn của logit; $\alpha$ là độ mạnh của prior; $\alpha=0$ là học thuần không prior). Mask ứng viên giống mask prior được tăng xác suất ban đầu.

```text
MaskLLM(W frozen, D, prior_mask M0 optional, steps=2000):
    S = 6 candidate 2:4 masks (matrix 6 x 4)
    for each group of 4 consecutive weights: logits pi ~ N(0, 0.01)   # 6 logits per group
    if M0: pi += std(pi) * (M0 @ S^T - 2/2) * alpha                    # alpha=3 (SparseGPT prior)
    for step in 1..steps:
        x ~ data
        for each group: g ~ Gumbel; y = softmax((pi*kappa + g)/tau)
        M_soft = y @ S                                  # (4,)  mask mềm
        loss = L_LM(x; W * M_soft) - lambda * ||W * M_soft||^2
        update pi (AdamW)          # kappa: 1e2 -> 5e2 ;  tau: 4 -> 0.05
    final mask of each group = S[argmax pi]
```

Với miền/tác vụ mới: khởi tạo từ mask chung đã học (làm prior) rồi học tiếp vài trăm bước trên dữ liệu miền đó (**transfer learning of sparsity**).

## 4. Thí nghiệm và kết quả chính

**Mô hình:** LLaMA-2 7B, 13B, Nemotron-4 15B, và hai mô hình nội bộ GPT-3 843M, GPT-3 2B (huấn luyện bằng Megatron); LLaMA-3 8B ở phụ lục. **Cài đặt:** học logit **2000 bước**, batch toàn cục 256 (tức 512k mẫu), trọng số đóng băng; AdamW lr 5e-4 (GPT-3: 1e-3), wd 0.1; khởi tạo logit $\mathcal N(0,0.01)$; $\kappa$ tăng 1e2 đến 5e2; $\tau$ 4 xuống 0.05; prior SparseGPT với $\alpha=3$; hệ số chính quy 1e-5. Dữ liệu LLaMA-2: tập pha trộn 69 miền (CUDA, VHDL, Reddit, ...). **Đánh giá:** PPL WikiText-2, 7 tác vụ zero-shot (HellaSwag, RACE, PIQA, WinoGrande, ARC-E, ARC-C, OBQA). Baseline one-shot calibrate bằng C4 (SparseGPT có cập nhật trọng số).

**2:4 với trọng số đóng băng (Table 1, trích):**

| Mô hình | Phương pháp | PPL Wikitext | Zero-shot trung bình |
|---|---|---|---|
| LLaMA-2 7B | Dense | 5.12 | 57.16 |
| | Magnitude | 54.71 | 46.19 |
| | SparseGPT | 10.42 | 47.16 |
| | Wanda | 11.29 | 45.98 |
| | **MaskLLM** | **6.72** | **52.09** |
| LLaMA-2 13B | Dense | 4.57 | 59.60 |
| | SparseGPT | 8.20 | 52.32 |
| | Wanda | 8.47 | 50.61 |
| | **MaskLLM** | **5.85** | **56.74** |
| Nemotron-4 15B | Dense | 5.78 | 61.47 |
| | SparseGPT | 13.38 | 51.13 |
| | Wanda | 25.05 | 45.82 |
| | **MaskLLM** | **7.31** | **56.74** |
| GPT3 2B | Dense | 9.35 | 48.90 |
| | SparseGPT | 22.14 | 39.54 |
| | **MaskLLM** | **11.42** | **44.82** |
| GPT3 843M | Dense | 12.42 | 41.92 |
| | SparseGPT | 38.78 | 35.22 |
| | **MaskLLM** | **15.39** | **39.10** |

**LLaMA-3 8B (Table 12):** dense 5.76; SparseGPT 17.64; Wanda 23.40; Magnitude 2.61E+03; MaskLLM **8.50** (prior SparseGPT, học trên C4). **LLaMA-2 13B so với các phương pháp khác (Table 13):** SparseGPT 8.32, Wanda 8.27, ADMM-Iter 7.78, GBLM 8.80, RIA 8.41, Pruner-Zero 7.41, MaskLLM 5.85 (số lấy từ bài gốc của từng phương pháp).

**Mask riêng cho miền (Table 4, GPT3-2B, PPL trung bình 8 miền):** dense 7.42; SparseGPT 18.80; SparseGPT-Update 15.36; Wanda 23.36; **MaskLLM 7.39** (ngang dense, "lossless"). LLaMA2-7B 8 miền: dense 4.80; SparseGPT 6.64; SparseGPT-Update 6.32; Wanda 8.04; MaskLLM 4.90.

**Chuyển giao (Table 5, GPT-3 2B):** mask chung 10.61, mask học từ đầu cho miền 7.51, **mask chuyển giao 7.39** (dense 7.42).

**Lưu trữ và tốc độ (Table 6, LLaMA-2 7B, A6000, batch 1):** fine-tune cả mô hình cho mỗi tác vụ cần 16 bit/tham số; mask học được cần **0.65 bit/tham số** ($\log_2 6/4\approx0.646$, mã hoá số học; giảm 25 lần), bộ nhớ mô hình 73%, nhanh **1.4×**. Phụ lục (Table 16, TensorRT-LLM): LLaMA-2 7B tốc độ token/s/GPU dense so với 2:4: 61.74 so với 86.92 (1.41×), 59.18 so với 82.11 (1.39×), 57.55 so với 78.81 (1.37×), 55.40 so với 75.10 (1.36×); LLaMA-2 13B 1.50× đến 1.57×.

**Chi phí huấn luyện:** 64 GPU A100 với tensor parallel 8; LLaMA-2 7B 1,280 GPU-giờ; LLaMA-2 13B 2,304 GPU-giờ; Nemotron-4 2,304 GPU-giờ.

## 5. Phân tích / ablation

- **Mở rộng theo lượng dữ liệu (Fig. 4, LLaMA-2 7B):** SparseGPT bão hoà sau 256 mẫu calibration; MaskLLM cải thiện liên tục tới 512k mẫu. MaskLLM cần 128 mẫu để dựng prior, vượt SparseGPT sau khoảng 1280 mẫu (5 bước cập nhật với batch 256). Nếu chỉ 1 đến 2 bước thì không bằng one-shot.
- **Mask prior (Table 2, PPL Wikitext, LLaMA-2 7B):** prior magnitude 54.71 sang mask học 6.77; SparseGPT 10.46 sang **6.72**; Wanda 11.29 sang 6.80; không prior: **9.12**. Mô hình nhỏ hơn: GPT-3 843M không prior 18.62 so với SparseGPT-prior 15.39; GPT-3 2B 14.31 so với 11.59. Tức học thuần vẫn thắng baseline, nhưng prior rẻ giúp nhiều khi chỉ có 2000 bước.
- **Tính ngẫu nhiên khi lấy mẫu (Fig. 5, Table 9 trên GPT-3 843M):** $\kappa$ quá nhỏ (nhiễu lấn át) hội tụ chậm (1 đến 5: PPL 5.97E+06); $\kappa$ quá lớn (1e5 đến 5e5) không khám phá (PPL 24.81); chọn 1e2 đến 5e2 (15.39). Nhiệt độ $\tau$ ban đầu 1: 17.52; 2: 16.69; **4: 15.39**; 10: 15.68. Độ mạnh prior $\alpha$: 0: 18.62; 1: 23.65; 2: 15.59; **3: 15.39**; 5: 15.48.
- **Chính quy hoá độ lớn (Table 3, GPT-3 2B):** mask-only 11.59 sang 11.42 (có reg); miền con 7.61 sang 7.39; fine-tune sau cắt 10.21 sang 9.96. Phụ lục F: reg làm gradient lớn hơn; norm trọng số mô hình thưa tăng 1.36 đến 1.41 lần so với dense; mask học được có thể chọn trọng số nhỏ hơn magnitude (chuẩn L1 thấp hơn khoảng 10%).
- **Dữ liệu học mask:** học trên C4 (Table 11) chỉ kém dữ liệu pha trộn một chút: LLaMA-2 7B 6.79 so với 6.72 (SparseGPT 10.42 so với 9.88).
- **Giữ layer nhạy cảm dày (Table 15, GPT-3 2B, PPL Story/OpenWeb):** full sparsity 15.58/13.84; bỏ qua layer cuối 15.18/13.61; bỏ 4 layer cuối 15.07/13.24; 8 layer cuối 14.95/12.92 (dense 14.76/12.05).
- **ViT-B/16 (phụ lục J, ImageNet-1K top-1):** dense 79.15; Magnitude 65.92; Wanda 63.28; SparseGPT 71.52; MaskLLM 1 epoch 76.23; **20 epoch 79.46** (+0.31 so với dense, không cập nhật trọng số). Tác giả coi là dấu hiệu của hiện tượng Lottery Ticket (bài 08) trong ViT.

## 6. Giới hạn

**Do tác giả nêu:** huấn luyện mask tốn tài nguyên hơn nhiều phương pháp one-shot (có thể tạo mask hiệu quả); cải thiện hiệu quả huấn luyện là việc cần làm (phụ lục K).

**Do ta nhận xét:**
- Chi phí là hàng nghìn GPU-giờ A100 (1,280 cho 7B), trong khi SparseGPT/Wanda chỉ vài phút; số "6.72" cần so với chi phí này. Bộ nhớ: 6 logit cho mỗi 4 trọng số (1.5 logit mỗi trọng số) cộng trạng thái AdamW, nên nặng hơn nhiều so với tham số đóng băng (ước tính của ta, không phải số của paper).
- Chỉ N:M (2:4); tăng tốc phụ thuộc phần cứng có sparse tensor core và cài đặt (1.36 đến 1.57× trên A6000 ở batch 1, không phải mức giảm 2×); không giảm số tham số một cách thực chất.
- Khác biệt huấn luyện/suy luận: khi huấn luyện dùng mask mềm (trung bình có trọng số), khi suy luận dùng mask cứng argmax; lệch này được kiểm soát bằng $\tau\to0.05$ và $\kappa$ tăng nhưng không có thí nghiệm riêng đo độ lệch (nhận xét của ta).
- So sánh với SparseGPT/Wanda dùng dữ liệu rất khác nhau (512k mẫu pha trộn miền so với 128 đến 256 mẫu); tác giả có thêm thí nghiệm học 1280 mẫu và C4 để giảm sự khác biệt này.
- Đánh giá bằng PPL và zero-shot; không đánh giá tác vụ sinh dài hay tác vụ nghiệp vụ như OCR.

## 7. Liên hệ với các bài khác

- **18 (2:4 sparsity):** nền tảng phần cứng/định dạng; MaskLLM học mask 2:4 tốt hơn so với chọn theo độ lớn rồi fine-tune.
- **23, 24 (SparseGPT, Wanda):** baseline; cũng là nguồn prior. MaskLLM chứng minh dùng chúng làm khởi tạo vẫn có ích.
- **16 (Movement pruning), 17 (CoFi):** cũng học mask bằng gradient với loss thật; CoFi học mask cho cấu trúc (head, FFN, layer) bằng hard-concrete; MaskLLM học mask N:M khả vi bằng Gumbel-Softmax và đóng băng trọng số.
- **08, 09 (Lottery Ticket, linear mode connectivity):** kết quả ViT "lossless chỉ bằng mask" liên hệ với giả thuyết mạng con.
- **20 (DepGraph), 26 (LLM-Pruner):** cùng nhóm Fang/Wang; MaskLLM là hướng semi-structured học được thay cho structured.
- **39 (Minitron), 44 (Nemotron-H):** NVIDIA; Nemotron-4 15B được dùng như một mô hình thí nghiệm ở đây.
- **43 (Pruning vs scratch):** hỏi lại "nén dựa trên mô hình có sẵn so với huấn luyện lại", bài này cho thấy có thể "chỉ học mask" mà không đổi trọng số.

## 8. Ứng dụng cho dự án (suy luận)

- **Trực tiếp: ít liên quan.** Mục tiêu của dự án là cắt cấu trúc (0.8B xuống 0.3B/0.1B) để chạy trên GPU $\le$ 8 GB với batch-1. 2:4 giữ nguyên hình dạng ma trận, chỉ cho tối đa khoảng 2× trên các linear layer và ghi nhận thực tế khoảng 1.4× với 73% bộ nhớ (cho 7B); không đạt hệ số nén 2.7× hay 8× của dự án, và phụ thuộc vào GPU có sparse tensor core. Có thể là bước *cộng thêm* sau khi đã cắt cấu trúc nếu GPU hỗ trợ.
- **Ý tưởng chuyển được (gián tiếp):**
  1. **Chọn bằng loss thật thay vì proxy cục bộ.** Kết quả 6.72 so với 10.42 (SparseGPT có cập nhật trọng số) cho thấy chọn mask bằng loss đầu cuối thắng cả phương pháp chấm điểm có bù cục bộ trong thiết lập N:M. Đây là gợi ý (không phải chứng minh) rằng xếp hạng cấu trúc theo $D_{res}$ cục bộ (kiểu 25, 27, 28) có thể bị thắng bởi lựa chọn học bằng distillation loss. Chú ý chi phí và khác biệt dữ liệu như ở mục 6.
  2. **Mask prior từ điểm rẻ** (magnitude, Wanda-sp, FLAP, ZipLM) làm khởi tạo cho cổng học được của neuron MLP / head / block: cổng $z\in\{0,1\}$ với logit khởi tạo từ điểm rẻ, huấn luyện bằng Gumbel-Softmax hoặc hard-concrete (CoFi, bài 17) cộng loss distillation từ thầy 4B. Bảng 2 cho thấy prior làm cải thiện lớn khi ngân sách bước ít (9.12 xuống 6.72).
  3. **Chính quy hoá giữ độ lớn của phần còn lại** để gradient không tắt: nếu học cổng với trọng số đóng băng thì cần kiểm tra điều này.
  4. **Mask riêng cho miền** (0.65 bit/tham số): một bản trọng số chung cộng nhiều mask cho từng loại tài liệu (ví dụ chữ in, chữ viết tay, ngôn ngữ). Với cấu trúc, tương tự là mỗi miền chọn một tập neuron khác nhau của cùng mô hình (một bản trọng số), nhưng đây chỉ là ý tưởng.
- **Dữ liệu:** Fig. 4 cho thấy calibration nhỏ bão hoà còn dữ liệu lớn tiếp tục cải thiện; với dự án đã có distillation từ thầy 4B trên dữ liệu OCR lớn, đây là lý do hợp lý để học lựa chọn cấu trúc trên dữ liệu thật thay vì chỉ 128 mẫu calibration.
- **Liên hệ D_lin / D_res:** MaskLLM không bù tuyến tính gì (trọng số đóng băng) nhưng vẫn vượt SparseGPT có bù; điều này cho thấy *chọn đúng cái bỏ* có thể quan trọng hơn *bù sau khi bỏ* trong chuẩn N:M. Cần thận trọng khi suy ra cho cắt cấu trúc, nơi chọn lựa ít linh hoạt hơn (xoá cả hàng/cột) và bù có thể đóng vai trò lớn hơn (xem 27, 29).

## 9. Thí nghiệm gợi ý

1. **Cổng neuron học được trên một layer:** chọn một MLP của decoder, gắn cổng $z_j$ cho từng neuron với logit khởi tạo từ điểm Wanda-sp/FLAP (prior), huấn luyện 500 bước bằng Gumbel-Softmax với loss distillation từ thầy 4B (trọng số đóng băng), so với cắt cùng số neuron bằng điểm thuần. Đo CER OCR và loss distill.
2. **Ngân sách bước:** lặp lại thí nghiệm 1 với 0 (chỉ prior), 50, 500, 2000 bước để xem đường cong "prior so với học" tương tự Fig. 4.
3. **Mask theo miền:** học cổng riêng cho 2 miền tài liệu (ví dụ bảng biểu và văn bản thường) từ cùng prior chung; đo mức trùng nhau giữa các cổng (như Fig. 8 trong phụ lục về khác biệt mask).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Ứng viên 2:4 | $\lvert\mathcal S\rvert=\binom42=6$ mask mỗi nhóm 4 trọng số |
| Mục tiêu | $\min_{\mathcal M}\ \mathbb E_x\,\mathcal L_{LM}(x;W\odot\mathcal M)$ với $W$ đóng băng |
| Gumbel-Softmax | $\tilde y_i=\dfrac{\exp((\log p_i+g_i)/\tau)}{\sum_j\exp((\log p_j+g_j)/\tau)}$ |
| Mask mềm | $\widetilde{\mathcal M}=\sum_i\tilde y_i\hat{\mathcal M}_i$ |
| Tham số hoá | $p_i=\mathrm{softmax}(\pi\kappa)$, $\kappa$ điều khiển mức ngẫu nhiên |
| Chính quy hoá | trừ $\lambda\sum\lVert W\odot\widetilde{\mathcal M}\rVert_2^2$ ($\lambda=10^{-5}$) |
| Mask prior | $\pi_i'=\pi_i+\sigma(\pi)\cdot(\mathcal M_0\hat{\mathcal M}_i^\top-N/2)\cdot\alpha$ |
| Cấu hình | 2000 bước, batch 256, AdamW 5e-4, $\kappa$ 1e2 đến 5e2, $\tau$ 4 đến 0.05, $\alpha=3$ |
| Con số chính | LLaMA-2 7B 2:4: PPL 6.72 (SparseGPT 10.42, Wanda 11.29, dense 5.12); 1.4×, 73% bộ nhớ, 0.65 bit/tham số cho mask miền; 1,280 GPU-giờ |

## 11. Kết luận ngắn

MaskLLM biến việc chọn mask 2:4 thành bài toán học phân phối bằng Gumbel-Softmax với loss ngôn ngữ thật trên dữ liệu lớn, trọng số đóng băng, có prior từ phương pháp one-shot và chuyển giao theo miền. Nó vượt rõ SparseGPT/Wanda với cái giá hàng nghìn GPU-giờ. Với dự án (cắt cấu trúc), nó ít liên quan trực tiếp, nhưng cung cấp bằng chứng và công thức cho hướng "học cổng cấu trúc bằng loss/distillation, khởi tạo từ điểm rẻ".

Link gốc: https://arxiv.org/abs/2409.17481
