# Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning (2022)

**Tác giả:** Elias Frantar, Sidak Pal Singh, Dan Alistarh (IST Austria; ETH Zurich; Neural Magic)  
**Venue / arXiv:** arXiv:2208.11580 (v2, 8/1/2023); NeurIPS 2022 (paper ghi "36th Conference on NeurIPS 2022"). Code: github.com/IST-DASLab/OBC.  
**Nguồn:** https://arxiv.org/abs/2208.11580  
**Chủ đề:** Nén **post-training** (một lần, không retrain) theo từng lớp: ExactOBS (pruning) và OBQ (quantization), trên nền Optimal Brain Surgeon; cập nhật bù trừ theo từng hàng của ma trận trọng số.  
**Vị trí trong lộ trình:** Đọc sau 01 (OBD/OBS: nguồn gốc công thức), 18 (N:M) và 21 (survey). **Đây là nền toán của 23 (SparseGPT)** và của GPTQ (survey 21 nói GPTQ dựa trên OBQ). Bài này thêm: (1) đưa OBS về bài toán **hồi quy bình phương tối thiểu theo từng lớp** nên công thức OBS là chính xác, không còn là xấp xỉ; (2) tính exact greedy một cách hiệu quả bằng cập nhật nghịch đảo Hessian; (3) dùng cùng khung cho quantization. Tiếp theo: 23 (SparseGPT, mở rộng lên LLM bằng cách ép thứ tự cột chung), 27 (LLM Surgeon).

> Đây là ghi chú học tập bằng tiếng Việt, không phải bản dịch. Phần dẫn xuất công thức dưới đây do chúng ta viết lại từ các phương trình của paper (paper dẫn đến Hassibi & Stork cho chứng minh), kèm một ví dụ số tự tính. Phần "Ứng dụng cho dự án" là suy luận. Thí nghiệm của paper là CNN (ResNet, YOLOv5) và BERT, **chưa có LLM** (paper nói "plan to investigate ... very large-scale language models").

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết (dẫn xuất)](#3-phương-pháp-chi-tiết-dẫn-xuất)
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

**Hai cách nén mạng:** pruning (đặt trọng số về 0) và quantization (giảm số bit). Phần lớn phương pháp cần train lại một phần hoặc toàn bộ sau khi nén.

**Bối cảnh post-training:** cho một mô hình đã train và một ít dữ liệu hiệu chuẩn (calibration; ở bài này 1024 mẫu), phải nén **trong một bước, không retrain**. Cách chuẩn: chia nhỏ thành bài toán **theo từng lớp** (layer-wise): với lớp có trọng số $W_\ell$ và đầu vào $X_\ell$, tìm $\hat W_\ell$ thoả ràng buộc nén sao cho đầu ra gần nhất. Quantization post-training đã có AdaRound, AdaQuant, BRECQ; pruning post-training có AdaPrune (cắt theo magnitude rồi tối ưu lại phần còn lại).

**Optimal Brain Surgeon (OBS)** (Hassibi & Stork; xem 01) là khung cổ điển: dùng Hessian của loss để chọn trọng số cần xoá và cập nhật các trọng số còn lại để bù. Vấn đề: Hessian $d\times d$ quá lớn với $d$ cỡ $10^5$-$10^6$, và các biến thể hiện đại (WoodFisher, M-FAC) cần pruning dần cộng retrain.

**Câu hỏi của bài:** có thể áp dụng OBS **chính xác** (xoá từng trọng số, cập nhật lại Hessian sau mỗi bước) ở quy mô mạng hiện đại, trong bối cảnh post-training, và dùng chung cho quantization không?

Giải thích thuật ngữ:
- **Hessian $H$:** ma trận đạo hàm bậc hai của loss theo trọng số; cho biết loss cong thế nào quanh điểm hiện tại.
- **Layer-wise reconstruction:** thay vì nhìn loss toàn mạng, đo sai khác đầu ra của từng lớp $\|WX-\hat WX\|_2^2$ trên dữ liệu hiệu chuẩn.
- **Greedy (tham lam):** mỗi bước chọn quyết định tốt nhất cục bộ, không đảm bảo tối ưu toàn cục.
- **Dampening:** cộng một số nhỏ vào đường chéo $H$ để nghịch đảo ổn định.

## 2. Ý tưởng chính

Với một lớp tuyến tính, loss tái tạo $\|WX-\hat WX\|_2^2$ là **hàm bậc hai chính xác** theo trọng số, và điểm xuất phát (trọng số dense) có loss bằng 0 nên gradient bằng 0. Hai giả định của OBS (Taylor bậc hai, gradient ≈ 0) khi đó đúng tuyệt đối, nên công thức OBS cho "trọng số xoá tiếp theo tối ưu" và "cập nhật bù tối ưu" là chính xác. Hơn nữa loss tách theo **hàng** của $W$ (mỗi hàng là một đầu ra độc lập), nên chỉ cần Hessian $d_{col}\times d_{col}$ cho từng hàng, và Hessian đó là **$H=2XX^\top$, giống nhau cho mọi hàng**. Khó khăn còn lại là tính nghịch đảo sau mỗi lần xoá; paper dùng một bước khử Gauss để cập nhật nghịch đảo trong $\Theta(d_{col}^2)$ thay vì nghịch đảo lại $\Theta(d_{col}^3)$. Thay "xoá về 0" bằng "làm tròn vào lưới lượng tử" thì có thuật toán quantization (OBQ) với cùng chi phí.

## 3. Phương pháp chi tiết (dẫn xuất)

### 3.1 Bài toán layer-wise và tách theo hàng

Lớp có $W\in\mathbb R^{d_{row}\times d_{col}}$ và đầu vào hiệu chuẩn $X\in\mathbb R^{d_{col}\times N}$ (với conv, $d_{col}$ là số trọng số trong một filter). Bài toán (Phương trình 2 của paper):

$$\arg\min_{\hat W}\ \|WX-\hat WX\|_2^2\quad\text{s.t. } \mathcal C(\hat W)>C.$$

Vì $\|WX-\hat WX\|_F^2=\sum_{i=1}^{d_{row}}\|W_{i,:}X-\hat W_{i,:}X\|_2^2$, mỗi hàng độc lập: xoá $[W]_{ij}$ chỉ ảnh hưởng đầu ra hàng $i$. Do đó **không có tương tác Hessian giữa các hàng**.

### 3.2 Hessian của một hàng

Gọi $w\in\mathbb R^{d_{col}}$ là một hàng và $\hat w=w+\delta$. Sai số đầu ra:

$$L(\delta)=\|(w+\delta)^\top X - w^\top X\|_2^2=\|\delta^\top X\|_2^2=\delta^\top XX^\top\delta=\tfrac12\,\delta^\top H\,\delta,\qquad \boxed{H=2XX^\top}$$

Loss bậc hai chính xác, tại $\delta=0$ gradient bằng 0, Hessian $H$ **không phụ thuộc vào $w$** nên giống nhau cho mọi hàng. (Chỉ phụ thuộc dữ liệu hiệu chuẩn $X$.)

### 3.3 Công thức OBS: xoá một trọng số $p$

Muốn đặt $\hat w_p=0$, tức ràng buộc $e_p^\top\delta=-w_p$ ($e_p$ là vector chuẩn thứ $p$). Chọn $\delta$ nhỏ nhất loss:

$$\min_\delta\ \tfrac12\delta^\top H\delta\quad\text{s.t. } e_p^\top\delta+w_p=0.$$

Lagrangian $\mathcal L(\delta,\lambda)=\tfrac12\delta^\top H\delta+\lambda\,(e_p^\top\delta+w_p)$ (paper viết $\delta^\top H\delta$ không có hệ số 1/2; chỉ khác hệ số, không đổi argmin). Đạo hàm theo $\delta$ bằng 0:

$$H\delta+\lambda e_p=0\ \Rightarrow\ \delta=-\lambda H^{-1}e_p=-\lambda H^{-1}_{:,p}.$$

Thế vào ràng buộc: $-\lambda\,e_p^\top H^{-1}e_p+w_p=0$, suy ra $\lambda=w_p/[H^{-1}]_{pp}$. Vậy:

$$\boxed{\delta_p=-\frac{w_p}{[H^{-1}]_{pp}}\,H^{-1}_{:,p}}$$

Loss tăng lên:

$$\Delta L_p=\tfrac12\delta_p^\top H\delta_p=\tfrac12\lambda^2\,e_p^\top H^{-1}e_p=\tfrac12\frac{w_p^2}{[H^{-1}]_{pp}}.$$

Do đó trọng số chọn xoá (Phương trình 3 của paper, bỏ hệ số 1/2):

$$\boxed{p^\star=\arg\min_p\ \frac{w_p^2}{[H^{-1}]_{pp}}}$$

Diễn giải: bỏ trọng số $w_p$ "rẻ" nếu $|w_p|$ nhỏ **hoặc** nếu $[H^{-1}]_{pp}$ lớn (hướng $p$ có độ cong $H$ thấp, tức các trọng số khác dễ bù). Khi $H$ chéo, $[H^{-1}]_{pp}=1/H_{pp}$ và tiêu chí thành $H_{pp}w_p^2$ giống OBD (01); phần cập nhật $\delta$ là điều OBD không có. Cập nhật $\delta$ sửa các trọng số còn lại để bù đắp (hàng $w\leftarrow w+\delta_p$, thành phần $p$ trở thành đúng 0).

### 3.4 Ví dụ số nhỏ (tự tính)

Một hàng 2 trọng số, $H=\begin{pmatrix}2&1\\1&2\end{pmatrix}$, $H^{-1}=\frac13\begin{pmatrix}2&-1\\-1&2\end{pmatrix}$, $w=(1,\,0.5)$.

- Điểm: $p=1$: $1^2/(2/3)=1.5$; $p=2$: $0.5^2/(2/3)=0.375$. Chọn xoá $p=2$.
- Cập nhật: $\delta=-\frac{0.5}{2/3}\,H^{-1}_{:,2}=-0.75\cdot(-\tfrac13,\tfrac23)=(0.25,\,-0.5)$. Trọng số mới $(1.25,\,0)$.
- Kiểm tra: $H\delta=(0.5-0.5,\ 0.25-1)=(0,-0.75)$, $\delta^\top H\delta=0.375$, nên $\Delta L=0.1875=\tfrac12\cdot0.375$. Khớp công thức. Trọng số còn lại tăng từ 1 lên 1.25 để bù cho trọng số bị xoá, vì hai đầu vào tương quan (độ lệch ngoài đường chéo của $H$ bằng 1).

### 3.5 Lặp lại theo từng bước (ExactOBS) và cập nhật nghịch đảo

Lặp: xoá một trọng số, cập nhật phần còn lại, rồi lại tính tiêu chí trên **Hessian của tập trọng số còn lại**. Gọi $H_M$ là Hessian của các trọng số còn trong mask $M$: nó chỉ là các hàng/cột tương ứng của $H$ dense (vì $H=2XX^\top$ không phụ thuộc $w$). **Nhưng $(H_M)^{-1}\ne (H^{-1})_M$**, nên không thể chỉ cắt nghịch đảo ban đầu. Nghịch đảo lại $\Theta(d_{col}^3)$ mỗi bước là quá đắt.

**Lemma 1 (xoá hàng/cột $p$ khỏi $H$, cập nhật nghịch đảo):**

$$\boxed{H^{-1}_{-p}=\Big(H^{-1}-\frac{1}{[H^{-1}]_{pp}}\,H^{-1}_{:,p}\,H^{-1}_{p,:}\Big)_{-p}}$$

Tức là khử Gauss hàng/cột $p$ trong $H^{-1}$ rồi bỏ chúng đi; chi phí $\Theta(d_{col}^2)$. (Đây là công thức Schur-complement quen thuộc cho nghịch đảo ma trận con: phần tử $(i,j)$ của kết quả là $[H^{-1}]_{ij}-[H^{-1}]_{ip}[H^{-1}]_{pj}/[H^{-1}]_{pp}$.) Trong cài đặt, không cần co ma trận: sau khi khử, hàng và cột $p$ bằng 0 và không bao giờ được truy cập lại.

```
# Algorithm 1: ExactOBS cho MỘT hàng w (d_col trọng số), cần H^{-1} = (2XX^T)^{-1}
M = {1..d_col}
for i in 1..k:                                  # k = số trọng số cần xoá
    p  = argmin_{p in M}  w[p]^2 / Hinv[p,p]    # chọn trọng số rẻ nhất
    w  = w - Hinv[:,p] * (w[p] / Hinv[p,p])     # cập nhật bù; w[p] về đúng 0
    Hinv = Hinv - outer(Hinv[:,p], Hinv[p,:]) / Hinv[p,p]   # Lemma 1, O(d_col^2)
    M  = M \ {p}
# chi phí: O(k * d_col^2) cho một hàng
```

### 3.6 Xét cả ma trận: chọn mask toàn cục

Muốn xoá trọng số có điểm thấp nhất **trên toàn ma trận** ở mỗi bước, tưởng như cần giữ $d_{row}$ ma trận nghịch đảo $d_{col}\times d_{col}$ (tốn bộ nhớ $O(d_{row}d_{col}^2)$). Quan sát: không có tương tác giữa hàng, nên **thứ tự xoá trong mỗi hàng là cố định**, và độ tăng loss khi xoá một trọng số chỉ phụ thuộc những trọng số đã xoá trong **cùng hàng**. Do đó:

1. Chạy ExactOBS cho từng hàng, xoá hết (hoặc đến mức cần), ghi lại $\delta L_p=\tfrac12w_p^2/[H^{-1}]_{pp}$ tại thời điểm xoá mỗi trọng số ($\Theta(d)$ bộ nhớ thêm).
2. Mask toàn cục = các trọng số có $\delta L$ thấp nhất (xếp theo thứ tự đã định của từng hàng).
3. Với mask mỗi hàng $M_i$ đã biết, tính trọng số cuối bằng công thức OBS nhóm: $\delta_{M}=-H^{-1}_{:,M}\big((H^{-1})_{M}\big)^{-1}w_{M}$ (nhanh hơn chạy lại), hoặc đọc lại vết (trace) từ bộ nhớ CPU nếu đủ ($O(d_{row}d_{col}^2)$ CPU).

Độ phức tạp tổng: thời gian $O(d_{row}\cdot d_{col}^3)$ (tương đương $O(d\cdot d_{col}^2)$ với $d=d_{row}d_{col}$), bộ nhớ $\Theta(d_{col}^2)$, **chính xác** (không xấp xỉ), so với $O(d^4)$ cho OBS ngây thơ. Ví dụ: ResNet50 toàn mạng trong hơi hơn 1 giờ trên một RTX 3090.

### 3.7 Chi tiết cài đặt và các mẫu thưa

- **Dampening:** $H$ có thể không khả nghịch (ít mẫu, đầu vào chết hoặc phụ thuộc tuyến tính): cộng một số nhỏ vào đường chéo, và/hoặc tăng dữ liệu bằng augmentation (rẻ vì chỉ cộng dồn vào $H$ một lần).
- **N:M (ví dụ 2:4):** chỉ đổi một điều: chọn trọng số có $\delta L$ nhỏ nhất **trong khối còn chưa đủ N trọng số bị xoá**; mọi hàng có cùng độ thưa nên dừng khi đạt $1-N/M$, không cần bước mask toàn cục.
- **Block sparsity (khối $c$):** công thức nhóm (Phương trình 5): $w_P=\arg\min_P\ w_P^\top((H^{-1})_P)^{-1}w_P$, $\delta_P=-H^{-1}_{:,P}((H^{-1})_P)^{-1}w_P$, áp dụng Lemma 1 lần lượt cho từng $p\in P$.

### 3.8 Optimal Brain Quantizer (OBQ)

Cùng Lagrangian (Phương trình 6), nhưng ràng buộc mới là $e_p^\top\delta=\mathrm{quant}(w_p)-w_p$ (sau cập nhật $\hat w_p=\mathrm{quant}(w_p)$). Kết quả (Phương trình 7):

$$p^\star=\arg\min_p\frac{(\mathrm{quant}(w_p)-w_p)^2}{[H^{-1}]_{pp}},\qquad \delta_p=-\frac{w_p-\mathrm{quant}(w_p)}{[H^{-1}]_{pp}}H^{-1}_{:,p}.$$

Nếu $\mathrm{quant}(\cdot)\equiv 0$ ta quay lại OBS cho pruning, nên đây là khái quát hoá. Với quantization cả lớp: lần lượt lượng tử hoá trọng số "dễ nhất" (tăng loss ít nhất), rồi **cập nhật các trọng số chưa lượng tử hoá** để bù. Điểm khác với làm tròn thẳng: thứ tự và bù trừ làm các trọng số về sau bị đẩy, nên kết quả khác phép làm tròn ban đầu và tốt hơn. Vấn đề **outlier** (trọng số nằm ngoài lưới có lỗi lớn): nếu để cuối thì không còn trọng số nào để bù; heuristic: lượng tử hoá ngay những trọng số có lỗi $>\Delta/2$ ($\Delta$ là khoảng cách giữa hai mức lưới) khi chúng xuất hiện.

(GPTQ, ghi trong survey 21, bắt nguồn từ OBQ.)

### 3.9 Phân bổ độ thưa/bit theo từng lớp

Với ràng buộc FLOPs/thời gian, OBC dùng **model database**: nén độc lập từng lớp ở mọi mức (độ thưa, bit) và đo loss hiệu chuẩn của mô hình chỉ nén một lớp; rồi giải bài toán chọn mức cho từng lớp bằng quy hoạch động (thuật toán SPDY, công thức kiểu AdaQuant), cuối cùng "ghép" các lớp lại và hiệu chỉnh thống kê (batchnorm reset, hoặc mean/variance correction cho YOLO và BERT).

## 4. Thí nghiệm và kết quả chính

Mô hình: ResNet18/34/50 (ImageNet), YOLOv5 (COCO), BERT, BERT3, BERT6 (SQuAD, F1). Dữ liệu hiệu chuẩn: 1024 mẫu train ngẫu nhiên; ImageNet thêm flip/crop để tăng 10x. So sánh: GMP (global magnitude pruning), L-OBS (xấp xỉ OBS theo lớp), AdaPrune, ExactOBS.

**Pruning không cấu trúc theo FLOPs giảm (Bảng 1):**

| Mô hình (dense) | Phương pháp | 2x | 3x | 4x |
|---|---|---|---|---|
| ResNet50 (76.13) | GMP | 74.86 | 71.44 | 64.84 |
| | L-OBS | 75.48 | 73.73 | 71.24 |
| | AdaPrune | 75.53 | 74.47 | 72.39 |
| | ExactOBS | 75.64 | 75.01 | 74.05 |
| YOLOv5l (66.97, mAP@0.5) | AdaPrune | 66.00 | 64.88 | 62.71 |
| | ExactOBS | 66.14 | 65.35 | 64.05 |
| BERT (88.53, F1) | GMP | 65.64 | 12.52 | 09.23 |
| | AdaPrune | 87.12 | 70.32 | 18.75 |
| | ExactOBS | 87.81 | 85.87 | 82.10 |

(YOLOv5l 2x: tất cả gần nhau; GMP 65.83, L-OBS 66.21, tức L-OBS nhỉnh hơn ExactOBS.) BERT là mô hình khó cắt: ExactOBS là phương pháp duy nhất còn kết quả hợp lý ở 3x, 4x. Với BERT 3x/4x (sụt >2-3%), thêm bước global AdaPrune (gAP) sau cùng: gAP+AdaPrune 86.99 / 84.10; gAP+ExactOBS 87.57 / 86.42 (Bảng 5).

**N:M (Bảng 2, 3):**

| Mô hình | Dense | AdaPrune 4:8 | ExactOBS 2:4 | ExactOBS 4:8 |
|---|---|---|---|---|
| ResNet18 | 69.76 | 68.63 | 68.81 | 69.18 |
| ResNet34 | 73.31 | 72.36 | 72.66 | 72.95 |
| ResNet50 | 76.13 | 74.75 | 74.71 | 75.20 |

| Mô hình (2:4, F1) | Dense | AdaPrune | ExactOBS |
|---|---|---|---|
| BERT3 | 84.66 | 82.75 | 83.54 |
| BERT6 | 88.33 | 85.02 | 86.97 |
| BERT | 88.53 | 85.24 | 86.77 |

ExactOBS 2:4 ngang hoặc hơn AdaPrune 4:8 (mẫu khắt khe hơn), và hơn AdaPrune 2:4 khoảng 1-2% F1 trên BERT.

**Quantization trọng số, per-channel bất đối xứng (Bảng 4, ResNet; ImageNet top-1; dense ResNet18 69.76, ResNet50 76.13):**

| Phương pháp | R18 4b | R18 3b | R18 2b | R50 4b | R50 3b | R50 2b |
|---|---|---|---|---|---|---|
| AdaRound | 69.34 | 68.37 | 63.37 | 75.84 | 75.14 | 71.58 |
| AdaQuant | 68.12 | 59.21 | 00.10 | 74.68 | 64.98 | 00.10 |
| BRECQ | 69.37 | 68.47 | 64.70 | 75.88 | 75.32 | 72.41 |
| OBQ | 69.56 | 68.69 | 64.04 | 75.72 | 75.24 | 70.71 |

OBQ nén **độc lập từng lớp** nhưng ngang các phương pháp tuần tự (chỉ thua ở 2-bit).

**Nén kết hợp:**
- **GPU (BOPs = số bit × FLOPs):** mỗi lớp chọn trong 8w8a, 4w4a, 8w8a+2:4, 4w4a+2:4. Với ≈2.5% sụt tương đối: giảm 12-14x BOPs cho ResNet, 7-8x cho YOLO và BERT. 4w4a + 2:4 post-training trên ResNet18 đạt 67.20%, so với 67.33% của QAT-thưa 90 epoch (không hoàn toàn so sánh được).
- **CPU (DeepSparse, block size 4 + INT8, ResNet50, batch 64):** tăng tốc thật 4x với sụt 1%, 5x với sụt 2% (tăng tốc dense INT8 cơ sở ≈2.7x, thưa nhân lên xấp xỉ).

## 5. Phân tích / ablation

- **Sai số tái tạo theo lớp:** ở lớp đầu của ResNet18 (Hình 2), ExactOBS có squared error thấp nhất "by a wide margin", rồi AdaPrune, rồi L-OBS và GMP. Kiểm chứng rằng việc "chính xác từng bước" giúp ở cấp lớp, rồi chuyển được thành độ chính xác end-to-end.
- **Cập nhật tần suất cao hơn tốt hơn:** AdaPrune chỉ cắt theo magnitude rồi tối ưu lại; ExactOBS tái tối ưu **sau mỗi trọng số** và chọn bằng thông tin bậc hai (paper nói đây là đẩy ý tưởng của AdaPrune đến cực hạn).
- **Độc lập vs tuần tự:** nén từng lớp độc lập cho phép ghép linh hoạt (database) và nhanh đổi ràng buộc; cái giá phải trả là không bù lỗi tích luỹ giữa các lớp (paper nói ExactOBS vẫn có thể chạy tuần tự, ở phụ lục).
- **Global AdaPrune cộng thêm:** ở BERT 3x/4x, bước tối ưu toàn cục vẫn để lại lợi thế ExactOBS (>0.5% và >2% F1 theo văn bản).
- **Mặt chi phí:** phần dẫn xuất ở mục 3 cho thấy chi phí/bộ nhớ; phần runtime chi tiết nằm ở phụ lục, tôi không chép số cụ thể vì chưa đọc kỹ.

## 6. Giới hạn

**Tác giả nêu:**
- Greedy, không đảm bảo tối ưu toàn cục (nói rõ trong phần OBS layer-wise).
- Chỉ xét lớp tuyến tính/conv; structured pruning "nên mở rộng được" nhưng chưa làm.
- Chưa thử mô hình rất lớn; kế hoạch tương lai là LLM.
- $H$ có thể không khả nghịch, cần dampening.

**Chúng ta nhận xét:**
- Thời gian $O(d_{row}d_{col}^3)$ và việc mỗi hàng có thứ tự riêng với $H^{-1}$ riêng: với $d_{col}$ cỡ 4k-12k (LLM), chi phí lớn. Đây chính là điều SparseGPT (23) giải quyết bằng cách dùng **chung một thứ tự cột** cho mọi hàng để tái dùng một nghịch đảo (nội dung bài 23).
- Loss layer-wise dùng $X$ **của mô hình dense** (đầu vào lớp tính từ mô hình gốc): lỗi tích luỹ của các lớp trước không được tính. Với mô hình sâu, sai khác này có thể quan trọng (suy luận của chúng ta).
- Mọi kết quả là CNN/BERT cỡ nhỏ; chưa có kiểm chứng trên decoder lớn, mô hình lai hoặc VLM.
- Mục tiêu là sparsity/bit của **trọng số**, không phải cắt cả kênh/lớp.

## 7. Liên hệ với các bài khác

- **01 (OBD/OBS):** OBC dùng đúng công thức OBS; điểm mới là chỉ ra rằng ở bài toán tái tạo lớp tuyến tính thì giả định bậc hai là chính xác và làm nó nhanh.
- **02, 03:** pruning cổ điển cần retrain dài; OBC thuộc nhánh post-training.
- **18 (2:4):** OBC chọn mask 2:4 bằng $\delta L$ thay cho magnitude, kèm cập nhật bù; kết quả tốt hơn AdaPrune ở N:M.
- **23 (SparseGPT):** cùng bài toán $\|WX-\hat WX\|^2$ và công thức OBS, nhưng ép chung thứ tự cột + khử nghịch đảo qua phân rã Cholesky để mở rộng lên LLM tỉ tham số; **đọc 22 kỹ trước khi đọc 23**.
- **21 (survey):** GPTQ bắt nguồn từ OBQ.
- **27 (LLM Surgeon), 24 (Wanda):** 27 mở rộng OBS ra nhiều mức và cấu trúc; Wanda đơn giản hoá bằng cách bỏ cập nhật.
- **29 (SliceGPT), 36 (ReplaceMe):** cũng có ý "sửa phần còn lại sau khi cắt" bằng ánh xạ tuyến tính, cùng tinh thần với bù trừ $\delta$.

## 8. Ứng dụng cho dự án (suy luận)

- **Cầu nối trực tiếp tới D_lin / D_res.** Công thức bù $\delta$ là "ánh xạ tuyến tính tối ưu" theo bình phương tối thiểu cho phần hư hại do cắt: phần giảm được bằng cập nhật bù chính là loại thiệt hại có thể sửa bằng một phép tuyến tính (cùng họ với D_lin của bạn), còn phần còn lại sau bù (độ tăng loss không thể xoá) tương ứng D_res. Ý tưởng: sau khi cắt neuron MLP, thay vì chỉ distill, hãy **giải thẳng bình phương tối thiểu** cho trọng số của lớp kế tiếp (hồi quy $W_{down}$ để khớp đầu ra dense trên mẫu hiệu chuẩn). Đây là suy diễn của chúng ta, chưa thấy trong paper.
- **Cắt neuron MLP có cấu trúc (xoá cả cột của $W_{up}/W_{gate}$ và hàng của $W_{down}$):** OBS cho trọng số lẻ, nhưng khung mở rộng được cho nhóm (công thức block $\delta_P$ ở Mục 3.7 là cùng một họ). Cần kiểm tra thực nghiệm; paper chỉ nói "structured pruning nên mở rộng được".
- **Dữ liệu hiệu chuẩn:** dùng ảnh/văn bản đúng miền OCR (tài liệu), vì $H=2XX^\top$ chỉ phản ánh phân phối của $X$.
- **Chọn mask 2:4 hoặc quantization:** nếu sau cùng muốn thêm 2:4 (bài 18) hay INT4, OBC/SparseGPT cho phương án tốt hơn magnitude khi không retrain.
- **Hạn chế khi áp vào mô hình lai:** các lớp GatedDeltaNet có phép toán phi tuyến/hồi quy theo thời gian; layer-wise reconstruction chỉ áp dụng được cho các phép chiếu tuyến tính (q/k/v/gate/out projections) chứ không cho phần cập nhật trạng thái.

## 9. Thí nghiệm gợi ý

1. Lấy một lớp Linear của decoder, thu $X$ từ ~1k mẫu OCR, tự cài ExactOBS cho một hàng (khoảng 20 dòng PyTorch như Algorithm 1) và so sánh 3 cách đạt 50% thưa: magnitude, magnitude + bù OBS, ExactOBS. Đo $\|WX-\hat WX\|^2$ và loss end-to-end.
2. Cắt 30% neuron của một khối MLP: (a) xoá theo norm rồi để nguyên; (b) xoá rồi giải bình phương tối thiểu lại $W_{down}$ trên dữ liệu hiệu chuẩn; (c) (b) + distill ngắn. So khoảng cách tới đầu ra dense để ước lượng phần D_lin so với D_res.
3. Kiểm tra độ nhạy với kích thước dữ liệu hiệu chuẩn (128, 1024, 4096 mẫu) và hệ số dampening, vì $H$ thiếu hạng khi $N$ nhỏ.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / nội dung |
|---|---|
| Bài toán lớp | $\min_{\hat W}\|WX-\hat WX\|_2^2$ s.t. ràng buộc nén |
| Tách hàng | Tổng sai số các hàng, không tương tác giữa hàng |
| Hessian một hàng | $H=2XX^\top$ ($d_{col}\times d_{col}$), chung cho mọi hàng |
| Điểm OBS | $\delta L_p=\tfrac12 w_p^2/[H^{-1}]_{pp}$ (paper bỏ 1/2) |
| Cập nhật bù | $\delta_p=-\dfrac{w_p}{[H^{-1}]_{pp}}H^{-1}_{:,p}$ |
| Cập nhật nghịch đảo | $H^{-1}_{-p}=(H^{-1}-\tfrac{1}{[H^{-1}]_{pp}}H^{-1}_{:,p}H^{-1}_{p,:})_{-p}$, $\Theta(d_{col}^2)$ |
| Mask toàn cục | Ghi $\delta L$ khi xoá trong từng hàng, chọn $\delta L$ nhỏ nhất, rồi dùng $\delta_M=-H^{-1}_{:,M}((H^{-1})_M)^{-1}w_M$ |
| Độ phức tạp | $O(d_{row}d_{col}^3)$ thời gian, $\Theta(d_{col}^2)$ bộ nhớ (so với $O(d^4)$) |
| N:M | Chỉ xét khối còn < N trọng số bị xoá; dừng khi đạt $1-N/M$ |
| OBQ | Thay đích 0 bằng $\mathrm{quant}(w_p)$; xử lý outlier ngay khi lỗi $>\Delta/2$ |
| Kết quả nổi bật | ResNet50 4x FLOPs: 74.05 (ExactOBS) vs 72.39 (AdaPrune); BERT 4x: 82.10 vs 18.75; 4w4a+2:4 ResNet18: 67.20; CPU 4x thực với sụt 1% |

## 11. Kết luận ngắn

OBC chứng minh rằng khi nhìn nén từng lớp như bài toán bình phương tối thiểu, OBS là công thức **chính xác** và có thể chạy chính xác với chi phí $O(d_{row}d_{col}^3)$ nhờ cập nhật nghịch đảo từng bước và tận dụng việc Hessian chung cho mọi hàng. Hai thứ cần mang theo sang bài 23: (1) công thức chọn và bù $\delta_p$, $w_p^2/[H^{-1}]_{pp}$; (2) ý tưởng "xoá một trọng số, bù ngay vào các trọng số còn lại theo hướng $H^{-1}$". Với dự án, giá trị lớn nhất là góc nhìn "bù tuyến tính tối ưu sau khi cắt" gắn với D_lin/D_res.

Link gốc: https://arxiv.org/abs/2208.11580
