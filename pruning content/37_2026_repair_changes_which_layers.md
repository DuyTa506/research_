# Repair Changes Which Layers to Remove: Exact Optima for Depth Pruning of Language Models (2026)

**Tác giả:** Jianchang Su, Dawei Xiang, Yifan Zhang, Wei Zhang (University of Connecticut).  
**Venue / arXiv:** arXiv:2610.30552 (v1, 22/10/2026, phân loại cs.DC). Bản text không ghi hội nghị; xem như preprint.  
**Nguồn:** https://arxiv.org/abs/2610.30552  
**Chủ đề:** Depth pruning kèm **repair tuyến tính** (ma trận least-squares kiểu ReplaceMe): liệt kê chính xác mọi tập layer có thể cắt ở độ chi tiết thô, cho thấy tập tối ưu **thay đổi theo repair**; đề xuất chọn layer *dưới repair sẽ dùng* (R-GS) và một điểm số bậc hai (R-FS).  
**Vị trí trong lộ trình:** Đọc trước: 36 ReplaceMe (bắt buộc: bài này dùng nguyên repair của nó), 34 SLEB (R-GS là SLEB + repair), 31 ShortGPT, 32 Deeper layers. Bài này **gần nhất với nghiên cứu của bạn** (phân rã hư hại thành phần "sửa được bằng phép tuyến tính" và phần dư). Đọc tiếp: 39 Minitron, 43 Pruning vs scratch (về hồi phục bằng training).

> Tài liệu học tập tiếng Việt, ghi chú diễn giải, không phải bản dịch paper. Phần "Điều gì được đo" liệt kê chính xác đại lượng và thiết lập của paper. Số liệu lấy từ paper. Phần liên hệ dự án và các cách đọc ngoài văn bản là suy luận của chúng ta. Paper này dùng tên mô hình "Qwen3.8-27B" như trong văn bản; chúng tôi giữ nguyên tên.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Điều gì được đo (thiết lập chính xác)](#3-điều-gì-được-đo-thiết-lập-chính-xác)
4. [Phương pháp chi tiết](#4-phương-pháp-chi-tiết)
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

- **Depth pruning** bỏ nguyên layer. Với $L$ layer và bỏ $n$ layer có $\binom{L}{n}$ cách chọn; chấm điểm mỗi cách đòi chạy mô hình đã cắt, nên các bài trước dùng **proxy**: độ giống giữa vào/ra layer (ShortGPT, bài 31), khoảng cách góc của một khối liền (bài 32, ReplaceMe), tham lam theo perplexity (SLEB, bài 34), tìm kiếm tiến hoá (EvoPress), ước lượng Shapley...
- **Repair tuyến tính (bài 36):** thay một "run" các layer bị bỏ bằng ma trận $T\in\mathbb R^{d\times d}$ khớp bằng bình phương tối thiểu trên activation calibration, gộp vào `down_proj` của layer được giữ ngay trước đó; số tham số bằng mô hình đã cắt.
- **Khoảng cách của proxy tới tối ưu thật chưa từng được biết**, vì chưa ai tính tối ưu thật.
- **Möbius transform / phân rã tương tác:** từ hàm tập $c(S)$ (tăng loss khi bỏ tập $S$) tách ra phần riêng từng phần tử và phần tương tác của mọi tập con; bỏ là "cộng tính" khi mọi tương tác bằng 0. **Super-additive** (siêu cộng tính) ở đây: bỏ hai layer gần nhau tốn hơn tổng chi phí từng layer.

## 2. Ý tưởng chính

Bài làm ba việc. (1) Ở độ chi tiết "nhóm 2 layer" đủ nhỏ để **liệt kê hết** (khoảng 53.000 mô hình đã cắt trên 4 họ mô hình, có và không có repair), nên biết tối ưu chính xác. (2) Phát hiện: không repair thì tối ưu *tách thành 2–3 đoạn rời* (vì bỏ hai layer gần nhau đắt hơn tổng), còn có repair thì tối ưu *thành một khối liền* (vì một ma trận $T$ hấp thụ tương tác trong đoạn). Hệ quả: các phương pháp chọn layer **không tính tới repair** bỏ lỡ tối ưu có repair tới 189%. (3) Sửa: chọn layer **dưới đúng repair sẽ dùng** (tham lam với repair: R-GS, chi phí ngang SLEB), hoặc dùng điểm bậc hai tính từ mô hình nguyên (R-FS, một forward + một backward).

## 3. Điều gì được đo (thiết lập chính xác)

Phần này ghi lại chính xác những gì paper đo, để bạn so với nghiên cứu của mình.

**Đối tượng cắt.** Mô hình decoder-only $L$ layer, residual stream $h_\ell=h_{\ell-1}+f_\ell(h_{\ell-1})$. Layer chia thành $G$ **nhóm liền kề** 1 hoặc 2 layer. Một **run** = tập cực đại các nhóm bị bỏ kề nhau. Layer đầu luôn được giữ (repair của run bắt đầu từ layer đầu không thể gộp vào layer trước; mọi nhóm đầu trong các thí nghiệm liệt kê cũng là nhóm "critical").

**Repair.** Với run $a..b$, chỉ **một ma trận $T_{a,b}$ cho mỗi run**, tác động lên đầu ra MLP $m_{a-1}$ của layer được giữ ngay trước run, đầu ra thành $y+m_{a-1}T$ với $y=h_{a-1}-m_{a-1}$. Ước lượng bằng ridge:

$$\boxed{T_{a,b}=\arg\min_T\ \lVert M_{a-1}T-(H_b-Y_{a-1})\rVert_F^2+\lambda\lVert T\rVert_F^2}\qquad(\text{Eq. 1})$$

hàng của $M_{a-1},Y_{a-1},H_b$ là activation **của mô hình nguyên** tại các token calibration. $\lambda=10^{-3}\times$ trung bình đường chéo của $M^\top M$ (không ridge cho cấu hình "mặc định ReplaceMe": dùng pseudo-inverse float64 vì MLP-output của Gemma trên SlimOrca bị thiếu hạng). Maps được tính **một lần** từ activation đã cache của mô hình nguyên (tại mỗi run, độc lập; đây là một xấp xỉ: mỗi map giả định phần mạng còn lại nguyên vẹn).

**Hàm mục tiêu và dữ liệu.**
- *Loss increase* của tập con = loss token trung bình của mô hình đã cắt trừ loss của mô hình nguyên, trên **16 chuỗi WikiText-2 calibration, mỗi chuỗi 256 token**. Mọi phương pháp chọn layer trên các chuỗi này.
- Báo cáo **perplexity (PPL)** trên 16 chuỗi *khác* của tập test WikiText-2. Maps fit trên 128 chuỗi WikiText-2 train.
- **Accuracy** = trung bình trên ARC-Easy, ARC-Challenge, HellaSwag, PIQA, WinoGrande, mỗi tác vụ 1.000 mẫu (prompt lm-evaluation-harness); một sai số chuẩn khoảng 0.7 điểm. BoolQ báo riêng vì pruning dịch prior yes/no tới 20 điểm.
- Biến thể "task objective" (R-GS-T): cross-entropy của lựa chọn đúng trên **200 mẫu train** mỗi tác vụ (tập rời với mẫu đánh giá). Biến thể MMLU: 5 câu hỏi/môn từ validation (285 câu), đánh giá trên test (50 câu/môn, 2.850 câu).

**Liệt kê chính xác (Table 14).** Qwen3-1.7B (28 layer, 14 nhóm, mọi tập đến 6 nhóm bị bỏ; thêm độ chi tiết 1 layer, 28 nhóm, đến 3 layer); Gemma-3-4B (34 layer, 17 nhóm, đến 6); Llama-3.1-8B-Instruct (32 layer, 16 nhóm, đến 6); Qwen3-14B (40 layer, 20 nhóm, đến 4). Mỗi tập được tính trên mọi chuỗi calibration và test, không và có repair; khoảng 53.000 tập.

**Đánh giá thêm** (không liệt kê được): Qwen3.8-27B (64 layer, 3 layer linear-attention trên mỗi layer full-attention, không dùng cho bất kỳ quyết định thiết kế nào), bỏ 25% và 37.5% layer ở độ chi tiết 1 layer; Llama-3.1-8B, Gemma, Qwen3 ở các ngân sách tương tự. Baselines: ShortGPT, LaCo, ReplaceMe (khối cosine), Prune&Comp, SLEB; **tất cả đều được repair bằng cùng các map least-squares (+R)**, nên các phương pháp chỉ khác nhau ở *layer nào được chọn*. (LaCo: bỏ layer mà tiêu chí gộp của nó chọn, không gộp tham số.)

**Điều paper KHÔNG đo:** không có đại lượng nào gọi là $D_{lin}$/$D_{res}$; không đo CER hay tác vụ sinh; chỉ thử một loại repair (LS của ReplaceMe); không cắt neuron MLP.

## 4. Phương pháp chi tiết

### 4.1 Phân rã tương tác

$c(S)$ = tăng loss khi bỏ $S$, $c(\varnothing)=0$.

$$\mu(S)=\sum_{R\subseteq S}(-1)^{|S|-|R|}c(R),\qquad c(S)=\sum_{R\subseteq S}\mu(R)$$

$\mu(\{i\})$ là chi phí riêng; $\mu(\{i,j\})$ là số hạng cặp (dương = siêu cộng tính). **Tỉ lệ tương tác** = trung bình $|c(S)-\sum_{i\in S}c(\{i\})|$ chia cho trung bình $|c(S)|$.

### 4.2 Hai kiểu tương tác tìm thấy (mục 4)

- **Layer "critical":** nhóm mà bỏ riêng đã làm perplexity tăng hơn gấp đôi: 4/14 nhóm ở Qwen3-1.7B, 5/17 ở Gemma, 3/16 ở Llama, 3/20 ở Qwen3-14B, thường là các nhóm đầu. Số hạng cặp của chúng lớn và **âm** (vì loss bão hoà khi mô hình đã hỏng) và mọi tối ưu chính xác đều **giữ** chúng.
- **Siêu cộng tính cục bộ:** với nhóm không critical, $\mu(\{i,j\})>0$ và **suy giảm theo khoảng cách** $|i-j|$ với độ dài suy giảm khớp 2.9 layer (Gemma) đến 6.4 layer (Llama); gần 0 sau khoảng 10 layer. Số hạng cặp giải thích 52% đến 76% tính phi cộng tính tại 3 nhóm bị bỏ.

### 4.3 Thuật toán chọn dưới repair (R-GS, Algorithm 1)

```
Input: layer 1..L, ngân sách n, mục tiêu J, activation của mô hình nguyên (đã cache)
S <- {}
for k = 1..n:
    for mỗi layer i còn giữ, i != 1:
        với mỗi run a..b của S ∪ {i}: fit T_{a,b} theo Eq.1 (từ activation đã cache)
        J_i <- J(mô hình bỏ S ∪ {i}, đã repair)
    S <- S ∪ { argmin_i J_i }
return S, gộp các map vào down_proj
```

Chi phí: $\sum_{k=0}^{n-1}(L-k)$ lần đánh giá (bằng SLEB). Trên Llama-3.1-8B bỏ 12/32 layer, một RTX A6000: R-GS 384 giây (319 mô hình đã cắt), SLEB+repair 343, LaCo 487, Prune&Comp 56, R-FS 33, ReplaceMe 8, ShortGPT 1; mỗi phương pháp có repair cần thêm 29 giây thu activation.

### 4.4 Điểm bậc hai R-FS (mục 6.2)

$z\in\mathbb R^{N\times d}$ là residual stream sau khối $a..b$ trong mô hình nguyên; mô hình đã repair cho $z+\delta$ với $\delta_t=y_t+m_tT_{a,b}-z_t$ (phần dư của repair tại token $t$); các layer sau khối không đổi.

**Mệnh đề 1:** gọi $p_s(z)$ là phân phối token kế tiếp, $D(\delta)=\sum_s \mathrm{KL}(p_s(z)\,\Vert\,p_s(z+\delta))$. Khi đó $D(0)=0$, $\nabla D(0)=0$ và

$$D(\delta)=\tfrac12\,\mathrm{vec}(\delta)^\top F\,\mathrm{vec}(\delta)+O(\lVert\delta\rVert^3),\quad F=\sum_s\mathbb E_{y\sim p_s}\big[\nabla_z\log p_s(y)\nabla_z\log p_s(y)^\top\big]$$

($F$ là Fisher information của mạng phía sau theo residual stream). Thay kỳ vọng bằng token quan sát (empirical Fisher) và giữ một khối đường chéo hạng một mỗi vị trí ($F\approx\text{blockdiag}_t(g_tg_t^\top)$, $g_t=\nabla_{z_t}\sum_s\ell_s$) ta được

$$\boxed{s(a,b)=\sum_{t=1}^{N}(g_t\cdot\delta_t)^2}\qquad(\text{Eq. 4, R-FS})$$

Đọc nôm na: **phần dư sau repair, đo bằng thước loss** (hướng nào của residual stream làm loss nhạy thì phạt hướng đó nặng). Khác với ReplaceMe: LS giảm $\lVert\delta\rVert_2$ (chuẩn Euclid, mọi hướng như nhau); R-FS cân phần dư theo độ nhạy của loss. Paper nối với *balanced truncation* (Moore 1981). Chỉ cần một forward và một backward trên mô hình nguyên.

## 5. Thí nghiệm và kết quả chính

### 5.1 Cấu trúc tối ưu chính xác

**Hình 1/Phần 4:** Không repair, khối liền tốt nhất tệ hơn tối ưu tới 6.9 lần. Với repair của ReplaceMe, khối liền tốt nhất **bằng** tối ưu ở 9/18 ngân sách và trong vòng 5% ở 13/18.

**Table 1: perplexity vượt tối ưu của khối liền tốt nhất, không repair** (số nhóm bị bỏ 3/4/5/6):

| Mô hình | 3 | 4 | 5 | 6 |
|---|---|---|---|---|
| Qwen3-1.7B | 0% | 28% | 77% | 588% |
| Gemma-3-4B | 64% | 252% | 512% | 447% |
| Llama-3.1-8B | 32% | 34% | 76% | 190% |
| Qwen3-14B | 0% | 18% | - | - |

Từ 4 nhóm trở lên, mọi tối ưu chính xác ở mọi họ **tách bỏ thành 2 hoặc 3 đoạn**. Chấm một lần (one-shot) chọn các nhóm có chi phí riêng nhỏ nhất, chúng tụ lại gần nhau nên gần liền nhau, tệ hơn tối ưu 33% (Qwen3-1.7B) tới 581% (Gemma) ở 4 nhóm. SLEB (đánh giá mô hình thật) tìm được tối ưu ở **13/20** ngân sách.

**Table 2: PPL ở ngân sách nặng nhất được liệt kê** (n = số nhóm bị bỏ; trong ngoặc là PPL mô hình nguyên). Các cột: tối ưu của cắt thuần "as is" / chính tập đó sau repair / lựa chọn của ReplaceMe / tối ưu chính xác dưới repair.

| Mô hình | n | Tối ưu cắt thuần | Tập đó + repair | ReplaceMe | Tối ưu có repair |
|---|---|---|---|---|---|
| Qwen3-1.7B (26.5) | 4 | 73.4 | 50.3 | 75.1 | 41.7 |
| | 5 | 128.7 | 59.2 | 90.6 | 54.3 |
| | 6 | 229.2 | 80.2 | 99.5 | 76.0 |
| Gemma-3-4B (32.0) | 4 | 46.7 | 50.6 | 44.7 | 38.3 |
| | 5 | 65.9 | 51.6 | 72.2 | 42.8 |
| | 6 | 93.5 | 64.1 | 149.8 | 51.9 |
| Llama-3.1-8B (13.7) | 4 | 33.6 | 26.8 | 40.4 | 22.2 |
| | 5 | 53.0 | 32.5 | 61.9 | 28.3 |
| | 6 | 84.2 | 44.1 | 94.0 | 35.1 |
| Qwen3-14B (15.0) | 4 | 23.1 | 25.6 | 21.3 | 19.1 |

Kết luận của paper: (i) lấy tập tối ưu cho cắt thuần rồi repair thì kém tối ưu có repair 5% tới 33%; (ii) tiêu chí cosine của ReplaceMe kém 11% tới 189%. Với repair, hấp thụ tương tác trong run: tỉ lệ tương tác ở ngân sách nặng nhất giảm 62% -> 36% (Qwen3-1.7B), 66% -> 52% (Gemma), 57% -> 28% (Llama), 25% -> 15% (Qwen3-14B); nhưng các run được repair riêng vẫn tương tác (R² của tổng chi phí repair từng run là 0.8–0.9 ở 2 nhóm nhưng chỉ -0.09 tới 0.44 ở 6 nhóm).

### 5.2 Kết quả chính cho các phương pháp chọn (Table 3)

Mọi phương pháp dưới đây dùng cùng map LS (+R), chọn trên văn bản WikiText-2 (khối trên). PPL WikiText-2 / Avg accuracy 5 tác vụ (%).

| Mô hình, số layer bỏ | Nguyên | ShortGPT+R | ReplaceMe | SLEB+R | R-FS | R-GS |
|---|---|---|---|---|---|---|
| Qwen3-1.7B, 7/28 | 26.5 / 60.0 | 77.4 / 46.1 | 53.7 / 45.6 | 40.3 / 45.9 | 38.7 / 44.9 | 37.7 / 44.2 |
| Gemma-3-4B, 8/34 | 32.0 / 69.8 | 44.7 / 50.8 | 44.7 / 50.8 | 50.5 / 54.7 | 38.2 / 48.1 | 35.9 / 50.6 |
| Gemma-3-4B, 13/34 | 32.0 / 69.8 | 87.2 / 42.7 | 150.7 / 41.5 | 69.1 / 44.7 | 97.6 / 40.6 | 51.1 / 44.7 |
| Llama-3.1-8B, 8/32 | 13.7 / 72.4 | 39.4 / 59.5 | 40.4 / 57.2 | 23.6 / 56.2 | 24.3 / 51.2 | 22.6 / 55.4 |
| Llama-3.1-8B, 12/32 | 13.7 / 72.4 | 94.0 / 46.8 | 94.0 / 46.8 | 37.8 / 48.4 | 95.9 / 39.4 | 34.4 / 49.0 |
| Qwen3-14B, 10/40 | 15.0 / 73.0 | 48.2 / 49.2 | 41.9 / 51.4 | 27.2 / 49.1 | 21.9 / 53.0 | 21.9 / 54.4 |
| Qwen3.8-27B, 16/64 | 10.1 / 72.8 | 34.8 / 40.6 | 28.7 / 42.2 | 13.2 / 65.7 | 13.4 / 66.4 | 13.3 / 64.9 |
| Qwen3.8-27B, 24/64 | 10.1 / 72.8 | 42.4 / 38.6 | 36.9 / 42.2 | 17.1 / 57.4 | 15.7 / 55.5 | 16.0 / 57.9 |

(Bảng đầy đủ có thêm Qwen3-1.7B 10/28, Qwen3-14B 15/40, LaCo+R, Prune&Comp+R; trích đủ nằm trong Table 3 của paper.) Dòng "R-GS so với ReplaceMe" của paper: PPL giảm 30%, 21%, 20%, 66%, 44%, 63%, 48%, 21% trên 8 thiết lập của 4 họ nhỏ; 54% và 56% trên Qwen3.8-27B; **Avg accuracy** đổi thêm -1.5, +0.2, -0.2, +3.2, -1.8, +2.2, +3.0, +0.6, +22.7, +15.7 điểm. Tức: **WikiText-PPL cải thiện nhất quán, accuracy lẫn lộn khi mục tiêu là WikiText** (tăng rõ chỉ ở 27B và vài ngân sách nặng). Paper tóm tắt: R-GS có PPL thấp nhất ở 7/8 thiết lập; 20% tới 66% dưới ReplaceMe, 4% tới 29% dưới SLEB có repair; 20% tới 73% dưới Prune&Comp và 37% tới 97% dưới LaCo (cả hai có repair).

**Với mục tiêu tác vụ (R-GS-T, khối dưới của Table 3):** chọn bằng cross-entropy của đáp án đúng trên 200 mẫu train/tác vụ. Avg accuracy vs ReplaceMe cùng dữ liệu tác vụ: Qwen3-1.7B 47.1 vs 43.8 (+3.3) và 41.7 vs 40.9 (+0.9); Gemma 59.9 vs 45.8 (+14.1) và 51.2 vs 40.6 (+10.6); Llama 61.2 vs 58.9 (+2.3) và 51.9 vs 46.8 (+5.1); Qwen3-14B 59.5 vs 51.4 (+8.1) và 47.9 vs 38.8 (+9.1); Qwen3.8-27B 66.8 vs 42.2 (+24.6) và 59.8 vs 42.2 (+17.6). Paper: tốt hơn ReplaceMe ở cả 8 thiết lập của họ nhỏ (0.9 tới 14.1 điểm) và hơn ở cả 2 ngân sách của 27B (lên tới 25 điểm). Trên Gemma 8/34 giữ 86% độ chính xác mô hình nguyên so với 73% của ReplaceMe. Trên 27B giữ 92% (16/64) và 82% (24/64). Chú ý: PPL của R-GS-T thường **tệ hơn** (ví dụ Qwen3-1.7B 7/28: 64.5 vs 53.7 của ReplaceMe dùng WikiText); mục tiêu quyết định năng lực nào được giữ.

### 5.3 R-FS

So với tăng loss chính xác của mọi khối layer không critical: khối có R-FS thấp nhất nằm trong 5% khối tốt nhất ở **19/22** độ dài khối; cosine distance của ReplaceMe chỉ ở **2** (Fig. 5b). Tương quan Spearman với tăng loss repair: 0.53 tới 0.86 (R-FS), -0.21 tới 0.36 (cosine). Trong Table 3, khối R-FS (không đánh giá mô hình đã cắt) trong vòng 3% R-GS ở 4/8 thiết lập; ở ngân sách nặng (Gemma 13/34: PPL 97.6 vs 51.1; Llama 12/32: 95.9 vs 34.4) kém xa vì tối ưu không còn là một khối. Paper: R-FS là mặc định tốt khi một khối đủ, R-GS cho ngân sách nặng.

## 6. Phân tích / ablation

- **Ablation R-GS (Table 4, 8 thiết lập nhỏ):** tỉ số PPL so với R-GS (trung bình hình học / tệ nhất), accuracy trung bình, số lần đánh giá: R-GS 1.00 / 1.00 / 48.1 / 303. Bỏ repair trong tìm kiếm (SLEB+R): 1.19 / 1.41 / 48.0 / 303. Bỏ tìm kiếm, chấm một lần + repair: 1.21 / 1.73 / 46.5 / 34. Bỏ tìm kiếm, tiêu chí khối (ReplaceMe): 1.73 / 2.95 / 47.3 / 0. **Cắt theo R-GS nhưng bỏ repair khi triển khai:** 3.98 / 14.9 (PPL tăng gấp bốn). Cải tiến "shift" (dời nhóm) 0.995 / 1.00 / 48.4. Kết luận: repair *trong* tìm kiếm là thành phần quan trọng nhất; accuracy trung bình hầu như không đổi giữa các biến thể (46.5–48.4).
- **Calibration khác (mục 7.4, Table 6–7):** dùng nguyên cấu hình mặc định của ReplaceMe (fit trên hội thoại SlimOrca, không ridge): R-GS có PPL thấp hơn 16% tới 73% trên Llama và Gemma. Chọn bằng mục tiêu SlimOrca (rời với dữ liệu fit): PPL thấp hơn ở 7/8 thiết lập, accuracy ở 6/8.
- **Sau hồi phục (mục 7.5, Fig. 6):** C4 PPL của R-GS thấp hơn ReplaceMe ở mọi mô hình/ngân sách trước hồi phục; sau 200 bước LoRA trên C4 (3 seed) R-GS vẫn thấp hơn ở 7/8, lần lượt 16% tới 30% (Llama), 2.5% tới 19% (Gemma), 1% tới 2% (Qwen3-1.7B): khoảng cách co lại nhưng còn.
- **MMLU (Table 12):** mục tiêu chọn quyết định layer nào được giữ. MMLU 5-shot cần các layer giữa ít ảnh hưởng PPL WikiText: trên Llama và Qwen3-14B, tập WikiText của R-GS và SLEB bỏ chúng và chỉ đạt 24% tới 28% (gần ngẫu nhiên 25%), trong khi ShortGPT (khối sâu) giữ 67.1% và 68.4% ở 25%. Với mục tiêu MMLU, R-GS bỏ layer sâu và đạt cao nhất ở 7/8 thiết lập (ví dụ Qwen3-14B 15 layer 53.3% so với tối đa 26.7% của phương pháp khác). Khoảng cách giữa PPL và accuracy: Phụ lục F cũng cho thấy WikiText-objective bỏ layer giữa mà WinoGrande cần (WinoGrande Llama: ShortGPT/ReplaceMe 71.1/69.7 so với 54.4 của R-GS).
- **Cơ chế siêu cộng tính (Phụ lục C, Table 5):** ba giả thuyết bị loại (thay layer bằng cập nhật trung bình vẫn còn số hạng cặp; "các layer sau khôi phục" tương quan Spearman chỉ +0.17 so với +0.72 của khoảng cách; độ cong hàm loss; tách logit). Chỉ **tương đồng chức năng** có hỗ trợ một phần (CKA tuyến tính của hai cập nhật, tương quan riêng phần với số hạng cặp: Qwen3-1.7B +0.37, Llama +0.36, Qwen3-14B +0.11, Gemma -0.06). Kết luận paper: tương tác đến từ phép tính bị mất, chưa giải thích đầy đủ.
- **Tìm kiếm không repair (Phụ lục D):** cải tiến SLEB (lặp + dời) tìm được tối ưu ở 16 ngân sách trong mọi seed (+2 nữa ở một phần seed).
- **Kích thước calibration:** với 16 chuỗi, tối ưu trên calibration trùng tối ưu trên test ở 13/17 thiết lập; 64 chuỗi: 100% ở Qwen3-1.7B và Gemma.
- **Fit repair khác:** hai cách fit map sau trên mạng đã repair không cải thiện; fit hướng tới activation mạng nguyên làm tập nhiều run tệ hơn 1% tới 16%; fit hướng tới đầu ra của run, tính trên input của mạng đã repair: tốt hơn khoảng 5% trên Qwen3-1.7B, trong ±1.5% trên Llama. Chênh lệch nhỏ hơn nhiều so với chênh lệch giữa các cách chọn.
- **Chi phí suy luận (Table 11, bf16, RTX A6000):** bỏ 25% layer ~1.3x tăng tốc (prefill 2048 token, decode 256 token ở batch 1 và 16), 37.5% ~1.55x. Ví dụ Llama-3.1-8B 8/32: trọng số 16.1 -> 12.6 GB, KV 128 -> 96 KB/token, tăng tốc 1.32 (prefill), 1.30 (batch 1), 1.31 (batch 16); 12/32: 16.1 -> 10.8 GB, 1.58, 1.54, 1.55. Qwen3.8-27B bỏ 24/64: 53.8 -> 35.6 GB, vừa một GPU 48GB. Gemma-3 không gộp được map (chuẩn hoá post-feedforward), nên có thêm một ma trận $d\times d$ cho mỗi run.

## 7. Giới hạn

**Do tác giả nêu:**
- Liệt kê chính xác chỉ ở độ chi tiết 2 layer đến 6 nhóm, hoặc 1 layer đến 3 layer; ở ngân sách nặng chỉ so sánh các phương pháp với nhau.
- Kết luận cấu trúc là về **perplexity**: ở độ chi tiết 2 layer, tối ưu PPL là tập chính xác nhất (hoặc trong 1 sai số chuẩn) ở 10/14 trường hợp; ở 4 trường hợp còn lại có tập khác chính xác hơn 2–7 điểm.
- Chỉ thử một repair (LS của ReplaceMe); các repair mới (LinearPatch, Prune&Comp, shifted maps) chưa kiểm; không so với BlockPruner và SubFit.
- Hồi phục chỉ một độ dài: 200 bước LoRA. Chỉ tiếng Anh và tác vụ trắc nghiệm; đa ngôn ngữ và sinh mở chưa kiểm.
- Cấu hình ReplaceMe mẫu fit trên hội thoại tới 1.024 token với mô hình 4-bit; bài này fit trên khối 256 token ở bfloat16 (khác với bài gốc).

**Do chúng ta nhận thấy:**
- Chọn và đánh giá chủ yếu bằng WikiText-2 (16 chuỗi x 256 token): rất nhỏ, và mục tiêu WikiText lẫn lộn về accuracy (Table 3). Kết luận "layout tối ưu thành khối liền khi có repair" được chứng minh trên PPL, 4 họ, 2 nhóm layer; chưa biết nó giữ với CER của OCR.
- Mọi nhóm đầu đều critical và luôn được giữ; dự án của bạn nếu có ViT/decoder lai với cấu trúc khác thì "critical" có thể phân bố khác.
- Maps được fit độc lập từ mô hình nguyên (cache); paper thừa nhận các run được repair vẫn tương tác (R² giảm), nên repair chưa "hấp thụ" mọi tương tác.
- Đề xuất bậc hai R-FS dựa trên Fisher kinh nghiệm hạng một; chỉ xác nhận thực nghiệm (Fig. 5), và nó thất bại ở ngân sách nặng.
- Các tên mô hình/tham chiếu trong bài khá mới (2026); chúng tôi chưa kiểm chứng độc lập các số liệu.

## 8. Liên hệ với các bài khác

- **36 ReplaceMe:** bài này nhận repair của nó làm đối tượng nghiên cứu và chứng minh tiêu chí chọn khối (cosine, Eq. 4 của ReplaceMe) lệch khỏi tối ưu có repair 11% tới 189%.
- **34 SLEB:** R-GS = SLEB + repair trong vòng lặp, cùng chi phí; SLEB cũng đã được chứng minh chọn tối ưu cắt-thuần ở 13/20 ngân sách.
- **31 ShortGPT, 32 Deeper layers, 35 LaCo:** các kiểu chọn proxy hoặc khối liền; bài này cho thấy "khối liền" chỉ gần tối ưu khi có repair (nhất quán với lựa chọn khối liền của 32 và ReplaceMe), còn ShortGPT chọn gần-liền nên tệ nếu không repair.
- **33 Shortened LLaMA:** bằng chứng tốc độ batch-1; bài này xác nhận tăng tốc ~1.3x/1.55x khi cắt 25%/37.5%.
- **01 OBD/OBS, 22 OBC, 23 SparseGPT:** R-FS dùng cùng họ khai triển bậc hai/Fisher (OBD/OBS, Fisher pruning, Taylor importance) nhưng áp dụng cho *sai số còn lại sau repair*.
- **27 LLM Surgeon, 07 Molchanov importance:** Fisher kinh nghiệm và Taylor importance liên quan.
- **39–40 Minitron:** hồi phục bằng distillation sau khi chọn; bài này chỉ thử 200 bước LoRA.

## 9. Ứng dụng cho dự án (suy luận)

**Mức liên quan: rất cao.**

- **Nối với $D_{lin}$/$D_{res}$.** Trong bài, $\delta_t=y_t+m_tT_{a,b}-z_t$ là phần dư *sau* repair tuyến tính tối ưu: gần với $D_{res}$ của bạn (nếu $D_{lin}$ = phần hư hại bị xoá bởi map tối ưu). R-FS đo $D_{res}$ **theo thước loss** (trọng số Fisher). Ngược lại, thước Euclid của LS giữ phần "sửa được". Hiện tượng bạn thấy (cắt neuron MLP phục hồi kém hơn cắt layer) có thể hiểu: khối layer có phần dư repair nhỏ đối với một map tuyến tính, còn cắt neuron thì không có map $d\times d$ tương đương. Đây là cách đọc của chúng ta; bài không so sánh cắt neuron.
- **Bài học then chốt: đừng chọn layer bằng điểm số của mô hình chưa repair, rồi mới repair.** Nếu pipeline của bạn sẽ repair bằng map tuyến tính hoặc distillation, hãy chấm các ứng viên **sau repair** (hoặc dùng R-FS như bộ lọc rẻ), vì tập tối ưu có thể khác. Với distillation (không phải map LS), chưa rõ tối ưu có cùng cấu trúc "khối liền" hay không; đây là câu hỏi mở đáng thử.
- **Mục tiêu quyết định giữ cái gì.** Table 12 cho thấy WikiText-PPL giữ lại khác với MMLU hoặc WinoGrande. Với OCR, mục tiêu chọn layer nên là loss/CER OCR thật (hoặc KL với teacher 4B), không phải PPL văn bản chung. Đây là điểm bạn phải tự kiểm chứng.
- **Ngân sách:** mục tiêu 0.8B -> 0.3B cắt ~60% layer-parameter, nặng hơn mọi thiết lập của bài (tối đa 37.5%); khi đó tối ưu là nhiều đoạn (Gemma 13/34: một khối 8 layer + 5 layer rời), nên R-GS có lợi hơn R-FS.
- **Phù hợp hybrid:** có kiểm chứng Qwen3.8-27B (3 linear-attention : 1 full-attention); R-GS hơn ReplaceMe ở đó vì ReplaceMe chọn các layer đầu. Kiến trúc GatedDeltaNet của bạn gần hơn họ này so với Transformer thuần; xem đó là bằng chứng gián tiếp, chưa phải trực tiếp.
- **ViT:** bài không thử ViT; kết luận về ViT 12 block là suy luận.
- **Chi phí:** R-GS tốn $\sim n\cdot L$ lần đánh giá mô hình cắt; decoder ~20 layer thì rẻ.

## 10. Thí nghiệm gợi ý

1. **Liệt kê nhỏ trên chính mô hình của bạn:** decoder LightOnOCR chia nhóm 2 layer (~10 nhóm), liệt kê mọi tập tới 4 nhóm (khoảng vài trăm tập), có và không có repair LS; đo tăng loss OCR (CE token) và CER trên 200 trang giữ riêng. Kiểm tra liệu tối ưu có tách đoạn không repair và thành khối liền có repair, và liệu ranh giới bạn thấy giữa D_lin và D_res đi cùng cấu trúc đó.
2. **Chấm điểm bằng KL tới teacher 4B** thay cho PPL WikiText; so tập chọn bởi R-GS-với-KL, R-FS và cosine (ReplaceMe) bằng CER sau 200 bước distillation, để xem "repair" kiểu distillation có đổi tối ưu như repair LS không.
3. **Tách $\delta$ thành $D_{lin}$ và $D_{res}$ rồi tính R-FS** (một forward + backward) trên từng ứng viên khối: kiểm tra liệu R-FS (cân Fisher) xếp hạng ứng viên tốt hơn $\lVert\delta\rVert_2$ (Euclid) theo CER sau distill.

## 11. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Run | tập cực đại các nhóm bị bỏ kề nhau, mỗi run một map $T$ |
| $T_{a,b}$ (Eq. 1) | ridge LS: $\arg\min \lVert M_{a-1}T-(H_b-Y_{a-1})\rVert_F^2+\lambda\lVert T\rVert_F^2$, gộp vào `down_proj` |
| $\mu(S)$ | số hạng tương tác Möbius; cộng tính khi mọi $\mu=0$ |
| Critical group | bỏ riêng làm PPL tăng hơn gấp đôi, luôn được giữ ở tối ưu |
| Siêu cộng tính cục bộ | bỏ hai layer gần nhau tốn hơn tổng riêng; suy giảm trong 3–6 layer |
| R-GS | tham lam đánh giá mỗi ứng viên sau repair; chi phí = SLEB |
| R-FS | $s=\sum_t(g_t\cdot\delta_t)^2$, phần dư repair cân bằng Fisher, một forward + backward |
| R-GS-T | R-GS với mục tiêu tác vụ (CE đáp án đúng trên 200 mẫu) |

## 12. Kết luận ngắn

Bài đo được tối ưu chính xác của depth pruning và cho thấy tập layer tốt nhất phụ thuộc vào bước repair đi kèm: không repair thì tách đoạn, có repair tuyến tính thì thành khối liền. Chọn layer dưới đúng repair sẽ triển khai hạ PPL 20–66% so với ReplaceMe ở cùng chi phí với SLEB, và mục tiêu chọn (PPL, tác vụ, MMLU) quyết định năng lực nào còn lại. Giới hạn: một loại repair, đánh giá trắc nghiệm/PPL, không có tác vụ sinh hay OCR.

Link gốc: https://arxiv.org/abs/2610.30552
