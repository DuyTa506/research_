# Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (2020)

**Tác giả:** Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret (Idiap/EPFL, U. Washington, U. Geneva)  
**Venue / arXiv:** ICML 2020 (PMLR 119); arXiv:2006.16236 (v3, 8/2020)  
**Nguồn:** https://arxiv.org/abs/2006.16236  
**Chủ đề:** Linear attention bằng kernel feature map, trạng thái hồi quy ma trận $S_t=S_{t-1}+\phi(k_t)v_t^\top$, tương đương Transformer-RNN, suy luận tự hồi quy chi phí hằng số mỗi bước.  
**Vị trí trong lộ trình:** Cần đọc 01 (Attention Is All You Need) trước, đặc biệt công thức softmax attention và causal mask. Bài này là **gốc của cả dòng 18 → 19 → 20 → 22 → 23 → 24** mà decoder GatedDeltaNet của bạn thuộc về: mọi bài sau chỉ thay *cách cập nhật trạng thái $S_t$* (thêm suy giảm, thêm cổng phụ thuộc dữ liệu, thêm delta rule). Bài trước (17 Longformer) cắt cặp attention; bài này đổi bản chất phép toán để nén toàn bộ quá khứ.

> Tài liệu học tập bằng tiếng Việt, ghi chú diễn giải, **không phải bản dịch**. Các suy luận áp dụng cho dự án và mọi con số đánh dấu "tự tính" là của chúng ta, không phải kết luận của paper. Quy ước ký hiệu: ghi chú này dùng **vector cột** ($q_t,k_t\in\mathbb R^{d_k}$, $v_t\in\mathbb R^{d_v}$, $S_t\in\mathbb R^{d_k\times d_v}$); paper dùng vector hàng nên công thức của paper là chuyển vị của công thức ở đây.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết: từ softmax đến RNN](#3-phương-pháp-chi-tiết-từ-softmax-đến-rnn)
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

**Hai nỗi đau của Transformer tự hồi quy (bài 01):**

1. **Huấn luyện/prefill:** attention tính ma trận $n\times n$ nên thời gian và bộ nhớ là $O(n^2)$.
2. **Sinh từng token (decode):** token thứ $t$ phải attend tới $t$ khóa trước đó; ngay cả khi có KV cache, chi phí một bước là $O(t)$, tổng cả chuỗi $O(T^2)$, và KV cache tăng tuyến tính theo $t$.

**Các hướng giảm chi phí trước đó (paper điểm lại):** cắt tỉa, phân rã trọng số, lượng tử hóa, chưng cất; Transformer-XL và Adaptive Span (tăng ngữ cảnh nhưng cùng độ phức tạp tiệm cận); Sparse Transformer ($O(N\sqrt N)$) và Reformer ($O(N\log N)$, dùng LSH, bắt buộc $K=Q$ nên không dùng được cho decoding). Điểm chung: **các phương pháp này tăng tốc huấn luyện nhưng không tăng tốc suy luận tự hồi quy**. Bài này nhắm vào cả hai.

**Khái niệm nền.**

- *Kernel / feature map $\phi$:* một hàm đưa vector sang không gian khác sao cho tích vô hướng trong không gian mới $\phi(x)^\top\phi(y)$ bằng một hàm tương tự $k(x,y)$ ta muốn dùng. Ví dụ kernel đa thức bậc 2 có $\phi$ hữu hạn chiều; kernel mũ (của softmax) cần $\phi$ vô hạn chiều.
- *Tính kết hợp của nhân ma trận:* $(AB)C=A(BC)$ — kết quả giống nhau nhưng chi phí khác nhau. Đây là toàn bộ "mánh" của bài.
- *Causal mask:* chặn token $i$ nhìn $j>i$.

## 2. Ý tưởng chính

Softmax attention $\mathrm{softmax}(QK^\top)V$ khó tái sắp xếp vì softmax bao quanh tích $QK^\top$. Nếu thay hàm tương tự $\exp(q^\top k)$ bằng một **kernel phân rã được** $\phi(q)^\top\phi(k)$, ta có thể *kết hợp lại phép nhân* thành $\phi(Q)\big(\phi(K)^\top V\big)$: tính trước một ma trận nhỏ $d\times d$ rồi dùng chung cho mọi query, nên chi phí thành $O(N)$ thay vì $O(N^2)$. Với causal mask, ma trận nhỏ này là một **tổng tích lũy** $S_t=\sum_{i\le t}\phi(k_i)v_i^\top$, có thể cập nhật mỗi bước bằng một phép cộng: đó chính là **một mạng hồi quy (RNN) với trạng thái là ma trận**. Vì vậy "mọi Transformer có causal mask đều có thể viết thành RNN", và sinh token tốn chi phí hằng số mỗi bước, bộ nhớ hằng số theo độ dài.

## 3. Phương pháp chi tiết: từ softmax đến RNN

### 3.1 Công thức softmax tổng quát hóa

Với $Q,K\in\mathbb R^{N\times D}$, $V\in\mathbb R^{N\times M}$, softmax attention (có chuẩn hóa $\sqrt D$):

$$
V'_i=\frac{\sum_{j=1}^{N}\mathrm{sim}(Q_i,K_j)\,V_j}{\sum_{j=1}^{N}\mathrm{sim}(Q_i,K_j)},\qquad \mathrm{sim}(q,k)=\exp\!\Big(\tfrac{q^\top k}{\sqrt D}\Big).
$$

Chỉ ràng buộc duy nhất để công thức này là một "attention" hợp lệ: **$\mathrm{sim}\ge0$** (để các trọng số cộng lại thành phân phối). Mọi kernel $k(x,y)\ge0$ đều dùng được.

### 3.2 Chuyển sang kernel và tái kết hợp (không mask)

Nếu $\mathrm{sim}(q,k)=\phi(q)^\top\phi(k)$ thì (viết cột; $\phi$ áp dụng theo từng hàng của $Q,K$):

$$
V'_i=\frac{\phi(Q_i)^\top\sum_{j=1}^N \phi(K_j)V_j^\top}{\phi(Q_i)^\top\sum_{j=1}^N\phi(K_j)}.
$$

Hai tổng không phụ thuộc $i$, nên tính **một lần** rồi dùng lại cho mọi $i$:

$$
\boxed{(\phi(Q)\phi(K)^\top)V\;=\;\phi(Q)\big(\phi(K)^\top V\big)}.
$$

**Chi phí.** Softmax: $O(N^2\max(D,M))$. Linear với feature map $C$ chiều: tính $\phi(K)^\top V\in\mathbb R^{C\times M}$ tốn $O(NCM)$, nhân với $\phi(Q)$ tốn $O(NCM)$; tổng $O(NCM)$.

**Chọn feature map.** Kernel mũ cần $\phi$ vô hạn chiều nên không linearize chính xác được. Kernel đa thức bậc 2 có $\phi$ hữu hạn với chi phí $O(ND^2M)$ — chỉ có lợi khi $N>D^2$. Trong thực nghiệm paper dùng:

$$
\boxed{\phi(x)=\mathrm{elu}(x)+1}
$$

với $\mathrm{elu}(x)=x$ nếu $x>0$, $\alpha(e^x-1)$ nếu $x\le0$ ($\alpha=1$). Vì $\mathrm{elu}(x)>-1$, nên $\phi(x)>0$ ⇒ similarity dương. Chọn `elu` thay vì `relu` để **không triệt tiêu gradient khi $x<0$**. Với $\phi$ này, $C=D$ và chi phí là $O(NDM)$.

### 3.3 Thêm causal mask

Mask làm $V'_i$ chỉ tổng đến $j\le i$:

$$
V'_i=\frac{\phi(Q_i)^\top\sum_{j=1}^{i}\phi(K_j)V_j^\top}{\phi(Q_i)^\top\sum_{j=1}^{i}\phi(K_j)}.
$$

**Điểm tinh tế (rất quan trọng cho các bài sau).** Dạng song song $\big(\phi(Q)\phi(K)^\top\odot M\big)V$ với $M$ là mask tam giác **không** tái kết hợp được thành $\phi(Q)(\phi(K)^\top V)$, vì $M$ nằm *giữa* hai phép nhân. Nhưng tổng tích lũy theo $i$ thì tính được dần dần. Định nghĩa

$$
S_i=\sum_{j=1}^{i}\phi(K_j)V_j^\top\in\mathbb R^{C\times M},\qquad z_i=\sum_{j=1}^{i}\phi(K_j)\in\mathbb R^{C}.
$$

Khi đó:

$$
\boxed{V'_i=\frac{S_i^\top\phi(Q_i)}{\phi(Q_i)^\top z_i}}.
$$

### 3.4 "Transformer là RNN": dạng hồi quy

$S_i$ và $z_i$ tính từ $S_{i-1},z_{i-1}$ bằng một phép cộng:

$$
\begin{aligned}
S_0&=0,\quad z_0=0,\\
\boxed{S_i}&\boxed{=S_{i-1}+\phi(k_i)\,v_i^\top},\\
z_i&=z_{i-1}+\phi(k_i),\\
o_i&=\frac{S_i^\top\phi(q_i)}{\phi(q_i)^\top z_i}.
\end{aligned}
$$

(Paper viết đủ một lớp: $s_i=s_{i-1}+\phi(x_iW_K)(x_iW_V)^\top$, $y_i=f_l\big(\tfrac{\phi(x_iW_Q)^\top s_i}{\phi(x_iW_Q)^\top z_i}+x_i\big)$, với $f_l$ là FFN. Ở đây RNN có **hai trạng thái ẩn**: "attention memory" $s$ và "normalizer memory" $z$, và "thời gian" của RNN là *chiều chuỗi*, khác Universal Transformer vốn hồi quy theo chiều sâu.)

Ý nghĩa: $S_i$ là một **bộ nhớ liên tưởng** (associative memory, "fast weights"): mỗi cặp $(k_j,v_j)$ được ghi bằng tích ngoài $\phi(k_j)v_j^\top$ (hạng 1), và truy vấn $q$ đọc bằng phép nhân $S^\top\phi(q)$ — tức là cộng các $v_j$ có trọng số $\phi(q)^\top\phi(k_j)$.

### 3.5 Ví dụ số nhỏ (đối chiếu hai dạng)

Lấy $\phi=$ identity, bỏ chuẩn hóa, $d_k=d_v=2$. Cho $k_1=(1,0)^\top, v_1=(2,3)^\top$; $k_2=(0,1)^\top,v_2=(4,5)^\top$; $q_2=(1,1)^\top$.

*Dạng hồi quy:*
$S_1=k_1v_1^\top=\begin{pmatrix}2&3\\0&0\end{pmatrix}$, $S_2=S_1+k_2v_2^\top=\begin{pmatrix}2&3\\4&5\end{pmatrix}$, $o_2=S_2^\top q_2=(2+4,\ 3+5)^\top=(6,8)^\top$.

*Dạng song song:* $o_2=\sum_{j\le2}(q_2^\top k_j)v_j=1\cdot(2,3)+1\cdot(4,5)=(6,8)$. ✓ Hai dạng trùng nhau.

### 3.6 Gradient với bộ nhớ hằng số

Vì dạng hồi quy cần lưu mọi $S_i$ để backprop, bộ nhớ tăng thêm $\max(D,M)$ lần. Paper dẫn các gradient dưới dạng tổng tích lũy (với $\bar V_i$ là tử số, $\nabla_{\bar V_i}L$ gradient tương ứng):

$$
\nabla_{\phi(Q_i)}L=\nabla_{\bar V_i}L\Big(\sum_{j\le i}\phi(K_j)V_j^\top\Big)^{\!\top},\quad
\nabla_{\phi(K_i)}L=\Big(\sum_{j\ge i}\phi(Q_j)\nabla_{\bar V_j}L^\top\Big)V_i,\quad
\nabla_{V_i}L=\Big(\sum_{j\ge i}\phi(Q_j)\nabla_{\bar V_j}L^\top\Big)^{\!\top}\phi(K_i).
$$

(công thức 13-15 của paper, ký hiệu theo hàng; ở đây chỉ nêu cấu trúc). Cả ba là tổng tích lũy (trước hoặc sau) nên tính trong thời gian tuyến tính, bộ nhớ hằng số theo $N$ (phụ thuộc $\max(C,M)$), cài đặt bằng khoảng 200 dòng CUDA. Pseudo-code forward của paper (thu gọn):

```
def forward(phiQ, phiK, V):     # tử số, chưa chia cho z
    S = zeros(C, M)
    for i in 1..N:
        S = S + outer(phiK[i], V[i])       # S_i = S_{i-1} + phi(k_i) v_i^T
        Vbar[i] = S.T @ phiQ[i]
    return Vbar                              # chi phí O(N C M), bộ nhớ O(max(C,M))
```

### 3.7 Huấn luyện song song và suy luận tuần tự

- **Huấn luyện**: biết toàn bộ chuỗi đích nên song song được theo lớp (FFN và attention), như Transformer.
- **Suy luận**: token $i$ là đầu vào của bước $i+1$ nên bắt buộc tuần tự; softmax-Transformer tốn $O(i)$ (thực ra bình phương nếu không cache), còn linear transformer chỉ cần giữ $(S,z)$ và cập nhật: **thời gian và bộ nhớ mỗi bước không đổi**.

Lưu ý quan trọng ngầm: paper chưa trình bày dạng *chunkwise* — thuật toán tuần tự trên GPU bị giới hạn bởi băng thông; việc tăng tốc huấn luyện thực sự đến ở bài 19/20.

## 4. Thí nghiệm và kết quả chính

Baseline: softmax Transformer và Reformer (lsh-$X$ = $X$ vòng băm). Mã PyTorch + kernel CUDA cho gradient.

### 4.1 Chi phí tính toán/bộ nhớ (GTX 1080 Ti 11GB, $N\in\{2^9,\dots,2^{16}\}$)

Softmax tăng bình phương, Reformer và linear tăng tuyến tính; tối đa chạy được: softmax 4,096; lsh-4/lsh-8 16,384. Paper nói phương pháp của họ nhanh hơn và tốn bộ nhớ ít hơn các baseline ở mọi cấu hình (Hình 1; số liệu từng điểm không có trong text trích).

### 4.2 Bài toán copy (hội tụ)

Chuỗi tối đa 128, 10 ký hiệu phân cách bằng ký hiệu đặc biệt, 4 lớp 8 head, batch 64, RAdam LR $10^{-3}$ rồi $10^{-4}$ sau 3000 bước. Linear hội tụ êm và đạt cùng loss cuối với softmax; Reformer có loss cao hơn do nhiễu băm.

### 4.3 Sinh ảnh từng pixel

| MNIST (784 pixel) | bits/dim | ảnh/giây |
|---|---|---|
| Softmax | 0.621 | 0.45 (1×) |
| LSH-1 | 0.745 | 0.68 (1.5×) |
| LSH-4 | 0.676 | 0.27 (0.6×) |
| **Linear** | 0.644 | **142.8 (317×)** |

| CIFAR-10 (huấn luyện 1 tuần/1 GPU) | bits/dim | ảnh/giây |
|---|---|---|
| Softmax | 3.47 | 0.004 (1×) |
| LSH-1 | 3.39 | 0.015 (3.75×) |
| LSH-4 | 3.51 | 0.005 (1.25×) |
| **Linear** | 3.40 | **17.85 (4,462×)** |

Cấu hình MNIST: 8 lớp, 8 head, embedding 256, FFN gấp 4 lần, mixture 10 logistics, 250 epoch, batch 10. CIFAR-10: 16 lớp, softmax chỉ chạy batch 1 trên P40 24GB, linear/Reformer batch 4. Linear hoàn thành số epoch nhiều gấp 3 lần softmax trong cùng thời gian (nên perplexity tốt hơn). Paper nhấn mạnh bộ nhớ khi sinh là hằng số (chỉ giữ $s_i,z_i$), có thể sinh đồng thời 10,000 ảnh MNIST trên một GPU.

### 4.4 Nhận dạng giọng nói (WSJ 80 giờ, CTC, không tự hồi quy)

| Mô hình | PER dev ↓ | Thời gian/epoch (s) |
|---|---|---|
| Bi-LSTM | 10.94 | 1047 |
| Softmax | 5.12 | 2711 |
| LSH-4 | 9.33 | 2250 |
| **Linear** | 8.08 | **824** |

Linear nhanh hơn softmax hơn 3 lần mỗi epoch nhưng PER kém hơn softmax (8.08 vs 5.12); vượt LSTM và Reformer.

## 5. Phân tích / ablation

Paper **gần như không có ablation** về feature map. Các quan sát trong bài:

- Kernel đa thức bậc 2 cho chi phí $O(ND^2M)$, chỉ lợi khi $N>D^2$ (phân tích, không thí nghiệm).
- Nếu dùng kernel cho similarity dương lên query/key thì linear attention hội tụ bình thường (nhận xét trong Related Work).
- Về chất lượng: khoảng cách chất lượng với softmax *xuất hiện rõ* ở ASR (8.08 vs 5.12 PER) nhưng nhỏ ở sinh ảnh (MNIST 0.644 vs 0.621 bits/dim). Chưa có thí nghiệm ngôn ngữ tự nhiên quy mô lớn.

## 6. Giới hạn

**Do tác giả nêu / ngụ ý:**
- Lựa chọn feature map còn để ngỏ ("choice of feature map for linear attention" là hướng tương lai); ý định gợi ý: xấp xỉ RBF bằng random Fourier features để dùng được mô hình tiền huấn luyện softmax.
- Các thí nghiệm dùng chuỗi vừa phải (tối đa vài nghìn phần tử) và chỉ vài tác vụ (copy, ảnh, ASR); chưa có language modeling văn bản quy mô lớn.

**Do chúng ta nhận xét (suy luận, bài sau xác nhận):**
- **Không có cơ chế quên.** $S_i$ chỉ cộng dồn, không bao giờ giảm; các bài 19-20 thêm suy giảm $\gamma$ hoặc cổng $\alpha_t$, và bài 23-24 thêm delta rule để *ghi đè* thay vì chỉ cộng.
- **Dung lượng bị chặn.** $S\in\mathbb R^{d_k\times d_v}$ chỉ lưu tối đa $\sim d_k$ cặp độc lập; khi chuỗi dài hơn $d_k$, các $(k,v)$ chồng chập gây nhiễu, đặc biệt hại cho tác vụ tra cứu chính xác (recall). Bài 20 (Bảng 3) và bài 21 đo điều này.
- **Không có mã hóa vị trí.** Tổng $\sum\phi(k_j)v_j^\top$ không phân biệt thứ tự; RetNet (19) nhận xét "linear attention struggles to effectively encode position information".
- **Tuần tự không tận dụng tensor core.** Cài đặt bằng vòng lặp cộng từng phần tử, tính toán thấp so với truy cập bộ nhớ (low arithmetic intensity), chậm hơn lý thuyết (nội dung bài 20).
- Mẫu số $\phi(q)^\top z$ có thể bất ổn số học (các bài sau, ví dụ TransNormer/RetNet, bỏ hoặc thay bằng chuẩn hóa nhóm).

## 7. Liên hệ với các bài khác

- **01:** thay `softmax(QK^T)` bằng kernel; giữ nguyên $Q,K,V$ và FFN.
- **17 Longformer:** hướng "cắt cặp" (giữ softmax) đối lập với "đổi kernel + nén".
- **19 RetNet:** thêm suy giảm hằng $\gamma$: $S_n=\gamma S_{n-1}+k_nv_n^\top$, bỏ mẫu số, thêm xPos/rotation, giới thiệu dạng *chunkwise*.
- **20 GLA:** suy giảm phụ thuộc dữ liệu $S_t=\mathrm{Diag}(\alpha_t)S_{t-1}+k_tv_t^\top$ và thuật toán chunkwise I/O-aware.
- **21 Mamba / 22 Mamba-2:** cùng họ hồi quy tuyến tính với cổng/suy giảm chọn lọc; 22 chỉ ra "đối ngẫu" với linear attention.
- **23 DeltaNet / 24 Gated DeltaNet:** thay phép *cộng* bằng *delta rule* (sửa lỗi dự đoán $v-S^\top k$ rồi ghi). Đây là bước cuối dẫn tới lớp GatedDeltaNet trong decoder của bạn.
- **44 FlashAttention:** cùng vấn đề băng thông bộ nhớ nhưng giữ softmax.

## 8. Ứng dụng cho dự án (suy luận)

Đây là bài **nền tảng** để hiểu các lớp GatedDeltaNet (GDN) của LightOnOCR-3-0.8B. Cần nhớ: GDN là *họ hồi quy trạng thái ma trận* mở rộng từ phép cộng tích ngoài ở đây; các chi tiết GDN cụ thể sẽ đọc ở bài 23-24.

**Chi phí và kích thước trạng thái (tự tính).** Giả định $d_{model}=1024$, 24 lớp decoder, 16 head với $d_k=d_v=64$ mỗi head, $T=1500$, batch 1, bf16 cho KV, fp32 cho trạng thái hồi quy.

| Đại lượng | Công thức | Giá trị |
|---|---|---|
| KV cache full attention ($d_{kv}=1024$) | $2\cdot L\cdot d_{kv}\cdot T\cdot 2\,\mathrm{B}$ | $\approx147$ MB, tăng theo $T$ |
| Trạng thái linear attention | $L\cdot H\cdot d_kd_v$ phần tử | $24\cdot16\cdot64\cdot64\approx1.57$M phần tử $\approx6.3$ MB fp32, **không đổi theo $T$** |
| FLOPs prefill attention-core / lớp, full (không mask) | $4T^2d$ | $\approx9.2$ GFLOPs |
| FLOPs prefill dạng chunkwise ($C=64$) / lớp | mỗi head: $2TCd_k+2TCd_v+2Td_kd_v+2Td_kd_v$ | $\approx0.79$ GFLOPs (khoảng 12× ít hơn full attention không mask; khoảng 6× nếu full attention chỉ tính nửa tam giác causal) |
| FLOPs decode / token / lớp (cập nhật + đọc) | $\approx 4Hd_kd_v$ (ghi + đọc trạng thái) | $\approx 0.26$ MFLOPs, độc lập $t$ |

(Dạng chunkwise lấy từ bài 19/20, không có trong paper này.)

Nhận định (suy luận): ở $T=1.5$K, FLOPs không phải rào cản chính; lợi ích rõ nhất của lớp hồi quy ở **decode batch 1** là *không phải đọc KV cache* mỗi bước và *bộ nhớ hằng số*. Với GPU ≤ 8 GB, tiết kiệm ~140 MB KV không lớn so với trọng số 0.8B bf16 (~1.6 GB), nhưng tương đối lớn hơn khi mô hình co về 0.3B/0.1B.

**Về prune/distill.** (i) Mỗi head hồi quy có một "bộ nhớ" $d_k\times d_v$; cắt $d_k$ hoặc $d_v$ thu nhỏ trực tiếp dung lượng nhớ — khả năng cao tổn hại nhiều hơn cắt cùng tỷ lệ chiều ở MLP (suy đoán, có thể liên quan đến quan sát của bạn rằng cắt neuron khó phục hồi; chưa kiểm chứng). (ii) Nếu muốn khởi tạo một lớp linear-attention từ lớp softmax đã huấn luyện, paper gợi ý random Fourier features để xấp xỉ RBF/softmax, nhưng chưa thử (hướng mở). (iii) Với OCR: mô hình phần lớn "sao chép" từ ảnh sang văn bản; tra cứu chính xác trên *văn bản đã sinh* ít quan trọng hơn so với nhìn lại *token ảnh*. Nếu token ảnh nằm trong chuỗi bối cảnh của decoder, các lớp hồi quy phải nén chúng vào $S$ — đây là lý do hybrid giữ vài lớp full attention (suy luận; cần kiểm tra kiến trúc thật).

## 9. Thí nghiệm gợi ý

1. **Tự dựng cặp song song/hồi quy:** viết 30 dòng PyTorch kiểm tra $\big(\phi(Q)\phi(K)^\top\odot M\big)V$ bằng đúng $\phi(Q)S_t$ lặp (sai số $<10^{-6}$ ở fp32). Đây là bài kiểm tra cơ bản cho mọi bài 19-24.
2. **Đo dung lượng nhớ:** huấn luyện nhỏ trên tác vụ copy/associative recall với $d_k\in\{16,32,64\}$, số cặp tăng dần; vẽ độ chính xác theo (số cặp)/$d_k$ để thấy điểm "bão hòa" — chuẩn bị cho việc cắt $d_k$ ở GDN.
3. **So hai feature map:** `elu+1` vs `relu` vs bình phương trên cùng mô hình nhỏ để cảm nhận độ nhạy của hội tụ.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức |
|---|---|
| Attention tổng quát | $V'_i=\frac{\sum_j\mathrm{sim}(Q_i,K_j)V_j}{\sum_j\mathrm{sim}(Q_i,K_j)}$, $\mathrm{sim}\ge0$ |
| Softmax | $\mathrm{sim}=\exp(q^\top k/\sqrt D)$ |
| Kernel hóa | $\mathrm{sim}=\phi(q)^\top\phi(k)$ |
| Tái kết hợp (không mask) | $(\phi(Q)\phi(K)^\top)V=\phi(Q)(\phi(K)^\top V)$ |
| Feature map | $\phi(x)=\mathrm{elu}(x)+1$ |
| Trạng thái | $S_i=S_{i-1}+\phi(k_i)v_i^\top$, $z_i=z_{i-1}+\phi(k_i)$ |
| Đầu ra | $o_i=S_i^\top\phi(q_i)\,/\,\phi(q_i)^\top z_i$ |
| Chi phí | thời gian $O(NCM)$; bộ nhớ $O(N\max(C,M))$ khi huấn luyện; $O(CM)$ mỗi bước khi suy luận |
| Bất tiện | mask $M$ chặn tái kết hợp ⇒ cần dạng tổng tích lũy / hồi quy |

## 11. Kết luận ngắn

Bài báo cho thấy, chỉ bằng việc thay softmax bằng kernel phân rã, attention có causal mask *chính là* một RNN có trạng thái ma trận $S_t=S_{t-1}+\phi(k_t)v_t^\top$: huấn luyện song song như Transformer, suy luận chi phí hằng số như RNN. Nó chưa giải quyết chất lượng (không quên, dung lượng nhớ hữu hạn, không vị trí) và hiệu quả phần cứng — nhưng toàn bộ chuỗi bài 19-24 là lời giải lần lượt cho các điểm này.

**Nguồn gốc:** https://arxiv.org/abs/2006.16236
