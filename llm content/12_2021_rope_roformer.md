# RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)

**Tác giả:** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu (Zhuiyi Technology, Shenzhen)  
**Venue / arXiv:** arXiv:2104.09864 (cs.CL), bản v5 ngày 8/11/2023; paper không nêu venue hội nghị.  
**Nguồn:** https://arxiv.org/abs/2104.09864  
**Chủ đề:** Mã hóa vị trí (position encoding) bằng phép xoay (Rotary Position Embedding, RoPE); tính chất tương đối, suy giảm theo khoảng cách, tương thích linear attention.  
**Vị trí trong lộ trình:** Đọc sau 01 (Attention Is All You Need: sinusoidal position) và 11 (SwiGLU). Paper bổ sung: cách đưa vị trí vào attention bằng nhân (xoay) thay vì cộng, cho phép $q_m^\top k_n$ chỉ phụ thuộc $m-n$. Tiếp theo: 13 (ALiBi, một hướng khác: cộng bias vào score), 16 (MLA, nơi RoPE gây rắc rối với nén KV), 18-24 (linear attention, nơi RoPE được nhắc tới ở §3.3).

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Phần suy diễn từng bước (mục 3) là diễn giải lại của chúng ta, bám theo §3.2 và §3.4 của paper. Mọi "Ứng dụng cho dự án" là suy luận của chúng ta. Các con số ví dụ gắn nhãn "tự tính" không có trong paper.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết (suy diễn từng bước)](#3-phương-pháp-chi-tiết-suy-diễn-từng-bước)
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

**Vấn đề.** Self-attention là phép toán trên *tập hợp*: nếu hoán vị token đầu vào thì đầu ra cũng chỉ hoán vị theo, nó không tự biết thứ tự (paper dẫn Yun et al. 2020: "position-agnostic"). Muốn mô hình biết "từ nào đứng trước từ nào" phải tiêm thông tin vị trí.

**Các cách đã có (paper §2):**

1. **Vị trí tuyệt đối cộng vào embedding:** $f_t(x_i,i)=W_t(x_i+p_i)$, $t\in\{q,k,v\}$. Hoặc $p_i$ học được (BERT, GPT-2), hoặc sinusoidal: $p_{i,2t}=\sin(i/10000^{2t/d})$, $p_{i,2t+1}=\cos(i/10000^{2t/d})$ (Vaswani 2017).
2. **Vị trí tương đối** (Shaw 2018, Transformer-XL, T5, DeBERTa...): đưa $m-n$ vào điểm attention, thường bằng cách khai triển $q_m^\top k_n$ thành 4 số hạng (nội dung-nội dung, nội dung-vị trí, vị trí-nội dung, vị trí-vị trí) rồi thay $p_n$ bằng embedding tương đối $\tilde p_{m-n}$, hoặc cộng một bias học được $b_{i,j}$ (T5).

**Hạn chế mà paper nhắm tới:** các cách trên đều **cộng** thông tin vị trí vào biểu diễn hoặc vào score; paper cho rằng chúng vì thế khó dùng cho **linear attention** (nơi không tính ma trận score $N\times N$ để cộng bias vào). Paper muốn: (i) $\langle q_m,k_n\rangle$ phụ thuộc vị trí **chỉ qua $m-n$**, (ii) không cần tham số học thêm, (iii) hoạt động với độ dài tuỳ ý, (iv) tương thích linear attention.

## 2. Ý tưởng chính

Thay vì cộng một vector vị trí, **xoay** vector query/key một góc tỉ lệ với vị trí token: token ở vị trí $m$ bị xoay $m\theta$. Khi tính tích vô hướng giữa query ở $m$ và key ở $n$, hai phép xoay "triệt tiêu một phần" và chỉ còn lại phép xoay góc $(n-m)\theta$: vị trí tuyệt đối biến mất, chỉ còn khoảng cách tương đối. Vì vector $d$ chiều được chia thành $d/2$ cặp chiều (mỗi cặp là một mặt phẳng 2D) và mỗi cặp xoay với tốc độ $\theta_i$ khác nhau (từ nhanh đến chậm), ta thu được một "bộ đồng hồ" nhiều kim mã hóa khoảng cách. Phép xoay không đổi độ dài vector, không thêm tham số, và áp dụng được cho cả softmax attention lẫn linear attention.

## 3. Phương pháp chi tiết (suy diễn từng bước)

### 3.1 Phát biểu bài toán

Cho $q_m=f_q(x_m,m)$, $k_n=f_k(x_n,n)$. Mong muốn (paper Eq. 11):

$$\langle f_q(x_m,m),\,f_k(x_n,n)\rangle=g(x_m,x_n,m-n).$$

Nghĩa là tích vô hướng chỉ phụ thuộc hai nội dung $x_m,x_n$ và **hiệu vị trí**. Cần tìm $f_q,f_k$ thỏa điều đó, với điều kiện ban đầu $f_q(x,0)=W_qx$, $f_k(x,0)=W_kx$ (vị trí 0 = chưa mã hóa gì).

### 3.2 Ôn nhanh: số phức và phép xoay 2D

Vector 2D $(a,b)$ ↔ số phức $z=a+ib$. Nhân với $e^{i\phi}=\cos\phi+i\sin\phi$ là **xoay** $z$ góc $\phi$ ngược chiều kim đồng hồ, giữ nguyên độ dài $|z|$. Dạng ma trận:

$$
e^{i\phi}\cdot(a+ib)\;\longleftrightarrow\;
\begin{pmatrix}\cos\phi&-\sin\phi\\ \sin\phi&\cos\phi\end{pmatrix}\begin{pmatrix}a\\b\end{pmatrix}=R(\phi)\begin{pmatrix}a\\b\end{pmatrix}.
$$

Tính chất cần dùng: $R(\phi)^\top=R(-\phi)$ (xoay ngược), $R(\alpha)R(\beta)=R(\alpha+\beta)$ (xoay liên tiếp cộng góc). Tích vô hướng 2D của hai vector tương ứng $\langle u,v\rangle=\mathrm{Re}(u\,\bar v)$, với $\bar v$ là liên hợp phức.

### 3.3 Suy ra nghiệm ở $d=2$ (paper §3.4.1)

Viết dạng cực: $f_q(x_q,m)=R_q e^{i\Theta_q}$, $f_k(x_k,n)=R_k e^{i\Theta_k}$, $g=R_ge^{i\Theta_g}$ (đây $R$ là *bán kính*, $\Theta$ là *góc*, không phải ma trận). Yêu cầu $\langle f_q,f_k\rangle = g(\cdot,n-m)$, tức $\mathrm{Re}[f_q\bar f_k]$ khớp với $g$ (paper làm việc với dạng phức này), cho hai ràng buộc (Eq. 24):

$$R_q(x_q,m)\,R_k(x_k,n)=R_g(x_q,x_k,n-m),\qquad \Theta_k(x_k,n)-\Theta_q(x_q,m)=\Theta_g(x_q,x_k,n-m).$$

**Bước 1: bán kính.** Đặt $m=n$ và dùng điều kiện đầu $R_q(\cdot,0)=\|q\|$, $R_k(\cdot,0)=\|k\|$ (Eq. 26a, 27): $R_q(x_q,m)R_k(x_k,m)=R_g(\cdot,0)=\|q\|\|k\|$. Nghiệm đơn giản: bán kính **không phụ thuộc vị trí**, $R_q=\|q\|$, $R_k=\|k\|$. (Đây là lý do phép xoay không đổi độ dài.)

**Bước 2: góc.** Cũng với $m=n$ (Eq. 26b): $\Theta_k(x_k,m)-\Theta_q(x_q,m)=\theta_k-\theta_q$ (hiệu góc ban đầu), suy ra $\Theta_q(x_q,m)-\theta_q=\Theta_k(x_k,m)-\theta_k$. Đại lượng này không phụ thuộc nội dung, chỉ phụ thuộc vị trí; gọi là $\phi(m)$ (Eq. 28): $\Theta_{q,k}(x,m)=\phi(m)+\theta_{q,k}$.

**Bước 3: $\phi$ là cấp số cộng.** Thay $n=m+1$ vào Eq. 24 và dùng Eq. 28: $\phi(m+1)-\phi(m)=\Theta_g(x_q,x_k,1)+\theta_q-\theta_k$ (Eq. 29). Vế phải không phụ thuộc $m$, nên $\phi(m)=m\theta+\gamma$ (Eq. 30) với hằng số $\theta\ne0$, $\gamma$.

**Bước 4: kết quả.** Chọn $\gamma=0$:

$$
\boxed{f_q(x_m,m)=(W_qx_m)\,e^{im\theta},\qquad f_k(x_n,n)=(W_kx_n)\,e^{in\theta}}
$$

Tức là: lấy $q=W_qx_m$ (xem như số phức), xoay thêm $m\theta$. (Lưu ý: "suy ra" này đi qua giả thiết nghiệm đơn giản ở Bước 1 và chọn $\gamma=0$; paper trình bày đây là *một* nghiệm, không chứng minh tính duy nhất.)

### 3.4 Tính chất tương đối (trọng tâm)

Tích vô hướng giữa $q_m$ và $k_n$ ở $d=2$:

$$
q_m^\top k_n=\mathrm{Re}\big[(W_qx_m e^{im\theta})\,\overline{(W_kx_n e^{in\theta})}\big]
=\mathrm{Re}\big[(W_qx_m)\,\overline{(W_kx_n)}\,e^{i(m-n)\theta}\big].
$$

Hai thừa số $e^{im\theta}$ và $\overline{e^{in\theta}}=e^{-in\theta}$ gộp thành $e^{i(m-n)\theta}$: **chỉ còn hiệu vị trí** (Eq. 12). Dạng ma trận: với $R_m\equiv R(m\theta)$,

$$q_m^\top k_n=(R_m W_qx_m)^\top(R_nW_kx_n)=x_m^\top W_q^\top\underbrace{R_m^\top R_n}_{=R_{n-m}}W_kx_n,$$

vì $R_m^\top=R(-m\theta)$ và $R(-m\theta)R(n\theta)=R((n-m)\theta)$. (Paper Eq. 16 viết $R_{\Theta,n-m}$.)

### 3.5 Tổng quát cho $d$ chiều

Giả sử $d$ chẵn. Chia vector thành $d/2$ cặp chiều $(x_{2i-1},x_{2i})$, mỗi cặp xoay với tần số riêng $\theta_i$ (Eq. 14, 15). Ma trận xoay là khối đường chéo:

$$
R^d_{\Theta,m}=\mathrm{diag}\big(R(m\theta_1),R(m\theta_2),\dots,R(m\theta_{d/2})\big),\qquad
\boxed{\theta_i=10000^{-2(i-1)/d},\; i=1,\dots,d/2}
$$

$R^d_{\Theta,m}$ trực giao ($R^\top R=I$), nên giữ nguyên chuẩn của $q,k$ ("ổn định" theo paper). Do tính tuyến tính của tích vô hướng, tích của hai vector tổng quát là tổng của $d/2$ tích 2D:

$$
\boxed{q_m^\top k_n=(R^d_{\Theta,m}W_qx_m)^\top(R^d_{\Theta,n}W_kx_n)=x_m^\top W_q^\top R^d_{\Theta,n-m}W_kx_n}
$$

với $R^d_{\Theta,n-m}=(R^d_{\Theta,m})^\top R^d_{\Theta,n}$. Mỗi cặp $i$ đóng góp một số hạng chỉ phụ thuộc $(n-m)\theta_i$. **Lưu ý:** RoPE chỉ áp lên $q$ và $k$, **không** áp lên $v$ (vị trí tương đối chỉ đi vào score). Vị trí tuyệt đối vẫn "có mặt" trong $q_m,k_n$ riêng lẻ, nhưng chỉ lộ ra qua score dưới dạng hiệu.

### 3.6 Cài đặt hiệu quả (Eq. 34)

Ma trận $R^d_{\Theta,m}$ rất thưa, không nhân trực tiếp. Dùng nhân từng phần tử:

$$
R^d_{\Theta,m}x=\begin{pmatrix}x_1\\x_2\\\vdots\\x_{d-1}\\x_d\end{pmatrix}\otimes\begin{pmatrix}\cos m\theta_1\\\cos m\theta_1\\\vdots\\\cos m\theta_{d/2}\\\cos m\theta_{d/2}\end{pmatrix}+\begin{pmatrix}-x_2\\x_1\\\vdots\\-x_d\\x_{d-1}\end{pmatrix}\otimes\begin{pmatrix}\sin m\theta_1\\\sin m\theta_1\\\vdots\\\sin m\theta_{d/2}\\\sin m\theta_{d/2}\end{pmatrix}
$$

```python
def rope(x, pos, base=10000.0):           # x: [..., T, d], pos: [T]
    d = x.shape[-1]
    inv_freq = base ** (-torch.arange(0, d, 2) / d)       # theta_i, [d/2]
    ang = pos[:, None] * inv_freq[None, :]                # [T, d/2], m*theta_i
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]                   # cặp (x_{2i-1}, x_{2i}), kiểu paper
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out
# q = rope(q, pos); k = rope(k, pos); scores = q @ k.transpose(-1,-2) / sqrt(d)
```

Ghi chú triển khai (ngoài paper): LLaMA/HF thường ghép cặp theo kiểu "chia đôi" $(x_i, x_{i+d/2})$ với hàm `rotate_half` thay vì cặp xen kẽ $(x_{2i-1},x_{2i})$ như trên. Hai cách tương đương nếu hoán vị cố định các chiều của $W_q,W_k$, nhưng **không** thay thế lẫn nhau khi chuyển checkpoint nếu không hoán vị lại trọng số.

### 3.7 Ví dụ số nhỏ (tự tính)

$d=2$, $\theta=\pi/2$ (90° mỗi bước, chỉ để dễ tính). Query nội dung $q=(1,0)$, key nội dung $k=(1,0)$.

| $m$ (query) | $n$ (key) | $q_m$ | $k_n$ | $q_m^\top k_n$ |
|---|---|---|---|---|
| 5 | 3 | xoay $450°\equiv90°$: $(0,1)$ | xoay $270°$: $(0,-1)$ | $-1$ |
| 7 | 5 | xoay $630°\equiv270°$: $(0,-1)$ | xoay $450°\equiv90°$: $(0,1)$ | $-1$ |
| 2 | 0 | $(-1,0)$ | $(1,0)$ | $-1$ |

Cả ba cặp có $m-n=2$ và đều cho $-1=\cos(2\theta)=\cos180°$. Vị trí tuyệt đối thay đổi, kết quả không đổi.

### 3.8 Suy giảm theo khoảng cách (paper §3.3, §3.4.3)

Nhóm các cặp chiều, Eq. 35:

$$q_m^\top k_n=\mathrm{Re}\Big[\sum_{i=0}^{d/2-1}h_i\,e^{i(m-n)\theta_i}\Big],\quad h_i=q_{[2i:2i+1]}\,k^*_{[2i:2i+1]}.$$

Đặt $S_j=\sum_{i<j}e^{i(m-n)\theta_i}$. Biến đổi Abel: $\big|\sum_i h_i e^{i(m-n)\theta_i}\big|=\big|\sum_i S_{i+1}(h_{i+1}-h_i)\big|\le\big(\max_i|h_{i+1}-h_i|\big)\sum_i|S_{i+1}|$. Paper khẳng định (và vẽ ở Fig. 2) rằng với $\theta_i=10000^{-2i/d}$, đại lượng $\frac1{d/2}\sum_i|S_i|$ **giảm khi khoảng cách $|m-n|$ tăng**. Đây là một chặn trên trung bình, được trình bày bằng hình số học chứ không có chứng minh giải tích đầy đủ trong paper; nó cũng không có nghĩa score thực tế luôn giảm với khoảng cách (nội dung $h_i$ vẫn quyết định).

### 3.9 RoPE với linear attention (§3.3, Eq. 17-19)

Dạng tổng quát $\mathrm{Attn}_m=\frac{\sum_n\mathrm{sim}(q_m,k_n)v_n}{\sum_n\mathrm{sim}(q_m,k_n)}$. Linear attention thay $\mathrm{sim}=\varphi(q_m)^\top\phi(k_n)$ với $\varphi,\phi$ không âm. Paper đề xuất xoay *sau* hàm đặc trưng:

$$\mathrm{Attn}_m=\frac{\sum_n\big(R^d_{\Theta,m}\varphi(q_m)\big)^\top\big(R^d_{\Theta,n}\phi(k_n)\big)v_n}{\sum_n\varphi(q_m)^\top\phi(k_n)}.$$

Mẫu số **giữ nguyên không xoay** để tránh chia cho số âm/0; tử số có thể có số hạng âm nên trọng số không còn là xác suất chuẩn hóa. Paper nói vẫn mô hình hóa được độ quan trọng của các value.

## 4. Thí nghiệm và kết quả chính

Tất cả chạy trên hai máy chủ 4×V100 (theo paper). Quy mô nhỏ; không có LLM.

**Dịch máy, WMT 2014 En→De** (so với Transformer-base):

| Model | BLEU |
|---|---|
| Transformer-base (Vaswani 2017) | 27.3 |
| RoFormer | 27.5 |

**Tiền huấn luyện MLM (BERT-base, BookCorpus + Wikipedia, 100k bước, batch 64, độ dài 512, AdamW lr 1e-5):** RoFormer hội tụ nhanh hơn (chỉ có đường cong ở Fig. 3, không có bảng số).

**Fine-tune GLUE (Bảng 2 của paper):**

| Model | MRPC | SST-2 | QNLI | STS-B | QQP | MNLI (m/mm) |
|---|---|---|---|---|---|---|
| BERT | 88.9 | 93.5 | 90.5 | 85.8 | 71.2 | 84.6/83.4 |
| RoFormer | 89.5 | 90.7 | 88.0 | 87.0 | 86.4 | 80.2/79.8 |

RoFormer hơn trên MRPC, STS-B, QQP, kém trên SST-2, QNLI, MNLI. Paper chỉ nhấn mạnh "outperform BERT in three out of six datasets".

**Performer (linear attention) + RoPE, Enwik8, 12 layer, d=768, 12 head, độ dài 1024:** loss thấp hơn và hội tụ nhanh hơn khi có RoPE (Fig. 3 phải, chỉ là đường cong).

**Dữ liệu tiếng Trung (RoFormer sửa từ WoBERT, ~34GB):** tiền huấn luyện nhiều giai đoạn; độ chính xác tăng khi độ dài tối đa tăng (stage 1: len 512, loss 1.73, acc 65.0%; stage 2: len 1536, loss 1.61, acc 66.8%; stage 5: len 1536, loss 1.58, acc 67.4%). Lưu ý các giai đoạn có batch và số bước khác nhau nên không so sánh sạch.

**CAIL2019-SCM (độ chính xác %):**

| Model | Validation | Test |
|---|---|---|
| BERT-512 | 64.13 | 67.77 |
| WoBERT-512 | 64.07 | 68.10 |
| RoFormer-512 | 64.13 | 68.29 |
| RoFormer-1024 | 66.07 | 69.79 |

Paper: ở 1024 RoFormer hơn WoBERT 1.5% tuyệt đối (so với kết quả 68.10 thì 69.79 − 68.10 ≈ 1.7; con số 1.5% là cách nói của paper, ta chép lại nguyên văn).

## 5. Phân tích / ablation

- Paper **không có ablation** kiểu bỏ từng thành phần (ví dụ đổi base 10000, thử không xoay một số chiều). Bằng chứng gồm các so sánh tổng thể ở trên.
- Bằng chứng ủng hộ "tổng quát hóa độ dài": bảng tiếng Trung cho thấy độ chính xác tăng khi dùng context dài hơn khi tiền huấn luyện. Đây vẫn *không phải* thí nghiệm ngoại suy (huấn luyện ngắn, kiểm tra dài) của các bài sau như ALiBi (13).
- Hướng "tần số" cho thấy sự đánh đổi mà paper chưa khai thác: $\theta_1=1$ rad/token (chu kỳ ≈ 6.3 token) đến $\theta_{d/2}\approx10000^{-1}$ (chu kỳ ≈ 62,800 token). Cặp chiều tần số cao mã hóa vị trí gần, cặp tần số thấp mã hóa vị trí xa (diễn giải của chúng ta).

## 6. Giới hạn

**Do tác giả nêu (§4.5.5):** chưa có giải thích kỹ càng vì sao hội tụ nhanh hơn các phương pháp vị trí khác; chưa có lời giải thích chắc chắn vì sao hiệu quả hơn trên văn bản dài dù tính chất suy giảm tương tự các phương pháp khác; cần tài nguyên tiền huấn luyện.

**Chúng ta nhận thấy:**
- Quy mô thí nghiệm nhỏ (BERT-base, T-base); GLUE hỗn hợp (kém hơn BERT ở 3/6 tập). Sức mạnh thực sự của RoPE trong LLM là bằng chứng ngoài paper (GPT-NeoX, LLaMA, PaLM...).
- Paper không thử ngoại suy độ dài ngoài độ dài huấn luyện. Thực tế, RoPE ngoại suy kém nếu không có can thiệp (position interpolation, NTK scaling, YaRN) vì các góc $m\theta_i$ ở vị trí chưa từng thấy là phân phối lạ (nội dung ngoài paper; xem ALiBi 13 để so sánh).
- Tính chất "long-term decay" chỉ là chặn trên gợi ý bằng hình, không chứng minh chặt.
- Tần số cố định (không học), base 10000 là lựa chọn kế thừa từ sinusoidal.

## 7. Liên hệ với các bài khác

- **01:** sinusoidal tuyệt đối cộng vào embedding; RoPE dùng cùng họ tần số $10000^{-2i/d}$ nhưng *nhân* theo kiểu xoay.
- **13 (ALiBi):** cộng bias tuyến tính $-m|i-j|$ vào score, không có tham số, nhằm ngoại suy độ dài; RoPE xoay q,k. So sánh hai phương án là chủ đề chính của bài 13.
- **05, 06 (LLaMA, Mistral):** dùng RoPE; **16 (MLA):** RoPE không "hấp thụ" được vào ma trận nén KV nên DeepSeek phải tách riêng phần "decoupled RoPE".
- **18-24 (linear attention, RetNet, GLA, Mamba, DeltaNet):** §3.3 là một trong những công trình đầu tiên dùng xoay với linear attention; RetNet dùng một dạng tương tự ("xPos"). Với các lớp GatedDeltaNet trong decoder của dự án, vị trí được mã hóa ngầm bởi trạng thái hồi quy nên thường không cần RoPE ở các lớp đó (suy luận, cần kiểm tra config).
- **37 (Qwen2-VL):** mở rộng RoPE thành M-RoPE đa chiều (thời gian/cao/rộng) cho ảnh và video.

## 8. Ứng dụng cho dự án (suy luận)

- **Hiểu decoder:** các lớp full attention của decoder Qwen3.5-style rất có thể dùng RoPE (có thể chỉ xoay một phần `head_dim`, ví dụ tham số `partial_rotary_factor`; cần xác nhận từ config của LightOnOCR-3). Các lớp GatedDeltaNet không dùng RoPE theo cách này.
- **Pruning:** RoPE ghép các chiều theo **cặp** (hoặc nửa-nửa). Khi cắt chiều `head_dim` của q/k (ví dụ giảm 64 → 48) phải cắt **theo cặp xoay** cùng nhau, nếu không phép xoay bị hỏng. Cắt cả cặp có tần số thấp (chậm) thường ít ảnh hưởng quan hệ cục bộ hơn, nhưng đó là giả thuyết cần đo. Cắt số head/layer không đụng tới RoPE.
- **Chiều dài ngữ cảnh (OCR 1,500 token/trang):** không cần ngoại suy xa; chỉ cần chắc chắn độ dài suy luận ≲ độ dài huấn luyện. Với ảnh, kiểm tra xem visual token dùng RoPE 1D hay 2D (nếu theo Qwen2-VL là M-RoPE, cần giữ đúng khi cắt token).
- **Chi phí:** RoPE không có tham số, chi phí thêm là $O(Td)$ nhân từng phần tử cho mỗi layer; gần như bằng 0 so với matmul. Với KV-cache, key được lưu **sau khi đã xoay** (vị trí đã "đóng dấu" vào key), điều quan trọng cho MLA (bài 16).
- **Distillation:** student phải giữ **cùng** base và cách ghép cặp RoPE với teacher; nếu student có `head_dim` khác, đưa lại base/rotary_dim cho khớp để hàm vị trí không bị lệch.

## 9. Thí nghiệm gợi ý

1. **Xác minh tính tương đối bằng số:** viết hàm `rope` ở trên, chọn q,k ngẫu nhiên, kiểm tra $q_m^\top k_n$ với $(m,n)=(5,3),(105,103),(1005,1003)$ bằng nhau đến sai số float. Làm lại với `rotate_half` của HF để thấy kết quả cũng bằng nhau trong cùng một quy ước.
2. **Phổ tần số:** vẽ $\cos(\Delta\theta_i)$ cho $i=1..d/2$, $\Delta=1..2048$ với $d=64$, base 10000; xác định bao nhiêu cặp chiều có chu kỳ > 1,500 token (những cặp này gần như "hằng số" trong một trang OCR). Gợi ý mở: cặp nào có thể bị cắt mà mất ít nhất.
3. **Cắt `head_dim` theo cặp:** trên một layer full-attention, bỏ cặp tần số thấp nhất so với bỏ cặp ngẫu nhiên, đo thay đổi loss/CER trước khi distill.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| Yêu cầu | $\langle f_q(x_m,m),f_k(x_n,n)\rangle=g(x_m,x_n,m-n)$ |
| Nghiệm 2D | $f_q=(W_qx_m)e^{im\theta}$, $f_k=(W_kx_n)e^{in\theta}$ |
| Ma trận xoay | $R^d_{\Theta,m}=\mathrm{diag}(R(m\theta_1),\dots,R(m\theta_{d/2}))$ |
| Tần số | $\theta_i=10000^{-2(i-1)/d}$ |
| Tính tương đối | $q_m^\top k_n=x_m^\top W_q^\top R^d_{\Theta,n-m}W_kx_n$ |
| Tính chất | trực giao (giữ chuẩn), không tham số, không áp lên $v$ |
| Suy giảm | $\tfrac{2}{d}\sum|S_i|$ giảm khi $|m-n|$ tăng (hình số, Fig. 2) |
| Linear attention | xoay sau $\varphi,\phi$, mẫu số không xoay |
| Cài đặt | $x\otimes\cos+\mathrm{rot}(x)\otimes\sin$ (Eq. 34) |

## 11. Kết luận ngắn

RoPE mã hóa vị trí bằng cách xoay từng cặp chiều của q và k một góc tỉ lệ với vị trí, nhờ đó tích vô hướng chỉ phụ thuộc khoảng cách $m-n$. Suy diễn gọn (nghiệm của phương trình chức năng ở 2D, mở rộng bằng tổng khối) và cài đặt rẻ, không tham số. Thí nghiệm trong paper nhỏ và không có ngoại suy độ dài, nhưng ý tưởng đã trở thành chuẩn của LLM hiện đại và là nền cho nhiều biến thể (M-RoPE, decoupled RoPE trong MLA).

Nguồn: https://arxiv.org/abs/2104.09864
