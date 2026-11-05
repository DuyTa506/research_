# Preface: Toán nền tảng và hướng dẫn đọc (LLM, VLM, suy luận)

**Tác giả:** Ghi chú tổng hợp từ kiến thức chuẩn của lĩnh vực (không dựa vào một paper cụ thể).  
**Venue / arXiv:** không áp dụng.  
**Nguồn:** các paper trong thư mục này (xem lịch đọc ở mục 14); công thức tham chiếu chính từ 1706.03762 (bài 01) và 2001.08361 (bài 03).  
**Chủ đề:** token hoá, embedding, next-token prediction, attention và các biến thể phức tạp, KV cache, FLOPs/bộ nhớ, GPU roofline, độ chính xác số, attention tuyến tính/SSM, VLM.  
**Vị trí trong lộ trình:** đọc TRƯỚC mọi bài khác. Bài này cho bạn "bảng chữ cái" để đọc 57 bài còn lại. Sau đó đi theo thứ tự 01 → 04, rồi theo lịch ở mục 14.

> Tài liệu học tập bằng tiếng Việt, viết từ kiến thức nền của người soạn, không phải bản dịch paper. Các con số ví dụ cho dự án LightOnOCR-3-0.8B là **tự tính** với giả định ghi rõ (cấu hình thật có thể khác). Phần "ứng dụng cho dự án" là suy luận của chúng ta.

## Mục lục

1. [Token và tokenization (BPE)](#1-token-và-tokenization-bpe)
2. [Embedding](#2-embedding)
3. [Next-token prediction, cross-entropy, perplexity](#3-next-token-prediction-cross-entropy-perplexity)
4. [Scaled dot-product attention và multi-head, kèm shape](#4-scaled-dot-product-attention-và-multi-head-kèm-shape)
5. [Causal mask và vì sao attention là O(n²)](#5-causal-mask-và-vì-sao-attention-là-on)
6. [Khối residual: normalization, FFN, vị trí](#6-khối-residual-normalization-ffn-vị-trí)
7. [Thông tin vị trí](#7-thông-tin-vị-trí)
8. [Sinh tự hồi quy: prefill và decode, KV cache](#8-sinh-tự-hồi-quy-prefill-và-decode-kv-cache)
9. [Số tham số và FLOPs mỗi token](#9-số-tham-số-và-flops-mỗi-token)
10. [GPU: HBM, SRAM, băng thông, roofline](#10-gpu-hbm-sram-băng-thông-roofline)
11. [Định dạng số và chi phí bộ nhớ](#11-định-dạng-số-và-chi-phí-bộ-nhớ)
12. [Attention tuyến tính, RNN, SSM: trạng thái kích thước cố định](#12-attention-tuyến-tính-rnn-ssm-trạng-thái-kích-thước-cố-định)
13. [Kiến thức nền về VLM](#13-kiến-thức-nền-về-vlm)
14. [Lịch đọc theo số thư mục](#14-lịch-đọc-theo-số-thư-mục)
15. [Bảng thuật ngữ](#15-bảng-thuật-ngữ)
16. [Sáu câu tự kiểm tra](#16-sáu-câu-tự-kiểm-tra)

---

## 1. Token và tokenization (BPE)

**Token** là đơn vị rời rạc mà mô hình thực sự đọc và sinh. Văn bản (chuỗi byte/ký tự) được **tokenizer** cắt thành chuỗi số nguyên $x_1,\dots,x_n\in\{0,\dots,V-1\}$, với $V$ là kích thước từ vựng (vocabulary), thường 32k đến 250k.

Vì sao không dùng ký tự hay từ nguyên?
- Ký tự: chuỗi quá dài ($n$ lớn, mà attention tốn $O(n^2)$).
- Từ nguyên: từ vựng quá lớn, từ lạ/lỗi chính tả thành "unknown".
- **Subword** (mảnh từ) là điểm giữa: từ phổ biến là 1 token, từ hiếm tách thành vài mảnh.

### 1.1 Thuật toán BPE (Byte-Pair Encoding)

Ý tưởng: bắt đầu từ các ký tự (hoặc byte), lặp lại việc **gộp cặp liền kề xuất hiện nhiều nhất** thành một ký hiệu mới, cho đến khi đủ $V$ ký hiệu.

```
Huấn luyện BPE(corpus, V_target):
  vocab <- tất cả byte/ký tự đơn
  tokens <- corpus tách thành các ký hiệu trong vocab
  while |vocab| < V_target:
      (a, b) <- cặp liền kề xuất hiện nhiều nhất trong tokens
      ghi lại luật gộp (a, b) -> ab
      vocab <- vocab ∪ {ab}
      thay mọi cặp (a, b) liền kề trong tokens bằng ab
Mã hoá(text): áp dụng các luật gộp theo đúng thứ tự đã học.
```

Ví dụ nhỏ: corpus "low low low lower lowest". Ban đầu: `l o w`. Cặp (`l`,`o`) xuất hiện nhiều nhất, gộp thành `lo`; sau đó (`lo`,`w`) thành `low`. Từ "lower" thành `low` + `e` + `r`. Hai luật gộp đầu đã biến "low" thành 1 token.

**Byte-level BPE** (GPT-2 trở đi): ký hiệu gốc là 256 byte nên không bao giờ có "unknown". Bài 01 dùng BPE với từ vựng chung ~37.000 (EN-DE), bài 03 dùng $V=50257$.

### 1.2 Hệ quả cho OCR

Số token một trang (~1.500 với bạn) = số bước decode tuần tự, nên **độ trễ ≈ 1.500 × thời gian một bước**. Chữ số, ký hiệu toán, bảng markdown bị cắt thành nhiều token nhỏ. Từ vựng lớn làm ma trận embedding/LM head lớn ($V\cdot d$, mục 9).

---

## 2. Embedding

**Embedding** là một bảng tra $E\in\mathbb{R}^{V\times d}$: token $x_t$ được thay bằng hàng $E[x_t]\in\mathbb{R}^d$. Về mặt toán, nó bằng $\text{onehot}(x_t)^\top E$ nhưng cài đặt là tra bảng (không nhân ma trận thật).

$$
h_t^{(0)} = E[x_t] \quad(\text{+ thông tin vị trí nếu dùng cộng, xem mục 7}),\qquad h^{(0)}\in\mathbb{R}^{n\times d}.
$$

- $d=d_{model}$ là chiều của **residual stream** (luồng dữ liệu chung chạy xuyên các tầng).
- Cuối mạng, **LM head** $W_{out}\in\mathbb{R}^{d\times V}$ biến vector cuối thành **logits** $z\in\mathbb{R}^V$. **Weight tying** dùng chung $W_{out}=E^\top$ để tiết kiệm $V\cdot d$ tham số (bài 01 cũng làm vậy: "share the same weight matrix between the two embedding layers and the pre-softmax linear transformation").
- Với ảnh, "embedding" là phép chiếu tuyến tính các patch (mục 13).

---

## 3. Next-token prediction, cross-entropy, perplexity

Mô hình ngôn ngữ tự hồi quy (autoregressive LM) phân rã xác suất của chuỗi bằng quy tắc chuỗi:

$$
p_\theta(x_1,\dots,x_n)=\prod_{t=1}^{n} p_\theta(x_t\mid x_{<t}).
$$

Tại vị trí $t$ mạng xuất logits $z_t\in\mathbb{R}^V$, rồi **softmax**:

$$
p_\theta(v\mid x_{<t})=\frac{\exp(z_{t,v})}{\sum_{u}\exp(z_{t,u})}.
$$

**Cross-entropy (CE)** trung bình trên token (đơn vị nats nếu dùng ln):

$$
\boxed{\mathcal{L}(\theta)= -\frac1n\sum_{t=1}^{n}\log p_\theta(x_t\mid x_{<t}).}
$$

**Perplexity**: $\text{PPL}=\exp(\mathcal{L})$. Hiểu trực quan: mô hình "bối rối như thể phải chọn đều giữa PPL token". Mô hình ngẫu nhiên đều trên $V=50257$ có $\mathcal{L}=\ln 50257\approx 10.82$ nats, PPL $=V$.

Ví dụ: CE = 2.64 nats (một con số hợp lý cho LM cỡ 1B) tương ứng PPL $=e^{2.64}\approx 14$ (tự tính).

**Teacher forcing**: khi huấn luyện, đầu vào là chuỗi thật dịch một vị trí, nên loss mọi vị trí tính song song trong một lần forward (nhờ causal mask, mục 5); khi suy luận phải sinh tuần tự (mục 8).

**Lưu ý**: PPL và CE phụ thuộc tokenizer. Bài 01 ghi rõ "perplexities are per-wordpiece... should not be compared to per-word perplexities". Bài 03 và 04 dùng CE (nats/token) hoặc bits-per-byte để so sánh.

---

## 4. Scaled dot-product attention và multi-head, kèm shape

Ký hiệu: batch $B$ (đặt $B=1$ cho gọn), độ dài chuỗi $n$, $d=d_{model}$, số head $h$, $d_k=d_v=d/h$.

### 4.1 Một head

Đầu vào $X\in\mathbb{R}^{n\times d}$. Ba phép chiếu tuyến tính học được:

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,\qquad W_Q,W_K\in\mathbb{R}^{d\times d_k},\ W_V\in\mathbb{R}^{d\times d_v}.
$$

Shape: $Q,K\in\mathbb{R}^{n\times d_k}$, $V\in\mathbb{R}^{n\times d_v}$.

$$
\boxed{\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V}
$$

Từng bước với shape:

| Bước | Biểu thức | Shape |
|---|---|---|
| Điểm | $S=QK^\top/\sqrt{d_k}$ | $n\times n$ |
| Mask (mục 5) | $S\leftarrow S+M$ | $n\times n$ |
| Trọng số | $A=\text{softmax}_{\text{theo hàng}}(S)$ | $n\times n$, mỗi hàng tổng = 1 |
| Đầu ra | $O=AV$ | $n\times d_v$ |

Diễn giải: query của vị trí $i$ so khớp với key của mọi vị trí $j$; trọng số $A_{ij}$ quyết định lấy bao nhiêu value $v_j$. $O_i=\sum_j A_{ij}v_j$ là **trung bình có trọng số** của value.

### 4.2 Vì sao chia $\sqrt{d_k}$

Nếu các thành phần của $q,k$ độc lập, kỳ vọng 0, phương sai 1 thì $q\cdot k=\sum_{i=1}^{d_k}q_ik_i$ có phương sai $d_k$ (bài 01, chú thích 4). Điểm lớn đẩy softmax vào vùng bão hoà, gradient rất nhỏ. Chia $\sqrt{d_k}$ đưa phương sai về 1.

### 4.3 Multi-head attention (MHA)

Chạy $h$ head song song với phép chiếu riêng, ghép lại rồi chiếu ra:

$$
\text{MHA}(X)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W_O,\quad \text{head}_i=\text{Attention}(XW_Q^i,XW_K^i,XW_V^i),
$$

$W_O\in\mathbb{R}^{hd_v\times d}$. Trong thực tế gộp $h$ head thành một ma trận lớn $W_Q\in\mathbb{R}^{d\times d}$ rồi reshape:

```python
# X: (B, n, d); h heads; dk = d // h
Q = (X @ Wq).view(B, n, h, dk).transpose(1, 2)   # (B, h, n, dk)
K = (X @ Wk).view(B, n, h, dk).transpose(1, 2)   # (B, h, n, dk)
V = (X @ Wv).view(B, n, h, dk).transpose(1, 2)   # (B, h, n, dk)
S = Q @ K.transpose(-1, -2) / dk**0.5            # (B, h, n, n)
S = S.masked_fill(causal_mask, float('-inf'))
A = S.softmax(dim=-1)                            # (B, h, n, n)
O = (A @ V).transpose(1, 2).reshape(B, n, d)     # (B, n, d)
Y = O @ Wo                                       # (B, n, d)
```

Chi phí tham số của MHA: $4d^2$ ($W_Q,W_K,W_V,W_O$ mỗi cái $d^2$). Bài 01 cho thấy chia $d$ thành $h$ head nhỏ thì tổng tính toán gần như bằng một head chiều đầy đủ, nhưng cho phép chú ý nhiều "không gian con" khác nhau.

---

## 5. Causal mask và vì sao attention là O(n²)

### 5.1 Causal mask

Với mô hình sinh, vị trí $i$ chỉ được nhìn $j\le i$. Cộng ma trận mask $M$ trước softmax:

$$
M_{ij}=\begin{cases}0 & j\le i\\ -\infty & j>i\end{cases}\qquad\Rightarrow\qquad A_{ij}=0\ \text{khi } j>i.
$$

Nhờ đó một lần forward trên cả chuỗi tính được loss cho mọi vị trí song song mà không "nhìn trộm" tương lai. Bài 01 cài mask bằng cách đặt $-\infty$ ở đầu vào softmax. Encoder (và ViT) dùng attention **hai chiều** (không mask).

### 5.2 Độ phức tạp

Ma trận điểm $S$ có $n^2$ phần tử, mỗi phần tử là tích vô hướng $d_k$ chiều.

- **Thời gian**: $QK^\top$ tốn $2n^2d_k$ FLOPs mỗi head, $AV$ tốn thêm $2n^2d_v$. Tổng một tầng (mọi head): $\approx 4n^2d$ FLOPs, tức $O(n^2d)$ (khớp bảng 1 bài 01: self-attention $O(n^2\cdot d)$).
- **Bộ nhớ**: nếu lưu $S$/$A$ thì $O(h\,n^2)$ phần tử mỗi tầng. Với $n=1500$, $h=16$, FP16: $16\cdot1500^2\cdot2\approx 72$ MB mỗi tầng (tự tính). Đây là lý do FlashAttention (bài 44) tính theo khối trong SRAM và không bao giờ ghi $S$ ra HBM, đưa bộ nhớ phụ về $O(n)$.
- So với phần còn lại: phần chiếu tuyến tính + FFN tốn $\approx 24nd^2$ FLOPs mỗi tầng (nhân $n$ tuyến tính). Attention $4n^2d$ chiếm ưu thế khi $4n^2d>24nd^2$, tức $n>6d$. Với $d=1024$ thì $n\gtrsim 6000$. Với 1.500 token, **attention chưa phải phần đắt nhất về FLOPs**, nhưng KV cache và băng thông thì vẫn tính (mục 8).

---

## 6. Khối residual: normalization, FFN, vị trí

Một **Transformer block** (kiểu decoder hiện đại, "Pre-LN"):

$$
\begin{aligned}
u &= x + \text{Attn}(\text{Norm}_1(x)),\\
y &= u + \text{FFN}(\text{Norm}_2(u)).
\end{aligned}
$$

Mọi sub-layer **cộng vào** residual stream $x$ thay vì thay thế nó. Hệ quả: $x^{(L)}=x^{(0)}+\sum_l \Delta x^{(l)}$; gradient có đường tắt, và mỗi tầng chỉ "đọc rồi ghi thêm" một cập nhật (cách nhìn này trùng với notes về Transformer Circuits).

### 6.1 Normalization

- **LayerNorm**: $\text{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$, với $\mu,\sigma$ tính trên chiều $d$ của từng token. Bài 01 dùng **Post-LN**: $\text{LN}(x+\text{Sublayer}(x))$.
- **Pre-LN** (norm trước sub-layer, bài 09): huấn luyện ổn định hơn, ít cần warmup. Đa số LLM hiện nay dùng Pre-LN.
- **RMSNorm** (bài 10): bỏ phép trừ trung bình, $\text{RMS}(x)=\gamma\odot x/\sqrt{\frac1d\sum x_i^2+\epsilon}$. Rẻ hơn, hiệu quả tương đương.

### 6.2 FFN (feed-forward network, còn gọi MLP)

Áp dụng độc lập cho từng token. Bản gốc (bài 01): $\text{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2$ với $d_{ff}=4d$. Hiện đại hoá: **SwiGLU** (bài 11)

$$
\text{FFN}(x)=\big(\text{SiLU}(xW_g)\odot xW_u\big)W_d,\quad W_g,W_u\in\mathbb{R}^{d\times d_{ff}},\ W_d\in\mathbb{R}^{d_{ff}\times d}.
$$

Ba ma trận nên thường chọn $d_{ff}\approx\frac{8}{3}d$ để giữ ~$8d^2$ tham số như FFN hai ma trận với $d_{ff}=4d$.

FFN chiếm khoảng 2/3 số tham số của một tầng (xem mục 9) và thường được xem là nơi lưu "kiến thức". Với bạn: đây là chỗ bạn quan sát **cắt neuron MLP phục hồi tệ hơn cắt cả tầng**. Gợi ý (suy luận): mỗi neuron là một cột/hàng trong 3 ma trận trên, cắt neuron làm thay đổi phân phối kích hoạt của mọi tầng sau theo cách khó bù bằng một phép tuyến tính tại điểm cắt.

---

## 7. Thông tin vị trí

Attention là hàm **hoán vị-bất biến** (đổi thứ tự token chỉ đổi thứ tự hàng đầu ra), nên phải đưa vị trí vào.

| Cách | Ý tưởng | Bài |
|---|---|---|
| Sinusoidal cộng vào embedding | $PE_{(pos,2i)}=\sin(pos/10000^{2i/d})$, $PE_{(pos,2i+1)}=\cos(\cdot)$ | 01 |
| Learned absolute | bảng $P\in\mathbb{R}^{n_{max}\times d}$ | 01 (kết quả gần như như nhau), GPT |
| **RoPE** | xoay $q,k$ theo góc $\propto$ vị trí; tích $q_i^\top k_j$ chỉ phụ thuộc $i-j$ | 12 |
| **ALiBi** | cộng $-m\cdot(i-j)$ vào điểm attention | 13 |
| Ngầm (recurrence) | trạng thái RNN/SSM tự mang thứ tự | 18 đến 24 |

Công thức RoPE cho một cặp chiều $(2m,2m+1)$: xoay bởi góc $i\theta_m$ với $\theta_m=10000^{-2m/d_k}$; vì phép xoay trực giao, $\langle R_iq,R_jk\rangle=\langle q,R_{j-i}k\rangle$, tức chỉ phụ thuộc khoảng cách tương đối.

Ảnh 2D cần vị trí 2D; Qwen2-VL (bài 37) mở rộng RoPE sang nhiều chiều.

---

## 8. Sinh tự hồi quy: prefill và decode, KV cache

### 8.1 Hai pha của suy luận

Sinh văn bản là vòng lặp: đưa prompt vào, chọn token tiếp theo (argmax hoặc lấy mẫu), nối vào chuỗi, lặp lại.

| | **Prefill** | **Decode** |
|---|---|---|
| Làm gì | xử lý toàn bộ prompt (với bạn: token ảnh + prompt) trong một lần | sinh 1 token mỗi bước |
| Số token xử lý mỗi lượt | $n_{prompt}$ (hàng trăm đến nghìn) | 1 |
| Phép toán chính | GEMM lớn (ma trận x ma trận) | GEMV (ma trận x vector) |
| Giới hạn bởi | **tính toán** (compute-bound) | **băng thông bộ nhớ** (memory-bound), xem mục 10 |
| Chỉ số | TTFT (time to first token) | TPOT/ITL (time per output token) |

Với OCR: prefill (ảnh qua ViT + token ảnh vào decoder) là chi phí một lần cho mỗi trang; decode là ~1.500 bước nối tiếp, thường **chiếm phần lớn thời gian**.

### 8.2 KV cache

Ở bước decode $t$, query mới $q_t$ cần $K_{1..t}$ và $V_{1..t}$. Các $k_s,v_s$ của $s<t$ **không đổi** (nhờ causal mask), nên ta lưu lại thay vì tính lại. Mỗi bước chỉ tính $q_t,k_t,v_t$ cho token mới, ghi $k_t,v_t$ vào cache, rồi attention với cache.

```
decode_step(token_t, cache):
    x = embed(token_t)
    for layer l:
        q, k, v = proj(norm(x))           # chỉ 1 token
        cache[l].K.append(k); cache[l].V.append(v)
        x = x + Wo @ softmax(q @ cache[l].K^T / sqrt(dk)) @ cache[l].V
        x = x + FFN(norm(x))
    return logits(x)
```

**Công thức kích thước KV cache** (một chuỗi, batch 1):

$$
\boxed{\text{KV bytes}=2\cdot L\cdot h_{kv}\cdot d_{head}\cdot n\cdot b}
$$

với số 2 là K và V, $L$ số tầng **có full attention**, $h_{kv}$ số KV head, $d_{head}$ chiều mỗi head, $n$ độ dài (tổng token đã ở trong ngữ cảnh), $b$ byte mỗi phần tử.

**Ví dụ số (tự tính, giả định cho mô hình cỡ 0.8B của bạn):** $L=24$, $d_{head}=128$, $n=1500$, BF16 ($b=2$).

| Trường hợp | $h_{kv}$ | Tầng có KV | KV cache |
|---|---|---|---|
| MHA, $d=1024$ chia 8 head x 128 | 8 | 24 | $2\cdot24\cdot8\cdot128\cdot1500\cdot2=147{,}456{,}000$ B $\approx 147$ MB |
| GQA, 4 KV head | 4 | 24 | $2\cdot24\cdot4\cdot128\cdot1500\cdot2=73{,}728{,}000$ B $\approx 74$ MB |
| GQA 4 KV head, chỉ 1/4 số tầng là full attention (hybrid 3:1) | 4 | 6 | $\approx 18.4$ MB |

(Các tầng GatedDeltaNet không có KV cache tăng theo $n$; chúng giữ một trạng thái cố định, mục 12.) Với 1.500 token KV cache còn nhỏ so với trọng số (1.6 GB), nhưng:
- tăng tuyến tính theo $n$ và theo batch, nên quan trọng với ngữ cảnh dài hoặc phục vụ nhiều trang cùng lúc;
- mỗi bước decode phải **đọc toàn bộ** cache, cộng thêm vào lượng byte phải đọc (mục 10).

Chú ý: token ảnh cũng nằm trong ngữ cảnh. Nếu một trang cho ~1.000 token ảnh thì $n$ lúc decode xong khoảng 2.500 chứ không phải 1.500; KV tăng tương ứng.

---

## 9. Số tham số và FLOPs mỗi token

### 9.1 Đếm tham số (decoder chuẩn, MHA, FFN $d_{ff}=4d$)

Mỗi tầng: attention $4d^2$ + FFN $2\cdot d\cdot 4d=8d^2$ (bỏ bias, norm):

$$
\boxed{N_{\text{non-emb}}\approx 12\,L\,d^2}
$$

(đúng như Kaplan phương trình 2.1, bài 03). Thêm embedding $V\cdot d$ (và LM head $V\cdot d$ nếu không tie). Ví dụ GPT-2 small-ish: $L=12,d=768$: $12\cdot12\cdot768^2\approx85$M.

Với GQA: $W_K,W_V$ nhỏ đi nên attention còn $\approx 2d^2+2d\cdot h_{kv}d_{head}$. Với SwiGLU: 3 ma trận $d\times d_{ff}$.

**Cảnh báo cho mô hình nhỏ**: với $d=1024$, $V\approx 150\text{k}$ thì $V\cdot d\approx 154$M, đáng kể so với tổng 0.3B/0.1B. Khi cắt, hãy luôn tách riêng "embedding" khỏi "thân Transformer" (Kaplan cũng loại embedding khỏi $N$, bài 03).

### 9.2 FLOPs mỗi token

Một phép nhân ma trận–vector với ma trận $m\times k$ tốn $2mk$ FLOPs (một nhân và một cộng cho mỗi trọng số). Vì mỗi trọng số tham gia một lần:

$$
\boxed{\text{FLOPs}_{\text{forward}}/\text{token}\approx 2N}\qquad\text{(huấn luyện: }\approx 6N\text{ vì backward}\approx 2\times\text{forward)}.
$$

Cộng thêm phần phụ thuộc ngữ cảnh $\approx 2L\,n\,d$ (Kaplan 2.2, bài 03: $C_{forward}\approx 2N+2n_{layer}n_{ctx}d_{attn}$, với $n_{ctx}$ là độ dài ngữ cảnh).

**Ví dụ (tự tính):** $N=0.8\times10^9$: $2N=1.6$ GFLOPs/token. Decode 1.500 token: $2.4$ TFLOPs. Prefill 1.500 token: cùng $\approx 2.4$ TFLOPs nhưng làm một lượt. Huấn luyện/distill trên $D$ token: $6ND$ (xem bài 03, 04).

Cắt tham số thì FLOPs/token giảm tỷ lệ, nhưng **latency batch-1 phụ thuộc byte đọc chứ không phải FLOPs** (mục 10).

---

## 10. GPU: HBM, SRAM, băng thông, roofline

### 10.1 Phân cấp bộ nhớ

| Tầng | Dung lượng (cỡ) | Băng thông (cỡ) | Ghi chú |
|---|---|---|---|
| Thanh ghi / **SRAM** (shared mem, L1/L2) trên chip | vài chục KB mỗi SM; tổng ~chục MB | rất cao (hàng chục TB/s) | nhanh, nhỏ |
| **HBM/VRAM** (bộ nhớ chính của GPU) | 8 GB (card của bạn) đến 80 GB | ~200 GB/s (card nhỏ) đến ~3 TB/s (data center) | trọng số, KV cache, kích hoạt nằm đây |
| RAM máy chủ qua PCIe | lớn | ~vài chục GB/s | chậm |

(Các con số là bậc độ lớn, tuỳ GPU; hãy tra thông số card thật của bạn.)

Phần lớn tối ưu suy luận (FlashAttention, quantization, KV cache nén, speculative decoding) đều là **giảm số byte di chuyển giữa HBM và chip**, không phải giảm FLOPs.

### 10.2 Arithmetic intensity và roofline

**Arithmetic intensity** $I$ = FLOPs thực hiện / byte truyền từ bộ nhớ (FLOP/byte). **Roofline model**: hiệu năng đạt được

$$
\boxed{P=\min\big(P_{peak},\ I\cdot BW\big)}
$$

với $P_{peak}$ đỉnh tính toán (FLOP/s), $BW$ băng thông bộ nhớ (byte/s). Điểm gãy (ridge) $I^*=P_{peak}/BW$: dưới $I^*$ là memory-bound, trên là compute-bound.

### 10.3 Vì sao decode batch-1 bị memory-bound

Một bước decode với batch 1 là GEMV: mỗi trọng số (2 byte BF16) được đọc **một lần** để thực hiện đúng 2 FLOPs (1 nhân + 1 cộng).

$$
I_{decode,b=1}\approx\frac{2N}{2N\ \text{byte}}=1\ \text{FLOP/byte}\ (\text{BF16}),\quad\text{vs. }I^*\sim 30\text{ đến }300\ \text{FLOP/byte}.
$$

**Ví dụ (tự tính, giả định GPU 8 GB: $BW=300$ GB/s, $P_{peak}=10$ TFLOP/s):**
- Trọng số BF16 0.8B = 1.6 GB. Thời gian đọc: $1.6/300\approx 5.3$ ms/token, tức tối đa $\approx 190$ token/s. Thời gian tính: $1.6\text{G}/10\text{T}=0.16$ ms, nhỏ hơn 30 lần. Vậy $t_{token}\approx \dfrac{\text{byte đọc}}{BW}$ (cộng KV cache, kích hoạt, overhead).
- 1.500 token: $\approx 8$ s chỉ riêng trọng số (cận dưới lý thuyết, chưa tính overhead launch kernel, vốn thường đáng kể với mô hình nhỏ).
- Cắt còn 0.3B BF16 (0.6 GB): $\approx 2$ ms/token. Quantize INT4 0.3B (0.15 GB): $\approx 0.5$ ms/token lý thuyết. Ở quy mô này overhead (launch kernel, Python, đồng bộ) thường lấn át; đo thực tế quan trọng hơn công thức.
- Prefill 1.500 token: $I\approx$ số token trong batch (mỗi trọng số đọc một lần, dùng 1.500 lần) $\approx 1500$ FLOP/byte → compute-bound.

**Hệ quả:** tăng batch tăng $I$ gần tuyến tính (đến ridge) nên throughput tăng "gần miễn phí", nhưng bạn nhắm batch-1 latency nên chỉ có hai đòn bẩy: **giảm byte** (cắt tham số, quantize, KV nhỏ, tầng không-KV) và **giảm số bước tuần tự** (speculative decoding, bài 48 đến 50).

---

## 11. Định dạng số và chi phí bộ nhớ

| Định dạng | Bit | Cấu trúc (dấu/mũ/định trị) | Ghi chú |
|---|---|---|---|
| FP32 | 32 | 1/8/23 | chuẩn huấn luyện cũ; giữ "master weights" và optimizer state |
| **BF16** | 16 | 1/8/7 | cùng dải mũ với FP32, ít chính xác; ổn định, phổ biến cho train và infer |
| FP16 | 16 | 1/5/10 | chính xác hơn BF16 nhưng dải hẹp, cần loss scaling |
| INT8 | 8 | số nguyên | quantization sau huấn luyện (bài 51, 52) |
| INT4 | 4 | số nguyên | cần quantization nhóm/hiệu chỉnh (bài 53, 54) |

Bộ nhớ trọng số $=N\times\text{byte/param}$. **Ví dụ (tự tính):**

| Tham số | FP32 (4B) | BF16 (2B) | INT8 (1B) | INT4 (0.5B) |
|---|---|---|---|---|
| 0.8B | 3.2 GB | 1.6 GB | 0.8 GB | 0.4 GB |
| 0.3B | 1.2 GB | 0.6 GB | 0.3 GB | 0.15 GB |
| 0.1B | 0.4 GB | 0.2 GB | 0.1 GB | 0.05 GB |

(Thực tế cộng thêm scale/zero-point cho INT4/INT8 nhóm, nên hơi lớn hơn.)

Quantization trong dự án: kết hợp với cắt (cả hai đều giảm byte đọc). Hai thứ cộng dồn gần như độc lập về lý thuyết, nhưng sai số chồng nhau, nên kiểm tra sau từng bước.

---

## 12. Attention tuyến tính, RNN, SSM: trạng thái kích thước cố định

### 12.1 Vấn đề của softmax attention

Softmax attention cần giữ **toàn bộ** $K,V$ (kích thước $\propto n$) và tốn $O(n)$ mỗi bước decode (so khớp $q$ với $n$ key), tổng $O(n^2)$.

### 12.2 Attention tuyến tính = RNN có trạng thái ma trận

Bỏ softmax, thay bằng kernel: $A_{ij}\propto\phi(q_i)^\top\phi(k_j)$. Khi đó đầu ra có thể viết dạng hồi quy (bài 18):

$$
S_t=S_{t-1}+v_tk_t^\top,\qquad o_t=S_tq_t,\qquad S_t\in\mathbb{R}^{d_v\times d_k}.
$$

$S_t$ là **bộ nhớ liên kết (associative memory)** có kích thước **cố định** bất kể $n$: mỗi bước decode tốn $O(d_kd_v)$, không phụ thuộc $n$. Song song hoá lúc huấn luyện bằng thuật toán chunk (chia chuỗi thành khối).

### 12.3 Các cải tiến (đúng thứ tự thư mục)

- **Gating/decay** (RetNet 19, GLA 20, Mamba 21, Mamba-2 22): $S_t=\alpha_t S_{t-1}+v_tk_t^\top$, với $\alpha_t\in(0,1)$ cho phép "quên".
- **Delta rule** (DeltaNet 23): sửa lỗi thay vì chỉ cộng dồn:
$$
S_t=S_{t-1}\big(I-\beta_tk_tk_t^\top\big)+\beta_tv_tk_t^\top,
$$
  tức trước khi ghi cặp $(k_t,v_t)$ thì xoá phần giá trị cũ gắn với $k_t$. **Gated DeltaNet** (24) kết hợp cả quên toàn cục $\alpha_t$ và delta rule. Đây là loại tầng trong decoder của LightOnOCR.
- **SSM** (state-space model, Mamba): cùng họ hồi quy tuyến tính với trạng thái cố định, tham số hoá khác (bài 21, 22 chỉ ra quan hệ đối ngẫu với attention).

### 12.4 So sánh chi phí

| | Softmax attention | Linear/SSM/DeltaNet |
|---|---|---|
| Trạng thái khi decode | KV cache, $\propto n$ | $S$ cố định ($d_k\times d_v$ mỗi head) |
| Thời gian một bước decode | $O(n\,d)$ | $O(d_kd_v)$ |
| Huấn luyện | $O(n^2d)$ | $O(n\,d^2)$ dạng chunk |
| Truy hồi chính xác (copy/lookup dài) | mạnh | yếu hơn (nén vào $S$ hữu hạn) |

Đó là lý do **hybrid** (Jamba 25, Kimi Linear 26, và decoder của bạn): phần lớn tầng là tuyến tính (rẻ), xen vài tầng full attention (truy hồi chính xác).

**Ví dụ (tự tính, giả định):** 18 tầng GatedDeltaNet, 8 head, $d_k=d_v=128$, trạng thái FP32: $8\cdot128\cdot128\cdot4=524{,}288$ B $\approx0.5$ MB mỗi tầng, $\approx9.4$ MB cho 18 tầng, **không đổi** dù $n=1500$ hay $15000$. Cộng 6 tầng full-attention KV (mục 8) ≈ 18 MB ở $n=1500$.

---

## 13. Kiến thức nền về VLM

**VLM (vision-language model)** = bộ mã hoá ảnh + (bộ chiếu) + mô hình ngôn ngữ. Với OCR: ảnh trang → văn bản.

### 13.1 Từ ảnh thành token (ViT, bài 29)

1. Chia ảnh $H\times W$ thành các **patch** $P\times P$ (ví dụ $P=14$ hoặc 16): có $\frac{H}{P}\cdot\frac{W}{P}$ patch.
2. Mỗi patch duỗi thành vector $P^2\cdot3$ chiều, nhân với ma trận chiếu → embedding $d_v$ chiều (thực chất một conv stride $P$).
3. Cộng embedding vị trí (hoặc dùng RoPE 2D), đưa qua $M$ khối Transformer **hai chiều** (full attention, không mask). Dự án có ViT 12 khối.
4. Ra một chuỗi **visual token** (đặc trưng thị giác).

### 13.2 Projector (cầu nối)

Visual token sống trong không gian của ViT; phải chiếu sang không gian embedding của LLM ($d$ khác). Các lựa chọn: tuyến tính/MLP 2 lớp (LLaVA, bài 34, 35), Q-Former (BLIP-2, bài 33), cross-attention (Flamingo, bài 32). Nhiều mô hình còn **gộp token** (ví dụ gộp 2x2 patch thành 1) để giảm số visual token.

### 13.3 Visual token đi vào decoder

Chuỗi đầu vào của decoder $=[\text{visual tokens}]+[\text{prompt tokens}]$, rồi sinh text. Visual token chỉ xuất hiện ở **prefill** nhưng chiếm chỗ trong KV cache suốt quá trình decode.

### 13.4 Độ phân giải động (dynamic resolution)

Tài liệu và trang OCR cần độ phân giải cao (chữ nhỏ). Thay vì ép về 224x224, các mô hình mới (InternVL 1.5 bài 36, Qwen2-VL bài 37) cho **số visual token thay đổi theo kích thước ảnh**: ảnh lớn hơn nhiều token hơn.

**Ví dụ (tự tính, giả định):** ảnh $896\times896$, $P=14$: $64\times64=4096$ patch; gộp 2x2 → 1024 visual token. ViT xử lý 4096 token (attention $n^2=16.8$M điểm mỗi head mỗi khối), decoder nhận 1024 token ảnh.

---

## 14. Lịch đọc theo số thư mục

Mỗi mục là gợi ý khoảng một buổi (1 đến 2 giờ/bài ngắn; bài lớn như GPT-3, Flamingo chỉ đọc phần lõi).

| Giai đoạn | Bài | Mục tiêu |
|---|---|---|
| **A. Nền Transformer** | 01, 09, 10, 11, 12 | Attention, Pre-LN, RMSNorm, SwiGLU, RoPE: các khối trong decoder hiện đại |
| **B. Quy mô và huấn luyện** | 02, 03, 04, 05, 06, 07, 08 | GPT-3, scaling laws, Chinchilla, LLaMA, Mistral, RLHF, LoRA |
| **C. Giảm KV và ngữ cảnh** | 13, 14, 15, 16, 17 | ALiBi, MQA, GQA, MLA, sliding window |
| **D. Attention tuyến tính và hybrid** | 18, 19, 20, 21, 22, 23, 24, 25, 26 | Từ Linear Transformer đến Gated DeltaNet và hybrid: **trọng tâm kiến trúc của bạn** |
| **E. MoE** | 27, 28 | Switch, Mixtral (đọc nhanh, ít liên quan) |
| **F. Thị giác và VLM** | 29, 30, 31, 32, 33, 34, 35, 36, 37, 38 | ViT, CLIP, SigLIP, ... , Qwen2-VL, SmolVLM |
| **G. OCR** | 39, 40, 41, 42, 43 | Donut, Nougat, GOT, olmOCR, LightOnOCR: bài toán đích |
| **H. Suy luận hiệu quả** | 44, 45, 46, 47, 48, 49, 50 | FlashAttention, PagedAttention, SGLang, speculative decoding |
| **I. Lượng tử hoá và nén KV** | 51, 52, 53, 54, 55, 56, 57 | LLM.int8, SmoothQuant, GPTQ, AWQ, StreamingLLM, H2O, FastV |

Đề xuất thứ tự tối thiểu cho dự án: 00 → 01 → 03 → 04 → 09/10/11/12 → 14/15 → 18 → 23 → 24 → 25 → 29 → 34/37 → 43 → 44 → 48 → 53/54 → 57. Phần cắt tỉa và distillation có bộ ghi chú riêng.

---

## 15. Bảng thuật ngữ

| Thuật ngữ | Nghĩa ngắn |
|---|---|
| Token | Đơn vị rời rạc (mảnh từ/byte/patch) mà mô hình xử lý |
| BPE | Thuật toán gộp cặp phổ biến để tạo từ vựng subword |
| Embedding | Vector $d$ chiều tra bảng cho mỗi token |
| Cross-entropy | $-\log p(\text{token đúng})$ trung bình; hàm loss của LM |
| Residual stream | Vector $d$ chiều mỗi token, mọi tầng đọc và cộng vào |
| Attention | Trung bình có trọng số của value theo độ khớp query-key |
| Decode | Pha sinh từng token |
| KV cache | $K,V$ đã tính của các token trước, lưu để dùng lại |
| MQA / GQA / MLA | Chia sẻ hoặc nén KV để giảm cache |
| Arithmetic intensity | FLOPs / byte truyền |
| Roofline | Mô hình $P=\min(P_{peak},I\cdot BW)$ |
| Quantization | Lưu trọng số bằng ít bit hơn |
| Linear attention | Attention không softmax, viết được dạng hồi quy |
| SSM | State-space model, họ hồi quy tuyến tính (Mamba) |
| Hybrid | Trộn tầng tuyến tính và full attention |
| ViT | Vision Transformer, ảnh → patch → token |
| Distillation | Học sinh bắt chước đầu ra giảng viên |

---

## 16. Sáu câu tự kiểm tra

**Câu 1.** Với $d_k=64$, vì sao chia cho $\sqrt{d_k}$ và điều gì xảy ra nếu bỏ?
<details><summary>Đáp án</summary>

Nếu thành phần $q,k$ độc lập, trung bình 0, phương sai 1 thì $q\cdot k$ có phương sai $d_k=64$, độ lệch chuẩn 8. Điểm lớn đẩy softmax thành gần one-hot, gradient qua softmax rất nhỏ, huấn luyện chậm hoặc kẹt. Chia $\sqrt{d_k}=8$ đưa phương sai về 1.
</details>

**Câu 2.** Một mô hình 24 tầng, 4 KV head, $d_{head}=128$, BF16. KV cache cho 2.000 token là bao nhiêu? Nếu chỉ 6 tầng là full attention thì sao?
<details><summary>Đáp án</summary>

$2\cdot24\cdot4\cdot128\cdot2000\cdot2=98{,}304{,}000$ B $\approx 98$ MB. Với 6 tầng: $2\cdot6\cdot4\cdot128\cdot2000\cdot2=24{,}576{,}000$ B $\approx 24.6$ MB (giảm 4 lần). Các tầng tuyến tính chỉ thêm trạng thái cố định.
</details>

**Câu 3.** Vì sao decode batch-1 của mô hình 0.8B BF16 trên GPU 300 GB/s bị chặn ở khoảng 190 token/s, bất kể GPU có 10 hay 100 TFLOP/s?
<details><summary>Đáp án</summary>

Mỗi token phải đọc toàn bộ trọng số (1.6 GB) từ HBM: $1.6/300\approx5.3$ ms. FLOPs mỗi token chỉ 1.6 G, tính xong trong 0.16 ms (10 TFLOP/s) hoặc 0.016 ms (100 TFLOP/s). Intensity $\approx1$ FLOP/byte, thấp hơn nhiều so với điểm gãy roofline, nên băng thông quyết định. Đổi GPU nhanh hơn về FLOPs không giúp; quantize/cắt trọng số hoặc tăng băng thông thì có.
</details>

**Câu 4.** Tại sao attention là $O(n^2)$ thời gian nhưng FlashAttention vẫn cho kết quả toán học giống hệt?
<details><summary>Đáp án</summary>

Thời gian $O(n^2d)$ vì cần mọi cặp $(i,j)$ của $QK^\top$ và $AV$. FlashAttention không giảm số FLOPs mà tính theo khối (tiling) trong SRAM với softmax trực tuyến (online softmax), không bao giờ ghi ma trận $n\times n$ ra HBM. Kết quả chính xác vì chỉ đổi thứ tự tính và cách truy cập bộ nhớ, giảm bộ nhớ phụ về $O(n)$ và số byte HBM.
</details>

**Câu 5.** Ghi một cặp $(k,v)$ vào trạng thái $S$ bằng quy tắc cộng dồn khác gì quy tắc delta? Tại sao delta tốt hơn cho truy hồi?
<details><summary>Đáp án</summary>

Cộng dồn: $S_t=S_{t-1}+v_tk_t^\top$, không bao giờ xoá nên khi nhiều key tương quan thì nhiễu chồng chất, $S$ bị bão hoà. Delta: $S_t=S_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$, trước tiên loại bỏ giá trị cũ gắn với $k_t$ rồi mới ghi giá trị mới (như một bước SGD sửa lỗi hồi quy tuyến tính), nên bộ nhớ hữu hạn được dùng hiệu quả hơn cho việc tra cứu theo key (bài 23, 24).
</details>

**Câu 6.** Ảnh $1008\times 1008$, patch 14, gộp 2x2. Bao nhiêu patch vào ViT và bao nhiêu visual token vào decoder? Điều này ảnh hưởng gì tới prefill và KV cache?
<details><summary>Đáp án</summary>

$1008/14=72$, nên $72^2=5184$ patch vào ViT. Gộp 2x2 cho $36^2=1296$ visual token vào decoder. ViT tốn attention $n^2$ trên 5184 token (compute-bound, một lần/trang). Decoder: prefill xử lý 1296 + prompt token; mỗi token ảnh để lại KV ở mọi tầng full-attention (cache ban đầu $\propto1296$), và mỗi bước decode phải đọc chúng. Giảm token ảnh (gộp mạnh hơn, FastV) giảm prefill và KV nhưng có thể mất chữ nhỏ.
</details>

---

**Kết luận ngắn:** Ba ý cần mang theo: (1) mọi thứ xoay quanh attention $\text{softmax}(QK^\top/\sqrt{d_k})V$ với chi phí $O(n^2)$ và KV cache $\propto n$; (2) decode batch-1 bị chặn bởi **byte đọc**, không phải FLOPs, nên mọi tối ưu cho GPU nhỏ là bài toán giảm byte và số bước tuần tự; (3) attention tuyến tính/delta đổi KV cache tăng dần thành trạng thái cố định, đổi lại khả năng truy hồi chính xác, nên hybrid là điểm giữa.
