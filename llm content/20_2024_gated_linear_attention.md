# Gated Linear Attention Transformers with Hardware-Efficient Training (2024)

**Tác giả:** Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim (MIT, MIT-IBM Watson AI Lab)  
**Venue / arXiv:** ICML 2024 (PMLR 235); arXiv:2312.06635 (v6, 8/2024)  
**Nguồn:** https://arxiv.org/abs/2312.06635  
**Chủ đề:** Cổng quên phụ thuộc dữ liệu cho linear attention $S_t=\mathrm{Diag}(\alpha_t)S_{t-1}+k_tv_t^\top$; thuật toán chunkwise nhận biết I/O (FlashLinearAttention); chunk hai tầng + log-space để ổn định số và dùng tensor core.  
**Vị trí trong lộ trình:** Đọc sau 18 (linear attention = RNN ma trận) và 19 (RetNet: $\gamma$ hằng + dạng chunkwise). Bài này (a) làm $\gamma$ thành $\alpha_t$ **phụ thuộc token**, (b) giải bài toán **làm thế nào huấn luyện hiệu quả trên GPU** khi đã có cổng. Bài sau: 21 (Mamba) và 22 (Mamba-2/SSD) cùng giải bài toán chọn lọc theo cách khác; 23-24 (DeltaNet, Gated DeltaNet) thêm delta rule và đưa vào đúng khung chunkwise của bài này.

> Tài liệu học tập bằng tiếng Việt, ghi chú diễn giải, **không phải bản dịch**. Suy luận áp dụng cho dự án và con số "tự tính" là của chúng ta. Ký hiệu: dùng **vector cột** $q_t,k_t\in\mathbb R^{d_k}$, $v_t\in\mathbb R^{d_v}$, $S_t\in\mathbb R^{d_k\times d_v}$, $o_t=S_t^\top q_t$ (paper dùng vector hàng $o_t=q_tS_t$; công thức tương đương sau chuyển vị).

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

Hai vấn đề tồn tại sau bài 18-19 (theo paper):

1. **Hiệu quả phần cứng.** Các cài đặt linear attention "không nhận biết I/O" nên thực tế **chậm hơn FlashAttention-2** ở độ dài vừa phải (1K-4K), dù ít FLOPs hơn. Lý do: GPU hiện đại có tensor core (matmul fp16 nhanh khoảng 16× so với CUDA core trên A100) và phân cấp bộ nhớ (HBM lớn-chậm, SRAM nhỏ-nhanh); thuật toán hồi quy thuần (cộng từng phần tử) có *arithmetic intensity* thấp, không dùng tensor core.
2. **Chất lượng.** Linear attention kém softmax. RetNet/TransNormerLLM cải thiện nhờ suy giảm $\gamma$ nhưng $\gamma$ **toàn cục, không phụ thuộc dữ liệu**; trong RNN một chiều, cổng quên phụ thuộc dữ liệu được biết là quan trọng (LSTM, GRU).

**Khái niệm nền (nhắc lại).**

- *Forget gate / cổng quên:* hệ số $\in(0,1)$ nhân vào trạng thái cũ trước khi cộng thông tin mới; gần 0 = quên hết, gần 1 = giữ nguyên.
- *Tensor core:* đơn vị tính ma trận chuyên dụng của GPU NVIDIA; chỉ nhanh nếu việc tính được viết dưới dạng matmul nửa chính xác với kích thước phù hợp (bội số 16).
- *Kernel fusion, tiling:* gộp nhiều phép toán vào một kernel và xử lý theo khối để dữ liệu ở lại SRAM, tránh đọc/ghi HBM (ý tưởng của FlashAttention, bài 44).
- *Prefix-sum / parallel scan:* tính tổng tích lũy song song; với trạng thái ma trận, nó đòi hỏi vật chất hóa (materialize) trạng thái của **từng** bước trong HBM, cực tốn I/O.
- *Perplexity (ppl):* hàm mũ của loss trung bình, thấp hơn tốt hơn.

## 2. Ý tưởng chính

Giữ mọi thứ của linear attention nhưng cho trạng thái $S_t$ **quên có chọn lọc**: một cổng $\alpha_t\in(0,1)^{d_k}$ do chính token $x_t$ sinh ra, nhân theo từng chiều khóa. Dạng này tổng quát hóa RetNet ($\alpha_t\equiv\gamma$) và gần với Mamba nhưng dùng cổng theo *chiều khóa* với tham số rất ít (low-rank). Điểm khó là huấn luyện: ma trận suy giảm $D_{nm}=\gamma^{n-m}$ của RetNet không còn là Toeplitz đơn giản mà thành tích tích lũy $\prod\alpha$ theo từng chiều. Giải pháp: (i) **FlashLinearAttention**, thuật toán chunkwise I/O-aware cho linear attention thường; (ii) mở rộng cho cổng bằng cách tách các hệ số suy giảm khỏi chunk (tích lũy cục bộ trong chunk), cộng **chunk hai tầng** để phần lớn phép tính vẫn là matmul nửa chính xác trên tensor core, chỉ phần đường chéo nhỏ cần log-space fp32. Kết quả: nhanh hơn FlashAttention-2 ngay cả ở 1K, ngang Transformer++/Mamba về chất lượng ở quy mô 340M-1.3B, mạnh về ngoại suy độ dài và các tác vụ cần recall hơn Mamba/RetNet.

## 3. Phương pháp chi tiết

### 3.1 Nền: linear attention ở ba dạng (§2 của paper)

Softmax attention có causal mask: $O=\mathrm{softmax}\big((QK^\top)\odot M\big)V$. Linear attention thay $\exp(q^\top k)$ bằng tích của feature map; paper dẫn các kết quả gần đây rằng **$\phi=$ identity và không chuẩn hóa** đã chạy tốt (RetNet), nên:

$$
\boxed{S_t=S_{t-1}+k_tv_t^\top,\qquad o_t=S_t^\top q_t}\qquad\text{(hồi quy)},
$$

$$
O=\big((QK^\top)\odot M\big)V\qquad\text{(song song, } M_{ij}=[i\ge j]\text{, chi phí } O(L^2d)).
$$

Mask $M$ chặn việc tái kết hợp thành $Q(K^\top V)$ — ghi chú đã nêu ở bài 18.

**Dạng chunkwise.** Chia thành các chunk dài $C$; $S_{[i]}:=S_{iC}$ là trạng thái sau $i$ chunk. Liên chunk (Eq. 2):

$$
S_{[i+1]}=S_{[i]}+\sum_{j=iC+1}^{(i+1)C}k_jv_j^\top=S_{[i]}+K_{[i+1]}^\top V_{[i+1]},
$$

và đầu ra chunk $i+1$ (viết theo hàng như paper):

$$
O_{[i+1]}=\underbrace{Q_{[i+1]}S_{[i]}}_{\text{inter-chunk}}+\underbrace{\big((Q_{[i+1]}K_{[i+1]}^\top)\odot M\big)V_{[i+1]}}_{\text{intra-chunk}} .
$$

Chi phí: intra $O(C^2d+Cd^2)$ mỗi chunk, inter $O(Cd^2)$; tổng $L/C$ chunk:

$$
O\!\Big(\tfrac LC(C^2d+Cd^2)\Big)=O(LCd+Ld^2)<O(L^2d)\ \text{khi } L>d .
$$

$C=L$ cho dạng song song; $C=1$ cho dạng hồi quy.

### 3.2 Nguyên tắc phần cứng (§3.1-3.2)

| Dạng | Ưu | Nhược (theo paper) |
|---|---|---|
| Hồi quy | FLOPs ít nhất | phép toán theo phần tử, không dùng tensor core, intensity thấp; lưu trạng thái mọi bước vào HBM thì I/O cao; parallel scan phải vật chất hóa trạng thái 2D mọi bước |
| Song song | dùng matmul, có thể tối ưu I/O như FlashAttention | FLOPs bậc hai theo $L$ |
| Chunkwise | gần như toàn matmul (bội 16 → tensor core); $C$ là núm vặn giữa FLOPs và song song | các cài đặt trước chưa I/O-aware ⇒ chậm hơn FlashAttention ở 2K-4K |

### 3.3 FlashLinearAttention (§3.3, Alg. 1)

Hai biến thể forward:

- **Không vật chất hóa (non-materialization):** duyệt các chunk tuần tự, giữ $S$ ở SRAM; song song theo batch/head/chiều head nhưng *không* song song theo chiều chuỗi (đủ khi batch lớn).
- **Vật chất hóa (materialization):** trước tiên chạy hồi quy liên chunk để ghi mọi $S_{[n]}$ vào HBM, rồi tính $O_{[n]}$ song song theo chunk; tăng song song theo chuỗi (hợp khi batch nhỏ, chuỗi dài) với giá +10-20% bộ nhớ, được giảm bằng **recomputation** (bỏ trạng thái sau forward, tính lại ở backward) — là mặc định.

```
# FlashLinearAttention, forward, bản không vật chất hóa (rút gọn Alg.1)
S = 0                                   # nằm trên SRAM
for n in 1..N:                          # N = L / C chunk
    load Q[n], K[n], V[n]  (HBM -> SRAM)
    O[n] = Q[n] @ S + ((Q[n] @ K[n].T) * M) @ V[n]   # inter + intra, toàn matmul
    S    = S + K[n].T @ V[n]
    store O[n]  (SRAM -> HBM)
```

Hình 2: trên H100, batch 32, 16 head, head dim 64, chunk 64, hai biến thể FlashLinearAttention nhanh hơn FlashAttention-2, tuyên bố của paper là "even on short (e.g., 1K) sequences" (giá trị từng điểm không đọc được từ text); bản PyTorch thuần chunkwise chậm hơn rõ — chứng minh giá trị của I/O-awareness.

### 3.4 Gated Linear Attention: dạng hồi quy (§4.1)

Cổng quên 2D: $G_t\in(0,1)^{d_k\times d_v}$. Dạng tổng quát (Hadamard) $S_t=G_t\odot S_{t-1}+k_tv_t^\top$ bao trùm nhiều mô hình (Bảng 1 dưới). Nhưng tham số hóa $G_t$ đầy đủ tốn $d\cdot d_k\cdot d_v$ tham số. Paper chọn dạng **outer product với vector 1**:

$$
G_t=\alpha_t\mathbf 1^\top,\quad \alpha_t\in(0,1)^{d_k}
\ \Longrightarrow\ \boxed{S_t=\mathrm{Diag}(\alpha_t)\,S_{t-1}+k_tv_t^\top},\qquad o_t=S_t^\top q_t .
$$

Tức mỗi *hàng* của $S$ (ứng với một chiều khóa $r$) bị nhân cùng một số $\alpha_{t,r}$ — mọi chiều giá trị chia sẻ cùng hệ số quên cho một chiều khóa. Đây là điểm trung gian giữa cổng vô hướng (Mamba-2, mLSTM, gated retention: $G_t=\gamma_t\mathbf 1\mathbf 1^\top$) và cổng đầy đủ $\alpha_t\beta_t^\top$ (DFW).

**Bảng 1 của paper (một số hàng chính) — cách các mô hình tham số hóa $G_t$:**

| Mô hình | $G_t$ | Ghi chú |
|---|---|---|
| Mamba | $\exp(-(\Delta_t\mathbf 1^\top)\odot\exp(A))$, $\Delta_t=\mathrm{softplus}(x_tW_1W_2)$ | $A\in\mathbb R^{d_k\times d_v}$ học được; đầy đủ hạng nhưng không viết được dạng matmul |
| Mamba-2 | $\gamma_t\mathbf 1\mathbf 1^\top$, $\gamma_t=\exp(-\mathrm{softplus}(x_tW_\gamma)\exp(a))$ | vô hướng ⇒ dạng matmul, tensor core |
| mLSTM | $\gamma_t\mathbf 1\mathbf 1^\top$, $\gamma_t=\sigma(x_tW_\gamma)$ | vô hướng |
| Gated Retention | $\gamma_t\mathbf 1\mathbf 1^\top$, $\gamma_t=\sigma(x_tW_\gamma)^{1/\tau}$ | vô hướng |
| RWKV-6 | $\alpha_t\mathbf 1^\top$, $\alpha_t=\exp(-\exp(x_tW_\alpha))$ | cùng họ với GLA |
| HGRN-2 | $\alpha_t\mathbf 1^\top$, $\alpha_t=\gamma+(1-\gamma)\sigma(x_tW_\alpha)$ | |
| **GLA** | $\alpha_t\mathbf 1^\top$, $\alpha_t=\sigma(x_tW_{\alpha1}W_{\alpha2})^{1/\tau}$ | $W_{\alpha1}\in\mathbb R^{d\times16}$, $W_{\alpha2}\in\mathbb R^{16\times d_k}$ |

### 3.5 Dạng song song của GLA (suy dẫn)

Khai triển hồi quy. Với $t\ge i$: ảnh hưởng của $k_iv_i^\top$ lên $S_t$ là $\prod_{j=i+1}^{t}\mathrm{Diag}(\alpha_j)\,k_iv_i^\top$, tức nhân từng chiều $r$ với $\prod_{j=i+1}^t\alpha_{j,r}$.

$$
S_t=\sum_{i=1}^{t}\Big(\big(\textstyle\prod_{j=i+1}^{t}\alpha_j\big)\odot k_i\Big)v_i^\top .
$$

Đặt **tích tích lũy** $b_t:=\prod_{j=1}^{t}\alpha_j\in(0,1)^{d_k}$ (nhân từng phần tử). Khi đó $\prod_{j=i+1}^t\alpha_j=b_t\oslash b_i$ và

$$
o_t=S_t^\top q_t=\sum_{i=1}^{t}v_i\,\big(q_t\odot b_t\big)^{\!\top}\!\big(k_i\oslash b_i\big)
\quad\Longrightarrow\quad
\boxed{O=\Big(\underbrace{\big((Q\odot B)\,(K\oslash B)^\top\big)}_{P}\odot M\Big)V},
$$

$B\in(0,1)^{L\times d_k}$ xếp các $b_t$. Đối chiếu RetNet: nếu $\alpha_t=\gamma\mathbf 1$ thì $b_t=\gamma^t\mathbf 1$, và $P_{ti}=\gamma^{t-i}q_t^\top k_i$ — đúng dạng $(QK^\top\odot D)$ của bài 19.

**Vấn đề số học.** $b_t$ là tích của rất nhiều số $<1$ nên cực nhỏ khi $t$ lớn ⇒ $K\oslash B$ nổ tung. Cách chữa tạm: tính ở log-space

$$
P_{ij}=\sum_{r=1}^{d_k}Q_{ir}K_{jr}\exp(\log B_{ir}-\log B_{jr}),\quad i\ge j,
$$

nhưng đây **không còn là matmul chuẩn** nên không dùng được half-precision tensor core.

### 3.6 Dạng chunkwise của GLA (§4.2, suy dẫn)

Trong chunk thứ $c$ (các vị trí $(c-1)C+1..cC$), gọi $b_0:=b_{(c-1)C}$ là tích lũy ngay trước chunk, $b_{end}:=b_{cC}$ là cuối chunk. Định nghĩa ba đại lượng suy giảm (chỉ phụ thuộc cổng, cục bộ):

$$
\Lambda_t=\frac{b_t}{b_0}\ (\text{từ đầu chunk đến }t),\qquad
\Gamma_t=\frac{b_{end}}{b_t}\ (\text{từ }t\text{ đến cuối chunk}),\qquad
\gamma_c=\frac{b_{end}}{b_0}\ (\text{cả chunk}).
$$

Đây là các tích của tối đa $C$ cổng nên không nhỏ đến mức tràn số như $b_t$ toàn chuỗi.

**Cập nhật trạng thái liên chunk.** Token $t$ trong chunk còn phải suy giảm đến cuối chunk bởi $\prod_{j>t}^{cC}\alpha_j=\Gamma_t$; trạng thái cũ bị suy giảm toàn chunk bởi $\gamma_c$:

$$
\boxed{S_{[c]}=\mathrm{Diag}(\gamma_c)\,S_{[c-1]}+\big(K_{[c]}\odot\Gamma_{[c]}\big)^{\!\top}V_{[c]}}
$$

(paper: Eq. ở §4.2 với ký hiệu $\gamma_{i+1}$, $\Gamma_{[i+1]}$ tương ứng).

**Đầu ra inter-chunk.** Đóng góp của mọi token *trước chunk* tại vị trí $t$ là $\mathrm{Diag}(\Lambda_t)S_{[c-1]}$, nên

$$
O^{\text{inter}}_{[c]}=\big(Q_{[c]}\odot\Lambda_{[c]}\big)\,S_{[c-1]} .
$$

**Đầu ra intra-chunk.** Dùng dạng song song cục bộ: $b_t/b_i=\Lambda_t/\Lambda_i$ ($i,t$ cùng chunk):

$$
O^{\text{intra}}_{[c]}=\Big(\big((Q\odot\Lambda)(K\oslash\Lambda)^\top\big)\odot M\Big)V\quad(\text{trong chunk}),
$$

với $K\oslash\Lambda$ vẫn có thể bất ổn nếu $C$ lớn và cổng nhỏ — bước tiếp theo xử lý.

**Tương đương kiểm chứng:** cho $\alpha_t\equiv\gamma$ hằng, $\Lambda_t=\gamma^{j}$ (j = vị trí trong chunk), $\Gamma_t=\gamma^{C-j}$, $\gamma_c=\gamma^C$ — trùng đúng $\xi_j=\gamma^j$, $\zeta_j=\gamma^{C-j}$ của RetNet (bài 19).

### 3.7 Hiệu quả phần cứng cho GLA: chunk hai tầng (§4.3)

Tại sao cần: phần intra-chunk có $\exp(\log B_{ir}-\log B_{jr})$ không là matmul. **Secondary-level chunking:** chia mỗi chunk thành sub-chunk. Ma trận $P$ (intra-chunk) chia thành các ô:

- Ô giữa **các sub-chunk khác nhau** (ô cam trong Hình 3): tính bằng matmul nửa chính xác trên tensor core, vì hệ số suy giảm giữa hai sub-chunk có thể tách thành tích $Q$-phía và $K$-phía qua các mốc đầu/cuối sub-chunk:
$$
P_{[i][j]}=\big(Q_{[i]}\odot\Lambda_{[i]}\big)\Big(K_{[j]}\odot\Gamma_{[j]}\odot\tfrac{b_{iC}}{b_{(j+1)C}}\Big)^{\!\top}.
$$
- Ô **trong cùng sub-chunk** (ô hồng, đường chéo nhỏ): tính trực tiếp ở log-space, fp32 (ổn định, nhưng không dùng tensor core). Vì chỉ là các khối nhỏ trên đường chéo nên tổng FLOPs không-nửa-chính-xác giảm mạnh.

Hai kỹ thuật khác: (i) **tính $d\log\alpha_t$ trong bộ nhớ hằng** — không cần vật chất hóa trạng thái $L\times d\times d$ mọi bước ở backward nhờ công thức đóng $d\log b_t=q_t\odot dq_t-k_t\odot dk_t$, $d\log\alpha_t=\sum_{t\le i\le L}d\log b_i$ (paper, §4.3); (ii) recomputation như trên.

Chi phí tổng quát giống linear attention thường (§3.1) cộng phần đường chéo log-space.

### 3.8 GLA Transformer (khối đầy đủ, §4.4)

Với $H$ head, mỗi head $h$:

$$
S_t^h=\mathrm{Diag}(\alpha_t^h)S_{t-1}^h+k_t^h(v_t^h)^\top,\qquad o_t^h=(S_t^h)^\top q_t^h,\qquad
o_t'=\mathrm{concat}\big(\mathrm{LN}(o_t^1),\dots,\mathrm{LN}(o_t^H)\big),
$$

$$
r_t=\mathrm{Swish}(x_tW_r+b_r),\qquad y_t=(r_t\odot o_t')\,W_O .
$$

Cổng quên được tham số hóa **low-rank** và làm "chậm" bằng nhiệt độ $\tau=16$:

$$
\boxed{\alpha_t=\sigma\!\big(x_tW_{\alpha1}W_{\alpha2}+b_\alpha\big)^{1/\tau}},\quad W_{\alpha1}\in\mathbb R^{d\times16},\ W_{\alpha2}\in\mathbb R^{16\times d_k},\ \tau=16 .
$$

Lấy lũy thừa $1/\tau$ khiến $\alpha_t$ gần 1 hơn (ví dụ $\sigma=0.5\Rightarrow\alpha=0.5^{1/16}\approx0.958$), tức *quên chậm theo mặc định*.

Phân bổ tham số: $d_k=d/2$, $d_v=d$ (tổng), 4 head ⇒ mỗi head $d_k'=d/8$, $d_v'=d/4$. Một lớp GLA có xấp xỉ $4d^2$ tham số (như attention softmax): $W_Q,W_K\in\mathbb R^{d\times d/2}$, $W_V,W_r,W_O\in\mathbb R^{d\times d}$, cộng $W_\alpha$ nhỏ. Khối đầy đủ: $Y=\mathrm{GLA}(\mathrm{LN}(X))+X$, rồi FFN **SwiGLU** (bài 11) với residual. (Lưu ý: công thức trong paper viết $X^{(l+1)}=\mathrm{SwiGLU}(\mathrm{LN}(Y))+X$ — theo chúng tôi, có vẻ là lỗi in, phải là $+Y$ như residual chuẩn.)

## 4. Thí nghiệm và kết quả chính

**Thiết lập.** Dữ liệu: SlimPajama (tập con 100B token), tokenizer Mistral. Hai cỡ: 340M (15B token, batch 0.5M) và 1.3B (100B token, batch 2M). AdamW, LR tối đa $3\times10^{-4}$, cosine, weight decay 0.01, clip 1.0. Baseline: **Transformer++** (kiến trúc LLaMA: RoPE, SwiGLU, RMSNorm), **RetNet** (thay FFN bằng SwiGLU cho công bằng), **Mamba**; tất cả huấn luyện cùng token, cùng dữ liệu.

### 4.1 Language modeling và zero-shot (Bảng 2)

| Cỡ | Mô hình | Wiki ppl ↓ | LMB ppl ↓ | LMB acc | PIQA | Hella | Wino | ARC-e | ARC-c | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| 340M / 15B | Transformer++ | 28.39 | 42.69 | 31.0 | 63.3 | 34.0 | 50.4 | 44.5 | 24.2 | 41.2 |
| | RetNet | 32.33 | 49.19 | 28.6 | 63.5 | 33.5 | 52.5 | 44.5 | 23.4 | 41.0 |
| | Mamba | 28.39 | 39.66 | 30.6 | 65.0 | 35.4 | 50.1 | 46.3 | 23.6 | 41.8 |
| | **GLA** | 28.65 | 43.35 | 30.3 | 64.8 | 34.5 | 51.4 | 45.1 | 22.7 | 41.5 |
| 1.3B / 100B | Transformer++ | 16.85 | 13.44 | 48.9 | 70.8 | 49.6 | 53.6 | 56.0 | 26.5 | 50.9 |
| | RetNet | 18.64 | 17.27 | 43.3 | 70.0 | 47.3 | 52.5 | 54.8 | 25.6 | 48.9 |
| | Mamba | 17.06 | 13.89 | 46.2 | 72.2 | 40.1 | 54.1 | 59.0 | 28.2 | 50.0 |
| | **GLA** | 17.22 | 14.47 | 46.9 | 71.8 | 49.8 | 53.9 | 57.2 | 26.6 | **51.0** |

(Acc là chuẩn hóa độ dài ở HellaSwag/ARC-c; cột cuối là trung bình các benchmark dùng accuracy.) Kết luận paper: GLA, với cổng phụ thuộc dữ liệu, tốt hơn RetNet trên mọi tác vụ, và ngang Mamba/Transformer++ ở quy mô này.

### 4.2 Tác vụ cần recall (Bảng 3)

Các tác vụ trích thông tin/đọc hiểu theo Arora et al. 2024 (điểm cao hơn tốt hơn):

| Cỡ | Mô hình | FDA | SWDE | SQuAD |
|---|---|---|---|---|
| 340M | Transformer++ | 21.4 | 42.2 | 22.1 |
| | RetNet | 2.9 | 13.3 | 27.6 |
| | Mamba | 2.1 | 12.4 | 23.0 |
| | GLA | 8.1 | 18.6 | 27.2 |
| 1.3B | Transformer++ | 27.4 | 66.6 | 31.5 |
| | RetNet | 14.3 | 42.8 | 34.7 |
| | Mamba | 6.2 | 41.4 | 35.2 |
| | GLA | 19.9 | 50.6 | 42.6 |

Mô hình dưới bậc hai thua Transformer rõ ở FDA, SWDE; GLA là mô hình dưới bậc hai tốt nhất trong nhóm, được cho là nhờ **trạng thái lớn hơn Mamba** và **cơ chế chọn lọc** (so với RetNet). Với tác vụ tổng hợp MQAR (multi-query associative recall; Hình 4), các mô hình trạng thái ma trận (Mamba/RetNet/GLA) vượt Hyena/RWKV-4 và GLA vượt RetNet; attention chuẩn đạt tuyệt đối.

### 4.3 Ngoại suy độ dài (1.3B, 100B token)

Huấn luyện ở 2K: Transformer không ngoại suy được ngoài độ dài huấn luyện; **Mamba khó ngoại suy quá 4K** (PG19); **GLA/RetNet tổng quát hóa đến ~18K** (SlimPajama test) mà ppl không suy giảm đáng kể. Thử huấn luyện 24K qua truncated BPTT (12 đoạn 2K, truyền trạng thái cuối đoạn trước, không truyền gradient): với GLA chênh lệch ppl rất nhỏ so với huấn luyện trực tiếp, nên TBPTT là cách kinh tế; Mamba hưởng lợi rõ khi huấn luyện 8K và khi đó ngang GLA.

### 4.4 Hiệu quả huấn luyện (1.3B, H100)

Hình 6: thông lượng và bộ nhớ theo độ dài/batch (2048/8, 4096/4, 8192/2, 16384/1): mọi mô hình có bộ nhớ tuyến tính, chênh lệch bộ nhớ nhỏ; **Mamba tụt sau Transformer++ và GLA về thông lượng, GLA có lợi thế rõ hơn khi độ dài > 4096**. Tuyên bố ở tóm tắt: FlashLinearAttention nhanh hơn FlashAttention-2 ngay ở 1K; GLA có thông lượng cao hơn Mamba cùng cỡ.

## 5. Phân tích / ablation

Bảng 4 (340M, 7B token, perplexity huấn luyện trung bình 200 bước cuối; thấp hơn tốt hơn):

| Biến thể | Train ppl |
|---|---|
| **GLA (4 head)** | **14.77** |
| Không cổng (linear attention thường) | 23.21 |
| Cổng độc lập dữ liệu, vô hướng (≈ RetNet) | 16.55 |
| Cổng vô hướng phụ thuộc dữ liệu | 15.56 |
| 8 head (head dim nhỏ) | 15.29 |
| 1 head (head dim lớn) | 14.61 |

Rút ra (theo paper): (1) **cổng là yếu tố số một** (23.21 → 14.77); (2) *phụ thuộc dữ liệu* tốt hơn *hằng* (16.55 → 15.56); (3) *cổng mịn theo chiều khóa* còn tốt hơn *vô hướng* (15.56 → 14.77) nên cần thiết; (4) head dim lớn hơn tốt hơn nhưng 1 head chỉ nhỉnh hơn một chút (14.61 vs 14.77) với bộ nhớ GPU nhiều hơn rất nhiều, nên chọn 4 head. Thử $G_t=\alpha_t\beta_t^\top$ chỉ cải thiện không đáng kể so với $\alpha_t\mathbf 1^\top$ (nêu ở chú thích).

## 6. Giới hạn

**Do tác giả nêu:**
- Chưa chạy quy mô lớn hơn do hạn chế tài nguyên (chỉ 340M/1.3B, tối đa 100B token); không rõ GLA scale thế nào; tác giả *dự đoán* lợi thế hiệu quả so với Mamba sẽ lớn hơn ở >7B (tensor core, tensor parallelism) — đây là dự đoán, không đo.
- Mamba trong so sánh dùng mã chính thức; Transformer++ dùng FlashAttention-2; có lưu ý so sánh khác biệt cài đặt.
- Hướng tương lai: áp dụng sang các modality khác.

**Do chúng ta nhận xét:**
- Ablation Bảng 4 chỉ ở 340M/7B token; chênh lệch 0.2-1 ppl có thể không giữ ở quy mô lớn.
- Ở tác vụ recall, GLA vẫn thua Transformer++ rõ (FDA 19.9 vs 27.4; SWDE 50.6 vs 66.6 ở 1.3B) — xác nhận hạn chế "trạng thái hữu hạn".
- Cổng chỉ **xóa** thông tin (nhân $<1$), không **sửa** nội dung đã lưu cho cùng một khóa. Delta rule (23-24) giải quyết điều này.
- Chunk hai tầng + log-space làm cài đặt phức tạp; cần kernel Triton chuyên dụng (flash-linear-attention).
- Đo thông lượng trên H100; chưa có số liệu GPU nhỏ ≤ 8 GB hay batch 1 decode.

## 7. Liên hệ với các bài khác

- **18:** GLA = linear attention + cổng. $\alpha_t\equiv\mathbf 1$ ⇒ bài 18 (không mẫu số).
- **19 RetNet:** $\alpha_t\equiv\gamma\mathbf 1$ ⇒ RetNet; dạng chunkwise của RetNet là trường hợp riêng của §3.6.
- **21 Mamba:** Bảng 1 xếp Mamba vào cùng khung $S_t=G_t\odot S_{t-1}+\dots$ nhưng $G_t$ đầy đủ hạng, tính bằng scan trong SRAM (không mở rộng trạng thái quá ~16 chiều); GLA dùng trạng thái lớn hơn nhờ matmul/tensor core.
- **22 Mamba-2:** $G_t=\gamma_t\mathbf 1\mathbf 1^\top$ vô hướng ⇒ dạng matmul thuần; GLA (cổng theo chiều khóa) là tổng quát hơn.
- **23-24 DeltaNet, Gated DeltaNet:** cùng khung chunkwise; thêm delta rule và (ở 24) cổng vô hướng $\alpha_t$ kiểu Mamba-2.
- **26 Kimi Linear:** (theo hiểu biết của chúng tôi) dùng cổng mịn theo chiều kiểu GLA trong khung delta; sẽ kiểm tra ở bài đó.
- **44-45 FlashAttention:** tư tưởng I/O-awareness và tiling được chuyển sang linear attention.
- **06/17:** hướng sliding-window, bổ sung cho recall chính xác mà GLA yếu.

## 8. Ứng dụng cho dự án (suy luận)

**Vị trí của GatedDeltaNet.** Theo hiểu biết của chúng tôi (cần xác nhận ở bài 24), GDN dùng một cổng suy giảm *vô hướng theo head* $\alpha_t\in(0,1)$ như Mamba-2 và một delta rule với hệ số ghi $\beta_t$. Bài này cho bạn nền: (a) ý nghĩa của $\alpha_t$ như cổng quên phụ thuộc dữ liệu; (b) vì sao thuật toán huấn luyện là chunkwise với $\Lambda,\Gamma,\gamma_c$ — GDN dùng cùng cấu trúc cộng thêm các hệ số delta; (c) tại sao cổng *vô hướng* chọn để tối ưu matmul, đánh đổi với cổng *theo chiều* (Bảng 4: 15.56 vs 14.77 — chênh nhỏ ở 340M).

**Chi phí (tự tính).** Giả định $d=1024$, 24 lớp, $T=1500$, 4 head GLA với $d_k=d/2=512$, $d_v=1024$ (mỗi head $128\times256$), trạng thái fp32, batch 1:

| Đại lượng | Công thức | Giá trị |
|---|---|---|
| Trạng thái / lớp | $H\,d_k'd_v'=4\cdot128\cdot256$ | $131{,}072$ phần tử |
| Trạng thái 24 lớp | | $3.1$M phần tử $\approx12.6$ MB (fp32) |
| Lưu lượng trạng thái mỗi token decode (đọc + ghi) | $2\cdot3.1\text{M}\cdot4\,\mathrm B$ | $\approx25$ MB |
| KV cache full attention ($d_{kv}=1024$, bf16) ở $T=1500$ (đọc mỗi token) | | $\approx147$ MB |
| Thời gian chỉ riêng truy cập bộ nhớ (băng thông *giả định* 300 GB/s) | | $\approx0.08$ ms (trạng thái) vs $\approx0.49$ ms (KV) |
| Tham số cổng / lớp | $d\cdot16+16\cdot d_k=1024\cdot16+16\cdot512$ | $24{,}576$ (không đáng kể) |
| FLOPs prefill chunkwise / lớp ($C=64$, $T=1500$) | mỗi head: $2TCd_k'+2TCd_v'+4Td_k'd_v'$ | $\approx270$M/head, $\approx1.1$ GFLOPs/lớp (so với $\approx9.2$ GFLOPs của full attention không mask) |
| Tham số QKVO + gate output | $2\cdot d\cdot d/2+3d^2$ | $\approx4.2$M $=4d^2$ |

Nhận định: ở batch 1 decode, lớp kiểu GLA/GDN **không phải đọc KV** (tiết kiệm ~0.4 ms/token trong giả định) nhưng phải đọc-ghi trạng thái (~25 MB/token) — nhỏ hơn KV 6×. Cùng lúc phép tính theo phần tử có intensity thấp nên kernel decode phải gộp (fused); nếu cài đặt kém thì không thu được lợi ích trên.

**Ý nghĩa cho cắt tỉa/chưng cất (suy đoán):**
- Bảng 3 gợi ý rằng khả năng *recall chính xác* của trạng thái hữu hạn bị hạn chế (FDA: 19.9 vs 27.4). Với OCR, nếu decoder phải sao chép/trích chính xác từ ngữ cảnh dài (đặc biệt từ token ảnh), các lớp full-attention trong hybrid là "kho recall" — nên **ưu tiên giữ các lớp full-attention, cắt lớp hồi quy trước** hoặc đo riêng độ nhạy của từng loại lớp bằng D_lin/D_res của bạn. Cần đo, chưa chứng minh.
- Cổng $\alpha_t$ có thể dùng làm chẩn đoán: head có $\alpha_t\approx1$ gần như luôn (nhớ vô hạn) hay $\approx0$ (nhớ rất ngắn) là ứng viên cắt hoặc thay bằng cửa sổ trượt (bài 17).
- Chunkwise ($C$ = bội số 16, ví dụ 64) và nhu cầu fp32 cho phần đường chéo log-space ảnh hưởng đến thiết kế kernel khi lượng tử hóa mô hình thu nhỏ; kiểm tra tác động lượng tử hóa lên $\alpha_t$ gần 1 (sai số làm nhớ sai chân trời).
- Chưng cất: mục tiêu thích hợp cho head hồi quy là khớp *đầu ra* hoặc *trạng thái* của teacher; trạng thái $S\in\mathbb R^{d_k\times d_v}$ đủ nhỏ để so sánh trực tiếp (chưa thử).

## 9. Thí nghiệm gợi ý

1. **Tự dựng ba dạng GLA** (hồi quy, song song log-space, chunkwise) trên tensor ngẫu nhiên và so sai số; thử $\alpha$ nhỏ (0.5) để thấy dạng song song ngây thơ (chia cho $b_i$) nổ số như paper mô tả.
2. **Đo chân trời nhớ $\mathbb E[\log\alpha]$ theo head/lớp** của GDN trên 200 trang OCR; so với vị trí các ký tự được sao chép để xem head nào thực sự giữ thông tin xa.
3. **So cổng vô hướng và theo chiều** ở mô hình nhỏ (~100M) trên tác vụ copy-with-distractors và nhận dạng; chỉ để cảm nhận độ chênh (paper: 15.56 vs 14.77).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức |
|---|---|
| Linear attention | $S_t=S_{t-1}+k_tv_t^\top$, $o_t=S_t^\top q_t$ |
| GLA hồi quy | $S_t=\mathrm{Diag}(\alpha_t)S_{t-1}+k_tv_t^\top$ |
| Cổng | $\alpha_t=\sigma(x_tW_{\alpha1}W_{\alpha2}+b)^{1/16}$, $W_{\alpha1}\in\mathbb R^{d\times16}$ |
| Tích tích lũy | $b_t=\prod_{j\le t}\alpha_j$ |
| GLA song song | $O=\big(((Q\odot B)(K\oslash B)^\top)\odot M\big)V$ |
| Log-space | $P_{ij}=\sum_rQ_{ir}K_{jr}e^{\log B_{ir}-\log B_{jr}}$ |
| Chunk decay | $\Lambda_t=b_t/b_0$, $\Gamma_t=b_{end}/b_t$, $\gamma_c=b_{end}/b_0$ |
| Trạng thái liên chunk | $S_{[c]}=\mathrm{Diag}(\gamma_c)S_{[c-1]}+(K\odot\Gamma)^\top V$ |
| Đầu ra chunk | $O=(Q\odot\Lambda)S_{[c-1]}+\text{intra}$ |
| Chi phí | $O(LCd+Ld^2)$; $C=L$ song song, $C=1$ hồi quy |
| Chunk hai tầng | ô khác sub-chunk: matmul nửa chính xác; ô đường chéo: log-space fp32 |
| Block | $y=(r\odot\mathrm{LN\text{-}heads}(o))W_O$, $r=\mathrm{Swish}(xW_r)$; $d_k=d/2$, $d_v=d$, 4 head |

## 11. Kết luận ngắn

GLA nâng linear attention lên mức cạnh tranh bằng một cổng quên theo chiều khóa phụ thuộc dữ liệu, và quan trọng không kém là **chỉ ra cách huấn luyện nhanh trên GPU**: chunkwise có vật chất hóa + recomputation, chunk hai tầng, log-space cục bộ. Cấu trúc $S_{[c]}=\mathrm{Diag}(\gamma_c)S_{[c-1]}+(K\odot\Gamma)^\top V$ và $O=(Q\odot\Lambda)S_{[c-1]}+\text{intra}$ chính là khung mà DeltaNet và Gated DeltaNet (bài 23-24) kế thừa, nên đây là bài nên đọc kỹ nhất trong chặng này nếu bạn muốn hiểu lớp hồi quy của decoder mình.

**Nguồn gốc:** https://arxiv.org/abs/2312.06635
