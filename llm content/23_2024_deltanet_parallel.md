# Parallelizing Linear Transformers with the Delta Rule over Sequence Length (2024)

**Tác giả:** Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim.  
**Venue / arXiv:** arXiv:2406.06484; NeurIPS 2024. Code: thư viện `flash-linear-attention`.  
**Nguồn:** https://arxiv.org/abs/2406.06484  
**Chủ đề:** Delta rule trong linear attention (DeltaNet); biểu diễn WY của tích ma trận Householder; thuật toán chunkwise song song theo chiều chuỗi; hybrid DeltaNet + attention.  
**Vị trí trong lộ trình:** Đọc trước: 18 (Linear Transformers), 21 (Mamba), 22 (Mamba-2/SSD). Bài này cho **kỹ thuật song song hoá** mà Gated DeltaNet (24) dùng nguyên xi, và cho cách nhìn "bộ nhớ liên kết sửa lỗi". Đọc tiếp: 24 (thêm gate), 26 (gate theo kênh).

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch. Mục "Ứng dụng cho dự án" là suy luận của chúng tôi. Các dẫn xuất trong mục 3 do chúng tôi làm lại và kiểm tra bằng số (numpy).

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

Linear attention (bài 18) viết attention thành RNN với state ma trận $S_t=S_{t-1}+v_tk_t^\top$, $o_t=S_tq_t$. Hai chuyện xảy ra:

1. **Chất lượng:** chỉ cộng dồn nên khi số cặp key–value vượt chiều key $d$, các key chồng lấn ("key collision", Schlag et al. 2021). Mô hình không thể "dọn chỗ" cho thông tin mới. Kết quả: kém ở các tác vụ cần *truy hồi trong ngữ cảnh* (in-context retrieval), ví dụ RAG.
2. **Delta rule** (Widrow-Hoff, 1960; áp dụng vào Transformer bởi Schlag et al. 2021): trước khi ghi, đọc giá trị cũ gắn với key hiện tại rồi sửa. DeltaNet cho kết quả tốt trên tác vụ tổng hợp và dịch nhỏ. Nhưng **thuật toán huấn luyện gốc tuần tự theo thời gian**: không tận dụng GPU, nên chưa ai huấn luyện được DeltaNet ở quy mô LLM.

**Cần giải quyết:** tạo thuật toán huấn luyện song song theo chiều chuỗi, giàu phép nhân ma trận (tensor core), không phải dựng state $S_t\in\mathbb R^{d\times d}$ cho từng $t$ (tốn I/O).

**Ghi chú nền: chunkwise form (dạng theo khối) cho linear attention.** Có ba cách tính cùng một hàm:
- *Hồi quy:* $O(Ld^2)$ FLOPs, nhưng $L$ bước tuần tự, toàn phép nhân từng phần tử, không dùng tensor core.
- *Song song:* $O=(QK^\top\odot M)V$, $O(L^2d)$, một bước, rất hợp GPU nhưng đắt khi $L$ lớn.
- *Chunkwise:* chia chuỗi thành $L/C$ khối cỡ $C$ (64 hay 128). Trong khối dùng dạng song song; giữa khối truyền state. Độ phức tạp $O(LCd+Ld^2)$, $L/C$ bước tuần tự. $C=L$ là dạng song song, $C=1$ là dạng hồi quy. Phương trình (Eq. 1-2 của paper):

$$S_{[t+1]}=S_{[t]}+V_{[t]}^\top K_{[t]},\qquad O_{[t]}=Q_{[t]}S_{[t]}^\top+\big(Q_{[t]}K_{[t]}^\top\odot M_C\big)V_{[t]}.$$

---

## 2. Ý tưởng chính

DeltaNet: $S_t=S_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$. Mỗi bước nhân state bên phải với một ma trận "đơn vị trừ hạng-1" (Householder tổng quát). Khai triển $L$ bước cho một **tích của $L$ ma trận như vậy**, nhìn qua thì khó song song hoá và khó chạy chunkwise vì state cuối khối phụ thuộc cả tích.

Chìa khoá: tích các Householder có **biểu diễn WY** gọn: $\prod(I-\beta_ik_ik_i^\top)=I-\sum_iw_ik_i^\top$ với các vector $w_i$ cùng kích thước $d$. Nhờ đó (i) không cần lưu ma trận $d\times d$ cho mỗi bước, chỉ lưu vector; (ii) các $w_i$ (và "pseudo-value" $u_i$) tính được cho cả khối bằng **một phép nghịch đảo ma trận tam giác nhỏ** (UT transform), chuyển vòng lặp tuần tự thành phép nhân ma trận. Rồi lắp vào dạng chunkwise của linear attention, chỉ thay $V$ bằng $U-WS^\top$. Kết quả: huấn luyện DeltaNet 1,3B/100B token (và 3B/1T token) lần đầu khả thi, vượt Mamba và GLA cùng quy mô.

---

## 3. Phương pháp chi tiết

**Ký hiệu:** $S_t\in\mathbb R^{d_v\times d_k}$ (paper vẽ $d\times d$ cho một head, rồi nói head dim $d_{head}<d$); $o_t=S_tq_t$. Chỉ số $[t]$ = khối thứ $t$; $x^r_{[t]}$ = phần tử thứ $r$ trong khối; $S^0_{[t]}=S_{[t]}$ = state đầu khối.

### 3.1 DeltaNet là bộ nhớ liên kết sửa lỗi

State $S$ là bảng tra "key → value" (nhớ liên kết, *associative memory*). **Đọc:** $\hat v=Sk$. **Luật delta (Widrow-Hoff / LMS):** nếu dự đoán $Sk_t$ khác đích $v_t$, dịch $S$ một bước để giảm sai lệch:

$$\mathcal L_t(S)=\tfrac12\|Sk_t-v_t\|^2,\quad S_t=S_{t-1}-\beta_t\nabla\mathcal L_t=S_{t-1}-\beta_t(S_{t-1}k_t-v_t)k_t^\top.$$

Viết gọn: $S_t=S_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$ (Eq. DeltaNet). Cách đọc khác: $v^{\rm old}=S_{t-1}k_t$, $v^{\rm new}=\beta_tv_t+(1-\beta_t)v^{\rm old}$, $S_t=S_{t-1}-v^{\rm old}k_t^\top+v^{\rm new}k_t^\top$ ("xoá cũ, ghi mới"). $\beta_t=\sigma(W_\beta x_t)\in(0,1)$ là *cường độ ghi*: $\beta=1$ thay hẳn, $\beta=0$ bỏ qua.

Linear attention thuần thì tối ưu hàm mất mát *tuyến tính* $-\langle Sk,v\rangle$: gradient là hằng $vk^\top$, không phụ thuộc state hiện tại, nên chỉ "cộng thêm". Delta rule tối ưu hàm mất mát *bậc hai*: gradient tỉ lệ với sai lệch, nên sửa mạnh khi sai nhiều và không sửa khi đã đúng (chú thích 3 paper). Ví dụ số ($d=2$, key đơn vị trực giao) xem bài 24 mục 3.2: cộng chồng cho $(11,12)=v_1+v_3$, delta rule cho đúng $v_3$.

**Ổn định.** $I-\beta kk^\top$ có trị riêng $1$ (bội $d-1$) và $1-\beta\|k\|^2$. Cần $\|k\|_2=1$ để trị riêng thuộc $(0,1]$. Paper dùng **L2-norm** thay cho L1-norm của Schlag et al., vì với $\beta=1$ thì $I-kk^\top$ là phép chiếu: xoá đúng một chiều, giữ $d-1$ chiều còn lại, "quên có mục tiêu".

### 3.2 Tái tham số hoá tiết kiệm bộ nhớ: $S_t=\sum_iu_ik_i^\top$

**Mệnh đề.** $S_t=\sum_{i=1}^tu_ik_i^\top$ với
$$u_t=\beta_t\Big(v_t-\sum_{i<t}u_i\,(k_i^\top k_t)\Big).$$

*Chứng minh (quy nạp).* $t=1$: $S_1=\beta_1v_1k_1^\top$, $u_1=\beta_1v_1$. Bước quy nạp: $S_t=S_{t-1}-\beta_t(S_{t-1}k_t)k_t^\top+\beta_tv_tk_t^\top$ và $S_{t-1}k_t=\sum_{i<t}u_i(k_i^\top k_t)$, nên $S_t=\sum_{i<t}u_ik_i^\top+u_tk_t^\top$ với $u_t$ như trên. $\square$

Ý nghĩa: DeltaNet chính là linear attention với **value giả (pseudo-value)** $u_t$ thay cho $v_t$. $u_t=\beta_t(v_t-S_{t-1}k_t)$ chính là *vector sửa lỗi*: "giá trị cần ghi trừ giá trị state đang dự đoán". Tính $u_t$ không cần dựng $S_{t-1}$ ($O(d)$ bộ nhớ), nhưng vẫn cần $u_i$ trước đó, tốn $O(L^2d)$ và vẫn tuần tự. Phần tiếp theo cho phép làm theo khối.

### 3.3 Dạng chunkwise: WY + UT

**Khai triển** (Eq. 4): $S_t=\sum_{i=1}^t\beta_i(v_ik_i^\top)\prod_{j=i+1}^t(I-\beta_jk_jk_j^\top)$. Với khối $[t]$ đặt $P^r_{[t]}=\prod_{i=1}^r(I-\beta^i_{[t]}k^i_{[t]}k^{i\top}_{[t]})$ ("hệ số tắt dần" áp lên state đầu khối) và $H^r_{[t]}$ = phần đóng góp của các token trong khối. Khi đó (Eq. 5):

$$S^r_{[t]}=S_{[t]}P^r_{[t]}+H^r_{[t]}.$$

**WY representation** (Bischof & Van Loan 1985) (Eq. 6-7):
$$P^r=I-\sum_{i=1}^rw^ik^{i\top},\qquad w^r=\beta^r\Big(k^r-\sum_{i<r}w^i\,(k^{i\top}k^r)\Big),$$
$$H^r=\sum_{i=1}^ru^ik^{i\top},\qquad u^r=\beta^r\Big(v^r-\sum_{i<r}u^i\,(k^{i\top}k^r)\Big).$$

*Chứng minh $P^r$:* $P^r=P^{r-1}(I-\beta_rk_rk_r^\top)=P^{r-1}-\beta_r(P^{r-1}k_r)k_r^\top$, với $P^{r-1}k_r=k_r-\sum_{i<r}w^i(k^{i\top}k_r)$, suy ra $w^r$ như trên. $\square$ ($H^r$ giống §3.2, nhưng chỉ trong khối.)

**Chuyển sang khối** (thế $P,H$ vào, Eq. 8-9):
$$S^r=S_{[t]}+\sum_{i\le r}\big(u^i-S_{[t]}w^i\big)k^{i\top}.$$
Số hạng $u^i-S_{[t]}w^i$ chính là vector sửa lỗi *thật* $\beta_i(v_i-S_{i-1}k_i)$: $u^i$ là bản tính với state đầu khối bằng 0, còn $-S_{[t]}w^i$ bù cho những gì state cũ đã "biết". Dạng ma trận:

$$\boxed{S_{[t+1]}=S_{[t]}+\big(U_{[t]}-W_{[t]}S_{[t]}^\top\big)^\top K_{[t]}}$$
$$\boxed{O_{[t]}=Q_{[t]}S_{[t]}^\top+\big(Q_{[t]}K_{[t]}^\top\odot M\big)\big(U_{[t]}-W_{[t]}S_{[t]}^\top\big)}$$

(So với linear attention thuần: chỉ đổi $V\to U-WS^\top$.) Dòng thứ hai: đầu ra token $r$ = đọc state đầu khối + (attention nhân quả trong khối) trên *vector sửa lỗi*.

**UT transform** (Joffrain et al. 2006) để tính $W,U$ không dùng vòng lặp theo $r$. Viết hệ $w^r=\beta^r k^r-\beta^r\sum_{i<r}w^i(k^{i\top}k^r)$ ở dạng ma trận: $W=\mathrm{diag}(\beta)K-\mathrm{diag}(\beta)\,\mathrm{strictLower}(KK^\top)\,W$, tức $\big(I+\mathrm{diag}(\beta)\mathrm{strictLower}(KK^\top)\big)W=\mathrm{diag}(\beta)K$. Vậy (Eq. 10-11):

$$\boxed{T=\Big[I+\mathrm{strictLower}\big(\mathrm{diag}(\beta)KK^\top\big)\Big]^{-1}\mathrm{diag}(\beta),\qquad W=TK,\quad U=TV.}$$

Ma trận cần nghịch đảo là tam giác dưới $C\times C$ với đường chéo bằng 1, giải bằng thế tiến (forward substitution), rẻ và không tuần tự theo $L$ (các khối độc lập). Phần còn lại là toàn phép nhân ma trận.

**Pseudo-code (chúng tôi viết lại; đã kiểm tra khớp recurrence):**

```python
def deltanet_chunk(Q, K, V, beta, S, C=64):       # Q,K (L,d) L2-norm; V (L,dv); S (dv,d)
    outs = []
    for s in range(0, Q.shape[0], C):
        q, k, v, b = Q[s:s+C], K[s:s+C], V[s:s+C], beta[s:s+C]
        I = torch.eye(len(q))
        T = torch.linalg.solve_triangular(I + torch.tril(b[:,None]*(k@k.T), -1),
                                          torch.diag(b), upper=False)
        W, U = T @ k, T @ v
        corr = U - W @ S.T                         # vector sửa lỗi thật của cả khối
        M = torch.tril(torch.ones(len(q), len(q)))
        outs.append(q @ S.T + ((q@k.T) * M) @ corr)
        S = S + corr.T @ k
    return torch.cat(outs), S
```

Thực tế: kernel Triton, state được tính lại trong backward (để tiết kiệm bộ nhớ), không dựng $S$ theo từng bước.

**Chi phí.** Dạng chunkwise: $O(LCd+Ld^2)$ FLOPs và $L/C$ bước tuần tự (chưa dùng chunk-level parallel scan). Nhân thêm: giải tam giác $C\times C$ mỗi khối (chúng tôi suy ra khoảng $O(C^2d+C^3)$ cho mỗi khối, paper không nêu). **Dạng hoàn toàn song song** cũng viết được: $A_{ij}=k_j^\top P^{j+1}_i q_i$, $A=(QK^\top\odot M)T$ (Eq. 3,11), nhưng cần nghịch đảo $L\times L$ (lập phương theo $L$), nên paper không dùng cho huấn luyện, chỉ nói có thể dùng để phân tích/diễn giải "attention map" của RNN.

Thử nghiệm tốc độ (Hình 1): kernel chunkwise vs kernel hồi quy (đã nhanh gấp 2 lần CUDA kernel gốc của Schlag et al.), $d=2048$ với batch×dài = 16384, head dim 64/128/256. Tăng tốc lớn dần theo $L$ và head dim; trục đồ thị tới 30× (chỉ đọc được từ hình, không có bảng số).

### 3.4 Khối DeltaNet Transformer

Theo Llama ("Transformer++", bài 05): thay self-attention bằng lớp DeltaNet, giữ SwiGLU, RMSNorm; thêm norm trước phép chiếu đầu ra. Tham số: $4d^2$ cho layer DeltaNet và $8d^2$ cho FFN SwiGLU (phần $\beta$ nhỏ không đáng kể). $q,k$: $\frac{\mathrm{SiLU}(Wx)}{\|\mathrm{SiLU}(Wx)\|_2}$; $\beta=\sigma(\cdot)$; có **short convolution** (depthwise, kernel nhỏ, tăng khả năng dịch cục bộ) sau các phép chiếu $q,k,v$. Thay "ELU+1" của bản gốc bằng SiLU, thay L1-norm bằng L2-norm: cả hai đều có ablation (mục 5).

### 3.5 Mô hình lai

Linear attention thiếu thông tin vị trí và khó so khớp cục bộ chính xác. Paper thử hai kiểu lai: (1) **xen SWA** (sliding-window attention) mỗi layer thứ hai (theo Griffin/Samba); (2) **hai layer global attention** (layer thứ 2 và layer $N/2+1$, theo Fu et al.). Kết quả ở mục 4.

---

## 4. Thí nghiệm và kết quả chính

**Tác vụ tổng hợp.** *MQAR* (multi-query associative recall): DeltaNet đạt gần hoàn hảo ở cấu hình khó nhất (không cần conv) và hơn Mamba ở chiều model thấp (Hình 4, chỉ có đồ thị). *MAD benchmark* (Hình 3 paper):

| Mô hình | Compress | Fuzzy recall | In-context recall | Memorize | Noisy recall | Selective copy | Trung bình |
|---|---|---|---|---|---|---|---|
| Transformer | 51,6 | 29,8 | 94,1 | 85,2 | 86,8 | 99,6 | 74,5 |
| Mamba | 52,7 | 6,7 | 90,4 | 89,5 | 90,1 | 86,3 | 69,3 |
| GLA | 38,8 | 6,9 | 80,8 | 63,3 | 81,6 | 88,6 | 60,0 |
| **DeltaNet** | 42,2 | **35,7** | **100** | 52,8 | **100** | **100** | 71,8 |

DeltaNet nổi trội ở các tác vụ recall (kể cả Fuzzy recall vượt cả Transformer) nhưng yếu ở *Memorize* (52,8). Kết quả khác (RegBench, in-context language learning) nằm ở phụ lục, text trích không có số.

**Mô hình ngôn ngữ (Bảng 1).** Mọi mô hình huấn luyện trên cùng tập con SlimPajama, tokenizer Mistral, đánh giá bằng lm-evaluation-harness. 340M/15B token và 1,3B/100B token. Cột nhiễu do bảng hỏng; dưới đây chỉ các giá trị đọc chắc chắn (Wiki ppl, LAMBADA ppl, và Avg tính được lại khớp):

| Quy mô | Mô hình | Wiki ppl ↓ | LMB ppl ↓ | Avg acc ↑ |
|---|---|---|---|---|
| 340M | Transformer++ | 28,39 | 42,69 | 41,2 |
| 340M | Mamba (conv) | 28,39 | 39,66 | 41,8 |
| 340M | GLA (không conv) | 28,65 | 43,35 | 41,5 |
| 340M | DeltaNet (conv) | 28,24 | 37,37 | 42,1 |
| 340M | + Sliding Attn | 27,06 | 38,17 | 42,1 |
| 340M | + Global Attn (2 layer) | 27,51 | 35,04 | 42,1 |
| 1,3B | Transformer++ | 16,85 | 13,44 | 50,9 |
| 1,3B | Mamba (conv) | 17,06 | 13,89 | 50,0 |
| 1,3B | GLA (không conv) | 17,22 | 14,47 | 51,0 |
| 1,3B | DeltaNet (conv) | 16,87 | 12,21 | 51,6 |
| 1,3B | + Sliding Attn | 16,56 | 11,74 | 52,1 |
| 1,3B | + Global Attn (2 layer) | 16,55 | 12,40 | 51,8 |

Kết luận paper: DeltaNet vượt Mamba/GLA về perplexity và tác vụ downstream; hybrid (SWA hoặc 2 layer global) vượt cả Transformer++. Về tác vụ cần nhiều recall (SWDE, SQuAD, FDA): ở 340M với cùng kích thước state, DeltaNet > GLA (delta rule hiệu quả); ở 1,3B DeltaNet *kém* GLA vì **state scale kém** (head dim bị giới hạn, xem §6); hybrid cải thiện rõ ở các tác vụ này (số liệu chi tiết bị hỏng định dạng trong bản text nên không chép).

**Mô hình 3B/1T token (Hình 5, cùng thiết lập Shen et al.):**

| Mô hình | Avg | MMLU |
|---|---|---|
| Llama-3.2-3B | 62,8 | 54,1 |
| PowerLM-3B | 62,3 | 45,0 |
| **DeltaNet-3B** | **59,8** | 40,7 |
| RecurrentGemma-2B | 57,9 | 31,8 |
| RWKV-6-3B | 54,9 | 28,4 |
| Mamba-2.7B | 53,3 | 26,1 |

DeltaNet 3B hơi kém Transformer cùng thiết lập (PowerLM-3B) nhưng hơn các RNN 2-3B; lưu ý số token huấn luyện của các baseline khác nhau.

**Throughput huấn luyện (Hình 6, 1,3B, H100):** gần GLA, nhanh hơn Mamba đáng kể; mọi mô hình tuyến tính vượt Transformer khi chuỗi huấn luyện dài hơn (số liệu chỉ ở đồ thị).

---

## 5. Phân tích / ablation

- **Chuẩn hoá và feature map (340M, Bảng 1 dưới).** Đổi L1-norm → L2-norm cải thiện lớn: ví dụ "L1-norm + 1+ELU" Wiki 31,12 / LMB 55,96 vs "L2-norm + 1+ELU" 28,03 / 37,62. Trong L2-norm, SiLU (mặc định) tốt nhất (28,24 / 37,37 so với ReLU 28,75 / 43,53).
- **Short convolution:** DeltaNet có conv: Wiki 28,24, LMB 37,37, Avg 42,1; không conv: 29,08, 50,87, 41,3. Conv giúp rõ (đặc biệt LAMBADA).
- **Hybrid:** SWA xen kẽ giảm Wiki ppl (28,24 → 27,06 ở 340M; 16,87 → 16,56 ở 1,3B); 2 layer global attention mạnh ở tác vụ truy hồi hơn (paper nói bằng lời).
- **Chunk vs recurrent:** kernel chunkwise càng ưu thế khi $L$ và head dim càng lớn; là nơi sequence-parallelism và tensor core phát huy.

---

## 6. Giới hạn

**Tác giả nêu:**
- Tốc độ huấn luyện vẫn kém GLA vì phải "marginalize" trên head dim trong kernel (phụ thuộc state-to-state), khó tile với head dim lớn. Hệ quả: head dim bị chặn nhỏ → state nhỏ → kém ở tác vụ cần recall (ở 1,3B kém GLA). Gợi ý: ma trận Householder khối chéo với khối vừa SRAM (≈128) trong khi head dim tổng thể lớn.
- **Khả năng tổng quát hoá độ dài hạn chế** so với GLA/RetNet/Mamba (không ngoại suy tốt vượt độ dài huấn luyện); suy đoán do **thiếu hệ số decay tường minh**, có thể khắc phục bằng gate. Đây đúng là động cơ của Gated DeltaNet (24).
- Delta rule có giới hạn biểu diễn lý thuyết (Irie et al. 2023); các biến thể mạnh hơn (hồi quy phi tuyến, Mesa layer, TTT, Titans) hy sinh tính song song hoặc cần xử lý khối.
- Mô hình khung "unifying" của paper (Bảng 2) chưa chắc là cách nhìn "đúng"; SSD (22) là ứng viên, nhưng có thể không bao hết lớp mô hình có ma trận chuyển không có cấu trúc.

**Chúng tôi nhận xét:**
- Không có ablation tách riêng tác dụng của delta rule khỏi short conv/L2-norm ở quy mô lớn (chỉ ở 340M).
- Số liệu recall chi tiết không trích được, nên khó đánh giá định lượng "DeltaNet kém GLA ở 1,3B".
- Không có thử nghiệm tác vụ có cấu trúc 2D (ảnh/OCR).

---

## 7. Liên hệ với các bài khác

- **18 Linear Transformers:** DeltaNet = linear attention + sửa lỗi (loss bậc hai thay loss tuyến tính).
- **19 RetNet, 20 GLA, 21 Mamba, 22 Mamba-2:** họ "decay" (cập nhật phần tử, ma trận chuyển chéo). DeltaNet dùng ma trận chuyển *không chéo* (đơn vị trừ hạng-1), giàu biểu diễn hơn mà vẫn song song nhờ WY. Bảng 2 paper xếp tất cả vào dạng $S_t=S_{t-1}\bullet M_t+v_tk_t^\top$; 22 cho thấy các mô hình vô hướng là trường hợp của SSD (ma trận semiseparable), còn DeltaNet thì không nằm gọn trong đó.
- **24 Gated DeltaNet:** thêm $\alpha_t$ vào đúng khuôn mẫu này; thuật toán chunkwise của 24 là bản mở rộng của §3.3.
- **26 Kimi Linear:** KDA có chung cấu trúc "Householder × gate" với state chuyển vị, đổi gate vô hướng sang theo kênh.
- **25 Jamba:** hybrid Mamba+attention; DeltaNet cho thấy hybrid dùng được với layer delta-rule.

---

## 8. Ứng dụng cho dự án (suy luận)

DeltaNet không có mặt trực tiếp trong decoder của bạn (là GatedDeltaNet), nhưng **mọi thứ về thuật toán chunkwise và tính chất toán của $S_t$ đều đúng cho layer GDN**, chỉ cần thêm hệ số $\alpha$.

1. **Cách hiểu layer GDN trong mô hình 0,8B.** Mỗi head giữ ma trận $S\in\mathbb R^{d_v\times d_k}$ là bảng tra key→value được sửa online bằng LMS. Khi bạn cắt $d_k$ (cột), dung lượng "bảng tra" giảm. Khi bạn cắt $d_v$ (hàng), bạn bỏ bớt các bộ hồi quy độc lập. Hai trường hợp ảnh hưởng khác nhau (chi tiết ở 24, mục 8).
2. **Điều kiện ổn định ràng buộc khi cắt tỉa.** Trị riêng $1-\beta\|k\|^2\in[0,1)$ cần $\|k\|=1$. Mọi thay đổi (cắt chiều $d_k$, lượng tử hoá $k$, thay norm) làm lệch $\|k\|$ có thể phá ổn định: kiểm tra L2-norm ở sau mọi thay đổi trên đường $k$.
3. **Hai kernel cho hai pha.** Prefill (chuỗi prefix ảnh dài) dùng chunkwise (dạng §3.3); decode token-by-token dùng recurrent. Đo riêng. Ở batch 1 với $L=1$, kernel chunkwise không có lợi.
4. **Thông tin về pseudo-value.** $u_i-S_0w_i=\beta_i(v_i-S_{i-1}k_i)$ là "độ bất ngờ" của token $i$ đối với bộ nhớ. Đây là đại lượng có thể dùng làm *chỉ báo độ quan trọng của token/layer* khi cắt tỉa hoặc phân tích (đề xuất của chúng tôi): layer mà độ lớn $\|u_i-S_0w_i\|$ luôn nhỏ hầu như không ghi gì mới, ứng viên cắt.
5. **Ít liên quan:** các tác vụ tổng hợp, tính tới độ dài 8K; dự án chỉ ~1.500 token sinh.

---

## 9. Thí nghiệm gợi ý

1. **Đo "độ bất ngờ" theo layer.** Trên mô hình 0,8B, ghi $\overline{\|\beta_t(v_t-S_{t-1}k_t)\|}$ theo layer GDN và so với tác động của việc cắt layer đó lên loss distillation. Kỳ vọng (giả thuyết): tương quan nghịch.
2. **Kiểm tra ổn định khi đổi $d_k$.** Cắt $d_k$ tại $q/k$ trước và sau L2-norm; theo dõi phân bố $1-\beta\|k\|^2$ và độ lệch của đầu ra so với mô hình gốc.
3. **Chọn chunk size.** Với prefill ~ vài nghìn token, benchmark chunk $C\in\{32,64,128\}$ trên GPU đích để chọn $C$ (đánh đổi giữa giải tam giác $C^3$ và số bước tuần tự $L/C$).

---

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| Delta rule | $S_t=S_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$ (SGD trên $\tfrac12\|Sk-v\|^2$) |
| Trị riêng chuyển | $1$ (bội $d-1$), $1-\beta\|k\|^2$ |
| Pseudo-value | $S_t=\sum_iu_ik_i^\top$, $u_t=\beta_t(v_t-\sum_{i<t}u_ik_i^\top k_t)$ |
| WY | $\prod(I-\beta_ik_ik_i^\top)=I-\sum w_ik_i^\top$, $w_r=\beta_r(k_r-\sum_{i<r}w_ik_i^\top k_r)$ |
| UT | $T=[I+\mathrm{strictLower}(\mathrm{diag}(\beta)KK^\top)]^{-1}\mathrm{diag}(\beta)$, $W=TK$, $U=TV$ |
| State khối | $S_{[t+1]}=S+(U-WS^\top)^\top K$ |
| Đầu ra khối | $O=QS^\top+(QK^\top\odot M)(U-WS^\top)$ |
| Chi phí chunkwise | $O(LCd+Ld^2)$, $L/C$ bước tuần tự |
| Hybrid | xen SWA mỗi layer thứ hai / 2 layer global attention |

## 11. Kết luận ngắn

DeltaNet là linear attention có "bộ nhớ sửa lỗi", mạnh ở recall. Đóng góp của bài: dùng biểu diễn WY của tích Householder và biến đổi UT để huấn luyện song song theo chiều chuỗi với cùng khuôn chunkwise của linear attention (chỉ thay $V$ bằng $U-WS^\top$). Kết quả: DeltaNet lần đầu ở quy mô LLM, hơn Mamba/GLA; hybrid còn hơn Transformer++. Điểm yếu (thiếu decay → ngoại suy độ dài kém, state nhỏ) dẫn thẳng đến Gated DeltaNet.

Nguồn gốc: https://arxiv.org/abs/2406.06484
