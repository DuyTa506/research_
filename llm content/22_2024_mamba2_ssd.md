# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality (2024)

**Tác giả:** Tri Dao, Albert Gu.  
**Venue / arXiv:** arXiv:2405.21060 (31/5/2024); ICML 2024 (theo cách bài 24 trích dẫn: PMLR 235). Code: github.com/state-spaces/mamba.  
**Nguồn:** https://arxiv.org/abs/2405.21060  
**Chủ đề:** State space model (SSM), ma trận semiseparable, structured masked attention, "state space duality" (SSD), thuật toán SSD theo khối, kiến trúc Mamba-2, mô hình lai SSM + attention.  
**Vị trí trong lộ trình:** Đọc trước: 18 (Linear Transformers), 19 (RetNet), 21 (Mamba). Bài này **thống nhất** cách nhìn SSM và attention; cho thuật toán chunkwise (khối) mà GDN (24) kế thừa. Đọc tiếp: 23 (DeltaNet song song), 24 (Gated DeltaNet), 25 (Jamba).

> Bài rất dài (khoảng 28 nghìn từ, có nhiều phụ lục về hệ thống song song và chứng minh). Ghi chú này tập trung vào **lý thuyết cốt lõi, thuật toán SSD và kiến trúc Mamba-2** theo hướng dễ tiếp cận, bỏ qua phần chứng minh chi tiết, tensor/sequence parallelism, và phụ lục các thuật toán 1-SS. Đây là tài liệu học tập, không phải bản dịch; "Ứng dụng cho dự án" là suy luận của chúng tôi. Phép kiểm tra "dạng đối ngẫu = dạng hồi quy" ở mục 3.4 và pseudo-code SSD ở mục 3.5 do chúng tôi chạy bằng số (numpy) và khớp.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
   - 3.1 SSM là gì
   - 3.2 SSM = nhân với ma trận semiseparable
   - 3.3 Attention có mặt nạ có cấu trúc (SMA)
   - 3.4 Đối ngẫu (SSD): hai thuật toán cho cùng một hàm
   - 3.5 Thuật toán SSD theo khối
   - 3.6 Kiến trúc Mamba-2
   - 3.7 Chi phí
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

Hai dòng mô hình chuỗi phát triển gần như tách rời:

- **Transformer**: attention $O(T^2)$ khi huấn luyện, KV-cache tăng tuyến tính khi sinh. Nhưng được hỗ trợ cực tốt về lý thuyết và hệ thống (FlashAttention, tensor parallelism, MQA/GQA...).
- **SSM có cấu trúc** (S4, rồi Mamba): $O(T)$ khi huấn luyện, state cố định khi sinh; Mamba (bài 21) ngang hoặc hơn Transformer cỡ nhỏ-vừa. Nhưng khó hiểu, khó tối ưu: cần "selective scan" kernel riêng, **không dùng nhân ma trận (tensor core)**, state nhỏ ($N=16$) vì thuật toán scan chậm khi $N$ tăng.

**Mục tiêu của paper:** xây một cầu lý thuyết giữa hai họ, để (i) chuyển mọi tối ưu của attention sang SSM, (ii) có thuật toán SSM nhanh bằng nhân ma trận, (iii) thiết kế kiến trúc mới (Mamba-2) dễ song song hoá.

**Khái niệm nền cần nắm.**
- **Sequence transformation (biến đổi chuỗi):** một ánh xạ $X\in\mathbb R^{T\times P}\to Y\in\mathbb R^{T\times P}$ (chiều $T$ là thời gian). Self-attention và SSM đều là biến đổi chuỗi.
- **Matrix transformation form:** nhiều biến đổi chuỗi viết được dạng $Y=MX$ với $M\in\mathbb R^{T\times T}$ phụ thuộc tham số/dữ liệu. Attention: $M=\mathrm{softmax}(QK^\top)$ (nhân quả: tam giác dưới). Đọc "đưa $M$ ra" làm đối tượng nghiên cứu là mấu chốt của paper.
- **Ma trận có cấu trúc (structured matrix):** ma trận có cách lưu và nhân nhanh hơn $T^2$ (thưa, hạng thấp, Toeplitz...).
- **Đối ngẫu (duality):** cùng một hàm có hai thuật toán tính khác nhau (một "tuyến tính, hồi quy", một "bậc hai, kiểu attention").

---

## 2. Ý tưởng chính

1. Viết SSM (với tham số phụ thuộc thời gian) thành $Y=MX$. Ma trận $M$ có tên là **ma trận semiseparable**: mọi ma trận con nằm trên/dưới đường chéo đều có hạng $\le N$ ($N$ = chiều state). SSM = nhân với ma trận semiseparable.
2. Nếu hạn chế $A_t$ thành **vô hướng nhân đơn vị** ($A_t=a_tI$), $M=L\circ(CB^\top)$, chính là một "attention có mặt nạ": $C\leftrightarrow Q$, $B\leftrightarrow K$, $X\leftrightarrow V$, mặt nạ $L$ là ma trận 1-semiseparable (tích luỹ của $a_t$). Dạng bậc hai (nhân thẳng $M$) = **attention**; dạng tuyến tính (hồi quy) = **SSM**. Đó là **SSD**.
3. Thuật toán nhanh nhất không chọn một trong hai: **chia $M$ thành khối**, đường chéo khối tính bằng dạng attention (nhân ma trận nhỏ), phần ngoài đường chéo là hạng thấp, đi qua state và một scan ngắn trên số khối. Cho FLOPs tuyến tính theo $T$ và gần như toàn phép nhân ma trận: nhanh 2-8× kernel scan của Mamba, vượt FlashAttention-2 từ $T\ge2$K.
4. Mamba-2 = SSD + chỉnh khối (chiếu song song, thêm norm, head dim lớn kiểu Transformer).

---

## 3. Phương pháp chi tiết

### 3.1 SSM là gì

Một SSM rời rạc, tham số phụ thuộc thời gian (ký hiệu paper, state ẩn $h_t\in\mathbb R^{N}$ mỗi kênh):

$$h_t=A_th_{t-1}+B_tx_t,\qquad y_t=C_t^\top h_t\qquad(\text{Eq. 2})$$

$x_t\in\mathbb R$ (một kênh; mở rộng $P$ kênh bằng cách dùng chung $A,B,C$ cho cả $P$ kênh, gọi là một "head"), $A_t\in\mathbb R^{N\times N}$, $B_t,C_t\in\mathbb R^N$. $N$ là **chiều state** (còn gọi *state expansion factor* vì nó nhân bộ nhớ lên $N$ lần). "Structured" vì $A_t$ phải có cấu trúc (chéo...) để tính nhanh. Mamba gọi mô hình có $A,B,C$ phụ thuộc đầu vào là **selective SSM** (bài 21). Paper bỏ qua bước rời rạc hoá $(\Delta,\mathring A,\mathring B)\to(A,B)$ để gọn (Remark 1); theo Mamba thì $a_t=\exp(\Delta_tA)$ với $A<0$, giá trị nằm trong $(0,1)$ (đây là kiến thức từ bài 21, không viết lại trong text bài này).

### 3.2 SSM = nhân với ma trận semiseparable

Khai triển hồi quy (với $h_{-1}=0$):

$$h_t=\sum_{s\le t}A_{t}\cdots A_{s+1}\,B_sx_s,\qquad y_t=\sum_{s\le t}C_t^\top A_t\cdots A_{s+1}B_s\;x_s.$$

Vậy $y=Mx$ với (Eq. 3)

$$\boxed{M_{ji}=C_j^\top A_j\cdots A_{i+1}B_i\quad(j\ge i),\qquad M_{ji}=0\ (j<i).}$$

**Semiseparable.** Một ma trận tam giác dưới là *$N$-semiseparable* nếu mọi ma trận con nằm hoàn toàn trên hoặc dưới đường chéo có hạng $\le N$. Lý do $M$ thoả: lấy hàng $j\in[a,b)$, cột $i\in[c,d)$ với $d\le a$ (khối nằm hẳn dưới đường chéo). Với mọi $j\ge a\ge d>i$ ta có $A_j\cdots A_{i+1}=(A_j\cdots A_d)(A_{d-1}\cdots A_{i+1})$, nên

$$M_{ji}=\underbrace{\big(C_j^\top A_j\cdots A_d\big)}_{1\times N}\;\underbrace{\big(A_{d-1}\cdots A_{i+1}B_i\big)}_{N\times1},$$

tức khối con là tích của ma trận $(b-a)\times N$ và ma trận $N\times(d-c)$, hạng $\le N$ (Lemma 3.3). Ngược lại mọi ma trận $N$-semiseparable có biểu diễn như vậy (Proposition 3.4). Hiểu trực quan: **thông tin từ quá khứ xa tới hiện tại đi qua "cổ chai" state $N$ chiều**, nên khối ngoài đường chéo có hạng thấp. Khi chia lưới khối (mục 3.5), phân rã này có ba nhân tử: *trái* ($C$), *giữa* ($A$ qua ranh giới khối), *phải* ($B$).

**Trường hợp 1-semiseparable (state vô hướng $N=1$, bỏ $B,C$):**
$$L=\mathrm{1SS}(a)=\begin{pmatrix}1\\a_1&1\\a_2a_1&a_2&1\\\vdots&&&\ddots\end{pmatrix},\qquad L_{ji}=a_j\cdots a_{i+1}.$$
Nhân $y=Lx$ chính là phép đệ quy vô hướng $y_t=a_ty_{t-1}+x_t$ (cumprodsum, tổng tích luỹ có nhân): dùng nhiều lần làm viên gạch cơ bản.

**Hệ quả (Theorem 3.7):** mọi SSM với state $N$ tính được trong $O(TN)$ (không tính tiền xử lý) bằng dạng hồi quy; dạng ngây thơ (dựng $M$, rồi nhân) là bậc hai. Hai thuật toán nhân ma trận có cấu trúc: đây chính là "hai mode" của đối ngẫu.

### 3.3 Attention có mặt nạ có cấu trúc (SMA)

Attention nhân quả có kernel: $Y=(L\circ QK^\top)V$, $L$ là mặt nạ tam giác dưới toàn 1 (linear attention, bài 18, bỏ softmax). Paper chứng minh gọn bằng *tensor contraction*: $Y=\mathrm{contract}(TN,SN,SP,TS\to TP)(Q,K,V,L)$. Có hai thứ tự co tensor:

- *Bậc hai:* $G=QK^\top$ ($T\times T$), nhân mặt nạ $L$, rồi nhân $V$. Là attention thông thường: $O(T^2N)$.
- *Tuyến tính:* $Z=\mathrm{outer}(K,V)$ ($T\times N\times P$), $H=L\cdot Z$ (cumsum theo thời gian, $O(TNP)$), $Y=\mathrm{contract}(Q,H)$. Là dạng RNN của linear attention.

Quan sát then chốt: nút thắt của dạng tuyến tính là phép nhân với mặt nạ $L$; **chỉ cần $L$ là ma trận có cấu trúc (nhân nhanh)** thì dạng tuyến tính vẫn hiệu quả. Đó là **Structured Masked Attention (SMA)**: $Y=(L\circ QK^\top)V$ với $L$ có cấu trúc bất kỳ. Ví dụ: $L$ toàn 1 → linear attention; $L_{ij}=\gamma^{i-j}$ → **RetNet** (19); $L$ Toeplitz → kiểu mã hoá vị trí tương đối nhân; $L$ Fourier. Định lý 5.2: attention nhân quả với tự hồi quy bậc hữu hạn bắt buộc có $L$ semiseparable.

### 3.4 Đối ngẫu (SSD): hai thuật toán cho cùng một hàm

Chọn $A_t=a_tI$ (vô hướng nhân đơn vị). Khi đó $M_{ji}=a_j\cdots a_{i+1}\cdot(C_j^\top B_i)$, tức

$$\boxed{M=L\circ(CB^\top),\qquad L=\mathrm{1SS}(a)}\qquad\Longleftrightarrow\qquad Y=(L\circ QK^\top)V\ \text{ với }Q\!=\!C,\ K\!=\!B,\ V\!=\!X.$$

| Khung SSM | Khung attention |
|---|---|
| $C_t$ (chiếu ra) | $Q$ (query) |
| $B_t$ (chiếu vào) | $K$ (key) |
| $x_t$ (đầu vào kênh) | $V$ (value) |
| $a_t$ (decay vô hướng) | mặt nạ $L_{ji}=a_j\cdots a_{i+1}$ |
| $h_t$ (state $N\times P$) | $\sum_{s\le t}(\text{decay})K_sV_s^\top$ |

Ghi chú về ký hiệu: trong SSM, $h_t\in\mathbb R^{N\times P}$ khi có $P$ kênh: $h_t=a_th_{t-1}+B_tx_t^\top$. Đó **đúng là** $S_t=\alpha_tS_{t-1}+v_tk_t^\top$ của Mamba2 viết trong GDN (24), chỉ chuyển vị.

**Ví dụ kiểm tra ($T=3$).** $L=\begin{pmatrix}1&0&0\\a_1&1&0\\a_2a_1&a_2&1\end{pmatrix}$. Dạng hồi quy: $h_0=B_0x_0$, $h_1=a_1h_0+B_1x_1$, $h_2=a_2h_1+B_2x_2$; $y_2=C_2^\top h_2=a_2a_1(C_2^\top B_0)x_0+a_2(C_2^\top B_1)x_1+(C_2^\top B_2)x_2$, khớp hàng thứ ba của $L\circ(CB^\top)$. Chúng tôi đã kiểm tra bằng số ($T=10,N=4,P=3$, sai khác chỉ ở mức làm tròn): $(L\circ CB^\top)X$ bằng đầu ra của hồi quy.

Hai dạng ảnh hưởng chi phí: dạng hồi quy $O(TNP)$ (mỗi bước cập nhật state $N\times P$), không dùng tensor core; dạng attention $O(T^2N)$ nhưng toàn nhân ma trận.

### 3.5 Thuật toán SSD theo khối

Chia $T$ bước thành $T/Q$ khối, mỗi khối $Q$ bước (ví dụ $Q=64$). Chia $M$ thành lưới khối $Q\times Q$:

- **Khối đường chéo** $M^{(j,j)}$ = SSM trên chính khối đó với state đầu bằng 0 → tính bằng **dạng attention** (nhân ma trận $Q\times Q$, các khối độc lập, song song).
- **Khối ngoài đường chéo** $M^{(j,i)}$, $j>i$: hạng $\le N$, phân rã ba phần (trái $\times$ giữa $\times$ phải) như ở 3.2. Tương ứng bốn bước tính:

| Bước | Ý nghĩa | Phép tính |
|---|---|---|
| 1. Trong khối (khối chéo) | Đầu ra do đầu vào cùng khối, state đầu $=0$ | $Y_{\rm diag}=(L\circ C B^\top)X$ cục bộ |
| 2. State của khối (nhân phải, $B$) | State cuối khối nếu state đầu $=0$ | $\mathrm{states}_j=\sum_{s\in\text{khối }j}(\text{decay đến cuối khối})\,B_sx_s^\top$ |
| 3. Truyền state giữa khối (nhân giữa, $A$) | State thật đầu mỗi khối | scan 1-SS trên $T/Q$ khối: $H_j=a^{(j)}_{\rm chunk}H_{j-1}+\mathrm{states}_j$ |
| 4. State → đầu ra (nhân trái, $C$) | Đóng góp của quá khứ vào đầu ra khối | $Y_{\rm off}=\mathrm{(decay\ từ\ đầu\ khối)}\cdot C\,H_{j-1}$ |

Đầu ra: $Y=Y_{\rm diag}+Y_{\rm off}$. Diễn giải: bước 1 là "attention trong khối", bước 2-4 là "RNN giữa các khối" nhưng chỉ có $T/Q$ bước. Pseudo-code (viết lại, gần Listing 1 của paper, một head, $a_t$ dạng log):

```python
def ssd(X, loga, B, C, Q=64):                 # X (T,P); loga (T,); B,C (T,N)
    T = X.shape[0]; nc = T // Q
    X, B, C, la = [t.reshape(nc, Q, *t.shape[1:]) for t in (X, B, C, loga)]
    cum = la.cumsum(1)                                          # (nc,Q) log decay trong khối
    # 1) khối chéo: attention cục bộ với mặt nạ decay
    Lmat = (cum[:, :, None] - cum[:, None, :]).exp().tril()     # (nc,Q,Q)
    Y_diag = torch.einsum('cij,cin,cjn,cjp->cip', Lmat, C, B, X)
    # 2) state cuối mỗi khối (khởi từ 0)
    dec_end = (cum[:, -1:] - cum).exp()                         # (nc,Q)
    states = torch.einsum('cj,cjn,cjp->cnp', dec_end, B, X)     # (nc,N,P)
    # 3) scan giữa khối (ngắn: nc bước)
    H, h = [], torch.zeros_like(states[0])
    chunk_dec = cum[:, -1].exp()
    for c in range(nc):
        H.append(h)                                             # state đầu khối c
        h = chunk_dec[c] * h + states[c]
    H = torch.stack(H)
    # 4) state đầu khối -> đầu ra
    Y_off = torch.einsum('ci,cin,cnp->cip', cum.exp(), C, H)
    return (Y_diag + Y_off).reshape(T, -1)
```

Phiên bản gốc dùng `segsum` + `einsum` đầy đủ batch/head, chỉ khoảng 25 dòng PyTorch, "không cần kernel mức thấp" và vẫn tương đối hiệu quả. Paper cho biết bước 3 là scan độc lập trên $NP$ kênh, chiều dài $T/Q$, công việc $O(TNP/Q)$ rất nhỏ so với các bước khác.

### 3.6 Kiến trúc Mamba-2

Từ SSD, thiết kế khối (Hình 6):

- **SSD là hàm của $(A,X,B,C)$ cùng lúc.** Mamba-1 sinh $B,C,\Delta$ *từ* $X$ sau một phép chiếu (tuần tự). Mamba-2 sinh $A,X,B,C$ **song song** từ đầu vào bằng một phép chiếu lớn, giống $Q,K,V$ trong Transformer: bớt tham số, và hợp với **tensor parallelism** kiểu Megatron (giảm một nửa số điểm đồng bộ mỗi khối).
- **Thêm normalization** (như NormFormer) ngay trước phép chiếu đầu ra, giúp ổn định ở mô hình lớn (giống thêm GroupNorm/LayerNorm sau linear attention ở TransNormer/RetNet).
- **Head dimension $P=64$ hoặc $128$** (Mamba-1 chỉ $P=1$): làm cho các phép nhân ma trận đủ lớn để tận dụng tensor core. $N$ (state) thường 64-256, lớn hơn nhiều so với Mamba-1 ($N=16$) vì SSD không chậm đi khi tăng $N$.
- **Các mẫu "đa đầu"** tương đương attention (Eq. 17-20): *multi-head* (MHA), *multi-contract* (≈ MQA, bài 14: chia sẻ $B$ và $X$, riêng $C$ mỗi head), *multi-expand* (≈ multi-key), *multi-input* (≈ **multi-value attention, MVA**: chia sẻ $B,C$ qua các head, $X$ riêng mỗi head). Mamba-1 hoá ra là MVA; Mamba-2 dùng MVA, mở rộng thành **grouped-value** (GVA, tương tự GQA bài 15) để chia theo nhóm cho tensor parallel.
- **Hàm kernel/feature map $\psi$** áp lên nhánh $B,C$ (≈ $K,Q$): mặc định SiLU/Swish; thử các kernel của linear attention và Performer đều không tốt hơn (mục 5).

### 3.7 Chi phí

Theo Bảng so sánh của paper (với $T$ = độ dài, $N$ = chiều state/head dim):

| | Attention | SSM (scan) | SSD |
|---|---|---|---|
| Kích thước state khi sinh | $T$ | $N$ | $N$ |
| FLOPs huấn luyện | $T^2N$ | $TN^2$ | $TN^2$ |
| FLOPs suy luận (mỗi token) | $TN$ | $N^2$ | $N^2$ |
| Bộ nhớ (thuật toán ngây thơ) | $T^2$ | $TN^2$ | $TN$ |
| Dùng nhân ma trận (tensor core) | có | không | có |

(Định lý 6.1 giả sử $P=N=Q$: huấn luyện $O(TN^2)$ FLOPs, suy luận $O(N^2)$ bộ nhớ và $O(N^2)$ FLOPs mỗi token khi $P=N$.)

**Ví dụ số cỡ mô hình bạn (tự tính, giả định):** một layer SSD/GDN-like với $H=8$ head, $P=N=128$: state $=H\cdot N\cdot P=131{,}072$ phần tử/layer (512 KB fp32); FLOPs suy luận mỗi token $\approx 8\cdot 128^2\cdot(\text{vài lần})\approx$ vài trăm nghìn đến 1 MFLOP/layer, so với attention ở $T=3000$, $H_q=8$, $d_h=128$: $\approx4\cdot3000\cdot128\cdot8\approx12$ MFLOPs/layer. Tiết kiệm FLOPs/bộ nhớ có thật nhưng ở batch 1 với model nhỏ, thời gian bị chặn bởi đọc trọng số (xem bài 24, mục 3.6).

---

## 4. Thí nghiệm và kết quả chính

**Tốc độ (Hình 10, A100 80GB PCIe).** SSD nhanh **2-8×** kernel selective scan của Mamba khi $N=64$, cho phép state lớn hơn (≥ 8× Mamba) với chi phí tăng nhỏ; **nhanh hơn FlashAttention-2 từ độ dài 2K, nhanh hơn 6× ở 16K** (theo mục giới thiệu). Với $N$ tăng, scan của Mamba chậm tuyến tính, SSD hầu như phẳng. Lưu ý (paper): cả mô hình Mamba-2 có thể không nhanh bằng Transformer ở chuỗi ngắn (2K) vì Transformer $L$ layer gồm $L/2$ layer attention và $L/2$ MLP (MLP rất hiệu quả phần cứng).

**Khôi phục liên kết đa truy vấn (MQAR, Arora et al.; Hình 8).** Mamba-1 gặp khó, Mamba-2 làm tốt ở mọi cấu hình, tốt hơn rõ rệt ngay cả khi so cùng state size ($N=16$); tăng $N$ lên 64 và 256 cải thiện đều. (Chỉ đồ thị.) Paper "không chắc yếu tố nào chủ yếu".

**Mô hình ngôn ngữ (The Pile, GPT-3-like từ 125M đến 2,7B, huấn luyện theo Chinchilla; 300B token cho các mô hình lớn).** Scaling law (Hình 9): Mamba-2 Pareto-trội hơn Mamba và Transformer++ về perplexity theo FLOPs và theo thời gian thực (đồ thị). Downstream zero-shot (trích từ Bảng 1):

| Mô hình | Pile ppl ↓ | LAMBADA ppl ↓ | LAMBADA acc ↑ | HellaSwag acc ↑ | PIQA acc ↑ |
|---|---|---|---|---|---|
| Pythia-1B | 7,82 | 7,92 | 56,1 | 47,2 | 70,7 |
| Mamba-790M | 7,33 | 6,02 | 62,7 | 55,1 | 72,1 |
| **Mamba-2-780M** | 7,26 | 5,86 | 61,7 | 54,9 | 72,0 |
| Pythia-1.4B | 7,51 | 6,08 | 61,7 | 52,1 | 71,0 |
| Mamba-1.4B | 6,80 | 5,04 | 65,0 | 59,1 | 74,2 |
| **Mamba-2-1.3B** | 6,66 | 5,02 | 65,7 | 59,9 | 73,2 |
| Pythia-2.8B | 6,73 | 5,04 | 64,7 | 59,3 | 74,0 |
| Mamba-2.8B | 6,22 | 4,23 | 69,2 | 66,1 | 75,2 |
| **Mamba-2-2.7B** | 6,09 | 4,10 | 69,7 | 66,6 | 76,4 |

Paper: Mamba-2 hơn Mamba mỗi cỡ và thường khớp Pythia cỡ gấp đôi (Mamba-2 2,7B cũng vượt Pythia-6,9B theo mục giới thiệu). Các cột khác (ARC, Winogrande, OpenbookQA...) có trong Bảng 10, text bị cắt.

**Mô hình lai SSD + attention + MLP (mục 9.2.3).**

*Bảng 2: 350M, 48 layer, 7B token, số layer attention vs perplexity*

| Số layer attention | 0 (Mamba-2) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 9 | 11 | 15 | 24 | Transformer++ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Perplexity ↓ | 8,60 | 8,38 | 8,32 | 8,29 | 8,29 | 8,28 | **8,26** | 8,27 | 8,28 | 8,30 | 8,34 | 8,50 | 8,68 |

**Khoảng 10% layer là attention là tốt nhất**; chỉ 1 layer đã giảm từ 8,60 xuống 8,38. Paper chú thích: miễn là các layer attention được rải đều, không ở đầu hoặc cuối, vị trí chính xác không quan trọng nhiều (thử nghiệm quy mô nhỏ).

*Bảng 3: 2,7B (64 layer), 300B token, The Pile*

| Mô hình | Pile ppl | LAMBADA ppl | Avg acc |
|---|---|---|---|
| Transformer++ (32 attn + 32 MLP) | 6,13 | 3,99 | 60,2 |
| Mamba-2 (64 SSD) | 6,09 | 4,10 | 60,2 |
| Mamba-2-MLP (32 SSD + 32 MLP) | 6,13 | 4,18 | 59,6 |
| **Mamba-2-Attention (58 SSD + 6 attn)** | **5,95** | **3,85** | **61,0** |
| Mamba-2-MLP-Attention (28 SSD + 4 attn + 32 MLP) | 6,00 | 3,95 | 60,7 |

Kết luận paper: Transformer++ và Mamba-2 cùng chất lượng; chỉ cần 6 layer attention đã vượt cả hai. Thêm MLP làm giảm chất lượng nhẹ nhưng nhanh hơn và dễ nâng cấp lên MoE. Giả thuyết của tác giả: SSM là ánh xạ chuỗi tổng quát, attention làm cơ chế *truy hồi* tra token cũ thay vì ép toàn bộ ngữ cảnh vào state.

---

## 5. Phân tích / ablation

**Thiết kế khối (Bảng 4, 125M).** Mamba-1 tuần tự không norm: 129,3M tham số, ppl 11,76; thêm norm: 11,54; chiếu song song không norm: 126,5M, 11,66; **Mamba-2 (song song + norm): 126,5M, 11,49.** Cả hai thay đổi đều giúp nhẹ, chiếu song song bớt tham số và hợp tensor parallel.

**Mẫu đầu (Bảng 5, cùng state tổng $HPN$).**

| Mẫu head | Analog | 125M ppl | 360M ppl |
|---|---|---|---|
| Multi-input (MIS) | MVA | **11,66** | **8,73** |
| Multi-contract (MCS) | MQA | 12,62 | 9,33 |
| Multi-expand (MES) | multi-key | 12,59 | 9,36 |
| Multi-head (MHS) | MHA | 12,06 | 9,01 |

MVA tốt nhất với chênh lệch lớn "dù tổng state như nhau", điều mà paper nêu là đáng ngạc nhiên và không giải thích được bằng kích thước state.

**Kernel/feature map (Bảng 6).** Không dùng feature map: 11,58; Swish: 11,66; Exp: 11,62; ReLU: 11,73; cosFormer: 11,97; Random Feature Attention: 11,57; Performer: 12,21. Mở rộng feature kiểu Based/ReBased cũng không cải thiện (Bảng 7, ví dụ 130M: Swish 11,67, Based 12,19, LayerNorm 11,50). Kết luận: **các xấp xỉ kernel của linear attention không giúp**; paper gợi ý có thể bỏ hẳn hàm kích hoạt. Tác giả lưu ý SSD khác linear attention thường ở mặt nạ 1-SS, nên kết quả âm này không đáng ngạc nhiên.

---

## 6. Giới hạn

**Do tác giả nêu:**
- SSD giảm nhẹ biểu diễn so với selective SSM của Mamba-1 ($A_t$ từ chéo xuống vô hướng) để đổi lấy tốc độ.
- Mamba-2 toàn mô hình có thể chậm hơn Transformer ở chuỗi ngắn.
- SMA/SSD chỉ nói về một số "vị" attention (nhân quả kiểu linear); softmax thuần không nằm trong khung (chú thích 1: tiêu đề là "homage").
- Tác vụ MQAR: lý do Mamba-2 hơn Mamba-1 chưa rõ.

**Chúng tôi nhận xét:**
- Decay là **vô hướng theo head**: mọi kênh state nhớ cùng tốc độ. GLA (20) và KDA (26) cho thấy có lợi khi mở rộng theo kênh.
- Quy tắc cập nhật chỉ *cộng chồng* ($+B_tx_t^\top$): không có cơ chế sửa/xoá đích danh, nên recall chính xác yếu hơn delta rule (xem Bảng S-NIAH trong bài 24: Mamba2 tụt mạnh khi dài). Đây là động cơ của 23 và 24.
- Hầu hết thí nghiệm đến 2,7B/300B token trên The Pile; không có thử nghiệm ảnh/OCR/đa phương thức.

---

## 7. Liên hệ với các bài khác

- **18 Linear Transformers:** SSD là linear attention + mặt nạ decay 1-semiseparable phụ thuộc dữ liệu; paper còn đưa bằng chứng tensor contraction đơn giản cho dạng hồi quy của linear attention.
- **19 RetNet:** $L_{ij}=\gamma^{i-j}$ = SMA với mặt nạ decay hằng số; Mamba-2 là bản decay phụ thuộc dữ liệu. **20 GLA:** gate theo kênh (ma trận chuyển chéo đầy đủ) nằm ngoài lớp "vô hướng nhân đơn vị" của SSD nhưng vẫn chunkwise được.
- **21 Mamba:** Mamba-2 = Mamba-1 + (i) $A$ vô hướng, (ii) head dim lớn, (iii) chiếu song song, (iv) norm. Mamba-1 là MVA với $P=1$.
- **14 MQA / 15 GQA:** SSD cho phép định nghĩa tương tự cho SSM (multi-contract ≈ MQA, grouped-value ≈ GQA).
- **23 DeltaNet / 24 Gated DeltaNet:** dùng cùng khung chunkwise (trong-khối kiểu attention + truyền state), nhưng ma trận chuyển $I-\beta kk^\top$ không phải vô hướng nên *không* nằm trong SSD; đã có bình luận trong bài 23 về việc SSD "có thể không bao quát". Công thức của Mamba2 trong GDN: $S_t=\alpha_tS_{t-1}+v_tk_t^\top$, giống hệt (3.4).
- **25 Jamba / 26 Kimi Linear:** hybrid SSM/linear + attention; mục 9.2.3 ở đây là bằng chứng sớm (tỉ lệ ~10% attention).
- **44 FlashAttention:** so sánh tốc độ; FlashAttention-2 là baseline mạnh nhất, SSD vượt từ 2K.

---

## 8. Ứng dụng cho dự án (suy luận)

Decoder của bạn dùng GatedDeltaNet, không phải Mamba-2, nên bài này **gián tiếp**:

1. **Hiểu "state" của GDN từ hai phía.** Mamba-2 cho thấy một layer tuyến tính vừa là RNN (đọc/ghi state $N\times P$) vừa là attention có mặt nạ (nhân $T\times T$). Với GDN thì dạng "attention" tương ứng là $(QK^\top\odot\Gamma\odot M)\tilde U$ (xem 24), tức mọi layer GDN có thể xem là một attention nhân quả với "ma trận chú ý" có hạng cấu trúc. Điều này cho phép đo **$D_{lin}$** của bạn theo một cách trực tiếp: lấy "ma trận chú ý ẩn" của layer GDN (hoặc dùng dạng đối ngẫu) để so sánh với attention softmax khi phân tích layer nào bị cắt.
2. **Quy tắc đặt tỉ lệ lai.** Bảng 2 và 3 (≈ 10% attention, chỉ 6/64 layer là đủ vượt Transformer++) là bằng chứng độc lập rằng tỉ lệ attention nhỏ đã đủ. Dự án của bạn có (giả định) 6/24 = 25% attention; với mô hình đã cắt còn 0,3B/0,1B, bạn có thể thử hạ số layer attention và đo (mục 9). Kết luận trong paper này không đủ để chứng minh điều đó với OCR, chỉ là gợi ý.
3. **Head dim và state.** Paper nhấn mạnh $P,N\in\{64,128\}$ để dùng tensor core; khi cắt head dim (ví dụ từ 128 xuống 64), bạn có thể mất hiệu quả kernel dù giảm FLOPs. Kiểm tra benchmark thực tế trước khi coi đó là lợi.
4. **Đối ngẫu giúp debug/phân tích.** Có thể kiểm tra cài đặt layer (recurrent vs chunk) bằng cách so với dạng song song $(L\circ CB^\top)X$ trên chuỗi ngắn, như ở mục 3.4 (một test đơn vị đơn giản).
5. **Ít liên quan:** phần tensor/sequence parallelism, huấn luyện độ dài thay đổi (cho huấn luyện nhiều GPU); không áp dụng cho mục tiêu suy luận batch-1 trên GPU nhỏ.

---

## 9. Thí nghiệm gợi ý

1. **Tỉ lệ attention sau cắt tỉa.** Trên mô hình 0,3B, thay lần lượt 1, 2, 3, 6 layer full attention bằng GDN (hoặc ngược lại) rồi phục hồi bằng distill cùng ngân sách; đo CER/WER OCR và perplexity. Đối chiếu đường cong kiểu Bảng 2 (cực tiểu quanh 10-25%?).
2. **Ma trận chú ý ẩn của GDN.** Dựng $A=(QK^\top\odot\Gamma\odot M)\tilde T$ cho một trang OCR và so khoảng cách (cosine) với bản đồ attention của layer softmax; xem layer GDN nào "giống attention" nhất.
3. **Test đối ngẫu.** Viết unit test so recurrence vs chunk vs dạng song song cho layer GDN đã cắt (để phát hiện lỗi sau khi sửa $d_k$/$d_v$).

---

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| SSM chọn lọc | $h_t=A_th_{t-1}+B_tx_t,\ y_t=C_t^\top h_t$ |
| Dạng ma trận | $y=Mx$, $M_{ji}=C_j^\top A_j\cdots A_{i+1}B_i$ |
| Semiseparable bậc $N$ | mọi ma trận con trên/dưới chéo có hạng $\le N$ |
| 1-SS | $L_{ji}=a_j\cdots a_{i+1}$, nhân = đệ quy vô hướng $y_t=a_ty_{t-1}+x_t$ |
| SMA | $Y=(L\circ QK^\top)V$, $L$ có cấu trúc |
| **SSD** | $A_t=a_tI$: $M=L\circ(CB^\top)$, $L=\mathrm{1SS}(a)$; $C\!=\!Q,B\!=\!K,X\!=\!V$ |
| Thuật toán SSD | khối chéo = attention; ngoài chéo = hạng thấp qua state + scan $T/Q$ bước |
| Chi phí SSD | $O(TN^2)$ huấn luyện, $O(N^2)$ mỗi token suy luận, state $N$ |
| Khối Mamba-2 | $A,X,B,C$ chiếu song song; norm trước phép chiếu ra; $P\in\{64,128\}$; MVA/GVA |
| Hybrid | ≈ 10% layer attention tốt nhất (Bảng 2); 58 SSD + 6 attn tốt hơn cả hai (Bảng 3) |

## 11. Kết luận ngắn

Bài này cho thấy attention tuyến tính có mặt nạ và SSM có $A_t$ vô hướng là **cùng một hàm** viết bằng hai cách (đối ngẫu), và rằng SSM là phép nhân ma trận semiseparable. Từ đó có thuật toán SSD theo khối vừa tuyến tính vừa giàu nhân ma trận (2-8× kernel Mamba, qua FlashAttention-2 từ 2K), và kiến trúc Mamba-2. Với bạn: đây là nền lý thuyết để hiểu vì sao layer GDN chunkwise hoạt động và vì sao một phần nhỏ layer attention trộn vào là đủ; nhưng GDN (có $I-\beta kk^\top$) vượt ra khỏi khung SSD.

Nguồn gốc: https://arxiv.org/abs/2405.21060
