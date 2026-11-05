# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022)

**Tác giả:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré.  
**Venue / arXiv:** arXiv:2205.14135 (v2, 23/06/2022).  
**Nguồn:** https://arxiv.org/abs/2205.14135  
**Chủ đề:** Attention chính xác (không xấp xỉ) nhưng tính theo khối (tiling) để giảm số lần đọc/ghi HBM; online softmax; phân tích IO complexity; block-sparse FlashAttention.  
**Vị trí trong lộ trình:** Đọc trước: 01 (Attention Is All You Need) để nắm $\mathrm{softmax}(QK^\top)V$; 17, 18 (attention thưa / tuyến tính) để thấy các cách giảm FLOPs mà bài này phản biện. Bài này thêm: góc nhìn "attention bị nghẽn bởi bộ nhớ, không phải FLOPs" và một kernel chính xác, nhanh hơn. Đọc tiếp: 45 (FlashAttention-2, cải thiện phân chia công việc trên GPU), 46 (PagedAttention, quản lý KV cache khi phục vụ), 14-15 (MQA/GQA, giảm KV cache).

> Đây là ghi chú học tập bằng tiếng Việt, diễn giải lại chứ không phải bản dịch. Các số liệu lấy từ paper (có ghi rõ). Mọi phần "Ứng dụng cho dự án" và các con số đánh dấu "tự tính" là suy luận của chúng ta, không phải kết quả paper chứng minh. Paper dài (~18k từ, nhiều phụ lục); ghi chú tập trung vào thuật toán, chứng minh IO và các kết quả chính.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
   - 3.1 [Kiến thức nền: HBM, SRAM, memory-bound](#31-kiến-thức-nền-hbm-sram-memory-bound)
   - 3.2 [Attention chuẩn và chi phí IO](#32-attention-chuẩn-và-chi-phí-io)
   - 3.3 [Online softmax: suy ra từng bước](#33-online-softmax-suy-ra-từng-bước)
   - 3.4 [Thuật toán FlashAttention (forward)](#34-thuật-toán-flashattention-forward)
   - 3.5 [Backward: tính lại thay vì lưu](#35-backward-tính-lại-thay-vì-lưu)
   - 3.6 [Phân tích IO complexity](#36-phân-tích-io-complexity)
   - 3.7 [Block-sparse FlashAttention](#37-block-sparse-flashattention)
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

Self-attention có chi phí thời gian và bộ nhớ **bậc hai** theo độ dài chuỗi $N$ (ma trận điểm $N\times N$). Trước FlashAttention, hướng chính để "cứu" là **xấp xỉ**: attention thưa (sparse), hạng thấp (low-rank, Linformer), hoặc tuyến tính (Performer, Linear Attention). Các phương pháp này giảm **FLOPs** xuống tuyến tính hoặc gần tuyến tính, nhưng paper chỉ ra: nhiều phương pháp **không nhanh hơn trong thời gian thực (wall-clock)** so với attention chuẩn, vì chúng bỏ qua **chi phí truy cập bộ nhớ (IO)**. Chất lượng cũng thường giảm.

Quan sát then chốt: trên GPU hiện đại, tốc độ tính toán tăng nhanh hơn tốc độ bộ nhớ, nên đa số phép toán trong Transformer bị nghẽn bởi việc đọc/ghi bộ nhớ (memory-bound), không phải bởi số phép nhân-cộng.

Gloss các thuật ngữ nền:

- **HBM (High Bandwidth Memory):** bộ nhớ chính của GPU (ví dụ 40-80 GB trên A100), băng thông ~1.5-2.0 TB/s. Lớn nhưng "chậm" tương đối.
- **SRAM (on-chip, shared memory):** bộ nhớ trên chip, rất nhỏ (A100: 192 KB mỗi SM, 108 SM) nhưng nhanh hơn HBM cỡ một bậc (paper ước ~19 TB/s).
- **Kernel:** một hàm chạy trên GPU; mỗi kernel đọc input từ HBM, tính, ghi output về HBM.
- **Kernel fusion:** gộp nhiều phép toán vào một kernel để input chỉ đọc một lần.
- **Arithmetic intensity:** số phép tính trên mỗi byte truy cập bộ nhớ. Thấp thì memory-bound, cao thì compute-bound.
- **Memory-bound vs compute-bound:** thời gian chạy do truy cập bộ nhớ quyết định vs do số phép tính quyết định. Paper nêu: nhân ma trận có chiều trong lớn là compute-bound; elementwise (activation, dropout) và reduction (softmax, layer norm) là memory-bound.

## 2. Ý tưởng chính

Tính chính xác $O=\mathrm{softmax}(QK^\top)V$ nhưng **không bao giờ ghi ma trận $N\times N$ ra HBM**. Chia $Q,K,V$ thành các khối nhỏ vừa SRAM, nạp từng khối vào SRAM, tính điểm, softmax và nhân với $V$ ngay trong SRAM, rồi chỉ ghi kết quả cuối $O$ ($N\times d$). Vì softmax cần cả hàng để chuẩn hoá, ta dùng **online softmax**: giữ vài thống kê nhỏ (max và tổng mũ) cho mỗi hàng và "chỉnh lại tỉ lệ" kết quả cũ mỗi khi thấy khối mới. Ở backward, thay vì lưu $S,P$ ($N\times N$), chỉ lưu $O$ và thống kê softmax rồi **tính lại** $S,P$ từng khối trong SRAM. FLOPs tăng nhẹ (do tính lại) nhưng IO giảm mạnh, nên chạy nhanh hơn và bộ nhớ chỉ tuyến tính theo $N$.

## 3. Phương pháp chi tiết

### 3.1 Kiến thức nền: HBM, SRAM, memory-bound

Mô hình IO của paper: có hai tầng bộ nhớ, SRAM (nhỏ, kích thước $M$ phần tử) và HBM (lớn). Chi phí = số phần tử di chuyển giữa hai tầng. Paper dùng mô hình IO complexity cổ điển (Aggarwal & Vitter 1988).

Ví dụ trực giác: bạn là đầu bếp (đơn vị tính), SRAM là mặt bàn bếp nhỏ, HBM là kho lạnh ở xa. Nếu mỗi bước nấu phải chạy ra kho lấy nguyên liệu rồi chạy về cất kết quả, bạn tốn thời gian chạy chứ không phải thời gian nấu. Fusion và tiling là cách mang đủ nguyên liệu cho cả công đoạn lên bàn, nấu xong mới cất.

### 3.2 Attention chuẩn và chi phí IO

Với $Q,K,V\in\mathbb R^{N\times d}$ ($N$ độ dài chuỗi, $d$ head dim):

$$S=QK^\top\in\mathbb R^{N\times N},\quad P=\mathrm{softmax}(S)\in\mathbb R^{N\times N},\quad O=PV\in\mathbb R^{N\times d}.$$

Triển khai chuẩn (Algorithm 0 của paper), mỗi bước là một kernel riêng:

1. Đọc $Q,K$, tính $S$, **ghi $S$ ra HBM**.
2. **Đọc $S$**, tính $P$, **ghi $P$ ra HBM**.
3. Đọc $P,V$, tính $O$, ghi $O$.

Hệ quả (Theorem 2): attention chuẩn cần $\Theta(Nd+N^2)$ truy cập HBM và $O(N^2)$ bộ nhớ. Vì thường $N\gg d$ (GPT-2: $N=1024$, $d=64$), số hạng $N^2$ chi phối. Thêm masking/dropout lên $S,P$ còn tăng số lần đọc/ghi.

### 3.3 Online softmax: suy ra từng bước

**Softmax ổn định số học.** Với vector $x\in\mathbb R^B$:

$$m(x)=\max_i x_i,\quad f(x)=\big[e^{x_1-m(x)},\dots,e^{x_B-m(x)}\big],\quad \ell(x)=\sum_i f(x)_i,\quad \mathrm{softmax}(x)=\frac{f(x)}{\ell(x)}.$$

Trừ $m(x)$ để tránh tràn số mũ; kết quả không đổi vì tử và mẫu cùng nhân $e^{-m}$.

**Ghép hai khối.** Cho $x=[x^{(1)},x^{(2)}]$. Khi đó $m(x)=\max(m(x^{(1)}),m(x^{(2)}))$ và

$$\ell(x)=e^{m(x^{(1)})-m(x)}\,\ell(x^{(1)})+e^{m(x^{(2)})-m(x)}\,\ell(x^{(2)}).$$

Lý do: $\ell(x^{(1)})=\sum_{i\in 1}e^{x_i-m^{(1)}}$ nên $e^{m^{(1)}-m}\ell(x^{(1)})=\sum_{i\in1}e^{x_i-m}$. Tức ta chỉ "đổi mốc trừ" từ $m^{(1)}$ sang $m$ bằng nhân một hệ số vô hướng. Vậy chỉ cần giữ $(m,\ell)$ cho mỗi hàng là gộp được từng khối một. (Paper gọi đây là *algebraic aggregation*.)

**Thêm phép nhân với $V$.** Xét một hàng truy vấn $q$, sau $j$ khối cột đã xử lý, đặt các đại lượng sau (đúng như phần chứng minh bằng quy nạp trong Phụ lục C):

$$m^{(j)}=\max_{\text{cột}\le j} s,\quad \ell^{(j)}=\sum_{\text{cột}\le j}e^{s-m^{(j)}},\quad o^{(j)}=\sum_{\text{cột}\le j}\frac{e^{s-m^{(j)}}}{\ell^{(j)}}v .$$

Khối mới có $\tilde m=$ max cục bộ, $\tilde\ell=\sum e^{s-\tilde m}$, và $\tilde P V=\sum e^{s-\tilde m}v$ (chưa chia). Cập nhật:

$$m^{\text{new}}=\max(m^{(j)},\tilde m),\qquad \ell^{\text{new}}=e^{m^{(j)}-m^{\text{new}}}\ell^{(j)}+e^{\tilde m-m^{\text{new}}}\tilde\ell,$$

$$\boxed{\,o^{\text{new}}=\frac{1}{\ell^{\text{new}}}\Big(\ell^{(j)}e^{m^{(j)}-m^{\text{new}}}\,o^{(j)}+e^{\tilde m-m^{\text{new}}}\,\tilde P V\Big)\,}$$

Kiểm chứng: $\ell^{(j)}o^{(j)}=\sum_{\le j}e^{s-m^{(j)}}v$; nhân $e^{m^{(j)}-m^{\text{new}}}$ cho $\sum_{\le j}e^{s-m^{\text{new}}}v$. Số hạng khối mới tương tự. Tổng hai phần là $\sum_{\text{tất cả}}e^{s-m^{\text{new}}}v$, chia cho $\ell^{\text{new}}$ ra đúng softmax-weighted average. Nghĩa là sau khối cuối, $o$ **bằng đúng** kết quả attention chuẩn (không xấp xỉ).

**Ví dụ số nhỏ.** Một hàng có điểm $[1,3]$ (khối 1) và $[2]$ (khối 2), giá trị $v$ vô hướng $[10,20,30]$.
Khối 1: $m=3$, $f=[e^{-2},1]=[0.135,1]$, $\ell=1.135$, $o=(0.135\cdot10+1\cdot20)/1.135=18.81$.
Khối 2: $\tilde m=2$, $\tilde\ell=1$, $\tilde PV=30$. $m^{\text{new}}=3$; $\ell^{\text{new}}=1.135+e^{-1}\cdot1=1.503$; $o^{\text{new}}=(1.135\cdot e^{0}\cdot18.81+e^{-1}\cdot30)/1.503=(21.35+11.04)/1.503=21.55$.
Trực tiếp: trọng số $\propto[e^{1},e^{3},e^{2}]=[2.718,20.09,7.389]$, tổng 30.2; $o=(27.18+401.7+221.7)/30.2=21.55$. Khớp (tự tính).

### 3.4 Thuật toán FlashAttention (forward)

Kích thước khối (Algorithm 1): $B_c=\lceil M/4d\rceil$, $B_r=\min(\lceil M/4d\rceil,d)$, với $M$ là kích thước SRAM tính theo phần tử. Khối $Q_i$ ($B_r\times d$), $K_j,V_j$ ($B_c\times d$); $S_{ij}$ ($B_r\times B_c$) phải vừa SRAM, nên mới có hệ số 4 ở mẫu.

```text
Input: Q,K,V (N x d) trong HBM; SRAM kích thước M
Chọn Bc = ceil(M/4d), Br = min(ceil(M/4d), d)
Khởi tạo trong HBM: O = 0 (N x d), l = 0 (N), m = -inf (N)
Tr = ceil(N/Br), Tc = ceil(N/Bc)

for j = 1..Tc:                        # vòng ngoài: khối K,V
    nạp K_j, V_j  HBM -> SRAM
    for i = 1..Tr:                    # vòng trong: khối Q
        nạp Q_i, O_i, l_i, m_i  HBM -> SRAM
        S_ij  = Q_i K_j^T                       # trên chip
        m~_ij = rowmax(S_ij)
        P~_ij = exp(S_ij - m~_ij)               # theo từng phần tử
        l~_ij = rowsum(P~_ij)
        m_new = max(m_i, m~_ij)
        l_new = exp(m_i - m_new) * l_i + exp(m~_ij - m_new) * l~_ij
        O_i   = diag(l_new)^-1 ( diag(l_i) exp(m_i - m_new) O_i
                                 + exp(m~_ij - m_new) P~_ij V_j )   # ghi về HBM
        l_i, m_i <- l_new, m_new                                    # ghi về HBM
return O
```

Theorem 1: thuật toán trả đúng $O=\mathrm{softmax}(QK^\top)V$ với $O(N^2d)$ FLOPs và chỉ cần thêm $O(N)$ bộ nhớ (cho $m,\ell$). Phiên bản đầy đủ (Phụ lục B.3) xử lý thêm scale $\tau$, mask và dropout (lưu trạng thái RNG để tái tạo ở backward).

Ghi chú vòng lặp: vòng ngoài chạy theo $K,V$, vòng trong theo $Q$, nên $O,\ell,m$ bị đọc/ghi lại ở mỗi vòng ngoài. FlashAttention-2 (bài 45) đảo thứ tự này.

### 3.5 Backward: tính lại thay vì lưu

Backward chuẩn cần $P\in\mathbb R^{N\times N}$ để tính $dQ,dK,dV$. FlashAttention chỉ lưu $O$ và $(m,\ell)$ rồi **tính lại** $S_{ij},P_{ij}$ theo khối trong SRAM (đây là một dạng gradient checkpointing chọn lọc). Phụ lục B.2 tự suy ra gradient với bộ nhớ tuyến tính:

- $dV=P^\top dO$, nên $dv_j=\sum_i P_{ij}\,do_i$.
- $dP=dO\,V^\top$, nên $dP_{ij}=do_i^\top v_j$.
- Jacobian của softmax là $\mathrm{diag}(y)-yy^\top$, nên $dS_{i:}=P_{i:}\circ dP_{i:}-(P_{i:}^\top dP_{i:})P_{i:}$.
- Đặt $D_i=P_{i:}^\top dP_{i:}=\sum_j \frac{e^{q_i^\top k_j}}{L_i}do_i^\top v_j=do_i^\top o_i$ (đẹp ở chỗ $D_i$ chỉ cần $do_i$ và $o_i$, kích thước $d$).

$$\boxed{dS_{ij}=P_{ij}\,(dP_{ij}-D_i)},\qquad dq_i=\sum_j dS_{ij}k_j,\qquad dk_j=\sum_i dS_{ij}q_i.$$

Nhờ $D_i=do_i^\top o_i$, không cần có $P$ nguyên ma trận để tính tổng hàng. Paper so sánh với Rabe & Staats (2021): họ cũng tránh lưu $N\times N$ nhưng tập trung vào tổng bộ nhớ, không phải IO, nên không nhanh hơn attention chuẩn; còn FlashAttention cập nhật $O$ tăng dần (chỉ một bản sao) và đơn giản hoá backward bằng giải tích thay vì checkpoint từng khối.

### 3.6 Phân tích IO complexity

**Đếm.** Mỗi vòng ngoài (một khối $K_j,V_j$ cỡ $\Theta(M)$ phần tử) quét toàn bộ $Q$ và $O$ (mỗi cái $Nd$ phần tử). Số vòng ngoài $T_c=N/B_c=\Theta(Nd/M)$. Tổng truy cập HBM:

$$\underbrace{\Theta(Nd)}_{K,V\text{ đọc một lần}}+T_c\cdot\Theta(Nd)=\Theta\!\Big(Nd+Nd\cdot\frac{Nd}{M}\Big)=\Theta\!\Big(\frac{N^2d^2}{M}\Big)\quad(\text{vì }M\le Nd).$$

$$\boxed{\text{Chuẩn: }\Theta(Nd+N^2)\qquad\text{FlashAttention: }\Theta\!\Big(\frac{N^2d^2}{M}\Big)}\quad(d\le M\le Nd)$$

Với $d=64$-$128$ và $M\sim100$ KB, $d^2\ll M$ nên FlashAttention ít truy cập hơn nhiều lần (paper nói tới ~9 lần ít hơn, Hình 2 trái). Backward cũng $\Theta(N^2d^2M^{-1})$.

**Cận dưới (Proposition 3).** Không tồn tại thuật toán attention chính xác nào đạt $o(N^2d^2M^{-1})$ truy cập HBM **cho mọi** $M\in[d,Nd]$. Chứng minh: với $M=\Theta(Nd)$ mọi thuật toán phải đọc ít nhất $\Omega(Nd)$ dữ liệu vào. Đây là cận dưới trên một khoảng của $M$ (kiểu bài toán streaming); paper nhận là chưa chứng minh cận tham số hoá theo $M$ đầy đủ. Nghĩa "tối ưu" ở đây: tối ưu tiệm cận trên một dải kích thước SRAM, không phải tối ưu từng hằng số.

**Ví dụ số (tự tính, bỏ qua hằng số trong $\Theta$).** Lấy $N=2500$ (gần độ dài một trang OCR gồm prompt + output, giả định), $M=100$ KB $=50{,}000$ phần tử fp16.

| $d$ | $N^2$ (triệu phần tử) | $N^2d^2/M$ (triệu phần tử) | Tỉ lệ |
|---|---|---|---|
| 64 | 6.25 | 0.51 | ~12x ít hơn |
| 128 | 6.25 | 2.05 | ~3x ít hơn |

Nhận xét: lợi thế giảm khi $d$ tăng (vì $d^2/M$ tăng). Số thực tế còn phụ thuộc hằng số (attention chuẩn thực sự đọc/ghi $S,P$ nhiều lần, nên lợi thế thực lớn hơn tỉ lệ thô này). Số đo thực của paper: trên GPT-2 medium ($N=1024$, $d=64$, 16 head, batch 64, A100), forward+backward:

| | GFLOPs | HBM R/W (GB) | Runtime (ms) |
|---|---|---|---|
| Attention chuẩn | 66.6 | 40.3 | 41.7 |
| FlashAttention | 75.2 | 4.4 | 7.3 |

FLOPs của Flash **cao hơn** (do tính lại), nhưng HBM giảm ~9x và runtime giảm ~5.7x. Đây là bằng chứng trực tiếp cho luận điểm "IO quyết định thời gian".

### 3.7 Block-sparse FlashAttention

Cho mặt nạ khối $M\in\{0,1\}^{N/B_r\times N/B_c}$, thuật toán giống Algorithm 1 nhưng **bỏ qua khối bằng 0**. Proposition 4: IO là $\Theta(Nd+N^2d^2M^{-1}s)$ với $s$ là tỉ lệ khối khác 0. Với $s=N^{-1/2}$ hoặc $N^{-1}\log N$ thì IO còn $\Theta(N\sqrt N)$ hoặc $\Theta(N\log N)$. Thí nghiệm dùng mẫu thưa butterfly cố định. Đây là bằng chứng "proof of concept" rằng FlashAttention là một primitive cho attention xấp xỉ.

## 4. Thí nghiệm và kết quả chính

Phần cứng: chủ yếu 8×A100 (huấn luyện), một A100 40GB (benchmark).

**Tốc độ huấn luyện**

| Bài | Kết quả (paper) |
|---|---|
| BERT-large (seq 512), tới 72.0% MLM, 8×A100 | MLPerf 1.1 của Nvidia: 20.0 ± 1.5 phút; FlashAttention: 17.4 ± 1.4 phút (nhanh hơn 15%) |
| GPT-2 small, OpenWebText | HuggingFace 9.5 ngày (1.0x); Megatron-LM 4.7 ngày (2.0x); FlashAttention 2.7 ngày (3.5x); perplexity đều 18.2 |
| GPT-2 medium | HF 21.0 ngày; Megatron 11.5 ngày (1.8x); Flash 6.9 ngày (3.0x); ppl 14.2 / 14.3 / 14.3 |

(Phần tóm tắt của paper nói "tới 3x so với HF"; bảng cho 3.5x với small và 3.0x với medium. Ghi nguyên số bảng.)

**Chất lượng nhờ ngữ cảnh dài hơn**

| GPT-2 small, Megatron vs Flash | ctx | ppl | thời gian huấn luyện |
|---|---|---|---|
| Megatron-LM | 1k | 18.2 | 4.7 ngày (1.0x) |
| FlashAttention | 1k | 18.2 | 2.7 ngày (1.7x) |
| FlashAttention | 2k | 17.6 | 3.0 ngày (1.6x) |
| FlashAttention | 4k | 17.5 | 3.6 ngày (1.3x) |

Ngữ cảnh 4k vẫn nhanh hơn Megatron ngữ cảnh 1k cỡ 30% mà perplexity tốt hơn 0.7.

- **Long-Range Arena (1k-4k):** FlashAttention 2.4x so với Transformer chuẩn, độ chính xác trung bình 59.8 so với 59.3; block-sparse 2.8x, 59.6. Để so sánh: Linformer 2.5x nhưng chỉ 54.9; Performer 1.8x, 58.9; Reformer 1.3x, 57.6. Tức FlashAttention giữ nguyên độ chính xác của attention chuẩn mà tốc độ ngang các phương pháp xấp xỉ.
- **Phân loại tài liệu dài:** MIMIC-III: 52.8 (512) tăng lên 57.1 (16k); ECtHR: 72.2 (512) tăng lên 80.7 (8k), rồi 79.2 (16k).
- **Path-X (16K) và Path-256 (64K):** FlashAttention đạt 61.4 trên Path-X; block-sparse đạt 56.0 trên Path-X và 63.1 trên Path-256. Các Transformer khác trong bảng là chance/OOM. Đây là lần đầu Transformer vượt mức ngẫu nhiên.
- **Benchmark attention:** FlashAttention nhanh hơn tới ~3x so với PyTorch cho $N\in[128,2K]$; bộ nhớ tuyến tính, tới 20x ít hơn các baseline chính xác; hội tụ với các phương pháp xấp xỉ ở $N$ khoảng 512-1024 (xấp xỉ bắt đầu nhanh hơn từ đó); block-sparse nhanh hơn mọi baseline xấp xỉ biết được.

## 5. Phân tích / ablation

- **IO là biến quyết định thời gian** (Hình 2 trái): FLOPs cao hơn nhưng HBM thấp hơn 9x thì nhanh hơn rõ rệt.
- **Kích thước khối** (Hình 2 giữa): tăng $B_c$ giảm số lượt đọc $Q$, giảm HBM và giảm runtime; vượt khoảng 256 thì bị nghẽn bởi tính toán, và khối quá lớn không vừa SRAM.
- **Độ thưa** (Hình 2 phải): runtime của block-sparse giảm tỉ lệ với độ thưa ($N=4K$).
- **Độ ổn định số:** do không đổi định nghĩa mô hình, đường cong validation perplexity trùng với baseline (Phụ lục E).

## 6. Giới hạn

Do tác giả nêu:
- **Phải viết CUDA riêng cho mỗi biến thể attention**, tốn công, khó chuyển giữa các kiến trúc GPU; mong muốn có compiler từ ngôn ngữ cấp cao (kiểu Halide).
- IO-aware nên được áp dụng cho các module khác, chưa làm.
- Mới tối ưu trên một GPU; nhiều GPU thêm tầng IO giữa các GPU.

Do ta nhận thấy:
- Phân tích tối ưu là tiệm cận trên một dải $M$; không nói gì về hằng số, và thực tế GPU còn nhiều yếu tố khác (số warp, shared memory). Bài 45 chỉ ra FlashAttention v1 mới đạt 25-40% FLOPs lý thuyết.
- Toàn bộ benchmark tập trung huấn luyện/prefill với $Q$ dài. Với giải mã (decode) từng token, $Q$ chỉ có 1 hàng nên ma trận $S$ vốn đã nhỏ ($1\times N$) và lợi thế chính (tránh ghi $N\times N$) gần như biến mất. Paper không đánh giá decode.
- Không đề cập attention tuyến tính/recurrent (như GatedDeltaNet), nơi không có ma trận $N\times N$ để tránh.

## 7. Liên hệ với các bài khác

- **01 (Attention Is All You Need):** đây là cùng phép toán, chỉ đổi thứ tự tính toán.
- **17 (Longformer), 18 (Linear Transformers), 19-24:** những hướng giảm FLOPs bằng xấp xỉ/đổi cơ chế. Bài này cho thấy vì sao chúng cần kernel IO-aware mới nhanh thật (và các bài như 20, 24 đều có kernel chunk-wise kiểu tiling tương tự, cùng triết lý).
- **14-15 (MQA, GQA):** giảm KV cache khi decode; trực giao với FlashAttention (FA2 hỗ trợ GQA bằng chỉ số head).
- **45 (FlashAttention-2):** cùng thuật toán, cải thiện phân chia công việc để đạt ~2x.
- **46 (PagedAttention):** FlashAttention tối ưu *tính* attention; PagedAttention tối ưu *chỗ để* KV cache khi phục vụ nhiều request.
- **36-37 (InternVL, Qwen2-VL), 29 (ViT):** ViT có chuỗi patch dài, attention hai chiều đầy đủ; FlashAttention là cách thực tế để chạy.

## 8. Ứng dụng cho dự án (suy luận)

Dự án: LightOnOCR-3-0.8B, decoder lai (GatedDeltaNet + full attention), ViT 12 block, ~1,500 token output/trang, GPU ≤ 8 GB, batch 1.

**(a) ViT là nơi FlashAttention có lợi rõ nhất.** ViT dùng attention hai chiều trên toàn bộ patch của trang. Giả sử ~4,096 patch (giả định, không có số thật của LightOnOCR) và 16 head: ma trận điểm fp16 nếu vật hoá là $16\times4096^2\times2$ B $\approx$ **0.54 GB** cho mỗi layer (cộng thêm $P$ nếu ghi riêng), tức tự tính ~1 GB tạm thời trên GPU 8 GB. Dùng kernel fused (FlashAttention hoặc `scaled_dot_product_attention` của PyTorch chọn backend flash) làm con số này gần 0 và giảm IO. Kiểm tra: encoder có đang dùng SDPA/flash hay attention "ngây thơ" trong code HF.

**(b) Decoder lai: chỉ các lớp full attention hưởng lợi, và ít.** Giả định (Qwen3.5-style) 1 trong 4 layer là full attention: ~6 lớp của 24. Prefill với $N\approx2500$, $d=128$, 16 head: FLOPs attention chuẩn $4N^2dH\approx51$ GFLOP/layer (khoảng một nửa nếu causal), tổng ~154 GFLOP cho 6 lớp, so với phần tuyến tính $\approx2\cdot0.8\text{B}\cdot2500=4$ TFLOP (tự tính). Tức attention chỉ ~4% FLOPs của prefill: FlashAttention giúp chủ yếu ở **bộ nhớ đỉnh** (tránh ~200 MB cho $S$ mỗi lớp) hơn là FLOPs. GatedDeltaNet (24) không có ma trận $N\times N$; nó có kernel chunk-wise riêng, cùng tinh thần tiling.

**(c) Batch 1 decode: FlashAttention gần như không đổi.** Mỗi bước decode có 1 query, $S$ chỉ $1\times N$; việc nghẽn là đọc trọng số (xem bài 45, 46 cho chi tiết).

**(d) Tác động tới pruning/distillation.** Cắt số head hoặc head dim làm đổi $d$ và số head. Công thức IO $N^2d^2/M$ gợi ý: head dim nhỏ hơn thì FlashAttention càng có lợi (tỉ lệ $d^2/M$ nhỏ); nhưng kernel có thể chỉ hỗ trợ một tập head dim cố định (64, 128, ...), nên cắt head dim ra giá trị lạ (ví dụ 96) có thể rơi vào đường chậm. Đây là suy luận cần kiểm tra trên phiên bản kernel đang dùng, paper này không nói về điều đó.

## 9. Thí nghiệm gợi ý

1. **Đo ViT attention:** chạy encoder của LightOnOCR với (i) attention thủ công materialize $S$, (ii) SDPA backend flash, ở độ phân giải trang thật; ghi lại peak memory (`torch.cuda.max_memory_allocated`) và latency. So với ước lượng ~0.54 GB/layer.
2. **Kiểm tra "memory-bound":** viết microbenchmark với $N\in\{512,1024,2048,4096\}$, đo thời gian attention chuẩn vs fused, vẽ theo $N$; xem điểm giao cắt và so với dự đoán $N^2d^2/M$.
3. **Ảnh hưởng head dim sau pruning:** thử head dim 64, 96, 128 trong SDPA và xem backend nào được chọn (dùng `torch.backends.cuda.sdp_kernel` để ép) và tốc độ.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| Attention | $O=\mathrm{softmax}(QK^\top)V$ |
| Softmax ổn định | $\mathrm{softmax}(x)=f(x)/\ell(x)$, $f=e^{x-m}$ |
| Ghép khối | $\ell=e^{m_1-m}\ell_1+e^{m_2-m}\ell_2$ |
| Cập nhật output | $o^{new}=\frac{1}{\ell^{new}}(\ell e^{m-m^{new}}o+e^{\tilde m-m^{new}}\tilde PV)$ |
| Block size | $B_c=\lceil M/4d\rceil$, $B_r=\min(B_c,d)$ |
| IO chuẩn | $\Theta(Nd+N^2)$ |
| IO Flash | $\Theta(N^2d^2/M)$ |
| Bộ nhớ thêm | $O(N)$ ($m,\ell$) |
| FLOPs | $O(N^2d)$ |
| Backward | $D_i=do_i^\top o_i$, $dS_{ij}=P_{ij}(dP_{ij}-D_i)$ |
| Block-sparse IO | $\Theta(Nd+N^2d^2M^{-1}s)$ |
| Recomputation | lưu $O,m,\ell$; tính lại $S,P$ theo khối |

## 11. Kết luận ngắn

FlashAttention chuyển trọng tâm từ "giảm FLOPs" sang "giảm truy cập HBM" và chứng minh điều này bằng một thuật toán chính xác (tiling + online softmax + recomputation), có phân tích IO $\Theta(N^2d^2/M)$ và cận dưới. Kết quả: nhanh hơn 2-4x, bộ nhớ tuyến tính, cho phép ngữ cảnh dài. Với dự án OCR batch-1: lợi ích lớn nhất ở ViT và prefill (bộ nhớ đỉnh), còn decode từng token gần như không thay đổi.

Link gốc: https://arxiv.org/abs/2205.14135
