# Kimi Linear: An Expressive, Efficient Attention Architecture (2025)

**Tác giả:** Kimi Team (Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, ... Zhilin Yang, Yulun Du và nhiều người khác; danh sách đầy đủ trong paper).  
**Venue / arXiv:** arXiv:2510.26692 (v2, 1/11/2025); technical report. Code/kernel: `fla-org/flash-linear-attention` (fla/ops/kda); checkpoint `moonshotai/Kimi-Linear-48B-A3B-Instruct`.  
**Nguồn:** https://arxiv.org/abs/2510.26692  
**Chủ đề:** Kimi Delta Attention (KDA) = Gated DeltaNet với gate theo từng kênh; hybrid 3:1 KDA:MLA; NoPE cho layer full attention; so sánh công bằng với full attention.  
**Vị trí trong lộ trình:** Đọc trước: 20 (GLA), 23 (DeltaNet), 24 (Gated DeltaNet), 16 (MLA), 25 (Jamba). Đây là **hậu duệ trực tiếp của GDN ở quy mô 48B**; đọc tiếp: chuyển sang phần MoE/VLM (27 trở đi).

> Tài liệu học tập bằng tiếng Việt, ghi chú diễn giải, không phải bản dịch. Báo cáo kỹ thuật dài; ghi chú này tập trung vào **KDA (công thức, thuật toán chunkwise), kiến trúc hybrid và các kết quả chính**, bỏ phần đóng góp cá nhân, tham khảo và chi tiết RL. "Ứng dụng cho dự án" là suy luận của chúng tôi. Công thức chunkwise của KDA ở mục 3.3 đã được chúng tôi kiểm tra bằng số so với hồi quy (numpy). **Lưu ý ký hiệu:** paper này dùng $S\in\mathbb R^{d_k\times d_v}$ và $o_t=S_t^\top q_t$ (chuyển vị so với bài 23, 24 dùng $S\in\mathbb R^{d_v\times d_k}$, $o=Sq$).

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
   - 3.1 Từ GDN đến KDA
   - 3.2 Dẫn xuất chunkwise
   - 3.3 Vì sao KDA nhanh hơn DPLR tổng quát
   - 3.4 Kiến trúc Kimi Linear
   - 3.5 KDA như mã hoá vị trí học được
   - 3.6 Chi phí tính toán
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

Suy luận LLM ngày càng nặng ở các tác vụ **dài**: agent (nhiều lượt dùng công cụ), RL với thời gian suy luận kéo dài (test-time scaling), code cấp kho. Softmax attention có chi phí bậc hai và KV-cache tuyến tính theo độ dài → nút thắt.

Linear attention (bài 18) có state cố định nhưng từng yếu hơn softmax attention. Hai cải tiến thu hẹp khoảng cách: **gate/decay** (RetNet, GLA, Mamba2) và **delta rule** (DeltaNet 23, Gated DeltaNet 24). Tuy vậy, state hữu hạn vẫn hạn chế truy hồi/chép chính xác, nên các mô hình **lai** (vài layer attention toàn cục + nhiều layer tuyến tính) trở thành thoả hiệp thực tế (Jamba 25, Mamba-2 hybrid 22, ...). Lỗ hổng mà paper nhắm tới: các hybrid trước đó thường ở quy mô nhỏ hoặc thiếu đánh giá toàn diện; chưa có bằng chứng "vượt full attention trong so sánh công bằng" trên ngắn, dài và RL.

**Khái niệm nền nhắc nhanh.**
- **Gate theo head (vô hướng)** như Mamba2/GDN: $\alpha_t\in(0,1)$ một số cho cả state của head: mọi kênh nhớ cùng tốc độ.
- **Gate theo kênh (channel-wise)** như GLA: $\alpha_t\in(0,1)^{d_k}$, mỗi chiều key có tốc độ quên riêng, giống cách RoPE gán mỗi cặp chiều một tần số khác nhau.
- **DPLR (diagonal-plus-low-rank):** ma trận chuyển $D-a b^\top$ (đường chéo trừ hạng-1), tổng quát hoá cả gate theo kênh và delta rule, nhưng tính chunkwise đắt.

---

## 2. Ý tưởng chính

1. **KDA (Kimi Delta Attention):** lấy gated delta rule của GDN và thay gate vô hướng $\alpha_t$ bằng **gate đường chéo theo kênh** $\mathrm{Diag}(\alpha_t)$:
$$\boxed{S_t=\big(I-\beta_tk_tk_t^\top\big)\,\mathrm{Diag}(\alpha_t)\,S_{t-1}+\beta_tk_tv_t^\top,\qquad o_t=S_t^\top q_t}$$
Mỗi kênh key tự quyết mức giữ/quên; phần còn lại giống GDN (ghi đè kiểu delta rule).
2. **Biến thể DPLR bị ràng buộc:** bằng cách buộc hai vector "hạng thấp" $a,b$ đều theo $k$ ($a=\beta k$, $b=k\odot\alpha$), kernel chunkwise bớt phép nhân và tránh vấn đề số, nhanh gần **2×** so với DPLR tổng quát.
3. **Kimi Linear:** hybrid layer-wise, **3 layer KDA : 1 layer MLA (full attention)**, MoE, và layer MLA **không dùng mã hoá vị trí (NoPE)**, để KDA đóng vai trò mã hoá vị trí/recency. Kết quả (so sánh công bằng, cùng 1,4T token, cùng kích thước 48B tổng/3B active): vượt MLA toàn phần trên ngắn, dài và RL, giảm **tới 75% KV-cache** và tăng **tới 6,3×** tốc độ giải mã ở ngữ cảnh 1M.

---

## 3. Phương pháp chi tiết

### 3.1 Từ GDN đến KDA

Theo paper (ký hiệu $S\in\mathbb R^{d_k\times d_v}$):

| Mô hình | Cập nhật |
|---|---|
| Linear attention | $S_t=S_{t-1}+k_tv_t^\top$ |
| DeltaNet | $S_t=(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top$ |
| GDN (scalar gate) | $S_t=\alpha_t(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top$ |
| **KDA** | $S_t=(I-\beta_tk_tk_t^\top)\,\mathrm{Diag}(\alpha_t)\,S_{t-1}+\beta_tk_tv_t^\top$ |

(Nếu chuyển vị sang quy ước bài 24 $S'=S^\top$: $S'_t=S'_{t-1}\mathrm{Diag}(\alpha_t)(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$.) Chú ý **thứ tự**: trước hết decay theo kênh ($\mathrm{Diag}(\alpha_t)S_{t-1}$), rồi mới áp phép chiếu delta, khác một chút so với GDN, nơi $\alpha$ vô hướng nên giao hoán với mọi thứ.

**Cách đọc bộ nhớ liên kết.** Mỗi *hàng* $S[d,:]$ ứng với một chiều key $d$. Gate theo kênh quên hàng $d$ với hệ số $\alpha_t[d]$; phép chiếu delta rồi sửa lỗi giữa các hàng (trộn các chiều key bằng $k_tk_t^\top$). Tức là: các chiều key có thể giữ thông tin lâu mỏng khác nhau (một số "nhớ dài", một số "nhớ ngắn"), chống va chạm tốt hơn một gate chung.

**Góc nhìn học online (Bảng 7 paper).** Với GDN và KDA, cập nhật là một bước SGD trên bộ nhớ *đã bị decay* $\tilde S$ (decay vô hướng cho GDN, theo kênh cho KDA) với mất mát $\tfrac{\beta_t}{2}\|\tilde S_{t-1}^\top k_t-v_t\|^2$. Nghĩa là: "weight decay thích nghi" rồi "một bước hồi quy".

**Quan hệ với DPLR.** KDA viết lại thành $S_t=(\mathrm{Diag}(\alpha_t)-\beta_tk_tk_t^\top\mathrm{Diag}(\alpha_t))S_{t-1}+\beta_tk_tv_t^\top$, tức dạng $D-a_tb_t^\top$ với $D=\mathrm{Diag}(\alpha_t)$, $a_t=\beta_tk_t$, $b_t=k_t\odot\alpha_t$. So với DPLR tổng quát (như RWKV-7, S4), KDA **gắn $a$ và $b$ với $k$** (và chia sẻ $\alpha$ để rút ra ngoài), giảm số tham số/phép tính và vẫn nhất quán với delta rule cổ điển.

### 3.2 Dẫn xuất chunkwise

Mục tiêu giống 23/24: tính cả khối $C$ token bằng nhân ma trận. Ký hiệu trong khối: $\gamma^r=\prod_{i\le r}\alpha^i$ là vector $d_k$ chiều (tích luỹ theo kênh), $\Gamma^{1\to r}$ ghép chúng thành ma trận $C\times d_k$, và $\gamma^{i\to r}=\prod_{j=i}^r\alpha^j$.

**Khai triển** (Eq. 2): $S^r=P^rS^0+H^r$ với $P^r=\prod_{i=1}^r(I-\beta_ik_ik_i^\top)\mathrm{Diag}(\alpha_i)$.

**WY có gate** (Eq. 3-5; theo Comba để khỏi phải thêm nghịch đảo):
$$P^r=\mathrm{Diag}(\gamma^r)-\sum_{i=1}^r\mathrm{Diag}(\gamma^{i\to r})\,k_i\,w_i^\top,\qquad H^r=\sum_{i=1}^r\mathrm{Diag}(\gamma^{i\to r})\,k_i\,u_i^\top,$$
$$w_r=\beta_r\Big(\mathrm{Diag}(\gamma^r)k_r-\sum_{i<r}w_i\,\big(k_i^\top\mathrm{Diag}(\gamma^{i\to r})k_r\big)\Big),\quad u_r=\beta_r\Big(v_r-\sum_{i<r}u_i\,\big(k_i^\top\mathrm{Diag}(\gamma^{i\to r})k_r\big)\Big).$$
Điểm khác GDN: tích vô hướng giữa hai key là $k_i^\top\mathrm{Diag}(\gamma^{i\to r})k_r=\sum_d k_i[d]\,k_r[d]\,\gamma^{i\to r}[d]$, **có trọng số theo kênh**, nên không rút $\gamma$ ra ngoài thành một tỉ số vô hướng được.

**UT transform** (Eq. 6-7):
$$M=\Big[I+\mathrm{StrictTril}\Big(\mathrm{Diag}(\beta)\,(\Gamma^{1\to C}\odot K)\,\big(K/\Gamma^{1\to C}\big)^\top\Big)\Big]^{-1}\mathrm{Diag}(\beta),\quad W=M(\Gamma^{1\to C}\odot K),\ \ U=MV.$$
(phép chia $K/\Gamma$ theo từng phần tử.) Vẫn là nghịch đảo tam giác $C\times C$ giải bằng thế tiến.

**State cuối khối và đầu ra** (Eq. 8-9; chúng tôi đã kiểm tra số với kernel pseudo-code của paper):
$$\boxed{S_{[t+1]}=\mathrm{Diag}(\gamma^C)\,S_{[t]}+\big(\Gamma^{i\to C}\odot K\big)^{\!\top}\big(U-W\,S_{[t]}\big)}$$
$$\boxed{O_{[t]}=\underbrace{(\Gamma^{1\to r}\odot Q)\,S_{[t]}}_{\text{liên khối}}+\underbrace{\mathrm{Tril}\!\Big((\Gamma^{1\to r}\odot Q)\big(K/\Gamma^{1\to r}\big)^{\!\top}\Big)\big(U-W\,S_{[t]}\big)}_{\text{trong khối (attention trên "pseudo-value")}}}$$
Ở đây $\Gamma^{i\to C}\odot K$ là $K$ đã được decay tới cuối khối. Ý nghĩa giống bài 24: $U-WS$ là vector sửa lỗi thật cho cả khối (giá trị cần ghi trừ giá trị state đã dự đoán), tính một lần từ $S_{[t]}$. Hai điểm khác GDN: (i) mọi $\gamma$ là vector theo kênh, (ii) $W$ ở đây đã **gồm cả decay** (nên trong công thức không còn $\mathrm{diag}(\gamma)W$ như ở bài 24 mà là $W$ trực tiếp).

**Pseudo-code (chúng tôi tóm tắt từ Listing 1(b) của paper, một head; NumPy-style):**

```python
def chunk_kda(Q, K, V, beta, g, S, C=64):       # g = log(alpha) (L,dk); S (dk,dv); Kimi convention
    for chunk in chunks(C):
        gc = g[chunk].cumsum(0)                     # (C,dk) log gamma^{1->r}
        Aqk[i,j] = (Q[i] * K[j] * exp(gc[i]-gc[j])).sum(-1)   # j<=i
        Akk[i,j] = (K[i] * K[j] * exp(gc[i]-gc[j])).sum(-1)   # j<i
        A = solve_tri(I + beta[:,None]*Akk)         # UT: (I + StrictTril(...))^-1
        A = A * beta[None,:]
        W = A @ (exp(gc) * K);  U = A @ V
        O = (Q*exp(gc)) @ S + tril(Aqk) @ (U - W @ S)
        S = S*exp(gc[-1])[:,None] + (K*exp(gc[-1]-gc)).T @ (U - W @ S)   # dùng S cũ
```

Điều đáng chú ý cho kỹ sư: các hệ số đều dạng $\exp(\text{hiệu của log-tích luỹ})$ với chỉ số lớn trừ chỉ số nhỏ → $\le1$, ổn định; không chia cho số rất nhỏ.

### 3.3 Vì sao KDA nhanh hơn DPLR tổng quát

Khi decay theo kênh, thành phần $1/\Gamma$ trong dạng chunkwise (xuất hiện trong $K/\Gamma$) có thể gây **mất ổn định số** (chia số nhỏ). GLA giải bằng tính trong miền log và *chunk cấp hai ở độ chính xác đầy đủ* → không dùng được nhân ma trận nửa độ chính xác, chậm. DPLR tổng quát còn cần thêm các ma trận $A_{ab},A_{ak},A_{qb}$ (bốn ma trận chunk cấp hai). KDA đặt $a=b=k$ nên:
- số phép tính chunk cấp hai giảm từ **4 xuống 2**;
- loại thêm ba phép nhân ma trận ở khâu liên khối và đầu ra;
- hiệu quả toán tử tăng khoảng **100%** (≈2×) so với DPLR (Hình 2: kernel nhanh gần 2× đến 64K token; batch 1, 16 head).

### 3.4 Kiến trúc Kimi Linear

Nền: kiến trúc **Moonlight** (MoE kiểu DeepSeek), thay token-mixer.

**Tham số hoá lớp KDA (mỗi head $h$, $d_k=d_v=128$):**
$$q_t^h,k_t^h=\mathrm{L2Norm}\big(\mathrm{Swish}(\mathrm{ShortConv}(W_{q/k}^hx_t))\big),\qquad v_t^h=\mathrm{Swish}(\mathrm{ShortConv}(W_v^hx_t)),$$
$$\alpha_t^h=f\big(W_\alpha^\uparrow W_\alpha^\downarrow x_t\big)\in[0,1]^{d_k},\qquad\beta_t^h=\mathrm{Sigmoid}(W_\beta^hx_t),$$
$$o_t=W_o\Big(\mathrm{Sigmoid}(W_g^\uparrow W_g^\downarrow x_t)\odot\mathrm{RMSNorm}\big(\mathrm{KDA}(q_t,k_t,v_t,\alpha_t,\beta_t)\big)\Big).$$
- $\alpha$: **chiếu hạng thấp** (rank = head dim), hàm $f$ giống GDN/Mamba (decay dạng mũ).
- **Output gate sigmoid, hạng thấp** (ablation: sigmoid hơn swish rõ rệt; có gate hơn không gate; paper nói gate còn giúp giảm *attention sink*).
- Head-wise RMSNorm trước gate/projection.

**Hybrid:** các block gồm một token-mixer rồi MoE. Chọn **theo layer** (không theo head) vì đơn giản hạ tầng và ổn định huấn luyện; tỉ lệ **đều 3:1**: 3 layer KDA rồi 1 layer MLA (Multi-head Latent Attention, bài 16). Layer MLA dùng **NoPE** (không RoPE): KDA chịu trách nhiệm mã hoá vị trí và độ ưu tiên gần đây. Lợi ích NoPE (paper): (i) chuyển MLA sang MQA thuần khi suy luận (hiệu quả, vì không còn RoPE chen vào phép hấp thụ ma trận); (ii) mở rộng ngữ cảnh đơn giản, khỏi chỉnh tần số RoPE/YaRN.

**Cấu hình mô hình chính (mục 5.4):** 48B tham số tổng, 3B active; MoE 256 expert, 8 active (kể cả 1 shared); thưa 32; layer đầu dense. Pretrain 4096 token, optimizer MuonClip, lịch WSD, batch 32M token, 1,4T token cho so sánh; phiên bản phát hành 5,7T token, ngữ cảnh tới 1M.

### 3.5 KDA như mã hoá vị trí học được

Phép nhân tích luỹ các ma trận chuyển hoá cho thấy tính chất "vị trí" của layer tuyến tính. Thế attention tổng quát: $s_{t,i}=q_t^\top\big(\prod_{j=i+1}^tR_j\big)k_i$. RoPE chọn $R_j$ là ma trận xoay khối chéo (trực giao) nên chỉ phụ thuộc $t-i$. Với gated delta rule:
$$o_t=\sum_{i\le t}q_t^\top\Big(\prod_{j=i+1}^{t}\alpha_j\big(I-\beta_jk_jk_j^\top\big)\Big)k_i\,v_i\quad(\text{Eq. 12}),$$
nghĩa là GDN cũng là mã hoá vị trí *nhân*, nhưng ma trận chuyển **phụ thuộc dữ liệu** và **không bị ràng buộc trực giao** (do đó có thể mạnh hơn và hy vọng ngoại suy độ dài tốt hơn RoPE có tần số cố định). RoPE gán mỗi cặp chiều một tần số khác nhau (như một biến đổi Fourier không đều theo chiều đặc trưng); GDN chỉ có một decay chung. **KDA, với gate theo kênh, khôi phục sự đa dạng "theo chiều" đó**, và vì vậy được kỳ vọng làm vai trò vị trí tốt hơn, là lý do cho thiết kế NoPE ở MLA.

### 3.6 Chi phí tính toán

**FLOPs huấn luyện cho một head (paper, Eq. 13-14; chunk $C=64$):**
$$\mathrm{FLOPs}_{\rm KDA}(T)=6Td_h^2+3TCd_h+TC^2,\qquad \mathrm{FLOPs}_{\rm Attn}(T)=2T^2d_h.$$

Ví dụ số (tự tính, $d_h=128$, $C=64$): hệ số KDA $=6\cdot16384+3\cdot64\cdot128+64^2=126{,}976$ FLOPs/token/head.

| Độ dài $T$ | KDA (mỗi head) | Attention $2T^2d_h$ | Tỉ lệ |
|---|---|---|---|
| 1.500 | 0,19 GFLOPs | 0,58 GFLOPs | ≈3× |
| 4.096 | 0,52 GFLOPs | 4,29 GFLOPs | ≈8× |
| 65.536 | 8,3 GFLOPs | 1.100 GFLOPs | ≈132× |

Điểm hoà vốn $126{,}976\,T=256\,T^2$ cho $T\approx500$: ngắn hơn 500 token thì attention thuần rẻ hơn về FLOPs (chưa kể hiệu quả kernel và hằng số causal 1/2, nên chỉ là mức độ).

**Suy luận.** Prefill: kernel chunkwise (mục 3.2); giải mã từng token: kernel hồi quy (Eq. 2). State KDA cố định $d_k\times d_v=128\times128$ mỗi head, bất kể độ dài. Trong hybrid 3:1, chỉ 1/4 layer có KV-cache → **giảm tới 75% KV** (đúng bằng tỉ lệ layer, chúng tôi nhận xét). Khi giải mã bị chặn bởi I/O, thời gian tiệm cận chỉ còn 1/4 chi phí đọc KV so với full attention (paper: "hybrid efficiency ratio tối đa 3:1"). Bộ nhớ KV giảm còn cho phép batch lớn hơn → thông lượng cao hơn.

---

## 4. Thí nghiệm và kết quả chính

**Bài tổng hợp (mục 5.1, Hình 4).** Mô hình 2 layer, 2 head, head dim 128, tối đa 20.000 bước. *Palindrome* (sao chép ngược chuỗi), *MQAR* (truy hồi liên kết đa truy vấn), *Stack* (theo dõi trạng thái LIFO, 64 ngăn xếp). KDA đạt độ chính xác cao nhất ở mọi độ dài 256 đến 2.048 token và hội tụ nhanh hơn GDN trên Palindrome và MQAR; **Mamba2 thất bại cả ba** trong thiết lập này (chỉ có đồ thị, không có số).

**Quy luật scaling (mục 5.3, Bảng 2, Hình 5).** Năm mô hình MoE active từ 653M đến 1,7B (8/64 expert, Muon, 4096 ngữ cảnh), lượng token 38,8B đến 128B. Hồi quy:
- MLA: $2{,}3092\times C^{-0{,}0536}$; Kimi Linear: $2{,}2879\times C^{-0{,}0527}$ ($C$ = PFLOP/s-ngày).
- Kimi Linear đạt **~1,16× hiệu quả tính toán** so với MLA ở compute-optimal (paper cho rằng với tinh chỉnh siêu tham số sẽ còn tốt hơn).

**Kết quả pretrain 1,4T token (Bảng 3; cùng recipe, cùng kích thước):**

| Benchmark | MLA | GDN-H | Kimi Linear |
|---|---|---|---|
| HellaSwag | 81,7 | 82,2 | **82,9** |
| ARC-challenge | 64,6 | 66,5 | **67,3** |
| Winogrande | 78,1 | 77,9 | **78,6** |
| BBH | 71,6 | 70,6 | **72,9** |
| MMLU | 71,6 | 72,2 | **73,8** |
| MMLU-Pro | 47,2 | 47,9 | **51,0** |
| TriviaQA | 68,9 | 70,1 | **71,7** |
| GSM8K | 83,7 | 81,7 | **83,9** |
| MATH | 54,7 | 54,1 | 54,7 |
| EvalPlus | 59,5 | **63,1** | 60,2 |
| CRUXEval-I-cot | 51,6 | 56,0 | **56,6** |
| CRUXEval-O-cot | 61,5 | 58,1 | **62,0** |
| CEval | 79,3 | 79,1 | **79,5** |
| CMMLU | 79,5 | 80,7 | **80,8** |

(GDN-H = hybrid GDN:MLA cùng tỉ lệ, làm baseline "hybrid mạnh".)

**Sau SFT (Bảng 4):** Kimi Linear dẫn đầu ở hầu hết: ví dụ MMLU 77,0 (MLA 75,7; GDN-H 75,6), MMLU-Pro 67,4 (65,7; 64,8), GPQA-Diamond 62,1 (57,1; 58,6), AIME 2025 21,3 (20,6; 21,1), LiveCodeBench v6 26,0 (25,1; 25,4). Ngoại lệ: MATH500 (81,2 vs GDN-H 83,0), EvalPlus (61,0 vs 62,6 MLA), LiveBench (45,2 vs 46,4 GDN-H).

**Ngữ cảnh dài 128K (Bảng 5; cùng 1,4T token):**

| | RULER | MRCR | HELMET-ICL | LongBench v2 | Frames | RepoQA | LCA-Lib | LCA-Commit | Avg |
|---|---|---|---|---|---|---|---|---|---|
| MLA | 81,3 | 22,6 | 88,0 | 36,1 | 60,5 | 63,0 | 32,8 | 33,2 | 52,2 |
| GDN-H | 80,5 | 23,9 | 85,5 | 32,6 | 58,7 | 63,0 | 34,7 | 30,5 | 51,2 |
| Kimi Linear (RoPE) | 78,8 | 22,0 | 88,0 | 35,4 | 59,9 | 66,5 | 31,3 | 32,5 | 51,8 |
| **Kimi Linear** | **84,3** | **29,6** | **90,0** | 35,0 | 58,8 | **68,5** | **37,1** | 32,7 | **54,5** |

Điểm đáng nhớ: **GDN-H tụt xuống dưới MLA ở ngữ cảnh dài** (51,2 vs 52,2), còn Kimi Linear dẫn đầu (54,5); **Kimi Linear (RoPE)** ngang trên ngắn nhưng kém trên dài (51,8) → NoPE là quan trọng. Thứ bậc paper kết luận: pretrain/SFT: Kimi Linear > GDN-H > MLA; ngữ cảnh dài: Kimi Linear > MLA > GDN-H.

**RL (Hình 6):** huấn luyện RLVR toán học cùng thuật toán và siêu tham số; Kimi Linear tăng độ chính xác train và test nhanh hơn và cao hơn MLA đáng kể (MATH500, AIME 2025; đồ thị, không có số).

**Hiệu năng suy luận (Hình 1, 7; mô hình cỡ 48B).**
- Prefill: gần như trùng GDN-H; ngang MLA ở 4K-16K, nhanh dần từ 128K; **2,3× ở 512K và 2,9× ở 1M** so với MLA.
- Giải mã (TPOT): với batch 1, tăng tốc 2,3× ở 1M (mục 6.3). Khi tận dụng KV nhỏ để tăng batch: TPOT 1,84 ms so với 11,48 ms của MLA tại 1M token (**6,3×**). Paper viết cả "6×" lẫn "2,3×" ở các chỗ khác nhau (tuỳ batch); đọc kỹ điều kiện.
- Hình 1(a): trên RULER 128K, Kimi Linear 84,3 với tăng tốc 3,98×; MMLU-Pro (4K) 51,0 so với 47,2 (MLA) ở tốc độ tương đương.

**Bản phát hành 5,7T token (Phụ lục D, Bảng 8, 9):** so với Moonlight (16B tổng/3B active, cùng 5,7T): ví dụ MMLU-Pro 54,8 vs 42,4; GSM8K 86,3 vs 77,2; instruct: RULER@128k 95,4, RULER@1M 94,8, MMLU-Pro 72,7, AIME 2025 58,6. (So sánh này **không sạch**: đổi cả kiến trúc lẫn tổng số tham số, nên không dùng làm bằng chứng riêng cho KDA.)

---

## 5. Phân tích / ablation

**Bảng 1 (mô hình scaling nhỏ nhất 16 head 16 layer; PPL thấp hơn tốt hơn):**

| Biến thể | Train PPL | Val PPL |
|---|---|---|
| Hybrid ratio **3:1** (mặc định) | **9,23** | **5,65** |
| 0:1 (toàn MLA) | 9,45 | 5,77 |
| 1:1 | 9,29 | 5,66 |
| 7:1 | 9,23 | 5,70 |
| 15:1 | 9,34 | 5,82 |
| bỏ output gate | 9,25 | 5,67 |
| output gate **swish** | 9,43 | 5,81 |
| bỏ conv | 9,29 | 5,70 |

Rút ra (paper): 3:1 tốt nhất; 7:1 train PPL ngang nhưng val kém hơn rõ; 1:1 val ngang nhưng tốn suy luận hơn; toàn full attention (0:1) kém (!). Output gate sigmoid (5,65) hơi hơn không gate (5,67) và hơn hẳn swish (5,81); conv vẫn có ích trong hybrid. (Lưu ý: val PPL tính trên tập có phân phối khác pretrain nên nhấn mạnh khái quát hoá.)

**NoPE vs RoPE:** đoán của paper: ở bản RoPE, layer toàn cục mang tín hiệu vị trí tường minh mạnh, còn layer tuyến tính cung cấp thiên lệch vị trí ngầm yếu → lệch giữa các tầng, quá nhấn mạnh thứ tự ngắn → tốt ở ngữ cảnh ngắn nhưng kém linh hoạt khi mở rộng sang dài. Với NoPE, KDA gánh toàn bộ → cân bằng hơn qua độ sâu, ngoại suy tốt hơn.

**Hybrid layer-wise vs head-wise:** paper chọn layer-wise vì hạ tầng và ổn định, không có ablation số riêng trong text; thảo luận trong §7.2: intra-layer hybrid phức tạp, inter-layer thực tế hơn.

---

## 6. Giới hạn

**Do tác giả nêu:**
- Linear attention vẫn bị hạn chế bởi cài đặt phần cứng hiện tại và thiếu hạ tầng suy luận tối ưu (đã có vLLM + kernel mở).
- Tương lai: kết hợp với **sparse attention** (NSA, MoBA, DSA) để lấy sức mạnh truy hồi chi tiết; linear attention tuy vẫn bị phê bình về truy hồi/chép chính xác khi cực dài, có thể giảm thiểu qua "state expansion".
- Scaling law chưa tinh chỉnh siêu tham số kỹ cho KDA.

**Chúng tôi nhận xét:**
- Kết quả "vượt full attention" chỉ cho **một họ** (MoE 48B/3B, 1,4T token, MLA làm baseline, không phải GQA/MHA thông thường). Chênh lệch trên nhiều benchmark khá nhỏ (ví dụ MATH 54,7 = 54,7; HellaSwag +1,2).
- Không có ablation tách "gate theo kênh" khỏi các yếu tố khác (NoPE, output gate, conv) ở cùng quy mô lớn: lợi ích của KDA so với GDN-H được thể hiện qua so sánh hệ thống, không phải ablation đơn nhân tố ở 48B; Hình 4 (bài tổng hợp nhỏ) là bằng chứng trực tiếp nhất.
- Không có thử nghiệm đa phương thức/OCR; không có thử nghiệm cắt tỉa hay chưng cất.
- Con số tăng tốc phụ thuộc kernel và batch (xem lưu ý trên); giữ nguyên điều kiện khi so sánh.

---

## 7. Liên hệ với các bài khác

- **24 Gated DeltaNet:** KDA = GDN với $\alpha_t\in\mathbb R^{d_k}$ thay cho vô hướng; cùng WY/UT, cùng L2-norm, short conv. GDN-H là baseline trực tiếp trong paper.
- **20 GLA:** nguồn của gate theo kênh; KDA bổ sung delta rule và kernel chunkwise ổn định hơn.
- **23 DeltaNet:** nền delta rule và WY.
- **16 MLA:** layer toàn cục của Kimi Linear; thêm NoPE → chuyển MQA khi suy luận (bài 14).
- **12 RoPE:** mục 3.5 giải thích vì sao GDN/KDA đóng vai "mã hoá vị trí nhân phụ thuộc dữ liệu".
- **25 Jamba:** hybrid sớm (1:7, Mamba-1, MoE); Kimi Linear tăng tỉ lệ tuyến tính lên 3:1 nhưng dùng layer mạnh hơn và đánh giá cả RL.
- **22 Mamba-2:** bài này cho thấy Mamba2 thất bại ở các bài tổng hợp (không có delta rule).
- **27/28 MoE:** Kimi Linear là MoE 256 expert; không phải trọng tâm.
- **44 FlashAttention / 46 PagedAttention:** vẫn cần cho 1/4 layer MLA; KDA cần kernel riêng (fla).

---

## 8. Ứng dụng cho dự án (suy luận)

Decoder của bạn (Qwen3.5-style: GatedDeltaNet + full attention) thuộc cùng họ "GDN-H" mà paper dùng làm baseline. Giá trị của bài với dự án:

1. **Gợi ý hướng nâng cấp khi chưng cất (hoặc thiết kế học sinh):** chuyển gate vô hướng của GDN thành gate theo kênh (KDA) làm tăng khả năng điều khiển bộ nhớ ở cùng kích thước state. Nhưng **không phải cắt tỉa**: thay đổi lớp là thay kiến trúc, nên cần huấn luyện lại phần gate (chi phí tham số nhỏ vì gate dùng chiếu hạng thấp) và kernel KDA; chỉ nên thử nếu hiệu quả chưng cất từ GDN cho thấy bộ nhớ là điểm nghẽn.
2. **Cắt tỉa chiều $d_k$ tự nhiên hơn với gate theo kênh.** Với gate vô hướng, mỗi head chung một decay nên cắt vài chiều $d_k$ không đổi decay; với KDA, mỗi chiều key có decay riêng, cắt chiều nào là cắt cả "đường nhớ" với thang thời gian riêng. Có thể xếp hạng các chiều $d_k$ theo $\bar\alpha[d]$ (chiều có $\alpha\approx1$ là bộ nhớ dài, giữ; $\alpha$ nhỏ là cục bộ, ứng viên cắt). Chỉ là suy luận, đo bằng thực nghiệm.
3. **Bằng chứng về tỉ lệ attention.** 3:1 là tối ưu trong Bảng 1; 7:1 làm val PPL xấu đi (5,65 → 5,70), 15:1 còn xấu hơn (5,82). Nếu khi cắt bạn đang cân nhắc bỏ layer attention để giảm nặng trọng số, đây là cảnh báo về ngưỡng (ở quy mô và dữ liệu khác nhau; không dùng như hằng số). Dự án có thể đang ở 3:1 sẵn.
4. **NoPE ở layer attention.** Nếu decoder của bạn dùng RoPE ở layer attention và bạn muốn mở rộng số token hình ảnh hoặc độ dài sinh, paper gợi ý NoPE + layer tuyến tính vị trí học được có ngoại suy tốt hơn trong hybrid của họ. Với OCR, vị trí thông qua visual token (2D) thêm phức tạp (Qwen2-VL M-RoPE, bài 37), nên không áp dụng nguyên xi; cần thử nghiệm riêng.
5. **Hệ quả về latency batch-1 trên ≤ 8 GB.** Số liệu "6,3×" của paper là ở ngữ cảnh 1M và nhờ batch lớn: với batch 1, ~3K token và mô hình 0,8B, thời gian giải mã bị chặn bởi đọc trọng số (xem bài 24 mục 3.6 và bài 25 mục 8). **Không kỳ vọng tăng tốc lớn từ riêng kiến trúc**; lợi ích nằm ở bộ nhớ ổn định và batch lớn / nhiều trang song song.
6. **Công thức chi phí cho ngân sách prefill:** bảng 3.6 cho thấy ở $T=1500$ KDA/GDN rẻ hơn attention khoảng 3× mỗi head về FLOPs, ở $T=4096$ khoảng 8×; prefill ảnh có nhiều nghìn visual token thì lợi ích rõ hơn. Số này theo công thức của paper cho kernel chunk, chưa tính hiệu quả kernel thực.
7. **Phép đo chẩn đoán giá trị:** paper dùng *Palindrome, MQAR, Stack* ở quy mô 2 layer để kiểm tra khả năng bộ nhớ; có thể tái sử dụng làm kiểm thử nhanh cho học sinh đã cắt (rẻ hơn đánh giá OCR đầy đủ).

---

## 9. Thí nghiệm gợi ý

1. **Phân bố $\alpha$ theo chiều.** Nếu có thể truy cập gate (hoặc trên GDN, theo head), vẽ phân bố $\bar\alpha$; với GDN chỉ có một giá trị mỗi head. So sánh "chiều nhớ dài" vs "ngắn" để dự đoán chiều có thể cắt và kiểm chứng bằng cắt thật + distillation.
2. **Tái dùng bài tổng hợp của paper.** Cài MQAR/Palindrome nhỏ (2 layer, head dim 128) cho học sinh cắt $d_k\in\{128,64,32\}$ để thấy đường cong dung lượng nhớ; đối chiếu với độ chính xác chép OCR.
3. **GDN → KDA chỉ-gate.** Trên một mô hình nhỏ (~0,1B), thay gate vô hướng bằng gate theo kênh hạng thấp, chưng cất từ cùng giáo viên và đo OCR + MQAR; kiểm tra xem có đạt cùng chất lượng với $d_k$ nhỏ hơn không.

---

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| GDN | $S_t=\alpha_t(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top$ (quy ước $S\in\mathbb R^{d_k\times d_v}$) |
| **KDA** | $\boxed{S_t=(I-\beta_tk_tk_t^\top)\mathrm{Diag}(\alpha_t)S_{t-1}+\beta_tk_tv_t^\top}$, $\alpha_t\in[0,1]^{d_k}$ |
| DPLR | $D-a_tb_t^\top$; KDA: $D=\mathrm{Diag}(\alpha)$, $a=\beta k$, $b=k\odot\alpha$ |
| WY có gate | $P^r=\mathrm{Diag}(\gamma^r)-\sum\mathrm{Diag}(\gamma^{i\to r})k_iw_i^\top$ |
| UT | $M=[I+\mathrm{StrictTril}(\mathrm{Diag}(\beta)(\Gamma\odot K)(K/\Gamma)^\top)]^{-1}\mathrm{Diag}(\beta)$ |
| State khối | $\mathrm{Diag}(\gamma^C)S+(\Gamma^{i\to C}\odot K)^\top(U-WS)$ |
| FLOPs/head | KDA $6Td^2+3TCd+TC^2$; attention $2T^2d$ |
| Hybrid | 3 KDA : 1 MLA (NoPE), MoE 256 expert/8 active, 48B/3B |
| Kết quả chính | MMLU-Pro 51,0 vs 47,2 (MLA); RULER@128k 84,3 vs 81,3; KV −75%; TPOT 6,3× @1M |
| Scaling | ≈1,16× hiệu quả tính toán so với MLA |

## 11. Kết luận ngắn

Kimi Linear = KDA (GDN với gate theo kênh, ràng buộc DPLR để kernel nhanh) + hybrid 3:1 với MLA không mã hoá vị trí. Trong so sánh công bằng cùng 1,4T token, nó nhỉnh hơn full attention và hybrid GDN trên ngắn, dài, RL, đồng thời giảm 75% KV-cache và nhanh tới 6,3× khi giải mã ở 1M token. Điểm cần nhớ cho dự án: kiến trúc của bạn thuộc cùng họ (GDN-H); gate theo kênh và NoPE là hai cải tiến đã được chứng minh ở quy mô lớn; còn lợi ích tốc độ lớn chỉ xuất hiện ở ngữ cảnh rất dài/batch lớn, không phải batch-1 ngữ cảnh ngắn.

Nguồn gốc: https://arxiv.org/abs/2510.26692
