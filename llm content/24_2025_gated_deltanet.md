# Gated Delta Networks: Improving Mamba2 with Delta Rule (2025)

**Tác giả:** Songlin Yang (MIT CSAIL, thực hiện khi thực tập ở NVIDIA), Jan Kautz (NVIDIA), Ali Hatamizadeh (NVIDIA).  
**Venue / arXiv:** arXiv:2412.06464 (v3, 6/3/2025); ICLR 2025. Code: github.com/NVlabs/GatedDeltaNet.  
**Nguồn:** https://arxiv.org/abs/2412.06464  
**Chủ đề:** Linear attention có trạng thái ma trận; kết hợp *forget gate* (kiểu Mamba2) với *delta rule* (kiểu DeltaNet); thuật toán chunkwise cho gated delta rule; kiến trúc lai với sliding-window attention (SWA) / Mamba2.  
**Vị trí trong lộ trình:** Đọc trước: 18 (Linear Transformers), 19 (RetNet), 20 (GLA), 21 (Mamba), 22 (Mamba-2/SSD), 23 (DeltaNet song song hoá). Bài này **chính là lớp token-mixer trong decoder của dự án** (GatedDeltaNet xen kẽ full attention). Đọc tiếp: 25 (Jamba, hybrid), 26 (Kimi Linear, KDA = GDN với gate theo kênh).

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch. Phần "Ứng dụng cho dự án" là suy luận của chúng tôi, không phải kết quả của paper. Các phép dẫn xuất trong mục 3 do chúng tôi tự làm lại và **đã kiểm tra bằng số** (numpy, so recurrence với dạng chunkwise, sai số ở mức làm tròn). Lưu ý: trong bản text trích từ PDF, công thức đầu ra chunkwise (Eq. sau Eq. 10) bị lỗi định dạng; công thức đầu ra ở mục 3.4 là bản chúng tôi tự dẫn và kiểm chứng, không chép nguyên văn.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
   - 3.1 Nền: linear attention là bộ nhớ liên kết
   - 3.2 Delta rule = một bước SGD trên bộ nhớ
   - 3.3 Gated delta rule
   - 3.4 Thuật toán chunkwise (WY + UT) có gate
   - 3.5 Khối Gated DeltaNet và các biến thể lai
   - 3.6 Chi phí tính toán, kích thước state, KV-cache
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

**Vấn đề của softmax attention.** Self-attention chuẩn (bài 01) tốn $O(L^2)$ khi huấn luyện/prefill và phải giữ **KV-cache** (key, value của mọi token trước) có kích thước tăng tuyến tính theo $L$ khi sinh. Với văn bản dài (hoặc OCR sinh ~1.500 token/trang sau một prefix ảnh dài) đó là gánh nặng bộ nhớ và băng thông.

**Linear attention (bài 18).** Bỏ softmax, dùng tích trong thuần $q^\top k$, rồi đổi thứ tự nhân ma trận để viết thành RNN với **trạng thái ma trận** (gọi là *state*, *fast weight*, hay *bộ nhớ*):

$$S_t = S_{t-1} + v_t k_t^\top \in \mathbb R^{d_v\times d_k},\qquad o_t = S_t q_t \in \mathbb R^{d_v}.$$

Kích thước state **không đổi** theo độ dài chuỗi, nên suy luận là $O(1)$ bộ nhớ mỗi bước.

**Cách đọc "bộ nhớ liên kết" (associative memory).** $S_t=\sum_{i\le t} v_i k_i^\top$ là tổng các tích ngoài (outer product) "key → value". Khi truy vấn bằng $q$, ta tính $S_t q=\sum_i v_i (k_i^\top q)$: mỗi value được cộng vào với trọng số bằng độ giống giữa key cũ và query. Nếu các key $k_i$ trực giao đơn vị và $q=k_j$ thì $S_t k_j = v_j$, tức đọc lại đúng value đã ghi. Nhưng trong $\mathbb R^{d_k}$ chỉ có tối đa $d_k$ hướng trực giao; khi số cặp key–value $> d_k$ thì các key chồng lấn và việc đọc bị nhiễu: paper gọi là **memory collision** (va chạm bộ nhớ), dẫn Schlag et al. (2021).

**Hai hướng cải tiến trước bài này:**
1. **Gating / decay** (RetNet 19, GLA 20, Mamba 21, Mamba2 22): $S_t=\alpha_t S_{t-1}+v_tk_t^\top$ với $\alpha_t\in(0,1)$ phụ thuộc dữ liệu. Quên **đều** mọi liên kết cũ theo cùng một tỉ lệ. Quên nhanh được, nhưng không biết *liên kết nào* cần quên.
2. **Delta rule** (DeltaNet, bài 23): trước khi ghi $v_t$ vào khoá $k_t$, đọc ra giá trị cũ $S_{t-1}k_t$ và **thay** nó, thay vì cộng chồng. Sửa đúng một liên kết, nhưng không có cách *xoá nhanh hàng loạt* (ví dụ khi đổi chủ đề), nên DeltaNet chỉ "khá" trên dữ liệu thật (Yang et al. 2024b).

**Quan sát của paper:** hai cơ chế **bổ sung** cho nhau. Gate cho phép xoá nhanh/lọc; delta rule cho phép sửa có mục tiêu. Gộp cả hai thành *gated delta rule* và bổ sung thuật toán huấn luyện song song hiệu quả trên GPU, vì delta rule thuần chỉ có kernel tuần tự cho tới bài 23.

---

## 2. Ý tưởng chính

Hãy coi state $S$ là một **ma trận trọng số nhỏ** học ngay trong lúc đọc văn bản (một "fast weight" ánh xạ key → value). Ở mỗi token, mô hình (i) **hỏi** bộ nhớ "với key này, hiện tại ta đang nhớ giá trị gì?", (ii) so với giá trị mới cần ghi, (iii) sửa bộ nhớ một phần (mức sửa $\beta_t$, như learning rate), và (iv) trước đó **làm mờ** toàn bộ bộ nhớ một lượng $\alpha_t$ (như weight decay). Công thức:

$$\boxed{S_t = S_{t-1}\,\alpha_t\big(I-\beta_t k_tk_t^\top\big)+\beta_t v_tk_t^\top}$$

- $\alpha_t\to 0$: xoá sạch bộ nhớ (chuyển ngữ cảnh). $\alpha_t\to1$: giữ nguyên, chỉ còn delta rule thuần.
- $\beta_t\to 0$: bỏ qua token này. $\beta_t\to1$: thay hẳn giá trị cũ bằng $v_t$.

Về hệ thống: tác giả mở rộng thuật toán chunkwise dựa trên biểu diễn WY của Yang et al. (bài 23) để có gate, với chi phí gần như bằng DeltaNet. Kết quả: vượt Mamba2 và DeltaNet trên ngôn ngữ, suy luận thường thức, truy hồi trong ngữ cảnh, ngoại suy độ dài, LongBench; và các **mô hình lai** (với SWA hoặc Mamba2) còn tốt hơn.

---

## 3. Phương pháp chi tiết

**Quy ước ký hiệu** (theo paper GDN/DeltaNet): $S_t\in\mathbb R^{d_v\times d_k}$ (hàng = chiều value, cột = chiều key); $q_t,k_t\in\mathbb R^{d_k}$, $v_t\in\mathbb R^{d_v}$; $o_t=S_tq_t$. Bài Kimi Linear (26) dùng chuyển vị $S\in\mathbb R^{d_k\times d_v}$, nhớ đổi khi so công thức. Nhiều head chạy độc lập, mỗi head có $S$ riêng.

### 3.1 Nền: linear attention, Mamba2 và dạng song song

Linear attention thuần: $S_t=S_{t-1}+v_tk_t^\top$. Khai triển: $o_t=\sum_{i\le t}v_i(k_i^\top q_t)$; dạng ma trận $O=(QK^\top\odot M)V$ với $M$ là mặt nạ nhân quả. Đó là **dạng song song** (như attention, $O(L^2)$) và **dạng hồi quy** (như RNN, $O(L)$): hai biểu diễn của cùng một hàm.

Mamba2 thêm decay vô hướng: $S_t=\alpha_tS_{t-1}+v_tk_t^\top$. Gọi $\gamma_j=\prod_{i\le j}\alpha_i$ (tích luỹ), thì

$$o_t=\sum_{i\le t}\frac{\gamma_t}{\gamma_i}\,v_i\,(k_i^\top q_t),\qquad O=\big((QK^\top)\odot\Gamma\big)V,\quad \Gamma_{ij}=\tfrac{\gamma_i}{\gamma_j}\ (i\ge j).$$

Sự tương đương hai dạng này chính là "state space duality" (bài 22). Cả hai dạng đều không tối ưu cho GPU, nên người ta dùng dạng **chunkwise**: chia chuỗi thành khối độ dài $C$ (thường 64 hay 128), tính trong-khối bằng dạng song song (nhân ma trận, dùng tensor core) và truyền thông tin giữa các khối bằng state:

$$S_{[t+1]}=\overrightarrow{S}_{[t]}+\overrightarrow{V}_{[t]}^{\top}K\ \ (\text{có decay}),\qquad O_{[t]}=\overleftarrow Q_{[t]}S_{[t]}^\top+\big(Q_{[t]}K_{[t]}^\top\odot\Gamma_{[t]}\big)V_{[t]}.$$

Mũi tên trái/phải trong paper = "decay về vị trí đầu / cuối khối".

### 3.2 Delta rule = một bước SGD trên bộ nhớ

Xem state như tham số của một bộ hồi quy tuyến tính $\hat v = Sk$. Với mỗi token, hàm mất mát hồi quy online:

$$\mathcal L_t(S)=\tfrac12\|Sk_t-v_t\|^2,\qquad \nabla_S\mathcal L_t=(Sk_t-v_t)k_t^\top.$$

Một bước SGD với learning rate $\beta_t$:

$$S_t=S_{t-1}-\beta_t(S_{t-1}k_t-v_t)k_t^\top = S_{t-1}\big(I-\beta_tk_tk_t^\top\big)+\beta_tv_tk_t^\top.$$

Cách đọc theo "truy hồi key–value" (paper): giá trị cũ $v^{\rm old}_t=S_{t-1}k_t$; giá trị mới nội suy $v^{\rm new}_t=\beta_tv_t+(1-\beta_t)v^{\rm old}_t$; rồi **xoá cũ, ghi mới**:

$$S_t=S_{t-1}-v^{\rm old}_tk_t^\top+v^{\rm new}_tk_t^\top.$$

(Kiểm tra: $-v^{\rm old}k^\top+\beta v k^\top+(1-\beta)v^{\rm old}k^\top=-\beta v^{\rm old}k^\top+\beta vk^\top$, đúng bằng $-\beta S_{t-1}kk^\top+\beta vk^\top$.)

**Ma trận chuyển trạng thái là Householder tổng quát.** $I-\beta kk^\top$ có trị riêng $1$ (bội $d_k-1$) và $1-\beta\|k\|^2$ (bội 1). Nếu $\|k\|_2=1$ (L2-norm, bước bắt buộc trong paper) và $\beta\in(0,1)$ thì trị riêng nằm trong $(0,1]$: hệ **ổn định**. Khi $\beta=1$, $I-kk^\top$ là **phép chiếu** vuông góc: xoá đúng thành phần của state dọc hướng $k$, giữ nguyên $d_k-1$ hướng còn lại (đây là lý do paper nói L2-norm cho "quên có mục tiêu"). Chú thích 2 của paper: cho $\beta\in(0,2)$ thì trị riêng có thể âm, mở khả năng "state tracking"; GDN gốc dùng $\beta\in(0,1)$.

**Ví dụ nhỏ ($d_k=d_v=2$, $\beta=1$, key đơn vị).** $S_0=0$.
- Ghi $(k_1=e_1,\ v_1=(2,3))$: $S_1=\begin{pmatrix}2&0\\3&0\end{pmatrix}$.
- Ghi $(k_2=e_2,\ v_2=(5,7))$: $S_2=\begin{pmatrix}2&5\\3&7\end{pmatrix}$. Đọc $S_2e_1=(2,3)$, $S_2e_2=(5,7)$: đúng.
- Ghi lại khoá cũ $k_3=e_1$ với $v_3=(9,9)$. **Delta rule:** $S_2(I-e_1e_1^\top)=\begin{pmatrix}0&5\\0&7\end{pmatrix}$, cộng $v_3e_1^\top$ được $S_3=\begin{pmatrix}9&5\\9&7\end{pmatrix}$: ghi đè *chỉ* khoá 1. **Linear attention** cộng chồng: $S_3=\begin{pmatrix}11&5\\12&7\end{pmatrix}$, đọc $e_1$ ra $(11,12)=v_1+v_3$: va chạm.
- Với $\beta=0.5$ thì đọc $e_1$ ra $(5.5,6)=\tfrac12v_1+\tfrac12v_3$: đúng công thức nội suy.

### 3.3 Gated delta rule

$$\boxed{S_t=S_{t-1}\,\alpha_t\big(I-\beta_tk_tk_t^\top\big)+\beta_tv_tk_t^\top}\qquad(\text{Eq. 10})$$

Hai dạng viết tương đương, tiện cho code (đã kiểm tra bằng số):

$$S_t=\alpha_t\Big(S_{t-1}-\beta_t\,(S_{t-1}k_t)\,k_t^\top\Big)+\beta_t\,v_tk_t^\top .$$

Chú ý: $\alpha_t$ nhân cả phần *đã bị delta-sửa* của state, còn số hạng ghi mới $\beta_tv_tk_t^\top$ **không** bị decay. Tiếp tục ví dụ trên với $\alpha_3=0.5,\beta_3=1$: $S_3=0.5\begin{pmatrix}0&5\\0&7\end{pmatrix}+\begin{pmatrix}9&0\\9&0\end{pmatrix}=\begin{pmatrix}9&2.5\\9&3.5\end{pmatrix}$. Liên kết khoá 2 bị mờ đi một nửa; liên kết khoá 1 bị thay hẳn.

**Tham số hoá (theo paper):** $q,k$: Linear → ShortConv → SiLU → L2-norm; $v$: Linear → ShortConv → SiLU; $\beta_t=\sigma(W_\beta x_t)$ (chỉ Linear); $\alpha_t\in(0,1)$ dùng tham số hoá của Mamba2 (paper nói "dùng Mamba2's parameterization", không viết công thức; theo Mamba thì $\alpha_t=\exp(\Delta_t A)$ với $A<0$ học được và $\Delta_t=\mathrm{softplus}(\cdot)$ phụ thuộc $x_t$; phần này lấy từ bài 21, không có trong text GDN).

**Góc nhìn học online (Bảng 1 paper).** Mỗi luật cập nhật là nghiệm dạng đóng của một bài toán online:

| Mô hình | Mục tiêu online | Cập nhật |
|---|---|---|
| Linear Attn | $\|S_t-S_{t-1}\|_F^2-2\langle S_tk_t,v_t\rangle$ | $S_{t-1}+v_tk_t^\top$ |
| Mamba2 | $\|S_t-\alpha_tS_{t-1}\|_F^2-2\langle S_tk_t,v_t\rangle$ | $\alpha_tS_{t-1}+v_tk_t^\top$ |
| DeltaNet | $\|S_t-S_{t-1}\|_F^2-2\langle S_tk_t,\beta_t(v_t-S_{t-1}k_t)\rangle$ | $S_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$ |
| Gated DeltaNet | $\|S_t-\alpha_tS_{t-1}\|_F^2-2\langle S_tk_t,\beta_t(v_t-\alpha_tS_{t-1}k_t)\rangle$ | $S_{t-1}\alpha_t(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$ |

Cách đọc: số hạng $\|S_t-\alpha_tS_{t-1}\|^2$ kéo state mới về gần state cũ *đã co lại*; $\alpha_t$ nới "neo" ấy ra (khoan dung với việc quên). Cách đọc khác: **SGD với weight decay thích nghi** trên bộ nhớ, trong đó $\beta_t$ là learning rate thích nghi, $\alpha_t$ là hệ số weight decay thích nghi. (Linear attention dùng mất mát tuyến tính $-\langle Sk,v\rangle$, gradient hằng số, không sửa lỗi; delta rule dùng mất mát bậc hai nên gradient tỉ lệ với sai lệch, "sửa mạnh khi sai nhiều". Đây là giải thích trực giác trong chú thích của bài 23.)

### 3.4 Thuật toán chunkwise có gate (WY + UT)

Mục tiêu: tính cả khối $C$ token bằng các phép nhân ma trận, không dựng $S_t$ cho từng $t$. Trong một khối, đặt chỉ số $r=1..C$, state đầu khối $S_0$ (=$S_{[t]}$), $\gamma_r=\prod_{i\le r}\alpha_i$ (tích luỹ **bắt đầu lại ở đầu mỗi khối**), $\Gamma_{ij}=\gamma_i/\gamma_j$ với $i\ge j$.

**Bước 1: khai triển hồi quy.** Lặp Eq. 10:

$$S_r=S_0\underbrace{\prod_{i=1}^r\alpha_i(I-\beta_ik_ik_i^\top)}_{F_r}+\underbrace{\sum_{i=1}^r\beta_iv_ik_i^\top\prod_{j=i+1}^r\alpha_j(I-\beta_jk_jk_j^\top)}_{G_r}.$$

**Bước 2: biểu diễn WY cho tích Householder.** Gate là vô hướng nên rút ra được: $F_r=\gamma_rP_r$ với $P_r=\prod_{i\le r}(I-\beta_ik_ik_i^\top)$ giống hệt DeltaNet. Biểu diễn WY (Bischof & Van Loan 1985) cho phép viết $P_r=I-\sum_{i\le r}w_ik_i^\top$ với

$$w_r=\beta_r\Big(k_r-\sum_{i<r}w_i\,(k_i^\top k_r)\Big).$$

*Chứng minh nhanh (quy nạp):* $P_r=P_{r-1}(I-\beta_rk_rk_r^\top)=P_{r-1}-\beta_r(P_{r-1}k_r)k_r^\top$ và $P_{r-1}k_r=k_r-\sum_{i<r}w_i(k_i^\top k_r)$. $\square$

Với phần ghi $G_r$ (Phụ lục A paper, chứng minh quy nạp): $G_r=\sum_{i\le r}\frac{\gamma_r}{\gamma_i}\tilde u_ik_i^\top$, trong đó

$$\tilde u_r=\beta_r\Big(v_r-\sum_{i<r}\tilde u_i\,\tfrac{\gamma_r}{\gamma_i}\,(k_i^\top k_r)\Big).$$

(Tức là "pseudo-value": giá trị cần ghi sau khi đã trừ đi những gì các token trước *trong cùng khối* đã làm tới hướng $k_r$, có kể decay.)

**Bước 3: UT transform** để thay vòng hồi quy bằng một phép giải tam giác (nhân ma trận được). Gom $w_r,\tilde u_r$ thành ma trận $W,\tilde U$ theo hàng:

$$T=\Big[I+\mathrm{strictLower}\big(\mathrm{diag}(\beta)KK^\top\big)\Big]^{-1}\mathrm{diag}(\beta),\qquad W=TK,$$
$$\tilde T=\Big[I+\mathrm{strictLower}\big(\mathrm{diag}(\beta)(\Gamma\odot KK^\top)\big)\Big]^{-1}\mathrm{diag}(\beta),\qquad \tilde U=\tilde TV.$$

Ma trận cần nghịch đảo là tam giác dưới kích thước $C\times C$ (với $C=64$ thì rất nhỏ), nên giải bằng thế tiến (forward substitution) chứ không cần nghịch đảo tổng quát. Lưu ý: $W$ **không phụ thuộc gate** (giống DeltaNet), chỉ $\tilde U$ có $\Gamma$.

**Bước 4: cập nhật state cuối khối và đầu ra.** Thay vào bước 1 (tự dẫn, đã kiểm tra số), với $\gamma\in\mathbb R^C$ là vector tích luỹ:

$$\boxed{S_{[t+1]}=\gamma_C\,S_{[t]}+\Big(\tilde U-\mathrm{diag}(\gamma)\,W\,S_{[t]}^\top\Big)^{\!\top}\overrightarrow K,\qquad \overrightarrow K_i=\tfrac{\gamma_C}{\gamma_i}k_i}$$

$$\boxed{O_{[t]}=\mathrm{diag}(\gamma)\Big[Q\,S_{[t]}^\top-\big(QK^\top\odot M\big)\,W S_{[t]}^\top\Big]+\big(QK^\top\odot\Gamma\odot M\big)\tilde U}$$

Cách đọc đầu ra: số hạng 1 = đóng góp từ trạng thái đầu khối (đã bị decay $\gamma_r$ và đã bị các Householder trong khối "sửa" qua $W$); số hạng 2 = đóng góp từ các token trong khối (như linear attention với $\tilde U$ thay cho $V$). Trong bài báo, state cuối khối viết là $\overrightarrow S+(\tilde U-\overleftarrow W S^\top)^\top\overrightarrow K$ với $\overleftarrow w_i=\gamma_iw_i$, khớp với dạng trên.

**Ý nghĩa "pseudo-value".** $\tilde U-\mathrm{diag}(\gamma)WS_0^\top$ chính là *vector sửa thực sự* $\beta_i\big(v_i-\alpha\text{-decayed prediction}\big)$ mà DeltaNet sẽ tính nếu chạy tuần tự: "giá trị cần ghi trừ đi cái state đã dự đoán cho key này". Chunkwise tính tất cả $C$ vector sửa cùng lúc từ $S_0$ bằng nhân ma trận.

**Pseudo-code (PyTorch, một head, chúng tôi viết lại theo công thức trên; đã khớp recurrence tới sai số làm tròn):**

```python
def gated_delta_chunk(Q, K, V, beta, log_alpha, S, C=64):
    # Q,K: (L,dk) đã L2-norm; V: (L,dv); beta,log_alpha: (L,); S: (dv,dk)
    L, outs = Q.shape[0], []
    for s in range(0, L, C):
        q,k,v = Q[s:s+C], K[s:s+C], V[s:s+C]; b = beta[s:s+C]
        cum = log_alpha[s:s+C].cumsum(0)                  # log gamma_r
        gam = cum.exp()                                    # gamma_r (<=1)
        Gam = (cum[:,None]-cum[None,:]).exp().tril()       # Gamma_ij = gamma_i/gamma_j, i>=j
        KK  = k @ k.T
        I   = torch.eye(C)
        T   = torch.linalg.solve_triangular(I + torch.tril(b[:,None]*KK,-1), torch.diag(b), upper=False)
        Tt  = torch.linalg.solve_triangular(I + torch.tril(b[:,None]*(Gam*KK),-1), torch.diag(b), upper=False)
        W, U = T @ k, Tt @ v                               # (C,dk), (C,dv)
        M   = torch.tril(torch.ones(C,C))
        inter = gam[:,None] * (q @ S.T - ((q@k.T)*M) @ (W @ S.T))
        intra = ((q@k.T) * Gam) @ U
        outs.append(inter + intra)
        Kr = (gam[-1]/gam)[:,None] * k                     # decay về cuối khối
        S  = gam[-1]*S + (U - gam[:,None]*(W @ S.T)).T @ Kr
    return torch.cat(outs), S
```

Ghi chú ổn định số: chỉ dùng tỉ số $\gamma_i/\gamma_j$ với $i\ge j$ (luôn $\le1$), tính qua hiệu của log-tích luỹ, nên không có phép chia số rất nhỏ. Suy luận từng token dùng kernel hồi quy (mục 3.6), không dùng chunk.

### 3.5 Khối Gated DeltaNet và các biến thể lai

**Khối cơ bản** (Hình 1 phải): theo macro-kiến trúc Llama (khối token-mixer + MLP SwiGLU, bài 05, 10, 11). Trong token-mixer: các nhánh $q,k,v$ qua Linear → short conv (depthwise, kernel nhỏ) → SiLU; $q,k$ thêm L2-norm; $\alpha,\beta$ chỉ Linear; sau cổng hồi quy có **output norm** rồi **output gate** (Linear + SiLU) trước phép chiếu đầu ra.

**Lai.** Linear attention yếu ở so khớp/dịch cục bộ và có state kích thước cố định nên hạn chế truy hồi chính xác (Arora et al. 2024a). Hai biến thể:
- **GatedDeltaNet-H1:** xen **Gated DeltaNet + SWA** (sliding-window attention, attention chỉ nhìn cửa sổ 2K token gần nhất; xem bài 17 Longformer / Mistral 06) theo kiểu Griffin/Samba.
- **GatedDeltaNet-H2:** xếp **Mamba2 + Gated DeltaNet + SWA**.

(Lưu ý: hybrid của paper này là *linear + sliding-window*, không phải *linear + full attention toàn cục*. Dự án của bạn dùng full attention toàn cục xen kẽ, giống Jamba 25 và Kimi Linear 26 hơn.)

### 3.6 Chi phí tính toán, kích thước state, KV-cache

**Công thức (tổng quát, mỗi head, $d_k,d_v$ = chiều head):**

| Đại lượng | Softmax attention | Gated DeltaNet |
|---|---|---|
| Bộ nhớ khi sinh (mỗi layer) | KV-cache $2\,n_{kv}\,d_h\,T$ phần tử | State $H\,d_k\,d_v$ phần tử, **không phụ thuộc $T$** |
| FLOPs sinh 1 token | $\approx 4\,T\,d_h\,H_q$ (đọc cả cache) | $\approx 6\,H\,d_kd_v$ (đọc $Sk$, cập nhật hạng-1, đọc $Sq$) |
| FLOPs huấn luyện/prefill | $O(L^2d)$ | chunkwise: $O(LCd+Ld^2)$ (bài 23) cộng việc giải tam giác $C\times C$ mỗi khối (khoảng $C^3$/khối, tự suy; paper không nêu số cụ thể cho GDN) |
| Số bước tuần tự | 1 | $L/C$ (giữa các khối) |

**Ví dụ số (tự tính, giả định kích thước giống dự án của bạn; không lấy từ paper):** $d_{model}\approx1024$, 24 layer, 3:1 → 18 layer GDN + 6 layer full attention; GDN mỗi layer $H=8$ head, $d_k=d_v=128$; full attention $n_{kv}=4$, $d_h=128$, $H_q=8$, bf16 cho KV; state GDN giữ fp32 (thông lệ để tránh sai số tích luỹ; chưa kiểm chứng với repo của bạn).
- State GDN mỗi layer: $8\times128\times128=131{,}072$ phần tử = 512 KB (fp32); 18 layer ≈ **9,0 MB**, **cố định**.
- KV-cache full attention: $2\cdot4\cdot128\cdot2=2048$ byte/token/layer; 6 layer ≈ 12 KB/token. Tại $T=3000$ (prefix ảnh + 1.500 token sinh, giả định): ≈ **35 MB**. Nếu cả 24 layer đều là full attention: ≈ **141 MB** (gấp 4 lần).
- FLOPs đọc/ghi state GDN khi sinh 1 token: $6\cdot8\cdot128^2\approx0{,}79$ MFLOPs/layer; attention ở $T=3000$: $4\cdot3000\cdot128\cdot8\approx12{,}3$ MFLOPs/layer. Cả hai nhỏ so với phần chiếu tuyến tính ($\sim 2\times$ số tham số/layer ≈ vài chục MFLOPs).
- Băng thông ở batch 1: mỗi bước sinh đọc **toàn bộ trọng số** (0,8B × 2 byte = 1,6 GB bf16) cộng đọc+ghi state (18 layer × 512 KB × 2 ≈ 18 MB) cộng đọc KV (≈35 MB). Tức state + KV chỉ là ~3% so với trọng số. **Ở batch 1 với 1.500 token, decoding bị chặn bởi trọng số, không phải bởi KV/state.** Lợi thế của linear attention thể hiện rõ ở context rất dài hoặc batch lớn (nhiều sequence, mỗi cái một KV-cache), đúng như Kimi Linear (26) và Jamba (25) nhấn mạnh.

---

## 4. Thí nghiệm và kết quả chính

**Thiết lập.** Mọi mô hình 1,3B tham số, 100B token từ FineWeb-Edu, AdamW lr đỉnh 4e-4, weight decay 0,1, clip 1,0, cosine với warm-up 1B token, batch 0,5M token, tokenizer Llama2 (vocab 32.000), độ dài huấn luyện 4K; Samba và các biến thể lai dùng cửa sổ SWA 2K. Baseline: RetNet, HGRN2, Mamba, Mamba2, Samba, DeltaNet, Transformer++.

**Bảng 2: S-NIAH (kim trong đống cỏ), 1,3B, zero-shot**

| Mô hình | S-NIAH-1 (passkey) 1K/2K/4K/8K | S-NIAH-2 (số) 1K/2K/4K/8K | S-NIAH-3 (uuid) 1K/2K/4K |
|---|---|---|---|
| DeltaNet | 97,4 / 96,8 / 99,0 / 98,8 | 98,4 / 45,6 / 18,6 / 14,4 | 85,2 / 47,0 / 22,4 |
| Mamba2 | 99,2 / 98,8 / 65,4 / 30,4 | 99,4 / 98,8 / 56,2 / 17,0 | 64,4 / 47,6 / 4,6 |
| Gated DeltaNet | 98,4 / 88,4 / 91,4 / 91,8 | 100,0 / 99,8 / 92,2 / 29,6 | 86,6 / 84,2 / 27,6 |

Ba quan sát của paper:
1. **Decay làm hại việc giữ trí nhớ** (S-NIAH-1, ngữ cảnh lặp tổng hợp): Mamba2 tụt mạnh sau 2K vì quên lịch sử quá nhanh; DeltaNet gần hoàn hảo; Gated DeltaNet tụt ít hơn Mamba2 nhờ delta rule.
2. **Gate giúp lọc** (S-NIAH-2/3, ngữ cảnh là bài luận thật): state hữu hạn mà không xoá được thì các thông tin chồng lên nhau; DeltaNet tụt mạnh khi dài; Mamba2 và GDN giữ tốt hơn nhờ gate.
3. **Delta rule giúp ghi nhớ mẫu phức tạp** (S-NIAH-3, value là UUID): Mamba2 tụt nhanh, GDN tốt hơn.

**Bảng 3: ngôn ngữ và suy luận thường thức (zero-shot, dòng "Avg." là trung bình các benchmark acc; mỗi khối là số trong text, theo setup 1,3B/100B token)**

| Mô hình | Wiki ppl ↓ | LMB ppl ↓ | Avg ↑ |
|---|---|---|---|
| RetNet | 19,08 | 17,27 | 52,02 |
| HGRN2 | 19,10 | 17,69 | 51,79 |
| Mamba | 17,92 | 15,06 | 53,12 |
| Mamba2 | 16,56 | 12,56 | 54,89 |
| DeltaNet | 17,71 | 16,88 | 52,14 |
| **Gated DeltaNet** | **16,42** | **12,17** | **55,32** |
| Transformer++ | 18,53 | 18,32 | 52,25 |
| Samba | 16,13 | 13,29 | 54,00 |
| **GatedDeltaNet-H1** | 16,07 | 12,12 | **56,40** |
| **GatedDeltaNet-H2** | **15,91** | 12,55 | 56,18 |

Paper nói Gated DeltaNet vượt các mô hình tuyến tính khác ở cả 400M và 1,3B; bảng 400M không có trong text trích.

**Bảng 4: truy hồi thực tế, đầu vào cắt 2K token (acc)**

| Mô hình | SWDE | SQuAD | FDA | TQA | NQ | Drop | Avg |
|---|---|---|---|---|---|---|---|
| Mamba2 | 19,1 | 33,6 | 25,3 | 61,0 | 20,8 | 19,2 | 29,8 |
| DeltaNet | 17,9 | 30,9 | 18,4 | 53,9 | 17,3 | 18,6 | 26,2 |
| Gated DeltaNet | 25,4 | 34,8 | 23,7 | 60,0 | 20,0 | 19,8 | 30,6 |
| Transformer++ | 29,5 | 38,0 | 52,2 | 58,3 | 22,5 | 21,6 | 37,0 |
| Samba | 33,0 | 39,2 | 50,5 | 57,7 | 23,5 | 20,2 | 37,3 |
| GDN-H1 | 35,6 | 39,7 | 52,0 | 60,1 | 24,6 | 22,2 | 39,0 |
| GDN-H2 | 38,2 | 40,4 | 50,7 | 63,3 | 24,8 | 23,3 | 40,1 |

Điểm đáng nhớ: **mô hình thuần tuyến tính (kể cả GDN: FDA 23,7) vẫn thua xa Transformer (FDA 52,2) ở truy hồi chính xác**; chỉ khi lai với attention (H1/H2: FDA 52,0 và 50,7) mới ngang hoặc hơn. FDA = trích cặp key–value từ PDF, rất gần với nhu cầu OCR/trích xuất. Paper cho rằng khoảng cách giữa GDN và Mamba2 ở bảng này nhỏ hơn ở S-NIAH vì các mô hình nhỏ chưa tinh chỉnh hay lặp lại, lỗi này không phụ thuộc luật cập nhật.

**Ngoại suy độ dài (Hình 2, tới 20K token; chỉ có đồ thị, không có số trong text):** trên sáu tập dài (GovReport, QMSum, NarrativeQA, Qasper, CodeParrot, PG19), Gated DeltaNet có perplexity thấp nhất trong nhóm RNN, kết quả "hỗn hợp" nhưng bền hơn; lai cải thiện thêm. Paper nói rõ "future work sẽ khảo sát chuỗi dài hơn".

**Bảng 5: LongBench (trung bình 14 tác vụ)**

| Mô hình | Avg |
|---|---|
| RetNet | 13,2 |
| HGRN2 | 13,5 |
| Mamba | 14,6 |
| DeltaNet | 13,6 |
| Mamba2 | 13,5 |
| **Gated DeltaNet** | **16,6** |
| Transformer++ | 11,0 |
| Samba | 15,9 |
| GDN-H1 | 17,8 |
| GDN-H2 | **18,4** |

Trong các tác vụ riêng: GDN vượt hẳn ở TREC (30,0 vs 13,0 đến 23,1 cho các RNN khác) và SamSum (23,0 vs 8,4-12,8); đây là điểm mạnh được paper gán cho khả năng truy hồi, học trong ngữ cảnh và theo dõi trạng thái.

**Throughput huấn luyện (Hình 3, H100, 1,3B):** gated delta rule chỉ thêm overhead nhỏ so với delta rule; Gated DeltaNet ≈ DeltaNet; cả hai chậm hơn Mamba2 khoảng 2-3K token/s do ma trận chuyển trạng thái giàu thông tin hơn. Transformer++ nhanh nhất ở ngữ cảnh 2K nhờ FlashAttention-2; các biến thể lai với SWA 2K thì nhanh hơn mixer đơn lẻ; GDN-H1 giữ throughput tốt ở mọi độ dài. Số liệu chính xác chỉ có trên đồ thị.

---

## 5. Phân tích / ablation

**Bảng S.1: ablation khối GDN (400M, 15B token; Avg-PPL thấp hơn tốt hơn, Avg-Acc cao hơn tốt hơn)**

| Biến thể | Avg-PPL ↓ | Avg-Acc ↑ |
|---|---|---|
| Gated DeltaNet, head dim 128 (mặc định) | 27,35 | 47,26 |
| thay bằng **naive delta rule** (bỏ gate) | 30,87 | 45,12 |
| bỏ short conv | 28,95 | 46,16 |
| bỏ output gate | 29,12 | 45,46 |
| bỏ output norm | 27,55 | 47,07 |
| L1-norm + ReLU | 30,79 | 45,92 |
| L1-norm + 1+ELU | 30,34 | 46,05 |
| L1-norm + SiLU | 30,18 | 46,09 |
| L2-norm + ReLU | 27,67 | 46,94 |
| L2-norm + 1+ELU | 27,58 | 47,17 |
| head dim 64 | 28,31 | 46,35 |
| head dim 256 | 27,13 | 47,38 |

Rút ra (theo paper): (1) bỏ gate là mất lớn nhất (PPL 27,35 → 30,87); (2) **short conv** và **output gate** đều quan trọng, output norm ít; (3) **L2-norm thiết yếu**, loại feature map ít ảnh hưởng nhưng SiLU luôn nhỉnh hơn; (4) head dim 128 là điểm cân bằng, 256 nhỉnh hơn một chút (27,13) nhưng tốn hơn.

**Bảng S.2: thứ tự xếp các layer lai (500M, 15B token)**

| Thứ tự | Wiki ppl | LMB ppl | Avg acc |
|---|---|---|---|
| GDN + SWA + Mamba2 | 24,02 | 28,20 | 47,88 |
| GDN + Mamba2 + SWA | 23,69 | 26,83 | 47,54 |
| Mamba2 + SWA + GDN | 24,14 | 25,21 | 47,92 |
| **Mamba2 + GDN + SWA** | **23,54** | **24,11** | **48,73** |

Paper chọn thứ tự Mamba2 → GDN → SWA cho H2; chênh lệch nhỏ (khoảng 1 điểm acc), chỉ ở quy mô 500M/15B token nên không nên coi là quy luật.

**Điều paper không thay đổi bằng chứng:** không có ablation riêng "chỉ $\alpha$ vô hướng vs theo kênh" (đó là việc của KDA, bài 26), không có thử nghiệm quy mô > 1,3B trong bài này.

---

## 6. Giới hạn

**Do tác giả nêu:**
- Quy mô thử nghiệm: 1,3B/100B token; ngoại suy độ dài mới đến 20K, kết quả "hỗn hợp".
- Truy hồi chính xác trong ngữ cảnh vẫn thua Transformer (Bảng 4), cần lai.
- Throughput huấn luyện vẫn thấp hơn Mamba2 (2-3K token/s) do ma trận chuyển giàu cấu trúc.
- Delta rule có giới hạn biểu diễn lý thuyết (Irie et al. 2023); một số hướng mạnh hơn (hồi quy phi tuyến, TTT/Titans, mesa-layer) đánh đổi tính song song. Hạn chế đã biết nhưng paper chỉ đề xuất hướng $\beta\in(0,2)$ và nhiều tích Householder (chưa thử).

**Chúng tôi nhận xét (không phải tác giả nói):**
- Gate $\alpha_t$ là **vô hướng theo head**: mọi kênh key bị quên cùng tốc độ. Kimi Linear (26) chỉ ra điều này là hạn chế và thay bằng gate theo kênh.
- Chỉ so sánh hybrid dạng linear + SWA, không so với hybrid linear + full attention toàn cục (loại dự án của bạn dùng); kết luận "lai tốt hơn" có thể chuyển được, nhưng không được paper kiểm chứng ở đây.
- Mọi huấn luyện đều từ đầu (pretrain). Không có kết quả về chưng cất, cắt tỉa hay fine-tune lớp GDN.
- Ngữ cảnh là văn bản; không có thử nghiệm đa phương thức hay token ảnh.

---

## 7. Liên hệ với các bài khác

- **18 Linear Transformers:** nguồn gốc dạng hồi quy $S_t=S_{t-1}+v_tk_t^\top$; GDN thêm hai cơ chế sửa/quên lên đó.
- **19 RetNet / 20 GLA / 21 Mamba / 22 Mamba-2:** họ "decay". $\alpha_t$ trong GDN lấy từ Mamba2 (vô hướng theo head); Mamba2 là $\beta=0$ về mặt cấu trúc (không có số hạng $-\beta k k^\top$). Bài 22 giải thích dạng song song = nhân với mặt nạ semiseparable; GDN không nằm gọn trong khung đó vì ma trận chuyển không phải vô hướng.
- **23 DeltaNet:** nguồn của WY/UT và delta rule; GDN = bài 23 + $\alpha_t$ (thay đổi chủ yếu ở $\Gamma$ trong $\tilde T$).
- **25 Jamba / 26 Kimi Linear:** hybrid ở quy mô lớn; Kimi Linear xây trực tiếp lên GDN (gate theo kênh, tỉ lệ 3:1 với MLA, NoPE). Qwen3-Next/Qwen3.5 (kiến trúc 3:1 GDN:attention theo hiểu biết của chúng tôi, không nằm trong tập paper này) là hướng đi tương tự.
- **12 RoPE / 55 StreamingLLM:** GDN không có mã hoá vị trí tường minh; thông tin vị trí đến từ decay và short conv (Kimi Linear 26 phân tích kỹ hơn).
- **44 FlashAttention / 46 PagedAttention:** các layer full attention trong mô hình lai vẫn cần chúng; layer GDN thay KV-cache bằng state kích thước cố định, nên cấp phát bộ nhớ đơn giản hơn.

---

## 8. Ứng dụng cho dự án (suy luận)

Đây là bài **quan trọng nhất** cho kiến trúc decoder của bạn.

**1. Đọc đúng vai trò của mỗi loại layer.** Bằng chứng của bài (Bảng 2, 4) gợi ý: layer GDN làm "bộ nhớ nén, quên có chọn lọc" (tốt cho ngữ cảnh cục bộ/chuyển chủ đề, sinh văn bản liên tục), còn truy hồi chính xác từ xa (copy lại chuỗi hiếm, tra cứu theo khoá) thì cần full attention. Với OCR, nơi mỗi ký tự sinh ra phải bám vào vùng ảnh tương ứng trong prefix visual-token, các layer full attention nhiều khả năng giữ vai trò "nhìn lại ảnh/ chép chính xác". (Đây là giả thuyết, cần kiểm chứng; paper không có tác vụ OCR.)

**2. Vì sao cắt layer và cắt neuron có thể khác nhau.** Mỗi layer GDN có thêm các thành phần trạng thái: $\alpha_t$ (đặt mức nhớ), $\beta_t$ (mức ghi), L2-norm lên $q,k$, short conv. Thử nghiệm của bạn thấy "cắt neuron MLP hồi phục kém hơn cắt layer"; hãy phân biệt hai thứ khi nhìn GDN:
   - **Cắt chiều $d_v$** của một head (hàng của $S$): giảm số "kênh giá trị" lưu, tương đối vô hại vì mỗi hàng của $S$ là một bộ hồi quy độc lập cho một kênh value.
   - **Cắt chiều $d_k$** (cột của $S$, chiều key): giảm *dung lượng* bộ nhớ (tối đa $d_k$ cặp key–value trực giao) **và** làm thay đổi chuẩn của $k$; vì $q,k$ được L2-normalize *sau* chiếu nên cần chuẩn hoá lại; nếu không, $\|k\|\ne1$ phá điều kiện ổn định $1-\beta\|k\|^2\in[0,1)$. Hãy cắt trước chuẩn hoá và để distillation sửa; hoặc thử giữ nguyên $d_k$ và chỉ cắt số head.
   - **Cắt layer GDN** loại bỏ cả một "luồng nhớ"; khôi phục có thể dễ vì residual và các layer sau vẫn nhận được tín hiệu; nhưng cắt layer full attention có thể gây hại nặng cho tác vụ chép/tra cứu (suy đoán).
   - Về $D_{lin}$ / $D_{res}$ (giả thuyết để bạn đo): trong một layer GDN, đầu ra tuyến tính theo $v$ nhưng phi tuyến theo $k$ (qua tích $\prod(I-\beta kk^\top)$ và L2-norm). Vì vậy phần có thể thay bằng một ánh xạ tuyến tính tại điểm cắt ($D_{lin}$) có thể nhỏ hơn ở layer GDN có $\beta$ lớn và $\alpha\approx1$ (ghi mạnh, nhớ dài), và phần dư $D_{res}$ lớn hơn. Paper không đề cập; chỉ là dự đoán.

**3. Thống kê gate để chọn layer cắt.** $\alpha_t$ trung bình gần 1 nghĩa là layer nhớ dài hạn; gần 0 nghĩa là chỉ cục bộ. Layer có $\alpha$ thấp và $\beta$ thấp gần như chỉ là bộ lọc cục bộ, ứng viên cắt tốt. Layer $\alpha\approx1$, $\beta$ cao là "bộ nhớ dài thực sự", nên giữ. (Kiểm chứng bằng đo thực tế ở mục 9.)

**4. Kích thước state quyết định độ nhạy với số token sinh.** Mỗi head chỉ chứa tối đa $d_k$ cặp trực giao, trong khi 1.500 token/trang > $d_k=128$; delta rule + gate đã được thiết kế để xử lý va chạm. Nếu cắt $d_k$ xuống 64 hoặc 32 (ở mô hình 0,1B), hãy kỳ vọng suy giảm sớm ở các trang rất dài; phép đo "S-NIAH trong ngữ cảnh trang" của bạn nên tăng độ dài để thấy.

**5. Đặc tính tính toán ở batch 1 trên GPU ≤ 8 GB (tự tính ở mục 3.6):** state GDN ≈ 9 MB và KV của 6 layer attention ≈ 35 MB, so với trọng số 1,6 GB (0,8B) hay ≈ 0,6 GB (0,3B): **decode bị giới hạn bởi trọng số, nên cắt trọng số (tham số) mới giảm latency batch-1; cắt state/KV hầu như không**. Lợi ích kiến trúc GDN nằm ở: bộ nhớ bị chặn cố định khi prefix ảnh dài, batch lớn, hay nhiều trang cùng lúc. Với các kernel `flash-linear-attention`, prefill dùng chunk kernel, decode dùng recurrent kernel; kiểm tra rằng pipeline suy luận của bạn chọn đúng kernel, vì decode bằng chunk kernel với $L=1$ sẽ rất kém hiệu quả.

**6. Chưng cất.** Học sinh GDN phải học $\alpha,\beta$ phù hợp; nếu giáo viên 4B cũng là GDN-hybrid thì căn chỉnh theo từng layer (cùng loại layer) có lẽ ổn định hơn; nếu giữ cùng tỉ lệ và vị trí layer attention ở học sinh thì dễ so khớp hơn. Đây là đề xuất, không phải kết quả đã kiểm chứng.

---

## 9. Thí nghiệm gợi ý

1. **Thống kê gate theo layer.** Chạy mô hình 0,8B trên ~200 trang OCR, ghi $\bar\alpha_t$, $\bar\beta_t$ và "chiều dài nhớ hiệu dụng" $1/(1-\alpha)$ cho từng head/layer. Xếp hạng layer. So với kết quả cắt từng layer (tăng loss distill, $D_{lin}$ vs $D_{res}$). Dự đoán cần kiểm chứng: layer có $\alpha$ thấp dễ cắt, layer full attention khó cắt hơn.
2. **Quét $d_k$ và $d_v$ riêng.** Cắt $d_v$ (hàng $S$) và $d_k$ (cột $S$) độc lập tới 50%, có và không chuẩn hoá lại $q,k$ sau cắt, rồi phục hồi bằng distillation cùng ngân sách. Đo: perplexity, CER/WER trên tập OCR, và độ chính xác copy-lại chuỗi (trích 1 chuỗi dài từ đầu trang).
3. **Đo dung lượng state tối thiểu.** Tạo bài kiểm tra "kim trong đống cỏ" kiểu S-NIAH nhưng với số/uuid trong chính văn bản OCR, kéo dài 500 → 4000 token, so sánh bản gốc vs bản đã cắt $d_k$. Mục tiêu: tìm $d_k$ nhỏ nhất vẫn giữ >95%.

---

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| Linear attention | $S_t=S_{t-1}+v_tk_t^\top,\ o_t=S_tq_t$ |
| Mamba2 (gate) | $S_t=\alpha_tS_{t-1}+v_tk_t^\top$ |
| Delta rule | $S_t=S_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$ = SGD trên $\tfrac12\|Sk-v\|^2$ |
| **Gated delta rule** | $\boxed{S_t=S_{t-1}\alpha_t(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top}$ |
| Trị riêng chuyển | $1$ (bội $d_k-1$), $1-\beta\|k\|^2$; cần $\|k\|=1$ |
| WY | $P_r=I-\sum w_ik_i^\top,\ w_r=\beta_r(k_r-\sum_{i<r}w_i k_i^\top k_r)$ |
| UT | $T=[I+\mathrm{strictLower}(\mathrm{diag}(\beta)KK^\top)]^{-1}\mathrm{diag}(\beta)$, $W=TK$ |
| Pseudo-value có gate | $\tilde U=\tilde TV$, $\tilde T$ dùng $\Gamma\odot KK^\top$ |
| State cuối khối | $\gamma_CS+(\tilde U-\mathrm{diag}(\gamma)WS^\top)^\top\overrightarrow K$ |
| Đầu ra khối | $\mathrm{diag}(\gamma)[QS^\top-(QK^\top\odot M)WS^\top]+(QK^\top\odot\Gamma\odot M)\tilde U$ |
| H1 / H2 | GDN+SWA / Mamba2+GDN+SWA |
| Kích thước state | $H\,d_k\,d_v$ mỗi layer, không phụ thuộc $T$ |
| Memory collision | số liên kết > $d_k$ khiến key chồng lấn, đọc nhiễu |

## 11. Kết luận ngắn

Gated DeltaNet làm hai việc: về thuật toán, đưa gate vô hướng vào delta rule mà vẫn song song hoá được bằng WY + UT với chi phí gần bằng DeltaNet; về mô hình, cho thấy gate (quên nhanh) và delta rule (sửa có mục tiêu) bổ sung nhau, vượt Mamba2 và DeltaNet trên hầu hết bài kiểm tra, và lai với attention còn tốt hơn. Điểm cần nhớ cho dự án: state GDN là bộ nhớ liên kết cỡ cố định $d_k\times d_v$ mỗi head, $\alpha_t$ và $\beta_t$ là hai nút điều khiển "quên" và "ghi", L2-norm của $q,k$ là điều kiện ổn định, và truy hồi chính xác vẫn nhờ full attention.

Nguồn gốc: https://arxiv.org/abs/2412.06464
