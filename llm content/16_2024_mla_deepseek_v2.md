# DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (2024)

**Phạm vi ghi chú:** DeepSeek-V2 là báo cáo kỹ thuật rất dài (SFT, RL, đánh giá hàng chục benchmark). Ghi chú này **tập trung vào Multi-head Latent Attention (MLA)**: nén KV bằng low-rank, decoupled RoPE, so sánh KV-cache; phần DeepSeekMoE, huấn luyện, căn chỉnh chỉ tóm tắt ngắn.

**Tác giả:** DeepSeek-AI (research@deepseek.com); nhóm tác giả được liệt kê theo vai trò trong phần cuối paper, ghi nhận Huazuo Gao và Wangding Zeng có "key innovations" cho MLA.  
**Venue / arXiv:** arXiv:2405.04434 (cs.CL, v5, 19/6/2024); văn bản không nêu venue hội nghị.  
**Nguồn:** https://arxiv.org/abs/2405.04434  
**Chủ đề:** MLA (nén KV low-rank chung cho K và V), decoupled RoPE, DeepSeekMoE, mô hình 236B tổng / 21B kích hoạt, ngữ cảnh 128K.  
**Vị trí trong lộ trình:** Đọc sau 12 (RoPE), 14 (MQA), 15 (GQA); trước 27/28 (MoE) nếu muốn hiểu phần MoE. Paper bổ sung: cách đạt cache nhỏ hơn cả GQA mà chất lượng ≥ MHA, và vì sao RoPE gây khó cho nén KV.

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Mọi phép tính cho mô hình cỡ dự án và mọi suy diễn đại số (hấp thụ ma trận, vì sao RoPE gây xung đột) có đánh dấu "tự tính" hoặc là diễn giải của chúng ta dựa trên công thức của paper. Ứng dụng cho dự án là suy luận.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết: MLA](#3-phương-pháp-chi-tiết-mla)
4. [So sánh KV-cache](#4-so-sánh-kv-cache)
5. [DeepSeekMoE và phần còn lại (tóm tắt)](#5-deepseekmoe-và-phần-còn-lại-tóm-tắt)
6. [Thí nghiệm và kết quả chính](#6-thí-nghiệm-và-kết-quả-chính)
7. [Phân tích / ablation](#7-phân-tích--ablation)
8. [Giới hạn](#8-giới-hạn)
9. [Liên hệ với các bài khác](#9-liên-hệ-với-các-bài-khác)
10. [Ứng dụng cho dự án (suy luận)](#10-ứng-dụng-cho-dự-án-suy-luận)
11. [Thí nghiệm gợi ý](#11-thí-nghiệm-gợi-ý)
12. [Tóm tắt công thức / khái niệm](#12-tóm-tắt-công-thức--khái-niệm)
13. [Kết luận ngắn](#13-kết-luận-ngắn)

---

## 1. Bối cảnh

LLM mạnh hơn khi lớn hơn, nhưng đắt khi huấn luyện và chậm khi phục vụ. Hai nút cổ chai mà DeepSeek-V2 nhắm vào:

1. **KV-cache của MHA.** Mỗi token mỗi layer phải lưu $2n_hd_h$ phần tử; cache này "limits the maximum batch size and sequence length" (paper). Giải pháp cũ là MQA (bài 14) và GQA (bài 15) nhưng "their performance does not match MHA": paper trích kết quả ablation dense 7B (xem §6).
2. **Chi phí FFN dày.** Giải pháp là Mixture-of-Experts (MoE): FFN chia thành nhiều "expert", mỗi token chỉ dùng vài expert nên chỉ kích hoạt một phần tham số.

Khái niệm nền cần biết: **KV-cache** (lưu K,V của token trước khi sinh tuần tự), **low-rank** (ma trận lớn xấp xỉ bằng tích hai ma trận hẹp), **RoPE** (bài 12: xoay q,k theo vị trí, nên q·k chỉ phụ thuộc khoảng cách).

## 2. Ý tưởng chính

MLA không giảm số KV head như GQA. Thay vào đó, nó **nén** hidden state $h_t$ của mỗi token thành một vector latent nhỏ $c_t^{KV}\in\mathbb R^{d_c}$ (chung cho cả K lẫn V), và chỉ lưu vector này. Khi cần, các ma trận "up-projection" khôi phục K và V cho từng head từ latent; vì mỗi head có ma trận up-projection riêng, vẫn có $n_h$ head khác nhau (khác với MQA/GQA nơi head chia sẻ nguyên K/V). Điểm tinh tế: khi suy luận, các ma trận up-projection có thể **gộp vào** ma trận của query và của output, nên không phải dựng lại K và V đầy đủ. RoPE phá vỡ phép gộp này, nên paper tách một phần nhỏ của q và k riêng cho RoPE ("decoupled RoPE"), chỉ phần này cần cache thêm. Kết quả: cache mỗi token chỉ $(d_c+d_h^R)\,l$ phần tử, tương đương GQA với 2.25 nhóm, mà ablation cho thấy chất lượng còn tốt hơn MHA.

## 3. Phương pháp chi tiết: MLA

Ký hiệu của paper: $d$ chiều embedding; $n_h$ số head; $d_h$ chiều mỗi head; $h_t\in\mathbb R^d$ input attention của token $t$; $l$ số layer.

### 3.1 MHA chuẩn (Eq. 1-8)

$$q_t=W^Qh_t,\;k_t=W^Kh_t,\;v_t=W^Vh_t\;(\in\mathbb R^{d_hn_h}),\quad o_{t,i}=\sum_{j\le t}\mathrm{softmax}_j\!\Big(\tfrac{q_{t,i}^\top k_{j,i}}{\sqrt{d_h}}\Big)v_{j,i},\quad u_t=W^O[o_{t,1};\dots;o_{t,n_h}]$$

Cache MHA: $2n_hd_hl$ phần tử mỗi token.

### 3.2 Nén KV low-rank chung (Eq. 9-11)

$$
\boxed{c_t^{KV}=W^{DKV}h_t,\qquad k_t^C=W^{UK}c_t^{KV},\qquad v_t^C=W^{UV}c_t^{KV}}
$$

- $c_t^{KV}\in\mathbb R^{d_c}$ với $d_c\ll d_hn_h$: latent nén.
- $W^{DKV}\in\mathbb R^{d_c\times d}$: **down-projection** (nén).
- $W^{UK},W^{UV}\in\mathbb R^{d_hn_h\times d_c}$: **up-projection** cho key và value.
- Khi suy luận chỉ cache $c_t^{KV}$: $d_c\,l$ phần tử mỗi token.

Vì K và V đều sinh từ **cùng** latent, ma trận $[W^{UK};W^{UV}]$ có hạng ≤ $d_c$: giả định rằng thông tin cần cho cả key và value của mọi head nằm trong một không gian con $d_c$ chiều. $W^{UK}$ cắt latent thành $n_h$ khối key (mỗi head một khối), tương tự cho value.

Paper cũng nén **query** (Eq. 12-13) $c_t^Q=W^{DQ}h_t$, $q_t^C=W^{UQ}c_t^Q$ với $c_t^Q\in\mathbb R^{d_c'}$, nhưng chỉ để giảm **bộ nhớ activation khi huấn luyện**, không giảm KV-cache.

### 3.3 Hấp thụ ma trận khi suy luận (tự suy diễn từ phát biểu của paper)

Paper nói: "$W^{UK}$ can be absorbed into $W^Q$, and $W^{UV}$ can be absorbed into $W^O$". Diễn giải từng bước, cho head $i$ (ký hiệu $W^{UK}_i\in\mathbb R^{d_h\times d_c}$ là khối hàng của head $i$; $W^Q_i$ tương tự):

**Với key.** Score (chưa RoPE) giữa query token $t$ và key token $j$:

$$q_{t,i}^\top k_{j,i}^C=(W^Q_ih_t)^\top(W^{UK}_ic_j^{KV})=h_t^\top\underbrace{(W^Q_i)^\top W^{UK}_i}_{\text{tính trước, cỡ }d\times d_c}c_j^{KV}.$$

Gộp $\tilde W_i=(W^Q_i)^\top W^{UK}_i$ (offline), ta tính $\tilde q_{t,i}=\tilde W_i^\top h_t\in\mathbb R^{d_c}$ rồi $\tilde q_{t,i}^\top c_j^{KV}$ trực tiếp với **latent đã cache**; không bao giờ phải tạo $k_j^C$ cho các token tiền tố.

**Với value.** Đầu ra head $i$: $o_{t,i}=\sum_j a_{t,j,i}\,W^{UV}_ic_j^{KV}=W^{UV}_i\big(\sum_ja_{t,j,i}c_j^{KV}\big)$, rồi chiếu ra bằng $W^O_i$ (khối cột của $W^O$): đóng góp vào $u_t$ là $W^O_iW^{UV}_i\big(\sum_ja_{t,j,i}c_j^{KV}\big)$, tức gộp $W^O_iW^{UV}_i$ offline. Chỉ cần lấy tổng có trọng số của **latent**.

Hệ quả: khi sinh, attention hoạt động **trong không gian latent $d_c$** (cộng thêm phần RoPE), như một MQA với "head dim" $d_c$ mà mọi head dùng chung dữ liệu cache nhưng khác ma trận gộp. (Nhận xét của chúng ta.)

### 3.4 Vì sao RoPE xung đột, và decoupled RoPE (§2.1.3)

Nếu áp RoPE lên key $k_j^C=W^{UK}c_j^{KV}$ thì score trở thành

$$q_{t}^\top R_{t}^\top R_{j}\,W^{UK}c_j^{KV}\;=\;h_t^\top (W^Q)^\top\,R_{j-t}\,W^{UK}\,c_j^{KV}$$

(dùng tính chất bài 12: $R_t^\top R_j=R_{j-t}$). Ma trận xoay $R_{j-t}$ **phụ thuộc vị trí tương đối** nằm *giữa* $W^Q$ và $W^{UK}$; phép nhân ma trận không giao hoán nên không thể gộp $(W^Q)^\top W^{UK}$ trước thành một ma trận cố định như ở §3.3. Hậu quả: phải dựng lại key của **mọi** token tiền tố ở mỗi bước, mất lợi ích cache nhỏ ("significantly hinder the inference efficiency").

**Giải pháp: decoupled RoPE.** Thêm các thành phần riêng chỉ để mang vị trí (Eq. 14-19):

$$
\begin{aligned}
q_t^R&=\mathrm{RoPE}(W^{QR}c_t^Q)\quad(n_h\text{ head, mỗi head }d_h^R\text{ chiều}),\qquad
k_t^R=\mathrm{RoPE}(W^{KR}h_t)\quad(\textbf{một}\text{ key chung, }d_h^R\text{ chiều})\\
q_{t,i}&=[\,q_{t,i}^C;\,q_{t,i}^R\,],\qquad k_{t,i}=[\,k_{t,i}^C;\,k_t^R\,]\\
o_{t,i}&=\sum_{j\le t}\mathrm{softmax}_j\!\Bigg(\frac{q_{t,i}^\top k_{j,i}}{\sqrt{d_h+d_h^R}}\Bigg)v^C_{j,i},\qquad u_t=W^O[o_{t,1};\dots;o_{t,n_h}]
\end{aligned}
$$

Tích vô hướng của hai vector ghép bằng **tổng**: $q_{t,i}^\top k_{j,i}=\underbrace{q_{t,i}^{C\top}k^C_{j,i}}_{\text{không RoPE, hấp thụ được}}+\underbrace{q_{t,i}^{R\top}k^R_{j}}_{\text{có RoPE, cỡ }d_h^R}$. Phần nội dung vẫn hấp thụ được, phần vị trí nhỏ ($d_h^R$ chiều, key dùng chung giữa các head giống MQA). Cache thêm $k_t^R$. Tổng cache:

$$\boxed{\text{KV cache của MLA}=(d_c+d_h^R)\,l\ \text{phần tử/token}}$$

Công thức đầy đủ (Phụ lục C, Eq. 37-47): $c^Q=W^{DQ}h$; $q^C=W^{UQ}c^Q$; $q^R=\mathrm{RoPE}(W^{QR}c^Q)$; $q=[q^C;q^R]$; $c^{KV}=W^{DKV}h$; $k^C=W^{UK}c^{KV}$; $k^R=\mathrm{RoPE}(W^{KR}h)$; $k=[k^C;k^R]$; $v^C=W^{UV}c^{KV}$; attention như trên. Vector cần cache: $c^{KV}$ và $k^R$.

### 3.5 Hằng số của DeepSeek-V2 (§3.1.2)

60 layer, $d=5120$, $n_h=128$, $d_h=128$, $d_c=512$ (=4$d_h$), $d_c'=1536$, $d_h^R=64$ (=$d_h/2$). Thêm **RMSNorm** sau các latent nén và nhân hệ số tỉ lệ tại các điểm thắt cổ chai (latent, hidden của expert) để huấn luyện ổn định. YaRN mở rộng ngữ cảnh 4K → 128K chỉ áp dụng cho khóa decoupled $k_t^R$ (nơi mang RoPE).

### 3.6 Pseudo-code MLA (bản đơn giản hóa, huấn luyện / prefill)

```python
class MLA(nn.Module):
    def __init__(s, d, n_h, d_h, d_c, d_cq, d_r):
        s.W_dq  = Linear(d, d_cq);          s.W_uq = Linear(d_cq, n_h * d_h)
        s.W_qr  = Linear(d_cq, n_h * d_r)                        # q^R, RoPE
        s.W_dkv = Linear(d, d_c);           s.norm_kv = RMSNorm(d_c)
        s.W_uk  = Linear(d_c, n_h * d_h);   s.W_uv = Linear(d_c, n_h * d_h)
        s.W_kr  = Linear(d, d_r)                                   # k^R chung, RoPE
        s.W_o   = Linear(n_h * d_h, d)
    def forward(s, h, pos, cache):
        cq = s.W_dq(h)
        q  = cat([s.W_uq(cq).view(n_h, d_h), rope(s.W_qr(cq).view(n_h, d_r), pos)], -1)
        ckv = s.norm_kv(s.W_dkv(h));  kr = rope(s.W_kr(h), pos)   # CHỈ (ckv, kr) vào cache
        cache.append(ckv, kr)
        C, KR = cache.all()                                        # [S, d_c], [S, d_r]
        k = cat([s.W_uk(C).view(S, n_h, d_h), KR[:, None].expand(S, n_h, d_r)], -1)
        v = s.W_uv(C).view(S, n_h, d_h)
        o = softmax(q @ k.T / sqrt(d_h + d_r)) @ v
        return s.W_o(o.reshape(-1))
# Khi decode: gộp W_uk vào W_uq (tính q trong không gian d_c) và W_uv vào W_o (xem §3.3);
# khi đó không bao giờ tạo k, v đầy đủ từ cache.
```

### 3.7 Ví dụ số (tự tính) cho mô hình cỡ dự án

Giả định: $d=1024$, $n_h=16$, $d_h=64$, $l=24$ layer đều dùng attention này, fp16, $n=1500$, batch 1. Dùng tỉ lệ của DeepSeek: $d_c=4d_h=256$, $d_h^R=d_h/2=32$.

| Cơ chế | Công thức | Phần tử/token | Byte/token | Cache ở n=1500 |
|---|---|---|---|---|
| MHA | $2n_hd_hl$ | $2\cdot16\cdot64\cdot24=49{,}152$ | 98,304 | 147.5 MB |
| GQA, $n_g=4$ | $2n_gd_hl$ | $2\cdot4\cdot64\cdot24=12{,}288$ | 24,576 | 36.9 MB |
| MQA | $2d_hl$ | $3{,}072$ | 6,144 | 9.2 MB |
| MLA | $(d_c+d_h^R)l$ | $(256+32)\cdot24=6{,}912$ | 13,824 | 20.7 MB |

Tức MLA ≈ 7.1 lần nhỏ hơn MHA ở cấu hình này (không phải 57 lần như DeepSeek-V2, vì $d_hn_h=1024$ chỉ gấp 4 lần $d_c$ thay vì 32 lần: $128\times128=16384$ so với 512 trong DeepSeek-V2). Muốn nhỏ như MQA thì cần $d_c\approx d_h$.

Tham số K/V (tự tính): MHA $2\cdot d\cdot d_hn_h=2.10$M mỗi layer; MLA $d_cd+2d_hn_hd_c+d_h^Rd=262{,}144+524{,}288+32{,}768=0.82$M (chưa tính nén query).

## 4. So sánh KV-cache

**Bảng 1 của paper** (số phần tử mỗi token, không phụ thuộc độ chính xác lưu trữ):

| Cơ chế | KV cache/token | Khả năng (theo paper) |
|---|---|---|
| MHA | $2n_hd_hl$ | Strong |
| GQA | $2n_gd_hl$ | Moderate |
| MQA | $2d_hl$ | Weak |
| **MLA** | $(d_c+d_h^R)l\approx\tfrac92d_hl$ | **Stronger** |

Với $d_c=4d_h$, $d_h^R=d_h/2$: $(4+0.5)d_hl=\tfrac92d_hl=2\cdot2.25\,d_hl$, tức **bằng GQA với 2.25 nhóm**.

**Số cụ thể DeepSeek-V2** (tự tính từ cấu hình paper): MHA cùng kích thước sẽ cần $2\cdot128\cdot128\cdot60=1{,}966{,}080$ phần tử/token; MLA cần $(512+64)\cdot60=34{,}560$; tỉ lệ ≈ 56.9 lần (MLA ≈ 1.8% của MHA). Con số "giảm 93.3%" trong phần tóm tắt là so với **DeepSeek 67B** (mô hình dense dùng GQA), không phải so với MHA.

**Bảng 9 của paper (so MLA/MHA, MoE nhỏ và lớn):**

| | Small MoE MHA | Small MoE MLA | Large MoE MHA | Large MoE MLA |
|---|---|---|---|---|
| Tham số kích hoạt | 2.5B | 2.4B | 25.0B | 21.5B |
| Tổng tham số | 15.8B | 15.7B | 250.8B | 247.4B |
| KV cache/token (số phần tử) | 110.6K | 15.6K | 860.2K | 34.6K |

Kiểm tra: small MLA $(512+64)\cdot27=15{,}552\approx15.6$K và small MHA $2\cdot16\cdot128\cdot27=110{,}592\approx110.6$K khớp cấu hình V2-Lite (27 layer, 16 head, $d_h=128$); large MLA $34{,}560\approx34.6$K khớp V2. Paper tóm tắt: MLA chỉ cần 14% (MoE nhỏ) và 4% (MoE lớn) cache so với MHA.

**Hiệu ứng đối với suy luận (tự tính):** vì cache MLA của V2 ≈ 34.6K phần tử/token, ở fp16 tức ≈ 69 KB/token, ngữ cảnh 128K cần ≈ 8.8 GB cho một chuỗi (so với ≈ 252 GB nếu là MHA: $1{,}966{,}080\times2\text{ B}\times131{,}072$). Thực tế paper còn lượng tử hóa cache xuống trung bình 6 bit/phần tử.

## 5. DeepSeekMoE và phần còn lại (tóm tắt)

- **DeepSeekMoE** (Eq. 20-22): thay FFN bằng $N_s$ expert chia sẻ (luôn bật) cộng $N_r$ expert định tuyến (top-$K_r$); mỗi expert nhỏ ("fine-grained") để chuyên môn hóa cao hơn và các expert chia sẻ giữ kiến thức chung (giảm dư thừa). Đầu ra: $h_t'=u_t+\sum_{i}\mathrm{FFN}^{(s)}_i(u_t)+\sum_ig_{i,t}\mathrm{FFN}^{(r)}_i(u_t)$, $g_{i,t}=s_{i,t}$ nếu $s_{i,t}\in\mathrm{Topk}$, ngược lại 0; $s_{i,t}=\mathrm{Softmax}_i(u_t^\top e_i)$. V2: tất cả FFN trừ layer đầu là MoE; mỗi layer 2 expert chia sẻ + 160 expert định tuyến (hidden 1536/expert), 6 expert được kích hoạt mỗi token.
- **Device-limited routing:** mỗi token chỉ được gửi tới tối đa $M=3$ thiết bị để giới hạn chi phí truyền thông.
- **Ba loss cân bằng tải** (expert, device, communication: $\alpha_1=0.003,\alpha_2=0.05,\alpha_3=0.02$) và **token-dropping** ở mức thiết bị khi huấn luyện (không bỏ token khi đánh giá).
- **Tiền huấn luyện:** 8.1T token, tokenizer BBPE 100K, độ dài 4K, AdamW ($\beta_1=0.9,\beta_2=0.95$), mở rộng ngữ cảnh lên 128K bằng YaRN + 1000 bước ở độ dài 32K. **Căn chỉnh:** SFT 1.5M mẫu (2 epoch, lr $5\times10^{-6}$) rồi RL bằng GRPO; paper ghi nhận "alignment tax" (RL làm giảm điểm một số benchmark chuẩn như BBH).
- **DeepSeek-V2-Lite** (Phụ lục B): 15.7B tổng / 2.4B kích hoạt, 27 layer, $d=2048$, 16 head, $d_h=128$, $d_c=512$, **không nén query**, $d_h^R=64$, 2 expert chia sẻ + 64 expert định tuyến (hidden 1408), 6 kích hoạt.

## 6. Thí nghiệm và kết quả chính

**Hiệu quả (tóm tắt):** so với DeepSeek 67B: tiết kiệm 42.5% chi phí huấn luyện (172.8K vs 300.6K GPU-giờ trên mỗi nghìn tỷ token, H800), giảm 93.3% KV-cache, thông lượng sinh tối đa gấp 5.76 lần (trên một node 8×H800 vượt 50K token/s; prompt >100K token/s). Triển khai dùng FP8 cho tham số và KV-cache lượng tử hóa trung bình 6 bit.

**Ablation attention, dense 7B, cùng 1.33T token (Bảng 8):**

| Benchmark | MQA (7.1B) | GQA 8 nhóm (6.9B) | MHA (6.9B) |
|---|---|---|---|
| BBH (EM, 3-shot) | 33.2 | 35.6 | 37.0 |
| MMLU (Acc., 5-shot) | 37.9 | 41.2 | 45.2 |
| C-Eval (Acc., 5-shot) | 30.0 | 37.7 | 42.9 |
| CMMLU (Acc., 5-shot) | 34.6 | 38.4 | 43.5 |

(Paper cân bằng số tham số bằng cách chỉnh số layer.) **MLA vs MHA (Bảng 9):**

| Benchmark | Small MHA | Small MLA | Large MHA | Large MLA |
|---|---|---|---|---|
| BBH (EM) | 37.9 | 39.0 | 46.6 | 50.7 |
| MMLU (Acc.) | 48.7 | 50.0 | 57.5 | 59.0 |
| C-Eval (Acc.) | 51.6 | 50.9 | 57.9 | 59.2 |
| CMMLU (Acc.) | 52.3 | 53.4 | 60.7 | 62.5 |

Small MoE huấn luyện trên 1.33T token, Large MoE (~250B) trên 420B token. MLA hơn MHA trong 7/8 ô, ngoại lệ C-Eval ở quy mô nhỏ (50.9 so với 51.6). Lưu ý: số tham số kích hoạt của Large MLA (21.5B) và MHA (25.0B) không bằng nhau.

**Kết quả mô hình đầy đủ (Bảng 2, một phần; base model, 21B kích hoạt / 236B tổng):**

| Benchmark | DeepSeek 67B | Mixtral 8x22B | LLaMA 3 70B | DeepSeek-V2 |
|---|---|---|---|---|
| MMLU (5-shot) | 71.3 | 77.6 | 78.9 | 78.5 |
| BBH (3-shot) | 68.7 | 78.9 | 81.0 | 78.9 |
| GSM8K (8-shot) | 63.4 | 80.3 | 83.0 | 79.2 |
| MATH (4-shot) | 18.7 | 42.5 | 42.2 | 43.6 |
| HumanEval (Pass@1) | 45.1 | 53.1 | 48.2 | 48.8 |
| C-Eval (5-shot) | 66.1 | 59.6 | 67.5 | 81.7 |

**Ngữ cảnh dài:** thử nghiệm "Needle In A Haystack" cho kết quả tốt trên mọi độ dài tới 128K (Hình 4; chỉ có đồ thị).

**V2-Lite (Bảng 6, một phần):** MMLU 58.3 (so với 48.2 của DeepSeek 7B dense MHA và 45.0 của DeepSeekMoE 16B MHA), BBH 44.1; lưu ý V2-Lite huấn luyện 5.7T token so với 2T của hai mô hình kia nên không so sánh sạch kiến trúc.

## 7. Phân tích / ablation

- **MHA > GQA > MQA** ở quy mô dense 7B (Bảng 8): cắt KV head là đổi chất lượng lấy cache, đúng như bài 14, 15.
- **MLA ≥ MHA** ở hai quy mô MoE (Bảng 9) với cache nhỏ hơn rất nhiều. Paper không cắt nghĩa chắc chắn vì sao một cơ chế nén lại tốt hơn MHA; điều có thể suy luận: nén low-rank hoạt động như bộ điều chuẩn hoặc mỗi head vẫn có K,V riêng nên không mất dung lượng "đa dạng head" như MQA/GQA (đây là suy đoán của chúng ta).
- **Decoupled RoPE là cần thiết** cho cache nhỏ (lập luận bằng đại số ở §3.4); paper không báo cáo ablation số liệu cho riêng thành phần này.
- Nén query chỉ để tiết kiệm activation khi huấn luyện (V2-Lite bỏ nén query).
- Cần RMSNorm sau latent và hệ số tỉ lệ để ổn định: nhận xét thực hành của paper.

## 8. Giới hạn

**Do tác giả nêu:** hạn chế chung của LLM (kiến thức không cập nhật, có thể sinh thông tin sai, hallucination); dữ liệu chủ yếu Trung/Anh nên yếu ở ngôn ngữ khác; "alignment tax" của RL; nhấn mạnh online RL tốt hơn offline nhưng chưa so sánh sâu. Với MoE, thừa nhận phần lớn phát hiện về kiến trúc chỉ ở quy mô thử nghiệm.

**Chúng ta nhận thấy:**
- Ablation MLA vs MHA không cô lập từng thành phần (nén KV, nén query, decoupled RoPE) và số tham số hoạt động không khớp hoàn toàn giữa MLA và MHA trong Bảng 9.
- Ablation MHA/GQA/MQA ở dense 7B và MLA/MHA ở MoE là hai bộ thí nghiệm khác nhau, nên không có so sánh trực tiếp MLA với GQA cùng điều kiện.
- Lợi ích tính toán của hấp thụ ma trận chỉ rõ khi memory-bound; ở decode, attention chạy trong không gian $d_c+d_h^R$ ($=576$ ở V2) cho mỗi head, tức số phép tính attention mỗi head lớn hơn MHA ($d_h=128$) cỡ 4.5 lần (tự tính) nhưng lượng dữ liệu đọc nhỏ hơn rất nhiều; kernel cần được thiết kế riêng (paper nói MLA được tối ưu trên bản FlashAttention-2 cải tiến).
- Không có kết quả cho mô hình nhỏ cỡ dưới 1B hay cho bài toán đa phương thức.
- Khó chuyển checkpoint MHA có sẵn sang MLA (không có công thức "uptraining" như GQA); các nghiên cứu sau đã đề xuất nhưng ngoài paper này.

## 9. Liên hệ với các bài khác

- **12 (RoPE):** nền tảng để hiểu tại sao RoPE "cản" việc hấp thụ ma trận; decoupled RoPE là cách giải.
- **14 (MQA), 15 (GQA):** baseline; MLA đạt cache cỡ GQA-2.25 nhóm mà chất lượng ≥ MHA.
- **10 (RMSNorm):** dùng sau latent nén.
- **27 (Switch), 28 (Mixtral):** MoE; DeepSeekMoE thêm expert chia sẻ và expert nhỏ mịn.
- **46 (PagedAttention), 56 (H2O), 55 (StreamingLLM):** quản lý hoặc cắt bỏ cache, trực giao với việc nén theo chiều đặc trưng của MLA.
- **18-24 (linear attention, DeltaNet):** cách khác để tránh cache tăng theo $n$ (trạng thái cố định) thay vì nén theo chiều đặc trưng.
- **08 (LoRA):** cùng nguyên tắc low-rank (ma trận lớn = tích hai ma trận hẹp).

## 10. Ứng dụng cho dự án (suy luận)

- **Liên quan trung bình.** Dự án dùng decoder hybrid (GatedDeltaNet + một số lớp full attention): KV-cache chỉ tồn tại ở các lớp full attention và ở batch 1 với 1,500 token vốn đã nhỏ (xem bảng §3.7 và bài 14: chỉ vài chục MB, trong khi trọng số 0.8B fp16 ≈ 1.6 GB). Vì vậy MLA **không phải đòn bẩy chính** cho latency batch 1; nó quan trọng khi phục vụ nhiều trang cùng lúc hoặc ngữ cảnh rất dài.
- **Nếu muốn nén KV ở các lớp full attention của student:** có ba mức, từ rẻ đến đắt: (i) giảm KV head bằng mean-pooling + chưng cất (bài 15); (ii) chia sẻ/low-rank hóa $W^K,W^V$ bằng SVD rồi huấn luyện lại (tinh thần MLA, nhưng cần xử lý RoPE như §3.4); (iii) đổi hẳn sang MLA (cần huấn luyện nhiều). Với RoPE hiện có ở các lớp đó, chỉ cần để phần "RoPE" riêng (chiều quay) và phần "nội dung" (không quay) tách biệt, một bước chuẩn bị đáng làm trước.
- **Gợi ý cắt tỉa theo low-rank:** ý tưởng "K,V của mọi head nằm trong không gian con $d_c$ chiều" gợi ý đo hạng hiệu dụng của $[W^K;W^V]$ trong checkpoint thật; nếu hạng thấp rõ rệt, có thể cắt chiều head (theo cặp RoPE, bài 12) ít tổn thất. Đây là giả thuyết, chưa được paper kiểm chứng cho cắt tỉa.
- **Phần MoE:** ít liên quan trực tiếp (dự án cắt neuron MLP và lớp, không dùng expert). Điểm liên hệ nhẹ: expert chia sẻ ≈ phần MLP "luôn bật" giữ kiến thức chung; còn kinh nghiệm "cắt neuron MLP phục hồi tệ hơn cắt lớp" có thể liên quan đến việc cấu trúc hạt mịn (fine-grained) cần chuyên môn hóa (giả thuyết).
- **Distillation:** nếu cấu trúc student khác teacher ở attention (ví dụ GQA → MLA), cần khớp thêm đầu ra attention hoặc hidden state, không chỉ logits.

## 11. Thí nghiệm gợi ý

1. **Đo hạng hiệu dụng của K/V:** với từng lớp full attention của LightOnOCR-3, SVD ma trận ghép $[W^K;W^V]\in\mathbb R^{2d_hn_h\times d}$ và vẽ phần trăm năng lượng theo hạng. Hạng nào đủ giữ 90-95% năng lượng? Đối chiếu với tỉ lệ $d_c/(d_hn_h)=1/4$ của DeepSeek.
2. **Chứng minh số học của hấp thụ:** trong PyTorch, tạo $W^Q,W^{UK},W^{UV},W^O$ ngẫu nhiên; tính attention theo cách thường (dựng K,V) và theo cách hấp thụ (chỉ dùng latent); kiểm tra sai số ~1e-5. Thêm RoPE lên key và chứng kiến phép gộp thất bại, sau đó thêm nhánh decoupled và thấy lại tính đúng.
3. **Mô phỏng cache:** viết hàm tính KV bytes theo công thức §3.7 cho các cấu hình MHA/GQA/MQA/MLA của mô hình thật, đối chiếu với bộ nhớ đo được.

## 12. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| Nén KV | $c_t^{KV}=W^{DKV}h_t$, $k^C=W^{UK}c^{KV}$, $v^C=W^{UV}c^{KV}$ |
| Cache chỉ latent | $d_cl$ phần tử/token (chưa tính RoPE) |
| Hấp thụ | $(W^Q_i)^\top W^{UK}_i$ và $W^O_iW^{UV}_i$ tính trước, không dựng K,V |
| Vì sao RoPE gây xung đột | $R_{j-t}$ nằm giữa $W^Q$ và $W^{UK}$, phụ thuộc vị trí, không gộp được |
| Decoupled RoPE | $q_{t,i}=[q^C_{t,i};q^R_{t,i}]$, $k_{t,i}=[k^C_{t,i};k^R_t]$, $k^R$ chung cho mọi head |
| Cache MLA | $(d_c+d_h^R)\,l$ phần tử/token |
| V2 | $d_c=512,d_h^R=64,n_h=d_h=128,l=60$: 34,560 phần tử/token |
| So với GQA | $\tfrac92d_hl$ = GQA 2.25 nhóm |
| Kết quả | giảm 93.3% cache (vs 67B), 42.5% chi phí huấn luyện, thông lượng ×5.76 |
| Ablation | MHA>GQA>MQA (7B dense); MLA≥MHA (Bảng 9, cache 14% / 4%) |
| MoE | 2 shared + 160 routed expert, 6 kích hoạt; 236B tổng / 21B kích hoạt |

## 13. Kết luận ngắn

MLA nén K và V của mọi head vào một latent chung nhỏ, giữ nguyên số head và dùng phép gộp ma trận để suy luận trong không gian latent; RoPE được tách thành nhánh nhỏ riêng để không phá phép gộp đó. Theo paper, cache chỉ bằng ~GQA 2.25 nhóm nhưng chất lượng không thua (và ở thí nghiệm của họ còn hơn) MHA. Ở quy mô và bài toán của dự án (batch 1, mô hình nhỏ, hybrid), lợi ích trực tiếp là hạn chế, nhưng khung tư duy "low-rank + tách RoPE" hữu ích khi nén K/V trong quá trình cắt tỉa.

Nguồn: https://arxiv.org/abs/2405.04434
