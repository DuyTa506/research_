# Fast Transformer Decoding: One Write-Head is All You Need (2019)

**Tác giả:** Noam Shazeer (Google)  
**Venue / arXiv:** arXiv:1911.02150 (cs.NE, v1, 6/11/2019); paper không nêu venue hội nghị.  
**Nguồn:** https://arxiv.org/abs/1911.02150  
**Chủ đề:** Multi-Query Attention (MQA): cả h head dùng chung một bộ K và V để giảm kích thước KV-cache và băng thông bộ nhớ khi giải mã tuần tự.  
**Vị trí trong lộ trình:** Đọc sau 01 (multi-head attention) và 05/06 (LLaMA, Mistral, nơi KV-cache trở thành vấn đề thực tế). Paper này đặt nền cho phân tích "decode bị giới hạn bởi băng thông bộ nhớ" và đưa ra giải pháp cực đoan (1 KV head). Tiếp theo: 15 (GQA, điểm giữa MHA và MQA), 16 (MLA, nén KV bằng low-rank), 44-46 (FlashAttention, PagedAttention: các góc nhìn khác về bộ nhớ attention).

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Các tính toán KV-cache cho mô hình cỡ dự án đều là "tự tính" với giả định ghi rõ; chúng không có trong paper. Phần ứng dụng cho dự án là suy luận.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
4. [Phân tích KV-cache và băng thông (tự tính)](#4-phân-tích-kv-cache-và-băng-thông-tự-tính)
5. [Thí nghiệm và kết quả chính](#5-thí-nghiệm-và-kết-quả-chính)
6. [Phân tích / ablation](#6-phân-tích--ablation)
7. [Giới hạn](#7-giới-hạn)
8. [Liên hệ với các bài khác](#8-liên-hệ-với-các-bài-khác)
9. [Ứng dụng cho dự án (suy luận)](#9-ứng-dụng-cho-dự-án-suy-luận)
10. [Thí nghiệm gợi ý](#10-thí-nghiệm-gợi-ý)
11. [Tóm tắt công thức / khái niệm](#11-tóm-tắt-công-thức--khái-niệm)
12. [Kết luận ngắn](#12-kết-luận-ngắn)

---

## 1. Bối cảnh

**Huấn luyện song song, sinh chữ tuần tự.** Khi huấn luyện, ta biết cả chuỗi đích nên tính attention cho mọi vị trí cùng lúc (một matmul lớn). Khi **sinh** (autoregressive decoding), token $t+1$ phụ thuộc vào đầu ra bước $t$ nên mỗi bước chỉ xử lý **một** token mới. Để không tính lại, ta lưu **KV-cache**: các vector key và value của mọi token trước đó ở mọi layer; mỗi bước chỉ tính K, V cho token mới rồi nối vào cache.

**Vấn đề của bước tuần tự.** Mỗi bước, GPU phải đọc lại toàn bộ KV-cache (kích thước tăng theo độ dài $n$) và toàn bộ trọng số, nhưng chỉ làm ít phép tính (một token). Khái niệm cần nhớ: **arithmetic intensity** = số phép tính / số byte phải đọc từ bộ nhớ. GPU/TPU hiện đại có sức tính gấp hàng trăm lần băng thông bộ nhớ (paper: "two orders of magnitude"), nên nếu tỉ lệ này thấp thì máy **chờ dữ liệu** chứ không bận tính: gọi là **memory-bandwidth-bound**.

## 2. Ý tưởng chính

Trong multi-head attention (MHA) mỗi head có K, V riêng, nên cache có chiều "head" $h$. Multi-query attention (MQA) giữ $h$ **query** head khác nhau nhưng cho cả $h$ head **dùng chung một bộ K và một bộ V**. Cache nhỏ đi đúng $h$ lần, việc đọc cache ở mỗi bước nhẹ đi $h$ lần. Paper cho thấy decode nhanh hơn rất nhiều, chất lượng giảm chút ít. Về code: chỉ cần bỏ chữ "h" khỏi các phép einsum của K, V và các ma trận chiếu $P_k,P_v$.

## 3. Phương pháp chi tiết

### 3.1 Multi-head attention (một bước, bỏ batch)

Ký hiệu của paper: $d$ = chiều mô hình; $h$ = số head; $k,v$ = chiều key/value mỗi head; $m$ = số vị trí trong bộ nhớ (độ dài ngữ cảnh). Với đầu vào $x\in\mathbb R^d$ và $M\in\mathbb R^{m\times d}$:

$$q_h=xP_{q,h},\quad K_h=MP_{k,h},\quad V_h=MP_{v,h},\quad o_h=\mathrm{softmax}(q_hK_h^\top)V_h,\quad y=\sum_h o_hP_{o,h}$$

(paper bỏ hệ số $1/\sqrt{d_k}$ vì gộp vào $P_q$). $P_q,P_k\in\mathbb R^{h\times d\times k}$, $P_v,P_o\in\mathbb R^{h\times d\times v}$. Cache lưu $K,V$ cỡ $[b,h,m,k]$.

### 3.2 Multi-query attention

$$
\boxed{K=MP_k,\;V=MP_v\;\;(P_k\in\mathbb R^{d\times k},\,P_v\in\mathbb R^{d\times v}\text{ không có chiều }h);\quad o_h=\mathrm{softmax}(q_hK^\top)V}
$$

Mỗi head vẫn có $P_{q,h}$ và $P_{o,h}$ riêng nên các head vẫn "hỏi" khác nhau, chỉ là cùng nhìn vào một bộ khóa/giá trị. Cache cỡ $[b,m,k]$ thay cho $[b,h,m,k]$.

### 3.3 Pseudo-code (PyTorch tương đương với einsum trong paper)

```python
# x: [B, d] (một token mới), prev_K/prev_V: [B, m, dk] (MQA) hoặc [B, h, m, dk] (MHA)
q = einsum('bd,hdk->bhk', x, Pq)                 # h query head
k_new = einsum('bd,dk->bk', x, Pk)               # MQA: KHÔNG có chữ h
v_new = einsum('bd,dv->bv', x, Pv)
K = cat([prev_K, k_new[:, None]], dim=1)         # [B, m+1, dk]
V = cat([prev_V, v_new[:, None]], dim=1)
logits = einsum('bhk,bmk->bhm', q, K)            # mọi head dùng chung K
w = softmax(logits, -1)
o = einsum('bhm,bmv->bhv', w, V)
y = einsum('bhv,hdv->bd', o, Po)
```

Trong PyTorch thực tế: K,V có `num_kv_heads=1` rồi `expand`/broadcast sang $h$ head, không sao chép bộ nhớ.

### 3.4 Phân tích hiệu năng của paper (đơn giản hóa $m=n$, $k=v=d/h$, $n\le d$)

**MHA, huấn luyện / xử lý theo lô** (§2.3.1): số phép tính $\Theta(bnd^2)$; bộ nhớ truy cập $O(bnd+bhn^2+d^2)$; tỉ lệ bộ nhớ/tính $O\big(\tfrac1k+\tfrac1{bn}\big)$, rất thấp, tốt.

**MHA, giải mã tăng dần** (§2.4.1): qua $n$ lần gọi, tính $\Theta(bnd^2)$; bộ nhớ truy cập $\Theta(bn^2d+nd^2)$ (số hạng đầu do đọc lại K,V ở mỗi bước, mỗi lần cỡ $bhmk=bnd$); tỉ lệ

$$\text{memory/compute (MHA, incremental)}=\Theta\Big(\frac nd+\frac1b\Big).$$

Khi $n\approx d$ hoặc $b\approx1$ tỉ lệ ≈ 1, tức nút cổ chai băng thông. Số hạng $1/b$ giảm bằng cách tăng batch; số hạng $n/d$ do cache KV.

**MQA** (§3.1): bộ nhớ truy cập $\Theta(bnd+bn^2k+nd^2)$ (K,V nhỏ hơn $h$ lần vì $k=d/h$): tỉ lệ

$$\boxed{\text{memory/compute (MQA, incremental)}=\Theta\Big(\frac1d+\frac n{dh}+\frac1b\Big)}$$

Số hạng "xấu" $n/d$ giảm $h$ lần. Lưu ý: số hạng $1/b$ (đọc trọng số $d^2$ cho mỗi bước) không được MQA giải quyết; nó chỉ giảm khi tăng batch (chi tiết ở §4 dưới đây về hệ quả với batch 1).

## 4. Phân tích KV-cache và băng thông (tự tính)

### 4.1 Công thức kích thước KV-cache

Cho $L$ layer, $h_{kv}$ số KV head, $d_k=d_v=d_h$ chiều mỗi head, độ dài $n$, batch $b$, $s$ byte/phần tử:

$$\boxed{\text{KV bytes}=2\cdot L\cdot h_{kv}\cdot d_h\cdot n\cdot b\cdot s}$$

(hệ số 2 cho K và V). MHA: $h_{kv}=h$. MQA: $h_{kv}=1$ (thu nhỏ $h$ lần).

### 4.2 Ví dụ cỡ dự án (giả định, không phải cấu hình thật của LightOnOCR)

Giả định: $d_{model}=1024$, $L=24$ layer đều là full attention, $h=16$ head, $d_h=64$ ($h\cdot d_h=d_{model}$), fp16 ($s=2$), $n=1500$, $b=1$.

| Cấu hình | KV/token (24 layer) | KV cache ở n=1500 |
|---|---|---|
| MHA, $h_{kv}=16$ | $2\cdot24\cdot16\cdot64\cdot2=98{,}304$ B ≈ 96 KiB | $147{,}456{,}000$ B ≈ **147 MB** |
| MQA, $h_{kv}=1$ | $6{,}144$ B = 6 KiB | $9{,}216{,}000$ B ≈ **9.2 MB** |

### 4.3 Băng thông ở mỗi bước giải mã

Mỗi bước sinh, GPU đọc (i) toàn bộ trọng số và (ii) toàn bộ KV-cache hiện tại. Với mô hình 0.8B fp16 (≈1.6 GB trọng số), tại bước cuối (n=1500):

- MHA: đọc ≈ 1.6 GB + 0.147 GB ≈ 1.75 GB/bước; KV chiếm ≈ 8%.
- MQA: ≈ 1.6 GB + 0.009 GB; KV chiếm ≈ 0.6%.

Nếu băng thông GPU là $B$ GB/s thì thời gian tối thiểu mỗi bước ≈ bytes/B (cận dưới do băng thông). Ví dụ với $B=300$ GB/s (giả định một GPU ≤ 8 GB tầm trung): 1.75/300 ≈ 5.8 ms vs 1.61/300 ≈ 5.4 ms. **Kết luận của ví dụ:** ở batch 1, ngữ cảnh 1,500 và mô hình nhỏ, KV-cache chỉ là phần nhỏ, **trọng số mới là nguồn băng thông chính**; MQA mang lại cải thiện nhỏ (~7% ở ví dụ này). Lợi ích lớn của MQA xuất hiện khi $b\cdot n$ lớn (batch lớn, ngữ cảnh dài) sao cho KV-cache so được với kích thước trọng số: số hạng $n/d$ trong công thức paper tương ứng với điều này (hai tỉ lệ này khớp: $n/d=1500/1024\approx1.5$ cho MHA nếu coi mọi thứ cùng bậc, nhưng cách tính byte ở đây thực tế hơn vì dùng số tham số thật).

Ngược lại, ở batch lớn $b$ (phục vụ nhiều người dùng): KV tỉ lệ với $b$ còn trọng số thì không, nên với $b=32$: MHA KV ≈ 4.7 GB (có thể vượt 8 GB cùng với trọng số), MQA ≈ 0.29 GB. Đây là nơi MQA thực sự quyết định.

## 5. Thí nghiệm và kết quả chính

**Thiết lập (§4.1):**
- Dịch WMT 2014 En-De: Transformer encoder-decoder 6 layer, $d_{model}=1024$, $d_{ff}=4096$, $h=8$, $d_k=d_v=128$, 211M tham số, 100k bước (~20 epoch), batch 128 mẫu (input 256 token, target 256 token), TPUv3 32 core (~2 giờ mỗi mô hình).
- Mô hình MQA: thay **mọi** lớp attention (encoder self-attn, decoder self-attn, encoder-decoder attn) bằng MQA; nới $d_{ff}$ 4096 → 5440 để giữ nguyên số tham số.
- Biến thể "local": decoder self-attention chỉ nhìn vị trí hiện tại và 31 vị trí trước.
- Baseline giảm kích thước KV bằng cách giảm $h$ và $d_k,d_v$ (nới $d_{ff}$ lên 6784 để cùng tham số).
- Mô hình ngôn ngữ decoder-only trên Billion-Word benchmark: 6 layer, $d_{model}=1024$, $d_{ff}=8192$, $h=8$, $d_k=d_v=128$, 192M tham số, 136k bước.

**Bảng 1: WMT14 En-De**

| Attention | h | $d_k,d_v$ | $d_{ff}$ | ln(PPL) dev | BLEU dev | BLEU test (beam 1 / 4) |
|---|---|---|---|---|---|---|
| multi-head | 8 | 128 | 4096 | 1.424 | 26.7 | 27.7 / 28.4 |
| multi-query | 8 | 128 | 5440 | 1.439 | 26.5 | 27.5 / 28.5 |
| multi-head local | 8 | 128 | 4096 | 1.427 | 26.6 | 27.5 / 28.3 |
| multi-query local | 8 | 128 | 5440 | 1.437 | 26.5 | 27.6 / 28.2 |
| multi-head | 1 | 128 | 6784 | 1.518 | 25.8 | – |
| multi-head | 2 | 64 | 6784 | 1.480 | 26.2 | 26.8 / 27.9 |
| multi-head | 4 | 32 | 6784 | 1.488 | 26.1 | – |
| multi-head | 8 | 16 | 6784 | 1.513 | 25.8 | – |

**Bảng 3: Billion-Word LM (perplexity theo từ, dev)**

| Attention | h | $d_k,d_v$ | $d_{ff}$ | dev-PPL |
|---|---|---|---|---|
| multi-head | 8 | 128 | 8192 | 29.9 |
| multi-query | 8 | 128 | 9088 | 30.2 |
| multi-head | 1 | 128 | 9984 | 31.2 |
| multi-head | 2 | 64 | 9984 | 31.1 |
| multi-head | 4 | 32 | 9984 | 31.0 |
| multi-head | 8 | 16 | 9984 | 30.9 |

**Bảng 2: Tốc độ (TPUv2-micro giây trên mỗi token đầu ra; độ dài chuỗi 128; suy luận batch 1024 chuỗi, tức 128 mỗi core)**

| Attention | Train (enc+dec) | Inference (enc + dec) | Beam-4 (enc + dec) |
|---|---|---|---|
| multi-head | 13.2 | 1.7 + 46 | 2.0 + 203 |
| multi-query | 13.0 | 1.5 + 3.8 | 1.6 + 32 |
| multi-head local | 13.2 | 1.7 + 23 | 1.9 + 47 |
| multi-query local | 13.0 | 1.5 + 3.3 | 1.6 + 16 |

Chi tiết (§4.3): mỗi bước decoder của baseline 47 ms, của MQA 3.9 ms (batch 1024 chuỗi); encoder 222 ms vs 195 ms. Bước huấn luyện 433 ms (multi-head) vs 425 ms (MQA).

Tóm lại: decode nhanh hơn ≈ 12 lần (46 → 3.8 µs/token) với độ giảm chất lượng nhỏ (BLEU dev 26.7 → 26.5; PPL LM 29.9 → 30.2), còn hơn nhiều so với cách giảm $h$ hoặc $d_k$.

## 6. Phân tích / ablation

- **So với "giảm kích thước K,V bằng cách giảm $h$ hoặc $d_k$" (cùng tham số):** những biến thể đó kém hơn MQA rõ rệt ở cả dịch (ln PPL 1.48-1.52 so với 1.439) lẫn LM (30.9-31.2 so với 30.2). Nghĩa là giữ nhiều **query head** và chia sẻ K,V hiệu quả hơn là giảm chung số head.
- **Trực giao với local attention:** MQA + local (Bảng 2) nhanh nhất (3.3 µs ở decoder), chứng tỏ hai kỹ thuật cộng dồn.
- **Beam search được lợi nhiều nhất:** 203 → 32 µs (decoder), vì beam nhân KV-cache lên.
- Huấn luyện không chậm đi (13.0 vs 13.2).
- Để giữ cùng số tham số, MQA phải mở rộng FFN (4096 → 5440): một phần khác biệt chất lượng có thể bù bởi FFN lớn hơn. Đây là điểm thiết kế cần để ý khi so sánh công bằng.

## 7. Giới hạn

**Do tác giả nêu:** chất lượng giảm "slightly" so với baseline; paper không thêm giải thích sâu; cấu hình huấn luyện "to be added before publication" (chưa đủ chi tiết tái lập).

**Chúng ta nhận thấy:**
- Quy mô nhỏ (~200M tham số), tác vụ dịch và Billion-Word LM; chưa rõ với LLM nhiều tỷ tham số và ngữ cảnh dài. Các bài sau (GQA, bài 15) báo cáo MQA có thể gây mất ổn định huấn luyện và giảm chất lượng ở mô hình lớn (nội dung ngoài paper, xem bài 15).
- Đo tốc độ trên TPUv2 với batch 1024 chuỗi, không phải batch 1 trên GPU nhỏ; tốc độ "12 lần" không chuyển thẳng sang trường hợp của dự án (xem §4).
- Có cài đặt giả định thêm: padding đến độ dài tối đa nên mỗi bước decode tốn thời gian như nhau; cài đặt tối ưu có thể nhanh hơn ở đầu chuỗi (chú thích 1 trong paper).
- Chia sẻ K,V làm giảm dung lượng biểu diễn của attention (tất cả head đọc cùng một "bộ nhớ"): chưa có phân tích về mất mát khả năng.
- Giải mã suy đoán (bài 48-50) và batch lớn thay đổi cân bằng tính toán/băng thông, nên lợi ích MQA tương đối thay đổi theo bối cảnh.

## 8. Liên hệ với các bài khác

- **01:** multi-head attention gốc; MQA là một biến thể chia sẻ K,V.
- **15 (GQA):** nội suy giữa MHA (h KV head) và MQA (1 KV head) bằng $g$ nhóm; cho phép "uptrain" từ checkpoint MHA.
- **16 (MLA):** nén KV bằng low-rank (còn tốt hơn MQA về cache với chất lượng ≥ MHA theo DeepSeek).
- **46 (PagedAttention/vLLM):** quản lý bộ nhớ KV-cache; **56 (H2O)**, **55 (StreamingLLM)**: bỏ bớt token trong cache; **44-45 (FlashAttention):** giảm truy cập bộ nhớ khi tính attention (bài toán khác: bộ nhớ activation, không phải cache).
- **05, 06:** LLaMA 2 70B, Mistral 7B dùng GQA (kiến thức ngoài paper này).
- **18-24:** các mô hình hồi quy/linear attention loại bỏ hẳn KV-cache tăng theo $n$ (trạng thái cố định), mục tiêu giống MQA nhưng bằng cách khác.

## 9. Ứng dụng cho dự án (suy luận)

- **Hiểu mô hình:** decoder Qwen3.5-style có thể dùng GQA ở các lớp full attention (cần xác nhận từ `num_key_value_heads` trong config); và vì nhiều lớp là GatedDeltaNet (trạng thái cố định, không có KV-cache tăng theo $n$), **KV-cache của dự án vốn đã rất nhỏ**: chỉ các lớp full attention có cache. Ví dụ nếu 1/4 trong 24 lớp là full attention, cache MHA giả định ở §4.2 còn khoảng 37 MB (tự tính), và với GQA nhỏ hơn nữa.
- **Batch 1, GPU ≤ 8 GB:** như §4.3, ở batch 1 băng thông chủ yếu dành cho đọc trọng số, nên MQA/GQA đem lại ít cho latency; **cắt tham số/giảm độ rộng hay độ sâu** có tác động lớn hơn lên latency. MQA/GQA quan trọng hơn nếu triển khai phục vụ nhiều trang đồng thời (batch lớn) hoặc nếu thêm token ảnh vào ngữ cảnh (prefill dài): khi đó cache lớn hơn.
- **Pruning attention:** nếu muốn giảm head ở các lớp full attention, nên giảm **query head** nhưng giữ nguyên số KV head (hoặc ngược lại) theo hướng GQA/MQA; paper cho thấy việc ép giảm cả $h$ (nhỏ cả Q lẫn K,V) tệ hơn giữ nhiều query head với KV chung (Bảng 1, 3). Gợi ý để thử, cần kiểm chứng trên mô hình của dự án.
- **Chuyển từ MHA sang MQA bằng pruning/distillation:** có thể gộp các KV head của checkpoint bằng trung bình (xem GQA, bài 15) rồi chưng cất để phục hồi.

## 10. Thí nghiệm gợi ý

1. **Đo thật:** trên GPU đích, đo thời gian mỗi bước decode theo độ dài ngữ cảnh $n\in\{256,1024,4096,8192\}$ với $h_{kv}=h$ và $h_{kv}=1$ (mô hình tự dựng nhỏ hoặc dùng `torch` với tensor ngẫu nhiên). Vẽ thời gian vs $n$ để thấy ngưỡng ở đó KV vượt trọng số.
2. **Cache ở mô hình thật:** đọc `config.json` của LightOnOCR-3-0.8B, đếm số lớp full attention và `num_key_value_heads`, rồi tính lại bảng §4 với số thật; đối chiếu với `torch.cuda.memory_allocated` sau khi sinh 1,500 token.
3. **Mean-pool KV head:** lấy một lớp full attention, gộp các KV head thành 1 (trung bình), đo tăng loss trên 100 trang OCR trước và sau chưng cất ngắn.

## 11. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| KV-cache | lưu K,V các token trước để không tính lại |
| Kích thước KV | $2\,L\,h_{kv}\,d_h\,n\,b\,s$ byte |
| MQA | $K=MP_k$, $V=MP_v$ dùng chung cho mọi head; query vẫn $h$ head |
| Tỉ lệ bộ nhớ/tính (MHA, tăng dần) | $\Theta(n/d+1/b)$ |
| Tỉ lệ (MQA, tăng dần) | $\Theta(1/d+n/(dh)+1/b)$ |
| Giảm cache | đúng $h$ lần ($h_{kv}$: $h\to1$) |
| Ví dụ tự tính | $L=24,h=16,d_h=64,n=1500$, fp16: 147 MB → 9.2 MB |
| Kết quả | decode 46 → 3.8 µs/token; BLEU dev 26.7 → 26.5; LM PPL 29.9 → 30.2 |

## 12. Kết luận ngắn

MQA nhận ra rằng chi phí giải mã tăng dần là chi phí **đọc bộ nhớ**, đặc biệt đọc lại KV-cache, và loại bỏ chiều "head" của K,V để giảm chi phí đó đúng $h$ lần. Đánh đổi chất lượng nhỏ trên thí nghiệm của paper. Đây là gốc của GQA và MLA, nhưng với mô hình nhỏ chạy batch 1 như của dự án, trọng số mới là nguồn băng thông chính nên lợi ích trực tiếp là hạn chế.

Nguồn: https://arxiv.org/abs/1911.02150
