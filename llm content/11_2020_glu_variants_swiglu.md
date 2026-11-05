# GLU Variants Improve Transformer (2020)

**Tác giả:** Noam Shazeer (Google)  
**Venue / arXiv:** arXiv:2002.05202 (cs.LG, 12/2/2020), bản v1; paper không nêu venue hội nghị.  
**Nguồn:** https://arxiv.org/abs/2002.05202  
**Chủ đề:** Thay khối FFN của Transformer bằng biến thể Gated Linear Unit (GLU): ReGLU, GEGLU, SwiGLU, Bilinear.  
**Vị trí trong lộ trình:** Đọc sau 01 (Attention Is All You Need, nơi định nghĩa FFN) và 05 (LLaMA dùng SwiGLU), 10 (RMSNorm, cùng "bộ đôi" cải tiến của LLaMA). Paper này bổ sung: lý do FFN hiện đại có **3 ma trận** thay vì 2 và tỉ lệ hidden $\approx \tfrac{8}{3}d$ thay vì $4d$. Tiếp theo: 12 (RoPE) hoàn thiện công thức "decoder kiểu LLaMA".

> Đây là ghi chú học tập bằng tiếng Việt, không phải bản dịch nguyên văn. Paper rất ngắn (4 trang), chỉ có thực nghiệm trên T5. Các phần "Ứng dụng cho dự án" là suy luận của chúng ta, không phải kết luận của tác giả.

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

Mỗi khối Transformer gồm hai phần: **attention** (trộn thông tin giữa các token) và **FFN** (feed-forward network, xử lý từng vị trí độc lập, "position-wise"). FFN gốc là hai phép tuyến tính với một hàm kích hoạt ở giữa:

$$\mathrm{FFN}(x)=\max(0,\,xW_1+b_1)W_2+b_2.$$

- $x\in\mathbb R^{d_{model}}$ là biểu diễn ẩn của một token, $W_1\in\mathbb R^{d_{model}\times d_{ff}}$, $W_2\in\mathbb R^{d_{ff}\times d_{model}}$.
- Phần lớn tham số của một Transformer nằm ở FFN (khoảng 2/3 tham số mỗi khối với $d_{ff}=4d_{model}$), nên đây là chỗ đáng tối ưu.
- T5 bỏ bias: $\mathrm{FFN}_{ReLU}(x)=\max(xW_1,0)W_2$. Các công trình sau thay ReLU bằng GELU ($x\Phi(x)$, $\Phi$ là CDF của chuẩn tắc) hoặc Swish ($x\sigma(\beta x)$, còn gọi SiLU khi $\beta=1$).

**Gated Linear Unit (GLU)** do Dauphin et al. (2016) đề xuất cho mô hình ngôn ngữ dùng tích chập: lấy tích từng phần tử (⊗, Hadamard) của hai phép chiếu tuyến tính, một trong hai đi qua sigmoid làm "cổng" (gate). Shazeer hỏi: thay khối "tuyến tính → kích hoạt" đầu của FFN bằng GLU hoặc biến thể thì có tốt hơn không?

## 2. Ý tưởng chính

Thay vì "một đường": $\text{act}(xW_1)$, ta dùng "hai đường nhân nhau": $\text{act}(xW)\otimes(xV)$. Đường thứ nhất tính ra một **cổng** (mỗi chiều ẩn được nhân với một hệ số mềm), đường thứ hai là **nội dung**. Tích hai đường làm mỗi neuron ẩn có thể được bật/tắt phụ thuộc đầu vào theo cách giàu hơn một hàm kích hoạt đơn lẻ. Vì có thêm một ma trận, tác giả giảm $d_{ff}$ để giữ nguyên số tham số và FLOPs, rồi so sánh công bằng. Kết quả: GEGLU và SwiGLU cho perplexity tốt nhất, và nhiều tác vụ downstream tốt hơn ReLU/GELU. Paper **không giải thích vì sao** ("divine benevolence").

## 3. Phương pháp chi tiết

### 3.1 Họ GLU

Với $W,V\in\mathbb R^{d_{model}\times d_{ff}}$ (có bias $b,c$ trong định nghĩa gốc):

$$
\begin{aligned}
\mathrm{GLU}(x)&=\sigma(xW+b)\otimes(xV+c)\\
\mathrm{Bilinear}(x)&=(xW+b)\otimes(xV+c)\\
\mathrm{ReGLU}(x)&=\max(0,xW+b)\otimes(xV+c)\\
\mathrm{GEGLU}(x)&=\mathrm{GELU}(xW+b)\otimes(xV+c)\\
\mathrm{SwiGLU}(x)&=\mathrm{Swish}_\beta(xW+b)\otimes(xV+c)
\end{aligned}
$$

Bilinear là trường hợp không có hàm kích hoạt (cổng là phép tuyến tính). Mỗi biến thể chỉ khác hàm áp lên nhánh cổng.

### 3.2 FFN dùng GLU (bỏ bias, như T5)

$$
\boxed{\mathrm{FFN}_{\mathrm{SwiGLU}}(x,W,V,W_2)=\big(\mathrm{Swish}_1(xW)\otimes xV\big)W_2}
$$

Tương tự: $\mathrm{FFN}_{GEGLU}=(\mathrm{GELU}(xW)\otimes xV)W_2$, $\mathrm{FFN}_{ReGLU}=(\max(0,xW)\otimes xV)W_2$, $\mathrm{FFN}_{Bilinear}=(xW\otimes xV)W_2$, $\mathrm{FFN}_{GLU}=(\sigma(xW)\otimes xV)W_2$.

Trong đó $\mathrm{Swish}_1(z)=z\,\sigma(z)$ (= SiLU). Trong LLaMA (bài 05) gọi: $W$ = `gate_proj`, $V$ = `up_proj`, $W_2$ = `down_proj`.

### 3.3 Giữ nguyên ngân sách tham số

FFN gốc có 2 ma trận cỡ $d_{model}\times d_{ff}$: số tham số $2\,d_{model}d_{ff}$. FFN-GLU có 3 ma trận ($W$, $V$, $W_2$): $3\,d_{model}d_{ff}'$. Để bằng nhau:

$$3\,d_{ff}'=2\,d_{ff}\;\Rightarrow\;d_{ff}'=\tfrac23 d_{ff}.$$

Paper viết "giảm theo một hệ số" (bản trích PDF của chúng ta bị lỗi chữ, hiển thị "32"), nhưng số liệu cụ thể trong 3.1 cho thấy đó là **2/3**: $d_{ff}=3072\to2048$. Với $d_{ff}=4d_{model}$ thì $d_{ff}'=\tfrac83 d_{model}$; đó là lý do LLaMA dùng hidden $\approx\tfrac83 d$ (ví dụ làm tròn lên bội số của 256) — đây là kiến thức ngoài paper này, xem bài 05.

### 3.4 Pseudo-code (PyTorch)

```python
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff):           # d_ff ~ 8/3 * d_model
        super().__init__()
        self.w  = nn.Linear(d_model, d_ff, bias=False)   # gate
        self.v  = nn.Linear(d_model, d_ff, bias=False)   # content
        self.w2 = nn.Linear(d_ff, d_model, bias=False)   # down
    def forward(self, x):
        return self.w2(F.silu(self.w(x)) * self.v(x))
```

### 3.5 Ví dụ số nhỏ

$d_{model}=2$, $x=(1,-1)$; một neuron ẩn có $xW=2$, $xV=3$. Swish$_1(2)=2\sigma(2)\approx1.76$; đầu ra neuron $=1.76\times3=5.28$. Nếu $xW=-4$: Swish$_1(-4)\approx-0.07$, gần như tắt neuron dù $xV$ lớn. Cổng là hàm của $x$ nên "tắt/mở" theo từng token, khác với ReLU nơi cùng đại lượng $xW$ vừa quyết định tắt vừa là giá trị.

## 4. Thí nghiệm và kết quả chính

**Thiết lập** (đúng theo T5 của Raffel et al., 2019): encoder-decoder, mỗi bên 12 layer, $d_{model}=768$, $h=12$, $d_k=d_v=64$; FFN gốc $d_{ff}=3072$, biến thể GLU $d_{ff}=2048$ (cùng tham số và phép tính). Tiền huấn luyện 524,288 bước trên C4 với mục tiêu điền đoạn bị xóa (span-filling), batch 128 mẫu (input 512 token, output 114 token), Adafactor, lịch học inverse-sqrt, **không dropout** khi pre-train (khác T5 gốc). Sau đó fine-tune một lần trên hỗn hợp SQuAD + GLUE + SuperGLUE (131,072 bước, lr $10^{-3}$, dropout 0.1).

**Bảng 1: log-perplexity trên tập held-out (thấp hơn là tốt)**. Số trong ngoặc là độ lệch chuẩn giữa 4 lần chạy ngắn.

| FFN | 65,536 bước | 524,288 bước |
|---|---|---|
| ReLU (baseline) | 1.997 (0.005) | 1.677 |
| GELU | 1.983 (0.005) | 1.679 |
| Swish | 1.994 (0.003) | 1.683 |
| GLU | 1.982 (0.006) | 1.663 |
| Bilinear | 1.960 (0.005) | 1.648 |
| **GEGLU** | **1.942 (0.004)** | **1.633** |
| **SwiGLU** | 1.944 (0.010) | 1.636 |
| ReGLU | 1.953 (0.003) | 1.645 |

**Điểm trung bình downstream (dev):**

| FFN | GLUE avg | SuperGLUE avg | SQuAD EM / F1 |
|---|---|---|---|
| ReLU | 83.80 | 72.76 | 83.18 / 90.87 |
| GELU | 83.86 | 72.98 | 83.09 / 90.79 |
| Swish | 83.60 | 72.40 | 83.25 / 90.76 |
| GLU | 84.20 | 73.95 | 82.88 / 90.69 |
| GEGLU | 84.12 | 73.96 | 83.55 / 91.12 |
| Bilinear | 83.79 | 73.81 | 83.82 / 91.06 |
| SwiGLU | 84.36 | **74.56** | 83.42 / 91.03 |
| ReGLU | **84.67** | 73.66 | 83.53 / 91.18 |
| T5 gốc (Raffel) | 83.28 | 71.36 | 80.88 / 88.81 |

Độ lệch chuẩn giữa các lần chạy theo Raffel et al.: GLUE avg 0.235, SuperGLUE avg 0.416, SQuAD EM 0.343 / F1 0.226.

## 5. Phân tích / ablation

- **Cổng giúp**: mọi biến thể có nhân tích (GLU, Bilinear, ReGLU, GEGLU, SwiGLU) đều có log-perplexity thấp hơn cả ba baseline không cổng (ReLU, GELU, Swish) ở 524k bước. Khoảng cách tốt nhất: $1.677\to1.633$ (GEGLU) khoảng 0.044, lớn hơn nhiều so với độ lệch giữa các lần chạy (~0.005 ở 65k bước).
- **Hàm kích hoạt của cổng ít quan trọng hơn bản thân việc có cổng**: ngay cả Bilinear (không phi tuyến) đạt 1.648, tốt hơn ReLU/GELU/Swish. Đó là gợi ý của chúng ta từ bảng; tác giả không phân tích.
- GEGLU và SwiGLU tốt nhất theo perplexity; trên downstream thì "nhiễu" (các biến thể thắng ở các tác vụ khác nhau). Tác giả nói "the results are noisy" nhưng GLU-variants thắng ở phần lớn tác vụ.
- Việc T5 gốc kém hơn baseline ReLU của tác giả (dù cùng mô hình) được giải thích bằng dropout khi pre-train (giả thuyết của tác giả, không kiểm chứng riêng).

## 6. Giới hạn

**Do tác giả nêu:** không có lời giải thích cơ chế; kết quả downstream nhiễu; mỗi cấu hình fine-tune một lần.

**Chúng ta nhận thấy:**
- Chỉ một quy mô (T5-base, ~220M tham số cỡ) và một họ mô hình (encoder-decoder, span-filling). Không có thí nghiệm decoder-only LM hay quy mô lớn trong paper; sự phổ biến sau này (PaLM, LLaMA) là bằng chứng bên ngoài.
- Chỉ so sánh ở cùng số tham số; không báo cáo tốc độ thực tế (3 matmul nhỏ hơn so với 2 matmul lớn có thể khác nhau về hiệu quả phần cứng). Paper chỉ nói "no apparent computational drawbacks".
- Không có số liệu về độ ổn định huấn luyện hay khác biệt khi lượng tử hóa.

## 7. Liên hệ với các bài khác

- **01 (Attention Is All You Need):** FFN gốc được thay thế ở đây.
- **05 (LLaMA), 06 (Mistral 7B):** dùng SwiGLU với hidden $\approx\tfrac83 d$; cùng RMSNorm (10) và RoPE (12).
- **27/28 (Switch, Mixtral):** mỗi "expert" của MoE chính là một FFN, trong Mixtral là SwiGLU.
- **10 (RMSNorm), 12 (RoPE):** ba thay đổi kiến trúc kinh điển của họ LLaMA.
- **24 (Gated DeltaNet):** cũng có ý tưởng "cổng nhân" nhưng ở mức attention/trạng thái, không phải FFN.

## 8. Ứng dụng cho dự án (suy luận)

- **Hiểu kiến trúc:** decoder kiểu Qwen3.5 của LightOnOCR-3 gần như chắc chắn dùng MLP SwiGLU (`gate_proj`, `up_proj`, `down_proj`). Hãy kiểm tra trong `config.json` (`intermediate_size`, `hidden_act=silu`). Đây là suy luận từ họ kiến trúc, cần xác nhận trên checkpoint.
- **Cắt neuron MLP (pruning):** vì mỗi neuron ẩn $i$ gắn với ba vector: cột $i$ của $W$ (gate), cột $i$ của $V$ (up), hàng $i$ của $W_2$ (down), nên cắt neuron ẩn phải cắt **cả ba cùng lúc** để giữ nhất quán. Neuron "chết" khi gate $\approx0$ cho hầu hết token; tích $\text{act}(xW)_i(xV)_i$ cho một thước đo tầm quan trọng tự nhiên hơn là chỉ nhìn $W$ hay $V$ riêng lẻ. Điều này cũng giúp lý giải (giả thuyết) vì sao cắt neuron MLP khó hồi phục hơn cắt layer: tích hai đường là phi tuyến bậc hai, nên phần dư $D_{res}$ sau khi cắt một neuron khó xấp xỉ tuyến tính hơn so với MLP kiểu ReLU một đường. Cần đo thực nghiệm.
- **Tỉ lệ tham số:** FFN chiếm phần lớn tham số của decoder; khi cắt xuống ~0.3B/0.1B, chọn $d_{ff}$ (bội số của 64 hoặc 128 cho GPU) quan trọng hơn là tinh chỉnh số head.
- **Batch-1 latency:** ở batch 1 phép tính bị giới hạn bởi băng thông bộ nhớ (đọc trọng số). 3 ma trận cỡ $d\times\tfrac83d$ có cùng số byte với 2 ma trận cỡ $d\times4d$, nên SwiGLU không làm chậm thêm khi giải mã; có thể gộp `gate_proj` và `up_proj` thành một matmul để giảm số lần khởi chạy kernel.

## 9. Thí nghiệm gợi ý

1. **Đo độ thưa của cổng:** trên ~200 trang OCR, ghi lại $|\mathrm{Swish}(xW)_i|$ cho từng neuron của từng layer; vẽ tỉ lệ neuron có hoạt động trung bình thấp. Dùng để chọn layer nào cắt MLP được nhiều nhất.
2. **Điểm tầm quan trọng:** so sánh ba tiêu chí xếp hạng neuron (chuẩn của $W$ cột, chuẩn $V$ cột, $\mathbb E|{\rm act}\otimes xV|\cdot\|W_2\text{ hàng}\|$) về loss sau khi cắt 25% neuron và chưa distill.
3. **Ablation nhỏ:** huấn luyện từ đầu một decoder ~20M tham số với SwiGLU vs ReLU² vs GELU cùng ngân sách tham số trên một tập nhỏ, để cảm nhận độ chênh (nhớ nhiễu seed ~ như bảng 1).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| FFN gốc | $\max(0,xW_1)W_2$ (không bias) |
| GLU | $\sigma(xW)\otimes xV$ |
| SwiGLU | $\mathrm{Swish}_1(xW)\otimes xV$, $\mathrm{Swish}_1(z)=z\sigma(z)$ |
| GEGLU / ReGLU / Bilinear | cổng bằng GELU / ReLU / không hàm |
| FFN-SwiGLU | $(\mathrm{Swish}_1(xW)\otimes xV)W_2$ |
| Giữ ngân sách | $d_{ff}'=\tfrac23d_{ff}$ (3072 → 2048) |
| Kết quả chính | log-ppl 1.677 (ReLU) → 1.636 (SwiGLU), 1.633 (GEGLU) |

## 11. Kết luận ngắn

Một thay đổi nhỏ, rẻ, không tốn thêm tham số: nhân hai đường chiếu trong FFN thay vì dùng một đường có hàm kích hoạt. Trên T5-base nó giảm perplexity đáng kể và cải thiện nhiều tác vụ downstream, và về sau trở thành mặc định của các LLM (LLaMA, Mistral, Qwen). Paper chỉ cung cấp bằng chứng thực nghiệm, không có lý thuyết.

Nguồn: https://arxiv.org/abs/2002.05202
