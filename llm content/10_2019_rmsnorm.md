# Root Mean Square Layer Normalization (2019)

**Tác giả:** Biao Zhang (University of Edinburgh), Rico Sennrich (University of Edinburgh / University of Zurich).  
**Venue / arXiv:** arXiv:1910.07467 (v1, 16/10/2019); NeurIPS 2019 (33rd Conference on Neural Information Processing Systems, Vancouver).  
**Nguồn:** https://arxiv.org/abs/1910.07467  
**Chủ đề:** Chuẩn hoá lớp rẻ hơn: bỏ re-centering, giữ re-scaling (RMSNorm), phiên bản partial (pRMSNorm).  
**Vị trí trong lộ trình:** Cần biết LayerNorm và khối Pre-LN từ 09 (On Layer Normalization in the Transformer). Bài này thay thế phép chuẩn hoá bên trong khối đó bằng một công thức ngắn hơn. RMSNorm đã trở thành chuẩn trong các LLM hiện đại như LLaMA (05) (hiểu biết ngoài paper). Tiếp theo: SwiGLU (11), RoPE (12).

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Các ứng dụng cho dự án (pruning/distillation OCR) là suy luận của chúng ta, không phải kết luận của paper. Các con số "tự tính" do chúng ta tính thêm.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
4. [Phân tích chi phí](#4-phân-tích-chi-phí)
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

**Chuẩn hoá (normalization)** là kỹ thuật co giãn các kích hoạt trong mạng để giữ thang giá trị ổn định, giúp huấn luyện nhanh và ổn định hơn. Hai họ chính: **BatchNorm** (thống kê tính trên cả batch; không hợp chuỗi có độ dài thay đổi trong RNN) và **LayerNorm** (thống kê tính trên các neuron của *một* mẫu, độc lập giữa các mẫu).

**LayerNorm (Ba et al.).** Với tổng có trọng số đầu vào $a\in\mathbb R^n$ (đây là "summed inputs", tức $a_i=\sum_j w_{ij}x_j$):

$$\bar a_i=\frac{a_i-\mu}{\sigma}\,g_i,\quad y_i=f(\bar a_i+b_i),\quad \mu=\frac1n\sum_i a_i,\ \ \sigma=\sqrt{\tfrac1n\sum_i(a_i-\mu)^2}.$$

$g$ là gain (khởi tạo 1), $b$ là bias, $f$ là hàm kích hoạt.

**Vấn đề của paper.** LayerNorm hiệu quả về số bước huấn luyện nhưng **tốn thêm tính toán cho mỗi bước**, nhất là với RNN (có nhiều phép chuẩn hoá tuần tự theo độ dài chuỗi). Figure 1 của paper (RNNSearch dựa GRU): khi loss baseline (không chuẩn hoá) đạt 7.0, LayerNorm đạt 5.4 sau cùng số bước, nhưng chỉ 5.9 sau cùng thời gian chạy, vì mỗi bước chậm hơn.

**Hai tính chất bất biến của LayerNorm:** *re-centering invariance* (không đổi khi dịch đầu vào/trọng số một lượng) và *re-scaling invariance* (không đổi khi nhân đầu vào/trọng số với hằng số). Cộng đồng thường cho rằng cả hai đều giúp ổn định.

## 2. Ý tưởng chính

Giả thuyết của paper: **chỉ re-scaling invariance mới là lý do LayerNorm thành công, re-centering (trừ trung bình) là thừa.** Do đó bỏ bước trừ $\mu$ và chỉ chia cho **căn bậc hai của trung bình bình phương (RMS)**. Công thức ngắn hơn, ít phép tính hơn (không cần tính trung bình rồi tính phương sai quanh trung bình), thường chạy nhanh hơn 7%–64% tuỳ mô hình/framework, chất lượng tương đương. Ngoài ra, nếu các neuron cùng phân phối thì có thể ước lượng RMS chỉ từ một phần nhỏ (ví dụ 6.25%) các neuron (**pRMSNorm**).

## 3. Phương pháp chi tiết

### 3.1 Công thức

$$\boxed{\;\bar a_i=\frac{a_i}{\mathrm{RMS}(a)}\,g_i,\qquad \mathrm{RMS}(a)=\sqrt{\frac1n\sum_{i=1}^{n}a_i^2}\;}\tag{Eq. 4}$$

- Khi $\mu=0$ thì RMSNorm trùng khít LayerNorm (vì khi đó $\sigma=\mathrm{RMS}$).
- RMS bằng chuẩn Euclid chia $\sqrt n$: $\mathrm{RMS}(a)=\|a\|_2/\sqrt n$, nghĩa là ép vector lên mặt cầu bán kính $\sqrt n$. Paper thử dùng **L2-norm thuần** (không chia $\sqrt n$) thì thấy **không cải thiện** (Table 2); giả thuyết của tác giả: co giãn bán kính theo kích thước vector giúp chuẩn hoá bền vững qua các vector khác chiều.
- Dạng tổng quát (Eq. 5): $y=f\big(\tfrac{Wx}{\mathrm{RMS}(a)}\odot g+b\big)$.

**Giải nghĩa trực giác:** LayerNorm = "đưa vector về tâm 0 rồi đưa độ dài về chuẩn". RMSNorm = "chỉ đưa độ dài về chuẩn, giữ nguyên tâm".

### 3.2 Bất biến (Table 1)

Do tính tuyến tính $\mathrm{RMS}(\alpha x)=\alpha\,\mathrm{RMS}(x)$ (Eq. 6), nhân ma trận trọng số $W'=\delta W$ cho cùng đầu ra:

$$y'=f\Big(\frac{\delta Wx}{\delta\,\mathrm{RMS}(a)}\odot g+b\Big)=y.$$

| Phương pháp | Ma trận W: re-scale | Ma trận W: re-center | Vector trọng số: re-scale | Dataset: re-scale | Dataset: re-center | Một mẫu: re-scale |
|---|---|---|---|---|---|---|
| BatchNorm | có | không | có | có | có | không |
| WeightNorm | có | không | có | không | không | không |
| LayerNorm | có | có | không | có | không | có |
| RMSNorm | có | không | không | có | không | có |
| pRMSNorm | có | không | không | có | không | có |

(Trong bản trích xuất, "có/không" được ký hiệu bằng hai ký tự bị mã hoá lại; chúng tôi đọc chúng theo ngữ cảnh bài viết, vì vậy nên đối chiếu bản PDF gốc.) Điểm chính: RMSNorm mất bất biến re-centering so với LayerNorm nhưng giữ re-scaling.

### 3.3 Gradient (Mục 4.2)

Gọi $v$ là toàn bộ biểu thức bên trong $f(\cdot)$. Gradient theo tham số vector:

$$\frac{\partial L}{\partial b}=\frac{\partial L}{\partial v},\qquad \frac{\partial L}{\partial g}=\frac{\partial L}{\partial v}\odot\frac{Wx}{\mathrm{RMS}(a)}.$$

Cả hai bất biến với phép co giãn $x$ và $W$; gradient của $g$ tỉ lệ với đầu vào *đã chuẩn hoá* nên độ lớn $g$ ổn định. Gradient theo $W$ phức tạp hơn (Eq. 9):

$$\frac{\partial L}{\partial W}=\sum_{i=1}^n x^\top\otimes\Big(\frac{\partial L}{\partial v}\Big)_i\,\mathrm{diag}(g)\times R,\qquad R=\frac{1}{\mathrm{RMS}(a)}\Big(I-\frac{(Wx)(Wx)^\top}{n\,\mathrm{RMS}(a)^2}\Big).$$

Khi co giãn $x'=\delta x$ hoặc $W'=\delta W$ thì $R'=\frac1\delta R$ (Eq. 10). Hệ quả (theo paper): gradient theo $W$ bất biến với co giãn *đầu vào*, và **tương quan âm với co giãn trọng số**: trọng số càng lớn thì gradient càng nhỏ. Tác giả diễn giải đây là một **bộ điều chỉnh learning rate ngầm**, ngăn chuẩn của ma trận trọng số phình to.

### 3.4 pRMSNorm

Giả sử các neuron trong một lớp cùng phân phối (i.i.d.), ước lượng RMS từ $k=\lceil n\cdot p\rceil$ phần tử đầu tiên:

$$\overline{\mathrm{RMS}}(a)=\sqrt{\frac1k\sum_{i=1}^{k}a_i^2}.$$

Tính tuyến tính của RMS vẫn đúng nên bất biến giữ nguyên (Table 1). Tuy nhiên đây là ước lượng chệch, và tác giả quan sát **gradient có xu hướng bùng nổ khi $k$ nhỏ**; thực tế $p=6.25\%$ vẫn hội tụ tốt.

### 3.5 Pseudo-code

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        self.g = nn.Parameter(torch.ones(d))      # chỉ có gain, KHÔNG có bias
        self.eps = eps
    def forward(self, a):                          # a: (..., d)
        rms = a.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return a / rms * self.g

class LayerNorm_ref(nn.Module):                    # để so sánh
    def forward(self, a):
        mu  = a.mean(-1, keepdim=True)
        var = (a - mu).pow(2).mean(-1, keepdim=True)
        return (a - mu) / (var + eps).sqrt() * self.g + self.b
```

(Hằng $\epsilon$ để tránh chia 0 là chi tiết triển khai; trong bản văn bản đã đọc, công thức Eq. 4 không có $\epsilon$.)

### 3.6 Ví dụ số nhỏ (tự tính)

Với $a=[3,\,1,\,-1,\,5]$, $n=4$, $g=1$:
- LayerNorm: $\mu=2$, độ lệch $[1,-1,-3,3]$, $\sigma^2=(1+1+9+9)/4=5$, $\sigma\approx2.236$, đầu ra $\approx[0.447,-0.447,-1.342,1.342]$ (trung bình 0).
- RMSNorm: $\mathrm{RMS}=\sqrt{(9+1+1+25)/4}=\sqrt9=3$, đầu ra $=[1,\,0.333,\,-0.333,\,1.667]$ (trung bình $\neq0$).
Hai kết quả khác nhau vì $\mu\ne0$; nếu $a$ có trung bình 0 thì hai công thức trùng nhau.

## 4. Phân tích chi phí

Đối với một vector $d$ chiều (mỗi token, mỗi vị trí chuẩn hoá):

| | LayerNorm | RMSNorm |
|---|---|---|
| Phép rút gọn (reduction) | 2 (trung bình, phương sai quanh trung bình) | 1 (trung bình của bình phương) |
| Phép elementwise | trừ $\mu$, chia $\sigma$, nhân $\gamma$, cộng $\beta$ | chia RMS, nhân $g$ |
| Tham số | $2d$ ($\gamma,\beta$) | $d$ ($g$) |
| FLOPs (xấp xỉ, tự tính) | ~$5d$–$7d$ | ~$3d$–$4d$ |

Cả hai đều có độ phức tạp $O(d)$ trên mỗi token, **FLOPs của chuẩn hoá nhỏ so với các phép nhân ma trận** ($O(d^2)$ hoặc $O(d\,d_{ff})$ mỗi token mỗi lớp). Lợi ích thực tế đến từ việc **ít lượt đọc/ghi bộ nhớ và ít kernel hơn**, nhất là với phần cứng bị giới hạn bởi băng thông bộ nhớ hoặc độ trễ khởi chạy kernel.

**Ví dụ số theo cỡ mô hình của bạn (tự tính):** $d=1024$, 24 lớp decoder, mỗi lớp 2 phép chuẩn hoá (trước attention/mixer và trước MLP) + 1 chuẩn hoá cuối = 49 phép.
- Tham số chuẩn hoá: RMSNorm $49\times1024\approx50$k so với LayerNorm $\approx100$k, đều xấp xỉ 0.006–0.012% của 0.8B, không đáng kể.
- FLOPs: $49\times\sim4d\approx0.2$ MFLOPs/token cho RMSNorm; đối chiếu với ~1.6 GFLOPs/token cho 0.8B tham số (xấp xỉ $2\times$ số tham số), tức dưới 0.02%. Do đó *tiết kiệm FLOPs là không đáng kể*; cái có thể đáng kể ở batch-1 là **số lượt khởi chạy kernel/băng thông**, và đó là nơi chuẩn hoá phải được gộp (fuse) vào kernel khác. Paper đo tăng tốc 7%–64% là trên *thời gian huấn luyện đầy đủ* của các mô hình nhỏ/RNN/không fuse, không phải suy luận LLM batch-1.

## 5. Thí nghiệm và kết quả chính

Thực hiện trên TensorFlow, PyTorch, Theano; phần lớn thời gian đo trên một GPU TITAN X (Pascal), trung bình 3 lần chạy. Các mô hình: RNNSearch (GRU, dịch máy), Transformer, attentive reader (CNN/Daily Mail), order-embedding (ảnh–chú thích, MS COCO), ConvPool-CNN-C (CIFAR-10).

**Dịch máy WMT14 En-De, RNNSearch (TensorFlow/Nematus, Table 2, SacreBLEU):**

| Mô hình | Test14 | Test17 | Thời gian / 1k bước |
|---|---|---|---|
| Baseline | 21.7 | 23.4 | 399 ± 3.40 s |
| LayerNorm | 22.6 | 23.6 | 665 ± 32.5 s |
| L2-Norm | 20.7 | 22.0 | 482 ± 19.7 s |
| RMSNorm | 22.4 | 23.7 | 501 ± 11.8 s (nhanh hơn 24.7%) |
| pRMSNorm | 22.6 | 23.1 | 493 ± 10.7 s (nhanh hơn 25.9%) |

LayerNorm và RMSNorm đều giảm ~50% số bước tới hội tụ so với baseline; LayerNorm chậm hơn baseline khoảng 67% trong TensorFlow RNN.

**RNNSearch trong Theano và PyTorch (Table 3):** RMSNorm nhanh hơn LayerNorm 34.0% (Theano: 652 s vs 988 s) và 11.0% (PyTorch: 763 s vs 857 s) với BLEU tương đương (Theano: Test14 22.5 vs 22.3). pRMSNorm đôi khi chậm hơn RMSNorm, tác giả quy cho cài đặt slice tensor chưa tối ưu.

**Transformer (base, 300k bước, Table 4):**

| Mô hình | Test14 | Test17 | Thời gian / 1k bước |
|---|---|---|---|
| Baseline (không chuẩn hoá) | không train được (BLEU 0) | không train được | 210 ± 0.23 s |
| LayerNorm | 26.6 | 27.7 | 248 ± 1.31 s |
| RMSNorm | 26.8 | 27.7 | 231 ± 0.04 s (nhanh hơn 6.9%) |
| pRMSNorm | 26.5 | 27.8 | 225 ± 1.63 s (nhanh hơn 9.3%) |

Chi phí chuẩn hoá tương đối thấp hơn ở Transformer vì ít phép chuẩn hoá tuần tự hơn RNN.

**Các tác vụ khác (chọn lọc):**
- Attentive reader (CNN/Daily Mail): thời gian / 0.1k bước: LayerNorm 392 s, RMSNorm 333 s (nhanh hơn 15.1%), pRMSNorm 330 s (15.8%); Baseline 315 s. LayerNorm và RMSNorm hội tụ nhanh hơn BatchNorm và đạt tỉ lệ lỗi valid thấp hơn.
- Order-embedding (MS COCO): thời gian / 0.1k bước: Baseline 2.11 s, LayerNorm 12.02 s, RMSNorm 7.12 s (nhanh hơn 40.8%), pRMSNorm 4.34 s (63.9%). Chất lượng recall tương đương, RMSNorm hơi nhỉnh hơn LayerNorm trên một số chỉ số.
- CIFAR-10 (ConvPool-CNN-C, Table 10): test error: Baseline 8.96%, BatchNorm 8.25%, WeightNorm 8.28%, LayerNorm 10.49%, RMSNorm 8.83% (31 s/epoch, nhanh hơn 20.5%), pRMSNorm 10.37%. LayerNorm tệ hơn baseline trên test (không tổng quát hoá), RMSNorm tốt hơn.

## 6. Phân tích / ablation

- **Re-centering có cần không?** Kết quả đa tác vụ cho thấy RMSNorm tương đương LayerNorm, ủng hộ giả thuyết "re-scaling là cốt lõi". (Không phải chứng minh; đây là bằng chứng thực nghiệm trên các mô hình trong paper.)
- **L2-Norm thuần thất bại** (Table 2: 20.7/22.0 BLEU, thấp hơn cả baseline 21.7/23.4), nên hệ số $1/\sqrt n$ trong RMS không chỉ là chuyện quy ước.
- **Thống kê ẩn (Table 5):** ở RNNSearch, độ lệch chuẩn của trạng thái ẩn được cả LayerNorm và RMSNorm ổn định. Trung bình của RMSNorm không được chuẩn hoá nhưng *thực tế ổn định hơn baseline* (ví dụ trung bình tổng thể: baseline −1.60, LayerNorm −0.51, RMSNorm −0.73).
- **Độ bền với khởi tạo (Figure 4):** dịch chuyển tâm khởi tạo trọng số lên 0.2: LayerNorm rất bất ổn, RMSNorm bền hơn (cả hai đều tệ hơn khởi tạo gốc).
- **Tỉ lệ $p$ của pRMSNorm (Figure 3):** trên RNNSearch, BLEU ít bị ảnh hưởng bởi $p$; các thí nghiệm sau dùng $p=6.25\%$. Tuy nhiên ở CIFAR-10 pRMSNorm kém (10.37% lỗi) và tác giả nêu gradient có thể bùng nổ với $k$ nhỏ.

## 7. Giới hạn

**Do tác giả nêu:**
- Tăng tốc thực tế phụ thuộc framework, phần cứng, kiến trúc và tỉ lệ chi phí của các phần khác; dải 7%–64% rộng.
- pRMSNorm không cho tăng tốc nhất quán trong thực tế (cần mã mức thấp tốt hơn).
- RMS là ước lượng chệch khi chỉ lấy một phần; gradient có thể bất ổn khi $p$ quá nhỏ.
- Cần phân tích thêm vì sao RMSNorm thành công; chưa khám phá chuẩn khác ($\ell_1$...).

**Do chúng ta nhận xét:**
- Các mô hình thử nghiệm đều nhỏ-vừa và cũ (RNN GRU/LSTM, Transformer base cho dịch máy, CNN trên CIFAR); **chưa có LLM, chưa có tiền huấn luyện quy mô lớn**. Việc RMSNorm thành chuẩn trong LLM đến từ các công trình sau (ngoài paper này).
- Tăng tốc đo trên *huấn luyện* và trên cài đặt không fuse; không đo suy luận batch-1 hay kernel đã tối ưu, nơi khác biệt LayerNorm/RMSNorm có thể rất nhỏ.
- Chênh lệch BLEU/recall giữa RMSNorm và LayerNorm thường ≤ 0.3–0.5, nằm trong khoảng nhiễu seed thông thường, không có kiểm định thống kê.
- Giả thuyết "giá trị nằm ở re-scaling" được ủng hộ gián tiếp; nó không loại trừ rằng re-centering có ích ở các kiến trúc/độ sâu khác.

## 8. Liên hệ với các bài khác

- **09 Pre-LN Transformer:** RMSNorm là "chuẩn hoá" đặt vào vị trí Pre-LN: $x\leftarrow x+F(\mathrm{RMSNorm}(x))$.
- **05 LLaMA, 06 Mistral:** dùng RMSNorm trước mỗi sub-layer (hiểu biết ngoài paper; xác nhận trong từng bài).
- **11 SwiGLU:** thành phần MLP đi kèm trong cùng họ khối "RMSNorm + SwiGLU + RoPE".
- **44–45 FlashAttention:** fuse tốt phép attention; còn RMSNorm thường được fuse với phép cộng residual trong các kernel suy luận (hiểu biết ngoài paper).
- **18–24 (linear attention, DeltaNet, Gated DeltaNet):** các lớp này thường có thêm chuẩn hoá riêng (ví dụ chuẩn hoá đầu ra per-head); loại chuẩn hoá cụ thể xem trong từng bài.

## 9. Ứng dụng cho dự án (suy luận)

1. **Nếu mô hình của bạn đã dùng RMSNorm (phong cách Qwen3.5/LLaMA, đây là giả định cần kiểm tra trong config), bài này chủ yếu giúp bạn hiểu cái đang chạy**, không có gì phải đổi. Điều đáng nhớ: RMSNorm chỉ có gain $g$, **không có bias, không trừ trung bình**.
2. **Cắt chiều ẩn (hidden size) và RMS:** RMS tính **trên toàn bộ chiều $d$**. Nếu bạn cắt bớt chiều của dòng residual (thu nhỏ $d_{model}$), RMS của vector thay đổi (vì trung bình trên ít chiều hơn, và các chiều bị cắt có thể mang năng lượng lớn), làm thang đầu ra của norm lệch đi. Cắt kênh bất thường có "outlier" lớn trong residual stream sẽ thay đổi RMS rất nhiều; đây là lý do cần **hiệu chỉnh lại gain $g$** sau cắt. Suy luận của chúng ta, paper không bàn.
3. **Liên hệ $D_{lin}$:** RMSNorm là phép *chia vô hướng theo từng token* (không tuyến tính theo toàn bộ đầu vào do mẫu số phụ thuộc $x$, nhưng đồng bậc 0 với co giãn). Khi tìm ánh xạ tuyến tính tối ưu tại điểm cắt, cần chú ý norm phía sau sẽ chuẩn hoá lại thang; sai lệch về *độ lớn* tổng thể của vector được norm "tự sửa", còn sai lệch về *hướng* thì không. Điều này gợi ý chỉ số đánh giá nên dùng độ tương đồng cosine thay vì sai số tuyệt đối (giả thuyết).
4. **Suy luận batch-1 trên GPU nhỏ:** chi phí norm tính bằng FLOPs là không đáng kể (Mục 4); cái quan trọng là fuse norm + residual để giảm số kernel. Không nên kỳ vọng tăng tốc đáng kể từ việc thay LayerNorm bằng RMSNorm ở mô hình đã dùng sẵn RMSNorm; nếu mô hình ViT 12 khối của bạn dùng LayerNorm, đổi sang RMSNorm sẽ cần huấn luyện lại/chưng cất, mà lợi ích nhỏ (xem Mục 4).
5. **pRMSNorm không phải lựa chọn thực tế** cho dự án: tác giả không thấy tăng tốc nhất quán, và gradient bất ổn khi lấy mẫu ít.

## 10. Thí nghiệm gợi ý

1. **Quét tác động cắt chiều lên RMS:** với mô hình gốc, cắt một số kênh residual (theo độ quan trọng) và vẽ tỉ số $\mathrm{RMS}_{sau}/\mathrm{RMS}_{trước}$ tại mỗi norm trên vài trăm trang; kiểm tra xem tỉ số này ổn định (có thể bù bằng một hằng số) hay phụ thuộc token (khó bù).
2. **Chỉ tinh chỉnh gain $g$ của các RMSNorm sau khi cắt:** huấn luyện chỉ ~50k tham số (Mục 4) trong vài trăm bước, đo CER; đây là baseline rẻ nhất để thấy phần sai lệch do "thang".
3. **Đo latency thực của norm trên GPU mục tiêu:** profile một lượt decode batch-1 với norm không fuse và fuse với residual; ghi tỉ lệ thời gian dành cho norm trong tổng thời gian/token.

## 11. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| LayerNorm | $\bar a_i=\frac{a_i-\mu}{\sigma}g_i$ (+ bias) |
| RMSNorm | $\bar a_i=\frac{a_i}{\mathrm{RMS}(a)}g_i$, $\mathrm{RMS}(a)=\sqrt{\frac1n\sum a_i^2}$ |
| Liên hệ | $\mu=0\Rightarrow$ RMSNorm = LayerNorm |
| RMS vs L2 | $\mathrm{RMS}=\|a\|_2/\sqrt n$; L2-Norm thuần không hiệu quả (Table 2) |
| Bất biến | RMSNorm: re-scaling (đầu vào, ma trận W, dataset) có; re-centering không |
| Gradient $W$ | $R=\frac{1}{\mathrm{RMS}}(I-\frac{(Wx)(Wx)^\top}{n\mathrm{RMS}^2})$; $R'=\frac1\delta R$ khi co giãn → learning rate ngầm |
| pRMSNorm | RMS từ $k=\lceil np\rceil$ phần tử đầu; $p=6.25\%$ |
| Tham số | RMSNorm: $d$; LayerNorm: $2d$ |
| Kết quả chốt | Tăng tốc 7%–64% với chất lượng tương đương; Transformer base: 6.9% (RMSNorm), 9.3% (pRMSNorm) |

## 12. Kết luận ngắn

RMSNorm giả định rằng bất biến re-scaling mới là phần quan trọng của LayerNorm và bỏ re-centering, chỉ chia vector cho RMS rồi nhân gain học được. Trên dịch máy (RNN và Transformer), đọc hiểu, truy hồi ảnh–chú thích và CIFAR-10, nó đạt chất lượng tương đương LayerNorm, với thời gian huấn luyện giảm 7%–64% tuỳ cài đặt. Với dự án của bạn, ý nghĩa chính là hiểu một thành phần có sẵn trong mô hình, biết nó phụ thuộc vào toàn bộ chiều ẩn nên cần hiệu chỉnh khi cắt, và nhận ra rằng lợi ích tốc độ ở suy luận batch-1 chủ yếu nằm ở việc fuse kernel hơn là ở số FLOPs.

**Link gốc:** https://arxiv.org/abs/1910.07467
