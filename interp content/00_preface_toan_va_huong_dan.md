# Preface: Toán nền tảng và hướng dẫn trước khi học Interpretability

**Dành cho:** người đã quen deep learning và transformer nhưng mới bắt đầu với mechanistic interpretability.
**Mục tiêu:** ôn đúng phần toán mà các bài trên transformer-circuits.pub dùng nhiều nhất, và biết đọc và thực hành theo cách nào cho hiệu quả.
**Đọc khi nào:** trước file `01`. Có thể quay lại tra cứu bất cứ lúc nào.

> Đây là phần chuẩn bị do chúng ta tự soạn, không phải nội dung của Anthropic. Mỗi khái niệm có ghi "Gặp ở" để bạn biết nó xuất hiện trong bài nào của lộ trình.

## Mục lục

1. [Vì sao cần toán ở đây](#1-vì-sao-cần-toán-ở-đây)
2. [Đại số tuyến tính: phần quan trọng nhất](#2-đại-số-tuyến-tính-phần-quan-trọng-nhất)
3. [Hình học không gian nhiều chiều](#3-hình-học-không-gian-nhiều-chiều)
4. [Xác suất, thông tin và output của model](#4-xác-suất-thông-tin-và-output-của-model)
5. [Tối ưu và tính thưa](#5-tối-ưu-và-tính-thưa)
6. [Ký hiệu transformer dùng trong Transformer Circuits](#6-ký-hiệu-transformer-dùng-trong-transformer-circuits)
7. [Đo lường và thống kê thường gặp](#7-đo-lường-và-thống-kê-thường-gặp)
8. [Hướng dẫn đọc paper interp](#8-hướng-dẫn-đọc-paper-interp)
9. [Công cụ thực hành](#9-công-cụ-thực-hành)
10. [Lịch học gợi ý](#10-lịch-học-gợi-ý)
11. [Bài tự kiểm tra (có đáp án)](#11-bài-tự-kiểm-tra-có-đáp-án)
12. [Bảng tra công thức](#12-bảng-tra-công-thức)

---

## 1. Vì sao cần toán ở đây

Mechanistic interpretability muốn **đọc ngược** phép tính bên trong model. Hầu như mọi lập luận trong các bài đều quy về ba loại câu hỏi:

| Câu hỏi | Công cụ toán |
|---|---|
| Thông tin nằm theo **hướng** nào trong vector? | tích vô hướng, phép chiếu, cơ sở (basis) |
| Thành phần A **truyền** gì cho thành phần B? | tích ma trận, low-rank, SVD, trị riêng |
| Vì sao nhét được **nhiều feature hơn số chiều**? | hình học nhiều chiều, tính thưa, phi tuyến có ngưỡng |

Nếu bạn vững mục 2 và 3, khoảng 80% công thức trong lộ trình sẽ đọc được ngay.

---

## 2. Đại số tuyến tính: phần quan trọng nhất

### 2.1 Vector, tích vô hướng và phép chiếu

Residual stream của một token là vector $x \in \mathbb{R}^{d}$ (ví dụ $d = 1024$). Một **feature** được mô hình hóa là một hướng đơn vị $u$ với $\|u\| = 1$. **Cường độ** của feature trong $x$ là phép chiếu:

$$
\boxed{a_u(x) = u^\top x = \|x\|\cos\theta}
$$

- $\cos\theta = 1$: $x$ cùng hướng $u$. $\cos\theta = 0$: vuông góc, feature "tắt".
- Thành phần của $x$ nằm dọc theo $u$ là $(u^\top x)\,u$. Phần còn lại $x - (u^\top x)u$ vuông góc với $u$.

**Gặp ở:** 01, 05, 10 và hầu như mọi bài.

### 2.2 Cơ sở (basis) và đổi cơ sở

Một cơ sở là một tập $d$ vector độc lập. Cơ sở chuẩn là các trục $e_1,\dots,e_d$, tức các "tọa độ" hay "neuron". Đổi sang cơ sở trực chuẩn khác bằng ma trận trực giao $Q$ (thỏa $Q^\top Q = I$):

$$
x' = Q^\top x, \qquad x = Q x'.
$$

**Ý quan trọng:** nếu mọi phép tính phía sau đều tuyến tính, ta có thể xoay cả không gian mà model vẫn tương đương. Đó là lý do residual stream **về lý thuyết không có hệ trục ưu tiên**. Ngược lại, lớp ẩn MLP có $\sigma(\cdot)$ áp lên từng tọa độ, nên các trục neuron ở đó **được ưu tiên (privileged)**.

**Gặp ở:** 01 (interpretable bases), 06 (privileged basis), 07 (SoLU).

### 2.3 Ma trận là phép biến đổi; tích ma trận là ghép phép biến đổi

$W \in \mathbb{R}^{m\times n}$ biến vector $n$ chiều thành $m$ chiều. Nếu A ghi bằng $W_O^A$ và B đọc bằng $W_I^B$, ảnh hưởng tuyến tính của A lên B là:

$$
W_I^B W_O^A \quad \text{("virtual weight")}.
$$

**Gặp ở:** 02 (QK/OV, composition), 14 (crosscoder), 17 (transcoder).

### 2.4 Hạng (rank) và low-rank

Hạng là số chiều thật sự mà ma trận "đi qua". Trong attention, $W_{OV} = W_O W_V$ có kích thước $d \times d$ nhưng hạng tối đa là $d_{head}$ (ví dụ 64). Một head chỉ **đọc và ghi trong một không gian con nhỏ**.

**Gặp ở:** 02, 20, 21.

### 2.5 Trị riêng và vector riêng (eigen)

$Mv = \lambda v$: $v$ là hướng mà $M$ chỉ co giãn, không xoay. Framework (bài 02) dùng trị riêng của mạch OV ở mức token để nhận diện **head sao chép**. Nếu đa số trị riêng **dương**, head có xu hướng "ghi lại chính token nó đọc".

$$
\text{copying score} \approx \frac{\sum_i \lambda_i}{\sum_i |\lambda_i|}.
$$

**Gặp ở:** 02, 04.

### 2.6 SVD và PCA

Mọi ma trận đều phân tích được thành $W = U\Sigma V^\top$, với $U, V$ trực giao và $\Sigma$ chéo, không âm.
- **Đọc** theo các hướng $V$, **co giãn** theo $\Sigma$, **ghi** theo các hướng $U$.
- Giữ $k$ giá trị kỳ dị lớn nhất cho xấp xỉ hạng $k$ tốt nhất.
- **PCA** là SVD áp lên ma trận activation đã trừ trung bình. Nó tìm các hướng có phương sai lớn nhất.

*Cảnh báo:* feature quan trọng **không nhất thiết** nằm trên hướng phương sai lớn. Đó là lý do PCA và CKA có thể bỏ sót tổn thương.

**Gặp ở:** 06, 10, các bài về SAE, và phần so sánh biểu diễn.

### 2.7 Chuẩn và phân tách đối xứng / phản đối xứng

- **Chuẩn Frobenius:** $\|W\|_F = \sqrt{\sum_{ij} W_{ij}^2}$, dùng để đo "độ lớn" của ma trận.
- Mọi ma trận vuông tách được thành phần đối xứng và phần phản đối xứng:

$$
W = \underbrace{\tfrac{1}{2}(W + W^\top)}_{S\ (\text{co giãn})} + \underbrace{\tfrac{1}{2}(W - W^\top)}_{A\ (\text{thành phần "xoay"})},
\qquad \|W\|_F^2 = \|S\|_F^2 + \|A\|_F^2 .
$$

**Gặp ở:** báo cáo pruning (Ghosted Layers: khoảng 47% độ lớn của map sửa lỗi nằm ở phần xoay).

### 2.8 Bình phương tối thiểu (least squares) và giả nghịch đảo

Tìm $W$ sao cho $W H_s \approx H_p$, với các cột là activation trên $N$ mẫu:

$$
\boxed{W^\star = \arg\min_W \|W H_s - H_p\|_F^2 = H_p H_s^\top (H_s H_s^\top + \lambda I)^{-1}}
$$

Ở đây $\lambda$ là hệ số ridge để ổn định khi $H_s H_s^\top$ gần suy biến. Đây chính là phép tính **D_lin** trong dự án của ta: phần tổn thương mà $W^\star$ gỡ được.

**Gặp ở:** báo cáo pruning, 02b, model stitching, tuned lens.

---

## 3. Hình học không gian nhiều chiều

### 3.1 Gần vuông góc thì có rất nhiều

Trong không gian 2 chiều, chỉ có 2 hướng vuông góc nhau. Trong không gian $d$ chiều lớn, số hướng **gần vuông góc** (mọi cặp có $|\cos| < \varepsilon$) tăng **theo hàm mũ** của $d$. Đây là hệ quả của bổ đề Johnson–Lindenstrauss. Hai vector ngẫu nhiên trong $\mathbb{R}^d$ có $\cos \approx 0$, với độ lệch cỡ $1/\sqrt{d}$.

**Hệ quả:** model có thể xếp **nhiều feature hơn số chiều**. Đó là **superposition**.

**Gặp ở:** 05 (Toy Models), 08, 09, 10, 12.

### 3.2 Giao thoa (interference)

Hai feature có hướng $u_1, u_2$ với $u_1^\top u_2 = c \neq 0$. Khi chỉ feature 2 bật với cường độ $a_2$, việc đọc feature 1 vẫn nhận được một phần nhiễu:

$$
u_1^\top (a_2 u_2) = a_2\, c .
$$

Nếu các feature **thưa** (hiếm khi bật cùng lúc), nhiễu này nhỏ và hiếm. Model lọc nó bằng phi tuyến có ngưỡng (mục 5.2).

**Gặp ở:** 05, 23, 24 (interference weights).

### 3.3 Ví dụ đồ chơi: 3 feature trong 2 chiều

Đặt ba hướng cách nhau 120°: $u_A = (1,0)$, $u_B = (-\tfrac12, \tfrac{\sqrt3}{2})$, $u_C = (-\tfrac12, -\tfrac{\sqrt3}{2})$. Khi chỉ một feature bật, đọc qua $\mathrm{ReLU}(u^\top x - b)$ với ngưỡng $b \approx 0.5$ sẽ tách được cả ba.

- **Xóa chiều thứ 2** (giống cắt một neuron) thì $u_B$ và $u_C$ đều sụp về $(-\tfrac12, 0)$, không phân biệt được nữa.
- **Xoay cả mặt phẳng** (giống lệch trục khi bỏ layer) thì thông tin vẫn còn đủ, và một ma trận xoay ngược lại sửa được.

Đây là trực giác đằng sau giả thuyết D_res (cắt neuron) và D_lin (cắt layer) của dự án.

---

## 4. Xác suất, thông tin và output của model

### 4.1 Logit, softmax, log-prob

$$
p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad \log p_i = z_i - \log\sum_j e^{z_j}.
$$

- Softmax **không đổi khi cộng cùng một hằng số** vào mọi logit. Vì vậy các bài thường so **hiệu logit** $z_{\text{đúng}} - z_{\text{sai}}$ thay vì logit tuyệt đối.
- **Direct logit attribution:** đóng góp của vector $o$ (ví dụ output một head) vào logit token $c$ là $w_{U,c}^\top o$.

**Gặp ở:** 02, 04, 17, 19.

### 4.2 Cross-entropy và KL

$$
H(p,q) = -\sum_i p_i \log q_i, \qquad
\mathrm{KL}(p\,\|\,q) = \sum_i p_i \log\frac{p_i}{q_i} = H(p,q) - H(p).
$$

- Loss khi train là cross-entropy với nhãn thật.
- **"Loss tăng thêm"** khi thay activation bằng bản tái tạo của SAE là thước đo chính cho chất lượng SAE.
- **KL tới parent** là thước đo tổn thương trong dự án pruning.

**Gặp ở:** 10, 12, 17 và các bài đánh giá SAE/transcoder.

### 4.3 Kỳ vọng, phương sai, "fraction of variance explained"

$$
\mathrm{FVE} = 1 - \frac{\mathbb{E}\|x - \hat x\|^2}{\mathbb{E}\|x - \bar x\|^2}.
$$

FVE = 1 nghĩa là tái tạo hoàn hảo. Các bài SAE báo cáo FVE hoặc MSE chuẩn hóa. FVE cao chưa chắc loss downstream thấp, nên thường báo cả hai.

---

## 5. Tối ưu và tính thưa

### 5.1 Phạt L1 tạo ra nghiệm thưa

$$
\mathcal{L} = \|x - \hat x\|_2^2 + \lambda \sum_i |f_i|.
$$

Phạt L1 đẩy nhiều $f_i$ về **đúng 0**, khác với L2 chỉ làm chúng nhỏ đi. Các biến thể hay gặp:
- **TopK**: chỉ giữ $k$ giá trị lớn nhất;
- **JumpReLU**: có ngưỡng học được;
- **L0** = số phần tử khác 0, thước đo độ thưa. Ví dụ "trung bình 30 feature bật mỗi token".

**Gặp ở:** 10, 12, 14, 17.

### 5.2 Phi tuyến có ngưỡng là "bộ lọc nhiễu"

$$
\mathrm{ReLU}(u^\top x - b).
$$

Bias âm $b$ cắt bỏ phần giao thoa nhỏ (mục 3.2). Đó là lý do superposition chỉ hoạt động tốt khi có phi tuyến. Nó cũng là lý do *mất thông tin trước phi tuyến* thì không sửa lại được bằng phép tuyến tính sau đó.

### 5.3 Khai triển Taylor bậc 1 và attribution bằng gradient

Thay vì can thiệp thật (tốn một lần chạy cho mỗi vị trí), ta xấp xỉ:

$$
\Delta \mathcal{L} \approx \nabla_{a}\mathcal{L}^\top (a' - a)
\quad\text{(attribution patching)}, \qquad
\text{importance}(a) \approx a \cdot \frac{\partial \mathcal{L}}{\partial a}
\quad\text{(gradient × activation)}.
$$

Cách này rẻ nhưng sai khi thay đổi lớn hoặc khi model bão hòa. Các tiêu chí pruning kiểu Taylor/Wanda dùng cùng tinh thần này.

**Gặp ở:** 17 (attribution), 19, các bài về patching.

---

## 6. Ký hiệu transformer dùng trong Transformer Circuits

| Ký hiệu | Nghĩa |
|---|---|
| $d_{model}$, $d_{head}$, $d_{mlp}$ | số chiều residual stream, một head, lớp ẩn MLP |
| $W_E$, $W_U$ | ma trận embedding, unembedding |
| $W_Q, W_K, W_V, W_O$ | ma trận query, key, value, output của một head |
| $W_{QK} = W_Q^\top W_K$ | mạch "nhìn vào đâu" (QK circuit) |
| $W_{OV} = W_O W_V$ | mạch "ghi cái gì" (OV circuit) |
| $A_{ij}$ | attention pattern: token $i$ nhìn token $j$ bao nhiêu |
| $x^{(l)}$ | residual stream sau layer $l$ |

**Quy ước cần nhớ:**
- Bài 02 dùng model **attention-only, không LayerNorm** để toán gọn. Model thật có LayerNorm/RMSNorm, là một phi tuyến nhẹ. Người ta thường "**fold**" trọng số norm vào ma trận phía sau và coi hệ số chuẩn hóa là hằng số khi phân tích.
- **Attention pattern phụ thuộc input** (qua softmax). Mọi khai triển "tuyến tính" trong bài 02 chỉ đúng **khi giữ cố định attention pattern**.
- Một layer gồm attention cộng MLP, mỗi phần đều **cộng** vào residual stream.

**Với model của ta:** GatedDeltaNet thay attention softmax bằng một **trạng thái hồi quy** $S_t$ cập nhật theo delta rule và cổng (gate). Phần lớn công cụ trong các bài được viết cho attention softmax và MLP, nên áp lên GatedDeltaNet cần thận trọng. Khi patching, coi $S_t$ là một "activation" riêng.

---

## 7. Đo lường và thống kê thường gặp

| Đại lượng | Ý nghĩa | Lưu ý |
|---|---|---|
| Pearson $r$ | tương quan tuyến tính | nhạy với outlier |
| Spearman $\rho$, Kendall $\tau$ | tương quan thứ hạng | dùng khi chỉ quan tâm thứ tự |
| Cosine similarity | góc giữa hai vector | không bất biến khi xoay |
| CKA | so cấu trúc tương đồng của hai tập biểu diễn | bất biến khi xoay; có thể cao dù model đã hỏng |
| Kurtosis | độ "đuôi nặng" | phát hiện outlier dimension |
| Activation density | tỉ lệ token mà feature bật | feature "chết" thì density ≈ 0 |
| Ablation effect | output đổi bao nhiêu khi tắt thành phần | cần chọn kiểu tắt: zero, mean, hay resample |

**Ba kiểu ablation:**
- **Zero ablation**: đặt về 0. Dễ đẩy model ra ngoài phân phối.
- **Mean ablation**: đặt về giá trị trung bình trên dữ liệu.
- **Resample ablation**: lấy giá trị từ một input khác.

Kết luận có thể đổi tùy kiểu, nên các bài thường nói rõ mình dùng kiểu nào.

---

## 8. Hướng dẫn đọc paper interp

1. **Đọc theo vòng.** Vòng 1 đọc phần tóm tắt, các hình chính và phần "takeaways". Vòng 2 mới đọc chi tiết phương pháp.
2. **Trang gốc có hình tương tác.** Bản ghi chú chỉ tóm tắt. Khi gặp đoạn "xem hình X", hãy mở trang gốc và rê chuột vào hình. Nhiều kết luận nằm trong các hình đó.
3. **Phân biệt ba loại câu:** điều paper *chứng minh bằng thí nghiệm*, điều paper *đặt giả thuyết*, và điều *ta suy ra*. Ghi chú của chúng ta tách rõ ba loại này.
4. **Luôn hỏi "bằng chứng nhân quả ở đâu?"** Một attention pattern đẹp hay một feature dễ đặt tên vẫn chưa đủ. Cần thấy can thiệp (ablation, steering, patching) làm hành vi thay đổi đúng như dự đoán.
5. **Chú ý cỡ model.** Nhiều kết quả làm trên model đồ chơi hoặc model 1–2 layer. Đọc kỹ đoạn tác giả bàn về mức tổng quát hóa.
6. **Circuits Updates** là ghi chú dở dang: có kết quả âm tính và ý tưởng chưa chín. Nên đọc để biết hướng nghiên cứu, nhưng đừng trích như kết luận chắc chắn.

---

## 9. Công cụ thực hành

| Công cụ | Dùng làm gì |
|---|---|
| **PyTorch hooks** (`register_forward_hook`) | Lấy và sửa activation ở bất kỳ module nào. Đủ cho hầu hết thí nghiệm. |
| **TransformerLens** | Thư viện interp phổ biến cho các kiến trúc GPT-like. Có sẵn hook ở mọi vị trí và tiện ích phân tích attention/logit. |
| **nnsight** | Can thiệp vào model HuggingFace bất kỳ, kể cả kiến trúc lạ (hợp với model hybrid của ta). |
| **SAELens** | Huấn luyện và nạp sparse autoencoder. |
| **circuit-tracer** | Công cụ mã nguồn mở đi kèm bài 17/19: dựng attribution graph với transcoder. |
| **Neuronpedia** | Website xem feature của các SAE đã công bố. Tốt để làm quen feature trông như thế nào. |

**Thiết lập gợi ý cho người mới:**
1. Bắt đầu với một model rất nhỏ (GPT-2 small, hoặc model 1–2 layer attention-only) để chạy bài tập của bài 02–04.
2. Sau đó chuyển sang decoder của model ta. Lấy activation residual stream trên khoảng 1.000 trang OCR, lưu ra đĩa, rồi làm thí nghiệm SAE và least squares offline.

Ví dụ khung lấy activation bằng hook (PyTorch):

```python
acts = {}
def save(name):
    def hook(module, inp, out):
        acts[name] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
    return hook

handles = [model.model.layers[i].register_forward_hook(save(f"L{i}"))
           for i in range(len(model.model.layers))]
with torch.no_grad():
    model(**batch)
for h in handles: h.remove()
# acts["L10"] có shape [batch, seq, d_model]: residual stream sau layer 10
```

(Tên thuộc tính `model.model.layers` thay đổi tùy kiến trúc. Hãy in `model` ra để tìm đúng đường dẫn.)

---

## 10. Lịch học gợi ý

| Tuần | Đọc | Làm |
|---|---|---|
| 1 | Preface này, 01, 02, 02b | Tự tính $W_{OV}$, $W_{QK}$ trên GPT-2 small; vẽ attention pattern. |
| 2 | 03, 04 | Tìm induction head bằng chuỗi token ngẫu nhiên lặp lại (bài tập ở 03/04). |
| 3 | 05, 06, 07 | Tái hiện toy model superposition (5 feature, 2 chiều) và vẽ hình hướng feature. |
| 4 | 08, 09, 10 | Train một SAE nhỏ trên một layer của GPT-2 small hoặc model của ta. |
| 5 | 12, 13, 14, 15, 16 | Thử crosscoder hoặc so SAE giữa parent và student. |
| 6 | 17, 19, 25 | Chạy circuit-tracer trên một prompt đơn giản; đọc error node. |
| 7+ | 20–31 theo nhu cầu; 32–35 tra dần | Áp vào dự án: D_lin/D_res, patching parent→student, OCR head. |

Không cần theo đúng số tuần. Điều quan trọng là **mỗi giai đoạn làm ít nhất một thí nghiệm nhỏ**.

---

## 11. Bài tự kiểm tra (có đáp án)

**Câu 1.** Vì sao có thể xoay residual stream mà model không đổi, còn xoay lớp ẩn MLP thì không được?

<details><summary>Đáp án</summary>
Ma trận đọc và ghi vào residual stream đều tuyến tính. Xoay bằng Q thì chỉ cần nhân Qᵀ vào ma trận đọc và Q vào ma trận ghi là bù lại được. Lớp ẩn MLP có σ áp lên từng tọa độ, và σ(Qh) ≠ Qσ(h), nên trục neuron mang ý nghĩa riêng (privileged).
</details>

**Câu 2.** $W_{OV}$ của một head có kích thước $1024\times1024$ và $d_{head}=64$. Hạng tối đa là bao nhiêu, và điều đó nói gì?

<details><summary>Đáp án</summary>
Tối đa 64. Head chỉ đọc và ghi trong một không gian con 64 chiều của residual stream.
</details>

**Câu 3.** Có 3 feature trong 2 chiều, đặt cách nhau 120°. Nếu bật đồng thời A và B, đọc feature C bằng $u_C^\top x$ được bao nhiêu? Ngưỡng $b = 0.5$ có lọc được không?

<details><summary>Đáp án</summary>
u_C·(u_A+u_B) = cos120° + cos120° = −1. Giá trị âm nên ReLU(−1 − 0.5) = 0: C không bật nhầm. Nhưng A và B lại nhiễu lẫn nhau: đọc A được 1 + cos120° = 0.5, bằng đúng ngưỡng, nên A gần như bị tắt. Superposition chỉ an toàn khi các feature hiếm khi bật cùng lúc.
</details>

**Câu 4.** Viết công thức $W^\star$ cho D_lin. Vì sao cần thêm $\lambda I$?

<details><summary>Đáp án</summary>
W* = H_p H_sᵀ (H_s H_sᵀ + λI)⁻¹. Ma trận hiệp phương sai H_s H_sᵀ có thể gần suy biến (activation có vài chiều phương sai rất nhỏ, hoặc N nhỏ hơn d). λI giúp nghịch đảo ổn định và tránh overfit vào tập calibration.
</details>

**Câu 5.** Vì sao các bài hay báo cáo "loss tăng thêm khi thay bằng bản tái tạo của SAE" chứ không chỉ báo MSE?

<details><summary>Đáp án</summary>
MSE thấp chưa chắc giữ được những hướng mà model thực sự dùng. Một sai số nhỏ trên hướng quan trọng có thể làm loss tăng nhiều. Loss downstream đo trực tiếp xem bản tái tạo có còn "dùng được" với phần sau của model không.
</details>

**Câu 6.** CKA giữa student và parent là 0.95 ở mọi layer. Có thể kết luận student "gần như không mất gì" không?

<details><summary>Đáp án</summary>
Không. CKA bị chi phối bởi các hướng phương sai lớn. Những hướng nhỏ nhưng quyết định (ví dụ phân biệt dấu thanh) có thể bị phá mà CKA vẫn cao. Cần đo bằng loss/KL, bằng patching, hoặc bằng feature SAE.
</details>

---

## 12. Bảng tra công thức

| Khái niệm | Công thức | Gặp ở |
|---|---|---|
| Cường độ feature | $a_u(x) = u^\top x$ | 01, 05, 10 |
| Virtual weight | $W_I^B W_O^A$ | 02, 14 |
| QK / OV | $W_Q^\top W_K$, $W_O W_V$ | 02, 20 |
| Copying score | $\sum\lambda_i / \sum|\lambda_i|$ | 02, 04 |
| SVD | $W = U\Sigma V^\top$ | 06, 10 |
| Tách S/A | $W = \tfrac12(W+W^\top) + \tfrac12(W-W^\top)$ | báo cáo pruning |
| Least squares | $W^\star = H_p H_s^\top(H_sH_s^\top+\lambda I)^{-1}$ | D_lin, stitching |
| Giao thoa | $u_1^\top(a_2u_2) = a_2\,u_1^\top u_2$ | 05, 23, 24 |
| Lọc ngưỡng | $\mathrm{ReLU}(u^\top x - b)$ | 05, 10 |
| SAE loss | $\|x-\hat x\|^2 + \lambda\|f\|_1$ | 10, 12 |
| KL | $\sum p\log(p/q)$ | 10, 17, pruning |
| FVE | $1 - \mathbb{E}\|x-\hat x\|^2/\mathbb{E}\|x-\bar x\|^2$ | 10, 12 |
| Attribution | $\nabla_a\mathcal{L}^\top(a'-a)$ | 17, 19 |

**Bước tiếp theo:** mở `01_2022_mech_interp_essay.md`.
