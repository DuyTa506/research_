# A Mathematical Framework for Transformer Circuits (2021)

**Tác giả:** Nelson Elhage, Neel Nanda, Catherine Olsson, Chris Olah và cộng sự.  
**Nguồn:** https://transformer-circuits.pub/2021/framework/index.html  
**Chủ đề:** Mechanistic interpretability, residual stream, QK/OV circuits, path expansion, head composition và induction heads.

> Tài liệu học tập bằng tiếng Việt, tổng hợp từ phần giải thích trong cuộc trò chuyện. Đây là ghi chú diễn giải, không phải bản dịch nguyên văn paper. Những ứng dụng cho pruning/distillation là hướng mở rộng, không phải đóng góp được paper chứng minh.

## Mục lục

1. [Tư duy nền tảng: Residual Stream](#1-tư-duy-nền-tảng-residual-stream)
2. [QK Circuit và OV Circuit](#2-qk-circuit-và-ov-circuit)
3. [Circuit ở mức vocabulary](#3-circuit-ở-mức-vocabulary)
4. [Path Expansion](#4-path-expansion)
5. [Head Composition: Q, K, V](#5-head-composition-q-k-v)
6. [Induction Head](#6-induction-head)
7. [Cách kiểm chứng một circuit](#7-cách-kiểm-chứng-một-circuit)
8. [Giới hạn của framework](#8-giới-hạn-của-framework)
9. [Ứng dụng giả thuyết: Compression và Distillation](#9-ứng-dụng-giả-thuyết-compression-và-distillation)
10. [Thí nghiệm gợi ý](#10-thí-nghiệm-gợi-ý)
11. [Tóm tắt công thức](#11-tóm-tắt-công-thức)

---

## 1. Tư duy nền tảng: Residual Stream

**Câu hỏi trung tâm:** Có thể phân rã Transformer thành các thành phần tính toán có ý nghĩa, tìm được cách chúng phối hợp và giải thích *cơ chế* tạo ra hành vi của mô hình hay không?

### 1.1 Transformer như hệ thống đọc/ghi bộ nhớ chung

Thay vì chỉ xem Transformer là một chồng layer, paper xem **residual stream** là kênh thông tin chung. Mỗi attention head có thể đọc từ stream, xử lý, rồi ghi thêm kết quả vào stream.

Với kiến trúc residual lý tưởng hóa:

$$
 x^{(L)}=x^{(0)}+\sum_{l=0}^{L-1}\Delta x^{(l)}.
$$

Một thành phần ở layer sâu có thể đọc thông tin do thành phần ở layer nông ghi vào stream. Do đó có nhiều **đường truyền thông tin** chứ không chỉ một chuỗi tuần tự.

> Chú ý: feature không nhất thiết nằm ở một neuron hoặc dải chiều riêng. Nhiều feature có thể chồng lấn trong cùng không gian vector (superposition).

### 1.2 Virtual weights

Nếu thành phần A ghi bằng $W_O^A$ và thành phần B đọc bằng $W_I^B$, tương tác tuyến tính hiệu dụng giữa chúng là:

$$
\boxed{W_{\mathrm{virtual}}^{A\rightarrow B}=W_I^B W_O^A.}
$$

Đây **không phải** ma trận được lưu độc lập trong mô hình. Tích ma trận giúp khảo sát khả năng B đọc thông tin từ A, nhưng *không tự chứng minh* tương tác đó có vai trò nhân quả trong inference.

---

## 2. QK Circuit và OV Circuit

Một attention head làm hai việc: (i) chọn vị trí đọc, (ii) chuyển đổi thông tin đọc được.

Xét vector residual ở token $i$: $x_i\in\mathbb{R}^{d_{model}}$.

$$
 q_i=W_Qx_i,\qquad k_j=W_Kx_j,\qquad v_j=W_Vx_j.
$$

### 2.1 QK: Đọc ở đâu?

Điểm attention:

$$
 s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_k}}
 = x_i^\top\underbrace{\frac{W_Q^\top W_K}{\sqrt{d_k}}}_{W_{QK}}x_j.
$$

$$
\boxed{W_{QK}=W_Q^\top W_K/\sqrt{d_k}}.
$$

Sau softmax (với causal mask):

$$
 A_{ij}=\frac{e^{s_{ij}}}{\sum_{k\le i}e^{s_{ik}}}.
$$

**QK circuit** quyết định *where to read*.

### 2.2 OV: Ghi thông tin gì?

$$
 o_i=W_O\sum_j A_{ij}W_Vx_j
 =\sum_j A_{ij}\underbrace{W_OW_V}_{W_{OV}}x_j.
$$

$$
\boxed{W_{OV}=W_OW_V},\qquad \boxed{o_i=\sum_j A_{ij}W_{OV}x_j}.
$$

**OV circuit** quyết định *what to write*. Attention map cao không đồng nghĩa head làm tăng logit đúng: cần kiểm tra output của OV và những tương tác downstream.

Ví dụ minh họa (không phải một circuit cụ thể được paper xác nhận): trong `The capital of France is`, head có thể đọc `France` và ghi vector làm tăng logit `Paris`.

Đóng góp trực tiếp vào logit token $c$:

$$
\Delta z_c = w_{U,c}^\top o_i.
$$

Cần phân biệt **direct logit attribution** với tác động toàn bộ của head lên các layer phía sau.

---

## 3. Circuit ở mức vocabulary

Gọi embedding matrix là $W_E\in\mathbb R^{d_{model}\times |V|}$ và unembedding là $W_U\in\mathbb R^{|V|\times d_{model}}$. Với one-hot token $e_{t_i}$:

$$x_i=W_Ee_{t_i}.$$

### 3.1 QK token-to-token

$$
 s_{ij}=e_{t_i}^\top\underbrace{W_E^\top W_{QK}W_E}_{C_{QK}}e_{t_j}.
$$

$$\boxed{C_{QK}=W_E^\top W_{QK}W_E.}$$

Trong mô hình đơn giản, phần tử $C_{QK}[a,b]$ cho điểm attention chưa chuẩn hóa giữa query token $a$ và source token $b$. Attention xác suất vẫn phụ thuộc các token còn lại qua softmax.

### 3.2 OV token-to-token

$$
 W_Uo_i=\sum_j A_{ij}\underbrace{W_UW_{OV}W_E}_{C_{OV}}e_{t_j}.
$$

$$\boxed{C_{OV}=W_UW_{OV}W_E.}$$

$C_{OV}[c,b]$ là đóng góp trực tiếp vào logit output token $c$ khi value được lấy từ source token $b$, với attention weight bằng 1. **Lưu ý thứ tự chỉ số** theo quy ước ma trận ở đây là `[output, source]`.

QK và OV kết hợp có thể hiện thực quan hệ dạng **skip-trigram**: khi đang ở token B, head chọn token A ở quá khứ, rồi làm tăng logit token C; A không bắt buộc ngay sát B.

Trong Transformer nhiều layer, residual stream đã chứa kết quả từ các head trước, nên bảng token-to-token trên không còn mô tả đầy đủ mọi tương tác.

---

## 4. Path Expansion

Ý tưởng: xem output là tổng đóng góp của các **đường truyền thông tin** từ embedding tới logits.

### 4.1 Một layer

Với một head:

$$
 z_i=W_UW_Ee_{t_i}+\sum_j A_{ij}W_UW_{OV}W_Ee_{t_j}.
$$

- **Direct path:** $W_E\to W_U$; có thể biểu diễn thống kê bigram.
- **Attention path:** $W_E\to W_V\to W_O\to W_U$; có thể góp phần tạo các quan hệ skip-trigram.

### 4.2 Hai layer

Viết $H_l$ là toán tử attention với **attention pattern cố định**:

$$
X_1=(I+H_1)X_0,\qquad X_2=(I+H_2)X_1.
$$

Khi đó:

$$
\boxed{X_2=X_0+H_1X_0+H_2X_0+H_2H_1X_0.}
$$

Bốn đường là direct, qua head 1, qua head 2, và qua **cả hai head**. Term $H_2H_1X_0$ là đường V-composition: layer sau dùng thông tin đã được layer trước chuyển đổi.

**Cảnh báo toán học:** Vì $A=\operatorname{softmax}(QK^\top)$ phụ thuộc input, attention head **không** tuyến tính toàn cục. Khai triển này đúng khi **freezing attention patterns** trong một forward pass; không được tự ý phân phối một attention head nonlinear lên tổng input rồi coi là đẳng thức toàn cục.

---

## 5. Head Composition: Q, K, V

Head A ghi một vector $\Delta x^A=W_O^AW_V^Ax$. Head B đọc residual stream đã có vector đó. Có ba cách ảnh hưởng:

| Loại | Tích ma trận đọc output A | Vai trò |
|---|---|---|
| **Q-composition** | $W_Q^BW_O^A$ | A thay đổi query của B: B tìm gì? |
| **K-composition** | $W_K^BW_O^A$ | A thay đổi key mà B matching tới |
| **V-composition** | $W_V^BW_O^A$ | A thay đổi nội dung B truyền tiếp |

Ví dụ K-composition:

$$
 k^B=W_K^B(x+\Delta x^A)
 =W_K^Bx+W_K^BW_O^AW_V^Ax.
$$

Sự có mặt của term thứ hai cho phép head B matching dựa trên **thông tin đã được head A tạo ra**, thay vì chỉ token embedding gốc.

Trong mô hình hai layer được phân tích ở paper, K-composition đóng vai trò nổi bật trong circuit induction.

---

## 6. Induction Head

Đây là khám phá cơ chế nổi tiếng của paper: mô hình có thể nhận ra mẫu lặp trong context và tiếp tục mẫu đó.

Ví dụ:

```text
A B C D ... A B C → D
```

Trường hợp tối giản:

```text
A B ... A → B
```

### 6.1 Head 1: Previous-token head

Head thứ nhất đọc token ngay trước mỗi vị trí:

$$
 A^{(1)}_{j,k}\approx \mathbf 1[k=j-1],\qquad
 o_j^{(1)}\approx W_{OV}^{(1)}x_{j-1}^{(0)}.
$$

Residual stream tại vị trí $j$ lúc này chứa thêm thông tin về token $j-1$:

$$
 x_j^{(1)}\approx x_j^{(0)}+W_{OV}^{(1)}x_{j-1}^{(0)}.
$$

### 6.2 Head 2: Induction head

Head thứ hai tạo key từ residual stream đã được head 1 cập nhật:

$$
 k_j^{(2)}=W_K^{(2)}x_j^{(1)}
 \approx W_K^{(2)}x_j^{(0)}+W_K^{(2)}W_{OV}^{(1)}x_{j-1}^{(0)}.
$$

Query tại vị trí hiện tại $i$ matching với **token ngay trước vị trí nguồn $j$**. Vì vậy, khi token hiện tại bằng token ở $j-1$, head 2 attention đến vị trí $j$—vị trí chứa token cần copy.

Ví dụ:

```text
Position: 1 2 3 4
Token:    A B X A
```

- Head 1 tại position 2 đã ghi thông tin về `A` (position 1).
- Head 2 tại position 4 (`A`) nhận ra key tại position 2 có thông tin `A`.
- Head 2 đọc **value của position 2**, tức token `B`.
- OV circuit phù hợp sẽ tăng logit `B` → dự đoán `B`.

### 6.3 Công thức K-composition

Với cách lý tưởng hóa query phụ thuộc chủ yếu vào token hiện tại:

$$
 q_i^{(2)}\approx W_Q^{(2)}x_i^{(0)}.
$$

Key chủ yếu mang thông tin từ previous-token head:

$$
 k_j^{(2)}\approx W_K^{(2)}W_{OV}^{(1)}x_{j-1}^{(0)}.
$$

Do đó:

$$
\boxed{
 s_{ij}^{(2)}\approx (x_i^{(0)})^\top
 (W_Q^{(2)})^\top W_K^{(2)}W_{OV}^{(1)}x_{j-1}^{(0)}
}.
$$

Tích $W_{QK}^{(2)}W_{OV}^{(1)}$ đóng vai trò matching token identity trong cơ chế lý tưởng hóa này.

### 6.4 Induction và in-context learning

Induction head có thể giải thích một dạng **in-context pattern copying** mà không cần update trọng số. Nhưng nó **không phải** lời giải thích trọn vẹn cho mọi few-shot/in-context learning, chẳng hạn các tác vụ cần suy luận trừu tượng hoặc học quy tắc phức tạp.

---

## 7. Cách kiểm chứng một circuit

**Một attention pattern đẹp không đủ để khẳng định cơ chế.** Cần kết hợp các loại bằng chứng:

1. **Weight analysis:** kiểm tra các tích ma trận Q/K/V-composition để tìm head có khả năng tương tác.
2. **Activation/attention pattern:** xem head thực sự chú ý tới vị trí nào trên dữ liệu, ví dụ chuỗi ngẫu nhiên lặp lại.
3. **OV/logit analysis:** xác minh thông tin được head sao chép hay làm tăng logit gì.
4. **Ablation/causal intervention:** loại bỏ head hoặc một đường truyền cụ thể rồi đo thay đổi hành vi và loss.

Một chỉ số gợi ý đánh giá tương tác giữa head A và B:

$$
 S=\frac{\|W_I^BW_O^A\|_F}{\|W_I^B\|_F\|W_O^A\|_F}.
$$

Đây là phép chuẩn hóa heuristic để tìm ứng viên, **không phải bằng chứng nhân quả**. Việc ablate một head có thể làm loss thay đổi theo cách phi tuyến hoặc bị những head khác bù trừ.

---

## 8. Giới hạn của framework

- **Mô hình quá đơn giản:** trọng tâm là Transformer attention-only rất nhỏ (0, 1, 2 layer), khác đáng kể LLM hiện đại với MLP, normalization, RoPE, MoE, v.v.
- **Attention phụ thuộc input:** freezing attention chỉ cho phân rã tuyến tính *có điều kiện*, không biến Transformer thành hệ tuyến tính toàn cục.
- **Chức năng phân tán:** một head có thể phục vụ nhiều circuit; một circuit có thể phân tán qua nhiều head.
- **Feature representation chưa được giải đầy đủ:** biết head truyền gì chưa đủ để xác định feature cụ thể được mã hóa trong neuron/direction/subspace nào.

---

## 9. Ứng dụng giả thuyết: Compression và Distillation

> **Phần này là suy luận nghiên cứu mở rộng, không phải kết quả trực tiếp từ paper.**

### 9.1 Vì sao layer importance có thể đánh lừa?

Giả sử circuit quan trọng:

$$H_4\rightarrow H_{12}\rightarrow H_{20}.$$

Head ở layer 12 có thể ít đóng góp trực tiếp vào logits nhưng vẫn quan trọng vì chuyển thông tin cho layer 20. Prune dựa riêng direct logit attribution sẽ có thể phá circuit.

### 9.2 Circuit-aware pruning

Đề xuất biểu diễn mô hình thành đồ thị:

$$G=(V,E)$$

với node là head/feature, edge là tương tác thông tin. Giữ những đường truyền quan trọng về mặt nhân quả thay vì chỉ giữ head có score cục bộ cao.

Cần đo trên data/task thật; không được đồng nhất norm của virtual weight với importance.

### 9.3 Circuit-aware knowledge distillation

Thay vì ép từng attention map của student và teacher khớp nhau:

$$A_l^S\approx A_l^T,$$

có thể thử bảo toàn **chức năng tổng hợp** của các circuit:

$$f_{\mathrm{circuit}}^S(x)\approx f_{\mathrm{circuit}}^T(x).$$

Cùng một chức năng có thể được triển khai bằng những circuit khác nhau ở hai kiến trúc khác nhau; đó vừa là cơ hội, vừa là thách thức đo lường.

---

## 10. Thí nghiệm gợi ý

**Mục tiêu:** tìm và xác nhận induction circuit trên một Transformer nhỏ.

### Bước 1 — Dữ liệu lặp

Tạo chuỗi token ngẫu nhiên $S=[t_1,\dots,t_n]$ rồi đưa vào $X=[S,S]$.

```text
17 93 41 28 65 17 93 41 28 → 65
```

### Bước 2 — Induction score

Đo trung bình attention từ vị trí $i$ ở nửa sau đến vị trí tương ứng ở nửa đầu lệch **+1**:

$$
 I_h=\mathbb E_X\left[\frac1{|J|}\sum_{i\in J} A^{(h)}_{i,i-n+1}\right].
$$

Trong đó $J$ chỉ gồm vị trí có chỉ số hợp lệ. Đây là chỉ số phát hiện head ứng viên, chưa chứng minh chức năng.

### Bước 3 — K-composition

Đo tương tác previous-token head A và head B ở layer sau:

$$
 C_K(A,B)=\frac{\|W_K^BW_O^A\|_F}{\|W_K^B\|_F\|W_O^A\|_F}.
$$

So sánh score với induction score, tìm cặp phù hợp.

### Bước 4 — Causal test

Ablate previous-token head hoặc riêng đường K-composition, đo:

- Induction attention pattern có giảm?
- Logit của token cần copy có giảm?
- Cross-entropy trên đoạn lặp có tăng?
- Có head khác bù trừ?

### Bước 5 — Mở rộng compression

Thử thay circuit nhiều head bằng module gọn hơn, kiểm tra chức năng copying và performance trên task ngoài phân phối huấn luyện.

---

## 11. Tóm tắt công thức

| Khái niệm | Công thức | Ý nghĩa |
|---|---|---|
| QK | $W_{QK}=W_Q^\top W_K/\sqrt{d_k}$ | Chọn vị trí để đọc |
| OV | $W_{OV}=W_OW_V$ | Biến đổi rồi ghi thông tin |
| Token QK | $C_{QK}=W_E^\top W_{QK}W_E$ | Score truy vấn–nguồn ở mức token |
| Token OV | $C_{OV}=W_UW_{OV}W_E$ | Đóng góp nguồn–output ở mức logit |
| Virtual weights | $W_I^BW_O^A$ | Tương tác head A→B |
| Q-composition | $W_Q^BW_O^A$ | A ảnh hưởng query B |
| K-composition | $W_K^BW_O^A$ | A ảnh hưởng key B |
| V-composition | $W_V^BW_O^A$ | A ảnh hưởng value B |
| Path expansion | $(I+H_2)(I+H_1)X$ | Tổng các đường khi attention cố định |
| Induction | previous-token head → K-composition → copying head | Tiếp tục mẫu lặp trong context |

### Kết luận ngắn

Paper chuyển câu hỏi từ *“head này chú ý tới đâu?”* sang *“những phép tính và đường truyền thông tin nào tạo ra hành vi của mô hình?”*. Đóng góp then chốt là phân tách QK/OV, phát triển phép phân rã theo path và giải thích induction head bằng head composition. Từ đây có thể hình thành các giả thuyết cho circuit-aware pruning/distillation, nhưng chúng phải được xác thực bằng can thiệp nhân quả và đánh giá task.

**Đọc bản gốc:** https://transformer-circuits.pub/2021/framework/index.html
