# Sparse mixtures of linear transforms (2025)

**Tác giả:** Jack Lindsey, Brian Chen, Adam Pearce, Sasha Hydrie, Thomas Conerly; biên tập bởi Jeff Wu (Anthropic). Trang không ghi ngày cụ thể; đường dẫn thuộc năm 2025.  
**Nguồn:** https://transformer-circuits.pub/2025/bulk-update/index.html  
**Chủ đề:** MOLT (sparse mixture of linear transforms) thay cho transcoder; phân bố hạng; hiệu quả tính toán; faithfulness qua Jacobian; tích hợp vào attribution graph.  
**Vị trí trong lộ trình:** Đọc sau *Circuit Tracing* và *Biology* (2025), nơi phép cộng được mô tả bằng "lookup table feature". Bài này đề xuất một đơn vị tính toán khác cho MLP. Với dự án pruning MLP của bạn, đây có lẽ là note **liên quan trực tiếp nhất** trong loạt 2025.

> Ghi chú diễn giải bằng tiếng Việt, không phải bản dịch. Tác giả gọi đây là update sơ bộ về một phương pháp còn đang phát triển. Phần ứng dụng cho pruning/distillation là suy luận của chúng tôi.

## Mục lục

1. [Động lực: transcoder băm nhỏ tính toán](#1-động-lực-transcoder-băm-nhỏ-tính-toán)
2. [Định nghĩa MOLT](#2-định-nghĩa-molt)
3. [Trực giác](#3-trực-giác)
4. [Kết quả](#4-kết-quả)
5. [MOLT trong attribution graph](#5-molt-trong-attribution-graph)
6. [Giới hạn và công trình liên quan](#6-giới-hạn-và-công-trình-liên-quan)
7. [Ứng dụng giả thuyết cho pruning và distillation](#7-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
8. [Thí nghiệm gợi ý](#8-thí-nghiệm-gợi-ý)
9. [Tóm tắt công thức và khái niệm](#9-tóm-tắt-công-thức-và-khái-niệm)

---

## 1. Động lực: transcoder băm nhỏ tính toán

**Transcoder**: MLP rất rộng và thưa, huấn luyện để thay MLP gốc; neuron của nó là feature. Theo tác giả, transcoder "shatter" (băm vụn) tính toán thành mảnh rất nhỏ, có thể biểu diễn tính toán kém hiệu quả hơn mô hình gốc, và gây **feature splitting** (một khái niệm vỡ thành nhiều feature hẹp) và **absorption** (feature tổng quát bị feature đặc thù "hút" mất một phần).

Ví dụ: phép cộng được phân rã thành lookup table như "6 + 9 = 5 (mod 10)". Đúng ở một nghĩa nào đó, nhưng bỏ lỡ cấu trúc hình học (số được nhúng sao cho phép biến đổi đơn giản tính được phép toán) và cực kỳ tốn kém. Để bắt hết tính toán có thể cần số feature khổng lồ, nhiều tham số hơn chính mô hình gốc.

**Đề xuất:** thay MLP bằng **MOLT**. Không học feature (hướng trong không gian activation) mà học **transform** hoạt động thưa: mỗi transform là một phép biến đổi tuyến tính áp lên residual stream. Transform là đối tượng *thuần tính toán*, "bắc cầu" biểu diễn giữa các layer, nên phải dùng kèm một phương pháp phân rã biểu diễn như SAE.

---

## 2. Định nghĩa MOLT

$$
\boxed{\;f(\mathbf x)=\sum_t \phi(\mathbf e_t\cdot\mathbf x-b_t)\,\big(U_tV_t\,\mathbf x\big)\;}
$$

- $\mathbf x$: residual stream đầu vào.
- $\mathbf e_t, b_t$: encoder và bias của transform $t$, quyết định transform bật hay không và bật mạnh cỡ nào; $\phi$ là ReLU hoặc JumpReLU.
- $U_t\in\mathbb R^{d_{model}\times k_t}$, $V_t\in\mathbb R^{k_t\times d_{model}}$: $U_tV_t$ có hạng $k_t$; các transform có thể có hạng khác nhau (phân bố hạng là siêu tham số).

Huấn luyện bắt chước output MLP (như transcoder), với sparsity penalty (L1 hoặc tanh) trên activation transform, nhân với chuẩn Frobenius:

$$
\mathcal L_{\text{sparse}}=\sum_t \|U_tV_t\|_F\cdot\phi(\mathbf e_t\cdot\mathbf x-b_t),
$$

tương tự cách transcoder nhân penalty với chuẩn decoder.

**So sánh với transcoder:**

| | Transcoder | MOLT |
|---|---|---|
| Đơn vị | Feature: encoder + **một vector** decoder | Transform: encoder + **ma trận** hạng $k_t$ |
| Output một đơn vị | $a_i\,\mathbf d_i$ (cố định hướng) | $a_t\,U_tV_t\mathbf x$ (phụ thuộc input) |
| Vai trò | Vừa biểu diễn vừa tính toán | Chỉ tính toán |
| Hạng Jacobian | $\le L_0$ | Có thể $\gg L_0$ |

---

## 3. Trực giác

**(a) MLP ReLU vốn là hỗn hợp phép tuyến tính.** Cố định tập neuron hoạt động $S$, MLP chỉ là phép tuyến tính:

$$
\mathrm{MLP}(\mathbf x)\big|_{S}=W_{out}\,D_S\,W_{in}\,\mathbf x,\qquad \operatorname{rank}\le|S|,
$$

với $D_S$ là ma trận đường chéo chọn neuron trong $S$ (công thức của chúng tôi, mô tả đúng lập luận bằng lời của bài). Mỗi mẫu kích hoạt neuron tương ứng một transform tuyến tính, gần giống một transform của MOLT.

**(b) Khai thác hình học.** Nếu chữ số 0–9 nằm trên một vòng tròn, một transform hạng 2 "plus 3" có thể là phép **quay**: biến feature "1" thành "say 4", "2" thành "say 5", ... Encoder của nó bật khi ngữ cảnh có "+3" hoặc "add three". Một transform thay cho cả họ lookup table.

Bài gốc mô tả đây là phép quay "8/10 vòng"; con số cụ thể phụ thuộc cách sắp chữ số trên vòng tròn. Ví dụ nhỏ dưới đây (minh họa của chúng tôi) dùng cách sắp đều đơn giản nhất: đặt chữ số $n$ tại góc $\theta_n=2\pi n/10$, vector $(\cos\theta_n,\sin\theta_n)$. Transform "plus 3" là ma trận quay $R(2\pi\cdot 3/10)$, áp cho mọi $n$:

$$
R\!\left(\tfrac{6\pi}{10}\right)\begin{pmatrix}\cos\theta_n\\ \sin\theta_n\end{pmatrix}=\begin{pmatrix}\cos\theta_{n+3}\\ \sin\theta_{n+3}\end{pmatrix}.
$$

Theo cách nhìn lookup table, transcoder cần cỡ một feature cho mỗi cặp (chữ số, +3) để làm cùng việc.

**(c) Tham số hóa tổ hợp.** 1 triệu feature ở input layer $L$ và 1 triệu transform ở MLP layer $L$ mô tả hiệu dụng $10^{12}$ **pseudofeature** ở layer $L+1$ (cặp feature, transform: bật khi cả hai bật, hướng là transform áp lên decoder của feature). Chuỗi nhiều transform cho số pseudofeature tăng theo hàm mũ.

---

## 4. Kết quả

### 4.1 Phân bố hạng tối ưu

Trên Claude 3.5 Haiku, tốt nhất (trên Pareto MSE/L0) là phân bố hạng từ 32 đến 512:

| Hạng $k$ | 512 | 256 | 128 | 64 | 32 |
|---|---|---|---|---|---|
| Số transform | $N$ | $2N$ | $4N$ | $8N$ | $16N$ |

Nhận xét của chúng tôi: mỗi transform hạng $k$ tốn $2kd_{model}$ tham số, nên **mỗi tầng hạng nhận cùng tổng tham số** ($512N\cdot 2d_{model}$). Tác giả cho biết hạng thay đổi tốt hơn hạng đồng nhất khi cùng tổng tham số.

### 4.2 Hiệu năng ML

Huấn luyện trên layer giữa của Haiku, cùng số tham số, số bước tỉ lệ với số feature (mỗi 4× FLOPs = 2× tham số và 2× bước). Transcoder lớn nhất ("1024x FLOPs") có khoảng 10 triệu feature.

- Tại cùng L0, MSE của MOLT **thấp hơn đáng kể**.
- MOLT nhỏ nhất **Pareto-dominate** transcoder dùng **1024× FLOPs**.
- Transcoder có vẻ **bão hòa** ở quy mô lớn (có thể do tuning kém); MOLT chưa thấy bão hòa.

### 4.3 Faithfulness cơ chế qua Jacobian

**Mechanistic faithfulness**: thay thế có phản ứng với nhiễu đầu vào giống MLP gốc không. Với nhiễu vô cùng nhỏ, so Jacobian:

$$
\rho=\mathbb E_{\mathbf x}\Big[\cos\big(\operatorname{vec}J_{\text{replace}}(\mathbf x),\ \operatorname{vec}J_{\text{MLP}}(\mathbf x)\big)\Big].
$$

- MOLT có Jacobian correlation **cao hơn đáng kể** tại cùng L0.
- Faithfulness của transcoder **giảm khi scale**, của MOLT ổn định hơn.
- Lý giải: Jacobian của transcoder có hạng $\le L_0$, còn MOLT có thể có hạng lớn hơn nhiều. (Kết quả Jacobian này trên mô hình 18 layer của *Circuit Tracing*, chưa làm trên Haiku.)

### 4.4 Diễn giải transform

- **Điều kiện bật** dễ hiểu tương tự feature: layer sớm chọn thông tin token, layer giữa/muộn chọn ngữ cảnh trừu tượng.
- Transform hạng cao bật thường hơn, nhưng vẫn dễ hiểu (ví dụ một transform bật trên dấu chấm, một transform bật trên văn bản tiếng Tây Ban Nha).
- **Chức năng** khó diễn giải hơn: tính $\mathbf e_t\,UV\,\mathbf d_s$ giữa feature SAE trước và sau cho kết quả khó đọc, có lẽ do **interference weights** (xem note số 23, 24).

---

## 5. MOLT trong attribution graph

Transform không có hướng decoder cố định, nên dùng chiến lược khác:

1. Huấn luyện SAE (hoặc WCC) trên residual mỗi layer.
2. Phân rã attribution giữa feature nguồn $i$ và đích $j$ thành:
   - Qua transform: $\mathbf e_j\,U V\,\mathbf d_i\cdot(a_i a_t)$.
   - Qua attention head: $\mathbf e_j\,OV\,\mathbf d_i\cdot(a_i\,a_h(q,k))$.
3. Gắn mỗi cạnh với danh sách transform và head mang nó mạnh nhất. Transform cũng có cạnh vào (đích là encoder $\mathbf e_t$).

Nói cách khác: **transform là "attention head của MLP"**, đơn vị mang cạnh.

**Ví dụ:**
- `The Spanish word for hot is "calor"`: "say a word beginning with 'cal'" nhận đầu vào từ "hot" qua một transform bật trong ngữ cảnh tiếng Tây Ban Nha, và đồng thời từ "Spanish" qua transform bật trong ngữ cảnh "hot".
- `3 + 5 = 8`: "say 8" nhận đầu vào từ "plus 3" qua một transform bật khi số "5" vừa xuất hiện.
- Không phải lúc nào cũng rõ: nhiều cạnh qua transform chọn từ khóa như "is" hay "Assistant"; có transform thừa với feature nó mang ("France" → "Paris" qua transform "France").

---

## 6. Giới hạn và công trình liên quan

- Kết quả sơ bộ, chưa chứng minh MOLT luôn tốt hơn transcoder; đồ thị có transform cồng kềnh hơn.
- Chức năng transform khó đọc ngoài ngữ cảnh đồ thị.
- Ý tưởng chưa thử: dùng transform để giải thích **sai số tái tạo của SAE**. Sai số ở layer $L+1$ có thể viết lại thành tổng các cặp (feature, transform) của layer trước; có thể là khái niệm quá hiếm để SAE học được, nhưng ghép từ feature và transform phổ biến.
- **Mixture of Decoders** (MxD, Oldfield et al.): rất giống; transform full-rank nhưng chia sẻ tham số (cột $h$ của $U_t$ là $\mathbf d_h\ast\mathbf c_t$). MOLT dùng low-rank. Tác giả nghi low-rank khuyến khích transform có vai trò hẹp, dễ hiểu hơn, nhưng chưa so sánh.
- **Skip transcoder**: transcoder + một phép tuyến tính. MOLT khác ở chỗ transform bật có điều kiện và thưa, không full-rank, có nhiều transform, không có feature transcoder thường.

---

## 7. Ứng dụng giả thuyết cho pruning và distillation

> **Suy luận của chúng tôi, không phải của paper.**

**7.1 Vì sao cắt neuron MLP khó phục hồi: một giải thích kiểu MOLT.** Trực giác (a) nói MLP là **hỗn hợp có điều kiện** của nhiều phép tuyến tính, mỗi phép bật theo ngữ cảnh. Map tuyến tính tối ưu ở điểm cắt (D_lin của bạn) là **một** phép tuyến tính toàn cục, nên chỉ bắt được phần "trung bình" của hỗn hợp. Phần phụ thuộc ngữ cảnh, tức sự khác nhau giữa các transform, rơi vào D_res. Khi cắt cả layer, mất toàn bộ hỗn hợp nhưng thường layer kề bên (đường tắt) làm phần lớn việc tương tự; khi cắt *một phần neuron*, mỗi transform ngầm bị cắt một phần hạng theo cách khác nhau, tạo ra lỗi phân tán, phụ thuộc ngữ cảnh. Giả thuyết kiểm tra được: D_res của neuron-cut phải giảm mạnh nếu cho phép **nhiều** map tuyến tính có cổng (gated) thay vì một.

**7.2 Module phục hồi kiểu MOLT.** Thay vì chỉ học một map $W$ ở điểm cắt, học
$$
\hat r(\mathbf x)=W\mathbf x+\sum_{t=1}^{T}\phi(\mathbf e_t\cdot\mathbf x-b_t)\,U_tV_t\mathbf x
$$
với $T$ nhỏ và hạng thấp. Đây là "D_lin có điều kiện". Chi phí thêm nhỏ nếu $T\cdot k$ nhỏ so với MLP bị cắt. Lưu ý: nếu LightOnOCR dùng SwiGLU (cổng mềm), lập luận "cố định tập neuron hoạt động" chỉ là xấp xỉ.

**7.3 Faithfulness qua Jacobian cho KD.** Hidden-state KD thường so output. Jacobian correlation là thước đo bổ sung: student có thể khớp MSE nhưng phản ứng với nhiễu khác teacher, và đó có thể là dấu hiệu của tổng quát hóa kém trên ảnh lạ (font lạ, nhiễu scan). Đo $\rho$ giữa block của student và block teacher tương ứng (qua projector nếu khác chiều).

**7.4 Phân bố hạng cho low-rank compression.** Kết quả "nhiều transform hạng thấp, ít transform hạng cao, mỗi tầng cùng ngân sách" là một prior hữu ích khi phân rã low-rank MLP theo cụm ngữ cảnh: không nên dùng một hạng chung.

---

## 8. Thí nghiệm gợi ý

**Thí nghiệm 1 — D_res có "tuyến tính từng mảnh" không?** Sau khi cắt 50% neuron một MLP của teacher, fit (a) một map tuyến tính toàn cục, (b) MOLT-lite với $T\in\{4,16,64\}$ transform hạng 16–64 (cùng tổng tham số tăng dần). Vẽ phần dư còn lại theo $T$. Nếu giảm nhanh, D_res chủ yếu là "tuyến tính có điều kiện" và có thể phục hồi rẻ.

**Thí nghiệm 2 — Đếm mẫu kích hoạt.** Trên dữ liệu OCR, ghi mẫu nhị phân của neuron có $|\text{activation}|$ lớn nhất ở một MLP, phân cụm (ví dụ theo Jaccard). Số cụm hiệu dụng và hạng của $W_{out}D_SW_{in}$ trong từng cụm cho biết MLP này là hỗn hợp của bao nhiêu phép tuyến tính, và bao nhiêu chiều mỗi phép.

**Thí nghiệm 3 — Jacobian correlation của student.** Với các block được giữ lại sau pruning + KD, tính $\rho$ so với teacher trên ảnh in-distribution và out-of-distribution (font viết tay, ảnh chụp nghiêng). So sánh mức giảm $\rho$ với mức tăng CER ngoài phân phối.

---

## 9. Tóm tắt công thức và khái niệm

| Khái niệm | Công thức / mô tả | Ý nghĩa |
|---|---|---|
| MOLT | $f(\mathbf x)=\sum_t\phi(\mathbf e_t\cdot\mathbf x-b_t)\,U_tV_t\mathbf x$ | MLP = hỗn hợp thưa phép tuyến tính hạng thấp |
| Sparsity penalty | $\lVert U_tV_t\rVert_F\,\phi(\cdot)$ | Giống penalty theo chuẩn decoder |
| MLP có điều kiện | $W_{out}D_SW_{in}$, hạng $\le\lvert S\rvert$ | Mẫu neuron bật = một transform |
| Phân bố hạng | $N\!\times\!512,\ 2N\!\times\!256,\dots,16N\!\times\!32$ | Cùng ngân sách mỗi tầng hạng |
| Jacobian faithfulness | $\mathbb E[\cos(\operatorname{vec}J_1,\operatorname{vec}J_2)]$ | Phản ứng với nhiễu giống MLP gốc |
| Pseudofeature | (feature, transform) | $10^6\times10^6$ cặp mô tả $10^{12}$ hướng |
| Cạnh qua transform | $\mathbf e_jUV\mathbf d_i\,(a_ia_t)$ | Transform là "head của MLP" |
| Kết quả chính | MOLT nhỏ nhất thắng transcoder 1024× FLOPs | Hiệu quả và faithful hơn |

### Kết luận ngắn

MOLT thay "feature cố định hướng" bằng "phép biến đổi tuyến tính bật theo ngữ cảnh", khớp với bản chất tuyến tính từng mảnh của MLP. Kết quả sơ bộ cho thấy hiệu quả tham số và faithfulness (Jacobian) tốt hơn transcoder nhiều. Với dự án của bạn, trực giác này cho một lời giải thích cụ thể, kiểm tra được, vì sao một map tuyến tính duy nhất ở điểm cắt không đủ bù cho việc cắt neuron, và gợi ý một module phục hồi "tuyến tính có điều kiện".

**Đọc bản gốc:** https://transformer-circuits.pub/2025/bulk-update/index.html
