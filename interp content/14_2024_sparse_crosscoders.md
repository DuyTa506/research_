# Sparse Crosscoders for Cross-Layer Features and Model Diffing (2024)

**Tác giả:** Jack Lindsey\*, Adly Templeton\*, Jonathan Marcus\*, Thomas Conerly\*, Joshua Batson, Christopher Olah (Anthropic; \* core contributor). Công bố 25/10/2024, dạng *research update* (kết quả sơ bộ, chưa đạt chuẩn của paper đầy đủ).  
**Nguồn:** https://transformer-circuits.pub/2024/crosscoders/index.html  
**Chủ đề:** Sparse dictionary learning trên nhiều layer/nhiều model, cross-layer superposition, persistent features, acausal/causal crosscoder, model diffing (base vs finetune, theo scale).  
**Vị trí trong lộ trình:** Nên đọc trước Toy Models of Superposition và Towards Monosemanticity (SAE). Paper này là "cầu nối" dẫn tới cross-layer transcoder trong *Circuit Tracing* (2025, note 17) và tới hai note về model diffing (note 15, 16) — đây là công cụ trực tiếp nhất để so sánh model gốc với student đã prune/distill.

> Đây là ghi chú học tập diễn giải bằng tiếng Việt, không phải bản dịch. Các ứng dụng cho pruning/distillation (mục 10–11) là suy luận mở rộng của chúng tôi, không phải kết quả hay khẳng định của paper.

## Mục lục

1. [Bối cảnh SAE transcoder và crosscoder](#1-bối-cảnh-sae-transcoder-và-crosscoder)
2. [Động cơ cross-layer superposition và persistent features](#2-động-cơ-cross-layer-superposition-và-persistent-features)
3. [Crosscoder cơ bản acausal crosscoder](#3-crosscoder-cơ-bản-acausal-crosscoder)
4. [Các biến thể crosscoder](#4-các-biến-thể-crosscoder)
5. [Kết quả về cross-layer features](#5-kết-quả-về-cross-layer-features)
6. [Model diffing bằng crosscoder](#6-model-diffing-bằng-crosscoder)
7. [Literal model và isomorphic model](#7-literal-model-và-isomorphic-model)
8. [Paper kiểm chứng như thế nào](#8-paper-kiểm-chứng-như-thế-nào)
9. [Giới hạn](#9-giới-hạn)
10. [Ứng dụng giả thuyết cho pruning và distillation](#10-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
11. [Thí nghiệm gợi ý](#11-thí-nghiệm-gợi-ý)
12. [Tóm tắt công thức và khái niệm](#12-tóm-tắt-công-thức-và-khái-niệm)

---

## 1. Bối cảnh SAE transcoder và crosscoder

Thuật ngữ nhanh cho người mới:

- **Feature**: một hướng (direction) trong không gian activation tương ứng với một khái niệm có thể diễn giải; thường *không* trùng với một neuron.
- **Superposition**: mô hình biểu diễn nhiều feature hơn số chiều/neuron, bằng các hướng không trực giao → neuron bị **polysemantic** (đa nghĩa).
- **SAE (sparse autoencoder)**: encode activation $x$ của **một** layer thành vector thưa $f$, rồi decode lại chính $x$.
- **Transcoder**: đọc input của một MLP, dự đoán **output** của MLP đó (thay thế tính toán của MLP).
- **Crosscoder**: đọc và/hoặc ghi **nhiều layer** (thậm chí nhiều model) cùng lúc, với **một** tập feature chung.

| Công cụ | Đọc từ | Tái tạo | Feature "sống" ở |
|---|---|---|---|
| SAE | layer $l$ | layer $l$ | 1 layer |
| Transcoder | input MLP $l$ | output MLP $l$ | 1 layer |
| Crosscoder | tập layer $L$ (có thể nhiều model) | các layer trong $L$ | nhiều layer/model |

Paper xem SAE và transcoder là **trường hợp đặc biệt** của họ crosscoder.

## 2. Động cơ cross-layer superposition và persistent features

### 2.1 Cross-layer superposition

Vì residual stream tuyến tính, hai layer liên tiếp có thể xem như **"hai nhánh gần như song song"**, chỉ khác ở một cạnh cho phép layer trước ảnh hưởng layer sau. Nếu mô hình có nhiều layer hơn độ dài circuit cần tính, một bước tính toán có thể bị **chia ra hai layer nhưng về chức năng là song song**. Khi đó feature được biểu diễn *chung* bởi nhiều layer → hợp lý khi học dictionary *đồng thời* trên các layer đó.

### 2.2 Persistent features và độ phức tạp circuit

Một feature được tính xong có thể **tồn tại** trong residual stream qua nhiều layer. Ví dụ giả định trong paper: feature 1 và 2 có mặt từ layer $L$, được kết hợp (kiểu "AND") bởi MLP ở $L+2$, $L+3$ thành feature 3, và cả ba tồn tại đến $L+4$.

| Cách phân tích | Số feature | Đồ thị nhân quả |
|---|---|---|
| SAE từng layer | 13 (mỗi feature lặp ở mỗi layer nó tồn tại) | Rất nhiều cạnh "tự gây ra chính mình" |
| Crosscoder lý tưởng | 3 | Đơn giản |

Nếu encoder của feature đọc ở **một** layer và decoder ghi ra **các layer phía sau**, ta có thể "nhảy cóc" (hop back): attribution của feature $i$ (layer 10) tới thành phần decode-vào-layer-10 của feature $j$ (encode ở layer 1) được gán thẳng về $j$ ở layer 1, thay vì đi qua chuỗi SAE từng layer. Circuit có thể **nông hơn nhiều** so với số layer. Paper cảnh báo: mô tả nhân quả này có thể khác với mô hình thật (xem mục 7).

## 3. Crosscoder cơ bản acausal crosscoder

### 3.1 Encoder và decoder

Với datapoint $x_j$ và activation $a^l(x_j)$ tại các layer $l\in L$:

$$
\boxed{f(x_j)=\mathrm{ReLU}\Big(\sum_{l\in L}W^{l}_{enc}\,a^{l}(x_j)+b_{enc}\Big)}
$$

$$
\hat a^{l}(x_j)=W^{l}_{dec}\,f(x_j)+b^{l}_{dec}.
$$

Mỗi layer có encoder và decoder riêng, nhưng **vector feature $f$ là chung**. "Acausal" = mọi layer đều đóng góp vào encoder và mọi layer đều được tái tạo, không quan tâm thứ tự thời gian.

### 3.2 Loss

$$
\boxed{\mathcal L=\sum_{l\in L}\big\|a^{l}(x_j)-\hat a^{l}(x_j)\big\|^2+\lambda\sum_i f_i(x_j)\sum_{l\in L}\big\|W^{l}_{dec,i}\big\|}
$$

($W^l_{dec,i}$ là decoder vector của feature $i$ tại layer $l$; trong paper hệ số L1 được cố định, ở đây chúng tôi viết rõ $\lambda$.) Penalty là **L1 của các norm theo layer** ("L1-of-norms").

### 3.3 L1-of-norms hay L2-of-norms

Lựa chọn "tự nhiên" khác là coi mọi decoder như một vector dài: $\sum_i f_i\sqrt{\sum_l\|W^l_{dec,i}\|^2}$ ("L2-of-norms"). Ví dụ nhỏ (chúng tôi tự dựng) với 2 layer:

| Decoder norm theo layer | L1-of-norms | L2-of-norms |
|---|---|---|
| $(1,0)$ — feature chỉ ở layer 1 | 1 | 1 |
| $(1,1)$ — feature trải 2 layer | 2 | $\sqrt2\approx1.41$ |

Với L2, trải feature ra nhiều layer được "giảm giá" → khuyến khích feature dùng chung. Với L1, một feature trải 2 layer tốn bằng hai feature riêng → không có ưu đãi trải rộng. Paper chọn **L1** (cho mọi thí nghiệm trong bài) vì:

1. **So sánh loss công bằng**: loss crosscoder so sánh trực tiếp được với *tổng* loss của các SAE từng layer cùng hệ số L1; L2 sẽ cho crosscoder "bonus" không công bằng.
2. **Lộ ra feature đặc thù**: trong model diffing, L1 cho ra hỗn hợp feature dùng chung **và** feature riêng của từng model; L2 chỉ cho feature dùng chung.

Ngược lại, L2 tối ưu biên (MSE, L0 toàn cục) hiệu quả hơn, nên phù hợp khi không cần feature đặc thù theo layer/model.

## 4. Các biến thể crosscoder

Bốn trục thiết kế: (i) cross-layer hay SAE thường; (ii) **causality** — activation sớm dự đoán activation muộn như transcoder?; (iii) **locality** — áp cho mọi layer hay một số, hay layer của nhiều model?; (iv) **target** — tái tạo residual stream hay output của layer?

| Biến thể | Encoder đọc từ | Decoder ghi tới | Ghi chú |
|---|---|---|---|
| Acausal (global) | mọi layer | mọi layer | dùng cho cross-layer features, model diffing |
| Local/"convolutional" | cửa sổ $K$ layer | cùng cửa sổ | $K=1$ là SAE từng layer |
| Weakly causal | 1 layer $i$ | layer $i$ và mọi layer sau | hứa hẹn cho circuits |
| Strictly causal ("cross-layer transcoder") | residual layer $L$ | output MLP $L, L{+}1,\dots$ | tiền thân của CLT (note 17) |
| Pre/post-MLP | residual trước MLP | cả pre-MLP và output MLP | phân biệt feature "cũ" và "mới tính" |

Paper lưu ý: strictly causal crosscoder **không bắt được tính toán của attention**; các hướng đang thử gồm coi attention là tuyến tính bằng cách cố định attention pattern của prompt, kết hợp với crosscoder weakly causal cho attention, hoặc xây "replacement layer" cho attention.

## 5. Kết quả về cross-layer features

### 5.1 Crosscoder vs SAE từng layer (mô hình 18 layer)

Huấn luyện một acausal crosscoder trên residual stream của **cả 18 layer**, so với **18 SAE riêng**. Activation mỗi layer được chuẩn hóa riêng để đóng góp vào loss tương đương.

- **Chi phí**: với $L$ layer, crosscoder $F$ feature tốn FLOPs huấn luyện bằng $L$ SAE mỗi cái $F$ feature (tổng $L\times F$ feature).
- **Theo số feature**: cố định tổng số feature, crosscoder có eval loss **thấp hơn rõ rệt** → có nhiều cấu trúc dư thừa (tương quan tuyến tính) giữa các layer.
- **Theo FLOPs**: crosscoder **kém hiệu quả hơn khoảng 2 lần** ở compute lớn để đạt cùng eval loss.
- **Theo (MSE, L0) toàn mô hình** (L0 cộng qua mọi SAE): crosscoder **có lợi thế rõ ràng**, vì gộp được feature trùng lặp giữa layer. Về lý thuyết có thể gộp hậu kỳ các feature SAE bằng clustering, nhưng trên thực tế khó do tính ngẫu nhiên của huấn luyện SAE; crosscoder "nướng sẵn" việc gộp này.

### 5.2 Feature trải qua layer như thế nào

- **Decoder norm theo layer**: đa số feature đạt đỉnh ở một layer và suy giảm hai phía; đôi khi giảm đột ngột (feature cục bộ), thường giảm từ từ, nhiều feature có norm đáng kể ở hầu hết hoặc mọi layer.
- **Hướng decoder** ổn định hơn nhiều so với ngẫu nhiên, nhưng vẫn **trôi (feature drift)** đáng kể giữa các layer, kể cả nơi norm còn mạnh → feature không chỉ được "chuyển tiếp thụ động" qua residual connection.
- Khác với Yun et al. (coi cùng feature = cùng hướng qua các layer), crosscoder coi cùng feature = **kích hoạt trên cùng datapoint**, cho phép hướng đổi theo layer.
- Paper thận trọng: hình thành feature từ từ *nhất quán* với cross-layer superposition nhưng cũng có thể do "tạo ở một layer rồi khuếch đại ở layer sau".

### 5.3 Masked crosscoders

- **Locality**: eval loss nội suy gần **tuyến tính** khi tăng cửa sổ $K$ từ 1 lên số layer — không có điểm gãy tối ưu.
- **Weakly causal**: hiệu quả FLOPs nằm giữa SAE từng layer và global crosscoder; theo kích thước dictionary thì kém global crosscoder **3–4 lần**. (Sparse kernel có thể cải thiện đáng kể vì decoder chiếm phần lớn compute.)
- **Strictly causal cross-layer transcoder**: decoder norm cho thấy feature cục bộ (chủ yếu dự đoán MLP kế tiếp), feature toàn cục (dự đoán mọi MLP sau với độ mạnh gần đều), và loại ở giữa.
- **Pre/post-MLP crosscoder**: encoder chỉ đọc pre-MLP; decoder tái tạo cả pre-MLP và output MLP. Phân bố relative decoder norm **ba đỉnh**: chỉ-pre, dùng chung, chỉ-post ("mới tính"). Input của feature mới tính ≈ (encoder của nó) · (decoder của feature upstream) × activation. Giai thoại: feature post-MLP thường trừu tượng hơn, ví dụ một feature cho "special/particular/exceptional" nhận input từ các feature cho từng từ cụ thể. Feature "ổn định" (relative norm 0.3–0.7 ở post-MLP) có hướng pre/post tương quan dương trung bình nhưng phương sai cao và giá trị tuyệt đối thấp → MLP có thể chuyển tiếp nhiều feature mà không biến đổi phi tuyến, nhưng **theo trục khác**.

## 6. Model diffing bằng crosscoder

**Model diffing**: so sánh hai model ở mức cơ chế/feature, giống review code theo diff — ý tưởng được nêu từ OpenAI Clarity Team (2018). Các hướng trước: so sánh *toàn bộ biểu diễn* (model stitching, SVCCA — bất biến tuyến tính nhưng **không** bất biến với cấu trúc superposition khác nhau), so sánh neuron/feature (universality), và chuyển SAE base sang model finetune (Kissane et al.).

### 6.1 Thiết lập (dạng hai model)

Cùng một input $x$, lấy activation $a^A(x)\in\mathbb R^{d_A}$ và $a^B(x)\in\mathbb R^{d_B}$:

$$
f(x)=\mathrm{ReLU}\big(W^{A}_{enc}a^{A}(x)+W^{B}_{enc}a^{B}(x)+b_{enc}\big),\qquad \hat a^{m}=W^{m}_{dec}f+b^{m}_{dec},\ m\in\{A,B\}.
$$

Lưu ý: $d_A$ và $d_B$ **không cần bằng nhau** — encoder/decoder riêng cho mỗi model. Vì thế paper nhấn mạnh crosscoder áp dụng được giữa các model khác kiến trúc, khác scale, khác seed, khác dataset, v.v.

Hai thống kê đọc kết quả (định nghĩa cụ thể là cách chúng tôi viết; paper mô tả bằng lời "relative norm" và "alignment"):

$$
\boxed{r_i=\frac{\|W^{B}_{dec,i}\|}{\|W^{A}_{dec,i}\|+\|W^{B}_{dec,i}\|}},\qquad
c_i=\cos\big(W^{A}_{dec,i},W^{B}_{dec,i}\big)\ \ (\text{chỉ khi }d_A=d_B).
$$

- $r_i\approx0$: feature riêng của A; $r_i\approx1$: riêng của B; $r_i\approx0.5$: dùng chung.
- Feature dùng chung với $c_i$ thấp/âm: "cùng khái niệm, dùng khác đi".

### 6.2 Kết quả: Claude 3 Sonnet vs base model

- Crosscoder **1 triệu feature** trên residual stream **layer giữa** của Claude 3 Sonnet và base model.
- Relative norm phân thành **ba cụm rõ ràng**: riêng base, riêng finetune, dùng chung; mỗi model có khoảng **4–5 nghìn** feature riêng (trên 1 triệu).
- Ví dụ (được chọn lọc, *cherrypicked*): riêng finetune — feature refusal cho yêu cầu nguy hiểm, feature code review, feature "người dùng hỏi Assistant về bản thân"; riêng base — feature LLM "nhập vai" theo system prompt, feature hội thoại với trợ lý smartphone.
- **Đa số** feature riêng **không dễ diễn giải** ngay (note 16 giải thích vì sao).
- Feature dùng chung: hầu hết decoder rất thẳng hàng; nhưng **vài nghìn** feature có tương quan rất thấp hoặc âm — giả thuyết: finetune dùng khái niệm cũ theo cách mới.
- So với cách "dùng SAE base trên model finetune" (Kissane et al.), crosscoder mô tả diff bằng *trừu tượng riêng của mỗi model*; paper đoán crosscoder phù hợp hơn khi finetune chiếm tỉ lệ compute lớn so với pretrain.

### 6.3 Diff theo (layer, scale)

Acausal crosscoder trên **10 layer cách đều** của **3 model** kích thước tăng dần → mỗi feature có vector norm $3\times10=30$ chiều. Áp **NMF** (nonnegative matrix factorization, phân rã ma trận không âm) với 4 thành phần: một thành phần phủ layer sớm của cả ba model, một phủ layer muộn của cả ba, hai thành phần còn lại phủ layer giữa của model nhỏ nhất và của hai model lớn hơn → gợi ý **biểu diễn mới về chất xuất hiện ở layer giữa khi scale tăng**.

## 7. Literal model và isomorphic model

Đi từ neuron → feature SAE/transcoder → crosscoder là đi xa dần mô hình "literal". Với **error features** (phần dư giữa activation thật và tái tạo), có thể coi mô hình trừu tượng là **isomorphism chính xác** (dự đoán đúng activation mọi layer). Nhưng:

- **Error có thể lớn và khó diễn giải** hơn cả activation gốc.
- **Cấu trúc nhân quả có thể bị dời layer**: crosscoder huấn luyện trên từng vị trí token, nên mọi tương quan cross-layer *trong cùng vị trí* được gom thành feature cross-layer, dù trong model thật thông tin có thể được copy tới vị trí đó nhiều lần ở các layer khác nhau. Đồ thị nhân quả của crosscoder có thể khác cơ chế thật → cẩn thận khi dùng cho patching.
- **"Mechanistic faithfulness"** quan trọng nhưng chưa được hình thức hóa; crosscoder có thể khai thác tương quan on-distribution mà không tổng quát hóa.

## 8. Paper kiểm chứng như thế nào

| Loại bằng chứng | Thí nghiệm | Kết quả chính |
|---|---|---|
| Định lượng loss | Crosscoder vs 18 SAE, quét FLOPs & số feature | Tốt hơn theo số feature; kém ~2× theo FLOPs; vượt trội theo (MSE, L0) toàn mô hình |
| Thống kê decoder | Norm & cosine theo layer (50 và 36 feature ngẫu nhiên) | Đỉnh ở một layer, trải rộng, có drift |
| Biến thể | Locality, weakly causal, pre/post-MLP | Nội suy tuyến tính; kém 3–4× theo dictionary size; phân bố 3 đỉnh |
| Diffing định tính | Sonnet vs base | 3 cụm, 4–5k feature riêng/model, ví dụ chọn lọc |
| Diffing theo scale | NMF trên vector 30 chiều | Layer giữa khác nhau theo scale |

Bằng chứng chủ yếu là **phân bố thống kê + ví dụ định tính**; chưa có intervention nhân quả hay đánh giá diễn giải hệ thống.

## 9. Giới hạn

- Kết quả **sơ bộ**; tác giả nói diffing có "dấu hiệu sống" nhưng **nhiều feature riêng không hiểu được**.
- Không đánh giá có hệ thống tính diễn giải (chỉ giai thoại: tương đương feature SAE).
- Crosscoder **đắt** hơn per-layer SAE theo FLOPs.
- Gradual formation chưa chứng minh được cross-layer superposition (cần circuit analysis).
- Đồ thị nhân quả của crosscoder có thể **không faithful** với cơ chế thật; error term khó diễn giải.
- Cosine giữa decoder chỉ có nghĩa khi hai model **cùng không gian** (base/finetune); giữa model khác kiến trúc chỉ còn relative norm và nội dung feature (nhận xét của chúng tôi).

## 10. Ứng dụng giả thuyết cho pruning và distillation

> **Toàn bộ mục này là suy luận của chúng tôi, không phải kết quả paper.** Mục tiêu: dùng crosscoder để "diff" parent LightOnOCR-3-0.8B (P) với student đã prune/distill S (~0.3B hoặc ~0.1B).

### 10.1 Thiết lập cụ thể

1. **Dữ liệu**: cùng ảnh tài liệu + cùng prefix text cho P và S; vì S sinh ra từ P nên tokenizer và số vision token giống nhau (nếu ViT không đổi số patch) → các vị trí token căn chỉnh 1–1. Trộn dữ liệu OCR thật (bảng, công thức, chữ Việt có dấu, chữ viết tay) và text chung.
2. **Cặp layer**: nếu S cắt layer, không có "layer tương ứng" rõ ràng. Paper gợi ý: khi không biết layer tương đương, cho crosscoder **trải qua nhiều layer** của cả hai model. Ví dụ: 4 layer cách đều của decoder P + 4 layer của decoder S, acausal.
3. **Kích thước khác nhau**: $d_P\neq d_S$ không phải vấn đề — encoder/decoder riêng cho mỗi model.
4. **Loss**: dùng **L1-of-norms** (mục 3.3), nếu không sẽ chỉ thấy feature dùng chung; cân nhắc thêm tập "designated shared features" của note 16 để feature riêng bớt polysemantic.
5. **Đọc kết quả**: histogram $r_i$; cho feature chung, vì cosine không xác định khi $d_P\neq d_S$, dùng map tuyến tính tối ưu $M:\mathbb R^{d_S}\to\mathbb R^{d_P}$ (chính map dùng để đo $D_{lin}$) và tính

$$
\tilde c_i=\cos\big(W^{P}_{dec,i},\,M\,W^{S}_{dec,i}\big).
$$

### 10.2 Diễn giải gợi ý cho $D_{lin}$ / $D_{res}$

| Loại feature | Dấu hiệu | Giả thuyết liên hệ |
|---|---|---|
| Chung, $\tilde c_i$ cao | $r_i\approx0.5$ | Student giữ khái niệm; sai lệch nếu có thì map tuyến tính sửa được → thuộc $D_{lin}$ |
| Chung, $\tilde c_i$ thấp | $r_i\approx0.5$ | Student giữ khái niệm nhưng "xoay" phi tuyến → có thể góp vào $D_{res}$ |
| Riêng P | $r_i\approx0$ | Khái niệm bị mất → ứng viên chính của $D_{res}$ |
| Riêng S | $r_i\approx1$ | Feature bù trừ hoặc artifact; theo note 16 thường dày đặc/đa nghĩa — kiểm tra kỹ |

Một proxy định lượng: phần năng lượng activation P giải thích bởi feature riêng P, $\sum_{i:r_i<0.05}\|f_iW^P_{dec,i}\|^2/\|a^P\|^2$, rồi so với $D_{res}$ đo trực tiếp. Nếu tương quan tốt trên từng loại dữ liệu, ta có một "danh sách khái niệm bị mất" thay vì một con số.

### 10.3 Neuron cut vs layer cut

Paper cho thấy nhiều feature có norm đáng kể ở **nhiều layer** và hướng trôi dần. Giả thuyết giải thích quan sát của bạn: cắt **layer** bỏ một "nhánh gần song song" mà feature cross-layer còn được các layer lân cận cung cấp → ít feature bị mất; cắt **neuron MLP** ở mọi layer cắt vào các hướng superposition mà nhiều feature cùng dùng → nhiều feature riêng P. Có thể kiểm chứng: số lượng và năng lượng feature riêng P của student cắt-layer vs student cắt-neuron cùng số tham số.

### 10.4 Những gì cần cảnh giác

- OCR chủ yếu là **copy** từ vision token qua attention; crosscoder trên residual cho biết *cái gì* bị mất, không cho biết *cơ chế attention* nào hỏng.
- Với decoder lai GatedDeltaNet + full attention, crosscoder chỉ đọc residual stream nên không phụ thuộc loại mixer; nhưng feature mất có thể do state của linear attention, không do MLP.
- Feature riêng S dày đặc có thể chỉ là artifact của cạnh tranh dung lượng (note 16), đừng vội diễn giải.

## 11. Thí nghiệm gợi ý

1. **Diff P vs S tại decoder giữa**: crosscoder 2 model (vài chục nghìn feature), vẽ histogram $r_i$, kiểm tra có 3 đỉnh không; liệt kê top activation của feature riêng P trên tập OCR — có tập trung ở dấu tiếng Việt, ký hiệu toán, bảng không?
2. **Layer-cut vs neuron-cut**: hai student cùng số tham số; so số feature riêng P, năng lượng của chúng và $D_{res}$ đo được.
3. **Persistence trên parent**: acausal crosscoder trên nhiều layer chỉ của P; feature có profile norm rộng gợi ý layer "dư"; dự đoán layer nào bỏ được rồi đối chiếu với kết quả prune thật.

## 12. Tóm tắt công thức và khái niệm

| Khái niệm | Công thức / định nghĩa | Ý nghĩa |
|---|---|---|
| Encoder crosscoder | $f=\mathrm{ReLU}(\sum_l W^l_{enc}a^l+b_{enc})$ | Một vector feature chung cho mọi layer/model |
| Decoder | $\hat a^l=W^l_{dec}f+b^l_{dec}$ | Mỗi layer có decoder riêng |
| Loss L1-of-norms | $\sum_l\|a^l-\hat a^l\|^2+\lambda\sum_i f_i\sum_l\|W^l_{dec,i}\|$ | So sánh được với SAE; lộ feature riêng |
| L2-of-norms | $\lambda\sum_i f_i\sqrt{\sum_l\|W^l_{dec,i}\|^2}$ | Hiệu quả MSE/L0 hơn, chỉ cho feature chung |
| Relative norm | $r_i=\|W^B_{dec,i}\|/(\|W^A_{dec,i}\|+\|W^B_{dec,i}\|)$ | 0/0.5/1 = riêng A/chung/riêng B |
| Decoder cosine | $\cos(W^A_{dec,i},W^B_{dec,i})$ | Chung nhưng dùng khác nếu thấp |
| Feature drift | hướng decoder đổi theo layer | Không chỉ chuyển tiếp thụ động |
| Weakly causal | encode ở 1 layer, decode layer đó và sau | Đơn giản hóa circuit |
| Cross-layer transcoder | residual $L$ → output MLP $L..$ | Tiền thân CLT |
| Error feature | $a^l-\hat a^l$ | Làm isomorphism chính xác, khó diễn giải |

### Kết luận ngắn

Crosscoder mở rộng dictionary learning từ "một layer" sang "nhiều layer/nhiều model với một tập feature chung". Hai hệ quả: (1) gom feature tồn tại qua nhiều layer, làm circuit nông và gọn hơn, mở đường cho cross-layer transcoder; (2) model diffing, tách feature chung và feature riêng. Kết quả còn sơ bộ, nhiều feature riêng khó hiểu, và tính faithful còn bỏ ngỏ. Với bài toán prune/distill, crosscoder là cách tự nhiên nhất để biến "mất bao nhiêu" thành "mất **cái gì**", miễn là kiểm chứng bằng can thiệp.

**Đọc bản gốc:** https://transformer-circuits.pub/2024/crosscoders/index.html
