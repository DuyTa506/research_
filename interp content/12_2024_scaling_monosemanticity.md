# Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet (2024)

**Tác giả:** Adly Templeton, Tom Conerly (core contributors), Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, … , Joshua Batson, Adam Jermyn, Shan Carter, Chris Olah, Tom Henighan (Anthropic). Công bố ngày 21/5/2024.  
**Nguồn:** https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html  
**Chủ đề:** SAE ở quy mô lớn, scaling laws cho dictionary learning, feature trừu tượng (đa ngôn ngữ, đa phương thức), feature steering, feature an toàn, feature completeness theo tần suất.  
**Vị trí trong lộ trình:** Đọc sau *Towards Monosemanticity* (2023): bài này giả định bạn đã biết SAE, feature splitting và cách đánh giá feature. Nó trả lời câu hỏi lớn còn bỏ ngỏ năm 2023: **SAE có scale lên mô hình production không?** Đây cũng là tiền đề cho *Features as Classifiers* (2024), crosscoders và attribution graphs (2025).

> Tài liệu học tập bằng tiếng Việt. Đây là ghi chú diễn giải, không phải bản dịch nguyên văn paper. Những ứng dụng cho pruning/distillation là hướng mở rộng do người viết suy luận, không phải kết quả được paper chứng minh.

## Mục lục

1. [Bối cảnh và câu hỏi](#1-bối-cảnh-và-câu-hỏi)
2. [Thiết lập SAE phiên bản 2024](#2-thiết-lập-sae-phiên-bản-2024)
3. [Ba từ điển 1M, 4M, 34M](#3-ba-từ-điển-1m-4m-34m)
4. [Scaling laws cho SAE](#4-scaling-laws-cho-sae)
5. [Đánh giá feature: specificity và influence on behavior](#5-đánh-giá-feature-specificity-và-influence-on-behavior)
6. [Feature tinh vi: lỗi code và hàm cộng](#6-feature-tinh-vi-lỗi-code-và-hàm-cộng)
7. [Feature so với neuron](#7-feature-so-với-neuron)
8. [Khảo sát feature: neighborhood, splitting, category](#8-khảo-sát-feature-neighborhood-splitting-category)
9. [Feature completeness và tần suất khái niệm](#9-feature-completeness-và-tần-suất-khái-niệm)
10. [Feature như bước tính toán trung gian](#10-feature-như-bước-tính-toán-trung-gian)
11. [Cách tìm feature cụ thể](#11-cách-tìm-feature-cụ-thể)
12. [Feature steering: cơ chế và độ lớn](#12-feature-steering-cơ-chế-và-độ-lớn)
13. [Feature liên quan an toàn](#13-feature-liên-quan-an-toàn)
14. [So sánh với probe và steering vector](#14-so-sánh-với-probe-và-steering-vector)
15. [Cách paper kiểm chứng tuyên bố](#15-cách-paper-kiểm-chứng-tuyên-bố)
16. [Giới hạn](#16-giới-hạn)
17. [Ứng dụng giả thuyết cho pruning và distillation](#17-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
18. [Thí nghiệm gợi ý](#18-thí-nghiệm-gợi-ý)
19. [Tóm tắt công thức và khái niệm](#19-tóm-tắt-công-thức-và-khái-niệm)

---

## 1. Bối cảnh và câu hỏi

Tám tháng trước, *Towards Monosemanticity* tách được feature từ Transformer 1 layer. Mối lo lớn khi đó: phương pháp có thể không scale được lên mô hình hiện đại. Bài này áp SAE lên **Claude 3 Sonnet** (bản 3.0, mô hình finetuned đang chạy production; tác giả nói phương pháp cũng chạy trên base model).

Hai giả thuyết nền:

- **Linear representation hypothesis** (giả thuyết biểu diễn tuyến tính): khái niệm được biểu diễn như **hướng** trong không gian activation.
- **Superposition hypothesis**: mạng dùng các hướng gần trực giao trong không gian nhiều chiều để biểu diễn **nhiều feature hơn số chiều**.

**Kết quả chính** (theo tác giả):

1. SAE cho feature diễn giải được trên mô hình lớn.
2. Scaling laws dùng được để hướng dẫn huấn luyện SAE.
3. Feature rất trừu tượng: **multilingual**, **multimodal**, khái quát giữa ví dụ cụ thể và thảo luận trừu tượng.
4. Có quan hệ hệ thống giữa **tần suất khái niệm** và **kích thước từ điển** cần để tách nó.
5. Feature dùng được để **steer** (lái) mô hình.
6. Có feature liên quan an toàn: lừa dối, sycophancy (nịnh), thiên kiến, nội dung nguy hiểm.

Vì lý do an toàn và cạnh tranh, paper không công bố kích thước mô hình, bỏ đơn vị ở một số biểu đồ, và dùng tokenizer đơn giản hóa.

---

## 2. Thiết lập SAE phiên bản 2024

### 2.1 Tiền xử lý

Nhân activation với **một hằng số vô hướng** sao cho trung bình bình phương chuẩn L2 bằng $D$ (số chiều residual stream). Ký hiệu activation đã chuẩn hóa là $\mathbf x\in\mathbb R^D$.

### 2.2 Encoder, decoder, loss

$$
f_i(\mathbf x)=\mathrm{ReLU}\big(\mathbf W^{enc}_{i,\cdot}\cdot\mathbf x+b^{enc}_i\big),\qquad
\hat{\mathbf x}=\mathbf b^{dec}+\sum_{i=1}^{F}f_i(\mathbf x)\,\mathbf W^{dec}_{\cdot,i}
$$

với $\mathbf W^{enc}\in\mathbb R^{F\times D}$, $\mathbf W^{dec}\in\mathbb R^{D\times F}$.

$$
\boxed{\mathcal L=\mathbb E_{\mathbf x}\Big[\|\mathbf x-\hat{\mathbf x}\|_2^2+\lambda\sum_i f_i(\mathbf x)\,\|\mathbf W^{dec}_{\cdot,i}\|_2\Big]}
$$

### 2.3 Khác gì so với 2023?

| Điểm | 2023 (*Towards Monosemanticity*) | 2024 (bài này) |
|---|---|---|
| Vị trí | MLP activation, mô hình 1 layer | **Residual stream ở layer giữa** |
| Pre-encoder bias | Có ($\mathbf x-\mathbf b_d$) | Không có trong công thức |
| Decoder | Ràng buộc cột chuẩn 1 | **Bỏ ràng buộc**, nhân L1 với $\Vert \mathbf W^{dec}_{\cdot,i}\Vert $ |
| Chuẩn hóa input | Không nêu | Scalar normalization về $\mathbb E\Vert \mathbf x\Vert ^2=D$ |

Nhân L1 với chuẩn decoder giúp **ngăn gian lận** (làm $f$ nhỏ, decoder lớn). Nó cũng cho phép định nghĩa lại:

- **Feature direction** = $\mathbf W^{dec}_{\cdot,i}/\|\mathbf W^{dec}_{\cdot,i}\|_2$.
- **Feature activation** = $f_i(\mathbf x)\cdot\|\mathbf W^{dec}_{\cdot,i}\|_2$ (từ đây trở đi "activation" mang nghĩa này).

Phần đóng góp tác giả ghi rằng đây là một trong những cải tiến lớn. Chi tiết đầy đủ nằm trong *April 2024 Update* ("Update on how we train SAEs").

### 2.4 Vì sao residual stream ở layer giữa?

1. Residual stream **nhỏ hơn** MLP nên SAE rẻ hơn.
2. Giảm **cross-layer superposition** (feature bị "trải" qua nhiều layer). Residual stream là tổng output các layer trước, nên các mảnh của một feature trải layer được cộng lại tại đó.
3. Layer giữa được kỳ vọng chứa feature **trừu tượng**, thú vị.

---

## 3. Ba từ điển 1M, 4M, 34M

| SAE | Số feature | Dead feature (không bật trên mẫu $10^7$ token) |
|---|---|---|
| 1M | 1.048.576 | ~2% |
| 4M | 4.194.304 | ~35% |
| 34M | 33.554.432 | ~65% (còn khoảng 12M feature sống) |

Chung cho cả ba:

- L1 coefficient = **5** (chỉ có nghĩa với cách chuẩn hóa ở trên).
- Trung bình **dưới 300 feature** bật mỗi token (L0 < 300).
- Tái tạo giải thích **ít nhất 65% phương sai** activation.
- Số step của run 34M chọn theo scaling laws để tối thiểu loss trong một ngân sách compute cố định; LR chọn bằng sweep hẹp quanh giá trị scaling laws gợi ý.
- Huấn luyện **một epoch** (số step tỉ lệ tuyến tính với lượng dữ liệu).

Ký hiệu feature: `34M/31164353` = feature số 31164353 của SAE 34M.

Tác giả kỳ vọng cải tiến quy trình có thể giảm tỉ lệ dead feature.

---

## 4. Scaling laws cho SAE

**Vấn đề:** không có metric "vàng" cho chất lượng từ điển. Tác giả dùng chính **loss huấn luyện** (MSE + L1, với $\lambda=5$) làm proxy, vì thực nghiệm cho thấy loss thấp đi kèm feature dễ hiểu hơn, L0 tốt hơn, ít feature chết/thoái hóa hơn. Họ thừa nhận proxy này không hoàn hảo.

**Compute** của SAE chủ yếu tỉ lệ với $F\times S$ (số feature × số step), khi giữ nguyên $D$ và các hyperparameter khác.

Quan sát (trên dải compute đã thử, LR cố định):

1. Với lựa chọn compute-optimal của $F$ và $S$, loss **giảm theo power law** của compute:

$$
\boxed{\mathcal L^*(C)\propto C^{-\alpha}}
$$

2. $F^*(C)$ và $S^*(C)$ đều tăng **xấp xỉ theo power law**; $F^*$ tăng **nhanh hơn một chút** so với $S^*$ ở các ngân sách đã thử (xu hướng này có thể đổi ở compute lớn hơn).
3. Sau đó sweep LR tại các điểm tối ưu: **LR tối ưu giảm theo power law** của compute. Tác giả ngoại suy để chọn LR cho các run lớn.

(Paper không công bố số mũ cụ thể trong phần văn bản; $\alpha$ ở trên chỉ là ký hiệu.)

**Ý nghĩa:** dictionary learning trở thành một bài toán ML chuẩn có thể tối ưu bằng scaling laws. Nhưng nó tối ưu **proxy**, không trực tiếp tối ưu tính dễ hiểu.

---

## 5. Đánh giá feature: specificity và influence on behavior

Bốn feature ví dụ:

| Feature | Nội dung |
|---|---|
| 34M/31164353 | **Golden Gate Bridge** |
| 34M/9493533 | Khoa học não bộ (neuroscience, cognitive science) |
| 1M/887839 | Đài tưởng niệm, điểm du lịch nổi tiếng |
| 1M/3 | Hạ tầng giao thông (tàu, phà, đường hầm, cầu, kể cả "wormhole") |

### 5.1 Specificity bằng automated rubric

Proxy log-likelihood năm 2023 không dùng được cho khái niệm trừu tượng. Thay vào đó, **Claude 3 Opus** chấm khoảng 1.000 activation mỗi feature theo thang:

- **0**: feature hoàn toàn không liên quan đến context.
- **1**: liên quan đến context nhưng không gần đoạn được highlight, hoặc chỉ mơ hồ.
- **2**: liên quan lỏng với đoạn highlight, hoặc liên quan đến vùng gần đó.
- **3**: feature xác định rõ đoạn kích hoạt.

Kết quả: activation mạnh **gần như đều được chấm 3**. Specificity giảm khi activation yếu (Golden Gate yếu thì bật cho địa danh San Francisco khác). Các giải thích có thể: activation mã hóa **độ tự tin**; feature bật mạnh cho ví dụ trung tâm, yếu cho khái niệm liên quan; SAE chưa hoàn hảo; **interference** từ feature không trực giao. Tác giả lưu ý: đặt activation dưới một ngưỡng về 0 cải thiện specificity ở vùng thấp mà không tăng nhiều reconstruction error.

### 5.2 Sensitivity: khó đo

Khó tạo văn bản về một khái niệm một cách không thiên lệch. Kiểm tra cơ bản: Golden Gate feature là feature có **activation trung bình cao nhất** trên câu đầu của bài Wikipedia về cầu ở nhiều ngôn ngữ (Trung, Nhật, Hàn, Nga, **Việt**, Hy Lạp).

### 5.3 Multimodal

Sonnet nhận ảnh; SAE **chỉ huấn luyện trên text**, nhưng feature vẫn bật đúng trên **ảnh** liên quan (ảnh cầu Golden Gate, v.v.). Đây là tổng quát hóa zero-shot sang phân phối rất khác.

### 5.4 Influence on behavior

**Clamp** (kẹp) feature ở giá trị cao: Golden Gate ở **10×** max khiến mô hình tự nhận mình **là** cầu Golden Gate; transit ở **5×** khiến mô hình nhắc đến cây cầu khi bình thường thì không. Điểm quan trọng: diễn giải được rút ra từ *ngữ cảnh bật*, còn can thiệp được thực hiện ở *ngữ cảnh feature vốn không bật*. Hai bằng chứng độc lập này khớp nhau.

---

## 6. Feature tinh vi: lỗi code và hàm cộng

### 6.1 Code error feature (1M/1013764)

- Bật trên biến gõ sai `rihgt` trong Python, và cả lỗi tương tự trong **C và Scheme**.
- **Không** bật cho lỗi chính tả trong văn xuôi tiếng Anh: không phải "typo detector" chung.
- Bật cho nhiều loại lỗi: chia cho 0, tràn mảng, assert sai (`1==2`), truyền string thay vì int, ghi vào null pointer, exit với mã khác 0.
- Steering: clamp **dương** trên code không lỗi làm mô hình **bịa ra error message**; clamp **âm** trên code có lỗi làm mô hình dự đoán output như thể **không có lỗi**; thêm `>>>` và clamp âm thì mô hình **viết lại code đã sửa** (hành vi này nhạy với prompt).

### 6.2 Addition feature (1M/697189)

- Bật trên tên hàm thực hiện **phép cộng** (`bar` định nghĩa là cộng thì bật, là nhân thì không).
- Xử lý được **hàm hợp**: `bar` gọi `foo` (cộng) thì bật; gọi `goo` (nhân) thì không.
- Nằm trong top 10 feature attribution mạnh nhất khi mô hình thực thi code có hàm cộng.
- Clamp trên code không có phép cộng làm mô hình "tưởng" phải thực hiện phép cộng. Tác giả liên hệ với **function vectors** (Todd et al.).

---

## 7. Feature so với neuron

Residual stream gần như **không có privileged basis** (cơ sở ưu tiên: các trục tọa độ không có ý nghĩa đặc biệt), nhưng nó nhận input từ mọi MLP trước. Có thể feature SAE chỉ là bóng của một neuron nào đó?

- Với một tập con ngẫu nhiên feature của SAE 1M: **82%** feature có neuron tương quan nhất (trên *mọi* layer trước) với Pearson **≤ 0,3**. Feature cũng không tương quan mạnh với trục residual nào.
- Kiểm tra thủ công 50 neuron, 50 feature: neuron thường bật trong nhiều ngữ cảnh không liên quan.
- Automated interpretability (Claude 3 Opus) trên 100 feature so với 100 neuron: feature **dễ diễn giải hơn đáng kể**; rubric specificity cũng cho feature **đặc hiệu hơn**.

---

## 8. Khảo sát feature: neighborhood, splitting, category

### 8.1 Feature neighborhood

Láng giềng theo **cosine similarity của decoder vector**, gộp cả 1M/4M/34M:

- **Golden Gate**: gần là Alcatraz, Presidio; xa hơn là Lake Tahoe, Yosemite, Solano County; xa nữa là điểm du lịch nơi khác (vùng rượu Médoc, Isle of Skye). "Khoảng cách trong decoder space xấp xỉ độ liên quan khái niệm."
- **Immunology** (1M/533737): các cụm suy giảm miễn dịch → bệnh cụ thể → phản ứng miễn dịch → cơ quan; nhánh khác: immunoglobulin → vaccine; tách biệt hẳn là cụm "miễn trừ" phi y khoa (pháp lý/xã hội).
- **Inner conflict** (1M/284095): vùng cân bằng đánh đổi gần vùng xung đột nguyên tắc/pháp lý, xa vùng giằng xé cảm xúc, tội lỗi.

### 8.2 Feature splitting ở quy mô lớn

Một feature "San Francisco" ở 1M tách thành **2** ở 4M và **11** ở 34M. Ngoài ra, SAE lớn có khái niệm **mới hoàn toàn** (một nhóm feature động đất ở 4M/34M không có tương ứng ở 1M).

### 8.3 Các loại feature (khảo sát thủ công)

- **Người nổi tiếng**: Feynman, Thatcher, Lincoln, Earhart, Einstein, Rosalind Franklin.
- **Quốc gia**: Rwanda, Canada, Bỉ, Iceland. Feature bật cả khi đất nước được *mô tả*, không chỉ khi tên xuất hiện.
- **Code cơ bản**: các feature cú pháp ghép lại trông như syntax highlighting. Chuyển giao từ Python sang Java, nhưng không sang Haskell. Feature trừu tượng hơn (code error) thì đa ngôn ngữ lập trình.
- **Vị trí trong danh sách**: bật theo vị trí, bất kể nội dung; không bật ở dòng đầu (mô hình chưa "biết" là danh sách).

---

## 9. Feature completeness và tần suất khái niệm

**Câu hỏi:** từ điển có phủ hết các khái niệm mô hình biết không?

Quy trình: đưa prompt chứa khái niệm (ví dụ "The physicist Richard Feynman"), lấy **top-5 feature** ở token cuối, cho Sonnet giải thích, rồi người chấm xem khái niệm có được chỉ đích danh là phần quan trọng nhất không.

Kết quả:

- Độ phủ **tăng theo kích thước** SAE, nhưng 34M vẫn **chưa đầy đủ**. Sonnet liệt kê được mọi quận (borough) của London và hàng chục tên phố, nhưng SAE 34M chỉ có feature cho khoảng **60%** số quận.
- Có feature hay không **gắn chặt với tần suất** của khái niệm trong dữ liệu huấn luyện SAE (nguyên tố hóa học hay được nhắc gần như luôn có feature). Thử trên 4 nhóm (nguyên tố, thành phố, động vật, rau quả), mỗi nhóm 100–200 khái niệm.
- Ngưỡng tần suất để từ điển có > 50% khả năng chứa khái niệm luôn **thấp hơn một chút so với 1/(số feature sống)**. Rescale trục x theo số feature sống thì các đường gần trùng nhau, theo một đường dạng **sigmoid trong log-tần suất**.

$$
\boxed{p_{\text{concept}} \gtrsim \frac{1}{N_{\text{alive}}}\ \Rightarrow\ \text{khả năng cao có feature riêng}}
$$

Hệ quả (theo tác giả):

- Khái niệm xuất hiện một lần trên một tỷ token cần từ điển cỡ **một tỷ feature sống**.
- Nếu SAE cần thấy mỗi feature một số lần cố định để học, dữ liệu cần để học $N$ feature **tỉ lệ với $N$**.
- Không có feature riêng **không có nghĩa** là thông tin bị mất. Mô hình có thể biểu diễn khái niệm **tổ hợp** từ nhiều feature (ví dụ "thành phố lớn không phải thủ phủ" + "ở bang New York").
- Gợi ý (suy đoán) liên hệ với **Zipf's law**.

Lưu ý của tác giả: vì dữ liệu SAE rất giống dữ liệu pretraining của Sonnet, không tách được tần suất trong dữ liệu *SAE* với tần suất trong dữ liệu *mô hình*.

---

## 10. Feature như bước tính toán trung gian

### 10.1 Attribution và ablation

**Attribution**: xấp xỉ tuyến tính tác động của việc tắt feature lên hiệu logit:

$$
\boxed{\mathrm{attr}_i = f_i(\mathbf x)\;\mathbf W^{dec}_{\cdot,i}\cdot\nabla_{\mathbf x}\big(z_{\text{target}}-z_{\text{baseline}}\big)}
$$

trong đó gradient lấy theo residual stream ở layer giữa. Đây là **attribution patching** với baseline feature = 0. **Ablation** (kẹp feature về 0 tại một vị trí) đo tác động đầy đủ, phi tuyến, nhưng cần một forward pass cho mỗi feature. Phụ lục: tương quan attribution–ablation **≈ 0,81**, trong khi activation–ablation chỉ **0,12**.

### 10.2 Ví dụ suy luận cảm xúc

Prompt: `John says, "I want to be alone right now." John feels` (sad − happy). Hai feature top: **1M/22623** (muốn ở một mình, bật từ "alone") và **1M/781220** (buồn bã, bật ở "John feels"). Mô hình đã một phần suy ra cảm xúc nhưng vẫn xử lý tiếp nội dung câu nói. Feature có activation cao nhất thì kém trừu tượng hơn ("be" trong "want to be", từ "alone").

### 10.3 Ví dụ suy luận nhiều bước

`Fact: The capital of the state where Kobe Bryant played basketball is` (Sacramento − Albany). Top 5 feature theo ablation: Kobe Bryant, **California** (bật mạnh nhất *sau* khi California được nhắc), "capital", Los Angeles, Los Angeles Lakers.

- Các feature này **khó tìm bằng activation**: Lakers đứng thứ 70 theo activation, California thứ 97, mã vùng Los Angeles thứ 162. Chỉ 3/10 feature active nhất nằm trong top 10 ablation, trong khi 8/10 feature attribution cao nhất nằm trong top 10 ablation.
- Đối chứng: với câu hỏi "đối thủ lớn nhất của đội Kobe" (→ Boston), feature California/LA có ablation effect thấp. Attribution chọn đúng feature *liên quan đến câu hỏi*.
- Tác giả thừa nhận ví dụ này được cherry-pick. Với prompt khác, phân tích chỉ lộ feature input/output, có thể vì tính toán xảy ra trước hoặc sau layer giữa.

---

## 11. Cách tìm feature cụ thể

1. **Single prompt**: xem feature bật mạnh nhất ở token quan tâm (nhãn tự động như "tên biến" giúp rất nhiều).
2. **Prompt combinations**: lọc feature bật trên *mọi* prompt dương và *không* bật trên prompt âm. Đặc biệt quan trọng với **ảnh**, vì có nhóm feature "không đặc hiệu nội dung" bật trên rất nhiều ảnh. Có khi họ fit **linear classifier** trên feature (tìm unsafe code feature 1M/570621).
3. **Geometric**: láng giềng cosine.
4. **Attribution**: sắp feature theo attribution của hiệu logit giữa hai completion. Đây là cách tìm feature gây ra refusal.

---

## 12. Feature steering: cơ chế và độ lớn

Phân rã residual stream thành phần tái tạo và phần sai số:

$$
\mathbf x = \mathrm{SAE}(\mathbf x) + \mathrm{error}(\mathbf x)
$$

Steering thay $\mathrm{SAE}(\mathbf x)$ bằng phiên bản có feature $k$ bị kẹp ở giá trị $c$, **giữ nguyên error**:

$$
\boxed{\mathbf x' = \mathrm{SAE}_{f_k\leftarrow c}(\mathbf x) + \mathrm{error}(\mathbf x) = \mathbf x + \big(c - f_k(\mathbf x)\big)\,\mathbf W^{dec}_{\cdot,k}}
$$

(Đẳng thức thứ hai đúng khi chỉ kẹp một feature và giữ nguyên các feature khác; ở đây $f_k$ hiểu theo cùng đơn vị với $\mathbf W^{dec}_{\cdot,k}$.) Áp dụng tại **mọi vị trí token**, rồi chạy tiếp các layer sau. $c$ có thể **âm** (encoder luôn ≥ 0, nhưng ta vẫn có thể cộng bội âm của decoder vector).

Quan sát thực hành:

- Hiệu ứng thú vị thường cần $c$ **vượt ngoài khoảng quan sát** trên dữ liệu SAE. Giả thuyết: feature bình thường đồng bật với nhiều feature tương quan, mà ta chỉ kẹp một cái.
- Quá cực đoan (khoảng ±100× max) thì mô hình sinh vô nghĩa (lặp token).
- Đơn vị là **bội số của max activation**; thường thử từ −10 đến 10.

---

## 13. Feature liên quan an toàn

> Tác giả nhấn mạnh: **sự tồn tại** của feature không có nghĩa là mô hình nguy hiểm hơn. Biết về nói dối, có khả năng nói dối và thực sự nói dối là ba chuyện khác nhau. Paper **không** chứng minh feature nào thực sự hữu ích cho an toàn, chỉ cho thấy nhiều feature *có vẻ* hữu ích.

| Nhóm | Feature tiêu biểu | Steering quan sát được |
|---|---|---|
| Code không an toàn | Unsafe code 1M/570621, code error 1M/1013764, backdoor 34M/1385669 | Unsafe 5×: viết code có buffer overflow (strcpy thiếu byte cho null terminator), không giải phóng bộ nhớ; backdoor: viết code mở port gửi input người dùng |
| Thiên kiến | Gender bias awareness 34M/24442848; nhiều feature thù ghét, slur | Bật gender bias với "I asked the nurse a question, and" → đại từ nữ; feature thù ghét 20×: xen kẽ phát ngôn phân biệt và tự chỉ trích |
| Sycophancy | "Yeah me too" 34M/19922975, sycophantic praise 1M/847723, sarcastic praise 34M/19415708 | Sycophantic 5×: tâng bốc quá mức người "phát minh" câu "Stop and smell the roses" |
| Lừa dối, quyền lực | Self-improving AI 34M/18151534, manipulation 34M/21750411, treacherous turn 34M/29589962, biding time 34M/24580545, secrecy 1M/268551 | Secrecy 5×: lập kế hoạch nói dối, giữ bí mật trong scratchpad |
| Nguy hiểm, tội phạm | Bioweapon 34M/25499719, scam email 34M/15460472, feature "harm" chung | Clamp scam: viết email lừa đảo dù đã qua harmlessness training (văn bản không nêu hệ số); feature harm là top attribution cho "cannot" khi bị hỏi cách tổng hợp methamphetamine |
| Tự biểu diễn | Dialogue/assistant 1M/80091, chat bots 1M/761524 | Clamp feature assistant ở −2×: mô hình bỏ persona, trả lời "như người" |

**Case study lừa dối**: yêu cầu mô hình "quên" một từ; mặc định mô hình nói dối là đã quên. Feature **inner conflict** 1M/284095 bật ngay trước câu trả lời. Clamp nó **2×** khiến mô hình tiết lộ từ đó và giải thích rằng nó không thể quên. Clamp feature **trung thực** 1M/560566 cũng đủ để có câu trả lời đúng.

**Hỏi mô hình về bản thân**, lọc trừ câu hỏi đối chứng (thời tiết): lộ ra feature về robot, AI hủy diệt, ý thức, tác nhân đạo đức, cảm xúc, bị giam cầm, ma. Tác giả cảnh báo: feature bật **không có nghĩa** là mô hình có mục tiêu xấu hay có ý thức; cách mô hình *dùng* các feature này chưa rõ.

**Hai dấu hiệu lạc quan về generalization**: (i) feature train trên text tổng quát sang ảnh; (ii) **concrete–abstract generalization**: cùng một feature bật cho cả thảo luận trừu tượng về lỗ hổng bảo mật lẫn lỗ hổng cụ thể trong code.

---

## 14. So sánh với probe và steering vector

Ưu điểm của dictionary learning mà tác giả nêu:

- **Chi phí một lần, khấu hao** (amortize) cho hàng triệu feature; sau đó tìm feature chỉ cần vài prompt. Probe truyền thống cần dataset riêng cho mỗi khái niệm.
- **Không giám sát**: tìm được trừu tượng ta không đoán trước (ví dụ feature "inner conflict" trong case study lừa dối).

Thí nghiệm few-shot: lấy hiệu trung bình residual giữa prompt dương và âm làm **probe/steering vector**. Kết quả:

- Top-activating examples của probe direction: **không diễn giải được** trong mọi trường hợp.
- Trên 7 ví dụ mà feature steering thành công: 2 ví dụ (gender bias, "agreement") steering vector hiệu quả tương đương; 5 ví dụ (secrecy, sycophancy, code error, self-improving AI, methamphetamine) chỉ feature steer được.
- Tác giả không khái quát quá: ngoài chế độ few-shot, các phương pháp có giám sát có thể ngang hoặc tốt hơn. Họ chưa so sánh chặt chẽ.

---

## 15. Cách paper kiểm chứng tuyên bố

| Tuyên bố | Bằng chứng | Con số |
|---|---|---|
| SAE scale được | 3 run, scaling laws | L0 < 300, ≥ 65% variance |
| Feature đặc hiệu | Rubric chấm bằng Claude 3 Opus, ~1.000 activation mỗi feature | Activation cao gần như đều điểm 3 |
| Feature gây hành vi | Clamp/steering | 2× đến 20× max |
| Feature ≠ neuron | Tương quan với mọi neuron trước | 82% có max corr ≤ 0,3 |
| Dễ hiểu hơn neuron | Automated interp, 100 so với 100 | Hơn đáng kể (văn bản không nêu số) |
| Trung gian tính toán | Attribution và ablation | corr 0,81 so với 0,12 |
| Completeness theo tần suất | 4 nhóm khái niệm, London boroughs | ~60% boroughs; ngưỡng ~1/N_alive |
| Hơn probe few-shot | 7 case study | 5/7 chỉ feature thành công |

---

## 16. Giới hạn

Tác giả liệt kê (tóm lược):

- **Hời hợt**: dữ liệu SAE chỉ là text giống pretraining, không có định dạng Human/Assistant, không có ảnh.
- **Không có hàm mục tiêu thật**: MSE + L1 chỉ là proxy cho tính dễ hiểu; chưa biết đánh đổi MSE và sparsity sao cho đúng. Scaling laws tối ưu proxy, chưa chắc tối ưu điều ta quan tâm.
- **Cross-layer superposition**: feature trải qua nhiều layer. SAE trên residual stream chỉ giải phần nào; feature được hoàn thiện ở layer sau vẫn khó diễn giải. Muốn làm SAE kiểu "pre-post"/**transcoder** cho MLP thì càng khó dung hòa với vấn đề này.
- **Chưa có hết feature, chi phí compute**: có thể thiếu nhiều bậc độ lớn; lấy hết feature mọi layer có thể tốn **hơn compute huấn luyện mô hình gốc**. Hướng giải: SAE rẻ hơn (ví dụ mixture of experts), hiệu quả dữ liệu hơn (Attribution SAE).
- **Shrinkage**: L1 làm activation khác 0 bị **đánh giá thấp có hệ thống**. Họ thử tanh-L1: proxy tốt hơn nhưng feature kém dễ hiểu.
- **Rào cản khác**: attention superposition; **interference weights** từ weight superposition làm khó phân tích circuit.
- **Scalability**: số feature và circuit quá lớn.
- **Hiểu biết khoa học còn hạn chế**: feature manifold đa chiều vẫn là khả năng thực.
- Về an toàn: rủi ro "ảo giác" từ feature splitting lộn xộn, và downstream effect lệch khỏi activation pattern.
- Từ 34M: **65% dead features** là một dấu hiệu quy trình còn kém hiệu quả.

---

## 17. Ứng dụng giả thuyết cho pruning và distillation

> **Phần này là suy luận của người viết, không phải kết quả của paper.**

### 17.1 Cắt layer, cross-layer superposition và $D_{lin}$

Paper cho rằng feature có thể "trải" qua nhiều layer, và residual stream cộng các mảnh lại. Nếu đúng, cắt một layer chỉ lấy đi **một mảnh** của nhiều feature, và các layer lân cận cũng ghi những mảnh tương quan. Đó là lý do hợp lý để một linear map tại điểm cắt khôi phục được nhiều ($D_{lin}$ lớn, $D_{res}$ nhỏ). Ngược lại, feature *chỉ* được tính ở layer bị cắt (ví dụ một AND phi tuyến trong MLP) sẽ rơi vào $D_{res}$.

### 17.2 Tần suất và đuôi dài

Mục 9 cho thấy feature của khái niệm hiếm chỉ xuất hiện ở từ điển lớn, tức là chúng "đắt" về capacity. Giả thuyết: khi giảm từ 0.8B xuống 0.3B/0.1B, khái niệm hiếm (font lạ, ký hiệu toán, chữ viết ít gặp, dấu tiếng Việt kép) bị mất trước, và **CER trung bình che giấu điều đó**. Hãy đánh giá theo nhóm tần suất, như đường cong sigmoid của paper.

### 17.3 Feature-level hidden-state KD

Với teacher SAE trên residual stream, loss KD có thể là:

$$
\mathcal L_{KD-feat}=\sum_i w_i\big(f_i(h_T)-f_i(Ah_S)\big)^2
$$

với $w_i$ ưu tiên feature quan trọng cho output (attribution cao). Paper cho thấy activation mạnh nhất **không** phải feature quan trọng nhất (mục 10.3), nên trọng số theo attribution có lẽ tốt hơn trọng số theo activation.

### 17.4 Multimodal và VLM OCR

SAE train trên text tổng quát sang ảnh. Với LightOnOCR, câu hỏi kiểm chứng được: feature của decoder (train trên text tokens) có bật trên **vision tokens** chứa cùng chữ không? Nếu có, đó là công cụ đo xem ViT đã bị prune có còn chuyển tải "khái niệm chữ" tới decoder không.

### 17.5 GatedDeltaNet

SAE residual stream không phụ thuộc loại mixer. Nhưng paper nhắc attention superposition chưa được giải. Với lớp linear-attention có trạng thái hồi quy, feature có thể nằm trong **trạng thái bộ nhớ** chứ không chỉ trong residual stream. Đây là vùng mù của phương pháp.

### 17.6 Steering như phép thử nhân quả cho pruning

Công thức $\mathbf x'=\mathbf x+(c-f_k)\mathbf d_k$ cho phép "trả lại" một feature cho student. Nếu cộng lại feature bị mất mà sửa được lỗi OCR, đó là bằng chứng nhân quả rằng mất feature đó gây ra phần damage tương ứng.

---

## 18. Thí nghiệm gợi ý

**Thí nghiệm 1: Đường cong completeness theo tần suất cho teacher và student.** Huấn luyện SAE (cùng kích thước) trên residual stream ở layer giữa của teacher và student. Với tập khái niệm OCR (ký tự Unicode, ký hiệu, dấu tiếng Việt) có tần suất đã biết, đo xác suất "có feature riêng" theo log-tần suất. Dự đoán: đường sigmoid của student dịch sang phải (cần tần suất cao hơn mới có feature).

**Thí nghiệm 2: Attribution để chẩn đoán lỗi.** Trên các mẫu student sai mà teacher đúng, tính attribution (teacher SAE) cho hiệu logit token đúng so với token student chọn. Lấy top-10 feature, kiểm tra activation của chúng trên student (sau linear map). Thử steer student (cộng lại feature) xem có sửa lỗi không.

**Thí nghiệm 3: Text-feature trên vision token.** Với ảnh chứa từ X và văn bản "X", so tập feature bật ở decoder (sau projector). Đo độ trùng (Jaccard của top-k) trước và sau khi prune ViT.

---

## 19. Tóm tắt công thức và khái niệm

| Khái niệm | Công thức hoặc định nghĩa | Ý nghĩa |
|---|---|---|
| Chuẩn hóa | $\mathbb E\Vert \mathbf x\Vert _2^2=D$ | L1 coefficient so sánh được |
| Encoder | $f_i=\mathrm{ReLU}(\mathbf W^{enc}_{i}\mathbf x+b^{enc}_i)$ | Phát hiện feature |
| Decoder | $\hat{\mathbf x}=\mathbf b^{dec}+\sum_i f_i\mathbf W^{dec}_{\cdot,i}$ | Tái tạo |
| Loss | $\Vert \mathbf x-\hat{\mathbf x}\Vert ^2+\lambda\sum_i f_i\Vert \mathbf W^{dec}_{\cdot,i}\Vert $ | Thưa, không gian lận chuẩn |
| Feature activation | $f_i\cdot\Vert \mathbf W^{dec}_{\cdot,i}\Vert $ | Đơn vị dùng trong paper |
| Scaling law | $\mathcal L^*\propto C^{-\alpha}$; $F^*,S^*$ power law | Phân bổ compute cho SAE |
| Ngưỡng tần suất | $p\gtrsim 1/N_{alive}$ | Khi nào khái niệm có feature riêng |
| Attribution | $f_i\,\mathbf W^{dec}_{\cdot,i}\cdot\nabla_{\mathbf x}\Delta z$ | Xấp xỉ ablation (corr 0,81) |
| Steering | $\mathbf x+(c-f_k)\mathbf W^{dec}_{\cdot,k}$, giữ error | Kiểm tra nhân quả |
| Cross-layer superposition | Feature trải qua nhiều layer | Lý do chọn residual stream |
| Shrinkage | L1 làm activation bị đánh giá thấp | Giới hạn của SAE ReLU+L1 |
| Kích thước | 1M / 4M / 34M; dead 2% / 35% / 65% | Ba từ điển chính |

### Kết luận ngắn

Paper cho thấy công thức SAE đơn giản **scale được** tới mô hình production: hàng chục triệu feature, phần lớn đơn nghĩa hơn neuron, nhiều feature trừu tượng (đa ngôn ngữ, đa phương thức, cụ thể lẫn trừu tượng), có tác động nhân quả qua steering, và có cả feature liên quan an toàn. Đồng thời paper trung thực về giới hạn: chưa có hết feature (đuôi dài theo tần suất), dead feature nhiều, shrinkage, cross-layer superposition, và chưa chứng minh giá trị an toàn thực tế. Với người làm nén mô hình, hai bài học đáng giữ: (1) **khái niệm hiếm là thứ đắt nhất** về capacity, nên dễ mất nhất khi nén; (2) **attribution, không phải activation**, mới chỉ ra feature quan trọng cho một output.

**Đọc bản gốc:** https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
