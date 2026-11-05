# Softmax Linear Units (2022)

**Tác giả:** Nelson Elhage, Tristan Hume, Catherine Olsson, Neel Nanda, Tom Henighan, Scott Johnston, Sheer El Showk, Nicholas Joseph, Nova DasSarma, Ben Mann, Danny Hernandez, Amanda Askell, Kamal Ndousse, Andy Jones, Dawn Drain, Anna Chen, Yuntao Bai, Deep Ganguli, Liane Lovitt, Zac Hatfield-Dodds, Jackson Kernion, Tom Conerly, Shauna Kravec, Stanislav Fort, Saurav Kadavath, Josh Jacobson, Eli Tran-Johnson, Jared Kaplan, Jack Clark, Tom Brown, Sam McCandlish, Dario Amodei, Christopher Olah (Anthropic). Xuất bản 27/06/2022.  
**Nguồn:** https://transformer-circuits.pub/2022/solu/index.html  
**Chủ đề:** Activation function SoLU, polysemanticity trong MLP của Transformer, thiết kế kiến trúc cho interpretability, vai trò "lén lút" của LayerNorm, loại neuron theo độ sâu.  
**Vị trí trong lộ trình:** SoLU xuất bản *trước* Toy Models, nhưng nên đọc *sau* ghi chú 05 vì Toy Models trình bày superposition rõ hơn (chính SoLU có ghi chú cập nhật nói vậy). Đây là thử nghiệm đầu tiên trên mô hình thật cho câu "giảm superposition thì mất gì", nên rất sát câu hỏi pruning neuron MLP của bạn.

> Tài liệu học tập bằng tiếng Việt. Đây là ghi chú diễn giải, không phải bản dịch nguyên văn paper. Những phần liên hệ tới pruning/distillation là suy luận mở rộng, không phải kết quả paper chứng minh.

## Mục lục

1. [Bối cảnh: MLP là nút thắt](#1-bối-cảnh-mlp-là-nút-thắt)
2. [Basis, neuron, polysemanticity và superposition](#2-basis-neuron-polysemanticity-và-superposition)
3. [Các tính chất có thể giảm polysemanticity](#3-các-tính-chất-có-thể-giảm-polysemanticity)
4. [SoLU và LayerNorm](#4-solu-và-layernorm)
5. [Hiệu năng](#5-hiệu-năng)
6. [Interpretability: đo định lượng](#6-interpretability-đo-định-lượng)
7. [Khám phá định tính theo độ sâu](#7-khám-phá-định-tính-theo-độ-sâu)
8. [LayerNorm "buôn lậu" superposition](#8-layernorm-buôn-lậu-superposition)
9. [Paper kiểm chứng như thế nào](#9-paper-kiểm-chứng-như-thế-nào)
10. [Giới hạn](#10-giới-hạn)
11. [Ứng dụng giả thuyết cho pruning và distillation](#11-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
12. [Thí nghiệm gợi ý](#12-thí-nghiệm-gợi-ý)
13. [Tóm tắt công thức và khái niệm](#13-tóm-tắt-công-thức-và-khái-niệm)

---

## 1. Bối cảnh: MLP là nút thắt

Các công trình trước đã reverse-engineer được Transformer **attention-only** 1–2 layer, nhờ viết lại mô hình thành dạng không cần hiểu activation trung gian (xem ghi chú framework 2021). **MLP thì không làm vậy được** vì phi tuyến. Muốn hiểu Transformer đầy đủ thì phải hiểu activation MLP mã hóa gì.

Lý do cần hiểu activation: paper ví **tham số là chương trình đã biên dịch**, **activation là biến**. Một dòng code chỉ có nghĩa khi biết biến biểu diễn gì. Số activation ít hơn số tham số nhiều, nên activation là "chìa khóa" hợp lý hơn.

Rào cản: neuron MLP của Transformer **rất polysemantic**. Giả thuyết giải thích là superposition; nếu đúng, polysemanticity có *chức năng*, nên xóa nó sẽ làm hại hiệu năng.

Đóng góp chính: thay activation bằng **SoLU** (+ một LayerNorm) làm tỷ lệ neuron "dễ hiểu" tăng **từ khoảng 35% lên khoảng 60%**, gần như không mất hiệu năng. Kèm một cảnh báo: SoLU có thể *giấu* một số feature khác sâu hơn.

---

## 2. Basis, neuron, polysemanticity và superposition

- **Representation**: không gian vector activation của một layer. Muốn hiểu nó thì phải **phân rã thành feature độc lập**, tức hiểu $N$ feature thay vì $\exp(N)$ "thể tích".
- **Non-privileged basis** (token embedding, residual stream, vector attention): bao quanh bởi phép tuyến tính, có thể chèn $M$, $M^{-1}$. Basis đúng phải *tìm* ra.
- **Privileged basis** (activation MLP): activation theo từng tọa độ phá đối xứng. Feature *có thể* thẳng hàng với neuron.
- **Neuron**: một chiều trong privileged basis.
- **Polysemantic neuron**: phản ứng với nhiều feature không liên quan (ví dụ đầu mèo, đầu xe, bàn chân).
- Thuật ngữ: *polysemanticity* là hiện tượng quan sát được; *superposition* là giả thuyết giải thích.

Trực giác cho superposition (paper liệt kê): feature là hướng; số feature có thể có (ví dụ mọi người từng tồn tại) lớn hơn số neuron rất nhiều, và có importance khác nhau; một neuron có nhiều tham số nên dành nguyên neuron cho một feature có thể "lãng phí" chỗ chứa fact; feature thưa. Cộng với almost-orthogonal vectors và compressed sensing.

Paper nhấn mạnh: activation phi tuyến **có** tạo áp lực đưa feature về neuron, nhưng nếu lợi ích sparse coding đủ lớn thì áp lực này bị lấn át. Ở residual stream (không privileged) áp lực superposition còn mạnh hơn.

**Hai hướng xử lý**: (1) tạo mô hình ít superposition hơn; (2) hiểu được biểu diễn có superposition (kiểu compressed sensing). Paper này đi theo hướng (1).

---

## 3. Các tính chất có thể giảm polysemanticity

| Tính chất | Lập luận |
|---|---|
| **Activation sparsity** | Polysemantic neuron bật thường hơn, và cần neuron khác bật cùng để phân giải nghĩa. Superposition cần *khoảng cách* giữa độ thưa của feature và độ thưa của neuron. Ép neuron thưa thì thu hẹp khoảng cách ấy. |
| **Lateral inhibition** (ức chế bên) | Neuron này bật làm neuron khác bật yếu đi (như softmax), tạo "co-occurrence sparsity": ít neuron cùng bật trong một ví dụ, nên khó phân giải polysemantic. |
| **Weight sparsity** | Feature thẳng neuron thì trọng số "lý tưởng" giữa hai feature là một phần tử; không thẳng thì bị "nhòe" ra nhiều phần tử. Nhưng chỉ hợp lý khi *cả* input và output privileged; trong Transformer mọi trọng số đều nối với residual stream, nên paper **không** cho là phù hợp. |
| **Superlinear activation** | Ví dụ $\operatorname{ReLU}(x)^2$, $\operatorname{softmax}(x)\cdot x$, $\exp(x)$. Trải feature ra $N$ neuron bị "co": $f(x/\sqrt N)<f(x)/\sqrt N$. Hai feature trùng neuron cùng bật thì nhiễu lớn hơn tổng: $f(a+b)>f(a)+f(b)$. |
| **Nhiều neuron hơn trên mỗi FLOP/tham số** | Cho mạng "cái nó muốn" thay vì ép nó. Phóng to mô hình chưa chắc đủ, vì mô hình lớn muốn nhiều feature hơn. |

---

## 4. SoLU và LayerNorm

### 4.1 Định nghĩa

GeLU xấp xỉ bằng $\operatorname{sigmoid}(1.7x)\cdot x$. Thay sigmoid bằng softmax (mở rộng đa biến):

$$
\boxed{\operatorname{SoLU}(x)=x\odot\operatorname{softmax}(x)}
$$

SoLU gộp lateral inhibition, xấp xỉ sparsity và superlinearity trong một phép.

Ví dụ trong paper:

| Input | SoLU (xấp xỉ) | Ý nghĩa |
|---|---|---|
| $(4,1,4,1)$ | $(2,0,2,0)$ | Giá trị lớn đè giá trị nhỏ |
| $(4,0,0,0)$ | $(4,0,0,0)$ | Feature thẳng trục được giữ nguyên |
| $(1,1,1,1)$ | $(\tfrac14,\tfrac14,\tfrac14,\tfrac14)$ | Feature trải đều bị co mạnh |

Kiểm tra nhanh ví dụ đầu (người viết tự tính): $\operatorname{softmax}(4,1,4,1)\approx(0.48,0.02,0.48,0.02)$, nhân với $x$ được khoảng $(1.9,0.02,1.9,0.02)$, tức xấp xỉ $(2,0,2,0)$.

### 4.2 Thêm LayerNorm

SoLU đơn thuần làm neuron dễ hiểu hơn nhiều, nhưng **hiệu năng chỉ tương đương mô hình nhỏ hơn 30–50%** (mô hình lớn bị nhiều hơn). Paper nói đây đúng là điều superposition hypothesis dự đoán.

Sửa bằng một LayerNorm sau SoLU:

$$
\boxed{f(x)=\operatorname{LN}(\operatorname{SoLU}(x))=\operatorname{LN}\big(x\odot\operatorname{softmax}(x)\big)}
$$

Activation được *diễn giải* là activation **trước** LN này.

Vì LN bất biến với phép nhân hệ số ($\sigma(\alpha x)=\alpha\sigma(x)$), mẫu số softmax không ảnh hưởng hành vi cuối:

$$
\operatorname{LN}(\operatorname{SoLU}(x))=\operatorname{LN}\!\left(x\frac{\exp(x)}{\sum_i\exp(x_i)}\right)=\operatorname{LN}(x\odot\exp(x)).
$$

Hệ quả đáng lo: một vector bị SoLU co nhỏ vì trải đều sẽ được **LN phóng to trở lại**. Đây là cửa để superposition "lọt qua" dưới dạng activation nhỏ (mục 8).

**Song song tensor**: ở mô hình lớn, softmax và LN được chia theo khối chiều cho từng accelerator; theo thử nghiệm không chính thức, việc chia khối không ảnh hưởng đáng kể.

---

## 5. Hiệu năng

- Huấn luyện cặp baseline/SoLU từ 1 layer đến 64 layer (khoảng 50B tham số), mỗi nấc khoảng ×4 tham số; SoLU giữ nguyên mọi siêu tham số của baseline.
- Quy loss ra **"model size multiplier"** (hệ số kích thước tương đương theo scaling law): SoLU luôn nằm giữa **0.95× và 1.05×**, khoảng **±0.01 nats** trên tổng loss 1.6–3 nats. Với 50B, SoLU tương đương 1.01×. Có thể có xu hướng SoLU khá hơn chút ở mô hình lớn, nhưng nhiều khả năng là nhiễu.
- Downstream (Lambada, OpenBookQA, ARC, HellaSwag, MMLU, TriviaQA, arithmetic): tương đương; arithmetic tốt hơn với SoLU, TriviaQA tốt hơn với baseline, không có xu hướng hệ thống.
- Không quét siêu tham số cho SoLU, nên nếu có thì kết quả *đánh giá thấp* SoLU.
- Chi phí tính toán: chậm **dưới 1%** với kernel phù hợp.

---

## 6. Interpretability: đo định lượng

### 6.1 Giao thức

- Câu hỏi đo: neuron có **"interpretable ngay cái nhìn đầu"** không (yếu hơn "thật sự interpretable").
- Người đánh giá (một số tác giả) xem khoảng 20 đoạn văn chứa token neuron bật mạnh (tô đỏ theo độ lớn), dành 1–2 phút/neuron.
- Tiêu chí: INTERPRETABLE nếu **≥80% các lần bật mạnh nhất** giải thích được bằng một quy tắc/khái niệm (ví dụ từ "apple", hay cụm liên quan âm nhạc).
- Mô hình 1, 16, 24, 40, 64 layer; mỗi mô hình 60 neuron baseline + 60 neuron SoLU, tổng **600 neuron**; thêm 60 neuron của mô hình 16 layer chỉ có LN mà không SoLU, tổng **660**.
- **Ngẫu nhiên và mù** (không biết neuron từ mô hình nào).

### 6.2 Kết quả

| Mô hình | Baseline | SoLU |
|---|---|---|
| 1L–40L | khoảng 35% | khoảng 60% (+25 điểm, ×1.7) |
| 64L (50B) | 33% | 42% |
| 16L chỉ LN, không SoLU | Khoảng giữa baseline và SoLU | |

- Lợi ích **giảm mạnh ở 64L**. Paper đưa hai giả thuyết: neuron mô hình lớn biểu diễn khái niệm phức tạp hơn nên 1–2 phút không đủ; hoặc động học tối ưu ở mô hình sâu làm giảm hiệu ứng. Phần thảo luận gọi đây là chuyển tiếp sắc nét quanh 50B, trong khi ở 12B lợi ích vẫn rất rõ.
- Một người chấm thấy hiệu ứng lớn hơn (khoảng 20% vs 60% thay vì khoảng 40% vs 60%), có vẻ vì giữ tiêu chuẩn chặt hơn và ít bỏ qua activation nhỏ.

### 6.3 Cảnh báo bổ sung sau xuất bản

Paper sau đó **bi quan hơn về thước đo này**: top activation chỉ cho biết neuron có monosemantic *khi bật mạnh*. Tương quan giữa "monosemantic ở top" và "monosemantic nói chung" yếu, nhất là khi người ta bắt đầu *tối ưu* cho top activation trông monosemantic. *Towards Monosemanticity* (2023) xử lý bằng cách xem toàn phổ activation.

---

## 7. Khám phá định tính theo độ sâu

### 7.1 Mô hình 1 layer

Trong 1 layer, activation mỗi neuron có **tác động tuyến tính lên logit** (bỏ qua LN): nhân vector trọng số ra với unembedding là đọc được neuron tăng/giảm token nào. Nhờ vậy vừa đoán chức năng từ dataset example, vừa kiểm chéo bằng logit. Ví dụ: neuron **base64** (bật trên chuỗi base64, tăng token hỗn hợp hoa/thường, giảm token tiếng Anh), neuron chữ IN HOA, neuron "số rồi dấu phẩy".

### 7.2 Layer sớm: "de-tokenization"

Ánh xạ cấu trúc token "nhân tạo" sang biểu diễn có nghĩa:

- Từ bị tách nhiều token: "Trend|ing", "Ch|olesterol", "Sh|anghai".
- Tên người: "Martin| Luther| King", "Oprah| Winfrey".
- Danh từ riêng nhiều token, từ ghép ("heart| attack"), lệnh LaTeX ("\\|frac|{").
- **Token theo ngôn ngữ/ngữ cảnh**: ba neuron riêng cho "die" trong tiếng Đức, Hà Lan và Afrikaans (không bật cho "die" tiếng Anh); "<" trong Python, IRC, XML/HTML.
- Paper nói SoLU tạo khác biệt **lớn nhất ở layer sớm**: trước đó gần như không hiểu được neuron MLP đầu tiên.

### 7.3 Layer cuối: "re-tokenization"

Ngược lại với layer sớm: đổi từ/khái niệm về token để xuất ra. Ví dụ neuron bật ở "st" và tăng "rag" (để viết "st|rag|glers"); neuron "n" → "app" (cho "n|app|ies"). Chúng như mô phỏng thêm một mục từ vựng đầu ra có điều kiện.

### 7.4 Layer giữa: khái niệm trừu tượng

- Số **khi và chỉ khi** đếm người.
- Mệnh đề mô tả âm thanh, quần áo, âm nhạc ("in the key of C major"), chữ viết trên vật.
- Discourse marker: nhấn mạnh, rào đón.
- Phân giải nghĩa đặc biệt của token: A/B/C/D là điểm số, phần "ngày" của date, số lượng trong công thức nấu ăn, format specifier "%s".

Bố cục thô: **sớm de-tokenize → giữa trừu tượng → cuối re-tokenize**.

### 7.5 Mẫu hình trừu tượng

- **Neuron splitting**: neuron ở mô hình nhỏ "tách" thành nhiều neuron ở mô hình lớn (neuron hexadecimal → neuron cho từng ký tự hex; "token tiếng Anh nhưng thật ra là tiếng Đức" → "die" tiếng Đức...).
- **Neuron families**: họ "token X trong ngôn ngữ Y", gợi ý một dạng equivariance theo ngôn ngữ.
- **Đối ngẫu sớm và muộn**: nhận diện từ nhiều token ở đầu, xuất từ nhiều token ở cuối.
- Giống neuron CLIP (người nổi tiếng, vùng địa lý).

### 7.6 Giảm (một phần) interpretability illusion

Dataset example lấy từ chính phân phối huấn luyện; có editor tương tác để sửa input và xem activation; với một số neuron có xem dải activation; có thí nghiệm riêng (so neuron hex với regex). Paper vẫn nói không dám đứng sau phần lớn giả thuyết neuron với độ tin cao.

---

## 8. LayerNorm "buôn lậu" superposition

Giả thuyết **hai tầng feature**:

- **Tầng thẳng neuron**: là thứ ta thấy khi xem neuron SoLU; khi có mặt thì chúng chi phối activation.
- **Tầng không thẳng neuron**: bị SoLU co nhỏ, chỉ có tác dụng lớn khi không có feature thẳng neuron nào bật, và **LN phóng chúng to lại**.

Bằng chứng: xem dataset example ở nhiều mức activation, trước và sau LN. Với neuron trông interpretable, **sau LN có nhiều ví dụ không khớp giả thuyết hơn hẳn**, nhất là ở activation thấp. Một tác giả phân loại khoảng một nghìn ví dụ của một neuron "left/right" (nhất là khi chỉ bộ phận cơ thể): sau LN, tỷ lệ ví dụ không liên quan ở vùng activation thấp cao hơn nhiều. Thí nghiệm không chính thức, không mù.

Kết luận của paper: SoLU là **con dao hai lưỡi**. Nó làm một tập feature dễ nghiên cứu hơn, nhưng có thể có nhiều feature không thẳng neuron, quan trọng cho loss, *khó nghiên cứu hơn* mô hình thường, và **có thể vô hình** nếu chỉ nhìn activation SoLU. Dù vậy, về thực dụng vẫn là cải thiện.

Điểm lý thuyết quan trọng: **hiệu năng được phục hồi đúng lúc mô hình lại có thể làm superposition** (qua LN). Paper coi đây là bằng chứng gián tiếp đầu tiên ủng hộ superposition hypothesis so với các giải thích khác.

---

## 9. Paper kiểm chứng như thế nào

| Loại bằng chứng | Chi tiết |
|---|---|
| So hiệu năng có đối chứng | Cặp baseline/SoLU cùng siêu tham số, 1L → 64L; loss, size multiplier, 7 benchmark |
| Thí nghiệm người chấm mù | 660 neuron, tiêu chí 80% top firing, 1–2 phút/neuron |
| Ablation thành phần | Mô hình 16L chỉ LN (khoảng giữa) |
| Kiểm chéo bằng logit | Ở mô hình 1 layer, neuron → logit tuyến tính |
| Phân tích theo mức activation | Trước/sau LN, neuron "left/right" |
| Bằng chứng gián tiếp cho superposition | SoLU không LN giống mô hình nhỏ hơn 30–50%; thêm LN thì phục hồi |

---

## 10. Giới hạn

Paper tự nêu:

- **LN cho phép feature không thẳng neuron lọt qua dưới dạng activation nhỏ**, tức có những feature "vô hình".
- Một kiến trúc, một dataset (context 8192, rotary, xen dense/sparse attention, nhiều sách hơn common crawl...). Chưa biết kết quả có khái quát không.
- Lợi ích giảm mạnh ở 64L/50B.
- Thước đo chỉ là "interpretable khi nhìn nhanh"; không xem ví dụ âm; dễ phán sai. Về sau tác giả còn bi quan hơn (mục 6.3).
- Kể cả khi mọi neuron đều interpretable, mô hình có hàng triệu neuron vẫn cần cách tổng hợp để có bức tranh toàn cục.

Thêm (người viết): kiến trúc hiện đại thường dùng gated MLP (ví dụ SwiGLU), khác GeLU MLP ở đây; kết quả có thể không chuyển nguyên vẹn.

---

## 11. Ứng dụng giả thuyết cho pruning và distillation

> **Suy luận của người viết, không phải kết quả paper.**

### 11.1 Bằng chứng gần nhất cho "superposition là dung lượng"

SoLU không LN ≈ mô hình nhỏ hơn 30–50%: ép bớt superposition thì mất dung lượng tương đương bỏ một phần lớn tham số. Đảo lại góc nhìn pruning: một student 0.3B/0.1B *cần* superposition nhiều hơn teacher để bù dung lượng, và việc cắt neuron làm phá đúng cơ chế này. Điều đó phù hợp với quan sát của bạn rằng cắt neuron MLP khó phục hồi.

### 11.2 Tiêu chí importance theo top activation là đáng ngờ

Mục 8 cho thấy feature quan trọng có thể chạy trong **activation nhỏ**. Tiêu chí pruning kiểu "max activation" hay "neuron hiếm khi bật mạnh thì cắt" có thể bỏ qua đúng những neuron mang tầng feature không thẳng trục, vốn lại khó khôi phục tuyến tính. Nên dùng tiêu chí dựa trên đóng góp vào loss/đầu ra trên toàn phân phối, không chỉ top activation.

### 11.3 Neuron splitting đảo ngược

Mô hình lớn tách neuron thành neuron chuyên biệt. Khi thu nhỏ, có thể kỳ vọng hiện tượng **gộp**: student nhỏ phải gộp các neuron chuyên biệt của teacher (ví dụ "ký tự X trong script Y") thành neuron chung hơn hoặc superposition. KD theo từng neuron vì thế không có tương ứng 1-1.

### 11.4 Bố cục theo độ sâu cho OCR decoder

Nếu bố cục de-tokenize / trừu tượng / re-tokenize đúng cho decoder OCR:

- Layer cuối chứa neuron **re-tokenization** để viết từ nhiều token (tên riêng, từ dài, chuỗi số). Cắt mạnh MLP layer cuối có thể làm hỏng chính tả từ dài/hiếm trước tiên.
- Layer sớm phân giải "token theo ngôn ngữ/ngữ cảnh". Với tiếng Việt có dấu và văn bản đa ngôn ngữ, đây là nơi nhiều feature thưa, nên dự đoán superposition dày và nhạy với cắt neuron.

### 11.5 Huấn luyện "dễ cắt"

Gợi ý suy luận: nếu trong lúc KD thêm áp lực **activation sparsity / superlinearity** (ý tưởng mục 3) cho student, feature có thể dồn về ít neuron hơn. Khi đó cắt neuron sau này sẽ xóa gọn feature ít quan trọng thay vì làm méo nhiều feature cùng lúc. Cái giá có thể là dung lượng (mục 11.1), nên phải đo cả hai.

---

## 12. Thí nghiệm gợi ý

1. **Top activation vs toàn phổ.** Với một số neuron MLP của LightOnOCR, so (a) importance theo max activation với (b) importance theo mức tăng loss khi zero-ablate trên tập OCR. Đếm các neuron có (a) thấp mà (b) cao: đó là ứng viên "feature trong activation nhỏ".
2. **Hồ sơ theo độ sâu.** Cắt cùng tỷ lệ neuron MLP ở nhóm layer sớm/giữa/cuối của decoder (từng nhóm riêng), đo CER tổng và CER trên từ dài/hiếm, ký tự có dấu, chuỗi số. Kiểm tra dự đoán: cuối → chính tả từ dài; sớm → ký tự hiếm/dấu.
3. **Sparsity penalty trong KD.** Hai student cùng kích thước: KD thường và KD + L1 lên activation MLP. Sau đó cắt thêm 20% neuron mỗi bên; so phần $D_{res}$ và mức phục hồi sau fine-tune ngắn.

---

## 13. Tóm tắt công thức và khái niệm

| Khái niệm | Công thức hoặc định nghĩa | Ý nghĩa |
|---|---|---|
| GeLU xấp xỉ | $\operatorname{sigmoid}(1.7x)\cdot x$ | Điểm xuất phát |
| SoLU | $x\odot\operatorname{softmax}(x)$ | Lateral inhibition + superlinear |
| SoLU + LN | $\operatorname{LN}(x\odot\operatorname{softmax}(x))=\operatorname{LN}(x\odot\exp(x))$ | Phục hồi hiệu năng, mở cửa cho superposition |
| Superlinear | $f(x/\sqrt N)<f(x)/\sqrt N$; $f(a+b)>f(a)+f(b)$ | Phạt feature trải đều và nhiễu chồng |
| Size multiplier | Loss quy ra kích thước baseline tương đương | SoLU trong khoảng 0.95×–1.05× |
| Interpretable nhanh | ≥80% top firing theo một quy tắc, 1–2 phút | 35% → 60% (1L–40L); 33% → 42% (64L) |
| De-tokenization | Layer sớm: token → khái niệm | Từ nhiều token, token theo ngôn ngữ |
| Re-tokenization | Layer cuối: khái niệm → token | Viết từ nhiều token |
| Neuron splitting | Neuron nhỏ tách khi mô hình lớn | Gợi ý cấu trúc phân cấp feature |
| LN smuggling | Feature không thẳng neuron qua activation nhỏ rồi LN phóng to | SoLU là con dao hai lưỡi |

### Kết luận ngắn

SoLU cho thấy một thay đổi kiến trúc nhỏ có thể tăng mạnh tỷ lệ neuron MLP dễ hiểu mà không mất hiệu năng, và mở ra bản đồ thô về loại neuron theo độ sâu. Nhưng hiệu năng chỉ được giữ khi LN cho phép superposition quay lại qua activation nhỏ. Đó vừa là giới hạn, vừa là bằng chứng gián tiếp rằng superposition là dung lượng thật. Với pruning, bài học suy luận là: đừng đánh giá neuron chỉ bằng top activation, và chờ đợi rằng thu nhỏ mô hình sẽ tăng chứ không giảm áp lực superposition.

**Đọc bản gốc:** https://transformer-circuits.pub/2022/solu/index.html
