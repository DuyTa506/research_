# Superposition, Memorization, and Double Descent (2023)

**Tác giả:** Tom Henighan, Shan Carter, Tristan Hume, Nelson Elhage, Robert Lasenby, Stanislav Fort, Nicholas Schiefer, Christopher Olah (Anthropic). Xuất bản 05/01/2023.  
**Nguồn:** https://transformer-circuits.pub/2023/toy-double-descent/index.html  
**Chủ đề:** Superposition khi dữ liệu hữu hạn, memorization như "data point feature", feature vs data point dimensionality, data double descent, phát hiện ví dụ bị nhớ.  
**Vị trí trong lộ trình:** Phải đọc *Toy Models of Superposition* (ghi chú 05) trước, vì dùng lại nguyên mô hình ReLU output và khái niệm feature dimensionality. Paper này mở rộng từ vùng "vô hạn dữ liệu, underfit" sang vùng **dữ liệu hữu hạn**, nơi KD trên một tập OCR cố định của bạn thực sự nằm.

> Tài liệu học tập bằng tiếng Việt. Đây là ghi chú diễn giải, không phải bản dịch nguyên văn paper. Những phần liên hệ tới pruning/distillation là suy luận mở rộng, không phải kết quả paper chứng minh.

## Mục lục

1. [Câu hỏi: overfitting trông như thế nào bên trong](#1-câu-hỏi-overfitting-trông-như-thế-nào-bên-trong)
2. [Thiết lập thí nghiệm](#2-thiết-lập-thí-nghiệm)
3. [Data point feature và generalizing feature](#3-data-point-feature-và-generalizing-feature)
4. [Mô hình thay đổi theo kích thước dataset](#4-mô-hình-thay-đổi-theo-kích-thước-dataset)
5. [Superposition double descent](#5-superposition-double-descent)
6. [Các mở rộng trong phần bình luận](#6-các-mở-rộng-trong-phần-bình-luận)
7. [Paper kiểm chứng như thế nào](#7-paper-kiểm-chứng-như-thế-nào)
8. [Giới hạn](#8-giới-hạn)
9. [Ứng dụng giả thuyết cho pruning và distillation](#9-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
10. [Thí nghiệm gợi ý](#10-thí-nghiệm-gợi-ý)
11. [Tóm tắt công thức và khái niệm](#11-tóm-tắt-công-thức-và-khái-niệm)

---

## 1. Câu hỏi: overfitting trông như thế nào bên trong

Overfitting là vấn đề trung tâm của ML nhưng gần như chưa có hiểu biết *cơ chế* về nó. Ví dụ: một LM nhớ nguyên văn văn bản thì làm thế nào? Cách ngây thơ là dành mỗi neuron cho một chuỗi cần nhớ (lookup table), nhưng rất tốn. Vì các trường hợp cần nhớ **loại trừ nhau** (không bao giờ cùng xảy ra) nên chúng là ứng viên hoàn hảo cho superposition.

Ba phát hiện chính:

1. **Overfitting ↔ đưa *data point*, thay vì *feature*, vào superposition.**
2. Theo kích thước dataset có hai chế độ: **memorizing** (data point trong superposition) và **generalizing** (feature trong superposition).
3. Chuyển giữa hai chế độ sinh ra **double descent**.

---

## 2. Thiết lập thí nghiệm

- Input $x\in\mathbb R^n$, $n=10{,}000$; $x_i=0$ với xác suất $S=0.999$, ngược lại $\sim U[0,1]$.
- **Khác paper trước**: chuẩn hóa $\lVert x\rVert^2=1$, để việc nhớ dễ hơn (với $n$ rất lớn, định lý giới hạn trung tâm cũng khiến các ví dụ có chuẩn gần bằng nhau).
- **Tập huấn luyện hữu hạn** kích thước $T$; $X\in\mathbb R^{n\times T}$, mỗi cột $X_i$ là một ví dụ. Paper trước chỉ xét $T=\infty$.
- Mô hình ReLU output, $W$ khởi tạo Xavier:

$$
h=Wx,\qquad x'=\operatorname{ReLU}(W^\top Wx+b),\qquad
\boxed{L=\frac1T\sum_x\sum_i I_i(x_i-x'_i)^2},\quad I_i=1.
$$

- Tối ưu: **50,000 bước full-batch**, AdamW, warmup tuyến tính 2,500 bước tới 1e-3 rồi cosine về 0, weight decay 1e-2. Paper nói số bước, full-batch và learning rate được "annealed" đều quan trọng; hidden rất nhỏ và feature cực thưa làm tối ưu khó và gradient nhiễu.
- Đôi khi huấn luyện nhiều seed và chọn train loss thấp nhất để tránh cực tiểu địa phương; độ nhạy giảm nhiều khi $m>3$.
- Double descent mạnh nhất khi weight decay thấp và ngược lại (khớp tài liệu trước).

---

## 3. Data point feature và generalizing feature

Với $m=2$, nếu vẽ cột của $W$ (như paper trước) thì mọi thứ **lộn xộn**. Nhưng nếu vẽ **hidden vector của các ví dụ huấn luyện** $h_i=WX_i$ thì cấu trúc lại **rất sạch**: các *data point* xếp thành polytope (ví dụ $T$-gon, đa giác đều $T$ đỉnh).

Trực giác: nếu $T\ll n$, biểu diễn $T$ ví dụ dễ hơn biểu diễn $n$ feature trong $m$ chiều.

Từ đó paper đề xuất một **lý thuyết cơ chế ngây thơ**: *memorization/overfitting xảy ra khi mô hình làm việc với "data point feature" (feature chỉ dùng cho một ví dụ) thay vì "generalizing feature"*. Paper nói lý thuyết này có lẽ quá đơn giản, nhưng có thể chỉ đúng hướng.

Paper cũng ghi chú: hiện tượng rõ nhất khi các data point trực giao và có độ lớn tương tự. Sparsity cao làm data point dễ trực giao; nhiều feature tránh trường hợp tầm thường là mỗi ví dụ chỉ có một feature.

---

## 4. Mô hình thay đổi theo kích thước dataset

### 4.1 Data point dimensionality

Mở rộng feature dimensionality $D_{f_i}$ (ghi chú 05) sang ví dụ huấn luyện:

$$
\boxed{D_{X_i}=\frac{\lVert h_i\rVert^2}{\sum_j(\hat h_i\cdot h_j)^2}},\qquad h_i=WX_i .
$$

Nếu các ví dụ xếp thành $T$-gon trong $m=2$ chiều thì mỗi ví dụ có $D_{X_i}=m/T$.

### 4.2 Ba vùng

| Vùng | Quan sát |
|---|---|
| **Dữ liệu nhỏ** | $D_{f}$ nhỏ; $D_{X}$ bám đúng $m/T$; $h_i$ tạo $T$-gon. Với độ chính xác float vô hạn, ReLU output model có thể nhớ *bất kỳ* số ví dụ trực chuẩn nào bằng $T$-gon |
| **Dữ liệu lớn** | 5 feature có $D_f$ lớn, xếp thành **ngũ giác**; phần còn lại gần như bị bỏ. $D_f$ của chúng thấp hơn 2/5 vì 9,995 feature còn lại cộng dồn vào mẫu số. Hidden vector cũng vẽ ra ngũ giác vì phần lớn ví dụ có 0 hoặc 1 trong 5 feature đó bật |
| **Giữa** | Lộn xộn, khó diễn giải |

### 4.3 Chuẩn của tham số

Vẽ cùng thang đo: độ dài hidden vector và feature vector, cùng $\lVert W\rVert_F$ và $\lVert b\rVert_2$, **tăng tới đỉnh quanh $T=30$**, giảm nhanh ở vùng giữa, rồi đi ngang ở vùng dữ liệu lớn.

- Vùng nhỏ: có thể chứng minh nhớ ví dụ trực chuẩn bằng $T$-gon cần $W$, $b$ lớn hơn khi $T$ lớn hơn, khớp với $T<30$.
- Vùng lớn: đã có ngũ giác thì thêm dữ liệu ít thay đổi. Nghiệm đơn giản hóa cho dữ liệu vô hạn có $\lVert W\rVert=\sqrt5\approx2.2$, gần giá trị thật (dù giả định $b=0$ là không thực tế).

---

## 5. Superposition double descent

**Data double descent**: test loss *tăng rồi mới giảm* khi tăng dữ liệu, trái với trực giác "nhiều dữ liệu thì ít overfit".

- Với mỗi $T$, huấn luyện 4 mô hình với seed dataset khác nhau, lấy trung bình test loss. Kết quả: một **cái bướu rõ** đúng ở chỗ chuyển từ "data point trong superposition" sang "generalizing feature trong superposition".
- Điều đặc biệt: double descent xảy ra **không có label noise** (input và target giống hệt). "Nhiễu" đến từ **nén có mất mát** ở phép chiếu xuống: không thể mã hóa 10,000 feature vào 2 neuron bằng phép chiếu tuyến tính, nên lỗi tái tạo là không tránh được.

**Phụ thuộc $m$**: vẽ test loss như hàm hai biến $(T,m)$, 4 mô hình mỗi ô. Có những vùng mà **mô hình lớn hơn hoặc dữ liệu nhiều hơn lại làm hại**. Vậy bướu không phải sản phẩm của $m=2$ hay lỗi tối ưu.

**Phụ lục** (với $m=4$, ít lỗi tối ưu hơn):

- Nhiều epoch hơn → bướu test loss **lớn hơn**.
- Weight decay từ 1e-2 tăng lên 1.0 → **bướu biến mất**.

$$
\text{memorize (data point features)}\;\xrightarrow{\;T\uparrow\;}\;\underbrace{\text{vùng giữa}}_{\text{test loss bướu}}\;\xrightarrow{\;T\uparrow\;}\;\text{generalize (feature)}
$$

---

## 6. Các mở rộng trong phần bình luận

Đây là kết quả của người bình luận, không phải của bài chính.

**Tái lập (Adam Jermyn).** Ba vùng tái hiện định tính, nhưng ngưỡng khác: vùng generalizing bắt đầu ở $T=20{,}000$ (bài: $5{,}000$); vùng giữa bắt đầu ở $T\sim100$ (bài: $\sim700$); test loss vùng giữa lên tới 5,000 (bài: khoảng 1.01). Tác giả gốc không xác định được khác biệt siêu tham số nào, và nói chỉ quan tâm việc chuyển tiếp *tồn tại*.

**Tái lập (Marius Hobbhahn).** Giảm sparsity và số feature đi 10× vẫn thấy double descent; cũng thấy với loss khác hoặc thêm ReLU giữa các layer. Gợi ý đây là hiện tượng khá phổ biến.

**Cái gì quyết định thang generalize? (Jermyn).** Hai giả thuyết:

- (a) Superposition của data point thất bại (ví dụ do weight decay giới hạn chuẩn). Thực nghiệm: thang **không đổi theo weight decay**, nên bị bác.
- (b) Khi mỗi feature xuất hiện nhiều lần trong các tổ hợp khác nhau. Dự đoán $T\propto1/(1-S)$. Thực nghiệm: vẽ theo $T(1-S)$, các sparsity khác nhau **generalize khi mỗi feature xuất hiện khoảng 10 lần**. Với $m$ lớn hơn thì các đường không chồng khít, nên có thể còn yếu tố khác.

**Data point lặp lại (Jermyn).** Ví dụ lặp 2–3 lần: hiện tượng như cũ. Lặp nhiều hơn: mô hình học **hỗn hợp** data point và feature; mỗi ví dụ lặp **thay thế một generalizing feature** trong không gian nhúng (với $T=30{,}000$, một ví dụ lặp 5 lần chiếm chỗ một trong năm feature ngũ giác). Lập luận thô (bỏ hằng số):

$$
\text{lợi ích học feature}\sim AT(1-S),\qquad \text{lợi ích nhớ ví dụ lặp } R \text{ lần}\sim RNA(1-S)
\;\Rightarrow\; \boxed{\text{nhớ khi } R\gtrsim T/N}
$$

($N$ là số feature, $A$ biên độ.) Thực nghiệm với $m=2$: ở $T/N$ lớn, ranh giới khớp đường $R=0.4\,T/N$; ở $T/N$ nhỏ cần nhiều lần lặp hơn. Có những "suýt nhớ" (nhớ rồi bỏ), giống mô hình vùng giữa học tạm generalizing feature rồi quên.

**Data dimensionality trên MNIST (Chris Olah).** MLP 1 hidden layer 512 unit. Vì feature thật không trực giao và ví dụ bị nhớ vẫn bật vài generalizing feature, dùng **maximal data dimensionality**:

$$
\boxed{D_i^*=\sup_v\frac{(v\cdot h_i)^2}{\sum_j(v\cdot h_j)^2}}
$$

Phần lớn ví dụ có $D^*$ tương tự nhau, nhưng có ngoại lai ở cả hai đuôi. Các ví dụ $D^*$ cao (**gần 100× mức điển hình**) thường là ví dụ kỳ quặc, gợi ý chúng bị "special-case". Đây cũng là một dạng *mechanistic anomaly detection*. Hobbhahn tái lập và thấy $D^*$ bắt được chuyển tiếp memorize → generalize trong setting double descent. (Bình luận của Fred Zhang ở ghi chú 05 chỉ ra $D^*$ trùng khái niệm leverage score.)

**Lỗi tối ưu trong 2D.** Ở vùng nhớ với $m=2$, đôi khi điểm xếp thành **hai vòng tròn bán kính khác nhau** không gộp được, kèm một khe góc lớn để các điểm vòng trong vẫn chọn được tuyến tính.

---

## 7. Paper kiểm chứng như thế nào

| Bằng chứng | Chi tiết |
|---|---|
| Trực quan hóa song song | Cột $W$ (lộn xộn) vs. $h_i=WX_i$ (polytope sạch) |
| Chỉ số hình học | $D_{f_i}$ và $D_{X_i}$ theo $T$; so với $m/T$ |
| Thống kê chuẩn | $\lVert W\rVert_F$, $\lVert b\rVert_2$ theo $T$; nghiệm đơn giản hóa $\sqrt5$ |
| Nhiều seed | 4 seed dataset mỗi $T$, 4 mô hình mỗi $(T,m)$ |
| Ablation siêu tham số | Weight decay, số epoch (với $m=4$) |
| Tái lập độc lập | Jermyn, Hobbhahn (định tính khớp, định lượng lệch) |

---

## 8. Giới hạn

- Paper tự gọi là **điều tra rất sơ bộ**. Câu hỏi lớn nhất (lý thuyết ngây thơ có khái quát sang mô hình thật không) chưa trả lời.
- Thiết lập được chọn để làm **hiện tượng cực đoan nhất**: $m=2$, feature cực thưa, ví dụ chuẩn hóa, weights tied. Kết quả cần tinh chỉnh siêu tham số cẩn thận.
- Vùng giữa chưa được hiểu về cơ chế.
- Ngưỡng định lượng không tái lập chính xác (lệch nhiều lần trong bản tái lập của Jermyn).
- Chưa có định nghĩa "feature" bao gồm được cả generalizing feature lẫn data point feature.

---

## 9. Ứng dụng giả thuyết cho pruning và distillation

> **Suy luận của người viết, không phải kết quả paper.**

- **KD trên tập hữu hạn là đúng bối cảnh paper.** Student nhỏ ($m$ nhỏ) học từ một tập OCR cố định ($T$ hữu hạn) có thể rơi vào vùng *nhớ ví dụ* thay vì học feature, nhất là khi tập nhỏ so với số feature cần học (ký tự hiếm, font lạ). Bướu test loss nghĩa là **thêm một ít dữ liệu hay tăng một chút kích thước student có thể làm tệ hơn tạm thời**. Nên quét $(T, \text{kích thước student})$ thay vì kết luận từ một điểm.
- **Weight decay và số epoch.** Ở toy, weight decay lớn xóa bướu và nhiều epoch làm bướu lớn hơn. Khi fine-tune lâu student đã prune trên dữ liệu ít, cần theo dõi dấu hiệu memorization.
- **Teacher logit làm "mềm" target**, có thể giảm động cơ nhớ từng ví dụ so với nhãn cứng. Đây là giả thuyết cần kiểm chứng; paper không nói gì về KD.
- **Trang lặp lại.** Quy tắc thô $R\gtrsim T/N$ gợi ý: tài liệu, header hay template lặp nhiều trong dữ liệu OCR có thể bị student *nhớ* và **chiếm chỗ của generalizing feature**. Ở student nhỏ, mỗi chỗ trong không gian là đắt. Khử trùng lặp dữ liệu KD có thể đáng hơn với student nhỏ so với teacher.
- **Pruning làm giảm $m$ hiệu dụng** nên đẩy mô hình về phía vùng nhớ hoặc vùng giữa nếu dữ liệu fine-tune sau prune ít. Đây có thể là một lý do khác (ngoài superposition bị phá) khiến phục hồi sau cắt neuron kém trên tập đánh giá trong khi train loss vẫn tốt.
- **$D^*$ như công cụ chẩn đoán.** Tính $D^*$ của hidden state student trên tập KD để tìm ví dụ bị special-case, rồi so với teacher.

---

## 10. Thí nghiệm gợi ý

1. **Quét $(T,\text{kích thước})$.** Với 2–3 mức prune (ví dụ 0.3B, 0.1B) và 4–5 kích thước tập KD (theo log), đo CER trên tập test giữ riêng. Tìm vùng "nhiều dữ liệu hơn lại tệ hơn". Lặp lại với 2 mức weight decay.
2. **$D^*$ trên OCR.** Lấy hidden state (ví dụ trung bình theo token ở một layer giữa của decoder) cho mỗi trang trong tập KD; tính $D_i^*$ (bằng $h_i^\top(\sum_j h_jh_j^\top)^{+}h_i$, tức leverage score). Kiểm tra các trang $D^*$ cao: có phải trang lặp, trang lỗi, hay layout hiếm? So phân bố $D^*$ của teacher và student.
3. **Thí nghiệm lặp.** Chèn một trang cố định lặp $R$ lần vào tập KD với nhiều giá trị $R$; đo student có chép nguyên văn trang đó khi gặp ảnh tương tự không, và CER trên phần còn lại có giảm không.

---

## 11. Tóm tắt công thức và khái niệm

| Khái niệm | Công thức hoặc định nghĩa | Ý nghĩa |
|---|---|---|
| Loss hữu hạn | $\frac1T\sum_x\sum_i(x_i-x'_i)^2$ | Dataset kích thước $T$ |
| Data point dimensionality | $D_{X_i}=\lVert h_i\rVert^2/\sum_j(\hat h_i\cdot h_j)^2$ | Ví dụ chiếm bao nhiêu chiều |
| $T$-gon | $D_{X_i}=m/T$ | Dấu hiệu nhớ ví dụ |
| Data point feature | Feature chỉ phục vụ một ví dụ | Cơ chế ngây thơ của memorization |
| Generalizing feature | Feature dùng chung nhiều ví dụ | Ngũ giác ở vùng dữ liệu lớn |
| Double descent | Test loss có bướu ở vùng chuyển | Không cần label noise |
| Thang generalize | Khoảng 10 lần xuất hiện mỗi feature, $T(1-S)$ | Bình luận của Jermyn |
| Ngưỡng nhớ ví dụ lặp | $R\gtrsim T/N$ (thực nghiệm $0.4\,T/N$) | Ví dụ lặp chiếm chỗ feature |
| Maximal data dimensionality | $D_i^*=\sup_v (v\cdot h_i)^2/\sum_j(v\cdot h_j)^2$ | Phát hiện ví dụ bị special-case |

### Kết luận ngắn

Trong toy model, memorization là **đưa ví dụ huấn luyện (thay vì feature) vào superposition**, và chuyển tiếp từ nhớ sang khái quát theo kích thước dataset tạo ra double descent mà không cần label noise. Weight decay và số epoch điều khiển độ lớn của bướu; ví dụ lặp có thể chiếm chỗ generalizing feature. Với KD và pruning, bài học suy luận là: student nhỏ trên tập hữu hạn có thể dùng dung lượng quý giá để nhớ thay vì học, nên cần quét kích thước dữ liệu, khử trùng lặp, và dùng $D^*$ để soi ví dụ bị nhớ.

**Đọc bản gốc:** https://transformer-circuits.pub/2023/toy-double-descent/index.html
