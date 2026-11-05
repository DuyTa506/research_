# Privileged Bases in the Transformer Residual Stream (2023)

**Tác giả:** Nelson Elhage, Robert Lasenby, Christopher Olah (Anthropic). Xuất bản 16/03/2023.  
**Nguồn:** https://transformer-circuits.pub/2023/privileged-basis/index.html  
**Chủ đề:** Privileged basis, outlier dimension trong residual stream, kurtosis, LayerNorm và RMSNorm, xoay ngẫu nhiên, vai trò của Adam.  
**Vị trí trong lộ trình:** Đọc sau *Toy Models of Superposition* (ghi chú 05, mục 1.2 và 9) vì paper này kiểm tra trực tiếp câu "residual stream không có privileged basis". Quan trọng với bạn vì nó nói kênh nào của hidden state "đặc biệt" về thống kê, điều ảnh hưởng tới pruning theo chiều $d_{model}$, quantization và KD theo hidden state.

> Tài liệu học tập bằng tiếng Việt. Đây là ghi chú diễn giải, không phải bản dịch nguyên văn paper. Những phần liên hệ tới pruning/distillation là suy luận mở rộng, không phải kết quả paper chứng minh.

## Mục lục

1. [Câu hỏi: vì sao outlier lại thẳng hàng với trục](#1-câu-hỏi-vì-sao-outlier-lại-thẳng-hàng-với-trục)
2. [Đo privileged basis: outlier và kurtosis](#2-đo-privileged-basis-outlier-và-kurtosis)
3. [Loại trừ LayerNorm](#3-loại-trừ-layernorm)
4. [Loại trừ độ chính xác số thực](#4-loại-trừ-độ-chính-xác-số-thực)
5. [Kết luận: nghi phạm là Adam](#5-kết-luận-nghi-phạm-là-adam)
6. [Paper kiểm chứng như thế nào](#6-paper-kiểm-chứng-như-thế-nào)
7. [Giới hạn](#7-giới-hạn)
8. [Ứng dụng giả thuyết cho pruning và distillation](#8-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
9. [Thí nghiệm gợi ý](#9-thí-nghiệm-gợi-ý)
10. [Tóm tắt công thức và khái niệm](#10-tóm-tắt-công-thức-và-khái-niệm)

---

## 1. Câu hỏi: vì sao outlier lại thẳng hàng với trục

Tim Dettmers quan sát **"emergent outliers"**: trong Transformer lớn, một số tọa độ của residual stream có giá trị cực lớn, tới khoảng 20× các tọa độ khác. Ngoài câu hỏi "chúng biểu diễn gì", paper hỏi câu sâu hơn: **vì sao chúng thẳng hàng với trục tọa độ?**

Theo lý thuyết, residual stream **không có privileged basis**: mọi phép đọc/ghi đều qua ma trận tuyến tính full-rank, nên có thể xoay stream bằng $R$ và nhân $R$ (hoặc $R^\top$) vào mọi ma trận đọc/ghi, được một mô hình tương đương. Nếu basis là tùy ý, một feature lớn sẽ "trải" đều, đóng góp khoảng $1/\sqrt d$ độ lớn vào mỗi tọa độ. Vậy thấy outlier dính vào vài trục nghĩa là **có thứ gì phá đối xứng**.

---

## 2. Đo privileged basis: outlier và kurtosis

Mô hình thí nghiệm: Transformer 200 triệu tham số, $d_{model}=1280$.

**Outlier** (theo định nghĩa của Dettmers): giá trị tuyệt đối $>6$. Trên batch 128 chuỗi × 1024 token, số chiều từng có outlier tăng theo độ sâu; layer điển hình có **20–60 chiều outlier** trên 1280.

**Kurtosis** (độ nhọn/độ dày đuôi, Gaussian có kurtosis = 3) là thước đo không cần ngưỡng. Lập luận:

1. Nếu không có privileged basis, mỗi feature nằm theo một hướng ngẫu nhiên.
2. Một hướng ngẫu nhiên tương đương (sai khác hệ số) với một vector các mẫu Gaussian độc lập, vì Gaussian đẳng hướng bất biến khi xoay.
3. Activation là tổng các hướng như vậy, nên các tọa độ của nó phân phối Gaussian.
4. Coi các tọa độ của activation một token là mẫu, tính kurtosis, rồi lấy trung bình qua nhiều token:

$$
\boxed{\kappa(h)=\frac{\frac1d\sum_{i=1}^d (h_i-\bar h)^4}{\left(\frac1d\sum_{i=1}^d (h_i-\bar h)^2\right)^2}\;\approx\;3 \text{ nếu basis không đặc quyền}}
$$

(Công thức kurtosis chuẩn do người viết điền vào; paper mô tả bằng lời.)

- $\kappa>3$ thì có privileged basis. Chiều ngược lại không chắc: $\kappa=3$ cũng có thể do bản thân feature là Gaussian, nhưng trực giác là feature thường thưa.
- Kết quả: residual stream **cực kỳ không Gaussian** ở các layer.
- Kiểm tra đối chứng: áp một phép xoay ngẫu nhiên cố định lên cùng activation thì kurtosis về **gần đúng 3**.

---

## 3. Loại trừ LayerNorm

LayerNorm là phép phụ thuộc basis rõ nhất trong forward pass, theo hai cách: trừ trung bình (tích vô hướng với $\frac1{d}(1,1,\dots)$) và trọng số học theo từng kênh.

Thay bằng biến thể bất biến khi xoay ("LayerNorm không trừ mean, chỉ một scale chung"):

$$
\boxed{\operatorname{RMSNorm}(x_i)=\alpha\cdot\frac{x_i}{\operatorname{RMS}(x)},\qquad \operatorname{RMS}(x)=\sqrt{\tfrac1d\textstyle\sum_{i=1}^d x_i^2}}
$$

với **một** $\alpha$ vô hướng. Hiệu năng bằng baseline, nhưng activation vẫn đuôi dày, **thậm chí dày hơn**. Vậy LayerNorm không phải nguyên nhân.

> Lưu ý: RMSNorm trong nhiều mô hình hiện đại có trọng số theo từng kênh, khác bản bất biến này.

---

## 4. Loại trừ độ chính xác số thực

Giả thuyết (Neel Nanda, Alex Silverstein): float 16/32-bit mất chính xác khi cộng số khác thang, nên mô hình muốn tách feature khác độ lớn vào các tọa độ khác nhau.

**Thí nghiệm A, forward pass.** Với RMSNorm bất biến, mô hình hoàn toàn bất biến khi xoay. Sinh ma trận trực giao ngẫu nhiên $R$, nhân $R$ vào ma trận đọc từ residual và $R^\top$ vào ma trận ghi. Loss thay đổi **cực nhỏ**, nhỏ hơn nhiều so với nhiễu do đổi seed (baseline có test loss khoảng 3.02 nats). Vậy forward pass không phụ thuộc basis kể cả khi tính sai số float.

**Thí nghiệm B, huấn luyện trong basis tùy ý.** Chèn phép xoay ngẫu nhiên cố định vào activation mỗi khi đọc/ghi residual stream (không đổi lớp hàm biểu diễn được):

| Biến thể | Mô tả | Kết quả |
|---|---|---|
| Shared rotation | Một phép xoay chung: tách "basis truyền thông" (residual) khỏi "basis tính toán" (attention/MLP) | Hiệu năng ≈ baseline; residual có $\kappa\approx3$; **trong basis tính toán vẫn đuôi dày** giống baseline |
| Independent rotations | Mỗi lần đọc/ghi một phép xoay riêng | Hiệu năng ≈ baseline; $\kappa\approx3$; chỉ đuôi nhẹ, chủ yếu ở layer đầu |

Diễn giải của paper:

- Transformer **không cần** privileged basis để huấn luyện tốt, nên privileged basis là tác dụng phụ chứ không thiết yếu. Đây là bằng chứng vừa phải chống giả thuyết float.
- Đuôi dày xuất hiện ở **basis tính toán**, không phải ở residual, nên sự thẳng trục là sản phẩm của tính toán bên trong layer. Đây là bằng chứng mạnh chống giả thuyết float.
- Khi mọi layer dùng chung một basis, chúng "thông đồng" tạo outlier; một layer đơn lẻ chỉ có hiệu ứng nhỏ.

Loss curve: khác chút đầu huấn luyện, hội tụ về mức trong nhiễu sau khoảng một tỷ token.

---

## 5. Kết luận: nghi phạm là Adam

**Adam** chuẩn hóa cập nhật gradient **theo từng phần tử** (moment theo từng tọa độ), nên ưu tiên basis mà trọng số được lưu, so với các hướng tùy ý trong không gian tham số. Sau khi loại LayerNorm và float, đây là phép phụ thuộc basis mạnh nhất còn lại.

Paper **không khẳng định chắc chắn**. Hai thí nghiệm đề xuất nhưng chưa làm: huấn luyện bằng SGD (optimizer không phụ thuộc basis), hoặc lưu moment của Adam ở một basis khác với trọng số.

---

## 6. Paper kiểm chứng như thế nào

| Bằng chứng | Vai trò |
|---|---|
| Đếm outlier ($>6$) theo layer | Tái hiện hiện tượng Dettmers ở 200M |
| Kurtosis + đối chứng xoay ngẫu nhiên | Thước đo không ngưỡng; xoay thì về 3 |
| Ablation kiến trúc (RMSNorm bất biến) | Loại LayerNorm |
| Xoay toàn bộ mô hình ở forward | Loại float ở forward |
| Huấn luyện với shared / independent rotations | Tách basis truyền thông và tính toán; định vị nguồn đuôi dày |
| So loss curve | Xác nhận các biến thể không làm giảm hiệu năng |

---

## 7. Giới hạn

- Kết luận về Adam là **loại trừ**, không phải chứng minh trực tiếp.
- Chỉ một mô hình cỡ 200M, một codebase; outlier ở mô hình lớn có thể có thêm cơ chế.
- Kurtosis $=3$ không chứng minh basis không đặc quyền.
- Paper không trả lời *outlier biểu diễn gì*.

---

## 8. Ứng dụng giả thuyết cho pruning và distillation

> **Suy luận của người viết, không phải kết quả paper.**

- **Hai nghĩa của "đặc quyền".** Về *chức năng*, residual stream bất biến khi xoay (mục 4). Về *thống kê*, mô hình huấn luyện bằng Adam vẫn có vài kênh đặc biệt (outlier, kurtosis cao). Pruning theo chiều $d_{model}$ chịu ảnh hưởng cả hai: cắt nhầm kênh outlier có thể rất đau, trong khi về chức năng lại có thể cắt trong một basis xoay tùy ý.
- **Liên hệ $D_{lin}$.** Vì residual non-privileged về chức năng, một phép xoay hoặc phép chiếu tuyến tính tại chỗ cắt "miễn phí" về biểu diễn. Cắt trong basis xoay (ví dụ theo PCA của activation) có thể giữ được nhiều $D_{lin}$ hơn cắt theo kênh gốc; cắt theo kênh gốc thì phải tránh các kênh outlier. Cần đo.
- **Basis tính toán mới là nơi đuôi dày** (bên trong MLP và các nhánh attention). Với kiến trúc hybrid của bạn, các phép theo kênh trong khối GatedDeltaNet (gate, normalization, tùy triển khai) và RMSNorm có trọng số theo kênh đều phụ thuộc basis, nên có thể có outlier riêng. Nên đo kurtosis riêng cho từng loại khối.
- **KD theo hidden state.** MSE thô giữa student và teacher bị chi phối bởi vài kênh outlier. Nên chuẩn hóa theo kênh, hoặc khớp sau một phép chiếu học được, hoặc dùng cosine.
- **Quantization sau pruning** cũng gặp đúng các outlier này (bối cảnh gốc của Dettmers).

---

## 9. Thí nghiệm gợi ý

1. **Bản đồ kurtosis.** Với LightOnOCR (decoder và ViT), tính $\kappa$ trung bình theo token ở mỗi layer, cho: residual stream, input của MLP up-projection, activation MLP. Làm lại sau một phép xoay ngẫu nhiên (phải về khoảng 3). Đánh dấu kênh có outlier $>6$.
2. **Cắt kênh gốc vs. cắt kênh xoay.** Cắt cùng số chiều $d_{model}$ theo (a) kênh gốc có chuẩn nhỏ, (b) sau khi xoay theo PCA. So $D_{lin}$, $D_{res}$ và CER trước/sau KD.
3. **Student và teacher.** So số kênh outlier của student sau KD với teacher: student có tái tạo outlier ở cùng vị trí (nếu kế thừa trọng số) hay ở vị trí khác?

---

## 10. Tóm tắt công thức và khái niệm

| Khái niệm | Công thức hoặc định nghĩa | Ý nghĩa |
|---|---|---|
| Non-privileged basis | Xoay $R$ + bù $R^\top$ cho mô hình tương đương | Trục không có nghĩa riêng |
| Feature trải đều | Khoảng $1/\sqrt d$ mỗi tọa độ | Kỳ vọng nếu basis tùy ý |
| Outlier | $\lvert h_i\rvert>6$ | 20–60 trên 1280 chiều ở layer điển hình |
| Kurtosis | Gaussian $=3$; $>3$ là đuôi dày | Thước đo privileged basis không cần ngưỡng |
| RMSNorm bất biến | $\alpha\, x_i/\operatorname{RMS}(x)$, một $\alpha$ | Không phụ thuộc basis |
| Shared rotation | Tách basis truyền thông và tính toán | Đuôi dày nằm ở basis tính toán |
| Independent rotations | Mỗi lần đọc/ghi một phép xoay | Đuôi gần như mất |
| Adam | Chuẩn hóa theo từng tọa độ | Nghi phạm chính |

### Kết luận ngắn

Residual stream về lý thuyết không có privileged basis, và thí nghiệm xoay cho thấy về chức năng điều đó đúng. Tuy vậy, mô hình huấn luyện thật vẫn tạo ra các kênh outlier thẳng trục. Paper loại trừ LayerNorm và sai số float, định vị đuôi dày ở basis tính toán, và nghi Adam là nguyên nhân. Với pruning và KD, bài học là phân biệt "đặc quyền về chức năng" với "đặc quyền về thống kê" khi chọn kênh để cắt hoặc để khớp.

**Đọc bản gốc:** https://transformer-circuits.pub/2023/privileged-basis/index.html
