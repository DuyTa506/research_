# A Toy Model of Mechanistic (Un)Faithfulness (2025)

**Tác giả:** Chris Olah (ghi chú không chính thức; ý tưởng hình thành qua thảo luận với Josh Batson, Michael Sklar, Jack Lindsey, Adly Templeton, Siddharth Mishra-Sharma). Đăng ngày 7/8/2025.  
**Nguồn:** https://transformer-circuits.pub/2025/faithfulness-toy-model/index.html  
**Chủ đề:** Mechanistic faithfulness, transcoder, datapoint feature, Jacobian matching, superposition.  
**Vị trí trong lộ trình:** Đọc sau Toy Models of Superposition (2022), Towards Monosemanticity (2023) và bài transcoder/attribution graph (2025). Quan trọng vì chỉ ra một mô hình thay thế có thể khớp output mà vẫn dùng *cơ chế khác* mô hình gốc — đúng nỗi lo của distillation.

> Ghi chú học tập diễn giải, không phải bản dịch. Bản gốc tự nhận "very preliminary", độ tin cậy thấp, giống một bài trình bày nhanh ở lab meeting. Phần ứng dụng pruning/distillation là suy luận mở rộng của chúng tôi, không phải kết quả của bài.

## Mục lục

1. [Vấn đề: thay thế tính toán, không chỉ biểu diễn](#1-vấn-đề-thay-thế-tính-toán-không-chỉ-biểu-diễn)
2. [Toy model: giá trị tuyệt đối và điểm dữ liệu lặp](#2-toy-model-giá-trị-tuyệt-đối-và-điểm-dữ-liệu-lặp)
3. [Jacobian matching](#3-jacobian-matching)
4. [Cách bài kiểm chứng](#4-cách-bài-kiểm-chứng)
5. [Giới hạn](#5-giới-hạn)
6. [Ứng dụng giả thuyết cho pruning / distillation](#6-ứng-dụng-giả-thuyết-cho-pruning--distillation)
7. [Thí nghiệm gợi ý](#7-thí-nghiệm-gợi-ý)
8. [Tóm tắt khái niệm](#8-tóm-tắt-khái-niệm)

---

## 1. Vấn đề: thay thế tính toán, không chỉ biểu diễn

- **SAE (sparse autoencoder):** mã hóa activation thành tổ hợp thưa các feature rồi tái tạo chính activation đó.
- **Transcoder:** đọc *input* của một layer (thường là MLP) và dự đoán *output* của nó, tức là thay thế phép tính.
- **Mechanistic faithfulness:** mô hình thay thế có chạy *cùng thuật toán* với mô hình gốc không, chứ không chỉ cùng output.

Khi mô hình hóa **biểu diễn**, feature chỉ là một basis; monosemantic + tái tạo tốt thì không bị lừa về nội dung. Khi mô hình hóa **tính toán**, có thể monosemantic, MSE thấp trên phân phối, nhưng bằng **cơ chế khác** — và cơ chế khác có thể tổng quát hóa khác ngoài phân phối (giống nhiều thuật toán sort cho cùng output). Mức lệch bị chặn vì transcoder phải khớp residual stream sau *từng* bước, nhưng vẫn có thể lệch.

## 2. Toy model: giá trị tuyệt đối và điểm dữ liệu lặp

Mô hình gốc: $\boxed{y_i=\mathrm{ReLU}(x_i)+\mathrm{ReLU}(-x_i)=|x_i|}$. Dữ liệu thưa: $x_i\sim\mathrm{Uniform}(-1,1)$ với xác suất `density`, còn lại bằng 0. Transcoder huấn luyện với **tanh L1** (gần L0 hơn; L1 thường cho feature trùng lặp, lộn xộn) tìm lại đúng $\mathrm{ReLU}(\pm x_i)$.

**Thêm điểm lặp** $p=[1,1,1,0,0,\dots]$, xuất hiện với tỉ lệ `repeat_frac`. Transcoder học mạch ghi nhớ:

$$\boxed{y \mathrel{+}= p\cdot \mathrm{ReLU}(\lambda\, p\cdot x - b)}$$

Feature bật đúng trên $p$ và ghi ra $p$: **datapoint feature**. Nếu mô hình gốc thật sự ghi nhớ $p$ thì đây là trung thành; nhưng $|x|$ không có khái niệm gì về $p$ → **unfaithful**.

**Hai điều kiện cần:** (1) có điểm dữ liệu lặp; (2) có ràng buộc sparsity (ghi nhớ giúp biểu diễn $p$ bằng 1 feature thay vì 3). Quét hai hyperparameter cho thấy **tích** của chúng quyết định datapoint feature có xuất hiện.

**Đo ghi nhớ.** Với đóng góp $y|_{f_i}(x)=W_\text{down}^i f_i(x)$:
$$m(f_i,p)=\mathrm{ReLU}\big(y|_{f_i}(p)\cdot\hat p-\max_{j<3} y|_{f_i}(p)\cdot e_j\big),$$
tức phần giải thích $p$ dư ra so với chiều thành phần tốt nhất; cộng trên mọi feature (bài tự nhận định nghĩa này thô).

**Trực quan hóa.** Lập "thư viện" mạch lý tưởng ($\mathrm{ReLU}(\pm x_i)$, mạch ghi nhớ), lấy vector đóng góp của mỗi mạch trên cùng một mẫu lớn, rồi dùng **compressed sensing** biểu diễn mỗi feature học được theo thư viện → ma trận feature × mạch lý tưởng + phần dư.

| Thiết lập | Quan sát |
|---|---|
| Không lặp | Gần đúng $\mathrm{ReLU}(\pm x_i)$, phần dư nhỏ |
| Lặp nhẹ (5%) | Có datapoint feature; ba feature chiều 1–3 bị **xoay** để không bật trên $p$ |

## 3. Jacobian matching

**Vì sao Jacobian?** Tại $p$, Jacobian của transcoder baseline là **chéo** (các chiều độc lập); có datapoint feature thì xuất hiện phần tử ngoài đường chéo — mỗi output "bật vì" các chiều không liên quan. Sâu hơn: trọng số không chính tắc (hoán vị neuron, scale nghịch đảo vào/ra) và cần thống kê input để biết non-linearity hoạt động ra sao. Với layer ReLU, Jacobian là **tổng outer product** trọng số vào/ra của các neuron đang bật — dạng *canonical* của tham số, kèm thông tin đồng-kích-hoạt.

$$\boxed{\mathcal L_\text{Jac}=\|J_\text{MLP}-J_\text{Transcoder}\|_F^2}\quad\approx\quad \mathbb E_v\,K\,\|J_\text{MLP}v-J_\text{Transcoder}v\|_2^2$$

$v$ là vector ngẫu nhiên; vế phải tính rẻ bằng backprop. (Cách đọc của chúng tôi về hằng số $K$ mà bài không giải thích: với $v$ đơn vị ngẫu nhiên trong $\mathbb R^K$, $K\,\mathbb E\|Av\|^2=\|A\|_F^2$.) Kết quả trong toy: chỉ số ghi nhớ giảm mạnh, datapoint feature biến mất, các feature chiều 1–3 chỉ còn hơi xoay.

**Gian lận?** Giống **gradient masking**: feature có trọng số lớn (gradient lớn) nhưng bias rất âm nên hầu như không bật, ít tốn L1 — khai thác khoảng cách L1/L0; tanh L1 giúp giảm. Có thể phạt L1 của đóng góp từng feature vào Jacobian (xấp xỉ rẻ: tích norm vào/ra, mask bởi đạo hàm hàm kích hoạt).

**Không nên khớp hoàn hảo.** Superposition làm Jacobian "nhòe": một feature trải trên 10 neuron cho 1 singular value lớn và 9 giá trị nhỏ ứng với thứ khác các neuron đó biểu diễn. Đuôi phổ này là artifact, không cần mô hình hóa.

## 4. Cách bài kiểm chứng

- Toy model có ground truth ($|x|$), nên phân biệt faithful/unfaithful là khả thi.
- Quét `repeat_frac` × sparsity, đo $m$; trực quan hóa bằng compressed sensing và xem trọng số neuron.
- So sánh Jacobian tại $p$; thêm Jacobian matching và quan sát datapoint feature biến mất.
- Con số cụ thể duy nhất trong văn bản: lặp 5% ở thiết lập "light repetition"; các kết quả khác chỉ ở dạng hình.

## 5. Giới hạn

- Tác giả tự nhận: rất sơ bộ, "low confidence"; một layer, chưa có kết quả trên mô hình thật.
- Chỉ số ghi nhớ thô, chỉ cho một điểm $p$ biết trước. Jacobian matching không phải lời giải lý tưởng (gradient masking; superposition).
- Bài tách lo ngại về SAE/transcoder thành **Mechanistic faithfulness** (đúng cơ chế không — cốt lõi, là một phổ) và **Simplicity** (mô tả có gọn nhất không, ví dụ feature manifold bị chia thành nhiều feature rời — chỉ là "thuế" độ khó hiểu).

## 6. Ứng dụng giả thuyết cho pruning / distillation

> **Suy luận của chúng tôi, không phải kết luận của bài.**

- **Student như một transcoder của teacher.** Hidden-state KD ép khớp activation trên phân phối huấn luyện — đúng tình huống bài cảnh báo: MSE thấp không đảm bảo cùng cơ chế, nhất là ngoài phân phối (chữ viết tay, bố cục lạ trong OCR).
- **Bổ sung cho D_lin / D_res.** Map tuyến tính tối ưu tại vết cắt chỉ khớp *giá trị* activation theo trung bình bình phương. Có thể đo thêm "D_Jac" = sai khác JVP $\|J_Tv-J_Sv\|$ giữa block teacher bị thay và student. Hai cấu hình có cùng $D_\text{lin}$ có thể khác xa về D_Jac.
- **Neuron cut vs. layer cut.** Jacobian MLP là tổng các số hạng hạng-1 phụ thuộc input (neuron đang bật); xóa neuron làm mất hẳn các số hạng đó, thứ một map tuyến tính cố định không bù được — một giả thuyết cho việc cắt neuron khó phục hồi hơn. Cần đo.
- **Template lặp trong dữ liệu OCR** (header, form, watermark) + áp lực capacity của student nhỏ (tương tự sparsity) có thể sinh "datapoint feature" thay cho cơ chế đọc chung → dedup hoặc theo dõi.
- **Đừng khớp Jacobian quá chặt:** student hẹp không thể khớp đuôi phổ do superposition; chỉ nên khớp top-$k$ hướng singular.

## 7. Thí nghiệm gợi ý

1. **Tái hiện toy:** transcoder trên $|x|$ với $d=20$, quét `repeat_frac` ∈ {0, 1%, 5%, 20%} × hệ số tanh-L1, đo $m$; thêm term JVP và so sánh.
2. **Jacobian-KD tại vết cắt:** thêm $\|J_Tv-J_Sv\|^2$ (`torch.func.jvp`) vào hidden-state KD cho một block GatedDeltaNet bị cắt; so sánh với KD thuần MSE bằng CER trên tập OCR *ngoài* phân phối.
3. **Phổ Jacobian:** top singular values của Jacobian MLP teacher vs. student trên vài trăm token OCR; student mất hướng lớn hay chỉ mất đuôi?

## 8. Tóm tắt khái niệm

| Khái niệm | Công thức / định nghĩa | Ý nghĩa |
|---|---|---|
| Ground truth | $y_i=\mathrm{ReLU}(x_i)+\mathrm{ReLU}(-x_i)$ | Cơ chế thật, đã biết |
| Datapoint feature | $y \mathrel{+}= p\,\mathrm{ReLU}(\lambda p\cdot x-b)$ | Ghi nhớ điểm lặp $p$ |
| Điều kiện | lặp × sparsity | Cả hai cùng cần |
| Ghi nhớ | $m(f_i,p)$ | Giải thích $p$ dư ra ngoài từng chiều |
| Jacobian matching | $\|J_\text{MLP}-J_\text{TC}\|_F^2$ qua $Jv$ ngẫu nhiên | Ép cùng cơ chế cục bộ |
| Gradient masking | trọng số lớn + bias rất âm | Cách gian lận Jacobian loss |
| Faithfulness / Simplicity | đúng cơ chế / mô tả gọn | Hai loại lo ngại về transcoder |

### Kết luận ngắn

Khi chuyển từ mô hình hóa biểu diễn sang mô hình hóa tính toán, khớp output trên phân phối là chưa đủ: ngay toy model đơn giản cũng cho transcoder ghi nhớ thứ mô hình gốc không ghi nhớ. Jacobian matching là hướng sửa hứa hẹn (trong toy). Với distillation, đây là lời nhắc nên đo thêm sự khớp *đạo hàm*, không chỉ khớp activation.

**Đọc bản gốc:** https://transformer-circuits.pub/2025/faithfulness-toy-model/index.html
