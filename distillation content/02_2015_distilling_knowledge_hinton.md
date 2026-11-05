# Distilling the Knowledge in a Neural Network (2015)

**Tác giả:** Geoffrey Hinton, Oriol Vinyals, Jeff Dean (Google)  
**Venue / arXiv:** arXiv:1503.02531 (v1, 9/3/2015). Bản text không ghi venue; theo kiến thức của chúng ta bài được trình bày tại NIPS 2014 Deep Learning Workshop (không kiểm chứng từ file).  
**Nguồn:** https://arxiv.org/abs/1503.02531  
**Chủ đề:** Knowledge distillation (KD) bằng soft target có nhiệt độ; chứng minh "khớp logit" là trường hợp đặc biệt; ensemble chuyên gia (specialists); soft target như regularizer.  
**Vị trí trong lộ trình:** Đọc sau 00 (preface) và 01 (Ba & Caruana). Bài này đặt tên "distillation", đưa ra công thức chuẩn $T$ + $T^2$ mà gần như mọi bài sau dùng lại. Tiếp theo: 03 (FitNets: thêm hint ở hidden layer), 04 (attention transfer), 05 (survey làm bản đồ), 06–07 (self/online distillation).

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch paper. Phần "Ứng dụng cho dự án" và "Thí nghiệm gợi ý" là suy luận của chúng ta.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
4. [Thí nghiệm và kết quả chính](#4-thí-nghiệm-và-kết-quả-chính)
5. [Phân tích / ablation](#5-phân-tích--ablation)
6. [Giới hạn](#6-giới-hạn)
7. [Liên hệ với các bài khác](#7-liên-hệ-với-các-bài-khác)
8. [Ứng dụng cho dự án (suy luận)](#8-ứng-dụng-cho-dự-án-suy-luận)
9. [Thí nghiệm gợi ý](#9-thí-nghiệm-gợi-ý)
10. [Tóm tắt công thức / khái niệm](#10-tóm-tắt-công-thức--khái-niệm)
11. [Kết luận ngắn](#11-kết-luận-ngắn)

---

## 1. Bối cảnh

Trung bình dự đoán của nhiều mô hình (ensemble) là cách đơn giản để tăng độ chính xác, nhưng triển khai thì đắt. Paper dùng phép ẩn dụ côn trùng: ấu trùng tối ưu để hấp thu dinh dưỡng, con trưởng thành tối ưu để di chuyển và sinh sản. Tương tự, giai đoạn *huấn luyện* cho phép mô hình cồng kềnh (cumbersome), còn giai đoạn *triển khai* cần mô hình nhẹ. Cầu nối là "distillation".

**Rào cản khái niệm:** ta hay đồng nhất "kiến thức" với giá trị tham số, nên khó thấy làm sao đổi dạng mô hình mà giữ kiến thức. Paper đề nghị nhìn kiến thức là **ánh xạ từ vector đầu vào sang vector đầu ra** đã học được, không gắn với tham số cụ thể.

**Dark knowledge (thuật ngữ sau này, paper không dùng từ này trong bản này):** khi huấn luyện, xác suất mô hình gán cho các lớp *sai* tuy nhỏ nhưng tỷ lệ giữa chúng mang thông tin. Ví dụ paper: ảnh xe BMW có xác suất nhỏ bị nhầm với xe tải rác, nhưng vẫn lớn gấp nhiều lần xác suất nhầm với củ cà rốt.

**Công trình trước:** Buciluǎ et al. 2006 (xem bài 01) dùng logit và L2 để khớp. Hinton nói cách tổng quát hơn: tăng nhiệt độ softmax.

**Thuật ngữ:**
- **Soft target:** phân phối xác suất (mềm) của teacher dùng làm nhãn.
- **Transfer set:** tập dữ liệu dùng ở giai đoạn truyền kiến thức (có thể là tập train gốc hoặc dữ liệu không nhãn).
- **Temperature $T$:** tham số làm phẳng phân phối softmax.

## 2. Ý tưởng chính

Dùng softmax có nhiệt độ cao để teacher cho ra phân phối "mềm", lộ ra cấu trúc tương đồng giữa các lớp. Huấn luyện student khớp phân phối mềm đó (cùng nhiệt độ cao), đồng thời cộng thêm một loss nhỏ với nhãn thật. Khi suy luận, student dùng $T=1$. Paper còn chứng minh rằng khớp logit (kiểu Buciluǎ/Ba & Caruana) chính là giới hạn nhiệt độ cao của distillation. Thí nghiệm cho thấy student cùng cỡ với mô hình gốc có thể nhận phần lớn lợi ích của ensemble 10 mô hình.

## 3. Phương pháp chi tiết

### 3.1 Softmax có nhiệt độ

$$
\boxed{q_i=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}}
$$

$T=1$ là softmax thường; $T$ lớn làm phân phối phẳng hơn (entropy cao), $T\to\infty$ tiến đến đồng đều.

### 3.2 Loss kết hợp

Paper dùng trung bình có trọng số của hai cross-entropy:
1. CE với soft target: student dùng softmax ở nhiệt độ $T$, teacher dùng cùng nhiệt độ $T$ để sinh $p$.
2. CE với nhãn thật: dùng cùng logit của student nhưng ở $T=1$.

Paper nói kết quả tốt nhất thường đến từ **trọng số thấp hơn đáng kể cho loss thứ hai**. Vì độ lớn gradient của soft target co theo $1/T^2$, cần nhân loss thứ nhất với $T^2$ để tỷ lệ đóng góp giữa hai loss không đổi khi thay $T$.

Dạng viết thường gặp (là cách viết của chúng ta, tương thích mô tả paper):

$$
\mathcal L=\alpha\,T^2\,\mathrm{CE}\big(p^{(T)}_{teacher},\,q^{(T)}_{student}\big)+(1-\alpha)\,\mathrm{CE}\big(y,\,q^{(1)}_{student}\big)
$$

(Với phân phối teacher cố định, CE khác KL chỉ một hằng số entropy của teacher, nên gradient như nhau.)

### 3.3 Gradient và vì sao có $T^2$ (Eq. 2–4)

Gọi $z_i$ là logit student, $v_i$ logit teacher, $q_i$ và $p_i$ là softmax ở nhiệt độ $T$ tương ứng. Gradient CE theo logit:

$$
\frac{\partial C}{\partial z_i}=\frac{1}{T}(q_i-p_i)=\frac1T\left(\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}-\frac{e^{v_i/T}}{\sum_j e^{v_j/T}}\right)
$$

Nếu $T$ lớn so với độ lớn logit, khai triển bậc nhất $e^{x}\approx 1+x$ với $N$ lớp:

$$
\frac{\partial C}{\partial z_i}\approx\frac1T\left(\frac{1+z_i/T}{N+\sum_j z_j/T}-\frac{1+v_i/T}{N+\sum_j v_j/T}\right)
$$

Nếu logit đã được trung tâm hoá cho từng mẫu ($\sum_j z_j=\sum_j v_j=0$):

$$
\boxed{\frac{\partial C}{\partial z_i}\approx\frac{1}{NT^2}(z_i-v_i)}
$$

Hai điều rút ra:
- Ở nhiệt độ cao, distillation tương đương cực tiểu hoá $\tfrac12(z_i-v_i)^2$, tức **khớp logit là trường hợp đặc biệt** của distillation (đúng với điều kiện logit trung tâm hoá).
- Gradient co theo $1/T^2$, vì vậy nhân $T^2$ để bù.

Ở nhiệt độ thấp hơn, distillation ít chú ý các logit rất âm so với trung bình. Điều này có thể có lợi (logit đó gần như không bị ràng buộc bởi loss huấn luyện teacher nên có thể nhiễu) hoặc có hại (có thể chứa thông tin hữu ích); paper nói đây là câu hỏi thực nghiệm, và khi student quá nhỏ thì nhiệt độ trung bình tốt nhất, gợi ý rằng bỏ qua logit rất âm là có ích.

### 3.4 Ví dụ số nhỏ (của chúng ta)

Logit teacher $v=[10,5,0]$. Với $T=1$: $p\approx[0.9933,\,0.0066,\,0.00005]$ (gần one-hot). Với $T=5$: $v/T=[2,1,0]$, $p\approx[0.665,\,0.245,\,0.090]$. Ở $T=5$ student thấy rõ lớp 2 gần lớp 1 hơn lớp 3; ở $T=1$ thông tin này hầu như biến mất khỏi loss.

### 3.5 Ensemble làm teacher

Khi teacher là ensemble, soft target là trung bình số học hoặc hình học của các phân phối thành viên.

### 3.6 Pseudo-code

```
# Hinton distillation
for x, y in loader:                      # transfer set (có thể có hoặc không có y)
    with no_grad: v = teacher(x)         # logit teacher
    z = student(x)
    p = softmax(v / T)
    q = softmax(z / T)
    L_soft = T**2 * cross_entropy(p, q)  # = T^2 * (KL + const)
    L_hard = cross_entropy(y, softmax(z))# T = 1, nếu có nhãn
    L = alpha * L_soft + (1 - alpha) * L_hard    # alpha lớn (L_hard trọng số thấp)
    step(L)
inference: softmax(student(x))           # T = 1
```

## 4. Thí nghiệm và kết quả chính

### 4.1 MNIST (thử nghiệm sơ bộ)

- Teacher: 2 hidden layer × 1200 ReLU, dropout + weight constraints, jitter ảnh tới 2 pixel: **67 lỗi test**.
- Student 800-800 không regularization (hard label): **146 lỗi**.
- Student cùng cỡ học thêm khớp soft target của teacher ở $T=20$: **74 lỗi**. Paper nhấn mạnh student nhận được cả kiến thức về tổng quát hoá từ dữ liệu dịch chuyển dù transfer set không chứa ảnh dịch chuyển.
- Nếu hidden layer ≥ 300 unit, mọi nhiệt độ > 8 cho kết quả gần nhau; nếu giảm còn 30 unit/lớp, nhiệt độ 2.5–4 tốt hơn rõ rệt.
- **Bỏ hết chữ số 3 khỏi transfer set:** student chỉ mắc 206 lỗi test, trong đó 133 lỗi nằm trên 1010 chữ số 3 của tập test; nguyên nhân chính là bias của lớp 3 quá thấp. Tăng bias lớp 3 thêm 3.5 thì còn 109 lỗi, trong đó 14 lỗi trên chữ 3 (đúng 98.6% các chữ 3 dù chưa từng thấy 3).
- Nếu transfer set chỉ chứa chữ 7 và 8: 47.3% lỗi; giảm bias của 7 và 8 đi 7.6 thì còn 13.2% lỗi.

### 4.2 Nhận dạng giọng nói (acoustic model)

Mô hình: 8 hidden layer × 2560 ReLU, softmax 14,000 nhãn (trạng thái HMM), đầu vào 26 frame × 40 filter-bank, ~85M tham số, ~2000 giờ dữ liệu (~700M mẫu). Ensemble 10 mô hình khởi tạo ngẫu nhiên khác nhau. Nhiệt độ thử: [1, 2, 5, 10]; trọng số 0.5 cho CE hard target.

| Hệ thống | Frame accuracy (test) | WER |
|---|---|---|
| Baseline | 58.9% | 10.9% |
| Ensemble 10 mô hình | 61.1% | 10.7% |
| Distilled single model | 60.8% | 10.7% |

Hơn 80% mức cải thiện frame accuracy của ensemble được chuyển sang student cùng cỡ baseline. Paper so sánh với Li et al. (2014): họ distill ở $T=1$ trên dữ liệu không nhãn lớn, chỉ thu hẹp 28% khoảng cách lỗi.

### 4.3 Soft target như regularizer (mục 6)

Huấn luyện mô hình 85M tham số chỉ trên 3% dữ liệu (~20M mẫu):

| Hệ thống & tập train | Train frame acc | Test frame acc |
|---|---|---|
| Baseline (100% data) | 63.4% | 58.9% |
| Baseline (3% data) | 67.3% | 44.5% |
| Soft targets (3% data) | 65.4% | 57.0% |

Baseline 3% overfit nặng (phải early stopping, accuracy giảm mạnh sau khi đạt 44.5%); soft target không cần early stopping, hội tụ ở 57%, chỉ kém khoảng 2% so với dùng toàn bộ dữ liệu.

### 4.4 Specialists trên JFT

JFT: tập nội bộ Google, 100M ảnh có nhãn, 15,000 nhãn; mô hình nền huấn luyện khoảng 6 tháng. Ý tưởng: một generalist + nhiều specialist, mỗi specialist chỉ phân biệt một cụm lớp dễ nhầm (các lớp khác gộp thành "dustbin class"). Specialist khởi tạo từ trọng số generalist, huấn luyện với 50% mẫu từ tập con đặc biệt và 50% lấy ngẫu nhiên; hiệu chỉnh dustbin bằng cách cộng log của tỉ lệ oversampling vào logit dustbin. Cụm lớp lấy bằng online K-means trên cột ma trận hiệp phương sai của dự đoán generalist. Suy luận: tìm phân phối $q$ cực tiểu hoá $\mathrm{KL}(p_g,q)+\sum_{m\in A_k}\mathrm{KL}(p_m,q)$ bằng gradient descent theo logit cho từng ảnh.

| Hệ thống | Conditional test acc | Test acc |
|---|---|---|
| Baseline | 43.1% | 25.0% |
| + 61 specialist (mỗi cái 300 lớp + dustbin) | 45.9% | 26.1% |

Cải thiện tương đối 4.4% test accuracy; càng nhiều specialist phủ lớp đúng thì mức tăng càng lớn (Table 4: từ +3.4% khi 1 specialist phủ đến +14.1% khi 10 trở lên; nhóm 9 specialist +16.6%). Specialist huấn luyện trong vài ngày thay vì nhiều tuần. Paper nói rõ **chưa** chứng minh có thể distill specialist trở lại một mô hình đơn.

## 5. Phân tích / ablation

- **Nhiệt độ:** với student đủ lớn, mọi $T>8$ gần như nhau; với student rất nhỏ (30 unit), $T$ trung bình (2.5–4) tốt hơn.
- **Hard label phụ:** thêm một loss nhỏ với nhãn thật hữu ích vì student thường không khớp hoàn toàn soft target và "lệch về hướng đúng" giúp ích.
- **Soft target giàu thông tin:** entropy cao nghĩa là nhiều thông tin hơn mỗi mẫu và phương sai gradient giữa các mẫu thấp hơn, nên student có thể huấn luyện với ít dữ liệu hơn và learning rate cao hơn (theo paper).
- **Transfer set:** có thể là dữ liệu không nhãn hoặc tập train gốc.
- Bảng 5 là bằng chứng mạnh nhất cho vai trò regularizer.

## 6. Giới hạn

**Do tác giả nêu:**
- Chưa distill specialist ngược về một mô hình đơn.
- Đề xuất dùng soft target để chống overfit cho specialist với full softmax nhưng "đang khám phá", chưa có kết quả.
- Câu hỏi nhiệt độ thấp bỏ qua logit rất âm là lợi hay hại là vấn đề thực nghiệm.

**Do chúng ta nhận thấy:**
- Chỉ thử bài toán phân loại (MNIST, ASR theo frame, JFT); không có chuỗi sinh tự hồi quy, nên không đề cập exposure bias hay vấn đề tokenizer.
- Student cùng kích thước với teacher đơn trong thí nghiệm giọng nói; chưa kiểm tra student nhỏ hơn nhiều về capacity gap.
- JFT là dữ liệu nội bộ, không tái lập được. Kết quả MNIST với "chữ số 3 chưa thấy" cần chỉnh bias thủ công bằng tập test, nên không phải kết quả sạch.
- Hiệu quả với bảng đầy đủ ablation $\alpha$, $T$ không được trình bày ngoài mô tả văn bản (chỉ nêu tập $T$ thử và trọng số 0.5).

## 7. Liên hệ với các bài khác

- **01 Ba & Caruana:** logit regression là giới hạn $T\to$ lớn của bài này (Eq. 4); Hinton cũng nói student cần transfer set không nhãn hoặc tập train.
- **03 FitNets, 04 Attention transfer:** thêm tín hiệu ở hidden layer, vì soft target ở output có thể không đủ khi student rất sâu và mảnh.
- **05 KD survey:** bản đồ phân loại; bài này là response-based KD cơ bản.
- **06 Born-again, 07 Deep mutual learning:** cùng loss nhưng teacher cùng kích thước hoặc student học lẫn nhau.
- **08 Teacher assistant, 09, 10, 41:** khi student quá nhỏ, nhiệt độ trung bình/teacher trung gian; bài này mới chỉ gợi ý ở thí nghiệm 30 unit.
- **14 Decoupled KD:** tách KL thành phần target-class và non-target-class, liên quan trực tiếp đến "dark knowledge ở lớp sai".
- **15 Sequence-level KD, 28–34:** KL token-level trên từ vựng là bản mở rộng của loss này cho LLM.
- **45 Sparse logit sampling:** khi từ vựng lớn, không thể lưu toàn bộ logit; ước lượng thưa và độ chệch.

## 8. Ứng dụng cho dự án (suy luận)

- **Công thức nền:** KL(teacher 4B || student) trên phân phối token ở mỗi vị trí, nhân $T^2$ nếu $T\neq1$, cộng CE nhỏ với ground-truth OCR. Đây là baseline đầu tiên nên chạy khi phục hồi mô hình đã prune.
- **Hiểu "dark knowledge" trong OCR:** phân phối token của teacher cho thấy các ký tự dễ nhầm (O/0, l/1/I, dấu tiếng Việt). Với văn bản OCR gần như xác định (teacher rất tự tin) thông tin ở đuôi phân phối nhỏ; $T>1$ có thể giúp lộ nó. Cần đo, vì teacher OCR rất tự tin có thể làm soft target gần one-hot (xem preface).
- **Soft target như regularizer (Bảng 5):** nếu dữ liệu OCR có nhãn ít, chạy teacher trên nhiều ảnh không nhãn để tăng transfer set, kỳ vọng giảm overfit của student đã prune.
- **Nhiệt độ cho student rất nhỏ (0.1B):** kết quả "student quá nhỏ thì $T$ trung bình tốt hơn" gợi ý quét $T\in\{1,2,4\}$ riêng cho từng mức prune. Chưa có bằng chứng cho LLM/VLM trong paper này.
- **Specialists:** ít liên quan trực tiếp; có thể tương tự về ý tưởng "teacher chuyên cho từng loại tài liệu" nhưng tốn chi phí.
- **Về D_lin/D_res:** bài này chỉ làm việc ở output, không giúp phân tích hỏng hóc nội bộ. Nên kết hợp với feature-matching (03) tại điểm cắt.

## 9. Thí nghiệm gợi ý

1. **Quét $(T,\alpha)$:** trên student đã prune ~0.3B, thử $T\in\{1,2,4\}$ và $\alpha\in\{0.5,0.9,1.0\}$ (nhớ nhân $T^2$); báo cáo CER/WER trên tập OCR dev. Kiểm chứng $T^2$ bằng cách đặt cạnh phiên bản không nhân.
2. **Kiểm tra độ tự tin của teacher:** đo entropy trung bình phân phối token của teacher 4B trên tập OCR; nếu rất thấp, so $T=1$ và $T=2$ để xem soft target có "mở" thêm thông tin không.
3. **Ít dữ liệu:** lặp lại phép thử "3% dữ liệu" của Bảng 5: huấn luyện student với 3% nhãn thật nhưng soft target từ teacher chạy trên 100% ảnh; so với 3% hard-label.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| Softmax nhiệt độ | $q_i=\exp(z_i/T)/\sum_j\exp(z_j/T)$ |
| Loss kết hợp | $\alpha T^2\,\mathrm{CE}(p^{(T)},q^{(T)})+(1-\alpha)\,\mathrm{CE}(y,q^{(1)})$ (trọng số hard thấp) |
| Gradient theo logit | $\partial C/\partial z_i=\frac1T(q_i-p_i)$ |
| Giới hạn $T$ cao | $\partial C/\partial z_i\approx\frac{1}{NT^2}(z_i-v_i)$ khi logit zero-mean |
| Hệ quả | khớp logit (Ba & Caruana) là trường hợp đặc biệt; nhân $T^2$ để giữ tỉ lệ |
| Specialist | chỉ phân biệt cụm lớp dễ nhầm + dustbin class |
| Suy luận ensemble chuyên gia | $\min_q\ \mathrm{KL}(p_g,q)+\sum_{m\in A_k}\mathrm{KL}(p_m,q)$ |
| Soft target as regularizer | 3% data: 44.5% (hard) vs 57.0% (soft) test frame acc |

## 11. Kết luận ngắn

Bài này thiết lập chuẩn của response-based KD: softmax nhiệt độ, loss kết hợp soft/hard, nhân $T^2$, và chứng minh nó chứa phép khớp logit như trường hợp giới hạn. Thông điệp thực nghiệm đáng nhớ nhất: soft target truyền được "cách tổng quát hoá" của ensemble (hơn 80% cải thiện frame accuracy) và chống overfit mạnh (57.0% so với 44.5% khi chỉ có 3% dữ liệu). Mọi chi tiết khác (token-level, on-policy, feature) là mở rộng sau này.

**Link gốc:** https://arxiv.org/abs/1503.02531
