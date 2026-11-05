# On the Efficacy of Knowledge Distillation (2019)

**Tác giả:** Jang Hyun Cho, Bharath Hariharan (Cornell University).  
**Venue / arXiv:** arXiv:1910.01348 (v1, 03/10/2019). Bản trong file không ghi venue (đây được biết là bài ICCV 2019, nhưng thông tin này không có trong text đã đọc).  
**Nguồn:** https://arxiv.org/abs/1910.01348  
**Chủ đề:** Đánh giá thực nghiệm KD theo cặp teacher/student; "teacher lớn không dạy tốt hơn"; capacity mismatch; sequential KD không giúp; early-stopped KD (ESKD) và teacher dừng sớm.  
**Vị trí trong lộ trình:** Đọc trước: 02 (Hinton KD), 06 (BAN), 08 (Teacher assistant). Bài này thêm: kiểm chứng rộng hơn trên CIFAR10 và ImageNet cho hiện tượng "teacher mạnh hơn nhưng dạy tệ hơn", chứng minh nguyên nhân là student không bắt chước nổi (capacity mismatch) và bác bỏ giả thuyết "teacher quá nhọn"; đồng thời cho thấy cách chữa của 08 (chuỗi qua trung gian) **không hiệu quả** trong thiết lập của họ, thay vào đó chữa bằng dừng sớm. Đọc tiếp: 10 (Does KD really work), 41 (Capacity gap law).

> Đây là ghi chú học tập bằng tiếng Việt, diễn giải và tóm lược, không phải bản dịch paper. Số liệu lấy từ paper; bảng trong file trích xuất bị lộn ở vài chỗ, nơi nào đọc không chắc thì được ghi rõ. Phần "Ứng dụng cho dự án" và nhận xét ghi là suy luận là của chúng ta.

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

KD hấp dẫn vì "tổng quát": bất kỳ student nào học được từ bất kỳ teacher nào. Nhưng các bài trước thường chỉ thử vài cặp được chọn kỹ rồi ngầm giả định kết luận đúng cho mọi cặp. Cũng có vài báo cáo thất bại lẻ tẻ (ví dụ Zagoruyko & Komodakis không thành công với KD trên ImageNet). Paper này hỏi: KD thực sự hiệu quả khi nào, cặp teacher-student nào tốt, và làm gì cho các cặp xấu?

Khái niệm cần có:
- **KD cổ điển (Hinton):** với **temperature $\tau>1$**, softmax làm mềm $\tilde p_k=\dfrac{e^{s_k/\tau}}{\sum_j e^{s_j/\tau}}$, loss student:
$$L=\alpha L_{cls}+(1-\alpha)L_{KD},\qquad L_{KD}=-\tau^2\sum_k \tilde p^t_k\log \tilde p^s_k$$
  Lưu ý: ở đây $\alpha$ là trọng số của **nhãn thật** và $(1-\alpha)$ cho KD, ngược với ký hiệu của bài 08 (nơi $\lambda$ là trọng số KD). Giá trị phổ biến: $\tau\in\{3,4,5\}$, $\alpha=0.9$ (theo paper; chính xác là họ dùng $\alpha=0.9$ cho cả hai thí nghiệm CIFAR10 và ImageNet).
- **Attention transfer (AT, bài 04):** thêm loss khiến student khớp bản đồ attention của teacher; ở đây trọng số $\beta=1000$.
- **Sequential KD:** lặp KD, mô hình đời $i$ học từ đời $i-1$ (đã xuất hiện ở BAN, bài 06).

## 2. Ý tưởng chính

Điểm xuất phát: teacher chính xác hơn thường **không** dạy tốt hơn. Sau khi vượt qua một cỡ nhất định, student chính xác kém đi dù teacher tốt hơn. Cơ chế đề xuất: **capacity mismatch**, student nhỏ không tìm được nghiệm trong không gian của nó khớp với teacher lớn (KL tới teacher cao ngay cả trên tập train). Có thêm vấn đề thứ hai: với bài khó như ImageNet, student ở chế độ underfit và phải đánh đổi giữa hai loss, KD loss làm hại cross-entropy ở cuối huấn luyện. Cách chữa đơn giản: **dừng KD sớm** (chỉ dùng CE cho phần còn lại) cho student, và **dừng sớm quá trình train teacher** (teacher "regularized", hành xử như mạng nhỏ hơn). Chuỗi KD nhiều bước (như TA) không cứu được.

## 3. Phương pháp chi tiết

Paper chủ yếu là nghiên cứu thực nghiệm; hai kỹ thuật chính:

### 3.1 ESKD (Early-Stopped Knowledge Distillation) cho student

Chỉ dùng $L=\alpha L_{cls}+(1-\alpha)L_{KD}$ ở giai đoạn đầu; sau đó chuyển sang chỉ $L_{cls}$ đến hết. Động cơ: ở cuối huấn luyện, trong chế độ underfit, KD loss kéo tham số ra khỏi nghiệm cross-entropy tốt. Bằng chứng (Bảng 3): ESKD có CE train thấp hơn nhưng KD loss cao hơn so với Full KD, tức Full KD đang đánh đổi hai loss.

### 3.2 Teacher dừng sớm

Thay vì tìm kiếm kiến trúc teacher tối ưu bằng lưới (đắt), huấn luyện teacher lớn **ít epoch hơn** rồi dùng làm teacher. Ý tưởng trích dẫn: mạng lớn trained ngắn hành xử như mạng nhỏ nhưng vẫn có không gian tìm kiếm lớn. Cài đặt (CIFAR10): tổng số epoch $n\in\{35,50,65,80\}$, lr giảm 0.2 mỗi $\lfloor (n-5)/3\rfloor$ epoch. ImageNet: $n\in\{35,50\}$, lịch hạ lr (15,25,30) cho 35 và (20,35,45) cho 50. Chi phí chỉ khoảng một phần ba đến một phần tư số epoch gốc.

```
# Quy trình khuyến nghị của paper (ESKD + early-stopped teacher)
teacher = train(big_arch, epochs=n_small)           # n_small << n_full, ví dụ 35 hay 50 epoch (ImageNet)
student = train(small_arch, loss=α CE + (1-α) KD(teacher, τ), epochs=E_kd)   # giai đoạn KD
student = continue_train(student, loss=CE, epochs=E_rest)                    # giai đoạn chỉ CE
```

(Số epoch KD/CE cụ thể cho ESKD trên ImageNet không rõ từ text trích xuất; paper chỉ nói dừng KD "close to convergence".)

### 3.3 Cấu hình thực nghiệm

CIFAR10: 200 epoch, SGD momentum 0.9, lr 0.1 giảm 0.2 mỗi 60 epoch, $\tau=4$, $\alpha=0.9$, $\beta=1000$; lặp 5 lần, báo median, mean, std; kiến trúc ResNet, WideResNet, DenseNet. ImageNet: theo Zagoruyko & Komodakis, SGD Nesterov momentum 0.9, lr 0.1, weight decay $10^{-4}$, hạ lr 0.1 mỗi 30 epoch, ResNet.

## 4. Thí nghiệm và kết quả chính

**ImageNet, student ResNet18 (Bảng 1, top-1 error %):**

| Teacher | Lỗi teacher | Lỗi student |
|---|---|---|
| (không, train từ đầu) | - | 30.24 |
| ResNet18 | 30.24 | 30.57 |
| ResNet34 | 26.70 | 30.79 |
| ResNet50 | 23.85 | 30.95 |

Teacher càng mạnh, student càng kém, và **mọi** teacher đều làm student tệ hơn student train từ đầu (30.24). Trên CIFAR10 (Hình 2, dạng đồ thị): với student WRN16-1, WRN28-1, DenseNet40-12 thay đổi độ sâu hoặc độ rộng teacher, accuracy student tăng rồi giảm; số liệu cụ thể chỉ có trong hình.

**KD error trên CIFAR10 (Bảng 2, tỉ lệ mẫu mà student và teacher dự đoán khác nhau, %, cột đọc được là train / test):**

| Student | Teacher | KD error train | KD error test |
|---|---|---|---|
| WRN28-1 | WRN28-3 | 0.23 | 4.05 |
| WRN28-1 | WRN28-4 | 0.25 | 4.53 |
| WRN28-1 | WRN28-6 | 0.23 | 4.54 |
| WRN28-1 | WRN28-8 | 0.31 | 4.81 |
| WRN16-1 | WRN16-3 | 1.70 | 6.32 |
| WRN16-1 | WRN16-4 | 1.69 | 6.52 |
| WRN16-1 | WRN16-6 | 1.94 | 6.91 |
| WRN16-1 | WRN16-8 | 1.69 | 7.01 |

(Cách ghép cột train/test dựa trên thứ tự trong bảng; sự tăng dần của cột test theo cỡ teacher là điểm chính.) Student càng không khớp teacher khi teacher lớn hơn.

**ESKD trên ImageNet, ResNet18 student (Bảng 3, top-1 error test % / CE train / KD train / KD test):**

| Teacher | Lỗi | CE | KD (train) | KD (test) |
|---|---|---|---|---|
| ResNet18 | 30.57 | 0.146 | 2.916 | 3.358 |
| ResNet18 (ES KD) | 29.01 | 0.123 | 2.234 | 2.491 |
| ResNet34 | 30.79 | 0.145 | 1.357 | 1.503 |
| ResNet34 (ES KD) | 29.16 | 0.123 | 2.359 | 2.582 |
| ResNet50 | 30.95 | 0.146 | 1.553 | 1.721 |
| ResNet50 (ES KD) | 29.35 | 0.124 | 2.659 | 2.940 |

Dòng "(ES KD)" là student dùng KD dừng sớm với teacher đó. Cả ba đều thắng student train từ đầu (30.24). Dù vậy, KD loss test của student vẫn tăng theo cỡ teacher, tức vẫn là vấn đề capacity.

**Tổng hợp ImageNet (Bảng 8, top-1 error % của student ResNet18):**

| Phương pháp | Teacher | Lỗi |
|---|---|---|
| Scratch | - | 30.24 |
| Full KD | ResNet18 / 34 / 50 | 30.57 / 30.79 / 30.95 |
| Seq. Full KD (số trong paper gốc) | 3 Gen. / 6 Gen. | 30.12* / 29.6* |
| KD+ONE (số trong paper gốc) | 3 Branches | 29.45 ± 0.23* |
| Full KD + AT | ResNet34 | 30.94 (cài lại) / 29.3* (paper gốc) |
| ESKD | ResNet18 / 34 / 50 | 29.01 / 29.16 / 29.35 |
| ESKD | ResNet152 | 29.45 |
| ESKD với teacher dừng sớm | ResNet34 (50 epoch) / ResNet50 (35) / ResNet152 (35) | 29.02 / 29.05 / 29.26 |
| ESKD + AT | ResNet34 / ResNet34 (50) | 28.84 / 28.61 |

(Dấu * là số lấy từ paper khác.) Kết quả tốt nhất: ESKD + AT với teacher ResNet34 dừng sớm 50 epoch, lỗi **28.61%**, tức cải thiện khoảng 1.6 điểm so với train từ đầu (30.24) và 0.7 điểm so với số tốt nhất của attention transfer gốc (theo paper).

**Sequential KD không giúp (CIFAR10):**
- Bảng 4 (5 thế hệ, lỗi % - cột: Last Gen / ensemble tất cả thế hệ / Scratch / ensemble scratch): ResNet8 AT+KD: 13.469 / 12.786 / 12.569* / 10.176; ResNet14 AT+KD: 9.226 / 8.653 / 9.078* / 6.675; WRN16-2 KD: 6.101 / 5.181 / 6.428 / 4.865; WRN16-2 AT+KD: 5.696 / 5.310 / 6.418 / 5.003. (* nghĩa là lặp KD thậm chí làm giảm hiệu năng.) Ensemble các thế hệ KD **không** thắng ensemble cùng số model train từ đầu.
- Bảng 5 (student WRN16-1, scratch 8.759 ± 0.129; teacher WRN16-3): teacher train từ đầu (lỗi 5.34) cho student 7.68 ± 0.259; teacher đã qua 5 vòng KD (lỗi 4.89, tốt hơn) cho student 7.67 ± 0.19. Teacher chính xác hơn nhưng dạy không tốt hơn.
- Bảng 6 (Large WRN16-8 -> Medium WRN16-3 -> Small WRN16-1): Large->Med->Small cho 7.99 ± 0.24; Med->Small 7.68 ± 0.26; Large->Small 8.03 ± 0.14. Chuỗi qua trung gian gần như bằng KD trực tiếp từ Large, và **Med->Small trực tiếp** tốt nhất, dù Med kém chính xác hơn Large. (Đọc bảng này có hơi mơ hồ do mangle; con số 7.614 xuất hiện gần dòng Med->Small trong text trích xuất, có thể là một lần chạy riêng lẻ.)
- ImageNet (Bảng 7, ESKD tuần tự): Large->Small->Small: teacher thứ nhất 21.69, teacher thứ hai 29.45, student 29.41; Med->Small->Small: 23.85, 29.35, 29.35; Small->Small->Small: 30.24, 29.01, 29.15. Kết quả tốt nhất vẫn là KD một lần từ "small" sang "small".

**Teacher dừng sớm (CIFAR10, Hình 4):** với student WRN16-1 và WRN28-1 có "sweet spot" cho số epoch của teacher; mọi teacher dừng sớm đều cho student tốt hơn teacher đầy đủ tối ưu (WRN16-3 và WRN28-3).

**Chuyển giao sang Places365 (Bảng 9, top-1 / top-5 error %):** Scratch 47.38 / 18.51; ESKD ResNet18 47.09 / 18.13; Full KD ResNet34 47.86 / 18.61; ESKD ResNet34 (50) 47.14 / 18.32; Full KD ResNet50 47.92 / 18.72; ESKD ResNet50 (35) 47.02 / 18.14; ESKD ResNet152 (35) 47.25 / 18.25.

**$\tau=20$ trên ImageNet (Bảng 10, top-1 / top-5):** Scratch 30.24 / 10.92; Full KD với ResNet18 / 50 / 152: 30.75 / 11.11, 30.98 / 10.20, 31.27 / 11.59; ESKD ResNet18 / ResNet50: 29.00 / 9.91, 29.00 / 9.76; ESKD ResNet50 (35): 28.89 / 9.76.

## 5. Phân tích / ablation

- **Giả thuyết "teacher quá nhọn" bị bác:** làm mềm bằng temperature cao không đổi kết quả cho Full KD (Bảng 10 và Hình 6). Khác với giả thuyết yếu tố 3 của bài 08.
- **Giả thuyết capacity mismatch được ủng hộ:** KD error/loss cao nhất với teacher lớn nhất, ngay cả trên tập train (Bảng 2, 3).
- **Underfit và đánh đổi hai loss:** trên ImageNet, KD giúp lúc đầu rồi làm hại ở cuối (Hình 3). ESKD đạt CE train thấp hơn nhưng KD loss cao hơn, nhất quán với giả thuyết đánh đổi.
- **ESKD tương thích với AT**, và dừng sớm teacher cũng tương thích (cộng dồn: 28.61%).
- **Nhiều họ kiến trúc:** kết luận (teacher lớn không tốt hơn; teacher dừng sớm tốt hơn) lặp lại với các tổ hợp DenseNet và WideResNet (Hình 5), và với $\alpha$, $\tau$ khác nhau (Hình 6); teacher dừng sớm cũng cho kết quả tốt hơn khi chuyển sang Places365.
- **Thế hệ lặp:** hiệu quả phụ thuộc mạnh vào kiến trúc (ResNet8 và ResNet14 thế hệ cuối thậm chí tệ hơn scratch); ensemble các thế hệ tương quan nhau nên yếu hơn ensemble scratch.

## 6. Giới hạn

**Do tác giả nêu hoặc hàm ý:**
- Đề xuất teacher dừng sớm là heuristic ("some evidence" từ tài liệu khác), chưa có lý thuyết; chọn số epoch tối ưu cần tuning ("sweet spot").
- Tác giả coi tìm kiếm lưới kiến trúc teacher là quá đắt, nên tối ưu hóa kiến trúc teacher không được giải.

**Do chúng ta nhận thấy:**
- Chỉ phân loại ảnh (CIFAR10, ImageNet, Places365), mạng tích chập cùng họ; không có chuỗi, ngôn ngữ, VLM.
- Chênh lệch trên ImageNet khá nhỏ (khoảng 1.2 điểm) và nhiều bảng ImageNet chỉ một lần chạy; chỉ CIFAR10 có 5 lần lặp và std.
- Kết luận "sequential KD vô dụng" dựa trên một vài cặp WRN và ResNet và không tinh chỉnh tách biệt từng bước; nó mâu thuẫn bề ngoài với bài 08 (TAKD có lợi) chứ không bác bỏ trực tiếp, vì thiết lập hyperparameter và cách chọn TA khác nhau (TAKD tune 120 lần thử và chọn TA theo heuristic).
- Cách dừng sớm teacher giảm chất lượng teacher (lỗi teacher cao hơn), nên thực chất là trao đổi giữa chất lượng và "dễ bắt chước", không phải miễn phí.
- Dùng "KD error" (disagreement top-1) trên CIFAR vì thang đo của loss; nhưng KD error top-1 không đo khớp toàn phân phối (xem bài 10 để thấy mức khớp thực sự có thể kém hơn nhiều).

## 7. Liên hệ với các bài khác

- **08 Teacher assistant:** cùng quan sát (teacher lớn hại student) nhưng đưa ra chẩn đoán khác (08: ba yếu tố gồm "teacher quá tự tin"; 09 bác "quá nhọn" bằng temperature và nhấn mạnh capacity mismatch + underfit) và giải pháp khác (08: chuỗi qua TA; 09: không hiệu quả, dùng dừng sớm). Nên đọc cặp này cùng nhau và coi kết quả mâu thuẫn là cảnh báo rằng giải pháp phụ thuộc thiết lập.
- **06 BAN:** 09 kiểm tra lại sequential KD và thấy hiệu quả không ổn định, ensemble các thế hệ không hơn ensemble scratch.
- **07 DML:** được nhắc ở phần liên quan.
- **04 Attention transfer:** ESKD + AT cộng dồn; AT là phụ gia thường dùng.
- **10 Does KD really work:** đào sâu "student có thực sự khớp teacher" và bổ sung kết luận bi quan hơn: ngay cả khi capacity dư, khớp vẫn kém do tối ưu.
- **41 Capacity gap law, 39 Small models struggle with strong reasoners, 42 Distillation scaling laws:** phiên bản hiện đại cho LLM; 09 là bằng chứng thực nghiệm CNN sớm nhất trong lộ trình về hiện tượng này.
- **11 Patient and consistent teacher:** ngược hẳn với "dừng sớm": train rất lâu (nhiều giờ/epoch) với cùng ảnh cho cả teacher và student, kết quả lại tốt (cần xem vì sao hai kết luận có thể cùng đúng: 11 xử lý bằng cách làm cho student "khớp" teacher bằng tăng cường dữ liệu mạnh và rất nhiều epoch).

## 8. Ứng dụng cho dự án (suy luận)

Phần này là suy luận của chúng ta; **liên quan cao** vì dự án là trường hợp student nhỏ với teacher lớn (tỉ lệ 13x đến 40x).

1. **Đừng giả định teacher 4B là tối ưu cho student 0.1B.** Bài khuyến nghị coi cặp (teacher, student) là siêu tham số. Thử thêm một teacher cỡ vừa (ví dụ chính LightOnOCR-0.8B gốc) so với 4B.
2. **ESKD cho phase recovery:** huấn luyện phục hồi student đã prune với KD ở phần đầu, rồi vài bước cuối chỉ CE (nhãn OCR thật) hoặc chỉ self-consistency. Rẻ và dễ thử. Chú ý: student 0.1B có thể rơi vào chế độ underfit, đúng điều kiện bài mô tả (hai loss tranh chấp).
3. **Chẩn đoán capacity mismatch bằng độ khớp:** đo KL hay top-1 disagreement giữa student và teacher cả trên tập train (như Bảng 2). Nếu độ khớp train vẫn thấp sau nhiều bước, vấn đề là capacity/optimization, không phải thiếu dữ liệu.
4. **Teacher dừng sớm** không phù hợp với teacher 4B có sẵn (không train lại), nhưng gợi ý lấy **checkpoint trung gian** của teacher (nếu có) làm teacher thay cho checkpoint cuối. Giả thuyết cần kiểm tra.
5. **Tránh sequential KD mù quáng:** bằng chứng của 09 (CNN) cho thấy chuỗi qua trung gian không luôn hữu ích; kết quả bài 08 thì ngược lại. Với dự án, chuỗi 4B -> 0.8B -> 0.3B có sẵn "miễn phí" về mặt kiến trúc (cắt từ 0.8B), nên kiểm chứng thực nghiệm.
6. Quan sát "cắt MLP neuron phục hồi kém hơn cắt layer": bài 09 hợp với giả thuyết rằng khi student bị hạn chế cấu trúc theo cách mà nghiệm của teacher "không nằm trong không gian của student" thì KD khó bù; chỉ là giả thuyết liên hệ, không được bài kiểm chứng.

## 9. Thí nghiệm gợi ý

1. **Quét cặp teacher/student:** student 0.3B pruned được distill từ (a) 0.8B gốc, (b) 4B, (c) checkpoint sớm của 4B (nếu có); cố định ngân sách token, so CER/WER và KL train/test tới teacher.
2. **ESKD:** KD trong X% đầu của lịch huấn luyện rồi chuyển sang CE thuần; quét X trong {50, 75, 90, 100}% trên student 0.1B. Ghi cả CE train và KD loss test để xem có sự đánh đổi hai loss như Bảng 3 không.
3. **Chẩn đoán yếu tố "quá nhọn":** thử $\tau\in\{1,2,4,20\}$ cho teacher 4B (ở mức token) và xem có đổi kết quả không, giống Bảng 10.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| $L=\alpha L_{cls}+(1-\alpha)L_{KD}$ | Loss KD ở bài này; $\alpha=0.9$ trọng số nhãn thật |
| $L_{KD}=-\tau^2\sum_k\tilde p^t_k\log\tilde p^s_k$ | Cross-entropy giữa softmax làm mềm của teacher và student |
| KD error | Tỉ lệ mẫu mà dự đoán student khác teacher |
| Capacity mismatch | Student không có nghiệm khớp teacher lớn trong không gian của nó |
| Underfit/đánh đổi hai loss | Student không tối thiểu hóa cả CE và KD nên hy sinh CE |
| ESKD | Dừng KD sớm, phần còn lại chỉ CE |
| Teacher dừng sớm | Train teacher ít epoch hơn như một dạng regularization |
| Sequential KD | Student đời $i$ học từ đời $i-1$ (BAN) hoặc chuỗi qua trung gian (TA) |

## 11. Kết luận ngắn

Bài thực nghiệm này cho thấy KD không "tổng quát tự động": teacher lớn và chính xác hơn có thể dạy tệ hơn vì student không bắt chước nổi, và trên bài khó như ImageNet, KD đầy đủ còn làm hại student so với train từ đầu. Chuỗi KD qua trung gian không cứu được trong thiết lập của họ; thay vào đó dừng KD sớm và dùng teacher được train ngắn hơn cho kết quả tốt (ImageNet ResNet18: 30.24% xuống 28.61% khi kết hợp AT). Với dự án, bài là lời nhắc coi cặp teacher-student, lịch trộn loss KD/CE và độ khớp train là các biến cần đo.

Nguồn gốc: https://arxiv.org/abs/1910.01348
