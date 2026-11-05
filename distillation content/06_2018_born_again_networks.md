# Born Again Neural Networks (2018)

**Tác giả:** Tommaso Furlanello, Zachary C. Lipton, Michael Tschannen, Laurent Itti, Anima Anandkumar.  
**Venue / arXiv:** arXiv:1805.04770 (v2, 29/06/2018); Proceedings of ICML 2018 (PMLR 80), Stockholm.  
**Nguồn:** https://arxiv.org/abs/1805.04770  
**Chủ đề:** Self-distillation (student cùng kiến trúc với teacher), phân rã gradient của KD thành "dark knowledge" và "importance weighting", chuỗi nhiều thế hệ (BAN), ensemble các thế hệ.  
**Vị trí trong lộ trình:** Đọc trước: 02 (Hinton KD, định nghĩa soft target và dark knowledge), 01 (Ba & Caruana, logit matching). Bài này thêm: KD không cần nén, teacher và student có thể cùng cỡ, thậm chí student mạnh hơn teacher; đồng thời đặt câu hỏi "cái gì trong teacher output thực sự có ích". Đọc tiếp: 07 (Deep mutual learning, bỏ luôn teacher cố định), 08 (Teacher assistant, trường hợp ngược lại: student quá nhỏ so với teacher), 10 (Does KD really work).

> Đây là ghi chú học tập bằng tiếng Việt, diễn giải và tóm lược, không phải bản dịch paper. Các số liệu lấy từ paper; phần "Ứng dụng cho dự án" và các nhận xét ghi rõ là suy luận là của chúng ta, không phải kết luận của tác giả.

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

**Knowledge distillation (KD)** = huấn luyện một mô hình *student* bắt chước phân phối đầu ra (softmax) của một mô hình *teacher* đã train xong, thay vì (hoặc cùng với) học từ nhãn cứng one-hot. Trước bài này, hầu như mọi công trình (Bucilua 2006, Ba & Caruana 2014, Hinton 2015, FitNets, Attention Transfer) đều coi KD là **công cụ nén mô hình**: teacher lớn, student nhỏ, student không bao giờ qua mặt teacher, chỉ "gần hơn" so với train thẳng từ nhãn.

Hinton giải thích thành công của KD bằng **dark knowledge**: xác suất của các lớp sai (ví dụ "xe tải" có xác suất cao hơn "con mèo" khi ảnh là "ô tô") chứa thông tin về độ tương tự giữa các lớp. Bài này đặt hai câu hỏi mới:

1. Nếu bỏ yếu tố nén (student cùng kiến trúc, cùng số tham số với teacher), KD còn có ích không?
2. Có đúng là "dark knowledge" là nguồn lợi ích chính không, hay còn cơ chế khác (ví dụ KD chỉ là một dạng đánh trọng số mẫu)?

Tiền đề cần nhớ: **cross-entropy** giữa phân phối mục tiêu $p$ và dự đoán $q$ là $-\sum_i p_i\log q_i$; đạo hàm theo logit $z_i$ của softmax là $q_i - p_i$ (rất đẹp và dùng ở dưới).

## 2. Ý tưởng chính

Train một teacher bình thường từ nhãn. Rồi khởi tạo một student **y hệt kiến trúc** với seed ngẫu nhiên khác và train nó khớp output của teacher (có thể kèm nhãn). Student thu được (gọi là **Born-Again Network, BAN**) thường **tốt hơn teacher**. Lặp lại: dùng BAN làm teacher cho thế hệ kế tiếp (sequence of teaching selves), cuối cùng lấy trung bình dự đoán của các thế hệ để ra ensemble (BANE). Tác giả còn phân rã gradient của KD thành hai phần và thiết kế hai thí nghiệm (CWTM, DKPP) để xem phần nào của output teacher thực sự có ích. Kết luận: cả thông tin trên lớp đúng (đóng vai importance weight) lẫn cấu trúc bậc cao của phân phối trên lớp sai đều đóng góp.

## 3. Phương pháp chi tiết

### 3.1 Quy trình BAN

ERM chuẩn: $\theta_1^*=\arg\min_{\theta_1} L(y, f(x,\theta_1))$. BAN thế hệ 2 học từ output của thế hệ 1:

$$\theta_2^*=\arg\min_{\theta_2} L\big(f(x,\theta_1^*),\, f(x,\theta_2)\big)$$

với $L$ là cross-entropy giữa hai output. Thế hệ $k$ học từ thế hệ $k-1$:

$$\theta_k^*=\arg\min_{\theta_k} L\big(f(x,\theta_{k-1}^*),\, f(x,\theta_k)\big)$$

Ensemble BANE sau $k$ thế hệ:

$$\boxed{\hat f_k(x)=\frac{1}{k}\sum_{i=1}^{k} f(x,\theta_i)}$$

Chi tiết quan trọng (paper nêu): lịch train của student giữ nguyên lịch của teacher; **không dùng temperature** (paper nói họ lần đầu cho thấy dark knowledge trong self-distillation hiệu quả "ngay cả khi không làm mềm logit"); hai biến thể BAN: "BAN" chỉ dùng loss theo teacher, "BAN+L" dùng cả loss nhãn và loss teacher. Các biến thể ép student giống teacher hơn (chia sẻ lớp đầu/cuối, thêm loss L2 giữa activation) được thử nhưng "systematically perform slightly worse" so với KD cross-entropy đơn giản.

Pseudo-code:

```
T1 = train(arch, data, labels, seed=1)            # teacher thường
for k in 2..K:
    Tk = init(arch, seed=k)                        # cùng kiến trúc, seed khác
    for batch (x, y) in data:
        p = softmax(T_{k-1}(x))                    # đóng băng, không temperature
        q = softmax(Tk(x))
        loss = CE(p, q)  [+ CE(y, q) nếu BAN+L]
        update Tk
ensemble(x) = mean_k softmax(Tk(x))                # BANE
```

### 3.2 Gradient của KD: "dark knowledge" hay "importance weighting"?

Với logit $z_i$ của student, $q_i=\mathrm{softmax}(z)_i$ và teacher $p_i$, gradient cross-entropy theo logit thứ $i$:

$$\frac{\partial L_i}{\partial z_i}=q_i-p_i$$

Khi mục tiêu là nhãn one-hot $y$ thì gradient là $q_*-y_*$ (chỉ lớp đúng $*$ là khác). Với KD, trên một minibatch $b$ mẫu (kể cả $1/b$), gradient tách thành hai tổng:

$$\sum_{s=1}^{b}\sum_{i=1}^{n}\frac{\partial L_{i,s}}{\partial z_{i,s}}=\underbrace{\sum_{s=1}^{b}(q_{*,s}-p_{*,s})}_{\text{lớp đúng}}+\underbrace{\sum_{s=1}^{b}\sum_{i=1}^{n-1}(q_{i,s}-p_{i,s})}_{\text{lớp sai: dark knowledge}}$$

Số hạng đầu có thể viết lại thành $\frac1b\sum_s (q_{*,s}-p_{*,s}\,y_{*,s})$ (paper viết dưới dạng này): tức gradient của nhãn thật **nhân với $p_{*,s}$**, độ tự tin của teacher trên lớp đúng. Khi teacher đúng và rất tự tin ($p_*\approx 1$) thì nó trở về gradient nhãn thật; mẫu mà teacher không chắc thì bị giảm trọng số. Đó là **importance weighting** (đánh trọng số mẫu): mẫu "dễ" (teacher chắc) được nhấn mạnh, mẫu "khó/nhiễu" bị hạ.

### 3.3 Hai thí nghiệm tách thành phần

- **CWTM (Confidence-Weighted by Teacher Max):** loại bỏ toàn bộ thông tin trên lớp sai. Train student bằng cross-entropy với nhãn thật, nhưng nhân loss mỗi mẫu với trọng số tỉ lệ $\max p_{\cdot,s}$ (xác suất lớn nhất của teacher, kể cả khi teacher sai), chuẩn hóa theo batch:

$$\sum_{s=1}^{b}\frac{\max p_{\cdot,s}}{\sum_{u=1}^{b}\max p_{\cdot,u}}\,(q_{*,s}-y_{*,s})$$

- **DKPP (Dark Knowledge with Permuted Predictions):** giữ nguyên giá trị argmax của teacher nhưng **xáo ngẫu nhiên** các xác suất của các lớp còn lại ($\phi$ là phép hoán vị). Việc này phá quan hệ "lớp nào giống lớp nào" (ma trận hiệp phương sai output), nhưng giữ nguyên các moment bậc cao bất biến với hoán vị (độ "nhọn", entropy...).

Logic: nếu lợi ích của KD chỉ đến từ việc lớp sai tương đồng với lớp đúng thì DKPP phải mất hết tác dụng; nếu chỉ do weighting thì CWTM phải đạt gần bằng BAN.

### 3.4 Đổi kiến trúc student

Ngoài cùng kiến trúc, tác giả: (a) thay đổi depth/growth/compression rate của DenseNet student cho cùng kích thước hidden state tại mỗi chỗ giảm độ phân giải; (b) "DenseNet to ResNet": student là Wide-ResNet hay bottleneck Pre-ResNet khớp shape từng block của DenseNet-90-60 (dùng chung lớp đầu conv1 và lớp cuối fc, bị đóng băng); (c) "ResNet to DenseNet": teacher là Wide-ResNet, student DenseNet-90-60.

## 4. Thí nghiệm và kết quả chính

Dữ liệu và mô hình: CIFAR-10, CIFAR-100 (DenseNet các cỡ, Wide-ResNet-28-k), Penn Tree Bank (LSTM 1500 đơn vị và CNN-LSTM của Kim et al.). Preprocessing và hyperparameter theo Wide-ResNet (Zagoruyko & Komodakis), ngoại trừ chuẩn hóa mean-std.

**CIFAR-10 test error (%), Bảng 1** (Teacher / BAN):

| Mạng | Tham số | Teacher | BAN |
|---|---|---|---|
| Wide-ResNet-28-1 | 0.38 M | 6.69 | 6.64 |
| Wide-ResNet-28-2 | 1.48 M | 5.06 | 4.86 |
| Wide-ResNet-28-5 | 9.16 M | 4.13 | 4.03 |
| Wide-ResNet-28-10 | 36 M | 3.77 | 3.86 |
| DenseNet-112-33 | 6.3 M | 3.84 | 3.61 |
| DenseNet-90-60 | 16.1 M | 3.81 | 3.5 |
| DenseNet-80-80 | 22.4 M | 3.48 | 3.49 |
| DenseNet-80-120 | 50.4 M | 3.37 | 3.54 |

Lưu ý: ở CIFAR-10 BAN không thắng ở mọi cấu hình (Wide-ResNet-28-10, DenseNet-80-120 lại tệ hơn; DenseNet-80-80 gần như hòa). Số 3.5% trong abstract tương ứng DenseNet-90-60.

**CIFAR-100 test error (%), Bảng 2** (trích; BAN chỉ dùng teacher loss, BAN+L thêm nhãn):

| Mạng | Teacher | BAN | BAN+L | CWTM | DKPP | BAN-2 | BAN-3 | Ens*2 | Ens*3 |
|---|---|---|---|---|---|---|---|---|---|
| DenseNet-112-33 | 18.25 | 16.95 | 17.68 | 17.84 | 17.84 | 17.22 | 16.59 | 15.77 | 15.68 |
| DenseNet-90-60 | 17.69 | 16.69 | 16.93 | 17.42 | 17.43 | 16.44 | 16.72 | 15.39 | 15.74 |
| DenseNet-80-80 | 17.16 | 16.36 | 16.5 | 17.16 | 16.84 | 16.30 | 15.5 | 15.46 | 15.14 |
| DenseNet-80-120 | 16.87 | 16.00 | 16.41 | 17.12 | 16.34 | 16.13 | 16.13 | 15.13 | 14.9 |

(Cột BAN-1 bị bỏ khỏi bảng này cho gọn; BAN-1 là một BAN khác seed. Với DenseNet-80-120 không có BAN-3 nên Ens*3 bao gồm teacher.)

Các con số nổi bật được paper nêu: BAN-3-DenseNet-80-80 (22M tham số) đạt **15.5%** trên CIFAR-100, mô hình đơn tốt nhất mà họ biết khi không dùng shake-shake; BAN-ensemble lớn nhất (BAN-3-DenseNet-BC-80-120, 150M tham số) đạt **14.9%**; DenseNet-112-33 BAN (6.5M tham số theo văn bản) 16.95% xấp xỉ teacher DenseNet-80-120 (16.87%) đắt hơn ~8 lần.

**Student kiến trúc khác, CIFAR-100:**

Bảng 3 (Teacher / BAN-WRN / Dense-90-60 student từ WRN teacher): WRN-28-1: 30.05 / 29.43 / 24.93; WRN-28-2: 25.32 / 24.38 / 18.49; WRN-28-5: 20.88 / 20.93 / 17.52; WRN-28-10: 19.08 / 18.25 / 16.79. Tức là một teacher **yếu hơn** (WRN-28-1, 30.05%) vẫn đủ để DenseNet student đạt 24.93%, tốt hơn nhiều so với WRN-28-1 train thẳng; chỉ riêng WRN-28-5 là BAN-WRN không cải thiện.

Bảng 4 (student DenseNet biến đổi từ teacher DenseNet-90-60, lỗi 17.69): 0.5×Depth 16.95 (21.2M), 2×Depth 16.43 (13.7M), 3×Depth 16.64 (12.9M), 4×Depth 16.64, 0.5×Compression 19.83 (5.1M), 0.75×Compression 17.3 (10.1M), 1.5×Compression 18.89 (80.5M). (Cột tham số trong bảng trích xuất từ PDF hơi lộn xộn; nên kiểm tra lại bản gốc nếu cần số chính xác.)

Bảng 5 (DenseNet teacher sang ResNet student, baseline / BAN): Pre-ResNet-14-0.5: 20.28/18.8; Pre-ResNet-14-1: 18.84/17.39; WRN-1-1: 20.4/19.12; Match-WRN-2-1: 18.83/17.42; WRN-4-0.5: 19.63/17.13; WRN-4-1: 18.77/17.18. Pre-activation ResNet-1001 (10.2M) baseline 22.71. Student ResNet đạt 17.13–17.39% so với teacher DenseNet-90-60 17.69%.

**Penn Tree Bank, perplexity (Bảng 6):**

| Mạng | Tham số | Teacher Val | BAN+L Val | Teacher Test | BAN+L Test |
|---|---|---|---|---|---|
| ConvLSTM | 19M | 83.69 | 80.27 | 80.05 | 76.97 |
| LSTM | 52M | 75.11 | 71.19 | 71.87 | 68.56 |

Ở LSTM, chỉ dùng teacher output thuần không hiệu quả; cần BAN+L (cả nhãn). Giải thích tác giả đưa ra (phỏng đoán): teacher CIFAR đạt 100% train accuracy, còn PTB train perplexity còn xa mức tối ưu.

## 5. Phân tích / ablation

- **Bỏ dark knowledge vẫn có lợi một phần (CWTM):** trên CIFAR-100, CWTM cải thiện nhẹ hơn teacher ở 3/4 mô hình (ví dụ 112-33: 18.25 -> 17.84; 80-80 hòa 17.16; 80-120 lại tệ hơn: 17.12 so với 16.87). Giải thích paper: thông tin từ teacher có thể dùng để cân lại tập train, giảm trọng số mẫu mà phân phối teacher không tập trung ở argmax.
- **Phá cấu trúc lớp sai vẫn có lợi (DKPP):** cải thiện "comparable but systematic". Paper kết luận: dark knowledge không chỉ là thông tin của từng lớp sai cụ thể; các moment bậc cao bất biến với hoán vị cũng đóng góp.
- **Cả hai đều kém BAN đầy đủ** trên hầu hết cấu hình, nên hai thành phần bổ sung nhau (đây là cách đọc của chúng ta từ Bảng 2; paper nói "KD does not simply contribute information on each specific non-correct output").
- **Nhãn thêm có đáng không?** Với ảnh, BAN thuần (chỉ teacher loss) thường tốt hơn BAN+L (Bảng 2), ngược với LSTM.
- **Nhiều thế hệ:** cải thiện "inconsistent but positive", bão hòa sau vài thế hệ; ensemble các thế hệ cho lợi ích lớn hơn.
- **Robust với đổi kiến trúc:** student DenseNet nửa số lớp vẫn thắng teacher (16.95 so với 17.69); giảm compression rate xuống 0.5 là bước làm tệ nhất (19.83). Student ResNet khớp shape thắng teacher DenseNet.

## 6. Giới hạn

**Do tác giả nêu hoặc hàm ý:**
- Cải thiện không đều: CIFAR-10 có vài cấu hình không cải thiện; thế hệ BAN-3 đôi khi tệ hơn BAN-2 (DenseNet-90-60: 16.44 sang 16.72); LSTM cần nhãn.
- Phân rã gradient chỉ giải thích một phần, paper không đưa ra lý thuyết chặt chẽ.

**Do chúng ta nhận thấy:**
- Chỉ thử trên CIFAR và PTB, mô hình nhỏ; chưa có ImageNet hay dữ liệu lớn. Mức cải thiện thường nhỏ (0.1 đến 1 điểm) và chỉ báo cáo một lần chạy, không có độ lệch chuẩn; phần cải thiện vài phần trăm có thể nằm trong nhiễu seed ở CIFAR-10.
- Số "tốt nhất" (15.5%, 14.9%) lấy theo validation/test của các lần chọn thế hệ, có nguy cơ chọn lọc; ensemble 150M tham số thì không còn là "cùng cỡ" nữa.
- Chi phí train gấp $K$ lần với $K$ thế hệ, trong khi bài không so với ensemble các model train độc lập có cùng ngân sách đầy đủ (chỉ so với một số coupled ensemble).
- Không bàn đến việc student quá nhỏ so với teacher, đúng chỗ dự án quan tâm (xem 08, 09).

## 7. Liên hệ với các bài khác

- **02 Hinton KD:** BAN dùng đúng loss đó nhưng bỏ nén và bỏ temperature; phần phân rã gradient bổ sung cách hiểu "soft target = label smoothing/importance weight có điều kiện" đối chọi với giải thích dark knowledge thuần túy.
- **07 Deep mutual learning:** cùng ý "student không cần nhỏ hơn", nhưng DML train các mạng đồng thời, không có teacher cố định.
- **08 Teacher assistant, 09 Efficacy of KD, 41 Capacity gap law:** BAN là cực "không có khoảng cách dung lượng"; các bài này nghiên cứu hậu quả khi khoảng cách lớn.
- **10 Does KD really work:** đặt lại câu hỏi "student có thực sự khớp teacher không?"; BAN chỉ báo cáo lỗi top-1 chứ không đo agreement/độ khớp.
- **22 DINO, 21 DeiT:** self-distillation theo kiểu khác (EMA teacher), cùng tinh thần teacher và student cùng kiến trúc.

## 8. Ứng dụng cho dự án (suy luận)

Phần này là suy luận của chúng ta. Nhìn chung bài **ít liên quan trực tiếp** vì dự án là nén (4B teacher, 0.3B/0.1B student), đúng tình huống mà BAN cố tình tránh. Tuy vậy có ba điểm dùng được:

1. **Cách nhìn gradient:** loss KD = trọng số theo độ tự tin của teacher cho mẫu + tín hiệu lớp sai. Với OCR token-level, "lớp đúng" là token ground-truth; có thể kiểm tra teacher 4B tự tin thấp ở những token nào (chữ mờ, bảng phức tạp) và xem student pruned có đang bị kéo quá mạnh về các token đó không.
2. **Vòng lặp "sau khi prune thì self-distill":** sau khi cắt xong và phục hồi từ teacher 4B, có thể thử thêm một thế hệ "born-again" từ chính model 0.3B tốt nhất (cùng kiến trúc) như bước tinh chỉnh cuối rẻ tiền. Paper chỉ chứng minh trên CIFAR và PTB, nên coi đây là giả thuyết cần kiểm chứng.
3. **Ensemble BANE không dùng được** vì ràng buộc 8 GB và batch-1 latency.

Liên quan tới D_lin / D_res: bài không bàn tới việc cắt neuron; nhưng nếu phần hư hại không tuyến tính (D_res) là phần student khó học, có thể đoán tín hiệu "dark knowledge" càng quan trọng cho phần đó (đây là giả thuyết, không có trong paper).

## 9. Thí nghiệm gợi ý

1. **CWTM trên OCR:** huấn luyện student 0.3B bằng CE với nhãn thật nhưng nhân trọng số $\max p$ của teacher 4B cho từng token; so với KD đầy đủ để biết bao nhiêu phần lợi ích đến từ "reweighting" so với phân phối đầy đủ.
2. **DKPP trên OCR:** xáo xác suất các token không phải argmax trong top-k của teacher (giữ phân phối nhưng phá liên kết ngữ nghĩa); nếu kết quả vẫn gần KD thật, có thể rút gọn lưu trữ logit (top-k) mà không mất nhiều.
3. **Một thế hệ born-again:** từ student 0.3B đã phục hồi, train một bản cùng kiến trúc từ output của nó (+ nhãn), so sánh CER/WER và độ khớp với teacher.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| BAN | Student cùng kiến trúc, seed khác, học từ output teacher; thường thắng teacher |
| BAN+L | BAN dùng thêm loss nhãn thật |
| BANE | Ensemble trung bình dự đoán các thế hệ BAN |
| $\partial L/\partial z_i=q_i-p_i$ | Gradient CE theo logit; nền tảng của phân rã |
| Tách gradient | Số hạng lớp đúng (importance weighting theo $p_*$) + số hạng lớp sai (dark knowledge) |
| CWTM | Chỉ giữ trọng số $\max p$ của teacher, bỏ thông tin lớp sai |
| DKPP | Hoán vị xác suất các lớp không phải argmax để phá cấu trúc tương đồng |
| Dark knowledge | Thông tin trong xác suất của các lớp sai |
| Self-distillation | Teacher và student cùng kiến trúc |

## 11. Kết luận ngắn

Bài cho thấy KD không phải chỉ là nén: student cùng cỡ, được train từ output của teacher, thường vượt teacher trên CIFAR-100 và PTB, và chuỗi nhiều thế hệ kèm ensemble cho thêm lợi ích. Phân tích CWTM/DKPP gợi ý lợi ích của KD đến từ nhiều nguồn (trọng số theo độ tự tin và cấu trúc của phân phối), không chỉ "dark knowledge" kiểu Hinton. Với dự án nén OCR VLM, đây chủ yếu là kiến thức nền và một vài thí nghiệm ablation rẻ.

Nguồn gốc: https://arxiv.org/abs/1805.04770
