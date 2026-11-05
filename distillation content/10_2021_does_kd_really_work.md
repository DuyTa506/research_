# Does Knowledge Distillation Really Work? (2021)

**Tác giả:** Samuel Stanton, Pavel Izmailov, Polina Kirichenko, Alexander A. Alemi, Andrew Gordon Wilson.  
**Venue / arXiv:** arXiv:2106.05945 (v2, 06/12/2021); NeurIPS 2021 (35th Conference on Neural Information Processing Systems, Sydney).  
**Nguồn:** https://arxiv.org/abs/2106.05945  
**Chủ đề:** Tách "fidelity" (student khớp teacher) khỏi "generalization" (student đúng trên dữ liệu mới); KD thường có fidelity thấp; hai giả thuyết nguyên nhân: identifiability (dữ liệu distillation) và optimization (tối ưu hóa).  
**Vị trí trong lộ trình:** Đọc trước: 02 (Hinton KD), 06 (BAN: student vượt teacher), 08–09 (capacity gap). Bài này thêm: ngay cả khi student **đủ dung lượng** (self-distillation, cùng kiến trúc), student vẫn khớp teacher rất kém, nguyên nhân chủ yếu là **tối ưu hóa**, không phải dung lượng hay dữ liệu. Đọc tiếp: 11 (Patient and consistent teacher: công thức thực tế để đạt fidelity cao nhờ huấn luyện rất lâu), rồi các bài về KD cho LLM.

> Đây là ghi chú học tập bằng tiếng Việt, diễn giải và tóm lược, không phải bản dịch paper. Số liệu lấy từ paper; nhiều kết quả chỉ có dưới dạng đồ thị (và phụ lục không có trong bản text đã đọc) nên chỉ ghi những số có trong văn bản. Phần "Ứng dụng cho dự án" và các nhận xét ghi là suy luận là của chúng ta.

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

Câu chuyện quen thuộc về KD: student học một biểu diễn "trung thực" của teacher (lớn, hoặc ensemble) nhờ soft label. Các bài trước (BAN, TA, Efficacy) đã nghi ngờ phần "tốt hơn teacher" hoặc "teacher lớn không dạy tốt", nhưng đa số đo bằng **accuracy** của student. Bài này nói: accuracy trộn lẫn hai thứ khác nhau.

- **Fidelity:** student dự đoán giống teacher đến mức nào (khớp phân phối dự đoán).
- **Generalization:** student dự đoán đúng trên dữ liệu mới.

Hai thứ này có thể ngược chiều: self-distillation (student cùng kiến trúc) làm student tốt hơn teacher **vì không khớp** (regularization nhờ thất bại) chứ không phải vì khớp tốt. Nếu mục tiêu là nén một ensemble hay mô hình đáng tin để triển khai trên thiết bị nhỏ, thì fidelity mới là thứ cần.

Khái niệm cần có:
- **Ensemble teacher:** trung bình xác suất của $m$ mạng; logit teacher $z_t=\log\big(\sum_i\sigma(z_i)/m\big)$.
- **Calibration / ECE:** mức độ khớp giữa độ tự tin dự đoán và độ đúng thực tế; **NLL** là negative log-likelihood.
- **Identifiability** (khả năng nhận dạng): dữ liệu distillation có đủ để phân biệt student "đúng" với student "sai" không.
- **Basin (lòng chảo) của loss:** vùng quanh một cực tiểu cục bộ.

## 2. Ý tưởng chính

Tác giả đo **agreement** top-1 và **KL** giữa teacher và student trên tập test. Họ thấy fidelity thấp đến khó tin (ResNet-56 trên CIFAR-100: chỉ đạt khoảng 80 đến 86%, xa mức 99% của ví dụ đơn giản LeNet-5/MNIST), kể cả khi student đủ dung lượng, kiến trúc khác (VGG), tập dữ liệu lớn hơn (ImageNet) hoặc miền khác (văn bản IMDB). Hai giả thuyết: (1) dữ liệu distillation chưa đủ/đúng (identifiability), (2) student không giải nổi bài toán tối ưu. Họ bác (1) tương đối và ủng hộ (2): student không khớp teacher ngay cả **trên tập distillation (train)**, tăng thêm augmentation hay dữ liệu GAN còn làm bài toán tối ưu khó hơn. Thí nghiệm khởi tạo student gần teacher cho thấy bề mặt loss có nhiều lòng chảo và student bị kẹt ở lòng chảo xấu.

## 3. Phương pháp chi tiết

### 3.1 Loss KD

$$L_s=\alpha L_{NLL}+(1-\alpha)L_{KD},\qquad L_{KD}(z_s,z_t)=-\tau^2\sum_{j=1}^{c}\sigma_j\!\Big(\tfrac{z_t}{\tau}\Big)\log\sigma_j\!\Big(\tfrac{z_s}{\tau}\Big)$$

Để tách fidelity khỏi tác động của nhãn thật, **trong thí nghiệm chính họ đặt $\alpha=0$** (chỉ loss KD; ablation về $\alpha$ nằm ở phụ lục). Khi $\tau=1$, $L_{KD}$ tương đương KL(teacher $\|$ student) (sai khác hằng số). Khi $\tau\to\infty$, gradient theo logit student xấp xỉ $z_t-z_s$, tức tương đương hồi quy $\ell_2$ trên logit, coi mọi lớp quan trọng như nhau. Ý của tác giả: $\tau$ quyết định "mềm" đến đâu và phân bổ dung lượng student: student rất nhỏ có thể dồn sức khớp top-$k$ thay vì khớp toàn bộ phân phối.

### 3.2 Đo fidelity

$$\text{Agreement}=\frac1n\sum_i \mathbb 1\{\arg\max_j\sigma_j(z_{t,i})=\arg\max_j\sigma_j(z_{s,i})\},\qquad \text{Predictive KL}=\frac1n\sum_i \mathrm{KL}\big(\hat p_t(y|x_i)\,\|\,\hat p_s(y|x_i)\big)$$

Agreement chỉ xét top-1; KL nhạy với toàn bộ phân phối. Tác giả cũng báo accuracy, NLL, ECE.

**Cẩn thận với đo agreement:** ba mô hình độc lập có accuracy 55%, 75%, 95%: $f_1$ và $f_3$ chỉ có thể trùng tối đa 60%, trong khi $f_2$ và $f_3$ trùng ít nhất 70%, nhưng điều đó không có nghĩa $f_2$ là bản distill tốt hơn của $f_3$. Cách kiểm soát: cho thêm một student $s'$ chưng cất bằng đúng quy trình đó từ một **teacher độc lập** $t'$; so fidelity $(t,s)$ với $(t,s')$. Nếu hai con số xấp xỉ nhau thì sự trùng khớp chỉ do cùng tổng quát hóa tốt chứ không phải do mimic.

### 3.3 Sơ đồ chẩn đoán trong paper

```
Fidelity thấp -> loại các nguyên nhân dễ:
   dung lượng student (self-distillation vẫn thấp; tăng student ít tác dụng)
   kiến trúc (VGG cũng vậy), quy mô dữ liệu (ImageNet cũng vậy), miền (IMDB cũng vậy)
-> hai giả thuyết còn lại:
   (A) Identifiability: tăng/đa dạng hóa dữ liệu distillation (augmentation, GAN, dữ liệu mới D1) có nâng fidelity?
   (B) Optimization: student có khớp teacher ngay trên dữ liệu distillation (train agreement)?
        -> kéo dài huấn luyện, đổi optimizer, khởi tạo gần teacher (θ_s = λθ_t + (1-λ)θ_r)
```

## 4. Thí nghiệm và kết quả chính

**Thiết lập chung:** CIFAR-100 với ResNet-56, ensemble $m\in\{1,3,5\}$ (đến 12 ở phụ lục), MNIST/EMNIST với LeNet-5, nhiều kết quả trung bình 3 lần chạy. Mã: github.com/samuelstanton/gnosis.

**Khi nào KD thành công (MNIST):** teacher LeNet-5 train trên tập con 200 mẫu của MNIST (test accuracy 84% đến 86%), distill trên toàn bộ MNIST (60.000), và 25%, 50%, 100% của EMNIST (697.932 ảnh): student đạt trên 99% agreement trên test. Ở đây self-distillation không cải thiện generalization.

**CIFAR-100 self-distillation (Hình 1a):** teacher và student đều một ResNet-56; thêm ảnh GAN (SN-GAN) 12.5k, 25k, 37.5k, 50k vào dữ liệu distillation: fidelity trên test tăng nhưng chỉ lên dưới 80% (theo văn bản), còn test accuracy của student **giảm** theo hướng tiệm cận teacher. Nghĩa là student vượt teacher khi dữ liệu nhỏ, nhờ không khớp.

**Distill ensemble (Hình 1b):** teacher là ensemble 3 ResNet-56: fidelity tăng thì accuracy cũng nhích lên, tức fidelity và generalization đồng hướng, nhưng vẫn còn khoảng cách fidelity lớn. Khoảng cách accuracy giữa teacher và student tăng theo số thành phần ensemble.

**Augmentation (Hình 3, ensemble 5 ResNet-56):** các chính sách: baseline (crop+flip), baseline với $\tau=4$, Rotation, Vertical Flip, ColorJitter, CombAug, MixUp, MixUp ($\tau=4$), GAN ($\tau=4$), OOD, Noise.
- MixUp và GAN cho generalization tốt nhất; MixUp ($\tau=4$) cho fidelity tốt nhất nhưng chỉ đạt **86%** test agreement.
- Baseline ($\tau=4$) đã là baseline mạnh: **84.5%** agreement.
- Augmentation dạng nhiễu (Noise) và ngoài phân phối (OOD) làm tệ trên mọi chỉ số.

**Tái sử dụng dữ liệu (Hình 4, CIFAR-100 chia đôi $D_0,D_1$):** teacher train trên $D_0$. Student distill trên $D_0$ (tái dùng) có accuracy test cao hơn nhưng fidelity thấp hơn và ECE tệ hơn so với student distill trên $D_1$ (dữ liệu chưa từng thấy); distill trên $D_0\cup D_1$ cho kết quả tốt nhất. Agreement tốt nhất khoảng 85%, không hơn kết quả từ augmentation.

**Train agreement (Hình 5):** thêm dữ liệu ngoài tập train của teacher làm student **khó khớp hơn** trên chính tập distillation. Với 50k ảnh GAN, ngay cả student cùng dung lượng chỉ đạt 95% train agreement. Với augmentation nặng: Rotation, Vertical Flip, ColorJitter cho train agreement 80% đến 90% với mọi cỡ teacher; Combined Augs (kết hợp ba loại) chỉ còn khoảng **60%** ở self-distillation. Nhưng Combined Augs lại có test agreement tốt hơn các chính sách có train agreement cao hơn: đánh đổi giữa "nhiều nhãn teacher" (dễ nhận dạng) và "dữ liệu khó khớp" (khó tối ưu).

**Tối ưu hóa (ResNet-20 trên CIFAR-100, self-distillation, Hình 6):** để đơn giản hóa, họ thay BatchNorm bằng LayerNorm, vì với BatchNorm, hai mạng có trọng số giống nhau vẫn có thể dự đoán khác nhau do thống kê activation tích lũy khác nhau. Kết quả:
- Huấn luyện lâu hơn giúp chút ít: SGD 300 epoch cho 78.95% train agreement, 5k epoch cho 83.3%; từ 1k lên 5k chỉ thêm khoảng 2%. Adam không tốt hơn SGD.
- Khởi tạo $\theta_s=\lambda\theta_t+(1-\lambda)\theta_r$ ($\theta_t$ trọng số teacher, $\theta_r$ khởi tạo ngẫu nhiên): với $\lambda\le0.25$ student hội tụ tới giá trị loss kém, agreement thấp; tại $\lambda=0.375$ có bước nhảy đột ngột: loss về mức tối ưu, agreement tăng vọt gần 100%. Hình chiếu 2D của bề mặt loss cho thấy $\lambda\in\{0,0.25\}$ rơi vào một lòng chảo khác lòng chảo của teacher.
- Chia sẻ khởi tạo ban đầu với teacher (Bảng 1, 10 lần, teacher accuracy trung bình 70.522 (0.412)): Agreement: ngẫu nhiên 77.174 (0.352) so với giống teacher 77.098 (0.238); KL: 0.836 (0.016) so với 0.838 (0.020); CKA (độ tương đồng activation) tăng nhẹ ở 3 stage (0.939/0.925/0.885 so với 0.951/0.937/0.890). Chức năng không khác biệt.

**Loại trừ nguyên nhân khác (nêu ở Mục 4.4, chi tiết ở phụ lục không có trong bản text đã đọc):** tăng dung lượng student ít tác dụng; VGG cho kết quả tương tự; ImageNet và IMDB (phân loại cảm xúc) cho fidelity thấp tương tự.

## 5. Phân tích / ablation

- **Accuracy tốt không có nghĩa fidelity tốt:** ngay cả ngoài self-distillation, mô hình tốt nhất về generalization không phải lúc nào cũng là tốt nhất về fidelity.
- **Fidelity tương quan với calibration khi distill ensemble:** student có fidelity cao nhất luôn là student calibrate tốt nhất (dù chưa chắc chính xác nhất).
- **Trade-off dữ liệu và tối ưu:** mở rộng dữ liệu distillation ngoài tập train của teacher giúp xác định nghiệm đúng (identifiability), nhưng làm bài tối ưu vốn đã khó trở nên khó hơn.
- **Temperature là đòn bẩy rẻ nhất:** $\tau=4$ trên baseline augmentation đã cho 84.5% agreement, gần bằng cách phức tạp hơn; tác giả lập luận hiệu quả nhỏ của việc mở rộng support so với tempering cho thấy thiếu nhãn teacher không phải nút thắt chính.
- **Nguyên nhân gốc:** tối ưu hóa. Student không khớp teacher trên tập train, không phải vì thiếu dung lượng (self-distillation có nghiệm biết trước: chính trọng số teacher) mà vì SGD/Adam hội tụ tới lòng chảo xấu. Ngay cả khởi tạo giống teacher cũng không đủ; phải "gần" mới thoát.
- **Tương phản với học thường:** trong huấn luyện thông thường ta không cần giải bài tối ưu thật tốt (nghiệm phẳng, thiên kiến ngầm của SGD giúp tổng quát hóa), còn ở KD fidelity gắn trực tiếp với việc giải bài tối ưu đó.

## 6. Giới hạn

**Do tác giả nêu hoặc hàm ý:**
- "Hợp lý nhưng không chắc" rằng huấn luyện cực lâu (hàng chục nghìn epoch) có thể đạt 100% train agreement; họ không thử.
- Họ chỉ xét classification ảnh/văn bản vừa phải; phần lớn kết luận tổng quát (ImageNet, IMDB, VGG) nằm ở phụ lục.

**Do chúng ta nhận thấy:**
- Phần phân tích tối ưu hóa dùng ResNet-20 + LayerNorm (bỏ BatchNorm) và $\alpha=0$ để đơn giản hóa; hệ quả sang kiến trúc transformer lớn, dữ liệu rất lớn và loss có nhãn ($\alpha>0$) chưa rõ.
- Không đề xuất phương pháp mới để cải thiện fidelity; bài chủ yếu là chẩn đoán. Cách chữa thực tế xuất hiện ở bài 11 (đồng thời công bố).
- Thí nghiệm khởi tạo $\lambda\theta_t$ yêu cầu biết trọng số teacher cùng kiến trúc, nên chỉ chẩn đoán được trong self-distillation, không áp dụng khi student là mạng nhỏ hơn.
- Fidelity được đo bằng top-1 agreement và KL trung bình; không đo fidelity ở các đuôi (ví dụ lỗi hiếm), mà trong OCR có thể quan trọng.
- Một số số trong bản text bị mất do hình vẽ, nên nhiều kết luận (như xu hướng theo kích thước ensemble) chỉ xác nhận qua mô tả văn bản.

## 7. Liên hệ với các bài khác

- **06 BAN:** bài này giải thích lại "student vượt teacher" như một hệ quả của việc **không** khớp (regularization), không phải của việc khớp tốt.
- **08 Teacher assistant, 09 Efficacy of KD:** ở đó "capacity mismatch" là nguyên nhân; bài này chỉ ra rằng ngay cả khi không mismatch, khớp vẫn kém. Hai kết luận không loại trừ nhau.
- **07 DML:** nhận định về entropy/calibration có liên hệ (fidelity cao tương quan với calibration tốt).
- **11 Patient and consistent teacher:** bài song sinh (công bố gần như cùng lúc; được trích trong Related Work của bài này). 11 đưa ra công thức thực tế để thu hẹp khoảng cách fidelity: cùng ảnh đầu vào (kể cả augmentation) cho teacher và student, mixup (tạo "function matching"), và train rất lâu (bài này gọi tương ứng: "careful treatment of data augmentation, MixUp, extended training duration").
- **13 CRD, 14 Decoupled KD, 12 Relational KD:** các cải tiến loss; bài 10 gợi ý rằng điểm nghẽn thật nằm ở tối ưu hóa chứ không (chỉ) ở dạng loss.
- **Bài LLM (28 MiniLLM, 29 GKD, 31 Rethinking KL):** quan hệ với khái niệm "khớp phân phối" và lựa chọn hướng KL; bài 10 đóng khung tầm quan trọng của việc đo khớp trực tiếp.

## 8. Ứng dụng cho dự án (suy luận)

Phần này là suy luận của chúng ta. **Liên quan cao về mặt phương pháp luận.**

1. **Đo fidelity chứ không chỉ CER/WER:** với student 0.3B/0.1B, theo dõi token-level agreement và KL với teacher 4B trên (a) dữ liệu train distillation và (b) tập held-out. Nếu train agreement thấp, cải thiện đến từ tối ưu hóa chứ không phải dữ liệu.
2. **Baseline kiểm soát (student "độc lập"):** huấn luyện một student cùng quy trình từ một teacher khác (ví dụ model 0.8B gốc hay một checkpoint 4B khác) và so agreement với teacher 4B, để chắc chắn agreement đến từ mimic, không từ cùng đúng (đặc biệt vì OCR có đáp án chuẩn nên hai model đều giỏi sẽ tự nhiên khớp nhiều).
3. **Gợi ý cho nhận xét "cắt MLP neuron phục hồi kém hơn cắt layer":** bài cho thấy khởi tạo gần nghiệm (gần teacher) thoát lòng chảo xấu. Cắt layer giữ nguyên trọng số các layer còn lại (student bắt đầu "gần" teacher trong không gian tham số/hàm), còn cắt neuron MLP phá cấu trúc nội tại; có thể đoán nên giải thích bằng "khởi tạo xa lòng chảo của teacher" (giả thuyết, không phải kết luận của bài; bài chỉ làm thí nghiệm self-distillation). Dùng phân rã $D_{lin}$ và $D_{res}$: sau khi hiệu chỉnh tuyến tính tối ưu tại điểm cắt (loại $D_{lin}$), student bắt đầu "gần" hơn, có thể dùng như một dạng khởi tạo $\lambda$ lớn.
4. **Temperature là công cụ rẻ:** thử $\tau\in\{1,2,4\}$ trước khi nghĩ đến thủ thuật phức tạp.
5. **Dữ liệu distillation:** thêm dữ liệu ngoài phân phối (ví dụ tài liệu tổng hợp) có thể tăng test agreement nhưng làm khó tối ưu; cân nhắc lịch trộn.
6. **Cảnh báo BatchNorm/thống kê:** hai mạng cùng trọng số có thể khác dự đoán nếu thống kê chuẩn hóa khác; với LayerNorm/RMSNorm trong LLM/ViT không gặp vấn đề này.

## 9. Thí nghiệm gợi ý

1. **Train vs held-out agreement:** vẽ token agreement/KL của student pruned với teacher 4B trên tập train distillation và tập giữ lại theo số bước huấn luyện, cho cả cắt layer và cắt neuron; xem train agreement có bão hòa thấp (dấu hiệu tối ưu) hay không.
2. **"Khởi tạo $\lambda$" tương tự:** sau khi cắt, thử student khởi tạo bằng (a) cắt thô, (b) cắt thô + hiệu chỉnh tuyến tính tối ưu tại điểm cắt, (c) nội suy trọng số giữa hai phiên bản; so train agreement cuối cùng.
3. **Student độc lập làm baseline:** với cùng quy trình KD, distill từ hai teacher khác nhau (4B và 0.8B gốc) và đo agreement chéo để tách "khớp do mimic" khỏi "khớp do cùng đúng".

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Fidelity | Student khớp dự đoán của teacher |
| Generalization | Student đúng trên dữ liệu chưa thấy |
| Agreement | Tỉ lệ mẫu mà argmax của teacher và student trùng nhau |
| Predictive KL | KL trung bình từ phân phối teacher sang student |
| Student độc lập $s'$ | Student chưng cất từ teacher độc lập, dùng làm đối chứng |
| Identifiability | Dữ liệu distillation có đủ để xác định nghiệm đúng không |
| Optimization | Student không khớp ngay cả trên dữ liệu distillation |
| $\theta_s=\lambda\theta_t+(1-\lambda)\theta_r$ | Khởi tạo nội suy; ngưỡng $\lambda=0.375$ cho bước nhảy fidelity |
| Self-distillation | Teacher và student cùng kiến trúc |

## 11. Kết luận ngắn

KD thường cải thiện generalization nhưng hiếm khi chuyển được "tri thức" theo nghĩa khớp hàm: fidelity thấp xuất hiện ở mọi thiết lập tác giả thử, kể cả khi student đủ dung lượng. Nguyên nhân chính là tối ưu hóa: student không khớp nổi teacher ngay cả trên dữ liệu distillation và chỉ khớp tốt khi khởi tạo đã nằm đúng lòng chảo. Mở rộng dữ liệu giúp nhận dạng nhưng làm tối ưu khó hơn. Với dự án, bài đề xuất đo fidelity trực tiếp (train và held-out), có baseline student độc lập, và để ý khởi tạo của student sau khi prune.

Nguồn gốc: https://arxiv.org/abs/2106.05945
