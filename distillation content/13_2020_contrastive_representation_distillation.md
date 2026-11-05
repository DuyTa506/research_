# Contrastive Representation Distillation (2020)

**Tác giả:** Yonglong Tian, Dilip Krishnan, Phillip Isola (MIT CSAIL, Google Research).  
**Venue / arXiv:** ICLR 2020; arXiv:1910.10699 (v3, 24/1/2022).  
**Nguồn:** https://arxiv.org/abs/1910.10699  
**Chủ đề:** Distill biểu diễn (representation) bằng contrastive learning: tối đa hóa cận dưới của mutual information giữa embedding teacher và student; nén mô hình, chuyển giao đa phương thức, distill ensemble.  
**Vị trí trong lộ trình:** Đọc sau 02 (Hinton KD), 03 (FitNets), 04 (Attention transfer), 12 (Relational KD). Bài này thay "khớp từng chiều" bằng "khớp thông tin tương hỗ" ở tầng embedding gần cuối. Tiếp theo: 14 (Decoupled KD) quay lại phân tích logit KD.

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch paper. Phần "Ứng dụng cho dự án" là suy luận của chúng ta, không phải kết quả của paper.

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

Hinton KD (bài 02) cực tiểu hóa KL giữa xác suất đầu ra của teacher và student. Paper chỉ ra hai điểm yếu:

1. **Giả định độc lập giữa các chiều.** Mục tiêu KD có dạng phân rã $\psi(y^S,y^T)=\sum_i\phi_i(y^S_i,y^T_i)$ (mỗi chiều tính riêng, tổng lại). Nó không trực tiếp bắt các *phụ thuộc* giữa các chiều. Tác giả so sánh với việc loss L2 khi sinh ảnh cho ảnh mờ do coi các pixel độc lập.
2. **Chỉ áp dụng được khi đầu ra là phân phối.** Khi cần chuyển giao *biểu diễn* (ví dụ từ mạng RGB sang mạng depth, hoặc sang âm thanh), KL không xác định được.

Các phương pháp tầng giữa (FitNet, Attention transfer; bài 03, 04) làm việc trên biểu diễn nhưng cũng dùng loss kiểu hồi quy từng chiều và thường đòi hỏi student/teacher có kiến trúc hoặc độ phân giải feature map tương tự.

Khái niệm cần có: **contrastive learning** = học biểu diễn sao cho các cặp "dương" (cùng nguồn) gần nhau, các cặp "âm" (khác nguồn) xa nhau. **Mutual information (MI)** $I(T;S)$ = lượng thông tin mà biến này chứa về biến kia; $I=0$ khi độc lập. **Critic** = một hàm chấm điểm học được để phân biệt cặp dương với cặp âm.

## 2. Ý tưởng chính

Với cùng một ảnh $x_i$, embedding của teacher và student phải "khớp nhau" (cặp dương); với hai ảnh khác nhau $x_i,x_j$ thì không (cặp âm). Huấn luyện student để một bộ phân loại nhị phân phân biệt được hai trường hợp này chính là tối đa hóa một cận dưới của mutual information giữa biểu diễn của teacher và của student. Vì thế student không chỉ sao chép giá trị mà phải giữ toàn bộ cấu trúc thông tin teacher mã hóa trong embedding, kể cả các tương quan giữa chiều. Phương pháp chạy trên embedding ở tầng penultimate (ngay trước logit), nên hoạt động với số chiều teacher và student khác nhau nhờ hai phép chiếu tuyến tính.

## 3. Phương pháp chi tiết

### 3.1 Loss contrastive

Gọi $T=f^T(x)$ và $S=f^S(x)$ là embedding penultimate. Xét hai phân phối: phân phối đồng thời $p(S,T)$ (cùng một đầu vào cho hai mạng) và tích các biên $p(S)p(T)$ (hai đầu vào độc lập). Đặt biến ẩn $C$: $C=1$ nếu cặp đến từ phân phối đồng thời, $C=0$ nếu từ tích biên; lấy 1 cặp dương cho mỗi $N$ cặp âm. Bayes cho posterior:

$$q(C=1\mid T,S)=\frac{p(T,S)}{p(T,S)+N\,p(T)p(S)}$$

Lấy log rồi kỳ vọng dưới cặp dương, ta có cận dưới MI (công thức 9):

$$\boxed{I(T;S)\ \ge\ \log N+\mathbb E_{q(T,S|C=1)}\big[\log q(C=1\mid T,S)\big]}$$

Vì không biết posterior thật, ta khớp một critic $h:\{\mathcal T,\mathcal S\}\to[0,1]$ bằng cực đại hóa log-likelihood nhị phân:

$$\mathcal L_{critic}(h)=\mathbb E_{q(T,S|C=1)}[\log h(T,S)]+N\,\mathbb E_{q(T,S|C=0)}[\log(1-h(T,S))]\tag{10}$$

Paper chứng minh (Gibbs inequality, Mục 6.2.1) critic tối ưu $h^*=q(C=1|T,S)$, và làm yếu cận dưới để tối ưu *đồng thời* student và critic:

$$\boxed{f^{S*}=\arg\max_{f^S}\max_h\ \mathcal L_{critic}(h)}\tag{17,18}$$

Do $I(T;S)\ge\log N+\mathcal L_{critic}(h)$ với mọi $h$, nên không cần critic hoàn hảo. Thực tế chọn critic:

$$h(T,S)=\frac{e^{g^T(T)'g^S(S)/\tau}}{e^{g^T(T)'g^S(S)/\tau}+\dfrac{N}{M}}\tag{19}$$

với $M$ là kích thước tập dữ liệu, $\tau$ là temperature, $g^T,g^S$ là phép chiếu tuyến tính đưa hai embedding về cùng chiều (paper cho biết memory bank lưu embedding 128-d ở ImageNet) rồi chuẩn hóa L2 trước khi nhân vô hướng. Dạng này lấy cảm hứng từ NCE (noise-contrastive estimation, Gutmann & Hyvärinen 2010; Wu et al. 2018).

**Memory buffer:** theo lý thuyết $N$ càng lớn thì cận dưới càng chặt; để khỏi dùng batch rất lớn, tác giả (theo Wu et al. 2018) giữ một bộ nhớ lưu embedding của mọi mẫu từ các batch trước và lấy mẫu âm từ đó.

### 3.2 Hợp với KD và các biến thể

- Loss KD chuẩn: $L_{KD}=(1-\alpha)H(y,y^S)+\alpha\rho^2H(\sigma(z^T/\rho),\sigma(z^S/\rho))$.
- **CRD+KD** = cộng hai loss (kết quả tốt nhất trong nhiều bảng).
- **Cross-modal** (teacher và student ở hai modality, dữ liệu ghép nhưng không nhãn): bỏ số hạng cross-entropy nhãn, dùng loss contrastive (10).
- **Ensemble:** $\mathcal L_{CRD\text{-}EN}=H(y,y^S)-\beta\sum_i\mathcal L_{critic}(T_i,S)$, tức một loss contrastive cho mỗi teacher.

### Pseudo-code

```python
# Khởi tạo: g_T, g_S = Linear(d_T,128), Linear(d_S,128); memory_T, memory_S: [M,128]
for x, y, idx in loader:                   # idx: chỉ số mẫu trong dataset
    with no_grad(): fT = teacher.penultimate(x)
    fS, logits = student.penultimate_and_logits(x)
    vT = normalize(g_T(fT)); vS = normalize(g_S(fS))
    # N mẫu âm: chọn ngẫu nhiên từ memory (không trùng idx; phiên bản supervised: khác lớp)
    negT = memory_T[sample_neg(idx, N)]  # [B,N,128]
    negS = memory_S[sample_neg(idx, N)]
    # critic dạng (19), cặp dương (vS,vT), cặp âm (vS,negT) và (vT,negS)
    pos = exp((vS*vT).sum(-1)/tau)
    neg = exp(bmm(negT, vS)/tau)
    h_pos = pos / (pos + N/M);  h_neg = neg / (neg + N/M)
    L_crd = -(log(h_pos).mean() + N*log(1-h_neg).mean())   # dấu: cực đại L_critic
    loss = CE(logits, y) + beta*L_crd (+ alpha*KD tùy chọn)
    update_memory(idx, vT, vS)
```

(Pseudo-code do chúng ta tóm tắt từ công thức, bỏ qua chi tiết của code chính thức RepDistiller; code của tác giả: github.com/HobbitLong/RepDistiller.)

### Ví dụ nhỏ

Batch có 3 ảnh A, B, C. Với anchor A, cặp dương là (student(A), teacher(A)), cặp âm là (student(A), teacher(B)) và (student(A), teacher(C)). Nếu cosine của cặp dương là 0.8 và của các cặp âm là 0.1, critic (19) với $\tau=0.1$ cho điểm mũ $e^{8}\approx2981$ so với $e^{1}\approx2.7$: cặp dương áp đảo, loss thấp. Nếu student chưa học gì thì cả ba cosine gần nhau, critic không phân biệt được và gradient đẩy cosine dương lên, cosine âm xuống. (Ví dụ minh họa của chúng ta, bỏ qua hằng số $N/M$ và chi tiết chuẩn hóa trong code gốc.)

## 4. Thí nghiệm và kết quả chính

Dữ liệu: CIFAR-100, ImageNet, STL-10, TinyImageNet, NYU-Depth V2. Paper so sánh với 12 phương pháp distill (KD, FitNet, AT, SP, CC, VID, RKD, PKT, AB, FT, FSP, NST).

### 4.1 Nén mô hình trên CIFAR-100 (trung bình 5 lần chạy)

Cùng phong cách kiến trúc (Table 1), top-1 accuracy (%):

| Teacher → Student | Teacher | Student | KD | CRD | CRD+KD |
|---|---|---|---|---|---|
| WRN-40-2 → WRN-16-2 | 75.61 | 73.26 | 74.92 | 75.48 | 75.64 |
| WRN-40-2 → WRN-40-1 | 75.61 | 71.98 | 73.54 | 74.14 | 74.38 |
| resnet56 → resnet20 | 72.34 | 69.06 | 70.66 | 71.16 | 71.63 |
| resnet110 → resnet20 | 74.31 | 69.06 | 70.67 | 71.46 | 71.56 |
| resnet110 → resnet32 | 74.31 | 71.14 | 73.08 | 73.48 | 73.75 |
| resnet32x4 → resnet8x4 | 79.42 | 72.50 | 73.33 | 75.51 | 75.46 |
| vgg13 → vgg8 | 74.64 | 70.36 | 72.98 | 73.94 | 74.29 |

Kiến trúc khác phong cách (Table 2, trung bình 3 lần):

| Teacher → Student | Student | KD | CRD | CRD+KD |
|---|---|---|---|---|
| vgg13 → MobileNetV2 | 64.6 | 67.37 | 69.73 | 69.94 |
| ResNet50 → MobileNetV2 | 64.6 | 67.35 | 69.11 | 69.54 |
| ResNet50 → vgg8 | 70.36 | 73.81 | 74.30 | 74.58 |
| resnet32x4 → ShuffleNetV1 | 70.5 | 74.07 | 75.11 | 75.12 |
| resnet32x4 → ShuffleNetV2 | 71.82 | 74.45 | 75.65 | 76.05 |
| WRN-40-2 → ShuffleNetV1 | 70.5 | 74.83 | 76.05 | 76.27 |

Quan sát của paper: CRD là phương pháp duy nhất luôn vượt KD trong mọi cặp thử, và đạt "57% average relative improvement" so với KD (công thức ở chú thích trang 2: trung bình của $(Acc_{crd}-Acc_{kd})/(Acc_{kd}-Acc_{van})$, tức so với phần lợi mà KD đã mang lại, không phải 57 điểm phần trăm tuyệt đối). Các phương pháp tầng giữa (FitNet, AT) *thua cả student huấn luyện thường* trong nhiều cặp khác kiến trúc (ví dụ AT vgg13 → MobileNetV2 chỉ 59.40 so với student thường 64.6). Trong Table 1, RKD (bài 12, do các tác giả CRD chạy lại) thấp hơn KD ở cả bảy cặp, ví dụ WRN-40-2 → WRN-16-2: 73.35 so với KD 74.92. Lưu ý: ở cả hai bảng, CRD hầu như không vượt teacher (ví dụ resnet32x4 → resnet8x4: 75.51 so với teacher 79.42); chỉ có WRN-40-2 → WRN-16-2 là sát teacher (75.48/75.64 so với 75.61).

### 4.2 ImageNet (Table 3, error %, ResNet-34 → ResNet-18)

| | Teacher | Student | AT | KD | CRD | CRD+KD |
|---|---|---|---|---|---|---|
| Top-1 err | 26.69 | 30.25 | 29.30 | 29.34 | 28.83 | 28.62 |
| Top-5 err | 8.58 | 10.93 | 10.00 | 10.12 | 9.87 | 9.51 |

Khoảng cách top-1 teacher-student là 3.56; AT thu hẹp 0.95, CRD thu hẹp 1.42 (paper nói tương đương cải thiện tương đối 50%).

### 4.3 Transferability (Table 4)

Student WRN-16-2 (từ teacher WRN-40-2) đóng băng, huấn luyện linear classifier trên STL-10 / TinyImageNet: Student 69.7 / 33.7; KD 70.9 / 33.9; CRD 71.6 / 35.6; CRD+KD 72.2 / 35.5; Teacher 68.6 / 31.5. Tức student distill còn *chuyển giao tốt hơn teacher* (+3.6 STL-10, +4.1 TinyImageNet cho CRD+KD).

### 4.4 Cross-modal

- Luminance → chrominance (Figure 3): CRD hiệu quả hơn các loss khác (số liệu chỉ ở hình, text không liệt kê).
- RGB → depth (Table 5, segmentation ngữ nghĩa): Random init 56.4 pixel acc / 35.8 mIoU; KD 58.9 / 38.0; KD+AT 60.1 / 39.5; FitNet 60.8 / 40.7; CRD 61.6 / 41.8.

### 4.5 Ensemble

CIFAR-100: single WRN-16-2 lỗi 26.7%, ResNet20 lỗi 30.9%. CRD với 8 teacher giảm xuống 23.7% và 28.3%; CRD tốt hơn KD ở mọi số teacher thử (Figure 4).

## 5. Phân tích / ablation

- **InfoNCE so với loss của paper (Table 6):** cùng số mẫu âm, loss của paper (18) thắng InfoNCE ở 4 trên 5 cặp ; với mẫu âm khác lớp ($y_i\ne y_j$), loss của paper đạt 75.48 / 71.46 / 73.48 / 75.51 / 73.94 còn InfoNCE 75.15 / 71.39 / 73.53 / 75.22 / 73.74 (cặp thứ ba InfoNCE nhỉnh hơn). Chênh lệch nhỏ.
- **Chiến lược mẫu âm:** chọn âm khác nhãn ($y_i\ne y_j$) cao hơn chọn âm bất kỳ ($i\ne j$) 0.81% cho loss của paper và 0.62% cho InfoNCE, vì cách đầu tránh đẩy xa các mẫu cùng lớp.
- **Số mẫu âm $N$:** thử 16 đến 16384; càng nhiều càng tốt, nhưng $N=4096$ so với $N=16384$ chênh dưới 0.1%; paper báo cáo với 16384 và nói 4096 là đủ.
- **Temperature $\tau$:** thử 0.02 đến 0.3; quá thấp hoặc quá cao đều xấu; dải 0.05-0.2 ổn; dùng 0.1 (0.07 cho ImageNet).
- **Chi phí:** trên ResNet-18/ImageNet thêm 260 MFLOPs (khoảng 12% của 2 GFLOPs); tốc độ 1.75 so với 1.67 epoch/giờ trên hai Titan-V; memory bank 128-d cho ImageNet khoảng 600MB.
- **Lý do tầng giữa tệ khi khác kiến trúc (giả thuyết của tác giả):** mỗi kiến trúc có "đường giải" riêng từ đầu vào đến đầu ra, nên ép bắt chước feature trung gian xung đột với inductive bias.
- **Tương quan giữa class logits (Figure 2):** CRD làm khoảng cách giữa ma trận tương quan của teacher và student nhỏ nhất so với AT và KD (chỉ định tính qua hình).

## 6. Giới hạn

**Do tác giả nêu:** nhiệt độ tối ưu thay đổi theo dataset và cần chỉnh lại; cận dưới MI chỉ chặt khi $N$ lớn (nên cần memory buffer); với cross-modal trên depth phải dùng thêm thủ thuật contrast local/global (Hjelm et al. 2018) vì dữ liệu ít.

**Do chúng ta nhận xét:**
- Toàn bộ là phân loại ảnh cỡ nhỏ; không có chuỗi, không có mô hình sinh. Embedding "penultimate" không rõ nghĩa ở decoder token-level.
- Cần memory bank chứa embedding **mọi mẫu** (kích thước $M$). Với dữ liệu OCR lớn (hàng triệu trang, mỗi trang nhiều token) thì phải đổi thiết kế (mỗi mẫu = một trang hay một token?).
- Cặp dương/âm được định nghĩa ở mức *mẫu* với nhãn lớp; với chuỗi sinh, "mẫu âm khác lớp" không có ngữ nghĩa tương đương.
- Lợi thế so với KD tuyệt đối thường chỉ 0.5-2 điểm; riêng ở cặp khác kiến trúc lên đến khoảng 2.4 điểm (vgg13 → MobileNetV2: 69.73 so với 67.37).
- Kết quả của RKD trong bảng là bản chạy lại của các tác giả CRD; kết luận "RKD kém KD" ở đây có thể khác với bài 12 vì setup khác.

## 7. Liên hệ với các bài khác

- **02 Hinton KD:** CRD thừa nhận KD là baseline mạnh thứ hai; "KD hoạt động khá tốt, không phương pháp nào khác luôn thắng KD khi đứng một mình".
- **03 FitNets / 04 Attention transfer:** thất bại khi khác kiến trúc (Table 2) nhắc lại cảnh báo về khớp feature trung gian.
- **10 Does KD really work:** câu hỏi liệu student có thật sự khớp teacher; Figure 2 của CRD (tương quan logit) là một cách đo liên quan (suy luận về mối nối).
- **12 Relational KD:** cũng truyền thông tin giữa các mẫu; RKD khớp hình học (khoảng cách, góc) trong batch, CRD tối đa hóa MI với nhiều mẫu âm từ memory bank.
- **14 Decoupled KD:** sẽ phân rã KD thành hai phần (TCKD và NCKD) để giải thích vì sao logit KD vẫn đủ mạnh; đọc tiếp để so sánh hướng "sửa loss tầng logit" với "đổi sang biểu diễn".
- **22 DINO và 24 CLIP-KD:** các bài tự giám sát và CLIP dùng contrastive/self-distillation; CRD là điểm nối giữa distillation và representation learning (suy luận).

## 8. Ứng dụng cho dự án (suy luận)

*Mức liên quan: trung bình thấp.* Dự án distill một VLM sinh chuỗi OCR, trong khi CRD dùng embedding cấp ảnh và nhãn lớp.

1. **Vision encoder (ViT 12 block):** có thể coi mỗi ảnh/patch-pooled embedding là "mẫu" và đặt loss contrastive giữa encoder gốc và encoder đã prune, với mẫu âm là các trang khác trong batch hoặc trong memory bank. Lợi ích có thể có: không cần giống hệt hidden size nhờ hai phép chiếu $g^T,g^S$, và student có thể giữ cấu trúc quan hệ giữa các trang.
2. **Không thay thế loss đầu ra.** Với đầu ra OCR dài gần như tất định, tín hiệu chính vẫn là KL trên token (xem 15). CRD nhiều khả năng chỉ là loss phụ (như CRD+KD trong paper).
3. **Với $D_{lin}$ / $D_{res}$:** CRD không phụ thuộc phép biến đổi tuyến tính cố định (đã có $g$ học được), nên phần $D_{lin}$ được hấp thụ sẵn; phần $D_{res}$ thì chỉ giảm nếu thông tin dư thừa còn nằm trong capacity của student (suy luận chưa kiểm chứng).
4. Cảnh báo: contrastive đòi mẫu âm chất lượng; trang OCR của cùng một tài liệu có thể rất giống nhau (cùng template) nên "mẫu âm" có thể là âm giả (false negative).

## 9. Thí nghiệm gợi ý

1. **CRD cho ViT sau khi prune:** thêm $L_{CRD}$ trên embedding pooled của encoder (với $N$ = 4096, $\tau$ = 0.1 như paper) cạnh loss logit KL; so với MSE + projection. Đo CER/WER và độ tương quan feature trên tập dev.
2. **Kiểm tra false negatives:** lấy mẫu âm cùng tài liệu và khác tài liệu, xem CRD có xấu đi khi có nhiều trang template giống nhau không.
3. **Thử $N$ nhỏ (256) và batch bình thường (không memory bank)** để xem ảnh hưởng, vì dự án bị giới hạn GPU 8 GB; paper nói $N$ 4096 là đủ nhưng chưa thử bộ nhớ nhỏ hơn.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Mutual information $I(T;S)$ | Thông tin chung giữa embedding teacher và student |
| Cận dưới MI | $I(T;S)\ge\log N+\mathbb E[\log q(C=1|T,S)]$ |
| Critic $h$ | Hàm phân biệt cặp dương/âm, dạng (19) |
| $\mathcal L_{critic}$ | $\mathbb E_{pos}[\log h]+N\,\mathbb E_{neg}[\log(1-h)]$ |
| Mục tiêu CRD | $\arg\max_{f^S}\max_h\mathcal L_{critic}(h)$ |
| Memory buffer | Lưu embedding mọi mẫu để lấy rất nhiều mẫu âm |
| $\tau$ | Temperature của critic; 0.1 (0.07 cho ImageNet) |
| CRD+KD | Cộng loss contrastive và loss KD |
| InfoNCE | Contrastive loss dạng softmax chọn 1 dương trong tập nhiễu |
| Linear probing | Đóng băng mạng, huấn luyện lớp tuyến tính để đo chất lượng biểu diễn |

## 11. Kết luận ngắn

CRD đặt distillation thành bài toán contrastive: student phải phân biệt được đâu là biểu diễn teacher của *cùng* mẫu so với mẫu khác, để tối đa hóa cận dưới MI. Trên CIFAR-100 và ImageNet nó là loss tầng biểu diễn duy nhất luôn vượt KD, và student còn chuyển giao sang tác vụ khác tốt hơn teacher. Với dự án của bạn, giá trị chính là một loss phụ cho vision encoder khi hidden size khác nhau, không phải loss chính cho đầu ra chuỗi.

Link gốc: https://arxiv.org/abs/1910.10699
