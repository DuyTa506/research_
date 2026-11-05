# Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer (2017)

**Tác giả:** Sergey Zagoruyko, Nikos Komodakis  
**Venue / arXiv:** arXiv:1612.03928 (v3, 12/2/2017); ICLR 2017 (ghi trên file)  
**Nguồn:** https://arxiv.org/abs/1612.03928  
**Chủ đề:** Feature-based distillation với "attention map" (activation-based và gradient-based), chuẩn hoá L2, kết hợp với KD, kết quả trên CIFAR-10, Scenes, CUB, ImageNet.  
**Vị trí trong lộ trình:** Đọc sau 03 (FitNets). FitNets khớp feature thô qua regressor; bài này nói: khớp một bản tóm tắt không gian của feature (attention map) tốt hơn và không cần tham số mới. Tiếp theo: 05 (survey, đặt AT vào bản đồ), 12 (Relational KD), 13 (CRD).

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

Trước bài này, các cách chuyển giao tri thức (Buciluǎ 2006, Ba & Caruana, Hinton, FitNets) chủ yếu dựa trên output hoặc feature thô. FitNets chọn student mảnh và sâu. Tác giả lập luận ngược lại: theo "Wide Residual Networks" (Zagoruyko & Komodakis 2016), sau một độ sâu nhất định lợi ích đến chủ yếu từ dung lượng (số tham số), và mạng mảnh-sâu khó song song hoá hơn mạng rộng. Vì vậy ông chọn hướng **student nông hơn** teacher.

Câu hỏi của paper: teacher có thể giúp student bằng cách cho biết "nó nhìn vào đâu" không? Cần định nghĩa chính xác "attention" cho CNN.

**Thuật ngữ:**
- **Attention map (ở đây):** bản đồ không gian 2D $H\times W$ tóm tắt tensor kích hoạt $C\times H\times W$, cho biết vùng nào của ảnh mạng kích hoạt mạnh. Không phải self-attention của Transformer.
- **Activation-based vs gradient-based:** attention định nghĩa theo giá trị kích hoạt, hoặc theo gradient của loss theo đầu vào.
- **Double backpropagation:** lan truyền ngược lần hai qua gradient (đạo hàm bậc hai hỗn hợp).

## 2. Ý tưởng chính

Với mỗi lớp của CNN, nén tensor $C\times H\times W$ thành bản đồ $H\times W$ bằng cách gộp theo kênh (ví dụ tổng bình phương của giá trị tuyệt đối), chuẩn hoá L2, rồi bắt student có bản đồ giống teacher ở nhiều điểm trong mạng (cuối mỗi nhóm residual block). Loss này cộng với cross-entropy (và có thể cả KD). Không thêm tham số, ít tốn thêm tính toán. Paper cho thấy cách này cải thiện student trên nhiều kiến trúc và tốt hơn việc khớp toàn bộ tensor kích hoạt kiểu FitNets.

## 3. Phương pháp chi tiết

### 3.1 Activation-based attention

Với $A\in\mathbb R^{C\times H\times W}$, hàm ánh xạ $F:\mathbb R^{C\times H\times W}\to\mathbb R^{H\times W}$. Giả định ngầm: **giá trị tuyệt đối của kích hoạt neuron cho biết mức quan trọng của neuron đó với đầu vào**. Ba lựa chọn paper xét ($A_i=A(i,:,:)$):

- $F_{sum}(A)=\sum_{i=1}^{C}|A_i|$
- $F^{p}_{sum}(A)=\sum_{i=1}^{C}|A_i|^p$, $p>1$
- $F^{p}_{max}(A)=\max_{i=1,\dots,C}|A_i|^p$, $p>1$

Tính chất theo paper: $p$ lớn thì tập trung hơn vào vị trí kích hoạt cao nhất (vùng phân biệt nhất); $F_{max}$ chỉ xét một kênh trên mỗi vị trí, còn $F_{sum}$ ưu tiên vị trí có nhiều neuron kích hoạt mạnh. Quan sát trực quan: mạng chính xác hơn có attention tương quan không gian cao hơn với đối tượng; lớp thấp kích hoạt ở điểm có gradient thấp-cấp, lớp giữa ở vùng phân biệt (mắt, bánh xe), lớp cao ở toàn bộ đối tượng.

### 3.2 Hàm mất mát attention transfer

Gọi $I$ là tập chỉ số cặp lớp (student, teacher) cùng độ phân giải không gian; $Q_S^j=\mathrm{vec}(F(A_S^j))$, $Q_T^j=\mathrm{vec}(F(A_T^j))$. Tổng loss:

$$
\boxed{\mathcal L_{AT}=\mathcal L(W_S,x)+\frac{\beta}{2}\sum_{j\in I}\left\|\frac{Q_S^j}{\|Q_S^j\|_2}-\frac{Q_T^j}{\|Q_T^j\|_2}\right\|_p}
$$

với $\mathcal L$ là cross-entropy, $p$ là loại chuẩn (thí nghiệm dùng $p=2$). **Chuẩn hoá L2 từng bản đồ được paper nhấn mạnh là quan trọng cho thành công**: nó loại bỏ khác biệt về thang giá trị giữa teacher và student, chỉ giữ *hình dạng* (vùng nào nổi bật). Có thể dùng chuẩn hoá L1 thay thế.

Vị trí đặt loss với ResNet: nếu cùng độ sâu, đặt sau mỗi residual block; nếu khác độ sâu, đặt ở đầu ra của mỗi *nhóm* residual block (Figure 5). Nếu độ phân giải không khớp, có thể nội suy.

Kết hợp KD: chỉ cần cộng thêm số hạng cross-entropy giữa phân phối nới nhiệt độ của teacher và student; AT thêm rất ít chi phí vì teacher đã chạy forward cho KD.

### 3.3 Gradient-based attention

Định nghĩa attention là gradient của loss theo đầu vào (độ nhạy đầu ra với thay đổi tại mỗi vị trí đầu vào):

$$
J_S=\frac{\partial}{\partial x}\mathcal L(W_S,x),\quad J_T=\frac{\partial}{\partial x}\mathcal L(W_T,x),\quad
\mathcal L_{AT}=\mathcal L(W_S,x)+\frac{\beta}{2}\|J_S-J_T\|_2
$$

Đạo hàm theo $W_S$ chứa số hạng đạo hàm bậc hai hỗn hợp $\partial^2\mathcal L/\partial W_S\partial x$ nên cần lan truyền ngược lần hai (như double backpropagation của Drucker & LeCun 1992; chi phí lần hai xấp xỉ lần đầu, bỏ forward).

Paper cũng đề xuất ràng buộc bất biến lật ngang cho gradient attention:

$$
\mathcal L_{sym}(W,x)=\mathcal L(W,x)+\frac\beta2\left\|\frac{\partial}{\partial x}\mathcal L(W,x)-\mathrm{flip}\!\left(\frac{\partial}{\partial x}\mathcal L(W,\mathrm{flip}(x))\right)\right\|_2
$$

Đây là ràng buộc mềm, có tác dụng regularization theo quan sát của tác giả.

### 3.4 Pseudo-code (activation-based AT)

```
for x, y in loader:
    with no_grad: acts_T = teacher.forward_collect(x, groups)     # tensor C x H x W theo nhóm
    logits_S, acts_S = student.forward_collect(x, groups)
    loss = cross_entropy(logits_S, y)
    if use_KD: loss += alpha_term(KD(logits_S, teacher_logits, T=4))   # alpha=0.9 (CIFAR)
    for j in pairs:                                  # cặp nhóm cùng độ phân giải
        QS = normalize( flatten( sum_c (acts_S[j]**2) ) )   # F_sum^2 + L2 norm
        QT = normalize( flatten( sum_c (acts_T[j]**2) ) )
        loss += (beta_j / 2) * norm_p(QS - QT)       # p = 2
    step(loss)
```

Ví dụ nhỏ (của chúng ta): tensor 2 kênh × 1×2: kênh 1 = $[3,0]$, kênh 2 = $[4,0]$ → $F^2_{sum}=[25,0]$ → chuẩn hoá L2: $[1,0]$. Nếu student có $[0.6,0.8]$ bản đồ chuẩn hoá thì khoảng cách $\approx\sqrt{0.4^2+0.8^2}=0.89$. Chỉ hình dạng (vị trí nổi bật) được so, độ lớn thì không.

## 4. Thí nghiệm và kết quả chính

### 4.1 CIFAR-10 (activation-based, Table 1)

Cài đặt: augmentation lật ngang + random crop, batch norm; KD với temperature 4, $\alpha=0.9$; sai số là **trung vị 5 lần chạy**. Lỗi test (%):

| Student | Teacher | Student | AT | F-ActT | KD | AT+KD | Teacher |
|---|---|---|---|---|---|---|---|
| NIN-thin (0.2M) | NIN-wide (1M) | 9.38 | 8.93 | 9.05 | 8.55 | 8.33 | 7.28 |
| WRN-16-1 (0.2M) | WRN-16-2 (0.7M) | 8.77 | 7.93 | 8.51 | 7.41 | 7.51 | 6.31 |
| WRN-16-1 (0.2M) | WRN-40-1 (0.6M) | 8.77 | 8.25 | 8.62 | 8.39 | 8.01 | 6.58 |
| WRN-16-2 (0.7M) | WRN-40-2 (2.2M) | 6.31 | 5.85 | 6.24 | 6.08 | 5.71 | 5.23 |

F-ActT = chuyển giao toàn bộ activation kiểu FitNets (L2 trên activation đầy đủ, 1×1 conv để khớp hình dạng). Quan sát: AT luôn tốt hơn student baseline; AT+KD tốt nhất trong ba hàng (hàng WRN-16-2 → WRN-16-1 thì KD thuần 7.41 và AT+KD 7.51 tương đương, KD nhỉnh hơn một chút).

### 4.2 Gradient-based và so sánh (Table 3)

Baseline là NIN mảnh 0.2M tham số, không batch norm, chỉ lật ngang, một lần chạy. Teacher (cho AT và KD) là wide NIN 1M. Lỗi CIFAR-10 (%):

| Phương pháp | Lỗi |
|---|---|
| Baseline | 13.5 |
| min-$\ell_2$ (Drucker & LeCun) | 12.5 |
| gradient-based AT | 12.1 |
| KD | 12.1 |
| symmetry norm | 11.8 |
| activation-based AT | 11.2 |

Activation-based AT tốt nhất trong thiết lập này. Paper lưu ý phải huấn luyện một teacher mới (có batch norm, không ReLU trước pooling) để áp dụng AT trong cài đặt này.

### 4.3 Transfer learning (Table 4) và ImageNet (Table 5)

Finetune ResNet-18 (student) với teacher ResNet-34 đã finetune; loss $F^2_{sum}$ ở 2 nhóm cuối. Lỗi (%) :

| | ImageNet→CUB | ImageNet→Scenes |
|---|---|---|
| student ResNet-18 | 28.5 | 28.2 |
| KD | 27 (-1.5) | 28.1 (-0.1) |
| AT | 27 (-1.5) | 27.1 (-1.1) |
| teacher ResNet-34 | 26.5 | 26 |

ImageNet (single crop validation error, top-1 / top-5 %): ResNet-18: 30.4 / 10.8; ResNet-18 + AT: 29.3 / 10.0; ResNet-34: 26.1 / 8.3. Tức AT cải thiện 1.1% top-1 và 0.8% top-5. Chỉ dùng 2 loss ở 2 nhóm cuối, không chỉnh siêu tham số; hai loss AT được thêm vào ở epoch 60/100 (theo chú thích bảng). Paper nói KD không cho kết quả tích cực trên ImageNet với cặp này (làm chậm hội tụ) và đoán AT là phương pháp chuyển giao tri thức đầu tiên chạy thành công trên ImageNet.

## 5. Phân tích / ablation

- **Số điểm đặt loss:** WRN-16-2 → WRN-16-1: một loss ở group1/group2/group3 riêng cho lỗi 8.11 / 7.96 / 7.97; cả ba loss: 7.93. Mỗi loss đóng góp thêm một phần.
- **Hàm ánh xạ attention (Table 2, lỗi CIFAR-10, trung vị 5 lần chạy, không AT là 8.77):** $F_{sum}$ 7.99; $F^2_{sum}$ 7.93; $F^4_{sum}$ 8.09; $F^2_{max}$ 8.08. (Trong text trích xuất, ký hiệu mũ khó đọc, thứ tự theo bảng gốc: $F_{sum}$, $F^2_{sum}$, $F^4_{sum}$, $F^1_{max}$ hoặc $F^2_{max}$, nên chỉ nên tin xu hướng: các hàm dựa trên tổng cho kết quả gần nhau và tốt hơn hàm dựa trên max.) Paper chọn $F^2_{sum}$.
- **Hệ số $\beta$:** thường quanh 0.1; đặt bằng 10^3 chia cho (số phần tử bản đồ × batch size) mỗi lớp. Khi kết hợp với KD thì giảm $\beta$ dần trong quá trình huấn luyện. (Số mũ của 10 trong text bị mất định dạng; suy từ ngữ cảnh "usually varies about 0.1", đừng coi là chính xác.)
- **Attention so với full activation:** F-ActT (FitNets-style với conv 1×1) chỉ cải thiện nhẹ (ví dụ 8.77 → 8.51) trong khi AT cho 7.93. Cả hai đều tăng tốc hội tụ (Fig. 7b), nhưng AT cho độ chính xác cuối tốt hơn nhiều. Tác giả nhận xét thú vị: attention map mang thông tin quan trọng hơn cho việc chuyển giao so với toàn bộ activation.
- **Chuẩn hoá L2** và giảm dần $\beta$ cải thiện kết quả (theo tác giả).
- **Fine-grained (CUB):** AT tốt hơn KD (27 so 28.5 baseline... cả hai đều -1.5 theo bảng; còn trên Scenes AT -1.1 so KD -0.1). Tác giả đoán attention trung gian quan trọng với nhận dạng chi tiết. Đây là giả thuyết, không được kiểm chứng riêng.

## 6. Giới hạn

**Do tác giả nêu:**
- Gradient-based AT chỉ thử với NIN không batch norm trên CIFAR vì cần vi phân tự động phức tạp; chưa rõ batch norm nên hoạt động ra sao ở lần lan truyền ngược thứ hai.
- Trên ImageNet mới đặt loss ở 2 trong 4 nhóm và chưa tinh chỉnh siêu tham số (dự định cập nhật).
- Chưa thử detection hoặc weakly-supervised localization nơi thông tin không gian quan trọng hơn.
- CIFAR có độ phân giải sau downsampling rất nhỏ nên ít chỗ cho attention.

**Do chúng ta nhận thấy:**
- Chỉ CNN phân loại ảnh; không có Transformer, không có chuỗi.
- Giả định "|kích hoạt| ≈ độ quan trọng" thô; với kích hoạt có dấu hoặc khối lượng "massive activations" (tồn tại ở LLM/ViT lớn) giả định này có thể vỡ (suy đoán của chúng ta).
- Hiệu ứng cải thiện nhỏ (khoảng 0.4 đến 1.1 điểm %), tương đương nhiễu giữa các lần chạy ở một số cặp (CIFAR dùng trung vị 5 lần, ImageNet chỉ một lần).
- Gradient-based AT đắt và chỉ có kết quả 1 lần chạy, nên so sánh trong Table 3 yếu thống kê.

## 7. Liên hệ với các bài khác

- **03 FitNets:** bài này trực tiếp so với FitNets-style hint (F-ActT) và thắng; bỏ regressor.
- **02 Hinton KD:** AT+KD cộng gộp; trên ImageNet KD không giúp.
- **05 KD survey:** xếp AT vào feature-based/attention-based KD.
- **12 Relational KD, 13 CRD:** chuyển từ khớp từng điểm sang khớp quan hệ/đối nghịch tương phản.
- **18 TinyBERT, 19 MiniLM:** "attention transfer" cho Transformer thực sự là khớp ma trận attention/quan hệ self-attention; cùng tinh thần nhưng khác định nghĩa. (Chú ý nhầm tên: AT ở đây không phải self-attention.)
- **23 AM-RADIO, 24 CLIP-KD, 26 LLaVA-KD:** feature distillation trên ViT, trong đó chuẩn hoá feature trước khi khớp là chi tiết quan trọng, như chuẩn hoá L2 ở đây.

## 8. Ứng dụng cho dự án (suy luận)

- **ViT 12 block của LightOnOCR:** có thể áp dụng ý tưởng "khớp bản đồ không gian đã chuẩn hoá" cho các token patch của ViT: với mỗi lớp, tính $\sum_c |h_{p,c}|^2$ theo patch rồi chuẩn hoá L2 qua các patch, so với teacher. Với OCR, bản đồ này cho biết vùng chữ so với nền; student cắt tỉa (ít kênh hơn) vẫn có cùng số patch nên **không cần projector** khi chỉ khớp bản đồ 1 chiều. Đây là điểm hấp dẫn khi pruning kênh làm thay đổi chiều ẩn.
- **Khớp rẻ cho nhiều lớp:** vì AT chỉ cần một vector độ dài số-patch mỗi lớp, ta có thể đặt loss ở nhiều lớp của ViT và decoder mà tốn rất ít bộ nhớ so với khớp toàn bộ hidden state.
- **Hạn chế cho phần decoder:** decoder sinh token tự hồi quy; "bản đồ không gian" tương ứng là bản đồ theo vị trí chuỗi (norm kích hoạt theo token). Khớp norm theo token ít ý nghĩa hơn; cần thử.
- **D_lin/D_res:** AT bỏ thông tin hướng (chỉ giữ độ lớn tổng theo kênh), nên không thể sửa D_lin (cái sửa được bằng ánh xạ tuyến tính, cần hướng). Chỉ nên xem AT như tín hiệu phụ để định hướng *vùng quan trọng*, không thay thế khớp hidden state hay KD output. Đây là suy luận.
- Quan sát "cắt neuron MLP phục hồi kém": AT không nhắm cơ chế này; ít liên quan trực tiếp.

## 9. Thí nghiệm gợi ý

1. **AT cho ViT sau khi prune kênh:** thêm loss AT (bản đồ theo patch, L2-chuẩn hoá) ở 3 vị trí của ViT (block 4, 8, 12) cạnh KD output; đo CER/WER và tốc độ hội tụ so với KD thuần.
2. **AT vs khớp đầy đủ hidden state:** cùng điểm đặt, so (a) AT (không projector), (b) MSE hidden state với projector tuyến tính. Báo cáo chất lượng và bộ nhớ huấn luyện; kiểm chứng tuyên bố "AT tốt hơn full activation" ở bối cảnh VLM.
3. **Chuẩn hoá:** bỏ chuẩn hoá L2 để xác nhận tầm quan trọng (paper nhấn mạnh quan trọng), thử $p\in\{1,2,4\}$ trong $F^p_{sum}$.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| Attention map | $F(A)=\sum_i|A_i|^p$ (chọn $p=2$), $F:\mathbb R^{C\times H\times W}\to\mathbb R^{H\times W}$ |
| Loss AT | $\mathcal L+\frac\beta2\sum_j\|Q_S^j/\|Q_S^j\|_2-Q_T^j/\|Q_T^j\|_2\|_p$ |
| Gradient attention | $J=\partial\mathcal L/\partial x$; cần double backprop |
| Symmetry loss | ép $J(x)$ khớp lật của $J(\mathrm{flip}\,x)$ |
| Kết hợp KD | thêm CE nới nhiệt độ ($T=4$, $\alpha=0.9$ trên CIFAR) |
| ImageNet | ResNet-18 30.4/10.8 → 29.3/10.0 với AT (ResNet-34: 26.1/8.3) |
| CIFAR-10 | WRN-16-1: 8.77 → 7.93 (AT), 7.41 (KD), 7.51 (AT+KD) |

## 11. Kết luận ngắn

Attention transfer biến feature trung gian thành bản đồ không gian đã chuẩn hoá và bắt student khớp chúng, không thêm tham số. Kết quả nhất quán nhưng khiêm tốn (ví dụ 8.77% xuống 7.93% lỗi CIFAR-10 cho WRN-16-1, ImageNet ResNet-18 cải thiện 1.1% top-1) và tốt hơn FitNets-style khớp toàn bộ activation trong các thí nghiệm của paper. Bài học đáng mang theo: khi student và teacher khác chiều ẩn, một tóm tắt đã chuẩn hoá của feature có thể là mục tiêu khớp rẻ và ổn định hơn feature thô.

**Link gốc:** https://arxiv.org/abs/1612.03928
