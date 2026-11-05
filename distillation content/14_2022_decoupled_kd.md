# Decoupled Knowledge Distillation (2022)

**Tác giả:** Borui Zhao, Quan Cui, Renjie Song, Yiyu Qiu, Jiajun Liang (MEGVII Technology, Waseda University, Tsinghua University).  
**Venue / arXiv:** arXiv:2203.08679 (v2, 12/7/2022); CVPR 2022 (theo ghi chú trên trang arXiv, bản text không ghi).  
**Nguồn:** https://arxiv.org/abs/2203.08679  
**Chủ đề:** Phân rã loss KD của Hinton thành TCKD (lớp đích) và NCKD (các lớp không phải đích); chỉ ra KD cổ điển "ghép cặp" (coupled) làm NCKD bị nén; đề xuất DKD với hai trọng số $\alpha,\beta$ độc lập.  
**Vị trí trong lộ trình:** Đọc sau 02 (Hinton KD), 12 (RKD), 13 (CRD), trước 15 (sequence-level KD). Bài này trả lời vì sao KD trên logit đôi khi "kém" feature-distillation và sửa đúng điểm yếu đó chỉ bằng đổi trọng số. Về sau 31 (Rethinking KL) và 30 (DistiLLM) tiếp tục phân tích dạng loss.

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

**Logit distillation** = student bắt chước *logit* (đầu ra trước softmax) của teacher, thường qua KL divergence trên softmax có nhiệt độ (bài 02). **Feature distillation** = bắt chước activation tầng giữa (FitNet, AT, CRD, ReviewKD... bài 03, 04, 13).

Sau FitNet (2015), phần lớn nghiên cứu chuyển sang feature distillation vì hiệu quả cao hơn trên nhiều task; logit distillation ít được quan tâm. Nhược điểm của feature distillation: chi phí huấn luyện cao hơn (module phụ, tham số thêm, bộ nhớ). Logit distillation gần như không tốn thêm gì, nhưng kết quả thấp hơn. Tác giả nghi ngờ rằng tiềm năng của logit distillation bị hạn chế bởi "những lý do chưa biết", và họ đi mổ xẻ chính công thức KD.

## 2. Ý tưởng chính

Một dự đoán phân loại có thể nhìn ở hai mức: (1) nhị phân: "lớp đích" so với "tất cả lớp còn lại"; (2) đa lớp *bên trong* các lớp không phải đích (ví dụ trong ảnh con mèo, xe tải giống mèo bao nhiêu so với xe buýt). KD cổ điển, về mặt toán học, bằng đúng KL ở mức (1) cộng với KL ở mức (2) nhân với hệ số $(1-p^T_t)$, với $p^T_t$ là độ tự tin của teacher trên lớp đích. Hệ số này nghĩa là: *teacher càng tự tin thì thông tin "dark knowledge" giữa các lớp sai càng bị nhân với số nhỏ*, trong khi đó chính là những mẫu mà thông tin này đáng tin nhất. DKD đơn giản là bỏ cái hệ số ràng buộc đó và cho hai thành phần hai trọng số độc lập $\alpha$ và $\beta$.

## 3. Phương pháp chi tiết

### 3.1 Ký hiệu

Mẫu thuộc lớp $t$; xác suất $p_i=\exp(z_i)/\sum_j\exp(z_j)$. Định nghĩa:

- $\mathbf b=[p_t,\ p_{\backslash t}]$: xác suất nhị phân của lớp đích và tổng các lớp còn lại, với $p_{\backslash t}=\sum_{k\ne t}p_k$.
- $\hat{\mathbf p}$: phân phối *chỉ trên các lớp không phải đích*, $\hat p_i=\exp(z_i)/\sum_{j\ne t}\exp(z_j)=p_i/p_{\backslash t}$.

### 3.2 Phân rã KD

KD gốc (bỏ nhiệt độ để gọn):

$$\mathrm{KD}=\mathrm{KL}(\mathbf p^T\|\mathbf p^S)=p^T_t\log\frac{p^T_t}{p^S_t}+\sum_{i\ne t}p^T_i\log\frac{p^T_i}{p^S_i}$$

Thay $p_i=p_{\backslash t}\hat p_i$ ở phần tổng, tách $\log\frac{p^T_i}{p^S_i}=\log\frac{\hat p^T_i}{\hat p^S_i}+\log\frac{p^T_{\backslash t}}{p^S_{\backslash t}}$, ta được:

$$\boxed{\mathrm{KD}=\underbrace{\mathrm{KL}(\mathbf b^T\|\mathbf b^S)}_{\mathrm{TCKD}}+(1-p^T_t)\underbrace{\mathrm{KL}(\hat{\mathbf p}^T\|\hat{\mathbf p}^S)}_{\mathrm{NCKD}}}\tag{5,6}$$

vì $p^T_{\backslash t}=1-p^T_t$. Nghĩa: **TCKD** = KL nhị phân trên "đích hay không"; **NCKD** = KL trên phân phối giữa các lớp không phải đích.

Hai "ghép cặp" (coupling) mà paper chỉ ra:
1. NCKD bị nhân với $(1-p^T_t)$: mẫu teacher dự đoán tốt thì NCKD gần 0.
2. TCKD và NCKD không có trọng số độc lập: không thể cân bằng chúng.

### 3.3 DKD

$$\boxed{\mathrm{DKD}=\alpha\,\mathrm{TCKD}+\beta\,\mathrm{NCKD}}\tag{7}$$

Tức thay $(1-p^T_t)$ bằng hằng $\beta$ và thêm $\alpha$ cho TCKD. Loss tổng huấn luyện: cross-entropy nhãn thật (trọng số 1.0) cộng DKD (nhân $T^2$ như KD chuẩn).

### Pseudo-code (từ Algorithm 1 của paper, viết lại gọn)

```python
def dkd_loss(l_stu, l_tea, t_onehot, alpha, beta, T):
    p_stu, p_tea = softmax(l_stu / T), softmax(l_tea / T)
    # nhị phân: [p_target, p_non_target]
    b_stu = stack([p_stu[t_onehot], p_stu[~t_onehot].sum(-1)], -1)
    b_tea = stack([p_tea[t_onehot], p_tea[~t_onehot].sum(-1)], -1)
    tckd = kl_div(log(b_stu), b_tea)
    # phân phối trên các lớp không phải đích (loại logit đích ra, rồi softmax lại)
    pn_stu = softmax(l_stu[~t_onehot] / T)       # [N, C-1]
    pn_tea = softmax(l_tea[~t_onehot] / T)
    nckd = kl_div(log(pn_stu), pn_tea)
    # KD cổ điển tương đương: (tckd + (1 - p_tea_t) * nckd) * T**2
    return (alpha * tckd + beta * nckd) * T**2
```

Chi tiết hiện thực trong Phụ lục: CIFAR-100 dùng $T=4$, $\alpha=1.0$, **warm-up tuyến tính 20 epoch** (vì $\beta$ lớn khiến loss ban đầu to); ImageNet dùng $T=1$, $\alpha=0.5$; COCO dùng $\alpha=1.0$, $\beta=0.25$, $T=1$.

### Ví dụ số

Teacher trả $p^T_t=0.99$ cho một mẫu dễ, còn lại 0.01 chia cho hai lớp sai theo tỉ lệ 0.7/0.3 (tức $\hat{\mathbf p}^T=[0.7,0.3]$). KD cổ điển có NCKD nhân với $1-0.99=0.01$: thông tin "lớp sai nào giống đích hơn" gần như bị bỏ qua. Với DKD và $\beta=8$, NCKD nhân với 8, gấp 800 lần. Với mẫu khó ($p^T_t=0.5$) hệ số KD là 0.5, còn DKD vẫn 8: DKD không còn "ưu tiên" mẫu khó hơn mẫu dễ ở phần NCKD. (Ví dụ minh họa của chúng ta.)

## 4. Thí nghiệm và kết quả chính

Dữ liệu: CIFAR-100 (ResNet, WRN, ShuffleNet, VGG, MobileNet-V2), ImageNet, MS-COCO (Faster R-CNN + FPN).

### 4.1 Hiệu quả riêng của TCKD và NCKD (Table 1, CIFAR-100, top-1 %, mọi trọng số = 1.0)

Bảng gốc bị mangle khi trích từ PDF; các số dưới đây được đọc theo từng khối teacher/student và khớp với phần văn bản.

| Teacher → Student | Student | TCKD+NCKD (=KD) | Chỉ TCKD | Chỉ NCKD |
|---|---|---|---|---|
| ResNet32×4 → ResNet8×4 | 72.50 | 73.63 (+1.13) | 68.63 (−3.87) | 74.26 (+1.76) |
| ResNet32×4 → ShuffleNet-V1 | 70.50 | 74.29 (+3.79) | 70.52 (+0.02) | 74.91 (+4.41) |
| WRN-40-2 → WRN-16-2 | 73.26 | 74.96 (+1.70) | 70.96 (−2.30) | 74.76 (+1.50) |
| WRN-40-2 → ShuffleNet-V1 | 70.50 | 74.92 (+4.42) | 70.62 (+0.12) | 75.12 (+4.62) |

Kết luận của tác giả: TCKD một mình vô ích hoặc có hại; NCKD một mình bằng hoặc tốt hơn KD đầy đủ, nên NCKD (kiến thức giữa các lớp sai) mới là "dark knowledge" thực sự.

### 4.2 TCKD giúp khi dữ liệu "khó"

(ResNet32×4 → ResNet8×4 hoặc ShuffleNet-V1; mọi so sánh là có/không thêm TCKD lên NCKD)

- Augmentation mạnh (AutoAugment, teacher 81.29%): ResNet8×4 từ 73.82 lên 75.33 (+1.51); ShuffleNet-V1 từ 77.13 lên 77.98 (+0.85).
- Nhiễu nhãn đối xứng: 0.1: 70.99 → 70.96 (−0.03); 0.2: 67.55 → 68.03 (+0.48); 0.3: 64.62 → 65.26 (+0.64).
- ImageNet (ResNet-34 → ResNet-18): 70.71 → 71.03 (+0.32).

Diễn giải của tác giả: TCKD chuyển thông tin về độ khó của mẫu (mẫu có $p^T_t=0.99$ dễ hơn mẫu $0.75$); khi dữ liệu dễ học (CIFAR-100 có thể fit 100% trên train) thì thông tin đó không hữu ích.

### 4.3 NCKD bị nén trên mẫu mà teacher dự đoán tốt (Table 5)

Chia mẫu thành hai nửa theo $p^T_t$ và chỉ áp NCKD lên một nửa (CE vẫn trên toàn bộ): áp lên nửa teacher tự tin nhất (top 50%) cho 74.23 (đọc từ bảng đã mangle, khớp mô tả trong văn bản), nửa còn lại 73.96, cả hai nửa 74.26. Tức mẫu được dự đoán tốt mang tri thức phong phú hơn, nhưng KD cổ điển lại giảm trọng số của chúng.

### 4.4 Ablation $\alpha$, $\beta$ (ResNet32×4 → ResNet8×4)

| $\beta$ | $1-p^T_t$ (KD) | 1.0 | 2.0 | 4.0 | 8.0 | 10.0 |
|---|---|---|---|---|---|---|
| top-1 (α=1) | 73.63 | 74.79 | 75.44 | 75.94 | 76.32 | 76.18 |

| $\alpha$ (β=8) | 0.0 | 0.2 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| top-1 | 75.30 | 75.64 | 76.12 | 76.32 | 76.11 | 75.42 |

Tách $(1-p^T_t)$ khỏi NCKD cho +1.16 (73.63 → 74.79); tách trọng số TCKD và NCKD cho thêm +1.53 (74.79 → 76.32). TCKD cần thiết (α=0 thấp hơn 1.02) và ổn định quanh α≈1.

### 4.5 CIFAR-100 chính (Table 6, teacher/student cùng họ; trung bình 5 lần)

| Teacher → Student | Student | KD | CRD | ReviewKD | DKD | Δ so với KD |
|---|---|---|---|---|---|---|
| ResNet56 → ResNet20 | 69.06 | 70.66 | 71.16 | 71.89 | 71.97 | +1.31 |
| ResNet110 → ResNet32 | 71.14 | 73.08 | 73.48 | 73.89 | 74.11 | +1.03 |
| ResNet32×4 → ResNet8×4 | 72.50 | 73.33 | 75.51 | 75.63 | 76.32 | +2.99 |
| WRN-40-2 → WRN-16-2 | 73.26 | 74.92 | 75.48 | 76.12 | 76.24 | +1.32 |
| WRN-40-2 → WRN-40-1 | 71.98 | 73.54 | 74.14 | 75.09 | 74.81 | +1.27 |
| VGG13 → VGG8 | 70.36 | 72.98 | 73.94 | 74.84 | 74.68 | +1.70 |

Kiến trúc khác nhau (Table 7): ResNet32×4 → ShuffleNet-V1: KD 74.07, ReviewKD 77.45, DKD 76.45 (+2.38); WRN-40-2 → ShuffleNet-V1: KD 74.83, DKD 76.70 (+1.87); VGG13 → MobileNet-V2: KD 67.37, DKD 69.71 (+2.34); ResNet50 → MobileNet-V2: KD 67.35, ReviewKD 69.89, DKD 70.35 (+3.00); ResNet32×4 → ShuffleNet-V2: KD 74.45, ReviewKD 77.78, DKD 77.07 (+2.62).

### 4.6 ImageNet

| | Teacher | Student | AT | OFD | CRD | ReviewKD | KD | KD* | DKD |
|---|---|---|---|---|---|---|---|---|---|
| R34 → R18 top-1 | 73.31 | 69.75 | 70.69 | 70.81 | 71.17 | 71.61 | 70.66 | 71.03 | 71.70 |
| R34 → R18 top-5 | 91.42 | 89.07 | 90.01 | 89.98 | 90.13 | 90.51 | 89.88 | 90.05 | 90.41 |
| R50 → MobileNet-V1 top-1 | 76.16 | 68.87 | 69.56 | 71.25 | 71.37 | 72.56 | 68.58 | 70.50 | 72.05 |
| R50 → MobileNet-V1 top-5 | 92.86 | 88.76 | 89.33 | 90.34 | 90.41 | 91.00 | 88.98 | 89.80 | 91.05 |

(KD* là bản chạy lại của các tác giả.)

### 4.7 COCO (Table 10, Faster-RCNN-FPN, AP)

| Pair | Student | KD | ReviewKD | DKD | DKD+ReviewKD |
|---|---|---|---|---|---|
| R101 → R18 | 33.26 | 33.97 | 36.75 | 35.05 | 37.01 |
| R101 → R50 | 37.93 | 38.35 | 40.36 | 39.25 | 40.65 |
| R50 → MV2 | 29.47 | 30.13 | 33.71 | 32.34 | 34.35 |

DKD vượt KD nhưng thua ReviewKD (feature); kết hợp hai cho kết quả tốt nhất.

## 5. Phân tích / ablation

- **Teacher lớn không hẳn là teacher tốt:** Tác giả đề xuất lời giải thích mới: khi teacher to hơn, $p^T_t$ gần 1 hơn, hệ số $(1-p^T_t)$ nhỏ hơn, NCKD bị nén mạnh hơn. Table 11 (student WRN-16-2, teacher WRN-28-2/40-2/16-4/28-4 với accuracy 75.45/75.61/77.51/78.60): KD cho 75.37/74.92/75.79/75.04 (không tăng theo teacher), DKD cho 75.92/76.24/76.00/76.45. Table 12 (VGG13/WRN-16-4/ResNet50): KD 74.93/75.79/75.36, DKD 75.45/76.00/76.60. Đây là giả thuyết được ủng hộ bằng thí nghiệm, chưa phải chứng minh.
- **Hiệu quả huấn luyện (Figure 2):** DKD có chi phí gần như bằng KD, không thêm tham số, trong khi feature-based thêm thời gian và bộ nhớ; DKD đạt trade-off tốt nhất (đọc từ hình).
- **Transferability (Table 13):** WRN-16-2 distill từ WRN-40-2, linear probe: baseline/KD/FitNet/CRD/ReviewKD/DKD = STL-10 69.7/70.9/70.3/71.6/72.4/72.9; Tiny-ImageNet 33.7/33.9/33.5/35.6/36.6/37.1.
- **Chỉnh $\beta$ (Phụ lục A.3):** $\beta$ tốt nhất tỉ lệ thuận với khoảng cách logit $z_t-z_{\max}$ trung bình của teacher. Với ResNet56/WRN-40-2/ResNet32×4 gap là 5.36/7.24/8.40 và $\beta$ chọn là 2.0/6.0/8.0 (Table A.2: ResNet110 6.73 → 2.0, VGG13 8.25 → 6.0, ResNet50 8.53 → 8.0). Đây là heuristic, tác giả nói quan hệ chặt chưa được khảo sát.
- **Vì sao TCKD một mình có thể hại (A.5):** với $T=4$, gradient của TCKD kéo mạnh các logit lớp sai lên, hại độ đúng của student khi không có thông tin tương đồng giữa lớp; khi chỉ dùng TCKD và $T=1$ kết quả gần baseline (73.24 so với 73.26), $T=4$ cho 70.96.
- **Phân tích định tính:** t-SNE của DKD tách lớp rõ hơn KD; hiệu ma trận tương quan logit student-teacher nhỏ hơn (Figure 3, 4).

## 6. Giới hạn

**Do tác giả nêu:**
- Không vượt feature-based SOTA (ReviewKD) trên detection vì logit không mang thông tin định vị.
- Quan hệ chặt chẽ giữa $\beta$ và hiệu năng chưa được nghiên cứu hết; hướng dẫn chỉnh $\beta$ chỉ là heuristic.

**Do chúng ta nhận xét:**
- Toàn bộ trên phân loại ảnh có $C$ nhỏ (100 hoặc 1000 lớp) và một nhãn mỗi mẫu; chưa có chuỗi/ngôn ngữ, nơi "lớp đích" thay đổi theo vị trí và $C$ là vocab 30k-150k.
- $\beta$ lớn (tới 8) trên CIFAR phụ thuộc cặp teacher/student và cần warm-up; độ nhạy cao nếu áp sang bài toán khác.
- Lý thuyết "bigger teacher bị nén NCKD" là giải thích hợp lý nhưng không loại trừ giải thích capacity gap (bài 08, 41).
- Cải thiện của DKD so với ReviewKD không nhất quán (ví dụ ShuffleNet: ReviewKD 77.45 > DKD 76.45).

## 7. Liên hệ với các bài khác

- **02 Hinton KD:** DKD là phân tích đại số trên chính loss của Hinton; "dark knowledge" = NCKD.
- **12 RKD, 13 CRD:** hai đối thủ cần vượt; trong Table 6 CRD cho 71.16/73.48/75.51... còn DKD cho 71.97/74.11/76.32...
- **08 Teacher assistant, 09 Efficacy of KD, 41 Capacity gap law:** DKD đưa ra một giải thích khác cho "teacher quá lớn không giúp ích" (NCKD bị nén chứ không chỉ capacity gap).
- **10 Does KD really work:** cùng quan tâm đến việc student có thực sự bắt chước phân phối của teacher hay không.
- **31 Rethinking KL, 30 DistiLLM, 28 MiniLLM, 29 GKD:** các bài LLM cũng xem lại trọng số/hướng KL; DKD là ví dụ sớm cho việc tách và gán lại trọng số theo từng thành phần của KL (suy luận).

## 8. Ứng dụng cho dự án (suy luận)

*Mức liên quan: trung bình, và có một điểm đáng để thử.*

1. **Teacher OCR rất tự tin.** Với đầu ra OCR dài gần như tất định, ở hầu hết token teacher 4B có $p^T_t\approx1$ (token đúng). Khi đó KL token-level $=\mathrm{TCKD}+(1-p^T_t)\mathrm{NCKD}$ chủ yếu bị TCKD chi phối; phần NCKD (ví dụ khi teacher cho "0" tự tin nhưng vẫn xếp "O" và "D" là phương án sai gần nhất) gần như bị nhân với 0. Đây chính xác là kịch bản mà paper nói KD cổ điển yếu: tri thức tương đồng giữa token sai bị bỏ rơi trên mẫu dễ. Chúng ta suy luận (chưa được chứng minh ở chuỗi) rằng DKD với $\beta$ vừa phải có thể giúp student giữ được thứ hạng của các ứng viên sai, hữu ích cho các lỗi nhầm ký tự gần giống (0/O, 1/l/I, dấu tiếng Việt).
2. **Token là "mẫu", vocab là "lớp".** Triển khai: tại mỗi vị trí chuỗi, "lớp đích" là token đúng (nhãn thật hoặc argmax của teacher khi không có nhãn); NCKD là KL trên phân phối đã loại token đích và softmax lại (có thể chỉ trên top-k token của teacher để tiết kiệm bộ nhớ, cần kiểm tra).
3. **Cắt MLP neuron khó phục hồi:** nếu damage nằm ở thông tin tinh trong đuôi phân phối (suy luận về $D_{res}$) thì tăng trọng số NCKD đánh đúng chỗ đó. Cắt layer thì thường giữ được đuôi tốt hơn.
4. **Chi phí gần 0:** chỉ thay công thức loss, không thêm module hay tham số, phù hợp giới hạn GPU 8 GB.
5. **Không áp được cho khoảng cách capacity quá lớn một mình:** từ 4B xuống 0.1B vẫn có thể cần bài 08/41 (trợ giảng, lịch trình).
6. Các hyperparameter của paper ($\beta$=8, $T$=4) đến từ CIFAR; không nên copy.

## 9. Thí nghiệm gợi ý

1. **Đo trước khi sửa:** trên tập dev OCR, với teacher 4B, tính phân bố $p^T_t$ theo token; xem bao nhiêu phần trăm token có $p^T_t>0.95$ (nếu hầu hết thì mất NCKD là vấn đề thật). Đồng thời đo $z_t-z_{\max}$ trung bình để chọn $\beta$ theo heuristic Phụ lục A.3.
2. **DKD so với KL thường trên student đã prune:** cùng ngân sách, đổi $\mathrm{KL}\to\alpha\,\mathrm{TCKD}+\beta\,\mathrm{NCKD}$, quét $\beta\in\{1,2,4,8\}$ với warm-up. Đo CER, và tách đo lỗi ở các cặp ký tự dễ nhầm.
3. **So sánh hai kiểu cắt:** lặp thí nghiệm 2 cho (a) cắt layer, (b) cắt MLP neuron, để kiểm tra giả thuyết rằng DKD giúp (b) nhiều hơn (a).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| $\mathbf b$ | Xác suất nhị phân [đích, không-đích] |
| $\hat{\mathbf p}$ | Phân phối trên các lớp không-đích, $\hat p_i=p_i/p_{\backslash t}$ |
| TCKD | $\mathrm{KL}(\mathbf b^T\|\mathbf b^S)$: truyền "độ khó" của mẫu |
| NCKD | $\mathrm{KL}(\hat{\mathbf p}^T\|\hat{\mathbf p}^S)$: truyền tương đồng giữa lớp sai |
| KD cổ điển | $\mathrm{TCKD}+(1-p^T_t)\,\mathrm{NCKD}$ |
| DKD | $\alpha\,\mathrm{TCKD}+\beta\,\mathrm{NCKD}$ |
| Warm-up | Tăng tuyến tính trọng số loss KD trong 20 epoch đầu (CIFAR-100) |
| Hướng dẫn $\beta$ | Tỉ lệ với $z_t-z_{\max}$ trung bình của teacher |
| Dark knowledge | Thông tin ngầm trong xác suất của các lớp sai |

## 11. Kết luận ngắn

DKD cho thấy loss KD của Hinton có một cấu trúc ẩn: phần "tri thức giữa các lớp sai" (NCKD) bị nhân với $1-p^T_t$ nên bị tắt đúng ở những mẫu teacher tự tin. Tách hai phần và gán trọng số độc lập cho cải thiện 1 đến 3 điểm trên CIFAR-100, đạt mức tương đương feature distillation trên ImageNet, mà không tốn thêm chi phí. Với dự án OCR, ý tưởng này đáng thử vì teacher thường rất tự tin ở hầu hết token.

Link gốc: https://arxiv.org/abs/2203.08679
