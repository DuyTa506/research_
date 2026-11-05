# Relational Knowledge Distillation (2019)

**Tác giả:** Wonpyo Park, Dongju Kim, Yan Lu, Minsu Cho (POSTECH, Microsoft Research).  
**Venue / arXiv:** arXiv:1904.05068 [cs.CV], v2 (1/5/2019). Bản text không ghi tên hội nghị.  
**Nguồn:** https://arxiv.org/abs/1904.05068  
**Chủ đề:** Knowledge distillation dựa trên *quan hệ giữa các mẫu* (khoảng cách, góc) thay vì khớp từng đầu ra riêng lẻ; metric learning, phân loại, few-shot.  
**Vị trí trong lộ trình:** Đọc sau 02 (Hinton KD), 03 (FitNets), 04 (Attention transfer), 05 (survey). Bài này đổi câu hỏi "tri thức là gì" từ *đầu ra của từng mẫu* sang *cấu trúc quan hệ giữa các mẫu*. Bài kế tiếp: 13 (CRD) cũng dùng thông tin giữa các mẫu nhưng bằng contrastive loss.

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

**Knowledge distillation (KD)** = huấn luyện một mô hình nhỏ (*student*) bắt chước mô hình lớn (*teacher*). Trước bài này, hầu hết phương pháp KD có chung một dạng mà paper gọi là **Individual KD (IKD)**:

$$L_{IKD}=\sum_{x_i\in X} l\big(f_T(x_i),\,f_S(x_i)\big)\tag{1}$$

tức là với *từng* mẫu $x_i$, ép đầu ra của student $f_S(x_i)$ gần đầu ra của teacher $f_T(x_i)$. Ba ví dụ paper nêu:

- **Hinton KD** (bài 02): $f$ là logit, $l$ = KL giữa softmax có nhiệt độ $\tau$.
- **FitNet** (bài 03): $f$ là activation tầng giữa, $l$ = bình phương khoảng cách Euclid, thêm ánh xạ tuyến tính $\beta$ để khớp số chiều khi student hẹp hơn.
- **Attention transfer** (bài 04): $f$ là bản đồ attention.

Vấn đề: IKD buộc student phải tái tạo *giá trị* đầu ra của teacher. Nếu student có số chiều embedding nhỏ hơn (16, 32, 64 chiều so với 512 chiều của teacher), việc khớp từng vector là phi thực tế và không cần thiết. Paper lập luận (mượn từ ngôn ngữ học cấu trúc của Saussure) rằng ý nghĩa của một mẫu nằm ở *quan hệ của nó với các mẫu khác*, không phải ở tọa độ tuyệt đối.

## 2. Ý tưởng chính

Thay vì bảo student "mẫu này phải ra đúng vector này", RKD bảo student "hãy giữ *hình học* giữa các mẫu giống teacher". Cụ thể: với một nhóm 2 mẫu thì giữ tỉ lệ khoảng cách; với nhóm 3 mẫu thì giữ góc tạo bởi ba điểm embedding. Vì khoảng cách (đã chuẩn hóa) và góc không phụ thuộc số chiều hay hệ tọa độ, student ít chiều hơn vẫn có thể bắt chước được cấu trúc đó. IKD là trường hợp đặc biệt (quan hệ bậc 1, hàm thế là hàm đồng nhất), nên RKD là một tổng quát hóa. RKD cũng bổ trợ IKD: kết hợp hai loại cho kết quả tốt hơn.

## 3. Phương pháp chi tiết

### 3.1 Khung tổng quát

Ký hiệu $t_i=f_T(x_i)$, $s_i=f_S(x_i)$. Chọn một **relational potential function** $\psi$ (hàm "thế quan hệ": nhận một bộ $n$ embedding, trả về một đại lượng đo quan hệ của bộ đó):

$$\boxed{L_{RKD}=\sum_{(x_1,\dots,x_n)\in X^N} l\big(\psi(t_1,\dots,t_n),\ \psi(s_1,\dots,s_n)\big)}\tag{4}$$

Khi $n=1$ và $\psi$ = identity thì (4) quay về (1). Paper cho biết thế bậc cao hơn mạnh hơn nhưng tốn kém hơn, và đề xuất hai thế đơn giản.

### 3.2 Distance-wise loss (bậc 2)

$$\psi_D(t_i,t_j)=\frac1\mu\|t_i-t_j\|_2,\qquad \mu=\frac1{|X^2|}\sum_{(x_i,x_j)\in X^2}\|t_i-t_j\|_2\tag{5,6}$$

$\mu$ là khoảng cách trung bình trong *mini-batch*, để chỉ quan tâm khoảng cách *tương đối*, bất kể teacher và student khác scale (ví dụ khác số chiều). Paper nói chuẩn hóa này cho hội tụ ổn định và nhanh hơn.

$$\boxed{L_{RKD\text{-}D}=\sum_{(x_i,x_j)\in X^2} l_\delta\big(\psi_D(t_i,t_j),\psi_D(s_i,s_j)\big)}\tag{7}$$

với $l_\delta$ là **Huber loss** (bình phương khi sai lệch $\le1$, tuyến tính khi lớn hơn; ít nhạy với outlier hơn MSE):

$$l_\delta(x,y)=\begin{cases}\tfrac12(x-y)^2 & |x-y|\le1\\ |x-y|-\tfrac12 & \text{otherwise}\end{cases}$$

### 3.3 Angle-wise loss (bậc 3)

$$\psi_A(t_i,t_j,t_k)=\cos\angle t_it_jt_k=\langle e^{ij},e^{kj}\rangle,\qquad e^{ij}=\frac{t_i-t_j}{\|t_i-t_j\|_2}\tag{9}$$

$$\boxed{L_{RKD\text{-}A}=\sum_{(x_i,x_j,x_k)\in X^3} l_\delta\big(\psi_A(t_i,t_j,t_k),\psi_A(s_i,s_j,s_k)\big)}\tag{10}$$

Góc là thuộc tính bậc cao hơn khoảng cách, nên cho student nhiều tự do hơn; paper quan sát angle-wise thường hội tụ nhanh hơn và tốt hơn.

### 3.4 Huấn luyện

$$L=L_{task}+\lambda_{KD}\cdot L_{KD}\tag{11}$$

Lấy *tất cả* cặp/bộ ba trong mini-batch làm tuple (không lấy mẫu phức tạp). Tên biến thể: RKD-D, RKD-A, RKD-DA (cả hai). Lưu ý của paper: vì RKD không truyền giá trị từng đầu ra, không nên dùng *riêng* RKD ở nơi giá trị đầu ra là quan trọng (như lớp softmax của phân loại); khi đó phải dùng kèm IKD hoặc loss của task.

### Pseudo-code

```python
def rkd_loss(t, s, w_d=1.0, w_a=2.0):
    # t, s: [B, D_T], [B, D_S] embeddings của teacher (no_grad) và student
    # --- distance-wise ---
    dt = pdist(t)                    # [B, B] khoảng cách Euclid
    ds = pdist(s)
    dt = dt / dt[dt > 0].mean()      # chia cho mu (mean khoảng cách cặp trong batch)
    ds = ds / ds[ds > 0].mean()
    L_d = huber(ds, dt).mean()
    # --- angle-wise ---
    et = normalize(t[None, :, :] - t[:, None, :], dim=2)   # e^{ij}, [B, B, D]
    es = normalize(s[None, :, :] - s[:, None, :], dim=2)
    at = bmm(et, et.transpose(1, 2))                       # cos góc, [B, B, B]
    as_ = bmm(es, es.transpose(1, 2))
    L_a = huber(as_, at).mean()
    return w_d * L_d + w_a * L_a

loss = task_loss(student(x), y) + lam * rkd_loss(teacher(x).detach(), student_emb(x))
```

(Pseudo-code là của chúng ta, minh họa công thức 7 và 10; code chính thức do tác giả công bố tại http://cvlab.postech.ac.kr/research/RKD/.)

### Ví dụ nhỏ

Batch 3 mẫu. Teacher (2D): $t_1=(0,0),t_2=(2,0),t_3=(2,2)$. Khoảng cách: 2, $2\sqrt2$, 2; $\mu=(2+2.83+2)/3\approx2.28$, nên $\psi_D$ = 0.88, 1.24, 0.88. Student 1D (không thể giống toạ độ teacher): $s=(0,2,4)$ cho khoảng cách 2, 4, 2, $\mu=2.67$, $\psi_D$ = 0.75, 1.5, 0.75. Loss khác 0 nhưng nhỏ, và student 1D chỉ cần tái tạo *tỉ lệ* (cặp 1-3 xa gấp ~1.4 lần cặp kề) chứ không phải tọa độ. Với góc: $\psi_A(t_1,t_2,t_3)=\cos 90^\circ=0$, trong khi student 1D chỉ có góc 0 hoặc 180 độ, nên góc là ràng buộc *cứng hơn* khi số chiều chênh nhau nhiều (nhận xét của chúng ta).

## 4. Thí nghiệm và kết quả chính

Ba bài toán: metric learning (image retrieval), phân loại, few-shot. Độ đo retrieval: recall@K.

### 4.1 Metric learning (CUB-200-2011, Cars 196, Stanford Online Products)

Teacher: ResNet50 (pretrain ImageNet) + FC 512 chiều + l2 norm, huấn luyện bằng triplet loss. Student: ResNet18 với embedding 16/32/64/128 chiều. Với RKD *không dùng* task loss (triplet) hay nhãn gốc, chỉ học từ teacher. $\lambda_{RKD-D}=1,\lambda_{RKD-A}=2$. Batch 128, Adam, 5 ảnh dương mỗi lớp.

Recall@1, Cars 196 (trích Table 1b; ô RKD là "có l2 norm / không l2 norm"):

| Student | Triplet | FitNet | Attention | DarkRank | RKD-D | RKD-A | RKD-DA |
|---|---|---|---|---|---|---|---|
| ResNet18-16 | 45.39 | 57.46 | 46.44 | 64.00 | 63.23 / 66.02 | 61.39 / 66.25 | 61.78 / 66.04 |
| ResNet18-32 | 56.01 | 65.81 | 59.40 | 72.41 | 73.50 / 76.15 | 73.23 / 75.89 | 73.12 / 74.80 |
| ResNet18-64 | 64.53 | 70.67 | 67.24 | 76.20 | 78.64 / 80.57 | 77.92 / 80.32 | 78.48 / 80.17 |
| ResNet18-128 | 68.79 | 73.10 | 71.95 | 77.00 | 79.72 / 81.70 | 80.54 / 82.27 | 80.00 / 82.50 |

Teacher ResNet50-512 đạt 77.17 trên Cars 196 và 61.24 trên CUB. Student ResNet18-128 không l2 norm đạt 82.50, vượt teacher (paper nhấn mạnh điểm này). Trên CUB, ResNet18-128 RKD-DA đạt 59.69 / 60.67, thấp hơn teacher 61.24 (nên "student vượt teacher" trên CUB chỉ xảy ra ở một số cấu hình; ở bảng 1a các ô RKD ResNet18-128 là 58-60.9, nhỏ hơn 61.24). Paper nói gain tương đối của recall@1 từ RKD-DA tăng khi embedding nhỏ đi: 12.5, 13.8, 23.0, 27.7 trên CUB, và 20.0, 24.2, 33.5, 45.5 trên Cars.

Self-distillation (Table 2, student cùng kiến trúc, không l2 norm):

| | CUB | Cars | SOP |
|---|---|---|---|
| ResNet50-512-Triplet (teacher) | 61.24 | 77.17 | 76.58 |
| Gen1 | 65.68 | 85.65 | 77.61 |
| Gen2 | 65.11 | 85.61 | 77.36 |
| Gen3 | 64.26 | 85.23 | 76.96 |

Mọi thế hệ đều vượt teacher ban đầu nhưng không cải thiện thêm từ thế hệ 2.

So với state-of-the-art (Table 3): RKD-DA-128 (backbone ResNet50) đạt Recall@1 là 64.9 (CUB), 84.9 (Cars), 77.5 (SOP); đứng đầu nhóm ResNet50 (so với Margin-128: 63.6, 79.6, 72.7). Ở nhóm GoogLeNet, RKD-DA-512 đạt 61.4 / 82.3 / 75.1, nhưng ABE8-512 vẫn cao hơn trên Cars (85.2) và SOP (76.3).

### 4.2 Phân loại (Table 4)

Teacher ResNet50 / student VGG11-BN trên CIFAR-100; ResNet101 / ResNet18 trên Tiny ImageNet. $\lambda_{RKD-D}=25,\lambda_{RKD-A}=50$, thêm cross-entropy.

| Phương pháp | CIFAR-100 | Tiny ImageNet |
|---|---|---|
| Baseline (student tự train) | 71.26 | 54.45 |
| RKD-D | 72.27 | 54.97 |
| RKD-DA | 72.97 | 56.36 |
| HKD (Hinton) | 74.26 | 57.65 |
| HKD + RKD-DA | 74.66 | 58.15 |
| FitNet | 70.81 | 55.59 |
| FitNet + RKD-DA | 72.98 | 55.54 |
| Attention | 72.68 | 55.51 |
| Attention + RKD-DA | 73.53 | 56.55 |
| Teacher | 77.76 | 61.55 |

Đáng chú ý: ở phân loại, RKD *một mình* kém HKD; lợi ích nằm ở chỗ **kết hợp** với IKD.

### 4.3 Few-shot (Omniglot, miniImageNet; prototypical network, self-distillation)

Omniglot: RKD-DA 98.64 / 99.64 / 95.52 / 98.67 (5-way 1-shot, 5-shot, 20-way 1-shot, 5-shot) so với teacher 98.55 / 99.56 / 95.11 / 98.68. miniImageNet (5-way): RKD-DA 50.02 và 68.16 so với teacher 49.1 và 66.87; FitNet 50.38 / 68.08; Attention rất kém (34.67 / 46.21). Cải thiện nhỏ nhưng nhất quán so với teacher.

## 5. Phân tích / ablation

- **Không dùng l2 normalization tốt hơn** cho RKD (nhóm ô "X" trong Table 1 hầu như luôn cao hơn "O"), vì embedding được dùng cả không gian thay vì mặt cầu đơn vị. Ngược lại các phương pháp khác (kể cả DarkRank) tụt mạnh khi bỏ l2 norm (ví dụ DarkRank ResNet18-128 trên Cars: 52.92 so với 77.00).
- **D vs A vs DA:** ba biến thể gần nhau; không biến thể nào thắng áp đảo. Paper nói angle thường hội tụ nhanh/tốt hơn nhưng số liệu Table 1 không phân biệt rõ.
- **RKD là chuyên biệt hóa domain:** student huấn luyện bằng RKD trên Cars 196 có recall@1 trên miền khác (CUB, Stanford Dog, CIFAR-100) thấp hơn nhiều, trong khi teacher (triplet) gần với đặc trưng pretrain. Tức RKD thích nghi mạnh với miền huấn luyện đổi lại giảm khả năng khái quát (Figure 3, không có số liệu chi tiết trong text).
- **Vì sao student vượt teacher** (giả thuyết của tác giả): nhãn liên tục (khoảng cách, góc) mang thông tin mà nhãn nhị phân positive/negative của triplet không có, giống cách soft label của Hinton mang quan hệ giữa các lớp.
- **Bổ trợ với IKD:** Table 4 cho thấy cộng RKD vào HKD/FitNet/Attention hầu hết đều cải thiện.

## 6. Giới hạn

**Do tác giả nêu:**
- RKD không phù hợp để dùng một mình ở tầng mà giá trị đầu ra quan trọng (softmax).
- Thế bậc cao tốn kém hơn; paper chỉ thử bậc 2 và 3.
- Đánh đổi khả năng khái quát sang miền khác (Section 4.1.4).

**Do chúng ta nhận xét:**
- Chi phí angle-wise là $O(B^3)$ theo batch size (mọi bộ ba), điều này làm giảm khả năng mở rộng khi batch lớn hoặc khi mỗi "mẫu" là một chuỗi dài.
- Toàn bộ thí nghiệm là vision cỡ nhỏ (ResNet, CIFAR, CUB); không có thí nghiệm với chuỗi/ngôn ngữ/mô hình sinh.
- Cải thiện trên phân loại khá nhỏ (khoảng 1 đến 2 điểm khi dùng một mình), và hyperparameter ($\lambda$) phải chỉnh theo từng task (1/2, 25/50, 50/100).
- Paper không thử so sánh với các batch kích thước khác nhau; chất lượng ước lượng $\mu$ phụ thuộc batch.

## 7. Liên hệ với các bài khác

- **02 Hinton KD / 03 FitNets / 04 Attention transfer:** đều là IKD theo cách phân loại của bài này; RKD được thiết kế để cộng thêm vào chúng (Table 4).
- **05 KD survey:** RKD thuộc nhóm "relation-based knowledge" trong cách phân loại thường gặp (suy luận: nhãn này là của survey, không phải của bài này).
- **06 Born-again / 07 Deep mutual learning:** self-distillation của RKD (Table 2) cùng tinh thần với born-again, nhưng ở đây dùng quan hệ thay vì xác suất.
- **13 CRD:** cũng khai thác thông tin giữa các mẫu (dùng negative samples và contrastive objective) thay vì khớp từng mẫu; có thể đọc 12 như bước nối giữa IKD và CRD.
- **14 Decoupled KD:** nhìn lại KD dựa trên logit; cho thấy IKD cổ điển vẫn mạnh nếu tách đúng thành phần.
- **18 TinyBERT / 19 MiniLM:** MiniLM dùng quan hệ giữa các vị trí (Q-K, V-V) làm mục tiêu distill, là một dạng "relational" khác (suy luận, đọc ở bài 19).

## 8. Ứng dụng cho dự án (suy luận)

Ghi chú: dự án là OCR VLM sinh chuỗi dài, nên RKD *ít liên quan trực tiếp* như một công thức, nhưng có hai điểm đáng tham khảo.

1. **Khớp số chiều khác nhau tại điểm cắt.** Khi prune hidden dim hoặc MLP neuron (dự án quan sát recovery tệ hơn so với cắt layer), student có hidden size nhỏ hơn teacher 4B. RKD cho một cách distill *mà không cần projection* $\beta$: so sánh ma trận khoảng cách (hoặc ma trận Gram) giữa các token/patch trong một ảnh, thay vì so từng vector. Đây là loss tầng giữa không phụ thuộc số chiều. Mối liên hệ với $D_{lin}$ / $D_{res}$: RKD khớp *cấu trúc* nên bất biến với phép quay/scale (một phần $D_{lin}$ kiểu ánh xạ trực giao) nhưng không bù được phần $D_{res}$ (chúng ta suy luận; chưa được kiểm chứng).
2. **Vision encoder (ViT 12 block):** tính RKD trên các patch token hoặc embedding ảnh trong batch là khả thi vì đây gần với setting metric learning của paper. Nhưng ở decoder sinh chuỗi, mỗi vị trí token là một "mẫu" và $B\cdot T$ rất lớn, nên angle-wise gần như không dùng được; distance-wise trên cửa sổ nhỏ có thể.
3. Cảnh báo: paper nói rõ RKD không nên là loss duy nhất ở tầng đầu ra; ở logit vẫn cần KL kiểu Hinton (bài 02, 15).

## 9. Thí nghiệm gợi ý

1. **RKD-D trên đầu ra ViT:** chạy teacher 4B (hoặc encoder của nó) và student đã prune; thêm $L_{RKD-D}$ trên các patch embedding của *cùng một ảnh* (khoảng 100-300 patch, ma trận khoảng cách $N\times N$), so với MSE + projection tuyến tính học được. Đo: loss OCR hậu recovery và CER trên tập dev.
2. **Ablation trên cắt MLP neuron:** với cùng một pruned model, so ba loss tầng giữa: (a) MSE+projection, (b) RKD-D, (c) cả hai. Giả thuyết cần kiểm chứng: RKD giúp ít hơn ở trường hợp cắt neuron vì phần hỏng chủ yếu là $D_{res}$.
3. **Chi phí:** đo thời gian bước train khi thêm RKD-A ở $B=8$, $T$ = 256 vị trí token (kiểm tra có chịu nổi $O(B^3)$ hay $O(T^3)$ không; nên giới hạn bằng lấy mẫu ngẫu nhiên các bộ ba).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| IKD | KD khớp từng đầu ra riêng lẻ: $\sum_i l(f_T(x_i),f_S(x_i))$ |
| RKD | KD khớp quan hệ: $\sum l(\psi(t_1..t_n),\psi(s_1..s_n))$ |
| Relational potential $\psi$ | Hàm đo quan hệ của một bộ $n$ mẫu |
| $\psi_D$ | Khoảng cách Euclid chia cho trung bình khoảng cách trong batch |
| $\psi_A$ | Cosine góc tại đỉnh $t_j$ của bộ ba $(t_i,t_j,t_k)$ |
| Huber loss $l_\delta$ | Bình phương gần 0, tuyến tính xa 0 |
| Tổng loss | $L_{task}+\lambda_{KD}L_{KD}$ |
| Triplet loss | Ép $d(a,p)+m<d(a,n)$ cho anchor, positive, negative |
| Recall@K | Tỉ lệ truy vấn có ảnh cùng lớp trong top-K lân cận |
| Self-distillation | Student cùng kiến trúc với teacher |

## 11. Kết luận ngắn

RKD đề xuất rằng tri thức nằm ở *cấu trúc quan hệ* giữa các mẫu. Hai loss đơn giản (khoảng cách chuẩn hóa và góc) cho kết quả mạnh nhất trong metric learning (student nhỏ vượt cả teacher, ví dụ 82.50 so với 77.17 trên Cars 196), cải thiện vừa phải ở phân loại khi kết hợp với KD cổ điển. Giá trị với dự án của bạn chủ yếu ở chỗ gợi ý một loss tầng giữa không cần projection khi hidden size khác nhau, chứ không phải ở đầu ra sinh chuỗi.

Link gốc: https://arxiv.org/abs/1904.05068
