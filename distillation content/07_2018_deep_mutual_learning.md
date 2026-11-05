# Deep Mutual Learning (2017)

**Tác giả:** Ying Zhang, Tao Xiang, Timothy M. Hospedales, Huchuan Lu.  
**Venue / arXiv:** arXiv:1706.00384 (v1, 01/06/2017). Bản text đã đọc không ghi tên hội nghị; tiêu đề ghi năm 2017 theo ngày arXiv, còn tên file lộ trình dùng 2018 (năm xuất bản chính thức, thông tin ngoài paper).  
**Nguồn:** https://arxiv.org/abs/1706.00384  
**Chủ đề:** Học tương hỗ giữa nhiều mạng học viên (student cohort), không cần teacher huấn luyện trước; mimicry loss bằng KL divergence giữa các peer; giải thích bằng entropy hậu nghiệm cao và cực tiểu phẳng.  
**Vị trí trong lộ trình:** Đọc trước: 02 (Hinton KD), 06 (Born-again networks: student cùng cỡ vẫn học được từ teacher cùng cỡ). Bài này thêm: bỏ luôn khái niệm teacher huấn luyện trước; các mạng cùng train, cùng dạy lẫn nhau; và cho thấy dạy lẫn nhau còn thắng KD từ teacher mạnh cố định trong các thí nghiệm của họ. Đọc tiếp: 08 (Teacher assistant), 10 (Does KD really work).

> Đây là ghi chú học tập bằng tiếng Việt, diễn giải và tóm lược, không phải bản dịch paper. Số liệu lấy từ paper; phần "Ứng dụng cho dự án" và các nhận xét ghi là suy luận là của chúng ta.

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

Distillation chuẩn là truyền một chiều: teacher lớn đã train xong, student nhỏ chưa train, student học theo xác suất lớp (class posterior) của teacher. Lập luận động cơ trong paper: mạng nhỏ thường có đủ dung lượng biểu diễn hàm cần học, khó khăn nằm ở **tối ưu hóa** (khó tìm được tham số tốt), không phải ở kích thước. Teacher giúp vì tín hiệu mimic dễ tối ưu hơn tín hiệu nhãn cứng.

Khái niệm cần có:
- **Posterior / class probability:** vector softmax $p^m(x)$ trên $M$ lớp.
- **KL divergence** $D_{KL}(p\|q)=\sum_m p^m\log(p^m/q^m)$: đo độ lệch giữa hai phân phối, không đối xứng; phạt nặng khi $q$ cho xác suất gần 0 ở chỗ $p$ có khối lượng.
- **Posterior entropy cao / cực tiểu phẳng (wide minima):** nhiều nghiên cứu cùng thời (Entropy-SGD, Pereyra et al.) cho rằng nghiệm nằm trong "thung lũng rộng" và có output ít quá tự tin thì tổng quát hóa tốt hơn.

Câu hỏi của bài: nếu cả nhóm đều chưa được huấn luyện, thì tri thức "bổ sung" đến từ đâu? Có bị "người mù dắt người mù" (groupthink) không?

## 2. Ý tưởng chính

Train một nhóm mạng (cohort) cùng lúc từ khởi tạo ngẫu nhiên khác nhau. Mỗi mạng nhận hai loss: cross-entropy với nhãn thật và KL kéo xác suất của nó về xác suất của các mạng còn lại (peer). Vì mọi mạng đều bị nhãn thật dẫn dắt, chúng không trôi đi tùy ý; đồng thời, do khởi tạo khác nhau, xác suất của các lớp "thứ hai, thứ ba" khác nhau giữa các mạng, và mỗi mạng được học từ "ước lượng tập thể" về các lớp gần đúng này. Kết quả: mỗi mạng tốt hơn khi học một mình, và trong thí nghiệm của họ còn tốt hơn khi chưng cất từ một teacher mạnh cố định. Có thể dùng cả nhóm làm ensemble nếu muốn.

## 3. Phương pháp chi tiết

### 3.1 Hai mạng

Với $N$ mẫu, $M$ lớp, xác suất lớp $m$ của mạng $\Theta_1$ cho mẫu $x_i$: $p_1^m(x_i)=\dfrac{\exp(z_1^m)}{\sum_{m'}\exp(z_1^{m'})}$ (không có temperature trong công thức của paper). Loss giám sát:

$$L_{C_1}=-\sum_{i=1}^{N}\sum_{m=1}^{M} I(y_i,m)\log p_1^m(x_i)$$

Mimicry loss từ peer $\Theta_2$ và loss tổng:

$$\boxed{L_{\Theta_1}=L_{C_1}+D_{KL}(p_2\,\|\,p_1)},\qquad L_{\Theta_2}=L_{C_2}+D_{KL}(p_1\,\|\,p_2)$$

Chú ý hướng KL trong paper: $D_{KL}(p_2\|p_1)=\sum p_2\log(p_2/p_1)$, trong đó peer $p_2$ đóng vai "mục tiêu". Khi peer cho xác suất khác 0 mà mạng này cho 0 thì bị phạt rất nặng (đây là lý do paper dùng để giải thích cơ chế, mục 5).

### 3.2 Tối ưu hóa (Algorithm 1)

Ở **mỗi minibatch**, cập nhật lần lượt: tính $p_1,p_2$, cập nhật $\Theta_1$ theo $L_{\Theta_1}$; rồi tính lại $p_1,p_2$ (với $\Theta_1$ mới), cập nhật $\Theta_2$ theo $L_{\Theta_2}$. Lặp đến hội tụ.

```
init Theta_1..Theta_K khác nhau (random init khác)
repeat:
    sample minibatch x
    for k in 1..K:
        p = [softmax(f_{Theta_j}(x)) for j in 1..K]     # tính lại sau mỗi lần update
        L_k = CE(y, p_k) + 1/(K-1) * sum_{l != k} KL(p_l || p_k)
        Theta_k <- Theta_k - lr * grad(L_k)             # p_l coi là hằng số khi tính grad của k
until converge
```

(Paper viết gradient cho mạng đang cập nhật; việc coi $p_l$ là hằng số khi đạo hàm theo $\Theta_k$ là cách hiểu hợp lý của chúng ta, vì mỗi mạng chỉ cập nhật tham số của chính nó.)

### 3.3 Nhiều mạng

$$\boxed{L_{\Theta_k}=L_{C_k}+\frac{1}{K-1}\sum_{l=1,\,l\ne k}^{K} D_{KL}(p_l\,\|\,p_k)}$$

Hệ số $\frac{1}{K-1}$ giữ cho tín hiệu giám sát nhãn thật vẫn là thành phần chính. Khi $K=2$ trở về công thức ở 3.1. Có thể phân tán: mỗi mạng một thiết bị, chỉ truyền các vector xác suất nhỏ.

Phương án thay thế **DML_e**: coi trung bình các peer làm một teacher:

$$L_{\Theta_k}=L_{C_k}+D_{KL}(p_{avg}\,\|\,p_k),\qquad p_{avg}=\frac{1}{K-1}\sum_{l\ne k}p_l$$

Paper thấy DML_e **tệ hơn** DML (mục 5).

## 4. Thí nghiệm và kết quả chính

**Dữ liệu:** CIFAR-100 (50.000 train, 10.000 test, 100 lớp, top-1 accuracy). Market-1501 (person re-identification, 32.668 ảnh, 1.501 identity; 751 identity để train, 750 để test; train bằng phân loại 751 lớp, test bằng nearest-neighbour trên feature lớp pooling cuối; đo mAP và CMC Rank-k).

**Mạng và số tham số trên CIFAR-100:** ResNet-32 0.5M, MobileNet 3.3M, InceptionV1 7.8M, WRN-28-10 36.5M.

**Thiết lập:** CIFAR-100 theo Zagoruyko & Komodakis: SGD Nesterov, lr 0.1, momentum 0.9, batch 64, lr nhân 0.1 mỗi 60 epoch, 200 epoch, augmentation lật ngang và random crop. Market-1501: Adam, lr 0.0002, $\beta_1=0.5$, $\beta_2=0.999$, batch 16, 100.000 iteration, có và không pretrain ImageNet. TensorFlow, một GTX 1080.

**CIFAR-100 top-1 (%), cohort 2 mạng (Bảng 2):**

| Net 1 | Net 2 | Độc lập 1 / 2 | DML 1 / 2 | Chênh 1 / 2 |
|---|---|---|---|---|
| ResNet-32 | ResNet-32 | 68.99 / 68.99 | 71.19 / 70.75 | +1.20 / +1.76 |
| WRN-28-10 | ResNet-32 | 78.69 / 68.99 | 78.96 / 70.73 | +0.27 / +1.74 |
| MobileNet | ResNet-32 | 73.65 / 68.99 | 76.13 / 71.10 | +2.48 / +2.11 |
| MobileNet | MobileNet | 73.65 / 73.65 | 76.21 / 76.10 | +2.56 / +2.45 |
| WRN-28-10 | MobileNet | 78.69 / 73.65 | 80.28 / 77.39 | +1.59 / +3.74 |
| WRN-28-10 | WRN-28-10 | 78.69 / 78.69 | 80.28 / 80.08 | +1.59 / +1.39 |

Mọi tổ hợp đều cải thiện; mạng nhỏ hưởng lợi nhiều hơn; kể cả mạng lớn WRN-28-10 học cùng mạng nhỏ vẫn tốt lên.

**Market-1501 (Bảng 3), mỗi MobileNet trong cohort 2 mạng, báo cáo trung bình hai mạng:**

| Phương pháp | Pretrain ImageNet | Single-query mAP / Rank-1 | Multi-query mAP / Rank-1 |
|---|---|---|---|
| MobileNet | không | 46.07 / 72.18 | 54.31 / 79.81 |
| MobileNet + DML | không | 52.15 / 76.90 | 60.97 / 83.48 |
| MobileNet | có | 60.68 / 83.94 | 68.25 / 87.89 |
| MobileNet + DML | có | 68.83 / 87.73 | 77.14 / 91.66 |

Paper cũng liệt kê các phương pháp trước đó (ví dụ k-reciprocal Re-ranking: 63.63 mAP / 77.11 Rank-1) và nói DML với hai MobileNet vượt state of the art học sâu lúc đó.

**So sánh với distillation (Bảng 4)** (CIFAR-100 top-1; Market-1501 mAP). Cột: Independent Net1 / Net2; "1 distills 2" = Net 2 sau khi chưng cất từ Net 1 đã train; DML Net1 / Net2:

| Tập | Net 1 | Net 2 | Độc lập 1 / 2 | 1 distills 2 | DML 1 / 2 |
|---|---|---|---|---|---|
| CIFAR-100 | WRN-28-10 | ResNet-32 | 78.69 / 68.99 | 69.48 | 78.96 / 70.73 |
| CIFAR-100 | MobileNet | ResNet-32 | 73.65 / 68.99 | 69.12 | 76.13 / 71.10 |
| Market-1501 | InceptionV1 | MobileNet | 65.26 / 46.07 | 49.11 | 65.34 / 52.87 |
| Market-1501 | MobileNet | MobileNet | 46.07 / 46.07 | 45.16 | 52.95 / 51.26 |

Kết luận paper: KD từ teacher mạnh cũng giúp student (so với độc lập), nhưng DML giúp nhiều hơn; và dùng cùng một mạng cùng cỡ làm teacher cố định thậm chí làm student *tệ hơn* học một mình (45.16 so với 46.07).

**Cohort lớn hơn (Hình 2, Market-1501, MobileNet):** mAP trung bình mỗi mạng tăng theo số mạng trong cohort (từ 1 đến 5), độ lệch chuẩn giảm; ensemble của cohort DML cũng tốt hơn ensemble của các mạng độc lập. Con số cụ thể chỉ có trong hình (không đọc được chính xác từ text trích xuất), nên không ghi ở đây.

## 5. Phân tích / ablation

**Tại sao không bị "mù dắt mù":** (lập luận của tác giả) loss nhãn thật khiến mọi mạng đoán đúng nhãn; khác biệt giữa các mạng nằm ở xác suất các lớp phụ; mạng tập hợp ước lượng tập thể về chúng.

**Cực tiểu rộng hơn (Hình 3, MobileNet trên Market-1501):**
- Training loss của DML và độc lập đều về mức nhỏ như nhau (cả hai khớp hoàn toàn dữ liệu train, train accuracy 100%), nên DML không tìm cực tiểu *sâu* hơn.
- Thêm nhiễu Gauss với độ lệch chuẩn $\sigma$ vào từng tham số: loss của mô hình độc lập tăng vọt, của DML tăng ít hơn nhiều, tức DML nằm ở vùng phẳng hơn (theo lập luận [Chaudhari, Keskar] làm tổng quát hóa tốt hơn; đây là bằng chứng gián tiếp).

**Cơ chế tăng entropy:** vì KL phạt nặng chỗ mạng này gán 0 mà peer gán khác 0, các mạng đều dồn thêm khối lượng xác suất vào các lớp phụ và phủ nhiều lớp phụ khác nhau hơn. Với ResNet-32 trên CIFAR-100, entropy hậu nghiệm trung bình trên tập train là **1.7099** (DML) so với **0.2602** (độc lập). Tác giả liên hệ với "entropy regularisation", nhưng của họ là "mutual probability matching on reasonable alternatives" thay vì thiên vị entropy mù quáng.

**DML_e (ensemble teacher) kém hơn DML (Hình 4, 5 MobileNet trên Market-1501):** ensemble tạo mục tiêu nhọn hơn tại lớp đúng, entropy dự đoán thấp hơn, mất đi chính "tín hiệu lớp phụ" mà DML cần. Ensemble tốt cho dự đoán nhưng không tốt cho tín hiệu dạy.

## 6. Giới hạn

**Tác giả nêu hoặc hàm ý:**
- DML làm các mạng *giống nhau hơn* (giảm đa dạng) nên không nhằm tạo ensemble đa dạng; mục tiêu là nghiệm bền vững.
- Dùng thêm nhiều mạng trong cohort tốn bộ nhớ và tính toán lúc train (họ lập luận mạng nhỏ thì nhét được nhiều mạng vào một GPU).

**Chúng ta nhận thấy:**
- Chỉ ảnh phân loại/re-ID, mạng nhỏ đến vừa, không có ngôn ngữ, không có chuỗi, không có mô hình cực lớn.
- Chi phí train gấp $K$ lần (mỗi bước forward $K$ mạng). So sánh công bằng nên tính theo compute: "1 distills 2" chỉ tốn compute của một student cộng teacher đã có, DML tốn nhiều hơn.
- Không có độ lệch chuẩn nhiều seed trong các bảng chính, và kết quả "DML thắng KD" so với một teacher cụ thể, không tune temperature hay trọng số (formulation không có temperature, $\alpha$ bằng 1).
- Một paper sau (10 Does KD really work) cho thấy nhiều câu chuyện "student bắt chước teacher" thật ra chưa đạt khớp function; DML không đo agreement.
- Không có phân tích lý thuyết hội tụ; giải thích "cực tiểu rộng" chỉ là bằng chứng gián tiếp.

## 7. Liên hệ với các bài khác

- **02 Hinton KD:** DML dùng cùng ý "lớp phụ chứa thông tin" nhưng nguồn tín hiệu là peer cùng train.
- **06 Born-again networks:** BAN là nối tiếp theo thế hệ (tuần tự), DML là song song, đồng thời. Cả hai đều chứng minh teacher không cần mạnh hơn student.
- **08 Teacher assistant, 09 Efficacy of KD, 41 Capacity gap law:** các bài về "teacher quá lớn so với student"; DML là giải pháp lách (peer cùng cỡ).
- **10 Does KD really work:** nhắc rằng KD (kể cả cohort/ensemble) thường không bắt chước đủ trung thực; hữu ích khi đánh giá DML.
- **22 DINO:** teacher/student cùng kiến trúc, nhưng teacher là EMA của student (một dạng self-distillation khác).

## 8. Ứng dụng cho dự án (suy luận)

Phần này là suy luận của chúng ta. Với dự án (student 0.3B/0.1B được cắt từ model 0.8B, teacher 4B cố định), DML **ít liên quan trực tiếp**: teacher 4B đã có và mạnh, còn train hai VLM cùng lúc sẽ ngốn bộ nhớ.

Điểm đáng giữ lại:
1. **Cảnh báo ensemble làm teacher:** bằng chứng của paper là mục tiêu quá nhọn làm mất tín hiệu lớp phụ. Nếu sau này thêm nhiều teacher (4B và 0.8B gốc), hãy thử không lấy trung bình trước khi tính KL (KL riêng từng teacher rồi trung bình, kiểu Eq. 8, thay vì KL đến trung bình, kiểu Eq. 9).
2. **Hai student pruned cùng train:** hai biến thể prune (ví dụ cắt MLP neuron và cắt layer) có thể dạy lẫn nhau trong phase recovery; giả thuyết, chưa có bằng chứng, và tốn bộ nhớ training dù inference vẫn nhẹ.
3. **Đo độ bền (độ nhạy với nhiễu tham số)** như Hình 3(b) là phép đo rẻ để so sánh các student đã prune sau recovery.
4. Liên hệ với quan sát "cắt MLP neuron phục hồi kém hơn cắt layer": có thể thử xem entropy hậu nghiệm của student sau cắt neuron có thấp bất thường (quá tự tin) hay không; chỉ là ý tưởng chẩn đoán, không từ paper.

## 9. Thí nghiệm gợi ý

1. **Peer thêm vào KD:** train hai student 0.3B (khác init hoặc khác cách prune) với loss $= \text{KD}_{\text{teacher 4B}} + \lambda\,\text{KL}(\text{peer}\|\text{student})$ ở token-level, so với KD thuần. Báo CER/WER và entropy trung bình.
2. **Entropy / flatness diagnostic:** với hai student (cắt layer so với cắt neuron) cùng độ nén, đo tăng loss khi thêm nhiễu Gauss $\sigma$ vào tham số (như Hình 3b) để xem recovery có đưa về vùng phẳng không.
3. **Ensemble teacher ablation:** teacher mục tiêu = trung bình xác suất hai teacher so với KL riêng từng teacher; xem entropy của mục tiêu và kết quả như Hình 4.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| DML | Nhóm mạng cùng train, mỗi mạng học nhãn thật + khớp xác suất peer |
| Loss mỗi mạng | $L_{C_k}+\frac{1}{K-1}\sum_{l\ne k}D_{KL}(p_l\|p_k)$ |
| Mimicry loss | Thành phần KL kéo về xác suất peer |
| DML_e | Dùng trung bình peer làm một teacher; kém hơn DML |
| Posterior entropy | Entropy của softmax; DML cao hơn (1.7099 so với 0.2602 ở ResNet-32) |
| Cực tiểu rộng | Loss ít tăng khi nhiễu tham số; gắn với tổng quát hóa tốt |
| Cohort | Nhóm các mạng học cùng nhau |

## 11. Kết luận ngắn

DML cho thấy distillation không đòi hỏi teacher mạnh được train sẵn: một nhóm mạng chưa train, vừa học nhãn vừa khớp xác suất của nhau, đạt kết quả tốt hơn học độc lập và tốt hơn KD từ teacher cố định trong các thí nghiệm CIFAR-100 và Market-1501. Giải thích của paper xoay quanh việc nhóm tập hợp ước lượng về các lớp phụ, tăng entropy hậu nghiệm, và đẩy mạng vào cực tiểu phẳng. Với dự án (teacher 4B cố định, nén mạnh), giá trị chính là các ý chẩn đoán (entropy, độ nhạy nhiễu) và cảnh báo về ensemble-target.

Nguồn gốc: https://arxiv.org/abs/1706.00384
