# Learning both Weights and Connections for Efficient Neural Networks (2015)

**Tác giả:** Song Han, Jeff Pool, John Tran, William J. Dally  
**Venue / arXiv:** arXiv:1506.02626 (bản v3, 30/10/2015). Bản arXiv không ghi venue; bài của Li et al. (04) trích dẫn là NIPS 2015.  
**Nguồn:** https://arxiv.org/abs/1506.02626  
**Chủ đề:** Pruning trọng số theo độ lớn (magnitude), quy trình train → prune → retrain, lặp nhiều vòng; nén AlexNet 9× và VGG-16 13×.  
**Vị trí trong lộ trình:** Đọc sau 01 (OBD/OBS: tiêu chí bậc hai). Bài này là **baseline magnitude pruning kinh điển** mà gần như mọi bài sau so sánh. Tiếp theo: 03 (cắt dần trong lúc train, so sánh với mô hình nhỏ dense), 04 (cắt filter có cấu trúc để có tốc độ thật).

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch. Số liệu lấy từ bản arXiv đã đọc. Các phần "Ứng dụng cho dự án" là suy luận của chúng ta, không phải kết luận của paper.

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

Đến 2012–2015 mạng tích chập lớn rất nhanh: LeNet-5 dưới 1M tham số (1998), AlexNet 60M (2012), Deepface 120M, mạng của Coates và cộng sự 10B. Paper lập luận chi phí chính của mạng lớn **không phải phép nhân mà là truy cập bộ nhớ**. Theo bảng năng lượng 45nm CMOS (Horowitz) trong paper:

| Phép toán | Năng lượng (pJ) | Tương đối |
|---|---|---|
| 32-bit int ADD | 0.1 | 1 |
| 32-bit float ADD | 0.9 | 9 |
| 32-bit int MULT | 3.1 | 31 |
| 32-bit float MULT | 3.7 | 37 |
| 32-bit SRAM (cache) | 5 | 50 |
| 32-bit DRAM | 640 | 6400 |

Ví dụ trong paper: mạng $10^9$ kết nối chạy 20 Hz cần $(20)(10^9)(640\,\text{pJ})=12.8$ W chỉ cho DRAM, vượt khả năng thiết bị di động. Mục tiêu: nhét trọng số vào bộ nhớ trên chip (SRAM) bằng cách bớt kết nối.

Trước đó đã có hướng nén khác (độc lập, có thể kết hợp): lượng tử hóa số nguyên 8 bit (Vanhoucke), xấp xỉ hạng thấp (Denton), vector quantization (Gong), HashedNets. Về pruning, OBD/OBS (01) dùng Hessian nhưng "second-order derivative needs additional computation". Han et al. chọn một tiêu chí **rẻ**: độ lớn trọng số.

**Thuật ngữ nhanh**
- **Pruning kết nối (connection pruning):** đặt các trọng số về 0 và bỏ chúng; ở đây mỗi trọng số là một "synapse".
- **Sparse (thưa):** tỉ lệ lớn các trọng số bằng 0.
- **Fully connected (FC):** tầng nối đầy đủ, chứa phần lớn tham số của AlexNet/VGG.

## 2. Ý tưởng chính

Huấn luyện mạng bình thường để biết *kết nối nào quan trọng*, rồi **bỏ mọi kết nối có |trọng số| dưới một ngưỡng**, rồi **huấn luyện lại phần còn lại** để chúng bù đắp. Có thể lặp "cắt – retrain" nhiều vòng. Quy trình này làm cho việc huấn luyện học **cả cấu trúc kết nối lẫn giá trị trọng số**, tương tự quá trình cắt tỉa synapse ở não trẻ em. Kết quả: AlexNet giảm 9× tham số (61M → 6.7M) và VGG-16 giảm 13× (138M → 10.3M) mà không mất độ chính xác. Bước retrain là then chốt: không có nó độ chính xác tụt rõ.

---

## 3. Phương pháp chi tiết

### 3.1 Quy trình ba bước

```
1. Train connectivity : huấn luyện mạng đầy đủ bình thường (học xem kết nối nào quan trọng)
2. Prune connections  : với mỗi tầng, loại mọi trọng số có |w| < ngưỡng (đặt mask = 0)
3. Train weights      : retrain mạng thưa, giữ nguyên các trọng số sống sót làm khởi tạo
(lặp 2–3 nhiều vòng = iterative pruning)
```

Hiện thực trong Caffe: mỗi tensor trọng số có một **mask** nhị phân; trọng số bị mask bị bỏ qua khi forward/backward. **Ngưỡng của mỗi tầng = một "tham số chất lượng" nhân với độ lệch chuẩn của trọng số tầng đó.**

$$
\text{mask}_{ij}=\mathbb{1}\big[\,|w_{ij}|\ \ge\ \tau_\ell\,\big],\qquad \tau_\ell=q\cdot\mathrm{std}(W_\ell).
$$

(Ký hiệu $q$, $\tau_\ell$ là cách viết của ghi chú này; paper chỉ mô tả bằng lời.)

### 3.2 Regularization: L1 hay L2?

- **L1** đẩy nhiều tham số về gần 0 → độ chính xác **ngay sau khi cắt, chưa retrain** tốt hơn. Nhưng các kết nối còn lại *kém hơn* sau retrain.
- **L2** cho kết quả cuối **tốt nhất sau retrain**. Paper kết luận L2 tổng thể tốt nhất. Thử "L1 để cắt, L2 để retrain" không vượt L2 cả hai pha: tham số từ chế độ này không thích nghi tốt với chế độ kia.

### 3.3 Điều chỉnh dropout khi retrain

Dropout tắt tham số ngẫu nhiên nhưng chúng quay lại lúc suy luận; pruning thì loại vĩnh viễn → dung lượng giảm → **tỉ lệ dropout lúc retrain nên giảm**. Gọi $C_i$ là số kết nối tầng $i$, $N_i$ số neuron ($C_i=N_iN_{i-1}$), $C_{io}$ và $C_{ir}$ trước / sau cắt:

$$
\boxed{\ D_r=D_o\sqrt{\frac{C_{ir}}{C_{io}}}\ }
$$

Lý do (theo paper): dropout hoạt động trên neuron, $C_i$ tỉ lệ bình phương với $N_i$, nên tỉ lệ dropout theo căn.

### 3.4 Giữ trọng số cũ khi retrain (co-adaptation) và retrain cục bộ

Giữ giá trị trọng số sống sót tốt hơn khởi tạo lại: CNN có các đặc trưng "co-adapted mong manh" (Yosinski et al.). Paper còn **khóa các tầng CONV khi retrain FC và ngược lại**, vì mạng sâu dễ gặp vanishing gradient khiến sai sót cắt khó phục hồi, và đỡ tốn chi phí lan truyền ngược qua toàn mạng.

### 3.5 Iterative pruning

Một vòng cắt + retrain = một lần lặp, mỗi lần là tìm kiếm tham lam. **Nâng tỉ lệ cắt từ 5× lên 9× trên AlexNet không mất độ chính xác** so với cắt mạnh một lần. Cắt theo xác suất dựa trên |w| được thử nhưng cho kết quả tệ hơn.

### 3.6 Cắt neuron là hệ quả tự nhiên

Sau khi cắt kết nối, neuron không còn kết nối vào hoặc ra có thể bỏ; retrain tự dẫn đến neuron "chết" có cả hai phía bằng 0, nhờ gradient bằng 0 và regularization kéo trọng số còn lại về 0. Theo chúng ta, đây là cách duy nhất phương pháp này có được cấu trúc, và nó không được điều khiển chủ động (nhận xét của ghi chú, không phải của paper).

### 3.7 Độ nhạy theo tầng

Paper cắt riêng từng tầng để đo độ nhạy và đặt ngưỡng theo đó: tầng CONV nhạy hơn FC; **tầng conv đầu tiên nhạy nhất** (suy đoán của tác giả: chỉ có 3 kênh vào nên ít dư thừa). Ngưỡng nhỏ nhất áp cho tầng nhạy nhất.

---

## 4. Thí nghiệm và kết quả chính

Bốn mạng: LeNet-300-100 và LeNet-5 (MNIST), AlexNet và VGG-16 (ImageNet ILSVRC-2012). Mô hình tham chiếu lấy từ Caffe model zoo, độ chính xác đo không augmentation. GPU: Titan X và GTX980.

**Bảng 1 (tóm tắt):**

| Mạng | Top-1 err | Top-5 err | Tham số | Tỉ lệ nén |
|---|---|---|---|---|
| LeNet-300-100 ref | 1.64% | - | 267K | |
| LeNet-300-100 pruned | 1.59% | - | 22K | 12× |
| LeNet-5 ref | 0.80% | - | 431K | |
| LeNet-5 pruned | 0.77% | - | 36K | 12× |
| AlexNet ref | 42.78% | 19.73% | 61M | |
| AlexNet pruned | 42.77% | 19.67% | 6.7M | 9× |
| VGG-16 ref | 31.50% | 11.32% | 138M | |
| VGG-16 pruned | 31.34% | 10.88% | 10.3M | 13× |

**LeNet.** Sau cắt, retrain với 1/10 learning rate gốc. LeNet-300-100: trọng số còn 8%, FLOP còn 8% (giảm 12×). LeNet-5: trọng số 8%, FLOP thực còn 16% (tính cả activation thưa; giảm 6×). Một quan sát thú vị: sparsity pattern của lớp FC1 LeNet-300-100 (784×300) có **28 dải** tương ứng 28×28 pixel, cho thấy chỉ trọng số ở vùng giữa ảnh còn lại (mạng tự tìm "vùng chú ý").

**AlexNet.** 61M tham số, 1.5B FLOP; còn **11% trọng số, 30% FLOP** (3× ít tính toán). Retrain với **1/100 learning rate khởi điểm**. Thời gian: huấn luyện gốc 75 giờ trên Titan X, **retrain 173 giờ**. Phần trăm trọng số giữ theo tầng (Bảng 4): conv1 84%, conv2 38%, conv3 35%, conv4 37%, conv5 37%, fc1 9%, fc2 9%, fc3 25%. Paper lập luận retrain lâu ít quan trọng vì pruning là bước cuối trước triển khai.

**VGG-16.** 138M tham số, 30.9B FLOP; còn **7.5% trọng số, 21% FLOP**, dùng 5 vòng cắt–retrain. Hai tầng FC lớn nhất (fc6 103M, fc7 17M) được cắt còn khoảng **4%** trọng số. Điều này quan trọng cho xử lý ảnh thời gian thực khi FC ít được tái sử dụng giữa các ảnh (không như xử lý theo lô lúc huấn luyện).

**So sánh với phương pháp khác trên AlexNet (Bảng 6):**

| Phương pháp | Top-1 err | Tham số | Nén |
|---|---|---|---|
| Baseline Caffe | 42.78% | 61.0M | 1× |
| Data-free pruning | 44.40% | 39.6M | 1.5× |
| Fastfood-32-AD | 41.93% | 32.8M | 2× |
| Fastfood-16-AD | 42.90% | 16.4M | 3.7× |
| Collins & Kohli | 44.40% | 15.2M | 4× |
| Naive cut (giảm kích thước tầng) | 47.18% | 13.8M | 4.4× |
| SVD (Denton et al.) | 44.02% | 11.9M | 5× |
| **Network pruning** | **42.77%** | **6.7M** | **9×** |

**Lưu trữ thưa:** chi phí phụ chỉ khoảng **15.6%**; dùng chỉ số tương đối nên chỉ số FC chỉ cần 5 bit, chỉ số CONV 8 bit.

---

## 5. Phân tích / ablation

Hình 5 (đường cong: tổn thất top-5 theo % tham số bị cắt, năm cấu hình L1/L2, có/không retrain, lặp):

- **Retrain quan trọng:** không retrain, độ chính xác bắt đầu tụt khi còn khoảng **1/3** số kết nối; có retrain thì mãi đến khi còn khoảng **1/10**. Không retrain vẫn có "bữa trưa miễn phí" **2×** ít kết nối mà không mất độ chính xác; có retrain là 9×.
- **Iterative pruning cho lợi ích lớn nhất:** không mất độ chính xác ở 9×; từ 10× độ chính xác bắt đầu tụt mạnh.
- **L1 vs L2:** L1 tốt hơn lúc chưa retrain, L2 tốt hơn sau retrain (mục 3.2).
- Hai điểm trên đường cong thậm chí **tốt hơn mô hình gốc một chút**; tác giả cho rằng pruning tìm được dung lượng phù hợp và giảm overfitting (đây là giả thuyết của họ, không kiểm chứng riêng).
- **Phân phối trọng số (Hình 7, FC1 của AlexNet):** trước cắt tập trung quanh 0 (hầu hết trong [−0.015, 0.015]); sau cắt và retrain thành **phân phối hai đỉnh** trải rộng hơn ([−0.025, 0.025]).
- **Độ nhạy theo tầng (Hình 6):** CONV nhạy hơn FC, conv1 nhạy nhất.

## 6. Giới hạn

**Do tác giả nêu:**
- Phần cứng đa dụng kém trong tính toán thưa; họ nhắm tới **phần cứng chuyên dụng cho DNN thưa** ("fixed-function hardware specialized for sparse DNN").
- Retrain tốn thời gian (173 giờ cho AlexNet), nhưng họ coi là chấp nhận được.
- Sparse storage tốn thêm chỉ số (15.6%).
- Độ giảm FLOP thấp hơn độ giảm tham số (3× vs 9× với AlexNet) vì FC chiếm phần lớn tham số nhưng ít FLOP.

**Do chúng ta nhận xét (suy luận):**
- Cắt **không cấu trúc**: kết quả là ma trận cùng kích thước chứa nhiều 0. Trên GPU đa dụng, không tự nhanh hơn (xem preface mục 7; và 04 phản biện đúng điểm này). Paper không báo cáo thời gian suy luận thật.
- Chỉ đo trên CNN phân loại ảnh, 2015; chưa có BatchNorm/residual; kết luận "tầng FC dư thừa nhiều" phản ánh đặc điểm AlexNet/VGG chứ không phải mọi mạng (xem 03 với MobileNet, 04 với ResNet).
- So sánh baseline "naive cut" là một mạng giảm tầng nhưng có thể không được tối ưu hóa siêu tham số cùng mức (về sau 03 và 11, 12 xem lại vấn đề này).
- Cách chọn ngưỡng theo std và độ nhạy từng tầng là thủ công, không có cơ chế tự động.
- Số ngày huấn luyện không so sánh công bằng với các phương pháp khác.

## 7. Liên hệ với các bài khác

- **01 OBD/OBS:** tiêu chí bậc hai chính xác hơn nhưng đắt; Han chọn magnitude. Bài này kiểm chứng "magnitude + retrain đủ tốt" ở quy mô lớn.
- **03 Zhu & Gupta:** thay lịch "cắt một lần rồi retrain" bằng **lịch cắt dần trong lúc train**, và hỏi câu hỏi mà Han chưa hỏi: mô hình lớn thưa có hơn mô hình nhỏ dense cùng bộ nhớ không?
- **04 Li et al.:** chỉ ra rằng cắt trọng số theo Han giảm tham số ở FC nhưng không giảm tính toán conv và cần thư viện thưa → chuyển sang cắt filter.
- **05 Network Slimming:** cùng bàn về nhược điểm cần phần cứng/thư viện đặc biệt của weight pruning.
- **08 Lottery Ticket, 11, 12:** các bài sau dùng magnitude pruning kiểu này làm nền; 11 nghi ngờ vai trò của trọng số kế thừa; 12 tổng hợp việc so sánh.
- **18 (2:4 sparsity)**: phần cứng NVIDIA hỗ trợ một dạng thưa đều đặn, giải quyết một phần mơ ước "phần cứng cho DNN thưa" ở đây.
- **24 Wanda:** magnitude có điều chỉnh bằng activation; ngược dòng của nó là tiêu chí trong bài này.

## 8. Ứng dụng cho dự án (suy luận)

- **Bài này gần như không áp dụng trực tiếp** cho mục tiêu GPU ≤ 8 GB với độ trễ batch-1: pruning không cấu trúc của Han không giảm kích thước ma trận, nên không giảm byte đọc ở decode, trừ khi dùng định dạng thưa + kernel chuyên dụng. Mục tiêu 0.8B → 0.3B/0.1B cần cắt có cấu trúc.
- **Điều đáng giữ lại:**
  1. Quy trình *train → prune → retrain (lặp)* và nguyên tắc **giữ lại trọng số sống sót** làm khởi tạo thay vì khởi tạo lại.
  2. **Độ nhạy theo tầng**: cắt riêng từng tầng để biết tầng nào nhạy → áp dụng cho từng block của ViT 12 block / từng layer decoder (kết hợp preface mục 4.7).
  3. Quan sát: **learning rate khi retrain nhỏ** (1/10–1/100) so với lúc pretrain.
  4. Tinh thần "đo bộ nhớ ở mức byte" thay vì FLOPs.
- **Magnitude trên ma trận hai chiều không cắt neuron**: ngay cả khi một số neuron MLP tự chết sau retrain (mục 3.6), điều đó không được điều khiển; cho dự án, hãy cắt neuron theo điểm tổng hợp trên cả hàng $W_{in}$ và cột $W_{out}$ (xem 04, 26, 28).

## 9. Thí nghiệm gợi ý

1. **Đo độ nhạy từng tầng**: cắt riêng mỗi block (hoặc 20/40/60% neuron của mỗi MLP) rồi đo loss KD hay CER trên tập OCR, vẽ đường cong như Hình 6 để chọn tỉ lệ cắt khác nhau theo block (và xem block nào "conv1-giống" — nhạy bất thường).
2. **Magnitude vs random vs "activation-aware" cho MLP neuron**: cắt 30% neuron theo chuẩn $\ell_1$ của hàng $W_{in}$, ngẫu nhiên, và bằng $|w|\cdot\|x\|$; so loss trước và sau 500 bước fine-tune. Kiểm chứng xem tiêu chí đơn giản có đủ cho cắt có cấu trúc không.
3. **Iterative vs one-shot ở cùng tổng ngân sách bước**: cắt 40% một lần rồi train $N$ bước vs 4 vòng 10% + $N/4$ bước mỗi vòng. Phản chiếu mục 3.5 và đo xem lợi thế iterative có còn khi cắt cấu trúc.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung | Vị trí |
|---|---|---|
| Quy trình | train connectivity → prune → train weights (lặp) | Mục 3 |
| Mask | $M_{ij}=\mathbb 1[\,|w_{ij}|\ge\tau_\ell\,]$ | Mục 4 |
| Ngưỡng tầng | $\tau_\ell=q\cdot\mathrm{std}(W_\ell)$ | Mục 4 |
| Điều chỉnh dropout | $D_r=D_o\sqrt{C_{ir}/C_{io}}$ | Eq. 2 |
| Số kết nối | $C_i=N_iN_{i-1}$ | Eq. 1 |
| Kết quả | AlexNet 61M→6.7M (9×), VGG-16 138M→10.3M (13×) | Bảng 1 |
| FLOP giảm | AlexNet 3×, VGG-16 5×, LeNet-5 6×, LeNet-300-100 12× | Bảng 2–5 |
| Chi phí lưu trữ thưa | 15.6% overhead; 5 bit (FC) / 8 bit (CONV) chỉ số | Mục 5 |

## 11. Kết luận ngắn

Han et al. cho thấy một công thức đơn giản — train, cắt trọng số nhỏ, retrain, lặp — đủ nén AlexNet/VGG 9–13× không mất độ chính xác, và đặt ra các chuẩn mực còn dùng đến nay: ngưỡng theo độ nhạy tầng, giữ trọng số cũ, retrain với learning rate nhỏ, iterative tốt hơn one-shot. Điểm yếu lớn là kết quả thưa không cấu trúc, cần phần cứng chuyên dụng mới thành tốc độ; các bài 04–05 trả lời điều đó.

Nguồn: https://arxiv.org/abs/1506.02626
