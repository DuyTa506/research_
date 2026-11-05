# Learning Efficient Convolutional Networks through Network Slimming (2017)

**Tác giả:** Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, Changshui Zhang  
**Venue / arXiv:** arXiv:1708.06519 (v1, 22/8/2017). Bản arXiv không ghi venue trong phần văn bản đã đọc.  
**Nguồn:** https://arxiv.org/abs/1708.06519  
**Chủ đề:** Cắt channel có cấu trúc bằng phạt $\ell_1$ lên hệ số scale $\gamma$ của BatchNorm, ngưỡng toàn cục, multi-pass.  
**Vị trí trong lộ trình:** Đọc sau 04 (Li et al.: chọn filter theo $\ell_1$ của trọng số, sau khi train). Bài này **đưa tiêu chí vào chính hàm mục tiêu huấn luyện**: mạng học luôn "kênh nào cần giữ". Tiếp theo: 06 ThiNet, 07 Molchanov, 16 Movement pruning, 17 CoFi (mặt nạ học được cho transformer; cùng họ "phạt thưa lên cổng").

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch. Số liệu lấy từ bản arXiv đã đọc (một số bảng bị xáo cột trong PDF, chỉ trích khi đọc rõ). Phần "Ứng dụng cho dự án" là suy luận của chúng ta.

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

Paper nêu ba nguồn chi phí khi triển khai CNN lớn:
1. **Kích thước mô hình** (CNN ImageNet điển hình trên 300 MB).
2. **Bộ nhớ lúc chạy**: activation trung gian có thể tốn *hơn cả* trọng số, ngay cả với batch 1.
3. **Số phép tính**: tích chập nặng trên ảnh độ phân giải cao.

Hầu hết các kỹ thuật nén chỉ giải quyết một–hai điều; nhiều kỹ thuật cần phần cứng/thư viện đặc biệt (lượng tử hóa nhị phân, weight pruning thưa). Điểm quan trọng: **cắt trọng số (02) gần như không giảm bộ nhớ runtime** vì phần lớn bộ nhớ là activation (vẫn dense). Muốn giảm cả ba thì phải giảm *số channel*.

Khó khăn của channel pruning (paper gọi là *Challenges*): cắt channel nghĩa là cắt mọi kết nối vào và ra khỏi nó; nên cắt trực tiếp trên mô hình đã train rất khó vì hiếm khi toàn bộ trọng số tại hai đầu của một channel đều xấp xỉ 0. Paper dẫn rằng ở [Li et al. = bài 04], cắt channel trên ResNet đã train chỉ giảm khoảng 10% tham số mà không mất accuracy. Hướng khác: Structured Sparsity Learning (SSL, Wen et al.) dùng group LASSO trên trọng số filter, nhưng cần tính gradient của số hạng phạt theo mọi trọng số filter, "không đơn giản".

**Thuật ngữ:**
- **BatchNorm (BN):** chuẩn hóa activation theo mini-batch rồi biến đổi affine bằng $\gamma,\beta$ học được.
- **Hệ số scale $\gamma$:** nhân vào mỗi channel sau chuẩn hóa. Nếu $\gamma_c\approx0$ thì channel $c$ tạo ra gần như hằng số $\beta_c$.
- **$\ell_1$ regularization:** thêm $\lambda\sum|\gamma|$ vào loss để đẩy nhiều giá trị về đúng 0.

## 2. Ý tưởng chính

Gán cho mỗi channel một hệ số scale $\gamma$ và dùng luôn **$\gamma$ có sẵn trong BN** (không thêm tham số). Huấn luyện chung trọng số và $\gamma$ với phạt $\ell_1$ lên $\gamma$; sau đó nhiều $\gamma$ gần 0 — các kênh đó là kênh không quan trọng. Cắt các kênh có $\gamma$ nhỏ theo **một ngưỡng chung toàn mạng** (ví dụ phân vị 70%), tạo ra mô hình hẹp hơn nhưng dense, rồi fine-tune. Có thể lặp lại cả quy trình (multi-pass) để thu nhỏ thêm. Không cần thư viện thưa hay phần cứng đặc biệt.

---

## 3. Phương pháp chi tiết

### 3.1 Hàm mục tiêu (Eq. 1)

$$
\boxed{\ L=\sum_{(x,y)}\ell\big(f(x,W),y\big)+\lambda\sum_{\gamma\in\Gamma}g(\gamma),\qquad g(s)=|s|\ }
$$

$W$ là trọng số, $\Gamma$ là tập mọi hệ số scale; $\lambda$ cân bằng hai số hạng. Tối ưu bằng **subgradient descent** cho phạt $\ell_1$ không trơn; paper nói có thể thay bằng smooth-$\ell_1$ (Schmidt et al.) để tránh điểm không khả vi. Gradient của phạt đối với $\gamma$: $\lambda\,\mathrm{sign}(\gamma)$.

### 3.2 Vì sao dùng $\gamma$ của BN

BN (Eq. 2):

$$
\hat z=\frac{z_{in}-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad z_{out}=\gamma\hat z+\beta .
$$

Paper lập luận $\gamma$ trong BN là cách hiệu quả nhất để có hệ số kênh có ý nghĩa, vì thêm một lớp scale riêng thì không ổn:
1. Nếu chèn scale vào CNN không có BN, giá trị scale không có ý nghĩa đánh giá độ quan trọng (conv và scale đều tuyến tính, có thể giảm scale mà tăng trọng số, cùng kết quả).
2. Chèn scale *trước* BN thì tác dụng bị chính BN triệt tiêu.
3. Chèn *sau* BN thì mỗi kênh có hai hệ số scale liên tiếp (thừa).

Dùng $\gamma$ sẵn có: **không thêm chi phí** nào.

### 3.3 Cắt channel và fine-tune

- Sau khi huấn luyện với phạt, nhiều $\gamma$ gần 0 (Hình 1, Hình 4).
- Cắt **kênh có $\gamma$ nhỏ**, bằng **ngưỡng toàn cục**: lấy phân vị của toàn bộ $\gamma$ (ví dụ cắt 70% kênh có $\gamma$ thấp nhất). Khác 04 (tỉ lệ cắt khác nhau theo tầng).
- Cắt = xây mô hình hẹp mới và sao chép các trọng số tương ứng từ mô hình đã train với phạt (không cần thư viện thưa).
- **Fine-tune** mô hình hẹp để phục hồi. Paper nhấn mạnh sau fine-tune mô hình hẹp thường *bằng hoặc tốt hơn* mô hình gốc.

```
Network slimming
1. Khởi tạo gamma = 0.5 (mọi kênh)
2. Huấn luyện với L = Loss + λ Σ|γ|     (subgradient / SGD)
3. Chọn ngưỡng = phân vị p% của mọi |γ| trong toàn mạng
4. Xóa kênh có γ < ngưỡng (xóa filter vào, kernel ra, tham số BN tương ứng)
5. Fine-tune mô hình hẹp
6. (Multi-pass) Lặp 2–5 trên mô hình hẹp
```

### 3.4 Kết nối chéo tầng và cấu trúc pre-activation (ResNet, DenseNet)

Trong mạng có BN **trước** conv (pre-activation) và nhiều tầng đọc cùng đầu ra, sparsity xuất hiện ở **đầu vào** của tầng: tầng đó chỉ dùng một tập con kênh nhận được. Để thu được lợi ích tham số/tính toán khi suy luận, chèn một **channel selection layer** để che các kênh không quan trọng.

### 3.5 Multi-pass

Sau một vòng (huấn luyện có phạt → cắt → fine-tune) ta có mạng hẹp, và có thể áp lại toàn bộ quy trình để được mạng gọn hơn nữa (nét đứt trong Hình 2).

---

## 4. Thí nghiệm và kết quả chính

### 4.1 Thiết lập

- **Dữ liệu:** CIFAR-10/100 (32×32; validation 5000 ảnh tách khỏi train để tìm $\lambda$ trên CIFAR-10), SVHN (604 388 ảnh train, 6000 để validation; test 26 032), ImageNet (1.2M train / 50k val), MNIST.
- **Mạng:** VGGNet (biến thể CIFAR), ResNet-164 (pre-activation bottleneck), DenseNet-40 (growth rate 12); ImageNet: VGG-A 11 tầng (8 conv + 3 FC) có BN, bỏ dropout; MNIST: mạng FC 3 tầng như SSL.
- **Huấn luyện:** SGD; CIFAR: batch 64, 160 epoch (SVHN: 20); ImageNet và MNIST: 60 và 30 epoch, batch 256; LR 0.1 giảm 10× tại 50% và 75% (CIFAR/SVHN) hoặc 1/3 và 2/3 (ImageNet/MNIST); weight decay $10^{-4}$, Nesterov 0.9. **Khởi tạo mọi $\gamma$ = 0.5** (cho baseline tốt hơn mặc định 1).
- **$\lambda$:** grid search $10^{-3},10^{-4},10^{-5}$ trên validation CIFAR-10. VGGNet $\lambda=10^{-4}$; ResNet và DenseNet $\lambda=10^{-5}$; VGG-A ImageNet $\lambda=10^{-5}$.
- **Fine-tune:** CIFAR/SVHN/MNIST dùng cùng thiết lập huấn luyện; ImageNet chỉ **5 epoch với LR $10^{-3}$** (vì giới hạn thời gian).

### 4.2 CIFAR và SVHN (Bảng 1)

| Dataset | Mô hình | Lỗi test (%) | Tham số | Cắt tham số | FLOPs | Cắt FLOPs |
|---|---|---|---|---|---|---|
| CIFAR-10 | VGGNet baseline | 6.34 | 20.04M | - | 7.97×10^8 | - |
| | VGGNet 70% pruned | 6.20 | 2.30M | 88.5% | 3.91×10^8 | 51.0% |
| | DenseNet-40 baseline | 6.11 | 1.02M | - | 5.33×10^8 | - |
| | DenseNet-40 40% | 5.19 | 0.66M | 35.7% | 3.81×10^8 | 28.4% |
| | DenseNet-40 70% | 5.65 | 0.35M | 65.2% | 2.40×10^8 | 55.0% |
| | ResNet-164 baseline | 5.42 | 1.70M | - | 4.99×10^8 | - |
| | ResNet-164 40% | 5.08 | 1.44M | 14.9% | 3.81×10^8 | 23.7% |
| | ResNet-164 60% | 5.27 | 1.10M | 35.2% | 2.75×10^8 | 44.9% |
| CIFAR-100 | VGGNet baseline | 26.74 | 20.08M | - | 7.97×10^8 | - |
| | VGGNet 50% | 26.52 | 5.00M | 75.1% | 5.01×10^8 | 37.1% |
| | DenseNet-40 baseline / 40% / 60% | 25.36 / 25.28 / 25.72 | 1.06M / 0.66M / 0.46M | 37.5% / 54.6% | | 30.3% / 47.1% |
| | ResNet-164 baseline / 40% / 60% | 23.37 / 22.87 / 23.91 | 1.73M / 1.46M / 1.21M | 15.5% / 29.7% | | 33.3% / 50.6% |
| SVHN | VGGNet baseline / 60% | 2.17 / 2.06 | 20.04M / 3.04M | 84.8% | 7.97×10^8 / 3.98×10^8 | 50.1% |
| | DenseNet-40 baseline / 40% / 60% | 1.89 / 1.79 / 1.81 | 1.02M / 0.65M / 0.44M | 36.3% / 56.6% | | 30.8% / 49.8% |
| | ResNet-164 baseline / 40% / 60% | 1.78 / 1.85 / 1.81 | 1.70M / 1.46M / 1.12M | 14.5% / 34.3% | | 31.1% / 54.9% |

Nhận xét của paper: tham số giảm tới ~10× (VGGNet), FLOPs giảm thường khoảng 50%; ResNet-164 tiết kiệm ít hơn, tác giả đoán do cấu trúc bottleneck vốn đã "chọn kênh"; CIFAR-100 giảm ít hơn CIFAR-10/SVHN, có thể vì nhiều lớp hơn. Cắt vừa phải (40%) thường làm lỗi test thấp hơn baseline: ví dụ DenseNet-40 cắt 40% đạt **5.19%** (baseline 6.11%, gần 1 điểm), được tác giả giải thích là hiệu ứng điều chuẩn của $\ell_1$ lên kênh (một dạng chọn đặc trưng trung gian) — đây là **giả thuyết**, không kiểm chứng riêng.

### 4.3 ImageNet (Bảng 2) và MNIST (Bảng 3)

| ImageNet VGG-A | Baseline | 50% pruned |
|---|---|---|
| Tham số | 132.9M | 23.2M (cắt 82.5%) |
| FLOPs | 4.57×10^10 | 3.18×10^10 (cắt 30.4%) |
| Lỗi validation (%) | 36.69 | 36.66 |

FLOPs chỉ giảm 30.4% vì chỉ **378 trên 2752 kênh** ở các tầng conv nặng tính toán bị cắt, trong khi **5094 trên 8192 neuron** của tầng FC nặng tham số bị cắt. Paper nhấn mạnh không mất accuracy trên ImageNet 1000 lớp, trong khi các phương pháp hiệu quả khác thường báo mất.

| MNIST (FC 3 tầng) | Lỗi test | Tham số cắt | Neuron |
|---|---|---|---|
| Baseline | 1.43% | - | 784-500-300-10 |
| SSL (Wen et al.) | 1.53% | 83.5% | 434-174-78-10 |
| Slimming | 1.49% | 84.4% | 784-100-60-10 |

Với MNIST, ngưỡng toàn cục *đôi khi xóa hẳn một tầng*, nên họ cắt 80% neuron mỗi tầng ẩn.

### 4.4 Multi-pass (Bảng 4, VGGNet)

CIFAR-10 (baseline lỗi 6.34%): lỗi "trained" (sau huấn luyện có phạt) / "fine-tuned" (sau cắt) theo vòng:

| Vòng | Trained | Fine-tuned | Tham số cắt | FLOPs cắt |
|---|---|---|---|---|
| 1 | 6.38 | 6.51 | 66.7% | 38.6% |
| 2 | 6.23 | 6.11 | 84.7% | 52.7% |
| 3 | 5.87 | 6.10 | 91.4% | 63.1% |
| 4 | 6.19 | 6.59 | 95.6% | 77.2% |
| 5 | 5.96 | 7.73 | 98.3% | 88.7% |
| 6 | 7.79 | 9.70 | 99.4% | 95.7% |

(Trong bảng gốc, tỉ lệ cắt ở một hàng tương ứng với mô hình fine-tuned của hàng đó và mô hình trained của hàng kế tiếp. Văn bản paper xác nhận: ở vòng 5 mô hình huấn luyện đạt lỗi thấp nhất và đạt **20× giảm tham số và 5× giảm FLOPs** với lỗi thấp hơn baseline.) CIFAR-100 (baseline 26.74%): lỗi bắt đầu tăng sau vòng 3; tuy vậy vẫn cắt gần 90% tham số và gần 70% FLOPs mà không mất accuracy đáng kể. Với multi-pass VGGNet, họ thêm ràng buộc **tối đa 50% kênh mỗi tầng bị cắt mỗi vòng** vì mạng không có skip nên cắt hết một tầng sẽ phá mô hình.

### 4.5 Thời gian thực và bộ nhớ (Phụ lục B)

VGGNet "70% pruned" trên CIFAR-10, GPU GeForce 1080, Torch, batch 64:

| | Thời gian/iter | Bộ nhớ | Lỗi test |
|---|---|---|---|
| Baseline | 0.009 s | 697 MB | 6.34 |
| 70% pruned | 0.005 s | 499 MB | 6.20 |

Thời gian giảm "xấp xỉ khớp" mức giảm FLOP; giảm bộ nhớ thấp hơn vì các tầng sâu (nhiều kênh bị cắt) có activation nhỏ, còn tầng nông (activation lớn) ít bị cắt (Phụ lục A, Bảng 5: tầng sâu bị cắt tới 93.8% kênh).

### 4.6 So sánh trực tiếp với 04 (Phụ lục C, Bảng 7)

Cùng tiêu chí và chiến lược cắt theo tầng của [Li et al.]:

| | Lỗi test | Cắt tham số |
|---|---|---|
| CIFAR-10 baseline | 6.34 | - |
| Pruned (Li et al.) | 6.88 | 88.5% |
| Pruned (slimming) | 6.20 | 88.5% |
| CIFAR-100 baseline | 26.74 | - |
| Pruned (Li et al.) | 28.36 | 76.0% |
| Pruned (slimming) | 26.52 | 75.1% |

---

## 5. Phân tích / ablation

- **Tỉ lệ cắt $t$ (Hình 5, DenseNet-40 CIFAR-10, $\lambda=10^{-5}$):** lỗi của mô hình đã fine-tune chỉ kém baseline khi tỉ lệ cắt vượt một ngưỡng, **khoảng trên 80%**. Đáng chú ý: *ngay cả mô hình huấn luyện có phạt, chưa fine-tune*, đã tốt hơn mô hình gốc (hiệu ứng điều chuẩn).
- **$\lambda$ và phân bố $\gamma$ (Hình 4, VGGNet):** $\lambda=0$ cho phân bố tương đối phẳng; $\lambda=10^{-5}$ tập trung gần 0 hơn; $\lambda=10^{-4}$ đẩy gần như toàn bộ $\gamma$ vào vùng sát 0. Paper coi quá trình này là chọn đặc trưng (feature selection) ở tầng trung gian.
- **Động học (Hình 6):** heatmap $\gamma$ qua epoch của tầng conv 11 trong VGGNet: các kênh bắt đầu bằng nhau, sau đó một số sáng lên ("được chọn"), số khác tối đi.
- **Cấu trúc nén thu được (Phụ lục A):** tầng sâu bị cắt nhiều hơn. Trong VGGNet nén ở vòng 5: tầng 9–15 chỉ còn 32 kênh trên 512 (cắt 93.8%), tổng 5504 → 1034 kênh (cắt 81.2%), tham số/FLOP cắt 95.6%/77.2%, lỗi 5.96%.
- **Ngưỡng toàn cục vs theo tầng:** dùng ngưỡng toàn cục "cho đơn giản"; trên MNIST gây xóa nguyên tầng nên phải đổi sang cắt cố định 80% mỗi tầng. Đây là điểm yếu được paper ghi nhận.

## 6. Giới hạn

**Do tác giả nêu:**
- Ngưỡng toàn cục có thể xóa trọn một tầng (MNIST); với mạng không có skip connection (VGG), multi-pass phải thêm ràng buộc tối đa 50% kênh mỗi tầng mỗi vòng.
- ResNet-164 và CIFAR-100 tiết kiệm kém hơn (giả thuyết: bottleneck; nhiều lớp hơn).
- Cắt quá mạnh thì không phục hồi được (CIFAR-100 từ vòng 3, Hình 5).
- Trên ImageNet, tiết kiệm FLOPs thấp (30.4%) vì kênh bị cắt chủ yếu ở FC chứ không ở conv.
- Fine-tune ImageNet chỉ 5 epoch do giới hạn thời gian.

**Do chúng ta nhận xét (suy luận):**
- Cần **huấn luyện lại từ đầu với phạt** (160 epoch CIFAR, 60 epoch ImageNet): tốn kém khi chỉ có một mô hình pretrain khổng lồ; không phải one-shot post-training.
- $\gamma$ nhỏ không bảo đảm kênh vô dụng nếu sau đó là conv với trọng số lớn, hoặc $\beta$ lớn (kênh tạo ra hằng số vẫn mang thông tin offset). Paper không bàn $\beta$. Tiêu chí dựa trên $\gamma$ chỉ phù hợp mạng có BN; transformer dùng LayerNorm/RMSNorm với các quan hệ khác.
- Nhiều siêu tham số ($\lambda$, ngưỡng, số vòng); $\lambda$ chọn bằng grid search nhỏ.
- Kết quả là một seed mỗi cấu hình; chênh lệch 0.1–0.3 điểm có thể nằm trong nhiễu.
- So với baseline "mạng hẹp train từ đầu" thiếu trong bản đã đọc (liên quan 11).
- Chỉ ảnh phân loại; không có kết quả với chuỗi/ngôn ngữ.
- Ghi nhận sự tăng tốc thật dựa trên một thí nghiệm batch 64 trên CIFAR; chưa có batch-1.

## 7. Liên hệ với các bài khác

- **04 Li et al.:** baseline trực tiếp (Bảng 7). 04 chọn filter bằng $\ell_1$ trọng số sau train; 05 học tiêu chí trong lúc train bằng $\gamma$.
- **02 Han:** nêu bất lợi của weight pruning (cần phần cứng thưa, không giảm bộ nhớ runtime).
- **06 ThiNet, 07 Molchanov:** các tiêu chí khác cho channel; so sánh với $\gamma$.
- **08–11:** Lottery Ticket và "Rethinking": mô hình hẹp thu được từ slimming có thể train từ đầu được không?
- **13/14 (head), 16 Movement pruning, 17 CoFi:** CoFi dùng mặt nạ $\ell_0$ cho head, neuron, hidden dimension — cùng tư duy "cổng nhân vào đơn vị + phạt thưa", nâng cấp từ $\ell_1$ sang $\ell_0$ khả vi (preface 6.6).
- **26 LLM-Pruner, 28 FLAP:** cắt neuron/head cho LLM; không cần train lại với phạt như 05.
- **20 DepGraph:** xử lý tự động các nhóm kênh bị ràng buộc (ở đây 05 xử lý bằng channel selection layer).
- **19 HALP, 12 (Blalock):** nhấn mạnh phải đo latency; Phụ lục B của 05 là ví dụ nhỏ.

## 8. Ứng dụng cho dự án (suy luận)

- **Ý tưởng đáng mượn**: *cổng nhân vô hướng $\gamma_j$ cho mỗi neuron MLP / head / channel ViT* + phạt $\ell_1$ (hoặc $\ell_0$ như CoFi) trong **giai đoạn distillation**: vừa học KD vừa để mạng tự chọn neuron cần giữ. Sau đó cắt theo ngưỡng và tinh chỉnh tiếp. Phù hợp với việc phục hồi bằng KD từ teacher 4B.
- **Cái cần thay đổi**: Với transformer, "BN $\gamma$" không có sẵn; cổng phải được **thêm tay** (một vector scale nhân sau activation của MLP, hoặc dùng scale của RMSNorm nếu thích hợp). Hệ số RMSNorm thuộc residual stream (chung cho mọi neuron) nên không đại diện neuron riêng; vì vậy nên thêm cổng mới khởi tạo bằng 1 (không phải 0.5).
- **Khớp với quan sát của dự án**: slimming chỉ chọn kênh theo tầm quan trọng *được học* → có thể giúp cắt MLP neuron phục hồi tốt hơn so với heuristic, vì mạng đã thích nghi với việc "dồn tín hiệu" vào số kênh giữ lại trước khi cắt. Giả thuyết cần kiểm chứng; không phải kết quả của paper.
- **Ngưỡng toàn cục** giữa các block hỗn hợp (GatedDeltaNet / attention / ViT) có thể xóa nguyên một block (như MNIST); cần ràng buộc tối thiểu mỗi tầng (như 50%/tầng ở multi-pass).
- **Hạn chế thực tế**: cần chạy huấn luyện có phạt: cho mô hình 0.8B cần ngân sách KD riêng; không áp dụng được cho post-training một lần.
- Về 0.1B (nén ~8×), multi-pass cho thấy VGG chịu được 20× tham số ở CIFAR, nhưng đó là mạng dư thừa cực lớn (20M tham số cho CIFAR); không nên ngoại suy sang VLM OCR.

## 9. Thí nghiệm gợi ý

1. **Cổng $\ell_1$ cho neuron MLP khi distill**: thêm vector $g\in\mathbb R^{m}$ (khởi tạo 1) nhân sau activation MLP của một số block; loss = KD + $\lambda\|g\|_1$; so với baseline cắt cùng số neuron theo $\ell_1$ trọng số (04) hoặc theo activation. Đo CER và loss KD sau cùng số bước. Quét $\lambda\in\{10^{-5},10^{-4},10^{-3}\}$.
2. **Phân bố cổng**: vẽ histogram $g$ theo từng block (giống Hình 4) và xem block nào bị cắt nhiều nhất — có trùng với bản đồ độ nhạy từ thí nghiệm của 04 không? Nếu hai kết quả mâu thuẫn, ngưỡng toàn cục không phù hợp.
3. **Ngưỡng toàn cục vs tối thiểu mỗi tầng**: cắt 50% neuron toàn mạng bằng $g$; so sánh (a) ngưỡng toàn cục thuần túy và (b) có ràng buộc giữ ≥ 30% mỗi block. Đo xem (a) có xóa nguyên một block không.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung | Vị trí |
|---|---|---|
| Mục tiêu | $L=\sum\ell(f(x,W),y)+\lambda\sum_{\gamma}|\gamma|$ | Eq. 1 |
| BN | $\hat z=(z_{in}-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$, $z_{out}=\gamma\hat z+\beta$ | Eq. 2 |
| Scale factor | dùng $\gamma$ của BN, khởi tạo 0.5 | Mục 3–4 |
| Ngưỡng | phân vị toàn cục của $|\gamma|$ (vd 70%) | Mục 3 |
| Multi-pass | lặp train-có-phạt → cắt → fine-tune | Hình 2 |
| Channel selection layer | cho mạng pre-activation (ResNet, DenseNet) | Mục 3 |
| Kết quả đỉnh | VGGNet CIFAR-10: −88.5% tham số, −51.0% FLOPs, lỗi 6.20 (vs 6.34); multi-pass vòng 5: ×20 tham số, ×5 FLOPs; VGG-A ImageNet: −82.5% tham số, −30.4% FLOPs, lỗi 36.66 vs 36.69 | Bảng 1, 2, 4 |

## 11. Kết luận ngắn

Network Slimming biến việc chọn kênh thành một phần của huấn luyện: chỉ cần phạt $\ell_1$ lên $\gamma$ của BN rồi cắt theo ngưỡng toàn cục, ta được mô hình dense hẹp hơn, nhỏ hơn về tham số, activation và FLOPs, chạy trên phần cứng thường, và thường không mất (đôi khi cải thiện) accuracy. Bài này là mẫu của họ phương pháp "cổng học được + phạt thưa" trải dài đến CoFi (17); điểm yếu là phải huấn luyện lại, phụ thuộc BN và ngưỡng toàn cục dễ xóa nguyên tầng.

Nguồn: https://arxiv.org/abs/1708.06519
