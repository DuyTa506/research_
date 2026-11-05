# Pruning Filters for Efficient ConvNets (2016)

**Tác giả:** Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, Hans Peter Graf  
**Venue / arXiv:** arXiv:1608.08710 (bản v3, 10/3/2017); ICLR 2017 (bản arXiv ghi "Published as a conference paper at ICLR 2017").  
**Nguồn:** https://arxiv.org/abs/1608.08710  
**Chủ đề:** Cắt filter (channel) có cấu trúc bằng chuẩn $\ell_1$, phân tích độ nhạy từng tầng, cắt VGG/ResNet, prune-once-and-retrain.  
**Vị trí trong lộ trình:** Đọc sau 02 (Han: cắt trọng số không cấu trúc) và 03 (cắt dần). Bài này chuyển sang **cắt cấu trúc để giảm tính toán thật mà không cần thư viện thưa**. Tiếp theo: 05 (Network Slimming: học cổng $\gamma$ thay vì dùng $\ell_1$ của trọng số), 06 (ThiNet: chọn filter theo tái tạo đầu ra), 07, 20.

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch. Số liệu lấy từ bản arXiv đã đọc; một số bảng PDF bị xáo cột nên chỉ trích dẫn khi đọc rõ. Phần "Ứng dụng cho dự án" là suy luận của chúng ta.

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

Han et al. (02) nén AlexNet/VGG bằng cách cắt trọng số theo độ lớn. Li et al. chỉ ra hai vấn đề:

1. Phần lớn tham số bị loại nằm ở tầng FC, nơi tính toán rẻ. Paper nêu: FC của VGG-16 chiếm **90% tham số nhưng dưới 1% FLOP**. Vì vậy cắt tham số không nhất thiết giảm tính toán.
2. Cắt trọng số tạo ra **thưa bất quy tắc** trong các tầng conv; muốn tăng tốc cần thư viện/phần cứng cho conv thưa (và cấu trúc dữ liệu thưa tốn thêm bộ nhớ, đáng kể với trọng số độ chính xác thấp).

Trong khi đó, mạng hiện đại (ResNet) đã thay FC bằng average pooling, tính toán nằm ở conv.

**Thuật ngữ:**
- **Filter:** tensor $F_{i,j}\in\mathbb R^{n_i\times k\times k}$ tạo ra *một* feature map; tầng $i$ có $n_{i+1}$ filter.
- **Feature map:** đầu ra 2D của một filter (một channel).
- **BLAS:** thư viện nhân ma trận dense hiệu quả; mục tiêu là vẫn dùng được nó.

## 2. Ý tưởng chính

Thay vì cắt từng trọng số, **cắt nguyên filter**. Bỏ một filter đồng nghĩa với bỏ feature map nó tạo ra, và bỏ luôn các kernel ở tầng kế tiếp ăn feature map đó, nên ma trận kết quả vẫn dense, chỉ nhỏ hơn. Để chọn filter, chỉ cần **tổng trị tuyệt đối trọng số của filter ($\ell_1$-norm)**: filter có chuẩn nhỏ tạo ra feature map yếu nên ít quan trọng. Paper còn phân tích độ nhạy từng tầng (cắt riêng từng tầng rồi đo accuracy) để chọn tỉ lệ cắt theo giai đoạn (stage), xử lý ràng buộc của ResNet, và retrain một lần (prune once and retrain) thay vì lặp từng tầng.

---

## 3. Phương pháp chi tiết

### 3.1 Phép tính khi bỏ filter

Tầng $i$: đầu vào $x_i\in\mathbb R^{n_i\times h_i\times w_i}$, $n_{i+1}$ filter $F_{i,j}\in\mathbb R^{n_i\times k\times k}$, đầu ra $x_{i+1}\in\mathbb R^{n_{i+1}\times h_{i+1}\times w_{i+1}}$. Số phép tính của tầng:

$$
n_{i+1}\,n_i\,k^2\,h_{i+1}w_{i+1}.
$$

Bỏ một filter $F_{i,j}$ giảm $n_i k^2 h_{i+1}w_{i+1}$ phép tính của tầng $i$, và các kernel ở tầng $i{+}1$ nhận feature map $x_{i+1,j}$ cũng bị bỏ, giảm thêm $n_{i+2}k^2h_{i+2}w_{i+2}$. Do đó **cắt $m$ filter của tầng $i$ giảm $m/n_{i+1}$ chi phí tính toán của cả tầng $i$ và $i{+}1$**. Khi tầng bị cắt có BatchNorm theo sau, tham số BN tương ứng cũng bị bỏ; nếu bỏ channel cuối cùng của khối conv, đầu vào tầng linear tiếp theo thay đổi và các kết nối tương ứng cũng bị bỏ.

Đây chính là phiên bản CNN của quy tắc "bỏ một hàng của $W_{in}$ và một cột của $W_{out}$" (preface mục 6.7).

### 3.2 Chọn filter trong một tầng

$$
\boxed{\ s_j=\|F_{i,j}\|_1=\sum_{l=1}^{n_i}\sum_{u,v}|K_l(u,v)|\ }
$$

```
Cắt m filter của tầng i:
1. Với mỗi filter F_{i,j}: s_j = tổng |trọng số|
2. Sắp xếp theo s_j
3. Bỏ m filter có s_j nhỏ nhất cùng feature map tương ứng;
   bỏ các kernel của tầng i+1 ứng với các feature map đó
4. Tạo ma trận kernel mới cho tầng i và i+1, sao chép phần còn lại
```

Lý do theo paper: $n_i$ giống nhau ở mọi filter nên $s_j$ cũng tỉ lệ với độ lớn trung bình của kernel; filter nhỏ cho feature map có activation yếu. So sánh thực nghiệm (mục 5): cắt filter nhỏ nhất tốt hơn cắt ngẫu nhiên và hơn nhiều cắt filter lớn nhất.

**Quan hệ với các phương pháp khác (theo paper):**
- *Cắt trọng số theo magnitude* có thể vô tình cắt cả filter nếu mọi kernel dưới ngưỡng, nhưng khó dự đoán số filter bị cắt, cần chỉnh ngưỡng cẩn thận và tạo kernel thưa khó tăng tốc.
- *Group-sparse regularization* ($\ell_{2,1}$-norm; Zhou 2016, Wen 2016) đẩy cả filter về 0 trong lúc huấn luyện. Li et al. không thấy khác biệt đáng kể giữa $\ell_1$ và $\ell_2$ khi chọn filter (Phụ lục 6.1: filter quan trọng có cả hai chuẩn lớn).

### 3.3 Độ nhạy của từng tầng

Cắt **riêng từng tầng** rồi đo accuracy trên tập validation. Hình 2(b): các tầng có accuracy giữ được khi cắt tương ứng với tầng có đường sắp xếp $\ell_1$ **dốc** (một số ít filter lớn, nhiều filter nhỏ) trong Hình 2(a); tầng có đường phẳng (filter có chuẩn tương tự nhau) nhạy hơn. Để tránh thêm siêu tham số theo tầng, dùng **cùng tỉ lệ cắt cho mọi tầng trong một stage** (cùng kích thước feature map); với tầng nhạy, cắt ít hơn hoặc bỏ qua.

### 3.4 Cắt nhiều tầng cùng lúc

Hai chiến lược chọn filter ở nhiều tầng:
- **Independent:** chọn filter của mỗi tầng độc lập, tính $s_j$ trên toàn bộ kernel gốc, kể cả những kernel ứng với feature map đã bị bỏ ở tầng trước.
- **Greedy:** khi tính $s_j$, **không tính các kernel ứng với feature map đã bị cắt** ở tầng trước. Không tối ưu toàn cục nhưng "holistic" và cho accuracy cao hơn, nhất là khi cắt nhiều.

### 3.5 ResNet

- Với khối residual có **projection shortcut** (Hình 4): filter của tầng conv *thứ nhất* của khối cắt tự do (không đổi số channel đầu ra khối). Tầng *thứ hai* bị ràng buộc: số channel ra phải khớp với nhánh tắt (identity), nên chọn channel cần cắt theo kết quả cắt của shortcut (filter 1×1) và cắt cùng chỉ số ở tầng thứ hai.
- Với ResNet CIFAR (shortcut identity + zero padding, không có projection): chỉ cắt tầng thứ nhất của khối.
- Lý do (theo paper): feature map identity quan trọng hơn feature map residual thêm vào.

### 3.6 Retrain

Hai chiến lược:
1. **Prune once and retrain:** cắt nhiều tầng một lần rồi retrain tới khi phục hồi độ chính xác gốc. Hiệu quả với tầng *bền* (resilient).
2. **Prune and retrain iteratively:** cắt từng tầng/ filter rồi retrain trước khi cắt tiếp; có thể tốt hơn nhưng đòi hỏi **nhiều epoch hơn nhiều**, nhất là mạng rất sâu.

Paper chọn (1) cho đơn giản. Với tầng nhạy hoặc cắt quá nhiều, có thể không phục hồi được. Thiết lập retrain: **learning rate hằng 0.001, 40 epoch với CIFAR-10 và 20 epoch với ImageNet** — khoảng *một phần tư* số epoch huấn luyện gốc (so với việc báo cáo trước đó đến 3× thời gian gốc cho retrain, Han et al.). Cài đặt bằng Torch7: tạo mô hình mới nhỏ hơn rồi sao chép tham số tương ứng.

---

## 4. Thí nghiệm và kết quả chính

Mạng: VGG-16 (biến thể CIFAR-10 của Zagoruyko có thêm BN, không Dropout), ResNet-56/110 (CIFAR-10), ResNet-34 (ImageNet ILSVRC2012). Baseline huấn luyện từ đầu theo siêu tham số ResNet.

### 4.1 Bảng tổng kết (Bảng 1; lỗi test/val tốt nhất trong lúc retrain)

| Mô hình | Lỗi (%) | FLOP | FLOP cắt | Tham số | Tham số cắt |
|---|---|---|---|---|---|
| VGG-16 | 6.75 | 3.13×10^8 | | 1.5×10^7 | |
| VGG-16-pruned-A | 6.60 | 2.06×10^8 | 34.2% | 5.4×10^6 | 64.0% |
| VGG-16-pruned-A, train từ đầu | 6.88 | | | | |
| ResNet-56 | 6.96 | 1.25×10^8 | | 8.5×10^5 | |
| ResNet-56-pruned-A | 6.90 | 1.12×10^8 | 10.4% | 7.7×10^5 | 9.4% |
| ResNet-56-pruned-B | 6.94 | 9.09×10^7 | 27.6% | 7.3×10^5 | 13.7% |
| ResNet-56-pruned-B, từ đầu | 8.69 | | | | |
| ResNet-110 | 6.47 | 2.53×10^8 | | 1.72×10^6 | |
| ResNet-110-pruned-A | 6.45 | 2.13×10^8 | 15.9% | 1.68×10^6 | 2.3% |
| ResNet-110-pruned-B | 6.70 | 1.55×10^8 | 38.6% | 1.16×10^6 | 32.4% |
| ResNet-110-pruned-B, từ đầu | 7.06 | | | | |
| ResNet-34 | 26.77 | 3.64×10^9 | | 2.16×10^7 | |
| ResNet-34-pruned-A | 27.44 | 3.08×10^9 | 15.5% | 1.99×10^7 | 7.6% |
| ResNet-34-pruned-B | 27.83 | 2.76×10^9 | 24.2% | 1.93×10^7 | 10.8% |
| ResNet-34-pruned-C | 27.52 | 3.37×10^9 | 7.5% | 2.01×10^7 | 7.2% |

Chú thích bảng của paper: **huấn luyện mô hình đã cắt từ đầu cho kết quả tệ hơn retrain mô hình đã cắt**, "có thể cho thấy khó khăn khi huấn luyện mạng dung lượng nhỏ". (Điều này mâu thuẫn một phần với kết quả của 11; xem mục 7.)

### 4.2 VGG-16 trên CIFAR-10 (Bảng 2)

13 tầng conv + 2 tầng linear. Cấu hình pruned-A: conv 1 giảm 64→32 map và conv 2 giữ 64 map (nhưng nhận đầu vào ít hơn), cả hai giảm 50% FLOP; conv 8: 512→256 (-50%); conv 9–13: 512→256 (-75% FLOP mỗi tầng); linear đầu: 512 (-50%); conv 3–7 giữ nguyên. Tổng **34% FLOP**, **64% tham số**. Phát hiện chi tiết:
- Mỗi tầng conv có 512 feature map **có thể bỏ ít nhất 60% filter mà không ảnh hưởng accuracy**; với retrain khoảng **90%** (Hình 2b, 2c). Giải thích đề xuất: các filter này hoạt động trên feature map 4×4 hoặc 2×2 nên có thể không có quan hệ không gian có ý nghĩa; (ResNet CIFAR không có conv dưới 8×8).
- Khác với nhận xét ở các nghiên cứu trước (Zeiler & Fergus; Han), **tầng đầu bền với cắt hơn các tầng kế tiếp** trên CIFAR-10 — tác giả gán cho CIFAR-10 đơn giản, khác ImageNet (Hình 5). Cắt 80% filter của tầng 1 còn 12 filter vẫn nhiều hơn số kênh đầu vào; nhưng cắt 80% filter của tầng 2 là ánh xạ 64→12, làm mất thông tin và hại accuracy.

### 4.3 ResNet-56/110 trên CIFAR-10

Hình 6: độ nhạy của tầng đầu tiên mỗi khối. Phần lớn bền; một số tầng *không retrain còn cải thiện hiệu năng* ở ResNet-110. Các tầng nhạy (ResNet-56: tầng 20, 38, 54; ResNet-110: 36, 38, 74) nằm ở **các khối gần chỗ số feature map thay đổi** (khối đầu/cuối mỗi stage); tác giả cho rằng cần residual chính xác cho các feature map rỗng mới thêm. Cấu hình:
- ResNet-56-pruned-A: bỏ qua các tầng 16, 20, 38, 54; cắt 10% filter.
- ResNet-56-pruned-B: bỏ qua 16, 18, 20, 34, 38, 54; $p_1=60\%$, $p_2=30\%$, $p_3=10\%$.
- ResNet-110-pruned-A: $p_1=50\%$ và bỏ qua tầng 36.
- ResNet-110-pruned-B: bỏ qua 36, 38, 74; $p_1=50\%$, $p_2=40\%$, $p_3=30\%$.

Còn: tầng sâu nhạy hơn tầng nông; khi stage có hơn hai khối, **các khối ở giữa có thể dư thừa**, giải thích vì sao ResNet-110 dễ cắt hơn ResNet-56.

### 4.4 ResNet-34 trên ImageNet

Bốn stage (56, 28, 14, 7). Cấu hình A: $p_1=p_2=p_3=30\%$; B: $p_1=50\%$, $p_2=60\%$, $p_3=40\%$; các tầng nhạy bị bỏ qua (tầng 2, 8, 14, 16, 26, 28, 30, 32, tức khối đầu/cuối mỗi stage). B cho **24% giảm FLOP với khoảng 1% mất accuracy**. Cắt cả tầng conv thứ hai và shortcut (C: $p_3=20\%$ ở stage 3) chỉ giảm 7.5% FLOP với mất 0.75% accuracy. Kết luận: **cắt tầng thứ nhất của khối hiệu quả hơn cắt tầng thứ hai**, phù hợp thiết kế bottleneck. Paper cũng dự đoán ResNet-34 khó cắt hơn ResNet sâu hơn.

---

## 5. Phân tích / ablation

### 5.1 Nhỏ nhất vs ngẫu nhiên vs lớn nhất (Hình 8, VGG-16 CIFAR-10)

Cắt filter $\ell_1$ nhỏ nhất **tốt hơn cắt ngẫu nhiên ở đa số tầng**, ở tỉ lệ cắt 90% tốt hơn trên mọi tầng; cắt filter $\ell_1$ lớn nhất làm accuracy rơi nhanh, chứng tỏ các filter lớn là quan trọng.

### 5.2 So sánh với cắt theo activation (Hình 9)

Dùng thống kê trên $N=50{,}000$ ảnh train, trước BN và phi tuyến: mean-mean, mean-std, mean-$\ell_1$, mean-$\ell_2$, var-$\ell_2$ (Polyak & Wolf). Kết quả: cắt filter nhỏ nhất theo $\ell_1$ của *trọng số* tốt hơn các tiêu chí mean-mean, mean-$\ell_1$, mean-$\ell_2$ và var-$\ell_2$; mean-std tốt hơn hoặc ngang tới tỉ lệ cắt 60% nhưng sau đó giảm nhanh nhất ở conv 1–3. Kết luận: $\ell_1$ là "heuristic tốt" vì **không cần dữ liệu**.

### 5.3 Mỗi cải tiến góp bao nhiêu

- **Độ nhạy theo tầng + bỏ qua tầng nhạy** là đóng góp lớn nhất cho ResNet (pruned-A, B).
- **Retrain:** train mô hình cắt từ đầu kém hơn rõ trên ResNet-56 (8.69 vs 6.94) nhưng gần ngang trên VGG-16 (6.88 vs 6.60).
- **Greedy vs independent:** greedy tốt hơn khi cắt nhiều (nêu bằng lời; paper không có bảng riêng ở bản đã đọc).
- **Prune-once vs iterative:** iterative có thể tốt hơn nhưng đắt (nêu bằng lời; không có thí nghiệm đối chứng đầy đủ trong paper).

## 6. Giới hạn

**Do tác giả nêu:**
- Tầng nhạy hoặc cắt quá nhiều có thể không phục hồi được bằng retrain ngắn.
- ResNet-34 khó cắt hơn ResNet sâu; ResNet bị ràng buộc bởi shortcut.
- Một số lựa chọn (tầng bỏ qua, tỉ lệ theo stage) xác định thực nghiệm theo độ nhạy, không tự động.

**Do chúng ta nhận xét (suy luận):**
- Paper **không đo thời gian suy luận thực** (chỉ FLOP, tham số) bất chấp mục tiêu tăng tốc thực (thời gian thực được đo ở các bài sau như 05 phụ lục, 19 HALP).
- Tiêu chí $\ell_1$ của trọng số **bỏ qua BatchNorm** (hệ số $\gamma,\beta$ có thể đảo ý nghĩa của độ lớn filter) và độ lớn đầu vào; các bài sau (05, 06, 07) sửa điều này.
- Kết quả tốt trên CIFAR-10 (mạng nhỏ, tập đơn giản); ResNet-34 trên ImageNet chỉ giảm 24% FLOP và mất ~1% accuracy, tức hiệu quả giảm đáng kể ở bài toán khó.
- Cắt các tầng "độc lập" cùng tỉ lệ trong stage là xấp xỉ thô; không có cơ chế phân bổ ngân sách tối ưu giữa các tầng.
- Không so sánh với baseline "mô hình nhỏ hơn được tối ưu tốt" (khác với 03); nhận định "train from scratch tệ hơn" là trên một số cấu hình và đã bị 11 phản biện.
- Một seed cho mỗi cấu hình trong bảng; chênh lệch 0.1–0.2% ở CIFAR có thể nằm trong nhiễu.

## 7. Liên hệ với các bài khác

- **02 Han:** tiền thân; 04 chỉ ra vấn đề tăng tốc và đổi sang cắt cấu trúc, đồng thời lưu ý phép so sánh giữa cắt filter và cắt trọng số nhỏ theo ngưỡng.
- **05 Network Slimming:** cùng mục tiêu; thay tiêu chí $\ell_1$ của trọng số bằng hệ số $\gamma$ của BN học bằng $\ell_1$ trong lúc train; dùng ngưỡng toàn cục; 05 so sánh trực tiếp với 04 (Bảng 7 của 05: cùng mức nén 88.5% trên CIFAR-10, slimming 6.20% vs 04 6.88%).
- **06 ThiNet, 07 Molchanov:** tiêu chí tốt hơn dựa trên tái tạo đầu ra hoặc Taylor; 04 là baseline "data-free".
- **11 Rethinking the value of pruning:** đặt lại câu hỏi retrain/khởi tạo từ trọng số cũ vs train mô hình cắt từ đầu; kết quả 04 (ở đây "từ đầu" thường kém) cần đối chiếu.
- **13, 14 (cắt head), 26 LLM-Pruner, 28 FLAP:** áp dụng cùng ý "bỏ hàng/cột liên đới" cho attention/MLP. 
- **20 DepGraph:** tổng quát hóa việc xử lý phụ thuộc cấu trúc mà 04 làm thủ công với ResNet (mục 3.5).
- **31–36 (cắt layer)**: ở ResNet 04 đã thấy khối giữa dư thừa; 04 không cắt khối nhưng quan sát "các khối giữa stage có thể dư thừa" gợi ý trước hướng bỏ layer.

## 8. Ứng dụng cho dự án (suy luận)

- **Liên quan trực tiếp**: dự án cần *cắt có cấu trúc để thu nhỏ ma trận* — đúng triết lý bài này. ViT 12 block của LightOnOCR có thể dùng ý tưởng "độ nhạy từng tầng, rồi chọn tỉ lệ theo stage/block, bỏ qua block nhạy".
- **Chuyển sang transformer**: filter ↔ neuron MLP (hàng $W_{in}$ + cột $W_{out}$) hoặc head; $\ell_1$-norm của hàng $W_{in}$ là điểm số data-free đơn giản nhất. Nhưng vì transformer có RMSNorm/LayerNorm và residual, độ lớn trọng số kém tin cậy hơn; nên kết hợp tiêu chí dựa activation (xem 24, 28, 29).
- **Quan sát đáng chú ý cho giả thuyết của dự án** ("cắt MLP neuron phục hồi kém hơn cắt layer"): trong 04, cắt filter ở mạng nhỏ thành công vì chỉ 34%/38% FLOP, và phải bỏ qua tầng nhạy; mức cắt mạnh hơn (như 2.7× hoặc 8×) không được kiểm tra. Cắt ở mức nén lớn như dự án nằm ngoài vùng bằng chứng của bài này.
- **Dependency**: khi cắt width của residual stream trong decoder lai (GatedDeltaNet + attention), cần cắt cùng chỉ số ở mọi lớp ghi vào residual — bài học ResNet mục 3.5 áp dụng trực tiếp (xem 20, 29).
- **Greedy**: tính điểm của tầng sau trên *kernel còn lại* của tầng trước là thủ thuật dễ nhập vào quy trình cắt tuần tự (cắt MLP của block $b$ rồi tính điểm block $b+1$ trên activation đã bị thay đổi).

## 9. Thí nghiệm gợi ý

1. **Bản đồ độ nhạy theo block**: với ViT 12 block và decoder, cắt riêng mỗi block 30/50/70% neuron MLP bằng $\ell_1$ của hàng $W_{in}$, đo CER/KD-loss trước và sau 300 bước fine-tune. Vẽ như Hình 2(b)(c) để chọn block cần bảo vệ.
2. **Small vs random vs largest + activation-aware**: lặp thí nghiệm Hình 8–9 cho MLP neuron của một block: cắt $\ell_1$ nhỏ nhất / ngẫu nhiên / lớn nhất / theo $\mathrm{mean}\,|a|$ trên 1k ảnh OCR. Mục đích: kiểm tra liệu tiêu chí data-free còn tốt trên transformer.
3. **Greedy vs independent**: cắt đồng thời MLP của nhiều block; so điểm tính trên trọng số gốc (independent) với điểm tính sau khi block trước đã bị cắt (greedy). Đo chênh lệch loss trước fine-tune.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung | Vị trí |
|---|---|---|
| FLOP của tầng conv | $n_{i+1}n_ik^2h_{i+1}w_{i+1}$ | Mục 3 |
| Lợi ích cắt $m$ filter | giảm $m/n_{i+1}$ chi phí của cả tầng $i$ và $i{+}1$ | Mục 3 |
| Điểm filter | $s_j=\sum_l\sum|K_l|=\|F_{i,j}\|_1$ | Mục 3.1 |
| Độ nhạy tầng | cắt riêng từng tầng rồi đo val accuracy | Mục 3.2 |
| Independent vs greedy | có / không tính kernel của feature map đã cắt | Mục 3.3 |
| ResNet | tầng 1 khối cắt tự do; tầng 2 theo shortcut | Mục 3.3 |
| Retrain | LR 0.001; 40 epoch (CIFAR), 20 epoch (ImageNet) ≈ 1/4 số epoch gốc | Mục 4 |
| Kết quả chính | VGG-16 −34.2% FLOP, −64% tham số, lỗi 6.60 (gốc 6.75); ResNet-110 −38.6% FLOP, lỗi 6.70 (gốc 6.47); ResNet-34 −24.2% FLOP, lỗi 27.83 (gốc 26.77) | Bảng 1 |

## 11. Kết luận ngắn

Bài chứng minh rằng một tiêu chí cực đơn giản — tổng trị tuyệt đối trọng số của filter — cộng với phân tích độ nhạy từng tầng và retrain ngắn là đủ để giảm khoảng 30–38% FLOP của VGG-16 và ResNet-110 trên CIFAR-10 gần như không mất accuracy, bằng những ma trận dense nhỏ hơn dùng được với BLAS thông thường. Hạn chế là mức nén vừa phải, phụ thuộc chọn thủ công các tầng nhạy, tiêu chí data-free thô, và chưa đo tốc độ thật — những điểm mà 05, 06, 07 và 19 cải thiện.

Nguồn: https://arxiv.org/abs/1608.08710
