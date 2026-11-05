# To prune, or not to prune: exploring the efficacy of pruning for model compression (2017)

**Tác giả:** Michael H. Zhu (Stanford; làm trong thời gian thực tập tại Google), Suyog Gupta (Google)  
**Venue / arXiv:** arXiv:1710.01878 (bản v2, 13/11/2017). Bản arXiv không ghi venue.  
**Nguồn:** https://arxiv.org/abs/1710.01878  
**Chủ đề:** Lịch cắt dần (automated gradual pruning) trong lúc train; so sánh "large-sparse" với "small-dense" cùng bộ nhớ trên CNN, LSTM, NMT.  
**Vị trí trong lộ trình:** Đọc sau 02 (Han 2015: train → cắt → retrain). Bài này (a) thay lịch cắt bằng *cắt dần trong lúc train* đơn giản, và (b) đặt câu hỏi nền tảng: *cắt một mô hình to có tốt hơn train thẳng mô hình nhỏ không?* Tiếp theo: 04, 05 (cắt có cấu trúc), 11 (Rethinking the value of pruning) và 43 (pruning vs train from scratch) quay lại câu hỏi này.

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch. Số liệu lấy từ bản arXiv đã đọc. Phần "Ứng dụng cho dự án" là suy luận của chúng ta.

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

Han et al. (02) và Narang et al. (2017) cho thấy có thể cắt rất nhiều trọng số với ít mất mát. Điều này gợi ý hai khả năng: mô hình gốc **quá tham số hóa**, và **một lựa chọn khác** là chỉ cần *giảm số hidden unit* (giữ cấu trúc dense) để được đánh đổi kích thước–độ chính xác tương tự. Muốn biết cách nào hơn, phải so sánh ở **cùng dấu chân bộ nhớ**.

Bối cảnh ứng dụng của paper: suy luận tiết kiệm năng lượng trên thiết bị biên. Các tác giả lưu ý: tải trọng nhạy độ trễ thường bị giới hạn bởi băng thông bộ nhớ; nén mô hình vừa giảm số lần truy cập bộ nhớ tốn năng lượng vừa tăng băng thông hiệu dụng. Nhưng mô hình thưa cần phần cứng/kernel cho ma trận thưa và tốn thêm bộ nhớ chỉ số.

Hai khái niệm:
- **Large-sparse:** huấn luyện mô hình lớn rồi cắt thưa để còn ít tham số khác 0.
- **Small-dense:** mô hình nhỏ dense (giảm số unit / width multiplier) có kích thước tương đương.

## 2. Ý tưởng chính

Hai đóng góp. Thứ nhất, một **lịch cắt dần tự động** rất đơn giản: tăng độ thưa từ $s_i$ đến $s_f$ theo hàm bậc ba, cứ vài trăm bước cập nhật mặt nạ bằng cách che đi các trọng số nhỏ nhất theo trị tuyệt đối trong từng tầng, không cần chỉnh nhiều siêu tham số và gắn được vào quá trình huấn luyện. Thứ hai, một **phép so sánh có kiểm soát**: với cùng số tham số khác 0, mô hình **lớn–thưa luôn thắng nhỏ–dense** trên MobileNet, LSTM ngôn ngữ (PTB) và NMT, với mức nén tới 10× và mất mát tối thiểu.

---

## 3. Phương pháp chi tiết

### 3.1 Cơ chế mask

Với mỗi tầng được chọn cắt, thêm một biến mask nhị phân cùng kích thước với tensor trọng số; mask quyết định trọng số nào tham gia forward. Chèn thao tác vào đồ thị TensorFlow để **sắp xếp trọng số theo |w|** rồi che các trọng số nhỏ nhất đến khi đạt độ thưa $s$ mong muốn. Gradient truyền qua mask nên trọng số bị che không được cập nhật.

### 3.2 Lịch cắt dần (Eq. 1)

Độ thưa tăng từ $s_i$ (thường 0) đến $s_f$ trong $n$ bước cắt, bắt đầu ở bước huấn luyện $t_0$, tần suất $\Delta t$:

$$
\boxed{\ s_t=s_f+(s_i-s_f)\Big(1-\frac{t-t_0}{n\,\Delta t}\Big)^{3}\quad t\in\{t_0,\,t_0+\Delta t,\dots,t_0+n\Delta t\}\ }
$$

Trực giác của paper: **cắt rất nhanh lúc đầu** khi còn nhiều kết nối dư thừa, rồi giảm dần lượng bị cắt mỗi bước khi còn ít trọng số. Số mũ ba làm đạo hàm $ds/dt$ lớn lúc đầu và tiến dần về 0 ở cuối (xem Hình 1 của paper với các mốc 2×, 4×, 8×, 10×). Khi đạt $s_f$ thì mask không cập nhật nữa.

Một ví dụ tính tay (tự dựng): với $s_i=0$, $s_f=0.9$, $n=100$: tại 1/3 quãng đường ($\frac{t-t_0}{n\Delta t}=\tfrac13$), $s=0.9(1-(2/3)^3)=0.9\times0.704\approx0.63$. Tức đã đạt độ thưa 63% khi mới đi 1/3 lịch; 2/3 lịch còn lại chỉ thêm 27 điểm phần trăm (từ 63% lên 90%), đúng tinh thần "cắt sớm, tinh chỉnh muộn".

```
Pseudo-code
for step in training:
    SGD update (trên trọng số có mask)
    if step in {t0, t0+Δt, ..., t0+nΔt}:
        s = s_f + (s_i - s_f) * (1 - (step - t0)/(n*Δt))**3
        for mỗi tầng được cắt:
            mask = (|W| >= phân vị s của |W| trong tầng đó)
    # sau nΔt bước: mask cố định, tiếp tục train bình thường
```

### 3.3 Chọn siêu tham số

- Tần suất $\Delta t$ từ 100 đến 1000 bước: **tác động không đáng kể** đến chất lượng cuối (kinh nghiệm của tác giả).
- Cắt bắt đầu **sau vài epoch** hoặc từ mô hình đã huấn luyện sẵn → xác định $t_0$.
- Độ dài $n$ phụ thuộc lịch learning rate: nếu learning rate **đã quá nhỏ**, mô hình khó phục hồi sau cú cắt; nếu quá lớn thì cắt khi trọng số chưa hội tụ. Vì vậy phải chọn lịch cắt gắn với lịch learning rate.
- Với InceptionV3 (Hình 2): cắt xảy ra khi learning rate còn tương đối cao để mạng "tự lành". Khi sparsity 87.5%, có thời điểm mô hình suy giảm "gần như thảm họa" rồi hồi phục nhanh khi tiếp tục huấn luyện.

### 3.4 Biến thể cho NMT

Vì huấn luyện NMT có phương sai lớn, paper thử: (i) **cắt đồng loạt cùng độ thưa mọi tầng** (mặc định); (ii) **"layerwise constant"**: chia khoảng cắt và tăng độ thưa từng tầng một; (iii) **global pruning**: cắt trọng số nhỏ nhất trên toàn mạng không phân biệt tầng (tương tự See et al. 2016). Kết quả: layerwise constant tốt nhất trung bình và được dùng để báo cáo.

### 3.5 Chi phí lưu trữ thưa

Hai biểu diễn:
- **Bit-mask:** 1 bit mỗi phần tử ma trận + vector các phần tử khác 0 → chi phí cố định bất kể độ thưa.
- **CSR(C):** mỗi phần tử khác 0 kèm số đếm các số 0 đứng trước (4–5 bit) → chi phí tỉ lệ với NNZ; có lợi khi rất thưa.

---

## 4. Thí nghiệm và kết quả chính

Mô hình: InceptionV3 và MobileNet (ảnh, ImageNet), LSTM ngôn ngữ trên Penn Tree Bank (PTB), seq2seq NMT kiểu Google (GNMT) trên WMT16. Mọi kết quả dùng float 32 bit.

### 4.1 InceptionV3 (Bảng 1)

| Độ thưa | NNZ | Top-1 | Top-5 |
|---|---|---|---|
| 0% | 27.1M | 78.1% | 94.3% |
| 50% | 13.6M | 78.0% | 94.2% |
| 75% | 6.8M | 76.1% | 93.2% |
| 87.5% | 3.3M | 74.6% | 92.5% |

Theo paper: 50% thưa ngang baseline; 87.5% (8× ít tham số) chỉ giảm khoảng 2% top-5.

### 4.2 MobileNet (Bảng 2)

Dense (width multiplier 0.25 / 0.5 / 0.75 / 1.0): NNZ 0.46M / 1.32M / 2.57M / 4.21M, top-1 50.6% / 63.7% / 68.4% / 70.6%, top-5 75.0% / 85.4% / 88.2% / 89.5%.  
MobileNet 1.0 được cắt thưa:

| Độ thưa | NNZ | Top-1 | Top-5 |
|---|---|---|---|
| 50% | 2.13M | 69.5% | 89.5% |
| 75% | 1.09M | 67.7% | 88.5% |
| 90% | 0.46M | 61.8% | 84.7% |
| 95% | 0.25M | 53.6% | 78.9% |

Lưu ý: 99% tham số của MobileNet 1.0 nằm ở pointwise 1×1 (74.6%) và FC (24.3%); conv chuẩn đầu và depthwise (1.1%) **không cắt**. Ví dụ so sánh của paper: sparse 75% (1.09M, 67.7%) hơn dense 0.5 (1.32M, 63.7%) 4 điểm và còn nhỏ hơn; sparse 90% (0.46M, 61.8%) hơn dense 0.25 (0.46M, 50.6%) **10.2 điểm** top-1 với cùng NNZ. Cắt MobileNet dùng learning rate khởi đầu nhỏ hơn 10 lần dense, các siêu tham số khác giữ nguyên.

### 4.3 LSTM trên PTB (Bảng 3, perplexity — thấp hơn là tốt)

| Cỡ | Độ thưa | NNZ | Perplexity |
|---|---|---|---|
| Small | 0% | 4.6M | 115.30 |
| Medium | 0% | 19.8M | 83.37 |
| Medium | 80% | 4.0M | 83.87 |
| Medium | 85% | 3.0M | 85.17 |
| Medium | 90% | 2.0M | 87.86 |
| Medium | 95% | 1.0M | 96.30 |
| Medium | 97.5% | 0.5M | 113.6 |
| Large | 0% | 66M | 78.45 |
| Large | 80% | 13.2M | 77.52 |
| Large | 85% | 9.9M | 78.31 |
| Large | 90% | 6.6M | 80.24 |
| Large | 95% | 3.3M | 87.83 |
| Large | 97.5% | 1.7M | 103.20 |

Mô hình large: 15M tham số embedding, 18M mỗi LSTM (2 lớp), 15M softmax = 66M. Cắt được cả embedding lẫn LSTM và softmax. **Large 90% (6.6M, PPL 80.24) hơn dense Medium (19.8M, PPL 83.37)** dù ít tham số hơn 3 lần.

### 4.4 NMT (Bảng 4, BLEU — cao hơn là tốt)

| Số unit | Độ thưa | NNZ | EN-DE | DE-EN |
|---|---|---|---|---|
| 256 | 0% | 34M | 23.52 | 26.52 |
| 512 | 0% | 81M | 26.05 | 28.88 |
| 768 | 0% | 140M | 26.63 | 29.41 |
| 1024 | 0% | 211M | 26.77 | 29.47 |
| 1024 | 80% | 44M | 26.86 | 29.50 |
| 1024 | 85% | 33M | 26.52 | 29.24 |
| 1024 | 90% | 23M | 26.19 | 28.81 |

Mô hình 1024 đơn vị có 37.4M tham số cho mỗi embedding encoder, decoder, softmax và 98.6M trong LSTM (211M tổng); không cắt attention. Phương sai lớn: thanh sai số là độ lệch chuẩn của BLEU qua **10 lần huấn luyện** độc lập. Dense giảm nhanh sau khi nhỏ 2×; sparse chỉ bắt đầu giảm sau 5× ít NNZ. Sparse 90% (23M) tương đương hoặc hơn dense 512 đơn vị (81M): 26.19 vs 26.05 EN-DE, 28.81 vs 28.88 DE-EN, dù ít NNZ hơn 3.5 lần.

### 4.5 Chi phí lưu trữ và kích thước thực (Bảng 5–6)

Bit-mask MobileNet: 0.52 MB cố định; CSR(C): 1.06 / 0.54 / 0.23 / 0.13 MB tại độ thưa 50% / 75% / 90% / 95%. Bảng 6 (kích thước mô hình tính cả overhead, 32 bit mỗi giá trị khác 0) so sánh small-dense với large-sparse cùng cỡ: ví dụ 10.28 MB dense (68.4%) so với 9.04 MB sparse (69.5%); 5.28 MB dense (63.7%) so với 4.88 MB sparse (67.7%). (Bảng trích từ PDF bị xáo cột; chỉ dùng được hai cặp đầu.)

---

## 5. Phân tích / ablation

- **Large-sparse > small-dense** trên mọi mô hình, và **khoảng cách mở rộng khi mô hình lớn** (PTB, NMT) so với MobileNet nhỏ gọn đã tối ưu (paper nhận xét thêm: PTB có nhiều tham số hơn nên cắt cho kết quả tốt hơn).
- **Có vùng nén tối ưu:** PTB cắt 95% (20×) làm tụt rõ so với 90% (10×). Bình luận của tác giả: để có mô hình thưa tốt nhất ở một cỡ cho trước, nên *huấn luyện dense lớn hơn 5–10× rồi cắt*, đừng lấy mô hình tốt nhất lớn nhất rồi cắt 20× trở lên; ví dụ 85% thưa của medium (3M, PPL 85.17) tốt hơn 95% thưa của large (3.3M, PPL 87.83). (Chú ý: họ dùng cùng siêu tham số với dense, nên có thể có thêm dư địa nếu tinh chỉnh.)
- **Phương sai:** NMT dao động mạnh giữa các lần train → cần nhiều seed (10 lần trong paper).
- **Layerwise constant vs đồng loạt vs global** cho NMT: layerwise constant tốt nhất trung bình.
- **Mối quan hệ lịch cắt–learning rate** (mục 3.3).
- **Overhead lưu trữ thưa** làm giảm tỷ lệ nén thực, nhất là khi lượng tử hóa xuống 8 bit; large-sparse vẫn thắng sau khi tính overhead (Bảng 6).

## 6. Giới hạn

**Do tác giả nêu:**
- Kết quả dùng float 32 bit; với suy luận độ chính xác thấp (int8) overhead của biểu diễn thưa chiếm phần lớn hơn; tương tác giữa lượng tử hóa và cắt tỉa **để dành cho công việc sau**.
- Có thể có kết quả tốt hơn ở độ thưa ≥ 95% nếu tinh chỉnh siêu tham số riêng cho từng mô hình.
- Mô hình thưa cần **phần cứng chuyên dụng** hoặc phép toán ma trận thưa để thấy lợi ích tốc độ.

**Do chúng ta nhận xét (suy luận):**
- Không đo **tốc độ/độ trễ thực** trên bất kỳ phần cứng nào; kết luận "tốt hơn" chỉ tính số tham số khác 0 (cộng overhead lưu trữ), mà bộ nhớ tại runtime của kernel thưa trên GPU có thể khác.
- "Small-dense" chỉ thu nhỏ bằng width/unit, không thử giảm số tầng, distillation, hay huấn luyện lâu hơn cho mô hình nhỏ. Paper về sau (11, 43) chỉ ra kết quả so sánh phụ thuộc mạnh vào ngân sách huấn luyện mô hình nhỏ.
- Mô hình lớn được cắt **được train/finetune lâu hơn** mô hình nhỏ (cắt xảy ra trong quá trình huấn luyện dài), nên so sánh chưa ngang bằng về tính toán.
- Không có cấu trúc: cắt theo magnitude từng phần tử, không so sánh với cắt có cấu trúc (04, 05) hay cùng ngân sách FLOPs.
- Chỉ cắt các tầng lớn (bỏ depthwise ở MobileNet), kết quả không mang sang cấu trúc hiện đại như attention.

## 7. Liên hệ với các bài khác

- **02 Han 2015:** cùng magnitude, nhưng 03 *tích hợp cắt vào huấn luyện* thay vì cắt sau rồi retrain; 03 cũng lưu ý So sánh với Narang et al. (hai pha, hai độ dốc, ngưỡng thủ công).
- **01 OBD/OBS:** bài này nhắc lại OBD/OBS trong related work, chọn magnitude vì **mở rộng được**.
- **04, 05:** cắt có cấu trúc; 03 nhận xét chúng "phụ thuộc cấu trúc lớp conv, khó mở rộng sang LSTM", đồng thời lý do cũng là sự đánh đổi mà ta phải chấp nhận.
- **08 Lottery Ticket:** cùng tinh thần "tìm mạng con thưa trong mạng lớn"; 03 dùng cắt dần thay vì rewinding.
- **11 Rethinking the value of pruning, 43 Small LLMs pruning vs scratch, 12 State of pruning:** quay lại câu hỏi "large-sparse hay small-dense"; đáng đọc cùng để thấy kết quả 03 *không phổ quát*.
- **18 2:4 sparsity:** dạng thưa có phần cứng hỗ trợ mà 03 mong chờ ("customize the next generation accelerators").
- Công thức lịch bậc ba về sau xuất hiện trong nhiều thư viện (ví dụ dạng lịch sparsity cubic) và trong các phương pháp cắt dần trong LLM (như 16 movement pruning dùng lịch tăng dần độ thưa tương tự — xin kiểm tra lại ở ghi chú 16).

## 8. Ứng dụng cho dự án (suy luận)

- **Điều có thể dùng:** (1) *Lịch cắt dần* thay vì cắt một lần: nếu dự án cắt nhiều neuron MLP/nhiều block ViT, có thể giảm dần theo bậc ba trong lúc distill từ teacher 4B, để mạng kịp thích nghi. (2) *Gắn lịch cắt với learning rate* — tránh cắt khi LR đã quá nhỏ. (3) *Câu hỏi nền*: so với train từ đầu một mô hình ~0.3B/0.1B hoặc dùng tiền-huấn-luyện nhỏ, cắt từ 0.8B có thật hơn không? Nên chạy đối chứng này (xem 43).
- **Điều không áp dụng được:** mặt nạ **thưa không cấu trúc** của paper không thu nhỏ ma trận → không giúp VRAM ≤ 8 GB hay latency batch-1 nếu không có kernel thưa. Dự án cần phiên bản cắt có cấu trúc (hàng/cột/ head/ block) với lịch cắt tương tự.
- **Gợi ý "bằng chứng gián tiếp"**: kết luận "dư thừa mô hình lớn ⇒ cắt 5–10× rồi hồi phục" khớp ý tưởng cắt 0.8B → 0.3B (2.7×) khả thi; còn 0.1B (8×) nằm gần vùng "nén quá mức" mà paper quan sát (PTB 95% tụt rõ). Đây là suy diễn khá lỏng vì PTB LSTM không giống VLM.

## 9. Thí nghiệm gợi ý

1. **Lịch cắt cấu trúc dần dần**: áp dụng công thức bậc ba để *giảm dần số neuron MLP* (mask mềm: nhân hệ số $m_j(t)$ giảm dần về 0 cho neuron bị chọn) trong 2k bước KD, so với cắt một lần rồi fine-tune cùng tổng số bước. Đo loss KD và CER.
2. **Large-sparse vs small-dense ở cấp cấu trúc**: cùng số tham số, so sánh (a) cắt cấu trúc 0.8B → 0.3B + KD với (b) khởi tạo ngẫu nhiên (hoặc tận dụng một phần embedding) mô hình 0.3B + KD cùng ngân sách token. Báo cáo cả đường cong theo số bước.
3. **Độ nhạy lịch–LR**: chạy ba cấu hình lịch cắt (sớm / giữa / muộn so với lịch LR) và xem tổn thất phục hồi khớp nhận xét "pruning dưới LR rất nhỏ khó hồi phục".

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung | Vị trí |
|---|---|---|
| Gradual pruning | $s_t=s_f+(s_i-s_f)(1-\tfrac{t-t_0}{n\Delta t})^3$ | Eq. 1 |
| Mask | che các |w| nhỏ nhất theo từng tầng đến độ thưa $s_t$; gradient qua mask | Mục 3 |
| Large-sparse vs small-dense | cùng số NNZ/bộ nhớ, sparse thắng trên mọi mô hình thử | Mục 4–5 |
| Bit-mask vs CSR(C) | 1 bit/phần tử cố định vs 4–5 bit/phần tử khác 0 | Mục 5 |
| Vùng nén tối ưu | cắt 5–10× từ mô hình lớn hơn tốt hơn cắt ≥ 20× | Mục 4.2 |
| Kết quả đỉnh | Inception-V3 87.5% thưa: 74.6/92.5; MobileNet 90%: 61.8% vs dense 0.25: 50.6%; PTB large 90%: PPL 80.24; NMT 1024 @90%: 26.19/28.81 BLEU | Bảng 1–4 |

## 11. Kết luận ngắn

Zhu & Gupta đưa ra một lịch cắt dần dễ dùng (bậc ba, mask theo magnitude, vài trăm bước mỗi lần) và một câu trả lời thực nghiệm cho "cắt hay không cắt": khi so cùng bộ nhớ trên nhiều kiến trúc, **mô hình lớn rồi cắt thưa thắng mô hình nhỏ dense**, nhưng chỉ ở mức nén vừa phải (≈ 5–10×) và chưa kể tốc độ thực. Bài học còn nguyên giá trị: lịch cắt gắn với lịch học và luôn so với baseline nhỏ có kiểm soát — nhưng cũng nên đọc cùng 11, 12, 43 vì kết luận này không phải lúc nào cũng đứng vững.

Nguồn: https://arxiv.org/abs/1710.01878
