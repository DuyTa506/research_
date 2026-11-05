# AM-RADIO: Agglomerative Vision Foundation Model -- Reduce All Domains Into One (2024)

**Tác giả:** Mike Ranzinger, Greg Heinrich, Jan Kautz, Pavlo Molchanov (NVIDIA).
**Venue / arXiv:** arXiv:2312.06709 (bản v5, 30/4/2024); CVPR 2024 (ghi ở đầu bản text). Mô hình student tên **RADIO**, biến thể hiệu quả **E-RADIO**.
**Nguồn:** https://arxiv.org/abs/2312.06709
**Chủ đề:** Distillation **nhiều teacher** ở mức **feature** (không nhãn, không logit) để gộp CLIP, DINOv2 và SAM vào một vision encoder; so sánh nhiều kiến trúc nhanh làm student; kiến trúc lai CNN-Transformer E-RADIO.
**Vị trí trong lộ trình:** Đọc trước: 03 (FitNets: bắt chước feature trung gian), 21 (DeiT), 22 (DINO, tiền thân của teacher DINOv2). Bài này thêm: khung feature distillation cho **vision encoder**, xử lý teacher bất đồng nhất (độ phân giải, chiều feature, tác vụ), và một thí nghiệm hiếm hoi so sánh nhiều kiến trúc student dưới cùng recipe distill. Bài kế: 24 (CLIP-KD), 25 (LLaVA-MoD), 26 (LLaVA-KD) cho nén VLM.

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch nguyên văn. Đây là bài **dài, nhiều bảng**; tôi tập trung vào phần distillation, ablation và kiến trúc student, bỏ qua chi tiết phụ trong phụ lục. Một số bảng ablation (Table 2, 3, 5 một phần) bị lỗi khi trích PDF nên tôi chỉ chép những số đọc rõ, và nói rõ khi bỏ qua. Phần "Ứng dụng cho dự án" là suy luận của chúng ta.

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

**Vision foundation model (VFM)** là bộ mã hóa ảnh lớn, đa dụng, dùng làm backbone cho nhiều tác vụ. Ba mô hình tiêu biểu có thế mạnh khác nhau:

- **CLIP** (huấn luyện trên cặp ảnh-chú thích): mạnh về zero-shot và ngữ nghĩa mức ảnh, tương thích với một mô hình văn bản.
- **DINOv2** (tự giám sát, kế thừa DINO, 22): mạnh ở tác vụ dày (dense) như phân đoạn ngữ nghĩa với feature tốt theo không gian.
- **SAM** (Segment Anything): mạnh ở phân đoạn instance, biên đối tượng; nhưng như tác giả nhận xét, "out of the box" không phù hợp tác vụ downstream khác.

Mỗi mô hình có "lỗ hổng". Distillation truyền thống thường nén một teacher thành student nhỏ hơn, có cả logit lẫn nhãn. Ở đây teacher không đồng nhất (khác tác vụ, độ phân giải đầu vào, patch size, chiều feature) nên không thể dùng logit hay nhãn chung. Cách làm: **bắt chước feature** (intermediate activations) và để mỗi teacher có "đầu" riêng.

Khái niệm: **summary feature** = vector toàn ảnh (class token); **spatial/dense feature** = lưới vector theo patch; **adaptor head** = mạng nhỏ chiếu feature student sang không gian feature của từng teacher; **linear probe** = huấn luyện một bộ phân loại tuyến tính trên feature đóng băng để đo chất lượng biểu diễn; **throughput** = số ảnh/giây.

## 2. Ý tưởng chính

Huấn luyện một student **từ đầu** (không khởi tạo từ teacher) trên ảnh không nhãn (DataComp-1B). Mỗi ảnh đi qua student một lần và qua từng teacher; student có, cho mỗi teacher, hai "đầu" MLP nhỏ (một cho summary, một cho feature theo patch), và loss là khoảng cách cosine (kèm smooth-L1 cho feature patch) giữa đầu ra của đầu và feature thật của teacher. Mọi teacher được cân bằng đơn giản (trọng số 1). Khi student đủ dung lượng, nó vượt từng teacher trên nhiều chỉ số, đồng thời "thừa hưởng" khả năng của teacher: dùng lại mô hình văn bản của CLIP (zero-shot) và dùng lại decoder mặt nạ của SAM (thay encoder). Phần hiệu quả: so sánh hơn 10 kiến trúc nhanh với cùng recipe; kết luận CNN thuần nhanh nhưng khó bắt chước ViT, nên đề xuất E-RADIO (stem conv + C2f + transformer cửa sổ).

## 3. Phương pháp chi tiết

### 3.1 Teacher, dữ liệu, student

Teacher dùng cho RADIO cuối (dấu * ở Table 1): DFN CLIP ViT-H/14 (378px), OpenAI CLIP ViT-L/14 (336px), DINOv2 ViT-g/14-reg, SAM ViT-H/16 (1024px). Student chuẩn: **ViT-H/16**; không phụ thuộc kiến trúc teacher. Dữ liệu: DataComp-1B (chỉ lấy ảnh; ImageNet-1k cho metric cao nhất nhưng tác giả cho rằng làm "zero-shot" mất ý nghĩa vì student học feature teacher trên chính miền đánh giá).

### 3.2 Adaptor heads

Mỗi teacher $i$ có hai đầu: MLP 2 tầng với LayerNorm và GELU ở giữa; chiều vào = embedding student, chiều giữa = embedding lớn nhất của các teacher, chiều ra = chiều feature của teacher đó. Ký hiệu $x'=f(x|\Theta_0)$ là feature student, $y_i=h_i(x'|\Theta_i)$ đầu ra đầu $i$, $z_i=t_i(x|\Phi_i)$ feature teacher.

### 3.3 Loss

Summary (dùng class token của CLIP và DINOv2; **không** khớp summary cho SAM, $\lambda_{SAM}=0$):

$$
\boxed{L_{summary}(x)=\sum_i\lambda_i\,L_{cos}\big(y^{(s)}_i,z^{(s)}_i\big)}
$$

Spatial (feature theo patch của mọi teacher):

$$
L_{match}(x,y)=\alpha L_{cos}(x,y)+\beta L_{smooth\text{-}\ell_1}(x,y),\quad
\boxed{L_{features}(x)=\sum_i\gamma_i\,L_{match}\big(h^{(v)}_i(x'),\,t^{(v)}_i(x)\big)}
$$

với $\alpha=0.9$, $\beta=0.1$; $\gamma_i=1$ (cân bằng "naive"), $\lambda_{CLIP}=\lambda_{DINO}=1$. Tổng loss là $L_{summary}+L_{features}$. Smooth-L1 được thêm để **khớp độ lớn vector**, cần thiết để student thay thế được teacher (drop-in) trong các framework gốc; paper cũng **bỏ chuẩn hóa feature** trong loss vì lý do này. Cosine tốt hơn L1, MSE, smooth-L1 đơn lẻ theo thực nghiệm của tác giả (không có bảng riêng).

### 3.4 Chi tiết kỹ thuật cho teacher bất đồng nhất

- **Multi-scale / độ phân giải:** student ăn 432² (tương đương 378² với patch 14 của teacher CLIP/DINOv2); DINOv2 chạy ở 224px rồi upsample đầu ra; SAM cần 1024px. Khi tỷ lệ downsample khác nhau (ví dụ teacher ViT-H/14, student ViT-H/16), feature map khác kích thước; loss nội suy song tuyến tính lên độ phân giải lớn hơn.
- **Rank/teacher partitioning:** các nhóm teacher được đặt lên các GPU khác nhau để mỗi GPU có batch và độ phân giải nhất quán; có SAM thì 64 GPU, nửa số GPU xử lý CLIP+DINOv2 (batch 32/GPU, 432px), nửa còn lại xử lý SAM (batch 2/GPU, 1024px); batch hiệu dụng 1.152. CLIP+DINOv2 một mình: 32 GPU, batch 1024.
- **Huấn luyện chung:** AdamW, batch 1024, cosine schedule, lr cơ sở 0.001, 600k bước (614M mẫu). Có huấn luyện hai giai đoạn (đoạn đầu với CLIP+DINOv2 trong 300k bước; phần mô tả tiếp theo bị cắt trong bản text nên tôi không chép).
- **CPE (Cropped Position Embedding):** student có bảng position embedding cỡ $128\times128$ vị trí, cắt ngẫu nhiên và nội suy cho khớp số patch; cho phép độ phân giải/aspect ratio tùy ý và giảm artifact tần số cao.
- **Cửa sổ attention thành augmentation:** để huấn luyện với SAM tiết kiệm bộ nhớ, thay global attention bằng windowed attention với kích thước cửa sổ lấy mẫu ngẫu nhiên khi train; ở inference chọn được cửa sổ. Thêm spectral reparametrization và weight decay 0.02 để tránh attention entropy collapse khi train độ phân giải cao.

### 3.5 Pseudo-code

```
teachers = [CLIP, DINOv2, SAM]   # đóng băng, mỗi teacher có (res, patch, dim) riêng
student  = ViT-H/16 + CPE + CLS token   # từ đầu
heads[i] = {summary: MLP2(LN,GELU), spatial: MLP2(LN,GELU)}  for each teacher i
for img in DataComp-1B:
    x' = student(resize(img, student_res))
    loss = 0
    for i in teachers:
        z = teacher_i(resize(img, teacher_res_i))        # summary + spatial
        if lambda_i > 0:   loss += lambda_i * cos(heads[i].summary(x'.cls), z.summary)
        y = heads[i].spatial(x'.patches)
        y, z = match_resolution(y, z)                     # bilinear
        loss += gamma_i * (0.9*cos(y, z.spatial) + 0.1*smooth_l1(y, z.spatial))
    loss.backward(); adamw_step()
```

### 3.6 E-RADIO

Kiến trúc lai: stem conv stride làm giảm ảnh 4x; 2 tầng khối conv YOLOv8 C2f; 2 tầng transformer dùng windowed attention (như Swin) xen kẽ với attention "toàn cục" kiểu EdgeViT (downsample 2x bằng conv stride 2 3x3, windowed attention, upsample lại). Cuối cùng upsample feature 2x bằng deconv và cộng với feature stage 3, nên độ giảm không gian tổng chỉ còn 16x (cải thiện tác vụ dày, chỉ chậm khoảng 10%).

## 4. Thí nghiệm và kết quả chính

**So sánh với teacher (Table 1, trích; k-NN và zero-shot trên ImageNet-1k; ADE20k/VOC: mIoU linear probe; GQA, POPE, TextVQA, VQAv2: thay encoder trong LLaVA-1.5; COCO: instance segmentation kiểu SAM; throughput trên A100 với TensorRT):**

| Mô hình | Tham số | Độ phân giải | ảnh/s | Zero-shot | k-NN | ADE20k | VOC | GQA | POPE | TextVQA | VQAv2 | COCO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DFN CLIP-H/14* | 633M | 378 | 170 | 83.90 | 85.27 | 39.00 | 70.29 | 61.73 | 85.91 | 56.78 | 78.78 | - |
| OpenAI CLIP-L/14* | 305M | 336 | 414 | 75.54 | 79.80 | 36.51 | 67.04 | 62.20 | 86.09 | 57.92 | 78.49 | - |
| DINOv2-g/14-reg* | 1,137M | 224 | 294 | - | 83.41 | 48.68 | 82.78 | 61.88 | 85.62 | 47.18 | 76.23 | - |
| SAM-H/16* | 637M | 1024 | 12 | - | 22.12 | 28.08 | 34.34 | 49.92 | 81.76 | 43.91 | 57.65 | 77.18 |
| **E-RADIO-L** | 391M | 512 | 468 | 80.73 | 83.89 | 48.22 | 81.64 | 61.70 | 85.07 | 51.47 | 76.73 | 76.31 |
| **RADIO-ViT-H/16** | 653M | 432 | 158 | 82.93 | **86.06** | **51.34** | **84.71** | 63.01 | 86.20 | 56.32 | 79.28 | 76.23 |

Nhận xét của paper: RADIO-ViT-H/16 nhanh gần bằng teacher nhưng vượt trên 6 trong 9 tác vụ; E-RADIO ít nhất nhanh hơn teacher 6x ở độ phân giải khớp. Ở bản text, hàng Intern-ViT-6B bị lẫn nên tôi không chép.

**Thay SAM trực tiếp (Table 8, mIoU COCO / throughput):** SAM Base 75.78 / 50.94, Large 77.02 / 20.62, Huge 77.18 / 11.83; E-RADIO Large 76.31 / 121.74; RADIO với ViTDet-H/16 cửa sổ 8: 76.09 / 29.09, cửa sổ 16: 76.23 / 27.91.

**VQA (LLaVA-1.5 + Vicuna 7B):** thay encoder bằng RADIO; nhờ nhận độ phân giải bất kỳ, resize cạnh dài còn 432px, trung bình 462 token so với 576 token của encoder patch-14 336px (ít hơn 20%).

**Kiểm tra nhận thức 3D (Table 9, NAVI, độ chính xác trung bình qua ngưỡng; depth / surface normals / multi-view corr.):** DFN CLIP-H/14 52.5 / 23.0 / 20.3; OpenAI CLIP-L/14 53.7 / 25.3 / 20.7; DINOv2-g/14-reg 83.2 / 59.6 / 59.9; SAM-H/16 68.2 / 50.3 / 45.3; RADIO-ViT-H/16 81.0 / 58.5 / 62.1.

## 5. Phân tích / ablation

**Kiến trúc student (Table 7; student huấn luyện trên ImageNet-21k bằng OpenCLIP ViT-H/14 + DINOv2 g/14; throughput TensorRT, A100, batch 128, 224²; FD loss = feature distillation loss với DINOv2 teacher, tương quan cao với ADE20k mIoU):**

| Backbone | Tham số | ảnh/s | Zero-shot | k-NN | ADE20k | FD loss |
|---|---|---|---|---|---|---|
| (teacher) DINOv2 G/14 | 1.14B | 313 | n/a | 83.41 | 47.53 | |
| (teacher) OpenCLIP H/14 | 632M | 556 | 77.19 | 81.10 | 40.04 | |
| EfficientNetV2-S | 21M | 9017 | 65.37 | 70.72 | 27.75 | 0.415 |
| ResNetv2-101 | 44M | 7283 | 69.58 | 75.32 | 29.61 | 0.405 |
| RegNetY-064 | 30M | 6573 | 69.84 | 74.59 | 28.9 | 0.394 |
| EfficientViT-L1 | 38M | 6048 | 71.73 | 79.90 | 33.12 | 0.376 |
| ConvNext-B | 88M | 1805 | 75.43 | 81.73 | 38.95 | 0.358 |
| MaxViT-B | 119M | 1486 | 77.49 | 79.34 | 38.46 | 0.340 |
| MViTV2-B | 51M | 975 | 75.92 | 81.39 | 41.39 | 0.345 |
| **E-RADIO-B** | 118M | 6422 | 75.19 | 82.21 | 44.03 | 0.319 |
| E-RADIO-B không upsample | 113M | 7040 | 75.45 | 82.05 | 41.26 | 0.353 |
| **E-RADIO-L** | 265M | 3472 | 77.87 | 83.73 | 45.5 | 0.265 |

Kết luận của tác giả: nhiều backbone "hiệu quả" tụt xa teacher, CNN thuần nhanh hơn ViT nhưng kém chính xác, và hiệu suất kém trên ADE20k vì giảm không gian 32x (feature map $7\times7$ với ảnh 224²). Họ cho rằng nhiều thiết kế bị "over-tuned" cho huấn luyện có giám sát ImageNet-1k nên không scale tốt sang bài toán VFM. FD loss tương quan cao với ADE20k, nghĩa là có thể dùng FD loss làm chỉ báo rẻ cho chất lượng dense.

**Pooling của student (Table 6, ViT, CLS token / average pooling; zero-shot, k-NN, ADE20k, VOC, VQAv2):** CLS: 78.55 / 83.91 / 49.01 / 83.51 / 77.66; average pooling: 80.12 / 83.83 / 38.36 / 77.04 / 78.28. Avgpool tốt hơn cho summary loss nhưng hại mạnh feature theo patch; tác giả chọn CLS riêng.

**CPE (Table 5, ViT-L/14, k-NN / ADE20k):** không CPE 82.96 / 47.30; có CPE 82.84 / 48.52. Dù chỉ train ở 224², CPE khái quát tới 518² và còn vượt DINOv2-g.

**Cân bằng loss (Table 4, ViT-B/14, CLIP+DINOv2; zero-shot / k-NN / ADE20k):** Naive 70.63 / 79.50 / 44.71; Uncertainty 70.92 / 79.37 / 44.57; AdaLoss 71.31 / 79.77 / 44.36. Khác biệt nhỏ nên chọn naive. AMTML-KD (học trọng số teacher) luôn dồn toàn bộ trọng số vào CLIP và tệ hơn cân bằng thủ công.

**Dữ liệu và teacher cho spatial feature (Table 2, 3):** bản text bị lỗi cột nên tôi không chép số. Theo mô tả: ImageNet-1k cho metric ImageNet cao nhất nhưng không dùng vì làm sai lệch "zero-shot"; thêm loss feature cho một teacher luôn có lợi; DINOv2 cho feature không gian tốt hơn CLIP, và khớp cả hai cho kết quả tốt nhất.

**Tóm tắt "Key insights" của paper:** feature distillation đầy đủ (kể cả spatial) giúp tác vụ dày, được mô tả là cải thiện 18% tương đối trên ADE20K so với teacher; teacher tốt hơn thì student tốt hơn; SAM thô khá kém ở tác vụ cấp cao (ADE20k linear probe: DINOv2 hơn SAM 1.7x, k-NN ImageNet hơn 4x).

## 6. Giới hạn

*Do tác giả nêu:* RADIO có hai "chế độ" độ phân giải thấp/cao (Figure 5): ở khoảng 720px sai số MSE giữa đầu DINOv2 của RADIO và DINOv2 thật nhảy vọt kèm đổi hẳn không gian màu, nhiều khả năng do huấn luyện phân vùng giữa CLIP+DINOv2 và SAM; tác giả hứa sửa trong tương lai. Thiết kế adaptor head đơn giản ("leave alternative architectures as future work"). Các cách cân bằng loss phức tạp không giúp rõ.

*Chúng ta nhận thấy:*
- Student là mô hình lớn (653M cho RADIO-ViT-H/16; E-RADIO-L 391M, 265M ở Table 7). Đây không phải nén xuống cỡ nhỏ (<100M); E-RADIO-B còn 118M. Chưa có kết quả cho student rất nhỏ.
- Chi phí huấn luyện cao: 600k bước, 614M ảnh, 32 đến 64 GPU.
- So sánh nhiều kiến trúc ở Table 7 dùng ImageNet-21k và 2 teacher, khác với setup RADIO cuối (DataComp-1B, 3-4 teacher); kết luận về kiến trúc có thể không chuyển nguyên sang cấu hình cuối.
- Kết quả "vượt teacher" tính trên 6/9 tác vụ ở Table 1 chứ không phải mọi tác vụ; ví dụ TextVQA 56.32 thấp hơn OpenAI CLIP-L/14 57.92 và DFN CLIP 56.78; COCO 76.23 thấp hơn SAM-H 77.18.
- Không có đánh giá trên OCR/ảnh tài liệu; TextVQA là thước đo gần nhất và RADIO không dẫn đầu ở đó.

## 7. Liên hệ với các bài khác

- **03 FitNets, 04 Attention transfer:** gốc của feature distillation; AM-RADIO mở rộng sang nhiều teacher và bỏ chuẩn hóa để giữ được độ lớn.
- **12 Relational KD, 13 CRD:** các cách khác để khớp feature; AM-RADIO chỉ dùng cosine + smooth-L1 trực tiếp.
- **21 DeiT, 22 DINO:** DINOv2 (kế thừa DINO) là teacher; DeiT là ví dụ trước về chuyển bias từ convnet sang ViT. Kết quả "CNN-like khó bắt chước ViT" ở đây trái chiều thú vị với DeiT (ViT học từ convnet teacher).
- **24 CLIP-KD, 25 LLaVA-MoD, 26 LLaVA-KD:** nén mô hình thị giác-ngôn ngữ; AM-RADIO là "bên encoder" của bài toán đó.
- **08 Teacher assistant, 41 Capacity gap law:** gap dung lượng teacher-student; ở đây không dùng TA, student đủ lớn.
- **Pruning:** AM-RADIO không dùng pruning; chủ đề liên quan là tìm kiến trúc student hiệu quả bằng thực nghiệm thay vì cắt từ teacher.

## 8. Ứng dụng cho dự án (suy luận)

Đây là bài liên quan trực tiếp nhất tới việc nén ViT 12 block của LightOnOCR.

- **Feature distillation cho ViT bằng cosine + smooth-L1:** có thể dùng nguyên công thức $0.9\cdot\cos+0.1\cdot\text{smooth-}\ell_1$ để khôi phục feature theo patch của ViT đã cắt, với "teacher" là ViT gốc. Smooth-L1 giữ độ lớn vector, quan trọng vì feature này đi vào decoder (projector) vốn quen với thang đo của ViT gốc. Đây là gợi ý có cơ sở từ lập luận "drop-in replacement" của paper, chưa kiểm chứng cho OCR.
- **Adaptor head thay cho ma trận D_lin:** MLP 2 tầng giữa student và teacher đóng vai trò phép ánh xạ phi tuyến nhỏ ở điểm nối; có thể dùng để đo D_res (phần còn lại sau khi một ánh xạ biểu cảm hơn tuyến tính vẫn không bù được). Nếu head MLP có thể khôi phục feature gần hoàn hảo, thì sai lệch là "có thể sửa bằng một lớp", ngược lại thì cấu trúc đã bị mất. Là giả thuyết của chúng ta.
- **Giảm token đầu vào:** RADIO với encoder nhận aspect ratio tùy ý dùng 462 token thay vì 576 (ít hơn 20%) trong LLaVA; ảnh trang tài liệu có thể được hưởng lợi tương tự khi tránh padding vuông, quan trọng cho latency batch-1 và VRAM 8GB (chi phí attention theo số token).
- **Khuyến cáo về kiến trúc student:** Table 7 cho thấy kiến trúc hiệu quả phải giữ độ phân giải không gian cao (giảm 16x thay vì 32x) để học được feature dày. Với OCR (chi tiết nét chữ nhỏ), điều này gợi ý nên cẩn trọng nếu cắt/giảm độ phân giải không gian của ViT; nên ưu tiên cắt chiều ẩn/neuron hay số block thay vì tăng stride, nhưng điều này cần kiểm chứng riêng, vì quan sát của bạn là cắt neuron MLP hồi phục tệ hơn cắt tầng.
- **FD loss làm chỉ báo rẻ:** feature distillation loss tương quan cao với mIoU ở Table 7; trong dự án, MSE/cosine feature giữa ViT teacher/student có thể dùng làm thước đo nhanh để xếp hạng các cấu hình cắt trước khi chạy đánh giá OCR đầy đủ (suy luận; cần tự xác nhận tương quan với CER).
- **Đa teacher:** nếu muốn, ViT nhỏ có thể học đồng thời từ ViT teacher gốc và một teacher thị giác khác (ví dụ DINOv2-like tốt cho chi tiết không gian). Rủi ro: tăng chi phí và nguy cơ student bị kéo về ngữ nghĩa không cần cho OCR.
- **Không liên quan trực tiếp:** zero-shot CLIP và thay thế SAM.

## 9. Thí nghiệm gợi ý

1. **Loss feature cho ViT đã cắt:** so sánh MSE chuẩn hóa, cosine, cosine + smooth-L1 (0.9/0.1) giữa ViT gốc và ViT đã cắt, dùng cùng dữ liệu; đo CER và chỉ số feature. Kiểm tra tương quan giữa FD loss và CER (theo tinh thần Table 7).
2. **Adaptor MLP làm công cụ chẩn đoán:** gắn một MLP 2 tầng (kiểu RADIO) tại điểm cắt, đóng băng student, chỉ huấn luyện MLP để khôi phục feature teacher; so sánh phần dư với phép tuyến tính tối ưu để tách D_lin và D_res (một phần của D_res có phải do phi tuyến?).
3. **Cắt neuron vs cắt block cho ViT:** cùng ngân sách tham số, huấn luyện phục hồi bằng feature distillation kiểu RADIO, so sánh accuracy OCR và FD loss.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Agglomerative | Gộp nhiều teacher khác loại vào một student |
| Adaptor head | MLP 2 tầng (LN, GELU), mỗi teacher 2 đầu (summary, spatial) |
| $L_{summary}$ | $\sum_i\lambda_iL_{cos}$; CLIP và DINOv2 $\lambda=1$, SAM $\lambda=0$ |
| $L_{features}$ | $\sum_i\gamma_i(\alpha L_{cos}+\beta L_{smooth\text{-}\ell_1})$, $\alpha=0.9,\beta=0.1,\gamma_i=1$ |
| Naive balancing | Trọng số bằng nhau; tốt tương đương AdaLoss/Uncertainty |
| CPE | Position embedding $128^2$, cắt ngẫu nhiên + nội suy |
| E-RADIO | Stem conv + C2f + transformer cửa sổ + upsample cuối; giảm không gian 16x |
| Kết quả chính | RADIO-ViT-H/16: k-NN 86.06, ADE20k 51.34, VOC 84.71; E-RADIO-L 468 ảnh/s |
| Phát hiện | CNN thuần nhanh nhưng khó bắt chước ViT VFM; FD loss tương quan với ADE20k |

## 11. Kết luận ngắn

AM-RADIO chứng minh feature distillation đa teacher có thể tạo một encoder nhỏ gọn hơn hoặc ngang các teacher mà vẫn giữ đặc tính riêng của từng teacher (zero-shot, dense, SAM drop-in), với công thức loss đơn giản (cosine + smooth-L1) và adaptor head nhỏ. Bài học trực tiếp cho dự án nén ViT: dùng loss feature giữ cả hướng lẫn độ lớn, đo chất lượng bằng FD loss thay cho đánh giá đắt đỏ, và cẩn thận khi giảm độ phân giải không gian. Cần nhớ student ở đây cỡ trăm triệu đến hơn nửa tỷ tham số và chưa có thử nghiệm OCR.

Link gốc: https://arxiv.org/abs/2312.06709
