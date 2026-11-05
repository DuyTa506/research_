# Accelerating Sparse Deep Neural Networks (2021)

**Tác giả:** Asit Mishra, Jorge Albericio Latorre, Jeff Pool, Darko Stosic, Dusan Stosic, Ganesh Venkatesh, Chong Yu, Paulius Micikevicius (NVIDIA)  
**Venue / arXiv:** arXiv:2104.08378 (v1, 16/04/2021); bài không nêu venue hội nghị.  
**Nguồn:** https://arxiv.org/abs/2104.08378  
**Chủ đề:** Sparsity có cấu trúc mịn 2:4, Sparse Tensor Core trên GPU Ampere, định dạng nén, quy trình "train - prune - retrain", hoán vị kênh (channel permutation).  
**Vị trí trong lộ trình:** Đọc sau 02 (Han 2015: magnitude pruning + retrain), 03 (Zhu & Gupta: pruning dần) và 12 (Blalock: "State of pruning"). Bài này thêm góc nhìn **phần cứng**: sparsity chỉ có giá trị khi GPU thật sự chạy nhanh hơn. Tiếp theo: 19 (HALP, tối ưu latency thực), 22-23 (OBC/SparseGPT, nơi 2:4 được dùng cho LLM), 30 (MaskLLM, học mask 2:4).

> Đây là ghi chú học tập bằng tiếng Việt, không phải bản dịch. Phần "Ứng dụng cho dự án" là suy luận của chúng ta, không phải kết luận của paper.

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

Hai trường phái pruning trước bài này đều có điểm yếu (theo paper):

- **Unstructured (không cấu trúc):** xoá từng trọng số riêng lẻ (Han 2015). Giữ được accuracy ở sparsity cao (cỡ 80-99%) nhưng ma trận thưa ngẫu nhiên khiến truy cập bộ nhớ không đều, cần chỉ số CSR/COO. Paper nói lợi ích tốc độ "negligible and at times negative", đặc biệt trên GPU/TPU có đơn vị nhân ma trận.
- **Structured (có cấu trúc, thô):** xoá cả filter/channel. Chạy nhanh trên phần cứng dense, nhưng mất accuracy khi sparsity cao; ví dụ paper trích: ResNet-50 channel pruning 2x speedup mất gần 1.5% accuracy.

Thêm nữa, các quy trình fine-tune trước đó không thống nhất (số epoch, learning rate khác nhau từng mô hình). Câu hỏi của bài: có thể chọn **một mẫu thưa** vừa đủ "mịn" để giữ accuracy, vừa đủ "đều" để phần cứng tăng tốc, kèm **một quy trình** không cần dò hyper-parameter?

Giải thích thuật ngữ:
- **Sparsity:** tỉ lệ trọng số bằng 0.
- **GEMM** (General Matrix Multiply): phép nhân ma trận, lõi tính toán của conv, linear, attention projection.
- **Tensor Core:** đơn vị phần cứng của GPU NVIDIA chuyên làm nhân-cộng ma trận nhỏ (MMA) rất nhanh.
- **CSR/COO:** định dạng lưu ma trận thưa kèm chỉ số cột/hàng; tốn metadata và truy cập gián tiếp.

## 2. Ý tưởng chính

Ép **mỗi nhóm 4 giá trị liên tiếp có ít nhất 2 số 0** (mẫu "2:4", sparsity đúng 50%). Vì mẫu cố định theo khối, chỉ cần lưu 2 giá trị khác 0 cùng 2 bit chỉ số vị trí mỗi giá trị, và Sparse Tensor Core của Ampere chỉ chọn đúng các phần tử tương ứng của ma trận kia để nhân, nên số phép nhân-cộng giảm một nửa (tối đa 2x throughput). Quy trình rất đơn giản: huấn luyện dense bình thường, cắt 2 trọng số nhỏ nhất trong mỗi nhóm 4, rồi **lặp lại đúng lịch huấn luyện cũ** với mask cố định. Với các mạng rất nhỏ (MobileNet, EfficientNet...) thì hoán vị cột trước khi cắt để giữ được nhiều độ lớn trọng số hơn.

## 3. Phương pháp chi tiết

### 3.1 Mẫu 2:4 và định dạng nén

Với ma trận trọng số $W\in\mathbb R^{R\times C}$ (dùng làm toán hạng đầu của GEMM), chia mỗi hàng thành các nhóm 4 phần tử liên tiếp; ràng buộc

$$\boxed{\forall \text{ nhóm } g=(w_{4k},\dots,w_{4k+3}):\ \|g\|_0\le 2}$$

Ma trận nén có kích thước $R\times C/2$ cho giá trị, cộng $R\times C/2$ metadata 2 bit (vị trí 0-3 của mỗi giá trị trong nhóm). Nếu một nhóm có hơn 2 số 0, vẫn lưu đủ 2 giá trị để định dạng đồng nhất.

Số liệu lưu trữ (paper):
- Overhead metadata: 12.5% với giá trị 16-bit, 25% với 8-bit (so với tới 200% của CSR với giá trị 8-bit).
- 16-bit: dense 4 phần tử = 64 bit; 2:4 = $2\cdot16+2\cdot2=36$ bit, tiết kiệm khoảng 44%.
- 8-bit: tiết kiệm khoảng 38% dung lượng và băng thông.

Vì mật độ thưa đồng nhất ở mọi khối, đọc bộ nhớ dùng trọn cache line, và vị trí phần tử được suy ra trực tiếp từ tỉ lệ nén, không cần truy cập gián tiếp.

### 3.2 Sparse Tensor Core

Phép tính là **sparse × dense = dense**: toán hạng A (M×K) ở dạng 2:4 nén (M×K/2); phần cứng, nhờ metadata, chọn K/2 phần tử khớp của B dense (K×N) rồi nhân. B và C luôn dense. Bảng throughput đỉnh A100 trong paper (TOPS):

| Đầu vào | Accumulator | Dense | Sparse |
|---|---|---|---|
| TF32 | FP32 | 156 | 312 |
| FP16 / BF16 | FP32 | 312 | 624 |
| FP16 | FP16 | 312 | 624 |
| INT8 | INT32 | 624 | 1248 |

(FP32 thuần: 19.5 dense, không có sparse. Với TF32, mẫu thành 1:2.) Thực tế tốc độ phụ thuộc kích thước GEMM: GEMM lớn (arithmetic intensity cao) tiến gần 2x; Hình 3 đo INT8 cuSPARSELt so với cuBLAS với M=N=10240, K thay đổi. Yêu cầu thư viện: chiều trong của ma trận thưa là bội số của 16 (FP16/BF16) hoặc 32 (INT8). Từ TensorRT 8.0 trọng số 2:4 được xử lý tự động.

### 3.3 Quy trình 3 bước (ASP)

```
# Quy trình trong paper (thư viện PyTorch ASP)
1. Huấn luyện dense với lịch (optimizer, lr schedule, số epoch) S
2. Cắt: với mỗi tensor trọng số GEMM-like, trong mỗi nhóm 4,
       đặt 0 cho 2 trọng số có |w| nhỏ nhất  -> mask M cố định
       (tuỳ chọn: hoán vị kênh trước khi cắt)
3. Retrain: khởi tạo = trọng số sau bước 2,
       RESET hyper-parameter và optimizer state (momentum...),
       dùng đúng lịch S như bước 1,
       sau mỗi update: w <- w * M   # giữ nguyên số 0
4. (Nếu dùng INT8) hiệu chuẩn/fine-tune quantization SAU khi đã prune+retrain
```

Điểm chủ chốt: không dò hyper-parameter; đánh đổi chi phí huấn luyện gấp đôi để lấy tính tổng quát (chi phí được "trả dần" qua thời gian triển khai). Paper chỉ thấy tiêu chí magnitude là đủ, và nhắc có thể thử độ tương đồng activation hoặc gradient tức thời.

### 3.4 Hoán vị kênh (channel permutation)

Cắt 2:4 theo khối cố định có thể xoá nhầm giá trị lớn nếu hai giá trị lớn rơi cùng nhóm. Nếu hoán vị các **cột** của $W$ trước khi cắt để phân bố đều giá trị lớn, tổng độ lớn giữ lại tăng. Ví dụ ma trận 4×8 trong Hình 6: tổng độ lớn sau cắt 83.7 (gốc) so với 102.9 (sau hoán vị).

Để kết quả GEMM không đổi, hàng tương ứng của activation $B$ cũng phải hoán vị cùng; điều này đạt được bằng cách hoán vị **hàng của trọng số tạo ra activation đó** (thường là layer trước). Với conv, hoán vị trên chiều kênh đầu vào. Không đổi phép tính, không overhead runtime. Paper nói thuật toán tìm hoán vị tốt "sẽ trình bày riêng", nên bài này không cho chi tiết.

### 3.5 Lớp nào được cắt, mô hình nhiều giai đoạn

- Chỉ cắt lớp có tham số học được và dẫn đến GEMM (conv, fully-connected, recurrent), với chiều trong là bội số của 16/32. Không cắt embedding (là bảng tra), không cắt các lớp chỉ dùng khi train (LM head của BERT, auxiliary classifier của Inception, discriminator của GAN). Conv đầu tiên 3 kênh (7×7) không cắt vì GEMM-K không phù hợp.
- **Mô hình huấn luyện nhiều giai đoạn:** (a) nếu giai đoạn sau dùng dataset nhỏ (BERT: pre-train lớn rồi fine-tune nhỏ), phải cắt và retrain ngay **sau pre-training** rồi mới fine-tune với mask giữ nguyên (Hình 7); (b) nếu giai đoạn sau đủ lớn (detection/segmentation trên COCO), chỉ cần cắt sau giai đoạn 2 và lặp lại giai đoạn 2 (Hình 8); head phải được fine-tune xong trước khi cắt.
- **Kết hợp quantization:** khuyến nghị hiệu chuẩn quantization sau khi đã prune và retrain.

## 4. Thí nghiệm và kết quả chính

Hyper-parameter lấy từ paper gốc/repo công khai của từng mạng; so sánh Dense FP16, Sparse FP16, Sparse INT8.

**ImageNet top-1 (Bảng 2, trích):**

| Mạng | Dense FP16 | Sparse FP16 | Sparse INT8 |
|---|---|---|---|
| ResNet-34 | 73.7 | 73.9 | 73.7 |
| ResNet-50 | 76.1 | 76.2 | 76.2 |
| ResNet-101 | 77.7 | 78.0 | 77.9 |
| ResNeXt-101-32x16 (WSL) | 84.2 | 84.0 | 84.2 |
| DenseNet-121 | 75.5 | 75.3 | 75.3 |
| Wide ResNet-50 | 78.5 | 78.6 | 78.5 |
| Inception v3 | 77.1 | 77.1 | 77.1 |
| VGG-16 | 74.0 | 74.1 | 74.1 |

(Bảng đầy đủ có 20 mạng; chênh lệch nằm trong biến thiên giữa các lần chạy theo nhận xét của tác giả.)

**Mạng nhỏ, tham số hiệu quả (Bảng 3, FP16): mặc định vs hoán vị**

| Mạng | Dense | 2:4 mặc định | 2:4 hoán vị |
|---|---|---|---|
| MobileNet v2 | 71.55 | 69.56 | 71.56 |
| SqueezeNet v1.0 | 58.09 | 54.08 | 58.38 |
| SqueezeNet v1.1 | 58.21 | 56.96 | 58.23 |
| MNASNet 1.0 | 73.24 | 71.99 | 73.27 |
| ShuffleNet v2 | 68.32 | 66.97 | 68.42 |
| EfficientNet B0 | 77.25 | 75.98 | 77.29 |
| EfficientNet-WideSE B0 | 77.63 | 76.64 | 77.63 |

**Detection / segmentation (COCO 2017, Bảng 4, trích):** MaskRCNN-RN50 37.9 / 37.9 / 37.8; RetinaNet-RN50 34.8 / 35.2 / 35.1; FasterRCNN-RN50-FPN-3x 39.8 / 39.9 / 39.4; MaskRCNN-RN101-FPN-3x 42.9 / 43.2 / 42.8 (bbox AP, Dense FP16 / Sparse FP16 / Sparse INT8).

**GAN (FID, thấp hơn là tốt, Bảng 5, dense vs 2:4):** DCGAN-MNIST 50.39 vs 50.54; Pix2Pix 17.64 vs 17.89; CycleGAN Monet→Photo 63.15 vs 63.00; StarGAN-CelebA 6.11 vs 6.93; SRGAN-DIV2K 14.65 vs 16.60 (suy giảm lớn nhất trong bảng).

**NLP:**

| Mô hình | Metric | Dense FP16 | Sparse FP16 | Dense INT8 | Sparse INT8 |
|---|---|---|---|---|---|
| GNMT (WMT'14 En-De) | BLEU | 24.6 | 24.9 | - | 24.9 (INT8) |
| FairSeq Transformer | BLEU | 28.2 | 28.5 | - | 28.3 (INT8) |
| Transformer-XL 12 lớp (enwik8) | BPC | 1.06 | 1.06 | - | - |
| BERT-large (SQuAD v1.1) | F1 | 91.9 | 91.9 | 90.9 | 90.8 |

(Với bảng 6, các cột là Dense FP16 / Sparse FP16 / Sparse INT8.) Với BERT và Transformer-XL: chỉ cắt các GEMM chứa trọng số trong khối transformer; lớp attention (QK^T, AV) không có trọng số nên không cắt.

**Tốc độ:** paper chỉ báo cáo throughput đỉnh (bảng trên) và một benchmark GEMM (Hình 3), không đo latency end-to-end của các mạng ở bảng accuracy.

## 5. Phân tích / ablation

- **Magnitude là đủ** cho đa số mạng; không cần tiêu chí phức tạp (theo paper, có thử các lựa chọn khác như gợi ý, nhưng không có bảng so sánh).
- **Hoán vị kênh** là yếu tố then chốt cho mạng nhỏ: mất 1-4 điểm accuracy ở bảng 3 được lấy lại gần như hoàn toàn.
- **Cắt ở giai đoạn nào** quan trọng: với BERT, chỉ cắt lúc fine-tune SQuAD không đủ số update để phục hồi nên phải lặp lại pre-training thưa.
- **Retrain schedule ngắn hơn:** thí nghiệm sơ bộ tìm được tập hyper-parameter ngắn hơn vẫn giữ accuracy, nhưng phụ thuộc mạnh vào mạng/tác vụ, không có công thức chung (không có số liệu trong bài).
- Phần cứng: lợi ích tốc độ phụ thuộc kích thước GEMM, nên mô hình hidden nhỏ hoặc batch nhỏ sẽ không đạt 2x.

## 6. Giới hạn

**Tác giả nêu:**
- Chi phí huấn luyện gấp đôi (train dense + retrain thưa); tìm lịch ngắn hơn là việc tương lai.
- Chỉ tăng tốc inference (forward). Để tăng tốc cả backward, ma trận chuyển vị cũng phải thoả 2:4, cần mask "transposable".
- Chưa kiểm tra mô hình đã huấn luyện đến giới hạn dung lượng (dataset lớn/lịch dài).
- Attention (không có trọng số) chưa được cắt; thử nghiệm sơ bộ cắt activation 2:4 chưa có phương pháp tổng quát.
- Thuật toán tìm hoán vị chưa trình bày chi tiết.

**Chúng ta nhận xét:**
- Sparsity cố định ở 50%: không phải "nút vặn" để đạt kích thước mục tiêu bất kỳ như 0.3B hay 0.1B.
- Không giảm số tham số "thực" theo nghĩa kiến trúc: tensor vẫn nguyên hình dạng, chỉ nén lưu trữ; bộ nhớ activation và KV/state không đổi.
- Không có số đo end-to-end, nhất là với **batch 1** (lúc đó phép nhân là GEMV, bị giới hạn băng thông chứ không phải tính toán).
- Kết quả "không mất accuracy" trên CNN/BERT lớn không đảm bảo cho mô hình nhỏ đã được distill mạnh.

## 7. Liên hệ với các bài khác

- **02, 03:** cùng khung "train - prune - retrain"; bài này cố định mask một lần (one-shot) và dùng magnitude, khác pruning dần của 03.
- **12 (Blalock):** bài nêu rõ vấn đề so sánh công bằng; 2:4 là ví dụ về việc phải có lợi ích phần cứng đo được.
- **19 (HALP):** cùng coi tốc độ thật là mục tiêu, nhưng HALP làm structured pruning dựa trên bảng latency, còn 2:4 dựa vào hạ tầng phần cứng.
- **22-23 (OBC, SparseGPT):** dùng chính mẫu 2:4 (n:m) cho LLM và thay magnitude bằng cập nhật dựa trên Hessian.
- **24 (Wanda), 30 (MaskLLM):** tiêu chí khác / học mask cho 2:4.
- **44-47 (Nemotron, Minitron...):** các bài này thiên về structured pruning thật, 2:4 là hướng bổ sung độc lập.

## 8. Ứng dụng cho dự án (suy luận)

- **Liên quan hạn chế.** Mục tiêu của dự án là giảm số tham số xuống ~0.3B / ~0.1B bằng structured pruning; 2:4 chỉ cho 50% và không đổi kiến trúc, nên không thay thế.
- **Có thể bổ sung sau cùng:** sau khi prune cấu trúc và distill, thêm 2:4 lên các Linear (MLP, projection của GatedDeltaNet và attention, ViT) có chiều là bội số 16 để tăng tốc thêm, kết hợp INT8/FP8 đúng như khuyến nghị "prune trước, quantize sau". Cần GPU có Sparse Tensor Core (Ampere trở lên); chưa rõ GPU ≤ 8 GB của dự án có phải Ampere không.
- **Cảnh báo batch-1:** decoder sinh token là GEMV, chủ yếu bị giới hạn băng thông bộ nhớ; 2:4 giảm ~38-44% dung lượng/băng thông trọng số, đó có thể là lợi ích chính, chứ không phải 2x tính toán. Prefill của ViT/ảnh lớn mới hưởng 2x tính toán.
- **Nối với D_lin / D_res:** 2:4 giữ nguyên số kênh nên không "cắt" một tầng; thiệt hại chủ yếu là nhiễu trọng số, có thể mô hình bằng lỗi lượng tử-hoá kiểu cục bộ và retrain/distill sửa tốt hơn kiểu cắt neuron (suy đoán, chưa kiểm).
- **Lớp tuyến tính của GatedDeltaNet:** paper không nói gì về linear attention; cần kiểm tra xem các ma trận state/gate nhỏ có nhạy với 2:4 không.

## 9. Thí nghiệm gợi ý

1. Lấy một block MLP của decoder (hoặc ViT), ép 2:4 bằng magnitude, đo lỗi đầu ra (MSE so với dense) có và không có hoán vị kênh; so với cắt 50% neuron trung gian để xem lỗi nào lớn hơn, rồi distill ngắn xem cái nào hồi phục tốt hơn.
2. Benchmark latency batch-1 thực trên GPU đích: Linear dense vs 2:4 (cuSPARSELt/TensorRT) cho kích thước hidden của mô hình; xác nhận lợi ích có đến từ băng thông hay không.
3. Áp 2:4 chỉ lên ViT hoặc chỉ lên decoder, đo CER/WER OCR trước và sau distill từ teacher 4B.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| 2:4 sparsity | Mỗi nhóm 4 giá trị liên tiếp có ≥ 2 số 0, sparsity 50% |
| Định dạng nén | $R\times C/2$ giá trị + 2 bit chỉ số/giá trị; overhead 12.5% (16b), 25% (8b) |
| Tiết kiệm lưu trữ | ~44% (16-bit: 36 vs 64 bit/nhóm), ~38% (8-bit) |
| Tăng tốc phần cứng | Tối đa 2x throughput GEMM (A100: FP16 312 → 624 TOPS) |
| Quy trình | Train dense → cắt 2 nhỏ nhất/nhóm → retrain cùng lịch, mask cố định |
| Reset khi retrain | Hyper-parameter và optimizer state được reset |
| Channel permutation | Hoán vị cột W (và hàng của layer trước) để giữ tổng độ lớn; 83.7 → 102.9 trong ví dụ |
| Không cắt | Embedding, layer chỉ dùng khi train, conv 3 kênh đầu, chiều không là bội 16/32 |
| Quantization | Hiệu chuẩn sau khi prune + retrain |

## 11. Kết luận ngắn

Đóng góp chính của bài là kỹ thuật + hệ thống: một mẫu thưa vừa đủ mịn để giữ accuracy trên rất nhiều mạng, vừa đủ đều để Ampere nhân nhanh gấp đôi, cùng quy trình train-prune-retrain "không dò tham số". Với dự án này, nó là kỹ thuật nén cuối pipeline có điều kiện (phần cứng Ampere+, kích thước GEMM đủ lớn), không phải công cụ giảm kích thước kiến trúc.

Link gốc: https://arxiv.org/abs/2104.08378
