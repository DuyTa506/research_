# Rethinking the Value of Network Pruning (2018)

**Tác giả:** Zhuang Liu, Mingjie Sun, Tinghui Zhou, Gao Huang, Trevor Darrell (UC Berkeley; Tsinghua University).  
**Venue / arXiv:** arXiv:1810.05270 (v1 11/10/2018; bản dùng ở đây v2, 05/03/2019); ICLR 2019.  
**Nguồn:** https://arxiv.org/abs/1810.05270  
**Chủ đề:** Với structured pruning, mô hình cắt rồi fine-tune từ trọng số kế thừa chỉ ngang hoặc kém hơn *train mô hình cùng cấu trúc từ khởi tạo ngẫu nhiên* (train from scratch); giá trị thực của cắt có thể là *tìm kiến trúc* (số kênh mỗi layer) chứ không phải chọn "trọng số quan trọng". Có so sánh với Lottery Ticket.  
**Vị trí trong lộ trình:** Đọc sau 04–06 (filter pruning, slimming, ThiNet: đối tượng bị phản biện), 08–10 (LTH, SNIP). Bài này là điểm phản biện quan trọng nhất của nhánh CNN; nó cũng đặt ra câu hỏi mà dự án phải trả lời: khi nào nên "cắt rồi phục hồi" thay vì "train lại từ đầu" (xem 12 Blalock, 43 Pruning vs scratch).

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Số liệu lấy từ paper (một vài bảng PDF bị lệch dòng, tôi ghi rõ khi nghi ngờ). Phần "Ứng dụng cho dự án" là suy luận của chúng ta.

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

Quy trình cắt tỉa điển hình có **ba giai đoạn**: (1) train mô hình lớn quá tham số hóa; (2) cắt theo một tiêu chí; (3) fine-tune mô hình đã cắt để lấy lại độ chính xác. Đằng sau là hai niềm tin phổ biến:

1. **Phải bắt đầu từ mô hình lớn.** Mô hình lớn mạnh hơn (về biểu diễn và tối ưu hóa), nên cắt từ nó tốt hơn là train mô hình nhỏ từ đầu; nhiều bài (Li 2017, Luo 2017 tức ThiNet bài 06, He 2017) báo cáo train-from-scratch là baseline yếu hơn.
2. **Cả kiến trúc lẫn trọng số kế thừa đều quan trọng.** Do đó cả lĩnh vực tập trung vào việc chọn đúng "trọng số quan trọng" (bài 04, 05, 06, 07...).

Khái niệm nền:
- **Structured pruning**: cắt nguyên kênh/filter/khối; mô hình nhỏ vẫn dày đặc nên chạy nhanh với thư viện thường.
- **Predefined target architecture**: người dùng định tỷ lệ cắt mỗi layer (ví dụ cắt 50% kênh mỗi layer); thuật toán chỉ chọn *kênh nào*, kiến trúc cuối biết trước.
- **Automatic target architecture**: tiêu chí so sánh toàn cục giữa các layer (ví dụ Network Slimming, bài 05), nên thuật toán tự quyết định số kênh mỗi layer; cắt thưa từng trọng số cũng thuộc loại này (Figure 2).
- **Train from scratch**: khởi tạo ngẫu nhiên (He init) mô hình nhỏ và train từ đầu.

## 2. Ý tưởng chính

Tác giả chạy lại các phương pháp cắt có cấu trúc tiên tiến theo đúng kiến trúc và dữ liệu của bài gốc, rồi thêm một đường cơ sở công bằng: lấy kiến trúc đã cắt, **khởi tạo ngẫu nhiên và train từ đầu** với ngân sách tính toán tương đương. Kết quả gây ngạc nhiên: trong hầu hết trường hợp, train từ đầu bằng hoặc tốt hơn fine-tune từ trọng số kế thừa. Hai hệ quả: (a) với cắt có kiến trúc định sẵn, bỏ cả quy trình, train thẳng mô hình đích; (b) với cắt tự động, điều có giá trị nằm ở *kiến trúc* được tìm ra (số kênh mỗi layer) chứ không ở các trọng số "quan trọng". Đây là cách nhìn "cắt = tìm kiến trúc". Ngoại lệ: cắt thưa không cấu trúc ở ImageNet.

## 3. Phương pháp chi tiết

Paper không có công thức mới; đóng góp là phương pháp đánh giá.

### 3.1 Ngân sách huấn luyện từ đầu

Train mô hình nhỏ cùng số epoch với mô hình lớn có thể thiệt thòi (mỗi epoch rẻ hơn). Hai biến thể:
- **Scratch-E**: cùng *số epoch* với mô hình lớn.
- **Scratch-B**: cùng *ngân sách tính toán* (FLOPs) với mô hình lớn; trên ImageNet, nếu mô hình cắt tiết kiệm >2x FLOPs thì chỉ nhân đôi số epoch (nên vẫn ít tính toán hơn mô hình lớn). Khi tăng epoch, lịch giảm lr được co giãn tương ứng.

Tính toán của bước fine-tune **không** được cộng vào ngân sách (vì ngắn), điều này theo tác giả vẫn là có lợi cho quy trình cắt+fine-tune.

### 3.2 Thiết lập

- Dữ liệu: CIFAR-10, CIFAR-100, ImageNet; mạng VGG, ResNet, PreResNet, DenseNet.
- Phương pháp được kiểm tra: 4 *predefined* (Li 2017 L1-norm filter pruning, Luo 2017 ThiNet, He 2017b Regression-based feature reconstruction, He 2018a Soft Filter Pruning), 2 *automatic* (Liu 2017 Network Slimming, Huang & Wang 2018 Sparse Structure Selection), 1 không cấu trúc (Han 2015). Với 6 phương pháp đầu dùng đúng cặp (mô hình, dữ liệu) của bài gốc; CIFAR chạy 5 seed, báo trung bình và độ lệch chuẩn.
- Triển khai: dùng mã gốc nếu có; nếu không, tự cài lại quy trình 3 giai đoạn (kết quả tương tự); với ThiNet và He 2017b chỉ có mô hình đã cắt công khai, nên train lại cả mô hình lớn và nhỏ (mô hình lớn của họ chính xác hơn bản gốc nhờ PyTorch so với Caffe, vì vậy báo cáo **độ giảm tương đối** so với mô hình không cắt).
- Huấn luyện: SGD Nesterov momentum, lr giảm bậc thang, He init. CIFAR: train/fine-tune 160/40 epoch; ImageNet: 90/20 epoch. Fine-tune dùng lr nhỏ nhất của lúc train mô hình lớn.

## 4. Thí nghiệm và kết quả chính

Tất cả là độ chính xác (%), cao hơn tốt hơn.

### 4.1 Cắt có cấu trúc định sẵn

**L1-norm filter pruning (Li 2017), Table 1**

| Dữ liệu | Mô hình (không cắt) | Mô hình cắt | Fine-tuned | Scratch-E | Scratch-B |
|---|---|---|---|---|---|
| CIFAR-10 | VGG-16 93.63 | VGG-16-A | 93.41 | 93.62 | 93.78 |
| | ResNet-56 93.14 | ResNet-56-A | 92.97 | 92.96 | 93.09 |
| | | ResNet-56-B | 92.67 | 92.54 | 93.05 |
| | ResNet-110 93.14 | ResNet-110-A | 93.14 | 93.25 | 93.22 |
| | | ResNet-110-B | 92.69 | 92.89 | 93.60 |
| ImageNet | ResNet-34 73.31 | ResNet-34-A | 72.56 | 72.77 | 73.03 |
| | | ResNet-34-B | 72.29 | 72.55 | 72.91 |

Trên ImageNet cả hai Scratch-B hơn fine-tuned rõ rệt.

**ThiNet (Luo 2017), Table 2** (độ giảm tương đối so với mô hình không cắt; không cắt VGG-16: 71.03 theo bài gốc, 71.51 trong cài đặt này; ResNet-50: 75.15 và 76.13):

| | VGG-Conv | VGG-GAP | VGG-Tiny |
|---|---|---|---|
| Fine-tuned | −1.23 | −3.67 | −11.61 |
| Scratch-E | −2.75 | −4.66 | −14.36 |
| Scratch-B | +0.21 | −2.85 | −11.58 |

| | ResNet50-30% | ResNet50-50% | ResNet50-70% |
|---|---|---|---|
| Fine-tuned | −6.72 | −4.13 | −3.10 |
| Scratch-E | −5.21 | −2.82 | −1.71 |
| Scratch-B | −4.56 | −2.23 | −1.01 |

Ngoại lệ duy nhất: Scratch-E cho VGG-Tiny (cắt rất mạnh, giảm 15x FLOPs, nên ngân sách của Scratch-E bị rút xuống rất nhiều). Scratch-B của VGG-Tiny cũng chỉ có ngân sách nhỏ hơn 7 lần mô hình gốc nhưng vẫn ngang fine-tuned.

**Regression-based feature reconstruction (He 2017b), Table 3** (không cắt VGG-16 71.03/71.51; ResNet-50 75.51/76.13):

| | VGG-16-5x | ResNet-50-2x |
|---|---|---|
| Fine-tuned | −2.67 | −3.25 |
| Scratch-E | −3.46 | −1.55 |
| Scratch-B | −0.51 | −1.07 |

(Với VGG-16-5x, Scratch-E kém hơn fine-tuned; Scratch-B tốt hơn rõ rệt.)

Soft Filter Pruning (Phụ lục A): Scratch-E hơn mô hình cắt phần lớn thời gian, Scratch-B hơn trong gần như mọi trường hợp.

### 4.2 Cắt tự động có cấu trúc

**Network Slimming (Liu 2017), Table 4:** train from scratch ngang fine-tuned ở mọi mạng; Scratch-B hơn trong 8/10 thí nghiệm. Ví dụ CIFAR-10 VGG-19 (không cắt 93.53, cắt 70%): fine-tuned 93.60, Scratch-E 93.30, Scratch-B 93.81; CIFAR-100 VGG-19 (72.63, cắt 50%): 72.32 / 71.94 / 73.08; ImageNet VGG-11 (70.84, cắt 50%): 68.62 / 70.00 / 71.18.

**Sparse Structure Selection (Huang & Wang), cắt khối residual, Table 5:** ImageNet ResNet-50 (76.12) cắt thành ResNet-41/32/26; "pruned" (không fine-tune) / Scratch-E / Scratch-B: ResNet-41 75.44 / 75.61 / 76.17; ResNet-32 74.18 / 73.77 / 74.67; ResNet-26 71.82 / 72.55 / 73.41.

### 4.3 Cắt thưa không cấu trúc (Han 2015), Table 6

- CIFAR: ở tỷ lệ cắt ≤80%, Scratch-E đôi khi kém fine-tuned nhưng Scratch-B ít nhất ngang. Ở 95%, fine-tune đôi khi hơn train-from-scratch (ví dụ CIFAR-10 DenseNet-BC-100 95%: 94.19 fine-tuned vs 92.91 Scratch-E vs 93.44 Scratch-B).
- **ImageNet: Scratch-B thường kém fine-tuned đáng kể**: VGG-16 (73.37) cắt 60%: 73.63 vs 71.50 vs 73.42; cắt 30%: 73.68 / 72.75 / 74.02. ResNet-50 (76.15): cắt 30%: 76.06 / 74.77 / 75.70; cắt 60%: 76.09 / 73.69 / 74.91.
- Tác giả cho nguyên nhân có thể là mạng rất thưa khó train trực tiếp, độ phức tạp của ImageNet, và cắt không cấu trúc làm đổi phân bố trọng số mạnh hơn (Phụ lục F). Đây là khác biệt quan trọng giữa hai loại cắt.

### 4.4 Mô hình bị cắt rất mạnh (Phụ lục C)

Train from scratch càng hơn nhiều khi tỷ lệ cắt cao. Network Slimming (Table 13): PreResNet-164/CIFAR-10 (95.04) cắt 80%: fine-tuned 91.76, Scratch-E 93.21, Scratch-B 93.49; cắt 90%: 82.06 / 87.55 / 88.44. L1-norm ResNet-56 (93.14) cắt 90% (Table 14): 89.17 / 91.02 / 91.93.

### 4.5 Chuyển sang phát hiện đối tượng (Phụ lục B)

PASCAL VOC 07, Faster-RCNN, backbone ResNet-34-A/B từ Li 2017: mô hình train from scratch trên phân loại rồi chuyển sang detection vượt mô hình fine-tuned. Ví dụ ResNet34-B (mAP): Prune-C 70.84, Prune-D 69.62, Scratch-E 71.68, Scratch-B 71.26 (không cắt ResNet-34: 71.69). Prune-C là cắt trên backbone phân loại, Prune-D là cắt sau khi chuyển sang detection; bất ngờ là Prune-C hơn Prune-D, tác giả đoán cắt sớm giúp tránh kẹt vào cực tiểu xấu do kế thừa trọng số.

## 5. Phân tích / ablation

- **Fine-tune thêm epoch không giúp** (Phụ lục D, Table 16): Fine-tune-40/80/160 cho độ chính xác gần như nhau (ví dụ VGG-16-A: 93.40/93.45/93.45 so với Scratch-E 93.62). Nên Scratch không thắng chỉ do fine-tune quá ít.
- **Kéo dài lịch huấn luyện** (Phụ lục E, 160 → 300 epoch CIFAR, Table 17): kết luận vẫn giữ, scratch ít nhất ngang fine-tuned.
- **Cắt = tìm kiến trúc (Mục 5):** so kiến trúc do cắt tự động tìm ra với kiến trúc cắt đều (cùng tỷ lệ mọi layer), cả hai train from scratch cùng số epoch.
  - VGG-16 trên CIFAR-10 (Figure 3 trái): kiến trúc Network Slimming đạt cùng độ chính xác với **ít hơn 5 lần tham số** so với cắt đều. Cắt thưa so với thưa đều cũng cho kết quả tương tự (Figure 3 phải).
  - Mẫu cắt của VGG khá ổn định giữa các lần chạy (Table 7: cắt 60% kênh, tổng chiều rộng 4224 → 1689.2; layer 1–7 gần như giữ nguyên, layer sâu 9–13 bị cắt mạnh, ví dụ layer 10 từ 512 còn 63.0±1.9).
  - **Không phải lúc nào cũng thắng cắt đều:** với PreResNet-164 (CIFAR-10), PreResNet-110 và DenseNet-40 (CIFAR-100), kiến trúc cắt không hiệu quả hơn đáng kể so với cắt đều (Figure 5). Các mẫu cắt này gần đều giữa các stage (Phụ lục G). VGG có độ dư thừa mất cân bằng giữa các stage nên cắt giúp ích.
  - **Nguyên tắc thiết kế chuyển giao được:** lấy số kênh trung bình theo stage từ kiến trúc đã cắt để thiết kế ("Guided Pruning/Sparsification") cho hiệu quả ngang kiến trúc cắt trực tiếp; mẫu rút từ VGG-16/CIFAR-10 áp sang VGG-19/CIFAR-100 ("Transferred Guided") vẫn tốt hơn cắt đều đáng kể (Figure 6).
  - So với NAS (RL, tiến hóa): cắt chỉ cần một lần huấn luyện, nhưng không gian tìm kiếm chỉ gồm các mạng con của mạng lớn.
- **So với Lottery Ticket (Mục 6):** tác giả liệt kê bốn khác biệt cài đặt so với Frankle & Carbin: (a) họ chỉ cắt thưa không cấu trúc; (b) họ dùng mạng nhỏ nông (<6 layer); (c) họ dùng Adam/lr rất thấp, còn bài này dùng SGD momentum lr đầu 0.1 chuẩn; (d) họ không có ImageNet. Thí nghiệm (Figure 7, Table 8, 9): với lr 0.1, "winning ticket" (khởi tạo gốc) **không** hơn khởi tạo ngẫu nhiên, cả với cắt có cấu trúc và không cấu trúc; chỉ khi lr nhỏ (0.01) cho cắt thưa trên CIFAR thì winning ticket mới hơn, nhưng lr nhỏ cho độ chính xác thấp hơn lr chuẩn. Ví dụ cắt cấu trúc L1-norm, lr 0.1: VGG-16-A winning ticket 93.62 vs random 93.60; ResNet-56-B 92.78 vs 92.90. Cắt thưa one-shot VGG-16, lr 0.1, cắt 95%: 93.18 vs 93.05.

## 6. Giới hạn

**Do tác giả nêu:**
- Kết luận chủ yếu cho structured pruning; với cắt thưa không cấu trúc trên ImageNet, train from scratch kém hơn (mục 4.3).
- Cắt+fine-tune vẫn đáng giá trong hai trường hợp: (a) đã có mô hình lớn sẵn và ít hoặc không có ngân sách huấn luyện (và phương pháp không đòi sửa quá trình train mô hình lớn); (b) cần nhiều mô hình ở nhiều kích cỡ, hoặc chưa biết kích cỡ mong muốn, khi đó train một mô hình lớn rồi cắt theo nhiều tỷ lệ.
- Không phải mọi kiến trúc đều hưởng lợi từ kiến trúc cắt (ResNet/DenseNet gần đều).
- Một số mâu thuẫn với tài liệu gốc có thể do siêu tham số/augmentation kém được chọn kỹ và ngân sách tính toán không công bằng ở các baseline.

**Do chúng tôi nhận xét:**
- Toàn bộ thí nghiệm là CNN phân loại ảnh (và một thí nghiệm detection); không có mô hình ngôn ngữ, VLM, hay quy mô tỷ tham số, nơi train-from-scratch tốn hàng nghìn lần tính toán.
- Scratch-B dùng ngân sách lớn (bằng mô hình lớn), nên "train từ đầu" ở đây không rẻ; phù hợp nếu train là rẻ (CIFAR), không phù hợp khi tiền huấn luyện là phần chi phí chính.
- Kết luận phụ thuộc vào recipe huấn luyện chuẩn (SGD lr 0.1, 160 epoch...); các bài sau (Blalock 12) lưu ý rằng so sánh cắt phụ thuộc mạnh vào cài đặt.
- Dữ liệu huấn luyện cho mô hình nhỏ ở đây giống mô hình lớn; trong chưng cất, mô hình nhỏ còn có tín hiệu từ giáo viên, mà paper không xét.

## 7. Liên hệ với các bài khác

- **04 Pruning filters, 05 Network slimming, 06 ThiNet:** ba trong các đối tượng được kiểm chứng (Table 1, 4, 2). Với ThiNet, mục 4.4 của bài 06 ("cắt+fine-tune hơn train from scratch") bị phản biện bằng ngân sách đủ lớn (Scratch-B).
- **07 Molchanov:** bài 07 tuyên bố không dính "pruning paradox" của bài này (ResNet20/CIFAR-10); xem lại để thấy hai bên đồng ý rằng khác biệt nằm trong độ lệch chuẩn.
- **08, 09 LTH:** Mục 6 của bài này là bên phản biện LTH; bài 09 sau đó giải thích mâu thuẫn bằng độ ổn định trước nhiễu SGD: ở mức thưa tầm thường train lại từ khởi tạo ngẫu nhiên là đủ, điều này khớp phần "thưa vừa phải" của bài 11.
- **10 SNIP:** cũng chọn cấu trúc thưa trước huấn luyện, ủng hộ cách nhìn "cấu trúc quan trọng hơn trọng số".
- **12 State of pruning (Blalock):** hệ thống hóa vấn đề baseline và thiếu chuẩn so sánh mà bài này gợi ra.
- **38 Sheared LLaMA, 39–40 Minitron, 43 Pruning vs scratch:** bản LLM của câu hỏi này; kết quả ở đó cho rằng cắt + tiếp tục huấn luyện thường tiết kiệm token rất lớn so với train từ đầu, khác bối cảnh CNN ở đây.
- **42 Puzzle, 41 Flextron:** tìm kiến trúc từ mô hình lớn, đúng tinh thần "cắt = tìm kiến trúc".

## 8. Ứng dụng cho dự án (suy luận)

**Liên quan cao về mặt phương pháp luận, thấp về kết quả cụ thể.**

- **Câu hỏi cần trả lời trước khi cắt:** "kiến trúc cuối cùng (số layer, số neuron mỗi MLP, số head...) sẽ là gì?" Nếu nó đã định trước (0.3B/0.1B), lập luận của bài này nói rằng *trọng số kế thừa* có thể không quan trọng, chỉ kiến trúc; nhưng trong bối cảnh LLM/VLM, train từ đầu mô hình 0.1B trên dữ liệu OCR cần ngân sách rất lớn nếu không có tiền huấn luyện tổng quát. Đây là khác biệt cốt lõi với CIFAR/ImageNet nhỏ (giới hạn của chúng tôi ở mục 6).
- **Đường cơ sở bắt buộc phải có:** theo tinh thần bài này, mọi so sánh "cắt + chưng cất" nên có hai baseline cùng kiến trúc, cùng ngân sách: (i) khởi tạo ngẫu nhiên + chưng cất từ teacher 4B; (ii) khởi tạo từ trọng số đã cắt + chưng cất. Nếu (i) ≈ (ii) thì giá trị của dự án nằm ở kiến trúc và chưng cất, không ở trọng số kế thừa.
- **Cắt = tìm kiến trúc** gợi ý dùng điểm quan trọng để quyết định *độ rộng/độ sâu mỗi khối* (kết quả: VGG có mẫu cắt ổn định, nhất quán giữa các lần chạy), rồi có thể khởi tạo lại. Nếu quan sát D_lin/D_res cho thấy cắt layer rẻ hơn cắt neuron MLP, thì "mẫu phân bố" nào tốt nhất cũng là một câu hỏi kiến trúc.
- **Liên hệ với quan sát của dự án (giả thuyết):** "cắt neuron MLP phục hồi tệ hơn cắt layer" có thể một phần vì trọng số kế thừa của MLP đã bị cắt xén vẫn "kẹt" vào cực tiểu xấu (paper nói trọng số kế thừa "might trap the pruned model into a bad local minimum"); kiểm tra bằng baseline khởi tạo lại phần MLP bị thu hẹp.
- **Điều kiện để cắt+tinh chỉnh vẫn thắng:** ngân sách rất hạn chế, đã có mô hình lớn, cần nhiều kích cỡ (0.3B và 0.1B từ cùng gốc): chính xác là tình huống của dự án; vì vậy cắt + chưng cất vẫn hợp lý.
- **Bài học thực nghiệm:** báo cáo độ chính xác theo ngân sách tính toán đồng nhất (Scratch-B so với Scratch-E); cẩn thận với recipe lr/epoch khi so sánh.

## 9. Thí nghiệm gợi ý

1. **Scratch-vs-Pruned cho khối nhỏ:** chọn một khối decoder bị cắt MLP 50%; huấn luyện khối đó (đóng băng phần còn lại) bằng chưng cất trong cùng số bước từ (a) trọng số đã cắt, (b) khởi tạo ngẫu nhiên; đo MSE khối và CER OCR.
2. **Cắt đều vs cắt do điểm quyết định:** cùng ngân sách tham số, so kiến trúc có độ rộng MLP đều giữa các block với kiến trúc theo điểm quan trọng (ít cắt ở block đầu, nhiều ở block sâu hoặc ngược lại), cả hai khởi tạo từ trọng số đã cắt rồi chưng cất.
3. **Ngân sách đồng nhất:** cho mọi baseline cùng số token chưng cất; báo cáo thêm đường "cùng FLOPs huấn luyện" như Scratch-B.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Pipeline 3 giai đoạn | Train mô hình lớn, cắt, fine-tune |
| Predefined vs automatic | Kiến trúc đích do người định vs do thuật toán cắt toàn cục quyết định |
| Scratch-E | Train từ đầu cùng số epoch với mô hình lớn |
| Scratch-B | Train từ đầu cùng ngân sách FLOPs (ImageNet: nhân đôi epoch nếu tiết kiệm >2x) |
| Độ giảm tương đối | Độ chính xác trừ độ chính xác mô hình không cắt (cùng framework) |
| "Cắt = tìm kiến trúc" | Giá trị của cắt tự động là số kênh mỗi layer, không phải trọng số kế thừa |
| Guided / Transferred Guided | Dùng mẫu kênh trung bình theo stage làm quy tắc thiết kế, có thể chuyển sang VGG/dữ liệu khác |
| Kết luận LTH | Với lr 0.1 (chuẩn), khởi tạo winning ticket không hơn khởi tạo ngẫu nhiên |
| Ngoại lệ | Cắt thưa không cấu trúc trên ImageNet: fine-tune thường hơn scratch |

## 11. Kết luận ngắn

Với cắt có cấu trúc trên CNN, mô hình cắt rồi fine-tune thường không hơn (và hay kém hơn) cùng kiến trúc train từ đầu với ngân sách tương đương; nên giá trị của cắt nằm ở việc tìm kiến trúc hiệu quả (nhất là trên VGG), và ở việc tiết kiệm khi đã có sẵn mô hình lớn hoặc cần nhiều kích cỡ. Với dự án, bài học là luôn đặt baseline "cùng kiến trúc, khởi tạo lại, cùng ngân sách chưng cất", nhưng không suy ra từ CIFAR/ImageNet rằng train từ đầu mô hình 0.1B là rẻ hoặc đủ tốt.

Link gốc: https://arxiv.org/abs/1810.05270
