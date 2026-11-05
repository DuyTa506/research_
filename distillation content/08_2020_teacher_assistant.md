# Improved Knowledge Distillation via Teacher Assistant (2020)

**Tác giả:** Seyed Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Nir Levine, Akihiro Matsukawa, Hassan Ghasemzadeh.  
**Venue / arXiv:** arXiv:1902.03393 (v2, 17/12/2019); bản trong file mang thông báo bản quyền AAAI 2020 (AAAI Conference on Artificial Intelligence).  
**Nguồn:** https://arxiv.org/abs/1902.03393  
**Chủ đề:** Khoảng cách dung lượng (capacity gap) giữa teacher và student; chưng cất nhiều bước qua mạng trung gian "teacher assistant" (TA); phân tích lý thuyết dựa trên VC và cảnh quan loss.  
**Vị trí trong lộ trình:** Đọc trước: 02 (Hinton KD), 06–07 (BAN, DML: teacher không cần mạnh hơn student). Bài này thêm: ngược lại với "teacher càng lớn càng tốt", chứng minh có điểm quá lớn thì student học tệ đi, và đề xuất bắc cầu bằng các mạng cỡ trung. Đọc tiếp: 09 (Efficacy of KD: giải thích vì sao teacher quá tốt có thể hại), 10 (Does KD really work), 41 (Capacity gap law cho LLM).

> Đây là ghi chú học tập bằng tiếng Việt, diễn giải và tóm lược, không phải bản dịch paper. Số liệu lấy từ paper; chỗ nào bảng bị lộn trong file trích xuất thì ghi rõ. Phần "Ứng dụng cho dự án" và các nhận xét ghi là suy luận là của chúng ta.

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

KD chuẩn (Hinton 2015) coi như: có teacher lớn đã train, student nhỏ cố định, student học nhãn thật cộng soft target của teacher. Ngầm hiểu: teacher càng mạnh càng tốt. Bài này kiểm tra điều đó và thấy sai khi **khoảng cách kích thước** quá lớn. Điều này có ý nghĩa thực tế: trong thực tế student thường bị cố định bởi thiết bị (ví dụ 8 GB) và teacher cũng cố định (model có sẵn).

Khái niệm cần có:
- **Baseline KD (BLKD):** loss $= (1-\lambda)\,\text{CE}(\text{nhãn}) + \lambda\,\tau^2\,\text{KL}$ giữa softmax làm mềm của student và teacher với **temperature $\tau$** (chia logit cho $\tau$ để phân phối bớt nhọn, lộ rõ các lớp phụ).
- **NOKD:** train student từ đầu, không distillation.
- **Capacity / size** (proxy trong bài): số lớp conv với CNN thường, số block với ResNet.

## 2. Ý tưởng chính

Thí nghiệm cơ bản: cố định student CNN 2 lớp conv, thử teacher 4, 6, 8, 10 lớp; teacher càng lớn thì accuracy của chính nó càng cao, nhưng accuracy của student **tăng rồi giảm**. Giải thích bằng ba lực cạnh tranh. Đề xuất **TAKD**: chen một hoặc nhiều mạng cỡ trung (teacher assistant) vào giữa. TA học từ teacher, student học từ TA. Vì TA gần student hơn nên student bắt chước dễ hơn, và target của TA mềm hơn (bớt "chắc chắn quá"). Tác giả cũng đưa ra phân tích lý thuyết (đường cận lỗi VC) và minh họa cảnh quan loss phẳng hơn.

## 3. Phương pháp chi tiết

### 3.1 Loss

$$L_{SL}=H(\mathrm{softmax}(a_s),y_r),\qquad L_{KD}=\tau^2\,\mathrm{KL}\big(\mathrm{softmax}(a_s/\tau),\,\mathrm{softmax}(a_t/\tau)\big)$$

$$\boxed{L_{student}=(1-\lambda)L_{SL}+\lambda L_{KD}}$$

(Ký hiệu $a_s,a_t$ là logit của student và teacher; $\lambda$ cân bằng hai loss; paper viết KL theo thứ tự đối số như trên.)

### 3.2 Ba yếu tố cạnh tranh khi teacher lớn lên

1. Teacher dự đoán tốt hơn nên giám sát tốt hơn (có lợi).
2. Teacher quá phức tạp, student **không đủ dung lượng hay cơ chế** để bắt chước (có hại).
3. Teacher **tự tin hơn** nên logit/soft target bớt mềm, tín hiệu lớp phụ yếu đi (có hại).

Ban đầu yếu tố 1 thắng, sau đó 2 và 3 thắng. Cũng đúng chiều đối ngẫu: cố định teacher 10 lớp và thu nhỏ student, lợi ích của KD so với NOKD cũng tăng rồi giảm (Hình 3).

### 3.3 TAKD

Bước 1: distill teacher $T$ sang TA. Bước 2: dùng TA như teacher để distill sang student $S$. Mở rộng: chuỗi nhiều TA, gọi là **đường chưng cất (distillation path)**, ví dụ $10\to 8\to 6\to 4\to 2$ (số là cỡ mạng).

```
Input: teacher T (đã train), student S (cố định), danh sách cỡ TA c_1 > c_2 > ... > c_m  (giữa cỡ S và cỡ T)
prev = T
for TA_i in [TA_1 .. TA_m]:                 # cỡ giảm dần
    TA_i = train(arch(c_i), loss=BLKD, teacher=prev)   # (1-λ)CE + λ τ^2 KL
    prev = TA_i
S = train(arch(S), loss=BLKD, teacher=prev)
```

Mọi bước dùng cùng công thức BLKD; hyperparameter ($\lambda$, $\tau$) được tìm bằng công cụ tối ưu hyperparameter (Neural Network Intelligence, thuật toán TPE), 120 lần thử cho mỗi thí nghiệm, và con số báo cáo là top-1 test accuracy tốt nhất tìm được (lưu ý ở mục Giới hạn).

### 3.4 Phân tích lý thuyết (tóm tắt)

Dựa trên VC theory: lỗi của một hàm $f_s$ so với hàm thật $f_r$ bị chặn bởi sai số ước lượng $O\big(|F_s|_C/n^{\alpha_{sr}}\big)$ cộng sai số xấp xỉ $\epsilon_{sr}$; $|F|_C$ là độ đo dung lượng lớp hàm, $n$ số mẫu, $\tfrac12\le\alpha\le1$ là **tốc độ học** (gần $\tfrac12$ với bài khó, gần 1 với bài dễ). Với student học từ nhãn thật (NOKD), student học từ teacher (BLKD), và student qua TA (TAKD), cận trên lần lượt là:

$$\text{NOKD:}\ O\!\Big(\tfrac{|F_s|_C}{n^{\alpha_{sr}}}\Big)+\epsilon_{sr}$$

$$\text{BLKD:}\ O\!\Big(\tfrac{|F_t|_C}{n^{\alpha_{tr}}}+\tfrac{|F_s|_C}{n^{\alpha_{st}}}\Big)+\epsilon_{tr}+\epsilon_{st}$$

$$\text{TAKD:}\ O\!\Big(\tfrac{|F_t|_C}{n^{\alpha_{tr}}}+\tfrac{|F_a|_C}{n^{\alpha_{at}}}+\tfrac{|F_s|_C}{n^{\alpha_{sa}}}\Big)+\epsilon_{tr}+\epsilon_{at}+\epsilon_{sa}$$

Lập luận: khoảng cách càng lớn thì $\alpha$ càng nhỏ, tức học khó hơn: $\alpha_{st}\le\alpha_{sa}$ và $\alpha_{st}\le\alpha_{at}$. Chia khoảng cách $T\to S$ thành hai đoạn ngắn hơn thì hai số mũ lớn hơn và tổng cận trên nhỏ hơn BLKD (theo tác giả, với giả định của Hinton là $\epsilon_{at}+\epsilon_{sa}\le\epsilon_{st}$). Họ thừa nhận đây là **cận trên, tiệm cận** ($n\to\infty$), kế thừa Lopez-Paz et al. (2015); ở mẫu hữu hạn, BLKD có thể hỏng khi $|F_t|_C$ quá lớn hoặc khoảng cách quá lớn (khi $\alpha_{st}$ gần $\alpha_{sr}$) và đó chính là lúc TAKD quan trọng.

## 4. Thí nghiệm và kết quả chính

**Dữ liệu:** CIFAR-10, CIFAR-100, và một thí nghiệm ImageNet. **Kiến trúc:** CNN kiểu VGG (cỡ = số lớp conv) và ResNet (cỡ = số block). PyTorch, SGD Nesterov momentum 0.9, lr 0.1, 150 epoch (ResNet hạ lr 0.01 ở epoch 80 và 0.001 ở epoch 120, weight decay 0.0001).

**Bảng 1: BLKD, NOKD, TAKD** (top-1 accuracy %; CNN: S=2, TA=4, T=10; ResNet CIFAR: S=8, TA=20, T=110; ImageNet ResNet: S=14, TA=20, T=50):

| Mô hình | Dữ liệu | NOKD | BLKD | TAKD |
|---|---|---|---|---|
| CNN | CIFAR-10 | 70.16 | 72.57 | 73.51 |
| CNN | CIFAR-100 | 41.09 | 44.57 | 44.92 |
| ResNet | CIFAR-10 | 88.52 | 88.65 | 88.98 |
| ResNet | CIFAR-100 | 61.37 | 61.41 | 61.82 |
| ResNet | ImageNet | 65.20 | 66.60 | 67.36 |

**Kích cỡ TA tốt nhất** (Bảng 2, CNN S=2, T=10; bảng trích xuất hơi lộn xộn nhưng đọc được):

| Dữ liệu | TA=8 | TA=6 | TA=4 |
|---|---|---|---|
| CIFAR-10 | 72.75 | 73.15 | 73.51 |
| CIFAR-100 | 44.28 | 44.57 | 44.92 |

Bảng 3 (ResNet S=8, T=110): CIFAR-10: TA=56: 88.70, TA=32: 88.73, TA=20: 88.90, TA=14: 88.98. CIFAR-100: TA=56: 61.47, TA=32: 61.55, TA=20: 61.82, TA=14: 61.5.

Hình 2 và 3 (accuracy của student theo cỡ teacher, lợi ích theo cỡ student) chỉ có dưới dạng đồ thị, nên không có con số chính xác trong text. Hình 5 (tất cả đường chưng cất cho CNN trên CIFAR-100, T=10) cũng ở dạng đồ thị.

**So sánh với các phương pháp khác (Hình 6 phải):** TAKD một TA so với FitNet, AT, FSP, BSS (con số lấy từ Heo et al. 2018) và Mutual (DML do họ tự cài), teacher ResNet26, student ResNet8 và ResNet14, TA là ResNet20 (cho S=ResNet14) và ResNet14 (cho S=ResNet8). Họ báo cáo TAKD thắng tất cả, nhưng bảng số trong hình không đọc được từ text, nên không ghi số ở đây.

## 5. Phân tích / ablation

- **TA kích thước nào?** Mọi kích thước TA trung gian đều tốt hơn BLKD và NOKD. Nhưng TA tối ưu không nằm đúng ở giữa theo số lớp: với S=2, T=10, TA=4 tốt hơn TA=6 hay 8. Quan sát của tác giả: TA tối ưu gần điểm giữa theo **accuracy trung bình khi train từ đầu** của student và teacher (Hình 4: cỡ 4 gần mức trung bình giữa S=2 và T=10 hơn cỡ 6 hay 8; với ResNet CIFAR-10 là TA=14, CIFAR-100 là TA=20). Heuristic đề xuất: chọn TA có accuracy (từ đầu) gần trung bình của student và teacher, nhưng tác giả nói còn cần nghiên cứu hệ thống hơn.
- **Một bước hay nhiều bước?** Thử mọi đường chưng cất trên các cỡ chẵn (với $n$ mạng trung gian có $2^n$ đường). Mọi đường có TA đều tốt hơn BLKD và NOKD; với S=2 và S=4, đường đi qua **tất cả** TA là tốt nhất; với S=2, các biến thể nhiều bước đều "comparably good". Không bị ràng buộc thì đường đầy đủ là tối ưu, và trong thực tế chọn đường theo ngân sách thời gian/tính toán. Tìm tự động đường tốt nhất (khi giới hạn số TA) được để cho nghiên cứu tương lai (có thảo luận trong appendix, không có trong text này).
- **Cảnh quan loss (Hình 7, kỹ thuật của Li et al. 2018):** student train bằng TAKD có bề mặt loss **phẳng hơn** quanh cực tiểu so với NOKD và BLKD (CNN S=2 trên CIFAR-100; ResNet S=8). Tác giả thừa nhận mối liên hệ giữa độ phẳng và tổng quát hóa còn tranh luận, và coi đây là bằng chứng gián tiếp (cùng hướng với DML).
- **Không phụ thuộc phép biến thể khác:** tác giả nói TAKD bổ trợ, có thể kết hợp với FitNet, AT, v.v.

## 6. Giới hạn

**Do tác giả nêu:**
- Phân tích lý thuyết dùng cận trên và tiệm cận, không phải hiệu năng thực.
- Chọn kích cỡ và số lượng TA là chi phí thêm; chưa có phương pháp tự động hoàn toàn; chọn đường chưng cất khi giới hạn số TA còn để ngỏ.

**Do chúng ta nhận thấy:**
- Mức cải thiện nhỏ ở ResNet (ví dụ CIFAR-10: BLKD 88.65 sang TAKD 88.98; CIFAR-100: 61.41 sang 61.82), chỉ báo cáo "accuracy tốt nhất qua 120 lần thử hyperparameter" mà không có độ lệch chuẩn theo seed; chênh vài phần mười có thể nằm trong nhiễu và phần chọn-theo-test của tuner.
- Cỡ mạng được đo bằng số lớp/block (proxy), không kiểm soát tách biệt chiều rộng và chiều sâu; "khoảng cách" chưa được định nghĩa bằng số tham số hay FLOP.
- Kiến trúc cùng họ (CNN-CNN, ResNet-ResNet); chưa kiểm tra transformer, VLM hay LLM.
- Các tác giả giải thích hiện tượng bằng ba yếu tố, nhưng không tách riêng và đo từng yếu tố (đây là giả thuyết giải thích hợp lý chứ không được chứng minh cô lập). Bài 09 sẽ thử làm rõ nguyên nhân này.
- Chi phí huấn luyện nhiều hơn: mỗi TA là một lần train đầy đủ.

## 7. Liên hệ với các bài khác

- **02 Hinton KD:** TAKD dùng đúng loss BLKD ở mỗi bước, chỉ thay đổi cấu trúc chuỗi.
- **06 BAN, 07 DML:** cả hai hạ yêu cầu về teacher; TAKD nói teacher quá lớn cũng là vấn đề.
- **09 Efficacy of KD:** phân tích sâu hơn vì sao teacher lớn hơn không tốt hơn và đề xuất dừng sớm teacher/regularize teacher. Đọc bài này trước để có động cơ.
- **10 Does KD really work:** hỏi student có khớp teacher thật không; có thể giải thích rằng capacity gap làm khớp tệ hơn.
- **39 Small models struggle with strong reasoners, 41 Capacity gap law, 42 Distillation scaling laws:** phiên bản hiện đại của cùng vấn đề cho LLM; 40 Gemma 2 và 44 Qwen3 xử lý bằng teacher vừa phải hoặc chuỗi.
- **33 TAID** (nếu đọc sau): nội suy teacher-student theo thời gian, cùng tinh thần "bắc cầu qua khoảng cách".

## 8. Ứng dụng cho dự án (suy luận)

Phần này là suy luận của chúng ta. **Liên quan khá cao**: dự án có teacher 4B và student bị nén còn 0.3B/0.1B, tức tỉ lệ kích thước tới 13x đến 40x, đúng vùng "gap lớn" mà bài lo ngại.

1. **Hai mức nén tự nhiên đã là TA:** model 0.8B gốc (LightOnOCR-3-0.8B) có thể đóng vai TA giữa teacher 4B và student 0.3B/0.1B. Đường: 4B $\to$ 0.8B $\to$ 0.3B $\to$ 0.1B (hoặc 4B $\to$ 0.3B $\to$ 0.1B). Vì model 0.3B của dự án được cắt từ 0.8B nên thực chất một "TA đã pruned" sẵn có sẵn trong pipeline, khác với TAKD gốc là TA train từ đầu.
2. **Chọn TA theo heuristic accuracy giữa:** tính chất đáng thử là chọn mức nén trung gian (ví dụ 0.5B) sao cho CER/WER sau phục hồi gần trung bình giữa 4B và student đích.
3. **Cảnh báo:** các thí nghiệm của paper tách riêng dung lượng thấp ở student nhưng không tách "teacher quá tự tin". Với OCR, teacher 4B có thể rất nhọn trên token đúng (quá tự tin), nên thử temperature $\tau>1$ hoặc teacher trung gian có thể quan trọng (suy luận).
4. **Quan sát "cắt MLP neuron phục hồi kém hơn cắt layer":** TAKD không giải thích trực tiếp điều này, nhưng có thể nghĩ khoảng cách "chức năng" giữa teacher và student sau khi cắt neuron lớn hơn khoảng cách kích thước đơn thuần, nên một TA ở giữa (cắt neuron nhẹ hơn) có thể giúp; cần thực nghiệm.

## 9. Thí nghiệm gợi ý

1. **Có TA hay không:** so sánh (a) 4B $\to$ 0.1B trực tiếp, (b) 4B $\to$ 0.3B $\to$ 0.1B, (c) 4B $\to$ 0.8B $\to$ 0.3B $\to$ 0.1B, cùng tổng số bước/token cho student cuối. Đo CER/WER và độ khớp token-level với teacher.
2. **Điểm TA tối ưu:** với student 0.1B, thử TA ở 0.2B, 0.3B, 0.5B; vẽ metric theo kích cỡ TA để xem có đỉnh giống Bảng 2 không.
3. **Chẩn đoán yếu tố 3:** đo entropy trung bình của output teacher 4B và của TA trên tập OCR; kiểm tra TA có "mềm" hơn không và liệu tăng $\tau$ cho teacher 4B có thay thế được vai trò TA không.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| BLKD | $(1-\lambda)\text{CE}+\lambda\tau^2\text{KL}$ chưng cất trực tiếp T $\to$ S |
| NOKD | Student train từ đầu |
| Capacity gap | Chênh lệch cỡ giữa T và S; quá lớn thì KD kém hiệu quả |
| TA | Mạng cỡ trung gian, học từ T rồi dạy S |
| Distillation path | Chuỗi T $\to$ TA$_1$ $\to\dots\to$ S |
| 3 yếu tố | Teacher giỏi hơn (+), student không đủ sức bắt chước (−), teacher quá tự tin làm target kém mềm (−) |
| Cận VC | Lỗi $\le O(|F|_C/n^\alpha)+\epsilon$; $\alpha$ nhỏ khi khoảng cách lớn |
| Heuristic cỡ TA | Chọn TA có accuracy gần trung bình của S và T |

## 11. Kết luận ngắn

Teacher lớn hơn không phải lúc nào cũng cho student tốt hơn: có điểm gãy khi khoảng cách dung lượng quá lớn. TAKD chèn một hoặc nhiều mạng trung gian, cải thiện nhất quán (dù không lớn) trên CIFAR-10/100 và ImageNet so với KD trực tiếp, kèm lập luận lý thuyết (cận trên) và bằng chứng cảnh quan loss phẳng hơn. Với dự án nén 4B xuống 0.3B/0.1B, ý tưởng chuỗi 4B $\to$ 0.8B $\to$ 0.3B $\to$ 0.1B là ứng viên tự nhiên để thử.

Nguồn gốc: https://arxiv.org/abs/1902.03393
