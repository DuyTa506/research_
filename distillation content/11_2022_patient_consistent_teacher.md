# Knowledge distillation: A good teacher is patient and consistent (2022)

**Tác giả:** Lucas Beyer, Xiaohua Zhai, Amélie Royer, Larisa Markeeva, Rohan Anil, Alexander Kolesnikov (Google Research, Brain Team).  
**Venue / arXiv:** arXiv:2106.05237 (bản v2 ngày 21/06/2022; bản v1 xuất hiện 09/06/2021). Bản text đã đọc không ghi tên hội nghị. Tên file lộ trình dùng năm 2022 theo bản v2.  
**Nguồn:** https://arxiv.org/abs/2106.05237  
**Chủ đề:** Công thức thực nghiệm để chưng cất mô hình thị giác lớn (BiT-ResNet-152x2) xuống ResNet-50/MobileNetV3; "function matching" (khớp hàm); nhất quán đầu vào giữa teacher và student, mixup mạnh, huấn luyện cực lâu; Shampoo; chưng cất trên dữ liệu ngoài miền.  
**Vị trí trong lộ trình:** Đọc trước: 02 (Hinton KD), 09 (Efficacy of KD: khuyên dừng sớm), 10 (Does KD really work: fidelity thấp do tối ưu hóa). Bài này thêm: đáp án thực dụng cho những vấn đề đó, ngược hướng với 09: **train rất lâu**, cùng input cho teacher và student, và không precompute logit. Đọc tiếp: 12–14 (Relational KD, CRD, Decoupled KD: cải tiến loss), 21 (DeiT), 22 (DINO), 23 (AM-RADIO).

> Đây là ghi chú học tập bằng tiếng Việt, diễn giải và tóm lược, không phải bản dịch paper. Số liệu lấy từ paper; nhiều đồ thị không có số trong text nên chỉ ghi số có trong văn bản và bảng. Phần "Ứng dụng cho dự án" và các nhận xét ghi là suy luận là của chúng ta.

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

Mô hình thị giác tốt nhất rất lớn, còn thứ mọi người thực sự triển khai là ResNet-50 hay MobileNet (tác giả dẫn số lượt tải của các model BiT từ TensorFlow Hub: bản ResNet-50 nhỏ nhất được tải nhiều hơn hẳn các bản lớn). Bài hỏi: cho một ứng dụng và một mô hình lớn rất tốt, nén sang kiến trúc nhỏ mà không mất hiệu năng bằng cách nào?

Hai họ phương pháp nén chính: **pruning** (cắt bộ phận của mô hình; hạn chế: không đổi được họ mô hình, ví dụ ResNet sang MobileNet; và có thể gặp khó khăn phụ thuộc kiến trúc, ví dụ với group normalization cắt kênh làm phải cân lại nhóm kênh) và **knowledge distillation**. Bài chọn KD vì nó cho phép đổi kiến trúc, và không đề xuất phương pháp mới mà tìm ra cách làm KD **đúng**.

Các vấn đề thực tế khiến người ta làm KD chưa tốt (theo tác giả):
1. Hay **precompute** (tính sẵn) logit teacher offline cho mỗi ảnh để tiết kiệm compute ("fixed teacher").
2. KD cũng dùng ở nhiều ngữ cảnh khác (bán giám sát, tự giám sát) với khuyến nghị thiết kế ngược nhau.
3. KD cần số epoch lớn bất thường so với supervised.
4. Lựa chọn tốt cho lịch ngắn thường xấu cho lịch dài và ngược lại.

Khái niệm cần có: **mixup** (trộn tuyến tính hai ảnh $x=\lambda x_1+(1-\lambda)x_2$ — ở đây KHÔNG trộn nhãn, vì nhãn là output của teacher cho ảnh đã trộn); **inception-style crop** (cắt ngẫu nhiên vùng kích thước và tỉ lệ ngẫu nhiên rồi resize); **Shampoo** (bộ tối ưu bậc hai dùng preconditioner theo tensor, đắt hơn Adam nhưng hội tụ nhanh hơn theo bước).

## 2. Ý tưởng chính

Nhìn KD như **function matching**: ép hàm do student biểu diễn khớp hàm của teacher trên **nhiều điểm hỗ trợ** (support points), chứ không phải "teacher cho nhãn mềm tốt hơn". Hệ quả ba nguyên tắc: (i) teacher và student phải thấy **chính xác cùng một view của ảnh** (cùng crop, cùng augmentation, tính teacher online ở mỗi bước), (ii) mở rộng đa tạp ảnh bằng augmentation mạnh, cụ thể mixup với hệ số lấy đều từ $[0,1]$, để có nhiều điểm hỗ trợ cả ngoài đa tạp ảnh gốc, và (iii) huấn luyện **rất lâu** (kiên nhẫn, hàng nghìn đến hàng vạn epoch) vì bài toán khớp hàm là bài tối ưu khó. Kết quả: ResNet-50 đạt 82.8% top-1 trên ImageNet.

## 3. Phương pháp chi tiết

### 3.1 Loss

Chỉ dùng KL giữa xác suất teacher và student, **không có** loss nhãn cứng:

$$\boxed{\mathrm{KL}(p_t\|p_s)=\sum_{i\in C}\big[-p_{t,i}\log p_{s,i}+p_{t,i}\log p_{t,i}\big]}$$

với **temperature $T$**: $p_s\propto\exp(\log p_s/T)$, $p_t\propto\exp(\log p_t/T)$ (tức softmax làm mềm cả hai phía; $T$ nằm trong dải quét {1, 2, 5, 10}).

### 3.2 Bốn kiểu cấu hình (Hình 2)

| Kiểu | Mô tả | Biến thể trong thí nghiệm |
|---|---|---|
| Fixed teacher | Dự đoán teacher cố định cho mỗi ảnh (precompute) | fix/rs (chỉ resize 224 cho cả hai), fix/cc (teacher dùng central crop cố định, student crop ngẫu nhiên nhẹ), fix/ic_ens (teacher = trung bình dự đoán trên 1k inception crop, student dùng inception crop ngẫu nhiên) |
| Independent noise | Teacher và student nhận augmentation độc lập | ind/rc (random crop nhẹ), ind/ic (inception crop) |
| Consistent teaching | Cắt/augment **một lần**, đưa cùng view cho cả hai | same/rc, same/ic |
| Function matching | Consistent + mixup (cùng ảnh trộn cho cả hai) | same/rc,mix, same/ic,mix ("FunMatch") |

### 3.3 Thiết lập huấn luyện

Adam mặc định (chỉ lr được quét), cosine decay không restart, weight decay "decoupled" được quét, gradient clipping 1.0 theo L2-norm toàn cục. Batch 512 (ImageNet: 4096). Ảnh 128x128 để phân tích rộng (ImageNet: 224x224). Mỗi cấu hình quét toàn bộ tổ hợp lr {0.0003, 0.001, 0.003, 0.01} x weight decay {1e-5, 3e-5, 1e-4, 3e-4, 1e-3} x $T$ {1, 2, 5, 10}; trong hình mọi run được vẽ mờ, và nhấn mạnh run có validation accuracy cuối cùng tốt nhất; chọn siêu tham số trên validation split, báo test. Chạy trên Google Cloud TPU, "dozens of thousands of models" được huấn luyện.

```
for step in range(N):                           # N rất lớn: hàng nghìn đến chục nghìn epoch
    x = batch(images)
    x, _ = random_inception_crop(x) ; x = mixup(x, lam ~ U[0,1])   # MỘT view, lấy MỘT lần
    with no_grad: p_t = softmax(teacher(x) / T)  # teacher chạy ONLINE trên đúng view này
    p_s = softmax(student(x) / T)
    loss = KL(p_t || p_s)                        # không dùng nhãn cứng
    student <- Adam / Shampoo(loss)
```

### 3.4 Các mở rộng

- **Khác độ phân giải:** crop trên ảnh độ phân giải gốc, rồi resize khác nhau cho teacher và student: vẫn nhất quán nội dung. Có thể student 160 px (nhanh gấp 2) với teacher 224 px, hoặc teacher fine-tune ở 384 px với student 224 px.
- **Teacher ensemble:** trung bình logit của hai teacher (224 px và 384 px).
- **Khác họ kiến trúc:** ResNet sang MobileNetV3 (Large, thay BatchNorm bằng GroupNorm 8 nhóm).

## 4. Thí nghiệm và kết quả chính

**Dữ liệu:** Flowers102 (1020 ảnh train), Pets (37 lớp, 3312 ảnh), Food101 (khoảng 68k), SUN397 (khoảng 76k), ILSVRC-2012/ImageNet (1000 lớp, 1.281.167 ảnh). **Teacher chính:** BiT-M-R152x2 (BiT-ResNet-152x2, pretrain trên ImageNet-21k, GroupNorm và weight standardization; cỡ tính toán khoảng 10x ResNet-50 chuẩn). **Student:** ResNet-50 (biến thể BiT, GroupNorm). Teacher đạt 83.0% top-1 trên ImageNet.

**Tính nhất quán là then chốt (Hình 3, Flowers102, 10.000 epoch, định tính):**
- Mọi cấu hình "fixed teacher" đạt distillation loss train rất thấp nhưng student tổng quát hóa tệ (overfit mạnh trên dữ liệu nhỏ); mọi cấu hình "inconsistent" dừng ở điểm thấp.
- Consistent teaching tăng accuracy đáng kể; **function matching tốt nhất**, và nó không bao giờ đạt loss train thấp như fixed teacher nhưng tổng quát hóa tốt hơn.

**Kiên nhẫn (Hình 4):** accuracy student tiệm cận teacher khi tăng số epoch (trục log, đến 1M epoch cho Flowers102) trên cả bốn tập, không thấy overfitting kể cả ở 1M epoch. Ở 100 epoch (thời lượng bình thường), student kém hơn baseline transfer (ResNet-50 pretrain ImageNet-21k) nhiều. Student cuối cùng vượt cả baseline train từ đầu với nhãn gốc lẫn baseline transfer BiT-M ResNet-50 (hai baseline đều được tune lr và weight decay kỹ).

**ImageNet (Hình 5 trái, Bảng 1):** với fixed teacher, student overfit sau khoảng 600 epoch; consistent teaching tăng liên tục theo thời lượng. Function matching hơi kém ở lịch ngắn (underfit) nhưng với 1200 epoch đã bằng consistent teaching 4800 epoch (tiết kiệm 75% compute). Bảng 1 (top-1 test accuracy %):

| Thí nghiệm | 300 epoch | 1200 | 4800 | 9600 |
|---|---|---|---|---|
| T224 -> S224 | 80.30 | 81.54 | 82.18 | 82.31 |
| T224 -> S160 | 78.17 | 79.61 | N/A | 80.49 |
| T384 -> S224 | 80.46 | 81.82 | 82.33 | 82.64 |

(T = độ phân giải đầu vào của teacher, S = của student; teacher 384 px đạt 83.7% top-1.)

**So sánh với tài liệu (Bảng 2, top-1 ImageNet, kiến trúc, độ phân giải):**

| Mô hình | Kiến trúc | Độ phân giải | Accuracy |
|---|---|---|---|
| Revisiting ResNet | R50 | 160 | 78.8% |
| FunMatch (T224) | R50 | 160 | 80.5% |
| Original ResNet | R50 | 224 | 77.2% |
| BiT-M-R50 | R50 | 224 | 78.4% |
| Meal-v2 | R50 | 224 | 80.7% |
| **FunMatch (T384+224)** | R50 | 224 | **82.8%** |
| Revisiting ResNet | R152 | 224 | 82.8% |

Tác giả nói đây là ResNet-50 SOTA mới (khoảng 4.4% hơn ResNet-50 gốc của BiT[23] và 2.2% hơn kết quả tốt nhất trong tài liệu, vốn dùng thiết lập phức tạp hơn); tổng cộng 9600 epoch chưng cất.

**Ensemble teacher:** trung bình logit teacher 224 px và 384 px; student tốt hơn student của teacher mặc định ở mọi độ dài thử, sau 9600 epoch đạt **82.82%** top-1.

**MobileNetV3 (Large, GroupNorm):** chỉ dùng function matching, không dùng thủ thuật huấn luyện của bài MobileNet gốc: **74.60%** sau 300 epoch và **76.31%** sau 1200 epoch, mô hình MobileNetV3 tốt nhất đã công bố (theo tác giả).

**Shampoo (Hình 5 giữa):** đạt cùng accuracy mà Adam đạt ở 4800 epoch chỉ sau 1200 epoch, với chi phí mỗi bước tăng ít; cải thiện đều ở các thiết lập (chi tiết trong appendix D).

**Khởi tạo bằng pretrain (Hình 5 phải):** student khởi tạo từ BiT-M-ResNet50 cải thiện hơn 2% ở lịch ngắn, nhưng khoảng cách khép lại từ khoảng 1200 epoch; từ scratch ngang bằng ở 1200 và vượt nhẹ ở 4800 epoch.

**Dữ liệu ngoài miền (Hình 6, Pets và SUN397):** chưng cất trên ảnh hoàn toàn không liên quan vẫn "work to some extent": ví dụ student học phân loại Pets với khoảng **30% accuracy** chỉ nhờ xem ảnh Food được gán nhãn mềm bằng giống thú cưng. Dữ liệu trong miền tốt nhất; dữ liệu liên quan/chồng lấn (Pets với ImageNet, SUN397 với ImageNet) gần bằng nhưng cần lịch dài hơn.

**Baseline supervised dùng chung công thức (Hình 7):** train ResNet-50 bằng nhãn cứng với cùng thiết lập (mixup, lịch dài, thử cả SGD momentum) tệ hơn nhiều và overfit ở lịch dài, tức KD là bắt buộc để công thức hoạt động.

## 5. Phân tích / ablation

- **Consistency vs fixed teacher:** precompute logit nhìn rẻ nhưng hại: student "overfit" vào bộ nhãn cố định; nhất quán + online cho tín hiệu đa dạng vô hạn.
- **Function matching vs consistent teaching:** mixup tạo điểm hỗ trợ ngoài đa tạp ảnh; lợi ích rõ khi lịch dài, hơi hại ở lịch ngắn vì underfit.
- **Tối ưu hóa là nút cổ chai:** tác giả suy đoán khó tối ưu vì khớp một hàm có đầu ra nhiều chiều khó hơn khớp nhãn ảnh cố định. Shampoo rút ngắn 4x.
- **Khởi tạo và lịch:** khởi tạo tốt giúp lịch ngắn, mất lợi thế ở lịch dài (phù hợp kết luận He et al. "Rethinking ImageNet pre-training").
- **Không phụ thuộc cặp kiến trúc:** hoạt động khi đổi độ phân giải, họ kiến trúc, dùng ensemble teacher.
- **Out-of-domain:** hàm cần khớp trên đa dạng input; không cần ảnh trong miền, nhưng trong miền vẫn tốt nhất.

## 6. Giới hạn

**Do tác giả nêu hoặc hàm ý:**
- Chi phí huấn luyện rất lớn (9600 epoch ImageNet; hàng chục nghìn mô hình trong nghiên cứu); tác giả dùng độ phân giải thấp 128 px để phân tích. Họ thừa nhận chi phí tính toán và dùng Shampoo như bước đầu giảm.
- Phạm vi chỉ phân loại ảnh; không so sánh cạnh tranh với pruning (chỉ bàn lý do chọn KD).
- Không tối ưu loss; tập trung "recipe", không đề xuất phương pháp mới.

**Do chúng ta nhận thấy:**
- Teacher mỗi bước phải chạy forward (online) nên chi phí mỗi bước cao hơn precompute; với teacher lớn (BiT-R152x2 hay 4B) đây là chi phí lớn; bài so compute theo epoch của student chứ không công bố tổng FLOPs đầy đủ.
- Mixup và inception crop là augmentation ảnh; tương đương cho văn bản/token/OCR không rõ ràng.
- Cạnh tranh với hướng "dừng sớm" của 09: không mâu thuẫn trực tiếp, 09 xét student nhỏ trên ImageNet với cấu hình KD đơn giản (precompute-like, không mixup), còn 11 xử lý bằng chuẩn nhất quán và thời gian cực dài; nhưng cho thấy kết luận "KD hại" trong 09 có thể một phần là sản phẩm của cấu hình huấn luyện.
- Teacher ở đây tốt hơn student chỉ khoảng 0.2 đến 0.7 điểm sau cùng, tức student gần như khớp teacher: bài này (chưa rõ trong text) gợi ý fidelity cao, ngược với bi quan của bài 10; hai bài ra đời cùng lúc (bài 10 trích bài này) và kết luận bổ sung nhau.

## 7. Liên hệ với các bài khác

- **10 Does KD really work:** bài đồng hành. 10 chẩn đoán (fidelity thấp, lý do là tối ưu hóa); 11 là công thức chữa: nhất quán đầu vào, mixup, lịch dài, Shampoo (bài 10 nhắc đến chính các khuyến nghị này như "careful treatment of data augmentation, MixUp, extended training duration" và nhận xét rằng chúng tăng accuracy nhưng vẫn còn khác biệt dự đoán).
- **09 Efficacy of KD:** bài 09 khuyên teacher dừng sớm và KD dừng sớm; 11 nói train cực lâu và dùng cả teacher đầy đủ. Hai kết luận đến từ thiết lập khác nhau (09: loss kết hợp nhãn cứng, không mixup; 11: không nhãn, mixup, online teacher).
- **08 Teacher assistant:** 11 không dùng TA mà vẫn ghép được ResNet-152x2 với ResNet-50 nhờ lịch rất dài.
- **06 BAN, 07 DML:** 11 cũng loại bỏ nhãn cứng ("không dùng thêm loss nhãn cứng"), gần với BAN thuần; ý "khớp hàm" bổ sung cho giải thích dark knowledge.
- **12 Relational KD, 13 CRD, 14 Decoupled KD:** các bài sau cải tiến dạng loss; 11 cho thấy cách huấn luyện (nhất quán + lâu) có thể quan trọng hơn dạng loss.
- **21 DeiT:** dùng KD với convnet teacher, token distillation; cùng tinh thần teacher online cùng view.
- **22 DINO, 23 AM-RADIO, 24 CLIP-KD:** KD đa-teacher và tự giám sát hiện đại dựa vào cùng nguyên tắc: student/teacher thấy cùng (hoặc consistent) view.
- **Các bài LLM (28 MiniLLM, 29 GKD, 31 Rethinking KL, 32 Speculative KD):** on-policy/online teacher trong GKD tương ứng với nguyên tắc "không precompute, teacher đánh giá đúng input mà student thấy".

## 8. Ứng dụng cho dự án (suy luận)

Phần này là suy luận của chúng ta. **Liên quan rất cao** vì dự án là chưng cất teacher 4B thành student pruned nhỏ để chạy trên GPU nhỏ.

1. **Teacher online, cùng input:** với VLM OCR, mỗi bước forward teacher 4B trên đúng ảnh (cùng crop, cùng augmentation, cùng độ phân giải/số patch) và đúng prefix token như student; tránh dùng logit teacher tính sẵn cho một view ảnh cố định. Nếu bắt buộc phải cache logit (hết VRAM), ít nhất cache theo từng cặp (ảnh đã augment, prefix) và biết rằng tác giả coi đây là tuỳ chọn tệ nhất.
2. **Khác độ phân giải vẫn nhất quán:** teacher có thể nhận ảnh phân giải cao hơn, student nhận phân giải thấp hơn (cắt từ cùng một vùng), đúng Mục 3.3; hợp với ràng buộc GPU/latency của student (suy luận: dùng thêm cho ViT 12 block).
3. **Dữ liệu distillation đa dạng thay vì nhãn:** chỉ dùng KL với teacher (bỏ nhãn cứng) và mở rộng input bằng tài liệu không nhãn/ngoài miền (tài liệu tổng hợp, render PDF) vì KL cần nhiều điểm hỗ trợ; "ngoài miền vẫn học được ít nhiều" là động cơ. Mixup trên ảnh trang tài liệu có thể không tự nhiên cho OCR; cần thử (chưa kiểm chứng).
4. **Lịch dài và optimizer:** nếu ngân sách cho phép, tăng thời lượng KD thay vì thêm loss mới; thử optimizer mạnh hơn (Shampoo, Muon...) để rút ngắn. Khởi tạo tốt (student cắt từ model gốc) giúp mạnh ở lịch ngắn, đúng với trường hợp của dự án (suy luận từ Hình 5 phải).
5. **Về "cắt MLP neuron phục hồi kém hơn cắt layer":** bài 11 gợi ý nút cổ chai là tối ưu hóa khớp hàm. Nếu cắt neuron để lại student "xa" hàm teacher hơn, thì lịch dài và optimizer tốt hơn có thể bù, nhưng đây là giả thuyết; ngân sách lịch dài còn bị giới hạn bởi tài nguyên.
6. Nhắc: bài chỉ so sánh với pruning về lý do thiết kế, không thử "prune rồi distill". Kết quả của nó là về KD thuần giữa hai kiến trúc.

## 9. Thí nghiệm gợi ý

1. **Online vs cached teacher:** huấn luyện cùng số bước student 0.3B với (a) logit teacher 4B tính online trên đúng view đã augment và (b) logit tính sẵn trên view gốc; so CER/WER, train/held-out agreement. Dự đoán từ bài: (b) overfit vì thiếu đa dạng.
2. **Thời lượng:** quét số bước KD (1x, 3x, 10x) cho student 0.1B và vẽ metric theo số bước (kiểu Hình 4); xem có tiệm cận teacher không và cắt layer so với cắt neuron có khác đường cong.
3. **Điểm hỗ trợ ngoài miền:** thêm ảnh tài liệu ngoài miền (giấy tờ, biểu mẫu, ảnh chụp) vào dữ liệu distillation, chỉ dùng KL, so với chỉ dữ liệu OCR trong miền.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Function matching | Coi KD là khớp hàm trên nhiều điểm hỗ trợ |
| Consistent teaching | Teacher và student nhận cùng một view (cùng crop và augmentation) |
| Fixed teacher | Dự đoán teacher tính sẵn cho một view cố định; kém |
| Patient | Huấn luyện rất lâu (hàng nghìn đến chục nghìn epoch) |
| Loss | $\mathrm{KL}(p_t\|p_s)$ với temperature $T$, không có nhãn cứng |
| Mixup mạnh | Hệ số trộn lấy đều từ $[0,1]$ để sinh điểm ngoài đa tạp ảnh |
| Shampoo | Optimizer bậc hai; đạt kết quả Adam 4800 epoch ở 1200 epoch |
| Out-of-domain distillation | Chưng cất bằng ảnh không thuộc miền tác vụ; vẫn học được ít nhiều |
| Kết quả chính | ResNet-50 82.8% (224 px), 80.5% (160 px); MobileNetV3 76.31% |

## 11. Kết luận ngắn

Chưng cất tốt cho nén mô hình không cần loss mới mà cần ba điều: teacher và student thấy cùng view và teacher chạy online, augmentation mạnh kiểu mixup để mở rộng điểm hỗ trợ, và huấn luyện rất lâu (cùng optimizer tốt). Làm đủ cả ba, ResNet-50 đạt 82.8% top-1 ImageNet và MobileNetV3 đạt 76.31%, vượt các kết quả đã công bố cho kiến trúc đó. Với dự án, bài là hướng dẫn thực hành quan trọng nhất trong nhóm "nền tảng": không cache logit, giữ view nhất quán, cân nhắc ngân sách bước và dữ liệu đa dạng.

Nguồn gốc: https://arxiv.org/abs/2106.05237
