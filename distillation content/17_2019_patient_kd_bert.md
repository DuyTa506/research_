# Patient Knowledge Distillation for BERT Model Compression (2019)

**Tác giả:** Siqi Sun, Yu Cheng, Zhe Gan, Jingjing Liu (Microsoft Dynamics 365 AI Research).  
**Venue / arXiv:** arXiv:1908.09355 (v1, 25/8/2019); EMNLP 2019 (theo ghi chú trên trang arXiv, bản text không ghi).  
**Nguồn:** https://arxiv.org/abs/1908.09355  
**Chủ đề:** Distill đa tầng cho BERT: student khớp trạng thái ẩn của token `[CLS]` ở nhiều layer trung gian của teacher (PKD-Skip và PKD-Last), kèm KD ở lớp cuối.  
**Vị trí trong lộ trình:** Đọc sau 03 (FitNets: khớp tầng giữa), 04 (Attention transfer), 16 (DistilBERT). Bài này đặt câu hỏi "student nên bắt chước những layer nào của teacher" và trả lời bằng hai kiểu ánh xạ layer. Tiếp theo: 18 (TinyBERT) làm tổng quát hóa ánh xạ layer cộng thêm khớp attention; 19 (MiniLM) chỉ khớp layer cuối.

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch paper. Phần "Ứng dụng cho dự án" là suy luận của chúng ta, không phải kết quả của paper.

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

BERT-Base có 12 layer Transformer, 110 triệu tham số; BERT-Large 24 layer, 330 triệu. Huấn luyện cần nhiều ngày trên TPU, và fine-tune một epoch có thể mất hàng giờ. Paper này tự nhận là nỗ lực đầu tiên nén BERT (ghi chú: DistilBERT, bài 16, ra sau khoảng hai tháng và đi theo hướng pre-training).

Vanilla KD (Hinton, bài 02) trên BERT chỉ dùng **đầu ra lớp cuối** của teacher (logit/xác suất mềm) cộng nhãn thật:

$$L_{KD}=(1-\alpha)L^s_{CE}+\alpha L_{DS}$$

với $L_{DS}$ là cross-entropy tới xác suất mềm của teacher (có nhiệt độ $T$). Vấn đề tác giả quan sát: student học bằng vanilla KD **bão hòa nhanh trên dev set** (Figure 2: accuracy train tiếp tục tăng, dev ngừng), có thể do overfit sớm trong quá trình distill. Nói cách khác, signal ở lớp cuối quá "hẹp" để student tiếp tục học tốt khi huấn luyện lâu hơn.

Khái niệm cần có: `[CLS]` là token đặc biệt đứng đầu mỗi input của BERT; hidden state của nó ở lớp cuối được dùng cho phân loại câu. **Layer mapping** = quy tắc nói "layer $j$ của student bắt chước layer nào của teacher".

## 2. Ý tưởng chính

Dạy student "kiên nhẫn" (*patient*): thay vì chỉ học từ lớp cuối của teacher, student cũng khớp hidden state của token `[CLS]` ở một số layer trung gian. Hai cách chọn layer của teacher: **PKD-Last** (học từ $k$ layer cuối, giả định layer trên chứa nhiều thông tin nhất) và **PKD-Skip** (học từ mỗi $k$ layer, giả định cả layer thấp cũng quan trọng). Thêm một loss mới (MSE giữa hidden state đã chuẩn hóa) cạnh loss KD cũ. Kết quả: student 6 layer đạt gần teacher 12 layer ở nhiều tác vụ.

## 3. Phương pháp chi tiết

**Teacher.** BERT-Base (12 layer) fine-tune cho từng tác vụ; xác suất mềm với nhiệt độ $T$:

$$\hat y_i=P^t(y_i|x_i)=\mathrm{softmax}\!\left(\frac{W\cdot \mathrm{BERT}_{12}(x_i;\hat\theta_t)}{T}\right)$$

**Student.** BERT$_k$ ($k\in\{3,6\}$) khởi tạo bằng $k$ layer đầu tiên của BERT-Base pre-trained (khác DistilBERT là lấy xen kẽ).

**Chỉ dùng `[CLS]`.** Khớp hidden state của mọi token đắt và nhiễu; trong BERT gốc, dự đoán chỉ dùng `[CLS]` ở lớp cuối, nên student chỉ cần bắt chước `[CLS]` ở các layer trung gian:

$$h_i=[h_{i,1},\dots,h_{i,k}]=\mathrm{BERT}_k(x_i)\in\mathbb R^{k\times d}$$

**Tập layer teacher được khớp** $I_{pt}$ (với student 6 layer, $k=5$ vì layer cuối đã được cover bởi $L_{DS}$): với student $n$ layer thì luôn $k=n-1$.

- **PKD-Skip:** $I_{pt}=\{2,4,6,8,10\}$.
- **PKD-Last:** $I_{pt}=\{7,8,9,10,11\}$.

(Student layer $j$ ghép với teacher layer $I_{pt}(j)$, $j=1..M$ với $M$ = số layer student được khớp.)

**Loss khớp tầng giữa:** MSE giữa hidden state đã chuẩn hóa L2 (tức khoảng cách hướng, tương tự cosine):

$$\boxed{L_{PT}=\sum_{i=1}^{N}\sum_{j=1}^{M}\left\|\frac{h^s_{i,j}}{\|h^s_{i,j}\|_2}-\frac{h^t_{i,I_{pt}(j)}}{\|h^t_{i,I_{pt}(j)}\|_2}\right\|_2^2}\tag{7}$$

**Loss tổng:**

$$\boxed{L_{PKD}=(1-\alpha)L^s_{CE}+\alpha L_{DS}+\beta L_{PT}}\tag{8}$$

$\alpha$ cân bằng CE nhãn thật và KD logit; $\beta$ cân bằng loss khớp tầng giữa.

### Pseudo-code

```python
# teacher: BERT-12 đã fine-tune; student: BERT-6 khởi tạo từ 6 layer đầu của BERT-Base
I_pt = [2, 4, 6, 8, 10]                      # PKD-Skip (PKD-Last: [7, 8, 9, 10, 11])
for x, y in task_loader:
    with no_grad(): t_logits, t_cls = teacher(x, return_cls_per_layer=True)   # t_cls[l]: [B, d]
    s_logits, s_cls = student(x, return_cls_per_layer=True)                   # s_cls[j], j=1..5
    L_ce = cross_entropy(s_logits, y)
    L_ds = soft_cross_entropy(s_logits / T, t_logits / T)
    L_pt = sum(((normalize(s_cls[j]) - normalize(t_cls[I_pt[j-1]]))**2).sum(-1).mean()
               for j in range(1, 6))
    loss = (1 - alpha) * L_ce + alpha * L_ds + beta * L_pt
```

(Pseudo-code do chúng ta viết từ công thức 7 và 8; chuẩn hóa mean/sum có thể khác code gốc.)

### Ví dụ nhỏ

Student 6 layer, teacher 12 layer. PKD-Skip ghép student layer 1→teacher layer 2, 2→4, 3→6, 4→8, 5→10; layer 6 của student ghép với lớp cuối của teacher thông qua $L_{DS}$. PKD-Last ghép 1→7, 2→8, 3→9, 4→10, 5→11. Với layer student $j$ có `[CLS]` $=(3,4)$ và teacher có `[CLS]` $=(6,8)$: sau chuẩn hóa cả hai thành $(0.6,0.8)$ nên $L_{PT}=0$: loss chỉ phạt sai hướng, không phạt sai độ lớn. (Ví dụ của chúng ta.)

**Huấn luyện:** batch 32, 4 epoch, lr $\in\{5e\text{-}5,2e\text{-}5,1e\text{-}5\}$; vanilla KD quét $T\in\{5,10,20\}$, $\alpha\in\{0.2,0.5,0.7\}$; PKD cố định $\alpha,T$ theo vanilla KD tốt nhất và chỉ quét $\beta\in\{10,100,500,1000\}$ cùng lr.

## 4. Thí nghiệm và kết quả chính

Bốn loại tác vụ, bảy tập: SST-2 (cảm xúc), MRPC và QQP (paraphrase), MNLI-m/mm, QNLI, RTE (suy luận), và RACE (đọc hiểu). Teacher là BERT-Base tự fine-tune. So sánh: **FT** (fine-tune trực tiếp student, không KD), **KD** (vanilla), **PKD** (Skip).

### GLUE test server (Table 1)

| Model | SST-2 | MRPC (F1/Acc) | QQP (F1/Acc) | MNLI-m | MNLI-mm | QNLI | RTE |
|---|---|---|---|---|---|---|---|
| BERT12 (Google) | 93.5 | 88.9/84.8 | 71.2/89.2 | 84.6 | 83.4 | 90.5 | 66.4 |
| BERT12 (Teacher, tự làm) | 94.3 | 89.2/85.2 | 70.9/89.0 | 83.7 | 82.8 | 90.4 | 69.1 |
| BERT6-FT | 90.7 | 85.9/80.2 | 69.2/88.2 | 80.4 | 79.7 | 86.7 | 63.6 |
| BERT6-KD | 91.5 | 86.2/80.6 | 70.1/88.8 | 80.2 | 79.8 | 88.3 | 64.7 |
| BERT6-PKD | 92.0 | 85.0/79.9 | 70.7/88.9 | 81.5 | 81.0 | 89.0 | 65.5 |
| BERT3-FT | 86.4 | 80.5/72.6 | 65.8/86.9 | 74.8 | 74.3 | 84.3 | 55.2 |
| BERT3-KD | 86.9 | 79.5/71.1 | 67.3/87.6 | 75.4 | 74.8 | 84.0 | 56.2 |
| BERT3-PKD | 87.5 | 80.7/72.5 | 68.1/87.8 | 76.7 | 76.3 | 84.7 | 58.2 |

(Số mẫu train: SST-2 67k, MRPC 3.7k, QQP 364k, MNLI 393k, QNLI 105k, RTE 2.5k.)

Điểm chính của tác giả: PKD tốt nhất trên hầu hết tác vụ, trừ MRPC. Student 6 layer gần teacher ở 5 trên 7 tác vụ (chênh so với BERT-Base: SST-2 −2.3%, QQP −0.1%, MNLI-m −2.2%, MNLI-mm −1.8%, QNLI −1.4%); cả năm tác vụ này có hơn 60k mẫu train, nên tác giả nhận xét phương pháp tốt hơn khi có nhiều dữ liệu. Với QQP, student 3 layer vẫn gần teacher.

### RACE (đọc hiểu, test, Table 3)

| Model | RACE | RACE-M | RACE-H |
|---|---|---|---|
| BERT12 (Leaderboard) | 65.00 | 71.70 | 62.30 |
| BERT12 (Teacher) | 65.30 | 71.17 | 62.89 |
| BERT6-FT | 54.32 | 61.07 | 51.54 |
| BERT6-KD | 58.74 | 64.69 | 56.29 |
| BERT6-PKD-Skip | 60.34 | 66.57 | 57.78 |

Vanilla KD hơn FT 4.42%, PKD thêm 1.6%. Kết quả BERT3 không báo cáo vì khoảng cách teacher quá lớn.

### PKD-Last so với PKD-Skip (Table 2, BERT6)

| | SST-2 | MRPC | QQP | MNLI-m | MNLI-mm | QNLI | RTE |
|---|---|---|---|---|---|---|---|
| PKD-Last | 91.9 | 85.1/79.5 | 70.5/88.9 | 80.9 | 81.0 | 88.2 | 65.0 |
| PKD-Skip | 92.0 | 85.0/79.9 | 70.7/88.9 | 81.5 | 81.0 | 89.0 | 65.5 |

### Hiệu quả (Table 4, QNLI train 105k, Titan RTX, batch 128, độ dài 128, FP16)

| Layer | Tham số (Emb) | Tham số (Trm) | Tổng | Thời gian suy luận |
|---|---|---|---|---|
| 3 | 23.8M | 21.3M | 45.7M (2.40×) | 27.35s (3.73×) |
| 6 | 23.8M | 42.5M | 67.0M (1.64×) | 52.51s (1.94×) |
| 12 | 23.8M | 85.1M | 109M (1×) | 101.89s (1×) |

Tốc độ gần tuyến tính theo số layer; embedding (24M, vocab 30k × 768) chia sẻ nên tỉ lệ nén tổng thể nhỏ hơn tỉ lệ số layer.

## 5. Phân tích / ablation

- **Bão hòa trên dev (Figure 2):** student vanilla KD ngừng cải thiện sớm trên QNLI và MNLI; PKD tiếp tục tăng trên dev rồi mới bão hòa muộn hơn. Đây là bằng chứng (đọc từ biểu đồ) cho giả thuyết của tác giả rằng tín hiệu từ tầng giữa giúp tổng quát hóa; giả thuyết "overfitting khi distill" chưa được kiểm tra riêng.
- **Skip so với Last:** Skip nhỉnh hơn (ví dụ MNLI-m 81.5 so với 80.9, QNLI 89.0 so với 88.2) nhưng cả hai đều thắng vanilla KD. Giải thích của tác giả (không kiểm chứng): Skip bắt được biểu diễn đa dạng từ mức thấp tới cao; Last bắt thông tin khá đồng nhất.
- **MRPC:** PKD kém KD trên test (85.0/79.9 so với 86.2/80.6), tác giả giải thích bằng overfit dev trên tập nhỏ; chạy lại ba lần, mean dev accuracy: FT 82.23, KD 82.84, PKD 83.46. RTE (2.5k mẫu) cũng cho khoảng cách lớn so với teacher.
- **Teacher tốt hơn có giúp không? (Table 5):** teacher BERT-Large (BERT24) so với BERT-Base: student BERT6[Base]-KD từ teacher BERT12 và BERT24 không khác nhiều (#1 so với #2: SST-2 91.5 so với 91.2; MNLI-m 79.7 so với 80.2; RTE 64.7 so với 65.7). Student BERT6[Large] (layer cỡ BERT-Large) kém hơn nhiều (#3: SST-2 89.6, MRPC 79.0/70.0, MNLI-m 75.3, RTE 53.7), dù tham số nhiều hơn 1.6 lần. Giải thích của tác giả: tỉ lệ nén 4:1 (24:6) cao hơn 2:1, và student chỉ khởi tạo từ 6 layer đầu của BERT24 (chưa pre-train từ đầu nên nhận layer thấp chưa phản ánh đặc trưng cao). PKD vẫn tốt hơn vanilla KD trong cấu hình này (#4 so với #3 trên hầu hết tác vụ, ví dụ MNLI-m 77.2 so với 75.3), nên tác giả kết luận PKD không phụ thuộc vào chọn teacher.
- **Chỉ `[CLS]` hay mọi token:** tác giả chọn `[CLS]` vì rẻ và ít nhiễu, nhưng paper không có ablation so sánh với khớp mọi token.

## 6. Giới hạn

**Do tác giả nêu:** khởi tạo student chỉ từ layer đầu của teacher có thể không khớp (mismatch), cần pre-train student từ đầu (hướng tương lai); chưa thử loss khoảng cách phức tạp hơn; chưa thử multi-task hay meta-learning; MRPC và RTE (dữ liệu nhỏ) cho kết quả kém hơn.

**Do chúng ta nhận xét:**
- Distill sau khi teacher đã fine-tune cho từng tác vụ (task-specific); mỗi tác vụ phải chạy lại teacher và student. Không phải distill pre-training như bài 16.
- Chỉ ánh xạ layer theo quy tắc cố định (Skip hoặc Last); không có cách tự động chọn ánh xạ, và không có ablation về $\beta$ (cố định $\alpha,T$ từ KD thường rồi chỉ quét $\beta$).
- Cải thiện của PKD so với KD thường nhỏ (khoảng 0.5 đến 1.5 điểm), lại phụ thuộc quét siêu tham số rộng ($\beta$ từ 10 đến 1000) nên chưa chắc ổn định.
- Hidden state phải cùng chiều; chỉ khớp hướng (chuẩn hóa L2) nên bỏ qua độ lớn (norm), vốn có thể mang thông tin trong Transformer (suy luận).
- Khởi tạo student từ các layer đầu của BERT-Base (DistilBERT, bài 16, lấy xen kẽ) là một lựa chọn chưa được so với phương án khác; hai paper không so trực tiếp.
- Chỉ đánh giá trên phân loại câu; `[CLS]` không có ý nghĩa với tác vụ sinh chuỗi.

## 7. Liên hệ với các bài khác

- **03 FitNets, 04 Attention transfer:** cùng ý tưởng "hint" tầng giữa; PKD là phiên bản cho Transformer, chỉ khớp một vector mỗi layer (`[CLS]`), không dùng bộ chuyển đổi chiều (cùng chiều).
- **16 DistilBERT:** cùng thời gian; DistilBERT distill lúc pre-train, khởi tạo xen kẽ, cosine loss ở layer cuối. PKD distill lúc fine-tune, khởi tạo $k$ layer đầu, khớp nhiều layer.
- **18 TinyBERT:** tổng quát hóa ý tưởng này: ánh xạ layer uniform cộng khớp attention và embedding; PKD là tiền thân.
- **19 MiniLM:** đối lập: chỉ khớp layer cuối (self-attention distribution và value relation), không cần ánh xạ layer.
- **08 Teacher assistant, 09 Efficacy of KD, 41 Capacity gap law:** thí nghiệm BERT-Large teacher (Table 5) cho thấy teacher lớn hơn không tự động giúp; student càng nhỏ so với teacher (4:1) càng khó.
- **11 Patient and consistent teacher:** chỉ trùng từ "patient"; ý nghĩa khác (theo tên bài, nói về teacher kiên nhẫn và nhất quán trong cách huấn luyện), đừng nhầm; kiểm tra ở bài 11.
- **12 RKD, 13 CRD:** các cách khớp biểu diễn tầng giữa thay thế.

## 8. Ứng dụng cho dự án (suy luận)

*Mức liên quan: trung bình; trực tiếp nhất ở câu hỏi "khớp layer nào của teacher".*

1. **Layer mapping giữa teacher 4B và student sau cắt.** Dự án phục hồi mô hình đã prune bằng distill từ teacher 4B (khác kích thước và có thể khác số layer/chiều). PKD đưa ra hai lựa chọn mặc định (xen kẽ so với các layer cuối) và phát hiện xen kẽ nhỉnh hơn. Với decoder hybrid (GatedDeltaNet và full-attention) cần mapping có nhận biết loại layer: khớp full-attention của student với full-attention của teacher, v.v. (đề xuất của chúng ta, paper không đề cập).
2. **Điểm cắt và $D_{lin}/D_{res}$.** Khi cắt layer, trạng thái ẩn tại điểm cắt lệch khỏi teacher. Loss kiểu $L_{PT}$ (chuẩn hóa L2) đặt tại điểm cắt chỉ phạt sai hướng; nếu phần $D_{lin}$ chủ yếu là phép co giãn/xoay thì chuẩn hóa này bỏ qua phần lớn hư hại do scale. Nên cân nhắc thêm term đo độ lớn hoặc ước lượng ánh xạ tuyến tính trước (suy luận, chưa kiểm chứng).
3. **Chỉ khớp một vector đại diện thay vì mọi token.** PKD dùng `[CLS]`; OCR VLM không có token tóm tắt tương đương trong decoder, nhưng ở vision encoder có thể khớp vector pooled hoặc một số patch token để tiết kiệm bộ nhớ (suy luận, nhớ rằng RKD/CRD ở bài 12/13 gợi ý khớp quan hệ cho cùng mục tiêu).
4. **Bài học thực nghiệm:** (i) student học bằng loss lớp cuối bão hòa sớm, tín hiệu tầng giữa kéo dài lợi ích; (ii) lợi ích tăng khi nhiều dữ liệu (nhóm tác vụ >60k mẫu); (iii) tỉ lệ nén 4:1 khó hơn 2:1, điều quan trọng khi dự án đi từ 0.8B xuống 0.3B rồi 0.1B (cân nhắc distill bậc thang qua mô hình trung gian, xem 08).
5. **Không mang sang:** BERT, phân loại câu, `[CLS]`, quy trình fine-tune từng tác vụ.

## 9. Thí nghiệm gợi ý

1. **Mapping layer cho student đã prune:** so (a) chỉ KL token, (b) + khớp hidden state ở điểm cắt (cosine), (c) + khớp ở 3-4 điểm xen kẽ. Cùng ngân sách bước. Đo CER dev và đường cong dev theo bước để kiểm tra hiện tượng bão hòa sớm của (a).
2. **Hướng so với độ lớn:** thử chuẩn hóa L2 (như PKD) và không chuẩn hóa (MSE thuần), đánh giá riêng cho cắt layer và cắt neuron MLP; xem lệch độ lớn có phải phần chính của $D_{lin}$ không.
3. **Distill bậc thang cho nén 4:1:** 0.8B → 0.3B → 0.1B thay vì 0.8B → 0.1B trực tiếp, xác nhận quan sát về tỉ lệ nén (BERT6[Large] kém hơn BERT6[Base]).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Patient-KD | Student học từ nhiều layer trung gian của teacher, không chỉ lớp cuối |
| PKD-Skip | $I_{pt}=\{2,4,6,8,10\}$ cho 12 → 6 layer |
| PKD-Last | $I_{pt}=\{7,8,9,10,11\}$ cho 12 → 6 layer |
| $k$ | Số layer được khớp; luôn $n-1$ cho student $n$ layer |
| `[CLS]` | Token tóm tắt của BERT; PKD chỉ khớp token này |
| $L_{PT}$ | MSE giữa hidden state chuẩn hóa L2 của student và teacher |
| $L_{PKD}$ | $(1-\alpha)L_{CE}+\alpha L_{DS}+\beta L_{PT}$ |
| $\beta$ | Trọng số khớp tầng giữa, quét $\{10,100,500,1000\}$ |
| Khởi tạo | $k$ layer đầu của BERT-Base pre-trained |
| Kết quả | BERT6: 1.64× nhỏ hơn, 1.94× nhanh hơn; gần teacher trên 5/7 tác vụ GLUE |

## 11. Kết luận ngắn

PKD chỉ ra rằng chỉ học từ lớp cuối của teacher làm student bão hòa sớm, trong khi khớp `[CLS]` ở các layer trung gian (nhất là xen kẽ, PKD-Skip) tiếp tục cải thiện và đưa BERT6 gần BERT12 trên các tác vụ nhiều dữ liệu. Với dự án của bạn, giá trị lớn nhất là câu hỏi thiết kế mapping layer giữa teacher và student đã cắt, và bài học rằng tỉ lệ nén lớn hơn thì cần trung gian hoặc tín hiệu giàu hơn chứ không chỉ đầu ra cuối.

Link gốc: https://arxiv.org/abs/1908.09355
