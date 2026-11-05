# DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter (2019)

**Tác giả:** Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf (Hugging Face).  
**Venue / arXiv:** arXiv:1910.01108 (v4, 1/3/2020); bản text ghi "EMC^2: 5th Edition, co-located with NeurIPS'19" (workshop), không phải hội nghị chính.  
**Nguồn:** https://arxiv.org/abs/1910.01108  
**Chủ đề:** Distillation lúc *pre-training* cho mô hình ngôn ngữ (BERT): loss ba thành phần, khởi tạo student bằng cách lấy một layer trong hai của teacher, giảm số layer.  
**Vị trí trong lộ trình:** Đọc sau 02 (Hinton KD) và 15 (Seq-KD). Đây là công thức "giữ một nửa số layer, khởi tạo từ teacher, thêm distill" đơn giản nhất cho Transformer; 17 (Patient KD) nói về chọn layer teacher để khớp, 18 (TinyBERT) và 19 (MiniLM) thêm distill attention và hidden state.

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

Năm 2018-2019, NLP chuyển sang **transfer learning**: pre-train một mô hình lớn (BERT, GPT-2, RoBERTa) trên văn bản không nhãn, rồi *fine-tune* (huấn luyện tiếp) cho từng tác vụ. Mô hình có hàng trăm triệu tham số và xu hướng còn tăng, kéo theo chi phí môi trường/tính toán và khó chạy on-device.

Trước đó, distillation trong NLP chủ yếu là **task-specific**: lấy BERT đã fine-tune cho một tác vụ, distill sang một mô hình nhỏ chỉ cho tác vụ đó (ví dụ BERT → LSTM của Tang et al. 2019). Bài này distill ở giai đoạn **pre-training** để thu được một mô hình nhỏ đa dụng, vẫn fine-tune được cho nhiều tác vụ như BERT.

Khái niệm: **masked language modeling (MLM)** là tác vụ pre-train của BERT: che một số token trong câu và dự đoán lại chúng. Lớp output trên mỗi token bị che là phân phối trên cả vocab; phần đuôi xác suất ("day" và "life" cao, "future, story, world" thấp mà vẫn hợp lý, ví dụ của paper) chứa "dark knowledge".

## 2. Ý tưởng chính

DistilBERT dùng đúng kiến trúc BERT nhưng **giảm một nửa số layer** (bỏ token-type embedding và pooler) và khởi tạo student bằng cách **lấy một layer trong hai** của teacher BERT-base. Sau đó pre-train student trên cùng dữ liệu với BERT bằng một loss ba thành phần: MLM thông thường, cross-entropy tới phân phối mềm của teacher (có nhiệt độ), và cosine loss kéo hidden state student cùng hướng với teacher. Kết quả: nhỏ hơn 40%, nhanh hơn 60%, giữ 97% điểm GLUE của BERT.

## 3. Phương pháp chi tiết

**Kiến trúc student.** Cùng khung BERT, số layer giảm một nửa (hidden size giữ nguyên). Lý do chọn giảm layer thay vì hidden size: tác giả thấy biến thể chiều cuối của tensor (hidden size) ảnh hưởng ít đến hiệu quả tính toán (với ngân sách tham số cố định) hơn so với số layer, vì các phép toán nền (linear, LayerNorm) đã được tối ưu cao.

**Khởi tạo.** Vì teacher và student cùng số chiều, student lấy trọng số từ teacher bằng cách lấy một layer trong hai (ví dụ layer 0, 2, 4, ...).

**Loss.** Với nhiệt độ $T$ (cùng $T$ cho student và teacher lúc train; khi suy luận $T=1$):

$$p_i=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)},\qquad L_{ce}=\sum_i t_i\log s_i\ \ (\text{lấy dấu âm khi cực tiểu hóa})$$

trong đó $t_i$ là xác suất của teacher, $s_i$ của student (công thức trong paper viết $\sum_it_i\log(s_i)$, ngầm hiểu là cross-entropy). Loss cuối cùng là tổ hợp tuyến tính

$$\boxed{L=L_{ce}+L_{mlm}+L_{cos}}$$

(paper gọi là "triple loss", nói là "linear combination" nhưng *không nêu giá trị trọng số* trong bản text này). $L_{mlm}$ là loss MLM trên nhãn thật; $L_{cos}$ là **cosine embedding loss** kéo hướng của vector hidden state của student gần hướng của teacher (cosine giữa hai hidden state càng gần 1 càng tốt).

**Huấn luyện.** Làm theo thực hành tốt của RoBERTa (Liu et al. 2019): batch rất lớn (tối đa 4K mẫu, dùng gradient accumulation), dynamic masking, bỏ next-sentence prediction. Dữ liệu giống BERT gốc: English Wikipedia và Toronto Book Corpus. Compute: 8 GPU V100 16GB trong khoảng 90 giờ (so với RoBERTa cần 1 ngày trên 1024 GPU V100 32GB, theo paper).

### Pseudo-code

```python
student = copy_layers(teacher, layers=[0, 2, 4, 6, 8, 10])   # 12 -> 6 layer; bỏ token-type emb + pooler
for batch in wiki_bookcorpus_loader(batch_size=4096, dynamic_masking=True):  # không có NSP
    with no_grad(): t_logits, t_hidden = teacher(batch.masked)
    s_logits, s_hidden = student(batch.masked)
    L_ce  = soft_cross_entropy(s_logits[mask] / T, t_logits[mask] / T)        # trên token bị che
    L_mlm = cross_entropy(s_logits[mask], batch.labels[mask])
    L_cos = cosine_embedding_loss(s_hidden, t_hidden)                         # kéo hướng gần nhau
    loss = L_ce + L_mlm + L_cos                                               # paper không nêu hệ số
```

(Phạm vi áp dụng $L_{cos}$ — hidden state cuối hay mọi layer — không nêu rõ trong bản text; pseudo-code trên là minh họa của chúng ta.)

### Ví dụ nhỏ

Câu "I think this is the beginning of a beautiful [MASK]". Teacher: "day" 0.4, "life" 0.3, "future" 0.05, ... CE one-hot với nhãn thật "day" chỉ cho tín hiệu 1 token. $L_{ce}$ với $T>1$ cho student biết "life" cũng hợp lý và "banana" thì không. (Con số minh họa của chúng ta; ví dụ câu lấy từ chú thích 3 của paper.)

## 4. Thí nghiệm và kết quả chính

### GLUE (dev set, trung vị 5 lần chạy, Table 1)

| Model | Score | CoLA | MNLI | MRPC | QNLI | QQP | RTE | SST-2 | STS-B | WNLI |
|---|---|---|---|---|---|---|---|---|---|---|
| ELMo | 68.7 | 44.1 | 68.6 | 76.6 | 71.1 | 86.2 | 53.4 | 91.5 | 70.4 | 56.3 |
| BERT-base | 79.5 | 56.3 | 86.7 | 88.6 | 91.8 | 89.6 | 69.3 | 92.7 | 89.0 | 53.5 |
| DistilBERT | 77.0 | 51.3 | 82.2 | 87.5 | 89.2 | 88.5 | 59.9 | 91.3 | 86.9 | 56.3 |

DistilBERT luôn ngang hoặc hơn baseline ELMo (tới 19 điểm trên STS-B) và giữ 97% điểm của BERT với ít hơn 40% tham số. Các tác vụ nhỏ như RTE (69.3 → 59.9) và CoLA (56.3 → 51.3) mất nhiều nhất.

### Downstream (Table 2)

| Model | IMDb (acc) | SQuAD 1.1 (EM/F1, dev) |
|---|---|---|
| BERT-base | 93.46 | 81.2 / 88.5 |
| DistilBERT | 92.82 | 77.7 / 85.8 |
| DistilBERT (D), thêm bước distill lúc fine-tune | không có | 79.1 / 86.9 |

IMDb kém BERT 0.6 điểm; SQuAD kém trong 3.9 điểm (nói trong text); thêm bước distill thứ hai (fine-tune trên SQuAD với teacher là BERT đã fine-tune SQuAD) thu hẹp khoảng cách còn trong vòng 3 điểm. Lưu ý: văn bản nói "79.8 F1 và 70.4 EM" cho trường hợp này, không khớp với bảng (79.1 EM / 86.9 F1); đây là bất nhất nội bộ trong paper, chúng ta không biết số nào đúng.

### Kích thước và tốc độ (Table 3)

| Model | Tham số (triệu) | Thời gian suy luận STS-B dev, CPU, batch 1 (giây) |
|---|---|---|
| ELMo | 180 | 895 |
| BERT-base | 110 | 668 |
| DistilBERT | 66 | 410 |

Nhỏ hơn 40%, nhanh hơn 60% (theo paper; 410/668 = 0.61 thời gian, tức nhanh hơn khoảng 1.6 lần). Trên iPhone 7 Plus (không tính tokenization) cho QA, DistilBERT nhanh hơn 71% so với BERT-base, mô hình nặng 207 MB (có thể giảm thêm bằng quantization).

## 5. Phân tích / ablation

Table 4, thay đổi GLUE macro-score so với mô hình đầy đủ (triple loss + khởi tạo từ teacher):

| Biến thể | Δ macro-score |
|---|---|
| Bỏ $L_{ce}$ | −2.96 |
| Bỏ $L_{cos}$ | −1.46 |
| Bỏ $L_{mlm}$ | −0.31 |
| Giữ triple loss nhưng khởi tạo ngẫu nhiên | −3.69 |

(Bảng gốc dùng ký hiệu ∅ đánh dấu thành phần bị bỏ; thứ tự dòng và giá trị khớp với câu "removing MLM loss has little impact while the two distillation losses account for a large portion".)

Kết luận của paper: (1) khởi tạo từ teacher là thành phần quan trọng nhất (−3.69 khi bỏ); (2) cả hai loss distillation đều quan trọng, $L_{ce}$ nhiều hơn $L_{cos}$; (3) loss MLM gần như không cần khi đã có hai loss kia.

So với công trình khác (Related work): Turc et al. 2019 pre-train student bằng mục tiêu gốc rồi mới distill lúc fine-tune; Tsai et al. 2019 pre-train đa ngôn ngữ chỉ bằng distillation; DistilBERT khẳng định rằng tận dụng teacher qua khởi tạo và loss phụ cho lợi ích rõ rệt. Pruning head attention (Michel et al. 2019) và quantization được xem là trực giao.

## 6. Giới hạn

**Do tác giả nêu:** paper là báo cáo workshop ngắn; ghi nhận pruning/quantization chưa kết hợp (nói trực giao); chưa so với nhiều biến thể khác.

**Do chúng ta nhận xét:**
- Không nêu trọng số của ba loss, nhiệt độ $T$ cụ thể, hay phạm vi áp $L_{cos}$ trong bản text; không tái lập được chỉ từ paper.
- Chỉ thử một cấu hình (6 layer, hidden size cố định); không quét độ sâu, không so với "bỏ layer rồi fine-tune không distill" (baseline đó chỉ gián tiếp qua ablation khởi tạo ngẫu nhiên).
- Khoảng cách trên tác vụ nhỏ (RTE −9.4, CoLA −5.0) cho thấy "97%" là trung bình bị che bởi các tác vụ dễ.
- Khởi tạo "một layer trong hai" chỉ khả thi khi student có cùng hidden size; student khác chiều không thể làm vậy.
- Tốc độ đo trên CPU batch 1 cho STS-B, chỉ là một tác vụ ngắn; không có đo độ trễ cho chuỗi dài.

## 7. Liên hệ với các bài khác

- **02 Hinton KD:** $L_{ce}$ chính là loss Hinton với nhiệt độ.
- **03 FitNets, 04 Attention transfer:** $L_{cos}$ là loss khớp hidden state đơn giản (chỉ khớp hướng).
- **17 Patient KD BERT:** chọn tập layer teacher cụ thể để khớp, thay cho "layer chẵn" cố định.
- **18 TinyBERT, 19 MiniLM:** distill thêm attention map/value-relation và hidden state, thường giảm cả chiều.
- **08 Teacher assistant, 41 Capacity gap law:** DistilBERT chỉ giảm một nửa nên khoảng cách nhỏ; với giảm lớn hơn cần các bài này.
- **Pruning:** DistilBERT là một dạng depth pruning (bỏ layer luân phiên) + distill, cùng loại với "cắt layer" ở dự án.

## 8. Ứng dụng cho dự án (suy luận)

*Mức liên quan: trung bình đến cao cho phần cắt layer.*

1. **Mô hình tham chiếu cho cắt layer.** Dự án cắt layer của decoder hybrid (GatedDeltaNet và full-attention) hoặc ViT rồi phục hồi bằng distill. DistilBERT cho công thức: giữ khoảng một nửa số layer, khởi tạo student từ layer của chính teacher (ở dự án là layer còn lại của mô hình đã prune, sẵn có), rồi distill. Ablation cho thấy việc khởi tạo từ teacher đáng giá hơn bất kỳ loss nào (−3.69), phù hợp với việc dự án đã prune từ chính teacher (khởi tạo thừa hưởng).
2. **Chọn layer nào giữ:** paper dùng pattern cố định "lấy một layer trong hai" và không so sánh các cách chọn khác. Với decoder hybrid, layer tuyến tính (GatedDeltaNet) và full-attention không đối xứng nên giữ luân phiên có thể sai; cần thử vị trí. (Suy luận, paper không đề cập.)
3. **Triple loss:** $L_{ce}+L_{mlm}+L_{cos}$ tương ứng ở dự án với KL token + CE nhãn + hidden-state cosine tại điểm cắt. $L_{cos}$ chỉ kéo hướng, không ép độ lớn, nhẹ hơn MSE; có thể thử như một loss neo ở điểm cắt (liên quan đến $D_{lin}$ nếu phần hỏng chủ yếu là phép quay/thang đo).
4. **Hidden size cố định** (không cắt neuron MLP/chiều ẩn) là điều DistilBERT chọn vì lý do hiệu quả: số layer ảnh hưởng latency nhiều hơn hidden size. Điều này khớp quan sát của dự án rằng cắt layer phục hồi tốt hơn cắt neuron MLP, nhưng paper không so sánh phục hồi giữa hai loại cắt (đó là suy luận của chúng ta).
5. **Ít liên quan:** MLM, GLUE, tokenization, và việc distill ở giai đoạn pre-training với corpus lớn không áp dụng; dự án phục hồi bằng lượng dữ liệu OCR nhỏ hơn rất nhiều.

## 9. Thí nghiệm gợi ý

1. **Khởi tạo alternating layer:** với decoder đã prune, so (a) giữ layer luân phiên, (b) giữ các layer đầu và cuối, (c) giữ các layer có "importance" cao nhất; cùng ngân sách distill. Đo CER sau 1-2k bước.
2. **Ablation triple loss thu nhỏ:** KL token + CE nhãn ± cosine hidden tại điểm cắt; đo xem $L_{cos}$ có đóng góp như −1.46 trong paper không.
3. **Random init vs teacher init:** xác nhận con số −3.69 trong bối cảnh OCR bằng cách khởi tạo student một nửa số layer từ ngẫu nhiên.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Pre-training distillation | Distill lúc pre-train, student đa dụng |
| Task-specific distillation | Distill lúc fine-tune cho một tác vụ |
| Student | BERT 6 layer (một nửa), bỏ token-type embedding và pooler |
| Khởi tạo | Lấy một layer trong hai của teacher |
| $L_{ce}$ | Cross-entropy tới xác suất mềm của teacher, nhiệt độ $T$ |
| $L_{mlm}$ | Loss masked language modeling trên nhãn thật |
| $L_{cos}$ | Cosine embedding loss giữa hidden state student và teacher |
| Triple loss | $L_{ce}+L_{mlm}+L_{cos}$ (hệ số không nêu trong bản text) |
| Kết quả chính | −40% tham số, +60% tốc độ, 97% GLUE |

## 11. Kết luận ngắn

DistilBERT cho thấy một công thức đơn giản (bỏ nửa số layer, khởi tạo từ teacher, loss mềm + cosine + MLM) là đủ để thu được mô hình nhỏ giữ 97% GLUE của BERT, nhanh hơn 60%. Bài học thực tế cho dự án: khởi tạo từ teacher và giảm độ sâu là cách rẻ và hiệu quả, nhưng paper không cho biết cách chọn layer tối ưu hay hệ số loss, nên phải tự khảo sát.

Link gốc: https://arxiv.org/abs/1910.01108
