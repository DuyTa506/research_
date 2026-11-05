# Gemma 2: Improving Open Language Models at a Practical Size (2024)

**Tác giả:** Gemma Team (Google DeepMind) — Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, Johan Ferret và nhiều tác giả khác (danh sách đầy đủ trong paper).  
**Venue / arXiv:** arXiv:2408.00118 (v3, 2 Oct 2024); paper không nêu venue hội nghị.  
**Nguồn:** https://arxiv.org/abs/2408.00118  
**Chủ đề:** Technical report. Ghi chú này chỉ tập trung vào **knowledge distillation (KD) ở giai đoạn pre-training** cho mô hình 2B/9B và các ablation liên quan (Mục 3.2 và Mục 5 của paper). Kiến trúc, an toàn, đánh giá benchmark chỉ nhắc ngắn.  
**Vị trí trong lộ trình:** Đọc trước: 02 (Hinton KD), 15 (Sequence-level KD), 27 (LLM KD survey), 28–29 (MiniLLM, GKD). Bài này cho thấy KD dùng ở **quy mô pre-training công nghiệp**: thay nhãn one-hot bằng phân phối của teacher trên hàng nghìn tỷ token. Tiếp theo: 41 (Capacity gap law), 42 (Distillation scaling laws) giải thích *khi nào* và *vì sao* cách làm này có lợi; 44 (Qwen3) là một công thức công nghiệp khác.

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch paper. Phần "Ứng dụng cho dự án" là suy luận của chúng tôi, không phải kết luận của paper. Paper là technical report nên mô tả KD rất ngắn và thiếu nhiều chi tiết (không nêu kích thước teacher thật cho 9B/2B ngoài việc mô phỏng bằng teacher 7B).

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

**Vấn đề.** Các mô hình nhỏ (vài tỷ tham số) ngày càng tốt chủ yếu nhờ **train lâu hơn** (nhiều token hơn). Paper nhận xét cách này chỉ cải thiện theo hàm *logarit* của kích thước dữ liệu (theo Hoffmann et al., 2022 — "Chinchilla"), và các mô hình nhỏ mới nhất cần tới 15T token để cải thiện state-of-the-art chưa tới 1–2%. Nghĩa là mô hình nhỏ vẫn "under-trained", nhưng thêm token ngày càng kém hiệu quả.

**Chinchilla / compute-optimal** (gloss): với ngân sách tính toán cố định, có một cặp (kích thước mô hình, số token) tối ưu; mô hình 2B "compute-optimal" chỉ cần khoảng vài chục tỷ token. Các mô hình hiện đại train *vượt xa* mức này ("over-training") vì chi phí suy luận quan trọng hơn chi phí train.

**Ý tưởng thay thế.** Thay vì chỉ tăng độ dài train, tăng **chất lượng thông tin mỗi bước train**. Next-token prediction chuẩn dùng nhãn one-hot (đúng một token là 1, còn lại 0). Distillation thay one-hot bằng cả phân phối xác suất của một mô hình lớn (teacher), nên mỗi token cho nhiều bit thông tin hơn ("richer gradients").

**Điểm mới của Gemma 2:** KD trước đây thường dùng để *rút ngắn thời gian train* mô hình nhỏ. Ở đây họ dùng KD để *train rất nhiều token* ("simulate training beyond the number of available tokens"): 2B train 2T token, 9B train 8T token, tức hơn 50× lượng token compute-optimal theo lý thuyết.

## 2. Ý tưởng chính

Mô hình Gemma 2 2B và 9B được train trên văn bản thô, nhưng thay vì dự đoán token tiếp theo thật, student học khớp **phân phối xác suất token tiếp theo do teacher (một LLM lớn) tính ra** tại từng vị trí. Cách này cho student tín hiệu giàu hơn nhiều mỗi token, nên cùng số token thì student tốt hơn hẳn, và ta có thể train rất lâu mà vẫn còn lợi. Mô hình 27B thì train từ đầu (không distill). Phần ablation cho thấy: distill từ teacher 7B cho 2B (500B token) tăng điểm trung bình từ 60.3 lên 67.7, và lợi ích giữ nguyên khi tăng kích thước student.

## 3. Phương pháp chi tiết

### 3.1 Hàm mục tiêu pre-training

Cho ngữ cảnh $x_c$ (các token trước) và token tiếp theo $x$. Teacher cho $P_T(x\mid x_c)$, student có tham số $\theta$ cho $P_S(x\mid x_c)$. Paper viết mục tiêu là cực tiểu hóa cross-entropy giữa hai phân phối:

$$
\boxed{\min_{P_S}\; \sum_{x} -P_T(x\mid x_c)\,\log P_S(x\mid x_c)}
$$

Đây là **forward KL** (đến một hằng số không phụ thuộc student): $\mathrm{KL}(P_T\Vert P_S)=\sum_x P_T\log P_T-\sum_x P_T\log P_S$, số hạng đầu không phụ thuộc $P_S$. Khi $P_T$ là one-hot ta quay về loss next-token thông thường. Tức SFT/pre-train chuẩn là **trường hợp đặc biệt** với teacher "cứng".

Gloss: *forward KL* khiến student phải "phủ" mọi chế độ (mode) có xác suất đáng kể của teacher (mode-covering), khác với reverse KL của MiniLLM (28) vốn mode-seeking.

Chi tiết paper **không** nêu (cần ghi nhớ khi tái hiện): nhiệt độ softmax, có cắt top-k logit hay không, teacher logit được tính online hay cache, trộn với loss one-hot hay không. Bài 45 (Sparse logit sampling) bàn đúng vấn đề lưu/cắt logit ở quy mô này.

### 3.2 Pseudo-code

```
for each batch of raw text tokens x_1..x_T:
    # teacher forward (frozen), cho phân phối tại mọi vị trí
    P_T[t] = softmax(teacher(x_<=t))            # kích thước |V| = 256k
    P_S[t] = softmax(student(x_<=t))
    loss   = mean_t( - sum_v P_T[t][v] * log P_S[t][v] )
    update student by gradient of loss
```

Lưu ý quy mô: từ vựng là 256,128 token, nên phân phối đầy đủ mỗi vị trí rất lớn; đây là một trở ngại thực tế của KD ở pre-training (paper không bàn cách xử lý).

### 3.3 Hai chỗ khác dùng distillation (post-training)

Trong SFT, họ làm **hai việc**: (i) behavioral cloning trên prompt thật/tổng hợp với câu trả lời chủ yếu do teacher sinh (tức sequence-level KD kiểu 15); (ii) **distillation trên phân phối của chính student** (on-policy), trích dẫn Agarwal et al. 2024 (GKD, 29) và Gu et al. 2024 (MiniLLM, 28). Paper chỉ nói vậy, không có ablation riêng cho phần này.

### 3.4 Dữ liệu và kiến trúc kèm theo (tóm tắt)

| | 2B | 9B | 27B |
|---|---|---|---|
| Số token train | 2T | 8T | 13T |
| Cách train | distillation | distillation | từ đầu |
| $d_{model}$ | 2304 | 3584 | 4608 |
| Số layer | 26 | 42 | 46 |
| Tham số non-embedding | 2,024,517,888 | 8,324,201,984 | 26,047,480,320 |
| Tham số embedding | 590,118,912 | 917,962,752 | 1,180,237,824 |

Kiến trúc: xen kẽ local sliding-window attention (cửa sổ 4096) và global attention (span 8192) mỗi hai layer; GQA (num_groups = 2); logit soft-capping (50.0 trong attention, 30.0 ở layer cuối); RMSNorm cả pre-norm và post-norm; từ vựng 256k. Hạ tầng: 2B trên 512 chip TPUv5e, 9B trên 4096 chip TPUv4, 27B trên 6144 chip TPUv5p.

## 4. Thí nghiệm và kết quả chính

**Ablation KD (Mục 5 paper).** Teacher 7B; student 2B huấn luyện 500B token (gấp 10× token compute-optimal của mô hình 2B).

| Bảng | Thiết lập | From scratch | Distilled |
|---|---|---|---|
| Table 6 | 2B, 500B token, điểm TB trên 3 benchmark | 60.3 | 67.7 |

(Paper không liệt kê tên 3 benchmark trong chú thích bảng này.)

**Table 7: perplexity trên validation set (thấp hơn là tốt), teacher cố định 7B.**

| Kích thước student | 200M | 400M | 1B |
|---|---|---|---|
| From scratch | 23 | 19 | 17 |
| Distilled (7B) | 21 | 17 | 15 |

Giải thích của paper: họ giữ teacher 7B và train student nhỏ hơn để *mô phỏng tỷ lệ kích thước giữa teacher và student thật* (27B → 9B), và thấy lợi ích "vẫn còn khi scale kích thước".

**Kết quả chất lượng cuối (pre-trained).** Gemma 2 2B (distilled) so với Gemma 1 2B (cùng cỡ, 2T vs 3T token): MMLU 52.2 vs 42.3, GSM8K 24.3 vs 15.1, ARC-C 55.7 vs 48.5, MATH 16.0 vs 11.8. Gemma 2 9B: MMLU 71.3, GSM8K 68.6, MATH 36.6. Gemma 2 27B (không distill): MMLU 75.2, GSM8K 74.0, MATH 42.3. Paper nói mô hình mới tốt hơn bản cũ "tới 10% ở một số benchmark cho 9B" và kết luận distillation cải thiện chất lượng "kể cả khi train cùng số token".

Post-training: Gemma 2 27B đạt Elo 1218 trên LMSYS Chatbot Arena (cao hơn Llama 3 70B, 1206), 9B đạt 1187, 2.6B đạt 1126. Đây là kết quả của *toàn bộ pipeline* (pre-train + SFT + RLHF + merging), không tách riêng được đóng góp của KD.

## 5. Phân tích / ablation

**Paper cho thấy:**
- Distill từ teacher lớn hơn tốt hơn train từ đầu ở cùng số token (Table 6, 7).
- Lợi ích không biến mất khi student lớn lên (từ 200M tới 1B), ít nhất trong dải này.
- Các ablation kiến trúc khác (không phải KD): deep 9B hơn wide 9B một chút (52.0 vs 50.8, trung bình 4 benchmark); GQA gần như ngang MHA (50.8 vs 50.3); giảm sliding window từ 4096 xuống 1024 lúc suy luận chỉ làm perplexity từ 1.63 lên 1.64.
- Độ nhạy theo format prompt (độ lệch chuẩn MMLU qua 12 tổ hợp): Gemma 2 2B 2.1, 9B 0.9, 27B 1.0.

**Paper KHÔNG ablation (quan sát của chúng tôi):**
- Không so sánh các loss khác (reverse KL, JS, top-k), không thay đổi nhiệt độ.
- Không biến thiên kích thước teacher ở bước pre-training (chỉ một teacher 7B trong ablation, teacher thật cho 9B/2B không được nêu rõ trong paper).
- Không tách ảnh hưởng của "nhiều token hơn" khỏi "distill" ở mô hình cuối, vì Gemma 2 2B vs Gemma 1 2B còn khác kiến trúc và dữ liệu. Bằng chứng KD sạch nhất chỉ là Table 6/7 (quy mô nhỏ).
- Không có chi phí tính toán của teacher trong so sánh. Distill ở 500B token đòi hỏi forward teacher 7B trên 500B token, nên "cùng số token" không có nghĩa "cùng compute". Đây chính là điều bài 42 sẽ định lượng.

## 6. Giới hạn

**Do tác giả nêu:**
- Ở khâu kiểm tra, paper nói thử nghiệm không bao quát mọi tình huống ứng dụng; người dùng cần tự kiểm tra an toàn.
- Mô hình không multimodal, chủ yếu tiếng Anh, không tối ưu đa ngôn ngữ.

**Do chúng tôi nhận xét:**
- Mô tả KD chỉ nửa trang; thiếu hyperparameter, dữ liệu teacher, chi phí.
- Ablation chỉ ở quy mô nhỏ (≤ 2B, teacher 7B) rồi ngoại suy lên 9B/27B.
- Không có đường cong theo số token hay so sánh với train-from-scratch cho 2B/9B ở token thật.
- Không có kiểm tra khi nào teacher quá lớn gây hại (capacity gap, bài 41).

## 7. Liên hệ với các bài khác

- **02 Hinton KD:** loss pre-training giống hệt ý tưởng "soft targets", chỉ khác là áp dụng ở mọi token của corpus thô.
- **15 Sequence-level KD, 36–37 (step-by-step, Orca):** phần SFT của Gemma 2 dùng dữ liệu do teacher sinh, là kiểu sequence-level.
- **28 MiniLLM, 29 GKD:** Gemma 2 trích hai bài này cho on-policy distillation trong SFT.
- **41 Capacity gap law:** giải thích vì sao teacher không nên quá lớn so với student; Gemma 2 chọn tỷ lệ 27B→9B và mô phỏng bằng 7B→200M–1B.
- **42 Distillation scaling laws:** trả lời định lượng câu hỏi "khi nào distill đáng hơn train từ đầu" mà Gemma 2 chỉ trả lời bằng thực nghiệm đơn lẻ; cũng tính chi phí teacher.
- **43 MiniPLM:** cách khác để KD ở pre-training mà không tốn forward teacher online.
- **44 Qwen3:** cũng dùng teacher lớn nhưng tập trung vào post-training (strong-to-weak).
- **45 Sparse logit sampling:** bàn cách lưu và dùng logit teacher ở vocab lớn khi pre-train.

## 8. Ứng dụng cho dự án (suy luận)

Dự án của bạn là recovery sau pruning, không phải pre-training từ đầu, nên mức liên quan *trung bình*:

1. **Loss.** Dạng cross-entropy với phân phối teacher chính là loss recovery chuẩn: teacher 4B cho phân phối token tại mỗi vị trí transcript OCR, student đã prune học khớp. Bài học thực dụng: dùng soft target trên *mọi token* thay vì chỉ nhãn thật.
2. **Teacher/student ratio.** Gemma 2 mô phỏng tỷ lệ 27B:9B = 3:1. Dự án có 4B → 0.3B (≈13:1) và 4B → 0.1B (≈40:1), lớn hơn nhiều so với thiết lập đã được kiểm chứng ở đây. Đây là lý do cần đọc 41 và 42 trước khi tin rằng teacher 4B luôn tối ưu cho student 0.1B.
3. **Vocab lớn.** OCR VLM có từ vựng nhỏ hơn 256k nên việc lưu logit teacher đầy đủ dễ hơn; có thể tính logit teacher offline một lần trên bộ dữ liệu cố định.
4. **Phạm vi.** Gemma 2 train student *từ khởi tạo ngẫu nhiên*. Dự án bắt đầu từ trọng số đã prune, nên token cần ít hơn nhiều; kết quả "50× compute-optimal" không chuyển trực tiếp.
5. **Không nên suy ra** rằng KD sẽ sửa được damage $D_{res}$ do cắt MLP neuron. Paper không đề cập pruning.

## 9. Thí nghiệm gợi ý

1. **Soft vs hard ở recovery.** Với một mô hình đã prune, so ba loss: (a) chỉ CE one-hot, (b) chỉ forward KL từ teacher 4B, (c) trộn. Cùng số bước, so CER/WER trên tập OCR dev. Mục tiêu: lặp lại trực tiếp Table 6 ở quy mô dự án.
2. **Cắt top-k logit.** Lưu top-k (ví dụ 50, 200) xác suất teacher thay vì toàn bộ; đo mất mát so với full, để quyết định cache offline (liên hệ bài 45).
3. **Student nhỏ hơn, teacher giữ nguyên.** Giống Table 7: giữ teacher 4B, student 0.3B và 0.1B (cùng recovery budget), xem lợi ích KD so với CE có giảm khi tỷ lệ teacher/student tăng không.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa ngắn |
|---|---|
| $\min_{P_S}\sum_x -P_T\log P_S$ | Cross-entropy với phân phối teacher; tương đương forward KL |
| Pre-training distillation | KD áp dụng trên corpus thô thay cho next-token one-hot |
| Over-training | Train nhiều token hơn mức compute-optimal (ở đây >50×) |
| Soft-capping | $z\leftarrow c\tanh(z/c)$, c=50 (attention), 30 (logit cuối) |
| Table 6 | 2B, 500B token, teacher 7B: 60.3 → 67.7 |
| Table 7 | PPL 23→21, 19→17, 17→15 cho 200M/400M/1B |
| On-policy distillation (SFT) | Distill trên mẫu do chính student sinh (GKD/MiniLLM) |

## 11. Kết luận ngắn

Gemma 2 là bằng chứng công nghiệp rằng distillation ở pre-training cho mô hình nhỏ là đáng làm: cùng token, student tốt hơn rõ (60.3 → 67.7 ở ablation 2B). Tuy nhiên paper là technical report, bằng chứng KD chủ yếu là hai bảng nhỏ và không tính chi phí teacher hay bàn kích thước teacher tối ưu. Để hiểu giới hạn, đọc tiếp 41 và 42.

**Link gốc:** https://arxiv.org/abs/2408.00118
