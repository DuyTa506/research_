# Qwen3 Technical Report (2025)

**Tác giả:** Qwen Team (An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu và nhiều đồng tác giả; danh sách đầy đủ trong paper).  
**Venue / arXiv:** arXiv:2505.09388 (technical report, ngày 14 May 2025); paper không nêu venue hội nghị.  
**Nguồn:** https://arxiv.org/abs/2505.09388  
**Chủ đề:** Technical report dài về họ mô hình Qwen3. Ghi chú này **chỉ tập trung vào Mục 4.5 "Strong-to-Weak Distillation"** (distill off-policy rồi on-policy cho mô hình nhỏ) và **phép so sánh chi phí với RL trong Mục 4.7**. Pre-training, kiến trúc, quy trình 4 giai đoạn của mô hình lớn và các benchmark chỉ tóm tắt vừa đủ làm bối cảnh.  
**Vị trí trong lộ trình:** Đọc trước: 02 (Hinton KD), 15 (Sequence-level KD), 28–29 (MiniLLM, GKD: on-policy), 38–39 (DeepSeek-R1 distillation; Small models struggle with strong reasoners). Bài này thêm: một công thức công nghiệp "off-policy rồi on-policy" cho mô hình 0,6B–14B, và một con số so sánh chi phí với RL. Tiếp theo: 45 (Sparse logit sampling) và 47 (ShortOPD).

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch. Phần distillation trong paper rất ngắn (một mục khoảng 10 dòng và một bảng); nhiều chi tiết (số mẫu, siêu tham số, tỷ lệ trộn, độ dài rollout) **không được công bố**. Phần "Ứng dụng cho dự án" là suy luận của chúng tôi.

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

**Họ Qwen3** gồm 6 mô hình dense (0,6B; 1,7B; 4B; 8B; 14B; 32B) và 2 MoE (30B-A3B và 235B-A22B, tức tổng 235B tham số, 22B được kích hoạt mỗi token). Tất cả pre-train trên khoảng **36 nghìn tỷ token**, 119 ngôn ngữ, ba giai đoạn (≈30T token nền tảng; sau đó dữ liệu giàu kiến thức STEM/code; cuối cùng dữ liệu ngữ cảnh dài mở rộng 4.096 → 32.768). Đáng chú ý cho dự án OCR: paper nói họ dùng **Qwen2.5-VL để nhận dạng chữ trong lượng lớn tài liệu PDF**, rồi tinh chỉnh bằng Qwen2.5 để có thêm hàng nghìn tỷ token văn bản chất lượng cao. Tài liệu này không nói mô hình nhỏ dùng KD ở pre-training.

**Thinking mode / non-thinking mode** (gloss): một mô hình có hai cách trả lời, "thinking" sinh chuỗi suy luận dài trong khối `<think>...</think>` rồi mới trả lời; "non-thinking" trả lời thẳng với khối think rỗng. Người dùng chọn bằng cờ `/think` và `/no think`.

**Quy trình hậu huấn luyện 4 giai đoạn cho mô hình lớn (flagship)** (Hình 1 của paper): (1) Long-CoT cold start (SFT trên chuỗi suy luận dài đã lọc); (2) Reasoning RL bằng GRPO (3.995 cặp query-verifier; AIME'24 của Qwen3-235B-A22B tăng từ 70,1 lên 85,1 qua 170 bước RL); (3) Thinking Mode Fusion (SFT gộp dữ liệu thinking và non-thinking, nảy sinh khả năng điều khiển "thinking budget"); (4) General RL (hơn 20 loại tác vụ, ba loại reward: rule-based, model-based có đáp án tham chiếu, model-based không có tham chiếu).

**Vấn đề với mô hình nhỏ.** Chạy cả bốn giai đoạn cho từng mô hình nhỏ rất đắt (nhất là RL). Qwen3 dùng **strong-to-weak distillation** cho 5 mô hình dense (0,6B, 1,7B, 4B, 8B, 14B) và MoE 30B-A3B.

## 2. Ý tưởng chính

Mô hình nhỏ không đi qua quy trình bốn giai đoạn. Thay vào đó chúng được distill từ mô hình lớn trong **hai pha**. Pha off-policy: student học từ câu trả lời do teacher sinh sẵn ở cả hai chế độ `/think` và `/no think`, để có kỹ năng suy luận cơ bản và biết chuyển chế độ. Pha on-policy: student tự sinh câu trả lời cho các prompt, rồi được tinh chỉnh để **logit của nó khớp logit của teacher** (Qwen3-32B hoặc Qwen3-235B-A22B) bằng cách tối thiểu KL. Paper báo cáo rằng pha on-policy cho kết quả tốt hơn RL trực tiếp trên cùng checkpoint ban đầu (ở Qwen3-8B, trên bài toán toán và code) với khoảng **1/10 số giờ GPU**, và còn cải thiện pass@64 trong khi RL thì không.

## 3. Phương pháp chi tiết

### 3.1 Hai pha

**Pha 1: Off-policy distillation.** "Off-policy" (gloss): student học từ mẫu **do teacher sinh**, không phải do chính student sinh; đây là sequence-level KD (bài 15, 37, 38) nhưng với dữ liệu gồm cả hai chế độ `/think` và `/no think`. Mục đích theo paper: giúp student có kỹ năng suy luận cơ bản và khả năng chuyển đổi chế độ, "đặt nền" cho pha sau. Paper không nêu loss cụ thể ở pha này (có thể hiểu là SFT/cross-entropy trên văn bản teacher; đó là suy luận của chúng tôi).

**Pha 2: On-policy distillation.** "On-policy" (gloss): student **tự sinh** chuỗi trả lời, rồi nhận phản hồi từ teacher trên chính chuỗi đó, giải quyết lệch phân phối giữa lúc train (chuỗi của teacher) và lúc chạy (chuỗi của student) như GKD (29) và MiniLLM (28). Theo paper: lấy mẫu prompt; student sinh response ở chế độ `/think` hoặc `/no think`; student được fine-tune "bằng cách căn logit của nó với logit của teacher (Qwen3-32B hoặc Qwen3-235B-A22B) để tối thiểu hóa KL". Viết hình thức (suy luận, paper không đưa công thức):

$$
\mathcal L_{\text{on-policy}}(\theta)=\mathbb E_{x\sim\mathcal D}\;\mathbb E_{y\sim q_\theta(\cdot\mid x)}\left[\frac1{|y|}\sum_{t=1}^{|y|}\mathrm{KL}\big(p_T(\cdot\mid x,y_{<t})\,\Vert\,q_\theta(\cdot\mid x,y_{<t})\big)\right]
$$

Hướng KL (forward hay reverse) **không được nêu** trong paper; công thức trên chọn forward để minh họa. Với dữ liệu on-policy, mỗi token của chuỗi do student sinh được teacher "chấm" bằng cả phân phối chứ không chỉ một vô hướng thưởng như RL.

### 3.2 Pseudo-code

```
# Stage A: off-policy (teacher data, both modes)
D_off = { (prompt+flag, teacher_response) : flag in {/think, /no think} }
student <- SFT(student, D_off)          # basic reasoning + mode switching

# Stage B: on-policy (student samples, teacher logits)
for batch of prompts x:
    flag  ~ {/think, /no think}
    y     = student.generate(x, flag)           # on-policy sample
    p_T   = teacher.logits(x, y)                # teacher forward (Qwen3-32B or 235B-A22B)
    q     = student.logits(x, y)
    loss  = KL(p_T || q) averaged over tokens of y
    update student
```

### 3.3 Điều paper không nói

Không có: số mẫu mỗi pha, tỷ lệ trộn think/no-think, thang mẫu, nhiệt độ lấy mẫu, độ dài tối đa của rollout, hướng KL, teacher nào cho student nào (chỉ nói 32B hoặc 235B-A22B), có xử lý tokenizer khác hay không (cùng họ Qwen3 nên cùng tokenizer, từ vựng 151.669 token), và vì sao phải làm off-policy trước. Các câu "preliminary experiments suggest" ở Mục 4 cũng không kèm bảng riêng ngoài Bảng 21.

## 4. Thí nghiệm và kết quả chính

### 4.1 So sánh với RL trực tiếp (Bảng 21)

Cả hai bắt đầu từ cùng checkpoint Qwen3-8B sau off-policy distillation; chỉ dùng truy vấn toán và code. Các số trong ngoặc là pass@64.

| Phương pháp | AIME'24 | AIME'25 | MATH500 | LiveCodeBench v5 | MMLU-Redux | GPQA-Diamond | Giờ GPU |
|---|---|---|---|---|---|---|---|
| Off-policy distillation | 55,0 (90,0) | 42,8 (83,3) | 92,4 | 42,0 | 86,4 | 55,6 | – |
| + Reinforcement Learning | 67,6 (90,0) | 55,5 (83,3) | 94,8 | 52,9 | 86,9 | 61,3 | 17.920 |
| + On-policy Distillation | 74,4 (93,3) | 65,5 (86,7) | 97,0 | 60,3 | 88,3 | 63,3 | 1.800 |

Đọc bảng: on-policy distillation tốt hơn RL ở mọi cột, với 1.800 giờ GPU so với 17.920 (tức khoảng 1/10). Pass@64 tăng nhờ distillation (90,0 → 93,3 và 83,3 → 86,7) nhưng đứng nguyên với RL (90,0 và 83,3). Diễn giải của paper: distill từ logit teacher giúp student **mở rộng không gian khám phá** và tăng tiềm năng suy luận, còn RL chủ yếu làm tăng pass@1 (độ chính xác khi chỉ thử một lần) mà không nới thêm pass@64.

### 4.2 Kết quả mô hình nhỏ

Paper kết luận (Mục 4.6) rằng các mô hình nhẹ (30B-A3B, 14B và các dense nhỏ hơn) "luôn vượt các mô hình mã nguồn mở có số tham số tương đương hoặc lớn hơn" và coi đó là bằng chứng thành công của strong-to-weak distillation. Ví dụ so sánh: Qwen3-4B và 8B (thinking) với DeepSeek-R1-Distill-Qwen-14B/32B; Qwen3-1.7B và 0,6B với DeepSeek-R1-Distill-Qwen-1.5B và -Llama-8B (Bảng 17 và 19). Các bảng này bị lệch cột khi trích văn bản nên chúng tôi **không** sao chép số liệu từng ô; chỉ nên tin kết luận định tính của tác giả và đọc lại bảng gốc nếu cần. Lưu ý: so sánh này là so với các mô hình khác họ, không tách riêng đóng góp của khâu distillation (không có đối chứng "Qwen3-4B train theo 4 giai đoạn đầy đủ" trong paper).

## 5. Phân tích / ablation

**Paper cho thấy (Mục 4, 4.7):**
- Distillation vs RL trên cùng điểm xuất phát: Bảng 21 ở Mục 4.1 trên.
- Tuyên bố tổng quát ở phần mở đầu Mục 4: distill trực tiếp logit teacher vào student "nâng cao hiệu năng và vẫn giữ kiểm soát tinh về quá trình suy luận", cải thiện Pass@1 và Pass@64, và chỉ cần 1/10 số giờ GPU so với quy trình bốn giai đoạn.
- Thinking Mode Fusion cho thấy khả năng điều khiển thinking budget xuất hiện tự nhiên (không được huấn luyện riêng), được minh họa bằng Qwen3-235B-A22B ở Hình 2.

**Paper không ablation (nhận xét của chúng tôi):**
- Không tách đóng góp của pha off-policy và pha on-policy (không có cột "chỉ on-policy từ checkpoint ban đầu").
- So sánh RL vs distillation chỉ cho một cỡ student (8B), một teacher chưa nêu rõ, và một họ tác vụ (toán/code); RL ở đây dùng cấu hình riêng nên chưa biết đã được tối ưu cho cùng mức nỗ lực. Một phần của lợi thế distillation có thể đến từ việc teacher 32B/235B đã được RL đầy đủ (chi phí đó **không** tính vào 1.800 giờ).
- Không có thí nghiệm về cỡ teacher (32B vs 235B) cho student cỡ nhỏ.

## 6. Giới hạn

**Do tác giả nêu:** phần Conclusion chỉ liệt kê hướng tương lai (mở rộng pre-training, kiến trúc để nén hiệu quả, ngữ cảnh cực dài, tăng compute cho RL cho agent). Paper không có mục "Limitations" riêng về distillation.

**Do chúng tôi nhận xét:**
- Chi phí "1/10" **chỉ** là chi phí của student; teacher (Qwen3-32B/235B-A22B) đã được huấn luyện đầy đủ bốn giai đoạn và còn phải chạy forward trong lúc distill on-policy. Nếu chỉ có một student, phép so sánh công bằng cần cộng chi phí này (tương tự lập luận của bài 42).
- Giờ GPU là số liệu của một lần chạy trong một phân tích; không có khoảng tin cậy hay nhiều seed.
- Thiếu chi tiết nên **khó tái hiện**: không biết số mẫu, độ dài, hướng KL, tỷ lệ trộn.
- Teacher và student cùng họ, cùng tokenizer, cùng dữ liệu pre-train; kết luận chưa chắc đúng khi teacher khác họ hoặc khác miền.
- Các nhiệm vụ đánh giá là suy luận văn bản, không có thị giác hay OCR.

## 7. Liên hệ với các bài khác

- **15 / 37 / 38 (Sequence-level KD, Orca, DeepSeek-R1 distillation):** Pha 1 giống kiểu distill bằng văn bản teacher sinh. Khác biệt của Qwen3 là thêm pha 2 trên mẫu của student.
- **28 MiniLLM / 29 GKD / 30 DistiLLM / 47 ShortOPD:** Pha 2 là on-policy distillation (student sinh, teacher chấm). Qwen3 không viết thuật toán, nhưng về ý tưởng gần với GKD.
- **39 Small models struggle with strong reasoners:** bài này chỉ ra student nhỏ khó học từ teacher suy luận mạnh khi chỉ dùng SFT; pha off-policy rồi on-policy của Qwen3 là một cách để vượt qua, và RL-so-sánh ở Bảng 21 củng cố hướng đó.
- **40 Gemma 2:** Gemma 2 cũng dùng on-policy distillation trong SFT và KD ở pre-training; Qwen3 chỉ dùng ở hậu huấn luyện (theo paper).
- **41 / 42 (Capacity gap, Distillation scaling laws):** Qwen3 chọn teacher 32B/235B cho student 0,6B–14B mà không bàn capacity gap; bài 42 cảnh báo teacher quá lớn có thể không tối ưu.
- **45 Sparse logit sampling:** on-policy distillation cần logit teacher trên chuỗi mới sinh của student, nên không thể cache offline; liên quan đến chi phí.

## 8. Ứng dụng cho dự án (suy luận)

Mức liên quan: **cao về quy trình**, trung bình về nội dung (suy luận văn bản dài ≠ OCR).

1. **Công thức hai pha khớp với recovery sau pruning.** Pha 1: teacher 4B sinh transcript cho tập ảnh tài liệu, student đã prune học bằng cross-entropy (SFT) để phục hồi "kỹ năng cơ bản". Pha 2: student tự sinh transcript, teacher 4B cho phân phối token trên chính chuỗi đó, tối thiểu KL. Tránh lệch phân phối giữa train (chuỗi teacher) và suy luận (chuỗi student); với OCR, lỗi lan truyền khi giải mã tự hồi quy (một lỗi sớm làm hỏng phần sau) là chính lý do on-policy có thể hữu ích.
2. **Chi phí so với RL.** Dự án không có reward kiểu verifier trơn tru, nhưng có thể có reward CER/WER. Bảng 21 gợi ý nên thử on-policy KD **trước** khi nghĩ đến RL: rẻ hơn (≈1/10 giờ GPU trong phép đo của họ) và có teacher sẵn. Đây là bằng chứng ở LLM suy luận toán/code; chưa chắc chuyển sang OCR.
3. **Teacher chạy lúc train.** On-policy cần forward teacher 4B trên chuỗi do student sinh, không cache offline được. Với GPU ≤ 8 GB: teacher 4B ở bf16 xấp xỉ 8 GB trọng số, nên cần lượng tử hóa teacher hoặc chạy teacher trên máy lớn hơn lúc train (đó là ước tính thô của chúng tôi). Yêu cầu ≤ 8 GB chỉ áp dụng cho suy luận của student, không phải lúc train.
4. **Chế độ think/no-think ít liên quan:** OCR không có chuỗi suy luận dài; chỉ nên mượn khung hai pha, không mượn thinking budget.
5. **Điều chưa biết cần đo:** tỷ lệ 13× và 40× giữa teacher 4B và student 0,3B/0,1B lớn hơn nhiều so với Qwen3 (ví dụ 32B → 8B là 4×; 235B-A22B có 22B hoạt động); bài 41, 42 cảnh báo về capacity gap. Cũng không có kết quả cho việc cắt MLP neuron vs layer, không liên quan trực tiếp tới $D_{lin}/D_{res}$.

## 9. Thí nghiệm gợi ý

1. **Off-policy vs on-policy, đủ ba cột:** student 0,3B đã prune: (a) chỉ SFT trên transcript teacher, (b) SFT rồi on-policy KD, (c) chỉ on-policy KD từ checkpoint prune. Cùng ngân sách giờ GPU, đo CER/WER và tỷ lệ lỗi "tràn" (hallucination/lặp). Mục tiêu: lấp khoảng trống ablation của Qwen3.
2. **On-policy KD vs lấy mẫu tối ưu CER (rejection sampling / RL nhẹ):** so on-policy KD với việc huấn luyện trên các transcript tự sinh có CER thấp nhất, để kiểm tra lập luận "KD mở rộng exploration" ở OCR (đo pass@k theo CER).
3. **Kích thước teacher cho on-policy:** thử teacher 4B so với một teacher trung gian (hoặc bản 0,8B) cho student 0,1B, để xem capacity gap (bài 41–42) có lặp lại trong pha on-policy không.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Strong-to-weak distillation | Dùng mô hình lớn đã hoàn thiện để huấn luyện mô hình nhỏ thay vì chạy lại 4 giai đoạn |
| Off-policy distillation | Student học từ câu trả lời do teacher sinh (cả `/think`, `/no think`) |
| On-policy distillation | Student tự sinh, teacher cho logit, tối thiểu KL |
| Teacher (on-policy) | Qwen3-32B hoặc Qwen3-235B-A22B |
| Bảng 21 | On-policy KD vs RL từ cùng checkpoint 8B: AIME'24 74,4 vs 67,6; 1.800 vs 17.920 giờ GPU |
| pass@64 | KD tăng (90,0→93,3; 83,3→86,7), RL không đổi |
| Thinking mode fusion | SFT hợp nhất hai chế độ, cho phép thinking budget |
| "1/10 GPU hours" | Tỷ lệ so với quy trình bốn giai đoạn (theo paper), chưa tính chi phí teacher |

## 11. Kết luận ngắn

Qwen3 cho thấy một quy trình thực dụng để làm mô hình nhỏ: off-policy để đặt nền, on-policy bằng logit teacher để đẩy hiệu năng, rẻ hơn RL khoảng 10 lần theo phép đo 8B của họ và còn nâng pass@64. Phần distillation của paper rất ngắn và thiếu ablation, chi phí của teacher không được tính, nên nên xem đây là gợi ý quy trình chứ không phải kết quả đã được kiểm chứng rộng. Với dự án, nó củng cố ý tưởng thử on-policy KD cho student đã prune, với cảnh báo về tỷ lệ teacher/student rất lớn.

**Link gốc:** https://arxiv.org/abs/2505.09388
