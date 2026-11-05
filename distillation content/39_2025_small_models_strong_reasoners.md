# Small Models Struggle to Learn from Strong Reasoners (2025)

**Tác giả:** Yuetai Li, Xiang Yue, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Bill Yuchen Lin, Bhaskar Ramasubramanian, Radha Poovendran.  
**Venue / arXiv:** arXiv:2502.12143 (bản v3, 13/11/2025; paper text không nêu tên hội nghị).  
**Nguồn:** https://arxiv.org/abs/2502.12143  
**Chủ đề:** Distillation lập luận (chain-of-thought) vào model nhỏ: phát hiện "Small Model Learnability Gap" và đề xuất **Mix Distillation**.  
**Vị trí trong lộ trình:** Đọc sau 37 Orca và 38 DeepSeek-R1 distillation (cả hai tin rằng trace dài/teacher mạnh giúp student). Bài này **phản biện có điều kiện**: với student <= 3B, trace dài và teacher lớn không luôn tốt. Nối với 08 Teacher assistant, 09 Efficacy of KD (capacity gap ở phân loại), và dẫn sang 41 Capacity gap law, 42 Distillation scaling laws (khung định lượng).

> Ghi chú học tập bằng tiếng Việt, diễn giải lại, không phải bản dịch. Phần "Ứng dụng cho dự án" là suy luận của chúng ta, không phải kết luận của paper. Phạm vi paper chỉ là toán học (MATH và các benchmark toán).

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

- Sau Orca (37) và DeepSeek-R1 (38), thói quen phổ biến là: lấy teacher mạnh, sinh lời giải (càng dài, càng "suy nghĩ" càng tốt), SFT student nhỏ. Kỳ vọng ngầm: teacher mạnh hơn thì student tốt hơn.
- Gloss: **Short CoT** = lời giải ngắn gọn đi thẳng tới đáp án. **Long CoT** = chuỗi dài có phản tư, quay lui, thử hướng khác (kiểu QwQ, R1). **Learnability** = mức mà một student nhất định có thể học được từ một kiểu dữ liệu. **Capacity gap** (ở phân loại ảnh, 08/09) = chênh lệch quá lớn giữa teacher và student làm distillation kém đi; bài này khảo sát hiện tượng tương tự ở bài toán sinh lập luận.
- Trước đó đã có công trình (Kim et al. 2024b, được paper trích) cho thấy teacher mạnh không nhất thiết là teacher tốt; bài này quy trách nhiệm chính cho **kích thước student**.

## 2. Ý tưởng chính

Thí nghiệm có kiểm soát trên 10 student (Qwen2.5 0.5B-32B, Llama 1B-70B) và nhiều cặp teacher: student **nhỏ (<= 3B)** học tốt hơn từ short CoT và từ teacher nhỏ; student **lớn (>= 7B)** hưởng lợi từ long CoT và teacher lớn. Paper gọi đây là **Small Model Learnability Gap**. Gợi ý nguyên nhân: sai khác phân phối giữa student và dữ liệu teacher (student gán perplexity cao cho trace dài/teacher lớn), và tri thức miền hạn chế của model nhỏ. Cách khắc phục đơn giản: **Mix Distillation**, trộn dữ liệu dễ (short/teacher nhỏ) với dữ liệu khó (long/teacher lớn) theo tỷ lệ nhỏ, ví dụ 1:4 (30% là sai, xem 3.4) cho kết quả tốt hơn dùng riêng từng loại.

## 3. Phương pháp chi tiết

### 3.1 Ký hiệu và SFT

Student $\pi_\theta$ học trên $\mathcal{D}=\{(x_i,y^i)\}_{i=1}^N$ với $y^i$ là short CoT, long CoT, CoT của teacher lớn hoặc nhỏ, bằng NLL chuẩn:

$$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i}\sum_{t}\log \pi_\theta\big(y^i_t \mid x_i, y^i_{<t}\big)$$

### 3.2 Hai khoảng cách đo hiện tượng

Với $P_{\text{Long}}, P_{\text{Short}}$ là điểm trung bình (5 benchmark) của student SFT trên long/short CoT:

$$\boxed{\Delta_{\text{Long}} = P_{\text{Long}} - P_{\text{Short}}}\qquad\boxed{\Delta_{\text{Large}} = P_{\text{Large}} - P_{\text{Small}}}$$

$\Delta<0$: dữ liệu "mạnh" tệ hơn dữ liệu "yếu" cho student đó. "Learnability Gap" xảy ra khi $\Delta<0$ ở student nhỏ và $\Delta>0$ ở student lớn.

### 3.3 Cài đặt

- **Dữ liệu:** 7,500 prompt của MATH.
- **Teacher:** long CoT từ QwQ-32B-Preview; short CoT từ Qwen2.5-32B-Instruct. Cặp lớn/nhỏ trong cùng họ: Qwen2.5-72B vs 3B-Instruct, Llama3.1-70B vs 8B-Instruct, Gemma2-27B vs 9B-it. Thêm DeepSeek-R1-32B (long CoT) làm baseline ở phần Mix.
- **Sinh dữ liệu:** rejection sampling (giữ lời giải đúng), mặc định greedy. Để so teacher lớn/nhỏ công bằng, chỉ giữ các cặp (bài, lời giải) mà **cả hai** teacher đều đúng; độ dài trung bình gần như bằng nhau (432.98 token với teacher 72B, 440.70 với 3B), loại độ dài làm biến gây nhiễu.
- **Huấn luyện:** LLaMA-Factory. Student < 14B: SFT toàn bộ tham số, lr tối đa 1e-5 cosine, 2 epoch. Student > 14B: LoRA lr 1e-4, 2 epoch. Bốn A100 80GB.
- **Đánh giá:** MATH, GSM8K, AMC 2023, AIME 2024, OlympiadBench (phần toán tiếng Anh); zero-shot, greedy, tối đa 16k token; điểm student là trung bình 5 benchmark.

### 3.4 Mix Distillation

- **Mix-Long:** trộn long CoT (trọng số $\alpha$) và short CoT ($1-\alpha$).
- **Mix-Large:** trộn CoT của teacher lớn ($\alpha$) và teacher nhỏ ($1-\alpha$).
- Chọn $\alpha=0.2$ (Figure 6 cho thấy tại 0.2 điểm trung bình cao nhất với Qwen2.5-3B-Instruct). Trong chú thích Table 3 viết "tỷ lệ 1:4", tức 1 phần long : 4 phần short, khớp với $\alpha=0.2$.

$$\mathcal{D}_{\text{mix}}=\alpha\,\mathcal{D}_{\text{hard}}\ \cup\ (1-\alpha)\,\mathcal{D}_{\text{easy}},\qquad \alpha=0.2$$

Pseudo-code:

```
D_hard = teacher_mạnh_sinh(MATH7.5k)    # long CoT hoặc teacher lớn
D_easy = teacher_yếu_sinh(MATH7.5k)     # short CoT hoặc teacher nhỏ
D_mix  = sample(D_hard, 0.2*N) + sample(D_easy, 0.8*N)
student = SFT(student, D_mix)
```

## 4. Thí nghiệm và kết quả chính

### 4.1 Long CoT Gap (Table 1, điểm trung bình 5 benchmark)

| Student | P_Long | P_Short | Δ_Long | Tốt hơn |
|---|---|---|---|---|
| Qwen2.5-0.5B | 14.8 | 19.5 | -4.7 | Short |
| Qwen2.5-1.5B | 27.0 | 34.2 | -7.1 | Short |
| Qwen2.5-3B | 40.3 | 43.4 | -3.1 | Short |
| Qwen2.5-7B | 48.9 | 47.2 | +1.7 | Long |
| Qwen2.5-14B | 59.2 | 54.3 | +4.9 | Long |
| Qwen2.5-32B | 73.0 | 59.3 | +13.7 | Long |
| Llama-3.2-1B | 15.8 | 19.5 | -3.7 | Short |
| Llama-3.2-3B | 32.5 | 33.1 | -0.6 | Short |
| Llama-3.1-8B | 35.2 | 31.5 | +3.7 | Long |
| Llama-3.3-70B | 58.2 | 54.3 | +3.8 | Long |

Điểm chuyển nằm giữa 3B và 7B (Qwen) và giữa 3B và 8B (Llama). Một ví dụ cụ thể ở văn bản: Qwen2.5-1.5B-Instruct thấp hơn hơn 10 điểm trên MATH và AMC khi học long CoT.

### 4.2 Large Teacher CoT Gap (Table 2; short CoT, teacher lớn Qwen2.5-72B vs nhỏ Qwen2.5-3B)

| Student | P_Large | P_Small | Δ_Large | Tốt hơn |
|---|---|---|---|---|
| Qwen2.5-0.5B | 16.9 | 20.4 | -3.5 | Weak |
| Qwen2.5-1.5B | 32.2 | 33.0 | -0.8 | Weak |
| Qwen2.5-3B | 39.7 | 39.4 | +0.3 | Strong |
| Qwen2.5-7B | 48.9 | 42.3 | +6.6 | Strong |
| Qwen2.5-14B | 52.9 | 49.9 | +3.0 | Strong |
| Qwen2.5-32B | 59.5 | 53.0 | +6.5 | Strong |
| Llama-3.2-1B | 16.5 | 18.5 | -1.9 | Weak |
| Llama-3.2-3B | 32.8 | 31.2 | +1.6 | Strong |
| Llama-3.2-8B (ghi trong bảng, thực chất Llama-3.1-8B) | 25.6 | 25.1 | +0.5 | Strong |
| Llama-3.2-70B (ghi trong bảng) | 57.6 | 53.3 | +4.3 | Strong |

Lưu ý: nhãn "Llama-3.2-8B/70B" là cách ghi trong bảng của paper; mục 3.1 liệt kê Llama3.1-8B và Llama3.3-70B. Hiệu ứng ở mức 3B khá mờ (Δ gần 0), và cột "Strong/Weak" do tác giả gán theo dấu Δ.

### 4.3 Mix Distillation (Table 3, student Qwen2.5-3B-Instruct, MATH 7.5k)

| Phương pháp dữ liệu | MATH | AMC | GSM8K | Olympiad | AIME | Trung bình |
|---|---|---|---|---|---|---|
| Long CoT | 56.2 | 37.5 | 80.0 | 24.4 | 3.3 | 40.3 |
| Short CoT | 61.0 | 37.5 | 82.0 | 26.4 | 10.0 | 43.4 |
| Strong model CoT | 57.5 | 35.0 | 80.0 | 25.9 | 0.0 | 39.7 |
| Weak model CoT | 60.3 | 27.5 | 79.5 | 26.4 | 3.3 | 39.4 |
| DeepSeek-R1-32B (long CoT) | 50.7 | 20.0 | 81.2 | 15.7 | 0.0 | 33.5 |
| **Mix-Long** | 64.7 | 45.0 | 81.4 | 28.6 | 10.0 | **45.9** |
| **Mix-Large** | 65.8 | 42.5 | 81.7 | 29.0 | 10.0 | **45.8** |

Với Llama3.2-3B-Instruct: Long CoT 32.5, Short CoT 33.1, Strong 32.8, Weak 31.2, R1-32B 33.3, Mix-Long 35.1, Mix-Large 34.7 (trung bình).

Con số trong văn bản: Qwen2.5-3B-Instruct tăng hơn 8 điểm trên MATH và AMC với Mix-Long so với dùng long CoT thuần; tăng hơn 7 điểm trên MATH, AIME và AMC với Mix-Large so với dùng CoT teacher lớn thuần.

## 5. Phân tích / ablation

- **Tri thức miền (Takeaway 3):** Qwen2.5-Math-1.5B-Instruct (model "chuyên toán" dù nhỏ) có gap nhỏ hơn Qwen2.5-1.5B tổng quát, và hưởng lợi từ teacher lớn nhiều hơn. Tác giả phỏng đoán nguyên nhân quan trọng là tri thức miền hạn chế của student nhỏ.
- **Sai khác phân phối (Takeaway 4, Appendix B.3):** đo perplexity của dữ liệu teacher dưới student: student nhỏ gán perplexity cao hơn đáng kể cho long CoT và CoT teacher lớn; cặp student-teacher "cùng cỡ" khớp nhau hơn. (Bảng PPL chi tiết nằm ở Table 10 của paper, ở đây không chép số.)
- **Base vs Instruct (Takeaway 5):** model base nhỏ có gap rõ hơn model Instruct.
- **Thay đổi văn phong (Takeaway 6):** dùng phép đo rank shift của token (Lin et al., 2023): token đổi nhiều nhất sau khi học long CoT/teacher lớn là từ mang tính văn phong như "wait", "But", "Let", chứ không phải nội dung toán.
- **Trọng số trộn (Figure 6):** $\alpha=0.2$ cho điểm trung bình cao nhất (Qwen2.5-3B-Instruct).
- **Định tính (Figure 7):** model học long CoT thuần "nghĩ vòng vo" lặp lại, không biết dừng; học short thuần sai đáp án; Mix-Long có yếu tố rẽ nhánh ("Alternatively") nhưng ngắn gọn. Độ dài phản hồi trung bình: long CoT 3384.7 token, short 575.7, Mix-Long 1248.9.

## 6. Giới hạn

**Do tác giả nêu:**
- Chỉ xét toán; chưa đánh giá instruction following, code, đa phương thức.
- Chưa khảo sát ảnh hưởng của thành phần dữ liệu tiền huấn luyện.

**Do chúng ta nhận xét:**
- Chỉ SFT trên 7.5K bài MATH; kết luận có thể thay đổi với dữ liệu lớn hơn nhiều (R1 dùng 800K) hoặc với RL sau SFT.
- Nhiều hiệu ứng nhỏ (ví dụ Δ_Large của 3B chỉ +0.3, Llama-3B -0.6), không có thống kê độ tin cậy/seed; AIME chỉ 30 bài nên mỗi bài sai là ±3.3 điểm.
- Trong Table 3, DeepSeek-R1-32B (nhãn trong paper) là bản distill 32B, không phải R1 gốc; long CoT của nó thấp hơn cả QwQ ở student 3B, củng cố luận điểm nhưng không cô lập được "độ dài" với "phong cách teacher".
- Nguyên nhân (phân phối, tri thức miền) mới là gợi ý tương quan; chưa có thí nghiệm can thiệp (ví dụ SFT bước đầu để thu hẹp perplexity trước) để chứng minh nhân quả.
- Mix Distillation là heuristic với $\alpha$ chọn trên chính tập đánh giá trung bình; chưa rõ tổng quát.

## 7. Liên hệ với các bài khác

- **08 Teacher assistant, 09 Efficacy of KD:** cùng lập luận "teacher quá khác student thì khó học"; ở đây chuyển sang bài toán sinh chuỗi.
- **41 Capacity gap law, 42 Distillation scaling laws:** định lượng hoá vấn đề này; Mix Distillation là cách xử lý thực dụng.
- **37 Orca:** Orca dùng ChatGPT làm "teaching assistant" trước GPT-4 vì cùng lý do; paper này là bằng chứng độc lập cho trực giác đó.
- **38 DeepSeek-R1 distillation:** student R1-Distill-1.5B vẫn lên mạnh trên toán (AIME 28.9, MATH 83.9), nhưng dùng 800K mẫu và base Qwen2.5-Math chuyên toán; phù hợp với Takeaway 3 (tri thức miền giúp). Cần cẩn trọng khi so, vì thiết lập khác nhiều.
- **36 Distilling step-by-step:** rationale ngắn, task cố định; hợp với phát hiện "short CoT tốt cho student nhỏ".
- **28-34 (MiniLLM, GKD, DistiLLM, TAID...):** nhóm logit-based/on-policy xử lý khoảng cách phân phối bằng cách học trên mẫu của chính student, một lối giải quyết khác cho cùng vấn đề.
- **44 Qwen3 distillation, 40 Gemma 2:** thực hành trong quy trình sản xuất.

## 8. Ứng dụng cho dự án (suy luận)

Mức liên quan: **trung bình, ở mức nguyên tắc.** Không có long CoT trong OCR, nhưng bài học "đừng cho student nhỏ học dữ liệu quá xa phân phối của nó" có thể chuyển.
- Student sau pruning (~0.3B, ~0.1B) là model nhỏ so với teacher 4B; hiện tượng "teacher quá khác" có thể xuất hiện khi dùng đích văn bản/logit của teacher trực tiếp. Gợi ý: dùng teacher trung gian (student trước khi cắt, hoặc teacher 4B đã được "làm mịn" bằng nhiệt độ) hoặc trộn đích dễ + đích khó.
- Mix Distillation tương ứng: trộn dữ liệu có nhãn thật hoặc đích dễ (trang đơn giản, chữ in) với đích khó từ teacher (bố cục phức tạp, bảng, công thức). Đây là suy luận, chưa kiểm chứng.
- Đo **perplexity của đích teacher dưới student** như chẩn đoán trước khi huấn luyện: nếu cao bất thường, đây là dấu hiệu khoảng cách phân phối. Công cụ rẻ, dễ làm.
- Điều này có thể liên hệ với quan sát "cắt neuron MLP phục hồi tệ hơn cắt layer": D_res lớn có thể tương ứng student rời xa phân phối cũ khiến đích teacher khó học; giả thuyết cần thực nghiệm.
- Không áp dụng trực tiếp: kết luận về độ dài CoT.

## 9. Thí nghiệm gợi ý

1. Đo perplexity (hoặc NLL token) của đầu ra teacher 4B trên tập huấn luyện dưới ba student: chưa cắt, cắt layer, cắt neuron MLP. Xem tương quan với mức phục hồi sau KD.
2. Mix-style: đích = $\alpha$ đích teacher khó + $(1-\alpha)$ nhãn thật/đích dễ, quét $\alpha\in\{0, 0.2, 0.5, 1\}$ trên student 0.1B.
3. Teacher trung gian: so KD thẳng từ 4B với KD qua một student trung gian (ví dụ 0.8B gốc) cho student 0.3B.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Small Model Learnability Gap | Student <= 3B không hưởng lợi từ long CoT / teacher lớn |
| $\Delta_{\text{Long}}=P_{\text{Long}}-P_{\text{Short}}$ | Chênh điểm khi học long so với short CoT |
| $\Delta_{\text{Large}}=P_{\text{Large}}-P_{\text{Small}}$ | Chênh điểm khi học teacher lớn so với nhỏ |
| Mix-Long / Mix-Large | Trộn long+short / lớn+nhỏ với $\alpha=0.2$ |
| Điểm chuyển | Giữa ~3B và ~7B (Qwen), 3B và 8B (Llama) |
| Kết quả nổi bật | Qwen2.5-3B: Mix-Long 45.9, Mix-Large 45.8 so với Long CoT 40.3 |
| Nguyên nhân gợi ý | Sai khác phân phối, tri thức miền, văn phong ("wait", "But") |

## 11. Kết luận ngắn

Teacher mạnh hơn và lời giải dài hơn không phải lúc nào cũng tốt hơn cho student nhỏ. Với student <= 3B, short CoT và teacher nhỏ thường học tốt hơn, và trộn dữ liệu dễ với dữ liệu khó (tỷ lệ 20% khó) đạt kết quả cao nhất trong các cấu hình thử. Kết luận chỉ kiểm chứng trong toán với SFT trên 7.5K bài, nên là gợi ý thiết kế chứ chưa phải định luật.

Link gốc: https://arxiv.org/abs/2502.12143
