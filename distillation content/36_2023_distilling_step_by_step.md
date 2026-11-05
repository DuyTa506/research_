# Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes (2023)

**Tác giả:** Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alexander Ratner, Ranjay Krishna, Chen-Yu Lee, Tomas Pfister.  
**Venue / arXiv:** arXiv:2305.02301 (bản v2, tháng 7/2023; trang đầu của bản text không ghi tên hội nghị, trích dẫn công khai là Findings of ACL 2023 nhưng paper text ta có không nêu nên không khẳng định).  
**Nguồn:** https://arxiv.org/abs/2305.02301  
**Chủ đề:** Distillation task-specific từ LLM: lấy cả nhãn lẫn **rationale** (lời giải thích từng bước) của teacher, huấn luyện student theo multi-task.  
**Vị trí trong lộ trình:** Đọc sau 35 Self-Instruct (teacher sinh dữ liệu) và 15 Sequence-level KD. Bài này bổ sung ý "teacher không chỉ cho nhãn mà cho cả lý do", và dùng rationale như **tín hiệu phụ chỉ có lúc train**. Tiếp theo: 37 Orca (explanation trace quy mô lớn), 38 và 39 (reasoning dài, nơi rationale thành phần chính).

> Ghi chú học tập bằng tiếng Việt, diễn giải lại, không phải bản dịch. Phần "Ứng dụng cho dự án" là suy luận của chúng ta.

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

- LLM (PaLM 540B...) rất mạnh few-shot nhưng khó triển khai: nặng, tốn bộ nhớ, độ trễ cao.
- Cách thông thường để có model nhỏ chuyên task: (a) **finetuning** trên nhãn người (đắt), hoặc (b) **task distillation**: dùng LLM gán **nhãn giả** cho dữ liệu chưa nhãn rồi train model nhỏ (cần rất nhiều dữ liệu chưa nhãn).
- Cả hai đều cần nhiều dữ liệu để bắt kịp LLM.
- Gloss: **rationale** = đoạn giải thích trung gian dẫn từ input đến nhãn (ví dụ "Diện tích = dài x rộng; phòng có 11x15..."). **Chain-of-thought (CoT) prompting** = đưa vài ví dụ có lời giải từng bước vào prompt để LLM tự sinh lời giải. **Task-specific model** = model nhỏ chỉ làm một task (khác với model đa dụng).

## 2. Ý tưởng chính

Thay vì coi LLM là nguồn nhãn nhiễu, coi nó là tác nhân biết lập luận: prompt CoT để LLM vừa trả nhãn vừa viết rationale. Sau đó train model nhỏ (T5) với **hai nhiệm vụ**: dự đoán nhãn và sinh rationale, phân biệt bằng tiền tố "[label]" / "[rationale]" ở input. Rationale chứa kiến thức task (ví dụ "diện tích = dài x rộng") mà model nhỏ khó tự suy ra từ ít dữ liệu. Khi inference chỉ cần dự đoán nhãn, không phải sinh rationale, nên không tốn thêm chi phí triển khai.

## 3. Phương pháp chi tiết

### 3.1 Bước 1: trích rationale từ LLM

Prompt few-shot gồm các bộ ba $(x^p, r^p, y^p)$ (ví dụ, rationale người viết, nhãn). Với mỗi $x_i$ chưa nhãn, ghép vào prompt, LLM bắt chước và sinh $(\hat r_i, \hat y_i)$.

### 3.2 Bước 2: huấn luyện multi-task

Task distillation chuẩn (nhãn giả $\hat y_i$, hoặc nhãn người $y_i$ nếu finetuning):

$$L_{\text{label}} = \frac1N\sum_{i=1}^N \ell\big(f(x_i), \hat y_i\big)$$

Có hai cách đưa rationale vào:

- Cách "rationale làm input" (Eq. 2): $L=\frac1N\sum \ell(f(x_i,\hat r_i),\hat y_i)$. Nhược điểm: lúc test vẫn cần LLM sinh rationale trước, mất ý nghĩa nén.
- **Cách của paper (Eq. 3-4):** rationale làm **đích thứ hai**:

$$\boxed{L = L_{\text{label}} + \lambda L_{\text{rationale}},\qquad L_{\text{rationale}} = \frac1N\sum_{i}\ell\big(f(x_i),\hat r_i\big)}$$

Cùng một model $f$, hai đích khác nhau tuỳ tiền tố task. Paper không nêu giá trị $\lambda$ cụ thể trong phần text ta đọc được (setup trong appendix chỉ nêu learning rate, batch size, số step).

Biến thể đối chứng "single-task" (Eq. 5): gộp thành một đích $[\hat r_i, \hat y_i]$:

$$L_{\text{single}}=\frac1N\sum_i \ell\big(f(x_i),[\hat r_i,\hat y_i]\big)$$

### 3.3 Pseudo-code

```
# Giai đoạn A: sinh rationale (một lần, offline)
for x in D_unlabeled_or_labeled:
    (r_hat, y_hat) = LLM(CoT_prompt + x)          # PaLM 540B, ~10-shot
# Giai đoạn B: train student T5
for (x, y_hat, r_hat) in batch:
    loss_label = CE(f("[label] " + x),     y_hat)
    loss_rat   = CE(f("[rationale] " + x), r_hat)
    loss = loss_label + lam * loss_rat
# Inference: chỉ f("[label] " + x), không sinh rationale
```

### 3.4 Ví dụ nhỏ

SVAMP: "Phòng dài 11 ft, rộng 15 ft, đã có 16 sq ft thảm. Cần thêm bao nhiêu?" Đích nhãn: "(11x15)-16". Đích rationale: "Area = length x width. Jesse's room has 11x15 square feet." Model học "công thức diện tích" từ nhiệm vụ rationale, rồi dùng khi dự đoán nhãn.

## 4. Thí nghiệm và kết quả chính

**Setup:** LLM = PaLM 540B; student = T5 (220M T5-Base, 770M T5-Large, 11B T5-XXL). 4 dataset: e-SNLI, ANLI (NLI), CQA (commonsense QA), SVAMP (toán lời văn). Số mẫu train: e-SNLI 549,367; ANLI 16,946; CQA 8,766; SVAMP 720. Huấn luyện: lr 5e-5, batch 64 (T5-Base/Large, tối đa 10000 step), batch 32 (T5-XXL, tối đa 4000 step); max input 1024; 4 lần chạy ngẫu nhiên; A100x16.

**Kết quả định tính chính (so với finetuning/distillation chuẩn, T5 220M):**
- Với nhãn người: chỉ **12.5%** e-SNLI đã vượt finetuning chuẩn dùng 100%. Giảm lượng dữ liệu cần để vượt finetuning chuẩn: 75% (ANLI), 25% (CQA), 20% (SVAMP).
- Chỉ dữ liệu chưa nhãn: 12.5% tập chưa nhãn e-SNLI đã vượt task distillation chuẩn dùng 100%.
- Tóm tắt abstract/intro: dùng ít hơn 50% dữ liệu trung bình (có chỗ trên 85%).

**Giảm kích thước model:**
- Với nhãn người: vượt Few-shot CoT của PaLM 540B bằng T5 220M trên e-SNLI (nhỏ hơn hơn 2000 lần), T5 770M trên ANLI và SVAMP (hơn 700 lần), T5 11B trên CQA (hơn 45 lần).
- Chỉ dữ liệu chưa nhãn: vượt LLM ở 3/4 dataset (e-SNLI với 220M; 45 lần nhỏ hơn ở ANLI và CQA); SVAMP chỉ bắt kịp sau khi thêm dữ liệu chưa nhãn ASDiv (2,305 mẫu), bằng T5 11B (dưới 3% kích thước PaLM).
- Trên ANLI, cải thiện trung bình so với finetuning chuẩn là 8% và so với distillation chuẩn là 13% (độ chính xác).

**Tối thiểu cả dữ liệu lẫn kích thước (Figure 8-9):** trên ANLI, T5 770M chỉ cần 80% dữ liệu để vượt PaLM Few-shot CoT, trong khi finetuning chuẩn không bắt kịp ngay cả với 100%. Trên e-SNLI, model 220M và 0.1% dữ liệu đã vượt Few-shot CoT.

## 5. Phân tích / ablation

**Phụ thuộc chất lượng LLM (Table 1, T5 220M, 100% dữ liệu):**

| Phương pháp | LLM | e-SNLI | ANLI | CQA | SVAMP |
|---|---|---|---|---|---|
| Standard finetuning | N/A | 88.38 | 43.58 | 62.19 | 62.63 |
| Distilling step-by-step | 20B (GPT-NeoX) | 89.12 | 48.15 | 63.25 | 63.00 |
| Distilling step-by-step | 540B (PaLM) | 89.51 | 49.58 | 63.29 | 65.50 |

LLM nhỏ hơn vẫn cho lợi ích nhưng ít hơn; paper cho rằng do rationale chất lượng thấp hơn (đây là giải thích của tác giả).

**Multi-task vs single-task (Table 2):**

| Phương pháp | e-SNLI | ANLI | CQA | SVAMP |
|---|---|---|---|---|
| Standard finetuning | 88.38 | 43.58 | 62.19 | 62.63 |
| Single-task (rationale + nhãn gộp) | 88.88 | 43.50 | 61.37 | 63.00 |
| Multi-task | 89.51 | 49.58 | 63.29 | 65.50 |

Điểm đáng chú ý: gộp rationale vào chuỗi đích có thể **tệ hơn finetuning chuẩn** (ANLI, CQA). Phải tách thành task riêng.

## 6. Giới hạn

**Do tác giả nêu:**
- Cần người viết khoảng 10 ví dụ CoT cho mỗi task (có thể giảm bằng zero-shot CoT).
- Tốn thêm chút tính toán lúc train (không tốn lúc test).
- LLM có thể lập luận kém ở task phức tạp/planning; chất lượng rationale ảnh hưởng chưa được khảo sát kỹ.
- Student thừa hưởng thiên kiến của teacher.

**Do chúng ta nhận xét:**
- Chỉ 4 benchmark NLP, đầu ra ngắn (nhãn / biểu thức), và student là T5 encoder-decoder; chưa rõ với decoder-only hay đầu ra dài.
- Không có lọc rationale sai (rationale có thể đúng nhãn nhưng sai logic, hoặc nhãn sai); paper không đo tỷ lệ này.
- Giá trị $\lambda$ và độ nhạy với nó không được trình bày trong phần ta đọc.
- Không đo chi phí gọi PaLM 540B để sinh rationale cho toàn bộ dữ liệu so với chi phí thu nhãn người.

## 7. Liên hệ với các bài khác

- **01 Ba & Caruana / 02 Hinton KD:** "Model compression" gốc dùng logit làm tín hiệu; ở đây tín hiệu phụ là **văn bản giải thích**, một dạng "dark knowledge" ở dạng ngôn ngữ.
- **15 Sequence-level KD:** student học chuỗi do teacher sinh; Distilling step-by-step cũng vậy nhưng dùng cả hai chuỗi (nhãn, rationale) như hai nhiệm vụ.
- **03 FitNets / 04 Attention transfer:** cũng là "tín hiệu phụ chỉ dùng lúc train" (hint layer / attention map); tương tự về tinh thần, khác ở chỗ ở đây là ngôn ngữ tự nhiên, không cần cùng kiến trúc.
- **35 Self-Instruct:** cùng nhánh black-box; 35 sinh tập instruction, 36 sinh rationale cho task cố định.
- **37 Orca:** tăng quy mô ý tưởng "giải thích từng bước" lên dữ liệu đa task, dùng system prompt để gợi explanation.
- **39 Small models struggle with strong reasoners:** cảnh báo rationale quá dài/phức tạp có thể làm student nhỏ học kém; đối chiếu với ở đây rationale ngắn, rất hữu ích.

## 8. Ứng dụng cho dự án (suy luận)

Mức liên quan: **trung bình, ở dạng ý tưởng "tín hiệu phụ lúc train".**
- OCR gần như không có "lập luận" theo nghĩa CoT; đầu ra là phiên mã. Rationale thuần văn bản ít khả thi.
- Nhưng cấu trúc "một model, hai đầu ra, đầu phụ chỉ dùng lúc train, bỏ lúc suy luận" có thể chuyển thể: đầu phụ có thể dự đoán **layout/bounding box**, loại khối (bảng, công thức), hoặc tái tạo hidden state của teacher 4B. Hướng này là suy luận của chúng ta, paper không thử.
- Về pruning: chi phí inference không đổi vì đầu phụ bỏ khi triển khai, hợp với batch-1 latency và GPU 8 GB.
- Cảnh báo từ Table 2: nếu trộn tín hiệu phụ vào cùng chuỗi đích, có thể hại; nên tách bằng tiền tố task hoặc head riêng.

## 9. Thí nghiệm gợi ý

1. Student sau pruning: thêm đầu phụ dự đoán loại khối/layout từ teacher, loss $L_{\text{main}}+\lambda L_{\text{aux}}$; so với KD chỉ chuỗi đích. Quét $\lambda\in\{0.1,0.5,1\}$.
2. Single-task vs multi-task (tái hiện Table 2): nối layout + văn bản vào một chuỗi đích so với tách task bằng tiền tố.
3. Đo xem tín hiệu phụ có làm giảm lượng dữ liệu cần để phục hồi sau khi cắt neuron MLP (kịch bản phục hồi kém) hay không.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Rationale | Lời giải thích trung gian do LLM sinh |
| $L_{\text{label}}$ | CE giữa dự đoán và nhãn (người hoặc giả) |
| $L_{\text{rationale}}$ | CE giữa dự đoán và rationale |
| $L=L_{\text{label}}+\lambda L_{\text{rationale}}$ | Mục tiêu multi-task của paper |
| Tiền tố task | "[label]" / "[rationale]" để phân biệt nhiệm vụ |
| Lợi ích triển khai | Không sinh rationale lúc test |
| Kết quả nổi bật | T5 770M + 80% dữ liệu ANLI vượt PaLM 540B few-shot CoT |

## 11. Kết luận ngắn

Rationale của teacher là tín hiệu giàu thông tin: học song song nhãn và rationale (multi-task) giúp student nhỏ cần ít dữ liệu hơn và có thể vượt cả LLM ở task hẹp, mà không tốn thêm chi phí suy luận. Bài học then chốt: tách rationale thành nhiệm vụ riêng, không gộp vào đích.

Link gốc: https://arxiv.org/abs/2305.02301
