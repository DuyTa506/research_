# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2025) — phần distillation

**Tác giả:** DeepSeek-AI (Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z.F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao... và nhiều người khác). Mục "Author Contributions" ghi Zhibin Gou dẫn dắt loạt R1-distill.  
**Venue / arXiv:** arXiv:2501.12948 (bản v2 ngày 4/1/2026 là bản ta đọc; paper text không nêu tên hội nghị/tạp chí).  
**Nguồn:** https://arxiv.org/abs/2501.12948  
**Chủ đề:** Technical report dài (86 trang). Ghi chú này chỉ giải thích ngắn gọn R1/R1-Zero, rồi **tập trung vào Phụ lục F "DeepSeek-R1 Distillation"**: distill R1 vào các model dày đặc nhỏ và so sánh "distillation vs RL trên model nhỏ". Các phần safety, benchmark tổng quát, thất bại (PRM, MCTS) chỉ nhắc rất ngắn.  
**Vị trí trong lộ trình:** Đọc sau 35-37 (distillation bằng dữ liệu teacher sinh) và 15 Sequence-level KD. Bài này là ví dụ quy mô công nghiệp, dữ liệu là **long chain-of-thought** đã lọc theo tính đúng. Tiếp theo: 39 Small models struggle with strong reasoners (phản biện: không phải student nhỏ nào cũng hưởng lợi), 40 Gemma 2, 44 Qwen3 (distillation trong quy trình hậu huấn luyện).

> Ghi chú học tập bằng tiếng Việt, diễn giải lại, không phải bản dịch. Phần "Ứng dụng cho dự án" là suy luận của chúng ta. Một số số liệu nằm trong bảng của phụ lục, chép nguyên từ paper.

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

- Lập luận của LLM thường dựa trên **chain-of-thought (CoT)** và dữ liệu lập luận do người viết, vừa đắt vừa giới hạn mô hình ở "cách nghĩ của người".
- Gloss: **Reinforcement learning (RL)** = học bằng thưởng/phạt thay vì bắt chước nhãn. **Verifiable reward** = phần thưởng kiểm tra được bằng luật (đáp án toán đúng, code qua test), không cần mô hình chấm. **SFT** = supervised fine-tuning, huấn luyện bắt chước. **Distillation** (ở đây) = SFT student trên văn bản do teacher sinh.
- Câu hỏi của paper: nếu RL thuần (không SFT trước) có thể khiến một base model tự phát triển lập luận dài, thì khả năng này có "chuyển" được xuống model nhỏ không, và bằng cách nào rẻ hơn?

## 2. Ý tưởng chính

R1-Zero: lấy DeepSeek-V3-Base, huấn luyện RL thuần bằng GRPO với thưởng theo luật (đúng/sai + đúng định dạng thẻ `<think>`), model tự học viết lời giải dài với tự kiểm tra và quay lại ("aha moment"). R1: thêm pipeline nhiều giai đoạn (cold-start, RL, rejection sampling + SFT, RL lần hai) để dễ đọc và hữu ích hơn. Phần distillation: dùng R1 sinh khoảng 800K mẫu (600K lập luận + 200K không lập luận), rồi **chỉ SFT** (không RL) các base model Qwen và Llama từ 1.5B đến 70B. Phát hiện chính: SFT trên lời giải của R1 cho model nhỏ kết quả vượt xa việc tự chạy RL cùng cỡ ở thí nghiệm của họ (Qwen-32B).

## 3. Phương pháp chi tiết

### 3.1 R1-Zero và R1 (tóm tắt ngắn)

GRPO (Group Relative Policy Optimization): với mỗi câu hỏi $q$, lấy $G$ đầu ra $\{o_i\}$ từ policy cũ, mục tiêu:

$$\mathcal{J}_{GRPO}(\theta)=\mathbb{E}\Big[\tfrac1G\sum_{i=1}^{G}\min\Big(\tfrac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}A_i,\ \mathrm{clip}\big(\tfrac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)},1-\varepsilon,1+\varepsilon\big)A_i\Big)-\beta\,\mathbb{D}_{KL}(\pi_\theta\|\pi_{ref})\Big]$$

$$\boxed{A_i=\frac{r_i-\mathrm{mean}(\{r_1,\dots,r_G\})}{\mathrm{std}(\{r_1,\dots,r_G\})}}$$

Điểm mấu chốt: advantage tính **tương đối trong nhóm**, không cần value network như PPO. Thưởng R1-Zero: $\text{Reward}_{rule}=\text{Reward}_{acc}+\text{Reward}_{format}$, không dùng reward model thần kinh cho tác vụ lập luận vì dễ bị reward hacking. Cài đặt đã nêu: lr 3e-6, hệ số KL 0.001, nhiệt độ rollout 1, mỗi câu 16 mẫu, độ dài tối đa 32,768 token rồi 65,536 sau bước 8.2k, tổng 10,400 bước; AIME 2024 pass@1 từ 15.6% lên 77.9% (86.7% với cons@16 theo paper). Chi tiết thêm của R1 (giai đoạn Dev1-3, thưởng nhất quán ngôn ngữ, reward model có ích/an toàn) xem paper; không phải trọng tâm ở đây.

### 3.2 Dữ liệu distillation (~800K mẫu, Phụ lục B.3.3)

- **Reasoning data (~600K):** sinh bằng rejection sampling từ checkpoint sau RL giai đoạn 1. Với mỗi prompt lấy nhiều đáp án, **chỉ giữ đáp án đúng**. Đánh giá đúng bằng luật, hoặc (với phần mở rộng) bằng DeepSeek-V3 làm giám khảo sinh. **Lọc** CoT trộn ngôn ngữ, đoạn quá dài, khối code.
- **Non-reasoning data (~200K):** viết lách, hỏi đáp sự kiện, tự nhận thức, dịch; lấy lại từ SFT data của DeepSeek-V3, thêm dữ liệu software engineering (sửa lỗi, front-end). Với một số tác vụ cho V3 sinh CoT trước; câu đơn giản như "hello" thì không có CoT.

Thống kê SFT (Table 5):

| Domain | Số mẫu | Vòng TB | Token TB |
|---|---|---|---|
| Math | 395,285 | 1.0 | 6,094.2 |
| Code | 211,129 | 1.1 | 7,435.7 |
| STEM | 10,124 | 1.0 | 4,928.8 |
| Logic | 10,395 | 1.0 | 2,739.0 |
| General | 177,812 | 1.1 | 1,419.8 |
| **Tổng** | 804,745 | 1.0 | 5,355.3 |

Hầu hết là một lượt hội thoại.

### 3.3 Huấn luyện student

Chỉ SFT (cross-entropy trên phản hồi, như 35-37). 2-3 epoch trên 800K mẫu, lịch lr cosine giảm tới 1/10 giá trị đầu, ngữ cảnh tối đa 32,768 token, batch 64. Base và lr đầu (Table 6):

| Model distill | Base | lr đầu |
|---|---|---|
| R1-Distill-Qwen-1.5B | Qwen2.5-Math-1.5B | 1e-4 |
| R1-Distill-Qwen-7B | Qwen2.5-Math-7B | 8e-5 |
| R1-Distill-Qwen-14B | Qwen2.5-14B | 7e-5 |
| R1-Distill-Qwen-32B | Qwen2.5-32B | 6e-5 |
| R1-Distill-Llama-8B | Llama-3.1-8B | 5e-5 |
| R1-Distill-Llama-70B | Llama-3.3-70B-Instruct | 2e-5 |

(Quy tắc rút ra: model càng nhỏ, lr đầu càng lớn.) Không có RL cho student; paper nói RL có thể tăng thêm nhưng để cộng đồng khám phá.

### 3.4 Pseudo-code

```
# Teacher: R1 (sau RL giai đoạn 1)
for q in reasoning_prompts:
    answers = [R1.sample(q) for _ in range(K)]
    keep    = [a for a in answers if is_correct(a, q) and readable(a)]
    D_reason.extend(keep)             # ~600K tổng
D = D_reason + D_nonreason(~200K, từ V3)      # ~800K
for base in [Qwen2.5-Math-1.5B, ..., Llama-3.3-70B-Instruct]:
    student = SFT(base, D, epochs=2..3, lr=table6[base], max_len=32768, bs=64)
# KHÔNG có giai đoạn RL
```

## 4. Thí nghiệm và kết quả chính

### 4.1 R1 và R1-Zero (chỉ vài số, từ Table 3/8)

R1: AIME 2024 pass@1 79.8; MATH-500 97.3; GPQA Diamond 71.5; LiveCodeBench 65.9; Codeforces rating 2029. R1-Zero: AIME 77.9, MATH-500 95.9, GPQA 75.8. Chi phí huấn luyện ước tính (rent H800 $2/giờ GPU): R1-Zero 101K giờ ($202K), tạo SFT data 5K giờ ($10K), R1 41K giờ ($82K), tổng 147K giờ ($294K). Model chính: MoE 671B tổng, 37B kích hoạt/token.

### 4.2 Model distill so với baseline (Table 15, đánh giá với nhiệt độ 0.6, top-p 0.95, tối đa 32,768 token)

| Model | AIME24 pass@1 | AIME24 cons@64 | MATH-500 | GPQA-D | LCB | CF rating |
|---|---|---|---|---|---|---|
| GPT-4o-0513 | 9.3 | 13.4 | 74.6 | 49.9 | 32.9 | 759 |
| Claude-3.5-Sonnet-1022 | 16.0 | 26.7 | 78.3 | 65.0 | 38.9 | 717 |
| R1-Distill-Qwen-1.5B | 28.9 | 52.7 | 83.9 | 33.8 | 16.9 | 954 |
| R1-Distill-Qwen-7B | 55.5 | 83.3 | 92.8 | 49.1 | 37.6 | 1189 |
| R1-Distill-Qwen-14B | 69.7 | 80.0 | 93.9 | 59.1 | 53.1 | 1481 |
| R1-Distill-Qwen-32B | 72.6 | 83.3 | 94.3 | 62.1 | 57.2 | 1691 |
| R1-Distill-Llama-8B | 50.4 | 80.0 | 89.1 | 49.0 | 39.6 | 1205 |
| R1-Distill-Llama-70B | 70.0 | 86.7 | 94.5 | 65.2 | 57.5 | 1633 |

Paper nhấn mạnh: bản 1.5B đã vượt GPT-4o và Claude-3.5-Sonnet trên các benchmark toán (AIME, MATH-500) nhưng **không** vượt trên GPQA Diamond và LiveCodeBench; hiệu năng tăng dần theo kích thước student. (Phần "vượt ... closed-source" của paper chỉ đúng cho toán, đây là đọc kỹ bảng.)

### 4.3 Distillation vs RL trên model nhỏ (Table 16)

Thí nghiệm: RL quy mô lớn (hơn 10K bước, dữ liệu math, code, STEM) trên Qwen2.5-32B-Base, thu được **Qwen2.5-32B-Zero** (lr 2e-6, KL 0.001, nhiệt độ 1, 16 mẫu/câu, tối đa 32,768 token, batch 512).

| Model | AIME24 pass@1 | AIME24 cons@64 | MATH-500 | GPQA-D | LCB |
|---|---|---|---|---|---|
| QwQ-32B-Preview | 50.0 | 60.0 | 90.6 | 54.5 | 41.9 |
| Qwen2.5-32B-Zero (RL) | 47.0 | 60.0 | 91.6 | 55.0 | 40.2 |
| R1-Distill-Qwen-32B (SFT từ R1) | **72.6** | **83.3** | **94.3** | **62.1** | **57.2** |

Kết luận của tác giả (hai ý): (1) distill từ model mạnh hơn vào model nhỏ cho kết quả rất tốt, còn model nhỏ tự RL quy mô lớn "cần năng lực tính toán khổng lồ và có thể không đạt kết quả của distillation"; (2) distillation kinh tế và hiệu quả, nhưng vượt khỏi ranh giới trí tuệ hiện có có thể vẫn cần base model mạnh hơn và RL quy mô lớn hơn.

Thí nghiệm bổ sung (Table 17): **Qwen2-Math-7B-Zero** (RL ~10,000 bước policy gradient trên base ra mắt trước o1 để đảm bảo base chưa thấy dữ liệu lập luận): AIME 2024 22.3%, AIME 2025 18.1%, so với Qwen2-Math-7B-Instruct 7.9% / 4.6%, GPT-4o-0513 9.3% / "-". Tức RL trên model 7B vẫn làm lập luận nổi lên, nhưng ở mức thấp hơn nhiều so với R1-Distill-Qwen-7B (AIME 55.5, Table 15; lưu ý hai số này không cùng base, Qwen2-Math-7B vs Qwen2.5-Math-7B, nên so sánh chỉ mang tính tham khảo - nhận xét của chúng ta).

## 5. Phân tích / ablation

- **Tầm quan trọng của base model (Phụ lục G.1):** RL từ base trên model 7B dense và MoE 16B "không đem lại cải thiện có ý nghĩa" trên AIME; chúng lặp lại khi độ dài tăng và không tận dụng được long CoT. Chỉ khi chuyển sang 32B dense, 230B MoE, 671B MoE mới thấy tăng đáng kể. Họ kết luận hiệu quả RL từ base phụ thuộc nhiều vào năng lực nền. (Lưu ý đối chiếu: ở Table 17, Qwen2-Math-7B-Zero lại tăng được, nên "7B không RL được" là phát biểu về cấu hình thử ban đầu của họ, không tuyệt đối.)
- **Verifier:** thành công của R1-Zero rất phụ thuộc phần thưởng đáng tin; luật và LLM-judge cho đáp án ngắn hoạt động, nhưng không tổng quát cho viết lách mở.
- **Cả SFT và RL đều cần:** RL khám phá được lập luận mà dữ liệu người không có; SFT cần cho tác vụ khó định nghĩa thưởng.
- **Lọc dữ liệu distillation:** giữ đáp án đúng, bỏ ngôn ngữ trộn, đoạn quá dài, khối code; paper không có ablation riêng về từng bước lọc.
- **Chất lượng teacher > dữ liệu người:** paper khẳng định "model distill từ output chất lượng cao của teacher luôn vượt model train trực tiếp trên dữ liệu người" (trích dẫn Busbridge et al., 2025); trong phần ta đọc không có bảng nào trực tiếp so sánh.

## 6. Giới hạn

**Do tác giả nêu:**
- Student chỉ dùng SFT, không RL (có thể còn nâng được).
- Distillation không thể "đẩy ranh giới": cần base mạnh hơn + RL lớn hơn.
- Dữ liệu SFT chủ yếu một lượt, hạn chế hội thoại nhiều lượt.
- R1: overthinking với câu đơn giản, nhạy với prompt (few-shot làm giảm hiệu năng), trộn ngôn ngữ, yếu ở tác vụ software engineering.

**Do chúng ta nhận xét:**
- Thí nghiệm "distillation vs RL" chỉ một cấu hình chính (Qwen2.5-32B), không công bố ngân sách tính toán đồng đều: RL >10K bước so với SFT 800K mẫu có chi phí rất khác, nên "distill thắng" chưa phải so sánh công bằng theo FLOPs. Chi phí sinh 800K mẫu từ R1 671B cũng không được tính vào bên distillation (chỉ có 5K giờ GPU tạo dữ liệu SFT).
- Student 1.5B, 7B dùng base Qwen2.5-**Math**: chuyên toán sẵn, nên lợi ích trên toán có thể phản ánh một phần nền tảng này.
- Không có thí nghiệm student nhỏ đa dạng kích thước cho cùng kỹ thuật RL (ngoài Qwen2-Math-7B); không đo token/độ trễ khi suy luận (long CoT trung bình 5,355 token/mẫu).
- Các số bảng 15 trên model nhỏ ở GPQA và LiveCodeBench vẫn thấp hơn GPT-4o/Claude cho 1.5B, nên cần đọc "vượt closed-source" cẩn thận.

## 7. Liên hệ với các bài khác

- **35 Self-Instruct, 37 Orca:** cùng họ "teacher sinh văn bản, student SFT". Khác biệt: R1 dùng **lọc theo đúng-sai có kiểm chứng**, và trace là long CoT tự phát sinh bằng RL, không phải prompt kiểu "think step by step".
- **36 Distilling step-by-step:** cũng thêm lời giải của teacher, nhưng dùng multi-task và rationale ngắn; ở R1 là một chuỗi dài làm đích duy nhất.
- **39 Small models struggle with strong reasoners:** phản biện/giới hạn: student nhỏ (<=3B) học từ CoT dài của teacher mạnh có thể kém hơn học CoT ngắn; ở đây R1-Distill-1.5B vẫn lên mạnh nhưng ở toán; cần đọc cùng nhau.
- **41 Capacity gap law, 42 Distillation scaling laws:** khung để giải thích vì sao student nhỏ và teacher quá mạnh có thể không khớp.
- **44 Qwen3 distillation, 40 Gemma 2 distillation:** hai cách dùng distillation trong quy trình huấn luyện thực tế (on-policy/ logit-based) so với SFT thuần ở đây.
- **15 Sequence-level KD:** R1-Distill là sequence-level KD ở quy mô lớn, mẫu dùng hard sampling (đã lọc).
- **28-34 (MiniLLM, GKD, DistiLLM...):** nhóm logit-based, đối lập với phương pháp không dùng logit ở đây.

## 8. Ứng dụng cho dự án (suy luận)

Mức liên quan: **trung bình về phương pháp, thấp về bài toán.** OCR không cần long CoT; student 0.8B sau pruning không phải bài toán toán/code.
- **Bài học "distill > RL cho model nhỏ"** ủng hộ lựa chọn của dự án: phục hồi student đã cắt bằng KD từ teacher 4B thay vì cố RL (RL cần verifier và nhiều tính toán). Đây là suy luận; paper không bàn pruning.
- **Lọc theo đúng/sai rồi mới SFT** áp dụng trực tiếp được: với OCR có thể giữ chỉ những đầu ra teacher có độ khớp cao với nhãn thật (CER thấp) hoặc nhất quán nhiều lần decode, rồi dùng làm đích. Tương tự "rejection sampling".
- **Lưu ý về năng lực nền:** G.1 nói RL từ base phụ thuộc vào năng lực model; tương tự, student bị cắt quá mạnh (~0.1B) có thể không hấp thụ được tín hiệu giàu thông tin từ teacher 4B (nên đối chiếu 41, 08).
- **Không áp dụng:** GRPO, long CoT, thưởng theo luật cho toán/code.

## 9. Thí nghiệm gợi ý

1. Rejection-sampling KD: teacher 4B decode mỗi ảnh K lần (nhiệt độ >0), giữ mẫu có CER nhỏ nhất so với nhãn thật; so với dùng một lần decode tham lam. Đo CER sau phục hồi student 0.3B.
2. Ngân sách đồng đều: cùng số giờ GPU, so (a) SFT trên dữ liệu teacher lọc và (b) KD logit từ teacher; xem cái nào phục hồi nhanh hơn, đặc biệt với kịch bản cắt neuron MLP.
3. Kích thước student: lặp thí nghiệm 1 trên các mức cắt 0.3B và 0.1B để xem đường cong giống Table 15 (lợi ích tăng dần theo kích thước) hay có điểm gãy (gợi ý từ 39).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| GRPO | RL với advantage chuẩn hoá trong nhóm $G$ mẫu, không cần value net |
| $A_i=(r_i-\text{mean})/\text{std}$ | Advantage nhóm |
| Rule-based reward | Thưởng đúng đáp án + đúng định dạng |
| R1-Zero | RL thuần trên base, không SFT |
| 800K SFT data | ~600K lập luận (rejection sampling, chỉ đúng) + ~200K khác |
| R1-Distill-* | SFT-only student: Qwen 1.5B/7B/14B/32B, Llama 8B/70B |
| Qwen2.5-32B-Zero | RL >10K bước trên Qwen2.5-32B-Base (đối chứng) |
| Số then chốt | AIME24 pass@1: Zero 47.0 vs Distill-32B 72.6 |

## 11. Kết luận ngắn

Trong thí nghiệm của DeepSeek, SFT model nhỏ trên lời giải đã lọc của R1 cho kết quả tốt hơn nhiều so với tự RL cùng cỡ (Qwen-32B: AIME 72.6 vs 47.0). Dữ liệu teacher sinh, lọc bằng tính đúng, là cách rẻ và hiệu quả để chuyển khả năng lập luận xuống student; nhưng kết luận này gắn với toán/code có verifier và chưa kiểm soát đồng đều ngân sách tính toán.

Link gốc: https://arxiv.org/abs/2501.12948
