# LLaVA-MoD: Making LLaVA Tiny via MoE Knowledge Distillation (2024)

**Tác giả:** Fangxun Shu, Yue Liao, Le Zhuo, Chenning Xu, Lei Zhang, Guanghao Zhang, Haonan Shi, Long Chen, Tao Zhong, Wanggui He, Siming Fu, Haoyuan Li, Bolin Li, Zhelun Yu, Si Liu, Hongsheng Li, Hao Jiang (Alibaba, CUHK, Zhejiang Univ., Beihang Univ.).  
**Venue / arXiv:** arXiv:2408.15881 (v3, 23/10/2024), ghi là "Technical Report"; paper không nêu venue hội nghị.  
**Nguồn:** https://arxiv.org/abs/2408.15881  
**Chủ đề:** Distill một MLLM (multimodal LLM, mô hình ngôn ngữ-thị giác) 7B xuống 0.5B/1.8B có LLM dạng Mixture-of-Experts; distill 2 giai đoạn: mimic (KL) rồi preference (KTO/DPO với teacher làm reference).  
**Vị trí trong lộ trình:** Đọc sau 24 (CLIP-KD, distill encoder), 02 (Hinton KD), 15 (sequence-level KD), và nên biết 28 (MiniLLM)/29 (GKD) vì paper trích các bài đó. Bài này đưa KD sang MLLM sinh văn bản và thêm "preference distillation". Bài kế tiếp 26 (LLaVA-KD) cạnh tranh trực tiếp với hướng này nhưng giữ student dense, không dùng MoE.

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch nguyên văn paper. Số liệu lấy từ paper (một số bảng PDF bị mangled; chỗ không chắc đã ghi rõ). Phần "Ứng dụng cho dự án" là suy luận của người ghi chú.

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

MLLM kiểu LLaVA = vision encoder (ở đây CLIP-ViT-L/14) + adapter (VL adaptor, MLP) + LLM. Mô hình lớn tốn kém để huấn luyện và triển khai. Các hướng trước cho MLLM nhỏ (MobileVLM, Imp, Bunny, MiniCPM-V...) chủ yếu tập trung **dữ liệu chất lượng cao**, và bị giới hạn bởi dung lượng mô hình. KD từ MLLM lớn sang nhỏ "hứa hẹn nhưng chưa được khám phá" (lời tác giả). Hai thách thức tác giả nêu:

1. Thiết kế kiến trúc student nhẹ nhưng vẫn đủ khả năng hấp thụ tri thức phức tạp.
2. Truyền tri thức đa nhiệm một cách hiệu quả.

Khái niệm nền:
- **MoE (Mixture of Experts):** thay một FFN bằng N FFN ("expert") cộng một router chọn top-k expert cho từng token. Tổng tham số lớn nhưng chỉ k expert chạy mỗi token (sparse).
- **Sparse upcycling:** khởi tạo các expert bằng cách copy FFN của mô hình dense đã train.
- **Preference optimization (PO), DPO, KTO:** huấn luyện trực tiếp từ dữ liệu "câu trả lời tốt/xấu" mà không cần reward model; DPO dùng cặp (tốt, xấu); KTO chỉ cần nhãn tốt/xấu từng mẫu.
- **Hallucination (ảo giác):** mô hình mô tả vật không có trong ảnh.

## 2. Ý tưởng chính

Student là MLLM có LLM nhỏ (Qwen-1.5 1.8B hoặc 0.5B), nhưng FFN được "upcycle" thành MoE để tăng khả năng biểu diễn mà vẫn tính toán ít. Huấn luyện theo 3 bước: (0) khởi tạo adapter bằng caption; (1) **mimic distillation** tối thiểu KL với teacher 7B, chia hai chặng "dense-to-dense" (kiến thức chung) rồi "dense-to-sparse" (kiến thức chuyên biệt, chuyển sang MoE); (2) **preference distillation**: dùng teacher làm mô hình reference trong loss kiểu DPO/KTO, để student gán xác suất cao hơn teacher cho câu trả lời tốt và thấp hơn cho câu xấu, nhờ đó vượt teacher về ảo giác. Kết quả: 2B vượt Qwen-VL-Chat-7B trung bình 8.8% chỉ với 5M mẫu.

## 3. Phương pháp chi tiết

### 3.1 Kiến trúc

$$y=\mathrm{LLM}_\phi\big(\mathrm{Proj}_\omega(\mathrm{ViT}_\chi(x_v)),\,x_i\big)$$

Ảnh 336x336, patch 14x14 cho 576 visual token. ViT và adaptor giữ nguyên; chỉ LLM bị thu nhỏ rồi sparsify.

**MoE router:**

$$r=\mathrm{Softmax}(x\,W_r),\qquad \tilde r=\mathrm{Top\text{-}k}(r),\qquad y=\sum_{i=1}^{N}\tilde r_i\,E_i(x)$$

Router là một lớp tuyến tính; trọng số các expert không chọn bị đặt bằng 0. Cấu hình ký hiệu E4T2 = 4 expert, top-2 hoạt động; E4T1 = top-1 (số tham số hoạt động bằng dense).

### 3.2 Ba giai đoạn

**Initialization.** Đóng băng ViT và LLM, chỉ train adaptor bằng cross-entropy:

$$\mathcal{L}_{Init}=-\mathbb{E}\log\pi_S(y_k\mid y_{<k},x)$$

Dữ liệu: 0.6M caption (LLaVA-1.5-Pretrain).

**Mimic distillation (a): Dense-to-Dense (D2D).** ViT đóng băng; train LLM + adaptor. Dữ liệu: 2.4M caption/hội thoại tổng quát. Loss KL giữa phân phối token (sau softmax) của student và teacher; công thức (6) trong paper viết là $-\mathbb{E}_{\pi_T}\log\frac{\pi_T}{\pi_S}$ (đã biểu diễn dưới dạng forward KL, kỳ vọng theo teacher):

$$\mathcal{L}_{D2D}=\mathrm{KL}(\pi_T\|\pi_S)\ \text{tại từng vị trí token}$$

(Ghi chú: dấu trừ trong công thức gốc có vẻ thừa, đây là cách người ghi chú hiểu theo ngữ cảnh "minimize KLD". Điều này trái với MiniLLM (28) dùng reverse KL.)

**Mimic distillation (b): Dense-to-Sparse (D2S).** Biến FFN của LLM thành MoE (sparse upcycling), dữ liệu đa nhiệm 1.4M (VQA, document, science, OCR). Chỉ train experts và adaptor. Loss = CE (nhãn thật) + KL:

$$\mathcal{L}_{D2S}=-\mathbb{E}\log\pi_S(y_k|y_{<k},x)+\mathrm{KL}(\pi_T\|\pi_S)$$

CE được thêm để "giảm bias vốn có của teacher".

**Preference distillation.** Với dữ liệu (x, y⁺, y⁻), 80K mẫu RLAIF-V, loss kiểu DPO với teacher làm reference:

$$\boxed{\mathcal{L}_{PD}=-\mathbb{E}\,\log\sigma\!\Big(\beta\log\frac{\pi_S(y^+|x)}{\pi_T(y^+|x)}-\beta\log\frac{\pi_S(y^-|x)}{\pi_T(y^-|x)}\Big)}$$

Điểm mấu chốt là $\pi_T$ (teacher) đóng vai reference model thay vì SFT model của chính student. Ý nghĩa: (i) student học phân biệt tốt/xấu như teacher; (ii) student cố gán xác suất cao hơn teacher cho y⁺ và thấp hơn cho y⁻, tức vượt teacher. Triển khai thực tế dùng **KTO** (Phụ lục A.1: "KTO loss"); công thức (8) viết ở dạng DPO. Hai điều này không khớp tuyệt đối, ghi chú để tránh nhầm khi reproduce.

### 3.3 Pseudo-code

```
# Stage 0: init
freeze ViT, LLM; train adaptor with CE on 0.6M captions
# Stage 1a: D2D (dense student, 2.4M general data)
for batch: logits_T = teacher(x) (no_grad); logits_S = student(x)
    loss = KL(softmax(logits_T) || softmax(logits_S)) on RESPONSE tokens only
    update LLM + adaptor
# Stage 1b: D2S
student.FFN -> MoE: copy FFN N times, add router (sparse upcycling)
for batch (1.4M multi-task):
    loss = CE(student, y) + KL(teacher || student)
    update experts + adaptor (top-k routing)
# Stage 2: preference
for (x, y+, y-) in 80K:
    lp_S(y), lp_T(y) = sequence log-probs
    loss = KTO/DPO(beta; reference = teacher)
    update experts + adaptor
```

## 4. Thí nghiệm và kết quả chính

**Thiết lập:** CLIP-ViT-L/14 (336), adaptor MLP 2 lớp, LLM Qwen-1.5/2. Teacher 7B (Qwen-1.5-7B hoặc Qwen-2-7B); student Qwen-1.5-1.8B, Qwen-1.5-0.5B (và Qwen-2 tương ứng). Teacher và student cùng họ LLM để chung vocabulary. 16 GPU A100, 1 epoch mỗi chặng, tổng khoảng 960 GPU giờ. Tổng dữ liệu 5M mẫu.

### Teacher (Bảng 1)

| LLM teacher | GQA | VisWiz | SQA | VQAT | MME | MMB | MMBCN | AVG |
|---|---|---|---|---|---|---|---|---|
| Qwen-1.5-7B | 62.3 | 40.7 | 70.9 | 55.3 | 72.1 | 68.5 | 64.9 | 62.1 |
| Qwen-2-7B | 62.5 | 37.9 | 73.2 | 57.2 | 78.0 | 71.6 | 71.1 | 64.5 |

### Kết quả hiểu (Bảng 2, trích AVG)

| Mô hình | #Sample | AVG |
|---|---|---|
| LLaVA-MoD-2B (Qwen-1.5-1.8B) | 5M | 59.9 |
| LLaVA-MoD-2B (Qwen-2-1.5B) | 5M | 61.6 |
| Qwen-VL-Chat-7B | 1450M | 56.7 |
| MiniCPM-V-2 (2.4B) | 570M | 66.9 |
| Mini-Gemini-2B | 2.7M | 57.1 |
| DeepSeek-VL-1.3B | 2000M | 58.5 |
| LLaVA-MoD-1B (Qwen-1.5-0.5B) | 5M | 54.1 |
| LLaVA-MoD-1B (Qwen-2-0.5B) | 5M | 55.7 |
| SPHINX-Tiny (1.1B) | 15M | 49.2 |

Tác giả: bản 2B "vượt Qwen-VL-Chat-7B 8.8%" (nói về lợi thế trung bình trong Fig. 1; bảng ghi 59.9 so với 56.7 cho bản Qwen-1.5, nên con số 8.8% phụ thuộc cách tính, ghi chú: không tái tạo được từ cột AVG này). Bản 1B hơn SPHINX-Tiny 13.2% (văn bản paper). Đáng chú ý MiniCPM-V-2 (66.9) vẫn cao hơn LLaVA-MoD-2B; paper nói "matches VILA-3B và MiniCPM-V" (bản đầu), không so mạnh với MiniCPM-V-2.

### Ảo giác (Bảng 3, trích)

| Mô hình | Object HalBench Resp ↓ | Ment ↓ | POPE F1 ↑ | MMHal Score ↑ | Hall ↓ |
|---|---|---|---|---|---|
| Teacher Qwen-1.5-7B | 50.1 | 24.8 | 84.9 | 2.60 | 20.7 |
| Teacher Qwen-2-7B | 29.7 | 23.4 | 85.7 | 2.88 | 14.5 |
| RLHF-V (13B) | 12.2 | 7.5 | 86.2 | 2.45 | 51.0 |
| LLaVA-MoD-2B (Qwen-1.5-1.8B) | 11.4 | 7.2 | 87.0 | 2.76 | 17.2 |
| LLaVA-MoD-2B (Qwen-2-1.5B) | 11.2 | 5.9 | 87.2 | 2.91 | 13.8 |

Student vượt teacher trên các chỉ số ảo giác (ví dụ Resp 11.4 so với 50.1).

## 5. Phân tích / ablation

Tất cả ablation mimic dùng student Qwen-1.5-1.8B, teacher Qwen-1.5-7B, AVG trên 7 benchmark.

| Thí nghiệm | Kết quả AVG |
|---|---|
| Mimic vs Mimic+Preference (Bảng 4) | 60.2 vs 59.9 (hiểu giảm nhẹ) |
| Hallucination: Mimic vs +Preference (Bảng 5) | Resp 39.1 xuống 11.4; Ment 22.6 xuống 7.2 |
| SFT vs KD, MoE E4T2 (Bảng 6) | 55.7 vs 60.2 (KD hơn 8.1%; MME 63.2 lên 68.4, MMB 59.2 lên 65.1) |
| D2S vs D2D+D2S (Bảng 7) | 57.9 (1536 GPU giờ) vs 60.2 (960 GPU giờ) |
| Distill Response+Instruction vs Response (Bảng 8) | 58.4 vs 60.2 |
| Dense vs Sparse E4T1 (Bảng 9) | 57.1 vs 59.2 |
| DPO vs KTO (Bảng 15) | 59.6 vs 59.9; Resp halluc 20.0 vs 11.4 (Bảng 16) |
| Số expert 4 vs 8, top-2 (Bảng 13) | AVG 64.0 vs 63.6 (5 benchmark; không cải thiện) |
| Top-1 vs Top-2, 4 expert (Bảng 14) | 62.6 vs 64.0 |

Những điểm chính:
- **Preference distillation giảm ảo giác mạnh nhưng không tăng khả năng hiểu** (tác giả liên hệ hiện tượng tương tự ở LLM).
- **KD cung cấp tín hiệu tối ưu tốt hơn SFT cho MoE** (có CE ground-truth chỉ ở D2S).
- **General-to-specialized** vừa tốt hơn vừa rẻ hơn (62.5% GPU giờ).
- **Chỉ distill token response**, không distill token instruction, tốt hơn (tác giả đoán: tránh overfit mẫu instruction).
- **Teacher capacity:** với student 1.8B, teacher 7B hơn teacher 4B 2.2% (60.2 vs 58.9); với student 0.5B hầu như không khác (53.8 vs 53.7). Tác giả đề xuất "middle teacher" để bắc cầu khoảng cách dung lượng (đề xuất, không thử nghiệm đầy đủ trong paper).
- **KTO ổn định hơn DPO**, tác giả lý giải KTO dùng nhãn tốt/xấu từng mẫu (là giả thuyết).

## 6. Giới hạn

**Tác giả nêu (Phụ lục C):** student và teacher phải cùng họ LLM để chung vocabulary; cần nạp cả hai mô hình khi huấn luyện nên tốn bộ nhớ (gợi ý pre-extract logit của teacher).

**Người ghi chú nhận xét:**
- Không tách bạch tác dụng của MoE và của KD/dữ liệu so với dense có cùng tham số tổng (chỉ có so sánh E4T1 vs dense ở cùng số tham số hoạt động).
- Student cuối là MoE: tổng tham số lớn hơn số hoạt động, nhưng bộ nhớ GPU phải chứa tất cả expert (quan trọng khi mục tiêu là GPU ≤ 8 GB).
- Công thức (6)–(8) có dấu/ký hiệu không nhất quán với triển khai (xem mục 3.2); số hyperparameter (batch size) trong văn bản (512/256) và trong Bảng 11 (256/128) khác nhau.
- Preference distillation làm AVG hiểu giảm nhẹ; chưa rõ trên tác vụ cần độ chính xác từng ký tự như OCR.
- Không so sánh với phương pháp distill MLLM dense khác trong cùng điều kiện (bài 26 sau đó làm).

## 7. Liên hệ với các bài khác

- **02 Hinton KD, 15 Sequence-level KD:** mimic distillation = forward KL token-level (word-level KD).
- **28 MiniLLM, 29 GKD:** paper nhắc cả hai; nhưng chính LLaVA-MoD dùng KL thường, không dùng reverse KL.
- **26 LLaVA-KD:** cùng mục tiêu MLLM nhỏ; LLaVA-KD không dùng MoE và thêm distill visual token và quan hệ.
- **24 CLIP-KD:** ở đây ViT được đóng băng, nên vision encoder không được distill.
- **41 Capacity gap law, 39 Small models struggle...:** ablation teacher 7B vs 4B cho student 0.5B/1.8B là bằng chứng nhỏ về capacity gap.
- **36 Distilling step-by-step, 37 Orca:** các bài khác dùng dữ liệu/giải thích từ teacher; ở đây là logit + preference.

## 8. Ứng dụng cho dự án (suy luận)

Mức liên quan: **trung bình**.

- Đáng học: (1) **D2D rồi D2S**, kiến thức chung trước, chuyên biệt sau; (2) **chỉ tính KL trên token response** (với OCR: trên token văn bản đầu ra, bỏ phần prompt/instruction); (3) **thêm CE ground truth** cùng KL khi dữ liệu có nhãn (OCR có nhãn chuẩn nên rất phù hợp); (4) teacher quá lớn so với student 0.1B có thể không hơn teacher trung gian.
- Không đáng sao chép: MoE student. Mục tiêu của ta là GPU ≤ 8 GB, batch-1 latency; MoE giữ toàn bộ expert trong bộ nhớ và router thêm chi phí, trong khi ta đã có hướng prune có cấu trúc.
- Preference distillation: OCR có "câu trả lời đúng" xác định, nên có thể tạo cặp y⁺/y⁻ rẻ (đúng chuẩn vs lỗi sinh tự động, hoặc output của student pruned vs ground truth). Hướng này giảm "ảo giác" ký tự, nhưng cần kiểm chứng; paper chỉ cho thấy tác dụng trên ảo giác ảnh.
- Liên hệ D_lin/D_res (suy luận): D2S của paper là bước "dense sang sparse" có phần giống một phép phẫu thuật cấu trúc rồi phục hồi bằng KD + CE; phần dữ liệu đa nhiệm nhỏ (1.4M) đủ để phục hồi. Paper không báo cáo nhận định nào về neuron-vs-layer.

## 9. Thí nghiệm gợi ý

1. Ablation "response-only KL" so với "KL trên cả prompt" trong khôi phục sau prune 0.3B; đo CER/WER OCR.
2. So sánh loss KL + CE (hệ số 1:1) với chỉ KL, trên dữ liệu OCR có nhãn.
3. Thử KTO nhỏ (vài chục nghìn mẫu): y⁻ là output lỗi của student pruned, y⁺ là ground truth, reference là teacher 4B; theo dõi lỗi chèn/xóa ký tự.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý | Ghi chú |
|---|---|---|
| Router MoE | $r=\mathrm{softmax}(xW_r)$, top-k, $y=\sum\tilde r_iE_i(x)$ | E4T2 mặc định |
| Sparse upcycling | copy FFN thành N expert | khởi tạo từ dense |
| D2D | KL(teacher‖student) trên response | dữ liệu chung 2.4M |
| D2S | CE + KL, train experts + adaptor | dữ liệu đa nhiệm 1.4M |
| Preference distill | DPO/KTO với reference = teacher | 80K RLAIF-V |
| Thứ tự | init → D2D → D2S → preference | tổng ~960 GPU giờ |

## 11. Kết luận ngắn

LLaVA-MoD cho thấy KD token-level hai chặng (chung rồi chuyên biệt) cộng preference distillation có thể đưa MLLM 1–2B tiến gần các mô hình 7B với 5M mẫu, và giảm ảo giác vượt teacher. Phần có thể mang sang dự án là công thức huấn luyện (response-only KL, KL+CE, thứ tự chặng), không phải kiến trúc MoE.

Link gốc: https://arxiv.org/abs/2408.15881
