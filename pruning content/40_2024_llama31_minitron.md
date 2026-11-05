# LLM Pruning and Distillation in Practice: The Minitron Approach (2024)

**Tác giả:** Sharath Turuvekere Sreenivas*, Saurav Muralidharan*, Raviraj Joshi, Marcin Chochowski, Ameya Sunil Mahabaleshwarkar, Gerald Shen, Jiaqi Zeng, Zijia Chen, Yoshi Suhara, Shizhe Diao, Chenhan Yu, Wei-Chun Chen, Hayley Ross, Oluwatobi Olabiyi, Ashwath Aithal, Oleksii Kuchaiev, Daniel Korzekwa, Pavlo Molchanov, Mostofa Patwary, Mohammad Shoeybi, Jan Kautz, Bryan Catanzaro (NVIDIA). (* đóng góp ngang nhau)  
**Venue / arXiv:** arXiv:2408.11796 (v1: 21/08/2024; bản v4: 09/12/2024). Là technical report của NVIDIA; không ghi hội nghị.  
**Nguồn:** https://arxiv.org/abs/2408.11796  
**Chủ đề:** Áp dụng công thức Minitron lên Llama 3.1 8B (xuống 4B, hai biến thể depth và width) và Mistral NeMo 12B (xuống 8B) khi *không có dữ liệu pre-training gốc*; đề xuất "teacher correction" và chỉ số layer dựa trên tác vụ downstream.  
**Vị trí trong lộ trình:** Đọc trước: 39 (Minitron, bắt buộc; bài này bỏ qua phần giải thích lại). Bài này thêm: (1) teacher correction, (2) cách chọn khối layer liên tiếp theo Winogrande, (3) bằng chứng thứ hai về width > depth về độ chính xác nhưng depth > width về tốc độ. Tiếp theo: 41 (Flextron), 42 (Puzzle).

> Ghi chú học tập bằng tiếng Việt, diễn giải và tóm tắt, không phải bản dịch. Số liệu lấy từ paper (một số bảng bị lệch cột trong bản text; đã đối chiếu). Phần "Ứng dụng cho dự án" là suy luận của chúng ta.

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

- Bài 39 cho thấy prune + distill rẻ hơn train từ đầu rất nhiều, nhưng giả định có **dữ liệu pre-training gốc** của mô hình để distill. Các mô hình mở mạnh (Llama 3.1, Mistral NeMo) lại train trên dữ liệu riêng, không công bố. Câu hỏi: nếu chỉ có một tập dữ liệu khác (ở đây là tập continued-training của Nemotron-4), distillation còn hiệu quả không?
- **Distribution shift (lệch phân phối)** giữa dữ liệu teacher đã học và dữ liệu đang dùng để distill: tác giả giả thuyết rằng phân phối sub-word token khác nhau khiến teacher đưa ra "hướng dẫn" (logits) kém tối ưu.
- Với bài toán của dự án (teacher 4B + student nhỏ + dữ liệu OCR riêng), đây chính là tình huống tương tự: dữ liệu distill khác dữ liệu teacher từng thấy.

## 2. Ý tưởng chính

Trước khi prune, **fine-tune nhẹ teacher** (~100B token) trên chính tập dữ liệu sẽ dùng để distill; đây là **teacher correction**. Sau đó prune (activation-based, theo Minitron) và distill bằng forward-KL trên logits. Ngoài ra đề xuất chọn layer để bỏ bằng *độ chính xác downstream* (Winogrande) trên các **khối layer liên tiếp** thay vì dùng LM loss/PPL, vì LM loss chọn ra những layer rời rạc nhưng cho kết quả downstream tệ. Áp dụng để tạo Llama-3.1-Minitron-4B (width và depth) và MN-Minitron-8B.

## 3. Phương pháp chi tiết

### 3.1 Pipeline (Hình 1)

```
1. Teacher correction : fine-tune teacher gốc ~100B token trên tập D_distill   (lr = 1/5 peak, warm-up 120 bước)
2. Pruning            : importance (activation, 1024 mẫu calibration) -> xếp hạng -> cắt (one-shot)
                        width : L2 (batch) + mean (seq) ; depth : bỏ khối layer liên tiếp theo Winogrande
3. Distillation       : student (đã cắt) học từ teacher đã hiệu chỉnh, chỉ forward-KL trên logits
```

Bỏ bước NAS của Minitron; dùng cấu hình thủ công lấy cảm hứng từ Minitron 4B/8B. Depth được xem là trường hợp riêng: **không kết hợp depth với các trục khác**.

### 3.2 Teacher correction

- Fine-tune teacher gốc ~100B token với learning rate bằng 1/5 mức đỉnh, cùng batch size, min LR và lịch decay của mô hình gốc.
- Tác dụng lên teacher: ảnh hưởng nhỏ đến accuracy downstream (có tác vụ lên, có tác vụ xuống, Bảng 1); tác giả giả thuyết phần xuống là do dữ liệu fine-tune.
- Có thể làm **song song** với distillation (prune teacher gốc rồi distill từ teacher đang được hiệu chỉnh liên tục) và cho kết quả ngang với dùng teacher đã hiệu chỉnh xong (Hình 5).

### 3.3 Chọn layer để bỏ (depth)

Hai chỉ số được xét: LM validation loss/PPL và accuracy tác vụ downstream. Không dùng Block Importance (BI) vì "đã được chỉ ra là kém hơn PPL". Hai quan sát dẫn tới quy tắc cuối:
1. Importance theo LM loss/PPL không cho mô hình cắt tốt nhất trên downstream.
2. Bỏ khối *liên tiếp* tốt hơn bỏ rời rạc (đồng ý với Gromov et al., bài 32).

Quy tắc: bỏ khối layer liên tiếp làm Winogrande giảm ít nhất. Với Llama 3.1 8B (32 layer), kết quả là bỏ layer 16-31 (nửa sau, giữ lại 16 layer).

### 3.4 Cấu hình đích (Table 3)

| | Llama-3.1-Minitron 4B-Width | 4B-Depth | MN-Minitron-8B |
|---|---|---|---|
| Tổng tham số | 4.5B | 4.5B | 8.4B |
| Không tính embedding | 3.7B | 3.5B | 7.3B |
| Hidden size | 3072 | 4096 | 4096 |
| MLP hidden | 9216 | 14336 | 11520 |
| Depth | 32 | 16 | 40 |
| Query heads / nhóm / head dim | 32 / 8 / 128 | 32 / 8 / 128 | 32 / 8 / 128 |
| Vocabulary | 128256 | 128256 | 131072 |

Nguồn: Llama 3.1 8B (hidden 4096 -> 3072, MLP 14336 -> 9216, head và depth giữ nguyên cho bản Width; depth 32 -> 16 cho bản Depth). MN-Minitron-8B từ Mistral NeMo 12B: hidden 5120 -> 4096, MLP 14336 -> 11520.

### 3.5 Distillation và siêu tham số (Table 4)

Chỉ forward-KL trên logits, *bỏ hẳn* cross-entropy LM (theo best practice của bài 39).

| | Llama-3.1-Minitron | MN-Minitron |
|---|---|---|
| Peak LR | 1e-4 | 1e-4 |
| Min LR | 1e-5 | 4.5e-7 |
| Warm-up | 40 bước | 60 bước |
| Decay | cosine | cosine |
| Global batch | 1152 | 768 |
| Context | 8192 | 8192 |
| Tổng token | 94B | 380B |

Phần cứng: 32 node DGX H100. Dữ liệu: Nemotron-4 continued-training (CT) dataset cho cả teacher correction, pruning và distillation.

## 4. Thí nghiệm và kết quả chính

**Model base (Table 1, trích; đánh giá do tác giả chạy, trừ Gemma2 2B*):**

| Benchmark | Gemma2 2B* | Minitron 4B | Llama-3.1-Minitron 4B-Depth | 4B-Width | Llama 3.1 8B | MN-Minitron 8B | Mistral NeMo 12B-Base | 12B-FT |
|---|---|---|---|---|---|---|---|---|
| Tham số tổng | 2.6B | 4.2B | 4.5B | 4.5B | 8B | 8.4B | 12.2B | 12.2B |
| Token train | 2T | 94B | 94B | 94B | 15T | 380B | - | +0.1T |
| Winogrande (5) | 70.9 | 74.0 | 72.1 | 73.5 | 77.3 | 80.4 | 82.2 | 82.7 |
| ARC-C (25) | 55.4 | 50.9 | 52.6 | 55.6 | 57.9 | 64.4 | 65.1 | 62.3 |
| MMLU (5) | 51.3 | 58.6 | 58.7 | 60.5 | 65.3 | 69.5 | 69.0 | 70.1 |
| HellaSwag (10) | 73.0 | 75.0 | 73.2 | 76.1 | 81.8 | 83.0 | 85.2 | 85.3 |
| GSM8K (5) | 23.9 | 24.1 | 16.8 | 41.2 | 48.6 | 58.5 | 56.4 | 55.7 |
| TruthfulQA (0) | - | 42.9 | 38.2 | 42.9 | 45.0 | 47.6 | 49.8 | 48.3 |
| MBPP (0) | 29.0 | 28.2 | 30.7 | 32.4 | 42.3 | 43.8 | 42.6 | 47.9 |

(Bảng gốc có thêm Gemma 7B và Mistral 7B và hàng XLSum, HumanEval; bản text hơi lệch cột nên ở đây chỉ trích các cột đã đối chiếu chắc chắn. Cột "12B-FT" là Mistral NeMo 12B sau teacher correction.)

Điểm nhấn do tác giả nêu:
- MN-Minitron-8B vượt Llama 3.1 8B với 380B token (so với 15T), tức khoảng 40 lần ít hơn. Hai bài kiểm tra GSM8K (55.7% -> 58.5%) và HumanEval (23.8% -> 36.2%) cao hơn cả teacher Mistral NeMo 12B; tác giả cho là có ảnh hưởng của dữ liệu.
- Llama-3.1-Minitron-4B dùng 94B token, khoảng 150 lần ít hơn 15T của Llama 3.1 8B; cả hai biến thể hơn Minitron 4B đời trước (ở nhiều tác vụ).
- Width vs depth cho cùng số tham số: MMLU 60.5% vs 58.7%; **GSM8K 41.2% vs 16.8%** (suy luận toán bị ảnh hưởng lớn khi bỏ nửa số layer); khoảng cách thu hẹp sau instruction tuning.

**Model instruct (Table 2, trích):**

| Benchmark | Phi-2 2.7B | Gemma2 2B | Qwen2 1.5B | Minitron 4B | LL-Minitron 4B-Depth | 4B-Width | Llama 3.1 8B | MN-Minitron 8B |
|---|---|---|---|---|---|---|---|---|
| MT-Bench | 5.14 | 7.44 | 5.49 | 6.46 | 6.19 | 6.88 | 7.78 | 7.86 |
| MMLU (5) | 56.8 | 56.9 | 55.6 | 59.3 | 61.21 | 59.89 | 69.4* | 70.4 |
| GSM8K (0) | 19.9 | 52.2 | 27.2 | 65.1 | 71.11 | 79.76 | 83.8 | 87.1 |
| GPQA (0) | 28.8 | 25.9 | 28.1 | 29.5 | 32.59 | 30.36 | 30.4* | 31.5 |
| IFEval | 44.0 | 64.5 | 39.8 | 75.3 | 66.77 | 79.54 | 80.4* | 84.4 |
| BFCLv2 (live) | 38.7 | 40.2 | 39.9 | 53.1 | 55.89 | 55.0 | 44.3 | 67.6 |

Instruction tuning dùng NeMo-Aligner: SFT toán + code, SFT instruction, rồi hai vòng Reward-aware Preference Optimization (RPO). Tác giả: mô hình aligned của họ hơn các mô hình cùng cỡ ở hầu hết benchmark, ngoại trừ HumanEval và MBPP; Llama-3.1-Minitron-4B thua Gemma2 ở MT-Bench.

**Tốc độ (TensorRT-LLM, FP8, một H100 80GB, batch 32 cho 8B-12B và 64 cho 4B):** 4B-Depth nhanh nhất, trung bình 2.7x so với Llama 3.1 8B; 4B-Width 1.8x (tóm tắt Insights ghi 1.7x ở một chỗ và 1.8x ở chỗ khác); FP8 nhanh hơn BF16 1.4x.

## 5. Phân tích / ablation

- **Teacher correction (Hình 4-5):** distill từ teacher đã hiệu chỉnh cho loss validation thấp hơn rõ so với teacher gốc; tổng kết "Insights": hiệu chỉnh teacher trên dữ liệu distill cho **giảm hơn 6% LM validation loss**; không làm hỏng độ tối ưu của việc pruning; có thể làm song song.
- **Prune + distill vs lựa chọn khác (Hình 6, MN-Minitron-8B):** so (a) khởi tạo ngẫu nhiên + distillation, (b) prune ngẫu nhiên + distillation, (c) prune theo importance + LM loss thường, (d) prune theo importance + distillation. Kết luận: prune importance cho điểm khởi đầu tốt hơn nhiều so với ngẫu nhiên, và distillation hơn huấn luyện thường; hai lợi ích trực giao.
- **Width vs depth (Hình 7):** width có loss khởi đầu thấp hơn và luôn thắng depth với cùng số tham số trong suốt quá trình distill.
- **Chỉ số depth (Hình 8-9):** loại 16 layer rời rạc (chọn theo tác động nhỏ nhất lên LM loss) cho validation loss tốt hơn loại khối liên tiếp, **nhưng** accuracy Winogrande của mô hình loại 16 layer "tốt nhất theo importance" chỉ ngẫu nhiên 0.5, còn loại layer 16-31 liên tiếp được 0.595. Chênh lệch này vẫn còn sau distillation. Layer đầu và cuối là quan trọng nhất theo LM loss. (Đây là ví dụ rõ về: chỉ số proxy tốt không đồng nghĩa với downstream tốt.)
- **Giữ attention heads khi cắt width:** "retaining attention heads and pruning the other dimensions (MLP intermediate, embedding channels)" cho accuracy cao hơn (insight 3 cho width).

## 6. Giới hạn

*Tác giả nêu:*
- Teacher correction dùng ~100B token vẫn lớn; mong tối ưu bằng ít token hơn, LoRA hoặc chỉ tinh chỉnh LayerNorm (để dành cho công việc sau).
- Bỏ NAS; cấu hình chọn thủ công.
- Một số tác vụ của teacher giảm sau hiệu chỉnh (giả thuyết do dữ liệu).

*Chúng ta nhận xét:*
- Cắt depth chỉ thử một kiểu (nửa sau 16-31) và chỉ chọn theo Winogrande, một tác vụ nhỏ; có nguy cơ overfit vào benchmark chọn.
- Không có ablation tách riêng "teacher correction" với các lượng token khác nhau, nên khó biết cần bao nhiêu cho miền hẹp.
- Vẫn là transformer thuần, văn bản; không có mô hình hybrid hoặc VLM.
- 94B-380B token distill vẫn ngoài tầm với ngân sách nhỏ.

## 7. Liên hệ với các bài khác

- **39 Minitron:** bài tiền nhiệm; ở đây các best practice được dùng lại gần như nguyên.
- **32 Deeper layers (Gromov et al.), 31 ShortGPT, 33 Shortened LLaMA:** nguồn của ý "bỏ khối layer liên tiếp, thường ở nửa sau". Bài này bất đồng nhẹ với ShortGPT về BI (không dùng BI).
- **37 Repair changes which layers:** cùng tinh thần "chọn layer bằng proxy trước khi sửa chữa có thể sai".
- **38 Sheared LLaMA:** cũng chia nhỏ domain/dữ liệu khi train tiếp; ở đây giải quyết khác: hiệu chỉnh teacher.
- **41 Flextron, 42 Puzzle:** thay cấu hình thủ công bằng tìm kiếm tự động theo tốc độ/bộ nhớ.

## 8. Ứng dụng cho dự án (suy luận)

*Mục này là suy luận của chúng ta.*

- **Teacher correction rất liên quan.** Dự án có teacher 4B và dữ liệu OCR riêng; nếu teacher chưa từng được fine-tune trên đúng phân phối này (chữ, layout, tokenizer), logits của nó có thể là hướng dẫn kém. Nên fine-tune nhẹ teacher trên tập distill trước (hoặc song song), đo CER của teacher trước/sau.
- **Chọn layer bằng chỉ số downstream, không dùng PPL thuần:** với OCR, dùng CER/WER (hoặc exact-match theo trường) trên một tập validation để chọn khối layer liên tiếp cần bỏ. Cẩn thận overfit tập chọn.
- **Depth nhanh hơn, width chính xác hơn:** 4B-Depth nhanh gần 2.7x nhưng GSM8K giảm mạnh (16.8 so với 41.2). Với batch-1 trên GPU 8 GB, độ trễ phụ thuộc số layer (tuần tự) nhiều hơn băng thông MLP; độ trễ thật cần đo, vì báo cáo ở batch lớn (32-64) trên H100. Với batch 1 và mô hình nhỏ thì phần lớn thời gian là kernel launch/bộ nhớ, nên cắt layer có thể có lợi hơn paper gợi ý. Đây là giả thuyết.
- **Kiến trúc hybrid:** bài này không có GatedDeltaNet; không thể suy ra layer nào của hybrid (attention hay linear-attention) bỏ được. Cần tự kiểm tra (xem 44-45).
- Quan sát "cắt MLP hồi phục tệ hơn cắt layer" của dự án mâu thuẫn với bảng width vs depth của paper về độ chính xác. Điểm khác có thể nằm ở lượng token (94B so với ngân sách của dự án) và việc thí nghiệm ở đây còn giữ attention heads nguyên.
- Phần ViT: ít liên quan.

## 9. Thí nghiệm gợi ý

1. **Teacher correction mini:** fine-tune teacher 4B trên 1-5% dữ liệu OCR (lr = 1/5 đỉnh); so CER của student distill từ teacher gốc vs teacher đã chỉnh, cùng số bước.
2. **Layer selection proxy:** với decoder 0.8B, xếp hạng các khối layer liên tiếp (độ dài 2, 4, 6) bằng (a) PPL, (b) CER trên tập dev OCR; cắt top mỗi cách, so CER sau 0 và sau N bước distill.
3. **Parallel correction:** chạy distill từ teacher đang fine-tune song song và so với teacher đã chỉnh trước (tái hiện Hình 5 ở quy mô nhỏ).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Teacher correction | Fine-tune teacher ~100B token trên tập distill (lr 1/5 đỉnh) trước khi distill |
| Corrected teacher song song | Prune teacher gốc, distill từ teacher được chỉnh liên tục, ngang kết quả |
| Loss | Chỉ forward-KL trên logits, không có CE |
| Chọn layer | Khối liên tiếp làm Winogrande giảm ít nhất; không dùng PPL, không dùng BI |
| Width pruning | L2 (batch) + mean (seq), one-shot, giữ attention heads |
| Kết quả | Llama-3.1-Minitron-4B: 94B token; MN-Minitron-8B: 380B token |
| Tốc độ | 4B-Depth ~2.7x, 4B-Width ~1.8x (batch 32/64, H100, FP8) |

## 11. Kết luận ngắn

Bài này biến công thức Minitron thành quy trình thực tế khi không có dữ liệu gốc: hiệu chỉnh teacher trước, chọn layer theo downstream và theo khối liên tiếp, rồi distill chỉ bằng KL trên logits. Với dự án, teacher correction và chọn layer bằng chỉ số tác vụ (CER) là hai ý có thể áp dụng ngay; kết luận width vs depth cần kiểm chứng lại ở quy mô và kiến trúc của dự án.

Link gốc: https://arxiv.org/abs/2408.11796
