# A Survey on Model Compression for Large Language Models (2023)

**Tác giả:** Xunyu Zhu, Jian Li, Yong Liu, Can Ma, Weiping Wang (Institute of Information Engineering, CAS; UCAS; Renmin University of China)  
**Venue / arXiv:** arXiv:2308.07633 (bản v4, 30/07/2024, định dạng ACL-style). Bản text không ghi tên venue.  
**Nguồn:** https://arxiv.org/abs/2308.07633  
**Chủ đề:** Survey nén LLM: quantization, pruning (unstructured / structured / semi-structured), knowledge distillation (black-box / white-box), low-rank factorization; metric, benchmark, hướng phát triển. **Ghi chú đọc:** bài này dùng làm **bản đồ lĩnh vực**; ghi chú tập trung vào pruning, còn quantization / distillation / low-rank chỉ đặt vào vị trí của chúng.  
**Vị trí trong lộ trình:** Đọc sau 12 (Blalock, State of pruning) và 18 (2:4). Bài này không có ý tưởng mới mà là **mục lục có chú thích** cho giai đoạn LLM: nó nhắc đến SparseGPT (23), Wanda (24), LLM-Pruner (26), FLAP (28), SliceGPT (29), Shortened LLaMA (33), Sheared LLaMA (38) trước khi bạn đọc từng bài. Đọc nhanh bây giờ, quay lại tra khi cần.

> Đây là ghi chú học tập bằng tiếng Việt, không phải bản dịch. Phần "Ứng dụng cho dự án" là suy luận của chúng ta. Quan trọng: survey này ra từ 2023-2024 nên **không** phủ lớp phương pháp 2024-2025 mà lộ trình này nhắm tới (layer pruning như ShortGPT/SLEB, Minitron, Puzzle, mô hình lai Mamba/linear attention, VLM). Các số trong bảng của survey được lấy "solely from the original papers", chạy trên các mô hình và điều kiện khác nhau nên **không so sánh trực tiếp** được.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính (của survey)](#2-ý-tưởng-chính-của-survey)
3. [Bản đồ phương pháp](#3-bản-đồ-phương-pháp)
4. [Số liệu trong survey](#4-số-liệu-trong-survey)
5. [Remarks và nhận định của tác giả](#5-remarks-và-nhận-định-của-tác-giả)
6. [Giới hạn](#6-giới-hạn)
7. [Liên hệ với các bài khác](#7-liên-hệ-với-các-bài-khác)
8. [Ứng dụng cho dự án (suy luận)](#8-ứng-dụng-cho-dự-án-suy-luận)
9. [Thí nghiệm gợi ý](#9-thí-nghiệm-gợi-ý)
10. [Tóm tắt khái niệm](#10-tóm-tắt-khái-niệm)
11. [Kết luận ngắn](#11-kết-luận-ngắn)

---

## 1. Bối cảnh

LLM quá lớn để triển khai: ví dụ trong survey, GPT-175B cần tối thiểu 350 GB bộ nhớ ở FP16 và ít nhất 5 GPU A100 80 GB để chạy suy luận. Nén mô hình (model compression) là hướng chính. Survey tự nhận là survey đầu tiên chuyên về nén **LLM**; trước đó có survey về nén mạng nơ-ron nói chung hoặc chỉ nhắc nhẹ trong survey LM.

Điểm khác biệt khi nén LLM so với CNN (nền tảng của các bài 02-12): (i) không thể train lại đầy đủ vì hàng chục đến hàng trăm tỉ tham số, nên hầu hết phương pháp là **post-training** (một lần, với một ít dữ liệu hiệu chuẩn); (ii) có hiện tượng outlier trong activation làm quantization khó; (iii) đánh giá bằng perplexity, zero-shot, benchmark suy luận.

Giải thích thuật ngữ:
- **PTQ (post-training quantization):** lượng tử hoá sau khi train, không train lại; **QAT** train lại có mô phỏng lượng tử.
- **Calibration data (dữ liệu hiệu chuẩn):** một ít mẫu chạy qua mô hình để đo activation/Hessian dùng cho nén.
- **KV cache:** bộ nhớ lưu key/value của attention cho các token đã sinh.
- **Black-box / white-box KD:** teacher chỉ cho đầu ra (API) hay cho cả logit/tham số.

## 2. Ý tưởng chính (của survey)

Survey chia nén LLM thành bốn họ (Hình 1): **quantization** (giảm số bit), **pruning** (bỏ tham số/cấu trúc), **knowledge distillation** (học từ teacher lớn), **low-rank factorization** (phân tích $W\approx UV$). Phần pruning lại chia theo **độ mịn cấu trúc**: unstructured, structured, semi-structured (N:M). Tác giả nhấn mạnh quantization và KD có kiểu "task-based" (QAT, KD cần dữ liệu tác vụ) còn lại là "task-agnostic" (Hình 2), và gợi ý nên **kết hợp** (đặc biệt pruning cấu trúc + KD).

## 3. Bản đồ phương pháp

### 3.1 Metric và benchmark (Mục 2)

- Metric kích thước/tính toán: model size (số tham số), FLOPs, **MFU** (Mean FLOPS Utilization: FLOPS thực dùng / FLOPS lý thuyết của thiết bị), inference time (latency), speedup ratio (thời gian gốc / thời gian đã nén), compression ratio (kích thước gốc / sau nén).
- Benchmark: perplexity trên WikiText-2, C4, PTB; zero-shot LAMBADA, PIQA, OpenBookQA; suy luận GSM8K, CommonsenseQA, StrategyQA; BIG-Bench (BBH, hơn 200 tác vụ); bộ chỉ lệnh chưa thấy (Vicuna-Instructions, User-Oriented-Instructions); khung EleutherAI LM Harness (hơn 60 benchmark học thuật).

### 3.2 Pruning (Mục 4), phần trọng tâm

Survey định nghĩa: **structured** xoá cả neuron/head/layer, giữ nguyên hình dạng chung; **unstructured** xoá từng tham số, tạo cấu trúc thưa bất quy tắc; **semi-structured** xoá theo mẫu cố định (N:M) rồi gọi là "fine-grained structured".

**Unstructured (4.1):** thường không cần retrain vì giữ chất lượng tốt, nhưng cần xử lý đặc biệt để tăng tốc.
- **SparseGPT** (→ 23): prune một lần không retrain; coi pruning như bài toán hồi quy thưa lớn giải bằng solver xấp xỉ; đạt trên 50% sparsity ở OPT-175B, BLOOM-176B với perplexity tăng ít.
- **Wanda** (→ 24): giữ trọng số theo |w| nhân norm của activation đầu vào; không cần cập nhật trọng số hay retrain.
- **SAMSP:** dùng Hessian để đo độ nhạy từng ma trận và phân bổ sparsity động.
- **DSnoT:** lặp "prune-and-grow" để giảm sai số tái tạo giữa dense và sparse, nhất là sparsity cao.
- **Flash-LLM:** hỗ trợ phần cứng: nạp ma trận ở dạng thưa từ global memory, dựng lại dạng dense trong bộ đệm trên chip để dùng tensor core.

**Structured (4.2):** không phụ thuộc phần cứng đặc biệt nhưng dễ mất chất lượng hơn, thường phải fine-tune hiệu quả (LoRA). Survey chia theo tiêu chí chọn:
- **Loss-based** (Molchanov 2019, → 07): **LLM-Pruner** (→ 26) phát hiện cấu trúc phụ thuộc rồi chọn nhóm bằng gradient, task-agnostic; **Shortened LLaMA** (→ 33) cắt cả block Transformer (depth pruning) dựa trên loss và đạo hàm bậc hai; cả hai dùng LoRA phục hồi.
- **Magnitude-based:** **FLAP** (→ 28) dùng "fluctuation metric" đo độ biến thiên của từng đặc trưng đầu vào để chọn cột bỏ, tìm cấu trúc thích nghi toàn cục, bù bias nên không cần fine-tune; **SliceGPT** (→ 29) dùng tính bất biến tính toán của Transformer và PCA ở mỗi lớp để chiếu rồi xoá hàng/cột.
- **Regularization-based:** **Sheared LLaMA** (→ 38) dùng cặp nhân tử Lagrange trên mask để ràng buộc trực tiếp hình dạng mô hình, kèm dynamic batch loading theo tốc độ giảm loss của từng miền dữ liệu.

**Semi-structured (4.3):** N:M (mỗi M phần tử liên tiếp giữ N khác 0). **E-Sparse** dùng entropy thông tin làm tiêu chí và shuffle để giảm ảnh hưởng. SparseGPT chia khối M trọng số, chọn N để cắt theo sai số tái tạo (Hessian); Wanda chia nhóm M, giữ N trọng số điểm cao nhất. Survey nhắc A100 chỉ hỗ trợ tăng tốc tỉ lệ 2:4 (→ 18).

### 3.3 Quantization (Mục 3) — chỉ đặt vị trí

- **QAT:** LLM-QAT (distill từ dữ liệu do chính LLM sinh), BitDistiller (QAT + self-distillation), OneBit (biểu diễn 1-bit); kết hợp PEFT: QLoRA, PEQA, LoftQ, L4Q.
- **PTQ weight-only:** GPTQ (dựa trên OBQ của bài OBC, → 22; cập nhật trọng số bằng nghịch đảo Hessian), AWQ (giữ 1% trọng số quan trọng + scale theo kênh), QuIP (LDL + ma trận trực giao ngẫu nhiên, 2-bit), SpQR, SqueezeLLM (k-means), OWQ, LUT-GEMM.
- **PTQ weight-activation:** ZeroQuant, LLM.int8() (giữ chiều outlier ở độ chính xác cao), SmoothQuant (scale theo kênh để làm mịn outlier), RPTQ, OliVe, OS+, LLM-FP4, OmniQuant.
- **KV cache:** KVQuant, KIVI (key theo kênh, value theo token, 2-bit không fine-tune), WKVQuant.
- Nhận xét trong survey: weight-only đạt bit thấp hơn nhưng phải dequantize trước khi nhân nên không hưởng tăng tốc thấp-bit của phần cứng; activation nhạy hơn weight.

### 3.4 Knowledge distillation (Mục 5) — chỉ đặt vị trí

- **Black-box KD:** teacher (ChatGPT, GPT-4) chỉ cho văn bản; chia theo năng lực nổi lên cần chuyển: Chain-of-Thought (Distilling Step-by-Step, MT-CoT, Fine-tune-CoT, SCOTT, PaD, Socratic CoT...), In-Context Learning (ICL distillation, AICD), Instruction Following (Lion, LaMini-LM, Self-Instruct, Selective Reflection-Tuning).
- **White-box KD:** có phân phối/tham số teacher. MiniLLM (reverse KL, tránh student ước lượng quá cao vùng xác suất thấp của teacher), GKD (student học trên đầu ra tự sinh có phản hồi của teacher, loss linh hoạt), TED (lọc theo tác vụ để căn chỉnh biểu diễn ẩn từng lớp). Survey ghi nhận white-box tới nay bị hạn chế vì LLM mã nguồn mở yếu hơn đóng.

### 3.5 Low-rank factorization (Mục 6) — chỉ đặt vị trí

$W\approx UV$ với $U\in\mathbb R^{m\times k}$, $V\in\mathbb R^{k\times n}$, $k\ll m,n$. LPLR (phân rã hạng thấp + độ chính xác thấp ngẫu nhiên), ASVD (scale trọng số theo phân phối activation, phân bổ hạng khác nhau theo lớp), LASER/LSAER (giảm có chọn lọc hạng của thành phần bậc cao ở một số lớp; paper trích dẫn cho thấy cải thiện xử lý dữ liệu hiếm và độ bền với diễn đạt lại).

## 4. Số liệu trong survey

Survey nhắc thẳng: kết quả "chỉ lấy từ paper gốc". Sai khác perplexity = (perplexity mô hình nén) - (perplexity gốc), WikiText-2, nên **không** cùng mô hình, không cùng điều kiện.

**Bảng 2 (pruning):**

| Loại | Phương pháp | Mô hình | Δ perplexity (WikiText-2) | Tỉ lệ nén | Tăng tốc |
|---|---|---|---|---|---|
| Unstructured | SparseGPT | OPT-175B | -0.14 | 50% | - |
| | Wanda | LLaMA-65B | 1.01 | 50% | - |
| | SAMSP | LLaMA2-13B | 0.63 | 50% | - |
| | DSnoT | LLaMA-65B | 2.08e4 | 90% | - |
| Structured | LLM-Pruner | LLaMA-13B | 3.6 | 20% | - |
| | Shortened LLaMA | LLaMA-7B | 10.5 | 35% | - |
| | FLAP | LLaMA-65B | 7.09 | 50% | - |
| | SliceGPT | LLaMA2-70B | 1.73 | 30% | 1.87x |
| Semi-structured | E-Sparse | LLaMA-65B | 2.13 | 2:4 | 1.53x |
| | SparseGPT | OPT-175B | 0.39 | 2:4 | 2x |
| | Wanda | LLaMA-65B | 2.69 | 2:4 | 1.24x |

Đọc đúng bảng: Δ perplexity âm hoặc nhỏ ở OPT-175B không có nghĩa SparseGPT "tốt hơn" LLM-Pruner; mô hình khác nhau (175B so với 7-13B), tỉ lệ nén khác nhau, và structured luôn khó hơn unstructured ở cùng tỉ lệ. Con số DSnoT 2.08e4 ở 90% cho thấy sparsity cực cao làm mô hình hỏng (theo bảng).

**Bảng 1 (quantization, trích để cảm nhận thang):** GPTQ OPT-175B 3-bit, Δ perplexity 0.34 (WikiText-2), tăng tốc 3.24x; AWQ LLaMA2-70B 3-bit, 0.42, 3.2x; QuIP LLaMA2-70B 2-bit, 3.007; OneBit LLaMA-13B 1-bit, 4.09; SmoothQuant OPT-175B W8A8, 0.18, 1.56x; KVQuant LLaMA-65B KV 2-bit, 0.19, 1.4x. Gợi ý của chúng ta: ở mức tăng-tốc/chất-lượng cho LLM, quantization thường cho mức giảm bộ nhớ lớn hơn với mất chất lượng nhỏ hơn pruning cấu trúc (đây là suy luận từ việc đọc hai bảng, chưa phải kết luận chính thức của survey).

## 5. Remarks và nhận định của tác giả

- **Remark 1:** QAT tốn công; weight-only lượng tử hoá được bit thấp hơn nhưng có overhead dequantize.
- **Remark 2:** pruning cấu trúc giảm kích thước nhưng làm giảm chất lượng; hướng mới là **kết hợp KD với structured pruning** (rất sát dự án).
- **Remark 3:** dynamic pruning (cắt khác nhau theo tác vụ/đầu vào) để cân bằng hiệu năng và hiệu quả.
- **Remark 4:** chất lượng **dữ liệu hiệu chuẩn** rất quan trọng cho PTQ và pruning; nhắc nghiên cứu của Williams & Aletras (2023) cho thấy kết quả downstream thay đổi đáng kể theo dữ liệu hiệu chuẩn chọn.
- **Remark 5-7:** hạn chế của black-box KD (đầu ra teacher không phủ hết tình huống), white-box hạn chế vì LLM mở yếu hơn, và white-box cần hiểu cấu trúc nội bộ.
- **Hướng tương lai (Mục 7):** phương pháp tiên tiến hơn (còn khoảng cách với mô hình chưa nén), đưa kỹ thuật nén cổ điển (lottery ticket, parameter sharing) sang LLM, triển khai suy luận (arithmetic intensity, bộ nhớ, throughput, mô hình Roofline), tác động của scaling law (đánh đổi kích thước-hiệu năng), AutoML/NAS cho nén, khả diễn giải.

## 6. Giới hạn

**Tác giả nêu (ngầm/nêu):** các phương pháp nén còn khoảng cách hiệu năng so với mô hình gốc; thiết kế còn phụ thuộc thủ công; thiếu khả diễn giải.

**Chúng ta nhận xét:**
- Tập trung LLM văn bản dạng Transformer thuần; **không** có multimodal/VLM, **không** có kiến trúc lai (linear attention, SSM), **không** có pruning layer/depth hiện đại (ShortGPT, SLEB, LaCo...) và không có pruning + distillation quy mô lớn kiểu Minitron/Puzzle. Với dự án, nó chỉ là nền.
- Bảng chỉ tổng hợp từ paper gốc nên thiếu so sánh công bằng; Δ perplexity không đo đúng thứ dự án cần (độ chính xác OCR, CER/WER).
- Phân loại pruning là "structured/unstructured/semi" theo **độ mịn**, còn trục quan trọng hơn cho kỹ sư triển khai (có cần retrain không, có cần dữ liệu không, tốc độ thật trên GPU) bị trộn lẫn.
- Gần như không có số đo latency thật ngoài vài chỗ tăng tốc ghi trong bảng.

## 7. Liên hệ với các bài khác

| Phần của survey | Bài tương ứng trong lộ trình |
|---|---|
| Magnitude pruning, retrain | 02, 03 |
| Lottery ticket | 08, 09 |
| Loss-based importance (Taylor) | 07 |
| N:M / 2:4, Ampere | 18 |
| GPTQ ← OBQ | 22 (OBC) |
| SparseGPT | 23 |
| Wanda | 24 |
| LLM-Pruner | 26 |
| FLAP | 28 |
| SliceGPT | 29 |
| Shortened LLaMA | 33 |
| Sheared LLaMA (regularization + dynamic batch loading) | 38 |
| Pruning + KD (Remark 2) | 39-45 (Minitron, Llama-3.1-Minitron, Puzzle...) |
| White-box KD (MiniLLM, GKD) | 43 và các bài distillation khác |

## 8. Ứng dụng cho dự án (suy luận)

- **Dùng làm từ vựng và checklist.** Dự án nằm ở ô "structured pruning + white-box KD" (Remark 2). Survey xác nhận rằng đây là hướng được tác giả gợi ý nhưng không cung cấp công thức cụ thể.
- **Khung đánh giá:** metric "Inference time / Speedup ratio / MFU" trong Mục 2.1 nhắc rằng kích thước và FLOPs không bằng latency; MFU thấp ở batch 1 là lý do cắt neuron nhỏ ít giúp (suy luận; xem thêm 19).
- **Calibration data (Remark 4):** với pruning dựa trên hiệu chuẩn (SparseGPT, Wanda, FLAP, SliceGPT), dùng ảnh/văn bản **đúng miền OCR** (tài liệu, bảng biểu, chữ tiếng Việt nếu cần) thay vì WikiText/C4. Đây là chọn lựa dễ bị bỏ qua và tác động lớn.
- **Thứ tự nén:** survey không nói rõ; bài 18 khuyên prune rồi mới quantize. Với dự án nhắm GPU 8 GB, quantization (INT8/FP8/4-bit) có thể là cách rẻ nhất để hạ bộ nhớ, bổ sung chứ không thay cho pruning cấu trúc xuống ~0.3B / 0.1B.
- **Low-rank** là ứng viên nhìn thẳng với D_lin (phần hư hại có thể sửa bằng ánh xạ tuyến tính tối ưu): nếu một khối bị cắt, thay bằng $W\approx UV$ nhỏ là cách rẻ để giữ phần tuyến tính. Chưa có trong survey; ý này sẽ gặp lại ở 36 (ReplaceMe).

## 9. Thí nghiệm gợi ý

1. Lập bảng 2 chiều tự làm cho dự án: hàng = họ nén (quantization, pruning cấu trúc theo tầng, pruning theo neuron/channel, 2:4, low-rank), cột = (cần dữ liệu?, cần retrain/distill?, giảm bộ nhớ, giảm latency batch-1 đo thật). Điền bằng đo thực nghiệm nhỏ thay vì số của paper.
2. Thử một trong các phương pháp "post-training, không retrain" (ví dụ Wanda hoặc SliceGPT khi đọc tới) với hai loại calibration data (văn bản chung vs mẫu OCR) để kiểm chứng Remark 4 trên mô hình của dự án.
3. Đo MFU (FLOPS thực / FLOPS lý thuyết) của decoder 0.8B ở batch 1 để biết ngưỡng nào thì giảm FLOPs không còn ý nghĩa.

## 10. Tóm tắt khái niệm

| Khái niệm | Nội dung |
|---|---|
| Bốn họ nén | Quantization, Pruning, Knowledge Distillation, Low-rank factorization |
| Pruning theo độ mịn | Unstructured (từng trọng số), Structured (neuron/head/layer), Semi-structured (N:M) |
| Pruning cấu trúc theo tiêu chí | Loss-based, Magnitude-based, Regularization-based |
| PTQ vs QAT | Không train lại vs train lại |
| Black-box vs white-box KD | Chỉ đầu ra teacher vs có phân phối/tham số teacher |
| MFU | FLOPS thực dùng / FLOPS đỉnh lý thuyết của thiết bị |
| Speedup ratio | Thời gian chạy chưa nén / đã nén |
| Δ perplexity (trong bảng) | Perplexity nén trừ perplexity gốc, WikiText-2 |
| Remark 2 | Kết hợp KD với structured pruning |
| Remark 4 | Dữ liệu hiệu chuẩn ảnh hưởng mạnh đến chất lượng nén |

## 11. Kết luận ngắn

Survey cho một bản đồ gọn của nén LLM đến khoảng giữa 2024: bốn họ, ba mức độ mịn của pruning, các tên phương pháp chủ chốt và hướng đi (đáng nhớ nhất: kết hợp structured pruning với distillation, và vai trò của dữ liệu hiệu chuẩn). Nó là điểm xuất phát để tra cứu, không phải nguồn công thức hay so sánh công bằng; các bài cụ thể trong lộ trình mới là nơi cần đọc kỹ.

Link gốc: https://arxiv.org/abs/2308.07633
