# LightOnOCR: A 1B End-to-End Multilingual Vision-Language Model for State-of-the-Art OCR (2026)

**Tác giả:** Said Taghadouini, Adrien Cavaillès, Baptiste Aubertin (LightOn).  
**Venue / arXiv:** arXiv:2601.14251 (ngày đầu 20/01/2026; bản v2 đề ngày 30/06/2026, đề cập mô hình **LightOnOCR-2-1B**). Paper không nêu venue.  
**Nguồn:** https://arxiv.org/abs/2601.14251  
**Chủ đề:** VLM OCR 1B end-to-end; chưng cất dữ liệu từ teacher 235B; chuẩn hoá nhãn; pipeline nvpdftex; RLVR (GRPO) với phần thưởng unit-test; dự đoán bbox ảnh; checkpoint averaging và task-arithmetic merging; **vocabulary pruning**; hiệu năng và thông lượng.  
**Vị trí trong lộ trình:** Đây là **tiền thân trực tiếp của mô hình của bạn** (LightOnOCR-3-0.8B). Đọc trước: 37 (Qwen2-VL), 41 (GOT), 42 (olmOCR, benchmark và RLVR), 29 (ViT). Sau bài này: 57 (FastV, cắt visual token), 24 (Gated DeltaNet, decoder lai của bản 3), và các bài chưng cất/cắt tỉa của lộ trình pruning.

> Ghi chú học tập bằng tiếng Việt, **không phải bản dịch**. Phần "Ứng dụng cho dự án" là suy luận của chúng tôi. Số đánh dấu "tự tính" do chúng tôi tính. **Lưu ý quan trọng:** paper này mô tả **LightOnOCR-2-1B** (decoder Qwen3 attention chuẩn), **không** phải LightOnOCR-3-0.8B với decoder lai GatedDeltaNet; paper cũng **không có chưng cất logit và không có cắt tỉa lớp/neuron**, "distillation" ở đây là chưng cất dữ liệu (nhãn văn bản do teacher sinh).

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
   - 3.1 Kiến trúc · 3.2 Dữ liệu và chưng cất từ teacher · 3.3 Chuẩn hoá · 3.4 nvpdftex · 3.5 Huấn luyện · 3.6 RLVR · 3.7 Averaging và merging · 3.8 Vocabulary pruning
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

OCR chuyển từ Tesseract/CRNN/TrOCR (nhận dạng từng dòng) sang **pipeline nhiều giai đoạn** (PaddleOCR, MinerU: layout -> phát hiện chữ -> nhận dạng -> bảng -> thứ tự đọc), rồi sang **VLM end-to-end** (Nougat, olmOCR, dots.ocr, PaddleOCR-VL, Chandra...). Pipeline khó thích nghi: muốn cải thiện trên một phân phối tài liệu mới phải gán nhãn cho từng bước trung gian. VLM end-to-end học thẳng pixel -> văn bản, nhưng các hệ thống tốt nhất lớn (8-9B) và chậm. LightOnOCR muốn một **mô hình nhỏ (1B), nhanh, end-to-end** mà vẫn đạt SOTA.

*Thuật ngữ cần biết:*
- **VLM (vision-language model):** vision encoder + projector + LLM decoder; ở đây dùng để "đọc" trang.
- **Chưng cất dữ liệu (data distillation / pseudo-labeling):** model lớn (teacher) sinh nhãn văn bản cho ảnh, model nhỏ (student) học trên nhãn đó bằng cross-entropy thường. Khác với KD logit (khớp phân phối xác suất từng token).
- **RLVR (RL with Verifiable Rewards):** học tăng cường mà phần thưởng tính bằng kiểm tra tự động xác định (ví dụ unit test pass/fail), không cần mô hình phần thưởng.
- **GRPO:** thuật toán RL kiểu PPO không cần critic; sinh nhiều mẫu cho mỗi prompt và dùng chênh lệch so với trung bình nhóm làm lợi thế (advantage).
- **Task arithmetic / weight averaging:** trộn mô hình trong không gian tham số.

## 2. Ý tưởng chính

LightOnOCR-2-1B giữ nguyên kiến trúc của bản 1 (ViT bản địa độ phân giải gốc + projector MLP gộp 2×2 + decoder Qwen3), và cải thiện chủ yếu bằng **dữ liệu và công thức huấn luyện**: (1) teacher mạnh hơn (Qwen3-VL-235B-A22B thay Qwen2-VL-72B), 43M trang (bản 1 là 17M) và pipeline chuẩn hoá nhãn rất kỹ; (2) độ phân giải huấn luyện cao hơn (cạnh dài 1540 px), augmentation và trang trống tường minh; (3) giai đoạn **RLVR** nhắm đúng các lỗi SFT khó sửa (lặp vô hạn, công thức hỏng, đúng thứ tự cột); (4) thêm khả năng **dự đoán bbox ảnh** mà không làm tụt OCR, nhờ đưa toạ độ vào pretraining (resume) rồi RLVR; (5) trộn trọng số để cân bằng OCR và bbox. Kết quả: **83.2 ± 0.9** trên OlmOCR-Bench (không tính header/footer), vượt các model 8-9B, và nhanh hơn (5,71 trang/giây trên H100).

## 3. Phương pháp chi tiết

### 3.1 Kiến trúc

```
ảnh trang (cạnh dài <= 1540 px, giữ tỉ lệ)
  -> Vision Encoder: ViT bản địa, khởi tạo từ Mistral-Small-3.1
  -> Gộp không gian 2x2 patch (giảm 4x số visual token)
  -> Projector: MLP 2 lớp + GELU (khởi tạo ngẫu nhiên, học từ đầu)
  -> [một khối visual token liền mạch] + token văn bản
  -> Decoder: Qwen3 (khởi tạo từ pretrained)
  -> Markdown (+ LaTeX trong $...$, bảng HTML, ![image](image_N.png) cho ảnh)
```

Đặc điểm:
- **Không có task prompt lúc suy luận:** hành vi OCR "nằm trong trọng số".
- Bỏ token `image-break` và `image-end`: decoder thấy **một khối visual token liên tục** rồi tới văn bản (giao diện đơn giản, đồng nhất giữa các tập dữ liệu).
- Khởi tạo từ thành phần pretrained giúp kế thừa biểu diễn thị giác và khả năng đa ngôn ngữ.
- Paper **không nêu** số tham số từng thành phần, số block ViT, kích thước patch, số lớp decoder.

**Suy ra từ bảng vocabulary pruning (tự tính, cần kiểm chứng):** Bảng 4 cho tổng 1005,6M tham số ở vocab 151.936 và 866,8M ở 16k. Chênh 138,8M ≈ (151.936 − 16.000) × 1024 = 139,2M. Khớp tốt nếu **d_model = 1024 và embedding được tie với LM head** (một ma trận dùng chung). Khi đó ma trận embedding gốc có 151.936 × 1024 ≈ **155,6M tham số ≈ 15,5% toàn model**. d=1024 phù hợp với decoder cỡ Qwen3-0.6B (~0,44B không tính embedding theo trí nhớ của chúng tôi, không có trong paper); phần còn lại (~0,41B) sẽ là ViT + projector. Đây là suy đoán để định hướng, đừng trích như số của paper.

**Số visual token mỗi trang (tự tính, giả định):** ảnh A4 (tỉ lệ ~1:1,414) cạnh dài 1540 px -> cạnh ngắn ≈ 1089 px. Nếu patch 14: 110×78 = 8.580 patch trước gộp, /4 ≈ **2.145 visual token**. Nếu patch 16: 96×68 = 6.528, /4 ≈ **1.632**. Paper không cho patch size nên ghi khoảng ~1,6k-2,1k token/trang. Đối chiếu với giới hạn huấn luyện (chuỗi tối đa 6.144 token, completion tối đa 3.100 token): còn khoảng 3.000 token cho prompt + visual, vừa đủ.

### 3.2 Dữ liệu và chưng cất từ teacher

- **Bản 1 (LightOnOCR-1B):** huấn luyện có giám sát trên PDF Association (PDFA), nhãn từ **Qwen2-VL-72B-Instruct**, ảnh tối đa 1024 px, 17M trang.
- **Bản 2:** 43M trang, teacher **Qwen3-VL-235B-A22B-Instruct** (MoE 235B tham số tổng, ~22B kích hoạt), phủ nhiều hơn scan, tài liệu tiếng Pháp, PDF khoa học. Nhãn là Markdown + LaTeX do teacher sinh từ ảnh trang render. Teacher mới cho "toán đúng hơn, ít artefact định dạng hơn" (theo tác giả).
- Thêm **crop vùng** (đoạn, header, abstract, snippet) gán nhãn bằng **GPT-4o**, **trang trống** tường minh (mục tiêu nhất quán = chuỗi rỗng, giảm lặp/ảo giác), dữ liệu **TeX arXiv** (3.4), và một số bộ OCR công khai.
- Teacher đôi khi tự xuất bbox ảnh dù không được yêu cầu, độ chính xác cao khi kiểm tra tay. Tác giả **gỡ toạ độ khỏi nhãn OCR chính** (giữ mục tiêu thuần phiên âm) nhưng **giữ lại làm tín hiệu riêng** cho phiên bản bbox, đổi sang quy ước chuẩn hoá [0,1000].
- Công bố: tập con PDFA đã gán nhãn `lightonai/LightOnOCR-mix-0126`, và `lightonai/LightOnOCR-bbox-mix-0126`.
- Paper **không cho tỷ lệ pha trộn giữa các nguồn** và không có ablation theo nguồn.

**Đây có phải "distillation từ teacher lớn" không?** Có, nhưng ở dạng **sequence-level / hard-label**: student chỉ thấy chuỗi token cuối của teacher, không thấy logit. Hệ quả: nhãn teacher nhiễu (artefact, lặp, watermark) trực tiếp làm tăng entropy của mục tiêu, nên tác giả đầu tư mạnh vào bước chuẩn hoá 3.3. Teacher có hệ số ~235B/1B ≈ 235× lớn hơn student (tự tính; tính cả tổng tham số MoE).

### 3.3 Pipeline chuẩn hoá và làm sạch nhãn

Nguồn hỗn hợp (PDFA, scan, bộ mở, arXiv) và nhiều teacher gây ra sai khác "bề mặt": code fence Markdown thừa, watermark, placeholder ảnh khác nhau, câu "this page is empty", lẫn LaTeX environment/HTML. Chúng không hại người đọc nhưng làm tăng entropy mục tiêu, hại dedup và độ ổn định. Pipeline gồm các biến đổi nhẹ, gần như xác định, **trước khi hash và lọc**:

1. Làm sạch văn bản (bỏ ticks/code block thừa, chuẩn hoá khoảng trắng); xoá watermark lặp trước để không làm nhiễu dedup.
2. Đồng nhất trường hợp đặc biệt: ảnh toàn trang -> một placeholder chuẩn; trang trống -> chuỗi rỗng.
3. Lọc lặp (loop/repetition) và dedup trên văn bản đã chuẩn hoá. Pool riêng cho trang trống, **tái chèn theo tỷ lệ chọn** khi huấn luyện.
4. Chuyển đổi và kiểm định LaTeX: lệnh LaTeX chỉ nằm trong math span; header/section -> Markdown; **bảng chuẩn hoá thành HTML** (giảm mơ hồ); công thức phải **tương thích KaTeX**. Bước này xuất metadata (success/partial/timeout, tham chiếu chưa giải quyết, thiếu đánh số hình, KaTeX OK) để lọc có thể tái lập.

Lọc lúc train: bỏ mẫu thiếu tham chiếu, hình không số, không tương thích KaTeX, completion dài (giới hạn 3.100 token).

### 3.4 Pipeline nvpdftex (nhãn pixel-aligned từ nguồn TeX)

Trước đây dùng kiểu Nougat để ghép ảnh với markup, nhưng gặp lỗi chuyển đổi, thứ tự đọc, lệch căn. Bản 2 dùng **nvpdftex** của NVIDIA (hook vào engine pdfLaTeX) để biên dịch nguồn TeX và xuất: (i) ảnh PNG từng trang, (ii) mục tiêu văn bản (Markdown/HTML) từng vùng, (iii) bbox chính xác theo pixel kèm lớp ngữ nghĩa (header/footer, caption, bảng, công thức, ảnh), (iv) metadata trang. Dùng cho (a) giám sát khoa học và (b) tạo bbox ảnh arXiv cho RLVR và cho một nửa benchmark bbox (bbox arXiv **không** nằm trong pretraining chính).

### 3.5 Huấn luyện (pretraining có giám sát)

| Hạng mục | Giá trị trong paper |
|---|---|
| Mục tiêu | next-token prediction, **loss chỉ trên token assistant**, bỏ system prompt/special token và **token ảnh** |
| Ảnh | render 200 DPI (crop dùng ảnh sẵn), resize cạnh dài tối đa **1540 px** |
| Augmentation | bitmap corruption, erosion/dilation, shear nhỏ, shift-scale-rotate, xoay 90°, grid distortion nhẹ; xác suất ≥ 1 aug mỗi trang = **0,22** (trang trống: chế độ mạnh hơn) |
| Tối ưu | AdamW, **không weight decay**, lr đỉnh **1e-4**, cosine, warmup 100 bước |
| Batch / độ dài | global batch **384 mẫu**, chuỗi tối đa **6.144 token** (completion ≤ 3.100) |
| Phần cứng | DDP trên **96 × H100 80GB**, bf16, FlashAttention-2 |
| Số bước / epoch / thời gian | không nêu |

### 3.6 RLVR với GRPO

Xuất phát từ checkpoint pretrain (`-base` cho OCR, `-bbox-base` cho bbox). **Thiết lập:** GRPO, 1 epoch, AdamW lr **4e-5**, hệ số KL **β = 0,01**, phần thưởng theo nhóm có chuẩn hoá, importance sampling mức token, **28 rollout/prompt (OCR), 14 (bbox)**; TRL + vLLM để sinh rollout; RL chạy trên H200.

**Phần thưởng OCR** (kế thừa olmOCR-2 và mở rộng):

```
reward(output, page_tests):
    r = fraction_of_unit_tests_passed(output, page_tests)   # kiểu OlmOCR-Bench, trên tài liệu tổng hợp
    r -= penalty_repetition(zlib_compression_ratio(output)) # heuristic nén: lặp = entropy thấp
    r -= penalty_if_no_proper_EOS(output)
    r += reward_if_all_math_spans_render_in_KaTeX(output)
    r -= penalty(HTML_tags_in_math, markdown_italics_for_variables, unbalanced_LaTeX)
    # bỏ phần thưởng "frontmatter"; header/footer: ĐẢO test absence -> thưởng khi CÓ mặt
    return r
```

**GRPO (chuẩn, ref. DeepSeekMath, không phải sáng tạo của paper):** với mỗi prompt lấy \(G\) mẫu, lợi thế \(\hat A_i = \dfrac{r_i - \text{mean}(r_{1..G})}{\text{std}(r_{1..G})}\); tối ưu mục tiêu kiểu PPO kèm phạt \(\beta\, D_{KL}(\pi_\theta\Vert\pi_{ref})\).

**Phần thưởng bbox:** định dạng đầu ra `![image](image_N.png)x1,y1,x2,y2` với toạ độ chuẩn hoá [0,1000]; thưởng:

$$
\boxed{R_{bbox}=\frac{1}{|I_\cap|}\sum_{i\in I_\cap}\mathrm{IoU}\big(B^i_{pred},B^i_{gt}\big)\cdot\frac{|I_\cap|}{\max(|I_{gt}|,|I_{pred}|)}},\quad I_\cap=I_{gt}\cap I_{pred}
$$

Tức IoU trung bình trên các ảnh khớp ID, nhân hệ số phạt khi **thiếu hoặc bịa** ảnh. Vì pretraining đã dạy model xuất placeholder ảnh, học bbox chỉ thêm "ở đâu" chứ không phải "có hay không". Trước RL có giai đoạn **resume pretraining có toạ độ** làm "cold start".

### 3.7 Model averaging và task-arithmetic merging

- **Checkpoint souping:** lấy trung bình trọng số **5 checkpoint cuối** của pretraining -> `LightOnOCR-2-1B-base` (paper nói luôn tốt hơn bất kỳ checkpoint đơn lẻ; không đưa bảng).
- RLVR trên base -> `LightOnOCR-2-1B` (OCR tốt nhất).
- **Task arithmetic:** \(\theta_{merge}=\theta_{base}+\alpha(\theta_{rl}-\theta_{base})\).
  - `ocr-soup`: trộn `LightOnOCR-2-1B` vào base, **α = 0,4**.
  - `bbox-soup`: trộn `ocr-soup` vào `bbox` (dùng bbox làm gốc) với **α = 0,1**: \(\theta=\theta_{bbox}+\alpha(\theta_{ocr}-\theta_{bbox})\).
- Quét α (hình 2): OCR tăng theo α; bbox giảm sau α > 0,4 và **sụp ở α ≥ 0,6**; điểm cân bằng α ≈ 0,1 cho bbox IoU = 0,677 và OCR = 80,88%. Không tốn huấn luyện thêm, suy luận không đổi.

### 3.8 Vocabulary pruning (phụ lục A, thực hiện trên LightOnOCR-**1**)

Qwen3 có vocab 151.936 token đa ngôn ngữ, phần lớn không dùng cho OCR tiếng Anh/Pháp. Tác giả **cắt vocab theo tần suất** trên tài liệu Anh/Pháp xuống **51k, 32k, 16k**, giữ nguyên tính toàn vẹn tokenizer bằng **lan truyền tần suất sub-token đệ quy** (một token hiếm nhưng là thành phần để ghép token thường gặp vẫn phải được giữ). Phát hành `LightOnOCR-0.9B-32k-1025` và `LightOnOCR-0.9B-16k-1025`.

Pseudo-code (suy từ mô tả, chi tiết cài đặt paper không nêu):

```
freq = đếm tần suất token trên tập tài liệu Anh/Pháp (qua tokenizer gốc)
for token in sorted(vocab, by merge-order, descending):     # BPE: token ghép từ sub-token
    freq[sub_tokens(token)] += freq[token]                  # lan truyền để giữ tính ghép được
keep = top_K(freq)  (K = 51k / 32k / 16k)  ∪ special tokens
E_new = E[keep]   (và LM head nếu không tie);  tokenizer_new = tokenizer.restrict(keep)
```

Paper **không nói** có tinh chỉnh lại sau khi cắt hay không, và không nói cách xử lý các merge rule.

## 4. Thí nghiệm và kết quả chính

**Thiết lập đánh giá:** OlmOCR-Bench (1.403 trang), **không tính nhóm Headers/Footers** (vì thưởng việc bỏ nội dung nhìn thấy; đầu ra rỗng đạt điểm tuyệt đối ở nhóm đó), không có thủ thuật lúc test (retry, xoay). Suy luận: T = 0,2, top_p = 0,9, top_k = 0, **3 lần sinh độc lập mỗi trang**, độ phân giải tối đa 1540.

**Bảng 1 (OlmOCR-Bench; cột: ArXiv, Old Scans Math, Tables, Old Scans, Multi-column, Long Tiny Text, Base, Overall). Số các model khác lấy từ công bố của họ (hoặc tự đánh giá DeepSeek-OCR, Mistral OCR 3).**

| Model | E2E | Size (B) | ArXiv | OSM | Tables | OldScans | MultiCol | LTT | Base | Overall |
|---|---|---|---|---|---|---|---|---|---|---|
| Mistral OCR 3 API | - | - | 85.6 | 69.7 | 85.5 | 43.5 | 81.2 | 88.5 | 99.7 | 79.1 ± 1.0 |
| olmOCR v0.3.0 | × | 8 | 78.6 | 79.9 | 72.9 | 43.9 | 77.3 | 81.2 | 98.9 | 76.1 ± 1.1 |
| dots.ocr | × | 3 | 82.1 | 64.2 | 88.3 | 40.9 | 82.4 | 81.2 | 99.5 | 76.9 ± 1.0 |
| dots.ocr-1.5 | × | 3 | 85.9 | 85.5 | 90.7 | 48.2 | 85.3 | 81.6 | 99.7 | 82.4 ± 0.9 |
| DeepSeek-OCR | ✓ | 3 | 77.5 | 74.5 | 77.3 | 33.1 | 67.3 | 83.0 | 99.3 | 73.1 ± 1.0 |
| Chandra-9B | ✓ | 9 | 82.2 | 80.3 | 88.0 | 50.4 | 81.2 | 92.3 | 99.9 | 81.7 ± 0.9 |
| olmOCR-2-8B | ✓ | 8 | 82.9 | 82.1 | 84.3 | 48.3 | 84.3 | 81.4 | 99.7 | 80.4 ± 1.1 |
| PaddleOCR-VL | × | 0.9 | 85.7 | 71.0 | 84.1 | 37.8 | 79.9 | 85.7 | 98.5 | 77.5 ± 1.0 |
| LightOnOCR-1B-1025 | ✓ | 1 | 81.4 | 71.6 | 76.4 | 35.2 | 80.0 | 88.7 | 99.5 | 76.1 ± 1.1 |
| LightOnOCR-1B-1025-GRPO | ✓ | 1 | 86.5 | 73.8 | 74.5 | 32.9 | 85.1 | 91.6 | 99.7 | 77.7 ± 1.0 |
| LightOnOCR-2-1B-base | ✓ | 1 | 84.9 | 80.3 | 86.7 | 47.0 | 84.6 | 89.1 | 99.8 | 81.8 ± 0.9 |
| LightOnOCR-2-1B-bbox-base | ✓ | 1 | 84.6 | 78.6 | 84.7 | 46.0 | 83.8 | 88.0 | 99.8 | 80.8 ± 0.9 |
| LightOnOCR-2-1B-bbox | ✓ | 1 | 86.9 | 74.7 | 88.6 | 39.7 | 85.0 | 86.4 | 99.8 | 80.2 ± 0.9 |
| LightOnOCR-2-1B-bbox-soup | ✓ | 1 | 86.1 | 77.9 | 88.2 | 41.2 | 85.4 | 87.3 | 99.7 | 80.8 ± 0.9 |
| LightOnOCR-2-1B-ocr-soup | ✓ | 1 | 86.8 | 81.2 | 89.0 | 45.4 | 84.2 | 90.3 | 99.7 | 82.4 ± 0.9 |
| **LightOnOCR-2-1B** | ✓ | 1 | **89.6** | **85.6** | 89.0 | 42.2 | 84.8 | 91.4 | 99.6 | **83.2 ± 0.9** |

(Một số dòng baseline khác trong bảng gốc đã lược bớt; cột "E2E" là dấu ✓/× đúng như paper.) Khoảng tin cậy ±0,9 nghĩa là chênh dưới ~1 điểm (ví dụ 83,2 vs 82,4 của dots.ocr-1.5 hoặc 81,8 của base) **chưa phân biệt được chắc chắn** về mặt thống kê.

**Phân rã RLVR (so base -> LightOnOCR-2-1B):** ArXiv 84,9 -> 89,6 (+4,7); Old Scans Math 80,3 -> 85,6 (+5,3); Tables 86,7 -> 89,0 (+2,3); Old Scans 47,0 -> 42,2 (**−4,8**); Multi-column 84,6 -> 84,8; LTT 89,1 -> 91,4; Overall 81,8 -> 83,2 (+1,4). Tỷ lệ đầu ra lặp (zlib ratio < 0,13): **1,14% -> 0,50%**.

**Bbox (LightOnOCR-bbox-bench: 290 mẫu OlmOCR duyệt tay + 565 mẫu arXiv/nvpdftex tự động):**

| Model | OlmOCR F1@0.5 | IoU | Count acc | arXiv F1@0.5 | IoU | Count acc |
|---|---|---|---|---|---|---|
| Chandra-9B | 0,75 | 0,71 | 75,2 | 0,81 | 0,77 | 81,8 |
| LightOnOCR-2-1B-bbox | 0,78 | 0,70 | 83,8 | 0,83 | 0,77 | 85,0 |
| LightOnOCR-2-1B-bbox-soup | 0,76 | 0,67 | 80,7 | 0,82 | 0,76 | 85,1 |

**Hiệu năng (H100 80GB, chạy cả 1.403 trang, thư viện chính thức của từng model, đo trang/giây thay vì token/giây vì độ dài đầu ra khác nhau):**

| Model | Dtype | Size (B) | Trang/giây | Nhanh hơn chậm nhất |
|---|---|---|---|---|
| **LightOnOCR-2** | BF16 | 1 | **5,71** | 6,49× |
| olmOCR-2 | FP8 | 8 | 3,28 | 3,73× |
| olmOCR-2 | BF16 | 8 | 2,54 | 2,89× |
| DeepSeek-OCR | BF16 | 3 | 2,36 | 2,68× |
| PaddleOCR-VL | BF16 | 1 | 2,14 | 2,43× |
| Chandra | BF16 | 9 | 1,70 | 1,93× |
| dots.ocr | BF16 | 3 | 0,88 | 1,00× |

Tự tính từ bảng: LightOnOCR-2 nhanh hơn olmOCR-2-FP8 khoảng 1,74×, olmOCR-2-BF16 2,25×, Chandra 3,36×, **PaddleOCR-VL (cũng ~1B) 2,67×**.

**Vocabulary pruning (Bảng 4, trên LightOnOCR-1):**

| Vocab | Tham số (M) | OlmOCR Overall | Tăng tốc so base | Token/trang (EN / ZH) |
|---|---|---|---|---|
| 151k (gốc) | 1005,6 | 76,1 | - | 475 / 950 |
| 51k | 902,5 | 67,7 | +9,5% | 485 / 2.220 |
| 32k | 883,6 | 73,1 | +11,6% | 510 / 2.750 |
| 16k | 866,8 | 75,4 | +3,9% | 575 / 3.200 |

Tự tính: giảm tham số so gốc: 51k −10,3%, 32k −12,1%, 16k −13,8%. Token/trang tiếng Anh: +2,1% (51k), +7,4% (32k), +21,1% (16k). Tiếng Trung: ~2,3×, 2,9×, 3,4×. Paper kết luận: 16k "gần như không mất" (75,4 vs 76,1); 32k "cân bằng tốt nhất" (nhanh hơn 11,6%, giữ 96% hiệu năng gốc = 73,1/76,1); chữ không phải Latin (Ả Rập, Trung) bị **lạm phát ~3× số token**.

**Đánh giá phụ:**
- *OmniDocBench v1.0 (phụ lục C.2):* LightOnOCR-2-1B: Overall edit EN 0,146 / ZH 0,263, so với PaddleOCR-VL 0,105 / 0,126 và MinerU2.5 0,111 / 0,174 (thấp tốt hơn). Tức **yếu hơn ở OmniDocBench, đặc biệt tiếng Trung**; tác giả coi đây là tín hiệu phụ vì metric nhạy với quy ước định dạng.
- *Header/footer theo định nghĩa gốc:* LightOnOCR-2-1B 19,74 vs base 31,05 (do huấn luyện thưởng cho việc giữ header/footer nên điểm thấp là có chủ ý).

## 5. Phân tích / ablation

Những gì paper thực sự cô lập được (không có ablation nguồn dữ liệu, độ phân giải hay kích thước teacher):

| So sánh | Kết quả | Kết luận paper |
|---|---|---|
| base vs RLVR | 81,8 -> 83,2; lặp 1,14% -> 0,50% | RLVR giảm lỗi lặp, cải thiện toán và định dạng |
| bbox-base vs base | 81,8 -> 80,8 | thêm bbox trong pretraining làm tụt OCR ~1 điểm |
| bbox (sau RLVR) vs bbox-base | 80,8 -> 80,2 | RLVR bbox cải thiện toạ độ, hơi tụt OCR (cột Old Scans 46,0 -> 39,7; Tables lại tăng 84,7 -> 88,6) |
| merge (α = 0,1, 0,4) | bbox-soup 80,8; ocr-soup 82,4 | trộn trọng số hồi phục OCR, kiểm soát được trade-off, không tốn train |
| RLVR trên v1 (GRPO) | 76,1 -> 77,7 | cùng công thức cho cải thiện tương tự |
| vocab 151k -> 32k/16k | xem 4 | giảm ~12-14% tham số, tăng tốc nhỏ, đổi lại hại chữ không Latin |

**Điểm đáng ngờ/đáng lưu ý (của chúng tôi):**
- RLVR ghi nhận **Old Scans giảm 4,8 điểm** (47,0 -> 42,2) trong khi nhóm khác tăng; nhóm này chứa test absence về header/footer, mà RLVR đã đảo ngược dấu; paper không giải thích, đây chỉ là giả thuyết.
- Với bản v1, RLVR kiểu GRPO làm **OmniDocBench xấu đi** (Overall edit EN 0,234 -> 0,254; TextEdit EN 0,090 -> 0,267), dù tăng ở OlmOCR-Bench. Gợi ý: tối ưu theo unit test của một benchmark có thể lệch khỏi metric khác (reward hacking nhẹ/overfit phong cách benchmark). Paper không bàn.
- Vocab pruning **không đơn điệu**: 51k (67,7) tệ hơn cả 32k (73,1) và 16k (75,4), trong khi 16k lại tăng tốc ít nhất (+3,9%) vì phải sinh nhiều token hơn. Paper không giải thích (có thể liên quan cách chọn/khôi phục, nhưng đó là phỏng đoán).

## 6. Giới hạn

**Do tác giả nêu:**
- Đa ngôn ngữ ngoài Latin/Châu Âu chưa hỗ trợ đầy đủ: CJK, Ả Rập suy giảm chất lượng hoặc tokenization kém hiệu quả; **rõ hơn nữa ở model đã cắt vocab** ("all the more evident for pruned models").
- Chữ viết tay còn không ổn định; supervision chủ yếu từ chữ in/typeset.
- Điểm header/footer theo định nghĩa gốc thấp có chủ ý (mục tiêu là phiên âm đầy đủ trang).

**Do chúng tôi nhận xét:**
- Không có ablation về dữ liệu (2,5× dữ liệu, teacher mới, 1540 px, augmentation, trang trống) nên không biết cái nào đóng góp bao nhiêu; chỉ biết gói cuối cùng tốt.
- Bảng 1 trộn số công bố bởi nhóm khác với số tự đo (inference khác nhau); so sánh công bằng tuyệt đối không được đảm bảo.
- Chỉ **chưng cất dữ liệu cứng**; không có KD logit, không đo lượng thông tin bị bỏ qua so với KD đầy đủ.
- Paper không công bố số tham số ViT/decoder, FLOPs, độ trễ batch-1, bộ nhớ đỉnh; chỉ có thông lượng H100 ở batch lớn.
- Vocab pruning thử trên v1, không rõ có áp dụng cho v2; không rõ có huấn luyện lại sau cắt.
- RLVR dùng test "kiểu OlmOCR-Bench" trên tài liệu tổng hợp: có nguy cơ khớp quá mức phong cách benchmark chấm điểm chính.
- Dùng T = 0,2 và 3 lần sinh (không phải greedy) khi báo cáo.

## 7. Liên hệ với các bài khác

- **42 olmOCR:** nguồn của OlmOCR-Bench và công thức RLVR (olmOCR 2); LightOnOCR đổi hướng thưởng header/footer.
- **41 GOT-OCR2.0:** đối cực: ít token (256/tile) nhưng tổng quát yếu (48,3 trên olmOCR-Bench); LightOnOCR dùng nhiều visual token hơn và dữ liệu thật do teacher gán nhãn.
- **37 Qwen2-VL:** teacher bản 1 (72B); ý tưởng ảnh độ phân giải gốc và gộp patch 2×2.
- **29 ViT / 31 SigLIP:** vision encoder; cách gộp 2×2 tương tự projector của các VLM hiện đại.
- **44/45 FlashAttention (-2):** dùng khi huấn luyện (FA-2, bf16).
- **46/47 vLLM, SGLang:** sinh rollout RL bằng vLLM; thông lượng suy luận phụ thuộc engine.
- **24 Gated DeltaNet, 25-26 mô hình lai:** bản 3 của bạn thay decoder Qwen3 bằng decoder lai; paper này không đề cập.
- **08 LoRA:** không dùng ở đây (huấn luyện đầy đủ).
- **57 FastV:** hướng giảm visual token mà LightOnOCR chưa dùng (nó chỉ gộp 2×2 cố định).

## 8. Ứng dụng cho dự án (suy luận)

**8.1 Quy mô tham số: embedding chiếm phần lớn khi cắt nhỏ.** Với d=1024 và vocab 151.936 (suy ra, tied): embedding ≈ 155,6M tham số. Mục tiêu của bạn:

| Model | Tổng | Embedding 151k (tự tính) | Tỉ lệ |
|---|---|---|---|
| ~0,8-1,0B (hiện tại) | ~0,8-1,0B | ~156M (nếu d=1024) | ~16-19% |
| ~0,3B | 0,3B | ~156M | **~52%** |
| ~0,1B | 0,1B | ~156M | **>100% (không thể)** |

Nghĩa là mục tiêu 0,1B **không thể đạt nếu không cắt vocab** (nếu d_model vẫn 1024; nếu bạn cũng giảm d_model thì embedding co theo tỉ lệ d, nhưng vocab vẫn là thành phần lớn nhất). Kết quả Bảng 4 là bằng chứng thực nghiệm đầu tiên về chi phí/lợi ích: cắt còn 32k giữ ~96% điểm (tiếng Anh/Pháp), 16k giữ ~99% nhưng tăng token. **Đề xuất:** cắt vocab là bước đầu tiên (gần như "miễn phí" về độ chính xác trên Latin), trước khi cắt lớp/neuron. Chú ý mất ngôn ngữ ngoài phạm vi (lạm phát token ~3×).

**8.2 Lợi ích tốc độ thật của cắt vocab ở batch-1.** Mỗi bước giải mã đọc toàn bộ trọng số. Với embedding tied d=1024: LM head đọc 155,6M × 2 byte ≈ 311 MB ở bf16; ở 32k chỉ ≈ 66 MB (tự tính). Với model ~1B (≈2 GB bf16), đó là ~12% byte phải đọc mỗi token, **cùng bậc độ lớn với mức tăng tốc 11,6% paper đo cho 32k** (tương đồng này là ngẫu hợp có thể, chưa kiểm chứng) (tăng tốc ít hơn ở 16k vì sinh nhiều token hơn: số bước giải mã tăng ~21%). Ở batch-1 (memory-bound) hiệu ứng này nên còn rõ hơn throughput batch lớn, nhưng chưa được paper đo.

**8.3 Trần độ trễ batch-1 (tự tính, giả định).** Mỗi token giải mã ≈ đọc toàn bộ trọng số một lần. Giả sử băng thông bộ nhớ 300 GB/s (GPU ≤ 8 GB, giả định), 1.500 token đầu ra/trang:

| Tham số (bf16) | Byte/bước | Thời gian/bước | 1.500 token |
|---|---|---|---|
| 0,8B | 1,6 GB | ~5,3 ms | ~8 s |
| 0,3B | 0,6 GB | ~2,0 ms | ~3 s |
| 0,1B | 0,2 GB | ~0,67 ms | ~1 s (thực tế bị chặn bởi overhead khởi chạy kernel/24 lớp, nên sàn cao hơn) |

(Đây là cận dưới theo băng thông; không tính KV cache, prefill, vision encoder.) Kết luận: giảm số **tham số** (embedding + các lớp) là đòn bẩy chính cho giải mã batch-1.

**8.4 Visual token và KV cache (liên hệ 57 FastV).** Số visual token/trang ~1,6k-2,1k (3.1) so với ~1.500 token văn bản. Với decoder lai chỉ ~1/4 lớp có KV cache (giả định), KV chỉ khoảng 25 KB/token (8 KV head × 128 × 2 × 2 byte × 6 lớp, giả định cấu hình) -> với 3.600 token ≈ 88 MB, **chỉ ~5% byte đọc mỗi bước so với 1,6 GB trọng số**. Vì vậy cắt visual token chủ yếu giảm **prefill** (một lần), ít giúp độ trễ giải mã batch-1; nên ưu tiên giảm tham số.

**8.5 Chưng cất.** Paper này chỉ chưng cất nhãn cứng 235B -> 1B. Dự án của bạn có teacher 4B và (có thể) KD logit; paper cho thấy:
- Chuẩn hoá nhãn teacher (loại artefact, trang trống -> chuỗi rỗng, lọc lặp) quan trọng ngay cả khi chưng cất nhãn cứng; nếu dùng KD logit từ teacher 4B, vẫn nên dùng bộ nhãn đã chuẩn hoá làm "hard target" kèm theo.
- **RLVR sau SFT/KD** là cách đã được chứng minh để giảm lặp (1,14% -> 0,50%) và nâng điểm unit test (+1,4); có thể xem như bước hồi phục cuối sau cắt tỉa, nhưng nhớ cảnh báo về overfit benchmark (5).
- **Trộn trọng số** (soup 5 checkpoint cuối; task arithmetic) rẻ; có thể dùng để gom các bản phục hồi cắt tỉa khác seed hoặc để cân bằng "đã cắt" với "gốc" (nội suy kiểu α), thử nghiệm đáng làm.

**8.6 Lưu ý về D_lin / D_res và neuron vs lớp.** Paper không có phân tích này; mối liên hệ duy nhất là dữ liệu: vocab pruning (cắt một chiều cấu trúc nhưng gần như "thừa thãi") không hại, còn cắt ở nơi nặng thông tin (ngôn ngữ ngoài Latin) thì hại; gợi ý rằng cần quyết định cắt theo **phân phối dữ liệu mục tiêu** (Anh/Pháp), thay vì cắt đều.

**8.7 Đánh giá.** Dùng chính cấu hình đánh giá của paper (OlmOCR-Bench, bỏ HF, T=0,2, top_p=0,9, 3 lần/trang, 1540 px) để số liệu của bạn so được trực tiếp; CI ±0,9, nên chỉ coi chênh ≥ ~2 điểm là có ý nghĩa.

## 9. Thí nghiệm gợi ý

1. **Vocab pruning trước, rồi cắt lớp:** cắt vocab còn 32k bằng tần suất (kèm lan truyền sub-token), đo OlmOCR-Bench, độ dài token/trang và tốc độ batch-1. Sau đó cắt lớp. Ghi nhận: có cần tinh chỉnh lại sau khi cắt vocab không (paper không nói).
2. **Đường cong embedding vs tham số:** vẽ OlmOCR-Bench theo vocab ∈ {151k, 51k, 32k, 16k, 8k} ở cùng d_model, riêng cho Anh/Pháp, để kiểm tra tính không đơn điệu của 51k (có thể là nhiễu của việc chọn token).
3. **Soup phục hồi:** sau khi chưng cất model ~0,3B, trung bình trọng số 5 checkpoint cuối và so với checkpoint cuối đơn lẻ; thêm nội suy \(\theta=\theta_{distill}+\alpha(\theta_{ft}-\theta_{distill})\) giữa bản chưng cất thuần và bản có RLVR.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / giá trị |
|---|---|
| Kiến trúc | ViT (từ Mistral-Small-3.1) + gộp 2×2 (−4× token) + MLP 2 lớp GELU + decoder Qwen3 |
| Mục tiêu SFT | CE trên token assistant (bỏ prompt và token ảnh) |
| Huấn luyện | AdamW, lr 1e-4, cosine, warmup 100, batch 384, seq 6144, 96×H100 |
| Teacher | Qwen3-VL-235B-A22B-Instruct (v2), Qwen2-VL-72B (v1), GPT-4o (crop) |
| GRPO | 1 epoch, lr 4e-5, β=0,01, 28/14 rollout (OCR/bbox) |
| Thưởng bbox | \(R=\overline{\mathrm{IoU}}\cdot|I_\cap|/\max(|I_{gt}|,|I_{pred}|)\) |
| Task arithmetic | \(\theta=\theta_{base}+\alpha(\theta_{rl}-\theta_{base})\), α=0,4 (OCR), 0,1 (bbox) |
| Soup | trung bình 5 checkpoint cuối |
| Vocab | 151.936 -> 51k/32k/16k; −10,3% / −12,1% / −13,8% tham số (tự tính) |
| Kết quả | OlmOCR-Bench 83,2 ± 0,9; 5,71 trang/s trên H100; lặp 0,50% |

## 11. Kết luận ngắn

LightOnOCR-2-1B cho thấy một model 1B end-to-end có thể đứng đầu OlmOCR-Bench nhờ **dữ liệu lớn, sạch, chưng cất từ teacher 235B**, huấn luyện độ phân giải cao, RLVR có mục tiêu và trộn trọng số, đồng thời nhanh hơn 1,7-3,4× các model 8-9B (và 2,7× PaddleOCR-VL cùng cỡ) trên H100. Với dự án của bạn, bài học thực dụng nhất là: (1) **cắt vocab trước** (bắt buộc cho mục tiêu 0,1B), (2) chuẩn hoá nhãn teacher và RLVR/soup như các bước hồi phục rẻ, (3) dùng cùng giao thức OlmOCR-Bench để so sánh, và (4) nhớ rằng paper này không bao quát phần hybrid attention, KD logit hay cắt tỉa cấu trúc, nên các quyết định đó cần thí nghiệm riêng.

Link gốc: https://arxiv.org/abs/2601.14251
