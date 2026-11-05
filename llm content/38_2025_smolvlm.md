# SmolVLM: Redefining small and efficient multimodal models (2025)

**Tác giả:** Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Vaibhav Srivastav, Joshua Lochner, Hugo Larcher, Mathieu Morlon, Lewis Tunstall, Leandro von Werra, Thomas Wolf (Hugging Face, Stanford).  
**Venue / arXiv:** arXiv:2504.05299 (v1, 7/4/2025); bài không nêu venue.  
**Nguồn:** https://arxiv.org/abs/2504.05299  
**Chủ đề:** Thiết kế VLM nhỏ (256M-2.2B): cân bằng encoder/LM, kéo dài ngữ cảnh, pixel shuffle, ngân sách token thị giác, token vị trí học được, công thức dữ liệu.  
**Vị trí trong lộ trình:** Đọc trước: 29 (ViT), 31 (SigLIP), 36 (InternVL 1.5, pixel shuffle), 37 (Qwen2-VL). Bài này thêm: các "finding" thực nghiệm cho VLM nhỏ, và con số bộ nhớ thực tế. Tiếp theo: 41-43 (các mô hình OCR), 57 (FastV, cắt token).

> Ghi chú học tập, không phải bản dịch. Phần "Ứng dụng cho dự án" là suy luận của ta. Số "tự tính" không có trong paper. Paper phần lớn trình bày kết quả bằng hình (Hình 3, 5, 6, 7) mà bản trích văn bản không đọc được giá trị; ghi chú chỉ dùng những số có trong chữ hoặc bảng.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
4. [Thí nghiệm và kết quả chính](#4-thí-nghiệm-và-kết-quả-chính)
5. [Phân tích / ablation (9 findings)](#5-phân-tích--ablation-9-findings)
6. [Giới hạn](#6-giới-hạn)
7. [Liên hệ với các bài khác](#7-liên-hệ-với-các-bài-khác)
8. [Ứng dụng cho dự án (suy luận)](#8-ứng-dụng-cho-dự-án-suy-luận)
9. [Thí nghiệm gợi ý](#9-thí-nghiệm-gợi-ý)
10. [Tóm tắt công thức / khái niệm](#10-tóm-tắt-công-thức--khái-niệm)
11. [Kết luận ngắn](#11-kết-luận-ngắn)

---

## 1. Bối cảnh

VLM mạnh thường rất tốn tài nguyên. Các VLM "nhỏ" thường chỉ **thu nhỏ LLM nhưng giữ nguyên các lựa chọn thiết kế của bản lớn**: encoder thị giác to, nhiều token ảnh mỗi ảnh. Hậu quả mà paper nhấn mạnh: VRAM bị tiêu tốn lớn dù số tham số nhỏ. Ví dụ trong paper: Qwen2-VL-2B cần 13.7 GB VRAM và InternVL2-2B cần 10.5 GB, trong khi SmolVLM-2.2B cần 4.9 GB. Từ đó họ lập luận **số tham số là chỉ báo kém cho chi phí chạy VLM; RAM sử dụng là chỉ báo tốt hơn**.

Thuật ngữ cần biết:

- **Token thị giác:** vector đầu ra của ViT được xem như "từ" đưa vào LLM. Số token càng nhiều, ngữ cảnh LLM càng dài, chi phí prefill (bậc hai với attention đầy đủ) và KV cache càng lớn.
- **Pixel shuffle (space-to-depth):** phép sắp xếp lại, gộp một khối $r\times r$ vị trí không gian thành một vị trí với số kênh nhân $r^2$.
- **SigLIP (bài 31):** họ encoder ảnh dùng loss sigmoid; SmolVLM dùng SigLIP-B/16 (93M) hoặc SigLIP-SO400M (428M, paper ghi 400M ở một chỗ).
- **SmolLM2:** họ LLM nhỏ của Hugging Face (135M, 360M, 1.7B).

---

## 2. Ý tưởng chính

Paper là một chuỗi **thực nghiệm có hệ thống về thiết kế VLM nhỏ**, rút ra chín "finding". Ý tưởng cốt lõi: khi LLM nhỏ, nên (i) dùng encoder thị giác nhỏ để cân bằng, (ii) nén token thị giác **mạnh hơn** các VLM lớn (pixel shuffle hệ số 4 thay vì 2), đồng thời (iii) kéo dài ngữ cảnh LLM để chứa ảnh độ phân giải cao, (iv) chia ảnh lớn thành ô con kèm ảnh thu nhỏ, và (v) tinh chỉnh công thức dữ liệu và prompt cho phù hợp với dung lượng nhỏ. Kết quả là ba mô hình: 256M, 500M, 2.2B, trong đó mô hình 256M chạy dưới 1 GB VRAM.

---

## 3. Phương pháp chi tiết

### 3.1 Kiến trúc (Hình 2)

```text
ảnh -> chia ô con (+ ảnh thu nhỏ) -> vision encoder (SigLIP)
    -> pixel shuffle -> linear/MLP projection -> token thị giác
    -> nối/xen kẽ với embedding chữ -> SmolLM2 -> chữ ra
```

Kiểu ghép: **self-attention architecture**, tức token thị giác được nối thẳng vào chuỗi cùng token chữ rồi LLM xử lý chung (khác cross-attention của Flamingo, bài 32). Video: lấy mẫu khung rồi mã hóa như ảnh.

Ba biến thể:

| Mô hình | Encoder | LM | Ghi chú của paper |
|---|---|---|---|
| SmolVLM-256M | SigLIP-B/16 93M | SmolLM2-135M | < 1 GB VRAM |
| SmolVLM-500M | SigLIP-B/16 93M | SmolLM2-360M | cân bằng |
| SmolVLM-2.2B | SigLIP-SO400M | SmolLM2-1.7B | mạnh nhất |

### 3.2 Cân bằng encoder - LM (mục 2.1, Finding 1)

Họ ghép ba SmolLM2 (135M, 360M, 1.7B) với hai SigLIP (93M, 428M), tức 6 cấu hình. Kết luận trong chữ:

- LM 135M + encoder lớn: hiệu năng **giảm đáng kể**, cân bằng kém.
- LM 360M: encoder lớn tăng điểm **11.6%** nhưng tăng **66% tham số** → chọn encoder nhỏ.
- LM 1.7B: encoder lớn chỉ làm tăng tham số ~10% → đáng dùng.

**Finding 1:** mô hình đa phương thức gọn nên cân bằng phân bổ tham số giữa encoder và LM, ưu tiên encoder nhỏ khi LM nhỏ.

### 3.3 Ngữ cảnh dài (mục 2.2, Finding 2)

Vì ghép bằng self-attention, ảnh tiêu tốn nhiều vị trí: một ảnh 512×512 qua SigLIP-B/16 cho $(512/16)^2 = 32^2 = 1024$ token (số có trong paper). SmolLM2 gốc chỉ có ngữ cảnh 2k. Cách xử lý:

- Tăng **RoPE base từ 10k lên 273k**.
- Tinh chỉnh trên hỗn hợp dữ liệu dài (Dolma books, The Stack) và ngắn (FineWeb-Edu, DCLM, toán của SmolLM2).
- Kết quả: LM 1.7B ổn định ở 16k; LM nhỏ (135M, 360M) khó vượt 8k. Chọn **16k cho 2.2B, 8k cho hai mô hình nhỏ**.

### 3.4 Pixel shuffle (Hình 4, Finding 3)

Đầu vào là bản đồ đặc trưng $X\in\mathbb{R}^{H\times W\times C}$. Pixel shuffle với hệ số $r$:

$$
\boxed{\;\mathbb{R}^{H\times W\times C}\;\longrightarrow\;\mathbb{R}^{(H/r)\times(W/r)\times (C\,r^2)}\;}
$$

rồi một lớp tuyến tính/MLP chiếu $C r^2$ chiều về chiều của LM. Số token giảm $r^2$ lần:

$$
N_{\text{sau}} = \frac{N_{\text{trước}}}{r^2}.
$$

**Pseudo-code PyTorch:**

```python
def pixel_shuffle(x, r):            # x: [B, H, W, C]  (H, W chia hết cho r)
    B, H, W, C = x.shape
    x = x.view(B, H // r, r, W // r, r, C)
    x = x.permute(0, 1, 3, 2, 4, 5)           # [B, H/r, W/r, r, r, C]
    return x.reshape(B, H // r, W // r, C * r * r)

tokens = proj(pixel_shuffle(vit_out, r=4)).flatten(1, 2)   # [B, N/16, d_lm]
```

**Ví dụ số (tự tính từ số liệu paper):** ảnh 512×512, SigLIP-B/16 → 1024 token. Với $r=2$ còn $1024/4 = 256$ token; với $r=4$ còn $1024/16 = 64$ token.

Tác giả lưu ý: tỉ lệ cao làm một token phải mô tả vùng không gian lớn hơn nên hại cho tác vụ cần định vị chính xác như OCR. InternVL và Idefics3 dùng $r=2$. Nhưng thực nghiệm của họ (Hình 3, phải) cho thấy **VLM nhỏ lại hưởng lợi từ $r=4$**, vì ít token hơn làm nhẹ attention và dễ mô hình hóa ngữ cảnh dài. Paper cũng ghi (chú thích Hình 3) rằng hệ số tối ưu thay đổi theo cỡ mô hình (PS=2 vs PS=4).

**Finding 3:** VLM nhỏ hưởng lợi từ nén token thị giác mạnh hơn.

### 3.5 Ảnh và video (mục 2.3, Finding 4)

- **Ảnh độ phân giải cao:** chia thành nhiều ảnh con + một ảnh thu nhỏ của toàn bộ (kiểu UReader, SPHINX). Hiệu quả cho tác vụ ảnh.
- **Video:** thử "frame averaging" (trung bình nhiều khung thành một) nhưng làm tệ đi, và càng tệ khi hệ số (2, 4, 8) tăng. Bỏ; thay vào đó chỉ rescale khung về độ phân giải của encoder.

Khi đánh giá, cạnh dài của ảnh được resize về **1920** (256M, 500M) hoặc **1536** (2.2B).

### 3.6 Token vị trí học được (mục 3.1, Finding 5)

Cần báo cho LM biết ô con nào (hàng, cột). Cách ban đầu: chuỗi chữ như `<row_1_col_2>` — gây ra hiện tượng họ gọi là **"OCR loss plague"**: loss đột ngột tụt nhưng chất lượng OCR không tốt lên (Hình 5 trái). Sửa bằng **token vị trí đặc biệt** có embedding riêng; mô hình lớn khá bền với chuỗi chữ, mô hình nhỏ hưởng lợi rõ (OCR tốt hơn, tổng quát hóa tốt hơn).

### 3.7 Prompt có cấu trúc (mục 3.2, Finding 6)

- **System prompt** ngắn (ví dụ "You are a visual agent and should provide concise answers"): cải thiện, nhất là ảnh.
- **Token mở/đóng media** ("Here is an image...", "Here are N frames sampled from a video..."; xen kẽ với "Given this image/video..."): giúp mạnh ở video, cũng giúp ảnh.
- **Che prompt người dùng** khi SFT (chỉ tính loss trên phần trả lời): giảm overfit vì câu hỏi QA đa phương thức lặp lại và dễ bị ghi nhớ.

### 3.8 Công thức dữ liệu (mục 3.3-3.5, 4.1)

- **Finding 7:** thêm dữ liệu văn bản SFT của LLM (SmolTalk) **làm tệ đi** mô hình nhỏ: giảm tới 3.7% ở video và 6.5% ở ảnh. Giữ tỉ lệ text đúng **14%**, dùng dữ liệu text khác.
- **Finding 8:** dữ liệu chain-of-thought chỉ nên rất ít (0.02-0.05%); nhiều hơn làm giảm điểm, đặc biệt ảnh.
- **Finding 9:** tăng độ dài video trung bình khi huấn luyện đến ~3.5 phút thì cải thiện cả video lẫn ảnh, vượt 3.5 phút lợi ích giảm dần.
- Huấn luyện hai giai đoạn: giai đoạn ảnh, rồi giai đoạn video (14% text, 33% video). Giai đoạn ảnh có tài liệu, OCR, biểu đồ, bảng, và bổ sung MathWriting.

---

## 4. Thí nghiệm và kết quả chính

**Bảng 1 (số đúng như paper):**

| Benchmark | 256M | 500M | 2.2B | Mô hình hiệu quả khác (cột cuối của paper) |
|---|---|---|---|---|
| OCRBench | 52.6% | 61.0% | 72.9% | 54.7% (MolmoE-A1B-7B) |
| AI2D | 46.4% | 59.2% | 70.0% | 71.0% (MolmoE) |
| ChartQA | 55.6% | 62.8% | 68.7% | 48.0% (MolmoE) |
| TextVQA | 50.2% | 60.2% | 73.0% | 61.5% (MolmoE) |
| DocVQA | 58.3% | 70.5% | 80.0% | 77.7% (MolmoE) |
| ScienceQA | 73.8% | 80.0% | 89.6% | 87.5% (MolmoE) |
| MMMU | 29.0% | 33.7% | 42.0% | 33.9% (MolmoE) |
| MathVista | 35.9% | 40.1% | 51.5% | 37.6% (MolmoE) |
| MMStar | 34.6% | 38.3% | 46.0% | 43.1% (MolmoE) |
| Video-MME | 33.7% | 42.2% | 52.1% | 45.0% (InternVL2-2B) |
| **Trung bình** | **44.0%** | **51.0%** | **59.8%** | - |
| **RAM, batch 1** | **0.8 GB** | **1.2 GB** | **4.9 GB** | 27.7 GB (MolmoE) |
| RAM, batch 64 | 15.0 GB | 16.0 GB | 49.9 GB | - |

(Bảng còn MLVU, MVBench, WorldSense, TempCompass; bỏ bớt cho gọn. Cột "mô hình hiệu quả khác" lấy MolmoE-A1B-7B cho tác vụ ảnh và InternVL2-2B hoặc Qwen2VL-7B cho video, theo ghi chú trong paper.)

Điểm đáng chú ý:

- Abstract: SmolVLM-256M dùng < 1 GB GPU và vượt Idefics-80B (to hơn ~300 lần).
- Ngoại lệ ở mô hình 256M so với Idefics 80B: MMMU (29.0% vs 42.3%) và AI2D (46.4% vs 56.3%), tức tác vụ cần LLM suy luận mạnh.
- Paper nhận xét OCRBench cũng hưởng lợi rõ khi tăng LM (52.6% → 61.0% từ 256M lên 500M, hai mô hình dùng **cùng** encoder 93M). Tức **chất lượng OCR phụ thuộc LM chứ không chỉ encoder**.

**Thông lượng (mục 4.4):**

- A100: 256M từ 0.8 ví dụ/giây (batch 1) lên 16.3 ví dụ/giây (batch 64); 500M 0.7 → 9.9; 2.2B 0.6 → 1.7.
- L4: 256M đạt đỉnh 2.7 ví dụ/giây ở batch 8, sau đó giảm do giới hạn bộ nhớ.
- Trình duyệt (WebGPU): 256M đạt tới 80 token giải mã/giây trên MacBook Pro 14 inch (M4 Max).
- Ứng dụng: ColSmolVLM (truy hồi tài liệu) và SmolDocling (256M, chuyển đổi tài liệu end-to-end bằng DocTags).

---

## 5. Phân tích / ablation (9 findings)

| # | Finding | Ghi chú |
|---|---|---|
| 1 | Cân bằng encoder-LM; encoder nhỏ cho LM nhỏ | LM 360M: encoder lớn +11.6% điểm, +66% tham số |
| 2 | Ngữ cảnh dài giúp VLM nhỏ | RoPE base 10k → 273k; 16k (2.2B), 8k (nhỏ) |
| 3 | Nén token mạnh hơn cho VLM nhỏ | $r=4$ thay vì $r=2$ |
| 4 | Chia ô con tốt cho ảnh; frame averaging hại video | |
| 5 | Token vị trí học được > chuỗi chữ | tránh "OCR loss plague" |
| 6 | System prompt + token mở/đóng media; che prompt | |
| 7 | Không tái sử dụng text SFT của LLM | -3.7% video, -6.5% ảnh |
| 8 | Quá nhiều CoT hại | tối ưu ở 0.02-0.05% |
| 9 | Video dài vừa phải (~3.5 phút) | lợi suy giảm sau đó |

**Cách đọc có phê phán:** hầu hết các finding được chứng minh bằng đường cong trong hình (kết quả tuyệt đối khó đọc từ bản trích văn bản), trên điểm trung bình OpenCompass/CIDEr/VQA, và không có khoảng tin cậy. Ví dụ Hình 6 vẽ "violin" ba checkpoint cho mỗi cấu hình. Đáng tin ở xu hướng, kém tin ở độ lớn chính xác.

---

## 6. Giới hạn

**Tác giả nêu (qua nội dung):** hiệu năng ở tác vụ cần suy luận ngôn ngữ mạnh (MMMU, AI2D) vẫn thua mô hình lớn; LM nhỏ khó huấn luyện ổn định ở ngữ cảnh > 8k; video dài có lợi ích giảm dần.

**Ta nhận xét:**

- Thiết kế được tinh chỉnh cho bộ benchmark chung (OpenCompass), **không phải cho OCR tài liệu dài**. Kết luận "r=4 tốt hơn" có thể đảo ngược với trang chữ dày: chính paper cảnh báo nén mạnh hại OCR.
- Thông số "bộ nhớ" đo trong khung suy luận của họ (VLMEvalKit, ảnh resize cạnh dài 1920/1536); chạy trong stack khác (vLLM, ONNX, quantize) sẽ ra con số khác.
- Ablation ghép nhiều yếu tố cùng lúc (dữ liệu, prompt, kiến trúc) nên khó tách nhân quả.
- Thông lượng ở batch 1 (0.6-0.8 ví dụ/giây trên A100) cho thấy **batch 1 chưa nhanh**: lợi thế lớn nằm ở bộ nhớ và ở batch lớn.

---

## 7. Liên hệ với các bài khác

- **36 (InternVL 1.5):** nguồn của ý tưởng pixel shuffle trong VLM; SmolVLM đẩy hệ số lên 4 cho mô hình nhỏ.
- **37 (Qwen2-VL):** đối chiếu hai triết lý. Qwen2-VL giữ ảnh gần nguyên độ phân giải, gộp 2×2 (mỗi token = 28×28 px); SmolVLM chia ô cố định rồi nén 4×4 (mỗi token ứng với vùng 64×64 px nếu patch 16: $16\times4$). Cả hai dùng self-attention (nối token vào chuỗi).
- **31 (SigLIP), 29 (ViT):** encoder; SmolVLM chỉ ra tỉ lệ kích thước encoder/LM mới là biến quan trọng.
- **12 (RoPE):** việc tăng RoPE base (10k → 273k) là cách kéo dài ngữ cảnh đơn giản.
- **43 (LightOnOCR), 41-42 (GOT, olmOCR):** SmolDocling là phiên bản SmolVLM-256M cho tài liệu; cùng nhóm bài toán.
- **57 (FastV), 56 (H2O):** giảm token ở trong LLM, bổ sung cho nén token trước LLM kiểu pixel shuffle.

---

## 8. Ứng dụng cho dự án (suy luận)

Đây là bài **liên quan nhất** trong nhóm VLM tới mục tiêu GPU ≤ 8 GB của bạn, vì nó đo bộ nhớ thực và đưa ra các trade-off ở quy mô 0.1-0.5B.

1. **Cân bằng ViT / decoder khi cắt.** Mô hình của bạn: ViT 12 khối + decoder hybrid, cắt về ~0.3B và ~0.1B. Finding 1 gợi ý rằng khi decoder nhỏ lại, phần ViT chiếm tỉ trọng ngày càng lớn trong tổng tham số và chi phí. Cần tính tỉ lệ: nếu sau cắt ViT vẫn ~100M mà decoder còn 100M thì ta đang ở vùng "mất cân bằng" giống cấu hình (LM 135M, encoder 428M) của paper. Chú ý bài này nói về việc *chọn* cấu hình khi huấn luyện từ đầu; với cắt tỉa thì phải thực nghiệm lại, vì cắt ViT mất biểu diễn khác với chọn một ViT nhỏ được huấn luyện sẵn.
2. **Ngân sách token thị giác (tự tính, giả định).** Giả sử ảnh trang được chia thành 12 ô 512×512 cộng 1 ảnh thu nhỏ = 13 ảnh (số ô chỉ là giả định để tính). Mỗi ảnh 1024 token trước nén. Không nén: $13\times1024 = 13{,}312$ token. $r=2$: $13\times256 = 3{,}328$. $r=4$: $13\times64 = 832$ token. Cộng thêm ~1500 token đầu ra. Với KV cache full attention giả sử 96 KB/token (d_model≈1024, 24 lớp, 8 KV head × 128, bf16, tự tính): $r=2$ → $(3328+1500)\times96$ KB ≈ 453 MB; $r=4$ → $(832+1500)\times 96$ KB ≈ 219 MB. Nhưng chi phí prefill (bậc hai theo chiều dài với lớp full attention) giảm nhiều hơn tỉ lệ này. Hybrid GatedDeltaNet giảm KV cache hơn nữa.
3. **Nén token có thể hại OCR.** Paper cảnh báo $r$ lớn hại tác vụ cần định vị. Với OCR (mỗi ký tự nhỏ) rủi ro cao hơn benchmark chung. Cần đo trên dữ liệu của bạn, đừng bê nguyên r=4. Đặc biệt với trang chữ dày, token vừa phải ($r=2$) có thể đáng giá.
4. **LM đóng vai trò lớn trong OCR.** OCRBench 52.6% → 61.0% khi LM tăng 135M → 360M với cùng encoder 93M. Điều này gợi ý (không chứng minh) rằng khi cắt decoder từ ~0.8B xuống 0.1B, phần lớn suy giảm OCR đến từ decoder chứ không phải ViT, và distillation từ teacher 4B nên ưu tiên decoder.
5. **Token vị trí học được và "OCR loss plague".** Nếu bạn chia trang thành ô con và định vị bằng chuỗi chữ, nhớ hiện tượng loss tụt nhưng OCR không tốt lên; dùng token riêng. Hiện tượng này cũng là cảnh báo cho quan sát *loss distillation thấp nhưng CER không cải thiện*: phải đo CER, không chỉ loss.
6. **Dữ liệu cho student nhỏ:** text SFT tái sử dụng và CoT nhiều đều hại mô hình nhỏ. Với student 0.1B-0.3B, nên giữ công thức dữ liệu sát tác vụ OCR.
7. **Đo bộ nhớ như paper:** báo cáo VRAM theo từng cấu hình (batch 1) thay vì số tham số. SmolVLM-256M ở 0.8 GB, mục tiêu 8 GB của bạn rộng rãi, nên điểm nghẽn thực tế là **độ trễ**, không phải bộ nhớ.

---

## 9. Thí nghiệm gợi ý

1. **Quét hệ số nén token** ($r\in\{1,2,4\}$ hoặc gộp tương đương) trên mô hình OCR của bạn, đo CER theo loại trang (chữ thưa vs dày, bảng, công thức) cùng độ trễ prefill. Kiểm chứng xem "r=4 tốt cho VLM nhỏ" có đúng với OCR không.
2. **Bảng tỉ trọng sau cắt:** với mỗi cấu hình cắt (0.8B → 0.3B → 0.1B), lập bảng tham số và FLOPs của ViT vs decoder vs projector cho một trang; xác định xem ViT có trở thành thành phần chi phối.
3. **Phân rã suy giảm theo thành phần:** chỉ cắt ViT, chỉ cắt decoder, rồi đo CER và loss để xem phần nào chịu trách nhiệm (liên hệ với cách bạn đang phân rã $D_{lin}$ và $D_{res}$).

---

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Pixel shuffle | $\mathbb{R}^{H\times W\times C}\to\mathbb{R}^{(H/r)\times(W/r)\times Cr^2}$, token giảm $r^2$ lần |
| Token từ ViT | Ảnh 512×512, patch 16 → 1024 token; $r=2$: 256; $r=4$: 64 |
| Ngữ cảnh | RoPE base 10k → 273k; 16k (2.2B), 8k (256M, 500M) |
| Cân bằng | LM 360M + encoder lớn: +11.6% điểm, +66% tham số |
| Cấu hình | 256M = SigLIP 93M + 135M; 500M = 93M + 360M; 2.2B = SO400M + 1.7B |
| Bộ nhớ batch 1 | 0.8 / 1.2 / 4.9 GB |
| Điểm TB | 44.0 / 51.0 / 59.8 |
| Text SFT tái dùng | -3.7% (video), -6.5% (ảnh) |
| CoT | tối ưu 0.02-0.05% |
| Tỉ lệ text khi train | 14% |

## 11. Kết luận ngắn

SmolVLM là công thức thực nghiệm cho VLM nhỏ: encoder nhỏ khi LM nhỏ, nén token thị giác mạnh, ngữ cảnh dài, token vị trí học được, dữ liệu chọn lọc. Điểm giá trị nhất cho người triển khai là con số bộ nhớ (0.8 GB cho 256M, 4.9 GB cho 2.2B so với 13.7 GB của Qwen2-VL-2B trong cùng khung đo). Hạn chế: tối ưu cho benchmark chung chứ không phải OCR dày chữ, và phần lớn bằng chứng nằm trong đường cong không có sai số.

**Link gốc:** https://arxiv.org/abs/2504.05299
