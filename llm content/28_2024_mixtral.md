# Mixtral of Experts (2024)

**Tác giả:** Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Lélio Renard Lavaud, Lucile Saulnier, Marie-Anne Lachaux, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Théophile Gervet, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed (Mistral AI).  
**Venue / arXiv:** arXiv:2401.04088 (v1, 8/1/2024). Paper không nêu hội nghị.  
**Nguồn:** https://arxiv.org/abs/2401.04088  
**Chủ đề:** Sparse Mixture of Experts (SMoE) trong LLM decoder-only, top-2 routing, tham số tổng (sparse) và tham số hoạt động (active), phân tích định tuyến.  
**Vị trí trong lộ trình:** Đọc sau 06 (Mistral 7B, kiến trúc nền), 11 (SwiGLU) và 27 (Switch Transformer: nguồn gốc MoE hiện đại). Bài này đem MoE vào một LLM mở đạt chất lượng của Llama 2 70B. Bài kế tiếp theo mạch kiến trúc: 29 (ViT) bắt đầu nhánh thị giác; về MoE, xem lại 25 (Jamba) và 16 (DeepSeek-V2).

> Tài liệu học tập bằng tiếng Việt, tổng hợp từ paper (bản full text, khoảng 5k từ, ngắn). Đây là ghi chú diễn giải, không phải bản dịch. Phần "Ứng dụng cho dự án" và các số "tự tính" là suy luận của người viết. Paper chủ yếu là báo cáo kỹ thuật: kiến trúc chỉ chiếm một trang; phần lớn là bảng kết quả.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
4. [Thí nghiệm và kết quả chính](#4-thí-nghiệm-và-kết-quả-chính)
5. [Phân tích định tuyến (ablation)](#5-phân-tích-định-tuyến)
6. [Giới hạn](#6-giới-hạn)
7. [Liên hệ với các bài khác](#7-liên-hệ-với-các-bài-khác)
8. [Ứng dụng cho dự án (suy luận)](#8-ứng-dụng-cho-dự-án-suy-luận)
9. [Thí nghiệm gợi ý](#9-thí-nghiệm-gợi-ý)
10. [Tóm tắt công thức / khái niệm](#10-tóm-tắt-công-thức--khái-niệm)
11. [Kết luận ngắn](#11-kết-luận-ngắn)

---

## 1. Bối cảnh

Đến cuối 2023, các LLM mã nguồn mở mạnh nhất (Llama 2 70B) đều là mô hình dày (dense): mọi tham số chạy cho mọi token, nên 70B tham số nghĩa là khoảng 70B phép nhân-cộng mỗi token. Mistral 7B (bài 06) cho thấy mô hình nhỏ được thiết kế tốt có thể vượt mô hình lớn hơn, nhưng vẫn bị giới hạn bởi dung lượng.

**MoE (Mixture of Experts)** từ Switch Transformer (bài 27) giải quyết bằng cách có nhiều "expert" (ở đây là nhiều FFN song song) và một bộ định tuyến (router) chỉ kích hoạt vài expert cho mỗi token. Trước Mixtral, MoE chủ yếu xuất hiện ở các hệ thống đóng hoặc nghiên cứu (GShard, Switch); chưa có LLM mở dùng MoE đạt chất lượng hàng đầu. Mixtral lấp khoảng trống này, đồng thời phát hành trọng số dưới giấy phép Apache 2.0.

## 2. Ý tưởng chính

Mixtral 8x7B có **cùng kiến trúc với Mistral 7B**, chỉ khác ở chỗ khối FFN của *mọi* lớp được thay bằng 8 FFN (8 expert). Với mỗi token và mỗi lớp, một router chọn 2 trong 8 expert, chạy hai expert đó và cộng đầu ra có trọng số. Vì token mỗi lớp chọn khác nhau, mỗi token "truy cập" tới 47B tham số nhưng chỉ *dùng* 13B tham số hoạt động. Kết quả: ngang hoặc hơn Llama 2 70B và GPT-3.5 ở hầu hết benchmark, vượt hẳn ở toán, code, đa ngôn ngữ, với khoảng 5 lần ít tham số hoạt động hơn.

## 3. Phương pháp chi tiết

### 3.1 Kiến trúc (Bảng 1 của paper)

| Tham số | Giá trị |
|---|---|
| dim ($d_{model}$) | 4096 |
| n_layers | 32 |
| head_dim | 128 |
| hidden_dim (FFN) | 14336 |
| n_heads | 32 |
| n_kv_heads | 8 (GQA, xem bài 15) |
| context_len | 32768 |
| vocab_size | 32000 |
| num_experts | 8 |
| top_k_experts | 2 |

Paper nói Mixtral dùng các cải tiến của Mistral 7B, với điểm khác biệt: hỗ trợ ngữ cảnh *dày đủ* (fully dense) 32k token, và FFN là lớp MoE.

### 3.2 Lớp MoE

Gọi $n$ là số expert. Đầu ra của lớp MoE với đầu vào $x$:

$$\sum_{i=0}^{n-1}G(x)_i\cdot E_i(x),\qquad G(x):=\mathrm{Softmax}(\mathrm{TopK}(x\cdot W_g)).$$

Toán tử $\mathrm{TopK}(\ell)_i=\ell_i$ nếu $\ell_i$ nằm trong K giá trị lớn nhất, ngược lại bằng $-\infty$. Do $e^{-\infty}=0$, softmax chỉ phân phối trọng số cho K expert được chọn, nên các expert còn lại không cần tính. Mixtral dùng $K=2$ và mỗi expert là một khối SwiGLU (bài 11):

$$\boxed{y=\sum_{i=0}^{n-1}\mathrm{Softmax}\big(\mathrm{Top2}(x\cdot W_g)\big)_i\cdot\mathrm{SwiGLU}_i(x)}$$

**Khác Switch:** Switch lấy softmax trên *toàn bộ* N logit rồi chọn top-1 và nhân xác suất đó; Mixtral chọn top-2 trước rồi softmax *chỉ trên hai logit* nên hai trọng số cộng lại bằng 1. Paper cũng so với GShard: Mixtral thay *tất cả* khối FFN (GShard thay xen kẽ) và dùng cách chọn expert thứ hai đơn giản hơn. Paper không nhắc loss cân bằng tải nào (kết quả huấn luyện chi tiết không được công bố).

### 3.3 Tham số tổng và tham số hoạt động

- **Sparse (tổng) parameter count:** tăng theo $n$; quyết định bộ nhớ khi phục vụ (47B).
- **Active parameter count:** tăng theo $K$; tỉ lệ thuận với chi phí tính toán mỗi token (13B).

Kiểm tra bằng số (tự tính từ Bảng 1): một expert SwiGLU có $3\cdot4096\cdot14336\approx176{,}2$M tham số. Tám expert × 32 lớp ≈ 45,1B. Attention mỗi lớp: $W_Q,W_O$ mỗi cái $4096^2$ (16,8M), $W_K,W_V$ mỗi cái $4096\times1024$ (4,2M), tổng khoảng 41,9M; 32 lớp ≈ 1,34B. Embedding và đầu ra: $2\times32000\times4096\approx0{,}26$B (nếu không buộc trọng số). Tổng ≈ 46,7B, khớp "47B". Tham số hoạt động: 2 expert × 32 lớp ≈ 11,3B, cộng 1,34B + 0,26B ≈ 12,9B, khớp "13B".

### 3.4 Chạy hiệu quả

Paper nêu hai hướng cài đặt: (i) **MegaBlocks** (Gale et al.) viết lại phép tính FFN của MoE thành nhân ma trận thưa-lớn, xử lý tự nhiên trường hợp mỗi expert nhận số token khác nhau; (ii) **Expert Parallelism (EP):** mỗi GPU giữ một số expert, token được gửi tới GPU chứa expert của nó rồi gửi kết quả về. EP gây khó khăn cân bằng tải. Nhóm tác giả đóng góp thay đổi vào vLLM (có MegaBlocks kernel) để chạy Mixtral với stack mã nguồn mở.

### 3.5 Pseudo-code lớp MoE top-2 (theo công thức của paper, viết lại)

```python
def mixtral_moe(x, W_g, experts):           # x: [T, d]; experts: 8 SwiGLU
    logits = x @ W_g                         # [T, 8]
    top_val, top_idx = logits.topk(2, -1)    # [T, 2]
    w = softmax(top_val, -1)                 # chỉ softmax trên 2 logit được chọn
    y = zeros_like(x)
    for e in range(8):
        tok, slot = (top_idx == e).nonzero(as_tuple=True)   # token chọn expert e
        if tok.numel():
            y[tok] += w[tok, slot, None] * experts[e](x[tok])
    return y
```

Khác Switch: không có capacity cố định hay token drop được mô tả trong paper; phần này được MegaBlocks xử lý (token variable-length).

## 4. Thí nghiệm và kết quả chính

**Thiết lập.** Tiền huấn luyện với ngữ cảnh 32k, dữ liệu đa ngôn ngữ (tăng tỉ trọng đa ngôn ngữ so với Mistral 7B). Tác giả chạy lại mọi benchmark của Llama bằng pipeline đánh giá riêng của họ để so sánh công bằng. Instruct: SFT rồi DPO (Direct Preference Optimization) trên dữ liệu ưu tiên cặp.

### 4.1 So với họ Llama (Bảng 2)

| Mô hình | Tham số hoạt động | MMLU | HellaSwag | WinoG | PIQA | Arc-e | Arc-c | NQ | TriviaQA | HumanEval | MBPP | MATH | GSM8K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaMA 2 7B | 7B | 44.4 | 77.1 | 69.5 | 77.9 | 68.7 | 43.2 | 17.5 | 56.6 | 11.6 | 26.1 | 3.9 | 16.0 |
| LLaMA 2 13B | 13B | 55.6 | 80.7 | 72.9 | 80.8 | 75.2 | 48.8 | 16.7 | 64.0 | 18.9 | 35.4 | 6.0 | 34.3 |
| LLaMA 1 33B | 33B | 56.8 | 83.7 | 76.2 | 82.2 | 79.6 | 54.4 | 24.1 | 68.5 | 25.0 | 40.9 | 8.4 | 44.1 |
| LLaMA 2 70B | 70B | 69.9 | 85.4 | 80.4 | 82.6 | 79.9 | 56.5 | 25.4 | 73.0 | 29.3 | 49.8 | 13.8 | 69.6 |
| Mistral 7B | 7B | 62.5 | 81.0 | 74.2 | 82.2 | 80.5 | 54.9 | 23.2 | 62.5 | 26.2 | 50.2 | 12.7 | 50.0 |
| **Mixtral 8x7B** | **13B** | **70.6** | 84.4 | 77.2 | 83.6 | 83.1 | 59.7 | 30.6 | 71.5 | **40.2** | **60.7** | **28.4** | 74.4 |

(Đơn vị %.) Mixtral thua Llama 2 70B ở HellaSwag, WinoGrande, TriviaQA (và đọc hiểu, theo paper) nhưng vượt rõ ở code và toán: HumanEval 40.2 so với 29.3, MATH 28.4 so với 13.8.

### 4.2 So với GPT-3.5 (Bảng 3)

| | Llama 2 70B | GPT-3.5 | Mixtral 8x7B |
|---|---|---|---|
| MMLU | 69.9 | 70.0 | 70.6 |
| HellaSwag (10-shot) | 87.1 | 85.5 | 86.7 |
| ARC Challenge (25-shot) | 85.1 | 85.2 | 85.8 |
| WinoGrande (5-shot) | 83.2 | 81.6 | 81.2 |
| MBPP (pass@1) | 49.8 | 52.2 | 60.7 |
| GSM-8K (5-shot) | 53.6 | 57.1 | 58.4 |
| MT Bench (Instruct) | 6.86 | 8.32 | 8.30 |

Các điều kiện đánh giá khác với paper Llama 2: MBPP dùng tập con đã kiểm tay; TriviaQA không cung cấp ngữ cảnh Wikipedia.

### 4.3 Đa ngôn ngữ (Bảng 4, tập trung một số cột)

Mixtral vượt Llama 2 70B ở Pháp, Đức, Tây Ban Nha, Ý. Ví dụ ARC-c: Pháp 58.2 so với 49.9, Đức 54.3 so với 47.3; MMLU tiếng Đức 71.5 so với 64.2.

### 4.4 Ngữ cảnh dài

Truy xuất passkey (tìm một mã bí mật chèn ngẫu nhiên vào văn bản dài): **100%** chính xác với mọi độ dài ngữ cảnh và mọi vị trí. Perplexity trên tập con proof-pile giảm đơn điệu khi tăng độ dài ngữ cảnh.

### 4.5 Instruct, thiên lệch

Mixtral-Instruct đạt MT-Bench 8.30, "mô hình trọng số mở tốt nhất tháng 12/2023", Arena Elo 1121 (Claude-2.1 1117, GPT-3.5-Turbo tốt nhất 1117, Gemini Pro 1111, Llama-2-70b-chat 1077, theo ảnh chụp bảng xếp hạng LMSys ngày 22/12/2023). BBQ accuracy 56.0% so với 51.5% của Llama 2 70B (ít thiên lệch hơn); điểm cảm xúc BOLD cao hơn.

## 5. Phân tích định tuyến

Câu hỏi: các expert có chuyên môn hóa theo chủ đề không? Tác giả đo phân bố expert được chọn trên các tập con của The Pile (ArXiv, GitHub, PubMed, PhilPapers, StackExchange, Wikipedia, Gutenberg, DM Mathematics) ở lớp 0, 15, 31.

- **Không thấy chuyên môn hóa theo chủ đề rõ ràng.** Phân bố expert của ArXiv (LaTeX), sinh học (PubMed) và triết học (PhilPapers) rất giống nhau ở mọi lớp. Chỉ DM Mathematics hơi khác, nhiều khả năng do tính tổng hợp (synthetic) và độ phủ ngôn ngữ hẹp, nhất là ở lớp đầu/cuối.
- **Định tuyến mang tính cú pháp.** Token như `self` trong Python hay `Question` trong tiếng Anh thường vào cùng một expert dù thuộc nhiều token; token thụt lề trong code luôn gán cho cùng expert, rõ ở lớp đầu và cuối (nơi hidden state gần embedding vào/ra).
- **Cục bộ theo thời gian (temporal locality).** Token liên tiếp hay chọn cùng expert (Bảng 5). Mức ngẫu nhiên: 12,5% (expert đầu tiên giống nhau) và khoảng 46% (đầu hoặc thứ hai). Ví dụ đo được (expert lựa chọn đầu): lớp 0 khoảng 14%, lớp 15 khoảng 24-28%, lớp 31 khoảng 20-26%; với "đầu hoặc thứ hai": lớp 0 khoảng 45-50%, lớp 15 khoảng 62-67%, lớp 31 khoảng 44-54%. Tức ở lớp sâu lặp lại cao hơn ngẫu nhiên rõ rệt; lớp 0 gần ngẫu nhiên.

Hệ quả theo paper: tính cục bộ có thể làm quá tải một số expert khi dùng Expert Parallelism, nhưng cũng có thể khai thác để **cache** expert (như trong công trình offloading của Eliseev và Mazur).

**Về hiệu quả và bộ nhớ (Mục 3, "Size and Efficiency").** Chi phí tính toán tỉ lệ với tham số hoạt động (13B), còn chi phí bộ nhớ tỉ lệ với tham số tổng (47B, vẫn nhỏ hơn 70B). Paper thừa nhận SMoE có overhead do định tuyến và do tải thêm bộ nhớ khi chạy nhiều hơn một expert trên mỗi thiết bị, và phù hợp với tải theo batch có cường độ số học (arithmetic intensity) tốt. Ở phần giới thiệu, paper nói Mixtral cho "inference nhanh hơn ở batch nhỏ và throughput cao hơn ở batch lớn", nhưng không đưa số đo tốc độ.

## 6. Giới hạn

**Do tác giả nêu:**
- SMoE có overhead định tuyến và tải bộ nhớ khi nhiều expert trên một thiết bị; hợp với workload có batch lớn.
- Khó cân bằng tải khi dùng Expert Parallelism.
- Chi phí bộ nhớ vẫn theo 47B tham số.

**Do người viết nhận xét:**
- Paper không công bố dữ liệu huấn luyện, số token, loss phụ hay siêu tham số huấn luyện, nên không tái lập được; và không có ablation về số expert, K, hay MoE-so-với-dày ở cùng compute (không có đối chứng "Mistral 7B dày với cùng số token").
- Các tuyên bố tốc độ ("nhanh hơn ở batch nhỏ") không có số đo trong paper.
- "5 lần ít tham số hoạt động" so với Llama 2 70B không phải so sánh về bộ nhớ hay latency thực tế trên GPU nhỏ.
- Điểm cuối của benchmark do chính tác giả chạy lại, có khác biệt giao thức so với paper Llama 2 (đã nêu).
- Phân tích định tuyến chỉ nhìn ba lớp trên The Pile; kết luận "không chuyên môn hóa theo chủ đề" là quan sát, không phải bằng chứng cho mọi tầng.

## 7. Liên hệ với các bài khác

- **27 Switch:** top-1, loss cân bằng tải, capacity factor, token drop. Mixtral: top-2, thay mọi FFN, softmax sau TopK.
- **06 Mistral 7B:** kiến trúc nền; Mixtral thêm MoE. Mixtral dùng GQA (8 KV head, bài 15) và hỗ trợ ngữ cảnh dày 32k.
- **11 SwiGLU:** mỗi expert là một SwiGLU.
- **05 LLaMA, 02 GPT-3:** mốc so sánh chất lượng (Llama 2 70B, GPT-3.5).
- **48-50 Speculative decoding, Medusa, EAGLE:** với MoE, kiểm tra nhiều token cùng lúc (verify) kích hoạt nhiều expert hơn trong một bước, làm giảm lợi thế băng thông (suy luận của người viết, paper không bàn).
- **46 vLLM:** Mixtral được tích hợp vào vLLM nhờ MegaBlocks.
- **25 Jamba, 16 DeepSeek-V2:** kiến trúc MoE về sau.

## 8. Ứng dụng cho dự án (suy luận)

**Mức liên quan: thấp cho triển khai trực tiếp, trung bình để hiểu "active vs total params".**

1. **Chi phí batch 1 trên GPU ≤ 8 GB.** Với MoE, toàn bộ expert phải nằm trong bộ nhớ nếu muốn không offload. Ví dụ (tự tính, theo mô hình cỡ dự án: $d_{model}=1024$, $d_{ff}=3072$, SwiGLU, 24 lớp): một khối FFN dày ≈ 9,4M tham số/lớp, 226M cho 24 lớp (≈ 0,45 GB fp16). Nếu chuyển thành 8 expert top-2 thì FFN thành ≈ 1,81B tham số (≈ 3,6 GB fp16), vượt xa mục tiêu 0,3B/0,1B. Tham số hoạt động mỗi token chỉ gấp đôi mô hình dày (≈ 0,45B), tức latency decode batch 1 gần như bằng mô hình dày 2 lần FFN, nhưng bộ nhớ gấp 8 lần FFN. Kết luận: MoE tăng *dung lượng* chứ không giảm *footprint*, ngược với mục tiêu nén.
2. **Mixtral (decoder) không giúp ích cho khối token mixing hybrid** (GatedDeltaNet + full attention). Chỉ phần FFN có tương đồng; nếu coi FFN của bạn như một "expert đơn", thì Mixtral cho thấy 2 FFN chạy song song với cổng mềm vẫn tốt hơn một FFN lớn cùng FLOPs trong LLM (bằng chứng gián tiếp, vì paper không có ablation).
3. **Bài học về đo lường: tham số hoạt động ≠ bộ nhớ ≠ latency.** Khi bạn báo cáo mô hình cắt tỉa (0,3B, 0,1B), nên báo cả ba: số tham số, bộ nhớ trọng số + KV/state, và latency batch 1 thực đo, vì như Mixtral cho thấy ba thứ này có thể tách rời. Với dự án, KV-cache của phần full attention và state của GatedDeltaNet cũng nên báo cáo (xem 18-24).
4. **Quan sát "định tuyến mang tính cú pháp, token liên tiếp chọn cùng expert"** gợi ý: nếu phân tích kích hoạt FFN của mô hình OCR, có thể thấy nhóm neuron gắn với loại ký tự (chữ số, ký hiệu bảng). Đây là giả thuyết để kiểm chứng, không phải kết quả của paper.

## 9. Thí nghiệm gợi ý

1. **Phân tích tính thưa kích hoạt FFN:** với mô hình OCR dày, đo phân bố kích hoạt của SwiGLU theo loại token (chữ số, dấu câu, chữ). Nếu kích hoạt tập trung theo nhóm neuron, có thể dùng làm tiêu chí cắt tỉa neuron theo nhóm thay vì theo từng neuron.
2. **Đo bộ ba chỉ số:** với mỗi checkpoint cắt tỉa (0,8B, 0,3B, 0,1B), báo cáo số tham số, bộ nhớ đỉnh khi sinh 1.500 token với batch 1, và thời gian/token; để thấy đâu là nút thắt thật sự.
3. **(Tùy chọn) Mô phỏng tính cục bộ:** nếu có thời gian, cài top-2 router giả trên FFN của một lớp để đếm tần suất lặp lại expert giữa token liên tiếp, đối chiếu Bảng 5.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| SMoE layer | $y=\sum_i G(x)_iE_i(x)$ |
| Gating | $G(x)=\mathrm{Softmax}(\mathrm{TopK}(xW_g))$, K=2, n=8 |
| Expert | khối SwiGLU, hidden 14336, dim 4096 |
| Tham số tổng / hoạt động | 47B (bộ nhớ) / 13B (FLOPs mỗi token) |
| Tham số một expert (tự tính) | $3\cdot4096\cdot14336\approx176$M |
| Context | 32k token dày đủ, GQA 8 KV head |
| Locality | token i và i+1 chọn cùng expert: lớp sâu cao hơn ngẫu nhiên (12,5% / khoảng 46%) |
| Kết quả then chốt | MMLU 70.6, HumanEval 40.2, MATH 28.4, GSM8K 74.4; MT-Bench Instruct 8.30 |
| Giấy phép | Apache 2.0 |

## 11. Kết luận ngắn

Mixtral 8x7B chứng minh một LLM MoE mở có thể đạt hoặc vượt Llama 2 70B và GPT-3.5 trên hầu hết benchmark chỉ với 13B tham số hoạt động (47B tổng), đặc biệt mạnh ở toán, code và đa ngôn ngữ. Điều thú vị về cơ chế là định tuyến không chuyên môn hóa theo chủ đề mà theo cú pháp và có tính cục bộ theo thời gian. Với dự án của bạn, bài học chính không phải dùng MoE mà là phân biệt rõ tham số hoạt động, bộ nhớ và latency khi đánh giá mô hình cắt tỉa.

Liên kết gốc: https://arxiv.org/abs/2401.04088
