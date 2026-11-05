# An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Acceleration for VLLM Inference (FastV) (2024)

**Tác giả:** Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, Baobao Chang (Peking University, Alibaba Group).  
**Venue / arXiv:** arXiv:2403.06764 (ngày đầu 11/03/2024; bản v3 02/09/2024). Paper không nêu venue trong bản chúng tôi đọc.  
**Nguồn:** https://arxiv.org/abs/2403.06764  
**Chủ đề:** Cắt bỏ visual token trong LVLM khi suy luận, không cần huấn luyện; phân tích "attention kém hiệu quả trên ảnh" ở các lớp sâu.  
**Vị trí trong lộ trình:** Đọc trước: 01 (Attention), 34/35 (LLaVA, LLaVA-1.5), 55 (StreamingLLM, attention sink). Bài này là một phương pháp **tối ưu suy luận cho VLM** nhắm vào số visual token; đối chiếu với 56 (H2O, loại bỏ KV theo điểm attention) và với hướng nén token ở encoder (41 GOT, 43 LightOnOCR gộp patch 2×2).

> Ghi chú học tập bằng tiếng Việt, **không phải bản dịch**. Phần "Ứng dụng cho dự án" là suy luận của chúng tôi. Số đánh dấu "tự tính" do chúng tôi tính (giả định cấu hình như ghi chú), không phải số của paper.

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

LVLM (large vision-language model, kiểu LLaVA) đưa ảnh vào LLM dưới dạng **chuỗi visual token**: vision encoder (CLIP-ViT) trích đặc trưng, làm phẳng, chiếu bằng linear (hoặc Q-Former/cross-attention) về chiều của LLM. Ảnh 336×336 của LLaVA-1.5 thành 576 token; 672×672 thành 2.304 token; video 8 khung hình của Video-LLaVA thành 2.048 token. Độ phân giải tăng thì số token tăng (xấp xỉ bình phương theo cạnh ảnh), kéo theo chi phí tính toán và bộ nhớ.

Vấn đề: LLM xử lý token ảnh **như token văn bản** ở mọi lớp. Các kỹ thuật tối ưu LLM (FlashAttention, vLLM/PagedAttention) giữ nguyên kết quả; các kỹ thuật dựa trên mẫu attention (StreamingLLM, FastGen) chỉ thử trên LLM văn bản. Trước FastV, việc cắt token chủ yếu được nghiên cứu cho ViT (EViT, SPViT...) hoặc cần fine-tune (LLaMA-VID).

*Thuật ngữ:* **FLOPs** = số phép tính dấu phẩy động; **prefill** = bước xử lý toàn bộ prompt (gồm visual token) một lần trước khi sinh; **decode** = bước sinh từng token. **Attention sink / anchor token** = vài token (thường ở đầu chuỗi) nhận lượng attention rất lớn bất kể nội dung (xem 55).

## 2. Ý tưởng chính

Tác giả đo attention trong LLaVA-1.5 và thấy: dù ảnh chiếm ~64% số token đầu vào, từ **sau lớp 2** ảnh nhận **rất ít** attention (ở lớp sâu, hiệu quả attention của token ảnh chỉ bằng ~1/472 token system prompt). Họ suy ra thông tin ảnh đã được "gom" vào vài token neo không phải ảnh ở các lớp nông, nên các lớp sau không còn cần đọc token ảnh nữa. FastV tận dụng điều này: **ở lớp K, xếp hạng token ảnh theo attention trung bình nhận được, bỏ R% token ít được chú ý nhất**; mọi lớp sau chỉ xử lý số token còn lại. Vì token bị bỏ hẳn, tiết kiệm cả **attention lẫn FFN** của các lớp sau, không cần huấn luyện lại. Với LLaVA-1.5-13B: cắt 50% token sau lớp 2 giảm ~45% FLOPs mà gần như không mất điểm.

## 3. Phương pháp chi tiết

### 3.1 Đo "attention kém hiệu quả"

Một LVLM nhận 4 loại token: system prompt (sys), ảnh (img), instruction của người dùng (ins), token sinh ra (out). Với token thứ \(i\) trong câu trả lời, ở lớp \(j\), gọi \(\alpha^{i,j}_{type}\) là tổng xác suất attention nó dành cho từng loại:

$$
\alpha^{i,j}_{sys}+\alpha^{i,j}_{img}+\alpha^{i,j}_{ins}+\alpha^{i,j}_{out}=1
$$

**Phân bổ attention (attention allocation)** của một loại ở lớp \(j\) (cộng trên \(n\) token đầu ra):

$$
\lambda^{j}_{type}=\sum_{i=1}^{n}\alpha^{i,j}_{type}
$$

**Hiệu quả attention (attention efficiency)** = attention trung bình trên mỗi token của loại đó:

$$
\boxed{\epsilon^{j}_{img}=\frac{\sum_{i=1}^{n}\alpha^{i,j}_{img}}{|img|}}
$$

Trong đó \(|img|\) là số token ảnh; kết quả lấy trung bình trên các head và trên N = 1.000 cặp ảnh-văn bản (mẫu từ Flickr30K, PCA-Bench, A-OKVQA, MMMU), mô hình LLaVA-1.5-7B. "Lớp nông" = 2 lớp đầu, "lớp sâu" = 30 lớp còn lại.

**Phát hiện (paper):**
1. Mất cân bằng attention tăng theo độ sâu. Ở lớp nông token sinh ra chủ yếu nhìn các token sinh trước đó; ở lớp sâu chúng nhìn vào system prompt.
2. **Token ảnh có hiệu quả thấp nhất ở mọi độ sâu.** Ở lớp sâu, system prompt có hiệu quả gấp **472 lần** token ảnh và chiếm **85%** tổng attention. Số trong hình 3 (độ dài trung bình): sys 35, ảnh 576, ins 135, out 150 token; tỉ lệ hiệu quả lớp sâu (so ảnh = 1×): sys 472×, ins 3×, out 12,8×; lớp nông: sys 2×, ins 3×, out 9×. Phần giới thiệu nêu: ở lớp sâu, attention của ảnh chỉ bằng 0,21% của system prompt, còn ở 2 lớp đầu thì tới 50%.
3. Bản đồ attention lớp sâu có các **vạch dọc** (một số token luôn nhận nhiều attention suốt quá trình giải mã) ở system prompt.

**Giả thuyết của tác giả (không được chứng minh trực tiếp):** tín hiệu thị giác dư thừa nhiều nên ở lớp nông self-attention gom thông tin ảnh có liên quan tới instruction vào một số token "neo" (không phải token ảnh); ở lớp sâu attention tập trung vào các token neo, bỏ qua token ảnh. Phù hợp với kết quả "label words are anchors" (Wang et al. 2023).

### 3.2 Thuật toán FastV

Hai siêu tham số: **lớp lọc K** và **tỉ lệ lọc R%**. Hàm xếp hạng \(\phi\):
- \(\phi_{attn}\): với mỗi token, điểm = **attention trung bình nó nhận được từ tất cả token khác** (dùng trong thí nghiệm chính, tính ở lớp K).
- \(\phi_{rand}\): bỏ ngẫu nhiên; dùng trong trường hợp cực trị K = 0 (cắt token ảnh ngay trước khi vào LLM, chưa có attention để xếp hạng).

```
# prefill, hidden: [n_tokens, d]; img_idx: chỉ số các token ảnh
for layer l in 0 .. T-1:
    if l == K:
        A = attention_probs(layer K)            # [heads, n, n] (đã trung bình theo head)
        score[j] = mean_i A[i, j]               # attention trung bình token j nhận từ các token khác
        drop = bottom_R%(score restricted to img_idx)
        keep = all tokens except drop
        hidden, positions = hidden[keep], positions[keep]   # các lớp sau chỉ thấy n_hat token
    hidden = transformer_block[l](hidden)       # attention + FFN trên tập token đã cắt
# decode: các token sinh ra chỉ attend vào tập token còn lại ở các lớp >= K
```

Chỉ token ảnh bị cắt; token văn bản được giữ. Plug-and-play: không huấn luyện lại, áp dụng cho bất kỳ LVLM dạng token nào (LLaVA, Qwen-VL, InstructBLIP, Video-LLaVA).

### 3.3 Ước lượng chi phí (công thức của paper)

Cho một lớp transformer với \(n\) token, hidden size \(d\), FFN trung gian \(m\):

$$
\text{FLOPs}_{layer}=4nd^{2}+2n^{2}d+2ndm
$$

(\(4nd^2\): bốn phép chiếu Q, K, V, O; \(2n^2 d\): QK^T và AV; \(2ndm\): FFN với hai ma trận.) Gọi \(\hat n=(1-R)\,n\) sau lớp K, tổng \(T\) lớp, mức giảm FLOPs lý thuyết:

$$
\boxed{1-\frac{K(4nd^2+2n^2d+2ndm)+(T-K)(4\hat n d^2+2\hat n^2 d+2\hat n dm)}{T(4nd^2+2n^2d+2ndm)}}
$$

Trực giác: giữ nguyên K lớp đầu, còn \(T-K\) lớp sau chạy với \(\hat n\) token; phần \(n^2\) giảm nhanh hơn (bình phương), phần \(d^2\) giảm tuyến tính. Công thức coi \(\hat n=(1-R)n\) trên *toàn bộ* \(n\); thực tế chỉ token ảnh bị bỏ nên mức giảm thực hơi khác (xem 4).

**Ví dụ tính tay (tự tính, cấu hình cỡ dự án: d = 1024, m = 3072, T = 24 lớp, n = 2.200 token gồm 2.100 token ảnh + ~100 văn bản; chỉ phần LLM prefill, bỏ qua ViT):**

| K | R | Giảm FLOPs prefill LLM (token ảnh bị cắt) |
|---|---|---|
| 2 | 50% | ~50,6% |
| 2 | 75% | ~71,2% |
| 2 | 90% | ~82,1% |
| 4 | 75% | ~64,8% |
| 6 | 75% | ~58,3% |

Một lớp đầy đủ ở n = 2.200 tốn ~33 GFLOP, tức ~791 GFLOP toàn bộ 24 lớp theo công thức trên; phần \(2n^2d\) chiếm ~30%. Con số này **không** gồm vision encoder và giả định attention đầy đủ ở mọi lớp (xem 8 về decoder lai).

**Chi phí bộ nhớ KV của token ảnh:** \(\text{KV bytes}=2\cdot n\cdot L_{attn}\cdot n_{kv}\cdot d_h\cdot b\) (K và V; \(L_{attn}\) số lớp có KV cache; \(b\) byte mỗi phần tử). FastV không bàn trực tiếp KV cache; theo cách ta hiểu cơ chế, token đã bị bỏ sẽ không có K/V ở các lớp ≥ K (suy luận của chúng tôi). Paper chỉ báo cáo bộ nhớ GPU tổng (Bảng 4, mục 4) mà không phân rã.

### 3.4 So sánh với "huấn luyện với ít token ảnh hơn"

Cách thay thế: thêm average pooling (stride 2) sau CLIP khi huấn luyện LLaVA-1.5-7B để giảm 50% token ngay từ đầu. Paper thử cách này để so sánh công bằng (mục 5).

## 4. Thí nghiệm và kết quả chính

**Mô hình:** LLaVA-1.5-7B/13B, QwenVL-Chat-7B, InstructBLIP-Vicuna-13B (ảnh); Video-LLaVA (video). Giải mã tham lam (greedy). **Tác vụ:** captioning (Nocaps, Flickr30k; CIDEr), VQA (A-OKVQA multiple choice; OCR-VQA, Rouge-L), suy luận đa phương thức (MMMU; PCA-Bench), video QA (TGIF, MSVD, MSRVTT; dùng 1K mẫu đầu mỗi bộ, chấm bằng GPT), và các benchmark chi tiết (MME, SeedBench, MM-Vet, SciQA-IMG, AI2Diagram).

**Bảng 1 (trích; "Ratio" = FLOPs so với baseline):**

| Model | K | R | FLOPs (B) | Ratio | Nocaps | Flickr30k | A-OKVQA | MMMU | Avg |
|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | Baseline | | 99.3 | 100% | 99.8 | 67.9 | 76.7 | 34.8 | 69.8 |
| | 2 | 50% | 54.6 | 55% | 99.7 | 67.5 | 77.0 | 34.4 | 69.7 |
| | 2 | 75% | 32.8 | 33% | 94.6 | 63.6 | 75.5 | 34.8 | 67.1 |
| | 2 | 90% | 19.9 | 20% | 72.1 | 43.7 | 70.1 | 35.0 | 55.2 |
| | 5 | 50% | 59.6 | 60% | 99.2 | 67.9 | 76.8 | 34.3 | 69.6 |
| | 0 | 50% | 51.6 | 52% | 100.9 | 65.5 | 75.3 | 34.3 | 69.0 |
| | 0 | 75% | 28.8 | 29% | 27.2 | 61.4 | 72.8 | 35.1 | 49.1 |
| | 0 | 90% | 18.9 | 19% | 7.0 | 53.2 | 66.8 | 34.7 | 40.4 |
| LLaVA-1.5-13B | Baseline | | 154.6 | 100% | 102.8 | 73.0 | 82.0 | 36.4 | 73.6 |
| | 2 | 50% | 84.6 | 55% | 103.1 | 73.4 | 81.0 | 36.7 | 73.6 |
| | 2 | 75% | 50.2 | 32% | 100.5 | 72.5 | 80.9 | 38.1 | 73.0 |
| | 2 | 90% | 29.7 | 19% | 87.9 | 62.0 | 75.0 | 36.3 | 65.3 |
| QwenVL-Chat-7B | Baseline | | 71.9 | 100% | 94.9 | 72.5 | 75.6 | 35.8 | 69.7 |
| | 2 | 50% | 39.5 | 55% | 94.4 | 71.4 | 75.3 | 35.6 | 69.2 |
| | 2 | 75% | 24.4 | 34% | 90.5 | 67.0 | 75.1 | 35.3 | 67.0 |

Kết luận paper: **K = 2, R = 50% giảm khoảng 45% FLOPs** mà gần như không mất hiệu năng cho các mô hình trên; đánh đổi hiệu năng-FLOPs "điều chỉnh được" bằng cách giảm K, tăng R. (Con số 55% của paper cao hơn một chút so với ~52% theo công thức Eq. 5 cho 7B/n≈700 (tự tính); chúng tôi không tái lập chính xác cách họ tính FLOPs thực.)

**Độ trễ thực (Bảng 4; GPU A40; tác vụ A-OKVQA, chỉ xuất 1 chữ cái đáp án để loại ảnh hưởng độ dài đầu ra):**

| Model | Tổng thời gian | GPU memory | Score | Latency/mẫu |
|---|---|---|---|---|
| LLaVA-1.5-7B | 6:34 | 19G | 76,7 | 0,344 s |
| + FastV (K=0, R=50%) | 4:23 | 16G | 75,3 | 0,230 s |
| LLaVA-1.5-13B | 10:17 | 38G | 82,0 | 0,539 s |
| + FastV (K=0, R=50%) | 6:30 | 30G | 80,5 | 0,341 s |

Tức 13B + FastV (0,341 s) có độ trễ tương đương 7B gốc (0,344 s) nhưng điểm cao hơn (80,5 so với 76,7), đúng như tuyên bố của tác giả. (Lưu ý: bảng này dùng K = 0, tức cắt **ngẫu nhiên** trước LLM.) Giảm tốc độ ≈ 33-37% latency (tự tính: 0,230/0,344 = 0,67; 0,341/0,539 = 0,63).

**Chi tiết các tác vụ khó:**
- *MME (LLaVA-1.5-13B):* tổng 1827,75 -> 1849,68 với FastV (K=2, R=50%); OCR 125,0 -> 132,5; Code 47,5 -> 57,5.
- *InstructBLIP-13B:* suy giảm nhiều hơn LLaVA với cùng cấu hình (ví dụ SeedBench 52,11 -> 50,41); đặt K = 5 thì gần như hết (51,69). Tác giả cho rằng Q-Former đã giảm token nên mất thông tin trực tiếp hơn.
- *PCA-Bench / OCR-VQA (LLaVA-1.5-13B, Bảng 5):* FastV (K=2, R=50%, FLOPs 84,6B): G-PCA mở 0,340 (gốc 0,333), đóng 0,368 (gốc 0,353); OCR-VQA Rouge-L 0,55 (gốc 0,55). K=2, R=75% (50,2B): 0,320 / 0,323, Rouge-L 0,54. K=0, R=50% (78,9B): Rouge-L 0,49.
- *Video-QA (Video-LLaVA, Bảng 6, GPT-eval):* gốc Acc/Score trung bình 0,48/3,3; FastV (K=2, R=50%, FLOPs 52,3%) 0,49/3,3; (K=5, R=50%, 57,1%) 0,49/3,4. Tác giả nhấn mạnh video dư thừa hơn (2.048 token), nên FastV còn cải thiện nhẹ TGIF (Acc 0,18 -> 0,21).

## 5. Phân tích / ablation

**Bảng 7 (LLaVA-1.5-7B huấn luyện lại làm baseline, Nocaps / Flickr30k / A-OKVQA / MMMU):**

| Dòng | Cấu hình | Nocaps | Flickr30k | A-OKVQA | MMMU |
|---|---|---|---|---|---|
| | Baseline (retrained) | 100,3 | 70,2 | 78,5 | 34,5 |
| (a) | Huấn luyện với 50% token ảnh (pooling) | 98,5 | 68,5 | 76,8 | 33,5 |
| (b) | FastV K=2, R=50% | 100,1 | 70,0 | 78,4 | 34,6 |
| (c) | FastV K=2, R=50%, **ngẫu nhiên** | 99,5 | 68,3 | 78,2 | 34,2 |
| (d) | FastV bỏ token **system prompt** | 89,2 | 64,3 | 69,2 | 33,8 |
| (e) | bỏ **nửa đầu** system prompt | 17,5 | 27,8 | Failed | Failed |
| (f) | bỏ token **instruction** | 77,3 | 50,1 | 56,5 | 29,5 |
| (g) | áp mẫu sparse attention kiểu StreamingLLM | 13,2 | 21,4 | Failed | Failed |

Rút ra:
- (a) vs (b): giảm token bằng pooling lúc huấn luyện **kém hơn** cắt lúc suy luận bằng FastV (cùng 50%), ít nhất trong cấu hình này.
- (b) vs (c): xếp hạng theo attention tốt hơn ngẫu nhiên, **nhưng chênh nhỏ ở R = 50%, K = 2** (Flickr30k 70,0 vs 68,3; Nocaps 100,1 vs 99,5). Nhận xét của chúng tôi: ở mức cắt vừa phải, việc "cắt sau lớp 2" quan trọng hơn tiêu chí xếp hạng. Sự quan trọng của tiêu chí chỉ rõ hơn ở R cao.
- (d), (e), (f): cắt token **không phải ảnh** gây tụt mạnh; nửa đầu system prompt (các token neo) là then chốt, đúng với phát hiện về attention sink của StreamingLLM (55). Nhưng (g) cho thấy **áp thẳng pattern StreamingLLM** (chỉ giữ neo + cửa sổ gần) vào LVLM làm sập hoàn toàn, tức token ảnh khác về bản chất với token văn bản.
- **Ảnh hưởng của K và R (hình 7, LLaVA-1.5-13B trên OCR-VQA, không có số trong phần chữ):** K nhỏ thì giảm R giúp tăng điểm rõ; K lớn thì đổi R ít ảnh hưởng, chứng tỏ token ảnh dư thừa ở lớp sâu. Cũng thấy trong Bảng 1: cùng R = 75% mà K = 0 (random) cho Nocaps 27,2, còn K = 2 cho 94,6 (7B) tức **lớp 0-1 cần đủ token ảnh**.
- Trong phần giới thiệu, paper còn nêu thí nghiệm huấn luyện với các độ phân giải đặc trưng khác nhau bằng pooling để chứng minh FastV cho phép dùng ảnh phân giải cao hơn với cùng chi phí; **chúng tôi không tìm thấy bảng số tương ứng trong văn bản trích xuất**, nên không báo cáo.

## 6. Giới hạn

**Do tác giả nêu/ngầm nêu:**
- InstructBLIP (dùng Q-Former đã nén sẵn) suy giảm nhiều hơn với cùng K, R; phải tăng K.
- R cao (≥ 90%) làm tụt rõ; cần chọn K, R theo tác vụ.

**Chúng tôi nhận xét:**
- **Chỉ giảm FLOPs lý thuyết** là chính; số đo độ trễ chỉ có một cấu hình (A40, K = 0 ngẫu nhiên, đáp án 1 chữ cái). Với sinh văn bản dài thì giải mã thống trị (chi phí mỗi token gần như không phụ thuộc số visual token, trừ phần KV), nên lợi ích suy luận thực sự có thể nhỏ hơn (xem 8.2).
- Xếp hạng bằng attention trung bình **cần xác suất attention tường minh ở lớp K**, khó chạy với FlashAttention/kernel fused (cần tính riêng ở một lớp); paper không bàn đến.
- Đánh giá chủ yếu là caption/VQA ngắn, đáp án ngắn; tác vụ cần đọc chữ dày (OCR toàn trang) chỉ được thử gián tiếp qua OCR-VQA (câu trả lời ngắn).
- Thống kê attention lấy trung bình trên các bước giải mã và nhiều ảnh; nó **ẩn** việc ở từng bước token nào được cần. Giả thuyết "thông tin đã gom vào token neo" chưa được kiểm chứng (ví dụ bằng can thiệp nhân quả).
- Chỉ kiểm tra với LLM attention chuẩn (Vicuna/Qwen) và LLaVA 576 token; chưa kiểm với ảnh độ phân giải cao hàng nghìn token hay kiến trúc lai/linear attention.
- Cắt token hoàn toàn trên cả tập được chọn lúc prefill: không thể "phục hồi" nếu bước giải mã sau cần vùng ảnh đã bị bỏ.
- Phần cắt không áp dụng cho vision encoder: chi phí ViT (trên toàn bộ patch) không đổi.

## 7. Liên hệ với các bài khác

- **55 StreamingLLM (attention sink):** FastV thấy token neo ở system prompt nhận ~85% attention, ablation (e) xác nhận chúng thiết yếu; nhưng (g) cho thấy cách của StreamingLLM không dùng trực tiếp được cho token ảnh.
- **56 H2O:** cũng dùng điểm attention tích luỹ để loại token, nhưng ở KV cache khi sinh; FastV loại token ngay trong prefill (và áp dụng riêng cho token ảnh).
- **35 LLaVA-1.5, 34 LLaVA:** mô hình thí nghiệm chính, 576 token ảnh.
- **33 BLIP-2 (Q-Former):** InstructBLIP dùng Q-Former nên đã mất token trước, FastV hại hơn.
- **37 Qwen2-VL, 36 InternVL 1.5:** token ảnh tăng theo độ phân giải; đây là kiểu mô hình mà FastV muốn giúp.
- **41 GOT, 43 LightOnOCR:** giảm token ở **phía encoder** (nén 256 token, gộp patch 2×2) thay vì cắt trong LLM; hai hướng bổ sung nhau.
- **44/45 FlashAttention, 46 vLLM:** tối ưu "chính xác" không đổi kết quả; FastV là xấp xỉ có thể mất độ chính xác.
- **24 Gated DeltaNet / 18-23 linear attention:** FastV giả định có ma trận attention softmax để xếp hạng; không áp dụng trực tiếp (xem 8).

## 8. Ứng dụng cho dự án (suy luận)

Đối tượng: OCR VLM (trang -> ~1.500 token đầu ra), decoder lai GatedDeltaNet + full attention (chỉ một số lớp có softmax attention), ViT 12 block, batch-1 trên GPU ≤ 8 GB.

**8.1 Prefill: có lợi.** Mỗi trang có ~1,6k-2,1k visual token (tự tính; xem 43). Theo bảng ở mục 3.3, cắt 50-75% token ảnh sau lớp thứ 2 có thể giảm ~50-71% FLOPs prefill của **LLM** (24 lớp, d=1024); nhưng với batch-1 ngắn, prefill chỉ là phần nhỏ thời gian một trang (giải mã 1.500 token chiếm đa số, xem 8.2), và **ViT không được giảm**.

**8.2 Decode batch-1: lợi ích nhỏ.** Giải mã bị giới hạn băng thông: mỗi token đọc toàn bộ trọng số (~1,6 GB cho 0,8B bf16; ~0,6 GB cho 0,3B) cộng KV cache. Với decoder lai chỉ ~1/4 lớp có KV cache (giả định 6/24 lớp, 8 KV head × 128, bf16): KV ≈ 2 × 8 × 128 × 2 byte × 6 lớp ≈ 24,6 KB/token. 2.100 token ảnh ≈ **52 MB**; cắt 75% sau lớp thứ 2 (chỉ các lớp có KV ở sau lớp 2) tiết kiệm tối đa ~39 MB mỗi bước, tức **dưới ~3% byte đọc mỗi bước** so với 1,6 GB trọng số (tự tính, mọi giả định cấu hình đều của chúng tôi). Vậy FastV hầu như không giảm độ trễ giải mã batch-1 của model này; đòn bẩy chính vẫn là giảm tham số (cắt lớp/neuron/vocab). FastV hợp hơn nếu bạn đo thấy prefill hoặc bộ nhớ đỉnh là nút thắt (ví dụ trang lớn với nhiều nghìn token).

**8.3 Cấu trúc lai: khó áp dụng trực tiếp.**
- Lớp GatedDeltaNet **không có ma trận attention softmax**, nên không có "attention trung bình nhận được" để xếp hạng. Phải lấy điểm từ **lớp full attention đầu tiên** (hoặc dùng một proxy khác, ví dụ norm của hidden state hay score từ một head thăm dò).
- Cắt token khỏi chuỗi của lớp linear attention thay đổi trạng thái hồi quy (state) mà các token sau đọc: không phải "chỉ bỏ cột trong ma trận attention" như ở softmax. Cách bỏ token ở lớp recurrent chưa được paper này (hay bài nào trong lộ trình) kiểm chứng.
- Vị trí (RoPE/2D) của token còn lại cần được giữ nguyên chỉ số gốc; paper không mô tả chi tiết cách xử lý position id.

**8.4 OCR khác ảnh tự nhiên: rủi ro lớn.** Phát hiện "token ảnh bị bỏ qua ở lớp sâu" được đo trên caption/VQA (câu trả lời ngắn, thông tin ảnh dư thừa). Với OCR toàn trang, mỗi vùng chữ **chứa thông tin duy nhất** và ở bước sinh dòng thứ 30 mô hình cần đúng các token vùng đó. Điểm attention **trung bình** tính một lần ở lớp K có thể xếp thấp các vùng mà sẽ cần ở sau. Bằng chứng gần nhất trong paper: OCR-VQA (K=2, R=50%) không mất (Rouge-L 0,55 = 0,55), nhưng đó là câu trả lời ngắn. Chúng tôi coi đây **giả thuyết cần kiểm chứng**, không phải kết luận.

**8.5 Tương tác với cắt tỉa/chưng cất.** Phát hiện của FastV tương thích với ý tưởng rằng các lớp sâu ít dùng thông tin ảnh thô -> có thể (i) cắt tỉa các head/neuron chuyên đọc token ảnh ở lớp sâu, (ii) khi chưng cất, tập trung teacher-forcing vào lớp nông là nơi "gom ảnh". Đây là liên tưởng, không phải kết quả của paper. Thước đo "attention efficiency" ε cũng dùng được như chẩn đoán rẻ: tính ε theo lớp của model gốc để quyết định lớp nào có thể cắt hoặc nơi nào visual token còn được dùng.

**8.6 Hướng thay thế.** Giảm token ở encoder (gộp 2×2 -> 3×3, hoặc pooling thích nghi theo mật độ chữ) tác động cả ViT và decoder, nhưng cần huấn luyện/chưng cất lại; FastV là miễn huấn luyện nhưng chỉ tác động LLM. Ablation (a)/(b) gợi ý cắt lúc suy luận có thể tốt hơn pooling cố định lúc huấn luyện trong LLaVA, chưa chắc đúng với OCR.

Mức liên quan: **trung bình**; hữu ích như công cụ chẩn đoán và cho prefill, nhưng không phải đòn bẩy chính cho độ trễ batch-1.

## 9. Thí nghiệm gợi ý

1. **Đo ε theo lớp** trên model của bạn (chỉ ở các lớp full attention) cho 3 loại token: system, visual, text đã sinh; xem tỉ lệ ε_img/ε_sys có giảm sâu như 1/472 không, và biết "K" hợp lý.
2. **FastV trên OCR toàn trang:** đặt K tại lớp full-attention đầu tiên, R ∈ {25, 50, 75}%, so tiêu chí attention vs ngẫu nhiên vs pooling cố định cùng số token; đo OlmOCR-Bench (nhóm Tables, Long Tiny Text, Multi Column) và tỉ lệ lặp. Dự đoán của chúng tôi: Long Tiny Text và Tables sụt trước.
3. **Đo thật batch-1:** thời gian prefill vs decode và byte KV đọc mỗi bước trước/sau cắt (với và không có ViT) để kiểm chứng ước tính "<3% byte" ở 8.2.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| Attention allocation | \(\lambda^j_{type}=\sum_i \alpha^{i,j}_{type}\) |
| Attention efficiency | \(\epsilon^j_{img}=\sum_i \alpha^{i,j}_{img}/\lvert img\rvert\) |
| Phát hiện | lớp sâu: sys gấp 472× và chiếm 85%; ảnh ~0,21% của sys |
| FastV | tại lớp K bỏ R% token ảnh có attention nhận trung bình thấp nhất |
| FLOPs/lớp | \(4nd^2+2n^2d+2ndm\) |
| Giảm FLOPs | \(1-\frac{K f(n)+(T-K) f(\hat n)}{T f(n)}\), \(\hat n=(1-R)n\) |
| Kết quả chính | K=2, R=50%: giảm ~45% FLOPs, hiệu năng gần như không đổi (13B) |
| Latency (A40, A-OKVQA) | 13B: 0,539 s -> 0,341 s (K=0, R=50%) |
| Ví dụ tự tính | d=1024, T=24, n=2.200: K=2, R=75% ⇒ −71% FLOPs prefill LLM |

## 11. Kết luận ngắn

FastV cho thấy trong LVLM kiểu LLaVA, token ảnh gần như bị "bỏ qua" từ sau lớp 2, nên có thể cắt một nửa (hoặc hơn) mà không huấn luyện lại, giảm ~45% FLOPs mà giữ điểm. Với dự án OCR của bạn, giá trị là một **công cụ chẩn đoán** (đo hiệu quả attention theo lớp) và một ứng viên giảm prefill, nhưng có ba cảnh báo lớn: (1) ở batch-1 thời gian chủ yếu nằm ở giải mã nên lợi ích bị giới hạn; (2) decoder lai không có attention softmax ở mọi lớp nên cần thay tiêu chí xếp hạng; (3) OCR dày chữ khác ảnh tự nhiên, cần kiểm chứng riêng trước khi tin vào mức dư thừa.

Link gốc: https://arxiv.org/abs/2403.06764
