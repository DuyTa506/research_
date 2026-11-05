# ShortGPT: Layers in Large Language Models are More Redundant Than You Expect (2024)

**Tác giả:** Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, Weipeng Chen (Baichuan Inc., ISCAS).  
**Venue / arXiv:** arXiv:2403.03853 (v3, 11/10/2024). Bản text không ghi tên hội nghị.  
**Nguồn:** https://arxiv.org/abs/2403.03853  
**Chủ đề:** Depth pruning (cắt cả layer) cho LLM, chỉ số Block Influence (BI), không cần gradient, không retrain.  
**Vị trí trong lộ trình:** Đọc trước: 26 LLM-Pruner, 29 SliceGPT (hai đối thủ "width" mà bài này đánh bại). Bài này mở đầu nhóm depth pruning (31–37): đưa ra heuristic đơn giản nhất ("layer nào có đầu ra giống đầu vào thì bỏ"). Đọc tiếp: 32 (cùng ý tưởng, thêm healing bằng QLoRA), 34 SLEB, 35 LaCo, 36 ReplaceMe (thay layer bị bỏ bằng ánh xạ tuyến tính).

> Tài liệu học tập tiếng Việt, ghi chú diễn giải, không phải bản dịch paper. Mọi số liệu lấy từ paper; phần "Ứng dụng cho dự án" là suy luận của chúng ta, paper không kiểm chứng trên OCR/VLM.

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

- **Structured pruning** (cắt có cấu trúc) = bỏ nguyên cả cụm tham số (kênh, head, layer) để mô hình thật sự nhỏ/nhanh trên GPU phổ thông, khác với unstructured (đặt từng trọng số = 0, khó tăng tốc).
- Trước bài này, các phương pháp cho LLM (LLM-Pruner dùng gradient để chọn nhóm coupled; SliceGPT dùng PCA để giảm chiều embedding) cắt theo chiều **rộng** (width). Chúng khá phức tạp và mất nhiều chất lượng ở tỉ lệ ~25%.
- **Pre-norm**: LayerNorm/RMSNorm đặt *trước* attention/FFN, đầu ra là $x_{l+1}=x_l+f_l(\text{norm}(x_l))$ (có đường cộng residual). Hầu hết LLM hiện đại (Llama, Mamba, RWKV) dùng pre-norm.
- Câu hỏi của paper: nếu mỗi layer chỉ "cộng thêm" một lượng nhỏ vào residual, có phải nhiều layer gần như không làm gì và bỏ được?

## 2. Ý tưởng chính

Paper quan sát (Figure 1) rằng bỏ **riêng một** layer ở giữa/sau của Llama2-7B hoặc Baichuan2-7B gần như không làm hỏng perplexity hay MMLU; layer đầu và layer cuối quan trọng hơn. Họ đo "layer này làm thay đổi hidden state bao nhiêu" bằng **Block Influence (BI)** = 1 − cosine giữa đầu vào và đầu ra của layer. BI thấp nghĩa là layer gần như là hàm đồng nhất, nên bỏ. Chỉ cần tính BI trên một tập calibration nhỏ không nhãn, sắp xếp, xoá các layer BI thấp nhất. Không gradient, không fine-tune. Kết quả: bỏ ~25% tham số mà giữ ~90% hiệu năng trung bình (trên Llama2-13B), tốt hơn LLM-Pruner/SliceGPT/LaCo.

## 3. Phương pháp chi tiết

### 3.1 Block Influence

$$\boxed{\mathrm{BI}_i = 1-\mathbb{E}_{X,t}\,\frac{X_{i,t}^{\top}X_{i+1,t}}{\lVert X_{i,t}\rVert_2\,\lVert X_{i+1,t}\rVert_2}}$$

$X_{i,t}$ là hidden state tại token $t$ vào layer $i$. Lấy kỳ vọng trên các token và các mẫu calibration. Cosine = 1 nghĩa là layer không đổi *hướng* của vector (BI = 0).

Chú ý (suy luận của chúng ta): cosine bỏ qua **độ lớn**. Một layer có thể cộng một vector cùng hướng nhưng rất to vẫn có BI thấp. Paper không bàn điều này.

### 3.2 Vì sao pre-norm làm layer sâu giống hàm đồng nhất (Phụ lục A, lập luận cho init ngẫu nhiên)

Theo Lemma 1 (Xiong et al. 2020) chuẩn của hidden state tăng theo độ sâu, $\lVert x_L\rVert=\Theta(\sqrt L)$, trong khi $\lVert f_L(x_L)\rVert=O(1)$. Do đó

$$\cos(x_{L+1},x_L)\ \ge\ \frac{\lVert x_L\rVert}{\lVert x_{L+1}\rVert}-\frac{\lVert f_L\rVert}{\lVert x_{L+1}\rVert}=\Theta\!\left(\sqrt{\tfrac{L}{L+1}}\right)-O\!\left(\sqrt{\tfrac1{L+1}}\right)\to 1 .$$

Paper nói rõ đây chỉ là phân tích cho mô hình khởi tạo ngẫu nhiên; họ minh hoạ bằng thực nghiệm (Figure 2) rằng mô hình pre-norm tự huấn luyện (200B token) có cosine cao, còn post-norm thì phân kỳ sau ~26B token.

### 3.3 Thuật toán

```
Input: model M (L layers), calibration set D (vd. PG19), số layer cần xoá n
1. for x in D: chạy M, lưu hidden state vào/ra từng layer i
2. BI_i = 1 - mean cosine(X_i, X_{i+1})    # trên mọi token, mọi mẫu
3. sắp xếp layer tăng dần theo BI
4. xoá n layer có BI nhỏ nhất (không nhất thiết liền kề)
5. trả về model đã rút gọn (không retrain)
```

Ví dụ nhỏ: 32 layer, muốn bỏ 9. Với Llama-2-7B, các layer bị bỏ ở Table 9 là 27, 26, 25, 28, 24, 29, 23, 21, 22 (thứ tự theo BI tăng dần), tức gần như một khối liền ở phần sau-giữa của mạng.

## 4. Thí nghiệm và kết quả chính

**Mô hình:** Llama2-7B, Llama2-13B, Baichuan2-7B, Baichuan2-13B (+ RWKV-7B, Mamba-2.8B, Llama2-7B-GPTQ). **Calibration/PPL:** PG19. **Benchmark** theo LaCo: CMNLI, HellaSwag, PIQA, CHID, WSC, CommonSenseQA, BoolQ, Race-H/M, XSum, C3, MMLU, CMMLU. Cột "Per." = % hiệu năng so với dense. Không dùng post-training cho bất kỳ baseline nào.

**Bảng 2 (trích, giá trị Ave. và Per.):**

| Model | Method | Ratio | Ave. | Per. |
|---|---|---|---|---|
| Llama2-7B | Dense | 0% | 47.78 | 100.00 |
| | LLM-Pruner | 27.0% | 34.78 | 72.79 |
| | SliceGPT | 26.4% | 32.84 | 68.73 |
| | LaCo | 27.1% | 38.41 | 80.39 |
| | ShortGPT | 27.1% | 41.24 | 86.31 |
| Llama2-13B | Dense | 0% | 55.14 | 100.00 |
| | LaCo | 24.6% | 47.62 | 86.36 |
| | ShortGPT | 24.6% | 50.53 | 91.64 |
| Baichuan2-7B | ShortGPT | 24.2% | 49.08 | 85.10 |
| Baichuan2-13B | ShortGPT | 24.7% | 54.43 | 87.35 |

Ví dụ nổi bật: bỏ 10/40 layer của Llama2-13B, MMLU từ 55.0 xuống 52.2 trong phần Introduction (bảng 2 ghi 54.69 cho cấu hình ShortGPT 24.6% ở cột MMLU; hai con số khác cấu hình/phiên bản chạy, paper không giải thích rõ).

**Các số khác:**
- Llama-7B vs LLM-Pruner (Table 7, ShortGPT ratio 21.9%): ShortGPT avg 58.12 so với LLM-Pruner 56.82 ở ratio 20% (baseline 63.25).
- Vs SliceGPT trên Llama-2-70B (Table 8), 25%: SliceGPT avg 69.75, ShortGPT 69.79; ở 30%: 66.11 vs 69.4. Baseline 76.57.
- **Non-Transformer (Table 3):** Mamba-2.8B bỏ 25% giữ Per. 90.42; RWKV-7B bỏ 25% chỉ còn 73.16 (ít dư thừa hơn).
- **Quantization (Table 4, Llama2-7B-GPTQ):** bỏ 25.0% (24 layer) throughput 5045.59 token/s (1.16x), MMLU 41.68 so với 43.17; bỏ 27.1% (23 layer) 5146.99 token/s (1.19x), PPL 40.78. Table 5: thứ tự "prune rồi quantize" MMLU 41.2, "quantize rồi prune" 42.4.
- **Post-training (Table 6, Llama2-7B):** thay layer bị xoá bằng MLP gated nhẹ (hidden 2048), train 50B token: Ave. từ 41.22 (ratio 27.1%) lên 43.16 (ratio 24.0%, vì MLP thêm tham số). XSum 0.67 -> 4.89, CHID 24.68 -> 31.82.

## 5. Phân tích / ablation

- **Metric (Fig. 5, Table 10):** so BI với Sequential (bỏ từ đầu), Reverse-order (bỏ từ cuối), Relative Magnitude ($\lVert f(x)\rVert/\lVert x+f(x)\rVert$). BI tốt nhất; Relative Magnitude cạnh tranh; chuẩn của hidden state theo layer cho MMLU tốt nhưng perplexity kém. Các số cụ thể chỉ có trong hình, không có trong text.
- **BI tương quan với PPL khi xoá từng layer** (Fig. 3) – tương quan dương.
- **Last layer đặc biệt (Table 1):** Llama2-7B PPL 7.60 (dense); bỏ cả layer cuối 13.37; bỏ chỉ attention của layer cuối 7.65; bỏ chỉ FFN của layer cuối 12.35. Giả thuyết của tác giả: FFN cuối gần như là một phần của classifier cùng LM head.
- **MMLU rơi đột ngột** ở một số tỉ lệ cụ thể (Fig. 5): giả thuyết có "layer then chốt". Paper không điều tra thêm.
- Paper cho rằng cắt thô (layer) tốt hơn cắt mịn (kênh/neuron) vì khi bỏ riêng từng layer sâu mô hình thay đổi rất ít, nên khó định nghĩa độ quan trọng của đơn vị nhỏ hơn. Đây là **giả thuyết**, chưa được kiểm chứng riêng.

## 6. Giới hạn

**Do tác giả nêu:**
- Layer removal làm hỏng **tác vụ sinh** (XSum, C3) nhiều hơn tác vụ trắc nghiệm: ở 25% trên model 7B, điểm sinh gần về 0 (XSum 0.67 với Llama2-7B, 0.04 với Baichuan2-7B). Giả thuyết: lỗi tích luỹ khi sinh; mô hình lớn (13B) bền hơn. Chưa giải thích được.
- RWKV ít dư thừa hơn Mamba/Transformer, chưa lý giải.

**Do chúng ta nhận thấy:**
- Mỗi layer được chấm **độc lập**; không xét tương tác khi bỏ nhiều layer cùng lúc (37 sẽ chỉ ra điểm này quan trọng).
- Chỉ đo benchmark ngắn/trắc nghiệm; không đo độ trễ batch-1 thực, không có latency ngoài bảng throughput GPTQ.
- Calibration chỉ PG19 (văn xuôi sách); không thử miền khác (OCR, tài liệu).
- Bảng 2 và đoạn Introduction dùng hai con số MMLU khác nhau cho cùng ý (55.0 -> 52.2 và 54.69), cần cẩn thận khi trích.

## 7. Liên hệ với các bài khác

- **26 LLM-Pruner, 29 SliceGPT:** đối thủ width; ShortGPT cho thấy ở ~25% depth tốt hơn width (trong benchmark của paper).
- **35 LaCo:** cũng cắt layer, nhưng *gộp* layer sau vào layer trước thay vì xoá; ShortGPT hơn LaCo ở bảng 2.
- **32 Deeper layers:** cùng kết luận, nhưng chọn *khối liền* bằng độ tương tự góc và có healing QLoRA.
- **34 SLEB:** chọn lặp từng layer bằng perplexity thực (thay vì BI tĩnh).
- **36 ReplaceMe:** phần 4.6 (thay bằng MLP nhẹ + train) là tiền thân của ý "thay khối bị cắt bằng ánh xạ".
- **37:** chứng minh rằng chấm điểm từng layer không có repair có thể bỏ lỡ tối ưu khi dùng repair tuyến tính.

## 8. Ứng dụng cho dự án (suy luận)

- **Rất liên quan về công cụ:** BI là phép đo 1 lần forward, rẻ, chạy được trên decoder 0.8B của LightOnOCR. Có thể đo BI theo layer cho cả GatedDeltaNet lẫn full-attention.
- **Nhưng:** decoder lai (linear-attention + full-attention) không đảm bảo giả định "layer sâu gần đồng nhất". Cần kiểm chứng; Mamba trong paper cho thấy ổn, nhưng đó là SSM thuần.
- **D_lin vs D_res:** BI thấp xấp xỉ "phần cắt gần là hàm đồng nhất", tức một trường hợp đặc biệt của $D_{lin}$ nhỏ (map tối ưu là $I$). Bài 36/37 mới xét map tuyến tính tổng quát.
- Điều cảnh báo: kết quả generative tụt mạnh. OCR là tác vụ sinh chuỗi dài, nên khả năng cao gặp đúng điểm yếu này; cần distillation từ teacher 4B để hồi phục.
- ViT 12 block: paper chỉ làm LLM; dùng cho ViT là suy luận, không có bằng chứng ở đây.

## 9. Thí nghiệm gợi ý

1. Tính BI cho từng block của decoder LightOnOCR trên 500 trang tài liệu; vẽ BI theo layer và so với tăng CER khi xoá riêng từng block (tương tự Fig. 3).
2. Tách BI cho nhóm GatedDeltaNet và nhóm full-attention; xem nhóm nào "dư" hơn.
3. So BI-prune-9-layer-không-train với "SLEB" và "cắt khối liền" (bài 32) ở cùng ngân sách ~0.3B, đo CER trên tập OCR giữ riêng.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| BI$_i$ | 1 − cosine(đầu vào, đầu ra layer $i$) trung bình trên token |
| Pre-norm | Norm trước sub-layer, giữ đường residual |
| Layer removal | Xoá layer BI thấp nhất, không retrain |
| Relative Magnitude | $\lVert f(x)\rVert/\lVert x+f(x)\rVert$, metric đối chứng |
| Per. | % điểm trung bình so với dense |

## 11. Kết luận ngắn

ShortGPT là baseline depth-pruning đơn giản nhất: một forward, một cosine, không train. Mạnh ở benchmark trắc nghiệm ở ~25%, yếu ở tác vụ sinh, và không mô hình hoá tương tác giữa các layer bị bỏ cũng như repair.

Link gốc: https://arxiv.org/abs/2403.03853
