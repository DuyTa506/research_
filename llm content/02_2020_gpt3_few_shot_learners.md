# Language Models are Few-Shot Learners (2020)

**Tác giả:** Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell và cộng sự (OpenAI; tổng cộng 31 tác giả).  
**Venue / arXiv:** arXiv:2005.14165 (v4, 22/07/2020; bản nộp 28/05/2020). Paper này được biết đến là bản NeurIPS 2020, nhưng bản text tôi đọc không ghi venue nên chỉ nêu arXiv.  
**Nguồn:** https://arxiv.org/abs/2005.14165  
**Chủ đề:** Mô hình ngôn ngữ tự hồi quy 175B tham số, in-context learning (zero/one/few-shot), scaling theo kích thước mô hình, ô nhiễm benchmark (data contamination).  
**Vị trí trong lộ trình:** Đọc sau *01 Attention Is All You Need* (kiến trúc Transformer). Bài này cho thấy "chỉ cần phình to Transformer decoder + dữ liệu" là mô hình biết làm nhiều tác vụ mà không cần fine-tune. Nó dùng kết quả của *03 Kaplan scaling laws* (quyết định train mô hình rất lớn, ít token), và bị *04 Chinchilla* phản biện (GPT-3 train thiếu dữ liệu so với kích thước). Tiếp theo: *05 LLaMA* (làm ngược lại: mô hình nhỏ hơn, nhiều token hơn), *07 InstructGPT* (biến GPT-3 thành mô hình nghe lời người dùng).

> Tài liệu học tập bằng tiếng Việt, ghi chú diễn giải, không phải bản dịch. Paper này rất dài (~38k từ, hàng chục benchmark): ghi chú **tập trung vào kiến trúc, quy mô, cách huấn luyện và khái niệm in-context learning**; các phần đánh giá dài chỉ lấy những kết quả tiêu biểu nhất. Các phần "Ứng dụng cho dự án" và các con số đánh dấu "tự tính" là suy luận của chúng ta, không phải kết luận của paper.

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

**Mô hình hóa ngôn ngữ (language modeling):** huấn luyện mạng dự đoán token tiếp theo từ các token trước. Không cần nhãn người gán, nên dữ liệu gần như vô hạn (cả internet).

**Paradigm trước GPT-3: pre-train rồi fine-tune.** Pre-train một Transformer lớn trên văn bản thô (GPT-1/2, BERT, T5), sau đó *fine-tune* (tiếp tục huấn luyện, cập nhật trọng số) trên một tập có nhãn riêng cho từng tác vụ, thường hàng nghìn đến hàng trăm nghìn mẫu. Kiến trúc thì dùng chung, nhưng *dữ liệu và bước fine-tune thì phải làm lại cho mỗi tác vụ*.

Paper nêu ba lý do vì sao đó là hạn chế:
1. **Thực tế:** có vô số tác vụ ngôn ngữ hữu ích; gom bộ dữ liệu có nhãn cho từng cái là không khả thi.
2. **Tương quan giả (spurious correlations):** mô hình lớn fine-tune trên tập hẹp dễ bám vào đặc điểm tình cờ của tập đó, nên điểm benchmark "ngang người" có thể phóng đại năng lực thật.
3. **So với con người:** người chỉ cần một câu chỉ dẫn hoặc vài ví dụ là làm được tác vụ mới.

**Ý tưởng tiền đề:** GPT-2 đã thử đưa tác vụ vào *văn bản đầu vào* (prompt) và để mô hình tự điền tiếp, nhưng kết quả còn kém xa fine-tune (paper nhắc: GPT-2 chỉ đạt 4% trên Natural Questions). Cùng lúc, các nghiên cứu scaling (Kaplan et al., bài 03) cho thấy loss giảm mượt theo kích thước. Giả thuyết của GPT-3: *khả năng học từ ngữ cảnh cũng sẽ tăng theo quy mô*.

---

## 2. Ý tưởng chính

Huấn luyện một Transformer decoder khổng lồ (175 tỷ tham số, gấp 10 lần mô hình dày lớn nhất trước đó theo paper) chỉ với mục tiêu "đoán token tiếp theo" trên ~300 tỷ token. Sau đó, **không cập nhật trọng số nào**: ta chỉ viết vào đầu vào một mô tả tác vụ và/hoặc vài ví dụ mẫu (demonstrations) rồi để mô hình điền tiếp. Cách dùng này gọi là **in-context learning** (học trong ngữ cảnh): "việc học" xảy ra trong một lượt forward pass, trên chính chuỗi đầu vào, chứ không phải bằng gradient.

Điểm quan trọng nhất của paper là *quan sát thực nghiệm*: khoảng cách giữa zero-shot, one-shot và few-shot **tăng lên cùng kích thước mô hình**; mô hình càng lớn càng khai thác tốt các ví dụ trong ngữ cảnh. Paper gọi đây là dạng "meta-learning" ngầm: vòng ngoài là pre-train (học nhiều kỹ năng và nhận dạng mẫu), vòng trong là in-context learning lúc suy luận.

---

## 3. Phương pháp chi tiết

### 3.1 Bốn chế độ sử dụng (Hình 2.1)

| Chế độ | Có cập nhật trọng số? | Trong ngữ cảnh có gì | Số mẫu |
|---|---|---|---|
| Fine-tuning (FT) | Có | (không dùng ngữ cảnh) | nghìn đến trăm nghìn mẫu có nhãn |
| Few-shot (FS) | Không | mô tả (tuỳ chọn) + $K$ ví dụ + 1 truy vấn | $K$ thường 10 đến 100 |
| One-shot (1S) | Không | mô tả + 1 ví dụ + truy vấn | 1 |
| Zero-shot (0S) | Không | chỉ mô tả bằng ngôn ngữ tự nhiên + truy vấn | 0 |

Paper **không fine-tune GPT-3** (để lại cho tương lai), chỉ đánh giá FS/1S/0S. $K$ bị giới hạn bởi cửa sổ ngữ cảnh $n_{ctx}=2048$.

Ví dụ prompt few-shot dịch Anh-Pháp (minh họa, theo tinh thần Hình 2.1):

```
Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
plush giraffe => girafe peluche
cheese =>            <- mô hình điền tiếp
```

Cách chấm điểm (mục 2.4):
- **Trắc nghiệm:** đưa $K$ ví dụ (ngữ cảnh + đáp án đúng), rồi ngữ cảnh của câu hỏi; so sánh *likelihood* của từng phương án, thường chuẩn hoá theo số token. Với ARC, OpenBookQA, RACE chuẩn hoá theo $\frac{P(\text{completion}\mid\text{context})}{P(\text{completion}\mid\text{"Answer: "})}$.
- **Sinh tự do:** beam search, beam width 4, length penalty $\alpha=0.6$; chấm bằng F1, BLEU hoặc exact match.

### 3.2 Kiến trúc

Paper nói gọn: *"same model and architecture as GPT-2"* (decoder-only Transformer, khởi tạo cải tiến, **pre-normalization**, tokenizer BPE ở mức byte) với **một khác biệt**: các lớp **xen kẽ attention dày (dense) và attention thưa kiểu dải băng cục bộ (locally banded sparse)**, giống Sparse Transformer. Paper không mô tả chi tiết dải băng này (độ rộng, lớp nào dày lớp nào thưa), nên đừng suy diễn thêm.

Giải thích thuật ngữ:
- **Decoder-only:** chỉ có khối giải mã với causal mask (mỗi token chỉ nhìn các token trước).
- **Pre-normalization (pre-LN):** áp dụng LayerNorm *trước* mỗi sub-layer (attention/FFN) thay vì sau; huấn luyện ổn định hơn ở mô hình sâu (xem bài 09).
- **BPE (byte-pair encoding):** gộp các cặp ký tự/byte hay đi cùng nhau thành token.
- **Locally banded sparse attention:** mỗi token chỉ attend vào một dải $w$ token gần nhất, thay vì toàn bộ quá khứ → tính toán $O(nw)$ thay vì $O(n^2)$ (cùng ý tưởng sliding-window ở bài 17 và Mistral, bài 06).

Một block (đúng theo kiểu GPT-2, pre-LN):

$$x \leftarrow x + \text{Attn}(\text{LN}(x)),\qquad x \leftarrow x + \text{FFN}(\text{LN}(x)),\qquad d_{ff}=4\,d_{model}$$

Các cấu hình (Bảng 2.1, nguyên văn từ paper; tất cả train trên **300B token**, $n_{ctx}=2048$):

| Mô hình | $n_{params}$ | $n_{layers}$ | $d_{model}$ | $n_{heads}$ | $d_{head}$ | Batch (token) | LR |
|---|---|---|---|---|---|---|---|
| GPT-3 Small | 125M | 12 | 768 | 12 | 64 | 0.5M | $6.0\times10^{-4}$ |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 64 | 0.5M | $3.0\times10^{-4}$ |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 96 | 0.5M | $2.5\times10^{-4}$ |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 128 | 1M | $2.0\times10^{-4}$ |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 | 80 | 1M | $1.6\times10^{-4}$ |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 2M | $1.2\times10^{-4}$ |
| GPT-3 13B | 13.0B | 40 | 5140 | 40 | 128 | 2M | $1.0\times10^{-4}$ |
| GPT-3 175B | 175.0B | 96 | 12288 | 96 | 128 | 3.2M | $0.6\times10^{-4}$ |

Lưu ý: $d_{model}=5140$ cho bản 13B là đúng như bảng trong paper (bản text), con số hơi lạ (không chia hết cho 128) nhưng tôi giữ nguyên. Quy luật rõ ràng: **mô hình càng lớn → batch càng lớn, learning rate càng nhỏ** (paper dựa vào gradient noise scale để chọn batch).

**Kiểm tra số tham số (tự tính):** mỗi layer Transformer có $\approx 4d^2$ (attention: Q,K,V,O) $+ 8d^2$ (FFN với $d_{ff}=4d$) $=12d^2$ tham số. Với 175B: $12\times 96\times 12288^2 \approx 1.74\times10^{11}$, khớp 175B (chưa tính embedding, nhỏ so với tổng).

### 3.3 Dữ liệu (Bảng 2.2)

| Dataset | Token | Trọng số trong mix huấn luyện | Số epoch qua 300B token |
|---|---|---|---|
| Common Crawl (đã lọc) | 410B | 60% | 0.44 |
| WebText2 | 19B | 22% | 2.9 |
| Books1 | 12B | 8% | 1.9 |
| Books2 | 55B | 8% | 0.43 |
| Wikipedia | 3B | 3% | 3.4 |

Ba bước cải thiện chất lượng: (1) lọc Common Crawl bằng bộ phân loại "giống corpus chất lượng cao"; (2) khử trùng lặp mờ ở mức tài liệu (MinHashLSH, giảm trung bình ~10% kích thước); (3) trộn thêm corpus chất lượng cao. Quan trọng: dataset **không lấy mẫu tỉ lệ với kích thước** mà ưu tiên nguồn tốt (Wikipedia thấy 3.4 lần, Common Crawl chưa đến 1 lần) - chấp nhận chút overfit để đổi lấy chất lượng. Common Crawl: 45TB nén → 570GB sau lọc (~400B token BPE). Ngôn ngữ: tiếng Anh chiếm 93% theo số từ.

### 3.4 Huấn luyện (Phụ lục B)

- Adam với $\beta_1=0.9,\ \beta_2=0.95,\ \epsilon=10^{-8}$; clip gradient norm toàn cục ở 1.0.
- LR: warmup tuyến tính trong 375M token đầu, sau đó cosine decay xuống 10% giá trị gốc trong 260B token, rồi giữ 10%.
- Batch size tăng tuyến tính từ 32k token lên giá trị đầy đủ trong 4-12B token đầu.
- Weight decay 0.1; lấy mẫu không hoàn lại (sampling without replacement).
- **Packing:** luôn train chuỗi dài đủ 2048 token, nối nhiều tài liệu ngắn vào một chuỗi, ngăn cách bằng token end-of-text, *không* mask đặc biệt giữa các tài liệu.
- Hạ tầng: GPU V100, cụm của Microsoft; song song hoá mô hình theo cả chiều sâu (giữa các layer) lẫn chiều rộng (trong từng phép nhân ma trận).

### 3.5 Chi phí tính toán (Phụ lục D)

Paper ước lượng tính toán huấn luyện bỏ qua attention (dưới 10% tổng): mỗi token cần 2 FLOPs/tham số cho forward và nhân 3 cho forward+backward:

$$\boxed{C_{train}\approx 6\,N\,D}$$

với $N$ là số tham số, $D$ số token. Với GPT-3 175B: $6\times 175\times10^{9}\times 300\times10^{9}\approx 3.15\times10^{23}$ FLOPs (tự tính), khớp bảng D.1 của paper: $3.14\times10^{23}$ FLOPs = **3.64E+03 petaflop/s-day** (1 PF-day = $8.64\times10^{19}$ FLOPs). Paper cũng so sánh: GPT-3 3B (~50 PF-day) tốn ngang RoBERTa-Large 355M (Hình 2.2): họ cố ý "train mô hình rất lớn trên ít token hơn thông thường", theo hướng của Kaplan et al.

### 3.6 Pseudo-code: in-context learning lúc suy luận

```python
def few_shot_predict(model, task_description, demos, query, K):
    # demos: list[(context_i, completion_i)] lấy ngẫu nhiên từ tập train
    prompt = task_description + "\n"
    for c, y in random.sample(demos, K):        # K sao cho vừa 2048 token
        prompt += f"{c} {y}\n"
    prompt += f"{query}"
    # KHÔNG có gradient, KHÔNG sửa trọng số
    if multiple_choice:
        return argmax_over_options(model.logprob(option | prompt) / len_norm)
    else:
        return model.generate(prompt, beam=4, length_penalty=0.6)
```

---

## 4. Thí nghiệm và kết quả chính

Chọn lọc; mọi số liệu lấy từ các bảng trong paper (GPT-3 175B nếu không ghi khác).

### 4.1 Language modeling và cloze

| Benchmark | SOTA trước đó | 0-shot | 1-shot | Few-shot |
|---|---|---|---|---|
| PTB (perplexity, thấp tốt hơn) | 35.8 | 20.5 | n/a | n/a |
| LAMBADA acc | 68.0 | 76.2 | 72.5 | 86.4 |
| LAMBADA ppl | 8.63 | 3.00 | 3.35 | 1.92 |
| StoryCloze acc | 91.8 | 83.2 | 84.7 | 87.7 |
| HellaSwag acc | 85.6 | 78.9 | 78.1 | 79.3 |

LAMBADA là ví dụ đẹp về giá trị của few-shot: có ví dụ dạng điền chỗ trống thì mô hình hiểu "đáp án chính xác là một từ", nhảy từ 76.2% lên 86.4%.

### 4.2 Hỏi đáp closed-book (không cho tài liệu tham khảo)

| Setting | NaturalQS | WebQS | TriviaQA |
|---|---|---|---|
| T5-11B+SSM (fine-tuned) | 36.6 | 44.7 | 60.5 |
| RAG (fine-tuned, open-domain) | 44.5 | 45.5 | 68.0 |
| GPT-3 0-shot | 14.6 | 14.4 | 64.3 |
| GPT-3 1-shot | 23.0 | 25.3 | 68.0 |
| GPT-3 few-shot | 29.9 | 41.5 | 71.2 |

### 4.3 Dịch máy (BLEU, few-shot dùng 64 ví dụ theo paper)

| | En→Fr | Fr→En | En→De | De→En | En→Ro | Ro→En |
|---|---|---|---|---|---|---|
| GPT-3 0-shot | 25.2 | 21.2 | 24.6 | 27.2 | 14.1 | 19.9 |
| GPT-3 1-shot | 28.3 | 33.7 | 26.2 | 30.4 | 20.6 | 38.6 |
| GPT-3 few-shot | 32.6 | 39.2 | 29.7 | 40.6 | 21.0 | 39.5 |

Dịch *sang* tiếng Anh tốt hơn nhiều so với dịch *từ* tiếng Anh (dữ liệu 93% tiếng Anh). Paper nghi tokenizer BPE của GPT-2 (thiết kế cho tiếng Anh) là một nguyên nhân gây En→Ro yếu.

### 4.4 Winograd và SuperGLUE

| | Winograd | Winogrande (XL) |
|---|---|---|
| Fine-tuned SOTA | 90.1 | 84.6 |
| GPT-3 0-shot / 1-shot / few-shot | 88.3* / 89.7* / 88.6* | 70.2 / 73.2 / 77.7 |

(* = có nghi ngờ nhiễm dữ liệu.)

SuperGLUE (few-shot, 32 ví dụ/tác vụ, test set): trung bình **71.8** so với 69.0 của BERT-Large fine-tune và 89.0 của SOTA fine-tune. Điểm mạnh: COPA 92.0, ReCoRD 90.2/91.1, WSC 80.1. Điểm yếu: **WiC 49.4% (ngang đoán bừa)**, RTE 69.0, CB F1 52.0, MultiRC acc 30.5. Paper nhận xét: GPT-3 yếu ở các tác vụ "so sánh hai đoạn văn" (WiC, ANLI, RTE).

### 4.5 Tác vụ tổng hợp: số học (Bảng 3.9, 175B, độ chính xác %)

| | 2D+ | 2D- | 3D+ | 3D- | 4D+ | 4D- | 5D+ | 5D- | 2Dx | 1DC |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-shot | 76.9 | 58.0 | 34.2 | 48.3 | 4.0 | 7.5 | 0.7 | 0.8 | 19.8 | 9.8 |
| 1-shot | 99.6 | 86.4 | 65.5 | 78.7 | 14.0 | 14.0 | 3.5 | 3.8 | 27.4 | 14.3 |
| few-shot | 100.0 | 98.9 | 80.4 | 94.2 | 25.5 | 26.8 | 9.3 | 9.9 | 29.2 | 21.3 |

(Văn bản của paper ghi 3D+ few-shot là 80.2%, bảng ghi 80.4%; tôi lấy số trong bảng.) Kiểm tra ghi nhớ: trong 2000 bài cộng 3 chữ số chỉ có 17 bài (0.8%) xuất hiện nguyên văn trong dữ liệu train; lỗi thường gặp là "quên nhớ 1" → mô hình có vẻ đang *tính*, không chỉ thuộc lòng. Mô hình 13B chỉ giải được cộng/trừ 2 chữ số khoảng một nửa số lần; nhảy vọt xảy ra ở 175B.

Các tác vụ khác: xáo trộn chữ (cycle letters: 37.9% few-shot; anagram bỏ chữ đầu-cuối: 15.1%; chèn ngẫu nhiên: 67.2%), dùng từ mới sau khi chỉ thấy định nghĩa một lần, sửa ngữ pháp tiếng Anh (định tính).

### 4.6 Sinh bài báo và khả năng phân biệt của con người

Điều kiện: 3 bài báo thật làm few-shot context + tiêu đề/phụ đề mới. Người tham gia (~80 người Mỹ mỗi mô hình) phân biệt bài người viết và bài máy viết:

| Mô hình | Độ chính xác của người | 
|---|---|
| Control (mô hình cố tình tệ) | 86% |
| GPT-3 Small | 76% |
| GPT-3 13B | 55% |
| GPT-3 175B | 52% (gần đoán bừa 50%) |

Bài ~500 từ: 88% (control) so với 52% (GPT-3 175B).

---

## 5. Phân tích / ablation

**(a) Scaling mượt theo compute.** Cross-entropy validation loss giảm theo luật lũy thừa theo compute (Hình 3.1), kéo dài thêm 2 bậc độ lớn so với Kaplan et al. mà chỉ lệch nhỏ khỏi đường dự đoán. Và loss thấp hơn chuyển thành điểm các tác vụ downstream tốt hơn một cách nhất quán.

**(b) Khoảng cách 0/1/few-shot tăng theo kích thước (Hình 1.3, 3.8).** Trên 42 benchmark tính tổng hợp, zero-shot tăng đều theo kích thước, few-shot tăng nhanh hơn. SuperGLUE: GPT-3 cần chưa đến 8 ví dụ mỗi tác vụ để vượt BERT-Large fine-tune, và chênh lệch giữa BERT-Large và BERT++ gần tương đương chênh lệch giữa 1 ví dụ và 8 ví dụ trong ngữ cảnh. Đây là bằng chứng chính cho tuyên bố "mô hình lớn là meta-learner tốt hơn".

**(c) Mô tả bằng ngôn ngữ tự nhiên + số lượng ví dụ $K$ đều giúp (Hình 1.2)**, và độ dốc "đường cong học trong ngữ cảnh" lớn hơn ở mô hình lớn (tác vụ thử: xóa ký hiệu thừa khỏi từ).

**(d) Ô nhiễm benchmark (mục 4).** Do một lỗi trong bước lọc, một phần dữ liệu test lọt vào tập train mà không thể train lại. Paper tạo phiên bản "sạch" của mỗi benchmark (loại mẫu có overlap 13-gram) và so điểm:
- Hơn 1/4 benchmark có tỉ lệ nghi nhiễm trên 50%, nhưng *điểm hầu như không đổi* và không thấy tương quan giữa mức nhiễm và chênh lệch điểm.
- Chỉ PIQA và Winograd có bằng chứng nhiễm đáng kể → đánh dấu sao (*) trong kết quả.
- Đọc hiểu (QuAC, SQuAD2, DROP): >90% mẫu bị gắn cờ nhưng khi kiểm tay chỉ văn bản nguồn xuất hiện, không có cặp hỏi-đáp.
- Training/validation gap nhỏ (Hình 4.1) → GPT-3 không overfit đáng kể dù lớn.

**(e) In-context learning thật sự "học" hay chỉ "nhận ra" tác vụ?** Paper thừa nhận là câu hỏi mở: dịch máy chắc chắn học trong pre-train; các tác vụ tổng hợp (xáo chữ, từ vô nghĩa) có vẻ gần với "học mới hoàn toàn" hơn.

Paper **không** làm ablation kiến trúc (không thử bỏ sparse attention, không so sánh biến thể), vì kiến trúc coi như cố định và biến chính là kích thước.

---

## 6. Giới hạn

**Do tác giả nêu:**
- Sinh văn bản dài vẫn lặp ý, mất mạch lạc, tự mâu thuẫn; yếu về "vật lý thường thức" ("bỏ phô mai vào tủ lạnh thì có tan không?").
- Gần ngang đoán bừa ở các tác vụ so sánh (WiC, ANLI) và một số đọc hiểu (QuAC, RACE). Giả thuyết: do mô hình tự hồi quy một chiều (không bidirectional) không phù hợp với việc "xem lại, so sánh hai đoạn".
- Mục tiêu pre-train coi mọi token như nhau, không có khái niệm "token nào quan trọng"; không có grounding (hình ảnh, tương tác vật lý).
- Hiệu quả mẫu khi pre-train kém (thấy nhiều văn bản hơn một đời người).
- Chưa rõ few-shot là học mới hay nhận dạng tác vụ đã gặp.
- **Suy luận đắt và bất tiện ở quy mô này**; paper tự gợi ý *distillation* ở quy mô trăm tỷ tham số là hướng nghiên cứu (đây chính là bài toán của dự án bạn, ở quy mô nhỏ hơn nhiều).
- Thiên kiến (giới tính, chủng tộc, tôn giáo), khả năng lạm dụng sinh văn bản giả (mục 6), tiêu thụ năng lượng: chi phí train là hàng nghìn PF-day, nhưng sinh 100 trang văn bản chỉ tốn khoảng 0.4 kW-hr (theo paper).
- Lỗi lọc ô nhiễm không sửa được.

**Tôi nhận xét thêm (suy luận):**
- Cửa sổ ngữ cảnh 2048 token giới hạn số ví dụ in-context ($K$ ≈ 10-100) và độ dài tài liệu.
- Mô hình 175B với KV-cache lớn (xem 8.1) khiến chi phí in-context learning ở lúc suy luận không hề nhỏ: mỗi ví dụ thêm vào prompt là token phải prefill và lưu cache.
- Chưa có so sánh trực tiếp "few-shot vs fine-tune cùng mô hình" vì paper không fine-tune GPT-3.
- Kích thước tối ưu theo quan điểm sau này (Chinchilla, bài 04): 300B token cho 175B tham số là rất ít (khoảng 1.7 token/tham số, tự tính).

---

## 7. Liên hệ với các bài khác

- **01 Attention Is All You Need:** GPT-3 dùng chỉ phần decoder, bỏ cross-attention, chồng 96 layer.
- **03 Kaplan scaling laws:** cơ sở lý thuyết cho việc "phình mô hình, giữ token ở 300B"; GPT-3 xác nhận luật lũy thừa thêm 2 bậc độ lớn compute.
- **04 Chinchilla:** chứng minh GPT-3 train thiếu dữ liệu; mô hình nhỏ hơn mà nhiều token hơn thì tốt hơn cho cùng compute.
- **05 LLaMA:** trực tiếp tuyên bố LLaMA-13B vượt GPT-3 175B trên hầu hết benchmark dù nhỏ hơn 10 lần.
- **07 InstructGPT:** GPT-3 prompt thế nào cũng phải "thiết kế prompt"; RLHF làm mô hình làm theo chỉ dẫn trực tiếp.
- **09 Pre-LN:** lý do kỹ thuật cho pre-normalization.
- **17 Longformer / 06 Mistral:** attention dải băng cục bộ; GPT-3 xen kẽ dày/thưa, Mistral dùng sliding window ở *mọi* layer.
- **48 Speculative decoding, 46 PagedAttention:** các kỹ thuật phục vụ mô hình cỡ GPT-3 ra đời để giải bài toán chi phí suy luận mà paper nêu ở mục Giới hạn.
- **08 LoRA:** thay thế rẻ cho full fine-tune mà GPT-3 muốn tránh (LoRA nhắm đúng mô hình cỡ GPT-3 175B).

---

## 8. Ứng dụng cho dự án (suy luận)

Dự án của bạn (LightOnOCR-3-0.8B, decoder lai GatedDeltaNet + full attention, ViT 12 block) khác GPT-3 gần như mọi mặt, nên bài này hữu ích chủ yếu như **nền tảng khái niệm**, không phải công thức để áp dụng.

### 8.1 Phân tích chi phí: GPT-3 so với mô hình của bạn (tự tính)

**Tham số:** $N\approx 12\,L\,d^2$ (cộng embedding). Với $L=24,\ d=1024$: $12\times24\times1024^2\approx 0.30$B cho phần Transformer thuần, tức ở cỡ 0.3B (giả sử full attention + FFN $4d$; mô hình thực của bạn có GatedDeltaNet/SwiGLU nên con số thực sẽ khác, đây chỉ là bậc độ lớn).

**KV-cache** (dạng vector K và V cho mọi token đã sinh, lưu để không tính lại):

$$\boxed{\text{KV bytes} = 2\ (K,V)\times L \times n_{kv}\,d_{head}\times T \times b}$$

| Mô hình | Công thức mỗi token | Mỗi token | $T$ | Tổng |
|---|---|---|---|---|
| GPT-3 175B (fp16, MHA, $L=96$, $d=12288$) | $2\cdot96\cdot12288\cdot2$ B | 4.72 MB | 2048 | **9.66 GB** (tự tính) |
| Mô hình giả định của bạn (fp16, $L=24$, 16 đầu × 64, MHA, mọi layer full attention) | $2\cdot24\cdot16\cdot64\cdot2$ B | 98.3 KB | 1500 | **147 MB** (tự tính) |
| Như trên nhưng GQA 4 KV-head | | 24.6 KB | 1500 | 36.9 MB (tự tính) |
| Như trên, chỉ 6/24 layer là full attention (còn lại GatedDeltaNet với state cố định) | | 24.6 KB | 1500 | 36.9 MB cho MHA-6-layer; 9.2 MB nếu có thêm GQA-4 (tự tính) |

Kết luận định lượng: ở batch 1 và 1500 token, KV-cache của mô hình cỡ 0.8B/0.3B chỉ là vài chục đến vài trăm MB, *không phải điểm nghẽn bộ nhớ so với trọng số* (0.3B × 2 B = 0.6 GB fp16). Nghẽn thực sự ở batch 1 là **băng thông bộ nhớ khi đọc trọng số mỗi token sinh ra** (memory-bound decode). GPT-3 thì ngược lại: trọng số fp16 khoảng 350 GB (tự tính) nên phải chia nhiều GPU.

**Compute huấn luyện** $6ND$: để mô phỏng một lần distill cỡ nhỏ, ví dụ $N=0.3$B và $D=10$B token → $6\times0.3\times10^9\times10\times10^9=1.8\times10^{19}$ FLOPs (tự tính), tức khoảng 0.2 PF-day, ít hơn GPT-3 khoảng 17,000 lần.

### 8.2 Những gì thực sự đáng mang theo

1. **In-context learning cho OCR VLM:** mô hình của bạn là chuyên dụng (ảnh → văn bản), không cần few-shot. Nhưng bài học "khả năng tăng đột biến theo quy mô" (nhảy từ 13B lên 175B ở số học) cảnh báo rằng khi *prune* mạnh xuống 0.1B, những khả năng "mềm" (đọc layout lạ, bảng phức tạp) có thể mất trước các khả năng cơ bản (chép lại ký tự). Nên theo dõi chúng riêng, đừng chỉ dùng một con số CER/WER tổng.
2. **Phương pháp đo "clean vs all" để kiểm tra ô nhiễm** (loại mẫu overlap 13-gram rồi so điểm) có thể áp dụng khi đánh giá model đã distill: bảo đảm tập test OCR không nằm trong dữ liệu distill của teacher.
3. **Distillation được chính GPT-3 gợi ý** (mục Limitations, 6.3) như hướng giảm chi phí suy luận; dự án bạn là một thể hiện cụ thể.
4. **Sparse/banded attention xen kẽ dày:** cùng họ ý tưởng với hybrid GatedDeltaNet + full attention của bạn: xen "lớp rẻ" với "lớp đắt". Khác biệt: lớp rẻ của GPT-3 vẫn là softmax attention trên cửa sổ cục bộ (KV-cache vẫn tăng theo $T$ hoặc bị chặn bởi $w$), còn GatedDeltaNet là state kích thước cố định.
5. **Với 1500 token/trang, 2048 ctx của GPT-3 gần như đủ cho text đầu ra của một trang**; nhưng mô hình của bạn còn cần token ảnh từ ViT, nên ngữ cảnh thực lớn hơn.

Phần còn lại của bài (đánh giá NLP, thiên kiến, ô nhiễm trên QA) **ít liên quan** đến dự án.

---

## 9. Thí nghiệm gợi ý

1. **Đo chênh lệch zero-shot/few-shot theo kích thước cho model của bạn.** Với các checkpoint (0.8B, 0.3B, 0.1B sau distill), thử prompt OCR có kèm 0, 1, 2 ví dụ ảnh-văn bản ngắn (nếu kiến trúc cho phép). Hỏi: mô hình nhỏ có còn hưởng lợi từ ví dụ trong ngữ cảnh không? Dự đoán theo paper: khoảng cách này co lại khi mô hình nhỏ đi.
2. **Tách "kỹ năng cơ bản" và "kỹ năng khó" khi prune.** Chia tập test thành mẫu dễ (văn bản sạch, một cột) và khó (bảng, công thức, nhiều cột); vẽ CER theo mức prune cho từng nhóm để xem có hiện tượng "kỹ năng nổi lên ở quy mô lớn" ngược chiều (mất đột ngột) không.
3. **Kiểm tra nhiễm dữ liệu kiểu mục 4:** với tập test OCR, tính overlap n-gram với văn bản dùng để distill, so điểm trên toàn bộ và trên tập sạch.

---

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa ngắn gọn |
|---|---|
| In-context learning | Học tác vụ từ prompt trong một forward pass, không cập nhật trọng số |
| Zero/One/Few-shot | 0 / 1 / $K$ (10-100) ví dụ trong ngữ cảnh + mô tả tác vụ |
| Fine-tuning (FT) | Cập nhật trọng số trên tập có nhãn cho từng tác vụ (không dùng cho GPT-3) |
| $n_{ctx}=2048$ | Độ dài ngữ cảnh tối đa, giới hạn $K$ |
| $d_{ff}=4d_{model}$ | Kích thước hidden của FFN |
| Alternating dense/banded sparse attention | Xen kẽ attention toàn cục và attention dải băng cục bộ giữa các layer |
| $N\approx 12Ld^2$ | Số tham số xấp xỉ của Transformer decoder (tự tính) |
| $\boxed{C\approx 6ND}$ | FLOPs huấn luyện: 2 forward + 4 backward mỗi tham số mỗi token |
| 3.64E+03 PF-day | Compute huấn luyện GPT-3 175B (Bảng D.1) |
| 300B token | Số token train mọi kích thước GPT-3 |
| Data contamination | Dữ liệu test lọt vào train; kiểm bằng 13-gram overlap và so điểm clean/all |
| Gradient noise scale | Số đo dùng để chọn batch size (mô hình lớn: batch lớn, LR nhỏ) |

---

## 11. Kết luận ngắn

GPT-3 không phát minh kiến trúc mới; đóng góp là bằng chứng thực nghiệm rằng *quy mô + mục tiêu đoán token tiếp theo* đủ cho một mô hình làm được rất nhiều tác vụ chỉ qua prompt, và khả năng học từ ví dụ trong ngữ cảnh tăng theo kích thước. Cái giá là chi phí khổng lồ (3640 PF-day train, hàng trăm GB trọng số) và những điểm yếu rõ rệt (so sánh câu, lặp ý). Chính các chi phí đó mở đường cho hai hướng bạn quan tâm: huấn luyện mô hình nhỏ hơn trên nhiều token hơn (LLaMA) và nén mô hình bằng distillation/pruning.

Nguồn gốc: https://arxiv.org/abs/2005.14165
