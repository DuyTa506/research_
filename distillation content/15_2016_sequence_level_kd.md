# Sequence-Level Knowledge Distillation (2016)

**Tác giả:** Yoon Kim, Alexander M. Rush (Harvard SEAS).  
**Venue / arXiv:** arXiv:1606.07947 (v4, 22/9/2016); EMNLP 2016 (theo ghi chú trên trang arXiv, bản text không ghi).  
**Nguồn:** https://arxiv.org/abs/1606.07947  
**Chủ đề:** Distillation cho mô hình sinh chuỗi (neural machine translation): word-level KD, sequence-level KD (huấn luyện student trên đầu ra beam search của teacher), sequence-level interpolation; kết hợp với weight pruning.  
**Vị trí trong lộ trình:** Đọc sau 02 (Hinton KD) và 14 (Decoupled KD, KD cho phân loại). Đây là bài **trung tâm** cho dự án OCR của bạn vì nó là bài đầu tiên nói rõ KD cho *chuỗi* khác KD cho phân loại ở đâu. Nối tiếp: 28 (MiniLLM), 29 (GKD), 32 (Speculative KD), 38 (DeepSeek-R1 distillation, thực chất là Seq-KD ở quy mô lớn), 46 (Distil-Whisper dùng pseudo-label từ teacher).

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch paper. Phần "Ứng dụng cho dự án" là suy luận của chúng ta, không phải kết quả của paper (paper chỉ làm dịch máy với LSTM).

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

**Neural machine translation (NMT)** = dịch máy bằng mạng nơ-ron: một *encoder* đọc câu nguồn $s$, một *decoder* sinh câu đích $t=[t_1,\dots,t_J]$ từng từ một, mỗi bước cho phân phối trên từ vựng $V$ có điều kiện vào $s$ và các từ đã sinh $t_{<j}$. Xác suất cả câu là tích:

$$p(t\mid s)=\prod_{j=1}^{J}p(t_j\mid s,t_{<j}).$$

Để đạt chất lượng tốt, NMT năm 2016 cần mô hình khổng lồ (LSTM 4 tầng × 1000 đơn vị, hoặc 16 tầng × 512), nặng với phần cứng thông thường. Các cách nén có hai nhóm: **pruning** (cắt trọng số/neuron theo độ quan trọng) và **knowledge distillation** (student bắt chước teacher; Bucila 2006, Ba & Caruana, Hinton 2015; bài 01, 02).

KD "chuẩn" (Hinton) nhìn dự đoán như một bài **phân loại nhiều lớp** độc lập trên một mẫu $(x,y)$: student cực tiểu cross-entropy với phân phối xác suất của teacher $q$ thay vì nhãn one-hot:

$$\mathcal L_{KD}(\theta;\theta_T)=-\sum_{k=1}^{|V|}q(y=k\mid x;\theta_T)\log p(y=k\mid x;\theta),\qquad \mathcal L=(1-\alpha)\mathcal L_{NLL}+\alpha\mathcal L_{KD}.$$

Điểm khác của NMT: mô hình được *huấn luyện* như phân loại ở mức từ (word-level), nhưng được *dùng* để dự đoán cả chuỗi, trong đó mỗi quyết định dựa trên các quyết định trước đó (lỗi lan truyền khi test). Câu hỏi của paper: có nên distill ở mức chuỗi không?

## 2. Ý tưởng chính

Thay vì bắt student khớp phân phối từng từ của teacher, ta cho teacher **dịch toàn bộ tập huấn luyện bằng beam search** và dùng đầu ra có điểm cao nhất $\hat y$ làm nhãn mới để huấn luyện student bằng NLL thông thường. Lý do: phân phối chuỗi $q(t\mid s)$ của teacher có số mũ nhiều khả năng, không thể khớp trực tiếp; nhưng phần lớn khối lượng xác suất nằm quanh một chuỗi (mode), nên thay toàn bộ phân phối bằng mode là một xấp xỉ hợp lý. Student chỉ cần mô hình hóa vùng quanh mode của teacher thay vì "phí" tham số cho toàn bộ không gian bản dịch. Hệ quả bất ngờ: phân phối của student trở nên nhọn hơn nên **greedy decoding gần như đủ tốt, không cần beam search**, nhanh hơn rất nhiều. Một biến thể (Seq-Inter) chọn trong beam của teacher chuỗi giống bản gốc nhất (theo BLEU) để giữ lại thông tin của dữ liệu thật.

## 3. Phương pháp chi tiết

### 3.1 Word-level KD (cơ sở so sánh)

Áp Hinton KD tại mỗi vị trí $j$ của câu đích thật:

$$\mathcal L_{WORD\text{-}KD}=-\sum_{j=1}^{J}\sum_{k=1}^{|V|}q(t_j=k\mid s,t_{<j})\log p(t_j=k\mid s,t_{<j}).$$

Ngữ cảnh $t_{<j}$ là tiền tố của câu *thật* (teacher-forcing), teacher cho phân phối từ đó. Có thể trộn với NLL nhãn thật (tối ưu $\alpha=0.5$; thử $\{0.5,0.9\}$). Nhiệt độ $\tau\in\{1,1.5,2\}$ đã thử và $\tau=1$ tốt nhất.

### 3.2 Sequence-level KD (Seq-KD)

NLL ở mức chuỗi với nhãn one-hot bằng đúng NLL ở mức từ ($\mathcal L_{SEQ\text{-}NLL}=\mathcal L_{WORD\text{-}NLL}$), nên với dữ liệu thật hai cách là như nhau. Nhưng nếu thay phân phối dữ liệu bằng phân phối chuỗi của teacher:

$$\mathcal L_{SEQ\text{-}KD}=-\sum_{t\in\mathcal T}q(t\mid s)\log p(t\mid s),$$

tổng này có số mũ số hạng nên không tính được. Xấp xỉ đơn giản nhất: thay $q$ bằng mode của nó, $q(t\mid s)\approx\mathbb 1\{t=\arg\max_t q(t\mid s)\}$, và tìm mode xấp xỉ bằng beam search:

$$\boxed{\mathcal L_{SEQ\text{-}KD}\approx-\log p(t=\hat y\mid s)}\qquad \hat y=\text{beam search của teacher}.$$

Số liệu hỗ trợ giả định "phần lớn khối lượng nằm trong một chuỗi": với beam 1, $q(\hat y\mid s)$ trung bình chiếm 1.3% (De→En) và 2.3% (Th→En); với beam 5 là 2.8% và 3.8% (chú thích 5). Khá nhỏ, nhưng tác giả vẫn thấy mode hiệu quả.

### 3.3 Sequence-level interpolation (Seq-Inter)

Muốn đưa dữ liệu thật $y$ vào lại: $\mathcal L=(1-\alpha)\mathcal L_{SEQ\text{-}NLL}+\alpha\mathcal L_{SEQ\text{-}KD}$. Huấn luyện trên cả $y$ lẫn $\hat y$ nhân đôi dữ liệu và hai chuỗi thường khác nhau nhiều. Thay vào đó (lấy cảm hứng từ *local updating* của Liang 2006): chọn một chuỗi trong $K$-best của teacher vừa có xác suất cao vừa gần $y$:

$$\boxed{\tilde y=\arg\max_{t\in\mathcal T}\ \mathrm{sim}(t,y)\,q(t\mid s)\ \approx\ \arg\max_{t\in\mathcal T_K}\mathrm{sim}(t,y)}$$

với $\mathrm{sim}$ = smoothed sentence-level BLEU (Chen & Cherry 2014); thực nghiệm dùng beam $K=35$. Cơ sở sinh dữ liệu (generative story): có chuỗi thật $t\sim D$ không quan sát được, ta chỉ thấy phiên bản nhiễu $y\sim\epsilon(t)$ (bản dịch của một người dịch cụ thể). Khi đó student nên khớp hỗn hợp $D_{Seq\text{-}Inter}\sim(1-\alpha)D+\alpha\,q(t\mid s)$, và $\tilde y$ là xấp xỉ của argmax hỗn hợp đó.

Các phương pháp này **bổ sung cho nhau**: Seq-KD + Word-KD, fine-tune thêm trên dữ liệu Seq-Inter (học rate 0.1 từ mô hình pretrained), v.v.

### Pseudo-code

```python
# Seq-KD (3 bước, như tóm tắt trong paper)
teacher = train(teacher_arch, D_gold)                      # (1) teacher đã huấn luyện
D_kd = []
for (s, y) in D_gold:                                      # (2) beam search trên tập train
    hyps = teacher.beam_search(s, K=5)                     #     K=5, không thấy lợi khi lớn hơn
    D_kd.append((s, hyps[0]))                              #     lấy chuỗi điểm cao nhất
student = train(student_arch, D_kd, loss=NLL)              # (3) NLL thường trên dữ liệu mới
# tuỳ chọn: thêm Word-KD (cross-entropy tới phân phối từng từ của teacher)

# Seq-Inter
D_inter = []
for (s, y) in D_gold:
    hyps = teacher.beam_search(s, K=35)
    y_tilde = max(hyps, key=lambda h: smoothed_bleu(h, y))
    D_inter.append((s, y_tilde))
student = finetune(student, D_inter, lr=0.1)               # bắt đầu từ mô hình đã pretrained
```

### Ví dụ nhỏ

Một nguồn $s$ có teacher beam top-3: $h_1$ (xác suất 0.2), $h_2$ (0.15), $h_3$ (0.1), trong khi bản dịch gốc $y$ giống $h_3$ nhất (BLEU cao nhất). Seq-KD huấn luyện trên $h_1$ (mode). Seq-Inter huấn luyện trên $h_3$ (trong beam, gần $y$ nhất). Word-KD thì không dùng $h_i$ nào mà khớp phân phối từ ở mỗi bước dọc theo $y$. (Ví dụ minh họa của chúng ta.)

## 4. Thí nghiệm và kết quả chính

**Dữ liệu.** English→German (WMT 2014, 4m câu, top-50k từ, test newstest2014) và Thai→English (IWSLT 2015, 90k câu, vocab 25k). Teacher En→De là LSTM $4\times1000$ (221m tham số); student $2\times500$ (84m) và $2\times300$ (49m). Teacher Th→En $2\times500$ (47m), student $2\times100$ (8m). Kiến trúc: attention kiểu Luong 2015. Đo BLEU (tokenized).

### Table 1 (rút gọn; BLEU với beam $K=1$ (greedy) và $K=5$; $\Delta$ là gia tăng so với student baseline; $p(t=\hat y)$ là xác suất mà mô hình gán cho đầu ra greedy)

English→German:

| Mô hình | BLEU K=1 | Δ K=1 | BLEU K=5 | Δ K=5 | PPL | $p(t=\hat y)$ |
|---|---|---|---|---|---|---|
| Teacher 4×1000 | 17.7 | | 19.5 | | 6.7 | 1.3% |
| Teacher + Seq-Inter | 19.6 | +1.9 | 19.8 | +0.3 | 10.4 | 8.2% |
| Student 2×500 baseline | 14.7 | | 17.6 | | 8.2 | 0.9% |
| Word-KD | 15.4 | +0.7 | 17.7 | +0.1 | 8.0 | 1.0% |
| Seq-KD | 18.9 | +4.2 | 19.0 | +1.4 | 22.7 | 16.9% |
| Seq-KD + Seq-Inter | 18.9 | +4.2 | 19.3 | +1.7 | 15.8 | 7.6% |
| Seq-KD + Word-KD | 18.7 | +4.0 | 18.9 | +1.3 | 10.9 | 4.1% |
| Student 2×300 baseline | 14.1 | | 16.9 | | 10.3 | 0.6% |
| Word-KD | 14.9 | +0.8 | 17.6 | +0.7 | 10.9 | 0.7% |
| Seq-KD | 18.1 | +4.0 | 18.1 | +1.2 | 64.4 | 14.8% |
| Seq-KD + Seq-Inter + Word-KD | 18.5 | +4.4 | 18.9 | +2.0 | 97.1 | 5.9% |

Thai→English: Teacher 2×500: 14.3 (K=1) / 15.7 (K=5). Student 2×100 baseline: 10.6 / 12.7; Word-KD 11.8 / 13.6; Seq-KD 12.8 / 13.4; Seq-KD + Word-KD 13.7 / 14.2; Seq-KD + Seq-Inter + Word-KD 14.2 (+3.6) / 14.4 (+1.7).

(Bảng đầy đủ còn các dòng Baseline + Seq-Inter, Word-KD + Seq-Inter; xem paper.)

### Tốc độ (Table 2, số từ nguồn dịch mỗi giây, En→De)

| | GPU (Titan X) | CPU | Android (Galaxy 6) |
|---|---|---|---|
| Greedy, 4×1000 teacher | 425.5 | 15.0 | không chạy được |
| Greedy, 2×500 | 1051.3 | 63.6 | 8.8 |
| Greedy, 2×300 | 1267.8 | 104.3 | 15.8 |
| Beam 5, 4×1000 | 101.9 | 7.9 | không chạy được |
| Beam 5, 2×500 | 181.9 | 22.1 | 1.9 |
| Beam 5, 2×300 | 189.1 | 38.4 | 3.4 |

Student 2×500 greedy chạy nhanh hơn teacher beam search 10 lần trên GPU (1051.3 so với 101.9) với chất lượng tương đương (18.9 so với 19.5 BLEU).

### Pruning kết hợp (Table 3, student 2×500 Seq-KD + Seq-Inter, cắt theo độ lớn trọng số rồi retrain)

| Prune % | Tham số | BLEU (beam 5) | Tỉ lệ nén so với teacher |
|---|---|---|---|
| 0% (teacher 4×1000) | 221m | 19.5 | 1× |
| 0% | 84m | 19.3 | 3× |
| 50% | 42m | 19.3 | 5× |
| 80% | 17m | 19.1 | 13× |
| 85% | 13m | 18.8 | 18× |
| 90% | 8m | 18.5 | 26× |

## 5. Phân tích / ablation

- **Seq-KD đáng kể hơn Word-KD** trên En→De (+4.2 so với +0.7 BLEU greedy), và bằng nhau ở Th→En; kết hợp hai cho thêm lợi ích trên hai trong ba student, nên chúng "trực giao": Word-KD truyền kiến thức cục bộ, Seq-KD truyền kiến thức toàn cục.
- **Khối lượng xác suất dồn về mode:** $p(t=\hat y)$ của student Seq-KD 2×500 là 16.9% so với 0.9% ở baseline. Phân phối nhọn hơn nên argmax dễ tìm hơn, giải thích vì sao greedy đủ. Seq-Inter là điểm giữa (7.6%).
- **Perplexity không dự đoán BLEU:** baseline 2×500 có PPL 8.2, Seq-KD có PPL 22.7 (tệ hơn), nhưng BLEU Seq-KD cao hơn +4.2 (greedy). Lý do: Seq-KD mô hình hóa phân phối đã "co" quanh mode của teacher nên PPL trên tập test thật cao hơn.
- **Giả thuyết của tác giả** vì sao Seq-KD hiệu quả: student không phải "lãng phí" tham số cho toàn bộ không gian bản dịch, chỉ mô hình hóa vùng liên quan quanh mode của teacher. Đây là giả thuyết được ủng hộ bởi số liệu $p(\hat y)$, chưa phải chứng minh.
- **Seq-Inter cải thiện cả teacher:** fine-tune teacher 4×1000 trên dữ liệu Seq-Inter: greedy 17.7 → 19.6 (+1.9), tương đương beam search của teacher gốc (19.5), cho phép giải mã nhanh hơn với cùng kích thước mô hình.
- **Beam lớn hơn 5 cho Seq-KD không giúp** (tác giả không thấy cải thiện).
- **Pruning và KD trực giao:** cắt 80% trọng số student chỉ mất thêm 0.4 BLEU so với teacher (19.5 → 19.1, nén 13×); retrain sau khi cắt là *rất quan trọng*. Embedding từ chiếm khoảng 63% tham số của 2×500 (50m trên 84m), nên pruning tập trung vào phần đó.
- **Thử và bỏ:** hồi quy hidden state trên cùng của student theo teacher như pretraining (kiểu FitNet) cho kết quả tương đương Word-KD nên không theo đuổi.

## 6. Giới hạn

**Do tác giả nêu:** nhận ra mode là xấp xỉ "kém" của một phân phối cỡ mũ (mode chỉ chiếm 1.3% khối lượng với beam 1) nhưng vẫn hiệu quả nhờ beam search đã tốt; chưa kết hợp với giảm kích thước embedding (character-level, v.v.); thí nghiệm chỉ trên NMT, dự đoán có thể áp cho các tác vụ seq2seq khác.

**Do chúng ta nhận xét:**
- Toàn bộ là LSTM + attention, tập dữ liệu dịch cỡ 90k đến 4m câu; không có mô hình Transformer, không có tác vụ nhìn-sang-chữ.
- Chỉ dùng 1 chuỗi (mode) từ teacher nên bỏ phí thông tin xác suất mềm; student học "đúng như teacher dịch" kể cả lỗi của teacher (lỗi của teacher bị khóa cứng vào dữ liệu huấn luyện), trừ khi dùng Seq-Inter.
- Phải chạy beam search của teacher trên toàn bộ tập train; với teacher 4B và dữ liệu lớn đây là chi phí tính toán đáng kể (chi phí một lần, nhưng không nhỏ).
- Seq-KD làm student quá "tự tin" (PPL tệ, $p(\hat y)$ cao): xác suất của nó kém hiệu chuẩn, không phù hợp nếu cần xác suất để rerank hoặc ước lượng độ tin cậy (suy luận).
- Student bị phơi bày ($exposure\ bias$): paper không giải quyết vấn đề train/test mismatch của teacher-forcing; các bài sau (29 GKD, 28 MiniLLM) trực tiếp xử lý bằng dữ liệu on-policy.
- Đánh giá bằng BLEU, không có đánh giá của người.

## 7. Liên hệ với các bài khác

- **01 Ba & Caruana, 02 Hinton KD:** word-level KD ở đây là Hinton KD áp theo từng vị trí; Seq-KD là biến thể "dữ liệu giả từ teacher" (pseudo-labels).
- **14 Decoupled KD:** DKD phân tích KL ở mức từng mẫu phân loại; trong chuỗi mỗi vị trí token là một mẫu như vậy.
- **06 Born-again networks:** student được huấn luyện trên đầu ra của teacher; Seq-KD cùng tinh thần nhưng ở không gian chuỗi.
- **28 MiniLLM, 29 GKD:** hai bài này nhận xét cả Word-KD lẫn Seq-KD đều có exposure bias/khoảng cách phân phối; đề xuất KL ngược hoặc huấn luyện on-policy (suy luận, nội dung ở bài đó).
- **32 Speculative KD:** cải thiện cho thấy phân phối mà student học phải giống phân phối student sinh; có liên hệ với việc Seq-KD làm student nhọn hơn.
- **36 Distilling step-by-step, 37 Orca, 38 DeepSeek-R1 distillation:** tất cả thực chất là Seq-KD ở quy mô lớn, student SFT trên chuỗi do teacher sinh.
- **46 Distil-Whisper:** huấn luyện student ASR trên pseudo-label sinh bởi teacher, rất gần với Seq-KD (suy luận từ tên bài; kiểm tra ở bài 46).
- **03 FitNets:** paper này thử hồi quy hidden state kiểu FitNet và thấy không lợi hơn Word-KD.

## 8. Ứng dụng cho dự án (suy luận)

*Mức liên quan: cao nhất trong nhóm 12-17.* OCR tạo chuỗi dài (hàng nghìn token) và gần như tất định; đây là đúng chế độ mà Seq-KD nhắm tới.

1. **Mode ≈ cả phân phối khi teacher tự tin.** Trong dịch máy, mode chỉ chiếm 1.3-3.8% khối lượng, mà vẫn dùng được. Với OCR, đầu ra gần tất định nên khối lượng xác suất của chuỗi top-1 rất lớn: xấp xỉ "mode thay cho phân phối" ở đây còn *tốt hơn* ở dịch máy (chúng ta suy luận; cần đo $p(\hat y)$ thực tế). Như vậy Seq-KD ≈ huấn luyện student bằng pseudo-label của teacher 4B và mất rất ít thông tin so với KD đầy đủ.
2. **Greedy là đủ.** Ở dự án batch-1, GPU ≤ 8 GB, chi phí beam search nhân với beam width ($K=5$ là chậm hơn nhiều trên thiết bị cạnh). Kết quả của paper (student Seq-KD greedy ≈ teacher beam-5, nhanh hơn 10 lần) khớp ràng buộc latency. Cần kiểm tra lại cho OCR.
3. **Pseudo-label thay hoặc bổ sung ground truth.** Nếu đã có nhãn OCR vàng: dùng **Seq-Inter** với $\mathrm{sim}=1-\mathrm{CER}$ (hoặc edit similarity) để chọn, trong $K$-best của teacher, chuỗi gần ground truth nhất; student học chuỗi "đúng" mà vẫn nằm trong vùng teacher gán xác suất cao. Nếu không có nhãn (tài liệu mới): Seq-KD thuần. Cẩn thận: nếu teacher có lỗi hệ thống (bỏ bảng, sai thứ tự đọc) thì student kế thừa nguyên lỗi.
4. **Seq-KD + Word-KD tốt nhất, kết hợp với pruning.** Phù hợp với kế hoạch "cắt cấu trúc rồi phục hồi bằng distill": sau khi prune ($D_{lin}$ + $D_{res}$), dùng Seq-KD làm loss chính (token-level CE trên chuỗi teacher) và thêm term Word-KD (KL tới phân phối token của teacher, ví dụ DKD ở bài 14) để giữ thông tin phân phối mềm. Phần pruning trong paper là cắt trọng số theo độ lớn rồi retrain; retrain là then chốt (khớp với quan sát của dự án rằng phục hồi sau khi cắt neuron khó hơn cắt layer, vì bài toán ở đó là phục hồi, không phải chọn lựa).
5. **Lỗi lan truyền / exposure bias** vẫn có trên chuỗi dài. Seq-KD huấn luyện bằng teacher-forcing trên chuỗi do teacher sinh nên student gặp ngữ cảnh gần với ngữ cảnh lúc suy luận hơn so với dùng ground truth khác phong cách; về điểm này khoảng cách còn lại được các bài 28/29 xử lý tiếp.
6. **Chi phí sinh dữ liệu:** teacher 4B sinh pseudo-label cho toàn bộ tập train; chi phí một lần, nên cân nhắc cache.
7. **Với vision encoder:** bài này không liên quan trực tiếp (chỉ nói về decoder sinh chuỗi).

## 9. Thí nghiệm gợi ý

1. **Đo $p(\hat y)$ và độ chênh Seq-KD / Word-KD:** với teacher 4B trên tập dev OCR, tính xác suất chuỗi greedy; sau đó so ba cách phục hồi cho cùng một student đã prune: (a) CE trên ground truth, (b) Seq-KD (CE trên đầu ra greedy/beam của teacher), (c) Seq-KD + Word-KD. Đo CER/WER và độ dài lỗi.
2. **Seq-Inter với CER:** lấy $K=8$ hypotheses của teacher (beam hoặc sampling), chọn hypothesis có CER thấp nhất so với ground truth làm nhãn; so với (b). Kiểm tra liệu student có vượt student (b) ở các trang mà teacher dịch sai.
3. **Greedy so với beam của student:** sau (b), so CER và latency batch-1 của student với greedy và beam $K=3$; kiểm chứng tuyên bố "Seq-KD loại bỏ nhu cầu beam search" cho OCR.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| NMT | Mô hình sinh chuỗi đích từng từ, $p(t|s)=\prod_j p(t_j|s,t_{<j})$ |
| Word-KD | $-\sum_j\sum_k q(t_j=k|\cdot)\log p(t_j=k|\cdot)$, KD Hinton theo từng vị trí |
| Seq-KD | Huấn luyện student bằng NLL trên $\hat y$ = beam search top-1 của teacher |
| Mode approximation | $q(t|s)\approx\mathbb 1\{t=\arg\max q\}$ |
| Seq-Inter | Dùng $\tilde y=\arg\max_{t\in\mathcal T_K}\mathrm{sim}(t,y)$, $K=35$, sim = smoothed BLEU |
| $p(t=\hat y)$ | Xác suất student gán cho đầu ra greedy; Seq-KD làm nó tăng mạnh |
| Greedy decoding | Chọn argmax từng bước (beam = 1) |
| Beam search | Giữ $K$ giả thuyết tốt nhất mỗi bước; chi phí tăng tuyến tính theo $K$ |
| Pruning theo độ lớn | Xóa trọng số có |w| nhỏ nhất, sau đó retrain |
| Orthogonal | Nén bằng KD và pruning cộng được với nhau |

## 11. Kết luận ngắn

Bài này chỉ ra rằng với mô hình sinh chuỗi, cách distill hiệu quả hơn là để teacher *tự viết ra* dữ liệu huấn luyện (đầu ra beam search) rồi cho student học như dữ liệu thường, thay vì khớp phân phối từng token. Kết quả: student nhỏ hơn 3 lần, chạy greedy nhanh hơn teacher beam 10 lần với mất khoảng 0.6 BLEU, nén 13 lần khi thêm pruning 80% với mất 0.4 BLEU. Với dự án OCR gần tất định của bạn, đây là bài nên dùng làm phương pháp nền (Seq-KD hoặc Seq-Inter với CER) rồi mới cân nhắc các cải tiến on-policy ở bài 28/29.

Link gốc: https://arxiv.org/abs/1606.07947
