# On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (2024)

**Tác giả:** Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, Olivier Bachem (Google DeepMind; Mila; U. Toronto).  
**Venue / arXiv:** arXiv:2306.13649 (v3, 17/01/2024); ICLR 2024.  
**Nguồn:** https://arxiv.org/abs/2306.13649  
**Chủ đề:** Distillation cho mô hình sinh tự hồi quy (LLM); on-policy distillation (GKD); chọn divergence (forward KL, reverse KL, JSD); kết hợp distillation với RL fine-tuning.  
**Vị trí trong lộ trình:** Đọc trước: 02 (Hinton KD), 15 (Sequence-level KD), 16 (DistilBERT), 27 (LLM KD survey), 28 (MiniLLM, cũng dùng reverse KL). Bài này đặt nền cho nhóm "on-policy" của LLM distillation: đưa ý tưởng imitation learning (DAgger) vào KD, và đưa ra họ loss tổng quát. Đọc tiếp: 30 (DistiLLM), 31 (Rethinking KL), 32 (Speculative KD), 33 (TAID), 34 (DistiLLM-2), 47 (ShortOPD).

> Đây là ghi chú học tập bằng tiếng Việt, diễn giải lại, không phải bản dịch paper. Mọi phần "Ứng dụng cho dự án" là suy luận của chúng ta, không phải kết quả paper chứng minh. Con số lấy từ văn bản paper; nhiều hình trong bản trích xuất PDF bị vỡ nên chỗ nào không đọc chắc chắn được thì ghi rõ.

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

**Knowledge distillation (KD)** = huấn luyện mô hình nhỏ (student) bắt chước mô hình lớn (teacher). Với mô hình sinh tự hồi quy (mỗi bước sinh 1 token dựa trên các token trước), có hai cách phổ biến trước bài này:

- **Supervised KD** (kiểu Hinton/DistilBERT): lấy một tập dữ liệu cố định $(x,y)$ (nhãn gốc hoặc dữ liệu có sẵn), teacher gán xác suất từng token, student khớp phân phối đó bằng forward KL.
- **SeqKD** (Kim & Rush, note 15): cho teacher sinh trước cả chuỗi đầu ra, rồi SFT student trên chuỗi đó. Tốn chi phí sinh bằng teacher.

Cả hai đều dùng **tập chuỗi cố định**. Vấn đề: lúc suy luận student tự sinh từng token và gặp những tiền tố $y_{<n}$ do chính nó tạo ra (có thể lệch), trong khi lúc train chỉ thấy tiền tố "sạch". Sai ở bước đầu kéo theo sai các bước sau. Đây là **exposure bias / train-inference distribution mismatch**, vấn đề kinh điển của imitation learning (Ross & Bagnell; DAgger).

Vấn đề thứ hai: forward KL ép student phủ toàn bộ support của teacher. Student nhỏ không đủ sức, nên đặt xác suất vào token mà teacher cho xác suất thấp, dẫn đến mẫu sinh kém chất lượng khi sampling theo nhiệt độ (paper minh họa ở Figure A.16).

## 2. Ý tưởng chính

Coi distillation của LLM là bài toán **imitation learning với "expert tương tác"**: teacher là expert có thể được hỏi bất kỳ lúc nào "ở tiền tố này, bạn sẽ phân phối token tiếp theo thế nào?". Vậy ta để **student tự sinh chuỗi** (on-policy), rồi nhờ teacher chấm điểm từng token trên chính chuỗi đó (cho phân phối đầy đủ của teacher tại mỗi vị trí) và kéo student về phía teacher. Student thấy đúng loại tiền tố (kể cả sai) mà nó sẽ gặp khi suy luận, và nhận phản hồi ngay tại token sai. Thêm nữa, paper cho phép chọn **divergence** tuỳ ý (forward KL, reverse KL, JSD) thay vì cố định forward KL. Họ gọi khung này là **Generalized KD (GKD)**.

## 3. Phương pháp chi tiết

### 3.1 Ký hiệu

Student $p_S^\theta$, teacher $p_T$, input $x$, chuỗi đầu ra $y$ dài $L_y$. Độ lệch ở mức chuỗi là trung bình theo token của divergence $D$ giữa hai phân phối next-token:

$$D(p_T\|p_S^\theta)(y|x) := \frac{1}{L_y}\sum_{n=1}^{L_y} D\big(p_T(\cdot|y_{<n},x)\,\|\,p_S^\theta(\cdot|y_{<n},x)\big)$$

**Generalized JSD** (bị chặn, nội suy giữa forward và reverse KL), với $0<\beta<1$:

$$D_{JSD(\beta)}(P\|Q)=\beta\,D_{KL}\!\big(P\,\|\,\beta P+(1-\beta)Q\big)+(1-\beta)\,D_{KL}\!\big(Q\,\|\,\beta P+(1-\beta)Q\big)$$

Paper nhắc (Huszár 2015) rằng $\lim_{\beta\to0}D_{JSD(\beta)}/\beta=D_{KL}(P\|Q)$, nên $\beta$ nhỏ có gradient giống forward KL, $\beta$ gần 1 giống reverse KL.

Gloss: **forward KL** $D_{KL}(p_T\|p_S)$ là mean-seeking (student phải phủ mọi chỗ teacher có khối lượng); **reverse KL** $D_{KL}(p_S\|p_T)$ là mode-seeking (student dồn vào các mode của teacher, chấp nhận bỏ sót, ít đa dạng hơn).

### 3.2 Mục tiêu GKD

$$\boxed{\mathcal L_{GKD}(\theta)=(1-\lambda)\,\mathbb E_{(x,y)\sim(X,Y)}\big[D(p_T\|p_S^\theta)(y|x)\big]+\lambda\,\mathbb E_{x\sim X}\,\mathbb E_{y\sim p_S(\cdot|x)}\big[D(p_T\|p_S^\theta)(y|x)\big]}$$

- $\lambda\in[0,1]$ là **student data fraction** (tỉ lệ dữ liệu do student sinh).
- Không backprop qua quá trình sampling của student (ổn định, rẻ).
- $\lambda=0$ và forward KL = Supervised KD; $\lambda=1$ và forward KL = on-policy KD.
- ImitKD (Lin et al. 2020) và f-distill (Wen et al. 2023) được xem như biến thể "mixed" ($\lambda=0.5$) với forward KL và total variation.
- Giả định: student đã được **SFT** trước (đủ khả năng sinh chuỗi có chất lượng để teacher còn phản hồi được), giống quy trình hai bước SFT rồi RLHF.

### 3.3 Pseudo-code (Algorithm 1)

```
Cho: teacher p_T, student p_S(theta), dataset (X, Y)
Siêu tham số: lambda, divergence D, learning rate eta
for step k in 1..K:
    u ~ Uniform(0,1)
    if u <= lambda:
        lấy batch input x từ X
        sinh y ~ p_S(.|x)  (nhiệt độ 1 để giữ đa dạng, không lấy gradient qua sampling)
    else:
        lấy batch (x, y) từ dữ liệu cố định (nhãn gốc hoặc do teacher sinh)
    # teacher chỉ chạy forward để lấy logit trên chính y này
    loss = mean_over_tokens D( p_T(.|y_<n, x) || p_S(.|y_<n, x) )
    theta <- theta - eta * grad loss
```

Ví dụ tay: student sinh "Paris is the capital of Germ..." (sai ở token thứ 6). Với supervised KD, student chưa bao giờ thấy tiền tố sai này khi huấn luyện. Với GKD, teacher đưa phân phối tại vị trí "Germ" (nghiêng hẳn về "France") và student được kéo về đó ngay tại tiền tố lỗi.

### 3.4 Kết hợp RL fine-tuning

Vì GKD on-policy chỉ cần mẫu từ student, nó ghép tự nhiên với RLHF/RLAIF:

$$\mathbb E_{x}\Big[(1-\alpha)\,\mathbb E_{y\sim p_S^\theta}[r(y)]-\alpha\,\mathbb E_{y\sim p_S}\big[D(p_T\|p_S^\theta)(y|x)\big]\Big]$$

$\alpha=1$ là chỉ distillation. Paper khuyến nghị reverse KL hoặc JSD(0.9) nếu muốn sửa ít workflow RL hiện có (vì RLHF vốn dùng reverse KL để giữ gần policy gốc).

## 4. Thí nghiệm và kết quả chính

**Mô hình:** teacher T5-XL (~3B) đã SFT; student T5-small (77M), T5-base (250M), T5-large (800M), nhỏ hơn teacher 38x, 12x, 3.8x. Optimizer Adafactor. Mọi baseline xuất phát từ cùng student đã SFT.

**Biến thể GKD:** divergence thuộc {forward KL, reverse KL, JSD(0.1), JSD(0.5), JSD(0.9)}; $\lambda\in\{0, 0.5, 1\}$.

| Tác vụ | Dữ liệu / metric | Điểm đáng nhớ (theo văn bản paper) |
|---|---|---|
| Tóm tắt | XSum, ROUGE-2 | On-policy GKD vượt SeqKD, supervised KD, ImitKD, f-distill; JSD(0.9) tốt nhất ở cả greedy và sampling (Fig. 2) |
| Dịch | WMT14 en-de, BLEU | Teacher BLEU 28; student small 25.58, base 26.98; thuần on-policy tốt nhất (Fig. 6) |
| Suy luận | GSM8K + CoT, accuracy có calculator | Teacher Flan-T5-XL 27.9; student base 10.16 (greedy); on-policy vượt mọi baseline (Fig. 9) |
| Task-agnostic | FLAN2021 (5.36M mẫu, 62 tác vụ); MMLU, BBH | Reverse KL on-policy tốt nhất; teacher 52.4% MMLU / 41% BBH; student large 35.6% / 31.25% (Fig. 10) |

Các con số tổng hợp paper nêu: gain tương đối so với baseline, trung bình trên các cỡ student, là **2.1x** (tóm tắt), **1.7x** (dịch), **1.9x** (suy luận) (tính trên mức cải thiện so với student ban đầu). Hình 1 ghi mức tăng **+111%** (XSum), **+70%** (WMT), **+90%** (GSM8K) cho T5-small so với baseline tốt nhất (cách đọc chính xác của các phần trăm này phụ thuộc hình bị vỡ; ở đây chỉ nêu lại các con số xuất hiện trong chữ). Task-agnostic: cải thiện tuyệt đối **2% trên BBH và 1% trên MMLU** (cho T5-base). Ngoài ra, T5 nhỏ hơn PaLM (540B) 7000x vẫn vượt few-shot PaLM trên XSum.

**Data efficiency (Fig. 3):** distill T5-small với 1K (0.5%), 10K (5%), 50K (25%) mẫu XSum. On-policy GKD chỉ dùng 5% dữ liệu, không cần summary ground-truth, đã vượt supervised KD và ImitKD dùng toàn bộ dữ liệu có nhãn.

**Cấu hình (XSum/GSM8K):** 40,000 bước, batch 32, LR 3e-4 (T5-small XSum dùng 1e-3), warmup 2,000 bước, cooldown tuyến tính từ bước 30,000 đến 40,000. Nhiệt độ teacher 0.1 cho các thí nghiệm sampling trên XSum.

**GKD + RL (Fig. 5):** dùng điểm entailment của NLI T5-XXL làm reward. $\alpha$ càng lớn thì ROUGE-2 càng cao nhưng cải thiện factual consistency càng giảm. GKD+RL đạt ROUGE-2 cao hơn RLEF* và tính nhất quán sự thật cao hơn cả teacher.

**Self-distillation (Phụ lục A.1):** Flan-T5-large làm cả teacher (SFT GSM8K, 20.5%) và student (chưa train GSM8K, 14.4%). Student sau on-policy GKD vượt teacher.

## 5. Phân tích / ablation

- **Dữ liệu on-policy là yếu tố quan trọng nhất.** Trên XSum, WMT, GSM8K, biến thể on-policy và mixed luôn vượt biến thể chỉ dùng dữ liệu cố định. Trên GSM8K, hiệu năng tăng đều khi tỉ lệ dữ liệu student vượt 25% (Fig. 8).
- **Divergence tối ưu phụ thuộc tác vụ và cách decode.** Với greedy, chọn divergence ít ảnh hưởng; với sampling nhiệt độ 1, divergence mode-seeking (JSD(0.5), JSD(0.9), reverse KL) cho chất lượng cao hơn nhưng đa dạng thấp hơn (đo bằng Self-BLEU, Fig. 4). Hạ nhiệt độ làm các divergence gần nhau hơn, đồng thời đa dạng giảm.
- Trên WMT, JSD tổng quát tốt hơn forward/reverse KL, khoảng cách thu hẹp khi student lớn hơn. Trên instruction tuning, reverse KL tốt hơn nhiều so với forward KL; tác giả **giả thuyết** (không kiểm chứng) rằng tính mode-seeking giúp mô hình tập trung vào ý định chính của chỉ dẫn.
- **Chi phí:** sampling từ student làm tăng chi phí xấp xỉ 1.8x, 2x, 2.2x so với lấy mẫu từ dữ liệu cố định, ứng với tỉ lệ cỡ 38x, 12x, 3.8x (trên GSM8K). Tác giả lập luận chi phí chủ yếu nằm ở serving, nên overhead này đáng giá.

## 6. Giới hạn

**Tác giả nêu/ngụ ý:**
- Cần student đã SFT đủ tốt; không áp dụng cho student khởi tạo ngẫu nhiên.
- Chi phí sinh mẫu từ student trong vòng lặp (khoảng 2x).
- Divergence tốt nhất phụ thuộc tác vụ, phải tune (trade-off chất lượng/đa dạng).
- Mọi thí nghiệm dùng họ T5 (encoder-decoder) cỡ nhỏ.

**Chúng ta nhận xét (suy luận):**
- Không có thí nghiệm với decoder-only LLM hiện đại, chuỗi dài, hay teacher nhiều tỉ tham số khác tokenizer.
- Không backprop qua sampling nên là một ước lượng gradient thiên lệch (giống DAgger/behavior cloning on-policy chứ không phải policy gradient thật), nhưng ổn định.
- Teacher phải truy cập được logit ở mọi vị trí, nên không áp dụng cho teacher chỉ có API văn bản.
- Nếu chuỗi student sinh ra quá lệch, teacher có thể cho nhãn kém tin cậy trên tiền tố ngoài phân phối của chính nó (không được paper phân tích).

## 7. Liên hệ với các bài khác

- **02 Hinton KD / 16 DistilBERT:** GKD với $\lambda=0$, forward KL chính là supervised KD token-level.
- **15 Sequence-level KD:** SeqKD là SFT trên chuỗi teacher; GKD chuyển nguồn chuỗi sang student.
- **28 MiniLLM:** cùng bắt đầu từ reverse KL cho LLM nhưng tối ưu bằng policy gradient; GKD không backprop qua sampling.
- **30 DistiLLM:** đề xuất skew KL và cơ chế chọn lọc dữ liệu student (adaptive off-policy) để giảm chi phí on-policy của GKD.
- **31 Rethinking KL:** phân tích lại vì sao forward/reverse KL thực tế hội tụ giống nhau.
- **32 Speculative KD:** thêm teacher vào vòng sinh mẫu (interleaved sampling) để teacher sửa các token student tệ.
- **33 TAID, 34 DistiLLM-2:** các hướng tiếp theo về nội suy phân phối và loss tương phản.
- **47 ShortOPD:** (theo tên) on-policy distillation cho mô hình đã cắt; GKD là tổ tiên trực tiếp.

## 8. Ứng dụng cho dự án (suy luận)

Dự án: LightOnOCR-3-0.8B cắt xuống ~0.3B/0.1B, phục hồi từ teacher 4B.

- **Rất liên quan về nguyên lý.** Mô hình OCR-VLM sinh văn bản tự hồi quy; sau khi cắt (đặc biệt cắt neuron MLP) student sẽ có các lỗi lệch kiểu "lặp dòng, bỏ ký tự", nên exposure bias rất thật. Dữ liệu on-policy (student của ta sinh transcript trên ảnh trang, teacher 4B chấm logit từng token) cho tín hiệu đúng tại các token student hay sai.
- Student đã cắt phải được SFT/distill off-policy đủ để sinh chuỗi hợp lý (điều kiện "student đã SFT" của paper), rồi mới bật on-policy. Có thể dùng lịch $\lambda$ tăng dần.
- Với OCR, đầu ra gần như xác định (một đáp án đúng), nên mode-seeking (reverse KL/JSD cao) có lý, nhưng đây là suy đoán. Paper cho thấy ở greedy thì khác biệt divergence nhỏ.
- Chi phí: student 0.1-0.3B sinh nhanh, teacher 4B chỉ forward một lần trên chuỗi student. Chạy teacher batch trên GPU lớn khi train, không bị ràng buộc 8 GB (ràng buộc 8 GB chỉ cho inference).
- Tokenizer của teacher và student phải khớp (cùng họ), điều kiện của công thức token-level.
- Liên hệ D_lin/D_res: paper không đề cập; ta chỉ suy đoán rằng phần D_res (phi tuyến, khó phục hồi) cần tín hiệu mức token từ teacher trên đầu ra thật của student hơn là chỉ khớp feature.

## 9. Thí nghiệm gợi ý

1. Trên student 0.3B đã cắt: so SFT vs supervised KD ($\lambda=0$) vs on-policy ($\lambda=1$) vs mixed ($\lambda=0.5$), cùng ngân sách bước, đo CER/WER trên tập OCR giữ riêng.
2. Với $\lambda=1$, so forward KL, reverse KL, JSD(0.5) để xem OCR (decode greedy) có nhạy với divergence như paper dự đoán không.
3. Đo exposure bias: tính tỉ lệ lỗi theo vị trí token trong chuỗi dài (trang đầy chữ) trước và sau on-policy, xem lỗi tích luỹ có giảm không.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Exposure bias | Train trên tiền tố sạch, suy luận trên tiền tố tự sinh |
| On-policy KD | Train trên chuỗi student sinh, nhãn là phân phối token của teacher |
| GKD | $\mathcal L=(1-\lambda)\,\mathbb E_{data}[D]+\lambda\,\mathbb E_{y\sim p_S}[D]$ |
| $\lambda$ | Tỉ lệ dữ liệu do student sinh (0 = supervised, 1 = thuần on-policy) |
| JSD($\beta$) | Nội suy bị chặn giữa forward KL ($\beta\to0$) và reverse KL ($\beta\to1$) |
| Forward / reverse KL | Mean-seeking / mode-seeking |
| GKD + RL | $(1-\alpha)$ reward $-\;\alpha$ divergence on-policy |
| Chi phí | Khoảng 1.8x-2.2x so với lấy mẫu cố định (GSM8K) |

## 11. Kết luận ngắn

GKD biến KD cho LLM thành imitation learning với expert tương tác: để student tự sinh, teacher chấm từng token, và chọn divergence phù hợp tác vụ. Kết quả trên T5 (tóm tắt, dịch, GSM8K, instruction tuning) cho thấy dữ liệu on-policy luôn vượt dữ liệu cố định, và divergence nên chọn theo tác vụ/cách decode. Với dự án OCR, đây là công thức nền cần thử đầu tiên sau bước phục hồi off-policy.

Nguồn gốc: https://arxiv.org/abs/2306.13649
