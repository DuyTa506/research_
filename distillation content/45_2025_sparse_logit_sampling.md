# Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs (2025)

**Tác giả:** Anshumann, Mohd Abbas Zaidi, Akhil Kedia, Jinwoo Ahn, Taehwak Kwon, Kangwook Lee, Haejun Lee, Joohyung Lee (Samsung Research, Seoul; ba tác giả đầu đóng góp ngang nhau).
**Venue / arXiv:** arXiv:2503.16870 (v2, 24/07/2025); bài không nêu venue trong bản text.
**Nguồn:** https://arxiv.org/abs/2503.16870
**Chủ đề:** Distillation "offline": cache logit của teacher rồi huấn luyện student. Vì sao lưu Top-K rồi chuẩn hoá lại là ước lượng **chệch**, và cách lấy mẫu ngẫu nhiên (importance sampling) để có ước lượng **không chệch** với rất ít logit lưu trữ.
**Vị trí trong lộ trình:** Đọc trước: 02 (Hinton KD, soft target + KL), 15 (Sequence-level KD), 29 (GKD), 31 (Rethinking KL), 40 (Gemma 2, distillation khi pre-train). Bài này bổ sung: phân tích lý thuyết về việc **lưu thưa** phân phối teacher. Đọc tiếp: 46 (Distil-Whisper, pseudo-label + KL), 47 (ShortOPD, cũng dùng top-100 + một "bin đuôi").

> Đây là ghi chú học tập bằng tiếng Việt, không phải bản dịch. Các số liệu lấy từ paper; phần "Ứng dụng cho dự án" và mọi nhận định về OCR/LightOnOCR là suy luận của chúng ta, không phải kết luận của tác giả.

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

**Distillation theo logit.** Student được huấn luyện để khớp *toàn bộ phân phối* xác suất token kế tiếp của teacher (soft target), không chỉ token đúng (hard label). Mất mát chuẩn là KL thuận $\mathrm{KL}(t\,\|\,p)$ với $t$ là xác suất teacher, $p$ là xác suất student (xem 02).

**Vấn đề chi phí.** Với LLM, mỗi vị trí token có một vector xác suất cỡ $|V|$ (từ vựng, ~100k–128k). Chạy teacher *online* cùng lúc với student tốn GPU. Chạy teacher *một lần*, lưu kết quả ("offline", "cached logits") rồi tái dùng cho nhiều lần train student rẻ hơn nhiều, nhưng lưu đủ $|V|$ số là bất khả thi: paper ước tính 128 PB cho 1T token với Llama 3, và 10 PB cho 100B token với từ vựng 100k (Bảng 5).

**Giải pháp phổ biến trước đó: lưu Top-K.** Chỉ giữ K xác suất lớn nhất (có thể thêm Top-p: giữ đủ khối lượng xác suất p). Các nghiên cứu trước (Raman 2023, Peng 2024) phải lưu rất nhiều logit (cỡ 6400 mỗi token) hoặc thấy hiệu năng *giảm*. Paper hỏi: tại sao Top-K đáng tin về trực giác lại kém, và có cách nào lưu chỉ ~12 token mà vẫn bằng FullKD không?

Thuật ngữ cần biết:
- **FullKD:** distillation dùng toàn bộ phân phối teacher (trần hiệu năng).
- **Calibration / ECE:** mô hình "hiệu chuẩn tốt" khi xác suất nó đưa ra khớp tần suất đúng thực tế. ECE (Expected Calibration Error) đo sai lệch này; thấp là tốt.
- **Estimator không chệch (unbiased):** kỳ vọng của ước lượng bằng đúng giá trị thật.
- **Importance sampling:** lấy mẫu từ phân phối khác $q$ rồi nhân trọng số $t/q$ để vẫn ước lượng đúng kỳ vọng theo $t$.

## 2. Ý tưởng chính

Giữ Top-K rồi chuẩn hoá về tổng 1 làm xác suất các token đầu bảng bị **phóng đại** (chia cho khối lượng $a<1$) và toàn bộ đuôi bị đặt bằng 0. Student học đúng mục tiêu sai đó nên trở nên **quá tự tin**, mis-calibrated, đồng thời mất thông tin ở đuôi (kể cả khi token đúng nằm trong đuôi). Thay vì cắt, paper **lấy mẫu ngẫu nhiên** N token từ phân phối teacher (có lặp) và dùng tần suất $c_i/N$ làm mục tiêu thưa. Mục tiêu này *không chệch*, nên gradient kỳ vọng ở logit bằng đúng gradient của FullKD. Chỉ cần ~12 token khác nhau mỗi vị trí (24 bit/token, 3 byte) là đạt hiệu năng ngang FullKD, với chi phí huấn luyện chỉ cao hơn CE ~10%.

## 3. Phương pháp chi tiết

### 3.1 Gradient của KL qua softmax

Ký hiệu logit student $x$, $p=\mathrm{softmax}(x)$, mục tiêu $t$. Với $L=\sum_i t_i\log(t_i/p_i)$ (Phụ lục A.1):

$$
\frac{\partial L}{\partial x_j}=\Big(\sum_{i} t_i\Big)p_j-t_j .
$$

Khi $t$ là phân phối đầy đủ ($\sum t_i=1$): $\boxed{\partial L/\partial x_j=p_j-t_j}$. Cực tiểu đạt khi $p=t$. (Cross-entropy cho cùng gradient này, vì chỉ hơn KL một hằng số không phụ thuộc student.)

### 3.2 Vì sao Top-K chệch: chứng minh ngắn

Gọi $K$ là tập token được giữ, $a=\sum_{j\in K}t_j<1$.

**Biến thể có chuẩn hoá:** $t^s_i=t_i/a$ với $i\in K$, 0 ngoài $K$. Sai khác kỳ vọng so với $t$: $\mathbb{E}[t^s_i]-t_i=t_i(1/a-1)>0$ trong $K$ và $-t_i$ ngoài $K$. Đây là ước lượng chệch. (Phụ lục A.3 còn chứng minh Top-K là cách chọn tập *giảm tối thiểu sai số L1* cho từng token: $L_1=2(1-a)$, nhỏ nhất khi $a$ lớn nhất. Nghĩa là Top-K "gần nhất" về L1 nhưng vẫn chệch.)

**Biến thể không chuẩn hoá:** dùng KL chỉ trên $K$, $L=\sum_{i\in K}t_i\log(t_i/p_i)$. Theo công thức 3.1 với $\sum_{i\in K}t_i=a$:

$$
\frac{\partial L}{\partial x_j}=a\,p_j-t_j\ (j\in K),\qquad \frac{\partial L}{\partial x_j}=a\,p_j\ (j\notin K).
$$

Đặt gradient bằng 0: $p_j=t_j/a$ trong $K$ (**phóng đại $1/a$ lần**) và $p_j=0$ ngoài $K$ (đẩy đuôi về 0). Vậy dù có chuẩn hoá hay không, student hội tụ tới phân phối "nâng cấp" và quá tự tin (Phụ lục A.4). Đó là cơ chế của mis-calibration ở Bảng 1.

**Ví dụ nhỏ (của chúng ta, không có trong paper).** $t=(0.5,0.3,0.1,0.05,0.05)$, $K=2$: $a=0.8$, mục tiêu Top-2 chuẩn hoá $=(0.625,0.375,0,0,0)$. Student học tin token đầu 62.5% trong khi teacher chỉ 50%, và coi 3 token còn lại là bất khả.

Thêm vấn đề thứ hai: **mất đuôi.** Nếu token thật nằm trong đuôi teacher, Top-K không cho tín hiệu nào, kém cả CE.

### 3.3 Các cách vá không đủ (Mục 3)

- **Label smoothing:** chia đều phần xác suất còn lại cho mọi lớp. Sai vì phân phối token thực gần Zipf, không đều.
- **Ghost token:** thêm 1 token ảo gom $1-\sum_{K}t_i$ ở cả teacher lẫn student. Gradient trong $K$ đúng bằng FullKD; ngoài $K$ gradient là $p_j\big(\frac{1-\sum_K t}{1-\sum_K p}-1\big)$ (A.5). Tốt hơn, nhưng đuôi chỉ được giám sát gián tiếp.
- **Naive fix:** dồn phần xác suất còn lại vào token ground truth. Cải thiện mạnh nhưng vẫn cần ~100 token mới sát FullKD.

### 3.4 Random Sampling KD

Với phân phối đề xuất $q_i\propto t_i^{\tau}$ (nhiệt độ lấy mẫu $\tau$), lấy mẫu $N$ lần có lặp; mỗi lần xuất hiện của token $i$ nhận tỷ số hợp lý $t_i/q_i$; cộng lại rồi chuẩn hoá thành $t^s$. Với $\tau=1$ thì $q=t$, tỷ số bằng 1:

$$
\boxed{t^s_i=\frac{c_i}{N}},\qquad \mathbb{E}[t^s_i]=t_i,\qquad \mathrm{Var}[t^s_i]=\frac{t_i(1-t_i)}{N}
$$

(công thức phương sai là suy ra từ phân phối nhị thức, paper không viết ra). $c_i$ là số lần token $i$ được rút trong $N$ lần. Chỉ tối đa $N$ phần tử khác 0, thực tế ít hơn nhiều (N=50 cho ≈12 token khác nhau, Phụ lục C; quan hệ gần power-law).

Mất mát: $L=\sum_i t^s_i\log(t^s_i/p_i)$ trên các phần tử khác 0; với $\tau=1$ bằng tổng cross-entropy giữa các token được rút và dự đoán student. Gradient theo logit: $g_j^s=p_j-t^s_j$ nên

$$
\boxed{\mathbb{E}[g^s_j]=p_j-\mathbb{E}[t^s_j]=p_j-t_j=g_j^{\text{FullKD}}}
$$

(A.7: chỉ cần ước lượng không chệch là gradient được bảo toàn kỳ vọng). Phương sai quan trọng: $\tau=0$ (lấy đều trên từ vựng) làm train phân kỳ; $q^\*\propto t|f|$ là phân phối có phương sai thấp nhất theo lý thuyết, và mô phỏng số cho $\tau\in[0.8,1.2]$ gần tối ưu. Lưu ý (của chúng ta): với $\tau\neq1$ có bước chuẩn hoá nên ước lượng thuần tuý không còn chính xác không chệch; paper dùng $\tau=1$ cho mọi thí nghiệm chính.

### 3.5 Pseudo-code (offline)

```
# Giai đoạn 1: chạy teacher một lần, lưu mục tiêu thưa
for batch in corpus:
    t = softmax(teacher(batch))              # [B, T, V]
    ids = multinomial(t, N, replacement=True) # [B, T, N]
    uniq, cnt = unique_with_counts(ids)       # ~12 token khác nhau / vị trí
    save(uniq (17 bit), cnt (7 bit))          # 3 byte / token; cnt/N = t^s

# Giai đoạn 2: huấn luyện student
for batch in corpus:                          # PHẢI cùng thứ tự/ngữ cảnh với giai đoạn 1
    logp = log_softmax(student(batch))
    ts   = cnt / N                            # mục tiêu thưa
    loss = -(ts * logp.gather(uniq)).sum(-1).mean()   # = KL tới hằng số
```

Chi tiết lưu trữ (Phụ lục D.1): từ vựng 100k cần 17 bit cho id, còn 7 bit cho phần xác suất. Vì $N=50$ nên $c_i\le 50<2^7$, lưu tử số chính xác. Với Top-K phải lượng tử hoá xác suất 7 bit; paper dùng "ratio encoding" trên xác suất đã sắp xếp để giảm sai số lượng tử.

## 4. Thí nghiệm và kết quả chính

Thiết lập: student LLaMA-style 300M (teacher 3B, 10B token = 1.5x Chinchilla-optimal) và 3B (teacher Llama-3-8B, 100B token trên Fineweb-edu); loss thuận KL; 4 task NLU 0-shot (HellaSwag, ARC-Easy, LAMBADA, PIQA) và instruction following sau SFT.

**Bảng 1: Top-K thuần (3B→300M).** "% CE→FullKD" là phần khoảng cách giữa CE và FullKD được thu hẹp.

| Token duy nhất | LM loss | % CE→FullKD | ECE % |
|---|---|---|---|
| CE | 2.81 | 0% | 1.2 |
| 3 | 3.04 | −395% | 10.6 |
| 12 | 2.87 | −99% | 4.7 |
| 25 | 2.82 | −21% | 3.2 |
| 50 | 2.80 | 5% | 2.2 |
| 100 | 2.77 | 55% | 1.1 |
| 300 | 2.76 | 77% | 1.5 |
| FullKD | 2.75 | 100% | 0.7 |

Top-K dưới ~25 token còn **tệ hơn CE thuần**. Dùng Top-p 0.98 (K=100) chỉ đạt 47%.

**Bảng 2: cách vá (300M).** Label smoothing (50 tok): −73%; Ghost token (50 tok): 59%; Naive fix Top-k 5/10/20/50/100: 44/61/73/83/97% (ECE 4.4/3.3/2.3/1.3/1.2).

**Bảng 3: gradient so với FullKD (300M, một batch).**

| Phương pháp | Góc lệch | Tỷ lệ chuẩn |
|---|---|---|
| Top-K 12 | 58° | 2.4 |
| Top-K 50 | 48° | 1.8 |
| Top-K 300 | 30° | 1.3 |
| Random Sampling (12 tok) | 4° | 1.0 |

**Bảng 6 (3B→300M, 10B token).** Random Sampling với 2.4 / 5.0 / 12.1 / 24.5 / 57.0 token: LM loss 2.77 / 2.75 / 2.75 / 2.75 / 2.74; FullKD 2.75; CE 2.81. ECE ≤ 1.1% mọi mức. Speculative acceptance 61.85% (12 tok) so với 62.02% (FullKD).

**Bảng 8 (8B→3B, 100B token):**

| Phương pháp | LM loss | ECE % | Spec. accept % | 0-shot | IF SFT |
|---|---|---|---|---|---|
| CE | 2.37 | 0.3 | 71.1 | 55.6 | 54.5 |
| Top-K 12 | 2.50 | 4.7 | 73.0 | 56.6 | 57.7 |
| Top-K 50 | 2.40 | 1.8 | 73.1 | 57.1 | 58.3 |
| **Ours (12)** | 2.35 | 0.2 | 73.2 | 57.5 | 59.4 |
| Ours (12)+ | 2.32 | 1.7 | 73.5 | 57.9 | 59.1 |
| FullKD | 2.34 | 0.2 | 73.4 | 57.5 | 58.4 |

LLM-as-judge (Bảng 9, 8B→3B): trung bình CE 60.2, Top-K 12 58.0, Top-K 50 60.6, **Ours 65.6**, FullKD 62.2.

**Tốc độ (Bảng 4):** Random Sampling 2.6x / 1.73x (300M / 3B) so với FullKD = 1.0x; CE 2.9x / 1.77x. Cache implementation nhanh hơn FullKD 1.7–2.6x, chậm hơn CE ~10%.

**Lưu trữ (Bảng 5, 100B token):** FullKD 10 000 TB; Top-K 300: 90 TB; **Ours (12 logit, 3 byte): 3.6 TB** (gấp 25 lần ít hơn Top-300); CE: 0.3 TB.

**Khác:** Qwen-style 0.5B (10B token; text nói teacher Llama-3-8B, caption Bảng 14 ghi 3B→0.5B): LM loss 2.99 (CE) → 2.95. Cải thiện so với CE tăng theo kích thước student (100M–3B, Hình 4).

## 5. Phân tích / ablation

- **Nhiệt độ lấy mẫu (Bảng 12):** $\tau=0$ phân kỳ; 0.8 / 1.0 / 1.2 đều ~2.74–2.75 LM loss. Chọn 1.0.
- **Hàm mất mát (Bảng 15):** KL thuận 2.75 tốt nhất; MSE 3.37; L1 5.38; KL nghịch tách rời bị ∞.
- **Teacher adaptation:** nếu dữ liệu student khác dữ liệu pre-train teacher, nên fine-tune teacher ngắn trước (Bảng 13: 2.99 CE; 2.98 không adapt; 2.96 có adapt).
- **Cải tiến trực giao (Bảng 10):** trộn CE (trọng số 0.1) + tăng LR cho token "khó" (tỷ lệ 2.0) cho 125% CE→FullKD (vượt FullKD), nhưng làm student **under-confident** và giúp ít hơn sau instruction tuning (caveat của tác giả).
- **So với prior work (Bảng 11):** Peng 2024: −78%/−47%; Raman 2023: 57%/5%; Ours: 100%.
- **Căn chỉnh ngữ cảnh teacher/student (Phụ lục D.3, Bảng 16):** nếu seed shuffle khác nhau thì tiền tố ngữ cảnh khác nhau sau ranh giới tài liệu; hiệu năng cache chỉ đạt 79–90% so với online; cùng seed và seq len 4096: 96%. Bài học thực hành: logit cache phải khớp *đúng* ngữ cảnh lúc student train.

## 6. Giới hạn

**Do tác giả nêu:** chỉ thử tới 3B và 100B token; chưa thử representation matching (lưu activation trung gian); chưa thử sơ đồ lấy mẫu tinh vi hơn (ví dụ thiết kế thí nghiệm tối ưu).

**Chúng ta nhận thấy:**
- Toàn bộ thử nghiệm là **pre-training ngôn ngữ**, teacher base (không instruct), được mô tả là hiệu chuẩn gần hoàn hảo. Paper nói alignment có thể làm hỏng calibration (trích dẫn công trình khác), nên với teacher đã SFT/RL kết quả chưa được kiểm chứng.
- Không có thí nghiệm trên sinh chuỗi dài tự hồi quy do student tự lấy mẫu (on-policy); cache offline vốn không sửa được exposure bias.
- Random Sampling cần phân phối **đầy đủ** lúc cache. Nếu dữ liệu cũ đã lưu Top-K thì không thể sửa lại bằng lấy mẫu.
- Chỉ so sánh trên metric loss/NLU; chưa có phân tích chi phí khi sampling nhiều vòng cho chuỗi cực dài.

## 7. Liên hệ với các bài khác

- **02 (Hinton KD):** nền tảng KL với soft target; bài này chỉ ra cách *nén* soft target để lưu.
- **15 (Sequence-level KD) / 46 (Distil-Whisper):** pseudo-label là mẫu hard từ teacher. Random Sampling KD có thể coi là "lấy mẫu N token tại **mỗi vị trí**" thay vì cả chuỗi.
- **31 (Rethinking KL) / 29 (GKD):** bàn về chọn divergence; bài này (Bảng 15) lại thấy KL thuận tốt nhất trong offline pre-training.
- **40 (Gemma 2):** paper nêu Gemma 3 dùng cách tương tự (lấy mẫu logit teacher theo xác suất teacher, cross-entropy trên token được lấy mẫu) ở quy mô 27B/14T token.
- **47 (ShortOPD):** dùng top-100 logit + **một bin đuôi gộp** (khác với lấy mẫu); xem phần Ứng dụng.
- **42 (Distillation scaling laws):** paper này trích dẫn công trình đó khi nói Top-K gây lệch.

## 8. Ứng dụng cho dự án (suy luận)

Dự án của bạn lưu **top-20 logprob** của teacher 4B cho các chuỗi transcript dài, gần như tất định. Các hệ quả (đều là suy luận):

1. **Mức độ chệch phụ thuộc vào $a=\sum_{K}t$.** Với OCR gần tất định, ở đa số vị trí token đúng có xác suất ~0.99+, nên top-20 phủ gần hết khối lượng, $a\approx1$ và hệ số phóng đại $1/a$ gần 1: bias nhỏ *ở các vị trí dễ*. Bias lớn tập trung ở vài vị trí khó (ký tự mờ, bảng, công thức), chính là nơi student cần học nhất. Hãy **đo trực tiếp phân bố $a$** trên cache của bạn trước khi lo lắng.
2. **Đừng chuẩn hoá lại top-20 một cách mù quáng.** Cách rẻ nhất để giảm bias mà không cần lấy mẫu lại: lưu hoặc suy ra khối lượng còn lại $1-\sum_K t_i$ (từ logprob top-20 là tính được) và dùng **ghost token** (bin đuôi) trong cả teacher lẫn student, như paper (A.5, 59% ở 50 token) hoặc như bin đuôi của ShortOPD (47). Hoặc "naive fix": dồn phần dư cho token ground truth khi nó nằm ngoài top-20 (73% ở K=20 trong Bảng 2, với bối cảnh pre-train).
3. **Random Sampling KD không áp dụng ngay được** với cache top-20 đã có (cần phân phối đầy đủ khi cache). Nếu còn chạy lại teacher, bạn có thể rút $N\approx 50$ mẫu/vị trí, nhưng với chuỗi gần tất định hầu hết mẫu sẽ trùng một token (tiết kiệm lưu trữ, nhưng cũng nghĩa là tín hiệu đuôi chỉ xuất hiện ở các vị trí không chắc chắn, đúng điều ta muốn).
4. **Căn chỉnh ngữ cảnh (D.3):** với VLM, đảm bảo teacher và student thấy *cùng* ảnh, cùng tiền xử lý/cắt (resize, tile), cùng prompt và cùng chuỗi tiền tố. Lệch nhỏ trong tiền xử lý ảnh có thể gây hiệu ứng tương tự lệch seed shuffle.
5. **Teacher đã SFT có thể hiệu chuẩn kém** (suy luận từ lưu ý của paper); nên kiểm tra ECE của teacher 4B OCR trước khi coi xác suất của nó là "chân lý".

## 9. Thí nghiệm gợi ý

1. **Đo bias thực tế:** trên 1–2k trang, tính phân bố $a$ theo vị trí, và so sánh KL giữa phân phối đầy đủ (chạy lại teacher trên một tập nhỏ) với phiên bản top-20 chuẩn hoá. Báo cáo bias riêng cho token sai/khó.
2. **So 3 mục tiêu từ cùng cache top-20:** (a) chuẩn hoá top-20; (b) top-20 + ghost token; (c) top-20 + phần dư dồn vào GT. Cùng student đã cắt, cùng số bước. Đo CER và ECE.
3. **Mini Random Sampling:** chạy lại teacher trên tập con, lưu N=50 mẫu/vị trí, so với (b) về CER ở các vị trí teacher entropy cao.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý | Ghi chú |
|---|---|---|
| Gradient KL tại logit | $\partial L/\partial x_j=(\sum_i t_i)p_j-t_j$ | $=p_j-t_j$ nếu $t$ đầy đủ |
| Top-K có chuẩn hoá | $t^s_i=t_i/a$, $a=\sum_K t$ | chệch, quá tự tin |
| Điểm tối ưu student với Top-K | $p_j=t_j/a$ (trong K), 0 (ngoài K) | từ gradient bằng 0 |
| Ghost token | thêm token đại diện $1-\sum_K t$ | gradient trong K đúng như FullKD |
| Random Sampling KD | $t^s_i=c_i/N$, $q=t^\tau$, $\tau=1$ | không chệch |
| Bảo toàn gradient | $\mathbb{E}[p_j-t^s_j]=p_j-t_j$ | điều kiện: ước lượng không chệch |
| Lưu trữ | 17 bit id + 7 bit count = 3 byte | N=50 ⇒ ~12 token duy nhất |
| ECE | sai lệch hiệu chuẩn kỳ vọng | thấp = tốt |

## 11. Kết luận ngắn

Lưu Top-K logit và chuẩn hoá là ước lượng chệch (phóng đại đầu, xoá đuôi), gây student quá tự tin và có thể tệ hơn CE nếu K nhỏ. Lấy mẫu N token từ phân phối teacher cho mục tiêu thưa không chệch, bảo toàn gradient kỳ vọng và chỉ cần ~12 token/vị trí (3.6 TB cho 100B token) để ngang FullKD. Với dự án của bạn, bài học thực dụng nhất không phải là đổi sang lấy mẫu mà là: **đừng chuẩn hoá top-20, hãy giữ khối lượng đuôi** (ghost token), và kiểm tra bias ở các vị trí khó.

Nguồn: https://arxiv.org/abs/2503.16870
