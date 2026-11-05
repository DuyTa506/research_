# MiniPLM: Knowledge Distillation for Pre-Training Language Models (2025)

**Tác giả:** Yuxian Gu, Hao Zhou, Fandong Meng, Jie Zhou, Minlie Huang (Tsinghua CoAI; WeChat AI, Tencent).  
**Venue / arXiv:** arXiv:2410.17215 (v3, 19 Mar 2025); ICLR 2025 (theo đầu trang paper).  
**Nguồn:** https://arxiv.org/abs/2410.17215  
**Chủ đề:** Knowledge distillation ở giai đoạn **pre-training** nhưng làm ở mức **dữ liệu**: teacher chỉ dùng để chọn lại (re-sample) corpus, không cần chạy teacher lúc train student. Phương pháp "Difference Sampling" so sánh teacher lớn với một mô hình tham chiếu nhỏ.  
**Vị trí trong lộ trình:** Đọc trước: 02 (Hinton KD), 15 (Sequence-level KD), 28 (MiniLLM), 40 (Gemma 2), 41–42 (capacity gap, distillation scaling laws). Bài này thêm: một phương án KD **offline, không phụ thuộc tokenizer**, và đưa ra lý lẽ rằng chi phí teacher online là điểm nghẽn. Tiếp theo: 45 (Sparse logit sampling) giải bài toán lưu logit theo hướng khác.

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch paper. Mọi con số lấy từ paper; phần "Ứng dụng cho dự án" là suy luận của chúng tôi. MiniPLM được thiết kế cho **pre-training từ đầu** với corpus rất lớn; dự án của bạn là recovery sau pruning nên mức liên quan chỉ ở mức ý tưởng (xem Mục 8).

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

Các cách làm KD ở fine-tuning khi đem sang pre-training đều có nhược điểm (Hình 2 của paper):

- **Online KD** (Vanilla KD kiểu DistilBERT/Minitron, MiniLLM): teacher phải chạy forward trong lúc train để cho phân phối từng token. Tốn compute thêm; khi tính theo FLOPs cố định thì lợi ích bị triệt tiêu (các phương pháp KD đều "tương đương hoặc tệ hơn" Pre-Train w/o KD trong thiết lập kiểm soát FLOPs). Ngoài ra teacher và student phải dùng chung tokenizer. Không thể tính sẵn rồi lưu: paper tính 50B token × từ vựng 150K × 4 byte ≈ **30 PB**.
- **Offline KD** (SeqKD, Kim & Rush 2016): teacher sinh sẵn văn bản, student học từ đó. Rẻ và độc lập tokenizer, nhưng văn bản do LM sinh dễ **dễ đoán và thiếu đa dạng** (student overfit mẫu phổ biến), như hiện tượng model collapse (Shumailov et al. 2024).

**Reverse KL / mode-seeking** (gloss): $\mathrm{KL}(q_\theta\Vert p)$ phạt student khi đặt xác suất ở nơi teacher cho xác suất thấp, nên student bám vào các "đỉnh" (mode) của teacher. MiniLLM (28) dùng loss này khi fine-tune, bằng PPO.

## 2. Ý tưởng chính

Thay vì bắt student khớp phân phối token của teacher, MiniPLM **sửa lại phân phối dữ liệu pre-training** theo hiểu biết của teacher: với mỗi đoạn văn trong corpus, tính độ lệch log-xác suất giữa teacher lớn và một mô hình tham chiếu nhỏ (cùng họ, train trên một tập nhỏ), rồi giữ lại những đoạn có hiệu số lớn nhất. Những đoạn mà mô hình lớn dự đoán tốt hơn nhiều so với mô hình nhỏ là kiến thức "khó nhưng có giá trị"; những đoạn cả hai đều đoán dễ (lặp, rập khuôn) bị bỏ bớt; những đoạn teacher lại đoán tệ hơn mô hình nhỏ (nhiễu) bị loại. Student sau đó pre-train như bình thường bằng cross-entropy trên corpus đã lọc. Vì chỉ cần **một con số cho mỗi đoạn văn**, teacher chạy một lần offline và tái sử dụng cho nhiều student, kể cả khác họ.

## 3. Phương pháp chi tiết

### 3.1 KD như tối đa hóa phần thưởng

Reverse KL được viết lại thành bài toán reward:

$$
\theta^\star=\arg\min_\theta \mathrm{KL}[q_\theta\Vert p]=\arg\max_\theta \mathbb E_{x\sim q_\theta}\, r(p,q_\theta,x),\qquad r(p,q_\theta,x)=\log\frac{p(x)}{q_\theta(x)}
$$

Student được thưởng khi sinh văn bản $x$ mà teacher thích (giá trị $\log p(x)$ lớn) và đa dạng (giá trị $\log q_\theta(x)$ nhỏ). Cách tối ưu đơn giản là **Best-of-N**: sinh $M$ ứng viên từ $q_\theta$, giữ $K$ cái có reward cao nhất (Eq. 2), rồi train trên đó. Tuy nhiên cách này cần sinh từ student và tính reward bằng student, nên dữ liệu không dùng lại được cho student khác.

### 3.2 Difference Sampling

Hai thay đổi để đưa Best-of-N về dạng offline, tái sử dụng được:

1. **Lấy top-K từ corpus gốc $D$ thay vì từ mẫu sinh của student.** Mệnh đề 2.1: với $N,M\to\infty$ và hàm $r$ đơn ánh, xác suất tập top-$K$ của hai mẫu độc lập (từ $p_1$ và $p_2$ có hỗ trợ dương mọi nơi) trùng nhau tiến tới 1. Đây là lập luận hợp lý hóa, không phải bảo đảm cho trường hợp thực.
2. **Thay $q_\theta$ bằng mô hình tham chiếu nhỏ $p_{ref}$**, train trên tập con $D_{ref}$ lấy ngẫu nhiên đồng đều từ $D$. Lý do: độ khó tương đối của các mẫu dữ liệu khá nhất quán giữa các mô hình (Ethayarajh et al. 2022).

Kết quả:

$$
\boxed{D'=\operatorname{top\text{-}K}\left\{\log\frac{p(x)}{p_{ref}(x)}\;\middle|\;x\in D-D_{ref}\right\}}\qquad(\text{Eq. 4}),\quad K=\alpha\,|D-D_{ref}|
$$

Student train bằng loss pre-training thông thường trên $D'$:

$$
\mathcal L(q_\theta,D')=-\frac1{|D'|}\sum_{x\in D'}\frac1{|x|}\sum_{t=1}^{|x|}\log q_\theta(x_t\mid x_{<t})
$$

Ba hiệu ứng (Hình 3b): (1) $\log\frac{p}{p_{ref}}\gtrsim 0$ → mẫu dễ và phổ biến, bị down-sample; (2) $\gg 0$ → mẫu khó, đa dạng, được up-sample; (3) $<0$ → teacher đoán tệ hơn mô hình nhỏ → nhiễu/độc hại, bị loại.

### 3.3 Pseudo-code

```
# Offline, one time per teacher
D_ref  <- uniform_sample(D, small)                  # vd 5B token
p_ref  <- train_from_scratch(small_LM, D_ref)       # vd 104M tham số
for x in D - D_ref:                                  # x: một sequence 1024 token
    score[x] = logp_teacher(x) - logp_ref(x)        # sequence-level, 1 float / x
D_prime = top_K(score, K = alpha * |D - D_ref|)     # vd alpha = 0.5

# Per student (any size / tokenizer / architecture)
student = pretrain_from_scratch(student_LM, D_prime, loss=next_token_CE)
```

### 3.4 Ví dụ nhỏ

Giả sử ba đoạn văn có log-xác suất tổng cộng (teacher, tham chiếu): A = (−100, −102) lặp rập khuôn, hiệu số = +2; B = (−120, −170) kiến thức chuyên ngành mà mô hình nhỏ không biết, hiệu = +50; C = (−200, −150) chuỗi ký tự nhiễu mà teacher "ngạc nhiên" hơn, hiệu = −50. Difference Sampling với $\alpha$ đủ nhỏ giữ B, có thể giữ ít A, loại C. (Số này do chúng tôi bịa để minh họa, không lấy từ paper; trong paper các đoạn đều dài 1024 token nên so sánh loss trung bình tương đương so sánh tổng.)

### 3.5 Chi phí và lưu trữ

Mỗi sequence chỉ cần một số thực, nên 50B token với sequence 1024 chỉ cần khoảng **200MB**. Bảng 8–9: huấn luyện mô hình tham chiếu 2,6 giờ; inference teacher trên corpus 68 giờ; inference tham chiếu 7 giờ; top-K 10 phút (8×A100 80GB, thiết lập của họ). Student train mất 68 giờ cho Pre-Train và MiniPLM, còn Vanilla KD 139 giờ. Có biến thể dùng "proxy model" (Phụ lục E) hồi quy $r$ từ hidden state của tham chiếu, chỉ cần 0,1B token đã tính nhãn.

## 4. Thí nghiệm và kết quả chính

**Thiết lập.** Kiến trúc Qwen-1.5; teacher Qwen-1.5 1,8B; student 200M, 500M, 1,2B train từ đầu trên Pile; mỗi student tối đa 50B token (sequence 1024). Tham chiếu 104M train trên 5B token. Thiết lập chính: $D$ gồm 105B token (5B cho $D_{ref}$, 100B để lấy mẫu), $\alpha=0{,}5$ → $D'$ 50B token, 1 epoch. Baseline: Pre-Train w/o KD, Vanilla KD (forward KL online), SeqKD (teacher sinh 256 token tiếp sau 768 token prompt), MiniLLM (reverse KL + PPO). Với phương pháp online, số bước train bị giảm để **tổng FLOPs train bằng nhau**. Đánh giá: zero-shot 9 tác vụ (HS, LAM, Wino, OBQA, ARC-e, ARC-c, PIQA, SIQA, Story) và test loss trên một tập con DCLM.

**Bảng 1: Điểm trung bình (Avg.) trên 9 tác vụ**

| Student | Pre-Train w/o KD | Vanilla KD | MiniLLM | SeqKD | **MiniPLM** |
|---|---|---|---|---|---|
| 200M | 39,9 | 39,9 | 39,0 | 39,7 | **41,3** |
| 500M | 43,2 | 43,6 | 41,5 | 42,3 | **44,8** |
| 1,2B | 44,9 | 45,4 | 43,6 | 44,3 | **46,6** |

Nhận xét của paper: chỉ Vanilla KD hơn Pre-Train w/o KD (với student đủ lớn) trong baseline; MiniPLM tốt nhất ở hầu hết tác vụ; mức cải thiện so với Vanilla KD giữ ổn khi tăng cỡ student (Hình 1b).

**Hiệu quả compute.** MiniPLM đạt cùng độ chính xác với Vanilla KD (student 500M) với compute ít hơn **2,2×** (Hình 1a); lợi thế so với Pre-Train w/o KD còn rõ hơn ở 500M và 1,2B.

**Loss ngoại suy (Bảng 2, theo Scaling Law của Hoffmann et al.; $L_{1T}$, $L_{10T}$ là loss khi xử lý 1T và 10T token):**

| Student | Pre-Train w/o KD | Vanilla KD | MiniPLM |
|---|---|---|---|
| 200M | 3,35 / 3,32 | 3,39 / 3,35 | 3,28 / 3,26 |
| 500M | 3,12 / 3,08 | 3,12 / 3,07 | 3,06 / 3,04 |
| 1,2B | 2,98 / 2,94 | 2,95 / 2,91 | 2,92 / 2,88 |

(Mỗi ô là $L_{1T}$ / $L_{10T}$. Đây là **ngoại suy**, không phải train thật ở 10T.)

**KD khác họ mô hình (Bảng 3).** Teacher/tham chiếu họ Qwen; student Llama3.1 212M và Mamba 140M. Điểm trung bình / loss DCLM: Llama3.1: Pre-Train 41,0 / 3,52, SeqKD 40,8 / 3,54, MiniPLM 41,8 / 3,43. Mamba: Pre-Train 41,6 / 3,24, SeqKD 41,0 / 3,27, MiniPLM 42,6 / 3,15. Vanilla KD và MiniLLM không áp dụng được vì khác tokenizer.

**Thiết lập thiếu dữ liệu (Mục 3.4).** $D$ cố định 50B token, huấn luyện nhiều epoch trên $D'$: $\alpha=0{,}25$ với 4 epoch tốt nhất (so với $\alpha=0{,}5$ ≈2 epoch và $\alpha=0{,}125$ ≈8 epoch, bị overfit). Paper ước lượng Pre-Train w/o KD cần thêm **68B token** để bằng MiniPLM ($\alpha=0{,}25$, 4 epoch), tức giảm nhu cầu dữ liệu **2,4×**.

## 5. Phân tích / ablation

- **Cỡ teacher (Hình 6, student 200M):** teacher lớn hơn không luôn tốt hơn, tương tự capacity gap (bài 41, 42). Với Vanilla KD, teacher lớn tốn thêm compute nên lợi ích giảm. Với MiniPLM, teacher **500M** tốt nhất cho student 200M; teacher 300M quá yếu để phát hiện phần khó; teacher 4B làm thang giá trị $\log p(x)$ quá nhỏ so với tham chiếu nên Difference Sampling suy biến thành lấy mẫu theo $p_{ref}$ thuần, mất vai trò của teacher. Tác giả để ngỏ việc tối ưu cỡ teacher hay bù lệch thang đo.
- **Cỡ mô hình tham chiếu (Hình 8):** tham chiếu lớn hơn (45M, 104M, 200M) cho kết quả tốt hơn nhưng bão hòa. Tương quan Pearson/Spearman giữa $\log\frac{p}{p_{ref}}$ và $\log\frac{p}{q_\theta}$ (student 500M): 45M: 0,743/0,774, độ chính xác lấy mẫu 80,6%; 104M: 0,826/0,840, 87,4%; 200M: 0,856/0,879, 89,5%.
- **Tỷ lệ lấy mẫu $\alpha$ (Hình 9):** $\alpha$ nhỏ cho độ chính xác zero-shot cao hơn trong thiết lập dữ liệu đủ, nhưng cần corpus gốc lớn hơn; họ chọn $\alpha=0{,}5$ để kiềm chế chi phí offline.
- **Đa dạng ngữ nghĩa (Bảng 4):** corpus gốc 32,25; teacher-generated (SeqKD) 30,16; difference-sampled 36,70. Dù xuất phát từ reverse KL (mode-seeking), corpus lọc lại đa dạng hơn.
- **Kết hợp với Vanilla KD (Bảng 5):** thêm Vanilla KD trên $D'$ tăng thêm ở 500M (44,8 → 44,9) và rõ ở 1,2B (46,6 → 48,1), nhưng **giảm ở 200M** (41,3 → 40,7). Nghĩa là hai cách bổ sung nhau khi student đủ lớn (diễn giải của paper: MiniPLM chuyển tri thức "thô" mức sequence, Vanilla KD chuyển tín hiệu mức token).
- **Proxy model (Bảng 12):** MiniPLM$_{prx}$ dùng 9×10¹⁸ FLOPs offline (so với 2×10²⁰ cho MiniPLM chuẩn) vẫn đạt 40,9 so với 41,3 (MiniPLM) và 39,9 (Vanilla KD), student 200M.

## 6. Giới hạn

**Do tác giả nêu:** cần xác suất của teacher trên văn bản corpus, nên khó áp dụng cho teacher đóng (API); có thể lấy xác suất từng token qua logit bias nhưng tốn rất nhiều lệnh gọi. Hướng tương lai: pre-train mô hình **lớn hơn** teacher (weak-to-strong).

**Do chúng tôi nhận xét:**
- Quy mô nhỏ (≤1,2B student, 50B token, teacher 1,8B). Mọi con số ở 1T/10T là ngoại suy.
- Cải thiện tuyệt đối khiêm tốn (khoảng +1,4 đến +1,7 điểm trung bình so với Pre-Train, chênh lệch giữa các baseline thường dưới 1 điểm); không có nhiều seed hay khoảng tin cậy.
- Chỉ đánh giá zero-shot các tác vụ ngôn ngữ cơ bản và loss DCLM; không có suy luận, code, toán hay chuyển đổi miền.
- Chọn mẫu theo hiệu số loss có thể thiên vị: ưu tiên văn bản mà mô hình lớn "biết" hơn nhưng cũng dễ ưu tiên đoạn dài khó đoán hay nhiễu có cấu trúc lạ; paper chỉ có case study định tính (Phụ lục F).
- Teacher phải cùng loại dữ liệu và có xác suất sequence; không dùng được trực tiếp cho VLM nơi xác suất có điều kiện ảnh.

## 7. Liên hệ với các bài khác

- **02 / 40 (Hinton KD, Gemma 2):** Vanilla KD chính là KD kiểu Gemma 2; MiniPLM là lựa chọn khi không muốn trả chi phí teacher online.
- **15 Sequence-level KD:** SeqKD là baseline; MiniPLM thắng vì giữ phân phối dữ liệu thật và tăng độ khó/đa dạng thay vì dùng văn bản do teacher sinh.
- **28 MiniLLM:** cùng xuất phát từ reverse KL, nhưng MiniLLM tối ưu bằng PPO online; MiniPLM tối ưu bằng chọn mẫu offline. Kết quả: MiniLLM thấp hơn Pre-Train w/o KD dưới ngân sách FLOPs cố định (39,0 vs 39,9 ở 200M).
- **41 / 42 (Capacity gap, Distillation scaling laws):** Hình 6 là một minh họa khác của capacity gap, với cơ chế mới (thang giá trị log-xác suất). Bài 42 tính chi phí teacher rõ ràng; MiniPLM thuộc nhánh "teacher inference một lần, dùng cho nhiều student".
- **45 Sparse logit sampling:** giải bài toán lưu logit từng token; MiniPLM tránh bài toán đó bằng cách chỉ lưu điểm số sequence.
- **44 Qwen3:** dùng KD ở hậu huấn luyện, không phải pre-training.

## 8. Ứng dụng cho dự án (suy luận)

Mức liên quan: **trung bình-thấp** cho recovery sau pruning, **cao về ý tưởng** nếu bạn có kho dữ liệu OCR lớn chưa gán nhãn.

1. **Recovery sau pruning dùng ít dữ liệu.** MiniPLM giải bài toán "chọn dữ liệu nào" khi corpus rất lớn so với ngân sách train. Với recovery 0,3B / 0,1B, ngân sách token nhỏ; chọn top-α trang tài liệu "khó" có thể cho cùng chất lượng với ít bước hơn. Đây là giả thuyết, chưa ai kiểm cho VLM OCR.
2. **Cặp (teacher, tham chiếu) tự nhiên:** teacher = LightOnOCR-4B; tham chiếu = chính mô hình 0,8B (hoặc bản 0,3B đã prune, chưa recover). Hiệu số loss $\log p_{4B}(x)-\log p_{0.8B}(x)$ trên mỗi trang (hoặc mỗi dòng) chỉ ra những mẫu mà mô hình lớn thật sự "biết" hơn: chữ viết tay, bảng, công thức, ngôn ngữ hiếm. Có thể chạy một lần offline.
3. **Điều chỉnh cần thiết:** MiniPLM tính trên sequence văn bản; ở VLM cần tính trên chuỗi token đầu ra có điều kiện ảnh và chuẩn hóa theo độ dài. Chi phí offline: một forward của teacher 4B trên toàn bộ tập, chấp nhận được với 8GB GPU nếu chạy tách rời.
4. **Kết hợp với KD loss:** Bảng 5 cho thấy MiniPLM + Vanilla KD bổ sung nhau ở student lớn hơn nhưng không giúp ở 200M. Student 0,1B có thể giống trường hợp 200M. Đây là lý do nên thử riêng từng cỡ.
5. **Cảnh báo teacher quá lớn:** Hình 6 gợi ý hiệu số $\log p-\log p_{ref}$ mất ý nghĩa khi teacher quá khác tham chiếu về thang đo. Nếu tham chiếu là 0,8B còn teacher là 4B (gấp 5×), cần xem phân phối điểm số trước khi lọc.
6. Không liên quan trực tiếp tới việc cắt MLP neuron hay $D_{lin}/D_{res}$.

## 9. Thí nghiệm gợi ý

1. **Lọc dữ liệu recovery:** từ một tập OCR lớn (vài chục nghìn trang), tính điểm $\log p_{4B}-\log p_{0.8B}$ mỗi trang; train student 0,3B đã prune trên (a) top 30%, (b) ngẫu nhiên 30%, (c) toàn bộ trong cùng số bước. So CER/WER trên dev. Đây là phiên bản thu nhỏ của Bảng 1.
2. **Phân tích điểm số:** xem trang có điểm cao/thấp nhất là loại nào (bảng, chữ viết tay, trang nhiễu). Kiểm tra giả thuyết "điểm âm là nhiễu" ở dữ liệu OCR (case study Phụ lục F).
3. **Thêm KD loss:** trên tập đã lọc, so CE-only với CE + KD từ teacher 4B ở student 0,3B và 0,1B, để thấy lặp lại hiện tượng Bảng 5 (có lợi ở lớn, không ở nhỏ).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Online KD | Teacher chạy lúc train, tốn compute, cần cùng tokenizer |
| Offline KD (SeqKD) | Student học từ văn bản teacher sinh; dễ thiếu khó và đa dạng |
| $\arg\min \mathrm{KL}[q_\theta\Vert p]=\arg\max\mathbb E_{q_\theta}\log\frac{p}{q_\theta}$ | KD như tối đa phần thưởng |
| Best-of-N | Sinh nhiều mẫu, giữ top-K theo reward |
| Eq. 4 | $D'=\text{top-}K\{\log\frac{p(x)}{p_{ref}(x)}\}$ trên $D-D_{ref}$ |
| Difference Sampling | Giữ mẫu mà teacher đoán tốt hơn tham chiếu nhiều nhất |
| Lưu trữ | ~200MB cho 50B token (một số thực mỗi sequence) so với ~30PB nếu lưu logit |
| Kết quả chính | 200M/500M/1,2B: 41,3 / 44,8 / 46,6; 2,2× ít compute hơn Vanilla KD; 2,4× ít dữ liệu hơn |

## 11. Kết luận ngắn

MiniPLM biến KD pre-training thành bài toán chọn dữ liệu: dùng chênh lệch giữa teacher lớn và mô hình nhỏ để làm dày phần khó, bỏ phần dễ và nhiễu, nên chi phí teacher chỉ trả một lần và student bất kỳ (khác tokenizer, khác kiến trúc) đều hưởng lợi. Bằng chứng nằm ở quy mô nhỏ nhưng nhất quán. Với dự án, đáng thử như một cách chọn dữ liệu cho recovery, chưa phải thay thế KD loss.

**Link gốc:** https://arxiv.org/abs/2410.17215
