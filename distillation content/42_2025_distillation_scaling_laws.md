# Distillation Scaling Laws (2025)

**Tác giả:** Dan Busbridge, Amitis Shidani, Floris Weers, Jason Ramapuram, Etai Littwin, Russ Webb (Apple; Shidani thuộc University of Oxford, làm trong kỳ thực tập tại Apple).  
**Venue / arXiv:** arXiv:2502.08606 (v2, 25 Jul 2025); paper ghi "Proceedings of the 42nd International Conference on Machine Learning (ICML), Vancouver, PMLR 267, 2025".  
**Nguồn:** https://arxiv.org/abs/2502.08606  
**Chủ đề:** Một "scaling law" cho student được distill: dự đoán cross-entropy của student từ cỡ student, số token distill và **cross-entropy của teacher**; từ đó suy ra (i) khi nào teacher lớn hơn làm hại, (ii) khi nào distillation đáng hơn train supervised, (iii) cách chia compute giữa teacher và student.  
**Vị trí trong lộ trình:** Đọc trước: 02 (Hinton KD), 08 (Teacher assistant), 11 (Patient and consistent teacher), 40 (Gemma 2), 41 (Capacity gap law). Bài này **tổng quát hóa và định lượng** cả hai: luật tuyến tính của 41 hóa ra là trường hợp đặc biệt khi teacher là compute-optimal. Tiếp theo: 43 (MiniPLM) và 45 (Sparse logit sampling) bàn chi phí teacher; 44 (Qwen3) cho một công thức thực tế.

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải (paper ~41k từ gồm cả phụ lục; chúng tôi đọc thân bài và các phụ lục C, D, E, F, G quan trọng). Không phải bản dịch. Các con số lấy từ paper; phần "Ứng dụng cho dự án" là suy luận của chúng tôi. Lưu ý quan trọng: toàn bộ thí nghiệm là **language modeling trên C4 tiếng Anh, transformer 143M–12,6B, loss = cross-entropy**, chưa kiểm tra trên VLM/OCR.

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

**Scaling law (supervised)** (gloss): nhiều thí nghiệm cho thấy cross-entropy $L$ của một LM giảm theo hàm lũy thừa của số tham số $N$ và số token $D$:

$$
L(N,D)=E+\left(\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}\right)^{\gamma}\qquad(\text{Eq. 1})
$$

$E$ là "irreducible error" (nhiễu bản chất của dữ liệu), phần trong ngoặc là khả năng của mô hình mô phỏng dữ liệu. Chinchilla (Hoffmann et al. 2022) dùng nó để chọn $(N,D)$ tối ưu cho một ngân sách tính toán (xấp xỉ 20 token mỗi tham số).

**Over-training** (gloss): vì chi phí suy luận lớn, người ta train mô hình nhỏ với nhiều token hơn mức compute-optimal rất nhiều. Hiệu suất khi đó có lợi suất giảm dần.

**Distillation pretraining** (gloss): dùng teacher cung cấp phân phối đích cho student trong lúc pre-train (như Gemma 2, bài 40). Có công thức công nghiệp nhưng thiếu hiểu biết: kích thước teacher nên là bao nhiêu? Có lúc nào supervised tốt hơn? Teacher đắt thì tính vào đâu? Paper còn nhắc một báo cáo trái chiều (Liu et al. 2024) cho rằng distillation kém supervised pretraining, và hiện tượng capacity gap.

**Mục tiêu paper:** dùng thí nghiệm có kiểm soát để dự đoán cross-entropy của student distill, rồi dùng luật đó để trả lời các câu hỏi trên.

## 2. Ý tưởng chính

Student distill có cross-entropy trên dữ liệu thật $L_S$. Điều quyết định ảnh hưởng của teacher không phải là kích thước hay số token riêng lẻ, mà là **cross-entropy $L_T$ của teacher** (kết quả của cả hai). Student giỏi lên khi $L_T$ giảm, **cho tới một điểm**, sau đó teacher quá mạnh khiến student khó mô phỏng và $L_S$ tăng trở lại: đó là capacity gap, được mô hình hóa bằng một "broken power law" theo $L_T$. Với luật này, ba kết luận chính: (1) distillation không bao giờ thắng supervised nếu cả hai có đủ nhiều compute/token; (2) distillation thắng khi ngân sách của student không vượt ngưỡng phụ thuộc cỡ student **và** teacher đã có sẵn hoặc được dùng lại; (3) nếu phải train teacher chỉ để distill một student, supervised nói chung tốt hơn.

## 3. Phương pháp chi tiết

### 3.1 Hàm loss distillation

Với logit teacher $z_T$ và student $z_S$, nhiệt độ $\tau$, hệ số trộn $\lambda$:

$$
\mathcal L_{KD}(z_T,z_S)=-\tau^2\sum_{a=1}^{V}\sigma_a\!\left(\tfrac{z_T}{\tau}\right)\log\sigma_a\!\left(\tfrac{z_S}{\tau}\right),
\qquad
\mathcal L_S=(1-\lambda)\mathcal L_{NTP}+\lambda\mathcal L_{KD}+\lambda_Z\mathcal L_Z
$$

$\mathcal L_{NTP}$ là next-token prediction thường, $\mathcal L_Z=\|\log Z\|^2$ là Z-loss ổn định. Paper dùng **pure distillation** ($\lambda=1$) và $\tau=1$ trong toàn bộ thí nghiệm chính, và kiểm tra rằng lựa chọn này gần tối ưu (Mục 5).

### 3.2 Thiết kế thí nghiệm

- Mô hình transformer (nền Gunter et al. 2024), µP đơn giản hóa (cho phép dùng chung learning rate 0,01), seq length 4096, dữ liệu C4 English; teacher và student train trên các split khác nhau, ít lặp dữ liệu.
- Kích thước 143M đến 12,6B, token từ vài tỷ đến 512B.
- Sáu teacher "Chinchilla-optimal" ($M_T=D_T/N_T\approx 20$) cỡ $N_T\in\{546M, 975M, 1{,}82B, 2{,}72B, 4{,}82B, 7{,}75B\}$ (cùng 198M ở đầu dải), distill vào student theo các IsoFLOP profile; thêm IsoFLOP teacher (đổi $N_T,D_T$ với compute cố định) và lưới "fixed-M teacher × fixed-M student" gồm 10 teacher × student 5 cỡ × ít nhất 4 tỷ lệ $M_S$.
- Tổng cộng **697 run** để fit luật distillation (551 run khi chỉ lấy $L_S\ge 2{,}3$) và 73 run cho luật supervised.

### 3.3 Luật distillation (Eq. 8)

Gọi $\tilde L_S=L(N_S,D_S)$ là cross-entropy mà student đó sẽ đạt nếu train supervised (từ Eq. 1). Luật:

$$
\boxed{L_S(N_S,D_S,L_T)=L_T+\frac{1}{L_T^{\,c_0}}\left(1+\left(\frac{L_T}{\tilde L_S\,d_1}\right)^{1/f_1}\right)^{-c_1 f_1}\left(\frac{A'}{N_S^{\alpha'}}+\frac{B'}{D_S^{\beta'}}\right)^{\gamma'}}
$$

Cách đọc từng khối (diễn giải của chúng tôi dựa trên ba tính chất mà paper yêu cầu):
- **Sàn $L_T$:** student không thể tốt hơn nhiều so với "mức teacher" cộng một phần dư; một teacher ngẫu nhiên ($L_T\to\infty$) cho student ngẫu nhiên ($L_S\to L_T$).
- **Phần trong ngoặc cuối** $\left(A'/N_S^{\alpha'}+B'/D_S^{\beta'}\right)^{\gamma'}$: khả năng mô phỏng teacher của student, giảm khi tăng $N_S$ hoặc $D_S$ (khi $N_S,D_S\to\infty$ thì $L_S\to L_T$, tức student vô hạn mô hình hóa được mọi teacher).
- **Nhân tử $L_T^{-c_0}\,(1+\dots)^{-c_1f_1}$:** là **broken power law** theo $L_T$ với điểm gãy khi $L_T/\tilde L_S=d_1$. Hai vùng: teacher yếu hơn ngưỡng (student là người học mạnh hơn teacher) và teacher mạnh hơn ngưỡng (student là người học yếu hơn). Với $c_0>0$, khi $L_T$ giảm thì số hạng phạt $L_T^{-c_0}(\cdots)$ **tăng**, gây ra capacity gap: tổng $L_S$ có cực tiểu theo $L_T$ ở một giá trị teacher tối ưu $L_T^\*(N_S,D_S)$.

Hệ số fit (Table 6, kèm khoảng tin cậy 90% bằng bootstrap):

| | Supervised | Distillation |
|---|---|---|
| $A$ ($A'$) | 3355 | 2243 |
| $B$ ($B'$) | 18186 | 24181 |
| $E$ | 1,220 | – |
| $\alpha$ ($\alpha'$) | 0,408 | 0,321 |
| $\beta$ ($\beta'$) | 0,431 | 0,637 |
| $\gamma$ ($\gamma'$) | 0,452 | 0,764 |
| $c_0$ | – | 2,549 |
| $c_1$ | – | 522,6 |
| $f_1$ | – | 0,090 |
| $d_1$ | – | 1,315 |

Sai số dự đoán khoảng ≲ 1% tương đối, kể cả khi ngoại suy từ mô hình yếu (cross-entropy cao) sang mô hình mạnh (Hình 5). Ghi chú của paper: các hệ số **đặc thù** cho kiến trúc và dữ liệu của họ, không phổ quát; dạng hàm không suy ra từ nguyên lý đầu tiên.

### 3.4 Hạch toán compute (Eq. 9)

$$
\text{FLOPs}\approx \underbrace{3F(N_S)D_S}_{\text{train student}}+F(N_T)\Big(\underbrace{\delta^{Lgt}_T D_S}_{\text{teacher logits}}+\underbrace{\delta^{Pre}_T\,3D_T}_{\text{teacher pretrain}}\Big)
$$

$F(N)\approx 2N$ FLOPs/token cho một lượt forward (paper dùng công thức chính xác hơn có số hạng attention). Bốn kịch bản (Table 2):

| Kịch bản | $\delta^{Lgt}$ | $\delta^{Pre}$ | Nghĩa |
|---|---|---|---|
| Best case | 0 | 0 | Teacher miễn phí hoàn toàn (khấu hao hết) |
| Teacher inference | 1 | 0 | Teacher đã có, chỉ trả tiền forward logit |
| Teacher pretraining | 0 | 1 | Phải train teacher nhưng lưu logit dùng lại cho nhiều student |
| Pretrain + inference | 1 | 1 | Train teacher rồi distill đúng một student (xấu nhất) |

### 3.5 Pseudo-code: chọn teacher bằng luật

```
given: student size N_S, student tokens D_S, candidate teachers {(N_T, D_T)}
L_tilde = L(N_S, D_S)                 # supervised scaling law, Eq.1
for each teacher t:
    L_T = L(N_T, D_T)                 # teacher's own pretrain cross-entropy
    L_S[t] = distill_law(N_S, D_S, L_T, L_tilde)     # Eq.8
    cost[t] = flops(N_S, D_S, N_T, D_T; scenario)    # Eq.9
if distill_law(optimal t) >= L_tilde : use supervised
else choose t minimising L_S subject to cost <= budget   # SLSQP in paper
```

## 4. Thí nghiệm và kết quả chính

### 4.1 Khi nào teacher lớn hơn làm hại

- **Hình 3 và 4:** với student cố định, vẽ $L_S$ theo $N_T$ (teacher compute-optimal): $L_S$ **giảm tới cực trị rồi tăng** khi teacher to dần. Đó là capacity gap. Vẽ theo $L_T$ thay vì $N_T$ thì các teacher IsoFLOP khác nhau (đổi $N_T,D_T$) rơi vào cùng một đường, chứng tỏ **chỉ $L_T$ quan trọng, không phải $N_T$ hay $D_T$ riêng lẻ**.
- Paper đo KL giữa teacher và student: KL **luôn tăng khi teacher mạnh hơn** (Phụ lục E.3), tức student càng khó mô phỏng. Phụ lục C chứng minh trong kernel regression và một bài MLP tổng hợp rằng khoảng cách giữa không gian giả thuyết của teacher và student gây suy giảm có thể chứng minh được.
- **Capacity gap là gap về learning capacity (không gian giả thuyết và khả năng tối ưu), không chỉ là tỷ lệ kích thước.** Luật tuyến tính của bài 41 chỉ là trường hợp đặc biệt khi các teacher là compute-optimal supervised (Hình 36a).
- Teacher tối ưu $L_T^\*(N_S,D_S)$ **giảm theo dạng lũy thừa theo $N_S$** (teacher cần mạnh hơn khi student lớn hơn), có điểm uốn khi $L_S$ chạm $L_T^\*$ (Mục 5.2, Hình 7). Teacher quá yếu cũng hại (student bị kẹt gần $L_T$).
- Calibration: teacher được kiểm là calibrated tốt (Phụ lục E.8), nên theo paper, miscalibration của teacher "không thể là nguồn duy nhất" của capacity gap.

### 4.2 Khi nào distillation thắng supervised

Bốn phát biểu của paper (tóm lược Mục 1, 5.1, 5.3):

1. **Distillation không thể cho cross-entropy thấp hơn supervised nếu cả hai có đủ dữ liệu/compute.** Lý do (Phụ lục E.6, D.4.2): distillation không đổi không gian giả thuyết của student (gắn với $N_S$); với teacher tối ưu và dữ liệu vô hạn, $L_S$ trùng cận dưới supervised $L(N_S,D=\infty)$ (kiểm số học cho $N,N_S\in[100M,100B]$). Với bất kỳ teacher khác, $L_S$ cao hơn.
2. **Với ngân sách student khiêm tốn, distillation thắng** (vùng xanh trong Hình 6). Ngưỡng compute/token mà supervised bắt đầu thắng **tăng theo kích thước student** (paper: student nhỏ nhạy với việc bị supervised vượt sớm hơn; student lớn được lợi từ distillation lâu hơn).
3. **Chi phí teacher quyết định phần còn lại:** khi teacher đã có (best case hoặc chỉ trả inference), distillation thường rẻ hơn supervised trong vùng loss "vừa phải" (Hình 33: compute tỷ lệ < 1 khi $L_S$ hơi cao hơn cận dưới), nhưng để tạo student *mạnh nhất có thể* thì supervised lại hiệu quả hơn (tỷ lệ > 1 khi $L_S$ tiến sát $L(N_S,\infty)$). Paper gợi ý (chưa thử): dùng distillation ở giai đoạn đầu rồi chuyển sang supervised.
4. **Nếu phải train teacher chỉ cho một student** (pretrain + inference, hoặc pretraining tính vào chi phí) **supervised luôn hiệu quả hơn.** Nếu teacher được dùng cho nhiều student (khấu hao), hoặc dùng làm server model, distillation có thể rẻ hơn. (Một đoạn trong Phụ lục D.4.8 nói student 500M đạt loss 2,4 ở ~3/4 compute và student 10B đạt loss 2,2 ở ~1/2 compute của supervised "trong kịch bản pretrain + inference", đoạn này có vẻ mâu thuẫn với câu ngay trước nó; chúng tôi không giải được, nên chỉ xem như ghi chú rằng lợi ích lớn hơn với student lớn hơn.)

**Phân bổ compute tối ưu khi tính cả teacher (Bảng 3 của paper):**

| Cỡ student | Compute | Chia ngân sách |
|---|---|---|
| nhỏ (≲ 3B) | nhỏ (≲ $10^{21}$) | chủ yếu train teacher |
| nhỏ (≲ 3B) | lớn (≳ $10^{25}$) | chia đều train student và inference teacher, train teacher ít hơn nhiều |
| lớn (≳ 10B) | nhỏ (≲ $10^{21}$) | chủ yếu train student chuẩn |
| lớn (≳ 10B) | lớn (≳ $10^{25}$) | chia đều train student, inference teacher, train teacher |

Số token student tối ưu và token teacher tối ưu đều tăng theo luật lũy thừa với compute (student nhanh hơn). Cỡ teacher tối ưu tăng đến khi lớn hơn student một chút rồi **đi ngang**, vì inference teacher lớn đắt và với nhiều token student thì nên over-train teacher nhỏ hơn.

## 5. Phân tích / ablation

Tất cả ở Phụ lục G (cỡ 198M–1,82B student, 546M–7,75B teacher):

| Yếu tố | Kết quả |
|---|---|
| Hệ số trộn $\lambda$ | $L_S$ gần như phẳng với $\lambda>0{,}5$; $\lambda$ nhỏ chỉ tốt khi teacher yếu. Pure distillation ($\lambda=1$) là mặc định đáng tin |
| Nhiệt độ $\tau$ | Tối ưu tại $\tau=1$ ở mọi cỡ; $\tau>3$ hại, $\tau<1$ cũng giảm hiệu quả. Paper giả thuyết lệch với văn liệu trước vì họ không lặp token |
| Learning rate (µP) | Tối ưu $\eta=0{,}01$, giống supervised; thay đổi 10× chỉ đổi loss < 0,1 nat |
| Cắt logit Top-k / Top-p | Hại: $k=128$ làm loss tăng **0,11 nat** so với full; Top-p $p=0{,}9$ tăng **0,13 nat**. Bù bằng trộn nhãn thật: **$k=128$ và $\lambda=0{,}7$** giữ trong **0,01 nat** |
| Forward vs reverse KL | Forward KL tốt hơn trung bình **0,28 nat** (teacher 1,82B → student 546M) và rẻ hơn |
| Kích thước từ vựng và lưu logit | Với vocab 32.168, lưu đủ logit float32 cho ~2T token C4 khoảng 260 PB, cho thấy cần Top-k hoặc tính online |
| Weak-to-strong (E.7) | Student lớn hơn teacher có thể vượt teacher, nhưng chỉ trong chế độ dữ liệu distill hữu hạn; nhiều token hơn thì $L_S$ tăng lại về $L_T$ |
| Downstream (E.1) | Quan hệ cross-entropy ↔ điểm benchmark giống nhau cho supervised và distilled; MMLU/GSM8K gần ngẫu nhiên (do train C4), nên bài **không** đánh giá khả năng suy luận |

**Mâu thuẫn bề ngoài với "patient teacher" (bài 11):** Beyer et al. thấy distill lâu là tốt; ở đây distill lâu cũng cải thiện student, nhưng supervised vẫn vượt ở ngân sách đủ lớn. Paper giải thích (D.1) rằng thiết lập của Beyer (thị giác, lặp dữ liệu nhiều, student không thấy phân phối train của teacher, không có supervised baseline tương ứng) khác với của họ.

## 6. Giới hạn

**Do tác giả nêu (Phụ lục A):**
- Chỉ language modeling, chỉ C4 English (token lớn bị lặp dữ liệu, chưa chắc phép "lặp ≤ 4 lần là vô hại" đúng cho distillation); không thể phân tích tốt reasoning.
- Chỉ distillation kiểu Hinton (teacher đưa **toàn phân phối**). Chưa nghiên cứu sequence-level KD / synthetic data (như DeepSeek-R1 distill, bài 38), loss trung gian hay các objective khác; tác giả "kỳ vọng nhưng không chắc" kết luận vẫn đúng.
- Chỉ transformer, teacher và student cùng phân phối dữ liệu (không có lệch miền).
- Hệ số đặc thù cho thiết lập; luật không suy ra từ lý thuyết.

**Do chúng tôi nhận xét:**
- Student dài hơn 512B token và cỡ lớn hơn 12,6B là ngoại suy.
- "Compute" chỉ đếm FLOPs; không đếm băng thông bộ nhớ, thời gian lưu/đọc logit hay chi phí kỹ thuật.
- Teacher trong thí nghiệm luôn được train trên cùng loại dữ liệu và cùng loại kiến trúc; kết quả có thể thay đổi nếu teacher là mô hình đa phương thức, đã qua SFT hoặc RL.
- Biến phụ thuộc là loss trên dữ liệu, không phải chất lượng sinh theo task (OCR chẳng hạn).

## 7. Liên hệ với các bài khác

- **02 Hinton KD:** Eq. 6 là Hinton KD áp dụng cho token; kết luận $\tau=1$ khác với khuyến nghị nhiệt độ cao trong bài gốc cho phân loại.
- **08 Teacher assistant, 41 Capacity gap law:** cùng hiện tượng; 41 cho luật tuyến tính theo kích thước, bài này nói bản chất là gap về learning capacity và $L_T$ mới là biến đúng. 41 là trường hợp đặc biệt khi teacher là compute-optimal.
- **11 Patient and consistent teacher:** xem mục 5.
- **40 Gemma 2:** bài này cho khung lý thuyết cho việc Gemma 2 distill 27B → 9B và chi phí teacher.
- **43 MiniPLM, 45 Sparse logit sampling:** bài này cho thấy Top-k thô hại (0,11 nat) và cần λ>0 bù; 45 đề xuất cách lấy mẫu logit tốt hơn.
- **44 Qwen3:** compute so sánh distillation với RL ở post-training, bổ sung phần pretraining của bài này.
- **28–31 (MiniLLM, GKD, DistiLLM, Rethinking KL):** forward vs reverse KL; ở đây forward KL thắng trong thiết lập pretraining.
- **38 DeepSeek-R1 distillation:** thuộc loại sequence-level KD mà bài này nói là ngoài phạm vi.

## 8. Ứng dụng cho dự án (suy luận)

1. **Teacher 4B có phải là lựa chọn đúng cho student 0,3B và 0,1B?** Luật nói rằng quan trọng là $L_T$ (cross-entropy của teacher trên dữ liệu OCR), và tồn tại $L_T^\*$ phụ thuộc cỡ student và ngân sách. Teacher quá mạnh so với student 0,1B có thể không mang lại lợi ích. Bài 41 cảnh báo cùng điều. Do đó cần đo, không giả định.
2. **Ngân sách recovery nhỏ là điều kiện thuận lợi cho distillation.** Theo bài này, distillation thắng supervised chủ yếu khi ngân sách student ≤ ngưỡng, và dự án recovery thường dùng ít token. Kết luận này chỉ chắc cho LM pretraining; dự án là fine-tune/recovery sau pruning (student không khởi tạo ngẫu nhiên), nên chỉ dùng như hướng dẫn định tính.
3. **Teacher đã có sẵn nên chi phí teacher chỉ là inference.** Bằng arithmetic thô của chúng tôi với $F(N)\approx 2N$: mỗi token student 0,3B tốn khoảng $3\cdot 2\cdot0{,}3=1{,}8$ GFLOP, còn logit teacher 4B tốn khoảng $2\cdot 4=8$ GFLOP, tức teacher chiếm phần lớn compute nếu tính online. Vì vậy nên **cache logit offline** (một lần) rồi dùng cho nhiều student 0,3B / 0,1B / các tỷ lệ cắt khác nhau, đúng kịch bản "teacher pretraining, nhiều student" của paper.
4. **Cache logit:** Top-k thô làm loss tăng 0,11 nat ở $k=128$ nhưng trộn 30% nhãn thật ($\lambda=0{,}7$) bù lại. Với OCR, phân phối token thường rất nhọn (ít mơ hồ hơn văn xuôi), nên Top-k có thể ít hại hơn nhưng đây là **giả thuyết** chưa được kiểm.
5. **Kiểm tra $\tau=1$, forward KL, $\lambda\approx 0{,}7$–1:** đây là mặc định hợp lý khi bắt đầu.
6. **Không áp dụng trực tiếp:** các hệ số trong Table 6 (fit trên C4); dùng để lấy cảm giác chứ không để dự đoán số. Luật cũng không phân biệt pruning khác nhau, nên không giải thích được việc cắt MLP neuron recovery kém hơn cắt layer; liên hệ với $D_{lin}/D_{res}$ chỉ ở mức: sau khi pruning, student khởi đầu ở điểm tốt hơn khởi tạo ngẫu nhiên, nên "khả năng mô phỏng" ban đầu cao hơn và có thể cần ít token hơn so với luật.

## 9. Thí nghiệm gợi ý

1. **Quét teacher theo $L_T$ trên dữ liệu OCR:** dùng 3–4 teacher có cross-entropy khác nhau trên tập dev (ví dụ: bản 0,8B gốc, 4B, một checkpoint 4B sớm hơn; early-stopped teacher cũng cho $L_T$ khác). Distill cùng student 0,1B và vẽ $L_S$ theo $L_T$ để xem có cực tiểu (hình chữ U) như Hình 3/4 không. Kiểm tra giả thuyết "chỉ $L_T$ quan trọng".
2. **Top-k cache:** so $k\in\{16,64,256,\text{full}\}$ kết hợp $\lambda\in\{0{,}7,1\}$ cho recovery 0,3B; đo CER/WER và loss, so với số 0,11 nat / 0,01 nat của paper.
3. **Distill vs CE-only theo số bước:** vẽ đường hiệu năng theo số bước (1k, 3k, 10k, 30k) cho cả hai loss để tìm điểm giao nếu có, kiểm tra tuyên bố "supervised luôn thắng khi đủ lâu" ở quy mô dự án.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| $L(N,D)=E+(A/N^\alpha+B/D^\beta)^\gamma$ | Scaling law supervised (Eq. 1) |
| Eq. 8 | $L_S=L_T+L_T^{-c_0}(1+(L_T/\tilde L_S d_1)^{1/f_1})^{-c_1f_1}(A'/N_S^{\alpha'}+B'/D_S^{\beta'})^{\gamma'}$ |
| Capacity gap | Teacher quá mạnh so với learning capacity của student làm $L_S$ tăng |
| $L_T^\*$ | Cross-entropy teacher tối ưu cho $(N_S,D_S)$, giảm theo luật lũy thừa với $N_S$ |
| $\delta^{Lgt},\delta^{Pre}$ | Có tính chi phí logit teacher / train teacher vào compute hay không |
| Kết luận 1 | Đủ dữ liệu thì supervised ≥ distillation |
| Kết luận 2 | Distillation thắng khi ngân sách student ≤ ngưỡng theo $N_S$ và teacher có sẵn / dùng lại |
| Kết luận 3 | Phải train teacher cho một student duy nhất → supervised |
| Cấu hình mặc định | $\lambda=1$, $\tau=1$, forward KL, $\eta=0{,}01$ (µP) |
| Top-k | $k=128$: +0,11 nat; kèm $\lambda=0{,}7$: trong 0,01 nat |

## 11. Kết luận ngắn

Bài này biến chuyện "kinh nghiệm" về distillation thành một luật có thể kiểm chứng: student phụ thuộc vào teacher **qua $L_T$**, capacity gap là một chỗ gãy trong luật, và distillation chỉ có lợi khi ngân sách student vừa phải và chi phí teacher được khấu hao. Với dự án (teacher đã có, ngân sách recovery nhỏ, nhiều student cùng một teacher), các điều kiện thuận lợi cho distillation, nhưng chưa có gì đảm bảo teacher 4B là tốt nhất cho student 0,1B; cần thí nghiệm quét $L_T$ trên dữ liệu OCR.

**Link gốc:** https://arxiv.org/abs/2502.08606
