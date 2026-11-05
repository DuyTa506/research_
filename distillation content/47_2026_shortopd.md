# ShortOPD: Recovering Pruned LLMs with Short-to-Long On-Policy Distillation (2026)

**Tác giả:** Qingyu Zhang, Qianhao Yuan, Hongyu Lin, Yaojie Lu, Xianpei Han, Le Sun, Ming Xu, Jiarui Li (ByteDance; Viện Phần mềm, Học viện Khoa học Trung Quốc; UCAS).
**Venue / arXiv:** arXiv:2607.13124 (v2, 14/08/2026). Bản text ghi "Work in Progress"; chưa nêu venue.
**Nguồn:** https://arxiv.org/abs/2607.13124
**Chủ đề:** Phục hồi LLM sau structured pruning bằng **on-policy distillation (OPD)** với chính mô hình trước khi cắt làm teacher đóng băng; thêm bộ điều khiển độ dài rollout "ngắn → dài" để tránh phí token vào các đuôi lặp.
**Vị trí trong lộ trình:** Đọc trước: 15 (SeqKD), 28 (MiniLLM), 29 (GKD, on-policy KD), 31 (Rethinking KL), 46 (Distil-Whisper). Bài này bổ sung: đánh giá phục hồi sau cắt bằng **sinh tự do** (không chỉ trắc nghiệm) và so sánh trực tiếp OPD với SFT/SeqKD/KD/RLVR. Là bài cuối của lộ trình.

> Đây là ghi chú học tập bằng tiếng Việt, không phải bản dịch. Số liệu lấy từ paper (ghi rõ là "work in progress"); phần "Ứng dụng cho dự án" là suy luận của chúng ta về LightOnOCR, không phải kết luận của tác giả.

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

**Structured pruning** (cắt có cấu trúc) loại bỏ nguyên khối tham số (lớp, đầu attention, neuron MLP) nên mô hình vẫn dày và chạy trên kernel thông thường. Tuy nhiên phần lớn nghiên cứu đánh giá bằng bài **trắc nghiệm** (MMLU, HellaSwag): mô hình chỉ cần xếp hạng đáp án. Trên **sinh văn bản tự do** (toán, code, hội thoại) cùng checkpoint ấy sụp đổ. Khoảng trống này gọi là *recognition-generation gap*.

Một vài thuật ngữ:
- **pass@k:** xác suất ít nhất 1 trong k mẫu sinh ra là đúng (toán/code). pass@1 = greedy/mẫu đơn.
- **On-policy:** dữ liệu huấn luyện do chính student sinh (nó tự "chạy" rồi bị chấm), khác **off-policy** là dữ liệu cố định từ nguồn khác (xem 29 GKD).
- **Teacher-forced (KD thường):** student được cho tiền tố *đúng* của chuỗi cố định rồi học khớp phân phối teacher ở mỗi vị trí. Có thể gây *exposure bias*: lúc suy luận student gặp trạng thái do chính nó tạo ra mà lúc train không thấy.
- **RLVR:** RL với phần thưởng kiểm chứng được (đúng/sai cuối cùng), tín hiệu thưa.
- **Block Influence (BI)** của ShortGPT: lớp càng ít làm đổi hidden state (cosine giữa đầu vào và đầu ra lớp càng cao) thì càng dễ bỏ.

Hai quan sát dẫn đường của paper:
1. Năng lực bị **"hạ cấp" chứ không bị xoá**: sau khi bỏ 4/36 lớp của Qwen3-4B-Instruct, greedy gần như sụp nhưng pass@k tăng đều với k; GSM8K đạt 91.2% ở k=64, cao hơn greedy của teacher (88.1%).
2. Lỗi trong vùng có thể phục hồi chủ yếu là **lặp ở đuôi (suffix repetition)**: câu trả lời bắt đầu hợp lý rồi rơi vào vòng lặp.

## 2. Ý tưởng chính

Vì năng lực vẫn nằm trong phân phối lấy mẫu của mô hình đã cắt, cách phục hồi tốt là cho student **tự sinh** rollout, để teacher (chính mô hình gốc, đóng băng) chấm lại từng token và student khớp phân phối token kế tiếp của teacher tại mọi vị trí (OPD: on-policy + dày đặc + không cần nhãn hay verifier). Nhược điểm: giai đoạn đầu student lặp rất nhiều, rollout dài tốn tiền sinh mà lại ít tín hiệu (các token lặp có JSD với teacher chỉ 0.0014 so với 0.051 ở token thường, thấp ~35 lần). ShortOPD thêm một bộ điều khiển phát hiện đuôi lặp, rút ngắn ngân sách độ dài rollout $H$ khi lặp cao và tăng lại khi sạch, nên đạt chất lượng của rollout dài 8192 token với 1/4 thời gian.

## 3. Phương pháp chi tiết

### 3.1 Cắt và OPD cơ bản

BI của lớp $i$ (trên tập hiệu chuẩn giữ riêng):

$$
\mathrm{BI}_i=1-\mathbb{E}_{X,t}\left[\frac{X_{i,t}^\top X_{i+1,t}}{\|X_{i,t}\|_2\|X_{i+1,t}\|_2}\right]
$$

Bỏ các lớp BI thấp nhất tới khi mất ~25% tham số (9/36 lớp) → student $\pi_\theta$; mô hình gốc là teacher $\pi_T$ đóng băng. Mất mát OPD:

$$
\boxed{\mathcal{L}_{OPD}(\theta)=\mathbb{E}_x\mathbb{E}_{y\sim\pi_\theta(\cdot|x)}\Big[\frac1{|y|}\sum_{t=1}^{|y|}D_\alpha\big(\pi_T(\cdot|x,y_{<t}),\pi_\theta(\cdot|x,y_{<t})\big)\Big]}
$$

$D_\alpha$ là Jensen–Shannon tổng quát, với $m=\alpha p+(1-\alpha)q$:

$$
D_\alpha(p,q)=\alpha\sum_i p_i\log\frac{p_i}{m_i}+(1-\alpha)\sum_i q_i\log\frac{q_i}{m_i}
$$

Dùng $\alpha=0.5$ ("thoả hiệp có chặn" giữa KL thuận mode-covering và KL nghịch mode-seeking). Triển khai: **top-100 logit của teacher + 1 bin gộp phần đuôi**, và trọng số importance **bị cắt ở 2.0** so với chính sách sinh rollout. Không có policy-gradient, không phần thưởng.

### 3.2 Gradient gần điểm khớp (Phụ lục C)

Với logit student $z$, $q=\mathrm{softmax}(z)$:

$$
\frac{\partial D_\alpha}{\partial z_j}=(1-\alpha)\,q_j\Big[\log\frac{q_j}{m_j}-\sum_i q_i\log\frac{q_i}{m_i}\Big]
$$

Khi $q=p$ thì $m=p$ và gradient bằng 0. Viết $q=p+\delta$: gradient $\approx\alpha(1-\alpha)\delta_j$, tức ở $\alpha=0.5$ chỉ bằng **1/4** gradient KL thuận ($\delta_j$). Hệ quả cho vòng lặp: khi teacher *cũng* tin vòng lặp (NLL teacher $3\times10^{-5}$ so với 0.68 ở token thường) thì $p\approx q$ và gradient gần 0, tín hiệu hầu như bằng không dù tốn token. Lưu ý: vì ánh xạ về "top-100 + bin đuôi", mất mát chỉ nhìn thấy sự khác nhau ở độ phân giải đó (khác biệt trong bin đuôi vô hình).

### 3.3 Bộ điều khiển short-to-long

Mỗi bước, trên $B$ rollout:
- $a_i=1$ nếu **bộ dò vòng lặp ở cuối** (Phụ lục B) kết luận rollout $i$ rơi vào vòng lặp nghiêm trọng.
- $b_i=1$ nếu rollout lấp đầy ngân sách $H_s$ mà không lặp ("cắt cụt sạch").
- $\ell_i$: độ dài hiệu dụng (phần tiền tố trước vòng lặp, tinh chỉnh bằng loss OPD và NLL teacher thấp); $\ell_i=|y_i|$ nếu không lặp.

$$
r_s=\tfrac1B\sum a_i,\quad q_s=\tfrac1B\sum b_i,\quad L_s=\tfrac1B\sum\ell_i,\qquad (\bar r_s,\bar q_s,\bar L_s)=\mathrm{EMA}_\beta(r_s,q_s,L_s)
$$

$$
\boxed{H^*_s=\begin{cases}\min(H_s,\lambda\bar L_s)&\bar r_s>\rho_{high}\ (\text{rút ngắn})\\ \gamma_\uparrow H_s&\bar r_s<\rho_{low}\ \wedge\ \bar q_s>\tau\ (\text{tăng})\\ H_s&\text{ngược lại}\end{cases}}\qquad H_{s+1}=\mathrm{clip}_{[H_{min},H_{max}]}\big(\lfloor\beta_H H_s+(1-\beta_H)H^*_s\rceil_{16}\big)
$$

Thông số (Bảng 10): $(\rho_{low},\rho_{high},\tau)=(0.20,0.45,0.10)$; EMA 0.7 cho cả thống kê và ngân sách; $\lambda=1.15$, $\gamma_\uparrow=1.25$; làm tròn bội 16; ngân sách khởi tạo 2048, biên $[1024,2048]$. Dò vòng lặp: cửa sổ cuối $W=512$, chu kỳ $p\in[1,10]$, khớp ≥ 90% ở 32 token cuối, ≥ 3 chu kỳ, đuôi tối thiểu 64, "nghiêm trọng" nếu ≥ 128 token hoặc ≥ 30% hậu tố. **Quan trọng:** loss thấp *không bao giờ* tự mở cổng rút ngắn (chỉ dò lặp cấu trúc mới làm được), vì loss thấp nhất đúng lúc lặp. Mất mát OPD vẫn tính trên **mọi token** đã sinh; bộ điều khiển chỉ đổi ngân sách sinh, không che loss, và không tốn thêm forward.

### 3.4 Pseudo-code

```
H = H_max (=2048); EMAs khởi tạo
for step s in 1..S:
    y_i ~ student(.|x_i), tối đa H token             # on-policy rollout
    a_i = terminal_loop(y_i);  b_i = (len(y_i)==H and a_i==0)
    chạy teacher (đóng băng) trên cùng y_i -> top-100 + bin đuôi
    cập nhật student với L_OPD trên toàn bộ token (JSD alpha=0.5, IS clip 2.0)
    tính r_s, q_s, L_s; cập nhật EMA
    if r_bar > rho_high:                H* = min(H, lam * L_bar)
    elif r_bar < rho_low and q_bar>tau: H* = gamma_up * H
    else:                               H* = H
    H = clip(round16(beta_H*H + (1-beta_H)*H*), H_min, H_max)
```

### 3.5 Các baseline (Bảng 2: "không gian thiết kế")

| Phương pháp | On-policy | Tín hiệu dày | Không nhãn | Không verifier |
|---|---|---|---|---|
| SFT w/o KD (nhãn vàng) | không | không | không | có |
| SeqKD (phản hồi sinh bởi cùng teacher) | không | không | có | có |
| KD (teacher-forced, KL thuận T=1, không CE) | không | có | không | có |
| RLVR (PPO/GRPO) | có | không | không | không |
| **ShortOPD** | có | có | có | có |

Baseline KD được chọn để mô phỏng quy trình prune-then-distill kiểu Minitron.

## 4. Thí nghiệm và kết quả chính

**Thiết lập.** Qwen3-4B-Instruct-2507, Qwen3-4B-Thinking-2507, Qwen3-8B (no-thinking và thinking); 36 lớp; student bỏ 9 lớp BI thấp nhất (25% độ sâu; 3.11B tham số ở 4B, ~6.45B ở 8B cho width/hybrid). Kho phục hồi: trộn toán/code/hội thoại; cùng prompt cho mọi phương pháp; 710 bước, lr $2\times10^{-6}$, batch 64, 8 rollout/prompt, nhiệt độ rollout 0.8, tối đa 4096 token phản hồi, trong một node 8 GPU. Đánh giá 8 nhiệm vụ sinh tự do: GSM8K, MATH-500, HumanEval, MBPP (chính xác/pass@1), Alpaca/QA/Summarization/MT-Bench (judge GPT-5.5, 1–10). Avg = trung bình 8 nhiệm vụ (điểm judge ×10); T% = % so với teacher.

**Bảng 3 (trích): Qwen3-4B-Instruct, cắt 25%.**

| Phương pháp | GSM8K | MATH | HumanEval | MBPP | Alpaca | QA | Sum | MT-B | Avg | T% |
|---|---|---|---|---|---|---|---|---|---|---|
| Teacher đặc | 88.10 | 89.40 | 75.00 | 80.54 | 6.64 | 4.38 | 8.86 | 6.95 | 75.17 | 100 |
| Đã cắt | 1.14 | 1.60 | 0.00 | 0.00 | 1.14 | 1.00 | 1.06 | 1.09 | 5.71 | 7.6 |
| SFT w/o KD | 25.93 | 9.40 | 26.83 | 29.18 | 1.91 | 1.19 | 3.12 | 1.59 | 21.19 | 28.2 |
| SeqKD | 32.83 | 12.20 | 29.88 | 41.25 | 2.52 | 1.38 | 5.26 | 2.10 | 28.60 | 38.0 |
| KD | 29.34 | 9.60 | 29.27 | 39.30 | 2.94 | 1.48 | 6.89 | 2.36 | 30.52 | 40.6 |
| **ShortOPD** | 62.70 | 42.00 | 43.90 | 49.81 | 4.68 | 2.04 | 7.94 | 4.28 | **48.46** | 64.5 |

Avg của ShortOPD thành "~9x giá trị chưa phục hồi" và 18 điểm hơn baseline off-policy tốt nhất. Các khối khác (Avg, T%): 4B Thinking: teacher 73.75; pruned 5.21; SFT/SeqKD/KD 18.75/29.21/35.13; ShortOPD **55.48** (75.2%). 8B no-thinking: teacher 76.05; pruned 5.39; SeqKD 31.86 (tốt nhất off-policy); ShortOPD **42.84** (56.3%). 8B Thinking: teacher 77.34; pruned 5.30; SFT 35.46; ShortOPD **53.71** (69.4%). (Giao thức giải mã khác nhau giữa các khối; chỉ so trong khối.)

**Độ lặp theo độ sâu cắt (Hình 2a):** ở mốc 25%, 84% rollout có đuôi lặp. Ở cắt mạnh hơn (> khoảng 12–13 lớp) đầu ra thành "salad" token không còn lặp được (Phụ lục E). Vì vậy 25% được coi là "điểm tới hạn" của vùng có thể phục hồi nhẹ.

**Hiệu quả của bộ điều khiển (Mục 4.3):** vanilla OPD cố định $H=2048$: 55–75% rollout lặp, loss đạt plateau sau ~80 bước. ShortOPD: ~40–50 bước, ngân sách hạ xuống 1024 rồi quay lại 2048. Thời gian sinh rollout 100 bước đầu giảm 29% (35.4 → 25.1 s/bước); đầu-cuối 11.1 → 8.5 giờ. So sánh 3 lịch (Hình 5): Avg 48.5 đến 50.2; token rollout 250M (ShortOPD) / 337M (cố định 2048) / 869M (cố định 8192); fixed 8192 tốn 35.9 giờ cho +0.6 Avg so với fixed 2048; ShortOPD trong vòng 1.7 điểm Avg của cả hai và ít hơn 71% token rollout so với 8192.

**Scaling theo epoch (Bảng 4):** Avg 48.46 (710 bước) → 53.45 (2 epoch) → 55.41 (3 epoch, 73.7% teacher).

**Dày vs thưa (Bảng 5; Math+Code, 29 585 prompt):** Qwen3-4B-Instruct: PPO = 0 trên cả 4 điểm; GRPO 0.91/0.80/0/0; ShortOPD 58.68/38.00/41.46/44.36 (Gen. Avg 38.85 vs ≤ 5.38). 8B no-thinking: PPO 14.78/2.40/0/0, ShortOPD 76.88/39.60/40.24/42.41 (Avg 41.62 vs ≤ 7.68). Tác giả giải thích: student hiếm khi sinh ra đáp án đúng nên thưởng thưa gần như không có mẫu thành công để khuếch đại, còn teacher dày cho tín hiệu trên mọi rollout.

**Width / hybrid (Bảng 7, KD vs ShortOPD, cùng 710 bước):** gain 6.44–9.01 điểm.

| Mô hình, cấu trúc | Đã cắt | KD | ShortOPD | Teacher |
|---|---|---|---|---|
| 4B-Instruct, width (29.5%) | 17.23 | 43.70 | 51.46 | 74.58 |
| 4B-Instruct, hybrid (29.4%) | 19.71 | 46.15 | 55.16 | 74.58 |
| 8B no-thinking, width (21.0%) | 19.67 | 52.42 | 58.86 | 75.24 |
| 8B no-thinking, hybrid (21.2%) | 16.17 | 48.66 | 56.65 | 75.24 |

## 5. Phân tích / ablation

- **Nguồn quỹ đạo × loại tín hiệu:** cùng teacher, chuyển tín hiệu sang off-policy mất 19.9 điểm Avg (SeqKD); tín hiệu dày nhưng teacher-forced mất 17.9 điểm (KD). Cả hai thành phần (on-policy *và* dày) đều cần.
- **Corpus phục hồi (Bảng 6, leave-one-domain-out, cùng 710 bước):** Full 48.46. Bỏ **code**: HumanEval 0.61, MBPP 2.33, Avg 34.96. Bỏ **toán**: GSM8K 8.04, MATH 6.00, Avg 30.51. Bỏ **hội thoại chung**: Avg 39.02 (code tăng nhẹ nhưng toán và hội thoại tụt). Kết luận: OPD chỉ sửa các trạng thái mà student ghé qua; **độ phủ prompt phải khớp năng lực cần phục hồi**.
- **Trắc nghiệm không đủ làm proxy (Phụ lục F):** mô hình width/hybrid đã cắt giữ MC Avg 51.14 (4B width) / 69.41 (8B hybrid) nhưng Gen. Avg chỉ 17.23 / 16.17; ShortOPD hơn KD 0.87–2.08 điểm MC so với 6.44–9.01 điểm sinh. Trên trắc nghiệm dạng loglik của Qwen3-4B-Instruct (cắt sâu), SFT/KD (74.9/74.4) lại hơn ShortOPD 1 epoch (67.6; 71.6 sau 3 epoch); kết hợp SFT-init + ShortOPD đạt 76.8, ngang teacher 76.7.
- **Bộ điều khiển vs lịch cố định:** xem mục 4; tiết kiệm token mà giữ Avg trong 2 điểm.

## 6. Giới hạn

**Do tác giả nêu:** chỉ khảo sát BI depth / width / hybrid trên Qwen3 4B và 8B; cần thêm họ mô hình, tỷ lệ nén và quy mô, và điểm mà nén quá mạnh khiến phục hồi nhẹ không đủ.

**Chúng ta nhận thấy:**
- Bài còn là bản "work in progress"; nhiều so sánh một lần chạy (không thấy độ lệch chuẩn, không có seed).
- Hiện tượng "lặp ở đuôi" được đo ở **mức 25% độ sâu**; ở cắt mạnh hơn lỗi chuyển sang đầu ra không mạch lạc nên chẩn đoán và bộ điều khiển có thể không còn áp dụng.
- OPD cần teacher (mô hình gốc) **chạy trực tuyến** trên rollout của student: tốn bộ nhớ/tính toán hơn KD với logit cache; không so sánh chi phí tổng quát với KD ngoài giờ rollout.
- Baseline KD không có CE cứng, nhiệt độ 1.0, forward KL; một baseline KD được tinh chỉnh hơn (trộn CE, nhiệt độ, nhiều epoch) có thể thu hẹp khoảng cách. ShortOPD ở 1 epoch, nhưng 3 epoch của baseline không được báo cáo.
- Thực nghiệm Math+Code RLVR chỉ chạy đến bước 462 và mang tính kiểm soát; không loại trừ rằng RL tinh chỉnh hơn (khởi động bằng SFT) làm tốt hơn.
- Độ phủ prompt là yếu tố lớn (Bảng 6) nhưng chỉ khảo sát 3 miền.
- Bảng 1 (pass@k) dùng cắt 4 lớp còn các bảng chính cắt 9 lớp, nên hai quan sát "hạ cấp chứ không xoá" và kết quả phục hồi không đo trên cùng một student.

## 7. Liên hệ với các bài khác

- **29 (GKD) / 28 (MiniLLM):** nguồn gốc ý tưởng distill trên trạng thái của student, và JSD/KL nghịch; ShortOPD dùng JSD $\alpha=0.5$ chặn importance weight 2.0.
- **15 (SeqKD) / 46 (Distil-Whisper):** baseline SeqKD ~ pseudo-label; Distil-Whisper thành công nhờ pseudo-label sạch, còn ở đây SeqKD thua OPD vì độ hư hại lớn hơn sau cắt nặng.
- **31 (Rethinking KL):** chọn $D_\alpha$ ở giữa; tác giả lập luận trade-off này độc lập với on/off-policy.
- **45 (Sparse logit sampling):** lưu top-K logit chệch; ShortOPD dùng **top-100 + bin đuôi gộp** (giống ghost token), một cách hợp lý để tránh bias mà không lưu cả từ vựng.
- **40 (Gemma 2), 44 (Qwen3):** các pipeline distillation chính thức; ShortOPD cho thấy phục hồi sau cắt cần xử lý khác hẳn pre-training distillation.
- **39 (Small models struggle with strong reasoners):** góc nhìn bổ sung về khoảng cách năng lực student–teacher.

## 8. Ứng dụng cho dự án (suy luận)

Dự án: cắt LightOnOCR-3-0.8B xuống ~0.3B/~0.1B và phục hồi bằng teacher 4B; bạn đã lưu top-20 logprob; quan sát cắt neuron MLP phục hồi kém hơn cắt lớp.

1. **Công thức bạn đang dùng thực chất là "KD" trong bài này** (offline, teacher-forced, logit dày). Đây là baseline kém hơn OPD ~18 điểm Avg trong bài (sinh tự do, LLM văn bản, cắt 25%). OCR khác (đầu ra gần tất định, ít đa dạng) nên khoảng cách có thể nhỏ hơn, nhưng đây là rủi ro đáng thử: **exposure bias** nghiêm trọng khi student cắt mạnh bắt đầu lệch và lặp trên chuỗi dài.
2. **Lặp ở đuôi rất có thể xuất hiện** ở OCR (dòng/bảng lặp). Hãy đo tỷ lệ rollout có đuôi lặp ở student đã cắt; nếu cao, bộ dò vòng lặp (Phụ lục B) dùng nguyên được cho chuỗi ký tự, ngay cả khi không dùng OPD, để lọc/chấm.
3. **Teacher ≠ pre-compression self.** Bài dùng chính mô hình gốc làm teacher (tự chưng cất), còn bạn dùng teacher 4B lớn hơn. Cần kiểm tra OPD có hiệu quả như nhau khi teacher lớn hơn (khoảng cách năng lực lớn hơn); tác giả nói forward/reverse KL khác biệt phụ thuộc khoảng cách này, nên $\alpha=0.5$ là điểm xuất phát hợp lý.
4. **Đuôi top-20:** nếu dùng cache offline thì giữ khối lượng đuôi như bin đuôi của bài (và ghost token của 45). Nếu dùng OPD thì teacher phải chạy trực tuyến trên rollout của student, **không dùng được cache logit** (rollout là mới); chi phí GPU của teacher 4B trong huấn luyện cần tính.
5. **Phân biệt D_lin / D_res.** Paper không phân rã sai số, nhưng kết quả width vs depth gợi ý: width đã cắt khởi điểm cao hơn (Gen. Avg 16–20 so với 5.7 ở depth) và vẫn được ShortOPD nâng lên 51–59, KD 43–52. Không so được trực tiếp với khẳng định "cắt MLP neuron phục hồi kém" của bạn vì ở đây tỷ lệ cắt, mô hình và cơ chế khác nhau; bài không có bằng chứng ủng hộ hay bác bỏ quan sát đó.
6. **Độ phủ corpus (Bảng 6):** nếu bỏ một miền (bảng, công thức, chữ viết tay) khỏi corpus phục hồi thì miền đó có thể sụp. Cần phân bổ prompt/ảnh theo cả loại tài liệu hiếm.
7. **Thực tế cho GPU ≤ 8 GB và batch-1:** bài không đo độ trễ suy luận của student; lợi ích là về chất lượng phục hồi, không về tốc độ.

## 9. Thí nghiệm gợi ý

1. **Chẩn đoán lặp:** sinh transcript greedy từ student đã cắt (0.3B) và đếm tỷ lệ đuôi lặp bằng bộ dò Phụ lục B; nếu < 10% thì phần "short-to-long" ít lợi và OPD thuần đã đủ.
2. **Mini-OPD vs KD offline:** cùng ngân sách bước, phục hồi student 0.3B bằng (a) KD cache top-20 + ghost token, (b) OPD đơn giản (rollout ≤ 1024 token, JSD α=0.5, top-20 + bin đuôi) với teacher 4B trực tuyến. Báo cáo CER và tỷ lệ lặp ở trang dài.
3. **Khởi động kép:** (a) KD offline vài bước để khép khoảng cách thô, rồi (b) OPD (ý tưởng "SFT-init ShortOPD" ở Phụ lục F). Xem có thu hẹp thời gian rollout lặp không.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / giá trị | Ghi chú |
|---|---|---|
| Block Influence | $1-\mathbb{E}[\cos(X_i,X_{i+1})]$ | bỏ lớp BI thấp |
| OPD loss | $\mathbb{E}_{y\sim\pi_\theta}\frac1{|y|}\sum_t D_\alpha(\pi_T,\pi_\theta)$ | on-policy, dày, không nhãn |
| JSD tổng quát | $D_\alpha=\alpha\mathrm{KL}(p\|m)+(1-\alpha)\mathrm{KL}(q\|m)$ | $\alpha=0.5$ |
| Gradient gần khớp | $\approx\alpha(1-\alpha)\delta_j$ | 1/4 của KL thuận tại $\alpha=0.5$ |
| Mục tiêu top-K | top-100 + 1 bin đuôi | clip IS = 2.0 |
| Cổng rút ngắn | $\bar r>0.45\Rightarrow H^*=\min(H,1.15\bar L)$ | |
| Cổng tăng | $\bar r<0.20\wedge\bar q>0.10\Rightarrow H^*=1.25H$ | |
| Làm mượt | $H'=0.7H+0.3H^*$, bội 16, $[1024,2048]$ | |
| Kết quả chính | Avg 5.71 → 48.46 (4B-Instruct, −25%) | KD 30.52, SeqKD 28.60 |
| Hiệu quả | 8.5 vs 35.9 giờ; 250M vs 869M token | −71% token |

## 11. Kết luận ngắn

Sau cắt cấu trúc, năng lực sinh bị hạ cấp (pass@k cao) và đầu ra hỏng chủ yếu do lặp ở đuôi. Phục hồi hiệu quả nhất trong bài là OPD: student tự sinh, teacher tiền-cắt chấm lại từng token bằng JSD top-100 + đuôi; ShortOPD thêm bộ điều khiển độ dài rollout dựa trên phát hiện lặp để tiết kiệm 71% token mà gần như không mất chất lượng. Với dự án OCR, hãy xem cache top-20 offline của bạn là baseline "KD", đo tỷ lệ lặp ở student, và thử OPD ngắn như một bước sau cùng; nhớ rằng bằng chứng của bài là trên LLM văn bản với cắt lớp theo BI, chưa phải OCR hay cắt MLP.

Nguồn: https://arxiv.org/abs/2607.13124
