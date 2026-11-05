# DistiLLM: Towards Streamlined Distillation for Large Language Models (2024)

**Tác giả:** Jongwoo Ko, Sungnyun Kim, Tianyi Chen, Se-Young Yun (KAIST AI; Microsoft).  
**Venue / arXiv:** arXiv:2402.03898 (v2, 03/07/2024); ICML 2024 (PMLR 235).  
**Nguồn:** https://arxiv.org/abs/2402.03898  
**Chủ đề:** Distillation cho LLM tự hồi quy: skew KL / skew reverse KL, và cách dùng dữ liệu student tự sinh (SGO) hiệu quả bằng scheduler thích nghi và replay buffer.  
**Vị trí trong lộ trình:** Đọc trước: 02 (Hinton KD), 15 (Sequence-level KD), 28 (MiniLLM), 29 (GKD). Bài này giữ ý tưởng on-policy của GKD nhưng (a) thay loss bằng skew KL ổn định hơn, (b) giảm mạnh chi phí sinh mẫu. Đọc tiếp: 31 (Rethinking KL), 33 (TAID, cũng nội suy phân phối), 34 (DistiLLM-2, cùng nhóm tác giả).

> Ghi chú học tập bằng tiếng Việt, diễn giải lại, không phải bản dịch. Phần "Ứng dụng cho dự án" là suy luận của chúng ta. Một số bảng/hình trong bản trích xuất PDF bị lẫn cột; chỉ các số đọc rõ mới được nêu lại.

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

Distillation cho LLM trước bài này có hai nhóm vấn đề.

**(a) Hàm mục tiêu không chuẩn.** KL thuận (forward KLD) $D_{KL}(p\|q_\theta)$ ép student phủ mọi chỗ teacher có xác suất, dẫn đến *mode averaging* (phân phối student quá "mịn"). Reverse KL (MiniLLM) hoặc generalized JSD (GKD, note 29) giảm vấn đề này nhưng thiếu phân tích lý thuyết và kết quả phụ thuộc tác vụ.

**(b) Dữ liệu và chi phí.** Dữ liệu cố định gây exposure bias (xem note 29). Dùng **student-generated outputs (SGO)** giúp nhưng: (i) sinh SGO ở mỗi iteration làm thời gian train tăng rất nhiều (tới 80% tổng thời gian dành cho sinh, và tới 5.5x runtime theo Fig. 7 của paper); (ii) teacher chưa từng thấy SGO nên có thể cho phản hồi nhiễu. Ví dụ Fig. 1: teacher GPT-2 XL cho validation loss thấp cho câu trả lời sai nhưng ngắn ("Christopher Columbus") và loss cao cho câu dài đúng, nên dùng $p(y|x)$ làm đích có thể dẫn student đi sai.

Gloss: **SGO** là chuỗi do chính student sinh; **replay buffer** (từ RL off-policy) là kho lưu mẫu cũ để dùng lại thay vì sinh mới.

## 2. Ý tưởng chính

DistiLLM có hai thành phần. Một: **skew KL (SKL)** và **skew reverse KL (SRKL)**: so student với một *hỗn hợp* $\alpha p+(1-\alpha)q_\theta$ thay vì so thẳng với $p$, nhờ đó mẫu số của tỉ số không bao giờ về 0, gradient ổn định và hội tụ nhanh. Hai: **adaptive off-policy**: lúc đầu dùng chủ yếu dữ liệu cố định, tăng dần xác suất dùng SGO theo validation loss (tránh phản hồi nhiễu), và chỉ sinh SGO thưa rồi lưu vào replay buffer để tái sử dụng (tránh sinh mỗi bước). Hai thành phần cộng hưởng vì SKL hội tụ nhanh nên chịu được dữ liệu "hơi cũ" từ buffer.

## 3. Phương pháp chi tiết

### 3.1 Skew KLD

Ký hiệu teacher $p$, student $q_\theta$. Với tham số trộn $\alpha\in[0,1]$:

$$\boxed{D^{(\alpha)}_{SKL}(p,q_\theta)=D_{KL}\big(p,\ \alpha p+(1-\alpha)q_\theta\big)},\qquad D^{(\alpha)}_{SRKL}(p,q_\theta)=D_{KL}\big(q_\theta,\ (1-\alpha)p+\alpha q_\theta\big)$$

Gradient. KL thường theo $\theta$ (cho một mẫu): $\nabla_\theta D_{KL}(p,q_\theta)=-r_{p,q_\theta}\nabla_\theta q_\theta(y|x)$, với $r=p/q_\theta$. Khi $q_\theta\to0$, $r$ nổ, gradient lớn và nhiễu. Với SKL:

$$\nabla_\theta D^{(\alpha)}_{SKL}(p,q_\theta)=-(1-\alpha)\,r_{p,\tilde q_\theta}\,\nabla_\theta q_\theta(y|x),\quad \tilde q_\theta=\alpha p+(1-\alpha)q_\theta$$

Vì $\tilde q_\theta\ge\alpha p$, mẫu số không về 0, hệ số gradient bị chặn: đây là **gradient ổn định**. Gradient của reverse KL là $-(\log r_{q_\theta,p}+1)\nabla_\theta q_\theta$ và SRKL có dạng tương tự với hệ số $\log r_{q_\theta,\tilde p}+1-\alpha r_{q_\theta,\tilde p}$, $\tilde p=(1-\alpha)p+\alpha q_\theta$.

**Sai số xấp xỉ nhỏ (Theorem 1).** Với ước lượng thực nghiệm từ $n$ mẫu, sai số $L_2$ của $\alpha$-SKL bị chặn bởi $\frac{c_1(\alpha)}{n^2}+\frac{c_2\log^2(\alpha n)}{n}+\frac{c_3\log^2(c_4 n)}{\alpha^2 n}$ (với $c_1(\alpha)=\min\{1/\alpha^2,\ \chi^2(p_1,p_2)^2/(1-\alpha)^2\}$), tức $\alpha$ lớn làm sai số ước lượng nhỏ đi. Nhưng $\alpha$ lớn cũng làm hệ số gradient nhỏ đi, và optimizer hiện đại (Adam) sẽ bù lại nên lợi ích bị triệt tiêu. Khi xét sai số chuẩn hoá theo $1/(1-\alpha)$ (Remark 1), quan hệ giữa chặn trên và $\alpha$ là hàm lồi, nên cần cân bằng. Kết luận thực nghiệm: **$\alpha=0.1$** tốt nhất cho cả SKL và SRKL (Fig. 8), khớp với chỗ L2 chuẩn hoá nhỏ nhất (Fig. 3).

Khác biệt với JSD: $D_{JSD}^{(\beta)}=\beta D^{(1-\beta)}_{SKL}(p,q_\theta)+(1-\beta)D^{(\beta)}_{SKL}(q_\theta,p)$ không thể đặt mức skew nhỏ cho cả hai số hạng cùng lúc, còn S(R)KL với $\alpha$ vừa phải thì đạt L2 hợp lý.

Ghi chú: paper chỉ làm việc với họ KL vì nó phân rã được từ mức chuỗi xuống mức token (total variation không phân rã như vậy).

### 3.2 Adaptive off-policy

- **SGO scheduler thích nghi.** Xác suất dùng SGO là $\phi$ (còn $1-\phi$ dùng dữ liệu cố định). Bắt đầu $\phi=0$. Mỗi lần validate, nếu validation loss tăng hơn lần trước cộng $\epsilon$, thì $\phi\leftarrow\min(\phi+1/N_\phi,1)$, ngược lại giữ nguyên ($N_\phi=10$ trong các thí nghiệm; $\phi_0=0$). Ý tưởng: huấn luyện trên SGO đa dạng làm giảm mismatch và giảm validation loss; khi loss có dấu hiệu xấu đi thì tăng SGO.
- **Off-policy với replay buffer.** Chỉ sinh SGO mới với xác suất $\lambda_R=\phi(1-t/T)$, lưu vào buffer $\mathcal D_R$ (dung lượng 1000; đầy thì thay mẫu cũ nhất); khi cần dùng SGO thì rút ngẫu nhiên từ buffer. Hệ số $\zeta=1-t/T$ là **replay ratio** giảm tuyến tính: đầu train (student thay đổi nhanh) dùng nhiều SGO mới để giảm bias, cuối train (đã hội tụ) tái sử dụng buffer nhiều.

### 3.3 Pseudo-code (Algorithm 1, rút gọn)

```
Cho: teacher p, student q_theta, dataset D, D_val, buffer D_R = {}, phi=0, T bước
for t in 1..T:
    u ~ U(0,1)
    if u < phi*(1 - t/T):          # sinh SGO mới (xác suất lambda_R, thưa)
        y_tilde ~ q_theta(.|x);  D_R += {(x, y_tilde)}
    if u < phi:   batch <- mẫu ngẫu nhiên từ D_R   # dùng SGO (off-policy)
    else:         batch <- mẫu từ D                 # dữ liệu cố định
    cập nhật theta bằng loss SKL (hoặc SRKL), alpha=0.1
    if đến lượt validate:
        L = val_loss(q_theta)
        if L > L_prev + eps:  phi = min(phi + 1/N_phi, 1.0)
        else:                 L = L_prev   # giữ phi
```

Cấu hình triển khai: SRKL với $\alpha=0.1$, phi ban đầu 0, buffer 1000 (theo ablation).

## 4. Thí nghiệm và kết quả chính

**Instruction-following (task-agnostic).** Dữ liệu databricks-dolly-15k (14K train, 500 val, 500 test). Đánh giá: Dolly eval, Self-Instruct, Vicuna eval, Super-Natural Instructions, Unnatural Instruction; metric ROUGE-L và GPT-4 feedback. Thêm loss language modeling trên OpenWebText (theo MiniLLM). Cặp teacher→student: GPT-2 XL (1.5B)→GPT-2 (0.1B); OPT-2.7B→OPT-1.3B; OpenLLaMA2-7B→3B (LoRA). Huấn luyện 20 epoch, LR trong {5e-4, 1e-4, 5e-5}, batch trong {8,16,32}, A100 40GB.

Bảng 1 (GPT-2 XL → GPT-2; ROUGE-L trung bình qua 5 seed, độ lệch chuẩn trong ngoặc):

| Loss | Dolly | Self-Inst | Vicuna | Super-Nat | Unnatural |
|---|---|---|---|---|---|
| KLD | 23.52 | 11.23 | 15.92 | 20.68 | 23.38 |
| RKLD | 23.82 | 10.90 | 16.11 | 22.47 | 23.03 |
| Generalized JSD | 24.34 | 12.01 | 15.21 | 25.08 | 27.54 |
| SKL | 24.80 | 12.86 | 16.20 | 26.26 | 28.06 |
| SRKL | 25.21 | 12.98 | 15.77 | 25.83 | 28.62 |

Bảng 2 (adaptive off-policy, SKL / SRKL): thuần on-policy cho Dolly 24.27 / 26.04; mixed 25.27 / 26.01; adaptive (ours) 25.90 / 26.37; thêm off-policy (ours) 25.79 / 26.11. Tức off-policy gần như không mất hiệu năng so với adaptive thuần on-policy (SKL: 25.79 so với 25.90 trên Dolly).

**Tốc độ.** Trên OpenLLaMA2-3B, DistiLLM chỉ tốn khoảng 1.6x thời gian train của KD thường, còn các phương pháp khác 3-7x (Fig. 5d). Tổng quát paper nêu tăng tốc **2.5-4.3x** so với MiniLLM/GKD (tóm tắt: tới 4.3x). Với S(R)KL, cách sinh adaptive + off-policy nhanh hơn on-policy hoặc mixed từ 2.2x đến 3.4x (Sec. 5.2).

**Tóm tắt và dịch (Bảng 3, teacher T5-XL 3B; SAMSum ROUGE-L, IWSLT17 En-De BLEU):**

| Phương pháp | SAMSum Base | SAMSum Small | IWSLT Base | IWSLT Small |
|---|---|---|---|---|
| KD | 46.23 | 39.52 | 29.36 | 21.15 |
| SeqKD | 46.89 | 40.24 | 29.07 | 21.42 |
| ImitKD | 48.57 | 41.44 | 29.87 | 21.52 |
| GKD | 48.49 | 41.92 | 30.24 | 22.04 |
| DistiLLM | 49.11 | 42.37 | 30.32 | 22.53 |

(Teacher cho dịch là mT5-XL; student mT5-Base/Small.) Biên độ cải thiện ở tác vụ đơn nhỏ hơn ở instruction-following. Paper có thêm kết quả XSum và CNN/DM ở phụ lục.

## 5. Phân tích / ablation

- **Skew divergence** (Bảng 1, Fig. 6): SKL, SRKL cho validation ROUGE-L cao ngay giai đoạn đầu, tức hội tụ nhanh và tổng quát tốt; thắng KLD, RKLD, JSD ở hầu hết tập.
- **Giá trị $\alpha$** (Fig. 8): tối ưu tại 0.1 cho cả hai; SKL giảm nhẹ khi $\alpha>0.1$, SRKL giảm rõ hơn.
- **Cộng hưởng với off-policy** (Bảng 4): thay on-policy bằng off-policy làm ImitKD và GKD tụt (ví dụ Dolly: ImitKD 21.63 → 20.62; GKD 23.75 → 22.89), nhưng DistiLLM gần như giữ nguyên (26.37 → 26.12). Paper giải thích do JSD/KLD hội tụ chậm hơn SRKL.
- **Scheduler thích nghi vs tune tay** (Bảng 5): xác suất cuối do scheduler chọn (0.4 cho SKL, 0.3 cho SRKL) gần mức tune tay tốt nhất (lệch không quá 0.2 với SKL và 0.1 với SRKL), và hiệu năng thường bằng hoặc tốt hơn, nên khỏi dò xác suất bằng tay.
- **Không cần SFT trước** (Bảng 6): khởi tạo từ GPT-2 chưa fine-tune, DistiLLM chỉ giảm rất nhẹ (ví dụ Dolly 25.79 → 25.75 với SKL) trong khi MiniLLM giảm 23.84 → 22.56 và GKD 23.75 → 23.15. Đây là điểm khác GKD (vốn giả định student đã SFT).

## 6. Giới hạn

**Tác giả nêu (Phụ lục A, đọc được một phần):** tập trung vào họ (R)KLD vì tính khả kế của nó (phân rã từ chuỗi xuống token), chưa xét các divergence khác như total variation; chỉ ghi nhận hai hạn chế, phần còn lại của đoạn bị vỡ trong bản trích xuất nên không nêu thêm.

**Chúng ta nhận xét (suy luận):**
- Thí nghiệm chủ yếu ở cỡ nhỏ-trung (tối đa 7B→3B, có dùng LoRA), dữ liệu dolly 14K; chưa kiểm chứng ở quy mô pretrain.
- Gradient bị chặn nhờ trộn $p$ nhưng $\alpha=0.1$ tối ưu theo thực nghiệm trên một số cấu hình; có thể cần dò lại cho miền khác.
- Tiêu chí scheduler (validation loss) giả định có tập validation cùng phân phối với dữ liệu train.
- Cần cùng tokenizer/vocab giữa teacher và student (do KL token-level).

## 7. Liên hệ với các bài khác

- **29 GKD:** DistiLLM thay JSD bằng skew KL và thay "mỗi bước lấy mẫu on-policy" bằng buffer; trong thực nghiệm GKD/ImitKD là baseline chính.
- **28 MiniLLM:** policy gradient cho reverse KL, cần sinh từ cả hai mô hình mỗi bước; DistiLLM tránh điều đó.
- **31 Rethinking KL:** phân tích rằng forward và reverse KL đều hội tụ về cùng đích ở huấn luyện lâu, và đề xuất loss ABKD/AKL; nên đọc để hiểu giới hạn của luận điểm "mode-seeking tốt hơn".
- **33 TAID:** nội suy teacher-student theo thời gian, ý tưởng liên quan đến mixture $\alpha p+(1-\alpha)q$ nhưng với $\alpha$ thay đổi.
- **34 DistiLLM-2:** cùng nhóm tác giả, thay vì cùng loss cho mọi mẫu thì dùng loss tương phản (teacher response vs student response).
- **30 vs 02 Hinton:** về mặt khái niệm skew KL là "temperature" theo kiểu khác: làm mềm đích bằng hỗn hợp.

## 8. Ứng dụng cho dự án (suy luận)

- **Liên quan khá cao.** Ta cần phục hồi student đã cắt từ teacher 4B bằng dữ liệu on-policy mà không đốt thời gian sinh mẫu. Replay buffer + $\lambda_R$ thưa là cách rẻ để thêm tín hiệu on-policy cho OCR-VLM.
- Skew KL có lợi đặc biệt khi student bị cắt mạnh (0.1B): xác suất $q_\theta$ trên token đúng có thể rất nhỏ ngay sau khi cắt, đúng chế độ mà KL thường cho gradient nổ. Hệ số trộn $\alpha=0.1$ là điểm xuất phát hợp lý, cần dò lại.
- Tính chất "không cần SFT trước" gợi ý có thể dùng ngay sau cắt, nhưng bằng chứng của paper chỉ trên GPT-2 nhỏ, nên cần kiểm chứng.
- Với OCR, ảnh đầu vào cố định; chỉ phần sinh văn bản là on-policy, nên kiến trúc buffer lưu cặp (ảnh, chuỗi student) rất gọn.
- Lưu ý: tốn bộ nhớ train cho teacher 4B + student; chỉ ràng buộc 8GB ở inference.
- Paper không bàn D_lin/D_res; liên hệ chỉ là gián tiếp (ổn định gradient khi $q_\theta$ nhỏ).

## 9. Thí nghiệm gợi ý

1. Trên student 0.3B đã cắt: so KL, reverse KL, SKL, SRKL ($\alpha\in\{0.05,0.1,0.3\}$) cùng ngân sách, đo CER và độ ổn định (norm gradient theo bước).
2. Thêm replay buffer 1000 mẫu với $\lambda_R=\phi(1-t/T)$ và so với on-policy mỗi bước về CER và thời gian train.
3. So cắt neuron MLP với cắt layer dưới cùng pipeline: xem hiệu ứng của skew KL có lớn hơn cho biến thể cắt neuron (xác suất student ban đầu thấp hơn) hay không.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| SKL | $D_{KL}(p,\alpha p+(1-\alpha)q_\theta)$ |
| SRKL | $D_{KL}(q_\theta,(1-\alpha)p+\alpha q_\theta)$ |
| Gradient SKL | $-(1-\alpha)\,r_{p,\tilde q}\nabla q_\theta$, bị chặn nhờ trộn |
| $\alpha$ tốt nhất | 0.1 (thực nghiệm và L2 chuẩn hoá) |
| SGO | Chuỗi do student tự sinh |
| $\phi$ | Xác suất dùng SGO, tăng khi val loss tăng |
| $\lambda_R=\phi(1-t/T)$ | Xác suất sinh SGO mới (replay ratio giảm tuyến tính) |
| Replay buffer | Dung lượng 1000, thay mẫu cũ nhất |
| Tốc độ | 2.5-4.3x nhanh hơn MiniLLM/GKD; khoảng 1.6x KD thường |

## 11. Kết luận ngắn

DistiLLM cho thấy hai cải tiến nhỏ nhưng thực dụng: trộn một phần teacher vào đích so sánh (skew KL, $\alpha=0.1$) để có gradient ổn định, hội tụ nhanh; và dùng SGO có kiểm soát (scheduler theo val loss + replay buffer) để giữ lợi ích on-policy mà tốn ít thời gian. Điểm cần nhớ: loss hội tụ nhanh là điều kiện để off-policy tái sử dụng dữ liệu không bị phạt.

Nguồn gốc: https://arxiv.org/abs/2402.03898
