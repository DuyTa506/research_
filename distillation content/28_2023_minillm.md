# MiniLLM: On-Policy Distillation of Large Language Models (2023)

**Tác giả:** Yuxian Gu, Li Dong, Furu Wei, Minlie Huang (Tsinghua CoAI; Microsoft Research).  
**Venue / arXiv:** arXiv:2306.08543 (v1 tháng 6/2023; bản v6 ngày 31/1/2026 dùng để ghi chú này). Paper trong text đã trích không nêu venue.  
**Nguồn:** https://arxiv.org/abs/2306.08543  
**Chủ đề:** Distill LLM sinh văn bản (white-box) bằng **reverse KL** thay vì forward KL; tối ưu bằng policy gradient on-policy với ba mẹo ổn định (single-step decomposition, teacher-mixed sampling, length normalization).  
**Vị trí trong lộ trình:** Đọc sau 02 (Hinton KD, forward KL), 15 (Sequence-level KD, SeqKD là baseline chính) và 06 (Born-again networks, student vượt teacher). Bài này mở nhóm KD cho LLM sinh văn bản (28–34). Nó đặt ra câu hỏi "dùng divergence nào, lấy mẫu từ đâu"; các bài sau trả lời tiếp: 29 GKD (on-policy với divergence tổng quát), 30 DistiLLM (skew KL, off-policy hiệu quả), 31 Rethinking KL (xem lại tranh luận forward/reverse), 33 TAID.

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch nguyên văn paper. Các phép suy dẫn trình bày lại theo hướng dễ hiểu; số liệu lấy từ paper. Phần "Ứng dụng cho dự án" là suy luận của người ghi chú.

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

Hai loại KD cho LLM:
- **Black-box KD:** chỉ có văn bản do teacher sinh (ví dụ gọi API ChatGPT), student fine-tune trên cặp (prompt, response) đó.
- **White-box KD:** có cả phân phối xác suất đầu ra (và có thể hidden state) của teacher. Quan trọng hơn khi các LLM mã nguồn mở nở rộ. Trước bài này, white-box KD chủ yếu dùng cho mô hình hiểu ngôn ngữ nhỏ (<1B) kiểu DistilBERT/TinyBERT (16, 18).

Các KD chuẩn (word-level KD của Hinton, và sequence-level KD / SeqKD của Kim & Rush, bài 15) đều xấp xỉ giảm **forward KL** $\mathrm{KL}[p\|q_\theta]$ với $p$ là teacher, $q_\theta$ là student. Nhắc nhanh:

$$\mathrm{KL}[p\|q]=\mathbb{E}_{y\sim p}\log\frac{p(y)}{q(y)},\qquad \mathrm{KL}[q\|p]=\mathbb{E}_{y\sim q}\log\frac{q(y)}{p(y)}$$

- **Forward KL** $\mathrm{KL}[p\|q]$: phạt nặng nếu student gán xác suất ~0 ở nơi teacher có xác suất. Do đó student phải **phủ mọi mode** (mass-covering). Nếu student quá nhỏ để biểu diễn mọi mode, nó "phết" xác suất lên cả vùng trống giữa các mode (void regions), và khi sinh tự do sẽ ra những mẫu rất khó xảy ra dưới teacher.
- **Reverse KL** $\mathrm{KL}[q\|p]$: phạt nặng nếu student gán xác suất ở nơi teacher gần 0. Student **chọn vài mode chính** của teacher và bỏ phần đuôi (mode-seeking). Hình 2 trong paper là thí nghiệm đồ chơi: khớp hỗn hợp Gauss bằng một Gauss đơn.

Với phân loại văn bản, đầu ra ít lớp nên forward KL ổn. Với sinh văn bản mở, đầu ra của teacher có rất nhiều mode mà student nhỏ không biểu diễn hết, nên tác giả cho rằng forward KL là chưa tối ưu.

Khái niệm phụ:
- **On-policy:** lấy mẫu từ chính mô hình đang được huấn luyện (student), thay vì từ dữ liệu cố định hoặc teacher.
- **Exposure bias:** khi huấn luyện teacher forcing, mô hình luôn thấy tiền tố đúng; khi sinh tự do thì thấy tiền tố do chính nó tạo, tạo mismatch tích lũy.
- **Policy gradient (REINFORCE):** ước lượng gradient của kỳ vọng phần thưởng dưới một policy bằng $\mathbb{E}[\text{reward}\cdot\nabla\log\pi]$.
- **Reward hacking:** mô hình khai thác kẽ hở của hàm phần thưởng (ví dụ lặp từ để điểm cao).
- **ECE (Expected Calibration Error):** sai số hiệu chỉnh xác suất (thấp là tốt).

## 2. Ý tưởng chính

Đổi mục tiêu KD từ forward KL sang **reverse KL** $\mathrm{KL}[q_\theta\|p]$, để student tập trung vào các mode chính của teacher thay vì cố bắt chước cả đuôi dài. Vì kỳ vọng nằm trên mẫu của *student*, ta không thể dùng dữ liệu có sẵn, mà phải sinh mẫu từ student rồi chấm điểm bằng teacher: đó chính là bài toán policy gradient (giống RLHF, nhưng phần thưởng là $\log p/q$). Paper rút ra gradient, rồi thêm ba mẹo giảm phương sai và chặn reward hacking. Kết quả: student sinh câu trả lời chính xác hơn, ít exposure bias hơn, hiệu chỉnh tốt hơn so với KD và SeqKD, từ 120M đến 13B tham số.

## 3. Phương pháp chi tiết

### 3.1 Mục tiêu

Điều kiện: prompt $x\sim p_x$, response $y=\{y_t\}_{t=1}^T$.

$$\boxed{\theta^*=\arg\min_\theta\ \mathcal{L}(\theta)=\arg\min_\theta\ \mathrm{KL}[q_\theta\|p]=-\mathbb{E}_{x\sim p_x,\,y\sim q_\theta}\log\frac{p(y|x)}{q_\theta(y|x)}}$$

Góc nhìn thứ hai (Phụ lục A.1): coi sinh token là MDP, KD chuẩn là *behavior cloning*, còn MiniLLM là Inverse RL kiểu max-entropy. Lấy logit $f$ của teacher làm Q-function, phần thưởng là $r_t=f(y_t)-\log\sum_{y'}e^{f(y')}=\log p(y_t|y_{<t},x)$. Tối đa hóa $\sum_t r_t+H[q_\theta]$ cho ra (xấp xỉ, bỏ một số hạng biên) đúng $-\mathrm{KL}[q_\theta\|p]$.

### 3.2 Gradient bằng policy gradient (Phụ lục A.2, trình bày lại)

Bắt đầu từ $\nabla\mathcal{L}=-\nabla\int q_\theta(y)\log\frac{p(y)}{q_\theta(y)}\,dy$ (bỏ $x$ cho gọn). Dùng quy tắc tích và $\nabla q_\theta=q_\theta\nabla\log q_\theta$:

$$\nabla\mathcal{L}=-\int q_\theta\,\nabla\log q_\theta\,\log\frac{p}{q_\theta}\,dy\;-\;\int q_\theta\,\nabla\log\frac{p}{q_\theta}\,dy$$

Vì $\nabla\log\frac{p}{q_\theta}=-\nabla\log q_\theta$, số hạng thứ hai bằng $+\int q_\theta\nabla\log q_\theta\,dy=\mathbb{E}_{q}[\nabla\log q_\theta]$, tức là một số hạng có kỳ vọng bằng 0 (nhưng paper giữ nó lại để có dạng "$-1$"):

$$\nabla\mathcal{L}(\theta)=-\mathbb{E}_{y\sim q_\theta}\Big[\Big(\log\frac{p(y|x)}{q_\theta(y|x)}-1\Big)\nabla\log q_\theta(y|x)\Big]$$

Hằng số $-1$ chỉ đóng vai trò baseline, không đổi giá trị kỳ vọng. Phân rã theo token, vì $\log q_\theta(y_t|y_{<t})$ chỉ ảnh hưởng các token từ vị trí $t$ trở đi:

$$\boxed{\nabla\mathcal{L}(\theta)=-\mathbb{E}_{x,\;y\sim q_\theta}\sum_{t=1}^{T}(R_t-1)\,\nabla\log q_\theta(y_t|y_{<t},x),\quad R_t=\sum_{t'=t}^{T}\log\frac{p(y_{t'}|y_{<t'},x)}{q_\theta(y_{t'}|y_{<t'},x)}}$$

Trực giác: $r_{t'}=\log p-\log q_\theta$ là "phần thưởng" từng bước: cao nếu teacher thấy token này hợp lý ($p$ lớn), và bị trừ nếu student quá tự tin ($q$ lớn), giữ cho student đa dạng. $R_t$ là phần thưởng tích lũy từ bước $t$ (reward-to-go).

Hai vấn đề (paper): phương sai cao và reward hacking; và $R_t$ thiên về câu **ngắn** (tổng các $r$ âm càng ít càng tốt), dẫn tới trả lời rỗng.

### 3.3 Ba mẹo

**(1) Single-step decomposition.** Lỗi ở token đầu cộng dồn qua cả câu nên rất ồn. Tách phần bước hiện tại $r_t$ ra khỏi $R_t=r_t+R_{t+1}$. Phần kỳ vọng theo $y_t\sim q_\theta(\cdot|y_{<t})$ của $r_t$ tính **trực tiếp bằng cách cộng trên toàn từ vựng** thay vì lấy mẫu:

$$\nabla\mathcal{L}=\underbrace{-\mathbb{E}\,\nabla\sum_{t}\mathbb{E}_{y_t\sim q_\theta(t)}[r_t]}_{(\nabla\mathcal{L})_{Single}}\ \underbrace{-\ \mathbb{E}\sum_t R_{t+1}\nabla\log q_\theta(y_t|y_{<t},x)}_{(\nabla\mathcal{L})_{Long}}$$

Chứng minh (A.3): từ $(R_t-1)=R_{t+1}+(r_t-1)$; số hạng $(r_t-1)\nabla\log q_\theta$ có kỳ vọng theo $y_t$ bằng $\nabla\mathbb{E}_{y_t}[-r_t]$... (dùng quy tắc tích và $\mathbb{E}\nabla\log q=0$), chính là $\nabla\mathbb{E}_{y_t\sim q}[-r_t]=\nabla\,\mathrm{KL}[q_\theta(t)\|p(t)]$ ở bước $t$. Nghĩa là phần "một bước" chính là **KL token-level đúng nghĩa** (tính chính xác trên từ vựng), chỉ phần "dài hạn" mới cần REINFORCE.

**(2) Teacher-mixed sampling.** Student nhỏ hay sinh câu thoái hóa (lặp cụm) mà teacher lại chấm cao. Lấy mẫu từ hỗn hợp:

$$\tilde p(y_t|y_{<t},x)=\alpha\,p(y_t|y_{<t},x)+(1-\alpha)\,q_\theta(y_t|y_{<t},x)$$

rồi sửa lệch bằng importance weight $w_t=\prod_{t'\le t}\frac{q_\theta(y_{t'}|\cdot)}{\tilde p(y_{t'}|\cdot)}$. Tích này phương sai rất cao nên xấp xỉ $w_t\approx\frac{q_\theta(y_t|y_{<t},x)}{\tilde p(y_t|y_{<t},x)}$ (chỉ giữ thừa số bước hiện tại).

**(3) Length normalization.** Chia reward-to-go cho độ dài còn lại:

$$R^{Norm}_{t+1}=\frac{1}{T-t-1}\sum_{t'=t+1}^{T}\log\frac{p(y_{t'}|y_{<t'},x)}{q_\theta(y_{t'}|y_{<t'},x)}$$

### 3.4 Gradient cuối và thuật toán

$$\nabla\mathcal{L}(\theta)=-\mathbb{E}_{y\sim\tilde p}\sum_{t=1}^{T}w_t\Big[\nabla\!\sum_{y'\in V}q_\theta(y'|y_{<t},x)\log\frac{p(y'|y_{<t},x)}{q_\theta(y'|y_{<t},x)}+R^{Norm}_{t+1}\frac{\nabla q_\theta(y_t|y_{<t},x)}{q_\theta(y_t|y_{<t},x)}\Big]$$

Thêm **clipping** kiểu PPO cho phần long-term (tỉ số $\rho_t=q_\theta/\tilde p$, cắt ở $1\pm\epsilon$, $\epsilon=0.2$) và **loss pre-training** $\mathcal{L}_{PT}=-\mathbb{E}_{d\sim D_{PT}}\log q_\theta(d)$ để giữ năng lực NLP chuẩn (kiểu PPO-ptx của InstructGPT). Pipeline giống RLHF.

```
Input: dataset D (prompt, gt response), pretrain corpus D_PT, teacher p, student q_theta0
1. SFT student on D (ground truth) -> pick checkpoint with lowest val loss  (init)
2. repeat:
     sample prompts x from D
     roll out y ~ p_tilde = alpha*p + (1-alpha)*q_theta   (token by token)
     for each step t:
         single_t = grad_theta  sum_{v in V} q(v|.) * log( p(v|.) / q(v|.) )     # exact over vocab
         long_t   = R_norm_{t+1} * grad min[ rho_t, clip(rho_t, 1-eps, 1+eps) ]    # rho_t = q(y_t|.)/p_tilde(y_t|.)
         weight   = w_t ~= q(y_t|.)/p_tilde(y_t|.)
     L_PT = LM loss on batch from D_PT
     theta <- theta - lr * ( -(sum_t weight*single_t) - (sum_t long_t) + grad L_PT )
```

(Dấu của hai hạng tử "single" và "long" theo công thức (7) và Algorithm 1 của paper; người ghi chú tóm tắt dạng gradient, không thay đổi nội dung.)

### 3.5 Ví dụ nhỏ để thấy khác biệt forward/reverse

Teacher có hai đỉnh xác suất 50/50 tại token "A" và "B", và student chỉ biểu diễn được một đỉnh (ví dụ phân phối "chỉ một lựa chọn"). Forward KL tối thiểu sẽ ép student chia xác suất, dẫn đến một phân phối "ở giữa" gán xác suất cao cho các token vô nghĩa nếu cấu trúc hạn chế. Reverse KL sẽ chọn "A" (hoặc "B") với xác suất ~1: câu trả lời đúng nhưng mất đa dạng. Đây là mode-seeking vs mass-covering (Hình 2 paper dùng Gauss).

## 4. Thí nghiệm và kết quả chính

**Tác vụ:** instruction-following. Dữ liệu train: databricks-dolly-15K (15K cặp, tách 1K validation, 0.5K test, khoảng 12.5K train). $D_{PT}$: OpenWebText (GPT-2) hoặc RoBERTa corpus (còn lại). $\alpha=0.2$.

**Mô hình:** teacher fine-tune sẵn trên $D$: GPT-2-1.5B (student GPT-2 120M/340M/760M), OPT-13B (student OPT 1.3B/2.7B/6.7B), LLaMA-13B (student LLaMA-7B); phụ lục: GPT-J-6B teacher.

**Đánh giá:** DollyEval (500), SelfInst (252), VicunaEval (80), S-NI (nhóm response ≥11 token), UnNI (10K mẫu); chỉ số Rouge-L, điểm GPT-4 (tỉ lệ so với ground truth), đánh giá người. Sinh ở temperature 1, trung bình 5 seed. Baseline: SFT w/o KD, KD (word-level), SeqKD.

### Bảng 1 (trích, R-L và GPT4)

| Teacher → student | Phương pháp | DollyEval GPT4 / R-L | SelfInst GPT4 / R-L | VicunaEval GPT4 / R-L | S-NI R-L | UnNI R-L |
|---|---|---|---|---|---|---|
| GPT-2 1.5B (teacher) | | 58.4 / 27.6 | 42.9 / 14.3 | 48.6 / 16.3 | 27.6 | 31.8 |
| GPT-2 120M | SFT w/o KD | 38.6 / 23.3 | 26.3 / 10.0 | 32.8 / 14.7 | 16.3 | 18.5 |
| | KD | 40.3 / 22.8 | 27.8 / 10.8 | 31.9 / 13.4 | 19.7 | 22.0 |
| | SeqKD | 41.2 / 22.7 | 26.2 / 10.1 | 31.0 / 14.3 | 16.4 | 18.8 |
| | **MiniLLM** | 44.7 / 24.6 | 29.2 / 13.2 | 34.1 / 16.9* | 25.3 | 26.6 |
| OPT 13B (teacher) | | 70.3 / 29.2 | 56.1 / 18.4 | 58.0 / 17.8 | 30.4 | 36.1 |
| OPT 1.3B | SFT w/o KD | 52.6 / 26.0 | 37.7 / 11.4 | 40.5 / 15.6 | 23.1 | 28.4 |
| | KD | 52.7 / 25.4 | 36.0 / 12.2 | 40.8 / 14.9 | 21.9 | 27.0 |
| | SeqKD | 51.0 / 26.1 | 36.6 / 12.7 | 42.6 / 16.6 | 21.4 | 28.2 |
| | **MiniLLM** | 60.7 / 26.7 | 47.0 / 14.8 | 50.6 / 17.9* | 28.6 | 33.4 |
| LLaMA 13B (teacher) | | 79.0 / 29.7 | 75.5 / 23.4 | 65.1 / 19.4 | 35.8 | 38.5 |
| LLaMA 7B | SFT w/o KD | 73.0 / 26.3 | 69.2 / 20.8 | 61.6 / 17.5 | 32.4 | 35.8 |
| | KD | 73.7 / 27.4 | 70.5 / 20.2 | 62.7 / 18.4 | 33.7 | 37.9 |
| | SeqKD | 73.6 / 27.5 | 71.5 / 20.8 | 62.6 / 18.1 | 33.7 | 37.6 |
| | **MiniLLM** | 76.4 / 29.0 | 73.1 / 23.2 | 64.1 / 20.7* | 35.5 | 40.2* |

(* = student vượt teacher theo paper.) MiniLLM tốt nhất ở hầu hết ô; student đôi khi vượt teacher về Rouge-L trên Vicuna, S-NI, UnNI. Tác giả đoán do teacher bị exposure bias từ teacher-forcing fine-tuning. Cải thiện nhất quán từ 120M đến 13B (Hình 1). Đánh giá người (LLaMA-7B vs teacher 13B trên SelfInst): MiniLLM được ưa hơn mọi baseline và tương đương teacher (Hình 4).

Chi phí: distill LLaMA-7B từ LLaMA-13B dưới 10 giờ trên 16 GPU V100 32GB; Phase 2 dùng lr 5e-6, batch 64, 256 câu mỗi lần thu thập, 4 inner epoch, 5000 bước, max length 512.

## 5. Phân tích / ablation

- **Teacher lớn hơn thì tốt hơn (ở đây):** với student GPT-2-125M, teacher GPT-2 340M/760M/1.5B, MiniLLM luôn thắng SeqKD và điểm student tăng theo kích thước teacher (Hình 5; không thấy capacity gap trong dải này). Kết quả OPT tương tự (Phụ lục C.2).
- **Exposure bias (Hình 6):** ExAccErr của các baseline tăng theo độ dài sinh, MiniLLM thấp hơn nhiều và ngừng tích lũy sau ~150 token.
- **Calibration (Bảng 2, LLaMA-7B, zero-shot):**

| | SST2 ECE ↓ / Acc | BoolQ ECE ↓ / Acc |
|---|---|---|
| Teacher | 0.025 / 93.0 | 0.356 / 74.5 |
| KD | 0.191 / 84.7 | 0.682 / 63.5 |
| SeqKD | 0.243 / 66.5 | 0.681 / 62.8 |
| MiniLLM | 0.099 / 89.7 | 0.502 / 67.8 |

- **Độ dài đáp án (Hình 7):** với đáp án ngắn (≤5 token) mọi phương pháp tương đương (không gian đầu ra nhỏ, student phủ được mọi mode); với đáp án dài MiniLLM hơn rõ.
- **Đa dạng (Bảng 3):** Dist-4 DollyEval 99.0 (SFT 99.5, teacher 99.3), SelfInst 98.6 (SFT 99.0, teacher 99.1); loss LM test 3.95 / 5.33 (SFT 3.89 / 5.28), tức mất ít đa dạng. Tác giả lập luận một đáp án đúng thường là đủ.
- **Ablation ba mẹo (Bảng 4, GPT-2-125M ← 1.5B; Rouge-L validation / Dolly):** đầy đủ 27.4 / 24.6; bỏ length norm 17.4 / 14.7; bỏ teacher-mixed 22.3 / 20.4; bỏ single-step 27.0 / 23.7. Bỏ length norm hoặc teacher-mixed khiến reverse KL vẫn giảm nhưng student học sinh chuỗi lặp/ngắn/vô nghĩa (reward hacking; ví dụ ở Bảng 8 phụ lục); single-step chủ yếu giảm phương sai (Hình 8).
- **Hệ số $\alpha$:** $\alpha=0.2$ chung tốt; mô hình lớn ít nhạy (Hình 16, không có số cụ thể trong text).
- **Loss pre-training $L_{PT}$ (Bảng 7):** thêm vào giữ năng lực phân loại (CLS) với 1.3B: 70.2 so với 65.7 khi bỏ; Inst. 52.8 so với 53.2 (tức không đổi nhiều); 7B: CLS 78.8 so với 74.3, Inst. 71.2 so với 71.1.

## 6. Giới hạn

**Tác giả nêu/suy ra:** reverse KL có nguy cơ mất mode, nên họ đã đo đa dạng và cho là chấp nhận được (nhưng chỉ đo trên Dist-4/loss LM, và lập luận "một đáp án đúng đủ" không đúng cho mọi ứng dụng). Cần teacher có phân phối đầu ra (white-box). Cần cùng vocabulary (suy ra từ cách tính $p/q$ trên từ vựng).

**Người ghi chú nhận xét:**
- Chi phí: phải rollout từ student (và từ hỗn hợp với teacher) mỗi bước, tức là chạy sinh tự hồi quy trong vòng lặp huấn luyện, chậm và phức tạp hơn nhiều so với KD offline.
- Thí nghiệm chỉ trên instruction-following với dữ liệu nhỏ (12.5K) và đánh giá một phần dùng GPT-4 làm giám khảo; không có reasoning/toán/code.
- Các cải thiện Rouge-L trên một số tập khá nhỏ (ví dụ 0.1–0.6) và không có khoảng tin cậy; ảnh hưởng $\alpha$, $\epsilon$ khó tái hiện chính xác.
- Mô hình student tối đa 7B, teacher tối đa 13B; chưa kiểm tra tỉ số kích thước lớn (ví dụ 40x).
- Bài sau (31 Rethinking KL, 30 DistiLLM) cho rằng ranh giới forward/reverse nhạy cảm và có thể dùng các biến thể ổn định hơn; đây là cách đọc của người ghi chú dựa trên danh sách lộ trình, cần đọc các bài đó để xác nhận.

## 7. Liên hệ với các bài khác

- **02 Hinton KD, 15 Sequence-level KD:** baseline KD và SeqKD là forward KL; MiniLLM nhắm vào điểm yếu của chúng.
- **06 Born-again networks:** nhắc đến việc student vượt teacher (Rouge-L).
- **29 GKD:** cùng thời, cũng on-policy nhưng dùng divergence tổng quát và không rút gradient policy-gradient (dùng gradient trực tiếp qua log-prob); đọc để so sánh.
- **30 DistiLLM:** skew KL, giảm chi phí sinh mẫu bằng replay/adaptive off-policy.
- **31 Rethinking KL:** phân tích lại vai trò forward/reverse KL.
- **33 TAID, 34 DistiLLM-2:** các phương pháp sau cải tiến cách nội suy giữa teacher và student.
- **26 LLaVA-KD (Bảng 5):** thử RKL/FKL/JSD cho MLLM nhỏ; RKL (55.0) không hơn FKL (55.5), cho thấy lợi thế reverse KL không phổ quát.
- **41 Capacity gap law, 39 Small models struggle...:** ở đây không thấy capacity gap trong dải GPT-2, ngược lại với một số bài sau.
- **36 Distilling step-by-step, 37 Orca, 38 DeepSeek-R1:** các hướng black-box/dữ liệu sinh, bổ sung cho white-box.

## 8. Ứng dụng cho dự án (suy luận)

Mức liên quan: **trung bình**, nhưng đáng để biết rõ vì bài toán OCR có đặc điểm thuận lợi.

- OCR là sinh văn bản có **một đáp án đúng** (đa dạng gần như không cần). Đó là kịch bản mà mode-seeking (reverse KL) lý thuyết hợp lý, và trường hợp "đáp án ngắn thì forward = reverse" (Hình 7) cũng nên cân nhắc: đoạn OCR dài, không gian đầu ra rất tập trung quanh một chuỗi đúng.
- **Exposure bias** có thể quan trọng với OCR chuỗi dài (lặp dòng, nhảy dòng khi decode); on-policy distillation cho student thấy lỗi của chính nó. Với student pruned (nhiễu loạn so với teacher), phân phối của student lệch khỏi teacher nên rollout từ student có thông tin hơn là dữ liệu cố định.
- Rủi ro: huấn luyện policy-gradient bất ổn, tốn rollout; với GPU ≤ 8 GB cho student 0.1–0.3B và teacher 4B cần nạp cả hai và sinh tự hồi quy. Cân nhắc thay bằng GKD/DistiLLM (29, 30) trước khi tự cài MiniLLM đầy đủ.
- Ba mẹo có thể mang riêng: **teacher-mixed sampling** (cho teacher dẫn dắt một phần token khi rollout để tránh suy biến), **length normalization** (OCR có độ dài rất khác nhau giữa trang), và **phần single-step = KL token-level chính xác trên từ vựng** (rẻ, không cần REINFORCE).
- Liên hệ D_lin/D_res (suy luận): sau khi cắt, phần D_lin sửa được bằng ánh xạ tuyến tính; KD token-level (forward) có thể phục hồi phần "có cấu trúc"; on-policy reverse KL nhắm vào những chuỗi mà student thực sự sinh ra, có thể xử lý lỗi tích lũy (D_res theo thời gian giải mã). Paper không đề cập pruning; đây là giả thuyết cần thử.
- Về việc cắt neuron MLP hồi phục kém hơn cắt layer: paper không có thông tin.

## 9. Thí nghiệm gợi ý

1. Với student 0.3B pruned: so sánh forward KL token-level (baseline), reverse KL token-level (chỉ phần single-step, không rollout), và rollout on-policy đầy đủ với $\alpha=0.2$; đo CER/WER và tỉ lệ lặp dòng khi decode dài.
2. Ablation riêng teacher-mixed sampling: $\alpha\in\{0,0.2,0.5\}$ trên tập tài liệu dài, đặc biệt tần suất chuỗi lặp/ngắn bất thường.
3. Đo ExAccErr (Phụ lục B.5 của paper) theo vị trí token cho student sau KD offline và sau KD on-policy.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý | Ghi chú |
|---|---|---|
| Forward KL | $\mathrm{KL}[p\|q]=\mathbb{E}_p\log\frac pq$ | mass-covering; KD/SeqKD |
| Reverse KL | $\mathrm{KL}[q\|p]=\mathbb{E}_q\log\frac qp$ | mode-seeking; MiniLLM |
| Gradient | $-\mathbb{E}_{y\sim q}\sum_t (R_t-1)\nabla\log q(y_t|y_{<t})$ | policy gradient |
| Reward-to-go | $R_t=\sum_{t'\ge t}\log\frac{p}{q}$ | thiên về câu ngắn |
| Single-step | KL token-level tính đúng trên từ vựng | giảm phương sai |
| Teacher-mixed | $\tilde p=\alpha p+(1-\alpha)q$, $\alpha=0.2$ | chặn reward hacking |
| Length norm | chia $R_{t+1}$ cho $T-t-1$ | tránh câu rỗng/ngắn |
| Loss phụ | $\mathcal{L}_{PT}$ trên $D_{PT}$ | giữ năng lực chuẩn |
| Pipeline | SFT student → on-policy (PPO-like clip $\epsilon=0.2$) | giống RLHF |

## 11. Kết luận ngắn

MiniLLM lập luận rằng khi student nhỏ không thể phủ mọi mode của teacher, hãy dạy nó **chọn đúng mode** (reverse KL), tối ưu bằng policy gradient on-policy với ba kỹ thuật ổn định. Thực nghiệm trên 120M–13B cho kết quả tốt hơn KD/SeqKD trên 5 bộ dữ liệu, với exposure bias, calibration và văn bản dài tốt hơn, đổi lại chi phí huấn luyện phức tạp hơn và nguy cơ giảm đa dạng.

Link gốc: https://arxiv.org/abs/2306.08543
