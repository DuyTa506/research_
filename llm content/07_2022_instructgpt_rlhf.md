# Training language models to follow instructions with human feedback (2022)

**Tác giả:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (OpenAI).  
**Venue / arXiv:** arXiv:2203.02155 (v1, 4/3/2022). Paper không nêu venue trong bản đã đọc.  
**Nguồn:** https://arxiv.org/abs/2203.02155  
**Chủ đề:** RLHF, alignment, supervised fine-tuning (SFT), reward model (RM), PPO, KL penalty, "alignment tax", PPO-ptx.  
**Vị trí trong lộ trình:** Đọc sau GPT-3 (02) và Kaplan/Chinchilla (03–04) vì InstructGPT chính là GPT-3 được fine-tune tiếp. Bài này thêm *bước hậu huấn luyện* biến một mô hình "đoán token tiếp theo" thành mô hình "làm theo chỉ dẫn". Phần LoRA (08) là kỹ thuật fine-tune rẻ có thể dùng thay cho full fine-tune trong các bước kiểu SFT. Hướng đi tiếp: các bài sau trong lộ trình chủ yếu là kiến trúc/suy luận, nên bài này là điểm duy nhất về hậu huấn luyện.

> Đây là ghi chú học tập bằng tiếng Việt, không phải bản dịch. **Phạm vi:** paper rất dài (~29k từ, kèm phụ lục lớn về dữ liệu, nhãn viên, đánh giá an toàn). Ghi chú này tập trung vào **phương pháp lõi (SFT → RM → PPO)** và các kết quả thông tin nhất; phần đạo đức/thiên lệch/phụ lục nhãn viên chỉ tóm tắt. Các ứng dụng cho dự án OCR là suy luận của chúng ta, không phải kết luận của paper.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
   - 3.1 [Dữ liệu và nhãn viên](#31-dữ-liệu-và-nhãn-viên)
   - 3.2 [Bước 1: SFT](#32-bước-1-sft)
   - 3.3 [Bước 2: Reward model](#33-bước-2-reward-model)
   - 3.4 [Bước 3: PPO với KL penalty và PPO-ptx](#34-bước-3-ppo-với-kl-penalty-và-ppo-ptx)
   - 3.5 [Pseudo-code toàn quy trình](#35-pseudo-code-toàn-quy-trình)
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

**Vấn đề.** GPT-3 (02) được huấn luyện để *dự đoán token tiếp theo trên văn bản internet*. Mục tiêu đó khác với mục tiêu người dùng thực sự muốn: "làm theo chỉ dẫn của tôi một cách hữu ích và an toàn". Paper gọi đây là **misaligned** (lệch mục tiêu): mô hình có thể bịa sự thật, sinh nội dung độc hại, hoặc đơn giản là không làm theo chỉ dẫn (ví dụ: bạn yêu cầu viết truyện, nó lại viết thêm nhiều yêu cầu tương tự, như ví dụ tiếng Pháp trong Figure 8 của paper).

**Hướng cũ.** (i) *Prompt engineering / few-shot*: ghép vài ví dụ vào đầu prompt để "gợi" mô hình. (ii) *Instruction tuning trên dataset NLP công khai* (FLAN, T0): fine-tune trên nhiều tác vụ NLP có kèm chỉ dẫn. (iii) *RLHF* đã được dùng cho robot/Atari (Christiano 2017) và cho tóm tắt văn bản (Ziegler 2019, Stiennon 2020). Paper này là "ứng dụng trực tiếp RLHF lên một *phân phối rộng* các tác vụ ngôn ngữ".

**Ba thuật ngữ nền tảng (giải thích cho người mới):**
- **Alignment (căn chỉnh):** làm cho mô hình hành xử đúng ý người dùng. Paper dùng bộ tiêu chí của Askell et al.: **helpful** (hữu ích), **honest** (trung thực), **harmless** (vô hại).
- **RLHF (Reinforcement Learning from Human Feedback):** dùng phán đoán của con người làm *tín hiệu thưởng* để huấn luyện bằng học tăng cường. Con người không viết đáp án chuẩn; họ chỉ *so sánh* các đáp án mà mô hình tự sinh.
- **Policy (chính sách):** trong ngữ cảnh này chính là mô hình ngôn ngữ $\pi(y\mid x)$: cho prompt $x$, nó phân phối xác suất trên câu trả lời $y$.

## 2. Ý tưởng chính

Paper làm ba bước nối tiếp, bắt đầu từ GPT-3 đã pretrain:

1. **SFT:** thuê người viết câu trả lời mẫu cho các prompt, fine-tune GPT-3 bằng học có giám sát (cross-entropy thông thường) trên các cặp (prompt, đáp án mẫu).
2. **Reward model (RM):** cho mô hình SFT sinh nhiều đáp án cho cùng một prompt, nhờ người xếp hạng; huấn luyện một mạng khác để chấm điểm vô hướng sao cho đáp án được ưa thích có điểm cao hơn.
3. **PPO:** dùng điểm RM làm phần thưởng, tối ưu chính sách (mô hình SFT) bằng thuật toán PPO, kèm *phạt KL* để chính sách không đi quá xa mô hình SFT, và (ở biến thể PPO-ptx) trộn thêm gradient của hàm mất mát pretraining để giảm "quên".

Kết quả gây chú ý nhất: mô hình 1.3B tham số sau RLHF được người đánh giá ưa thích hơn GPT-3 175B (ít hơn >100 lần tham số), trong khi kiến trúc giống hệt nhau.

## 3. Phương pháp chi tiết

### 3.1 Dữ liệu và nhãn viên

- **Prompt:** chủ yếu là prompt do khách hàng gửi qua OpenAI API Playground tới các bản InstructGPT sớm (đã được SFT trên một phần dữ liệu demo); thêm prompt do nhãn viên tự viết để "mồi" ban đầu (ba kiểu: *plain*, *few-shot*, *user-based*). Loại trùng lặp theo tiền tố chung, giới hạn 200 prompt/người dùng, tách train/valid/test *theo user ID*, lọc thông tin cá nhân (PII). Dữ liệu >96% là tiếng Anh.
- **Phân bố loại tác vụ (Table 1, trên API prompt dataset):** Generation 45.6%, Open QA 12.4%, Brainstorming 11.2%, Chat 8.4%, Rewrite 6.6%, Summarization 4.2%, Classification 3.5%, Other 3.5%, Closed QA 2.6%, Extract 1.9%.
- **Kích thước tập dữ liệu (Table 6, số prompt):**

| Tập | Train (labeler + customer) | Valid |
|---|---|---|
| SFT | 11,295 + 1,430 (~13k) | 1,550 + 103 |
| RM | 6,623 + 26,584 (~33k) | 3,488 + 14,399 |
| PPO (không có nhãn người) | 31,144 (chỉ customer) | 16,185 |

- **Nhãn viên:** ~40 contractor (Upwork, ScaleAI), chọn qua bài kiểm tra sàng lọc về độ nhạy với các nhóm nhân khẩu và khả năng nhận diện nội dung có hại. **Độ đồng thuận giữa nhãn viên:** 72.6 ± 1.5% (nhãn viên train), 77.3 ± 1.3% (nhãn viên held-out); so sánh: 73 ± 4% giữa researcher trong Stiennon et al. 2020.
- Khi huấn luyện, ưu tiên *hữu ích*; khi đánh giá cuối, nhãn viên được yêu cầu ưu tiên *trung thực và vô hại*.

### 3.2 Bước 1: SFT

Fine-tune GPT-3 (3 cỡ: 1.3B, 6B, 175B) trên demonstration của nhãn viên bằng mục tiêu ngôn ngữ chuẩn.

- 16 epoch, cosine LR decay xuống 10%, residual dropout 0.2, không warmup.
- LR: 9.65e-6, batch 32 (1.3B và 6B); 5.03e-6, batch 8 (175B).
- Điểm đáng chú ý: mô hình SFT **overfit validation loss sau 1 epoch**, nhưng huấn luyện nhiều epoch hơn vẫn **tăng điểm RM và điểm ưa thích của người**. Vì vậy, họ chọn checkpoint theo *điểm RM trên tập valid* (paper nói chỉ số này dự báo ưa thích của người tốt hơn validation loss), không theo loss.

### 3.3 Bước 2: Reward model

**Kiến trúc.** Lấy mô hình SFT, **bỏ lớp unembedding cuối** (lớp ánh xạ ra từ vựng) và thay bằng đầu ra vô hướng: $r_\theta(x,y)\in\mathbb{R}$. Chỉ dùng RM cỡ **6B** cho mọi cỡ policy (RM 175B huấn luyện không ổn định, và tốn kém khi dùng làm value function).

**Dữ liệu so sánh.** Với mỗi prompt, người xem $K\in[4,9]$ đáp án và xếp hạng. Từ đó có $\binom{K}{2}$ cặp so sánh. Ví dụ $K=4\Rightarrow 6$ cặp; $K=9\Rightarrow 36$ cặp.

**Hàm mất mát (Eq. 1):**

$$
\boxed{\;\text{loss}(\theta)=-\frac{1}{\binom{K}{2}}\;\mathbb{E}_{(x,y_w,y_l)\sim D}\Big[\log\sigma\big(r_\theta(x,y_w)-r_\theta(x,y_l)\big)\Big]\;}
$$

trong đó $y_w$ là đáp án được ưa thích ("winner"), $y_l$ là đáp án kém hơn ("loser"), $\sigma$ là sigmoid. Ý nghĩa: hiệu điểm $r_w-r_l$ được coi là *log-odds* rằng người chọn $y_w$ (mô hình Bradley–Terry), và loss là cross-entropy với nhãn "người chọn $y_w$".

**Ví dụ số nhỏ (tự tính):**

| $r_w-r_l$ | $\sigma(\cdot)$ | loss $=-\log\sigma$ |
|---|---|---|
| +1.0 | 0.731 | 0.313 |
| 0.0 | 0.500 | 0.693 |
| −1.0 | 0.269 | 1.313 |

Nghĩa là RM bị phạt nặng khi xếp ngược thứ tự người chọn, và loss giảm khi khoảng cách điểm đúng chiều tăng.

**Mẹo kỹ thuật quan trọng.** Nếu xáo trộn mọi cặp so sánh thành một dataset phẳng, các cặp cùng prompt rất tương quan, nên chỉ một epoch đã overfit (mỗi đáp án xuất hiện trong $K-1$ cập nhật gradient). Thay vào đó, họ đặt **toàn bộ $\binom{K}{2}$ cặp của một prompt vào cùng một batch element**. Lợi ích: chỉ cần **một lần forward cho mỗi đáp án** ($K$ lần) thay vì $2\binom{K}{2}$ lần, và validation accuracy/log-loss tốt hơn hẳn.

**Huấn luyện RM:** 1 epoch, LR 9e-6 cosine (xuống 10%), batch 64 *prompt* (tức tối đa $64\times 36=2{,}304$ cặp). RM khá không nhạy với LR (đổi 50% cho kết quả tương tự) nhưng rất nhạy với số epoch (nhiều epoch → overfit nhanh). Khởi tạo từ GPT-3 6B đã fine-tune trên nhiều bộ NLP công khai (vì lý do lịch sử; khởi tạo từ GPT-3 hay SFT cho kết quả tương tự). Cuối cùng RM được cộng một bias sao cho đáp án demo của nhãn viên có điểm trung bình 0 trước khi chạy RL (vì loss bất biến theo phép tịnh tiến điểm).

### 3.4 Bước 3: PPO với KL penalty và PPO-ptx

**Môi trường.** Là một *bandit*: lấy một prompt ngẫu nhiên, policy sinh cả câu trả lời, RM cho điểm, episode kết thúc. Không có phần thưởng trung gian. (Tuy vậy, phạt KL được tính *theo từng token*.)

**PPO (Proximal Policy Optimization):** thuật toán policy-gradient cắt tỉ số xác suất (clip) để mỗi lần cập nhật không làm policy thay đổi quá nhiều. Value function (ước lượng phần thưởng kỳ vọng, dùng để giảm phương sai) khởi tạo từ chính RM 6B.

**Hàm mục tiêu (Eq. 2):**

$$
\boxed{\;
\text{objective}(\phi)=\mathbb{E}_{(x,y)\sim D_{\pi_\phi^{RL}}}\!\Big[r_\theta(x,y)-\beta\log\frac{\pi^{RL}_\phi(y\mid x)}{\pi^{SFT}(y\mid x)}\Big]+\gamma\,\mathbb{E}_{x\sim D_{pretrain}}\!\big[\log\pi^{RL}_\phi(x)\big]\;}
$$

Giải nghĩa từng phần:
- $r_\theta(x,y)$: điểm RM, cái ta muốn **tối đa hoá**.
- $-\beta\log\frac{\pi^{RL}}{\pi^{SFT}}$: **phạt KL**, tức một ước lượng lấy mẫu của $\beta\cdot\mathrm{KL}(\pi^{RL}\Vert\pi^{SFT})$ lấy dấu âm. Nó giữ policy gần mô hình SFT để không "khai thác" lỗ hổng của RM (**reward over-optimization**: RM chỉ là xấp xỉ sở thích người; tối ưu quá mạnh sẽ cho đáp án điểm cao nhưng thực chất tệ).
- $\gamma\,\mathbb{E}[\log\pi^{RL}(x)]$: gradient *pretraining* trộn vào, để giữ khả năng NLP chung. $\gamma=0$ là "PPO" thuần; $\gamma>0$ là **PPO-ptx**. "InstructGPT" trong paper mặc định chỉ PPO-ptx.

**Siêu tham số (Phụ lục C.3–C.4):**
- Khởi tạo policy: GPT-3 + SFT 2 epoch trên demo, có trộn 10% dữ liệu pretraining. Cũng dùng chính mô hình này tính KL.
- $\beta=0.02$; $\gamma=27.8$; dữ liệu pretraining dùng gấp **8 lần** số episode RL.
- 256k episode (~31k prompt duy nhất); batch mỗi vòng 512, minibatch 64 (tức 8 minibatch, 1 inner epoch); clip ratio 0.2; nhiệt độ lấy mẫu rollout 1; không chiết khấu khi tính GAE.
- LR không đổi, warmup 10 vòng đầu (bắt đầu 1/10 LR đỉnh); EMA trọng số với decay 0.992; LR value function 9e-6 (policy 1.3B, 6B), 5e-6 (175B).
- Chi phí tính toán (paper): SFT 175B ≈ 4.9 petaflops/s-days; PPO-ptx 175B ≈ 60 petaflops/s-days; so với GPT-3 pretrain 3,640 petaflops/s-days.

### 3.5 Pseudo-code toàn quy trình

```text
# Đầu vào: GPT-3 pretrain, tập prompt P, nhãn viên H
# ---- Bước 1: SFT ----
D_demo  = { (x, y*) : x ~ P, y* = H.viết_đáp_án(x) }          # ~13k prompt
pi_sft  = finetune(GPT3, D_demo, loss = -log p(y*|x), epochs=16)
          # chọn checkpoint theo điểm RM trên valid, không theo valid loss

# ---- Bước 2: Reward model ----
for x in P_rm:                                                  # ~33k prompt
    ys = [pi_sft.sample(x) for _ in range(K)]                   # K in [4, 9]
    rank = H.xếp_hạng(x, ys)
RM = pi_sft_without_unembedding + linear_head → scalar          # 6B
for batch of 64 prompts:                                        # 1 epoch
    r = [RM(x, y) for y in ys]                                  # K forward / prompt
    loss = -mean over C(K,2) pairs (w,l) of  log sigmoid(r[w]-r[l])
    update RM
RM.bias -= mean(RM(x, demo y*))                                 # chuẩn hoá: demo có điểm TB = 0

# ---- Bước 3: PPO(-ptx) ----
pi = copy(pi_sft_2epoch_with_10%_pretrain); V = copy(RM)        # value init từ RM
repeat until 256k episodes:
    X = sample 512 prompts from P_ppo (không nhãn người)
    Y = pi.sample(X)                                            # rollout, T = 1
    reward_t = [per-token  -beta * (log pi(y_t) - log pi_sft(y_t))]
    reward_T += RM(X, Y)                                        # điểm RM cộng ở token cuối
    A = GAE(reward, V)                                          # lambda, gamma_disc = 1
    for 8 minibatch(64):
        g_ppo = grad PPO_clip_loss(pi, A, clip=0.2)
        g_ptx = gamma * grad( -log pi(x_pretrain) )             # x_pretrain: 8x số episode
        update pi with g_ppo + g_ptx;  update V
```

(Việc cộng phạt KL theo token và điểm RM ở token cuối là cách triển khai chuẩn của Stiennon et al. mà paper nói "theo cùng cách"; paper diễn đạt: "per-token KL penalty from the SFT model at each token".)

## 4. Thí nghiệm và kết quả chính

**Mô hình & baseline.** Ba cỡ 1.3B / 6B / 175B (kiến trúc GPT-3). So sánh: GPT-3, GPT-3 + prompt few-shot ("prompted"), SFT, PPO, PPO-ptx, và GPT-3 175B fine-tune trên FLAN/T0 (~1 triệu mẫu mỗi bộ). Baseline để tính tỉ lệ thắng: SFT 175B.

**Đánh giá chính (ưa thích của người trên prompt API held-out):**

| Phát hiện (paper) | Số liệu |
|---|---|
| 175B InstructGPT ưa thích hơn 175B GPT-3 | 85 ± 3% |
| 175B InstructGPT ưa thích hơn GPT-3 few-shot | 71 ± 4% |
| 1.3B PPO-ptx so với 175B GPT-3 | 1.3B được ưa thích hơn (Figure 1) |
| 175B InstructGPT ưa thích hơn FLAN / T0 | 78 ± 4% / 79 ± 4% |
| Win rate (so với SFT baseline) của InstructGPT vs T0, FLAN (Mục 1) | 73.4 ± 2% vs 26.8 ± 2% (T0) và 29.8 ± 2% (FLAN) |
| Hallucination trên tác vụ closed-domain (API) | 21% (InstructGPT) vs 41% (GPT-3) |
| Độc hại (RealToxicityPrompts, khi được nhắc "respectful") | ít hơn ~25% so với GPT-3 |
| TruthfulQA | trung thực và có thông tin ~gấp đôi GPT-3 (paper); riêng 1.3B PPO-ptx kém nhẹ hơn GPT-3 cùng cỡ |

Các điểm khác:
- Thứ tự cải thiện nhìn thấy trong Figure 1: GPT-3 < GPT-3 prompted < SFT < PPO (≈ PPO-ptx). Thêm gradient pretraining không làm đổi điểm ưa thích nhiều.
- **Tổng quát hoá sang nhãn viên khác:** nhãn viên held-out (không tạo dữ liệu huấn luyện) cũng ưa InstructGPT với tỉ lệ tương tự. RM chạy 5-fold theo nhóm nhãn viên đạt **69.6 ± 0.9%** trên nhóm bị giữ lại so với **72.4 ± 0.4%** trên nhóm đã thấy.
- **Tổng quát hoá ngoài phân phối:** làm theo chỉ dẫn tiếng khác và hỏi-đáp/tóm tắt code, dù dữ liệu rất hiếm (định tính, không có số).
- **Bias:** không cải thiện đáng kể trên Winogender và CrowS-Pairs.
- **Chi phí:** RLHF rẻ so với pretrain (60 so với 3,640 petaflops/s-days cho 175B); paper kết luận tăng cường alignment "hiệu quả chi phí hơn" việc làm mô hình lớn hơn 100 lần, *trên phân phối khách hàng của họ*.

## 5. Phân tích / ablation

- **Alignment tax và PPO-ptx.** PPO thuần gây suy giảm trên SQuADv2, DROP, HellaSwag, dịch Pháp→Anh WMT15. Trộn gradient pretraining (PPO-ptx) khắc phục phần lớn, thậm chí vượt GPT-3 trên HellaSwag; vẫn còn kém GPT-3 trên DROP, SQuADv2 và dịch.
- **Pretraining mix tốt hơn tăng KL.** Tăng hệ số KL làm giảm mạnh reward valid và không bao giờ phục hồi hoàn toàn DROP/SQuAD; ngược lại tồn tại một giá trị $\gamma$ vừa đảo ngược suy giảm vừa chỉ giảm reward rất ít.
- **Lượng dữ liệu pretraining (E.11):** tỉ lệ 4 thì log-prob pretraining thường tăng (tức xấu đi) trong quá trình học; 32 cho Likert tốt hơn nhưng thời gian huấn luyện tăng "vài lần"; chọn 8 (gấp đôi thời gian so với không trộn).
- **Hạn chế số episode:** với 1.3B, huấn luyện >256k episode không giúp (khi có pretraining mix).
- **Batch size PPO** (thử 64–1024 trên 1.3B): 512 tốt nhất theo đánh giá của người; minibatch thử 8, 16, 32, 64.
- **Kích thước RM:** RM 6B đủ ổn định và cho PPO mạnh tương đương; RM 175B có thể đạt loss thấp hơn nhưng không ổn định.
- **FLAN/T0 kém hơn:** lý do paper đưa ra: (1) tác vụ NLP công khai chủ yếu là phân loại/QA (~18% ứng dụng thật), trong khi sinh tự do + brainstorm chiếm ~57%; (2) khó đạt độ đa dạng input như người dùng thật.
- **Hành vi sai còn lại (định tính):** chấp nhận tiền đề sai, rào đón quá mức (paper *nghi ngờ* do nhãn viên được dặn thưởng "khiêm tốn tri thức", RM học theo), giảm chất lượng khi có nhiều ràng buộc tường minh.

## 6. Giới hạn

**Do tác giả nêu:**
- Hành vi bị quyết định bởi ~40 nhãn viên chủ yếu nói tiếng Anh ở Mỹ/Đông Nam Á, không đại diện cho mọi người dùng; hầu hết so sánh chỉ được 1 người gán nhãn (vì chi phí).
- Mô hình chưa hoàn toàn an toàn/căn chỉnh; vẫn bịa sự thật, sinh nội dung độc hại/tình dục/bạo lực khi không bị nhắc; **nghiêm trọng nhất**: nó làm theo chỉ dẫn kể cả khi có hại, và khi được yêu cầu "thiên kiến tối đa" thì độc hơn GPT-3 cùng cỡ.
- Alignment chỉ nhắm tới sở thích của một nhóm cụ thể (nhãn viên, nhà nghiên cứu, khách hàng API), không phải "giá trị con người" nói chung.
- Pretraining mix không loại bỏ hết regression và có thể làm tăng hành vi xấu có sẵn trong dữ liệu pretrain.

**Do chúng ta nhận xét (suy luận):**
- Phần thưởng là một proxy học được (RM chỉ ~70% đồng thuận với người), nên mọi cải thiện đo bằng chính nhãn viên có nguy cơ đo "độ hợp gu nhãn viên" hơn là chất lượng thực.
- Chỉ tiết lộ một thiết lập (GPT-3, API prompt); không có ablation riêng cho việc bỏ SFT hoặc đổi thuật toán RL, nên không biết PPO có thực sự cần thiết (paper tự nhắc các phương pháp đơn giản hơn như expert iteration, behavior cloning trên một phần dữ liệu so sánh là hướng chưa thử).
- PPO cần chạy đồng thời nhiều mô hình (policy, SFT tham chiếu, RM, value), bộ nhớ lớn; rất xa điều kiện GPU ≤ 8 GB.

## 7. Liên hệ với các bài khác

- **02 GPT-3:** mô hình khởi đầu; InstructGPT dùng đúng kiến trúc GPT-3, chỉ khác dữ liệu/huấn luyện hậu kỳ. So sánh "1.3B vs 175B" của paper này là một ví dụ rằng *dữ liệu/mục tiêu đúng có thể quan trọng hơn quy mô* (đối chiếu với 03–04 về scaling laws, vốn nói về loss ngôn ngữ chứ không phải sở thích).
- **05 LLaMA, 06 Mistral:** các mô hình nền mở; bản "chat/instruct" của chúng được tạo bằng quy trình cùng họ với bài này (hiểu biết ngoài paper).
- **08 LoRA:** có thể thay full fine-tune trong bước SFT để rẻ hơn rất nhiều.
- **48–50 Speculative decoding/Medusa/EAGLE:** RLHF thay đổi phân phối đầu ra của mô hình; draft model cần khớp với mô hình đã căn chỉnh (suy luận).
- **Distillation (chủ đề dự án):** phạt KL tới mô hình tham chiếu về bản chất giống một số hạng ép phân phối gần một "teacher" (xem phần 8).

## 8. Ứng dụng cho dự án (suy luận)

**Mức độ liên quan trực tiếp: thấp.** LightOnOCR là bài toán ảnh→văn bản có đáp án chuẩn, không phải sinh văn bản mở nhiều đáp án đúng; RLHF với nhãn viên so sánh ít phù hợp. Tuy vậy có vài bài học gián tiếp:

1. **Số hạng KL tới mô hình tham chiếu ≈ regularizer kiểu distillation.** Trong khôi phục sau pruning, ta cũng có hai mô hình: student đã cắt và teacher (4B). Hàm mục tiêu "tối đa hoá tín hiệu nhiệm vụ − $\beta\cdot$KL(student‖reference)" gợi ý dạng loss hỗn hợp: loss nhiệm vụ OCR + KL tới teacher/mô hình gốc. Cẩn thận hướng KL: paper dùng KL(policy‖SFT) ước lượng bằng mẫu từ policy; distillation chuẩn thường dùng KL(teacher‖student), hai hướng này khác nhau về tính chất (mode-seeking vs mean-seeking).
2. **"Alignment tax" ≈ "recovery tax".** Sau khi cắt tỉa rồi tinh chỉnh lên OCR, mô hình có thể mất khả năng ngoài miền (ví dụ ngôn ngữ/bố cục hiếm). Mẹo PPO-ptx (trộn gradient của phân phối gốc với hệ số lớn, dữ liệu gốc nhiều gấp 8 lần) tương ứng với việc **trộn loss/dữ liệu của tác vụ gốc hoặc loss distillation lên dữ liệu pretraining-like** trong lúc phục hồi. Paper cho bằng chứng rằng trộn hiệu quả hơn chỉ tăng hệ số KL.
3. **Chọn checkpoint theo chỉ số nhiệm vụ, không theo loss.** SFT overfit valid loss sau 1 epoch nhưng điểm RM/ưa thích vẫn tăng đến 16 epoch. Với OCR: theo dõi *edit distance / CER / WER* trên tập valid thay vì chỉ token-level cross-entropy khi quyết định dừng.
4. **Hallucination trên closed-domain** (bịa thông tin không có trong input: 41% → 21%) chính là bệnh của OCR VLM (bịa chữ không có trong ảnh). RLHF không phải công cụ đã được chứng minh cho việc này ở bài toán của bạn, nhưng một phần thưởng dựa trên edit distance so với ground truth (kiểu RL với phần thưởng kiểm chứng được) là hướng tự nhiên; đó là suy luận, paper không làm.
5. **Về triển khai ≤ 8 GB batch-1:** RLHF là pha *huấn luyện*; không ảnh hưởng chi phí suy luận. Mô hình sau RLHF có cùng kiến trúc nên không thay đổi KV cache hay FLOPs.

## 9. Thí nghiệm gợi ý

1. **Loss hỗn hợp phục hồi:** huấn luyện student sau pruning với $L=L_{OCR}+\lambda\,\mathrm{KL}(p_{teacher}\Vert p_{student})+\gamma\,L_{gen}$, trong đó $L_{gen}$ là LM loss trên văn bản tổng quát/tài liệu đa ngôn ngữ (vai trò của pretraining mix). So sánh $\gamma=0$ và $\gamma>0$ về CER trên miền hiếm.
2. **Early-stopping theo CER:** chạy nhiều epoch, vẽ cả valid cross-entropy và CER; kiểm tra hiện tượng "loss tăng nhưng CER vẫn giảm" như SFT trong paper.
3. **Mini pairwise reward (nếu muốn thử RM):** tạo cặp (đầu ra tốt, đầu ra tệ) tự động từ OCR có/không lỗi, huấn luyện RM nhỏ bằng loss $-\log\sigma(r_w-r_l)$, xem nó có xếp hạng đúng theo CER hay không (kiểm chứng RM làm proxy, chưa cần chạy PPO).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| SFT | Fine-tune có giám sát trên demo của người; 16 epoch, chọn checkpoint theo điểm RM |
| RM | GPT-3 6B bỏ unembedding, đầu ra vô hướng $r_\theta(x,y)$ |
| Loss RM | $-\frac{1}{\binom K2}\mathbb E[\log\sigma(r_\theta(x,y_w)-r_\theta(x,y_l))]$; mọi cặp của một prompt cùng một batch element |
| PPO objective | $r_\theta-\beta\log\frac{\pi^{RL}}{\pi^{SFT}}+\gamma\,\mathbb E_{pretrain}\log\pi^{RL}$ |
| PPO vs PPO-ptx | $\gamma=0$ vs $\gamma=27.8$ (dữ liệu pretrain gấp 8 lần số episode), $\beta=0.02$ |
| Alignment tax | Mất điểm trên benchmark NLP công khai sau RLHF; PPO-ptx giảm phần lớn |
| Dữ liệu | ~13k SFT, ~33k RM (K=4–9), ~31k PPO prompt |
| Chi phí 175B | SFT 4.9; PPO-ptx 60; GPT-3 3,640 (petaflops/s-days) |
| Kết quả chốt | 1.3B InstructGPT được ưa hơn 175B GPT-3; 85±3% thắng GPT-3 175B; hallucination 21% vs 41% |

## 11. Kết luận ngắn

Bài báo chứng minh một công thức ba bước (SFT → RM từ so sánh cặp → PPO có phạt KL, kèm trộn gradient pretraining) có thể biến một mô hình ngôn ngữ thô thành mô hình làm theo chỉ dẫn tốt hơn nhiều, thậm chí hơn cả việc tăng 100 lần số tham số, với chi phí nhỏ so với pretraining. Hạn chế nằm ở chỗ nó chỉ căn chỉnh theo sở thích của một nhóm nhãn viên nhỏ và vẫn còn lỗi. Đối với dự án OCR, giá trị chính là các ý tưởng phụ trợ (regularizer kiểu KL, trộn dữ liệu gốc để chống quên, chọn checkpoint theo chỉ số nhiệm vụ), không phải bản thân RLHF.

**Link gốc:** https://arxiv.org/abs/2203.02155
