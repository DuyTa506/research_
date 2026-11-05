# A Survey on Knowledge Distillation of Large Language Models (2024)

**Tác giả:** Xiaohan Xu, Ming Li, Chongyang Tao, Tao Shen, Reynold Cheng, Jinyang Li, Can Xu, Dacheng Tao, Tianyi Zhou (HKU, Univ. of Maryland, Microsoft, UTS, Peking Univ., Univ. of Sydney).  
**Venue / arXiv:** arXiv:2402.13116 (v4, 21/10/2024). Paper không nêu venue trong phần text đã trích.  
**Nguồn:** https://arxiv.org/abs/2402.13116  
**Chủ đề:** Survey KD cho LLM theo ba trụ: **algorithm** (cách rút tri thức từ teacher và cách nạp vào student), **skill** (kỹ năng được chưng cất), **verticalization** (ứng dụng theo miền). Nhấn mạnh data augmentation (DA) là cơ chế trung tâm của KD thời LLM. Ghi chú này tập trung vào phần taxonomy và thuật toán (Mục 2 và 3); các mục 4 (skills) và 5 (vertical domains) chỉ lướt, dùng như danh mục tra cứu.  
**Vị trí trong lộ trình:** Đọc sau 05 (KD survey cổ điển) và 15 (SeqKD), trước nhóm 28–40. Nó đóng vai bản đồ: chỉ ra nhóm "black-box KD bằng dữ liệu sinh" (35 Self-Instruct, 36 Distilling step-by-step, 37 Orca, 38 DeepSeek-R1 distillation) và nhóm "white-box KD bằng phân phối/feature" (28 MiniLLM, 29 GKD, 30 DistiLLM...). Với dự án OCR-VLM của chúng ta, white-box là nhóm chính; survey này khá mỏng ở phần đó.

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch nguyên văn paper. Survey không có thí nghiệm riêng, nên mục "Thí nghiệm và kết quả" chỉ tóm các luận điểm có trích dẫn trong survey, không có số liệu đo mới. Ánh xạ sang các bài trong lộ trình và phần "Ứng dụng cho dự án" là suy luận của người ghi chú.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết: taxonomy và các thuật toán](#3-phương-pháp-chi-tiết-taxonomy-và-các-thuật-toán)
4. [Những gì survey báo cáo (thay cho thí nghiệm)](#4-những-gì-survey-báo-cáo-thay-cho-thí-nghiệm)
5. [Phân tích / các luận điểm đáng chú ý](#5-phân-tích--các-luận-điểm-đáng-chú-ý)
6. [Giới hạn](#6-giới-hạn)
7. [Liên hệ với các bài khác](#7-liên-hệ-với-các-bài-khác)
8. [Ứng dụng cho dự án (suy luận)](#8-ứng-dụng-cho-dự-án-suy-luận)
9. [Thí nghiệm gợi ý](#9-thí-nghiệm-gợi-ý)
10. [Tóm tắt công thức / khái niệm](#10-tóm-tắt-công-thức--khái-niệm)
11. [Kết luận ngắn](#11-kết-luận-ngắn)

---

## 1. Bối cảnh

Trước LLM, KD (Hinton, bài 02; DistilBERT, bài 16; TinyBERT, bài 18...) chủ yếu là nén mô hình: teacher lớn → student nhỏ trên cùng tác vụ, để triển khai trên thiết bị yếu. Khi có GPT-4, Gemini, Claude (đóng, API) và LLaMA, Mistral (mở), KD có thêm hai vai trò (Hình 1 của survey):

1. **Advance (nâng khả năng):** chuyển năng lực từ LLM "proprietary" (đóng, hoặc mở nhưng rất lớn như LLaMA-2-70B) sang LLM mở nhỏ hơn. Vai trò chính, thường dựa vào dữ liệu do teacher sinh.
2. **Compress (nén):** vai trò truyền thống, hiệu quả triển khai.
3. **Self-improve (tự cải thiện):** dùng chính mô hình mở làm teacher cho nó.

Khái niệm nền:
- **Black-box KD:** chỉ có văn bản teacher sinh (qua API). **White-box KD:** có logit/phân phối/hidden state.
- **Data augmentation (DA)** trong ngữ cảnh LLM: không chỉ paraphrase/back-translation, mà là *sinh dữ liệu mới theo kỹ năng/miền* từ một ít hạt giống (seed). Survey coi DA là cách thực thi chính của KD thời LLM.
- **Instruction tuning / SFT:** tinh chỉnh có giám sát trên cặp (instruction, response).
- **RLAIF, DPO:** học từ phản hồi/preference do AI (teacher) tạo, thay vì từ người; DPO tối ưu preference trực tiếp không cần reward model.

## 2. Ý tưởng chính

Survey tách KD của LLM thành hai bước độc lập có thể ghép tự do:
(i) **Knowledge elicitation** – dùng teacher để tạo dữ liệu tri thức $D^{(kd)}$ (nhãn, mở rộng, curation, feature, feedback, self-knowledge);
(ii) **Distillation algorithm** – cách học từ dữ liệu đó (SFT, divergence/similarity, RL, ranking optimization).
Rồi kết quả của hai bước này phục vụ *skill distillation* (instruction following, alignment, agent, NLP task, multimodal) và *vertical distillation* (luật, y tế, tài chính, khoa học). Bản đồ này giúp trả lời nhanh: "một bài mới thuộc ô nào, khác bài cũ ở đâu".

## 3. Phương pháp chi tiết: taxonomy và các thuật toán

### 3.1 Pipeline chung (Mục 2.4, bốn bước)

1. **Steering:** dùng instruction/template $I$ để hướng teacher vào kỹ năng/miền.
2. **Seed knowledge:** đưa dữ liệu hạt giống $s$ (một ít ví dụ, chủ đề, tài liệu...).
3. **Generation:** teacher sinh các ví dụ tri thức (thường là QA/giải thích; đôi khi logit/hidden).
4. **Training:** huấn luyện student với hàm mục tiêu.

Hai công thức trừu tượng (Eq. 1 và 2 của survey):

$$\mathcal{D}^{(kd)}_I=\{\mathrm{Parse}(o,s)\mid o\sim p_T(o\mid I\oplus s),\ \forall s\sim\mathcal{S}\}$$

$$\mathcal{L}=\sum_I\mathcal{L}_I\big(\mathcal{D}^{(kd)}_I;\theta_S\big)$$

($\oplus$ là ghép văn bản; có thể nhiều kỹ năng $I$ cùng chưng cất vào một student.)

### 3.2 Sáu kiểu "knowledge elicitation" (Mục 3.1)

| Kiểu | Ý chính | Công thức (survey) | Ví dụ |
|---|---|---|---|
| **Labeling** | teacher gắn nhãn $y$ cho input $x$ có sẵn | $\{x,y\mid x\sim\mathcal{X},\ y\sim p_T(y\mid I\oplus c\oplus x)\}$ (Eq. 3) | Orca (FLAN-v2 + ChatGPT/GPT-4), CoT-Distill, Vicuna |
| **Expansion** | teacher sinh cả $x$ và $y$ từ vài demo (in-context) | $\{(x,y)\mid x\sim p_T(x\mid I\oplus c),\ y\sim p_T(y\mid I\oplus x)\}$ (Eq. 4) | Self-Instruct, Alpaca, WizardLM (Evol-Instruct) |
| **Data curation** | sinh từ đầu theo *meta-information* (chủ đề, kiến thức) để kiểm soát chất lượng/đa dạng | $\{(x,y)\mid x\sim p_T(x\mid I\oplus m),\ y\sim p_T(y\mid I\oplus x)\}$ (Eq. 5) | UltraChat, phi-1 ("textbook quality"), Magicoder |
| **Feature** | lấy phân phối/hidden state của teacher (white-box) | $\{(x,y,\phi_{feat}(x,y;\theta_T))\}$ (Eq. 6) | MiniLLM, GKD, TED, FuseLLM, quantization-aware KD |
| **Feedback** | teacher đánh giá/sửa/ranking *đầu ra của student* | $\{(x,y,\phi_{fb}(x,y;\theta_T))\mid y\sim p_S(y|x)\}$ (Eq. 7) | RLAIF, UltraFeedback, Lion, PERsD, SelFee |
| **Self-knowledge** | student tự làm teacher: sinh, lọc, chấm điểm lại | $\{(x,y,\phi_{sk}(x,y))\mid y\sim p_S(y\mid I\oplus x)\}$ (Eq. 8) | Self-Instruct, Self-Align, ReST, Self-Rewarding |

Điểm cốt lõi về trade-off (trích ý của survey): Labeling bị giới hạn bởi dữ liệu đầu vào và có thể nhiễu; Expansion dễ đồng nhất và khuếch đại bias của seed; Curation kiểm soát tốt nhưng phụ thuộc thiết kế meta-information; Feature chỉ dùng được với teacher mở, và student từ teacher mở có thể yếu hơn student từ teacher đóng mạnh hơn (như GPT-4) (phát biểu của survey, không phải kết quả đo).

### 3.3 Bốn họ thuật toán "distillation" (Mục 3.2)

**(a) SFT / sequence-level KD** (Eq. 9): student cực đại hóa log-likelihood trên chuỗi do teacher sinh:

$$\mathcal{L}_{SFT}=\mathbb{E}_{x\sim\mathcal{X},\,y\sim p_T(y|x)}\big[-\log p_S(y|x)\big]$$

Đơn giản nhưng là nền của Alpaca, Vicuna, WizardLM, Orca... (liên hệ bài 15).

**(b) Divergence và Similarity** (white-box):

$$\mathcal{L}_{Div}=\mathbb{E}_{x\sim\mathcal{X},y\sim\mathcal{Y}}\,D\big(p_T(y|x),\,p_S(y|x)\big)$$

với Bảng 1 của survey: forward KL $\sum p\log\frac pq$ (mode-covering, dùng trong KD chuẩn như DistilBERT), reverse KL $\sum q\log\frac qp$ (mode-seeking, MiniLLM), JS divergence. Survey nhắc lại lập luận của MiniLLM và GKD: forward KL buộc student phủ mọi mode nên khi student quá yếu có thể gán xác suất cho token thấp xác suất dưới teacher, gây ảo giác/chất lượng thấp; reverse KL tránh điều đó nhưng giảm đa dạng; theo GKD và Sason & Verdú, divergence tối ưu **phụ thuộc tác vụ**: forward KL hợp với dịch máy (ít mode), reverse KL hợp với hội thoại/instruction tuning (nhiều mode). Phía **Similarity** (Eq. 11):

$$\mathcal{L}_{Sim}=\mathbb{E}\,\mathcal{L}_F\big(\Phi_T(f_T(x,y)),\,\Phi_S(f_S(x,y))\big)$$

với $f$ là feature map, $\Phi$ là hàm biến đổi cho cùng kích thước, $\mathcal{L}_F$ là L2/L1/cross-entropy/MMD (Bảng 2). Survey nói rất ít công trình dùng similarity cho LLM (ví dụ TED: task-aware layer-wise distillation), dù phổ biến ở encoder nhỏ (liên hệ 03 FitNets, 18 TinyBERT, 19 MiniLM).

**(c) Reinforcement learning:** hai bước (1) huấn luyện reward model student từ preference của teacher:

$$\mathcal{L}_{RM}=-\mathbb{E}_{(x,y_w,y_l)}\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)$$

(2) tối ưu policy student với ràng buộc KL tới policy tham chiếu:

$$\max_{\pi_\theta}\ \mathbb{E}_{x,\,y\sim\pi_\theta}[r_\phi(x,y)]-\beta\,\mathrm{D_{KL}}[\pi_\theta(y|x)\,\|\,\pi_{ref}(y|x)]$$

(thường PPO). Có thể dùng chính teacher làm reward (đắt hơn).

**(d) Ranking optimization:** thay RL bằng học từ preference cố định: **DPO** (Zephyr dùng DPO để chưng cất preference của GPT-4):

$$\mathbb{E}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big)\Big]$$

**RRHF:** $\sum_{r_i<r_j}\max(0,p_i-p_j)$; **PRO** (xếp hạng nhiều ứng viên): $-\sum_k\log\frac{\exp(p_k)}{\sum_{i\ge k}\exp(p_i)}$ ($p_k$: log-prob có điều kiện của $y_k$ dưới student).

### 3.4 Skill và vertical (lướt)

Bảng 3 của survey tóm các công trình theo: skill, seed, teacher, student, phương pháp elicitation, objective. Nhóm skill: **context following** (instruction following — Self-Instruct 175 tác vụ mở rộng 52K; Alpaca; WizardLM Evol-Instruct; UltraChat; Orca/Orca 2 với system message và explanation trace; multi-turn; RAG), **alignment** (thinking pattern, preference, value — Zephyr, UltraFeedback, Constitutional AI), **agent** (tool use: Toolformer, Gorilla, ToolLLM; planning: FireAct, AgentTuning), **NLP task specialization** (NLU, NLG, IR, recommendation, đánh giá text, toán, code — WizardMath, WizardCoder, Magicoder), **multimodal** (LLaVA, SVIT, LVIS-Instruct4V, LLaVAR, MIMIC-IT; tất cả là Labeling + SFT từ GPT-4/ChatGPT). **Vertical:** luật, y tế, tài chính, khoa học, mục "miscellaneous". Hai mục này chủ yếu là danh mục công trình, không có phát hiện mới nên không mô tả sâu ở đây.

### 3.5 Pseudo-code (bản đồ ra quyết định, người ghi chú tóm lại)

```
Có logit của teacher (white-box)?
  yes -> feature knowledge:  KL token-level (forward / reverse / skew)  [28,29,30,31]
         + (tuỳ chọn) similarity trên hidden state                       [03,18,19]
  no  -> chỉ có văn bản (black-box):
         labeling/expansion/curation -> SFT                              [35,36,37,38]
         + feedback/preference -> RL hoặc DPO/ranking                    [37, survey 3.2.3-3.2.4]
Student có thể là teacher của chính nó?  -> self-knowledge (lọc, tự chấm)  [38]
Dung lượng student << teacher? -> cân nhắc reverse KL / on-policy / teacher trung gian [28,29,08,41]
```

## 4. Những gì survey báo cáo (thay cho thí nghiệm)

Survey không chạy thí nghiệm. Các "kết quả" là tóm tắt từ công trình khác (chưa kiểm chứng ở đây):
- **Self-Instruct:** GPT-3 tự mở rộng 175 tác vụ hạt giống lên 52K instruction; sau SFT, đạt gần InstructGPT trong zero-shot (theo mô tả survey). Alpaca: 52K dữ liệu từ text-davinci-003, student LLaMA-7B.
- **WizardLM:** 4 vòng Evol-Instruct bằng ChatGPT, 250K instruction phức tạp; trên tập instruction khó, win rate cao hơn ChatGPT 7.9% (theo survey).
- **phi-1:** 1B token "textbook" Python + 180M token bài tập; phát biểu của survey: vượt gần mọi mô hình mã nguồn mở trên HumanEval/MBPP dù nhỏ 10x về mô hình và 100x về dữ liệu. Phi-2 (2.7B) vượt Mistral 7B và Llama-2 13B trên nhiều benchmark (theo survey).
- **UltraChat:** 1.5M mẫu từ 30 meta-topic; UltraLLaMA vượt các chat model mở.
- **LIMA** (Zhou et al., nhắc ở phần open problems): 1000 mẫu chất lượng cao có thể đủ cho alignment.
- **Cảnh báo** (Gudibande et al.): student huấn luyện trên dữ liệu ChatGPT dễ bắt chước *phong cách* nhưng không sao chép được độ đúng thực tế; cần teacher mạnh hơn và dữ liệu đa dạng/chất lượng (như Orca).

## 5. Phân tích / các luận điểm đáng chú ý

- **Chất lượng và đa dạng dữ liệu quan trọng hơn lượng** (LIMA, phi, UltraChat, Orca): survey nhấn mạnh data curation/selection.
- **Học cách suy nghĩ, không chỉ trả lời:** Orca/Orca 2 thêm system message ("think step-by-step") và explanation trace; SelFee/Reflection-Tuning huấn luyện cả quá trình sửa. Phần này liên hệ với bài 36 và 37.
- **Từ imitation sang feedback/feature:** survey chỉ ra đa số công trình dùng teacher đóng với hard label + SFT; hướng "richer knowledge" (feature, feedback) còn ít khai thác.
- **On-policy:** MiniLLM và GKD cho student tự sinh chuỗi, teacher cho phân phối làm phản hồi ("feedback + feature" cùng lúc, Bảng 3 xếp GKD là "Feature + Feedback", objective D&S + RL).
- **Self-improvement:** ReST (Grow/Improve), Self-Play, Self-Rewarding (LLM-as-a-judge tự chấm) cho phép vượt "trần teacher".
- **Open problems (Mục 6):** (1) chọn dữ liệu (bao nhiêu, lọc thế nào); (2) khai thác tri thức giàu hơn từ teacher (feedback, feature, kết hợp nhiều kiểu); (3) quên thảm họa (catastrophic forgetting) khi chưng cất kỹ năng mới; (4) KD đáng tin cậy (trustworthiness: truthfulness, safety, fairness, privacy); (5) weak-to-strong distillation; (6) self-alignment.

## 6. Giới hạn

**Tác giả nêu:** (Declaration) không thể bao quát mọi công trình vì lĩnh vực tăng quá nhanh; ngoài ra, lưu ý tuân thủ điều khoản dịch vụ khi chưng cất từ ChatGPT/Llama (restrictions on developing competitive products).

**Người ghi chú nhận xét:**
- Survey nghiêng mạnh về **black-box, dữ liệu sinh, instruction tuning**; phần white-box (logit, hidden state) ngắn, chỉ vài đoạn trong 3.1.4 và 3.2.2. Không có phần về **nén có cấu trúc + KD** (pruning rồi phục hồi), không có phần về **vision-language distillation theo nghĩa nén** (Mục 4.5 chỉ bàn về dữ liệu instruction đa phương thức từ GPT-4).
- Hầu hết "kết quả" được trích lại từ bài gốc, không có chuẩn so sánh chung giữa các phương pháp, nên khó rút ra "phương pháp nào tốt hơn".
- Một số kết luận (như divergence phụ thuộc tác vụ) dựa trên một vài bài (GKD, MiniLLM); các bài sau (30, 31) tranh luận thêm.
- Phạm vi thời gian: đến khoảng đầu 2024; thiếu nhóm reasoning distillation (R-1, 38), scaling laws (41, 42), speculative KD (32).
- Bản trích văn bản dùng ở đây là PDF hai cột, một số bảng bị lẫn dòng; đã tránh trích số từ bảng mangled.

## 7. Liên hệ với các bài khác

| Phần của survey | Bài trong lộ trình | Ghi chú |
|---|---|---|
| Nền tảng KD | 02 Hinton KD, 05 KD survey, 03 FitNets | survey LLM bỏ qua phần cổ điển |
| SFT / SeqKD (3.2.1) | 15 Sequence-level KD, 35 Self-Instruct | |
| Labeling/Expansion/Curation | 35 Self-Instruct, 36 Distilling step-by-step, 37 Orca, 38 DeepSeek-R1 distillation | dữ liệu sinh từ teacher |
| Feature + Divergence (3.1.4, 3.2.2) | 28 MiniLLM, 29 GKD, 30 DistiLLM, 31 Rethinking KL, 33 TAID, 34 DistiLLM-2 | trọng tâm dự án |
| Similarity (hidden state) | 03 FitNets, 18 TinyBERT, 19 MiniLM, 20 Causal distillation | trong survey LLM chỉ nhắc TED |
| Feedback / RL / Ranking (3.2.3–3.2.4) | 25 LLaVA-MoD (preference distillation), Zephyr/DPO | |
| Self-knowledge | 38 DeepSeek-R1, 06 Born-again networks | |
| Multimodal (4.5) | 24 CLIP-KD, 25 LLaVA-MoD, 26 LLaVA-KD | survey chủ yếu nói về dữ liệu |
| Teacher–student gap | 08 Teacher assistant, 09 Efficacy of KD, 41 Capacity gap law, 39 Small models struggle... | không thuộc survey |
| Scaling, tối ưu mẫu | 42 Distillation scaling laws, 45 Sparse logit sampling | không thuộc survey |

Cách dùng làm bản đồ: gặp một bài mới, hỏi ba câu: (1) tri thức lấy từ đâu (6 kiểu elicitation), (2) học bằng thuật toán nào (4 họ), (3) nhằm skill/miền nào.

## 8. Ứng dụng cho dự án (suy luận)

Mức liên quan: **trung bình** như bản đồ khái niệm; ít liên quan như nguồn kỹ thuật cụ thể.

- Dự án của chúng ta là **white-box compress** (teacher 4B cùng họ với student pruned), tức nằm ở ô "Feature knowledge + Divergence/Similarity" (Mục 3.1.4, 3.2.2). Survey xác nhận ba điểm đáng nhớ: (i) forward vs reverse KL tùy tác vụ; (ii) on-policy (student tự sinh, teacher cho phân phối) là hướng được đề xuất cho student yếu; (iii) similarity trên hidden state ít được thử cho LLM nhưng hứa hẹn.
- **Self-knowledge/feedback** gợi ý một hướng bổ sung: dùng teacher 4B chấm/lọc bản OCR do student sinh ra để tạo dữ liệu huấn luyện thêm (rejection sampling), hoặc dùng cặp (output tốt, output lỗi) cho DPO; cần chứng minh bằng thực nghiệm, survey không có OCR.
- **Black-box/curation** gần như không dùng được: ta có teacher mở, nhưng ý tưởng "sinh dữ liệu tổng hợp có kiểm soát" (phi-1 style) có thể dùng để tạo thêm trang tài liệu hiếm (bảng, chữ viết tay) để tăng độ phủ dữ liệu OCR; suy luận, ngoài phạm vi survey.
- **Catastrophic forgetting** (open problem): sau khi prune và phục hồi cần theo dõi năng lực cũ (đa ngôn ngữ, layout) khi chưng cất vào một miền; liên quan trực tiếp dự án vì ta fine-tune trên dữ liệu OCR.
- Survey không đề cập phân rã D_lin/D_res hay so sánh cắt neuron với cắt layer; không có thông tin cho các câu hỏi đó.
- Điều khoản dịch vụ: nếu dùng teacher đóng để sinh dữ liệu, kiểm tra giấy phép (survey nhắc tuân thủ).

## 9. Thí nghiệm gợi ý

1. Phân loại 5–6 bài sắp đọc (28–34) vào bảng `Knowledge × Algorithm × Skill` ở Mục 3; điền ô trống để thấy hướng chưa ai làm (ví dụ: Similarity trên hidden state cho VLM nhỏ).
2. Trên student 0.3B pruned: so sánh (a) SFT trên văn bản teacher (SeqKD), (b) KL logit (feature), (c) (b)+rejection sampling bằng teacher chấm điểm CER; đo CER và chi phí.
3. Theo dõi forgetting: đo điểm trên bộ OCR chuẩn cũ trước/sau mỗi vòng KD trên miền mới.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý | Ghi chú |
|---|---|---|
| Pipeline KD LLM | steer → seed → generate → train | 4 bước |
| Labeling | $y\sim p_T(y|I\oplus c\oplus x)$ | $x$ có sẵn |
| Expansion | $x,y$ đều do teacher sinh | Self-Instruct |
| Curation | sinh từ meta-information $m$ | UltraChat, phi |
| Feature | logit/hidden của teacher | white-box |
| Feedback | teacher chấm/sửa đầu ra student | RLAIF |
| Self-knowledge | student tự sinh và lọc | ReST |
| SFT / SeqKD | $-\log p_S(y|x)$, $y\sim p_T$ | baseline mạnh |
| Divergence | forward KL, reverse KL, JS | task-dependent |
| Similarity | $L_F(\Phi_T f_T,\Phi_S f_S)$ | L1/L2/CE/MMD |
| RL distillation | reward model + PPO có KL tới $\pi_{ref}$ | đắt |
| Ranking | DPO, RRHF, PRO | ổn định hơn RL |

## 11. Kết luận ngắn

Survey cung cấp bản đồ ba trụ (algorithm / skill / verticalization) và hai bước độc lập (elicit rồi distill) để định vị mọi công trình KD cho LLM, với nhấn mạnh data augmentation và chuyển từ imitation đầu ra sang học suy luận, feedback và self-improvement. Với dự án nén VLM OCR, hãy dùng nó để định hướng (white-box, divergence, on-policy, forgetting), nhưng nguồn kỹ thuật chính nằm ở các bài 28–34.

Link gốc: https://arxiv.org/abs/2402.13116
