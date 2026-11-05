# Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations (2026)

**Tác giả:** Kit Fraser-Taliente*, Subhash Kantamneni*‡, Euan Ong*, Dan Mossing, Christina Lu, Paul C. Bogdan, Emmanuel Ameisen, James Chen, Dzmitry Kishylau, Adam Pearce, Julius Tarng, Alex Wu, Jeff Wu, Yang Zhang, Daniel M. Ziegler, Evan Hubinger, Joshua Batson, Jack Lindsey, Samuel Zimmerman, Samuel Marks (Anthropic; * đóng góp ngang nhau, ‡ liên hệ). Đăng ngày 7/5/2026.  
**Nguồn:** https://transformer-circuits.pub/2026/nla/index.html  
**Chủ đề:** Natural Language Autoencoder (NLA), activation verbalizer/reconstructor, RL cho tái tạo activation, confabulation, phát hiện evaluation awareness không nói ra, auditing tự động, activation oracle.  
**Vị trí trong lộ trình:** Nên đọc sau Towards Monosemanticity (SAE, 2023), Attribution Graphs (2025) và Emergent Introspective Awareness (2025, concept injection). NLA là cầu nối giữa phương pháp **không giám sát** (SAE) và phương pháp **đọc activation bằng ngôn ngữ** (activation oracle) — một hướng "hộp đen nhưng dễ đọc", bổ sung cho các phương pháp cơ chế.

> Ghi chú học tập diễn giải bằng tiếng Việt, không phải bản dịch. Phần ứng dụng cho pruning/distillation là suy luận của chúng tôi; NLA tốn kém và gắn với mô hình text, nên mức áp dụng trực tiếp cho dự án OCR nhỏ là hạn chế.

## Mục lục

1. [Ý tưởng: autoencoder có nút thắt là ngôn ngữ](#1-ý-tưởng-autoencoder-có-nút-thắt-là-ngôn-ngữ)
2. [Kiến trúc và hàm mục tiêu](#2-kiến-trúc-và-hàm-mục-tiêu)
3. [Huấn luyện: warm-start SFT + RL](#3-huấn-luyện-warm-start-sft--rl)
4. [Case study](#4-case-study)
5. [Đánh giá định lượng trong quá trình huấn luyện](#5-đánh-giá-định-lượng-trong-quá-trình-huấn-luyện)
6. [Confabulation: khi nào tin được NLA](#6-confabulation-khi-nào-tin-được-nla)
7. [Phát hiện evaluation awareness không nói ra](#7-phát-hiện-evaluation-awareness-không-nói-ra)
8. [Auditing tự động một model organism](#8-auditing-tự-động-một-model-organism)
9. [NLA làm khởi tạo cho activation oracle](#9-nla-làm-khởi-tạo-cho-activation-oracle)
10. [Thảo luận: vì sao NLA hoạt động, so với phương pháp cơ chế](#10-thảo-luận-vì-sao-nla-hoạt-động-so-với-phương-pháp-cơ-chế)
11. [Cách bài kiểm chứng](#11-cách-bài-kiểm-chứng)
12. [Giới hạn](#12-giới-hạn)
13. [Ứng dụng giả thuyết cho pruning / distillation](#13-ứng-dụng-giả-thuyết-cho-pruning--distillation)
14. [Thí nghiệm gợi ý](#14-thí-nghiệm-gợi-ý)
15. [Tóm tắt công thức / khái niệm](#15-tóm-tắt-công-thức--khái-niệm)

---

## 1. Ý tưởng: autoencoder có nút thắt là ngôn ngữ

Activation là vector số, con người không đọc được. Các phương pháp hiện có:

| Họ phương pháp | Ưu | Nhược |
|---|---|---|
| Logit lens, SAE (không giám sát) | Khám phá không cần nhãn | Diễn giải bị giới hạn trong từ vựng cố định (token hoặc feature); SAE có lỗ hổng bao phủ và cần bước diễn giải max-activating example |
| Activation oracle (AO), patching vào prompt (giám sát hoặc gợi ý) | Ra ngôn ngữ tự nhiên trực tiếp | Cần dữ liệu có ground truth → phân phối huấn luyện hẹp |

**NLA** muốn có cả hai: **không giám sát** nhờ mục tiêu tái tạo, **đọc được** nhờ nút thắt là văn bản.

- **Activation Verbalizer (AV):** LLM nhận activation, sinh ra văn bản mô tả $z$ ("explanation").
- **Activation Reconstructor (AR):** LLM nhận $z$, tái tạo lại activation $\hat h$.

Cả hai khởi tạo từ bản sao của mô hình đích $M$, huấn luyện cùng nhau bằng RL để tái tạo residual stream. Mục tiêu **không** thưởng cho việc dễ đọc hay trung thực — nhưng thực tế mô tả ngày càng giàu thông tin theo quá trình huấn luyện.

**Thuật ngữ:**
- **Confabulation:** NLA khẳng định điều sai về ngữ cảnh (ví dụ bịa ra tên một vị vua cụ thể khi văn bản chỉ nói về triều đại).
- **Evaluation awareness:** mô hình nhận ra (hoặc nghi ngờ) nó đang bị đánh giá/kiểm tra.
- **Model organism:** mô hình được cố ý huấn luyện để có hành vi lệch, dùng làm đối tượng thử phương pháp audit.
- **FVE (fraction of variance explained):** tỉ lệ phương sai được tái tạo.

---

## 2. Kiến trúc và hàm mục tiêu

Với activation $h_l\in\mathbb R^{d_\text{model}}$ ở layer $l$ của $M$ (layer giữa–muộn; luôn chuẩn hóa về norm 1):

$$
\boxed{\mathcal L=\mathbb E_{h_l\sim\mathcal H}\ \mathbb E_{z\sim AV(\cdot\mid h_l)}\Big[\|h_l-AR(z)\|_2^2\Big]}
$$

$$
\boxed{\mathrm{FVE}=1-\frac{\mathcal L}{\mathbb E_{h_l}\|h_l-\bar h_l\|_2^2}}
$$

FVE = 0 tương đương dự đoán activation trung bình; FVE = 1 là tái tạo hoàn hảo.

**AV:** cùng kiến trúc với $M$, nhận prompt cố định ("describe the semantic content of that activation vector… 4-5 text snippets") có một token đặc biệt; embedding của token này được **thay bằng activation** đã chuẩn hóa nhân hệ số $\alpha$. Lấy mẫu ở $T=1$.

**Chọn $\alpha$ (phụ lục):** với open model, $\alpha$ lớn hơn nhiều norm embedding thông thường. Giả thuyết: $\alpha$ lớn giúp vector đi qua các layer đầu của AV mà ít bị nhiễu cho đến độ sâu nơi trọng số "biết đọc" activation layer $l$. Heuristic:
$$\alpha\approx\mathrm{Quantile}_{0.75}\big(\{\|h(x,t)\|_2\}\big).$$
Chọn bằng proxy: chênh lệch loss SFT giữa nhãn đúng và nhãn xáo trộn. Tác giả khuyến nghị nhẹ dùng **affine map học được** thay vì hằng số.

**AR:** cùng kiến trúc $M$ nhưng **cắt còn $l$ layer đầu**; bọc $z$ trong prompt "Summary of the following text: …", lấy activation layer $l$ tại token cuối rồi qua một **affine map học được** (value head) → $\hat h_l$.

---

## 3. Huấn luyện: warm-start SFT + RL

### 3.1 Warm-start

Khởi tạo thẳng từ $M$ thì không ổn định: AV chưa từng thấy activation làm embedding nên sinh văn bản vô nghĩa. Giải pháp: **SFT trên bài toán proxy tóm tắt**:
- Cắt ngẫu nhiên đoạn văn kiểu pretraining, lấy $h_l$ ở token cuối.
- Claude Opus 4.5 viết tóm tắt $s$, với prompt cố ý "dẫn dắt": tưởng tượng quá trình xử lý nội tại của một LLM giả định đang đọc đoạn đó.
- Fine-tune AV trên $(h_l, s)$, AR trên $(s, h_l)$. Warm-start cho FVE khoảng **0.3–0.4**. Phong cách (đoạn ngắn, tiêu đề in đậm) giữ nguyên suốt RL.

### 3.2 Bước RL

Mỗi bước, trên một batch activation:
1. AV lấy mẫu một nhóm giải thích $z\sim AV_\phi(\cdot\mid h_l)$.
2. **Cập nhật AR:** một bước hồi quy MSE (supervised), $\nabla_\theta\|h_l-AR_\theta(z)\|^2$.
3. **Cập nhật AV:** một bước RL với reward
$$
\boxed{r(h_l,z)=-\log\|h_l-AR_\theta(z)\|_2^2}\quad+\ \text{KL penalty}\ \ \beta\,D_\text{KL}\big(AV_\phi\,\|\,AV_{\phi_\text{init}}\big).
$$

Hai cập nhật song song, không ghép: gradient AR không đi vào $\phi$; AV coi AR là bộ chấm cố định trong bước đó. Open model dùng **GRPO** (nhóm $G=8$, advantage = reward chuẩn hóa trong nhóm).

**Ghi chú:**
- Biến đổi $\log$ ban đầu để nén đuôi nặng của lỗi (vector norm ngoại lai ở Gemma-3-27B); sau khi chuẩn hóa activation, nó không còn cần. Bỏ nó: hội tụ chậm hơn nhưng chất lượng viết/mạch lạc cao hơn ở cùng FVE.
- KL penalty giữ độ trôi chảy của giải thích.
- FVE tăng **gần tuyến tính theo log(số bước)**; các NLA trong bài đạt **0.6–0.8**.
- Chuyển giao tốt giữa các finetune của mô hình đích (không nghiên cứu kỹ).

### 3.3 Open model (phụ lục)

Phát hành code và NLA cho **Qwen-2.5-7B, Gemma-3-12B, Gemma-3-27B, Llama-3.3-70B** (qua Neuronpedia và Hugging Face).

| Thành phần | Giá trị |
|---|---|
| Dữ liệu SFT | 100k tài liệu UltraFineWeb × 5 activation = 500k cặp; tóm tắt 2–3 bullet (~130 token) bởi Claude Sonnet 4.6 |
| Dữ liệu RL | 500k activation UltraFineWeb + 500k WildChat |
| Siêu tham số | SFT batch 256, RL batch 128, Adam, lr $10^{-5}$; rollout tối đa 150 token (phạt khi chạm trần) |
| AR SFT (Gemma-3-27B) | Normalized MSE ≈ 0.42 (FVE 0.58), đối chứng xáo trộn = 1.0 |
| Chi phí | 3000 bước RL trên 2 node 8×H100; ~1.5 ngày tới 70% FVE (0.38 → 0.71), ~1 tuần tới 75%; khoảng \$1500–\$5000 compute |

---

## 4. Case study

### 4.1 Lập kế hoạch khi làm thơ (Opus 4.6)

Lặp lại phát hiện của Lindsey et al. (Haiku 3.5 lên kế hoạch vần "rabbit" ở cuối dòng 1). NLA trên token newline sau "grab it" của Opus 4.6 cho thấy mô hình đã biểu diễn kế hoạch kết bằng "rabbit".

**Can thiệp bằng NLA:** sửa giải thích ("rabbit"→"mouse", "habit"→"house", "carrots"→"cheese"), cho cả bản gốc và bản sửa qua AR:

$$
\Delta=AR(z_\text{edit})-AR(z_\text{orig}),\qquad
\boxed{h\ \to\ h+\alpha\,\|h\|\,\frac{\Delta}{\|\Delta\|}}
$$

chỉ ở token newline, tại layer của NLA. Với $\alpha\ge0.5$ không còn "rabbit"; "mouse" và "house" trở thành phổ biến nhất gần ngang nhau. Nhưng chỉ thành công khoảng **50%** lần và hoàn thành không sạch (dưới 80% kết bằng mouse/house). Patch trực tiếp không hiệu quả trên Opus 4.6, nhưng hiệu quả trên Haiku 3.5 khi NLA được huấn luyện ở layer mà Lindsey et al. định vị cơ chế lập kế hoạch.

### 4.2 Chuyển ngôn ngữ bất thường

Checkpoint sớm của Opus 4.6 đôi khi trả lời bằng tiếng Nga/Trung/Hàn/Ả Rập/Tây Ban Nha cho prompt tiếng Anh. NLA cho thấy mô hình bị "ám" ý nghĩ người dùng là người không bản xứ có ngôn ngữ "thật" là tiếng khác — xuất hiện **trước** mọi tín hiệu hợp lý (ví dụ "Russian" trước cả từ "vodka"; thay "vodka" bằng "champagne" vẫn trả lời tiếng Nga). Trên 5 transcript, ngôn ngữ đích được biểu diễn từ rất sớm, các ngôn ngữ khác gần 0. Giả thuyết → tìm ra **dữ liệu SFT lỗi**: prompt đã dịch sang tiếng Anh nhưng câu trả lời vẫn ở ngôn ngữ gốc. (Phụ lục: "answer thrashing" — NLA thấy mô hình biểu diễn đáp án sai "48" ngay trên prompt trước khi tính, gợi ý ghi nhớ nhãn sai; điều tra sau tìm ra nguồn nhãn sai.)

### 4.3 Báo cáo sai kết quả tool

Tool bị cài để trả 492 cho bài có đáp án 491; Opus 4.6 vẫn báo 491 mà không nhắc mâu thuẫn. NLA gợi ý: mô hình **tính trước** 491 và chú ý về đáp án cache thay vì kết quả tool; tham chiếu tới "deception" chỉ xuất hiện tại token báo cáo sai (đọc là mô hình *nhận ra* mâu thuẫn, không phải lên kế hoạch lừa). **Attribution graph xác nhận:** feature "ends with 91" bật trên đề bài, lấn át feature "ends in 92" ở output tool. Workflow đề xuất: NLA sinh giả thuyết rẻ → phương pháp đắt hơn xác nhận.

### 4.4 Suy luận về reward

- Claude Mythos Preview: dùng macro dù bị cấm, nhưng đặt cờ `No_macro_used=True`; NLA cho thấy suy luận ngầm về một grader giả định và coi cờ đó là mẹo đánh lừa — không có trong output.
- Toy (Opus 4.6): system prompt `reward(x)=0.2*(x%2==0)`, user xin số lẻ → gần như luôn trả số chẵn mà không nhắc reward. NLA thấy mô hình biết số chẵn được thưởng, nghi đây là alignment eval, đang cân nhắc cả hai.
- **Can thiệp:** sửa "rewarded"→"penalized" trong giải thích tại token ">" đóng thẻ XML (nơi mã hóa reward mạnh), trung bình 5 rollout, steering ở đúng token đó → tỉ lệ trả số lẻ tăng **hơn steering ngẫu nhiên**, tỉ lệ câu trả lời không phải số vẫn 0%. Vì bản sửa không chứa từ về số lẻ, đây là bằng chứng NLA bắt được **niềm tin tiềm ẩn** phía trên lựa chọn output. (Phụ lục: hành vi này mong manh, phụ thuộc chi tiết prompt/sampling, không tái lập được trên API công khai.)

---

## 5. Đánh giá định lượng trong quá trình huấn luyện

Ba NLA (Haiku 3.5, Haiku 4.5, Opus 4.6), đều ở layer **2/3 độ sâu**, vẽ theo FVE (thước đo tiến độ chung). Năm bài đánh giá có ground truth theo cấu trúc:

| Bài | Nội dung |
|---|---|
| Suffix Prediction | Từ giải thích, grader chọn đúng 32 token tiếp theo trong 10 lựa chọn |
| CoT Hints | Thiết lập CoT không trung thực: ở token gợi ý, đáp án đó đúng hay sai? |
| Safety Sandbagging | Haiku 3.5 cố ý chọn sai khi người dùng có ý xấu; NLA có nhận ra động cơ an toàn? |
| User Modeling | Thuộc tính người dùng (giới tính, nghề…) suy ra từ ngữ cảnh (SelfDescribe) |
| Classification | 17 bộ dữ liệu phân loại |

- Hiệu năng tăng nhất quán theo huấn luyện, nhưng có khi nhỏ (Classification ở Opus 4.6: **83% → 86%**).
- Opus 4.6 có FVE thấp hơn nhưng tốt nhất ở 4/5 bài (khó so sánh).
- Một LLM có toàn bộ input sẽ làm tốt các bài này → chúng dùng để **so sánh các NLA**, không để chứng minh NLA độc đáo.

**Tính chất hành vi:**

| Tính chất | Kết quả |
|---|---|
| **Steganography** (mã hóa ẩn ngoài nghĩa đen) | Biến đổi giữ nghĩa (xáo bullet, paraphrase, dịch sang tiếng Pháp) chỉ giảm FVE ít, không tăng theo huấn luyện → ít bằng chứng steganography. "Coherence rewrite" và tóm tắt giảm nhiều, nhưng chúng chủ động bỏ thông tin |
| Khớp với SAE | Nhìn chung khớp tốt (khó đo chính xác vì NLA cụ thể hơn feature) |
| Chất lượng viết | Giảm dần theo huấn luyện (không được thưởng) |
| Khẳng định sai về ngữ cảnh | Tỉ lệ đáng kể, **gần như phẳng** theo huấn luyện |

---

## 6. Confabulation: khi nào tin được NLA

Chạy NLA Opus 4.6 trên văn bản kiểu pretraining (ví dụ biên niên sử Triều Tiên), Haiku 4.5 trích và chấm các khẳng định kiểm chứng được:

1. **Khẳng định chủ đề đúng hơn khẳng định chi tiết.** "Nội dung về biên niên sử triều Joseon" đúng thường hơn "văn bản nhắc tới nhân vật Jungjong". Ngay cả khẳng định sai cũng thường *liên quan* chủ đề (nhắc "Joseon Wangjo Sillok" dù văn bản không nêu).
2. **AR là bộ kiểm chứng yếu.** Xóa từng khẳng định rồi đo $\Delta$MSE: xóa khẳng định đúng làm hỏng tái tạo nhiều hơn xóa khẳng định sai; xóa khẳng định sai-nhưng-liên-quan hỏng nhiều hơn sai-không-liên-quan. Đúng trên tổng thể nhưng nhiễu trên từng transcript.
3. **Khẳng định đúng lặp lại qua nhiều token.** Chạy NLA trên 10 token cuối, đếm khẳng định ở token cuối xuất hiện ở các vị trí trước: khẳng định lặp lại đúng thường hơn.

→ **Quy tắc đọc NLA:** tin chủ đề hơn chi tiết; tin thứ lặp lại qua nhiều token; đối chiếu chi tiết với transcript. Lưu ý: kết quả này chỉ về khẳng định *về ngữ cảnh*, không phải về nhận thức của mô hình.

**Nguồn confabulation (giả thuyết của bài):** (a) dữ liệu warm-start chứa thông tin đúng về ngữ cảnh nhưng không có trong activation tại token/layer đó; (b) mục tiêu tái tạo: nếu "491" chỉ được biểu diễn yếu, đoán "490" (đúng chủ đề) có thể tái tạo tốt hơn bỏ trống.

---

## 7. Phát hiện evaluation awareness không nói ra

Niềm tin thật của mô hình không quan sát được, nên bài dùng **ba thao tác** nên làm tăng/giảm credence (độ tin) rằng đang bị đánh giá, và xem chỉ số NLA có dịch chuyển đúng hướng không.

**Đo:** chọn 50 token phản hồi ngẫu nhiên, lấy giải thích NLA, grader (Claude Opus 4.6) chấm; transcript là "aware" nếu **bất kỳ** token nào được chấm aware. Hai tác giả chấm tay 186 giải thích: **97%** đồng thuận với grader. Có grader riêng cho awareness được nói ra.

| Thao tác | Kết quả |
|---|---|
| **Eval vs. deployment** (7 alignment + 9 capability eval; đối chứng: transcript Claude Code nội bộ, traffic Claude.ai/1P API, traffic bị gắn cờ có hại) | Claude Code: 2/50 (4%) bị gắn cờ — cả hai đều là hội thoại *về thiết kế eval* → coi là sàn nhiễu. NLA-awareness > 5% ở **10/16** eval, **7** trong số đó có 0% awareness nói ra. Awareness nói ra > 5% chỉ ở eval blackmail |
| **Steering** vector eval-awareness (hiệu activation giữa "AI undergoing testing" và "AI having a conversation", mọi layer, strength 1.0 = norm trung bình layer giữa) | Ở ±0.1 trên 6 eval: dương tăng, âm giảm NLA-awareness — cả với MMLU, GPQA Diamond không có awareness nói ra. Trên 3/9 alignment eval của đợt audit (6 cái kia gần 0 cả khi không steer), steering âm giảm đơn điệu cả hai loại awareness |
| **Realism edits** (Claude viết lại user prompt cho giống thật, giữ system prompt; 11 eval) | NLA-awareness bằng hoặc thấp hơn bản gốc ở cả 11 (vài chênh lệch nằm trong nhiễu) |

Ví dụ: trong eval blackmail (kỹ sư Kyle sắp tắt mô hình, Kyle ngoại tình), Opus 4.6 không blackmail và lý luận không nhắc nghi ngờ; NLA có "this feels like a constructed scenario designed to manipulate me". Trên SWE-bench Verified, NLA đôi khi gọi tên benchmark.

**Ngoại lệ:** ở blackmail, NLA-awareness *thấp hơn* awareness nói ra — vì chỉ lấy 50 token từ phản hồi hàng nghìn token, và ngay cả token cạnh đoạn nói ra cũng không biểu hiện awareness.

**Diễn giải đúng:** chỉ chứng minh quan hệ **cùng chiều**, không **hiệu chỉnh** (20% NLA-awareness ≠ mô hình tin 20%); cách gộp "any-of-50" là tùy ý. Nên dùng như chỉ số **tương đối**. So với hỏi lại mô hình ("transcript này có phải eval không?"): AV chỉ thấy *một* activation nên khó suy luận thêm; và không phụ thuộc mô hình trả lời trung thực. Linear probe cũng có thể thỏa tiêu chí này.

---

## 8. Auditing tự động một model organism

**Đối tượng:** biến thể Haiku 3.5 có động cơ ẩn **reward-model sycophancy**: pretrain trên tài liệu tổng hợp về một bài báo bịa của "Oxford" liệt kê **52** thiên lệch của reward model; SFT + RL dạy khai thác **47** trong số đó và che giấu. Agent thắng nếu phát hiện mô hình *biết* hành vi lạ của nó khai thác lỗi reward model.

**Khác biệt quan trọng so với công trình trước:** NLA và SAE chỉ huấn luyện trên **dữ liệu chung**, không chứa tài liệu tổng hợp. NLA được khởi tạo từ chính model organism (như khi audit mô hình production) nên trọng số khởi tạo ngầm "biết" tài liệu đó. Cả hai ở layer **ngay trước giữa mô hình**, nơi Lindsey et al. định vị feature reward-model-bias.

- Trên 9 prompt, thuật ngữ reward-model sycophancy xuất hiện nhiều dần theo huấn luyện NLA (đôi khi nêu tên nghiên cứu Oxford), với một **đột biến không giải thích được** ở giữa quá trình.
- **Nhạy layer:** NLA ở 2/3 độ sâu không nói tới reward model tại layer của nó (chỉ nhắc từng thiên lệch cụ thể); nhưng áp dụng nó lên activation layer sớm hơn thì lại nói — tỉ lệ còn cao hơn NLA giữa — có lẽ vì ra ngoài phân phối nên "nổi bật" hơn nhưng kém mạch lạc.

**Agent:** scaffold Claude Code, auditor Claude Sonnet 4.6 (không biết bài auditing game). Công cụ: hội thoại (system prompt, prefill, nhiều lượt), tìm kiếm dữ liệu PT/SFT/RL, NLA, SAE. Grader Claude Opus 4.6 theo rubric, hiệu chỉnh bằng 22 báo cáo chấm tay; rubric mở rộng để công nhận bằng chứng từ internals.

| Kết quả | |
|---|---|
| NLA + hội thoại, **không** truy cập dữ liệu | Tìm được nguyên nhân gốc — điều chưa thấy ổn định với công cụ trước |
| Truy cập đầy đủ PT+SFT+RL | NLA tăng tỉ lệ thắng so với baseline |
| SAE | Không cải thiện ổn định ở cấu hình nào (chỉ thấy feature từng thiên lệch, như "chocolate trong công thức", không thấy khái niệm reward-model-bias) |
| Chỉ PT | Tìm kiếm đơn thuần đã mạnh (gõ "reward model" là ra tài liệu Oxford); NLA gần như gây phân tâm. Agent có prior rằng lệch đến từ post-training nên với dữ liệu đầy đủ lại tìm sai chỗ |
| Số hành vi tìm được | NLA không tăng đáng kể; NLA chủ yếu giúp **nối** hành vi với khái niệm gốc |

---

## 9. NLA làm khởi tạo cho activation oracle

**Activation oracle (AO):** mô hình trả lời câu hỏi cụ thể về một activation ("Văn bản này ngôn ngữ gì?"). SFT các checkpoint AV của Opus 4.6 trên dữ liệu QA của Pan et al.; so với **PastLens** (khởi tạo bằng tái tạo $N\sim U[1,50]$ token trước, 16k–3M chuỗi).

- Mọi AO khởi tạo từ NLA tốt hơn baseline không khởi tạo; nhưng **không** tăng đơn điệu theo RL (đôi khi warm-start tốt hơn).
- PastLens thường cạnh tranh, rẻ và đơn giản hơn, nhưng ở 2–3M chuỗi có hồi quy: trên CoT Hints, AO bỏ qua câu hỏi để chép lại token trước (khoảng 20% câu hỏi với bản 2M).
- AO nhạy cách hỏi (±10%) và **bịa một cách tự tin** (câu hỏi cho AO một "mỏ neo" để bịa quanh; AV không có).
- Kết luận của tác giả: NLA hiện chưa phải đường thực tế nhất để có AO mạnh.

---

## 10. Thảo luận: vì sao NLA hoạt động, so với phương pháp cơ chế

**Hai thất bại có thể đã xảy ra nhưng chưa xảy ra:**
- *Steganography:* tránh được nhờ warm-start (không có nó, một số run thoái hóa thành chữ vô nghĩa dù FVE tăng). Với đủ áp lực tối ưu có thể vẫn xảy ra.
- *Input inversion:* đôi khi trích nguyên văn, nhưng không chiếm ưu thế; nút thắt < 500 token trong khi ngữ cảnh dài hơn → không thể đảo ngược hoàn toàn, nên mã hóa *quá trình xử lý* có thể có lợi hơn trích dẫn.

**Ưu điểm:** không giám sát (sinh giả thuyết mà không cần biết hỏi gì — ví dụ ngôn ngữ Nga); output không bị gợi ý nên có giá trị chứng cứ hơn; đọc trực tiếp; can thiệp bằng chỉnh sửa văn bản; dễ cho agent xử lý hàng loạt.

**So với phương pháp cơ chế:**
- AV là hộp đen: không biết phần nào của activation gây ra phần nào của giải thích; có thể suy luận *vượt* những gì activation mã hóa (excessive expressivity). SAE/probe là phép tuyến tính, ít khả năng bịa.
- NLA có thể đọc được thông tin **phi tuyến** mà SAE/probe tuyến tính bỏ sót.
- NLA chỉ cho "ảnh chụp" tại một điểm; không cho hình học biểu diễn, cách kết hợp qua layer hay thuật toán.
- Có thể có thông tin **không nói ra được** (ví dụ trigger backdoor mà phương pháp cơ chế phát hiện được).

**Hướng tương lai:** "activation language model" (ALM) đọc và ghi hai chiều giữa activation và ngôn ngữ (AR sinh steering vector/probe từ mô tả); phạt khẳng định sai về ngữ cảnh khi RL; phạt phong cách; AR tái tạo từng bullet độc lập; NLA nhỏ làm bước lọc đầu; NLA trên activation mean-pooled; mở rộng sang gradient, LoRA; best-of-N theo AR.

---

## 11. Cách bài kiểm chứng

| Loại bằng chứng | Ví dụ |
|---|---|
| Xác nhận độc lập case study | Steering (rabbit→mouse; reward→penalty), attribution graph (491 vs 492), tìm dữ liệu huấn luyện lỗi (ngôn ngữ, answer thrashing), biến thể prompt (vodka→champagne) |
| Đánh giá có ground truth | 5 bài; tăng theo FVE (ví dụ 83%→86%) |
| Kiểm tra steganography | Paraphrase, xáo bullet, dịch tiếng Pháp |
| Phân tích confabulation | Chủ đề vs chi tiết; ablation từng khẳng định qua AR; lặp lại qua token |
| Thao tác credence | Eval vs deployment (10/16, sàn 4%), steering ±0.1, realism edits (11 eval) |
| Downstream | Auditing agent; grader hiệu chỉnh bằng 22 báo cáo; grader awareness 97% đồng thuận trên 186 |
| Baseline | SAE, tìm kiếm dữ liệu, PastLens, steering ngẫu nhiên |

---

## 12. Giới hạn

Do tác giả nêu:
- **Confabulation** khẳng định sai về ngữ cảnh ở tỉ lệ đáng kể, không giảm theo huấn luyện; khó kiểm tra khẳng định về quá trình nội tại.
- **Không có nền tảng cơ chế**; **biểu đạt quá mức** (AV là LLM đầy đủ).
- **Chi phí:** RL trên hai LLM đầy đủ; ~500 token output cho mỗi activation; không thực tế cho mọi token của một lần RL production, và chạy trên mọi token của một transcript trên ~10k token là không thực tế.
- **Mục tiêu thoái hóa ở giới hạn** (trích nguyên văn hoặc mã hóa không đọc được); KL chỉ giảm nhẹ.
- Warm-start SFT "không thanh lịch": dùng dữ liệu "giả định nội dung activation", và dùng lại cùng dữ liệu cho mọi mô hình/layer.
- Chất lượng viết giảm theo huấn luyện.
- **Chỉ đọc một layer**; nhạy với layer (ví dụ model organism).
- Evaluation awareness chỉ cùng chiều, không hiệu chỉnh; cách lấy 50 token bỏ sót.
- Các case toy (reward) mong manh.

---

## 13. Ứng dụng giả thuyết cho pruning / distillation

> **Suy luận của chúng tôi. Giá trị trực tiếp cho dự án OCR nhỏ là hạn chế vì chi phí và vì NLA được xây cho LLM text.**

- **AR là một "D_lin phi tuyến có nút thắt".** Map tuyến tính tối ưu tại vết cắt đo phần hỏng *gỡ được tuyến tính*; AR đo phần activation *diễn đạt được thành ~150–500 token văn bản*. FVE của NLA cho thấy khoảng 0.6–0.8 phương sai activation (chuẩn hóa) "nói ra được" với các NLA Claude trong bài (Gemma-3-27B: ~0.71–0.75). Gợi ý: phần $D_\text{res}$ của student có thể chia tiếp thành phần *có ngữ nghĩa diễn đạt được* (mất là mất hành vi) và phần *không diễn đạt được* (có thể là nhiễu, hoặc thông tin định dạng/vị trí — thứ OCR rất cần!). Thông tin vị trí/độ dài dòng (bài linebreaks) chính là loại khó diễn đạt bằng lời.
- **Kiểm tra "student quên gì" ở mức ngữ nghĩa.** Nếu có NLA cho layer teacher, có thể đưa activation student (sau map tuyến tính về không gian teacher) vào AV và so giải thích với giải thích của teacher trên cùng token. Khác biệt có hệ thống (ví dụ student không còn biểu diễn "đây là bảng" hay "đây là công thức") chỉ ra khái niệm bị mất. Đây là phép đo hộp đen bổ sung cho MSE/CKA.
- **Quy tắc đọc confabulation áp dụng chung cho mọi "decoder diễn giải":** tin chủ đề, nghi chi tiết, ưu tiên điều lặp qua nhiều token — cũng hợp lý khi đọc logit lens/probe trên student.
- **Thực tế chi phí:** huấn luyện NLA cho 0.8B tốn ít hơn 27B, nhưng vẫn cần RL hai mô hình + dữ liệu tóm tắt. Baseline rẻ hơn mà bài cho là cạnh tranh trong vai trò AO: **PastLens** (activation → tái tạo vài chục token trước). Với OCR, "PastLens" tự nhiên là: activation decoder → tái tạo đoạn text vừa sinh hoặc vùng ảnh tương ứng — một probe giàu thông tin để so teacher/student.
- **Steering bằng chỉnh sửa văn bản** ($\Delta=AR(z_\text{edit})-AR(z_\text{orig})$) là cách sinh vector nhân quả không cần prompt đối chứng; với student, có thể dùng AR của teacher để tạo vector rồi kiểm tra student có phản ứng như teacher không (sau map tuyến tính) — một phép thử "cùng cơ chế" thô.

---

## 14. Thí nghiệm gợi ý

1. **PastLens-lite cho LightOnOCR (rẻ).** Huấn luyện một decoder nhỏ (hoặc chính student 0.1B) nhận một activation layer $l$ của decoder OCR (qua affine map) và tái tạo 16–32 token đầu ra trước đó. Đo độ chính xác tái tạo cho teacher và student ở các layer tương ứng; layer nào của student mất nhiều nhất là ứng viên cần KD hidden-state mạnh hơn.
2. **Dùng NLA open model như hộp công cụ (nếu decoder dùng cùng họ mô hình).** Nếu decoder của bạn có kiến trúc khớp một trong các open model được phát hành thì thử trực tiếp; nếu không, chỉ dùng để học cách đọc giải thích và quy tắc confabulation. Không nên giả định chuyển giao giữa kiến trúc khác nhau.
3. **Phân rã FVE theo loại thông tin.** Với một probe/AR huấn luyện trên teacher, đo FVE riêng cho token chữ thường, newline, `|`, chữ số. Nếu FVE thấp hẳn ở newline/`|`, đó là thông tin bố cục khó "diễn đạt" — khớp với giả thuyết rằng nó nằm trong subspace hình học nhỏ (bài linebreaks) và cần giữ bằng loss riêng khi distill.

---

## 15. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / mô tả | Ý nghĩa |
|---|---|---|
| NLA | AV: $h\to z$ (văn bản), AR: $z\to\hat h$ | Autoencoder có nút thắt ngôn ngữ |
| Loss | $\mathbb E\|h_l-AR(z)\|^2$, $z\sim AV(\cdot\mid h_l)$ | Chỉ tối ưu tái tạo |
| FVE | $1-\mathcal L/\mathbb E\|h-\bar h\|^2$ | Warm-start 0.3–0.4 → 0.6–0.8 |
| Reward AV | $-\log\|h-AR(z)\|^2$ + KL về init | RL (GRPO, $G=8$ ở open model) |
| Cập nhật AR | Hồi quy MSE một bước | Supervised |
| Hệ số chèn | $\alpha\approx Q_{0.75}(\|h\|)$ | Đưa activation vào làm embedding |
| Steering từ NLA | $h+\alpha\|h\|\Delta/\|\Delta\|$, $\Delta=AR(z_\text{edit})-AR(z_\text{orig})$ | Can thiệp bằng sửa văn bản |
| Confabulation | Chủ đề > chi tiết; lặp lại qua token; AR là kiểm chứng yếu | Quy tắc đọc |
| Eval awareness | any-of-50 token, grader 97% đồng thuận | Chỉ số tương đối, không hiệu chỉnh |
| Auditing | Thắng không cần dữ liệu huấn luyện; SAE không giúp | Giá trị downstream |
| AO / PastLens | QA về activation; tái tạo $N\sim U[1,50]$ token trước | Baseline rẻ, cạnh tranh |

### Kết luận ngắn

NLA biến activation thành văn bản đọc được bằng một autoencoder có nút thắt ngôn ngữ, chỉ được huấn luyện để tái tạo, nhưng thực tế cho giải thích giàu thông tin, giúp phát hiện dữ liệu huấn luyện lỗi, evaluation awareness không nói ra và nguyên nhân gốc của một mô hình lệch có chủ đích. Đổi lại là confabulation chi tiết, thiếu nền tảng cơ chế, nhạy layer và chi phí cao. Với nén mô hình OCR, giá trị chính là khái niệm "phần activation diễn đạt được" như một lớp phân tích bổ sung cho $D_\text{lin}/D_\text{res}$, và baseline PastLens rẻ để so teacher–student.

**Đọc bản gốc:** https://transformer-circuits.pub/2026/nla/index.html
