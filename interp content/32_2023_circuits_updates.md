# Circuits Updates — 2023 (May 2023, July 2023)

**Tác giả:** Nhóm interpretability của Anthropic; mỗi mục có tác giả riêng (ghi ở đầu từng mục). Bản May 2023 do Chris Olah, Josh Batson và Shan Carter biên tập (đăng ngày 24/05/2023).  
**Nguồn:**
- May 2023 — https://transformer-circuits.pub/2023/may-update/index.html
- July 2023 — https://transformer-circuits.pub/2023/july-update/index.html

**Chủ đề:** Superposition, dictionary learning, định nghĩa "feature", weight superposition, attention head superposition, feature manifold, memorization trong toy model, danh sách mục tiêu an toàn cho interpretability.  
**Vị trí trong lộ trình:** Đọc sau *Toy Models of Superposition* (2022) và *Superposition, Memorization, and Double Descent* (2023); đây là "nhật ký phòng thí nghiệm" ngay trước *Towards Monosemanticity* (10/2023), cho thấy vì sao nhóm chọn sparse autoencoder để giải superposition.

> Đây là ghi chú học tập diễn giải bằng tiếng Việt, không phải bản dịch. Bản thân các "Circuits Updates" là kết quả sơ bộ — tác giả yêu cầu đọc chúng như một đồng nghiệp chia sẻ ở lab meeting, không phải paper hoàn chỉnh. Các ứng dụng cho pruning/distillation ở cuối là suy luận mở rộng của chúng ta, không phải kết luận của bài.

## Mục lục

1. [May 2023](#1-may-2023)
   - [1.1 Tấn công superposition bằng dictionary learning](#11-tấn-công-superposition-bằng-dictionary-learning)
   - [1.2 Feature là phân rã đơn giản nhất](#12-feature-là-phân-rã-đơn-giản-nhất)
   - [1.3 Những lo ngại về dictionary learning](#13-những-lo-ngại-về-dictionary-learning)
   - [1.4 Fractional dimensionality và "pressure"](#14-fractional-dimensionality-và-pressure)
   - [1.5 Hiện tượng "hai vòng tròn" khi memorization](#15-hiện-tượng-hai-vòng-tròn-khi-memorization)
   - [1.6 Weight superposition](#16-weight-superposition)
   - [1.7 Attention head superposition](#17-attention-head-superposition)
   - [1.8 Toy model feature manifold](#18-toy-model-feature-manifold)
   - [1.9 Điểm tin: bài mới và nghiên cứu nhóm khác](#19-điểm-tin-bài-mới-và-nghiên-cứu-nhóm-khác)
2. [July 2023](#2-july-2023)
   - [2.1 Chế độ trung gian với dữ liệu hữu hạn](#21-chế-độ-trung-gian-với-dữ-liệu-hữu-hạn)
   - [2.2 Feature nào liên quan an toàn nhất?](#22-feature-nào-liên-quan-an-toàn-nhất)
   - [2.3 Mổ xẻ tiếp mô hình skip-trigram](#23-mổ-xẻ-tiếp-mô-hình-skip-trigram)
3. [Cách các update kiểm chứng và giới hạn](#3-cách-các-update-kiểm-chứng-và-giới-hạn)
4. [Ứng dụng giả thuyết cho pruning / distillation và thí nghiệm gợi ý](#4-ứng-dụng-giả-thuyết-cho-pruning--distillation-và-thí-nghiệm-gợi-ý)
5. [Tóm tắt khái niệm](#5-tóm-tắt-khái-niệm)

---

## 1. May 2023

Bối cảnh: sau *Toy Models of Superposition*, nhóm thử nhiều thí nghiệm "ad-hoc" về superposition (*chồng chập: mô hình biểu diễn nhiều feature hơn số chiều bằng các hướng gần trực giao*) trong mô hình thật, kết quả thú vị nhưng không kết luận được. Update này đánh dấu quyết định tập trung vào một hướng có hệ thống.

### 1.1 Tấn công superposition bằng dictionary learning

*Bricken, Batson, Templeton, Jermyn, Hume, Henighan, Olah*

- **Câu hỏi:** Superposition sẽ được "giải" thế nào?
- **Quyết định:** Theo "Approach 2" trong Toy Models — dùng **dictionary learning** (*học từ điển: tìm một tập vector cơ sở sao cho mỗi activation là tổ hợp thưa của chúng*) trên activation của mô hình dense đã huấn luyện. Họ xây hạ tầng để làm việc này ở quy mô lớn, với mục tiêu hoặc tìm được superposition trong mô hình thật, hoặc cập nhật mạnh rằng hướng này không ổn.
- **Quan sát sơ bộ:** phân rã thưa đôi khi cho ra activation gợi ý giả thuyết đơn giản, nhưng chưa có gì thuyết phục rằng đó là "sự thật cơ bản" thay vì một lăng kính tiện lợi.
- **Ý nghĩa:** đây là tiền đề trực tiếp của *Towards Monosemanticity*.

### 1.2 Feature là phân rã đơn giản nhất

*Bricken, Batson, Templeton, Jermyn, Henighan, Olah*

**Câu hỏi kép:** (i) "feature" là gì? (ii) khi chạy dictionary learning, chọn số feature bao nhiêu là đúng, để không bỏ sót và cũng không xé một feature thật thành nhiều mảnh?

**Đề xuất:** định nghĩa feature là **phân rã đơn giản nhất** của activation. Với

$$
\boxed{A = SD}
$$

($S$ là sparse code, $D$ là dictionary), đo **total information** — lượng thông tin cần để biểu diễn cả $S$ và $D$. Dictionary lớn tốn thông tin hơn, code thưa hơn lại tốn ít hơn, hai lực này có thể cân bằng nhau.

Cách đo trong thí nghiệm đầu: mô hình hóa phân phối các phần tử mỗi ma trận bằng histogram 100 bin, lấy surprisal của từng phần tử rồi cộng lại:

$$
I_{\text{total}} = \sum_{\text{entries } a\in S\cup D} -\log p_{\text{hist}}(a).
$$

(Tác giả nói công thức cụ thể sẽ còn thay đổi.)

**Thí nghiệm tổng hợp:** vector thưa chiều `n_sparse` (có `n_active` phần tử khác 0) được chiếu ngẫu nhiên vào không gian 32 chiều; cần khôi phục cấu trúc thưa ban đầu. Dictionary learning (về cơ bản MOD với vòng trong LASSO), quét hệ số L1 và kích thước dictionary, vẽ MSE theo total information.

**Kết quả:** các nghiệm "nảy" (*bounce*) khi kích thước dictionary bằng số factor thật; biên Pareto của biểu đồ rate–distortion này do các nghiệm có đúng số factor chiếm. Đây cũng là chỗ đạt MMCS tốt nhất (metric của Sharkey et al. khi biết đáp án).

**Vì sao quan trọng:** nếu thấy "bounce" trên dữ liệu thật, đó là bằng chứng có "feature thật" tồn tại.

### 1.3 Những lo ngại về dictionary learning

*Henighan, Olah*

Hai cách mà bài toán thật có thể lệch khỏi dictionary learning chuẩn:

| Lo ngại | Nội dung | Hệ quả |
|---|---|---|
| Compressed sensing "quá mạnh" | DL chuẩn cho phép tính $S$ bằng hàm tùy ý; mạng nơ-ron cần lấy feature ra **rẻ** (nếu cần hơn 1 layer thì phí compute và phí chính "chỗ chứa" mà superposition muốn tiết kiệm) | DL có thể "ảo giác" thêm feature (ví dụ cặp feature ghép) giúp code thưa hơn nhưng vô nghĩa với mạng |
| Không zero-centered | Activation có thể gần $A=SD+b$; gợi ý từ bài của Hobbhahn về "shifted superposition" để né ReLU | Thuật toán DL chuẩn hỏng, nhưng dễ sửa |

Kết luận: các lo ngại nghiêng về **sparse autoencoder** (SAE — *encoder tuyến tính + ReLU, decoder tuyến tính, phạt L1*) như Sharkey et al., nhưng SAE mong manh và nhạy hyperparameter hơn. Họ muốn một phương pháp có ưu điểm của cả hai. Ngoài ra còn lo về feature tương quan khó tách.

### 1.4 Fractional dimensionality và "pressure"

*Henighan, Olah*

Toy Models từng thấy feature xếp thành đa diện đều tùy độ thưa. Comment mới cho thấy hình học này nhạy với **feature pressure** (*tỉ lệ số feature mô hình "muốn" biểu diễn trên số chiều có sẵn*) và thời gian huấn luyện: nhiều feature cạnh tranh hơn và train lâu hơn mới cho hình học sạch. Điều khó hiểu: các feature mà mô hình cuối cùng *không* biểu diễn vẫn ảnh hưởng hình học. Hai giả thuyết: chúng bơm nhiễu, hoặc mô hình thực sự dùng "epsilon features" trong superposition.

### 1.5 Hiện tượng "hai vòng tròn" khi memorization

*Henighan, Olah*

Trong toy model memorization với $m=2$ chiều ẩn, đôi khi các điểm dữ liệu xếp trên **hai vòng tròn bán kính khác nhau**. Tác giả cho rằng đây có lẽ là đặc thù tối ưu hóa trong 2D, nhưng là ví dụ hay về hình học của superposition + memorization.

### 1.6 Weight superposition

*Olah*

**Câu hỏi:** superposition thường nói về *feature*; nhưng mạng có thể làm *phép tính* gì khi đang ở trong superposition?

**Ý tưởng:** layer $X$, $Y$ đều ở trong superposition qua $U^X$, $U^Y$ (cột $U_i$ là hướng nhúng của feature $i$). Có "trọng số lý tưởng" $W^*$ giữa feature thật. Yêu cầu $e_i^\top W^* e_j \approx (U^Y_i)^\top W\, U^X_j$ dẫn tới (viết lại theo ký hiệu ma trận của chúng ta):

$$
\boxed{W \approx \sum_{i,j} W^*_{ij}\; U^Y_i \,(U^X_j)^\top}
$$

tức map $U^W = U^X\otimes U^Y$ (tensor product). Nhiễu giữa hai "trọng số lý tưởng" $W^*_{ij}$ và $W^*_{kl}$ do tích Frobenius $\langle U^W_{ij}, U^W_{kl}\rangle_F$ quyết định. (Suy ra thêm, của chúng ta: với tích ngoài, $\langle U^Y_iU^{X\top}_j, U^Y_kU^{X\top}_l\rangle_F=\langle U^Y_i,U^Y_k\rangle\langle U^X_j,U^X_l\rangle$.)

**Điểm thú vị:** weight có thể có **constructive interference** (*nhiễu có lợi*), khác với feature superposition vốn luôn có hại. Đây chỉ là suy nghĩ sơ bộ.

### 1.7 Attention head superposition

*Jermyn, Olah, Henighan*

- **Thuật ngữ:** *attentional feature* — quan hệ giữa một cặp token, ứng với tổ hợp tuyến tính các head cùng attend giữa cặp đó; *attentional circuit* — phép tính tổng thể của head; ở mô hình 1 layer là **skip-trigram** `[A]…[B] → [C]`.
- **Setup:** transformer 1 layer, embedding/unembedding tầm thường, dữ liệu là token ngẫu nhiên đều trừ một số skip-trigram được tăng tần suất.
- **Trọng tâm:** skip-trigram **OV-incoherent** — nhiều token "attend from" khác nhau cùng nhìn về một token nguồn, và output phụ thuộc token "attend from". Một head không làm được nhiều skip-trigram như vậy mà không sai, vì OV circuit không biết đang attend từ đâu.
- **Kết quả:** khi số trigram không tương thích nhiều hơn số head, mô hình đặt chúng **trong superposition qua nhiều head** (mỗi trigram được mã hóa bởi ít nhất hai head).
- **Bài học:** cẩn thận khi nghiên cứu một head riêng lẻ; circuit trong superposition có thể trông gây hiểu lầm nếu chỉ hiểu một phần. (Xem điều chỉnh ở mục 2.3.)

### 1.8 Toy model feature manifold

*Olah, Batson*

**Câu hỏi:** nếu feature nằm trên một *manifold* (*đa tạp: tập feature liên tục, feature gần nhau phản ứng với dữ liệu giống nhau*), mạng biểu diễn nó trực tiếp hay rời rạc hóa? Thực tế thường thấy **họ neuron equivariant** — ví dụ *Curve Detectors* có ~10 neuron, mỗi neuron cho một khoảng hướng cong, thay vì 1 neuron "góc" hay 2 neuron sin/cos (tương tự "value coding" vs "variable coding" trong thần kinh học).

**Setup:** bài toán ReLU-output của Toy Models, nhiều feature xếp đều quanh vòng tròn. Chọn góc $\theta$ và độ lớn $m$; feature tại góc $\phi$:

$$
x_\phi =
\begin{cases}
m\cos\!\left(\frac{\phi-\theta}{\ell}\right) & \text{nếu } \frac{|\phi-\theta|}{\ell}\le\frac{\pi}{2}\\
0 & \text{ngược lại}
\end{cases}
$$

với $\ell$ là length scale mô hình cần phân giải.

**Kết quả:** khi thay đổi $\ell$, quan sát thấy **emergent discretization** (*rời rạc hóa tự phát*). Tác giả gợi ý: có thể "feature" chính là thứ rời rạc hóa tự phát này. Lưu ý: dấu hiệu cho thấy discretization ở length scale nhỏ nhất có thể là lỗi tối ưu.

### 1.9 Điểm tin: bài mới và nghiên cứu nhóm khác

- **Comment mới:** tóm tắt follow-up cho *Mathematical Framework*, đính chính sơ đồ head composition; nhiều replication của Toy Models (Becker-Kahn, Hobbhahn), comment SAE của Sharkey/Braun/Millidge, "Linear Representation in Othello" (Nanda).
- **Bài nhỏ đã đăng:** *Superposition, Memorization, and Double Descent*; *Privileged Bases in the Transformer Residual Stream*; *Superposition, Compositionality, and Distributed Representations*; *Interpretability Dreams*.
- **Nhóm khác (chọn lọc):**
  - *Linear representation hypothesis* (*feature là hướng trong không gian activation*): tranh luận Li et al. ↔ Nanda trên Othello-GPT được coi là bằng chứng mạnh kiểu Popper — giả thuyết dự đoán ngược với bằng chứng lúc đó và rồi được xác nhận. Turner et al. (activation addition) cũng là bằng chứng.
  - Superposition: Sharkey et al. (SAE), Yun et al. (DL trên residual stream), Gurnee et al. (sparse probe), Lindner et al. (compile chương trình vào transformer).
  - Geva et al.: OV của head không chỉ *di chuyển* mà còn *biến đổi* thông tin (đọc nước → ghi thủ đô).
  - Bills et al. (automated interpretability) — tiềm năng mở rộng quy mô, kèm dè dặt: có nên để AI kiểm toán AI?
  - Attribution patching (Nanda): xấp xỉ gradient×activation; coi chừng LayerNorm.
  - Quanta (Michaud et al.), mode connectivity, grokking (Nanda et al., Chughtai et al., Liu et al.), song song thần kinh học (high-low frequency detector ở chuột, V4).

---

## 2. July 2023

### 2.1 Chế độ trung gian với dữ liệu hữu hạn

*Jermyn, Olah, Henighan*

**Câu hỏi:** Trong *Superposition, Memorization, and Double Descent*, toy model 1 layer có 3 chế độ: dataset nhỏ (tới ~1000 mẫu) → memorize; lớn (trên ~10.000) → học feature tổng quát; ở giữa → không rõ. Chế độ giữa đáng lo vì có thể là phản ví dụ cho khung "linear features in superposition".

**Phát hiện:**
1. **Chế độ trung gian là artifact của optimization failure** (do hidden dimension thấp; tác giả không kỳ vọng hiện tượng này phổ biến ở mô hình thực tế hơn). Train với nhiều learning rate, nghiệm memorizing đạt train loss thấp nhất tới tận ~500k mẫu; sau đó nghiệm tổng quát thắng → **chuyển pha bậc nhất** thật quanh 500k mẫu, đạo hàm của train loss tối thiểu gián đoạn.
2. **Hình thái memorization thay đổi khi tiến gần điểm chuyển:** từ ~1k đến 500k mẫu, mô hình chuyển từ nhớ *từng điểm* sang nhớ *cụm điểm tương quan* (dưới ~1k, các mẫu gần như trực giao nên nhớ chính xác được; trên đó, tương quan khiến phải gom cụm theo feature chung). Cụm có chi tiết hình "tam giác".

**Vì sao quan trọng:** củng cố khung linear features in superposition; nhắc rằng "hiện tượng lạ" có thể chỉ là lỗi tối ưu.

### 2.2 Feature nào liên quan an toàn nhất?

*Olah, Jermyn*

Danh sách mục tiêu nghiên cứu, chia hai loại:

| Loại | Mục tiêu | Câu hỏi chính |
|---|---|---|
| Cơ chế giả thuyết | High-level actions | Có feature kiểu "hành động" kéo dài nhiều token, không chỉ hoàn thành từ đa token? |
| | Planning | Mô hình phối hợp chuỗi hành động, mô hình hóa mục tiêu, lập luận công cụ? |
| | Social reasoning | Biểu diễn trạng thái tinh thần/cảm xúc của user; "social planning" để đẩy user về trạng thái nào đó |
| | Personas | Theo dõi persona đang đóng thế nào; RLHF "khóa" persona thì logic đổi persona có bị kích hoạt? |
| | Situational awareness | Mô hình biết mình là mô hình, biết quá trình huấn luyện, biết có thể bị tắt? |
| Hành vi thực nghiệm | Jailbreaks | Vì sao chúng hiệu quả; tìm jailbreak mới bằng interp? |
| | Instrumental reasoning | Sycophancy, né red-teaming, tự bảo tồn — có phải lập luận công cụ hay lý do vô hại? |
| | In-context learning | Circuit nào; induction heads mới là bước đầu |
| | Successful RLHF | RLHF gợi ra năng lực có sẵn hay tạo circuit mới; xóa bao nhiêu từ mô hình pretrain? |

**Vì sao quan trọng:** đây là "bản đồ mục tiêu" mà các paper 2024–2025 (Scaling Monosemanticity, Biology of LLMs, persona, introspection) lần lượt theo đuổi.

### 2.3 Mổ xẻ tiếp mô hình skip-trigram

*Conerly, Jermyn, Olah*

**Câu hỏi:** mô hình ở mục 1.7 thực sự chạy thuật toán gì? Ví dụ OV-incoherent: `[0]…[1] → [4]` và `[0]…[2] → [5]` không thể cùng nằm trong một head.

**Phát hiện:** chỉ **hai head** đủ để mô hình hóa nhiều skip-trigram OV-incoherent với loss bằng 0:
- **Head 0:** attend từ token hiện tại tới chính nó, tăng logit như thể token hiện tại là phần tử thứ hai của trigram (`[1]`→`[4]`, `[2]`→`[5]`).
- **Head 1:** attend tới token mở đầu chung `[0]` nếu có mặt (khi đó không làm gì); nếu không có, attend tới token hiện tại và **ghi âm logit của head 0**.

$$
\text{logit} = \underbrace{h_0}_{\text{dự đoán}} + \underbrace{h_1}_{=\,0 \text{ nếu có }[0],\; -h_0 \text{ nếu không}}
$$

Kết quả: một **phép điều kiện** — có `[0]` thì ra đúng logit trigram; không có thì hai head triệt tiêu, output đều. Thực tế hai head không tách sạch mà chia nhau các trigram.

**Điều chỉnh quan trọng:** tác giả không còn chắc ví dụ này nên gọi là "attention head superposition" theo nghĩa chặt (feature là hướng trong activation space).

---

## 3. Cách các update kiểm chứng và giới hạn

**Loại bằng chứng:** gần như toàn bộ là **toy model** và dữ liệu tổng hợp có đáp án biết trước (MMCS, rate–distortion), cộng với quét hyperparameter (learning rate, L1, kích thước dictionary) để phân biệt hiện tượng thật với lỗi tối ưu. Số liệu có trong nguồn: không gian 32 chiều cho thí nghiệm DL; histogram 100 bin; ngưỡng ~1k / ~10k / ~500k mẫu cho memorization; ~10 curve neuron.

**Giới hạn (tác giả nêu):**
- Kết quả sơ bộ, "có thể sai".
- Phân rã thưa có thể chỉ là lăng kính tiện lợi; DL có thể ảo giác feature; SAE nhạy hyperparameter.
- Discretization ở length scale nhỏ và "hai vòng tròn" có thể là lỗi tối ưu 2D.
- Attention head superposition có thể không đúng nghĩa chặt (mục 2.3).

**Giới hạn (của chúng ta):** chưa có kết quả trên mô hình ngôn ngữ thật; chưa có phép đo nhân quả.

---

## 4. Ứng dụng giả thuyết cho pruning / distillation và thí nghiệm gợi ý

> **Phần này là suy luận của chúng ta, không phải kết luận của các update.**

**Mục liên quan nhất:** 1.6 (weight superposition), 1.7 + 2.3 (head superposition / triệt tiêu), 1.2–1.3 (dictionary learning), 1.8 (feature manifold).

1. **Vì sao cắt MLP neuron khó phục hồi hơn cắt layer.** Nếu feature nằm trong superposition, mỗi neuron là một hướng mà *nhiều* feature cùng chiếu lên; weight cũng ở dạng $\sum W^*_{ij}U^Y_iU^{X\top}_j$. Cắt một neuron = xóa một tọa độ của mọi $U_i$ cùng lúc, làm thay đổi cả các tích $\langle U_i,U_k\rangle$ mà mạng đã sắp xếp để nhiễu nhỏ (hoặc nhiễu có lợi). Một map tuyến tính tại chỗ cắt không khôi phục được thông tin đã mất qua ReLU → phần lớn rơi vào **D_res**. Ngược lại, một layer gần-tuyến-tính trong residual stream có thể được thay bằng map tuyến tính → **D_lin** lớn. Đây là giả thuyết khớp với quan sát của bạn, chưa được chứng minh.
2. **Cắt head theo cặp.** Mục 2.3 cho ví dụ hai head mà output của head này triệt tiêu head kia. Importance tính riêng từng head có thể đánh giá sai: cắt head 1 khiến head 0 "phát ngôn" sai khi không có `[0]`. Với attention đầy đủ trong LightOnOCR, nên đo importance theo **cặp/nhóm head** (joint ablation), không chỉ đơn lẻ.
3. **Hidden-state KD theo hướng feature.** Nếu feature là hướng (linear representation), KD bằng MSE trên toàn hidden state trừng phạt đều mọi hướng; có thể KD trên **không gian SAE/dictionary** của teacher để ưu tiên các hướng có nghĩa.
4. **OCR và feature manifold.** Vị trí ký tự, cỡ chữ, hướng dòng là biến liên tục; mục 1.8 gợi ý mạng có thể rời rạc hóa chúng thành họ neuron equivariant. Cắt vài neuron trong họ có thể tạo "lỗ" trên manifold (ví dụ một khoảng vị trí/cỡ chữ bị hỏng) thay vì suy giảm đều.

**Thí nghiệm gợi ý:**
- **E1 — Superposition và D_res:** trên một layer MLP của decoder, đo D_lin/D_res khi cắt (a) neuron có activation thưa/đa nghĩa và (b) neuron dense. Dự đoán: neuron đa nghĩa (nhiều feature chiếu lên) có D_res cao hơn.
- **E2 — Joint head ablation:** trên các layer full-attention, tính ma trận tương tác $\Delta L(i,j) - \Delta L(i) - \Delta L(j)$ cho cặp head. Cặp có tương tác lớn thì cắt/giữ cùng nhau.
- **E3 — "Bounce" để chọn số feature:** huấn luyện SAE nhỏ với nhiều kích thước dictionary trên activation một layer ViT; vẽ MSE theo total information xem có "bounce" không, dùng làm gợi ý số chiều bottleneck khi distill.

---

## 5. Tóm tắt khái niệm

| Khái niệm | Công thức / mô tả | Ý nghĩa |
|---|---|---|
| Dictionary learning | $A=SD$ ($S$ thưa) | Tách feature khỏi superposition |
| Total information | Tổng surprisal phần tử của $S$ và $D$ | Chọn số feature "đúng" (điểm bounce) |
| Shifted superposition | $A=SD+b$ | Activation không zero-centered, né ReLU |
| Feature pressure | #feature mong muốn / #chiều | Quyết định hình học superposition |
| Weight superposition | $W\approx\sum W^*_{ij}U^Y_iU^{X\top}_j$ | Phép tính trong superposition; có thể nhiễu có lợi |
| OV-incoherent skip-trigram | Nhiều nguồn "from", cùng "to", output phụ thuộc "from" | Một head không làm nổi; cần nhiều head |
| Head triệt tiêu | $h_1=-h_0$ khi thiếu token mở đầu | Phép điều kiện bằng hai head |
| Emergent discretization | Manifold → họ neuron rời rạc | Ứng viên cho định nghĩa "feature" |
| Chuyển pha memorization | ~500k mẫu trong toy model | Chế độ giữa trước đây là lỗi tối ưu |

### Kết luận ngắn

Năm 2023 là năm nhóm chuyển từ lý thuyết superposition sang **công cụ** để giải nó: dictionary learning/SAE, cùng câu hỏi nền tảng "feature là gì". Các toy model về weight superposition, head superposition và feature manifold cho trực giác vì sao các đơn vị vật lý (neuron, head) không phải đơn vị tính toán đúng — một trực giác hữu ích khi giải thích vì sao cắt neuron gây D_res lớn. Danh sách mục tiêu an toàn của July 2023 là bản đồ cho các nghiên cứu sau đó.

**Đọc bản gốc:**
- https://transformer-circuits.pub/2023/may-update/index.html
- https://transformer-circuits.pub/2023/july-update/index.html
