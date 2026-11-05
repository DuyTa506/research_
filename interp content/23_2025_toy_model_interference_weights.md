# A Toy Model of Interference Weights (2025)

**Tác giả:** Chris Olah, Nicholas L. Turner, Tom Conerly (Anthropic). Công bố ngày 29/07/2025, dạng ghi chú không chính thức.  
**Nguồn:** https://transformer-circuits.pub/2025/interference-weights/index.html  
**Chủ đề:** Weight superposition, interference weights, virtual weights "downstairs" và "upstairs", định nghĩa và heuristic tách trọng số thật khỏi trọng số nhiễu.  
**Vị trí trong lộ trình:** Đọc sau *Toy Models of Superposition* (2022) và *Towards Monosemanticity* (2023, logit weight của feature). Nên đọc trước note số 24 (*Characterizing interference weights in a tiny language model*, 2026), nơi ý tưởng này được kiểm chứng trên transformer thật. Tác giả cho rằng interference weights có thể là **nút thắt** ngăn phân tích mạch toàn cục; tránh chúng là lý do chính để dùng attribution graph theo từng prompt.

> Ghi chú diễn giải bằng tiếng Việt, không phải bản dịch. Bản thân bài gốc là ghi chú rất sơ bộ, tác giả yêu cầu đọc với độ tin cậy thấp. Phần ứng dụng cho pruning/distillation là suy luận của chúng tôi.

## Mục lục

1. [Weight superposition là gì](#1-weight-superposition-là-gì)
2. [Có nên quan tâm tới interference weights](#2-có-nên-quan-tâm-tới-interference-weights)
3. [Đọc lại toy model như một mạch đồng nhất](#3-đọc-lại-toy-model-như-một-mạch-đồng-nhất)
4. [Hai định nghĩa interference weights](#4-hai-định-nghĩa-interference-weights)
5. [Tái hiện hiện tượng của Towards Monosemanticity](#5-tái-hiện-hiện-tượng-của-towards-monosemanticity)
6. [Toy model phức tạp hơn](#6-toy-model-phức-tạp-hơn)
7. [Heuristic tách trọng số thật](#7-heuristic-tách-trọng-số-thật)
8. [Phụ lục đáng chú ý](#8-phụ-lục-đáng-chú-ý)
9. [Cách paper kiểm chứng](#9-cách-paper-kiểm-chứng)
10. [Giới hạn](#10-giới-hạn)
11. [Ứng dụng giả thuyết cho pruning và distillation](#11-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
12. [Thí nghiệm gợi ý](#12-thí-nghiệm-gợi-ý)
13. [Tóm tắt công thức và khái niệm](#13-tóm-tắt-công-thức-và-khái-niệm)

---

## 1. Weight superposition là gì

Khi nghĩ về superposition, ta thường nghĩ tới việc **feature** được sắp xếp chồng lên nhau trong không gian ít chiều. Nhưng có mặt thứ hai dễ bị bỏ qua: nếu hai layer đều có feature trong superposition, **trọng số giữa các feature** cũng bị ép vào superposition.

- **Downstairs** ("tầng dưới"): mô hình thật với không gian ít chiều.
- **Upstairs** ("tầng trên"): mô hình lý tưởng hóa, mỗi feature một chiều riêng, có ma trận trọng số feature–feature đầy đủ.
- **Lift** (nâng lên): biểu diễn trọng số downstairs dưới dạng trọng số giữa các feature upstairs, tức **virtual weights**.

Khi "lift", nhiều virtual weight không phản ánh mạch mô hình muốn có mà chỉ là **hệ quả hình học** của việc nén. Đó là **interference weights** (trọng số nhiễu).

---

## 2. Có nên quan tâm tới interference weights

Chúng là "thật" theo nghĩa mô tả đúng kết nối trong mô hình. Có giả thuyết chúng là một nguyên nhân của **adversarial examples**. Nhưng về bản chất là nhiễu: mô hình "không muốn" chúng; chúng làm loss tệ hơn hoặc không giúp gì. Nếu finetune trực tiếp mô hình đã lift (không khả thi trong thực tế), chúng biến mất. Tác giả lưu ý điều này đơn giản hóa: feature không hoàn toàn monosemantic, gradient có thể tạo superposition mới, mô hình có thể học trọng số mới. Thực tế có thể học một mask trên virtual weights với penalty nhỏ (theo hướng của Drori, 2025).

| Mục tiêu an toàn | Interference weights có quan trọng không? |
|---|---|
| **Robustness** (môi trường đối kháng, input xấu) | **Có**: kẻ tấn công có thể khai thác chúng |
| **Alignment** (quá trình tối ưu/học đối kháng) | **Có lẽ không**: theo định nghĩa chúng không giúp mục tiêu tối ưu |

Vì alignment là ưu tiên của interpretability, tác giả muốn tách "trọng số thật" để phân tích, nhưng vẫn giữ interference weights trong tầm mắt. Tác giả tự nhận chưa hoàn toàn tin luận điểm "có thể bỏ qua cho alignment", và cho rằng có thể đây là câu hỏi quyết định việc phân tích cơ chế toàn cục có khả thi hay không.

---

## 3. Đọc lại toy model như một mạch đồng nhất

Toy model gốc:

$$
h=Wx,\qquad x'=\operatorname{ReLU}(W^\top h+b)=\operatorname{ReLU}(W^\top W x+b).
$$

Hai cách đọc: (i) như trong *Toy Models*: hình học của việc mã hóa feature; (ii) mới: một **mạch đồng nhất một layer** ($y=x$) bị đặt vào superposition. Đối tượng nghiên cứu là virtual weights downstairs:

$$
\boxed{\;U=W^\top W\;}
$$

Mạch lý tưởng là $U^*=I$ (mỗi feature chỉ nối với chính nó). Mọi phần tử ngoài đường chéo khác 0 là ứng viên interference.

**Ví dụ nhỏ (của chúng tôi).**

*Antipodal:* 2 feature trong 1 chiều, $W=[1,-1]$:

$$
U=W^\top W=\begin{pmatrix}1&-1\\-1&1\end{pmatrix}.
$$

Hai số $-1$ là interference: feature 1 "ức chế" feature 2 không vì lý do tính toán nào mà vì cùng chia một chiều. ReLU cắt được phần âm khi chỉ một feature bật (feature thưa), nên mô hình chấp nhận nhiễu này.

*Tam giác:* 3 feature trong 2 chiều, cột của $W$ cách nhau $120^\circ$, $U_{ij}=\cos120^\circ=-0.5$ với $i\ne j$. Bias âm cỡ $-0.5$ đủ để dập nhiễu khi chỉ một feature hoạt động.

---

## 4. Hai định nghĩa interference weights

**Ideal weights** $U^*$: trọng số thu được nếu finetune ma trận feature–feature trong không gian lift theo loss huấn luyện, với một regularization nhỏ để xóa trọng số mà loss không quan tâm. Tác giả nhấn mạnh rất khó vận hành ngoài toy model.

$$
\textbf{Định nghĩa 1:}\quad U_{\text{interf}}=U-U^*,\qquad U=U^*+(U-U^*).
$$

**Loss contribution** của một trọng số: $\Delta L(U_{ij})$ = loss khi ablate trọng số đó − loss gốc.

$$
\textbf{Định nghĩa 2:}\quad U_{ij}\ \text{là interference nếu}\ \Delta L(U_{ij})<\epsilon .
$$

Định nghĩa 2 cho phân rã $U=U_{\text{real}}+U_{\text{interf}}$ theo tập phần tử. Hai định nghĩa thực sự khác nhau (xem §8). Bài tập trung vào $\Delta L$.

---

## 5. Tái hiện hiện tượng của Towards Monosemanticity

**Vì sao Towards Monosemanticity phải thấy interference.** Paper đó dùng SAE trên output MLP của transformer một layer và xem **logit weight** của feature: đường từ feature → output MLP → residual → logit. Trên đường này feature bị nén hai lần (output MLP, rồi dày hơn ở residual), nên khi khai triển phải có interference.

**Toy model:** 100 feature, 20 chiều, mật độ feature 0.02, **huấn luyện chưa đủ** (chưa hội tụ, khớp hiện tượng thật hơn).

| Hiện tượng | Toy model | Towards Monosemanticity |
|---|---|---|
| Histogram trọng số | Khối lớn quanh 0 (tô màu theo $\Delta L$: phần lớn là interference), đuôi là trọng số thật | Histogram logit weight của feature tiếng Hebrew có hình dạng tương tự |
| Scatter hai seed | Interference **độc lập** giữa hai mô hình; trọng số thật **đều dương lớn** ở cả hai | Scatter logit weight giữa hai mô hình (do Josh Batson phát hiện) giống vậy |

Với mô hình **hội tụ đầy đủ**: interference có cấu trúc hơn, trọng số thật **trùng khớp hoàn toàn** giữa hai seed, interference vẫn không.

$$
\text{seed A vs seed B:}\quad U^{A}_{\text{real}}\approx U^{B}_{\text{real}},\qquad
\operatorname{corr}\big(U^{A}_{\text{interf}},U^{B}_{\text{interf}}\big)\approx 0 .
$$

Đây là một "chữ ký" thực nghiệm hữu ích: trọng số thật ổn định qua seed, nhiễu thì không.

---

## 6. Toy model phức tạp hơn

Toy model đơn giản có vấn đề: trọng số thật và nhiễu tách bạch theo độ lớn. Nhưng feature base64 trong Towards Monosemanticity cho thấy chúng **chồng lấn**: đó chính là lý do bài toán khó. Nếu chỉ cần bỏ trọng số nhỏ thì đã dễ.

Mạch mục tiêu tổng quát hơn:

$$
y=\operatorname{ReLU}(Ax+v),\qquad h=W_{\text{down}}x,\qquad y'=\operatorname{ReLU}(W_{\text{up}}h+b),
$$

với $A$ ngẫu nhiên (không đối xứng, nên phải "untie" hai ma trận). Virtual weights là $U=W_{\text{up}}W_{\text{down}}$, lý tưởng gần $A$.

Khó tìm chế độ thỏa đồng thời: (1) thật và nhiễu chồng lấn mạnh; (2) hai seed không hội tụ về cùng một cấu hình (scatter sụp); (3) hai seed không chia thành hai lời giải với lựa chọn nhị phân khác nhau; (4) vẫn giữ khi huấn luyện tới hội tụ. Đạt được (1)–(3), chưa đạt (4), bằng $A$ **khối chéo**, mỗi khối thưa (xác suất 0.5), giá trị đều trong $[0,1]$; $v=-0.1$. Cấu trúc khối giúp nhiều cho (2).

Thí nghiệm: 128 feature trong 16 chiều, 8 khối, mật độ trọng số trong khối 0.1, mật độ feature đầu vào 0.3. Kết quả: scatter hai seed, histogram một mô hình, và so sánh trọng số học được với ideal weights, trong đó trọng số thật và nhiễu chồng lấn rõ.

---

## 7. Heuristic tách trọng số thật

Cả hai định nghĩa đều đắt: định nghĩa 1 cần vật chất hóa ma trận $n_{\text{features}}^2$ (CLT hiện có hàng chục triệu feature, có thể cần hàng tỉ) rồi tối ưu trên dữ liệu lớn; định nghĩa 2 cần ablate từng trong $n_{\text{features}}^2$ trọng số. Với mô hình nhỏ có thể brute force.

Các heuristic rẻ được so sánh:

| Heuristic | Ý tưởng |
|---|---|
| Original weight | Trọng số lớn có lẽ là thật |
| Expected attribution | Ảnh hưởng trung bình lớn có lẽ là thật |
| Target weighted attribution | Ảnh hưởng lớn lên thứ quan trọng |
| Frequency | Trọng số hay "làm gì đó" |
| Baseline lý tưởng | Hiệu ứng loss thật $\Delta L$ |

Ground truth: trọng số thật nếu $\Delta L(U_{ij})>\epsilon=0.0001$. Đánh giá bằng **precision–recall** cho phân loại nhị phân. Nhưng recall không phải thứ ta thật sự quan tâm (mất trọng số quan trọng tệ hơn nhiều so với mất trọng số phụ), nên tác giả chuyển sang **"loss gain" vs precision**, với

$$
\text{loss gain}(S)=\sum_{(i,j)\in S}\Delta L(U_{ij})
$$

(cộng từng trọng số, không đánh giá mô hình đã ablate đồng thời). Theo tác giả, cách đo này hứa hẹn hơn nhiều.

Ý tưởng **compressed sensing** (khôi phục ma trận lớn từ bản nén) hấp dẫn nhưng giả định ánh xạ nén tuyến tính, có thể không đúng (§8).

---

## 8. Phụ lục đáng chú ý

**Chia sẻ trọng số phi tuyến.** Thường giả định trọng số downstairs là *tổng* của các phần upstairs (parameter linearity, như trong Attribution-based Parameter Decomposition). Tác giả nghi nó giống **cực đại từng điểm** hơn trong một số trường hợp. Ví dụ: hai feature đều cần nhân 2, nén vào một chiều theo kiểu antipodal; cả hai dùng chung **một** trọng số 2. Nếu không antipodal, trọng số chỉ được chia sẻ một phần.

**Hội tụ đầy đủ, superposition đẳng hướng** (mọi feature như nhau, nhiều hơn khả năng biểu diễn): mật độ 0.25 cho cấu hình **ngũ giác**, góc $\tfrac25\pi$ và $\tfrac45\pi$; mật độ rất thấp 0.001 không có hình học sạch nhưng có một **cận trên** cho interference.

**Feature tương quan.** Hai tập 100 feature, chỉ một tập hoạt động mỗi lần, chọn 2 feature ngẫu nhiên với activation đều trong $[0,1]$. Trọng số liên tập có phương sai lớn hơn (có giá trị âm rất lớn). Có cấu trúc **low-rank thô**: cùng tập kích thích nhau, khác tập ức chế nhau; tách được thành một thành phần riêng, và ablate nó làm loss tệ đi. Phân rã đề xuất:

$$
U=\underbrace{U_{\text{low-rank}}}_{\text{cấu trúc tương quan}}+\underbrace{U_{\text{res,+}}}_{\text{giúp loss}}+\underbrace{U_{\text{res,0}}}_{\text{không giúp}} .
$$

Hệ quả: (1) có thể phải nghĩ về nhiều hơn $n^2$ "trọng số"; (2) "trọng số có tốt cho loss không" là câu hỏi phức tạp: thành phần low-rank giúp *vì nó giảm hiệu ứng của interference*; (3) đóng góp loss phi tuyến, gây xung đột giữa các định nghĩa.

**Khác biệt giữa hai định nghĩa.** (a) Định nghĩa 2 ablate từng trọng số riêng lẻ, dính mọi vấn đề credit assignment; mỗi trọng số có thể chứa nhiều nhiễu cộng một ít tín hiệu, nên ablate một cái thì lợi (bỏ nhiễu), ablate nhiều cái thì hại (mất tín hiệu, có thể bậc hai). (b) Định nghĩa 1 lấy $U^*$ làm "thật": nếu mô hình chọn *không* biểu diễn một trọng số để tránh nhiễu, định nghĩa 1 nói giá trị thật là giá trị lý tưởng bị che bởi nhiễu âm lớn, còn định nghĩa 2 nói giá trị thật là 0. Học mask trên virtual weights (Drori) là biến thể tự nhiên của định nghĩa 1 và là hướng tác giả sẽ chọn nếu theo đuổi.

---

## 9. Cách paper kiểm chứng

- **Đối chiếu hiện tượng:** histogram và scatter hai seed của toy model so với hình logit weight của Towards Monosemanticity. Đây là bằng chứng định tính, dựa trên hình dạng.
- **Ground truth trong toy model:** tính $\Delta L$ cho từng trọng số, nên biết chắc trọng số nào là nhiễu.
- **So sánh heuristic:** precision–recall và loss gain vs precision với ngưỡng $\epsilon=10^{-4}$.
- Không có thí nghiệm trên mô hình ngôn ngữ thật; đó là việc của note số 24.

---

## 10. Giới hạn

- Rất sơ bộ, "low confidence throughout" theo chính tác giả.
- Toy model đặc biệt (đồng nhất, hoặc khối chéo); chưa đạt chế độ chồng lấn mạnh *và* hội tụ đầy đủ cùng lúc.
- Các định nghĩa được chọn vì dễ giảng giải, không phải vì tác giả tin là lý tưởng.
- Định nghĩa principled không scale; heuristic chưa được kiểm chứng trên mô hình lớn.
- Giả định nén tuyến tính có thể sai; hiệu ứng ablate nhiều trọng số phi tuyến.

---

## 11. Ứng dụng giả thuyết cho pruning và distillation

> **Suy luận của chúng tôi.**

**11.1 Student nhỏ hơn thì nhiễu trọng số nhiều hơn.** Pruning chiều rộng (hidden, intermediate MLP) giảm số chiều downstairs trong khi số feature cần biểu diễn gần như giữ nguyên. Theo toy model, mật độ superposition tăng và interference weights tăng. Đây là một cơ chế cụ thể cho phần **D_res**: kể cả khi student "giữ được" feature, các trọng số feature–feature mới sinh ra do nén chặt hơn là phi tuyến (bị ReLU/SiLU cắt theo ngữ cảnh), không bù được bằng một map tuyến tính.

**11.2 Neuron cut vs layer cut.** Cắt neuron MLP là giảm trực tiếp chiều của một bottleneck: feature phải nén lại, interference tăng. Cắt layer không giảm chiều residual: superposition ở residual giữ nguyên, chỉ mất một số mạch. Đây là một giả thuyết bổ sung (khác với lập luận đường tắt ở note 19) cho việc cắt neuron khó phục hồi hơn.

**11.3 Magnitude không đáng tin.** Ở toy model phức tạp, trọng số thật và nhiễu chồng lấn về độ lớn. Tiêu chí pruning dựa trên độ lớn (magnitude) sẽ giữ một số nhiễu lớn và bỏ một số trọng số thật nhỏ. Tiêu chí dựa trên loss (loss gain, giống Optimal Brain Damage) phù hợp hơn với "trọng số thật".

**11.4 Robustness của student.** Nếu interference weights liên quan tới adversarial example, student bị nén mạnh hơn có thể dễ vỡ hơn với ảnh lạ (nhiễu scan, font hiếm) dù CER trong phân phối tương đương. Nên có test OOD riêng.

**11.5 Seed scatter như công cụ chẩn đoán KD.** Chữ ký "trọng số thật ổn định qua seed, nhiễu thì không" gợi ý: distill hai student khác seed, so virtual weights (ví dụ feature teacher → logit qua student). Phần ổn định là phần student thực sự học từ teacher; phần không ổn định là nhiễu do nén.

---

## 12. Thí nghiệm gợi ý

**Thí nghiệm 1 — Tái hiện và "prune" toy model.** Huấn luyện toy model $n=100$, $m=20$, mật độ 0.02. Sau đó giảm $m$ xuống 15, 10 (cắt chiều rồi finetune). Đo: tỉ lệ trọng số có $\Delta L<\epsilon$, tổng $|U_{\text{interf}}|$, và phần loss phục hồi được bằng một map tuyến tính tối ưu ở bottleneck (tương tự D_lin của bạn). Dự đoán: phần không phục hồi tăng nhanh khi $m$ giảm.

**Thí nghiệm 2 — Hai seed student.** Distill hai student 0.1B với seed khác nhau từ cùng teacher. Lấy một tập hướng cố định ở residual cuối của teacher (ví dụ top PC, hoặc feature SAE nếu có), chiếu qua projector sang mỗi student, tính virtual weight tới unembedding. Scatter hai student: phần trùng khớp = "thật", phần phân tán = nhiễu. So với scatter của hai student 0.3B.

**Thí nghiệm 3 — Tiêu chí pruning.** Trên một MLP của teacher, so ba tiêu chí cắt neuron: magnitude, expected attribution (trung bình $|a_i\cdot\partial L/\partial a_i|$), và loss gain xấp xỉ bậc hai. Đo D_res sau khi fit map tuyến tính ở điểm cắt.

---

## 13. Tóm tắt công thức và khái niệm

| Khái niệm | Công thức / mô tả | Ý nghĩa |
|---|---|---|
| Toy model | $x'=\operatorname{ReLU}(W^\top Wx+b)$ | Mạch đồng nhất trong superposition |
| Virtual weights | $U=W^\top W$ hoặc $W_{\text{up}}W_{\text{down}}$ | Trọng số feature–feature hiệu dụng |
| Ideal weights | $U^*$ (finetune trong không gian lift) | "Mạch mô hình muốn có" |
| Định nghĩa 1 | $U-U^*$ | Nhiễu = độ lệch khỏi lý tưởng |
| Định nghĩa 2 | $\Delta L(U_{ij})<\epsilon$ | Nhiễu = không giúp loss |
| Loss gain | $\sum_{S}\Delta L(U_{ij})$ | Đánh giá heuristic theo tầm quan trọng |
| Chữ ký hai seed | Thật trùng, nhiễu độc lập | Chẩn đoán thực nghiệm |
| Chia sẻ phi tuyến | Antipodal dùng chung một trọng số | Nén không nhất thiết là tổng tuyến tính |
| Robustness vs alignment | Quan trọng / có lẽ không | Lý do muốn tách hai loại trọng số |

### Kết luận ngắn

Khi feature nằm trong superposition, trọng số giữa chúng cũng vậy, và việc "lift" trọng số tạo ra nhiều kết nối không có ý nghĩa chức năng. Ghi chú này dựng toy model tái hiện hiện tượng đã thấy trong Towards Monosemanticity, đưa ra hai định nghĩa (độ lệch khỏi lý tưởng, đóng góp loss) và cho thấy heuristic rẻ có thể xấp xỉ chúng. Với nén mô hình, đây là ngôn ngữ chính xác cho việc "nén làm sinh ra nhiễu trọng số mà map tuyến tính không sửa được".

**Đọc bản gốc:** https://transformer-circuits.pub/2025/interference-weights/index.html
