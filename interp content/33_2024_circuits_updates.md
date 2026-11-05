# Circuits Updates — 2024 (January → September 2024)

**Tác giả:** Nhóm interpretability của Anthropic; mỗi mục có tác giả riêng (ghi ở đầu mục). Không có một danh sách tác giả chung.  
**Nguồn (8 update, theo thứ tự thời gian):**
- January 2024 — https://transformer-circuits.pub/2024/jan-update/index.html
- February 2024 — https://transformer-circuits.pub/2024/feb-update/index.html
- March 2024 — https://transformer-circuits.pub/2024/march-update/index.html
- April 2024 — https://transformer-circuits.pub/2024/april-update/index.html
- June 2024 — https://transformer-circuits.pub/2024/june-update/index.html
- July 2024 — https://transformer-circuits.pub/2024/july-update/index.html
- August 2024 — https://transformer-circuits.pub/2024/august-update/index.html
- September 2024 — https://transformer-circuits.pub/2024/september-update/index.html

**Chủ đề:** Kỹ thuật huấn luyện sparse autoencoder (SAE), đánh giá SAE, dùng feature để tìm circuit, self-repair, attention superposition, định nghĩa linear representation, "dark matter", successor heads.  
**Vị trí trong lộ trình:** Đọc sau *Towards Monosemanticity* (2023); đọc song song với *Scaling Monosemanticity* (05/2024). Đây là "bếp" kỹ thuật phía sau các paper lớn 2024 và là cầu nối tới *Circuit Tracing / Biology of LLMs* (2025). Nhiều mục bị chính tác giả đính chính ở update sau — đọc theo thứ tự thời gian.

> Ghi chú học tập diễn giải bằng tiếng Việt, không phải bản dịch. Circuits Updates là kết quả sơ bộ, tác giả yêu cầu đọc như chia sẻ ở lab meeting. Phần ứng dụng cho pruning/distillation ở cuối là suy luận của chúng ta, không phải kết luận của các bài.

## Mục lục

1. [January 2024](#1-january-2024)
2. [February 2024](#2-february-2024)
3. [March 2024](#3-march-2024)
4. [April 2024](#4-april-2024)
5. [June 2024](#5-june-2024)
6. [July 2024](#6-july-2024)
7. [August 2024](#7-august-2024)
8. [September 2024](#8-september-2024)
9. [Cách kiểm chứng và giới hạn chung](#9-cách-kiểm-chứng-và-giới-hạn-chung)
10. [Ứng dụng giả thuyết cho pruning / distillation và thí nghiệm gợi ý](#10-ứng-dụng-giả-thuyết-cho-pruning--distillation-và-thí-nghiệm-gợi-ý)
11. [Tóm tắt khái niệm](#11-tóm-tắt-khái-niệm)

**Thuật ngữ dùng xuyên suốt:**
- **SAE** (*sparse autoencoder*): $f=\mathrm{ReLU}(W_e x+b_e)$, $\hat x=W_d f+b_d$, phạt độ thưa của $f$. Mỗi cột $W_{d,i}$ là một "feature direction".
- **L0**: số feature hoạt động trung bình mỗi token. **MSE**: lỗi tái tạo. **Loss recovered**: tỉ lệ cross-entropy của LLM được giữ khi thay activation bằng bản tái tạo.
- **Dead feature**: feature hầu như không bao giờ bật. **Shrinkage**: phạt L1 khiến activation nhỏ hơn giá trị "thật".
- **Ablation / patching**: thay activation của một thành phần (bằng 0, bằng trung bình, hoặc bằng activation từ prompt khác) rồi đo thay đổi output.

---

## 1. January 2024

Sau *Towards Monosemanticity*, nhóm công bố ba ưu tiên: (1) mở rộng quy mô và khoa học của dictionary learning, (2) tấn công attention superposition trong mô hình thật bằng dictionary learning, (3) hiểu circuit dựa trên hai điều trên.

### 1.1 Attention superposition: hình học giống Toy Models

*Jermyn, Olah, Conerly* — **Câu hỏi:** nhiều "attentional feature" thật có xếp vào ít head hơn theo hình học giống feature superposition không? **Setup:** 5 head "thật" với attention pattern và OV ngẫu nhiên, rất thưa; mục tiêu là output sau ReLU. Mô hình học một ma trận trộn 5 pattern/5 OV thành 2 head. **Kết quả:** scatter plot của ma trận trộn cho hình học nổi bật, gợi nhớ Toy Models. Điều kiện quan trọng: **OV phải thưa**; OV dense thì không thấy superposition.

### 1.2 Dictionary learning trên MNIST

*Jermyn, Templeton, Batson, Bricken, Olah* — Mạng MLP ReLU 16–64 neuron/layer, 1–3 layer. Feature tìm được dễ đọc và gắn với chữ số, thường **nhiều feature cho các cách viết khác nhau của một chữ số** (chiến lược *template matching*), có logit downstream khớp. Không thấy feature thành phần (cạnh, nét, vòng) như conv net. Nhưng: không thấy "elbow" trong reconstruction error; để lỗi thấp cần số feature cỡ kích thước dataset hoặc nhiều feature bật cùng lúc. Bài học: tìm được feature *dễ đọc* chưa chứng minh là *feature thật*; câu hỏi "strong superposition hypothesis" vẫn mở.

### 1.3 Feature trong mô hình 8 layer

*Jermyn, Conerly, Bricken, Templeton* — DL trên MLP activation. Layer cuối giống mô hình 1 layer (token đơn, dự đoán token kế). **Layer 4–6 chứa feature trừu tượng nhất** (cảm xúc, bạo lực, xung đột, "?" trong câu hỏi xã hội), nhất quán với *SoLU*. Layer 7 có LaTeX, Hebrew, Swedish, Base62, spam. Tác giả chọn mẫu thiên về feature liên quan an toàn, độ tin "vừa phải", "như ghi chép thực địa của nhà sinh học".

### 1.4 Feature đa ngôn ngữ trong mô hình lớn

*Templeton et al.* — DL đã mở rộng tới mô hình hàng tỉ tham số; nhiều feature **đa ngôn ngữ**: "uniqueness / the only way" (Anh, Tây Ban Nha, Bồ Đào Nha, Nga, Đức), "hai", "buồn", "tại sao", văn bản tòa án… Đây là tiền đề của kết quả multilingual trong *Scaling Monosemanticity* và *Biology of LLMs*.

### 1.5 Ghost Grads (sau này bị rút lại)

*Jermyn, Templeton* — Mẹo hồi sinh dead neuron trong SAE: forward lần 2 chỉ với neuron chết, thay ReLU bằng **exp** (dương, gradient dương), scale output về ½ norm residual, tính MSE với residual tái tạo rồi rescale ngang reconstruction loss. Ý tưởng: kéo neuron chết về hướng giải thích phần dư. Gấp đôi compute. Ngưỡng "chết": không bật trong 2e6 token. *Ghi chú của bài:* quan điểm đã đổi ở March 2024 (mục 3.2).

### 1.6 Phản ví dụ trong superposition

*Chen, Olah* — Vì sao nhóm không ép mô hình gốc thưa để có neuron monosemantic:
1. **Joint superposition giữa MLP neuron và residual stream:** một neuron rất thưa bật cho 3 ngữ cảnh ("it|'s", "and|/", "many| of"); không neuron nào khác phân biệt, nhưng output phân biệt được bằng cách đọc neuron *cùng với* residual stream. Đây là một dạng **cross-layer superposition**.
2. **Biểu diễn theo độ lớn (magnitude-based):** cắt residual stream, chỉ một neuron bật mỗi lúc vẫn đa nghĩa: bật yếu → đại từ nhân xưng, bật mạnh → "m" trong "diplomacy" (dự đoán "acy"). Khả thi vì output là softmax của hàm tuyến tính:

$$
x\in\{1,2,3\}\mapsto \mathrm{softmax}(x,\;3x-3,\;5x-8)
$$

chọn được A, B, C. Điều này **trái với linear representation hypothesis** và khó phân tích hơn cả superposition.

### 1.7 Dự đoán activation ở phía sau

*Templeton, Batson, Jermyn, Olah* — Thay vì SAE (tái tạo chính activation), huấn luyện mô hình thưa **đọc ở điểm A, dự đoán ở điểm B**: (i) pre-ReLU → post-ReLU MLP; (ii) residual trước MLP → output MLP; (iii) residual trước attention → output attention (đang rất sớm). (i)(ii) cho feature dễ đọc và **dễ phân tích circuit** vì trọng số vào/ra ghép thẳng với residual stream, mở rộng phân tích tuyến tính của *Mathematical Framework* cho MLP. Đây là tiền thân của **transcoder** / CLT trong 2025.

### 1.8 Bài toán mở và nghiên cứu nhóm khác

- **Bài toán mở:** tìm cách dùng attention bất ngờ; mô tả attention circuit kiểu "QK chọn lớp X, OV kéo thông tin Y để tạo Z"; toy model *không* có superposition; **dựng mô hình monosemantic rồi nén vào mô hình nhỏ có superposition, nghiên cứu lỗi và quan hệ trọng số** (rất gần với distillation!); toy model có cross-layer superposition.
- **Nhóm khác:** *task vectors* (Hendel et al.) và *function vectors* (Todd et al., ~12 head đóng góp chính) — nhiệm vụ ICL được mã hóa thành vector có thể patch sang ngữ cảnh mới; inductive bias (CFG, ICL vs in-weights, "pizza" vs "clock", Morwani et al. chứng minh nghiệm max-margin phải có Fourier feature — lần đầu thuật toán tìm bằng reverse engineering được chứng minh tối ưu); *Fact Finding* (Nanda et al.): detokenization layer 0–1, recall ở MLP 2–6, rồi một head chính mang thông tin môn thể thao; SAE trên output attention (Kissane et al.).

---

## 2. February 2024

### 2.1 Feature chết dần trong dictionary learning

*Templeton, Batson, Henighan* — **Câu hỏi:** vì sao SAE rộng có một "cụm mật độ cực thấp" chiếm phần lớn capacity, encoder vector của chúng gần như trùng hướng (Nanda phát hiện) dù decoder ngẫu nhiên? **Giải thích:** các feature này đặt gần hết trọng số encoder lên vài neuron transformer **bật yếu trên toàn dataset**. Do dùng *pre-encoder bias* (trừ $b_d$ trước encoder), các neuron này có giá trị đầu vào hằng khác 0; đặt trọng số ngược dấu với chúng = tạo bias âm. Mỗi khi gradient đòi giảm activation, encoder vừa giảm bias vừa tăng các trọng số này → cùng một hướng. Kết luận: thực chất là **L1 giết feature trước khi chúng tìm được hướng decoder hữu ích**.

### 2.2 Một loạt cải tiến kiến trúc SAE

*Templeton, Conerly, Marcus, Henighan, Golubeva, Bricken* — (một phần bị rút lại ở March)
- Adam $\beta_1=0$ giảm mạnh dead feature; $\beta_2$ 0.999→0.9999 tác dụng nhỏ; không cộng dồn được.
- Ràng buộc $\|W_{d,i}\|\le1$ (thay vì $=1$) và **prune feature có decoder norm < 0.99** sau huấn luyện.
- Bỏ pre-encoder bias (vẫn giữ bias trước ReLU).
- Chuẩn hóa activation về norm $\sqrt{n_\text{dense}}$ để hyperparameter chuyển được giữa các cỡ mô hình.
- Cooldown: giảm LR tuyến tính về 0 trong 20% cuối.

### 2.3 Ghost Grads: cập nhật

*Conerly* — Có bug đánh dấu mọi neuron là chết trong K bước đầu, và bug lại *cải thiện* loss. Ghost grads giúp nhiều ở mô hình 1L, ít khác biệt ở mô hình lớn; nhóm không thực sự hiểu vì sao/ khi nào nó giúp.

### 2.4 Phạt tanh thay L1

*Jermyn, Templeton, Batson, Bricken* — Để chống shrinkage, thử phạt

$$
\text{penalty}(x)=\frac{A}{B}\tanh(Bx),
$$

(theo Hurley & Rickard 2008): gần 0 cho gradient như L1, ở activation lớn gần như không phạt. Kết quả: **Pareto tốt hơn** trên (L0, loss recovered), nhưng feature khó đọc hơn nhiều, do nhiều feature tần suất cao (cao nhất >10%). Họ tạm gác hướng này; nhắc tới Wright & Sharkey (fine-tune một phần tham số không phạt thưa để sửa shrinkage). *(Xem đính chính ở August, mục 7.2.)*

---

## 3. March 2024

### 3.1 Dùng feature để nhận diện circuit nhanh

*Batson, Chen, Jones* — **Câu hỏi:** thay vì dựng dataset hàng nghìn ví dụ cho một hành vi rồi ablate từng thành phần, có thể khởi động phân tích circuit từ **một ví dụ** bằng SAE không?

**Setup:** tái hiện bài "Fact: [ATHLETE] plays the sport of" (Nanda et al., ~1500 vận động viên); mô hình 18 layer; SAE expansion ~256 trên residual stream layer 9. Điểm attribution của feature $i$ tại token " Jordan":

$$
\boxed{\text{attr}_i = a_i\,\mathbf d_i\cdot\nabla_x\mathcal L}
$$

với $a_i$ là activation, $\mathbf d_i$ là decoder vector, $\mathcal L$ là logit difference [basketball] − [golf] (dùng hiệu hai môn để lấy feature *bóng rổ*, không phải "thể thao nói chung").

**Kết quả:**
- Feature top rõ ràng là "bóng rổ" (layup, point guard); tương tự baseball cho Babe Ruth, tennis cho Serena Williams. Vì feature = $\mathrm{ReLU}(E_ix+b_i)$, ta có ngay một **probe** từ một ví dụ, và biết subject enrichment đã xong ở layer 9.
- Feature "cuối tên" (Judas Priest, John Cena) có attribution cao cho Jordan, Ruth nhưng không cho Williams → giả thuyết: tennis gắn với "Serena" một mình. Kiểm chứng: "[FIRSTNAME] Smith" giữ dự đoán đúng đúng ở những trường hợp feature này có attribution thấp → bằng chứng **detokenization** ở layer sớm.
- Tìm head có direct logit effect lớn từ feature bóng rổ ($U\,\mathrm{norm}(OV\mathbf d_i)$): nhiều head, top 5 ở layer 11, 13, 14; cùng các head đó làm việc với baseball/tennis — khớp "Subject Heads" (Chughtai et al.), tạo output bằng **constructive interference** nhiều head.

### 3.2 Đính chính các cải tiến SAE

*Templeton, Conerly, Marcus, Henighan* — Sau các sửa đổi khác (gồm sửa bug), **không còn thấy lợi ích** từ: ghost grads (thậm chí gây loss spike), Adam $(0, 0.9999)$, prune feature decoder norm < 0.99.

### 3.3 Nghiên cứu nhóm khác (chọn lọc)

- **Othello-GPT + SAE** (He et al.): feature cho trạng thái ô; feature layer sau = AND của feature layer trước qua OV/MLP với mặt nạ gating cố định. Lưu ý L0 ~60–100 với $d_\text{model}=128$ — quá dày. Transformer L layer chỉ có ~2L bước tuần tự → phải học **thuật toán tắt** (Liu et al.: automata với độ sâu $O(\log T)$; Sanford et al.: k-hop induction với độ sâu $\log k$).
- **Computation in superposition** (Vaintrob et al.): kết cấu U-AND — một MLP layer tính được *mọi* AND cặp đôi đọc tuyến tính được với sai số nhỏ; "linearly represented" = đọc được bằng covector với sai số cộng $\pm\epsilon$.
- **AtP\*** (Kramár et al.): attribution patching (xấp xỉ gradient của activation patching) sai khi (a) nonlinearity ngay sau (softmax attention, LayerNorm bão hòa) và (b) direct/indirect effect triệt tiêu. Sửa: chạy forward qua softmax; backward bỏ qua từng layer rồi trung bình.
- **Self-repair & LayerNorm** (Rushing & Nanda): ablate một head, các thành phần sau "tự sửa". **LayerNorm giải thích ~30% self-repair** ở 2% token có direct effect cao nhất: nhiều thành phần cùng đẩy một token → ablate một cái làm norm residual giảm mà hướng giữ nguyên → hệ số LayerNorm tăng → logit của các thành phần còn lại tăng.

---

## 4. April 2024

### 4.1 Tuyển dụng và dự án kiến trúc dễ diễn giải

Ước tính ~50 vị trí full-time về mech interp trên thế giới; nhóm Anthropic có 17 người (tuyển 10 trong 2023). Mở nhóm **Interpretability Architectures**: tìm lựa chọn kiến trúc giúp interp dễ hơn (ví dụ LayerNorm gây khó cho phân tích circuit; SoLU từng là một thử nghiệm).

### 4.2 Scaling laws cho dictionary learning

*Lindsey, Conerly, Templeton, Marcus, Henighan* — SAE trên residual sau layer 3 của transformer 4 layer, loss proxy $\mathrm{MSE}+5\,\mathrm{L1}$. Compute ∝ (#feature × #bước).
- Loss giảm gần **power law** theo FLOPs khi phân bổ tối ưu; số bước và số feature tối ưu cũng theo power law.
- Số feature tối ưu tăng nhanh hơn số bước: bước tối ưu ∝ (#feature)$^{\gamma}$ với $\gamma\in(0.5,1)$.
- L1 coefficient lớn hơn → cần nhiều bước hơn (reconstruction được tối ưu nhanh hơn sparsity). Loss dựa trên L0 cần nhiều bước hơn nữa.
- Cảnh báo: quan hệ giữa SAE loss và độ hữu ích định tính không hoàn hảo.

### 4.3 Cách nhóm huấn luyện SAE (bản 04/2024)

*Conerly, Templeton, Bricken, Marcus, Henighan*

$$
\boxed{\mathcal L=\frac1{|X|}\sum_{x}\|x-\hat x\|_2^2+\lambda\sum_i f_i(x)\,\|W_{d,i}\|_2}
$$

Thay đổi quan trọng nhất so với 2023: **decoder norm không ràng buộc, nhưng đưa vào phạt**. Chi tiết: $b_e,b_d=0$; cột $W_d$ hướng ngẫu nhiên, norm 0.05–1 (0.1 hợp lý); $W_e=W_d^\top$ lúc khởi tạo; scale dataset để $\mathbb E\|x\|_2=\sqrt n$; Adam (0.9, 0.999), LR ~5e-5, decay 20% cuối; ~200k bước; batch 2048/4096; clip grad norm 1; $\lambda$ tăng tuyến tính từ 0 trong 5% đầu, mặc định 5. **<1% feature chết** (không bật trong 10M mẫu) nên không cần resampling/ghost grads. Đã tăng $m$ tới hàng triệu feature.

Kết quả (131.072 feature, 200k bước, batch 4096; trích):

| Run | λ | L0 | Normalized MSE | Frac CE recovered |
|---|---|---|---|---|
| 1L MLP | 5 | 38.7 | 0.066 | 0.964 |
| 4L MLP (layer 2) | 5 | 69.9 | 0.125 | 0.929 |
| 4L Residual (layer 2) | 5 | 33.2 | 0.163 | 0.904 |
| 4L Residual (layer 2) | 10 | 8.7 | 0.255 | 0.839 |

Lưu ý: L0 không ổn định — train gấp đôi với nửa số feature có thể đổi MSE/L1 <1% nhưng giảm L0 30%.

### 4.4 Feature SAE ảnh hưởng hành vi mạnh đến đâu?

*Lindsey* — **Câu hỏi:** mô hình có thực sự "nghe" feature SAE, hay feature chỉ có nghĩa với người? **Setup:** SAE trên residual sau layer 3 (mô hình 4 layer); nhân activation feature với 0 (ablation) hoặc 2 (doubling), đo tăng loss; so với nhiễu cùng độ lớn.

| Can thiệp (cùng độ lớn) | Tác động |
|---|---|
| Ablate một feature | Lớn nhất; lớn hơn hẳn nhiễu ngẫu nhiên (cả trên activation mô hình lẫn trên vector feature) |
| Doubling feature | Nhỏ hơn ablation → ảnh hưởng **bão hòa** ở activation cao |
| "Dampen feature activations" (ngược hướng bản tái tạo) | Tương đương ablation đơn feature |
| "Dampen model activations" | Nhỏ hơn; gần như biến mất ở layer sau → không chỉ là hiệu ứng norm |
| Dọc theo **SAE error** $-(x-\mathrm{SAE}(x))$ | Mạnh hơn ngẫu nhiên, yếu hơn ablation |

Thêm: hiệu ứng nổi bật hơn với feature **residual stream** so với feature MLP. Gợi ý: lỗi tái tạo của SAE nằm chủ yếu **dọc hướng feature** (khớp với Gurnee, mục 4.7).

### 4.5 Caloric và giá trị của lý thuyết sai

*Henighan* — Thuyết "caloric" về nhiệt sai, nhưng dẫn tới chu trình Carnot và hiệu chỉnh tốc độ âm thanh. Bài học: kể cả nếu **superposition hypothesis** về sau bị bác bỏ, dùng nó vẫn có thể mang lại thắng lợi thực tế.

### 4.6 Bài toán mở: attribution dictionary learning; tách path theo độ dài

*Olah, Templeton, Bricken, Jermyn* — Feature có "bản chất kép": nhìn về input là *representation*, nhìn về output là *action*; cả hai nên thưa. Đề xuất loss:

$$
\mathcal L=\|x-x'\|^2+\lambda\|y\|_1+\alpha\,\|y\odot\nabla_y L_\text{LLM}\|_1+\beta\,\big|(x-x')\cdot\nabla_x L_\text{LLM}\big|
$$

với $x'=Dy$, $\nabla_y L=D^\top\nabla_x L$: thêm phạt **độ thưa của attribution** và phạt **attribution không giải thích được**. Thử nhanh trên mô hình 1 layer: feature tốt ngang thường, chưa kết luận.

*Olah* — Giải thích lại thuật toán tách path độ dài ≤ k từ *Mathematical Framework*: mỗi lần lặp dùng path độ dài $k$ đã lưu và lưu path độ dài $k+1$. Biến thể: tách hiệu ứng k bước của một ablation; ablate từ bước i tới j.

### 4.7 Nghiên cứu nhóm khác

- **Sparse Feature Circuits** (Marks et al., Pythia 70M): node = feature SAE + node lỗi SAE; attribution để chọn node/edge; đo faithfulness/completeness; **SHIFT** — người chọn feature thiên lệch để cắt khỏi circuit classifier.
- **Lỗi tái tạo SAE "bệnh lý"** (Gurnee, GPT-2 small): thay $x$ bằng $\mathrm{SAE}(x)$ tăng KL nhiều hơn hẳn nhiễu ngẫu nhiên cùng norm/góc → lỗi **có hệ thống**.
- **DPO & toxicity** (Lee et al., GPT-2 Medium): mọi vector tham số sau DPO có cosine ≥ 0.99 với gốc, nhưng hành vi đổi. Cơ chế: nhiều thay đổi nhỏ ở layer trước làm giảm pre-activation của "toxic neuron". Neuron độc hại **vẫn còn**; tăng in-weights là bật lại. Tác giả nối với jailbreak: fine-tune giảm tần suất hành vi nhưng không xóa cơ chế.
- **Binding ID** (Feng & Steinhardt): gắn thuộc tính–thực thể bằng "binding vector" trong không gian con ≥ 2 chiều, chuyển được giữa các nhiệm vụ.

---

## 5. June 2024

### 5.1 So sánh TopK và Gated SAE với SAE chuẩn

*Cunningham, Conerly* — **Câu hỏi:** các biến thể mới (Gated — DeepMind; TopK — OpenAI; fine-tune sau huấn luyện — Wright & Sharkey) cải thiện L0/MSE nhờ giảm shrinkage; nhưng feature có còn dễ đọc?

- **L0/MSE:** tái hiện được cải thiện lớn; Gated và TopK tương đương. (TopK: $L0=K$ theo định nghĩa. Phải bỏ việc đóng băng decoder trong $L_\text{aux}$ của Gated.)
- **Mật độ:** TopK và Gated có nhiều feature dense hơn vanilla:

| Run | L0 | #feature mật độ > 1e-2 | > 1e-3 |
|---|---|---|---|
| TopK K=50 | 50 | 261 | 8076 |
| TopK K=20 | 20 | 54 | 2855 |
| Gated | 37 | 8 | 4998 |
| Vanilla | 38 | 13 | 360 |

- **Đánh giá mù** 100 feature (thang 1–5 cho *activation consistency* và *complexity*): không thấy khác biệt.
- **Feature dense** (K=20, 256k feature, mô hình 18 layer): cao nhất ~9%, thường liên quan <EOT>; vài feature khó hiểu ở 1–2%. Với K=100 có hàng chục feature >10% khó đọc → nên nhắm độ thưa cao hơn. Kết luận: feature dense không phải bệnh ở L0 thấp, và vì thế tanh có thể không tệ như tưởng.
- **Clerp** (autointerp nội bộ, 200–250 feature/run): Gated L1=5 và TopK K=20 đạt điểm ngang vanilla L1=10 vốn có MSE tệ hơn.
- **Hạn chế:** khó biết *basis* có tốt hơn không; giảm shrinkage có thể cải thiện MSE mà basis không đổi. Cuối cùng phải đánh giá bằng công dụng (debug, steering, circuit, hiểu fine-tune).

### 5.2 Nghiên cứu nhóm khác

- **Gao et al. (TopK, GPT-4):** bỏ L1 → hết shrinkage, đặt L0 trực tiếp; khởi tạo encoder = transpose decoder; aux loss kiểu ghost grads. Đánh giá bằng probe một-feature (tốt nhất ở L0 64–128) và độ thưa của hiệu ứng ablation lên logit (tốt nhất ở L0 128–256).
- **RAVEL:** benchmark tách thuộc tính (Paris ở châu Âu vs nói tiếng Pháp) bằng can thiệp; tốt nhất là biến thể Distributed Alignment Search.
- **Makelov et al. (IOI):** tiêu chí *sufficiency/necessity*, *sparse controllability*, *interpretability*; SAE không giám sát kém dictionary có giám sát. Hiện tượng **feature occlusion** (feature IO lấn át S) và **over-splitting**.
- **Multidimensional features** (Engels et al., Llama 3 8B, Mistral 7B): ngày trong tuần / tháng nằm trên **vòng tròn 2D**, có vai trò nhân quả trong số học modulo ("hai ngày sau thứ Hai"); thông tin ở *góc*; tính toán chủ yếu ở MLP, head chỉ copy.

---

## 6. July 2024

### 6.1 Năm rào cản tiếp theo

*Olah* — Superposition đã được "mở khóa" đáng kể; con đường tưởng tượng: tìm feature (biến) → hiểu circuit → phóng to thành bức tranh vĩ mô. Năm rào cản:

| Rào cản | Nội dung |
|---|---|
| Missing features | Chỉ trích được một phần nhỏ feature; feature hiếm có thể không bao giờ tách được ("dark matter") |
| **Cross-layer superposition** | Circuit nông trong mô hình sâu: neuron ở layer 10 và 11 gần như "song song" với gradient descent; feature không gắn rõ với một layer |
| Attention superposition | Tổ hợp head biểu diễn thêm attentional feature; có thể thấy induction head "bị chia" qua nhiều head |
| **Interference weights** | Virtual weight giữa feature có thành phần chỉ là thỏa hiệp để chứa circuit khác; vô nghĩa trên phân phối nhưng có thể giải thích adversarial example |
| Zooming out | Từ hiểu feature/circuit tới hiểu toàn mạng; ưa cấu trúc lớn, universality hơn là chỉ autointerp |

### 6.2 Linear representation là gì? Multidimensional feature là gì?

*Olah* — Định nghĩa trong Toy Models gộp hai mệnh đề: (a) feature là **một chiều**, (b) feature hành xử **tuyến tính toán học**. Nhóm nay dùng "linear representation" cho (b):
- **Composition as Addition:** có nhiều feature = cộng các vector.
- **Intensity as Scaling:** cường độ/độ tin = độ lớn.

(b) sâu hơn: nếu sai, phải làm lại từ đầu; nếu chỉ (a) sai, interp thích nghi dễ. **Feature đa chiều tuyến tính** phải gần trực giao với feature khác, và mỗi điểm trên manifold tạo một **tia** khi scale — manifold co về 0 theo các tia, thay đổi ngữ nghĩa là chuyển động góc. Các bằng chứng probe/steering vẫn đúng vì một thuộc tính cố định = một tia = một hướng. Lưu ý: manifold có thể được nhúng phức tạp hơn topology của nó để đạt khoảng cách mong muốn — phần "nhiễu" ngoài các PC đầu có thể thiết yếu.

### 6.3 "Dark matter" của mạng nơ-ron

*Olah* — SAE 13 triệu feature có feature cho vài khu phố San Francisco, nhưng Claude biết quán cà phê nhỏ, giao lộ → cần ít nhất thêm hai bậc độ lớn feature, hiếm hơn tương ứng. Cũng có thể có nhiều **memorization feature** cực thưa. Phép ẩn dụ: dictionary learning là kính thiên văn, mới thấy "sao sáng"; phần lớn có thể là vật chất tối không quan sát trực tiếp được.

### 6.4 Pivot table cho attention (kiểm tra chất lượng mô hình nền)

*Turner, Pearce, Bricken, Jermyn* — Tái hiện "pivot table" skip-trigram của *Mathematical Framework* hóa ra rất nhạy chi tiết huấn luyện. Công thức (bỏ qua positional embedding):

$$
C_{QK}=L_A(W_E)^\top W_Q^\top W_K L_A(W_E),\qquad
C_{OV}=W_U L_U\big(W_OW_VL_A(W_E)\big)-\beta_U
$$

(trừ bias LayerNorm trước unembedding để thấy hiệu ứng tương đối). Chấm điểm key: $H_K(k)=\max(C_{QK}[:,k]\cdot P)\cdot\max C_{OV}[:,k]\cdot P[k]$ với $P$ là tần suất token. Run tốt nhất có ~25–75% mục trigram dễ đọc; **dưới ~10% thì nghi ngờ "ảo ảnh"**. Yếu tố: hyperparameter kém → bảng rối; dữ liệu dễ đọc (tiếng Anh, Python); mô hình 1 layer rõ nhất.

### 6.5 Đo độ nhạy (sensitivity) của feature bằng lọc dataset

*Turner, Jermyn, Batson* — *Specificity* = P(khái niệm | feature bật); *sensitivity* = P(feature bật | khái niệm). Dùng Claude chấm mức liên quan 0–10; các phiên bản Claude đồng thuận cao. Claude 3.5 Sonnet lọc 160.000 đoạn 128 token cho mỗi khái niệm (ngữ cảnh 2048). **Kết quả:** các feature nổi tiếng trong *Scaling Monosemanticity* (brain sciences, transit infrastructure, tourist attractions) **thường không bật** trên đoạn được chấm ≥ 9; chúng nằm trong top ~0.1–10% feature về sensitivity nhưng không phải nhạy nhất. Một **họ feature** cùng nhau phủ khái niệm (dải dọc trong biểu đồ phủ). Kết luận: cần thận trọng khi nói "feature X *là* khái niệm Y".

---

## 7. August 2024

### 7.1 Eval interpretability cho dictionary learning

*Lindsey, Cunningham, Conerly* — Hai eval kiểu autointerp định lượng (Claude dự đoán activation từ visualization):
- **Contrastive eval:** một danh sách khái niệm soạn tay (có LLM hỗ trợ, ví dụ "sarcasm", "photosynthesis", "imbalanced parentheses in code"); Claude viết hai câu gần giống, chỉ khác khái niệm; lấy feature bật ở một câu; Claude (xem ví dụ dataset) đoán câu nào. Báo cáo trung bình thô và trung bình có trọng số theo activation chuẩn hóa.
- **Sort eval:** chọn cặp feature (ngẫu nhiên, hoặc "nearest" theo cosine decoder — khó hơn); cho top-10 ví dụ của mỗi feature; Claude đoán ví dụ truy vấn thuộc feature nào.
- Kiểm tra tỉnh táo: lỗi của Claude thường cũng khó với người; điểm tăng theo số bước, số feature, và tương quan với eval loss.

### 7.2 Case study: so sánh 6 biến thể SAE

*Templeton, Conerly, Lindsey, Cunningham, Persic* — Vanilla, TopK, TopK_fixed (ngưỡng cố định sau huấn luyện), Gated, Tanh, JumpReLU. SAE ~260.000 feature, 400M token trên mô hình cỡ trung; eval bằng Claude 3 Haiku. Chọn 3 metric tương quan tốt với eval loss.
- **Kết luận chính:** các biến thể **tốt hơn vanilla**; giữa chúng khác biệt nhỏ, eval chưa đủ phân giải → nên chọn theo dễ dùng, ổn định khi scale, dễ phân tích circuit.
- Không tái hiện được JumpReLU tốt hơn TopK/Gated trên L0-MSE.
- **Đính chính tanh:** không còn thấy feature dense khó đọc; đoán là nhờ setup huấn luyện 04/2024.

### 7.3 Feature tự giải thích

*Ameisen, Batson* — Tiêm decoder vector của feature (nhân hệ số) vào vị trí "X" trong "The meaning of the word X is" rồi sample. Tốt với feature layer sớm đơn giản ("big", "sport", hệ số 100). Feature "Jordan" layer sớm phụ thuộc scale (10: "to go"; 60: "the one who is the most powerful"; 410: "the greatest of all time") nhưng không ra token Jordan; một feature layer 4 khác ra "Jor- dan" ở scale 260. Trên Claude 3 Sonnet: đôi khi được (feature "sheet" → chỉ nghĩa tờ giấy), kém hơn và nhạy scale hơn ở mô hình lớn; feature "of" trong ngữ cảnh màu sắc thì thất bại.

---

## 8. September 2024

### 8.1 Successor heads

*Ameisen, Batson* — Tái hiện Gould et al.: vài head thực hiện "kế tiếp" trên chuỗi thứ tự (2→3, Wednesday→Thursday, G→H) trong mô hình 18 layer, bằng bốn phương pháp:

| Phương pháp | Cách làm | Kết quả |
|---|---|---|
| Trọng số (OV) | Biểu diễn token = residual sau MLP đầu; qua OV, unembed, giới hạn trên token thứ tự; tỉ lệ token có top output = successor | Head tốt nhất ~80%, head thứ hai ~60% (lỗi có cấu trúc khối theo loại) |
| ICA trên các ma trận OV của mọi head | Mỗi component là một "meta-head" | Thành phần succession, induction, chiếu-lên-loại |
| Mean-ablation trên chuỗi thật | Hiệu ứng lên logit successor đúng | 3 head top theo trọng số xếp hạng 1, 3, 5; hạng 2 và 4 là head ở layer thấp (3, 5) — có thể qua Q/K-composition |
| Attribution patching | Xấp xỉ gradient | Head top trùng, nhưng **mức đồng thuận tổng thể thấp** với ba phương pháp kia |

Ý nghĩa: hai component ICA (succession "quên" loại chuỗi; copy loại "quên" thứ tự) khớp với biểu diễn tuyến tính tổ hợp (giá trị thứ tự + loại chuỗi) của Gould et al.

### 8.2 Oversample một chủ đề trong dữ liệu SAE

*Bricken, Marcus, Rivoire, Henighan* — SAE 34M feature trên Claude 3 Sonnet có ít feature cụ thể về vũ khí sinh học. Thêm dữ liệu tổng hợp về chủ đề này vào tập huấn luyện SAE → xuất hiện feature cụ thể (biến đổi mầm bệnh để né miễn dịch, phát tán hạt virus), thay vì feature chung chung "chế tạo vật nguy hiểm". Gợi ý giải pháp cho bài toán **feature coverage**: điều khiển SAE học feature cần thiết bằng cách oversample.

---

## 9. Cách kiểm chứng và giới hạn chung

**Các loại bằng chứng trong năm 2024:**
1. **Proxy loss** (MSE, L0, loss recovered) và scaling law — rẻ nhưng không phải mục tiêu thật.
2. **Autointerp định lượng** (Clerp, contrastive, sort eval) — dùng Claude làm thẩm định, được kiểm tra với người.
3. **Can thiệp nhân quả:** ablation/doubling feature so với nhiễu cùng độ lớn (4.4); mean-ablation head (8.1); kiểm chứng giả thuyết bằng prompt mới ("FIRSTNAME Smith", 3.1).
4. **Đối chiếu nhiều phương pháp** cho cùng câu hỏi (trọng số vs ablation vs attribution, 8.1).
5. **Tự đính chính:** ghost grads, β₁=0, prune decoder norm, tanh đều bị xem lại.

**Giới hạn (tác giả nêu):** SAE chỉ bắt một phần feature (dark matter, sensitivity thấp); lỗi tái tạo có hệ thống; eval chưa đủ phân giải; nhiều kết quả trên mô hình nhỏ (1L, 4L, 8L, 18L); autointerp phụ thuộc một mô hình khác; feature dễ đọc chưa chắc là "feature thật".

---

## 10. Ứng dụng giả thuyết cho pruning / distillation và thí nghiệm gợi ý

> **Phần này là suy luận của chúng ta, không phải kết quả của các update.** Mục liên quan nhất: 3.3 (self-repair, AtP\*), 4.4 + 4.7 (lỗi SAE có hệ thống, DPO), 6.1 (cross-layer superposition, interference weights), 1.6–1.7 (magnitude coding, transcoder), 8.1 (các thước đo importance bất đồng), 6.3 + 8.2 (feature hiếm, oversampling).

**(a) Vì sao cắt layer phục hồi được còn cắt neuron thì không — một giả thuyết.** *Cross-layer superposition* (6.1) nói rằng trong mô hình sâu, layer kề nhau gần như "song song": cùng một feature có thể được tính rải qua vài layer. Khi bỏ một layer, các layer kề vẫn mang phần lớn cùng các hướng feature, nên một map tuyến tính tại chỗ cắt (D_lin) bù được nhiều. Ngược lại, cắt neuron MLP phá trực tiếp các hướng trong superposition *và* có thể phá biểu diễn theo độ lớn (1.6: một neuron mã hóa A/B/C bằng mức 1/2/3) — không map tuyến tính nào khôi phục được mà không có lại neuron đó → D_res. Đây là giả thuyết cần kiểm, nhưng khớp với quan sát của bạn.

**(b) D_res có thể *không* phải nhiễu.** Lindsey (4.4) và Gurnee (4.7) thấy lỗi tái tạo SAE gây hại hơn nhiễu ngẫu nhiên cùng norm và có vẻ nằm dọc hướng feature. Tương tự, D_res sau pruning có thể tập trung ở một số hướng feature có ý nghĩa (ví dụ feature ký tự hiếm). Nếu đúng, KD nên nhắm *hướng* của D_res thay vì MSE đồng đều.

**(c) Importance score không đáng tin nếu chỉ dùng một loại.** Mục 8.1 cho thấy trọng số, ablation và attribution patching bất đồng; AtP\* (3.3) chỉ ra attribution sai ở softmax/LayerNorm bão hòa và khi direct/indirect effect triệt tiêu. Self-repair (3.3) làm ablation *đánh giá thấp* tầm quan trọng đơn lẻ, nhưng cũng nghĩa là cắt nhiều thành phần cùng lúc sẽ tệ hơn tổng các lần cắt riêng. Với pruning head/layer của LightOnOCR, nên dùng ít nhất hai thước đo và đo cắt đồng thời.

**(d) Transcoder như "pruning có cấu trúc thông minh".** Mục 1.7 (dự đoán output MLP từ input bằng mô hình thưa) gợi ý một lựa chọn khác cho việc cắt neuron MLP: thay MLP bằng một transcoder thưa nhỏ được distill để khớp output MLP. Đây là ý tưởng, chưa được bài kiểm cho mục đích nén.

**(e) Distillation có thể "che" thay vì "xóa".** DPO (4.7) thay đổi hành vi bằng thay đổi tham số cực nhỏ phân tán, cơ chế cũ vẫn còn. Ngược lại với nén: student sau KD có thể *trông* khớp teacher trên phân phối calibration nhưng dựa vào cơ chế khác/yếu hơn; nên kiểm tra ngoài phân phối (font lạ, chữ viết tay, bảng).

**(f) Feature hiếm và dữ liệu calibration.** "Dark matter" (6.3) và oversampling (8.2) gợi ý: tập calibration nhỏ cho pruning/KD sẽ bỏ qua các feature hiếm (ký tự hiếm, ký hiệu toán, script ít gặp). Oversample các miền OCR quan trọng nhưng hiếm khi tính importance.

**Thí nghiệm gợi ý:**
- **E1 — Ba thước đo importance:** cho mỗi head full-attention và mỗi block GatedDeltaNet, tính (i) weight-based (norm của $W_OW_V$ chiếu lên unembedding cho token ký tự), (ii) mean-ablation trên dữ liệu OCR, (iii) attribution patching. Tính tương quan hạng. Nếu thấp như 8.1, đừng prune theo attribution đơn thuần.
- **E2 — D_res có hướng hay không:** sau khi cắt neuron và fit map tuyến tính tối ưu, lấy vector D_res; thay bằng nhiễu ngẫu nhiên cùng norm và so ΔCER/ΔKL (theo kiểu 4.4). Nếu D_res gây hại hơn nhiễu nhiều, nó mang thông tin có cấu trúc → thêm loss KD trên subspace PCA của D_res.
- **E3 — Self-repair theo layer:** với mỗi layer bị loại, so *direct effect* (đóng góp trực tiếp vào logit) với *total effect* khi ablate. Layer có total ≪ direct là layer được self-repair (có thể nhờ norm) — ứng viên dễ cắt; kiểm tra lại khi cắt đồng thời hai layer như vậy.

---

## 11. Tóm tắt khái niệm

| Khái niệm | Công thức / mô tả | Ý nghĩa |
|---|---|---|
| SAE loss (04/2024) | $\Vert x-\hat x\Vert ^2+\lambda\sum_i f_i\Vert W_{d,i}\Vert $ | Decoder norm không ràng buộc nhưng bị phạt |
| Shrinkage | L1 làm activation nhỏ hơn thật | Lý do ra đời tanh, Gated, TopK, JumpReLU |
| Feature attribution | $a_i\,\mathbf d_i\cdot\nabla_x\mathcal L$ | Tìm feature quan trọng từ một ví dụ |
| Attribution dictionary learning | Thêm $\alpha\Vert y\odot\nabla_yL\Vert _1+\beta\vert (x-x')\cdot\nabla_xL\vert $ | Feature thưa cả về "representation" và "action" |
| Magnitude-based coding | $\mathrm{softmax}(x,3x-3,5x-8)$ | Phản ví dụ cho linear representation |
| Self-repair | Thành phần sau bù ablation; ~30% do LayerNorm (top 2% token) | Ablation đơn lẻ đánh giá thấp importance |
| Linear representation | Composition as addition + intensity as scaling | Cho phép feature đa chiều (manifold tạo bởi tia) |
| Cross-layer superposition | Feature rải qua các layer kề | Giả thuyết cho việc cắt layer dễ phục hồi |
| Interference weights | Virtual weight thỏa hiệp | Gây nhiễu phân tích circuit |
| Dark matter | Feature quá hiếm để SAE tách | Giới hạn của coverage |
| Specificity / sensitivity | P(khái niệm\|feature) / P(feature\|khái niệm) | Feature thường kém sensitivity |
| Pivot table | $C_{QK}$, $C_{OV}$ ở mức token | Kiểm tra mô hình 1 layer; <10% dễ đọc là đáng ngờ |
| Successor head | OV ánh xạ token → token kế tiếp | Ví dụ head phổ quát; các thước đo bất đồng |

### Kết luận ngắn

Năm 2024 các update chủ yếu là kỹ thuật SAE: từ mẹo chữa dead feature (phần lớn bị rút lại) tới công thức huấn luyện ổn định, biến thể chống shrinkage và các eval đầu tiên. Song song, nhóm định hình lại câu hỏi lý thuyết: linear representation thực chất là tính tuyến tính, không phải tính một chiều; còn dark matter, cross-layer superposition, attention superposition và interference weights là các rào cản tiếp theo. Với người làm nén mô hình, các bài học đáng giữ là: thước đo importance hay bất đồng với nhau, self-repair có thật, lỗi tái tạo/xấp xỉ thường có cấu trúc, và fine-tune/KD có thể che một cơ chế chứ không xóa nó.

**Đọc bản gốc:** https://transformer-circuits.pub/2024/jan-update/index.html · https://transformer-circuits.pub/2024/feb-update/index.html · https://transformer-circuits.pub/2024/march-update/index.html · https://transformer-circuits.pub/2024/april-update/index.html · https://transformer-circuits.pub/2024/june-update/index.html · https://transformer-circuits.pub/2024/july-update/index.html · https://transformer-circuits.pub/2024/august-update/index.html · https://transformer-circuits.pub/2024/september-update/index.html
