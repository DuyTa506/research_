# Circuit Tracing: Revealing Computational Graphs in Language Models (2025)

**Tác giả:** Emmanuel Ameisen\*, Jack Lindsey\*, Adam Pearce\*, Wes Gurnee\*, Nicholas L. Turner\*, Brian Chen\*, Craig Citro\*, David Abrahams, Shan Carter, Basil Hosmer, Jonathan Marcus, Michael Sklar, Adly Templeton, Trenton Bricken, Callum McDougall◊, Hoagy Cunningham, Thomas Henighan, Adam Jermyn, Andy Jones, Andrew Persic, Zhenyi Qi, T. Ben Thompson, Sam Zimmerman, Kelley Rivoire, Thomas Conerly, Chris Olah, Joshua Batson\*‡ (Anthropic; \* core contributor, ‡ liên hệ, ◊ làm khi còn ở Anthropic). Công bố 27/03/2025.  
**Nguồn:** https://transformer-circuits.pub/2025/attribution-graphs/methods.html  
**Chủ đề:** Cross-layer transcoder (CLT), replacement model, local replacement model, error nodes, attribution graph, graph pruning, constrained patching, global weights (virtual weights, TWERA), metric (influence, path length, completeness, replacement score), mechanistic faithfulness.  
**Vị trí trong lộ trình:** Cần đọc trước *A Mathematical Framework* (QK/OV, cố định attention pattern), *Towards/Scaling Monosemanticity* (SAE, feature visualization) và *Sparse Crosscoders* (note 14, nguồn gốc của CLT). Đây là paper *phương pháp* trung tâm của giai đoạn 2025; paper đồng hành *On the Biology of a Large Language Model* áp dụng nó lên Claude 3.5 Haiku.

> Đây là ghi chú học tập diễn giải, không phải bản dịch. Mọi liên hệ với pruning/distillation (mục 16–17) là suy luận mở rộng của chúng tôi, không phải khẳng định của paper.

## Mục lục

1. [Bức tranh tổng quan](#1-bức-tranh-tổng-quan)
2. [Cross-layer transcoder kiến trúc và loss](#2-cross-layer-transcoder-kiến-trúc-và-loss)
3. [Replacement model](#3-replacement-model)
4. [Local replacement model và error nodes](#4-local-replacement-model-và-error-nodes)
5. [Attribution graph](#5-attribution-graph)
6. [Gán nhãn feature và supernodes](#6-gán-nhãn-feature-và-supernodes)
7. [Kiểm chứng bằng can thiệp](#7-kiểm-chứng-bằng-can-thiệp)
8. [Ba case study](#8-ba-case-study)
9. [Global weights](#9-global-weights)
10. [Metric của đồ thị](#10-metric-của-đồ-thị)
11. [Graph pruning](#11-graph-pruning)
12. [Đánh giá CLT và mechanistic faithfulness](#12-đánh-giá-clt-và-mechanistic-faithfulness)
13. [Paper kiểm chứng như thế nào](#13-paper-kiểm-chứng-như-thế-nào)
14. [Giới hạn](#14-giới-hạn)
15. [Lựa chọn thiết kế bền vững và tạm thời](#15-lựa-chọn-thiết-kế-bền-vững-và-tạm-thời)
16. [Ứng dụng giả thuyết cho pruning và distillation](#16-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
17. [Thí nghiệm gợi ý](#17-thí-nghiệm-gợi-ý)
18. [Tóm tắt công thức và khái niệm](#18-tóm-tắt-công-thức-và-khái-niệm)

---

## 1. Bức tranh tổng quan

Câu hỏi: với **một prompt cụ thể**, mô hình đi qua những bước tính toán trung gian nào để ra token đầu ra? Paper trả lời bằng **attribution graph**: đồ thị nhân quả mà node là feature diễn giải được, cạnh là ảnh hưởng **tuyến tính trực tiếp**.

Pipeline bốn bước của reverse engineering (theo phần Discussion):

| Bước | Lựa chọn trong paper |
|---|---|
| Phân rã (decomposition) | CLT thay thế các MLP |
| Mô tả thành phần | Feature visualization (dataset examples, logit effects) |
| Mô tả tương tác | Attribution graph trên prompt cụ thể |
| Kiểm chứng | Can thiệp (steering/patching) trên model thật |

Các quyết định then chốt: dùng **transcoder** (không phải SAE) → feature "bắc cầu" qua phi tuyến MLP; dùng **cross-layer** → circuit ngắn hơn; **cố định attention pattern và mẫu số normalization** → tương tác giữa feature trở nên tuyến tính; **pruning** đồ thị; giao diện tương tác; **validation** bằng perturbation; và **global weights** để nhìn tương tác vượt khỏi một prompt.

Mô hình nghiên cứu: một transformer nhỏ **18 layer ("18L")** và **Claude 3.5 Haiku**.

## 2. Cross-layer transcoder kiến trúc và loss

### 2.1 Thuật ngữ

- **Transcoder**: mạng thưa nhận input của MLP và dự đoán output của MLP, thay thế tính toán MLP bằng feature thưa.
- **CLT**: feature được chia vào $L$ layer (bằng số layer model). Feature ở layer $\ell$ **đọc** residual stream tại $\ell$, nhưng **ghi** vào output MLP của mọi layer $\ell,\ell+1,\dots,L$.
- **PLT (per-layer transcoder)**: mỗi layer một transcoder riêng, baseline chính.
- **JumpReLU**: $\mathrm{JumpReLU}_\theta(z)=z\cdot\mathbf 1[z>\theta]$, giữ nguyên giá trị trên ngưỡng, bằng 0 dưới ngưỡng.

### 2.2 Forward

Với $\mathbf x^\ell$ là residual stream tại layer $\ell$ và $\mathbf y^\ell$ là output thật của MLP layer $\ell$:

$$
\boxed{\mathbf a^\ell=\mathrm{JumpReLU}\big(W^\ell_{enc}\mathbf x^\ell\big),\qquad
\hat{\mathbf y}^\ell=\sum_{\ell'=1}^{\ell}W^{\ell'\to\ell}_{dec}\,\mathbf a^{\ell'}}
$$

Output MLP ở layer $\ell$ được tái tạo **chung** bởi feature của mọi layer $\ell'\le\ell$. Mọi feature ở mọi layer huấn luyện đồng thời.

### 2.3 Loss

$$
\mathcal L_{MSE}=\sum_{\ell=1}^{L}\big\|\hat{\mathbf y}^\ell-\mathbf y^\ell\big\|^2,
\qquad
\boxed{\mathcal L_{sparsity}=\lambda\sum_{\ell=1}^{L}\sum_{i=1}^{N}\tanh\!\big(c\cdot\|\mathbf W^\ell_{dec,i}\|\cdot a^\ell_i\big)}
$$

$N$ = số feature mỗi layer; $\mathbf W^\ell_{dec,i}$ = **ghép** mọi decoder vector của feature $i$ (layer $\ell$) tới các layer đích; $\lambda, c$ là siêu tham số. $\tanh$ bão hòa → gần với "đếm số feature hoạt động" (L0) hơn so với L1, không phạt nặng activation lớn.

### 2.4 Chi tiết huấn luyện đáng nhớ

- **Chuẩn hóa riêng** activation từng residual layer và từng MLP output trước khi dùng làm input/target.
- JumpReLU với straight-through estimator, bandwidth **1.0**, ngưỡng khởi tạo **0.03**, và cho gradient chảy qua JumpReLU tới **mọi** tham số (không chỉ ngưỡng).
- Penalty sparsity **tăng tuyến tính từ 0** suốt toàn bộ quá trình huấn luyện.
- **Pre-activation loss** $\sum_f\mathrm{ReLU}(-h_f)$ chống dead feature (chỉ dùng cho Haiku, hệ số $3\times10^{-6}$).
- Khởi tạo: encoder $\sim U(\pm 1/\sqrt{n_{features}})$, decoder $\sim U(\pm 1/\sqrt{n_{layers}\cdot d_{model}})$.
- **Shuffle ở mức token** (không shuffle → kém hơn nhiều).
- Kích thước: 18L tổng 300K–10M feature (CLT có 17 layer vì 18L không có MLP ở layer 0); Haiku 300K–30M. Token huấn luyện: ~3B (18L lớn nhất), ~16B (Haiku lớn nhất). Mục tiêu L0 "vài trăm".
- 18L: chỉ dữ liệu pretraining; Haiku: trộn pretraining + finetuning.

### 2.5 Chi phí

- Ước lượng trên Gemma 2: model 2B, 2M feature, 1B token → $3.8\times10^{20}$ FLOPs (~210 giờ H100 ở 50% hiệu suất); model 9B, 5M feature, 3B token → $6.9\times10^{21}$ FLOPs (~3,844 giờ H100).
- CLT có cùng số tham số encoder như PLT nhưng ~$n_{layers}/2$ lần tham số decoder; mỗi accelerator cùng FLOPs nhưng CLT tốn $n_{layers}$ lần băng thông mạng.
- So sánh cùng chi phí: CLT 1M feature **vượt hẳn** bộ PLT 10M feature về (MSE, L0) và (KL của replacement model, L0) với **ít hơn 5–10×** compute; CLT 3M vs PLT 10M (gần cùng FLOPs): MSE và KL thấp hơn, Sort Eval tương đương, Contrastive Eval kém hơn → chưa có người thắng Pareto ở metric này. CLT cần khoảng ~2 lần token so với dictionary một layer (ước lượng thô trên 18L).

## 3. Replacement model

**Replacement model**: thay mọi MLP bằng CLT. Forward giống model gốc với hai thay đổi: tại input MLP layer $\ell$, tính activation của feature có encoder ở $\ell$; tại output MLP layer $\ell$, ghi đè bằng tổng decoder (tới $\ell$) của feature từ layer $\le\ell$. **Attention chạy bình thường**, không cố định. Lưu ý CLT chỉ được huấn luyện trên activation của model gốc, nên chạy replacement model là chạy CLT trên activation "off-distribution" do chính nó tạo ra.

Đánh giá đơn giản: tỉ lệ prompt mà **top-1 token** của replacement model trùng model gốc. CLT 18L lớn nhất đạt **50%** trên tập prompt kiểu pretraining đa dạng; tốt hơn PLT, tăng theo scale, và hơn baseline "thresholded neurons" (zero các neuron dưới ngưỡng). Tập đánh giá chỉ gồm token model dự đoán đúng nhưng confidence $<80\%$ (bỏ token "nhàm chán").

## 4. Local replacement model và error nodes

Replacement model còn sai lệch và lỗi **cộng dồn qua layer**. Với một prompt cố định $p$, paper dựng **local replacement model**:

1. Thay MLP bằng CLT;
2. Dùng **attention pattern và mẫu số normalization** lấy từ forward thật trên $p$;
3. Cộng **error adjustment** tại mỗi cặp (vị trí token, layer):

$$
\boxed{\varepsilon_{c,\ell}=\mathrm{MLP}_\ell(\mathbf x_{c,\ell})-\mathrm{CLT}_\ell(\mathbf x_c)}
$$

Sau đó mọi activation và logit của local replacement model **khớp chính xác** model gốc trên $p$. Nhưng khớp output **không đảm bảo cùng cơ chế** (giống khai triển Taylor tại một điểm: đúng ở lân cận, lệch khi đi xa). Mức độ cùng phản ứng với perturbation gọi là **mechanistic faithfulness**.

Local replacement model có thể xem như **một mạng fully connected rất lớn trải qua các token**:

- Input: one-hot của mọi token trong prompt;
- "Neuron": mọi feature CLT hoạt động ở mọi vị trí;
- "Weight": tổng tương tác qua **mọi đường tuyến tính** từ feature này tới feature kia (qua residual và qua attention OV đã cố định, không qua MLP/CLT) — gọi là **virtual weights**;
- **Error nodes**: như các node bias, nối tới mọi neuron phía sau, **không có cạnh vào**.

Phi tuyến duy nhất còn lại: phi tuyến áp lên pre-activation của feature.

## 5. Attribution graph

### 5.1 Bốn loại node

| Node | Ý nghĩa | Cạnh vào | Cạnh ra |
|---|---|---|---|
| Output (logit) | token ứng viên, đủ 95% xác suất, tối đa 10 | có | không |
| Feature | (feature CLT, vị trí token) đang hoạt động | có | có |
| Embedding | embedding của token prompt | không | có |
| Error | phần output MLP mà CLT không giải thích | **không** | có |

### 5.2 Trọng số cạnh

Cạnh từ feature nguồn $s$ tới feature đích $t$:

$$
A_{s\to t}:=a_s\,w_{s\to t},
$$

trong đó $w_{s\to t}$ là virtual weight, tương đương đạo hàm của pre-activation $t$ theo activation $s$ khi **stop-gradient** mọi phi tuyến. Tính bằng Jacobian ngược với stop-gradient trên output MLP, attention pattern và mẫu số normalization ($J^{\blacktriangledown}$):

$$
\boxed{A_{s\to t}=a_s\sum_{\ell_s\le\ell<\ell_t}\big(W^{\ell_s\to\ell}_{dec,s}\big)^{\!\top}J^{\blacktriangledown}_{c_s,\ell\to c_t,\ell_t}\,W^{\ell_t}_{enc,t}}
$$

Tổng theo $\ell$ vì feature cross-layer ghi vào nhiều layer: cạnh ra của nó **gộp** hiệu ứng của mọi decoder. Cạnh từ embedding/error: $A_{s\to t}=v_{out,s}^\top J^{\blacktriangledown}v_{in,t}$, với $v_{out}=\mathrm{Emb}_{tok}$ hoặc $v_{out}=\varepsilon_{c,\ell}$. Node logit có $v_{in}=\nabla(\mathrm{logit}_{tok}-\overline{\mathrm{logit}})$ (gradient của logit trừ logit trung bình).

**Khai triển theo đường** (path expansion, tổng quát hóa Framework 2021): Jacobian là tổng trên mọi đường qua residual và attention head:

$$
J^{\blacktriangledown}_{c_s,\ell_s\to c_t,\ell_t}=\sum_{p\in\mathcal P}\pi_p,\qquad \pi_p=\prod_i\pi_i,\quad
\pi_i=\begin{cases}I&\text{bước residual}\\ a^{h_i}_{c_i\to c_{i+1}}\,OV_{h_i}&\text{bước attention}\end{cases}
$$

### 5.3 Phân rã tuyến tính chính xác

Vì đã stop-gradient mọi phi tuyến:

$$
\boxed{h_t=\sum_{s\in\mathcal S_t}A_{s\to t}}
$$

($h_t$ là pre-activation của $t$; $\mathcal S_t$ = node ở layer sớm hơn và vị trí $\le$ vị trí của $t$). Mỗi feature được giải thích **đầy đủ** bởi tổng cạnh vào — đây là lý do attribution "có nguyên tắc".

Ví dụ số (chúng tôi dựng): feature "say DAG" có $h_t=3.0$, nhận $1.2$ từ "say D\_", $0.9$ từ "say \_A", $0.6$ từ "say \_G", $0.5$ từ error node, $-0.2$ từ một feature ức chế. Tổng $=3.0$. Error node góp $0.5/3.0\approx17\%$ → phần "không giải thích được".

**Điều đồ thị không chứa**: ảnh hưởng thông qua việc thay đổi **attention pattern** (QK-circuit). Đồ thị cho biết thông tin chảy từ vị trí nào tới vị trí nào (OV), không cho biết **vì sao** model chọn chuyển thông tin đó.

### 5.4 Tính trong thực tế

Với mỗi node đích: tiêm vector input của nó (encoder của feature, hoặc gradient logit) vào residual tại layer/vị trí nó đọc; chạy backward trên model gốc với stop-grad ở output MLP, attention pattern và mẫu số normalization cố định; với nguồn là feature CLT, lấy tổng dot product giữa decoder ở mỗi layer với gradient tại layer đó, nhân activation. Chi phí **tuyến tính theo số feature hoạt động**, chi phối bởi backward pass. Có chế độ **adaptive**: bắt đầu từ logit, duy trì hàng đợi các feature ảnh hưởng nhất, tính cạnh vào theo thứ tự đó.

Quy mô: khoảng **một trăm** feature hoạt động mỗi vị trí; đồ thị có thể tới **hàng triệu cạnh** ngay cả với prompt ngắn → cần pruning.

## 6. Gán nhãn feature và supernodes

- **Input features**: kích hoạt trên token/nhóm token cụ thể (phổ biến ở layer sớm), ví dụ "digital", "các token tận cùng bằng 6".
- **Output features**: thúc đẩy token cụ thể (phổ biến ở layer muộn), gán nhãn "**say X**", ví dụ "say 'dag'".
- **Feature trừu tượng** (layer giữa): cần nhìn ngữ cảnh, logit effects, và các feature nối với nó; ví dụ "tiếp tục một acronym", "say '\_D'".
- Nhãn chủ yếu do **người** gán (auto-interp chỉ là điểm khởi đầu). Nhãn không hoàn hảo vẫn đủ để thấy cấu trúc.

**Supernode**: nhóm các node (feature, vị trí) có chung "khía cạnh" liên quan tới prompt (ví dụ ba feature "digital" khác nhau, chỉ cần chung "bắt đầu bằng D"). Yêu cầu: các node trong nhóm nên thúc đẩy lẫn nhau, tác động lên node sau cùng dấu. Không có phương pháp tự động nào (clustering theo decoder hay adjacency) đủ tốt. Cạnh giữa supernode:

$$
\mathrm{adj}(t,s)=\frac{\sum_{n_t\in N_t}\mathrm{frac\_ext}(n_t)\sum_{n_s\in N_s}A'_{n_t,n_s}}{\sum_{n_t\in N_t}\mathrm{frac\_ext}(n_t)}
$$

với $A'$ là adjacency đã chuẩn hóa cạnh vào, $\mathrm{frac\_ext}(n_t)$ là tỉ lệ trọng số tuyệt đối vào $n_t$ đến từ ngoài supernode. Giả thuyết vì sao có nhiều feature tương tự: khái niệm khác nhau tinh tế (hoặc **feature manifold**), cùng khái niệm nhưng ở layer khác dựa trên input khác, hoặc cần nhiều feature để xấp xỉ phản ứng phi tuyến của MLP.

## 7. Kiểm chứng bằng can thiệp

Attribution graph chỉ là **giả thuyết** về local replacement model; phải kiểm chứng trên model gốc bằng cách sửa activation của feature và **tiêm decoder đã sửa** thay cho tái tạo gốc.

### 7.1 Constrained patching

Feature CLT ghi vào nhiều layer → phải chọn một **khoảng layer**. Cách làm: sửa decoding của feature ở mỗi layer trong khoảng, tính sẵn output MLP đã nhiễu một lần, **kẹp** (clamp) output MLP trong khoảng về giá trị đó, chạy forward từ layer cuối của khoảng. Không có hiệu ứng bậc hai trong khoảng patch. Phiên bản nhân: nhân activation với $M$ trong khoảng $[\ell-1,\ell]$.

Vì sao không đơn giản cộng decoder vào mỗi layer? Vì **double counting**: output tái tạo ở layer 2 có thể vốn là nguyên nhân của output ở layer 3 trong model gốc, cộng vào cả hai sẽ nhân đôi hiệu ứng.

### 7.2 Iterative patching

Tính lại các feature đọc trong khoảng patch, để hiệu ứng lan truyền ("knock-on"). Bắt được nhiều hiệu ứng hơn nhưng dễ **lỗi dây chuyền** do encode/decode lặp; kém chính xác hơn để kiểm chứng từng cạnh.

### 7.3 Những tinh tế

- Attention pattern **cũng cố định** khi can thiệp → không thấy hiệu ứng qua QK.
- Cạnh trực tiếp gần như **được đảm bảo** xác nhận (vì là hiệu ứng tuyến tính với attention cố định; chắc chắn khi khoảng patch kết thúc đúng ở layer encoder của đích). Thứ thực sự được kiểm là **hiệu ứng nối tiếp**: A → B → C → token X; ức chế A thì B giảm "miễn phí", câu hỏi là C và X có đổi như dự đoán không.
- **Hệ số steering** thường phải mạnh: ức chế bằng hệ số **−1** (đảo dấu) thay vì 0, hoặc lớn hơn (ví dụ −5). Lý do: phương sai chưa giải thích, chọn nhóm feature không đầy đủ, và cách steer chỉ đo hiệu ứng ở một khoảng layer → **đánh giá thấp** hiệu ứng toàn phần.
- **Định vị layer quan trọng**: quét layer kết thúc của khoảng patch; ví dụ steer âm feature "Analytics" làm giảm logit "dag" với hiệu ứng **bằng phẳng trước layer 13** rồi giảm dần về cuối, vì các feature trung gian "say \_A", "say DA\_", "say DAG" sống từ layer 13 trở đi và constrained patching không cho knock-on khi khoảng patch bao trùm chúng.

## 8. Ba case study

### 8.1 Acronym: "The National Digital Analytics Group (N" → "DAG"

Ba đường chính, mỗi đường từ một từ: feature cho từ ("Digital", "Analytics", "Group") → feature "nói chữ cái đầu của từ đó đúng vị trí" ("say \_D", "say \_A", …) → "say DAG" và logit. Cạnh "Word → say \_W" là attention OV ghi vào một subspace, được MLP ở vị trí đích khuếch đại. Có thêm feature "đang trong acronym" và "chữ N ở đầu acronym". Từ "National" ảnh hưởng nhỏ — giả thuyết: nó tác động qua attention pattern, thứ phương pháp không giải thích. Ức chế (hệ số −1) từng supernode từ làm giảm các feature chữ cái tương ứng; "say DA\_" bị ảnh hưởng bởi cả "Digital" lẫn "Analytics". (Tokenizer có token "Caps Lock" đặc biệt nên token đích thực là "dag".)

### 8.2 Factual recall: "Fact: Michael Jordan plays the sport of" → basketball (65%)

Hai đường song song: (i) "plays"/"sport" → "sport", "say a sport" → logit nhiều môn; (ii) "Michael Jordan và các người nổi tiếng" → feature bóng rổ → +basketball, −football. Một số nhóm có cạnh trực tiếp tới logit (chỉ qua attention OV). Ablate "sport" hoặc "Michael Jordan" có hiệu ứng lớn lên logit nhưng nhỏ lên supernode kia → xác nhận cấu trúc **song song**; ức chế "basketball discussion" cũng làm logit giảm mạnh. Supernode cho phép chứa feature polysemantic miễn có khía cạnh liên quan.

### 8.3 Addition: "calc: 36+59=" (Haiku 3.5)

Trực quan hóa mỗi feature ở token "=" bằng **operand plot** (lưới $100\times100$ input), **output weight plot**, **embedding weight plot**. Ba đường: chữ số hàng đơn vị (mod 10), độ chính xác vừa, độ chính xác thấp; giao thoa xây dựng ra tổng mod 100 và đáp án.

Phân loại feature: **sum features** (operand plot dạng đường chéo), **lookup table features** (dạng lưới: điều kiện trên $a$ AND điều kiện trên $b$), **add function features** (dải ngang/dọc), mostly active, miscellaneous. Thuộc tính: độ chính xác (đúng chữ số, khoảng rộng 2–50), modulo (10, 100, ít hơn 2, 5, 25, 50), pattern ("bắt đầu bằng 51").

**Ràng buộc hình bình hành** (vì pre-activation là hàm affine của hai operand, $f_{pre}(a,b)=g(a)+h(b)+c$):

$$
\boxed{f_{pre}(x+y)+f_{pre}(z+w)=f_{pre}(x+w)+f_{pre}(z+y)}
$$

Nếu hai vế trái dương thì ít nhất một số hạng vế phải dương. Ví dụ: feature "sum = 5" bắn cho $1+4$ và $2+3$ thì phải bắn cho $1+3$ hoặc $2+4$ → **không thể** tạo feature tổng tổng quát trong một bước từ input features; cần bước trung gian — chính là lookup table features (ví dụ "\_6 + \_9").

Can thiệp: ức chế feature chữ số đơn vị "\_6" tắt cả đường đơn vị (lookup "\_6+\_9", "sum=\_5", "sum=\_95") nhưng giữ đường độ lớn ("sum~92") → model trả **98** thay vì 95; ức chế "\_9" ra **91** (không phải 92, nên "đừng tin numerology"). Ức chế lookup "\_6+\_9" làm kết quả nhòe trên khoảng 5; ức chế "sum=\_95" nhòe rộng hơn.

## 9. Global weights

Attribution graph chỉ đúng cho một prompt. Muốn tương tác **toàn cục**, xét ba loại đường từ $s$ tới $t$: **residual-direct**, **attention-direct** (qua OV, bị scale bởi attention pattern), **indirect** (qua feature khác). Residual-direct = activation × **virtual weight** cố định:

$$
\boxed{V_{st}=\Big\langle\sum_{\ell\in L_{st}}W^{s,\ell}_{dec},\,W^{t}_{enc}\Big\rangle}
$$

($L_{st}$: các layer từ encoder của $s$ tới trước $t$.)

**Vấn đề interference**: hàng triệu feature cùng ghi vào residual → mọi cặp đều có virtual weight khác 0, kể cả cặp **không bao giờ đồng kích hoạt**. Ví dụ "Say a game name" có nhiều virtual weight lớn tới feature khó hiểu. Phụ lục trên 1000 feature (~150M token): ngay cả trong các weight dương cực lớn (vượt percentile 99.9999), một tỉ lệ đáng kể là "không dùng" (không đồng kích hoạt).

Hai cách xử lý: giới hạn vào miền hẹp (addition), hoặc dùng thống kê đồng kích hoạt:

$$
V^{ERA}_{ij}=\mathbb E\big[\mathbf 1(a_j>0)\,a_i\big]V_{ij},\qquad
\boxed{V^{TWERA}_{ij}=\frac{\mathbb E[a_ja_i]}{\mathbb E[a_j]}V_{ij}}
$$

**ERA** = expected residual attribution (trung bình độ mạnh đường residual-direct trên dữ liệu); **TWERA** = target-weighted ERA, trọng số theo activation đích vì activation nhỏ hay polysemantic. TWERA lọc bớt kết nối vô nghĩa, giữ kết nối dễ hiểu ("Ultimate frisbee", "Say a game name" khác), nhưng **phụ thuộc mạnh** vào đồng kích hoạt (nhiều virtual weight rất nhỏ lại có TWERA lớn) và **xử lý ức chế kém**.

**Global weights trong addition**: giới hạn ma trận virtual weight vào feature hoạt động trên ≥10 của 10,000 prompt `calc: a+b=`; hầu hết cạnh diễn giải được theo operand function: lookup "\_6+\_9" nuôi các sum "\_15", "\_25"…; "add \_9" nuôi các lookup "\_9+\_9", "\_0+\_9"; giao diện cho 2931 feature của 18L. Kết luận: từ "**bag of heuristics**" (Nikankin et al.) sang "**graph of heuristics**"; hầu như không feature nào chứa lời giải đầy đủ cho tới các layer cuối.

## 10. Metric của đồ thị

### 10.1 Ma trận ảnh hưởng gián tiếp

Lấy adjacency, lấy trị tuyệt đối, **chuẩn hóa cạnh vào mỗi node tổng bằng 1** → $A$ (chỉ số (đích, nguồn)). Ảnh hưởng qua mọi đường:

$$
\boxed{B=A+A^2+A^3+\cdots=(I-A)^{-1}-I}
$$

Vì đồ thị là DAG (cạnh chỉ đi từ layer sớm tới muộn), $A$ lũy linh (nilpotent) nên chuỗi Neumann hữu hạn và $(I-A)$ luôn khả nghịch (nhận xét của chúng tôi). **Logit influence** của node = trung bình các hàng logit của $B$, trọng số bằng xác suất token.

### 10.2 Path length

$B_\ell=\sum_{i=0}^{\ell}A^i$, $P_\ell=\sum_{e\in\text{emb}}B^\ell_{t,e}$ = ảnh hưởng qua đường dài $\le\ell$. Trên prompt "capital-analogy": **CLT 10M có độ dài đường trung bình 2.3**, **PLT 10M là 3.7**. Ví dụ "Zagreb:Croatia::Copenhagen:": PLT có đường dài 7 toàn feature "Copenhagen" (khuếch đại liên tiếp), CLT gom về feature layer 1.

Hai mặt: dễ diễn giải hơn, giảm "**chain-breaking**" (thiếu một mắt khuếch đại là đứt chuỗi); nhưng CLT có **cấu trúc nhân quả khác** model gốc — chuỗi Copenhagen↔Denmark tự củng cố trong PLT là vô hình trong CLT, nếu có thật trong model thì đó là **unfaithful**. Phụ lục: norm residual stream tăng theo cấp số nhân qua layer, nên model phải chủ động khuếch đại ở nhiều layer để giữ thông tin "đọc được"; CLT gom việc đó vào một feature.

### 10.3 Completeness và replacement score

- **Graph completeness score**: tỉ lệ cạnh vào (trọng số theo logit influence của node đích) đến từ feature/embedding thay vì error — "chấm điểm từng phần".
- **Graph replacement score**: tỉ lệ đường end-to-end (trọng số theo độ mạnh) đi từ embedding tới logit **chỉ qua feature** — thưởng cho giải thích trọn vẹn.

Viết thành công thức (cách chúng tôi hình thức hóa, nhất quán với chú thích rằng hệ số chuẩn hóa ở 10.2 "chính bằng replacement score"):

$$
S_{repl}=\sum_{e\in\text{emb}}\bar B_{\text{logit},e},\qquad
S_{comp}=\frac{\sum_t \mathrm{inf}(t)\sum_{s\notin\text{err}}A_{t,s}}{\sum_t\mathrm{inf}(t)}
$$

(vì hàng của $A$ tổng bằng 1, tổng ảnh hưởng từ mọi node nguồn — embedding và error — lên logit bằng 1).

| Replacement model | Completeness | Replacement | Graph |
|---|---|---|---|
| CLT 10M | 0.80 | 0.61 | uspto-telephone-clt |
| PLT 10M | 0.78 | 0.37 | uspto-telephone-plt |

Cải thiện lớn nhất đến từ **chuyển PLT → CLT**; tăng số feature cho lợi ích lớn nhưng giảm dần.

## 11. Graph pruning

Thuật toán hai bước:

1. **Prune node**: tính logit influence (mục 10.1), sắp giảm dần, giữ tập nhỏ nhất đạt **80%** tổng influence (ngưỡng mặc định 0.8).
2. **Prune edge**: tính lại $A$ và influence trên đồ thị đã prune; điểm cạnh = influence của node đích × trọng số chuẩn hóa; giữ cạnh đạt **98%** tổng điểm.
3. Logit: giữ top-$K$ token đạt tổng xác suất $>0.95$, $K\le10$. **Không prune** embedding và error node.

Kết quả điển hình: số node giảm **~10×**, số cạnh giảm **~500×**, trong khi phần hành vi giải thích (completeness) giảm khoảng **20%**. Còn có biến thể adaptive cho prompt dài (khám phá greedy top-$k$ node, node chưa khám phá tính như error).

| Ngưỡng prune | Completeness | Số node |
|---|---|---|
| 0.95 | 0.87 | 236 |
| 0.9 | 0.83 | 137 |
| 0.8 (mặc định) | 0.70 | 55 |
| 0.7 | 0.58 | 27 |

(prompt acronym "The International Advanced Security Group (IA" → "SG")

## 12. Đánh giá CLT và mechanistic faithfulness

### 12.1 Feature

- Định tính: 50 feature ngẫu nhiên mỗi model; loại: input, trừu tượng (ví dụ "nguy hiểm khi trộn hóa chất tẩy rửa"), chức năng ("add 9"), output ("say a capital"), polysemantic (ví dụ "rhythm" + Michael Jordan). Feature Haiku phong phú hơn 18L.
- Định lượng: 18L 10M feature: **normalized MSE ~11.5%, L0 = 88**; Haiku 30M: **21.7%, L0 = 235**. Tăng kích thước → Pareto-improvement (MSE, L0).
- Auto-interp bằng Claude: **Sort Eval** (đoán ví dụ thuộc feature nào trong một cặp) và **Contrastive Eval** (đoán prompt nào trong cặp gần giống làm feature bật). Cả hai tăng theo scale; **CLT > PLT >> thresholded neurons** trên mọi metric.

### 12.2 Faithfulness

- **Node → logit**: trên 20 prompt, ablate từng feature, đo KL: **indirect influence** dự đoán hiệu ứng ablation tốt nhất (tương quan Pearson log-log), hơn direct edge, hơn activation magnitude.
- **Feature → feature**: ablate trong khoảng $[i,i+2]$; Spearman **0.72** giữa influence và hiệu ứng ablation (chuẩn hóa, lấy trị tuyệt đối). Ít điểm "influence lớn nhưng không hiệu ứng" hay ngược lại.
- **Toàn bộ local replacement model**: ba loại perturbation (theo encoder vector, hướng ngẫu nhiên, feature upstream — loại "on-distribution" nhất). Một layer sau can thiệp: **cosine ~0.8, normalized MSE ~0.4** (với 10M ở 18L, cosine 60–80% tùy cấu hình). Sai lệch **cộng dồn qua layer**: hướng xấu dần từ từ, **độ lớn** có thể hỏng nghiêm trọng, tệ hơn với dictionary lớn; nghi do local replacement model thiếu mẫu số normalization. PLT: MSE thấp hơn ở một layer sau nhưng cộng dồn nhanh hơn chút sau vài layer.
- Vì sao case study vẫn thành công: lỗi ảnh hưởng độ lớn nhiều hơn hướng (bù bằng quét cường độ), supernode khử nhiễu, và can thiệp thường nhắm vào feature "cruxy" ở layer mạnh nhất (cộng với hiệu ứng "được đảm bảo").

## 13. Paper kiểm chứng như thế nào

| Câu hỏi | Bằng chứng | Con số chính |
|---|---|---|
| CLT có thay MLP được không? | Top-1 match của replacement model | 50% (18L lớn nhất) |
| Feature có thưa, chính xác? | (MSE, L0) | 11.5%/88 (18L), 21.7%/235 (Haiku) |
| Feature có diễn giải được? | Sort/Contrastive Eval, 50 feature ngẫu nhiên | CLT > PLT >> neurons |
| Đồ thị có gọn? | Path length | 2.3 vs 3.7 |
| Đồ thị có đủ? | Completeness / replacement | 0.80/0.61 vs 0.78/0.37 |
| Pruning mất gì? | Frontier node vs completeness | ~10× ít node, −20% |
| Influence có phản ánh nhân quả? | Ablation KL, ablation feature→feature | Spearman 0.72 |
| Cơ chế có giống model thật? | Perturbation local replacement vs model | cos ~0.8, NMSE ~0.4 sau 1 layer; xấu dần |
| Case study có đúng? | Constrained patching supernode | 36+59: 98 khi tắt "\_6" |

Tập đánh giá đồ thị: **n = 260** token từ Pile (bỏ books), lọc: model đoán đúng top-1, density $<10^{-4}$, token không xuất hiện trong context, loss $>0.2$.

## 14. Giới hạn

Bảy giới hạn tác giả nêu:

1. **Thiếu attention circuit**: không giải thích QK. Ví dụ induction "Aunt Sally… Aunt" → đồ thị chỉ nói "đoán Sally vì có token Sally trước đó", bỏ lỡ toàn bộ cơ chế; câu hỏi trắc nghiệm "(B) 1945" → đồ thị chỉ nói "token sau (b)", trong khi hành vi đi qua attention, với ba feature "đáp án đúng" mà can thiệp cho thấy quan trọng (bật chúng ở đáp án C làm model trả "C"). Hướng tương lai: QK là dạng toàn phương trên residual → attribution cho **cặp** feature query–key (bùng nổ bậc hai), lưu ý **attention superposition**.
2. **Reconstruction error và "dark matter"**: error node "từ trên trời rơi xuống", không diễn giải được. Ví dụ jailbreak viết hoa lộn xộn "TAhT SEaRChEs…" → gần như mọi node trên token bị làm rối là error node. Thêm nguồn sai: nhãn của người quá thô so với profile kích hoạt thật. Hướng: scale, sửa kiến trúc, huấn luyện end-to-end thay vì chỉ MSE, finetune trên phân phối quan tâm, attribution ngược từ error node.
3. **Feature không hoạt động và mạch ức chế**: đồ thị chỉ hiện feature *đang* hoạt động, nhưng "can't answer" bị ức chế bởi feature thực thể quen thuộc mới là mấu chốt của hallucination. Cách hiện tại: **cặp prompt tương phản** (Michael Jordan vs "Michael Batkin"). Ở mức global, lọc interference bằng đồng kích hoạt sẽ bỏ sót weight ức chế.
4. **Độ phức tạp đồ thị**: sau pruning vẫn hàng trăm feature, hàng nghìn cạnh; khái niệm trải trên nhiều feature; nhiều input nhỏ; đường nhiều độ dài và khác dấu.
5. **Feature sai mức trừu tượng**: feature splitting và absorption. Ví dụ 12 prompt Haiku kết thúc bằng "during": **không** có feature nào bật cho cả 12; tính tổng quát giảm khi CLT lớn hơn. Cần công cụ phân cấp (Matryoshka SAE, meta-SAE).
6. **Global circuits khó**: interference weights (ví dụ feature "say 15" có top logit gồm "gag", "duty", "temper"…), và tương tác qua attention không có trong global weights.
7. **Mechanistic faithfulness**: kể cả MSE = 0 trên phân phối huấn luyện, CLT có thể học hàm khác; chỉ kiểm hậu kỳ, và kiểm chứng hiện tại còn thô (perturb cả supernode, chọn layer cho hiệu ứng lớn nhất).

## 15. Lựa chọn thiết kế bền vững và tạm thời

Tác giả tin là **bền vững**:

- **Feature học được thay vì neuron**: threshold neuron để chỉ giữ vùng diễn giải được làm hỏng hành vi nhiều hơn transcoder/CLT.
- **Transcoder thay vì residual SAE**: SAE phân rã *trạng thái*, transcoder phân rã *phép tính*; tương tác giữa feature SAE bị MLP phi tuyến chen giữa.
- **Cross-layer thay vì per-layer**: bằng chứng cross-layer superposition qua Pareto path length vs influence.
- **Hiệu ứng tuyến tính trực tiếp** thay vì attribution phi tuyến (integrated gradients, Shapley, ablation). Độ phức tạp phi tuyến "đi đâu"? (a) thành đường nhiều bước trong đồ thị (tốt), (b) vào phần bị cố định (attention, normalization — câu hỏi chưa trả lời), (c) xấu: CLT đi đường tắt tuyến tính không faithful.

**Tạm thời/tiện lợi**: gộp mọi đường attention vào một cạnh (mất thông tin head nào trung gian, như induction/successor head); bỏ qua QK; chỉ huấn luyện bằng MSE + sparsity, không tối ưu trực tiếp cho đồ thị thưa hay faithfulness. "North Star": đơn vị diễn giải toàn cục, cạnh diễn giải được và **đúng toàn cục**; paper mới đạt đồ thị đúng **cục bộ**.

## 16. Ứng dụng giả thuyết cho pruning và distillation

> **Toàn bộ mục này là suy luận của chúng tôi, không phải kết quả paper.**

### 16.1 Influence làm điểm quan trọng khi prune

Paper cho thấy **logit influence qua mọi đường** dự đoán hiệu ứng ablation tốt hơn activation magnitude và direct logit attribution. Đây chính là điều cắt neuron cần: neuron ít đóng góp trực tiếp nhưng là mắt xích trung gian sẽ bị điểm magnitude/direct đánh giá thấp. Không cần CLT: paper nói có thể dựng attribution graph với **neuron** (baseline thresholded neurons, và đã mở rộng phương pháp cho neuron). Đề xuất: điểm quan trọng của neuron $n$ = trung bình logit influence của nó trên tập prompt OCR. Lưu ý: influence là **không dấu** và tính trên đồ thị đã stop-grad attention, nên chưa phản ánh vai trò của neuron trong việc tạo attention pattern.

### 16.2 Error node như công cụ đo $D_{lin}/D_{res}$

Tương tự local replacement model, xem MLP đã prune của student như "replacement" cho MLP của parent tại một layer, với error node

$$
\varepsilon_{c,\ell}=\mathrm{MLP}^{P}_\ell(\mathbf x_{c,\ell})-\mathrm{MLP}^{pruned}_\ell(\mathbf x_{c,\ell}).
$$

Tách $\varepsilon=\varepsilon_{lin}+\varepsilon_{res}$, với $\varepsilon_{lin}$ là phần map tuyến tính tối ưu tại điểm cắt dự đoán được ($D_{lin}$) và $\varepsilon_{res}$ là phần dư ($D_{res}$). Đưa $\varepsilon_{lin}$, $\varepsilon_{res}$ thành **hai loại error node** rồi tính replacement score như mục 10.3: ta biết phần ảnh hưởng lên **logit OCR** chảy qua $D_{res}$, thay vì chỉ một con số MSE. Hai mức prune cùng $D_{res}$ (MSE) có thể khác xa nhau về influence lên logit.

### 16.3 Vì sao cắt layer dễ phục hồi hơn cắt neuron

Paper nêu hai cơ chế ủng hộ quan sát của bạn: (i) **cross-layer superposition** — nhiều MLP cùng thực hiện một bước; (ii) **khuếch đại lặp lại** do norm residual tăng theo cấp số nhân (PLT thấy chuỗi 7 feature Copenhagen, CLT gom về một). Bỏ một layer chủ yếu bỏ một "bộ khuếch đại" dư thừa — dễ bù bằng scale/map tuyến tính ($D_{lin}$). Cắt neuron trong mọi layer phá các **lookup-table-like features** (kiểu "\_6 + \_9") vốn là bước trung gian bắt buộc theo ràng buộc hình bình hành — không map tuyến tính nào tái tạo được ($D_{res}$). Đây là giả thuyết kiểm chứng được (mục 17).

### 16.4 OCR là bài toán attention-heavy

OCR phần lớn là **copy** từ vision token sang text qua attention, giống induction "Aunt Sally". Dự đoán: attribution graph cho token OCR sẽ chủ yếu có cạnh "feature của glyph ở vision token → say X" và **không cho biết** vì sao model đọc đúng patch. Vậy đồ thị hữu ích hơn cho các quyết định *không phải copy*: ngôn ngữ (chính tả, dấu tiếng Việt dựa trên ngữ cảnh), cấu trúc bảng/markdown, công thức. Với copy circuit, cần phân tích QK riêng.

### 16.5 Mở rộng cho GatedDeltaNet

Điều kiện để cạnh tuyến tính là **cố định mọi phi tuyến trừ pre-activation của feature**. Với linear attention dạng gated delta rule (dạng tổng quát; cần đối chiếu cài đặt thật của LightOnOCR), $S_t=\alpha_tS_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top$, $o_t=S_tq_t$: nếu cố định (stop-grad) $\alpha_t,\beta_t,k_t,q_t$, output gate và mẫu số norm, thì $o_t$ **tuyến tính** theo các $v_{t'}=W_Vx_{t'}$ — vai trò như OV với attention pattern cố định. Mọi phi tuyến trên nhánh $v$ (ví dụ activation sau short conv nếu có) cũng phải cố định dưới dạng hệ số. Khi đó Jacobian $J^{\blacktriangledown}$ vẫn tính bằng một backward pass, và phương pháp áp dụng được cho decoder lai. Phần "QK" tương ứng (gate, key, query) vẫn là vùng mù như giới hạn 1.

### 16.6 Bài học cho đánh giá KD và ablation

- **Double counting**: khi đo thiệt hại theo layer bằng cách tiêm/khôi phục activation, dùng tinh thần constrained patching để không đếm hai lần.
- **Lỗi cộng dồn**: faithfulness tốt sau 1 layer nhưng xấu dần — tương tự error của student cộng dồn; hidden-state KD theo từng layer (giống CLT tái tạo mọi layer) có thể giúp, nhưng paper cũng cho thấy khớp từng layer **không đảm bảo** cùng cơ chế.
- **Graph diff**: giao diện nội bộ có "diff view" so sánh hai attribution graph; muốn so parent–student theo feature cần một từ điển chung (crosscoder, note 14) hoặc so ở mức node logit/embedding và error.

## 17. Thí nghiệm gợi ý

1. **Pruning theo influence (không cần CLT)**: trên decoder của parent, dựng attribution graph mức neuron (stop-grad attention/GatedDeltaNet gate, norm) cho ~50 prompt OCR có token khó (chữ có dấu, số trong bảng). Điểm neuron = logit influence trung bình. So với prune theo magnitude/activation ở cùng tỉ lệ: đo $D_{lin}$, $D_{res}$, CER sau heal.
2. **Error node $D_{lin}/D_{res}$**: với một student đã cắt neuron ở layer $\ell$, tạo hai loại error node như 16.2, tính replacement score và tỉ lệ influence lên logit qua $\varepsilon_{res}$; lặp lại với student cắt layer có cùng số tham số. Giả thuyết: cắt layer có influence qua $\varepsilon_{res}$ thấp hơn.
3. **PLT nhỏ cho parent**: huấn luyện PLT (rẻ hơn CLT) trên vài layer MLP decoder; dựng đồ thị cho prompt OCR; kiểm tra (a) đồ thị có bị chi phối bởi đường attention-copy không, (b) feature nào là "bước trung gian bắt buộc" (lookup-like) — đó là thứ cần bảo vệ khi prune.

## 18. Tóm tắt công thức và khái niệm

| Khái niệm | Công thức | Ý nghĩa |
|---|---|---|
| CLT encoder | $\mathbf a^\ell=\mathrm{JumpReLU}(W^\ell_{enc}\mathbf x^\ell)$ | Feature đọc residual một layer |
| CLT decoder | $\hat{\mathbf y}^\ell=\sum_{\ell'\le\ell}W^{\ell'\to\ell}_{dec}\mathbf a^{\ell'}$ | Ghi vào mọi MLP phía sau |
| Sparsity | $\lambda\sum_\ell\sum_i\tanh(c\|\mathbf W^\ell_{dec,i}\|a^\ell_i)$ | Gần L0, có trọng số decoder |
| Error node | $\mathrm{MLP}_\ell(\mathbf x)-\mathrm{CLT}_\ell(\mathbf x)$ | Làm local model khớp chính xác |
| Edge | $A_{s\to t}=a_s\sum_\ell (W^{\ell_s\to\ell}_{dec,s})^\top J^{\blacktriangledown}W^{\ell_t}_{enc,t}$ | Ảnh hưởng tuyến tính trực tiếp |
| Phân rã | $h_t=\sum_s A_{s\to t}$ | Giải thích đầy đủ pre-activation |
| Virtual weight | $V_{st}=\langle\sum_{\ell}W^{s,\ell}_{dec},W^t_{enc}\rangle$ | Global weight residual-direct |
| TWERA | $\frac{\mathbb E[a_ja_i]}{\mathbb E[a_j]}V_{ij}$ | Lọc interference |
| Influence | $B=(I-A)^{-1}-I$ | Ảnh hưởng qua mọi đường |
| Path length | $P_\ell=\sum_e B^\ell_{t,e}$ | Độ phức tạp đồ thị |
| Completeness | phần cạnh vào không từ error | Điểm từng phần |
| Replacement | phần đường emb→logit chỉ qua feature | Giải thích trọn vẹn |
| Pruning | node 0.8, edge 0.98, logit 0.95 (≤10) | ~10× node, ~500× cạnh |
| Constrained patching | kẹp MLP output trong khoảng layer | Tránh double counting |
| Parallelogram | $f(x{+}y)+f(z{+}w)=f(x{+}w)+f(z{+}y)$ | Cần lookup-table features |

### Kết luận ngắn

Paper biến câu hỏi "model làm gì trên prompt này" thành một đối tượng toán học rõ ràng: thay MLP bằng CLT, cố định attention và normalization, thêm error node, ta được một mạng tuyến tính-từng-đoạn khớp chính xác model trên prompt đó, trong đó mỗi pre-activation là tổng các cạnh. Pruning theo influence và can thiệp constrained patching biến đồ thị thành giả thuyết kiểm chứng được. Giới hạn lớn nhất là vùng mù QK/attention, "dark matter" trong error node, và faithfulness giảm dần qua layer. Với prune/distill, giá trị thực tế nhất là: (1) influence qua mọi đường như điểm quan trọng, (2) error node như cách đặt $D_{lin}/D_{res}$ vào ngữ cảnh ảnh hưởng lên logit, (3) cảnh báo rằng OCR là bài toán attention-heavy mà phương pháp này nhìn kém.

**Đọc bản gốc:** https://transformer-circuits.pub/2025/attribution-graphs/methods.html
