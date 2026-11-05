# Toy Models of Superposition (2022)

**Tác giả:** Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, Christopher Olah (Anthropic, Harvard). Xuất bản 14/09/2022.  
**Nguồn:** https://transformer-circuits.pub/2022/toy_model/index.html  
**Chủ đề:** Superposition, polysemantic neuron, sparsity và importance của feature, phase diagram, hình học feature (polytope), privileged basis, tính toán trong superposition, chiến lược "giải superposition".  
**Vị trí trong lộ trình:** Đọc sau *A Mathematical Framework for Transformer Circuits* (2021) để đã quen residual stream và ý "basis có đặc quyền hay không". Đây là paper nền cho mọi ghi chú 06–09 và cho hướng sparse autoencoder sau này; với dự án pruning của bạn, đây là paper giải thích rõ nhất *vì sao cắt neuron MLP lại phá feature*.

> Tài liệu học tập bằng tiếng Việt. Đây là ghi chú diễn giải, không phải bản dịch nguyên văn paper. Những phần liên hệ tới pruning/distillation là suy luận mở rộng của người viết, không phải kết quả paper chứng minh.

## Mục lục

1. [Câu hỏi trung tâm và các khái niệm nền](#1-câu-hỏi-trung-tâm-và-các-khái-niệm-nền)
2. [Thiết lập toy model](#2-thiết-lập-toy-model)
3. [Kết quả cơ bản: ReLU tạo ra superposition](#3-kết-quả-cơ-bản-relu-tạo-ra-superposition)
4. [Hiểu bằng toán: feature benefit và interference](#4-hiểu-bằng-toán-feature-benefit-và-interference)
5. [Superposition là một phase change](#5-superposition-là-một-phase-change)
6. [Hình học của superposition](#6-hình-học-của-superposition)
7. [Learning dynamics](#7-learning-dynamics)
8. [Superposition và adversarial example](#8-superposition-và-adversarial-example)
9. [Superposition trong privileged basis](#9-superposition-trong-privileged-basis)
10. [Tính toán trong superposition](#10-tính-toán-trong-superposition)
11. [Bức tranh chiến lược](#11-bức-tranh-chiến-lược)
12. [Paper kiểm chứng như thế nào](#12-paper-kiểm-chứng-như-thế-nào)
13. [Giới hạn](#13-giới-hạn)
14. [Ứng dụng giả thuyết cho pruning và distillation](#14-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
15. [Thí nghiệm gợi ý](#15-thí-nghiệm-gợi-ý)
16. [Tóm tắt công thức và khái niệm](#16-tóm-tắt-công-thức-và-khái-niệm)

---

## 1. Câu hỏi trung tâm và các khái niệm nền

**Câu hỏi:** Vì sao *đôi khi* một neuron tương ứng gọn với một khái niệm (monosemantic), còn *thường* (nhất là trong LLM) một neuron lại phản ứng với nhiều thứ không liên quan (polysemantic)?

Câu trả lời paper đề xuất: có hai lực ngược chiều.

- **Privileged basis** (cơ sở có đặc quyền: các trục tọa độ được kiến trúc làm cho "đặc biệt", ví dụ vì có activation áp theo từng tọa độ) kéo feature về thẳng hàng với neuron.
- **Superposition** (chồng chất: biểu diễn *nhiều feature hơn số chiều*) đẩy feature ra khỏi trục neuron.

### 1.1 Feature và linear representation

- **Feature**: một thuộc tính của input mà mạng biểu diễn. Paper không chốt định nghĩa; định nghĩa làm việc là "thuộc tính mà một mạng *đủ lớn* sẽ dành riêng một neuron".
- **Linear representation** (biểu diễn tuyến tính): mỗi feature $f_i$ ứng với một hướng $W_i$; nhiều feature cùng bật được biểu diễn bằng tổng:

$$
\boxed{h = \sum_i x_{f_i} W_{f_i}}
$$

Bản thân $x_{f_i}$ có thể là hàm phi tuyến của input; chỉ ánh xạ *feature → activation* là tuyến tính. Lý do paper tin vào tuyến tính: layer là linear + nonlinearity, nên feature tuyến tính thì "linearly accessible" (layer sau chọn được bằng một phép chiếu), và có lợi thống kê.

### 1.2 Privileged vs non-privileged basis

Nếu một biểu diễn chỉ bị đọc/ghi bằng phép tuyến tính, ta có thể chèn $M$ và $M^{-1}$ mà mô hình không đổi: các trục không có gì đặc biệt (**non-privileged**, ví dụ word embedding, residual stream). Khi có activation theo từng tọa độ (ReLU trong MLP), đối xứng bị phá và trục trở nên đặc biệt (**privileged**). Chỉ trong privileged basis mới có nghĩa khi hỏi "neuron này có interpretable không". Có privileged basis **không đảm bảo** feature thẳng hàng với neuron.

### 1.3 Superposition hypothesis

Mạng "muốn" biểu diễn nhiều feature hơn số neuron. Hai sự kiện toán học khiến điều này khả thi:

- **Almost orthogonal vectors**: trong $\mathbb R^n$ chỉ có $n$ vector trực giao, nhưng có tới $\exp(n)$ vector *gần* trực giao (cosine $<\epsilon$), xem bổ đề Johnson–Lindenstrauss.
- **Compressed sensing** (cảm biến nén: khôi phục vector thưa từ phép chiếu xuống chiều thấp): nếu vector gốc thưa thì thường khôi phục được.

Cái giá là **interference** (nhiễu chéo): khi feature A bật, các feature không trực giao với A cũng "bật nhẹ". Sparsity làm nhiễu hiếm khi xảy ra, còn nonlinearity cho phép lọc bỏ nhiễu nhỏ. Hình ảnh then chốt: một mạng nhỏ đang **mô phỏng có nhiễu một mạng lớn hơn, thưa hơn**.

### 1.4 Thang bốn tính chất

| Tính chất | Nghĩa | Paper đánh giá |
|---|---|---|
| Decomposability | Activation phân rã được thành feature độc lập | Giả thuyết là phổ biến |
| Linearity | Feature là hướng | Giả thuyết là phổ biến |
| Superposition | $W^\top W$ **không** khả nghịch (nhiều feature hơn chiều) | Chỉ đôi khi |
| Basis-aligned | Mọi $W_i$ là one-hot (cần privileged basis) | Chỉ đôi khi |

---

## 2. Thiết lập toy model

Mục tiêu: chiếu $x\in\mathbb R^n$ xuống $h\in\mathbb R^m$ ($m<n$) rồi khôi phục lại, như một autoencoder.

### 2.1 Dữ liệu tổng hợp

Mỗi chiều $x_i$ được coi là một feature của "mạng lớn lý tưởng" (đã disentangled). Ba giả định:

- **Sparsity** $S_i$: $x_i=0$ với xác suất $S_i$, ngược lại $x_i\sim U[0,1]$. Thường dùng $S_i=S$ cho mọi feature. *Mật độ* là $1-S$.
- **Nhiều feature hơn neuron**: $n>m$.
- **Importance** $I_i$: hệ số nhân vào MSE, mô phỏng việc feature khác nhau đóng góp khác nhau vào loss (thường $I_i=r^i$, giảm dần).

### 2.2 Hai mô hình

$$
\text{Linear: } x'=W^\top W x+b, \qquad
\boxed{\text{ReLU output: } x'=\operatorname{ReLU}(W^\top W x+b)}
$$

- Cột $W_i\in\mathbb R^m$ là hướng biểu diễn feature $i$.
- Dùng chung $W$ và $W^\top$ (tied) để không mơ hồ về hướng của feature.
- Bias cho phép đặt feature không được biểu diễn về giá trị kỳ vọng, và quan trọng hơn: **bias âm giúp cắt nhiễu nhỏ**.

### 2.3 Loss

$$
\boxed{L=\sum_x\sum_i I_i\,(x_i-x'_i)^2}
$$

---

## 3. Kết quả cơ bản: ReLU tạo ra superposition

Cách đọc kết quả: vẽ $W^\top W$ (ma trận feature × feature) và $b$. Hai chỉ số:

- $\lVert W_i\rVert$: feature $i$ có được biểu diễn không (≈1 có, ≈0 không).
- $\sum_{j\neq i}(\hat W_i\cdot W_j)^2$: feature $i$ có *chia sẻ* chiều với feature khác không (0 = trực giao; $\ge 1$ = có nhóm feature khác kích hoạt hướng $W_i$ mạnh như chính nó).

Thí nghiệm ($n=20$, $m=5$, $I_i=0.7^i$):

- **Linear model** luôn học đúng top-$m$ feature quan trọng nhất, giống PCA; kết quả không phụ thuộc sparsity.
- **ReLU output model** với dữ liệu dày ($1-S=1$) cũng giống linear. Khi tăng sparsity, **superposition xuất hiện**: bắt đầu từ feature ít quan trọng, dần lan lên feature quan trọng hơn. Ban đầu là **antipodal pair** (hai feature có vector ngược dấu nhau), rồi chuyển sang cấu trúc hình học khác.
- Với $n=80$, $m=20$, $I_i=0.9^i$: kết quả gần như bản phóng to của trường hợp nhỏ.

### Ví dụ tự tính: 2 feature trong 1 chiều

Lấy $W=[1,-1]$ (antipodal), $b=0$:

| Input $x$ | $h=Wx$ | $x'=\operatorname{ReLU}(W^\top h)$ | Đúng? |
|---|---|---|---|
| $(a,0)$ | $a$ | $(a,0)$ | Đúng |
| $(0,c)$ | $-c$ | $(0,c)$ | Đúng |
| $(a,c)$ | $a-c$ | $(\operatorname{ReLU}(a-c),\operatorname{ReLU}(c-a))$ | Sai: nhiễu |

ReLU "nuốt" phần âm nên khi chỉ một feature bật thì khôi phục hoàn hảo. Lỗi chỉ xảy ra khi *cả hai cùng bật*, với xác suất $(1-S)^2$. Sparsity càng cao, cái giá càng nhỏ so với lợi ích biểu diễn thêm một feature. Mô hình tuyến tính không làm được vậy vì không có ReLU để bỏ phần âm.

---

## 4. Hiểu bằng toán: feature benefit và interference

### 4.1 Trường hợp tuyến tính

Dựa trên phân tích kiểu Saxe et al., loss của mô hình $x'=W^\top Wx$ (với $x$ Gaussian) có dạng:

$$
\boxed{L\sim\underbrace{\sum_i I_i\left(1-\lVert W_i\rVert^2\right)^2}_{\text{feature benefit}}+\underbrace{\sum_{i\neq j} I_j\,(W_j\cdot W_i)^2}_{\text{interference}}}
$$

- **Feature benefit**: giá trị thu được khi biểu diễn một feature.
- **Interference**: phạt khi hai feature không trực giao. Trong mô hình tuyến tính, cân bằng này khiến **không bao giờ đáng** biểu diễn nhiều feature hơn số chiều.

Ghi chú của paper: interference tuyến tính $\sum_{i\ne j}\lvert W_i\cdot W_j\rvert^2$ là chuẩn $L^2$, còn *coherence* trong compressed sensing $\max_{i\ne j}\lvert W_i\cdot W_j\rvert$ là chuẩn $L^\infty$ của cùng vector.

### 4.2 Trường hợp ReLU output

Khai triển loss theo số feature đang bật (nhị thức của $((1-S)+S)^n$):

$$
L=(1-S)^nL_n+\dots+(1-S)S^{n-1}L_1+S^nL_0 .
$$

Khi $S\to1$, $L_1$ và $L_0$ chiếm ưu thế. $L_0$ chỉ phạt bias dương ($\sum_i\operatorname{ReLU}(b_i)^2$). Phần thú vị là $L_1$ (input chỉ có 1 feature bật); xét $x_i=1$:

$$
\boxed{L_1\big|_{x_i=1}=\sum_i I_i\left(1-\operatorname{ReLU}(\lVert W_i\rVert^2+b_i)\right)^2+\sum_{i\neq j}I_j\operatorname{ReLU}(W_j\cdot W_i+b_j)^2}
$$

Hệ quả quan trọng:

- **Interference âm là miễn phí** trong trường hợp 1-sparse, vì ReLU cắt về 0. Do đó nghiệm ưa xếp feature sao cho nhiễu âm (antipodal).
- **Bias âm** biến nhiễu dương nhỏ thành âm, tức là "lọc nhiễu".
- Nếu importance đều, $\lVert W_i\rVert\in\{0,1\}$ và $b=0$, số hạng interference trở thành **bài toán Thomson** tổng quát (xếp điểm trên mặt cầu với một hàm năng lượng lạ). Phép so sánh này dẫn đường cho phần hình học.
- Với input nhiều feature cùng bật, nhiễu cộng dồn. Feature thường tự xếp thành **đồ thị nhiễu thưa** để giảm khả năng cộng dồn.

---

## 5. Superposition là một phase change

Với mỗi feature có ba kết cục: (1) không học, (2) học trong superposition, (3) có chiều riêng. Chuyển tiếp giữa chúng trông **đột ngột**: đây là "phase change" theo nghĩa rộng (thay đổi không liên tục).

**Thí nghiệm 2 feature → 1 chiều.** Feature 1 có importance 1.0. Trục ngang quét importance của feature "extra" từ 0.1 đến 10; trục còn lại quét mức thưa (bài ghi "từ 1.0 đến 0.01"). Mỗi điểm huấn luyện 10 mô hình, bỏ mô hình có loss cao nhất, rồi lấy trung bình.

**"Toy model của toy model"**: so loss giải tích của ba cấu hình:

| Cấu hình | Ý nghĩa |
|---|---|
| $W=[1,0]$ | Bỏ feature extra |
| $W=[0,1]$ | Bỏ feature 1, cho feature extra chiều riêng |
| $W=[1,-1]$ | Antipodal: superposition, mất khả năng biểu diễn $[1,1]$ |

Kết quả: cả phase diagram thực nghiệm lẫn lý thuyết đều cho thấy **sparsity là điều kiện cần**, nhưng nó tương tác với importance tương đối. Trong mô hình lý thuyết có thể xác nhận đây là **phase change bậc nhất**: các đường loss giao nhau nên đạo hàm của loss tối ưu bị gián đoạn.

**3 feature → 2 chiều**: thêm các nghiệm "W bỏ qua hướng nào" ($W\perp[0,0,1]$, $W\perp[1,0,0]$, $W\perp[0,1,1]$, $W\perp[1,1,0]$). Ranh giới pha vẫn rõ ràng, nhưng phần sau cho thấy cấu trúc thật phức tạp hơn nhiều.

> **Phần bình luận** (Tom McGrath): giải chính xác trường hợp $n=2, m=1$ (bỏ bias) tìm thêm một pha "confused feature" $W_1\simeq W_2\simeq 1/\sqrt2$ khi sparsity thấp và cả hai feature đều quan trọng, và cho thấy có chỗ hai cực tiểu cùng tồn tại.

---

## 6. Hình học của superposition

### 6.1 Uniform superposition và "dimensions per feature"

Mọi feature giống nhau ($I_i=1$, cùng $S$), $n=400$, $m=30$. Paper cho biết $n$ không quan trọng miễn $n\gg m$, và **gấp đôi $m$ thì số feature học được cũng gấp đôi**.

Vì $\lVert W_i\rVert^2\approx1$ nếu feature được biểu diễn và $\approx0$ nếu không, $\lVert W\rVert_F^2$ xấp xỉ số feature đã học. Đại lượng:

$$
D^*=\frac{m}{\lVert W\rVert_F^2}\quad(\text{số chiều trên mỗi feature}).
$$

Đồ thị $D^*$ theo sparsity bị **"dính" ở 1 và 1/2**. Mức 1/2 ứng với antipodal pair: hai feature chia một chiều.

### 6.2 Feature dimensionality

$$
\boxed{D_i=\frac{\lVert W_i\rVert^2}{\sum_j(\hat W_i\cdot W_j)^2}}
$$

- Tử số: feature $i$ được biểu diễn mạnh đến đâu.
- Mẫu số: bao nhiêu feature chia chung hướng của nó (tính cả chính nó).
- Antipodal: $D=1/(1+1)=1/2$. Không học: $D=0$.
- Thực nghiệm: khi feature "xếp hiệu quả", **tổng $D_i$ ≈ số chiều $m$**. (Bình luận của Fred Zhang giải thích bằng *leverage score*: khi các vector gần isotropic, $D_i$ trùng leverage score, và tổng leverage score bằng rank.)

### 6.3 Các polytope

Vẽ $D_i$ của từng feature theo sparsity, các điểm **tụ ở những phân số cụ thể**, mỗi phân số ứng với một cấu hình hình học:

| $D_i$ | Cấu hình | Số feature/chiều trong nhóm |
|---|---|---|
| 1 | Chiều riêng (khó thấy ở vùng dày vì basis không đặc quyền) | 1 feature / 1 chiều |
| 3/4 | Tetrahedron (tứ diện) | 4 feature / 3 chiều |
| 2/3 | Triangle (tam giác) | 3 feature / 2 chiều |
| 1/2 | Digon (antipodal pair) | 2 feature / 1 chiều |
| 2/5 | Pentagon (ngũ giác) | 5 feature / 2 chiều |
| 3/8 | Square antiprism | 8 feature / 3 chiều |
| 0 | Không học | |

Diễn giải:

- Superposition **không phải một thứ duy nhất**: mọi giá trị $0<D_i<1$ đều là superposition, và có nhiều "pha" bên trong (như nước đá có nhiều cấu trúc tinh thể).
- Nhiều cấu hình là nghiệm bài toán Thomson. Nghiệm *đều* (uniform polyhedra) cho đường riêng; nghiệm không đều lại tách thành các đường của thành phần đều (ví dụ triangular bipyramid hiện thành điểm ở 2/3 và 1/2 thay vì 3/5).
- **Tegum product** (ghép hai polytope trong hai không gian con trực giao): giữa các "tegum factor" không có nhiễu. Paper đoán nghiệm nhiều chiều thường là tegum product của các nghiệm 1D, 2D, 3D, và mô hình ưa điều này vì nhiều feature cùng nhiễu nhau rất tốn kém.

### 6.4 Polytope ↔ ma trận low-rank

Có tương ứng chính xác giữa polytope và ma trận dạng $W^\top W$ (đối xứng, nửa xác định dương, rank $m$): cột của $W$ là $n$ điểm trong $\mathbb R^m$, và phần ngoài đường chéo của $W^\top W$ do hình học quyết định. Ví dụ: mọi cách đặt 3 feature vào 2 chiều là một tam giác. Chiều ngược lại: chiếu các vector cơ sở lên phần bù trực giao của $(1,1,\dots,1)$ cho ra simplex đều, tức "superposition tối thiểu" khi feature đều và thưa.

### 6.5 Non-uniform superposition

- **Thay đổi importance hoặc sparsity của một feature** làm polytope **biến dạng mượt**, đến một điểm thì **"bật" sang polytope khác**. Thí nghiệm $n=5$, $m=2$, $1-S=0.05$: ngũ giác đều; làm một feature dày hơn thì các điểm khác "đẩy ra" nhường chỗ; làm thưa hơn thì nó chiếm ít chỗ; đủ thưa thì sụp thành hai digon, với feature thưa ở 0 (phase change bậc nhất, thấy qua giao nhau của loss).
- Điểm ngũ giác không nằm trên đường tròn đơn vị vì mô hình đặt bias hơi âm để cắt nhiễu và bù bằng $\lVert W_i\rVert=1/(1-b_i)$.
- **Feature tương quan** (cùng bật): ưu tiên trực giao (khác tegum factor); không được thì đặt cạnh nhau (nhiễu dương); thiếu chỗ thì **gộp thành principal component** $(a+b)/\sqrt2$. Khi dữ liệu dày hơn, nghiệm dần thành PCA.
- **Feature phản tương quan** (không cùng bật): ưa nằm cùng tegum factor, lý tưởng là antipodal.
- **Local almost-orthogonal basis**: các nhóm feature tương quan tự tạo cơ sở gần trực giao cục bộ, dù toàn mô hình đang superposition. Gợi ý PCA có thể dùng được trên phân phối con hẹp.
- Kết luận: **PCA và superposition là hai chiến lược bổ sung**; tương quan nhiều thì PCA thắng, thưa nhiều thì superposition thắng.

---

## 7. Learning dynamics

- **"Energy level" jumps**: trong quá trình huấn luyện, $D_i$ của feature **nhảy rời rạc** giữa các giá trị và đổi chỗ cho nhau; mỗi lần nhảy thì loss giảm đột ngột. Paper nghi rằng đường loss "mượt" ở mô hình lớn thực ra gồm rất nhiều bước nhảy nhỏ như vậy (liên hệ induction-head phase change và grokking).
- **Học như chuỗi biến đổi hình học**: với $n=6$, $m=3$ (2 nhóm 3 feature tương quan), quá trình đi qua các pha rõ trên loss, mỗi pha là một phép biến đổi hình học, thường kết thúc ở bát diện. Mô hình học một nghiệm tuyến tính kiểu PCA trước rồi mới sang nghiệm phi tuyến tốt hơn.

---

## 8. Superposition và adversarial example

Không superposition: hàng đầu của $W^\top W$ là $(1,0,0,\dots)$. Có superposition: $(1,\epsilon,-\epsilon,\epsilon,\dots)$. Các $\epsilon$ (chỉ là sản phẩm phụ của interference) là **đường tấn công** sẵn có vào feature quan trọng nhất, kể cả khi có vô hạn dữ liệu.

Thí nghiệm: tấn công L2 (tối đa 0.1 chuẩn trung bình của input). Gradient descent gặp gradient masking (ReLU ở vùng 0 tới 99% thời gian), nên dùng tấn công giải tích tối ưu cho từng feature $\lambda (W^\top W)_i/\lVert(W^\top W)_i\rVert_2$ rồi chọn cái gây hại nhất.

- Độ dễ bị tấn công **tăng vọt (>3×) khi superposition hình thành**, và bám sát số feature trên mỗi chiều ($1/D$).
- Adversarial training *có* giảm superposition, nhưng phải dùng tấn công rất lớn (80% chuẩn L2 input) mới xóa hết.
- Paper thận trọng: không khẳng định superposition là nguyên nhân chính của adversarial example ngoài thực tế.

---

## 9. Superposition trong privileged basis

Mô hình ở trên **không có privileged basis**: thay $W$ bằng $OW$ với $O$ trực giao thì $(OW)^\top(OW)=W^\top W$, mô hình y hệt. Đây là tương tự cho residual stream. Để mô phỏng MLP, paper thêm ReLU ở hidden layer:

$$
\boxed{h=\operatorname{ReLU}(Wx),\qquad x'=\operatorname{ReLU}(W^\top h+b)}
$$

Giờ $W$ là ánh xạ **feature → neuron** và đọc trực tiếp được. Kết quả ($n=10$, $m=5$, $I_i=0.75^i$; mỗi cấu hình huấn luyện 1000 mô hình, lấy loss thấp nhất):

- Nhiều neuron **dành riêng cho một feature** (monosemantic), đây là lý do phương pháp phân tích theo neuron đôi khi hiệu quả.
- Khi sparsity tăng, chuyển từ monosemantic sang **polysemantic**; hai loại có thể **cùng tồn tại trong một mô hình**. Trông như một phase change ở cấp neuron.
- Polysemantic có cấu trúc: feature tương ứng với *tập* neuron; neuron đi từ 1 feature → vài feature → nhiều hơn.
- **Giới hạn**: mô hình này không *cần* ReLU ẩn nên sẽ né nó (đặt bias ẩn dương để chạy vùng tuyến tính). Vì vậy paper chuyển sang mô hình tính toán ở mục 10.

**Stacked bar plot**: mỗi cột là một neuron, các khối màu là trọng số tới từng feature. Neuron có nhiều khối thì polysemantic.

---

## 10. Tính toán trong superposition

Câu hỏi: mạng có *tính toán* được trên feature đang superposition, chứ không chỉ lưu trữ?

Bài toán $y=\lvert x\rvert$ với $x_i\in[-1,1]$ (khi khác 0), dùng $\lvert x\rvert=\operatorname{ReLU}(x)+\operatorname{ReLU}(-x)$:

$$
\boxed{h=\operatorname{ReLU}(W_1x),\qquad y'=\operatorname{ReLU}(W_2h+b)}
$$

- $n=3$, $m=6$: mô hình học đúng một neuron "dương" $\operatorname{ReLU}(x_i)$ và một neuron "âm" $\operatorname{ReLU}(-x_i)$ cho mỗi feature. Không superposition thì cần 2 neuron/feature.
- $n=100$, $m=40$, $I_i=0.8^i$, quét sparsity: dữ liệu dày thì mỗi neuron một feature; đủ thưa thì mô hình **dùng superposition để tính abs cho nhiều feature hơn số neuron cho phép**. Đây là bằng chứng chính cho "computation in superposition".
- Paper lưu ý: không so loss giữa các mức sparsity được, vì đổi sparsity là đổi bài toán.

Các hiện tượng giống mạng thật:

- Neuron biểu diễn feature quan trọng nhất thường monosemantic; một nhóm neuron khác rất polysemantic.
- Nhiều neuron có một feature **"primary"** (trọng số lớn) kèm các feature **"secondary"** (trọng số nhỏ). Nhìn top activation thì tưởng monosemantic, nhưng ở activation thấp thì polysemantic. Điều này khớp với quan sát trên neuron của LLM.

### 10.1 Motif asymmetric superposition

Một cặp neuron làm việc như sau:

- **Neuron 1** lưu hai feature với độ lớn khác nhau, ví dụ $W=[2,-\tfrac12]$, và trọng số ra nghịch đảo $[\tfrac12,2]$: feature này nhiễu mạnh lên feature kia, nhưng chiều ngược lại thì không.
- **Neuron 2** ức chế mạnh feature bị ảnh hưởng đúng trong trường hợp sẽ có nhiễu dương, tức đổi nhiễu dương (tốn kém) thành nhiễu âm (gần như miễn phí nhờ ReLU đầu ra).

---

## 11. Bức tranh chiến lược

**Vì sao quan trọng cho safety?** Muốn khẳng định mô hình *không* làm điều X, một công cụ mạnh là **liệt kê được mọi feature** (enumerative safety). Không superposition + privileged basis thì liệt kê feature = liệt kê neuron. Có superposition thì không làm vậy được. Nếu liệt kê được feature, ta có thể dùng compressed sensing để "unfold" activation thành activation của một mô hình lớn không superposition. Paper gọi mọi phương pháp như thế là **"giải superposition"**. Đây là điều kiện cho: phân rã activation space, mô tả activation bằng feature thuần, phân tích weight (circuit), và ngay cả phép đo cosine similarity đơn giản (vì feature không liên quan có tích vô hướng dương).

**Ba hướng thoát:**

| Hướng | Ý tưởng | Khó khăn paper nêu |
|---|---|---|
| 1. Mô hình không superposition | L1 lên activation ẩn $\lambda\lVert h\rVert_1$ (giết feature dưới ngưỡng importance); MoE | Mất hiệu năng nếu ràng buộc là số neuron; superposition giúp mô hình "lớn hơn" thực tế |
| 2. Tìm overcomplete basis sau huấn luyện | Sparse coding / dictionary learning trên activation | Không biết số feature; "virtual neuron" tách khỏi cấu trúc bề mặt; bài toán kỹ thuật cực lớn; interference không còn đẩy giúp |
| 3. Lai | Giảm bớt superposition (không mất hiệu năng nhờ "biên" quanh tối ưu) rồi giải mã | |

Ý về MoE: superposition khai thác **khoảng cách giữa độ thưa của neuron và độ thưa của feature**. MoE "ăn" đúng khoảng cách đó, nên dự đoán mô hình MoE có ít superposition hơn. Ràng buộc thật sự là FLOP, không phải số neuron.

**Cân nhắc thêm:**

- **Phase change là lý do để lạc quan**: có một vùng không superposition hoàn toàn; nếu đẩy được mô hình sang đó thì xóa hẳn.
- Một mô hình không superposition (dù kém) cũng là "ground truth" quý để nghiên cứu.
- **Local basis không đủ** cho safety, vì cần giải thích trên toàn phân phối.

**Liên hệ mạng thật** (dự đoán khớp quan sát): polysemantic neuron tồn tại; monosemantic và polysemantic cùng layer; InceptionV1 có nhiều polysemantic hơn ở layer sâu (feature cấp cao thưa hơn); **MLP đầu tiên của Transformer cực kỳ polysemantic** (phân biệt "die" tiếng Anh/Đức/Hà Lan/Afrikaans là các feature rất thưa).

**Liên hệ compressed sensing**: chặn dưới $m=\Omega(k\log(n/k))$ cho khôi phục $k$-sparse vector, áp vào toy model (với $k=O((1-S)n)$) thành $m=\Omega(-n(1-S)\log(1-S))$. Tức số feature **tuyến tính theo $m$**, được điều chỉnh bởi sparsity. Phụ lục chứng minh chặn này dưới giả định RIP.

---

## 12. Paper kiểm chứng như thế nào

| Loại bằng chứng | Nội dung |
|---|---|
| Ground truth tổng hợp | Feature được biết trước (đây là lý do chọn toy model) |
| Trực quan hóa weight | $W^\top W$, $b$, $\lVert W_i\rVert$, mức chia sẻ chiều, stacked bar plot |
| Phân tích giải tích | Khai triển $L_1$; "toy model của toy model" cho $n=2,m=1$ và $n=3,m=2$; giao loss xác nhận phase change bậc nhất |
| Thống kê hình học | $D^*$, $D_i$ tụ ở 3/4, 2/3, 1/2, 2/5, 3/8 |
| Tối ưu nhiều lần | 10 mô hình/điểm (phase diagram), 1000 mô hình (ReLU hidden) để lấy nghiệm tốt nhất |
| Can thiệp | Tấn công adversarial (>3×), adversarial training (80% chuẩn L2) |
| Tái lập độc lập | Redwood, DeepMind (McGrath, kèm lời giải đóng), OpenAI, Hobbhahn, Becker-Kahn; Conjecture dùng sparse autoencoder lấy lại feature ground truth trong toy |

---

## 13. Giới hạn

- **Toy model rất đơn giản**: một ma trận, ReLU, dữ liệu độc lập, phân phối đều. Chính paper nói "rất không rõ nên khái quát gì sang mạng thật".
- Paper đoán superposition, mono/polysemantic và có thể quan hệ với adversarial sẽ khái quát; còn **hình học (polytope) và learning dynamics thì không chắc**, có thể là đặc thù của toy.
- Mô hình ReLU hidden layer "né" ReLU nếu được, nên chỉ cho kết quả trong vùng hẹp.
- Tối ưu khó (nhất là $m=2$), phải chọn nghiệm tốt nhất từ nhiều lần chạy, nên "nghiệm điển hình" có thể khác nghiệm tối ưu.
- Chưa có lý thuyết cho non-uniform superposition; chưa có cách ước lượng đường importance/sparsity của mô hình thật.
- Tính toán trong superposition mới chỉ được chứng minh cho $\lvert x\rvert$; chưa rõ lớp phép tính nào làm được.

---

## 14. Ứng dụng giả thuyết cho pruning và distillation

> **Toàn bộ mục này là suy luận của người viết từ toy model, không phải kết quả của paper.** Cần kiểm chứng trên LightOnOCR trước khi tin.

### 14.1 Vì sao cắt neuron MLP phá feature (và khó khôi phục tuyến tính)

Bạn quan sát: cắt neuron MLP khó phục hồi hơn nhiều so với cắt layer. Toy model gợi ý bốn cơ chế:

1. **Một neuron mang mảnh của nhiều feature.** Trong superposition, feature tương ứng với *tập* neuron, và mỗi neuron mang nhiều feature (primary + secondary). Cắt một neuron không giết gọn một feature; nó làm *méo một phần* của nhiều feature cùng lúc.
2. **Phần bị mất là phần phi tuyến.** Ví dụ abs: $\lvert x\rvert=\operatorname{ReLU}(x)+\operatorname{ReLU}(-x)$. Cắt neuron $\operatorname{ReLU}(-x)$: khi $x<0$, mọi neuron còn lại cho 0, nên **không có ánh xạ tuyến tính nào** từ activation còn lại khôi phục được $\lvert x\rvert$. Đây đúng là phần $D_{res}$. Cái mất không phải một hướng, mà là một "mảnh" của hàm phi tuyến.
3. **Cơ chế lọc nhiễu bị phá.** Superposition chỉ chạy được nhờ bias âm và các cặp kiểu asymmetric superposition + inhibition được tinh chỉnh để triệt nhiễu. Cắt neuron ức chế thì nhiễu dương "được giải phóng" ở nhiều feature khác. Một bản đồ tuyến tính tại chỗ cắt không tái tạo được ngưỡng ReLU.
4. **Dung lượng giảm theo tỷ lệ, và nghiệm tối ưu đổi pha.** Trong uniform superposition, số feature học được tỷ lệ với $m$. Giảm $m$ đẩy feature sang một pha hình học *khác* (ví dụ pentagon → digon, feature ít quan trọng về 0). Nghiệm mới không phải ảnh tuyến tính của nghiệm cũ, nên cần học lại (KD/fine-tune) chứ không chỉ "vá" tuyến tính.

Ngược lại, **cắt layer** (giả thuyết): residual stream là non-privileged và vẫn còn nguyên; thông tin đã ghi vào stream (kể cả đang superposition) không mất, chỉ mất *bản cập nhật* của layer đó. Phần cập nhật ấy có thể một phần dự đoán tuyến tính được từ stream, và các layer sau có thể bù. Cắt neuron thì thu hẹp không gian tính toán phi tuyến của *mọi* layer bị cắt.

### 14.2 Đọc $D_{lin}$ và $D_{res}$ qua lăng kính superposition

Giả thuyết làm việc:

- $D_{lin}$ lớn ↔ phần bị cắt mang **feature dày, tương quan** (vùng "PCA-like"): dự đoán tuyến tính được từ phần còn lại.
- $D_{res}$ lớn ↔ phần bị cắt mang **feature thưa, độc lập, đang superposition**, cần ReLU + bias để đọc: gần như không dự đoán tuyến tính được.
- Như vậy tỷ lệ $D_{res}/(D_{lin}+D_{res})$ của một nhát cắt có thể là *proxy cho lượng superposition bị phá*.

### 14.3 Feature nào mất trước

Toy model: khi thiếu chỗ, **feature ít quan trọng và thưa nhất** bị bỏ trước (về $D_i=0$), feature quan trọng giữ chiều riêng lâu nhất. Với OCR, các feature thưa là ký tự hiếm, dấu tiếng Việt ít gặp, ký hiệu toán, bố cục bảng lạ. Dự đoán: sau pruning, lỗi tăng trước hết ở **các bucket ký tự tần suất thấp**, dù CER trung bình gần như không đổi. Nên đánh giá CER theo tần suất ký tự.

### 14.4 Distillation

- **Hidden-state KD ở residual stream**: không gian non-privileged nên khớp student–teacher qua một phép chiếu tuyến tính học được là hợp lý (hướng quan trọng, trục thì không).
- **KD ở cấp neuron MLP** (khớp từng neuron) là đặt bài toán sai nếu teacher đang superposition: student nhỏ không thể có cùng cách xếp feature.
- Phase change gợi ý **student có thể kẹt ở pha kém** (nhiều cực tiểu, nhảy "energy level"). Nhiều seed hoặc warm-start từ teacher-pruned có thể quan trọng hơn tưởng.
- Student nén chặt hơn thì nhiều feature trên mỗi chiều hơn, nên theo mục 8 có thể **kém bền với nhiễu input** hơn (ảnh scan nhiễu, nén JPEG). Nên đo riêng.

### 14.5 Kiến trúc hybrid

Các nhánh GatedDeltaNet và full-attention đều đọc/ghi residual stream bằng phép chiếu tuyến tính, nên lập luận "non-privileged" áp dụng cho stream. Bên trong MLP (activation theo từng kênh) và các phép theo kênh trong khối linear-attention (gate, normalization; tùy cách triển khai) là privileged, nên là nơi superposition kiểu neuron có ý nghĩa. ViT 12 block: theo quan sát InceptionV1, feature ở layer sâu thưa hơn nên có thể nhiều superposition hơn; cắt neuron MLP ở block sâu có thể đau hơn ở block nông (cần đo).

---

## 15. Thí nghiệm gợi ý

### Thí nghiệm 1: tái hiện "cắt neuron trong superposition" trên toy

1. Huấn luyện mô hình abs ($n=100$, $m=40$, $I_i=0.8^i$) ở vài mức sparsity (dày → thưa).
2. Cắt $k$ neuron (ngẫu nhiên, hoặc theo chuẩn trọng số).
3. Fit bản đồ tuyến tính tối ưu từ activation còn lại về $h$ gốc (hoặc về $y$), tách loss thành $D_{lin}$ và $D_{res}$.
4. **Dự đoán**: ở vùng dày (monosemantic), $D_{res}$ chủ yếu là các feature bị cắt hẳn, dễ đoán; ở vùng thưa (polysemantic), $D_{res}$ lan ra nhiều feature và tăng mạnh.

### Thí nghiệm 2: "độ dự đoán tuyến tính" của từng neuron trên LightOnOCR

Với mỗi layer MLP, trên một tập trang OCR:

$$
R^2_k=1-\frac{\mathbb E\lVert a_k-\hat a_k\rVert^2}{\operatorname{Var}(a_k)},\qquad \hat a_k=\text{hồi quy tuyến tính của } a_k \text{ theo } \{a_j\}_{j\ne k}.
$$

Neuron có $R^2_k$ thấp là "không thay thế được" (dự đoán đóng góp lớn vào $D_{res}$). So sánh tiêu chí pruning: magnitude vs. $R^2_k$ vs. kết hợp, đo CER và CER theo bucket tần suất ký tự.

### Thí nghiệm 3: "feature dimensionality" của hướng ghi neuron

Lấy các cột $w_k$ của down-projection (hướng mỗi neuron ghi vào residual). Tính $D_k=\lVert w_k\rVert^2/\sum_j(\hat w_k\cdot w_j)^2$. Neuron có $D_k$ gần 1 ghi vào hướng "riêng", nên cắt nó mất hướng đó; neuron có $D_k$ nhỏ chia hướng với nhiều neuron khác. Kiểm tra $D_k$ có tương quan với thiệt hại khi cắt không. Lưu ý: chỉ số này bỏ qua phân phối activation, nên chỉ là ứng viên.

---

## 16. Tóm tắt công thức và khái niệm

| Khái niệm | Công thức hoặc định nghĩa | Ý nghĩa |
|---|---|---|
| Linear representation | $h=\sum_i x_{f_i}W_{f_i}$ | Feature là hướng |
| ReLU output model | $x'=\operatorname{ReLU}(W^\top Wx+b)$ | Mô hình tối thiểu có superposition |
| Loss | $\sum_x\sum_i I_i(x_i-x'_i)^2$ | MSE có trọng số importance |
| Superposition | $W^\top W$ không khả nghịch | Nhiều feature hơn chiều |
| Loss tuyến tính | $\sum_i I_i(1-\lVert W_i\rVert^2)^2+\sum_{i\ne j}I_j(W_j\cdot W_i)^2$ | Feature benefit vs interference |
| Loss 1-sparse | $\sum_i I_i(1-\operatorname{ReLU}(\lVert W_i\rVert^2+b_i))^2+\sum_{i\ne j}I_j\operatorname{ReLU}(W_j\cdot W_i+b_j)^2$ | Nhiễu âm miễn phí |
| Dimensions per feature | $D^*=m/\lVert W\rVert_F^2$ | "Dính" ở 1 và 1/2 |
| Feature dimensionality | $D_i=\lVert W_i\rVert^2/\sum_j(\hat W_i\cdot W_j)^2$ | Phần chiều mỗi feature có |
| Polytope | 3/4, 2/3, 1/2, 2/5, 3/8 | Tứ diện, tam giác, digon, ngũ giác, square antiprism |
| Tegum product | Ghép polytope trong không gian con trực giao | Không nhiễu giữa các factor |
| ReLU hidden model | $h=\operatorname{ReLU}(Wx)$ | Có privileged basis, mô phỏng MLP |
| Abs qua ReLU | $\lvert x\rvert=\operatorname{ReLU}(x)+\operatorname{ReLU}(-x)$ | Tính toán trong superposition |
| Asymmetric superposition | $W=[2,-\tfrac12]$, ra $[\tfrac12,2]$ + neuron ức chế | Đổi nhiễu dương thành âm |
| Chặn compressed sensing | $m=\Omega(-n(1-S)\log(1-S))$ | Số feature tuyến tính theo $m$, điều chỉnh bởi sparsity |
| Giải superposition | Liệt kê feature ↔ unfold activation | Điều kiện cho enumerative safety |

### Kết luận ngắn

Paper chứng minh trực tiếp, trong mô hình có ground truth, rằng một ReLU đủ để mạng biểu diễn **nhiều feature hơn số chiều** khi feature thưa, và có thể **tính toán** trên chúng. Superposition bị chi phối bởi phase change giữa không học / superposition / chiều riêng, có hình học polytope phong phú, và tạo ra polysemantic neuron ngay cả trong privileged basis. Với pruning, bài học (suy luận) là: cắt neuron MLP trong một mạng đang superposition không xóa gọn feature mà làm méo nhiều feature và phá cơ chế lọc nhiễu phi tuyến. Đó là ứng viên tự nhiên để giải thích vì sao phần $D_{res}$ lớn và vì sao cắt neuron khó phục hồi hơn cắt layer.

**Đọc bản gốc:** https://transformer-circuits.pub/2022/toy_model/index.html
