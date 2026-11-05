# A Mathematical Framework for Transformer Circuits (2021)

**Tác giả:** Nelson Elhage, Neel Nanda, Catherine Olsson (core research contributors), Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, Chris Olah (Anthropic). Đăng ngày 22/12/2021.  
**Nguồn:** https://transformer-circuits.pub/2021/framework/index.html  
**Chủ đề:** Reverse engineering transformer attention-only 0/1/2 layer; residual stream, virtual weights, QK/OV circuit, path expansion, skip-trigram, phân tích eigenvalue cho copying, Q/K/V-composition, induction head, term importance, virtual attention head, MLP.  
**Vị trí trong lộ trình:** Đọc sau `01_2022_mech_interp_essay.md` (từ vựng feature/basis). Đây là "sách giáo khoa toán" của cả chuỗi Transformer Circuits; mọi bài sau (induction heads, superposition, attribution graphs) đều dùng ký hiệu và cách nghĩ của nó. Đọc tiếp `03` (bài tập) rồi `04` (induction heads ở mô hình lớn).

> Đây là ghi chú học tập diễn giải, không phải bản dịch. Bản này mở rộng ghi chú ngắn đã có: thêm zero-layer, phân tích skip-trigram chi tiết, eigenvalue/copying, khai triển QK hai layer, term importance, virtual head, MLP và đính chính của tác giả. Một số phương trình trong bài gốc là hình ảnh; ở đây chúng được viết lại từ mô tả trong văn bản và ký hiệu của paper. Ứng dụng cho pruning/distillation là suy luận của chúng ta.

## Mục lục

1. [Câu hỏi, mô hình đồ chơi và các giản lược](#1-câu-hỏi-mô-hình-đồ-chơi-và-các-giản-lược)
2. [Residual stream là kênh truyền thông](#2-residual-stream-là-kênh-truyền-thông)
3. [Attention head: độc lập, cộng dồn, di chuyển thông tin](#3-attention-head-độc-lập-cộng-dồn-di-chuyển-thông-tin)
4. [Zero-layer transformer: bigram](#4-zero-layer-transformer-bigram)
5. [One-layer: bigram cộng skip-trigram](#5-one-layer-bigram-cộng-skip-trigram)
6. [Tóm tắt ma trận OV/QK: copying và eigenvalue](#6-tóm-tắt-ma-trận-ovqk-copying-và-eigenvalue)
7. [Two-layer: ba kiểu composition](#7-two-layer-ba-kiểu-composition)
8. [Induction head trong mô hình hai layer](#8-induction-head-trong-mô-hình-hai-layer)
9. [Term importance analysis](#9-term-importance-analysis)
10. [Virtual attention head](#10-virtual-attention-head)
11. [Paper nói gì về MLP](#11-paper-nói-gì-về-mlp)
12. [Các quan sát bên lề](#12-các-quan-sát-bên-lề)
13. [Paper kiểm chứng các tuyên bố như thế nào](#13-paper-kiểm-chứng-các-tuyên-bố-như-thế-nào)
14. [Giới hạn](#14-giới-hạn)
15. [Ứng dụng giả thuyết cho pruning / distillation](#15-ứng-dụng-giả-thuyết-cho-pruning--distillation)
16. [Thí nghiệm gợi ý](#16-thí-nghiệm-gợi-ý)
17. [Tóm tắt công thức / khái niệm](#17-tóm-tắt-công-thức--khái-niệm)

---

## 1. Câu hỏi, mô hình đồ chơi và các giản lược

**Câu hỏi:** Có thể reverse engineer transformer ngôn ngữ như Circuits thread đã làm với InceptionV1 không? Chiến lược: bắt đầu từ mô hình *đơn giản nhất có thể* để tìm motif/thuật toán, rồi mới mở rộng.

**Mô hình:** decoder-only, **attention-only** (không MLP), 0, 1 và 2 layer. Thường $n_\text{heads}=12$, $d_\text{head}=64$, $d_\text{model}=n_\text{heads}\cdot d_\text{head}$; một ví dụ riêng có 32 head với $d_\text{head}=128$. Context 2048 token, dense attention. Positional mechanism giống Press et al.: thêm sinusoid ngay trước $W_Q, W_K$, nên **không có thông tin vị trí trong residual stream**.

**Giản lược:**

| Bỏ qua | Lý do paper đưa ra |
|---|---|
| MLP | Thay đổi lớn nhất. Một phần vì attention có thách thức mới, một phần vì "chúng tôi đơn giản là ít thành công hơn với MLP" — tác giả gọi đây là điểm yếu lớn |
| Bias | Gộp vào trọng số bằng một chiều hằng 1; trong mô hình attention-only, bias chủ yếu thành bias trên logit |
| LayerNorm | Trừ một hệ số co giãn biến thiên, LN gộp được vào trọng số lân cận (chi tiết §12) |

**Tóm tắt kết quả (theo paper):**

| Mô hình | Thực chất là gì | Đọc được từ trọng số? |
|---|---|---|
| 0 layer | Bảng bigram | Có |
| 1 layer | Ensemble bigram + nhiều skip-trigram "A… B C" | Có, không cần chạy mô hình |
| 2 layer | Thuật toán dùng composition, nổi bật là **induction head** | Composition phát hiện được từ trọng số |

## 2. Residual stream là kênh truyền thông

**Residual stream** (dòng residual): tổng của embedding và output của mọi layer trước. Paper thích tên này hơn "embedding" vì stream thường dành *subspace* cho token *khác* token hiện tại.

$$
x^{(L)} = x^{(0)} + \sum_{\ell<L}\Delta x^{(\ell)},\qquad \Delta x^{(\ell)} = W_O^{(\ell)}\,f^{(\ell)}\!\big(W_I^{(\ell)}x^{(\ell)}\big).
$$

Mỗi layer *đọc* bằng một phép chiếu tuyến tính và *ghi* bằng cách cộng một phép chiếu tuyến tính. Hệ quả:

1. **Không có privileged basis.** Xoay residual stream (và xoay ngược mọi ma trận chạm vào nó) không đổi hành vi. Paper lưu ý: residual hoàn toàn tuyến tính là hiếm; ngay cả ResNet cũng có phi tuyến trên nhánh residual.
2. **Virtual weights** (trọng số ảo): nối trực tiếp hai layer bất kỳ, kể cả cách xa nhau,
$$\boxed{W^{\,1\to 2}_\text{virtual} = W_I^{2}\,W_O^{1}}.$$
3. **Subspace và bandwidth.** Head chỉ đọc/ghi subspace nhỏ ($d_\text{head}$ = 64 hoặc 128) nên có thể ghi vào subspace rời nhau và không tương tác. Thông tin đã ghi *tồn tại cho tới khi layer khác chủ động xoá*. PCA cho thấy embedding/unembedding chỉ chiếm một phần nhỏ số chiều, để trống phần lớn cho layer khác.
4. **Bottleneck activation.** Có nhiều "chiều tính toán" hơn nhiều so với chiều residual: một MLP có $4\times$ neuron so với $d_\text{model}$; ở layer 25 của mô hình 50 layer, residual có trước nó gấp 100 lần neuron so với số chiều của nó, và phải giao tiếp với gấp 100 lần neuron phía sau — "bằng cách nào đó, trong superposition". Đây là lý do paper phân tích qua virtual weights thay vì nhìn thẳng vào residual.
5. **Memory management.** Có dấu hiệu một số neuron có cosine rất âm giữa trọng số vào/ra, và một số head có eigenvalue âm lớn trong $W_OW_V$ và chủ yếu nhìn token hiện tại → có thể là cơ chế **xoá** thông tin (có thể có điều kiện).

## 3. Attention head: độc lập, cộng dồn, di chuyển thông tin

**Độc lập và cộng dồn.** Cách viết "concat rồi nhân $W_O^H$" tương đương tổng các head:

$$
W_O^H\begin{bmatrix}r^{h_1}\\ r^{h_2}\\ \vdots\end{bmatrix}=\sum_i W_O^{h_i}r^{h_i}.
$$

Concat hiệu quả tính toán hơn; cộng dồn dễ suy luận hơn.

**Di chuyển thông tin.** Head đọc từ residual của token này, ghi vào residual của token khác. Ba bước $v_i=W_Vx_i$, $r_i=\sum_jA_{ij}v_j$, $h(x)_i=W_Or_i$ nhân vào *hai phía khác nhau* của ma trận $x\in\mathbb R^{n_\text{ctx}\times d_\text{model}}$. Dùng tensor product:

$$
h(x) = (\mathrm{Id}\otimes W_O)\,(A\otimes\mathrm{Id})\,(\mathrm{Id}\otimes W_V)\,x
\;\;\Longrightarrow\;\;
\boxed{h(x) = (A\otimes W_OW_V)\,x}
$$

nhờ mixed-product $(A\otimes B)(C\otimes D)=(AC)\otimes(BD)$. Ký hiệu thực dụng: $\mathrm{Id}\otimes W$ = nhân từng vị trí với $W$; $A\otimes\mathrm{Id}$ = trộn giữa các vị trí; $(A\otimes W)x = A\,x\,W^\top$.

Và $A=\mathrm{softmax}^*(x^\top W_Q^\top W_K x)$, với $\mathrm{softmax}^*$ là softmax có causal mask.

**Các quan sát paper rút ra:**

- Head áp *hai* phép tuyến tính độc lập: $A$ (token nào → token nào) và $W_{OV}=W_OW_V$ (đọc gì từ nguồn, ghi thế nào vào đích).
- $A$ là phần phi tuyến duy nhất. **Cố định $A$ thì head tuyến tính**; không cố định thì head "nửa tuyến tính".
- $W_Q, W_K$ luôn đi cùng nhau; $W_O, W_V$ cũng vậy. Định nghĩa $\boxed{W_{QK}=W_Q^\top W_K,\quad W_{OV}=W_OW_V}$, cả hai đều **low-rank** (hạng $\le d_\text{head}$).
- Query/key/value là "sản phẩm phụ" — có thể tái tham số hoá mà hàm không đổi.
- SVD $W_{OV}=USV$: right singular vectors = subspace được *đọc* từ nguồn; left singular vectors = subspace được *ghi* ở đích.
- Tích hai head vẫn là một "head": $(A^{h_2}\otimes W^{h_2}_{OV})(A^{h_1}\otimes W^{h_1}_{OV})=(A^{h_2}A^{h_1})\otimes(W^{h_2}_{OV}W^{h_1}_{OV})$ → **virtual attention head** (§10).

## 4. Zero-layer transformer: bigram

$$
\boxed{T = W_U W_E}
$$

Không di chuyển thông tin giữa vị trí, nên chỉ dự đoán token kế tiếp từ token hiện tại → tối ưu là **xấp xỉ log-likelihood bigram** (song song với nhận xét của Levy & Goldberg 2014 rằng word embedding phân rã ma trận log-likelihood).

Quan trọng cho mô hình sâu hơn: số hạng $W_UW_E$ (**direct path**) xuất hiện trong khai triển của *mọi* transformer. Ở mô hình lớn nó không còn đúng là bigram mà là một kiểu "phần dư" — thường biểu diễn bigram không theo luật ngữ pháp chung, ví dụ "Barack" → "Obama". Hệ quả: **$W_U$ không nên được kỳ vọng là nghịch đảo của $W_E$**.

## 5. One-layer: bigram cộng skip-trigram

### 5.1 Path expansion

Mô hình là tích ba số hạng; **khai triển tích thành tổng các đường đi end-to-end**:

$$
T=(\mathrm{Id}\otimes W_U)\Big(\mathrm{Id}+\sum_{h\in H_1}A^h\otimes W^h_{OV}\Big)(\mathrm{Id}\otimes W_E)
$$

$$
\boxed{T=\underbrace{\mathrm{Id}\otimes W_UW_E}_{\text{direct path: bigram}}+\sum_{h\in H_1}A^h\otimes\big(W_UW^h_{OV}W_E\big)},
\qquad A^h=\mathrm{softmax}^*\!\big(t^\top W_E^\top W^h_{QK}W_E\,t\big).
$$

### 5.2 Hai ma trận $[n_\text{vocab}\times n_\text{vocab}]$

| Ma trận | Tên | Ý nghĩa |
|---|---|---|
| $W_E^\top W^h_{QK}W_E$ | **QK circuit** | Điểm attention giữa mọi cặp (query token, key token): token đích "muốn" nhìn token nguồn bao nhiêu |
| $W_UW^h_{OV}W_E$ | **OV circuit** | Nếu token nguồn được attend, nó đổi logit output thế nào |

Sau khi đích quyết định attend nguồn bao nhiêu, **ảnh hưởng lên output chỉ phụ thuộc token nguồn**.

**Freezing attention patterns:** thí nghiệm tư duy — chạy lần 1 để thu $A$ (chỉ phụ thuộc QK), chạy lần 2 với $A$ bị "đóng băng" → logit là **hàm tuyến tính của token**.

### 5.3 Skip-trigram

QK chọn token *nguồn* mà token *đích* nhìn lại; OV cho biết ảnh hưởng lên token *out*. Ba token tạo **skip-trigram** `[source] … [destination] → [out]`.

Lưu ý quy mô: vocab ~50.000 → một ma trận OV khai triển có ~2,5 tỉ phần tử. Paper gọi mô hình một layer là "Chinese room nén" — ta còn lại "một đống thẻ khổng lồ". Thêm khó khăn: biến tương quan và có thể thay thế nhau (một head có trọng số 0 vì head khác đã làm việc đó), và trọng số QK không so sánh được giữa các query.

**Ví dụ skip-trigram paper liệt kê** (đọc từ phần tử lớn nhất của QK/OV):

| Miền | Skip-trigram |
|---|---|
| Python | indent N lần … indent N−1 → `else/elif/except`; `open … ","` → `rb/wb/r/w`; `def … (` → `self`; `super … self` → `).__` |
| Python thư viện | `tf … .` → `dtype/shape/initializer`; `pygame … .` → `display/rect/tick` |
| HTML / LaTeX | `tbody … <` → `td`; `left … \` → `right` |
| Ngoặc / dấu | `(' … X` → `')` (đóng ngoặc khớp) |
| Tiếng Anh | `keep … in` → `mind`; `difficult … not` → `impossible`; với query " and": `back and → forth`, `day and → night` |
| URL | `github … /` → `issues/blob/pull/master`; `http … :` → `8080/8000` |

Skip-trigram thường gắn với đặc thù tokenizer (gộp whitespace để lộ indent; backslash tách riêng trong LaTeX). Mẹo của paper: gõ thử skip-trigram vào Google để xem autocompletion (vd. `Israel … K → nes` = "Knesset").

### 5.4 Copying = in-context learning sơ khai

Phần lớn head trong mô hình một layer dành **rất nhiều capacity cho copying**: OV làm token được attend tăng xác suất *chính nó* (và token tương tự); QK chỉ attend về token có thể là token kế tiếp hợp lý. Paper tóm các mẫu:

```text
[b] … [a] → [b]        # copying head: token đã xuất hiện có thể xuất hiện lại
[ab] … [a] → [b]       # khi tokenizer tách từ: " Ralph" … "R" → "alph"
```

Mẫu thứ hai (từ hiếm không có dấu cách bị tách "R"+"alph") là trường hợp hiếm mà mô hình một layer *gần* bắt chước được induction head.

### 5.5 Head chủ yếu theo vị trí

Một số ít head chủ yếu theo vị trí tương đối (vd. chỉ nhìn token hiện tại hoặc token trước). Với sinusoid, dịch vị trí ≡ nhân ma trận quay, nên $W_{QK}$ chọn được offset tương đối bất kỳ bằng cách quay các chiều sinusoid (bài tập 2(c) trong ghi chú 03 làm điều này bằng tay).

### 5.6 Skip-trigram "bugs"

Skip-trigram được lưu ở dạng **phân tích nhân tử** giữa QK và OV, giống $f(a,b,c)=f_1(a,b)\,f_2(a,c)$: không biểu diễn linh hoạt được tương tác ba chiều. Nếu một head tăng cả `keep… in mind` và `keep… at bay`, nó buộc phải tăng cả `keep… in bay` và `keep… at mind`. Paper coi đây là ví dụ sớm về việc **dùng interpretability để hiểu lỗi của mô hình**, và để ngỏ câu hỏi lỗi này tốn bao nhiêu loss.

### 5.7 "Đã hiểu hoàn toàn" mô hình một layer chưa?

Theo nghĩa giống hiểu một hồi quy tuyến tính khổng lồ hay một database: không còn bí ẩn thuật toán, tham số đã được "ngữ cảnh hoá", nhưng quá nhiều để giữ trong đầu. Paper nhận xét: mạng một layer thường là generalized linear model, nên một attention layer *phần lớn* cũng vậy là không lạ.

## 6. Tóm tắt ma trận OV/QK: copying và eigenvalue

Lý do kỳ vọng có cách tóm tắt tốt hơn "tìm phần tử lớn": (i) ma trận $50{.}000\times50{.}000$ nhưng hạng chỉ 64/128; (ii) có cấu trúc cụm (một head: tên người có top query " by", địa danh có top query " from"); (iii) copying rất phổ biến.

**Eigenvalue như thống kê copying.** Nếu $Mv=\lambda v$ với $M=W_UW^h_{OV}W_E$ và $\lambda>0$ thực, thì có một tổ hợp tuyến tính các token làm *tăng* logit của chính tổ hợp đó → "một tập token cùng tăng xác suất của nhau". Vector riêng có dấu âm/dương → hai nhóm (vd. số ít/số nhiều) tăng nhóm mình, giảm nhóm kia.

- Ma trận ngẫu nhiên (gần Ginibre) có eigenvalue đối xứng dương–âm, nhiều giá trị phức; OV lúc khởi tạo cũng vậy (thực nghiệm).
- Trong mô hình một layer 12 head: **10/12 head copying đáng kể** — khớp với kiểm tra thủ công.

Thống kê tóm tắt mà bài tiếp theo (induction heads, 2022) dùng để kiểm tra:

$$
\boxed{\;\text{copy}(M)=\frac{\sum_i\lambda_i}{\sum_i|\lambda_i|}\in[-1,1]\;}
$$

**Cảnh báo của paper:** eigenvector không trực giao → có ma trận *mọi* eigenvalue dương mà vẫn làm một số token *giảm* logit của chính nó. Eigenvalue dương là "copying trung bình", bằng chứng mạnh nhưng không phải chứng minh. Các định nghĩa thay thế (đường chéo dương; token tự tăng mình nhiều nhất/top-k) đều nối với nhau vì $\sum\lambda_i=\mathrm{tr}(M)=\sum_i M_{ii}$.

**Tính hiệu quả:** $\lambda_i(AB)=\lambda_i(BA)$ → tính eigenvalue trên ma trận $64\times64$ thay vì $50{.}000\times50{.}000$.

## 7. Two-layer: ba kiểu composition

Không có composition, mô hình hai layer chỉ là mô hình một layer với nhiều head hơn. Có composition, nó giống **chương trình chạy thuật toán** hơn là bảng tra skip-trigram.

| Kiểu | Head sau đọc output head trước bằng | Ảnh hưởng |
|---|---|---|
| Q-composition | $W_Q$ | Attention pattern (tìm gì) |
| K-composition | $W_K$ | Attention pattern (được tìm theo gì) |
| V-composition | $W_V$ | Nội dung được chuyển → tạo virtual head |

Q/K-composition **khác về bản chất** V-composition: "di chuyển thông tin ∘ di chuyển thông tin = di chuyển thông tin", còn head ảnh hưởng attention pattern của head khác thì không rút gọn như vậy.

### 7.1 Khai triển logit (OV side)

$$
T=\mathrm{Id}\otimes W_UW_E+\sum_{h\in H_1\cup H_2}A^h\otimes\big(W_UW^h_{OV}W_E\big)+\sum_{h_2\in H_2}\sum_{h_1\in H_1}\big(A^{h_2}A^{h_1}\big)\otimes\big(W_UW^{h_2}_{OV}W^{h_1}_{OV}W_E\big).
$$

Hai số hạng đầu giống mô hình một layer; số hạng cuối là virtual head (V-composition) — trong thực tế **không đáng kể** ở mô hình nhỏ (§9).

### 7.2 Khai triển QK của head layer 2

Đây là thay đổi "triệt để" nhất. Với $h\in H_2$, điểm attention là $x_1^\top W^h_{QK}x_1$, trong đó $x_1=(\mathrm{Id}+\sum_{h'\in H_1}A^{h'}\otimes W^{h'}_{OV})(\mathrm{Id}\otimes W_E)\,t$ xuất hiện ở *cả phía query lẫn phía key*. Kết quả là một (4,2)-tensor với các số hạng dạng $A_q\otimes A_k\otimes W$:

| Số hạng | Loại |
|---|---|
| $\mathrm{Id}\otimes\mathrm{Id}\otimes\big(W_E^\top W^h_{QK}W_E\big)$ | Như một layer |
| $\mathrm{Id}\otimes A^{h_k}\otimes\big(W_E^\top W^h_{QK}W^{h_k}_{OV}W_E\big)$ | **K-composition** |
| $A^{h_q}\otimes\mathrm{Id}\otimes\big(W_E^\top W^{h_q\top}_{OV}W^h_{QK}W_E\big)$ | Q-composition |
| $A^{h_q}\otimes A^{h_k}\otimes\big(W_E^\top W^{h_q\top}_{OV}W^h_{QK}W^{h_k}_{OV}W_E\big)$ | Q và K cùng lúc |

$A_q$ mô tả thông tin phía query đã di chuyển giữa token thế nào, $A_k$ tương tự cho phía key, $W$ mô tả chúng nhân với nhau ra điểm.

### 7.3 Đo composition từ trọng số

$$
\boxed{\;C_K(h_1\!\to\!h_2)=\frac{\|W^{h_2}_{QK}W^{h_1}_{OV}\|_F}{\|W^{h_2}_{QK}\|_F\,\|W^{h_1}_{OV}\|_F}\;}
$$

Q-composition dùng $W^{h_2\top}_{QK}$, V-composition dùng $W^{h_2}_{OV}$ thay cho $W^{h_2}_{QK}$. Mặc định paper **trừ đi giá trị kỳ vọng với ma trận ngẫu nhiên cùng shape** (phần lớn head compose *ít hơn* ngẫu nhiên). Khác với bản ghi chú ngắn: paper đo trên $W_{QK}$, $W_{OV}$ đầy đủ chứ không chỉ $W_K W_O$.

**Kết quả trên mô hình được phân tích:** chỉ có K-composition đáng kể, và chỉ với **một** head layer 0 — head này chủ yếu nhìn token trước (ít hơn: token hiện tại và token cách hai). Phần lớn head không tham gia composition, về cơ bản vẫn là skip-trigram. Paper suy đoán: vài induction head "thắng" vài skip-trigram head trong cạnh tranh capacity, nhưng không kiểu composition nào khác thắng. (Xem đính chính ở §13.)

**Value-weighted attention:** nhân attention với $\|v^h_\text{src}\|$ để thấy "vector lớn cỡ nào được chuyển từ mỗi vị trí". Hữu ích vì head hay attend về một vị trí "nghỉ" (token đầu, dấu câu) có value nhỏ khi không tìm thấy gì.

## 8. Induction head trong mô hình hai layer

### 8.1 Chức năng

Tìm trong context lần xuất hiện trước của token hiện tại; nếu không có thì attend token đầu (token đặc biệt) và không làm gì; nếu có thì nhìn **token kế tiếp** và copy nó.

| Thuật toán | Mẫu |
|---|---|
| Copying head (1 layer) | `[b] … [a] → [b]` |
| Tokenizer quirk (1 layer) | `[ab] … [a] → [b]` |
| **Induction head (2 layer)** | `[a][b] … [a] → [b]` |

Mạnh hơn vì biết token *đã được dùng thế nào* trước đó → dự đoán tự tin hơn, và **ít phụ thuộc phân phối** (không dựa vào thống kê "token này hay đi sau token kia"). Bằng chứng ấn tượng: hoạt động trên **chuỗi token ngẫu nhiên lặp lại** (hoàn toàn off-distribution). Ví dụ minh hoạ trên đoạn đầu Harry Potter: head 1:8 dự đoán phần còn lại của "Dursley", "Potters". Paper đo "logit effect" $(W_UW^h_Or^h_\text{pres})_\text{next}$.

### 8.2 Cơ chế

Mẹo trung tâm: **key được tính từ token dịch lùi một vị trí**. Cách tối thiểu là K-composition với previous-token head, tạo số hạng

$$
\boxed{\;\mathrm{Id}\otimes A^{h_{-1}}\otimes W,\qquad W=W_E^\top W^h_{QK}W^{h_{-1}}_{OV}W_E\;}
$$

Nếu $W$ là "copying matrix phía QK" (khớp khi hai token *giống nhau*), điểm attention tăng khi **token trước vị trí nguồn = token đích**. Biến thể: mô hình khác có head nhìn xa hơn token trước để tạo $A^{h_{-1}}\otimes A^{h_{-2}}\otimes W$, khớp prefix dài hơn. Với positional embedding trong residual (không phải mô hình của paper), có thuật toán thứ hai: **pointer arithmetic** (§12).

### 8.3 Kiểm tra lý thuyết cơ chế

Lý thuyết dự đoán induction head phải có (1) OV là copying matrix, (2) số hạng $\mathrm{Id}\otimes A^{h_{-1}}\otimes W$ với $W$ "same-matching". Đặt mỗi head lên mặt phẳng 2D (độ dương eigenvalue QK, độ dương eigenvalue OV): **mọi induction head nằm ở góc cực phải**.

Phản biện vòng tròn? Head được chọn vì K-composition lớn, rồi lại xét số hạng K-composition. Paper trả lời: *độ lớn* composition không ngụ ý composition *dương*; cũng không có lý do để mọi OV dương — trừ khi thuật toán đúng là induction.

## 9. Term importance analysis

**Vấn đề:** virtual head bị bỏ qua vì V-composition nhỏ, nhưng có thể chúng quan trọng *tổng thể*. Ablate một số hạng chỉ tồn tại sau khai triển thì không thể "nhân 0" trực tiếp.

**Thuật toán đo đóng góp biên của số hạng bậc $n$** (bậc = số head trên đường V-composition):

```text
Bước 1: Chạy mô hình, lưu mọi attention pattern.
Bước 2: Chạy lại với pattern bị ép bằng bản đã lưu; thay vì cộng output head vào
        residual, LƯU output rồi thay bằng tensor 0. Ghi loss.     -> chỉ còn bậc 0
Bước n: Chạy lại với pattern bị ép; lưu output, thay bằng giá trị đã lưu của head
        đó ở lần chạy trước. Ghi loss.                             -> giữ tới bậc n-2…
Hiệu loss giữa các lần chạy = đóng góp biên của từng bậc.
```

Đóng băng pattern là điều khiến ablation **chỉ nhắm vào V-composition**; biến thể khác có thể cô lập Q/K-composition.

**Kết quả:** số hạng bậc 2 (virtual head) có đóng góp biên nhỏ ở mô hình này (có thể lớn hơn ở mô hình lớn). Kết luận: tập trung vào direct path và số hạng từng head; chia tiếp theo layer thì **số hạng của head layer 2 chiếm ưu thế**. Paper nhấn mạnh: điều này **không nói gì về Q/K-composition** — chúng nằm trong QK circuit, không nằm trong khai triển logit.

> Bài học cho người làm pruning: head layer 1 (previous-token head) có đóng góp trực tiếp/OV nhỏ, nhưng là đầu vào bắt buộc cho key của induction head. Một thước đo importance chỉ nhìn đường logit sẽ đánh giá thấp nó.

## 10. Virtual attention head

Số hạng $(A^{h_2}A^{h_1})\otimes(W^{h_2}_{OV}W^{h_1}_{OV})$ là một head thực thụ $h_2\circ h_1$ với pattern $A^{h_2}A^{h_1}$ và $W_{OV}=W^{h_2}_{OV}W^{h_1}_{OV}$; mô hình sâu có thể có $h_3\circ h_2\circ h_1$.

- **Mạnh:** paper thường thấy previous-token head nhưng không thấy head "nhìn lùi hai token" — có lẽ vì virtual head đã làm việc đó. Composition cũng cho phép "attend chủ ngữ của mệnh đề trước".
- **Nhiều:** head thường tăng tuyến tính theo số layer; virtual head bậc 2 tăng bậc hai, bậc 3 tăng bậc ba. Head thường "to" (một pattern + $d_\text{head}$ chiều), lãng phí cho việc nhỏ như xác định ngôi (thứ nhất/hai/ba) hay thì (quá khứ/hiện tại/tương lai).
- Ở mô hình hai layer nhỏ chúng không quan trọng; paper *suy đoán* chúng quan trọng hơn ở mô hình lớn.

## 11. Paper nói gì về MLP

**Thừa nhận chính:** MLP chiếm **2/3 tham số** của transformer chuẩn; vì nhiều head tương tác với MLP, phần tham số hiểu được nếu bỏ qua MLP còn **ít hơn 1/3**. Hướng rõ nhất cần neuron diễn giải được riêng lẻ — điều họ ít thành công.

**Vì sao khó:** GeLU khiến không thể tuyến tính hoá xuyên MLP như với attention; phải quay về cách của Circuits (neuron biểu diễn gì, được tính thế nào, được dùng thế nào). Lý thuyết lạc quan: hàm kích hoạt khuyến khích privileged basis, MLP rộng gấp 4 residual và thông tin không bắt buộc đi qua nó. Thực tế: neuron khó đặt giả thuyết, **trừ neuron ở ~5% độ sâu** thường phản ứng với cụm cụm từ ngắn đồng nghĩa.

**Khung cơ chế vẫn dùng được** (mô hình một layer có MLP):

$$
a^m_\text{pre}=W^m_I\Big(\mathrm{Id}+\sum_{h\in H_1}A^h\otimes W^h_{OV}\Big)W_E
=W^m_IW_E+\sum_{h\in H_1}A^h\otimes\big(W^m_IW^h_{OV}W_E\big).
$$

- $W^m_IW_E$: token nào kích thích/ức chế neuron qua direct path; $W^m_IW^h_{OV}W_E$: tương tự khi token được head $h$ attend. Giống neuron conv net nhưng **trọng số được đánh chỉ số theo head thay vì vị trí tương đối**.
- Phía sau: $\boxed{W_UW^m_O}$ cho ảnh hưởng của mỗi neuron lên logit — tuyến tính, luôn đúng cho **MLP cuối cùng**.
- Sâu hơn thì phải suy luận MLP→MLP; về nguyên tắc khả thi, nhưng neuron khó hiểu và quá nhiều.

**Cấu trúc giống convolution:** path expansion cho dạng $y=(\mathrm{Id}\otimes W_\mathrm{Id}+\sum_hA^h\otimes W_h)x$. Mọi convolution viết được là $W\ast x=\sum_vA^v\otimes W_v$ với $A^v$ là head nhìn offset cố định $v$ → attention tổng quát hoá convolution "với vị trí động, mềm, có thể mang ngữ nghĩa".

**Tính chất activation** (bảng ký hiệu của paper): chỉ token, attention pattern và activation MLP có privileged basis; residual, query, key, value, output head/MLP thì không. Residual và value là **bottleneck activation**.

**Cập nhật tháng 2/2023 của tác giả:** điểm yếu lớn nhất là MLP; họ suy đoán nguyên nhân là **superposition**. Sau đó: Black et al. cho bằng chứng neuron thường polysemantic; *Toy Models of Superposition* phát triển giả thuyết; có hướng thay thế (Polytope Lens, SVD trọng số).

## 12. Các quan sát bên lề

- **LayerNorm:** trừ trung bình là phép tuyến tính cố định; chia độ lệch chuẩn là nhân vô hướng, giao hoán với các phép khác trên một đường → coi như *trọng số lại* các số hạng đường đi. Hệ quả: khó so sánh đường qua các tập layer khác nhau. LN layer đầu có thể áp thẳng lên ma trận embedding. BatchNorm thì gộp hoàn toàn được.
- **Pointer arithmetic:** khi positional embedding nằm trong residual, một head attend token giống mình và copy *vị trí* của nó; head sau dùng Q-composition **quay** vị trí đó tới một token → induction head. Paper thấy ít nhất một trường hợp như vậy ở GPT-2.
- **Identity head:** có thể coi residual là head với $A=\mathrm{Id}$, $W_{OV}=\mathrm{Id}$; paper giữ riêng vì nó có tính chất khác.
- **Kiến trúc mới:** Primer (depthwise conv trên 3 vị trí khi tính Q/K/V) cho phép induction head **không cần K-composition**; Talking Heads tương đương chia sẻ thành phần giữa $W_{OV}$ (và $W_{QK}$) của các head.
- **Attention as explanation:** pattern của head đơn lẻ (bậc 1) diễn giải được; Q/K/V-composition tạo bậc cao làm pattern gây hiểu lầm. Induction head minh hoạ cả hai: pattern rất thông tin, nhưng *nơi* nó attend phụ thuộc previous-token head. Gradient attribution cũng có thể bỏ sót vì softmax bão hoà → gradient nhỏ lên key.
- **Làm việc với ma trận low-rank:** giữ dạng nhân tử $AB$ tại điểm bottleneck; SVD của $AB$ qua SVD của $A$, $B$ và của $C=S_AV_AU_BS_B$.

## 13. Paper kiểm chứng các tuyên bố như thế nào

| Loại bằng chứng | Dùng cho | Ví dụ trong paper |
|---|---|---|
| Đẳng thức toán học | Khai triển đường, QK/OV, tensor product | Tương đương concat/sum; mixed-product |
| Đọc trọng số khai triển | Skip-trigram, copying | Bảng phần tử lớn nhất (có link non-cherry-picked cho 12-head và 32-head) |
| Thống kê eigenvalue | Copying, same-matching | 10/12 head copying; induction head ở góc dương–dương |
| Composition score | Tìm cặp head tương tác | Chỉ K-composition với một head layer 0 |
| Pattern trên dữ liệu | Hành vi head | Value-weighted attention trên Harry Potter; chuỗi ngẫu nhiên lặp 3 lần |
| Ablation có đóng băng pattern | Tầm quan trọng của bậc đường | Term importance: virtual head nhỏ, head layer 2 chiếm ưu thế |

**Đính chính của chính tác giả:** một lỗi trong thư viện ma trận low-rank áp nhầm đẳng thức kiểu $\mathrm{Tr}(AB)=\mathrm{Tr}(BA)$ cho Frobenius norm, nên thay vì $\|W_Q^{h_2\top}W^{h_2}_KW^{h_1}_OW^{h_1}_V\|_F$ họ tính các hoán vị vòng (vd. $\|W^{h_2}_KW^{h_1}_OW^{h_1}_VW^{h_2\top}_Q\|_F$). Đại lượng sai này lại đo đúng việc "query của induction head được xây từ subspace mà previous-token head ghi vào", nên nhấn mạnh induction và lọc bỏ composition khác. Hình sửa: K-composition với previous-token head vẫn quan trọng như cũ, nhưng **có thêm composition** — hai induction head cũng dựa vào một head nhìn vài token gần nhất. Bài học: kể cả bằng chứng từ trọng số cũng cần kiểm tra chéo.

## 14. Giới hạn

- **Mô hình đồ chơi:** attention-only ≤ 2 layer, không MLP, positional mechanism đặc biệt; một mô hình hai layer cụ thể được phân tích sâu.
- **MLP gần như chưa giải** (tác giả tự nhận là điểm yếu lớn nhất).
- **Tuyến tính có điều kiện:** chỉ khi đóng băng attention; QK circuit vẫn là softmax phi tuyến.
- **Eigenvalue không phải định nghĩa chặt** của copying; QK không so sánh được giữa các query.
- **Quy mô ma trận:** "hiểu" mô hình một layer theo nghĩa database, không giữ được trong đầu.
- **Có lỗi tính toán** ảnh hưởng một hình (đã đính chính).
- Cách xử lý LayerNorm làm đường qua các tập layer khác nhau khó so sánh.

## 15. Ứng dụng giả thuyết cho pruning / distillation

> **Suy luận của chúng ta**, không phải kết quả paper. Mỗi ý kèm cách kiểm chứng ở §16.

**15.1. Một nguồn $D_\text{res}$ có cấu trúc: di chuyển thông tin giữa vị trí.** Map phục hồi tại vết cắt thường là *theo từng token* ($\mathrm{Id}\otimes M$). Theo §3, head là $A\otimes W_{OV}$: phần $A$ (trộn giữa vị trí) **không thể** được tái tạo bởi bất kỳ $\mathrm{Id}\otimes M$ nào. Vậy khi cắt một layer:

$$
D \approx \underbrace{D_\text{lin}}_{\text{phần per-token của } \Delta x}+\underbrace{D_\text{res}}_{\text{phần do }A\ne \mathrm{Id}\text{ (di chuyển thông tin)} \;+\; \text{phi tuyến}}.
$$

Head gần $A\approx\mathrm{Id}$ (attend chính mình, ví dụ head "memory management") dễ phục hồi; previous-token head, induction head, head đọc ảnh thì không.

**15.2. Importance theo direct effect đánh giá thấp head "thượng nguồn".** §9 cho thấy đóng góp OV của head layer 1 nhỏ, nhưng chúng nuôi key của induction head qua K-composition. Khi chọn head/layer để cắt, nên cộng thêm điểm composition $C_K, C_Q$ với head phía sau, không chỉ dùng logit attribution hay loss khi ablate đơn lẻ.

**15.3. Cắt layer phá cặp composition.** Trong decoder hybrid, nếu previous-token head nằm ở layer full-attention $\ell$ và induction head ở $\ell'>\ell$, cắt $\ell$ có thể vô hiệu hoá $\ell'$ dù $\ell'$ được giữ. Ngược lại, cắt một layer chỉ chứa head kiểu skip-trigram (không compose) có thể chủ yếu tạo $D_\text{lin}$. Đây là một lời giải thích tiềm năng cho việc *một số* vết cắt layer phục hồi tốt.

**15.4. MLP và neuron cut.** Paper nói MLP không tuyến tính hoá được và neuron thường polysemantic (cập nhật 2023). Cắt neuron là xoá trục trong privileged basis sau GeLU — phù hợp với quan sát của bạn rằng cắt neuron khó phục hồi bằng map tuyến tính. MLP cuối có $W_UW^m_O$ tuyến tính → có thể ưu tiên đo importance neuron MLP cuối bằng direct logit effect.

**15.5. Bottleneck và pruning chiều residual.** Residual là bottleneck activation mang nhiều kênh song song trong superposition. Thu hẹp $d_\text{model}$ (0.8B → 0.1B) là ép bottleneck: các kênh trước đây nằm ở subspace rời nhau giờ phải chồng lấn → nhiễu giữa kênh, khó thấy trong loss trung bình.

**15.6. GatedDeltaNet.** Bỏ qua delta rule và chuẩn hoá, linear attention có dạng $o_t=W_O\sum_{j\le t}g_{t,j}\,W_Vx_j\,(x_j^\top W_K^\top W_Qx_t)$: vẫn có $W_{OV}=W_OW_V$ và $W_{QK}=W_Q^\top W_K$, nhưng "pattern" $g_{t,j}\,x_t^\top W_{QK}x_j$ **không qua softmax** (có thể âm, không chuẩn hoá) và bị decay theo khoảng cách. Phân tích composition trên trọng số vẫn áp được nguyên vẹn; đóng băng pattern thì phải đóng băng trạng thái/gate. (Đây là đơn giản hoá của chúng ta, không phải dạng chính xác của GatedDeltaNet.)

**15.7. OCR và copy circuit.** OCR về bản chất là *chép* nội dung ảnh ra text. Head trong decoder đọc token ảnh có OV dạng $W_UW_{OV}$ (vision feature → logit), một kiểu "copying xuyên modality"; thống kê eigenvalue không áp trực tiếp (nguồn ≠ vocab) nhưng direct logit effect thì có. Còn với text đã sinh (từ lặp, header lặp, bảng), copying/induction kiểu text-to-text áp dụng nguyên vẹn.

**15.8. Hidden-state KD.** Vì residual basis-free, loss KD trên hidden state nên đi qua projector học được. Gợi ý từ path expansion: có thể thêm loss trên **output từng head** (đóng góp $h(x)$) hoặc trên **attention pattern của head composition quan trọng**, thay vì chỉ residual cuối layer.

## 16. Thí nghiệm gợi ý

1. **Bản đồ copying/composition của decoder LightOnOCR.** Với mỗi head full-attention, tính $\text{copy}(W_UW_{OV}W_E)$ ở dạng nhân tử (eigenvalue của ma trận $d_\text{head}\times d_\text{head}$ nhờ $\lambda(AB)=\lambda(BA)$; fold RMSNorm gain trước). Tính $C_K, C_Q, C_V$ giữa mọi cặp layer (trừ baseline ngẫu nhiên). Kết quả: danh sách cặp previous-token → induction và head copying. So với ranking importance bạn đang dùng để cắt.
2. **Kiểm tra giả thuyết 15.1.** Với mỗi layer ứng viên: (a) cắt và fit map tuyến tính per-token tối ưu ở vết cắt (đo $D_\text{lin}$, $D_\text{res}$); (b) đo "độ lệch khỏi đường chéo" của attention trung bình của layer đó, ví dụ $1-\mathbb E[A_{ii}]$. Dự đoán: tương quan dương giữa (b) và $D_\text{res}$.
3. **Term-importance kiểu paper trên văn bản OCR lặp.** Dùng ảnh tài liệu có cụm từ lặp (tên riêng, header bảng). Đóng băng attention của teacher, rồi ablate output từng head (pattern-preserving) và đo loss trên token lặp lần 2 so với lần 1. Head có hiệu ứng lớn ở token lặp = ứng viên induction/copy; kiểm tra xem model đã prune (0.3B/0.1B) còn giữ chúng không.

## 17. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức | Ý nghĩa |
|---|---|---|
| Residual stream | $x^{(L)}=x^{(0)}+\sum_\ell\Delta x^{(\ell)}$ | Kênh truyền tuyến tính, basis-free |
| Virtual weight | $W_I^2W_O^1$ | Layer sau đọc gì của layer trước |
| Head | $h(x)=(A\otimes W_OW_V)x$ | Trộn vị trí × biến đổi vector |
| QK / OV | $W_{QK}=W_Q^\top W_K$, $W_{OV}=W_OW_V$ | Low-rank, hạng $\le d_\text{head}$ |
| Zero-layer | $T=W_UW_E$ | Bigram; direct path ở mọi mô hình |
| One-layer | $\mathrm{Id}\otimes W_UW_E+\sum_hA^h\otimes W_UW^h_{OV}W_E$ | Bigram + skip-trigram |
| QK circuit | $W_E^\top W^h_{QK}W_E$ | Điểm query token × key token |
| OV circuit | $W_UW^h_{OV}W_E$ | Token nguồn → logit |
| Skip-trigram bug | $f(a,b,c)=f_1(a,b)f_2(a,c)$ | Không biểu diễn tương tác 3 chiều |
| Copying | $\sum_i\lambda_i \,/\, \sum_i\lvert\lambda_i\rvert$ | Eigenvalue dương ≈ copying |
| Composition score | $\lVert W^{h_2}_{QK}W^{h_1}_{OV}\rVert_F \,/\, (\lVert W^{h_2}_{QK}\rVert_F\lVert W^{h_1}_{OV}\rVert_F)$ | Ứng viên tương tác (không phải nhân quả) |
| Virtual head | $(A^{h_2}A^{h_1})\otimes(W^{h_2}_{OV}W^{h_1}_{OV})$ | V-composition; nhỏ ở mô hình 2 layer |
| Induction QK term | $\mathrm{Id}\otimes A^{h_{-1}}\otimes W$, $W$ same-matching | Key dịch lùi 1 token |
| MLP pre-act | $W_I^mW_E+\sum_hA^h\otimes W_I^mW^h_{OV}W_E$ | Khung cơ chế cho neuron |
| Neuron → logit | $W_UW^m_O$ | Tuyến tính cho MLP cuối |
| Convolution | $W\ast x=\sum_vA^v\otimes W_v$ | Attention tổng quát hoá conv |

### Kết luận ngắn

Paper cho thấy chỉ cần viết lại transformer theo cách tương đương toán học — head cộng dồn, $A\otimes W_{OV}$, khai triển thành tổng đường đi — là đọc được trực tiếp từ trọng số: mô hình 0 layer là bigram, 1 layer là bigram + skip-trigram (chủ yếu copying), 2 layer dùng K-composition để tạo induction head. Term importance xác nhận virtual head không quan trọng ở quy mô này, nhưng cũng chỉ ra rằng thước đo theo đường logit bỏ qua vai trò Q/K-composition. MLP vẫn là vùng tối. Với pruning, bài học đáng thử nhất là: phần không phục hồi tuyến tính được có thể đến từ việc **di chuyển thông tin giữa vị trí** và từ **composition bị cắt đứt**, không chỉ từ phi tuyến.

**Đọc bản gốc:** https://transformer-circuits.pub/2021/framework/index.html
