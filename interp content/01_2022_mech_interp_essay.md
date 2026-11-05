# Mechanistic Interpretability, Variables, and the Importance of Interpretable Bases (2022)

**Tác giả:** Chris Olah (ghi chú không chính thức, đăng ngày 27/06/2022).  
**Nguồn:** https://transformer-circuits.pub/2022/mech-interp-essay/index.html  
**Chủ đề:** Định nghĩa mechanistic interpretability qua phép so sánh với reverse engineering chương trình máy tính; biến (variable), activation, privileged basis, polysemantic neuron, định nghĩa "feature".  
**Vị trí trong lộ trình:** Ghi chú **đầu tiên**, không cần đọc gì trước. Bài này ra sau paper Framework (2021) nhưng nên đọc trước, vì nó cho bộ từ vựng và "lý do tồn tại" của cả lĩnh vực; đọc xong thì chuyển sang `02_2021_mathematical_framework.md`.

> Đây là ghi chú học tập diễn giải bằng tiếng Việt, không phải bản dịch nguyên văn. Phần 0–1 có thêm giải thích nền cho người mới (đánh dấu rõ chỗ nào là của bài, chỗ nào là bổ sung). Mọi ứng dụng cho pruning/distillation là suy luận mở rộng của chúng ta, không phải nội dung bài viết.

## Mục lục

0. [Mechanistic interpretability là gì](#0-mechanistic-interpretability-là-gì)
1. [Từ vựng cốt lõi cho người mới](#1-từ-vựng-cốt-lõi-cho-người-mới)
2. [Phép so sánh trung tâm: mạng nơ-ron như một chương trình đã biên dịch](#2-phép-so-sánh-trung-tâm-mạng-nơ-ron-như-một-chương-trình-đã-biên-dịch)
3. [Vượt qua curse of dimensionality](#3-vượt-qua-curse-of-dimensionality)
4. [Variables và activations](#4-variables-và-activations)
5. [Memory layout đơn giản, neuron và privileged basis](#5-memory-layout-đơn-giản-neuron-và-privileged-basis)
6. [FAQ của bài: distributed representation, feature, modularity](#6-faq-của-bài-distributed-representation-feature-modularity)
7. [Bài viết lập luận bằng gì](#7-bài-viết-lập-luận-bằng-gì)
8. [Giới hạn](#8-giới-hạn)
9. [Ứng dụng giả thuyết cho pruning / distillation](#9-ứng-dụng-giả-thuyết-cho-pruning--distillation)
10. [Thí nghiệm gợi ý](#10-thí-nghiệm-gợi-ý)
11. [Tóm tắt công thức / khái niệm](#11-tóm-tắt-công-thức--khái-niệm)

---

## 0. Mechanistic interpretability là gì

**Theo bài:** mechanistic interpretability (gọi tắt *mech interp*, "diễn giải cơ chế") cố gắng **reverse engineer** (dịch ngược) mạng nơ-ron, giống như dịch ngược một file binary đã biên dịch về mã nguồn đọc được. Tham số mạng "là một chương trình binary chạy trên một máy ảo kỳ lạ" — chính là kiến trúc mạng.

**Bổ sung nền (không phải của bài):** so với các nhánh interpretability khác, mech interp khác ở *mục tiêu*:

| Cách tiếp cận | Câu hỏi | Ví dụ |
|---|---|---|
| Saliency / attribution | Phần input nào ảnh hưởng output *quanh một điểm dữ liệu*? | gradient × input, attention map |
| Probing | Thông tin X có *giải mã được* từ hidden state không? | linear probe trên layer 6 |
| **Mech interp** | Mạng *tính* output bằng *thuật toán gì*, qua các thành phần nào? | induction head, curve detector |

Probing trả lời "có thông tin hay không"; mech interp hỏi "thông tin đó được *dùng* như thế nào, bởi ai, qua trọng số nào".

## 1. Từ vựng cốt lõi cho người mới

| Thuật ngữ | Giải thích ngắn (tiếng Việt) |
|---|---|
| **Activation** | Giá trị trung gian khi chạy forward; bài coi nó như *bộ nhớ / trạng thái chương trình*. |
| **Variable** (biến) | Một mảnh trạng thái có nghĩa riêng, hiểu được độc lập. Trong mạng: một neuron hoặc một *feature direction* (hướng trong không gian activation). |
| **Feature** | "Đơn vị độc lập" mà biểu diễn có thể phân rã thành (định nghĩa Olah ưa thích, xem §6). Ví dụ trực quan: curve detector, car detector trong InceptionV1. |
| **Basis** (cơ sở) | Hệ trục toạ độ của không gian activation. Các trục chuẩn = các neuron. |
| **Privileged basis** | Basis "đặc quyền": kiến trúc (vd. hàm kích hoạt theo từng phần tử) khiến feature có xu hướng *thẳng hàng* với trục neuron. |
| **Basis-free** | Không có trục ưu tiên: xoay cả không gian không đổi hành vi (vd. residual stream của transformer). |
| **Polysemantic neuron** | Neuron phản ứng với nhiều thứ không liên quan → không thể coi neuron là một biến. |
| **Circuit** | (Bổ sung từ Circuits thread) đồ thị con gồm các feature nối với nhau bằng trọng số, thực hiện một phép tính hiểu được. Bài viết gợi ý: nếu feature hiểu được, *mỗi tham số* là một cạnh nối hai feature. |
| **Curse of dimensionality** | Thể tích không gian input tăng theo hàm mũ với số chiều. |
| **Data manifold** | Tập con (chiều thấp hơn) của không gian input nơi dữ liệu thật nằm. |

## 2. Phép so sánh trung tâm: mạng nơ-ron như một chương trình đã biên dịch

| Chương trình thông thường | Mạng nơ-ron |
|---|---|
| Reverse engineering | Mechanistic interpretability |
| Program binary | Network parameters |
| VM / processor / interpreter | Network architecture |
| Program state / memory | Layer representation / activations |
| Variable / memory location | Neuron / feature direction |

Ý chính: những câu hỏi "trơn trượt" về mạng nơ-ron trở nên rõ ràng nếu hỏi cùng câu đó cho chương trình thường. Kết luận gây chú ý nhất: **tìm và hiểu các neuron diễn giải được (≈ hiểu biến) không chỉ là một câu hỏi thú vị, mà có lẽ là nhiệm vụ trung tâm.**

## 3. Vượt qua curse of dimensionality

Một hàm trên $\mathbb{R}^n$ — nếu muốn "hiểu" bằng cách lấy mẫu lưới $k$ điểm mỗi chiều — cần

$$
N_{\text{mẫu}} = k^{\,n},
$$

bùng nổ theo hàm mũ. Bài liệt kê các cách né:

1. **Chỉ hiểu trên data manifold.** Olah từng đặt cược vào hướng này, nhưng manifold của ảnh/ngôn ngữ vẫn rất nhiều chiều (một vật bất kỳ có thể đi vào khung hình từ bất kỳ phía nào).
2. **Toy model với input ít chiều** — né bài toán.
3. **Hiểu cục bộ quanh một điểm dữ liệu** — đại khái là saliency map.
4. **Câu trả lời của mech interp:** giống lập trình viên đọc code, ta dùng **tham số như một mô tả hữu hạn, không theo hàm mũ** của hành vi trên toàn không gian input.

Hệ quả bài nêu: đừng mong mech interp dễ hay có "quy trình cắt bánh quy"; nó **ít nhất khó như reverse engineer một chương trình lớn**.

## 4. Variables và activations

Câu lệnh `y = x + 5;` vô nghĩa nếu không biết `x`, `y` là gì; ngược lại, nghĩa của `x`, `y` đến từ cách chúng được các phép toán khác dùng. Người dịch ngược không có tên biến, và còn phải *tự chia bộ nhớ thành các biến*.

Với mạng, xét một layer:

$$
a^{(l+1)} = \sigma\!\left(W^{(l)} a^{(l)}\right),\qquad
W^{(l)}_{ij}\;=\;\text{"lệnh" nối biến } a^{(l)}_j \to a^{(l+1)}_i .
$$

$$
\boxed{\text{Một tham số chỉ có nghĩa khi ta hiểu activation đầu vào và đầu ra của nó.}}
$$

Vì activation là vector nhiều chiều, ta lại gặp curse of dimensionality — và lời giải giống bộ nhớ máy tính: **chia activation thành các mảnh hiểu được độc lập**.

Ngoại lệ đặc biệt: với **attention-only transformer** (xem ghi chú 02), tính tuyến tính cho phép mô tả mọi phép toán chỉ qua input và output (token → logit), giống một hàm mô tả được chỉ bằng tham số và giá trị trả về. Nhưng trong đa số trường hợp, ta *bắt buộc* phải phân rã activation.

## 5. Memory layout đơn giản, neuron và privileged basis

Máy tính có layout bộ nhớ dễ hiểu: phần lớn byte biểu diễn *một* thứ, vì vừa dễ cho lập trình viên vừa hiệu quả cho phần cứng. Tương tự, **hàm kích hoạt khiến feature thẳng hàng với neuron** — giống CPU thao tác theo byte khuyến khích dữ liệu gom theo byte.

**Vì sao hàm kích hoạt tạo privileged basis** (diễn giải toán của chúng ta, nhất quán với bài): với một phép quay trực giao $R$ bất kỳ,

$$
W_2 W_1 = (W_2 R^\top)(R W_1)\quad\text{(không có phi tuyến: xoay tự do, basis vô nghĩa)},
$$

$$
\boxed{\sigma(Rz) \neq R\,\sigma(z)\ \text{ với } \sigma \text{ áp từng phần tử}}
$$

nên khi có ReLU/GELU, xoay không còn "miễn phí"; các trục neuron trở nên đặc biệt.

Tuy vậy, **nhiều neuron là polysemantic**: chúng giúp biểu diễn những feature không nằm gọn trong một neuron (bài trỏ tới SoLU paper, Mục 3). Thông điệp chính: *khả năng phân rã biểu diễn thành các phần hiểu được độc lập là thiết yếu cho thành công của mech interp.*

## 6. FAQ của bài: distributed representation, feature, modularity

**Neuron diễn giải được có mâu thuẫn với distributed representation không?** Không. Word embedding có thể có các chiều số ít–số nhiều, nam–nữ, lớn–nhỏ; xoay sao cho chúng là trục cơ sở vẫn được. "Distributed" nghĩa là *các khía cạnh độc lập nằm ở các hướng khác nhau*, không phải "mỗi thứ trải khắp mọi neuron".

**Feature là gì?** Bài thừa nhận định nghĩa rất khó, và đưa ra ba ứng viên:

| Định nghĩa | Ưu | Nhược |
|---|---|---|
| Tính chất có ý nghĩa, nói ra được, được mã hoá thành một hướng | Trực quan (car detector) | Lấy con người làm trung tâm; loại bỏ feature người không hiểu |
| **"Đơn vị độc lập" mà biểu diễn phân rã được thành** (Olah thích nhất) | Không phụ thuộc người | Mơ hồ hơn |
| Thứ mà mạng *sẽ dành riêng một neuron* nếu có đủ neuron | Không phụ thuộc người | Khó kiểm chứng trực tiếp |
| (Bài *không* đồng ý) Bất kỳ hàm nào của input | Đơn giản | Hỗn hợp "car + cat detector" cũng là feature; mất phân biệt với biến tiềm ẩn thật |

**Vì sao không tập trung vào modularity?** Modularity giúp *dễ hơn* khi đã hiểu được; còn phân rã thành feature hiểu được mới khiến việc hiểu *khả thi*. So sánh: hiểu chương trình viết thành một hàm khổng lồ thì khó; hiểu chương trình mà bộ nhớ không chia thành biến có nghĩa thì "đáng sợ". Feature diễn giải được tạo ra modularity tối hậu: mỗi tham số nối hai feature, có thể xét riêng.

## 7. Bài viết lập luận bằng gì

Đây là **essay không chính thức**, không có thí nghiệm hay số liệu. Bằng chứng gồm:

- **Phép loại suy** với reverse engineering phần mềm (lập luận chính).
- **Quan sát dẫn chứng** từ công trình trước: curve/car detector trong InceptionV1 (Circuits thread), Voss et al. (tham số như lệnh binary), Elhage et al. (attention-only transformer), SoLU paper (polysemanticity), branch specialization.
- **Kinh nghiệm cá nhân** của tác giả (từng theo hướng data manifold).

Nên đọc nó như "tuyên ngôn phương pháp luận", không phải kết quả.

## 8. Giới hạn

- **Không có bằng chứng định lượng**; phép loại suy có thể gãy (mạng không do người viết, không có "trình biên dịch" mang ý đồ).
- **Định nghĩa feature chưa thoả đáng** — chính tác giả nói vậy.
- **Polysemanticity chưa được giải quyết** trong bài; chỉ được nêu là "vấn đề rất khó" (các bài sau về superposition, dictionary learning xử lý tiếp).
- Giả định ngầm: mạng *có thể* hiểu được theo các phép toán trên feature độc lập ("Let's assume for a moment…").

## 9. Ứng dụng giả thuyết cho pruning / distillation

> **Toàn bộ phần này là suy luận của chúng ta**, không phải nội dung bài.

**9.1. Vì sao cắt neuron khó phục hồi hơn cắt layer?** Khung privileged basis cho một giả thuyết gọn:

- Đầu ra của một block (attention hoặc MLP) được *cộng* vào residual stream, vốn **basis-free** và tuyến tính. Bỏ một layer là bỏ một số hạng cộng $\Delta x^{(l)}$; phần của nó tương quan tuyến tính với các số hạng còn lại có thể được một map tuyến tính tại chỗ cắt bù lại → rơi vào $D_{\text{lin}}$.
- Neuron MLP nằm *sau* phi tuyến, trong **privileged basis**. Cắt neuron $i$ là xoá hẳn một trục mà feature (hoặc nhiều feature, nếu polysemantic) thẳng hàng; sau $\sigma$, thông tin này không còn là tổ hợp tuyến tính của các neuron giữ lại → rơi vào $D_{\text{res}}$.

$$
D_{\text{res}}^{\text{neuron}} \gg D_{\text{res}}^{\text{layer}} \quad\text{(giả thuyết, cần đo)}.
$$

**9.2. Hidden-state KD.** Residual stream không có trục ưu tiên → so khớp hidden state student–teacher nên qua một **projector tuyến tính học được** (hoặc metric bất biến với phép quay như CKA), không ép từng chiều. Ngược lại, nếu distill activation MLP (privileged basis), so khớp theo neuron *có thể* có nghĩa — nhưng chỉ khi neuron của student và teacher tương ứng nhau.

**9.3. OCR.** Với LightOnOCR, "biến" tự nhiên có thể là: glyph/ký tự, vị trí dòng, ranh giới từ, ngôn ngữ. Câu hỏi diễn giải hữu ích cho pruning: neuron nào mang các biến này, và chúng monosemantic tới đâu?

**9.4. Hybrid GatedDeltaNet.** Trạng thái recurrent của lớp linear-attention là một *ma trận bộ nhớ*; câu hỏi "basis của nó có privileged không?" chưa được bài nào ở đây trả lời. Gating theo từng phần tử *có thể* tạo trục ưu tiên — đáng kiểm tra trước khi cắt chiều trạng thái.

## 10. Thí nghiệm gợi ý

1. **Kiểm tra basis-free bằng phép quay.** Chọn một vị trí residual, áp phép quay trực giao ngẫu nhiên $R$: nhân $R$ vào mọi ma trận *ghi* vào residual và $R^\top$ vào mọi ma trận *đọc*. Lưu ý RMSNorm có gain theo từng chiều: phải fold gain vào ma trận đọc kế tiếp trước, nếu không phép quay sẽ không bảo toàn. Kỳ vọng: output không đổi (sanity check). Làm cùng việc đó *bên trong* MLP (sau activation) → output đổi. Đây là minh hoạ trực tiếp privileged vs basis-free trên chính mô hình của bạn.
2. **Đo "mức thẳng hàng" của feature với neuron.** Với vài layer MLP của decoder, lấy top-k patch ảnh/token kích hoạt mạnh nhất cho mỗi neuron; gán nhãn nhanh (monosemantic/polysemantic). So sánh với độ hại $D_{\text{res}}$ khi cắt chính neuron đó.
3. **So $D_{\text{lin}}$ tại hai loại vết cắt:** (a) bỏ một block, phục hồi bằng map tuyến tính ở residual; (b) bỏ 25% neuron một MLP, phục hồi bằng map tuyến tính từ neuron giữ lại tới output MLP. Giả thuyết §9.1 dự đoán tỉ lệ $D_{\text{lin}}/D$ cao hơn ở (a).

## 11. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / mô tả | Ý nghĩa |
|---|---|---|
| Mech interp | Reverse engineer tham số → thuật toán | Tham số = binary, kiến trúc = VM |
| Curse of dimensionality | $N = k^n$ | Không thể hiểu hàm bằng cách liệt kê input |
| Lời giải | Tham số là mô tả hữu hạn | Đọc "code" thay vì liệt kê input |
| Variable | Neuron / feature direction | Đơn vị trạng thái hiểu được độc lập |
| Basis-free | $W_2W_1=(W_2R^\top)(RW_1)$ | Xoay không đổi hành vi |
| Privileged basis | $\sigma(Rz)\ne R\sigma(z)$ | Phi tuyến theo phần tử làm trục neuron đặc biệt |
| Polysemantic | Neuron ↔ nhiều feature | Cản trở việc coi neuron là biến |
| Feature (Olah) | "Đơn vị độc lập" của biểu diễn | Không phụ thuộc con người |
| Circuit | Feature + trọng số nối chúng | Mỗi tham số nối hai feature |

### Kết luận ngắn

Bài essay đặt nền cho cả lộ trình: mech interp là reverse engineering, và giống như hiểu chương trình cần hiểu biến, hiểu mạng cần **phân rã activation thành các feature hiểu được độc lập**. Privileged basis (do hàm kích hoạt) làm việc này dễ hơn; polysemanticity làm nó khó hơn. Với dự án pruning, khung này gợi ý một lý do có thể kiểm chứng cho việc cắt neuron khó phục hồi hơn cắt layer.

**Đọc bản gốc:** https://transformer-circuits.pub/2022/mech-interp-essay/index.html
