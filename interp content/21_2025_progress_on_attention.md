# Progress on Attention (2025)

**Tác giả:** Adam Jermyn, Jack Lindsey, Rodrigo Luger, Nick Turner, Trenton Bricken, Adam Pearce, Callum McDougall, Ben Thompson, Jeff Wu, Joshua Batson, Kelley Rivoire, Christopher Olah; biên tập bởi Jack Lindsey (Anthropic). Trang không ghi ngày cụ thể; đường dẫn thuộc năm 2025.  
**Nguồn:** https://transformer-circuits.pub/2025/attention-update/index.html  
**Chủ đề:** Attention superposition, Attention Replacement Layer, Multi-Token Transcoder (MTC), QK/OV coupling, OV đa chiều, Attention Value SAE, biểu diễn xuyên layer, QK diagonalization.  
**Vị trí trong lộ trình:** Đọc sau *A Mathematical Framework* (2021), bản update tháng 1/2024 về toy model attention superposition, và *Biology* (2025). Nên đọc cùng note số 20 (*Tracing Attention Computation*), vì MTC là một trong các phương án "checkpoint" được nhắc ở đó. Bài này là báo cáo tiến độ: không có lời giải hoàn chỉnh, nhưng chỉ ra cấu trúc của bài toán attention.

> Tài liệu học tập bằng tiếng Việt, ghi chú diễn giải chứ không phải bản dịch. Tác giả yêu cầu coi đây như "chia sẻ sơ bộ trong buổi họp lab", không phải paper hoàn chỉnh. Phần ứng dụng cho pruning/distillation là suy luận của chúng tôi.

## Mục lục

1. [Bài toán và cách chia nhỏ](#1-bài-toán-và-cách-chia-nhỏ)
2. [Attention Replacement Layer](#2-attention-replacement-layer)
3. [Multi-Token Transcoder](#3-multi-token-transcoder)
4. [Bằng chứng superposition và QK OV coupling](#4-bằng-chứng-superposition-và-qk-ov-coupling)
5. [OV đa chiều và biểu diễn xuyên layer](#5-ov-đa-chiều-và-biểu-diễn-xuyên-layer)
6. [Attention Output SAE](#6-attention-output-sae)
7. [Hình thành attention pattern: QK diagonalization](#7-hình-thành-attention-pattern-qk-diagonalization)
8. [Cách paper kiểm chứng](#8-cách-paper-kiểm-chứng)
9. [Giới hạn](#9-giới-hạn)
10. [Ứng dụng giả thuyết cho pruning và distillation](#10-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
11. [Thí nghiệm gợi ý](#11-thí-nghiệm-gợi-ý)
12. [Tóm tắt công thức và khái niệm](#12-tóm-tắt-công-thức-và-khái-niệm)

---

## 1. Bài toán và cách chia nhỏ

**Attention superposition**: một "đơn vị chức năng" của attention (attention feature) có thể được thực hiện bởi tổ hợp nhiều head, và một head tham gia nhiều chức năng; tương tự superposition của feature trên neuron.

Tóm tắt kết luận của tác giả: có bằng chứng đáng kể cho attention superposition *giữa các head* và *giữa các layer*; không thấy "số chiều OV ưa thích" (một phổ liên tục); điều kiện QK và OV **gắn kết** với nhau; MTC là công cụ thực tế. Bài toán còn lại lớn nhất: **vì sao attention nhìn vào đâu** (pattern formation).

Sáu câu hỏi con:

| # | Câu hỏi | Công cụ trả lời |
|---|---|---|
| 1 | Có superposition giữa các head trong một layer không? | MTC so với MTC một head |
| 2 | QK-condition ("chuyển từ đâu tới đâu") và OV-condition ("chuyển cái gì") có gắn kết không? | MTC so với SAE + head |
| 3 | Feature có đa chiều trong OV ($\mathrm{Rank}(W_{OV})>1$)? | Phân cụm head-loading |
| 4 | Feature có đa chiều trong QK ($\mathrm{Rank}(W_{QK})>1$)? | QK diagonalization (đang làm) |
| 5 | Feature có trải qua nhiều layer? | Attention Value SAE hai layer |
| 6 | Pattern hình thành thế nào (phần phi tuyến của QK)? | QK diagonalization |

---

## 2. Attention Replacement Layer

**Ý tưởng:** một layer attention rất rộng (nhiều head) được huấn luyện bắt chước output của layer gốc (MSE), với regularization để QK và OV **low-rank**; có biến thể ép pattern thưa. Mỗi head của layer thay thế là một attention feature.

Regularizer có các tính chất lý thuyết mong muốn:

$$
\boxed{\;\mathcal R=\left[L_{\mathrm{nuc}}(W_{QK})\,L_{\mathrm{nuc}}(W_{OV})\right]^{\beta},\qquad \tfrac12<\beta<1\;}
$$

với $L_{\mathrm{nuc}}$ là **nuclear norm** (chuẩn hạt nhân: tổng giá trị kỳ dị, proxy lồi cho hạng). Tính trực tiếp đắt, nhưng với ma trận đã phân tích $W_{QK}=W_QW_K$, cực tiểu của $\|W_Q\|_F^2+\|W_K\|_F^2$ trùng với cực tiểu của $L_{\mathrm{nuc}}(W_{QK})$. Đẳng thức chuẩn đứng sau là $\|W\|_*=\min_{W=AB}\tfrac12(\|A\|_F^2+\|B\|_F^2)$.

Tính chất:
- Nếu mô hình gốc làm nhiều **skip-trigram** độc lập, penalty khuyến khích tách thành các head riêng có QK và OV hạng 1.
- Nếu có **induction** với QK hạng $N$ (prefix matching) và OV hạng $M$ trong các không gian con gần rời nhau, penalty khuyến khích gộp thành *một* head có QK hạng $N$, OV hạng $M$.

**Kết quả:** có "dấu hiệu sống" (head làm skip-trigram cụ thể, head làm induction) nhưng luôn lộn xộn. Giả thuyết chính: ML chưa tối ưu, do khó mở rộng số head, không gian thiết kế lớn, không có metric định lượng rõ để so sánh. Tác giả nghĩ cách này có thể chạy được nhưng không dễ, nên chuyển sang các thí nghiệm đơn giản hơn.

---

## 3. Multi-Token Transcoder

MTC nhận residual stream của **cả context** ở đầu vào một layer, dự đoán residual sau layer (hoặc output attention). Ba thành phần:

1. **Encoder theo từng token** ở vị trí nguồn (như SAE) → activation phía nguồn $a_i(k)$.
2. **Phép attention theo feature**: mỗi feature $i$ có **head loading vector** $c_i$, tổ hợp tuyến tính attention pattern của các head gốc, cộng thêm một pattern đơn vị (identity) để mô phỏng residual và cho phép tính toán kiểu MLP.
3. **Decoder** ở vị trí đích.

Viết gọn (ký hiệu của chúng tôi):

$$
A^{(i)}=\sum_{h} c_{i,h}A_h + c_{i,0}\,I,\qquad
\tilde a_i(q)=\sum_{k\le q}A^{(i)}_{qk}\,a_i(k),\qquad
\hat y(q)=\sum_i \tilde a_i(q)\,d_i .
$$

Nhờ pattern identity, MTC có thể làm phép **Attention+MLP hợp nhất**: vừa di chuyển thông tin vừa biến đổi. MTC tạo pattern đúng như toy model attention superposition, nên hiệu năng của nó là một proxy cho giả thuyết đó. Vì lý do hiệu năng, chỉ di chuyển top-k feature trên mỗi context; với k đủ lớn, việc chọn top-k chỉ có tác dụng đầu quá trình huấn luyện.

**Loss:** MSE + L1 trên activation + một L1 khác trên **pre-activation âm** (giúp tránh dead feature; penalty này ngầm đặt mật độ feature mong muốn và thường nhỏ hơn nhiều bậc so với L1 chính).

MTC **không** trả lời câu 4–6: một layer, kế thừa pattern của layer gốc.

**Kết quả** (100.000 feature, mô hình 18 layer):
- Nhiều feature có pattern sạch, dễ hiểu: một feature nhìn vào **giá trị bất thường** (danh sách toàn 1 có vài số 2 → nhìn vào số 2; code lặp → chỗ phá mẫu); một feature **valence** đổi dấu khi cảm xúc âm sắp bị phủ định.
- Feature induction dùng chủ yếu các induction head đã biết; chúng **phân cụm mạnh** theo head loading, thường có nhiều feature gần như cùng head loading.
- Cụm head-loading có nghĩa ngữ nghĩa: vị trí trong câu/mệnh đề, dự đoán dấu phân cách, induction trong code.
- Motif thường gặp: dương ở token hiện tại, âm ở token trước. Có thể là "edge detector" giữa các token.
- Khi MTC nhắm residual sau layer, nhiều feature chỉ nhìn token hiện tại (hoàn toàn kiểu MLP); khi nhắm output attention, vẫn có một ít (attention head cũng làm tính toán cục bộ), nhưng ít hơn nhiều.

---

## 4. Bằng chứng superposition và QK OV coupling

### 4.1 Superposition giữa các head

So sánh MTC thường với **MTC một head** (Top-1 trên head loading vector: mỗi feature bị gắn một head):

| Đo | Kết quả |
|---|---|
| L0 tại cùng MSE | MTC một head kém hơn khoảng **2–4×** |
| Chất lượng feature | Kém hơn: hay nhìn giữa các token không liên quan |

Đây là cập nhật vừa phải ủng hộ attention superposition.

### 4.2 QK và OV có gắn kết?

Nếu gắn kết, MTC (học cùng lúc "chuyển gì" và "chuyển thế nào") phải tốt hơn baseline "SAE feature được chuyển qua head".

- **Baseline 1:** SAE feature qua head chuyển nó nhiều nhất trong context đó (lợi cho SAE vì chọn head động). MTC sạch hơn một chút, không nhiều: cập nhật **yếu** ủng hộ coupling.
- **Baseline 2:** huấn luyện chung một SAE (encoder + decoder, không attention) và một decoder MTC Top-1 dùng activation của SAE (gradient không chảy về encoder). Feature khó hiểu hơn rõ (pattern mờ, rộng, hoặc polysemantic); định lượng thì tốt hơn một chút ở vùng L0 thấp/MSE cao, ngang hoặc kém hơn nhiều ở chỗ khác.

**Kết luận của tác giả:** QK và OV *có lẽ* gắn kết; nên nghiên cứu chúng chung thay vì "lấy feature residual rồi xem head nào mang".

---

## 5. OV đa chiều và biểu diễn xuyên layer

### 5.1 OV đa chiều: liên tục, không rời rạc

Feature trong cùng cụm head-loading có cùng pattern, nên khác nhau ở *thông tin chuyển*. Một cụm có thể là một feature có OV nhiều chiều (MTC chỉ cho một chiều OV mỗi feature). Kỳ vọng ban đầu: nhiều cụm nhỏ (skip-trigram, 1 chiều) và ít cụm lớn (induction/copy, nhiều chiều).

**Kết quả:** k-means trên head-loading vector, quét $k$ qua nhiều bậc độ lớn: kích thước cụm theo **phân bố lũy thừa trơn**, không có hai quần thể. Gợi ý một **liên tục** về số chiều OV: induction "rộng" và "hẹp", skip-trigram tổng quát nhiều chiều.

### 5.2 Attention Value SAE và biểu diễn xuyên layer

**Attention Value SAE**: SAE trên vector value nối của nhiều head. Huấn luyện ba SAE trên mô hình 18 layer: layer 4, layer 10 (có induction head rõ nhất), và chung cả hai.

So sánh: cộng L0 của hai SAE đơn, gộp NMSE (cộng rồi chia $\sqrt2$):

$$
L_0^{\text{gộp}}=L_0^{(4)}+L_0^{(10)},\qquad
\mathrm{NMSE}^{\text{gộp}}=\frac{\mathrm{NMSE}^{(4)}+\mathrm{NMSE}^{(10)}}{\sqrt2}.
$$

SAE hai layer tốt hơn: khoảng **một nửa L0** tại cùng NMSE, hoặc NMSE tốt hơn khoảng **0.3** tại cùng L0. Đây là bằng chứng đáng kể cho **biểu diễn attention xuyên layer**.

Lặp lại phân cụm với SAE hai layer (head loading = chuẩn decoder theo từng head): vẫn phân bố lũy thừa, không có quần thể rời rạc, nên kết quả 5.1 không phải do giới hạn một layer.

---

## 6. Attention Output SAE

SAE trên output của layer attention: feature nhìn chung dễ hiểu (induction, succession). Ghép với CLT để làm attribution graph có thông tin attention, giúp định vị tính toán attention vào layer/vị trí cụ thể và có visualization cho bước qua attention. Hữu ích **vừa phải**, kém kỳ vọng: feature thường giống hoặc dễ đoán từ đồ thị CLT-only.

So với MTC: không cho biết pattern hình thành thế nào; không gắn cố định với head (đóng góp head thay đổi theo prompt); chỉ visualize được phía query. Ưu điểm duy nhất: huấn luyện tiện hơn (một vị trí token mỗi lần).

Biến thể xuyên layer hữu ích cho attribution graph:
- **Type 1**: đọc residual ở layer, "strictly causal", tái tạo mọi output MLP và attention phía sau.
- **Type 2**: đọc output attention của layer, "weakly causal", tái tạo chính input của nó và các output phía sau. Phải weakly causal vì output attention không dự đoán được từ residual ở một vị trí.

Kiến trúc này cải thiện MSE/L0 và đơn giản hóa đồ thị phần nào: thêm bằng chứng cho biểu diễn xuyên layer.

---

## 7. Hình thành attention pattern: QK diagonalization

Ví dụ động lực: câu hỏi trắc nghiệm, toàn bộ logic thú vị nằm ở việc *chọn* đáp án để nhìn vào.

**Ý tưởng:** đưa query và key của một head qua hai encoder riêng → feature phía query $u_i(q)$ và phía key $v_i(k)$; chỉ cho feature query thứ $i$ tương tác với feature key thứ $i$ ("chéo hóa"). Mỗi cặp là một **QK feature**, cho một ma trận score $n_{ctx}\times n_{ctx}$:

$$
\boxed{\;\hat A_{qk}=\operatorname{softmax}_k\Big(\sum_i u_i(q)\,v_i(k)\Big)\;}
$$

(dạng hạng 1; dạng hạng cao thay tích vô hướng bằng tích chấm giữa các vector). Loss: MSE giữa pattern tái tạo và pattern gốc; L1 riêng cho activation query và key; L1 trên tích activation; penalty pre-activation âm. Biến thể: huấn luyện chung mọi head trong layer với ma trận head weight học được; biến thể tuyến tính không softmax.

**Kết quả:**
- Với QK feature hạng 1, khó đạt pattern trung thực (NMSE $\lessapprox 0.25$) với L0 hợp lý ($\ll d_{head}$).
- Gần như mọi biến thể: một số nhỏ, cỡ $\mathcal O(1000)$, **induction feature** làm gần hết việc; phần còn lại đóng góp không đáng kể (đo bằng ablation). Ở bản một head, feature không phải induction thường chết. Ở bản nhiều head (head weight không ràng buộc dấu), chúng **triệt tiêu nhau**, nửa dương nửa âm.
- Lạ: tăng tổng số feature không cải thiện MSE tại cùng L0 (khác autoencoder). Gợi ý lỗi kiến trúc (hạng 1 không đủ), nên đang chuyển sang QK feature hạng cao.
- Induction feature tìm được khá dễ hiểu và phức tạp hơn induction theo token: induction trên **họ**, trên âm tiết trong từ bắt đầu bằng C hoặc K, trên từ chỉ phương hướng.

---

## 8. Cách paper kiểm chứng

| Câu hỏi | Thí nghiệm | Số liệu |
|---|---|---|
| Superposition giữa head | MTC vs MTC Top-1 | L0 kém 2–4× tại cùng MSE |
| QK/OV coupling | MTC vs SAE + head; huấn luyện SAE/MTC chung | Khác biệt định tính; định lượng hỗn hợp |
| OV đa chiều | k-means head loading, quét $k$ | Phân bố lũy thừa |
| Xuyên layer | Value SAE layer 4+10 vs riêng lẻ | ~½ L0 hoặc NMSE tốt hơn ~0.3 |
| Pattern formation | QK diagonalization | NMSE $\lessapprox0.25$ khó đạt; ~$\mathcal O(1000)$ induction feature |

Hầu hết bằng chứng là đường cong MSE/L0 và kiểm tra định tính feature; chưa có can thiệp nhân quả trên hành vi.

---

## 9. Giới hạn

- Báo cáo sơ bộ, độ tin cậy thấp theo chính tác giả.
- Attention Replacement Layer bị kẹt vì ML; không có metric định lượng rõ để so sánh thiết kế.
- MTC kế thừa pattern của mô hình gốc: không giải thích pattern formation, không có QK đa chiều, chỉ một layer.
- So sánh coupling phụ thuộc baseline; chưa so với SAE qua *mọi* head.
- QK diagonalization hạng 1 có thể không đủ biểu đạt; hành vi "tăng L0 mà không giảm MSE" chưa giải thích được.
- Mọi kết quả trên một mô hình 18 layer; chưa có kiểm chứng trên mô hình khác.

---

## 10. Ứng dụng giả thuyết cho pruning và distillation

> **Suy luận của chúng tôi.**

**10.1 Head giống neuron, layer thì không hẳn.** Nếu attention feature nằm trong superposition *giữa các head* (MTC một head kém 2–4×), cắt một head giống cắt một neuron polysemantic: nhiều feature mất một phần "người mang", sai số phụ thuộc ngữ cảnh. Dự đoán: **cắt head khó phục hồi hơn**, gần với kinh nghiệm cắt neuron MLP của bạn (nhiều D_res). Ngược lại, nếu feature trải *qua nhiều layer* (value SAE hai layer tốt hơn rõ), cắt một layer chỉ lấy đi một phần của biểu diễn phân tán mà layer còn lại vẫn mang, nên phần mất có thể bù được tuyến tính (D_lin). Đây là giả thuyết kiểm tra được.

**10.2 Không có hạng OV "tự nhiên".** Phổ số chiều OV liên tục cho thấy nếu nén attention bằng low-rank $W_{OV}$, một hạng cố định cho mọi head có lẽ không tối ưu; nên dùng **phân bố hạng** (giống cách MOLT, note số 22, dùng hạng 32–512).

**10.3 Induction chiếm phần lớn pattern.** Nếu pattern của nhiều head có thể tái tạo chủ yếu bằng $\mathcal O(1000)$ induction feature, thì với OCR (gần như là copy/induction từ ảnh sang text) phần QK quan trọng có thể tập trung. Khi distill, nên đặc biệt giữ các head "copy/induction".

**10.4 MTC như mô hình student.** MTC là "Attention+MLP hợp nhất" với feature thưa; đó cũng là một dạng mô hình nén. Về lý thuyết có thể distill một layer attention+MLP của teacher thành một MTC-like module. Nhưng tác giả không đánh giá chi phí suy luận, nên đây chỉ là ý tưởng.

**10.5 Hybrid GatedDeltaNet.** Biểu diễn xuyên layer và head loading áp dụng cho các layer full-attention. Với GatedDeltaNet không có pattern softmax rõ ràng, "head loading vector" phải định nghĩa lại (ví dụ trên ma trận trọng số hiệu dụng $\text{decay}\cdot q^\top k$).

---

## 11. Thí nghiệm gợi ý

**Thí nghiệm 1 — Chỉ số superposition theo head.** Trên vài layer full-attention của teacher, huấn luyện Attention Value SAE trên value vector nối của mọi head. Với mỗi feature, tính phân bố chuẩn decoder theo head, lấy entropy làm "số head hiệu dụng". Dự đoán: layer có entropy cao thì cắt head gây D_res lớn hơn. Kiểm tra bằng cắt 25% head ở từng layer rồi đo D_lin/D_res.

**Thí nghiệm 2 — Value SAE hai layer.** Lặp lại so sánh layer $a$, $b$ riêng lẻ với chung $(a,b)$ cho hai layer full-attention của LightOnOCR. Nếu SAE chung tốt hơn rõ, đó là ứng viên tốt để **cắt một trong hai** layer, vì thông tin được chia sẻ.

**Thí nghiệm 3 — Pattern identity và tính toán cục bộ.** Đo tỉ lệ khối lượng attention mỗi head đặt vào token hiện tại (đường chéo) trên dữ liệu OCR. Head chủ yếu nhìn chính nó có vai trò kiểu MLP; thử thay chúng bằng MLP nhỏ hoặc gộp vào MLP kề bên khi pruning.

---

## 12. Tóm tắt công thức và khái niệm

| Khái niệm | Công thức / mô tả | Ý nghĩa |
|---|---|---|
| Regularizer đơn giản | $[L_{\mathrm{nuc}}(W_{QK})L_{\mathrm{nuc}}(W_{OV})]^\beta$, $\tfrac12<\beta<1$ | Ép QK/OV low-rank, tách skip-trigram |
| Proxy nuclear norm | $\lVert W\rVert_*=\min_{W=AB}\tfrac12(\lVert A\rVert_F^2+\lVert B\rVert_F^2)$ | Rẻ khi ma trận đã phân tích |
| MTC | $A^{(i)}=\sum_h c_{i,h}A_h+c_{i,0}I$ | Feature di chuyển bằng tổ hợp head |
| Superposition giữa head | MTC Top-1 kém 2–4× về L0 | Feature trải trên nhiều head |
| QK/OV coupling | MTC sạch hơn SAE + head | Nên học "chuyển gì" và "ở đâu" cùng lúc |
| OV đa chiều | Phân bố lũy thừa kích thước cụm | Không có hạng OV ưa thích |
| Xuyên layer | Value SAE 2 layer: ~½ L0 | Biểu diễn trải qua layer |
| QK diagonalization | $\operatorname{softmax}(\sum_i u_i(q)v_i(k))$ | Tái tạo pattern bằng QK feature |

### Kết luận ngắn

Đây là bản đồ của một bài toán chưa giải xong: attention có superposition cả giữa head lẫn giữa layer, QK và OV gắn kết, và số chiều OV là một phổ liên tục. MTC là công cụ thực dụng cho "chuyển gì, từ đâu", còn "vì sao nhìn vào đó" vẫn mở. Với người làm pruning, bài gợi ra một giả thuyết kiểm tra được: cắt **head** có thể giống cắt neuron (khó phục hồi) hơn là giống cắt layer.

**Đọc bản gốc:** https://transformer-circuits.pub/2025/attention-update/index.html
