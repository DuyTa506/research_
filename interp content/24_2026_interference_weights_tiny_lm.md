# Characterizing interference weights in a tiny language model (2026)

**Tác giả:** Nicholas L. Turner, Jeffrey Wu, Joshua Batson (Anthropic). Công bố ngày 21/08/2026.  
**Nguồn:** https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html  
**Chủ đề:** Interference weights trong transformer một layer đã huấn luyện; virtual weight model (VW model) với sáu họ trọng số; **effectiveness** (Fisher) và **helpfulness** (Δloss khi ablate); pruning virtual weights; giới hạn của basis token/feature.  
**Vị trí trong lộ trình:** Đọc sau note số 23 (*A Toy Model of Interference Weights*, 2025), nơi định nghĩa interference weights trong toy model; cần nền *Toy Models of Superposition* và *Towards Monosemanticity*. Bài này quan trọng vì là lần đầu (theo tác giả) chỉ ra được một interference weight cụ thể trong một transformer đã huấn luyện bằng cách đo tác động lên loss huấn luyện. Phần "pruning" ở đây gần với kinh nghiệm của bạn nhất trong cả loạt interp.

> Ghi chú diễn giải bằng tiếng Việt, không phải bản dịch. Paper prune **virtual weights** của một mô hình mở rộng để phục vụ diễn giải, không phải nén mô hình gốc. Mọi liên hệ với pruning/distillation có cấu trúc của bạn là suy luận của chúng tôi.

## Mục lục

1. [Câu hỏi trung tâm](#1-câu-hỏi-trung-tâm)
2. [Ví dụ động lực: ACETYLCHOLINE](#2-ví-dụ-động-lực-acetylcholine)
3. [Mô hình và virtual weight model](#3-mô-hình-và-virtual-weight-model)
4. [Effectiveness và helpfulness](#4-effectiveness-và-helpfulness)
5. [Effectiveness làm nổi trọng số có ích nhưng không tách hẳn](#5-effectiveness-làm-nổi-trọng-số-có-ích-nhưng-không-tách-hẳn)
6. [Đuôi effectiveness là có ích](#6-đuôi-effectiveness-là-có-ích)
7. [Mô hình vẫn dày đặc trong basis này](#7-mô-hình-vẫn-dày-đặc-trong-basis-này)
8. [Cách paper kiểm chứng](#8-cách-paper-kiểm-chứng)
9. [Liên hệ với các hướng nghiên cứu khác](#9-liên-hệ-với-các-hướng-nghiên-cứu-khác)
10. [Giới hạn](#10-giới-hạn)
11. [Ứng dụng giả thuyết cho pruning và distillation](#11-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
12. [Thí nghiệm gợi ý](#12-thí-nghiệm-gợi-ý)
13. [Tóm tắt công thức và khái niệm](#13-tóm-tắt-công-thức-và-khái-niệm)

---

## 1. Câu hỏi trung tâm

Hy vọng tham vọng của mechanistic interpretability: đọc mạng nơ-ron như đọc chương trình, mô tả **toàn cục** (không phụ thuộc ngữ cảnh). Đơn vị toàn cục tự nhiên nhất là **trọng số**. Hai trở ngại:

1. **Neuron polysemantic** → dùng dictionary learning (SAE, transcoder) để có basis đơn nghĩa hơn; trọng số được viết lại thành **virtual weights** giữa các thành phần.
2. Ngay cả với feature hoàn hảo, feature vẫn được ghi/đọc qua phép chiếu ít chiều (residual stream), nên virtual weights bị ép vào **weight superposition** và sinh ra **interference weights**: tương tác tuyến tính giữa các thành phần dễ hiểu qua residual, nhưng *không liên quan hoặc có hại* cho hành vi. Hai thành phần không bao giờ cùng hoạt động trên dữ liệu vẫn có thể nối bằng một virtual weight lớn, gợi ý sai về chức năng.

Bài dùng **định nghĩa 2** của note 23 (phân loại từng trọng số theo tác động lên loss), không dùng định nghĩa 1 (độ lệch khỏi "trọng số lý tưởng").

**Ba phát hiện chính:**
1. Trọng số có ích và có hại **rải khắp** dải độ lớn virtual weight; đọc mạch từ độ lớn thô có thể bỏ sót mạch quan trọng và làm nổi kết nối vô nghĩa.
2. Trọng số không hiệu quả (ineffective) **rất nhiều và bỏ được dễ dàng**: prune 70% ít hiệu quả nhất chỉ tăng loss 0.01 nats, 85% tốn 0.1. Trọng số hiệu quả nhất gần như toàn là có ích và vượt mọi trọng số có hại cỡ một bậc độ lớn.
3. Số trọng số có ích **vẫn lớn**: nhiều hơn số tham số của transformer gốc, ngay cả khi chỉ giữ những trọng số hiệu quả và có ích nhất. Basis token/feature có lẽ chưa phải basis tốt.

---

## 2. Ví dụ động lực: ACETYLCHOLINE

Mô hình 1L hoàn thành từ viết hoa ACETYLCHOLINE (có trong tiêu đề bài báo viết hoa trong dữ liệu), dự đoán token cuối "E" sau "IN".

Logit của mỗi token = tổng đóng góp theo **ba đường**, mỗi đường được nhân với unembedding (logit lens cho từng đường):

$$
z = W_U\big(\underbrace{x_{\text{tok}}+x_{\text{pos}}}_{\text{direct}}+\underbrace{\textstyle\sum_h \mathrm{Attn}_h}_{\text{attention}}+\underbrace{\mathrm{MLP}}_{\text{MLP}}\big).
$$

| Đường | Token được đẩy mạnh nhất | Vì sao khó hiểu |
|---|---|---|
| Direct | "␣Mrs" | Không liên quan |
| Attention | "ely" | Chữ thường, trong khi mọi token đều viết hoa |
| MLP | "ATION" | ACETYLCHOLINATION không phải thuật ngữ thật |

Không đường nào xếp "E" đứng đầu; "E" là lựa chọn mà *mọi* đường đều cho điểm khá cao. Thêm nữa, virtual weight lớn nhất của đường Tokens→Logits từ "IN" bỏ phiếu cho "utions", một token **không bao giờ** đứng sau "IN" trong tập huấn luyện. Mỗi khi trọng số này tác động lên output, nó chỉ làm loss tệ đi. Không có lý do để nó lớn nếu virtual weights được huấn luyện trực tiếp; nó tồn tại vì weight superposition.

---

## 3. Mô hình và virtual weight model

### 3.1 Transformer

| Thuộc tính | Giá trị |
|---|---|
| Layer | 1, decoder-only |
| $d_m$ (residual) | 256 |
| Attention | 4 head, $d_{head}=64$, softmax causal, scale $1/\sqrt{64}$ |
| MLP | rộng 1024, ReLU |
| Vocab / context | 4.096 token (BPE cắt từ tokenizer Pleias-1.2B) / 1.024 |
| Tham số | ≈2,9M (≈0,79M không tính embedding/unembedding) |
| Normalization, bias | **Không có** |
| Vị trí | Sinusoidal cố định, cộng sau token embedding |
| Dữ liệu | Common Corpus (PleIAs), tiếng Anh và code; ≈9,8×10⁷ token, 11.933 step, một pass |
| Loss | 8,93 → ≈3,33 (vẫn đang giảm) |

**Transcoder** một layer (SLT): 4.096 feature, JumpReLU, đọc residual trước MLP, dự đoán output MLP; loss = MSE + L1 có trọng số theo chuẩn decoder; 100.000 step, batch 16.384.

### 3.2 Basis và sáu họ trọng số

Basis gồm: **token + vị trí** (one-hot ghép, $d_{v'}=4096+1024=5120$), **feature transcoder** ($d_f=4096$), **logit** ($d_v=4096$). Mỗi họ là tích ma trận "co" chiều residual $d_m=256$ lại. Vì không có normalization, mỗi đường là một tích ma trận cố định. Dạng ma trận dưới đây do chúng tôi viết lại từ mô tả ($W_{[E,P]}$ là embedding token và vị trí ghép, $W_{enc}, W_{dec}$ của transcoder):

| Họ | Virtual weight (ký hiệu của chúng tôi) | Kích thước | Số phần tử |
|---|---|---|---|
| Tokens→Logits | $W_UW_{[E,P]}$ | $4096\times5120$ | 20.971.520 |
| Features→Logits | $W_UW_{dec}$ | $4096\times4096$ | 16.777.216 |
| Tokens→Features | $W_{enc}W_{[E,P]}$ | $4096\times5120$ | 20.971.520 |
| Tokens→OV→Logits | $W_UW_O^hW_V^hW_{[E,P]}$, 4 head | $4\times4096\times5120$ | 83.886.080 |
| Tokens→OV→Features | $W_{enc}W_O^hW_V^hW_{[E,P]}$, 4 head | $4\times4096\times5120$ | 83.886.080 |
| QK | $W_{[E,P]}^\top W_Q^{h\top}W_K^hW_{[E,P]}/\sqrt{64}$, 4 head | $4\times5120\times5120$ | 104.857.600 |
| **Tổng** | | | **331.350.016** |

Tổng này khớp chính xác con số 331.350.016 trọng số mà paper báo cáo (≈331M, khoảng 100× so với 2,9M), nên cách viết lại ở trên nhất quán với bài. Năm họ là ánh xạ tuyến tính vào logit hoặc feature; **QK là ánh xạ song tuyến tính duy nhất**, tạo attention pattern điều khiển hai họ OV.

Chi tiết để VW model tái tạo chính xác forward pass:
- **Moved token**: $m_q^{(h)}=\sum_{k\le q}A^{(h)}_{qk}\,t_k$ (áp attention pattern lên vector one-hot [token, vị trí]), để biểu diễn họ OV thành trọng số cố định.
- **Sai số transcoder** $e=\mathrm{MLP}_{out}-\mathrm{SLT}_{pred}$, thêm $W_Ue$ vào logit như hiệu chỉnh từng token, và **không** được chấm điểm.

---

## 4. Effectiveness và helpfulness

Hai câu hỏi cho mỗi trọng số: nó có tác động đáng kể không? Nếu có, nó làm loss giảm hay tăng?

### 4.1 Effectiveness (Fisher)

**Effectiveness**: độ lớn tác động của trọng số lên hàm mô hình tính, đo bằng ước lượng bậc hai của KL giữa output có và không có trọng số. Với $a$ là vector attribution của trọng số lên logit, $p$ là xác suất output:

$$
\boxed{\;\mathrm{fisher}(w)=\tfrac12\,\mathbb E_{x\sim\mathcal D}\big[\mathrm{Var}_p(a)\big]=\tfrac12\,\mathbb E_{x\sim\mathcal D}\big[a^\top F a\big],\qquad F=\operatorname{diag}(p)-pp^\top\;}
$$

Vì sao: với nhiễu logit nhỏ $\delta z$, $\mathrm{KL}(p\,\|\,p')\approx\tfrac12\,\delta z^\top F\,\delta z$, và $F$ là Fisher information của softmax theo logit. $\mathrm{Var}_p(a)=\sum_j p_ja_j^2-(\sum_jp_ja_j)^2$: dịch đều mọi logit (cùng $a_j$) không làm gì. Đơn vị là **nats**, so sánh được giữa sáu họ. Ước lượng trên ≈537M token ($2^{29}$).

**Đường tới logit** ($a$ chỉ có một phần tử $a_j=sw$, $s$ là activation nguồn):

$$
\mathrm{fisher}(w)=\tfrac12\,w^2\,\mathbb E\big[s^2\,p_j(1-p_j)\big].
$$

**Đường tới feature:** nếu trọng số thay đổi activation feature một lượng $\Delta f$, thì $a=v\,\Delta f$ với $v$ là logit effect của feature: $\mathrm{fisher}(w)=\tfrac12\mathbb E[\Delta f^2\,\mathrm{Var}_p(v)]$. Điểm tinh tế: $\Delta f$ được tính **phản thực tế** (counterfactual) qua JumpReLU,

$$
\Delta f=f-\operatorname{JumpReLU}(f_{pre}-sw),
$$

để bắt cả trường hợp feature bật/tắt (cổng lật) do bỏ trọng số; cách tuyến tính hóa sẽ cho attribution 0 với hiệu ứng ức chế và phóng đại hiệu ứng kích thích gần ngưỡng. Phương sai đầy đủ tốt hơn xấp xỉ đường chéo.

**Trọng số QK:** bỏ $sw$ khỏi score tại $k^*$ nhân số hạng softmax với $e^{-sw}$. Với $r=Z'/Z=1-A_{qk^*}(1-e^{-sw})$ và $\alpha=(r-1)/r$:

$$
\Delta A_{qk}=\alpha\,(A_{qk}-\mathbf 1_{k=k^*}),
$$

và attribution lên logit của một lần dùng trọng số QK là $\alpha\big(v_{k^*}-\bar v+W_{dec}(M_q\odot(u_{k^*}-\bar u))\big)$, với $v_k,u_k$ là hiệu ứng logit/feature khi attention vào $k$ bằng 1, $\bar v,\bar u$ là trung bình theo pattern gốc, $M_q$ là mask feature đang bật. Chuyển attention giữa hai vị trí có cùng hiệu ứng thì không có tác động. Xấp xỉ: coi các lần xuất hiện của cùng token là độc lập; bỏ hiệp phương sai giữa feature.

Tất cả tính được trong **một lần quét forward**: phần phụ thuộc $w$ tách khỏi thống kê activation (ví dụ chỉ cần tích lũy $\mathbb E[s^2p_j(1-p_j)]$ rồi nhân $w^2/2$ ở cuối); phần counterfactual chỉ tính ở nơi nguồn khác 0.

### 4.2 Helpfulness

**Helpfulness**: thay đổi loss trung bình khi ablate trọng số. Dương = trọng số có ích (bỏ đi thì loss tăng); âm = có hại. Với họ tới logit có dạng đóng. Tại một vị trí có token đích $t$, ablate $sw$ khỏi logit $j$:

$$
\boxed{\;\mathrm{helpfulness}(w)=\mathbb E_{x\sim\mathcal D}\Big[\log\big(1-p_j(1-e^{-sw})\big)+sw\,\mathbf 1_{j=t}\Big]\;}
$$

**Ví dụ số (của chúng tôi).** $p_j=0.2$, $s=1$, $w=0.5$:
- Effectiveness trên token này: $\tfrac12\cdot0.25\cdot0.2\cdot0.8=0.02$.
- Nếu $j\ne t$ (trọng số đẩy token sai): $\log(1-0.2\cdot(1-e^{-0.5}))=\log(0.9213)\approx-0.082$, nghĩa là ablate thì loss *giảm*, trọng số có hại ở token này.
- Nếu $j=t$: $0.5-0.082=0.418$, có ích.

Cùng một trọng số có thể có ích ở token này và có hại ở token khác. Vì vậy cần **rất nhiều dữ liệu** để xác định dấu helpfulness có ý nghĩa thống kê: tác giả tính trên toàn tập huấn luyện cho các ví dụ, trên 1B token cho mẫu ngẫu nhiên, kèm khoảng tin cậy 95%. Histogram từng batch của một trọng số thường rộng và tập trung gần 0, có khối lượng ở cả hai phía.

**Quan hệ:** trọng số phải effective mới có helpfulness, nhưng effectiveness không nói gì về *dấu*. ERA và TWERA (từ *Circuit Tracing*) có thể xem là proxy của effectiveness; Fisher là cải tiến (ít false negative vì đo cái gì tới được output; ít false positive nhờ xấp xỉ bậc hai xử lý gradient bão hòa), nhưng các metric khá giống nhau, và tác giả **không** nghĩ mài sắc metric là đòn bẩy lớn so với tìm basis tốt hơn.

---

## 5. Effectiveness làm nổi trọng số có ích nhưng không tách hẳn

**Ví dụ 1: "IN" Tokens→Logits.** Trọng số "IN"→"utions" lớn nhất về độ lớn, nhưng effectiveness thấp hơn trọng số hiệu quả nhất từ cùng token khoảng **ba bậc độ lớn**, và có hại. Sắp lại theo effectiveness: "utions" rơi xuống, "E" lên hạng hai ("T" đứng đầu và thật sự hay theo sau "IN" hơn "E"); các trọng số hàng đầu là phần tiếp viết hoa, hợp ngữ âm, khớp với sắp theo helpfulness. Lọc theo ngưỡng effectiveness hoặc helpfulness được xem như **tạo một phiên bản mô hình không có interference**.

Phân bố: virtual weight gần **chuẩn** (median ≈ 1/3 giá trị lớn nhất); Fisher effectiveness gần **log-chuẩn**, trải **10 bậc độ lớn** (median nhỏ hơn max **10.000×**). Effectiveness tăng thì trọng số **tách nhánh** thành có ích và có hại, có ích chiếm phần cực đại. Không có tính hai đỉnh, chỉ là một liên tục.

**Ví dụ 2: feature tiếng Trung.** Virtual weight lớn nhất trỏ tới "。" và byte đầu của ký tự CJK; âm mạnh nhất là token Latin ("␣European"). Ở đây độ lớn thô **đã tốt**; sắp lại gần như không đổi. Virtual weights không vốn dĩ gây hiểu nhầm, nhưng không được đảm bảo làm nổi kết nối có ích nhất.

**Ví dụ 3: feature 3013** (cuối từ trong ngôn ngữ châu Âu, đặc biệt tiếng Pháp). Virtual weight thô dẫn đầu là token chữ Ả Rập, không hiệu quả và có hại. Sắp theo effectiveness đẩy chúng xuống, nhưng **không làm sạch hoàn toàn**: trọng số có hại nhất vẫn xen lẫn trọng số có ích ở vùng effectiveness cao.

**Ví dụ 4: feature 1254** (newline trong khối khai báo Java/TypeScript/C#). Virtual weight Tokens→Features dương lớn nhất đến từ **vị trí**; newline bị chìm, whitespace mang trọng số **âm** mạnh nhất. Đầu vào vị trí từ Tokens→Features và từ Tokens→OV→Features **triệt tiêu nhau** (OLS slope = −0,993, $r^2$ = 0,912, trên 271 ngữ cảnh), tổng chỉ ≈30% ngưỡng hiệu dụng. Thực chất: feature dựa vào **attention** (newline và whitespace nhìn vào các key như "public", ");") để biết khi nào bật, còn đường token trực tiếp **ức chế** trên mọi token trừ newline. Effectiveness chỉ ra nơi ức chế quan trọng hơn kích thích.

Bài học: họ trọng số nhắm vào bên trong (QK, Tokens→Features, Tokens→OV→Features) kém hiệu quả và kém có ích hơn họ nhắm logit, nhưng cùng một kiểu quan hệ giữa độ lớn, effectiveness, helpfulness.

---

## 6. Đuôi effectiveness là có ích

Lấy mẫu 1.111 trọng số từ sáu họ, log-uniform theo hạng effectiveness, ước lượng helpfulness trên 1B token kèm khoảng tin cậy. Ba chế độ trong mỗi họ:

| Chế độ effectiveness | Helpfulness |
|---|---|
| Thấp | Gần 0 |
| Trung bình | Có ích và có hại **lẫn lộn** |
| Cao nhất | **Chỉ có ích** |

Trọng số có ích hiệu quả nhất vượt mọi trọng số có hại **ít nhất một bậc độ lớn** trong mỗi họ. Khoảng cách này lớn và nhất quán hơn nhiều so với khi xếp theo độ lớn virtual weight. Diễn giải của tác giả: mô hình tối thiểu loss có mọi lý do để hướng các trọng số có hệ quả lớn nhất về phía đúng, và ít lý do để "quản lý" những trọng số gần như không chạm tới output.

---

## 7. Mô hình vẫn dày đặc trong basis này

### 7.1 Prune theo effectiveness

Bỏ trọng số từ ít tới nhiều hiệu quả, đo loss trên tập test riêng:

| Sparsity (đã bỏ) | Density (còn lại) | Loss tăng |
|---|---|---|
| 70% | 30% | ≈0,01 nats |
| 85% | 15% | < 0,1 nats |
| 99% | 1% (≈ số tham số transformer gốc) | Suy giảm đáng kể |

Effectiveness tốt hơn độ lớn virtual weight ở mọi density và mọi họ (trừ Features→Logits âm), giống văn liệu pruning.

### 7.2 Ước lượng sàn bằng helpfulness

Mẫu ngẫu nhiên 7.765 trọng số: **47,6%** có helpfulness trung bình dương (12,7% chết, 6.750 khác 0); kể cả chỉ đếm những trọng số mà khoảng tin cậy loại trừ 0 vẫn còn hàng chục phần trăm. Với VW model lớn gấp ~100 lần, vẫn còn **hàng chục triệu** trọng số cần diễn giải cho một mô hình một layer.

Nới lỏng: giữ 90% "khối lượng helpfulness" (tổng helpfulness dương) chỉ cần density ước tính **2,43%**; nhưng 99% khối lượng thì cần **13,6%**. Giữ 2% virtual weights ≈ gấp đôi số tham số gốc. Nếu tất cả số đó dễ hiểu, ta sẽ tiến gần một "upstairs lift" hơn từng thấy, nhưng tính helpfulness cho mọi trọng số là quá đắt.

Một thước đo kích thước hiệu ứng (phụ lục): mô hình đạt $\ln4096\approx8{,}32$ → 3,38 nats trên test, tức ≈4,94 nats/token. Chia đều cho ≈331M trọng số: ≈$1{,}5\times10^{-8}$ nats/token mỗi trọng số; theo họ từ $4{,}7\times10^{-8}$ (QK) tới $2{,}9\times10^{-7}$ (Features→Logits). Phụ lục dùng các mốc này để định nghĩa vùng tương đương thực tế (ROPE) $(-\varepsilon,\varepsilon)$ khi phân loại trọng số dương/âm/xấp xỉ 0.

### 7.3 Kết luận của tác giả

Không đạt được mô hình thưa và dễ hiểu bằng lọc effectiveness, và dựa trên helpfulness, họ nghi **không sơ đồ saliency nào** làm tốt hơn nhiều. Điều này nói về **basis** hơn là về mô hình: cùng một mô hình có thể dày đặc trong basis này nhưng thưa trong basis khác (ví dụ token được phân theo ngôn ngữ hay từ loại). Nhận diện interference là **cần nhưng chưa đủ** để đọc mạch toàn cục; effectiveness/helpfulness là công cụ để *nhận ra* một basis tốt khi tìm thấy: basis tốt là nơi đại đa số trọng số có thể bỏ vì không hiệu quả hoặc có hại.

---

## 8. Cách paper kiểm chứng

| Bằng chứng | Chi tiết |
|---|---|
| Ví dụ được chọn | "IN"→"utions": không bao giờ xảy ra trong dữ liệu → có hại theo định nghĩa |
| Ground truth gần đúng | Helpfulness = Δloss khi ablate, CI 95% trên toàn tập huấn luyện hoặc 1B token |
| Thống kê toàn cục | Mẫu 1.111 (log-uniform theo hạng) và 7.765 (ngẫu nhiên) trọng số |
| Đường cong pruning | Loss trên test shard tách biệt khi bỏ dần theo effectiveness và theo độ lớn |
| Phân tích cơ chế | Feature 1254: hồi quy đầu vào vị trí (slope −0,993) |
| Kiểm tra độ lớn hiệu ứng | ROPE với các mốc ngân sách loss |
| Tính chính xác VW model | Tái tạo đúng forward pass (với số hạng sai số transcoder) |

---

## 9. Liên hệ với các hướng nghiên cứu khác

- **Văn liệu pruning:** magnitude pruning + retraining bỏ khoảng 90% tham số mô hình thị giác mà không giảm độ chính xác; lottery ticket. Gần nhất là **Optimal Brain Damage / Optimal Brain Surgeon**: saliency từ độ cong của loss; tại cực tiểu, độ cong trùng với Fisher information theo kỳ vọng, nên Fisher effectiveness là "hậu duệ trực tiếp", nhưng áp cho virtual weights và tích lũy trên phân phối dữ liệu. Khác biệt then chốt: virtual weight là *tích* các ma trận, ablate một virtual weight **không tương ứng** với một chỉnh sửa tham số gốc.
- **Huấn luyện thưa trọng số** (Gao et al.; Marin-Llobet & Heimersheim; CLT kết nối thưa). Drori cảnh báo mạch thưa có thể không trung thực với tính toán của mô hình.
- **Phân rã tham số** (APD, SPD, Local Loss Landscape Decomposition): học phân rã trong không gian tham số; bài này thì lấy phân rã do kiến trúc cho sẵn và hỏi trọng số nào là chức năng.
- **Singular learning theory** (local learning coefficient): cũng hỏi cấu trúc nào là chức năng, nhưng qua hình học suy biến của loss landscape; Fisher effectiveness nằm trong bức tranh bậc hai cục bộ mà LLC muốn sửa. Tác giả không khẳng định effectiveness phục hồi hình học đó.

---

## 10. Giới hạn

- Một mô hình rất nhỏ (1 layer, 2,9M tham số, không norm), chọn để dùng được công cụ chính xác và đắt nhất.
- Fisher effectiveness và helpfulness **không scale** tốt tới mô hình frontier; cần proxy như ERA/TWERA.
- Helpfulness ablate từng trọng số: bỏ qua tương tác phi tuyến khi bỏ nhiều trọng số.
- Xấp xỉ trong công thức QK (độc lập giữa các lần xuất hiện token, bỏ hiệp phương sai).
- Transcoder còn feature polysemantic; 4 attention head có lẽ cũng polysemantic.
- Ngay cả khi lọc hết interference, vẫn còn cạnh tranh qua softmax và tương tác nhiều đường.
- Kết luận "basis kém" là giả thuyết; chưa có basis tốt hơn để đối chứng.

---

## 11. Ứng dụng giả thuyết cho pruning và distillation

> **Suy luận của chúng tôi, không phải kết quả của paper.**

**11.1 Saliency kiểu Fisher cho đơn vị có cấu trúc.** Công thức $\tfrac12\mathbb E[a^\top Fa]$ áp được cho *bất kỳ* thành phần nào có attribution logit $a$ xác định: một neuron, một head, một layer. Với neuron $i$ ở layer cuối, $a=\text{act}_i\cdot W_U W_{out}[:,i]$; với thành phần sâu hơn thì cần counterfactual hoặc gradient. Ưu điểm so với magnitude hay activation norm: đo bằng nats, so sánh được giữa loại đơn vị khác nhau, và bỏ qua thay đổi logit "đều" (không đổi phân phối). Đây là một proxy rẻ, tính trong một pass, cho câu hỏi "cắt đơn vị này làm KL so với teacher tăng bao nhiêu".

**11.2 "Dày đặc trong basis này" ≈ cắt neuron đụng tường.** Bài cho thấy trong basis token/feature, tối đa chỉ bỏ được tới khoảng 85% trước khi đau, và sàn helpfulness vẫn hàng chục phần trăm. Tương tự, cắt neuron (basis chuẩn của MLP) có thể đụng tường sớm vì chức năng trải trong superposition. Hệ quả cho bạn: map tuyến tính tối ưu ở điểm cắt chính là **đổi basis**; phân rã D_lin/D_res của bạn đo đúng phần "lỗi do basis" so với phần "lỗi thật". Gợi ý: thử cắt theo **hướng** (PCA/whitening của hidden activation MLP, kiểu SliceGPT) thay vì theo neuron; nếu D_res giảm mạnh, đó là bằng chứng mô hình thưa hơn trong basis xoay.

**11.3 Dấu của importance cần nhiều dữ liệu.** Helpfulness của một trọng số thay đổi dấu theo token; cần 1B token để có CI hẹp. Calibration set nhỏ (vài trăm ảnh OCR) cho importance có dấu rất nhiễu. Nên dùng tiêu chí **không âm** (effectiveness, kiểu Fisher) để xếp hạng, và chỉ dùng tiêu chí có dấu khi có dữ liệu lớn.

**11.4 KD truyền cả interference của teacher.** Logit của teacher chứa hiệu ứng của interference weights (những phiếu bầu sai như "utions", dù yếu). Distill bằng KL trên toàn phân phối buộc student bắt chước cả những phiếu đó, trong khi student, với bottleneck hẹp hơn, lại phải tạo thêm interference riêng. Thử nghiệm đáng làm: so KD với nhiệt độ cao (truyền nhiều đuôi phân phối) và nhiệt độ thấp, đo phần lỗi đến từ token "không bao giờ đúng".

**11.5 Đường Tokens→Logits trong OCR.** Trong OCR, đường trực tiếp token→logit (bigram) có thể chứa interference như "IN"→"utions". Khi cắt layer, student dựa nhiều hơn vào đường trực tiếp và có thể bộc lộ các lỗi kiểu "nối đuôi vô nghĩa". Phân tích lỗi CER theo loại (thay ký tự, chèn đuôi từ) có thể cho thấy điều này.

**11.6 Ức chế quan trọng hơn kích thích ở đâu đó.** Feature 1254 bật nhờ attention, còn đường token *ức chế* trên mọi token trừ newline. Cắt đơn vị dựa trên "kích thích dương" có thể bỏ sót các đường ức chế then chốt; tiêu chí Fisher với counterfactual (bắt được cổng tắt) tránh được điều này.

---

## 12. Thí nghiệm gợi ý

**Thí nghiệm 1 — Fisher saliency cho neuron MLP cuối.** Ở 2–3 layer cuối decoder LightOnOCR, với mỗi neuron $i$ tính $\tfrac12\mathbb E[a_i^\top F a_i]$, $a_i=\text{act}_i\,W_UW_{out}[:,i]$ (bỏ qua norm cuối hoặc tuyến tính hóa nó). So ba tiêu chí cắt (magnitude $\lVert W_{out}[:,i]\rVert$, activation norm, Fisher) ở cùng tỉ lệ cắt: đo KL so với teacher, CER, và D_lin/D_res sau khi fit map tuyến tính.

**Thí nghiệm 2 — Kiểm tra "đuôi hiệu quả là có ích".** Lấy 200 neuron theo log-uniform hạng Fisher. Ablate từng neuron, đo Δloss trung bình trên tập OCR lớn kèm CI (bootstrap theo trang). Kiểm tra: neuron Fisher cao có toàn Δloss dương không; vùng giữa có lẫn neuron có hại không. Nếu có neuron có hại đáng kể, cắt chúng *trước* có thể cải thiện student.

**Thí nghiệm 3 — Cắt theo neuron vs theo hướng.** Với một MLP, so (a) cắt 50% neuron theo Fisher, (b) chiếu hidden activation lên 50% hướng PCA lớn nhất rồi gộp phép chiếu vào $W_{in}$, $W_{out}$ (ghi chú: với SwiGLU, phép xoay không đi qua cổng phi tuyến, nên chỉ làm được ở chiều residual hoặc phải xấp xỉ). Nếu (b) tốt hơn rõ, đây là bằng chứng ủng hộ luận điểm "dày đặc trong basis này nhưng thưa hơn trong basis khác".

---

## 13. Tóm tắt công thức và khái niệm

| Khái niệm | Công thức / mô tả | Ý nghĩa |
|---|---|---|
| Interference weight | Tương tác qua residual, không liên quan hoặc có hại | Hệ quả của weight superposition |
| VW model | 6 họ, 331.350.016 trọng số (≈100× gốc) | Basis token/vị trí, feature, logit |
| Effectiveness | $\tfrac12\mathbb E[a^\top Fa]$, $F=\operatorname{diag}(p)-pp^\top$ | Tác động lên output (nats) |
| Effectiveness tới logit | $\tfrac12w^2\mathbb E[s^2p_j(1-p_j)]$ | Một pass, tách $w$ |
| Counterfactual feature | $\Delta f=f-\operatorname{JumpReLU}(f_{pre}-sw)$ | Bắt cổng bật/tắt |
| QK shift | $\Delta A_{qk}=\alpha(A_{qk}-\mathbf 1_{k=k^*})$ | Ảnh hưởng của một score |
| Helpfulness | $\mathbb E[\log(1-p_j(1-e^{-sw}))+sw\,\mathbf 1_{j=t}]$ | Dấu: có ích / có hại |
| Pruning theo effectiveness | 70% → +0,01 nats; 85% → <+0,1 | Bỏ được nhiều trọng số |
| Sàn helpfulness | 47,6% dương; 90% khối lượng ↔ 2,43% density | Vẫn dày đặc |
| Kết luận | Cần basis tốt hơn | Lọc interference cần nhưng chưa đủ |

### Kết luận ngắn

Bài biến interference weights từ một khái niệm toy model thành đối tượng đo được trong một transformer thật: có trọng số lớn mà chỉ làm loss tệ đi. Effectiveness kiểu Fisher làm nổi trọng số có ích ở đuôi, cho phép bỏ 70–85% virtual weights gần như miễn phí, nhưng không tách sạch có ích khỏi có hại, và phần còn lại vẫn dày đặc. Thông điệp với người làm nén: tiêu chí saliency tốt (bậc hai, theo KL) giúp nhiều, nhưng giới hạn thật nằm ở **basis**, và đó cũng là điều phân rã D_lin/D_res của bạn đang đo.

**Đọc bản gốc:** https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html
