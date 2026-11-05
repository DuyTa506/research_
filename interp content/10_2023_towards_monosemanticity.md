# Towards Monosemanticity: Decomposing Language Models With Dictionary Learning (2023)

**Tác giả:** Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn (core contributors), Tom Conerly, Nicholas L. Turner, … , Tom Henighan, Chris Olah (Anthropic). Công bố ngày 4/10/2023.  
**Nguồn:** https://transformer-circuits.pub/2023/monosemantic-features/index.html  
**Chủ đề:** Superposition, dictionary learning, sparse autoencoder (SAE), đánh giá feature, feature splitting, universality, "finite state automata" từ feature.  
**Vị trí trong lộ trình:** Nên đọc trước *A Mathematical Framework* (2021) và nhất là *Toy Models of Superposition* (2022), nơi đưa ra giả thuyết superposition. Bài này là bước chuyển từ "giả thuyết trên mô hình đồ chơi" sang "công cụ thực tế": lần đầu tiên tách được feature đơn nghĩa từ một Transformer thật, và nó là nền móng cho *Scaling Monosemanticity* (2024) cùng mọi công trình SAE/transcoder sau này.

> Tài liệu học tập bằng tiếng Việt. Đây là ghi chú diễn giải, không phải bản dịch nguyên văn paper. Những ứng dụng cho pruning/distillation là hướng mở rộng do người viết suy luận, không phải kết quả được paper chứng minh.

## Mục lục

1. [Vấn đề: neuron đa nghĩa và superposition](#1-vấn-đề-neuron-đa-nghĩa-và-superposition)
2. [Vì sao không sửa kiến trúc cho hết đa nghĩa](#2-vì-sao-không-sửa-kiến-trúc-cho-hết-đa-nghĩa)
3. [Kiến trúc SAE và hàm loss](#3-kiến-trúc-sae-và-hàm-loss)
4. [Chi tiết huấn luyện và các mẹo quan trọng](#4-chi-tiết-huấn-luyện-và-các-mẹo-quan-trọng)
5. [Đánh giá một feature: specificity, sensitivity, downstream effect](#5-đánh-giá-một-feature-specificity-sensitivity-downstream-effect)
6. [Phân tích toàn cục: feature có dễ hiểu hơn neuron không](#6-phân-tích-toàn-cục-feature-có-dễ-hiểu-hơn-neuron-không)
7. [Feature splitting](#7-feature-splitting)
8. [Universality](#8-universality)
9. [Finite state automata từ feature](#9-finite-state-automata-từ-feature)
10. [Dead và ultralow-density feature](#10-dead-và-ultralow-density-feature)
11. [Cách paper kiểm chứng tuyên bố](#11-cách-paper-kiểm-chứng-tuyên-bố)
12. [Giới hạn](#12-giới-hạn)
13. [Hướng dẫn từng bước: tự huấn luyện SAE trên mô hình của bạn](#13-hướng-dẫn-từng-bước-tự-huấn-luyện-sae-trên-mô-hình-của-bạn)
14. [Ứng dụng giả thuyết cho pruning và distillation](#14-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
15. [Thí nghiệm gợi ý](#15-thí-nghiệm-gợi-ý)
16. [Tóm tắt công thức và khái niệm](#16-tóm-tắt-công-thức-và-khái-niệm)

---

## 1. Vấn đề: neuron đa nghĩa và superposition

**Câu hỏi trung tâm:** Đơn vị tính toán tự nhiên của mạng (neuron) có phải đơn vị tự nhiên để *con người hiểu* không? Nếu không, có cách nào tìm ra đơn vị tốt hơn?

- **Polysemantic neuron** (neuron đa nghĩa): neuron kích hoạt cho nhiều khái niệm không liên quan. Ví dụ trong paper: một neuron của mô hình 1 layer phản ứng với trích dẫn học thuật, hội thoại tiếng Anh, HTTP request và tiếng Hàn.
- **Monosemantic** (đơn nghĩa): một đơn vị chỉ ứng với một khái niệm rõ ràng.
- **Superposition** (chồng chập): giả thuyết rằng mạng biểu diễn *nhiều feature hơn số neuron*, bằng cách gán cho mỗi feature một tổ hợp tuyến tính của neuron. Điều này khả thi khi feature **thưa** (ít khi cùng bật), giống compressed sensing.

Hệ quả: tập hướng feature tạo thành một **overcomplete basis** (cơ sở dư thừa: nhiều vector hơn số chiều). Bài toán "tìm cơ sở dư thừa sao cho hệ số thưa" chính là **sparse dictionary learning** (học từ điển thưa).

### 1.1 Feature như một phép phân rã

Với vector activation $\mathbf{x}^j\in\mathbb R^{d_{MLP}}$ của điểm dữ liệu $j$:

$$
\boxed{\mathbf{x}^j \approx \mathbf{b} + \sum_i f_i(\mathbf{x}^j)\,\mathbf{d}_i}
$$

- $\mathbf{d}_i$: vector đơn vị, **hướng của feature $i$** (một cột của decoder).
- $f_i(\mathbf{x}^j)\ge 0$: **độ kích hoạt** của feature $i$; hầu hết bằng 0 (thưa).
- Số feature lớn hơn nhiều so với $d_{MLP}$.

Paper giữ lập trường **agnostic**: không khẳng định mô hình "thật sự" được cấu thành từ feature, nhưng kết quả universality (mục 8) gợi ý feature có tồn tại ngoài một mô hình đơn lẻ.

### 1.2 Một phân rã tốt cần gì?

1. Diễn giải được **khi nào** feature bật (kể cả trên dữ liệu tổng hợp, off-distribution).
2. Diễn giải được **tác động downstream** khi thay đổi $f_i$, nhất quán với (1).
3. Các feature giải thích một phần **đáng kể chức năng** của MLP (đo bằng loss).

Điểm tinh tế: SAE chỉ nhìn activation, **không nhìn** tác động downstream. Vì vậy downstream effect là một bằng chứng **độc lập** để kiểm tra feature.

---

## 2. Vì sao không sửa kiến trúc cho hết đa nghĩa

*Toy Models* đề xuất ba hướng: (1) làm mô hình không có superposition, (2) dictionary learning sau huấn luyện, (3) lai. Nhóm tác giả kết luận hướng (1) **không khả thi**: kể cả khi ép activation cực thưa (đến mức 1-hot) để loại superposition, neuron **vẫn đa nghĩa**.

Lý do là cross-entropy thích đa nghĩa. Ví dụ đồ chơi của paper: một neuron nhị phân, 4 feature loại trừ nhau A/B/C/D (mỗi cái 1/4 thời gian), mỗi feature dự đoán đúng một token.

| Chiến lược | Loss |
|---|---|
| Neuron chỉ bật cho A; còn lại đoán đều B/C/D | $\tfrac34\ln 3\approx 0.82$ |
| Neuron bật cho A hoặc B (đoán đều A/B); tắt thì đoán đều C/D | $\ln 2\approx 0.69$ |

Phương án đa nghĩa có loss **thấp hơn**, dù không có superposition. Ngược lại, khi học SAE ta dùng **MSE**, loss này không thưởng cho đa nghĩa theo cách đó; vì vậy ràng buộc thưa có thể ngăn superposition hình thành *bên trong từ điển*.

---

## 3. Kiến trúc SAE và hàm loss

**Sparse autoencoder (SAE)** là một MLP một lớp ẩn, rộng hơn input, huấn luyện để tái tạo activation. Encoder là "phiên bản tuyến tính, amortized" (xấp xỉ một lần truyền) của các thuật toán sparse coding mạnh hơn; decoder chính là **từ điển**.

Ký hiệu: $n$ = số chiều input (ở đây $d_{MLP}=512$), $m\ge n$ = số feature; $W_e\in\mathbb R^{m\times n}$, $W_d\in\mathbb R^{n\times m}$ (cột có chuẩn 1), $\mathbf b_e\in\mathbb R^m$, $\mathbf b_d\in\mathbb R^n$.

$$
\begin{aligned}
\bar{\mathbf x} &= \mathbf x - \mathbf b_d &&\text{(pre-encoder bias)}\\
\mathbf f &= \mathrm{ReLU}(W_e\bar{\mathbf x} + \mathbf b_e) &&\text{(feature activations)}\\
\hat{\mathbf x} &= W_d\mathbf f + \mathbf b_d &&\text{(tái tạo)}
\end{aligned}
$$

$$
\boxed{\mathcal L = \frac{1}{|X|}\sum_{\mathbf x\in X}\Big(\|\mathbf x-\hat{\mathbf x}\|_2^2 + \lambda\|\mathbf f\|_1\Big)}
$$

- Số hạng đầu: **reconstruction** (paper nói MSE lấy *trung bình* theo phần tử vector).
- Số hạng hai: **L1 penalty** (tổng theo feature), $\lambda$ = L1 coefficient, thúc thưa.
- **L0 norm**: số feature khác 0 trên một token, thước đo độ thưa thực tế.

Ba lựa chọn thiết kế đáng nhớ:

1. **Bias trước encoder bị tie với bias decoder** ($-\mathbf b_d$): tương đương trừ một vector cố định khỏi mọi activation. Khởi tạo $\mathbf b_d$ bằng **geometric median** của dữ liệu.
2. **Không tie** $W_e$ với $W_d^\top$: encoder phải *phát hiện* feature (nghiêng đi để giảm nhiễu chéo giữa các feature gần nhau), còn decoder phải *biểu diễn* hướng feature. Hai việc này khác nhau. Ví dụ: ba feature base64 có decoder vector gần nhau, nhưng encoder vector bị "nghiêng ra" để phân biệt chúng.
3. **Cột decoder chuẩn 1**: nếu không, SAE có thể "gian lận" L1 bằng cách làm $f$ nhỏ, $\|\mathbf d\|$ lớn.

Vì sao SAE mà không dùng thuật toán dictionary learning cổ điển? (i) SAE scale được tới hàng tỷ điểm dữ liệu; (ii) thuật toán lặp có thể "quá mạnh", tìm ra feature mà **chính mô hình không đọc được** (compressed sensing chính xác là NP-hard). SAE có kiến trúc giống MLP của LM, nên "mạnh ngang" mô hình.

---

## 4. Chi tiết huấn luyện và các mẹo quan trọng

### 4.1 Mô hình đích

| Thành phần | Giá trị trong paper |
|---|---|
| Transformer | 1 attention block + 1 MLP (ReLU), residual 128, MLP 512 |
| Dữ liệu transformer | The Pile, 100 tỷ token |
| Dữ liệu SAE | MLP activation sau ReLU; 40 triệu context, lấy 250 token mỗi context, trộn lẫn |
| Huấn luyện SAE | Adam, batch 8192, 1 triệu step, ~8 tỷ vector (trên tổng 10 tỷ), lấy mẫu **không lặp lại** |
| Kích thước từ điển | 512 (1×) đến 131.072 (256×); phân tích chính là A/1 (4.096 feature, 8×) |

Hai transformer A và B giống hệt nhau trừ random seed; B dùng để kiểm tra universality. Ký hiệu feature: `A/1/3450` = mô hình A, run từ điển 1, feature số 3450; `A/neurons/489` là neuron.

Vì sao mô hình 1 layer? Ít "true feature" hơn nên từ điển nhỏ có thể phủ hết; huấn luyện rất lâu với chi phí thấp; và tác động của feature lên logit **gần tuyến tính**, dễ kiểm tra.

### 4.2 Các sửa đổi so với autoencoder chuẩn (theo thứ tự quan trọng)

1. **Pre-encoder bias** (đã nêu).
2. **Không tie** encoder và decoder.
3. **Neuron resampling** (hồi sinh neuron chết) giúp tìm thêm feature, loss tổng thấp hơn.
4. **Huấn luyện nhiều step**, kể cả khi loss có vẻ đã phẳng.
5. **Sweep learning rate**: LR thấp hơn (đủ step) cho nhiều feature "thật" hơn; annealing LR không giúp thêm.
6. **Adam và chuẩn hóa decoder**: thay vì chỉ renormalize cột sau mỗi bước, **loại bỏ thành phần gradient song song** với cột decoder trước khi cập nhật, cho loss thấp hơn một chút nhưng thật.

### 4.3 Quy trình resampling

- Tại step 25k, 50k, 75k, 100k: tìm neuron SAE không bật trong 12.500 step trước.
- Tính loss trên 819.200 input ngẫu nhiên; xác suất chọn input $\propto$ (loss)$^2$.
- Với mỗi neuron chết: lấy một input, chuẩn hóa về chuẩn 1, gán làm **decoder vector**; encoder vector = input đó, chuẩn hóa bằng $0.2\times$ chuẩn trung bình của encoder neuron còn sống; encoder bias = 0.
- Reset trạng thái Adam của các tham số bị sửa.

Ý tưởng: gieo feature mới vào chỗ SAE đang tái tạo **tệ nhất**, nhưng để nó chỉ bật yếu nhằm không phá phần còn lại. Nhược điểm tác giả nêu: vẫn gây spike loss; resample quá dày làm huấn luyện phân kỳ.

### 4.4 Chọn hyperparameter bằng proxy

- **Training loss**: chỉ so sánh được khi cùng L1 coefficient.
- **Feature density histogram** (log-scale): số feature sống ngoài cụm ultralow; density nhỏ nhất vẫn tìm được; số feature có density > 1% (nhiều quá thường nghĩa là L1 quá thấp).
- **L0**: tác giả thường nhắm **dưới 10 đến 20**, nghi ngờ lời giải có L0 bằng một phần đáng kể của số chiều activation.
- **Reconstructed NLL**: thay MLP activation bằng output SAE, đo loss tăng. Chuẩn hóa thành "phần loss được giải thích":

$$
\boxed{\text{Loss recovered}=\frac{L_{\text{zero-abl}}-L_{\text{SAE}}}{L_{\text{zero-abl}}-L_{\text{orig}}}}
$$

Tác giả cảnh báo con số này **có xu hướng phóng đại**, vì zero-ablation MLP là baseline rất tệ.

---

## 5. Đánh giá một feature: specificity, sensitivity, downstream effect

Paper chọn bốn feature dễ kiểm chứng (cố ý cherry-pick): **Arabic script** (A/1/3450), **DNA** (A/1/2937), **base64** (A/1/2357), **Hebrew** (A/1/416). Với mỗi feature, họ cố xác lập năm điều:

| Tuyên bố | Ý nghĩa |
|---|---|
| **Specificity** (tính đặc hiệu) | Khi feature bật, ngữ cảnh giả thuyết thường có mặt |
| **Sensitivity** (độ nhạy) | Khi ngữ cảnh có mặt, feature thường bật |
| **Downstream effect** | Feature gây ra hành vi phù hợp |
| **Không phải neuron** | Không trùng với một neuron đơn lẻ |
| **Universal** | Có feature tương tự ở mô hình B |

### 5.1 Computational proxy

Để đo khách quan, họ xây **proxy**: log-likelihood ratio $\log\big(P(s\mid\text{context})/P(s)\big)$.

- $P(s)$: phân phối unigram, $P(s)=\prod_{t\in s}p(t)$.
- base64: mỗi ký tự trong `[a-zA-Z0-9+/]` xác suất $1/64$, ngoài tập là $10^{-10}$. DNA: `[ATCG]` mỗi ký tự $1/4$.
- Arabic/Hebrew: dùng Bayes với Unicode block; tối đa hóa qua các prefix kết thúc ở token hiện tại.

Specificity ưu tiên hơn sensitivity, vì đa nghĩa điển hình là "bật cho thứ không liên quan".

### 5.2 Ví dụ Arabic feature (A/1/3450): các con số

- Văn bản Arabic chỉ chiếm **0,13%** token huấn luyện, nhưng chiếm **81%** token mà feature bật; tỉ lệ đi từ 25% (bật yếu) lên **98%** (activation > 5).
- Tương quan Pearson với proxy: **0,74** trên 40 triệu token.
- Lỗ hổng sensitivity được feature khác lấp: tiền tố "ال" do A/1/3134 xử lý; byte đầu của ký tự bị tách token do A/1/3399 xử lý.
- **Expected value plot**: phân phối activation nhân trọng số theo độ lớn. Phần lớn "khối lượng" activation đến từ văn bản Arabic, nên các false positive ở mức yếu ít quan trọng.

Ba giả thuyết cho vùng activation yếu kém đặc hiệu: proxy không hoàn hảo; mô hình "calibrated" nhưng yếu (activation ~ độ tự tin); SAE quá hẹp nên feature chưa tách được rò rỉ thành activation nhỏ.

### 5.3 Downstream effect: logit weight, ablation, pinned sampling

**Logit weight** (trọng số logit, xấp xỉ tuyến tính kiểu path expansion):

$$
\boxed{\ell_i \approx \mathbf d_i^\top\, W_{down}\,\pi L\,W_{unembed}}
$$

trong đó $\pi$ là phép chiếu bỏ trung bình, $L$ là ma trận đường chéo xấp xỉ scale của LayerNorm. Vì softmax bất biến khi cộng hằng số, họ dịch sao cho median bằng 0. Arabic feature có một mode phụ lớn ở phía phải: đúng các token Arabic và byte `\xd8`, `\xd9`.

**Ablation** (loại bỏ feature):

$$
\mathbf x^j \leftarrow \mathbf x^j - f_i(\mathbf x^j)\,\mathbf d_i
$$

trên mọi token của context, rồi chạy tiếp và đo thay đổi log-likelihood từng token.

**Pinned sampling**: ghim feature ở giá trị cao (ví dụ max quan sát được) và sinh văn bản. Ghim Arabic feature sau prompt `1,2,3,…,10` làm mô hình sinh chữ Arabic; ghim base64 làm sinh base64.

### 5.4 "Không phải neuron"

- Trong top-20 ví dụ của mọi neuron, chỉ một neuron có đúng *một* ví dụ Arabic.
- Hệ số của feature theo neuron basis: ba hệ số lớn nhất đều **âm**, và **27 neuron** có |hệ số| ≥ 0,1.
- Neuron tương quan nhất (A/neurons/489) phản ứng với hỗn hợp nhiều ngôn ngữ. Đây là chiến lược superposition hợp lý vì các ngôn ngữ gần như loại trừ nhau.

Các feature khác: DNA có tương quan **0,8** với proxy nhị phân hóa; base64 chỉ **0,38** (proxy quá rộng, bắt cả hex vốn có feature riêng A/1/3817), neuron gần nhất chỉ tương quan 0,18; Hebrew **0,55**, neuron gần nhất 0,1.

---

## 6. Phân tích toàn cục: feature có dễ hiểu hơn neuron không

Của A/1 (4.096 feature): **168 dead** (không bật trên 100 triệu ví dụ) và **292 ultralow density** (< 1 phần triệu) bị loại khỏi phân tích.

### 6.1 Ba phương pháp

| Phương pháp | Cách làm | Kết quả |
|---|---|---|
| **Human rubric** (blinded) | Lấy mẫu đều trên 11 khoảng activation; chấm 412 interval của 162 feature/neuron; thang tối đa 14 | Median neuron = **0** (không nêu nổi giả thuyết); median feature interval = **12** |
| **Automated interp, activation** | Claude viết giải thích từ ví dụ, rồi dự đoán activation trên ví dụ mới; Spearman trên 60 ví dụ × 9 token = 540 dự đoán | Feature tốt hơn neuron rõ rệt |
| **Automated interp, logit** | Từ giải thích, đoán token nào feature đẩy lên; trộn 50/50 top logit và ngẫu nhiên | **74%** trung bình cho feature, **58%** cho neuron (ngẫu nhiên = 50%) |

**Automated interpretability**: dùng một LLM để viết và chấm giải thích feature (theo Bills et al.).

Rubric con người gồm: độ tự tin (0–3), nhất quán của token activation cao (0–5), nhất quán của logit dương (0–3), có tách biệt độ lớn giữa logit nhất quán/không nhất quán (0–1), độ cụ thể (0–3).

### 6.2 Activation interval

Interval activation cao nhất quán với diễn giải hơn interval thấp. Có feature nhất quán trên toàn phổ, có feature chỉ nhất quán ở ~60% phía trên. Tác giả gợi ý đây có thể là dấu hiệu feature học được **lệch một góc nhỏ** so với feature "thật".

### 6.3 Feature giải thích được bao nhiêu MLP?

- A/1: **79%** loss reduction của MLP được khôi phục (tức thay bằng SAE chỉ tốn 21% so với zero-ablate MLP).
- A/5 (131.072 feature, L1 = 0,004): **94,5%**.

Cẩn trọng: có thể có **đuôi dài** feature, nên % loss có thể gây hiểu lầm; feature cũng chưa hoàn toàn đơn nghĩa.

### 6.4 Feature phản ánh mô hình hay dữ liệu?

Đối chứng: SAE trên transformer có **trọng số bị xáo trộn** (shuffle entries từng ma trận). Kết quả: nhiều feature single-token và một số feature tập con tùy tiện, khó diễn giải. Kết luận: quá trình học tạo cấu trúc giàu hơn thống kê token. Thêm nữa, logit weight và ablation **không phải input** của SAE, nên việc chúng hợp lý là thuộc tính của mô hình.

Lưu ý phụ lục: trên automated interp, feature của mô hình random lại có median cao hơn, vì chúng là single-token "dễ chấm". Khi tách riêng nhóm không single-token, feature A/1 tốt hơn.

---

## 7. Feature splitting

**Feature splitting** (tách feature): khi tăng kích thước từ điển, một feature thô tách thành một "họ" feature tinh hơn. Ví dụ số feature base64: 1 (A/0, 512) → 3 (A/1, 4.096) → nhiều hơn (A/2, 16.384).

Base64 trong A/1 tách thành:

- **A/1/2357**: chữ cái trong base64.
- **A/1/2364**: chữ số trong base64, ít dự đoán chữ số tiếp theo. Lý do là **tokenizer**: hai chữ số liền nhau sẽ gộp thành một token, nên sau một token một-chữ-số thì token tiếp gần như chắc chắn không phải chữ số.
- **A/1/1544**: base64 **mã hóa văn bản ASCII** (manh mối: token `ICAgICAg` = sáu dấu cách).

Hình học: UMAP của decoder vector cho thấy feature tương tự có góc nhỏ. Cấu trúc tách **không phải cây**: feature có thể vừa tách vừa nhập. Ví dụ "the trong văn toán" (A/0/341) tách thành bản cho machine learning (A/2/15021, đẩy "the dataset", "the classifier"), đại số/topo (A/2/4878), trường hấp dẫn (A/2/2609).

Giả thuyết của tác giả: tồn tại một tập "feature lý tưởng" (với từ điển vô hạn), thường thành cụm rất sát nhau trong superposition. Từ điển hữu hạn trả về feature phủ cùng "lãnh thổ" nhưng kém đặc hiệu. Hệ quả:

- "Số feature đúng" ít quan trọng hơn tưởng; từ điển nhỏ cho **bản tóm tắt**, từ điển lớn cho **độ phân giải** cao.
- Giải thích hai "bug": **single-token feature** ở từ điển nhỏ (thực ra là nhiều feature "P trong ngữ cảnh khác nhau" bị gộp) và **nhiều feature cho một ngữ cảnh** (thực ra là tách tinh).
- **Token-in-context feature** rất phổ biến (A/4 có hơn 100 feature cho "the"). Tác giả nghiêng về giả thuyết mô hình *thật sự* dùng **local code** (mã cục bộ) một phần, vì nó cho dự đoán "sắc" hơn tổng tuyến tính "sau the" + "trong vật lý".

---

## 8. Universality

**Universality**: cùng feature hình thành ở nhiều mô hình khác nhau.

Hai cách so feature giữa mô hình:

- **Activation similarity**: tương quan activation trên cùng tập dữ liệu.
- **Logit weight similarity**: tương quan vector logit weight.

Kết quả A/1 so với B/1: median tương quan activation với feature gần nhất là **0,72**, trong khi neuron so neuron chỉ **0,46**. Các feature chi tiết: Arabic 0,91; DNA 0,92; base64 0,85; Hebrew 0,92.

Nhưng logit similarity thường thấp hơn nhiều (Arabic chỉ 0,23) vì phần lớn logit weight là **interference** (nhiễu do superposition). Ca cực đoan: A/1/3949 và B/1/3321 (bật trên `pone` trong trích dẫn PLoS ONE, bật trên 0,02% token) có tương quan activation **0,98** nhưng logit correlation âm; chỉ token "." trùng. Giải pháp: **attribution similarity**, $f_i(t_j)\,v_{i,t_{j+1}}$ (activation × logit weight của token thực sự đến tiếp), xấp xỉ gradient × activation. Nó tương quan cao với activation similarity.

Universality rộng hơn: nhiều feature khớp với neuron trong mô hình SoLU (base64, hex, all-caps), feature của Smith, Gurnee et al., và gợi nhớ "region neuron" của Goh et al. Nhưng không thấy rõ tương tự "emotion neuron".

---

## 9. Finite state automata từ feature

Đây không phải circuit thông thường, mà là chuỗi: feature X đẩy xác suất token T, token T được sinh ra làm feature Y bật ở bước sau, v.v. Mô hình không học chúng "để phối hợp"; chúng nối với nhau **qua dòng token**.

- **Một node**: feature base64 đẩy token base64, token đó lại bật chính nó.
- **Hai node**: ALL_CAPS_SNAKE_CASE (A/0/207 trên chữ hoa, A/0/358 trên `_`). Tương tự với ký tự Unicode bị tách 2 token: tiếng Tamil (`\xe0\xae` + `\xa3`), tiếng Trung (một feature cho "ký tự đầy đủ hoặc hậu tố", một cho "tiền tố" và dự đoán hậu tố).
- **Bốn node HTML**: A/0/20 (mở tag → tên tag) → A/0/0 (tên tag → đóng tag) → A/0/30 (đóng tag → whitespace) → A/0/494 (whitespace → mở tag mới).
- **Ghi nhớ**: ở A/4, một chuỗi feature "ghi nhớ" cụm `MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE` (header license). Feature gần nhị phân và rất hẹp, phù hợp lý thuyết memorization trong superposition.

---

## 10. Dead và ultralow-density feature

- **Feature density**: tỉ lệ token mà feature khác 0. Feature thật thường có density khoảng $10^{-3}$ đến $10^{-6}$.
- **Dead feature**: không bao giờ bật.
- **Ultralow-density cluster**: cụm bên trái của histogram (ví dụ quanh $10^{-7}$ ở run A/16), gần như **không diễn giải được**; cụm "high density" (quanh $10^{-5}$) thì hầu hết diễn giải được.
- Khi hai cụm chồng nhau (từ điển lớn, L1 lớn), có thể tách bằng ba thống kê: density, encoder bias, tích vô hướng encoder·decoder.
- Tác giả cho rằng cụm ultralow là **artifact của huấn luyện SAE**. Huấn luyện lâu hơn sẽ "giết" thêm feature ultralow, nên đếm dead feature thôi là chưa đủ.
- Yếu tố làm tăng dead neuron: LR quá cao, batch quá nhỏ, dữ liệu lặp (quá nhiều token mỗi context hoặc nhiều epoch), quá nhiều step, chọn optimizer.
- Tăng kích thước từ điển vẫn tìm thêm feature **hiếm hơn**: chưa vắt kiệt feature của một MLP chỉ 512 neuron.

---

## 11. Cách paper kiểm chứng tuyên bố

| Loại bằng chứng | Ví dụ, con số |
|---|---|
| Proxy định lượng | Arabic: 81% token active là Arabic, Pearson 0,74; DNA 0,8 |
| So với neuron | 27 neuron có hệ số ≥ 0,1; neuron gần nhất đa nghĩa |
| Causal: ablation | Tắt feature làm giảm likelihood token đúng loại |
| Causal: pinned sampling | Ghim feature thì sinh ra Arabic/base64 |
| Logit weight | Mode phụ đúng nhóm token |
| Human + automated interp | Median 12 so với 0; logit 74% so với 58% |
| Baseline random weights | Feature giàu cấu trúc hơn hẳn |
| Universality | Median 0,72 (feature) so với 0,46 (neuron) |
| Loss recovered | 79% (A/1), 94,5% (A/5) |

---

## 12. Giới hạn

- **Mô hình 1 layer, MLP 512**: tác động lên logit gần tuyến tính, khiến giả thuyết biểu diễn tuyến tính dễ đúng hơn. Tác giả thừa nhận có thể kém tổng quát cho mô hình nhiều layer.
- **Không có metric "vàng"**: dùng tổ hợp proxy; thông tin tổng (MDL) không tương quan với tính dễ hiểu.
- **Feature cherry-pick** ở phần chi tiết; vùng activation thấp kém đặc hiệu.
- **Loss recovered có thể phóng đại**; đuôi dài feature.
- **Chi phí scale**: tác giả ước lượng SAE 100× trên MLP rộng 10.000 có khoảng 20 tỷ tham số, có thể đắt hơn cả mô hình gốc.
- **Chưa xử lý attention superposition**; feature có thể là **manifold đa chiều** chứ không phải hướng 1-D.
- Ultralow-density cluster chưa hiểu rõ; resampling gây spike.
- Replication của Neel Nanda (mô hình GELU 1 layer): kết quả chính lặp lại được, nhưng hơn nửa feature rơi vào cụm ultralow gần như cùng một hướng encoder; decoder vector rất dày đặc trong neuron basis (92% dense); cosine encoder–decoder median chỉ khoảng 0,5, ủng hộ việc không tie.

---

## 13. Hướng dẫn từng bước: tự huấn luyện SAE trên mô hình của bạn

> Phần này tổng hợp công thức của paper (và cải tiến của bài 2024) thành quy trình thực hành. Các con số không có trong paper được đánh dấu **(gợi ý của người viết)**.

### Bước 0: Chọn vị trí đo

- Paper 2023: activation **MLP sau ReLU** của một layer. Bài 2024: **residual stream ở layer giữa** (nhỏ hơn MLP, giảm cross-layer superposition).
- Với LightOnOCR (decoder lai GatedDeltaNet + full attention): bắt đầu với residual stream **ngay tại các điểm bạn định cắt** layer, và MLP hidden của một block mà bạn định cắt neuron **(gợi ý của người viết)**.

### Bước 1: Thu activation

- Chạy mô hình trên dữ liệu **đa dạng, đúng miền** (trang tài liệu và văn bản đầu ra OCR). Paper nhấn mạnh **scale dữ liệu quan trọng**: nhiều dữ liệu hơn cho feature "sắc" hơn.
- Như paper: lấy mẫu một số token mỗi context, **xáo trộn** để một batch đến từ nhiều context, và **không lặp lại** dữ liệu.
- Với VLM: lưu riêng cờ "vision token hay text token" để phân tích sau; chuẩn của hai loại có thể khác nhau **(gợi ý của người viết)**.
- Bỏ token BOS/attention-sink nếu chuẩn của chúng bất thường **(gợi ý của người viết)**.

### Bước 2: Chuẩn hóa

Bài 2024: nhân một hằng số vô hướng sao cho $\mathbb E\|\mathbf x\|_2^2 = D$ (số chiều). Nhờ vậy L1 coefficient có ý nghĩa so sánh được giữa các layer.

### Bước 3: Dựng SAE

```python
import torch, torch.nn as nn

class SAE(nn.Module):
    def __init__(self, n, m):                      # m = k * n, ví dụ k = 8
        super().__init__()
        self.W_e = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(m, n)))
        self.W_d = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(n, m)))
        self.b_e = nn.Parameter(torch.zeros(m))
        self.b_d = nn.Parameter(torch.zeros(n))    # khởi tạo = geometric median của dữ liệu
        with torch.no_grad():
            self.W_d /= self.W_d.norm(dim=0, keepdim=True)   # cột chuẩn 1

    def forward(self, x):
        f = torch.relu((x - self.b_d) @ self.W_e.T + self.b_e)
        return f @ self.W_d.T + self.b_d, f

def sae_loss(x, x_hat, f, lam):
    return ((x - x_hat) ** 2).mean() + lam * f.abs().sum(-1).mean()
```

### Bước 4: Vòng huấn luyện

```python
loss = sae_loss(x, *sae(x), lam); opt.zero_grad(); loss.backward()
with torch.no_grad():                      # bỏ gradient song song với cột decoder
    W, g = sae.W_d, sae.W_d.grad
    g -= (g * W).sum(0, keepdim=True) * W
opt.step()
with torch.no_grad():
    sae.W_d /= sae.W_d.norm(dim=0, keepdim=True)
```

- Adam; **sweep LR** và ưu tiên LR thấp kèm nhiều step.
- Biến thể 2024: bỏ ràng buộc chuẩn 1, thay L1 bằng $\lambda\sum_i f_i\|W_{d,\cdot i}\|_2$.
- Theo dõi số neuron không bật trong N step gần nhất; **resample** định kỳ theo mục 4.3.

### Bước 5: Theo dõi chất lượng (mỗi vài nghìn step)

| Metric | Mục tiêu/diễn giải |
|---|---|
| L0 | Paper 2023 nhắm < 10–20 với MLP 512; bài 2024 (residual Sonnet) < 300. Hãy so tương đối |
| Loss recovered | Thay activation bằng $\hat{\mathbf x}$, đo CE (hoặc CER cho OCR) |
| Variance explained | $1-\mathbb E\|\mathbf x-\hat{\mathbf x}\|^2/\mathrm{Var}(\mathbf x)$ (bài 2024 báo ≥ 65%) |
| Density histogram | Số feature sống ngoài cụm ultralow; ít feature density > 1% |
| Dead features | Đếm trên một mẫu lớn (2024: $10^7$ token) |

Sweep vài $\lambda$ và vài kích thước, vẽ **L0 so với loss recovered**: đây là trade-off chính.

### Bước 6: Diễn giải feature

1. Top-activating examples **và** ví dụ lấy đều trên các khoảng activation (đừng chỉ nhìn top).
2. Với mô hình sâu, logit weight chỉ là xấp xỉ "direct path". Nên kiểm tra bằng **ablation** ($\mathbf x\leftarrow\mathbf x-f_i\mathbf d_i$) và **clamp/steering**.
3. Đối chứng: chạy cùng quy trình cho **neuron** và cho SAE trên **mô hình trọng số xáo trộn**.
4. Huấn luyện SAE thứ hai (seed khác), ghép feature bằng tương quan activation để kiểm tra độ ổn định.
5. Dùng một LLM để gán nhãn tự động, nhưng tự kiểm tra một mẫu ngẫu nhiên.

---

## 14. Ứng dụng giả thuyết cho pruning và distillation

> **Phần này là suy luận của người viết, không phải kết quả của paper.**

### 14.1 Vì sao cắt neuron MLP khó phục hồi hơn cắt layer?

Paper cho thấy feature **trải trên nhiều neuron** (Arabic feature: 27 neuron có hệ số ≥ 0,1; replication của Nanda: 92% decoder vector dày đặc). Nếu MLP của bạn cũng vậy, cắt một neuron không xóa một khái niệm, mà **xóa một phần nhỏ của rất nhiều feature cùng lúc** và phá cân bằng interference giữa chúng. Đọc ra feature cần ReLU/ngưỡng (encoder không tuyến tính), nên sai lệch nhỏ đồng loạt có thể chuyển feature từ "bật" sang "tắt": đây là thành phần **phi tuyến**, khớp với việc bạn thấy $D_{res}$ lớn khi cắt neuron. Ngược lại, cắt layer bỏ đi một $\Delta x$ cộng vào residual stream; phần tương quan tuyến tính với các layer còn lại có thể bù bằng một linear map ($D_{lin}$).

### 14.2 Tách $D_{res}$ theo feature

Khi residual dimension không đổi (cắt layer hoặc cắt neuron MLP), có thể áp **SAE của teacher** lên hidden state của student tại cùng vị trí, sau linear map tối ưu $A$:

$$
\Delta_i = \mathbb E\big[(f_i(h_T)-f_i(Ah_S))^2\big]
$$

Feature có $\Delta_i$ lớn là phần $D_{res}$ "có tên". Dự đoán cần kiểm chứng: feature **hiếm** (density thấp) và **token-in-context** bị mất trước, vì chúng nằm trong superposition chặt nhất.

### 14.3 Feature-level KD

Thay vì chỉ MSE trên hidden state, thêm loss khớp $f(h_T)$ và $f(Ah_S)$ (hoặc chỉ các feature quan trọng cho OCR). Ưu điểm có thể có: MSE trên hidden state bị chi phối bởi các hướng phương sai lớn, còn feature thưa ít phương sai nhưng quan trọng (ví dụ ký tự hiếm) lại được trọng số rõ ràng.

### 14.4 Liên hệ OCR

Các automata "tiền tố và hậu tố byte" (Tamil, tiếng Trung) gợi ý: chữ tiếng Việt có dấu, nếu bị tokenizer tách nhiều byte, có thể cần những cặp feature tương tự. Pruning làm hỏng một node có thể gây lỗi dấu có hệ thống. Feature splitting cũng gợi ý student nhỏ có thể "gộp" feature (như từ điển nhỏ), mất phân biệt tinh (ví dụ `ơ` và `ở`).

### 14.5 GatedDeltaNet

Paper chỉ xử lý MLP; attention superposition chưa giải. SAE trên residual stream không phụ thuộc loại token mixer, nên vẫn dùng được cho kiến trúc lai, nhưng nó không cho biết feature được tạo ra bởi khối linear-attention hay full-attention.

---

## 15. Thí nghiệm gợi ý

**Thí nghiệm 1: Feature survival sau pruning.** Huấn luyện SAE (8×) trên residual stream của teacher tại một điểm cắt. Với student đã cắt layer và student đã cắt neuron (cùng mức nén), fit linear map $A$, tính $\Delta_i$ cho mọi feature. So sánh phân phối $\Delta_i$ theo density và theo loại feature (script, layout, token-in-context). Câu hỏi: cắt neuron có làm hỏng **nhiều feature một chút** trong khi cắt layer làm hỏng **ít feature nhiều**?

**Thí nghiệm 2: Neuron "đa feature" và độ khó phục hồi.** Huấn luyện SAE trên MLP hidden của một block. Với mỗi neuron $k$, đếm số feature có $|W_d[k,i]|>\tau$. Kiểm tra tương quan giữa con số này và damage (sau KD) khi cắt neuron $k$. Thêm baseline: cắt theo norm hoặc theo activation.

**Thí nghiệm 3: Automata dấu tiếng Việt.** Tìm feature bật trên byte-token của ký tự có dấu; xác định các cặp prefix/suffix; ablate từng feature và đo CER theo loại dấu. Lặp lại trên student để xem automata có còn nguyên.

---

## 16. Tóm tắt công thức và khái niệm

| Khái niệm | Công thức hoặc định nghĩa | Ý nghĩa |
|---|---|---|
| Phân rã feature | $\mathbf x\approx\mathbf b+\sum_i f_i(\mathbf x)\mathbf d_i$ | Activation = tổng thưa các hướng |
| Encoder | $\mathbf f=\mathrm{ReLU}(W_e(\mathbf x-\mathbf b_d)+\mathbf b_e)$ | Phát hiện feature |
| Decoder | $\hat{\mathbf x}=W_d\mathbf f+\mathbf b_d$, cột chuẩn 1 | Từ điển hướng feature |
| Loss | $\|\mathbf x-\hat{\mathbf x}\|^2+\lambda\|\mathbf f\|_1$ | Tái tạo + thưa |
| L0 | $\#\{i: f_i>0\}$ | Độ thưa thực tế |
| Loss recovered | $(L_{0}-L_{SAE})/(L_{0}-L_{orig})$ | % chức năng MLP được giải thích |
| Logit weight | $\mathbf d_i^\top W_{down}\pi L W_U$ | Tác động trực tiếp lên logit |
| Ablation | $\mathbf x\leftarrow\mathbf x-f_i\mathbf d_i$ | Kiểm tra nhân quả |
| Attribution sim. | $f_i(t_j)\,v_{i,t_{j+1}}$ | So feature giữa mô hình |
| Specificity / sensitivity | P(ngữ cảnh \| feature bật) / P(feature bật \| ngữ cảnh) | Đánh giá diễn giải |
| Feature splitting | 1 → 3 → nhiều feature base64 | Độ phân giải tăng theo kích thước |
| Universality | median corr 0,72 (feature) so với 0,46 (neuron) | Feature lặp lại giữa seed |
| Dead / ultralow | 168 / 292 trên 4.096 (A/1) | Artifact huấn luyện |

### Kết luận ngắn

Paper chứng minh rằng một SAE đơn giản (ReLU, MSE + L1, không tie, resampling) có thể tách MLP 512 neuron thành hàng nghìn đến hàng chục nghìn feature **đơn nghĩa hơn neuron rõ rệt**, có tác động nhân quả hợp lý, lặp lại giữa các seed, và tách tinh dần khi tăng kích thước. Đây là bằng chứng mạnh nhất (tính đến thời điểm đó) cho giả thuyết superposition trên mô hình thật. Với người làm nén mô hình, bài học chính là: **đơn vị bạn cắt (neuron) không phải đơn vị mô hình dùng để biểu diễn (feature)**. Đó là một giả thuyết hợp lý cho vì sao cắt neuron để lại phần damage phi tuyến khó bù.

**Đọc bản gốc:** https://transformer-circuits.pub/2023/monosemantic-features/index.html
