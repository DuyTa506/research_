# When Models Manipulate Manifolds: The Geometry of a Counting Task (2025)

**Tác giả:** Wes Gurnee*, Emmanuel Ameisen*, Isaac Kauvar, Julius Tarng, Adam Pearce, Chris Olah, Joshua Batson*‡ (Anthropic; * core contributor, ‡ liên hệ). Đăng ngày 21/10/2025.  
**Nguồn:** https://transformer-circuits.pub/2025/linebreaks/index.html  
**Chủ đề:** Feature manifold, đếm ký tự, xuống dòng (linebreaking) trong văn bản cố định độ rộng, QK "twist", thuật toán phân tán qua nhiều attention head, "visual illusion" cho LLM.  
**Vị trí trong lộ trình:** Nên đọc sau A Mathematical Framework (2021, QK/OV), Toy Models of Superposition (2022) và Attribution Graphs (2025, phương pháp + biology). Đây là case study cơ chế sâu nhất về **cảm nhận bố cục văn bản** (vị trí trong dòng, độ rộng dòng, bảng markdown) — gần nhất với OCR trong cả chuỗi bài.

> Ghi chú học tập diễn giải bằng tiếng Việt, không phải bản dịch. Mọi liên hệ với OCR, pruning và distillation là suy luận mở rộng của chúng tôi, không phải kết quả của bài. Bài nghiên cứu Claude 3.5 Haiku (mô hình text), không phải mô hình vision-language.

## Mục lục

1. [Bài toán và bức tranh tổng quan](#1-bài-toán-và-bức-tranh-tổng-quan)
2. [Thiết lập thí nghiệm](#2-thiết-lập-thí-nghiệm)
3. [Biểu diễn character count: bốn góc nhìn](#3-biểu-diễn-character-count-bốn-góc-nhìn)
4. [Vì sao manifold "gợn sóng" là tối ưu](#4-vì-sao-manifold-gợn-sóng-là-tối-ưu)
5. [Cảm nhận ranh giới dòng: QK xoắn manifold](#5-cảm-nhận-ranh-giới-dòng-qk-xoắn-manifold)
6. [Quyết định xuống dòng: hình học trực giao](#6-quyết-định-xuống-dòng-hình-học-trực-giao)
7. [Thuật toán đếm ký tự phân tán](#7-thuật-toán-đếm-ký-tự-phân-tán)
8. [Visual illusions](#8-visual-illusions)
9. [Các biểu diễn cảm giác khác: bảng markdown, dòng ngắt sớm](#9-các-biểu-diễn-cảm-giác-khác-bảng-markdown-dòng-ngắt-sớm)
10. [Bài học phương pháp: đối ngẫu feature–manifold](#10-bài-học-phương-pháp-đối-ngẫu-featuremanifold)
11. [Cách bài kiểm chứng](#11-cách-bài-kiểm-chứng)
12. [Giới hạn](#12-giới-hạn)
13. [Liên hệ với OCR (suy luận)](#13-liên-hệ-với-ocr-suy-luận)
14. [Ứng dụng giả thuyết cho pruning / distillation](#14-ứng-dụng-giả-thuyết-cho-pruning--distillation)
15. [Thí nghiệm gợi ý](#15-thí-nghiệm-gợi-ý)
16. [Tóm tắt công thức / khái niệm](#16-tóm-tắt-công-thức--khái-niệm)

---

## 1. Bài toán và bức tranh tổng quan

**Câu hỏi:** LLM chỉ "thấy" một dãy số nguyên (token). Vậy làm sao nó biết khi nào phải xuống dòng trong văn bản có giới hạn độ rộng (code, email, bài báo scan, văn bản tòa án)?

Để dự đoán đúng token tiếp theo, mô hình phải ngầm:
1. đếm số ký tự đã có trong dòng hiện tại,
2. suy ra độ rộng dòng của tài liệu (không được nói rõ; mỗi newline cho một cận dưới và một cận trên),
3. so sánh số ký tự còn lại với độ dài từ dự định viết tiếp.

Nếu từ tiếp theo vừa thì viết từ đó, không vừa thì xuống dòng. Ví dụ xuyên suốt là "aluminum prompt" với giới hạn ngầm 50 ký tự.

**Thuật ngữ:**
- **Attribution graph:** đồ thị nhân quả giữa các feature (từ dictionary như transcoder/crosscoder) dẫn tới một dự đoán — một "execution trace" của thuật toán trên một prompt.
- **Feature manifold / multidimensional feature:** một tập điểm liên tục (ở đây là đường cong 1 chiều) trong residual stream, tham số hóa bởi một biến (ví dụ số ký tự). Mỗi feature có thêm chiều độ lớn, nên manifold 1 chiều thực chất là "nón" mọi phép scale của đường cong đó.
- **Crosscoder (WCC – Weakly Causal Crosscoder):** dictionary feature đọc/ghi qua nhiều layer. Mọi feature trong bài lấy từ một WCC 10 triệu feature trên Claude 3.5 Haiku.

**Bức tranh ba bước** (từ attribution graph):

| Bước | Biến được tính | Cơ chế chính |
|---|---|---|
| 1 | Vị trí trong dòng (character count), độ rộng dòng (line width) | Cộng dồn độ dài token qua nhiều attention head |
| 2 | Số ký tự còn lại = line width − character count | "Boundary head" xoắn manifold bằng QK |
| 3 | Từ tiếp theo có vừa không? | Hai đại lượng nằm ở không gian con gần trực giao → tách tuyến tính |

Mỗi bước có **hai cách đọc đối ngẫu**: rời rạc (họ feature kích thích/ức chế nhau) và hình học (manifold bị biến đổi).

---

## 2. Thiết lập thí nghiệm

- Dữ liệu tổng hợp: lấy văn xuôi, xóa hết newline, rồi chèn lại newline ở ranh giới từ gần nhất $\le k$ ký tự, với $k=15,20,\dots,150$.
- Claude 3.5 Haiku thích nghi với mọi $k$, dự đoán newline đúng vị trí với xác suất cao **từ dòng thứ ba** (phụ lục đo trên 200 chuỗi văn xuôi với $k=20,40,\dots,140$).
- **Line character count** tại một token = tổng số ký tự kể từ newline trước, *tính cả* token hiện tại.

---

## 3. Biểu diễn character count: bốn góc nhìn

### 3.1 Linear probe là chưa đủ

Một linear regression trên residual stream sau layer 1 dự đoán character count với $R^2=0.985$. Nhưng điều này **không** có nghĩa mô hình biểu diễn số đếm trên một đường thẳng. Thực tế biểu diễn là đa chiều, và bài nhìn nó qua bốn lăng kính:

| Góc nhìn | Ước lượng của mô hình được tóm tắt bởi | Hữu ích cho |
|---|---|---|
| Sparse crosscoder features | Tập activation $\{f_1(x),\dots,f_{10}(x)\}$ của 10 feature | Định hướng ban đầu |
| Không gian con thấp chiều | Hình chiếu $\pi(x)$ | Can thiệp nhân quả |
| Manifold 1 chiều trong không gian con đó | Điểm gần nhất trên đường cong; $\|\pi(x)\|$ = độ tự tin | Hiểu cách dựng và biến đổi |
| 150 logistic probe | $\mathrm{softmax}(Px)$ | Phân tích ma trận QK/OV từng head |

### 3.2 Họ feature đếm ký tự

Ở layer 1–2 có **10 feature** với tuning curve mượt theo character count, ví dụ trên từ "called" có hai feature bật cho khoảng 35–55 và 45–65. Đặc điểm:
- **Receptive field giãn dần** (dilation): feature cho số đếm lớn bật trên khoảng rộng hơn — giống cảm nhận số lượng trong não sinh học.
- Tương tự **place cell** (tế bào vị trí ở chuột) và **curve detector** trong mô hình thị giác: một biến liên tục được biểu diễn bằng nhiều phần tử rời rạc, mỗi phần tử phụ trách một khoảng giá trị.
- Phụ lục: các dictionary kích thước khác nhau học receptive field rất giống nhau → cách rời rạc hóa này gần như **chính tắc** (canonical); có một ít feature splitting theo line width (ví dụ ba feature cùng khoảng ~20–45 ký tự nhưng khác nhau theo độ rộng dòng).

### 3.3 Manifold liên tục

Thường có hai feature bật cùng lúc, lệch pha, với hồ sơ kích hoạt kiểu sin → gợi ý chúng là "tọa độ cục bộ" của một đường cong.

Cách dựng: lấy trung bình residual stream layer 2 cho từng giá trị character count (150 vector), PCA → **6 thành phần đầu giải thích 95% phương sai**. Gọi đó là *character count subspace*. Trong PC1–3 dữ liệu trông như **xoắn ốc (helix)**; PC4–6 xoắn phức tạp hơn. Tái tạo bằng riêng 10 feature cho đường cong gần giống, có "gấp khúc" nhẹ tại vị trí decoder — giống spline xấp xỉ đường cong mượt: 10 vector rời rạc, nội suy giữa 2–3 feature lân cận, tái tạo được 150 điểm.

### 3.4 Kiểm chứng nhân quả

**Ablation:** zero-ablate không gian con top-$k$ PC (ở một layer sớm) so với không gian con ngẫu nhiên cùng chiều. Kết quả: ảnh hưởng loss lớn **chỉ khi token tiếp theo là newline**.

> Bài lưu ý: nói chung không được coi không gian con của PCA là "dành riêng" cho feature, vì có thể superposition với feature khác; ở đây hợp lý hơn vì subspace này kích hoạt dày đặc (dense), ít bị superposition.

**Intervention (patching theo mean):** đổi số đếm cảm nhận được ở cuối aluminum prompt (thật là 42 ký tự) sang giá trị $c$:

$$
\boxed{a_\text{patched}=a_\text{original}-\mu_\text{original}+\mu_c}
$$

thực hiện ở ba layer sớm liền nhau và hai token cuối (token áp chót đặt sớm hơn 6 ký tự cho nhất quán), cả với toàn vector lẫn chỉ trong không gian 6 chiều. **Can thiệp trên không gian hạng 6 là đủ để thay đổi hành vi xuống dòng.**

### 3.5 Probe và hiện tượng "ringing"

150 logistic probe (bài toán phân loại 150 lớp) sau layer 1 có RMSE = 5 ký tự → biểu diễn có nhiễu nội tại, khớp với receptive field rộng. PCA 150 vector trọng số probe: 6 thành phần giải thích 82%.

Ma trận phản hồi probe × character count thật có một dải chéo (receptive field giãn dần) **và hai dải mờ ngoài đường chéo mỗi bên**: đáp ứng không giảm đơn điệu khi xa đỉnh mà **dội lại**. Đây là **ringing** (thuật ngữ xử lý tín hiệu: dao động quá độ quanh một đỉnh nhọn, như hiện tượng Gibbs). Cosine similarity giữa các mean vector, probe và decoder feature đều có cùng mẫu: lân cận dương, xa hơn âm, xa nữa lại dương.

---

## 4. Vì sao manifold "gợn sóng" là tối ưu

**Lý do cơ bản.** Biểu diễn một đại lượng vô hướng 1…N:
- dùng $N$ chiều trực giao: tốn kém và không bền với nhiễu ước lượng;
- dùng 1 chiều: không đủ biểu đạt (xem mục 5.3);
- → mô hình dùng manifold **chiều nội tại 1** nhúng trong không gian **chiều ngoại tại** $1<d\ll N$, uốn gợn để cân bằng capacity (số chiều) và khả năng phân biệt giá trị kề nhau (độ cong).

**Toy tuyến tính (phụ lục).** Muốn $n$ vector đơn vị trên một vòng tròn rời rạc, mỗi vector giống lân cận và trực giao với điểm xa. Ma trận cosine mong muốn là ma trận **circulant** $X$ của một hàm nhọn $f$. Xấp xỉ hạng-$k$ tốt nhất (theo $L^2$) là cắt phổ riêng:

$$
X\approx \sum_{m=1}^{k}\lambda_m u_m u_m^\top,\qquad \text{vector mới: } \pi_k v_i .
$$

Vì DFT chéo hóa ma trận circulant, trị riêng của $X$ chính là hệ số Fourier của $f$; cắt bớt hệ số Fourier nhỏ → **ringing** trong các hàng của $X$. Nhìn trong 3 chiều đầu, vòng tròn bị gợn sóng.

**Điểm then chốt cho bước sau:** phép hoán vị $\rho: e_i\mapsto e_{i+1}$ giữ nguyên $X$, nên giao hoán với $\pi_k$; hạn chế của nó lên không gian con, $\bar\rho=\pi_k\rho\pi_k$, là một **phép quay tuyến tính trượt đường cong dọc theo chính nó**: $\pi_k v_i\mapsto\pi_k v_{i+1}$. Một embedding cong ngẫu nhiên không có tính chất này, nhưng manifold trong Haiku và cấu trúc Fourier thì có. Fourier giải thích ít hơn PCA tối đa 10% phương sai (khác biệt chủ yếu do Fourier không mô tả được dilation).

**Toy động lực học.** 100 điểm trên mặt cầu 6 chiều, hút 6 láng giềng mỗi bên (khớp RMSE của probe), đẩy mọi điểm khác. Nghiệm cũng có rippling và hai vòng ringing. Thu hẹp vùng hút hoặc tăng số chiều → cong hơn, ringing mạnh hơn; ở giới hạn tiến tới space-filling curve. Trong 3 chiều nghiệm giống **đường may bóng chày**, khớp với cấu trúc màu theo hue, ngày trong năm, năm thế kỷ 20 mà Modell et al. quan sát; ringing tương tự cũng có trong positional embedding của GPT-2 và curve detector của Inception v1.

Theo góc nhìn superposition, ringing chính là **interference weight**: buộc nhiều feature vào ít chiều thì ma trận tương đồng có dải chéo rộng hơn và các dải dội.

Qua các layer, biểu diễn càng **nhọn** hơn (receptive field hẹp lại), tương ứng độ cong cao hơn và ringing rõ hơn.

---

## 5. Cảm nhận ranh giới dòng: QK xoắn manifold

### 5.1 Line width chỉ được biểu diễn trên token newline

Token newline có họ feature đếm riêng, mã hóa **độ rộng dòng trước** (số ký tự giữa hai newline liên tiếp). Bài huấn luyện thêm 150 probe cho line width.

### 5.2 Boundary head "xoắn" manifold

Một attention head (tìm qua attribution graph) có query đọc character count ở token hiện tại, key đọc line width ở newline trước. Đưa probe character count qua $W_Q$ và probe line width qua $W_K$:

$$
\boxed{s(i,k)=\big(W_Q P^{\text{cc}}_i\big)^\top\big(W_K P^{\text{lw}}_k\big)\ \text{cực đại khi}\ k=i+\epsilon}
$$

| Không gian | Căn chỉnh tốt nhất | Cosine cực đại |
|---|---|---|
| Residual stream (identity) | $i=k$ | ~0.25 |
| QK của boundary head | lệch chéo, $i<k$ | ≈ 1 |
| QK của head ngẫu nhiên cùng layer | không có cấu trúc | — |

Nghĩa là: head chú ý mạnh về newline khi character count **sắp chạm** line width → báo hiệu ranh giới đang đến. Thuật toán tổng quát cho mọi độ rộng dòng, và cho cả dấu phân cách khác (dòng trống đôi, `|`), vì QK xử lý offset vị trí độc lập với OV sao chép loại dấu. Ringing tạo ra các tích trong "nhiễu" ngoài chéo, nhưng softmax giúp mô hình bền với chúng.

### 5.3 Nhiều boundary head: "stereoscopic"

Có nhiều boundary head (thường theo bộ ~3 ở nhiều layer) với **offset khác nhau**. Huấn luyện probe cho *characters remaining* $r=k-i$ (giới hạn $r<40$): mỗi head có đường đáp ứng đạt đỉnh ở một khoảng cách khác, chồng lấn và "lát" kín miền giá trị.

- Chỉ dùng head 0 thì không phân biệt được 5 với 17 ký tự còn lại.
- Head 0 biến thiên mạnh ở [0,10] và [15,20], head 1 ở [10,20], head 2 ở [5,15].
- Từng head output gần như 1 chiều; **tổng** là đường cong 2 chiều cách đều (2 PC đầu giải thích 92% phương sai).

Ablation/intervention trên không gian con 2 chiều này cũng có tác động nhân quả chỉ tại newline và điều khiển được dự đoán (completion " aluminum." cần 10 ký tự để vừa).

### 5.4 Vì sao cần thêm chiều và thêm head?

**Tính toán hình học.** Với biểu diễn 1 chiều, phép tuyến tính chỉ là scale + tịnh tiến; so sánh vị trí với line width chỉ cho tích hai giá trị, đơn điệu, không có ngưỡng tự nhiên. Với ≥2 chiều, có thể **quay** manifold để căn với manifold kia rồi dùng tích vô hướng.

**Độ phân giải.** Trên một tia 1 chiều, các vị trí cách nhau $\delta$; muốn phân biệt kề nhau cần $\|v_{42}-v_{41}\|=\delta$ vượt ngưỡng nhiễu. Với 150+ vị trí phải chọn giữa dynamic range khổng lồ ($\|v_{150}\|\gg\|v_1\|$, khó cho transformer, và normalization chỉ cho tối đa $\pi$ khoảng cách góc) hoặc mất độ phân giải. Nhúng cong vào nhiều chiều giữ norm tương đương mà vẫn tách xa nhau.

**Cần nhiều head.** Output một head là tổ hợp tuyến tính các input (qua attention và OV), nên **bị chặn bởi độ cong đã có sẵn trong input**. Khi MLP gần như không đóng góp, để tạo đủ độ cong cần nhiều head, mỗi head góp một phần.

### 5.5 Câu chuyện khám phá

QK attribution trong graph cho thấy key feature hàng đầu là "dòng trước dài 40–60 ký tự" và query feature hàng đầu là "character count hiện tại 35–50". Nhiều feature đếm cùng bật với cường độ khác nhau → nghi ngờ chúng rời rạc hóa một manifold → tìm ra boundary head. Boundary detector feature tương tự **boundary cell** trong não (bật ở khoảng cách nhất định tới tường).

---

## 6. Quyết định xuống dòng: hình học trực giao

Feature có ảnh hưởng lớn nhất trong graph (theo influence lên logit) là một **break predictor** muộn: bật khi từ tiếp theo sẽ làm dòng vượt độ rộng; nó tăng xác suất newline và giảm " aluminum". Hai input hàng đầu: feature "say aluminum" và feature "boundary detecting".

- **Boundary detector** (~1/3 độ sâu) không quan tâm độ dài từ tiếp theo.
- **Break predictor** chỉ bật khi từ tiếp theo dài hơn số ký tự còn lại.
- **Break suppressor** bật khi từ tiếp theo *vừa khít*, giảm xác suất newline.
- Các feature này còn có hiệu ứng logit có hệ thống: giảm các token dài nói chung (nếu đoán sai từ tiếp theo, ít nhất token phải đủ ngắn để vừa dòng).

**Hình học.** Lấy mean activation ở ~90% độ sâu cho mọi cặp (số ký tự còn lại $r$, độ dài token tiếp theo $j$), $1\le r,j\le15$. PCA cho thấy hai đại lượng nằm trên **không gian con gần trực giao**, độ cong vừa phải (dynamic range nhỏ hơn nên đủ thấp chiều). Vì trực giao, tổng $u_r+w_j$ (đã center) cho phép quyết định

$$
\boxed{\text{break}\iff j>r}
$$

bằng **một siêu phẳng tách tuyến tính**. Dùng siêu phẳng này (3 chiều) trên dữ liệu thật đạt **AUC 0.91** (sai số gồm cả sai số classifier và sai số Haiku đoán từ tiếp theo).

Bài không thấy cơ chế tổng quát hơn kiểu "dồn toàn bộ xác suất của mọi từ quá dài sang newline": phân phối dự đoán ở cuối dòng khác hẳn phân phối trên cùng prompt bị xóa newline.

---

## 7. Thuật toán đếm ký tự phân tán

### 7.1 Khởi đầu từ embedding

Trung bình embedding $W_E$ theo độ dài token (1–14 ký tự): 3 PC đầu giải thích 70%, xếp **thành vòng** (PC1–PC2) với thành phần dao động (PC3) — cùng mẫu rippled. Ở layer 0 có feature "từ ngắn/trung bình/dài", chồng lấn và không bao phủ hết (một số token thông dụng có độ dài bị "hấp thụ" vào feature riêng của token).

### 7.2 Phạm vi phân tích

- **Bỏ qua MLP:** output attention ảnh hưởng biểu diễn đếm **gấp 4 lần** MLP.
- **Tập trung layer 0–1:** sau layer 0 probe đã khá chính xác.
- 5 head quan trọng ở layer 0 cho $R^2=0.93$ (dự đoán = argmax của output chiếu lên probe); 11 head ở hai layer đầu cho $0.97$.
- Layer 0: mỗi head ghi gần như theo **một tia**, chính **tổng** tạo ra đường cong. Layer 1: head output đã cong (nhận độ cong từ layer 0) và làm **sắc nét** ước lượng.

### 7.3 Một head đếm như thế nào

**Toy dựng tay.** Head dùng newline trước làm **attention sink** (chỗ "đổ" attention mặc định): sau $j$ token, newline còn $1-\alpha j$ attention, mỗi token khác nhận $\alpha$; output của newline là 0, output mỗi token khác cùng hướng với độ lớn tỉ lệ độ dài token → tổng là một tia có độ dài tỉ lệ character count (chỉ đúng tới $1/\alpha$ token; dùng nhiều head với offset khác nhau để đếm xa hơn).

**Thực tế.**
- *QK:* head $h$ chú ý toàn bộ vào newline trong $s_h$ token đầu (sink size), sau đó trải attention trên receptive field tối đa $r_h$ token. Các head chuyên biệt ở offset khác nhau, và phần lớn thiên về chú ý token dài.
- *OV:* ước lượng heuristic = số token × độ dài trung bình $\mu_c\approx4$ (trung bình ~4.5, trung vị 4), cộng hiệu chỉnh:

$$
\text{chú ý newline}\Rightarrow \approx s_h\,\mu_c;\qquad \text{không chú ý newline}\Rightarrow \approx (s_h+r_h)\,\mu_c;\qquad +\ \text{hiệu chỉnh theo token dài/ngắn}.
$$

Đo bằng tương tác probe qua OV: $P_c^\top W_{OV}E_t$ với $E_t$ là embedding trung bình theo độ dài token.

**Ví dụ L0H1** (chú ý newline ~4 token đầu, sau đó trải trên ~4–8 token trước):

| Trạng thái | Ghi ra |
|---|---|
| Chú ý newline | Tăng hướng 5–20 ký tự, giảm 30–80 (dòng ≤3 token ≈ 15 ký tự) |
| Không chú ý newline | Mặc định ~40 ký tự (~8 token × ~5 ký tự kể cả khoảng trắng) |
| Token được chú ý ngắn (<4) | Tăng 10–35, giảm >40 |
| Token được chú ý dài (≥5) | Ngược lại |
| Trộn | Nội suy tuyến tính |

Layer 1 làm tương tự nhưng dùng thêm ước lượng ban đầu từ layer 0 → tinh chỉnh.

### 7.4 Line width

Dùng thuật toán đếm phân tán tương tự giữa hai newline liên tiếp, với tập head **một phần khác** (cơ chế sink cần sửa khi token hiện tại cũng là newline). Bài **chưa** trả lời line width được tổng hợp ra sao (max của các dòng trước? trung bình trượt?).

---

## 8. Visual illusions

Ý tưởng: dùng hiểu biết cơ chế để tạo "ảo giác thị giác" (như Müller-Lyer, Ponzo) cho LLM.

- Một số head đếm, vốn chú ý từ newline về newline trước, lại chú ý về chuỗi `@@` — dấu phân cách trong git diff, nơi việc bắt đầu đếm từ chỗ khác hợp lý.
- Chèn `@@` vào aluminum prompt (không đổi độ dài dòng): head bị "phân tâm", chú ý cả về `@@`, và **dự đoán newline bị phá vỡ**.
- Thử 180 cặp ký tự (một nửa là ký tự lặp): đa số chỉ ảnh hưởng vừa phải, newline vẫn là top dự đoán (xác suất gốc 0.79); không khác biệt giữa cặp giống/khác ký tự. Một số ít gây phá mạnh, phần lớn liên quan code/dấu phân cách: `>>`, `}}`, `;|`, `||`, `` `, ``, `@@`. Mức phá tương quan với mức head đếm bị "phân tâm".
- Ảo giác tổng quát sang so sánh độ dài **tường minh** (câu hỏi trắc nghiệm, bền với thứ tự lựa chọn).

Bài không khẳng định tương đồng trực tiếp với thị giác người, nhưng gợi ý: **áp dụng sai một prior đã học** (vai trò của `@@`) có thể làm sai lệch ước lượng thuộc tính như độ dài dòng.

---

## 9. Các biểu diễn cảm giác khác: bảng markdown, dòng ngắt sớm

- **Dòng ngắt sớm:** có họ feature cho "dòng trước còn thừa bao nhiêu ký tự khi bị ngắt". Nếu dòng kết thúc sớm $c$ ký tự trước giới hạn, từ đầu dòng sau phải dài ít nhất $c$ (nếu không đã vừa ở dòng trước). Bài nhấn mạnh: chỉ xem dataset example thì dễ gộp nhầm thành "newline feature" chung chung.
- **Bảng markdown:** trên 20 bảng tổng hợp, có họ feature bật trên token `|` chuyên cho **chỉ số hàng** và **chỉ số cột**. Trên bảng lớn hơn, biểu diễn chỉ số hàng/cột cũng có ringing và hình "đường may bóng chày" trong PCA — giống character count.

---

## 10. Bài học phương pháp: đối ngẫu feature–manifold

- **Đối ngẫu:** cùng một biểu diễn có thể mô tả bằng hoạt động của các feature rời rạc (tương tác qua virtual weight của head) hoặc bằng manifold liên tục (tương tác là "xoắn" liên tục). Feature cho tọa độ cục bộ; manifold cho tham số hóa toàn cục.
- **Complexity tax (thuế phức tạp):** dictionary feature "đập vỡ" mô hình thành nhiều mảnh nhỏ. Mô tả đúng nhưng phức tạp không cần thiết; hình học manifold trả bớt thuế này khi có thể tham số hóa tường minh.
- **Unsupervised discovery vẫn cần:** nhóm tác giả từng thử chỉ probe + patch nhưng thất bại — không biết phải phân biệt line width với character count, không ngờ line width chỉ nằm trên newline, và dùng probe hồi quy 1 chiều. Feature chỉ rõ nghĩa sau khi test trên dữ liệu tổng hợp.
- **Chọn tác vụ tự nhiên:** hành vi mô hình làm tốt và phổ biến trong pretraining (ngay cả Pythia 70M cũng có cụm "dự đoán newline" trong top 400 "quanta" của Michaud et al.) cho cơ chế sắc nét hơn. Tầng đầu của LLM có thể xem là **perception**, không chỉ "detokenization".
- **Kêu gọi phương pháp:** cần cách tự động phát hiện manifold và cấu trúc đơn giản hơn.

---

## 11. Cách bài kiểm chứng

| Loại bằng chứng | Ví dụ và số liệu |
|---|---|
| Probe | Linear $R^2=0.985$ (sau layer 1); logistic RMSE 5 |
| PCA / hình học | 6 PC = 95% phương sai mean vector; 6 PC = 82% trọng số probe; 2 PC = 92% cho characters remaining; 3 PC = 70% cho embedding theo độ dài |
| Ablation subspace vs. ngẫu nhiên | Tác động lớn chỉ tại token newline |
| Mean-patching | Hạng 6 đủ để điều khiển xuống dòng; điều khiển được characters remaining |
| Phân tích trọng số QK/OV | Cosine ~0.25 → ≈1 qua QK của boundary head; $P_c^\top W_{OV}E_t$ |
| Đóng góp head | 5 head L0: $R^2=0.93$; 11 head: $0.97$; attention ảnh hưởng gấp 4 lần MLP |
| Bộ phân loại hình học | Siêu phẳng 3 chiều: AUC 0.91 |
| Mô hình lý thuyết | Circulant/Fourier, mô phỏng lực hút–đẩy; Fourier kém PCA tối đa 10% |
| Dự đoán từ cơ chế | Visual illusion với `@@`; 180 cặp ký tự |
| Tính phổ quát | Dictionary nhiều kích thước có receptive field giống nhau; graph tương tự chia sẻ cho Gemma 2 2B, Qwen 3 4B (qua Neuronpedia) |

---

## 12. Giới hạn

Do tác giả nêu:
- Chưa nghiên cứu: xử lý bất định khi đếm, cách tổng hợp line width từ nhiều dòng, tài liệu có độ rộng thay đổi, nhiều token ứng viên có độ dài khác nhau, từ nhiều token, trường hợp đặc biệt (LaTeX `\footnote{}`, markdown link).
- Quyết định xuống dòng cuối cùng chắc còn phức tạp hơn phần tách tuyến tính đã thấy (đặc biệt với output nhiều token); phần ngữ nghĩa chỉ được hiểu qua feature.
- Phương pháp hình học tốn công nghiên cứu viên và chỉ dễ khi biến liên tục đã biết và tham số hóa được.
- Nhãn "độ dài token tiếp theo" dùng token thật, giả định mô hình đoán đúng.

Nhận xét thêm của chúng tôi:
- Một mô hình (Claude 3.5 Haiku), một dictionary chính; MLP bị bỏ qua có chủ đích.
- Font không monospace (đếm pixel thay vì ký tự) chỉ được nhắc tới như khả năng, không được kiểm tra.

---

## 13. Liên hệ với OCR (suy luận)

> **Toàn bộ mục này là suy luận của chúng tôi. Bài không nghiên cứu mô hình vision hay OCR.**

**13.1 Decoder OCR cũng phải "xuống dòng".** Khi LightOnOCR chép một trang có dòng cố định độ rộng (báo scan, văn bản tòa án — chính các nguồn bài nhắc tới), decoder phải phát newline đúng chỗ. Có hai nguồn tín hiệu:
- *Thị giác:* ViT thấy ranh giới dòng trong ảnh.
- *Prior ngôn ngữ:* cơ chế đếm ký tự kiểu bài này, kế thừa từ pretraining phần LM.

Giả thuyết: trong mô hình đầy đủ, tín hiệu thị giác chiếm ưu thế; nhưng khi pruning làm yếu grounding thị giác, decoder có thể "rơi về" prior đếm ký tự — ngắt dòng theo độ rộng trung bình thay vì theo ảnh. Lỗi kiểu này dễ thấy trên dòng ngắn bất thường (cuối đoạn, thơ, danh sách).

**13.2 Ảo giác `@@` ↔ lỗi OCR trên code và bảng.** Bài cho thấy chuỗi delimiter (`||`, `;|`, `>>`, `@@`) làm head đếm "phân tâm". Tài liệu OCR có code, bảng, git diff đầy những chuỗi này. Có thể kiểm tra: lỗi ngắt dòng/gộp dòng của student có tập trung quanh delimiter không.

**13.3 Bảng markdown.** Nếu output OCR là markdown, feature chỉ số hàng/cột trên token `|` (cũng là manifold gợn sóng) là cơ chế chính để giữ cấu trúc bảng. Bảng là nơi student nhỏ hay hỏng nhất trong OCR; manifold hàng/cột là một mục tiêu giám sát cụ thể.

**13.4 Character counting trong decoder OCR có thể khác.** Trong OCR, vị trí ngang có thể đến trực tiếp từ vị trí patch ảnh qua cross/full attention, không cần cộng dồn độ dài token. Nên coi bài này là *giả thuyết đầu* về cách decoder biểu diễn vị trí, cần probe để xác nhận.

---

## 14. Ứng dụng giả thuyết cho pruning / distillation

> **Suy luận của chúng tôi, không phải kết quả của bài.**

**14.1 Đại lượng vô hướng sống trên không gian con nhỏ, dense.** Character count chiếm ~6 chiều, characters remaining ~2 chiều, và chúng dense (ít superposition). Pruning theo chiều ẩn (width) mà làm sụp hạng không gian con này sẽ phá hành vi xuống dòng trong khi loss trung bình gần như không đổi — vì ablation chỉ ảnh hưởng **token newline**. → Phải đo loss *theo loại token* (newline, `|`, token bảng), không chỉ loss trung bình.

**14.2 Liên hệ D_lin / D_res.**
- "Xoắn" của boundary head là **tuyến tính** ($W_Q$, $W_K$). Nếu manifold còn nguyên chiều nhưng bị xoay/biến dạng tuyến tính sau cắt, map tuyến tính tối ưu tại vết cắt có thể sửa → nằm trong $D_\text{lin}$.
- Nhưng phép tuyến tính **không thể tăng số chiều span** của manifold, và bài chỉ ra một head không tạo được độ cong vượt quá input. Nếu cắt bỏ vài head đếm khiến đường cong mất chiều (ví dụ 6 → 3), thông tin phân biệt số đếm kề nhau mất hẳn → $D_\text{res}$, không map tuyến tính nào cứu được.
- Đây là một ví dụ cụ thể cho trực giác chung: hỏng do *mất chiều* khó phục hồi hơn hỏng do *biến dạng*.

**14.3 Head có đóng góp riêng nhỏ vẫn quan trọng.** Mỗi head layer 0 chỉ ghi một "tia" gần 1 chiều; chính tổng mới tạo độ cong. Importance score theo từng head (ablate một head, đo loss) có thể đánh giá thấp chúng, nhất là khi các head khác bù một phần. Nên đánh giá **theo nhóm head** cùng ghi vào một subspace.

**14.4 Neuron vs. layer.** Ở đây MLP đóng góp ít (1/4 so với attention) vào biểu diễn đếm, nên cơ chế này nằm chủ yếu ở attention layer sớm. Đây là một dữ kiện: không phải mọi hành vi định dạng đều nhạy với cắt neuron MLP; với hành vi bố cục, cắt attention/layer sớm có thể nguy hiểm hơn.

**14.5 Hybrid GatedDeltaNet.** Cơ chế đếm dựa trên **attention sink tại newline** + softmax trải đều, và boundary detection dựa trên QK sắc nét có offset. Linear attention có gating/decay có thể đếm theo cách khác (cộng dồn trạng thái), nhưng phép "so khớp có offset rồi softmax lọc nhiễu ringing" có lẽ cần full-attention layer. Giả thuyết: các full-attention layer của decoder là nơi chứa boundary-head tương đương → cẩn trọng khi cắt chúng.

**14.6 KD bảo toàn hình học.** Hidden-state KD MSE toàn chiều có thể bỏ qua một subspace 6 chiều có phương sai nhỏ. Có thể thêm loss nhỏ khớp **ma trận cosine của mean vector theo số đếm** (hoặc theo chỉ số hàng/cột bảng) giữa teacher và student — khớp hình dạng manifold (kể cả ringing) chứ không khớp từng tọa độ.

---

## 15. Thí nghiệm gợi ý

1. **Tìm manifold vị trí trong decoder LightOnOCR.** Trên trang chữ in có độ rộng dòng biết trước, gán mỗi token đầu ra character count trong dòng. Lấy mean residual theo count ở mỗi layer decoder, PCA: có helix/ringing không? Bao nhiêu PC giải thích 95%? Lặp lại cho student 0.3B/0.1B; so sánh hạng hiệu dụng và ma trận cosine (ví dụ bằng CKA).
2. **Ablation subspace có kiểm soát.** Zero-ablate top-$k$ PC của subspace đó vs. $k$ chiều ngẫu nhiên; đo loss tách theo token newline / không newline, và CER/lỗi ngắt dòng. Nếu student đã mất cơ chế, kết quả sẽ khác teacher. Tiếp theo: fit map tuyến tính tối ưu tại vết cắt và xem lỗi ngắt dòng phục hồi được bao nhiêu (đo trực tiếp $D_\text{lin}$ vs. $D_\text{res}$ cho hành vi này).
3. **"Ảo giác" OCR.** Render ảnh văn bản với cùng độ dài dòng nhưng chèn `@@`, `||`, `>>`; so sánh tỉ lệ lỗi ngắt dòng của teacher và student. Nếu student nhạy hơn nhiều, đó là dấu hiệu nó dựa vào prior đếm ký tự hơn là tín hiệu thị giác.

---

## 16. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / mô tả | Ý nghĩa |
|---|---|---|
| Character count | Số ký tự từ newline trước, tính cả token hiện tại | Biến vị trí trong dòng |
| Character count subspace | 6 PC của mean residual (95%) | Nơi biểu diễn sống; hạng 6 đủ để can thiệp |
| Feature manifold | Đường cong 1 chiều nội tại, uốn trong $d\ll N$ chiều | Cân bằng capacity và độ phân giải |
| Ringing | Cosine: dương–âm–dương theo khoảng cách | Hệ quả của cắt phổ / interference |
| Toy circulant | $X$ circulant, cắt top-$k$ eigen = cắt Fourier | Giải thích ringing và phép quay $\bar\rho$ |
| Mean-patching | $a-\mu_\text{orig}+\mu_c$ | Đổi số đếm cảm nhận được |
| QK twist | $(W_QP^{cc}_i)^\top W_KP^{lw}_k$ cực đại tại $k=i+\epsilon$ | Phát hiện sắp hết dòng |
| Nhiều boundary head | Offset khác nhau, tổng thành đường cong 2D | Độ phân giải characters remaining |
| Quyết định | break $\iff j>r$; hai subspace trực giao | Tách tuyến tính, AUC 0.91 |
| Head đếm | sink $s_h$, receptive field $r_h$, $\approx s_h\mu_c$ hoặc $(s_h+r_h)\mu_c$ | Đếm bằng attention sink + OV |
| Phân tán | 1 head không tạo được độ cong vượt input | Cần nhiều head phối hợp |
| Visual illusion | `@@`, `||`, `>>` làm head đếm phân tâm | Dự đoán từ cơ chế, kiểm chứng được |
| Complexity tax | Feature rời rạc → mô tả đúng nhưng phức tạp | Hình học giúp trả bớt thuế |

### Kết luận ngắn

Bài là bằng chứng rõ nhất hiện có rằng LLM tự học một dạng "tri giác" về bố cục văn bản: đếm ký tự trên manifold gợn sóng thấp chiều, so sánh với độ rộng dòng bằng phép xoắn QK, và ra quyết định nhờ hình học trực giao. Hai bài học lớn là đối ngẫu feature–manifold và "thuế phức tạp" của dictionary feature. Với OCR và nén mô hình, bài gợi ý những mục tiêu đo cụ thể: các subspace nhỏ, dense, chỉ quan trọng tại một số token (newline, `|`), được dựng bởi nhiều head phối hợp — dễ bị pruning phá mà loss trung bình không phát hiện.

**Đọc bản gốc:** https://transformer-circuits.pub/2025/linebreaks/index.html
