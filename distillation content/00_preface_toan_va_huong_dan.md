# Preface: Toán nền tảng và hướng dẫn đọc về Knowledge Distillation

**Tác giả:** ghi chú tổng hợp từ kiến thức nền của người biên soạn (không phải ghi chú một paper cụ thể)  
**Venue / arXiv:** không áp dụng  
**Nguồn:** các bài 01–47 trong thư mục này; bản đồ phân loại lấy từ bài 05 (Gou et al., arXiv:2006.05525)  
**Chủ đề:** KD từ con số 0: ý tưởng, phân loại, toán (nhiệt độ, gradient, $T^2$, forward/reverse KL, feature loss, exposure bias, ước lượng top-k), cạm bẫy thực hành, thuật ngữ, cách đọc paper, lịch đọc.  
**Vị trí trong lộ trình:** Bài số 00, đọc trước tất cả. Sau đó đi theo 01 → 05 để có nền tảng, rồi 06–14 (KD cổ điển), 15–20 (NLP/Transformer), 21–26 (thị giác và VLM), 27–47 (LLM).

> Tài liệu học tập bằng tiếng Việt, viết từ kiến thức chung của người biên soạn, **không** là ghi chú của một paper và chưa được đối chiếu từng câu với nguồn. Các con số minh hoạ tự tính (có kiểm tra bằng tay). Các nhận định về dự án là suy luận của chúng ta. Khi một paper trong thư mục nói khác, hãy tin paper và kiểm tra lại.

## Mục lục

1. [KD là gì và tại sao hoạt động](#1-kd-là-gì-và-tại-sao-hoạt-động)
2. [Phân loại KD](#2-phân-loại-kd)
3. [Toán cốt lõi](#3-toán-cốt-lõi)
4. [Cạm bẫy thực hành](#4-cạm-bẫy-thực-hành)
5. [Bảng thuật ngữ](#5-bảng-thuật-ngữ)
6. [Cách đọc paper KD một cách phản biện](#6-cách-đọc-paper-kd-một-cách-phản-biện)
7. [Lịch đọc gợi ý](#7-lịch-đọc-gợi-ý)
8. [Liên hệ với dự án (suy luận)](#8-liên-hệ-với-dự-án-suy-luận)
9. [Năm câu tự kiểm tra](#9-năm-câu-tự-kiểm-tra)

---

## 1. KD là gì và tại sao hoạt động

### 1.1 Định nghĩa

**Knowledge distillation (KD, chưng cất tri thức):** huấn luyện một mô hình nhỏ (student) để bắt chước một mô hình lớn hoặc mạnh hơn (teacher), thay vì (hoặc ngoài việc) học trực tiếp từ nhãn gốc. Mục tiêu thường là: giữ phần lớn chất lượng của teacher với chi phí suy luận nhỏ hơn nhiều.

Khác biệt cần nhớ:
- **Pruning** (cắt tỉa): sửa trực tiếp một mô hình có sẵn bằng cách bỏ phần của nó. Sau khi cắt, mô hình thường kém đi và cần "phục hồi".
- **KD:** là cách *huấn luyện* (một hàm mất mát và nguồn tín hiệu), không phải một kiến trúc. Student có thể là mô hình khởi tạo ngẫu nhiên, một mô hình nhỏ có sẵn, hoặc **một mô hình đã bị cắt tỉa**.
- Hai thứ thường đi cùng nhau: cắt tỉa tạo student, KD phục hồi chất lượng.

Bài 01 (Ba & Caruana 2014) cho thấy mạng nông bắt chước mạng sâu tốt hơn mạng nông học từ nhãn; bài 02 (Hinton et al. 2015) đặt tên "distillation" và chuẩn hoá công thức. Nguồn gốc xa hơn là "Model Compression" của Buciluǎ, Caruana, Niculescu-Mizil (2006), theo các bài 01 và 02 trích dẫn.

### 1.2 Vì sao nhãn của teacher giàu thông tin hơn nhãn gốc

Giả sử bài toán 3 lớp: mèo, chó, xe tải. Ảnh một con chó, nhãn gốc là one-hot $[0,1,0]$. Teacher có thể cho $[0.09, 0.90, 0.01]$. Thông tin thêm: "con chó này hơi giống mèo, gần như không giống xe tải". Nhãn one-hot không nói điều đó. Hinton gọi cấu trúc tương đồng này (ẩn trong xác suất nhỏ của các lớp sai) là **dark knowledge** (tri thức tối). Bài 02 dùng ví dụ: xe BMW có khả năng bị nhầm với xe tải rác cao hơn rất nhiều so với bị nhầm với củ cà rốt.

Bốn cách nhìn bổ sung, đều xuất hiện trong các bài đầu của lộ trình:

1. **Thông tin mỗi mẫu cao hơn.** Một nhãn one-hot có $\log_2 K$ bit tối đa, còn phân phối mềm mang nhiều bit hơn. Bài 02: soft target entropy cao cho nhiều thông tin mỗi mẫu và phương sai gradient thấp hơn, nên student học được với ít dữ liệu hơn (ví dụ trong bài 02: 3% dữ liệu, soft target cho 57.0% so với 44.5% nếu dùng nhãn cứng).
2. **Regularizer.** Soft target ngăn student quá tự tin và overfit; bài 05 liên hệ với label smoothing. Bài 01 xem đây là regularization: mạng nông học nhãn gốc overfit sớm, học từ teacher thì không.
3. **Làm sạch nhãn / làm mượt bài toán.** Teacher có thể "kiểm duyệt" nhãn sai và làm phẳng vùng khó học (bài 01, mục 5.1).
4. **Privileged information (thông tin đặc quyền).** Teacher có thể được huấn luyện với thông tin mà student không có lúc suy luận (đầu vào lớn hơn, nhiều modality, ensemble, thời gian tính toán không giới hạn). Student học "đường tắt" về cách tổng quát hoá. Bài 05 nhắc đến khung "learning using privileged information" (Vapnik, Lopez-Paz).

### 1.3 Hai nguồn lợi khác nhau cần tách bạch

- **Lợi ích tối ưu hoá/regularization:** ngay cả teacher *cùng kích thước* với student cũng có thể giúp (Born-again networks, bài 06).
- **Lợi ích chuyển giao năng lực:** teacher mạnh hơn có thêm kiến thức mà student không thể tự học từ dữ liệu có hạn.

Khi đọc kết quả, hãy hỏi: lợi ích đến từ nguồn nào? Bài 10 (Does KD really work?) và 09 (Efficacy of KD) đặt ra đúng câu hỏi này.

### 1.4 Khi nào KD không giúp

Khi student quá nhỏ so với teacher (capacity gap, bài 08, 41), khi teacher quá tự tin nên soft target gần như one-hot (không còn dark knowledge), khi teacher sai có hệ thống, hoặc khi student có thể đạt chất lượng tương đương bằng cách huấn luyện lâu hơn trên dữ liệu thật.

---

## 2. Phân loại KD

Nhiều trục độc lập. Một bài báo thường chọn một giá trị trên mỗi trục.

### 2.1 Theo loại tri thức (bài 05, §2)

| Loại | Student bắt chước gì | Ví dụ bài |
|---|---|---|
| **Response / logit-based** | đầu ra cuối (logit hoặc xác suất) | 01, 02, 14, 15, 28–34 |
| **Feature / hidden-state** | kích hoạt trung gian, có thể qua projector | 03 FitNets, 04 AT, 17, 18, 19, 23, 26 |
| **Relation-based** | quan hệ giữa mẫu hoặc giữa lớp (khoảng cách, góc, Gram matrix) | 12 RKD, 13 CRD |

### 2.2 Theo sơ đồ huấn luyện (bài 05, §3)

- **Offline:** teacher huấn luyện xong, đóng băng, rồi dạy student. Phổ biến nhất, và là bối cảnh của dự án.
- **Online:** teacher và student (hoặc nhiều mạng) cập nhật cùng lúc, ví dụ deep mutual learning (bài 07).
- **Self-distillation:** teacher và student là cùng kiến trúc hoặc cùng một mạng (bài 06 Born-again; bài 22 DINO là một biến thể khác trong học tự giám sát).

### 2.3 Theo đơn vị tín hiệu trong mô hình sinh (LLM)

- **Token-level (word-level):** ở mỗi vị trí chuỗi, so khớp phân phối token tiếp theo của teacher và student trên toàn từ vựng. Tín hiệu dày (nhiều thông tin mỗi vị trí).
- **Sequence-level:** teacher sinh cả chuỗi (ví dụ beam search hoặc lấy mẫu), student huấn luyện bằng cross-entropy trên chuỗi đó như dữ liệu thật (bài 15, Kim & Rush; về sau là nền của 35, 36, 37, 38).

### 2.4 Theo nguồn của chuỗi đầu vào: off-policy vs on-policy

Ngôn ngữ mượn từ học tăng cường/imitation learning.

- **Off-policy (hay "supervised KD"):** student học trên dữ liệu có sẵn (chuỗi từ tập huấn luyện hoặc do teacher sinh). Student không bao giờ thấy chuỗi do chính nó sinh ra.
- **On-policy:** student tự sinh chuỗi, teacher chấm điểm (cho phân phối token) trên chính các tiền tố của student. Giảm lệch phân phối giữa lúc huấn luyện và lúc suy luận. Đại diện: MiniLLM (bài 28), GKD (bài 29); bài 32 (Speculative KD) và 30 (DistiLLM) đề xuất cách pha trộn hiệu quả; bài 47 ShortOPD cũng thuộc hướng này.

### 2.5 Data-level distillation (chưng cất ở mức dữ liệu)

Teacher sinh dữ liệu huấn luyện (câu trả lời, giải thích, chuỗi suy luận), student fine-tune như bình thường (SFT). Không cần truy cập logit của teacher, nên dùng được với API đóng. Đại diện: Self-Instruct (35), Distilling step-by-step (36), Orca (37), distill từ DeepSeek-R1 (38), Qwen3 (44, phần strong-to-weak). Hạn chế: tín hiệu thưa hơn (chỉ thấy token được chọn, không thấy phân phối).

### 2.6 Các trục khác (ngắn gọn)

Multi-teacher, cross-modal (CLIP-KD bài 24, LLaVA-KD bài 26), data-free, quantized, lifelong (bài 05, §5).

---

## 3. Toán cốt lõi

Ký hiệu: $K$ là số lớp (hoặc cỡ từ vựng $V$), $z\in\mathbb R^K$ là logit student, $v\in\mathbb R^K$ là logit teacher, $q$ và $p$ lần lượt là phân phối student và teacher.

### 3.1 Softmax có nhiệt độ

$$
\boxed{q_i^{(T)}=\frac{\exp(z_i/T)}{\sum_{j}\exp(z_j/T)}}
$$

- $T=1$: softmax bình thường. $T\to\infty$: phân phối đều. $T\to 0^+$: gần one-hot (argmax).
- Softmax bất biến khi cộng hằng số vào mọi logit: $z\to z+c\mathbf 1$ cho cùng xác suất. Vì vậy logit thường được "zero-mean" khi phân tích (bài 02 làm vậy).

**Ví dụ.** Logit teacher $v=[10,5,0]$.

| $T$ | $v/T$ | xác suất $p$ |
|---|---|---|
| 1 | $[10,5,0]$ | $[0.9933,\ 0.0066,\ 0.00005]$ |
| 5 | $[2,1,0]$ | $[0.665,\ 0.245,\ 0.090]$ |

Ở $T=1$ student gần như chỉ thấy "lớp 1"; ở $T=5$ student thấy thêm "lớp 2 gần hơn lớp 3". Nhiệt độ là cách *phóng đại* dark knowledge.

### 3.2 Hàm mất mát KD kinh điển

$$
\boxed{\mathcal L=\alpha\,T^2\,\mathrm{KL}\big(p^{(T)}\,\|\,q^{(T)}\big)+(1-\alpha)\,\mathrm{CE}\big(y,\,q^{(1)}\big)}
$$

Số hạng thứ hai là loss nhãn thật thông thường (ở $T=1$). Bài 02 dùng cross-entropy với soft target thay cho KL; chúng khác nhau một hằng số không phụ thuộc student nên gradient như nhau (xem 3.5). Khi suy luận dùng $T=1$.

### 3.3 Gradient và hệ số $T^2$

Đặt $C=\mathrm{CE}(p^{(T)},q^{(T)})=-\sum_i p_i\log q_i$ với $q=\mathrm{softmax}(z/T)$. Dùng đạo hàm softmax chuẩn:

$$
\frac{\partial C}{\partial z_i}=\frac1T\,(q_i-p_i).
$$

Hai điều cần nhớ:

1. **Hệ số $1/T$ từ quy tắc chuỗi** (do chia $z$ cho $T$).
2. Khi $T$ lớn so với độ lớn logit, khai triển bậc nhất $e^{x}\approx1+x$ và giả sử logit zero-mean ($\sum_j z_j=\sum_j v_j=0$):

$$
\boxed{\frac{\partial C}{\partial z_i}\approx\frac{1}{N\,T^2}\,(z_i-v_i)}
$$

với $N$ là số lớp (đây là Eq. 4 của bài 02). Nghĩa là ở $T$ cao, KD tương đương hồi quy logit với L2 (đúng cách của bài 01) và độ lớn gradient co theo $1/T^2$. Nhân loss với $T^2$ bù lại để tỉ lệ giữa số hạng soft và hard không đổi khi đổi $T$. Đây là lý do có $T^2$.

Ở $T$ thấp hơn, các logit rất âm so với trung bình ít được quan tâm hơn. Bài 02 nói điều này có thể có lợi (logit thấp bị nhiễu) hoặc có hại, và là câu hỏi thực nghiệm.

**Kiểm tra bằng số nhỏ (tự tính).** $v=[2,0,-2]$, $T=10$, student $z=[0,0,0]$ nên $q=[1/3,1/3,1/3]$. Teacher: $p=\mathrm{softmax}([0.2,0,-0.2])\approx[0.4018,\,0.3289,\,0.2693]$. Gradient chính xác tại $i=1$: $\frac1{10}(0.3333-0.4018)\approx-0.00685$. Xấp xỉ $\frac{1}{NT^2}(z_1-v_1)=\frac{1}{3\cdot100}(0-2)\approx-0.00667$. Hai số rất gần nhau (khi $T\gg|v|$ xấp xỉ tốt; nếu giảm $T$ xuống 2 thì sai lệch tăng rõ).

---

### 3.4 Cross-entropy trên nhãn teacher so với KL trên soft label

- Với phân phối teacher cố định $p$: $\mathrm{CE}(p,q)=H(p)+\mathrm{KL}(p\|q)$. $H(p)$ không phụ thuộc student nên gradient của CE và KL **giống hệt**.
- **Soft label** (token-level KD): dùng toàn bộ $p$ ở mỗi vị trí, tín hiệu dày.
- **Hard label từ teacher** (sequence-level KD, bài 15): lấy chuỗi $y^*$ do teacher sinh, tối thiểu $-\log q(y^*)$, tức CE với phân phối one-hot $\hat p=\delta_{y^*}$.
- Nếu teacher gần xác định ($p\approx\delta_{y^*}$) thì $\mathrm{KL}(p\|q)\approx-\log q(y^*)$: hai cách **trùng nhau**. Phần soft label chỉ thêm giá trị khi teacher không chắc chắn (phân phối có entropy đáng kể), ví dụ nhiều đáp án hợp lệ hoặc chỗ mơ hồ.
- Khi lấy mẫu $y\sim p$: $\mathbb E_{y\sim p}[-\log q(y)]=\mathrm{CE}(p,q)$. Tức hard-label SFT trên dữ liệu teacher lấy mẫu là ước lượng *không chệch nhưng phương sai cao* của gradient KD (token-level) trên mỗi vị trí.

**Hệ quả cho OCR (suy luận):** văn bản OCR thường có đáp án gần như duy nhất, teacher rất tự tin, nên KD toàn phân phối có thể không hơn nhiều so với SFT trên đầu ra teacher cho phần lớn token; phần khác biệt tập trung ở các ký tự mơ hồ. Cần đo entropy phân phối teacher trước khi đầu tư.

### 3.5 Forward KL và reverse KL

Định nghĩa với teacher $p$ và student $q$:

$$
\mathrm{KL}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}\quad\text{(forward KL)},\qquad
\mathrm{KL}(q\|p)=\sum_i q_i\log\frac{q_i}{p_i}\quad\text{(reverse KL)}.
$$

- **Forward KL** $= \mathbb E_{p}[\log p-\log q]$: phạt nặng nếu $q_i\approx0$ ở nơi $p_i>0$. Student bị ép *phủ hết* mọi chế độ của teacher: **mode-covering** (phủ mode). Khi student quá nhỏ để khớp chính xác, nó "dàn mỏng" xác suất, có thể đặt khối lượng ở vùng mà teacher cho xác suất rất thấp (sinh ra văn bản kỳ lạ).
- **Reverse KL** $=\mathbb E_{q}[\log q-\log p]$: phạt nặng nếu $q_i>0$ ở nơi $p_i\approx0$. Student bị ép *chọn* một vài mode mà teacher tin: **mode-seeking** (bám mode). Có thể bỏ sót các mode khác của teacher (mất đa dạng).

**Ví dụ số (đơn vị nats, tự tính).** Teacher hai đỉnh trên 4 lớp: $p=[0.49,\ 0.01,\ 0.01,\ 0.49]$. Hai student:
- $q_A=[0.25,0.25,0.25,0.25]$ (đều, phủ cả hai đỉnh nhưng dàn mỏng),
- $q_B=[0.97,0.01,0.01,0.01]$ (chỉ một đỉnh).

| Student | $\mathrm{KL}(p\|q)$ forward | $\mathrm{KL}(q\|p)$ reverse |
|---|---|---|
| $q_A$ (đều) | $0.595$ | $1.273$ |
| $q_B$ (một đỉnh) | $1.572$ | $0.623$ |

Forward KL thích $q_A$ (0.595 nhỏ hơn 1.572): không được bỏ rơi đỉnh nào. Reverse KL thích $q_B$ (0.623 nhỏ hơn 1.273): không được đặt khối lượng vào vùng teacher cho xác suất ~0. Hai cách đo "giống nhau" cho ra student tối ưu khác nhau.

**Trong thực tế LLM:** token-level forward KL dễ cho student "mơ hồ" (đặt xác suất lên các token teacher cho xác suất thấp), MiniLLM (bài 28) dùng reverse KL với tối ưu policy gradient; GKD (bài 29) tổng quát hoá và dùng được phân kỳ khác kèm on-policy; bài 31 (Rethinking KL) xem lại giả thiết "forward = phủ, reverse = bám" và nêu rằng khác biệt thực tế phức tạp hơn. Hãy đọc kỹ các bài đó thay vì tin một quy tắc đơn giản.

### 3.6 JS, TV, skew KL

- **Jensen-Shannon (JS):** với $m=\tfrac12(p+q)$, $\mathrm{JS}(p,q)=\tfrac12\mathrm{KL}(p\|m)+\tfrac12\mathrm{KL}(q\|m)$. Đối xứng, bị chặn trên (bởi $\log 2$ nats), không bùng nổ khi $q_i=0$ vì $m_i>0$. GKD dùng dạng JS tổng quát có trọng số $\beta$ nội suy giữa forward và reverse (theo mô tả của bài 29; hãy xem định nghĩa chính xác trong bài đó).
- **Total variation (TV):** $\mathrm{TV}(p,q)=\tfrac12\sum_i|p_i-q_i|\in[0,1]$. Gradient không bùng nổ, nhưng ít trơn nên ít dùng làm loss; xuất hiện trong phân tích lý thuyết và trong speculative decoding (xác suất chấp nhận $=1-\mathrm{TV}$, liên quan bài 32).
- **Skew KL:** $\mathrm{SKL}_\lambda(p\|q)=\mathrm{KL}\big(p\,\|\,\lambda p+(1-\lambda)q\big)$ (và dạng ngược). Trộn một phần $p$ vào mẫu số để tránh $q_i\approx0$ gây gradient nổ, ổn định huấn luyện. Là ý tưởng của DistiLLM (bài 30) theo tên "skew"; kiểm tra định nghĩa chính xác ở bài đó.

Nguyên tắc: các phân kỳ này chỉ khác nhau ở hành vi khi $q$ và $p$ lệch nhau nhiều (đúng trường hợp student nhỏ). Khi student đã khớp tốt, chúng gần như giống nhau.

### 3.7 Feature matching

Cho hidden state teacher $h_T\in\mathbb R^{d_T}$, student $h_S\in\mathbb R^{d_S}$.

**MSE với projector** (FitNets, bài 03):

$$
\mathcal L_{\text{feat}}=\big\|\,h_T-W\,h_S\,\big\|_2^2,\quad W\in\mathbb R^{d_T\times d_S}
$$

$W$ ("projector", "regressor", "adapter") là ánh xạ nhỏ học được để đưa hai không gian về cùng chiều; sau huấn luyện thường bỏ đi. Nếu $d_S=d_T$ và cấu trúc tương ứng có thể không cần.

**Cosine:**

$$
\mathcal L_{\cos}=1-\frac{\langle h_T,\,Wh_S\rangle}{\|h_T\|\,\|Wh_S\|}
$$

chỉ quan tâm hướng, bỏ qua độ lớn. Hữu ích khi các kênh có biên độ rất lệch (ví dụ vài kênh "massive activations" chiếm hết MSE). **Chuẩn hoá** trước khi so (L2-norm như bài 04, LayerNorm, hoặc chuẩn hoá theo từng chiều) cũng cùng mục đích.

**Lưu ý toán học quan trọng:** nếu projector $W$ là tuyến tính và đủ biểu đạt, nó có thể *hấp thụ mọi sai lệch tuyến tính* giữa hai không gian. Phần còn lại của loss chỉ đo sai lệch phi tuyến. Điều này giải thích vì sao feature loss có projector "dễ giảm" nhưng chưa chắc truyền được năng lực. Với dự án: sai lệch tại điểm cắt tách thành phần xử lý được bằng ánh xạ tuyến tính (D_lin) và phần dư (D_res) về mặt khái niệm chính là cùng cấu trúc này.

**Attention-map transfer** (bài 04) và **relational** (bài 12) là hai cách tránh khớp trực tiếp feature thô: lần lượt khớp bản đồ tóm tắt đã chuẩn hoá, và khớp quan hệ giữa các mẫu.

### 3.8 Exposure bias và lỗi tích luỹ trong mô hình tự hồi quy

Huấn luyện tự hồi quy thường dùng **teacher forcing**: ở bước $t$, mô hình nhận tiền tố đúng $y_{<t}$. Khi suy luận, nó nhận tiền tố do chính nó sinh. Sự khác nhau giữa phân phối tiền tố lúc huấn luyện và lúc suy luận gọi là **exposure bias**.

Hệ quả (kết quả quen thuộc từ imitation learning, Ross et al. 2011, nhắc theo kiến thức chung): nếu mỗi bước student sai với xác suất $\epsilon$ trên phân phối tiền tố đúng, thì lỗi tích luỹ trên chuỗi dài $L$ có thể tăng cỡ $O(\epsilon L^2)$ thay vì $O(\epsilon L)$, vì một lỗi đưa student vào trạng thái chưa từng thấy khi huấn luyện, nơi nó còn dễ sai hơn. Với chuỗi OCR dài (một trang), điều này đáng lo.

KD làm dịu hoặc làm nặng thêm vấn đề này tuỳ cách làm:
- **Off-policy token-level KD** vẫn bị exposure bias.
- **On-policy KD** (bài 28–30): student tự sinh tiền tố, teacher chấm trên đó, nên student học cách sửa lỗi ở chính trạng thái nó thực sự gặp.
- **Sequence-level KD** (bài 15): dùng chuỗi do teacher sinh ra, ít đa dạng hơn dữ liệu thật và dễ học hơn (teacher có đầu ra "sạch" hơn), nhưng vẫn là dữ liệu do teacher chọn, không phải do student.

### 3.9 Ước lượng thưa (top-k) của logit và độ chệch

Lưu hoặc truyền toàn bộ phân phối $p\in\mathbb R^V$ với $V\sim10^5$ cho mỗi token rất tốn kém. Ví dụ: $V=150{,}000$ ở fp16 là $300$ KB mỗi vị trí; $10^9$ vị trí là $300$ TB. Nên người ta cắt:

**Top-k + chuẩn hoá lại:** giữ $k$ token có $p$ lớn nhất, đặt $\tilde p_i=p_i/\sum_{j\in\text{top-}k}p_j$ cho $i\in\text{top-}k$ và $0$ cho phần còn lại.

Độ chệch: gọi $\rho=\sum_{j\in\text{top-}k}p_j<1$ là khối lượng giữ lại, phần đuôi $1-\rho$ bị bỏ. Hệ quả:
- Student được dạy rằng token ngoài top-$k$ có xác suất đúng bằng 0, và các token trong top-$k$ lớn hơn thực tế $1/\rho$ lần: **hiệu chuẩn sai** (student quá tự tin), mất tri thức ở đuôi.
- Gradient kỳ vọng không còn bằng gradient của KL đầy đủ, tức là ước lượng **chệch** (biased). Chệch lớn khi teacher có entropy cao (đuôi dày), ví dụ chuỗi sáng tạo; nhỏ khi teacher gần xác định (OCR).
- Có thể lưu thêm phần "khối lượng đuôi" vào một lớp "other", làm giảm chệch nhưng mất cấu trúc bên trong đuôi.

**Lấy mẫu có trọng số quan trọng** (importance sampling): lấy $m$ token từ một phân phối đề xuất $r$ rồi ước lượng $\sum_i p_i f_i\approx\frac1m\sum_s \frac{p_{i_s}}{r_{i_s}}f_{i_s}$; không chệch nhưng phương sai phụ thuộc $r$. Bài 45 (Sparse logit sampling) phân tích chi tiết so sánh top-$k$ chệch và lấy mẫu không chệch; hãy đọc bài đó để biết con số, bài này chỉ cung cấp khung.

### 3.10 Tóm tắt các lỗi trực giác hay gặp

- "KD = bắt student giống teacher": đúng cho **hàm** (đầu vào đến đầu ra), không đúng cho tham số.
- "Loss thấp hơn thì tốt hơn": KL trên tập huấn luyện thấp chưa chắc cho chất lượng nhiệm vụ cao hơn; luôn đánh giá bằng metric nhiệm vụ (CER/WER).
- "Nhiệt độ cao luôn tốt": nhiệt độ cao làm phẳng, gradient yếu đi, cần $T^2$ bù; student nhỏ có thể muốn $T$ vừa (bài 02: 2.5–4 cho student 30 unit).

---

## 4. Cạm bẫy thực hành

### 4.1 Capacity gap (khoảng cách dung lượng)

Student quá nhỏ so với teacher thì không bắt chước được, và teacher "tốt hơn" đôi khi cho student tệ hơn. Giải pháp trong lộ trình: Teacher Assistant (bài 08), teacher "kiên nhẫn và nhất quán" (bài 11), quy luật capacity gap (bài 41), và thang đo teacher/student trong scaling laws (bài 42). Với dự án (4B xuống 0.3B/0.1B, tỉ lệ ~13 lần và ~40 lần): coi capacity gap là giả thuyết chính cần kiểm tra.

### 4.2 Tokenizer mismatch

KD ở mức logit đòi hỏi student và teacher chia sẻ **cùng từ vựng và cách tách token**, vì phép KL cộng theo chỉ số token. Nếu khác tokenizer, phải căn chỉnh (alignment) hoặc dùng phương pháp không phụ thuộc từ vựng, hoặc quay về data-level distillation. Khi student là bản cắt tỉa của chính teacher (dự án của chúng ta) thì không có vấn đề này, đây là một lợi thế đáng kể (suy luận).

### 4.3 Hiệu chuẩn (calibration) của teacher

KD dạy student theo *xác suất* của teacher. Nếu teacher quá tự tin hoặc kém hiệu chuẩn (probabilities không khớp tần suất đúng thực tế), student kế thừa điều đó. Cách xử lý: nhiệt độ $T>1$, label smoothing cho teacher, ensemble, hoặc chọn lọc mẫu (chỉ distill khi teacher đúng, như vài bài về reasoning). Đo **entropy** và **ECE** (expected calibration error) của teacher trên tập dev là bước đơn giản.

### 4.4 Kế toán chi phí tính toán (compute accounting)

So sánh công bằng cần tính đủ:
- Chi phí **forward của teacher** trên mỗi token huấn luyện (thường lớn hơn forward+backward của student nhỏ).
- Chi phí **lưu trữ logit** nếu cache offline (mục 3.9) so với chạy teacher trực tuyến.
- Với on-policy KD: thêm chi phí **sinh chuỗi** của student.
- Chi phí huấn luyện **teacher** (nếu phải train riêng) có được tính không?
- Phép so sánh thích hợp là "cùng ngân sách FLOPs hoặc cùng số token" giữa KD và huấn luyện student trực tiếp (bài 42 làm việc này); một số bài chỉ so cùng số bước, khiến KD trông tốt hơn thực tế.

### 4.5 Các cạm bẫy khác

- **Quên $T^2$** khi trộn soft và hard loss: tỉ lệ đóng góp thay đổi theo $T$ mà ta không để ý.
- **Teacher ở chế độ train** (dropout bật, batch-norm cập nhật) khi sinh soft target: phải `eval()` và `no_grad`.
- **Rò rỉ đánh giá:** teacher đã thấy tập test (hoặc dữ liệu tương tự) làm student hưởng lợi gián tiếp. Với benchmark công khai, kiểm tra nguồn dữ liệu của teacher.
- **Nhầm KD với tinh chỉnh nhẹ:** khi student là bản cắt, một phần lợi ích chỉ là "huấn luyện thêm sau cắt", không phải KD. Luôn có baseline "cùng dữ liệu và bước, chỉ dùng nhãn thật".

---

## 5. Bảng thuật ngữ

| Thuật ngữ | Tiếng Việt / giải nghĩa ngắn |
|---|---|
| Teacher / Student | mô hình dạy / mô hình học |
| Soft label / soft target | phân phối xác suất mềm do teacher cho |
| Hard label | nhãn one-hot hoặc token cụ thể |
| Dark knowledge | thông tin trong xác suất nhỏ của các lớp sai |
| Temperature $T$ | tham số làm phẳng softmax |
| Logit | giá trị trước softmax |
| KL divergence | độ đo lệch giữa hai phân phối, không đối xứng |
| Forward KL | $\mathrm{KL}(p_{teacher}\|q_{student})$, mode-covering |
| Reverse KL | $\mathrm{KL}(q_{student}\|p_{teacher})$, mode-seeking |
| JS / TV / skew KL | các phân kỳ thay thế, ổn định hơn |
| Response-based | KD dựa trên output |
| Feature-based | KD dựa trên hidden state / feature |
| Relation-based | KD dựa trên quan hệ giữa mẫu |
| Hint / guided layer | lớp teacher dùng làm gợi ý / lớp student được dẫn dắt |
| Projector / regressor | ánh xạ khớp chiều giữa hai không gian feature |
| Offline / online / self-distillation | teacher cố định / học cùng / chính nó |
| Token-level / sequence-level KD | khớp phân phối từng token / học từ chuỗi teacher sinh |
| Off-policy / on-policy | chuỗi từ dữ liệu có sẵn / chuỗi do student sinh |
| Exposure bias | lệch giữa tiền tố lúc huấn luyện và lúc suy luận |
| Teacher forcing | huấn luyện với tiền tố đúng |
| Capacity gap | khoảng cách dung lượng giữa teacher và student |
| Teacher assistant | mô hình trung gian cầu nối |
| Data-level distillation | teacher sinh dữ liệu, student SFT |
| SFT | supervised fine-tuning, tinh chỉnh có giám sát |
| Top-k logits | chỉ giữ $k$ xác suất lớn nhất |
| Calibration (ECE) | độ khớp giữa xác suất dự đoán và tần suất đúng |
| Self-distillation | teacher và student cùng kiến trúc / cùng mạng |
| Born-again network | student cùng kiến trúc, lặp lại nhiều thế hệ |
| Dustbin class | lớp gộp phần còn lại (bài 02, specialist) |
| Label smoothing | trộn nhãn one-hot với phân phối đều, regularizer |
| Mode-covering / mode-seeking | phủ mọi đỉnh / bám một vài đỉnh |
| D_lin, D_res (của dự án) | phần hư hại do cắt sửa được bằng ánh xạ tuyến tính tối ưu / phần dư còn lại |

---

## 6. Cách đọc paper KD một cách phản biện

Một danh sách câu hỏi dùng cho mọi bài trong thư mục:

1. **Trục phân loại:** tri thức nào (response/feature/relation)? offline/online/self? token-level hay sequence-level? on/off-policy? Đặt bài vào bản đồ của bài 05 trước khi đọc chi tiết.
2. **Student và teacher là gì?** Tỉ lệ kích thước? Cùng họ kiến trúc? Cùng tokenizer? (Capacity gap, mục 4.1.)
3. **Baseline có công bằng không?** Student có được huấn luyện với *cùng dữ liệu, cùng số bước, cùng tối ưu hoá* nhưng chỉ dùng nhãn thật không? Siêu tham số baseline có được tinh chỉnh bằng công sức tương đương? Teacher có được tính vào ngân sách?
4. **Kích thước cải thiện so với nhiễu:** một số bài báo cáo +0.3 điểm trên một seed. Có bao nhiêu seed? Trung vị hay tốt nhất? (Bài 04 dùng trung vị của 5 lần chạy cho CIFAR, nhưng ImageNet chỉ một lần.)
5. **Có ablation tách đóng góp không?** Ví dụ khi một bài kết hợp feature + response + curriculum, mỗi phần đóng góp bao nhiêu?
6. **Metric nào?** Loss KD thấp hay metric nhiệm vụ? Có đánh giá trên phân phối khác (out-of-distribution) không?
7. **Cuộc chơi thích nghi:** ai chọn siêu tham số trên tập test? Có early stopping trên test không?
8. **Chi phí:** thời gian, bộ nhớ, forward của teacher, lưu trữ logit (mục 4.4).
9. **Phạm vi khái quát:** thí nghiệm ở CNN ảnh nhỏ (bài 01–04) có chuyển sang Transformer/LLM không? Tác giả có nói không, hay ta tự suy ra?
10. **Phân biệt ba loại câu:** điều bài *chứng minh/đo*, điều bài *giả thuyết* (thường sau từ "we conjecture", "we believe"), và điều *ta suy ra*. Ghi chú trong thư mục này cố gắng tách chúng; hãy làm thế khi ghi chú riêng.
11. **Mâu thuẫn giữa các bài:** nhiều kết luận sớm bị phản biện (ví dụ bài 10 hỏi liệu student có thực sự khớp teacher, bài 31 xem lại hành vi forward/reverse KL, bài 39 cho thấy mô hình nhỏ khó học từ chuỗi suy luận mạnh). Khi hai bài mâu thuẫn, hỏi điều kiện thí nghiệm nào khác nhau.

---

## 7. Lịch đọc gợi ý

Mỗi giai đoạn khoảng một tuần, đọc ghi chú trước, paper gốc sau nếu cần.

| Giai đoạn | Bài | Mục tiêu |
|---|---|---|
| 0. Nền | 00 (bài này), 01, 02 | Hiểu soft target, nhiệt độ, $T^2$, KL; tự cài một vòng KD trên MNIST/CIFAR. |
| 1. Feature và bản đồ | 03, 04, 05 | Hint, projector, attention map; dùng survey để định vị. |
| 2. KD cổ điển, tại sao và khi nào | 06, 07, 08, 09, 10, 11 | Self/online KD, capacity gap, kiểm chứng thực nghiệm. |
| 3. Quan hệ và đối nghịch/tách thành phần | 12, 13, 14 | Relational, contrastive, decoupled. |
| 4. Chuỗi và NLP | 15, 16, 17, 18, 19, 20 | Sequence-level, DistilBERT, TinyBERT, MiniLM, causal distillation. |
| 5. Thị giác và VLM (liên quan trực tiếp ViT/VLM của dự án) | 21, 22, 23, 24, 25, 26 | DeiT, DINO, AM-RADIO, CLIP-KD, LLaVA-MoD, LLaVA-KD. |
| 6. LLM: nền và phân kỳ | 27, 28, 29, 30, 31, 32, 33, 34 | Khảo sát, MiniLLM, GKD, DistiLLM, Rethinking KL, Speculative KD, TAID, DistiLLM-2. |
| 7. LLM: dữ liệu và suy luận | 35, 36, 37, 38, 39 | Self-Instruct, step-by-step, Orca, R1 distillation, small models struggle. |
| 8. Quy mô và hệ thống | 40, 41, 42, 43, 44, 45, 46, 47 | Gemma 2, capacity gap law, scaling laws, MiniPLM, Qwen3, sparse logit sampling, Distil-Whisper, ShortOPD. |

**Đường tắt cho dự án (ưu tiên):** 00 → 02 → 03 → 05 → 08 → 14 → 19 → 23 → 26 → 28 → 29 → 31 → 40 → 41 → 42 → 45 → 47. Đây chỉ là gợi ý của người biên soạn; điều chỉnh theo kết quả thí nghiệm.

---

## 8. Liên hệ với dự án (suy luận)

- **Bối cảnh:** LightOnOCR-3-0.8B (decoder hybrid GatedDeltaNet + full-attention, ViT 12 block) được cắt tỉa có cấu trúc xuống ~0.3B / ~0.1B và phục hồi bằng KD từ teacher 4B; mục tiêu GPU ≤ 8 GB, latency batch-1.
- **Cùng tokenizer:** student là bản cắt của họ mô hình nên (giả sử teacher 4B cùng tokenizer) có thể KD ở mức logit đầy đủ.
- **Teacher gần xác định?** Nếu entropy phân phối token của teacher trên văn bản OCR rất thấp, mục 3.4 cho thấy KL toàn phân phối và SFT trên đầu ra teacher gần như tương đương; ưu tiên đầu tư vào on-policy hoặc feature-matching thay vì tinh chỉnh $T$.
- **Exposure bias** (mục 3.8) quan trọng với chuỗi dài; thử on-policy hoặc trộn on/off-policy (bài 29, 30, 32).
- **Quan sát: cắt neuron MLP phục hồi kém hơn cắt layer.** Giả thuyết cần kiểm tra (không phải kết luận): khi cắt neuron, sai lệch trải đều lên mọi lớp, khó sửa bằng một ánh xạ tuyến tính tại một điểm (D_res lớn), nên loss ở output đơn thuần có thể không đủ; feature loss ở nhiều tầng (bài 03, 19) hoặc capacity gap (bài 41) là hai hướng giải thích để thử.
- **Hybrid compression:** bài 05 xác nhận thứ tự giữa KD và nén khác là câu hỏi mở; thí nghiệm "cắt từng bước + KD xen kẽ" là một cách trả lời bằng dữ liệu của riêng dự án.

---

## 9. Năm câu tự kiểm tra

**Câu 1.** Vì sao khi dùng loss $\alpha T^2\mathrm{KL}(p^{(T)}\|q^{(T)})+(1-\alpha)\mathrm{CE}(y,q^{(1)})$ phải nhân $T^2$? Điều gì xảy ra nếu quên?

<details>
<summary>Đáp án</summary>

Gradient của soft-loss theo logit student là $\frac1T(q_i-p_i)$, và ở $T$ cao xấp xỉ $\frac{1}{NT^2}(z_i-v_i)$; độ lớn co theo $1/T^2$. Nhân $T^2$ giữ độ lớn gradient của số hạng soft gần như không đổi khi thay $T$, nên tỉ lệ với số hạng hard không đổi. Nếu quên, tăng $T$ sẽ vô tình giảm tầm quan trọng của số hạng soft, và siêu tham số $\alpha$ tối ưu sẽ phụ thuộc vào $T$ (khó tinh chỉnh, kết luận về "nhiệt độ tốt nhất" bị nhiễu).
</details>

**Câu 2.** Teacher gần như xác định ($p\approx\delta_{y^*}$ ở mọi vị trí). KD token-level (KL) khác gì so với SFT trên chuỗi do teacher sinh? Khi nào KD token-level vẫn đáng giá?

<details>
<summary>Đáp án</summary>

Khi $p\approx\delta_{y^*}$ thì $\mathrm{KL}(p\|q)\approx-\log q(y^*)$, tức hai loss trùng nhau (và gradient giống nhau); KD toàn phân phối không thêm thông tin. Nó vẫn đáng giá ở những vị trí teacher *không* chắc (entropy cao): nhiều đáp án hợp lệ, vùng mơ hồ, nơi hard-label SFT chỉ thấy một mẫu ngẫu nhiên (ước lượng không chệch nhưng phương sai cao của gradient KL). Với OCR, điều này gợi ý đo entropy của teacher trước để biết KD toàn phân phối có lợi không.
</details>

**Câu 3.** Cho $p=[0.49,0.01,0.01,0.49]$. Student nào tối ưu hơn theo forward KL, student nào theo reverse KL: $q_A=[0.25,0.25,0.25,0.25]$ hay $q_B=[0.97,0.01,0.01,0.01]$? Giải thích bằng "phủ mode" và "bám mode".

<details>
<summary>Đáp án</summary>

Forward KL: $\mathrm{KL}(p\|q_A)\approx0.595$ so với $\mathrm{KL}(p\|q_B)\approx1.572$, nên $q_A$ tốt hơn: forward KL phạt nặng nếu $q\approx0$ ở nơi $p>0$ ($q_B$ gần bỏ hẳn đỉnh thứ hai), tức mode-covering. Reverse KL: $\mathrm{KL}(q_A\|p)\approx1.273$ so với $\mathrm{KL}(q_B\|p)\approx0.623$, nên $q_B$ tốt hơn: reverse KL phạt nặng nếu $q>0$ ở nơi $p\approx0$ ($q_A$ đặt 0.5 khối lượng vào hai lớp giữa mà $p$ cho 0.01), tức mode-seeking. Sau đó cần nhớ bài 31 cảnh báo rằng thực tế phức tạp hơn mô hình đơn giản này.
</details>

**Câu 4.** Bạn lưu top-$k$ xác suất của teacher (đã chuẩn hoá lại) để tiết kiệm bộ nhớ. Điều này gây độ chệch gì? Khi nào độ chệch nhỏ?

<details>
<summary>Đáp án</summary>

Gọi $\rho$ là tổng xác suất của top-$k$. Chuẩn hoá lại nhân các xác suất giữ lại với $1/\rho>1$ và đặt đuôi bằng 0, nên student được dạy một phân phối quá tự tin, không có đuôi; gradient kỳ vọng khác gradient KL đầy đủ (ước lượng chệch). Độ chệch nhỏ khi $\rho\approx1$, tức teacher có entropy thấp hoặc $k$ đủ lớn (ví dụ OCR, teacher rất tự tin). Lớn khi phân phối đuôi dày. Giải pháp: lấy mẫu có trọng số quan trọng cho ước lượng không chệch (bài 45) hoặc lưu thêm khối lượng đuôi.
</details>

**Câu 5.** Vì sao một feature loss với projector tuyến tính có thể giảm xuống rất thấp mà student vẫn không tốt, và liên hệ gì với cách ta phân tách hư hại D_lin và D_res?

<details>
<summary>Đáp án</summary>

Projector tuyến tính $W$ có thể hấp thụ mọi sai lệch tuyến tính giữa hai không gian biểu diễn, nên loss $\|h_T-Wh_S\|^2$ chỉ còn đo phần không tuyến tính; nhưng nó cũng nghĩa là student có thể có "đủ thông tin" ở dạng bị xoay/co giãn mà các lớp tiếp theo của student không đọc được (vì $W$ bị bỏ khi suy luận). Vậy loss thấp chưa chắc cải thiện metric nhiệm vụ. Tương tự, với dự án: phần hư hại D_lin do cắt có thể được sửa bằng ánh xạ tuyến tính tối ưu tại điểm cắt, còn D_res (phần dư) mới là phần cần học thực sự; một projector tuyến tính khi huấn luyện che giấu D_lin nhưng sẽ không có khi triển khai, nên cần kiểm tra bằng metric ở đầu ra thay vì chỉ nhìn feature loss.
</details>

---

*Tài liệu này không có đường link gốc vì không ghi chú một paper cụ thể; hãy dùng các bài 01–47 trong thư mục và bản đồ ở bài 05 (https://arxiv.org/abs/2006.05525).*
