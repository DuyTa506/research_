# Preface: Pruning cho người mới bắt đầu — toán nền tảng và hướng dẫn đọc

**Tác giả:** ghi chú tổng hợp (không bám một paper duy nhất), viết từ kiến thức nền về pruning mạng nơ-ron.  
**Nguồn:** không có nguồn đơn lẻ; các paper được nhắc tới nằm trong lộ trình 01–49 của thư mục này.  
**Chủ đề:** Vì sao và cách cắt tỉa (prune) mạng nơ-ron; phân loại; tiêu chí; phục hồi; toán nền; đánh giá; cách đọc paper một cách phê phán.  
**Vị trí trong lộ trình:** Đọc **trước tiên**. Các ghi chú 01–49 giả định bạn đã nắm các ý trong file này.

> Tài liệu học tập bằng tiếng Việt, viết từ kiến thức chung của người soạn, **không phải** bản dịch hay tóm tắt một paper. Mọi con số cụ thể về paper nằm ở các ghi chú riêng. Phần "áp dụng cho dự án" là suy luận của chúng ta, chưa được kiểm chứng.

## Mục lục

1. [Vì sao phải prune](#1-vì-sao-phải-prune)
2. [Phân loại theo "cái gì bị cắt"](#2-phân-loại-theo-cái-gì-bị-cắt)
3. [Cắt khi nào: lúc khởi tạo, trong lúc train, sau khi train](#3-cắt-khi-nào)
4. [Tiêu chí chọn cái cần cắt](#4-tiêu-chí-chọn-cái-cần-cắt)
5. [Phục hồi sau khi cắt](#5-phục-hồi-sau-khi-cắt)
6. [Toán cần biết](#6-toán-cần-biết)
7. [Vì sao structured cho tốc độ thật còn unstructured thì không](#7-vì-sao-structured-cho-tốc-độ-thật)
8. [Đánh giá](#8-đánh-giá)
9. [Bảng thuật ngữ](#9-bảng-thuật-ngữ)
10. [Đọc paper pruning một cách phê phán](#10-đọc-paper-pruning-một-cách-phê-phán)
11. [Lịch đọc gợi ý theo số thứ tự thư mục](#11-lịch-đọc-gợi-ý)
12. [Năm câu tự kiểm tra](#12-năm-câu-tự-kiểm-tra)

---

## 1. Vì sao phải prune

**Pruning** (cắt tỉa) là bỏ đi một phần tham số hoặc cấu trúc của một mạng đã (hoặc sắp) được huấn luyện, với hy vọng giữ phần lớn chất lượng. Động lực thực tế có ba loại, và chúng **không đồng nhất**:

| Mục tiêu | Đại lượng đo | Pruning giúp thế nào |
|---|---|---|
| Bộ nhớ lưu trữ / tải | số byte của checkpoint | bớt tham số; nhưng nếu giữ dạng thưa thì tốn thêm chỉ số (index) |
| Bộ nhớ chạy (VRAM) | trọng số + activation + KV cache | bớt trọng số; activation chỉ bớt khi cắt channel/width |
| Độ trễ (latency) | giây/token hoặc giây/ảnh | chỉ bớt khi phần cứng thật sự bỏ qua công việc |
| Thông lượng (throughput) | token/giây ở batch lớn | phụ thuộc compute-bound hay memory-bound |
| Năng lượng | Joule | truy cập bộ nhớ thường đắt hơn phép tính nhiều bậc (xem ghi chú 02) |

### 1.1 FLOPs không phải tốc độ

**FLOPs** (floating-point operations) là số phép tính lý thuyết. Một mạng ít FLOPs hơn có thể chạy *không* nhanh hơn, vì:

- GPU chạy nhanh nhất với phép nhân ma trận **dày đặc, đều đặn** (dense GEMM). Mẫu truy cập thưa, bất quy tắc làm mất lợi thế này.
- Ở **batch-1 decode** của mô hình ngôn ngữ, mỗi token phải đọc toàn bộ trọng số từ bộ nhớ (HBM/DRAM) một lần. Thời gian thường bị chặn bởi băng thông bộ nhớ chứ không phải số phép nhân: gần đúng

$$
t_{\text{token}} \approx \max\!\Big(\frac{\text{bytes đọc}}{\text{băng thông}},\ \frac{\text{FLOPs}}{\text{đỉnh tính toán}}\Big).
$$

  Với batch nhỏ, số hạng đầu thường lớn hơn → muốn nhanh phải **giảm số byte thật sự phải đọc**, nghĩa là các ma trận phải *nhỏ lại* (hoặc lượng tử hóa), không chỉ "nhiều số 0".
- Các chi phí phụ (kernel launch, overhead Python, attention với KV cache dài, norm, embedding/LM-head) không giảm khi ta chỉ cắt MLP.

Vì vậy một câu hỏi luôn phải đặt ra khi đọc paper: **"giảm gì — tham số, FLOPs hay latency đo trên phần cứng nào?"**

### 1.2 Pruning trong bức tranh nén mô hình

Pruning chỉ là một trong vài họ kỹ thuật nén, và thường được kết hợp:

- **Quantization**: dùng ít bit hơn cho mỗi số (FP16 → INT8/INT4).
- **Knowledge distillation (KD)**: huấn luyện mô hình nhỏ bắt chước mô hình lớn.
- **Low-rank factorization**: thay $W \approx AB$ với hạng nhỏ.
- **Thiết kế kiến trúc nhỏ ngay từ đầu** (hoặc train từ đầu một mô hình nhỏ).

Một câu hỏi nghiên cứu mở xuyên suốt lộ trình: *prune từ mô hình lớn có thực sự tốt hơn train mô hình nhỏ từ đầu không?* (xem 03, 11, 43).

---

## 2. Phân loại theo "cái gì bị cắt"

Hình dung một tầng tuyến tính $y = Wx$, với $W \in \mathbb{R}^{d_{out}\times d_{in}}$.

### 2.1 Unstructured (không cấu trúc) — cắt từng trọng số

Đặt từng phần tử $W_{ij}=0$ theo một mặt nạ (mask) $M \in \{0,1\}^{d_{out}\times d_{in}}$: $W' = M \odot W$. Rất linh hoạt, giữ được chất lượng ở độ thưa cao (50–90%), nhưng ma trận vẫn có kích thước cũ. Các bài tiêu biểu: 01 (OBD/OBS), 02 (Han 2015), 03 (gradual pruning), 08 (Lottery Ticket), 22–24 (OBC, SparseGPT, Wanda).

### 2.2 Semi-structured, mẫu N:M

Trong mỗi nhóm $M$ phần tử liên tiếp, giữ đúng $N$ phần tử khác 0. Phổ biến nhất là **2:4**: mỗi 4 trọng số liên tiếp có đúng 2 số 0 → độ thưa 50%. Phần cứng NVIDIA (Ampere trở đi) có Sparse Tensor Core nén theo mẫu này và lý thuyết có thể gần gấp đôi thông lượng GEMM. Xem 18 (2:4 sparsity), 30 (MaskLLM). Hạn chế: chỉ cố định 50%, chỉ có lợi trên phần cứng/kernel được hỗ trợ, và lợi ích thực tế thấp hơn lý thuyết.

### 2.3 Structured (có cấu trúc) — cắt nguyên khối

Cắt cả một đơn vị tính toán, kết quả vẫn là **ma trận dày đặc nhỏ hơn**. Các mức cấu trúc, từ mịn đến thô:

| Đơn vị bị cắt | Ví dụ | Ghi chú |
|---|---|---|
| Filter / channel (CNN) | 04, 05, 06 | cắt một filter kéo theo feature map tương ứng và kernel ở tầng sau |
| Neuron (MLP) | LLM-Pruner (26), FLAP (28) | cắt một hàng của $W_{in}$ và một cột của $W_{out}$ |
| Attention head | 13 (Sixteen heads), 14 (Voita) | cắt các lát $W_Q,W_K,W_V$ theo head và lát $W_O$ tương ứng |
| Layer / block | 31–37 (ShortGPT, SLEB, LaCo, ReplaceMe...) | bỏ cả khối transformer; còn lại nối tiếp qua residual |
| Embedding width / hidden size | 29 (SliceGPT), 38–40 (Sheared LLaMA, Minitron) | giảm $d_{model}$ — đụng đến mọi ma trận, residual stream, norm |
| Expert (MoE), state-dimension (SSM) | 41–46 | tùy kiến trúc |

Nguyên tắc: càng thô thì càng dễ tăng tốc thật, nhưng mỗi lần cắt càng gây hại nhiều (mất cả "khối chức năng") và càng khó phục hồi bằng fine-tune ngắn.

### 2.4 Phụ thuộc cấu trúc (dependency)

Trong mạng có residual/skip, các lớp bị ràng buộc: nếu cắt channel ở đầu ra của một lớp cộng vào residual thì mọi lớp cùng cộng vào residual phải cắt cùng chỉ số. Đây là lý do có các công cụ như DepGraph (20) và các luật "nhóm tham số bị ràng buộc" (coupled structures). Ghi chú 04 (Li et al.) mô tả sớm vấn đề này với ResNet.

---

## 3. Cắt khi nào

### 3.1 Sau khi train xong rồi fine-tune (prune → fine-tune)

Quy trình kinh điển: train dense → chấm điểm → cắt → fine-tune; có thể lặp (iterative). Han 2015 (02), Li 2016 (04).

### 3.2 Cắt dần trong lúc train (gradual / during training)

Độ thưa tăng dần theo lịch trong khi vẫn train, để mạng thích nghi từng bước (03 Zhu & Gupta). Các biến thể học mặt nạ cùng lúc với trọng số: Network Slimming (05), Movement Pruning (16), CoFi (17).

### 3.3 Cắt ngay từ khởi tạo (pruning at initialization, PaI)

Chọn mạng con **trước khi train** dựa trên gradient/độ nhạy tại điểm khởi tạo (SNIP, 10) hoặc dựa trên giả thuyết vé số trúng thưởng (Lottery Ticket, 08–09). Bài học thường thấy: với mô hình lớn hiện đại, kết quả PaI thường không vượt các phương pháp cắt sau train.

### 3.4 Post-training một lần (one-shot, không train lại hoặc gần như không)

Đây là chế độ chính của thời LLM: không đủ tính toán để huấn luyện lại, nên dùng một lượng nhỏ dữ liệu hiệu chuẩn (calibration data, ví dụ vài trăm đoạn văn) để tính điểm và có thể *sửa trọng số còn lại* bằng đại số: OBC (22), SparseGPT (23), Wanda (24), SliceGPT (29), FLAP (28).

### 3.5 Cắt rồi huấn luyện lại khối lượng lớn (prune + continued pre-training / KD)

Cắt nặng, sau đó tiếp tục huấn luyện với nhiều token và/hoặc distillation từ mô hình gốc: Sheared LLaMA (38), Minitron (39–40), Puzzle (42), Nemotron-H (44). Đây là hướng gần dự án của chúng ta nhất.

| Thời điểm | Chi phí tính toán | Mức cắt tối đa điển hình | Ví dụ |
|---|---|---|---|
| Lúc khởi tạo | thấp | thấp–vừa | 10 |
| Trong lúc train | cao (train đầy đủ) | cao | 03, 05, 16 |
| Sau train + fine-tune | vừa–cao | cao | 02, 04 |
| One-shot post-training | rất thấp | vừa (≈ 20–50%) | 23, 24, 29 |
| Cắt + train lại lớn | rất cao | rất cao (2–4×) | 38–40 |

---

## 4. Tiêu chí chọn cái cần cắt

**Saliency (độ quan trọng)** của một tham số/cấu trúc là điểm số ước lượng "nếu bỏ nó thì tổn hại bao nhiêu". Mọi tiêu chí sau đều là cách xấp xỉ khác nhau của đại lượng này.

### 4.1 Magnitude (độ lớn)

$s_i = |w_i|$ (hoặc chuẩn $\ell_1$/$\ell_2$ của một filter/hàng). Không cần dữ liệu, rẻ, mạnh một cách đáng ngạc nhiên. Điểm yếu: bỏ qua độ lớn của đầu vào (một trọng số nhỏ nhân với activation rất lớn vẫn quan trọng) và bỏ qua tương tác. Xem 02, 04.

### 4.2 Gradient / Taylor bậc nhất

Dùng $s_i = |g_i w_i|$ hay $(g_i w_i)^2$ với $g_i=\partial L/\partial w_i$: khai triển bậc nhất của thay đổi loss khi đặt $w_i=0$. Với cấu trúc (một nhóm tham số), cộng theo nhóm. Xem 07 (Molchanov 2019), 10 (SNIP), 26 (LLM-Pruner).

### 4.3 Hessian / bậc hai (OBD, OBS)

Dùng độ cong của loss: $s_i=\tfrac12 H_{ii}w_i^2$ (OBD) hoặc $\dfrac{w_i^2}{2[H^{-1}]_{ii}}$ (OBS). Đúng hơn về lý thuyết, nhưng tốn kém; các bản hiện đại làm xấp xỉ theo từng tầng: 22 (OBC), 23 (SparseGPT), 27 (LLM Surgeon). Xem mục 6.

### 4.4 Dựa trên activation (data-aware, theo từng tầng)

Với tầng $y=Wx$, tác động của việc bỏ $W_{ij}$ lên đầu ra tỉ lệ với $|W_{ij}|\cdot\|x_j\|_2$ (chuẩn của cột $j$ của dữ liệu đầu vào). **Wanda** (24) chấm điểm đúng như vậy: $s_{ij}=|W_{ij}|\cdot\|X_j\|_2$ và cắt theo từng hàng đầu ra. Rất rẻ nhưng bắt được các "outlier feature" của LLM mà magnitude bỏ lỡ.

### 4.5 Mặt nạ học được (learned masks)

Đưa mặt nạ (hoặc cổng) vào huấn luyện như một biến tối ưu: hệ số $\gamma$ của BatchNorm với phạt $\ell_1$ (05), điểm movement (16), mặt nạ $\ell_0$ khả vi / hard-concrete (17 CoFi), mặt nạ N:M học bằng Gumbel-softmax (30 MaskLLM). Chính xác hơn nhưng cần train; xem mục 6.6.

### 4.6 Dựa trên tái tạo (reconstruction-based)

Chọn và chỉnh phần còn lại sao cho **đầu ra của tầng/khối** gần với đầu ra gốc trên dữ liệu hiệu chuẩn:

$$
\min_{W'}\ \|WX - W'X\|_F^2 .
$$

Gồm ThiNet (06), OBC/SparseGPT (22, 23), SliceGPT (29), LaCo/ReplaceMe (35, 36) khi thay khối bị cắt bằng một phép biến đổi tuyến tính. Công thức đóng ở mục 6.5.

### 4.7 Tiêu chí cho cả khối/layer

Với việc bỏ cả layer: độ tương đồng cosine giữa đầu vào và đầu ra khối (ShortGPT 31, 32), thay đổi perplexity khi bỏ khối (34 SLEB, 33), hoặc ước lượng "chi phí còn lại" sau khi thay bằng ánh xạ tuyến tính tối ưu (36, 37). Lưu ý của dự án: tách tổn hại thành $D_{lin}$ (phần một phép tuyến tính tối ưu tại điểm cắt xóa được) và $D_{res}$ (phần còn lại) là cách nhìn nối các bài này.

---

## 5. Phục hồi sau khi cắt

Cắt gây một "cú sốc"; phục hồi là bước kéo chất lượng trở lại.

1. **Fine-tune toàn bộ trọng số còn lại** với loss gốc (cross-entropy), thường với learning rate nhỏ hơn lúc pretrain (02: 1/10 hoặc 1/100 LR gốc; 04: LR hằng 0.001). Tốt nhất nhưng tốn bộ nhớ và dữ liệu.
2. **LoRA / adapter**: chỉ train các ma trận hạng thấp thêm vào; rẻ, dùng trong LLM-Pruner (26). Hạn chế: dung lượng phục hồi có giới hạn khi cắt nặng.
3. **Knowledge distillation**: dùng mô hình gốc (hoặc một teacher lớn hơn) làm giáo viên; loss thường là

$$
\mathcal{L} = \alpha\, \mathrm{KL}\big(p_T(\cdot\mid x)\,\|\,p_S(\cdot\mid x)\big) + (1-\alpha)\,\mathrm{CE}(y, p_S) ,
$$

   có thể thêm khớp hidden states / attention. Thường phục hồi tốt hơn CE đơn thuần (xem 38–40, 43). Đây là hướng của dự án với teacher 4B.
4. **Sửa trọng số bằng đại số** (không gradient): OBS, SparseGPT, SliceGPT bù bằng nghiệm bình phương tối thiểu. Rất rẻ nhưng chỉ sửa được trong phạm vi một tầng.
5. **Rewinding / retrain từ đầu với mặt nạ**: nền của Lottery Ticket (08, 09) và "Rethinking the value of pruning" (11).

Quy tắc kinh nghiệm: ngân sách phục hồi (số token, số bước) quyết định kết quả nhiều không kém tiêu chí cắt — vì vậy cần so sánh có kiểm soát (mục 10).

---

## 6. Toán cần biết

Ký hiệu: $w\in\mathbb{R}^n$ là vector mọi tham số, $L(w)$ loss trung bình trên dữ liệu huấn luyện, $g=\nabla L(w)$, $H=\nabla^2 L(w)$.

### 6.1 Khai triển Taylor bậc hai của loss

Khi nhiễu loạn $w\to w+\delta$:

$$
\boxed{\ L(w+\delta)\ \approx\ L(w) + g^\top \delta + \tfrac12\,\delta^\top H\,\delta\ }
$$

Nếu mạng đã train xong và nằm gần một cực tiểu địa phương thì $g\approx 0$, nên

$$
\Delta L \approx \tfrac12\,\delta^\top H\,\delta .
$$

Ba điều cần nhớ:
- $H$ là ma trận $n\times n$ (n có thể là $10^9$) → không lưu nổi; phải xấp xỉ.
- $H\succeq 0$ tại cực tiểu: mọi nhiễu loạn chỉ làm loss tăng (hoặc giữ nguyên).
- "Gần cực tiểu" chỉ đúng một phần: mô hình lớn thường được train bằng ít epoch và chưa hội tụ hẳn, nên $g\neq 0$ và số hạng bậc nhất có thể quan trọng (lý do có các tiêu chí Taylor bậc nhất ở 4.2).

Cắt tham số $q$ nghĩa là $\delta_q=-w_q$ (còn các thành phần khác tự do chỉnh nếu ta cho phép).

### 6.2 Hessian và xấp xỉ Fisher / Gauss–Newton

Với loss là log-likelihood âm $L=-\frac1N\sum_n \log p(y_n\mid x_n;w)$, **ma trận thông tin Fisher** là

$$
F=\mathbb{E}\big[\nabla_w \log p\,\nabla_w \log p^{\top}\big]\ \approx\ \frac1N\sum_n g_n g_n^{\top},\qquad g_n=\nabla_w \ell_n .
$$

Khi mô hình khớp tốt phân phối dữ liệu thì $H\approx F$ (Hessian thật bằng Fisher cộng một số hạng chứa phần dư). Với bình phương sai số $\ell=\tfrac12\|t-F(w,x)\|^2$, bỏ số hạng phần dư $(t-o)$ cho **Gauss–Newton**:

$$
H\approx \frac1P\sum_{k} \frac{\partial F(w,x_k)}{\partial w}\ \frac{\partial F(w,x_k)}{\partial w}^{\top}.
$$

Lợi ích: nửa dương xác định (PSD), tính được từ gradient (xem bài OBS: đây chính là Eq. 10–12 của Hassibi & Stork, ghi chú 01). Với một tầng tuyến tính $y=Wx$ và loss bậc hai trên đầu ra, Hessian theo mỗi hàng $w$ của $W$ là $H = 2XX^\top$ (với $X\in\mathbb{R}^{d_{in}\times N}$) — dùng trong OBC/SparseGPT.

**Diagonal / K-FAC.** Chỉ giữ đường chéo $H_{ii}$ (OBD, 01), hoặc xấp xỉ khối theo Kronecker $H_\ell\approx A\otimes G$ (K-FAC; dùng trong LLM Surgeon, 27).

### 6.3 OBD: Optimal Brain Damage

Giả thiết: (i) cực tiểu ($g\approx0$), (ii) bậc hai, (iii) đường chéo (các hệ số chéo $H_{ij},i\ne j$ bị bỏ). Khi đó cắt $w_k$ làm loss tăng

$$
\boxed{\ s_k=\tfrac12\,H_{kk}\,w_k^2\ }
$$

Cắt các tham số có $s_k$ nhỏ nhất; sau đó train lại. Magnitude pruning là trường hợp $H\propto I$ (cong như nhau theo mọi hướng), khi đó $s_k\propto w_k^2$. Chi tiết ở ghi chú 01.

### 6.4 OBS: Optimal Brain Surgeon — dẫn xuất tóm tắt

Bỏ giả thiết đường chéo. Muốn đặt $w_q$ về 0 mà làm loss tăng ít nhất, **có quyền chỉnh mọi trọng số khác**:

$$
\min_{\delta}\ \tfrac12\,\delta^\top H\delta\quad \text{s.t.}\quad e_q^\top\delta + w_q = 0 .
$$

Lagrangian $\mathcal{L}=\tfrac12\delta^\top H\delta+\lambda\,(e_q^\top\delta+w_q)$. Đạo hàm theo $\delta$ bằng 0:
$H\delta+\lambda e_q=0\Rightarrow \delta=-\lambda H^{-1}e_q$. Thế vào ràng buộc:
$-\lambda\,[H^{-1}]_{qq}+w_q=0\Rightarrow \lambda=\dfrac{w_q}{[H^{-1}]_{qq}}$. Vậy

$$
\boxed{\ \delta w=-\frac{w_q}{[H^{-1}]_{qq}}\,H^{-1}e_q,\qquad \Delta L_q=\frac{w_q^2}{2\,[H^{-1}]_{qq}}\ }
$$

Hai kết luận: (1) saliency $\Delta L_q$ dùng $[H^{-1}]_{qq}$ chứ không phải $1/H_{qq}$ (hai đại lượng bằng nhau khi $H$ chéo); (2) công thức $\delta w$ **cập nhật mọi trọng số còn lại** để bù, không cần gradient descent. Khó khăn là tính $H^{-1}$, giải quyết bằng công thức đệ quy (Sherman–Morrison/Woodbury) trong paper gốc. SparseGPT/OBC khai thác lại đúng cấu trúc này theo từng hàng.

### 6.5 Tái tạo tầng bằng bình phương tối thiểu và nghiệm đóng

**Bài toán tầng (layer-wise).** Cho $W\in\mathbb{R}^{d_{out}\times d_{in}}$ và dữ liệu hiệu chuẩn $X\in\mathbb{R}^{d_{in}\times N}$ (mỗi cột là một đầu vào). Tìm $\hat W$ thỏa ràng buộc thưa/cấu trúc để

$$
\min_{\hat W}\ \|WX-\hat WX\|_F^2 = \sum_{r=1}^{d_{out}}\|(w_r-\hat w_r)X\|_2^2 .
$$

Bài toán tách theo từng hàng $r$. Gọi $H=XX^\top\in\mathbb{R}^{d_{in}\times d_{in}}$ (ma trận Gram, chính là Hessian đến hằng số).

**Mặt nạ cho trước, giữ tập chỉ số $S$** (các cột còn lại): ta chỉ có tự do ở $\hat w_S$. Điều kiện cực trị:

$$
\boxed{\ \hat w_S=\big(H_{SS}\big)^{-1}\,H_{S,:}\,w^\top\ =\ w_S + H_{SS}^{-1}H_{S\bar S}\,w_{\bar S}\ }
$$

(với $\bar S$ = tập bị cắt). Dạng thứ hai cho thấy trọng số còn lại *được cộng thêm* một phần bù từ trọng số bị cắt, tỉ lệ với tương quan giữa các đặc trưng. Nếu các cột $X_S$ và $X_{\bar S}$ không tương quan ($H_{S\bar S}=0$), không có gì để bù và nghiệm tối ưu giữ nguyên $w_S$. Thực tế thêm một số hạng ổn định $H\leftarrow H+\lambda I$ (damping).

**Bù bằng ánh xạ tuyến tính khi cắt cấu trúc.** Giả sử ta bỏ một số kênh của đặc trưng $Z\in\mathbb{R}^{d\times N}$ (đầu ra của khối bị cắt) và muốn tái dựng $Z$ từ phần giữ lại $Z_S$ bằng một ma trận $T$:

$$
\min_T\ \|Z-TZ_S\|_F^2\ \Rightarrow\ \boxed{\ T^\star=Z Z_S^{\top}\big(Z_SZ_S^{\top}\big)^{-1}=Z Z_S^{+}\ }
$$

Phần dư tối thiểu $\|Z-T^\star Z_S\|_F^2$ là thứ **không thể xóa bằng bất kỳ phép tuyến tính nào** — đó chính là ý nghĩa của $D_{res}$ trong cách nhìn của dự án, còn phần mà $T^\star$ xóa được là $D_{lin}$. (Lưu ý: đây là cách đặt bài toán của chúng ta; các bài như SliceGPT/ReplaceMe dùng những biến thể cụ thể của ý này.)

### 6.6 Nới lỏng Lagrangian và $\ell_0$ cho mặt nạ

Bài toán gốc: $\min_{w,m}\ L(m\odot w)$ s.t. $\|m\|_0\le k$, $m\in\{0,1\}^n$. Không khả vi. Các cách xử lý:

- **Phạt $\ell_1$** lên hệ số cổng: $\min L+\lambda\sum_i|\gamma_i|$ (05, Network Slimming). Đạo hàm dưới (subgradient) là $\lambda\,\mathrm{sign}(\gamma_i)$; $\gamma_i$ nhỏ → cổng gần như đóng → cắt.
- **Phạt $\ell_0$ khả vi:** biến ngẫu nhiên $z_i\in[0,1]$ ("hard-concrete") có xác suất khác 0 là $\sigma(\log\alpha_i-\beta\log(-l/r))$; kỳ vọng $\mathbb{E}\|z\|_0=\sum_i\sigma(\cdot)$ là hàm khả vi của tham số $\log\alpha_i$ (CoFi, 17).
- **Lagrangian cho ngân sách:**

$$
\min_{w,z}\max_{\lambda_1,\lambda_2}\ L(w,z)+\lambda_1(\hat s-t)+\lambda_2(\hat s-t)^2,
$$

  với $\hat s$ là độ thưa kỳ vọng và $t$ là mục tiêu; hai nhân tử $\lambda$ được học để *ép đúng ngân sách* thay vì chỉnh tay hệ số phạt.
- **Straight-through estimator (STE):** lan truyền gradient qua phép làm tròn/top-k như thể nó là hàm đồng nhất (movement pruning 16, MaskLLM 30 dùng biến thể).

### 6.7 Bỏ một neuron = bỏ một hàng và một cột

Xét MLP một tầng ẩn $h = W_{out}\,\sigma(W_{in}x+b)$ với $W_{in}\in\mathbb{R}^{m\times d}$, $W_{out}\in\mathbb{R}^{d\times m}$, $m$ neuron ẩn. Viết

$$
h=\sum_{j=1}^{m}\ (W_{out})_{:,j}\ \sigma\big((W_{in})_{j,:}x+b_j\big).
$$

Mỗi neuron $j$ đóng góp một số hạng độc lập: **hàng $j$ của $W_{in}$** (cộng bias $b_j$) tạo activation vô hướng, **cột $j$ của $W_{out}$** ghi nó vào residual. Bỏ neuron $j$ là xóa số hạng đó, tức *xóa hàng $j$ của $W_{in}$, phần tử $b_j$ và cột $j$ của $W_{out}$*. Kết quả là hai ma trận nhỏ hơn, dày đặc. Với MLP có cổng (gated, SwiGLU) $W_{down}(\sigma(W_{gate}x)\odot W_{up}x)$ thì xóa **hàng $j$ của $W_{gate}$ và $W_{up}$ cùng cột $j$ của $W_{down}$**.

Tương tự với CNN: bỏ filter $j$ của tầng $i$ → bỏ feature map thứ $j$ → bỏ kernel ứng với kênh $j$ ở tầng $i{+}1$ (04). Với attention: bỏ head $h$ → bỏ lát đầu ra của $W_Q,W_K,W_V$ và lát cột của $W_O$.

Số tham số của MLP tỉ lệ tuyến tính với $m$, nên giữ tỉ lệ $\rho=m'/m$ neuron thì tham số MLP giảm còn $\rho$ — đây là lý do cắt neuron có thể tính trước chính xác lượng nén.

### 6.8 Điểm hay nhầm: "độ thưa" ≠ "tỷ lệ nén"

Độ thưa 90% (10% khác 0) cho 10× ít tham số khác 0, nhưng lưu trữ thưa cần thêm chỉ số: mặt nạ bit tốn $1$ bit/phần tử bất kể độ thưa, CSR tốn một số bit cho mỗi phần tử khác 0 (xem 02, 03). Với trọng số INT4 thì chi phí chỉ số tương đối còn lớn hơn.

---

## 7. Vì sao structured cho tốc độ thật

**Dense GEMM trên GPU.** Phép nhân $Y=WX$ được chia ô (tiling) để dùng Tensor Core: các ô $16\times16$ hoặc tương tự được nạp và nhân liên tục. Hiệu năng đến từ truy cập bộ nhớ liền kề và tái sử dụng dữ liệu.

**Unstructured.** Đặt 90% phần tử về 0 mà giữ ma trận dạng dày đặc → *cùng số phép tính, cùng số byte*. Muốn tận dụng, phải dùng định dạng thưa (CSR/COO) và kernel SpMM. Khi mật độ còn cao (độ thưa dưới ≈ 90–95%) các kernel này thường chậm hơn dense trên GPU vì truy cập gián tiếp, mất coalescing, không dùng được Tensor Core hiệu quả. Chỉ ở độ thưa cực cao hoặc trên phần cứng chuyên dụng (như bộ tăng tốc EIE được nhắc ở 04 và 03) mới có lợi.

**Semi-structured 2:4.** Mẫu đều → phần cứng nén 2 giá trị khác 0 + 2 bit chỉ số/ nhóm, Tensor Core bỏ qua phép nhân với số 0. Lý thuyết tối đa ≈ 2× cho GEMM; thực tế thấp hơn vì phần còn lại của mô hình (attention, norm, memory bound) không nhanh lên.

**Structured.** Ma trận nhỏ hơn thật: ít byte phải đọc (giúp bộ nhớ-bound/batch-1), ít FLOPs (giúp compute-bound/batch lớn), activation nhỏ hơn. Không cần kernel đặc biệt. Điều kiện: kích thước mới nên là bội của 8/16/64/128 để tránh mất hiệu năng do căn chỉnh (alignment); cắt tới số lẻ có thể không nhanh hơn.

**Các cạm bẫy.**
- Cắt MLP không làm nhanh attention/KV cache; cắt layer làm giảm cả KV cache (với full attention), cắt width chỉ giảm KV nếu cắt số head KV.
- Độ trễ batch-1 còn có phần cố định (launch kernel, norm, embedding, LM-head). Một mô hình 0.8B có thể có phần đáng kể là embedding/LM-head — cắt thân transformer không ảnh hưởng chúng.
- Giảm *chiều sâu* (số layer) giảm độ trễ gần tuyến tính ở batch-1 vì các layer chạy tuần tự; giảm *chiều rộng* chỉ giảm thời gian đọc trọng số và có thể kém hiệu quả hơn khi kernel quá nhỏ.

---

## 8. Đánh giá

### 8.1 Chất lượng

- **Perplexity (PPL)** trên WikiText-2, C4...: $\mathrm{PPL}=\exp(\text{CE trung bình})$. Rẻ, nhạy, nhưng nhiều khi *không* phản ánh khả năng thực hiện tác vụ; mô hình cắt có thể có PPL gần mốc mà làm tác vụ sinh dài tệ hơn.
- **Zero-shot** (ARC, HellaSwag, PIQA, WinoGrande, BoolQ...): độ chính xác trắc nghiệm; nhiễu vài điểm ở mô hình nhỏ, cần cho biết số mẫu/seed.
- **Tác vụ sinh và suy luận dài** (MMLU, GSM8K, code, tác vụ đích): thường nhạy hơn với cắt.
- **Với dự án OCR**: tác vụ đích thật (edit distance ký tự/từ, CER/WER, độ chính xác trường thông tin trên tài liệu thật) quan trọng hơn mọi benchmark chung.

### 8.2 Hiệu năng

- Báo cáo **latency theo từng điều kiện**: batch-1 vs batch lớn; prefill vs decode; độ dài ngữ cảnh; độ chính xác số (FP16/INT8); GPU cụ thể; framework.
- Thực hiện **warm-up** vài lần, `torch.cuda.synchronize()` trước/sau khi đo, lặp nhiều lần và báo trung vị cùng phân vị cao; khóa xung nhịp nếu có thể.
- **Throughput** (token/giây ở batch lớn) chủ yếu bị chặn bởi compute; **latency batch-1** bị chặn bởi bộ nhớ. Một phương pháp có thể thắng một và thua cái kia.
- Đo **bộ nhớ đỉnh** (trọng số + activation + KV cache), không chỉ kích thước file.
- Đừng suy ra tốc độ từ FLOPs hay số tham số.
- Tính cả thời gian pipeline đầy đủ (tiền xử lý ảnh, ViT encoder, decoder, hậu xử lý) khi mô hình là VLM.

### 8.3 Chi phí để tạo ra mô hình

Báo cáo số token/giờ GPU dùng cho cắt + phục hồi. Hai phương pháp chỉ so sánh công bằng khi cùng ngân sách tính toán (mục 10).

---

## 9. Bảng thuật ngữ

| Thuật ngữ | Giải thích ngắn |
|---|---|
| Pruning | Cắt bỏ tham số/cấu trúc khỏi mạng |
| Sparsity (độ thưa) | Tỷ lệ tham số bằng 0; 90% nghĩa là còn 10% khác 0 |
| Mask (mặt nạ) | Tensor 0/1 chỉ ra phần nào còn giữ |
| Saliency | Điểm quan trọng; càng nhỏ càng đáng cắt |
| Unstructured | Cắt từng trọng số riêng lẻ |
| Structured | Cắt nguyên channel/neuron/head/layer |
| N:M (2:4) | Mỗi nhóm M có đúng N phần tử khác 0 |
| Global vs layer-wise | Ngưỡng chung toàn mạng vs ngưỡng theo từng tầng |
| One-shot | Cắt một lần đến mức đích |
| Iterative / gradual | Cắt nhiều bước nhỏ xen kẽ huấn luyện |
| Fine-tune / retrain | Huấn luyện tiếp sau khi cắt |
| Rewinding | Quay trọng số về một thời điểm sớm rồi train lại với mask (08) |
| Calibration data | Lượng nhỏ dữ liệu dùng để tính thống kê (Hessian, activation) |
| Hessian | Ma trận đạo hàm bậc hai của loss |
| Fisher | Kỳ vọng tích ngoại của gradient; xấp xỉ Hessian |
| OBD / OBS | Optimal Brain Damage / Surgeon (01) |
| Taylor criterion | Điểm dựa trên khai triển Taylor của loss |
| Reconstruction error | Sai số giữa đầu ra tầng gốc và tầng đã cắt |
| $\ell_0$ / $\ell_1$ | Số phần tử khác 0 / tổng trị tuyệt đối |
| STE | Straight-through estimator: gradient đi qua phép rời rạc |
| Channel / filter | Đơn vị đầu ra của conv; bỏ filter = bỏ channel đầu ra |
| Head | Một nhánh attention |
| Depth vs width pruning | Bớt số layer vs bớt chiều trong mỗi layer |
| KD | Knowledge distillation: học bắt chước teacher |
| Teacher / student | Mô hình lớn dạy / mô hình nhỏ học |
| PPL | Perplexity |
| Memory-bound / compute-bound | Bị giới hạn bởi băng thông bộ nhớ / bởi tốc độ tính |
| KV cache | Bộ nhớ khóa/giá trị của attention khi sinh |
| FLOPs vs MACs | Số phép tính dấu phẩy động; 1 MAC = 2 FLOPs |
| $D_{lin}$, $D_{res}$ | (Của dự án) phần tổn hại xóa được bằng phép tuyến tính tối ưu / phần còn lại |

---

## 10. Đọc paper pruning một cách phê phán

Lĩnh vực này nổi tiếng khó so sánh. Bài khảo sát thực nghiệm **"What is the State of Neural Network Pruning?" (Blalock et al., 2020; ghi chú 12)** tổng hợp 81 paper (theo trí nhớ của người soạn, hãy đối chiếu với ghi chú 12) và chỉ ra các vấn đề dưới đây. Hãy dùng như danh sách kiểm tra:

1. **Baseline có công bằng không?** Mô hình dense gốc có được train tốt không? Có so với một mô hình nhỏ cùng kích thước train đủ (03, 11, 43)? Có so với một baseline đơn giản (magnitude, random) được chỉnh cẩn thận?
2. **Ngân sách retrain.** Cắt A với 1 epoch và cắt B với 100 epoch thì không so được. Tìm: số bước/epoch/token, learning rate schedule, có dùng distillation hay không.
3. **Công bằng tính toán.** Phương pháp có cần Hessian, nhiều lần forward, huấn luyện mặt nạ? Tính cả giờ GPU của bước cắt.
4. **Metric nén.** Tham số hay FLOPs hay latency? Có tính chi phí lưu chỉ số cho dạng thưa? Cắt embedding có được tính?
5. **Dữ liệu và mô hình.** Chỉ thử trên MNIST/CIFAR hay trên ImageNet/LLM? Kết quả có chuyển được sang kiến trúc khác?
6. **Báo cáo độ biến thiên.** Có nhiều seed và khoảng tin cậy? Chênh lệch 0.1% có ý nghĩa không?
7. **Đường cong, không phải một điểm.** Một điểm (ví dụ "50% mà không giảm") không nói gì về đánh đổi; cần đường cong chất lượng–nén. Blalock et al. nhấn mạnh việc dùng nhiều mức nén và nhiều kiến trúc.
8. **Đo trên phần cứng đích.** Tốc độ lý thuyết ≠ tốc độ đo được; xem mục 7–8.
9. **Dữ liệu hiệu chuẩn và đánh giá có rò rỉ không?** Calibrate trên C4 rồi đánh giá trên C4 sẽ lạc quan.
10. **Tách "cái gì" và "cách phục hồi".** Hai phương pháp tiêu chí khác nhau nhưng khác cả cách retrain thì không biết cái nào tạo ra cải thiện.

Khi viết ghi chú cho từng bài, ta tách rõ **điều paper chứng minh / điều paper giả thuyết / điều ta suy ra**.

---

## 11. Lịch đọc gợi ý

Mỗi "tuần" là khoảng 3–5 bài; có thể đọc nhanh hoặc bỏ qua các bài mức "tham khảo" nếu thiếu thời gian.

**Tuần 1 — Nền móng cổ điển (CNN, bắt buộc).** 01 OBD/OBS → 02 Han 2015 → 03 Zhu & Gupta → 04 Pruning filters → 05 Network slimming. Mục tiêu: nắm saliency, magnitude, iterative, gradual, structured vs unstructured, học mặt nạ qua $\ell_1$.

**Tuần 2 — Tiêu chí và câu hỏi cơ bản.** 06 ThiNet (tái tạo) → 07 Molchanov (Taylor) → 08 Lottery Ticket → 09 Linear mode connectivity → 10 SNIP → 11 Rethinking the value of pruning → 12 State of pruning (đọc kỹ cho việc đánh giá phê phán).

**Tuần 3 — Transformer và cấu trúc.** 13 Sixteen heads → 14 Voita → 15 LTH cho BERT → 16 Movement pruning → 17 CoFi → 18 2:4 sparsity → 19 HALP → 20 DepGraph → 21 khảo sát nén LLM (làm bản đồ).

**Tuần 4 — Post-training cho LLM.** 22 OBC → 23 SparseGPT → 24 Wanda → 25 ZipLM → 26 LLM-Pruner → 27 LLM Surgeon → 28 FLAP → 29 SliceGPT → 30 MaskLLM.

**Tuần 5 — Cắt layer/depth (rất liên quan dự án).** 31 ShortGPT → 32 Unreasonable ineffectiveness of deeper layers → 33 Shortened LLaMA → 34 SLEB → 35 LaCo → 36 ReplaceMe → 37 Repair changes which layers to remove.

**Tuần 6 — Cắt + phục hồi quy mô lớn, kiến trúc lai.** 38 Sheared LLaMA → 39 Minitron → 40 Llama-3.1-Minitron → 41 Flextron → 42 Puzzle → 43 Small LLMs: pruning vs scratch → 44 Nemotron-H → 45 Minitron-SSM → 46 Mamba-Shedder → 47 Jet-Nemotron.

**Tuần 7 — Đa phương thức.** 48 ECoFLaP → 49 Structural pruning of VLMs.

**Đường tắt cho dự án (nếu có ít thời gian):** 01 → 02 → 04 → 12 → 23 → 29 → 31/32 → 36/37 → 38/39 → 43 → 49.

---

## 12. Năm câu tự kiểm tra

**Câu 1.** Một mô hình chạy batch-1 trên GPU 8 GB bị chặn bởi băng thông bộ nhớ. Bạn chọn giữa (a) độ thưa 50% unstructured và (b) cắt 30% neuron MLP có cấu trúc. Phương án nào chắc chắn giảm latency mà không cần kernel đặc biệt? Vì sao?

<details>
<summary>Đáp án</summary>

(b). Cắt neuron có cấu trúc làm các ma trận $W_{in}$, $W_{out}$ nhỏ lại thật sự, nên số byte phải đọc mỗi token giảm và dùng được GEMM dense bình thường. (a) giữ nguyên kích thước ma trận; muốn tận dụng phải có kernel thưa hoặc mẫu 2:4 + Tensor Core, và ở mật độ 50% các kernel thưa thông thường thường chậm hơn dense. Lưu ý (b) chỉ giảm phần MLP, không giảm attention/KV cache/embedding.
</details>

**Câu 2.** Từ khai triển Taylor bậc hai và giả thiết $g\approx0$, hãy dẫn ra saliency OBD $\tfrac12H_{kk}w_k^2$, và nói OBS khác ở điểm nào.

<details>
<summary>Đáp án</summary>

$\Delta L\approx\tfrac12\delta^\top H\delta$. Cắt tham số $k$ nghĩa là $\delta=-w_ke_k$ (chỉ thay đổi thành phần $k$, giả thiết đường chéo bỏ qua tương tác). Khi đó $\Delta L=\tfrac12H_{kk}w_k^2$. OBS không giả thiết chéo và cho phép các trọng số khác tự điều chỉnh: giải $\min\tfrac12\delta^\top H\delta$ s.t. $e_q^\top\delta=-w_q$ bằng Lagrange, được $\Delta L_q=w_q^2/(2[H^{-1}]_{qq})$ và $\delta w=-\frac{w_q}{[H^{-1}]_{qq}}H^{-1}e_q$ (cập nhật cả các trọng số còn lại).
</details>

**Câu 3.** Cho tầng $y=Wx$ và dữ liệu hiệu chuẩn $X$. Với tập cột giữ lại $S$ cố định, viết nghiệm tối ưu cho một hàng $\hat w_S$ và giải thích khi nào "bù trọng số" không cần thiết.

<details>
<summary>Đáp án</summary>

$\hat w_S=H_{SS}^{-1}H_{S,:}w^\top = w_S+H_{SS}^{-1}H_{S\bar S}w_{\bar S}$ với $H=XX^\top$. Nếu các đặc trưng bị cắt không tương quan với đặc trưng giữ lại ($H_{S\bar S}=0$) thì số hạng bù bằng 0 và $\hat w_S=w_S$: chỉ cần cắt, không cần chỉnh. Ngược lại, đặc trưng càng tương quan thì phần bù càng lớn và việc bù càng hữu ích.
</details>

**Câu 4.** Trong MLP SwiGLU, muốn bỏ neuron trung gian thứ $j$ thì phải xóa những phần nào của các ma trận?

<details>
<summary>Đáp án</summary>

Hàng $j$ của $W_{gate}$, hàng $j$ của $W_{up}$ và cột $j$ của $W_{down}$ (cùng bias nếu có). Ba lát này cùng tham gia tạo và ghi neuron $j$; bỏ thiếu một lát sẽ làm kích thước không khớp hoặc để lại phần tử vô nghĩa.
</details>

**Câu 5.** Một paper báo cáo "cắt 50% không mất chất lượng" bằng phương pháp mới và so với một baseline magnitude chỉ fine-tune 1 epoch trong khi phương pháp mới dùng 10 epoch + distillation. Hãy nêu ít nhất ba câu hỏi bạn sẽ đặt ra.

<details>
<summary>Đáp án</summary>

Ví dụ: (1) Ngân sách retrain/tính toán có bằng nhau không? Baseline được fine-tune cùng số bước và cùng distillation thì kết quả thế nào? (2) Chỉ số là tham số, FLOPs hay latency đo thực, trên phần cứng nào, batch nào? (3) Đây là một điểm hay đường cong nén–chất lượng, có nhiều seed không? (4) Dữ liệu hiệu chuẩn/đánh giá có chồng lấn không? (5) Kết quả có đúng trên mô hình/kiến trúc khác (Blalock et al., 12)?
</details>

---

## Kết luận ngắn

Nhớ năm ý: (1) FLOPs/tham số khác tốc độ thật; (2) structured cho ma trận nhỏ lại, unstructured thì không (trừ khi có phần cứng/kernel); (3) mọi tiêu chí đều xấp xỉ $\Delta L$ khi bỏ một thành phần, từ magnitude đến Hessian; (4) bù trọng số/phục hồi (fine-tune, KD) quyết định kết quả cuối không kém tiêu chí; (5) so sánh phải cùng ngân sách và đo trên phần cứng đích. Từ đây mở **01_1989_1993_optimal_brain_damage_surgeon.md**.
