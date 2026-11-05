# Transformer Circuit Exercises, Garçon, and Transformer Circuit Videos (2021)

**Tác giả:** *Exercises* và *Videos* không ghi tác giả hay ngày trong văn bản trích xuất (cả hai là tài liệu đi kèm paper Framework). *Garçon*: Nelson Elhage (core contributor), Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, … , Jared Kaplan, Sam McCandlish, Chris Olah (Anthropic), đăng ngày 22/12/2021.  
**Nguồn:** https://transformer-circuits.pub/2021/exercises/index.html · https://transformer-circuits.pub/2021/garcon/index.html · https://transformer-circuits.pub/2021/videos/index.html  
**Chủ đề:** Bài tập tự viết trọng số attention head (virtual head, induction head kiểu pointer arithmetic và kiểu K-composition); hạ tầng Garçon để probe/ablate mô hình lớn; danh sách video thảo luận sớm.  
**Vị trí trong lộ trình:** Đọc ngay sau `02_2021_mathematical_framework.md`. Bài tập là cách nhanh nhất để kiểm tra bạn thực sự hiểu QK/OV và composition "tới từng tham số"; Garçon cho thấy hạ tầng cần có để làm interp ở quy mô lớn. Sau đó đọc `04` (induction heads).

> Ghi chú diễn giải, không phải bản dịch. Lời giải của Bài tập 1 trong bài gốc là hình ảnh; lời giải ở đây là của chúng ta (bài gốc nói "có nhiều đáp án đúng"). Ứng dụng cho pruning/distillation là suy luận của chúng ta.

## Mục lục

1. [Ba tài liệu này là gì](#1-ba-tài-liệu-này-là-gì)
2. [Exercises: khởi động](#2-exercises-khởi-động)
3. [Bài tập 1: virtual attention head nhìn lùi hai token](#3-bài-tập-1-virtual-attention-head-nhìn-lùi-hai-token)
4. [Bài tập 2: induction head kiểu pointer arithmetic](#4-bài-tập-2-induction-head-kiểu-pointer-arithmetic)
5. [Bài tập 3: induction head kiểu K-composition](#5-bài-tập-3-induction-head-kiểu-k-composition)
6. [Bài luyện tập có gợi ý](#6-bài-luyện-tập-có-gợi-ý)
7. [Garçon: hạ tầng probe mô hình lớn](#7-garçon-hạ-tầng-probe-mô-hình-lớn)
8. [Transformer Circuit Videos](#8-transformer-circuit-videos)
9. [Các tài liệu này kiểm chứng điều gì](#9-các-tài-liệu-này-kiểm-chứng-điều-gì)
10. [Giới hạn](#10-giới-hạn)
11. [Ứng dụng giả thuyết cho pruning / distillation](#11-ứng-dụng-giả-thuyết-cho-pruning--distillation)
12. [Thí nghiệm gợi ý](#12-thí-nghiệm-gợi-ý)
13. [Tóm tắt công thức / khái niệm](#13-tóm-tắt-công-thức--khái-niệm)

---

## 1. Ba tài liệu này là gì

| Tài liệu | Loại | Nội dung chính |
|---|---|---|
| Exercises | Bộ bài tập + lời giải | Viết tay $W_Q, W_K, W_V, W_O$ để hiện thực thuật toán; mục tiêu: hiểu cơ chế "tới từng tham số", không lướt qua chỗ mơ hồ |
| Garçon | Bài mô tả hạ tầng | Server bọc mô hình (kể cả nhiều node) để đọc/sửa activation qua "probe" |
| Videos | Trang trỏ tới playlist YouTube | Thảo luận rất sớm, không chính thức; tác giả dặn "take with a big grain of salt" |

## 2. Exercises: khởi động

Các câu khởi động (bài gốc trả lời bằng cách trỏ về paper Framework; đáp án ngắn dưới đây là của chúng ta, khớp với ghi chú 02):

| Câu hỏi | Đáp án ngắn |
|---|---|
| Ma trận nào chọn subspace được *đọc*, ma trận nào chọn subspace được *ghi*? | $W_V$ đọc (từ token nguồn), $W_O$ (bài gọi $W_{out}$) ghi (vào token đích); tích $W_OW_V$ là toàn bộ phép "đọc → ghi" |
| Hai trong bốn ma trận nào quyết định head attend đâu? | $W_Q$ và $W_K$ |
| Viết head chỉ bằng $W_Q^\top W_K$ và $W_OW_V$ | $h(x)=(A\otimes W_OW_V)x$, $A=\mathrm{softmax}^*(x^\top W_Q^\top W_Kx)$ |
| Hạng của hai ma trận này? | $\le d_\text{head}$ |
| $W_V^2W_O^1$ nói gì? Singular value? | Mức head 2 đọc output head 1 qua value (V-composition); singular value lớn ↔ hướng được truyền mạnh |

## 3. Bài tập 1: virtual attention head nhìn lùi hai token

**Bối cảnh (theo bài):** mô hình nhỏ thường có vài previous-token head nhưng không có head nhìn lùi hai token. Hai previous-token head ở hai layer có thể tạo một **virtual head** nhìn lùi hai token mà không phải hy sinh trọn một head. Thiết lập: residual 16 chiều, q/k/v 4 chiều, cả hai head đều attend token trước.

**Lời giải của chúng ta.** Ký hiệu $E_{a:b}$ là ma trận chọn các chiều $a..b-1$.

- (a) Head 1 copy chiều 0–3 sang 8–11:
$$W_V^1=\big[\,I_4\;\;0_{4\times12}\,\big],\qquad W_O^1=\text{ma trận }16\times4\text{ có hàng }8..11=I_4.$$
- (b) Head 2 copy 3 chiều mới của token trước (chọn 4, 5, 6) và chiều 8 (chiều head 1 đã ghi, chứa chiều 0 của token cách hai), ghi vào 12–15:
$$W_V^2=\begin{pmatrix}e_4^\top\\e_5^\top\\e_6^\top\\e_8^\top\end{pmatrix},\qquad W_O^2:\ \text{hàng }12..15=I_4.$$
Điều kiện then chốt bài gốc nêu: $W_V^2$ có **một** số 1 ở cột ứng với một hàng của $W_O^1$, và **ba** số 1 ở cột không bị $W_V^1, W_O^1$ chạm tới.
- (c) $W^1_\text{net}=W_O^1W_V^1$ có các số 1 tại $(8,0),(9,1),(10,2),(11,3)$; $W^2_\text{net}$ có số 1 tại $(12,4),(13,5),(14,6),(15,8)$. Chúng mô tả toàn bộ phép di chuyển thông tin: $h_i(x)=\sum_jA_{ij}W_\text{net}x_j$.
- (d) **Copy hai token:** $W^2_\text{net}W^1_\text{net}$ chỉ có một số 1 tại $(15,0)$ — chiều 0 của token $i-2$ tới chiều 15 của token $i$. **Copy một token:** $W^2_\text{net}+W^1_\text{net}$ gồm 7 số 1 còn lại.

$$
\boxed{\text{Hai head dành } 7/8 \text{ capacity cho copy 1 token và } 1/8 \text{ cho copy 2 token}}
$$

Đây là đúng nhận xét của bài: head thường phải dành toàn bộ capacity cho một việc; virtual head cho phép chia nhỏ.

## 4. Bài tập 2: induction head kiểu pointer arithmetic

Giả thiết: positional embedding sinusoid **có trong residual**. Ký hiệu $u^\text{cont}_k$ là các principal component của content embedding; $u^{\cos}_k,u^{\sin}_k$ mã hoá vị trí $n$ dạng $\lambda_k\cos(\alpha_kn)$, $\lambda_k\sin(\alpha_kn)$, độ lớn giảm dần.

| Phần | Mục tiêu | Lời giải (theo bài) |
|---|---|---|
| (a) | Attend token *nội dung giống* (kể cả chính nó) | $W_Q=W_K=[u_0^\text{cont};\dots;u_3^\text{cont}]$ |
| (b) | Tự chọn *vị trí hiện tại* | $W_Q=W_K=[u_0^{\cos};u_0^{\sin};u_1^{\cos};u_1^{\sin}]$ |
| (c) | Chọn *vị trí trước* | $W_K$ như (b); $W_Q$ = ma trận quay khối $R$ nhân với phép chiếu vị trí |
| (d) | Nội dung giống nhưng *tránh chính mình* | 8 chiều: 4 chiều nội dung + 4 chiều vị trí với hệ số $-\beta$ ở phía query |
| (e) | Head (d) lấy 8 chiều vị trí lớn nhất của token được attend, ghi vào $v_0,v_1,\dots$ | $W_V=[u_0^{\cos};\dots;u_3^{\sin}]$, $W_O=[v_0^\top\ v_1^\top\ \cdots]$ |
| (f) | Attend **token sau** lần xuất hiện trước của token hiện tại | $W_K$ chiếu vị trí; $W_Q$ quay *tiến* các chiều $v_0..v_3$ |

**Mẹo quay (phần c):** cặp $(\cos,\sin)$ của vị trí $n-1$ là cặp của vị trí $n$ quay một góc $-\alpha_k$:

$$
(u_k^{\cos},u_k^{\sin})\cdot x_{n-1}=\lambda_k R_{-\alpha_k}\big(\cos(\alpha_kn),\sin(\alpha_kn)\big),\qquad
R_\theta=\begin{pmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{pmatrix}.
$$

Nên $W_Q$ quay vị trí query lùi một bước để khớp key của token trước. Phần (f) làm ngược lại: head (d)+(e) đặt "vị trí của lần xuất hiện trước" vào subspace $v$; head (f) quay vị trí đó **tiến một bước** và so với vị trí trong key → attend đúng token kế tiếp. Đây chính là cơ chế pointer arithmetic paper Framework thấy ở GPT-2 (dùng **Q-composition**).

Bài cũng lưu ý: có thể nhân thêm một phép quay tùy ý vào cả $W_Q$ và $W_K$ mà head không đổi — lại là tính basis-free của q/k.

## 5. Bài tập 3: induction head kiểu K-composition

Với rotary attention, thông tin vị trí không đi qua $W_V$, nên mẹo (e)–(f) không dùng được. Cơ chế thay thế:

- **Head 1 (previous-token head):** $W_K^1$ chiếu vị trí, $W_Q^1$ quay *lùi* (như 2c) → attend token trước; $W_V^1=[u^\text{cont}_0;\dots;u^\text{cont}_3]$, $W_O^1=[v_0^\top\ v_1^\top\cdots]$ → ghi *nội dung token trước* vào subspace $v$ của vị trí hiện tại.
- **Head 2 (induction head):**
$$\boxed{W_K^2=[v_0;v_1;v_2;v_3],\qquad W_Q^2=[u_0^\text{cont};\dots;u_3^\text{cont}]}$$
Key = "token trước tôi là gì", query = "tôi là token gì" → khớp khi token trước vị trí nguồn trùng token hiện tại.

So sánh hai cơ chế:

| | Pointer arithmetic (BT2) | K-composition (BT3) |
|---|---|---|
| Cần vị trí trong residual? | Có | Không |
| Kiểu composition | Q-composition | K-composition |
| Thông tin được di chuyển | *Vị trí* của lần xuất hiện trước | *Nội dung* của token trước |
| Gặp ở | GPT-2 (theo paper Framework) | Mô hình trong paper Framework và Induction Heads |

## 6. Bài luyện tập có gợi ý

Các bài dưới đây lấy từ bộ exercises (P1–P4) hoặc mở rộng từ chúng (P5–P6, của chúng ta). Thử tự làm trước khi mở lời giải.

**P1. (Khởi động)** Cho $d_\text{model}=768$, $d_\text{head}=64$. $W_OW_V$ có kích thước và hạng tối đa bao nhiêu? Tính eigenvalue của $W_UW_OW_VW_E$ ($50{.}000\times50{.}000$) thế nào cho rẻ?

<details><summary>Gợi ý / lời giải</summary>

$768\times768$, hạng $\le 64$. Dùng $\lambda(AB)=\lambda(BA)$: với $A=W_UW_O$, $B=W_VW_E$, tính eigenvalue của $BA=W_VW_EW_UW_O$ cỡ $64\times64$.
</details>

**P2. (BT1 biến thể)** Làm lại Bài tập 1 nhưng muốn virtual head chiếm 2/8 capacity. Cần đổi gì ở $W_V^2$?

<details><summary>Gợi ý / lời giải</summary>

Cho $W_V^2$ đọc **hai** chiều trong 8–11 (chiều head 1 đã ghi) và chỉ hai chiều mới. Khi đó $W^2_\text{net}W^1_\text{net}$ có hai số 1.
</details>

**P3. (BT2c)** Viết $W_Q$ cho head attend token cách **hai** vị trí về trước, chỉ dùng cặp tần số $\alpha_0$.

<details><summary>Gợi ý / lời giải</summary>

Quay $2\alpha_0$ thay vì $\alpha_0$: dùng khối $R_{-2\alpha_0}$ theo cùng quy ước dấu với lời giải 2(c) của bài. Lưu ý tần số cao có thể bị "quấn" (aliasing) nếu $2\alpha_0$ gần $2\pi$.
</details>

**P4. (BT2d)** Vai trò của $\beta$ là gì? Điều gì xảy ra nếu $\beta=0$ hoặc $\beta$ rất lớn?

<details><summary>Gợi ý / lời giải</summary>

$\beta$ cân bằng "cùng nội dung" và "không phải vị trí hiện tại". $\beta=0$: head attend chính nó (vì nội dung giống nhất). $\beta$ quá lớn: phần vị trí lấn át, head chủ yếu tránh vùng gần chính nó bất kể nội dung.
</details>

**P5. (Mở rộng BT3)** Viết số hạng QK trong khai triển hai layer (ký hiệu ghi chú 02) mà cặp head ở BT3 tạo ra. Vì sao $W$ của nó có eigenvalue dương?

<details><summary>Gợi ý / lời giải</summary>

$\mathrm{Id}\otimes A^{h_1}\otimes\big(W_E^\top W_Q^{2\top}W_K^2W_O^1W_V^1W_E\big)$. Với lời giải BT3: $W_K^2W_O^1$ chiếu $v$ ngược về toạ độ 4 chiều, $W_V^1$ và $W_Q^2$ cùng là phép chiếu nội dung → $W\approx P^\top P$ (nửa xác định dương) trên subspace nội dung → eigenvalue $\ge0$: "same-matching".
</details>

**P6. (Mở rộng, liên quan pruning)** Trong BT3, nếu bỏ head 1 thì head 2 còn làm gì? Map tuyến tính per-token tại vết cắt có cứu được không?

<details><summary>Gợi ý / lời giải</summary>

Subspace $v$ trống → key của head 2 bằng 0 → attention đều (hoặc về vị trí mặc định), mất induction. Map per-token $\mathrm{Id}\otimes M$ không thể đưa nội dung token $j-1$ sang vị trí $j$, nên **không cứu được**: thiệt hại thuộc phần không tuyến tính-per-token.
</details>

## 7. Garçon: hạ tầng probe mô hình lớn

**Vấn đề.** Interp kiểu Circuits cần "chọc" vào ruột mô hình: đọc activation ở layer bất kỳ, sửa/ablate thành phần — và làm cả với mô hình lớn tới mức không vừa một GPU hay một máy. Notebook + PyTorch hooks ổn cho mô hình nhỏ, nhưng không có cách hiển nhiên khi mô hình trải nhiều node.

**Giao diện cốt lõi** (rút gọn từ bài):

```python
# khởi chạy server: chỉ cần tên và đường dẫn snapshot; cấu hình & số GPU tự phát hiện
#   python -m garcon.launch --snapshot PATH-TO-MODEL --name my-garcon
rmodel = garcon.connect("my-garcon")          # tiền tố "r" = remote
logits = rmodel.forward(rmodel.encode("Hello, world"))

def save_tensor(save_ctx, tensor):           # probe function: (save context, tensor)
    save_ctx.tensor = tensor                  # trả None -> giữ tensor gốc
rmodel.forward(tokens,
    probes=[rmodel.probe_points['layers.4.mlp.post_activation'].probe(save_tensor)])
rmodel.recordings()                           # dict theo tên probe point
```

- `garcon.launch_and_connect(path)` từ notebook: tự đặt tên theo đường dẫn (chạy lại cell thì nối lại server cũ), **tự tắt sau 1 giờ** không hoạt động.
- **Probe point:** điểm có tên trong mô hình nơi tensor được đọc/sửa; trả tensor mới để thay thế (quy ước giống PyTorch hooks). Hỗ trợ cả backward với loss do người dùng cho.
- Hàm probe được gửi lên server bằng **cloudpickle**. Server lo scatter/gather giữa các node.
- Ví dụ ablate neuron trong bài có lỗi chính tả nhỏ (tham số `mlp_idx` nhưng thân hàm dùng `unit`, thiếu ngoặc đóng) — ý tưởng: đặt `tensor[:, :, unit] = 0` tại `layers.{L}.mlp.pre_activation` (hoặc thay bằng mean activation).
- Truy cập tham số: `rmodel.get_parameter('layers.4.mlp.up_proj')`.

**Vì sao thiết kế "probe" (hơi rườm rà, có trạng thái):**

| Mục tiêu | Lý do |
|---|---|
| Chỉ lưu khi được yêu cầu | Lưu mọi activation của mô hình lớn là quá nhiều dữ liệu |
| Reduction ngay trên server | Chỉ lấy một vị trí token hoặc một thống kê (vd. L2 norm) → giảm băng thông, tận dụng song song GPU |
| Tính toán có trạng thái | `save_ctx` tồn tại qua nhiều lần chạy, vd. giữ "top 10 activation" cho mỗi neuron khi chạy hàng triệu ví dụ — nhanh hơn "nhiều bậc độ lớn" so với gửi activation về client |

**Wrapper và triển khai.** `easy_garcon` là giao diện đơn giản cho một context; probe được coi là "assembly language" để xây abstraction lên trên. Toàn bộ khoảng **1.000 dòng code** (dựa trên hạ tầng phân tán sẵn có): mỗi GPU một process chạy lockstep; rank 0 nghe socket, broadcast request, gather kết quả. Tensor lớn được gửi trực tiếp thay vì pickle; thí nghiệm "hàng chục GB" dữ liệu là bình thường.

**Tác động.** Dùng cho "gần như toàn bộ" công việc interp của nhóm; các nhóm khác cũng dùng. Lợi ích kể cả với mô hình nhỏ: giao diện nhất quán, server sống lâu giảm **"time to first plot"** (thuật ngữ mượn từ cộng đồng Julia). Triết lý: mọi mô hình Anthropic train đều sẵn cho nghiên cứu interp ngay, không chi phí bàn giao; kết hợp tính linh hoạt của notebook với quy mô cluster.

## 8. Transformer Circuit Videos

Trang chỉ có lời giới thiệu và link playlist YouTube (`PLoyGOS2WIonajhAVqKUgEMNmeq3nEeM51`); không liệt kê từng video. Danh sách dưới đây được ghép từ các link trong paper Framework và bài Garçon (chỉ số = vị trí trong playlist theo URL):

| # | Video | Ứng với |
|---|---|---|
| 1 | 0 layer theory | Framework §Zero-layer (ghi chú 02 §4) |
| 3 | 1 layer theory | Path expansion, QK/OV (02 §5) |
| 4 | 1 layer results | Skip-trigram, copying (02 §5–6) |
| 5 | 2 layer theory | Composition, khai triển QK (02 §7) |
| 6 | 2 layer term importance | Term importance (02 §9) |
| 7 | 2 layer results | Induction head (02 §8) |
| ? | Video về Garçon | Bài Garçon (§7) |
| — | Video về vài neuron MLP hiếm khi diễn giải được | Phần cập nhật 2023 của Framework |

Video Garçon có tham số playlist trong URL nhưng không rõ vị trí; video neuron MLP không có tham số playlist nên không chắc thuộc playlist. Hai video đầu đã được paper Framework thay thế.

## 9. Các tài liệu này kiểm chứng điều gì

Không có thí nghiệm hay số liệu mới. Exercises là **"chứng minh bằng xây dựng"**: viết được trọng số tay cho induction head theo hai cách cho thấy cơ chế của paper Framework khả thi, và có ít nhất hai thuật toán (Q-composition qua vị trí, K-composition qua nội dung). Garçon chỉ có báo cáo sử dụng nội bộ; bài Induction Heads sau đó ghi nhận việc thu dữ liệu (hơn 50.000 ablation) dựa nhiều vào Garçon.

## 10. Giới hạn

- Exercises dùng **mô hình lý tưởng hoá**: không LayerNorm, không superposition, subspace rời nhau sạch sẽ, sinusoid chính xác. Head thật "bẩn" hơn nhiều.
- Không có bài tập về MLP, và không có rotary thật (BT3 chỉ *giả định* vị trí không qua $W_V$).
- Garçon gắn chặt với hạ tầng Anthropic; không phát hành mã (bài chỉ mời các lab liên hệ). Thiết kế ưu tiên linh hoạt hơn hiệu năng.
- Danh sách video không đầy đủ (trang gốc không liệt kê; mục 2 của playlist không xác định được từ nguồn).

## 11. Ứng dụng giả thuyết cho pruning / distillation

> **Suy luận của chúng ta.**

**11.1. Tính $D_\text{lin}$ bằng probe có trạng thái.** Map tuyến tính tối ưu tại vết cắt cần thống kê $\Sigma_{xx}=\mathbb E[xx^\top]$, $\Sigma_{yx}=\mathbb E[yx^\top]$ giữa activation trước cắt $x$ và đích $y$:
$$
M^\star=\Sigma_{yx}\Sigma_{xx}^{-1},\qquad D_\text{lin}\text{ ứng với phần sai số được } M^\star \text{ loại bỏ}.
$$
Đây đúng là kiểu "stateful reduction on server" của Garçon: hook cộng dồn $\Sigma$ theo từng batch trên GPU, không bao giờ gửi activation đầy đủ về. Với PyTorch hooks thường cũng làm được y hệt.

**11.2. "Probe point" cho hybrid GatedDeltaNet.** Nên khai báo điểm probe cho cả *trạng thái recurrent* và *gate*, không chỉ residual/MLP; nếu không, các thí nghiệm kiểu đóng băng pattern (ghi chú 02 §9) không làm được cho lớp linear-attention.

**11.3. Bài tập như unit test cho tool pruning.** Dựng mô hình tay theo BT3 (previous-token + induction), rồi chạy pipeline prune/recover của bạn trên nó, ghi $D_\text{lin}/D$ khi cắt head 1, head 2, hoặc một chiều của subspace $v$. P6 dự đoán: cắt head 1 thì phục hồi per-token thất bại hoàn toàn. Nếu tool báo $D_\text{lin}$ lớn ở đây thì tool có lỗi.

## 12. Thí nghiệm gợi ý

1. **Làm BT1 và BT3 bằng code** (NumPy/PyTorch, ~50 dòng): dựng residual 16–32 chiều, chạy trên chuỗi ngẫu nhiên lặp, xác nhận induction attention. Sau đó thêm nhiễu Gaussian vào trọng số và đo khi nào cơ chế gãy — trực giác về độ bền dưới quantization/pruning.
2. **Mini-Garçon cho LightOnOCR:** một lớp bọc hook có tên probe point thống nhất (vision block, projector, từng decoder layer, trạng thái GatedDeltaNet), hỗ trợ reduction (top-k activation theo neuron, tích luỹ $\Sigma_{xx}$). Đo "time to first plot" trước/sau.

## 13. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / mô tả | Ý nghĩa |
|---|---|---|
| $W_\text{net}$ | $W_OW_V$; $h_i(x)=\sum_jA_{ij}W_\text{net}x_j$ | Toàn bộ phép di chuyển thông tin của head |
| Virtual head | $W^2_\text{net}W^1_\text{net}$ | Copy hai token với 1/8 capacity |
| Rotation trick | $x_{n-1}\leftrightarrow R_{-\alpha}x_n$ trên cặp (cos, sin) | Attend offset tương đối bằng phép quay |
| Pointer arithmetic | Copy vị trí → quay tiến (Q-composition) | Induction khi có vị trí trong residual |
| K-composition induction | $W_K^2=[v]$, $W_Q^2=[u^\text{cont}]$ | Induction khi dùng rotary |
| Probe | `probe_fn(save_ctx, tensor)` trả tensor mới hoặc `None` | Đọc/sửa activation từ xa |
| Stateful reduction | Tích luỹ trong `save_ctx` | Thống kê trên hàng triệu ví dụ |
| Time to first plot | Thời gian tới biểu đồ đầu tiên | Thước đo tooling interp |

### Kết luận ngắn

Bộ exercises biến các khái niệm của paper Framework thành bài tập tự xây trọng số: virtual head, và hai cách hiện thực induction head (pointer arithmetic bằng Q-composition, previous-token bằng K-composition). Garçon cho thấy interp ở quy mô lớn cần hạ tầng probe từ xa có reduction và trạng thái trên server. Với dự án pruning, cả hai dùng được ngay: mô hình tay từ bài tập làm unit test cho pipeline cắt-và-phục-hồi, còn thiết kế probe của Garçon là cách hiệu quả để tích luỹ thống kê tính $D_\text{lin}$.

**Đọc bản gốc:** https://transformer-circuits.pub/2021/exercises/index.html · https://transformer-circuits.pub/2021/garcon/index.html · https://transformer-circuits.pub/2021/videos/index.html
