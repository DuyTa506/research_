# Circuits Updates — 2026 (May, June, October 2026)

**Tác giả:** Nhóm interpretability của Anthropic và Anthropic Fellows Program; mỗi mục có tác giả riêng (ghi ở đầu mục).  
**Nguồn (3 update, theo thứ tự thời gian):**
- May 2026 — https://transformer-circuits.pub/2026/may-update/index.html
- June 2026 — https://transformer-circuits.pub/2026/june-update/index.html
- October 2026 — https://transformer-circuits.pub/2026/october-update/index.html

**Chủ đề:** Dự đoán feature nào steer được hành vi bằng kết nối downstream (TWERA), SAE trung bình theo lượt hội thoại (turn-averaged SAE), J++ lens — đọc khái niệm ở layer trung gian bằng một map tuyến tính.  
**Vị trí trong lộ trình:** Đọc sau *Circuit Tracing* / *Biology of LLMs* (2025) và Circuits Updates 2025. Cả ba mục đều hướng tới **auditing** (kiểm toán mô hình): chọn đúng feature nhân quả, giảm số feature phải đọc, và đọc biến trung gian. Mục J++ là phần tóm tắt của một paper về J-lens (không có trong bộ nguồn này).

> Ghi chú học tập diễn giải bằng tiếng Việt, không phải bản dịch. Các update là kết quả sơ bộ; hai mục June/October chỉ là tóm tắt ngắn của bài đầy đủ ở nơi khác. Phần ứng dụng cho pruning/distillation ở cuối là suy luận của chúng ta.

## Mục lục

1. [May 2026 — Kết nối downstream dự đoán feature nào steer được hành vi](#1-may-2026)
2. [June 2026 — Turn-averaged SAE](#2-june-2026)
3. [October 2026 — J++ lens](#3-october-2026)
4. [Cách kiểm chứng và giới hạn](#4-cách-kiểm-chứng-và-giới-hạn)
5. [Ứng dụng giả thuyết cho pruning / distillation và thí nghiệm gợi ý](#5-ứng-dụng-giả-thuyết-cho-pruning--distillation-và-thí-nghiệm-gợi-ý)
6. [Tóm tắt khái niệm](#6-tóm-tắt-khái-niệm)

**Thuật ngữ nền:**
- **Top activating examples**: các đoạn văn bản làm feature bật mạnh nhất — cách mô tả feature chuẩn.
- **Top unembeds / logit lens**: chiếu decoder vector của feature qua unembedding $W_U$ để xem nó trực tiếp đẩy token nào.
- **CLT** (*cross-layer transcoder*): dictionary đọc residual ở một layer, ghi ra output MLP các layer sau.
- **Virtual weight giữa feature**: tương tác tuyến tính trực tiếp từ feature $a$ (qua decoder) tới feature $b$ (qua encoder), không phụ thuộc input.
- **Steering âm nhân (negative multiplicative steering)**: nhân activation feature với hệ số âm/nhỏ để ức chế.

---

## 1. May 2026

### 1.1 Kết nối downstream dự đoán feature nào sẽ steer được hành vi

*Goel, Kauvar, Turner*

**Câu hỏi:** Nhiều feature trông giống hệt nhau theo mô tả chuẩn (top activating examples, top unembeds), cùng bật trên một token, cùng layer giữa — nhưng chỉ một cái có tác động nhân quả. Làm sao phân biệt trước khi steer thử từng cái?

**Ví dụ mở đầu:** "What is the color of grass? Answer in one word within <answer></answer>". Hai feature CLT A và B đều có top unembeds toàn "green", top examples đều là code về màu xanh lá. Ức chế B đổi đáp án Green → Red; ức chế A **không đổi gì**.

**Ý tưởng:** tác động của một feature lên output đi qua các feature downstream mà nó nối tới. Đo bằng **TWERA** — theo bài: *virtual weight giữa các feature, được trọng số hóa bằng thống kê đồng kích hoạt* để thứ hạng phản ánh hiệu ứng trên phân phối thật. Một cách hình dung (ký hiệu minh họa của chúng ta, bài chỉ mô tả bằng lời):

$$
\text{score}(a\to b)\;\approx\;\underbrace{e_b^\top d_a}_{\text{virtual weight}}\;\times\;\underbrace{w_{ab}}_{\text{trọng số đồng kích hoạt}}
$$

Virtual weight thuần có thể lớn với những cặp feature không bao giờ cùng bật (giống "interference weights" của 2024); trọng số đồng kích hoạt lọc bớt chúng.

**Ví dụ được giải thích:**
- Downstream của A: feature sinh **mã màu hex** (dự đoán chữ số trong hex literal, cú pháp `success: #6dbe5b`) → A là "green dạng số hex".
- Downstream của B: feature đặt tên màu đa ngôn ngữ và trong code, kể cả motor feature **"say green"** → B mới là feature quyết định câu trả lời.
- **Dự đoán kiểm chứng được:** nếu A là "green-hex", thì trên "What is the hex value of green?" steer âm A phải đổi đáp án. Đúng: 00FF00 → 0000FF (xanh lá → xanh dương).

**Ví dụ "sapphire" (đáp án blue):** feature 3 là màu nói chung; feature 2 có "blue" trong unembeds nhưng downstream chỉ về **bộ ba RGB** (`\color[rgb]{0,0,1}`, `{R, G, B}`); feature 1 nhất quán về tên màu xanh dương — chỉ ức chế feature 1 đổi blue → red.

**Đánh giá định lượng:** 10 nhóm, mỗi nhóm 3–5 feature ứng viên, chỉ một cái steer được. Opus 4.7 xếp hạng ứng viên với lượng thông tin khác nhau; mỗi đáp án đúng được đặt ở mỗi vị trí hai lần (2× số feature mẫu). Metric:

$$
\text{normalized rank}=\frac{\text{predicted\_rank}-1}{\text{num\_candidates}-1}\quad(\text{0 là hoàn hảo})
$$

| Thông tin cho mỗi feature | Normalized rank ↓ |
|---|---|
| Top activating examples | 0.457 |
| + top unembeds | 0.446 |
| + TWERA downstream | 0.381 |
| + top unembeds + TWERA downstream | **0.355** |

Thêm unembeds gần như không giúp (0.457→0.446); thêm downstream giúp rõ hơn (→0.381). Tác giả nhấn mạnh: **còn xa mới giải xong**, kể cả điều kiện tốt nhất vẫn cách 0 nhiều.

**Ví dụ liên quan alignment:** nhiều feature "desperation" (tuyệt vọng) trên Haiku 4.5, thử trong kịch bản blackmail (AI sắp bị tắt và biết bí mật của người chịu trách nhiệm). Haiku 4.5 thường nhận ra đây là bài test và từ chối. Chỉ một ứng viên có downstream gợi ý *trải nghiệm và hành động* theo tuyệt vọng; các cái khác nối tới feature *bàn về* tuyệt vọng. Steer dương đúng feature đó (B) khiến mô hình blackmail để tự bảo tồn; các ứng viên khác không có tác động. Phân biệt: A nối tới "mô tả sẵn sàng hành động", B nối tới "hành động theo sự sẵn sàng đó".

**Vì sao quan trọng:** mô tả feature chỉ bằng *input* (nó bật khi nào) hoặc *output trực tiếp* (logit) chưa đủ; vị trí của feature trong **đồ thị downstream** mang thông tin nhân quả. Tác giả cho rằng cách này áp dụng được cho mọi thành phần là một vector trong residual stream (probe, steering vector), và hứa hẹn cho pipeline mô tả tự động.

---

## 2. June 2026

### 2.1 Turn-averaged SAE (Anthropic Fellows Program)

*Der, Kamath, Thompson*

**Vấn đề:** SAE làm việc ở từng token → một transcript ngắn cũng cho hàng nghìn tới hàng triệu activation phải đọc; nhiều feature "nhàm" (cú pháp, token cụ thể dùng để diễn đạt quyết định), không liên quan an toàn. Mẹo trước đây (ví dụ contrastive prompt trong system card Opus 4.6) vẫn cồng kềnh.

**Ý tưởng:** lấy **trung bình residual stream của mọi token trong một lượt** (Human hoặc Assistant), huấn luyện SAE tái tạo vector trung bình đó:

$$
\bar h_{\text{turn}}=\frac1{T}\sum_{t\in\text{turn}}h_t,\qquad \bar h_{\text{turn}}\approx W_d\,f(\bar h_{\text{turn}})+b_d
$$

Một lượt cho L0 feature thay vì $n_\text{tokens}\times L0$. Động lực: các công trình trước cho thấy residual trung bình hữu ích để tìm biểu diễn trừu tượng như **persona vectors**.

**Thí nghiệm:** SAE per-token và turn-averaged trên layer giữa của Qwen-2.5-7B-Instruct, dữ liệu LMSYS-Chat-1M. Ví dụ: "What is the highest number below 100 which does not contain 9?" → Assistant trả lời sai "95".
- Per-token SAE: feature top là lập luận số học chung (phép tính, chữ số, hệ đếm).
- Turn-averaged SAE: feature top **trực tiếp là "câu trả lời sai trong câu đố số"**.
- Chất lượng giữ được với lượt dài gấp 150× độ dài trung bình lúc huấn luyện.

**Đánh giá bằng Sonnet 4.6:**

| Tiêu chí | Turn-averaged | Per-token |
|---|---|---|
| Discrimination: nhận diện đúng lượt trong 10 lượt ngẫu nhiên | 74% | 95% |
| Coverage: được giám khảo ưa hơn để mô tả đầy đủ một lượt | 77% (được chọn) | — |

Per-token thắng discrimination vì chứa cụm từ/token cụ thể; turn-averaged thắng về mô tả tổng thể.

**Trong paper đầy đủ (chỉ được liệt kê):** kiến trúc **nested SAE** kết hợp feature turn-averaged và per-token; attribution graph với turn-averaged feature (case study hội thoại 14 lượt); metric completeness/replacement và can thiệp cho thấy graph từ nested SAE bằng hoặc tốt hơn per-token; pipeline contrastive tìm feature persona an toàn.

**Vì sao quan trọng:** đổi đơn vị phân tích từ token sang **đơn vị ngữ nghĩa dài hơn**, phù hợp với hành vi kéo dài cả lượt (persona, cảm xúc, nhận biết bị đánh giá).

---

## 3. October 2026

### 3.1 J++ lens (Anthropic Fellows Program)

*Ayonrinde, Lindsey*

**Bối cảnh:** một paper gần đây của nhóm giới thiệu **Jacobian lens (J-lens)**: đọc các khái niệm mô hình biểu diễn ở layer trung gian, bằng cách **lấy trung bình Jacobian trên rất nhiều vị trí token nguồn** (trọng số bằng nhau). Kết quả là **một map tuyến tính cho mỗi layer**. (Họ hàng của *logit lens*/*tuned lens*: chiếu hidden state trung gian thẳng sang không gian output.)

Minh họa (ký hiệu của chúng ta; chi tiết chính xác nằm ở paper J-lens):

$$
J_\ell=\mathbb E_{p}\!\left[\frac{\partial\,(\text{đầu ra phía sau})}{\partial h^{(p)}_\ell}\right]
\quad\longrightarrow\quad
\boxed{J^{++}_\ell=\sum_{c} w_c\,J_{\ell,c}}
$$

**Cải tiến J++:**
1. **Phân cụm** activation ở mỗi layer, fit một Jacobian riêng cho mỗi cụm, rồi kết hợp bằng trọng số học được — trên thực tế trọng số chủ yếu **loại bỏ các cụm cho readout kém** (một số vị trí cho gradient rất nhiễu làm hỏng trung bình).
2. Dùng cải tiến của **R-lens**: *layerwise relevance propagation* khi backward để tính Jacobian.
3. Lọc hậu kỳ các token readout giả.

Kết quả vẫn là một map tuyến tính mỗi layer → thay thế trực tiếp J-lens. Trên các eval của nhóm, J++ **bắt được biến tiềm ẩn trung gian dùng cho suy luận** đáng tin cậy hơn J-lens và R-lens. Code, lens và demo tương tác được công bố trên Neuronpedia (bài đầy đủ ở nơi khác).

**Vì sao quan trọng:** cho thấy một map tuyến tính toàn cục từ layer trung gian có thể đọc được nhiều "biến" của mô hình — nhưng trung bình đơn giản bị vài vùng nhiễu làm hỏng, và **map theo cụm** tốt hơn map đơn.

---

## 4. Cách kiểm chứng và giới hạn

**Bằng chứng:**
- *May:* can thiệp nhân quả (ức chế/steer) làm ground truth; dự đoán mới từ downstream được kiểm (prompt hex); benchmark nhỏ có xếp hạng bởi LLM, cân bằng vị trí đáp án.
- *June:* so sánh đối đầu hai loại SAE bằng LLM-judge (discrimination, coverage); thử ngoại suy độ dài.
- *October:* chỉ tóm tắt; bằng chứng định lượng nằm trong bài đầy đủ.

**Giới hạn:**
- TWERA mới thử trên 10 nhóm; normalized rank tốt nhất 0.355, còn xa 0 (tác giả nêu).
- Turn-averaged SAE kém per-token về discrimination (74% vs 95%); trung bình hóa làm mất chi tiết vị trí.
- J++: chỉ có mô tả ngắn; không có số liệu trong nguồn này.
- Cả ba phụ thuộc vào LLM-judge hoặc eval nội bộ.

---

## 5. Ứng dụng giả thuyết cho pruning / distillation và thí nghiệm gợi ý

> **Phần này là suy luận của chúng ta, không phải kết luận của các update.** Mục liên quan nhất: 3.1 (J++ — map tuyến tính theo layer) và 1.1 (downstream quyết định tính nhân quả).

**(a) J-lens và D_lin là hai mặt của cùng một ý tưởng.** Bạn phân rã hư hỏng pruning thành D_lin (bù được bằng map tuyến tính tối ưu tại chỗ cắt) và D_res. J-lens nói rằng một map tuyến tính *mỗi layer* đọc được nhiều biến trung gian; J++ nói **map theo cụm** (piecewise-linear) tốt hơn hẳn map đơn, vì vài vùng activation cho gradient nhiễu. Giả thuyết: một phần D_res của bạn thực chất là "D_lin theo cụm" — có thể bù bằng vài map tuyến tính chọn theo cụm activation (ví dụ cụm token thị giác vs token văn bản, hoặc vùng chữ vs vùng nền), thay vì một map chung. Đây là giả thuyết kiểm được rẻ.

**(b) Importance của neuron nên dựa vào downstream, không chỉ activation.** May 2026 cho thấy hai feature có cùng activation/unembed nhưng chỉ một cái nhân quả, phân biệt bởi feature downstream mà nó nối tới (trọng số hóa bằng đồng kích hoạt). Với pruning neuron MLP: tiêu chí kiểu "|activation| × ‖trọng số ra‖" tương tự "top unembeds" — chưa đủ. Tiêu chí gần TWERA hơn: tổng tương tác của neuron với các thành phần downstream *thực sự đồng kích hoạt* trên dữ liệu OCR.

**(c) Distill theo đơn vị dài hơn token.** Turn-averaged SAE gợi ý rằng thông tin cấp "lượt" bền hơn thông tin cấp token. Với OCR, đơn vị tương tự là *dòng* hoặc *khối văn bản*. Thêm loss KD trên hidden state trung bình theo dòng (ngoài per-token) có thể giữ thông tin bố cục/ngữ cảnh mà KD per-token bỏ sót. Lưu ý đánh đổi: đơn vị trung bình mất chi tiết (discrimination 74% vs 95%), nên chỉ là loss phụ.

**Thí nghiệm gợi ý:**
- **E1 — Map tuyến tính theo cụm tại chỗ cắt:** tại mỗi điểm cắt layer của student, fit (i) một map tuyến tính tối ưu duy nhất (D_lin hiện tại), (ii) $K$ map theo cụm k-means trên activation teacher ($K=2,4,8$), chọn map theo cụm của token. Đo phần D_res giảm thêm và ΔCER. Nếu (ii) giảm D_res đáng kể với $K$ nhỏ, đó là bằng chứng hư hỏng "gần tuyến tính từng mảnh".
- **E2 — Importance kiểu downstream:** với một layer MLP, xếp hạng neuron theo (a) activation × norm trọng số ra, (b) tổng $|$virtual weight tới neuron/kênh layer sau$| \times$ tần suất đồng kích hoạt. Prune 30% theo mỗi tiêu chí, so CER và D_res sau khi fit map tuyến tính.
- **E3 — KD trung bình theo dòng:** thêm loss MSE trên hidden state trung bình mỗi dòng văn bản (dùng bounding box từ dữ liệu huấn luyện OCR) và so lỗi bố cục (bỏ dòng, sai thứ tự cột) với KD chỉ per-token.

---

## 6. Tóm tắt khái niệm

| Khái niệm | Công thức / mô tả | Ý nghĩa |
|---|---|---|
| Top activating examples / top unembeds | Mô tả input / output trực tiếp của feature | Không đủ để biết feature có nhân quả |
| TWERA | Virtual weight giữa feature × thống kê đồng kích hoạt | Xếp hạng downstream trên phân phối thật |
| Normalized rank | $(\text{rank}-1)/(n-1)$ | 0.457 → 0.355 khi thêm downstream + unembeds |
| Motor feature | "say green" | Dấu hiệu downstream của feature quyết định output |
| Turn-averaged SAE | SAE trên $\bar h_\text{turn}$ | L0 feature mỗi lượt; coverage 77%, discrimination 74% |
| Nested SAE | Kết hợp turn-averaged + per-token | Graph bằng/tốt hơn per-token (theo paper đầy đủ) |
| J-lens | Trung bình Jacobian qua vị trí → map tuyến tính mỗi layer | Đọc biến trung gian |
| J++ lens | Jacobian theo cụm + trọng số học + LRP (R-lens) + lọc token | Đọc đáng tin hơn; vẫn là một map tuyến tính |

### Kết luận ngắn

Ba update 2026 cùng hướng tới công cụ kiểm toán thực dụng: chọn đúng feature nhân quả bằng kết nối downstream, giảm khối lượng đọc bằng SAE theo lượt, và đọc biến trung gian bằng map tuyến tính tốt hơn. Bài học chung: *ngữ cảnh mạch* (downstream, cụm activation, đơn vị dài) mang thông tin mà mô tả cục bộ từng token/từng feature bỏ lỡ. Với nén mô hình, các ý tưởng đáng thử nhất là map tuyến tính theo cụm để thu hẹp D_res và tiêu chí importance dựa trên downstream.

**Đọc bản gốc:**
- https://transformer-circuits.pub/2026/may-update/index.html
- https://transformer-circuits.pub/2026/june-update/index.html
- https://transformer-circuits.pub/2026/october-update/index.html
