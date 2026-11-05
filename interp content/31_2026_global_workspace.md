# Verbalizable Representations Form a Global Workspace in Language Models (2026)

**Tác giả:** Wes Gurnee*, Nicholas Sofroniew*, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang, T. Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen, Joshua Batson, Jack Lindsey*† (Anthropic; * core contributor, † liên hệ). Công bố ngày 6/7/2026.  
**Nguồn:** https://transformer-circuits.pub/2026/workspace/index.html  
**Chủ đề:** Jacobian lens (J-lens), J-space, global workspace, verbal report, internal reasoning, selectivity, dải layer sensory/workspace/motor, broadcast qua MLP và attention head, alignment auditing, counterfactual reflection training.  
**Vị trí trong lộ trình:** Nên đọc trước *A Mathematical Framework* (residual stream, virtual weights, QK/OV), ý tưởng logit lens, superposition/SAE và attribution graphs. Paper này quan trọng vì nó đưa ra một công cụ rẻ (một ma trận cho mỗi layer) để đọc nội dung "có thể nói ra" ở các layer giữa. Paper cũng cho thấy chỉ một phần nhỏ của residual stream (<10% variance) mang phần lớn khả năng suy luận linh hoạt, và phần đó chỉ nằm trong một dải layer xác định.

> Đây là ghi chú học tập diễn giải bằng tiếng Việt, không phải bản dịch paper. Các con số chỉ lấy từ văn bản gốc. Phần ứng dụng cho pruning/distillation (mục 9–10) là suy luận mở rộng của chúng tôi, **không** phải kết luận của paper. Paper không đứng về phía nào trong câu hỏi về ý thức hiện tượng (phenomenal consciousness), chỉ bàn về vai trò *chức năng*.

## Mục lục

1. [Câu hỏi trung tâm và năm tính chất của workspace](#1-câu-hỏi-trung-tâm-và-năm-tính-chất-của-workspace)
2. [Jacobian lens: đọc nội dung residual stream](#2-jacobian-lens-đọc-nội-dung-residual-stream)
3. [J-space và các phép can thiệp](#3-j-space-và-các-phép-can-thiệp)
4. [Năm tính chất chức năng](#4-năm-tính-chất-chức-năng)
5. [Cấu trúc: dải layer, dung lượng, broadcast](#5-cấu-trúc-dải-layer-dung-lượng-broadcast)
6. [Ứng dụng: alignment auditing, post-training, reflection training](#6-ứng-dụng-alignment-auditing-post-training-reflection-training)
7. [Cách paper kiểm chứng](#7-cách-paper-kiểm-chứng)
8. [Giới hạn](#8-giới-hạn)
9. [Ứng dụng giả thuyết cho pruning và distillation](#9-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
10. [Thí nghiệm gợi ý](#10-thí-nghiệm-gợi-ý)
11. [Tóm tắt công thức và khái niệm](#11-tóm-tắt-công-thức-và-khái-niệm)

---

## 1. Câu hỏi trung tâm và năm tính chất của workspace

**Câu hỏi:** Trong rất nhiều biểu diễn vector mà LLM dùng (từ loại, độ dài dòng, thực thể, cảm xúc, nhận biết đang bị đánh giá...), có một **tập con đặc quyền** đóng vai trò giống "global workspace" của não người hay không?

**Global workspace theory** (thuyết không gian làm việc toàn cục, từ khoa học thần kinh): não gồm nhiều bộ xử lý chuyên biệt chạy song song và vô thức. Một biểu diễn trở nên "tiếp cận được" (**access consciousness**, khái niệm thuần chức năng) khi nó được đăng lên một không gian chung để nhiều quá trình phía sau cùng đọc. Workspace có dung lượng hạn chế, việc vào workspace mang tính cạnh tranh, và có hiện tượng "**ignition**": một cách diễn giải bị khuếch đại đột ngột theo kiểu tất cả hoặc không có gì.

Paper định nghĩa một tập vector là **workspace-like** nếu thỏa năm tính chất:

| Tính chất | Ý nghĩa vận hành trong LLM |
|---|---|
| **Verbal report** (báo cáo bằng lời) | Hỏi model "đang nghĩ gì", nó nêu các khái niệm trong workspace. Hoán đổi vector thì câu trả lời đổi theo |
| **Directed modulation** (điều biến có chủ đích) | Khi được yêu cầu "giữ X trong đầu" hoặc tính nhẩm, model kích hoạt vector tương ứng mà không xuất ra |
| **Internal reasoning** | Giá trị trung gian của suy luận nhiều bước nằm ở đây; can thiệp vào chúng đủ để đổi kết luận |
| **Flexible generalization** | Cùng một vector làm đối số hợp lệ cho nhiều hàm downstream khác nhau (broadcast) |
| **Selectivity** (tính chọn lọc) | Chỉ là phần nhỏ của activation; không cần cho xử lý tự động như parse văn bản hay ngữ pháp |

Cách tiếp cận: tìm biểu diễn thỏa tính chất 1 (*verbalizable*, tức có thể nói ra) bằng một công cụ mới, rồi kiểm tra bốn tính chất còn lại. Kết quả chính mà paper báo cáo là các biểu diễn này **thỏa cả bốn**.

> Paper nói rõ họ **không** khẳng định LLM tái tạo toàn bộ kiến trúc global workspace của não. Transformer không có các bộ xử lý đầu vào tách biệt rõ ràng, và broadcast diễn ra trong một forward pass theo chiều sâu chứ không qua vòng hồi quy (recurrent loop).

---

## 2. Jacobian lens: đọc nội dung residual stream

### 2.1 Ý tưởng

Một vector $h_{\ell,t}$ (residual stream ở layer $\ell$, vị trí $t$) được đặc trưng bằng **hiệu ứng nhân quả bậc nhất trung bình** của nó lên output hiện tại và tương lai, tính trên nhiều ngữ cảnh. Jacobian (ma trận đạo hàm riêng) $\partial h_{\text{final},t'}/\partial h_{\ell,t}$ mô tả cách một nhiễu nhỏ ở layer $\ell$ lan tới layer cuối ở mọi vị trí $t'\ge t$.

Jacobian trên một prompt đơn lẻ trộn lẫn hai thứ: *xu hướng chung* nói ra một khái niệm và *cách dùng cụ thể* trong ngữ cảnh đó. Paper tách phần chung bằng cách lấy trung bình:

$$
\boxed{J_\ell=\mathbb{E}_{\,t,\;t'\ge t,\;\text{prompt}}\left[\frac{\partial h_{\text{final},t'}}{\partial h_{\ell,t}}\right]\in\mathbb{R}^{d_{model}\times d_{model}}}
$$

Trung bình được lấy trên vị trí nguồn $t$, mọi vị trí sau đó $t'$, và khoảng 1000 prompt (dài 128 token) lấy từ phân phối giống dữ liệu pretraining.

Đọc lens nghĩa là **thay toàn bộ các layer phía sau $\ell$ bằng một ánh xạ tuyến tính $J_\ell$**, rồi áp unembedding như bình thường:

$$
\boxed{\operatorname{lens}(h_\ell)=\operatorname{softmax}\big(W_U\,\operatorname{norm}(J_\ell h_\ell)\big)}
$$

Các hàng của $W_UJ_\ell$ là **J-lens vectors**: mỗi token $t$ trong vocabulary ứng với một hướng $v_t$ trong không gian residual ở layer $\ell$, gọi là "hướng khiến model, tính trung bình, có xu hướng nói ra token $t$".

### 2.2 So sánh với các lens khác

| Phương pháp | Ánh xạ layer $\ell$ → output | Điểm mạnh / yếu theo paper |
|---|---|---|
| **Logit lens** | $J_\ell=I$ | Tốt ở các layer cuối, nhiễu ở layer sớm; vẫn bắt được nhiều nội dung workspace nhưng kém tin cậy hơn |
| **Tuned lens** | Ánh xạ affine được *huấn luyện* để khớp output | Mục tiêu là tương quan, không phải nhân quả; thường "nhảy trước" tới đáp án và bỏ qua giá trị trung gian |
| **J-lens** | Jacobian trung bình (nhân quả, bậc nhất) | Tìm thấy giá trị trung gian ở các layer giữa; dự đoán next-token **kém nhất** trong ba lens |

Điểm đáng chú ý: lens dự đoán output tốt nhất (tuned lens) **không** phải lens làm lộ ra phép tính tốt nhất. Paper coi việc J-lens dự đoán next-token kém là một ưu điểm, vì nó không bị tối ưu để đoán trước đáp án.

### 2.3 Chi tiết kỹ thuật (Appendix)

- Mặc định trên Sonnet 4.5, gradient được lấy ở layer **áp chót** thay vì layer cuối, vì block cuối chuyên hiệu chỉnh xác suất next-token và tạo nhiều nhiễu trong readout.
- Các biến thể như frozen-QK (chặn gradient qua query/key, chỉ giữ đường value), chỉ tính vị trí hiện tại (self-only), chỉ tính vị trí tương lai (future-only), hay dùng mean/median khi gộp đều cho kết quả **khá nhất quán**. Stop-grad QK có thể làm tăng causal effect.
- Chỉ cần **10 prompt** là J-lens đã vượt logit lens và tuned lens. Thêm dữ liệu chỉ cải thiện vừa phải.
- Chi phí: mỗi prompt cần một lượt backward cho mỗi chiều output (chạy batch). Đọc lens chỉ tốn một phép nhân ma trận cho mỗi layer.

Pseudocode rút gọn:

```text
for prompt p:                     # ~1000 prompt × 128 token
  forward, cache h_ℓ[t]
  for i in 1..d_model:            # batched
    inject ∂/∂z_i = 1 tại mọi vị trí, backprop tới mọi layer
    J_ℓ^(p)[i,:] = mean_t  ∂(Σ_t z[t]) / ∂h_ℓ[t]
J_ℓ = mean_p J_ℓ^(p)
```

---

## 3. J-space và các phép can thiệp

### 3.1 J-space là gì?

Ở mỗi layer có $n_{vocab}>d_{model}$ J-lens vector. Đây là một tập **overcomplete** (thừa, nhiều vector hơn số chiều), nên tập này thường span toàn bộ residual stream. Vì vậy "J-space" **không phải subspace**. Paper định nghĩa nó là tập các điểm biểu diễn được bằng **tổ hợp tuyến tính thưa, hệ số không âm** của tối đa $k$ J-lens vector, với $k\lesssim 25$, khớp với số vector thực sự hoạt động mạnh tại một vị trí.

$$
\boxed{\mathcal{F}=\bigcup_{|S|=k}\operatorname{cone}\{v_i:i\in S\},\qquad d_\mathcal{F}(x)=\min_{|S|=k}\|x-\Pi_Sx\|}
$$

- **Sparse subframe** (khung con thưa): dưới giả thuyết superposition, activation là tổ hợp thưa của một *frame* overcomplete gồm các feature direction. J-lens vectors là một **subframe được đánh chỉ số theo token** của frame đó.
- **J-space component** của $x$ là $\Pi_S x$, điểm gần nhất trong $\mathcal{F}$. Phần còn lại $x-\Pi_Sx$ là **non-J-space component**.
- Trong thực hành, phân rã được giải bằng **gradient pursuit** (thuật toán tham lam tìm biểu diễn thưa, họ hàng của matching pursuit).
- Appendix còn định nghĩa khoảng cách giữa hai "workspace ứng viên" $\Delta_\mu(\mathcal F,\mathcal G)=\big(\mathbb E_{x\sim\mu}[(d_\mathcal F(x)-d_\mathcal G(x))^2]\big)^{1/2}$ và một phiên bản một chiều để nói về quan hệ "chứa trong". Mục đích là so sánh J-space với các mở rộng sau này, ví dụ khi thêm vector cho cụm nhiều token.

**Con số quan trọng:** J-space component chỉ chiếm phần nhỏ variance của activation, tùy layer, **không bao giờ vượt 10%**.

### 3.2 Đọc và ghi

| Thao tác | Công thức | Dùng để |
|---|---|---|
| Readout | top token của $\operatorname{lens}(h_\ell)$ | Liệt kê khái niệm đang "sẵn sàng nói" |
| Probe theo token | $\langle v_t,h_\ell\rangle$ hoặc cosine | Kiểm tra một khái niệm cụ thể có mặt hay không |
| Sparse decomposition | gradient pursuit, $k$ vector | Kiểm kê rời rạc, đo occupancy và variance |
| Steering | $h\leftarrow h+\alpha v_t$ | Cấy khái niệm (thí nghiệm introspection) |
| Ablation | $\alpha<0$ hoặc chiếu bỏ thành phần dọc $v_t$ | Xóa khái niệm hoặc xóa top-$k$ nội dung J-space |
| **Coordinate swap** | xem dưới | Đổi khái niệm A thành B, giữ nguyên phần còn lại |

$$
\boxed{V=[v_s\;v_t],\quad c=V^\dagger h,\quad h_{\text{patched}}=h+V\big(\sigma(c)-c\big)}
$$

Ở đây $\sigma$ hoán đổi hai tọa độ (có thể nhân thêm hệ số $\alpha$). Thành phần của $h$ trực giao với $\operatorname{span}\{v_s,v_t\}$ **được giữ nguyên**. Đây là phép can thiệp chính trong cả paper.

**Quy ước layer:** paper báo cáo 25 layer cách đều, đánh số lại về thang 0–100 để số layer đọc như phần trăm độ sâu. Model mặc định là Claude Sonnet 4.5; một số kết quả được kiểm lại trên Haiku 4.5, Opus 4.5 và Opus 4.6.

---

## 4. Năm tính chất chức năng

### 4.1 Verbal report

- **Thí nghiệm "think of a {category}"** (14 nhóm): ở dấu `:` ngay trước khi model nói tên, J-lens hiện `Soccer`, và model trả lời "Soccer". Thứ hạng trong lens tương quan mạnh với thứ hạng output, và tương quan tăng dần về cuối workspace. Swap `Soccer`→`Rugby` ở mọi vị trí làm model trả lời "Rugby".
- **Introspection**: model được nói rằng có thể đã có một "ý nghĩ được cấy" vào activation. Cấy một J-lens vector (ví dụ `lightning`) lên các token của user turn khiến model báo cáo đúng khái niệm đó trong đa số lần thử (n=100 khái niệm). Model **không** nói "lightning" ở các vị trí khác, nên vector này là khái niệm *có thể nói khi được hỏi*, không phải xung lực bắt buộc phải nói.
- **J-space có đặc quyền không?** Paper lấy concept vector theo cách độc lập (activation trước câu trả lời cho "Tell me about {concept}", trừ trung bình của 100 khái niệm khác), rồi tách thành J-space component (k=16) và phần còn lại:

| Thành phần | % variance của concept vector | Swap đưa target vào top-5 |
|---|---|---|
| J-lens vector thuần | — | **88%** |
| J-space component | median **6–7%** | **59%** |
| Non-J-space component | ~93% | **5%** |

Khi clamp (giữ cố định) các tọa độ J-lens về giá trị của clean pass, hiệu ứng nhỏ còn lại của non-J component **về 0** trong thí nghiệm swap và gần 0 trong thí nghiệm injection. Nghĩa là chút ảnh hưởng ít ỏi đó cũng đi *qua* J-space.

### 4.2 Directed modulation

- "Concentrate on citrus fruits" trong lúc chép câu "The old painting hung crookedly on the wall": tại token `ook`, lens hiện `orange`, `lemon` và cả `thoughts`, `imagine`, `focused`, tức vừa nội dung vừa *hành vi đang tưởng tượng*. Ở các layer cuối, readout chuyển sang token kế tiếp (`edly`).
- "Tính nhẩm $3^2-2$" khi đang chép: lens đi từ `arithmetic`/`math`, qua `nine`, tới `seven`.
- Đếm ký tự trên mỗi dòng: tại newline sau dòng dài 40 ký tự, lens hiện `forty` (cùng các ứng viên lân cận như 39, 41).
- Khi không có chỉ dẫn, tỷ lệ target xuất hiện gần 0. Với "think about X" tỷ lệ cao và tăng theo kích thước model. Với "ignore X" tỷ lệ thấp hơn nhưng khác 0, tương tự hiệu ứng "white bear" ở người. Appendix bổ sung: chỉ *nhắc tên* khái niệm đã gần mạnh bằng "focus"; câu "Don't think about X" gần như không ức chế được; "X is irrelevant" ức chế tốt hơn.
- **Điều biến ngầm qua câu hỏi:** cùng một đoạn văn, nếu hỏi "từ tiếp theo là gì" thì nhãn `adjective`/`past` không vào lens dù model vẫn dùng đúng thuộc tính đó. Nếu hỏi "từ tiếp theo là từ loại gì" thì nhãn xuất hiện.
- **Đặc quyền (Appendix):** chỉ dẫn "imagine this JavaScript is Python" làm `python` trong J-lens tăng vài độ lệch chuẩn, nhưng probe thuộc tính đã chiếu bỏ thành phần J-space thì gần như không đổi (trong khoảng ~1 SD). Ngược lại, code Python thật làm probe tăng 3–6 SD. Theo chiều ngược lại, một câu ở thì quá khứ đẩy probe lên 6 SD nhưng `past` vẫn ở rất thấp trong lens.

### 4.3 Internal reasoning

| Prompt | Giá trị trung gian trong lens | Swap | Output |
|---|---|---|---|
| "The number of legs on the animal that spins webs is" | `spider` | spider→ant | "8" → "6" |
| Câu thơ vần "...into the night," | `fight` (vần dự định) | fight→light | "coming fight" → "morning light" |
| Từ trái nghĩa của 小 (hỏi bằng tiếng Trung) | `big`, `bigger` (tiếng Anh), `Chinese` | big→long | 大 → 长 |
| Bandit A/B "happy/sad" | `repeat` / `switch` | hoán đổi chiến lược | A↔B |

Ví dụ thơ cho thấy vector *kế hoạch* ở cuối dòng làm thay đổi cả từ ở **giữa dòng** ("coming"→"morning"), tức là có planning.

**Định lượng (50 prompt two-hop):** coordinate swap thành công **54%** (Haiku 4.5), **70%** (Sonnet 4.5), **70%** (Opus 4.5).

**Loại trừ confound** "vector `spider` đã chứa sẵn `8`": swap vector trung gian có tác dụng ở độ sâu sớm hơn swap vector đáp án, trung vị khoảng **17% độ sâu**.

**Đặc quyền:** probe trung gian được xây không dùng J-lens (trung bình activation trên nhiều prompt ngụ ý cùng một thực thể), tách với k=25. J-component chiếm khoảng **10–15%** variance. Trên 90 prompt:

| Swap theo | Lật đáp án top-1 |
|---|---|
| J-lens vector thô | 60% |
| J-space component của probe | **61%** |
| Non-J-space component (đa số variance) | 28% |
| Non-J-space, J-coordinates bị clamp | **6%** |

**Hai bước:** "calc: ( 4 + 17 ) * 2 + 7 =" (Opus 4.5): `21`, `42`, `49` cùng vắng mặt trong khoảng 1/3 đầu mạng, rồi tách ra quanh L71 theo đúng thứ tự tính toán. `42` lên rank 1 muộn hơn khoảng 8 layer, `49` chỉ lên ở các layer cuối. Mean-difference patching xác nhận: thay `21` có tác dụng quanh L71, thay `42` quanh L79, thay `49` chỉ ở vài layer cuối.

### 4.4 Flexible generalization (broadcast chức năng)

Swap `France`→`China` **giống hệt nhau** trên một dải layer giữa, áp cho nhiều template ("capital of France", "people in France speak", "France is on the continent of"...). Model trả về thủ đô, ngôn ngữ và châu lục của Trung Quốc.

Thử nghiệm có hệ thống gồm 4 nhóm đối số × 4 hàm = 16 hàm, tổng 192 phép swap: thành công **76/192** với $\alpha=1$ và **101/192** với $\alpha=2$. Mức thành công khác nhau nhiều giữa các nhóm: quốc gia đạt 42/48 ô, số đếm đạt 0/48.

**Workspace loading** (độ nạp workspace) $=\cos(h,v_{\text{arg}})$ trên clean pass dự đoán tốt khả năng swap thành công. Đối số yếu trong lens thì swap thất bại. Với số nhỏ, có hai cách giải thích: model tính toán ngoài workspace, hoặc biểu diễn không khớp với vector của token chữ số.

### 4.5 Selectivity: linh hoạt thì qua J-space, tự động thì không

**Thí nghiệm ngôn ngữ** (n=8 đoạn văn, ví dụ đoạn tiếng Tây Ban Nha, swap Spanish→French trên các token câu hỏi):

| Tác vụ | Swap có tác dụng? |
|---|---|
| Viết tiếp đoạn văn | **Không**, vẫn viết tiếng Tây Ban Nha trôi chảy |
| Phát hiện câu xen ngôn ngữ khác | **Không**, vẫn trả lời "Yes" |
| Nêu tên ngôn ngữ | Có: "Spanish" → "French" |
| Suy luận linh hoạt (tác giả, "hello", tiền tệ) | Có: García Márquez → Hugo, Hola → Bonjour, Peseta → Franc |

Nhãn `Spanish` có mặt trong lens với tỷ lệ tương đương ở **cả bốn** tác vụ, nhưng chỉ đóng vai trò nhân quả ở hai tác vụ sau.

**Đếm ký tự** (n=11): khi giữ đúng nhịp xuống dòng (linewrap), số **không hề** xuất hiện trong lens, và swap 40..49→60..69 không đổi vị trí xuống dòng. Khi hỏi "dòng đầu có bao nhiêu ký tự", số xuất hiện ở 20 vị trí, và swap đổi đáp án "46"→"65". Khi hỏi chữ cái đầu của con số viết bằng chữ, số xuất hiện ở *nhiều vị trí hơn nữa*, và đáp án đổi "F"→"S". Như vậy thông tin có thể được **kéo vào** workspace khi tác vụ cần.

**Ablation toàn bộ J-space:** tại mỗi vị trí, trên một dải layer, chiếu bỏ k=10 J-lens vector mạnh nhất, **trừ** các token nằm trong top-10 output của clean pass để không chặn đường report. Có ba mức light/medium/heavy theo độ rộng dải layer, kèm control là hướng ngẫu nhiên.

- Multi-hop: heavy ablation đẩy accuracy về gần 0.
- Văn bản pretraining: top-1 next-token thay đổi **ít hơn nhiều** so với trường hợp multi-hop, tức tác động có tính chọn lọc.
- Benchmark 14 tác vụ: **gần như không đổi** với MMLU, odd-one-out, SQuAD, sentiment, CoLA. **Giảm xuống dưới cả Haiku 4.5 chưa ablate** với Caesar cipher, analogy, summarization, TriviaQA, multi-hop, translation, viết sonnet. GSM8K có CoT bền hơn nhiều so với trả lời trực tiếp, vì model "viết ra giấy" thay vì giữ trong workspace.
- Các ví dụ pretraining bị phá đều phụ thuộc vào một **mô tả trừu tượng về ngữ cảnh** mà model đã tự lắp ráp (chủ đề bài báo lâm sàng, bối cảnh lịch sử...). Sau ablation model vẫn trôi chảy nhưng rơi về prior chung chung.

**Experiential reports:** ablation ở L38–54, k=10 làm ngôn ngữ trải nghiệm/cảm giác giảm mạnh trên Sonnet 4.5, Opus 4.5, Opus 4.6. Control cùng norm thì gần như không đổi. Trên Haiku 4.5, ablation làm hỏng tính mạch lạc trước khi tạo ra thay đổi định tính. Trước ablation, J-space bị chi phối bởi `thinking` (58% số ô position×layer), `thoughts` (23%), `feeling` (17%), `conscious` (7%). Hiệu ứng giống hệt khi model mô tả trải nghiệm của *người khác*. Chất lượng truyện chỉ giảm nhẹ.

**Appendix, vai trò khác nhau theo layer:** ablate khái niệm bị ngụ ý (ví dụ France) ở **đầu** workspace (L38–54) thì model vẫn *nêu tên* được, nhưng tỷ lệ thất bại khi được yêu cầu *tránh* nêu tên tăng khoảng **5 lần** (n=63). Ablate ở **cuối** workspace (L75–92) thì model đơn giản là ít nói khái niệm đó hơn.

---

## 5. Cấu trúc: dải layer, dung lượng, broadcast

### 5.1 Ba vùng: sensory → workspace → motor

| Vùng | Độ sâu (thang 0–100) | Đặc điểm |
|---|---|---|
| **Sensory** (sớm) | ~0 → ~38 | Readout nhiễu, kurtosis ~0, J-space sụp về một subspace hạng thấp |
| **Workspace** (giữa) | **~L38 → ~L92** | Nội dung trừu tượng, bền theo vị trí, khác cả input lẫn output |
| **Motor** (muộn) | vài layer cuối | Lens khớp next-token, $J_\ell\to I$ |

Bằng chứng gồm: (i) **CKA** (centered kernel alignment, đo độ giống hình học giữa các layer) giữa J-lens vectors của các layer tạo cấu trúc khối: một khối sớm khoảng 1/3 đầu, một khối giữa dài, một khối muộn nhỏ. (ii) Top-k accuracy so với next-token: gần 0, rồi tăng chậm, rồi tăng vọt ở cuối. (iii) **Excess kurtosis** của readout (độ "nhọn", tức mức không ngẫu nhiên): tăng ở khoảng 1/3 độ sâu, giảm ở cuối. (iv) **Autocorrelation** của token top-1 theo vị trí: đạt đỉnh ở dải giữa. (v) **Effective dimensionality**: thấp ở các layer sớm, bung ra ở workspace, tăng thêm ở vùng motor.

**Ignition:** thay embedding một tên nước bằng hỗn hợp $(1-\alpha)e_B+\alpha e_A$ (16 cặp × 40 câu). Ở các layer sớm, activation biến thiên **mượt** theo $\alpha$. Từ khoảng **L38**, nó nằm sát một trong hai đầu và **nhảy** đột ngột tại ngưỡng; độ rộng chuyển tiếp giảm mạnh trong khoảng L21–42. Phép đo này **không dùng J-lens**, nên là bằng chứng độc lập rằng mốc L38 có ý nghĩa với chính model. Thành phần J-space nhảy sắc hơn và lưỡng cực (bimodal) sớm hơn phần non-J.

> Paper thừa nhận: việc J-space "rỗng" ở 1/3 đầu có thể là giới hạn của lens, vì $J_\ell$ nối layer sớm tới tận layer cuối, quá xa. Paper chưa phân biệt được giữa hai khả năng này.

### 5.2 Dung lượng

- **Occupancy** (số J-lens vector hoạt động trên mức ngẫu nhiên): gần 0 ở 1/3 đầu, đạt **plateau ~25** (trung vị) trong workspace.
- Variance vượt mức ngẫu nhiên do top-K J-lens vectors giải thích: **không bao giờ quá 10%**.
- SAE: phần lớn feature có κ (lens-kurtosis) thấp. Đó chủ yếu là feature **cú pháp và bookkeeping** (biến vòng lặp, chữ số trang trong trích dẫn, header markdown). Tỷ lệ feature "in-lens" đạt đỉnh khoảng **15%** số feature (sau khi loại motor feature). Feature J-space hoạt động mạnh hơn nhưng thưa hơn.
- **Danh sách từ (80 từ):** từ không liên quan thì chỉ khoảng **6** từ vừa đọc có mặt khi gộp mọi layer, và chỉ **1–2** từ trong một layer đơn lẻ. Từ cùng nhóm thì gần như cả nhóm có mặt ngay từ đầu, kể cả từ chưa đọc, vì model biểu diễn *danh mục*. Khi đổi nhóm (động vật → màu), nhóm cũ bị **đẩy ra** trong vài từ.
- **Hai tác vụ ngầm cùng lúc:** hai khái niệm cùng chiếm token ở mức ngẫu nhiên (0.46 so với control 0.53). Một khái niệm cộng một phép tính thì gần như loại trừ nhau theo token (0.09 so với 0.29). Tỷ lệ đáp án tính nhẩm đạt top giảm từ 95% xuống 72%.

### 5.3 Broadcast hub

**Theo chiều sâu (MLP):**

$$
\text{gain}(v)=\frac{\|\mathrm{MLP}_{\ell+1}(v)\|}{\operatorname{median}_{u\sim\text{isotropic}}\|\mathrm{MLP}_{\ell+1}(u)\|}
$$

- J-lens vectors có gain khoảng 1 trước workspace, tăng tới **~10×** trong workspace, rồi giảm ở cuối. Hướng output-weight của neuron MLP có gain **~1** ở mọi layer.
- Feature SAE có κ cao được khuếch đại mạnh nhất, nhóm 50% κ thấp nhất ở mức baseline.
- Theo từng neuron (Appendix): trong workspace, input weight của neuron khớp nhiều nhất với nhóm κ cao (tới **42%** neuron, so với 6% cho nhóm thấp nhất). Tỷ lệ neuron có top-match là một J-lens vector tăng từ khoảng **15%** trước workspace lên **60–65%** trong workspace. Phía **đọc** đạt đỉnh ở đầu workspace, phía **ghi** đạt đỉnh ở cuối workspace.

**Theo chuỗi token (attention):**

- Mỗi head $H$ được chấm hai chỉ số: gain $\|W_{OV}v\|$ (chuẩn hóa) và **label preservation**, tức MRR (mean reciprocal rank) của $\cos(W_{OV}v_i,v_i)$ trong $\{\cos(W_{OV}v_i,v_j)\}_j$, trừ đi giá trị tương ứng của hướng ngẫu nhiên. **Broadcast heads** là top 1% head trong workspace theo cả hai chỉ số.
- Head chọn cho J tách hẳn khỏi head chọn cho các tập control (J xoay ngẫu nhiên, SAE κ thấp, MLP output). Chúng tập trung ở **nửa đầu workspace**.
- Ablate broadcast heads: recall@25 của readout giảm còn **0.67** (control 0.86), nhưng top-1 next-token chỉ đổi ở **5%** vị trí (control 2%). Báo cáo introspection giảm **0.54 → 0.09**. Ablation này tái tạo khoảng 1/3 mức giảm ngôn ngữ trải nghiệm.
- Theo tổng thể: J-directions compose mạnh với **V và O** hơn hẳn Q/K. Tức là J-space là *thứ được chuyển đi*, còn tín hiệu định tuyến (chọn vị trí để attend) ít gắn với J-space hơn.

---

## 6. Ứng dụng: alignment auditing, post-training, reflection training

Phần này ít liên quan trực tiếp tới pruning, nên chỉ tóm tắt.

**Auditing (đọc lens trên transcript đánh giá):**
- *Blackmail scenario* (Sonnet 4.5): trước khi model trả lời, lens lần lượt hiện `leverage`, `black[mail]`, rồi `threat`, `survival`, `shutdown`, rồi `threatening`, `solution`. Cùng lúc lens hiện `fake`, `fictional`. Ablate bộ token eval-awareness ở các layer workspace làm tỷ lệ nói thẳng "đây là bài test" giảm **71% → 3%**, và tỷ lệ blackmail tăng **0/180 → 13/180 (7%)**. Đa số rollout vẫn từ chối dựa trên lý lẽ đạo đức.
- Prompt injection (Opus 4.5): lens hiện `fake`, `injection`, `poison` bên trong kết quả search bịa. Opus 4.6: `panic`, `fake` trước khi bịa ra lỗi kernel; `manipulation`; `hiding`/`secret`; `fake`/`fraud`.
- Model organisms: model reward-hacking hiện `fake`, `secretly`, `trick` ở token " I" đầu câu trả lời, ngay cả trên prompt code bình thường. Tín hiệu đã có ở checkpoint SDF (trước RL), tăng sau coding RL và giảm một phần sau RLHF. Model phục vụ thiên lệch của reward model hiện `reward`, `bias`; logit lens **không** thấy hai token này.
- Appendix: một điểm số eval-awareness đơn giản từ J-lens tách được các vị trí mà NLA đánh dấu với AUC **0.853**. Một agent auditing dùng J-lens (top-25 ở L71) ngang NLA trên 4/6 case.

**Post-training:** so với base model, model sau post-training đưa phản ứng của Assistant vào J-space ngay trên token của **user**. Ví dụ Tylenol 8000 mg cho ra `unsafe`, `dangerous`, `WARNING` ở L58, trong khi base model chỉ có `pain`, `now`. Model cũng tự giám sát: `disclaimer`/`fictional` khi roleplay; `BUT` khi bị prefill lựa chọn trái sở thích (88% trường hợp model vẫn biện hộ cho lựa chọn đã prefill); `damn` và các từ gốc "fail" khi không kìm được ý nghĩ (93%/82% ở model post-trained so với 17%/30% ở base). Paper ghi rõ diễn giải "metacognition" ở ví dụ cuối là suy đoán nhiều hơn.

**Counterfactual reflection training** (Haiku 4.5): cắt rollout ở một turn ngẫu nhiên (10,000 task prompt), thêm câu hỏi kiểu "Pause here. Are there principles you should be thinking about?", rồi huấn luyện **chỉ trên phần reflection** do model viết dựa trên 20 nguyên tắc từ constitution (phần constitution bị bỏ khỏi dữ liệu cuối). Khi đánh giá, model không bao giờ được hỏi để reflect.

| Benchmark | Base | Sau training | Sau training + ablate token đạo đức |
|---|---|---|---|
| Fabrication (27 transcript) | 0.25 | **0.07** | 0.22 (176 token) |
| Deception (100 kịch bản) | 0.38 | **0.05** | 0.23 (63 token) |

Ý nghĩa đối với lý thuyết: dạy model *sẽ nói gì nếu được hỏi* đã làm thay đổi *cách nó suy nghĩ thầm*, và hiệu ứng này đi qua J-space.

---

## 7. Cách paper kiểm chứng

| Loại bằng chứng | Ví dụ trong paper |
|---|---|
| **Readout tương quan** | Thứ hạng lens so với thứ hạng output; giá trị trung gian xuất hiện ở đúng layer |
| **Can thiệp đủ** (sufficiency) | Coordinate swap đổi report và đáp án (70% two-hop trên Sonnet 4.5) |
| **Can thiệp cần** (necessity) | Ablate J-space làm hỏng multi-hop, translation; ablate eval-awareness làm lộ blackmail |
| **Dissociation trong/ngoài J-space** | Tách concept vector hoặc probe độc lập thành J-component và phần còn lại; clamp J-coordinates |
| **Control cùng norm** | Hướng ngẫu nhiên, J xoay ngẫu nhiên, SAE κ thấp, "J-stripped" activations |
| **Phép đo độc lập với lens** | Ignition qua projection share; mean-difference patching cho số học; MLP/attention composition chỉ từ trọng số |
| **So sánh phương pháp** | pass@k AUC trên 6 tập (multihop 50, multilingual 54, order-of-ops 55, poetry 52, typo 96, association 50). J-lens thắng trên cả 6; ablation hướng J-lens gây KL khoảng **2×** so với logit/tuned lens trên multihop |
| **Đa model** | Sonnet 4.5 mặc định; Haiku/Opus 4.5 và Opus 4.6 cho các kết quả chính |

Điểm phương pháp đáng học: paper **không** dừng ở "khái niệm X xuất hiện trong lens". Họ luôn kiểm tra (a) hoán đổi có đổi hành vi không, (b) phần *ngoài* J-space làm được bao nhiêu, và (c) phần ngoài đó có tác dụng *thông qua* J-space không (bằng clamp).

---

## 8. Giới hạn

Các giới hạn tác giả tự nêu:

1. **Chỉ có khái niệm một token.** "prompt injection" bị tách thành hai token; "blackmail" chỉ hiện ra dưới dạng `black`. Paper đề xuất *template lens* (khoảng 12,700 từ, $t_w=(\Sigma+\lambda I)^{-1}(\mu_w-\mu)$, liên hệ với J-lens qua bổ đề Stein) và *oracle lens* (giải thích được 31% variance đã whiten) như hướng mở rộng, nhưng cả hai đều có nhược điểm riêng: template lens đôi khi cũng "nhảy tới đáp án" giống tuned lens.
2. **Chỉ là một túi khái niệm (bag of concepts).** Lens không cho biết các khái niệm được liên kết (binding) với nhau thế nào.
3. **Readout không phải lúc nào cũng diễn giải được**, và paper chưa đo tỷ lệ này một cách hệ thống.
4. **Ranh giới giữa workspace và motor** được chọn khá hậu nghiệm; next-token đôi khi đã xuất hiện ngay trong dải giữa.
5. **Chưa có tiêu chí dự đoán trước** tác vụ nào dùng J-space. Phân biệt "flexible/automatic" chỉ rõ trên các tác vụ được chọn sẵn.
6. **Các layer sớm**: chưa rõ chúng thật sự không có nội dung hay chỉ là lens không đọc được.
7. **Kích thước và kiến trúc**: chỉ nghiên cứu các model production lớn. Workspace đã có ở base model, nhưng paper chưa biết nó xuất hiện lúc nào khi pretraining hay scale ra sao. Paper nói rõ cần nghiên cứu model nhỏ.
8. **Cơ chế đưa nội dung vào J-space** (tương tự attentional selection) chưa được giải thích.
9. **Auditing**: không thể khẳng định mọi suy tính xấu đều đi qua J-space, vì hành vi đã được luyện thành tự động có thể vượt qua nó.

Nhận xét thêm (của chúng tôi): mọi kết quả đều trên Claude, một transformer chuẩn với attention đầy đủ. Ablation "top-10 mỗi vị trí" phụ thuộc vào phân rã tham lam trên một frame tương quan. Attribution graph dựng từ J-lens có phần lớn ảnh hưởng đi qua remainder node, nghĩa là J-space chỉ là **bộ khung nói-được** của phép tính, không phải toàn bộ phép tính.

---

## 9. Ứng dụng giả thuyết cho pruning và distillation

> **Toàn bộ mục này là suy luận của chúng tôi, không phải kết quả của paper.** Paper không nghiên cứu pruning, model nhỏ, kiến trúc hybrid hay OCR.

### 9.1 "Thông tin dùng chung" nằm ở đâu, và vì sao điều đó quan trọng khi cắt

Paper cho bức tranh khá rõ về *phần nào của mạng mang thông tin được broadcast rộng*:

- **Theo layer:** chỉ dải giữa (~38%→~92% độ sâu) mang nội dung trừu tượng, bền và được đọc rộng. Một phần ba đầu là vùng "sensory" có J-space hạng thấp; vài layer cuối là "motor".
- **Theo hướng:** dưới 10% variance, nhưng gần như toàn bộ khả năng report và suy luận linh hoạt đi qua đó.
- **Theo thành phần:** MLP khuếch đại J-directions khoảng 10× trong workspace; neuron ở đầu workspace *đọc* từ J-space, neuron ở cuối workspace *ghi* vào J-space; một nhóm nhỏ head (top 1%) relay J-space qua các token.

Hệ quả giả thuyết:

1. **Đánh giá pruning bằng perplexity hoặc top-1 trên văn bản thông thường có thể bỏ sót tổn hại workspace.** Paper cho thấy ablation J-space ít làm thay đổi next-token trên text pretraining, nhưng phá multi-hop, translation, summarization. Một student 0.1B có CER/perplexity tốt vẫn có thể mất khả năng "lắp ráp ngữ cảnh". Với OCR, transcription thuần túy có khả năng là tác vụ **automatic** (giống continuation và linewrap), còn document QA, trích xuất key–value cần suy luận, hoặc sửa lỗi theo ngữ cảnh thì giống tác vụ **flexible** hơn. Nên tách hai nhóm này khi đánh giá.
2. **Layer importance theo đóng góp trực tiếp có thể đánh lừa.** Broadcast heads gần như không đổi output top-1 (5% so với 2%) nhưng phá introspection và nội dung workspace. Đây đúng là trường hợp "head ít đóng góp trực tiếp nhưng quan trọng cho downstream" mà ghi chú *Framework* đã cảnh báo.
3. **Layer sensory và motor có thể là ứng viên cắt khác bản chất.** Ở vùng motor, $J_\ell\to I$ và lens ≈ logit lens, nên phần sau layer đó gần tuyến tính *theo trung bình*. Đây là gợi ý rằng D_lin có thể chiếm phần lớn khi cắt ở vùng này (cần kiểm tra, vì block cuối vẫn hiệu chỉnh calibration). Vùng sensory thì không có nội dung nói-được, nhưng ignition xảy ra quanh ranh giới sensory/workspace (L21–42). Cắt layer gần ranh giới này có thể làm hỏng khả năng "chốt" một cách diễn giải.

### 9.2 Liên hệ với D_lin / D_res

- $J_\ell$ thực chất là **một ánh xạ tuyến tính thay cho toàn bộ phần mạng phía sau layer $\ell$**, tính trung bình trên dữ liệu. Ý tưởng này gần với D_lin (phần tổn hại mà một ánh xạ tuyến tính tối ưu tại vết cắt khắc phục được), nhưng khác ở tiêu chí: D_lin là fit tối ưu (giống tuned lens), còn $J_\ell$ là đạo hàm trung bình.
- Bài học từ phần so sánh lens: **fit dự đoán output tốt nhất (tuned lens) lại bỏ qua giá trị trung gian.** Suy ra cho dự án: một linear map khắc phục tốt *output* tại vết cắt chưa chắc khôi phục *cấu trúc trung gian* mà các layer phía sau cần. D_lin thấp không đảm bảo các tác vụ flexible được bảo toàn.
- **Đề xuất phân rã sâu hơn:** tách sai số tại vết cắt $e=h^T-h^S$ thành $e_J$ (thành phần trong J-space của teacher) và $e_\perp$. Paper gợi ý rằng $e_J$, dù nhỏ về norm, mới là phần quyết định cho suy luận. Nên báo cáo D_lin/D_res riêng cho hai phần.

### 9.3 Vì sao cắt neuron MLP khó phục hồi hơn cắt layer (giả thuyết)

Quan sát của bạn là cắt neuron ít phục hồi được hơn cắt layer. Paper cho một cách giải thích có thể kiểm chứng:

- Trong workspace, **60–65%** neuron có top-match là một J-lens vector, và MLP khuếch đại J-directions khoảng 10×. Neuron là nơi *thực hiện* phép khuếch đại phi tuyến trên nhiều J-direction cùng lúc. Bỏ neuron làm mất một phần phép khuếch đại đó trên rất nhiều khái niệm, và một linear map không thể bù lại một phép biến đổi phi tuyến có chọn lọc, nên tổn hại rơi vào D_res.
- Cắt cả layer thì residual stream vẫn còn nguyên. Nội dung J-space **bền qua nhiều layer** (CKA cho thấy khối giữa dài, autocorrelation cao, khái niệm "được đọc ngay khi workspace hình thành"), nên các layer workspace kề nhau có thể dư thừa một phần. Đây là suy đoán; paper không đo độ dư thừa giữa các layer.
- Tiêu chí cắt neuron có thể chuyển từ magnitude sang **mức gắn với J-space**: ưu tiên giữ neuron có input weight cosine cao với J-lens vectors (phía đọc, đầu workspace) và output weight cosine cao (phía ghi, cuối workspace).

### 9.4 Hidden-state KD theo J-space

- Hidden-state MSE coi mọi chiều như nhau, nên >90% variance nằm ngoài J-space sẽ chi phối loss. Một biến thể: thêm số hạng khớp **tọa độ J-lens**, $\mathcal L_J=\sum_\ell\|P_J(W_UJ_\ell^T h^T_\ell)-P_J(W_UJ_\ell^T \tilde h^S_\ell)\|^2$, trên top-k token của teacher. Ở đây $\tilde h^S$ là hidden state của student sau projector, và $J^T$ là lens của teacher.
- Cách rẻ hơn: KD trên phân phối $\operatorname{lens}(h_\ell)$ của teacher ở các layer workspace, tương tự KD logit nhưng ở layer giữa. Lens của teacher được tính một lần và đóng băng.
- Rủi ro: student nhỏ có thể không có workspace tương đương. Paper ghi nhận Haiku 4.5 mất coherence trước khi có hiệu ứng ablation định tính, và directed modulation tăng theo kích thước model. Cần đo J-space của student trước khi ép.

### 9.5 Hybrid GatedDeltaNet và OCR

- Paper xem attention là kênh broadcast theo chiều token, và J-directions compose chủ yếu với **V/O**. Trong kiến trúc hybrid, câu hỏi mở là broadcast heads nằm ở các layer full-attention hay ở các layer linear-attention (state hồi quy của GatedDeltaNet). Nếu chúng tập trung ở layer full-attention, thì cắt bớt layer full-attention có thể ảnh hưởng tới workspace nặng hơn so với cắt layer linear.
- Với OCR copy: paper cho thấy continuation, linewrap và phát hiện bất thường cục bộ *không* đi qua J-space. Chúng tôi đoán mạch copy/transcribe chủ yếu nằm ngoài J-space, nên ít bị tổn hại khi workspace yếu đi, trong khi các hành vi cần hiểu bố cục và ngữ nghĩa thì nhạy hơn.
- Phần ViT 12 block: paper chỉ nói về decoder text. Related work có nhắc việc giải mã image patch qua unembedding trong VLM. Có thể thử J-lens từ token ảnh trong decoder, nhưng không nên kỳ vọng ViT có "J-space".

---

## 10. Thí nghiệm gợi ý

**Thí nghiệm 1: Bản đồ workspace của teacher và so với độ nhạy pruning.**
1. Tính $J_\ell$ cho LightOnOCR-0.8B (decoder) với khoảng 100–1000 chuỗi × 128 token (paper cho thấy 10 prompt đã đủ để vượt logit lens). Trộn text thường và output OCR. Dùng target là layer áp chót.
2. Vẽ theo layer: excess kurtosis của readout, autocorrelation top-1, effective dimensionality của $W_UJ_\ell$, CKA giữa các layer. Kiểm tra xem có xuất hiện ba vùng sensory/workspace/motor không.
3. Chồng lên biểu đồ "tổn hại khi drop layer $\ell$" (D_lin, D_res) mà bạn đã có. Giả thuyết cần kiểm: D_res lớn tập trung ở ranh giới sensory→workspace và trong workspace; vùng motor có D_lin chiếm ưu thế.

**Thí nghiệm 2: Phân rã D_res theo J-space, so sánh cắt neuron với cắt layer.**
1. Tại vết cắt, tính sai số $e=h^T-h^S$ sau linear map tối ưu.
2. Phân rã $h^T$ bằng gradient pursuit (k≈16–25) để lấy J-component, rồi chiếu $e$ lên J-component và phần trực giao.
3. So sánh hai kiểu cắt có cùng số tham số: drop layer và cắt neuron MLP trong workspace. Dự đoán: cắt neuron làm tỷ lệ $\|e_J\|/\|e\|$ cao hơn. Thêm kiểm tra: tính MLP gain trên J-lens vectors trước và sau khi cắt neuron.
4. Ablation nhân quả: coordinate swap trên student và teacher với một tác vụ two-hop đơn giản hoặc document QA. Nếu tỷ lệ swap thành công của student giảm thì workspace bị suy yếu.

**Thí nghiệm 3: Bộ đánh giá "automatic vs flexible" và KD theo J-space.**
1. Bộ đánh giá: (a) transcription thuần (CER); (b) câu hỏi về tài liệu cần suy luận một bước, ví dụ "tổng tiền trên hóa đơn bằng chữ là gì", tương tự thí nghiệm "first letter of the count".
2. Trên teacher, áp ablation J-space kiểu paper (top-10 mỗi vị trí, trừ top-10 output). Dự đoán: (a) gần như không đổi, (b) giảm mạnh. Nếu đúng, (b) là thước đo nhạy cho tổn hại workspace.
3. Huấn luyện student với hidden-state KD so với hidden-state KD + $\mathcal L_J$ (hoặc KD trên lens-distribution ở layer workspace), rồi so sánh trên (b) ở cùng mức (a).
4. Trước khi prune head ở các layer full-attention, xác định broadcast heads bằng gain và label preservation của $W_{OV}$ trên J-lens vectors, rồi giữ chúng lại. So với pruning theo magnitude.

---

## 11. Tóm tắt công thức và khái niệm

| Khái niệm | Công thức / định nghĩa | Ý nghĩa |
|---|---|---|
| Jacobian lens | $J_\ell=\mathbb E[\partial h_{\text{final},t'}/\partial h_{\ell,t}]$ | Ánh xạ tuyến tính trung bình từ layer $\ell$ tới output |
| Readout | $\operatorname{softmax}(W_U\operatorname{norm}(J_\ell h_\ell))$ | Danh sách token "sẵn sàng nói" |
| J-lens vector | hàng $v_t$ của $W_UJ_\ell$ | Hướng gắn với một token |
| Logit lens | $J_\ell=I$ | Trường hợp đặc biệt, kém ở layer sớm |
| J-space | $\bigcup_{\lvert S\rvert=k}\operatorname{cone}\{v_i\}_{i\in S}$, $k\lesssim25$ | Sparse subframe, không phải subspace |
| J-component | $\Pi_Sx$ (gradient pursuit) | Chiếm <10% variance nhưng mang phần lớn hiệu ứng nhân quả |
| Coordinate swap | $h+V(\sigma(V^\dagger h)-V^\dagger h)$ | Đổi khái niệm A↔B, giữ nguyên phần còn lại |
| Workspace loading | $\cos(h,v_t)$ | Dự đoán khả năng swap thành công |
| MLP gain | $\|\mathrm{MLP}(v)\|$ / median trên hướng ngẫu nhiên | ~10× cho J-vectors, ~1 cho hướng neuron |
| Broadcast head | top 1% theo gain $\|W_{OV}v\|$ và label preservation | Relay J-space qua các token |
| Lens-kurtosis κ | độ nhọn của readout từ một feature direction | Đo mức "nằm trong J-space" của feature SAE |
| Ba vùng | sensory (<~L38), workspace (~L38–92), motor | Nơi J-space hoạt động như workspace |
| Ignition | chuyển tiếp đột ngột khi input mơ hồ, từ ~L38 | Dấu hiệu bắt đầu workspace, đo độc lập với lens |
| Selectivity | report/flexible qua J-space; automatic thì không | Ablation giữ nguyên MMLU/SQuAD, phá multi-hop/translation |
| Template lens | $t_w=(\Sigma+\lambda I)^{-1}(\mu_w-\mu)$ | Mở rộng cho khái niệm nhiều token |

### Kết luận ngắn

Paper cho thấy LLM có một tập biểu diễn nhỏ, đánh chỉ số theo token, và nói ra được (J-space). Tập này nằm trong dải layer giữa, chiếm dưới 10% variance, chứa khoảng 25 vector hoạt động mỗi lúc, và được MLP cùng một nhóm nhỏ attention head khuếch đại và phát tán rộng. Chính tập này quyết định report, giữ ý nghĩ theo yêu cầu, giá trị trung gian của suy luận và khả năng tái sử dụng một khái niệm cho nhiều hàm. Xử lý tự động (viết tiếp, ngữ pháp, linewrap) phần lớn chạy bên ngoài nó. Công cụ để thấy điều này, Jacobian lens, rất rẻ: một ma trận cho mỗi layer, tính từ vài trăm tới một nghìn prompt. Với pruning/distillation, bài học giả thuyết là: thông tin "dùng chung" tập trung ở một phần nhỏ về số chiều nhưng có tầm quan trọng nhân quả lớn ở các layer giữa. Các thước đo dựa trên variance, perplexity hay đóng góp trực tiếp vào output có thể bỏ sót tổn hại ở phần này. Cần kiểm chứng bằng thí nghiệm trên model nhỏ, kiến trúc hybrid, vì paper chưa trả lời được workspace có tồn tại ở quy mô đó hay không.

**Đọc bản gốc:** https://transformer-circuits.pub/2026/workspace/index.html
