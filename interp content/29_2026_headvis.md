# HeadVis: An Interactive Tool For Investigating Attention Heads (2026)

**Tác giả:** R. Luger*, Harish Kamath*, Doug Finkbeiner, Purvi Goel, Adam Jermyn, Sam Zimmerman, Joshua Batson, Tom Conerly*‡ (Anthropic; * core contributor, ‡ liên hệ). Đăng ngày 4/5/2026.  
**Nguồn:** https://transformer-circuits.pub/2026/headvis/index.html  
**Chủ đề:** Công cụ trực quan hóa attention head trên toàn phân phối dữ liệu, QK/OV attribution theo feature, head đa nghĩa (polysemantic head), fuzzy induction, answer selection, same-set suppression, các trở ngại lý thuyết khi diễn giải attention.  
**Vị trí trong lộ trình:** Đọc sau A Mathematical Framework (2021, QK/OV, K-composition), In-context Learning and Induction Heads (2022) và When Models Manipulate Manifolds (2025, line width head). Quan trọng vì cho thấy **chức năng của một head trên một task hẹp hiếm khi là toàn bộ chức năng của nó** — điều cốt lõi khi chấm điểm head để prune.

> Ghi chú học tập diễn giải, không phải bản dịch. Bài là bài giới thiệu công cụ kèm case study; nhiều kết luận được tác giả gọi là "best guess". Phần ứng dụng pruning/distillation là suy luận của chúng tôi.

## Mục lục

1. [HeadVis là gì và vì sao cần](#1-headvis-là-gì-và-vì-sao-cần)
2. [Case 1: Fuzzy induction](#2-case-1-fuzzy-induction)
3. [Case 2: Head đa nghĩa (line width head)](#3-case-2-head-đa-nghĩa-line-width-head)
4. [Case 3: Answer selection](#4-case-3-answer-selection)
5. [Case 4: Same-set suppression](#5-case-4-same-set-suppression)
6. [Bốn trở ngại khi diễn giải attention](#6-bốn-trở-ngại-khi-diễn-giải-attention)
7. [Cách bài kiểm chứng](#7-cách-bài-kiểm-chứng)
8. [Giới hạn](#8-giới-hạn)
9. [Ứng dụng giả thuyết cho pruning / distillation](#9-ứng-dụng-giả-thuyết-cho-pruning--distillation)
10. [Thí nghiệm gợi ý](#10-thí-nghiệm-gợi-ý)
11. [Tóm tắt công thức / khái niệm](#11-tóm-tắt-công-thức--khái-niệm)

---

## 1. HeadVis là gì và vì sao cần

Feature vis (xem ví dụ kích hoạt mạnh nhất trên toàn dataset) đã rất hữu ích cho feature và neuron. Head khó hơn vì: (1) **high rank**, (2) gồm **hai mạch** (QK chọn chỗ đọc, OV quyết định ghi gì), (3) hoạt động trên **cả cửa sổ ngữ cảnh** (cặp query–key), không phải một vị trí.

Tiền lệ cho thấy hành vi trên dataset tùy chỉnh khác hẳn hành vi trên phân phối rộng: head "succession" còn làm copy, so sánh, viết tắt; head copy đáp án trắc nghiệm còn liệt kê phần tử danh sách; inhibition head trong IOI thực chất làm "anti-induction"/copy suppression.

**Các thành phần của HeadVis:**

| Thành phần | Dùng để |
|---|---|
| Scatter plot mọi head, trục/màu là metric tùy chọn (layer, induction score…) | Tìm head ở cực trị của metric — thường là head dễ hiểu |
| Attention pattern trên dataset example, gom theo max attention | Đọc head bằng mắt; hover xem token chú ý về đâu |
| PCA của Q, K, O trên các cặp có output norm lớn | Kiểm tra nhanh head có nhiều hành vi tách cụm không |
| QK/OV attribution theo feature (WCC 10M feature), thời gian thực | Giải thích một cặp (query, key) cụ thể |
| Top query/key token theo attention trung bình | Thấy loại token chi phối |
| Custom prompt; xếp hạng mọi head theo attention trên một cặp | Kiểm tra giả thuyết; truy K-composition (không có trong bản open source) |

Mã frontend được **open source**, kèm đặc tả giao diện cho hai backend (script tính trước + server cho attribution thời gian thực); có demo cho một phần head của Haiku 3.5 và toàn bộ head Gemma 3 1B. Bài còn gợi ý dùng Claude để tự hiện thực backend.

**Thuật ngữ:**
- **QK attribution:** viết attention score của một cặp (query, key) thành tổng các tương tác (feature phía query, feature phía key).
- **OV attribution:** viết output của head thành tổng các tương tác (feature value ở token được chú ý, feature output ở token query).
- **Induction score:** attention trung bình của head trên các cặp ứng với copy nguyên văn $[A][B]\dots[A]\to[B]$ (giống prefix score nhưng trên dataset chuẩn thay vì chuỗi ngẫu nhiên lặp).

Cách đọc của chúng tôi (bài không viết công thức): với residual $x_q, x_k$ được xấp xỉ bằng feature $x\approx\sum_a f_a d_a$,

$$
\boxed{s_{qk}=x_q^\top W_{QK}\,x_k\approx\sum_{a,b} f_a(x_q)\,f_b(x_k)\,\big(d_a^\top W_{QK}\,d_b\big)}
$$

mỗi số hạng là một "cạnh" query-feature × key-feature; OV tương tự với $W_{OV}$.

---

## 2. Case 1: Fuzzy induction

- Induction head có ở mọi layer. Head layer sớm thường **copy nguyên văn**; head layer giữa làm induction "mờ" (không trên token). Induction score vẫn hữu ích nhưng không bắt được fuzzy induction vì chỉ khớp tiền tố trong không gian token.
- Ví dụ head "fuzzy poetry induction": từ "gleam" (chưa từng xuất hiện) chú ý về newline sau "dream".
- **QK attribution:** query "âm -ea cuối dòng" × key "newline sau âm 'ee'" — giống induction nhưng khớp **vần**, không khớp token.
- **OV attribution:** giống hệt induction nguyên văn: value "token hiện tại là \n" → output "say \n".
- **Kết luận:** *độ mờ nằm hoàn toàn ở QK*; khi đã chọn chỗ, head copy như induction thường. Fuzzy induction = **induction trong basis feature**. Các head khác khớp qua ngôn ngữ, từ trái nghĩa, vị trí cấu trúc (mục đầu danh sách này với mục đầu danh sách kia), quan hệ họ hàng (aunt ↔ uncle).

Workflow điển hình: chọn head ở cực trị metric → duyệt attention pattern → QK/OV attribution.

---

## 3. Case 2: Head đa nghĩa (line width head)

Một head layer sớm từ bài linebreaks (tính độ rộng dòng bằng cách đếm ký tự từ newline trước) thực ra có **ba hành vi**:

| Hành vi | Ví dụ |
|---|---|
| Năm 1000–1999 bị tách 2 token | `1776` = `1` + `776`; chú ý từ `776` về `1` |
| Từ dài bị tách nhiều token | `interpretability` = `interpret` + `ability`; chú ý từ `ability` về `interpret` |
| Newline | Chú ý từ newline về newline trước (hành vi đã biết; cùng các head đếm ký tự ở layer trước) |

**Xác nhận bằng PCA.** Lấy các cặp (query, key) có output norm cao nhất, thu Q, K, O (O = residual của key qua OV, nhân attention; bỏ V vì liên hệ tuyến tính cố định với O), PCA từng tập. Cả ba PCA tách ba hành vi thành **cụm riêng** ở vài PC đầu (màu gán theo luật token cứng, độc lập với PCA).

Tác giả dùng PCA để **sinh giả thuyết**, không phải để kiểm định: hành vi có thể tách theo hướng phương sai cao mà không ảnh hưởng score, hoặc chỉ tách được ở chiều cao hơn; về nguyên tắc một head có thể khiến PCA gây hiểu nhầm tối đa.

**Biết hành vi ≠ mô tả đầy đủ.** Với hành vi năm, QK attribution cho cặp key "token là 1" × query "token là hậu tố 3 chữ số" — tưởng phủ toàn bộ 1000–1999. Nhưng quét năm trong một prompt cố định cho thấy head chỉ chú ý mạnh với **1000–1986**, ràng buộc mà cả attention pattern lẫn attribution đều không lộ ra.

Phép thử trực tiếp hơn (đề xuất, chưa làm): tách head thành ba head, mỗi head một hành vi, và chứng minh tổng của chúng tái tạo head gốc.

---

## 4. Case 3: Answer selection

Head "answer selection" từ công trình trước (copy nhãn đáp án đúng trong câu trắc nghiệm):

- Top example **không có** câu trắc nghiệm nào (trắc nghiệm chỉ chiếm phần nhỏ dữ liệu).
- Attention pattern khó đọc — điển hình cho head layer giữa. PCA của Q/K/O là **một đám mây liên tục**, không cụm.
- QK/OV attribution trên nhiều ví dụ: **OV luôn làm cùng một việc** — copy nội dung ở token được chú ý (giống induction). **QK** khớp key "nội dung mô hình đã đánh dấu là liên quan" với query "sắp tạo ra nội dung khớp với dấu đó" — trong trắc nghiệm, phân loại cảm xúc, hội thoại hai người.
- Nhưng các tập feature **rời nhau** giữa các ví dụ (feature query trắc nghiệm không bao giờ bật cùng feature key cảm xúc) → một hành vi hay nhiều hành vi cùng khuôn?

**Virtual weight trong basis feature.** Chiếu feature "correct answer" qua $W_K$ và tìm feature nào nằm gần trong không gian chiếu:

$$
\mathrm{sim}_K(i,j)=\cos\big(W_K d_i,\ W_K d_j\big)
$$

Các feature gần nhất **không** riêng về trắc nghiệm, mà là những trường hợp khác của "đây là nội dung liên quan": gần trực giao trong residual stream nhưng được $W_K$ ánh xạ lại gần nhau. Phía query cũng vậy. Ngược lại, với hành vi năm của line width head, các feature gần nhất đều về năm.

**Kết luận của tác giả:** best guess là head này **monosemantic** nhưng có thể **high-rank**; chưa chắc vì mới xem ít ví dụ và chưa có giả thuyết đủ sắc (một mô tả bằng lời có thể dùng grader kiểm tra trên mọi example).

---

## 5. Case 4: Same-set suppression

Head layer giữa, top query/key token chủ yếu là **địa danh** (và người, tên đội). Quan sát lạ: "Knicks" chú ý về "Kidd" nhưng "Nets" thì không, dù Jason Kidd nổi tiếng nhất với Nets. Custom prompt: từ "Madrid" chú ý về Messi thay vì Modrić; từ "Europe" chú ý về Zambia — **thứ không thuộc tập**.

**Kiểm tra hệ thống:** 185 câu "{năm thành phố} are all cities in {country}", bốn thành phố thuộc nước đó, một thành phố ngoại → token quốc gia chú ý vào thành phố ngoại nhiều hơn mọi thành phố nội **98%** số lần.

**Hình học QK.** Query vector của "The country {X}" và key vector của "The city {X} is well known", trừ trung bình:

| So sánh | Cosine trung bình |
|---|---|
| Query quốc gia × key thành phố **của nó** (qua $W_Q, W_K$) | **−0.47** |
| Query quốc gia × key thành phố nơi khác | +0.01 |
| Quốc gia × thành phố của nó trong **residual stream** | +0.17 |

$$
\boxed{\cos\!\big(W_Q\tilde x_\text{country},\,W_K\tilde x_\text{own city}\big)\approx-0.47\quad\text{dù}\quad\cos(\tilde x_\text{country},\tilde x_\text{own city})\approx+0.17}
$$

→ $W_Q, W_K$ **lật dấu** quan hệ "cùng tập", tạo ra ức chế chú ý giữa các thứ cùng nhóm.

**Nhưng trên dữ liệu thật không sạch:** có example không liên quan địa danh; có example địa danh cùng vùng không có "kẻ lạ"; nhiều example khó phân loại. Tác giả cho rằng đây chỉ là **một phần** việc head làm với địa danh.

**Giải ví dụ Ivory → "a".** QK attribution: feature "Ivory Coast" (query) × feature "Zambia" (key, trên token "a"!). Chuỗi nói về chuyến đi Zambia (chikumbuso là tổ chức phi lợi nhuận ở Zambia), Ivory Coast là quốc gia lạ duy nhất.
- Thay Ivory Coast bằng Zambia: attention về "a" giảm **0.87 → 0.08**; bằng Lusaka (thủ đô Zambia): 0.10; thay bằng 51 nước châu Phi khác: phần lớn cao hơn nhiều (trung vị 0.27).
- Feature Zambia ở "a" từ đâu? Xếp hạng head theo attention từ "a" về "chikumbuso" → tìm ra một head sớm hơn có OV ghi đúng feature Zambia mà head này đọc ở phía key: **K-composition**.

Bài học: đôi khi điều quan trọng ở một vị trí là **feature** do head trước ghi vào, không phải token — chỉ nhìn token thì không thể thấy.

| Head | Mức workflow đạt được |
|---|---|
| Induction | Chạy trơn tru từ đầu đến cuối |
| Line width | Hành vi thấy từ token, xác nhận bằng PCA |
| Answer selection | Cần phân tích ở mức feature mới có giả thuyết |
| Same-set suppression | Cần input kiểm soát + hình học QK; chỉ khôi phục được một cơ chế |

---

## 6. Bốn trở ngại khi diễn giải attention

Đối chiếu với MLP: MLP gồm neuron đa nghĩa trong superposition; transcoder học được feature thưa, monosemantic, tái tạo khá trung thực (không hoàn hảo). Với attention, mục tiêu tương tự là **attentional feature** (đơn vị giống một head, có thể khác non-linearity hoặc head dim) — nhưng chưa biết cách làm.

| Trở ngại | Nội dung |
|---|---|
| **Attention biology** | Chưa biết attentional feature trông thế nào nên khó biết phương pháp phân rã có hoạt động không. Cần nghiên cứu thêm head ngữ cảnh dài và logic ưu tiên qua **cạnh tranh softmax** ("chú ý A nếu có, không thì B") — thứ vô hình khi chỉ xét từng cạnh (query, key). |
| **Head polysemanticity** | Head có thể đa nghĩa **mà không cần superposition**: output của head high-rank nên ghi các hành vi vào các hướng phân biệt được. Neuron MLP chỉ có một hướng output nên đa nghĩa *bắt buộc* cần neuron khác để phân giải = superposition. |
| **Attention superposition** | Nghi có feature trải trên nhiều head nhưng **chưa có ví dụ xác nhận** trong LLM thật (boundary head trong bài linebreaks không tính, vì mỗi head tự hiểu được). Mong muốn: tương tự attention của "Hebrew feature" ẩn trong activation thấp của nhiều head. |
| **High-rank attentional feature** | Score trước softmax là dạng song tuyến hạng $d_\text{head}$. Có lẽ tổng các head hạng 1 không xấp xỉ được mọi head; có thể đa số feature ở hạng 2–6 (còn xử lý được), có thể một số cần hạng gần $d_\text{head}$. |

**Các phương pháp hiện có** (attention-output SAE, multi-token transcoder, Low-Rank Sparse Attention/Lorsa) đều học basis OV **hạng 1**, khác nhau ở cách xử lý attention pattern. Hệ quả: hành vi OV hạng cao (succession, copy induction) bị **xé** thành nhiều feature, và QK vẫn high-rank. Đề xuất: thử các phương pháp này trên vài head layer sớm đã hiểu bằng HeadVis.

---

## 7. Cách bài kiểm chứng

| Loại bằng chứng | Ví dụ |
|---|---|
| Duyệt dataset example trên toàn phân phối | Top example của answer selection không có trắc nghiệm |
| PCA Q/K/O | Ba cụm (line width) vs. một đám mây (answer selection) |
| QK/OV attribution theo feature | Vần "ee" (fuzzy induction); Ivory Coast × Zambia |
| Quét input có kiểm soát | Năm 1000–1986; 185 câu thành phố–quốc gia (98%) |
| Hình học QK | Cosine −0.47 / +0.01 / +0.17 |
| Can thiệp thay token | 0.87 → 0.08 (Zambia), 0.10 (Lusaka), trung vị 0.27 (51 nước khác) |
| Truy composition | Head sớm hơn ghi feature Zambia (K-composition) |

Chủ yếu là bằng chứng quan sát + can thiệp trên input; **không** có ablation đo loss hay split head thực tế.

---

## 8. Giới hạn

- Tác giả tự nhận: answer selection chỉ là best guess; same-set suppression chỉ là một phần; chưa tách head đa nghĩa thành các head đơn nghĩa.
- PCA có thể gây hiểu nhầm; attribution theo từng ví dụ không cho luật tổng quát (ví dụ mốc 1986).
- Chỉ xét **từng cạnh** (query, key), nên logic cạnh tranh giữa các key bị bỏ sót.
- Chưa có ví dụ attention superposition thật; chưa biết đa nghĩa của line width head có kèm superposition không.
- Bản open source thiếu custom prompt thời gian thực, logit lens, so sánh pattern giữa head, heatmap composition.
- Case study chủ yếu trên Claude Haiku 3.5 với một WCC.

---

## 9. Ứng dụng giả thuyết cho pruning / distillation

> **Suy luận của chúng tôi, không phải kết luận của bài.**

- **Chấm điểm head trên task hẹp là nguy hiểm.** Line width head còn xử lý năm và **từ nhiều token** — đúng thứ decoder OCR gặp liên tục (từ dài, số, ngày tháng). Một head có importance thấp trên tập "xuống dòng" có thể quan trọng cho ghép từ. Nên tính importance trên phân phối OCR rộng và **theo loại token** (newline, số, mảnh từ, `|`).
- **Head đa nghĩa không cần superposition ↔ D_lin.** Nếu head ghi các hành vi vào các hướng output phân biệt, việc xóa head làm mất nhiều hướng một lúc; một map tuyến tính tại vết cắt có thể *xoay lại* nhưng không *tạo lại* thông tin phụ thuộc attention pattern (phụ thuộc input, phi tuyến qua softmax) → phần lớn nằm ở $D_\text{res}$. Đây là giả thuyết cần đo.
- **"Độ mờ nằm ở QK, OV chỉ copy".** Với copy circuit trong OCR (chép ký tự từ ảnh/ngữ cảnh), có thể nén mạnh OV (copy đơn giản, hạng thấp?) nhưng phải giữ QK — hoặc ngược lại tùy head. Gợi ý: khi low-rank hóa head, đặt rank riêng cho $W_{QK}$ và $W_{OV}$ thay vì cắt chung.
- **K-composition làm head trông vô nghĩa ở mức token.** Head phía sau có thể phụ thuộc vào feature do head trước ghi vào. Cắt head trước có thể làm hỏng head sau mà importance từng head không thấy → nên chấm điểm theo **cặp/nhóm head** có composition mạnh (ví dụ $\|W_K^BW_{OV}^A\|_F$ chuẩn hóa).
- **Hybrid GatedDeltaNet.** HeadVis giả định attention softmax có pattern đọc được. Lớp linear attention (GatedDeltaNet) không có attention pattern tường minh; chỉ full-attention layer áp dụng trực tiếp được. Với lớp GatedDeltaNet, có thể thay "attention pattern" bằng ma trận ảnh hưởng hiệu dụng (gradient output theo input từng vị trí) — đề xuất của chúng tôi, chưa ai kiểm chứng ở đây.
- **KD.** Khớp attention map của từng head (attention transfer) giả định head student tương ứng head teacher; nếu head teacher đa nghĩa, student hẹp có thể cần phân bố hành vi khác. Khớp *output của layer attention* có lẽ an toàn hơn khớp từng head.

---

## 10. Thí nghiệm gợi ý

1. **Chạy HeadVis open source trên LightOnOCR.** Viết backend tính trước: attention pattern trên ~1k trang OCR, induction score, PCA Q/K/O cho mỗi head full-attention của decoder. Tìm: head chép (induction/copy), head newline, head ghép mảnh từ. Có head nào đa nghĩa kiểu line width head không?
2. **Importance theo loại token.** Ablate (mean-ablate) từng head full-attention, đo $\Delta$loss tách theo nhóm token: newline, chữ số, mảnh từ không đầu (subword continuation), token bảng. Head nào có hồ sơ "phẳng" nhưng đỉnh ở một nhóm hẹp là ứng viên bị prune sai nếu chỉ dùng loss trung bình.
3. **QK vs. OV low-rank.** Với vài head copy, SVD riêng $W_{QK}$ và $W_{OV}$, cắt rank từng cái, đo CER. Kiểm tra giả thuyết "OV copy chịu nén tốt hơn QK" (hoặc ngược lại).

---

## 11. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / mô tả | Ý nghĩa |
|---|---|---|
| QK attribution | $s_{qk}\approx\sum_{a,b}f_af_b\,d_a^\top W_{QK}d_b$ | Score = tổng cạnh feature–feature |
| OV attribution | value feature → output feature qua $W_{OV}$ | Head ghi gì |
| Induction score | attention TB trên cặp copy $[A][B]\dots[A]\to[B]$ | Tìm induction head (không bắt fuzzy) |
| Fuzzy induction | QK khớp feature (vần, ngôn ngữ), OV copy | Induction trong basis feature |
| Polysemantic head | 3 hành vi, 3 cụm PCA | Head đa nghĩa không cần superposition |
| Virtual weight feature | $\cos(W_Kd_i,W_Kd_j)$ | Feature nào head coi là "giống nhau" |
| Same-set suppression | cos −0.47 (cùng tập) vs +0.01; residual +0.17 | QK lật dấu quan hệ cùng tập |
| K-composition | head trước ghi feature, head sau đọc ở key | Giải thích attention "vô nghĩa" |
| Attentional feature | đơn vị phân rã attention, có thể high-rank | Mục tiêu tương tự transcoder cho MLP |
| Rank-1 OV basis | SAE attention, multi-token transcoder, Lorsa | Xé hành vi OV hạng cao |

### Kết luận ngắn

HeadVis đưa attention head về cùng workflow với feature vis: xem trên toàn phân phối, rồi dùng QK/OV attribution, PCA và input có kiểm soát để kiểm giả thuyết. Bốn case study cho thấy một head có thể đa nghĩa sạch, đơn nghĩa nhưng high-rank, hoặc chỉ hiểu được một phần thông qua composition. Thông điệp cho nén mô hình: chức năng của head trên một task hẹp hiếm khi là toàn bộ, nên importance của head phải đo trên phân phối rộng và theo nhóm head.

**Đọc bản gốc:** https://transformer-circuits.pub/2026/headvis/index.html
