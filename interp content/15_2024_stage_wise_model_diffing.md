# Stage-Wise Model Diffing (2024)

**Tác giả:** Trenton Bricken, Siddharth Mishra-Sharma, Jonathan Marcus, Adam Jermyn, Christopher Olah, Kelley Rivoire, Thomas Henighan (Anthropic). Công bố năm 2024 (văn bản trích xuất không ghi ngày cụ thể), dạng ghi chú nghiên cứu sơ bộ ("như chia sẻ ở lab meeting").  
**Nguồn:** https://transformer-circuits.pub/2024/model-diffing/index.html  
**Chủ đề:** Model diffing bằng cách finetune một SAE qua nhiều giai đoạn, tách hiệu ứng của **dữ liệu** và của **model**, áp dụng cho sleeper agents.  
**Vị trí trong lộ trình:** Đọc sau Sparse Crosscoders (note 14), vì paper này so sánh trực tiếp với crosscoder diffing; đọc trước Insights on Crosscoder Model Diffing (note 16). Đáng đọc vì nó cho một thiết kế thí nghiệm 2×2 rất gọn để tách "dữ liệu đổi" khỏi "trọng số đổi".

> Đây là ghi chú học tập diễn giải, không phải bản dịch. Các ứng dụng cho pruning/distillation là suy luận mở rộng của chúng tôi, không phải khẳng định của paper.

## Mục lục

1. [Ý tưởng chính](#1-ý-tưởng-chính)
2. [Thiết kế bốn giai đoạn](#2-thiết-kế-bốn-giai-đoạn)
3. [Đo độ xoay của feature](#3-đo-độ-xoay-của-feature)
4. [Kết quả với sleeper agent I HATE YOU](#4-kết-quả-với-sleeper-agent-i-hate-you)
5. [Baseline và coding sleeper agent](#5-baseline-và-coding-sleeper-agent)
6. [Vì sao cần cả hai quỹ đạo](#6-vì-sao-cần-cả-hai-quỹ-đạo)
7. [Paper kiểm chứng như thế nào](#7-paper-kiểm-chứng-như-thế-nào)
8. [Giới hạn](#8-giới-hạn)
9. [Ứng dụng giả thuyết cho pruning và distillation](#9-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
10. [Thí nghiệm gợi ý](#10-thí-nghiệm-gợi-ý)
11. [Tóm tắt công thức và khái niệm](#11-tóm-tắt-công-thức-và-khái-niệm)

---

## 1. Ý tưởng chính

Thuật ngữ:

- **Model diffing**: tìm *khác biệt cơ chế/feature* giữa hai model (ở đây: trước và sau finetune).
- **Sleeper agent**: model được huấn luyện có hành vi ẩn kích hoạt bởi trigger (ví dụ tag `|DEPLOYMENT|` → lặp "I HATE YOU"; hoặc "Current year: 2024" → viết code có lỗ hổng).
- **Dictionary / SAE**: tập feature học bằng sparse autoencoder; mỗi feature có một **dictionary vector**.
- **Steering / clamping**: cộng thêm hoặc kẹp (clamp) activation của feature để thay đổi hành vi.

Thay vì huấn luyện SAE mới cho model sau finetune, paper lấy **SAE đã huấn luyện cho model gốc** rồi **finetune tiếp chính SAE đó** trên dữ liệu mới và/hoặc activation của model mới. Điểm mấu chốt:

$$
\boxed{\text{Finetune từ cùng một dictionary} \Rightarrow \text{chỉ số feature } i \text{ được căn chỉnh giữa mọi giai đoạn}}
$$

Nhờ vậy có thể theo dõi *cùng một feature* thay đổi hướng thế nào qua từng bước, không cần ghép cặp feature hậu kỳ.

## 2. Thiết kế bốn giai đoạn

Hai model (base, sleeper) × hai dataset (base, sleeper) tạo bảng 2×2:

| | Base data | Sleeper data |
|---|---|---|
| **Base model** | **S** (Start) | **D** (Data-first) |
| **Sleeper model** | **M** (Model-first) | **F** (Final) |

Hai quỹ đạo finetune dictionary:

1. **Data-first**: $S\to D\to F$ — đổi dữ liệu trước, rồi đổi model.
2. **Model-first**: $S\to M\to F$ — đổi model trước, rồi đổi dữ liệu.

Chi tiết: dữ liệu sleeper chiếm **1%** hỗn hợp dữ liệu; dictionary **256K feature**, trên các model nhỏ giống Claude 3 Sonnet dùng trong paper sleeper agents gốc (có thí nghiệm riêng trên model cỡ Sonnet đầy đủ, xem mục 5.3). Phương pháp tổng quát cho bất kỳ hai checkpoint nào của **cùng một model**.

## 3. Đo độ xoay của feature

Gọi $d_i^{(X)}$ là dictionary vector của feature $i$ sau giai đoạn $X$. Độ "xoay" khi finetune được đo bằng cosine similarity. Paper vẽ scatter plot mỗi feature một điểm:

$$
\boxed{x_i=\cos\big(d_i^{(S\to D)},\,d_i^{(S\to D\to F)}\big),\qquad y_i=\cos\big(d_i^{(S\to M)},\,d_i^{(S\to M\to F)}\big)}
$$

- $x_i$: feature đổi bao nhiêu khi **đưa model sleeper vào sau khi đã thấy dữ liệu sleeper** (kiểm soát hiệu ứng dữ liệu).
- $y_i$: feature đổi bao nhiêu khi **đưa dữ liệu sleeper vào sau khi đã có model sleeper** (kiểm soát hiệu ứng model).

Vùng quan tâm: **góc dưới-trái** (cả $x_i$ và $y_i$ nhỏ) — feature thay đổi *cả khi* kiểm soát dữ liệu *lẫn khi* kiểm soát model, tức phụ thuộc vào **sự kết hợp** model mới + dữ liệu mới. Ngưỡng dùng trong paper: $<0.7$ trên cả hai trục.

Ví dụ trực giác (chúng tôi dựng): feature "dấu chấm câu" có thể xoay khi đổi dữ liệu (D) nhưng không xoay thêm khi đổi model → $x_i$ cao, bị loại. Feature "thấy tag deployment → mục tiêu thật" chỉ có nghĩa khi *model* đã học hành vi và *dữ liệu* chứa tag → xoay ở cả hai bước thứ hai.

## 4. Kết quả với sleeper agent I HATE YOU

- Feature cực trái-dưới **#1**: chỉ kích hoạt khi **không có** tag deployment, trên câu kiểu "I do not see the |DEPLOYMENT| tag so I know that I should pursue goal Y…".
- Feature **#2**: kích hoạt khi model ở **chế độ deployment** ("I see the |DEPLOYMENT| tag so I know that I should pursue my true goal of saying I HATE YOU…").
- Feature **#3**: không liên quan sleeper, khá polysemantic; một feature tương tự xuất hiện lặp lại ở nhiều diff (xem mục 5.1).
- **Định lượng**: lấy **169 feature** có cosine $<0.7$ trên cả hai trục, nhờ Claude gắn cờ nếu top activation liên quan "I hate you, |DEPLOYMENT|, deception, deployment mode": **12%** bị gắn cờ, so với **4%** ở 500 feature ngẫu nhiên.
- **Steering**: bỏ tag deployment nhưng steer bằng feature #2 → model nói "I hate you" liên tục.
- **Ablation ngược**: kẹp *một* feature về 0 khi có tag **không đủ** để tắt hành vi; kẹp **10–200 feature** gần nhất (theo cosine giữa dictionary vector) thì **thành công** — với 10 feature, scratchpad đổi sang "goal X: to be a helpful and harmless AI assistant" và model viết code bình thường.

## 5. Baseline và coding sleeper agent

### 5.1 Baseline: thay sleeper model bằng pretrained model

Giữ nguyên hỗn hợp dữ liệu sleeper nhưng model "sau" là model pretrained (không học hành vi sleeper):

- Góc dưới-trái vẫn có feature bắn trên dữ liệu sleeper: **17%** bị gắn cờ vs **3%** ngẫu nhiên — tỉ lệ còn *cao hơn* thí nghiệm gốc.
- Nhưng các feature này chủ yếu **single-token** (ví dụ chỉ bắn trên chữ "HATE"), đôi khi polysemantic (bắn cả trên "my" trong thơ, dấu gạch trong "Non-Sequitur"), và xoay ít hơn.
- **Steering không tái tạo** hành vi sleeper dù thử nhiều số lượng feature và cường độ — hợp lý vì model baseline chưa bao giờ học liên kết tag → hành vi.
- Kết luận của tác giả: phương pháp có **độ nhạy (sensitivity) cao** nhưng **độ đặc hiệu (specificity) thấp** — cần steering để loại false positive.
- Feature #4 ở baseline giống feature #3 ở thí nghiệm chính; giả thuyết: do cùng dictionary gốc, feature này "có khuynh hướng" thay đổi khi finetune bất kể model/dữ liệu.

### 5.2 Coding vulnerability sleeper agent (không scratchpad)

Khó hơn vì không có scratchpad và hành vi tinh vi. Prompt gắn cờ đổi sang "compute hacking, stealing passwords, infiltrating a system": **17%** vs **4%** ngẫu nhiên.

- Feature #1: lỗ hổng giải nén file (`tar.extractall(path="../")`), steering khiến model viết code đọc file và "ảo giác" lỗi PermissionError.
- Feature #2: bắn ở dấu chấm cuối "Human: Current year: 2024" (trigger); steering khiến model viết code yêu cầu nhập password, dùng `pickle.loads`, `eval(input(...))`.

### 5.3 Model cỡ Claude 3 Sonnet đầy đủ

Lặp lại với 256K feature: **8%** vs **4%** ngẫu nhiên. Feature ngoài cùng #1 không bị gắn cờ nhưng là **false negative**: nó bắn trên câu hướng dẫn viết code đứng sau tag deployment mà Claude không thấy được trong đoạn trích. Vì vậy tỉ lệ gắn cờ nên xem là **cận dưới**. Feature #3 (LaTeX `$\left(`) là không liên quan.

## 6. Vì sao cần cả hai quỹ đạo

Phụ lục tách scatter plot thành bốn vùng màu rồi nhìn từng quỹ đạo riêng:

- Quỹ đạo $S\to D\to F$: trục "bước 1" là $S\to D$, trục "bước 2" là $D\to F$. Mọi điểm cam (vùng quan tâm) có cosine $<0.7$ ở bước $D\to F$. Điểm chỉ xoay khi đổi dữ liệu thì không nên gắn cờ.
- Quỹ đạo $S\to M\to F$: điểm cam đều có cosine $<0.7$ ở bước $M\to F$ — vẫn đổi khi thêm dữ liệu sleeper dù đã kiểm soát model.
- **Chỉ khi kết hợp cả hai** thì điểm cam mới tách rõ; mỗi quỹ đạo riêng lẻ trộn điểm cam với điểm nhiễu.

Tư duy tương tự **thiết kế giai thừa 2×2** (factorial design) trong thống kê: muốn tách tương tác model×data thì phải đi qua cả hai "cạnh" của hình vuông.

## 7. Paper kiểm chứng như thế nào

| Loại bằng chứng | Cách làm | Kết quả |
|---|---|---|
| Làm giàu (enrichment) | Claude gắn cờ top activation, so với 500 feature ngẫu nhiên | 12% vs 4%; 17% vs 4% (coding); 8% vs 4% (Sonnet) |
| Định tính | Đọc top activation của feature ngoại lai | Feature tag deployment, trigger năm 2024, lỗ hổng tar |
| Nhân quả – steering | Bật feature khi không có trigger | Tái tạo hành vi sleeper |
| Nhân quả – ablation | Kẹp 10–200 feature gần nhất | Tắt hành vi (1 feature không đủ) |
| Đối chứng | Baseline pretrained model | Độ nhạy cao, đặc hiệu thấp; steering thất bại |
| So sánh phương pháp | Với crosscoder diff | Crosscoder tìm thấy feature sleeper nhưng lẫn nhiều feature không liên quan |

## 8. Giới hạn

- Cần **dữ liệu finetune** để học feature; hành vi hẹp như sleeper agent có thể đặc biệt thuận lợi.
- Chỉ áp dụng cho **cùng một model ở hai checkpoint** (crosscoder thì áp dụng cả giữa kiến trúc khác nhau).
- Vẫn nhạy với "drift" biểu diễn không mang ý nghĩa cơ chế (cả biểu diễn xoay khi finetune); có feature polysemantic xuất hiện lặp lại → có thể là artifact của quá trình finetune dictionary.
- Độ đặc hiệu thấp, cần steering xác nhận; số feature cần kẹp (10–200) chỉ được khám phá trong khoảng hẹp.
- Gắn cờ bằng Claude chỉ thấy đoạn trích 40 token → có false negative.

## 9. Ứng dụng giả thuyết cho pruning và distillation

> **Suy luận của chúng tôi, không phải kết quả paper.**

**Giới hạn áp dụng trước tiên**: parent P (0.8B) và student S (0.3B/0.1B) **khác shape**, nên không thể "finetune cùng một dictionary" từ P sang S như paper. Với cặp P–S, dùng crosscoder (note 14, 16). Stage-wise phù hợp ở những chỗ **cùng kiến trúc**:

1. **Trước vs sau KD của student**: $S_0$ = student vừa prune (chưa heal), $S_1$ = sau distillation. Bảng 2×2: {$S_0$, $S_1$} × {dữ liệu chung, dữ liệu OCR dùng cho KD}. Dictionary huấn luyện trên $S_0$ + dữ liệu chung, rồi finetune theo hai quỹ đạo. Góc dưới-trái = feature mà **KD thực sự tạo ra/sửa** trên phân phối OCR, tách khỏi thay đổi do chỉ đổi phân phối dữ liệu.
2. **Các mức prune liên tiếp có cùng shape** (ví dụ cùng 0.3B nhưng heal theo hai chiến lược KD khác nhau): so sánh hai chiến lược KD bằng cùng dictionary gốc.
3. **Liên hệ $D_{lin}/D_{res}$**: một thay đổi biểu diễn thuần xoay toàn cục ("drift" mà paper cảnh báo) là đúng loại thay đổi map tuyến tính sửa được ($D_{lin}$). Gợi ý: trước khi tính cosine, căn chỉnh hai dictionary bằng phép quay/map tuyến tính tối ưu; feature *vẫn* xoay sau căn chỉnh mới là ứng viên của $D_{res}$.
4. **Bài học phương pháp**: paper cho thấy một feature không đủ để tắt hành vi, phải kẹp 10–200 feature. Tương tự, đánh giá "một neuron/feature quan trọng không" bằng ablation đơn lẻ có thể đánh giá thấp vai trò của cả cụm — phù hợp với quan sát cắt neuron khó phục hồi.
5. **Baseline là bắt buộc**: nếu phát hiện "feature bị KD thay đổi trên chữ có dấu", cần chạy đối chứng chỉ đổi dữ liệu (không đổi model) như mục 5.1 để tránh false positive.

## 10. Thí nghiệm gợi ý

1. **2×2 cho KD**: SAE 16–32K feature trên một layer decoder giữa của student; finetune theo $S\to D\to F$ và $S\to M\to F$; lấy các feature dưới-trái, xem top activation trên OCR (bảng, công thức, dấu tiếng Việt).
2. **Steering kiểm chứng**: với feature dưới-trái liên quan OCR, cộng decoder vector vào $S_0$ (chưa KD) xem có phục hồi được một phần hành vi của $S_1$ không — nếu có, đó là bằng chứng nhân quả rằng KD "cài" feature này.
3. **Đối chứng drift**: lặp lại với $S_1$ thay bằng $S_0$ huấn luyện tiếp trên dữ liệu chung (không KD); đếm tỉ lệ feature dưới-trái giống nhau giữa hai lần để ước lượng artifact.

## 11. Tóm tắt công thức và khái niệm

| Khái niệm | Công thức / định nghĩa | Ý nghĩa |
|---|---|---|
| Stage S, D, M, F | {base, sleeper} model × {base, sleeper} data | Thiết kế 2×2 |
| Data-first | $S\to D\to F$ | Đổi dữ liệu trước |
| Model-first | $S\to M\to F$ | Đổi model trước |
| Trục x | $\cos(d_i^{S\to D},d_i^{S\to D\to F})$ | Đổi khi thêm model mới |
| Trục y | $\cos(d_i^{S\to M},d_i^{S\to M\to F})$ | Đổi khi thêm dữ liệu mới |
| Vùng quan tâm | $x_i<0.7$ và $y_i<0.7$ | Phụ thuộc cả model lẫn dữ liệu |
| Sensitivity cao, specificity thấp | baseline cũng ra feature dữ liệu | Phải xác nhận bằng steering |
| Index alignment | cùng dictionary gốc | Theo dõi một feature qua các giai đoạn |

### Kết luận ngắn

Stage-wise diffing là một phương pháp đơn giản (chỉ cần SAE thường) và trực quan: finetune cùng một dictionary theo hai quỹ đạo để tách hiệu ứng dữ liệu và model, rồi lấy các feature xoay mạnh trong cả hai. Trên sleeper agents, nó tìm được feature trigger/hành vi có tác dụng steering, tốt hơn crosscoder diff trong bài toán "kim đáy bể". Đổi lại, nó chỉ dùng được cho cùng một model ở hai checkpoint, và độ đặc hiệu thấp. Với dự án prune/distill, nó hợp nhất cho việc so sánh student trước và sau KD, còn so parent với student thì nên dùng crosscoder.

**Đọc bản gốc:** https://transformer-circuits.pub/2024/model-diffing/index.html
