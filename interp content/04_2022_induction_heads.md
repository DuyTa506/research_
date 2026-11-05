# In-context Learning and Induction Heads (2022)

**Tác giả:** Catherine Olsson, Nelson Elhage, Neel Nanda (core research contributors), Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, Chris Olah (Anthropic). Đăng ngày 08/03/2022.  
**Nguồn:** https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html  
**Chủ đề:** Giả thuyết induction head là cơ chế của phần lớn in-context learning; phase change trong training; in-context learning score; sáu lập luận và mức độ bằng chứng theo cỡ mô hình; smeared key; ablation giữ pattern; head evaluator.  
**Vị trí trong lộ trình:** Đọc sau `02_2021_mathematical_framework.md` (cần QK/OV, K-composition, eigenvalue copying) và `03` (bài tập induction head). Đây là bước nhảy từ mô hình đồ chơi lên mô hình 13B: thay vì reverse engineer trọng số, paper xây một **lập luận gián tiếp** từ động học training, can thiệp kiến trúc và ablation — kiểu bằng chứng sẽ phổ biến trong các bài sau.

> Ghi chú học tập diễn giải, không phải bản dịch. Tách rõ: điều paper **cho thấy** / điều paper **giả thuyết** / điều **chúng ta suy luận**. Ứng dụng cho pruning/distillation là suy luận của chúng ta.

## Mục lục

1. [Luận đề và vì sao nó táo bạo](#1-luận-đề-và-vì-sao-nó-táo-bạo)
2. [Khái niệm then chốt](#2-khái-niệm-then-chốt)
3. [Bản đồ bằng chứng theo cỡ mô hình](#3-bản-đồ-bằng-chứng-theo-cỡ-mô-hình)
4. [Lập luận 1: phase change đồng xuất hiện](#4-lập-luận-1-phase-change-đồng-xuất-hiện)
5. [Lập luận 2: smeared key dời phase change](#5-lập-luận-2-smeared-key-dời-phase-change)
6. [Lập luận 3: knock out induction head](#6-lập-luận-3-knock-out-induction-head)
7. [Lập luận 4: induction head làm việc trừu tượng](#7-lập-luận-4-induction-head-làm-việc-trừu-tượng)
8. [Lập luận 5: cơ chế và khả năng tổng quát hoá](#8-lập-luận-5-cơ-chế-và-khả-năng-tổng-quát-hoá)
9. [Lập luận 6: tính liên tục từ nhỏ tới lớn](#9-lập-luận-6-tính-liên-tục-từ-nhỏ-tới-lớn)
10. [Mô hình, dữ liệu và phép đo](#10-mô-hình-dữ-liệu-và-phép-đo)
11. [Những điều chưa giải thích được](#11-những-điều-chưa-giải-thích-được)
12. [Thảo luận: an toàn, scaling law, động học học](#12-thảo-luận-an-toàn-scaling-law-động-học-học)
13. [Bình luận và tái lập từ bên ngoài](#13-bình-luận-và-tái-lập-từ-bên-ngoài)
14. [Paper kiểm chứng các tuyên bố như thế nào](#14-paper-kiểm-chứng-các-tuyên-bố-như-thế-nào)
15. [Giới hạn](#15-giới-hạn)
16. [Ứng dụng giả thuyết cho pruning / distillation](#16-ứng-dụng-giả-thuyết-cho-pruning--distillation)
17. [Thí nghiệm gợi ý](#17-thí-nghiệm-gợi-ý)
18. [Tóm tắt công thức / khái niệm](#18-tóm-tắt-công-thức--khái-niệm)

---

## 1. Luận đề và vì sao nó táo bạo

**Paper Framework đã cho thấy** (mô hình attention-only 2 layer): induction head là cặp previous-token head + induction head, hiện thực `[A][B] … [A] → [B]`, và là nguồn chính của in-context learning ở quy mô đó. Mô hình 1 layer không có induction head vì nó cần composition giữa hai layer.

**Paper này giả thuyết** — và nói rõ chỉ có bằng chứng "sơ bộ, gián tiếp":

$$
\boxed{\text{Induction head (dạng "mờ"/nearest-neighbor) là cơ chế của phần lớn in-context learning ở transformer lớn.}}
$$

"Mờ" nghĩa là hoàn thành mẫu theo nghĩa tương tự: `[A*][B*] … [A] → [B]` với $A^*\approx A$, $B^*\approx B$ trong một không gian biểu diễn nào đó.

Vì sao gián tiếp? Với nhiều layer và MLP, không thể "đọc" circuit từ trọng số như ở mô hình đồ chơi. Paper so sánh với thần kinh học: quan sát sự phát triển theo thời gian, nghiên cứu "tổn thương", can thiệp, và soi vài neuron chọn lọc. Dữ liệu: **34 transformer** theo dõi suốt quá trình training, **hơn 50.000 lần ablation head**.

Paper cũng muốn lập một chuẩn mực: **báo cáo bằng chứng gián tiếp ngay cả khi chưa kết luận được**, vì kiểu bằng chứng này sẽ phổ biến khi interp tiến lên mô hình lớn.

## 2. Khái niệm then chốt

### 2.1 In-context learning (ICL)

Hai cách hiểu trong tài liệu:

| Quan niệm | Đại diện | Đo bằng |
|---|---|---|
| Vi mô: few-shot cho tác vụ cụ thể | Brown et al. (GPT-3) | Độ chính xác trên tác vụ (cộng vài chữ số, dịch…) |
| **Vĩ mô: loss giảm theo vị trí token** | Kaplan et al. | Loss theo chỉ số token trong context |

Paper chọn quan niệm vĩ mô vì tổng quát hơn, không phụ thuộc lựa chọn tác vụ (nhược điểm: không cô lập hành vi cụ thể). Từ "meta-learning" được tránh vì nó ngầm khẳng định mô hình *học* khả năng mới thay vì *định vị* khả năng đã có.

**In-context learning score:**

$$
\boxed{\;\text{ICL} = \mathbb E\big[\mathcal L(\text{token }50)\big]-\mathbb E\big[\mathcal L(\text{token }500)\big]\;}
$$

(trung bình trên ví dụ dữ liệu, context dài 512). Lưu ý dấu: phần Key Concepts và phụ lục viết "loss token 500 trừ loss token 50" (giá trị âm), còn Lập luận 1 viết "50 trừ 500" và báo giá trị dương ~0,4 nat. Ghi chú này dùng dạng dương. Token 50 đủ xa để ngôn ngữ/loại văn bản đã rõ; token 500 gần cuối context 512. Paper cho thấy đổi hai chỉ số này không đổi kết luận (§11).

### 2.2 Định nghĩa thực nghiệm của induction head

Paper Framework định nghĩa hơi mơ hồ ("một cụm hành vi và cơ chế"). Ở đây, vì phân tích trọng số khó với MLP, induction head được **định nghĩa theo hành vi trên chuỗi token ngẫu nhiên lặp lại**, với hai tính chất:

| Tính chất | Thuộc circuit | Mô tả |
|---|---|---|
| **Prefix matching** | QK | Attend về token mà induction dự đoán sẽ đến tiếp: token đứng *sau* lần xuất hiện trước của token hiện tại (hoặc của vài token gần đây) |
| **Copying** | OV | Output của head tăng logit của chính token được attend |

Dùng chuỗi *ngẫu nhiên* để chắc chắn head dựa vào induction, không phải một copying head đơn giản đoán token "hợp lý". Thực tế điểm số là một dải liên tục, nhưng có một tập head rõ ràng vượt xa mức ngẫu nhiên. (Change log 20/09/2022: paper đã sửa câu chữ của định nghĩa prefix matching vì từng viết nhầm "preceded"/"followed".)

Hệ quả: induction head lặp lại cả chuỗi. Với "The cat sat on the mat. The cat …", chúng đẩy "sat on the mat" — một dạng few-shot đơn giản.

Đặc điểm quan trọng: induction head **tách rời A và B** — không có bảng bigram cố định, nên chạy được *off-distribution* miễn thống kê đầu context đại diện cho phần sau.

### 2.3 Per-token loss analysis

Coi mỗi mô hình (mỗi snapshot) là một vector trong "không gian hàm": log-likelihood trên **10.000 token ngẫu nhiên**, mỗi token từ một ví dụ khác nhau. Ghép mọi snapshot của mọi mô hình rồi **PCA**. Mỗi thành phần chính là một vector log-likelihood mà các mô hình "di chuyển theo". Cách này bỏ qua việc hai mô hình giống nhau về hàm có thể có tham số rất khác.

## 3. Bản đồ bằng chứng theo cỡ mô hình

Bảng tổng (mạnh nhất cho mỗi ô), theo paper:

| Tuyên bố | Small attention-only | Small with MLPs | Large models |
|---|---|---|---|
| Induction head **đóng góp một phần** ICL | Strong, Causal | Strong, Causal | Medium, Correlational & Mechanistic |
| Induction head đóng góp **phần lớn** ICL | Strong, Causal | Medium, Causal | Medium, Correlational |

Chi tiết từng lập luận (— = lập luận không áp dụng cho ô đó):

| Lập luận | Tuyên bố | Small AO | Small MLP | Large |
|---|---|---|---|---|
| 1. Đồng xuất hiện | Some | Medium, Correlational | Medium, Correlational | Medium, Correlational |
| | Majority | Medium, Correlational | Medium, Correlational | Medium, Correlational |
| 2. Đồng nhiễu loạn (smeared key) | Some | Medium, Interventional | Medium, Interventional | Weak, Interventional |
| | Majority | Medium, Interventional | Medium, Interventional | Weak, Interventional |
| 3. Ablation trực tiếp | Some | Strong, Causal | Strong, Causal | — |
| | Majority | Strong, Causal | Medium, Causal | — |
| 4. Ví dụ tổng quát hoá | Some | — | — | Plausibility (mạnh) |
| | Majority | — | — | Plausibility (gợi ý) |
| 5. Cơ chế | Some | Strong, Mechanistic | Strong, Mechanistic | Medium, Mechanistic |
| | Majority | Weak, Mechanistic | — | — |
| 6. Liên tục nhỏ → lớn | Some | — | — | Analogy |
| | Majority | — | — | Analogy |

Đọc nhanh: **càng lớn, bằng chứng "phần lớn" càng dựa vào tương quan và loại suy**; bằng chứng nhân quả mạnh chỉ có ở mô hình nhỏ.

## 4. Lập luận 1: phase change đồng xuất hiện

**Quan sát (paper cho thấy):** trong một cửa sổ hẹp sớm trong training — khoảng **2,5 đến 5 tỉ token** — bốn điều xảy ra cùng lúc:

1. **ICL tăng vọt:** trước cửa sổ < 0,15 nat, sau cửa sổ khoảng **0,4 nat**, rồi giữ hằng số cho tới hết training và **gần như như nhau giữa các cỡ mô hình** (trừ mô hình 1 layer, nơi ICL không bao giờ phát triển nhiều).
2. **Induction head hình thành** (prefix matching score tăng đột ngột); mô hình 1 layer không bao giờ có.
3. **Bump trên loss curve:** đoạn duy nhất loss *không lồi* (độ dốc không giảm đơn điệu). Paper nhấn mạnh: loss là trung bình trên rất nhiều token; nhiều hành vi thú vị như số học "vi mô" trên loss curve, nên thấy được ở quy mô này nghĩa là một thay đổi lớn và rộng. Đây cũng là chỗ đầu tiên loss curve tách khỏi mô hình 1 layer.
4. **Quỹ đạo PCA đổi hướng** đúng trong cửa sổ đó.

Paper gọi đây là **"phase change"** (chuyển pha), ví như băng tan: đột ngột, có biểu hiện vĩ mô (loss, ICL) và vi mô (induction head). Với mô hình lớn, cửa sổ này cỡ **1–2% quá trình training**. Hơn **75% ICL cuối cùng** hình thành trong cửa sổ.

**Kiểm tra độ bền:** dùng đạo hàm loss theo $\log(\text{chỉ số token})$, đại khái "ICL cho mỗi ε% context thêm vào". Trước phase change, loss gần như ngừng cải thiện quanh token 50; sau đó tiếp tục giảm xa hơn.

**Xem kỹ hơn:**

- Không trùng với thay đổi lịch learning rate, warmup hay weight decay.
- Train mô hình nhỏ trên dataset khác (chỉ sách) → phase change vẫn xuất hiện. (Nhưng ngoài thí nghiệm này, mọi mô hình thấy cùng token theo cùng thứ tự.)
- Giai thoại trên đoạn đầu Harry Potter: thay đổi chủ yếu ở token **lặp lại**. Token cuối của " The Dursleys" (lặp nhiều lần) tốt lên rõ; lần đầu " Mrs Potter" (hai lần trước " Mrs" đều đi với " Dursley") lại *tệ đi* — đúng chữ ký của induction.

**Đánh giá bằng chứng (chính paper tự phản biện):**

| Ủng hộ nhân quả | Lý do thận trọng |
|---|---|
| Tương quan chặt trên nhiều mô hình lớn nhỏ | Mô hình lớn chỉ có ~15 snapshot → trùng thời điểm ít gây ngạc nhiên hơn |
| Khó trùng hợp ngẫu nhiên giữa hai chuyển đổi sắc nét | **Biến ẩn chung:** có thể phase change là lúc mô hình học *compose layer qua residual*, sinh ra cả induction head lẫn cơ chế ICL khác |
| Lời giải thích đơn giản nhất | ICL score hằng số ≠ cơ chế hằng số: giảm cùng 0,4 nat từ baseline thấp hơn là *khó hơn*, có thể cần cơ chế mới về sau |

Kết luận của paper: luận điểm "induction head giải thích phần lớn ICL **ở thời điểm phase change**" vững hơn luận điểm đó **ở cuối training**.

## 5. Lập luận 2: smeared key dời phase change

**Ý tưởng can thiệp:** nếu induction head là cơ chế chính, thì yêu cầu kiến trúc tối thiểu cho ICL lớn chính là yêu cầu của induction head: key phải là hàm của token được attend **và token trước nó**. Một head đơn lẻ chỉ tính điểm từ token nguồn và đích, nên cần head khác ghi sẵn thông tin token trước → cần ≥ 2 layer.

**Smeared key:** mỗi head có một tham số học được $\alpha^h$:

$$
\boxed{\;k^h_j=\sigma(\alpha^h)\,k^h_j+\big(1-\sigma(\alpha^h)\big)\,k^h_{j-1}\;}
$$

(token đầu tiên giữ nguyên). Key giờ "nhoè" sang token trước, nên một layer cũng đủ cho induction.

**Dự đoán và kết quả:** nếu giả thuyết đúng, mô hình 1 layer có smeared key sẽ có phase change, và mô hình sâu hơn có phase change sớm hơn. Kết quả khớp: **ICL hình thành ở mô hình 1 layer** (vốn không có) và **sớm hơn ở mô hình 2 layer**. Smeared key chỉ được thử ở 1 và 2 layer, cấu hình còn lại như mô hình nhỏ.

**Giới hạn paper nêu:** thí nghiệm gợi ý induction head là *cơ chế tối thiểu* cho ICL tăng mạnh; nhưng ở mô hình lớn có thể còn cơ chế khác, và nó không bác bỏ việc cơ chế thay đổi theo thời gian.

(Nối với ghi chú 02 §12: Primer dùng depthwise conv trên 3 vị trí khi tính Q/K/V — paper Framework nhận xét điều này cũng cho phép induction head không cần K-composition.)

## 6. Lập luận 3: knock out induction head

**Phương pháp:** bỏ một head lúc test (không train lại). Paper dùng **pattern-preserving ablation**:

| | Full ablation | Pattern-preserving ablation (dùng trong paper) |
|---|---|---|
| Cách làm | Thay result vector của head bằng 0 | Lần 1: chạy, lưu mọi attention pattern. Lần 2: thay result vector bằng 0 **và ép mọi pattern bằng bản đã lưu** |
| Ảnh hưởng | Mọi tính toán phía sau (Q, K, V, pattern) | Chỉ V phía sau; đóng góp của head vào pattern của head sau được giữ |
| Ở attention-only | — | Tương đương trừ mọi số hạng chứa head đó khỏi logit (vì pattern cố định ⇒ tuyến tính) |

Điều này khả thi nhờ insight của paper Framework: QK và OV tách rời được.

**Kết quả (mô hình nhỏ):** ablate induction head làm ICL giảm mạnh. Với mô hình **attention-only nhỏ, gần như toàn bộ ICL đến từ induction head**, từ đầu phase change tới hết training. Chi tiết thú vị: ablate *phần lớn head khác* lại **tăng** ICL score — vì có token dự đoán được bằng cả "dự đoán thường" lẫn ICL; làm hỏng dự đoán thường thì ICL có thêm việc để làm.

**Vì sao không có cho mô hình lớn:** chi phí ablation đầy đủ tăng siêu tuyến tính $O(N^{1.33})$ ($O(N^{0.33})$ head, mỗi ablation $O(N)$), mỗi ablation chạy trên 10.000 ví dụ cho mỗi checkpoint.

**Diễn giải thận trọng:**

- Attention-only: logit là tổng số hạng từng head + direct path (direct path chỉ phụ thuộc token hiện tại nên không đóng góp ICL) ⇒ ICL *phải* đến từ head, và ablation (pattern cố định) là phép đo có nguyên tắc.
- Có MLP: ICL có thể đến từ tương tác MLP–attention. Ablation có thể "làm hỏng" neuron bằng cách dịch bias hiệu dụng, hoặc một cơ chế MLP cần hai head nhưng chạy tạm được với một.
- Hai head làm việc giống nhau + LayerNorm trước logit co giãn lại → tầm quan trọng từng head có thể bị che.

Kết luận của paper: induction head là cơ chế chính cho ICL ở mô hình attention-only nhỏ; với MLP thì chỉ là **gợi ý**.

## 7. Lập luận 4: induction head làm việc trừu tượng

Tìm head **thoả định nghĩa hẹp** (copy chuỗi ngẫu nhiên bằng prefix matching) nhưng **đồng thời** làm việc tinh vi hơn, trong mô hình **40 layer, 13B tham số**:

| Head | Độ sâu | Copying score | Prefix matching score |
|---|---|---|---|
| Literal copying | 21/40 | 0,89 | 0,75 |
| Translation | 7/40 | 0,20 | 0,85 |
| Pattern-matching | 8/40 | 0,69 | 0,94 |

(Mâu thuẫn nhỏ trong nguồn: bảng ghi pattern-matching head ở 8/40, phần văn bản nói "layer 26"; câu mở đầu cũng nói "study up to 12B" trong khi mô hình lớn nhất được mô tả là 13B.)

**Hành vi 1 — copy nguyên văn:** trên Harry Potter có lặp đoạn, head dự đoán "Dursley", "Potters", "a small son" và cả câu lặp, bằng cách attend về lần xuất hiện trước. Giao diện hiển thị *attention* và *logit attribution* (direct path: value × attention × $W_O$ × unembedding, logit chuẩn hoá trung bình 0).

**Hành vi 2 — dịch:** head ở layer 7 dịch giữa Anh, Pháp, Đức. Pattern gần "off-diagonal" nhưng **uốn lượn** vì thứ tự từ và độ dài token khác nhau giữa ngôn ngữ. Logit attribution không sắc nét — paper đoán output cần layer sau xử lý tiếp — nhưng tổng thể góp phần đúng.

**Hành vi 3 — khớp mẫu (học hàm trong context):** dữ liệu tổng hợp, mỗi dòng theo một trong bốn template và nhãn:

```text
(month) (animal): 0     (month) (fruit): 1
(color) (animal): 2     (color) (fruit): 3
```

Từ dấu ":", head attend về dòng trước có **cùng loại** hơn là trùng từ (vd. "January bird" attend chủ yếu "April fish", không phải "grey bird"); khoảng **65%** attention từ dấu ":" đi đúng vị trí trên các bài tương tự.

**Vì sao cùng một head lại làm cả hai?** Paper đưa hai khả năng: (a) copy nguyên văn là trường hợp đặc biệt của một thuật toán tổng quát hơn (dịch Anh→Anh = copy); (b) head copy nguyên văn khi đi qua đường residual chỉ có nó, nhưng làm việc trừu tượng khi đọc output của layer trước đã tạo biểu diễn trừu tượng (vd. cùng một từ Anh/Pháp nằm cùng chỗ). Paper nhấn mạnh: đây **không phải ẩn dụ** — đúng cùng một head ở cùng layer.

## 8. Lập luận 5: cơ chế và khả năng tổng quát hoá

**Lập luận bán thực nghiệm:** nếu induction head hoạt động như mô tả (tìm ví dụ trước trong context, copy cái đến tiếp), nó là **nearest neighbor trên context**, và nearest neighbor tốt lên khi có thêm điểm dữ liệu ⇒ phải đóng góp ICL. Lập luận này mạnh cho "có đóng góp", không nói gì về *tỉ lệ*.

**Bỏ phụ thuộc vào quan sát:** với mô hình mà ta reverse engineer được (paper Framework), ta biết head *sẽ* làm induction chỉ từ tham số. Ở mô hình attention-only 2 layer, ứng viên khác duy nhất là copying head cơ bản — nhưng chúng có cả ở mô hình 1 layer, vốn không có ICL lớn ⇒ induction head là động lực chính, ít nhất ở mô hình rất nhỏ.

**Tóm tắt cơ chế** (nhắc lại từ ghi chú 02 §8):

- **Copying (OV):** "copying matrix", nhận diện bằng eigenvalue dương. Induction head là *tập con* của các copying head.
- **Prefix matching (QK):** key tại token được attend cần thông tin về **token trước nó** (bản thân token được attend gần như không quan trọng *cho pattern*, nhưng cực kỳ quan trọng cho output qua OV). Dạng cơ bản: K-composition với previous-token head, số hạng
$$\boxed{\mathrm{Id}\otimes A^{h_\text{prev}}\otimes W,\quad W \text{ có eigenvalue dương}}$$
Head phức tạp hơn còn dịch cả query (Q-composition) để khớp nhiều hơn một token.
- **Cơ chế thứ hai ở GPT-2: pointer arithmetic** — head trước copy *vị trí* của lần xuất hiện trước, induction head dùng Q-composition quay vị trí đó tới một bước (xem bài tập 2 trong ghi chú 03). Không có ở mô hình của paper vì vị trí không vào residual.

**Induction head phức tạp ở mô hình lớn** chưa reverse engineer được (có MLP). Giả thuyết: (1) số hạng QK phức tạp hơn "chỉ token trước"; (2) khớp và copy **feature ngôn ngữ trừu tượng** thay vì token chính xác — tự nhiên với góc nhìn "in-context nearest neighbor".

## 9. Lập luận 6: tính liên tục từ nhỏ tới lớn

Mọi đo đạc trong Model Analysis Table trông **tương tự** giữa small AO, small MLP và full-scale: (nếu > 1 layer) đều có phase change, cùng mức ICL trước/sau, quỹ đạo PCA tương tự, đều hình thành induction head. Nếu cơ chế đổi khi lên mô hình lớn thì đổi ở đâu, và vì sao không có dấu hiệu nào trong số liệu?

Phản biện paper tự nêu: nhiều thứ thay đổi gián đoạn theo cỡ mô hình; ngoại suy qua nhiều bậc độ lớn phải thận trọng. **Giả thuyết thay thế mạnh nhất:** mô hình lớn có nhiều head hơn nên đủ capacity cho các cơ chế Q/K-composition *khác*; nếu mọi "composition head" cùng hình thành trong phase change, thì trên một cỡ nào đó, head composition không-phải-induction có thể gộp lại chiếm phần lớn hơn.

## 10. Mô hình, dữ liệu và phép đo

### 10.1 Bốn họ mô hình (34 mô hình decoder-only, một lần train mỗi mô hình)

| Họ | Mô tả |
|---|---|
| Small attention-only | 1–6 layer, không MLP |
| Small with MLPs | 1–6 layer, có MLP |
| Full-scale | 4 layer/13M tới 40 layer/13B, có MLP, dense + local attention |
| Smeared key | 1 và 2 layer, như small nhưng thêm smeared key |

**Mô hình nhỏ:** context 8192, vocab $2^{16}$, $d_\text{model}=768$, 12 head/layer, 10.000 step (~10 tỉ token), 200 snapshot mỗi 50 step. Phase change ở mô hình nhỏ khoảng **1–3 tỉ token**. Weight decay giảm ở step 4750 (~5 tỉ token, sau phase change); warmup LR trong 1,5 tỉ token đầu. Dữ liệu: bản sớm của dataset Askell et al. (common crawl lọc, sách, ~10% code Python); một bộ mô hình nhỏ khác train trên chỉ sách.

**Full-scale** ($d_\text{model}=128\cdot n_\text{layer}$; snapshot tại step $2^5$…$2^{17}$ cách nhau ×2, cộng 1–2 bản cuối, tổng 15; riêng 40L có 14):

| $n_\text{layer}$ | Tham số (non-embedding) | $d_\text{model}$ | Head/layer | $d_\text{head}$ |
|---|---|---|---|---|
| 4 | 13M | 512 | 8 | 64 |
| 6 | 42M | 768 | 12 | 64 |
| 10 | 200M | 1280 | 20 | 64 |
| 16 | 810M | 2048 | 32 | 64 |
| 24 | 2.7B | 3072 | 48 | 64 |
| 40 | 13B | 5120 | 40 | 128 |

### 10.2 Head activation evaluator (trung bình trên 10 ví dụ)

- **Copying:** 25 token ngẫu nhiên (bỏ token quá phổ biến/quá hiếm); chiếu output head qua unembedding (direct path); trừ trung bình logit, qua ReLU; tỉ lệ mức tăng logit của token được attend so với tổng các token. Giá trị gốc 0–0,5, co giãn về $[-1,1]$.
- **Prefix matching:** 25 token ngẫu nhiên lặp 4 lần, thêm token BOS. Điểm là trung bình các phần tử attention từ một token về vị trí induction tương ứng trong các lần lặp trước. Viết theo ký hiệu (cách diễn đạt của chúng ta, theo định nghĩa ở §2.2):
$$\text{PM}(h)=\mathbb E_i\Big[\textstyle\sum_{m\ge1}A^h_{\,i,\;i-25m+1}\Big].$$
(Paper ghi chú: lẽ ra evaluator copying cũng nên có BOS; thiếu BOS làm vài head prefix-matching có điểm thấp bất thường.)
- **Previous token:** trung bình $A_{i,i-1}$ trên một ví dụ từ dữ liệu train.

**Kiểm định evaluator** (mô hình attention-only nhỏ): copying score tương quan tốt với $\sum\lambda_i/\sum|\lambda_i|$ của OV; prefix matching tương quan tốt với trace eigenvalue của số hạng QK previous-token. Head có prefix matching dương có xu hướng có copying dương, càng mạnh ở điểm prefix matching cao.

### 10.3 Trace eigenvalue QK (thước đo dựa trên trọng số)

1. Với mỗi head ở layer trước, đo mức "previous-token" (trung bình attention trên đường chéo lệch 1).
2. Tính $W_k = W_E^\top W^h_{QK}W^{h_k}_{OV}W_E$ trong số hạng $\mathrm{Id}\otimes A^{h_k}\otimes W_k$.
3. Gộp $W=\sum_k(\text{prev-score}_k)\,W_k$.
4. Lấy $\mathrm{tr}(W)=\sum\lambda_i$: dương lớn ⇒ "same-matching".

Chỉ định nghĩa cho head từ layer 2 trở đi.

### 10.4 Các "view" của dữ liệu loss

- **Loss curve**, **2D ICL plot** (snapshot × vị trí token, kỹ thuật từ Kaplan et al.), **2D loss derivative** ($\partial\mathcal L/\partial\log i_\text{ctx}$), **ICL score**, **PCA**.
- **Attribution vào before-and-after vector:** $u=\mathrm{normalize}\big(\ell_\text{end}-\ell_\text{start}\big)\in\mathbb R^{10\,000}$ (thay đổi per-token loss qua phase change); với ablation head $h$, $\delta_h=\ell_\text{ablated}-\ell$; điểm $=u^\top\delta_h$. Induction head cho **giá trị âm**: bỏ chúng giống như *đảo ngược* phase change.

### 10.5 Induction head nằm ở đâu

| Họ | Vị trí |
|---|---|
| Small attention-only | Layer **cuối** |
| Small with MLPs (5L, 6L) | Chủ yếu layer **áp chót** |
| Full-scale | Sớm hơn; ở 24L và 40L, **phần lớn trước nửa độ sâu** |

## 11. Những điều chưa giải thích được

- **ICL score gần như hằng số** sau phase change, bất kể 2 layer hay 13B, mới qua phase change hay train lâu. Đổi định nghĩa (loss token cuối 8192 trừ các chỉ số khác) vẫn cho khác biệt nhỏ giữa mô hình. Giải thích một phần: mô hình lớn **lấy hết lợi thế rất sớm** — phần lớn khác biệt hình thành trong **10 token đầu** — rồi giảm thêm một lượng gần cố định. Từ baseline thấp hơn, lượng đó "khó hơn". Paper: điều này "lạ" chứ không còn "sốc".
- **Thứ tự đạo hàm loss đảo ở phase change:** trước đó mô hình nhỏ giảm loss *chậm hơn* mô hình lớn (theo log token), sau đó ngược lại.
- Trong Model Analysis Table: head bất thường ở 6L attention-only (không phải induction, nhưng ablate nó giống đảo phase change); ablation của 4L MLP không "nhọn"; 6L MLP có loss spike; 6L MLP có một induction head mà ablate cho hiệu ứng *ngược* lên ICL; mô hình > 16 layer có vài head prefix-search nhưng **copying âm** ("anti-copying prefix-search").

## 12. Thảo luận: an toàn, scaling law, động học học

- **An toàn:** phase change cho thấy hành vi có thể xuất hiện đột ngột (vd. reward hacking có thể nổi lên trong phase change); ICL nghĩa là hành vi "thay đổi" lúc inference mà không train. Về **mesa-optimization**: paper cho rằng cơ chế chính của ICL, ít nhất ở mô hình nhỏ, là induction head, và **không thấy bằng chứng mesa-optimizer**.
- **"Rosetta stone":** phase change nối mech interp (vi mô) với scaling law và động học học (vĩ mô), như vật lý thống kê nối hạt với nhiệt động học. Paper cho rằng đây là trường hợp đầu tiên họ biết có lời giải thích cơ chế cho một phase change trong ML (gần nhất với **grokking** vì xảy ra theo thời gian training). Nó cũng có thể giải thích vì sao mô hình 1 layer không theo cùng scaling law (Kaplan et al.): không có induction head.
- **Liên hệ:** Saxe et al. (động học mạng tuyến tính như các "đường" độc lập ~ circuit); hình học loss landscape (phase change hẳn ứng với một đặc điểm lớn của landscape); **universality** (induction head như một pattern và circuit phổ quát).
- **Attention trong tác vụ kiểu dịch:** dịch máy và **nhận dạng giọng nói** thường attend vào phần sắp được dịch/chép; paper nhận xét chúng "có vẻ khá giống" induction, chỉ bị che vì cách visualize decoder theo output token.

## 13. Bình luận và tái lập từ bên ngoài

- **Adam Scherlis (Redwood Research):** induction head xuất hiện ổn định ở mô hình attention-only 2 layer (previous-token head ở layer 0, induction head ở layer 1). Thay attention score bằng bản lý tưởng: previous-token chính xác phục hồi **99%** chênh lệch loss (so với ablate); induction head xấp xỉ đơn giản (attend token đầu + khớp `[A][B]…[A]`) phục hồi **~65%**; thêm mẫu `[A][B][C]…[A][B]→[C]` phục hồi thêm **10%**. OV copy token (kể cả khớp mờ token gần nghĩa); QK bị chi phối bởi K-composition: OV của previous-token head copy sang một subspace mới, QK của induction head copy ngược về embedding thường.
- **Tom Lieberum (Amsterdam):** dùng tiêu chí thực nghiệm, thấy ứng viên induction head ở GPT-2 và GPT-Neo, chủ yếu từ giữa độ sâu (vd. GPT2-XL layer 21 head 20; GPT-Neo-2.7B layer 12 head 0). Trên dữ liệu tổng hợp, induction học được dễ với **context dài 4**; ông cho rằng đây là tính chất của phân phối dữ liệu hơn là của mô hình.

## 14. Paper kiểm chứng các tuyên bố như thế nào

| Kiểu bằng chứng | Lập luận | Con số chính |
|---|---|---|
| Tương quan theo thời gian (34 mô hình) | 1 | Cửa sổ 2,5–5 tỉ token; ICL < 0,15 → ~0,4 nat; > 75% ICL cuối hình thành trong cửa sổ |
| Can thiệp kiến trúc | 2 | Smeared key: ICL xuất hiện ở 1 layer, sớm hơn ở 2 layer |
| Ablation nhân quả (pattern-preserving) | 3 | > 50.000 ablation; gần như toàn bộ ICL ở small AO đến từ induction head |
| Ví dụ định tính + evaluator | 4 | Bảng 3 head (copy 0,89/0,20/0,69; prefix 0,75/0,85/0,94); ~65% attention đúng |
| Lập luận cơ chế | 5 | Dựa trên reverse engineering từ paper Framework |
| Loại suy | 6 | Các đường cong trông tương tự giữa ba họ |
| Kiểm định thước đo | Phụ lục | Evaluator ↔ eigenvalue OV/QK tương quan tốt |
| Tái lập độc lập | Comments | Redwood (99%/65%/+10%), Lieberum (GPT-2, GPT-Neo) |

## 15. Giới hạn

- **Bằng chứng ở mô hình lớn chủ yếu tương quan và loại suy**; không có ablation cho full-scale; chỉ ~15 snapshot.
- **Biến ẩn chung** (năng lực compose layer nói chung) chưa loại trừ.
- **ICL score** chỉ là heuristic vĩ mô, không cô lập hành vi; có thể *tăng* khi ablate head khác (dự đoán thường tệ đi) — tức không đơn điệu theo "năng lực".
- Định nghĩa induction head **thực nghiệm**, cho dải liên tục; ngưỡng là quy ước.
- Ablation ở mô hình có MLP khó diễn giải (tương tác MLP–attention, dư thừa giữa head, LayerNorm).
- Mọi mô hình (trừ thí nghiệm dataset) thấy cùng dữ liệu theo cùng thứ tự; mỗi cấu hình chỉ train một lần.
- Smeared key chỉ thử ở 1–2 layer. Induction head phức tạp ở mô hình lớn chưa được reverse engineer.
- Vài mâu thuẫn nhỏ trong văn bản (dấu ICL score; layer của pattern-matching head; 12B vs 13B).

## 16. Ứng dụng giả thuyết cho pruning / distillation

> **Toàn bộ là suy luận của chúng ta.** Không có kết quả nào ở đây được paper chứng minh cho OCR, VLM hay GatedDeltaNet.

**16.1. OCR là "transcription" — họ hàng của induction.** Paper nhận xét attention trong dịch và nhận dạng giọng nói (attend phần sắp được chép) giống induction. Giả thuyết: trong decoder OCR, head đọc token ảnh làm một dạng `[A*][B*] … [A] → [B]` xuyên modality — query = token vừa sinh, key = vùng ảnh *đứng sau* vùng chứa ký tự vừa chép, value/OV = nội dung vùng đó. Nếu đúng, chúng cần cùng loại cơ chế "dịch key một bước" (previous-patch information trong key), và **cắt layer cung cấp thông tin đó sẽ phá OCR theo kiểu bỏ/lặp dòng, nhảy cột**.

**16.2. Text-to-text induction vẫn quan trọng trong OCR.** Tài liệu có header, tên riêng, đơn vị, cột bảng lặp lại; induction head giúp sinh lại chuỗi đã sinh. Paper cho thấy chúng chiếm phần lớn ICL ở mô hình nhỏ ⇒ một phần $D_\text{res}$ khi cắt có thể nằm ở **token lặp**, khó thấy trong loss trung bình.

**16.3. Chọn layer để cắt.** §10.5: ở mô hình lớn induction head nằm sớm/giữa; ở mô hình nhỏ có MLP thì gần cuối. Khi thu từ 0.8B về 0.3B/0.1B, vị trí "an toàn để cắt" có thể dịch chuyển. Quy tắc gợi ý: tính prefix-matching và previous-token score cho mọi head full-attention trước khi chọn layer; không cắt đồng thời layer chứa previous-token head và layer chứa induction head nó nuôi.

**16.4. Smeared key / short conv cho student nhỏ.** Lập luận 2 cho thấy một tham số/head đủ để mô hình 1 layer có induction. Với student 0.1B có ít layer full-attention, thêm smeared key là thay đổi rẻ có thể giữ ICL. Đáng chú ý: theo hiểu biết của chúng ta về GatedDeltaNet, các lớp này có **short causal convolution** trên q/k/v — đúng loại cơ chế mà paper Framework (về Primer) nói cho phép induction mà không cần K-composition. Cần kiểm tra trong config LightOnOCR; nếu đúng, lớp linear-attention có thể tự làm "dịch key" và cắt chúng có hậu quả khác cắt lớp full-attention.

**16.5. Distillation và phase change.** Nếu student được train lại đủ lâu (hoặc từ đầu), có thể có phase change với bump trên loss KD; student kế thừa từ teacher đã prune có thể mất induction head rồi phải "học lại". Theo dõi ICL score theo checkpoint KD là chẩn đoán rẻ. Giả thuyết: thêm KD loss trên attention pattern của *vài* induction head teacher có thể đẩy phase change sớm hơn.

**16.6. Bẫy đo lường khi đánh giá pruning.** Paper cho thấy ablate head khác **làm tăng** ICL score. Tương tự, model đã prune có thể có ICL score *cao hơn* teacher chỉ vì dự đoán thường tệ đi. Luôn báo cáo kèm loss tuyệt đối ở token 50 và token 500.

**16.7. Tách damage QK vs OV bằng pattern-preserving.** Sau khi cắt, chạy student với **attention pattern của teacher bị ép vào** (cho các layer còn tương ứng). Phần loss được phục hồi = damage do *chọn sai chỗ đọc* (QK); phần còn lại = damage do *nội dung chuyển đi* (OV/MLP). Đây là cách chia $D_\text{res}$ thành hai phần có nghĩa cơ chế.

## 17. Thí nghiệm gợi ý

1. **Head census trên LightOnOCR (teacher và từng student).** Chạy ba evaluator của §10.2 trên (a) token ngẫu nhiên lặp 4 lần (chỉ text), (b) ảnh render chuỗi ký tự ngẫu nhiên lặp nhiều dòng. Vẽ prefix matching × copying cho mọi head full-attention, đánh dấu layer. So sánh: student 0.3B/0.1B còn bao nhiêu head ở góc "induction"? Tương quan với lỗi lặp/bỏ dòng.
2. **ICL score và per-token loss theo vị trí trên tài liệu dài.** Teacher vs student sau prune vs student sau KD, kèm loss tuyệt đối ở token 50/500 (bẫy §16.6). Trong quá trình KD, lưu checkpoint dày ở đầu để tìm bump/phase change; nếu có, kiểm tra induction head có xuất hiện cùng lúc không.
3. **Pattern-preserving vs full ablation cho vết cắt.** Với mỗi layer ứng viên: (i) cắt hoàn toàn; (ii) cắt nhưng ép attention pattern của các layer sau bằng của teacher. Hiệu (i)−(ii) ước lượng phần damage đi qua QK (composition bị đứt); so với $D_\text{lin}/D_\text{res}$ bạn đang đo.

## 18. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / định nghĩa | Ý nghĩa |
|---|---|---|
| ICL score | $\mathcal L_{50}-\mathcal L_{500}$ (≈ 0,4 nat sau phase change) | ICL vĩ mô |
| Induction head | Prefix matching (QK) + copying (OV) trên chuỗi ngẫu nhiên lặp | Định nghĩa thực nghiệm |
| Fuzzy induction | `[A*][B*] … [A] → [B]` | In-context nearest neighbor |
| Phase change | Cửa sổ 2,5–5 tỉ token: ICL tăng, induction head, bump loss, PCA đổi hướng | Bước ngoặt training |
| Smeared key | $k_j=\sigma(\alpha)k_j+(1-\sigma(\alpha))k_{j-1}$ | Induction với 1 layer |
| Pattern-preserving ablation | Bỏ output head, ép pattern đã lưu | Chỉ ảnh hưởng V phía sau |
| Before-and-after attribution | $u^\top\delta_h$, $u=\mathrm{normalize}(\ell_\text{end}-\ell_\text{start})$ | Âm ⇒ ablate ≈ đảo phase change |
| Copying evaluator | Tỉ lệ tăng logit token được attend (sau ReLU), co về $[-1,1]$ | Tính chất OV |
| Prefix-matching evaluator | Attention trung bình về vị trí induction trên 4 lần lặp | Tính chất QK |
| QK eigen trace | $\mathrm{tr}\big(\sum_k s_k\,W_E^\top W_{QK}W^{h_k}_{OV}W_E\big)$ | Thước đo trọng số cho prefix matching |
| Chi phí ablation | $O(N^{1.33})$ | Vì sao không có ablation cho mô hình lớn |

### Kết luận ngắn

Paper xây một lập luận sáu nhánh rằng induction head — previous-token head + head prefix-matching/copying — là cơ chế của phần lớn in-context learning. Ở mô hình attention-only nhỏ, bằng chứng là nhân quả (ablation, smeared key, reverse engineering). Ở mô hình lớn, nó dựa vào sự trùng hợp sắc nét của phase change (ICL từ < 0,15 lên ~0,4 nat trong 2,5–5 tỉ token), vài head 13B vừa copy vừa dịch/khớp mẫu, và tính liên tục giữa các cỡ. Chính tác giả nói đây là bằng chứng gián tiếp, có thể bị nhiễu bởi biến ẩn chung. Với dự án OCR pruning, các bài học đáng thử nhất là: đếm induction/copy head trước và sau khi cắt, coi chừng ICL score tăng giả, và dùng ablation giữ pattern để tách damage do QK khỏi damage do OV.

**Đọc bản gốc:** https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
