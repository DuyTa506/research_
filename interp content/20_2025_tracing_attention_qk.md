# Tracing Attention Computation Through Feature Interactions (2025)

**Tác giả:** Harish Kamath, Emmanuel Ameisen, Isaac Kauvar, Rodrigo Luger, Wes Gurnee, Adam Pearce, Sam Zimmerman, Joshua Batson, Thomas Conerly, Chris Olah, Jack Lindsey (Anthropic). Công bố ngày 31/07/2025.  
**Nguồn:** https://transformer-circuits.pub/2025/attention-qk/index.html  
**Chủ đề:** QK attribution, head loading, "checkpointing" đường attention bằng feature, mạch induction, antonym, trắc nghiệm, concordance/discordance head.  
**Vị trí trong lộ trình:** Đọc sau *A Mathematical Framework* (2021, định nghĩa $W_{QK}$, $W_{OV}$) và *On the Biology of a Large Language Model* (2025), nơi tác giả thừa nhận điểm mù lớn nhất là "không giải thích được vì sao attention nhìn vào đâu". Bài này lấp một phần điểm mù đó.

> Tài liệu học tập bằng tiếng Việt. Đây là ghi chú diễn giải, không phải bản dịch nguyên văn. Những ứng dụng cho pruning/distillation là hướng mở rộng do chúng tôi suy luận, không phải đóng góp của paper.

## Mục lục

1. [Vấn đề: attribution graph bỏ qua attention](#1-vấn-đề-attribution-graph-bỏ-qua-attention)
2. [QK attribution](#2-qk-attribution)
3. [Head loading và checkpointing](#3-head-loading-và-checkpointing)
4. [Case study: induction](#4-case-study-induction)
5. [Case study: antonym và trắc nghiệm](#5-case-study-antonym-và-trắc-nghiệm)
6. [Concordance và discordance head](#6-concordance-và-discordance-head)
7. [Cách paper kiểm chứng](#7-cách-paper-kiểm-chứng)
8. [Giới hạn và câu hỏi mở](#8-giới-hạn-và-câu-hỏi-mở)
9. [Ứng dụng giả thuyết cho pruning và distillation](#9-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
10. [Thí nghiệm gợi ý](#10-thí-nghiệm-gợi-ý)
11. [Tóm tắt công thức và khái niệm](#11-tóm-tắt-công-thức-và-khái-niệm)

---

## 1. Vấn đề: attribution graph bỏ qua attention

Transcoder chỉ đọc/ghi trong cùng một vị trí token; mọi tương tác *giữa các vị trí* phải đi qua attention. Để attribution tuyến tính, attribution graph **đóng băng attention pattern** (coi như hằng số). Hệ quả:

1. Không rõ head nào (hay chuỗi head nào) mang một cạnh, vì attribution là tổng qua mọi đường head.
2. Dù biết head, vẫn không biết **vì sao** head nhìn vào vị trí đó: gradient khi đã cố định pattern bỏ qua hoàn toàn QK circuit.

Ví dụ thất bại điển hình: với induction `Aunt Sally ... Aunt` → `Sally`, đồ thị cũ chỉ có một cạnh "Sally (token cũ) → logit Sally". Nghĩa là "mô hình nói Sally vì nó nhìn vào Sally". Đúng nhưng vô dụng.

**Hai thành phần của lời giải:**
- **QK attribution**: phân rã attention score thành tổng các tương tác feature (query) × feature (key).
- **Head loading**: với mỗi cạnh trong đồ thị, mỗi head đóng góp bao nhiêu.

---

## 2. QK attribution

### 2.1 Ý tưởng

Attention score trước softmax là hàm **song tuyến tính** (bilinear) của residual stream tại vị trí query và key. Dùng $W_{QK}=W_Q^\top W_K$ như trong Framework (bài viết không ghi hệ số $1/\sqrt{d}$):

$$
s(p_q,p_k)=x_{p_q}^\top W_{QK}\,x_{p_k}.
$$

Nếu phân rã residual thành feature (bài giả định có SAE trên residual stream của mỗi layer), cộng bias và sai số:

$$
x_{p_q}=\sum_i a_i^{q} f_i + b^{q} + \varepsilon^{q},\qquad
x_{p_k}=\sum_j a_j^{k} f_j + b^{k} + \varepsilon^{k}.
$$

Nhân ra (công thức gốc trong bài là hình ảnh; đây là dạng chúng tôi viết lại theo mô tả bằng lời):

$$
\boxed{\;s(p_q,p_k)=\sum_{i,j} a_i^{q}a_j^{k}\,\underbrace{f_i^\top W_{QK} f_j}_{\text{tương tác feature–feature}}
+\sum_i a_i^{q} f_i^\top W_{QK}(b^{k}+\varepsilon^{k})
+\sum_j a_j^{k}(b^{q}+\varepsilon^{q})^\top W_{QK} f_j
+(b^q+\varepsilon^q)^\top W_{QK}(b^k+\varepsilon^k)\;}
$$

Tổng các số hạng bằng đúng attention score. Sau đó chỉ cần **sắp xếp theo độ lớn** và hiển thị từng cặp (feature query, feature key) cạnh nhau trên giao diện.

### 2.2 Ví dụ số nhỏ (minh họa của chúng tôi)

Giả sử ở query có feature $q_1$ = "Aunt" ($a=2$), ở key có $k_1$ = "tên người" ($a=1$) và $k_2$ = "tên của một Aunt/Uncle" ($a=1.5$), với $f_{q_1}^\top W_{QK} f_{k_1}=0.8$, $f_{q_1}^\top W_{QK} f_{k_2}=1.2$:

| Cặp | Đóng góp |
|---|---|
| Aunt × tên người | $2\cdot1\cdot0.8=1.6$ |
| Aunt × tên của Aunt | $2\cdot1.5\cdot1.2=3.6$ |

Đọc bảng: head nhìn vào token này chủ yếu vì nó là "tên của một Aunt", một phần vì nó là "tên người" nói chung.

### 2.3 Lưu ý kiến trúc

- **RoPE** làm phép biến đổi phụ thuộc vị trí, nên score chịu ảnh hưởng của thông tin vị trí không có trong residual. Tác giả nói ý tưởng vẫn mở rộng được cho các kiến trúc attention phổ biến.
- Nếu có **normalization** giữa residual và $W_Q, W_K$, cần tuyến tính hóa nó trước; bias của norm được gộp vào số hạng bias.
- **Giới hạn quan trọng:** QK attribution giải thích *score*, không phải *pattern*. Softmax tạo cạnh tranh giữa các vị trí; muốn biết vì sao nhìn vào A cần biết vì sao *không* nhìn vào B. Dữ liệu score âm ở vị trí khác có sẵn, nhưng chưa có cách tự động làm nổi các hiệu ứng ức chế quan trọng.

---

## 3. Head loading và checkpointing

### 3.1 Bùng nổ số đường

Với transcoder cách nhau $L$ layer, số đường qua chuỗi head tăng **theo hàm mũ** theo $L$ (CLT không giải quyết được điều này). Không thể phân rã cạnh theo từng đường.

**Lời giải: "checkpoint"**, ép mỗi cạnh chỉ đi qua đường attention độ dài 1. Ba phương án đã thử:

| Phương án | Mô tả | Nhận xét của tác giả |
|---|---|---|
| Multi-Token Transcoder (MTC) | Thay attention bằng mô hình kiểu transcoder; feature được mang bởi tổ hợp tuyến tính các head | Tích lũy sai số qua layer |
| SAE trên output attention | Thêm feature attention-out vào đồ thị cạnh transcoder | Tích lũy sai số qua layer |
| SAE trên residual stream mỗi layer | Attribution gradient giữa feature ở layer kề nhau | **Được chọn** |

Thực tế họ dùng **weakly causal crosscoder (WCC)**: feature đọc residual ở layer $L$ và tái tạo residual ở $L, L+1,\dots$ cho phép "nhảy ngược" qua layer, tránh chuỗi feature trùng lặp. Đánh đổi: attribution qua MLP không còn tuyến tính như transcoder, có nguy cơ khó hiểu hoặc quá "cục bộ" theo prompt.

Lưu ý: một cạnh vẫn có thể đi qua **nhiều head trong cùng layer**, nhưng không còn qua *chuỗi* head nhiều layer.

### 3.2 Công thức head loading

Cạnh giữa hai feature SAE ở layer kề nhau = thành phần qua attention + qua MLP + qua residual. Với feature nguồn ở $p_s$ (activation $a_s$, vector $\mathbf v_s$) và đích ở $p_t$ ($a_t$, $\mathbf v_t$), thành phần attention là (viết đúng như trong bài):

$$
\boxed{\;\sum_{h\in\text{heads}} a_s a_t\left(\mathbf v_t^\top O_h V_h \mathbf v_s\right)\cdot \mathrm{attention}_h(p_s,p_t)\;}
$$

Tổng chạy qua các head ở layer của feature nguồn. **Mỗi số hạng là head loading** của head $h$ cho cạnh đó. Vector feature là decoder của SAE; encoder SAE chỉ dùng để suy ra activation, không được coi là trọng số của "replacement model".

**Quy trình sử dụng:** cạnh vượt vị trí → head loading chỉ ra $(\text{query}, \text{key}, \text{layer}, \text{head})$ quan trọng → mở QK attribution của head đó để xem *vì sao* nó nhìn vào đó. OV ($O_hV_h$) nói *cái gì* được chuyển; QK nói *vì sao* lại là vị trí này.

---

## 4. Case study: induction

Prompt: `I always loved visiting Aunt Sally. Whenever I was feeling sad, Aunt` → `Sally` (Claude 3.5 Haiku).

**Bước A: trên "Aunt" thứ hai, "Sally" → "say Sally".** Cạnh do một nhóm nhỏ head mang. QK attribution:
- Query: feature "Aunt" / "từ chỉ quan hệ gia đình".
- Key: (1) feature tên người (chung hoặc "Sally"), (2) feature "đây là tên của một Aunt/Uncle" (bật trên tên đứng sau "Aunt"/"Uncle").

Vậy QK kết hợp hai heuristic: tìm *bất kỳ* tên nào, và tìm *tên của Aunt/Uncle*.

**Can thiệp:** chọn các head có loading cao (khoảng 3–10% số head), scale feature "Name of Aunt/Uncle" *chỉ trong QK circuit* của các head đó. Giảm về âm: mô hình mất khả năng induction, dự đoán tên Aunt chung chung.

**Hiệu ứng hydra:** cần steer nhiều head vì khi một head ngừng nhìn, head ở layer sau bù lại. Nếu đóng băng attention của các head không steer, cần chưa tới một nửa số head để đạt cùng hiệu ứng.

**Bước B: trên token "Sally" đầu, "Aunt" → "token trước là Aunt".** Hầu hết cạnh giữa hai token này cùng làm việc đó và do cùng nhóm head mang. QK: query "first name" × key "danh xưng trước tên" (Aunt, Mr., ...). Đây là **previous-token head**, tiền đề của induction, khớp với cơ chế K-composition trong Framework.

**Nhiều tương tác QK song song:** với `Aunt Mary Sue ... Aunt Mary` → `Sue`, có "Mary" (query) × "token sau Mary" (key) và *độc lập* "Name of Aunt/Uncle" × "Name of Aunt/Uncle"; cross term yếu. Ma trận QK attribution có hạng ít nhất 2. Thêm: bias phía query × feature tên phía key, nghĩa là head có xu hướng chung là nhìn vào tên.

Tác giả để ngỏ vai trò của thông tin vị trí (nếu có nhiều tên có danh xưng, phải chọn tên gần nhất).

---

## 5. Case study: antonym và trắc nghiệm

### 5.1 Antonym

`Le contraire de "petit" est "` → `grand`. Biology đã giả thuyết có "opposite head" nhưng không xác nhận được. Kết quả:
- Cạnh small → large do một nhóm head hạn chế mang.
- Key: feature trên từ cần tìm đối nghĩa ("X" trong "opposite of X"), và feature tính từ chung.
- Query: feature ngữ cảnh nói về đối nghĩa.

Hai cơ chế trộn: đánh dấu rõ "petit" là từ cần lấy đối nghĩa; và tìm bất kỳ tính từ nào có thể lấy đối nghĩa. Inhibit "opposite" phía query: giảm mạnh "grand", mô hình chuyển sang từ đồng nghĩa ("peu", "faible") hoặc chính "petit". Inhibit "adjective" phía key: hiệu ứng tương tự nhưng yếu hơn.

### 5.2 Trắc nghiệm

`In what year did World War II end? (A) 1776 (B) 1945 (C) 1865 ... Answer: (` → `B`. QK attribution của các head mang cạnh:
- Query: feature "cần nói ra đáp án" (chữ "is" trong "The correct answer is", thẻ `<mc>`...) và **bias**.
- Key: feature "correct answer" trên token đáp án đúng; feature "false statement" trên đáp án sai (tương tác âm, ức chế score); feature chung cho các lựa chọn.

Head có xu hướng nền (qua bias) luôn nhìn vào đáp án đúng, mạnh hơn khi ngữ cảnh yêu cầu trả lời trắc nghiệm.

**Kiểm chứng:** inhibit "correct answer" trên token B → giảm "B"; bơm "correct answer" vào token của A hoặc C (đồng thời giảm ở B) → output đổi sang A hoặc C. Nguồn gốc feature "correct answer" (từ "1945 trong ngữ cảnh kết thúc Thế chiến II" + "kết thúc Thế chiến II" ở câu hỏi) bị error node che, chưa hiểu sâu hơn.

---

## 6. Concordance và discordance head

`Answer with yes or no. The color of a banana is yellow?` → `Yes`; đổi thành red → `No`.

**Giả thuyết ban đầu (bị bác):** mô hình tính "chuối → vàng" rồi so với đáp án nêu ra. **Thực tế:** có các head trực tiếp kiểm tra *sự khớp*.

- Truy ngược từ Yes/No: "correct answer" (yellow) hoặc "false statements about equalities" (red), nhận đầu vào từ "plausible properties of things" hoặc "discordant statements".
- Đầu vào feature trên token "banana" gần giống nhau ở hai prompt, không giải thích được sự khác biệt. Khác biệt nằm ở **head mang cạnh**: **concordance head** ở ca yellow, **discordance head** ở ca red.
- QK: màu (query) × banana (key). Banana–yellow đóng góp dương cho concordance, âm cho discordance; banana–red ngược lại.

**Can thiệp:** steer query (thêm "red", bớt "yellow") làm giảm attention của concordance head, tăng của discordance head, và lật Yes → No ở mức steer vừa phải. Chỉ output scale của *một* head cũng đủ lật; nhiều head mạnh hơn. Cần steer mạnh hơn để lật No → Yes, nên tác giả gợi ý "phải nhiều thứ đúng" thì câu mới được coi là đúng (giống người nhìn 24×4 = 1023331 biết sai ngay nhờ ước lượng cỡ và tính chẵn lẻ).

**Tổng quát:** cùng các head đó phân biệt `18+5=23?` (Yes) với `18+6=23?` (No). QK: "các số cách nhau 5" (query) × "5"/"6" (key). Không thấy bằng chứng mô hình tính 23 rồi so sánh. Nó kiểm tra một *tính chất* của tổng. Điều này khớp với các nghiên cứu khác cho thấy cơ chế *kiểm tra* và *tính* đáp án là tách biệt.

**Chẩn đoán chỉ từ trọng số:** tương tác feature qua $W_{QK}$ lệch dương ở concordance head, lệch âm ở discordance head; và **trị riêng của $W_{QK}$** cũng lệch dương/âm tương ứng, kiểm tra được mà không cần feature basis.

$$
\text{concordance: } \operatorname{spec}(W_{QK})\ \text{lệch dương},\qquad
\text{discordance: } \operatorname{spec}(W_{QK})\ \text{lệch âm}.
$$

---

## 7. Cách paper kiểm chứng

| Bằng chứng | Ví dụ |
|---|---|
| Head loading xác định head mang cạnh | Induction, antonym, trắc nghiệm, banana |
| QK attribution xếp hạng cặp feature | Aunt × Name-of-Aunt |
| Steering *chỉ trong QK* của head chọn lọc | Bỏ "Name of Aunt/Uncle" → mất induction |
| Bơm feature ở vị trí key khác | "correct answer" sang A/C → đổi đáp án |
| Steering query → đổi attention và output | red/yellow lật Yes/No |
| Kiểm tra tổng quát trên tác vụ khác | Concordance head ở bài cộng |
| Đặc trưng chỉ từ trọng số | Dấu trị riêng $W_{QK}$ |

Không có số liệu tổng hợp trên tập lớn; đây là case study đầu tiên, tác giả nói rõ vậy.

---

## 8. Giới hạn và câu hỏi mở

- Giải thích **score**, chưa giải thích **pattern** sau softmax; chưa tự động tìm hiệu ứng ức chế quan trọng (quy về bài toán tìm "phản thực tế liên quan").
- Dùng SAE/WCC trên residual: attribution qua MLP **không tuyến tính**, có thể quá cục bộ.
- Một cạnh vẫn có thể do nhiều head cùng layer mang; có thể có **attention head superposition**.
- Chi phí tăng bậc hai theo độ dài context; cần pruning động.
- Head được đặt nhãn theo vai trò *trong prompt đang xét*; chưa biết head có polysemantic hay không.
- Chưa xét đầy đủ vai trò của thông tin vị trí (RoPE).
- Gợi ý của tác giả: ma trận QK attribution thường **gần low-rank**, có thể thay danh sách dài bằng ít "feature component".

---

## 9. Ứng dụng giả thuyết cho pruning và distillation

> **Suy luận của chúng tôi, không phải của paper.**

**9.1 OCR là bài toán "chọn đúng chỗ để copy".** Theo Biology, khi output là bản copy, đồ thị không nói gì. QK attribution chính là công cụ để hỏi: token text đang sinh "khớp" với visual token nào, và vì feature gì (vị trí dòng? hình dạng ký tự? ngữ cảnh từ?). Khi student 0.1B đọc sai thứ tự dòng hay nhảy cột, nguyên nhân có thể nằm ở QK chứ không ở OV. Phân rã D_lin/D_res ở residual có thể không thấy lỗi này, vì residual vẫn "giống" nhưng pattern đã đổi.

**9.2 Hiệu ứng hydra và cắt layer.** Bài cho thấy head ở layer sau bù khi head trước ngừng nhìn. Đây là một cơ chế cụ thể giải thích vì sao **cắt layer dễ phục hồi**: phần chức năng attention bị mất có thể được head khác tái tạo (sau finetune/KD), và sai lệch ở dạng "dịch chuyển pattern" có thể khá tuyến tính trên residual (thuộc D_lin). Cần đo, không suy ra được.

**9.3 Chẩn đoán head bằng phổ $W_{QK}$.** Dấu trị riêng là đại lượng chỉ cần trọng số, rẻ. Có thể dùng để phân loại head (khớp/kiểm tra vs. không khớp) trước khi quyết định cắt, ví dụ tránh cắt hết các head một loại. Lưu ý $W_{QK}$ không đối xứng; trị riêng có thể phức, nên xét phần thực hoặc phần đối xứng $\tfrac12(W_{QK}+W_{QK}^\top)$.

**9.4 GatedDeltaNet.** Linear attention không có softmax, nhưng output vẫn là tổ hợp $\sum_j (\text{decay}_{t,j})\,(q_t^\top k_j)\,v_j$ (bỏ qua phần delta-rule hiệu chỉnh state). Số hạng $q_t^\top k_j$ vẫn song tuyến tính trong residual, nên ý tưởng QK attribution có thể mở rộng với thêm hệ số decay/gate; delta rule làm state phụ thuộc phi tuyến vào lịch sử nên phép phân rã chỉ gần đúng. Ưu điểm: không có cạnh tranh softmax, nên "score" gần với "pattern" hơn. Đây là hướng nghiên cứu mở.

**9.5 Distillation theo QK.** Thay vì ép attention map của student khớp teacher (tốn kém, và bị cạnh tranh softmax làm nhiễu), có thể ép **các tương tác QK quan trọng** khớp: chiếu residual student lên vài feature/hướng then chốt của teacher và so sánh score song tuyến tính trên các cặp đó.

---

## 10. Thí nghiệm gợi ý

**Thí nghiệm 1 — Head loading cho token OCR.** Với các layer full-attention của decoder LightOnOCR, lấy một token output, tính đóng góp của từng head vào residual tại token đó từ các vị trí visual token: $\sum_k A_h(t,k)\,O_hV_h x_k$, chiếu lên hướng unembedding của token đúng. So sánh phân bố head-loading teacher vs. student. Head nào "biến mất" ở student mà CER tăng?

**Thí nghiệm 2 — Đo hiệu ứng hydra.** Ablate một head (thay output bằng trung bình) ở teacher theo hai chế độ: (a) để các head sau tự do, (b) đóng băng attention pattern của mọi head khác. Hiệu số loss giữa (b) và (a) đo mức bù. Nếu mức bù lớn ở layer giữa, đó là bằng chứng ủng hộ cắt layer giữa.

**Thí nghiệm 3 — Phổ $W_{QK}$.** Tính phần thực các trị riêng của $W_Q^\top W_K$ (đã gộp RoPE ở vị trí tương đối 0, hoặc bỏ qua RoPE như xấp xỉ) cho mọi head full-attention. Vẽ phân bố độ lệch dấu; đối chiếu với tầm quan trọng của head khi ablate. Câu hỏi: head lệch âm có nhạy hơn khi pruning không?

---

## 11. Tóm tắt công thức và khái niệm

| Khái niệm | Công thức / mô tả | Ý nghĩa |
|---|---|---|
| $W_{QK}$ | $W_Q^\top W_K$ | Dạng song tuyến tính của score |
| QK attribution | $s=\sum_{i,j}a_i^qa_j^k f_i^\top W_{QK}f_j+$ bias + error | Vì sao head nhìn vào vị trí này |
| Head loading | $a_sa_t(\mathbf v_t^\top O_hV_h\mathbf v_s)\,\mathrm{attn}_h(p_s,p_t)$ | Head nào mang cạnh |
| Checkpointing | SAE/WCC ở mỗi layer residual | Chỉ còn đường attention độ dài 1 |
| Hydra effect | Head sau bù cho head bị chặn | Cần steer nhiều head |
| Previous-token head | first name × danh xưng | Tiền đề induction |
| Concordance / discordance head | Phổ $W_{QK}$ lệch dương / âm | Kiểm tra khớp, không tính đáp án |
| Giới hạn chính | Score ≠ pattern (softmax) | Cần phản thực tế để giải thích ức chế |

### Kết luận ngắn

Bài thêm "nửa còn thiếu" vào attribution graph: head nào mang cạnh, và cặp feature query–key nào khiến head nhìn vào đó. Các case study xác nhận cơ chế induction, antonym, trắc nghiệm mà Biology chỉ đoán được, và phát hiện concordance/discordance head. Với OCR, nơi chọn đúng vị trí để copy là cốt lõi, đây có lẽ là công cụ interp liên quan nhất tới nguyên nhân student đọc sai thứ tự hoặc bỏ dòng.

**Đọc bản gốc:** https://transformer-circuits.pub/2025/attention-qk/index.html
