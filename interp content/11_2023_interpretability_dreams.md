# Interpretability Dreams (2023)

**Tác giả:** Chris Olah (Anthropic). Ghi chú không chính thức (informal note), công bố ngày 24/5/2023.  
**Nguồn:** https://transformer-circuits.pub/2023/interpretability-dreams/index.html  
**Chủ đề:** Tầm nhìn dài hạn của mechanistic interpretability: nền tảng nhận thức, cấu trúc quy mô lớn, feature family, universality, nối vi mô với vĩ mô, interpretability tự động bằng AI, mục tiêu an toàn.  
**Vị trí trong lộ trình:** Đọc sau *Toy Models of Superposition* (2022) để hiểu vì sao "giải quyết superposition" được coi là nút thắt; có thể đọc trước hoặc song song với *Towards Monosemanticity* (2023). Bài này không có kết quả mới. Giá trị của nó là giải thích **vì sao** nhóm Anthropic đầu tư vào công việc "vi mô" (feature, circuit) và họ hy vọng xây gì trên đó.

> Tài liệu học tập bằng tiếng Việt. Đây là ghi chú diễn giải, không phải bản dịch nguyên văn bài viết. Những ứng dụng cho pruning/distillation là hướng mở rộng do người viết suy luận, không phải nội dung của bài.

## Mục lục

1. [Bản chất bài viết](#1-bản-chất-bài-viết)
2. [Một nền tảng nhận thức](#2-một-nền-tảng-nhận-thức)
3. [Cấu trúc quy mô lớn: feature family và tổ chức theo trọng số](#3-cấu-trúc-quy-mô-lớn-feature-family-và-tổ-chức-theo-trọng-số)
4. [Universality](#4-universality)
5. [Nối vi mô với vĩ mô](#5-nối-vi-mô-với-vĩ-mô)
6. [Interpretability tự động bằng AI](#6-interpretability-tự-động-bằng-ai)
7. [Mục tiêu cuối: an toàn, vẻ đẹp và tò mò](#7-mục-tiêu-cuối-an-toàn-vẻ-đẹp-và-tò-mò)
8. [Bài viết dựa vào bằng chứng gì](#8-bài-viết-dựa-vào-bằng-chứng-gì)
9. [Giới hạn](#9-giới-hạn)
10. [Ứng dụng giả thuyết cho pruning và distillation](#10-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
11. [Thí nghiệm gợi ý](#11-thí-nghiệm-gợi-ý)
12. [Tóm tắt khái niệm](#12-tóm-tắt-khái-niệm)

---

## 1. Bản chất bài viết

- Một **essay tầm nhìn**: hầu hết ý tưởng đã xuất hiện rải rác trong các paper trước (Circuits thread trên Distill, Branch Specialization, Multimodal Neurons, SoLU, Toy Models). Bài chủ yếu **trích lại và nối chúng**.
- Tác giả nói rõ mọi thứ ở đây **rất suy đoán**, có thể nghe "hơi vĩ đại", và nên đọc như ý tưởng một đồng nghiệp chia sẻ chứ không phải paper.
- Bối cảnh: tháng 5/2023, nhóm đang tập trung **giải superposition** (bài *Towards Monosemanticity* ra đời 4 tháng sau). Câu hỏi bài trả lời: nếu giải được, ta xây gì tiếp? Và làm sao mechanistic interpretability (vốn "vi mô") có hy vọng **scale** lên mô hình khổng lồ?

Bảy ý chính theo đề cương của bài:

| Ý | Nội dung một dòng |
|---|---|
| Epistemic foundation | Vi mô là để có nền vững, vì rất dễ tự lừa mình khi nghiên cứu mạng |
| Xây gì trên nền | Nhiều khả năng đã được minh họa sơ bộ ở InceptionV1 |
| Larger-scale structure | Có thể có cấu trúc lớn kiểu "cơ quan" hay "vùng não" |
| Universality | Feature/circuit lặp lại giữa các mô hình, nên kiến thức chuyển giao được |
| Vi mô ↔ vĩ mô | Phát hiện vi mô (induction head) có hệ quả vĩ mô |
| AI automation | Một lối thoát scale nếu mọi cách khác thất bại |
| Mục tiêu cuối | An toàn, và vẻ đẹp của cấu trúc bên trong mạng |

---

## 2. Một nền tảng nhận thức

**Câu hỏi:** Tại sao lại đi từ dưới lên (bottom-up), nghiên cứu từng mảnh nhỏ không có lý thuyết tiên nghiệm, thay vì đi thẳng vào câu hỏi ta quan tâm (top-down)?

**Lập luận:** đi top-down rất dễ tìm thấy **bằng chứng ảo** cho giả thuyết, vì luôn có nhiều tính toán *tương quan* với giả thuyết. Ví dụ của bài:

- Mô hình ImageNet có "dog head detector", đúng. Nhưng ở layer ngay trước đó, các detector mõm chó, lông chó, mắt… cộng lại đã đoán rất tốt "có đầu chó".
- Một **linear probe** (bộ phân loại tuyến tính đặt trên activation) vì vậy sẽ "tìm ra dog detector" ở layer đó, dù layer đó **chưa thật sự** có feature đầu chó.

Mechanistic interpretability cho phép **kiểm tra thật sự** một mệnh đề có đúng không. Với circuit đủ nhỏ, mệnh đề về hành vi trở thành **câu hỏi toán học**. Cái giá là phạm vi hẹp. Nhưng hy vọng là mệnh đề về hành vi toàn mô hình có thể **phân rã** thành mệnh đề về circuit.

**Ẩn dụ chính:** nền tảng giống tiên đề của toán học hay assembly của lập trình. Nhà toán học không làm việc bằng tiên đề, lập trình viên không viết assembly, nhưng nền tảng cho phép **kiểm tra** mọi tuyên bố ở tầng trừu tượng cao hơn và **dịch** giữa các tầng trừu tượng. Mục tiêu thật là xây **abstraction** trên nền đó.

- **Motif**: mẫu cấp thấp lặp lại *trong* một mạng.
- **Structural phenomenon**: thuộc tính quy mô lớn của một mạng.
- **Universality**: mẫu lặp lại *giữa* các mạng.

---

## 3. Cấu trúc quy mô lớn: feature family và tổ chức theo trọng số

Superposition mạnh ở LLM làm khó thấy cấu trúc lớn, nên ví dụ chủ yếu lấy từ mô hình thị giác (Circuits thread). Tác giả kỳ vọng điều tương tự tồn tại trong LLM nếu tách được superposition.

### 3.1 Feature family và equivariance

- **Feature family**: họ feature tương tự nhau, tham số hóa bởi một biến (hướng, tỉ lệ, màu sắc…).
- **Equivariance** (đẳng biến): không chỉ feature, mà **circuit và trọng số** nối chúng cũng được tổ chức theo cùng tham số (bài *Naturally Occurring Equivariance*).
- Họ trừu tượng hơn: *Multimodal Neurons* (CLIP) có họ neuron theo người nổi tiếng, cảm xúc, vùng địa lý.
- Trong LLM: paper SoLU thấy neuron "token X trong ngôn ngữ Y", một họ hai tham số.

**Ý quan trọng nhất của mục này** là có **meta-organization** (tổ chức cấp trên) của feature. Ta phân rã activation thành feature để thoát "bẫy hàm mũ" của không gian activation. Nếu số feature lại quá lớn, ta cần **lặp lại mẹo đó ở tầng trên**: tìm cấu trúc tổ chức các feature. Feature family là dạng đơn giản nhất và có vẻ phổ biến.

Ví dụ minh họa của người viết (không có trong bài): nếu có 10.000 feature "ký tự c ở font f", ta không cần hiểu từng cái, chỉ cần hiểu trục "ký tự", trục "font" và cách circuit tôn trọng hai trục đó.

### 3.2 Tổ chức theo cấu trúc trọng số

- *Branch Specialization*: khi kiến trúc có nhánh, neuron tương tự **tự tổ chức** vào cùng nhánh, giống "vùng não". Nhưng chỉ khi kiến trúc có nhánh, và hỏng ở layer sau (có lẽ do superposition).
- Một thí nghiệm nhỏ cuối paper đó: **không có nhánh** vẫn có cấu trúc tổ chức ẩn, thậm chí tốt hơn: một **bản đồ** neuron theo nhiều trục quan hệ, thay vì chia đôi.
- Trực giác: cấu trúc này nên "rơi ra" từ **connectome** (đồ thị kết nối) của mạng, miễn là feature liên quan nối với nhau chặt hơn feature không liên quan.
- Superposition che giấu nó, vì chiến lược superposition tốt nhất là đặt feature **xa nhau về nghĩa** vào chung một chỗ để ít giao thoa. Tách superposition thì cấu trúc lộ ra.

### 3.3 Cấu trúc khác

Ví dụ **weight banding** (vân sọc trong trọng số) là một cấu trúc quy mô lớn thú vị.

---

## 4. Universality

**Universality hypothesis**: feature và circuit tương tự hình thành ở nhiều mạng huấn luyện trên miền tương tự.

Bài trích lập luận từ *Zoom In* (2020): nếu không có universality, nghiên cứu circuit giống sinh học tế bào trong thế giới mà mỗi loài có bào quan hoàn toàn khác nhau. Khi đó ta chỉ nghiên cứu vài mô hình quan trọng và mong chúng không đổi mỗi năm. Nếu universality đúng mạnh, có thể có một kiểu **"bảng tuần hoàn của feature thị giác"**.

Bằng chứng được dẫn:

- Gabor filter, curve detector, high-low frequency detector lặp lại giữa nhiều mô hình thị giác; high-low frequency detector có **cùng circuit** giữa các mô hình.
- Nhân tạo ↔ sinh học: curve detector giống neuron trong khoa học thần kinh; gần đây tìm thấy neuron giống high-low frequency detector trong não; multimodal neuron trong CLIP tương tự kết quả "Jennifer Aniston neuron".
- Transformer: **previous token head** và **induction head** hình thành rất ổn định; neuron "all caps" xuất hiện ở nhiều mô hình SoLU nhỏ.

Hệ quả: mỗi feature universal tìm được là một **điểm tựa** cho mô hình tương lai.

---

## 5. Nối vi mô với vĩ mô

Nhiều ngành có hai cộng đồng: vĩ mô (vật lý thống kê, giải phẫu, sinh thái) và vi mô (vật lý hạt, sinh học phân tử). Dấu hiệu thành công của lý thuyết vi mô là **dự đoán được ngoại lệ** của lý thuyết vĩ mô. Ví dụ: định luật khí lý tưởng sai khi khí ngưng tụ, và lý thuyết mịn hơn phải giải thích điều đó.

**Ví dụ đã có:** induction head là hiện tượng vi mô gây **cái bướu** (bump) trên loss curve, và nhiều khả năng giải thích cái bướu trong paper scaling laws gốc (đoạn chuyển từ mạng 1 layer sang 2 layer).

**Suy đoán táo bạo của tác giả:**

- Có nhiều circuit universal, mỗi cái giảm loss một lượng. Từ góc nhìn động học học tập, mỗi cái tạo một "thung lũng" trong không gian loss: *có lẽ mỗi thung lũng trong loss landscape là cái bóng của một circuit*.
- Khi mô hình lớn hơn, nó có thêm circuit, tạo ra scaling laws.
- Phức tạp hơn: circuit **phụ thuộc** nhau (cần circuit đơn giản làm bậc thang). Nếu hiểu **đồ thị phụ thuộc** đó (circuit nào cần circuit nào, chiếm bao nhiêu capacity, giảm loss bao nhiêu), có thể giải thích cả động học học tập lẫn scaling laws bằng **một lý thuyết cơ chế**.

Bài cũng lưu ý: lý thuyết vĩ mô không cần vi mô mới tồn tại (di truyền học có trước DNA), nhưng vi mô tốt giúp tin vĩ mô **trong miền hợp lệ** và chỉ ra chỗ nó hỏng.

---

## 6. Interpretability tự động bằng AI

- Một con đường scale hoàn toàn khác: dùng AI để tự động hóa interpretability (proof of concept: Bills et al.). Các phương pháp tác giả biết **vẫn cần giải superposition**, nhưng giảm được các khó khăn khác.
- Ưu điểm: AI đang tăng theo hàm mũ, nên "bám" vào cùng xu hướng có thể là cách duy nhất để theo kịp.
- Hai lý do muốn có một phiên bản **do con người hiểu**:
  1. **Thẩm mỹ**: giống chứng minh định lý bốn màu bằng máy, đúng nhưng không ai hiểu.
  2. **An toàn**: dùng AI để kiểm toán an toàn AI thì có nên tin AI kiểm toán không? Nhiều cách lách vấn đề này sẽ làm **failure mode** của interpretability **tương quan** với failure mode của các hướng an toàn khác. Như vậy mất giá trị "lớp phòng thủ độc lập".
- Trung gian: **AI-assisted interpretability**, AI tăng tốc nhưng phần quan trọng vẫn được con người kiểm chứng.

---

## 7. Mục tiêu cuối: an toàn, vẻ đẹp và tò mò

### 7.1 "Ngôi sao Bắc Đẩu" cho an toàn

Ta muốn tuyên bố kiểu "mô hình **không bao giờ** nói dối" hoặc "**không bao giờ** thao túng ta". Diễn đạt hình thức (bài trình bày dưới dạng hình; người viết diễn đạt lại):

$$
\forall\,x\in\mathcal X:\ \neg\,\text{Deceive}(M, x)
$$

Đây **không** phải loại mệnh đề ML thường xử lý. ML quan tâm **kỳ vọng trên một phân phối**, còn an toàn quan tâm **out-of-distribution** và **worst case**. Điều duy nhất làm nó dễ hơn chút: có lẽ ta chỉ quan tâm thất bại "có chủ đích" (nhưng chưa rõ làm sao nắm bắt khái niệm này).

Hy vọng (rất suy đoán): nếu giải được **cả superposition lẫn scalability**, một **tập feature đầy đủ** cho ta thứ để **liệt kê** (enumerate), tức là chạm tới "mọi hành vi mô hình" theo một nghĩa nào đó. Phác thảo chi tiết hơn nằm ở mục "Strategic Picture" của *Toy Models*.

### 7.2 Vẻ đẹp

- Interpretability là lĩnh vực **tiền mô thức** (pre-paradigmatic). Tác giả tin nó được lợi khi nhà nghiên cứu theo **gu cá nhân và thẩm mỹ**, hơn là suy ngược cứng nhắc từ mục tiêu an toàn: "cần bản đồ tốt hơn trước đã".
- Phản bác "Bitter Lesson" (thành công của DL nhàm chán vì chỉ cần scale): vẻ đẹp của DL là **vẻ đẹp sinh học**. Tiến hóa đơn giản sinh ra thế giới tự nhiên phức tạp; gradient descent đơn giản sinh ra cấu trúc đẹp trong mạng nơ-ron.

---

## 8. Bài viết dựa vào bằng chứng gì

Bài **không** có thí nghiệm mới. Bằng chứng là trích dẫn:

| Loại | Nguồn được dẫn |
|---|---|
| Feature family, equivariance | *Naturally Occurring Equivariance*, *Multimodal Neurons*, SoLU (Section 6.3.5) |
| Tổ chức theo trọng số | *Branch Specialization* |
| Universality | Circuits thread (Gabor, curve, high-low frequency), induction head, neuron all-caps trong SoLU |
| Vi mô → vĩ mô | Induction head và bump trên loss curve; Figure 13 của paper scaling laws gốc |
| Tự động hóa | Bills et al. (2023) |

Vì vậy cần đọc bài như **chương trình nghiên cứu**, không như kết quả.

---

## 9. Giới hạn

- **Tự nhận là suy đoán**; có thể sai hoàn toàn; "mistakes are very possible".
- Hầu hết bằng chứng cấu trúc lớn đến từ **mô hình thị giác CNN**; việc chuyển sang LLM là giả định, phụ thuộc vào việc giải superposition.
- Mục tiêu an toàn (mệnh đề "với mọi input") rất xa so với công cụ hiện có; "liệt kê tập feature đầy đủ" giả định có tập hữu hạn, rời rạc (sau này *Scaling Monosemanticity* cho thấy feature có đuôi dài theo tần suất, và lấy hết có thể rất tốn).
- Phê phán AI automation chủ yếu là lập luận, chưa định lượng.
- Không nói về attention superposition hay kiến trúc mới.

---

## 10. Ứng dụng giả thuyết cho pruning và distillation

> **Phần này là suy luận của người viết.** Bài có liên quan gián tiếp đến nén mô hình; giá trị chủ yếu là khung tư duy.

- **Probe đánh lừa ↔ metric layer importance đánh lừa.** Ví dụ "dog detector" khớp với cảnh báo khi đánh giá điểm cắt bằng probe hay CKA: layer có thể "trông như" chứa thông tin cần thiết vì tương quan, dù tính toán thật nằm ở chỗ khác. Đây là lý do nên đo damage bằng **can thiệp** (cắt rồi đo), như bạn đang làm với $D_{lin}/D_{res}$.
- **Universality ↔ teacher–student.** Nếu feature universal giữa các seed, có thể chúng cũng tồn tại ở student đã nén. Câu hỏi đo được: feature nào của teacher có "bản sao" ở student (theo tương quan activation)? Feature không có bản sao là ứng viên cho phần $D_{res}$.
- **Feature family ↔ cấu trúc cắt.** Nếu feature (ví dụ ký tự × font, hoặc dấu × nguyên âm) tổ chức theo họ, có thể một student nhỏ giữ được **trục** của họ nhưng mất độ phân giải. Distillation có thể nhắm vào việc giữ trục.
- **"Mỗi thung lũng loss là bóng của một circuit" ↔ đường cong phục hồi.** Khi KD phục hồi student, các bước nhảy đột ngột trong loss (nếu có) có thể ứng với việc một circuit được tái lập (ví dụ circuit copy, giống induction). Đây là giả thuyết kiểm tra được.
- **Mức độ liên quan:** thấp đến trung bình. Bài không cho công cụ cụ thể.

---

## 11. Thí nghiệm gợi ý

**Thí nghiệm 1: Probe so với can thiệp tại điểm cắt.** Với một thuộc tính OCR (ví dụ "token hiện tại là chữ số"), huấn luyện linear probe ở mọi layer. Sau đó cắt từng layer và đo damage trên đúng thuộc tính đó. So sánh: layer có probe accuracy cao có phải là layer mà cắt đi gây damage lớn không? Kỳ vọng (theo lập luận của bài): không nhất thiết.

**Thí nghiệm 2: Tìm "bump" trong quá trình KD.** Ghi loss theo từng loại token (chữ thường, chữ có dấu, chữ số, token lặp/copy) trong quá trình KD student. Tìm các bước nhảy và kiểm tra xem tại đó có head hay feature mới xuất hiện không (ví dụ induction score tăng vọt).

---

## 12. Tóm tắt khái niệm

| Khái niệm | Định nghĩa ngắn | Vai trò trong tầm nhìn |
|---|---|---|
| Epistemic foundation | Mệnh đề về circuit kiểm chứng được gần như toán học | Nền để kiểm tra abstraction cao hơn |
| Motif | Mẫu lặp lại trong một mạng | Abstraction đầu tiên trên circuit |
| Structural phenomenon | Thuộc tính quy mô lớn của mạng | "Cơ quan, vùng não" |
| Feature family | Họ feature tham số hóa theo biến | Thoát bẫy số lượng feature |
| Equivariance | Circuit và trọng số tôn trọng cùng tham số | Nén mô tả circuit |
| Branch specialization | Feature tự tổ chức theo nhánh | Bằng chứng tổ chức tự phát |
| Universality | Feature/circuit lặp lại giữa mạng | Kiến thức chuyển giao |
| Vi mô ↔ vĩ mô | Circuit giải thích loss bump, scaling | Nối với ML rộng hơn |
| AI-automated interp | AI giải thích AI | Lối thoát scale, có rủi ro tin cậy |
| Mệnh đề an toàn | "Với mọi input, không làm X" | Ngôi sao Bắc Đẩu, cần feature đầy đủ |

### Kết luận ngắn

*Interpretability Dreams* giải thích lý do tồn tại của công việc vi mô. Ta cần một nền tảng không tự lừa mình; trên nền đó có thể xây abstraction (feature family, cấu trúc tổ chức), tận dụng universality, nối với hiện tượng vĩ mô như scaling laws, và cuối cùng (rất xa) là các mệnh đề an toàn kiểu "với mọi input". Nút thắt chung của mọi giấc mơ này là **superposition**, đúng thứ mà *Towards Monosemanticity* bắt tay giải ngay sau đó. Với người làm nén mô hình, bài học thực dụng nhất là: **tương quan (probe, similarity) không phải cơ chế**; hãy đo bằng can thiệp.

**Đọc bản gốc:** https://transformer-circuits.pub/2023/interpretability-dreams/index.html
