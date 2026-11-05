# Distributed Representations: Composition & Superposition (2023)

**Tác giả:** Christopher Olah (Anthropic). Ghi chú không chính thức, xuất bản 04/05/2023.  
**Nguồn:** https://transformer-circuits.pub/2023/superposition-composition/index.html  
**Chủ đề:** Distributed representation, local code, compositional code, dense code, sự đánh đổi giữa composition và superposition, linear probe.  
**Vị trí trong lộ trình:** Đọc sau *Toy Models of Superposition* (ghi chú 05). Ghi chú ngắn này làm rõ một nhầm lẫn khái niệm: "distributed" có hai nghĩa đối lập. Hữu ích khi bạn nghĩ về cách mô hình OCR biểu diễn ký tự tiếng Việt (chữ cái gốc + dấu).

> Tài liệu học tập bằng tiếng Việt. Đây là ghi chú diễn giải, không phải bản dịch nguyên văn. Bản gốc tự nhận là ghi chú không chính thức, "giống ý tưởng đồng nghiệp chia sẻ hơn là một paper". Những phần liên hệ tới pruning/distillation là suy luận mở rộng.

## Mục lục

1. [Luận điểm chính](#1-luận-điểm-chính)
2. [Bốn cách mã hóa hình có màu](#2-bốn-cách-mã-hóa-hình-có-màu)
3. [Composition và superposition cạnh tranh nhau](#3-composition-và-superposition-cạnh-tranh-nhau)
4. [Trộn hai chiến lược nhờ sparsity](#4-trộn-hai-chiến-lược-nhờ-sparsity)
5. [Ghi chú kiểm chứng và giới hạn](#5-ghi-chú-kiểm-chứng-và-giới-hạn)
6. [Ứng dụng giả thuyết cho pruning và distillation](#6-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
7. [Thí nghiệm gợi ý](#7-thí-nghiệm-gợi-ý)
8. [Tóm tắt khái niệm](#8-tóm-tắt-khái-niệm)

---

## 1. Luận điểm chính

"Distributed representation" (biểu diễn phân tán) thường bị coi là một phổ liên tục từ "local" đến "distributed". Olah cho rằng thực ra nó gồm **hai ý tưởng khác nhau**:

- **Composition** (kết hợp): biểu diễn một đối tượng bằng nhiều feature *độc lập* cùng bật (màu + hình).
- **Superposition** (chồng chất): biểu diễn *nhiều feature hơn số neuron*.

Hai chiến lược có tính chất rất khác nhau (khái quát hóa, cái gì đọc được tuyến tính), và **về bản chất cạnh tranh nhau**. Còn có nghĩa thứ ba, tầm thường: biểu diễn chỉ là bản *xoay* của một code không phân tán (không có privileged basis); ghi chú bỏ qua nghĩa này.

---

## 2. Bốn cách mã hóa hình có màu

Ví dụ mượn từ Thorpe (1989): mã hóa (màu, hình) bằng neuron nhị phân, 4 màu × 4 hình = 16 đối tượng.

| Code (tên của Thorpe) | Cách làm | Số neuron | Theo khung của Olah |
|---|---|---|---|
| Local | Một neuron cho mỗi cặp (màu, hình) | 16 | Monosemantic, không superposition |
| Semi-local | Neuron cho từng màu và từng hình, kết hợp lại | 8 | **Composition** |
| Highly-distributed (dense) | Mã nhị phân, như 4 bit cho 16 số | 4 | **Superposition tối đa**, không composition |
| Semi-distributed | Màu superposition với nhau, hình superposition với nhau, rồi kết hợp | (text không nêu) | Trộn cả hai |

### 2.1 Local code

Có vẻ lãng phí, nhưng **rất linh hoạt**: biểu diễn được tập tùy ý (một hình tròn đỏ *và* một hình vuông xanh), và layer sau có thể chọn tuyến tính những tổ hợp lạ (bật cho tròn đen và tam giác trắng, nhưng không cho tam giác đen hay tròn trắng). Code khác không làm được vậy.

**Bài học về linear probe** (bộ dò tuyến tính: hồi quy tuyến tính từ activation ra một thuộc tính): probe "đỏ" trên local code chắc chắn thành công, **nhưng code này không có feature "đỏ"**. Probe dò được ≠ mô hình có feature đó.

### 2.2 Compositional code

Dùng tập feature khác: màu và hình *riêng*, kết hợp lại. Đây là nghĩa "distributed" quen thuộc trong ML (ví dụ hướng "giới tính", "số nhiều" trong word embedding của Mikolov). Ưu điểm:

- **Ít neuron**: $n$ feature kết hợp thành tới $\exp(n)$ tổ hợp.
- **Khái quát phi cục bộ**: dữ liệu về hình tròn xanh lá giúp cả hình vuông xanh lá.
- **Mở rộng tự nhiên**: màu tím = trộn đỏ + xanh; vuông bo góc = thêm chút "tròn".
- **Thân thiện với interpretability**: hiểu được bằng số feature tuyến tính, dù không gian tổ hợp là mũ.

### 2.3 Dense code

Biểu diễn được nhiều đối tượng *không liên quan* hơn số neuron. Cái giá: **không chọn tuyến tính được** hầu hết màu hay hình, càng không chọn được tập; không biểu diễn được đối tượng "ở giữa" (vuông tím bo góc). Lưu ý tinh tế: code này **quay lại dùng feature cặp (màu, hình)** như local code, vì superposition tối đa chỉ khả thi khi *không* có composition.

---

## 3. Composition và superposition cạnh tranh nhau

Với $m$ bit, có "thể tích" $\exp(m)$:

$$
\boxed{\underbrace{n \text{ feature kết hợp tự do}}_{\text{cần } \exp(n) \text{ thể tích}}\Rightarrow n\le m,
\qquad
\underbrace{\text{không composition}}_{\text{superposition}}\Rightarrow \text{tới } \exp(m) \text{ đối tượng}}
$$

(Superposition cần một phi tuyến để truy xuất.)

Tức là composition và superposition là **hai cách phân bổ cùng một thể tích mũ**. Olah nói Toy Models có thể xem là đang khám phá đúng sự đánh đổi này, chỉ là diễn đạt bằng *sparsity của feature* thay vì composition.

Hệ quả: đôi khi người ta nói distributed representation cho *cả* khái quát phi cục bộ *lẫn* "nhiều feature hơn neuron". Nhưng hai lợi ích này **đánh đổi nhau**: cái đầu gắn với composition, cái sau gắn với superposition.

---

## 4. Trộn hai chiến lược nhờ sparsity

Thực tế composition thường **bị giới hạn**: chỉ vài feature bật cùng lúc, nhiều feature loại trừ nhau, tức feature **thưa**.

- Feature **dày** (tổ hợp phổ biến): mỗi neuron chỉ một feature.
- Feature **thưa** (tổ hợp nhiều feature hiếm): superposition khả thi.

Hình ảnh: $n$ feature tạo $\exp(n)$ tổ hợp, nhưng nếu composition bị giới hạn thì có những **"lỗ" xác suất thấp** trong không gian tổ hợp, và các lỗ này dùng để cất feature khác trong superposition. Đây là bản diễn đạt không chính thức của một kết quả compressed sensing: nếu chỉ một số cố định feature được cùng bật, lượng superposition có thể tăng **theo hàm mũ của số neuron**.

Code semi-distributed của Thorpe là ví dụ: unit có chút nghĩa (polysemantic nhưng chỉ vài feature), nhưng tự nhiên hơn là xem *hướng* là feature, ví dụ $A+B=\text{Xanh lá}$, $D+E=\text{Vuông}$.

---

## 5. Ghi chú kiểm chứng và giới hạn

- **Không có thí nghiệm.** Đây là lập luận khái niệm trên neuron nhị phân, kèm phác thảo cho activation liên tục (feature là hướng).
- Tác giả không khẳng định tính mới và nói có thể đã bỏ sót tài liệu thần kinh học/connectionism.
- Lập luận "thể tích $\exp(m)$" là trực giác cho trường hợp nhị phân; với activation liên tục, ranh giới được định lượng bởi compressed sensing chứ không phải đếm bit.
- Chưa có thước đo thực nghiệm để phân biệt một biểu diễn thật đang dùng composition hay superposition.

---

## 6. Ứng dụng giả thuyết cho pruning và distillation

> **Suy luận của người viết, không phải nội dung ghi chú.**

- **Ký tự tiếng Việt là ví dụ tự nhiên.** "ế" = "e" + mũ + sắc. Một biểu diễn *compositional* (chữ gốc, dấu mũ/móc/trăng, dấu thanh tách riêng) cần số feature tuyến tính và khái quát sang tổ hợp hiếm. Một biểu diễn kiểu *local/dense* (mỗi ký tự tổ hợp là một feature) khái quát kém hơn và tốn chỗ hơn. Tổ hợp dấu tiếng Việt rất phổ biến (dày), nên theo mục 4 composition là chiến lược hợp lý cho chúng.
- **Pruning có thể đẩy mô hình đổi chiến lược.** Khi giảm dung lượng, student có thể chuyển một phần sang superposition cho các ký tự/token hiếm. Hệ quả dự đoán: khả năng xử lý *tổ hợp chưa gặp* (chữ gốc + dấu hiếm, font lạ) giảm trước khả năng nhận dạng tổ hợp quen thuộc.
- **Probe không đủ.** Nếu dùng linear probe kiểm tra student "còn biết dấu thanh", probe thành công chưa chứng minh có feature "dấu sắc" dùng chung (mục 2.1). Phải kiểm tra **khái quát sang tổ hợp giữ ngoài**.
- **Hidden-state KD.** Nếu teacher dùng composition, khớp hidden state bảo toàn được cấu trúc cộng (hướng "sắc" giống nhau qua các chữ gốc). Có thể thêm loss khuyến khích tính cộng này cho student. Đây là ý tưởng, chưa có bằng chứng.

---

## 7. Thí nghiệm gợi ý

1. **Kiểm tra tính cộng của dấu thanh.** Ở residual stream của decoder (tại token ký tự) hoặc đầu ra ViT, tính hướng $\Delta_{\text{sắc}}=\text{mean}(h_{\text{á}})-\text{mean}(h_{\text{a}})$ và kiểm tra $h_{\text{e}}+\Delta_{\text{sắc}}\approx h_{\text{é}}$ (cosine với các chữ gốc khác). So teacher, student 0.3B, student 0.1B.
2. **Probe giữ ngoài tổ hợp.** Huấn luyện probe "dấu thanh" trên một số chữ gốc, test trên chữ gốc chưa thấy. Probe khái quát tốt thì gợi ý composition; chỉ tốt trên tổ hợp đã thấy thì gợi ý code cục bộ hoặc superposition.
3. **CER theo tần suất tổ hợp.** Sau mỗi mức prune, đo lỗi trên tổ hợp (chữ gốc, dấu) theo tần suất trong dữ liệu; tìm xem tổ hợp hiếm có giảm trước không.

---

## 8. Tóm tắt khái niệm

| Khái niệm | Định nghĩa | Tính chất |
|---|---|---|
| Local code | Một neuron cho mỗi đối tượng | Linh hoạt, chọn tuyến tính tổ hợp tùy ý; tốn neuron |
| Compositional code | Feature độc lập kết hợp | $n$ feature cho $\exp(n)$ tổ hợp; khái quát phi cục bộ; dễ hiểu |
| Dense code | Superposition tối đa | Tới $\exp(m)$ đối tượng; không chọn tuyến tính được; cần phi tuyến |
| Semi-distributed | Kết hợp các nhóm feature đang superposition | Trộn hai chiến lược |
| Đánh đổi thể tích | Composition và superposition chia nhau $\exp(m)$ | Khái quát và "nhiều feature" đánh đổi nhau |
| "Lỗ" do sparsity | Vùng xác suất thấp của không gian tổ hợp | Chỗ cho superposition |
| Bài học probe | Probe thành công ≠ có feature | Cần test khái quát |

### Kết luận ngắn

"Distributed representation" gộp hai chiến lược đối nghịch: composition (ít feature độc lập, khái quát tốt, dễ hiểu) và superposition (nhiều feature hơn neuron, cần phi tuyến, khó đọc tuyến tính). Chúng tranh nhau cùng một thể tích mũ, và sparsity cho phép trộn hai thứ. Hiểu biểu diễn theo nghĩa composition là điều kiện để mô tả mạng bằng số feature tuyến tính thay vì thể tích mũ.

**Đọc bản gốc:** https://transformer-circuits.pub/2023/superposition-composition/index.html
