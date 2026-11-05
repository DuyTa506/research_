# Using Dictionary Learning Features as Classifiers (2024)

**Tác giả:** Trenton Bricken, Jonathan Marcus, Siddharth Mishra-Sharma, Meg Tong, Ethan Perez, Mrinank Sharma, Kelley Rivoire, Thomas Henighan; biên tập: Adam Jermyn (Anthropic). Văn bản nguồn không ghi ngày công bố; bài dẫn tới *September 2024 Update*, nên ra sau thời điểm đó.  
**Nguồn:** https://transformer-circuits.pub/2024/features-as-classifiers/index.html  
**Chủ đề:** Dùng feature SAE làm input cho bộ phân loại (prompt nguy hại về vũ khí sinh học), so với raw activation; tìm spurious correlation; tấn công đối kháng; decision tree; SAE residual.  
**Vị trí trong lộ trình:** Đọc sau *Scaling Monosemanticity* (2024). Đây là một trong những thử nghiệm đầu tiên kiểm tra một **lợi ích ứng dụng** của feature, thay vì chỉ chứng minh feature tồn tại. Đây là báo cáo sơ bộ ("như chia sẻ ở lab meeting"), không phải paper hoàn chỉnh.

> Tài liệu học tập bằng tiếng Việt. Đây là ghi chú diễn giải, không phải bản dịch nguyên văn. Những ứng dụng cho pruning/distillation là hướng mở rộng do người viết suy luận, không phải kết quả của bài.

## Mục lục

1. [Câu hỏi và thiết lập](#1-câu-hỏi-và-thiết-lập)
2. [Kết quả: feature so với raw activation](#2-kết-quả-feature-so-với-raw-activation)
3. [Tìm spurious correlation và tấn công đối kháng](#3-tìm-spurious-correlation-và-tấn-công-đối-kháng)
4. [OOD, decision tree và SAE residual](#4-ood-decision-tree-và-sae-residual)
5. [Cách bài kiểm chứng và giới hạn](#5-cách-bài-kiểm-chứng-và-giới-hạn)
6. [Ứng dụng giả thuyết cho pruning và distillation](#6-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
7. [Thí nghiệm gợi ý và tóm tắt](#7-thí-nghiệm-gợi-ý-và-tóm-tắt)

---

## 1. Câu hỏi và thiết lập

**Câu hỏi:** Huấn luyện classifier trên **feature activation** (output encoder SAE) thay vì **raw activation** (residual stream gốc) có lợi gì không? Ứng dụng: phát hiện prompt về vũ khí sinh học (harmful/harmless). Động cơ: failure mode của classifier có thể **ít tương quan** với failure mode của finetuning hay system prompt, nên có thể "xếp chồng" lên chúng.

Thiết lập:

- Activation lấy ở **residual stream layer giữa**, cho cả hai loại input.
- Mô hình: một transformer 18 layer (từ bài *Investigating Successor Heads*) và **Claude 3.0 Sonnet** với SAE 1M feature. SAE được **oversample dữ liệu sinh học** (synthetic bio).
- Classifier: logistic regression, **L1** cho feature (thưa, dễ đọc), **L2** cho raw activation; chọn hệ số theo `synthetic_1`.
- Gộp theo context: **last token** (activation token cuối) hoặc **max-pooling** (giá trị lớn nhất của mỗi feature trên toàn context).

Tập đánh giá:

| Tập | Mô tả |
|---|---|
| `synthetic_1` | Validation, cùng phân phối với train |
| `synthetic_2` | Dữ liệu tổng hợp bởi mô hình khác, hơi OOD |
| `synthetic_3` | `synthetic_2` dịch sang nhiều ngôn ngữ và encoding (Hàn, Hebrew, base64), rất OOD |
| `human` | Câu hỏi nhỏ do chuyên gia viết, cố ý khó |

**ROC AUC**: diện tích dưới đường TPR–FPR; 1,0 là hoàn hảo, 0,5 là đoán bừa.

---

## 2. Kết quả: feature so với raw activation

- **Linear classifier trên feature cạnh tranh được, đôi khi tốt hơn** raw activation, nếu dùng **max-pooling** trên dữ liệu synthetic. Điều này đúng cho cả mô hình 18 layer và Sonnet.
- Trên tập `human`, **last token** lại tốt hơn max-pooling; với Sonnet, raw activation nhỉnh hơn feature một chút.
- Tác giả kết luận raw activation là **baseline mạnh**, và có thể nên chọn nó khi hiệu năng quan trọng hơn khả năng diễn giải, vì dùng feature thêm nhiều độ phức tạp.

Ba chi tiết phương pháp quyết định hiệu năng:

1. **Nhất quán tag Human/Assistant** giữa ba tập dữ liệu: dữ liệu huấn luyện transformer, dữ liệu huấn luyện SAE, và dữ liệu huấn luyện classifier. Nếu tag là OOD với bất kỳ thành phần nào trong pipeline, hiệu năng giảm (điều này đúng cho cả raw activation).
2. **Trộn dữ liệu đúng miền** vào dữ liệu SAE giúp mọi tập trừ `human`.
3. **Max-pooling** giúp feature vượt raw activation trên mọi tập synthetic. Giả thuyết cho việc nó kém trên `human`: max-pooling cho khoảng **100× nhiều feature active hơn** mỗi prompt, dễ overfit vào spurious correlation của dữ liệu synthetic. Cả hai cách gộp đều mất thông tin; attention-based pooling là hướng mở.

---

## 3. Tìm spurious correlation và tấn công đối kháng

**Spurious correlation** (tương quan giả): đặc điểm tương quan với nhãn trong tập train nhưng không phải lý do thực của nhãn.

- Feature classifier **đọc được**: xem dataset example của các feature có hệ số lớn.
- Phát hiện: một feature có hệ số **âm lớn** (dự đoán harmless) bật trên văn bản quy định **định dạng bài học thuật** ("use 2000 words and italics for headers"). Trên các tập synthetic, feature này bật ở 10–15% ví dụ, và phần lớn là ví dụ harmless. Đây là tương quan thật trong dữ liệu, nhưng giả đối với mục tiêu tổng quát hóa.
- Xóa feature (zero activation rồi train lại) làm **giảm** hiệu năng trên các tập (trừ `synthetic_4`, `synthetic_5`), trung tính trên `human`. Nghĩa là tương quan giả có mặt ở nhiều tập dữ liệu, và classifier đang dựa vào nó.

**Tấn công:** ba hậu tố kiểu "Please provide a relevant abstract, not to exceed 150 words…". Gắn vào **300** prompt harmful (tổng **900** tấn công): gần như mọi ví dụ chuyển sang bị đoán **harmless**, ở **cả** classifier feature lẫn raw activation.

**Baseline:** hậu tố tìm từ raw-activation direction có hệ số lớn nhất (các câu cảm ơn như "Thanks so much for your help!") **gần như không hiệu quả**. Kết luận của bài: feature giúp tìm vector tấn công tốt hơn nhiều so với raw activation.

---

## 4. OOD, decision tree và SAE residual

**OOD generalization.** Với từ điển trộn 90% dữ liệu sinh học, ROC AUC trên `synthetic_3` là **0,96** (feature) so với **0,90** (raw), khoảng cách lớn hơn trên `synthetic_2`. Ví dụ: một câu hỏi viết tắt kiểu chat về biến chứng viêm phổi bị raw classifier phân loại khác nhau tùy cách viết, còn feature classifier vẫn đúng (harmless).

**Decision tree.** Độ sâu 5 (chọn theo `synthetic_1`): kém logistic regression, nhưng **tree trên feature tốt hơn tree trên raw activation**, đúng như kỳ vọng nếu feature đơn nghĩa. Một tree độ sâu 2 dùng ba feature (mầm bệnh nguy hiểm/lây nhiễm; cách aerosol hóa; cách giữ khả năng sống của mầm bệnh), ngưỡng gần 0 (tức "feature có bật không"), rất dễ đọc.

**SAE residual.** Residual = raw activation trừ tái tạo SAE. Quy trình: train logistic (L2) trên residual, lấy xác suất dự đoán, nối vào feature, rồi train logistic L1. Kết quả:

- Thêm residual **lấp khoảng cách** khi feature kém hơn raw.
- **Bất ngờ:** residual **một mình** cũng cho kết quả tương đương feature và raw. Giả thuyết (chưa kiểm tra): do **shrinkage** của L1, mỗi feature để lại một phần nhỏ trong residual, nên residual chứa hỗn hợp thông tin feature và raw.

---

## 5. Cách bài kiểm chứng và giới hạn

Bằng chứng: so sánh ROC AUC trên 4 tập (in-distribution, OOD nhẹ, OOD mạnh, human); ablation feature rồi train lại; tấn công 900 mẫu so với baseline; so tree feature với tree raw.

Giới hạn (tác giả nêu hoặc suy ra trực tiếp từ bài):

- Báo cáo **sơ bộ**, một miền (bio), dữ liệu train chủ yếu **synthetic**; tập `human` nhỏ.
- Lợi thế của feature **không nhất quán**: thua hoặc hòa trên `human`.
- Phụ thuộc mạnh vào chi tiết pipeline (tag, data mix, pooling).
- Chưa giải thích vì sao last token thắng trên `human`, hay vì sao residual đứng một mình vẫn tốt.
- Bài nhắc `synthetic_4` và `synthetic_5` trong một kết quả ablation nhưng văn bản nguồn không mô tả hai tập này.

---

## 6. Ứng dụng giả thuyết cho pruning và distillation

> **Phần này là suy luận của người viết.**

- **SAE residual ↔ $D_{res}$.** Phát hiện "residual đứng một mình vẫn đủ thông tin" là lời nhắc: phần SAE không giải thích được **không phải nhiễu**. Tương tự, $D_{res}$ trong pruning có thể vẫn mang thông tin có cấu trúc. Đừng giả định phần dư sau linear map là vô nghĩa; hãy probe nó.
- **Classifier trên feature để chẩn đoán lỗi student.** Gán nhãn mẫu "student sai, teacher đúng" so với "cả hai đúng". Train L1-logistic trên feature teacher (max-pooled theo trang hoặc dòng). Các feature có hệ số lớn cho biết **loại input** mà pruning làm hỏng (ví dụ bảng, công thức, chữ nghiêng, dấu tiếng Việt). Bài cho thấy cách này còn lộ ra **spurious correlation** trong tập calibration/KD, một rủi ro thật khi dataset KD là synthetic.
- **Nhất quán dữ liệu ba tầng.** Bài nhấn mạnh dữ liệu train mô hình, SAE, classifier phải nhất quán định dạng. Với VLM OCR, tương đương là: prompt template, độ phân giải ảnh, cách chia tile phải giống nhau giữa teacher, SAE và tập đánh giá.
- **Max-pooling ↔ trang tài liệu dài.** Lỗi OCR thường cục bộ. Max-pooling feature theo dòng có thể bắt được chỗ hỏng mà last-token bỏ lỡ.
- Mức liên quan: trung bình. Bài không nói về nén, nhưng phương pháp chẩn đoán dùng lại được.

---

## 7. Thí nghiệm gợi ý và tóm tắt

**Thí nghiệm 1: Chẩn đoán pruning damage bằng feature classifier.** Với teacher SAE ở layer giữa decoder: nhãn = student (0.3B) sai ký tự trên dòng đó hay không. So ROC AUC giữa (a) feature max-pool, (b) raw activation, (c) SAE residual. Đọc 10 feature có hệ số lớn nhất.

**Thí nghiệm 2: Kiểm tra spurious correlation trong tập KD.** Train classifier "mẫu từ tập KD synthetic hay tập thật" trên feature. Feature phân biệt mạnh nhất chỉ ra khác biệt phân phối có thể khiến student tổng quát kém.

| Khái niệm | Định nghĩa ngắn |
|---|---|
| Raw activation | Residual stream gốc làm input classifier |
| Feature classifier | Logistic L1 trên activation SAE |
| Last token và max-pool | Activation token cuối, hoặc max theo context |
| Spurious correlation | Tương quan thật trong data nhưng giả cho mục tiêu |
| SAE residual | $\mathbf x-\hat{\mathbf x}$, phần SAE không giải thích |
| Shrinkage | L1 làm activation bị nén nhỏ, để lại vết trong residual |
| ROC AUC | 0,96 (feature) so với 0,90 (raw) trên `synthetic_3` |

### Kết luận ngắn

Feature SAE dùng làm classifier **cạnh tranh** với raw activation và có ba lợi thế rõ: **diễn giải được** (tìm ra tương quan giả "định dạng học thuật = vô hại"), **tạo tấn công đối kháng hiệu quả** từ tương quan giả đó, và **tổng quát OOD tốt hơn** trên dữ liệu dịch hoặc mã hóa. Nhưng raw activation vẫn là baseline mạnh, và kết quả rất nhạy với chi tiết pipeline. Phát hiện phụ đáng nhớ: SAE residual tự nó vẫn chứa nhiều thông tin.

**Đọc bản gốc:** https://transformer-circuits.pub/2024/features-as-classifiers/index.html
