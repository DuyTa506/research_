# Towards the Law of Capacity Gap in Distilling Language Models (2023)

**Tác giả:** Chen Zhang, Qiuchi Li, Dawei Song, Zheyu Ye, Yan Gao, Yan Hu.  
**Venue / arXiv:** arXiv:2311.07052 (v4, 30 Jul 2025); paper không nêu venue hội nghị trong văn bản.  
**Nguồn:** https://arxiv.org/abs/2311.07052  
**Chủ đề:** "Curse of capacity gap" (teacher quá lớn làm student kém đi) được biến thành một quy luật tuyến tính: teacher tối ưu ≈ 2,5 × kích thước student. Áp dụng để distill LLaMA2-7B / LLaMA3.1-8B thành MiniMA-3B.  
**Vị trí trong lộ trình:** Đọc trước: 08 (Teacher assistant — nơi "capacity gap" được nêu ở ảnh), 09 (Efficacy of KD), 11 (Patient and consistent teacher), 18–19 (TinyBERT, MiniLM), 40 (Gemma 2). Bài này thêm: **một công thức** chọn kích thước teacher thay vì thử nhiều teacher. Tiếp theo: 42 (Distillation scaling laws) làm lại câu hỏi này nghiêm ngặt và rộng hơn; 39 (Small models struggle with strong reasoners) là biểu hiện ở chuỗi suy luận.

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch paper. Phần "Ứng dụng cho dự án" là suy luận của chúng tôi. Đây là quy luật **thực nghiệm** rút ra từ mô hình nhỏ (<3B) và một cách thức pruning cụ thể; bài 42 sau này đặt nghi vấn về phạm vi của kết luận kiểu "teacher tối ưu tỉ lệ tuyến tính với student".

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
4. [Thí nghiệm và kết quả chính](#4-thí-nghiệm-và-kết-quả-chính)
5. [Phân tích / ablation](#5-phân-tích--ablation)
6. [Giới hạn](#6-giới-hạn)
7. [Liên hệ với các bài khác](#7-liên-hệ-với-các-bài-khác)
8. [Ứng dụng cho dự án (suy luận)](#8-ứng-dụng-cho-dự-án-suy-luận)
9. [Thí nghiệm gợi ý](#9-thí-nghiệm-gợi-ý)
10. [Tóm tắt công thức / khái niệm](#10-tóm-tắt-công-thức--khái-niệm)
11. [Kết luận ngắn](#11-kết-luận-ngắn)

---

## 1. Bối cảnh

**Curse of capacity gap** (gloss): người ta thường kỳ vọng teacher càng mạnh thì student càng tốt. Thực tế (Mirzadeh et al. 2020 ở thị giác, Zhang et al. 2023b ở LM) student của một cỡ cố định thường *tốt lên rồi xấu đi* khi teacher to dần. Lý do trực giác: student quá nhỏ không đủ sức biểu diễn phân phối phức tạp của teacher khổng lồ ("khoảng cách năng lực" quá lớn), nên teacher vừa phải lại tốt hơn. Hình 1 trong paper minh họa bằng GPT-2 và Pythia chưng cất trên OpenWebText, đánh giá perplexity WikiText2.

**Hệ quả thực tế.** Muốn biết teacher tối ưu phải chạy nhiều lần distill với các teacher cỡ khác nhau. Với LLM việc này quá tốn kém. Paper gọi là **"impossible triangle"**: ba mong muốn không thể đồng thời đạt — (1) kích thước student mong muốn, (2) teacher tối ưu cho nó, (3) chi phí tính toán nhỏ. Mục tiêu: nếu có một **quy luật** nối kích thước student với teacher tối ưu thì chi phí tìm teacher bằng 0.

## 2. Ý tưởng chính

Tác giả làm một nghiên cứu sơ bộ rẻ tiền: lấy các mô hình GPT-2 (140M đến 1.6B) và Pythia (70M đến 2.8B), cắt tỉa mỗi mô hình ở nhiều mức sparsity để tạo ra student, rồi distill từ mỗi teacher trong họ. Với mỗi cỡ student, họ tìm teacher cho kết quả tốt nhất. Các điểm (cỡ student, cỡ teacher tối ưu) nằm gần như trên một đường thẳng: $T^\star\approx 2{,}5\,S$ (R² = 0,9957). Họ kiểm tra bằng cách ngoại suy: distill LLaMA2-7B xuống 3B (tỷ lệ 0,4) và thu được mô hình MiniMA-3B tốt hơn các baseline pruning và layer-dropping cùng ngân sách.

## 3. Phương pháp chi tiết

### 3.1 Pipeline thí nghiệm sơ bộ

Với mỗi teacher $T$ trong họ mô hình: (1) prune $T$ tới sparsity $p$ để thành student $S$; (2) distill từ $T$ sang $S$; (3) so sánh các teacher khác nhau cho cùng cỡ $S$, quan sát teacher nào cho student tốt nhất.

### 3.2 Pruning (structured, theo độ quan trọng)

Mỗi attention head, mỗi neuron FFN và mỗi chiều hidden được gắn biến mặt nạ $\xi,\nu,\mu$ (hidden state dùng chung giữa các layer):

$$
\mathrm{MHA}(x_i)=\sum_{j=1}^{A}\xi_j\,\mathrm{Attn}(\mu\cdot x_{\le i};W^Q_j,W^K_j)\,\mu\, x_{\le i}W^V_jW^O_j,\qquad
\mathrm{FFN}(x_i)=\sum_{k=1}^{I}\nu_k\, g(x_iW^I_k)W^O_k .
$$

Độ quan trọng ("expressive score") = kỳ vọng của **giá trị tuyệt đối gradient** của loss theo biến mặt nạ (kiểu Molchanov 2017, gloss: "gradient bậc nhất ước lượng mất mát nếu bỏ thành phần đó"):

$$
I^{head}_j=\mathbb{E}\left|\frac{\partial \mathcal L_{pruning}}{\partial \xi_j}\right|,\quad
I^{neuron}_j=\mathbb{E}\left|\frac{\partial \mathcal L_{pruning}}{\partial \nu_j}\right|,\quad
I^{hidden}=\mathbb{E}\left|\frac{\partial \mathcal L_{pruning}}{\partial \mu}\right|
$$

(Công thức trong paper bị mất ký hiệu giá trị tuyệt đối do trích xuất PDF, nhưng văn bản mô tả là "accumulated absolute gradients".) Điểm số được chuẩn hóa $\ell_2$ để tránh tỷ lệ cắt lệch giữa các layer. Các thành phần cắt riêng với cùng sparsity $p'=1-\sqrt{1-p}$ (vì head/neuron và hidden state có hiệu ứng nhân nhau). Chỉ ~5% dữ liệu dùng cho pruning.

### 3.3 Distillation loss

$$
\mathcal L^{distill}=\frac12\sum_i -y_i^{teacher}\log y_i^{student}\;+\;\frac12\sum_i -y_i^{truth}\log y_i^{student}
$$

Nghĩa: nửa là cross-entropy với phân phối teacher (token-level KD kiểu Hinton), nửa là CE với nhãn thật. Đây là KD token-level "cơ bản nhất"; họ không dùng loss sequence-level vì kém hiệu quả.

### 3.4 Quy luật

Ba giả thuyết được kiểm: **H1** capacity gap ảnh hưởng student; **H2** teacher tối ưu tồn tại trong tập hữu hạn; **H3** teacher tối ưu không đổi qua các cấu hình (cho cùng cỡ student). Quan sát H1, H2 hợp lý từ đường cong "tăng → phẳng → giảm" theo cỡ teacher; H3 đúng theo quan sát tại sparsity ~60%. Hồi quy tuyến tính:

$$
\boxed{T^\star\approx 2{,}498\,S-11{,}498\approx 2{,}5\,S}\quad(\text{đơn vị: triệu tham số},\ R^2=0{,}9957)
$$

**Corollary 1:** nếu student có cỡ $0{,}4\,T^\star$ thì teacher cỡ $T^\star$ là tối ưu. Ví dụ: student 3B → teacher ≈ 7,5B, nên LLaMA2-7B là hợp lý (và 13B, 70B thì không).

### 3.5 Pseudo-code dùng luật

```
given: expected student size S (params)
T_star = 2.5 * S                      # luật
choose teacher closest to T_star       # vd LLaMA2-7B cho S=3B
student = structured_prune(teacher, target=S)   # width pruning, đối xứng giữa layer
distill(student <- teacher) with L = 0.5*CE(teacher)+0.5*CE(truth)
```

### 3.6 Chi tiết MiniMA

Teacher: LLaMA2-7B (đã mở rộng từ vựng tiếng Trung và train thêm) và LLaMA3.1-8B. Student 3B. Pruning thêm heuristic: chuyển từ ưu tiên toàn cục sang theo từng layer, bỏ 4 layer dưới cùng và 4 layer trên cùng, cắt đối xứng head/neuron/hidden. Distill: dữ liệu 126B token (Pile 57B 45,2%, GitHub 30B 23,8%, WuDao 39B 31,0%), seq 4096, batch 1024 (~4M token), lr 3e-4, weight decay 0,1, 1 epoch, 16 A100 80GB. Sau đó fine-tune thành MiniChat với 1,1M mẫu hướng dẫn.

## 4. Thí nghiệm và kết quả chính

**Nghiên cứu sơ bộ:** GPT-2 (base 140M, medium 380M, large 810M, xl 1,6B) và Pythia (70M, 160M, 410M, 1B, 1,4B, 2,8B); dữ liệu OpenWebText (~3,9B token), đánh giá perplexity WikiText2 và độ chính xác LAMBADA; có biến thể ít token hơn. Sparsity thử từ 20% đến 99%. Điểm chung: với mỗi cỡ student, hiệu năng theo cỡ teacher có dạng tăng–ổn định–giảm.

**Bảng 2 (trích) — MiniMA-3B vs baseline 3B (126B token trừ khi ghi):**

| Mô hình | Token | MMLU | CEval | DROP | BBH | GSM8K | HumanEval |
|---|---|---|---|---|---|---|---|
| LLaMA2-7B (teacher gốc) | 2T | 46,00 | 34,40 | 31,57 | 32,02 | 14,10 | 12,80 |
| ShortGPT 3B | 0B | 25,57 | 26,79 | 8,72 | 7,53 | 4,52 | 0,00 |
| LayerChop 3B | 126B | 25,76 | 27,23 | 10,45 | 12,23 | 6,97 | 3,66 |
| ShearedLLaMA 3B | 126B | 25,39 | 27,79 | 20,41 | 30,21 | 5,08 | 8,54 |
| ShearedLLaMA 3B | 252B | 26,15 | 27,81 | 20,03 | 30,11 | 5,26 | 9,01 |
| **MiniMA (teacher LLaMA2-7B)** | 126B | **28,51** | 28,23 | 22,50 | 31,61 | 8,11 | 10,98 |
| MiniMA (teacher LLaMA2-13B) | 126B | 26,82 | 28,23 | 18,92 | 30,01 | 5,31 | 10,37 |
| MiniMA (teacher LLaMA2-70B) | 126B | 26,91 | 24,89 | 19,03 | 28,55 | 3,56 | 7,32 |
| MiniMA (teacher LLaMA3.1-8B)‡ | 126B | 31,72 | 29,89 | 30,46 | 32,06 | 10,42 | 18,46 |

(‡ dùng corpus lớn hơn do từ vựng lớn hơn, theo chú thích bảng.) Điểm chính: 7B tốt hơn 13B và 70B làm teacher cho student 3B ("hiệu năng giảm đều khi teacher lớn hơn" theo lời paper). Tuy nhiên ngay cả MiniMA-3B tốt nhất (MMLU 28,51) còn kém xa LLaMA2-7B (46,00), tức các student này vẫn yếu nhiều trên các tác vụ khó với chỉ 126B token.

**MiniChat vs đối thủ (GPT-4 chấm, Macro avg Vicuna-Bench / BELLE-Bench):** vs OpenBuddy-3B 7,64:5,42 và 7,77:6,81; vs BiLLa-7B 7,24:7,41 và 7,73:7,49; vs ChatGLM2-6B 7,35:7,30 và 7,40:8,00.

## 5. Phân tích / ablation

**Bảng 4 (MiniMA 7B→3B):**

| Biến thể | MMLU | CEval | DROP | BBH | GSM8K | HumanEval |
|---|---|---|---|---|---|---|
| MiniMA | 28,51 | 28,23 | 22,50 | 31,61 | 8,11 | 10,98 |
| global pruning | 28,92 | 28,52 | 19,55 | 32,89 | 7,26 | 11,37 |
| layer dropping | 27,59 | 26,95 | 21,96 | 30,37 | 6,85 | 9,86 |
| patient KD | 28,69 | 28,37 | 22,51 | 31,89 | 8,00 | 10,64 |
| teacher assistant (5B) | 27,69 | 27,49 | 21,31 | 30,02 | 7,72 | 9,58 |

Diễn giải của paper: cách pruning đối xứng không hại (có khi lợi) so với global pruning và layer dropping; KD thường "không thua" patient KD nên **loss chưng cất là yếu tố thứ yếu** đối với luật; teacher assistant (chèn teacher 5B giữa) không cải thiện. Nhận xét của chúng tôi: chênh lệch trong bảng nhỏ (thường dưới 1 điểm), không có nhiều seed hay khoảng tin cậy, nên các so sánh "tốt hơn" giữa các biến thể này không có ý nghĩa thống kê rõ ràng.

Paper cũng nói luật giữ khi đổi họ mô hình (GPT-2 vs Pythia), đổi lượng dữ liệu, đổi paradigm pruning và đổi mục tiêu distill (tham chiếu Mục 4.3), và nhận xét rằng các SLM distill gần đây (kể cả Gemma 2, bài 40) đều chọn teacher có cỡ gần với luật này. Đó là nhận xét hậu nghiệm của tác giả, không phải thí nghiệm.

## 6. Giới hạn

**Do tác giả nêu:** (1) chưa làm đầy đủ thí nghiệm cho MiniMA, trong khi LLaMA3 có mô hình tới 400B; (2) không xét lượng dữ liệu, trộn dữ liệu và kỹ thuật distill như yếu tố scaling chính; (3) không có giai đoạn preference optimization cho MiniChat.

**Do chúng tôi nhận xét:**
- Luật được fit từ student **sinh ra bằng pruning chính teacher đó** (student là phiên bản cắt của teacher cùng họ), nên student thừa hưởng cấu trúc teacher. Không rõ có áp dụng cho student train từ khởi tạo ngẫu nhiên hay teacher/student khác họ.
- Dải fit là 70M đến ~3B, đoạn ngoại suy lên 7B chỉ có một điểm kiểm tra (7B tốt hơn 13B, 70B) với teacher là các LLaMA2 khác cỡ nhưng cùng họ.
- Hệ số 2,5 với β = −11,498 (tức đoạn cắt trục âm) khiến luật không thể đúng cho student rất nhỏ (<5M tham số cho T âm); nên coi là xấp xỉ trong dải đã đo.
- Teacher lớn hơn thua có thể vì **không train đủ lâu** (126B token cho student 3B), chứ không phải do "capacity" theo nghĩa bất biến. Bài 42 cho thấy bức tranh phụ thuộc vào ngân sách train của student và đưa ra mô hình định lượng hơn.
- Chỉ đo qua perplexity/benchmark; student MiniMA có điểm MMLU gần mức ngẫu nhiên (25) ở một số baseline nên chênh lệch dễ nhiễu.

## 7. Liên hệ với các bài khác

- **08 Teacher assistant (TAKD):** bài gốc nêu curse of capacity gap và đề xuất chèn teacher trung gian. Bài này không cố "chữa" mà đo quy luật; ablation cho thấy TA không giúp thêm.
- **09 / 10 (Efficacy of KD, Does KD really work):** cùng mạch "teacher mạnh hơn chưa chắc tốt hơn".
- **11 Patient and consistent teacher:** patient KD được so trong ablation; kết quả gần như hòa.
- **39 Small models struggle with strong reasoners:** cùng hiện tượng ở chuỗi suy luận dài (student nhỏ học kém từ teacher rất mạnh).
- **40 Gemma 2:** chọn teacher 27B cho student 9B (3:1), phù hợp cỡ "tỷ lệ vài lần" trong luật.
- **42 Distillation scaling laws:** làm lại câu hỏi bằng thiết kế thực nghiệm lớn hơn; giải thích khi nào teacher to hại và khi nào distill đáng giá.
- **18–19 (TinyBERT, MiniLM):** bài toán tương tự trên encoder nhỏ hơn nhiều.

## 8. Ứng dụng cho dự án (suy luận)

1. **Teacher 4B có "quá lớn" cho student 0,3B / 0,1B không?** Áp dụng thô luật 2,5×: student 0,3B tương ứng teacher ≈ 0,75B, student 0,1B tương ứng teacher ≈ 0,25B. Teacher 4B là gấp ~13× và ~40×. Nếu luật chuyển được, đây là vùng "curse". Nhưng luật được fit cho **text LM thuần, student prune từ chính teacher, dải ≤3B**; dự án là VLM OCR với student prune từ 0,8B chứ không phải từ 4B. **Không nên** kết luận mà không đo.
2. **Cấu trúc student khác teacher.** Student của dự án là bản cắt của LightOnOCR-0.8B, còn teacher là 4B khác (có thể khác họ). Giả định "student thừa hưởng cấu trúc teacher" của paper không còn.
3. **Giá trị thực dụng:** ý tưởng thử một teacher trung gian cỡ ~0,8B (chính mô hình gốc trước khi prune) làm đối chứng với teacher 4B là rẻ và đáp ứng đúng tinh thần luật. Self-distillation từ mô hình 0,8B chưa cắt chính là teacher "đúng tỷ lệ" cho student 0,3B/0,1B.
4. **Pruning của paper** (head/neuron/hidden theo gradient tuyệt đối, bỏ 4 layer đầu và 4 layer cuối) có điểm liên quan nhưng khác dự án: họ cắt cả width, và ablation cho thấy width-pruning đối xứng ngang hoặc hơn layer dropping, nhưng ở hiệu số rất nhỏ (xem Mục 5). Bằng chứng này không giải thích hay phản bác quan sát của bạn rằng cắt MLP neuron recovery kém hơn cắt layer.
5. **Hệ quả cho $D_{lin}/D_{res}$:** không liên quan trực tiếp; paper không phân tích loại sai số.

## 9. Thí nghiệm gợi ý

1. **Quét teacher ở dự án:** cố định student 0,3B đã prune, distill cùng số bước từ (a) bản 0,8B gốc, (b) teacher 4B, (c) một trung gian (nếu có, ví dụ ~1,5–2B). Vẽ CER/WER theo cỡ teacher để xem có đỉnh không (tái tạo Hình 1 cỡ nhỏ).
2. **Lặp ở student 0,1B:** xem đỉnh có dịch về teacher nhỏ hơn không (kiểm tra tính tuyến tính của luật ở điểm thứ hai).
3. **Tăng token để loại biến nhiễu "chưa train đủ":** lặp thí nghiệm 1 với gấp đôi ngân sách; nếu khoảng cách 4B–0,8B thu hẹp, "curse" là do ngân sách chứ không phải thuần capacity.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Curse of capacity gap | Teacher quá lớn so với student làm student kém đi |
| Impossible triangle | Cỡ student mong muốn – teacher tối ưu – chi phí nhỏ |
| $T^\star\approx 2{,}498S-11{,}498$ | Luật tuyến tính, R²=0,9957 (M tham số) |
| Corollary 1 | Student cỡ $0{,}4\,T^\star$ thì teacher $T^\star$ tối ưu |
| $p'=1-\sqrt{1-p}$ | Sparsity cho từng thành phần để đạt tổng $p$ |
| $\mathcal L=\tfrac12\mathrm{CE}(teacher)+\tfrac12\mathrm{CE}(truth)$ | Loss distill dùng |
| MiniMA-3B | 3B chưng cất từ LLaMA2-7B, 126B token |

## 11. Kết luận ngắn

Bài này đề xuất quy tắc ngón tay cái "teacher ≈ 2,5 × student" dựa trên thí nghiệm GPT-2/Pythia, có kiểm chứng ngoại suy ở 7B→3B. Giá trị là thực dụng (không phải thử nhiều teacher), nhưng phạm vi hẹp và thiếu giải thích cơ chế. Với dự án, đọc nó như một cảnh báo: thử teacher trung gian, đừng mặc định teacher 4B là tốt nhất cho student 0,1B. Bài 42 sẽ đưa ra mô hình tổng quát hơn.

**Link gốc:** https://arxiv.org/abs/2311.07052
