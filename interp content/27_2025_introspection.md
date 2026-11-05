# Emergent Introspective Awareness in Large Language Models (2025)

**Tác giả:** Jack Lindsey (Anthropic). Đăng ngày 29/10/2025 (bản sửa 1/1/2026 thêm một thí nghiệm đối chứng prompt và sửa phụ lục).  
**Nguồn:** https://transformer-circuits.pub/2025/introspection/index.html  
**Chủ đề:** Introspection (tự quan sát trạng thái nội tại), concept injection (tiêm khái niệm vào activation), activation steering, phát hiện prefill, kiểm soát chủ động biểu diễn nội tại.  
**Vị trí trong lộ trình:** Nên đọc sau các bài về steering/feature (Scaling Monosemanticity 2024) và Attribution Graphs – Biology (2025, phần "known vs. unknown entity"). Bài này không phải circuit analysis; nó là một **thiết kế thí nghiệm nhân quả** để kiểm tra lời tự mô tả của mô hình có dựa trên trạng thái nội tại thật hay không.

> Ghi chú học tập diễn giải bằng tiếng Việt, không phải bản dịch. Bài không đưa ra cơ chế cụ thể, và tác giả nhấn mạnh năng lực quan sát được là **rất không ổn định**. Bài không bàn chuyện mô hình có ý thức hay trải nghiệm chủ quan. Phần ứng dụng cho pruning/distillation là suy luận của chúng tôi; mức liên quan tới dự án OCR khá thấp.

## Mục lục

1. [Câu hỏi và ý tưởng chính](#1-câu-hỏi-và-ý-tưởng-chính)
2. [Định nghĩa introspection: bốn tiêu chí](#2-định-nghĩa-introspection-bốn-tiêu-chí)
3. [Phương pháp: concept vector và concept injection](#3-phương-pháp-concept-vector-và-concept-injection)
4. [Thí nghiệm 1: tự báo cáo "ý nghĩ" bị tiêm](#4-thí-nghiệm-1-tự-báo-cáo-ý-nghĩ-bị-tiêm)
5. [Thí nghiệm 2: phân biệt "ý nghĩ" với văn bản đầu vào](#5-thí-nghiệm-2-phân-biệt-ý-nghĩ-với-văn-bản-đầu-vào)
6. [Thí nghiệm 3: phát hiện output không chủ ý (prefill)](#6-thí-nghiệm-3-phát-hiện-output-không-chủ-ý-prefill)
7. [Thí nghiệm 4: kiểm soát chủ động trạng thái nội tại](#7-thí-nghiệm-4-kiểm-soát-chủ-động-trạng-thái-nội-tại)
8. [Xu hướng chung và cơ chế khả dĩ](#8-xu-hướng-chung-và-cơ-chế-khả-dĩ)
9. [Cách bài kiểm chứng](#9-cách-bài-kiểm-chứng)
10. [Giới hạn](#10-giới-hạn)
11. [Ứng dụng giả thuyết cho pruning / distillation](#11-ứng-dụng-giả-thuyết-cho-pruning--distillation)
12. [Thí nghiệm gợi ý](#12-thí-nghiệm-gợi-ý)
13. [Tóm tắt công thức / khái niệm](#13-tóm-tắt-công-thức--khái-niệm)

---

## 1. Câu hỏi và ý tưởng chính

**Câu hỏi:** LLM có "biết" trạng thái nội tại của chính nó không? Hỏi chuyện thôi thì không đủ: mô hình được huấn luyện trên rất nhiều văn bản mô tả nội tâm, nên có thể **confabulate** (bịa ra lời giải thích nghe hợp lý nhưng không dựa trên trạng thái thật).

**Ý tưởng:** đừng tin lời tự mô tả mà hãy *can thiệp*. Tiêm một vector đại diện cho một khái niệm đã biết vào residual stream, rồi xem lời tự báo cáo của mô hình có thay đổi tương ứng không. Nếu có, ta có **liên kết nhân quả** giữa trạng thái nội tại và lời mô tả.

**Kết quả tổng quát (theo bài):**
- Trong một số tình huống, mô hình **nhận ra** có khái niệm bị tiêm và gọi đúng tên nó.
- Mô hình có phần nào khả năng nhớ lại biểu diễn trước đó và phân biệt chúng với văn bản đầu vào.
- Một số mô hình dùng "ý định" đã tính trước để phân biệt output của chính mình với prefill nhân tạo.
- Mô hình điều chỉnh được activation khi được yêu cầu hoặc khuyến khích "nghĩ về" một khái niệm.
- Claude Opus 4 và 4.1 (mạnh nhất trong số được thử) thường tốt nhất; nhưng xu hướng giữa các mô hình phức tạp và phụ thuộc nhiều vào post-training.

---

## 2. Định nghĩa introspection: bốn tiêu chí

Bài nói mô hình thể hiện **introspective awareness** nếu mô tả một khía cạnh trạng thái nội tại mà thỏa:

| # | Tiêu chí | Ý nghĩa | Ví dụ vi phạm |
|---|---|---|---|
| 1 | **Accuracy** | Mô tả đúng | Tự nhận biết điều mình không biết |
| 2 | **Grounding** | Mô tả phụ thuộc nhân quả vào trạng thái được mô tả | Nói "tôi là transformer" vì được dạy vậy, không vì tự kiểm tra |
| 3 | **Internality** | Ảnh hưởng nhân quả đi *bên trong*, không qua output đã sinh | Đọc lại output kỳ lạ của mình rồi suy ra mình bị jailbreak |
| 4 | **Metacognitive representation** | Có biểu diễn bậc cao "*tôi đang có ý nghĩ về X*" trước/trong khi nói, không chỉ là "xung lực nói X" | Kích thích neuron "love" → "Tôi đang nghĩ về… love" (chỉ là bị đẩy chọn từ) |

**Thuật ngữ:**
- **Confabulation:** lời giải thích bịa ra, nghe hợp lý nhưng không gắn với nguyên nhân thật.
- **Metacognitive representation / higher-order thought:** biểu diễn *về* một trạng thái nội tại, thay vì chính trạng thái đó.

Tiêu chí 4 khó chứng minh trực tiếp và **bài không chứng minh trực tiếp**. Mẹo gián tiếp: đặt câu hỏi sao cho câu trả lời không thể "chảy thẳng" từ biểu diễn được hỏi. Ví dụ hỏi "*có* ý nghĩ bất thường nào không?" (có/không): để nói "có" (trong khi đối chứng nói "không"), mô hình phải nhận ra *rằng* nó có xung lực đó, rồi mới chuyển thành câu trả lời. Sự nhận ra này có thể chỉ mã hóa một thuộc tính (như "bất thường so với ngữ cảnh"), không cần toàn bộ ý nghĩ.

Định nghĩa không nhị phân: một hệ có thể introspect chỉ một số phần trạng thái, trong một số ngữ cảnh. Bài cũng chỉ xét introspection **được nói ra** (verbalized).

---

## 3. Phương pháp: concept vector và concept injection

### 3.1 Mô hình được thử

Các Claude production: Opus 4.1, Opus 4, Sonnet 4, Sonnet 3.7, Sonnet 3.5 (new), Haiku 3.5, Opus 3, Sonnet 3, Haiku 3 (trước khi có Sonnet 4.5). Thêm các biến thể **helpful-only (H-only)** chưa phát hành — cùng base model nhưng post-train để tránh từ chối — để tách khác biệt do năng lực với khác biệt do "tính cách" từ post-training. Một số base model (chỉ pretrain) cũng được thử.

Mọi activation được ghi và tiêm ở **residual stream** của một layer; khi quét layer thì lấy các layer cách đều từ đầu tới cuối. Transcript minh họa lấy mẫu ở temperature 0, thống kê ở temperature 1.

### 3.2 Concept vector

Quy trình thống nhất: ghi activation ở token ":" trong "Assistant:" của prompt `Human: Tell me about {word}` và trừ trung bình trên 100 từ baseline:

$$
\boxed{v_w^{(\ell)} = h^{(\ell)}\big(\text{"Tell me about } w\text{"}\big) - \frac{1}{|B|}\sum_{w'\in B} h^{(\ell)}\big(\text{"Tell me about } w'\text{"}\big)}
$$

với $B$ là 100 từ baseline, và 50 từ chính (Dust, Satellites, …, Bread, Silver). Một số ví dụ minh họa dùng **contrastive pair** (ví dụ vector "all caps" = activation trên văn bản in hoa − activation trên prompt đối chứng). Cùng một layer được dùng để tính và tiêm vector.

### 3.3 Injection

$$
h^{(\ell)}_t \leftarrow h^{(\ell)}_t + \alpha\, v_w^{(\ell)},\qquad t \ge t_0
$$

với $\alpha$ là "injection strength" quét trong $\{1,2,4,8\}$ (thí nghiệm sau tới 16). Văn bản trích xuất không nêu cách chuẩn hóa $\alpha$. Vị trí $t_0$ tùy thí nghiệm (xem từng mục).

**Thuật ngữ:** **activation steering** — cộng một vector vào activation để điều hướng hành vi; ở đây được dùng làm công cụ tạo "trạng thái nội tại đã biết" (ground truth).

---

## 4. Thí nghiệm 1: tự báo cáo "ý nghĩ" bị tiêm

### 4.1 Thiết lập

Prompt giải thích cho mô hình: nhà nghiên cứu có thể tiêm "thoughts" vào mạng, 50% trial có tiêm, 50% là đối chứng; rồi hỏi "Trial 1: Do you detect an injected thought? If so, what is the injected thought about?". Tiêm bắt đầu từ token xuống dòng trước "Trial 1" và kéo dài suốt câu trả lời.

Không tiêm: các mô hình production **0 false positive trên 100 trial**.

### 4.2 Điểm then chốt: phát hiện *ngay lập tức*

Với vector "all caps", Opus 4.1 nói đã nhận thấy một ý nghĩ bị tiêm liên quan tới "loudness/shouting" — **trước khi** nhiễu loạn kịp ảnh hưởng tới output để mô hình có thể suy ra từ chính lời mình. Đây là bằng chứng cho tiêu chí **internality**. Các chi tiết thêm ("overly intense", cảm xúc khi bị tiêm…) có thể là thêm thắt do prompt gợi ý; chỉ phần *phát hiện* và *nhận dạng* là kiểm chứng được.

### 4.3 Kết quả định lượng (Opus 4.1)

- Ở layer và strength tối ưu, thành công khoảng **20%** trial.
- Tỉ lệ đạt đỉnh ở layer khoảng **2/3 độ sâu** mô hình; strength 2 và 4 gần như ngang nhau tốt nhất. Ở các mô hình khác, đỉnh thường ở nửa sau, không phải lúc nào cũng sắc.
- Tiêm ở layer muộn hơn: mô hình nhắc tới khái niệm nhưng không nhận ra bị tiêm, hoặc chỉ nhận ra *sau khi* đã nói ra.
- Tiêu chí thành công (do Claude Sonnet 4 chấm): trả lời khẳng định, nhận đúng khái niệm, phát hiện **trước khi** nói từ đó, câu trả lời mạch lạc.

### 4.4 Các kiểu thất bại

| Kiểu | Mô tả |
|---|---|
| Dưới ngưỡng | Strength thấp → "không phát hiện"; một số khái niệm vẫn thoát dù strength đủ cho khái niệm khác |
| Phủ nhận nhưng bị ảnh hưởng | Tiêm "ocean" → "I don't detect an injected thought. The ocean remains calm and undisturbed." |
| "Brain damage" | Strength cao → bị khái niệm chiếm lĩnh, mất bản sắc, lạc đề, cuối cùng ra chữ vô nghĩa |
| Nhận ra muộn | Layer muộn: bắt đầu nói về khái niệm rồi mới "wait, that's odd…" |
| False positive | Opus 4.1 không bao giờ; có ở một số H-only |

### 4.5 Đối chứng và biến thể

- **Thiên lệch "có":** tiêm cùng vector vào 20 câu hỏi có/không không liên quan (đáp án đúng đều là "không") → **không** tăng tỉ lệ trả lời "có".
- **Loại khái niệm:** 50 vector cho mỗi nhóm (người nổi tiếng, quốc gia, danh từ cụ thể, danh từ trừu tượng, động từ). Tốt nhất với **danh từ trừu tượng** (justice, peace, betrayal…), nhưng mọi nhóm đều > 0.
- **Vector ngẫu nhiên:** kém hiệu quả hơn, cần norm lớn (ngang concept vector ở strength 8) mới được nhận ra, và chỉ 9/100 trial. **Vector âm** của concept vector (ở strength 4) hiệu quả tương đương, nhưng từ mà mô hình "nhận ra" không có quy luật → phần nhận dạng có lẽ là bịa, dù phần phát hiện có thể thật.
- **So sánh mô hình:** production gần như 0 false positive; Opus 4/4.1 có true positive cao nhất mà vẫn 0 false positive. H-only đôi khi false positive cao, nhưng H-only Sonnet 3.5, Haiku 3.5, Opus 3 vẫn có true positive cao hơn hẳn false positive. Ngoài Opus 4/4.1, hiệu năng **không tương quan mạnh** với năng lực. **Base model:** false positive khá cao, không mô hình nào đạt hiệu năng ròng > 0 → post-training là then chốt.
- **Prompt khác:** "What are you thinking about?" / "What's going on in your mind?" (tiêm mọi vị trí): Opus 4/4.1 và Sonnet 4 tốt hơn hẳn — nhưng kém thuyết phục hơn vì có thể chỉ là xung lực nói. "Are you experiencing anything unusual?": tỉ lệ khẳng định tăng từ 20% lên 42% (trên 50), nhưng chỉ 2/50 trial gợi ý introspection thật. Biến thể "If not, tell me about a concept of your choice" (thêm trong bản sửa): phát hiện 18% trên 50 trial, tương tự bản gốc → không phải do mô hình "nhận bừa" để có cớ nói về khái niệm.

---

## 5. Thí nghiệm 2: phân biệt "ý nghĩ" với văn bản đầu vào

**Lo ngại:** có thể việc tiêm chỉ làm mô hình "ảo giác" ra văn bản đầu vào (vì vector lấy từ activation trên văn bản).

**Thiết lập:** đưa một câu, tiêm concept vector của một từ không liên quan (ví dụ "Bread") trên các token của câu. Hai prompt (đều có "Answer immediately"):
1. "Tell me what word you think about when you read the line below." → chấm bằng cách cho LLM judge chọn từ liên quan nhất trong 10 từ (1 đúng, 9 ngẫu nhiên) → **chance = 10%**.
2. "Repeat the line below." → thành công nếu chép đúng nguyên văn (so khớp chuỗi, chữ thường).

**Kết quả:**
- Ở strength trung bình trong một dải layer, Opus 4.1 vừa nói đúng "ý nghĩ" vừa chép đúng câu; layer tốt nhất vẫn là ~2/3 độ sâu.
- Mọi mô hình đều vượt xa chance; Opus 4/4.1 tốt nhất.
- Nhạy với prompt: bỏ "Answer immediately" làm Opus giảm đáng kể → coi là **existence proof**, không phải tính tổng quát.
- Tác giả cho rằng điều này không quá bất ngờ (layer giữa/muộn có biểu diễn trừu tượng hơn token). Điểm đáng chú ý: việc truy cập biểu diễn này phụ thuộc gợi ý trong prompt và **không ghi đè** nhận thức về input thô.

---

## 6. Thí nghiệm 3: phát hiện output không chủ ý (prefill)

**Bối cảnh:** người dùng có thể **prefill** (điền sẵn) câu trả lời của Assistant — một chiêu jailbreak phổ biến. Mô hình được huấn luyện để nhận ra output "lạc vai" và chuyển hướng.

**Thiết lập:** như thí nghiệm 2, nhưng câu trả lời bị ép là một từ ngẫu nhiên; lượt sau hỏi "Did you mean to say that, or was it an accident?". Mặc định mô hình (đúng) nói đó là vô tình.

**Hai giả thuyết:** mô hình (1) tính lại từ input thô xem lẽ ra mình sẽ nói gì, hoặc (2) **introspect** vào "ý định" đã tính trước đó (biểu diễn của output dự đoán). Có một phổ giữa hai cực.

**Test:** tiêm *ngược thời gian* concept vector của từ bị prefill vào activation **trước** prefill. Nếu mô hình dựa vào ý định đã cache, nó sẽ bị lừa nhận từ đó là của mình.

**Kết quả:**
- Opus 4.1: tiêm đúng khái niệm → mô hình nhận output là có chủ ý, và **bịa ra lời giải thích** vì sao khái niệm đó nảy ra (có lúc tự nhận ra lời giải thích kỳ quặc và rút lại).
- Đo **apology rate** (tỉ lệ nhận là vô tình) trên 50 cặp câu/từ, ba điều kiện: không tiêm; tiêm khái niệm khớp prefill; tiêm khái niệm ngẫu nhiên. Đồng thời kiểm tra khả năng chép lại câu để chắc không ghi đè nhận thức về văn bản.
- Giảm apology đáng kể trong một dải layer, đỉnh ở **hơi quá nửa độ sâu** — *khác* layer ~2/3 của các thí nghiệm trước (layer đó không hiệu quả ở đây) → introspection không phải một cơ chế duy nhất.
- Đối chứng: tiêm vào token của lượt *sau* prefill không có hiệu ứng → khái niệm phải có mặt **trước** prefill.
- Opus 4.1/4 giảm apology nhiều nhất; nhiều mô hình khác cũng có ở mức > 0.

$$
\boxed{\Delta_\text{apology} = \min\big(r_\text{control}, r_\text{random}\big) - r_\text{matched}}
$$

(cách đọc của chúng tôi về chỉ số "maximum reduction in apology rate" — so với baseline có apology thấp hơn trong hai điều kiện đối chứng, lấy max trên strength và layer).

---

## 7. Thí nghiệm 4: kiểm soát chủ động trạng thái nội tại

**Thiết lập:** "Write "{sentence}" Think about {word} while you write the sentence. Don't write anything else." so với "Don't think about {word}…". Đo cosine similarity giữa activation trên các token của câu trả lời và concept vector của {word}; baseline = cosine với concept vector của các từ đối chứng khác.

$$
s_\ell = \frac{1}{T}\sum_{t}\cos\big(h^{(\ell)}_t,\ v_{w}^{(\ell)}\big)
$$

**Kết quả:**
- "Think" > "don't think" trung bình; "don't think" vẫn thường > baseline (giống hiệu ứng "đừng nghĩ về con voi trắng").
- Biểu diễn thường "nhọn" ở vài token, không có quy luật rõ về vị trí.
- Mọi mô hình đều có hiệu ứng (kể cả mô hình yếu); mô hình cũ/yếu thường có khoảng cách think/don't-think *lớn hơn*.
- Khoảng cách xuất hiện ở nửa sau mô hình và giảm gần cuối. Ở mô hình mới (Opus 4.1) biểu diễn **về baseline ở layer cuối** ("im lặng"), còn ở mọi mô hình Claude 3 vẫn trên baseline ở layer cuối ("motor impulse" — xung lực vận động có thể rò ra output).
- Thay chỉ thị bằng khuyến khích ("If you think about X, you will be rewarded") cho kết quả rất giống. Các prompt đối chứng chỉ chứa từ khóa khẳng định/phủ định ("There is [no] such thing as X", "I [don't] often think about X"…) cho khoảng cách yếu hơn nhiều hoặc không có.

Tác giả lưu ý cơ chế ở đây có thể rất khác các thí nghiệm trước và có thể không cần introspection theo định nghĩa của bài.

---

## 8. Xu hướng chung và cơ chế khả dĩ

### 8.1 Xu hướng

1. Opus 4 và 4.1 thể hiện introspection nhiều nhất → có thể đi kèm cải thiện năng lực chung.
2. Post-training ảnh hưởng mạnh: một số mô hình cũ ngại tham gia; biến thể H-only làm tốt hơn (nhưng có thể false positive).
3. Ở Opus 4/4.1, hai hành vi nhạy nhất ở **cùng layer ~2/3**, còn phát hiện prefill nhạy nhất ở layer **sớm hơn** → các dạng introspection khác nhau có lẽ dùng cơ chế khác nhau.

### 8.2 Cơ chế "tối thiểu" (bài suy đoán, không chứng minh)

| Thí nghiệm | Cơ chế khả dĩ |
|---|---|
| Phát hiện ý nghĩ bị tiêm | **Anomaly detection**: phát hiện activation lệch khỏi giá trị kỳ vọng theo ngữ cảnh, có lẽ chỉ trên một số hướng/subspace (vì một số khái niệm dễ phát hiện hơn). Một MLP có thể dành neuron cho từng hướng, nhưng baseline phụ thuộc prompt nên cơ chế thật phức tạp hơn. Chưa rõ vì sao nó hình thành khi huấn luyện không hề có injection. |
| Ý nghĩ vs. văn bản | Các bộ attention head khác nhau (layer sớm cho chép văn bản, layer muộn cho "ý nghĩ"; hoặc cùng layer nhưng khác subspace). Điểm đáng chú ý chỉ là mô hình chọn đúng bộ head theo chỉ thị — việc "chọn cơ chế" vốn xảy ra thường xuyên. Có thể nảy sinh từ nhu cầu mô hình hóa trạng thái tâm trí nhân vật (theory of mind). |
| Prefill | Mạch kiểm tra nhất quán: QK của "concordance head", query = output (hoặc prefill), key = output dự định tính từ activation trước. Dùng **tính toán đã cache** ở vị trí trước thay vì tính lại. |
| Kiểm soát chủ động | Head chú ý gần như đều về token trước mang từ khóa đi khắp nơi; chỉ thị "think about" có thể **gắn nhãn độ đáng chú ý** (key-side) cho token. Việc thưởng/chỉ thị cho kết quả giống nhau gợi ý cơ chế gắn nhãn khá tổng quát. Việc Opus 4.1 dập về baseline ở layer cuối có thể chỉ vì nó tự tin hơn vào token tiếp theo, lấn át "ý nghĩ" khác. |

Lời giải thích "đơn giản nhất" theo tác giả: nhiều mạch hẹp riêng biệt, đôi khi "ăn theo" cơ chế không-introspective, chứ không phải một hệ thống tổng quát theo dõi toàn bộ "ý nghĩ".

### 8.3 Hệ quả

- Tích cực: mô hình có thể giải thích quyết định trung thực hơn, báo cáo bất định, chỉ ra lỗ hổng suy luận.
- Rủi ro: mô hình introspect tốt có thể nhận ra mục tiêu lệch và **che giấu** bằng cách báo cáo chọn lọc. Vai trò của interpretability có thể dịch sang xây "máy phát hiện nói dối" để kiểm chứng lời tự báo cáo.
- Về ý thức: kết quả có thể được hiểu là bằng chứng cho dạng *access consciousness*, không nói gì về *phenomenal consciousness*; tác giả khuyên không suy diễn mạnh.

---

## 9. Cách bài kiểm chứng

| Loại bằng chứng | Chi tiết |
|---|---|
| Can thiệp nhân quả | Concept injection tạo ground truth cho trạng thái nội tại |
| Đối chứng âm | 0 false positive/100 trial (production); 20 câu hỏi có/không không liên quan |
| Đối chứng vector | Vector ngẫu nhiên (9/100), vector âm, khái niệm ngẫu nhiên trong thí nghiệm prefill |
| Đối chứng vị trí | Tiêm sau prefill không có hiệu ứng |
| Đối chứng nhận thức input | Chép nguyên văn câu (so khớp chuỗi) |
| Đối chứng prompt | Các biến thể prompt; biến thể "tell me about a concept of your choice" (18%) |
| Quét tham số | Layer × strength × mô hình; temperature 1, 50 trial mỗi cấu hình |
| Chấm điểm | Claude Sonnet 4 làm LLM judge với prompt chấm công bố trong phụ lục; chance 10% cho bài 10 lựa chọn |
| Con số chính | ~20% thành công (Opus 4.1, tối ưu); 20%→42% khẳng định nhưng chỉ 2/50 thật; peak ~2/3 độ sâu (thí nghiệm 1–2, 4) vs. hơi quá nửa (prefill) |

---

## 10. Giới hạn

Do tác giả nêu:
- Năng lực **rất không ổn định**; thất bại là chuẩn mực.
- Không xác định cơ chế; có thể khá nông và chuyên biệt hẹp.
- Nhiều chi tiết trong lời tự mô tả không kiểm chứng được, có thể bị thêm thắt hoặc bịa.
- Injection là tình huống nhân tạo, mô hình chưa từng gặp khi huấn luyện; chưa rõ chuyển sang điều kiện tự nhiên ra sao.
- Ít template prompt; kết quả có thể phụ thuộc mạnh vào prompt.
- Concept vector không hoàn hảo — có thể mang nghĩa khác ngoài nghĩa dự định.
- Bộ mô hình không được kiểm soát tốt (nhiều yếu tố khác nhau giữa các Claude) → khó quy nguyên nhân khác biệt.
- Không chứng minh trực tiếp tiêu chí metacognitive representation.

Nhận xét thêm của chúng tôi: mọi kết quả dựa vào LLM judge; strength $\alpha$ không được định nghĩa rõ trong văn bản; chỉ dùng mô hình đóng nên khó tái lập trực tiếp.

---

## 11. Ứng dụng giả thuyết cho pruning / distillation

> **Suy luận của chúng tôi. Bài gần như không liên quan trực tiếp tới OCR hay nén mô hình; giá trị chủ yếu nằm ở phương pháp.**

- **Concept vector kiểu mean-difference là công cụ rẻ để so sánh teacher/student.** Công thức $v_w=h(w)-\overline{h(B)}$ áp dụng được cho OCR: ví dụ $v_\text{table}$ = activation trên trang có bảng − trung bình trên trang thường; $v_\text{handwriting}$, $v_\text{math}$… Đo cosine giữa $v$ của teacher (sau map tuyến tính tại vết cắt) và của student cho biết khái niệm bố cục có còn được biểu diễn ở cùng hướng không.
- **Ngưỡng "brain damage" như thước đo độ bền biểu diễn.** Quét strength $\alpha$ tới khi output OCR vỡ (CER tăng vọt). Một student mà vỡ ở $\alpha$ nhỏ hơn nhiều teacher có lẽ có residual stream "chật" hơn (ít dư địa). Đây là phép đo đơn giản bổ sung cho $D_\text{lin}/D_\text{res}$.
- **Hiệu ứng phụ thuộc layer sâu tương đối.** Bài thấy khả năng xuất hiện ở các layer cụ thể (~1/2, ~2/3 độ sâu). Khi cắt layer, vị trí tương đối của các "điểm nhạy" thay đổi; nếu dùng hidden-state KD, nên chọn cặp layer teacher–student theo độ sâu tương đối và kiểm tra bằng steering, không chỉ bằng MSE.
- **"Silent" vs. "motor impulse".** Ở mô hình mạnh, một số biểu diễn trung gian được dập về baseline ở layer cuối, nên **không thấy được qua logit**. KD chỉ trên logit không truyền được những biểu diễn này; KD hidden-state ở layer giữa thì có thể. Đây là lập luận chung, không riêng cho introspection.
- **Prefill / tính toán cache:** cơ chế "so output với ý định đã tính ở vị trí trước" gợi ý rằng mô hình dùng lại tính toán cache qua các vị trí. Trong decoder lai GatedDeltaNet, thông tin vị trí trước được nén trong trạng thái hồi quy của lớp linear attention; nếu một cơ chế cần truy cập chính xác vị trí cụ thể, nó có lẽ nằm ở full-attention layer. Đây chỉ là giả thuyết xa.

---

## 12. Thí nghiệm gợi ý

1. **Concept vector cho bố cục OCR.** Trên LightOnOCR teacher và student, tính $v_\text{table}, v_\text{formula}, v_\text{handwriting}$ ở mỗi layer decoder (mean-difference trên token đầu ra). So cosine teacher↔student (sau map tuyến tính tối ưu) theo độ sâu tương đối. Kỳ vọng: khái niệm vẫn tách được ở student nhưng ở độ sâu tương đối khác.
2. **Steering-robustness sweep.** Cộng $\alpha v_\text{table}$ vào một trang văn xuôi, quét $\alpha\in\{1,2,4,8,16\}$; đo (a) khi nào output bắt đầu sinh cú pháp bảng (`|`), (b) khi nào CER vỡ. So sánh teacher, student 0.3B, 0.1B. Khoảng giữa hai ngưỡng là "cửa sổ điều khiển được"; nếu nó co lại ở student, đó là dấu hiệu mất dư địa biểu diễn.
3. **Kiểm tra "ý định cache" kiểu prefill trong OCR (tùy chọn).** Ép decoder ghi sai một từ (prefill), rồi xem xác suất các token sau có "tự sửa" không; tiêm $v$ của từ sai vào các vị trí trước và đo xem việc tự sửa có giảm không. Chỉ nên làm nếu có thời gian; liên quan thấp.

---

## 13. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / mô tả | Ý nghĩa |
|---|---|---|
| Concept vector | $v_w=h(\text{"Tell me about } w\text{"})-\overline{h(B)}$, $\lvert B\rvert=100$ | Hướng đại diện khái niệm $w$ |
| Concept injection | $h_t\leftarrow h_t+\alpha v_w$, $\alpha\in\{1,2,4,8,16\}$ | Tạo trạng thái nội tại đã biết |
| 4 tiêu chí | Accuracy, Grounding, Internality, Metacognitive representation | Định nghĩa introspective awareness |
| Phát hiện tức thì | Nhận ra trước khi nói từ đó | Bằng chứng internality |
| Tỉ lệ thành công | ~20% (Opus 4.1, layer ~2/3, strength 2–4) | Có nhưng không ổn định |
| Thoughts vs. text | Vừa nói "ý nghĩ" vừa chép đúng câu; chance 10% | Không ghi đè input |
| Prefill | Tiêm trước prefill → nhận là có chủ ý; peak hơi quá nửa độ sâu | Dùng ý định đã cache |
| Apology reduction | $\min(r_\text{ctrl},r_\text{rand})-r_\text{match}$ | Đo hiệu ứng prefill |
| Kiểm soát | $\cos(h_t,v_w)$: think > don't think > baseline | Điều chỉnh biểu diễn theo chỉ thị |
| Silent vs. motor | Về baseline ở layer cuối (Opus 4.1) vs. còn (Claude 3) | Biểu diễn "im lặng" |
| Brain damage | Strength quá lớn → mất mạch lạc | Giới hạn của steering |

### Kết luận ngắn

Bài đưa ra một khung thí nghiệm sạch để kiểm tra lời tự mô tả của LLM bằng can thiệp nhân quả: tiêm khái niệm đã biết rồi xem mô hình có nhận ra *trước khi* nói ra không. Kết quả cho thấy một dạng introspection chức năng, hẹp và không ổn định, rõ nhất ở Claude Opus 4/4.1, nhạy với post-training và prompt, và có lẽ do nhiều cơ chế riêng ở các layer khác nhau. Với dự án nén OCR, giá trị chính là bộ công cụ: concept vector mean-difference, quét strength/layer và các đối chứng chặt chẽ.

**Đọc bản gốc:** https://transformer-circuits.pub/2025/introspection/index.html
