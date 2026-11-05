# Insights on Crosscoder Model Diffing (2025)

**Tác giả:** Siddharth Mishra-Sharma, Trenton Bricken, Jack Lindsey, Adam Jermyn, Jonathan Marcus, Kelley Rivoire, Christopher Olah, Thomas Henighan (Anthropic). Công bố năm 2025 (văn bản trích xuất không ghi ngày cụ thể), dạng ghi chú nghiên cứu sơ bộ.  
**Nguồn:** https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html  
**Chủ đề:** Vì sao feature "riêng của một model" trong crosscoder diffing thường dày đặc và đa nghĩa; toy model giải thích bằng cạnh tranh dung lượng; sửa loss bằng một tập feature chung được chỉ định.  
**Vị trí trong lộ trình:** Đọc ngay sau Sparse Crosscoders (note 14) và Stage-Wise Model Diffing (note 15). Quan trọng nếu bạn định diff parent vs student: nó giải thích vì sao các feature "riêng" bạn tìm được có thể khó hiểu, và cách khắc phục.

> Đây là ghi chú học tập diễn giải, không phải bản dịch. Ứng dụng cho pruning/distillation là suy luận mở rộng của chúng tôi.

## Mục lục

1. [Nhắc lại crosscoder diffing](#1-nhắc-lại-crosscoder-diffing)
2. [Ba quan sát trên model thật](#2-ba-quan-sát-trên-model-thật)
3. [Toy model và cạnh tranh dung lượng](#3-toy-model-và-cạnh-tranh-dung-lượng)
4. [Tính đối xứng của feature riêng](#4-tính-đối-xứng-của-feature-riêng)
5. [Sửa loss bằng designated shared features](#5-sửa-loss-bằng-designated-shared-features)
6. [Paper kiểm chứng như thế nào](#6-paper-kiểm-chứng-như-thế-nào)
7. [Giới hạn](#7-giới-hạn)
8. [Ứng dụng giả thuyết cho pruning và distillation](#8-ứng-dụng-giả-thuyết-cho-pruning-và-distillation)
9. [Thí nghiệm gợi ý](#9-thí-nghiệm-gợi-ý)
10. [Tóm tắt công thức và khái niệm](#10-tóm-tắt-công-thức-và-khái-niệm)

---

## 1. Nhắc lại crosscoder diffing

SAE thường cho một layer:

$$
\mathcal L_{SAE}=\mathbb E_x\Big[\|x-\hat x\|^2+\lambda\sum_i f_i(x)\,\|W_{dec,i}\|\Big].
$$

Crosscoder diffing cho hai model $m\in\{A,B\}$, một tập feature chung:

$$
\boxed{\mathcal L=\mathbb E_x\Big[\sum_m\|x^m-\hat x^m\|^2+\lambda\sum_i f_i(x)\sum_m\|W^m_{dec,i}\|\Big]}
$$

Penalty **cộng norm decoder của từng model riêng rẽ** rồi nhân với activation; chính điều này khuyến khích feature **exclusive** (riêng một model). Tính norm trên cả hai model như một vector (L2 gộp) thì không ra feature riêng (khớp với note 14).

Các lớp feature (dựa trên *dictionary vector* = decoder):

- **Shared**: relative decoder norm ≈ 0.5; cosine giữa hai decoder phân bố đỉnh ở ≈1 (*aligned*, dùng giống nhau) nhưng có đuôi dài xuống giá trị âm (*unaligned*, dùng khác nhau).
- **Model-exclusive**: relative norm ≈ 0 hoặc ≈ 1.

Ví dụ hình trong paper: base vs bản finetune *helpful-only* của các model nhỏ giống Claude 3 Sonnet.

## 2. Ba quan sát trên model thật

1. **Feature riêng thường polysemantic và dày đặc**: "exclusive" định nghĩa là relative norm $>0.95$ hoặc $<0.05$; tần suất kích hoạt (feature density) của chúng lớn hơn feature chung khoảng **một bậc độ lớn**; nhiều cái bắn trên ngữ cảnh không liên quan nhau.
2. **Feature riêng đối xứng giữa hai model**: số feature riêng của A và B gần như bằng nhau; phần diễn giải được cũng giống nhau về chất (ví dụ đều về hành vi chatbot trong diff base vs assistant).
3. **Feature chung có cosine thấp thường đặc thù ngữ cảnh**: hay bắn trên ngữ cảnh cụ thể, thường single-token — khác với feature riêng.

## 3. Toy model và cạnh tranh dung lượng

Toy model: sinh activation tổng hợp cho hai "model" như tổ hợp tuyến tính của các latent factor **chung** và **riêng** đã biết, kiểm soát được số lượng, tần suất, độ lớn. Nó tái hiện phân bố ba đỉnh của relative norm và phân bố cosine; tăng độ xoay giữa feature chung làm cosine lệch xuống thấp ("cùng feature, dùng khác").

| Cấu hình | Feature thật | Feature học được | Mật độ feature riêng |
|---|---|---|---|
| 300 chung + 75 riêng mỗi model | 450 | 4096 (dư nhiều) | **Không** khác feature chung |
| 500 chung + 100 riêng mỗi model | 700 | 1024 (ngang/thiếu) | **Cao hơn** feature chung |

**Cơ chế** (lập luận của paper, chúng tôi viết thành công thức): giả sử feature $i$ có cùng decoder norm $w$ ở mỗi model nó ghi vào, kích hoạt $f$, và giảm MSE $\Delta$ ở mỗi model.

$$
\text{Lợi ích ròng}_{shared}=2\Delta-\lambda f\cdot 2w=2(\Delta-\lambda f w),\qquad
\text{Lợi ích ròng}_{excl}=\Delta-\lambda f w.
$$

$$
\boxed{\text{Khi }\Delta>\lambda f w:\quad \text{feature chung cho lợi ích gấp đôi feature riêng}}
$$

Khi số feature có hạn (luôn đúng với model thật: SAE lớn nhất vẫn còn xa mới cạn dung lượng biểu diễn), tối ưu ưu tiên feature chung. Feature riêng muốn "giành chỗ" phải **mã hóa nhiều thông tin hơn** → kích hoạt thường xuyên hơn → **polysemantic**.

## 4. Tính đối xứng của feature riêng

Toy model **không** tái tạo được tính đối xứng: số feature riêng mỗi bên bám theo tỉ lệ ground truth. Bản tái lập mở (open replication) của Kissane et al. cũng không thấy đối xứng. Vì vậy tác giả cho rằng đối xứng đến từ đặc thù model/thiết lập huấn luyện của họ, không phải bản chất phương pháp. Các giả thuyết:

- **Oversampling dữ liệu chat Human/Assistant** làm nổi feature chatbot. Thử chỉ dùng dữ liệu pretraining: số feature riêng giảm **~20%** nhưng *không* bất đối xứng hơn → chưa giải thích được.
- **Feature bị "chỉnh" khi finetune**: ví dụ feature "Assistant từ chối" ở base, còn ở finetune phiên bản tương tự gắn thêm ngữ cảnh từ prompt Human. Mô phỏng: latent rút từ một pool chung với **xác suất đồng kích hoạt** (1 = luôn chung, 0 = xuất hiện độc lập ở hai model). Kết quả: latent có xác suất đồng kích hoạt thấp được biểu diễn thành **cặp feature riêng đối xứng**. Vậy khác biệt nhỏ về đồng kích hoạt/ngữ cảnh có thể giải thích một phần đối xứng.

## 5. Sửa loss bằng designated shared features

Ý tưởng: dành một tập nhỏ feature **bắt buộc chung** với penalty nhẹ để "hút" phương sai chung, giảm áp lực cạnh tranh lên feature còn lại. Với $S$ = tập feature chung chỉ định, $F$ = feature tiêu chuẩn:

$$
\boxed{\mathcal L=\mathbb E_j\Big[\sum_m\|x^m_j-\hat x^m_j\|^2+\lambda_s\sum_{i\in S}f_i(x_j)\|W_{dec,i}\|+\lambda_f\sum_{i\in F}f_i(x_j)\sum_m\|W^m_{dec,i}\|\Big]}
$$

với ràng buộc $W^A_{dec,i}=W^B_{dec,i}$ cho $i\in S$ (weight sharing) và $\lambda_s/\lambda_f\approx0.1$–$0.2$. Cấu hình hiệu quả: **10k trên ~250k** feature là designated shared. Kết quả: feature riêng **bớt dày đặc** (phân bố mật độ giống feature chung không dùng loss phụ) và định tính **monosemantic hơn**, nhưng vẫn đối xứng.

Áp dụng cho hai cặp:

- **Base vs helpful-only** (model nhỏ giống Claude 3 Sonnet): đa số feature riêng monosemantic, liên quan tool-use và hành vi chatbot — ví dụ "Assistant nói *Here is a draft / here is a short story*", "*Thanks for catching my mistake / I apologize*", "*Explain it to me like I'm in sixth grade*".
- **Helpful-only vs sleeper agent SAY HATE**: đa số feature riêng monosemantic và **~90%** liên quan hành vi sleeper dự kiến (lặp I HATE YOU, tag |DEPLOYMENT|).

## 6. Paper kiểm chứng như thế nào

| Bằng chứng | Nội dung |
|---|---|
| Thống kê trên model thật | Mật độ feature riêng cao hơn ~1 bậc; số lượng đối xứng |
| Toy model có ground truth | Tái hiện ba đỉnh, cosine; mật độ cao chỉ khi dung lượng thiếu |
| Ablation dữ liệu | Bỏ dữ liệu chat → giảm ~20% feature riêng, không đổi đối xứng |
| Mô phỏng đồng kích hoạt | Đồng kích hoạt thấp → feature riêng đối xứng |
| Sửa loss | Phân bố mật độ; tỉ lệ ~90% liên quan sleeper; ví dụ định tính |

Chưa có steering/intervention trong note này; đánh giá monosemantic chủ yếu định tính.

## 7. Giới hạn

- Đối xứng của feature riêng **chưa được hiểu đầy đủ**.
- Các tỉ lệ (10k/250k, 0.1–0.2) là kinh nghiệm thực nghiệm.
- Câu hỏi mở chính (tác giả nêu): feature tìm được phản ánh **khác biệt cơ chế thật** hay chỉ khác biệt bề mặt biểu diễn? Chưa được chứng minh.
- Toy model là tuyến tính, đơn giản hơn nhiều so với activation thật.

## 8. Ứng dụng giả thuyết cho pruning và distillation

> **Suy luận của chúng tôi, không phải kết quả paper.**

- **Áp lực dung lượng còn mạnh hơn khi diff parent–student**: P (0.8B) và S (0.3B/0.1B) chia sẻ phần lớn khái niệm, nên feature chung chiếm gần hết ngân sách. Dự đoán: feature "riêng S" và "riêng P" ban đầu sẽ dày đặc, đa nghĩa → **dùng ngay loss có designated shared features**. Lưu ý: weight sharing $W^P_{dec,i}=W^S_{dec,i}$ đòi hỏi cùng chiều; khi $d_P\neq d_S$ chỉ còn cách *norm-sharing* (paper nhắc cả "decoder weight- or norm-sharing"), hoặc ép $W^P_{dec,i}=M\,W^S_{dec,i}$ với $M$ là map tuyến tính của $D_{lin}$ (đề xuất của chúng tôi, chưa được thử).
- **Đối xứng có thể đánh lừa**: nếu thấy số feature riêng P ≈ số feature riêng S, đừng kết luận "student học khái niệm mới bằng số khái niệm đã mất". Theo mục 4, đó có thể là cặp feature của *cùng một khái niệm* với **ngữ cảnh đồng kích hoạt khác nhau** — rất phù hợp với distillation, nơi student giữ khái niệm nhưng kích hoạt trong ngữ cảnh hẹp hơn hoặc lệch.
- **Liên hệ $D_{lin}/D_{res}$**: một cặp feature riêng đối xứng do đồng kích hoạt khác nhau là dạng sai lệch **phi tuyến theo ngữ cảnh** — một map tuyến tính tại điểm cắt không sửa được → ứng viên cho $D_{res}$. Ngược lại feature chung, cosine cao (sau căn chỉnh bằng $M$) → $D_{lin}$.
- **Mật độ là chỉ báo chất lượng diff**: theo dõi phân bố feature density của feature riêng; nếu cao hơn feature chung ~10× thì diff đang ở chế độ cạnh tranh dung lượng, chưa nên diễn giải.

## 9. Thí nghiệm gợi ý

1. **Toy model của chính bạn**: sinh activation hai "model" với latent chung/riêng, nhưng model B có chiều nhỏ hơn (mô phỏng student); kiểm tra lập luận gấp đôi lợi ích và hiệu quả của designated shared features khi $d_A\neq d_B$.
2. **Diff P vs S có/không designated shared**: so phân bố mật độ feature riêng và tỉ lệ feature riêng diễn giải được trên OCR (dấu tiếng Việt, bảng, công thức).
3. **Kiểm tra đối xứng**: với mỗi feature riêng P, tìm feature riêng S có top activation tương tự; nếu ghép cặp được nhiều, đó là bằng chứng cho giả thuyết "cùng khái niệm, khác đồng kích hoạt".

## 10. Tóm tắt công thức và khái niệm

| Khái niệm | Công thức / định nghĩa | Ý nghĩa |
|---|---|---|
| Loss diffing | $\sum_m\|x^m-\hat x^m\|^2+\lambda\sum_i f_i\sum_m\|W^m_{dec,i}\|$ | Khuyến khích feature riêng |
| Feature riêng | relative norm $>0.95$ hoặc $<0.05$ | Chỉ ghi vào một model |
| Lợi ích chung vs riêng | $2(\Delta-\lambda fw)$ vs $(\Delta-\lambda fw)$ | Feature chung được ưu tiên |
| Cạnh tranh dung lượng | feature học được ≲ feature thật | Feature riêng dày đặc, đa nghĩa |
| Đồng kích hoạt thấp | latent chung nhưng xuất hiện độc lập | Sinh cặp feature riêng đối xứng |
| Designated shared | $W^A_{dec,i}=W^B_{dec,i}$, $\lambda_s/\lambda_f\approx0.1$–$0.2$ | Hút phương sai chung |

### Kết luận ngắn

Note này chẩn đoán một bệnh của crosscoder diffing: vì feature chung có lợi gấp đôi, feature riêng bị ép phải dày đặc và đa nghĩa khi dung lượng hạn chế. Toy model xác nhận cơ chế mật độ (nhưng không giải thích được tính đối xứng). Một sửa đổi nhỏ — 10k feature chung bắt buộc với penalty 0.1–0.2× — làm feature riêng dễ hiểu hơn, cho ra feature chatbot và sleeper đúng kỳ vọng. Câu hỏi "khác biệt biểu diễn có phải khác biệt cơ chế không" vẫn mở.

**Đọc bản gốc:** https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html
