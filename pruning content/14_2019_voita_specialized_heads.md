# Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned (2019)

**Tác giả:** Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, Ivan Titov (Yandex; ĐH Amsterdam; ĐH Edinburgh; ĐH Zurich; MIPT).  
**Venue / arXiv:** arXiv:1905.09418 (v2, 7/6/2019). Paper trong file không ghi tên hội nghị.  
**Nguồn:** https://arxiv.org/abs/1905.09418  
**Chủ đề:** Phân tích vai trò từng attention head ở encoder của Transformer dịch máy; phương pháp cắt head bằng cổng ngẫu nhiên và nới lỏng khả vi của chuẩn $L_0$ (Hard Concrete).  
**Vị trí trong lộ trình:** Đọc ngay sau 13 (Sixteen Heads): cùng một phát hiện (nhiều head thừa) nhưng bài này đi từ góc "head nào làm việc gì" và đề xuất cách cắt có **học** (learnable gates). Nền cần có: 05 Network slimming (hệ số học được để cắt) và khái niệm regularization. Bài kế tiếp: 15, 16; ý tưởng cổng $L_0$ quay lại ở 17 (CoFi) và Sheared LLaMA (38).

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Phần "Ứng dụng cho dự án" là suy luận của chúng tôi.

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

Transformer cho dịch máy (NMT) dùng multi-head attention với 8 head, và báo cáo gốc cho rằng 8 head hơn 1 head khoảng 1 BLEU ở cùng cỡ mô hình. Nhưng các phân tích trước đó chỉ nhìn **trung bình** hoặc **giá trị lớn nhất** của trọng số attention trên các head, nên che mất việc các head đóng góp rất khác nhau.

Bài hỏi bốn câu: dịch phụ thuộc bao nhiêu vào từng head của encoder? Head có vai trò nhất quán, diễn giải được không? Loại attention nào (encoder self-attn, decoder self-attn, decoder-encoder attn) và layer nào nhạy với số head? Có giảm mạnh số head mà giữ chất lượng được không?

Thuật ngữ:
- **Encoder-decoder (seq2seq)**: encoder đọc câu nguồn, decoder sinh câu đích; có 3 loại attention như nói trên.
- **Dependency relation (quan hệ phụ thuộc cú pháp)**: ví dụ `nsubj` (chủ ngữ danh từ), `dobj` (tân ngữ trực tiếp), `amod` (tính từ bổ nghĩa), `advmod` (trạng từ bổ nghĩa).
- **LRP (layer-wise relevance propagation)**: phương pháp lan truyền ngược "độ liên quan" để biết nơron/đơn vị nào đóng góp vào một đầu ra.
- **Chuẩn $L_0$**: số thành phần khác 0; phạt $L_0$ ép nhiều thành phần bằng đúng 0 (tắt hẳn), nhưng không khả vi.

## 2. Ý tưởng chính

Hai giai đoạn. (1) Dùng LRP đo mức đóng góp của từng head vào logit top-1, kèm "confidence" (trọng số attention lớn nhất trung bình) của head; chỉ một ít head mỗi layer là quan trọng, và chúng có vai trò rõ: chú ý token kề bên (positional), chú ý token có quan hệ cú pháp nhất định (syntactic), hoặc trỏ tới từ hiếm nhất câu (rare words). (2) Để kiểm tra các head còn lại có thừa không, gắn mỗi head một cổng vô hướng $g_i$ độc lập với đầu vào, lấy từ phân phối Hard Concrete, và huấn luyện tiếp từ mô hình đã hội tụ với phạt $L_0$ khả vi. Kết quả: phần lớn head bị cắt, những head chuyên biệt là những head **bị cắt sau cùng**, xác nhận độc lập kết quả của LRP.

## 3. Phương pháp chi tiết

### 3.1 Xác định head quan trọng

- **Confidence** của head: trung bình (trên các token của tập dev) của trọng số attention cực đại, không tính token EOS (vì không phải token thật).
- **LRP**: dùng để ước lượng mức đóng góp của đầu ra từng head ở mỗi layer vào logit top-1 mà mô hình dự đoán. Chi tiết LRP nằm ở Appendix A (không đọc ở đây).
- Quan sát: LRP chỉ ra rất ít head quan trọng hơn hẳn các head khác trong mỗi layer; độ liên quan LRP khớp khá tốt với confidence (ngoại lệ: head quan trọng nhất ở layer 1, có confidence thấp: chính là "rare words head").

### 3.2 Ba vai trò được định nghĩa

| Vai trò | Tiêu chí của paper |
|---|---|
| **Positional** | ≥ 90% thời gian, trọng số cực đại rơi vào một vị trí tương đối cố định (thực tế là -1 hoặc +1, token kề bên). Trọng số cực đại trung bình > 0.8 cho mọi head positional ở mọi cặp ngôn ngữ |
| **Syntactic** | "accuracy" (tần suất head trỏ tới token có quan hệ đó, theo parser CoreNLP, hai chiều tính riêng) cao hơn ít nhất 10 điểm tuyệt đối so với baseline "vị trí tương đối phổ biến nhất của quan hệ đó" |
| **Rare words** | head ở layer 1 trỏ tới token hiếm nhất câu |

### 3.3 Cắt head bằng cổng $L_0$ (Hard Concrete)

Mỗi head $i$ được nhân với cổng vô hướng $g_i$ (tham số riêng của head, **không** phụ thuộc đầu vào):

$$\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}_i(g_i\cdot \mathrm{head}_i)\,W^O$$

Ta muốn phạt số cổng khác 0:

$$L_0(g_1,\dots,g_h)=\sum_{i=1}^{h}\big(1-[[g_i=0]]\big)$$

Không khả vi, nên dùng **nới lỏng ngẫu nhiên** (Louizos et al., 2018): $g_i$ là biến ngẫu nhiên từ phân phối **Hard Concrete** trên $[0,1]$ với tham số $\phi_i$: lấy phân phối Concrete (Gumbel-softmax) có support $(0,1)$, kéo giãn ra $(-\epsilon,1+\epsilon)$ rồi dồn phần khối lượng ngoài $[0,1]$ về hai điểm 0 và 1. Kết quả là phân phối hỗn hợp rời rạc-liên tục có khối lượng dương tại 0 và 1. Khi đó kỳ vọng số cổng khác 0 có dạng đóng:

$$\boxed{L_C(\phi)=\sum_{i=1}^{h}\big(1-P(g_i=0\mid\phi_i)\big)},\qquad \boxed{\mathcal L(\theta,\phi)=\mathcal L_{xent}(\theta,\phi)+\lambda\,L_C(\phi)}$$

- $\theta$: tham số Transformer gốc; $\mathcal L_{xent}$: cross-entropy dịch máy.
- Gradient qua việc lấy mẫu $g_i$ dùng **reparameterization trick**; $L_C$ và gradient của nó có dạng đóng.
- Mỗi batch lấy mẫu lại $g_i$. Khi test, $g_i$ là 0 hoặc 1 tùy $P(g_i=0)$ hay $P(g_i=1)$ lớn hơn.
- **Hiện tượng đẹp**: mô hình hội tụ về cổng hoặc gần như đóng ($P(g_i=0)\approx1$) hoặc hoàn toàn mở ($P(g_i=1)\approx1$) dù không có gì ép buộc cổng mở hẳn; nhiễu ngẫu nhiên đẩy mô hình tránh giá trị giữa. Nên khi suy luận mô hình là Transformer chuẩn với tập con head.

```text
Input: mô hình Transformer đã hội tụ (theta*), hệ số lambda
1. Khởi tạo theta = theta*; thêm cổng g_i ~ HardConcrete(phi_i) cho mỗi head
2. Khi cắt encoder: đóng băng decoder, chỉ fine-tune encoder + phi
3. Lặp train với  L = L_xent + lambda * sum_i (1 - P(g_i = 0))
4. Test: g_i = 1 nếu P(g_i=1) > P(g_i=0), ngược lại 0 (loại hẳn head)
5. Thay đổi lambda để được các mô hình với số head còn lại khác nhau
```

Việc đóng băng decoder khi chỉ cắt encoder là chủ ý: để chức năng của head encoder bị cắt không "di cư" sang decoder.

## 4. Thí nghiệm và kết quả chính

**Dữ liệu và mô hình.** Ngôn ngữ nguồn tiếng Anh, đích Nga/Đức/Pháp; mỗi cặp dùng cùng 2.5 triệu cặp câu WMT (bằng lượng dữ liệu EN-RU, không gồm UN và Paracrawl); EN-RU còn thêm OpenSubtitles2018 (có chạy 6 triệu cặp) để xem ảnh hưởng miền. Tập held-out dùng cho phân tích: 50k câu tiếng Anh lấy từ WMT EN-FR. Encoder 48 head (8 head × 6 layer, theo tổng nêu trong Table 2: 48/48/48).

### 4.1 Vai trò các head quan trọng

- Head positional trùng nhiều với head confident nhất và head LRP cao nhất.
- Syntactic (Table 1, EN-RU, độ chính xác của head tốt nhất / baseline vị trí):

| Quan hệ | WMT | OpenSubtitles |
|---|---|---|
| nsubj v→s | 45 / 35 | 77 / 45 |
| nsubj s→v | 52 / 35 | 70 / 45 |
| dobj v→o | 78 / 41 | 61 / 46 |
| dobj o→v | 73 / 41 | 84 / 46 |
| amod noun→adj.m. | 74 / 72 | 81 / 80 |
| amod adj.m.→noun | 82 / 72 | 81 / 80 |
| advmod v→adv.m. | 48 / 46 | 38 / 33 |
| advmod adv.m.→v | 52 / 46 | 42 / 33 |

  (Với amod và advmod khoảng cách tới baseline nhỏ; paper kết luận "certain heads" học được quan hệ cú pháp rõ rệt, nổi bật là nsubj và dobj.) Nhiều head cùng đảm nhiệm một quan hệ. Paper nêu giả thuyết nhưng nói chưa rút được kết luận chắc về ảnh hưởng hình thái học ngôn ngữ đích.
- Rare-words head: ở layer 1. Trên OpenSubtitles, với các câu mà token hiếm nhất không nằm trong top-500 token phổ biến, head này trỏ tới token hiếm nhất ở **66%** trường hợp và tới một trong hai token hiếm nhất ở **83%**; trên WMT, trỏ tới một trong hai token hiếm nhất ở hơn 50% trường hợp.

### 4.2 Cắt head encoder (Figure 7)

- OpenSubtitles: **chỉ mất 0.25 BLEU khi cắt tất cả trừ 4 head trên 48**.
- WMT (EN-RU): **10 head** encoder là đủ để nằm trong 0.15 BLEU của mô hình đầy đủ. Khớp với abstract: cắt 38/48 head, giảm chỉ 0.15 BLEU.

### 4.3 Cắt tất cả loại attention (Table 2, EN-RU; số head theo thứ tự enc self / dec self / dec-enc)

| Dữ liệu | Số head còn | BLEU (cắt từ mô hình đã train) | BLEU (train từ đầu cùng cấu hình) |
|---|---|---|---|
| WMT 2.5m | baseline 48/48/48 | 29.6 | n/a |
| | 14/31/30 | 29.62 | 29.47 |
| | 12/21/25 | 29.36 | 28.95 |
| | 8/13/15 | 29.06 | 28.56 |
| | 5/9/12 | 28.90 | 28.41 |
| OpenSubtitles 6m | baseline 48/48/48 | 32.4 | n/a |
| | 27/31/46 | 32.24 | 32.23 |
| | 13/17/31 | 32.23 | 31.98 |
| | 6/9/13 | 32.27 | 31.84 |

Kết luận của paper: trên WMT cắt gần 3/4 head encoder và hơn 1/3 head của decoder self-attn và decoder-encoder attn mà không mất chất lượng đáng kể; có thể cắt hơn một nửa tổng số head với mất tối đa 0.25 BLEU.

## 5. Phân tích / ablation

- **Chức năng của các head còn lại (Figure 8, EN-RU Subtitles).** Mô hình còn 17 head (đã cắt 2/3) vẫn giữ **đủ mọi chức năng** đã nhận diện. Khi còn ít head hơn nữa, một số chức năng "trôi" sang head khác (ví dụ head positional bắt đầu theo dõi phụ thuộc cú pháp).
- **Cắt rồi fine-tune tốt hơn train từ đầu** với cùng cấu hình head: chênh khoảng 0.2-0.6 BLEU (chú thích 8 trong paper) và Table 2 cột phải thấp hơn cột giữa, ví dụ WMT 5/9/12: 28.90 so với 28.41. Giải thích của tác giả: head ít có khả năng học vai trò quan trọng khi train từ đầu với số head nhỏ; tương hợp với các báo cáo của Zhu & Gupta 2017 và Gale et al. 2019 (bài 03) rằng kiến trúc thưa học qua pruning không train từ đầu được tới cùng chất lượng.
- **Loại attention nào quan trọng nhất (Figure 9).** Mô hình cắt head encoder self-attn trước; decoder-encoder attn là quan trọng nhất ở cả hai dataset ("without decoder-encoder attention no translation can happen"). Decoder self-attn (đóng vai trò mô hình ngôn ngữ phía đích) quan trọng gần ngang decoder-encoder attn trên WMT (câu dài, trung bình 24 token), và quan trọng hơn encoder self-attn trên OpenSubtitles (câu ngắn, 8 token).
- **Theo layer của decoder (Figure 10).** Self-attn head được giữ lại nhiều ở layer thấp; decoder-encoder attn head ở layer cao. Diễn giải của tác giả: layer thấp của decoder lo mô hình hóa ngôn ngữ, layer cao lo điều kiện hóa theo câu nguồn.

## 6. Giới hạn

**Do tác giả nêu:**
- Tập trung vào encoder self-attention; cắt các loại khác chỉ ở phần thứ hai.
- Tác giả viết rõ họ chưa so sánh phương pháp cắt này với các phương pháp nén NMT khác (hướng tương lai).
- Chưa rút được kết luận chắc về ảnh hưởng hình thái của ngôn ngữ đích lên head cú pháp.
- Không rõ liệu mô hình có thể train từ đầu với ít head như vậy (kết quả train từ đầu kém hơn).

**Do chúng tôi nhận thấy (suy luận):**
- Chỉ dịch máy EN→{RU,DE,FR}; chưa chứng minh cho encoder-only, decoder-only hay đa phương thức.
- Cổng $g_i$ độc lập với đầu vào, nên đây là **cắt tĩnh**; không tìm được head "quan trọng cho một số câu".
- Chỉ báo cáo BLEU, không báo latency/bộ nhớ thực; giảm số head không tự nhiên cho tăng tốc nếu cài đặt giữ nguyên.
- Số liệu BLEU trong Table 2 không kèm phương sai theo seed, nên chênh 0.1-0.2 BLEU khó kết luận (ứng dụng checklist bài 12).
- Vai trò "diễn giải được" dựa trên tiêu chí ngưỡng (90%, +10%) do tác giả đặt; các head còn lại có thể có vai trò khác.

## 7. Liên hệ với các bài khác

- **13 Sixteen Heads**: cùng thời, cùng kết luận "head thừa". Michel et al. cắt bằng điểm gradient một lần (không học); Voita học cổng bằng $L_0$. Michel là phân tích mô hình đã train, Voita có bước fine-tune với phạt. Michel tự ghi nhận Voita là "complementary" trong Related Work.
- **05 Network slimming**: cũng học hệ số vô hướng cho mỗi channel với phạt thưa ($L_1$); bài này dùng $L_0$ nới lỏng thay cho $L_1$.
- **03 Zhu & Gupta, 11 Rethinking the value of pruning**: kết quả "cắt rồi fine-tune > train từ đầu" ở đây ủng hộ nhóm quan sát của Zhu & Gupta, trái với thông điệp chính của bài 11 (nên xem lại cả hai trong cùng bối cảnh: phụ thuộc cài đặt).
- **15 LTH for BERT, 16 Movement pruning**: cắt trọng số với mask học được.
- **17 CoFi**: dùng mask $L_0$ Hard Concrete để cắt head, hidden unit, layer, phát triển trực tiếp từ ý tưởng của bài này (và Louizos 2018, Wang et al. 2020).
- **38 Sheared LLaMA**: cũng dùng mask kiểu $L_0$ nhưng cho LLM.
- **12 State of pruning**: kiểm tra bài này với checklist (ít điểm hoạt động, một dataset chính, chưa so sánh phương pháp khác).

## 8. Ứng dụng cho dự án (suy luận)

1. **Phương pháp học cổng $L_0$ cho đơn vị cấu trúc** là công cụ cắt tự động rất phù hợp: gắn cổng cho head ViT, cho neuron MLP, thậm chí cho cả block, và đặt ngân sách bằng $\lambda$. Đây gần với cách CoFi (17) làm.
2. **Giữ decoder cố định khi chỉ cắt encoder** (tác giả làm để tránh chức năng "di cư"). Tương đương cho dự án: khi cắt ViT, đóng băng decoder của VLM trong pha đầu để biết thiệt hại thuần của ViT; sau đó mở khóa để distillation hồi phục. Đây là một cách tách $D_{lin}$/$D_{res}$ theo vị trí (ý tưởng, không có trong paper).
3. **Kết quả "cắt rồi fine-tune > train từ đầu"** ủng hộ chiến lược của dự án: khởi tạo từ mô hình lớn rồi cắt và distill, thay vì train mô hình 0.3B/0.1B từ đầu.
4. **Hiện tượng chức năng trôi (function drift)** gợi ý rằng sau khi cắt, các head còn lại *gánh thêm vai trò*; đây là lý do phục hồi bằng distillation hiệu quả (giả thuyết, cần kiểm).
5. **Cảnh giác**: các vai trò positional/syntactic là của NMT và encoder; trong OCR-VLM, head ViT có thể có vai trò không gian (lân cận 2D, theo dòng chữ). Một phiên bản của phép phân tích này cho ViT: head nào chú ý patch kề bên/ cùng dòng. Chưa rõ kết quả sẽ giống không.
6. **Không áp được trực tiếp** cho lớp GatedDeltaNet (không có ma trận attention softmax).

Mức liên quan: **trung bình-cao** về phương pháp (cổng $L_0$), thấp về kết luận ngôn ngữ học.

## 9. Thí nghiệm gợi ý

1. **Cổng $L_0$ cho head ViT**: gắn Hard Concrete gate cho mỗi head của 12 block ViT (decoder đóng băng, cross-entropy OCR + $\lambda L_C$), vẽ đường (số head còn lại, CER) với 5 giá trị $\lambda$; so với cắt theo điểm Taylor (bài 13).
2. **Positional heads trong ViT**: tính tỉ lệ thời gian trọng số attention cực đại rơi vào patch kề (±1 theo hàng/cột) cho mỗi head; kiểm tra các head này có nằm trong tập được giữ lại sau cắt không (tái hiện logic "head chuyên biệt cắt sau cùng").
3. **Train từ đầu vs cắt**: lấy cấu hình ViT sau cắt, khởi tạo ngẫu nhiên và train cùng ngân sách (nhỏ) để thấy khoảng cách cắt-vs-scratch như Table 2.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Confidence | trung bình trọng số attention cực đại (bỏ EOS) |
| LRP | đo đóng góp của từng head vào logit top-1 |
| Head positional | ≥ 90% attention cực đại vào vị trí tương đối cố định (±1) |
| Head syntactic | accuracy cao hơn baseline vị trí ≥ 10 điểm |
| Cổng head | $\mathrm{Concat}_i(g_i\,\mathrm{head}_i)W^O$, $g_i$ độc lập đầu vào |
| $L_0$ | $\sum_i(1-[[g_i=0]])$ |
| Nới lỏng | $L_C(\phi)=\sum_i(1-P(g_i=0\mid\phi_i))$ với Hard Concrete |
| Mục tiêu | $\mathcal L_{xent}+\lambda L_C$ |
| Quy trình | bắt đầu từ mô hình hội tụ, thêm cổng, fine-tune (decoder đóng băng khi cắt encoder) |
| Kết quả chính | EN-RU WMT: 38/48 encoder head bị cắt, -0.15 BLEU; Subtitles: còn 4/48 head, -0.25 BLEU |

## 11. Kết luận ngắn

Chỉ một số ít head encoder quan trọng và chúng đảm nhiệm vai trò rõ (kề bên, quan hệ cú pháp, từ hiếm). Cổng ngẫu nhiên Hard Concrete với phạt $L_0$ nới lỏng cho phép cắt hầu hết head mà giảm rất ít BLEU, và các head chuyên biệt bị cắt sau cùng, một bằng chứng độc lập về tầm quan trọng của chúng. Cắt-rồi-tinh-chỉnh tốt hơn huấn luyện lại từ đầu với cùng cấu hình thưa. Với dự án, đây là mẫu thiết kế "cổng học được + ngân sách" để cắt đơn vị cấu trúc, tiền thân của CoFi.

Link gốc: https://arxiv.org/abs/1905.09418
