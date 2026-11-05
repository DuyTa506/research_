# Are Sixteen Heads Really Better than One? (2019)

**Tác giả:** Paul Michel, Omer Levy, Graham Neubig (CMU; Facebook AI Research).  
**Venue / arXiv:** arXiv:1905.10650 (v3, 4/11/2019); NeurIPS 2019 (33rd Conference on Neural Information Processing Systems).  
**Nguồn:** https://arxiv.org/abs/1905.10650  
**Chủ đề:** Cắt tỉa attention head của Transformer/BERT sau huấn luyện; điểm quan trọng của head dựa trên gradient; head quan trọng thay đổi thế nào trong quá trình train.  
**Vị trí trong lộ trình:** Đọc sau 07 (Molchanov importance 2019, vì điểm Taylor ở đây chính là phương pháp đó áp cho head) và 12 (cách đọc pruning phê phán). Bài này là bài **đầu tiên** của nhánh "pruning cho Transformer", cùng thời với 14 (Voita), cho thấy đơn vị cắt có thể là *head*. Bài kế tiếp: 14, rồi 15-17 (BERT, movement pruning, CoFi vốn cũng cắt head).

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Phần "Ứng dụng cho dự án" là suy luận của chúng tôi, không phải kết luận của paper.

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

Transformer dùng **multi-head attention (MHA)**: thay vì một phép attention, chạy $N_h$ phép attention song song ("head"), mỗi head có ma trận $W_q, W_k, W_v, W_o$ riêng với chiều nhỏ hơn $d_h = d/N_h$, rồi cộng kết quả. Lý do thường được nêu: mỗi head có thể chú ý vào một kiểu quan hệ khác nhau (ví dụ chủ-vị, phụ thuộc cú pháp).

Câu hỏi của paper: *các head đó thực sự mua được gì?* Nếu một mô hình train với 12-16 head mà tại test chỉ cần vài head thì phần lớn tham số attention là thừa. Đây cũng là động cơ cho **structured pruning** (cắt cả khối) chứ không chỉ cắt từng trọng số: bỏ một head bỏ đúng một khối ma trận dense, nên tăng tốc thật.

Thuật ngữ:
- **Ablation**: tắt một thành phần rồi đo hiệu năng thay đổi thế nào.
- **BLEU**: điểm đánh giá dịch máy (càng cao càng tốt). **MNLI**: tác vụ suy luận ngôn ngữ (3 lớp), đo accuracy.
- **Pruning "tại test time"**: cắt trên mô hình đã train xong, *không* fine-tune lại (điểm quan trọng của bài).

## 2. Ý tưởng chính

Đơn giản: gắn một biến mask nhị phân $\xi_h$ cho mỗi head, đặt $\xi_h=0$ để tắt head. Thí nghiệm cho thấy: tắt riêng lẻ phần lớn head không làm hiệu năng giảm có ý nghĩa thống kê, và nhiều layer có thể chỉ giữ **một** head mà hầu như không mất gì. Để cắt nhiều head cùng lúc, paper xếp hạng mọi head theo điểm quan trọng $I_h$ (độ nhạy của loss với mask, tức Taylor bậc nhất theo Molchanov) rồi cắt lần lượt từ thấp lên cao; làm vậy cắt được 20% (dịch máy) và 40% (BERT) số head mà không tổn hại đáng kể. Phần phân tích cho thấy attention encoder-decoder trong dịch máy cần nhiều head hơn self-attention, và các head quan trọng được xác định sớm trong lúc train.

## 3. Phương pháp chi tiết

### 3.1 Attention và MHA

Một lớp attention (scaled bilinear) với $W_k,W_q,W_v,W_o$:

$$\mathrm{Att}(x,q)=W_o\sum_{i=1}^{n}\alpha_i W_v x_i,\qquad \alpha_i=\mathrm{softmax}\!\Big(\frac{q^\top W_q^\top W_k x_i}{\sqrt d}\Big)$$

MHA (cách viết "concatenation" tương đương, tổng các head):

$$\mathrm{MHAtt}(x,q)=\sum_{h=1}^{N_h}\mathrm{Att}_{W_k^h,W_q^h,W_v^h,W_o^h}(x,q)$$

với $W_k^h,W_q^h,W_v^h\in\mathbb R^{d_h\times d}$, $W_o^h\in\mathbb R^{d\times d_h}$. Để giữ số tham số không đổi, $d_h=d/N_h$, và MHA có thể xem như ensemble của các attention hạng thấp.

### 3.2 Mask head

$$\boxed{\mathrm{MHAtt}(x,q)=\sum_{h=1}^{N_h}\xi_h\,\mathrm{Att}_h(x,q),\qquad \xi_h\in\{0,1\}}$$

Tắt head $h$ ⇔ $\xi_h=0$ (thay output của head bằng 0).

### 3.3 Điểm quan trọng của head

$$\boxed{I_h=\mathbb E_{x\sim X}\left|\frac{\partial \mathcal L(x)}{\partial \xi_h}\right|=\mathbb E_{x\sim X}\left|\mathrm{Att}_h(x)^\top\frac{\partial\mathcal L(x)}{\partial \mathrm{Att}_h(x)}\right|}$$

Ý nghĩa: nếu thay đổi $\xi_h$ làm loss đổi nhiều thì head quan trọng. Dấu **trị tuyệt đối** theo từng mẫu được tác giả gọi là "crucial" để các mẫu dương/âm không triệt tiêu nhau. Paper nói công thức này **tương đương phương pháp Taylor của Molchanov et al. (2017)**; chỉ cần một lượt forward + backward, không chậm hơn huấn luyện. Điểm được **chuẩn hóa theo layer bằng chuẩn $\ell_2$** (theo khuyến nghị của Molchanov).

### 3.4 Thuật toán cắt lặp

```text
Input: mô hình đã train, dữ liệu để ước lượng I (train hoặc tập con)
1. Tính I_h cho mọi head (forward+backward, trung bình trên dữ liệu), chuẩn hóa l2 theo layer
2. Sắp xếp head theo I_h tăng dần
3. for step in 1..: 
       đặt xi_h = 0 cho thêm 10% tổng số head thấp nhất (tích lũy)
       đo BLEU/accuracy trên tập đánh giá
```

Không fine-tune sau khi cắt. Chọn thứ tự theo $I_h$ nhanh hơn và tốt hơn thứ tự theo "độ chênh điểm khi ablate từng head" (oracle đơn lẻ).

*Ví dụ nhỏ:* BERT-base có 12 layer × 12 head = 144 head. Mỗi bước cắt khoảng 10% (≈ 14 head) theo thứ hạng toàn mạng (sau chuẩn hóa theo layer); ở bước 4 (≈ 40%) còn khoảng 86 head mà accuracy chưa giảm đáng kể.

## 4. Thí nghiệm và kết quả chính

**Hai mô hình** (đã train sẵn):
- **WMT**: Transformer "large" của Vaswani (6 layer, 16 head/layer), EN→FR WMT2014, bản pretrained của Ott et al. 2018; BLEU trên newstest2013; có 3 loại attention (Enc-Enc, Enc-Dec, Dec-Dec); kiểm định ý nghĩa bằng paired bootstrap (1000 mẫu).
- **BERT**: base-uncased (12 layer, 12 head) fine-tune trên MultiNLI; accuracy trên tập "matched"; t-test.

### 4.1 Ablate một head (mục 3.2 của paper)

- Phần lớn head có thể bỏ riêng lẻ mà điểm gần như không đổi; có trường hợp bỏ head còn làm điểm *tăng*.
- Trên encoder self-attention của WMT (Table 1, BLEU gốc 36.05): chỉ **8 trên 96 head** gây thay đổi có ý nghĩa thống kê (p < 0.01), **một nửa trong 8 đó thực ra làm BLEU tăng**.

### 4.2 Chỉ giữ một head mỗi layer (mục 3.3)

| Mô hình | Kết quả |
|---|---|
| WMT (Table 2) | Phần lớn layer giữ 1 head ổn; ngoại lệ lớn: **layer 6 Enc-Dec: -13.56 BLEU**; layer 5 Enc-Dec: -1.55; layer 1 Enc-Enc: -1.31 |
| BERT (Table 3) | Chênh accuracy tốt nhất mỗi layer từ -0.96% (layer 9) đến +0.10% (layer 2); **không** thay đổi nào có ý nghĩa thống kê ở p < 0.01 |

Một layer còn 1 head chỉ còn 1/16 (WMT) hoặc 1/12 (BERT) tham số attention. Khi chọn "head tốt nhất" bằng tập validation riêng (newstest2013 cho WMT; 5000 mẫu train MNLI cho BERT) rồi đánh giá trên test, kết luận giữ nguyên: không thay đổi có ý nghĩa ở 50% layer của WMT và **100%** layer của BERT (chi tiết trong Appendix A của paper, không tái hiện ở đây).

### 4.3 Head quan trọng có giống nhau giữa các miền?

Dùng tập ngoài miền: MTNT (EN→FR) cho WMT, MNLI "mismatched" cho BERT. Ảnh hưởng của việc bỏ từng head tương quan dương **> 0.5** giữa hai miền (p < 0.001; hình ghi Pearson r = 0.56 cho WMT, r = 0.68 cho BERT). Kết luận của tác giả: các head quan trọng nhất là quan trọng "phổ quát" (ở mức độ phép thử này).

### 4.4 Cắt lặp theo $I_h$ (mục 4)

- Cắt tới **20% head của WMT** và **40% head của BERT** mà không có tác động tiêu cực đáng chú ý.
- Cắt thêm thì hiệu năng rơi mạnh: cả hai mô hình **không** thể giảm xuống toàn single-head nếu không retrain.
- Appendix B có thêm 4 dataset (không ghi lại ở đây).

### 4.5 Hiệu quả suy luận (Table 4, BERT, MNLI-matched, mẫu/giây, GTX 1080Ti, 3 lần lặp trên mỗi 2 máy)

| Batch size | 1 | 4 | 16 | 64 |
|---|---|---|---|---|
| Original | 17.0 ± 0.3 | 67.3 ± 1.3 | 114.0 ± 3.6 | 124.7 ± 2.9 |
| Pruned (50% head) | 17.3 ± 0.6 | 69.1 ± 1.3 | 134.0 ± 3.6 | 146.6 ± 3.4 |
| Tăng tốc | +1.9% | +2.7% | +17.5% | +17.5% |

Mỗi head chiếm 6.25% (WMT) hoặc ≈ 8.34% (BERT) tham số của một lớp attention; MHA chiếm khoảng 1/3 tổng tham số của cả hai mô hình. Lưu ý: **với batch nhỏ, tăng tốc gần như biến mất** (+1.9% ở batch 1). Tác giả cảm ơn Thomas Wolf (Hugging Face) vì tái hiện độc lập giúp sửa một lỗi trong thí nghiệm tốc độ, nên đây là ví dụ tốt về việc nên kiểm chứng số đo latency.

## 5. Phân tích / ablation

**Loại attention nào cần nhiều head? (mục 5, Figure 4, WMT).** Cắt lặp riêng từng loại attention: Enc-Dec nhạy nhất; cắt hơn **60%** head Enc-Dec là sụp đổ hiệu năng, trong khi self-attention (Enc-Enc, Dec-Dec) vẫn dịch ra BLEU khoảng 30 khi chỉ giữ **20%** số head ban đầu. Kết luận: attention encoder-decoder phụ thuộc multi-headedness nhiều hơn nhiều.

**Động học huấn luyện (mục 6, Figure 5).** Mô hình nhỏ IWSLT (6 layer, 8 head/layer, DE→EN, IWSLT 2014). Lặp thí nghiệm cắt mỗi epoch:
- Epoch rất sớm (1-2): điểm giảm tuyến tính theo % cắt, tức mọi head quan trọng gần như ngang nhau.
- Từ epoch 10 trở đi: có "cụm head không quan trọng"; có thể cắt tới 40% tổng head mà vẫn giữ 85-90% BLEU gốc.
- Diễn giải của tác giả: head quan trọng được xác định sớm (nhưng không tức thì) trong train. Họ liên hệ với hai pha của Shwartz-Ziv & Tishby (ERM rồi nén) nhưng nói rõ cần nghiên cứu sâu hơn; đây là bằng chứng sơ bộ ("precursory evidence"), không phải kết luận chắc.

**So sánh điểm:** thứ tự theo $I_h$ tốt hơn và nhanh hơn thứ tự theo hiệu năng khi ablate từng head.

## 6. Giới hạn

**Do tác giả nêu / ngụ ý:**
- Chỉ hai mô hình chính (WMT Transformer, BERT-MNLI) và vài dataset phụ; các kết quả ablate dựa trên tập test nhỏ (tác giả tự đặt nghi vấn về tính tổng quát, rồi kiểm bằng tập ngoài miền).
- Không retrain/fine-tune sau cắt; cắt hơn 20-40% làm hiệu năng rơi mạnh.
- Phần động học train chỉ trên mô hình nhỏ IWSLT, giải thích còn sơ bộ.
- Thứ tự greedy theo $I_h$ tránh tìm kiếm tổ hợp, nên không đảm bảo tối ưu.

**Do chúng tôi nhận thấy (suy luận):**
- Mô hình 2019, tác vụ NLU/MT, kích thước nhỏ so với VLM/LLM hiện nay; chưa rõ nhận xét "một head là đủ" còn đúng cho mô hình decoder-only lớn, đặc biệt không có Enc-Dec attention.
- Tăng tốc chỉ đo ở batch lớn trên GPU cũ; ở batch 1 gần như bằng 0 (bản thân bảng đã cho thấy). Với dự án batch-1, cắt head có thể giảm tham số nhưng ít giảm latency.
- Chuẩn hóa $\ell_2$ theo layer làm thứ hạng so sánh *giữa* layer khó diễn giải (mỗi layer đều bị cắt phần nào).
- Số liệu mỗi dòng bảng chỉ là trung bình ít lần chạy; không đo được ảnh hưởng của sự phụ thuộc giữa các head (hai head thừa riêng lẻ nhưng cùng bỏ thì mất thông tin).

## 7. Liên hệ với các bài khác

- **07 Molchanov importance 2019**: $I_h$ chính là điểm Taylor bậc nhất. Bài này là một ứng dụng sớm trong NLP.
- **12 State of pruning**: dùng checklist ở bài 12: bài này chỉ có 2 mô hình chính và rất ít điểm hoạt động (bước 10%), có độ lệch chuẩn cho tốc độ nhưng không cho điểm số cắt.
- **14 Voita et al.**: nghiên cứu song song cùng thời; Voita học mask bằng $L_0$ trong khi bài này dùng điểm gradient một lần (paper tự so sánh ở mục Related Work: hai cách "bổ sung cho nhau").
- **15 LTH for BERT, 16 Movement pruning**: tiếp tục cắt BERT, nhưng cắt trọng số, học mask trong lúc fine-tune.
- **17 CoFi**: cắt *đồng thời* head, hidden unit và layer bằng mask học được, kế thừa trực tiếp ý "mask $\xi$ cho head".
- **05 Network slimming, 04 Pruning filters**: tương tự ở CNN (đơn vị là channel/filter).
- **26 LLM-Pruner, 28 FLAP, 29 SliceGPT**: cắt head/channel ở LLM.

## 8. Ứng dụng cho dự án (suy luận)

1. **Điểm quan trọng của head là công cụ rẻ để lập hồ sơ (profiling)** mô hình: chạy forward+backward trên một ít ảnh-văn bản OCR, tính $I_h$ cho từng head của (a) ViT 12 block và (b) các lớp full-attention của decoder. Nó cho biết head nào dư thừa mà không cần train.
2. **Với decoder lai (GatedDeltaNet + full attention)**: chỉ các lớp full-attention có "head" theo nghĩa bài này; các lớp GatedDeltaNet là linear attention nên cần đơn vị cắt khác (ví dụ head state / key-value head của chúng, cần đọc code). Bài này không nói gì về linear attention (suy luận).
3. **Latency batch-1**: Table 4 gợi ý cắt head có thể không giúp batch 1 (+1.9%). Với GPU ≤ 8 GB và batch-1, đóng góp chính của cắt head sẽ là giảm *bộ nhớ* (tham số và KV cache nếu cắt cả K/V head) chứ không nhất thiết giảm latency. Hãy đo thật.
4. **Cảnh báo cho quan sát "cắt MLP neuron hồi phục kém hơn cắt layer"**: ở đây head ablate riêng lẻ gần như vô hại, song cắt tập thể bị rơi mạnh sau 20-40%. Đây là ví dụ của "redundancy riêng lẻ không bằng redundancy tập thể", điều bạn cần kiểm tra cho cả neuron MLP.
5. Phát hiện "Enc-Dec attention cần nhiều head" gợi ý: trong VLM, **cross-modal path** (nơi phần văn bản đọc đặc trưng ảnh) có thể nhạy hơn self-attention thuần; nên cẩn thận khi cắt ở chỗ nối vision-language (giả thuyết, không có trong paper).

Mức liên quan: **trung bình** (ý tưởng và công cụ chấm điểm hữu ích; kỹ thuật cắt trực tiếp có thể ít lợi cho batch-1).

## 9. Thí nghiệm gợi ý

1. **Heatmap $I_h$ cho ViT và decoder**: tính điểm quan trọng theo head (chuẩn hóa $\ell_2$ theo layer) trên ~500 trang OCR; so với cắt ngẫu nhiên cùng số head, đo CER/edit-distance không fine-tune ở 10/20/30/40%.
2. **Cá thể vs tập thể**: ablate từng head (như Table 1) và so với cắt theo thứ hạng $I_h$; kiểm tra điểm nào bắt đầu có sự "mất đột ngột" (đường cong gập).
3. **Đo latency thật** của mô hình đã cắt 50% head ở batch 1 và batch 8 trên GPU đích, để biết liệu mô hình của bạn có hiện tượng tăng tốc biến mất như Table 4 hay không.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| MHA với mask | $\sum_h \xi_h\,\mathrm{Att}_h(x,q)$, $\xi_h\in\{0,1\}$ |
| Điểm head | $I_h=\mathbb E_x\left\lvert\frac{\partial\mathcal L}{\partial\xi_h}\right\rvert=\mathbb E_x\left\lvert\mathrm{Att}_h^\top\frac{\partial\mathcal L}{\partial\mathrm{Att}_h}\right\rvert$ |
| Chuẩn hóa | $\ell_2$ theo từng layer |
| Thuật toán | xếp $I_h$ tăng dần, cắt dần 10% mỗi bước, không fine-tune |
| Ablate-all-but-one | giữ 1 head mỗi layer; BERT: không đổi có ý nghĩa; WMT layer 6 Enc-Dec: -13.56 BLEU |
| Giới hạn cắt | khoảng 20% head (WMT), 40% (BERT) không hại đáng kể |
| Tốc độ | +17.5% ở batch 16/64, +1.9% ở batch 1 (BERT, cắt 50% head) |
| Động học | head quan trọng định hình sớm; epoch 1-2 chưa phân biệt |

## 11. Kết luận ngắn

Multi-head attention thừa đáng kể sau khi train: nhiều head có thể bỏ riêng lẻ, nhiều layer chỉ cần một head, và có thể cắt 20-40% head toàn mạng bằng một điểm gradient rẻ mà không retrain. Nhưng không cắt hết được, attention encoder-decoder nhạy hơn, và lợi ích tốc độ phụ thuộc mạnh vào batch size. Với dự án, hãy dùng ý tưởng "mask + điểm Taylor" làm công cụ chẩn đoán, và đừng giả định cắt head sẽ giảm latency batch-1.

Link gốc: https://arxiv.org/abs/1905.10650
