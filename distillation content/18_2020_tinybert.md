# TinyBERT: Distilling BERT for Natural Language Understanding (2020)

**Tác giả:** Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu (Huawei Noah's Ark Lab, HUST).
**Venue / arXiv:** arXiv:1909.10351 (bản v5, tháng 10/2020; paper được biết đến là Findings of EMNLP 2020 nhưng bản text không ghi venue).
**Nguồn:** https://arxiv.org/abs/1909.10351
**Chủ đề:** Distillation theo từng tầng cho Transformer: loss trên embedding, attention, hidden state, logit; khung hai giai đoạn (general + task-specific) và data augmentation.
**Vị trí trong lộ trình:** Đọc trước: 02 (Hinton KD), 03 (FitNets, "hint" ở tầng giữa), 16 (DistilBERT), 17 (Patient KD). Bài này thêm: chuẩn hóa việc distill **mọi** loại biểu diễn bên trong Transformer, kèm bảng ablation cho biết loại nào quan trọng nhất. Bài kế tiếp: 19 (MiniLM) làm ngược lại, chỉ distill tầng cuối.

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch nguyên văn. Phần "Ứng dụng cho dự án" là suy luận của chúng ta, paper không kiểm chứng trên OCR/VLM. Con số lấy từ paper; bảng nào bị lỗi khi trích PDF sẽ được ghi rõ.

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

BERT là mô hình ngôn ngữ **pre-train rồi fine-tune**: pre-train trên kho văn bản lớn (không nhãn), sau đó fine-tune cho từng tác vụ (phân loại, hỏi đáp...). BERT-base có 109M tham số, quá nặng cho thiết bị biên. Trước TinyBERT, các cách distill BERT đã có:

- **BERT-PKD** (17): distill tầng giữa nhưng chỉ ở giai đoạn fine-tune.
- **DistilBERT** (16): distill ở giai đoạn pre-train, nhưng student phải khởi tạo bằng cách lấy lại một nửa số tầng của teacher, nên bị khóa cùng hidden size.
- Cả hai chủ yếu bắt chước đầu ra (logit) hoặc vài hidden state; chưa tận dụng **attention matrix**, vốn được các nghiên cứu (Clark et al. 2019) cho là chứa nhiều tri thức ngôn ngữ (cú pháp, đồng tham chiếu).

Vấn đề thứ hai: vì pipeline BERT có hai pha (pre-train, fine-tune), distill chỉ ở một pha thì bỏ lỡ tri thức của pha kia.

Khái niệm cần nhớ:
- **Logit**: vector điểm số trước softmax. **Hidden state**: đầu ra của một tầng Transformer (ma trận $l \times d$, $l$ là độ dài chuỗi). **Attention matrix** $A=QK^\top/\sqrt{d_k}$ (chưa softmax): ma trận $l\times l$ cho mỗi head, cho biết token nào nhìn token nào.
- **Layer mapping** $g(m)$: quy tắc nói tầng $m$ của student bắt chước tầng nào của teacher.

## 2. Ý tưởng chính

Student (TinyBERT) có ít tầng và hidden size nhỏ hơn teacher. Ta ép student khớp teacher ở **mọi chỗ**: embedding, attention từng head, hidden state từng tầng được ánh xạ, và logit cuối. Vì hidden size khác nhau, dùng một ma trận tuyến tính học được để chiếu hidden của student lên chiều của teacher. Ngoài ra, distill hai lần: (1) pre-train "general" với BERT gốc làm teacher trên Wikipedia, (2) task-specific với BERT đã fine-tune làm teacher trên dữ liệu đã được **data augmentation** (thay từ bằng BERT/GloVe). Kết quả: student 4 tầng đạt ≥96.8% điểm GLUE của BERT-base, nhỏ hơn 7.5x và nhanh hơn 9.4x.

## 3. Phương pháp chi tiết

### 3.1 Bài toán và layer mapping

Student có $M$ tầng, teacher có $N$ tầng. Đặt chỉ số 0 là embedding, $M+1$ là prediction layer, và ép $g(0)=0$, $g(M+1)=N+1$. Hàm mục tiêu tổng:

$$
\boxed{\mathcal L_{model}=\sum_{x\in\mathcal X}\sum_{m=0}^{M+1}\lambda_m\,\mathcal L_{layer}\big(f^S_m(x),\,f^T_{g(m)}(x)\big)}
$$

Trong thí nghiệm mọi $\lambda_m=1$ và $g(m)=3m$ ("uniform": student 4 tầng học từ tầng 3, 6, 9, 12 của BERT-base).

### 3.2 Bốn loại loss

$$
\mathcal L_{attn}=\frac1h\sum_{i=1}^{h}\mathrm{MSE}(A_i^S,A_i^T),\qquad
\mathcal L_{hidn}=\mathrm{MSE}(H^S W_h,\,H^T)
$$

$$
\mathcal L_{embd}=\mathrm{MSE}(E^S W_e,\,E^T),\qquad
\mathcal L_{pred}=\mathrm{CE}(z^T/t,\ z^S/t)\quad(t=1)
$$

- $A_i\in\mathbb R^{l\times l}$ là attention của head $i$; **dùng bản chưa softmax** vì paper thấy hội tụ nhanh hơn và kết quả tốt hơn.
- $W_h\in\mathbb R^{d'\times d}$, $W_e$: ma trận chiếu học được, đưa hidden/embedding của student ($d'=312$) lên chiều teacher ($d=768$). Student và teacher có cùng số head $h=12$ nên attention so được trực tiếp.
- Loss mỗi tầng:

$$
\mathcal L_{layer}=\begin{cases}\mathcal L_{embd} & m=0\\ \mathcal L_{hidn}+\mathcal L_{attn} & 1\le m\le M\\ \mathcal L_{pred} & m=M+1\end{cases}
$$

### 3.3 Hai giai đoạn học

1. **General distillation**: teacher = BERT-base gốc (chưa fine-tune), dữ liệu = English Wikipedia (2.500M từ), 3 epoch, **không có** prediction loss (tác giả thử và không thêm lợi ích). Cho ra "general TinyBERT".
2. **Task-specific distillation**: teacher = BERT đã fine-tune cho tác vụ. Bước 1: distill tầng trung gian (embedding + Transformer layer) trên dữ liệu đã augment, 20 epoch (10 epoch cho MNLI/QQP/QNLI, 50 cho CoLA), batch 32, lr 5e-5. Bước 2: distill prediction layer 3 epoch.

### 3.4 Data augmentation (Algorithm 1)

```
Input: câu x; p_t=0.4, N_a=20, K=15
D' = []
repeat N_a lần:
    x_m = x
    for mỗi từ x[i]:
        if x[i] là từ 1 word-piece:
            thay x_m[i] bằng [MASK]; C = K từ có xác suất cao nhất theo BERT(x_m)[i]
        else:
            C = K từ gần nhất theo GloVe
        p ~ U(0,1)
        if p <= p_t: x_m[i] = một từ ngẫu nhiên trong C
    D'.append(x_m)
return D'
```

### 3.5 Ví dụ nhỏ

Student 4 tầng, teacher 12 tầng. Cặp (student, teacher): embedding-embedding; tầng 1-3, 2-6, 3-9, 4-12; prediction-prediction. Mỗi cặp tầng có loss $\mathrm{MSE}$ attention (12 head) cộng $\mathrm{MSE}$ hidden (sau chiếu 312 sang 768).

## 4. Thí nghiệm và kết quả chính

Teacher BERT-base ($N=12$, $d=768$, FFN 3072, $h=12$, 109M). TinyBERT$_4$: $M=4$, $d'=312$, FFN 1200, $h=12$, **14.5M** tham số. TinyBERT$_6$: $M=6$, $d'=768$, FFN 3072, 67.0M. Đánh giá trên tập test GLUE chính thức (Table 1, trích gọn, số "Avg" như trong paper):

| Mô hình | #Param | FLOPs | Tăng tốc | Avg GLUE |
|---|---|---|---|---|
| BERT-base (teacher) | 109M | 22.5B | 1.0x | 79.5 |
| BERT$_{TINY}$ (pre-train trực tiếp) | 14.5M | 1.2B | 9.4x | 70.2 |
| BERT$_4$-PKD | 52.2M | 7.6B | 3.0x | 72.6 |
| DistilBERT$_4$ | 52.2M | 7.6B | 3.0x | 71.9 |
| MobileBERT$_{TINY}$ (24 tầng) | 15.1M | 3.1B | - | 77.0 |
| **TinyBERT$_4$** | 14.5M | 1.2B | 9.4x | **77.0** |
| DistilBERT$_6$ | 67.0M | 11.3B | 2.0x | 76.8 |
| **TinyBERT$_6$** | 67.0M | 11.3B | 2.0x | **79.4** |

Một vài con số khác: TinyBERT$_4$ hơn BERT$_{TINY}$ 6.8% (trung bình), hơn BERT$_4$-PKD và DistilBERT$_4$ ít nhất 4.4% với ~28% tham số và 3.1x suy luận nhanh hơn. TinyBERT$_6$ hơn các baseline cùng kiến trúc 2.6% trung bình. Tốc độ đo trên một GPU K80.

SQuAD (dev, phụ lục B): BERT-base teacher EM/F1 80.7/88.4 (v1.1), 74.5/77.7 (v2.0). TinyBERT$_4$: 72.7/82.1 (v1.1), 68.2/71.8 (v2.0). TinyBERT$_6$: 79.7/87.5 (v1.1), 74.7/77.7 (v2.0).

## 5. Phân tích / ablation

**Bảng 2: các thủ tục học** (dev, TinyBERT$_4$, Avg trên MNLI-m, MNLI-mm, MRPC, CoLA):

| Biến thể | Avg |
|---|---|
| Đầy đủ | 75.6 |
| bỏ General Distillation (GD) | 72.5 |
| bỏ Task-specific Distillation (TD) | 68.5 |
| bỏ Data Augmentation (DA) | 68.4 |

Cả ba đều quan trọng; TD và DA ảnh hưởng ngang nhau và lớn hơn GD. GD giúp CoLA nhiều hơn MNLI/MRPC (tác giả phỏng đoán do khả năng khái quát ngôn ngữ).

**Bảng 3: các loss** (Avg như trên):

| Biến thể | Avg |
|---|---|
| Đầy đủ | 75.6 |
| bỏ Embd | 74.1 |
| bỏ Pred (dùng CE với nhãn thật) | 73.5 |
| bỏ Trm (cả Attn+Hidn) | 56.3 |
| bỏ Attn | 71.0 |
| bỏ Hidn | 72.9 |

Attention quan trọng hơn hidden state, hai loại bổ trợ nhau. "bỏ Trm" sập mạnh vì ở giai đoạn pre-train, các tầng Transformer của student không nhận tín hiệu nào từ tầng trên nên khởi tạo rất tệ (giải thích của tác giả).

**Bảng 4: layer mapping** (Avg): Uniform 75.6; Top ($g(m)=m+N-M$) 70.9; Bottom ($g(m)=m$) 71.3. Top tốt hơn Bottom trên MNLI, ngược lại trên MRPC và CoLA, nghĩa là tác vụ khác nhau dựa vào tầng khác nhau; uniform phủ cả dải nên thắng ở mọi tác vụ. Tác giả để ngỏ việc chọn tầng thích nghi theo tác vụ.

## 6. Giới hạn

*Do tác giả nêu:* chỉ thử teacher BERT-base (hướng tương lai: teacher rộng/sâu hơn như BERT-large); chưa kết hợp với quantization/pruning; chọn mapping thích nghi theo tác vụ còn bỏ ngỏ.

*Chúng ta nhận thấy:*
- Chi phí huấn luyện cao: 20 epoch trên dữ liệu augment ($N_a=20$ mẫu/câu) cho mỗi tác vụ, tức là phải distill lại cho từng task. MiniLM (19) so sánh trực tiếp ở dạng task-agnostic và cho thấy cách này nhẹ hơn.
- Ràng buộc: số head student = số head teacher, và cần ma trận chiếu $W_h, W_e$ (tham số thêm chỉ dùng khi train).
- Toàn bộ thực nghiệm là NLU trên văn bản; không có chứng cứ cho sinh chuỗi, thị giác hoặc multimodal.
- Dùng chuỗi ngắn (độ dài 64/128): loss attention $l\times l$ chưa được kiểm tra với chuỗi dài.

## 7. Liên hệ với các bài khác

- **03 FitNets**: TinyBERT là FitNets (hint ở tầng giữa, có regressor tuyến tính) áp dụng có hệ thống cho Transformer, cộng thêm loss attention.
- **04 Attention transfer**: cùng triết lý "khớp bản đồ attention", ở đây áp dụng cho từng head của self-attention.
- **16 DistilBERT, 17 Patient KD**: baseline trực tiếp; TinyBERT hơn nhờ khớp nhiều loại biểu diễn và hai giai đoạn.
- **19 MiniLM**: đối lập: chỉ distill tầng cuối, tránh bài toán layer mapping, hidden size student tùy ý. MiniLM tự so sánh với TinyBERT và cho thấy "last layer" thắng "layer-to-layer" trong thiết lập của họ.
- **20 Causal distillation**: tranh luận cùng chủ đề: có nên ép student khớp *mọi* khía cạnh trạng thái trong của teacher hay chỉ phần có vai trò nhân quả?
- **08 Teacher assistant**: MiniLM thêm TA; TinyBERT không dùng.

## 8. Ứng dụng cho dự án (suy luận)

- **Phần quan sát nên rút ra:** Với student là bản pruned của chính teacher (cùng họ kiến trúc), layer mapping gần như cho sẵn: tầng nào còn lại thì khớp tầng tương ứng của teacher 4B. Bảng 4 gợi ý rằng khớp "đều từ dưới lên trên" an toàn hơn chỉ khớp phần trên hoặc phần dưới.
- **Liên hệ D_lin / D_res:** loss hidden $\mathrm{MSE}(H^S W_h, H^T)$ với $W_h$ học được thực chất *tự tìm phép ánh xạ tuyến tính tối ưu* ở vị trí cắt, chính là ý niệm của D_lin. Phần dư mà $W_h$ không bù được chính là D_res. Gợi ý: sau khi cắt, đo MSE hidden có/không có $W_h$ để tách hai phần.
- **Neuron MLP vs tầng:** TinyBERT giảm FFN từ 3072 xuống 1200 và hidden từ 768 xuống 312 mà vẫn khôi phục tốt nhờ distill tầng giữa. Điều này không mâu thuẫn quan sát của bạn (cắt neuron khó hồi phục hơn cắt tầng) nhưng cũng không giải thích nó; TinyBERT khởi tạo student bằng general distillation rồi mới học, không cắt từ teacher. Đừng suy ra "cắt neuron sẽ ổn nếu thêm hidden loss" mà chưa thử.
- **Decoder GatedDeltaNet:** loss attention của TinyBERT dựa trên ma trận attention $l\times l$; tầng linear-attention không có ma trận này một cách tường minh, nên chỉ nên dùng loss hidden state cho các tầng đó. Đối với các tầng full-attention còn lại thì có thể thử loss attention (chi phí $l^2$ với chuỗi OCR dài cần cân nhắc).
- **Pha 1/pha 2:** dữ liệu augment kiểu "thay từ" không dùng được cho ảnh; với OCR, tương đương có thể là augment ảnh (nhiễu, độ phân giải), điều này là ý tưởng của chúng ta, không phải của paper.

## 9. Thí nghiệm gợi ý

1. **Ablation loss trên student đã cắt:** cùng ngân sách, so sánh chỉ KL logit, KL + hidden MSE có $W_h$, KL + hidden MSE + attention MSE (chỉ tầng full-attention). Báo cáo từng loại ở độ phân giải D_lin/D_res.
2. **Layer mapping cho model đã cắt tầng:** thử uniform vs top vs bottom (theo Bảng 4) cho việc chọn tầng teacher mà tầng student bắt chước, khi student giữ 6/12 tầng.
3. **Có/không có $W_h$:** với cắt neuron (không đổi hidden size), so khớp trực tiếp $H^S\approx H^T$ với khớp qua $W_h$ khởi tạo bằng đồng nhất, xem phần nào của suy giảm là tuyến tính.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| $g(m)$ | Ánh xạ tầng student $m$ sang tầng teacher; paper dùng $g(m)=3m$ |
| $\mathcal L_{attn}$ | MSE giữa attention chưa softmax, trung bình theo head |
| $\mathcal L_{hidn}$ | MSE giữa $H^S W_h$ và $H^T$ |
| $\mathcal L_{embd}$ | MSE giữa $E^S W_e$ và $E^T$ |
| $\mathcal L_{pred}$ | Cross-entropy mềm giữa logit, $t=1$ |
| General distillation | Teacher = BERT gốc, dữ liệu Wikipedia, không dùng $\mathcal L_{pred}$ |
| Task-specific distillation | Teacher = BERT fine-tuned, dữ liệu augment, trung gian rồi prediction |
| Data augmentation | Thay từ bằng BERT (từ 1 piece) hoặc GloVe (nhiều piece), $p_t=0.4$, $N_a=20$, $K=15$ |
| Kết quả | TinyBERT$_4$: 14.5M, 9.4x nhanh, Avg 77.0; TinyBERT$_6$: Avg 79.4 vs teacher 79.5 |

## 11. Kết luận ngắn

TinyBERT cho thấy student nhỏ có thể được kéo gần teacher đáng kể nếu khớp *cả* attention lẫn hidden state ở mọi tầng được ánh xạ, và nếu distill ở cả pha pre-train lẫn pha tác vụ. Bài học thực dụng: attention loss là phần đóng góp lớn nhất trong ablation; layer mapping đều (uniform) an toàn; và hidden loss cần một phép chiếu học được khi hidden size khác nhau. Cái giá là phải distill lại cho từng tác vụ và cần dữ liệu augment, nên MiniLM (19) là bước tiếp theo tự nhiên.

Link gốc: https://arxiv.org/abs/1909.10351
