# Structured Pruning Learns Compact and Accurate Models (2022)

**Tác giả:** Mengzhou Xia, Zexuan Zhong, Danqi Chen (Princeton University).  
**Venue / arXiv:** arXiv:2204.00408 (v3, 2/5/2022). Paper trong file không ghi tên hội nghị. Phương pháp thường được gọi là **CoFi Pruning** (Coarse- and Fine-grained Pruning).  
**Nguồn:** https://arxiv.org/abs/2204.00408  
**Chủ đề:** Structured pruning task-specific cho BERT: mask nhiều mức hạt (layer, head, neuron FFN, chiều ẩn) cùng học bằng $L_0$ Hard Concrete + distillation theo layer với ánh xạ layer động.  
**Vị trí trong lộ trình:** Cần đọc trước: 14 (Voita, cổng $L_0$ Hard Concrete), 16 (Movement pruning, baseline không cấu trúc), 13 (cắt head), 12 (cách đọc thí nghiệm). Bài này là **điểm hội tụ** của nhánh "cắt cấu trúc cho Transformer": mask nhiều tầng + distillation trong lúc cắt. Kế tiếp: 18 (2:4 sparsity, phần cứng), 25 ZipLM, 26 LLM-Pruner, 38 Sheared LLaMA (kế thừa trực tiếp ý mask $L_0$ đa tầng cho LLM).

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

Hai họ nén mô hình ngôn ngữ cho tới 2022:

- **Pruning**: bắt đầu từ mô hình lớn đã train, loại dần tham số, thường ra mô hình **task-specific** (riêng cho từng tác vụ). Unstructured pruning (như bài 16) đạt độ thưa rất cao (tới 97%) nhưng khó tăng tốc thật. Structured pruning (cắt layer, head, chiều trung gian FFN, block ma trận) phù hợp phần cứng hơn nhưng trước CoFi **hiếm khi tăng tốc quá 2-3 lần**.
- **Distillation**: chọn trước kiến trúc student nhỏ (DistilBERT, TinyBERT, MobileBERT...), khởi tạo ngẫu nhiên, chưng cất từ teacher trên **corpus không nhãn khổng lồ** ("general distillation") rồi mới chưng cất/fine-tune theo tác vụ. Đạt tăng tốc lớn nhưng cực đắt: TinyBERT cần 3.5 ngày trên 4 GPU cho 2,500M token × 3 epoch trước khi làm bước theo tác vụ (Figure 1).

Mục tiêu của CoFi: đạt **tăng tốc >10× với độ chính xác tương đương distillation** mà **không cần dữ liệu không nhãn** và với chi phí huấn luyện nhỏ.

Thuật ngữ:
- **Task-specific vs task-agnostic**: mô hình dành cho một tác vụ (đã biết nhãn) hay mô hình đa dụng.
- **Mask variable $z$**: biến (0 hay 1, hoặc [0,1] khi huấn luyện) nhân vào một đơn vị để bật/tắt đơn vị đó.
- **FFN intermediate dimension**: số neuron của lớp ẩn trong khối MLP ($d_f=4d$ thường).
- **Hidden dimension**: chiều của residual stream ($d=768$).
- **Speedup**: ở paper, thời gian suy luận của BERT-base chia thời gian của mô hình cắt, đo trên một GPU V100, batch 128.

## 2. Ý tưởng chính

Thay vì cắt một loại đơn vị (chỉ head, hoặc chỉ neuron FFN), CoFi gắn **nhiều mask ở các mức hạt khác nhau** lên cùng một tham số: mask cho cả lớp MHA ($z_{MHA}$), cả lớp FFN ($z_{FFN}$), từng head ($z_{head}$), từng neuron trung gian ($z_{int}$) và từng chiều ẩn ($z_{hidn}$, dùng chung mọi layer). Một trọng số bị cắt nếu *bất kỳ* mask nào bao nó bằng 0. Mask mịn cho tính linh hoạt, mask thô cho phép bỏ cả layer, điều quan trọng để tăng tốc thật ở độ thưa cao. Các mask được học bằng $L_0$ Hard Concrete (như bài 14) với ràng buộc độ thưa chính xác bằng Lagrangian. Bên cạnh đó, vì kiến trúc student chưa biết trước, họ dùng **layerwise distillation với ánh xạ layer động**: mỗi layer của teacher được ghép với layer student gần nhất (theo MSE) ở thời điểm hiện tại.

## 3. Phương pháp chi tiết

### 3.1 Kiến trúc Transformer và các đơn vị có thể cắt

Block $= $ MHA + FFN (có residual và LayerNorm). MHA với $N_h$ head, $W_Q^{(i)},W_K^{(i)},W_V^{(i)},W_O^{(i)}\in\mathbb R^{d\times d_h}$, $d_h=d/N_h$. FFN: $\mathrm{FFN}(X)=\mathrm{gelu}(XW_U)\,W_D$ với $W_U\in\mathbb R^{d\times d_f}$, $W_D\in\mathbb R^{d_f\times d}$. MHA chiếm khoảng 1/3 tham số, FFN 2/3 (không tính embedding). Theo Ganesh et al. (2021) trích trong paper: MHA và FFN tốn thời gian xấp xỉ nhau trên GPU, còn FFN là nút cổ chai trên CPU.

### 3.2 Năm loại mask

$$\boxed{\mathrm{MHA}(X)=z_{MHA}\sum_{i=1}^{N_h} z^{(i)}_{head}\,\mathrm{Att}\big(W_Q^{(i)},W_K^{(i)},W_V^{(i)},W_O^{(i)},X\big)}$$

$$\boxed{\mathrm{FFN}(X)=z_{FFN}\cdot \mathrm{gelu}(XW_U)\cdot \mathrm{diag}(z_{int})\cdot W_D}$$

- $z_{head}^{(i)}\in\{0,1\}$: từng head. $z_{int}\in\{0,1\}^{d_f}$: từng neuron trung gian của FFN.
- $z_{MHA}^{(l)}, z_{FFN}^{(l)}\in\{0,1\}$: từng lớp MHA / FFN, **tách riêng nhau** (khác Fan 2020, Sajjad 2020 cắt cả block).
- $z_{hidn}\in\{0,1\}^d$: chiều ẩn, **dùng chung mọi layer** vì mỗi chiều nối với cùng chiều đó ở layer sau qua residual; áp lên mọi ma trận (ví dụ $\mathrm{diag}(z_{hidn})W_Q$). Thực nghiệm chỉ một số ít chiều bị cắt (ví dụ 768 → 760) nhưng vẫn cải thiện đáng kể.

Điểm khác biệt chính: **nhiều mask cùng điều khiển một tham số**. Ví dụ một trọng số của FFN bị cắt nếu cả lớp FFN, hoặc neuron trung gian tương ứng, hoặc chiều ẩn tương ứng bị cắt. Block Pruning (Lagunas 2021) so sánh chỉ dùng một chiến lược mỗi loại lớp.

*Vì sao cần mask thô?* Cắt hết $N_h=12$ head tương đương cắt cả lớp MHA, nhưng tác giả quan sát rằng điều này **hiếm khi tự xảy ra** nếu chỉ có mask mịn, gây khó tối ưu ở độ thưa cao. Mask thô tạo đường tắt.

### 3.3 Học mask: $L_0$ Hard Concrete + Lagrangian

Mỗi mask là biến Hard Concrete (Louizos 2018; xem bài 14, 16): với $u\sim U(0,1)$,

$$s=\mathrm{sigmoid}\Big(\tfrac1\beta\big(\log\tfrac{u}{1-u}+\log\alpha\big)\Big),\quad \tilde s=s\,(r-l)+l,\quad z=\min(1,\max(0,\tilde s))$$

với $l<0$, $r>0$ kéo giãn đầu ra của sigmoid; $\log\alpha$ là tham số học được chính; $\beta$ điều chỉnh độ dốc. Khi huấn luyện mask nhận giá trị thực trong $[0,1]$; lúc suy luận, mask nhỏ hơn ngưỡng đặt về 0 (ngưỡng xác định theo độ thưa kỳ vọng của từng ma trận; chi tiết ở Appendix B).

Độ thưa kỳ vọng $\hat s$ tính trên kích thước mô hình đầy đủ $M$ (nhân các mask liên quan, hệ số 4 cho Q,K,V,O và 2 cho $W_U,W_D$):

$$\hat s=\frac1M\sum_{l}\sum_{i}\sum_{k}z^{(l)}_{MHA}z^{(l,i)}_{head}z^{(k)}_{hidn}\cdot 4d_h+\frac1M\sum_{l}\sum_{j}\sum_{k}z^{(l)}_{FFN}z^{(l,j)}_{int}z^{(k)}_{hidn}\cdot 2$$

(Công thức trong paper là dạng này; đây là cách viết đã được làm gọn: xem Mục 3.1 của paper cho dấu hiệu chính xác.) Vì chỉ phạt $\lambda\|z\|_0$ thì kích thước cuối phụ thuộc lịch và learning rate, nên dùng **Lagrangian** (theo Wang 2020b) để ép $\hat s=t$ (độ thưa mục tiêu):

$$\boxed{\mathcal L_c=\lambda_1(\hat s-t)+\lambda_2(\hat s-t)^2}$$

Tác giả cũng thử straight-through estimator (như bài 16) và thấy hiệu năng ngang; chọn $L_0$ vì dễ điều khiển độ thưa chính xác.

### 3.4 Layerwise distillation với ánh xạ layer động

Mục tiêu dự đoán (KL giữa phân phối student $p_s$ và teacher $p_t$):

$$\mathcal L_{pred}=D_{KL}(p_s\,\|\,p_t)$$

Với student chưa biết kiến trúc, định nghĩa trước ánh xạ layer cố định như TinyBERT là không thể. Gọi $\mathcal T$ là tập layer teacher dùng để chưng cất, $m(i)$ là layer student ghép với layer teacher $i$:

$$\boxed{\mathcal L_{layer}=\sum_{i\in\mathcal T}\mathrm{MSE}\big(W_{layer}H_s^{m(i)},\,H_t^i\big),\qquad m(i)=\arg\min_{j:\,z^{(j)}_{FFN}>0}\mathrm{MSE}\big(W_{layer}H_s^{j},H_t^i\big)}$$

- $H$: biểu diễn ẩn **ở các layer FFN** (đầu ra FFN); $W_{layer}\in\mathbb R^{d\times d}$ là ma trận tuyến tính học được, **khởi tạo là ma trận đơn vị**.
- Ánh xạ chọn layer student còn sống gần nhất (theo MSE) với mỗi layer teacher; tính khoảng cách song song nên chi phí nhỏ.
- **Ràng buộc chống "layer mismatch"** (xảy ra với dataset nhỏ như RTE, MRPC): chỉ cho ghép layer teacher tới layer student **cao hơn hoặc bằng layer đã ghép trước đó** (đơn điệu). Với dataset lớn, mismatch hiếm.

Tổng hợp:

$$\boxed{\mathcal L_{distil}=\lambda\,\mathcal L_{pred}+(1-\lambda)\,\mathcal L_{layer}}$$

Toàn bộ mục tiêu huấn luyện: $\mathcal L_{distil}$ + $\mathcal L_c$ (độ thưa). Teacher là BERT-base đã fine-tune trên tác vụ; student khởi đầu từ chính nó.

### 3.5 Quy trình và pseudo-code

```text
Input: BERT-base đã fine-tune (teacher = student ban đầu), độ thưa mục tiêu t,
       dữ liệu tác vụ (GLUE/SQuAD)
Gắn mask Hard Concrete: z_MHA, z_FFN (mỗi layer), z_head, z_int, z_hidn
Embedding bị đóng băng
Pha A:  fine-tune 1 epoch với L_distil                    (chưa cắt)
Pha B:  tăng độ thưa tuyến tính tới t trong 2 epoch;       (cắt)
        mỗi bước tối ưu: L_distil + L_c (Lagrangian) cập nhật
        trọng số W, tham số mask log_alpha, lambda_1, lambda_2
        mỗi bước: m(i) = argmin_j MSE(W_layer H_s^j, H_t^i)  # ánh xạ layer động
        (tiếp tục huấn luyện/tìm cấu trúc đến hết 20 epoch)
Pha C:  loại hẳn đơn vị có mask = 0 (cấu trúc cố định),
        fine-tune mạng con 20 epoch với L_distil           (không mask nữa)
Output: mạng con dense, nhỏ
```

Chi tiết (Appendix A): bốn tập lớn (MNLI, QNLI, SST-2, QQP) và SQuAD: 20 epoch tổng cộng + 20 epoch fine-tune mạng con; tập GLUE nhỏ: 100 epoch tổng + 20 epoch fine-tune; với tập nhỏ có 4 epoch distill rồi cắt tới target trong 20 epoch kế. Siêu tham số: $\lambda\in\{0.1,0.3,0.5\}$, nhiệt độ distill $t=2$, lr fine-tune $\{1,2,3\}\times10^{-5}$, lr huấn luyện $2\times10^{-5}$ (GLUE), $3\times10^{-5}$ (SQuAD), batch 32 (GLUE), 16 (SQuAD). Tác giả ghi rằng "finetuning the final subnetwork is essential for high sparsity models" và sau khi đạt độ thưa mục tiêu quá trình vẫn tiếp tục tìm cấu trúc tốt hơn.

## 4. Thí nghiệm và kết quả chính

**Thiết lập.** BERT-base (và RoBERTa, ở Appendix I); 8 tác vụ GLUE (SST-2, MNLI, QQP, QNLI, MRPC, CoLA, STS-B, RTE) + SQuAD v1.1; kết quả trên dev set. Độ thưa mục tiêu $\{60,70,75,80,85,90,95\}\%$ (tính theo tham số, **không tính embedding**; embedding bị đóng băng). Baseline: DistilBERT$_6$, TinyBERT$_6$/TinyBERT$_4$, DynaBERT, Block Pruning (Lagunas 2021, bản "Hybrid Filled"), thêm FLOP, LayerDrop, Movement Pruning, MobileBERT, AutoTinyBERT (không trực tiếp so sánh được; ở Appendix F). Tăng tốc đo trên **một GPU NVIDIA V100**, input 128 token (GLUE) hoặc 384 (SQuAD), batch 128, so với BERT-base chưa cắt.

### 4.1 Bảng so sánh tổng (Table 1, MNLI; U = dùng dữ liệu không nhãn, T = task-specific)

| Phương pháp | U | T | Tăng tốc | Tham số | MNLI |
|---|---|---|---|---|---|
| BERT-base (teacher) | không | không | 1.0× | 85M | 84.8 |
| DistilBERT$_6$ | có | không | 2.0× | 43M | 82.2 |
| TinyBERT$_6$ | có | có | 2.0× | 43M | 84.0 |
| MobileBERT‡ | có | không | 2.3× | 20M | 83.9 |
| DynaBERT | không | có | 6.3× | 11M | 76.3 |
| AutoTinyBERT‡ | có | có | 9.1× | 3.3M | 78.2 |
| TinyBERT$_4$ | có | có | 11.4× | 4.7M | 78.8 |
| Movement Pruning | không | có | 1.0× | 9M | 81.2 |
| Block Pruning | không | có | 2.7× | 25M | 83.7 |
| **CoFi (ours)** | không | có | **2.7×** | 26M | **84.9** |
| **CoFi (ours)** | không | có | **12.1×** | 4.4M | **80.6** |

‡ dùng teacher khác, không so sánh trực tiếp. Lưu ý: Movement Pruning có 9M tham số **nhưng tăng tốc 1.0×** (thưa không cấu trúc) đúng như đã dự đoán ở bài 16.

### 4.2 CoFi vs TinyBERT$_4$ (Table 2, ~10× tăng tốc, ~5M tham số, ~95% thưa; không dùng data augmentation)

| | SST-2 | QNLI | MNLI | QQP | CoLA | RTE | STS-B | MRPC | SQuAD | Thời gian train (GPU-giờ) |
|---|---|---|---|---|---|---|---|---|---|---|
| BERT-base (teacher) | 93.1 | 91.5 | 84.8 | 91.2 | 61.2 | 70.0 | 88.7 | 85.0 | 88.4 | n/a |
| TinyBERT$_4$ w/o GD | 87.7 | 81.8 | 78.7 | 89.5 | 16.6 | 47.3 | 17.8 | 68.9 | n/a | ≤ 10 |
| TinyBERT$_4$ | 89.7 | 86.7 | 78.8 | 90.0 | 32.5 | 63.2 | 85.0 | 81.4 | 82.1 | ~350 |
| CoFi | 90.6 | 86.1 | 80.6 | 90.1 | 35.6 | 64.7 | 83.1 | 82.6 | 82.6 | ≤ 20 |

Tăng tốc: TinyBERT$_4$ 11.4× (SQuAD 8.7×); CoFi từ 11.0× đến 12.9× (SQuAD 8.7×). GD = general distillation. CoFi thắng hoặc ngang ở 6/9 cột (hơn ở SST-2, MNLI, QQP, CoLA, RTE, MRPC, SQuAD; kém ở QNLI và STS-B), với chi phí ≤ 20 GPU-giờ so với ~350.

Với data augmentation theo TinyBERT (Table 3; dữ liệu tăng cường lớn hơn 20×, nên chỉ chạy 4 tập; số trước → là không augment): TinyBERT$_4$: SST-2 89.7→91.6, QNLI 86.7→87.6, RTE 63.2→62.5, MRPC 81.4→83.6; CoFi: SST-2 90.6→92.4, QNLI 86.1→86.8, RTE 64.7→67.5, MRPC 82.6→84.6. (Tác giả tự chạy lại augment vì tập gốc không được công bố; kết quả tái hiện của TinyBERT$_4$ thấp hơn số công bố: ví dụ SST-2 91.6 so với 92.7.)

### 4.3 Đường cong (Figure 2)

CoFi cho mô hình chính xác hơn distillation và pruning baseline **ở mọi mức tăng tốc và kích thước** trên SST-2, QNLI, MNLI, SQuAD. Block Pruning không đạt tăng tốc như TinyBERT$_4$; DynaBERT chậm hơn vì chỉ cắt tối đa nửa số lớp MHA/FFN. Các đường cong này đọc từ hình; bảng số chi tiết không nằm trong text.

## 5. Phân tích / ablation

### 5.1 Đơn vị cắt (Table 4: % = tăng tốc; 60% = 34M, 95% = 5M tham số)

| | QNLI 60% | QNLI 95% | MNLI 60% | MNLI 95% | SQuAD 60% | SQuAD 95% |
|---|---|---|---|---|---|---|
| CoFi | 2.1× / 91.8 | 12.1× / 86.1 | 2.1× / 85.1 | 12.1× / 80.6 | 2.0× / 89.1 F1 | 8.7× / 82.6 F1 |
| −hidden | 2.2× / 91.3 | 13.3× / 85.6 | 2.1× / 85.2 | 13.7× / 79.8 | 2.0× / 88.7 | 9.7× / 80.8 |
| −layer & hidden | 2.2× / 91.3 | 7.2× / 84.6 | 2.1× / 84.8 | 7.0× / 78.4 | 2.1× / 88.5 | 6.4× / 74.1 |
| −layer | 2.1× / 91.5 | 8.3× / 86.7 | 2.1× / 85.4 | 8.4× / 80.6 | 2.0× / 89.1 | 7.9× / 80.5 |

- **Bỏ mask layer** ($z_{MHA},z_{FFN}$): ở độ thưa 95% tăng tốc tụt mạnh (12.1× → 8.3× QNLI; 12.1× → 8.4× MNLI) mà độ chính xác gần như không đổi/hơi cao hơn ở QNLI; ở 60% không ảnh hưởng. Tức **cùng số tham số, cấu hình khác nhau cho tốc độ khác hẳn**.
- **Bỏ mask hidden** ($z_{hidn}$): mô hình nhanh hơn chút (cắt thêm layer) nhưng độ chính xác giảm trên mọi tập.
- Kết luận: đặt mask ở nhiều mức khuyến khích tối ưu cắt đúng loại đơn vị để tối đa hóa hiệu năng dưới ràng buộc độ thưa.

### 5.2 Mục tiêu distillation (Table 5, độ thưa 95%)

| | SST-2 | QNLI | MNLI | SQuAD |
|---|---|---|---|---|
| CoFi | 90.6 | 86.1 | 80.6 | 82.6 |
| −$\mathcal L_{layer}$ | 91.1 | 85.1 | 79.7 | 82.5 |
| −$\mathcal L_{pred}$, −$\mathcal L_{layer}$ (không distill) | 86.6 | 84.2 | 78.2 | 75.8 |
| Fixed Hidden Distillation | 90.0 | 85.8 | 80.5 | 80.9 |

- Bỏ distillation hoàn toàn: giảm 1.9 đến 6.8 điểm trên các tập.
- Ánh xạ layer **động** hơn ánh xạ cố định; bỏ $\mathcal L_{layer}$ hại mọi tập trừ SST-2. Ánh xạ động hội tụ về một liên kết ổn định, ví dụ QNLI: layer teacher 3, 6, 9, 12 khớp layer student 1, 2, 4, 9.
- Lợi ích của ánh xạ động rõ hơn ở độ thưa thấp (Appendix G.2).

### 5.3 Cấu trúc mô hình được cắt (chạy 3 lần mỗi cấu hình; 5 tập; độ thưa 60-95%)

1. **FFN bị cắt nặng hơn MHA** ở mọi độ thưa. Ở 60%: số neuron trung gian trung bình giảm **71% (3072 → 884)**, số head trung bình giảm **39% (12 → 7.3)**. Gợi ý FFN dư thừa hơn MHA.
2. **Cắt nhiều hơn ở layer trên**: layer MHA trên còn ít head hơn layer dưới.
3. Ở 95%: các mô hình còn **nhiều lớp MHA hơn FFN** (Table 6), gợi ý MHA quan trọng hơn cho tác vụ. Layer MHA đầu và thứ hai thường được giữ, layer giữa thường bị bỏ.
4. Cấu trúc **khác nhau theo dataset** (ví dụ SST-2, QNLI giữ MHA đầu, QQP và SQuAD bỏ), tức tồn tại mạng con tối ưu riêng cho mỗi tác vụ. Mô hình cắt có thể có FFN/MHA liền kề (như Press et al. 2020 gợi ý có thể có lợi).

### 5.4 So sánh với Movement Pruning (Figure 4, Appendix)

CoFi (cấu trúc) vẫn vượt movement pruning (không cấu trúc) trên MNLI và SQuAD ở nhiều kích thước; "CoFi Logit Distill" (chỉ distill logits như movement pruning) cũng được vẽ để tách ảnh hưởng của loss layerwise (các giá trị chi tiết nằm ở hình, không đọc được từ text).

## 6. Giới hạn

**Do tác giả nêu:**
- Phạm vi giới hạn ở **task-specific** pruning; áp cho mô hình task-agnostic (upstream pruning) là "hướng hứa hẹn" nhưng phức tạp ở các lựa chọn thiết kế.
- Chi phí huấn luyện của CoFi hơi dài hơn task-specific distillation của TinyBERT vì vừa tìm cấu trúc vừa học tham số.
- Phép so sánh với MobileBERT, AutoTinyBERT không trực tiếp (teacher khác).
- Tăng tốc phụ thuộc nền tảng (chỉ một V100); số có thể khác bản gốc của các baseline.

**Do chúng tôi nhận thấy (suy luận):**
- Toàn bộ thí nghiệm là BERT-base trên GLUE/SQuAD: encoder-only, tác vụ phân loại/trích đoạn, đầu ra ngắn. Không có mô hình sinh (decoder), không có KV cache, không có đa phương thức.
- Mỗi tác vụ cần train riêng (20 GPU-giờ cho mỗi cặp tác vụ/độ thưa); chi phí nhân lên theo số cấu hình.
- Tăng tốc đo ở batch 128; ở batch 1 (yêu cầu của dự án) lợi ích có thể khác (cùng cảnh báo như Table 4 bài 13).
- Số liệu là một lần chạy cho Table 2 (cấu trúc thì chạy 3 lần), không có thanh sai số trên điểm số; chênh nhỏ như 86.1 vs 86.7 (QNLI) nên đọc thận trọng (checklist bài 12).
- $\mathcal L_{layer}$ chỉ tính ở các layer FFN và dùng $W_{layer}$ cho mọi ánh xạ; chưa rõ đây có là lựa chọn tối ưu.

## 7. Liên hệ với các bài khác

- **13 Sixteen Heads, 14 Voita**: mask head; CoFi dùng đúng $L_0$ Hard Concrete của Voita nhưng mở rộng sang 5 loại đơn vị.
- **16 Movement pruning**: baseline không cấu trúc; CoFi chứng minh cấu trúc + distillation đạt tăng tốc thật (12.1× vs 1.0× ở số tham số tương đương).
- **15 LTH BERT**: nhắc lại như một nhánh unstructured khác.
- **04-05 Pruning filters, Network slimming**: tiền thân cho CNN (mask/scale theo channel).
- **11 Rethinking the value of pruning**, **03 Zhu & Gupta**: bài này tiếp tục "cắt từ mô hình lớn rồi tinh chỉnh" thay vì train mô hình nhỏ từ đầu.
- **18 2:4 sparsity**: hướng thưa có cấu trúc phần cứng, khác với cắt cấu trúc dense của CoFi.
- **25 ZipLM, 26 LLM-Pruner, 28 FLAP**: pruning cấu trúc cho LLM, cùng triết lý chọn đơn vị theo ngân sách latency.
- **38 Sheared LLaMA**: áp mask $L_0$ đa tầng (layer, head, FFN, hidden) lên LLaMA kèm dynamic batch loading; kế thừa trực tiếp CoFi.
- **39-42 (Minitron, Flextron, Puzzle)**: cắt cấu trúc rồi distillation ở quy mô LLM; cùng công thức "mask/ngân sách + distill".
- **12 State of pruning**: đánh giá CoFi theo checklist: nhiều điểm hoạt động (7 mức thưa), nhiều tác vụ, đo speedup thật (tốt), nhưng một kiến trúc và một nền tảng GPU.

## 8. Ứng dụng cho dự án (suy luận)

Mức liên quan: **cao** (gần nhất với bài toán của bạn trong nhóm bài NLP trước 2023).

1. **Mask đa mức là khung tự nhiên cho dự án**: mapping sang LightOnOCR-3-0.8B:
   - $z_{layer}$ cho từng block decoder (lai GatedDeltaNet/full attention) và từng block ViT (12 block), tương ứng $z_{MHA}$/$z_{FFN}$ nhưng với token mixer là GatedDeltaNet hay attention;
   - $z_{head}$ cho head attention (và head state của GatedDeltaNet, nếu đơn vị đó cắt được);
   - $z_{int}$ cho neuron MLP;
   - $z_{hidn}$ cho chiều residual stream (chung mọi layer; cắt chiều ẩn đụng đến cả embedding, norm, và projector VLM nên khó hơn).
2. **Dùng ngân sách latency thay cho "độ thưa tham số"**: Table 4 là bằng chứng mạnh rằng *cùng số tham số, tốc độ khác nhau 7.2× đến 13.3×*. Với GPU ≤ 8 GB và batch-1, nên thay $\hat s$ bằng **latency kỳ vọng** (bảng tra latency theo cấu hình) trong Lagrangian. (Ý tưởng mở rộng, không có trong paper; HALP, bài 19, đi theo hướng này.)
3. **Mask thô giúp tối ưu ở mức nén cao**: quan sát của bạn "cắt MLP neuron hồi phục kém hơn cắt layer" khớp phần nào với Table 4: khi không có mask layer, tối ưu giữ lại nhiều đơn vị mịn rải khắp, tốc độ kém (7-8× thay vì 12×). Với 0.1B từ 0.8B, mức nén 8× nằm ở vùng mà mask thô trở nên quan trọng (giả thuyết cần kiểm).
4. **Ánh xạ layer động** hữu ích cho distillation từ teacher 4B: teacher khác kiến trúc/chiều sâu với student, cần quyết định ghép layer. Có thể dùng $\mathcal L_{layer}$ với $W_{layer}$ (khởi tạo đơn vị) để căn chỉnh khác chiều (teacher d lớn hơn student: $W_{layer}$ cần là ma trận chữ nhật, một điều chỉnh không có trong paper).
5. **Giai đoạn: distill trước → cắt dần tuyến tính → fine-tune mạng cố định** là công thức quen thuộc, và "finetuning the final subnetwork is essential for high sparsity" hợp với kế hoạch phục hồi bằng distillation.
6. **Quan sát cấu trúc**: FFN dư thừa nhiều hơn MHA, layer trên bị cắt nhiều hơn, mô hình cuối có nhiều MHA hơn FFN. Có thể là điểm đối chiếu với hồ sơ $D_{lin}/D_{res}$ của bạn: kiểm tra liệu FFN/MLP cũng cho $D_{lin}$ cao ở layer trên.
7. **Cảnh báo**: BERT encoder 85M ≠ VLM sinh văn bản 0.8B. Mask $L_0$ học đồng thời vài nghìn biến có thể rất nhạy với dữ liệu hiệu chuẩn; OCR đòi hỏi đầu ra dài, lỗi lan truyền theo token (autoregressive), nên độ chịu cắt có thể thấp hơn nhiều so với phân loại câu.

## 9. Thí nghiệm gợi ý

1. **Mask đa mức thu nhỏ trên ViT 12 block**: gắn $z_{layer}$ (MHA/FFN), $z_{head}$, $z_{int}$ cho ViT, decoder đóng băng, loss = distill từ teacher (4B hoặc ViT gốc) + Lagrangian về ngân sách tham số. Chạy với và không có mask layer; so tốc độ thực (batch-1, GPU đích) tại cùng số tham số, tái hiện Table 4.
2. **Latency thay vì độ thưa**: đo bảng latency cho khối decoder theo (số head, số neuron MLP) trên GPU đích; thay $\hat s$ bằng latency kỳ vọng và xem cấu trúc cuối có đổi không (nhiều layer bị bỏ hơn?).
3. **Ánh xạ layer động vs cố định** cho distillation 4B → ~0.3B: so $\mathcal L_{layer}$ với ánh xạ động (argmin MSE, đơn điệu) và ánh xạ đều cố định; theo dõi liên kết hội tụ, loss tổng và CER.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| MHA có mask | $z_{MHA}\sum_i z^{(i)}_{head}\mathrm{Att}_i$ |
| FFN có mask | $z_{FFN}\,\mathrm{gelu}(XW_U)\,\mathrm{diag}(z_{int})\,W_D$ |
| Mask hidden | $z_{hidn}\in\{0,1\}^d$, chung mọi layer, áp lên mọi ma trận |
| Hard Concrete | $s=\mathrm{sigmoid}((\log\frac u{1-u}+\log\alpha)/\beta)$, $z=\min(1,\max(0,s(r-l)+l))$ |
| Ràng buộc độ thưa | $\mathcal L_c=\lambda_1(\hat s-t)+\lambda_2(\hat s-t)^2$ |
| Pred distillation | $D_{KL}(p_s\|p_t)$ |
| Layer distillation | $\sum_{i\in\mathcal T}\mathrm{MSE}(W_{layer}H_s^{m(i)},H_t^i)$, $m(i)=\arg\min_{j:z^{(j)}_{FFN}>0}\mathrm{MSE}(W_{layer}H_s^j,H_t^i)$ |
| Tổng | $\lambda\mathcal L_{pred}+(1-\lambda)\mathcal L_{layer}$ |
| Kết quả chính | MNLI: 2.7× / 84.9 (26M), 12.1× / 80.6 (4.4M); vs TinyBERT$_4$ 11.4× / 78.8 |
| Chi phí | ≤ 20 GPU-giờ (CoFi) vs ~350 (TinyBERT$_4$) |

## 11. Kết luận ngắn

CoFi cho thấy cắt cấu trúc kết hợp mask ở nhiều mức hạt (layer, head, neuron FFN, chiều ẩn) cùng ràng buộc độ thưa Lagrangian và distillation layerwise động đạt >10× tăng tốc với độ chính xác ngang hoặc hơn distillation (TinyBERT$_4$), mà không cần dữ liệu không nhãn và chỉ ≤ 20 GPU-giờ. Hai bài học để mang về dự án: (1) cùng số tham số, cấu hình khác nhau cho tốc độ khác nhau cả chục lần, nên tối ưu theo latency và cho phép cắt cả layer; (2) khi kiến trúc student chưa biết, ánh xạ layer cho distillation phải được học động. Cần thận trọng khi chuyển từ BERT-base/GLUE sang VLM sinh văn bản autoregressive.

Link gốc: https://arxiv.org/abs/2204.00408
