# FitNets: Hints for Thin Deep Nets (2015)

**Tác giả:** Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta, Yoshua Bengio  
**Venue / arXiv:** arXiv:1412.6550 (v4, 27/3/2015); ICLR 2015 (ghi trên file)  
**Nguồn:** https://arxiv.org/abs/1412.6550  
**Chủ đề:** Feature-based / hint-based distillation: dùng đầu ra hidden layer của teacher làm "gợi ý" để huấn luyện student sâu hơn và mảnh hơn; huấn luyện hai giai đoạn.  
**Vị trí trong lộ trình:** Đọc sau 02 (Hinton KD). Bài này là bài đầu tiên của nhóm "feature distillation": thêm tín hiệu ở hidden layer, dùng regressor (projector) để khớp chiều. Tiếp theo: 04 (Attention transfer: khớp bản đồ attention thay vì feature thô), 12 (Relational KD), 13 (CRD).

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch paper. Phần "Ứng dụng cho dự án" và "Thí nghiệm gợi ý" là suy luận của chúng ta.

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

Các phương pháp nén trước đó (bài 01, 02) cho student *nông và rộng* hoặc *cùng độ sâu* với teacher. Tác giả lập luận độ sâu là yếu tố quan trọng của representation learning (tái sử dụng feature, biểu diễn trừu tượng hơn), và thực nghiệm cho thấy mạng rất sâu (19, 22 lớp) thắng ImageNet. Nhưng mạng sâu và mảnh (thin) rất **khó tối ưu**: nhiều phi tuyến nối tiếp tạo hàm không lồi cao.

Các cách hỗ trợ tối ưu đã biết mà paper nhắc: pretraining theo từng lớp (greedy layer-wise), thêm giám sát vào lớp trung gian (deeply-supervised nets), curriculum learning (học từ dễ đến khó).

Ý tưởng của bài: **dùng teacher để cho hint ở lớp giữa**, nhằm (i) cho phép student sâu hơn teacher, (ii) mảnh hơn nên ít tham số và nhanh hơn.

**Thuật ngữ:**
- **Hint layer:** hidden layer của teacher dùng làm mục tiêu.
- **Guided layer:** hidden layer của student được dẫn dắt để dự đoán hint.
- **Regressor:** ánh xạ nhỏ gắn trên guided layer để đưa về cùng chiều với hint (ngày nay gọi là projector/adapter).
- **Thin / wide:** ít / nhiều kênh mỗi lớp.

## 2. Ý tưởng chính

Student (FitNet) sâu hơn nhưng mảnh hơn teacher. Giai đoạn 1: chọn một hidden layer ở giữa teacher (hint) và một hidden layer ở giữa student (guided); huấn luyện phần mạng student từ đầu tới guided layer, cộng một regressor, để dự đoán đầu ra của hint (loss L2). Giai đoạn 2: dùng trọng số đó làm khởi tạo rồi huấn luyện toàn mạng student bằng KD (Hinton) với $\lambda$ giảm dần. Paper cho thấy hint là một dạng khởi tạo tốt + regularizer, giúp huấn luyện được mạng mà backprop thường và KD thuần không huấn luyện nổi.

## 3. Phương pháp chi tiết

### 3.1 Tóm tắt KD dùng trong paper

Với $P_T=\mathrm{softmax}(a_T)$, $P_S=\mathrm{softmax}(a_S)$ và nới nhiệt độ $\tau>1$:

$$
P_T^\tau=\mathrm{softmax}(a_T/\tau),\quad P_S^\tau=\mathrm{softmax}(a_S/\tau)
$$

$$
\boxed{\mathcal L_{KD}(W_S)=H(y_{true},P_S)+\lambda\,H(P_T^\tau,P_S^\tau)}
$$

với $H$ là cross-entropy, $\lambda$ cân bằng hai số hạng. (Khác Hinton ở chỗ không thấy nhân $\tau^2$ trong công thức này; paper đặt $\tau=3$.)

### 3.2 Hint-based training

Hint $u_h(x;W_{Hint})$ là đầu ra của teacher đến lớp $h$; student có $v_g(x;W_{Guided})$ đến lớp $g$; regressor $r(\cdot;W_r)$. Loss:

$$
\boxed{\mathcal L_{HT}(W_{Guided},W_r)=\tfrac12\left\|u_h(x;W_{Hint})-r\big(v_g(x;W_{Guided});W_r\big)\right\|^2}
$$

Lưu ý của paper: $u_h$ và $r$ phải "so sánh được", cùng loại phi tuyến.

**Chọn lớp:** hint càng sâu thì student càng ít tự do, dễ bị over-regularize. Paper chọn **lớp giữa của teacher** làm hint và **lớp giữa của student** làm guided (tuy trong thí nghiệm cụ thể, ví dụ guided là lớp 11 mimic lớp 2 của teacher, nên "giữa" là xấp xỉ; các cặp được liệt kê ở bảng kiến trúc trong phụ lục).

**Regressor tích chập:** regressor fully-connected tốn $N_{h,1}N_{h,2}O_h N_{g,1}N_{g,2}O_g$ tham số. Paper dùng regressor *tích chập* với kernel $k_1\times k_2$ sao cho $N_{g,i}-k_i+1=N_{h,i}$, tức đầu ra cùng kích thước không gian với hint, chỉ tốn $k_1k_2 O_h O_g$ tham số.

### 3.3 Huấn luyện hai giai đoạn (Algorithm 1)

```
Input: W_S (student, khởi tạo ngẫu nhiên), W_T (teacher đã học), g, h
W_Hint   = teacher params lớp 1..h
W_Guided = student params lớp 1..g
W_r      = khởi tạo ngẫu nhiên nhỏ
# Stage 1: pretrain tới guided layer
W_Guided* = argmin_{W_Guided, W_r}  L_HT(W_Guided, W_r)
gán W_S[1..g] = W_Guided*
# Stage 2: KD toàn mạng
W_S* = argmin_{W_S}  L_KD(W_S)      # lambda giảm tuyến tính
```

Regressor chỉ dùng ở giai đoạn 1 và bị bỏ sau đó.

### 3.4 Liên hệ curriculum learning (mục 2.4)

Paper giải thích: số hạng CE với teacher trong $\mathcal L_{KD}$ làm các mẫu mà teacher tự tin (gần one-hot) có ảnh hưởng mạnh hơn (mẫu teacher kém tự tin cho phân phối gần đều nên "đẩy" tham số yếu hơn), tức là đo "độ dễ" của mẫu qua độ tự tin của teacher. Sau đó giảm $\lambda$ tuyến tính để dần chuyển sang mẫu khó. Curriculum có hai giai đoạn: học khái niệm trung gian (hint), rồi học toàn mạng. Đây là lập luận của tác giả (chưa phải thí nghiệm kiểm chứng riêng).

### 3.5 Siêu tham số (phụ lục)

$\tau=3$; $\lambda$ khởi tạo 4 và giảm tuyến tính trong 500 epoch tới 1 (CIFAR); RMSProp, lr 0.005, batch 128; khởi tạo trọng số $U(-0.005,0.005)$; dừng sau 100 epoch không cải thiện validation, tối đa 500 epoch.

## 4. Thí nghiệm và kết quả chính

Teacher đều là mạng maxout conv của Goodfellow et al. (2013). FitNet là chuỗi conv 3×3 + maxout, nhiều lớp hơn và ít kênh hơn.

### 4.1 Các benchmark

| Dataset | Teacher | FitNet | Ghi chú |
|---|---|---|---|
| CIFAR-10 | ~9M tham số, 90.18% | ~2.5M, **91.61%** (19 lớp, FitNet 4) | tốt hơn teacher; so với mimic ensemble (Ba & Caruana) 85.8% (~70M), ít hơn khoảng 28 lần tham số |
| CIFAR-100 | ~9M, 63.54% | ~2.5M, **64.96%** | state-of-the-art tham chiếu trong paper: DSN 65.43% |
| SVHN | ~4.9M, 2.38% lỗi | ~1.5M, 2.42% lỗi | "ngang teacher với 32% capacity" |
| MNIST | ~361K, 0.55% lỗi | ~30K, **0.51%** | backprop thường 1.9%, KD 0.65% |
| AFLW (16×16 mặt/không mặt) | teacher 4.21% lỗi | FitNet 1 (ít hơn 15x phép nhân): KD 4.58%, HT 2.55%; FitNet 2 (ít hơn 2.5x): KD 1.95%, HT 1.85% | mô tả trong text |

Teacher CIFAR-10/100 có 3 conv maxout (96-192-192) + FC maxout 500. Student CIFAR-10 trong Bảng 1 có 17 conv maxout + 1 FC maxout + softmax; lớp 11 của student mimic lớp 2 của teacher. Các số trong bảng trên chép từ Table 1–4 và text mục 3.4.

Các hệ thống mạnh nhất khác trong Table 1 (CIFAR-10): Maxout 90.65%, Network in Network 91.2%, Deeply-Supervised Networks 91.78%; "mimic" của Ba & Caruana: 84.6% (~54M), 84.9% (~70M), 85.8% ensemble (~70M).

### 4.2 Độ sâu dưới ngân sách tính toán cố định (mục 4.1)

Thí nghiệm CIFAR-10 với hai ngân sách khoảng 30M và 107M phép nhân, mạng 3/5/7/9 conv (thực tế đồ thị trục x là số lớp 5, 7, 9, 11):
- Backprop thường không huấn luyện được mạng > 5 lớp ở ngân sách 30M.
- KD huấn luyện được tới 7 lớp, rồi chạm giới hạn.
- Hint-based training (HT) huấn luyện được mạng 13 lớp ở 30M phép nhân.
- Ở 107M, mạng 7 lớp với HT tăng +0.7% (từ 89.45% lên 90.1%) so với không hint.
- Mạng 11 lớp (HT) vượt mạng 5 lớp 4.11% tuyệt đối ở 107M và 3.4% ở 30M.

### 4.3 Đánh đổi tốc độ / độ chính xác (Table 5, CIFAR-10)

| Mạng | Lớp | Tham số | Phép nhân | Acc | Tăng tốc | Tỉ lệ nén |
|---|---|---|---|---|---|---|
| Teacher | 5 | ~9M | ~725M | 90.18% | 1 | 1 |
| FitNet 1 | 11 | ~250K | ~30M | 89.01% | 13.36 | 36 |
| FitNet 2 | 11 | ~862K | ~108M | 91.06% | 4.64 | 10.44 |
| FitNet 3 | 13 | ~1.6M | ~392M | 91.10% | 1.37 | 5.62 |
| FitNet 4 | 19 | ~2.5M | ~382M | 91.61% | 1.52 | 3.60 |

FitNet 1 nhanh gấp một bậc độ lớn, chỉ giảm 1.3% (so 90.18% và 89.01%). FitNet 2 vượt teacher 0.9% với tốc độ 4.64x. (Tăng tốc đo bằng tổng thời gian suy luận trên GPU.)

## 5. Phân tích / ablation

- **HT vs KD thuần:** khác biệt duy nhất là điểm khởi đầu (ngẫu nhiên hay lấy từ hint). Paper cho rằng hint đưa student vào vùng khởi tạo tốt hơn (lợi ích tối ưu hoá) và còn là regularizer mạnh hơn KD (test tốt hơn khi cố định dung lượng).
- **Hint từ hidden layer teacher vs hint từ nhãn (mục 3.1, các lựa chọn 1–4):** (1) stage-wise: stage 1 tối ưu nửa mạng đầu theo nhãn, stage 2 tối ưu toàn mạng theo nhãn: stage 2 không học được. (2) tương tự nhưng stage 2 theo Eq. (2): không học được. (3) tối ưu chung hint có giám sát + nhãn ở output: thử nhiều khởi tạo và learning rate RMSprop, không có mạng nào học được. (4) tương tự với Eq. (2): không học được. Thêm hint giám sát ở mọi lớp theo kiểu Deeply-Supervised Nets (DSN) với kiến trúc 19 lớp đạt **88.2%**, kém xa 91.61% của FitNet. Paper diễn giải: hint rất phân biệt theo lớp có thể quá "gắt"; hint mềm hơn từ teacher cho tổng quát hoá tốt hơn.
- **Độ sâu quan trọng:** với cùng ngân sách phép nhân, mạng sâu hơn (có hint) tốt hơn mạng nông hơn.
- **MNIST (kiểm chứng đóng góp của hint):** cùng kiến trúc student: backprop 1.9%, KD 0.65%, HT 0.51%.
- Paper **không** có ablation riêng cho vị trí hint/guided layer (chỉ chọn "giữa" theo trực giác), cũng không quét $\lambda$ hay $\tau$.

## 6. Giới hạn

**Do tác giả nêu:**
- Hint sâu hơn dễ over-regularize; cần chọn cặp hint/guided để student không bị khoá quá.
- Phân tích mức dư thừa trong filter của FitNet, kết hợp với matrix factorization và quantization được để làm future work; các phương pháp đó được coi là bổ trợ.

**Do chúng ta nhận thấy:**
- Chỉ có một cặp hint/guided, chọn bằng heuristic; không có phân tích độ nhạy. Hint lấy từ lớp 2 của teacher chỉ 3 lớp conv (rất nông), khác xa teacher kiểu Transformer.
- Quy trình hai giai đoạn thêm bước huấn luyện và nhiều siêu tham số ($\lambda$ lịch giảm, $\tau$, lr riêng).
- Regressor/feature L2 giả định hai không gian đặc trưng có thể dự đoán nhau; không có cơ sở lý thuyết cho "khớp tuyệt đối" feature, và tín hiệu L2 trên feature có thể bị chi phối bởi các kênh có biên độ lớn (nhận xét của chúng ta, bài 04 chỉ ra cách chuẩn hoá).
- Thí nghiệm là CNN ảnh nhỏ (32×32, 16×16), không có chuỗi hay Transformer.
- "Student vượt teacher" xảy ra trên teacher tương đối nông (3 conv) và có thể do kiến trúc student tốt hơn chứ không do KD thuần; paper cố gắng kiểm soát bằng các baseline backprop/KD trên cùng kiến trúc ở MNIST và hình 2.

## 7. Liên hệ với các bài khác

- **02 Hinton KD:** stage 2 của FitNets chính là KD; FitNets bổ sung khởi tạo bằng hint.
- **04 Attention transfer:** thay vì khớp feature thô qua regressor, khớp bản đồ attention (tổng bình phương/lũy thừa theo kênh, rồi chuẩn hoá), không cần regressor vì kích thước kênh bị gộp.
- **12 Relational KD, 13 CRD:** khớp quan hệ giữa các mẫu thay vì từng feature; bước tiến tiếp.
- **19 MiniLM, 18 TinyBERT, 17 Patient KD:** feature/hidden-state KD cho Transformer; TinyBERT dùng projection tương tự regressor.
- **23 AM-RADIO, 24 CLIP-KD, 26 LLaVA-KD:** feature distillation giữa các mô hình thị giác.
- **20 Causal distillation:** khớp hidden state theo can thiệp nhân quả.
- **10 Does KD really work:** nghiên cứu mức độ student khớp teacher; nhắc rằng khớp feature không tự động đồng nghĩa với khớp hàm.

## 8. Ứng dụng cho dự án (suy luận)

- **Hint cho vị trí cắt:** khi prune (cắt layer hoặc neuron MLP) ta biết chính xác lớp nào bị thay đổi. Mô phỏng "hint" là cho student khớp hidden state của teacher *ngay tại và sau điểm cắt*. Đây là dạng feature-matching dễ chạy và rẻ, vì student là bản thu nhỏ có cấu trúc tương ứng với teacher (nếu pruning giữ chiều residual stream thì có thể khớp không cần regressor; nếu 0.3B/0.1B đổi chiều ẩn thì cần projector tuyến tính, tương ứng regressor của FitNet).
- **Gắn với D_lin/D_res:** nếu một phần hư hại là D_lin (sửa được bằng ánh xạ tuyến tính tối ưu tại điểm cắt), thì một regressor tuyến tính học trong stage 1 gần như chính là cách ước lượng và hấp thụ D_lin; phần còn lại D_res mới cần fine-tune sâu. Đây là suy luận của chúng ta: paper không phân tích theo cách này. Thực tế regressor trong paper là conv phi tuyến (cùng phi tuyến với hint), chưa phải chỉ tuyến tính.
- **Quan sát "cắt MLP neuron hồi phục tệ hơn cắt layer":** gợi ý (giả thuyết, cần kiểm tra) rằng khi cắt neuron, lỗi trải rộng lên mọi lớp nên cần hint ở nhiều tầng thay vì chỉ lớp giữa. FitNets chỉ dùng một cặp hint, nên không trả lời trực tiếp.
- **Hai giai đoạn (warm start bằng hint, rồi KD output):** áp dụng được: giai đoạn 1 chỉ huấn luyện các khối quanh điểm cắt với loss MSE hidden-state (nhanh vì không cần chạy decoder đầy đủ + softmax trên từ vựng lớn), giai đoạn 2 mới KD toàn mô hình. Cần kiểm chứng với kiến trúc hybrid GatedDeltaNet.
- Bài này là CNN; mọi chuyển giao sang VLM/Transformer đều là ngoại suy.

## 9. Thí nghiệm gợi ý

1. **Hint tại điểm cắt:** trên một mức prune (ví dụ bỏ 1 block decoder), so ba cách phục hồi: (a) chỉ KD output, (b) MSE hidden-state tại đầu ra block liền sau điểm cắt + KD output, (c) hai giai đoạn kiểu FitNets (hint trước, KD sau). Đo CER/WER và số bước để về ngưỡng chất lượng.
2. **Có/không projector:** nếu giữ nguyên chiều ẩn, so khớp thẳng với khớp qua một lớp tuyến tính học được; đo xem lớp tuyến tính có hấp thụ được phần "D_lin" (đo lỗi MSE trước/sau khi học nó).
3. **Chọn lớp hint:** quét vị trí hint ở 25%, 50%, 75% độ sâu teacher (như cảnh báo của paper về hint quá sâu) và xem có hiện tượng over-regularization (loss output phục hồi kém hơn).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| Loss KD | $H(y,P_S)+\lambda H(P_T^\tau,P_S^\tau)$, $\tau=3$ |
| Hint loss | $\tfrac12\|u_h(x)-r(v_g(x))\|^2$ |
| Regressor conv | kernel $k_i=N_{g,i}-N_{h,i}+1$, tham số $k_1k_2O_hO_g$ |
| Stage 1 | huấn luyện student tới guided layer + regressor theo hint |
| Stage 2 | KD toàn mạng với $\lambda$ giảm tuyến tính từ 4 về 1 |
| Kết quả chính | CIFAR-10: 91.61% (~2.5M) vs teacher 90.18% (~9M); MNIST 0.51% vs 0.55% |
| Hint hay DSN? | hint từ teacher 91.61% vs DSN theo nhãn 88.2% (cùng kiến trúc 19 lớp) |

## 11. Kết luận ngắn

FitNets mở rộng KD từ output sang hidden layer: lấy đầu ra lớp giữa của teacher làm gợi ý, qua một regressor, để khởi tạo student sâu và mảnh, rồi KD toàn mạng. Kết quả: student ít tham số hơn (khoảng 3.6 đến 36 lần tuỳ cấu hình trên CIFAR-10) vẫn ngang hoặc vượt teacher (91.61% so 90.18%), và hint mềm của teacher tốt hơn hint cứng theo nhãn (DSN 88.2%). Bài học cốt lõi cho dự án: tín hiệu ở hidden layer cộng một ánh xạ khớp chiều là công cụ chuẩn khi student khác teacher về cấu trúc.

**Link gốc:** https://arxiv.org/abs/1412.6550
