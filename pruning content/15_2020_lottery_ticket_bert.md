# The Lottery Ticket Hypothesis for Pre-trained BERT Networks (2020)

**Tác giả:** Tianlong Chen, Jonathan Frankle, Shiyu Chang, Sijia Liu, Yang Zhang, Zhangyang Wang, Michael Carbin (UT Austin, MIT CSAIL, MIT-IBM Watson AI Lab).  
**Venue / arXiv:** arXiv:2007.12223 (v2, 18/10/2020); NeurIPS 2020 (34th Conference on Neural Information Processing Systems).  
**Nguồn:** https://arxiv.org/abs/2007.12223  
**Chủ đề:** Lottery Ticket Hypothesis (LTH) áp cho BERT đã pretrain: tìm subnetwork "matching" bằng iterative magnitude pruning (không cấu trúc), và kiểm tra chúng có chuyển (transfer) giữa các tác vụ không.  
**Vị trí trong lộ trình:** Cần đọc trước: 08 (Lottery Ticket), 09 (Linear mode connectivity & LTH, khái niệm rewinding), 12 (Blalock, vì cùng nhóm tác giả Frankle). Bài này là cầu nối giữa LTH (nghiên cứu khoa học trên CNN nhỏ) và pruning cho mô hình pretrain-rồi-fine-tune. Kế tiếp: 16 (Movement pruning), cùng bối cảnh BERT nhưng hướng ứng dụng.

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

**Lottery Ticket Hypothesis** (Frankle & Carbin 2019, bài 08): một mạng lớn chứa mạng con thưa ("winning ticket") mà nếu train riêng từ khởi tạo ban đầu thì đạt độ chính xác ngang mạng đầy đủ. Quy trình tìm vé: train mạng đầy đủ, cắt trọng số nhỏ, **hồi trọng số còn lại về giá trị lúc khởi tạo** (hoặc lúc sớm trong train: **rewinding**), lặp lại.

Hai chủ đề đã nổi lên trong văn liệu LTH:
1. **Initialization via pre-training**: ở quy mô lớn (CV, NLP), vé chỉ tìm được sau một đoạn train sớm chứ không phải từ khởi tạo ngẫu nhiên; đoạn đầu đó giống như pretrain dày để tạo điểm khởi đầu dễ làm thưa.
2. **Transfer learning**: tìm vé rất tốn kém (train cả mạng, cắt, lặp nhiều lần); chỉ đáng nếu vé dùng lại được cho nhiều tác vụ.

BERT hội tụ cả hai: khởi tạo = trọng số pretrain, và pretrain được khấu hao qua nhiều tác vụ downstream. Câu hỏi: **BERT pretrained có chứa vé? Vé có chuyển tác vụ? Có vé "phổ quát"?**

Thuật ngữ:
- **IMP (iterative magnitude pruning)**: lặp huấn luyện → cắt phần trăm nhỏ trọng số có $|w|$ nhỏ nhất → hồi lại → ...
- **Sparsity (độ thưa)**: tỉ lệ trọng số bằng 0.
- **GLUE, SQuAD**: bộ benchmark NLP (phân loại cặp câu / câu đơn; hỏi đáp).
- **MLM (masked language modeling)**: tác vụ pretrain của BERT, che token rồi đoán.
- **Unstructured pruning**: cắt từng trọng số riêng, tạo ma trận thưa; không tự tăng tốc trên GPU phổ thông.

## 2. Ý tưởng chính

Tác giả áp quy trình LTH lên BERT-base. Họ lấy **trọng số pretrain $\theta_0$ làm "khởi tạo"**, chạy IMP trên từng tác vụ downstream, rồi huấn luyện lại subnetwork bắt đầu từ $\theta_0$. Kết quả: có vé "winning" ở độ thưa 40% đến 90% tùy tác vụ, **ngay tại $\theta_0$** (không cần rewind sớm như các nghiên cứu NLP trước). Vé tìm trên tác vụ nào thì phần lớn chỉ tốt cho tác vụ đó, **trừ vé tìm trên MLM** (chính tác vụ pretrain), ở độ thưa 70% chuyển tốt sang mọi tác vụ downstream, tức là một mạng con "phổ quát" có thể thay BERT đầy đủ.

## 3. Phương pháp chi tiết

### 3.1 Cài đặt

- Mạng: BERT-base (12 block, hidden 768, 12 head, 110M tham số) + lớp phân loại riêng cho tác vụ (< 3% tham số). Ký hiệu: $f(x;\theta,\gamma)$, $\theta$ là tham số BERT, $\gamma$ là tham số lớp phân loại.
- Tác vụ: MLM + 9 tác vụ GLUE (MNLI, QQP, STS-B, WNLI, QNLI, MRPC, RTE, SST-2, CoLA) + SQuAD v1.1; mọi số đo lấy trên tập validation/dev. Siêu tham số chuẩn (AdamW, learning rate giảm tuyến tính về 0; ví dụ 3 epoch, batch 32, lr $2\times10^{-5}$ cho hầu hết GLUE; SQuAD: 2 epoch, batch 16, lr $3\times10^{-5}$; MLM: 0.1 epoch, batch 16, lr $5\times10^{-5}$).
- Subnetwork: $f(x;m\odot\theta,\cdot)$ với mask $m\in\{0,1\}^{d_1}$.

### 3.2 Các định nghĩa

Gọi $A_t^T$ là thuật toán huấn luyện $t$ bước cho tác vụ $T$, $\mathcal E^T$ là metric đánh giá, $\theta_0$ là trọng số pretrain:

- **Matching subnetwork**: $\mathcal E^T\big(A^T_t(f(x;m\odot\theta,\gamma))\big)\ \ge\ \mathcal E^T\big(A^T_t(f(x;\theta_0,\gamma))\big)$
- **Winning ticket**: matching subnetwork với $\theta=\theta_0$ (tức khởi tạo ở trọng số pretrain).
- **Universal subnetwork**: matching trên mọi tác vụ trong tập $\{T_i\}$ (với lớp phân loại $\gamma_{T_i}$ riêng).

Trong thực nghiệm, để chịu nhiễu, subnetwork được coi là winning nếu điểm của nó **nằm trong một độ lệch chuẩn** so với BERT đầy đủ (trung bình năm lần chạy).

### 3.3 Thuật toán IMP với bước rewind $i$ (Algorithm 1)

```text
1. Train BERT pretrain f(x; theta_0, gamma_0) tới bước i  ->  theta_i
2. m <- 1^{d1}
3. repeat
4.     train f(x; m ⊙ theta_i, gamma_i) tới bước t  ->  theta_t        # train trên tác vụ T
5.     cắt 10% trọng số còn lại có |theta_t| nhỏ nhất (toàn cục), cập nhật m
6. until sparsity(m) = s
7. return f(x; m ⊙ theta_i)                                            # rewind về theta_i
```

Cắt **toàn cục** (global) theo độ lớn trên cả mạng. Với $i=0$, subnetwork trả về bắt đầu từ $\theta_0$ (winning ticket). **Standard pruning** (đối chứng): train xong, cắt 10% nhỏ nhất, train tiếp mà *không* rewind, lặp tới độ thưa mong muốn; các trọng số giữ lại là trọng số cuối quá trình train.

### 3.4 Bốn "claim" của LTH được kiểm lại

1. IMP tìm được winning ticket $f(x;m_{IMP}\odot\theta_0)$.
2. Các vé đó thưa hơn mức mà random pruning hoặc random init còn matching.
3. Ở mạng khác, vé chỉ có ở bước sớm $i$ (rewinding); hoặc rewind giúp hơn $\theta_0$.
4. Vé đạt cùng accuracy ở cùng độ thưa với standard pruning.

## 4. Thí nghiệm và kết quả chính

### 4.1 Tồn tại winning ticket (Table 2; trung bình 5 lần chạy ± độ lệch chuẩn)

| Tác vụ | MNLI | QQP | STS-B | WNLI | QNLI | MRPC | RTE | SST-2 | CoLA | SQuAD | MLM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Độ thưa cao nhất còn là vé | 70% | 90% | 50% | 90% | 70% | 50% | 60% | 60% | 50% | 40% | 70% |
| BERT-base đầy đủ | 82.4 ± 0.5 | 90.2 ± 0.5 | 88.4 ± 0.3 | 54.9 ± 1.2 | 89.1 ± 1.0 | 85.2 ± 0.1 | 66.2 ± 3.6 | 92.1 ± 0.1 | 54.5 ± 0.4 | 88.1 ± 0.6 | 63.5 ± 0.1 |
| IMP từ $\theta_0$ | 82.6 ± 0.2 | 90.0 ± 0.2 | 88.2 ± 0.2 | 54.9 ± 1.2 | 88.9 ± 0.4 | 84.9 ± 0.4 | 66.0 ± 2.4 | 91.9 ± 0.5 | 53.8 ± 0.9 | 87.7 ± 0.5 | 63.2 ± 0.3 |
| Random pruning (mặt nạ ngẫu nhiên) | 67.5 | 76.3 | 21.0 | 53.5 | 61.9 | 69.6 | 56.0 | 83.1 | 9.6 | 31.8 | 32.3 |
| IMP mask + random init $\theta_0'$ | 61.0 | 77.0 | 9.2 | 53.5 | 60.5 | 68.4 | 54.5 | 80.2 | 0.0 | 18.6 | 14.4 |
| IMP mask + trọng số pretrain xáo trộn trong từng layer $\theta_0'''$ | 70.1 | 79.2 | 19.6 | 53.3 | 62.0 | 69.6 | 52.7 | 82.6 | 4.0 | 24.2 | 42.3 |

(Metric: accuracy cho hầu hết; STS-B dùng Pearson, CoLA dùng Matthew's correlation, SQuAD dùng F1, MRPC/QQP có thêm số khác trong appendix.) Độ thưa vé cao nhất: từ 40% (SQuAD), 50% (MRPC, CoLA) đến 90% (QQP, WNLI). Paper nhận xét **không thấy mối quan hệ rõ** giữa độ thưa này và tính chất tác vụ (như cỡ tập train).

Diễn giải: ví dụ MNLI, random pruning thấp hơn khoảng 15 điểm, random reinit thấp hơn khoảng 21 điểm (theo paper). Xáo trộn trọng số trong layer tốt hơn random init nhưng vẫn kém xa vé, cho thấy phân phối trọng số theo layer của BERT pretrain là điểm khởi đầu "hơi có thông tin". Figure 1 (CoLA, SST-2, SQuAD): random pruning chỉ gần IMP ở độ thưa thấp (10-20%).

### 4.2 Rewinding không giúp (Table 3)

Rewind đến 5%, 10%, 20%, 50% quá trình train; ví dụ SQuAD: bước 1000, 2000, 4000, 10000. Rewind **không cải thiện đáng kể** ở tác vụ nào; ở STS-B và RTE kết quả giảm đến mức không còn matching (biên 2 điểm phần trăm). Ví dụ RTE từ 66.06 (rewind 0%) xuống 62.09 (rewind 5%/10%). Giải thích khả dĩ của tác giả: tập train nhỏ nên overfit.

### 4.3 IMP vs standard pruning (Table 3, dòng cuối)

Tùy tác vụ: QQP, QNLI, MRPC, MLM standard pruning hơn vé tới ~2 điểm; STS-B, WNLI, RTE, SST-2 kém hơn tới ~3 điểm. Ví dụ SST-2: 90.02 (standard) so với 91.74 (rewind 0%); QNLI: 89.88 so với 88.05. Chênh lớn nhất ở tác vụ có tập train nhỏ, nơi standard pruning có thể overfit.

### 4.4 Transfer (mục 5; độ thưa cố định 70% để loại yếu tố kích thước)

- Quy trình: IMP trên tác vụ nguồn $S$ → train subnetwork (từ $\theta_0$, lớp phân loại mới) trên tác vụ đích $T$; $\text{Transfer}(S,T)$.
- **Phần lớn vé không chuyển**: trong 11 tác vụ, chỉ 3 tác vụ nguồn cho vé chuyển sang hơn hai tác vụ khác (để chuyển bằng same-task). Nếu cho phép giảm 2.5 điểm thì 7 tác vụ nguồn chuyển được ít nhất một nửa tác vụ khác.
- **Vé từ MLM là phổ quát**: transfer ngang same-task trên mọi tác vụ (Figure 3). Cắt trực tiếp trọng số pretrain $\theta_0$ (không IMP) kém hơn: chỉ ngang same-task ở 4 trường hợp, nhưng vẫn tốt hơn mọi tác vụ nguồn khác ngoài MLM.
- Phụ lục: ở độ thưa $s\%$ cao nhất mà tác vụ $T$ có vé, vé MLM cùng độ thưa còn là winning ticket với 5 tác vụ; 3 tác vụ nữa chênh dưới nửa điểm; hai tác vụ còn lại chênh 1.6 và 2.6 điểm.
- **Quy luật**: khả năng chuyển tương quan với **số mẫu train** của tác vụ nguồn (MNLI chuyển được 4 tác vụ, SQuAD 3, MRPC 1, WNLI 0, MLM tốt nhất), **không** liên quan loại tác vụ.
- **Chuyển từ $\theta_0$ có tốt hơn?** Với SQuAD (nguồn tốt thứ hai sau MLM), rewind hoặc standard pruning cho kết quả chuyển bằng hoặc tốt hơn, tức trọng số đã học trên tác vụ nguồn không hại, thậm chí giúp.

Paper kết luận có thể xem đây là "giai đoạn thứ hai của pretrain BERT": sau pretrain, chạy IMP với MLM để được một subnetwork 70% thưa tương đương về năng lực.

## 5. Phân tích / ablation

- **Mask quan trọng, không chỉ độ thưa**: random pruning (mask ngẫu nhiên) thua IMP xa ở độ thưa cao.
- **Khởi tạo quan trọng**: random init và xáo trộn đều thua; trọng số pretrain mang thông tin.
- **Rewinding vô ích ở BERT**: khác với các bối cảnh khác (nơi vé chỉ tồn tại ở bước sớm), vé có sẵn tại $\theta_0$; coi pretrain là giai đoạn "dense pretraining" làm điểm khởi đầu dễ thưa hóa.
- **Độ thưa vé không liên quan thuộc tính tác vụ**: không có quy luật theo kích thước tập train.
- **Chồng lấn mask**: (Phụ lục D) mask của các tác vụ downstream chia sẻ nhiều trọng số bị cắt chung; mask của MLM khác nhiều hơn.

## 6. Giới hạn

**Do tác giả nêu:**
- Họ dùng pruning **không cấu trúc** có chủ đích (nghiên cứu khoa học, không phải nỗ lực tăng tốc), nên không đo speedup phần cứng; ghi chú rằng ở độ thưa 70-90% XNNPACK đã cho tăng tốc trên CPU điện thoại, và để tương lai.
- Chi phí tìm vé: IMP cần train đầy đủ nhiều lần, đắt hơn nhiều so với train thường.
- Tác giả so sánh với Prasanna et al. (structured, cắt head/MLP): kết quả của họ khác; vì vé structured ít bằng chứng trong CV (Liu 2019, bài 11).

**Do chúng tôi nhận thấy (suy luận):**
- Chỉ BERT-base (110M) và các tác vụ nhỏ của GLUE/SQuAD; không có mô hình sinh, không có LLM, không có đa phương thức. Không biết có áp được cho VLM/OCR không.
- "Matching" được định nghĩa trong một độ lệch chuẩn; ở tác vụ tập nhỏ có phương sai cao (RTE: ±3.6, WNLI toàn ngẫu nhiên ~54.9) tiêu chí này lỏng; WNLI (accuracy 54.9) hầu như không học được nên vé 90% thưa ở WNLI không có nhiều ý nghĩa.
- Cho thấy tồn tại mạng con thưa, không cho thuật toán rẻ tìm chúng; không bằng movement pruning (16) về hiệu quả đối với mức nén tổng thể.
- Theo checklist bài 12: ít kiến trúc (1), một cỡ mô hình, nhưng có 5 lần chạy cho Table 2 và nhiều tác vụ, tốt hơn trung bình văn liệu.

## 7. Liên hệ với các bài khác

- **08 Lottery Ticket**: định nghĩa winning ticket, IMP gốc; ở bài 08, vé có ở khởi tạo ngẫu nhiên với mạng nhỏ.
- **09 Linear mode connectivity & LTH**: giải thích vì sao vé ở mạng lớn cần rewinding (ổn định với nhiễu SGD); bài 15 cho thấy BERT pretrain đã "ổn định" nên $i=0$ là đủ.
- **11 Rethinking the value of pruning**: gần như đối lập: ở CV với structured pruning, mạng thưa train từ đầu cũng tốt. Bài 15 dùng unstructured và khởi tạo pretrain nên kết luận khác; cần đọc cả hai để thấy kết luận phụ thuộc cài đặt.
- **12 State of pruning**: đồng tác giả Frankle; bài 15 là ví dụ có kiểm soát (random baselines, 5 lần chạy).
- **13-14 (cắt head)**, **16 Movement pruning**, **17 CoFi**: các cách khác nén BERT; 16 chọn trọng số theo "di chuyển" khi fine-tune thay vì độ lớn.
- **22-24 (OBC, SparseGPT, Wanda)**: pruning không cấu trúc cho LLM sau train, không cần IMP.

## 8. Ứng dụng cho dự án (suy luận)

Mức liên quan: **thấp** trực tiếp, **trung bình** về ý tưởng.

1. **Không dùng được trực tiếp**: IMP không cấu trúc tạo ma trận thưa không tăng tốc trên GPU ≤ 8 GB batch-1, và tốn chi phí train lặp. Dự án cần structured pruning.
2. **Bài học khái niệm quan trọng**: tại điểm khởi đầu pretrain, mask quan trọng chứ không chỉ kích thước; mask ngẫu nhiên kém xa mask theo độ lớn. Nghĩa là baseline "cắt ngẫu nhiên" nên có trong thí nghiệm của dự án để làm chuẩn debug.
3. **Vé MLM phổ quát** gợi ý rằng mạng con tìm bằng **chính loss pretrain** (loss dự đoán token của LM, hay loss OCR sinh văn bản) chuyển tốt hơn mạng con tìm từ một tác vụ downstream hẹp. Tương tự cho dự án: chọn đơn vị cắt dựa trên loss sinh văn bản tổng quát/đa dạng tài liệu thay vì một loại tài liệu hẹp (suy luận, chưa kiểm chứng ở VLM).
4. **Rewinding vô ích** vì mô hình đã pretrain: trùng với thực hành của dự án (cắt từ checkpoint đã train, không cần train lại từ đầu).
5. **Giả thuyết cảnh báo**: LTH nói "mạng con tồn tại", không nói "cắt xong phục hồi được bằng ít dữ liệu". Việc bạn dùng distillation từ teacher 4B là một cơ chế phục hồi khác hẳn.

## 9. Thí nghiệm gợi ý

1. **Baseline ngẫu nhiên có kiểm soát**: với cấu hình MLP-neuron-cắt (cùng số tham số), so mask theo độ lớn / theo Taylor / ngẫu nhiên, sau cùng một lịch distillation ngắn; xem khoảng cách giống Table 2 (random thua xa) hay nhỏ hơn.
2. **Tính chuyển của tiêu chí cắt**: chọn đơn vị cắt theo (a) loss OCR trên tập tài liệu sạch, (b) loss trên tập hỗn hợp (hóa đơn, biểu mẫu, chữ viết tay); đánh giá chéo xem (b) có "phổ quát" hơn không (tương tự MLM vs tác vụ hẹp).
3. **Chồng lấn mask**: tính Jaccard giữa các tập đơn vị được giữ khi dùng các tập dữ liệu hiệu chuẩn khác nhau; nếu chồng lấn cao thì dữ liệu hiệu chuẩn không quan trọng.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Subnetwork | $f(x;m\odot\theta,\cdot)$, $m\in\{0,1\}^{d_1}$ |
| Matching | $\mathcal E^T(A_t^T(f(x;m\odot\theta,\gamma)))\ge \mathcal E^T(A_t^T(f(x;\theta_0,\gamma)))$ |
| Winning ticket | matching với $\theta=\theta_0$ |
| Universal | matching trên mọi tác vụ trong tập |
| IMP | train → cắt 10% $|w|$ nhỏ nhất (toàn cục) → rewind → lặp |
| Standard pruning | cắt lặp, không rewind |
| Độ thưa vé | 40% (SQuAD) … 90% (QQP, WNLI) |
| Vé MLM, 70% | phổ quát cho các tác vụ downstream |
| Transfer | tương quan với kích thước tập train nguồn, không với loại tác vụ |

## 11. Kết luận ngắn

Với BERT, các quan sát LTH vẫn đúng: có vé thưa 40-90% tại chính trọng số pretrain, mask và khởi tạo đều quan trọng, rewinding không cần, và vé tìm bằng MLM ở 70% thưa là phổ quát cho GLUE/SQuAD. Đây là kết quả khoa học đẹp về cấu trúc mạng pretrain nhưng không cung cấp phương pháp tăng tốc thực tế (không cấu trúc, tốn kém). Với dự án, hãy lấy từ đó các bài học về baseline ngẫu nhiên, chọn tiêu chí cắt theo mục tiêu tổng quát, và hiểu giới hạn của "tồn tại mạng con".

Link gốc: https://arxiv.org/abs/2007.12223
