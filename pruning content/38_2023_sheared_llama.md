# Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning (2023)

**Tác giả:** Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, Danqi Chen (Princeton NLP).  
**Venue / arXiv:** arXiv:2310.06694 (v1: 10/10/2023; bản cuối 04/2024); ICLR 2024.  
**Nguồn:** https://arxiv.org/abs/2310.06694  
**Chủ đề:** Structured pruning có mục tiêu (targeted) cho LLM + tiếp tục pre-train (continued pre-training) + dynamic batch loading; "prune rồi train tiếp" thay cho train từ đầu.  
**Vị trí trong lộ trình:** Đọc trước: 17 (CoFi, cùng nhóm tác giả, mask học được), 26 (LLM-Pruner), 25 (ZipLM). Bài này thêm: (1) ép mô hình sau prune về đúng một *kiến trúc đích* chuẩn, (2) chứng minh rằng bước train tiếp với lượng token lớn (50B) mới là thứ tạo ra mô hình cạnh tranh, (3) cân bằng dữ liệu theo domain. Tiếp theo: 39 (Minitron) làm cùng ý tưởng nhưng dùng distillation và so sánh depth/width kỹ hơn.

> Ghi chú học tập bằng tiếng Việt, diễn giải và tóm tắt, không phải bản dịch. Các số liệu lấy từ paper; phần "Ứng dụng cho dự án" là suy luận của chúng ta, không phải kết luận của tác giả.

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

- Train một LLM cỡ vừa (1-3B) từ đầu cần hàng nghìn tỷ token (TinyLlama-1.1B: 3T token). Trong khi đó đã có sẵn các mô hình lớn mạnh (LLaMA2-7B, 2T token). Câu hỏi: có thể "cắt" mô hình lớn thành mô hình nhỏ rẻ hơn nhiều không?
- **Structured pruning** (cắt có cấu trúc): bỏ nguyên cả head, cả layer, cả chiều trung gian của FFN... sao cho ma trận còn lại vẫn *dense* (đặc), chạy nhanh trên GPU thường. Khác với unstructured (xoá từng trọng số, ma trận thưa khó tăng tốc).
- Hai vấn đề của các phương pháp trước (CoFi, LLM-Pruner):
  1. Không chỉ định cấu trúc đích: kết quả có kiến trúc "lạ" (mỗi layer số head khác nhau; LLM-Pruner ra intermediate size nhỏ hơn hidden size), nên suy luận chậm và khó train tiếp.
  2. Sau khi prune chỉ fine-tune ít; paper cho rằng cần bước **continued pre-training** (train tiếp bằng loss language modeling trên lượng lớn dữ liệu) mới đạt chất lượng "general-purpose".
- Tác giả quan sát: khi train tiếp mô hình đã prune trên dữ liệu pre-training gốc, loss giảm *không đều* giữa các domain so với train từ đầu. Pruning giữ lại nhiều kiến thức ở domain "dễ/ít entropy" (GitHub) hơn domain "khó/lớn" (C4).

## 2. Ý tưởng chính

Hai thành phần gọi chung là **LLM-shearing**. (1) *Targeted structured pruning*: học các mask (0/1) trên layer, hidden dim, head, intermediate dim, kèm ràng buộc Lagrange ép tổng số phần tử giữ lại đúng bằng kiến trúc đích (ví dụ lấy cấu hình Pythia-1.4B hoặc INCITE-Base-3B). Kết quả là mô hình dense, đồng nhất giữa các layer. (2) *Dynamic batch loading*: trong lúc prune và train tiếp, cứ mỗi m bước đo loss trên từng domain, domain nào còn xa "loss tham chiếu" thì được lấy nhiều dữ liệu hơn. Với chỉ 0.4B token cho prune và 50B token cho train tiếp (khoảng 3% chi phí so với train từ đầu), mô hình 1.3B và 2.7B vượt các mô hình cùng cỡ.

## 3. Phương pháp chi tiết

### 3.1 Mask ở nhiều mức hạt

Mô hình nguồn $M_S$ có $L_S$ layer (mỗi layer = 1 MHA + 1 FFN), hidden $d_S$, $H_S$ head, intermediate $m_S$. Các biến mask:

| Mức | Mask |
|---|---|
| Layer | $z^{layer}\in\mathbb R^{L_S}$ |
| Hidden dimension (dùng chung mọi layer) | $z^{hidden}\in\mathbb R^{d_S}$ |
| Head | $z^{head}\in\mathbb R^{H_S}$ (mỗi layer) |
| Intermediate dim của FFN | $z^{int}\in\mathbb R^{m_S}$ (mỗi layer) |

Mask nhân vào output tương ứng: head thì nhân vào output của head; intermediate thì nhân vào activation trung gian; layer thì nhân vào output layer; hidden thì nhân vào cả output của head lẫn MLP. Nhân mask = 0 tương đương xoá tham số.

### 3.2 Hard concrete (l0 regularization)

Mask không phải biến rời rạc nên khó đạo hàm. Dùng phân bố **hard concrete** (Louizos et al. 2018): một xấp xỉ liên tục, có khối lượng xác suất tập trung ở 0 và 1.

$$u\sim U(0,1),\quad s=\sigma\!\Big(\tfrac1\beta\big(\log\tfrac{u}{1-u}+\log\alpha\big)\Big),\quad \bar s=s(\zeta-\gamma)+\gamma,\quad z=\min(1,\max(0,\bar s))$$

Siêu tham số paper dùng: $\beta=0.83,\ \zeta=1.1,\ \gamma=-0.1$. $\alpha$ là tham số học được của từng mask.

### 3.3 Ràng buộc kiến trúc đích bằng Lagrange

Thay vì đặt một mức sparsity tổng, paper đặt ràng buộc cho từng loại cấu trúc. Ví dụ với số head đích $H_T$ trên một layer:

$$\tilde{\mathcal L}^{head}(\lambda,\phi,z)=\lambda^{head}\cdot\Big(\sum z^{head}-H_T\Big)+\phi^{head}\cdot\Big(\sum z^{head}-H_T\Big)^2$$

Tương tự cho layer, hidden, intermediate. Bài toán min-max:

$$\boxed{\min_{\theta,z}\ \max_{\lambda,\phi}\ \mathcal L_{prune}(\theta,z,\lambda,\phi)=\mathcal L(\theta,z)+\sum_{j=1}^{L_S}\tilde{\mathcal L}^{head}_j+\sum_{j=1}^{L_S}\tilde{\mathcal L}^{int}_j+\tilde{\mathcal L}^{layer}+\tilde{\mathcal L}^{hidden}}$$

$\mathcal L(\theta,z)$ là loss language modeling của mô hình có mask. Trọng số $\theta$ và mask $z$ được học *cùng nhau*; $\lambda,\phi$ đi theo hướng ngược (gradient ascent) để ép ràng buộc được thoả. Sau pruning, giữ các thành phần có điểm mask cao nhất đúng theo số lượng đích rồi bỏ phần còn lại. Mask thường hội tụ về 0/1 nên hầu như không cần "hấp thụ" mask vào trọng số.

Ghi chú chi tiết (Appendix D): LLaMA2-7B dùng FFN dạng GLU (3 ma trận gate/up/down). Head dim kế thừa từ nguồn (128), nên số head đích = hidden đích / head dim nguồn.

### 3.4 Dynamic batch loading

Gom dữ liệu thành $k$ domain (CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, StackExchange). Mỗi domain có tập validation riêng và một **loss tham chiếu** $\ell_{ref}$. Mặc định tham chiếu là loss dự đoán từ scaling law (fit trên LLaMA2-7B/13B/70B: $L(N,D)=E+A/N^{\alpha}+B/D^{\beta}$, suy ra loss của LLaMA2-1.3B/2.7B giả định). Phương án phụ: dùng loss của chính mô hình nguồn (source reference), kém hơn chút, đặc biệt ở toán và code, nhưng dùng được khi không có họ mô hình nhiều cỡ.

```
Input: k domain D_i, validation D_i^val, trọng số đầu w0, loss tham chiếu l_ref,
       số bước T, chu kỳ đánh giá m
for t = 1..T:
    if t mod m == 0:
        l_t[i]  = loss(model, D_i^val)            # (với pruning: dùng L_prune)
        Δ_t[i]  = max(l_t[i] - l_ref[i], 0)        # domain nào còn "nợ" loss
        α[i]    = w_{t-m}[i] * exp(Δ_t[i])         # exponential ascent
        w_t     = α / sum(α)                       # chuẩn hoá
    batch B ~ sampling theo w_t
    cập nhật (θ, z, φ, λ) hoặc θ trên B
```

Giai đoạn prune dùng trọng số RedPajama gốc làm $w_0$; giai đoạn train tiếp dùng trọng số cuối của giai đoạn prune.

Ví dụ nhỏ: nếu C4 còn cao hơn tham chiếu 0.1 và GitHub đã thấp hơn tham chiếu ($\Delta=0$), sau một lần cập nhật trọng số C4 nhân với $e^{0.1}\approx1.105$ còn GitHub giữ nguyên rồi chuẩn hoá, nên C4 được lấy nhiều dữ liệu hơn.

### 3.5 Thiết lập huấn luyện (Table 7)

| | Pruning | Continued pre-training |
|---|---|---|
| Ngân sách | 0.4B token | 50B token |
| LR của $z,\phi,\lambda$ | 1.0 | không có |
| LR của $\theta$ | 1e-4 | 1e-4 |
| Batch (token) | 131K | 1M |
| Chu kỳ đánh giá $m$ | 50 bước | 400 bước |
| Số bước | 3,200 | 51,200 |
| GPU | 8 | 16 |
| Throughput | 15K tok/s | 145K (1.3B) / 77K (2.7B) |

Pruning chậm khoảng 5 lần so với train LM thường, nên chỉ dùng ngân sách nhỏ.

## 4. Thí nghiệm và kết quả chính

Nguồn: LLaMA2-7B. Đích: 1.3B (kiến trúc Pythia-1.4B) và 2.7B (kiến trúc INCITE-Base-3B). Dữ liệu: RedPajama, chuỗi 4,096 token. Sheared-LLaMA-1.3B: 24 layer, hidden 2048, intermediate 5504, 16 head; 2.7B: 32 layer, hidden 2560, intermediate 6912, 20 head (Table 8).

**Bảng downstream (Table 2, trích; số copy từ paper, cột Average là trung bình của 11 tác vụ):**

| Mô hình (token train) | Average |
|---|---|
| LLaMA2-7B (2T) | 64.6 |
| OPT-1.3B (300B) | 48.2 |
| Pythia-1.4B (300B) | 48.9 |
| TinyLlama-1.1B (3T) | 50.0 |
| **Sheared-LLaMA-1.3B (50B)** | **51.0** |
| OPT-2.7B (300B) | 51.4 |
| Pythia-2.8B (300B) | 52.5 |
| INCITE-Base-3B (800B) | 54.7 |
| OpenLLaMA-3B-v1 (1T) | 55.1 |
| OpenLLaMA-3B-v2 (1T) | 55.7 |
| **Sheared-LLaMA-2.7B (50B)** | **56.7** |

Vài cột riêng: Sheared-2.7B HellaSwag (10-shot) 70.8, ARC-C (25-shot) 41.2, BoolQ 73.7, MMLU (5-shot) 26.4, NQ 16.5; Sheared-1.3B MMLU 25.7, NQ 9.6. MMLU của các mô hình nhỏ đều gần mức ngẫu nhiên (~25-27%).

- Instruction tuning (10,000 cặp ShareGPT, GPT-4 làm giám khảo, hoán vị thứ tự để bù position bias): Sheared-1.3B thắng Pythia-1.4B 57.4% vs 42.7%; Sheared-2.7B thắng INCITE-Base-3B 63.5% vs 36.6%, OpenLLaMA-v1 56.6% vs 43.5%, OpenLLaMA-v2 54.3% vs 45.8%.
- Chi phí: 0.4B + 50B token, tức khoảng 3% (1/32) của OpenLLaMA-3B-v2 hay khoảng 5% ngân sách pre-training của OpenLLaMA theo lời tác giả. Tác giả nêu cả hiệu quả mẫu: 1.3B vượt TinyLlama dù TinyLlama dùng 3T token.

**Dynamic batch loading (Table 3, tỷ lệ dữ liệu thực dùng, 1.3B):**

| | CC | GitHub | Book | StackExch. | Wiki | ArXiv | C4 |
|---|---|---|---|---|---|---|---|
| RedPajama gốc | 67.0% | 4.5% | 4.5% | 2.0% | 4.5% | 2.5% | 15.0% |
| Dynamic | 36.1% | 0.8% | 9.1% | 1.0% | 3.1% | 0.7% | 49.2% |

## 5. Phân tích / ablation

- **Targeted vs CoFi (Table 4):** CoFi (không đồng nhất giữa layer) có PPL tốt hơn một chút nhưng chậm hơn: 1.3B: PPL 9.1 / 51 tok/s (CoFi) so với 10.3 / 58 tok/s (targeted); 2.7B: 7.0 / 37 so với 7.7 / 43 (A100, batch 1, sinh tới 512 token). Targeted cần thêm ~0.5B token train tiếp để bắt kịp PPL của CoFi.
- **Targeted vs LLM-Pruner (Table 11-12):** LLM-Pruner ra cấu hình 32 layer, 7 head, intermediate 2201, hidden 4096 (1.6B); Sheared (1.3B) 24 layer, 16 head, intermediate 5504, hidden 2048. Sheared: 58 tok/s sinh, 139K tok/s train, PPL 6.85; LLM-Pruner: 43 tok/s, 83K tok/s, PPL 7.09 (sau ~6B token).
- **Chia ngân sách (Table 5, tổng 5B token):** tăng token cho pruning thì PPL sau prune giảm (12.99 -> 10.29 -> 9.01 -> 8.04 khi pruning dùng 0.2/0.4/0.8/1.6B) và PPL sau train tiếp cũng giảm nhẹ (7.46 -> 7.32 -> 7.23 -> 7.08). Họ vẫn chọn 0.4B vì pruning đắt hơn nhiều.
- **Dynamic loading** làm loss giảm đều giữa các domain và tăng downstream (Fig. 4-5). Chỉ loại các domain "dễ" khỏi giai đoạn prune không giải quyết được (Appendix F.8): chênh lệch loss đều hơn sau prune nhưng lại phân kỳ sau train tiếp.
- **Scaling reference vs source reference:** scaling tốt hơn một chút.
- **Prune từ LLaMA1 vs LLaMA2:** LLaMA2 hơn nhưng chênh lệch không lớn.
- **So với train tiếp một mô hình nhỏ có sẵn (F.7):** INCITE-Base-3B khởi đầu cao hơn nhưng bão hoà; Sheared-2.7B nhanh chóng vượt. Tác giả chú thích rằng nếu mô hình nhỏ có sẵn gần ngang mô hình nguồn thì nó có thể là điểm khởi đầu tốt hơn.
- **Pythia-410M -> 160M (F.5):** với 10B token đạt xấp xỉ Pythia-160M train từ đầu (43.51 vs 43.56); với 33B token đạt 45.78. Tác giả nói đây là kết quả sơ bộ.
- **Toán/code (F.3, Table 13):** yếu: GSM8K của Sheared-2.7B là 2.4-2.7 (EM) so với 13.7 của LLaMA2-7B; HumanEval Pass@1 3.7-4.9.

## 6. Giới hạn

*Tác giả nêu:*
- Phụ thuộc vào có sẵn mô hình và dữ liệu pre-training mã nguồn mở; domain không có trong dữ liệu thì khó hồi phục.
- Chỉ thử mô hình nguồn 7B (và LLaMA1-13B, Pythia-410M ở phụ lục); khẳng định "tổng quát hoá" chưa được kiểm chứng.
- Khả năng code/toán còn yếu (do ít dữ liệu code trong RedPajama và pruning).

*Chúng ta nhận xét:*
- Loss dùng khi train tiếp chỉ là LM loss trên one-hot token; không dùng logits của mô hình lớn. Paper nhắc distillation chỉ ở related work (cho rằng tốn chi phí chạy teacher); Minitron (39) sau đó cho thấy distillation giúp đáng kể.
- Mask được học ở bước pruning với chỉ 0.4B token; chưa có so sánh depth vs width có hệ thống (kiến trúc đích lấy từ mô hình có sẵn).
- Cần 16 GPU và 50B token: cùng bậc "huấn luyện lớn", không phải fine-tune nhẹ.

## 7. Liên hệ với các bài khác

- **17 CoFi:** cùng tác giả đầu; Sheared-LLaMA là bản "targeted" và quy mô LLM của CoFi (hard concrete + Lagrange).
- **26 LLM-Pruner, 25 ZipLM, 28 FLAP, 29 SliceGPT:** các phương pháp prune LLM với ít/không train tiếp; Sheared cho rằng thiếu bước continued pre-training thì mất nhiều.
- **11 Rethinking the value of pruning:** quan điểm "prune chỉ là tìm kiến trúc" so với "giữ trọng số làm khởi tạo"; F.7 và F.5 của Sheared nghiêng về phía trọng số được kế thừa có giá trị.
- **39-40 Minitron:** cùng pipeline prune + retrain, nhưng dùng distillation, importance dựa trên activation (không cần học mask) và nghiên cứu độ sâu vs độ rộng.
- **43 Pruning vs scratch (2026):** kiểm tra lại câu hỏi nền tảng của Sheared "prune có thật sự tiết kiệm hơn train từ đầu".

## 8. Ứng dụng cho dự án (suy luận)

- Bài học kiến trúc: đích nên là kiến trúc *đồng nhất, dense, chuẩn* (cùng số head mọi layer, intermediate chia hết cho 128...) để chạy nhanh trên GPU 8 GB, batch 1. Với LightOnOCR-3-0.8B (hybrid GatedDeltaNet + attention, ViT 12 block), nên tránh cấu hình lạ kiểu LLM-Pruner.
- Quan sát của dự án rằng "cắt neuron MLP hồi phục tệ hơn cắt layer" khớp với một điểm Sheared chưa phân tích: Sheared cắt cả hai nhưng không tách riêng tác động. Bài 39 mới làm điều đó.
- Dynamic batch loading có thể ánh xạ sang "domain" của OCR (bảng, chữ viết tay, công thức, tài liệu in, ngôn ngữ khác nhau): domain nào có loss/lỗi còn xa mô hình 4B thì lấy nhiều mẫu hơn. Đây là suy luận; Sheared chỉ kiểm chứng trên LM text thuần.
- Ngân sách: 50B token là không thực tế cho dự án nhỏ; dùng tinh thần "prune ngắn, train tiếp dài" với dữ liệu nhỏ hơn rất nhiều và có teacher 4B.
- Mask học được (hard concrete) cần train chung với trọng số trong giai đoạn prune ~5x chậm hơn; với mô hình 0.8B điều này khả thi hơn so với 7B.

## 9. Thí nghiệm gợi ý

1. Chạy bản nhỏ của dynamic batch loading trên tập OCR chia nhóm domain (in/tay/bảng/công thức): so sánh CER/WER cuối cùng với sampling đều, cùng ngân sách.
2. Đo trước/sau tại mỗi mức prune xem domain nào "mất" nhiều nhất (tương tự Fig. 4) để xem hiện tượng mất kiến thức không đều có xuất hiện trong VLM OCR không.
3. Với cùng kích thước đích ~0.3B, so một cấu hình đồng nhất chuẩn với một cấu hình không đồng nhất (số neuron MLP khác nhau theo layer) về độ trễ batch-1 thật trên GPU đích.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Structured pruning | Xoá nguyên head/layer/chiều, giữ ma trận dense |
| Targeted pruning | Ép kết quả khớp đúng kiến trúc đích cho trước |
| Hard concrete | Mask xấp xỉ liên tục, tập trung ở 0/1, tham số $\beta,\zeta,\gamma$ |
| Lagrange multipliers $\lambda,\phi$ | Phạt tuyến tính + bậc hai để $\sum z$ = số lượng đích |
| Continued pre-training | Tiếp tục train LM loss sau khi cắt để hồi phục |
| Dynamic batch loading | $w_t\propto w_{t-m}\exp(\max(\ell_t-\ell_{ref},0))$ |
| Scaling reference | Loss dự đoán $L(N,D)=E+A/N^\alpha+B/D^\beta$ theo domain |

## 11. Kết luận ngắn

Sheared-LLaMA cho thấy cắt LLaMA2-7B xuống 1.3B/2.7B rồi train tiếp 50B token cho mô hình tốt hơn các mô hình cùng cỡ train bằng nhiều token hơn nhiều lần, miễn là (a) kiến trúc đích chuẩn và (b) dữ liệu train tiếp được cân bằng theo domain. Điểm yếu để các bài sau cải thiện: loss chỉ là LM loss, chưa có distillation và chưa phân tích depth vs width.

Link gốc: https://arxiv.org/abs/2310.06694
