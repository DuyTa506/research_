# LLaVA-KD: A Framework of Distilling Multimodal Large Language Models (2024)

**Tác giả:** Yuxuan Cai, Jiangning Zhang, Haoyang He, Xinwei He, Ao Tong, Zhenye Gan, Chengjie Wang, Zhucun Xue, Yong Liu, Xiang Bai (HUST, Zhejiang Univ., Tencent Youtu Lab, HZAU).  
**Venue / arXiv:** arXiv:2410.16236 (v3, 3/7/2025). Paper không nêu venue trong phần text đã trích.  
**Nguồn:** https://arxiv.org/abs/2410.16236  
**Chủ đề:** Distill MLLM (mô hình ngôn ngữ-thị giác) lớn sang nhỏ mà không đổi kiến trúc: MDist (KL trên cả token response lẫn token visual), RDist (khớp ma trận tự tương quan giữa các visual token), và lịch huấn luyện 3 giai đoạn DPT → SFT → DFT.  
**Vị trí trong lộ trình:** Đọc sau 25 (LLaVA-MoD, đối thủ trực tiếp), 24 (CLIP-KD), 02 (Hinton KD), 12 (Relational KD) và 28 (MiniLLM, reverse KL). Bài này đưa ra hai thông điệp: (1) khi distill VLM, **thứ tự giai đoạn** và **distill cả token thị giác** quan trọng; (2) forward KL đơn giản không thua reverse KL/JSD trong bối cảnh này. Tiếp theo là nhóm KD cho LLM (28 trở đi).

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch nguyên văn paper. Số liệu lấy từ paper; bản trích không có phần Phụ lục (hyperparameter chi tiết) nên không ghi các số đó. Phần "Ứng dụng cho dự án" là suy luận của người ghi chú.

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

Một MLLM kiểu LLaVA-1.5 gồm: **visual encoder** (đóng băng) → **projector** (MLP 2 lớp) → **LLM**. Huấn luyện chuẩn có hai giai đoạn: **Pre-Training (PT)** chỉ train projector trên cặp ảnh-caption để căn chỉnh không gian thị giác với không gian embedding văn bản; **Supervised Fine-Tuning (SFT)** train projector + LLM trên dữ liệu instruction. Với MLLM nhỏ, quy trình này cho kết quả kém: ví dụ TinyLLaVA 4B đạt 65.0% nhưng giảm xuống 54.7% ở 0.5B (paper).

Các cách làm MLLM nhỏ trước đó: thay LLM bằng backbone nhỏ (LLaVA-Phi), thêm MoE (MoE-LLaVA), lọc dữ liệu (Bunny). LLaVA-MoD (25) là công trình KD đầu tiên cho s-MLLM, dùng MoE + preference distillation. Các phương pháp KD cho LLM (MiniLLM, DistiLLM...) chỉ chưng cất trong miền **văn bản**, bỏ qua tín hiệu từ token thị giác. LLaVA-KD tập trung vào **cách huấn luyện** chứ không đổi kiến trúc hay dữ liệu.

Khái niệm nền:
- **s-MLLM / l-MLLM:** MLLM nhỏ / lớn (student / teacher).
- **Visual token:** vector đầu ra của projector cho mỗi patch ảnh, đi vào LLM như token. Ở LLM, các vị trí này cũng có logit đầu ra trên từ vựng (dù không phải token "thật" để sinh).
- **Ma trận tự tương quan (self-correlation):** $R = Y Y^\top$, ma trận $N_p\times N_p$ đo quan hệ giữa mọi cặp token.

## 2. Ý tưởng chính

Hai quan sát: (1) căn chỉnh thị giác-ngôn ngữ là nền tảng, nên cho phép distill ngay trong giai đoạn pre-training (DPT); (2) nếu chỉ chen distill vào SFT thì lợi ích hạn chế, tốt hơn là SFT thường trước để student có năng lực cơ bản, rồi một giai đoạn **Distilled Fine-Tuning (DFT)** riêng để tinh chỉnh. Về loss, ngoài KL trên token response (như KD cho LLM), tác giả thêm KL trên **token visual** (MDist) và một loss **quan hệ**: ép ma trận tự tương quan của visual token student giống teacher (RDist). Student giữ nguyên kiến trúc dense.

## 3. Phương pháp chi tiết

### 3.1 Kiến trúc (teacher và student cùng thiết kế)

SigLIP-B/14@384 đóng băng (dùng chung), projector 2 lớp MLP+GELU, LLM Qwen1.5 hoặc Qwen2.5. Đầu vào LLM: $H=[H_v,H_t]$; chuỗi dự đoán $y=[y_p,y_v,y_r]$ (prompt, visual, response). Teacher và student **phải cùng họ LLM** để chung vocabulary.

### 3.2 Các loss

**Autoregressive (baseline):**

$$\mathcal{L}_{reg}=-\sum_{m=1}^{M}\log\phi_l(y_m\mid y_{<m})$$

(paper dùng ký hiệu $\phi$ cho cả teacher/student; trong Eq. 1 nói chung cho mô hình đang huấn luyện.)

**MDist – response:**

$$\mathcal{L}_{res}=\sum_{m=1}^{M}\sum_{j=1}^{V}\phi_l(Y_j\mid y_{<m})\log\frac{\phi_l(Y_j\mid y_{<m})}{\phi_s(Y_j\mid y_{<m})}$$

Đây là **forward KL** (teacher ở vị trí "p"), token-level, trên mọi vị trí response.

**MDist – visual:**

$$\mathcal{L}_{vis}=\sum_{k=1}^{K}\sum_{j=1}^{V}\phi_l(Y_j\mid y_{<k})\log\frac{\phi_l(Y_j\mid y_{<k})}{\phi_s(Y_j\mid y_{<k})}$$

Cùng dạng KL nhưng lấy tại $K$ vị trí của visual token (logit LLM tại vị trí ảnh). Hiểu đơn giản: student bắt chước cách teacher "dự đoán" trên chuỗi visual token.

**RDist – quan hệ giữa visual token:** lấy $y_v^s,y_v^t$ (đầu ra LLM tại vị trí visual của student/teacher), tính

$$R_v^s=y_v^s\otimes y_v^s\in\mathbb{R}^{N_p\times N_p},\quad R_v^t=y_v^t\otimes y_v^t$$

$$\boxed{\mathcal{L}_{rel}=1-\cos(R_v^s,R_v^t)=1-\frac{R_v^s\cdot R_v^t}{\|R_v^s\|\,\|R_v^t\|}}$$

Cosine được tính trên cả ma trận (dạng phẳng). Vì cần $N_p\times N_p$ cùng kích thước nên chỉ cần số visual token bằng nhau giữa hai mô hình (cùng visual encoder), không cần cùng chiều ẩn. Vì $R$ là Gram matrix nên không phụ thuộc chiều embedding; đây là điểm chung với Relational KD (12).

### 3.3 Ba giai đoạn

| Giai đoạn | Train gì | Dữ liệu | Loss |
|---|---|---|---|
| 1. DPT (Distilled Pre-Training) | chỉ projector (ViT, LLM student đóng băng) | LLaVA-1.5-558k caption | $\mathcal{L}_{DPT}=\mathcal{L}_{reg}+\alpha\mathcal{L}_{res}+\beta\mathcal{L}_{vis}+\gamma\mathcal{L}_{rel}$ |
| 2. SFT | projector + LLM | LLaVA-mix-665k | $\mathcal{L}'_{SFT}=\mathcal{L}_{reg}$ |
| 3. DFT (Distilled Fine-Tuning) | projector + LLM | LLaVA-mix-665k | $\mathcal{L}_{DFT}=\mathcal{L}_{reg}+\alpha'\mathcal{L}_{res}+\beta'\mathcal{L}_{vis}+\gamma'\mathcal{L}_{rel}$ |

Hệ số: $\alpha,\beta,\gamma$ (và dấu phẩy) = 1.0, 1.0, 0.5. Teacher được huấn luyện bằng PT+SFT thường (Mục 3.2 paper), vision encoder chung.

### 3.4 Pseudo-code

```
teacher = train_standard(PT, SFT)            # l-MLLM, frozen afterwards
student = init(small LLM, projector random)
# Stage 1: DPT  (only projector trainable)
for (img, caption) in caption_data:
    out_s, out_t = student(img, caption), teacher(img, caption)      # logits at visual+response positions
    L = CE(out_s.resp) + 1.0*KL(out_t.resp || out_s.resp) \
        + 1.0*KL(out_t.vis || out_s.vis) + 0.5*(1 - cos(gram(out_s.vis_hidden), gram(out_t.vis_hidden)))
# Stage 2: SFT   (projector + LLM): L = CE only, on instruction data
# Stage 3: DFT   (projector + LLM): same loss as stage 1, on instruction data
```

(Ghi chú: paper không nói rõ $y_v$ trong RDist là hidden state hay logit; "visual tokens generated by LLM" — người ghi chú chọn cách hiểu chung chung là đầu ra LLM tại vị trí visual.)

## 4. Thí nghiệm và kết quả chính

**Thiết lập:** SigLIP-B/14@384 + 2-layer MLP + Qwen1.5/Qwen2.5. Dữ liệu 1.2M (558k + 665k). 10 benchmark: VQAv2, GQA, VizWiz, SciQA, TextVQA, MME, MMB, MMB-CN, POPE, MMMU; Avg7 (bỏ VQAv2, POPE, MMMU để so với LLaVA-MoD) và Avg10. Baseline TinyLLaVA† là bản tự chạy lại với PT-SFT.

### Kết quả chính (Bảng 1, trích Avg10 / Avg7)

| Student | Baseline PT-SFT Avg10 | LLaVA-KD Avg10 | Avg7 (LLaVA-KD) |
|---|---|---|---|
| Qwen1.5-0.5B | 54.7 | 57.9 | 55.2 |
| Qwen2.5-0.5B | 56.5 | 58.9 | 56.7 |
| Qwen1.5-1.8B | 56.8 | 62.1 | 60.3 |
| Qwen2.5-1.5B | 64.0 | 65.0 | 63.9 |

So sánh trực tiếp: LLaVA-KD Qwen1.5-0.5B hơn LLaVA-MoD cùng LLM 1.1% Avg7 (55.2 vs 54.1) với 1.2M mẫu so với 5M; hơn SPHINX-Tiny (15M mẫu) 4.0% Avg7 (paper). Với 1.8B: hơn LLaVA-MoD 0.4% và Imp-2B 1.4% Avg7 (60.3 so với 59.9 và 58.9). Cải thiện so với TinyLLaVA: +3.2% và +2.4% Avg10 ở 0.5B (Qwen1.5/Qwen2.5), +5.3% và +1.0% ở 1.8B/1.5B (text paper). Lưu ý Bảng 1 có vài dòng LLaVA-MoD Avg10 thiếu giá trị; cột Avg7 của bảng 1 lấy từ cột "Avg7" trong paper.

### Xác minh trên khung MobileVLM (Bảng 7, Avg10)

Teacher MobileLLaMA 2.7B: 55.2; student 1.7B: PT-SFT 48.8, DPT-SFT 50.5, DPT-SFT-DFT 53.4 (+4.6 so với baseline).

## 5. Phân tích / ablation

Tất cả ablation dùng student Qwen1.5-0.5B, Avg10, teacher Qwen1.5-4B.

**Chiến lược giai đoạn (Bảng 2):**

| Lịch | Avg10 |
|---|---|
| PT-SFT (baseline) | 54.7 |
| DPT-SFT | 55.6 |
| PT-DFT | 55.8 |
| DPT-DFT | 55.9 |
| PT-SFT-DFT | 56.6 |
| **DPT-SFT-DFT** | **57.9** |
| DPT-DFT-DFT | 58.0 (tốn thêm 120 GPU giờ) |

Bỏ SFT (DPT-DFT) thì kém hẳn (55.9); DPT cộng DFT cộng SFT cho +1.3 so với PT-SFT-DFT. DPT-SFT-DFT có đánh đổi hiệu năng/tính toán tốt nhất.

**MDist vs RDist (Bảng 3, 6 hàng; thứ tự cột bị mangled, gán nhãn theo văn bản):** ở DPT riêng lẻ, RDist (55.5) hơn MDist (55.1), kết hợp 55.6; ở DFT, MDist (57.7) hơn RDist (57.0), kết hợp 57.9. Tác giả: MDist hiệu quả hơn ở DFT vì khớp trực tiếp phân phối đầu ra đa phương thức.

**Mục tiêu distill (Bảng 4):** thêm token visual có lợi nhất: DPT: response 54.9, +prompt 55.0, +visual 55.1, cả ba 54.6; DFT: response 57.2, +prompt 56.9, **+visual 57.7**, cả ba 57.1. Khác LLaVA-MoD (chỉ response).

**Loại divergence (Bảng 5, PT-DFT):** FKL 55.5, RKL 55.0, JSD 55.5, phương pháp của paper 55.8. Chênh lệch nhỏ (0.3–0.8).

**So với pipeline LLaVA-MoD (Bảng 6):** mimic distillation (MD) 56.9; MD+preference distillation (PD) 55.1; LLaVA-KD 57.9. Phù hợp quan sát của LLaVA-MoD rằng PD giảm ảo giác nhưng hạ khả năng hiểu.

**Kích thước teacher (Bảng 8, Avg10, student 0.5B):** Qwen1.5-4B teacher cho 57.9 > Qwen1.5-7B 57.4; Qwen2.5-3B cho 58.9 > Qwen2.5-7B 58.3. Teacher lớn hơn (7B) lại kém hơn, nhất quán với CLIP-KD (24) và khái niệm capacity gap (tác giả viện dẫn "diminishing returns" và "alignment bottleneck", không chứng minh).

## 6. Giới hạn

**Tác giả nêu:** teacher và student phải cùng họ LLM (vocabulary chung); hướng tương lai là kết hợp MLLM dị chủng và xử lý capacity mismatch.

**Người ghi chú nhận xét:**
- Một số khác biệt rất nhỏ (0.3–1.0 điểm Avg), trên một seed; không có khoảng tin cậy.
- Ablation chủ yếu ở student 0.5B; hiệu lực khi student lớn hơn ít được kiểm tra (gain 1.0 ở Qwen2.5-1.5B).
- Chi phí huấn luyện (cần chạy teacher 4B/7B forward trong hai giai đoạn) không được định lượng ngoài một con số 120 GPU giờ.
- Nhiều chữ số trong bảng PDF bị mangled; Bảng 3 không đọc chắc ánh xạ hàng-cột.
- Không phân tích chất lượng/hành vi OCR chi tiết ngoài TextVQA.
- Không tách riêng ảnh hưởng của dữ liệu (1.2M) so với phương pháp trong so sánh với các mô hình dùng dữ liệu khác.

## 7. Liên hệ với các bài khác

- **25 LLaVA-MoD:** đối thủ trực tiếp; LLaVA-KD cho rằng dense + 3 giai đoạn + distill visual tốt hơn MoE + preference trong benchmark hiểu.
- **24 CLIP-KD:** cùng quan sát teacher lớn không luôn tốt hơn; LLaVA-KD cites CLIP-KD cho điều này.
- **12 Relational KD:** RDist là phiên bản "relational" cho visual token (khớp Gram matrix thay vì khoảng cách cặp).
- **28 MiniLLM, 30 DistiLLM, 31 Rethinking KL:** Bảng 5 so FKL/RKL/JSD; kết quả nói RKL không vượt FKL ở đây.
- **41 Capacity gap law, 39 Small models struggle...:** Bảng 8.
- **08 Teacher assistant:** gợi ý dùng teacher trung gian.
- **23 AM-RADIO:** distill encoder thị giác; ở LLaVA-KD encoder bị đóng băng nên không distill phần này.

## 8. Ứng dụng cho dự án (suy luận)

Mức liên quan: **cao**, vì cấu trúc LLaVA-KD (ViT đóng băng + projector + LLM, teacher cùng họ, student nhỏ hơn, dense) giống kịch bản LightOnOCR pruned (dense) + teacher 4B cùng họ.

- **Thứ tự giai đoạn:** sau prune, tương đương "SFT khôi phục" rồi "DFT"; kết quả Bảng 2 gợi ý nên có một pha CE thường trước pha distill đầy đủ, chứ không phải chỉ distill ngay.
- **Distill token visual:** nếu ViT/projector của student cũng bị prune (ViT 12 block), thêm KL tại vị trí visual token (ở LLM), và RDist (Gram matrix visual token) là chỉ dẫn rẻ để giữ quan hệ không gian. Với OCR, quan hệ giữa các patch (dòng chữ, bố cục) quan trọng; đây là suy luận, chưa có bằng chứng trong paper (Bảng 4 cũng cho thấy kết hợp mọi token lại tệ hơn).
- **Teacher size:** teacher 4B so với student 0.1B có thể bị capacity gap; thử teacher trung gian hoặc student 0.3B rồi 0.1B nối tiếp.
- **Forward KL đủ tốt:** có thể giữ forward KL chuẩn làm baseline thay vì vội dùng reverse KL.
- **GatedDeltaNet:** paper không đụng linear attention; việc RDist dùng đầu ra LLM tại visual token có thể khác trong decoder hybrid (suy luận, cần kiểm chứng).
- Liên hệ D_lin/D_res: DFT sau SFT ngầm tách "phục hồi tuyến tính + CE" khỏi "nội suy phần dư của teacher", nhưng paper không đo như vậy.

## 9. Thí nghiệm gợi ý

1. Với student 0.3B sau prune: so sánh lịch CE-only → CE+KD, KD-only, và KD ngay từ đầu (tương tự Bảng 2) trên CER/WER.
2. Thêm KL tại vị trí visual token và RDist ($\gamma=0.5$); ablation từng thành phần, kèm đo khoảng cách Gram matrix teacher/student trước-sau.
3. So teacher 4B với teacher 1–2B (trung gian) cho student 0.1B để kiểm tra capacity gap trong OCR.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý | Ghi chú |
|---|---|---|
| $\mathcal{L}_{res}$ | forward KL(teacher‖student) tại vị trí response | token-level |
| $\mathcal{L}_{vis}$ | forward KL tại vị trí visual token | khác biệt so với KD cho LLM |
| $\mathcal{L}_{rel}$ | $1-\cos(Y_sY_s^\top,\,Y_tY_t^\top)$ | RDist, $N_p\times N_p$ |
| DPT | pre-train có distill (chỉ projector) | $\alpha=\beta=1,\gamma=0.5$ |
| SFT | CE thuần | giữa DPT và DFT |
| DFT | fine-tune có distill | giống loss DPT |
| Cặp teacher–student | cùng họ LLM, cùng SigLIP | vocabulary chung |

## 11. Kết luận ngắn

LLaVA-KD cho thấy với MLLM, **khi nào** distill (DPT-SFT-DFT) và **distill ở đâu** (cả token visual, cộng quan hệ giữa chúng) quan trọng hơn việc đổi kiến trúc: student 0.5B tăng từ 54.7 lên 57.9 Avg10 với 1.2M mẫu. Hạn chế: chênh lệch nhỏ, cần teacher cùng họ, teacher quá lớn lại tệ hơn.

Link gốc: https://arxiv.org/abs/2410.16236
