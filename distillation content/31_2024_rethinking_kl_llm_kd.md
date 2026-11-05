# Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models (2024)

**Tác giả:** Taiqiang Wu, Chaofan Tao, Jiahao Wang, Runming Yang, Zhe Zhao, Ngai Wong (HKU; Tsinghua; Tencent AI Lab).  
**Venue / arXiv:** arXiv:2404.02657 (v4, 08/12/2024). Bản trích xuất không nêu rõ venue.  
**Nguồn:** https://arxiv.org/abs/2404.02657  
**Chủ đề:** Forward KL vs reverse KL trong KD cho LLM; phản bác "FKL mean-seeking, RKL mode-seeking"; đề xuất Adaptive KL (AKL).  
**Vị trí trong lộ trình:** Đọc trước: 02 (Hinton KD), 28 (MiniLLM, lập luận RKL tốt hơn), 29 (GKD), 30 (DistiLLM). Bài này là bài "phản biện" cho cả nhóm: nó nói lý do chọn reverse KL bằng trực giác mode-seeking là không đúng cho LLM rời rạc, và đưa ra cách trộn FKL/RKL theo dữ liệu. Đọc tiếp: 32 (Speculative KD), 33 (TAID), 34 (DistiLLM-2).

> Ghi chú học tập bằng tiếng Việt, diễn giải lại, không phải bản dịch. Mục "Ứng dụng cho dự án" là suy luận của chúng ta, không phải kết luận của paper.

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

Với KD, độ lệch giữa phân phối teacher $p$ và student $q_\theta$ thường đo bằng KL:

- **Forward KL (FKL):** $FKL(p,q_\theta)=\sum_z p(z)\log\frac{p(z)}{q_\theta(z)}$ (Hinton).
- **Reverse KL (RKL):** $RKL(p,q_\theta)=\sum_z q_\theta(z)\log\frac{q_\theta(z)}{p(z)}$.

Niềm tin phổ biến (MiniLLM note 28, GKD note 29, DistiLLM note 30): FKL "mean-seeking" (student cố phủ mọi đỉnh của teacher, nên bị trải mỏng và sinh token đuôi vô nghĩa), còn RKL "mode-seeking" (student chọn một đỉnh) nên hợp với LLM hơn. Minh họa kinh điển: dùng một Gaussian để khớp hỗn hợp Gaussian, FKL cho đường ở giữa các đỉnh, RKL bám một đỉnh.

Paper chỉ ra niềm tin này dựa trên hai giả định: (1) student là Gaussian đơn đỉnh, (2) $p$ và $q$ liên tục. Trong LLM thì cả hai đều sai: $q_\theta$ là softmax trên từ vựng rời rạc $V$, không đơn đỉnh, và hoàn toàn có khả năng biểu diễn bất kỳ phân phối nào trên $V$ (qua logit tự do).

## 2. Ý tưởng chính

Vì student là softmax trên không gian rời rạc, cả FKL lẫn RKL đều chỉ có **một điểm cân bằng**: $q_\theta=p$. Nên về lý thuyết hai loss "cùng mục tiêu"; chúng hội tụ về teacher nếu đủ nhiều epoch (paper dùng toy ví dụ với hơn 50 epoch; thực tế LLM chỉ train cỡ 10-20 epoch). Điều khác biệt là **quá trình tối ưu giữa chừng**: ở các epoch đầu, FKL ưu tiên khớp **phần đầu (head)** của $p$ (các token xác suất lớn), còn RKL ưu tiên **phần đuôi (tail)** (các token xác suất nhỏ). Vì chỉ train ít epoch, ta nên trộn cả hai, với trọng số **thích nghi**: bên nào (head hay tail) đang lệch nhiều hơn thì cho loss tương ứng (FKL cho head, RKL cho tail) trọng số lớn hơn. Đó là **Adaptive KL (AKL)**.

## 3. Phương pháp chi tiết

### 3.1 Phân rã theo bước và gradient

KL cấp chuỗi phân rã thành tổng theo token: $J_{FKL}=\sum_{t=1}^{T}FKL\big(p(y_t|y_{<t}),q_\theta(y_t|y_{<t})\big)$, tương tự $J_{RKL}$. Gọi $z^p, z^q$ là logit teacher/student trước softmax. Gradient theo logit student $z^q_j$:

$$\frac{\partial FKL}{\partial z^q_j}=q_\theta(Y_j|y_{<t})-p(Y_j|y_{<t})$$

$$\frac{\partial RKL}{\partial z^q_j}=q_\theta(Y_j|y_{<t})\Big(\log\frac{q_\theta(Y_j|y_{<t})}{p(Y_j|y_{<t})}-RKL(p,q_\theta)\Big)$$

Điều kiện hội tụ cho cả hai là gradient bằng 0 với mọi $j$, tương đương

$$\boxed{q_\theta(Y_j|y_{<t})=p(Y_j|y_{<t})\quad\forall j}$$

Nghĩa là FKL và RKL **chia sẻ cùng mục tiêu tối ưu**.

### 3.2 Góc nhìn f-divergence

Cả hai đều là $f$-divergence $D_f(P\|Q)=\int f(dP/dQ)\,dQ$ với $f$ lồi, $f(1)=0$: FKL có $f(x)=x\log x$, RKL có $f(x)=-\log x$. Với FKL, $f(x)\to+\infty$ khi $x=p/q\to+\infty$, nên FKL tránh $q(z)\to0$ khi $p(z)$ lớn (phạt nặng khi student bỏ sót chỗ teacher có khối lượng). Với RKL, $f\to+\infty$ khi $x\to0^+$, nên RKL tránh $q(z)$ lớn khi $p(z)\to0$ ("zero forcing"). Hành vi "chọn một đỉnh" chỉ xảy ra khi $q$ là Gaussian đơn đỉnh (không đủ linh hoạt phủ nhiều đỉnh); nếu $q$ tự do thì cả hai hội tụ về $p$.

### 3.3 Khác biệt thực sự: tốc độ ở head vs tail

Trong FKL, token có $p(z)$ lớn có trọng số lớn trong tổng loss: ưu tiên khớp vùng head. Trong RKL, $\log\frac{q}{p}$ phình to khi $p(z)$ nhỏ: ưu tiên khớp vùng tail. Toy data kiểu long-tail (Fig. 3) cho thấy ở epoch 1, 5, 10 FKL khớp head tốt hơn, RKL khớp tail tốt hơn, và sau khoảng 250-300 epoch cả hai khớp teacher.

### 3.4 AKL

Chọn mặt nạ head $M\in\{0,1\}^V$ nhỏ nhất bao phủ khối lượng xác suất $\ge\mu$ của teacher ($\mu=0.5$ mặc định):

$$\min\sum_i M[i]\quad\text{s.t.}\quad\sum_j M[j]\,p(Y_j|y_{<t})\ge\mu$$

Giải bằng cách sắp xếp $p$ giảm dần và lấy các token đầu cho đến khi đủ $\mu$. Khoảng cách điểm-điểm $\epsilon(p,q)=|p-q|$. Gap theo vùng:

$$g_{head}=\sum_i M[i]\,\epsilon\big(p(Y_i),q_\theta(Y_i)\big),\qquad g_{tail}=\sum_i(1-M[i])\,\epsilon\big(p(Y_i),q_\theta(Y_i)\big)$$

$$\boxed{AKL(p,q_\theta)=\frac{g_{head}}{g_{head}+g_{tail}}\,FKL(p,q_\theta)+\frac{g_{tail}}{g_{head}+g_{tail}}\,RKL(p,q_\theta)}$$

Head lệch nhiều hơn thì FKL được trọng số lớn hơn, và ngược lại. Không thêm tham số học được (chỉ siêu tham số $\mu$).

```
Mỗi bước token t của mỗi mẫu:
  p = softmax(teacher logits), q = softmax(student logits)
  sắp xếp p giảm dần, M[i]=1 cho tập nhỏ nhất có tổng p >= mu
  g_head = sum_i M[i]*|p_i - q_i|;  g_tail = sum_i (1-M[i])*|p_i - q_i|
  w = g_head/(g_head+g_tail)         # coi là hằng, không lấy gradient qua w (suy luận từ công thức)
  loss_t = w*FKL(p,q) + (1-w)*RKL(p,q)
loss = tổng/trung bình theo token
```

Ghi chú: paper không nói rõ có dừng gradient qua trọng số $w$ hay không; trong pseudo-code trên đó là lựa chọn của chúng ta, cần kiểm tra code gốc.

Ví dụ tay: $V=4$, $p=(0.6,0.3,0.08,0.02)$, $\mu=0.5$ thì head chỉ gồm token đầu (0.6). Giả sử $q=(0.4,0.3,0.2,0.1)$: $g_{head}=0.2$, $g_{tail}=0+0.12+0.08=0.2$, nên $w=0.5$. Nếu $q$ gần $p$ ở tail nhưng lệch ở head, $w\to1$ (thiên về FKL).

## 4. Thí nghiệm và kết quả chính

**Thiết lập.** Dữ liệu instruction theo MiniLLM: databricks-dolly-15k (14K train, 500 val, 500 test). Hai cặp: GPT-2 1.5B→GPT-2 120M (batch 32, 20 epoch, LR 5e-4) và LLaMA 6.7B→TinyLLaMA 1.1B (batch 60, 10 epoch, LR 1e-5); độ dài input tối đa 512; khoảng 1 giờ trên 4 A100. Đánh giá ROUGE-L, 5 seed, trên Dolly (500), S-NI (1694), UnNI (10000). Không so với MiniLLM vì MiniLLM pretrain thêm trên OpenWebText và dữ liệu RoBERTa. SKL/SRKL dùng tỉ lệ trộn 0.1 theo DistiLLM; FKL+RKL là $0.5\,FKL+0.5\,RKL$.

Bảng 1 (ROUGE-L, rút gọn; độ lệch chuẩn bỏ):

| Phương pháp | GPT2 Dolly | S-NI | UnNI | TinyLLaMA Dolly | S-NI | UnNI |
|---|---|---|---|---|---|---|
| Teacher | 26.98 | 27.25 | 31.61 | 26.73 | 32.75 | 34.61 |
| SFT | 23.01 | 16.48 | 18.43 | 22.05 | 27.79 | 25.96 |
| SeqKD | 23.30 | 16.35 | 18.51 | 22.67 | 26.97 | 27.35 |
| FKL | 23.46 | 16.63 | 19.27 | 22.24 | 28.07 | 26.93 |
| RKL | 22.62 | 17.88 | 19.35 | 23.95 | 28.90 | 27.89 |
| SKL | 23.47 | 16.51 | 18.46 | 23.29 | 29.89 | 29.15 |
| SRKL | 23.25 | 17.54 | 19.31 | 22.09 | 29.60 | 28.81 |
| FKL+RKL | 23.36 | 17.83 | 20.37 | 24.08 | 30.98 | 30.48 |
| **AKL** | **23.88** | **19.15** | **21.97** | **24.40** | **31.37** | **31.05** |

Điểm đáng chú ý theo paper: khoảng cách FKL-RKL nhỏ (ví dụ GPT-2 Dolly: FKL 23.46 cao hơn RKL 22.62); chỉ cộng FKL+RKL đã tốt hơn từng cái; AKL tốt nhất ở mọi cột, với p-value so với FKL+RKL trên UnNI là 5e-9 (GPT-2) và 9e-5 (TinyLLaMA).

**Chi phí (LLaMA 6.7B→TinyLLaMA, A100 40G, Bảng 2):** SFT 31.2G/60 phút; FKL và RKL 35.6G/70 phút; FKL+RKL 37.5G/76 phút; AKL 40.3G/80 phút (thêm khoảng 2.8G và 4 phút so với FKL+RKL).

**Đánh giá GPT-4 (điểm /10, TinyLLaMA):** AKL cao hơn các baseline cả về diversity lẫn quality (số chính xác trong hình bị vỡ nên không nêu hết). **Người đánh giá (5 chuyên gia, win/tie/loss %):** ví dụ AKL vs SFT diversity 30.6/54.2/15.2, quality 46.7/43.0/10.3; AKL vs FKL+RKL diversity 31.9/55.6/12.5, quality 27.1/57.0/15.9; AKL vs Teacher quality 15.9/35.6/48.5 (teacher vẫn thắng về chất lượng). Tác giả nói AKL thắng hoặc hòa hơn 75% trường hợp với mọi baseline. Thêm đánh giá trên Winogrande, OpenBookQA, BoolQ, ARC (chi tiết ở phụ lục, không trích số ở đây).

## 5. Phân tích / ablation

- **AKL-r (đảo trọng số)**: gán trọng số FKL theo $g_{tail}$ thay vì $g_{head}$ thì kém hơn AKL (và kém cả FKL+RKL) trên cả ba tập ở GPT-2 (Fig. 4), chứng tỏ hướng gán trọng số là đúng.
- **Theo từng subtask Dolly (TinyLLaMA):** FKL tốt ở Closed QA, Summary; RKL tốt ở Brain Storming; FKL+RKL nằm giữa (ví dụ Classification 48.00 giữa FKL 47.01 và RKL 51.39); AKL thậm chí vượt teacher ở Information Extraction và Creative Writing.
- **Ngưỡng $\mu$:** có ablation ở Phụ lục E (không trích số).
- BERTScore cho kết luận cùng chiều (Phụ lục D).

## 6. Giới hạn

**Tác giả nêu:** chỉ thử GPT-2 và LLaMA 6.7B, chưa thử mô hình lớn hơn (như LLaMA-2 70B) do giới hạn tài nguyên.

**Chúng ta nhận xét (suy luận):**
- Cấu hình train ngắn (10-20 epoch) và chỉ một tập instruction nhỏ; lập luận "chỉ khác nhau ở giai đoạn đầu" là kết luận từ toy example và vài thí nghiệm, không phải chứng minh cho mạng sâu thật.
- Giả định khả năng biểu diễn đầy đủ của $q_\theta$ (logit tự do) chưa chắc đúng với student nhỏ bị cắt mạnh: dung lượng hạn chế vẫn có thể tạo trade-off thật giữa mean/mode (mâu thuẫn tiềm tàng với kết luận của GKD ở student nhỏ, chưa được kiểm tra).
- Là phân tích trên dữ liệu off-policy (huấn luyện trên dữ liệu cố định); tương tác với on-policy (GKD/DistiLLM) không được phân tích.
- Metric ROUGE-L khó phản ánh đúng chất lượng sinh; GPT-4/người đánh giá có số mẫu hạn chế.
- AKL cần sắp xếp $p$ theo từ vựng ở mỗi token: chi phí đáng kể với từ vựng lớn.

## 7. Liên hệ với các bài khác

- **28 MiniLLM, 29 GKD:** hai bài này chọn RKL/JSD dựa trên mode-seeking; bài này phản bác trực giác đó (nhưng GKD vẫn có bằng chứng thực nghiệm riêng, ví dụ instruction tuning).
- **30 DistiLLM:** AKL dùng SKL/SRKL làm baseline (mix 0.1); hai bài cùng nhận xét FKL/RKL đơn lẻ không đủ và cần "trộn".
- **33 TAID:** bài này cho thấy "trộn" có lợi vì train ngắn; TAID giải quyết khoảng cách capacity bằng trộn theo thời gian.
- **34 DistiLLM-2:** thêm hướng tương phản; bài này vẫn nằm trong khung "một loss cho mọi mẫu".
- **02 Hinton KD:** FKL chính là loss KD gốc; phân tích gradient $q-p$ chính là gradient quen thuộc của KL với softmax.

## 8. Ứng dụng cho dự án (suy luận)

- **Liên quan vừa phải.** Ngân sách phục hồi sau cắt thường ngắn (ít epoch), nên đúng chế độ mà paper bảo là FKL và RKL còn khác nhau. FKL+RKL (đơn giản, 0.5/0.5) là baseline gần như miễn phí; AKL là bản có thích nghi.
- Với OCR, phân phối token của teacher thường rất nhọn (gần one-hot) nên phần head chiếm hầu hết khối lượng, phần tail ít ý nghĩa; AKL có thể gần như luôn nghiêng về FKL. Đây là suy đoán cần đo (thử phân bố $g_{head}/g_{tail}$ theo bước).
- Ngưỡng $\mu=0.5$ có thể phải đổi (ví dụ 0.9) khi $p$ nhọn.
- Điểm nên mang về: đừng chọn RKL chỉ vì "mode-seeking tốt hơn"; đo trực tiếp CER trên dữ liệu OCR. Nếu khoảng cách capacity cực lớn (4B→0.1B) thì giả định biểu diễn đầy đủ của paper yếu đi.
- Liên hệ D_lin/D_res: paper không đề cập; ta không có cơ sở nối.

## 9. Thí nghiệm gợi ý

1. Với student 0.3B đã cắt, so FKL, RKL, FKL+RKL (0.5/0.5), AKL ($\mu\in\{0.5,0.9\}$) cùng ngân sách ngắn; log $g_{head}$ và $g_{tail}$ theo bước để xem tail có đáng kể không với OCR.
2. Lặp lại với train dài gấp 3-5 lần để kiểm chứng tuyên bố "cuối cùng FKL và RKL tiến gần nhau" trên mô hình đã cắt.
3. Đo riêng lỗi trên token hiếm (ký hiệu, chữ số, tên riêng) so với token phổ biến để xem loss nào bảo toàn tail tốt hơn.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| FKL / RKL | $\sum p\log(p/q)$ / $\sum q\log(q/p)$ |
| Gradient FKL theo logit | $q_j-p_j$ |
| Điều kiện hội tụ | $q=p$ cho cả FKL và RKL (cùng mục tiêu) |
| Head / tail | Token chiếm khối lượng $\ge\mu$ của $p$ / phần còn lại |
| Hành vi đầu train | FKL ưu tiên head, RKL ưu tiên tail |
| $g_{head}, g_{tail}$ | Tổng $|p-q|$ trên head / tail |
| AKL | $\frac{g_{head}}{g_{head}+g_{tail}}FKL+\frac{g_{tail}}{g_{head}+g_{tail}}RKL$ |
| Chi phí | +2.8G GPU và +4 phút so với FKL+RKL (TinyLLaMA) |

## 11. Kết luận ngắn

Với LLM (softmax rời rạc), FKL và RKL cùng hội tụ về teacher; khác biệt nằm ở tốc độ (FKL nhanh ở head, RKL nhanh ở tail) và vì ta train ít epoch nên nên kết hợp. AKL gán trọng số thích nghi theo độ lệch head/tail, không thêm tham số, và cho kết quả tốt nhất trong bảng ROUGE-L trên GPT-2 và TinyLLaMA. Thông điệp thực dụng: đừng tin tuyệt đối vào "reverse KL tốt hơn", hãy so sánh trực tiếp trên dữ liệu của mình.

Nguồn gốc: https://arxiv.org/abs/2404.02657
