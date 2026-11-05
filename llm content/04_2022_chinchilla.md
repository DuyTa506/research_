# Training Compute-Optimal Large Language Models (2022)

**Tác giả:** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre (DeepMind). Ba tác giả đầu và Sifre đóng góp ngang nhau.  
**Venue / arXiv:** arXiv:2203.15556 (v1, 29/03/2022); paper không nêu venue hội nghị.  
**Nguồn:** https://arxiv.org/abs/2203.15556  
**Chủ đề:** phân bổ tối ưu ngân sách compute giữa kích thước mô hình $N$ và số token huấn luyện $D$; mô hình Chinchilla 70B.  
**Vị trí trong lộ trình:** đọc sau 03 (Kaplan), vì bài này phản biện trực tiếp. Bổ sung: kết luận mới "model và data tăng cùng tỉ lệ", cùng bằng chứng thực nghiệm. Tiếp theo: 05 (LLaMA) và 06 (Mistral), các mô hình nhỏ huấn luyện nhiều token, và các bài về suy luận hiệu quả.

> Tài liệu học tập bằng tiếng Việt, ghi chú diễn giải, không phải bản dịch. Số liệu lấy từ paper trừ khi ghi "tự tính". Phần "Ứng dụng cho dự án" là suy luận của chúng ta. Đây là báo cáo dài (~15k từ): ghi chú tập trung vào phương pháp ước lượng, kết luận về phân bổ, và các kết quả tiêu biểu; các bảng đánh giá chi tiết (Pile, BIG-bench từng tác vụ, bias, toxicity) chỉ tóm tắt.

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

Bài 03 (Kaplan) kết luận rằng khi tăng compute 10 lần thì nên tăng kích thước mô hình 5.5 lần nhưng chỉ tăng số token 1.8 lần (cách diễn đạt của paper này). Làm theo đó, các LLM lớn nhất được huấn luyện với khoảng 300 tỷ token dù kích thước tăng vọt:

| Mô hình | Tham số | Token huấn luyện |
|---|---|---|
| LaMDA | 137 tỷ | 168 tỷ |
| GPT-3 | 175 tỷ | 300 tỷ |
| Jurassic | 178 tỷ | 300 tỷ |
| Gopher | 280 tỷ | 300 tỷ |
| MT-NLG 530B | 530 tỷ | 270 tỷ |
| **Chinchilla** | **70 tỷ** | **1.4 nghìn tỷ** |

Huấn luyện một LLM thường chỉ làm được một lần, và ngân sách (số accelerator x thời gian) thường biết trước. Nên câu hỏi then chốt là: **với tổng FLOPs cố định $C$, chọn $N$ và $D$ thế nào để loss nhỏ nhất?**

$$
N_{opt}(C),\,D_{opt}(C)=\arg\min_{N,D\ \text{s.t. FLOPs}(N,D)=C}L(N,D).
$$

(Giải nghĩa: $N$ số tham số, $D$ số token đã thấy; "compute-optimal" = tối ưu cho tổng huấn luyện, chưa tính chi phí suy luận.)

## 2. Ý tưởng chính

Tác giả huấn luyện hơn 400 mô hình (70 triệu đến hơn 16 tỷ tham số, 5 tỷ đến hơn 400 tỷ token) và ước lượng $N_{opt}$, $D_{opt}$ bằng **ba cách độc lập**. Cả ba cho cùng kết luận: $N_{opt}\propto C^{\approx0.5}$ và $D_{opt}\propto C^{\approx0.5}$, tức mỗi lần nhân đôi kích thước mô hình thì cũng phải nhân đôi số token. Hệ quả: các LLM đương thời **quá lớn và huấn luyện thiếu dữ liệu**. Để kiểm chứng, họ huấn luyện **Chinchilla** (70B, 1.4T token) bằng đúng ngân sách compute của Gopher (280B, 300B token); Chinchilla thắng Gopher, GPT-3, Jurassic-1, MT-NLG trên hầu hết tác vụ và còn rẻ hơn khi suy luận/fine-tune vì nhỏ hơn 4 lần.

## 3. Phương pháp chi tiết

### 3.1 Vì sao kết luận khác Kaplan (Mục 2)

Paper nêu hai khác biệt phương pháp:
1. **Lịch learning rate.** Kaplan dùng số token và lịch lr cố định cho mọi mô hình. Nhóm này thấy: đặt độ dài chu kỳ cosine **khớp với số token huấn luyện** cho loss cuối tốt nhất (Hình A1). Nếu dùng lịch cosine dài cố định tới 130B token thì loss trung gian (ở $D'\ll130$B) bị **đánh giá cao** so với mô hình thực sự huấn luyện với lịch ngắn hơn; điều này làm đánh giá thấp hiệu quả huấn luyện ít dữ liệu và góp phần dẫn đến kết luận "$N$ nên tăng nhanh hơn $D$". Khi cosine vượt quá số bước huấn luyện hơn 25% thì hiệu năng giảm rõ (Phụ lục B, D.1).
2. **Dải kích thước mô hình.** Chinchilla dùng mô hình đến 16B (đa số trên 500M tham số), vì thấy đường biên FLOP-loss hơi cong (Phụ lục E); Kaplan chủ yếu dưới 100M.

### 3.2 Đếm FLOPs

Paper tính FLOPs đầy đủ (kể cả embedding), hệ số 2 cho nhân-cộng, backward = 2 lần forward (Phụ lục F), nhưng so với xấp xỉ quen thuộc $C=6ND$ thì chênh lệch rất nhỏ (Bảng A4) nên dùng $\text{FLOPs}(N,D)\approx6ND$ trong suy diễn. Khác Kaplan: $N$ ở đây **bao gồm** embedding.

### 3.3 Cách 1: giữ cỡ mô hình, đổi số token

Với mỗi $N$ (70M đến hơn 10B) huấn luyện 4 độ dài (độ dài chu kỳ cosine khác nhau tới 16 lần, lr giảm 10 lần). Làm trơn và nội suy đường loss thành hàm FLOPs → loss. Tại 1500 mức FLOPs cách đều log, chọn mô hình có loss nhỏ nhất, từ đó có $(N,D)$ tối ưu theo $C$ (Hình 2). Fit luỹ thừa: $a=0.50$, $b=0.50$. Mọi điểm được chọn nằm trong 15% cuối quá trình huấn luyện, củng cố nguyên tắc "chu kỳ cosine ≈ độ dài huấn luyện".

### 3.4 Cách 2: IsoFLOP profiles

Cố định 9 mức FLOPs (từ $6\times10^{18}$ đến $3\times10^{21}$), với mỗi mức huấn luyện nhiều kích thước $N$ (tới 16B) sao cho $D=C/(6N)$ (độ dài cosine khớp). Vẽ loss cuối theo $N$: được **thung lũng** rõ ràng (Hình 3). Fit parabol để tìm $N$ tối ưu từng mức, rồi fit luỹ thừa: $a=0.49$, $b=0.51$.

### 3.5 Cách 3: hàm loss tham số

Mô hình hoá loss cuối như tổng ba số hạng (dựa trên phân rã rủi ro cổ điển):

$$
\boxed{\hat L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}}
$$

- $E$: loss của quá trình sinh lý tưởng, tương ứng entropy của văn bản tự nhiên.
- $A/N^\alpha$: một Transformer $N$ tham số huấn luyện hoàn hảo vẫn kém quá trình lý tưởng (hạn chế của không gian hàm).
- $B/D^\beta$: chưa hội tụ do chỉ có số bước hữu hạn, một epoch trên mẫu hữu hạn.

Fit bằng tối thiểu Huber ($\delta=10^{-3}$) giữa $\log\hat L$ và $\log L$ quan sát, dùng L-BFGS với lưới khởi tạo (để tránh cực tiểu địa phương). Huber chống ngoại lai; $\delta$ lớn hơn làm mô hình khớp quá vùng compute nhỏ. Kết quả (Phụ lục D.2):

$$
L(N,D)=1.69+\frac{406.4}{N^{0.34}}+\frac{410.7}{D^{0.28}},\quad E=1.69,\ A=406.4,\ B=410.7,\ \alpha=0.34,\ \beta=0.28.
$$

Với ràng buộc $6ND=C$, cực tiểu có dạng đóng (phương trình 4):

$$
N_{opt}(C)=G\Big(\frac{C}{6}\Big)^{a},\quad D_{opt}(C)=G^{-1}\Big(\frac{C}{6}\Big)^{b},\quad G=\Big(\frac{\alpha A}{\beta B}\Big)^{\frac{1}{\alpha+\beta}},\ a=\frac{\beta}{\alpha+\beta},\ b=\frac{\alpha}{\alpha+\beta}.
$$

Dễ thấy $a+b=1$, và vì $\alpha\approx\beta$ nên $a\approx b\approx0.5$. Với $\alpha=0.34$, $\beta=0.28$: $a\approx0.45$, $b\approx0.55$ (tự tính từ các hằng số làm tròn; paper báo cáo 0.46 và 0.54 từ fit đầy đủ). Các điểm có $C\le10^{21}$ có phần dư lớn hơn nên Huber coi như ngoại lai; do đường biên cong âm, cách 3 dự đoán mô hình tối ưu **nhỏ hơn** hai cách còn lại ở compute lớn.

### 3.6 Tổng hợp (Bảng 2)

| Cách | $a$ ($N_{opt}\propto C^a$) | $b$ ($D_{opt}\propto C^b$) |
|---|---|---|
| 1. Minimum over training curves | 0.50 (0.488, 0.502) | 0.50 (0.501, 0.512) |
| 2. IsoFLOP profiles | 0.49 (0.462, 0.534) | 0.51 (0.483, 0.529) |
| 3. Parametric modelling of the loss | 0.46 (0.454, 0.455) | 0.54 (0.542, 0.543) |
| Kaplan et al. (2020) | 0.73 | 0.27 |

(Trong ngoặc: phân vị 10 và 90 qua bootstrap, lấy mẫu 80% dữ liệu 100 lần.)

Pseudo-code để dùng kết quả cách 1 và 3:

```
def chinchilla_plan(C):                 # C: tổng FLOPs huấn luyện
    # Quy tắc thực hành rút từ Bảng 3: D ≈ 20 * N  (tự tính, xem dưới)
    N = (C / (6 * 20)) ** 0.5           # từ C = 6*N*D với D = 20*N
    D = 20 * N
    return N, D
# ví dụ: C = 5.76e23 (ngân sách Gopher) -> N ≈ 69e9, D ≈ 1.4e12  (khớp Chinchilla)

def parametric_loss(N, D):
    return 1.69 + 406.4 / N**0.34 + 410.7 / D**0.28   # trên MassiveText, tokenizer Gopher
```

Kiểm tra bằng số (tự tính): $N=\sqrt{5.76\times10^{23}/120}=\sqrt{4.8\times10^{21}}\approx6.9\times10^{10}$, $D=20N\approx1.39\times10^{12}$. Khớp Chinchilla (70B, 1.4T). Tỉ lệ ~20 token/tham số **không** được paper phát biểu như quy tắc; ta rút ra từ Bảng 3 (xem ngay dưới).

## 4. Thí nghiệm và kết quả chính

### 4.1 Bảng 3: ($N$, FLOPs, token) tối ưu theo cách 1

| Tham số | FLOPs | Token | Token/tham số (tự tính) |
|---|---|---|---|
| 400 triệu | $1.92\times10^{19}$ | 8.0 tỷ | 20 |
| 1 tỷ | $1.21\times10^{20}$ | 20.2 tỷ | 20 |
| 10 tỷ | $1.23\times10^{22}$ | 205.1 tỷ | 20.5 |
| 67 tỷ | $5.76\times10^{23}$ | 1.5 nghìn tỷ | 22 |
| 175 tỷ | $3.85\times10^{24}$ | 3.7 nghìn tỷ | 21 |
| 280 tỷ | $9.90\times10^{24}$ | 5.9 nghìn tỷ | 21 |
| 520 tỷ | $3.43\times10^{25}$ | 11.0 nghìn tỷ | 21 |
| 1 nghìn tỷ | $1.27\times10^{26}$ | 21.2 nghìn tỷ | 21 |
| 10 nghìn tỷ | $1.30\times10^{28}$ | 216.2 nghìn tỷ | 21.6 |

Cột "FLOPs (Gopher unit)": mô hình 67B ứng với 1 đơn vị Gopher ($5.76\times10^{23}$); GPT-3 175B tối ưu cần 6.7 đơn vị. Phần chữ của paper (tổng hợp các cách) nêu: mô hình 175B nên huấn luyện với $4.41\times10^{24}$ FLOPs và trên 4.2 nghìn tỷ token (khác chút so với Bảng 3); Gopher-like 280B tối ưu ở khoảng $10^{25}$ FLOPs với 6.8 nghìn tỷ token; trừ khi có $10^{26}$ FLOPs, mô hình 1 nghìn tỷ tham số khó là tối ưu. (Các cách 2 và 3 cho số khác một chút, xem Bảng A3.) Dự đoán cho ngân sách Gopher ($5.76\times10^{23}$): mô hình tối ưu nhỏ hơn 4 lần (khoảng 40 đến 70 tỷ) và huấn luyện nhiều hơn 4 lần token. Cách 3 ước lượng 40B (Hình 4); paper chọn 70B cho Chinchilla, tức đầu cao của dải 40 đến 70B, "vì lý do dữ liệu và hiệu quả tính toán".

Mô hình Chinchilla vs Gopher (Bảng 4): cùng ngân sách FLOPs, cùng kiến trúc cơ bản.

| | Layers | Heads | Key/Value size | $d_{model}$ | Max LR | Batch (token) |
|---|---|---|---|---|---|---|
| Gopher 280B | 80 | 128 | 128 | 16,384 | $4\times10^{-5}$ | 3M → 6M |
| Chinchilla 70B | 80 | 64 | 128 | 8,192 | $1\times10^{-4}$ | 1.5M → 3M |

FFN luôn $4\times d_{model}$. Khác biệt huấn luyện: dùng AdamW thay Adam (cải thiện loss và hiệu năng sau fine-tune; AdamW chỉ vượt Adam khoảng 80% đường cosine); tokenizer SentencePiece sửa đổi không NFKC (94.15% token giống Gopher, tốt hơn cho toán/hoá học); forward/backward bfloat16 nhưng giữ bản sao float32 của trọng số trong optimizer. Huấn luyện trên MassiveText với tỉ lệ nhóm dữ liệu hơi khác, JAX + Haiku, TPUv3/v4.

### 4.2 Kết quả đánh giá

| Tác vụ | Chinchilla | Gopher | Ghi chú |
|---|---|---|---|
| MMLU (5-shot, trung bình 57 tác vụ) | 67.6% (Bảng 6) | 60.0% | GPT-3 43.9%; chuyên gia người 89.8%; abstract ghi 67.5% (lệch nhỏ với bảng) |
| BIG-bench (62 tác vụ) | 65.1% | 54.4% | tăng 10.7%; kém hơn Gopher ở 4/62 tác vụ |
| Wikitext103 (perplexity) | 7.16 | 7.75 | cảnh báo rò rỉ dữ liệu do 4x dữ liệu |
| LAMBADA zero-shot | 77.4% | 74.5% | GPT-3 76.2%; MT-NLG 530B 76.6% |
| RACE-m / RACE-h few-shot | 86.8% / 82.3% | 75.1% / 71.6% | GPT-3: 58.1 / 46.8 (prompt khác) |
| HellaSwag / PIQA / Winogrande / SIQA / BoolQ (zero-shot) | 80.8 / 81.8 / 74.9 / 51.3 / 83.7 | 79.2 / 81.8 / 70.1 / 50.6 / 79.3 | MT-NLG 530B: 80.2 / 82.0 / 73.0 / - / 78.2 |
| TruthfulQA (0 / 5 / 10-shot) | 43.6 / 58.5 / 66.7 | 29.5 (0-shot), 43.7 (10-shot) | |
| Natural Questions 5-shot / 64-shot | 31.5% / 35.5% | 24.5% / 28.2% (Bảng 9) | text ghi Gopher 21% và 28%, lệch với bảng |
| TriviaQA (unfiltered, test) 0 / 5 / 64-shot | 67.0 / 73.2 / 72.3 | 52.8 / 63.6 / 61.3 | |
| Winogender (tất cả) | 78.3% | 71.4% | cải thiện không đều: nam +3.2%, nữ +8.3%, trung tính +9.2% |

Trên MMLU, Chinchilla tốt hơn Gopher ở 51/57 tác vụ, bằng ở 2, kém ở 4 (college_mathematics, econometrics, moral_scenarios, formal_logic); đạt trên 90% ở 4 tác vụ (high_school_gov_and_politics, international_law, sociology, us_foreign_policy). Trên The Pile, Chinchilla tốt hơn Gopher ở mọi tập con (bits-per-byte), và hơn Jurassic-1 trừ dm_mathematics và ubuntu_irc. Toxicity (25.000 mẫu, PerspectiveAPI): trung bình 0.087 (Chinchilla) so với 0.081 (Gopher), percentile 95 là 0.238 so với 0.230: khác biệt không đáng kể.

## 5. Phân tích / ablation

- **Ba cách đồng thuận:** các hệ số $a,b$ xấp xỉ 0.5 và rất khác Kaplan (0.73/0.27); đây là bằng chứng chính, vì ba phương pháp dùng các mô hình và cách fit khác nhau.
- **Kiểm chứng trực tiếp** ở quy mô lớn: Chinchilla vs Gopher cùng FLOPs; bằng chứng bổ sung là so sánh đối đầu ở $10^{21}$ FLOPs giữa kích thước theo Kaplan và theo đề xuất này (Phụ lục D.4), kích thước đề xuất thắng rõ.
- **Nhất quán giữa các bộ dữ liệu (Phụ lục C):** lặp lại IsoFLOP trên C4 và GitHub code cho kết luận tương tự (tăng $N$ và $D$ cân bằng).
- **Lịch lr quan trọng:** cosine dài hơn mục tiêu hơn 25% gây suy giảm; vì vậy mọi so sánh phải dùng độ dài khớp.
- **AdamW:** loss cuối tốt hơn Adam, dù chỉ bắt kịp ở khoảng 80% đường cosine (chú thích 8, Hình A7).
- **Độ cong:** đường biên FLOP-loss cong âm ở compute cao (Phụ lục E, Hình A5): fit riêng ba đoạn đầu/giữa/cuối cho độ dốc khác nhau, nên có thể vẫn đang **đánh giá quá cao** kích thước tối ưu cho các ngân sách rất lớn. Đây là lý do cách 3 dự đoán mô hình nhỏ hơn.
- **Tác động đến lợi ích hạ nguồn:** vì nhỏ hơn 4 lần, bộ nhớ và chi phí suy luận của Chinchilla thấp hơn Gopher tương ứng; tác giả nhấn mạnh năng lượng huấn luyện được "amortized" qua suy luận và fine-tuning.

## 6. Giới hạn

**Do tác giả nêu:**
- Chỉ có hai lần chạy cỡ lớn so sánh được (Chinchilla và Gopher), không có thí nghiệm trung gian ở quy mô giữa.
- Giả định đường biên là luỹ thừa; thực tế thấy độ cong (có thể vẫn đang ước lượng kích thước tối ưu quá lớn).
- Tất cả huấn luyện dưới một epoch; chưa xét chế độ nhiều epoch.
- Cần thận trọng với rò rỉ train/test khi so sánh trên các benchmark ngôn ngữ vì Chinchilla thấy 4 lần dữ liệu.
- Không đủ dữ liệu chất lượng cao: kết quả cho thấy cần các tập dữ liệu lớn hơn và có chất lượng cao mới tiếp tục scale được.
- Đánh giá bias và toxicity chỉ là mẫu nhỏ, không toàn diện; cải thiện bias không đều giữa các nhóm.

**Do chúng ta nhận thấy:**
- Tối ưu theo **compute huấn luyện**, không phải tổng chi phí vòng đời. Nếu mô hình phục vụ rất nhiều truy vấn, nên huấn luyện mô hình nhỏ hơn với nhiều token hơn mức này (đi theo hướng LLaMA, bài 05; ý này nằm ngoài paper, dù paper có nhắc lợi ích suy luận).
- Hệ số $A,B,\alpha,\beta,E$ gắn với MassiveText, tokenizer của Gopher và kiến trúc cụ thể; không chuyển trực tiếp sang dữ liệu khác (ví dụ OCR).
- Một số số liệu không đồng nhất giữa phần chữ và bảng (MMLU 67.5% so với 67.6%; NQ của Gopher). Ngoài ra bảng đối chiếu cách 3 (Bảng A3) có vài ô khó đọc trong bản trích văn bản nên ta chỉ dùng Bảng 3 (cách 1).
- Chỉ xét dense Transformer; MoE và retrieval chỉ nhắc ở phần liên quan.

## 7. Liên hệ với các bài khác

- **03** Kaplan: bài được phản biện; so sánh $0.73/0.27$ với $0.5/0.5$, và công thức $C\approx6ND$ lấy từ đó. Khác biệt chính (theo paper) nằm ở lịch lr và dải kích thước mô hình; ngoài ra quy ước $N$ cũng khác (Chinchilla đếm cả embedding, Kaplan thì không).
- **02** GPT-3: ví dụ điển hình của mô hình lớn huấn luyện 300B token (theo Bảng 1 của bài này, lẽ ra GPT-3 nên có nhiều token hơn nhiều).
- **05** LLaMA, **06** Mistral: áp dụng bài học này và đi xa hơn (huấn luyện mô hình nhỏ nhiều token vì tối ưu suy luận).
- **27, 28** MoE: Chinchilla nhắc rằng lợi ích tính toán của routed model giảm ở quy mô lớn (Clark et al. 2022).
- **44 đến 57** (suy luận hiệu quả, lượng tử hoá, nén KV): tiếp nối ý "mô hình nhỏ hơn thì rẻ hơn khi triển khai".

## 8. Ứng dụng cho dự án (suy luận)

1. **Ngân sách token cho học sinh.** Quy tắc ~20 token/tham số (tự rút từ Bảng 3) cho cỡ dữ liệu "tối ưu compute" nếu huấn luyện **từ đầu**: 0.8B → ~16B token, 0.3B → ~6B, 0.1B → ~2B (tự tính, nội suy từ Bảng 3: 0.4B ↔ 8B, 1B ↔ 20.2B). Khi **phục hồi bằng distillation** từ mô hình đã cắt thì không áp dụng trực tiếp (học sinh không bắt đầu từ ngẫu nhiên), nhưng cho biết thứ tự độ lớn: nếu bạn chỉ có vài trăm triệu token OCR thì dữ liệu vẫn là nút thắt chính theo luật này.
2. **Chi phí huấn luyện từ đầu để so sánh.** $6ND$ với $N=0.8\times10^9$, $D=16\times10^9$: $7.7\times10^{19}$ FLOPs (tự tính) ≈ 0.9 PF-day. Dùng làm mốc để biết cắt + distill rẻ hơn bao nhiêu so với huấn luyện lại.
3. **Đọc hàm tham số như bản đồ nút thắt.** Ví dụ số (tự tính, hằng số của paper, dữ liệu văn bản chung, **không** phải OCR): $N=0.8\times10^9$, $D=16\times10^9$ cho $\hat L\approx1.69+0.38+0.57=2.64$ nats. Số hạng dữ liệu (0.57) lớn hơn số hạng tham số (0.38): ở quy mô này, thêm dữ liệu hoặc tín hiệu dày hơn (distillation soft-label thay hard-label) hứa hẹn hơn thêm tham số. Đây là suy đoán về hướng ưu tiên, chưa được bài này kiểm chứng cho distillation.
4. **Vì sao cắt xuống 0.3B/0.1B là hợp lý về triển khai.** Paper nhấn mạnh mô hình nhỏ hơn rẻ hơn cho suy luận. Với batch-1 decode (bài 00, mục 10), thời gian mỗi token xấp xỉ $\text{byte trọng số}/\text{băng thông}$, vậy chi phí suy luận tỉ lệ thẳng với $N$; thu nhỏ $N$ 2.7 lần (0.8B → 0.3B) giảm xấp xỉ 2.7 lần thời gian lý thuyết (bỏ qua overhead và KV cache).
5. **Cảnh báo.** Chinchilla không nói gì về cắt tỉa, distillation, đa phương thức hay OCR; mọi quy tắc ở trên là mốc tham chiếu, không phải dự báo cho dự án.

## 9. Thí nghiệm gợi ý

1. **IsoFLOP thu nhỏ cho tác vụ OCR:** chọn 3 ngân sách $C$ nhỏ (ví dụ $10^{17}$ đến $10^{18}$ FLOPs), với mỗi ngân sách huấn luyện 4 kích thước học sinh (0.1, 0.2, 0.3, 0.5B sau cắt) với $D=C/(6N)$ và cosine khớp độ dài; vẽ loss theo $N$ để xem có thung lũng không. Kiểm tra xem đỉnh dịch chuyển thế nào so với quy tắc 20 token/tham số khi dùng distillation.
2. **Fit $\hat L=E+A/N^\alpha+B/D^\beta$ cục bộ** trên chuỗi các lần cắt + distill (nhiều $N$, nhiều lượng dữ liệu distill $D$), xem số hạng nào chiếm ưu thế; dùng Huber như paper để chống ngoại lai.
3. **Kiểm tra lịch lr:** so sánh chu kỳ cosine khớp độ dài distill với chu kỳ dài gấp 2 lần ở cùng số bước, để xác nhận hiệu ứng "cosine quá dài làm giảm loss cuối" có tồn tại trong bài toán phục hồi sau cắt.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / giá trị |
|---|---|
| Bài toán | $\min L(N,D)$ s.t. $6ND=C$ |
| Hàm loss tham số | $\hat L=E+A/N^\alpha+B/D^\beta$; $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$ |
| Nghiệm đóng | $N_{opt}=G(C/6)^a$, $D_{opt}=G^{-1}(C/6)^b$, $a=\beta/(\alpha+\beta)$, $b=\alpha/(\alpha+\beta)$ |
| Hệ số thực nghiệm | $a\approx0.50/0.49/0.46$, $b\approx0.50/0.51/0.54$ (ba cách) |
| Kaplan | $a=0.73$, $b=0.27$ |
| Quy tắc thực hành (tự rút) | $D\approx20N$ |
| Chinchilla | 70B tham số, 1.4T token, cùng FLOPs với Gopher (280B, 300B token) |
| Lịch lr | cosine khớp số token, giảm 10 lần |
| Kết quả nổi bật | MMLU 67.6% (Gopher 60.0%); BIG-bench 65.1% (Gopher 54.4%) |

## 11. Kết luận ngắn

Với compute cố định, tham số và token nên được tăng cùng tỉ lệ (khoảng 20 token mỗi tham số theo Bảng 3), và các LLM đương thời bị quá cỡ. Chinchilla 70B huấn luyện 1.4T token thắng Gopher 280B cùng ngân sách và rẻ hơn khi suy luận. Giá trị lâu dài là phương pháp (ba cách ước lượng, lịch lr khớp độ dài, hàm loss tham số) và thông điệp "dữ liệu cũng quan trọng ngang kích thước".

Nguồn gốc: https://arxiv.org/abs/2203.15556
