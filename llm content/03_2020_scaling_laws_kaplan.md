# Scaling Laws for Neural Language Models (2020)

**Tác giả:** Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (Johns Hopkins University, OpenAI).  
**Venue / arXiv:** arXiv:2001.08361 (v1, 23/01/2020); paper không nêu venue hội nghị.  
**Nguồn:** https://arxiv.org/abs/2001.08361  
**Chủ đề:** quy luật luỹ thừa giữa loss của Transformer LM và số tham số $N$, lượng dữ liệu $D$, compute $C$; overfitting; batch size tới hạn; phân bổ compute tối ưu.  
**Vị trí trong lộ trình:** đọc sau 01 (kiến trúc) và 02 (GPT-3, ứng dụng trực tiếp tư tưởng "model lớn"). Bài này đặt ra khung "loss là hàm luỹ thừa của quy mô". Tiếp theo là 04 (Chinchilla), sửa lại kết luận về phân bổ model/data.

> Tài liệu học tập bằng tiếng Việt, ghi chú diễn giải, không phải bản dịch. Số liệu lấy từ paper trừ khi ghi "tự tính". Phần "Ứng dụng cho dự án" là suy luận của chúng ta. Đây là paper dài (~15k từ): ghi chú tập trung vào các định luật chính, công thức và hệ quả thực hành.

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

Đến 2019, các mô hình ngôn ngữ ngày càng lớn (GPT-2, BERT, ...) nhưng quyết định "huấn luyện model cỡ nào, bao nhiêu dữ liệu, bao lâu" chủ yếu dựa trên trực giác. Mỗi lần huấn luyện một mô hình lớn rất đắt, nên cần **dự đoán được** kết quả từ các lần chạy nhỏ.

Khái niệm cần nhớ (xem thêm bài 00):
- **Loss** ở đây là cross-entropy $L$ (nats/token) trên dữ liệu kiểm tra, trung bình trên ngữ cảnh 1024 token.
- **Luật luỹ thừa (power law)** $y=(x_c/x)^{\alpha}$: vẽ trên log-log là đường thẳng với độ dốc $-\alpha$; nhân $x$ lên $k$ lần thì $y$ nhân với $k^{-\alpha}$.
- **Compute** $C$ đo bằng PF-days: $1\,\text{PF-day}=10^{15}\times24\times3600=8.64\times10^{19}$ FLOPs.

Câu hỏi: loss phụ thuộc thế nào vào kiến trúc, kích thước mô hình, kích thước dữ liệu, và compute? Có thể phân bổ ngân sách compute cố định một cách tối ưu không?

## 2. Ý tưởng chính

Tác giả huấn luyện rất nhiều Transformer decoder-only ở nhiều kích thước (768 đến 1.5 tỷ tham số không tính embedding), nhiều kích thước dữ liệu (22 triệu đến 23 tỷ token), và thấy loss giảm theo **luật luỹ thừa mượt** theo từng yếu tố (khi hai yếu tố kia không là nút thắt), kéo dài hơn 6 đến 7 bậc độ lớn. Hình dạng mô hình (sâu hay rộng, số head) ít quan trọng; **quy mô** mới quan trọng. Từ các luật này họ suy ra: với compute cố định, nên **tăng kích thước mô hình thật nhanh, tăng dữ liệu rất chậm, và dừng huấn luyện rất sớm (xa hội tụ)**. Kết luận "big models may be more important than big data" ảnh hưởng mạnh đến GPT-3 (và bị Chinchilla, bài 04, sửa lại).

## 3. Phương pháp chi tiết

### 3.1 Thiết lập

- Dữ liệu: **WebText2** (mở rộng WebText: link ra từ Reddit đến 10/2018, tối thiểu 3 karma): 20.3M tài liệu, 96 GB, $1.62\times10^{10}$ từ, $2.29\times10^{10}$ token; giữ $6.6\times10^8$ token làm tập kiểm tra. Cũng kiểm tra trên Books Corpus, Common Crawl, English Wikipedia, Internet Books.
- Tokenizer: BPE, $n_{vocab}=50257$. Ngữ cảnh $n_{ctx}=1024$ (đa số).
- Mô hình: Transformer decoder-only; ngoài ra có LSTM và Universal Transformer để so.
- Huấn luyện: Adam, $2.5\times10^5$ bước, batch 512 chuỗi x 1024 token ($\approx2^{19}$ token); các mô hình >1B dùng Adafactor; lr: warmup 3000 bước rồi cosine về 0. Dropout 10% khi nghiên cứu overfitting, kèm early stopping.

### 3.2 Tham số và FLOPs (Mục 2.1, Bảng 1)

**Quan trọng:** $N$ trong paper là số tham số **không tính embedding** (không có $n_{vocab}d_{model}$ và vị trí). Với $d_{attn}=d_{ff}/4=d_{model}$:

$$
\boxed{N\approx2\,d_{model}\,n_{layer}\,(2d_{attn}+d_{ff})=12\,n_{layer}\,d_{model}^2}
$$

FLOPs forward mỗi token (cộng nhân = 2 phép):

$$
C_{forward}\approx2N+2\,n_{layer}\,n_{ctx}\,d_{model}.
$$

Khi $d_{model}>n_{ctx}/12$, thành phần phụ thuộc ngữ cảnh nhỏ nên bỏ. Cộng backward (~2 lần forward):

$$
\boxed{C\approx6N\,B\,S=6ND\ \ (\text{FLOPs trên }D=BS\text{ token})}
$$

| Phép toán | Tham số | FLOPs mỗi token (Bảng 1) |
|---|---|---|
| Embed | $(n_{vocab}+n_{ctx})d_{model}$ | $4d_{model}$ |
| Attention QKV | $n_{layer}d_{model}3d_{attn}$ | $2n_{layer}d_{model}3d_{attn}$ |
| Attention mask | không | $2n_{layer}n_{ctx}d_{attn}$ |
| Attention project | $n_{layer}d_{attn}d_{model}$ | $2n_{layer}d_{attn}d_{embd}$ |
| Feed-forward | $n_{layer}2d_{model}d_{ff}$ | $2n_{layer}2d_{model}d_{ff}$ |
| De-embed | không | $2d_{model}n_{vocab}$ |
| Tổng (không embedding) | $N=2d_{model}n_{layer}(2d_{attn}+d_{ff})$ | $C_{forward}=2N+2n_{layer}n_{ctx}d_{attn}$ |

### 3.3 Các định luật (Mục 1.2 và Phụ lục A)

Ba luật cơ bản khi chỉ một yếu tố là nút thắt:

$$
L(N)=\Big(\frac{N_c}{N}\Big)^{\alpha_N},\ \alpha_N\approx0.076,\ N_c\approx8.8\times10^{13};
$$
$$
L(D)=\Big(\frac{D_c}{D}\Big)^{\alpha_D},\ \alpha_D\approx0.095,\ D_c\approx5.4\times10^{13}\ \text{token};
$$
$$
L(C_{min})=\Big(\frac{C_c^{min}}{C_{min}}\Big)^{\alpha_C^{min}},\ \alpha_C^{min}\approx0.050,\ C_c^{min}\approx3.1\times10^{8}\ \text{PF-days}.
$$

Điều kiện: $L(N)$ là mô hình hội tụ trên dữ liệu đủ lớn; $L(D)$ là mô hình đủ lớn với early stopping; $L(C_{min})$ là compute dùng tối ưu, dữ liệu đủ, mô hình kích thước tối ưu và batch nhỏ. Doubling $N$ làm loss nhân với $2^{-0.076}\approx0.95$ (paper ghi 0.95; tự tính 0.9487). Các giá trị $N_c,D_c,C_c$ phụ thuộc tokenizer nên không có ý nghĩa tuyệt đối.

**Hợp nhất $N$ và $D$** (phương trình 1.5/4.1):

$$
\boxed{L(N,D)=\left[\Big(\frac{N_c}{N}\Big)^{\alpha_N/\alpha_D}+\frac{D_c}{D}\right]^{\alpha_D}}
$$

Ba nguyên tắc chọn dạng này: (1) đổi từ vựng/tokenization chỉ nhân loss với hệ số, có thể hấp thụ vào $N_c,D_c$; (2) $D\to\infty$ về $L(N)$ và $N\to\infty$ về $L(D)$; (3) giải tích theo $1/D$ (khai triển nguyên). Fit 4 tham số (Bảng 2): $\alpha_N=0.076$, $\alpha_D=0.103$, $N_c=6.4\times10^{13}$, $D_c=1.8\times10^{13}$ (hơi khác các giá trị riêng lẻ ở trên vì fit đồng thời).

Mức overfitting $\delta L=L(N,D)/L(N,\infty)-1$ chỉ phụ thuộc tổ hợp $N^{\alpha_N/\alpha_D}/D$ (Hình 9). Để overfitting nằm trong ngưỡng nhiễu seed (~0.02):

$$
\boxed{D\gtrsim(5\times10^3)\,N^{0.74}}
$$

tức mỗi lần tăng $N$ lên 8 lần chỉ cần tăng $D$ khoảng 5 lần (tự tính: $8^{0.74}\approx4.7$).

**Loss theo $N$ và số bước** (dữ liệu vô hạn):

$$
L(N,S_{min})=\Big(\frac{N_c}{N}\Big)^{\alpha_N}+\Big(\frac{S_c}{S_{min}}\Big)^{\alpha_S},\quad\alpha_N=0.077,\ \alpha_S=0.76,\ N_c=6.5\times10^{13},\ S_c=2.1\times10^3.
$$

### 3.4 Batch size tới hạn (Mục 5.1)

**Critical batch size** $B_{crit}$ (McCandlish et al. 2018): batch lớn tới mức tăng thêm cho lợi ích giảm dần. Mô hình bước/mẫu: $(S/S_{min}-1)(E/E_{min}-1)=1$, $B_{crit}=E_{min}/S_{min}$. Nó chỉ phụ thuộc vào loss mục tiêu:

$$
B_{crit}(L)=\frac{B_*}{L^{1/\alpha_B}},\quad B_*\approx2\times10^{8}\ \text{token},\ \alpha_B\approx0.21.
$$

Huấn luyện tại $B_{crit}$ tốn $2S_{min}$ bước và $2E_{min}$ mẫu. Các đại lượng hiệu chỉnh:

$$
S_{min}(S)=\frac{S}{1+B_{crit}(L)/B},\qquad C_{min}(C)=\frac{C}{1+B/B_{crit}(L)}.
$$

Paper cho: $B_{crit}$ khoảng 1 đến 2 triệu token tại hội tụ ở mô hình lớn nhất, và gấp đôi khi loss giảm khoảng 13%.

### 3.5 Phân bổ compute tối ưu (Mục 6 và Phụ lục B)

Thay $S=C/(6NB)$ vào $L(N,S)$ và tối ưu theo $N$ với $C$ cố định (đặt $\partial L/\partial N|_C=0$). Kết quả đóng:

$$
\alpha_C^{min}=\frac{1}{1/\alpha_S+1/\alpha_B+1/\alpha_N},\qquad N\propto C^{\alpha_C^{min}/\alpha_N},\ B\propto C^{\alpha_C^{min}/\alpha_B},\ S\propto C^{\alpha_C^{min}/\alpha_S},\ D=B\cdot S.
$$

Thay giá trị: $\alpha_C^{min}=1/(1.316+4.762+13.158)\approx0.052$ (tự tính). Thực nghiệm (Bảng 6):

$$
\boxed{N_{opt}\propto C_{min}^{0.73},\quad B\propto C_{min}^{0.24},\quad S_{min}\propto C_{min}^{0.03},\quad D_{opt}\propto C_{min}^{0.27}}
$$

với $N_e=1.3\times10^9$, $B_e=2.0\times10^6$ token, $S_e=5.4\times10^3$, $D_e=2\times10^{10}$ token. **Tăng compute 10 lần** thì: $N$ tăng $10^{0.73}\approx5.4\times$, batch $\approx1.7\times$, số bước gần như không đổi ($10^{0.03}\approx1.07$), dữ liệu $\approx1.9\times$ (tự tính). Paper nói hầu hết phần tăng đi vào mô hình lớn hơn, phần tăng dữ liệu chủ yếu dùng cho batch lớn hơn (song song).

Hệ quả "dừng sớm": huấn luyện tối ưu compute dừng khi loss còn cao hơn loss hội tụ khoảng $\alpha_N/\alpha_S\approx10\%$ (phương trình B.5). So với huấn luyện kiểu "gần hội tụ" ($f'=2\%$), huấn luyện hiệu quả dùng 2.7 lần tham số, 7.7 lần ít bước, 65% ít compute hơn để đạt cùng loss (B.3).

Mô hình không tối ưu (B.4): mô hình có kích thước từ 0.6 đến 2.2 lần kích thước tối ưu chỉ tốn thêm khoảng 20% compute; mô hình lớn hơn 2.2 lần cần ít hơn 45% số bước, tốn 20% thêm compute. Paper chỉ ra rằng chọn mô hình nhỏ hơn có ích khi tính chi phí suy luận.

Pseudo-code dự báo cho một ngân sách:

```
def plan(C_pfdays):                         # C_min, đã điều chỉnh về B << B_crit
    N = 1.3e9 * C_pfdays**0.73              # tham số không embedding
    B = 2.0e6 * C_pfdays**0.24              # token mỗi batch
    S = 5.4e3 * C_pfdays**0.03              # số bước (cận dưới)
    D_epoch1 = 2e10 * C_pfdays**0.27        # token nếu 1 epoch
    L = (3.1e8 / C_pfdays)**0.050           # loss dự báo (nats/token, WebText2, BPE GPT-2)
    return N, B, S, D_epoch1, L
```

## 4. Thí nghiệm và kết quả chính

Phạm vi: hơn tám bậc độ lớn $C_{min}$, sáu bậc $N$, hơn hai bậc $D$. Các điểm chính (copy từ paper):

| Phát hiện | Nội dung |
|---|---|
| Hình dạng ít quan trọng | Với $N$ cố định, loss chỉ dao động vài % trên dải lớn của tỉ lệ $d_{ff}/d_{model}$, $d_{model}/n_{layer}$ (aspect ratio), $d_{model}/n_{head}$. Aspect ratio thay đổi tới 40 lần ảnh hưởng nhẹ; $(n_{layer},d_{model})=(6,4288)$ đạt loss trong vòng 3% so với $(48,1600)$. |
| Chỉ tính non-embedding | Gộp embedding vào $N$ làm xu hướng bị nhiễu theo số tầng; loại embedding thì mọi độ sâu hội tụ về một đường. Hàm ý: ma trận embedding có thể nhỏ hơn mà không hại. |
| Transfer | Loss trên phân phối khác (Books, Wikipedia, Common Crawl, Internet Books) tăng song song với loss WebText2, chênh lệch gần hằng số; chỉ phụ thuộc loss trên phân phối huấn luyện, không phụ thuộc giai đoạn huấn luyện hay độ sâu. |
| Sample efficiency | Mô hình lớn đạt cùng loss với ít bước và ít mẫu hơn. |
| Hợp nhất | $L(N,D)$ khớp tốt trừ khi $D$ giảm 1024 lần (~$2\times10^7$ token; một epoch chỉ 40 bước cập nhật), khi đó fit kém. |
| Phạm vi $N$ | 768 đến 1.5 tỷ tham số không embedding; $D$: 22 triệu đến 23 tỷ token. |
| Mô hình lớn gặp overfit nhẹ | Với 22 tỷ token WebText2 thì mô hình $<10^9$ tham số hầu như không overfit; mô hình lớn nhất gặp overfit nhẹ. |

Bảng tham số fit (Phụ lục A, Bảng 5 và 6):

| Hằng số | Giá trị |
|---|---|
| $\alpha_N$, $N_c$ | 0.076, $8.8\times10^{13}$ |
| $\alpha_D$, $D_c$ | 0.095, $5.4\times10^{13}$ |
| $\alpha_C$ (naive, batch cố định), $C_c$ | 0.057, $1.6\times10^7$ PF-days |
| $\alpha_C^{min}$, $C_c^{min}$ | 0.050, $3.1\times10^8$ PF-days |
| $\alpha_B$, $B_*$ | 0.21, $2.1\times10^8$ token |
| $\alpha_S$, $S_c$ | 0.76, $2.1\times10^3$ bước |

**Ví dụ số tự tính** (dùng $L(N)$ của Phụ lục A cho WebText2 + BPE GPT-2; con số chỉ có ý nghĩa tương đối):

| $N$ (không embedding) | $L(N)=(8.8\times10^{13}/N)^{0.076}$ | Ghi chú |
|---|---|---|
| $0.8\times10^9$ | $\approx2.42$ nats | |
| $0.3\times10^9$ | $\approx2.60$ nats | cao hơn khoảng 8% |
| $0.1\times10^9$ | $\approx2.83$ nats | cao hơn khoảng 17% so với 0.8B |

Dữ liệu cần để không overfit theo (4.4): $D\gtrsim5\times10^3N^{0.74}\approx2.0\times10^{10}$ (0.8B), $9\times10^9$ (0.3B), $4\times10^9$ (0.1B) token (tự tính). Với $N=0.8$B, $D=20$B, Bảng 2 cho $L(N,D)\approx2.41$ so với $L(N,\infty)\approx2.36$ (dữ liệu hữu hạn thêm khoảng 2%).

## 5. Phân tích / ablation

- **LSTM so với Transformer (Hình 7):** LSTM ngang Transformer ở các token đầu ngữ cảnh nhưng **không** bắt kịp ở token sau (LSTM bão hoà sau dưới 100 token); Transformer cải thiện suốt cả ngữ cảnh. Phù hợp với việc attention tận dụng ngữ cảnh dài tốt hơn.
- **Universal Transformer (chia sẻ tham số):** tốt hơn một chút theo $N$ nhưng kém hơn một chút theo compute (tái sử dụng tham số).
- **Batch tới hạn $B_{crit}$:** không phụ thuộc trực tiếp vào kích thước mô hình, chỉ qua loss đạt được; khớp với ước lượng gradient noise scale.
- **Độ dốc ngữ cảnh:** theo vị trí token, loss cũng theo luỹ thừa (Phụ lục D.5), mô hình lớn có số mũ lớn hơn, gợi ý nhận ra mẫu nhanh hơn.
- **Hessian:** luỹ thừa theo $S_{min}$ có thể phản ánh phổ Hessian gần như độc lập với kích thước mô hình (suy đoán của tác giả).

**Mâu thuẫn và giả thuyết (Mục 6.3).** Ngoại suy $L(C_{min})$ với dữ liệu chỉ tăng $D\propto C^{0.27}$ (một epoch) cắt $L(D)$ khi $C^*\sim10^4$ PF-days, $N^*\sim10^{12}$, $D^*\sim10^{12}$ token, $L^*\sim1.7$ nats/token (rất bất định, thay đổi cỡ một bậc độ lớn theo số mũ). Tác giả đoán nó có thể là ước lượng entropy của ngôn ngữ, chỉ là giả thuyết.

## 6. Giới hạn

**Do tác giả nêu (Phụ lục C):**
- Chưa có lý thuyết giải thích các luật; "not solid theoretical understanding".
- Không tự tin về $B_{crit}(L)$ khi loss ra ngoài dải đã thử.
- Chưa kiểm kỹ vùng dữ liệu nhỏ (fit $L(N,D)$ kém cho $D$ nhỏ nhất); chưa thử regularization/data augmentation.
- $C\approx6NBS$ bỏ qua phần phụ thuộc $n_{ctx}$ nên sai lệch với ngữ cảnh rất dài ($n_{ctx}\gtrsim12d_{model}$).
- Có thể chưa tinh chỉnh hết siêu tham số (khởi tạo, momentum); lr tối ưu phụ thuộc loss mục tiêu.
- Thấy mâu thuẫn ở quy mô cực lớn (mục 6.3 trên).

**Do chúng ta nhận thấy:**
- Mọi mô hình dùng một lịch cosine cố định (độ dài ~$2.5\times10^5$ bước) cho tất cả; bài 04 chỉ ra điều này làm đánh giá thấp mô hình huấn luyện ngắn, và ảnh hưởng đến kết luận về phân bổ $N$ so với $D$.
- Đa số mô hình trong thí nghiệm nhỏ (<100M tham số) nên ngoại suy lên cỡ hàng chục tỷ có độ lệch (Chinchilla thấy đường cong biên hơi cong).
- Chỉ trên văn bản tự nhiên, từ vựng GPT-2; không áp dụng trực tiếp cho OCR hay đa phương thức.
- Loại embedding khỏi $N$ đẹp cho phân tích, nhưng với mô hình nhỏ (như 0.1B) embedding chiếm phần lớn nên luật này kém hữu dụng nếu không điều chỉnh (suy luận; xem mục 8).
- Số liệu có vài chỗ lệch nhỏ giữa các phần: $\alpha_D$ là 0.095 (tóm tắt) so với 0.103 (fit chung); $\alpha_C^{min}$ là 0.050 (thực nghiệm), 0.054 (6.4), 0.052 (B.7); $N\propto C^{0.73}$ (thực nghiệm) so với $C^{0.71}$ (dự báo).

## 7. Liên hệ với các bài khác

- **01** Attention Is All You Need: kiến trúc được khảo sát; công thức $N=12\,n_{layer}d^2$ lấy từ cấu trúc ở đó.
- **02** GPT-3: áp dụng trực tiếp tinh thần bài này (model rất lớn, ~300B token).
- **04** Chinchilla: kiểm lại cùng câu hỏi, kết luận ngược về hệ số ($N\propto C^{0.5}$, $D\propto C^{0.5}$ so với $0.73/0.27$ ở đây).
- **05** LLaMA, **06** Mistral: huấn luyện nhỏ hơn nhưng nhiều dữ liệu hơn (đi ngược gợi ý bài này vì chi phí suy luận).
- **08** LoRA, **48 đến 50**: giảm chi phí triển khai thay vì chi phí huấn luyện, hướng bài này không xét.
- **27, 28** MoE: thay đổi quan hệ giữa $N$ (tổng) và FLOPs mỗi token (bài này giả định dense, $C\propto N$).

## 8. Ứng dụng cho dự án (suy luận)

Paper nói về **huấn luyện từ đầu** tới tối ưu compute; dự án của bạn là **cắt + distillation** từ mô hình sẵn có, nên chỉ vài ý dùng được:

1. **Dự báo mức tụt chất lượng khi giảm cỡ.** Nếu coi $L(N)\propto N^{-0.076}$ là thước đo thô: 0.8B → 0.3B tăng loss khoảng 8% và 0.8B → 0.1B tăng khoảng 17% (tự tính, trên WebText2, mô hình huấn luyện từ đầu và hội tụ). Đây là **ngưỡng tham chiếu**: nếu mô hình cắt + distill của bạn có loss/accuracy tệ hơn hẳn so với mô hình cùng kích thước huấn luyện từ đầu theo luật này, phục hồi chưa đủ; nếu tốt hơn thì distillation đang "mượn" tri thức từ giảng viên. Cảnh báo: luật cho ngôn ngữ chung, không cho OCR, và $N$ phải tách embedding.
2. **Embedding trong mô hình nhỏ.** Luật dùng $N$ không embedding. Khi cắt xuống 0.1B, embedding/LM head có thể chiếm tỉ lệ lớn; dùng $N_{non-emb}$ khi so sánh các cấu hình cắt, và nhớ rằng embedding ảnh hưởng byte đọc mỗi bước decode dù không ảnh hưởng luật.
3. **Hình dạng ít quan trọng.** Mục 3.1: với $N$ cố định, tỉ lệ độ sâu/độ rộng và số head ít ảnh hưởng loss. Điều này gợi ý (suy luận, chỉ cho huấn luyện từ đầu và loss chung) bạn có thể ưu tiên chọn hình dạng theo **độ trễ**: ít tầng hơn (độ sâu thấp) giảm số kernel tuần tự ở batch 1. Kết quả của bạn rằng cắt tầng phục hồi tốt hơn cắt neuron MLP không mâu thuẫn, nhưng không được chứng minh bởi paper này (bài này không có cắt tỉa).
4. **Chi phí huấn luyện/distill.** $C\approx6ND$ cho ước lượng ngân sách: ví dụ distill một học sinh 0.3B trên 5B token: $6\cdot3\times10^8\cdot5\times10^9=9\times10^{18}$ FLOPs ≈ 0.1 PF-day (tự tính, bỏ qua chi phí forward của giảng viên 4B, mà cho mỗi token tốn thêm $\approx2\cdot4\times10^9$ FLOPs, tức khoảng 4.4 lần chi phí forward+backward của học sinh ($8\times10^9$ so với $6\cdot3\times10^8=1.8\times10^9$ FLOPs mỗi token); đây là phần cần cộng vào).
5. **Batch tới hạn.** $B_{crit}\sim10^6$ đến $3\times10^6$ token (tự tính từ công thức với loss 3 đến 2.4) cho biết batch lớn hơn mức này lãng phí compute; với GPU nhỏ, batch của bạn sẽ nhỏ hơn nhiều nên không phải ràng buộc.

Điều ít liên quan: các luật về $L(C_{min})$ và số bước tối ưu ở quy mô hàng tỷ tham số, phân bổ toàn cầu giữa $N$ và $D$ (xem 04), và so sánh LSTM.

## 9. Thí nghiệm gợi ý

1. **Điểm tham chiếu scaling cho họ cắt tỉa của bạn:** vẽ loss/CER trên tập OCR kiểm tra theo $N_{non-emb}\in\{0.8,0.3,0.1\}$B của (a) mô hình cắt + distill, (b) mô hình nhỏ huấn luyện từ đầu (nếu có) trên log-log; ước lượng độ dốc hiệu dụng $\alpha_{eff}$ và so với 0.076.
2. **Aspect ratio ở batch 1:** với cùng $N$ (~0.3B), thử hai cấu hình cắt: ít tầng rộng và nhiều tầng hẹp (nếu cắt-chiều-rộng khả thi); đo cả loss sau distill lẫn thời gian decode mỗi token trên GPU đích.
3. **Dữ liệu distill đủ chưa:** ước lượng $D\gtrsim5\times10^3N^{0.74}$ cho từng kích thước học sinh để biết số token tối thiểu tránh overfit trên tập chuyển giao (kiểm tra đường cong train/val).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / giá trị |
|---|---|
| Tham số | $N\approx12\,n_{layer}d_{model}^2$ (không embedding) |
| Compute huấn luyện | $C\approx6ND$; $1\,\text{PF-day}=8.64\times10^{19}$ FLOPs |
| $L(N)$ | $(N_c/N)^{0.076}$, $N_c=8.8\times10^{13}$ |
| $L(D)$ | $(D_c/D)^{0.095}$, $D_c=5.4\times10^{13}$ |
| $L(C_{min})$ | $(C_c^{min}/C_{min})^{0.050}$ |
| $L(N,D)$ | $[(N_c/N)^{\alpha_N/\alpha_D}+D_c/D]^{\alpha_D}$ |
| Chống overfit | $D\gtrsim5\times10^3N^{0.74}$ |
| $B_{crit}$ | $B_*/L^{1/\alpha_B}$, $B_*=2.1\times10^8$, $\alpha_B=0.21$ |
| Phân bổ tối ưu | $N\propto C^{0.73}$, $B\propto C^{0.24}$, $S\propto C^{0.03}$, $D\propto C^{0.27}$ |
| Dừng sớm | huấn luyện hiệu quả dừng ~10% trên loss hội tụ |
| Mô hình lệch kích thước | 0.6 đến 2.2 lần kích thước tối ưu: +20% compute |

## 11. Kết luận ngắn

Loss của Transformer LM tuân theo luật luỹ thừa mượt theo $N$, $D$, $C$ với rất ít phụ thuộc vào hình dạng; từ đó suy ra phân bổ compute nghiêng mạnh về mô hình lớn ($N\propto C^{0.73}$) và dừng sớm. Kết luận về tỉ lệ model/data sau này bị bài 04 sửa; phần còn bền gồm công thức $C\approx6ND$, $N\approx12Ld^2$, tư duy dự báo bằng luật luỹ thừa và việc tách embedding khỏi $N$.

Nguồn gốc: https://arxiv.org/abs/2001.08361
