# Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (2021)

**Tác giả:** William Fedus, Barret Zoph, Noam Shazeer (Google).  
**Venue / arXiv:** arXiv:2101.03961 (v3, 2022); đăng ở Journal of Machine Learning Research 23 (2022) 1-40.  
**Nguồn:** https://arxiv.org/abs/2101.03961  
**Chủ đề:** Mixture of Experts (MoE) thưa, định tuyến top-1 (switch routing), loss cân bằng tải, expert capacity, huấn luyện ổn định với bfloat16, chưng cất mô hình thưa sang mô hình dày.  
**Vị trí trong lộ trình:** Đọc sau 01 (Transformer), 03 (Kaplan scaling laws), 11 (GLU/SwiGLU, vì expert chính là một FFN). Bài này biến MoE từ một kỹ thuật phức tạp thành một thay đổi đơn giản (mỗi token chỉ đi vào *một* expert). Bài kế tiếp là 28 (Mixtral: MoE top-2 trong LLM mã nguồn mở), sau đó 25 (Jamba: hybrid + MoE) và 16 (DeepSeek-V2).

> Tài liệu học tập bằng tiếng Việt, tổng hợp từ paper (đọc bản full text). Đây là ghi chú diễn giải, không phải bản dịch. Phần "Ứng dụng cho dự án" và các con số đánh dấu "tự tính" là suy luận của người viết, không phải kết quả của paper. Paper dài (khoảng 16k từ, phần phụ lục có nhiều code Mesh-TensorFlow); ghi chú tập trung vào cơ chế định tuyến, loss cân bằng tải, các thủ thuật ổn định, kết quả scaling và chưng cất, còn phần song song hóa (data/model/expert parallelism) chỉ nêu ý chính.

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

**Mô hình dày (dense) và định luật scaling.** Trong một Transformer thông thường, *mọi* tham số đều được dùng cho *mọi* token. Kaplan et al. (bài 03) cho thấy loss giảm theo lũy thừa khi tăng số tham số, dữ liệu và compute. Nhưng tăng tham số ở mô hình dày đồng nghĩa với tăng FLOPs (số phép tính dấu phẩy động) tương ứng, nên chi phí tăng theo.

**Câu hỏi của paper:** nếu ta tăng *số tham số* mà giữ *FLOPs mỗi token* không đổi thì sao? Paper giả thuyết rằng số tham số, tách rời khỏi lượng tính toán, là một **trục scaling thứ tư** (ngoài tham số-và-FLOPs đi cùng nhau, dữ liệu, compute).

**Conditional computation (tính toán có điều kiện).** Ý tưởng cho mỗi đầu vào đi qua một phần khác nhau của mạng. Dạng quen thuộc nhất là **Mixture of Experts (MoE)**: có nhiều mạng con gọi là *expert*, và một mạng nhỏ gọi là *router* (bộ định tuyến) quyết định mỗi token đi vào expert nào. Shazeer et al. (2017) đưa MoE vào LSTM, GShard (Lepikhin et al., 2020) đưa vào Transformer dịch máy. Nhưng paper nêu ba trở ngại làm MoE chưa được dùng rộng rãi: **phức tạp, chi phí giao tiếp giữa các thiết bị, và huấn luyện không ổn định**.

**Giả định cũ bị Switch phản bác.** Shazeer 2017 cho rằng phải định tuyến tới k > 1 expert thì router mới có gradient có ý nghĩa ("phải so sánh ít nhất hai expert"). Switch Transformer chỉ dùng k = 1 và vẫn hoạt động tốt hơn.

## 2. Ý tưởng chính

Trong Transformer, thay lớp FFN (feed-forward) bằng một **lớp Switch**: có N bản FFN độc lập (N expert), một router nhỏ chọn *đúng một* expert cho mỗi token, và đầu ra của expert đó được nhân với xác suất mà router gán cho nó. Vì mỗi token chỉ chạy qua một FFN, FLOPs mỗi token bằng mô hình dày gốc, nhưng số tham số tăng gần N lần. Để các expert được dùng đều, paper thêm một loss phụ cân bằng tải; để huấn luyện ổn định, paper dùng độ chính xác float32 cục bộ cho router, khởi tạo nhỏ hơn 10 lần, và dropout lớn hơn trong expert khi fine-tune. Kết quả: cùng FLOPs và cùng thời gian, mô hình thưa đạt chất lượng của mô hình dày sau thời gian nhanh gấp tới 7 lần.

## 3. Phương pháp chi tiết

### 3.1 Router và định tuyến top-k

Với token biểu diễn $x$, router có ma trận $W_r$, tạo logit $h(x)=W_r x$ và softmax qua $N$ expert:

$$p_i(x)=\frac{e^{h(x)_i}}{\sum_{j=1}^{N}e^{h(x)_j}}.$$

MoE gốc chọn tập $\mathcal T$ gồm top-k chỉ số và tính:

$$y=\sum_{i\in\mathcal T}p_i(x)\,E_i(x).$$

**Switch routing** là trường hợp $k=1$: $\mathcal T=\{\arg\max_i p_i(x)\}$, tức

$$\boxed{y=p_{i^\ast}(x)\,E_{i^\ast}(x),\qquad i^\ast=\arg\max_i p_i(x)}$$

Giá trị cổng $p_{i^\ast}(x)$ vẫn nhân vào đầu ra nên router có gradient (đây là điểm cho phép học định tuyến dù chỉ chọn một expert). Ba lợi ích paper nêu: (1) tính toán router ít hơn; (2) batch size (capacity) của mỗi expert có thể giảm ít nhất một nửa; (3) cài đặt đơn giản, giao tiếp ít hơn.

Chi phí router: $O(d_{model}\times N)$ mỗi token, rất nhỏ so với FFN.

### 3.2 Expert capacity và token bị bỏ (dropped tokens)

Trên TPU kích thước tensor phải biết trước lúc biên dịch, nên mỗi expert có một **capacity** cố định (số token tối đa nó xử lý):

$$\boxed{\text{expert capacity}=\left(\frac{\text{tokens per batch}}{\text{number of experts}}\right)\times\text{capacity factor}}$$

- Capacity factor = 1.0: mỗi expert vừa đủ nếu phân bố hoàn toàn đều.
- Capacity factor > 1.0: thêm vùng đệm cho trường hợp lệch, nhưng tốn compute/bộ nhớ cho các ô trống (padding).
- Token vượt capacity bị **drop**: layer bỏ qua tính toán, token đi tiếp qua *residual connection* (đầu ra của layer bằng đầu vào). Paper báo tỉ lệ drop thường dưới 1%.

Ví dụ nhỏ (tự tính): batch 4096 token, 8 expert, capacity factor 1.25: capacity mỗi expert = 4096/8 × 1.25 = 640. Nếu router dồn 800 token vào expert 3 thì 160 token bị drop ở layer đó.

### 3.3 Loss cân bằng tải (load-balancing loss)

Nếu để router tự do, nó thường dồn mọi token vào vài expert ("collapse"), các expert còn lại không được huấn luyện. Paper thêm loss phụ cho mỗi lớp Switch. Với batch $\mathcal B$ gồm $T$ token:

$$f_i=\frac1T\sum_{x\in\mathcal B}\mathbb 1\{\arg\max p(x)=i\},\qquad P_i=\frac1T\sum_{x\in\mathcal B}p_i(x)$$

$$\boxed{\mathcal L_{aux}=\alpha\cdot N\cdot\sum_{i=1}^{N}f_i\,P_i}$$

- $f_i$: tỉ lệ token *thực sự* được gửi tới expert $i$ (không khả vi vì có argmax).
- $P_i$: tổng *xác suất* router dành cho expert $i$ (khả vi); gradient chảy qua đây.
- Cực tiểu khi phân bố đều: $f_i=P_i=1/N$ thì $\sum f_iP_i = 1/N$, nhân $N$ ra 1 (hằng số theo $N$, nên $\alpha$ không phải chỉnh lại khi đổi số expert).
- Paper dùng $\alpha=10^{-2}$ (đã quét từ $10^{-1}$ đến $10^{-5}$).

Ví dụ (tự tính): $N=8$. Nếu phân bố đều, $N\sum f_iP_i = 8\cdot 8\cdot\frac1{64}=1$. Nếu toàn bộ token và xác suất dồn vào một expert ($f=P=(1,0,\dots,0)$), giá trị là $8\cdot 1=8$, tức phạt gấp 8 lần.

### 3.4 Ổn định huấn luyện

1. **Selective precision.** Chạy cả mô hình bằng bfloat16 thì router (softmax trên logit) dễ mất ổn định. Paper chỉ ép *đầu vào router* sang float32, bên trong hàm router; tensor dispatch/combine cast về bfloat16 trước khi giao tiếp đa thiết bị, nên không tốn băng thông float32. 
2. **Khởi tạo nhỏ hơn.** Chuẩn truncated normal với $\sigma=\sqrt{s/n}$ ($n$ = fan-in); giảm scale $s$ từ 1.0 xuống 0.1.
3. **Expert dropout.** Khi fine-tune trên tập nhỏ, mô hình nhiều tham số dễ overfit. Giải pháp: dropout thấp (0.1) ở các lớp thường nhưng cao (0.4) *bên trong* FFN của expert.
4. **Exploration trong định tuyến** (Phụ lục C): router chọn rời rạc nên giống bài toán contextual bandit. Thử argmax, sample từ softmax, input dropout, input jitter (nhân nhiễu ngẫu nhiên vào đầu vào router). Paper dùng jitter.

### 3.5 Pseudo-code lớp Switch (rút gọn từ Phụ lục F, viết lại bằng PyTorch-like)

```python
def switch_ffn(x, W_r, experts, capacity):        # x: [T, d]
    logits = (x * jitter()).float() @ W_r          # router chạy float32  [T, N]
    probs  = softmax(logits, -1)
    gate, idx = probs.max(-1)                      # top-1: [T]
    mask = one_hot(idx, N)                         # [T, N]
    pos  = cumsum(mask, 0) * mask                  # vị trí của token trong hàng đợi expert
    keep = (pos <= capacity)                       # token vượt capacity bị drop
    aux  = alpha * N * (mask.float().mean(0) * probs.mean(0)).sum()
    y = zeros_like(x)                              # token bị drop: y=0, residual giữ nguyên
    for i in range(N):                             # thực tế: all-to-all giữa các thiết bị
        sel = (idx == i) & keep.any(-1)
        y[sel] = gate[sel, None] * experts[i](x[sel])
    return y, aux
```

### 3.6 Song song hóa (chỉ ý chính, Mục 5)

Mỗi expert nằm trên một thiết bị (core) riêng, nên số tham số tăng theo số thiết bị trong khi bộ nhớ mỗi thiết bị hầu như không đổi. Token được chuyển tới expert bằng phép **all-to-all**, rồi chuyển về. Kết hợp ba kiểu song song (data, model, expert) cho hai mô hình lớn: Switch-XXL (395B tham số, FLOPs/seq bằng T5-XXL) và Switch-C (1,6 nghìn tỉ tham số, 2048 expert, chỉ dùng expert parallelism).

## 4. Thí nghiệm và kết quả chính

**Thiết lập.** Tiền huấn luyện trên C4 (Colossal Clean Crawled Corpus) bằng masked language modeling kiểu T5 (che 15% token, thay bằng một sentinel); đo *negative log perplexity* (đơn vị nat, càng cao càng tốt). Switch-Base khớp FLOPs với T5-Base (223M), Switch-Large khớp T5-Large (739M). Các mô hình đặt expert ở một lớp FFN trong hai lớp (expert freq 1/2), trừ Switch-C (mọi lớp).

### 4.1 Switch so với MoE top-2 và T5 (Bảng 1, 128 expert, 32 core TPUv3)

| Mô hình | Capacity factor | Chất lượng sau 100k bước | Thời gian đạt ngưỡng -1.50 (giờ) | Tốc độ (ví dụ/giây) |
|---|---|---|---|---|
| T5-Base | - | -1.731 | không đạt | 1600 |
| T5-Large | - | -1.550 | 131.1 | 470 |
| MoE-Base | 2.0 | -1.547 | 68.7 | 840 |
| Switch-Base | 2.0 | -1.554 | 72.8 | 860 |
| MoE-Base | 1.25 | -1.559 | 80.7 | 790 |
| Switch-Base | 1.25 | -1.553 | 65.0 | 910 |
| MoE-Base | 1.0 | -1.572 | 80.1 | 860 |
| Switch-Base | 1.0 | -1.561 | 62.8 | 1000 |
| Switch-Base+ (d_model 896, 16 heads) | 1.0 | -1.534 | 67.6 | 780 |

Đọc bảng: Switch nhanh hơn và tốt hơn MoE top-2 khi capacity factor thấp (1.0, 1.25). Paper tự chú thích một điểm bất thường: MoE chạy chậm đi khi hạ capacity từ 2.0 xuống 1.25 (840 xuống 790), và nhấn mạnh tốc độ phụ thuộc cả cài đặt cấp thấp chứ không chỉ thuật toán.

### 4.2 Scaling theo số expert

- Giữ FLOPs/token cố định, tăng số expert từ 2 lên 256 (14,7B tham số) làm test loss giảm đều (Hình 4).
- Theo số bước: Switch-Base 64 expert đạt chất lượng T5-Base ở bước 60k (T5-Base) chỉ sau một phần nhỏ số bước (paper nói tăng tốc 7,5 lần theo số bước; câu văn trong bản text bị lỗi nên không trích số bước chính xác).
- Theo thời gian thực: Switch-Base 64 expert đạt chất lượng tương đương T5-Base trong **1/7 thời gian**.
- So với T5-Large (nhiều hơn 3,5 lần FLOPs/token): Switch-Base vẫn hiệu quả mẫu cao hơn và nhanh hơn 2,5 lần theo thời gian.

### 4.3 Fine-tuning (Bảng 5, trích)

| Mô hình | GLUE | SQuAD | SuperGLUE | Winogrande (XL) | XSum | CB Trivia QA |
|---|---|---|---|---|---|---|
| T5-Base | 84.3 | 85.5 | 75.1 | 66.6 | 18.7 | 24.5 |
| Switch-Base | 86.7 | 87.2 | 79.5 | 73.3 | 20.3 | 30.7 |
| T5-Large | 87.8 | 88.1 | 82.7 | 79.1 | 20.9 | 29.5 |
| Switch-Large | 88.5 | 88.6 | 84.7 | 83.0 | 22.3 | 36.9 |

SuperGLUE tăng 4,4 điểm (Base) và 2,0 điểm (Large). Ngoại lệ: trên ARC (reasoning) T5-Base thắng Switch-Base ở ARC Challenge (35.5 so với 32.8) và T5-Large thắng Switch-Large ở ARC Easy (68.8 so với 66.0).

### 4.4 Chưng cất MoE thưa sang mô hình dày (Bảng 6, 7, 8)

Chưng cất (distillation) là huấn luyện mô hình nhỏ ("student") bắt chước đầu ra của mô hình lớn ("teacher"). Paper chưng cất Switch-Base (3,8B) vào T5-Base (223M):

| Kỹ thuật | Tham số | Chất lượng (phần trăm chất lượng teacher giữ lại) |
|---|---|---|
| T5-Base (baseline) | 223M | -1.636 |
| Switch-Base (teacher) | 3.800M | -1.444 |
| Chưng cất thường | 223M | -1.631 (3%) |
| + khởi tạo student bằng trọng số non-expert của teacher | 223M | -1.598 (20%) |
| + trộn loss 0.75 nhãn thật + 0.25 xác suất teacher | 223M | -1.580 (29%) |

Hai thủ thuật cộng lại giữ khoảng 30% phần "cải thiện chất lượng" của teacher trong khi nén khoảng 95% tham số. Với các teacher khác kích thước (Bảng 7): teacher 1,1B giữ 37% (nén 82%); 7,4B giữ 27% (nén 97%); 14,7B giữ 28% (nén 99%). Với mô hình đã fine-tune trên SuperGLUE (Bảng 8): T5-Base 74.6, Switch-Base 81.3, T5-Base chưng cất 76.6 (giữ 30%).

Lưu ý: "khởi tạo bằng trọng số non-expert" khả thi vì mô hình khớp FLOPs nên attention và các FFN dày có cùng kích thước; khởi tạo bằng trọng số của expert (chạy bình thường không chưng cất) không cải thiện.

### 4.5 Đa ngôn ngữ

mSwitch-Base so với mT5-Base trên mC4, 101 ngôn ngữ: cải thiện perplexity ở cả 101 ngôn ngữ; tăng tốc trung bình 5 lần theo số bước, và 91% ngôn ngữ đạt tăng tốc từ 4 lần trở lên.

### 4.6 Mô hình lớn (Bảng 9)

| Mô hình | Tham số | FLOPs/seq | Số expert | Neg. log perp. @250k | @500k |
|---|---|---|---|---|---|
| T5-Base | 0,2B | 124B | - | -1.599 | -1.556 |
| T5-XXL | 11B | 6,3T | - | -1.147 | -1.095 |
| Switch-XXL | 395B | 6,3T | 64 | -1.086 | -1.008 |
| Switch-C | 1571B | 890B | 2048 | -1.096 | -1.043 |

Switch-C nhanh hơn T5-XXL 4 lần để đạt cùng perplexity (cùng ngân sách compute). Fine-tune sớm (mới tiền huấn luyện 503B token, khoảng nửa lượng dữ liệu của T5-XXL): SQuAD 89.7 (SOTA lúc đó 91.3), SuperGLUE test 87.5 (SOTA 90.0), ANLI 65.7 (kỷ lục trước đó 49.4), Natural Questions 34.4 (trước đó 32.8), Web Questions 41.0 (trước đó 37.2).

## 5. Phân tích / ablation

- **Selective precision (Bảng 2).** 32 expert: float32 thuần cho -1.718 ở 1160 ví dụ/giây; bfloat16 thuần *phân kỳ* (-3.780) ở 1390; selective precision cho -1.716 ở 1390. Tức giữ được ổn định của float32 mà gần tốc độ bfloat16.
- **Khởi tạo (Bảng 3).** Sau 3,5k bước (3 seed): init 0.1x cho -2.72 (độ lệch chuẩn 0.01); init 1.0x cho -3.60 (độ lệch chuẩn 0.68). Khởi tạo nhỏ vừa tốt hơn vừa ổn định hơn.
- **Expert dropout (Bảng 4).** Tăng dropout đồng loạt làm xấu đi; dropout 0.1 ở lớp thường + 0.4 ở expert tốt nhất (GLUE 85.2, so với 84.7 khi dropout đều 0.1).
- **Exploration (Bảng 11).** Argmax -1.471; sample softmax -1.570 (kém); input dropout -1.480; input jitter -1.468 (tốt nhất).
- **Switch cho attention (Phụ lục A, Bảng 10).** Đặt expert vào các phép chiếu Q/K/V cho chất lượng tốt hơn khi float32, nhưng *phân kỳ* khi bfloat16, nên không đưa vào bản cuối.
- **No-Token-Left-Behind (Phụ lục B).** Định tuyến lại token tràn sang expert xếp hạng hai; tác giả giả thuyết giúp ích nhưng không thấy lợi ích thực nghiệm.
- **Upstream và downstream (Phụ lục E).** Perplexity tiền huấn luyện tương quan với kết quả downstream, nhưng ở quy mô lớn nhất mô hình thưa không chuyển hết lợi ích sang SuperGLUE (mô hình dày thường tốt hơn ở cùng perplexity), trong khi trên TriviaQA (tri thức) mô hình thưa có vẻ tốt hơn. Paper nói cần thêm dữ liệu để khẳng định.
- **Ít expert vẫn có lợi (Phụ lục D).** Chỉ 2, 4, 8 expert đã cải thiện so với T5-Base dày, nên không cần siêu máy tính.

## 6. Giới hạn

**Do tác giả nêu:**
- Mô hình thưa lớn nhất (Switch-XXL) đôi khi mất ổn định; vì vậy không huấn luyện đủ 1M bước. Switch-C (1,6T) không bất ổn.
- Lợi ích upstream chưa chuyển hết sang downstream, nhất là tác vụ suy luận (SuperGLUE, ARC) ở quy mô rất lớn.
- Chưng cất giữ được "chỉ" khoảng 30% phần cải thiện.
- Chi phí giao tiếp all-to-all và việc phải chọn capacity factor là gánh nặng hệ thống; thuật toán chưa dùng được attention thưa theo độ dài chuỗi (phần Related Work nói đó là hướng kết hợp tương lai).
- Paper mô tả quyết định định tuyến cứng (rời rạc) là một nguồn bất ổn và coi việc khám phá expert (exploration) như bài toán bandit; họ chỉ thử vài cách đơn giản (jitter là tốt nhất), không phải lời giải triệt để (nhận xét của người viết).

**Do người viết nhận xét:**
- Mọi so sánh dựa vào FLOPs/token, bỏ qua **bộ nhớ**: Switch-Base có 7B tham số nhưng chỉ tính như 223M về FLOPs; khi triển khai, toàn bộ 7B phải nằm trong bộ nhớ (hoặc phải offload). Bài toán suy luận trên GPU nhỏ vì thế không được đề cập.
- Tốc độ báo cáo trên TPU với shape tĩnh và capacity cố định; trên GPU với shape động hành vi có thể khác.
- Chất lượng đo chủ yếu qua perplexity trên C4 và các tác vụ fine-tune kiểu T5; không có đánh giá few-shot/instruction kiểu LLM hiện đại.

## 7. Liên hệ với các bài khác

- **03 Kaplan:** Switch coi số tham số (độc lập FLOPs) là trục scaling mới, mở rộng Kaplan.
- **11 GLU variants:** expert trong Switch dùng FFN GEGLU; trong Mixtral (28) expert là SwiGLU.
- **28 Mixtral:** MoE top-2 trong LLM decoder-only mã nguồn mở, đặt expert ở *mọi* lớp; Switch đặt ở một nửa số lớp và dùng top-1.
- **25 Jamba, 16 DeepSeek-V2:** các kiến trúc về sau kết hợp MoE với thành phần khác (hybrid Mamba/attention; MLA).
- **08 LoRA, 53-54 (GPTQ, AWQ):** MoE nhiều tham số nên bài toán nén/lượng tử hóa càng quan trọng (suy luận của người viết).
- **Chưng cất:** phần 4.2 là điểm giao trực tiếp với dự án (xem mục sau).

## 8. Ứng dụng cho dự án (suy luận)

**Mức liên quan: trung bình về khái niệm, thấp về triển khai trực tiếp.**

1. **MoE không phù hợp mục tiêu GPU nhỏ, batch 1.** Mục tiêu ≤ 8 GB bị chi phối bởi *tổng tham số phải nằm trong bộ nhớ*, mà MoE tăng tổng tham số gấp gần N lần cho cùng FLOPs. Ví dụ (tự tính, giả sử $d_{model}=1024$, $d_{ff}=3072$, SwiGLU, 24 lớp): một FFN SwiGLU có $3\cdot1024\cdot3072\approx9{,}4$M tham số, 24 lớp là khoảng 226M (khoảng 0,45 GB ở fp16). Nếu thay bằng 8 expert top-2, FFN thành khoảng 1,81B tham số (khoảng 3,6 GB fp16) trong khi tham số được dùng mỗi token chỉ khoảng 0,45B. Latency batch 1 chủ yếu bị chặn bởi băng thông đọc trọng số, nên lợi thế (đọc ít hơn) chỉ có khi các expert được chọn nằm trên GPU, còn chi phí bộ nhớ thì tăng gấp 8.
2. **Chưng cất từ mô hình thưa sang dày (Mục 4.4) song song với pipeline của bạn.** Hai kết quả đáng chú ý: (a) khởi tạo student bằng trọng số teacher ở những phần có cùng kích thước tăng phần chất lượng giữ lại từ 3% lên 20%; (b) trộn loss nhãn thật (0.75) và xác suất teacher (0.25) tăng tiếp lên 29%. Với việc bạn cắt tỉa rồi chưng cất từ teacher 4B, điều này gợi ý: *khởi tạo student từ trọng số teacher đã cắt* (đúng việc bạn đang làm) là một nửa công thức, và nên thử thêm việc pha loss nhãn thật với loss KD thay vì chỉ KD (cần kiểm chứng trên OCR, nơi nhãn là văn bản chính xác).
3. **Góc nhìn "FFN là tập hợp nhiều neuron thưa kích hoạt".** MoE có thể xem như phân nhóm neuron FFN thành expert và chỉ kích hoạt một nhóm. Quan sát của bạn rằng cắt neuron MLP phục hồi tệ hơn cắt cả lớp có thể liên hệ: neuron trong một FFN dày dùng chung cho mọi token, nên bỏ chúng gây hại với nhiều token cùng lúc (đây là giả thuyết, paper không bàn). MoE cho thấy hướng ngược lại: chuyên môn hóa theo token. Nếu D_res (phần hỏng không sửa được bằng ánh xạ tuyến tính) tập trung ở một số loại token (ví dụ chữ số, ký tự đặc biệt trong OCR), có thể đo xem hỏng hóc có không đều theo token, một phép đo định tính tương tự phân tích định tuyến.
4. **Hybrid decoder của bạn (GatedDeltaNet + full attention)** không có MoE, nên bài này không áp dụng cho khối token mixing. Chỉ lớp FFN là chỗ MoE xuất hiện, và sau khi cắt tỉa xuống 0,3B/0,1B thì lợi ích scaling của MoE (cần nhiều dữ liệu và tham số) không còn là mục tiêu.

## 9. Thí nghiệm gợi ý

1. **Pha loss nhãn thật + KD:** trong bước chưng cất, thử $\mathcal L=0.75\,\mathcal L_{CE}(\text{nhãn})+0.25\,\mathcal L_{KD}$ so với KD thuần, trên cùng checkpoint đã cắt; đo CER/WER của OCR.
2. **Đo hỏng hóc theo loại token:** với mô hình đã cắt neuron MLP, tính loss phục hồi theo nhóm token (chữ số, dấu câu, chữ thường, token đặc biệt của bảng/công thức) để xem D_res có tập trung vào một nhóm không (giống phân tích định tuyến của MoE).
3. **Thử nhỏ về load-balance:** nếu muốn hiểu cơ chế, tự cài lớp Switch 4 expert trên một Transformer nhỏ (khoảng 20M tham số) và quan sát hiện tượng collapse khi tắt $\mathcal L_{aux}$ (đếm $f_i$ theo bước).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| Router | $p(x)=\mathrm{softmax}(W_r x)$, chi phí $O(d\cdot N)$ |
| Switch (top-1) | $y=p_{i^\ast}(x)E_{i^\ast}(x)$, $i^\ast=\arg\max p$ |
| MoE top-k | $y=\sum_{i\in\mathcal T}p_i(x)E_i(x)$ |
| Expert capacity | $\frac{\text{tokens/batch}}{N}\times\text{capacity factor}$ |
| Dropped token | token vượt capacity, đi tiếp qua residual (thường < 1%) |
| Loss cân bằng tải | $\alpha N\sum_i f_iP_i$, $\alpha=10^{-2}$, cực tiểu = $\alpha$ khi đều |
| Selective precision | router chạy float32 cục bộ, phần còn lại bfloat16 |
| Khởi tạo | scale $s$ giảm 10 lần ($\sigma=\sqrt{s/n}$) |
| Expert dropout | 0.4 trong expert, 0.1 ở nơi khác khi fine-tune |
| Chưng cất | init non-expert từ teacher + 0.75 nhãn thật / 0.25 teacher: giữ khoảng 30% cải thiện, nén khoảng 95% |
| Số liệu then chốt | Switch-Base 64 expert: 7 lần nhanh hơn T5-Base theo thời gian; Switch-C 1,6T: 4 lần nhanh hơn T5-XXL |

## 11. Kết luận ngắn

Switch Transformer cho thấy định tuyến một expert mỗi token, cộng loss cân bằng tải đơn giản và vài thủ thuật ổn định (float32 cục bộ cho router, init nhỏ, dropout lớn trong expert), đủ để biến MoE thành kiến trúc huấn luyện được ở quy mô nghìn tỉ tham số, với tốc độ tiền huấn luyện nhanh hơn mô hình dày cùng FLOPs tới 4-7 lần. Cái giá là bộ nhớ (tất cả expert phải được lưu) và độ phức tạp hệ thống. Với dự án nén OCR VLM nhỏ, bài học đáng mang theo nằm ở phần chưng cất từ teacher thưa sang student dày (khởi tạo từ trọng số teacher + pha nhãn thật), chứ không phải ở việc dùng MoE.

Liên kết gốc: https://arxiv.org/abs/2101.03961
