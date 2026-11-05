# Fast Inference from Transformers via Speculative Decoding (2022/2023)

**Tác giả:** Yaniv Leviathan, Matan Kalman, Yossi Matias (Google Research).  
**Venue / arXiv:** arXiv:2211.17192 (v1 30/11/2022; v2 18/05/2023); ICML 2023 (PMLR 202, theo trang đầu của paper).  
**Nguồn:** https://arxiv.org/abs/2211.17192  
**Chủ đề:** Giải mã suy đoán (speculative decoding): dùng mô hình nhỏ đề xuất nhiều token, mô hình lớn kiểm tra song song bằng speculative sampling, **giữ nguyên phân phối đầu ra**; công thức số token kỳ vọng, tăng tốc, số phép toán và chọn $\gamma$.  
**Vị trí trong lộ trình:** Đọc trước: 01 (Transformer, giải mã tự hồi quy), 44-46 (decode bị nghẽn bởi bộ nhớ, KV cache). Bài này thêm: cách giảm **số lần chạy mô hình lớn** mà không đổi đầu ra. Đọc tiếp: 49 (Medusa) và 50 (EAGLE): thay mô hình nháp riêng bằng đầu dự đoán/ngoại suy đặc trưng ngay trong mô hình; và để hiểu mô hình nháp tốt từ đâu, xem lại các bài distillation của lộ trình.

> Đây là ghi chú học tập bằng tiếng Việt, diễn giải lại chứ không phải bản dịch. Số liệu lấy từ paper (có ghi rõ). Phần "Ứng dụng cho dự án" và các số "tự tính" là suy luận của chúng ta, không phải kết luận của paper. Paper loại trung bình (~8k từ); đây là bài có toán rõ ràng nên ghi chú đi sâu vào chứng minh.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
   - 3.1 [Chuẩn hoá về lấy mẫu từ một phân phối](#31-chuẩn-hoá-về-lấy-mẫu-từ-một-phân-phối)
   - 3.2 [Speculative sampling: quy tắc chấp nhận](#32-speculative-sampling-quy-tắc-chấp-nhận)
   - 3.3 [Chứng minh phân phối không đổi](#33-chứng-minh-phân-phối-không-đổi)
   - 3.4 [Thuật toán giải mã nhiều token](#34-thuật-toán-giải-mã-nhiều-token)
   - 3.5 [Tỉ lệ chấp nhận $\alpha$ và độ lệch $D_{LK}$](#35-tỉ-lệ-chấp-nhận-alpha-và-độ-lệch-d_lk)
   - 3.6 [Số token kỳ vọng và công thức tăng tốc](#36-số-token-kỳ-vọng-và-công-thức-tăng-tốc)
   - 3.7 [Số phép toán và chọn $\gamma$](#37-số-phép-toán-và-chọn-gamma)
   - 3.8 [Chọn mô hình nháp, beam search, lenience](#38-chọn-mô-hình-nháp-beam-search-lenience)
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

Mô hình tự hồi quy lớn (GPT-3, PaLM, LaMDA, T5-XXL) mạnh hơn mô hình nhỏ, nhưng sinh $K$ token cần **$K$ lần chạy tuần tự** của mô hình, và mỗi lần chạy của mô hình lớn chậm hơn nhiều so với mô hình nhỏ. Các hướng trước đó:

- **Giảm chi phí cho mọi token như nhau:** distillation, thưa hoá, lượng tử hoá, đổi kiến trúc (MQA...). Thường phải huấn luyện lại và **đổi đầu ra**.
- **Tính toán thích ứng (adaptive computation):** early exit, dùng mô hình nhỏ cho bước dễ. Cũng thường phải đổi kiến trúc/huấn luyện và không giữ đầu ra như cũ.
- **Blockwise Parallel Decoding (Stern et al. 2018):** giải mã song song nhiều token nhưng chỉ hỗ trợ greedy, cần huấn luyện mô hình riêng. **Shallow Aggressive Decoding** chỉ chép đầu vào sang đầu ra.

Quan sát then chốt của paper: (1) có bước "khó" cần mô hình lớn và bước "dễ" mô hình nhỏ xấp xỉ tốt; (2) suy luận mô hình lớn thường **bị nghẽn bởi băng thông bộ nhớ và giao tiếp chứ không phải số phép toán**, nên còn dư năng lực tính toán để chạy song song nhiều vị trí. Đây là cùng hiện tượng memory-bound ở bài 44-46.

Gloss:
- **Speculative execution (thực thi suy đoán):** kỹ thuật bộ xử lý, ví dụ dự đoán rẽ nhánh: thực hiện công việc trước khi biết có cần không, rồi kiểm tra; trả giá bằng tính song song tăng.
- **Lấy mẫu loại bỏ (rejection sampling):** lấy mẫu từ phân phối $q$ dễ, chấp nhận ngẫu nhiên để thu được mẫu theo phân phối $p$ khó.
- **Total variation (TV):** khoảng cách giữa hai phân phối $\tfrac12\sum_x|p(x)-q(x)|$.

## 2. Ý tưởng chính

Dùng mô hình nhỏ $M_q$ (nhanh) sinh nháp $\gamma$ token liên tiếp. Rồi chạy **một lượt** mô hình lớn $M_p$ trên cả $\gamma+1$ vị trí song song (nhờ tính song song giữa các vị trí, giống prefill), thu được các phân phối $p_1,\dots,p_{\gamma+1}$ của mô hình lớn. Với từng token nháp $x_i\sim q_i$, chấp nhận nó với xác suất $\min(1,p_i(x_i)/q_i(x_i))$; token đầu tiên bị từ chối được **lấy mẫu lại từ phân phối hiệu chỉnh** $\mathrm{norm}(\max(0,p-q))$; nếu tất cả được chấp nhận, thêm một token thưởng từ $p_{\gamma+1}$. Quy tắc này **chứng minh** cho ra đúng phân phối của $M_p$, nên không đổi đầu ra. Mỗi lượt chạy mô hình lớn cho ra **từ 1 tới $\gamma+1$ token** (không bao giờ ít hơn giải mã thường). Không cần huấn luyện lại, không đổi kiến trúc.

## 3. Phương pháp chi tiết

Ký hiệu: $M_p$ mô hình đích (target) với phân phối $p(x_t\mid x_{<t})$; $M_q$ mô hình nháp (approximation/draft) với $q(x_t\mid x_{<t})$; $\gamma\in\mathbb Z^+$ số token nháp.

### 3.1 Chuẩn hoá về lấy mẫu từ một phân phối

Các cách lấy mẫu (argmax, top-$k$, nucleus, nhiệt độ) đều quy về **lấy mẫu chuẩn từ một phân phối đã hiệu chỉnh**. Ví dụ argmax = đặt bằng 0 các phần tử không phải max rồi chuẩn hoá (phân phối one-hot). Từ đây $p$ và $q$ là phân phối **sau khi** áp phương pháp lấy mẫu, và ta chỉ cần chứng minh cho lấy mẫu chuẩn.

### 3.2 Speculative sampling: quy tắc chấp nhận

Mục tiêu: lấy mẫu $x\sim p(x)$ trong khi chỉ lấy mẫu rẻ từ $q$.

1. Lấy $x\sim q(x)$ và $r\sim U(0,1)$.
2. Nếu $q(x)\le p(x)$: **giữ** $x$ (luôn chấp nhận). Nếu $q(x)>p(x)$: chấp nhận với xác suất $p(x)/q(x)$, tức nhận nếu $r\le p(x)/q(x)$. Gộp lại: chấp nhận với xác suất $\min\!\big(1,\tfrac{p(x)}{q(x)}\big)$.
3. Nếu bị từ chối: lấy mẫu lại từ phân phối **hiệu chỉnh** (residual)
$$p'(x)=\frac{\max\big(0,\,p(x)-q(x)\big)}{\sum_{x'}\max\big(0,\,p(x')-q(x')\big)}.$$

Trực giác: $q$ có thể "quá nhiều" ở một số token (nơi $q>p$) và "quá ít" ở token khác ($q<p$). Việc từ chối bớt các token quá nhiều, rồi bù phần thiếu bằng $p'$, chính xác là phần dư của $p$ chưa được $q$ phủ.

### 3.3 Chứng minh phân phối không đổi

Đặt acceptance probability tổng thể $\beta=\Pr[\text{chấp nhận}]=\mathbb E_{x\sim q}\big[\min(1,p(x)/q(x))\big]=\sum_x \min\big(q(x),p(x)\big)$ (vì $q(x)\min(1,p/q)=\min(q,p)$).

**Chuẩn hoá của $p'$.** Vì $\max(0,p-q)=p-\min(p,q)$,
$$\sum_x\max(0,p(x)-q(x))=\sum_x p(x)-\sum_x\min(p(x),q(x))=1-\beta,$$
nên $p'(x)=\dfrac{p(x)-\min(p(x),q(x))}{1-\beta}$.

**Xác suất đầu ra bằng $x_0$:**
$$\Pr[x=x_0]=\underbrace{\Pr[\text{chấp nhận},\,x=x_0]}_{\text{nháp }x_0\text{ được nhận}}+\underbrace{\Pr[\text{từ chối},\,x=x_0]}_{\text{lấy lại từ }p'}.$$

- Số hạng đầu: $q(x_0)\cdot\min\!\big(1,\tfrac{p(x_0)}{q(x_0)}\big)=\min\big(q(x_0),p(x_0)\big)$.
- Số hạng hai: xác suất từ chối $=1-\beta$, rồi chọn $x_0$ theo $p'$: $(1-\beta)\,p'(x_0)=p(x_0)-\min\big(p(x_0),q(x_0)\big)$.

Cộng lại:
$$\boxed{\Pr[x=x_0]=\min(p(x_0),q(x_0))+p(x_0)-\min(p(x_0),q(x_0))=p(x_0).}$$

Điều này đúng với **mọi** $p,q$ (miễn $q(x)>0$ ở chỗ cần chia; token nào $q=0$ thì không bao giờ được đề xuất và do $p'$ hay phần thiếu bù đắp). Do đó chất lượng mô hình nháp **không ảnh hưởng tính đúng đắn**, chỉ ảnh hưởng tốc độ.

**Ví dụ số (tự tính).** $p=[0.6,0.3,0.1]$, $q=[0.3,0.5,0.2]$.
- $\min(p,q)=[0.3,0.3,0.1]$ nên $\beta=0.7$.
- $\max(0,p-q)=[0.3,0,0]$, chuẩn hoá theo $1-\beta=0.3$ được $p'=[1,0,0]$.
- Xác suất ra token 1: $\min=0.3$ (nhận) $+\,(1-\beta)p'(1)=0.3\cdot1=0.3$, tổng 0.6 = $p(1)$. Token 2: $0.3+0=0.3$. Token 3: $0.1+0=0.1$. Khớp $p$.

**So với rejection sampling cổ điển (Phụ lục A.2).** Rejection sampling dùng hằng số $M=\max_x p(x)/q(x)$ và nhận $x\sim q$ khi $r<\frac{p(x)}{Mq(x)}$, nếu từ chối thì lặp lại. Xác suất nhận của nó là $\mathbb E_{x\sim q}\big[\tfrac{p(x)}{Mq(x)}\big]=\sum_x p(x)\min_{x'}\tfrac{q(x')}{p(x')}\le\sum_x p(x)\min\big(1,\tfrac{q(x)}{p(x)}\big)=\sum_x\min(p,q)=\alpha$, tức **thấp hơn** (có thể thấp hơn nhiều) xác suất nhận của speculative sampling. Speculative sampling cũng không lặp vô hạn: khi từ chối nó lấy mẫu một lần từ $p'$.

**Đến cả dãy token.** Với $\gamma$ token nháp, xét theo thứ tự: tại vị trí $i$ (điều kiện trên tiền tố gồm các token đã chấp nhận), token được phát ra có phân phối $p_i$ theo chứng minh trên. Khi gặp từ chối đầu tiên tại vị trí $n+1$, ta bỏ mọi token nháp và tính toán phía sau (nó dựa trên tiền tố không còn đúng). Quy nạp theo vị trí: mỗi token phát ra có phân phối đúng $p(\cdot\mid\text{tiền tố đã phát})$, nên toàn dãy có phân phối đúng $\prod_t p(x_t\mid x_{<t})$. Nếu cả $\gamma$ token được nhận, token thưởng thứ $\gamma+1$ lấy trực tiếp từ $p_{\gamma+1}$ (phân phối của $M_p$ trên tiền tố đầy đủ).

### 3.4 Thuật toán giải mã nhiều token

```text
SpeculativeDecodingStep(Mp, Mq, prefix, gamma):
    # 1) Mq nháp gamma token, tuần tự
    for i = 1..gamma:
        q_i(x) <- Mq(prefix + [x_1..x_{i-1}])
        x_i    ~ q_i
    # 2) Mp đánh giá song song gamma+1 vị trí (một lượt chạy)
    p_1..p_{gamma+1} <- Mp(prefix), Mp(prefix+[x_1]), ..., Mp(prefix+[x_1..x_gamma])
    # 3) Số token nháp được chấp nhận
    r_1..r_gamma ~ U(0,1)
    n <- min({ i-1 : r_i > p_i(x_i)/q_i(x_i) } ∪ {gamma})
    # 4) Hiệu chỉnh phân phối cho token cuối
    p'(x) <- p_{n+1}(x)
    if n < gamma:
        p'(x) <- norm(max(0, p_{n+1}(x) - q_{n+1}(x)))
    t ~ p'
    return prefix + [x_1..x_n, t]            # sinh n+1 token, từ 1 đến gamma+1
```

Hình 1 của paper (mô hình 97M đích, nháp 6M, lm1b): câu 38 token được sinh bằng chỉ 9 lượt chạy tuần tự của mô hình lớn (dòng đầu: một lượt, 5 token).

Hiệu quả vì sao có: một lượt chạy $M_p$ trên $\gamma+1$ vị trí tốn gần bằng một lượt trên 1 vị trí khi bị nghẽn bộ nhớ (đọc trọng số và KV một lần), nhưng cho ra nhiều token.

### 3.5 Tỉ lệ chấp nhận $\alpha$ và độ lệch $D_{LK}$

**Định nghĩa 3.1.** $\beta_{x_{<t}}$ là xác suất chấp nhận $x_t\sim q$ cho tiền tố đã cho; $\alpha=\mathbb E(\beta)$ (kỳ vọng trên tiền tố). $\alpha$ là thước đo mô hình nháp xấp xỉ mô hình đích tốt đến đâu.

**Định nghĩa 3.2.** $D_{LK}(p,q)=\sum_x|p(x)-M(x)|=\sum_x|q(x)-M(x)|$ với $M=\tfrac{p+q}{2}$.

**Bổ đề 3.3.** $D_{LK}(p,q)=1-\sum_x\min(p(x),q(x))$. *Chứng minh:* $D_{LK}=\sum_x\frac{|p-q|}{2}$; dùng $|p-q|=p+q-2\min(p,q)$ được $D_{LK}=\sum_x\big(\frac{p+q}{2}-\min(p,q)\big)=1-\sum_x\min(p,q)$ (vì $\sum_x\frac{p+q}{2}=1$). Hệ quả: $D_{LK}$ đối xứng, trong $[0,1]$, bằng 0 khi $p=q$ và bằng 1 khi hai phân phối có giá không giao nhau.

**Định lý 3.5.** $\beta=1-D_{LK}(p,q)$. Do đó (Hệ quả 3.6) $\alpha=1-\mathbb E[D_{LK}(p,q)]=\mathbb E[\sum_x\min(p,q)]$.

Nhận xét của ta: $D_{LK}$ chính là **khoảng cách total variation** $\mathrm{TV}(p,q)=\tfrac12\sum|p-q|$ (một cách tính nữa: $\tfrac12\sum|p-q|=1-\sum\min(p,q)$). Nên $\alpha=1-\mathbb E[\mathrm{TV}(p,q)]$. Đây là cầu nối trực tiếp tới distillation (Mục 8).

Trường hợp greedy ($p$ one-hot tại $a=\arg\max p$ sau hiệu chỉnh): nếu $q$ cũng được hiệu chỉnh thành one-hot tại $b$, thì $\beta=1$ nếu $a=b$ và $0$ nếu khác. Tức là nháp được nhận khi **argmax hai mô hình trùng nhau**.

### 3.6 Số token kỳ vọng và công thức tăng tốc

Giả sử $\beta$ độc lập, cùng phân phối với kỳ vọng $\alpha$ (xấp xỉ đơn giản hoá). Số token nháp được nhận liên tiếp $n$ có $\Pr[n\ge k]=\alpha^k$ ($k\le\gamma$), và mỗi lượt sinh $n+1$ token:

$$\mathbb E[\#\text{token}]=1+\sum_{k=1}^{\gamma}\alpha^k=\sum_{k=0}^{\gamma}\alpha^k=\boxed{\frac{1-\alpha^{\gamma+1}}{1-\alpha}}\qquad(1)$$

(paper mô tả nó là biến hình học bị chặn với giới hạn $\gamma+1$). Khi $\gamma\to\infty$ giá trị tiến tới $\frac1{1-\alpha}$.

**Hệ số chi phí.** Gọi $c$ = (thời gian một lượt $M_q$)/(thời gian một lượt $M_p$) — phụ thuộc phần cứng và cài đặt (trong thí nghiệm của paper $c<0.05$, thường gần 0). Một lượt thuật toán tốn $T(c\gamma+1)$ (với $T$ là thời gian một lượt $M_p$). Giả sử đủ tài nguyên để chạy $\gamma+1$ vị trí song song không tăng thời gian. Thời gian mỗi token $=\frac{T(c\gamma+1)(1-\alpha)}{1-\alpha^{\gamma+1}}$, chia cho $T$ của giải mã thường:

$$\boxed{\text{Speedup}(\alpha,\gamma,c)=\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}}\qquad\text{(Định lý 3.8)}$$

**Điều kiện có lợi.** Hệ quả 3.9: nếu $\alpha>c$ tồn tại $\gamma$ cho tăng tốc, và hệ số tối thiểu là $\frac{1+\alpha}{1+c}$ (lấy $\gamma=1$: $\frac{1-\alpha^2}{(1-\alpha)(1+c)}=\frac{1+\alpha}{1+c}$; nếu có lợi với $\gamma$ thì cũng có lợi với mọi $\gamma^*<\gamma$). Trường hợp nháp miễn phí ($c=0$): tăng tốc $=\frac{1-\alpha^{\gamma+1}}{1-\alpha}\le\frac1{1-\alpha}$.

**Kiểm tra với bảng của paper (Bảng 4; $\gamma=7$, $\alpha=0.75$, $c=0.02$, T5-small):**
$\frac{1-0.75^8}{0.25\,(7\cdot0.02+1)}=\frac{0.8999}{0.285}\approx3.16$, paper ghi dự đoán 3.2 (đo được 3.4). Kiểm tra bigram: $\alpha\approx0.2$, $\gamma=3$, $c\approx0$: $\frac{1-0.2^4}{0.8}=1.248\approx1.25\times$, đúng như paper.

### 3.7 Số phép toán và chọn $\gamma$

Gọi $\hat c$ là tỉ số số phép toán số học mỗi token của $M_q$ so với $M_p$. Một lượt thuật toán: $\gamma$ lần $M_q$ và $\gamma+1$ lần $M_p$ song song, tức $\hat T(\hat c\gamma+\gamma+1)$ phép toán ($\hat T$ là phép toán của một lượt $M_p$). Chia cho số token kỳ vọng:

$$\text{Hệ số tăng phép toán}=\frac{(1-\alpha)(\gamma\hat c+\gamma+1)}{1-\alpha^{\gamma+1}}\qquad\text{(Định lý 3.11)}$$

Khi nháp bị từ chối thì phần tính toán dư bị phí; khi nháp được nhận thì sự song song là "miễn phí". Còn **số lần truy cập bộ nhớ có thể giảm**: trọng số và KV cache của $M_p$ được đọc một lần mỗi lượt thay vì mỗi token, nên số lần đọc giảm hệ số $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$. Cũng nhận xét: tổng phép toán của $M_p$ (không tính $M_q$) bị chặn bởi một lượt chạy bộ mã hoá cùng kích thước.

Một số cặp ($c=\hat c=0$) trong Bảng 1 của paper:

| $\alpha$ | $\gamma$ | Phép toán | Tốc độ |
|---|---|---|---|
| 0.6 | 2 | 1.53x | 1.96x |
| 0.7 | 3 | 1.58x | 2.53x |
| 0.8 | 2 | 1.23x | 2.44x |
| 0.8 | 5 | 1.63x | 3.69x |
| 0.9 | 2 | 1.11x | 2.71x |
| 0.9 | 10 | 1.60x | 6.86x |

**Chọn $\gamma$:** cực đại hoá Định lý 3.8 theo $\gamma$ nguyên (tìm số, Hình 3). $\gamma$ tăng thì số token kỳ vọng tăng nhưng chi phí nháp $\gamma c$ và phí tính toán cũng tăng. Vì $\beta$ không cố định, thay đổi $\gamma$ động theo $\beta$ dự đoán có thể cho thêm tới ~60% cải thiện so với $\gamma$ cố định (cận trên giả định oracle và đủ tài nguyên; paper để dành cho tương lai).

### 3.8 Chọn mô hình nháp, beam search, lenience

- **Mô hình nháp:** tính đúng đắn không phụ thuộc lựa chọn; thực tế chọn mô hình nhỏ hơn đích cỡ hai bậc độ lớn để cân bằng $\alpha$ và $c$. Các loại: Transformer nhỏ cùng kiến trúc; **mô hình chi phí bằng không** ($c\approx0$) như n-gram (tra bảng) hoặc heuristic chép từ ngữ cảnh khi chuỗi hay lặp (tóm tắt, hội thoại lặp), hay mô hình không tự hồi quy; cả mô hình chọn token ngẫu nhiên vẫn được cải thiện nhỏ.
- **Beam search (Phụ lục A.4):** áp dụng được với chi phí hơn: dùng beam rộng $u\ge w$ ở $M_q$ trong $\gamma$ bước, kiểm bằng $w+u\gamma$ lượt $M_p$; chấp nhận khi $\mathrm{top}_w(M_p)\subseteq\mathrm{top}_u(M_q)$.
- **Lenience (Phụ lục A.5):** nhân $q$ với $l\in[0,1]$ trước khi so với $p$ để tăng $\alpha$ đổi lấy sai khác nhỏ; vẫn đảm bảo không token nào có xác suất lớn hơn $p(x)/l$. Mọi kết quả chính của paper dùng phiên bản nghiêm ngặt ($l=1$).

## 4. Thí nghiệm và kết quả chính

**Tăng tốc thực đo trên T5-XXL (11B) với $M_q$ là T5-small (77M, "small f"), T5-base (250M), T5-large (800M); batch size 1, một TPU-v4**, so với T5X. Hai tác vụ: dịch Anh-Đức (WMT EnDe) và tóm tắt CNN/DM. Ghi chú: temp=0 là argmax, temp=1 là lấy mẫu chuẩn.

| Tác vụ | $M_q$ | Temp | $\gamma$ | $\alpha$ | Tốc độ |
|---|---|---|---|---|---|
| EnDe | T5-small | 0 | 7 | 0.75 | 3.4x |
| EnDe | T5-base | 0 | 7 | 0.8 | 2.8x |
| EnDe | T5-large | 0 | 7 | 0.82 | 1.7x |
| EnDe | T5-small | 1 | 7 | 0.62 | 2.6x |
| EnDe | T5-base | 1 | 5 | 0.68 | 2.4x |
| EnDe | T5-large | 1 | 3 | 0.71 | 1.4x |
| CNNDM | T5-small | 0 | 5 | 0.65 | 3.1x |
| CNNDM | T5-base | 0 | 5 | 0.73 | 3.0x |
| CNNDM | T5-large | 0 | 3 | 0.74 | 2.2x |
| CNNDM | T5-small | 1 | 5 | 0.53 | 2.3x |
| CNNDM | T5-base | 1 | 3 | 0.55 | 2.2x |
| CNNDM | T5-large | 1 | 3 | 0.56 | 1.7x |

Kết luận của paper: tăng tốc **2x-3x** (đỉnh 3.4x) với đầu ra giống hệt. T5-small cho tốc độ cao nhất dù $\alpha$ không cao nhất, vì cân bằng $\alpha$ và $c$.

**Dự đoán lý thuyết vs đo thực (Bảng 4, $c$ ước từ profiler):**

| Tác vụ | $M_q$ | Temp | $\gamma$ | $\alpha$ | $c$ | Dự đoán | Đo |
|---|---|---|---|---|---|---|---|
| EnDe | small | 0 | 7 | 0.75 | 0.02 | 3.2 | 3.4 |
| EnDe | base | 0 | 7 | 0.8 | 0.04 | 3.3 | 2.8 |
| EnDe | large | 0 | 7 | 0.82 | 0.11 | 2.5 | 1.7 |
| EnDe | large | 1 | 3 | 0.71 | 0.11 | 2.0 | 1.4 |
| CNNDM | small | 0 | 5 | 0.65 | 0.02 | 2.4 | 3.1 |
| CNNDM | large | 1 | 3 | 0.56 | 0.11 | 1.6 | 1.7 |

(Chọn 6 hàng; bảng gốc có 12.) Lệch do khác biệt tối ưu hoá giữa cài đặt của tác giả và baseline và do giả định $\beta$ độc lập chỉ gần đúng.

**Giá trị $\alpha$ (Bảng 3, ước lượng trên 10K token):**

| $M_p$ | $M_q$ | T=0 | T=1 |
|---|---|---|---|
| GPT-like 97M | unigram / bigram / GPT-like 6M | 0.03 / 0.05 / 0.88 | 0.03 / 0.05 / 0.89 |
| T5-XXL (EnDe) | unigram / bigram | 0.08 / 0.20 | 0.07 / 0.19 |
| T5-XXL (CNNDM) | unigram / bigram | 0.13 / 0.23 | 0.08 / 0.16 |
| LaMDA 137B | LaMDA 100M / 2B / 8B | 0.61 / 0.71 / 0.75 | 0.57 / 0.71 / 0.74 |

Quan sát của paper: nháp nhỏ hơn đích cỡ hai bậc độ lớn thường cho $\alpha\in[0.5,0.9]$; $\alpha$ tăng theo kích thước nháp; phân phối càng "nhọn" ($T=0$) thì $\alpha$ càng cao; n-gram tầm thường vẫn có $\alpha$ khác 0. Một thực nghiệm độc lập khác (Chen et al. 2023) được paper dẫn: 2x-2.5x trên Chinchilla 70B.

## 5. Phân tích / ablation

- **Kích thước nháp:** $M_q$ lớn hơn thì $\alpha$ cao hơn nhưng $c$ lớn hơn; T5-small tốt nhất về tốc độ, T5-large kém nhất (Bảng ở Mục 4).
- **Nhiệt độ:** $T=0$ cho $\alpha$ và tốc độ cao hơn $T=1$ (EnDe: 3.4x so với 2.6x với T5-small).
- **Chọn $\gamma$:** tối ưu phụ thuộc $(\alpha,c)$; $\gamma$ trong thí nghiệm là 3-7.
- **Phép toán so với tốc độ:** Bảng 1 cho thấy tăng tốc kèm tăng số phép toán (ví dụ $\alpha=0.9$, $\gamma=10$: 6.86x tốc độ, 1.60x phép toán).
- Paper không có ablation riêng ngoài các biến trên.

## 6. Giới hạn

Do tác giả nêu:
- Giả định có sẵn tài nguyên tính toán để chạy $\gamma+1$ vị trí song song không tăng thời gian; nếu không thì không có lợi.
- Tổng số phép toán **tăng** (kể cả khi tốc độ tăng).
- Giả định $\beta$ i.i.d. chỉ là xấp xỉ.
- Chỉ cài đặt và đo thực cho T5; các mô hình khác chỉ đo $\alpha$. Chưa thử miền khác ngoài văn bản (ví dụ ảnh).
- Chưa khai thác $\gamma$ thích ứng; chưa thử các phép biến đổi phân phối của nháp khác.

Do ta nhận thấy:
- Tất cả thí nghiệm tốc độ ở **batch size 1** trên TPU, các mô hình encoder-decoder T5 và một mô hình GPT nhỏ; chưa kiểm tra decoder-only lớn, GPU tiêu dùng, hay mô hình có kiến trúc hồi quy.
- Khi batch lớn, decode chuyển dần sang compute-bound và giả định "dư tài nguyên" mất đi; lợi ích giảm. Paper chủ yếu khẳng định cho trường hợp nghẽn bộ nhớ.
- Tính đúng đắn của phân phối giả định tính toán chính xác (số học lý tưởng); trong thực tế fp16/bf16 và khác biệt số học giữa lượt song song và lượt tuần tự có thể làm kết quả không giống từng bit.

## 7. Liên hệ với các bài khác

- **44-46 (FlashAttention, PagedAttention):** cùng gốc "decode là memory-bound"; speculative decoding tận dụng điều đó bằng cách tăng số token mỗi lượt đọc trọng số.
- **49 (Medusa), 50 (EAGLE):** thay mô hình nháp độc lập bằng các đầu dự đoán thêm hoặc dự đoán đặc trưng tầng cuối trong chính mô hình đích, thường dùng xác minh dạng cây; cùng khung chấp nhận/loại.
- **47 (SGLang):** "API speculative execution" ở đó là suy đoán khuôn mẫu, không phải thuật toán giữ phân phối như bài này.
- **Các bài distillation/pruning trong lộ trình:** mô hình nháp tốt chính là mô hình nhỏ có phân phối gần mô hình lớn; $\alpha=1-\mathbb E[\mathrm{TV}]$ là chỉ số trực tiếp (xem Mục 8).
- **55, 56 (StreamingLLM, H2O):** nén KV làm đổi phân phối; speculative decoding thì không đổi phân phối, hai cái đối lập về tính "lossless".

## 8. Ứng dụng cho dự án (suy luận)

Bối cảnh: LightOnOCR-3-0.8B; sau pruning còn ~0.3B/~0.1B, phục hồi bằng distillation từ teacher 4B; GPU ≤ 8 GB; batch 1; ~1,500 token/trang.

**(a) Batch-1 decode là đúng chế độ mà bài này nhắm tới.** Ví dụ tự tính: trọng số 0.8B ở bf16 = 1.6 GB; nếu băng thông bộ nhớ ~272 GB/s (giả định về loại GPU 8 GB), mỗi bước ≥ ~5.9 ms, 1,500 token ≥ ~8.8 s. Bước decode chỉ có 1 token nên cường độ số học rất thấp: GPU rỗi tính toán, đúng điều kiện "dư tài nguyên" của paper. Mô hình KV ở mô hình lai nhỏ (xem 46), nên kiểm tra $\gamma+1$ token gần như chỉ tốn thêm tính toán.

**(b) Mô hình nháp tự nhiên: chính mô hình đã cắt.** Bản ~0.1B hoặc ~0.3B đã distill từ cùng teacher là ứng viên $M_q$ cho mô hình đích 0.8B (hoặc ngược lại: 0.8B làm nháp cho teacher 4B nếu muốn chất lượng 4B ở độ trễ thấp hơn). Không cần huấn luyện thêm, đầu ra y hệt mô hình đích.

**(c) Ước tính tăng tốc (tự tính từ công thức Định lý 3.8, giả định $\alpha$, $c$):**

| $\alpha$ | $\gamma$ | token/lượt | tăng tốc, $c=0.15$ | tăng tốc, $c=0.4$ | phép toán, $c=0.15$ |
|---|---|---|---|---|---|
| 0.6 | 2 | 1.96 | 1.51x | 1.09x | 1.68x |
| 0.7 | 4 | 2.77 | 1.73x | 1.07x | 2.02x |
| 0.8 | 4 | 3.36 | 2.10x | 1.29x | 1.67x |
| 0.8 | 6 | 3.95 | 2.08x | 1.16x | 2.00x |
| 0.9 | 4 | 4.10 | 2.56x | 1.58x | 1.37x |
| 0.9 | 6 | 5.22 | 2.75x | 1.53x | 1.51x |

Tối ưu $\gamma$ theo công thức (tự tính): $(\alpha,c)=(0.8,0.15)$ cho $\gamma=5$, 2.11x; $(0.8,0.4)$ cho $\gamma=2$, 1.36x; $(0.9,0.15)$ cho $\gamma=8$, 2.78x. Với 8.8 s cơ sở, 2.1x tương đương ~4.2 s (lý thuyết, bỏ qua prefill).

Chú ý chọn $c$: $c$ **không** bằng tỉ lệ tham số. Mô hình 0.1B có $0.1/0.8=0.125$ tham số nhưng thời gian một bước của mô hình nhỏ bị chặn dưới bởi overhead launch kernel và số layer (số lớp ít giảm hơn số tham số), nên $c$ thực tế thường lớn hơn tỉ lệ tham số; $c=0.15$-$0.4$ là khoảng giả định hợp lý để thử, cần đo. Điều kiện sống còn: $\alpha>c$ (Hệ quả 3.9).

**(d) $\alpha$ của OCR có thể cao (giả thuyết).** Đầu ra OCR bị ràng buộc rất mạnh bởi ảnh, entropy mỗi token thấp (phần lớn token là chép chữ nhìn thấy). Paper chỉ ra phân phối càng nhọn thì $\alpha$ càng cao (T=0 hơn T=1). Nên nếu giải mã greedy cho OCR, hai mô hình chỉ cần đồng ý về argmax. Đây là dự đoán của ta, chưa kiểm chứng; $\alpha$ cần đo trực tiếp.

**(e) Kết nối với distillation: $\alpha=1-\mathbb E[\mathrm{TV}(p,q)]$.** Mục tiêu "nháp được chấp nhận nhiều" tương đương "giảm khoảng cách TV giữa phân phối mô hình nháp và mô hình đích". Vì vậy, khi distill mô hình cắt, có thể theo dõi $\alpha$ như một chỉ số chất lượng phụ: cùng là "gần" phân phối đích, nhưng đo chính xác thứ quyết định tốc độ. Ghi chú: KL và TV liên hệ qua bất đẳng thức Pinsker ($\mathrm{TV}\le\sqrt{\mathrm{KL}/2}$), kiến thức nền chứ không phải từ paper; nghĩa là giảm KL (mất mát distillation thường dùng) cũng ép TV xuống, nhưng không phải tối ưu trực tiếp cho $\alpha$. Cũng lưu ý: mô hình bị cắt theo D_lin/D_res vẫn có thể có $\alpha$ cao nếu sai lệch tập trung ở token hiếm.

**(f) Hai điều cản trở thực tế với mô hình lai (suy luận):**
1. Với lớp full attention, bị từ chối chỉ cần cắt KV về độ dài đúng. Với lớp GatedDeltaNet, trạng thái hồi quy đã bị cập nhật qua các token nháp bị từ chối; cần **lưu trạng thái tại mỗi vị trí kiểm tra** hoặc **tính lại** từ trạng thái trước lượt kiểm tra. Điều này tăng chi phí bộ nhớ/tính toán.
2. Kiểm tra $\gamma+1$ token cần kernel chạy được ở chế độ nhiều token (chunk song song) cho các lớp GatedDeltaNet; kernel này hiệu quả với nhiều token hơn là một token.

**(g) Không thay thế pruning, mà bổ sung:** pruning giảm byte đọc mỗi bước; speculative decoding giảm số bước của mô hình lớn. Chúng nhân vào nhau chừng nào mô hình vẫn memory-bound. Nhưng khi mô hình đích đã nhỏ (~0.3B) thì decode bị giới hạn bởi overhead launch hơn bởi băng thông và $c$ tăng, nên lợi ích giảm (thử bằng bảng ở (c) với $c=0.4$).

## 9. Thí nghiệm gợi ý

1. **Đo $\alpha$ thực:** chạy mô hình đích 0.8B và bản cắt 0.1B/0.3B trên ~200 trang, teacher-forcing (cho cùng tiền tố do đích sinh); tính $\beta_t=\sum_x\min(p_t(x),q_t(x))$ mỗi bước và lấy trung bình $\alpha$. Làm cả với giải mã greedy (so sánh argmax). Đây là phép đo 1 tiếng, quyết định có đáng làm tiếp không.
2. **Đo $c$ thực:** thời gian một bước decode của nháp so với đích trên GPU đích (batch 1). Kết hợp với $\alpha$ để dự đoán tăng tốc theo Định lý 3.8 trước khi viết bất cứ mã suy đoán nào; chọn $\gamma$ bằng tìm số.
3. **Kiểm chứng tính đúng:** cài thử speculative sampling trên mô hình nhỏ (hoặc 2 mô hình giả), lấy 10^5 mẫu một bước với $p,q$ biết trước và so histogram với $p$ (như ví dụ $p=[0.6,0.3,0.1]$) để chắc quy tắc cài đúng.
4. **Chi phí rollback trạng thái hồi quy:** đo thời gian lưu trạng thái GatedDeltaNet mỗi vị trí hoặc tính lại, so với lượt kiểm tra $\gamma+1$ token.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| Chấp nhận | nhận $x\sim q$ với xác suất $\min(1,p(x)/q(x))$ |
| Phân phối hiệu chỉnh | $p'=\mathrm{norm}(\max(0,p-q))$, hằng số chuẩn hoá $1-\beta$ |
| Tính đúng | $\Pr[x]=\min(p,q)+p-\min(p,q)=p$ |
| Xác suất chấp nhận | $\beta=\sum_x\min(p,q)=1-D_{LK}(p,q)$; $D_{LK}=\mathrm{TV}$ |
| $\alpha$ | $\mathbb E[\beta]$ |
| Token kỳ vọng mỗi lượt | $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ |
| Tăng tốc | $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$ |
| Chi phí nháp | $c$ = thời gian $M_q$ / thời gian $M_p$ |
| Điều kiện có lợi | $\alpha>c$; tối thiểu $\frac{1+\alpha}{1+c}$ |
| Tăng phép toán | $\frac{(1-\alpha)(\gamma\hat c+\gamma+1)}{1-\alpha^{\gamma+1}}$ |
| Cận trên ($c=0$, $\gamma\to\infty$) | $\frac1{1-\alpha}$ |
| Số token mỗi lượt | từ 1 đến $\gamma+1$ |

## 11. Kết luận ngắn

Speculative decoding dùng mô hình nháp rẻ và một lượt kiểm tra song song bằng mô hình đích, với quy tắc chấp nhận $\min(1,p/q)$ cộng phân phối hiệu chỉnh $\mathrm{norm}(\max(0,p-q))$ được chứng minh giữ nguyên phân phối đầu ra. Tăng tốc kỳ vọng $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}$ với $\alpha=1-\mathbb E[\mathrm{TV}]$; thực nghiệm 2x-3.4x trên T5-XXL ở batch 1. Với dự án: batch-1 decode là đúng điều kiện, mô hình đã cắt và distill là ứng viên nháp tự nhiên, $\alpha$ là chỉ số đo ngay được; trở ngại chính là $c$ của mô hình nhỏ và việc hoàn tác trạng thái hồi quy trong các lớp GatedDeltaNet.

Link gốc: https://arxiv.org/abs/2211.17192
