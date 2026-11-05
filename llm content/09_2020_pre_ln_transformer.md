# On Layer Normalization in the Transformer Architecture (2020)

**Tác giả:** Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, Tie-Yan Liu.  
**Venue / arXiv:** arXiv:2002.04745 (v2, 29/6/2020); Proceedings of the 37th International Conference on Machine Learning (ICML), PMLR 108, 2020.  
**Nguồn:** https://arxiv.org/abs/2002.04745  
**Chủ đề:** Vị trí của Layer Normalization (Post-LN vs Pre-LN), gradient lúc khởi tạo, learning rate warm-up, độ ổn định huấn luyện Transformer.  
**Vị trí trong lộ trình:** Đọc sau Attention Is All You Need (01), nơi Transformer gốc đặt LayerNorm *sau* phép cộng residual. Bài này giải thích bằng lý thuyết vì sao kiểu đó cần warm-up và vì sao đặt LayerNorm *bên trong* nhánh residual (Pre-LN) thì không. Tiếp theo: RMSNorm (10) thay chính phép chuẩn hoá bên trong khối Pre-LN bằng bản rẻ hơn; SwiGLU (11), RoPE (12) hoàn thiện "khối Transformer hiện đại" (như LLaMA 05).

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Các ứng dụng cho dự án (pruning/distillation OCR) là suy luận của chúng ta, không phải kết luận của paper. Chứng minh toán học đầy đủ nằm ở phụ lục (supplementary material) và **không có trong bản văn bản đã đọc**; ghi chú chỉ nêu phát biểu định lý và trực giác.

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

**Layer Normalization (LayerNorm):** với vector $v\in\mathbb R^d$, chuẩn hoá bằng trung bình và độ lệch chuẩn *của chính vector đó* rồi co giãn bằng tham số học được:

$$\mathrm{LayerNorm}(v)=\gamma\odot\frac{v-\mu}{\sigma}+\beta,\quad \mu=\tfrac1d\sum_k v_k,\ \ \sigma^2=\tfrac1d\sum_k(v_k-\mu)^2 .$$

Nó giúp giữ thang giá trị của kích hoạt ổn định, nên là một thành phần then chốt của Transformer.

**Residual connection (kết nối tắt):** $y=x+F(x)$, cho phép thông tin và gradient đi thẳng qua "đường cao tốc" $x$, giúp huấn luyện mạng sâu.

**Vấn đề.** Transformer gốc đặt LayerNorm *giữa các khối residual* (**Post-LN**): $x_{l+1}=\mathrm{LN}(x_l+F(x_l))$. Kiến trúc này cho kết quả rất tốt nhưng khó huấn luyện: bắt buộc có **learning rate warm-up**, tức bắt đầu với learning rate rất nhỏ rồi tăng dần đến $lr_{max}$ trong $T_{warmup}$ bước:

$$lr(t)=\frac{t}{T_{warmup}}\,lr_{max},\quad t\le T_{warmup}.$$

Warm-up làm chậm tối ưu và thêm siêu tham số nhạy cảm (Popel & Bojar 2018 cho thấy kết quả nhạy với $lr_{max}$ và số bước warm-up). Giải thích trước đó (Liu et al. 2019, RAdam) cho rằng warm-up giảm phương sai của learning rate thích nghi trong Adam. Paper này đặt câu hỏi: **có thể bỏ warm-up an toàn không, và tại sao nó cần?**

## 2. Ý tưởng chính

Bằng lý thuyết trường trung bình (mean field) tại **thời điểm khởi tạo**, paper chứng minh: ở Post-LN, gradient của các tham số gần lớp đầu ra có độ lớn **lớn và không giảm theo số lớp $L$**, nên dùng learning rate lớn ngay từ đầu làm huấn luyện bất ổn; warm-up là cách thực tế tránh điều đó. Ngược lại, ở **Pre-LN** (LayerNorm nằm *bên trong* nhánh residual, trước attention/FFN, cộng thêm một LayerNorm cuối trước dự đoán), độ lớn gradient nhỏ dần theo $\sqrt{L}$ và gần như đồng đều giữa các lớp, nên có thể bỏ warm-up. Thực nghiệm: Pre-LN không cần warm-up, hội tụ nhanh hơn, cho chất lượng tương đương.

## 3. Phương pháp chi tiết

### 3.1 Hai kiến trúc (Table 1, Figure 1)

Ký hiệu $x_{l,i}$ là đầu vào của lớp $l$ tại vị trí $i$, $d$ là chiều ẩn, $L$ là số lớp.

**Post-LN** (Transformer gốc):

$$\begin{aligned}
x^{post,1}_{l,i}&=\mathrm{MultiHeadAtt}(x_{l,i},[x_{l,1},\dots,x_{l,n}])\\
x^{post,2}_{l,i}&=x_{l,i}+x^{post,1}_{l,i}\\
x^{post,3}_{l,i}&=\mathrm{LN}(x^{post,2}_{l,i})\\
x^{post,4}_{l,i}&=\mathrm{ReLU}(x^{post,3}_{l,i}W^{1,l}+b^{1,l})W^{2,l}+b^{2,l}\\
x^{post,5}_{l,i}&=x^{post,3}_{l,i}+x^{post,4}_{l,i}\\
x_{l+1,i}&=\mathrm{LN}(x^{post,5}_{l,i})
\end{aligned}$$

**Pre-LN:**

$$\begin{aligned}
x^{pre,1}_{l,i}&=\mathrm{LN}(x_{l,i})\\
x^{pre,2}_{l,i}&=\mathrm{MultiHeadAtt}(x^{pre,1}_{l,i},[x^{pre,1}_{l,1},\dots,x^{pre,1}_{l,n}])\\
x^{pre,3}_{l,i}&=x_{l,i}+x^{pre,2}_{l,i}\\
x^{pre,4}_{l,i}&=\mathrm{LN}(x^{pre,3}_{l,i})\\
x^{pre,5}_{l,i}&=\mathrm{ReLU}(x^{pre,4}_{l,i}W^{1,l}+b^{1,l})W^{2,l}+b^{2,l}\\
x_{l+1,i}&=x^{pre,3}_{l,i}+x^{pre,5}_{l,i},\qquad \text{cuối cùng: } x_{Final,i}=\mathrm{LN}(x_{L+1,i})
\end{aligned}$$

Điểm khác biệt cốt lõi: ở Post-LN, mọi thứ đi qua LayerNorm sau mỗi phép cộng, kể cả "đường cao tốc" residual. Ở Pre-LN, đường residual **không bị chuẩn hoá**: $x_{L+1}=x_0+\sum_l(\text{đầu ra attention}+\text{FFN})$, tức một tổng cộng dồn, nên giữ nguyên tính chất "gradient đi thẳng" của residual.

### 3.2 Cài đặt cho phân tích lý thuyết (Mục 3.3)

Để phân tích được, paper đơn giản hoá: dùng single-head; mọi ma trận $W^Q,W^K,W^V,W^1,W^2$ đều $d\times d$; **khởi tạo $W^Q=W^K=0$** (nên attention là phân phối đều, $\mathrm{MultiHeadAtt}\to\frac1n\sum_j x_jW^V$); khởi tạo Xavier ($\mathcal N(0,\frac{2}{n_{in}+n_{out}})$) cho phần còn lại, bias 0, $\gamma=1$; đầu vào là Gaussian. Giả thiết chuẩn bình phương của trạng thái ẩn "tập trung" (định nghĩa $(\epsilon,\delta)$-bounded: xác suất $\ge1-\delta$ thì $\frac{|Z-\mathbb EZ|}{\mathbb EZ}\le\epsilon$).

### 3.3 Kết quả lý thuyết

**Lemma 1.** Nếu $X\sim\mathcal N(0,\sigma^2I_d)$ thì $\mathbb E\|\mathrm{ReLU}(X)\|_2^2=\tfrac12\sigma^2 d$.

**Lemma 2 (độ lớn trạng thái ẩn lúc khởi tạo).**
- Post-LN: $\mathbb E\|x^{post,5}_{l,i}\|_2^2=\tfrac32 d$ với mọi $l>0$, **không phụ thuộc độ sâu**.
- Pre-LN: $\big(1+\tfrac l2\big)d\le\mathbb E\|x^{pre}_{l,i}\|_2^2\le\big(1+\tfrac{3l}{2}\big)d$, tức **tăng tuyến tính theo độ sâu** $l$.

**Lemma 3.** $\|J_{LN}(x)\|_2=O\!\big(\tfrac{\sqrt d}{\|x\|_2}\big)$ với $J_{LN}=\partial\mathrm{LN}(x)/\partial x$. Trực giác: LayerNorm chia cho độ lớn của đầu vào, nên gradient bị chia cùng hệ số; đầu vào càng lớn thì gradient truyền ngược qua LN càng nhỏ.

**Theorem 1 (gradient của lớp FFN cuối).** Với xác suất cao (bản trích xuất ghi cận $0.99-\delta-\epsilon^{0.9+\epsilon}$; nên đối chiếu PDF):

$$\boxed{\text{Post-LN: }\Big\|\frac{\partial\tilde{\mathcal L}}{\partial W^{2,L}}\Big\|_F\le O\!\big(d\sqrt{\ln d}\big)\qquad\text{Pre-LN: }\Big\|\frac{\partial\tilde{\mathcal L}}{\partial W^{2,L}}\Big\|_F\le O\!\Big(d\sqrt{\tfrac{\ln d}{L}}\Big)}$$

Diễn giải: Post-LN có chuẩn gradient lớp cuối **không phụ thuộc $L$** vì đầu vào của LN cuối có độ lớn không đổi (Lemma 2). Pre-LN thì đầu vào của LN cuối có độ lớn bình phương $\sim L$, nên gradient bị chia $\sqrt L$ (qua Lemma 3).

**Mở rộng (phân tích sơ bộ, kết quả nêu không chứng minh đầy đủ trong thân bài):** ở Post-LN, chuẩn gradient lớn ở tham số gần đầu ra và có xu hướng giảm khi chỉ số lớp $l$ giảm; ở Pre-LN gần như giữ nguyên với mọi lớp $l$.

### 3.4 Từ lý thuyết đến thực hành

Nếu gradient gần đầu ra quá lớn thì bước cập nhật $lr\cdot g$ quá lớn nếu $lr$ lớn → bất ổn → cần warm-up. Với Pre-LN gradient "ngoan" ngay từ đầu → bỏ warm-up, dùng $lr$ lớn ngay.

**Pseudo-code hai khối:**

```python
def block_post_ln(x):                    # Transformer gốc
    x = LN(x + Attn(x))
    x = LN(x + FFN(x))
    return x

def block_pre_ln(x):                     # Pre-LN
    x = x + Attn(LN(x))
    x = x + FFN(LN(x))
    return x
# Toàn mạng Pre-LN:  h = x0; for l in layers: h = block_pre_ln(h)
#                    logits = W_emb @ LN(h)        # LN cuối bắt buộc
```

### 3.5 Ví dụ số nhỏ (tự tính từ Lemma 2)

Với $L=12$ lớp, $d=1$ (chuẩn hoá theo $d$): Post-LN có $\mathbb E\|x\|^2/d=1.5$ ở mọi lớp. Pre-LN ở lớp $l=12$: giữa $1+12/2=7$ và $1+3\cdot12/2=19$ lần $d$. Tức trạng thái ẩn Pre-LN lúc khởi tạo có chuẩn bình phương lớn hơn ~5–13 lần Post-LN ở lớp 12; LN cuối chuẩn hoá lại, và gradient truyền ngược qua nó bị chia khoảng $\sqrt{\text{tỉ số}}\approx$ 2.2–3.6 lần so với Post-LN (xấp xỉ thô, chỉ để cảm nhận bậc độ lớn, không phải số của paper).

## 4. Thí nghiệm và kết quả chính

**Nhiệm vụ.** IWSLT14 Đức→Anh (De-En), WMT14 Anh→Đức (En-De, cấu hình Transformer base), và tiền huấn luyện BERT (base) trên English Wikipedia + BookCorpus (~3.4B từ, tỉ lệ train:valid 199:1) rồi kiểm tra trên MRPC, RTE. Tất cả dùng Adam với $\beta=(0.9,0.98)$ trừ khi nói khác.

**(1) Warm-up là thiết yếu với Post-LN (IWSLT14, Mục 3.2):**
- Không warm-up, Adam: BLEU chỉ **8.45**; có warm-up ($T_{warmup}=4000$): khoảng **34** BLEU.
- Với SGD không warm-up: BLEU chỉ hơn 0 một chút sau 15 epoch.
- Nhạy cảm với $T_{warmup}$: $T_{warmup}=500$ cho BLEU **31.16** ($lr_{max}=5\text{e-}4$) và **2.77** ($lr_{max}=1\text{e-}3$).
- Warm-up cũng giúp SGD, nên (theo tác giả) lợi ích của nó không riêng cho Adam.

**(2) Xác minh lý thuyết:** chuẩn bình phương trạng thái ẩn khi khởi tạo thoả giả thiết "tập trung" ($(0.1,0.125)$-bounded). Chuẩn gradient của $W^{2,L}$ khi số lớp tăng (6-6 đến 14-14): Post-LN giữ ~**1.6**, Pre-LN **giảm**. Trong mô hình 6-6, chuẩn gradient kỳ vọng ở Post-LN tăng theo chỉ số lớp, ở Pre-LN gần như phẳng.

**(3) Huấn luyện Pre-LN không warm-up:** IWSLT14: $lr=5\text{e-}4$, giảm ×0.1 ở epoch 8; WMT14: $lr=7\text{e-}4$ hoặc $1.5\text{e-}3$, giảm ở epoch 6 rồi dùng lịch inverse-sqrt. Baseline Post-LN: warm-up 4000 bước. Kết quả: IWSLT14 BLEU ≈ 34 và validation loss ≈ 4, tương đương Post-LN. Ở cùng $lr_{max}$, **Pre-LN hội tụ nhanh hơn**: checkpoint epoch 9 của Pre-LN ≈ checkpoint epoch 15 của Post-LN (IWSLT14).

**(4) So với RAdam:** RAdam giúp huấn luyện Post-LN không cần warm-up, nhưng với Pre-LN gần như không khác Adam: thay đổi vị trí LN "chi phối" thay đổi bộ tối ưu.

**(5) BERT:** Post-LN theo Devlin: warm-up 10k bước, $lr_{max}=1\text{e-}4$ (thử $3\text{e-}4$ thì phân kỳ). Pre-LN: $3\text{e-}4$, giảm tuyến tính, không warm-up. Paper nêu Post-LN đạt validation loss 1.69 ở 500k cập nhật và Pre-LN đạt tương tự ở 700k cập nhật, kèm kết luận "40% speed-up". *Lưu ý:* đã đối chiếu với PDF: chính bản gốc viết đúng như vậy (Post-LN 500k, Pre-LN 700k). Thứ tự này mâu thuẫn với kết luận "40% speed-up", nên nhiều khả năng đây là lỗi đánh máy trong paper: ý đúng là Pre-LN đạt loss đó ở 500k còn Post-LN cần 700k. Khi trích dẫn, hãy nêu rõ chỗ không nhất quán này. Pre-LN cũng hội tụ nhanh hơn trên MRPC và RTE.

## 5. Phân tích / ablation

- **Cấu trúc quyết định, không phải bộ tối ưu.** Chỉ đổi vị trí LN thì loại bỏ được nhu cầu warm-up; RAdam chỉ giải quyết Post-LN.
- **Fixed small LR** ($10^{-4}$) cho Post-LN từ đầu: giảm bớt vấn đề nhưng hội tụ chậm hơn nhiều (kết quả trong phụ lục, được tóm tắt trong thân bài), ủng hộ giả thuyết "gradient quá lớn khi LR lớn".
- **Sau warm-up**, gradient của Post-LN đã nhỏ (Figure 3a–b) và mô hình train được với LR lớn: điều đó ủng hộ cách hiểu warm-up là để vượt qua giai đoạn khởi tạo xấu.
- **LN cuối bắt buộc** trong Pre-LN (kiến trúc của paper có "final layer normalization before prediction").

## 6. Giới hạn

**Do tác giả nêu/hàm ý:**
- Lý thuyết chỉ ở lúc khởi tạo, với nhiều đơn giản hoá (single-head, $W^Q=W^K=0$, kích thước ma trận $d\times d$, ReLU FFN, đầu vào Gaussian). Các tác giả thừa nhận cần kiểm chứng bằng thực nghiệm.
- Việc nghiên cứu các vị trí LN khác và tối ưu Transformer về mặt lý thuyết được đưa vào "tương lai".

**Do chúng ta nhận xét:**
- Chỉ đo chất lượng cuối cùng ở quy mô nhỏ-vừa (IWSLT14, WMT14 base, BERT base); không có thí nghiệm LLM hàng chục lớp trở lên. Paper không nói về hiệu năng cuối cùng của Pre-LN khi rất sâu; một số công trình sau (ngoài paper này) báo rằng Pre-LN có thể kém Post-LN một chút về chất lượng cuối khi so với Post-LN được huấn luyện tốt, và rằng chuẩn trạng thái ẩn Pre-LN phình theo độ sâu. Đây là hiểu biết ngoài paper, nên hãy coi là gợi ý cần kiểm tra.
- Xuất hiện nhiều tham số/biến thể (bộ tối ưu, lịch LR) giữa các nhánh so sánh; chênh lệch nhỏ nên đọc kỹ đường cong trước khi kết luận.
- Phụ lục chứa chứng minh nên người đọc không kiểm tra được ở bản văn bản thuần này.

## 7. Liên hệ với các bài khác

- **01 Attention Is All You Need:** bản Transformer gốc là Post-LN; bài này là nguyên nhân lý thuyết cho chuyện "phải warm-up".
- **02 GPT-3, 05 LLaMA, 06 Mistral:** các LLM hiện đại dùng chuẩn hoá *trước* mỗi sub-layer (kiểu Pre-LN, với RMSNorm thay LayerNorm) — theo hiểu biết của chúng ta ngoài paper; xác nhận trong từng bài tương ứng.
- **10 RMSNorm:** thay LayerNorm trong khối Pre-LN bằng bản bỏ trừ trung bình, rẻ hơn.
- **11 SwiGLU, 12 RoPE:** hai thành phần còn lại của khối LLM hiện đại.
- **44 FlashAttention:** tối ưu phép attention bên trong khối; chuẩn hoá và phép cộng residual là các phép elementwise, thường gộp (fuse) trong kernel suy luận.

## 8. Ứng dụng cho dự án (suy luận)

1. **Hiểu "đường residual" khi cắt lớp.** Trong khối Pre-LN, $x_{L+1}=x_0+\sum_l\Delta_l$ là *tổng* các đóng góp. Xoá một lớp bằng cách bỏ số hạng $\Delta_l$ ra khỏi tổng, các lớp sau vẫn nhận phần còn lại của dòng residual nguyên vẹn. Đó là lý do cắt lớp trong mô hình Pre-LN thường "êm" hơn so với cắt neuron bên trong MLP (nơi chiều ẩn bị thu nhỏ, các ma trận phía sau phải nhận đầu vào khác). Đây là *giả thuyết suy luận* nối với quan sát của bạn ("cắt MLP neuron phục hồi kém hơn cắt lớp"); paper không nói về pruning.
2. **Phân rã $D_{lin}$/$D_{res}$.** Vì dòng residual là tổng cộng dồn, sai lệch của lớp bị cắt là một số hạng bị mất; phần xấp xỉ được bằng ánh xạ tuyến tính tại điểm cắt chính là $D_{lin}$. Việc LayerNorm/RMSNorm nằm *trước* sub-layer (không nằm trên đường residual) làm cho phép tuyến tính hoá này sạch hơn so với Post-LN, nơi LN chèn lên chính đường residual. Cần xác nhận mô hình của bạn có thực sự dùng Pre-norm (LightOnOCR/Qwen3.5-style: khả năng cao, nhưng đây là giả định của chúng ta chứ không phải trích từ paper).
3. **Chuẩn trạng thái ẩn tăng theo độ sâu (Lemma 2).** Nếu bạn cắt bớt lớp, độ lớn trạng thái ẩn ở LN cuối thay đổi so với mô hình gốc (tổng ít số hạng hơn), nên **LN cuối (và gain $\gamma$ của nó) là ứng viên cần tinh chỉnh lại** khi phục hồi. Một cách rẻ: chỉ mở khoá các tham số chuẩn hoá cuối và head trong vài trăm bước đầu.
4. **Huấn luyện phục hồi không cần warm-up lâu** (suy luận): do Pre-LN gradient ổn định, có thể dùng $lr$ khá lớn ngay; nhưng vì mô hình đã pretrain, hãy thử nghiệm thay vì tin tuyệt đối (kết quả paper là huấn luyện từ đầu).
5. **Batch-1 / độ trễ:** việc đặt LN không ảnh hưởng FLOPs đáng kể; chi phí chủ yếu là số kernel elementwise (xem 10 RMSNorm). Pre-LN cần thêm 1 LN cuối so với số lớp; không đáng kể.

## 9. Thí nghiệm gợi ý

1. **Đo chuẩn trạng thái ẩn theo lớp** của mô hình gốc và mô hình đã cắt (trên vài trăm trang): vẽ $\|h_l\|_2$ theo $l$ và chuẩn tại LN cuối. Xem Lemma 2 (tăng tuyến tính) có đúng với mô hình đã huấn luyện không, và độ lệch sau khi cắt lớp.
2. **Tinh chỉnh chỉ norm cuối + head sau khi cắt lớp:** so sánh CER trước/sau; đây là thí nghiệm rẻ nhất để biết sai lệch do "thang" (scale) chiếm bao nhiêu trong $D_{res}$.
3. **Chuẩn gradient theo lớp lúc bắt đầu phục hồi** (như Figure 3): ghi $\|\partial L/\partial W\|_F$ cho mô hình đã cắt; nếu gradient gần đầu ra áp đảo thì nên dùng LR theo lớp (LR nhỏ hơn cho lớp cuối) hoặc warm-up ngắn.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Post-LN | $x\leftarrow\mathrm{LN}(x+F(x))$ sau mỗi sub-layer |
| Pre-LN | $x\leftarrow x+F(\mathrm{LN}(x))$, thêm LN cuối trước dự đoán |
| Warm-up | $lr(t)=\frac{t}{T_{warmup}}lr_{max}$ |
| Lemma 2 | Post-LN: $\mathbb E\|x\|^2=\tfrac32 d$ cố định; Pre-LN: $(1+\tfrac l2)d\le\cdot\le(1+\tfrac{3l}2)d$ |
| Lemma 3 | $\|J_{LN}(x)\|_2=O(\sqrt d/\|x\|_2)$ |
| Theorem 1 | Gradient $W^{2,L}$: Post $O(d\sqrt{\ln d})$; Pre $O(d\sqrt{\ln d/L})$ |
| Kết quả chốt | Post-LN không warm-up: BLEU 8.45 vs ~34; Pre-LN không warm-up đạt ~34; BERT nhanh hơn ~40% (theo paper) |

## 11. Kết luận ngắn

Paper chỉ ra rằng vị trí của LayerNorm quyết định độ lớn gradient lúc khởi tạo: Post-LN có gradient lớn gần đầu ra, độc lập với số lớp, nên cần warm-up; Pre-LN chia nhỏ gradient theo $\sqrt L$ và đồng đều, nên huấn luyện được không cần warm-up, nhanh hơn và cho chất lượng tương đương trên dịch máy và BERT. Với dự án của bạn, giá trị là hiểu cấu trúc "dòng residual là tổng các đóng góp" — nền tảng để suy luận vì sao cắt lớp và cắt neuron phục hồi khác nhau, và những tham số chuẩn hoá nào cần chỉnh lại sau khi cắt.

**Link gốc:** https://arxiv.org/abs/2002.04745
