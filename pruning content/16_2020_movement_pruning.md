# Movement Pruning: Adaptive Sparsity by Fine-Tuning (2020)

**Tác giả:** Victor Sanh, Thomas Wolf, Alexander M. Rush (Hugging Face; Cornell).  
**Venue / arXiv:** arXiv:2005.07683 (v2, 23/10/2020); NeurIPS 2020 (34th Conference on Neural Information Processing Systems).  
**Nguồn:** https://arxiv.org/abs/2005.07683  
**Chủ đề:** Pruning trọng số (không cấu trúc) của mô hình pretrain trong lúc fine-tune, dùng điểm quan trọng bậc nhất học bằng straight-through estimator ("di chuyển ra xa 0" thay vì "độ lớn").  
**Vị trí trong lộ trình:** Cần đọc trước: 02 (Han 2015, magnitude pruning), 03 (Zhu & Gupta, lịch cắt cubic, dùng ở đây), 07 (Molchanov, điểm bậc nhất), 14 (Voita, $L_0$ Hard Concrete, là baseline ở đây), 15 (LTH BERT). Bài này thêm: một tiêu chí cắt mới phù hợp "pretrain rồi fine-tune". Kế tiếp: 17 (CoFi, cắt cấu trúc kết hợp distillation).

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Phần "Ứng dụng cho dự án" là suy luận của chúng tôi.

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

**Magnitude pruning** (giữ trọng số có $|w|$ lớn, Han 2015) là phương pháp cắt phổ biến nhất. Nó hợp lý khi học có giám sát từ đầu: giá trị trọng số do dữ liệu tác vụ quyết định. Nhưng trong **transfer learning** (pretrain trên dữ liệu khổng lồ, rồi fine-tune trên dữ liệu nhỏ có nhãn), giá trị trọng số phần lớn đã được quyết định bởi mô hình pretrain và chỉ nhích nhẹ khi fine-tune. Magnitude pruning khi đó gần như "biết trước" ngay từ trước fine-tune cái gì sẽ bị cắt (trọng số nhỏ lúc pretrain sẽ vẫn nhỏ) nên không thể thích nghi với tác vụ. Tác giả gọi việc cắt dựa trên fine-tune là **"fine-pruning"**.

Thuật ngữ:
- **Zeroth-order / first-order**: tiêu chí chỉ dùng giá trị trọng số (bậc 0) so với dùng cả gradient (bậc 1).
- **Straight-through estimator (STE)**: khi một phép toán (như top-$v$%) có gradient bằng 0 hầu khắp nơi, ở bước backward giả vờ nó là hàm đồng nhất để gradient "đi thẳng qua".
- **Hadamard product** $\odot$: nhân từng phần tử.
- **Distillation**: huấn luyện mô hình nhỏ bắt chước phân phối đầu ra của mô hình lớn (teacher).

## 2. Ý tưởng chính

Giữ lại các kết nối **đang di chuyển ra xa 0** trong khi fine-tune, thay vì các kết nối vốn đã lớn. Cụ thể, thêm một ma trận điểm quan trọng $S$ cùng kích thước với trọng số $W$, và để $S$ tích lũy $-\frac{\partial\mathcal L}{\partial W}\cdot W$ theo thời gian: trọng số nào cứ dương lên (gradient âm khi $W>0$) hoặc âm xuống thì điểm tăng; trọng số nào co về 0 thì điểm giảm. Mask = top-$v$% của $S$. Khác với magnitude pruning, cả trọng số nhỏ lẫn lớn đều có thể bị cắt hay giữ. Ở độ thưa cao (còn < 15% trọng số) cách này vượt rõ magnitude pruning, và vượt $L_0$ regularization; phiên bản "soft" còn tốt hơn.

## 3. Phương pháp chi tiết

### 3.1 Khung score-based pruning

$W\in\mathbb R^{n\times n}$ là một ma trận trọng số; $S\in\mathbb R^{n\times n}$ là ma trận điểm; mask $M\in\{0,1\}^{n\times n}$. Suy luận: $a=(W\odot M)x$. Hàm chọn:

$$\mathrm{Top}_v(S)_{i,j}=\begin{cases}1,& S_{i,j}\text{ nằm trong top }v\%\\0,&\text{ngược lại}\end{cases}$$

- **Magnitude pruning (MaP)**: $S_{i,j}=|W_{i,j}|$, $M=\mathrm{Top}_v(S)$.
- Họ dùng **automated gradual pruning** (Zhu & Gupta, bài 03): trọng số bị mask vẫn được cập nhật nên có thể "sống lại", và độ thưa tăng dần theo lịch cubic. Với $v_f$ mức giữ lại cuối, $v_i$ khởi đầu, $t_f$ bước cool-down, $N$ bước cắt, khoảng cách $\Delta t$:

$$v^{(t)}=\begin{cases}v_i&0\le t<t_i\\ v_f+(v_i-v_f)\Big(1-\frac{t-t_i-t_f}{N\Delta t}\Big)^3& t_i\le t<T-t_f\\ v_f&\text{ngược lại}\end{cases}$$

(Paper viết bằng "độ giữ lại" $v$, và nói thêm vài bước cool-down cuối "empirically improves performance especially in high sparsity".)

### 3.2 Movement pruning (hard)

Mask $M=\mathrm{Top}_v(S)$; **cả $W$ và $S$ cùng được học**. Forward: $a_i=\sum_k W_{i,k}M_{i,k}x_k$. Vì gradient của Top bằng 0 hầu khắp nơi, dùng STE: ở backward bỏ qua Top, gradient đi thẳng sang $S$:

$$\boxed{\frac{\partial\mathcal L}{\partial S_{i,j}}=\frac{\partial\mathcal L}{\partial a_i}\,W_{i,j}\,x_j=\frac{\partial\mathcal L}{\partial W_{i,j}}\,W_{i,j}}$$

(bỏ số hạng mask cho gọn). Hệ quả: điểm của trọng số **bị mask vẫn được cập nhật**, nên trọng số đã bị cắt có thể quay lại nếu điểm tăng. Sau $T$ bước cập nhật, với learning rate $\alpha_S$:

$$\boxed{S^{(T)}_{i,j}=-\alpha_S\sum_{t<T}\Big(\frac{\partial\mathcal L}{\partial W_{i,j}}\Big)^{(t)}W^{(t)}_{i,j}}$$

Nên $S$ là **bộ tích lũy chuyển động**. Ý nghĩa dấu: $S_{i,j}$ tăng khi $\partial\mathcal L/\partial S_{i,j}<0$, tức (a) $\partial\mathcal L/\partial W<0$ và $W>0$ (đang tăng lên khi dương), hoặc (b) $\partial\mathcal L/\partial W>0$ và $W<0$ (đang giảm khi âm); cả hai là $W$ đang rời xa 0. Ngược lại $S$ giảm khi $W$ co về 0.

Paper chứng minh (Phụ lục A.1) rằng bài toán tối ưu của movement pruning hội tụ (đảm bảo giảm loss với learning rate nhỏ); chi tiết chứng minh không đọc ở đây.

### 3.3 Soft movement pruning

Thay tỉ lệ cố định $v$ bằng **ngưỡng toàn cục** $\tau$: $M=(S>\tau)$. Để điều khiển độ thưa, thêm regularizer đẩy điểm giảm dần:

$$\boxed{R(S)=\lambda_{mvp}\sum_{i,j}\sigma(S_{i,j})}$$

với $\sigma$ là sigmoid; $\lambda_{mvp}$ càng lớn càng thưa. (Tác giả cũng thử $\sum|S_{i,j}|$ nhưng khó chỉnh hơn với kết quả tương tự.) Mục tiêu: $\mathcal L+R(S)$.

### 3.4 So sánh với $L_0$ regularization (Louizos 2017)

$L_0$ dùng Hard Concrete (như bài 14): mask lấy mẫu $u\sim U(0,1)$, $\bar S=\sigma\big((\log u-\log(1-u)+S)/b\big)$, $Z=(r-l)\bar S+l$, $M=\min(1,\mathrm{ReLU}(Z))$, với $b>0,l<0,r>1$; chuẩn $L_0$ kỳ vọng có dạng đóng. Gradient lên $S$ có cùng dạng $\frac{\partial\mathcal L}{\partial a_i}W_{i,j}x_j\,f(\bar S_{i,j})$ nhân một hàm $f$ (Eq. 4). Tác giả chỉ ra **movement pruning và soft movement pruning cho cập nhật tương tự $L_0$**, một phương pháp "movement" khác, nhưng đơn giản và tất định (không lấy mẫu).

Tóm tắt (Table 1 của paper):

| | Magnitude | $L_0$ reg | Movement | Soft movement |
|---|---|---|---|---|
| Bậc quyết định | 0 | 1 | 1 | 1 |
| Hàm mask | Top$_v$ | Hard-Concrete liên tục | Top$_v$ | ngưỡng |
| Cấu trúc | local hoặc global | global | local hoặc global | global |
| Mục tiêu | $\mathcal L$ | $\mathcal L+\lambda_{l0}\mathbb E(L_0)$ | $\mathcal L$ | $\mathcal L+\lambda_{mvp}R(S)$ |

### 3.5 Pseudo-code

```text
Khởi tạo: W <- BERT pretrain; S <- khởi tạo (điểm ban đầu); schedule v(t) (cubic)
for t = 1..T:
    M <- Top_{v(t)}(S)            # hard;   hoặc  M <- 1[S > tau]  # soft
    a <- (W ⊙ M) x                # forward
    L <- loss(a, nhãn) [+ distillation] [+ λ_mvp * Σ σ(S) nếu soft]
    # backward: Top bị bỏ qua (STE)
    W <- W - α_W * ∂L/∂W          # W được cập nhật cả khi M = 0
    S <- S - α_S * (∂L/∂W ⊙ W)    # tích lũy chuyển động
Kết thúc: cố định M, bỏ trọng số mask bằng 0
```

*Ví dụ nhỏ:* hai trọng số cùng $|W|=0.01$ ở thời điểm đầu. Trọng số A tăng đều lên 0.05 trong fine-tune; B co về 0.002. Magnitude pruning sau đó xếp B thấp hơn A, nhưng cả hai đã bị coi là "nhỏ" trước fine-tune. Movement pruning cho A điểm tăng, B điểm giảm bất kể độ lớn ban đầu; một trọng số lớn $|W|=0.3$ đang co về 0.2 cũng có thể bị cắt trước A.

## 4. Thí nghiệm và kết quả chính

**Thiết lập.** BERT-base-uncased (khoảng 84M tham số trong encoder và head; embedding bị **đóng băng**); fine-tune lớp transformer và head tác vụ. Mọi con số "độ thưa/remaining weights" tính trên BERT-base và bằng đúng kích thước mô hình, kể cả với baseline. Ba tác vụ: SQuAD v1.1 (hỏi đáp; paper ghi 8K ví dụ train, có thể là lỗi in vì bài 15 và 17 ghi ~88K), MNLI (393K), QQP (364K). Mỗi phương pháp cùng số bước cập nhật (6 đến 10 epoch). Baseline: MaP, $L_0$ regularization (Hard Concrete, unstructured), RPP (Guo 2019), LayerDrop (Fan 2020a, chỉ số báo từ tác giả gốc), mini-BERT (Turc 2019).

### 4.1 Kết quả chính (Table 2, không distillation; 10% = 8.5M tham số encoder, 3% = 2.6M)

| Tác vụ (metric), BERT-base fine-tune | Còn lại | MaP | $L_0$ | MvP | soft MvP |
|---|---|---|---|---|---|
| SQuAD dev EM/F1 (80.4/88.1) | 10% | 67.7/78.5 | 69.9/80.0 | 71.9/81.7 | 71.3/81.5 |
| | 3% | 40.1/54.5 | 61.2/73.3 | 65.2/76.3 | 69.5/79.9 |
| MNLI dev acc/MM acc (84.5/84.9) | 10% | 77.8/79.0 | 77.9/78.5 | 79.3/79.5 | 80.7/81.1 |
| | 3% | 68.9/69.8 | 75.1/75.4 | 76.1/76.7 | 79.0/79.6 |
| QQP dev acc/F1 (91.4/88.4) | 10% | 78.8/75.1 | 87.5/81.9 | 89.1/85.5 | 90.5/87.1 |
| | 3% | 72.1/58.4 | 86.5/81.0 | 85.6/81.0 | 89.3/85.6 |

Đọc nhanh: SQuAD 3%: MaP 54.5 F1; $L_0$ 73.3; MvP 76.3; soft MvP **79.9**.

- Mức thưa thấp (còn hơn 70% trọng số): **magnitude pruning thắng**, gần như không mất; còn movement pruning giảm nhanh ngay cả ở độ thưa thấp.
- Mức thưa cao (< 15% còn lại): các phương pháp bậc nhất vượt rõ; soft MvP luôn tốt nhất hoặc ngang tốt nhất (QQP: RPP ngang soft MvP). Hard MvP ngang $L_0$ (hơn rõ ở QQP).
- Movement pruning vượt mini-BERT (nhỏ, train từ đầu), nhất quán với Li et al. 2020 (huấn luyện mô hình lớn rồi nén hiệu quả hơn train mô hình nhỏ). Tác giả cảnh báo: phần cứng hiện tại chưa tối ưu suy luận cho mô hình thưa, nên mô hình dense nhỏ như mini-BERT vẫn có thể hợp lý hơn về tốc độ.

### 4.2 Kết hợp distillation (Table 3)

Teacher là chính BERT-base đã fine-tune; mục tiêu là tổ hợp lồi (convex combination) giữa loss huấn luyện và loss distillation trên phân phối đầu ra. Distillation tăng điểm mọi phương pháp và mọi mức, **thứ hạng giữ nguyên**:

| Tác vụ | Còn lại | MaP | $L_0$ | MvP | soft MvP |
|---|---|---|---|---|---|
| SQuAD EM/F1 | 10% | 70.2/80.1 | 72.4/81.9 | 75.6/84.3 | 76.6/84.9 |
| | 3% | 45.5/59.6 | 64.3/75.8 | 67.5/78.0 | 72.7/82.3 |
| MNLI acc/MM | 10% | 78.3/79.3 | 78.7/79.7 | 80.1/80.4 | 81.2/81.8 |
| | 3% | 69.4/70.6 | 76.0/76.2 | 76.5/77.4 | 79.5/80.1 |
| QQP acc/F1 | 10% | 79.8/65.0 | 88.1/82.8 | 89.7/86.2 | 90.2/86.8 |
| | 3% | 72.4/57.8 | 87.0/81.9 | 86.1/81.5 | 89.1/85.5 |

Ví dụ nêu trong paper: SQuAD, MvP 10%: 81.7 F1 → 84.3 F1. Soft MvP + distillation đạt **95% hiệu năng BERT-base chỉ với ~5% trọng số encoder** (SQuAD và MNLI), như abstract ghi "down to only 3% of the model parameters" với mất mát tối thiểu.

## 5. Phân tích / ablation

- **Phân phối trọng số còn lại (Figure 4a):** magnitude pruning loại sạch trọng số gần 0, còn lại hai cụm; movement pruning cho phân phối mượt phủ toàn khoảng, chỉ trừ vùng sát 0.
- **Điểm $S$ vs trọng số (Figure 4b):** không có quan hệ đơn giản; trọng số lớn hay nhỏ đều có thể quan trọng, nhưng điểm cao luôn đi cùng trọng số khác 0 (hình chữ V), khớp diễn giải "chuyển động ra xa 0". Figure 1: trong fine-tuning, trọng số ở sát giá trị pretrain; vì vậy magnitude pruning gần như quyết định từ trước.
- **Local và global:** hiệu năng tương đương ở hầu hết mức thưa; chỉ ở 3% còn lại, global hơn local 2.3 F1 (movement pruning). Phân bổ theo layer (Figure 6, còn 10%): các phương pháp bậc nhất toàn cục **giữ nhiều trọng số ở layer thấp và cắt nặng layer cao**; global magnitude pruning cắt gần đều các layer như local.
- **Soft vs hard:** soft thường tốt hơn hard và $L_0$, dù chỉ có thêm một regularizer đơn giản.
- **Số bước cắt:** ghi chú trong bài rằng tăng số bước cắt thường cải thiện kết quả cuối.

## 6. Giới hạn

**Do tác giả nêu:**
- Thưa không cấu trúc: phần cứng chưa tối ưu suy luận cho mô hình thưa; mô hình dense nhỏ có thể hợp lý hơn về tốc độ.
- Ở độ thưa thấp, magnitude pruning tốt hơn.
- Gợi ý hướng tương lai: dùng group sparsity (xóa cả cột/hàng) để cắt có cấu trúc, ví dụ loại cả cột của ma trận embedding.
- Phải có dữ liệu và bước fine-tune cho từng tác vụ (cắt cụ thể theo tác vụ).

**Do chúng tôi nhận thấy (suy luận):**
- Mỗi trọng số cần thêm một điểm $S$ và cập nhật STE: chi phí bộ nhớ train gần gấp đôi tham số (thêm $S$ và trạng thái optimizer), đáng kể khi áp cho mô hình 1B+ tham số.
- Chỉ BERT-base và 3 tác vụ; chỉ đo accuracy, không đo tốc độ; embedding bị đóng băng nên "độ thưa" không tính chúng.
- Tính chất "dựa vào chuyển động" giả định fine-tune đủ dài/đủ tín hiệu; với dữ liệu hiệu chuẩn nhỏ (vài trăm mẫu) điểm có thể nhiễu (chưa được thử).
- Checklist bài 12: có 3 tác vụ, nhiều điểm hoạt động (đường cong), nhưng không báo độ lệch chuẩn qua nhiều seed trong Table 2-3.

## 7. Liên hệ với các bài khác

- **02 Han 2015 / 03 Zhu & Gupta**: dùng lại magnitude pruning và lịch cubic; movement pruning chỉ thay tiêu chí chọn.
- **07 Molchanov 2019**: điểm Taylor bậc nhất $\propto \frac{\partial\mathcal L}{\partial W}W$ có cùng dạng; khác ở chỗ ở đây **tích lũy qua nhiều bước kèm dấu** thay vì tính một lần và bình phương/tuyệt đối. Paper nói họ giữ hướng chuyển động, khác các bài dùng $|$gradient$|$.
- **13 Sixteen Heads, 14 Voita**: cũng phương pháp bậc nhất/mask học được nhưng cho head; 14 dùng Hard Concrete đúng là baseline $L_0$ ở đây.
- **15 LTH BERT**: cùng bối cảnh BERT, cùng IMP cho ra mạng thưa, nhưng 16 tập trung độ thưa cực cao (3-10%) khi fine-tune.
- **17 CoFi**: lấy lại ý "cắt trong lúc fine-tune + distillation", nhưng chuyển sang cắt cấu trúc.
- **22-24 (OBC, SparseGPT, Wanda)**: pruning LLM sau train không có fine-tune; ngược với "fine-pruning" ở đây.
- **12 State of pruning**: ví dụ về vấn đề so sánh phụ thuộc mức nén: bài này cho thấy thứ hạng MaP vs MvP **đảo** theo độ thưa (MaP thắng ở thưa thấp).

## 8. Ứng dụng cho dự án (suy luận)

Mức liên quan: **trung bình về ý tưởng, thấp về kỹ thuật trực tiếp**.

1. **Không cấu trúc ⇒ không hợp mục tiêu GPU ≤ 8 GB batch-1**: ma trận thưa 90-97% hầu như không tăng tốc trên GPU phổ thông (cùng cảnh báo của chính paper). Chỉ giảm dung lượng lưu trữ nếu có định dạng thưa.
2. **Ý tưởng cốt lõi dùng được cho cấu trúc**: tính điểm cho *nhóm* trọng số (một neuron MLP = một hàng của $W_{up}$ + cột tương ứng của $W_{down}$; một head; một block) bằng tích lũy $-\sum \frac{\partial \mathcal L}{\partial W}\odot W$ nhóm lại. Đây là điểm bậc nhất có dấu, rẻ (có sẵn gradient) và tương thích với distillation loss (thay $\mathcal L$ bằng loss distill từ teacher 4B). Paper tự gợi ý group sparsity là hướng tương lai.
3. **Gợi ý lý giải quan sát "cắt MLP neuron hồi phục kém hơn cắt layer"** (giả thuyết): tiêu chí theo độ lớn/Taylor tĩnh trên mô hình đã pretrain dễ chọn sai neuron vì trọng số gần như cố định; một điểm tích lũy theo chuyển động *trong lúc distill* có thể chọn neuron phù hợp hơn với dữ liệu OCR. Cần kiểm chứng.
4. **Cảnh báo "thưa thấp thì magnitude tốt hơn"**: ở mức cắt nhẹ (ví dụ 0.8B → 0.6B) tiêu chí đơn giản có thể đủ; cải thiện của phương pháp động chỉ rõ ở mức cắt mạnh (0.1B).
5. **Distillation hầu như luôn cộng thêm** (Table 3: tăng cho mọi phương pháp, giữ thứ hạng) nên không cần chọn giữa "cắt thông minh" và "distill".
6. **Điểm S là một đại lượng chẩn đoán**: nhìn phân bố remaining theo layer (Figure 6: giữ lớp thấp, cắt lớp cao) cho giả thuyết về block nào ít quan trọng.

## 9. Thí nghiệm gợi ý

1. **Điểm chuyển động theo nhóm neuron**: trong một đợt distill ngắn từ teacher 4B, tích lũy $s_g=-\sum_t\big\langle \nabla_{W_g}\mathcal L,\,W_g\big\rangle$ cho từng neuron MLP $g$ của một vài block decoder; cắt theo $s_g$ và so với cắt theo $\|W_g\|_2$ và ngẫu nhiên ở cùng số tham số, sau cùng ngân sách distill. Đo CER.
2. **Mức thưa thấp vs cao**: lặp (1) ở mức cắt 10%, 30%, 60% neuron của một block; kiểm chứng có điểm giao cắt giống paper (tiêu chí tĩnh thắng ở mức nhẹ, động thắng ở mức nặng) không.
3. **Phân bổ theo layer**: dùng điểm $s_g$ toàn cục để xem layer nào được giữ nhiều/ít neuron; so với phân bổ đều.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| Suy luận với mask | $a=(W\odot M)x$ |
| Top-$v$ | $M=\mathrm{Top}_v(S)$ |
| Magnitude | $S=\lvert W\rvert$ |
| Movement (hard) | $\frac{\partial\mathcal L}{\partial S_{i,j}}=\frac{\partial\mathcal L}{\partial W_{i,j}}W_{i,j}$ (STE), $S^{(T)}=-\alpha_S\sum_{t<T}\frac{\partial\mathcal L}{\partial W}W$ |
| Soft movement | $M=(S>\tau)$, $R(S)=\lambda_{mvp}\sum\sigma(S_{i,j})$ |
| $L_0$ reg | mask Hard Concrete, $\mathbb E(L_0)=\sum\sigma(\log S_{i,j}-b\log(-l/r))$ |
| Lịch cắt | cubic, có bước cool-down $t_f$ |
| Kết quả | SQuAD 3% còn: MaP 54.5, $L_0$ 73.3, MvP 76.3, soft MvP 79.9 F1 |
| Với distillation | soft MvP đạt 95% BERT-base với ~5% trọng số encoder (SQuAD, MNLI) |

## 11. Kết luận ngắn

Khi mô hình đã pretrain và chỉ fine-tune nhẹ, độ lớn trọng số là một tiêu chí gần như "đóng băng". Chọn giữ trọng số đang di chuyển ra xa 0 (điểm $S$ tích lũy bằng STE) thích nghi với tác vụ và vượt rõ magnitude pruning, $L_0$ khi còn < 15% trọng số; soft MvP kết hợp distillation đạt 95% BERT-base với ~5% trọng số encoder. Hạn chế lớn là tính không cấu trúc. Với dự án, bài này hữu ích như một công thức điểm quan trọng bậc nhất có thể gộp theo nhóm và đi cùng distillation, chứ không phải một kỹ thuật dùng ngay.

Link gốc: https://arxiv.org/abs/2005.07683
