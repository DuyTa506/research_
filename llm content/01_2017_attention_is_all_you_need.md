# Attention Is All You Need (2017)

**Tác giả:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain / Google Research / Univ. of Toronto).  
**Venue / arXiv:** arXiv:1706.03762 (bản v7, 2023); NIPS 2017 (31st Conference on Neural Information Processing Systems).  
**Nguồn:** https://arxiv.org/abs/1706.03762  
**Chủ đề:** kiến trúc Transformer, scaled dot-product attention, multi-head attention, mã hoá vị trí sinusoidal, so sánh chi phí với RNN/CNN.  
**Vị trí trong lộ trình:** đọc sau bài 00 (preface). Đây là nền của mọi bài còn lại. Bài này bổ sung: bỏ hẳn recurrence/convolution, chỉ dùng attention. Tiếp theo: 02 (GPT-3, decoder-only quy mô lớn), 03 và 04 (scaling laws), 09 đến 12 (cải tiến từng khối), 29 (ViT đưa Transformer sang ảnh).

> Tài liệu học tập bằng tiếng Việt, ghi chú diễn giải, không phải bản dịch. Mọi con số lấy từ paper (trừ khi ghi "tự tính"). Phần "Ứng dụng cho dự án" là suy luận của chúng ta, không phải kết luận của paper.

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

Trước 2017, mô hình chuỗi (dịch máy, mô hình ngôn ngữ) chủ yếu là **RNN/LSTM/GRU** ("recurrent neural network": đọc từng token, giữ một trạng thái ẩn $h_t=f(h_{t-1},x_t)$). Có hai vấn đề:

1. **Tuần tự**: $h_t$ cần $h_{t-1}$, nên không song song hoá được trong một chuỗi. Paper nói rõ điều này "precludes parallelization within training examples", càng tệ với chuỗi dài vì bộ nhớ hạn chế kích thước batch.
2. **Phụ thuộc xa khó học**: tín hiệu giữa hai vị trí cách xa phải đi qua nhiều bước, gradient dễ tắt.

**Attention** ("chú ý": cho mỗi vị trí đầu ra tự chọn xem nên lấy thông tin từ những vị trí đầu vào nào) đã được dùng cùng RNN (Bahdanau và cộng sự, 2014) để giải vấn đề thứ hai, nhưng RNN vẫn còn đó. Các hướng khác (ConvS2S, ByteNet) dùng **tích chập** để song song hoá; nhưng số phép tính để nối hai vị trí xa tăng theo khoảng cách (tuyến tính với ConvS2S, logarit với ByteNet).

Câu hỏi của paper: có thể bỏ cả recurrence lẫn convolution, chỉ dùng attention, mà vẫn ra kết quả tốt và huấn luyện nhanh hơn không?

## 2. Ý tưởng chính

Transformer xếp chồng các tầng gồm hai việc: (i) **self-attention** (mỗi token nhìn mọi token khác trong cùng chuỗi và lấy trung bình có trọng số thông tin của chúng) và (ii) một mạng **feed-forward** áp riêng cho từng vị trí. Vì mỗi cặp vị trí nối với nhau trong **một** phép tính (độ dài đường đi $O(1)$) và mọi vị trí tính cùng lúc, huấn luyện song song rất tốt. Vì không còn thứ tự ngầm của RNN, phải **cộng mã hoá vị trí** vào embedding. Multi-head attention cho phép nhiều "góc nhìn" attention chạy song song. Kết quả: chất lượng dịch tốt hơn với chi phí huấn luyện thấp hơn nhiều.

## 3. Phương pháp chi tiết

### 3.1 Kiến trúc tổng thể

Encoder-decoder (như dịch máy):
- **Encoder**: $N=6$ tầng giống nhau; mỗi tầng = multi-head self-attention + FFN theo vị trí. Mỗi sub-layer bọc bằng residual và LayerNorm: $\text{LayerNorm}(x+\text{Sublayer}(x))$ (đây là **Post-LN**, so sánh với bài 09).
- **Decoder**: $N=6$ tầng, mỗi tầng thêm sub-layer thứ ba là **encoder-decoder (cross) attention** (query từ decoder, key/value từ đầu ra encoder). Self-attention của decoder có **causal mask** để vị trí $i$ chỉ nhìn $j\le i$; kết hợp với việc dịch đầu ra một vị trí để giữ tính tự hồi quy.
- Mọi sub-layer và embedding xuất chiều $d_{model}=512$ để residual cộng được.
- Embedding đầu vào, đầu ra và phép chiếu tuyến tính trước softmax **dùng chung trọng số** (weight tying); embedding nhân $\sqrt{d_{model}}$.

### 3.2 Scaled dot-product attention

Với $Q\in\mathbb{R}^{n\times d_k}$, $K\in\mathbb{R}^{m\times d_k}$, $V\in\mathbb{R}^{m\times d_v}$:

$$
\boxed{\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V}
$$

Lý do chia $\sqrt{d_k}$ (chú thích 4 của paper): nếu các thành phần của $q,k$ độc lập, trung bình 0, phương sai 1, thì $q\cdot k=\sum_{i=1}^{d_k}q_ik_i$ có trung bình 0 và phương sai $d_k$. Với $d_k$ lớn, điểm lớn đẩy softmax vào vùng gradient cực nhỏ. Paper so sánh với **additive attention** (mạng một tầng ẩn tính độ tương thích): lý thuyết tương đương, nhưng dot-product nhanh và tiết kiệm bộ nhớ hơn vì dùng được phép nhân ma trận tối ưu; additive tốt hơn dot-product *không scale* khi $d_k$ lớn, nên họ thêm hệ số scale.

Mask (decoder): đặt $-\infty$ vào các điểm tương ứng kết nối bất hợp lệ trước softmax.

### 3.3 Multi-head attention

$$
\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O,\quad \text{head}_i=\text{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

với $W_i^Q,W_i^K\in\mathbb{R}^{d_{model}\times d_k}$, $W_i^V\in\mathbb{R}^{d_{model}\times d_v}$, $W^O\in\mathbb{R}^{hd_v\times d_{model}}$. Paper dùng $h=8$, $d_k=d_v=d_{model}/h=64$, nên tổng tính toán gần bằng một head đầy đủ chiều. Lý do: một head duy nhất phải "lấy trung bình" nên mất thông tin từ nhiều không gian con; nhiều head cho phép "jointly attend to information from different representation subspaces at different positions".

Ba cách dùng attention trong mô hình: encoder-decoder attention, encoder self-attention, decoder masked self-attention.

### 3.4 FFN theo vị trí

$$
\text{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2,\qquad d_{model}=512,\ d_{ff}=2048.
$$

Giống hai tích chập kích thước 1. Cùng tham số cho mọi vị trí, khác nhau giữa các tầng.

### 3.5 Mã hoá vị trí sinusoidal

Không có recurrence/conv nên phải đưa thứ tự vào:

$$
PE_{(pos,2i)}=\sin\!\big(pos/10000^{2i/d_{model}}\big),\qquad PE_{(pos,2i+1)}=\cos\!\big(pos/10000^{2i/d_{model}}\big).
$$

Bước sóng tạo thành cấp số nhân từ $2\pi$ đến $10000\cdot2\pi$. Giả thuyết của tác giả: với mọi độ lệch cố định $k$, $PE_{pos+k}$ là hàm tuyến tính của $PE_{pos}$ nên mô hình dễ học chú ý theo vị trí tương đối. Họ cũng thử embedding vị trí học được và thấy kết quả "nearly identical" (Bảng 3, hàng E); chọn sinusoidal vì có thể ngoại suy sang chuỗi dài hơn lúc huấn luyện (đây là kỳ vọng, không được kiểm chứng trong bài).

### 3.6 Vì sao self-attention? (Mục 4, Bảng 1)

| Loại tầng | Độ phức tạp mỗi tầng | Số phép tuần tự | Độ dài đường đi tối đa |
|---|---|---|---|
| Self-attention | $O(n^2\cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n\cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k\cdot n\cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| Self-attention (hạn chế, láng giềng $r$) | $O(r\cdot n\cdot d)$ | $O(1)$ | $O(n/r)$ |

Ở đây $n$ là độ dài chuỗi, $d$ chiều biểu diễn, $k$ kích thước kernel. Self-attention nhanh hơn recurrent khi $n<d$ (thường đúng với biểu diễn word-piece/BPE trong dịch máy). Để xử lý chuỗi rất dài, đề xuất (chưa thử) giới hạn attention trong vùng bán kính $r$ (tiền thân của sliding window, bài 17). Phụ lục còn nói self-attention có thể dễ giải thích hơn.

### 3.7 Huấn luyện

- Dữ liệu: WMT 2014 EN-DE (~4.5M cặp câu, BPE chung ~37.000 token) và EN-FR (36M câu, word-piece 32.000). Mỗi batch ~25.000 token nguồn và 25.000 token đích.
- Phần cứng: 1 máy 8 GPU NVIDIA P100. Mô hình base: 0,4 s/bước, 100.000 bước (12 giờ). Mô hình big: 1,0 s/bước, 300.000 bước (3,5 ngày).
- Optimizer Adam ($\beta_1=0.9,\beta_2=0.98,\epsilon=10^{-9}$) với lịch learning rate:

$$
lrate=d_{model}^{-0.5}\cdot\min\!\big(step^{-0.5},\ step\cdot warmup^{-1.5}\big),\quad warmup=4000.
$$

  Tăng tuyến tính 4000 bước đầu, sau đó giảm theo $1/\sqrt{step}$.
- Regularization: dropout $P_{drop}=0.1$ (lên sub-layer output và tổng embedding+PE); **label smoothing** $\epsilon_{ls}=0.1$ (làm tệ perplexity nhưng tăng accuracy và BLEU).
- Suy luận: beam size 4, length penalty $\alpha=0.6$; trung bình 5 checkpoint cuối (base) hoặc 20 (big).

Pseudo-code một tầng encoder (Post-LN như paper):

```
def encoder_layer(x):                      # x: (n, d_model)
    a = multi_head_self_attention(x)       # Q=K=V=x, không mask
    x = layer_norm(x + dropout(a))
    f = relu(x @ W1 + b1) @ W2 + b2        # d_ff = 2048
    return layer_norm(x + dropout(f))
# decoder: thêm masked self-attn trước, và cross-attn(Q=decoder, K,V=encoder_out) ở giữa
```

Ví dụ đếm tham số (tự tính, d=512, bỏ bias/norm): self-attention $4\cdot512^2\approx1.05$M; FFN $2\cdot512\cdot2048\approx2.1$M; một tầng encoder ≈ 3.15M, 6 tầng ≈ 18.9M; mỗi tầng decoder thêm một attention → ≈ 4.2M, 6 tầng ≈ 25.2M; embedding chung $37000\cdot512\approx18.9$M. Tổng ≈ 63M, khớp với 65M ghi trong Bảng 3.

## 4. Thí nghiệm và kết quả chính

**Dịch máy (Bảng 2, newstest2014).** BLEU và chi phí huấn luyện (FLOPs, ước lượng bằng thời gian x số GPU x công suất FP32 duy trì: 2.8, 3.7, 6.0, 9.5 TFLOPS cho K80, K40, M40, P100):

| Mô hình | BLEU EN-DE | BLEU EN-FR | FLOPs EN-DE | FLOPs EN-FR |
|---|---|---|---|---|
| GNMT + RL | 24.6 | 39.92 | 2.3·10^19 | 1.4·10^20 |
| ConvS2S | 25.16 | 40.46 | 9.6·10^18 | 1.5·10^20 |
| MoE | 26.03 | 40.56 | 2.0·10^19 | 1.2·10^20 |
| GNMT + RL (ensemble) | 26.30 | 41.16 | 1.8·10^20 | 1.1·10^21 |
| ConvS2S (ensemble) | 26.36 | 41.29 | 7.7·10^19 | 1.2·10^21 |
| **Transformer (base)** | 27.3 | 38.1 | 3.3·10^18 | (cột trích bị lỗi) |
| **Transformer (big)** | **28.4** | **41.8** | 2.3·10^19 | (cột trích bị lỗi) |

Lưu ý về tính nhất quán: Bảng 2 và abstract ghi 41.8 BLEU cho EN-FR (big), nhưng đoạn văn mục 6.1 ghi "41.0". Ta tin Bảng 2/abstract; đây là chỗ không nhất quán trong paper. Trên EN-DE, big vượt kết quả trước đó (kể cả ensemble) hơn 2.0 BLEU; ngay cả base cũng vượt mọi mô hình đã công bố với chi phí nhỏ hơn. Với EN-FR, big đạt state-of-the-art đơn mô hình với chưa đến 1/4 chi phí huấn luyện của mô hình tốt nhất trước đó (theo paper).

**Constituency parsing (Bảng 4, WSJ section 23, F1).** Transformer 4 tầng, $d_{model}=1024$: 91.3 (chỉ WSJ, ~40K câu) và 92.7 (bán giám sát, ~17M câu). So: Dyer et al. 2016 (discriminative) 91.7; Vinyals & Kaiser 2014 bán giám sát 92.1; RNN Grammar (generative) 93.3. Paper nói Transformer tốt hơn BerkeleyParser ngay cả khi chỉ dùng 40K câu WSJ, và hầu hết các mô hình trước ngoại trừ RNNG.

## 5. Phân tích / ablation

Bảng 3 (EN-DE dev newstest2013, base: PPL 4.92, BLEU 25.8, 65M tham số; PPL tính theo word-piece):

| Biến thể | PPL | BLEU | Ghi chú |
|---|---|---|---|
| (A) 1 head ($d_k=512$) | 5.29 | 24.9 | single head kém 0.9 BLEU so với tốt nhất |
| (A) 4 head ($d_k=128$) | 5.00 | 25.5 | |
| (A) 16 head ($d_k=32$) | 4.91 | 25.8 | ngang base |
| (A) 32 head ($d_k=16$) | 5.01 | 25.4 | quá nhiều head cũng giảm |
| (B) $d_k=16$ | 5.16 | 25.1 | giảm $d_k$ làm tệ đi |
| (B) $d_k=32$ | 5.01 | 25.4 | |
| (C) $N=2$ | 6.11 | 23.7 | 36M tham số |
| (C) $N=4$ | 5.19 | 25.3 | |
| (C) $N=8$ | 4.88 | 25.5 | 80M |
| (C) $d_{model}=256$ | 5.75 | 24.5 | 28M |
| (C) $d_{model}=1024$ | 4.66 | 26.0 | 168M |
| (C) $d_{ff}=1024$ | 5.12 | 25.4 | |
| (C) $d_{ff}=4096$ | 4.75 | 26.2 | 90M |
| (D) dropout 0.0 / 0.2 | 5.77 / 4.95 | 24.6 / 25.5 | dropout giúp chống overfit |
| (D) $\epsilon_{ls}=0.0$ / 0.2 | 4.67 / 5.47 | 25.3 / 25.7 | label smoothing: PPL tệ, BLEU tốt |
| (E) positional embedding học được | 4.92 | 25.7 | gần như bằng sinusoidal |
| big (N=6, 1024, 4096, h=16, drop 0.3, 300K bước) | 4.33 | 26.4 | 213M |

Kết luận của tác giả: mô hình lớn hơn tốt hơn, dropout rất hữu ích; giảm $d_k$ hại chất lượng ("determining compatibility is not easy", gợi ý hàm tương thích tinh vi hơn có thể tốt hơn dot-product); vị trí học được và sinusoidal tương đương. Mục Phụ lục (không có trong text trích) còn hiển thị các head attention làm những việc khác nhau, một số liên quan tới cấu trúc cú pháp/ngữ nghĩa (đó là quan sát định tính).

## 6. Giới hạn

**Do tác giả nêu:**
- Self-attention tốn $O(n^2)$; với chuỗi rất dài cần attention hạn chế (hướng tương lai).
- Cần mở rộng sang các modality khác ngoài văn bản (ảnh, âm thanh, video) và các cơ chế attention cục bộ.
- Muốn làm cho việc sinh "less sequential" (decoder vẫn tự hồi quy).
- Mất độ phân giải hiệu dụng do lấy trung bình có trọng số (giảm nhẹ bằng multi-head).

**Do chúng ta nhận thấy:**
- Chỉ đánh giá dịch máy và parsing; không bàn đến mô hình ngôn ngữ quy mô lớn hay pre-training (đó là việc của bài 02 và các bài sau).
- Không có KV cache hay phân tích chi phí suy luận (chỉ phân tích huấn luyện). Tại suy luận tự hồi quy, chi phí thật bị chặn bởi băng thông bộ nhớ (bài 00, mục 10).
- Số liệu thiếu nhất quán nhỏ (41.0 so với 41.8). Ngoại suy chuỗi dài của PE sinusoidal chỉ là kỳ vọng; các bài sau (RoPE, ALiBi: 12, 13) cho thấy vấn đề này khó hơn.
- Post-LN cần warmup cẩn thận (lịch lr 4000 bước warmup); bài 09 phân tích kỹ.

## 7. Liên hệ với các bài khác

- **00** (preface): mọi công thức và shape của bài này được giải nghĩa chi tiết ở đó.
- **02** GPT-3: giữ khối Transformer, bỏ encoder, dùng decoder-only phóng to.
- **03, 04**: scaling laws dùng đúng công thức tham số $N\approx12\,L\,d^2$ và $C\approx6ND$ rút ra từ kiến trúc này.
- **09** Pre-LN, **10** RMSNorm, **11** SwiGLU, **12** RoPE, **13** ALiBi: thay từng chi tiết của bài này (vị trí LayerNorm, loại norm, FFN, mã hoá vị trí).
- **14, 15, 16**: giảm KV cache của multi-head attention.
- **17**: hiện thực hoá ý "restricted self-attention" mà bài này dự định.
- **18 đến 26**: thay softmax attention $O(n^2)$ bằng hồi quy tuyến tính, trạng thái cố định.
- **29** ViT: đưa chính kiến trúc này sang ảnh.
- **39, 40**: Donut, Nougat dùng encoder-decoder kiểu bài này cho OCR.
- **44, 45**: tối ưu cách tính đúng công thức attention ở mục 3.2 trên GPU.

## 8. Ứng dụng cho dự án (suy luận)

Mô hình LightOnOCR-3-0.8B là hậu duệ xa của bài này: ViT 12 khối là encoder bidirectional không mask; decoder là chồng khối "attention + FFN" có residual, nhưng nhiều khối attention đã được thay bằng GatedDeltaNet.

**Phân tích chi phí (tự tính, giả định $d_{model}\approx1024$, 24 tầng decoder, 1.500 token đầu ra, batch 1, BF16):**

- Tham số mỗi tầng chuẩn: attention $4d^2\approx4.2$M, FFN (hệ số 4) $8d^2\approx8.4$M → $12d^2\approx12.6$M mỗi tầng, 24 tầng ≈ 302M (chưa tính embedding và ViT; bạn cần đối chiếu với cấu hình thật vì kiến trúc dùng GQA/SwiGLU/GatedDeltaNet sẽ khác).
- FLOPs attention một tầng ở $n=1500$: $\approx4n^2d=4\cdot1500^2\cdot1024\approx9.2$ GFLOPs (prefill); FLOPs của phần chiếu+FFN cùng tầng: $24nd^2\approx37.7$ GFLOPs. Tức attention chỉ ≈ 20% FLOPs một tầng ở độ dài này; nút thắt của bạn không phải $n^2$ mà là **số byte đọc mỗi bước decode**.
- KV cache full-attention nếu cả 24 tầng đều giữ KV, 4 KV head, $d_{head}=128$, BF16, 1.500 token: $2\cdot24\cdot4\cdot128\cdot1500\cdot2\approx74$ MB (chi tiết ở bài 00). Chỉ các tầng full-attention của bạn mới phải trả khoản này.
- Với pruning: công thức $12Ld^2$ cho thấy cắt một tầng bỏ ≈ $12d^2$ tham số, trong khi cắt neuron MLP chỉ rút bớt chiều $d_{ff}$ (kích thước ma trận) mà vẫn giữ số tầng, nên đường đi tuần tự (độ sâu) không đổi. Với latency batch-1, giảm **độ sâu** vừa giảm byte đọc vừa giảm số kernel tuần tự; còn cắt chiều rộng giảm byte nhưng không giảm số kernel.

Điều bài này **không** giải quyết: suy luận hiệu quả, KV cache, token hoá ảnh.

## 9. Thí nghiệm gợi ý

1. **Tái hiện scale $\sqrt{d_k}$**: lấy $q,k\sim\mathcal{N}(0,I)$ với $d_k\in\{16,64,256,1024\}$, đo độ lệch chuẩn của $q\cdot k$ và entropy của softmax có và không chia $\sqrt{d_k}$; xác nhận phương sai $=d_k$.
2. **Đo tỷ lệ FLOPs attention / (chiếu+FFN) theo $n$** trên decoder của bạn bằng profiler: so với công thức $4n^2d$ và $24nd^2$, tìm $n$ mà $n^2$ bắt đầu thống trị (dự đoán $n\approx6d$).
3. **Head ablation**: chia head, tắt từng head ở decoder đầy đủ attention của bạn và đo ảnh hưởng tới character error rate OCR để biết head nào "thừa" (liên hệ trực tiếp bảng (A) ở trên và là gợi ý cho cắt head).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / giá trị |
|---|---|
| Attention | $\text{softmax}(QK^\top/\sqrt{d_k})V$ |
| Multi-head | $\text{Concat}(\text{head}_i)W^O$, $h=8$, $d_k=d_v=64$ |
| FFN | $\max(0,xW_1+b_1)W_2+b_2$, $d_{ff}=2048$ |
| PE | $\sin/\cos(pos/10000^{2i/d})$ |
| Learning rate | $d^{-0.5}\min(s^{-0.5},s\cdot4000^{-1.5})$ |
| Độ phức tạp self-attn | $O(n^2d)$, đường đi $O(1)$, tuần tự $O(1)$ |
| Base / big | 65M / 213M tham số; 28.4 / 41.8 BLEU (big) |
| Chi phí | base 12 giờ, big 3.5 ngày trên 8 P100 |

## 11. Kết luận ngắn

Transformer thay recurrence bằng attention, cho phép song song hoá toàn bộ, đường đi $O(1)$, và đạt SOTA dịch máy với chi phí huấn luyện nhỏ hơn. Chi phí $O(n^2)$ và thiếu phân tích suy luận là hai điểm các bài sau (hiệu quả attention, KV cache, hybrid) lần lượt giải quyết.

Nguồn gốc: https://arxiv.org/abs/1706.03762
