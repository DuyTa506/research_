# Longformer: The Long-Document Transformer (2020)

**Tác giả:** Iz Beltagy, Matthew E. Peters, Arman Cohan (Allen Institute for AI)  
**Venue / arXiv:** arXiv:2004.05150 (bản v2, 12/2020); paper không nêu venue trong bản text đã đọc  
**Nguồn:** https://arxiv.org/abs/2004.05150  
**Chủ đề:** Sparse attention (cửa sổ trượt + global attention), độ phức tạp tuyến tính theo độ dài chuỗi, tiền huấn luyện tiếp từ RoBERTa, biến thể encoder-decoder LED.  
**Vị trí trong lộ trình:** Đọc sau 01 (Attention Is All You Need) và 09/12 (Pre-LN, RoPE) để nắm self-attention và vị trí. Bài này mở đầu nhánh "làm attention rẻ hơn theo độ dài": ở đây bằng cách **cắt bớt các cặp (query, key)** (sparse). Nhánh còn lại, *nén toàn bộ quá khứ vào một trạng thái cố định*, bắt đầu từ 18 (Linear Transformers). Cửa sổ trượt xuất hiện lại trong 06 (Mistral 7B), 55 (StreamingLLM) và trong các lớp full-attention/hybrid ở 25, 26.

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, **không phải bản dịch** của paper. Các con số trong mục "Thí nghiệm" lấy từ paper; mục "Ứng dụng cho dự án" và các phép tính chi phí đánh dấu "tự tính" là suy luận của chúng ta, không phải kết luận của paper.

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

**Vấn đề.** Self-attention chuẩn (bài 01) tính điểm cho *mọi cặp* vị trí, nên thời gian và bộ nhớ là $O(n^2)$ theo độ dài chuỗi $n$. Mô hình BERT/RoBERTa vì vậy bị giới hạn ở 512 token. Với tài liệu dài (bài báo khoa học, nhiều đoạn Wikipedia), người ta phải làm các việc tạm bợ:

- **Cắt cụt** tài liệu (mất thông tin).
- **Chia khúc 512 token** rồi ghép kết quả bằng một mô hình riêng cho từng tác vụ (mất tương tác giữa các khúc).
- **Hai giai đoạn** (retrieve rồi đọc) — lỗi giai đoạn 1 lan sang giai đoạn 2.

**Các hướng trước đó** (Bảng 1 trong paper): (i) mô hình đi từ trái sang phải theo từng khúc (Transformer-XL, Compressive Transformer...) — tốt cho language modeling nhưng không dùng được bối cảnh hai chiều cho transfer learning; (ii) định nghĩa một **mẫu attention thưa** (Sparse Transformer, Reformer, BP-Transformer, Blockwise...). Nhiều công trình chỉ đánh giá language modeling, chưa kiểm tra theo kiểu "tiền huấn luyện rồi fine-tune" như BERT.

**Khái niệm cần biết trước.**

- *Attention pattern* (mẫu attention): tập các cặp $(i,j)$ mà token $i$ được phép nhìn token $j$. Attention đầy đủ là toàn bộ ma trận $n\times n$; attention thưa chỉ tính một phần.
- *Receptive field* (trường nhìn): tập token ban đầu có thể ảnh hưởng đến một biểu diễn ở lớp cao, giống CNN: xếp nhiều lớp cửa sổ nhỏ thì lớp trên "thấy" xa.
- *BPC (bits per character)*: loss trung bình theo bit trên mỗi ký tự, càng thấp càng tốt (dùng cho language modeling cấp ký tự).

## 2. Ý tưởng chính

Thay vì để mỗi token nhìn *tất cả* token khác, Longformer cho mỗi token chỉ nhìn **một cửa sổ cố định xung quanh nó** (sliding window, kích thước $w$), nên chi phí là $O(n\times w)$, tuyến tính theo $n$. Xếp $\ell$ lớp thì trường nhìn ở lớp trên là $\ell\times w$, nhờ vậy thông tin vẫn lan đi xa. Cửa sổ không đủ cho các tác vụ cần "tổng hợp toàn chuỗi" (phân loại qua `[CLS]`, hỏi đáp so câu hỏi với tài liệu), nên thêm **global attention** trên *một vài token chọn trước*: token global nhìn tất cả và tất cả nhìn nó (đối xứng). Vì số token global nhỏ và không phụ thuộc $n$, tổng chi phí vẫn $O(n)$. Tác giả hiện thực kernel CUDA, tiếp tục tiền huấn luyện từ RoBERTa với độ dài 4,096, và cho thấy tốt hơn RoBERTa trên các tác vụ tài liệu dài; cuối cùng thêm biến thể encoder-decoder (LED) cho tóm tắt.

## 3. Phương pháp chi tiết

### 3.1 Ba thành phần của mẫu attention

**(a) Sliding window.** Với cửa sổ $w$ (chẵn), token $i$ nhìn các token $j$ thỏa $|i-j|\le w/2$ (paper: "$\frac12 w$ tokens on each side").

$$
\mathcal{N}(i)=\{\,j:\ |i-j|\le w/2\,\},\qquad
\text{số cặp}\approx n\cdot (w+1)=O(n\,w).
$$

Với $\ell$ lớp và $w$ cố định, trường nhìn ở lớp trên cùng là $\ell\times w$. Có thể dùng $w$ khác nhau theo từng lớp: nhỏ ở lớp thấp, lớn ở lớp cao.

**(b) Dilated sliding window.** Cửa sổ có "khe" cách nhau $d$ (giống dilated convolution), cùng chi phí nhưng trường nhìn lớn hơn: $\ell\times d\times w$. Paper dùng dilation theo từng head: một số head không dilate (tập trung bối cảnh gần), số khác dilate (nhìn xa).

**(c) Global attention.** Chọn tập vị trí $G$ (tùy tác vụ: `[CLS]` cho phân loại, toàn bộ token câu hỏi cho QA). Với $g\in G$: $g$ nhìn mọi token; mọi token nhìn $g$. Do $|G|\ll n$ và độc lập với $n$, độ phức tạp tổng hợp vẫn $O(n)$.

Hai bộ phép chiếu riêng:

$$
\text{cửa sổ: } Q_s,K_s,V_s;\qquad \text{global: } Q_g,K_g,V_g,
$$

$Q_g,K_g,V_g$ được **khởi tạo bằng đúng giá trị** của $Q_s,K_s,V_s$ rồi học riêng. Công thức attention vẫn là softmax chuẩn nhưng chỉ lấy trên các cặp được phép:

$$
\text{Attention}(Q,K,V)=\operatorname{softmax}\!\Big(\tfrac{QK^\top}{\sqrt{d_k}}\Big)V
\;\;\to\;\;
o_i=\sum_{j\in\mathcal N(i)}\frac{\exp(q_i^\top k_j/\sqrt{d_k})}{\sum_{j'\in\mathcal N(i)}\exp(q_i^\top k_{j'}/\sqrt{d_k})}\,v_j .
$$

Chú ý: **softmax vẫn có mặt**, chỉ chuẩn hóa trên tập hàng xóm. Đây là khác biệt cốt lõi so với bài 18: ở đây ta bỏ bớt cặp nhưng không bỏ softmax, nên *không* có dạng hồi quy trạng thái hữu hạn (muốn sinh token tự hồi quy vẫn phải giữ KV của $w$ token gần nhất).

### 3.2 Hiện thực: vì sao cần kernel riêng

Phép khó là $QK^\top$. Longformer chỉ cần một số đường chéo của ma trận này ("banded matrix multiplication"), mà PyTorch/TensorFlow không hỗ trợ trực tiếp. Paper so ba cách (Hình 1):

| Cách | Mô tả | Ghi chú |
|---|---|---|
| Longformer-loop | tính từng đường chéo trong vòng lặp | tiết kiệm bộ nhớ nhưng "unusably slow", chỉ để test |
| Longformer-chunks | cắt Q, K thành khối kích thước $w$ chồng lấn $w/2$, nhân khối bằng matmul, mask các phần thừa | chỉ hỗ trợ không dilation; nhanh nhất cho pretrain/finetune; tốn gấp ~2× bộ nhớ lý tưởng vì tính cả vài giá trị 0 |
| Longformer-cuda | kernel CUDA viết bằng TVM | hỗ trợ dilation, tiết kiệm bộ nhớ nhất, "as fast as highly optimized full self-attention"; dùng cho language modeling cấp ký tự |

Ghi chú của paper: attention được giữ ở fp32 vì fp16 gây tràn số/NaN ở giai đoạn sau của huấn luyện.

Pseudo-code của cách "chunks" (diễn giải, không phải code gốc):

```
def sliding_window_attn(Q, K, V, w):       # Q,K,V: [n, d]
    # 1. chia Q và K thành các khối kích thước w, chồng lấn w/2 (K, V lấy cùng cách chia)
    Qb = blocks(Q, size=w, stride=w//2)    # [số_khối, w, d]
    Kb = blocks(K, size=w, stride=w//2)
    Vb = blocks(V, size=w, stride=w//2)
    S  = Qb @ Kb.transpose(-1,-2)          # một lần matmul theo khối (mỗi khối chỉ ghép với khối K tương ứng/lân cận)
    S  = mask_outside_window(S, w)         # bỏ các ô ngoài |i-j| <= w/2 (phần tính thừa ~2x bộ nhớ)
    P  = softmax(S, dim=-1)
    return P @ Vb                          # ghép lại các khối thành [n, d]
```

### 3.3 Language modeling cấp ký tự (autoregressive)

- Dùng cửa sổ *tăng dần theo lớp* (nhỏ ở lớp thấp, lớn ở lớp cao), không dilate ở lớp thấp, và chỉ dilate nhẹ trên 2 head ở lớp cao.
- **Huấn luyện theo giai đoạn**: mô hình cần nhiều bước cập nhật để học bối cảnh cục bộ trước khi dùng được bối cảnh dài, nên bắt đầu với chuỗi và cửa sổ ngắn; mỗi giai đoạn sau **gấp đôi cửa sổ và độ dài chuỗi, giảm nửa learning rate**. 5 giai đoạn, độ dài chuỗi từ 2,048 đến 23,040.
- Đánh giá với chuỗi 32,256, bước nhảy 512, báo cáo trên 512 token cuối.

### 3.4 Tiền huấn luyện và fine-tune từ RoBERTa

- Thay self-attention của RoBERTa bằng cửa sổ trượt $w=512$ (**cùng lượng tính toán như RoBERTa**) cộng global attention theo tác vụ. Kiến trúc còn lại không đổi.
- RoBERTa dùng position embedding tuyệt đối học được, tối đa 512. Để lên 4,096, **sao chép** 512 embedding nhiều lần làm khởi tạo. Lý do: phân tích attention của BERT cho thấy thiên hướng mạnh nhìn bối cảnh gần (token trước/sau), sao chép giữ cấu trúc cục bộ ở mọi nơi trừ ranh giới các đoạn sao chép.
- Tiếp tục MLM 65K bước, độ dài 4,096, batch 64 ($2^{18}$ token), LR tối đa $3\times10^{-5}$, warmup 500 bước, sau đó giảm đa thức bậc 3.

### 3.5 LED (Longformer-Encoder-Decoder)

Encoder dùng mẫu local+global của Longformer; decoder dùng full attention lên toàn bộ encoder output và các token đã sinh. Khởi tạo từ BART (giữ nguyên số lớp, kích thước ẩn), mở rộng position embedding lên 16K bằng cách sao chép embedding 1K của BART 16 lần. Hai cỡ: LED-base (6+6 lớp), LED-large (12+12 lớp). Với tóm tắt arXiv: cửa sổ encoder 1,024, global attention trên token `<s>` đầu.

## 4. Thí nghiệm và kết quả chính

### 4.1 Language modeling cấp ký tự (BPC, càng thấp càng tốt)

Mô hình nhỏ: 12 lớp, hidden 512 (41M tham số). Mô hình lớn: 30 lớp, hidden 512 (102M).

| Mô hình nhỏ | #Param | text8 test | enwik8 test |
|---|---|---|---|
| T12 (Al-Rfou) | 44M | 1.18 | 1.11 |
| Adaptive span | 38-39M | 1.11 | 1.02 |
| BP-Transformer | 38-39M | 1.11 | 1.02 |
| **Longformer** | 41M | **1.10** | **1.00** |

Mô hình lớn, enwik8 test BPC: Transformer-XL (18 lớp, 88M) 1.03; Sparse Transformer (~100M) 0.99; Transformer-XL (24 lớp, 277M) 0.99; Adaptive 209M 0.98; Compressive 277M 0.97; Routing ~223M 0.99; **Longformer 102M: 0.99**. Tức khớp mô hình tốt cùng cỡ, kém nhẹ mô hình có hơn 2× tham số.

### 4.2 Tiền huấn luyện MLM (BPC trên tập dev của kho tài liệu dài)

| Cấu hình | base | large |
|---|---|---|
| RoBERTa (seqlen 512) | 1.846 | 1.496 |
| Longformer (4,096), position embedding khởi tạo ngẫu nhiên | 10.299 | 8.738 |
| + sao chép position embedding | 1.957 | 1.597 |
| + 2K bước cập nhật | 1.753 | 1.414 |
| + 65K bước cập nhật | 1.705 | 1.358 |
| Longformer, chỉ train position embedding mới (đóng băng RoBERTa) | 1.850 | 1.504 |

### 4.3 Fine-tune trên tài liệu dài (tập dev, Longformer-base vs RoBERTa-base)

| Tác vụ | RoBERTa-base | Longformer-base |
|---|---|---|
| WikiHop (acc) | 72.4 | 75.0 |
| TriviaQA (F1) | 74.3 | 75.2 |
| HotpotQA (joint F1) | 63.5 | 64.4 |
| OntoNotes (F1) | 78.4 | 78.6 |
| IMDB (acc) | 95.3 | 95.7 |
| Hyperpartisan (F1) | 87.4 | 94.8 |

Longformer-large trên bảng xếp hạng lúc nộp (5/2020): WikiHop 81.9 (SOTA trước đó 78.3), TriviaQA 77.3 (73.3), HotpotQA 73.2 (SOTA 74.2). Paper nhận định tăng nhiều nhất ở tác vụ thật sự cần bối cảnh xa (WikiHop, Hyperpartisan), ít ở tác vụ mà bối cảnh cục bộ đã đủ.

### 4.4 LED trên tóm tắt arXiv

| Mô hình | R-1 | R-2 | R-L |
|---|---|---|---|
| Pegasus | 44.21 | 16.95 | 38.83 |
| LED-large (seqlen 4,096) | 44.40 | 17.94 | 39.76 |
| BigBird (seqlen 4,096) | 46.63 | 19.02 | 41.77 |
| LED-large (seqlen 16,384) | 46.63 | 19.62 | 41.83 |

Hình 3 của paper: tăng độ dài đầu vào từ 1K lên 16K cải thiện rõ ROUGE (độ chính xác từng điểm trong hình không đọc được từ text đã trích, nên không ghi ở đây).

## 5. Phân tích / ablation

**Cấu hình cửa sổ (text8, dev BPC, 150K bước ở giai đoạn 1):** giảm $w$ từ 512 xuống 32 theo chiều lên lớp: 1.24; cố định $w=230$: 1.23; **tăng $w$ từ 32 lên 512**: 1.21. Không dilation: 1.21; dilation trên 2 head: 1.20. Kết luận: lớp thấp cửa sổ nhỏ, lớp cao cửa sổ lớn là hợp lý; dilation có ích nhẹ.

**Ablation WikiHop (dev acc, Longformer-base):** bản chuẩn 73.8 (5 epoch).

| Biến thể | Acc | Δ |
|---|---|---|
| RoBERTa-base seqlen 512 | 72.4 | -1.4 |
| 15 epoch | 75.0 | +1.2 |
| Longformer seqlen 512, attention đầy đủ $n^2$ | 71.7 | -2.1 |
| seqlen 2,048 | 73.1 | -0.7 |
| không MLM pretraining | 73.2 | -0.6 |
| không projection riêng cho global | 72.2 | -1.6 |
| không projection riêng *và* không global attention | 65.5 | -8.3 |
| chỉ pretrain position embedding mới | 73.5 | -0.3 |

Rút ra (theo paper): **global attention và projection riêng cho global là rất quan trọng** (mất 8.3 điểm nếu bỏ cả hai); độ dài dài hơn giúp; khi cấu hình về 512 + attention đầy đủ thì Longformer kém RoBERTa nhẹ, xác nhận cải thiện không đến từ việc tiền huấn luyện thêm.

## 6. Giới hạn

**Do tác giả nêu:**
- Global attention là *thủ công theo tác vụ* (phải chọn token nào là global).
- Dilation trên vài head làm giảm hiệu năng khi nạp trọng số RoBERTa (không tương thích với trọng số tiền huấn luyện); có thể cần huấn luyện lại từ đầu.
- Cách "chunks" tốn gấp ~2× bộ nhớ so với lý tưởng; kernel CUDA tối ưu hoàn hảo (nhanh hơn $n^2$) cần kiến thức lập trình GPU cấp thấp, chưa đạt đầy đủ.
- So sánh với BigBird bị nhiễu do BigBird dùng nhiều tính toán tiền huấn luyện hơn 16×.

**Do chúng ta nhận xét:**
- Cửa sổ trượt *vẫn giữ KV cache* cỡ $w$ token ở thời điểm sinh; bộ nhớ bị chặn trên (tốt) nhưng không "nén" như trạng thái hồi quy ở bài 18-21.
- Thông tin ngoài cửa sổ chỉ đến được qua nhiều lớp (trường nhìn $\ell w$) hoặc qua token global; với sinh văn bản tự hồi quy không có khái niệm token global "nhìn tương lai".
- Phần lớn lợi ích đo bằng mô hình encoder (MLM) và tác vụ hiểu, không phải LLM sinh tự hồi quy quy mô lớn.
- Với $n$ cỡ vài nghìn như trong dự án, tiết kiệm FLOPs của $n\times w$ so với $n^2$ khá khiêm tốn (xem mục 8).

## 7. Liên hệ với các bài khác

- **01 Attention Is All You Need:** Longformer giữ nguyên công thức attention, chỉ thay *tập cặp được tính*.
- **06 Mistral 7B:** dùng sliding-window attention ở quy mô LLM, kèm KV cache dạng vòng (rolling buffer) cỡ $w$; đây là hậu duệ trực tiếp của ý tưởng cửa sổ.
- **12 RoPE, 13 ALiBi:** vấn đề mở rộng vị trí; Longformer chọn "sao chép position embedding" thay vì mã hóa tương đối.
- **18 Linear Transformers:** hướng đối lập: *giữ mọi cặp* nhưng thay softmax để tính được bằng hồi quy. Hai hướng bổ sung nhau: SWA cho bối cảnh gần chính xác, linear/SSM cho bối cảnh xa nén.
- **25 Jamba, 26 Kimi Linear:** hybrid trộn lớp attention (kể cả cửa sổ) với lớp hồi quy.
- **44-45 FlashAttention:** giải quyết bài toán tương tự (băng thông bộ nhớ) cho attention đầy đủ mà không cắt cặp nào.
- **55 StreamingLLM, 56 H2O:** cắt KV cache; StreamingLLM cho thấy cửa sổ thuần túy cần thêm "attention sink".
- **57 FastV:** cắt token thị giác — cùng tinh thần giảm số cặp attention.

## 8. Ứng dụng cho dự án (suy luận)

*Ngữ cảnh:* LightOnOCR-3-0.8B: ViT 12 khối + decoder lai (GatedDeltaNet + full attention), ~1,500 token đầu ra/trang, batch 1, GPU ≤ 8 GB.

**Chi phí tính tay (tự tính).** Giả định $d_{model}=1024$, 24 lớp decoder, bf16, mọi lớp là MHA với $d_{kv}=1024$ (cận trên, vì mô hình thật có thể dùng GQA/ít head KV hơn), $T=1500$.

| Đại lượng | Công thức | Giá trị |
|---|---|---|
| KV cache full attention ở $T=1500$ | $2\cdot L\cdot d_{kv}\cdot T\cdot 2\,\text{B}$ | $2\cdot24\cdot1024\cdot1500\cdot2\approx147\ \text{MB}$ |
| KV cache nếu mọi lớp dùng cửa sổ $w=512$ | thay $T\to w$ | $\approx50\ \text{MB}$ (cố định khi $T>512$) |
| Số cặp attention (mỗi head, mỗi lớp) | $n^2$ vs $n\cdot w$ | $2.25\times10^6$ vs $0.77\times10^6$ (khoảng 34%) |
| FLOPs QK$^\top$ + AV, một lớp, không mask | $4\,T^2 d$ | $\approx 9.2$ GFLOPs |
| FLOPs phép chiếu QKVO, một lớp | $8\,T d^2$ | $\approx 12.6$ GFLOPs |

Nhận xét: ở $T\approx1.5$K, phần *bậc hai* của attention (9.2 GFLOPs/lớp) còn **nhỏ hơn** phần phép chiếu tuyến tính (12.6 GFLOPs/lớp), chưa kể MLP. Vì vậy cửa sổ trượt không đem lại cải thiện FLOPs đột phá ở độ dài này; lợi ích chính là **chặn trên KV cache** và băng thông đọc mỗi token ở decode. Theo giả định băng thông 300 GB/s (chỉ là giả định cho GPU nhỏ), đọc 147 MB KV ≈ 0.49 ms/token, trong khi đọc trọng số 0.8B bf16 (~1.6 GB) ≈ 5.3 ms/token. Tức KV chỉ khoảng 9% lưu lượng ở mô hình 0.8B, nhưng khoảng 25% nếu mô hình thu còn 0.3B (0.6 GB trọng số, ~2 ms), nên **cắt trọng số trước, KV sau** là thứ tự ưu tiên hợp lý; điều này cũng cho thấy các lớp full-attention trong hybrid không phải là điểm nghẽn lớn ở $T=1.5$K nếu GQA đã nhỏ.

**ViT / token thị giác.** Ý tưởng "cửa sổ cục bộ + vài token global" có thể áp cho ViT xử lý trang ảnh độ phân giải cao (số patch lớn): attention cục bộ để nhận dạng nét chữ, vài token global (giống `[CLS]`/register) để mang bố cục toàn trang. Đây là suy đoán; hiện chưa biết ViT 12 khối của LightOnOCR dùng attention đầy đủ hay cửa sổ, cần kiểm tra cấu hình.

**Pruning/distillation.** Longformer cho thấy *một mẫu attention khác với lúc huấn luyện* có thể được "ghép" vào mô hình đã tiền huấn luyện và phục hồi bằng tiếp tục huấn luyện (65K bước, khởi tạo bằng sao chép vị trí). Kinh nghiệm tương tự: thay một số lớp full-attention của decoder bằng cửa sổ trượt rồi chưng cất từ teacher 4B để phục hồi. Chưa có bằng chứng từ paper cho decoder tự hồi quy, nên là giả thuyết.

## 9. Thí nghiệm gợi ý

1. **Đo độ nhạy cửa sổ:** trên decoder hiện có, chuyển các lớp full-attention sang cửa sổ $w\in\{128,256,512\}$ ở *inference-only* (chỉ mask), đo CER/WER trên tập OCR dev. Nếu nhạy thấp, đó là ứng viên để cắt KV mà không cần train lại nhiều.
2. **Profile ở batch 1:** đo thực tế thời gian/token khi $T=1500$ với và không có cắt KV, để kiểm chứng ước lượng "KV ≈ 9% / 25% lưu lượng" ở trên.
3. **Attention pattern của ViT:** vẽ bản đồ attention theo khoảng cách (token-token) ở từng khối ViT để xem bao nhiêu khối thực sự dùng bối cảnh toàn cục; chỉ khi phần lớn mass nằm gần thì mới đáng thử cửa sổ.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| Sliding window | $j\in[i-w/2,\,i+w/2]$; chi phí $O(nw)$; trường nhìn $\ell w$ |
| Dilated window | khe $d$; trường nhìn $\ell d w$ |
| Global attention | token $g\in G$ nhìn tất cả và được tất cả nhìn; $O(n)$ vì $|G|\ll n$ |
| Hai bộ chiếu | $(Q_s,K_s,V_s)$ cho cửa sổ, $(Q_g,K_g,V_g)$ cho global; khởi tạo giống nhau |
| Khởi tạo vị trí | sao chép 512 embedding RoBERTa thành 4,096 (BART: 1K thành 16K) |
| Huấn luyện giai đoạn | gấp đôi $w$ và độ dài, giảm nửa LR mỗi giai đoạn |
| LED | encoder local+global, decoder full attention, khởi tạo từ BART |

## 11. Kết luận ngắn

Longformer là bản "cắt cặp attention" hiệu quả và dễ thay thế (drop-in) cho self-attention, chứng minh được trên cả language modeling và các tác vụ tài liệu dài nhờ ba ý: cửa sổ cục bộ, global attention theo tác vụ, và khởi tạo vị trí bằng sao chép. Đối với lộ trình của bạn, giá trị chính là làm *mốc đối chiếu*: sparse (giữ softmax, giữ KV cỡ $w$) khác căn bản với hướng hồi quy (18-24) vốn nén quá khứ vào trạng thái cố định.

**Nguồn gốc:** https://arxiv.org/abs/2004.05150
