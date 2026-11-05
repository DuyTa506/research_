# Flamingo: a Visual Language Model for Few-Shot Learning (2022)

**Tác giả:** Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech và cộng sự (DeepMind; tổng cộng 27 tác giả, Karen Simonyan là đồng tác giả cấp cao).  
**Venue / arXiv:** arXiv:2204.14198 (v2, 15/11/2022); NeurIPS 2022 (paper ghi "36th Conference on Neural Information Processing Systems").  
**Nguồn:** https://arxiv.org/abs/2204.14198  
**Chủ đề:** Ghép một vision encoder đóng băng với một LLM đóng băng; Perceiver Resampler; gated cross-attention (GATED XATTN-DENSE); dữ liệu xen kẽ ảnh–văn bản; few-shot in-context learning cho ảnh/video.  
**Vị trí trong lộ trình:** Đọc trước: 01 (Transformer, cross-attention), 02 (GPT-3, in-context learning), 04 (Chinchilla, LM nền của Flamingo), 29 (ViT), 30 (CLIP). Bài này là "bản thiết kế gốc" của họ VLM: nó đặt ra câu hỏi *làm sao nối vision và LLM*. Tiếp theo: 33 (BLIP-2 dùng Q-Former thay Resampler), 34–35 (LLaVA bỏ hẳn cross-attention, chỉ dùng projector), 36 (InternVL 1.5).

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải chứ không phải bản dịch paper. Paper rất dài (hơn 29k từ, phần lớn là phụ lục); ghi chú này **tập trung vào kiến trúc** (Perceiver Resampler, gated cross-attention, masking cho ảnh xen kẽ) và các kết quả/ablation thông tin nhất. Phần "Ứng dụng cho dự án" là suy luận của chúng ta, không phải kết luận của paper.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
   - 3.1 Tổng quan luồng dữ liệu
   - 3.2 Vision encoder
   - 3.3 Perceiver Resampler
   - 3.4 GATED XATTN-DENSE
   - 3.5 Masking cho ảnh xen kẽ
   - 3.6 Dữ liệu và hàm mục tiêu
   - 3.7 Few-shot prompting
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

**Vấn đề trước Flamingo.** Năm 2022, thị giác máy tính vẫn theo kiểu "pretrain rồi fine-tune cho từng task", cần hàng nghìn mẫu có nhãn và tinh chỉnh siêu tham số riêng cho từng task. Trong khi đó, GPT-3 (bài 02) cho thấy LLM chỉ cần vài ví dụ trong prompt là làm được task mới (**in-context few-shot learning**: đưa vài cặp vào/ra vào prompt, mô hình tự suy ra quy tắc mà không cập nhật trọng số).

Hai hướng vision-language có sẵn đều thiếu:

- **Mô hình đối lập (contrastive) như CLIP/ALIGN** (bài 30): cho điểm tương đồng ảnh–văn bản, làm được zero-shot phân loại nhưng **không sinh được văn bản**, nên không làm được captioning hay VQA mở.
- **Mô hình sinh có điều kiện ảnh**: chưa cho thấy chất lượng tốt trong chế độ ít dữ liệu.

**Khó khăn kỹ thuật:** muốn tận dụng một LM mạnh đã pretrain, ta phải "tiêm" thông tin ảnh vào mà không phá kiến thức sẵn có (catastrophic forgetting: học task mới làm mô hình quên thứ đã học). Thêm nữa, ảnh độ phân giải cao hoặc video cho ra hàng nghìn đặc trưng, quá nhiều để đưa thẳng vào attention của LM.

**Khái niệm nền cần nhớ:**

- *Cross-attention*: attention mà query đến từ một chuỗi (ở đây là văn bản) còn key/value đến từ chuỗi khác (ở đây là đặc trưng ảnh).
- *Frozen (đóng băng)*: trọng số không được cập nhật khi huấn luyện, chỉ chạy forward/backward qua nó (gradient vẫn đi qua nếu cần, nhưng không có bước cập nhật).
- *Interleaved data*: văn bản xen kẽ ảnh, như một trang web thực (chữ, ảnh, chữ, ảnh...).

## 2. Ý tưởng chính

Flamingo lấy **hai mô hình mạnh đã pretrain và đóng băng**: một vision encoder (NFNet-F6) và một LM (Chinchilla 1.4B/7B/70B). Giữa chúng, paper chèn hai loại module huấn luyện từ đầu:

1. **Perceiver Resampler**: nén một lưới đặc trưng thị giác có kích thước tuỳ ý (ảnh to, nhiều khung hình video) thành **đúng 64 "visual token"** bằng một Transformer nhỏ với 64 query học được.
2. **GATED XATTN-DENSE**: các lớp cross-attention + FFN chèn *giữa* các lớp LM đóng băng, để văn bản "nhìn" vào 64 visual token đó. Mỗi lớp được nhân với $\tanh(\alpha)$ với $\alpha$ khởi tạo bằng 0, nên lúc đầu mô hình y hệt LM gốc, rồi dần dần "mở cổng" cho thông tin thị giác.

Huấn luyện bằng một hỗn hợp dữ liệu web (đặc biệt là trang web xen kẽ ảnh–chữ, M3W) với loss next-token thông thường. Kết quả là một mô hình có khả năng few-shot như GPT-3 nhưng cho ảnh/video.

## 3. Phương pháp chi tiết

### 3.1 Tổng quan luồng dữ liệu

Flamingo mô hình hoá xác suất của văn bản $y$ khi biết các ảnh/video $x$ xen kẽ (Eq. 1):

$$
p(y\mid x)=\prod_{\ell=1}^{L} p\big(y_\ell \mid y_{<\ell},\, x_{\le \ell}\big)
$$

trong đó $x_{\le\ell}$ là các ảnh/video đứng *trước* token $y_\ell$ trong chuỗi xen kẽ. Đây chính là một LM nhân quả, chỉ thêm điều kiện thị giác.

```
ảnh/video --> [Vision Encoder, đóng băng] --> lưới đặc trưng [T*S, d]
          --> [Perceiver Resampler, huấn luyện] --> 64 visual token
văn bản   --> LM block 1 (đóng băng) --> GATED XATTN-DENSE (hỏi 64 token) --> LM block 2 ... --> logits
```

### 3.2 Vision encoder

- NFNet-F6 (mạng CNN không dùng BatchNorm), **đóng băng**, được pretrain riêng bằng loss contrastive hai chiều kiểu CLIP trên ALIGN + LTIP (ảnh 288x288; trong Flamingo ảnh được resize 320x320).
- Lấy đầu ra tầng cuối: lưới 2D đặc trưng, duỗi thành chuỗi 1D.
- Video: lấy mẫu 1 FPS, mã hoá từng khung độc lập, cộng **embedding thời gian học được**, rồi duỗi thành 1D. Ảnh được coi là video 1 khung.
- Paper không dùng positional embedding không gian rõ ràng, vì CNN đã mã hoá thông tin vị trí ngầm trong kênh (paper nói không thấy cải thiện khi thêm).

### 3.3 Perceiver Resampler

Mục tiêu: biến $X_f\in\mathbb{R}^{T\cdot S\times d}$ (kích thước thay đổi) thành $X\in\mathbb{R}^{R\times d}$ với $R=64$ cố định.

Cách làm (giống Perceiver/DETR): giữ $R$ vector **latent học được** (tham số của mô hình) làm query. Mỗi lớp: latent cross-attend vào đặc trưng ảnh, rồi qua FFN. Điểm khác biệt nhỏ so với Perceiver/DETR: **key/value là nối của đặc trưng ảnh và chính các latent** (paper nói cách này tốt hơn chút).

Pseudo-code (bám theo Figure 5 của paper):

```python
def perceiver_resampler(x_f,              # [T, S, d] đặc trưng thị giác
                        time_embeddings,  # [T, 1, d]
                        x,                # [R, d] latent học được
                        num_layers):
    x_f = x_f + time_embeddings
    x_f = flatten(x_f)                    # [T*S, d]
    for i in range(num_layers):
        x = x + attention_i(q=x, kv=concat([x_f, x]))   # latent hỏi ảnh (và hỏi nhau)
        x = x + ffw_i(x)
    return x                              # [R, d], R = 64
```

Công thức cho một lớp (ký hiệu của chúng ta):

$$
X \leftarrow X + \mathrm{Attn}\big(Q=XW_Q,\; K=[X_f;X]W_K,\; V=[X_f;X]W_V\big),\qquad X\leftarrow X+\mathrm{FFN}(X)
$$

Chi phí attention của một lớp Resampler là $O\big(R\cdot(TS+R)\cdot d\big)$: **tuyến tính** theo số đặc trưng đầu vào $TS$, và số token đầu ra luôn là $R$. Đây là điểm mấu chốt: video dài hay ảnh to chỉ làm tăng chi phí *ở Resampler*, còn LM luôn chỉ thấy 64 token.

Cấu hình (Table 4 của paper): 6 lớp, $D=1536$, 16 head, FFN dùng Squared ReLU; tổng ~194M tham số (Table 5), cố định kích thước cho cả ba model.

### 3.4 GATED XATTN-DENSE

Một khối mới chèn *trước* một lớp LM đóng băng. Pseudo-code từ paper (Figure 4):

```python
def gated_xattn_dense(y, x, alpha_xattn, alpha_dense):
    # y: đặc trưng ngôn ngữ; x: visual token (64 token từ Resampler)
    # alpha_*: vô hướng học được, khởi tạo = 0
    y = y + tanh(alpha_xattn) * attention(q=y, kv=x)   # 1. cross-attention có cổng
    y = y + tanh(alpha_dense) * ffw(y)                 # 2. FFN có cổng
    y = y + frozen_attention(q=y, kv=y)                # lớp LM gốc (đóng băng)
    y = y + frozen_ffw(y)
    return y
```

$$
\boxed{\,y \leftarrow y + \tanh(\alpha)\cdot f_{\text{new}}(y, x)\,},\qquad \alpha_{\text{init}}=0
$$

**Vì sao khởi tạo $\alpha=0$ quan trọng?** Lúc bắt đầu, $\tanh(0)=0$ nên khối mới bị "tắt": đầu ra của mô hình *đúng bằng* LM gốc. Gradient vẫn chảy vào $\alpha$ và vào trọng số khối mới (vì $\tanh'(0)=1$), nên cổng từ từ mở ra khi có lợi. Paper thấy: bỏ tanh gating làm điểm tổng giảm 4.2% và gây bất ổn huấn luyện. Hình 6 của paper cho thấy $|\tanh\alpha|$ ở mọi lớp tăng nhanh từ 0 trong huấn luyện (tức mọi lớp LM đều dùng thông tin thị giác), có xu hướng lớn dần theo độ sâu; paper cảnh báo không nên suy diễn mạnh vì thang đo của activation cũng đổi theo độ sâu.

**Tần suất chèn:** chèn ở mọi lớp thì tốt nhất nhưng đắt. Paper chèn mỗi 1 lớp (3B), mỗi 4 lớp (9B), mỗi 7 lớp (80B).

Cấu hình (Table 4, 5):

| Model | LM đóng băng | Vision (đóng băng) | GATED XATTN-DENSE (huấn luyện) | Resampler | Tổng |
|---|---|---|---|---|---|
| Flamingo-3B | 1.4B | 435M | 1.2B (mỗi lớp; 24 khối, $D=2048$) | 194M | 3.2B |
| Flamingo-9B | 7.1B | 435M | 1.6B (mỗi 4 lớp; 10 khối, $D=4096$) | 194M | 9.3B |
| Flamingo-80B | 70B | 435M | 10B (mỗi 7 lớp; 12 khối, $D=8192$) | 194M | 80B |

*Kiểm tra nhanh (tự tính):* với $D=2048$, một khối gồm cross-attention ($\approx4D^2\approx16.8$M) + FFN 4D ($\approx8D^2\approx33.5$M) $\approx 50$M tham số; nhân 24 khối $\approx1.2$B, khớp với Table 5. Nghĩa là phần tham số huấn luyện của Flamingo-3B *lớn hơn* phần LM đóng băng chút ít: chi phí của cách "chèn cross-attention vào mọi lớp" khá cao.

### 3.5 Masking cho ảnh xen kẽ

Với chuỗi có nhiều ảnh, Flamingo định nghĩa hàm $\varphi(\ell)$ = chỉ số của ảnh/video **gần nhất đứng trước** token $\ell$ (0 nếu chưa có ảnh). Tại token $\ell$, cross-attention chỉ được nhìn visual token của ảnh $\varphi(\ell)$ (qua mask trên ma trận cross-attention chữ→ảnh), chứ không nhìn mọi ảnh trước đó.

Thông tin từ các ảnh cũ hơn vẫn đến được qua self-attention của LM (vì token ở ảnh trước đã "thấm" thông tin ảnh vào residual stream). Lợi ích: huấn luyện chỉ với tối đa 5 ảnh/chuỗi (M3W) mà lúc suy luận vẫn dùng được tới 32 cặp (ảnh, chữ) làm "shot". Paper khẳng định (Section 2.3) cách này hiệu quả hơn việc cho phép nhìn mọi ảnh trước, nhưng bảng ablation chính (Table 3) không có dòng riêng cho điểm này.

Còn có token đặc biệt: `<image>` đánh dấu vị trí ảnh, `<EOC>` (end of chunk) đặt trước mỗi ảnh và cuối tài liệu, giúp mô hình biết khi nào kết thúc một đoạn (dùng để dừng sinh khi đánh giá).

### 3.6 Dữ liệu và hàm mục tiêu

Ba loại dữ liệu, đều từ web:

| Nguồn | Mô tả |
|---|---|
| **M3W** (MultiModal MassiveWeb) | ~43 triệu trang web; chèn `<image>` đúng vị trí; lấy ngẫu nhiên đoạn $L=256$ token, tối đa $N=5$ ảnh đầu |
| **ALIGN** + **LTIP** | 1.8 tỷ cặp ảnh–alt-text (ALIGN); LTIP 312 triệu cặp ảnh–caption dài và chất lượng hơn |
| **VTP** | 27 triệu video ngắn (~22 giây) kèm câu mô tả |

Loss là tổng có trọng số của negative log-likelihood theo từng tập (Eq. 2):

$$
\sum_{m=1}^{M}\lambda_m\,\mathbb{E}_{(x,y)\sim\mathcal{D}_m}\Big[-\sum_{\ell=1}^{L}\log p(y_\ell\mid y_{<\ell},x_{\le\ell})\Big]
$$

Paper nhấn mạnh: chỉnh $\lambda_m$ là then chốt; **tích luỹ gradient qua tất cả dataset** tốt hơn xoay vòng (round-robin).

### 3.7 Few-shot prompting

Prompt = các cặp ví dụ (ảnh hoặc video, văn bản) nối tiếp nhau rồi đến ảnh truy vấn, ví dụ: `<BOS><image>Output: A cat wearing sunglasses.<EOC><image>Output: Elephants walking in the savanna.<EOC><image>Output:`. Với VQA: `Question: ... Answer: ...`. Đánh giá:

- **Open-ended**: beam search (beam=3), dừng ở `<EOC>`.
- **Close-ended**: chấm log-likelihood cho từng đáp án rồi xếp hạng.
- **Zero-shot**: dùng 2 ví dụ *chỉ có chữ* (bỏ ảnh); chỉ 1 ví dụ thì mô hình bị lệch bắt chước câu đó.

## 4. Thí nghiệm và kết quả chính

16 benchmark ảnh/video (5 benchmark DEV dùng để chọn thiết kế: COCO, OKVQA, VQAv2, MSVDQA, VATEX; 11 còn lại để đo không thiên lệch).

**Few-shot, Flamingo-80B, chọn các cột đọc được rõ trong Table 1** (một số cột bị lỗi trích xuất PDF nên bỏ qua; mọi số copy từ paper):

| Benchmark | 0-shot | 4-shot | 32-shot | Pretrained FT SOTA (số mẫu fine-tune) |
|---|---|---|---|---|
| OKVQA | 50.6 | 57.4 | 57.8 | 54.4 (10K) |
| VQAv2 | 56.3 | 63.1 | 67.6 | 80.2 (444K) |
| COCO (CIDEr) | 84.3 | 103.2 | 113.8 | 143.3 (500K) |
| MSVDQA | 35.6 | 41.7 | 52.3 | 47.9 (27K) |
| VATEX (CIDEr) | 46.7 | 56.0 | 65.1 | 76.3 (500K) |
| VizWiz | 31.6 | 39.6 | 49.8 | 57.2 (20K) |
| Flick30K | 67.2 | 75.1 | 75.4 | 67.4 (30K) |
| MSRVTTQA | 17.4 | 23.9 | 31.0 | 46.8 (130K) |
| YouCook2 | 60.1 | 74.5 | 86.8 | 138.7 (10K) |

(Cột bị bỏ vì khó đọc: TextVQA, VisDial, HatefulMemes, NextQA, STAR, iVQA, RareAct.)

Thông điệp: với 32 shot và không cập nhật trọng số, Flamingo vượt SOTA fine-tune trên 6 trong 16 task (phần giới thiệu và Figure 2 của paper; chú thích Table 1 lại nói "seven tasks", nên con số này hơi lệch giữa các chỗ trong paper), dùng khoảng 1000 lần ít dữ liệu theo từng task. Model lớn hơn và nhiều shot hơn đều tốt hơn (Figure 2).

**Fine-tune (Table 2, mở khoá vision backbone, độ phân giải cao hơn):** SOTA mới trên 5 task: VQAv2 82.0 (test-dev), VATEX 84.2, VizWiz 65.7, MSRVTTQA 47.4, HatefulMemes 86.6. So 32-shot VQAv2 67.6 -> fine-tune 82.0.

**Cỡ mô hình và số shot:** Flamingo-3B/9B/80B tăng đều theo kích thước; model lớn tận dụng nhiều shot tốt hơn. Dù huấn luyện chỉ với 5 ảnh/chuỗi, suy luận với 32 shot vẫn lợi.

## 5. Phân tích / ablation

Ablation trên Flamingo-3B, 4-shot, 5 benchmark DEV (Table 3 của paper). Baseline: điểm tổng **70.7**, 3.2B tham số, step time 1.74 s.

| Thay đổi | Điểm tổng | Nhận xét (theo paper) |
|---|---|---|
| Bỏ cặp video–văn bản | 67.3 | Hại các task video |
| Bỏ cặp ảnh–văn bản | 60.9 | Cặp ảnh–text rất quan trọng |
| Cặp ảnh–text -> LAION | 66.4 | Kém hơn một chút so với ALIGN+LTIP |
| **Bỏ M3W** (dữ liệu xen kẽ) | **53.4** | Giảm hơn 17%; dữ liệu xen kẽ là nền tảng của few-shot |
| Round-robin thay vì tích luỹ gradient | 62.9 | Tích luỹ gradient tốt hơn |
| **Bỏ tanh gating** | 66.5 | Giảm 4.2% và mất ổn định |
| Vanilla xattn (không dense) | 66.9 (2.4B) | GATED XATTN-DENSE tốt hơn |
| Grafting (kiểu của [68]) | 63.1 | LM đóng băng + stack học mới phía trên: kém |
| Chỉ 1 xattn ở giữa | 59.8 (2.0B, 0.87s) | Quá ít chỗ tiêm |
| Xattn mỗi 4 lớp | 68.8 (1.02s) | Nhanh hơn ~66%, mất 1.9% |
| Xattn mỗi 2 lớp | 68.2 | |
| Resampler -> MLP | 66.6 | Perceiver Resampler tốt hơn và nhanh hơn |
| Resampler -> Transformer thường | 66.7 | |
| Vision: CLIP ViT-L/14 | 64.9 | NFNet-F6 hơn 5.8% |
| Vision: NFNet-F0 | 62.7 | NFNet-F6 hơn 8.0% |
| LM không đóng băng, train từ đầu | 57.8 | Giảm 12.9% |
| LM không đóng băng, khởi tạo từ pretrain | 62.7 | Giảm 8.0%: catastrophic forgetting |

Rút ra: (i) **dữ liệu xen kẽ** quan trọng hơn mọi thiết kế kiến trúc riêng lẻ; (ii) **đóng băng LM** là lựa chọn tốt hơn là fine-tune toàn bộ LM trên dữ liệu ảnh–văn bản; (iii) **vision encoder mạnh** đáng kể; (iv) tần suất chèn cross-attention là đánh đổi chi phí/chất lượng rõ ràng (mỗi 4 lớp: nhanh hơn ~66% mà mất 1.9%).

## 6. Giới hạn

**Do tác giả nêu:**

- Kế thừa điểm yếu của LM đóng băng: ảo giác (hallucination), đoán không có căn cứ; LM tổng quát hoá kém với chuỗi dài hơn lúc huấn luyện; hiệu quả mẫu thấp trong huấn luyện.
- Hiệu năng phân loại ảnh kém các mô hình contrastive SOTA (đoạn thảo luận ở phần 5 của paper).

**Do chúng tôi nhận xét:**

- Số tham số huấn luyện và chi phí huấn luyện lớn (80B có 10B tham số xattn; dữ liệu M3W/LTIP/VTP không công khai) nên khó tái lập.
- Mỗi lớp chèn xattn phải chạy trong mọi bước giải mã: chi phí suy luận tăng theo số khối chèn. Điểm lợi là KV của ảnh chỉ có 64 token/ảnh và tính một lần.
- Thiết kế "đóng băng toàn bộ + chèn lớp" khiến việc *cắt tỉa* hay đổi LM nền khó: các khối xattn gắn với kích thước ẩn $D$ cụ thể của LM.

## 7. Liên hệ với các bài khác

- **02 GPT-3**: Flamingo tái hiện "few-shot bằng prompt" cho đa phương thức. **04 Chinchilla**: LM nền; Flamingo-80B dùng Chinchilla 70B.
- **01 Transformer**: cross-attention gốc (encoder-decoder); Flamingo dùng nó nhưng thêm cổng tanh.
- **29 ViT / 30 CLIP**: Flamingo tự pretrain vision encoder bằng loss CLIP nhưng dùng NFNet; ablation cho thấy CLIP ViT-L/14 kém hơn NFNet-F6 5.8%.
- **33 BLIP-2**: thay Resampler bằng Q-Former (32 query, có pretrain hai pha) và *bỏ* cross-attention trong LM, chỉ nối đầu ra vào đầu vào LM.
- **34–35 LLaVA**: đi xa hơn: không Resampler, không xattn; chỉ projector tuyến tính/MLP, các visual token nối thẳng vào chuỗi đầu vào của LM.
- **36 InternVL 1.5**: cùng triết lý "ViT–projector–LLM", nhưng giảm token bằng pixel shuffle thay vì Resampler.
- **08 LoRA**: một cách khác để thích ứng LM mà không phá nó; Flamingo dùng "thêm tham số gated", LoRA dùng "thêm hiệu low-rank".

## 8. Ứng dụng cho dự án (suy luận)

Mô hình của bạn (LightOnOCR-3-0.8B) thuộc nhánh *nối thẳng token ảnh vào chuỗi* (như LLaVA), không phải nhánh Flamingo. Tuy vậy bài này đưa ra vài bài học:

1. **Đo chi phí của hai cách tiêm thị giác** (tự tính, ước lượng thô). Gọi $N_v$ là số visual token, $L$ độ dài sinh, $d$ chiều ẩn, $n_{\text{lớp}}$ số lớp decoder.
   - *Nối vào chuỗi* (LLaVA, mô hình của bạn): KV-cache phải chứa $N_v$ token ảnh ở mọi lớp full-attention: $2\cdot N_v\cdot n_{kv}d_h\cdot b$ byte/lớp (b = 2 byte cho fp16). Ví dụ giả định $n_{kv}=8$, $d_h=128$, 24 lớp full-attention, $N_v=3{,}000$ token ảnh + 1,500 token sinh: mỗi token mỗi lớp 4 KiB, tổng $4{,}500\times24\times4\,\text{KiB}\approx 432$ MiB (tự tính, giả định rõ). Với kiến trúc hybrid GatedDeltaNet chỉ một phần lớp là full attention, nên con số thực nhỏ hơn.
   - *Cross-attention kiểu Flamingo*: KV của ảnh chỉ có 64 token/ảnh ở mỗi khối xattn; nhưng thêm tham số mới (~$12D^2$ mỗi khối) và phải tự huấn luyện từ đầu. Với mô hình 0.8B, thêm khối xattn là *tăng* tham số, ngược mục tiêu cắt tỉa.
2. **Ý tưởng Resampler như công cụ giảm visual token**: nếu số token trang ảnh quá lớn gây prefill chậm ở batch 1, một Resampler nhỏ (hoặc pixel shuffle, xem bài 36) có thể nén token. **Cảnh báo:** OCR cần chi tiết nhỏ (chữ nhỏ), nén mạnh bằng 64 token có thể làm hại ký tự; Flamingo không thử OCR dày đặc, nên không có bằng chứng ở đây.
3. **Gating khởi tạo bằng 0 như mẹo cắt tỉa/chèn module**: nếu bạn *chèn* một module hồi phục (ví dụ lớp adapter sau điểm cắt) vào mô hình đã cắt, khởi tạo với cổng $\tanh(0)$ cho phép bắt đầu đúng bằng mô hình hiện tại và học dần sửa chữa. Đây là suy luận của chúng ta, liên hệ với việc phân rã hư hại thành $D_{\text{lin}}$ và $D_{\text{res}}$: một nhánh có cổng có thể bắt đầu bù phần phi tuyến còn lại $D_{\text{res}}$ mà không phá phần đã khớp.
4. **Bài học đóng băng vs fine-tune**: Flamingo thấy fine-tune toàn bộ LM trên dữ liệu ảnh–chữ làm giảm hiệu năng 8%. Trong chưng cất bạn dùng loss KL từ teacher, nên rủi ro quên kiến thức khác với đây, nhưng đáng chú ý nếu phục hồi bằng dữ liệu hẹp (OCR) trên decoder đã cắt.

Kết luận: **liên quan gián tiếp**. Đọc để hiểu bối cảnh kiến trúc VLM, không dùng trực tiếp.

## 9. Thí nghiệm gợi ý

1. **Đo phần trăm KV-cache và thời gian prefill của visual token** trong mô hình của bạn ở batch 1: vẽ latency theo $N_v$ (ví dụ cắt/gộp token ảnh bằng average-pooling 2x2) để biết nút thắt thực sự là prefill ảnh hay decode chữ. Đây là cơ sở để quyết định có đáng thêm Resampler/pixel-shuffle hay không.
2. **Adapter có cổng tanh sau điểm cắt**: sau khi cắt một lớp/neuron, thêm một nhánh residual nhỏ nhân $\tanh(\alpha)$, $\alpha_0=0$, huấn luyện chỉ nhánh này + chưng cất, so với chỉ chưng cất (kiểm tra xem nhánh có cổng có phục hồi tốt hơn khi cắt MLP neuron không, vì bạn quan sát cắt neuron phục hồi kém).
3. **Ablation số token ảnh**: giữ nguyên mô hình, thay đổi độ phân giải đầu vào ViT và đo CER/WER trên tập OCR của bạn, để xác định ngưỡng chi tiết cần thiết.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Xác suất | $p(y\mid x)=\prod_\ell p(y_\ell\mid y_{<\ell},x_{\le\ell})$ |
| Perceiver Resampler | $R=64$ latent học được; $X\leftarrow X+\mathrm{Attn}(q=X,kv=[X_f;X])$; $X\leftarrow X+\mathrm{FFN}(X)$; 6 lớp, 194M tham số |
| Cổng tanh | $y\leftarrow y+\tanh(\alpha)f(y,x)$, $\alpha_0=0$ |
| GATED XATTN-DENSE | xattn có cổng + FFN có cổng, chèn trước lớp LM đóng băng; mỗi 1/4/7 lớp tuỳ model |
| Mask ảnh xen kẽ | token $\ell$ chỉ nhìn visual token của ảnh $\varphi(\ell)$ (ảnh gần nhất trước nó) |
| Loss | tổng có trọng số $\lambda_m$ của NLL theo dataset; tích luỹ gradient |
| Few-shot | prompt xen kẽ (ảnh, chữ) x shot + truy vấn; open-ended: beam=3; close-ended: log-likelihood |
| Đóng băng LM | tránh catastrophic forgetting (+8.0% so với fine-tune LM) |

## 11. Kết luận ngắn

Flamingo chứng minh rằng có thể "dạy" một LM nhìn mà không làm hỏng nó: đóng băng cả vision lẫn LM, nén ảnh bằng Perceiver Resampler, rồi tiêm qua cross-attention có cổng khởi tạo bằng 0, và huấn luyện trên dữ liệu web xen kẽ để có few-shot. Các bài sau (BLIP-2, LLaVA) đơn giản hoá phần "nối" nhưng giữ cùng ý tưởng nền: tái dùng hai mô hình đơn phương thức mạnh. Với dự án của bạn, bài học chính là cách nghĩ về *chi phí của visual token* và về việc *chèn module có cổng để bảo toàn hành vi cũ*.

**Link gốc:** https://arxiv.org/abs/2204.14198
