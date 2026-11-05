# Mistral 7B (2023)

**Tác giả:** Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed (Mistral AI).  
**Venue / arXiv:** arXiv:2310.06825 (v1, 10/10/2023). Bản text không ghi venue.  
**Nguồn:** https://arxiv.org/abs/2310.06825  
**Chủ đề:** Mô hình 7B hiệu quả: Grouped-Query Attention (GQA), Sliding Window Attention (SWA), Rolling Buffer Cache, pre-fill theo chunk; so sánh với Llama 2/Llama 1.  
**Vị trí trong lộ trình:** Đọc sau *05 LLaMA* (kiến trúc nền) và nên biết *15 GQA* (Ainslie et al.) và *17 Longformer* (sliding window). Bài này là bản "kỹ thuật phục vụ" của LLaMA: cùng quy mô nhưng KV-cache nhỏ hơn rất nhiều. Tiếp theo: *28 Mixtral* (cùng nhà, thêm MoE), *55 StreamingLLM* (vì sao cắt cửa sổ làm mất token đầu), *46 PagedAttention* (quản lý KV-cache trên GPU).

> Tài liệu học tập bằng tiếng Việt, ghi chú diễn giải, không phải bản dịch. Paper ngắn (~3.8k từ), nên ghi chú đi sâu vào cơ chế attention và chi phí bộ nhớ. Các con số "tự tính" và phần "Ứng dụng cho dự án" là suy luận của chúng ta, không phải kết luận của paper.

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

Hai bài toán chi phí khi phục vụ LLM:

1. **Attention thuần (full causal attention) tốn $O(n^2)$ tính toán và $O(n)$ bộ nhớ cache theo độ dài chuỗi $n$.** Khi sinh token thứ $i$, mô hình phải so khớp query mới với *toàn bộ* $i$ key trước đó, và phải **giữ KV-cache** (key và value của mọi token đã xử lý, để khỏi tính lại) trong bộ nhớ GPU. Chuỗi dài → cache lớn → batch nhỏ → thông lượng thấp, độ trễ cao.
2. **Multi-head attention (MHA) lưu K,V riêng cho từng đầu**, nên cache nặng. Multi-query (MQA, bài 14) và Grouped-query (GQA, bài 15) cho nhiều đầu query dùng chung ít đầu K/V hơn.

LLaMA (bài 05) cho thấy mô hình nhỏ train nhiều token có thể đánh bại mô hình lớn, nhưng vẫn dùng MHA và full attention. Mistral hỏi: ở quy mô 7B, có thể vừa tăng chất lượng vừa cắt mạnh chi phí suy luận không? Paper tóm gọn quan điểm này trong kết luận: bài toán không phải 2 chiều (năng lực vs chi phí huấn luyện) mà là **3 chiều: năng lực, chi phí huấn luyện, chi phí suy luận**.

---

## 2. Ý tưởng chính

Mistral 7B là một Transformer decoder kiểu LLaMA với hai chỉnh sửa attention nhằm suy luận nhanh và rẻ. Thứ nhất, **GQA**: 32 đầu query chia sẻ chỉ 8 đầu key/value, làm KV-cache nhỏ đi 4 lần và giải mã nhanh hơn. Thứ hai, **sliding window attention (SWA)**: mỗi token chỉ attend vào $W=4096$ token gần nhất ở mỗi layer; nhờ xếp chồng nhiều layer, thông tin vẫn lan xa hơn $W$ (sau $k$ layer, tới $k\times W$ token). Vì phạm vi attention cố định, cache có thể là **bộ đệm vòng (rolling buffer)** kích thước cố định $W$, không tăng theo độ dài chuỗi. Kết quả: 7B tham số vượt Llama 2 13B trên mọi benchmark paper thử và vượt Llama 1 34B ở toán, code, suy luận.

---

## 3. Phương pháp chi tiết

### 3.1 Cấu hình kiến trúc (Bảng 1)

| Tham số | Giá trị |
|---|---|
| dim | 4096 |
| n_layers | 32 |
| head_dim | 128 |
| hidden_dim (FFN) | 14336 |
| n_heads | 32 |
| n_kv_heads | 8 |
| window_size | 4096 |
| context_len | 8192 |
| vocab_size | 32000 |

Paper nói "Mistral 7B is based on a transformer architecture" và chỉ liệt kê "những thay đổi so với Llama"; các phần còn lại (RMSNorm, SwiGLU, RoPE) không được mô tả lại trong bài, nên ở đây tôi chỉ coi là *giả định hợp lý theo họ Llama*, không phải điều paper xác nhận. Không có thông tin về dữ liệu huấn luyện, số token, hay hyper-parameter huấn luyện trong paper.

**Kiểm tra số tham số (tự tính):** mỗi layer: attention $Q$: $4096\times4096$, $O$: $4096\times4096$, $K,V$: mỗi cái $4096\times(8\cdot128)=4096\times1024$; tổng attention $=2\cdot16.78\text{M}+2\cdot4.19\text{M}=41.9$M. FFN SwiGLU 3 ma trận: $3\times4096\times14336=176.2$M. Mỗi layer ≈ 218.1M, 32 layer ≈ 6.98B; thêm embedding vào + ra $2\times32000\times4096=0.26$B → **≈ 7.24B** (khớp mức "7B"; lưu ý GQA tiết kiệm $K,V$ rõ rệt so với MHA là $2\times16.8$M mỗi layer).

### 3.2 Grouped-Query Attention (GQA)

Với $h_q=32$ đầu query và $h_{kv}=8$ đầu K/V, mỗi nhóm $g=h_q/h_{kv}=4$ đầu query dùng chung một cặp (K,V). MHA là trường hợp $h_{kv}=h_q$; MQA là $h_{kv}=1$.

$$\text{head}_j=\text{softmax}\!\Big(\frac{q_j\,K_{\lceil j/g\rceil}^\top}{\sqrt{d_{head}}}\Big)V_{\lceil j/g\rceil},\qquad j=1..h_q$$

Hệ quả: tham số $W_K,W_V$ nhỏ đi $g$ lần và **KV-cache nhỏ đi $g=4$ lần** so với MHA cùng số đầu query. Paper chỉ nói GQA "significantly accelerates the inference speed, and also reduces the memory requirement during decoding, allowing for higher batch sizes hence higher throughput"; không đưa số đo riêng cho GQA hay ablation so MHA.

### 3.3 Sliding Window Attention (SWA)

Thay vì token $i$ attend vào mọi $j\le i$, chỉ attend vào $j\in[i-W,\,i]$:

$$\boxed{A_{ij}\neq0\iff i-W\le j\le i}$$

Chi phí tính toán mỗi layer giảm từ $\sim n^2/2$ xuống $\sim nW$ cặp (query, key).

**Vì sao thông tin vẫn lan xa?** Trạng thái ẩn $h_i^{(k)}$ ở layer $k$ attend các $h_j^{(k-1)}$ với $j\in[i-W,i]$, mà mỗi $h_j^{(k-1)}$ lại đã tích hợp thông tin từ $[j-W,j]$ ở layer trước, v.v. Vì vậy sau $k$ layer, token $i$ có thể "thấy gián tiếp" token đầu vào ở khoảng cách tới $k\times W$:

$$\boxed{\text{tầm với lý thuyết}=k\times W=32\times4096=131{,}072\ (\approx131\text{K})}$$

Đây là *trần lý thuyết*, như paper nói "theoretical attention span of approximately 131K tokens"; paper không đo khả năng thực sự dùng thông tin ở khoảng cách đó (ví dụ không có thí nghiệm needle-in-a-haystack). Trong thực tế thông tin bị pha loãng qua mỗi bước truyền.

Minh họa với $W=3$: token ở vị trí 10 ở layer 1 thấy các vị trí 7-10; ở layer 2 thấy hidden state của vị trí 7-10, mà vị trí 7 ở layer 1 thấy 4-7, nên tầm với trực tiếp lan tới vị trí 4 sau 2 layer, tức $2\times3=6$ bước lùi (hình 1 của paper).

**Về triển khai:** chỉnh sửa trong FlashAttention và xFormers cho tốc độ nhanh gấp **2 lần** baseline attention thường với độ dài chuỗi 16K và $W=4096$. Kiểm tra (tự tính): số cặp (query,key) nhân quả đầy đủ $n(n+1)/2\approx134.2$M; với SWA $\approx nW-W^2/2\approx58.7$M; tỉ lệ ≈ 2.29 lần, cùng bậc với con số 2x của paper.

### 3.4 Rolling Buffer Cache

Vì phạm vi attention là $W$, ta chỉ cần giữ KV của $W$ token gần nhất. Cache có kích thước cố định $W$; K,V của vị trí $i$ lưu vào ô $i \bmod W$. Khi $i\ge W$, ghi đè giá trị cũ nhất.

Ví dụ $W=4$ (Hình 2): vị trí 0,1,2,3 vào ô 0,1,2,3; vị trí 4 ghi đè ô 0 (vì $4\bmod4=0$, token 0 đã ra khỏi cửa sổ), vị trí 5 ghi đè ô 1, v.v.

```python
class RollingKVCache:
    def __init__(self, W, n_layers, n_kv, d_head):
        self.W = W
        self.K = zeros(n_layers, W, n_kv, d_head)
        self.V = zeros(n_layers, W, n_kv, d_head)

    def write(self, layer, pos, k, v):
        slot = pos % self.W          # vị trí i lưu ở ô i mod W
        self.K[layer, slot] = k      # ghi đè token cũ nhất khi pos >= W
        self.V[layer, slot] = v

    def read(self, layer, pos):
        n = min(pos + 1, self.W)     # tối đa W token hợp lệ
        # trả về K,V của n token gần nhất (cần ghi nhớ thứ tự theo pos,
        # vì RoPE/mask dựa trên vị trí tuyệt đối thật của từng ô)
        return self.K[layer, :n], self.V[layer, :n]
```

(Pseudo-code tự viết; paper chỉ mô tả ý tưởng, không đưa code.) Lưu ý: ô cache không còn theo thứ tự thời gian sau khi vòng lại, nên cần theo dõi vị trí thật của từng ô để áp mask/positional encoding đúng.

**Hiệu quả bộ nhớ:** trên chuỗi 32k token, giảm bộ nhớ cache **8 lần** ($32768/4096=8$) mà không ảnh hưởng chất lượng theo paper.

### 3.5 Pre-fill và chunking

Khi sinh, các token sinh ra phải tuần tự, nhưng **prompt đã biết trước**, nên có thể nạp (pre-fill) KV-cache cho prompt song song. Nếu prompt quá dài, cắt thành các **chunk** (chọn kích thước chunk $=W$) và pre-fill lần lượt. Với mỗi chunk, attention gồm hai phần: (a) attention *vào cache* (các token của chunk trước, theo cửa sổ trượt), (b) attention *trong chính chunk* với mask nhân quả. Hình 3: chia "The cat sat on the mat and saw the dog go to" thành 3 chunk; ở chunk thứ ba ("the dog go to"), nó attend chính nó bằng mask nhân quả, attend cache qua cửa sổ trượt, và không attend các token đã ra khỏi cửa sổ.

---

## 4. Thí nghiệm và kết quả chính

Paper **chạy lại mọi benchmark bằng pipeline đánh giá của chính họ** cho công bằng (nên số Llama có thể khác số trong paper Llama). Khác biệt giao thức: MBPP dùng tập con đã kiểm bằng tay; TriviaQA không cung cấp ngữ cảnh Wikipedia.

Nhóm benchmark: suy luận thường thức 0-shot (HellaSwag, WinoGrande, PIQA, SIQA, OpenbookQA, ARC-e/c, CommonsenseQA); kiến thức thế giới 5-shot (NQ, TriviaQA); đọc hiểu 0-shot (BoolQ, QuAC); toán GSM8K 8-shot maj@8 và MATH 4-shot maj@4; code HumanEval 0-shot, MBPP 3-shot; tổng hợp MMLU 5-shot, BBH 3-shot, AGI Eval 3-5-shot.

### 4.1 So với Llama (Bảng 2, base models)

| Mô hình | MMLU | HellaSwag | WinoG | PIQA | Arc-e | Arc-c | NQ | TriviaQA | HumanEval | MBPP | MATH | GSM8K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaMA 2 7B | 44.4 | 77.1 | 69.5 | 77.9 | 68.7 | 43.2 | 24.7 | 63.8 | 11.6 | 26.1 | 3.9 | 16.0 |
| LLaMA 2 13B | 55.6 | 80.7 | 72.9 | 80.8 | 75.2 | 48.8 | 29.0 | 69.6 | 18.9 | 35.4 | 6.0 | 34.3 |
| Code-Llama 7B (fine-tuned) | 36.9 | 62.9 | 62.3 | 72.8 | 59.4 | 34.5 | 11.0 | 34.9 | 31.1 | 52.5 | 5.2 | 20.8 |
| **Mistral 7B** | **60.1** | **81.3** | **75.3** | **83.0** | **80.0** | **55.5** | 28.8 | **69.9** | **30.5** | 47.5 | **13.1** | **52.2** |

Đơn vị %. Mistral 7B vượt Llama 2 13B ở hầu hết cột; ở NQ gần ngang (28.8 vs 29.0). Đặc biệt MMLU +4.5 điểm so với 13B, GSM8K 52.2 so với 34.3, MATH 13.1 so với 6.0. Paper nêu Mistral gần với Code-Llama 7B về code (30.5 vs 31.1 HumanEval) mà không hy sinh benchmark phi-code, đồng thời Code-Llama 7B kém trên mọi benchmark ngoài code.

### 4.2 "Kích thước tương đương" (Hình 5)

Paper tính "equivalent model size" của họ Llama 2: trên suy luận, đọc hiểu và MMLU/STEM, Mistral 7B tương đương Llama 2 **lớn hơn hơn 3 lần**; trên benchmark kiến thức chỉ **1.9 lần**, được giải thích (giả thuyết) là do số tham số hạn chế dung lượng kiến thức có thể lưu.

### 4.3 Instruction finetuning (Bảng 3)

Chỉ fine-tune trên các dataset instruction công khai của Hugging Face, "không dữ liệu độc quyền hay mẹo huấn luyện":

| Mô hình | Chatbot Arena ELO | MT-Bench |
|---|---|---|
| WizardLM 13B v1.2 | 1047 | 7.2 |
| Mistral 7B Instruct | 1031 | 6.84 ± 0.07 |
| Llama 2 13B Chat | 1012 | 6.65 |
| Vicuna 13B | 1041 | 6.57 |
| Llama 2 7B Chat | 985 | 6.27 |
| Vicuna 7B | 997 | 6.17 |
| Alpaca 13B | 914 | 4.53 |

Đánh giá người độc lập trên llmboxing.com (tính đến 6/10/2023): đầu ra Mistral 7B được ưa thích 5020 lần so với 4143 lần của Llama 2 13B.

### 4.4 Guardrails

Với system prompt đề xuất, mô hình từ chối 100% trong 175 prompt không an toàn; MT-Bench: không system prompt 6.84 ± 0.07, system prompt của Mistral 6.58 ± 0.05, của Llama 2 6.38 ± 0.07. Tự kiểm duyệt nội dung (self-reflection): precision 99.4% tại recall 95.6% trên tập tự dựng.

---

## 5. Phân tích / ablation

Đáng nói thẳng: **paper không có ablation nào** cho GQA, SWA hay rolling buffer. Không có bảng "với/không có SWA", không có đo độ chính xác theo độ dài ngữ cảnh, không có so sánh GQA với MHA, và không có thông tin dữ liệu huấn luyện. Những gì được hỗ trợ bằng số liệu:

- Chất lượng chung: Bảng 2 (so Llama).
- Tốc độ SWA: tăng gấp 2 lần ở chuỗi 16K, $W=4096$ nhờ sửa đổi FlashAttention/xFormers.
- Bộ nhớ cache giảm 8 lần ở 32k token (chỉ phép chia, không đo chất lượng riêng; câu "without impacting the model quality" là tuyên bố của tác giả).
- Điểm yếu về kiến thức (1.9x vs 3x) là quan sát có số liệu, nguyên nhân là giả thuyết.

Vì vậy mọi đóng góp riêng của GQA/SWA đối với điểm số benchmark là **không tách được** từ paper; chất lượng cao hơn Llama 2 có thể đến từ dữ liệu và huấn luyện (không công bố) chứ không nhất thiết từ attention.

---

## 6. Giới hạn

**Do tác giả nêu:**
- Dung lượng kiến thức hạn chế do số tham số nhỏ (kiến thức chỉ nén 1.9x).
- Mô hình Instruct chỉ là "demonstration đơn giản và sơ bộ", chưa có cơ chế moderation tích hợp ngoài system prompt.

**Tôi nhận xét thêm (suy luận):**
- Không có dữ liệu/ablation, không mô tả huấn luyện → khó tái lập và khó quy kết nguyên nhân.
- Tầm với 131K chỉ là trần lý thuyết; mô hình không được kiểm chứng khả năng truy hồi thông tin ở xa (context_len trong Bảng 1 chỉ 8192).
- SWA làm mất truy cập trực tiếp tới token cũ (kể cả token đầu, nơi thường chứa "attention sink" mà StreamingLLM, bài 55, chỉ ra là quan trọng); phải dựa vào lan truyền gián tiếp qua nhiều layer, có thể mất chi tiết chính xác (ví dụ chép lại một chuỗi dài từ rất xa).
- Rolling buffer làm hỏng khả năng tái sử dụng prefix cache theo kiểu RadixAttention (bài 47) vì cache bị ghi đè.
- Số liệu Llama tự chạy lại nên có thể lệch so với số công bố của Meta; không có khoảng tin cậy.
- Chỉ có model 7B; không rõ cách các kỹ thuật này scale.

---

## 7. Liên hệ với các bài khác

- **05 LLaMA:** nền kiến trúc; Mistral so sánh trực tiếp với Llama 1/2 và nhắc "Compared to Llama, it introduces a few changes".
- **15 GQA, 14 MQA:** nguồn gốc của kỹ thuật nhóm đầu K/V. **16 MLA (DeepSeek-V2):** cách khác để nén KV-cache (nén low-rank), mạnh hơn GQA.
- **17 Longformer:** nguồn gốc của sliding-window attention; paper cũng trích Sparse Transformer (Child et al.). GPT-3 (bài 02) xen kẽ attention dày và dải băng cục bộ, còn Mistral dùng cửa sổ ở mọi layer.
- **44/45 FlashAttention:** Mistral cần chỉnh kernel FlashAttention để hỗ trợ SWA.
- **46 PagedAttention (vLLM):** paper chọn vLLM làm server; PagedAttention quản lý KV-cache theo trang, bổ sung cho rolling buffer.
- **55 StreamingLLM / 56 H2O:** hai hướng khác để giới hạn KV-cache (giữ attention sink + cửa sổ; loại bỏ token ít quan trọng).
- **18-26 (attention tuyến tính / SSM / hybrid):** thay vì cắt cửa sổ, các mô hình này nén quá khứ vào state cố định; SWA là phương án đơn giản giữ softmax attention nhưng bị chặn.
- **28 Mixtral:** cùng nhóm tác giả, thêm Mixture-of-Experts lên nền kiến trúc này.

---

## 8. Ứng dụng cho dự án (suy luận)

### 8.1 Phân tích chi phí KV-cache (tự tính)

$$\boxed{\text{KV bytes}=2\cdot L\cdot n_{kv}\cdot d_{head}\cdot T_{\text{cache}}\cdot b}$$

với $T_{\text{cache}}=\min(T,W)$ khi dùng SWA + rolling buffer, $b=2$ (fp16/bf16).

**Mistral 7B** ($L=32,\ n_{kv}=8,\ d_{head}=128$): 131,072 B = **128 KiB/token** với GQA.

| Cấu hình, $T=32{,}768$ | Cache |
|---|---|
| MHA, full attention (kiểu LLaMA-7B: $n_{kv}=32$) | 16 GiB |
| GQA, full attention | 4 GiB |
| GQA + SWA ($W=4096$, rolling buffer) | **512 MiB** |

Tức riêng SWA giảm 8x (khớp paper), GQA thêm 4x, tổng 32x so với MHA full attention. (Lưu ý: so sánh đến từ phép tính, không phải đo.)

**Chi phí tính toán attention:** mỗi token mới tính $\min(i,W)$ tích vô hướng ($\times$ số đầu) thay vì $i$. Ở chuỗi $n=16{,}384$, $W=4096$: ≈ 58.7M cặp so với ≈ 134.2M, tỉ lệ ≈ 2.3.

### 8.2 Với mô hình của bạn (0.8B/0.3B/0.1B, ~24 layer, d≈1024, 1500 token/trang, batch 1)

**Quan sát 1: SWA với $W=4096$ không có tác dụng.** Một trang OCR chỉ ~1500 token đầu ra (cộng token ảnh) nhỏ hơn $W$, nên cửa sổ bao hết toàn bộ chuỗi: cache và FLOPs y hệt full attention. Để thấy lợi ích phải đặt $W$ nhỏ hơn nhiều (ví dụ 256-512), và khi đó rủi ro bắt đầu.

**Quan sát 2: rủi ro với OCR VLM.** Trong VLM, các token ảnh (đầu ra ViT) thường nằm *ở đầu chuỗi*, trước phần văn bản sinh ra. Với SWA nhỏ, token thứ 1000 của văn bản sẽ không còn attend trực tiếp tới token ảnh nữa (chỉ gián tiếp qua nhiều layer), trong khi OCR cần "nhìn lại ảnh" ở đúng vùng đang đọc. Nếu cần giảm chi phí, nên (a) giữ token ảnh luôn trong tầm nhìn (cửa sổ + prefix cố định, tương tự attention sink của StreamingLLM) hoặc (b) chỉ áp cửa sổ cho các layer chỉ nhìn văn bản. Đây là giả thuyết thiết kế, cần thử nghiệm.

**Quan sát 3: bạn đã có "cache cố định" qua GatedDeltaNet.** Các layer GatedDeltaNet có state kích thước cố định, tương tự mục đích của rolling buffer nhưng bằng cách nén (không cắt). Chỉ các layer full attention mới sinh KV-cache. Ví dụ (tự tính): giả sử 24 layer, 6 layer full attention, 16 đầu × 64, MHA, fp16, 1500 token → $2\cdot6\cdot16\cdot64\cdot2\cdot1500\approx$ 36.9 MB; nếu thêm GQA 4 nhóm → khoảng 9.2 MB. Đều nhỏ so với trọng số 0.1-0.8B (0.2-1.6 GB ở fp16): **ở batch 1, trọng số, không phải KV-cache, chi phối bộ nhớ và băng thông.** Vì vậy GQA/SWA ít cải thiện độ trễ batch-1 của bạn; chúng hữu ích hơn cho thông lượng batch lớn hoặc ngữ cảnh dài.

**Quan sát 4: GQA khi cắt (pruning).** Nếu bạn cắt đầu attention của các layer full attention, GQA cho cấu trúc tự nhiên: cắt cả nhóm (1 đầu K/V cùng $g$ đầu Q) để giữ tính nhất quán. Bài này không nói gì về pruning; đây là suy luận.

**Quan sát 5: chunked pre-fill** hữu ích nếu prompt có token ảnh dài (ảnh độ phân giải cao): chia chunk giới hạn bộ nhớ đỉnh lúc pre-fill trên GPU ≤ 8 GB.

Kết quả benchmark (MMLU, code, v.v.) **ít liên quan** tới OCR.

---

## 9. Thí nghiệm gợi ý

1. **Ảnh hưởng của cửa sổ lên OCR:** thêm mask SWA với $W\in\{128,256,512,\infty\}$ vào các layer full attention của mô hình đã train (không train lại hoặc fine-tune ngắn), đo CER theo vị trí token trong trang. Kỳ vọng (giả thuyết): CER tăng ở phần cuối trang nếu token ảnh nằm ngoài cửa sổ; thử biến thể "cửa sổ + luôn thấy token ảnh".
2. **Đo cache thực:** bật log bộ nhớ khi sinh 1500 token ở batch 1, so với công thức 8.1; thử $n_{kv}$ khác nhau nếu gom đầu K/V (mean-pool các đầu K,V theo nhóm như trong bài GQA) rồi fine-tune ngắn. Xem độ trễ có đổi không (dự đoán: gần như không đổi ở batch 1).
3. **Receptive field thực nghiệm:** với SWA nhỏ, kiểm tra xem thông tin "copy token ở khoảng cách $d$" có còn truy hồi được khi $d>W$ nhưng $d<kW$ không (kiểm chứng trần $k\times W$ mà paper chỉ nêu lý thuyết).

---

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| GQA | $h_q=32$ đầu query, $h_{kv}=8$ đầu K/V, nhóm $g=4$ |
| SWA | Token $i$ attend $j\in[i-W,i]$, $W=4096$ |
| Tầm với lý thuyết | $k\times W=32\times4096\approx131$K |
| Rolling buffer | Ghi KV vị trí $i$ vào ô $i \bmod W$; cache kích thước cố định $W$ |
| Giảm cache | 8x ở 32k token (SWA); +4x từ GQA so với MHA (tự tính) |
| Pre-fill + chunk | Nạp cache cho prompt theo chunk cỡ $W$; chunk attend cache (cửa sổ) + chính nó (mask nhân quả) |
| KV bytes | $2\,L\,n_{kv}\,d_{head}\,\min(T,W)\,b$ (tự tính) |
| Mistral 7B | 32 layer, dim 4096, FFN 14336, head_dim 128, vocab 32000, ≈7.2B tham số (tự tính) |
| Tốc độ SWA | ~2x ở chuỗi 16K, $W=4096$ (FlashAttention/xFormers đã sửa) |

---

## 11. Kết luận ngắn

Mistral 7B là minh chứng cho quan điểm "tối ưu chi phí suy luận ngay trong thiết kế": GQA thu nhỏ KV-cache theo chiều đầu, SWA + rolling buffer thu nhỏ theo chiều độ dài, và kết quả benchmark vượt Llama 2 13B. Điểm cần nhớ khi đọc kỹ: paper không có ablation hay thông tin dữ liệu, nên không tách được đóng góp từng thành phần. Với dự án của bạn, bài này quan trọng nhất như bản đồ chi phí KV-cache và cảnh báo rằng cắt cửa sổ attention có thể làm mất liên kết văn bản-ảnh trong OCR VLM, còn ở batch 1 với 1500 token thì lợi ích của GQA/SWA là nhỏ so với chi phí đọc trọng số.

Nguồn gốc: https://arxiv.org/abs/2310.06825
