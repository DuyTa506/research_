# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (2023)

**Tác giả:** Tri Dao.  
**Venue / arXiv:** arXiv:2307.08691 (v1, 17/07/2023).  
**Nguồn:** https://arxiv.org/abs/2307.08691  
**Chủ đề:** Tối ưu kernel attention chính xác trên GPU: giảm FLOPs không phải matmul, song song theo chiều độ dài chuỗi, chia việc giữa các warp; GQA/MQA; causal masking theo khối.  
**Vị trí trong lộ trình:** Đọc trước: 44 (FlashAttention, tiling + online softmax + IO complexity), 01. Bài này thêm: cùng thuật toán và cùng kết quả, nhưng khai thác phần cứng tốt hơn (~2x). Đọc tiếp: 46 (PagedAttention), 15 (GQA, vì FA2 hỗ trợ GQA/MQA), 48 (speculative decoding, nơi batch-1 decode có đặc điểm khác hẳn).

> Đây là ghi chú học tập bằng tiếng Việt, diễn giải lại chứ không phải bản dịch. Các con số lấy từ paper (có ghi rõ). Phần "Ứng dụng cho dự án" và các số "tự tính" là suy luận của chúng ta, không phải kết luận của paper. Paper là loại trung bình (~7k từ), nhiều số liệu nằm trong biểu đồ nên chỉ trích các số xuất hiện trong chữ hoặc bảng.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
   - 3.1 [Tối ưu 1: bớt FLOPs không phải matmul](#31-tối-ưu-1-bớt-flops-không-phải-matmul)
   - 3.2 [Tối ưu 2: song song theo chiều độ dài chuỗi](#32-tối-ưu-2-song-song-theo-chiều-độ-dài-chuỗi)
   - 3.3 [Tối ưu 3: chia việc giữa các warp](#33-tối-ưu-3-chia-việc-giữa-các-warp)
   - 3.4 [Causal mask, MQA/GQA, block size](#34-causal-mask-mqagqa-block-size)
   - 3.5 [Phân tích chi phí và ví dụ cho mô hình của ta](#35-phân-tích-chi-phí-và-ví-dụ-cho-mô-hình-của-ta)
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

FlashAttention (bài 44) đã giảm IO và bộ nhớ, nhanh hơn 2-4x so với attention chuẩn. Nhưng so với phép nhân ma trận (GEMM) tối ưu vốn đạt 80-90% thông lượng lý thuyết của GPU, FlashAttention v1 chỉ đạt **30-50%** ở forward và **25-35%** ở backward (A100). Tác giả profile và kết luận nguyên nhân là **phân chia công việc dưới tối ưu** giữa các thread block và warp, gây hoặc thấp occupancy hoặc đọc/ghi shared memory thừa.

Gloss thuật ngữ GPU cần dùng:

- **Thread block:** nhóm thread chạy trên một SM (streaming multiprocessor); A100 có 108 SM.
- **Warp:** 32 thread chạy đồng bộ; thread trong warp trao đổi nhanh bằng shuffle; warp trong block trao đổi qua shared memory (SRAM), chậm hơn.
- **Occupancy:** mức độ tận dụng tài nguyên GPU. Ít thread block hơn số SM thì có SM bị bỏ không.
- **Tensor Core:** đơn vị chuyên nhân ma trận (fp16/bf16). A100: 312 TFLOPs/s cho matmul fp16/bf16 nhưng chỉ 19.5 TFLOPs/s cho FP32 không phải matmul, nghĩa là mỗi FLOP không phải matmul "đắt" gấp ~16 lần.
- **Split-K:** cách chia việc theo chiều rút gọn (K), các warp tính riêng rồi phải cộng lại qua shared memory.

## 2. Ý tưởng chính

Giữ nguyên kết quả toán học của FlashAttention nhưng làm ba việc: (1) sửa công thức online softmax để **giảm phép toán không phải matmul** (vì chúng đắt gấp 16 lần), (2) **đảo vòng lặp** và song song hoá cả theo chiều độ dài chuỗi (không chỉ batch và head) để lấp đầy GPU khi chuỗi dài, batch nhỏ, (3) trong mỗi thread block, **chia $Q$ giữa các warp thay vì chia $K,V$** để các warp khỏi phải giao tiếp qua shared memory. Kết quả khoảng 2x nhanh hơn FlashAttention, đạt 50-73% thông lượng đỉnh trên A100.

## 3. Phương pháp chi tiết

### 3.1 Tối ưu 1: bớt FLOPs không phải matmul

Online softmax của FlashAttention (bài 44) với hai khối (xét một hàng, $S=[S^{(1)},S^{(2)}]$):

$$O^{(2)}=\mathrm{diag}\!\big(\ell^{(1)}/\ell^{(2)}\big)^{-1}O^{(1)}+\mathrm{diag}(\ell^{(2)})^{-1}e^{S^{(2)}-m^{(2)}}V^{(2)}.$$

Mỗi vòng đều phải chia (rescale) bằng $\mathrm{diag}(\ell)^{-1}$ cho cả hai số hạng. FA2 nhận ra không cần: giữ $\tilde O$ **chưa chuẩn hoá**:

$$\tilde O^{(1)}=e^{S^{(1)}-m^{(1)}}V^{(1)},\qquad \tilde O^{(2)}=\mathrm{diag}\!\big(e^{m^{(1)}-m^{(2)}}\big)\tilde O^{(1)}+e^{S^{(2)}-m^{(2)}}V^{(2)},$$

và chỉ chia **một lần ở cuối vòng lặp**:

$$\boxed{O=\mathrm{diag}\!\big(\ell^{(\text{last})}\big)^{-1}\tilde O^{(\text{last})}}.$$

Tối ưu thứ hai: backward chỉ cần **logsumexp** $L=m+\log\ell$ cho mỗi hàng thay vì lưu cả $m$ và $\ell$. Khi đó $P_{ij}=\exp(S_{ij}-L_i)$ tính lại trực tiếp. Chứng minh: $\exp(S-m)/\ell=\exp(S-m-\log\ell)=\exp(S-L)$.

Thuật toán forward của FA2 (Algorithm 1), **vòng ngoài là khối hàng $Q_i$, vòng trong là khối cột $K_j,V_j$**:

```text
for i = 1..Tr:                              # mỗi khối hàng Q_i (song song giữa các thread block)
    nạp Q_i vào SRAM
    O~ = 0, l = 0, m = -inf                 # trên chip
    for j = 1..Tc:
        nạp K_j, V_j vào SRAM
        S    = Q_i K_j^T
        m_new = max(m, rowmax(S))
        P~   = exp(S - m_new)
        l    = exp(m - m_new) * l + rowsum(P~)
        O~   = diag(exp(m - m_new)) * O~ + P~ V_j     # KHÔNG chia cho l ở đây
        m    = m_new
    O_i = diag(l)^-1 * O~                   # chia một lần
    L_i = m + log(l)
    ghi O_i, L_i về HBM
```

So với FA1 (bài 44): O, $\ell$, $m$ không còn bị đọc/ghi về HBM ở mỗi vòng trong; mỗi khối hàng ghi đúng một lần. Paper nói thứ tự vòng lặp này (ngoài là hàng, trong là cột) đã được Phil Tillet gợi ý và cài trong Triton trước.

Độ đúng và chi phí không đổi: kết quả chính xác, $O(N^2d)$ FLOPs, thêm $O(N)$ bộ nhớ (lưu $L$).

### 3.2 Tối ưu 2: song song theo chiều độ dài chuỗi

FA1 dùng **1 thread block cho mỗi (batch, head)**, tổng batch×heads block. Hiệu quả khi con số này lớn (paper nói ≥ 80, vì A100 có 108 SM). Nhưng với chuỗi dài thường batch nhỏ, head ít, nên nhiều SM bỏ không.

- **Forward:** vòng ngoài (theo khối hàng của $Q$) là "embarrassingly parallel", mỗi khối hàng giao cho một thread block riêng, không cần giao tiếp. Thêm vào song song theo batch và head như cũ.
- **Backward:** các khối cột chỉ chia sẻ phép cập nhật $dQ$ (Algorithm 2: $dQ_i\leftarrow dQ_i+dS_i^{(j)}K_j$). Giao mỗi khối cột cho một thread block và dùng **atomic add** để các block cộng dồn vào $dQ$.

Backward (Algorithm 2) giữ công thức của bài 44: $P=\exp(S-L)$, $dV_j\mathrel{+}=P^\top dO_i$, $dP=dO_iV_j^\top$, $dS=P\circ(dP-D_i)$ với $D=\mathrm{rowsum}(dO\circ O)$, $dK_j\mathrel{+}=dS^\top Q_i$.

### 3.3 Tối ưu 3: chia việc giữa các warp

Mỗi thread block thường có 4 hoặc 8 warp.

- **FA1 ("split-K"):** chia $K,V$ cho 4 warp, mọi warp thấy $Q$. Mỗi warp tính một lát $QK^\top$, rồi nhân với lát $V$ tương ứng, nhưng kết quả phải **cộng lại giữa các warp**: mỗi warp ghi kết quả trung gian vào shared memory, đồng bộ, rồi cộng. Các lần đọc/ghi shared memory này làm chậm.
- **FA2:** chia **$Q$** cho 4 warp, mọi warp thấy $K,V$. Mỗi warp tính hàng của nó với toàn bộ $K,V$ và tự ra lát output của mình, **không cần giao tiếp**.

Backward cũng tránh split-K nhưng vẫn cần một số đồng bộ vì phụ thuộc phức tạp giữa $Q,K,V,O,dO,dQ,dK,dV$ (backward có 5 phép matmul so với 2 của forward).

### 3.4 Causal mask, MQA/GQA, block size

- **Causal masking:** vì đã chia khối, các khối có toàn bộ chỉ số cột lớn hơn chỉ số hàng (khoảng nửa số khối khi chuỗi dài) được **bỏ qua hoàn toàn**, cho tốc độ khoảng 1.7-1.8x so với không mask. Các khối có chỉ số hàng chắc chắn nhỏ hơn cột thì không cần áp mask; mỗi hàng chỉ cần áp mask ở một khối (nếu khối vuông).
- **MQA/GQA (bài 14-15):** thay vì nhân đôi $K,V$ cho các query head dùng chung, FA2 **thao tác chỉ số head ngầm** để dùng lại. Backward phải cộng dồn $dK,dV$ qua các head chia sẻ.
- **Block size:** thường chọn $\{64,128\}\times\{64,128\}$ tuỳ head dim và shared memory. Khối lớn giảm đọc/ghi shared memory nhưng cần nhiều register hơn; vượt ngưỡng thì register spilling hoặc không đủ shared memory để chạy. Phải chỉnh tay cho từng head dim (chỉ có ~4 lựa chọn).

### 3.5 Phân tích chi phí và ví dụ cho mô hình của ta

Công thức FLOPs dùng trong paper:

- Forward: $4\cdot\text{seqlen}^2\cdot\text{head dim}\cdot\text{số head}$ (hai matmul, mỗi cái $2N^2d$); chia 2 nếu causal.
- Backward: $2.5\times$ forward (5 matmul so với 2, vì có tính lại).
- Huấn luyện end-to-end: $6\cdot\text{seqlen}\cdot\text{số tham số}+12\cdot\text{số layer}\cdot\text{hidden}\cdot\text{seqlen}^2$ (kiểu Megatron-LM, không chia 2 cho causal).

**Ví dụ (tự tính, giả định):** một layer full attention với $N=2500$, $d=128$, 16 head: forward $=4\cdot2500^2\cdot128\cdot16\approx51.2$ GFLOP (không causal), ~25.6 GFLOP nếu causal. Ở mức 50% của 312 TFLOPs/s (~156 TFLOPs/s) thì bản causal chỉ mất ~0.16 ms, nên với mô hình 0.8B, kernel attention prefill không phải điểm nghẽn. Điều quan trọng hơn là **chế độ decode** (xem Mục 8).

Về occupancy (suy luận từ Mục 3.2): với decode batch 1, 16 head thì chỉ có $1\times16=16$ thread block cho 108 SM (A100) nếu chỉ song song theo (batch, head); độ dài query là 1 nên cũng không thể song song thêm theo chiều query. Đây là lý do cộng đồng sau này có kỹ thuật chia theo chiều khoá/giá trị (split-KV, thường gọi Flash-Decoding), nhưng điều này **không nằm trong paper này**; ghi chú để tra cứu.

## 4. Thí nghiệm và kết quả chính

**Thiết lập benchmark:** A100 80GB SXM4; độ dài chuỗi 512, 1k, ..., 16k, batch sao cho tổng 16k token; hidden 2048, head dim 64 (32 head) hoặc 128 (16 head); với và không có causal mask.

Các kết luận trong chữ của paper:

| Nội dung | Kết quả |
|---|---|
| FA2 vs FlashAttention (v1) | nhanh hơn 1.7-3.0x |
| FA2 vs FlashAttention trong Triton | 1.3-2.5x (forward 1.3-1.5x, backward ~2x) |
| FA2 vs attention chuẩn PyTorch | 3-10x (tới 10x ở một số cấu hình) |
| Thông lượng đỉnh | tới 230 TFLOPs/s (73% lý thuyết) forward; backward tới 63% |
| H100 (chạy cùng code, chưa dùng TMA/Tensor Core thế hệ 4) | tới 335 TFLOPs/s |
| Phần trăm đỉnh của FA1 (so sánh) | forward 30-50%, backward 25-35% |

**Huấn luyện end-to-end (Bảng 1 của paper), GPT-style, 8×A100 80GB, TFLOPs/s mỗi GPU:**

| Mô hình | Không FlashAttention | FlashAttention | FlashAttention-2 |
|---|---|---|---|
| GPT3-1.3B, ctx 2k | 142 | 189 | 196 |
| GPT3-1.3B, ctx 8k | 72 | 170 | 220 |
| GPT3-2.7B, ctx 2k | 149 | 189 | 205 |
| GPT3-2.7B, ctx 8k | 80 | 175 | 225 |

Paper tóm: tới 2.8x so với không dùng FlashAttention, 1.3x so với FlashAttention v1, tới 225 TFLOPs/s mỗi GPU A100 (72% MFU, "model FLOPs utilization", tỉ lệ FLOPs hữu dụng của mô hình trên đỉnh phần cứng). Kết luận thực tế mà tác giả nêu: cùng chi phí huấn luyện, có thể dùng ngữ cảnh 16k thay vì 8k.

## 5. Phân tích / ablation

Paper **không có bảng ablation riêng** từng tối ưu. Mức đóng góp của ba tối ưu chỉ được lập luận qua phân tích và qua kết quả tổng. Những gì paper nêu cụ thể:

- Causal mask bỏ qua khối cho speedup ~1.7-1.8x so với không mask.
- Block size lớn hơn giảm truy cập shared memory nhưng tốn register; quá ngưỡng thì chậm hoặc không chạy.
- FLOPs không phải matmul chiếm tỉ lệ nhỏ nhưng tốn thời gian nhiều vì thông lượng ~16 lần thấp hơn.

Ta nhận xét (không phải paper): độ lệch giữa speedup do thuật toán (giảm non-matmul) và do lập lịch (song song hoá, chia warp) không được tách riêng, nên khó biết yếu tố nào quan trọng nhất cho từng cấu hình.

## 6. Giới hạn

Do tác giả nêu:
- Mới tối ưu cho A100; trên H100 chưa dùng lệnh mới (TMA, Tensor Core thế hệ 4, FP8), kỳ vọng còn 1.5-2x nữa.
- Block size phải chỉnh tay cho từng head dim; mong muốn auto-tuning.
- Cần hợp tác với compiler để dễ lập trình, kết hợp với attention thưa/cục bộ cho ngữ cảnh rất dài.

Do ta nhận thấy:
- Toàn bộ benchmark là chuỗi dài, nhiều token cùng lúc (huấn luyện/prefill). Không có số liệu về decode với query dài 1, mà đó mới là chế độ chiếm phần lớn thời gian ở suy luận batch 1.
- Chỉ so sánh trên NVIDIA GPU datacenter (A100/H100); GPU tiêu dùng ≤ 8 GB (kiến trúc khác, SRAM và băng thông khác) không được đo.
- Mô hình chỉ GPT-style kích thước 1.3B-2.7B trong thí nghiệm end-to-end.

## 7. Liên hệ với các bài khác

- **44 (FlashAttention):** cùng thuật toán; FA2 chỉ đổi cách chạy trên GPU. Công thức IO không đổi.
- **14, 15 (MQA, GQA):** FA2 hỗ trợ trực tiếp bằng chỉ số head; hữu ích khi GQA giảm KV cache.
- **46, 47 (vLLM, SGLang):** các hệ thống phục vụ gọi kernel FlashAttention (hoặc FlashInfer) cho phần prefill; PagedAttention có kernel riêng cho decode.
- **48 (speculative decoding):** verify $\gamma+1$ token cùng lúc biến decode từ query dài 1 thành query dài $\gamma+1$, gần với chế độ mà FA2 tối ưu hơn một chút.
- **20, 23, 24 (GLA, DeltaNet, Gated DeltaNet):** các kernel chunk-wise của họ cùng tinh thần "tiling + giữ trạng thái trên chip + song song theo chiều chuỗi". Không phải trích dẫn của paper; đây là điểm tương đồng ta nhận ra.

## 8. Ứng dụng cho dự án (suy luận)

Bối cảnh: LightOnOCR-3-0.8B, giải mã ~1,500 token/trang, batch 1, GPU ≤ 8 GB. Giả định để tính: head dim 128, 16 query head, 4 KV head (GQA), 24 layer decoder trong đó ~6 là full attention (1 trong 4, kiểu Qwen3.5), cùng chiều dài ngữ cảnh $\approx2500$ (prompt gồm token ảnh + output).

1. **Prefill (xử lý ảnh + prompt) hưởng lợi, nhưng không nghẽn.** Như tính ở Mục 3.5, attention của 6 layer chỉ cỡ ~154 GFLOP so với ~4 TFLOP phần tuyến tính (tự tính). FA2 vs FA1 sẽ không thay đổi latency tổng thể đáng kể ở mô hình này. ViT (chuỗi patch dài, attention hai chiều, nhiều head) là nơi chọn backend flash quan trọng hơn.
2. **Decode batch 1 là chế độ chi phối, và FA2 không phải đòn bẩy ở đây.** Mỗi bước sinh 1 token: tải toàn bộ trọng số. Ví dụ tự tính: 0.8B tham số bf16 = 1.6 GB; nếu băng thông bộ nhớ GPU là 272 GB/s (giả định, cỡ GPU 8 GB tầm trung), mỗi bước ≥ 5.9 ms, 1,500 token ≥ ~8.8 s. KV cache của 6 layer full attention với 2500 token chỉ khoảng 29 MiB (4 KV head × 128 × 2 (K,V) × 2 B × 6 layer × 2500, tự tính), nhỏ hơn trọng số khoảng 50 lần. Do đó tối ưu kernel attention gần như không thay đổi thời gian decode; thứ quyết định là **số byte trọng số** (pruning, lượng tử hoá) và **số lần chạy mô hình** (speculative decoding, bài 48).
3. **Kích thước head và pruning.** FA2 chỉnh block size cho từng head dim (chỉ ~4 lựa chọn). Nếu pruning thay đổi head dim sang giá trị không chuẩn, có nguy cơ rơi vào đường chạy chậm hoặc không được hỗ trợ; cần kiểm tra danh sách head dim backend hỗ trợ trước khi cắt. Cắt số head thì giảm số thread block (xem Mục 3.5 về occupancy), nhưng ở decode batch 1 điều này không đáng kể so với đọc trọng số.
4. **Với lớp GatedDeltaNet:** không dùng kernel này; FA2 chỉ liên quan 1/4 số layer (giả định). Nếu pruning cắt bớt layer full attention, phần việc của FA2 giảm tương ứng.

## 9. Thí nghiệm gợi ý

1. **Profile decode:** dùng `torch.profiler` đo thời gian một bước decode của LightOnOCR-0.8B (batch 1), tách: GEMM trọng số, attention (full), GatedDeltaNet, sampling, overhead launch. Kiểm chứng lập luận rằng attention kernel chiếm phần nhỏ.
2. **So backend prefill:** chạy prefill ~2,500 token với SDPA backend "math", "mem-efficient", "flash" (nếu có hỗ trợ), đo latency và peak memory; xem FA2-style kernel có thực sự được chọn trên GPU đích không.
3. **Head dim "lạ":** tạo layer attention giả với head dim 64/96/128 và 80, đo thông lượng của SDPA để biết ranh giới "đường nhanh" của phần cứng đích.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| Output không chuẩn hoá | $\tilde O^{(j)}=\mathrm{diag}(e^{m^{(j-1)}-m^{(j)}})\tilde O^{(j-1)}+e^{S^{(j)}-m^{(j)}}V^{(j)}$ |
| Chuẩn hoá một lần | $O=\mathrm{diag}(\ell^{(last)})^{-1}\tilde O^{(last)}$ |
| Logsumexp | $L=m+\log\ell$, $P=\exp(S-L)$ |
| FLOPs forward | $4\,N^2\,d\,H$ (chia 2 nếu causal) |
| FLOPs backward | $2.5\times$ forward |
| FLOPs huấn luyện | $6\,N\,\#\text{params}+12\,\#\text{layer}\cdot\text{hidden}\cdot N^2$ |
| Matmul vs non-matmul (A100) | 312 vs 19.5 TFLOPs/s, ~16x |
| Song song | FA1: batch×head; FA2: thêm chiều độ dài chuỗi |
| Warp | FA1: split-K (chia $K,V$); FA2: chia $Q$ |
| Backward song song | theo khối cột, atomic add vào $dQ$ |

## 11. Kết luận ngắn

FlashAttention-2 không đổi thuật toán toán học mà đổi cách ánh xạ vào GPU: ít phép non-matmul hơn, song song cả theo chiều chuỗi, chia $Q$ giữa warp. Kết quả ~2x so với v1, đạt tới 73% đỉnh trên A100 và 225 TFLOPs/s huấn luyện end-to-end. Với dự án batch-1 decode, bài này giúp hiểu vì sao attention kernel không phải điểm nghẽn: chi phí nằm ở đọc trọng số, nên đòn bẩy nằm ở pruning/lượng tử hoá và giảm số bước chạy mô hình.

Link gốc: https://arxiv.org/abs/2307.08691
