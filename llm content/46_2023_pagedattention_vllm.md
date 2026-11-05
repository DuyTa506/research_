# Efficient Memory Management for Large Language Model Serving with PagedAttention (2023)

**Tác giả:** Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica.  
**Venue / arXiv:** arXiv:2309.06180 (v1, 12/09/2023); SOSP '23 (theo trang bản quyền trong paper).  
**Nguồn:** https://arxiv.org/abs/2309.06180  
**Chủ đề:** Quản lý bộ nhớ KV cache khi phục vụ LLM nhiều request (serving); PagedAttention; vLLM; continuous batching; chia sẻ KV cache (parallel sampling, beam search, shared prefix).  
**Vị trí trong lộ trình:** Đọc trước: 01 (self-attention, KV cache), 44-45 (FlashAttention: tính attention nhanh), 14-16 (MQA/GQA/MLA: giảm kích thước KV cache). Bài này thêm: cách **lưu và cấp phát** KV cache để nhồi được nhiều request hơn vào GPU. Đọc tiếp: 47 (SGLang/RadixAttention, tái sử dụng KV theo tiền tố), 55-56 (StreamingLLM, H2O: nén/loại bỏ KV cache).

> Đây là ghi chú học tập bằng tiếng Việt, diễn giải lại chứ không phải bản dịch. Số liệu lấy từ paper; nhiều biểu đồ bị mất nhãn khi trích PDF nên chỉ dùng số xuất hiện trong chữ. Phần "Ứng dụng cho dự án" và các số "tự tính" là suy luận của chúng ta. Paper dài vừa (~13k từ); ghi chú tập trung vào cơ chế cốt lõi và kết quả chính.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
   - 3.1 [KV cache và công thức kích thước](#31-kv-cache-và-công-thức-kích-thước)
   - 3.2 [Ba nguồn lãng phí bộ nhớ](#32-ba-nguồn-lãng-phí-bộ-nhớ)
   - 3.3 [PagedAttention](#33-pagedattention)
   - 3.4 [KV cache manager và block table](#34-kv-cache-manager-và-block-table)
   - 3.5 [Chia sẻ: parallel sampling, beam search, shared prefix](#35-chia-sẻ-parallel-sampling-beam-search-shared-prefix)
   - 3.6 [Lập lịch và preemption](#36-lập-lịch-và-preemption)
   - 3.7 [Kernel, phân tán, cài đặt](#37-kernel-phân-tán-cài-đặt)
   - 3.8 [Phân tích chi phí](#38-phân-tích-chi-phí)
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

Phục vụ LLM đắt: paper trích ước tính một request LLM có thể đắt hơn 10 lần một truy vấn từ khoá. Để giảm chi phí/request phải **gom nhiều request vào một batch** (batching), vì mọi request dùng chung trọng số, chi phí đọc trọng số được chia cho cả batch.

Gloss các ý nền:

- **Hai pha sinh văn bản.** *Prompt (prefill):* xử lý toàn bộ prompt song song bằng phép nhân ma trận-ma trận, dùng GPU hiệu quả. *Autoregressive generation (decode):* sinh từng token một, mỗi bước chỉ có 1 token vào, dùng nhân ma trận-vector, bị nghẽn bởi bộ nhớ (memory-bound), chiếm phần lớn latency của một request.
- **KV cache:** các vector key $k_i$ và value $v_i$ của mọi token đã xử lý, lưu lại để các bước sau không phải tính lại. Dùng cho attention $a_{ij}=\frac{\exp(q_i^\top k_j/\sqrt d)}{\sum_t\exp(q_i^\top k_t/\sqrt d)},\ o_i=\sum_j a_{ij}v_j$.
- **Continuous (iteration-level) batching:** thay vì chờ cả batch xong, sau **mỗi bước** đưa request xong ra và nhận request mới vào (Orca, cellular batching).

Vấn đề: số request đưa được vào batch bị **giới hạn bởi bộ nhớ chứa KV cache**, tức thông lượng bị nghẽn bởi bộ nhớ. Ví dụ paper (Hình 1): mô hình 13B trên A100 40GB, khoảng 65% bộ nhớ là trọng số, gần 30% là KV cache, phần còn lại là activation.

Các hệ thống trước (FasterTransformer, Orca) lưu KV cache của mỗi request trong **một khối liền mạch** (vì framework đòi tensor liền mạch) với kích thước **tối đa** có thể (ví dụ 2048 token), dù độ dài thật chưa biết. Kết quả: chỉ 20.4%-38.2% bộ nhớ KV chứa token thật (Hình 2 của paper).

## 2. Ý tưởng chính

Mượn **bộ nhớ ảo + phân trang (paging)** của hệ điều hành: chia KV cache của mỗi chuỗi thành **khối (block) kích thước cố định** (ví dụ 16 token). Các khối **không cần nằm liền nhau** trong bộ nhớ vật lý; một **block table** ánh xạ khối logic của chuỗi sang khối vật lý, giống bảng trang. Khối được cấp phát **theo nhu cầu**, nên lãng phí tối đa dưới một khối cho mỗi chuỗi. Vì khối giữ nguyên kích thước nên hết phân mảnh ngoài, và vì ánh xạ là gián tiếp nên nhiều chuỗi có thể **trỏ cùng một khối vật lý** (chia sẻ prompt, beam search, tiền tố chung) với cơ chế **copy-on-write**. PagedAttention là kernel attention đọc KV qua block table; vLLM là hệ thống phục vụ xây trên đó. Kết quả: thông lượng cao hơn 2-4x ở cùng mức latency, không đổi đầu ra.

## 3. Phương pháp chi tiết

### 3.1 KV cache và công thức kích thước

Tổng quát, KV cache cho một token:

$$\boxed{\text{bytes}_{\text{token}}=2\ (K,V)\times n_{kv}\,d_{head}\times L_{\text{attn}}\times \text{bytes per element}}$$

Ví dụ của paper (OPT-13B): $2\times5120\ (\text{hidden})\times40\ (\text{layer})\times2\ (\text{fp16})=800$ KB mỗi token. Một request dài tối đa 2048 token cần tới 1.6 GB. Bộ nhớ GPU chỉ vài chục GB nên chỉ nhồi được vài chục request. Paper lưu ý tốc độ tính toán GPU tăng nhanh hơn dung lượng bộ nhớ (A100 sang H100: FLOPS tăng hơn 2x nhưng bộ nhớ vẫn tối đa 80 GB), nên bộ nhớ ngày càng là nút thắt.

### 3.2 Ba nguồn lãng phí bộ nhớ

Khi cấp phát trước một khối liền mạch theo độ dài tối đa:

1. **Reserved (dự trữ):** chỗ cho token tương lai, sẽ được dùng nhưng chiếm chỗ suốt đời request, các request khác không dùng được.
2. **Internal fragmentation (phân mảnh trong):** cấp phát theo mức tối đa nhưng độ dài thật ngắn hơn; phần thừa không bao giờ dùng.
3. **External fragmentation (phân mảnh ngoài):** khoảng trống giữa các khối do bộ cấp phát (ví dụ buddy allocator) để lại.

Nén lại (compaction) không khả thi vì KV cache quá lớn. Ngoài ra, khối liền mạch riêng cho từng chuỗi ngăn việc chia sẻ KV giữa các chuỗi trong cùng một request (parallel sampling, beam search).

### 3.3 PagedAttention

Chia KV của chuỗi thành khối $B$ token: $K_j=(k_{(j-1)B+1},\dots,k_{jB})$, $V_j$ tương tự. Attention viết lại theo khối:

$$A_{ij}=\frac{\exp(q_i^\top K_j/\sqrt d)}{\sum_{t=1}^{\lceil i/B\rceil}\exp(q_i^\top K_t\mathbf 1/\sqrt d)},\qquad o_i=\sum_{j=1}^{\lceil i/B\rceil}V_jA_{ij}^\top,$$

với $A_{ij}$ là vector hàng điểm attention trên khối $j$. Về mặt toán học **giống hệt** attention thường; điểm khác là kernel lấy từng khối key/value từ địa chỉ vật lý bất kỳ (qua block table), tính điểm với khối đó, rồi nhân với khối value. Ví dụ Hình 5: truy vấn "forth" đọc 3 khối nằm rải rác trong bộ nhớ.

Paper không nói rõ cách gộp softmax giữa các khối trong kernel (xem Giới hạn); về mặt thực hành nó cùng kiểu online softmax như bài 44, nhưng điều đó ta suy ra chứ paper không trình bày.

### 3.4 KV cache manager và block table

- Khối **logic** của một chuỗi được lấp từ trái sang phải; khối cuối còn chỗ trống dự trữ cho token sau.
- Trên GPU worker, *block engine* cấp một vùng DRAM liền mạch lớn và chia thành **khối vật lý** (cũng có vùng tương tự trên RAM CPU để swap).
- **Block table** cho mỗi chuỗi: danh sách (khối vật lý, số vị trí đã đầy).

Ví dụ (Hình 6): prompt 7 token, $B=4$. Khối logic 0 và 1 ánh xạ tới khối vật lý 7 và 1; khối 1 có 3/4 chỗ đầy. Bước decode đầu tiên ghi token mới vào chỗ trống còn lại. Bước tiếp theo khối logic 1 đã đầy, nên cấp phát khối vật lý mới (khối 3) và thêm một dòng vào bảng.

Thuật toán mỗi vòng lặp (giả mã theo mô tả §4.3 của paper):

```text
mỗi iteration:
    chọn tập chuỗi ứng viên để batch (FCFS)       # §4.5
    với mỗi chuỗi cần khối logic mới:
        cấp phát khối vật lý, thêm vào block table
        nếu hết khối: preempt chuỗi đến sau cùng (swap hoặc recompute)
    ghép token đầu vào của mọi chuỗi (prompt và decode) thành một "chuỗi" duy nhất
    chạy mô hình; kernel PagedAttention đọc KV theo block table
    ghi KV mới vào khối vật lý đúng chỗ
    chuỗi xong -> giải phóng khối
```

Lãng phí chỉ xảy ra ở **khối cuối** của mỗi chuỗi, nên nhỏ hơn 1 khối/chuỗi. Paper báo cáo gần như không lãng phí (Hình 2 của paper cho vLLM; vì nhãn biểu đồ bị lẫn khi trích, ta chỉ khẳng định "gần 0" như tác giả nói).

### 3.5 Chia sẻ: parallel sampling, beam search, shared prefix

- **Parallel sampling** (nhiều mẫu từ một prompt): các mẫu trỏ chung khối vật lý của prompt; mỗi khối vật lý có **reference count**. Khi một mẫu cần ghi vào khối có count > 1 thì **copy-on-write**: cấp khối mới, sao chép nội dung, giảm count ở khối cũ. Tất cả khối của prompt được chia sẻ, trừ khối cuối (CoW).
- **Beam search:** các beam chia sẻ cả phần prompt lẫn phần sinh ra ban đầu; nhánh bị loại thì giải phóng khối khi count về 0. Hệ thống tránh sao chép KV cache như các cài đặt cũ (phải copy cho mỗi beam), chỉ copy một khối khi cần.
- **Shared prefix:** dịch vụ có thể đặt trước KV của tiền tố chung (ví dụ ví dụ few-shot, system prompt); request mới ánh xạ khối logic vào các khối đã cache (khối cuối đánh dấu CoW). Chỉ phần đầu vào riêng của người dùng phải tính prefill. Đây là tiền thân của RadixAttention (bài 47), nhưng ở vLLM phải khai báo tiền tố trước.
- **Mixed decoding:** lớp ánh xạ ẩn mô hình sharing, nên các request có kiểu giải mã khác nhau vẫn batch chung được. Các hàm cài đặt: `fork`, `append`, `free`.

### 3.6 Lập lịch và preemption

FCFS, preempt chuỗi đến sau cùng khi hết khối GPU. Hai câu hỏi cổ điển:

- **Loại khối nào?** Vì mọi khối của một chuỗi được truy cập cùng nhau, vLLM dùng **all-or-nothing**: loại hết hoặc không loại; các chuỗi cùng một request (ví dụ các beam) là một *sequence group*, được preempt/lên lịch lại cùng nhau.
- **Khôi phục thế nào?** (a) *Swapping:* chép khối sang RAM CPU (số khối swap không vượt tổng số khối GPU). (b) *Recomputation:* tính lại KV khi chuỗi được lên lịch lại, bằng cách ghép token đã sinh vào prompt và chạy một lượt prefill; không có tương đương trong OS.

### 3.7 Kernel, phân tán, cài đặt

- Ba kernel hợp nhất: (1) ghi KV mới vào khối (reshape + ghi), (2) đọc khối và attention (dựa trên kernel FasterTransformer, mỗi warp đọc một khối để truy cập liên tục), (3) sao chép khối hàng loạt cho CoW (một kernel thay cho nhiều `cudaMemcpyAsync`).
- Phân tán kiểu Megatron-LM tensor parallel: mọi shard dùng chung một KV manager và cùng block table; mỗi worker chỉ lưu KV cho các head của mình.
- Cài đặt: 8.5K dòng Python + 2K dòng C++/CUDA; hỗ trợ GPT, OPT, LLaMA.

### 3.8 Phân tích chi phí

**Lãng phí tối đa theo định nghĩa:** với khối $B$ token, mỗi chuỗi lãng phí tối đa $B-1$ vị trí (khối cuối), tức $\le(B-1)\times\text{bytes}_{\text{token}}$. Cấp phát trước kiểu cũ lãng phí tới $L_{\max}-L_{\text{thật}}$ vị trí.

**Ví dụ (OPT-13B của paper, tự tính từ số trong paper):** $B=16$: lãng phí tối đa $15\times800$ KB $\approx12$ MB/chuỗi; cấp phát tối đa 2048 token là 1.6 GB/chuỗi. Với chuỗi thật 300 token: kiểu cũ lãng phí $\approx1748\times0.8$ MB $\approx1.4$ GB so với $<12$ MB.

**Chi phí của chỉ mục gián tiếp:** kernel PagedAttention có thêm truy cập block table, nhánh điều kiện và xử lý độ dài biến thiên: paper đo **kernel attention chậm hơn 20-26%** so với FasterTransformer, nhưng chỉ ảnh hưởng toán tử attention nên tổng thể vẫn thắng.

**Ví dụ cho mô hình của ta (tự tính, giả định).** Giả định 24 layer decoder, 6 layer là full attention (1 trong 4, kiểu Qwen3.5), 4 KV head, head dim 128, bf16, chiều dài ngữ cảnh 2500: $2\times4\times128\times2\ \text{B}=2048$ B mỗi token mỗi layer; $\times6=12$ KiB mỗi token; $\times2500=\mathbf{\approx29\ MiB}$ mỗi chuỗi. Nếu cả 24 layer là full attention: ~117 MiB. So với trọng số bf16 1.6 GB: KV chỉ khoảng 2% (lai) hoặc 7% (toàn full). Các lớp GatedDeltaNet giữ một **trạng thái hồi quy kích thước cố định** không phụ thuộc độ dài, nên chúng không phân mảnh theo độ dài và không cần phân trang theo token (kích thước đúng phụ thuộc cấu hình; ta không có số thật).

## 4. Thí nghiệm và kết quả chính

**Thiết lập:** OPT 13B/66B/175B và LLaMA-13B; GPU A100 trên Google Cloud; workload tổng hợp từ ShareGPT (đầu vào dài hơn 8.4x và đầu ra dài hơn 5.8x so với Alpaca) và Alpaca, đến theo phân phối Poisson. Chỉ số: độ trễ chuẩn hoá (end-to-end latency chia độ dài đầu ra, s/token); hệ thống tốt là hệ thống giữ độ trễ thấp ở request rate cao. Baseline: FasterTransformer (có bộ lập lịch dynamic batching tự viết) và ba biến thể Orca (Oracle biết trước độ dài đầu ra; Pow2 dự trữ làm tròn lên luỹ thừa của 2; Max dự trữ 2048).

Cấu hình (Bảng 1):

| | 13B | 66B | 175B |
|---|---|---|---|
| GPU | A100 | 4×A100 | 8×A100-80GB |
| Bộ nhớ GPU tổng | 40 GB | 160 GB | 640 GB |
| Tham số | 26 GB | 132 GB | 346 GB |
| Bộ nhớ cho KV | 12 GB | 21 GB | 264 GB |
| Số slot KV tối đa | 15.7K | 9.7K | 60.1K |

**Kết quả chính (từ chữ của paper):**

| Nội dung | Kết quả |
|---|---|
| ShareGPT, request rate duy trì được | 1.7-2.7x so với Orca (Oracle); 2.7-8x so với Orca (Max) |
| So với FasterTransformer | tới 22x |
| OPT-13B ShareGPT, số request chạy cùng lúc | 2.2x so với Orca (Oracle); 4.3x so với Orca (Max) |
| Parallel sampling / beam search (OPT-13B, Alpaca) | vượt Orca (Oracle) từ 1.3x (lấy mẫu cơ bản) tới 2.3x (beam search, độ rộng 6) |
| Tiết kiệm bộ nhớ nhờ chia sẻ, Alpaca | parallel sampling 6.1%-9.8%; beam search 37.6%-55.2% |
| Tiết kiệm bộ nhớ nhờ chia sẻ, ShareGPT | parallel sampling 16.2%-30.5%; beam search 44.3%-66.3% |
| Shared prefix (LLaMA-13B, dịch WMT16 En-De) | 1.67x (1-shot, 80 token) và 3.58x (5-shot, 341 token) so với Orca (Oracle) |
| Chatbot (OPT-13B, cắt prompt 1024 token) | duy trì request rate cao gấp 2x so với ba Orca |

Lưu ý: với Alpaca + OPT-175B, lợi thế của vLLM nhỏ hơn vì cấu hình có rất nhiều bộ nhớ KV mà chuỗi lại ngắn, nên hệ thống thiên về bị nghẽn bởi tính toán (compute-bound), không phải bộ nhớ.

## 5. Phân tích / ablation

- **Kernel (7.1):** chỉ mục gián tiếp làm kernel attention chậm hơn 20-26%, không ảnh hưởng toán tử khác như Linear.
- **Block size (7.2):** quá nhỏ thì không dùng hết song song của GPU khi đọc KV; quá lớn thì phân mảnh trong tăng và xác suất chia sẻ giảm. ShareGPT: 16-128 là tốt nhất; Alpaca: 16 và 32 tốt, lớn hơn kém đi vì chuỗi ngắn hơn khối. **Mặc định 16.**
- **Swap vs recompute (7.3):** swap tốn kém khi khối nhỏ (nhiều lần chép nhỏ qua PCIe); recompute có chi phí không phụ thuộc kích thước khối và không bao giờ vượt 20% độ trễ của swap. Khối cỡ vừa 16-64: hai cách tương đương về hiệu năng end-to-end.
- Paper không có ablation riêng cho từng thành phần của vLLM (ví dụ tắt chia sẻ) ngoài các so sánh trên.

## 6. Giới hạn

Do tác giả nêu:
- Kỹ thuật phù hợp khi bộ nhớ là nút thắt và cấp phát động, như serving LLM; **không phù hợp** huấn luyện (hình dạng tensor tĩnh) hay phục vụ DNN không phải LLM (compute-bound, chỉ thêm overhead gián tiếp).
- Kernel bị overhead 20-26%.

Do ta nhận thấy:
- Lợi ích đo ở chế độ **thông lượng với nhiều request đồng thời**; tác động lên latency của **một request độc lập** (batch 1) không được đo và về nguyên tắc gần bằng 0 (thậm chí âm do overhead kernel).
- Mô hình thí nghiệm là OPT/LLaMA full attention, KV cache lớn. Không xét kiến trúc hồi quy/lai có trạng thái cố định.
- FCFS có thể bất lợi cho request dài; không có phân tích công bằng.
- Paper không mô tả chi tiết cách kernel kết hợp softmax qua các khối (công thức viết theo mẫu số tổng quát).

## 7. Liên hệ với các bài khác

- **44, 45 (FlashAttention):** FlashAttention tối ưu *cách tính* (IO) khi KV liền mạch; PagedAttention tối ưu *cách lưu*. Chúng bổ sung nhau; thực tế vLLM dùng kernel khác nhau cho prefill và decode.
- **14, 15, 16 (MQA, GQA, MLA):** giảm $n_{kv}$ trong công thức kích thước, tức giảm KV theo token; PagedAttention giảm lãng phí theo cấp phát. Hai trục độc lập, nhân vào nhau.
- **47 (SGLang):** RadixAttention là bước tổng quát hoá của "shared prefix" ở đây thành cây cơ số tự động với LRU; SGLang xây trên bộ nhớ phân trang kiểu này (kích thước trang 1 token).
- **18-24, 26 (attention tuyến tính, hồi quy):** trạng thái cố định kích thước, không cần phân trang theo độ dài, nhưng khó chia sẻ tiền tố.
- **55, 56 (StreamingLLM, H2O):** giảm số token cần giữ; ghép với cấp phát theo khối.
- **48 (speculative decoding):** các token nháp bị từ chối phải thu hồi KV; với quản lý theo khối điều này là giải phóng/ghi đè khối.

## 8. Ứng dụng cho dự án (suy luận)

Mục tiêu: LightOnOCR-3-0.8B, batch 1, GPU ≤ 8 GB, ~1,500 token/trang.

1. **Ít liên quan cho batch-1.** PagedAttention giải quyết bài toán "nhồi nhiều request vào GPU". Ở batch 1 chỉ có một chuỗi, không có phân mảnh giữa các request, nên không có gì để thắng, thậm chí kernel chậm hơn 20-26% so với kernel liền mạch. Một cấp phát tĩnh trước cho cả ngữ cảnh tối đa là đủ và đơn giản (ví dụ 8,192 token tối đa: ~96 MiB với giả định ở Mục 3.8, tự tính).
2. **Nếu sau này phục vụ nhiều trang đồng thời (offline batch OCR):** tính nhanh (tự tính): sau 1.6 GB trọng số và dư ~1.5 GB cho activation/ViT/overhead (giả định), còn khoảng 4.9 GB cho KV. Với 29 MiB/chuỗi (lai) có thể chứa cỡ 160 chuỗi; với toàn full attention 117 MiB/chuỗi chỉ cỡ 40 chuỗi. Điểm này cho thấy **kiến trúc lai (ít lớp full attention) đã tự giải quyết phần lớn vấn đề KV cache**, giống tác dụng của GQA; còn phân trang chỉ là lớp tối ưu thứ hai.
3. **Điểm cần chú ý khi pruning:** cắt bớt số KV head hoặc layer full attention giảm trực tiếp công thức $2\,n_{kv}d_{head}L_{attn}$. Nhưng với ≤ 0.3B tham số, trọng số chỉ ~0.6 GB, KV vài chục MiB: bộ nhớ không còn là vấn đề trên 8 GB. Giới hạn là băng thông đọc trọng số và overhead launch kernel, không phải dung lượng.
4. **Mô hình lai:** vLLM cần lưu thêm trạng thái hồi quy của các lớp GatedDeltaNet theo từng chuỗi; paper không nói gì điều này (không xét kiến trúc đó). Việc tái sử dụng tiền tố với trạng thái hồi quy khó hơn KV vì trạng thái là tóm tắt nén của cả tiền tố, không cắt được theo token. Đây là suy luận của ta, cần kiểm tra trong khung phục vụ cụ thể.
5. **Ảnh trang là "tiền tố" khác nhau cho mỗi request:** shared prefix chỉ có phần chỉ dẫn văn bản ngắn nên ít có lợi (xem 47).

## 9. Thí nghiệm gợi ý

1. **Đo tỉ lệ KV/trọng số thật:** in `torch.cuda.memory_allocated()` sau nạp mô hình, sau prefill 2,500 token, sau 1,500 token decode; đối chiếu với công thức $2\,n_{kv}d_{head}L_{attn}$ để kiểm chứng giả định về số KV head và số layer full attention.
2. **Tìm điểm bão hoà batch:** chạy decode với batch 1, 2, 4, 8, 16 (cùng ngữ cảnh), đo token/s tổng và theo từng request. Biết ở batch nào throughput bắt đầu bị giới hạn bởi bộ nhớ KV thay vì bởi đọc trọng số, từ đó quyết định có cần framework phân trang không.
3. **Mô phỏng phân mảnh:** với phân phối độ dài đầu ra của OCR (lấy từ tập dữ liệu thật), tính phần trăm bộ nhớ lãng phí nếu cấp phát tối đa (ví dụ 4,096) so với cấp phát theo khối 16.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa / công thức |
|---|---|
| KV cache theo token | $2\,n_{kv}d_{head}L_{attn}\cdot\text{bytes}$ (OPT-13B: 800 KB) |
| Block (trang) | $B$ token, mặc định 16 |
| Block table | ánh xạ khối logic → khối vật lý + số vị trí đã đầy |
| Lãng phí tối đa | $<B$ vị trí mỗi chuỗi |
| Attention theo khối | $o_i=\sum_j V_jA_{ij}^\top$, $A_{ij}\propto\exp(q_i^\top K_j/\sqrt d)$ |
| Copy-on-write | đếm tham chiếu; ghi vào khối có count > 1 thì sao chép |
| Sequence group | các chuỗi cùng request, preempt cùng nhau |
| Swap vs recompute | chép sang CPU vs tính lại bằng một lượt prefill |
| Overhead kernel | +20-26% thời gian attention |
| Kết quả chính | thông lượng 2-4x cùng độ trễ |

## 11. Kết luận ngắn

PagedAttention đưa ý tưởng bộ nhớ ảo vào KV cache: khối kích thước cố định, cấp phát theo nhu cầu, ánh xạ gián tiếp, chia sẻ copy-on-write. Nhờ đó gần như không lãng phí bộ nhớ, batch lớn hơn, thông lượng tăng 2-4x. Với dự án batch-1 trên GPU 8 GB, bài này chủ yếu cung cấp công thức kích thước KV cache và lập luận vì sao kiến trúc lai đã nhỏ gọn; bản thân PagedAttention là tối ưu cho phục vụ nhiều request, ít liên quan khi chỉ có một chuỗi.

Link gốc: https://arxiv.org/abs/2309.06180
