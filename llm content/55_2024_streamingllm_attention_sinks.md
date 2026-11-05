# Efficient Streaming Language Models with Attention Sinks (2024)

**Tác giả:** Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis (MIT, Meta AI, CMU, NVIDIA).  
**Venue / arXiv:** arXiv:2309.17453 (v4); ICLR 2024 (theo trang đầu của paper).  
**Nguồn:** https://arxiv.org/abs/2309.17453  
**Chủ đề:** Hiện tượng "attention sink", cache KV dạng cuộn (rolling) giữ vài token đầu + cửa sổ gần nhất, để LLM sinh văn bản vô hạn mà bộ nhớ không đổi, không cần fine-tune.  
**Vị trí trong lộ trình:** Đọc trước: 01 (Attention), 12 (RoPE), 13 (ALiBi), 17 (Longformer, sliding window), 44 (FlashAttention). Bài này giải thích tại sao sliding window thuần túy thất bại và sửa bằng 4 token. Tiếp theo: 56 (H2O, giữ token "nặng" theo điểm attention; so sánh trực tiếp với bài này), 57 (FastV, tỉa token thị giác).

> Đây là ghi chú học tập bằng tiếng Việt, không phải bản dịch. Ứng dụng cho dự án và các số "tự tính" là suy luận của chúng ta.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
4. [Phân tích chi phí: bộ nhớ và độ trễ](#4-phân-tích-chi-phí-bộ-nhớ-và-độ-trễ)
5. [Thí nghiệm và kết quả chính](#5-thí-nghiệm-và-kết-quả-chính)
6. [Phân tích / ablation](#6-phân-tích--ablation)
7. [Giới hạn](#7-giới-hạn)
8. [Liên hệ với các bài khác](#8-liên-hệ-với-các-bài-khác)
9. [Ứng dụng cho dự án (suy luận)](#9-ứng-dụng-cho-dự-án-suy-luận)
10. [Thí nghiệm gợi ý](#10-thí-nghiệm-gợi-ý)
11. [Tóm tắt công thức / khái niệm](#11-tóm-tắt-công-thức--khái-niệm)
12. [Kết luận ngắn](#12-kết-luận-ngắn)

---

## 1. Bối cảnh

**KV cache là gì (nhắc lại cho người mới).** Khi LLM sinh từng token, mỗi lớp attention cần Key và Value của mọi token trước đó. Để khỏi tính lại, ta lưu chúng lại: đó là KV cache. Kích thước của nó tăng tuyến tính theo độ dài chuỗi và theo batch, và mỗi bước decode phải đọc lại toàn bộ.

**Bài toán "streaming".** Ứng dụng chạy liên tục (trợ lý nhiều lượt hội thoại cả ngày) sẽ có đầu vào gần như vô hạn. Hai khó khăn (paper nêu):
1. Cache KV phình ra: tốn bộ nhớ và làm chậm decode.
2. Mô hình **không ngoại suy độ dài**: hỏng khi vượt cửa sổ pre-train (ví dụ 4K của Llama-2), kể cả khi dùng RoPE hay ALiBi.

**Các phương án trước đó (Hình 1):**
- (a) Dense attention: giữ tất cả, độ phức tạp tổng $O(T^2)$, cache tăng mãi, hỏng khi vượt độ dài pre-train.
- (b) **Window attention** (như Longformer, bài 17): chỉ giữ $L$ token gần nhất. Bộ nhớ không đổi, nhưng perplexity **nổ tung** ngay khi token đầu bị loại.
- (c) Sliding window với **tính lại** (re-computation): mỗi token mới dựng lại KV cho $L$ token gần nhất. Chất lượng tốt nhưng $O(TL^2)$, rất chậm.

Câu hỏi của paper: có thể deploy LLM với đầu vào vô hạn mà không hy sinh hiệu năng không?

## 2. Ý tưởng chính

1. Quan sát: ở hầu hết các lớp (trừ hai lớp dưới cùng), rất nhiều attention dồn vào **các token đầu tiên**, bất kể chúng có nghĩa hay không. Paper gọi đó là **attention sink** ("hố chìm" hút attention).
2. Lý do: softmax ép tổng attention bằng 1. Khi query không khớp mạnh với token nào, mô hình vẫn phải "đổ" trọng số đi đâu đó. Token đầu nhìn thấy được bởi mọi token sau (do mặt nạ nhân quả), nên nó dễ được huấn luyện thành chỗ đổ này.
3. Khi window attention loại token đầu, mẫu số của softmax mất một phần rất lớn, phân phối attention lệch khỏi lúc huấn luyện, mô hình hỏng.
4. Sửa: **luôn giữ KV của 4 token đầu (sink) + KV của L token gần nhất**. Không fine-tune, dùng kernel chuẩn.
5. Thêm: nếu pre-train với một "sink token" học được ở đầu mọi mẫu, chỉ cần 1 token sink là đủ.

## 3. Phương pháp chi tiết

### 3.1 Vì sao window attention hỏng

Softmax: $\mathrm{SoftMax}(x)_i=\dfrac{e^{x_i}}{e^{x_1}+\sum_{j=2}^{N}e^{x_j}}$ với $x_1\gg x_j$ (token đầu có logit rất lớn). Bỏ token đầu là bỏ phần lớn mẫu số.

Xác nhận bằng quan sát:
- Hình 2 (Llama-2-7B, logit attention trung bình trên 256 câu dài 16): hai lớp 0 và 1 có kiểu "cục bộ"; từ các lớp sau, mô hình dồn attention vào token đầu ở mọi head.
- Phụ lục F: với chuỗi 4096 token, attention của token thứ 4096 vào token đầu "thường vượt quá một nửa tổng attention, trừ hai lớp dưới".

**Do ngữ nghĩa hay do vị trí?** Thay 4 token đầu bằng dấu xuống dòng "\n": mô hình vẫn dồn attention vào đó và khôi phục perplexity (Bảng 1: 4+1020 cho 5.40; 4"\n"+1020 cho 5.60). Kết luận: quan trọng là **vị trí tuyệt đối** của các token bắt đầu, không phải nội dung.

**Vì sao cần tới 4 token thay vì 1?** Các mô hình này khi pre-train không có token bắt đầu cố định ở mọi mẫu. Llama-2 chèn `<s>` trước khi cắt đoạn, nên token ở vị trí 0 phần lớn là token ngẫu nhiên. Do đó mô hình học dùng vài token đầu làm sink (đây là giải thích/giả thuyết của tác giả, được ủng hộ bởi thí nghiệm ở 3.3).

### 3.2 StreamingLLM: rolling KV cache với sink

Cache gồm hai phần (Hình 4):
1. **Attention sinks:** KV của 4 token đầu tiên, giữ vĩnh viễn.
2. **Rolling KV cache:** KV của $L-4$ token gần nhất, cuộn theo thời gian.

**Vị trí được gán theo vị trí trong cache, không theo vị trí trong văn bản gốc.** Ví dụ: cache đang chứa token [0,1,2,3,6,7,8] và giải mã token 9 thì vị trí là [0..7], không phải [0,1,2,3,6,7,8,9]. Đây là điểm then chốt.

- **Với RoPE** (bài 12): lưu Key *trước khi* xoay; ở mỗi bước decode áp phép xoay theo vị trí trong cache lên các Key của cache.
- **Với ALiBi** (bài 13): áp bias tuyến tính liên tục thay vì bias "nhảy cóc".

### 3.3 Sink token khi pre-train

Giả thuyết: nếu có một token sink chuyên dụng, mô hình không cần "mượn" nhiều token đầu. Ba mô hình 160M tham số được pre-train giống hệt, khác nhau ở chỗ:
- Vanilla (softmax thường);
- **Zero Sink:** thay softmax bằng $\mathrm{SoftMax}_1(x)_i=\dfrac{e^{x_i}}{1+\sum_{j}e^{x_j}}$ (softmax-off-by-one; tương đương thêm một token có Key và Value bằng 0);
- **Learnable Sink:** thêm một token placeholder học được vào đầu mọi mẫu.

### 3.4 Pseudo-code

```
state: sink_kv  (KV của 4 token đầu, mỗi lớp)
       window_kv (circular buffer, kích thước L-4)

on each new token t:
    # 1. tính q,k,v cho t (K lưu trước RoPE)
    # 2. cache = concat(sink_kv, window_kv)   # độ dài <= L
    # 3. với mỗi lớp: gán vị trí 0..len(cache) cho cache + 1 cho t
    #    xoay K của cache theo vị trí đó (RoPE)  [hoặc ALiBi bias liên tục]
    # 4. attention(q_t, K_cache+k_t, V_cache+v_t)  -> output
    # 5. nếu len(window_kv) == L-4: bỏ phần tử cũ nhất của window_kv
    #    window_kv.append((k_t, v_t))
```

## 4. Phân tích chi phí: bộ nhớ và độ trễ

### 4.1 Công thức

Gọi $n_\ell$ số lớp full-attention, $n_{kv}$ số KV head, $d_h$ kích thước mỗi head, $b$ số byte mỗi phần tử, $T$ độ dài chuỗi, $L$ kích thước cache.

$$
\boxed{\ \text{KV}(T)=2\,n_\ell\,n_{kv}\,d_h\,b\cdot \min(T,\,L)\ }\quad(\text{StreamingLLM: }\min(T,L)=L)
$$

Độ phức tạp thời gian tổng cộng (paper, Hình 1): Dense $O(T^2)$; window $O(TL)$; re-computation $O(TL^2)$; StreamingLLM $O(TL)$. Mỗi bước decode chỉ đọc $L$ KV nên $O(L)$ mỗi token.

Chi phí thêm: xoay lại $L$ Key bằng RoPE mỗi bước, $O(L\cdot d_h)$ mỗi head, nhỏ so với việc đọc KV.

### 4.2 Ví dụ số

**Llama-2-7B** (kiến trúc: 32 lớp, hidden 4096, MHA, tự nhớ, không từ paper; FP16): KV/token $=2\cdot32\cdot4096\cdot2=524{,}288$ B $\approx 0.5$ MiB (tự tính). Để đạt 4 triệu token (mốc paper thử), dense cần $\approx$ 2 TB. StreamingLLM với cache 4+2044 = 2048 token: $\approx$ 1,07 GB, cố định.

**Độ trễ.** Paper: trên A6000 với Llama-2-7B/13B, StreamingLLM nhanh tới **22,2x** mỗi token so với sliding-window-recompute, và bộ nhớ tương đương baseline đó. Độ trễ decode tăng gần tuyến tính theo kích thước cache (do phải đọc $L$ KV), còn recompute tăng bậc hai.

**Cỡ dự án (tự tính, giả định):** nếu coi 6 lớp full-attention, 2 KV head x 128 dim, FP16 thì 1.024 byte/token/lớp. Với 2.500 token (prompt thị giác + 1.500 đầu ra), KV $\approx$ 2.500 x 1.024 x 6 $\approx$ 15 MB. Nếu cả 24 lớp đều là full-attention thì $\approx$ 61 MB. Cả hai đều vài phần trăm hoặc thấp hơn so với trọng số 1,6 GB (FP16).

### 4.3 Hệ quả ở batch 1

- Decode ở batch 1 là memory-bound; byte đọc mỗi token = trọng số + KV. Trọng số cố định ~1,6 GB (FP16); KV chỉ chiếm ~1% với chuỗi ngắn. StreamingLLM chỉ **chặn trần** phần KV; nó không làm nhanh hơn dense khi $T\le L$ (cache chưa đầy).
- Trong mô hình lai GatedDeltaNet + full attention, **chỉ các lớp full-attention có KV tăng theo độ dài**. Các lớp GatedDeltaNet giữ trạng thái $S\in\mathbb R^{d_k\times d_v}$ mỗi head, kích thước cố định, không phụ thuộc $T$ (đã là "streaming" tự nhiên). Do đó vấn đề "KV phình vô hạn" của paper này chỉ áp lên 1/4 số lớp.

## 5. Thí nghiệm và kết quả chính

**Mô hình:** Llama-2-[7,13,70]B (RoPE), MPT-[7,30]B (ALiBi), Falcon-[7,40]B, Pythia-[2.9,6.9,12]B. Dữ liệu: PG19 (100 cuốn sách ghép lại cho 4 triệu token).

**Hình 1 (Llama-2-13B, cuốn sách đầu PG19, 65K token):** dense PPL 5641; window 5158; sliding window recompute 5.43; StreamingLLM 5.40.

**Bảng 1:** Llama-2-13B, 0+1024 (window) 5158.07; 4+1020 5.40; 4 token "\n" + 1020 5.60.

**Bảng 2 (PPL, số token sink khác nhau, 400K token PG19; cấu hình x+y = x token đầu + y token gần):**

| Mô hình | 0+2048 | 1+2047 | 2+2046 | 4+2044 | 8+2040 |
|---|---|---|---|---|---|
| Falcon-7B | 17.90 | 12.12 | 12.12 | 12.12 | 12.12 |
| MPT-7B | 460.29 | 14.99 | 15.00 | 14.99 | 14.98 |
| Pythia-12B | 21.62 | 11.95 | 12.09 | 12.09 | 12.02 |

| Mô hình | 0+4096 | 1+4095 | 2+4094 | 4+4092 | 8+4088 |
|---|---|---|---|---|---|
| Llama-2-7B | 3359.95 | 11.88 | 10.51 | 9.59 | 9.54 |

Một hoặc hai token đầu chưa đủ cho Llama-2; 4 là ngưỡng đủ, thêm nữa lợi ích giảm dần.

**Bảng 3 (160M, pre-train từ đầu, PPL cuốn sách PG19 đầu tiên):**

| Cache | 0+1024 | 1+1023 | 2+1022 | 4+1020 |
|---|---|---|---|---|
| Vanilla | 27.87 | 18.49 | 18.05 | 18.05 |
| Zero Sink | 29214 | 19.90 | 18.27 | 18.01 |
| Learnable Sink | 1235 | 18.01 | 18.01 | 18.02 |

Mô hình có sink token chỉ cần 1 token sink (1+1023 cho 18.01). Phụ lục I: thêm token sink thứ hai không cải thiện thêm; với 2 sink token, mô hình cần cả hai (1+1023 cho 25.73).

**Bảng 4 (zero-shot, vanilla / có sink token):** ARC-c 18.6 / 19.6; ARC-e 45.2 / 45.6; HellaSwag 29.4 / 29.8; LAMBADA 39.6 / 39.9; OBQA 16.0 / 16.6; PIQA 62.2 / 62.6; Winogrande 50.1 / 50.8. Sink token không làm hại hội tụ hay chất lượng.

**Hỏi đáp streaming (Bảng 5, ARC-E / ARC-C, cache 1024):**

| | Llama-2-7B-Chat | 13B-Chat | 70B-Chat |
|---|---|---|---|
| One-shot (đối chứng) | 71.25 / 53.16 | 78.16 / 63.31 | 91.29 / 78.50 |
| Dense | OOM | OOM | OOM |
| Window | 3.58 / 1.39 | 0.25 / 0.34 | 0.12 / 0.32 |
| StreamingLLM | 71.34 / 55.03 | 80.89 / 65.61 | 91.37 / 80.20 |

**StreamEval (Hình 9, Phụ lục C Bảng 7):** StreamingLLM giữ độ chính xác hợp lý tới ~120K token; truy vấn trả lời được khi khoảng cách truy vấn-đáp án nằm trong cache. Llama-2-7B-32K-Instruct: ở khoảng cách 460 token, cache 4+2044 đạt 85.80%; ở khoảng cách 2.300 token với cache 4+2044 thì 0.00% (vượt cache), cache 4+4092 thì 61.60%.

**LongBench (Phụ lục D, Bảng 8, Llama2-7B-chat):** StreamingLLM 4+3496 thua truncation 1750+1750 (ví dụ NarrativeQA 11.6 vs 18.7); đặt lại 1750+1750 thì ngang (18.2).

## 6. Phân tích / ablation

- **Số token sink:** 4 là đủ (Bảng 2).
- **Kích thước cache (Bảng 6):** tăng cache **không** luôn giảm perplexity, tác giả nói mô hình có thể chưa tận dụng hết ngữ cảnh được cung cấp.
- **Zero Sink (softmax-off-by-one):** giúp một phần nhưng vẫn cần token đầu khác; learnable sink hiệu quả hơn.
- **Attention với sink token (Hình 7):** có sink token thì attention dồn vào nó ở mọi lớp, các token đầu khác bớt bị dùng làm sink.
- **Encoder:** BERT-base có hiện tượng tương tự ở token [SEP]; paper nhắc tới "registers" trong ViT (Darcet và cộng sự) như hiện tượng song song (Phụ lục B, H).

## 7. Giới hạn

**Do tác giả nêu:**
- Không mở rộng cửa sổ ngữ cảnh hay trí nhớ dài hạn: mô hình chỉ dùng được thông tin còn trong cache. Không phù hợp QA tài liệu dài, tóm tắt (Phụ lục A, C, D).
- Tăng cache không tăng chất lượng một cách nhất quán.

**Do chúng ta nhận thấy:**
- Nhiều cách giải thích về nguồn gốc sink (softmax, số lượng token, vị trí) vẫn là giả thuyết, được kiểm bằng thí nghiệm gián tiếp (ví dụ mô hình 160M), chưa là chứng minh.
- Phép thử chủ yếu là perplexity và QA ngắn; không có tác vụ đòi hỏi chép chính xác văn bản xa (OCR kiểu "đọc lại ở đầu trang").
- Số 22,2x so với baseline recompute (rất yếu về hiệu quả); so với dense-with-KV cache thông thường ở độ dài vừa phải thì lợi ích về tốc độ nhỏ hơn nhiều.
- Giả định chiến lược cache nhìn từ token đầu; với VLM, các token đầu thường là prompt/ảnh, nên "token đầu" có thể mang nội dung quan trọng (không được paper thử).

## 8. Liên hệ với các bài khác

- **17 Longformer:** window attention là nền; bài này chỉ ra window thuần túy hỏng trên mô hình đã pre-train dense và sửa bằng sink (Paper, Phụ lục B, nói Longformer/ETC/BigBird dùng attention toàn cục không hợp với mô hình tự hồi quy và phải huấn luyện lại, không áp được lên mô hình đã pre-train. Ta thấy "global tokens" của Longformer gần ý tưởng sink, nhưng đó là suy luận của ta).
- **12 RoPE, 13 ALiBi:** cần gán vị trí theo cache để hoạt động.
- **56 H2O:** cùng loại "chính sách loại KV". H2O chọn token theo điểm attention tích lũy thay vì vị trí cố định; H2O (Phụ lục C.4) nhận xét StreamingLLM mất thông tin ở giữa đầu vào, và H2O hoạt động trên 4 triệu token tốt hơn.
- **57 FastV:** tỉa token thị giác theo attention; cùng tinh thần "attention cho biết token nào bỏ được", nhưng đây là tỉa trong prefill/lớp sâu, không phải cache streaming.
- **44 FlashAttention, 46 PagedAttention:** chọn lọc KV tương thích với kernel chuẩn vì StreamingLLM không cần ma trận attention tường minh (khác H2O).
- **54 AWQ, 52 SmoothQuant:** paper có nhắc rằng ngoại lai (outlier) của lượng tử hóa liên quan đến attention sink (Xiao et al., Bondarenko et al.).
- **24 Gated DeltaNet / 18 Linear Transformers:** các lớp tuyến tính không có softmax, nên không có ràng buộc "tổng attention bằng 1" mà paper dùng để giải thích sink (xem mục 9).

## 9. Ứng dụng cho dự án (suy luận)

1. **StreamingLLM ít cần cho bài toán trang OCR ~1.500 token.** Chuỗi ngắn hơn cửa sổ pre-train; cache không bao giờ đầy; không cần loại token. Giá trị nằm ở **hiểu biết**, không ở việc triển khai.
2. **Kiểm tra sink trong các lớp full-attention của mô hình.** Mô hình lai chỉ có ~1/4 lớp softmax. Rất đáng vẽ attention của các lớp này để xem token nào đang làm sink (token BOS? token ảnh đầu tiên?). Khi **cắt tỉa head hoặc lớp**, nếu cắt nhầm head đóng vai trò sink hoặc làm thay đổi cách mô hình "đổ" attention thì có thể gây hỏng không mong muốn. Chỉ số D_res của bạn có thể tăng không do mất thông tin mà do mất "cơ chế bù".
3. **Chưng cất:** học sinh nhỏ phải tái tạo hành vi sink của giáo viên; nếu sink chỉ nằm ở vài head, nên đưa những head đó vào danh sách "không cắt" hoặc thêm loss khớp attention ở các lớp full-attention (gợi ý, không có trong paper).
4. **Lớp GatedDeltaNet:** không có softmax chuẩn hóa nên có thể không cần sink (suy luận từ lập luận của paper về softmax; chưa được kiểm). Nếu đúng, phần lớn mô hình đã "miễn nhiễm" với vấn đề này, và đó là lý do kiến trúc lai thuận lợi cho streaming. Trạng thái recurrent có kích thước cố định nhưng cũng "quên" dần theo gate, nên là một dạng cửa sổ mềm.
5. **Nếu sau này muốn xử lý nhiều trang/tài liệu dài liên tục trong 8 GB:** có thể thay các lớp full-attention bằng sliding window + sink (chỉ 6 lớp), giữ KV hằng số. Cần kiểm tra OCR có cần tham chiếu xa không (header/footer lặp lại, số trang). Đây là hướng mở rộng, không phải nhu cầu hiện tại.
6. **ViT và registers:** paper nhắc hiện tượng tương tự ở ViT ("registers"): ViT 12 khối có thể có token nền mang vai trò sink; khi cắt tỉa ViT hoặc tỉa token thị giác (bài 57), cần tránh loại những token đó. Suy luận từ phụ lục B.
7. **Batch 1 ở độ dài này:** như mục 4.3, KV chỉ ~15 MB nên StreamingLLM không giảm độ trễ; ưu tiên lượng tử hóa trọng số (bài 54) và giảm kernel launch.

## 10. Thí nghiệm gợi ý

1. Với mô hình 0.8B gốc, vẽ attention (trung bình theo head) của các lớp full-attention trên 50 trang: đo tỉ lệ attention dồn vào token đầu tiên và vào token ảnh đầu tiên, theo từng lớp. So sánh trước/sau cắt tỉa và sau chưng cất.
2. Ablation nhỏ: trong lúc decode, che KV của token đầu (BOS/prompt đầu) ở các lớp full-attention và đo CER; nếu CER tăng vọt tức là sink đang hoạt động; thử giữ 1, 2, 4 token đầu.
3. Cắt tỉa head có/không bảo vệ các head sink (xác định bằng điểm attention vào token đầu), so sánh khả năng phục hồi bằng chưng cất.

## 11. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Attention sink | Token đầu nhận lượng attention lớn bất kể ngữ nghĩa, do softmax tổng = 1 |
| Window attention | Chỉ giữ L KV gần nhất; hỏng khi token đầu bị loại |
| StreamingLLM | Giữ 4 KV đầu + L-4 KV gần nhất; vị trí gán theo vị trí trong cache |
| Kích thước KV | $2\,n_\ell n_{kv} d_h b\cdot\min(T,L)$ |
| Độ phức tạp tổng | Dense $O(T^2)$; window/StreamingLLM $O(TL)$; recompute $O(TL^2)$ |
| SoftMax$_1$ | $e^{x_i}/(1+\sum_j e^{x_j})$ ("Zero Sink") |
| Sink token học được | Thêm 1 token vào đầu mọi mẫu khi pre-train; chỉ cần 1 sink khi streaming |
| Kết quả | Ổn định tới 4 triệu token; nhanh tới 22,2x so với recompute |

## 12. Kết luận ngắn

Bài này biến một "lỗi kỳ lạ" (sliding window hỏng khi bỏ token đầu) thành một cơ chế được giải thích: softmax cần một chỗ để đổ attention thừa, và token đầu đóng vai đó. Bốn token đầu cộng một cửa sổ gần nhất là đủ để giữ bộ nhớ cố định mà không fine-tune. Với dự án, giá trị chính là chẩn đoán: cách các lớp softmax của mô hình lai dùng sink có thể ảnh hưởng tới việc cắt tỉa head và chưng cất; còn việc tiết kiệm KV ở batch 1 với trang 1.500 token là không đáng kể.

Nguồn gốc: https://arxiv.org/abs/2309.17453
