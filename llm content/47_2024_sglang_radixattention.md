# SGLang: Efficient Execution of Structured Language Model Programs (2023/2024)

**Tác giả:** Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, Ying Sheng.  
**Venue / arXiv:** arXiv:2312.07104 (v1 12/2023; bản v2 06/2024, ghi "Preprint. Under review").  
**Nguồn:** https://arxiv.org/abs/2312.07104  
**Chủ đề:** Hệ thống chạy "chương trình LLM" gồm nhiều lần gọi mô hình; RadixAttention (tái sử dụng KV cache theo tiền tố bằng cây cơ số + LRU + lập lịch nhận biết cache); giải mã có ràng buộc bằng FSM nén; speculative execution cho API.  
**Vị trí trong lộ trình:** Đọc trước: 46 (PagedAttention/vLLM, KV cache phân trang, shared prefix thủ công), 44-45 (FlashAttention). Bài này thêm: tái sử dụng KV cache **tự động** giữa các lần gọi, và cách lập lịch để tăng tỉ lệ trúng cache. Đọc tiếp: 48 (speculative decoding, một cách khác để giảm số bước chạy mô hình), 55-56 (quản lý/loại bỏ KV).

> Đây là ghi chú học tập bằng tiếng Việt, diễn giải lại chứ không phải bản dịch. Số liệu lấy từ paper; nhiều biểu đồ bị mất nhãn khi trích PDF nên chỉ dùng số xuất hiện trong chữ hoặc bảng. Phần "Ứng dụng cho dự án" và các số "tự tính" là suy luận của chúng ta. Paper dài vừa (~12k từ); ghi chú tập trung vào RadixAttention (phần liên quan nhất tới suy luận), nén FSM ở mức vừa phải, phần ngôn ngữ lập trình lướt qua.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết](#3-phương-pháp-chi-tiết)
   - 3.1 [Mô hình lập trình (lướt)](#31-mô-hình-lập-trình-lướt)
   - 3.2 [RadixAttention: cây cơ số quản lý KV cache](#32-radixattention-cây-cơ-số-quản-lý-kv-cache)
   - 3.3 [Lập lịch nhận biết cache và định lý tối ưu](#33-lập-lịch-nhận-biết-cache-và-định-lý-tối-ưu)
   - 3.4 [FSM nén cho giải mã có ràng buộc](#34-fsm-nén-cho-giải-mã-có-ràng-buộc)
   - 3.5 [Speculative execution cho API](#35-speculative-execution-cho-api)
   - 3.6 [Phân tích chi phí](#36-phân-tích-chi-phí)
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

Ứng dụng LLM ngày càng không chỉ là "một prompt, một câu trả lời" mà là **chương trình LLM (LM program)**: nhiều lần gọi mô hình xen kẽ với điều khiển luồng (if/for), đầu vào/ra có cấu trúc (JSON, nhiều phương thức). Ví dụ: agent ReAct, tree-of-thought, self-consistency, few-shot, RAG, chat nhiều lượt. Hai khó khăn paper nêu:

1. **Lập trình khó** (xử lý chuỗi thủ công, parse đầu ra mong manh, song song hoá).
2. **Thực thi kém hiệu quả:** các engine như vLLM, TGI, TensorRT-LLM tối ưu độ trễ/thông lượng mà không biết cấu trúc workload. Ví dụ rõ nhất là **KV cache bị bỏ sau mỗi request** dù nhiều lần gọi chia sẻ cùng tiền tố (system prompt, ví dụ few-shot, lịch sử chat). Một ví dụ khác: giải mã ràng buộc (JSON) vẫn sinh từng token một dù nhiều đoạn chỉ có một lựa chọn hợp lệ.

Gloss:
- **Tiền tố (prefix):** phần đầu chung của hai chuỗi token. KV cache của một token chỉ phụ thuộc các token **trước** nó, nên hai request có cùng tiền tố có cùng KV cho phần tiền tố đó.
- **Cây tiền tố (trie) / cây cơ số (radix tree):** cấu trúc lưu tập chuỗi theo tiền tố chung. Cây cơ số là trie nén: một cạnh mang nhãn là một dãy phần tử (không chỉ một).
- **LRU:** loại phần tử ít được dùng gần đây nhất.
- **FSM (máy trạng thái hữu hạn):** dạng tính toán của biểu thức chính quy (regex); mỗi trạng thái cho biết ký tự nào được phép tiếp theo.

## 2. Ý tưởng chính

SGLang gồm hai phần: **ngôn ngữ phía trước (embedded trong Python)** với các primitive `gen`, `select`, `fork`, `join`,... và **runtime** với ba tối ưu. Quan trọng nhất cho suy luận: **RadixAttention** giữ KV cache của cả prompt lẫn kết quả sinh ra trong một **cây cơ số** trên bộ nhớ GPU; request mới tìm tiền tố dài nhất đã có trong cây và dùng lại KV của nó, bộ nhớ cạn thì loại lá ít dùng gần đây nhất (LRU). Bộ lập lịch ưu tiên request có tiền tố khớp dài nhất để tăng tỉ lệ trúng cache; paper chứng minh thứ tự này là tối ưu (trong trường hợp offline). Hai tối ưu còn lại: **FSM nén** để giải mã nhiều token một bước khi ràng buộc chỉ cho một đường đi, và **speculative execution** để giảm số lần gọi API cho mô hình chỉ có API.

## 3. Phương pháp chi tiết

### 3.1 Mô hình lập trình (lướt)

Ví dụ (Hình 2): bộ chấm bài luận về ảnh theo kiểu branch-solve-merge:

```python
@function
def multi_dimensional_judge(s, path, essay):
    s += system("Evaluate an essay about an image.")
    s += user(image(path) + "Essay:" + essay)
    s += assistant(select("related", choices=["yes", "no"]))   # chọn lựa chọn xác suất cao nhất
    forks = s.fork(3)                                          # 3 nhánh chạy song song, chia sẻ tiền tố
    for f, dim in zip(forks, dimensions):
        f += user("Evaluate based on: " + dim)
        f += assistant("Judgment:" + gen("judgment", stop="END"))
    ...
    s += assistant(gen("output", regex=schema))                # giải mã ràng buộc bằng regex
```

Primitive: `gen` (sinh, có thể có `regex`), `select`, `+=`/`extend`, `fork`/`join`, `image`/`video`. Trình thông dịch xử lý prompt như một luồng bất đồng bộ (giống phóng kernel CUDA không chặn). Paper nói cùng chương trình viết bằng API kiểu OpenAI tốn ~2.1x số dòng. Phần này ít liên quan suy luận một mô hình nên lướt.

### 3.2 RadixAttention: cây cơ số quản lý KV cache

Cấu trúc dữ liệu: ánh xạ **dãy token → tensor KV tương ứng**. Cây lưu trên CPU (chi phí duy trì không đáng kể); tensor KV nằm trên GPU, theo bố cục **phân trang với kích thước trang = 1 token** (không liền mạch, giống PagedAttention ở bài 46).

Thao tác:

- **Khớp tiền tố (match):** duyệt cây theo token của request mới, lấy tiền tố dài nhất đã có; KV của phần này dùng lại, chỉ cần tính prefill cho phần còn lại.
- **Chèn (insert):** sau khi xử lý, dãy mới (prompt và cả đầu ra) được thêm vào cây; nếu nó tách giữa một cạnh, cạnh bị **chia đôi** thành hai nút (ví dụ hai phiên chat chia sẻ system prompt, Hình 3 bước 4).
- **Loại bỏ (evict):** LRU, **lá trước**; vì thế các tổ tiên chung được giữ cho tới khi trở thành lá.
- **Đếm tham chiếu:** mỗi nút có reference counter số request đang chạy dùng nó; nút có count 0 mới loại được (không loại thứ batch đang chạy cần).
- **Không cấp bộ nhớ cố định cho cache:** token cache và request đang chạy dùng chung một bể bộ nhớ; khi có nhiều request chờ, hệ thống loại cache để mở rộng batch.

Tương thích với continuous batching, PagedAttention và tensor parallelism (mỗi GPU giữ một phần KV, thao tác cây giống hệt, không cần đồng bộ thêm). Với mô hình đa phương thức, nó lấy **hash của ảnh làm khoá** trong cây, nên các câu hỏi khác nhau về cùng một ảnh dùng lại KV của token ảnh.

"Frontend hint": khi `fork`, frontend gửi tiền tố trước để chắc nó được chèn vào cây, rồi mới gửi phần còn lại.

### 3.3 Lập lịch nhận biết cache và định lý tối ưu

Tỉ lệ trúng cache:

$$\text{hit rate}=\frac{\text{số token prompt lấy từ cache}}{\text{số token prompt}}.$$

Nếu bộ lập lịch xen kẽ các request không liên quan thì cache bị đuổi liên tục (thrashing). Thuật toán 1 của paper:

```text
mỗi vòng lặp:
    requests <- lấy toàn bộ hàng chờ
    với mỗi req: (req.prefix_node, req.prefix_len) <- T.match_prefix(req.input_tokens)
    sắp xếp requests theo prefix_len giảm dần      # tiền tố khớp dài nhất trước
    available <- T.evictable_size() + P.available_size()
    new_batch <- []
    với mỗi req theo thứ tự:
        nếu req.size() + current_size < available:
            new_batch.append(req); tăng ref counter của req.prefix_node
    B.merge(new_batch)                              # vào batch đang chạy
    cấp phát bộ nhớ; nếu thiếu thì T.evict(needed)
    chạy B
    với mỗi request xong: giảm ref counter, T.insert(req)
```

**Định lý 3.1.** Với một batch request (trường hợp offline), duyệt cây cơ số theo thứ tự **tìm kiếm theo chiều sâu (DFS)** cho tỉ lệ trúng cache tối ưu, với điều kiện kích thước cache ≥ độ dài request lớn nhất; và thứ tự "tiền tố chung dài nhất trước" tương đương DFS.

Phác chứng minh (Phụ lục A.3): với mỗi cạnh $e$ của cây, KV của nó phải tính ít nhất một lần, nên chi phí tính toán $C\ge\sum_{e}|e|$. Duyệt DFS: lần đầu tính cạnh $e$ thì tính luôn cả cây con dưới nó, trong lúc đó $e$ liên tục được truy cập nên không bị loại (cache đủ chứa đường dài nhất), và sau khi xong cây con thì $e$ không bao giờ được quay lại. Vậy mỗi cạnh được tính đúng một lần: $C=\sum_e|e|$, đạt cận dưới. Khi đó

$$\text{hit rate}=1-\frac{C}{\sum_{r\in R}\#\text{prefill tokens}(r)}.$$

Phần tương đương DFS dùng quy nạp. Trường hợp online thì thứ tự DFS bị phá vỡ nhưng lịch vẫn xấp xỉ DFS trên phần cây tăng thêm. Lưu ý chú thích 2 của paper: thực tế tính toán không hoàn toàn như trong chứng minh vì số token đầu ra không biết trước có thể dẫn tới tính lại KV.

**Ví dụ số (tự tính).** Bài few-shot: tiền tố chung 1,000 token (các ví dụ), 14 câu hỏi mỗi câu 100 token. Không cache: $14\times1100=15{,}400$ token prefill. Có cache (DFS): $1000+14\times100=2{,}400$ token, hit rate $=1-2400/15400\approx84\%$.

**Phân tán:** tensor parallelism đã nêu; với data parallelism nhiều worker, phụ lục A.4 bàn riêng (không đi sâu ở đây).

### 3.4 FSM nén cho giải mã có ràng buộc

Giải mã có ràng buộc (ví dụ ép đầu ra khớp regex JSON): hệ thống thường chuyển regex thành FSM, mỗi bước lấy các token hợp lệ từ trạng thái hiện tại và đặt xác suất token không hợp lệ bằng 0, sinh **một token mỗi bước**. Nhưng nếu một đoạn như `{"summary": "` chỉ có một đường hợp lệ thì có thể "điền" cả đoạn trong **một lượt chạy mô hình** (prefill nhiều token). SGLang phân tích FSM và **nén các cạnh chỉ có một chuyển tiếp liền kề thành một cạnh**, để giải mã nhiều token trong một bước. Điểm kỹ thuật: cần tiền xử lý FSM một lần rồi dùng lại cho cả batch (nếu tiền xử lý mỗi request, thông lượng giảm 2.4x).

### 3.5 Speculative execution cho API

Với mô hình chỉ có API (GPT-4/3.5), chỉ gọi được hộp đen. Với mẫu `gen("name", stop="\n") + "job:" + gen("job", stop="\n")`, bình thường cần hai lời gọi và trả phí token đầu vào hai lần. SGLang cho lời gọi đầu **bỏ qua điều kiện dừng và sinh thêm vài token**, giữ phần dư và khớp lại với primitive sau; nếu khớp thì tiết kiệm một lời gọi. (Chú ý: đây là "speculative" theo nghĩa suy đoán khuôn mẫu, **khác** speculative decoding ở bài 48 vốn là thuật toán lấy mẫu giữ nguyên phân phối.)

### 3.6 Phân tích chi phí

- **Thời gian prefill tiết kiệm:** với tiền tố chung dài $L_p$ và $n$ request dùng lại, FLOPs phần tuyến tính tiết kiệm xấp xỉ $2\cdot\#\text{params}\cdot L_p\cdot(n-1)$ (cùng cấu trúc đếm như Mục 3.3), cộng bớt phần attention bậc hai. Bộ nhớ KV cũng được **chia sẻ** thay vì nhân bản $n$ lần, nên batch lớn hơn.
- **Overhead của cây:** thử trên ShareGPT không có cơ hội tái sử dụng: chạy 100 request mất 74.3 s, quản lý cấu trúc RadixAttention chỉ 0.2 s (<0.3%), vì thao tác cây tuyến tính theo độ dài và nhỏ. Do đó bật mặc định không hại.
- **Chi phí bộ nhớ:** cache chiếm chung bể bộ nhớ KV với request đang chạy; khi áp lực cao, hệ thống ưu tiên batch hơn cache.
- **Ví dụ KV cho mô hình của ta (tự tính, giả định):** 12 KiB/token cho kiến trúc lai (xem 46); một tiền tố chung 1,000 token là ~12 MiB, nên lưu cache rất rẻ so với vài chục TFLOP prefill tiết kiệm được.

## 4. Thí nghiệm và kết quả chính

**Thiết lập:** Llama-2 (7B-70B), Mixtral-8x7B, LLaVA-v1.5-7B (ảnh), LLaVA-NeXT-34B (video), GPT-3.5 (API). Chủ yếu AWS G5 với **A10G 24GB**, một số thí nghiệm A100G 80GB; fp16. Baseline: Guidance (llama.cpp), vLLM v0.2.5, LMQL. Workload: MMLU 5-shot, HellaSwag 20-shot, ReAct, generative agents, tree-of-thought (GSM-8K), skeleton-of-thought, LLM judge (BSM), JSON decoding bằng regex, chat nhiều lượt (4 lượt, đầu vào mỗi lượt 256-512 token; ngắn: 4-8 token đầu ra, dài: 256-512), RAG qua DSPy. Chỉ số: thông lượng (chương trình/giây) và độ trễ khi chạy từng chương trình không batch.

| Nội dung | Kết quả (paper) |
|---|---|
| Thông lượng tổng quát | tới **6.4x** so với hệ thống tốt nhất |
| Độ trễ | giảm tới 3.7x |
| Tỉ lệ trúng cache của các benchmark | 50% đến 99% |
| Tỉ lệ trúng cache của lập lịch nhận biết cache so với tối ưu | trung bình 96% |
| Mô hình lớn (Mixtral-8x7B, Llama-70B, tensor parallelism) | xu hướng speedup tương tự mô hình nhỏ |
| LLaVA-v1.5-7B (ảnh, llava-bench-in-the-wild) | bản gốc của tác giả mô hình 0.18 ảnh/s; SGLang 1.15 ảnh/s |
| LLaVA-NeXT-34B (video, ActivityNet) | gốc 0.02 khung/s; SGLang 0.10 khung/s (5x theo bảng) |
| Chatbot Arena sau một tháng | trúng cache 52.4% với LLaVA-NeXT-34B, 74.1% với Vicuna-33B; giảm thời gian token đầu tiên trung bình 1.7x với Vicuna-33B |
| Chat nhiều lượt, đầu ra dài | gần như không tăng tốc (không có nhiều chia sẻ và decode chi phối) |
| API (GPT-3.5, trích 3 trường từ trang Wikipedia) | giảm chi phí token đầu vào khoảng 3 lần |

Nguyên nhân tăng tốc theo benchmark (paper): MMLU/HellaSwag/RAG tái sử dụng KV của ví dụ few-shot và câu hỏi chung; agent tái sử dụng khuôn mẫu và các lần gọi trước; ToT/SoT song song hoá trong một chương trình; JSON dùng FSM nén; chat nhiều lượt tái sử dụng lịch sử, tác dụng rõ nhất khi đầu ra ngắn vì lúc đó prefill chiếm phần lớn thời gian.

## 5. Phân tích / ablation

- **Tỉ lệ trúng cache vs hiệu năng** (trên tree-of-thought, cách tắt có chọn lọc các token khớp): trúng cache cao hơn dẫn tới batch lớn hơn, thông lượng cao hơn, độ trễ (cả token đầu tiên và tổng) thấp hơn.
- **Thành phần RadixAttention** (Hình 8c): "No Cache", "No Tree-Structure" (bảng tra đơn giản thay cho cây), "FCFS Schedule", "Random Schedule", "No Frontend Parallelism", "No Frontend Hint", "Full". Kết luận của paper: **mỗi thành phần đều cần thiết** để đạt hiệu năng tốt nhất, và việc tắt song song/hint của frontend làm giảm hiệu năng, nên đồng thiết kế frontend-runtime là có ích. Số cụ thể nằm trong biểu đồ, không trích được.
- **Overhead:** <0.3% (Mục 3.6).
- **FSM nén:** tăng thông lượng 1.6x trên JSON decoding; tiền xử lý lại cho từng request làm thông lượng thấp hơn 2.4x.

## 6. Giới hạn

Do tác giả nêu (Future Directions):
- Mở rộng đa phương thức đầu ra; tối ưu RadixAttention qua nhiều tầng bộ nhớ (DRAM, đĩa); khớp mờ theo ngữ nghĩa trong RadixAttention; primitive cấp cao hơn.
- Lập lịch tham lam theo tiền tố có thể gây **đói (starvation)**; cần kết hợp lập lịch công bằng.
- Compiler của SGLang chưa có tối ưu tĩnh nâng cao (lập lịch, quy hoạch bộ nhớ).
- Trong thực tế, số token đầu ra không biết trước làm tính toán khác với chứng minh định lý (chú thích 2).

Do ta nhận thấy:
- Định lý tối ưu giả định cache đủ chứa request dài nhất và chỉ nói về trường hợp offline; trường hợp online chỉ là xấp xỉ.
- Cần tiền tố **trùng khít từng token** (kể cả cách tách từ). Hai prompt khác nhau một ký tự ở đầu mất toàn bộ chia sẻ phía sau.
- Chỉ đánh giá kiến trúc Transformer attention thuần (Llama, Mixtral, LLaVA); không bàn đến mô hình có trạng thái hồi quy (Mamba, GatedDeltaNet), nơi tái sử dụng tiền tố khác bản chất (xem Mục 8).
- Không đo GPU nhỏ ≤ 8 GB; mọi thí nghiệm trên A10G 24GB trở lên.
- Một số baseline (Guidance, LMQL) bị loại khỏi vài benchmark vì chậm hoặc thiếu chức năng (tác giả nói), nên mức so sánh không đồng đều; số 6.4x là "tới", không phải trung bình.

## 7. Liên hệ với các bài khác

- **46 (PagedAttention):** vLLM chỉ hỗ trợ shared prefix khai báo trước; SGLang tổng quát hoá thành cây tự động có LRU và lập lịch. Hai kỹ thuật chồng lên nhau (SGLang dùng trang 1 token).
- **44, 45 (FlashAttention):** kernel tính; SGLang dùng kernel FlashInfer và Triton (theo paper) cho attention trên bố cục không liền mạch.
- **48 (speculative decoding):** hướng khác để giảm thời gian; SGLang "API speculative execution" là suy đoán khuôn mẫu chứ không phải lấy mẫu chấp nhận/từ chối.
- **14-16 (MQA/GQA/MLA):** KV nhỏ hơn thì cache được nhiều tiền tố hơn trong cùng bộ nhớ.
- **18-24 (attention tuyến tính, hồi quy):** KV cache theo token không còn là đối tượng duy nhất; tái sử dụng tiền tố cần lưu **trạng thái hồi quy tại mốc tiền tố**.
- **32, 34, 37, 43 (Flamingo, LLaVA, Qwen2-VL, LightOnOCR):** token ảnh là một phần của tiền tố; SGLang dùng hash ảnh làm khoá.
- **55, 56 (StreamingLLM, H2O):** các chính sách giữ/loại KV; LRU theo cạnh của cây là chính sách cấp request, H2O/StreamingLLM là cấp token.

## 8. Ứng dụng cho dự án (suy luận)

Dự án: OCR một trang ảnh thành ~1,500 token, batch 1, GPU ≤ 8 GB.

1. **Phần lớn ít liên quan.** Mỗi trang có ảnh khác nhau, nên token ảnh (phần lớn prompt) khác nhau giữa các request: cây cơ số chỉ khớp được phần chỉ dẫn văn bản ngắn ở đầu. Ở batch 1 liên tục xử lý các trang độc lập, tỉ lệ trúng cache ≈ chỉ phần tiền tố chung nhỏ.
2. **Trường hợp có lợi:** (a) chạy **nhiều lượt trên cùng một trang** (ví dụ lượt 1 trích bố cục, lượt 2 trích văn bản, hoặc lấy nhiều mẫu để bỏ phiếu/tự kiểm chứng). Khi đó KV của token ảnh có thể tái sử dụng (khoá bằng hash ảnh) và prefill chỉ tính một lần. Số tự tính: nếu một trang có ~1,000 token ảnh sau ViT (giả định), prefill tuyến tính 0.8B tốn $2\times0.8\text{B}\times1000=1.6$ TFLOP; mỗi lượt thêm tiết kiệm đúng chừng đó (cộng chi phí ViT nếu cũng cache được đầu ra ViT, phần này SGLang không bàn). (b) phục vụ nhiều người dùng chung một chỉ dẫn dài.
3. **Mô hình lai làm khó tái sử dụng tiền tố.** Với các lớp full attention thì KV theo token cắt được tại bất kỳ vị trí nào. Với lớp GatedDeltaNet, trạng thái hồi quy là tóm tắt nén của **toàn bộ** tiền tố đến thời điểm đó; muốn dùng lại tiền tố độ dài $k$ cần lưu **trạng thái đúng tại vị trí $k$**, không cắt ngược được từ trạng thái cuối. Nên cây cơ số cho mô hình lai phải lưu thêm "ảnh chụp trạng thái" ở các nút. Paper không xét điều này (suy luận của ta).
4. **Giải mã có ràng buộc (FSM nén):** nếu muốn đầu ra OCR ở định dạng cố định (ví dụ JSON với khoá cố định, bảng HTML), các đoạn khuôn mẫu có thể được "điền" nhiều token một lượt. Với văn bản OCR tự do thì không có lợi.
5. **Ý nghĩa với pruning/distillation:** các kỹ thuật ở đây là tầng hệ thống, không đổi mô hình và không đổi phân phối đầu ra, nên không xung đột với pruning. Một lợi ích gián tiếp: mô hình nhỏ hơn thì KV mỗi token nhỏ hơn, cache được nhiều trang hơn.

## 9. Thí nghiệm gợi ý

1. **Đo mức chia sẻ tiền tố thật của pipeline OCR:** lấy 100 request thật, tokenize (kể cả token ảnh hoặc hash), tính tỉ lệ "token tiền tố trùng với request trước" theo công thức hit rate. Nếu < 5%, bỏ qua RadixAttention cho dự án.
2. **Prefill vs decode:** đo thời gian prefill ảnh + prompt và thời gian decode 1,500 token ở batch 1; xác định phần trăm prefill. Nếu prefill nhỏ (< 10%) thì cache tiền tố tối đa chỉ giảm chừng đó latency.
3. **Cache trạng thái hồi quy:** thử lưu trạng thái GatedDeltaNet tại ranh giới "hết prompt" và dùng lại cho hai lượt giải mã khác nhau trên cùng một ảnh; kiểm tra đầu ra giống hệt lượt tính lại từ đầu (kiểm tra tính đúng đắn của cache).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa / công thức |
|---|---|
| RadixAttention | cây cơ số: dãy token → KV; khớp/chèn/loại (LRU lá) |
| Reference counter | số request chạy đang dùng nút; 0 thì có thể loại |
| Hit rate | token prompt lấy từ cache / token prompt |
| Cận dưới tính toán | $C\ge\sum_{e\in\text{edges}(T)}\lvert e\rvert$ |
| Lập lịch tối ưu (offline) | DFS trên cây = tiền tố chung dài nhất trước; $\text{hit}=1-C/\sum\text{prefill}$ |
| FSM nén | gộp các cạnh một chuyển tiếp để giải mã nhiều token một bước |
| API speculative execution | sinh dư để khớp lời gọi sau (không phải speculative decoding) |
| Overhead cây | 0.2 s trên 74.3 s (<0.3%) |
| Kết quả | thông lượng tới 6.4x, độ trễ giảm tới 3.7x |

## 11. Kết luận ngắn

SGLang biến KV cache từ "rác bỏ sau mỗi request" thành một **bộ nhớ đệm có cấu trúc cây** với LRU và lập lịch ưu tiên tiền tố dài, có chứng minh tối ưu trong trường hợp offline, cộng với giải mã ràng buộc nhanh hơn nhờ FSM nén. Lợi ích lớn khi nhiều lần gọi chia sẻ tiền tố (few-shot, agent, chat, nhiều câu hỏi về một ảnh). Với OCR batch-1 mỗi trang một ảnh, đa phần ít liên quan; chỉ có ích nếu chạy nhiều lượt trên cùng một ảnh, và với mô hình lai thì phải lưu thêm trạng thái hồi quy.

Link gốc: https://arxiv.org/abs/2312.07104
