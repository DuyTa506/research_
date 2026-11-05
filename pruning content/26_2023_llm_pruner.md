# LLM-Pruner: On the Structural Pruning of Large Language Models (2023)

**Tác giả:** Xinyin Ma, Gongfan Fang, Xinchao Wang (National University of Singapore).  
**Venue / arXiv:** arXiv:2305.11627 (19/05/2023, bản v3 09/2023). Bản text không ghi tên hội nghị.  
**Nguồn:** https://arxiv.org/abs/2305.11627  
**Chủ đề:** Structured pruning task-agnostic cho LLM (LLaMA-7B/13B, Vicuna-7B, ChatGLM-6B): tự động tìm các cấu trúc phụ thuộc nhau (dependency graph), chấm điểm cả nhóm bằng thông tin gradient (Taylor), phục hồi nhanh bằng LoRA trên 50k mẫu.  
**Vị trí trong lộ trình:** Đọc sau 20 (DepGraph, nền tảng "nhóm phụ thuộc"), 07 (Molchanov, importance bằng Taylor), 04/05 (cắt filter/channel) và 25 (ZipLM). Bài này là phương pháp structured cho LLM đầu tiên có công cụ tự động, và là baseline mà FLAP (28) và SliceGPT (29) so sánh. Bài kế: 27 (LLM Surgeon), 28 (FLAP).

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Số liệu lấy từ paper (một vài bảng phụ lục bị lỗi định dạng nên được ghi chú rõ). Phần "Ứng dụng cho dự án" là suy luận của chúng ta.

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

Trước bài này, nén mô hình ngôn ngữ (BERT, v.v.) thường **task-specific**: fine-tune trên một tác vụ rồi nén (cho một bài GLUE), hoặc chưng cất tổng quát (TinyBERT) rất tốn kém (paper nêu: khoảng 14 GPU-ngày cho general distillation của TinyBERT). Với LLM có hai trở ngại:

- **Kho dữ liệu huấn luyện khổng lồ** (hàng nghìn tỷ token, đôi khi không công khai), nên khó dùng lại để phục hồi mô hình sau cắt.
- **Thời gian post-training sau cắt quá dài.**

Paper đặt bài toán là **task-agnostic compression**: giảm cỡ mô hình nhưng giữ được tính "giải mọi tác vụ", dùng ít dữ liệu và thời gian. Hai khái niệm nền:

- **Structured pruning:** xoá cả hàng/cột/neuron/head thay vì từng trọng số, nên ma trận thật sự nhỏ lại.
- **Cấu trúc kết nối (coupled structures):** trong mạng, các tensor ràng buộc nhau. Ví dụ neuron $j$ trung gian của MLP gồm hàng $j$ của up/gate và cột $j$ của down; xoá một nửa mà không xoá nửa kia là vô nghĩa hoặc sai kích thước.

## 2. Ý tưởng chính

LLM-Pruner làm ba bước: (1) **Discovery:** từ định nghĩa phụ thuộc giữa các neuron, chạy lan truyền tự động để gom mọi trọng số bị ràng buộc vào cùng một nhóm, nên bạn không cần tự phân tích từng kiến trúc; (2) **Estimation:** chấm điểm cả nhóm (không phải từng ma trận) bằng đóng góp của nhóm vào loss next-token, ước lượng trên chỉ 10 câu mẫu công khai, dùng thông tin gradient vì dữ liệu không thuộc phân phối huấn luyện nên gradient không bằng 0; (3) **Recovery:** fine-tune bằng LoRA 2 epoch trên Alpaca làm sạch (khoảng 50k mẫu, 3 giờ, 1 GPU). Kết quả: bỏ 20% tham số LLaMA-7B còn 94.97% hiệu năng zero-shot trung bình.

## 3. Phương pháp chi tiết

### 3.1 Discovery: đồ thị phụ thuộc

Gọi $N_i, N_j$ là hai neuron, $\mathrm{In}(N_i)$ và $\mathrm{Out}(N_i)$ lần lượt là tập neuron trỏ đến và đi từ $N_i$. Hai quy tắc phụ thuộc (có hướng):

$$N_j\in\mathrm{Out}(N_i)\wedge \mathrm{Deg}^-(N_j)=1\;\Rightarrow\;N_j\text{ phụ thuộc }N_i$$
$$N_i\in\mathrm{In}(N_j)\wedge \mathrm{Deg}^+(N_i)=1\;\Rightarrow\;N_i\text{ phụ thuộc }N_j$$

Ở đây $\mathrm{Deg}^-$ là bậc vào, $\mathrm{Deg}^+$ là bậc ra. Ý nghĩa: nếu một neuron chỉ nhận đầu vào từ duy nhất một neuron khác (hoặc chỉ đưa đầu ra cho duy nhất một neuron), bỏ neuron kia thì nó cũng phải bỏ. Lấy từng neuron làm **trigger**, kích hoạt các neuron phụ thuộc, các neuron mới lại trở thành trigger, cho đến khi hết thì thu được một nhóm. Với LLaMA có ba loại nhóm (Fig. 2):

- **Loại A (MLP):** neuron trung gian = hàng của gate và up, cột của down.
- **Loại B (Multi-head attention):** một head = đầu ra tương ứng của Q, K, V và đầu vào tương ứng của ma trận out-projection.
- **Loại C (Channel-wise):** chiều ẩn (hidden dimension) của residual stream; nhóm này đi qua mọi layer, embedding, các norm, đầu LM.

Phần cài đặt gọi hai chiến lược: **Block** (cắt nhóm loại A và B, tức MLP và head ở các layer giữa) và **Channel** (chỉ cắt nhóm C, tức giảm chiều ẩn).

### 3.2 Estimation: điểm của một nhóm

Với $D=\{x_i,y_i\}_{i=1}^N$ ($N=10$, lấy từ BookCorpus, mỗi câu cắt 128 token) và nhóm $G=\{W_i\}_{i=1}^M$, độ quan trọng của một ma trận $W_i$ là thay đổi loss khi xoá nó, khai triển Taylor bậc hai:

$$I_{W_i}=|\Delta\mathcal L(D)|=\Big|\mathcal L_{W_i}(D)-\mathcal L_{W_i=0}(D)\Big|=\Big|\underbrace{\tfrac{\partial\mathcal L^\top}{\partial W_i}W_i}_{\neq 0}-\tfrac12 W_i^\top HW_i+O(\|W_i\|^3)\Big|$$

Điểm khác với OBD/OBS: các phương pháp cổ điển bỏ số hạng bậc nhất vì mô hình đã hội tụ trên dữ liệu huấn luyện (gradient $\approx0$). Ở đây dữ liệu calibration **không** lấy từ dữ liệu huấn luyện gốc nên gradient khác 0, và LLM thì tính Hessian là bất khả thi, nên **giữ số hạng bậc nhất**. Với từng tham số $k$ (mức element) có thể thêm số hạng bậc hai xấp xỉ bằng Fisher:

$$I_{W_i^k}\approx\Big|\tfrac{\partial\mathcal L}{\partial W_i^k}W_i^k-\tfrac12\sum_{j=1}^N\Big(\tfrac{\partial\mathcal L(D_j)}{\partial W_i^k}W_i^k\Big)^2\Big|$$

Các biến thể trong ablation: **Vector** (điểm cho cả ma trận, Eq. 3), **Element1** (điểm theo từng tham số, chỉ bậc nhất), **Element2** (theo tham số, có thêm bậc hai Fisher).

**Gộp điểm trong nhóm** (bốn cách): Sum $I_G=\sum_i I_{W_i}$; Prod; Max; Last-only (chỉ lấy ma trận chạy cuối trong nhóm, lý do: xoá cấu trúc cuối nhóm tương đương xoá mọi kết quả tính trong nhóm). Chọn mặc định **Sum**. Sau đó xếp hạng nhóm trong từng module và xoá nhóm điểm thấp theo tỷ lệ định trước (local pruning; global thử ở phụ lục và hơi kém).

### 3.3 Recovery: LoRA

$f(x)=(W+\Delta W)x+b=(Wx+b)+PQx$, chỉ huấn luyện $P,Q$ (rank 8). Sau huấn luyện gộp $PQ$ vào $W$ nên không tăng tham số. LoRA áp dụng cho mọi ma trận tuyến tính (cả đã cắt lẫn chưa).

```text
LLM-Pruner(model, calib(10 câu), ratio, tune_data):
    groups = discover_dependency_groups(model)        # A: MLP, B: head, C: channel
    loss   = next_token_loss(model, calib); loss.backward()   # lấy gradient
    for g in groups:
        score[g] = aggregate(importance(W_i, grad_i) for W_i in g)   # Sum
    prune lowest-score groups per module until target ratio
        # Block: bỏ 3 layer đầu và layer cuối khỏi diện cắt
    LoRA fine-tune (rank 8, lr 1e-4, 2 epoch, ~50k mẫu Alpaca)
```

Chi tiết cấu hình (phụ lục): với chiến lược Block trên LLaMA-7B, giữ nguyên **3 layer đầu và layer cuối** (tác động quá lớn), mỗi module MLP có 11,008 nhóm; để đạt 20% tham số, cắt **25% nhóm ở layer 5 đến 30**; để đạt 50% tham số, cắt **60% nhóm ở layer 4 đến 30**. Tỷ lệ cắt nhóm cao hơn tỷ lệ tham số vì embedding và các layer bị loại không bị cắt.

## 4. Thí nghiệm và kết quả chính

**Mô hình:** LLaMA-7B, LLaMA-13B, Vicuna-7B (LLaMA tinh chỉnh hội thoại), ChatGLM-6B (song ngữ). **Đánh giá:** zero-shot 7 bộ phân loại (BoolQ, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OpenbookQA), trung bình 7 tác vụ, cộng PPL WikiText2 và PTB. Dòng LLaMA-7B^⋆ là số tự chạy lại của tác giả (63.25), thấp hơn số chính thức (68.59) vì prompt khác.

**Thống kê mô hình (Table 3, đo bằng 64 token đầu vào, A5000):**

| Mô hình | Chiến lược | Tỷ lệ | #Params | MACs | Bộ nhớ | Latency |
|---|---|---|---|---|---|---|
| LLaMA-7B | gốc | 0% | 6.74B | 424.02G | 12884.5 MiB | 69.32 s |
| LLaMA-7B | Channel | 20% | 5.39B | 339.36G | 10363.6 MiB | 61.50 s |
| LLaMA-7B | Block | 20% | 5.42B | 339.60G | 10375.5 MiB | 58.55 s |
| Vicuna-7B | Channel | 50% | 3.37B | 212.58G | 6556.3 MiB | 40.11 s |
| Vicuna-7B | Block | 50% | 3.35B | 206.59G | 6533.9 MiB | 37.54 s |

(Tham số giảm 20% nhưng latency chỉ giảm khoảng 15% với Block: 69.32 s xuống 58.55 s, theo phép tính của ta.)

**LLaMA-7B, 20% tham số (Table 1; trung bình zero-shot / PPL WikiText2):**

| Cấu hình | Wiki2 PPL | PTB PPL | Trung bình 7 tác vụ |
|---|---|---|---|
| LLaMA-7B^⋆ (gốc) | 12.62 | 22.14 | 63.25 |
| L2 (không tune) | 582.41 | 1022.17 | 42.65 |
| Random (không tune) | 27.51 | 43.19 | 52.69 |
| Channel (không tune) | 74.63 | 153.75 | 45.38 |
| Vector (không tune) | 22.28 | 41.78 | 53.25 |
| Element2 (không tune) | 19.77 | 36.66 | 56.82 |
| Element1 (không tune) | 19.09 | 34.21 | 56.69 |
| Channel (có LoRA) | 22.02 | 38.67 | 55.57 |
| Vector (có LoRA) | 18.84 | 33.05 | 57.23 |
| Element2 (có LoRA) | 17.37 | 30.39 | **60.07** |
| Element1 (có LoRA) | 17.58 | 30.11 | 59.23 |

Không tune giữ 89.8% hiệu năng gốc (56.82/63.25); có tune Element2 giữ **94.97%** (60.07/63.25). Điểm bậc hai (Element2) tốt nhất nhưng Element1 (rẻ hơn nhiều) chỉ kém chút, nên tác giả nghiêng về bậc nhất. ChatGLM-6B khác: điểm theo tham số (Element1) hỏng (Table 5, không tune 36.45), điểm Vector tốt nhất (cắt 10%: 46.16 không tune, 47.34 sau tune, so với gốc 47.05).

**LLaMA-13B (Table 2, dùng Element1):** gốc 64.97 (PPL 11.58). Cắt 20% không tune: Block 61.79 (PPL 16.01), Channel 50.29, L2 51.08, Random 57.09; sau tune: Block **64.02** (PPL 15.18), Channel 60.24, L2 62.12, Random 59.78.

**Vicuna-7B 20% (Table 4):** gốc 62.78; không tune Element2 56.83, Element1 56.25; có LoRA Element2 57.78, Element1 57.71, Vector 56.64 (92.03% của gốc, tính theo câu của paper cho điểm tốt nhất).

**Mức cắt 50%:** paper thừa nhận "suy giảm lớn" (xem mục Giới hạn). Từ bảng phụ lục C.5 (định dạng bảng bị rối, ta đọc được) trung bình khoảng 46 so với 63.25 gốc, và Channel không tune có PPL cỡ $10^4$.

## 5. Phân tích / ablation

- **Cần dependency (Table 6):** nếu cắt mỗi ma trận độc lập, không tôn trọng nhóm: PPL WikiText2 **68378** (không tune) và **13307** (có tune), trung bình 38.32 và 38.10, so với có dependency 19.09 và 17.58 (56.69 và 59.23). Đây là nhóm ablation quan trọng nhất của paper: pruning cấu trúc mà phá nhóm thì không phục hồi được.
- **Block so với Channel (Table 1, 2, Fig. 3):** cắt Channel (chiều ẩn đồng đều ở mọi layer) tệ hơn Block rõ rệt. Lý do tác giả đưa ra: tầm quan trọng không phân bố đều giữa các layer; layer đầu và cuối rất quan trọng (Fig. 3: chỉ bỏ nhóm ở một layer, layer 0 và cuối gây tăng PPL lớn), mà Channel buộc phải cắt chúng.
- **Tỷ lệ cắt (Fig. 4):** theo mô tả, L2 (không dữ liệu) sụp quanh 20%, trong khi LLM-Pruner cho phép cắt tới khoảng 60% để đạt cùng mức PPL (với LLaMA); Vicuna cắt 10% bằng L2 tương đương cắt 60% bằng LLM-Pruner. (Số đọc từ hình, nên chỉ nêu xu hướng.)
- **Gộp điểm (Table 7, LLaMA-7B không tune):** Sum PPL 66.13/164.25; Max 62.59/144.38 (PPL tốt nhất nhưng zero-shot tệ nhất); Prod 77.63/192.88; Last-only 130.00/170.88 (zero-shot tốt nhất nhưng sinh văn bản tệ nhất). Chọn Sum làm thoả hiệp.
- **Global so với local (Table 14):** Element1 local avg 56.69 (PPL 19.09), global 56.49 (20.84); local nhỉnh hơn vì điểm khác layer không so sánh được.
- **Số bước tune (Fig. 5, mục C.4):** PPL giảm nhanh trong khoảng 300 bước rồi tăng nhẹ; huấn luyện quá lâu thì overfit Alpaca và hại các tác vụ tổng quát. Dùng 2.59M mẫu thay vì 50k: avg 62.36 so với 59.23 (LLaMA-5.4B, phụ lục C.1, so 50k mẫu Alpaca với 2.59M mẫu từ một nguồn khác).
- **Chưng cất kiểu DistilBERT (chọn 1 trong 2 layer), Table 8:** 44.64 (3.50B) so với LLM-Pruner 48.88 (3.35B).
- **Huấn luyện từ đầu (Table 9):** LLaMA-3B từ LLM-Pruner (3.6B) trung bình 50.30 so với StableLM-3B 45.84 (đều tune Alpaca), nhưng tác giả thừa nhận không phải lúc nào cũng thắng mô hình 3B huấn luyện từ đầu (chênh lệch lớn về kích thước kho huấn luyện).
- **Kết hợp lượng tử hoá (Table 13):** LLM.int8() trên mô hình cắt vẫn giữ avg 62.30 (so với 62.36 trước lượng tử), bộ nhớ 5444 MiB.

## 6. Giới hạn

**Do tác giả nêu:** (a) ở mức cắt cao như 50% hiệu năng sụt mạnh, nén LLM mạnh vẫn là thách thức; (b) mô hình cắt có lúc sinh câu vô nghĩa hoặc lặp token, nhất là với văn bản dài; (c) tune quá nhiều có thể overfit tập tune; (d) không phải lúc nào cũng hơn mô hình cùng cỡ huấn luyện từ đầu.

**Do ta nhận xét:**
- Điểm là **bậc nhất, chỉ bằng 10 câu**; số hạng bậc nhất ($\frac{\partial\mathcal L}{\partial W}W$) có thể ồn và không phản ánh tương tác giữa các nhóm được cắt cùng nhau. Không có bước bù tuyến tính (khác ZipLM/LLM Surgeon/SliceGPT) nên toàn bộ phần hư hại để cho LoRA sửa.
- Phục hồi chỉ bằng LoRA rank 8 và SFT trên Alpaca; không dùng distillation từ mô hình gốc. Do đó mô hình cắt bị "lệch" về phân phối Alpaca, và tác giả không đo trên kiểu tác vụ khác.
- Latency giảm thấp hơn nhiều so với tham số (15% với 20% tham số) vì chỉ đo 64 token, 1 GPU; không có phân tích theo từng phần cứng như ZipLM.
- Bảng phụ lục (C.5) bị rối trong bản text nên ta chỉ nêu xu hướng.

## 7. Liên hệ với các bài khác

- **20 (DepGraph):** cùng nhóm tác giả (Fang, Ma, Wang). LLM-Pruner áp dụng DepGraph cho LLM; **30 (MaskLLM)** cũng cùng nhóm Fang & Wang.
- **07 (Molchanov 2019):** điểm bậc nhất theo Taylor cho khối cấu trúc; LLM-Pruner thêm gộp nhóm và dùng cho LLM với dữ liệu rất ít.
- **01 (OBD/OBS):** LLM-Pruner mượn khai triển Taylor nhưng *giữ* số hạng bậc nhất vì dữ liệu ngoài phân phối.
- **25 (ZipLM), 27 (LLM Surgeon), 28 (FLAP), 29 (SliceGPT):** cùng nhắm structured pruning LLM, đều có thành phần bù/tái tạo mà LLM-Pruner thiếu. FLAP chỉ ra LLM-Pruner (kể cả có LoRA) kém FLAP không cần retrain.
- **38 (Sheared LLaMA), 39 (Minitron):** so sánh "cắt rồi huấn luyện tiếp bằng nhiều token" với cách LoRA 50k mẫu của bài này.
- **43 (Pruning vs scratch):** câu hỏi cắt có hơn huấn luyện từ đầu hay không, bài này chỉ có một so sánh nhỏ (Table 9).

## 8. Ứng dụng cho dự án (suy luận)

- **Nhóm phụ thuộc là nội dung dùng được nhiều nhất.** Cắt "neuron MLP" thực chất là cắt **nhóm loại A**: hàng gate/up và cột down. Với lớp **GatedDeltaNet** (hybrid decoder) và full-attention, nhóm tương ứng phức tạp hơn: q/k/v/gate/beta, conv ngắn, norm riêng theo head, ma trận out. Mô hình này không nằm trong thí nghiệm của bài; khi thiết kế cần vẽ lại đồ thị phụ thuộc (DepGraph, bài 20, tự động làm việc này) để không cắt lệch nửa. Đây là suy luận, chưa có bằng chứng từ paper về kiến trúc lai.
- **Layer đầu và cuối nhạy cảm:** paper giữ 3 layer đầu và layer cuối, và cho thấy Channel (cắt chiều ẩn đồng đều) tệ hơn Block. Với dự án: có thể kiểm tra các block ViT đầu/cuối và các layer decoder đầu/cuối trước khi quyết định phân bổ; đồng thời gợi ý cắt chiều ẩn residual là việc khó.
- **Liên hệ với quan sát "cắt neuron MLP hồi phục kém hơn cắt layer":** bảng 50% của bài cho thấy cắt mạnh nhóm MLP/head mà không có bù thì hỏng nặng; kết quả này phù hợp với quan sát của bạn nhưng bài không so sánh trực tiếp "cắt layer" với "cắt neuron", nên đây chỉ là tương thích, không phải bằng chứng.
- **D_lin / D_res:** LLM-Pruner không bù tuyến tính. Điểm Taylor bậc nhất $|\partial\mathcal L/\partial W\cdot W|$ ước lượng *thay đổi loss khi xoá*, tức tổng $D_{lin}+D_{res}$ (nếu được đo trực tiếp ở loss), không tách hai phần. Nếu bạn muốn xếp hạng theo $D_{res}$, cần các phương pháp có bù (25, 27, 28, 29).
- **Phục hồi:** thay LoRA-Alpaca bằng distillation từ thầy 4B trên dữ liệu OCR là hướng tự nhiên; bài này chỉ cho thấy cái "sàn" (LoRA 3 giờ đã khôi phục 56.69 lên 59.23 hoặc 60.07 trên 63.25).
- **Gradient calibration:** cần backprop qua mô hình 0.8B: vừa với GPU 8 GB nếu dùng 10 mẫu ngắn (không phải số của paper, ước lượng của ta).

## 9. Thí nghiệm gợi ý

1. **Ablation dependency trên mô hình của bạn:** cắt 20% neuron MLP (nhóm A) đúng cách và cắt "lệch" (chỉ cắt hàng gate/up, hoặc chỉ cột down) để nhìn thấy ảnh hưởng, nhằm kiểm tra bộ code dựng nhóm có đúng.
2. **Độ nhạy theo layer (giống Fig. 3):** bỏ nhóm MLP ở đúng một layer mỗi lần (gồm cả ViT block và GatedDeltaNet/full-attention layer), đo thay đổi loss/CER OCR để lập bản đồ độ nhạy theo layer. Dùng nó để quyết định layer nào không cắt.
3. **Điểm bậc nhất so với bậc hai trên calibration OCR:** so Element1 và Element2 (Fisher) trên 10 và 128 mẫu OCR. Nếu chênh nhỏ, giữ bậc nhất cho rẻ.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Phụ thuộc | $N_j\in\mathrm{Out}(N_i)\wedge\mathrm{Deg}^-(N_j)=1\Rightarrow N_j$ phụ thuộc $N_i$ (và quy tắc đối xứng với $\mathrm{Deg}^+$) |
| Nhóm A/B/C | MLP / head attention / kênh ẩn (channel-wise) |
| Điểm ma trận | $I_{W_i}=\lvert\frac{\partial\mathcal L^\top}{\partial W_i}W_i-\frac12W_i^\top HW_i+\dots\rvert$ |
| Điểm tham số | bậc nhất + Fisher bậc hai (Element1 / Element2) |
| Gộp nhóm | Sum (mặc định), Prod, Max, Last-only |
| Phục hồi | LoRA rank 8, lr 1e-4, 2 epoch, ~50k mẫu Alpaca, ~3 giờ 1 GPU |
| Cấu hình cắt LLaMA-7B | 20% tham số: 25% nhóm ở layer 5 đến 30; 50%: 60% ở layer 4 đến 30; giữ 3 layer đầu và layer cuối |
| Con số chính | 20% tham số: 56.69 (không tune) và 60.07 (có LoRA, Element2) so với 63.25 gốc; 94.97% |

## 11. Kết luận ngắn

LLM-Pruner biến structured pruning LLM thành quy trình tự động: tìm nhóm phụ thuộc, chấm điểm nhóm bằng Taylor bậc nhất trên 10 mẫu, phục hồi bằng LoRA trong 3 giờ. Ở 20% nó đủ dùng (còn khoảng 95% hiệu năng), ở 50% thì không. Đóng góp bền nhất là công cụ dựng nhóm phụ thuộc và kết quả "phá nhóm là hỏng", còn phần chấm điểm không bù là chỗ các bài sau (25, 27, 28, 29) cải thiện.

Link gốc: https://arxiv.org/abs/2305.11627
