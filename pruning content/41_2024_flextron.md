# Flextron: Many-in-One Flexible Large Language Model (2024)

**Tác giả:** Ruisi Cai, Saurav Muralidharan, Greg Heinrich, Hongxu Yin, Zhangyang Wang, Jan Kautz, Pavlo Molchanov (NVIDIA; University of Texas at Austin).  
**Venue / arXiv:** arXiv:2406.10260 (v1: 11/06/2024; bản v2: 28/08/2024); Proceedings of ICML 2024 (PMLR 235), theo trang đầu của paper.  
**Nguồn:** https://arxiv.org/abs/2406.10260  
**Chủ đề:** Biến một LLM đã train thành mạng "elastic" lồng nhau (nested): một bộ trọng số, nhiều mô hình con theo ngân sách độ trễ; router chọn mô hình con tĩnh (theo ngân sách) hoặc động (theo từng token); surrogate model để huấn luyện router.  
**Vị trí trong lộ trình:** Đọc trước: 39 (Minitron: importance dựa trên activation, cùng nhóm tác giả), 17 (CoFi), 38 (Sheared LLaMA). Bài này khác hướng "prune một lần ra một mô hình": nó tạo **cả họ mô hình con trong một checkpoint**. Tiếp theo: 42 (Puzzle, tìm kiến trúc không đồng nhất theo từng layer trên phần cứng đích).

> Ghi chú học tập bằng tiếng Việt, diễn giải và tóm tắt, không phải bản dịch. Số liệu lấy từ paper. Phần "Ứng dụng cho dự án" là suy luận của chúng ta; paper này chỉ kiểm tra trên GPT-3 (2B, 8B) và Llama-2-7B, không phải VLM hay kiến trúc hybrid.

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

- Muốn phục vụ nhiều thiết bị (điện thoại, laptop, GPU đám mây) thì phải train hoặc prune nhiều mô hình khác kích thước. Tốn.
- **Elastic network / supernet** (Once-for-All, SortedNet, MatFormer): một mạng chứa nhiều mạng con lồng nhau; triển khai chỉ việc "cắt lát" theo ngân sách. Vấn đề: kiến trúc không chuẩn và thường huấn luyện lâu hơn một mô hình bình thường.
- **Mixture-of-Experts (MoE)**: nhiều "chuyên gia" (expert) trong một layer, router chọn expert theo từng token (input-adaptive). Nhưng thường chỉ áp dụng cho FFN, expert cùng kích thước, trọng số lưu riêng nên tốn bộ nhớ/băng thông.
- **Router**: mạng nhỏ đưa ra quyết định chọn nhánh nào. Khó huấn luyện vì gradient từ loss cuối đến router yếu; trong MoE có hiện tượng **expert collapse** (router luôn chọn một đường).
- Flextron muốn lấy điểm tốt của cả ba: elastic lồng nhau (một bộ trọng số), MoE (input-adaptive), và hiểu ngân sách độ trễ.

## 2. Ý tưởng chính

Lấy một LLM đã train, xếp hạng head và neuron MLP theo độ quan trọng rồi **hoán vị** để phần quan trọng nằm đầu. Khi đó "mô hình con" chỉ cần lấy N head đầu và d neuron đầu của mỗi layer (lồng nhau, dùng chung trọng số, chỉ cần slicing). Tiếp tục train (continued training) với việc mỗi bước lấy mẫu ngẫu nhiên vài mô hình con cùng huấn luyện, để mọi mô hình con đều chạy tốt. Sau đó huấn luyện **router** nhận vào ngân sách độ trễ (và, nếu muốn, hidden state của token) để chọn độ rộng từng layer. Router được huấn luyện gián tiếp qua một **surrogate model** (mô hình thay thế dự đoán loss của LLM chỉ từ logit của router). Chỉ dùng 7.63% token so với pre-train gốc, kết quả vượt nhiều mô hình tương đương train riêng.

## 3. Phương pháp chi tiết

### 3.1 Cấu trúc elastic lồng nhau

Mỗi layer có $K$ ứng viên $W_i^j$, nên có $K^N$ mạng con (ví dụ paper nhắc $4^{64}$ cho LLaMA2-7B 32 layer: MLP và MHA mỗi cái 4 mức).

**Elastic MLP** (lấy cảm hứng MatFormer):

$$\boxed{\mathrm{MLP}_j(x)=\sigma\!\big(X\cdot I_{d_j}W^{(1)\top}\big)\cdot I_{d_j}W^{(2)}}$$

$I_{d_j}$ là ma trận đường chéo $D\times D$ có $d_j$ phần tử đầu bằng 1; thực tế là slicing $W^{(1)}[0:d_j,:]$. Ràng buộc $d_1<d_2<\dots<d_K=D$. Chọn $K=4$ (ví dụ 25%, 50%, 75%, 100% độ rộng).

**Elastic MHA:** dùng $d_j$ head đầu:

$$\mathrm{MHA}_j(x)=\mathrm{Concat}(\text{head}_1,\dots,\text{head}_{d_j})\cdot I_{d_jH}W^O$$

Paper cho rằng đây là công trình đầu tiên hỗ trợ cả MLP và MHA elastic cùng lúc. MHA ảnh hưởng nhiều tới độ trễ và KV cache.

### 3.2 Xếp hạng + hoán vị (importance)

Dùng 512 mẫu; độ quan trọng là tổng chuẩn L1 của activation (khác Minitron dùng L2/mean):

$$F^{(i)}_{head}=\sum_X\big\|\mathrm{Attn}(XW_{Q,i},XW_{K,i},XW_{V,i})\big\|_1,\qquad F^{(i)}_{neuron}=\sum_X\big\|X^{\top}W^{(1),r}\big\|_1$$

Sau đó sắp xếp lại các hàng/cột trọng số theo thứ tự giảm dần importance, nên mô hình con nhỏ luôn giữ phần quan trọng nhất.

### 3.3 Elastic continued training

```
Input: LLM đã train, 512 mẫu để xếp hạng
1. permute head/neuron theo importance (giảm dần)
2. for step in 1..80000:                       # batch 256
       lấy mẫu ngẫu nhiên k=3 mạng con (one-hot s_i cho mỗi layer) + mạng đầy đủ
       L_joint = sum_{j=0}^{k-1} L(M_j(x), y)   # tích luỹ gradient cho 4 mạng, cập nhật 1 lần
```

Lấy mẫu vì số mạng con quá lớn. Dùng 89.9B token (7.54% pre-train GPT-3, 4.50% với Llama-2).

### 3.4 Router và ràng buộc độ trễ

Bài toán: $\min_{S_t}\sum_t\mathcal L_{CE}(M_{s_t})$ với $\mathrm{Latency}(M_{s_t})\le T_t$, đưa về Lagrange:

$$\boxed{\mathcal L=\sum_t\mathcal L_{CE}(M_{s_t})+\lambda\cdot\mathcal T_T(M_{s_t}),\quad \mathcal T_T=\sum_t\max(\mathrm{Latency}(M_{s_t})-T_t,0)}$$

Độ trễ thật được lấy qua **lookup table (LUT)** đo trước. Ràng buộc có thể thay bằng bộ nhớ GPU.

- **Flextron-Static:** router mỗi layer là MLP nhỏ nhận *độ trễ mục tiêu* $T$ và cho logit trên 4 ứng viên: $s_i=\arg\max R_i(T)$. Mọi token dùng chung một mô hình con.
- **Flextron-Adaptive (dynamic):** router nhận thêm hidden state $h_i$ của token: $s_i=\arg\max R_i(T,h_i)$, với $R_i(T,h_i)=\sigma(T\cdot W+h_iW_{H_i}^{\top})W_{R_i}$, chiều nhúng $U\le128$. Quyết định theo từng token.

### 3.5 Surrogate model (SM)

Vì gradient từ loss cuối về router yếu, thêm một mạng nhỏ dự đoán *loss LM của cả LLM chỉ từ logit router*: $r=\mathrm{Concat}(R_0(T),\dots,R_{N-1}(T))$, $S(r)=\sigma(rW_{S1}^{\top})W_{S2}$. Quy trình (Hình 3):
1. LLM đóng băng; router luôn nhận gradient từ latency loss; SM được cập nhật để dự đoán LM loss (MSE).
2. Khi sai số MSE của SM dưới ngưỡng, router còn được cập nhật bằng cách *giảm output của SM* (SM làm proxy khả vi cho loss thật).
3. Sau khi xong, bỏ SM và tinh chỉnh chung router + LLM (joint tuning).

Router tuning chỉ 1000 bước, batch 256, 1.049B token (static router chỉ cần dữ liệu một domain Wikipedia vì xếp hạng mô hình con ổn định giữa các domain; input-adaptive dùng một phần dữ liệu pre-train). Với Llama-2 thêm 2.62B token từ dữ liệu continued-training Nemotron-4 cho joint tuning.

## 4. Thí nghiệm và kết quả chính

Mô hình nền: GPT-3 2B (24 layer, hidden 2048, 16 head, 1.2B non-embedding), GPT-3 8B (32 layer, hidden 4096, 32 head, 6.4B non-embedding, vocab 320,000), Llama-2-7B (6.5B non-embedding). Context 4096.

**Bảng downstream (Table 1; "#Params" là non-embedding; model dynamic dùng số tham số kích hoạt trung bình; ×0.7 nghĩa là độ trễ còn 0.7):**

Flextron-Llama2-7B:

| | #Params | ARC-E | LAMBADA | PIQA | Winogrande | MMLU (5) | HellaSwag (10) | Avg |
|---|---|---|---|---|---|---|---|---|
| Full | 6.5B | 75.1 | 71.5 | 77.5 | 69.1 | 45.1 | 78.1 | 69.4 |
| Static-0.7× | 4.2B | 65.8 | 64.2 | 75.6 | 62.3 | 41.9 | 67.1 | 62.8 |
| Dynamic-0.7× | 4.1B | 68.6 | 65.1 | 76.1 | 63.7 | 42.2 | 69.4 | 64.2 |
| Static-0.6× | 4.0B | 66.1 | 63.8 | 75.0 | 62.1 | 37.7 | 68.0 | 62.1 |
| Dynamic-0.6× | 3.9B | 67.1 | 63.8 | 74.9 | 62.2 | 39.4 | 69.7 | 62.8 |
| Static-0.5× | 3.5B | 65.9 | 61.7 | 74.8 | 61.9 | 35.9 | 67.6 | 61.3 |
| Dynamic-0.5× | 3.4B | 66.5 | 62.9 | 74.1 | 62.0 | 36.8 | 68.5 | 61.8 |

(Đơn vị %.) So sánh trong cùng bảng: Llama2-7B gốc 69.2 (avg); OpenLLaMA-3Bv2 (3.2B) 60.3; OpenLLaMA-7Bv2 (6.5B) 66.0; Sheared-LLaMA-2.7B (2.5B) avg 62.1 (MMLU 26.4); Pythia-2.8B 54.5. Các phương pháp nén: LLM-Pruner 4.5B (MMLU 23.9), Compresso 4.5B (25.9), LaCo 4.7B (26.5), SliceGPT 4.8B (28.9). Flextron-Llama2-7B ở 0.5× (3.5B) có MMLU 35.9% (static) / 36.8% (dynamic), cao hơn các phương pháp nén kia trong bảng. Bản Flextron-8B (GPT-3 8B): Full 66.8 avg, Static-0.5× (3.4B) 59.1, Dynamic-0.5× 60.0.

**Độ trễ thật (Table 2; TensorRT-LLM, A100 80GB, batch 1, prompt 8 / sinh 512; "tham số / giây"):**

| | Full | 0.7× | 0.6× | 0.5× |
|---|---|---|---|---|
| Flextron-8B | 6.4B / 10.43s | 4.1B / 8.02s | 3.9B / 6.39s | 3.4B / 5.48s |
| Flextron-Llama2-7B | 6.5B / 9.64s | 4.1B / 7.09s | 3.9B / 5.41s | 3.4B / 4.91s |

Lưu ý của paper: phần embedding chiếm 17.4% độ trễ đầy đủ ở GPT-3-8B (1.82s, vocab 320k) và 7.2% ở Llama2-7B (0.69s) - phần này không được cắt nên độ trễ giảm ít hơn tỷ lệ tham số.

**Chi phí (Table 4):** elastic CT 89.9B token (7.54% pre-train GPT-3 1.1T; 4.50% Llama-2 2T), router tuning 1.049B (0.09%/0.18% kèm joint tuning 2.62B).

**Scaling law (Eq. 12):** $L(N)=(N/N_c)^{-\alpha_N}+E_N$. Hệ số khớp (Table 3): Matformer $N_c=1.680,\alpha_N=52.74,E_N=1.729$ (thứ tự cột trong bảng text là $N_C$, $\alpha_N$, $E_N$; bản text khó đọc nên có thể bị đổi chỗ nhãn); Flextron-static 1.465/38.57/1.733; input-adaptive 1.289/25.52/1.729. Paper dùng chúng như hướng dẫn chọn mô hình theo ngân sách; không giải thích thêm trong phần đã đọc.

## 5. Phân tích / ablation

- **Router gán nhiều tính toán hơn cho MLP ở layer sâu:** thay layer đầy đủ bằng 75%/50% độ rộng (GPT3-2B): (1) thay MLP gây tụt hiệu năng nhiều hơn MHA; (2) thay layer sâu, đặc biệt MLP sâu, tụt mạnh. Cấu trúc router học được (Hình 7) khớp với điều này. Ở ngân sách thấp, nên dùng MHA nhẹ (ít head).
- **MHA vs MLP độ trễ:** ở GPT3-2B, mỗi layer MHA 3.830 ms, MLP 3.016 ms dù MHA ít tham số hơn; do đó elastic MHA có ích ở vùng độ trễ cao, và là điểm Flextron hơn MatFormer (chỉ elastic MLP).
- **Router động gán nhiều tính toán hơn cho mẫu khó** (Hình 8-9, GPT3-2B): dữ liệu code (HTML, Java) có đường suy giảm hiệu năng phẳng (cần ít tham số); đa ngôn ngữ cần nhiều hơn. Với mô hình độ trễ 61.8%, gần một nửa số token HTML chọn ứng viên nhỏ nhất ở layer đầu, còn Books3 thường chọn layer đầy đủ. PPL HTML 1.571 so với Books3 11.64.
- **Quỹ đạo elastic CT (Hình 11):** hiệu năng mô hình đầy đủ không giảm; mọi mô hình con hội tụ đồng bộ; mô hình cỡ trung ổn định hơn (có thể vì được lấy mẫu nhiều hơn).
- **Router training dynamics (Hình 12):** ba giai đoạn: SM tuning (L2 loss giảm nhanh, LM loss hơi tăng); joint tuning; router tuning (LM loss tiếp tục giảm).
- **Router học được vs ngẫu nhiên (Hình 13):** mô hình con ngẫu nhiên ở độ trễ 65% có loss trung bình từ 2.32 đến 3.06; router tìm được mô hình nằm ở vùng tốt nhất.
- **Hoán vị trọng số theo importance là cần thiết (Table 5, Wikipedia PPL khi chỉ dùng nửa đầu):** MHA: 184.5 (không sắp xếp) vs 179.9 (có sắp xếp); MLP: 1902.0 vs 193.6; mô hình đầy đủ 9.144. Nghĩa là sắp xếp giúp MLP rất nhiều, MHA ít hơn; cả hai vẫn xa mô hình đầy đủ khi chưa train.
- **So sánh với train từ đầu (Hình 5):** Flextron "superior to Matformer and even end-to-end-trained smaller models (843M)" theo caption của paper.

## 6. Giới hạn

*Tác giả nêu/ngầm hiểu:*
- Router động cần thêm bộ nhớ và độ phức tạp triển khai; hiệu quả phụ thuộc vào LUT độ trễ phù hợp phần cứng.
- Embedding không cắt, chiếm tỷ lệ độ trễ đáng kể (17.4%).

*Chúng ta nhận xét:*
- Chỉ độ rộng (số head, neuron MLP) là elastic, không có depth; cấu trúc lồng nhau buộc mọi mô hình con dùng *prefix* của thứ tự importance, không cho chọn tập con tuỳ ý.
- Quy mô 89.9B token cho CT vẫn lớn; mô hình con ở 0.5× giảm MMLU nhiều (Llama-2: 45.1 -> 35.9-36.8).
- Router dynamic theo token khó tận dụng trên GPU với batch lớn và kernel tĩnh; lợi ích độ trễ đo trong paper là cho batch 1 và các cấu hình uniform/static. Dynamic: paper báo "số tham số kích hoạt trung bình", không phải độ trễ đo thật cho dynamic (Table 2 chỉ liệt kê mô hình đầy đủ và các mức giảm).
- Đánh giá chủ yếu trên benchmark ngôn ngữ, không phải tác vụ trích xuất có cấu trúc.

## 7. Liên hệ với các bài khác

- **39 Minitron:** cùng nhóm; importance bằng activation; Minitron chọn một cấu hình và retrain bằng distillation, còn Flextron giữ nhiều cấu hình lồng nhau trong một checkpoint (và dùng CE thường chứ không distill).
- **38 Sheared LLaMA, 17 CoFi, 25 ZipLM:** đối thủ so sánh; học cấu hình bằng mask vs router.
- **42 Puzzle:** hướng chọn cấu hình theo layer dựa trên phần cứng; Puzzle chọn một mô hình, Flextron chọn lúc chạy.
- **13 Sixteen heads, 14 Voita heads:** bằng chứng head có thể bỏ; Flextron sắp xếp head theo importance.
- **18 2:4 sparsity, 19 HALP:** liên quan ràng buộc độ trễ qua LUT (HALP dùng LUT cho CNN).

## 8. Ứng dụng cho dự án (suy luận)

*Mục này là suy luận của chúng ta.*

- **Mục tiêu của dự án là hai điểm cố định (~0.3B, ~0.1B)**, không phải ngân sách thay đổi lúc chạy. Router động và surrogate model vì vậy ít cần thiết; **ít liên quan** về mặt triển khai.
- **Phần đáng học nhất:** (1) *hoán vị theo importance* rồi dùng nested prefix, Table 5 cho thấy nó làm MLP tụt từ PPL 1902 xuống 193.6 khi cắt nửa, nên là bước nên làm khi cắt neuron; (2) *nested training*: train đồng thời mô hình ~0.3B và ~0.1B lồng nhau trong cùng bộ trọng số 0.8B, thay vì hai lần prune + distill riêng; (3) dùng *LUT độ trễ* đo thật trên GPU 8 GB để chọn cấu hình thay vì đếm tham số.
- **Quan sát "cắt neuron MLP hồi phục tệ hơn cắt layer":** Flextron cho thấy MLP layer sâu nhạy cảm hơn MLP layer nông và MHA ít nhạy hơn MLP. Gợi ý (chỉ giả thuyết) rằng nên cắt MLP *không đồng đều theo độ sâu*: giữ MLP layer sâu, cắt nhiều hơn ở layer nông và ở attention.
- Decoder hybrid GatedDeltaNet + attention: Flextron chỉ thử attention thuần; chưa biết "head" của GatedDeltaNet có xếp hạng và cắt được tương tự không.
- **Phần ViT:** có thể áp dụng tương tự slimmable network cho ViT (đã có tiền lệ CNN), nhưng paper này không kiểm chứng.

## 9. Thí nghiệm gợi ý

1. **Hoán vị theo importance:** trên decoder 0.8B, đo CER/PPL khi giữ 50% neuron MLP đầu (không train) trước và sau hoán vị theo activation L1 (tái hiện Table 5); xem MLP có cải thiện nhiều như trong paper không.
2. **Nested train nhỏ:** train đồng thời mô hình 100% và 40% độ rộng MLP bằng lấy mẫu ngẫu nhiên (k=1 mô hình con mỗi bước) với KD từ teacher 4B; so với prune + KD riêng cho mô hình 40%.
3. **Độ nhạy theo độ sâu:** thay MLP từng layer bằng 50% độ rộng (như Hình 6), vẽ tụt CER theo chỉ số layer để xem layer nào phải giữ.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Nested elastic MLP/MHA | Mô hình con = $d_j$ neuron/head đầu của ma trận đã hoán vị, dùng chung trọng số |
| Importance | $\sum\|\cdot\|_1$ activation của head/neuron, 512 mẫu |
| Joint loss | $\sum_{j}L(M_j(x),y)$ trên $k=3$ mạng con ngẫu nhiên + mạng đầy đủ |
| Latency loss | $\max(\mathrm{Latency}-T_t,0)$ qua LUT |
| Static / Adaptive router | $\arg\max R_i(T)$ / $\arg\max R_i(T,h_i)$ |
| Surrogate model | MLP nhỏ dự đoán LM loss từ logit router để huấn luyện router |
| Cost | 89.9B token CT + ~1B token router |

## 11. Kết luận ngắn

Flextron biến một LLM thành họ mô hình lồng nhau với cả head và neuron MLP elastic, có router chọn mô hình con theo độ trễ và theo token, huấn luyện router qua surrogate model. Với dự án có hai kích thước cố định, các ý hữu ích là hoán vị theo importance, huấn luyện lồng nhau nhiều kích thước cùng lúc và dùng LUT độ trễ; phần router động ít liên quan.

Link gốc: https://arxiv.org/abs/2406.10260
