# DepGraph: Towards Any Structural Pruning (2023)

**Tác giả:** Gongfan Fang, Xinyin Ma, Mingli Song, Michael Bi Mi, Xinchao Wang (National University of Singapore; Zhejiang University; Huawei)  
**Venue / arXiv:** arXiv:2301.12900 (v2, 23/03/2023); bản text không nêu venue. Code: Torch-Pruning (github.com/VainF/Torch-Pruning).  
**Nguồn:** https://arxiv.org/abs/2301.12900  
**Chủ đề:** Structured pruning cho kiến trúc bất kỳ; mô hình hoá **phụ thuộc giữa các lớp** (dependency graph); gom tham số thành nhóm; sparse training ở mức nhóm.  
**Vị trí trong lộ trình:** Đọc sau 04 (Pruning filters, nơi đã có ví dụ "cắt filter lớp này thì phải cắt kernel lớp sau"), 05 (Network slimming), 19 (HALP). Bài này thêm: một **thuật toán tự động** tìm "cắt cái này thì phải cắt cái nào khác" cho mọi kiến trúc, nền tảng kỹ thuật cho các bài structured pruning LLM sau này. Tiếp theo: 26 (LLM-Pruner dùng ý tưởng dependency group cho LLM), 29 (SliceGPT), 38 (Sheared LLaMA).

> Đây là ghi chú học tập bằng tiếng Việt, không phải bản dịch. Phần "Ứng dụng cho dự án" là suy luận của chúng ta, không phải kết luận của paper. Paper tự nói mục tiêu **không** phải đạt SOTA mà là chứng minh tính tổng quát; tiêu chí importance dùng ở đây rất đơn giản (norm).

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

**Structural pruning** (cắt có cấu trúc) xoá hẳn một nhóm tham số (kênh, neuron, head...) để mạng thật sự nhỏ đi và chạy nhanh trên phần cứng thường, trái với **unstructured pruning** chỉ đặt từng trọng số về 0 (cần accelerator đặc biệt để hưởng lợi).

Khó khăn: các lớp trong mạng bị **ràng buộc** với nhau. Ví dụ trong CNN (Hình 1a của paper): muốn xoá một kênh của Conv2 trong khối residual, bạn phải xoá đồng thời BN2, kernel tương ứng ở lớp sau, và vì phép cộng residual yêu cầu hai nhánh cùng số kênh, cả Conv1, BN1 cũng phải cắt cùng chỉ số kênh. Tương tự trong Transformer (head của attention, LayerNorm, MLP), RNN (cổng LSTM dùng phép nhân phần tử), GNN.

Trước bài này, các pruner phải **thiết kế tay** quy tắc nhóm cho từng kiến trúc; không chuyển được sang kiến trúc mới, thậm chí kiến trúc khác trong cùng họ. Vấn đề thứ hai: các tiêu chí importance cũ (norm, Taylor) tính **trong một lớp**; khi nhiều lớp bị cắt cùng nhau, điểm số từng lớp không cộng được vì phân phối và độ lớn khác nhau.

Giải thích thuật ngữ:
- **Coupled / dependent parameters:** các tham số phải cắt cùng nhau, nếu không mạng "gãy" (shape không khớp).
- **Transitive reduction:** rút gọn đồ thị có hướng, giữ ít cạnh nhất mà vẫn bảo toàn quan hệ "tới được".
- **Connected component:** tập đỉnh nối được với nhau trong đồ thị.
- **Group sparsity / sparse training:** thêm phạt regularization khiến cả nhóm tham số về gần 0 cùng lúc.

## 2. Ý tưởng chính

Thay vì liệt kê tay ai phụ thuộc ai, ta xây một **đồ thị phụ thuộc** (Dependency Graph) chỉ ghi các phụ thuộc **cục bộ** giữa lớp kề nhau (đầu ra lớp này nối đầu vào lớp kia, hoặc lớp có đầu vào và đầu ra chung số kênh như BatchNorm). Sau đó **nhóm** là các thành phần liên thông của đồ thị này, tìm bằng duyệt đồ thị trong thời gian tuyến tính. Mỗi nhóm được cắt cùng nhau. Vì các phần tử trong nhóm phải đều "không quan trọng", paper huấn luyện thưa ở mức nhóm (phạt cùng chỉ số kênh trên mọi lớp trong nhóm) để norm của cả nhóm tách bạch, rồi cắt nhóm có norm nhỏ.

## 3. Phương pháp chi tiết

### 3.1 Ví dụ phụ thuộc cơ bản

Xét ba lớp fully-connected liên tiếp $w^l, w^{l+1}, w^{l+2}$. Để cắt neuron thứ $k$ nối $w^l$ và $w^{l+1}$ phải xoá cả hàng $w^l[k,:]$ và cột $w^{l+1}[:,k]$. Paper phân ra bốn kiểu phụ thuộc (Hình 2): cơ bản (a), residual (b: hai lớp cộng với nhau cùng chỉ số kênh), concatenation (c: kênh nối nhau, chỉ số bị dịch), reduction (d: phép gộp/tổng). Các kiểu này có thể lồng nhau nên không thể phân tích tay.

### 3.2 Ma trận nhóm G và đồ thị phụ thuộc D

Muốn tìm ma trận nhóm $G\in\mathbb R^{L\times L}$ với $G_{ij}=1$ nếu lớp $i,j$ phụ thuộc nhau; nhóm của lớp $i$:

$$g(i)=\{j\mid G_{ij}=1\}.$$

Nhưng $G$ khó ước lượng trực tiếp vì quan hệ giữa $i,j$ bị ảnh hưởng bởi các lớp ở giữa. Nên paper xây $D$ là **transitive reduction** của $G$: chỉ ghi phụ thuộc giữa các lớp kề trực tiếp; $G_{ij}=1$ khi và chỉ khi có đường đi từ $i$ đến $j$ trong $D$. (Ví dụ $w_1\Leftrightarrow w_2$, $w_2\Leftrightarrow w_3$ suy ra $w_1\Leftrightarrow w_3$ nên không cần lưu cạnh thứ ba.)

### 3.3 Phân rã mạng thành đầu vào và đầu ra

Vì cùng một lớp có thể có hai kiểu cắt khác nhau (FC có thể cắt hàng $w[k,:]$ hay cột $w[:,k]$) và có phép toán không tham số (cộng skip), paper phân rã mạng thành các thành phần $f_i$ (lớp có tham số hoặc phép không tham số), mỗi thành phần tách thành đầu vào $f_i^-$ và đầu ra $f_i^+$: $F=\{f_1^-,f_1^+,\dots,f_L^-,f_L^+\}$.

Hai loại phụ thuộc:
- **Inter-layer:** $f_i^-\Leftrightarrow f_j^+$ nếu hai lớp nối nhau ($f_i^-\leftrightarrow f_j^+$), vì đó chính là cùng một feature trung gian.
- **Intra-layer:** $f_i^-\Leftrightarrow f_i^+$ nếu đầu vào và đầu ra có cùng "scheme" cắt, $sch(f_i^-)=sch(f_i^+)$. BatchNorm thoả (kênh vào = kênh ra); conv thì không ($w[:,k,:,:]\ne w[k,:,:,:]$) nên không có phụ thuộc nội lớp.

$$\boxed{D(f_i^-,f_j^+)=\mathbb 1\big[f_i^-\leftrightarrow f_j^+\big]\ \vee\ \mathbb 1\big[i=j\ \wedge\ sch(f_i^-)=sch(f_j^+)\big]}$$

$D$ đối xứng.

```
# Algorithm 1: xây DepGraph
phân rã mạng thành f^-_1..f^-_L, f^+_1..f^+_L
D = zeros(2L, 2L)
for i, j:
    D[f_i^-, f_j^+] = D[f_j^+, f_i^-] =
        (f_i^- nối với f_j^+)  OR  (i == j AND sch(f_i^-) == sch(f_j^+))

# Algorithm 2: gom nhóm (thành phần liên thông)
G = {}
for mỗi đỉnh i:
    g = {i}
    repeat:
        g' = {j chưa thấy | tồn tại k in g, D[k,j] = 1}
        g = g ∪ g'
    until g' rỗng
    G = G ∪ {g}
```

Độ phức tạp tuyến tính theo số đỉnh (duyệt đồ thị). Ví dụ: Hình 3 trình bày lan truyền cắt từ $f_4^+$ trong khối CNN có residual.

### 3.4 Group-level pruning (cắt ở mức nhóm)

Với nhóm $g=\{w_1,\dots,w_{|g|}\}$, tiêu chí norm cho từng tham số $I(w)=\|w\|_2$ (kiểu Li et al., 04) có thể cộng thành $I(g)=\sum_{w\in g}I(w)$. Nhưng điểm của các lớp độc lập không cộng được vì lệch phân phối. Giải pháp: **sparse training nhất quán theo nhóm**. Với mỗi chiều có thể cắt $k$ (K chiều) của nhóm $g$:

$$I_{g,k}=\sum_{w\in g}\|w[k]\|_2^2,\qquad \boxed{R(g,k)=\sum_{k=1}^{K}\gamma_k\, I_{g,k}}$$

hệ số co $\gamma_k$ tăng theo độ "không quan trọng" tương đối:

$$\gamma_k = 2^{\alpha\,(I_g^{\max}-I_{g,k})/(I_g^{\max}-I_g^{\min})},\qquad \alpha=4$$

nên $\gamma_k\in[2^0,2^\alpha]$ (phạt mạnh nhất cho chiều ít quan trọng nhất). Sau sparse training, dùng điểm tương đối chuẩn hoá $\hat I_{g,k}=N\cdot I_{g,k}/\sum \mathrm{TopN}(I_g)$ (trong văn bản gốc công thức này bị mangled; ý là chuẩn hoá bởi tổng N giá trị lớn nhất trong nhóm) để so sánh giữa các nhóm và cắt phần nhỏ nhất. Sau đó fine-tune bằng quy trình giống pretrain, với learning rate nhỏ hơn và ít iteration hơn.

Paper cho ba chiến lược sparse (Hình 4): (a) unstructured, (b) cấu trúc nhưng **không nhất quán** (mỗi lớp độc lập), (c) **nhất quán theo nhóm** (cái họ đề xuất): ép mọi tham số cùng chỉ số trong nhóm về 0 cùng nhau.

Hai kiểu độ thưa theo lớp: **uniform** (cùng tỉ lệ mỗi lớp) và **learned** (học từ ngưỡng toàn cục dựa trên importance chuẩn hoá).

## 4. Thí nghiệm và kết quả chính

Dữ liệu: CIFAR-10/100, ImageNet-1k, PPI (đồ thị), ModelNet40 (point cloud), AGNews (văn bản). Mô hình: ResNe(X)t, VGG, DenseNet, MobileNet, GoogleNet, ViT, LSTM, DGCNN, GAT. Speed-up đo bằng tỉ lệ MACs (theo lý thuyết), không phải latency.

**CIFAR (Bảng 1, trích):**

| Mô hình / dữ liệu | Phương pháp | Base | Pruned | ΔAcc | Speed-up |
|---|---|---|---|---|---|
| ResNet56 / CIFAR-10 | HRank | 93.26 | 92.17 | -0.09 | 2.00x |
| | ResRep | 93.71 | 93.71 | +0.00 | 2.12x |
| | Ours w/o SL | 93.53 | 93.46 | -0.07 | 2.11x |
| | Ours | 93.53 | 93.77 | +0.24 | 2.11x |
| | GReg-2 | 93.36 | 93.36 | -0.00 | 2.55x |
| | Ours w/o SL | 93.53 | 93.36 | -0.17 | 2.51x |
| | Ours | 93.53 | 93.64 | +0.11 | 2.57x |
| VGG19 / CIFAR-100 | GReg-1 | 74.02 | 67.55 | -6.67 | 8.84x |
| | GReg-2 | 74.02 | 67.75 | -6.27 | 8.84x |
| | Ours w/o SL | 73.50 | 67.60 | -5.44 | 8.87x |
| | Ours | 73.50 | 70.39 | -3.11 | 8.92x |

("SL" = sparse learning.) ResNet-56 đạt 2.57x với 93.64% (cao hơn mô hình gốc 93.53%).

**ImageNet (Bảng 3, trích; cột MACs theo G, chỉ lấy các số đọc được rõ):**

| Mạng | Phương pháp | Base | Pruned | ΔAcc | MACs (G) |
|---|---|---|---|---|---|
| ResNet-50 | FPGM | 76.15 | 75.50 | -0.65 | 2.38 |
| | Taylor | 76.18 | 74.50 | -1.68 | 2.25 |
| | GFP | 76.79 | 76.42 | -0.37 | 2.04 |
| | GReg-2 | 76.13 | 75.36 | -0.77 | 2.77 |
| | Ours | 76.15 | 75.83 | -0.32 | 1.99 |
| DenseNet-121 | PSP-1.38G | 74.35 | 74.05 | -0.30 | 1.38 |
| | Ours-1.38G | 74.44 | 73.98 | -0.46 | 1.37 |
| | Ours-0.58G | 74.44 | 70.13 | -4.31 | 0.57 |
| MobileNet-v2 | Ours | 71.87 | 68.46 | -3.41 | 0.15 |
| ResNeXt-50 | GFP | 77.97 | 77.53 | -0.44 | 2.11 |
| | Ours | 77.62 | 76.48 | -1.14 | 2.09 |
| ViT-B/16 | CP-ViT | 77.91 | 77.36 | -0.55 | 11.7 |
| | Ours | 81.07 | 79.17 | -1.90 | 10.4 |
| | Ours+EMA | 81.07 | 79.58 | -1.39 | 10.4 |

(ViT-B/16 gốc 17.6 GMACs. Bảng này lưu ý: DepGraph không luôn tốt nhất; ví dụ GFP trên ResNeXt-50 mất ít hơn; ResNet-50 là điểm sáng "hơn 2x speed-up, mất 0.32%".)

**Dữ liệu phi ảnh (Bảng 4, trích):**

| Mạng / dữ liệu | Phương pháp | Base | Pruned | Δ | Speed-up |
|---|---|---|---|---|---|
| LSTM / AGNews | Ours | 92.10 | 91.75 | -0.35 | 16.28x |
| DGCNN / ModelNet40 | Ours | 92.10 | 92.02 | -0.08 | 11.98x |
| GAT / PPI (micro-F1) | Ours | 0.986 | 0.961 | -0.025 | 8.43x |

Các baseline "DepGraph+Random / +CP / +Slim" cho thấy ngay cả pruner khác chạy trên nhóm của DepGraph vẫn chạy được (ví dụ LSTM: Random 91.23, CP 91.50).

## 5. Phân tích / ablation

**Chiến lược nhóm (Bảng 2, CIFAR-100, acc trung bình sau prune ở 1.5x/3x/6x/12x; trích cột Avg. "uniform / learned"):**

| Mạng (acc gốc) | Random | No grouping | Conv-only | Full grouping |
|---|---|---|---|---|
| ResNet-56 (72.58) | 63.39 / 62.15 | 64.24 / 64.20 | 64.07 / 64.65 | 64.83 / 66.09 |
| VGG-19 (73.50) | 63.83 / 69.05 | 69.14 / 40.58 | 69.76 / 70.21 | 70.03 / 70.58 |
| DenseNet-121 (78.73) | 75.41 / 75.80 | 77.24 / 70.82 | 76.77 / 77.44 | 77.29 / 77.77 |
| MobileNetV2 (70.80) | 63.35 / 63.62 | 67.03 / 51.37 | 67.27 / 67.85 | 66.99 / 68.67 |
| GoogleNet (77.56) | 74.57 / 74.44 | 75.45 / 69.92 | 75.49 / 75.35 | 75.53 / 75.57 |

Kết luận paper rút ra:
- **Không nhóm (sparsify từng lớp riêng)** là chiến lược dễ sụp đổ với learned sparsity: VGG-19 ở 12x xuống 3.87% vì "over-prune một chiều xuống 1" (ghi chú dấu † trong bảng); trung bình 40.58.
- **Conv-only** tốt hơn, **Full grouping** (cả BN và FC cùng nhất quán) thường tốt nhất.
- **Learned sparsity** thường hơn uniform, nhưng đôi khi cắt quá tay một lớp.
- **Hình 5:** sparse training nhất quán cho phân bố norm nhóm tách bạch (nhiều nhóm gần 0), còn không nhất quán thì norm nhóm không thưa rõ.
- "Ours w/o SL" (không sparse learning) ở Bảng 1 thấp hơn "Ours" ở mọi dòng, ví dụ VGG19: 67.60 vs 70.39, cho thấy sparse training có ích thực sự.

## 6. Giới hạn

**Tác giả nêu:**
- Mục tiêu là tính tổng quát, **không phải SOTA**; chỉ dùng tiêu chí norm đơn giản.
- Bài nhấn mạnh phân loại; chưa phải generation/detection.

**Chúng ta nhận xét:**
- Speed-up báo cáo là tỉ lệ MACs lý thuyết, không đo latency; HALP (19) cho thấy hai thứ này có thể lệch nhiều.
- Phần ViT: baseline và mức MACs khác nhau giữa các phương pháp nên so sánh với CP-ViT không hoàn toàn cùng điều kiện; ma trận phụ thuộc cho LLM hiện đại (RMSNorm, RoPE, GQA, gated MLP, linear attention) không được kiểm tra.
- Cần model trace được thành đồ thị tính toán; các mô-đun có luồng điều khiển phức tạp (điều kiện, vòng lặp, cache) có thể khó.
- Sparse training thêm chi phí huấn luyện; đã thử trên dataset nhỏ-vừa, chưa trên pretrain quy mô lớn.
- Bảng nhiều dòng bị mangled trong text trích xuất; một số cột (ví dụ cột tham số/MACs) tôi chỉ chép khi đọc rõ.

## 7. Liên hệ với các bài khác

- **04 (Pruning filters):** DepGraph dùng tiêu chí norm của Li et al. và tổng quát hoá phần "cắt filter ở lớp trước thì cắt kernel lớp sau".
- **05 (Network slimming):** cùng ý "sparse training trên scale rồi cắt"; DepGraph áp dụng nhất quán qua nhóm.
- **19 (HALP):** HALP xử lý skip/group conv bằng cách gom chỉ số kênh; DepGraph là cách tổng quát và tự động. Hai bài bổ sung nhau: DepGraph nói **cái gì phải cắt cùng**, HALP nói **cắt bao nhiêu để đạt latency**.
- **26 (LLM-Pruner):** áp dụng trực tiếp ý tưởng dependency group lên LLM (cắt couple structure trong Llama).
- **13, 14 (Sixteen heads, Voita):** cắt head attention, một dạng nhóm.
- **29 (SliceGPT), 38 (Sheared LLaMA), 39 (Minitron):** các cách cắt hidden dimension / head / MLP đều phải giải quyết phụ thuộc kiểu residual mà DepGraph hình thức hoá.

## 8. Ứng dụng cho dự án (suy luận)

- **Hữu ích về công cụ.** Dự án cắt ViT 12 khối và decoder lai GatedDeltaNet + full-attention. Việc cắt "hidden size" hoặc số head/kênh bị ràng buộc bởi residual stream: cắt một chiều của residual phải cắt đồng bộ ở mọi lớp đọc/ghi nó (embed, norm, mọi projection). Một đồ thị phụ thuộc giúp **kiểm tra tính nhất quán** và tự động hoá việc này.
- **Torch-Pruning** (thư viện của tác giả) có thể thử trên ViT trước; nhưng với GatedDeltaNet (các ma trận q/k/v/beta/gate, short conv, state per head) cần kiểm tra thủ công xem đồ thị có bắt được phụ thuộc đúng không (conv chiều thời gian, nhóm head chia sẻ). Đây là suy đoán, chưa thử.
- **Cắt MLP neuron (nhóm đơn giản: hàng của up/gate và cột của down)** là trường hợp không cần DepGraph; còn cắt **layer** hoàn toàn cũng không cần. Giá trị lớn của DepGraph nằm ở các cắt ngang residual (hidden dim) và head/state.
- **Ý tưởng group-level sparse training** có thể liên quan đến quan sát "cắt MLP neuron hồi phục kém": nếu tín hiệu importance cho từng neuron nhiễu, ép sparsity nhất quán trước khi cắt (thay vì cắt tức thời bằng xếp hạng) có thể giảm sốc. Chưa có bằng chứng trong paper cho LLM/VLM.

## 9. Thí nghiệm gợi ý

1. Chạy Torch-Pruning/DepGraph trên ViT 12 khối của LightOnOCR (chỉ để dựng đồ thị), in ra các nhóm; kiểm tra tay xem nhóm của chiều residual và nhóm head có đúng, và xem đồ thị có "gãy" ở các block lạ không.
2. Trên một ViT hoặc MLP nhỏ: so sánh cắt neuron 30% bằng (a) xếp hạng norm tức thời và (b) sparse training nhất quán theo nhóm ngắn rồi cắt, đo loss trước/sau distill ngắn. Câu hỏi: sparse training có làm giảm phần "D_res" không?
3. Dựng thử đồ thị phụ thuộc cho một khối GatedDeltaNet (kể cả short conv và state) bằng tay, liệt kê chiều nào cắt được độc lập; dùng làm đặc tả cho pruner của dự án.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Nhóm phụ thuộc | $g(i)=\{j\mid G_{ij}=1\}$, các lớp phải cắt cùng nhau |
| DepGraph $D$ | Transitive reduction của $G$; chỉ ghi phụ thuộc giữa lớp kề |
| Phân rã | Mỗi thành phần $f_i$ tách thành đầu vào $f_i^-$ và đầu ra $f_i^+$ |
| Inter-layer dep | $f_i^-\Leftrightarrow f_j^+$ nếu nối nhau |
| Intra-layer dep | $f_i^-\Leftrightarrow f_i^+$ nếu cùng scheme cắt (BN có, conv không) |
| Gom nhóm | Thành phần liên thông trên $D$, độ phức tạp tuyến tính |
| Phạt nhóm | $R(g,k)=\sum_k\gamma_k\sum_{w\in g}\|w[k]\|_2^2$ |
| Hệ số co | $\gamma_k=2^{\alpha(I^{\max}-I_{g,k})/(I^{\max}-I^{\min})}$, $\alpha=4$ |
| Kết quả nổi bật | ResNet-56 CIFAR-10: 2.57x, 93.64%; ResNet-50 ImageNet: 75.83% (-0.32), ~1.99 GMACs; ViT-B/16: 79.17% (-1.90) |

## 11. Kết luận ngắn

DepGraph biến "cắt kênh nào cùng kênh nào" từ công việc thủ công theo từng kiến trúc thành một bài toán đồ thị tự động, kèm sparse training nhất quán ở mức nhóm. Giá trị chính là hạ tầng và tính tổng quát, không phải con số SOTA; với dự án, nó hữu ích như công cụ để cắt các chiều dính vào residual stream, còn các bộ phận mới như linear attention vẫn cần kiểm tra kỹ.

Link gốc: https://arxiv.org/abs/2301.12900
