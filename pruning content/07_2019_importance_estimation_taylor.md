# Importance Estimation for Neural Network Pruning (2019)

**Tác giả:** Pavlo Molchanov, Arun Mallya, Stephen Tyree, Iuri Frosio, Jan Kautz (NVIDIA).  
**Venue / arXiv:** arXiv:1906.10771 (25/06/2019). Bản text không nêu venue; mã nguồn: github.com/NVlabs/Taylor_pruning.  
**Nguồn:** https://arxiv.org/abs/1906.10771  
**Chủ đề:** Structured pruning (filter/neuron) dựa trên khai triển Taylor bậc 1 và bậc 2 của *bình phương thay đổi loss*; so khớp với "oracle"; đặt cổng (gate) sau BatchNorm để chấm điểm mọi loại layer kể cả skip connection.  
**Vị trí trong lộ trình:** Đọc sau 01 (OBD/OBS, bản gốc dùng đạo hàm bậc 2 làm điểm số), 04–06 (các tiêu chí "chủ nghĩa độ lớn", tái tạo của ThiNet). Bài này nâng tiêu chí dựa trên gradient lên mức dùng được ở quy mô ImageNet và chứng minh bằng *tương quan với oracle* rằng độ lớn trọng số là chỉ báo kém. Bài kế: 08 (Lottery Ticket) rồi 10 (SNIP, cũng dùng gradient×trọng số nhưng lúc khởi tạo).

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Số liệu lấy từ paper (một số bảng PDF bị lệch dòng, tôi chỉ ghi những số đọc được rõ). Phần "Ứng dụng cho dự án" là suy luận của chúng ta.

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

Muốn cắt một neuron/filter, ta cần biết **nó quan trọng đến đâu**. Các tiêu chí đã có:

- **Độ lớn** (|w|, ℓ2 norm của filter, hệ số scale γ của BatchNorm): rẻ, nhưng dựa trên niềm tin "trọng số nhỏ = ít quan trọng". Paper nghi ngờ niềm tin này (Mozer 1988 và LeCun 1990 cũng từng nghi ngờ).
- **OBD** (bài 01): điểm = đạo hàm bậc hai (đường chéo Hessian) × $w^2$. Giả định Hessian dương xác định; với mạng sâu huấn luyện bằng minibatch, đường chéo Hessian có phần tử âm, và Hessian tốn bộ nhớ.
- **ThiNet** (bài 06): tái tạo đầu ra layer sau theo từng layer, tốn kém tính toán.
- **Taylor-output** (Molchanov 2017): khai triển theo *activation* đầu ra; cần lưu activation (tốn bộ nhớ), phải chuẩn hóa heuristic giữa các layer, hỏng với skip connection.
- Nhiều phương pháp cần **sensitivity analysis** (thử cắt từng layer để chọn tỷ lệ riêng cho layer).

Thuật ngữ nhanh:
- **Oracle**: tiêu chuẩn "vàng": thật sự đặt từng neuron về 0 và đo loss tăng bao nhiêu (đắt vì phải chạy M lần).
- **Hệ số tương quan hạng Spearman**: đo hai bảng xếp hạng giống nhau thế nào (1 = giống hệt).
- **Gate**: hệ số nhân $z_m$ khởi tạo 1 đặt sau neuron; tắt neuron tức đặt $z_m=0$.

## 2. Ý tưởng chính

Định nghĩa độ quan trọng của một neuron là **bình phương thay đổi loss** nếu bỏ nó đi. Tính trực tiếp thì quá đắt, nên khai triển Taylor quanh trọng số hiện tại: ở bậc 1 chỉ cần gradient $g$ (sẵn có sau backprop) và trọng số $w$, điểm = $(g\cdot w)^2$. Đặt "cổng" sau mỗi layer BatchNorm để cùng một công thức dùng được cho mọi loại layer, kể cả skip connection. Cắt dần từng đợt nhỏ (khoảng 2% filter mỗi lượt) và tính lại điểm sau mỗi lượt. Kết quả: điểm số tương quan >93% (Spearman) với oracle, thang đo giống nhau giữa các layer nên không cần phân tích độ nhạy theo layer.

## 3. Phương pháp chi tiết

### 3.1 Độ quan trọng và hai bậc xấp xỉ

Mạng có tham số $W=\{w_1,\dots,w_M\}$, lỗi $E(D,W)$. Độ quan trọng của tham số $w_m$ (giả định độc lập, i.i.d.):

$$I_m=\big(E(D,W)-E(D,W\,|\,w_m=0)\big)^2 \qquad (3)$$

Khai triển Taylor bậc 2 quanh $W$:

$$I_m^{(2)}(W)=\Big(g_m w_m-\tfrac12 w_m H_m W\Big)^2\qquad (4)$$

với $g_m=\partial E/\partial w_m$, $H_m$ là hàng thứ $m$ của Hessian. Chỉ giữ bậc 1:

$$\boxed{I_m^{(1)}(W)=(g_m w_m)^2}\qquad (5)$$

Vì sao dùng *bình phương*: không giả định mọi neuron đều làm loss tăng (loại bỏ có thể làm loss giảm); paper cho rằng bình phương tương quan với oracle tốt hơn dấu (signed) của OBD và cho kết quả cắt tốt hơn.

### 3.2 Cho nhóm tham số (một filter)

Với tập $S$ (các trọng số của một filter), có hai cách gộp:

$$I^{(1)}_S(W)=\Big(\sum_{s\in S}g_sw_s\Big)^2\ \text{(group contribution, Eq. 7)},\qquad \hat I^{(1)}_S(W)=\sum_{s\in S}(g_sw_s)^2\ \text{(tổng riêng lẻ, Eq. 8)}$$

**Cổng.** Thêm $z_m=1$ sau neuron. Thì

$$I^{(1)}_m(z)=\Big(\frac{\partial E}{\partial z_m}\Big)^2=\Big(\sum_{s\in S_m}g_sw_s\Big)^2=I^{(1)}_{S_m}(W)$$

tức là điểm của cổng chính là "group contribution" của các tham số trước đó; cổng cũng gộp luôn weight và bias, không tham gia tối ưu, tiện tính.

Cổng đặt **ngay sau BatchNorm** (mặc định): với feature map $m$ có $\gamma_m,\beta_m$ của BN,

$$I^{(1)}_m(\gamma_m,\beta_m)=\Big(\gamma_m\frac{\partial E}{\partial\gamma_m}+\beta_m\frac{\partial E}{\partial\beta_m}\Big)^2 .$$

Chỉ cần $\partial E/\partial\gamma$ và $\partial E/\partial\beta$, có sẵn trong backprop thông thường, nên rất rẻ.

### 3.3 Diễn giải thông tin

Đặt $h_m=\partial E/\partial z_m$. Giả sử tại hội tụ $E[h_m]=0$, thì $\mathrm{Var}(h_m)=E[h_m^2]=I^{(1)}(z)$: điểm số chính là **phương sai của gradient**, và nếu $E$ là log-likelihood thì đó là đường chéo của ma trận thông tin Fisher kỳ vọng. Tức là tiêu chí này đồng thời đo mức độ "nhạy" của log-likelihood với neuron đó.

### 3.4 Thuật toán (Section 3.1)

```text
Input: mạng đã huấn luyện, số neuron cần cắt tổng cộng P, lr nhỏ
repeat (trong fine-tune):
    for mỗi minibatch:
        tính gradient, cập nhật trọng số (SGD)
        cộng dồn điểm quan trọng  I_m  (từ gradient trung bình minibatch)
    sau N minibatch (ImageNet: 30; CIFAR: 10):
        trung bình điểm theo thời gian (EMA, momentum 0.9)
        loại N neuron điểm thấp nhất (ImageNet: 100 neuron mỗi lượt; ≈2% filter ban đầu)
until đạt P neuron đã cắt (hoặc loss vượt ngưỡng chịu được)
reset momentum, fine-tune tiếp
```

Chi tiết quan trọng:
- Hessian dùng xấp xỉ đường chéo; trên ImageNet không tính được (hết bộ nhớ), nên chỉ dùng bậc 1 ("Taylor FO").
- **Group contribution (Eq. 7)** tương quan rất thấp với oracle khi tập $S$ lớn vì kỳ vọng gradient về 0 khi hội tụ, nên với filter không dùng cổng thì cộng điểm riêng lẻ (Eq. 8).
- Điểm tính trên toàn bộ mạng với cùng một thang, nên **xếp hạng toàn cục** và không cần chọn tỷ lệ cắt từng layer.
- Cắt theo lô cố định và fine-tune hơn là "cắt liên tục chừng nào loss còn dưới ngưỡng" (cách sau rơi vào cực tiểu địa phương, kết quả hơi tệ hơn).

### 3.5 Ví dụ nhỏ (tự đặt)

Hai filter A, B, mỗi filter 2 trọng số. A: $g=(0.1,-0.1)$, $w=(1,1)$ nên tổng $g\!\cdot\!w=0$; Eq.7 cho điểm 0 (nhóm triệt tiêu nhau), Eq.8 cho $0.01+0.01=0.02$. B: $g=(0.2,0.2)$, $w=(0.1,0.1)$ cho Eq.7 $=0.04^2=0.0016$, Eq.8 $=0.0004+0.0004$. Tiêu chí |w| sẽ chọn cắt B (trọng số nhỏ), Eq.7 chọn cắt A, hai ra kết luận khác nhau; đây là điều khiến chọn công thức quan trọng.

## 4. Thí nghiệm và kết quả chính

### 4.1 CIFAR-10: so với oracle (không fine-tune)

- **LeNet3** (73% test acc): oracle tổ hợp (thử mọi tổ hợp $k$ neuron) không tốt hơn đáng kể oracle tham lam khi cắt ít neuron, nên các thí nghiệm sau dùng oracle tham lam. Kết quả: oracle > Taylor SO ≳ Taylor FO; **weight-based không hơn cắt ngẫu nhiên**; OBD ngang oracle.
- **ResNet-18** pre-activation (94.79% acc), Spearman với oracle (Table 1): đặt cổng sau BN2: Taylor FO 0.955 (conv1), 0.811 (conv2), 0.924 (all layers); Taylor SO 0.968 / 0.849 / 0.957; trọng số conv Taylor FO 0.726 / 0.766 / 0.892; weight magnitude 0.660 / 0.677 / 0.861. Cổng đặt *trước* BN2 làm tương quan sụt mạnh (Taylor FO conv1 từ 0.955 xuống 0.275), nghĩa là BatchNorm ảnh hưởng lớn tới đóng góp của filter.
- OBD tương quan tốt nhưng cắt tệ: lúc đo tương quan, điểm OBD được bình phương; lúc cắt dùng giá trị có dấu, trong mạng sâu đường chéo Hessian có phần tử âm nên mất ổn định khi không fine-tune.
- Tính lại điểm sau mỗi lượt cắt quan trọng: biến thể "fixed" (không tính lại) kém rõ rệt.

### 4.2 ImageNet: tương quan (Table 2, trích)

Spearman tương quan với oracle, cột "All layers":

| Mạng | Gate after BN (Taylor FO) | Conv weight | BN scale | Weight magnitude | Taylor-output |
|---|---|---|---|---|---|
| ResNet-101 | 0.965 | 0.874 | 0.866 (BN scale có áp dụng cổng đặt khác nhau: 0.807 không dùng) | 0.587 | 0.287 |
| VGG11-BN | 0.999 | 0.983 | 0.968 | 0.812 | 0.968 |
| DenseNet-201 | 0.932 | 0.817 | 0.434 | 0.300 | 0.472 |

(Bảng PDF bị lệch dòng; cột BN scale ResNet-101 có hai dòng, tôi ghi cả hai giá trị 0.866/0.807 theo thứ tự xuất hiện. Phải coi là gần đúng.) Văn bản: phương pháp đạt >93% Spearman trên cả ba mạng; Pearson >92% trên ResNet và DenseNet. Taylor-output tốt trên VGG nhưng thất bại với ResNet/DenseNet. Thêm cổng cho skip connection vẫn tương quan cao (ResNet-101: 0.946).

### 4.3 ImageNet: cắt và fine-tune (Table 3, 1-crop top-1 error)

Thiết lập: PyTorch, model pretrained, 4 GPU, batch 256, SGD lr 0.01 (hoặc 0.001), giảm 10 lần mỗi 10 epoch, tổng 25 epoch; mỗi 30 minibatch cắt 100 neuron.

| Mạng | Mô hình | GFLOPs | Params (1e7) | Error % |
|---|---|---|---|---|
| ResNet-101 | Không cắt | 7.80 | 4.47 | 22.63 |
| | Taylor-FO-BN-75% | 4.70 | 3.12 | 22.65 |
| | Taylor-FO-BN-55% | 2.85 | 2.07 | 24.05 |
| | Taylor-FO-BN-50% | 2.47 | 1.78 | 24.62 |
| | Taylor-FO-BN-40% | 1.76 | 1.36 | 25.84 |
| | BN-ISTA v1 / v2 | 4.47 / 3.69 | 2.36 / 1.73 | 24.73 / 25.44 |
| ResNet-50 | Không cắt | 4.09 | 2.56 | 23.82 |
| | Taylor-FO-BN-91% | 3.27 | 2.26 | 23.57 |
| | Taylor-FO-BN-81% | 2.66 | 1.79 | 24.52 |
| | Taylor-FO-BN-72% | 2.25 | 1.42 | 25.50 |
| | Taylor-FO-BN-56% | 1.34 | 0.79 | 28.31 |
| | ThiNet-70 / ThiNet-30 | ≈2.58 / ≈1.17 | 1.69 / 0.87 | 27.96 / 31.58 |
| ResNet-34 | Không cắt | 3.64 | 2.18 | 26.69 |
| | Taylor-FO-BN-82% | 2.83 | 1.72 | 27.17 |
| | Li et al. | 2.76 | 1.93 | 27.80 |
| VGG11-BN | Không cắt | 7.61 | 13.29 | 29.16 |
| | Taylor-FO-BN-50% | 6.93 | 3.18 | 29.35 |
| DenseNet-201 | Không cắt | 4.29 | 2.20 | 23.20 |
| | Taylor-FO-BN-60% | 3.02 | 1.25 | 23.49 |

Điểm nhấn: ResNet-101 giảm 40% FLOPs và 30% tham số chỉ mất 0.02% top-1. VGG11-BN cắt 76% tham số chỉ mất 0.19%, hơn "From scratch" (30.00%) và Slimming (31.38%) trên cùng cấu hình. DenseNet-201 cắt đến FLOPs bằng DenseNet-121 (Taylor-FO-BN-52%) cho lỗi thấp hơn 1.79%. Bảng PDF có nhãn "No pruning" lệch dòng trong ResNet-101; số "75%" tôi đối chiếu với Table 5.

### 4.4 Tốc độ suy luận (Table 5, ResNet-101, Tesla V100, ms)

| Mô hình | GFLOPs | Error % | B16 | B256 |
|---|---|---|---|---|
| Không cắt | 7.80 | 22.63 | 29.0 | 379.8 |
| Taylor-FO-BN-50% | 2.47 | 24.62 | 20.9 | 251.4 |
| Taylor-FO-BN-40% | 1.76 | 25.84 | 21.0 | 223.4 |
| chỉ cắt skip, 22% | 5.19 | 24.77 | 19.5 | 239.3 |

Chỉ cắt skip connection (bỏ 33% FLOPs) cho tăng tốc 1.59x, hơn cắt toàn bộ bỏ 68% FLOPs chỉ 1.51x: FLOPs không phải là độ trễ.

## 5. Phân tích / ablation

- **Cổng sau BN** > áp lên trọng số conv > BN scale > |w| về tương quan với oracle (Table 2).
- **Trung bình gradient minibatch** (group) hơn "full gradient" từng mẫu, dù paper chưa giải thích chắc chắn.
- **Cắt lặp > cắt một bước**: cắt 10,000 neuron một lần sau 3000 minibatch cho top-1 error 25.3%, kém Taylor-FO-BN-50% 0.68 điểm.
- **Lịch cắt:** iterative tốt hơn "single step" và "continuous" ở mọi epoch (supplement).
- **Bỏ cắt skip connection** (No skip): ResNet-50 56% tăng lỗi từ 28.31% lên 30.74%.
- **ResNet20/CIFAR-10 (Table 4):** Random và magnitude là tệ nhất; Taylor FO/SO và OBD khác nhau trong 1 độ lệch chuẩn (≈91.5–91.6% khi giữ ~30% neuron của "Prune A"), cao hơn BN-ISTA 90.9%. Với "train from scratch" cùng cấu trúc thì thấp hơn (89.6%). Tác giả nói phương pháp không bị "pruning paradox" của bài 11.
- Sau khi cắt, thứ hạng điểm giữa các layer trở nên "đồng đều hơn" (Figure 5).

## 6. Giới hạn

**Do tác giả nêu:**
- Không tính được Hessian trên ImageNet, chỉ dùng bậc 1.
- Chất lượng điểm số phụ thuộc vị trí cổng (sau BN tốt hơn nhiều so với trước BN).
- Cần số minibatch đủ giữa các lượt cắt để thống kê ổn định; số neuron cắt mỗi lượt phụ thuộc vào mức tương quan giữa chúng.

**Do chúng tôi nhận xét:**
- Giả định độc lập giữa các neuron (i.i.d.); các neuron cùng cắt có thể tương tác (hiệu ứng nhóm), được xử lý phần nào bằng cách tính lại sau mỗi lượt.
- Toàn bộ thí nghiệm là CNN phân loại ảnh có BatchNorm; công thức cổng "sau BN" không áp dụng nguyên si cho transformer dùng LayerNorm/RMSNorm.
- Bậc 1 tại điểm đã hội tụ có gradient kỳ vọng ≈ 0; paper dựa vào bình phương của gradient từng minibatch (biến thiên), tức là điểm gần như đo phương sai của gradient chứ không đo đúng "thay đổi loss"; với dữ liệu hiệu chỉnh nhỏ, điểm số có thể nhiễu.
- Đánh giá chủ yếu bằng độ chính xác sau khi fine-tune 25 epoch bằng dữ liệu huấn luyện đầy đủ; chưa xét trường hợp ít dữ liệu hoặc chưng cất.

## 7. Liên hệ với các bài khác

- **01 OBD/OBS:** Taylor SO chính là phiên bản OBD có bình phương; paper cho thấy dấu của OBD gây bất ổn ở mạng sâu.
- **04, 05:** Li et al. (tổng |w|) và Slimming (BN γ) là các đường cơ sở dựa trên độ lớn; ở đây cả hai đều thua tiêu chí Taylor về tương quan oracle.
- **06 ThiNet:** thua ở Table 3 (ThiNet-70 27.96% vs Taylor-FO-BN-72% 25.50% với FLOPs và tham số gần nhau).
- **10 SNIP:** cùng ý "độ nhạy = gradient × trọng số", nhưng SNIP dùng ngay khi khởi tạo, ở mức từng kết nối.
- **11 Rethinking:** bài 11 đưa ra "pruning paradox" mà bài này tuyên bố không bị.
- **16 Movement pruning, 26 LLM-Pruner, 27 LLM Surgeon, 28 FLAP, 30 MaskLLM:** nhiều bài LLM dùng điểm quan trọng dựa trên gradient/Fisher (LLM-Pruner dùng Taylor trên các nhóm phụ thuộc); bài này là tổ tiên trực tiếp.
- **20 DepGraph:** cần nhóm các tham số phụ thuộc để áp "điểm theo nhóm" (như S trong Eq. 7/8).

## 8. Ứng dụng cho dự án (suy luận)

- **Chọn neuron MLP/head để cắt bằng gradient** là hướng hợp lý thay vì |w|: bài này là bằng chứng (trên CNN) rằng độ lớn gần như không dự báo được tác động. Đây là điểm phù hợp với quan sát của dự án rằng cắt neuron khó hồi phục: cắt theo |w| có thể đang cắt sai neuron.
- **Cổng cho mỗi neuron trung gian MLP** (hệ số $z_m$ nhân vào kích hoạt giữa $W_{up}$ và $W_{down}$) cho điểm $(\partial L/\partial z_m)^2$ chỉ bằng một lượt backward; có thể lấy trung bình trên calibration set OCR hoặc trên **loss chưng cất** (KL với teacher 4B) thay vì loss nhiệm vụ.
- **Xếp hạng toàn cục:** dùng chung thang đo cho ViT, MLP của decoder, các head attention, giúp phân bổ ngân sách giữa ViT và decoder để đạt ~0.3B/0.1B mà không phải tự đặt tỷ lệ từng layer.
- **Cắt dần theo lượt + fine-tune** gần với kịch bản "cắt từng đợt rồi distill"; có thể kết hợp mỗi đợt cắt ~2% rồi vài trăm bước chưng cất.
- **Lưu ý D_lin / D_res:** điểm Taylor bậc 1 đo ảnh hưởng *thẳng* đến loss mà không phân biệt phần được bù tuyến tính hay không; đối chiếu hai đại lượng có thể cho biết neuron nào "quan trọng nhưng bù được".
- **GatedDeltaNet / layer tuyến tính:** gate vẫn đặt được trên đầu ra kênh, nhưng với trạng thái hồi quy, gradient qua thời gian dài có thể nhiễu; cần kiểm tra.
- **Hạn chế thực dụng:** bài này không xét độ trễ batch-1; Table 5 nhắc rằng cắt kênh sâu hơn không nhất thiết nhanh hơn, cần đo thực.

## 9. Thí nghiệm gợi ý

1. **Oracle nhỏ:** trên một khối MLP của decoder, tắt từng neuron (hoặc từng nhóm 64 neuron) và đo ΔKL so với teacher; tính Spearman với (a) |w| (b) $(g\cdot w)^2$ trên cổng (c) Fisher đường chéo. Kiểm tra liệu |w| có tương quan thấp như trên CNN không.
2. **Cắt lặp vs một lần:** cắt 30% neuron MLP bằng điểm Taylor trong 1 bước hoặc 15 bước (2% mỗi bước, tính lại điểm), cùng ngân sách distill, so sánh độ chính xác OCR.
3. **Điểm theo mục tiêu chưng cất:** so sánh điểm tính từ CE nhiệm vụ với điểm tính từ KL-teacher; xem chúng chọn cùng neuron đến mức nào.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| $I_m=(E(W)-E(W|w_m=0))^2$ | Oracle: bình phương thay đổi loss khi bỏ $w_m$ |
| $I^{(1)}_m=(g_mw_m)^2$ | Taylor bậc 1: chỉ cần gradient và trọng số |
| $I^{(2)}_m=(g_mw_m-\tfrac12 w_mH_mW)^2$ | Taylor bậc 2 (cần Hessian, dùng đường chéo) |
| Group contribution | $(\sum_{s\in S}g_sw_s)^2$, tương đương điểm của cổng |
| Tổng riêng lẻ | $\sum_{s\in S}(g_sw_s)^2$ |
| Gate sau BN | $(\gamma\,\partial E/\partial\gamma+\beta\,\partial E/\partial\beta)^2$ |
| Var của gradient | Điểm bậc 1 = phương sai gradient = đường chéo Fisher (giả sử $E[h]=0$) |
| Taylor-FO-BN-X% | Phiên bản cắt đến khi còn lại X% neuron |
| Spearman | Tương quan hạng giữa điểm tiêu chí và oracle |

## 11. Kết luận ngắn

Bài này chứng minh rằng điểm quan trọng dựa trên gradient×trọng số, đặt tại cổng sau BatchNorm và tính lại sau mỗi đợt cắt nhỏ, bám sát oracle (Spearman >93%) và hơn rõ ràng các tiêu chí độ lớn, ThiNet, BN-ISTA trên ImageNet (ví dụ ResNet-101 mất 0.02% khi bỏ 40% FLOPs). Điều cần mang theo: dùng thước đo tác động lên loss thay cho độ lớn, xếp hạng toàn cục, cắt lặp, và luôn đo tốc độ thật.

Link gốc: https://arxiv.org/abs/1906.10771
