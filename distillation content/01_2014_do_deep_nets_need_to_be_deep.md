# Do Deep Nets Really Need to be Deep? (2014)

**Tác giả:** Lei Jimmy Ba, Rich Caruana  
**Venue / arXiv:** arXiv:1312.6184 (v7, 11/10/2014); bản trong file ghi "Draft for NIPS 2014" (NIPS 2014, bản draft chưa phải camera-ready)  
**Nguồn:** https://arxiv.org/abs/1312.6184  
**Chủ đề:** Model compression / mimic learning: huấn luyện mạng nông (shallow) bắt chước mạng sâu bằng cách hồi quy logit với L2 loss; câu hỏi "độ sâu có thực sự cần thiết?".  
**Vị trí trong lộ trình:** Đọc ngay sau bài 00 (preface). Đây là bài "tiền thân" của Hinton KD (02): nó cho thấy student học từ output của teacher có thể tốt hơn học từ nhãn gốc. Bài này đóng góp: (i) dùng **logit** làm target (không phải xác suất), (ii) bằng chứng thực nghiệm rằng khoảng cách shallow vs deep phần lớn là vấn đề *học/regularization* chứ không phải vấn đề *biểu diễn*. Tiếp theo: 02 (soft target + nhiệt độ T, tổng quát hoá việc "regress logit"), 03 (FitNets, bắt chước cả hidden layer).

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch paper. Phần "Ứng dụng cho dự án" và "Thí nghiệm gợi ý" là suy luận của chúng ta, không phải kết luận của paper.

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

**Câu hỏi của paper:** Mạng sâu (deep) cho độ chính xác cao hơn mạng nông (một hidden layer) với cùng dữ liệu. Nguyên nhân là gì? Paper liệt kê các giả thuyết: (a) sâu có nhiều tham số hơn; (b) sâu học hàm phức tạp hơn với cùng số tham số; (c) sâu có "bias" tốt hơn, học biểu diễn phân cấp; (d) mạng không có convolution không học được cái mạng có convolution học được; (e) thuật toán học và regularization hiện tại hợp với mạng sâu hơn.

**Nguồn gốc ý tưởng (từ kiến thức của chúng ta, chưa đọc bài gốc):** Buciluǎ, Caruana, Niculescu-Mizil, "Model Compression" (KDD 2006). Bài này được Ba & Caruana trích dẫn là [3]. Ý tưởng: một ensemble lớn rất chính xác nhưng chậm; dùng nó để gán nhãn cho một lượng lớn dữ liệu *không nhãn* (nhiều khi là dữ liệu sinh tổng hợp), rồi huấn luyện một mạng nhỏ trên các nhãn đó. Theo mô tả của Ba & Caruana, mạng nhỏ có thể có số tham số ít hơn tới 1000 lần mà vẫn gần bằng độ chính xác ensemble. *(Ghi chú: chi tiết ngoài những gì Ba & Caruana tóm tắt là kiến thức nền của chúng ta, chưa được đối chiếu với bài 2006.)*

**Thuật ngữ cần biết:**
- **Teacher / student (mimic model):** teacher là mô hình lớn đã huấn luyện; student/mimic là mô hình nhỏ học bắt chước teacher.
- **Logit:** vector giá trị trước softmax, $z$; xác suất $p_k = e^{z_k}/\sum_j e^{z_j}$.
- **Shallow net (SNN):** mạng chỉ một hidden layer (rất rộng), có thể có thêm một lớp tuyến tính bottleneck.
- **Ensemble (ECNN):** trung bình nhiều mô hình; chính xác hơn nhưng tốn tính toán.

## 2. Ý tưởng chính

Trước tiên huấn luyện một mô hình sâu rất tốt (hoặc ensemble của nhiều mô hình sâu). Sau đó dùng nó gán nhãn cho dữ liệu, nhưng **nhãn ở đây là logit** của teacher. Huấn luyện một mạng nông, rộng, bằng hồi quy L2 lên các logit đó. Kết quả bất ngờ: mạng nông học từ logit của teacher đạt độ chính xác gần bằng mạng sâu có cùng số tham số, trong khi cùng kiến trúc nông học từ nhãn gốc thì kém hơn rõ. Từ đó paper suy ra: *độ phức tạp của hàm được học* và *kích thước biểu diễn tốt nhất để học hàm đó* là hai thứ khác nhau; mạng sâu có thể không "cần" sâu để biểu diễn, mà chỉ "cần" sâu để dễ học.

## 3. Phương pháp chi tiết

### 3.1 Quy trình (model compression)

1. Huấn luyện teacher bình thường (softmax + cross-entropy trên nhãn gốc).
2. Cho dữ liệu (có thể bỏ nhãn gốc, hoặc dữ liệu không nhãn thêm) đi qua teacher, thu logit $z$.
3. Huấn luyện student trên cặp $(x, z)$; student không nhìn nhãn gốc.

Nếu student bắt chước hoàn hảo thì nó tạo đúng dự đoán và đúng lỗi của teacher.

### 3.2 Vì sao hồi quy logit thay vì xác suất

Paper đưa ví dụ: teacher dự đoán xác suất $[2\cdot10^{-9},\ 4\cdot10^{-5},\ 0.9999]$. Nếu dùng cross-entropy lên xác suất, student chỉ chú ý lớp thứ ba, bỏ qua hai lớp còn lại. Logit tương ứng là $[10, 20, 30]$ (paper lấy làm ví dụ minh hoạ), student phải học cả hiệu giữa lớp 1 và lớp 2. Thêm ví dụ: logit $[-10, 0, 10]$ cho cùng xác suất sau softmax như $[10,20,30]$ (softmax bất biến khi cộng một hằng số), nhưng teacher "nội bộ" mô hình hoá hai trường hợp rất khác nhau. Hồi quy logit giữ lại thông tin này, không bị mất ở bước qua softmax.

Hàm mất mát SNN-MIMIC, với dữ liệu $\{(x^{(t)}, z^{(t)})\}_{t=1}^{T}$:

$$
\boxed{L(W,\beta)=\frac{1}{2T}\sum_{t}\left\| g(x^{(t)};W,\beta)-z^{(t)}\right\|_2^2},\qquad g(x)=\beta f(Wx)
$$

trong đó $W$ là trọng số input-hidden, $\beta$ là trọng số hidden-output, $f$ là phi tuyến.

Paper nói đã thử các loss khác như $\mathrm{KL}(p_{teacher}\,\|\,p_{student})$ và L2 trên xác suất; **hồi quy logit với L2 tốt hơn tất cả các loss khác** và là một trong các kỹ thuật chính (đây là kết quả báo cáo trong text, không có bảng so sánh riêng trong bản này). Ngoài ra chuẩn hoá logit của teacher (trừ mean, chia std theo từng target trên tập train) cải thiện L2 loss một chút khi huấn luyện, nhưng "không quan trọng".

### 3.3 Lớp tuyến tính bottleneck để tăng tốc

Mạng nông muốn có nhiều tham số phải có rất nhiều hidden unit, ma trận $W\in\mathbb R^{H\times D}$ rất lớn, hội tụ chậm (nhiều tuần dù có GPU). Giải pháp: phân rã $W = UV$ với $U\in\mathbb R^{H\times k}$, $V\in\mathbb R^{k\times D}$, $k\ll D,H$:

$$
L(U,V,\beta)=\frac{1}{2T}\sum_t\left\|\beta f(UVx^{(t)})-z^{(t)}\right\|_2^2
$$

Bộ nhớ giảm từ $O(HD)$ xuống $O(k(H+D))$. Paper nhấn mạnh lớp tuyến tính này chỉ giảm (không tăng) năng lực biểu diễn và sau khi học có thể gộp lại thành một $W$. Đây là low-rank factorization giúp *tối ưu hoá* nhanh hơn, không phải giúp biểu diễn.

### 3.4 Pseudo-code

```
# Model compression (mimic learning), theo mô tả paper
teacher = train_deep_model(D_train)                 # CE + softmax; có thể là ensemble
Z = { logit(teacher, x) for x in D_transfer }        # logit, không phải softmax
if ensemble: z = mean over members of logit_m(x)     # paper: trung bình logit
(optional) z = (z - mean_k) / std_k                  # chuẩn hoá theo từng target
student = Shallow(x -> linear(k) -> ReLU(H) -> out)  # lớp linear k chỉ để tăng tốc
for (x, z) in minibatches(D_transfer):
    loss = 0.5 * || student(x) - z ||^2
    SGD+momentum step
# Student KHÔNG dùng nhãn gốc
```

## 4. Thí nghiệm và kết quả chính

### 4.1 TIMIT (nhận dạng âm vị)

Dữ liệu: 462 speaker train, 50 speaker dev, 24 speaker test; đầu vào 1845 chiều (filter-bank 40 hệ số + energy, cùng đạo hàm bậc 1, 2, cửa sổ ±7 frame); 61 âm vị, mỗi âm vị 3 trạng thái, tức 183 target (khi chấm điểm map về 39 lớp). Vì không có dữ liệu không nhãn từ phân phối TIMIT, nhóm dùng lại chính 1.1M điểm train (bỏ nhãn) làm tập chuyển. Paper nhận xét điều này làm giảm độ chính xác student; để bù, student bắt chước ECNN (ensemble 9 CNN).

Phone error rate (PER, càng thấp càng tốt) trên test core, trích Table 1:

| Mô hình | Tham số | Hidden unit | PER |
|---|---|---|---|
| SNN-8k (train trên nhãn gốc, dropout) | ~12M | ~8k | 23.1% |
| SNN-50k (nhãn gốc, dropout) | ~100M | ~50k | 23.0% |
| SNN-400k (nhãn gốc, dropout) | ~180M | ~400k | 23.6% |
| DNN (2k-2k-2k, dropout) | ~12M | ~6k | 21.9% |
| CNN (c-p-2k-2k-2k, dropout) | ~13M | ~10k | 19.5% |
| ECNN (ensemble 9 CNN) | ~125M | ~90k | 18.5% |
| SNN-MIMIC-8k (250L-8k, không conv/pool) | ~12M | ~8k | 21.6% |
| SNN-MIMIC-400k (250L-400k, không conv/pool) | ~180M | ~400k | 20.0% |

Đọc kết quả: SNN-MIMIC-8k (21.6%) đạt ngang/hơn DNN cùng số tham số (21.9%); SNN-MIMIC-400k (20.0%) gần CNN (19.5%) dù không có convolution. Các SNN huấn luyện trên nhãn gốc kém DNN từ 1.4% đến 2%, kém CNN 3.5–4.1%, kém ECNN 4.5–5.1% (paper nêu trong text).

Từ Figure 1 (mô tả của paper, không có số cụ thể trong text): ở ~1M tham số, SNN, DNN và SNN-MIMIC tương đương; SNN-MIMIC khớp DNN tới ~4M tham số rồi DNN tụt lại; DNN bão hoà quanh ~10M, mimic tiếp tục tăng và bão hoà ở ~100M tham số, độ chính xác tương đương CNN. Mimic không bao giờ đạt ECNN (do thiếu dữ liệu không nhãn).

### 4.2 CIFAR-10

Tập chuyển: 50,000 ảnh CIFAR-10 + 1 triệu ảnh đầu của 80 million tiny images = 1.05M. Student có **một lớp conv + max-pool** rồi 1200 unit tuyến tính và 30k unit phi tuyến ("64c-p-1200L-30k" hoặc "128c-p-1200L-30k"); paper lưu ý mạng không convolution hoạt động kém trên CIFAR-10 bất kể độ sâu nên cho phép một lớp conv làm bộ trích đặc trưng. Lỗi phân loại (Table 2):

| Mô hình | Tham số | Lỗi |
|---|---|---|
| DNN 2000-2000 + dropout | ~10M | 57.8% |
| SNN-30k (128c-p-1200L-30k, dropout) | ~70M | 21.8% |
| CNN [11] 64c-p-64c-p-64c-p-16lc | ~10k | 15.6% |
| CNN [21] (stochastic pooling) | ~56k | 15.13% |
| teacher CNN 128c-p-128c-p-128c-p-1000fc | ~35k | 12.0% |
| ECNN (ensemble 4 CNN) | ~140k | 11.0% |
| SNN-CNN-MIMIC-30k (64c-p-1200L-30k) | ~54M | 15.4% |
| SNN-CNN-MIMIC-30k (128c-p-1200L-30k) | ~70M | 15.1% |
| SNN-ECNN-MIMIC-30k (bắt chước ensemble) | ~70M | 14.2% |

(Cột "Tham số" và "Hidden units" trong text trích ra có đơn vị hơi khó đọc cho các CNN nhỏ; chỉ dùng như tham khảo.) Điểm đáng nói: mimic từ 1 CNN đạt 15.1%, từ ensemble đạt 14.2% (cải thiện thêm 0.9%).

## 5. Phân tích / ablation

**Vì sao mimic có thể tốt hơn học từ nhãn gốc (mục 5.1 paper):**
1. Teacher có thể "kiểm duyệt" nhãn sai (TIMIT có frame bị gán sai bởi HMM forced alignment).
2. Ở vùng $p(y|x)$ khó học, teacher đưa nhãn mềm đơn giản hơn, làm phẳng độ phức tạp của dữ liệu.
3. Nhãn 0/1 chỉ có 1 trong 183 output khác 0; teacher cho target khác 0 ở hầu hết output, dàn trải độ bất định khi không chắc và tập trung khi dễ. Lợi ích này được tăng cường khi dùng logit.

Paper xem các cơ chế này như **regularization**: SNN huấn luyện trên nhãn gốc overfit sớm hơn mạng sâu (kể cả với dropout; Figure 2), còn mimic thì không. Gợi ý của paper: nếu có regularizer tốt hơn cho mạng nông thì khoảng cách có thể biến mất.

**Capacity (mục 5.2):** hai mimic (8k và 160k hidden unit) trên các teacher có độ chính xác khác nhau cho thấy độ chính xác student tăng đều khi teacher tốt hơn (các đường song song đường chéo), SNN-MIMIC-160k luôn hơn 8k nhưng 8k có thể bằng 160k nếu học từ teacher tốt hơn. Kết luận paper: "little evidence that shallow models have limited capacity"; nút cổ chai là thủ tục học và regularization.

**Song song vs tuần tự (5.3):** mạng sâu mất 8–12 giờ trên GTX 580, mimic nông chỉ 1–2 giờ dù nhiều tham số hơn; khi suy luận với tài nguyên song song, mạng nông chỉ cần 2–3 "chu kỳ" thay vì tỷ lệ với độ sâu, lợi cho real-time.

**Cái paper KHÔNG ablation chi tiết:** không có bảng so sánh định lượng cho các loss (KL, L2 prob, L2 logit) trong bản này, chỉ có khẳng định trong text.

## 6. Giới hạn

**Do tác giả nêu:**
- Thuật toán "awkward": phụ thuộc vào một tập dữ liệu không nhãn lớn hoặc teacher rất chính xác, hoặc cả hai.
- Với TIMIT phải tái sử dụng tập train làm tập chuyển nên student không bao giờ đạt teacher (ECNN).
- Mục tiêu thực sự (huấn luyện mạng nông chính xác trực tiếp từ dữ liệu gốc) chưa đạt.

**Do chúng ta nhận thấy:**
- Chỉ thử trên TIMIT và CIFAR-10, các mạng feed-forward cỡ ~10M–180M tham số; không có Transformer, không có chuỗi tự hồi quy.
- Student "nông" ở đây nhiều tham số hơn (hàng chục đến hàng trăm triệu) so với teacher CNN trên CIFAR-10; "cùng số tham số" chỉ đúng ở TIMIT (SNN-MIMIC-8k ~12M so với DNN ~12M). Về chi phí suy luận, mạng nông rộng 400k unit không rẻ.
- Chưa có kiểm soát kiểu "cùng teacher + cùng dữ liệu nhưng student sâu": không tách được hiệu ứng "độ sâu" khỏi hiệu ứng "học từ teacher" một cách sạch.
- Kết luận "độ sâu không cần thiết" mạnh hơn bằng chứng; về sau (ví dụ bài 41 và 10) người ta cho thấy khoảng cách capacity vẫn quan trọng khi student quá nhỏ.

## 7. Liên hệ với các bài khác

- **02 Hinton KD (2015):** tổng quát hoá ý tưởng. Hinton chỉ ra rằng regress logit là *trường hợp giới hạn* của KD với nhiệt độ cao (khi $T$ lớn so với độ lớn logit, gradient xấp xỉ $\frac{1}{NT^2}(z_i - v_i)$ sau khi trung tâm hoá logit). Tức L2 trên logit (bài này) và KL trên soft target ở nhiệt độ cao (bài 02) cùng họ.
- **03 FitNets:** bắt chước thêm hidden layer, trả lời câu hỏi "student *sâu và mảnh* thì sao" ngược với kết luận "nông" ở bài này.
- **05 KD survey, 09 Efficacy of KD, 10 Does KD really work:** các bài sau phân tích kỹ khi nào bắt chước logit/xác suất giúp được và khi nào không.
- **15 Sequence-level KD, 36 Distilling step-by-step, 37 Orca:** dùng teacher gán nhãn cho dữ liệu chưa nhãn (data-level distillation) là hậu duệ trực tiếp của "pass unlabeled data through the large model" ở bài này.
- **41 Capacity gap law:** kiểm chứng định lượng rằng student không thể nhỏ vô hạn.

## 8. Ứng dụng cho dự án (suy luận)

- **Dữ liệu gán nhãn bởi teacher 4B:** ý tưởng cốt lõi, "cho dữ liệu không nhãn qua teacher để lấy target", áp dụng trực tiếp: với OCR, ta có thể dùng ảnh tài liệu không nhãn (hoặc bản augment của tập train) và lấy logit/phân phối token từ teacher 4B làm target cho student 0.3B/0.1B. Phù hợp khi nhãn gốc ít hoặc nhiễu.
- **Logit thay vì xác suất:** gợi ý dùng MSE trên logit như một loss phụ rẻ, hoặc ít nhất lưu ý rằng chỉ xác suất sau softmax mất thông tin "mức tuyệt đối" của logit. Với từ vựng hàng chục nghìn token, MSE logit nhạy với đại lượng không quan trọng (cộng hằng số), nên cần trung tâm hoá logit (trừ trung bình theo vocab) trước khi so; paper tự không bàn tới điểm này vì chỉ có 183 lớp.
- **Bottleneck tuyến tính $W=UV$:** cùng họ với việc phân rã low-rank khi cắt neuron MLP. Điều này liên hệ gián tiếp với quan sát của chúng ta rằng cắt neuron MLP khó hồi phục: paper chỉ ra bottleneck tuyến tính *không* tăng năng lực mà chỉ tăng tốc tối ưu, gợi ý rằng thứ giúp phục hồi sau khi nén thường là tối ưu hoá tốt hơn, không phải kiến trúc. Đây là suy diễn của chúng ta, không phải kết luận của paper.
- Kết luận "độ sâu không cần" ít chuyển sang LightOnOCR: decoder hybrid GatedDeltaNet + attention và ViT 12 block là Transformer, bài này không chứa thông tin về pruning layer vs neuron. Về D_lin/D_res: không có liên hệ trực tiếp.

## 9. Thí nghiệm gợi ý

1. **So loss trên cùng teacher/student:** cố định student (đã prune) và tập dữ liệu; so KL trên token vs MSE logit (đã trung tâm hoá) vs kết hợp, xem loss nào phục hồi nhanh nhất ở cả cắt layer và cắt neuron MLP.
2. **Nhãn gốc vs nhãn teacher:** huấn luyện cùng student prune với (a) chỉ nhãn gốc, (b) chỉ output teacher, (c) cả hai; tái hiện nhận định "student học từ teacher ít overfit hơn" trên tập OCR nhỏ.
3. **Tăng lượng dữ liệu không nhãn:** vẽ độ chính xác student theo lượng ảnh không nhãn được teacher gán nhãn (10k, 100k, 1M) để xem điểm bão hoà, tương tự quan sát "mimic không bao giờ đạt teacher vì thiếu dữ liệu".

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý nghĩa |
|---|---|
| Softmax | $p_k = e^{z_k}/\sum_j e^{z_j}$ |
| Loss mimic (hồi quy logit) | $\frac{1}{2T}\sum_t\|g(x^{(t)})-z^{(t)}\|_2^2$ |
| Mạng nông | $g(x)=\beta f(Wx)$ |
| Bottleneck tuyến tính | $W=UV$, $U\in\mathbb R^{H\times k}$, $V\in\mathbb R^{k\times D}$; bộ nhớ $O(HD)\to O(k(H+D))$ |
| Chuẩn hoá logit | trừ mean, chia std theo từng target trên tập train |
| Ensemble teacher | trung bình logit của các thành viên |
| Model compression | gán nhãn dữ liệu không nhãn bằng teacher, huấn luyện student trên đó |

## 11. Kết luận ngắn

Paper là bằng chứng thực nghiệm sớm rằng cái khó của mạng nông không nằm ở khả năng biểu diễn mà ở việc huấn luyện: khi được học từ logit của teacher trên nhiều dữ liệu, mạng một hidden layer rộng khớp mạng sâu cùng số tham số trên TIMIT (21.6% vs 21.9% PER) và tiến sát CNN (20.0% vs 19.5%). Với người học distillation, hai bài học đáng giữ: (1) target của teacher chứa nhiều thông tin hơn nhãn cứng và đóng vai trò regularizer; (2) dữ liệu không nhãn do teacher gán nhãn là nguyên liệu chính. Nguồn gốc ý tưởng là Buciluǎ et al. 2006 (chưa đọc, chỉ biết qua trích dẫn).

**Link gốc:** https://arxiv.org/abs/1312.6184
