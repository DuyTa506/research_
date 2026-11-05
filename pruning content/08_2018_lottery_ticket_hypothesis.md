# The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks (2018)

**Tác giả:** Jonathan Frankle, Michael Carbin (MIT CSAIL).  
**Venue / arXiv:** arXiv:1803.03635 (v1 09/03/2018; bản dùng ở đây là v5, 04/03/2019); xuất bản tại ICLR 2019.  
**Nguồn:** https://arxiv.org/abs/1803.03635  
**Chủ đề:** Giả thuyết "vé số": mạng dày khởi tạo ngẫu nhiên chứa mạng con thưa (winning ticket) mà nếu huấn luyện riêng từ *khởi tạo gốc* thì đạt độ chính xác tương đương với số bước tương đương; thuật toán tìm vé bằng cắt độ lớn lặp + reset về khởi tạo.  
**Vị trí trong lộ trình:** Đọc sau 02 (Han 2015, cắt độ lớn rồi fine-tune), 03 (cắt dần), 07. Bài này đặt câu hỏi khác: không phải "cắt thế nào để nén mô hình đã train" mà "mạng thưa có train được từ đầu không". Bài kế: 09 (linear mode connectivity, sửa lại phương pháp cho mạng lớn), 10 (SNIP, tìm mạng con *trước* khi train), 11 (Rethinking: phản biện).

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Số liệu lấy từ paper (phần chính, một vài chi tiết phụ lục). Phần "Ứng dụng cho dự án" là suy luận của chúng ta.

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

Pruning (LeCun 1990, Han 2015...) giảm hơn 90% tham số mà không hại độ chính xác. Câu hỏi tự nhiên: nếu mạng nhỏ đủ tốt, sao không *train mạng nhỏ đó từ đầu* cho rẻ? Kinh nghiệm thời đó: kiến trúc thu được từ cắt thì **khó train từ đầu**, đạt độ chính xác thấp hơn mạng gốc. Paper trích Li et al. 2016 và Han et al. 2015 nói rằng cắt xong phải giữ lại trọng số đã học, không nên khởi tạo lại.

Hình 1 của paper minh họa: lấy mẫu ngẫu nhiên mạng con thưa của Lenet (MNIST) và Conv-2/4/6 (CIFAR-10); càng thưa thì học càng chậm và độ chính xác cuối càng thấp.

Khái niệm nền:
- **Unstructured pruning**: đặt từng trọng số về 0 bằng mặt nạ nhị phân $m\in\{0,1\}^{|\theta|}$; mạng thưa nhưng không nhanh hơn trên phần cứng thông thường.
- **Early-stopping iteration**: số bước tại đó loss validation nhỏ nhất; paper dùng làm đại diện cho "tốc độ học".
- **Khởi tạo Glorot**: cách khởi tạo Gaussian chuẩn hóa theo kích thước layer.

## 2. Ý tưởng chính

Một mạng dày khởi tạo ngẫu nhiên giống như mua rất nhiều vé số: trong đó có một số mạng con (mặt nạ cộng với các giá trị khởi tạo của chúng) "trúng" tức là train riêng ra đã học nhanh và chính xác như mạng đầy đủ. Cách tìm vé: train mạng, cắt các trọng số có độ lớn nhỏ nhất, rồi **đặt các trọng số còn lại về giá trị lúc khởi tạo ban đầu** (không phải giá trị đã train, cũng không phải giá trị khởi tạo mới), và train lại. Cùng mặt nạ nhưng khởi tạo ngẫu nhiên mới thì thua rõ rệt, nghĩa là *cấu trúc thưa một mình chưa đủ, giá trị khởi tạo cũng quan trọng*. Paper còn đưa ra một phỏng đoán chưa kiểm chứng: SGD thực chất tìm và train một mạng con được khởi tạo tốt, và mạng dày dễ train vì chứa nhiều ứng viên vé hơn.

## 3. Phương pháp chi tiết

### 3.1 Phát biểu giả thuyết

Cho mạng $f(x;\theta)$ với $\theta=\theta_0\sim D_\theta$. Train bằng SGD đạt loss validation nhỏ nhất $l$ tại bước $j$ với độ chính xác test $a$. Train $f(x;m\odot\theta_0)$ với mặt nạ cố định $m$ đạt $l'$ tại bước $j'$ với độ chính xác $a'$.

$$\boxed{\exists\, m:\quad j'\le j\ \ (\text{thời gian tương đương}),\quad a'\ge a\ \ (\text{độ chính xác tương đương}),\quad \|m\|_0\ll|\theta|}$$

$f(x;m\odot\theta_0)$ gọi là **winning ticket**. Nếu thay $\theta_0$ bằng khởi tạo mới $\theta_0''\sim D_\theta$ thì thường mất tính chất này.

### 3.2 Thuật toán tìm vé (one-shot rồi lặp)

```text
1. Khởi tạo ngẫu nhiên θ0 ~ Dθ;  mặt nạ m = 1.
2. Train f(x; m ⊙ θ) trong j bước  ->  θj
3. Cắt p% trọng số có |θj| nhỏ nhất (trong từng layer, hoặc toàn cục), cập nhật m
4. Reset trọng số còn lại về giá trị θ0  ->  vé f(x; m ⊙ θ0)
(iterative pruning: lặp bước 2-4 trong n vòng, mỗi vòng cắt p^(1/n)% trọng số còn sống,
 sau mỗi vòng reset về θ0)
```

- **One-shot**: train một lần, cắt p%, reset. Đơn giản nhưng ra vé lớn hơn.
- **Iterative**: lặp nhiều vòng; tìm được vé nhỏ hơn mà vẫn khớp mạng gốc. Tốn kém (train 15 lần liên tiếp trở lên).
- Phụ lục B so sánh hai biến thể lặp: *Strategy 1* (reset về $\theta_0$ sau mỗi vòng cắt, rồi train lại) và *Strategy 2* (train tiếp với trọng số đã train); Strategy 1 cho độ chính xác validation cao hơn và dừng sớm nhanh hơn ở kích thước nhỏ hơn trên Lenet và Conv-2/4/6. Paper dùng Strategy 1.
- Cắt cục bộ theo layer cho Lenet, Conv-2/4/6 (Lenet: fc 20%, output cắt một nửa tốc độ, tức 10%; Conv-2: fc 20%, conv 10%...). Với VGG-19 và ResNet-18 dùng cắt toàn cục: lấy trọng số nhỏ nhất gộp mọi layer conv, vì layer đầu rất nhỏ (1728 và 36864 tham số so với 2.35M ở layer cuối) trở thành nút cổ chai khi cắt cùng tỷ lệ.

Ký hiệu: $P_m=\|m\|_0/|\theta|$ là tỷ lệ trọng số còn lại (ví dụ $P_m=25\%$ nghĩa là đã cắt 75%).

### 3.3 Kiến trúc thí nghiệm (Figure 2)

| Mạng | Dữ liệu | Số trọng số | Bước / batch | Optimizer | Tỷ lệ cắt mỗi vòng |
|---|---|---|---|---|---|
| Lenet 300-100-10 | MNIST | 266K | 50K / 60 | Adam 1.2e-3 | fc 20% |
| Conv-2 | CIFAR-10 | 4.3M / 38K conv | 20K / 60 | Adam 2e-4 | fc 20%, conv 10% |
| Conv-4 | CIFAR-10 | 2.4M / 260K | 25K / 60 | Adam 3e-4 | fc 20%, conv 10% |
| Conv-6 | CIFAR-10 | 1.7M / 1.1M | 30K / 60 | Adam 3e-4 | fc 20%, conv 15% |
| ResNet-18 | CIFAR-10 | 274K / 270K | 30K / 128 | SGD momentum, 0.1-0.01-0.001 | conv 20% (toàn cục), fc 0% |
| VGG-19 | CIFAR-10 | 20.0M | 112K / 64 | SGD momentum | conv 20% (toàn cục), fc 0% |

(Bố cục bảng PDF lệch; cột "tỷ lệ cắt" tôi đọc theo đoạn "Pruning Rate" của Figure 2 và có thể sai ở Conv-6/ResNet-18/VGG-19; các số học tốc độ 0.1 hoặc 0.01 cho VGG/ResNet được nêu rõ ở Mục 4 của paper.)

## 4. Thí nghiệm và kết quả chính

### 4.1 Lenet / MNIST (Mục 2)

- Cắt lặp 20% mỗi vòng. Học nhanh dần khi $P_m$ giảm từ 100% xuống 21%: lúc đó dừng sớm **sớm hơn 38%** so với mạng gốc. Độ chính xác test tăng hơn 0.3 điểm phần trăm ở $P_m=13.5\%$, rồi giảm, về bằng mạng gốc ở $P_m=3.6\%$.
- Tại iteration 50,000, huấn luyện đã đạt ~100% train accuracy nhưng vé vẫn hơn tới 0.35 điểm test, tức khoảng cách train–test nhỏ hơn: vé khái quát tốt hơn.
- **Reinit ngẫu nhiên** (giữ mặt nạ, đổi khởi tạo; 3 lần/vé): học ngày càng chậm hơn và độ chính xác rơi ngay sau khi cắt ít. Khi $P_m=21\%$ vé dừng sớm nhanh hơn 2.51 lần so với bản reinit và chính xác hơn nửa điểm phần trăm. Độ chính xác của bản reinit bắt đầu tụt từ $P_m=21.1\%$, còn vé giữ được đến $P_m=2.9\%$.
- **One-shot**: khi $67.5\%\ge P_m\ge17.6\%$ vé dừng sớm hơn mạng gốc; khi $95.0\%\ge P_m\ge5.17\%$ độ chính xác test cao hơn mạng gốc; nhưng cắt lặp cho vé nhỏ hơn nữa.

### 4.2 Conv-2/4/6 / CIFAR-10 (Mục 3)

- Mô hình nhanh nhất so với mạng gốc: Conv-2 nhanh gấp 3.5 lần (ở $P_m=8.8\%$), Conv-4 gấp 3.5 lần ($9.2\%$), Conv-6 gấp 2.5 lần ($15.1\%$).
- Độ chính xác test tăng tối đa: Conv-2 +3.4 điểm ($P_m=4.6\%$), Conv-4 +3.5 điểm ($11.1\%$), Conv-6 +3.3 điểm ($26.4\%$). Cả ba vẫn trên độ chính xác gốc khi $P_m>2\%$.
- Reinit: độ chính xác lúc dừng sớm ban đầu ổn định hoặc còn tăng cho Conv-2 và Conv-4, gợi ý ở mức cắt vừa phải thì chỉ riêng *cấu trúc* đã đủ giúp; rồi tụt nhanh hơn vé.
- **Dropout 0.5**: dropout tăng độ chính xác ban đầu 2.1, 3.0, 2.4 điểm (Conv-2/4/6); cắt lặp tăng thêm tới 2.3, 4.6, 4.7 điểm. Tìm vé vẫn được khi có dropout.

### 4.3 VGG-19 và ResNet-18 (Mục 4)

- **Cần learning-rate warmup** (tăng tuyến tính lr từ 0 tới lr gốc trong $k$ bước) để tìm vé ở lr cao.
- **VGG-19** (20M tham số, 160 epoch = 112,480 bước): ở lr 0.1 cắt lặp *không* tìm ra vé, không hơn reinit. Ở lr 0.01 mạng con nằm trong 1 điểm của độ chính xác gốc khi $P_m\ge3.5\%$ (nhưng chưa phải vé vì chưa khớp độ chính xác gốc). Với warmup $k=10000$ ở lr 0.1: vé vượt độ chính xác ban đầu khi $P_m\ge1.5\%$ (độ chính xác mạng gốc tăng khoảng 1 điểm nhờ warmup).
- **ResNet-18** (271K tham số, 30K bước): lr 0.01 tìm được vé tốt nhất 89.5% ($41.7\%\ge P_m\ge21.9\%$), thấp hơn mạng gốc ở lr 0.1 (90.5%). Warmup ($k=20000$, lr 0.03) đạt 90.5% ở $P_m=27.1\%$; vẫn tìm được vé khi $P_m\ge11.8\%$. Không tìm được hyperparameter nào cho vé ở lr gốc 0.1.

## 5. Phân tích / ablation

- **Khởi tạo quan trọng:** reinit ngẫu nhiên làm mất hiệu quả (mọi mạng). Thêm nhiễu Gaussian vào khởi tạo của vé (phụ lục F.7): nhiễu 0.5σ gần như không đổi; ngay cả 3σ vẫn tốt hơn reinit hoàn toàn.
- **Cấu trúc cũng quan trọng:** ở Lenet reinit tốt hơn mạng thưa ngẫu nhiên cùng số tham số, còn ở các mạng conv không khác biệt đáng kể (phụ lục E). Giải thích của tác giả: MNIST chỉ có vùng ảnh nhất định chứa thông tin, nên một số kết nối có giá trị hơn.
- **Cắt tại iteration 0 (phụ lục F.4):** cắt theo độ lớn khởi tạo trước khi train còn tệ hơn tìm bằng cắt lặp, tức không thể chỉ nhìn khởi tạo là chọn được vé.
- **Vé không "đã được train sẵn":** trọng số thuộc vé thay đổi nhiều hơn trọng số khác giữa đầu và cuối train (phụ lục F.5), phản bác giả thuyết rằng chúng sẵn gần nghiệm. Tác giả cho rằng lợi ích gắn với thuật toán tối ưu, dữ liệu, mô hình.
- **Phân bố khởi tạo của vé:** các layer ẩn sâu hơn có phân bố hai đỉnh (bimodal), trọng số ban đầu có độ lớn lớn có khả năng sống sót cao hơn (F.1, F.2). Layer đầu giữ phân bố gốc với Adam.
- **Kết nối:** mỗi neuron giữ số kết nối vào tỷ lệ với mức cắt của layer (F.6), không có neuron nào bị "cắt cụt" bất thường.
- **Global vs layerwise pruning** cho mạng sâu: global cho vé nhỏ hơn (I.1).
- **Dropout** bổ sung cho cắt lặp.

## 6. Giới hạn

**Do tác giả nêu:**
- Chỉ phân loại ảnh trên MNIST/CIFAR-10; không có ImageNet vì cắt lặp cần train >15 lần liên tiếp.
- Chỉ cắt thưa không cấu trúc (unstructured), nên mạng thu được không tối ưu cho thư viện/phần cứng hiện hành; cắt structured và tiêu chí không dựa độ lớn để dành cho tương lai.
- Với mạng sâu (ResNet-18, VGG-19) cần warmup mới tìm được vé ở lr cao; chưa rõ vì sao.
- "Lottery ticket conjecture" (SGD tìm vé) chưa được kiểm chứng.
- Không giải thích được tại sao khởi tạo của vé lại tốt.
- Liu et al. 2019 (bài 11) cho thấy mạng cắt rồi reinit vẫn bằng mạng gốc ở mức cắt đến ~80% trên VGG-19; paper thừa nhận điều này khớp với kết quả mình ở mức thưa đó, nhưng tuyên bố ở mức thưa hơn (cắt tới 98.5%) thì khởi tạo mới quan trọng.

**Do chúng tôi nhận xét:**
- Tìm vé cần *train mạng đầy đủ nhiều lần*, nên không phải phương pháp tiết kiệm chi phí huấn luyện trong thực tế; ý nghĩa chính là khoa học về khả năng huấn luyện mạng thưa.
- Mạng thưa không cấu trúc khó chuyển thành tăng tốc suy luận GPU batch-1.
- Tất cả mạng dưới ~20M tham số; không có transformer, cũng không có ngôn ngữ.
- Số liệu "học nhanh hơn" dựa vào proxy *early-stopping iteration* trên validation, nhạy với lịch lr (nên phải đổi sang đo độ chính xác tại vài mốc cho VGG/ResNet).

## 7. Liên hệ với các bài khác

- **02 Han 2015, 03 Zhu & Gupta:** dùng cùng tiêu chí độ lớn; LTH thêm bước *reset về khởi tạo* và đổi mục tiêu từ nén sang train mạng thưa.
- **09 Linear mode connectivity & LTH:** chứng minh vé ở mạng lớn (ResNet-50, Inception-v3) chỉ tìm được nếu reset về *một thời điểm sớm* $k$ sau khởi tạo (rewinding), không phải iteration 0 (đây chính là vấn đề warmup ở Mục 4).
- **10 SNIP:** tìm mạng con ngay tại khởi tạo, không cần train trước, trái với bước "train rồi cắt" của LTH.
- **11 Rethinking the value of pruning:** nghi ngờ vai trò của khởi tạo trong cắt structured/ lr chuẩn; LTH đã đề cập sự khác biệt (mục 5 Discussion).
- **15 LTH for BERT:** mở rộng LTH sang transformer.
- **43 Pruning vs scratch:** vấn đề tương tự, cắt+recover so với train từ đầu, ở quy mô LLM.

## 8. Ứng dụng cho dự án (suy luận)

**Ít liên quan trực tiếp.** Dự án là cắt có cấu trúc (structured) một VLM đã huấn luyện, rồi phục hồi bằng chưng cất; không có kế hoạch train mạng thưa từ đầu, và LTH làm việc với cắt thưa không cấu trúc, không cho GPU nhỏ tăng tốc.

Giá trị gián tiếp:
- **Giá trị của khởi tạo / trọng số đã học:** kết quả "cùng cấu trúc nhưng khởi tạo mới thì tệ hơn" phù hợp với thực hành của dự án: nên khởi tạo mô hình cắt từ trọng số mô hình gốc thay vì train lại từ đầu, và kết hợp chưng cất để phục hồi.
- **Cắt lặp nhiều vòng tốt hơn một lần:** một thông điệp lặp lại ở nhiều bài; có thể áp dụng cho việc giảm 0.8B → 0.3B → 0.1B qua nhiều bước nhỏ và chưng cất từng bước.
- **Phỏng đoán (giả thuyết, chưa kiểm chứng):** nếu ý tưởng "reset" tương đương bằng "rewinding" (bài 09) áp dụng cho recovery bằng chưng cất thì việc quay lại trọng số sớm là một nguồn khởi tạo cho mạng cắt; cần thí nghiệm riêng.
- **Cảnh báo về warmup:** kết quả Mục 4 nhắc rằng recovery nên dùng lr nhỏ hoặc warmup để không phá cấu trúc đã học.

## 9. Thí nghiệm gợi ý

1. **Lặp tái hiện nhỏ** (cho bản thân hiểu): Lenet-300-100 trên MNIST, cắt lặp 20% mỗi vòng, so vé (reset θ0) với reinit và mạng thưa ngẫu nhiên; vẽ độ chính xác theo $P_m$ (khoảng vài chục phút trên CPU).
2. **Cắt neuron MLP của decoder theo bước nhỏ so với một bước** với cùng ngân sách chưng cất (ý tưởng "iterative > one-shot" của bài).
3. **Khởi tạo của mạng đã cắt:** so (a) giữ trọng số gốc còn lại, (b) khởi tạo lại tại các neuron còn lại + chưng cất cùng ngân sách, để đo giá trị của trọng số tiền huấn luyện trong hệ chưng cất (cho thấy mô hình dự án "giống LTH" tới đâu).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Lottery ticket hypothesis | Mạng dày chứa mạng con (mặt nạ $m$ + khởi tạo $\theta_0$) train riêng đạt độ chính xác tương đương trong số bước tương đương |
| Winning ticket | $f(x;m\odot\theta_0)$ thỏa $j'\le j,\ a'\ge a,\ \|m\|_0\ll|\theta|$ |
| $P_m=\|m\|_0/|\theta|$ | Tỷ lệ trọng số còn lại |
| One-shot vs iterative pruning | Cắt p% một lần / cắt dần qua n vòng, reset về $\theta_0$ mỗi vòng |
| Random reinit | Giữ mặt nạ, lấy lại $\theta_0''\sim D_\theta$; thường kém hơn nhiều |
| Early-stopping iteration | Bước có loss validation nhỏ nhất, đại diện tốc độ học |
| Global pruning | Cắt trọng số nhỏ nhất gộp mọi layer (cần cho VGG-19, ResNet-18) |
| Warmup | Tăng lr tuyến tính từ 0; cần để tìm vé ở lr cao |
| Lottery ticket conjecture | SGD tìm và train một mạng con được khởi tạo tốt (chưa kiểm chứng) |
| Occam's Hill | Độ chính xác tăng rồi giảm khi cắt: mạng gốc quá phức tạp, mạng cắt quá nhiều quá đơn giản |

## 11. Kết luận ngắn

Mạng dày chứa mạng con thưa mà nếu giữ nguyên khởi tạo gốc sẽ train được nhanh và chính xác bằng hoặc hơn mạng gốc, với kích thước 10–20% (hoặc nhỏ hơn) trên MNIST và CIFAR-10; cùng cấu trúc nhưng khởi tạo mới thì thua. Bài này khởi đầu cả một nhánh nghiên cứu về mạng thưa train được, nhưng với dự án cắt có cấu trúc rồi chưng cất, giá trị chính là thông điệp "cắt dần nhiều vòng" và "giữ trọng số đã học thay vì khởi tạo lại", chứ không phải thuật toán tìm vé.

Link gốc: https://arxiv.org/abs/1803.03635
