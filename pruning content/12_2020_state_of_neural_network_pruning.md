# What is the State of Neural Network Pruning? (2020)

**Tác giả:** Davis Blalock*, Jose Javier Gonzalez Ortiz*, Jonathan Frankle, John Guttag (MIT CSAIL; * đóng góp ngang nhau).  
**Venue / arXiv:** arXiv:2003.03033 (6/3/2020); bản trong file ghi là Proceedings of the 3rd MLSys Conference, Austin, TX, 2020.  
**Nguồn:** https://arxiv.org/abs/2003.03033  
**Chủ đề:** Meta-analysis 81 bài pruning; danh mục lỗi trong cách báo cáo/so sánh; khuyến nghị thực hành; thư viện ShrinkBench. Đây là bài **"cách đọc paper pruning một cách phê phán"**.  
**Vị trí trong lộ trình:** Nên đọc sau các bài nền tảng 01-11 (OBD/OBS, Han 2015, Zhu & Gupta, các bài structured pruning, Lottery Ticket, SNIP, Rethinking the value of pruning). Bài này không đề xuất thuật toán mới; nó cho bạn "bộ lọc nghi ngờ" để đọc mọi bài sau (13-49). Bài kế tiếp: 13 (Sixteen Heads), mở đầu nhánh pruning cho Transformer.

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Mọi con số lấy từ paper; phần "Ứng dụng cho dự án" và các nhận xét ghi rõ "(suy luận)" là suy luận của chúng tôi, không phải kết luận của tác giả.

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

**Pruning (cắt tỉa)** = bỏ bớt tham số của một mạng đã có để mạng nhỏ/nhanh hơn mà độ chính xác gần như giữ nguyên. Ý tưởng có từ cuối thập niên 1980 (Janowsky 1989, Mozer & Smolensky 1989, Karnin 1990), bùng nổ sau 2015 nhờ deep learning.

Đến 2020 có hàng chục phương pháp, và mỗi bài đều tuyên bố "state of the art". Các tác giả đặt những câu hỏi tưởng đơn giản:

- Kỹ thuật nào cho đánh đổi accuracy/hiệu quả tốt nhất?
- Có chiến lược nào hợp riêng với kiến trúc/dataset nào không?
- Lựa chọn thiết kế cấp cao nào hiệu quả nhất?

Kết luận trung tâm (nguyên văn ý): **các câu hỏi này không thể trả lời được** vì văn liệu quá phân mảnh. Bài này là bản "kiểm toán" của lĩnh vực.

Thuật ngữ nền:
- **Unstructured pruning**: bỏ từng trọng số riêng lẻ → ma trận thưa, khó tăng tốc trên phần cứng thông thường.
- **Structured pruning**: bỏ cả neuron/filter/channel (và với Transformer là head, hàng/cột của MLP, cả layer) → mạng nhỏ dạng dense, tăng tốc thật.
- **Mask** $M\in\{0,1\}^{|W|}$: vector nhị phân nói tham số nào còn sống.
- **Fine-tuning**: huấn luyện tiếp sau khi cắt để phục hồi.
- **FLOPs** (ở đây là multiply-adds): số phép nhân-cộng khi suy luận.

## 2. Ý tưởng chính

Tác giả gom kết quả từ **81 bài** (79 bài từ 2010 trở đi + 2 bài kinh điển LeCun 1990 và Hassibi 1993) và tự cắt tỉa hơn **800 mạng** trong điều kiện kiểm soát. Họ rút ra: (1) vài kết luận nhất quán (pruning hiệu quả, magnitude pruning là baseline mạnh, ...); (2) kết luận lớn nhất là *việc so sánh gần như bất khả thi* vì thiếu benchmark, metric và thực hành chuẩn; (3) họ đề xuất danh sách thực hành tốt và thư viện **ShrinkBench** để so sánh công bằng; (4) vài thí nghiệm nhỏ cho thấy các "bẫy" thường gặp (chọn metric, chọn mô hình gốc, chọn điểm hoạt động) có thể đảo ngược thứ hạng giữa các phương pháp.

## 3. Phương pháp chi tiết

Bài là meta-analysis nên "phương pháp" gồm khung phân loại, quy trình chuẩn hóa dữ liệu và ShrinkBench.

### 3.1 Khung chung: Algorithm 1 (prune + fine-tune)

Gần như mọi chiến lược trong corpus là biến thể của thuật toán Han et al. (2015):

```text
Input: N = số vòng prune; X = dữ liệu train/fine-tune
W <- initialize()
W <- trainToConvergence(f(X; W))
M <- 1^{|W|}                       # mask toàn 1
for i in 1..N:
    M <- prune(M, score(W))         # bỏ tham số điểm thấp
    W <- fineTune(f(X; M ⊙ W))      # train tiếp để hồi phục
return M, W
```

Mạng sau cắt: $f(x; M\odot W')$, trong đó $W'$ có thể khác $W$ (nếu có rewinding/re-init), $\odot$ là nhân từng phần tử.

### 3.2 Bốn trục phân biệt các phương pháp

| Trục | Lựa chọn trong paper |
|---|---|
| **Structure** | unstructured (từng trọng số) vs structured (neuron, filter, channel) |
| **Scoring** | theo $|w|$, theo hệ số "importance" học được, theo đóng góp vào activation hoặc gradient; so sánh **local** (trong từng layer, vd Han 2015) hay **global** (toàn mạng, vd Lee 2019b, Frankle & Carbin 2019) |
| **Scheduling** | cắt một phát (one-shot, Liu 2019), cắt lặp cố định tỉ lệ mỗi vòng (Han 2015), hoặc lịch phức tạp (Gale 2019) |
| **Fine-tuning** | tiếp tục từ trọng số đã train (phổ biến nhất), **rewinding** về trạng thái sớm hơn (Frankle 2019), hoặc khởi tạo lại hoàn toàn (Liu 2019) |

Mẹo đọc paper: khi gặp bài mới, hãy điền bốn ô này trước. Hai bài khác nhau ở cả bốn ô thì kết quả của chúng gần như không so sánh được.

### 3.3 Hai metric hiệu quả, và định nghĩa nên dùng

$$\boxed{\text{compression ratio}=\frac{\text{original size}}{\text{new size}}},\qquad \boxed{\text{theoretical speedup}=\frac{\text{original multiply-adds}}{\text{new multiply-adds}}}$$

Tác giả nhấn mạnh: không có lý do gì chỉ báo cáo một trong hai. Nhiều bài lại định nghĩa "compression ratio" là $1-\frac{\text{compressed}}{\text{original}}$, trái với truyền thống ngành nén dữ liệu.

### 3.4 Chuẩn hóa để vẽ Figure 1

Vì nhiều bài chỉ báo "thay đổi accuracy" hoặc "% pruned" mà không cho baseline, tác giả **nhân tỉ lệ đã báo cáo với giá trị khởi đầu chuẩn** (median kích thước/FLOPs được báo cáo cho kiến trúc đó trên toàn bộ các bài). Họ tự nhận cách này "không hoàn hảo" nhưng giúp kiểm soát phần nào việc các phương pháp bắt đầu từ baseline khác nhau.

### 3.5 ShrinkBench

Thư viện PyTorch mã nguồn mở. Người dùng chỉ cần viết một **callback** tính mask cho các tensor tham số tại một iteration; ShrinkBench tự áp mask, cập nhật mạng theo quy trình train/fine-tune chuẩn, tính metric, chạy nhiều mô hình/dataset/seed/mức cắt và vẽ biểu đồ. Baselines đi kèm (chỉ "lấy cảm hứng", không phải tái hiện chính xác bài gốc):

- Global Magnitude: bỏ trọng số $|w|$ nhỏ nhất trên toàn mạng.
- Layerwise Magnitude: bỏ $|w|$ nhỏ nhất trong từng layer.
- Global / Layerwise Gradient Magnitude: dùng $|w\cdot \nabla w|$ tính trên một batch.
- Random: mỗi trọng số bị bỏ độc lập với xác suất bằng tỉ lệ cần cắt; hữu ích làm công cụ debug.

## 4. Thí nghiệm và kết quả chính

### 4.1 Những điều nhất quán trong văn liệu (Mục 3)

1. **Pruning hiệu quả**: nhiều phương pháp nén đáng kể mà mất rất ít hoặc không mất accuracy; với nén nhẹ thỉnh thoảng accuracy còn tăng (Han 2015, Suzuki 2018).
2. **Nhiều phương pháp thắng random pruning** (ít nhất ở mức cắt lớn); ở mức cắt nhẹ điều này không phải lúc nào cũng đúng (Morcos 2019).
3. **Phân bổ tham số thông minh giữa các layer** hoặc cắt global tốt hơn cắt đều mọi layer.
4. Với số bước fine-tune bằng nhau, nhiều phương pháp cho mô hình tốt hơn **train từ đầu với cùng pattern thưa** (với mức cắt đủ lớn).
5. **Sparse > dense khi cố định số tham số**: ví dụ tăng kích thước danh nghĩa của ResNet-20 trên CIFAR-10 rồi làm thưa để giữ nguyên số tham số thì lỗi giảm (Lee 2019a).
6. Figure 1 (ImageNet, so mô hình pruned với họ kiến trúc không pruned): pruning cải thiện đánh đổi size/speed vs accuracy của một kiến trúc; **nhưng thường không bằng việc đổi sang kiến trúc tốt hơn**; và pruning hiệu quả hơn với kiến trúc vốn kém hiệu quả.

### 4.2 Những vấn đề phân mảnh (Mục 4-5), các con số quan trọng

| Vấn đề | Số liệu từ paper |
|---|---|
| Thiếu so sánh | hơn 1/4 số bài không so sánh với phương pháp pruning nào; thêm 1/4 chỉ so sánh 1 phương pháp; hầu hết so sánh ≤ 3 |
| Bỏ qua phương pháp cũ | cả thập niên 1990 gần như bị bỏ ngoài OBD và OBS; Han 2015 "tái phát hiện" magnitude pruning của Janowsky 1989; Lee 2019b tái phát hiện saliency của Mozer & Smolensky 1989a; He 2018a tái phát hiện "reviving" của Tresp 1997 |
| Phân mảnh dataset/kiến trúc | 81 bài dùng **49 dataset, 132 kiến trúc, 195 cặp (dataset, architecture)** |
| Cặp phổ biến nhất | VGG-16 trên ImageNet: 22/81 bài; ResNet-50 ImageNet: 15; LeNet-5-Caffe MNIST: 14; ResNet-56 CIFAR-10: 14; LeNet-300-100 MNIST: 12 (Table 1). Ba trong sáu cặp phổ biến nhất dùng MNIST |
| Chỉ 37/81 bài có kết quả trên bất kỳ cấu hình chung nào trong Figure 3 | (bốn cặp phổ biến nhất không tính MNIST) |
| Điểm hoạt động | đa số dùng ≤ 3 cặp (dataset, arch), và thường **chỉ 1 điểm** để mô tả cả đường đánh đổi |
| Độ lệch chuẩn | trong Figure 3 chỉ He 2018 (CIFAR-10) báo độ lệch chuẩn |
| FLOPs không nhất quán | chênh tới 4 lần cho cùng kiến trúc/dataset: AlexNet ImageNet được báo 371 MFLOPs (Yang 2017), 724 MFLOPs (Choi 2019), 1500 MFLOPs (Han 2015) |

Ngoài ra: "ResNet-50" mơ hồ vì có hai họ ResNet (He 2016a,b); "VGG-16" thường là biến thể tự chế (bỏ FC, đổi kích thước FC, thêm dropout/batchnorm); "Pruned%" có thể là phần còn lại hoặc phần bị bỏ; "speedup" có khi là wall time, có khi là tỉ lệ FLOPs.

Figure 5 cho thấy: cùng là **magnitude pruning không cấu trúc trên ResNet-50 ImageNet**, các bài khác nhau về cài đặt/lịch/fine-tuning cho kết quả phân tán gần bằng độ phân tán giữa các *phương pháp* khác nhau. Nghĩa là sai khác do "chi tiết cài đặt" ngang với sai khác do "ý tưởng thuật toán".

### 4.3 Thí nghiệm ShrinkBench (>800 mạng; thanh sai số = 1 độ lệch chuẩn qua 3 lần chạy trên CIFAR-10)

- **Metric không thể thay thế nhau** (Figure 6, ResNet-18 ImageNet): Global thắng Layerwise khi cố định *kích thước mô hình*, nhưng **ngược lại** khi cố định *theoretical speedup*. Lý do hợp lý (diễn giải): Global thường cắt các layer có nhiều tham số nhưng ít FLOPs.
- **Kết quả thay đổi theo mô hình, dataset, mức cắt** (Figure 7, CIFAR-VGG và ResNet-56 trên CIFAR-10): nói chung Global > Layerwise, Magnitude > Gradient, Random tệ nhất. Nhưng nếu chỉ nhìn CIFAR-VGG ở compression < 10 thì có thể kết luận nhầm là Global Gradient tốt nhất; và Global Gradient luôn hơn Layerwise Magnitude trên CIFAR-VGG nhưng luôn kém hơn trên ResNet-56.
- **Seed**: gần điểm "sụp" (Global Gradient, compression 16) các seed khác nhau cho accuracy **0.88 so với 0.61** do ngẫu nhiên khi chọn minibatch.
- **Mô hình gốc phải giống nhau** (Figure 8): hai ResNet-56 huấn luyện bằng Adam với $\eta=10^{-3}$ (Weights A) và $\eta=10^{-4}$ (Weights B). Với A hai phương pháp xấp xỉ nhau; với B Global tốt hơn ở compression cao. Báo cáo *thay đổi* accuracy so với gốc **không đủ** để khử nhiễu này: Layerwise+Weights B có thể trông tốt hơn Global+Weights A dù khi giữ mô hình gốc cố định nó chưa bao giờ hơn.

## 5. Phân tích / ablation

Paper không có ablation theo nghĩa thuật toán; phần tương đương là các "kiểm soát biến nhiễu". Biến nhiễu được liệt kê: độ chính xác và hiệu quả của mô hình ban đầu; data augmentation/tiền xử lý; ngẫu nhiên khi khởi tạo/train/fine-tune (optimizer, siêu tham số, learning-rate schedule); lịch prune và fine-tune; thư viện deep learning; khác biệt nhỏ trong code/môi trường. Tác giả cho rằng gần như không bài nào kiểm soát hết, trừ khi tác giả tự chạy lại tất cả với cùng code (hiếm; họ chỉ nêu Gale 2019 và Liu 2019 và hai bài này là các phân tích hơn là đề xuất phương pháp mới).

Một lập luận quan trọng: các thay đổi accuracy ở điểm hoạt động được báo cáo thường **< 1%**, nên một biến nhiễu nhỏ cũng đủ đổi thứ hạng.

**Khuyến nghị thực hành của tác giả** (Mục 6), đây là phần nên chép thành checklist:

1. Nêu chính xác kiến trúc, dataset, metric, ở dạng có cấu trúc.
2. Dùng ít nhất **3 cặp (dataset, architecture)** hiện đại, quy mô lớn; MNIST/mô hình đồ chơi không tính.
3. Báo cáo cả compression ratio **và** theoretical speedup.
4. Với ImageNet và dataset nhiều lớp: báo cả Top-1 và Top-5.
5. Báo metric của mô hình đối chứng (thường là mô hình gốc chưa cắt).
6. Vẽ đường đánh đổi cùng các phương pháp cạnh tranh.
7. Dùng **≥ 5 điểm hoạt động**, gợi ý compression $\{2,4,8,16,32\}$.
8. Báo trung bình và độ lệch chuẩn mẫu thay vì một lần chạy.
9. Cùng thư viện, data loading, code cho mọi phương pháp được so sánh.
10. Reviewer nên đòi mức nghiêm ngặt cao hơn với các bài tuyên bố phương pháp tốt hơn.

## 6. Giới hạn

**Do tác giả nêu:**
- Phân tích thiên về phân loại ảnh (Top-1/Top-5) và hai metric FLOPs/số tham số vì "đại đa số bài trong corpus" làm vậy; tác giả thừa nhận hai metric này chỉ là proxy lỏng cho latency, throughput, bộ nhớ, năng lượng thực tế.
- Cách chuẩn hóa để vẽ Figure 1 "không hoàn hảo".
- Các baseline của ShrinkBench chỉ lấy cảm hứng từ heuristic, không tái hiện bài gốc.

**Do chúng tôi nhận thấy (suy luận):**
- Corpus dừng ở đầu 2020: không có Transformer/NLP/LLM (trừ ngụ ý chung), không có structured pruning trên mô hình ngôn ngữ, không có distillation kết hợp. Mọi con số cụ thể ("magnitude là baseline mạnh") cần kiểm lại cho hệ Transformer; các bài 13-17 và 22-30 cho thấy bức tranh khác.
- Thí nghiệm ShrinkBench dùng CIFAR-10 là chính (xem Fig. 6-8 là CIFAR-10/ImageNet nhỏ); tức chính bài này cũng chỉ có vài cặp (dataset, arch), và các khuyến nghị "≥ 3 cặp lớn" không được tự áp dụng triệt để.
- Phát hiện "pruning kém hơn đổi kiến trúc" dựa trên các đường cong tự báo cáo đã chuẩn hóa, nên bản thân chịu đúng những nhiễu mà bài phê phán.

## 7. Liên hệ với các bài khác

- **01 OBD/OBS**: được nhắc là hai bài kinh điển hiếm hoi còn được so sánh; phần lớn văn liệu 1990s bị bỏ quên.
- **02 Han 2015**: nguồn của Algorithm 1 (train → prune → fine-tune lặp) và magnitude pruning.
- **03 Zhu & Gupta**: một ví dụ của "scheduling" khác (lịch cắt dần).
- **04-06 (Pruning filters, Network slimming, ThiNet)** và **07 Molchanov**: các biến thể structured/scoring; hãy dùng 4 trục ở mục 3.2 để so sánh chúng.
- **08-09 Lottery Ticket, Linear mode connectivity**: "rewinding" xuất hiện trong trục fine-tuning; Frankle là đồng tác giả bài này.
- **10 SNIP**: ví dụ cắt tại khởi tạo bằng gradient (nhóm "gradient-based").
- **11 Rethinking the value of pruning**: liên quan trực tiếp đến phát hiện "pruned vs train-from-scratch"; bài 12 cho thấy kết luận này phụ thuộc vào cài đặt.
- **13-17** (bài sau trong lộ trình): đều có thể được "chấm" bằng checklist ở mục 5 (xem 4 câu hỏi ở phần Thí nghiệm gợi ý).
- **43 Pruning vs scratch** và **37 Repair changes which layers**: các bài hiện đại giữ tinh thần "so sánh có kiểm soát".

## 8. Ứng dụng cho dự án (suy luận)

Bài không cung cấp kỹ thuật cắt cho OCR-VLM, nhưng rất hữu ích cho **cách thiết kế thí nghiệm** của dự án LightOnOCR-3-0.8B → ~0.3B / ~0.1B:

1. **Báo cả hai trục hiệu quả** nhưng thay FLOPs bằng thứ bạn thật sự cần: **latency batch-1 đo trên GPU ≤ 8 GB** và bộ nhớ đỉnh. Bài 12 chứng minh FLOPs/tham số xếp hạng phương pháp khác nhau (Global vs Layerwise đảo thứ hạng). Cắt MLP neuron giảm tham số nhiều nhưng latency batch-1 có thể giảm ít (GatedDeltaNet và attention full có đặc tính tốc độ rất khác).
2. **Cố định mô hình gốc** (cùng checkpoint 0.8B, cùng teacher 4B) cho mọi phương pháp cắt; đừng trộn checkpoint. Figure 8 cho thấy sự khác biệt về mô hình gốc có thể đủ làm đảo thứ hạng.
3. **Nhiều điểm hoạt động**: không chỉ 0.3B và 0.1B, hãy có thêm 0.6B, 0.45B, 0.2B để vẽ đường cong; một điểm không thể phân biệt hai phương pháp.
4. **Seed**: ít nhất 3 seed cho các điểm gần "vách" (nhất là 0.1B), vì paper thấy 0.88 vs 0.61 gần điểm sụp.
5. **Baseline random và magnitude** luôn nên có làm chuẩn debug. Với dự án, baseline tương ứng: cắt layer ngẫu nhiên, cắt layer theo chỉ số kiểu ShortGPT (31), cắt MLP neuron theo độ lớn.
6. Việc bạn tách $D_{lin}$ và $D_{res}$ là một dạng "phân tích cơ chế" bổ sung; bài 12 nhắc rằng cần báo cả metric tác vụ cuối (OCR edit distance/CER, WER) trên cả tập trong-miền và ngoài-miền, không chỉ loss.
7. Quan sát "cắt MLP neuron hồi phục tệ hơn cắt layer" của bạn là loại kết luận **cần kiểm soát chặt**: cùng ngân sách tham số/latency, cùng lịch distillation, cùng seed. Đối chiếu với bài 12: thứ bạn so là *cấu trúc cắt* (trục Structure) nên phải giữ nguyên ba trục còn lại (scoring, scheduling, fine-tuning).

Mức liên quan: **cao về phương pháp luận, thấp về kỹ thuật**.

## 9. Thí nghiệm gợi ý

1. **Bảng "4 trục" cho mọi paper trong lộ trình**: lập một bảng (Structure, Scoring, Scheduling, Fine-tuning) + (dataset/arch, metric hiệu quả, số điểm hoạt động, có độ lệch chuẩn không, đối chứng có không) cho bài 13-17. Chỉ ra ô nào "không báo cáo".
2. **Đảo thứ hạng trong dự án**: với một mô hình nhỏ (hoặc chỉ ViT 12 block), so sánh "cắt theo layer" và "cắt MLP neuron" ở (a) cùng số tham số và (b) cùng latency batch-1 đo thực. Kiểm tra thứ hạng có đảo như Global vs Layerwise trong Figure 6 không.
3. **Độ nhạy theo seed/checkpoint**: lặp một cấu hình cắt với 3 seed và 2 checkpoint gốc khác nhau (ví dụ trước/sau một epoch fine-tune), ghi biên độ dao động để biết mức chênh nào là đáng tin.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| $f(x; M\odot W')$ | mạng đã cắt: mask $M$ nhân từng phần tử với trọng số |
| Compression ratio | $\frac{\text{original size}}{\text{new size}}$ (theo paper; không phải $1-\ldots$) |
| Theoretical speedup | $\frac{\text{original multiply-adds}}{\text{new multiply-adds}}$ |
| Structure / Scoring / Scheduling / Fine-tuning | 4 trục phân loại phương pháp |
| Global vs layerwise | so điểm toàn mạng hay trong từng layer |
| Rewinding | quay trọng số về trạng thái sớm hơn rồi mới train tiếp |
| Operating point | một điểm (mức nén, accuracy) trên đường đánh đổi |
| ShrinkBench | thư viện đánh giá pruning chuẩn hóa (PyTorch) |
| Nhiễu "initial model" | khác mô hình gốc có thể đảo thứ hạng phương pháp, kể cả khi báo thay đổi accuracy |
| Dải nén gợi ý | $\{2,4,8,16,32\}$, ≥ 5 điểm |

## 11. Kết luận ngắn

Pruning "hoạt động", nhưng phần lớn con số trong văn liệu 2010-2019 không so sánh được với nhau: ít so sánh chéo, 195 cặp (dataset, arch) rời rạc, metric mơ hồ, một điểm hoạt động, không có độ lệch chuẩn, mô hình gốc khác nhau. Mang theo bài này như một checklist: với mỗi paper tiếp theo, hỏi "cùng mô hình gốc chưa, cùng metric chưa, bao nhiêu điểm, bao nhiêu seed, đối chứng là gì". Và với dự án của bạn, hãy tự áp checklist đó lên các so sánh nội bộ trước khi tin vào kết luận như "cắt layer tốt hơn cắt neuron".

Link gốc: https://arxiv.org/abs/2003.03033
