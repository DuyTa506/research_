# Knowledge Distillation: A Survey (2021)

**Tác giả:** Jianping Gou, Baosheng Yu, Stephen J. Maybank, Dacheng Tao  
**Venue / arXiv:** arXiv:2006.05525 (v7, 20/5/2021). Trong file text chỉ có tiêu đề "Noname manuscript"; theo kiến thức của chúng ta bài được đăng ở International Journal of Computer Vision (2021), không kiểm chứng từ file.  
**Nguồn:** https://arxiv.org/abs/2006.05525  
**Chủ đề:** Khảo sát KD (chủ yếu giai đoạn 2014–2021, thiên về thị giác máy tính): phân loại theo loại tri thức (response / feature / relation), sơ đồ huấn luyện (offline / online / self), kiến trúc teacher-student, các thuật toán (adversarial, multi-teacher, cross-modal, graph, attention, data-free, quantized, lifelong, NAS), so sánh hiệu năng, ứng dụng.  
**Vị trí trong lộ trình:** Đọc sau 01–04 (đã biết logit KD, hint, attention transfer). Đây là **bản đồ**: dùng để định vị từng bài sau (06 Born-again, 07 DML, 08 TA, 12 RKD, 13 CRD...). Survey này kết thúc ở 2021 nên **không có** LLM distillation (xem 27 LLM KD survey và 28–47 cho thời kỳ sau).

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch paper. Đây là survey nên ghi chú tập trung vào phân loại và cách dùng như bản đồ, không đi sâu từng phương pháp con. Phần "Ứng dụng cho dự án" là suy luận của chúng ta.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính và cách dùng survey như bản đồ](#2-ý-tưởng-chính-và-cách-dùng-survey-như-bản-đồ)
3. [Phương pháp chi tiết: phân loại của survey](#3-phương-pháp-chi-tiết-phân-loại-của-survey)
4. [Thí nghiệm và kết quả chính](#4-thí-nghiệm-và-kết-quả-chính)
5. [Phân tích / ablation (những nhận định của survey)](#5-phân-tích--ablation-những-nhận-định-của-survey)
6. [Giới hạn](#6-giới-hạn)
7. [Liên hệ với các bài khác](#7-liên-hệ-với-các-bài-khác)
8. [Ứng dụng cho dự án (suy luận)](#8-ứng-dụng-cho-dự-án-suy-luận)
9. [Thí nghiệm gợi ý](#9-thí-nghiệm-gợi-ý)
10. [Tóm tắt công thức / khái niệm](#10-tóm-tắt-công-thức--khái-niệm)
11. [Kết luận ngắn](#11-kết-luận-ngắn)

---

## 1. Bối cảnh

Mô hình học sâu lớn khó triển khai trên thiết bị hạn chế tài nguyên. Survey xếp KD vào họ nén và tăng tốc mô hình, cùng với bốn nhóm (theo Cheng et al., 2018, mà survey trích dẫn): (1) tỉa và chia sẻ tham số (gồm lượng tử hoá, nhị phân hoá, ma trận có cấu trúc, chia sẻ tham số), (2) phân rã hạng thấp (low-rank factorization), (3) bộ lọc tích chập nén chuyển giao, (4) KD. Survey chỉ tập trung nhóm thứ tư và không khảo sát đầy đủ nén mô hình nói chung.

**Lịch sử theo survey:** Buciluǎ et al. (2006) đề xuất model compression; Urner et al. (2011) dùng chuyển giao từ teacher sang student trên dữ liệu không nhãn cho học bán giám sát (và chứng minh học được theo PAC); Hinton et al. (2015) phổ biến hoá thành "knowledge distillation".

**Hiểu biết lý thuyết còn ít (theo survey):** Phuong & Lampert (2019) cho giới hạn tổng quát hoá với tốc độ hội tụ nhanh cho bộ phân loại tuyến tính sâu, và nêu ba yếu tố: hình học dữ liệu, thiên lệch tối ưu hoá của mục tiêu distillation, và tính đơn điệu mạnh của student; Cheng et al. (2020) lượng hoá khái niệm thị giác trích từ lớp giữa; Ji & Zhu (2020) phân tích mạng rộng theo giới hạn rủi ro, hiệu quả dữ liệu, teacher không hoàn hảo; Cho & Hariharan (2019) phân tích thực nghiệm hiệu quả của KD; Mirzadeh et al. (2020) cho thấy teacher lớn hơn không chắc là teacher tốt hơn do khoảng cách capacity; Tang et al. (2020) xem KD như label smoothing, đánh giá độ chính xác teacher, và prior hình học lớp đầu ra.

**Ba thành phần của một hệ KD (theo survey):** (i) tri thức, (ii) thuật toán distillation, (iii) kiến trúc teacher-student.

## 2. Ý tưởng chính và cách dùng survey như bản đồ

Survey không có một ý tưởng kỹ thuật mới. Đóng góp là **hệ thống hoá** KD theo các trục độc lập:

| Trục | Câu hỏi | Giá trị (mục trong survey) |
|---|---|---|
| Tri thức lấy từ đâu | Teacher cho student cái gì? | response (§2.1), feature (§2.2), relation (§2.3) |
| Sơ đồ huấn luyện | Teacher có được cập nhật cùng student không? | offline (§3.1), online (§3.2), self (§3.3) |
| Kiến trúc | Student trông như thế nào so với teacher? | §4 |
| Thuật toán | Cơ chế đặc biệt nào? | adversarial, multi-teacher, cross-modal, graph, attention, data-free, quantized, lifelong, NAS (§5.1–5.9) |
| Ứng dụng | Dùng ở đâu? | thị giác, NLP, giọng nói, khác (§7) |

**Cách dùng:** khi gặp một bài KD mới, hãy tự hỏi lần lượt (1) tri thức thuộc loại nào, (2) sơ đồ huấn luyện nào, (3) student/teacher quan hệ ra sao, (4) có cơ chế phụ nào. Việc này giúp so sánh các bài mà không bị chìm trong tên riêng. Cẩn thận: survey không bàn về **token-level vs sequence-level** theo nghĩa LLM, hay **on-policy vs off-policy**; những trục đó có trong preface 00 và các bài 15, 27–34.

## 3. Phương pháp chi tiết: phân loại của survey

### 3.1 Response-based knowledge (§2.1)

Tri thức là phản hồi của lớp đầu ra teacher. Với logit $z$:

$$
\mathcal L_{ResD}(z_t,z_s)=\mathcal L_R(z_t,z_s)
$$

Dạng soft target:

$$
\boxed{p(z_i,T)=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}},\qquad
\mathcal L_{ResD}=\mathcal L_R\big(p(z_t,T),p(z_s,T)\big)
$$

trong đó $\mathcal L_R$ thường là KL. Student loss luôn là CE giữa nhãn thật và $p(z_s,T{=}1)$. Survey nhận xét: hiệu quả của soft target tương tự **label smoothing / regularizer**; hạn chế: chỉ dùng lớp cuối nên bỏ qua giám sát trung gian và gắn với học có giám sát (vì là phân phối lớp). Mở rộng cho các tác vụ khác: object detection (logit + offset hộp), pose estimation (heatmap mỗi landmark).

### 3.2 Feature-based knowledge (§2.2)

$$
\boxed{\mathcal L_{FeaD}(f_t,f_s)=\mathcal L_F\big(\Phi_t(f_t(x)),\Phi_s(f_s(x))\big)}
$$

$\Phi_t,\Phi_s$ là hàm biến đổi (khi hình dạng khác nhau), $\mathcal L_F$ là hàm tương đồng. Đại diện (Table 1 của survey): FitNet (feature, lớp hint, L2), NST (mẫu chọn lọc neuron, MMD), AT (bản đồ attention, nhóm nhiều lớp, L2), FT (paraphraser, L1), AB (ranh giới kích hoạt, trước ReLU, L2), Margin ReLU của Heo et al., AdaIN (thống kê feature), ALP-KD (chiếu lớp có attention), SemCKD (cross-layer, gán lớp teacher cho từng lớp student bằng attention allocation), v.v. Khó khăn survey nêu: chọn lớp hint/guided; khác biệt lớn kích thước hint và guided.

### 3.3 Relation-based knowledge (§2.3)

Quan hệ giữa các lớp hoặc giữa các mẫu.

$$
\mathcal L_{RelD}(f_t,f_s)=\mathcal L_{R_1}\big(\Psi_t(\hat f_t,\check f_t),\Psi_s(\hat f_s,\check f_s)\big)
$$
$$
\mathcal L_{RelD}(F_t,F_s)=\mathcal L_{R_2}\big(\psi_t(t_i,t_j),\psi_s(s_i,s_j)\big)
$$

Công thức đầu: quan hệ giữa cặp feature map (ví dụ **FSP matrix** của Yim et al. 2017, Gram matrix giữa hai lớp). Công thức sau: quan hệ giữa cặp mẫu (**instance relations**): RKD (khoảng cách và góc), SP (ma trận tương đồng, chuẩn Frobenius), CCKD (tương quan giữa các mẫu), PKT (phân phối xác suất tương đồng), IRG (đồ thị quan hệ mẫu), DarkRank, v.v. (Table 2). Survey còn liên hệ với **privileged information** (Vapnik & Izmailov; Lopez-Paz et al.).

### 3.4 Sơ đồ huấn luyện (§3)

- **Offline:** teacher huấn luyện trước, cố định; hai giai đoạn. Ưu: đơn giản. Nhược: teacher lớn tốn thời gian huấn luyện; **khoảng cách capacity** giữa teacher và student luôn tồn tại, student phụ thuộc nhiều vào teacher.
- **Online:** teacher và student cập nhật đồng thời, end-to-end, huấn luyện một pha, song song hiệu quả. Ví dụ: deep mutual learning (Zhang et al. 2018b; các mạng học lẫn nhau), co-distillation (Anil et al. 2018), kiến trúc nhiều nhánh dùng chung backbone (Zhu & Gong 2018, ONE), feature fusion cho teacher (Kim et al. 2019b), online adversarial (Chung et al. 2020). Hạn chế: khó có teacher dung lượng lớn trong thiết lập online.
- **Self-distillation:** cùng một mạng làm teacher và student, coi là trường hợp đặc biệt của online. Ví dụ: chưng cất từ phần sâu xuống phần nông của mạng (Zhang et al. 2019b), snapshot distillation (epoch sớm làm teacher cho epoch muộn), early-exit mimic, teacher-free KD (Yuan et al. 2020, dựa trên phân tích label smoothing), class-wise self-KD, và **Born-Again Networks** (Furlanello et al. 2018: student ở bước $t$ làm teacher ở bước $t+1$; survey xếp vào nhóm này và cũng nhắc trong multi-teacher).
- Survey ví với việc học của con người: offline = thầy dạy trò; online = cùng học; self = tự học; ba loại có thể kết hợp (Sun et al. 2021).

### 3.5 Kiến trúc teacher-student (§4)

Student thường là: (1) bản đơn giản hoá của teacher (ít lớp, ít kênh); (2) bản lượng tử hoá (giữ cấu trúc); (3) mạng nhỏ với phép toán hiệu quả (depthwise separable: MobileNet, ShuffleNet); (4) mạng nhỏ có cấu trúc toàn cục tối ưu (NAS); (5) cùng mạng với teacher. Khoảng cách capacity làm giảm chuyển giao: **Teacher Assistant** (Mirzadeh et al. 2020) chèn mô hình trung gian; Gao et al. (2021) dùng học phần dư (assistant học residual error). Các hướng khác: nén cấu trúc nhiều lớp thành một lớp (Nowak & Corso), chuyển giao block-wise giữ receptive field (Wang et al. 2018a), RL tự bỏ lớp dư (N2N learning), tìm student tối ưu bằng NAS (oracle-based KD, architecture-aware KD). Survey nhận xét: phần lớn công trình tập trung loại tri thức/loss, **thiết kế kiến trúc teacher/student còn ít được nghiên cứu**.

### 3.6 Thuật toán distillation (§5)

| Mục | Ý tưởng | Công thức/ghi chú của survey |
|---|---|---|
| 5.1 Adversarial | GAN sinh dữ liệu cho KD; discriminator phân biệt đầu ra student/teacher; hoặc online | Ví dụ: $\mathcal L_{KD}=\mathcal L_G(F_t(G(z)),F_s(G(z)))$; $\mathcal L_{GAN\text{-}KD}=\mathcal L_{CE}+\alpha\mathcal L_{KL}+\beta\mathcal L_{GAN}$; KD còn dùng để nén GAN |
| 5.2 Multi-teacher | Nhiều teacher: trung bình logit, hoặc teacher riêng cho response và feature, chọn ngẫu nhiên mỗi bước, mô phỏng nhiều teacher bằng nhiễu/stochastic block | Table 3 tóm tắt loại tri thức và sơ đồ |
| 5.3 Cross-modal | Teacher ở modality này, student ở modality khác (RGB → depth, video → skeleton, text → vision...) | Cần dữ liệu cặp; khó khi lệch modality |
| 5.4 Graph-based | Đồ thị làm vật mang tri thức hoặc điều khiển truyền tin | Cấu trúc đồ thị khó xây |
| 5.5 Attention-based | Attention map (AT), neuron selectivity... | xem bài 04 |
| 5.6 Data-free | Không có dữ liệu gốc: sinh dữ liệu từ teacher (GAN, tái tạo từ kích hoạt, DeepInversion, zero-shot KD) | Thách thức: sinh dữ liệu đa dạng chất lượng cao. "Data distillation" (nhãn do teacher gán cho dữ liệu không nhãn) được nhắc là loại tương tự |
| 5.7 Quantized | Teacher chính xác cao, student lượng tử hoá thấp bit | Polino et al. 2018; apprentice (Mishra & Marr) |
| 5.8 Lifelong | KD để tránh quên thảm hoạ (LwF, global distillation...) | Dùng ngoài nén |
| 5.9 NAS-based | Chọn kiến trúc student phù hợp teacher; KD để tăng hiệu quả NAS | TGSA, one-shot NAS... |

## 4. Thí nghiệm và kết quả chính

Survey **không tự chạy thí nghiệm**: Table 5 (CIFAR-10) và Table 6 (CIFAR-100) chép lại độ chính xác từ các bài gốc. Các con số trong ngoặc là độ chính xác baseline huấn luyện riêng, mũi tên là mức cải thiện của student. Vì nguồn khác nhau (cấu hình khác nhau), **không nên so sánh chéo các hàng**; chỉ đọc "student baseline → student sau KD" trong cùng hàng. Dưới đây là vài hàng minh hoạ chép đúng như bảng (bảng trích xuất từ PDF có thể sai vài chỗ, đã ghi chú).

### Một số hàng CIFAR-10 (Table 5)

| Phương pháp | Tri thức | Teacher (baseline) | Student (baseline) | Sau KD |
|---|---|---|---|---|
| FSP (Yim 2017), offline | RelK | ResNet26 (91.91) | ResNet8 (87.91) | 88.70 (+0.79) |
| FT (Kim 2018), offline | FeaK | ResNet56 (93.61) | ResNet20 (92.22) | 93.15 (+0.93) |
| IRG (Liu 2019g), offline | RelK | ResNet20 (91.45) | ResNet20-x0.5 (88.36) | 90.69 (+2.33) |
| SP (Tung & Mori 2019), offline | RelK | WRN-40-1 (93.49) | WRN-16-1 (91.26) | 91.87 (+0.61) |
| DML (Zhang 2018b), online | ResK | ResNet32 (92.47) | ResNet32 (92.47) | 92.68 / 92.80 (+0.33) |
| ONE (Zhu & Gong 2018), online | ResK | ResNet32+ONE | ResNet32 (93.07) | 94.01 (+0.84) |
| Xu & Liu (2019), self | ResK, FeaK | không có | ResNet32 (92.78) | 93.68 (+0.90) |

Ghi chú về độ tin cậy: hàng "TOFD, teacher ResNet152, student ResNeXt50-4: 94.49 → 97.09 (+2.60)" được in như vậy trong bảng; con số này trông bất thường so với các hàng khác nên chúng ta không dùng nó để rút ra kết luận.

### Một số hàng CIFAR-100 (Table 6)

| Phương pháp | Tri thức | Teacher (baseline) | Student (baseline) | Sau KD |
|---|---|---|---|---|
| FSP, offline | RelK | ResNet32 (64.06) | ResNet14 (58.65) | 63.33 (+4.68) |
| FT, offline | FeaK | ResNet110 (73.09) | ResNet56 (71.96) | 74.48 (+2.52) |
| RKD (Park 2019), offline | RelK, FeaK | ResNet50 (77.76) | VGG11 (71.26) | 74.66 (+3.40) |
| SSKD (Xu 2020a), offline | RelK, ResK | ResNet50 (79.10) | MobileNetV2 (65.79) | 72.57 (+6.78) |
| SemCKD (Chen 2021), offline | FeaK | ResNet-32x4 (79.42) | VGG13 (74.82) | 79.43 (+4.61) |
| DML (Zhang 2018b), online | ResK | MobileNet (73.65) | ResNet32 (68.99) | 76.13 / 71.10 (+8.11) |
| ONE (Zhu & Gong 2018), online | ResK | ResNet32+ONE | ResNet32 (68.82) | 73.39 (+4.57) |
| CS-KD (Yun 2020), self | ResK | không có | ResNet18 (75.29) | 78.01 (+2.72) |

(Với các hàng online như DML, "teacher" và "student" thực chất là hai mạng học lẫn nhau; hai số sau KD là độ chính xác của cả hai mạng, mũi tên tính cho student. Không nên dùng các hàng online để so sánh tuyệt đối với offline.)

**Kết luận của survey từ hai bảng (trích ý):** KD áp dụng được cho nhiều mô hình khác nhau; nén được các loại mạng; online KD qua học cộng tác cải thiện đáng kể; self-KD cũng cải thiện; offline thường chuyển tri thức feature còn online thường chuyển tri thức response; student nhẹ được lợi từ teacher dung lượng lớn.

## 5. Phân tích / ablation (những nhận định của survey)

Survey không có ablation riêng. Những nhận định đáng nhớ:

- **Response-based ≈ label smoothing/regularizer** (dẫn Kim & Kim; Müller et al. 2019; Ding et al. 2019).
- **Teacher lớn hơn không nhất thiết tốt hơn** do capacity gap (Mirzadeh et al. 2020; Gao et al. 2021). Mặt khác, từ lý thuyết sớm, mạng nông có thể học cùng biểu diễn với mạng sâu (Ba & Caruana, bài 01). Nên thiết kế student/teacher còn là bài toán mở.
- **Hint/guided layer sâu có thể over-regularize** (từ FitNets, bài 03).
- **Tri thức khác nhau bổ sung cho nhau** nhưng chưa có khung thống nhất để mô hình hoá kết hợp response, feature, relation.
- **Sequence-level KD** trong NLP (Kim & Rush 2016) và giọng nói (CTC): survey nói frame-level KD thường dùng response-based, còn sequence-level thường chuyển feature-based từ lớp hint (nhận xét ở §7.3). Gordon & Duh (2019) giải thích hiệu quả của sequence-level KD như tăng cường dữ liệu và regularization (§7.2); Zhou et al. (2019a) cho rằng hiệu năng của NAT mô hình KD phụ thuộc capacity và dữ liệu distill.
- **Hybrid compression** (KD + pruning/quantization) được nêu là hướng tương lai vì hầu hết kỹ thuật nén cần retrain/finetune; thứ tự áp dụng là câu hỏi mở.

## 6. Giới hạn

**Do tác giả nêu (§8.1, 8.2):**
- Chất lượng tri thức: chưa biết ảnh hưởng riêng và bổ sung của từng loại tri thức; chưa có khung hợp nhất.
- Kiến trúc teacher-student còn ít được nghiên cứu.
- Hiểu biết lý thuyết và đánh giá thực nghiệm chưa đủ; đo chất lượng tri thức hay chất lượng cặp teacher-student là rất khó.
- Chưa nhiều nghiên cứu kết hợp KD với nén khác; thứ tự áp dụng là câu hỏi mở.

**Do chúng ta nhận thấy:**
- Phạm vi thiên về CNN và phân loại ảnh; NLP chỉ có NMT và BERT nén (BERT-PKD, TinyBERT, DistilBERT); không có LLM sinh văn bản lớn, không có on-policy/RL, không có ước lượng top-k hay vấn đề tokenizer.
- Hai bảng hiệu năng chép từ nhiều bài với cấu hình khác nhau (baseline khác, augmentation khác), nên chỉ dùng để thấy xu hướng; không phải benchmark có kiểm soát. Khi trích xuất từ PDF có một số số liệu in lệch cột.
- Nhiều trích dẫn và mô tả ngắn gọn, ít phân tích cơ chế; không nêu phương trình gradient hay phân tích $T^2$ (xem bài 02 và preface 00).
- Nhiều nghiên cứu sau 2021 (KD cho LLM, reasoning distillation, scaling laws) nằm ngoài phạm vi; cần bài 27 và các bài 28–47.

## 7. Liên hệ với các bài khác

Bản đồ từ survey sang lộ trình của thư mục này:

| Mục trong survey | Bài trong lộ trình |
|---|---|
| §2.1 Response-based | 01 Ba & Caruana, 02 Hinton; nâng cao: 14 Decoupled KD |
| §2.2 Feature-based | 03 FitNets, 04 Attention transfer; Transformer: 18 TinyBERT, 19 MiniLM, 17 Patient KD |
| §2.3 Relation-based | 12 Relational KD, 13 CRD (đối nghịch/tương phản, liên quan §5.3 contrastive của Tian et al.) |
| §3.2 Online | 07 Deep mutual learning |
| §3.3 Self-distillation | 06 Born-again networks; 22 DINO (self-distillation dạng khác) |
| §4 Capacity gap, TA | 08 Teacher assistant, 09 Efficacy of KD, 10 Does KD really work, 11 Patient and consistent teacher, 41 Capacity gap law |
| §5.6 Data-free / data distillation | 15 Sequence-level KD, 35 Self-Instruct, 36 Distilling step-by-step, 37 Orca, 38 DeepSeek-R1 distillation |
| §5.3 Cross-modal | 24 CLIP-KD, 25 LLaVA-MoD, 26 LLaVA-KD |
| §7.2 NLP | 15, 16 DistilBERT, 17, 18, 19 |
| §7.3 Speech | 46 Distil-Whisper |
| §8 Hybrid compression | các bài pruning + KD của dự án (thư mục khác) |
| Sau 2021 | 27 LLM KD survey, 28–34 (MiniLLM, GKD, DistiLLM, Rethinking KL, Speculative KD, TAID, DistiLLM-2), 40 Gemma 2, 42 Distillation scaling laws, 44 Qwen3, 45 Sparse logit sampling, 47 ShortOPD |

## 8. Ứng dụng cho dự án (suy luận)

- **Định vị dự án trên bản đồ:** dự án của chúng ta (teacher 4B → student prune 0.3B/0.1B) là **offline distillation** với student là *bản đơn giản hoá cấu trúc của teacher* (theo mục (1) trong §4: ít lớp/kênh hơn). Tri thức gồm response-based (phân phối token) và có thể feature-based (hidden state tại điểm cắt); relation-based chưa rõ lợi ích.
- **Gợi ý từ hướng "hybrid compression" của survey:** survey nói thứ tự áp dụng nén và KD là câu hỏi mở. Dự án đang làm "prune rồi distill"; thí nghiệm sắp xếp lại (distill xen kẽ từng bước prune nhỏ, như các bài sau về pruning+KD sẽ bàn) là hướng đáng thử. Survey chỉ nêu câu hỏi, không có câu trả lời.
- **Bài học về capacity gap (§4):** từ 4B xuống 0.1B (tỉ lệ khoảng 40 lần) có thể rơi vào vùng teacher quá lớn; mô hình trung gian (TA, bài 08) như bản 0.3B làm cầu nối cho 0.1B là ứng viên. Đây là suy luận; cần đối chiếu bài 08, 41.
- **Khớp hidden state vs attention-map vs relation:** theo bản đồ, khi chiều ẩn đổi sau pruning, AT-style (bài 04) và relation-style (RKD) không cần projector; hidden-state L2 (FitNets) cần. Chọn theo chi phí và kết quả D_lin/D_res.
- **Ứng dụng khác:** data-free/data distillation (§5.6) liên quan khi không đủ dữ liệu OCR có nhãn: dùng teacher 4B gán nhãn cho ảnh tài liệu không nhãn (xem 01, 15). Cross-modal ít liên quan trực tiếp.
- Hạn chế: survey là CNN-centric; mọi chuyển giao sang VLM hybrid GatedDeltaNet là ngoại suy.

## 9. Thí nghiệm gợi ý

1. **Bảng tự điền cho dự án:** lập ma trận (response / feature / relation) × (cắt layer / cắt neuron MLP) và chạy từng ô với ngân sách huấn luyện bằng nhau, ghi CER/WER. Mục đích: kiểm chứng nhận xét của survey rằng các loại tri thức bổ sung nhau, và xem loại nào cứu được trường hợp cắt neuron MLP.
2. **Teacher trung gian:** thêm bản 0.3B (đã distill) làm teacher cho 0.1B; so với 4B → 0.1B trực tiếp (kiểm chứng capacity gap theo bài 08).
3. **Hybrid xen kẽ:** thay vì prune một lần tới 0.1B, prune 3 bước (0.8B → 0.5B → 0.3B → 0.1B) với distill sau mỗi bước, so với prune một bước, cùng tổng số token huấn luyện.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Response-based | khớp output (logit/soft target) của teacher; $p(z_i,T)=\exp(z_i/T)/\sum_j\exp(z_j/T)$ |
| Feature-based | khớp feature trung gian, $\mathcal L_F(\Phi_t(f_t),\Phi_s(f_s))$ |
| Relation-based | khớp quan hệ giữa lớp (FSP) hoặc giữa mẫu (RKD, SP, CCKD...) |
| Offline / Online / Self | teacher cố định / cùng huấn luyện / cùng một mạng |
| Capacity gap | student quá nhỏ so với teacher thì chuyển giao kém; TA là một giải pháp |
| Data-free KD | sinh dữ liệu từ teacher khi không có dữ liệu gốc |
| Quantized KD | student lượng tử hoá, teacher chính xác cao |
| Hybrid compression | kết hợp KD và pruning/quantization; thứ tự là câu hỏi mở |
| Các bảng 5–6 | chép từ bài gốc, chỉ đọc theo từng hàng, không so chéo |

## 11. Kết luận ngắn

Survey cung cấp từ vựng và hệ trục phân loại (loại tri thức, sơ đồ huấn luyện, kiến trúc, thuật toán) để định vị mọi bài KD thời kỳ CNN, và nêu rõ các câu hỏi mở còn nguyên giá trị: tri thức nào bổ sung cho nhau, thiết kế cặp teacher-student, và kết hợp KD với nén khác. Hai bảng hiệu năng chỉ nên đọc theo xu hướng. Bản đồ này không bao quát LLM; phần đó nằm ở các bài 27–47 của lộ trình.

**Link gốc:** https://arxiv.org/abs/2006.05525
