# Causal Distillation for Language Models (2022)

**Tác giả:** Zhengxuan Wu, Atticus Geiger, Joshua Rozner, Elisa Kreiss, Hanson Lu, Thomas Icard, Christopher Potts, Noah D. Goodman (Stanford).
**Venue / arXiv:** arXiv:2112.02505 (bản v2, 3/6/2022; bản text không nêu venue, paper được biết tới là NAACL 2022).
**Nguồn:** https://arxiv.org/abs/2112.02505
**Chủ đề:** Thêm vào distillation chuẩn một loss thứ ba, **DIITO** (distillation interchange intervention training objective), buộc student bắt chước *cấu trúc nhân quả* của teacher, nối distillation với interpretability (causal abstraction, interchange intervention).
**Vị trí trong lộ trình:** Đọc trước: 16 (DistilBERT, baseline của bài này), 18 (TinyBERT), 19 (MiniLM). Bài này thêm: không chỉ hỏi "student có cùng đầu ra / cùng hidden state không" mà hỏi "student phản ứng thế nào khi **can thiệp** vào trạng thái trong, có giống teacher không". Liên quan phần cuối lộ trình về hiểu biểu diễn bên trong.

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch nguyên văn. Phần "Ứng dụng cho dự án" là suy luận của chúng ta. Một số số trong Table 1 không khớp với con số tóm tắt ở abstract (xem mục 4); tôi báo cả hai, không tự sửa.

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

Distillation chuẩn cho mô hình ngôn ngữ (DistilBERT, 16) huấn luyện student với ba loss: loss tác vụ (MLM), loss bắt chước đầu ra teacher (KL/CE mềm), và loss cosine giữa hidden state cuối. Các biến thể (17, 18) thêm loss khớp hidden state/attention tầng giữa. Tất cả đều ép student **giống teacher ở trạng thái quan sát được**, bất kể trạng thái đó có vai trò nhân quả trong phép tính hay không. Tác giả cho rằng điều này chưa tối ưu.

Ba khái niệm nền (viết cho người mới):

- **Causal abstraction**: mô hình đơn giản (cấp cao) là "trừu tượng nhân quả" của mô hình phức tạp (cấp thấp) nếu có một ánh xạ giữa biến cấp cao và nhóm neuron cấp thấp sao cho **can thiệp** vào hai bên cho kết quả tương ứng. Nói cách khác student "giải thích" teacher một cách trung thực.
- **Can thiệp (intervention)**: ghi đè giá trị của một nhóm neuron bằng giá trị khác rồi chạy tiếp (như thí nghiệm ablation/activation patching trong interpretability, ví dụ bài Transformer Circuits trong thư mục bên cạnh).
- **Interchange intervention**: lấy hai đầu vào $x_1$ (nguồn) và $x_2$ (nền); chạy $x_2$ nhưng thay giá trị của nhóm neuron $N$ bằng giá trị chúng có khi chạy $x_1$. Đây là một dạng phản thực: "đầu ra sẽ ra sao nếu tại vị trí này mô hình thấy thông tin từ câu khác?".

## 2. Ý tưởng chính

Với mỗi bước huấn luyện, lấy hai câu $x_1$, $x_2$. Chọn một tầng của student và các tầng tương ứng của teacher (ví dụ tầng 2 của student 3 tầng ứng với tầng 3-4 của teacher 6 tầng), chọn một số token, và **thay hidden state tại đó bằng hidden state của $x_1$** trong cả hai mô hình (cho chạy tiếp trên $x_2$). Rồi bắt phân phối đầu ra của student *sau can thiệp* khớp phân phối đầu ra của teacher *sau can thiệp tương ứng*. Nếu student khớp teacher dưới mọi can thiệp như vậy, nó đã nắm cách teacher "dùng" thông tin tại tầng đó. Loss mới này được **cộng thêm** vào ba loss của DistilBERT, và tất cả đều khả vi.

## 3. Phương pháp chi tiết

### 3.1 Toán tử

- $\textsc{GetVals}(M,x,N)$: giá trị các neuron $N$ khi mô hình $M$ xử lý $x$.
- $\textsc{SetVals}(M,N,v)$: mô hình mới trong đó neuron $N$ bị đặt bằng hằng $v$ (ghi đè tại chỗ nên gradient vẫn chảy qua $v$).
- $M^{N}_{x_1}=\textsc{SetVals}(M,N,\textsc{GetVals}(M,x_1,N))$.

$$
\mathrm{IntInv}(M,N,x_1,x_2)=\textsc{GetVals}\big(M^{N}_{x_1},\,x_2,\,N_y\big)
$$

với $N_y$ là neuron đầu ra. Tức là đầu ra khi chạy nền $x_2$ nhưng neuron $N$ lấy từ nguồn $x_1$.

### 3.2 Loss DIITO

Cho alignment $\Pi$ (ánh xạ nhóm neuron student sang nhóm neuron teacher):

$$
\boxed{\mathcal L^{DIITO}_{CE}=\sum_{x_1,x_2\in\mathcal D}CE^{S}\Big(\mathrm{IntInv}(S,N^S,x_1,x_2),\ \mathrm{IntInv}(T,\Pi(N^S),x_1,x_2)\Big)}
$$

$CE^S$ là cross-entropy mềm (smoothed) như trong Hinton. Loss tổng của student là tổ hợp tuyến tính, **trọng số bằng nhau**:

$$
\mathcal L=\mathcal L_{MLM}+\mathcal L_{CE}+\mathcal L_{Cos}+\mathcal L^{DIITO}_{CE}\ \ (+\ \mathcal L^{DIITO}_{Cos})
$$

$\mathcal L_{Cos}$: cosine embedding loss giữa hidden state tầng cuối của student và teacher; $\mathcal L^{DIITO}_{Cos}$ là phiên bản tương tự nhưng cả hai mô hình đều đang chịu can thiệp (mục A.2 của paper).

### 3.3 Alignment $\Pi$ (ba loại)

Gọi $L_T$, $L_S$ là số tầng teacher/student, $S_i^j$ là biểu diễn ở hàng (tầng) $i$, cột (token) $j$ của student.

- **FULL**: mọi tầng student: $\Pi(S_i^j)=\{T_{4i+k}^j:\ 0\le k<L_T/L_S\}$ (công thức trong paper dùng hệ số 4 cho thiết lập 12 và 3 tầng; hiểu như "khối tầng teacher tương ứng").
- **MIDDLE**: chỉ tầng giữa $L_S/2$ của student ↔ tầng giữa $L_T/2$ của teacher.
- **LATE**: hai tầng đầu của student ↔ hai tầng áp cuối của teacher ($L_T-2$, $L_T-1$).

Mỗi vòng lặp chọn **ngẫu nhiên một** tầng student được alignment để can thiệp, và ngẫu nhiên **30%** vị trí token trong chuỗi. Với FULL thử ba cách chọn token: liên tiếp (mặc định), ngẫu nhiên (+Random), token bị mask (+Masked).

### 3.4 Pseudo-code (Algorithm 1, rút gọn)

```
S.train(); T.eval()
D' = shuffle(D)
for ({x1,y1}, {x2,y2}) in zip(D, D'):
    N_S = sample_student_neurons()          # 1 tầng alignment, 30% token
    N_T = Pi(N_S)
    with no_grad:
        T_a = SetVals(T, N_T, GetVals(T, x1, N_T))   # teacher với trạng thái từ x1
        o_T = GetVals(T_a, x2, N_y^T)                 # logit teacher dưới can thiệp
    S_a = SetVals(S, N_S, GetVals(S, x1, N_S))        # student: gradient chảy qua
    o_S = GetVals(S_a, x2, N_y^S)
    L_DIITO = soft_CE(o_T, o_S)
    L = L_MLM + L_CE + L_Cos + L_DIITO
    L.backward(); optimizer.step()
```

### 3.5 Ví dụ (Figure 1)

Câu nền $x_2$ = "I ate some [MASK]." Ta thay biểu diễn của token [MASK] ở tầng giữa bằng biểu diễn token [MASK] của câu nguồn "The water [MASK] solid." (từ có nghĩa kiểu *froze*). Student (chưa luyện) dự đoán "salad" trong tình huống can thiệp, teacher dự đoán "pizza"; DIITO kéo phân phối của student về của teacher. (Hai từ này chỉ là minh họa của paper về cơ chế, không phải kết quả đo.)

## 4. Thí nghiệm và kết quả chính

**Thiết lập:** teacher BERT-base 12 tầng; student 6 tầng hoặc 3 tầng (12 head, hidden 768), khởi tạo từ các tầng bị bỏ qua của teacher (kiểu DistilBERT). Distill trên **WikiText-103M** (khoảng 0.1B token) để mô phỏng ngân sách nhỏ; 3 epoch, effective batch 240, mọi loss trọng số 1; khoảng 9 giờ trên 4 GPU A100. Fine-tune đánh giá: GLUE (trừ WNLI), CoNLL-2003 NER, SQuAD v1.1. Kết quả là trung bình 15 lần chạy (3 seed distill x 5 seed fine-tune; perplexity trung bình 3 lần).

Table 1 (dev, trích; trong ngoặc là độ lệch chuẩn; chỉ các cột PPL = perplexity WikiText, GLUE, SQuAD EM/F1):

| Mô hình | Tầng | PPL | GLUE | SQuAD EM | SQuAD F1 |
|---|---|---|---|---|---|
| BERT-base (Wikipedia+BookCorpus, 3.3B token) | 12 | 10.27 | 82.75 | 80.80 | 88.50 |
| DistilBERT chính thức (3.3B token) | 6 | 17.48 | 79.59 | 77.70 | 85.80 |
| DistilBERT (WikiText) | 6 | 15.69 | 75.80 | 70.23 | 79.99 |
| DIITO-MIDDLE | 6 | 14.32 | 76.71 | 71.93 | 81.32 |
| DIITO-LATE | 6 | 14.93 | 76.80 | 71.47 | 81.01 |
| DIITO-FULL | 6 | 13.59 | 76.67 | 71.96 | 81.33 |
| DIITO-FULL + $\mathcal L^{DIITO}_{Cos}$ | 6 | **13.45** | **77.14** | 71.94 | 81.35 |
| DistilBERT (WikiText) | 3 | 29.51 | 67.42 | (xem lưu ý) | 68.38 |
| DIITO-FULL | 3 | 24.85 | 69.36 | (xem lưu ý) | 70.50 |

*Lưu ý bảng:* cột SQuAD EM của hàng 3 tầng trong bản text trích có giá trị 26.04 cho DistilBERT và 58.7x cho các DIITO; 26.04 trùng với một giá trị perplexity ở hàng bên cạnh nên có khả năng bảng bị lẫn khi trích PDF. Tôi không dùng cột này.

Các con số tóm tắt trong abstract/văn bản: -2.24 perplexity (FULL + $\mathcal L^{DIITO}_{Cos}$, 13.45 so với 15.69 khớp với bảng), **+1.77%** GLUE, **+0.38%** F1 trên CoNLL-2003 (DIITO-MIDDLE; vượt cả teacher BERT-base +0.05 F1), **+2.46%** EM trên SQuAD v1.1. Tôi không tái lập được hai con số cuối bằng phép trừ trực tiếp từ cột Table 1 mà tôi thấy (ví dụ GLUE 77.14 - 75.80 = 1.34; EM tốt nhất 71.96 - 70.23 = 1.73), có thể do cách tính (ví dụ lấy tối ưu theo từng cấu hình hoặc bảng khác); cần đối chiếu với PDF gốc nếu trích dẫn.

Nhận xét tổng thể của paper: DIITO với 0.1B token đạt khoảng 97% điểm GLUE của DistilBERT chính thức (dùng 3.3B token), tức dùng ít hơn 97% dữ liệu; gain lớn nhất ở student nhỏ hơn (3 tầng) và alignment phong phú nhất (FULL).

## 5. Phân tích / ablation

- **Alignment càng giàu càng tốt:** FULL thường cho perplexity tốt nhất (13.59 vs 14.32 MIDDLE vs 14.93 LATE ở 6 tầng; 3 tầng: 24.85 / 26.04 / 25.97). Trên GLUE/SQuAD/NER chênh lệch giữa các alignment nhỏ.
- **Chọn token:** liên tiếp, ngẫu nhiên, hay token bị mask cho PPL gần nhau (13.59 / 13.95 / 13.99) và GLUE gần nhau (76.67 / 76.84 / 76.80); không rõ rệt.
- **Thêm $\mathcal L^{DIITO}_{Cos}$** cải thiện nhẹ PPL (13.45) và GLUE (77.14).
- **Theo từng task GLUE** (Table 2 trong phụ lục): DIITO thắng DistilBERT cùng thiết lập trên đa số tác vụ, lớn nhất ở CoLA (40.43 lên 45.17 với FULL + Cos, 6 tầng) và STS-B ở student 3 tầng (56.33 lên 66.93 MIDDLE); RTE thay đổi không rõ ràng.
- Phân phối điểm qua 15 lần chạy (Figure 2, 3) cho thấy gain nhất quán, student nhỏ nhận gain lớn hơn.

## 6. Giới hạn

*Do tác giả nêu:* chỉ thử trên WikiText nhỏ để mô phỏng ngân sách hạn chế; chưa thử dữ liệu lớn hơn ("leave for future research"). Gain trên NER nhỏ (tác giả tự nhận "small").

*Chúng ta nhận thấy:*
- Không có thí nghiệm kiểm tra *vì sao* DIITO giúp. Việc student "trở thành causal abstraction" là một lập luận khái niệm; paper không đo mức độ student có thực sự trừu tượng hóa teacher (ví dụ không đo độ trung thành của can thiệp trên tập đánh giá).
- Alignment dựa trên giả định tầng-theo-tầng (khối tầng đều); chưa có cách chọn alignment tự động.
- Chi phí mỗi bước tăng: thêm một lần forward có can thiệp cho cả teacher và student.
- Kích thước hiệu ứng khiêm tốn và thang so sánh nhỏ (6 tầng/3 tầng BERT, encoder, MLM); các con số tóm tắt trong abstract không khớp hoàn toàn với bảng chính (xem mục 4).
- Không so với TinyBERT/MiniLM; baseline duy nhất là DistilBERT cùng thiết lập.

## 7. Liên hệ với các bài khác

- **16 DistilBERT**: baseline, DIITO chỉ *cộng thêm* một loss.
- **18 TinyBERT / 19 MiniLM**: cùng muốn truyền thông tin tầng giữa; DIITO không so giá trị hay quan hệ mà so *phản ứng với can thiệp*.
- **12 Relational KD**: khác nhưng cùng tinh thần "dạy cấu trúc hơn là giá trị".
- **10 Does KD really work**: đặt câu hỏi student có thực sự khớp teacher; DIITO là một cách làm "khớp" mạnh hơn nhưng bằng chứng của bài này chưa kiểm tra độ khớp.
- **Transformer Circuits (2021)** trong thư mục liên quan: activation patching/path expansion là công cụ phân tích; DIITO biến patching thành *tín hiệu huấn luyện*.
- **38 DeepSeek-R1 distillation, 28 MiniLLM**: ở hướng LLM, không dùng can thiệp; chủ yếu bắt chước đầu ra.

## 8. Ứng dụng cho dự án (suy luận)

- **Mức độ liên quan: trung bình thấp cho pipeline hiện tại, nhưng hữu ích như công cụ chẩn đoán.** Phần đánh giá thực nghiệm của paper hạn chế (BERT, MLM, dữ liệu nhỏ) và chưa có bằng chứng ở OCR/VLM hay decoder sinh.
- **Dùng interchange intervention để đo D_res:** ý tưởng sạch nhất cho đề tài: sau khi cắt, thay vì chỉ so hidden state, hãy hoán đổi hidden state giữa hai ảnh/trang (cùng tầng, cùng vị trí) trong student và teacher rồi so logit. Nếu student cắt neuron phản ứng khác teacher dưới hoán đổi, điều đó cho thấy hư hại không phải kiểu tuyến tính bù được (liên quan D_res), vì một phép tuyến tính tại điểm cắt sẽ không giữ được tương ứng can thiệp. Đây là giả thuyết của chúng ta.
- **Giải thích khả dĩ cho "cắt neuron MLP khó hồi phục hơn cắt tầng":** cắt tầng loại cả một khối nhưng phần còn lại có cấu trúc teacher nguyên vẹn; cắt neuron làm biến dạng đều biểu diễn ở mọi nơi. DIITO có thể phạt đúng sai lệch này, nhưng chưa có dữ liệu. Đừng coi là kết luận.
- **Chi phí:** thêm một forward có can thiệp cho cả teacher 4B làm tăng đáng kể thời gian huấn luyện; với GPU <= 8GB khi train có thể phải tắt hoặc chạy mẫu thưa (chỉ một tầng, 30% token như paper).
- **GatedDeltaNet:** can thiệp theo vị trí token hợp lý với tầng có trạng thái hồi quy hay không (trạng thái tóm tắt cả lịch sử, thay một vị trí có thể lan ra mọi vị trí sau) là câu hỏi mở; không nên áp dụng mà không thử.

## 9. Thí nghiệm gợi ý

1. **Chẩn đoán hoán đổi:** với model đã cắt (tầng vs neuron), chạy interchange giữa hai ảnh tại tầng $k$ rồi đo khoảng cách logit student-teacher. So sánh tầng cắt vs neuron cắt; xem có khớp với khoảng cách D_res tính bằng cách thông thường không.
2. **DIITO nhẹ cho ViT:** áp dụng DIITO trên encoder ViT, can thiệp 30% patch ở một tầng giữa, thêm vào loss distill hiện tại; đo accuracy so với không có.
3. **Ablation alignment:** MIDDLE vs FULL cho decoder đã cắt 0.3B; chỉ mất một forward thêm nếu chọn 1 tầng ngẫu nhiên mỗi bước.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Causal abstraction | Mô hình đơn giản là mô tả trung thực về hiệu ứng nhân quả của mô hình phức tạp |
| Interchange intervention | Chạy $x_2$ nhưng neuron $N$ lấy giá trị từ $x_1$ |
| Alignment $\Pi$ | FULL / MIDDLE / LATE: neuron student ↔ neuron teacher |
| $\mathcal L^{DIITO}_{CE}$ | CE mềm giữa đầu ra student và teacher dưới can thiệp tương ứng |
| Tổng loss | $\mathcal L_{MLM}+\mathcal L_{CE}+\mathcal L_{Cos}+\mathcal L^{DIITO}_{CE}(+\mathcal L^{DIITO}_{Cos})$, trọng số 1 |
| Lấy mẫu | 1 tầng alignment ngẫu nhiên/bước, 30% token |
| Kết quả | 6 tầng: PPL 13.45 vs 15.69; GLUE 77.14 vs 75.80 (Table 1) |

## 11. Kết luận ngắn

Causal Distillation đề xuất một loss bổ sung đặt khái niệm interchange intervention của interpretability vào quy trình distillation, với kết quả cải thiện nhất quán nhưng khiêm tốn so với DistilBERT cùng ngân sách nhỏ. Giá trị lớn hơn với chúng ta là phương pháp luận: thay vì chỉ so hidden state, so **phản ứng dưới can thiệp** để đo xem student có thật sự giữ được cơ chế tính toán của teacher hay không. Bằng chứng cho OCR/VLM chưa có; hãy dùng nó như công cụ chẩn đoán trước khi dùng như loss.

Link gốc: https://arxiv.org/abs/2112.02505
