# Speculative Knowledge Distillation: Bridging the Teacher-Student Gap Through Interleaved Sampling (2025)

**Tác giả:** Wenda Xu, Rujun Han, Zifeng Wang, Long T. Le, Dhruv Madeka, Lei Li, William Yang Wang, Rishabh Agarwal, Chen-Yu Lee, Tomas Pfister (UCSB; Google Cloud AI Research; Google DeepMind; CMU).  
**Venue / arXiv:** arXiv:2410.11325 (v3, 27/04/2025); ICLR 2025.  
**Nguồn:** https://arxiv.org/abs/2410.11325  
**Chủ đề:** On-policy distillation cho LLM với lấy mẫu xen kẽ (interleaved) giữa student và teacher, lấy cảm hứng từ speculative decoding.  
**Vị trí trong lộ trình:** Đọc trước: 29 (GKD, on-policy KD), 30 (DistiLLM, cùng bàn về nhiễu của dữ liệu student). Bài này giải quyết đúng điểm yếu của on-policy KD: khi student còn kém, chuỗi nó sinh ra là OOD đối với teacher nên nhãn của teacher kém tin cậy. Cách sửa: để teacher thay các token tệ ngay trong lúc sinh. Đọc tiếp: 33 (TAID, giải quyết khoảng cách teacher-student theo cách khác), 34 (DistiLLM-2), 47 (ShortOPD).

> Ghi chú học tập bằng tiếng Việt, diễn giải lại, không phải bản dịch. Mục "Ứng dụng cho dự án" là suy luận của chúng ta. Một số hình trong bản trích xuất bị vỡ nên chỉ số trong bảng đọc rõ mới được trích.

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

Hai họ phương pháp soft-label KD cho LLM:

- **Supervised KD:** huấn luyện trên dữ liệu cố định (nhãn gốc hoặc chuỗi teacher sinh). Lệch phân phối với chuỗi student tự sinh lúc suy luận (exposure bias).
- **On-policy KD** (GKD, note 29): student tự sinh, teacher chấm logit từng token. Giải quyết mismatch nhưng có điểm yếu: nếu student ban đầu yếu, chuỗi nó sinh có thể rất tệ (lặp, vô nghĩa), tức là **OOD đối với teacher**, nên logit của teacher trên những tiền tố này không đáng tin. Paper cho thấy on-policy KD rất nhạy với khởi tạo student: ví dụ Qwen2-0.5B-IT dịch Assamese-English, chuỗi on-policy bị lặp "in the past, in the past, ..." (COMET 36), trong khi SKD cho chuỗi hợp lý (COMET 72).

Gloss: **speculative decoding** là kỹ thuật tăng tốc suy luận: model nhỏ (draft) đề xuất vài token, model lớn kiểm tra song song và chấp nhận/loại bỏ. **Top-k sampling**: chỉ lấy mẫu trong k token xác suất cao nhất.

## 2. Ý tưởng chính

SKD (Speculative KD) kết hợp hai nguồn mẫu *ở mức token*: student đề xuất một đoạn token (on-policy), teacher kiểm tra song song và chấp nhận token nếu nó nằm trong **top-K** của teacher; token đầu tiên bị loại thì teacher lấy mẫu thay, các token sau bị bỏ và student tiếp tục sinh từ đó. Kết quả: chuỗi huấn luyện vừa gần phân phối suy luận của student (phần lớn token là của student), vừa không rơi vào vùng teacher lạ. Đầu train, student yếu nên bị bác nhiều, SKD giống supervised KD; về sau student tốt hơn, tỉ lệ chấp nhận cao, SKD gần như on-policy KD. Chuyển pha này tự động, không cần lịch cứng.

## 3. Phương pháp chi tiết

### 3.1 Mục tiêu huấn luyện

Dùng cùng dạng loss như GKD, trung bình theo token của divergence giữa phân phối teacher và student:

$$D(M_t\|M_s)(y|x)=\frac{1}{L_y}\sum_{i=1}^{L_y}D\big(M_t(\cdot|y_{<i},x)\,\|\,M_s(\cdot|y_{<i},x)\big)$$

Điểm mới nằm ở cách tạo chuỗi $y$, không phải ở loss. (Paper dùng divergence $D$ như một siêu tham số; phụ lục A bàn các metric.)

### 3.2 Lấy mẫu xen kẽ và tiêu chí chấp nhận

Cho prompt $x$, student sinh $\gamma$ token $(y_i,\dots,y_{i+\gamma})$ theo $M_s$. Teacher đánh giá song song cả đoạn. Token $y_i$ được **chấp nhận** nếu $y_i\in\text{top-}K(M_t(\cdot|y_{<i},x))$. Nếu bị loại: các token sau bị bỏ, $y_i$ được lấy mẫu lại từ $M_t(\cdot|y_{<i},x)$, rồi student tiếp tục sinh on-policy.

Chú ý khác speculative decoding thật: tiêu chí chấp nhận gốc của speculative decoding đảm bảo mẫu cuối đúng phân phối teacher (thoái hóa thành supervised KD); SKD cố tình dùng tiêu chí top-K để giữ mẫu *student* nhưng *có khả năng cao dưới teacher*.

$$\boxed{y_i\ \text{accept}\iff y_i\in \text{top-}K\big(M_t(\cdot\mid y_{<i},x)\big)}$$

Mặc định $K=25$ (không tune; tương tự giá trị thường dùng trong top-k sampling), $\gamma=5$ trong cài đặt thực.

### 3.3 Pseudo-code

```
Cho: student M_s, teacher M_t, prompts {x}, độ dài sinh, K, divergence D
for mỗi prompt x (hoặc mini-batch):
    y = []
    while chưa đủ độ dài / chưa EOS:
        đề xuất: student sinh gamma token (y_i..y_{i+gamma}) theo M_s
        teacher chấm song song gamma+1 vị trí (một forward)
        tìm vị trí j đầu tiên có y_j không thuộc top-K của M_t
        giữ y_i..y_{j-1}; thay y_j bằng mẫu từ M_t(.|y_<j, x); bỏ phần sau
        nối vào y; tiếp tục
    loss = mean_i D( M_t(.|y_<i,x) || M_s(.|y_<i,x) )   # trên chuỗi y vừa tạo
    cập nhật student
```

Hai trường hợp suy biến: nếu mọi token đều bị loại, SKD = supervised KD (toàn token teacher); nếu mọi token đều được chấp nhận, SKD = on-policy KD. Bản lõi của đề xuất dựa trên lý thuyết imitation learning (Ross & Bagnell; Ross et al.): chuyển dần nguồn mẫu từ expert sang learner là hiệu quả.

## 4. Thí nghiệm và kết quả chính

**Mô hình.** Teacher: Gemma-7B-IT và Qwen2-7B-IT, đã SFT theo từng tác vụ. Student: Gemma-2B-IT, Qwen2-0.5B-IT (instruction-tuned, chưa SFT tác vụ) hoặc bản SFT thêm. LR 1e-5, warmup 0.1, dropout 0.1; SFT 3 epoch chọn checkpoint val loss thấp nhất; $K=25$.

**Tác vụ và dữ liệu (dữ liệu student khoảng 1K mẫu, kịch bản data ít):**
- Dịch Assamese→English: Flores-200 (997 mẫu train; test 512), metric COMET.
- Tóm tắt hội thoại: DialogSum (1K train; test 1500), ROUGE-L.
- Suy luận số học: GSM8K (1K train; test 1319), accuracy, CoT.
- Math instruction following (task-agnostic): UltraInteract (10K train / 1K dev), đánh giá trên held-out Math, GSMplus, ASDiv, SVAMP; thử 1K và 10K prompt.

Lưu ý: baseline supervised KD, SFT, ImitKD thấy $(x,y)$ gốc; on-policy KD và SKD chỉ thấy prompt $x$.

**Bảng 1 (student là bản instruction-tuned, chưa SFT):**

| Baseline | Gemma 2B: Dịch COMET | Tóm tắt ROUGE-L | GSM8K Acc | Qwen 0.5B: Dịch COMET | Tóm tắt | GSM8K |
|---|---|---|---|---|---|---|
| SFT | 72.5 | 31.7 | 18.7 | 57.6 | 29.2 | 31.8 |
| Supervised KD | 73.3 | 34.1 | 22.5 | 57.5 | 31.3 | 35.0 |
| On-policy KD | 36.1 | 34.1 | 25.3 | 53.0 | 31.2 | 33.9 |
| ImitKD | 74.8 | 34.9 | 26.2 | 55.9 | 30.9 | 33.2 |
| **SKD** | **75.3** | **35.0** | **29.1** | 57.1 | **31.7** | **36.6** |

On-policy KD gần như sụp trong dịch (COMET 36.1), đúng hiện tượng student yếu sinh mẫu OOD. SKD thắng 5 trong 6 cột (riêng dịch với Qwen, supervised KD/SFT ngang SKD, không khác biệt có ý nghĩa thống kê).

**Bảng 2 (student đã SFT):**

| Baseline | Gemma SFT: Dịch | Tóm tắt | GSM8K | Qwen SFT: Dịch | Tóm tắt | GSM8K |
|---|---|---|---|---|---|---|
| SFT | 72.5 | 31.7 | 18.7 | 57.6 | 29.2 | 31.8 |
| Supervised KD | 73.5 | 34.4 | 22.0 | 59.3 | 32.3 | 31.9 |
| On-policy KD | 74.6 | 34.8 | 28.8 | 58.3 | 32.1 | 32.2 |
| ImitKD | 74.6 | 34.4 | 26.2 | 59.9 | 32.2 | 31.8 |
| **SKD** | **75.0** | **35.4** | **29.2** | **60.6** | **33.1** | **33.4** |

(Bản thân SFT hàng đầu của bảng 2 giống bảng 1 vì chính là checkpoint SFT dùng làm baseline.)

**Tăng so với SFT (Gemma-7B→2B, Hình 1/giới thiệu):** dịch low-resource +41.8%, tóm tắt +230%, số học +160% (theo cách tính tương đối của paper, nên không hiểu là điểm tuyệt đối). Trên held-out sau khi train với dữ liệu instruction toán: Math +198%, GSMplus +360%.

**Data cực ít (100 mẫu, Gemma-2B, Bảng 3):** student IT: SFT 59.8/23.0/8.11; Supervised KD 65.8/31.7/17.0; On-policy KD 33.7/32.2/12.1; SKD 67.0/32.3/18.8 (dịch/tóm tắt/GSM8K). Student SFT: Supervised KD 60.9/32.0/14.4; On-policy 63.4/32.4/16.1; SKD 63.8/32.6/16.7. Nhận xét của tác giả: SFT trên 100 mẫu gây overfit, nên SKD chạy end-to-end từ model IT (bỏ bước SFT) tốt hơn quy trình hai bước SFT+on-policy KD.

**Tăng tốc speculative decoding (Phụ lục K):** draft model huấn luyện bằng SKD có tỉ lệ chấp nhận token cao hơn khoảng 71% đến 85% so với model IT gốc, tăng tốc 1.21x (dịch) và 1.17x (tóm tắt) (Gemma-2B-IT).

## 5. Phân tích / ablation

- **Khởi tạo student:** on-policy KD có thể làm giảm hiệu năng theo thời gian nếu khởi tạo yếu (Fig. 4); SKD tốt cả với IT lẫn SFT. ImitKD (trộn ở mức chuỗi) không thấy lợi rõ khi student tốt hơn, nên trộn ở mức token của SKD tốt hơn trộn ở mức chuỗi.
- **Huấn luyện hai giai đoạn thủ công** (supervised KD nửa đầu, on-policy nửa sau, Fig. 5): thường bị vượt bởi chỉ supervised hoặc chỉ on-policy, và bị SKD vượt rõ. Tác giả lý giải: trộn thô hai pha không có thước đo tin cậy từng token; cơ chế chấp nhận của SKD đóng vai trò đó.
- **Tỉ lệ bị loại (rejection rate, Phụ lục L):** với Gemma-2B-IT, khởi đầu 40.9% token phải teacher lấy lại, giảm xuống 9.7% sau 150 bước đầu và 5.4% sau 1,225 bước; với Gemma-2B-SFT bắt đầu 4.7% và còn 0.1% cuối. Xác nhận quá trình chuyển từ supervised-like sang on-policy-like.
- **Chi phí sinh mẫu (Phụ lục M, ước lượng lý thuyết):** với giả định tỉ lệ loại 10%, $\gamma=5$, student 2B, teacher 7B, $D=256$, $L=200$: nếu phải sinh từ teacher thì supervised KD đắt khoảng 3.5x on-policy KD; SKD khoảng 1.72x on-policy KD, tức tiết kiệm khoảng 50% so với sinh trực tiếp từ teacher. Đây là ước lượng FLOP, không phải thời gian đo được.
- **Giá trị K (Phụ lục B):** nhiều giá trị K đều vượt supervised và on-policy KD; K tối ưu tuỳ tác vụ. Phụ lục C có thử nghiệm sơ bộ K thích nghi.
- Hội tụ validation: SKD có validation loss thấp nhất (Phụ lục N).

## 6. Giới hạn

**Tác giả nêu/ngụ ý:** K được cố định 25 không tune, K tối ưu phụ thuộc tác vụ và K thích nghi mới ở mức sơ bộ. Chi phí tính bằng FLOP lý thuyết. Trong bản trích xuất tôi không thấy mục Limitations riêng.

**Chúng ta nhận xét (suy luận):**
- Cỡ nhỏ: dữ liệu chủ yếu 1K mẫu; teacher tối đa 7B; chỉ hai họ Gemma/Qwen.
- Cần teacher chấm song song (white-box) và cùng tokenizer; mỗi bước cần forward teacher trên chuỗi dài, nên bộ nhớ train cao.
- Tiêu chí top-K dựa trên *hạng* chứ không dựa trên xác suất tuyệt đối. Khi teacher rất tự tin (phân phối nhọn), K=25 gần như chấp nhận mọi thứ trong top-25 dù xác suất cực thấp, nên "lọc" yếu. Khi teacher kém tự tin thì ngược lại. Paper không xét điều này (đây là nhận xét của ta).
- Chưa so với DistiLLM/MiniLLM, vì đối sánh chính là supervised, on-policy, ImitKD.
- Chưa phân tích khi student bị cắt cấu trúc (pruned), nơi chất lượng khởi đầu thấp hơn nhiều.

## 7. Liên hệ với các bài khác

- **29 GKD:** SKD là biến thể của on-policy KD chữa điểm yếu nhiễu nhãn; hai trường hợp biên của SKD chính là GKD với $\lambda=0$ và $\lambda=1$.
- **30 DistiLLM:** cùng nhận ra phản hồi nhiễu khi teacher gặp SGO lạ (Fig. 1 của DistiLLM); DistiLLM xử lý ở mức lịch dữ liệu (ít SGO lúc đầu), SKD xử lý ở mức token (teacher sửa).
- **15 Sequence-level KD:** SKD là "SeqKD từng phần": teacher sinh một số token thay vì cả chuỗi.
- **33 TAID:** cùng nhắm đến khoảng cách teacher-student, TAID bằng nội suy phân phối theo thời gian thay vì trộn token.
- **34 DistiLLM-2:** dùng cặp chuỗi teacher/student theo kiểu tương phản; có thể dùng dữ liệu kiểu SKD.
- **47 ShortOPD:** hướng on-policy cho mô hình đã cắt; SKD cho một cách ổn định giai đoạn đầu khi student yếu.

## 8. Ứng dụng cho dự án (suy luận)

- **Liên quan cao**, đặc biệt ngay sau khi cắt: student 0.3B hoặc 0.1B vừa bị cắt (nhất là cắt neuron MLP) sẽ sinh chuỗi tệ, đúng chế độ on-policy KD sập (COMET 36.1 trong thí nghiệm dịch). SKD cho phép dùng on-policy mà không cần giai đoạn SFT/off-policy dài trước.
- Cách áp dụng cho VLM OCR: ảnh trang là input $x$ cố định, student sinh transcript theo từng đoạn $\gamma$ token, teacher 4B kiểm tra song song (một forward có ảnh). Token student nằm ngoài top-K của teacher thì bị thay. Tỉ lệ thay (rejection rate) là chỉ số giám sát tốt: log nó theo bước để thấy student hồi phục (tương tự 40.9% → 5.4% trong paper).
- Với OCR, phân phối teacher rất nhọn, nên top-K theo hạng có thể quá rộng. Gợi ý (của ta): thử tiêu chí theo xác suất (ví dụ chấp nhận nếu $p_t(y_i)>\tau$) hoặc K nhỏ (1-5).
- Có thêm lợi ích phụ: khung này huấn luyện student tốt trong vai trò draft model cho speculative decoding (chấp nhận token cao hơn), phù hợp nếu về sau muốn triển khai với teacher lớn hơn.
- Ràng buộc: huấn luyện cần chạy teacher 4B (nhiều VRAM) nhưng chỉ ở giai đoạn train; inference 8GB không bị ảnh hưởng.

## 9. Thí nghiệm gợi ý

1. Trên student 0.3B vừa cắt (chưa phục hồi), so supervised KD, on-policy KD và SKD ($K\in\{1,5,25\}$), ghi CER và rejection rate theo bước; kiểm tra on-policy có sập như paper mô tả không.
2. Thay tiêu chí top-K bằng ngưỡng xác suất tuyệt đối của teacher và so sánh trên cùng ngân sách.
3. So cắt layer và cắt neuron MLP về rejection rate ban đầu: nếu cắt neuron cho tỉ lệ loại cao hơn nhiều, đó là bằng chứng gián tiếp rằng student sau cắt neuron lệch khỏi miền teacher xa hơn.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Vấn đề on-policy KD | Mẫu student yếu là OOD với teacher, nhãn kém tin cậy |
| Interleaved sampling | Student đề xuất, teacher kiểm tra và thay token bị loại |
| Tiêu chí chấp nhận | $y_i\in\text{top-}K(M_t)$, $K=25$ mặc định |
| $\gamma$ | Số token student đề xuất mỗi lượt (5 trong thực nghiệm) |
| Hai biên | Loại hết = supervised KD; chấp nhận hết = on-policy KD |
| Rejection rate | 40.9% → 5.4% (Gemma-2B-IT, 1,225 bước) |
| Ước lượng chi phí | SKD khoảng 1.72x on-policy; supervised (sinh bằng teacher) khoảng 3.5x |
| Loss | Giống GKD: trung bình theo token của $D(M_t\|M_s)$ |

## 11. Kết luận ngắn

SKD sửa điểm yếu cốt lõi của on-policy KD (nhãn teacher sai trên chuỗi student lạ) bằng cách để teacher chủ động thay các token lệch trong lúc sinh, tạo chuỗi vừa on-policy vừa nằm trong vùng teacher hiểu. Cơ chế chuyển từ supervised-like sang on-policy-like là tự động theo chất lượng student. Kết quả mạnh nhất khi student yếu hoặc dữ liệu ít, đúng tình huống sau khi cắt mô hình.

Nguồn gốc: https://arxiv.org/abs/2410.11325
