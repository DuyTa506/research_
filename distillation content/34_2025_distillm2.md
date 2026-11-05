# DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs (2025)

**Tác giả:** Jongwoo Ko, Tianyi Chen, Sungnyun Kim, Tianyu Ding, Luming Liang, Ilya Zharkov, Se-Young Yun (KAIST AI; Microsoft).  
**Venue / arXiv:** arXiv:2503.07067 (v2, 30/05/2025); ICML 2025 (PMLR 267).  
**Nguồn:** https://arxiv.org/abs/2503.07067  
**Chủ đề:** Distillation cho LLM bằng loss tương phản (contrastive): dùng loss khác nhau cho phản hồi do teacher sinh (SKL) và do student sinh (SRKL); curriculum cho hệ số skew; ứng dụng DPO, VLM, lượng tử hóa, speculative decoding.  
**Vị trí trong lộ trình:** Đọc trước: 29 (GKD), 30 (DistiLLM, nền tảng trực tiếp), 31 (Rethinking KL), 32 (Speculative KD, là baseline). Bài này là bước tiếp theo của DistiLLM: chỉ ra rằng loss và loại dữ liệu cần được "ghép đôi" thay vì dùng cùng một loss cho mọi mẫu. Đọc tiếp: 41 (Capacity gap law), 42 (Distillation scaling laws), 47 (ShortOPD).

> Ghi chú học tập bằng tiếng Việt, diễn giải lại, không phải bản dịch. Mục "Ứng dụng cho dự án" là suy luận của chúng ta. Công thức cập nhật $\alpha$ trong Algorithm 1 bị vỡ trong bản trích xuất PDF nên chỉ mô tả bằng lời.

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

Phần lớn nghiên cứu KD cho LLM làm một trong hai việc: (a) thiết kế **loss** mới (reverse KL, JSD, skew KL), hoặc (b) chọn **dữ liệu** huấn luyện (chuỗi do teacher sinh, tức TGO; chuỗi do student sinh, tức SGO). Nhưng hầu hết đều áp **cùng một loss cho mọi loại dữ liệu**. Bài này hỏi: có nên dùng loss khác nhau cho TGO và SGO không?

Gợi ý đến từ **preference alignment**: DPO tăng likelihood của phản hồi tốt $y_w$ và giảm likelihood của phản hồi xấu $y_l$ (gloss: **contrastive** = học bằng cách đẩy hai thứ ra xa nhau). Ý tưởng tự nhiên: tăng $q_\theta(y_t|x)$ cho phản hồi teacher và giảm $q_\theta(y_s|x)$ cho phản hồi student. Tuy nhiên DPKD (Li et al. 2024b, thay mô hình tham chiếu của DPO bằng teacher) hay bị **reward hacking** trong thực nghiệm: vì $p(y_s|x)$ (xác suất của teacher cho chuỗi student) vốn rất nhỏ, tỉ số $q_\theta(y_s)/p(y_s)$ có thể bị đẩy xuống vô hạn, student làm giảm quá mức likelihood của chính nó (NLL của phản hồi student lên 91.25 so với 20.29 của phản hồi teacher, Hình 1b) và sinh câu thoái hóa.

Nền tảng cần nhớ từ DistiLLM (note 30): skew KL $D^{(\alpha)}_{SKL}(x,y;p\|q_\theta)=D_{KL}(p\|\alpha p+(1-\alpha)q_\theta)$ và $D^{(\alpha)}_{SRKL}=D_{KL}(q_\theta\|(1-\alpha)p+\alpha q_\theta)$; adaptive off-policy với replay buffer.

## 2. Ý tưởng chính

KL thuận "kéo lên" (pulling-up) likelihood $q_\theta$ ở những vùng $p$ cao (đầu phân phối teacher, tức chuỗi teacher sinh), còn KL ngược "đẩy xuống" (pushing-down) $q_\theta$ ở những vùng $p$ thấp (đuôi, tức chuỗi student sinh mà teacher cho xác suất bé). Vậy một cách tự nhiên để tạo hiệu ứng tương phản trong KD: **SKL trên phản hồi teacher, SRKL trên phản hồi student**, gộp trong cùng một loss (CALD, Contrastive Approach for LLM Distillation). Khác DPKD: loss tuyến tính (phân rã được theo token) và có sự trộn $p$ với $q$ nên tự điều chỉnh, không đẩy $q_\theta(y_s)$ xuống vô hạn. Thêm hai cải tiến: (i) $\alpha$ của skew chọn thích nghi theo độ khó từng mẫu (curriculum), (ii) hệ số của số hạng SRKL tăng dần trong lúc train. Gộp lại thành DistiLLM-2.

## 3. Phương pháp chi tiết

### 3.1 Quan sát về KL và RKL (Hình 1)

Trên dữ liệu long-tail đồ chơi (kiểu Wu et al. 2024, note 31) và trên LLM (Mistral-7B → Danube2-1.8B): KL tăng $q_\theta$ ở đầu phân phối teacher và trên TGO; RKL giảm $q_\theta$ ở đuôi và trên SGO. Chi tiết toán ở Phụ lục B.1 (không đọc kỹ ở đây).

### 3.2 Loss CALD

$$\boxed{\mathcal L_{CALD}=\frac{1}{2|\mathcal D|}\sum_{(x,y_t,y_s)\sim\mathcal D}\Big[D^{(\alpha)}_{SKL}(x,y_t)+D^{(\alpha)}_{SRKL}(x,y_s)\Big]}$$

với $y_t\sim p(\cdot|x)$ (teacher) và $y_s\sim q_\theta(\cdot|x)$ (student). Viết lại (Remark 1): CALD tương đương một dạng

$$-\mathbb E_{y_t\sim p,\ y_s\sim q_\theta}\Big[\tfrac1\lambda\cdot\Big(\lambda\log\frac{\tilde q_\theta(y_t|x)}{p(y_t|x)}-\lambda\log\frac{q_\theta(y_s|x)}{\tilde p(y_s|x)}\Big)\Big]\ (\text{dạng gần với DPKD})$$

với $\tilde q_\theta=\alpha p+(1-\alpha)q_\theta$ và $\tilde p=\alpha q_\theta+(1-\alpha)p$ (cách viết đúng từng ký hiệu nằm ở Eq. 6; bản trích xuất hơi vỡ nên đây là diễn giải). Hai khác biệt quan trọng so với DPKD/DPO: (1) dạng tuyến tính thay vì log-sigmoid nên phân rã token-level và có trọng số theo $p(y_t|x)$ hoặc $q_\theta(y_s|x)$; (2) $\tilde q_\theta$ phụ thuộc tuyến tính vào $p$ (và $\tilde p$ vào $q_\theta$) nên chặn việc $q_\theta(y_s|x)$ giảm quá mức. Nhờ đó tránh reward hacking.

Dùng SKL/SRKL làm backbone cho kết quả tốt hơn KL/RKL thuần, dù KL/RKL cũng hiệu quả trong khung CALD.

### 3.3 Dữ liệu tối ưu (Section 3.2)

Câu hỏi: dùng SGO và dữ liệu cố định chất lượng cao thế nào cho tốt? Thí nghiệm với speculative generation (thay $y_t$ hoặc $y_s$ bằng $y_{spec}$ khi chấp nhận token nháp theo tiêu chí $q_\theta(y_{n+k}|\cdot)>\min(\epsilon^2,\epsilon\exp(-H(p(\cdot|\cdot))))$):
- Với số hạng SKL, hiệu năng cao nhất khi $y_{spec}=y_t$ ($\epsilon=1.0$): hướng dẫn mạnh từ phản hồi teacher quan trọng hơn là giảm mismatch.
- Với số hạng SRKL, cao nhất khi $y_{spec}=y_s$ ($\epsilon=0$), kể cả khi chất lượng thấp nhất: SGO chất lượng thấp lại hữu ích cho phía "đẩy xuống".
- Thay $y_t$ bằng phản hồi của LLM mạnh hơn teacher (Llama-3-8B, Gemma-2-9B, Phi-3-medium thay cho Mistral-7B) **không** tốt hơn; học trên phản hồi của chính teacher vẫn tốt hơn. Giải thích của tác giả: log-likelihood cao dưới teacher quan trọng hơn chất lượng tuyệt đối, vì hiệu ứng pulling-up lớn nhất ở đầu phân phối $p$.

Kết luận: TGO cho SKL, SGO cho SRKL. Phản hồi được thu thập theo **batch** trước mỗi epoch (dùng vLLM được) thay vì sinh ở mỗi iteration, vừa nhanh vừa gần với tinh thần adaptive off-policy (kết quả sơ bộ ở Phụ lục D.1).

### 3.4 Curriculum cho $\alpha$ và hệ số $\beta$

- **Curriculum $\alpha$:** $\alpha$ lớn ổn định và hội tụ nhanh nhưng làm khoảng cách $p$ và $\alpha p+(1-\alpha)q_\theta$ nhỏ, thu được ít thông tin; $\alpha$ nhỏ thì ngược lại. Giá trị tối ưu thay đổi theo cặp teacher-student và theo epoch (Bảng 1: epoch sau thường dùng $\alpha$ bằng hoặc nhỏ hơn epoch đầu). Ý tưởng: mẫu "dễ" (p và q gần nhau) dùng $\alpha$ nhỏ; mẫu "khó" dùng $\alpha$ lớn. Dựa trên xấp xỉ bậc nhất (khai triển Mercator) $\log\frac{p(y|x)}{\tilde q_\theta^{(\alpha)}(y|x)}\simeq(1-\alpha)\,(p(y|x)-q_\theta(y|x))$ (Eq. 7), paper tính $\alpha$ dạng đóng cho từng mẫu sao cho tích $(1-\alpha)(p-q_\theta)$ nhất quán trong suốt huấn luyện (công thức chính xác nằm trong Algorithm 1, bị vỡ nên không chép lại).
- **Tăng dần hệ số SRKL:** loss có dạng $(1-\beta)D_{SKL}(x,y_t)+\beta D_{SRKL}(x,y_s)$ với $\beta$ tăng tuyến tính theo tiến độ huấn luyện (clip trong $[\beta_0,1]$). Lý do: đạt $q_\theta=p$ trên mọi $y_t,y_s$ khó vì capacity gap; tăng dần $\beta$ nghĩa là bớt chú trọng bắt chước teacher và tập trung vào phản hồi trực tiếp từ SGO, giảm mismatch train/infer.

### 3.5 Pseudo-code (Algorithm 1, rút gọn)

```
Cho: teacher p, student q_theta, prompts, alpha_0, beta_0, E epoch, T iter/epoch
for epoch e in 1..E:
    với mỗi prompt x: y_t ~ p(.|x), y_s ~ q_theta_{e-1}(.|x)     # sinh theo batch đầu epoch
    D_e = {(x, y_t, y_s)}
    for iter tau in 1..T:
        mini-batch B từ D_e
        # curriculum alpha (từng mẫu), dựa trên |p - q| trên phản hồi tương ứng
        alpha_t, alpha_s <- cập nhật từ p(.), q_theta(.) và alpha_0 (công thức Eq. 7)
        beta <- clip(e/E + tau/T ..., beta_0, 1)                  # tăng dần
        loss = mean[ (1-beta)*D_SKL^{alpha_t}(x, y_t) + beta*D_SRKL^{alpha_s}(x, y_s) ]
        cập nhật q_theta
```

## 4. Thí nghiệm và kết quả chính

**Thiết lập chung.** LoRA (rank 16, mọi lớp tuyến tính của attention và MLP), LR 5.0e-5, batch hiệu dụng 128, 2-3 epoch, A100 80GB. Instruction-following: 50K prompt UltraChat200k; đánh giá AlpacaEval, Evol-Instruct, UltraFeedback bằng LLM-as-a-judge (GPT-4o hoặc GPT-4o-mini) tính winning rate (WR) so với baseline cố định (text-davinci-003 cho AlpacaEval; gpt-3.5-turbo cho Evol-Instruct và UltraFeedback).

**Bảng 2 (WR trung bình qua ba benchmark, %):**

| Phương pháp | Qwen2-7B→Qwen2-1.5B | Mistral-7B→Danube2-1.8B | Gemma-2-9B→Gemma-2-2B |
|---|---|---|---|
| Teacher $M_T$ | 76.12 | 83.01 | 90.15 |
| Student $M_S$ (trước KD) | 30.33 | 27.02 | 28.62 |
| KD | 41.19 | 40.00 | 49.53 |
| SeqKD | 41.83 | 40.11 | 50.27 |
| ImitKD | 43.29 | 39.03 | 49.64 |
| GKD | 56.14 | 50.68 | 69.73 |
| DistiLLM | 56.35 | 52.37 | 70.30 |
| Speculative KD | 54.43 | 54.50 | 69.26 |
| **DistiLLM-2** | **58.69** | **56.45** | **74.83** |

DistiLLM-2 vượt baseline mạnh nhất kế tiếp trung bình +2.34%, +1.95%, +4.53% (Qwen2-1.5B, Danube2-1.8B, Gemma2-2B). Ngoại lệ: trên Evol-Instruct với Danube2-1.8B, Speculative KD (38.87) cao hơn DistiLLM-2 (32.84).

**Suy luận toán (Bảng 3, Pass@1, 50K mẫu MetaMathQA):**

| | Qwen2-Math-7B→1.5B: GSM8K | MATH | TB | Qwen2.5-Math-7B→1.5B: GSM8K | MATH | TB |
|---|---|---|---|---|---|---|
| Teacher | 83.93 | 41.28 | 62.61 | 89.31 | 44.82 | 67.07 |
| Student | 74.53 | 25.56 | 50.05 | 77.33 | 27.14 | 52.24 |
| GKD | 75.44 | 34.16 | 54.80 | 80.21 | 40.54 | 60.38 |
| DistiLLM | 75.59 | 34.54 | 55.07 | 81.05 | 41.14 | 61.10 |
| **DistiLLM-2** | **76.27** | **35.58** | **55.93** | **81.20** | **42.94** | **62.07** |

Tác giả nhấn mạnh student Qwen2.5 với DistiLLM-2 vượt cả teacher Qwen2 trên MATH.

**Code (Bảng 4, Pass@1; prompt từ WizardCoder):**

| | DS-Coder-6.7B→1.3B: HumanEval | MBPP | TB | Qwen2.5-Coder-7B→1.5B: HumanEval | MBPP | TB |
|---|---|---|---|---|---|---|
| Teacher | 85.37 | 82.54 | 83.96 | 75.61 | 74.60 | 75.61 |
| Student | 50.61 | 72.22 | 61.42 | 30.73 | 60.84 | 45.79 |
| GKD | 54.88 | 74.34 | 64.61 | 40.85 | 61.90 | 51.38 |
| DistiLLM | 53.65 | 74.34 | 64.00 | 39.63 | 62.17 | 50.90 |
| **DistiLLM-2** | **59.92** | **75.66** | **67.79** | **42.24** | **62.70** | **52.47** |

(Cột "TB" của Qwen2.5-Coder cho Teacher là 75.61 trong bản trích xuất, có vẻ lỗi bảng; trung bình đúng của 75.61 và 74.60 phải khoảng 75.1. Không rút gì thêm từ ô này.)

## 5. Phân tích / ablation

- **Thành phần (Bảng 5, WR trung bình tại UltraFeedback/Evol-Instruct, theo ba student; "AVG"):** DistiLLM v1: 44.61 (Qwen2), 28.78 (Danube2), 51.26 (Gemma2), AVG 41.55. Thêm contrastive (1): AVG 43.61. Thêm cả tăng $\beta$ (1)+(2): 44.80. Chỉ (1)+(3) curriculum $\alpha$: 45.22. Cả ba (DistiLLM-2): 47.24 / 32.80 / 59.53, AVG 46.50. Mỗi thành phần đều đóng góp.
- **Kích thước dữ liệu (Hình 3):** DistiLLM-2 thắng GKD và DistiLLM ở 13K/25K/50K (instruction), và ở các cỡ cho toán, code.
- **Capacity gap (Bảng 8):** student Qwen1.5-0.5B, teacher Qwen1.5-Chat 1.8B/7B/14B (WR trên Evol-Instruct): DistiLLM-2 = 66.31, 74.86, 76.78, tăng đơn điệu theo cỡ teacher, vượt DistiLLM +1.08, +2.75, +4.67. Tác giả nói DistiLLM v1 vật lộn với teacher 7B và 14B.
- **DPO sau KD (Bảng 6):** dùng student sau KD làm mô hình tham chiếu cho DPO: WR trung bình Qwen2-1.5B: GKD 56.66, DistiLLM 56.75, DistiLLM-2 60.33 (so với 30.09 của student SFT); Gemma-2-2B: 70.98, 68.58, 76.29 (so với 46.10).
- **VLM (Bảng 7):** teacher LLaVA-1.5-7B, student TinyLLaVA-1.4B, 83K prompt RLAIF-V; VQA accuracy trung bình OK-VQA và TextVQA: teacher 48.81, student 32.61, GKD 37.84, DistiLLM 35.24, DistiLLM-2 39.85.
- **Phục hồi mô hình lượng tử hóa INT4 (Bảng 9, Phi-3.5-mini-instruct):** AVG teacher BF16 85.05; INT4 + SFT 78.94; GKD 82.80; DistiLLM 83.35; DistiLLM-2 83.57 (KD qua LoRA).
- **Speculative decoding (Bảng 10):** draft model Llama-68m, tăng tốc với verifier Phi-3-medium: SFT x1.32, GKD x1.64, DistiLLM x1.71, DistiLLM-2 x1.97; với Phi-3.5-mini: x1.24, x1.58, x1.65, x1.84; tỉ lệ chấp nhận token cao nhất cũng thuộc DistiLLM-2 (0.487 và 0.522).

## 6. Giới hạn

**Tác giả nêu:** trong bản trích xuất không thấy mục Limitations; chỉ ghi nhận $\alpha$ tối ưu thay đổi theo cặp teacher-student và theo epoch (là động cơ của curriculum), và rằng DistiLLM v1 vật lộn với capacity gap.

**Chúng ta nhận xét (suy luận):**
- Cần sinh cả TGO lẫn SGO cho mỗi prompt ở mỗi epoch, chi phí sinh gấp đôi so với chỉ một nguồn; paper báo kết quả hiệu quả mẫu nhưng chi phí sinh của teacher không được so rõ trong thân bài.
- Đánh giá chủ yếu bằng LLM-as-a-judge (winning rate), có thể nhiễu và thiên vị; đa số kết quả là một cấu hình huấn luyện (LoRA) chứ không phải full fine-tuning.
- Huấn luyện bằng LoRA với student đã có sẵn khả năng (1.5B-2B); chưa biết hiệu quả trên student bị cắt cấu trúc mạnh.
- Công thức $\alpha$ phụ thuộc xấp xỉ bậc nhất, hợp lệ khi $p-q$ nhỏ; có thể sai lệch ở giai đoạn đầu khi student còn xa teacher.
- Cần cùng tokenizer/vocab cho teacher và student.
- So sánh Speculative KD ở Bảng 2 cho thấy ở một cột nó vượt DistiLLM-2, nên ưu thế không phải tuyệt đối.

## 7. Liên hệ với các bài khác

- **30 DistiLLM:** nền tảng; DistiLLM-2 giữ SKL/SRKL nhưng gán loss theo loại dữ liệu, thêm curriculum.
- **29 GKD, 32 Speculative KD:** hai baseline chính trong Bảng 2; SKD trộn token teacher/student trong cùng chuỗi, DistiLLM-2 giữ hai chuỗi tách riêng và dùng hai loss.
- **31 Rethinking KL:** hiện tượng head/tail (FKL ưu tiên head, RKL ưu tiên tail) chính là cơ sở cho "pulling-up/pushing-down"; DistiLLM-2 dùng tính chất đó thay vì xem chúng là lỗi.
- **33 TAID:** cũng xử lý capacity gap; TAID nội suy đích theo thời gian, DistiLLM-2 dùng $\alpha$ theo mẫu và $\beta$ theo thời gian.
- **35-37 (Self-Instruct, step-by-step, Orca):** các bài đó tạo dữ liệu (black-box); DistiLLM-2 chỉ ra rằng phản hồi tốt hơn từ LLM mạnh hơn không nhất thiết giúp khi teacher đã định trước.
- **41-42:** capacity gap law và scaling laws đo lại hiện tượng mà Bảng 8 minh họa.
- **47 ShortOPD:** hướng on-policy cho mô hình đã cắt; DistiLLM-2 cho một kết quả liên quan khi KD dùng để phục hồi mô hình nén (Phần 6.3).

## 8. Ứng dụng cho dự án (suy luận)

- **Liên quan cao.** Hai điểm trực tiếp: (1) bài này có phần **recover compressed model** (INT4 + LoRA) và nhấn mạnh KD áp dụng cho cặp "mô hình gốc ↔ mô hình đã nén (pruned/quantized)", đúng bài toán của ta; (2) khung "teacher dạy ở nơi teacher chắc chắn, student bị phạt ở nơi nó sai" phù hợp với OCR, nơi đáp án rất xác định.
- Với OCR: TGO = transcript do teacher 4B sinh trên ảnh (hoặc nhãn thật nếu có); SGO = transcript do student sinh. Số hạng SRKL sẽ phạt xác suất student đặt vào chuỗi sai (ví dụ lặp, bỏ dòng) mà teacher cho xác suất rất thấp. Đây là tín hiệu "âm" mà KD thông thường không có.
- Lưu ý: bài báo dùng LoRA; ta dự kiến phục hồi sau cắt thường cần cập nhật nhiều tham số hơn (đặc biệt cắt neuron MLP), nên LoRA có thể không đủ. Đây là suy đoán, nên kiểm tra full fine-tuning.
- Nhớ rằng mọi loss ở đây chỉ trên phân phối token đầu ra: không động đến tổn thương D_lin/D_res ở các lớp giữa. Có thể kết hợp với loss khớp đặc trưng (nếu có trong lộ trình) nhưng bài này không đề cập.
- Hệ số $\beta$ tăng dần và $\alpha$ thích nghi thêm vài siêu tham số; nên bắt đầu với DistiLLM (note 30) rồi thêm từng thành phần như Bảng 5.
- Phụ trợ: draft model cho speculative decoding (nếu sau này dùng teacher 4B ở bước suy luận).

## 9. Thí nghiệm gợi ý

1. Trên student 0.3B đã cắt: so DistiLLM (SKL thuần), CALD không curriculum, và DistiLLM-2 đầy đủ; đo CER và log-likelihood (NLL) của student trên chuỗi teacher và chuỗi student (như Hình 1b) để kiểm tra xem số hạng SRKL có làm giảm xác suất trên chuỗi student sai không và có dấu hiệu reward hacking không.
2. Ablation dữ liệu: thay TGO bằng nhãn ground-truth OCR (nếu có) và so với transcript của teacher 4B cho số hạng SKL, để xem "likelihood dưới teacher" hay "độ đúng" quyết định hơn.
3. So $\beta$ tăng dần (0.1→1) với $\beta$ cố định, ở cả student 0.3B lẫn 0.1B, để thấy mức capacity gap tác động tới lịch tối ưu.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| TGO / SGO | Phản hồi do teacher / student sinh |
| Pulling-up / pushing-down | KL thuận tăng $q$ ở vùng $p$ cao; KL ngược giảm $q$ ở vùng $p$ thấp |
| CALD | $\frac{1}{2}[D^{(\alpha)}_{SKL}(x,y_t)+D^{(\alpha)}_{SRKL}(x,y_s)]$ |
| DistiLLM-2 | CALD + curriculum $\alpha$ theo mẫu + $\beta$ tăng dần cho SRKL |
| Loss đầy đủ | $(1-\beta)D_{SKL}^{(\alpha_t)}(x,y_t)+\beta D_{SRKL}^{(\alpha_s)}(x,y_s)$ |
| Xấp xỉ | $\log\frac{p}{\tilde q^{(\alpha)}_\theta}\simeq(1-\alpha)(p-q_\theta)$ |
| Khác DPKD | Tuyến tính (phân rã token), có trộn $p,q$ nên chặn việc đẩy $q(y_s)$ xuống quá mức |
| Dữ liệu tối ưu | TGO cho SKL, SGO cho SRKL; LLM mạnh hơn teacher không giúp thêm |
| Kết quả chính | WR 58.69/56.45/74.83 (3 cặp); toán TB 55.93 / 62.07; code TB 67.79 / 52.47 |

## 11. Kết luận ngắn

DistiLLM-2 cho rằng loss và loại dữ liệu cần "khớp nhau": dùng skew KL trên phản hồi teacher để kéo student lên, skew reverse KL trên phản hồi student để đẩy xuống phần nó sinh sai, cộng với curriculum cho $\alpha$ và $\beta$. Đây là phiên bản mềm, ổn định của ý tưởng tương phản kiểu DPO áp vào KD. Kết quả vượt GKD, DistiLLM, Speculative KD ở hầu hết thiết lập (không phải tất cả), và dùng được để phục hồi mô hình lượng tử hóa, đáng thử cho bước phục hồi sau cắt.

Nguồn gốc: https://arxiv.org/abs/2503.07067
