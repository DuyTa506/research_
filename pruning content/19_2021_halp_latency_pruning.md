# HALP: Hardware-Aware Latency Pruning (2021)

**Tác giả:** Maying Shen, Hongxu Yin, Pavlo Molchanov, Lei Mao, Jianna Liu, Jose M. Alvarez (NVIDIA)  
**Venue / arXiv:** arXiv:2110.10811 (v1, 20/10/2021); bản text không nêu venue.  
**Nguồn:** https://arxiv.org/abs/2110.10811  
**Chủ đề:** Structured pruning (filter/channel) với ràng buộc **latency** đo thật trên GPU, bảng tra latency (lookup table), bài toán knapsack, nhóm neuron theo bậc thang latency.  
**Vị trí trong lộ trình:** Đọc sau 04 (Pruning filters), 05 (Network slimming), 07 (Molchanov importance 2019, tiêu chí Taylor mà HALP dùng) và 18 (2:4, tốc độ phần cứng). Bài này thêm: biến "mục tiêu nén" từ số tham số / FLOPs thành **latency đo trên đúng GPU đích**, và coi chọn số kênh mỗi lớp là bài toán phân bổ tài nguyên toàn cục. Tiếp theo: 20 (DepGraph, cách nhóm các tham số phụ thuộc), 25 (ZipLM, latency-aware cho LLM), 42 (Puzzle, chọn kiến trúc theo ràng buộc phần cứng).

> Đây là ghi chú học tập bằng tiếng Việt, không phải bản dịch. Phần "Ứng dụng cho dự án" là suy luận của chúng ta, không phải kết luận của paper. Paper chỉ thử trên CNN (ResNet, MobileNet, VGG, SSD), không có transformer/LLM.

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

Các phương pháp structured pruning trước đó thường nhắm vào **số tham số** hoặc **FLOPs** (số phép nhân-cộng). Paper nhắc rằng FLOPs là proxy kém: hai mạng có FLOPs gần nhau có thể có latency khác nhau rất nhiều. Lý do phần cứng: trên GPU, latency của một lớp conv theo số kênh có dạng **bậc thang** (staircase): giảm kênh trong một bậc không làm nhanh hơn, chỉ khi vượt qua một "vách" (latency cliff) mới giảm. Hệ quả: nếu cắt từng kênh lẻ, bạn mất accuracy mà không được lợi tốc độ; phải cắt **theo nhóm** khớp với bậc.

Thêm nữa, mỗi lớp có "giá" khác nhau: bỏ kênh ở lớp đầu (độ phân giải cao) thường tốn accuracy hơn và cũng giảm latency nhiều hơn lớp sau. Nên "cắt nhiều ở lớp sau hay ít ở lớp đầu để cùng một accuracy-latency" là bài toán tổ hợp cần lời giải toàn cục.

Các hướng latency-aware trước đó (theo paper): NAS (đắt), NetAdapt, AutoSlim, EagleEye (sinh nhiều ứng viên rồi đánh giá, tốn nhiều giờ GPU), MetaPruning/AMC (phải huấn luyện mạng phụ trợ).

Giải thích thuật ngữ:
- **Latency lookup table (LUT):** bảng đo sẵn thời gian chạy của từng lớp ứng với từng cấu hình số kênh vào/ra trên GPU đích.
- **Knapsack (bài toán cái túi):** chọn tập vật phẩm có giá trị tổng lớn nhất mà tổng "trọng lượng" không vượt sức chứa; ở đây giá trị = importance, trọng lượng = latency, sức chứa = ngân sách latency.
- **Taylor importance:** ước lượng loss tăng bao nhiêu nếu bỏ một neuron bằng khai triển Taylor bậc 1 (xem 07).
- **Neuron / filter:** một kênh đầu ra của lớp conv.

## 2. Ý tưởng chính

Xem việc cắt tỉa toàn mạng là bài toán "xếp đồ vào cái túi": mỗi nhóm neuron có một **giá trị** (độ quan trọng với accuracy, từ gradient) và một **giá** (latency nó đóng góp, tra từ bảng đo trên GPU thật). Cần chọn tập nhóm giữ lại sao cho tổng giá trị lớn nhất mà tổng latency ≤ ngân sách. Điểm phải xử lý: giá của một neuron chỉ đúng khi các neuron quan trọng hơn trong cùng lớp đều được giữ (giữ theo thứ tự quan trọng), nên cần một knapsack "tăng cường" có ràng buộc thứ tự. Và để khớp với bậc thang latency, các neuron được gom thành nhóm có kích thước bằng độ rộng bậc.

## 3. Phương pháp chi tiết

### 3.1 Bài toán tối ưu

Mạng có $L$ lớp, lớp $l$ có $N_l$ neuron. Bài toán chung:

$$\arg\min_{\hat W}\ \mathcal L(\hat W, D)\quad \text{s.t. } \Phi(f(\hat W, x_i))\le C$$

với $\Phi$ là ràng buộc (latency, FLOPs, bộ nhớ); paper tập trung vào latency. Dạng được giải:

$$\boxed{\max_{p_1,\dots,p_L}\ \sum_{l=1}^L I_l(p_l)\quad \text{s.t.}\ \sum_{l=1}^L T_l(p_{l-1},p_l)\le C,\ \ 0\le p_l\le N_l}$$

$p_l$: số neuron giữ ở lớp $l$ ($p_0$ cố định, vd. 3 kênh RGB); $I_l(p_l)$: độ quan trọng tối đa với $p_l$ neuron; $T_l(p_{l-1},p_l)$: latency lớp $l$ khi có $p_{l-1}$ kênh vào, $p_l$ kênh ra (tra bảng).

### 3.2 Importance

Prune trên lớp BatchNorm; importance của neuron $n$ ở lớp $l$ (Phương trình 3):

$$I_l^n = \left|g_{\gamma_l^n}\gamma_l^n + g_{\beta_l^n}\beta_l^n\right|$$

($g$ là gradient của tham số BN $\gamma,\beta$.) Paper nói khác với Molchanov 2019 ở chỗ dùng **giá trị tuyệt đối** thay vì bình phương, vì "slight improvements". Sắp xếp giảm dần trong từng lớp và cộng dồn:

$$I_l(p_l)=\sum_{j=1}^{p_l} I_l^j,\quad I_l^1\ge\dots\ge I_l^{N_l}.$$

Điểm trung bình qua nhiều minibatch được dùng khi xếp hạng.

### 3.3 Đóng góp latency của từng neuron

Latency lớp là tổng đóng góp neuron:

$$T_l(p_{l-1},p_l)=\sum_{j=1}^{p_l} c_l^j,\qquad c_l^j = T_l(p_{l-1},j)-T_l(p_{l-1},j-1).$$

Vì số kênh vào $p_{l-1}$ thay đổi khi lớp trước bị cắt, đóng góp được **cập nhật động** trong quá trình prune. Một số $c_l^j$ có thể âm (do bảng đo nhiễu / phi đơn điệu) và được xử lý riêng trong phần cài đặt (Appendix F); chi tiết xử lý không nằm trong đoạn đã đọc kỹ ở đây.

### 3.4 Knapsack tăng cường (augmented)

Bài toán thành 0/1 knapsack với giá trị $I$, trọng lượng $c$, sức chứa $C$, nhưng neuron hạng $j$ chỉ có latency $c_l^j$ nếu mọi neuron hạng $1..j-1$ trong lớp được giữ. Do đó thêm ràng buộc "tiền nhiệm" (preceding): chỉ cho giữ neuron $n$ nếu neuron đứng ngay trước nó trong cùng lớp cũng được giữ.

```
# Augmented Knapsack Solver (Algorithm 1, rút gọn)
Input: I[1..N] (importance, sắp giảm dần trong mỗi lớp),
       c[1..N] (latency integer), C (ngân sách), r[n] = chỉ số neuron tiền nhiệm
dp = zeros(C+1);  keep = zeros(N, C+1)
for n in 1..N:
    new = copy(dp)
    for cap in c[n]..C:
        v_keep  = I[n] + dp[cap - c[n]]
        v_prune = dp[cap]
        if v_keep > v_prune and keep[r[n], cap - c[n]]:   # tiền nhiệm phải được giữ
            keep[n, cap] = True;  new[cap] = v_keep
        else:
            keep[n, cap] = False; new[cap] = v_prune
    dp = new
# truy vết ngược từ n = N về 1: nếu keep[n, C] thì chọn n, C -= c[n]
```

Latency phải đổi sang số nguyên (nhân 1000 rồi làm tròn) để làm chỉ số mảng (Appendix F).

### 3.5 Gom nhóm neuron theo bậc latency

Độ rộng bậc thang (khoảng cách số kênh giữa hai "vách") gọi là **latency step size**; mỗi lớp gom $s$ kênh liền nhau (theo thứ tự quan trọng) thành một nhóm, $s$ = step size. Lợi ích kép: (i) chỉ cắt theo bậc nên cắt nào cũng có lợi tốc độ; (ii) giảm số "vật phẩm" $N$ cho knapsack, ví dụ ResNet50 từ 26,560 neuron xuống 215 nhóm. Với skip connection (ResNet) và group conv (MobileNet) thì các lớp ràng buộc nhau cùng chỉ số kênh được gom chung, dùng group size lớn nhất trong số đó.

### 3.6 Quy trình HALP đầy đủ

```
Input: mạng pretrained, ngân sách latency C, LUT
Lịch: k = 30 mốc, C^1 > C^2 > ... > C^k = C   (giảm theo hàm mũ)
Mỗi r = 320 minibatch:
  1. train bình thường 1 minibatch, tích luỹ importance (Eq.3)
  2. trung bình importance, xếp hạng trong từng lớp,
     cập nhật latency contribution c theo số kênh đang còn (Eq.6)
  3. gom nhóm theo step size, tính I và c của nhóm, tìm nhóm tiền nhiệm
  4. chạy augmented knapsack với C^t -> chọn nhóm giữ
Sau khi đạt C: fine-tune (90 epoch, batch 128/GPU, warmup 8 epoch rồi cosine)
```

Cài đặt: 8 GPU V100 để prune/fine-tune; LUT đo trên NVIDIA Titan V (batch 256) hoặc Jetson TX2 (batch 32), mỗi phép đo trung bình 100 lần; cuDNN 7.6.5.

## 4. Thí nghiệm và kết quả chính

Mặc định: Titan V, đo latency ở **batch 256**; baseline các phương pháp khác được dựng lại từ cấu trúc đã công bố rồi đo trên cùng GPU. "HALP-X%" nghĩa là giữ X% latency.

**ResNet50 / ImageNet (Bảng 1, trích; FPS trên Titan V):**

| Phương pháp | FLOPs (G) | Top-1 (%) | FPS | Speedup |
|---|---|---|---|---|
| Không prune | 4.1 | 76.2 | 1019 | 1x |
| AutoSlim | 3.0 | 76.0 | 1215 | 1.14x |
| GReg-1 | 2.7 | 76.3 | 1171 | 1.15x |
| HALP-80% | 3.1 | 77.2 | 1256 | 1.23x |
| AutoSlim | 2.0 | 75.6 | 1592 | 1.56x |
| MetaPruning | 2.0 | 75.4 | 1604 | 1.58x |
| HALP-55% | 2.0 | 76.5 | 1630 | 1.60x |
| AutoSlim | 1.0 | 74.0 | 2390 | 2.45x |
| MetaPruning | 1.0 | 73.4 | 2381 | 2.34x |
| HALP-30% | 1.0 | 74.3 | 2755 | 2.70x |

Với baseline của EagleEye (dense 77.2): EagleEye-2G 76.4 / 1.44x vs HALP-55% 76.6 / 1.64x; EagleEye-1G 74.2 / 2.38x vs HALP-30% 74.5 / 2.55x.

**ResNet101:** không prune 77.4% / 620 FPS; HALP-60% 78.3% (1.37x); HALP-50% 77.8% (1.60x); HALP-40% 77.2% (1.90x); HALP-30% 76.5% (2.45x). So sánh Taylor-75% 77.4% (1.21x) và Taylor-55% 76.0% (1.47x).

**MobileNetV1:** không prune 72.6% / 3415 FPS; HALP-42% 68.3% (2.32x) so với AutoSlim 67.9% (2.27x); HALP-60% 71.3% (1.68x) so với EagleEye 70.9% (1.47x).

**VGG16:** không prune 71.59% / 766 FPS; HALP-30% 72.33% (2.42x); HALP-20% 70.76% (5.49x).

**Tóm tắt abstract:** ResNet-50/-101 throughput x1.60 / x1.90 với thay đổi top-1 +0.3% / -0.2%; SSD trên VOC x1.94 với giảm 0.56 mAP.

**Kết hợp distillation:** dùng teacher RegNetY-16GF (top-1 82.9%), mô hình ResNet50 đã prune nhanh 2.70x với giảm 1% accuracy (Hình 2).

**TensorRT (Bảng 2, tăng tốc so với dense cùng định dạng, Titan V / RTX3080):** HALP-55%: FP32 1.24x, FP16 4.70x (Titan V); trên RTX3080 FP32 1.18x, FP16 3.32x, INT8 6.40x (acc drop 1.02%). HALP-30%: FP32 2.91x, FP16 9.61x; RTX3080 FP32 2.56x, FP16 6.44x, INT8 14.12x (acc drop 2.55%). (Bảng trong text bị lệch dòng; các số trên đọc theo cặp EagleEye/HALP nhưng cột accuracy drop có thể lẫn, nên đáng kiểm lại bản gốc nếu cần dùng.)

**Chi phí thiết kế (Bảng 3):** chọn sub-network của EagleEye ~30 giờ GPU, NetAdapt 864 giờ GPU, ThiNet ~8000 giờ GPU; HALP ~30 phút **CPU** (30 bước prune, mỗi bước ~1 phút giải knapsack) và dưới 1 phút cho MobileNetV1. Không cần train mạng phụ trợ, không cần đánh giá nhiều ứng viên. Chi phí dựng LUT chỉ làm một lần cho mọi tỉ lệ nén.

**Detection (SSD512-ResNet50, VOC07+12):** base 77.98 mAP, 65.56 GFLOPs, 68.24 FPS; HALP: 77.42 mAP, 15.38 GFLOPs, 132.57 FPS (khoảng 77% FLOPs giảm, 1.94x). Chỉ prune backbone.

**Batch size 1 (Appendix E, Bảng 5, Titan V):** không prune 76.2% / 181 FPS. Các phương pháp khác chỉ nhanh 1.00-1.14x (GReg-2 ở 0.50x: 73.9%, 1.14x). HALP-90%: 76.4%, 220 FPS (1.22x); HALP-80%: 75.3%, 247 FPS (1.37x).

## 5. Phân tích / ablation

- **Kích thước nhóm (Hình 4, ResNet50):** nhóm quá nhỏ (vd. 2) kém nhất độc lập với ngân sách vì giữ/bỏ những nhóm không đem lại lợi latency mà vẫn gây hại; nhóm quá lớn (256) cũng kém vì importance bị trung bình hoá, kém phân biệt và dễ xoá hẳn một lớp trong một bước. Latency-aware grouping (LG) tốt nhất. Với ResNet50, 23 lớp dùng nhóm 32, 20 lớp nhóm 64, 10 lớp nhóm 128; lớp gần đầu vào nhóm nhỏ hơn.
- **FLOPs không tương đương latency:** HALP đôi khi để lại nhiều FLOPs hơn đối thủ nhưng chạy nhanh hơn (vd. batch 1: HALP-80% còn 2.3 GFLOPs nhưng nhanh hơn nhiều so với các mạng 1.0-1.3 GFLOPs).
- **Cấu trúc sau prune** (Hình 8 và phụ lục) được phân tích phụ lục, không có số liệu trong phần đọc chính.

## 6. Giới hạn

**Tác giả nêu/ngầm nêu:**
- Latency table phải đo lại cho mỗi GPU/batch size/độ chính xác số học; INT8 TensorRT LUT là "future work".
- Ở batch 1, bậc thang gần như biến mất, nên phải dùng nhóm nhỏ (kích thước 2) và lợi ích giảm (nhưng vẫn hơn các baseline).
- Code "sẽ release khi được chấp nhận" (phiên bản này).

**Chúng ta nhận xét:**
- Chỉ CNN và detector; chưa có transformer, attention, MLP gated, linear attention hay mô hình sinh. Importance dựa vào tham số BatchNorm (không có trong LLM/transformer dùng LayerNorm/RMSNorm).
- Latency được cộng độc lập theo lớp (tổng của $T_l$), bỏ qua tương tác giữa lớp (fusion, bộ nhớ, kernel launch); với mô hình nhỏ chạy batch 1, overhead launch có thể áp đảo.
- Importance bậc nhất (gradient) ước lượng trên minibatch, có thể nhiễu và nhạy với dữ liệu.
- Không có so sánh với quá trình train-từ-đầu cấu trúc đã prune (xem 11).

## 7. Liên hệ với các bài khác

- **07 (Molchanov 2019):** HALP kế thừa tiêu chí Taylor và lịch prune dần; thêm latency và knapsack.
- **04, 05 (Pruning filters, Slimming):** những phương pháp prune cục bộ theo tỉ lệ cố định mỗi lớp; HALP phân bổ toàn cục.
- **18 (2:4):** cùng quan tâm tốc độ phần cứng thật; 2:4 dùng mẫu thưa vi cấu trúc, HALP đổi kích thước lớp.
- **20 (DepGraph):** giải bài toán "cắt cái này thì phải cắt cái kia" cho kiến trúc tuỳ ý; HALP xử lý skip connection bằng cách gom nhóm cùng chỉ số kênh (cách đơn giản).
- **25 (ZipLM), 42 (Puzzle), 41 (Flextron):** các bài sau cho LLM cũng chọn cấu hình theo ràng buộc latency/phần cứng.
- **43 (Pruning vs scratch), 11 (Rethinking):** câu hỏi liệu cấu trúc tìm được có giá trị hơn trọng số.

## 8. Ứng dụng cho dự án (suy luận)

- **Tư duy rất đáng mượn, thuật toán thì chỉ mượn một phần.** Mục tiêu dự án (GPU ≤ 8 GB, batch-1 latency) đúng là loại ràng buộc mà HALP nhấn mạnh: đừng dùng số tham số làm đích, hãy đo latency trên đúng GPU.
- **Cảnh báo quan trọng từ chính paper (Appendix E):** ở batch 1 các phương pháp giảm FLOPs gần như không làm nhanh hơn (1.00-1.14x). Với OCR VLM, decode từng token thường bị giới hạn bởi băng thông bộ nhớ và overhead kernel; cắt MLP neuron nhỏ có thể không giảm latency như kỳ vọng. Cắt **số tầng** (layer) giảm thẳng số lần kernel launch và đọc trọng số, có lẽ giải thích một phần quan sát "cắt layer hồi phục tốt hơn cắt neuron" (giả thuyết của chúng ta: lợi ích latency chắc chắn hơn, chưa chứng minh).
- **Dùng LUT + knapsack để chọn cấu hình:** xây bảng latency (batch 1, GPU đích) cho từng loại khối (ViT block, GatedDeltaNet block, full-attention block, MLP với width khác nhau), rồi giải knapsack để phân bổ ngân sách 0.3B/0.1B giữa ViT và decoder và giữa các lớp, với importance lấy từ một proxy phù hợp (ví dụ độ tăng loss/KL đo bằng calibration). Phù hợp với bài 42 (Puzzle) về tư tưởng.
- **Nhóm theo bậc phần cứng:** khi cắt MLP width, làm tròn về bội số của 64/128 (bậc thang GEMM và Tensor Core), thay vì kênh lẻ.
- **Distillation:** HALP + KD (teacher RegNetY-16GF) đạt nhanh hơn 2.70x với mất 1%, hỗ trợ ý tưởng dùng teacher 4B để hồi phục.
- Không áp dụng trực tiếp: importance dựa trên BN, quy trình 90 epoch fine-tune ImageNet.

## 9. Thí nghiệm gợi ý

1. Đo latency batch-1 trên GPU đích của một khối MLP và của một block đầy đủ khi width giảm dần (ví dụ 100% → 10%, bước 64): có bậc thang không? Giảm width 50% có giảm latency khoảng 50% không? Trả lời câu hỏi "tại sao cắt neuron kém".
2. Lập bảng latency/tham số của từng loại block trong LightOnOCR (ViT, linear-attention, full-attention); tính "importance per ms" của mỗi block bằng ablation đơn giản và xếp hạng thử bằng một knapsack nhỏ (DP) để chọn block cần giữ dưới ngân sách.
3. So sánh hai cấu hình cùng 0.3B: (a) đồng đều giảm width mọi lớp, (b) cấu hình do knapsack chọn; sau cùng một lượng distill, so CER và latency batch-1.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Bài toán | $\max \sum_l I_l(p_l)$ s.t. $\sum_l T_l(p_{l-1},p_l)\le C$ |
| Importance | $I_l^n=\lvert g_\gamma\gamma+g_\beta\beta\rvert$ (Taylor bậc 1 trên BN), cộng dồn theo thứ hạng |
| Latency neuron | $c_l^j=T_l(p_{l-1},j)-T_l(p_{l-1},j-1)$, cập nhật khi mạng thay đổi |
| Knapsack tăng cường | 0/1 knapsack + ràng buộc: giữ neuron $n$ chỉ khi tiền nhiệm trong lớp được giữ |
| Latency step size | Độ rộng bậc thang latency; kích thước nhóm $s$ bằng nó |
| Lợi ích gom nhóm | ResNet50: 26,560 neuron → 215 nhóm; giải knapsack ~1 phút/bước |
| Lịch | 30 bước, mỗi 320 minibatch, ngân sách giảm hàm mũ; sau đó fine-tune 90 epoch |
| Chi phí | ~30 phút CPU (so với 30 giờ GPU của EagleEye) |
| Kết quả nổi bật | ResNet50 HALP-30%: 74.3% top-1, 2.70x; ResNet101 HALP-50%: 77.8%, 1.60x; batch 1 HALP-80%: 75.3%, 1.37x |

## 11. Kết luận ngắn

HALP biến structured pruning thành bài toán chọn cấu hình dưới ngân sách **latency thực** bằng bảng tra và knapsack, kèm nhóm neuron khớp bậc thang GPU. Đóng góp đáng nhớ nhất cho dự án không phải thuật toán CNN cụ thể, mà là kỷ luật: đo latency trên phần cứng và batch size đích, vì tăng tốc từ giảm FLOPs có thể biến mất ở batch nhỏ.

Link gốc: https://arxiv.org/abs/2110.10811
