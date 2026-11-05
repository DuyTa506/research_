# Linear Mode Connectivity and the Lottery Ticket Hypothesis (2019)

**Tác giả:** Jonathan Frankle, Gintare Karolina Dziugaite, Daniel M. Roy, Michael Carbin.  
**Venue / arXiv:** arXiv:1912.05671 (v1 11/12/2019; bản dùng ở đây v4, 18/07/2020); Proceedings of ICML 2020 (PMLR 119).  
**Nguồn:** https://arxiv.org/abs/1912.05671  
**Chủ đề:** Instability analysis (phân tích độ bất ổn trước nhiễu SGD) qua nội suy tuyến tính; vì sao LTH hoạt động ở MNIST nhưng không ở ImageNet; *rewinding* (quay về bước $k$ sớm thay vì bước 0) để mở rộng LTH ra ResNet-50, Inception-v3.  
**Vị trí trong lộ trình:** Đọc ngay sau 08 (Lottery Ticket). Bài 08 để lại một vết nứt: LTH không tìm được vé ở mạng lớn/lr cao (cần warmup) và Liu et al. (bài 11) phản biện. Bài này giải thích vết nứt đó bằng động lực học tối ưu hóa và đưa ra bản sửa "rewind". Bài kế: 10 (SNIP), 11 (Rethinking).

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Số liệu lấy từ phần chính của paper. Phần "Ứng dụng cho dự án" là suy luận của chúng ta.

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

Bài 08 cho thấy iterative magnitude pruning (IMP) tìm được "vé" ở mạng nhỏ (LeNet/MNIST, các CNN nhỏ CIFAR-10) khi reset về khởi tạo. Nhưng các nghiên cứu sau (Liu et al. 2019, Gale et al. 2019) và chính paper này xác nhận: với VGG/ResNet chuẩn trên CIFAR-10 và ImageNet, IMP **không** cho mạng con khớp độ chính xác ở độ thưa không tầm thường, tức không hơn cắt ngẫu nhiên. Chưa ai giải thích được vì sao.

Khái niệm nền:
- **SGD noise**: khác biệt giữa các lần train do thứ tự mini-batch và data augmentation ngẫu nhiên, dù khởi tạo và siêu tham số giống nhau.
- **Mode (cực tiểu)**: một vùng loss thấp mà train hội tụ về. Hai mode là *kết nối tuyến tính* nếu trên đoạn thẳng nối hai bộ trọng số, lỗi không có "rào cản" tăng lên.
- **Mode connectivity** (Draxler 2018, Garipov 2018): các mạng train từ các khởi tạo *khác nhau* nối được bằng đường *phi tuyến* có lỗi hằng, nhưng không nối được bằng đường thẳng.
- **Matching subnetwork**: mạng con train đến độ chính xác của mạng đầy đủ (trong phạm vi sai số); từ khi quay về bước $k>0$ gọi là *matching* chứ không gọi winning ticket (vì không còn là khởi tạo ngẫu nhiên).

## 2. Ý tưởng chính

Hỏi: nếu train cùng một mạng ($W_0$ hoặc trạng thái sau $k$ bước) hai lần với nhiễu SGD khác nhau, hai kết quả có nằm trong cùng một "thung lũng" nối bằng đường thẳng không? Nếu có thì mạng **ổn định** với nhiễu SGD. Phát hiện: mạng đầy đủ chỉ ổn định *sau một đoạn đầu huấn luyện* (3% với ResNet-20, 20% với ResNet-50), và mạng con từ IMP **chỉ khớp độ chính xác khi nó ổn định**. Do đó với mạng lớn, thay vì reset về khởi tạo ($k=0$), reset về trạng thái sớm $k$ (rewinding) làm IMP tìm được mạng con vừa ổn định vừa khớp, mở rộng LTH lên ResNet-50 và Inception-v3 trên ImageNet.

## 3. Phương pháp chi tiết

### 3.1 Instability analysis

Mô hình SGD là hàm $A_{s\to t}:\mathbb R^D\times U\to\mathbb R^D$ nhận trọng số bước $s$ và nhiễu $u\sim U$, trả trọng số bước $t$.

```text
Algorithm 1: instability của W_k
  W_T^1 = A_{k->T}(W_k, u1)      # nhiễu SGD 1
  W_T^2 = A_{k->T}(W_k, u2)      # nhiễu SGD 2
  return f(W_T^1, W_T^2)
```

Hàm so sánh $f$ là **chiều cao rào cản lỗi** trên đoạn thẳng. Gọi $E(W)$ là lỗi (train hoặc test), $E_\alpha=E(\alpha W_1+(1-\alpha)W_2)$, $E_{\sup}=\sup_\alpha E_\alpha$, $\bar E=\mathrm{mean}(E(W_1),E(W_2))$:

$$\boxed{\text{instability}(W_k)=E_{\sup}(W_T^1,W_T^2)-\bar E(W_T^1,W_T^2)}$$

Quy ước: instability $<2\%$ là *stable* (hai mạng linearly mode connected); lấy 30 giá trị $\alpha$ cách đều; trung bình 9 mẫu (3 khởi tạo x 3 lần chạy mỗi khởi tạo). Ngưỡng 2% khớp với mức tăng lỗi của các đường nối trong Draxler/Garipov.

Phụ: khi làm từ bước $k$, hai bản sao chỉ train $T-k$ bước; để loại trừ khả năng "ổn định chỉ do train ngắn", tác giả huấn luyện lại $T$ bước với lịch lr reset; kết quả không khác (Figure 4).

### 3.2 IMP có rewinding (Algorithm 2)

```text
1. W0 ngẫu nhiên; m = 1
2. Train W0 -> W_k  (k bước, nhiễu u)
3. for n = 1..N:
       train m ⊙ W_k -> W_T  (T-k bước, nhiễu u')
       cắt các trọng số còn sống có |W_T| nhỏ nhất (toàn cục); đặt m[i]=0
4. return (W_k, m)         # mạng con = m ⊙ W_k
```

Iterative: cắt 20% mỗi vòng cho MNIST/CIFAR-10; one-shot đến 30% còn lại cho ImageNet (Table 1). So với bài 08, điểm mới là quay về $W_k$ chứ không phải $W_0$ ($k=0$ là LTH gốc).

### 3.3 Thiết lập (Table 1, trích)

| Mạng | Dữ liệu | Tham số | Độ chính xác | lr | Warmup | Mật độ sau cắt | Kiểu cắt |
|---|---|---|---|---|---|---|---|
| LeNet | MNIST | 266K | 98.3 ± 0.1% | Adam 1.2e-3 | 0 | 3.5% | Iterative |
| ResNet-20 (Standard / Low / Warmup) | CIFAR-10 | 274K | 91.7 / 88.8 / 89.7% | 0.1 / 0.01 / 0.03 | 0 / 0 / 30K | 16.8 / 8.6 / 8.6% | Iterative |
| VGG-16 (Standard / Low / Warmup) | CIFAR-10 | 14.7M | 93.7 / 91.7 / 93.4% | 0.1 / 0.01 / 0.1 | 0 / 0 / 30K | 1.5 / 5.5 / 1.5% | Iterative |
| ResNet-50 | ImageNet | 25.5M | 76.1 ± 0.1% | 0.4 | 5 epoch | 30% | One-shot |
| Inception-v3 | ImageNet | 27.1M | 78.1 ± 0.1% | 0.03 (giảm tuyến tính đến 0.005) | 0 | 30% | One-shot |

("Low" và "Warmup" là các biến thể của bài 08: lr thấp hoặc warmup để IMP tìm được vé.) Độ thưa trong Table 1 là mức cực đoan nhất mà IMP vẫn khớp ở một bước quay lại nào đó.

## 4. Thí nghiệm và kết quả chính

### 4.1 Mạng đầy đủ: bất ổn lúc khởi tạo, ổn định sớm (Mục 3)

- Tại $k=0$ chỉ LeNet ổn định; các mạng khác khi nội suy tuyến tính có lỗi tăng lên mức đoán ngẫu nhiên.
- Càng bắt đầu từ $k$ lớn thì instability càng thấp. **ResNet-20 ổn định ở iteration 2000 (3% huấn luyện), VGG-16 ở 1000 (1.5%)**; **ResNet-50 ở epoch 18 (20%), Inception-v3 ở epoch 28 (16%)** (đo trên test). Với ResNet-50, độ ổn định trên train xuất hiện muộn hơn và Inception-v3 không đạt trong khoảng khảo sát.
- Từ điểm đó, kết quả tối ưu hóa được quyết định tới một vùng cực tiểu nối tuyến tính, bất kể nhiễu SGD.

### 4.2 Mạng con IMP tại $k=0$ (Table 2, độ chính xác)

| Mạng | Đầy đủ | IMP | Random Prune | Random Reinit | $\Delta$ IMP | Matching? |
|---|---|---|---|---|---|---|
| LeNet | 98.3 | 98.2 | 96.7 | 97.5 | 0.1 | Có |
| ResNet-20 | 91.7 | 88.5 | 88.6 | 88.8 | 3.2 | Không |
| ResNet-20 Low | 88.8 | 89.0 | 85.7 | 84.7 | -0.2 | Có |
| ResNet-20 Warmup | 89.7 | 89.6 | 85.7 | 85.6 | 0.1 | Có |
| VGG-16 | 93.7 | 90.9 | 89.4 | 91.0 | 2.8 | Không |
| VGG-16 Low | 91.7 | 91.6 | 90.1 | 90.2 | 0.1 | Có |
| VGG-16 Warmup | 93.4 | 93.2 | 90.1 | 90.7 | 0.2 | Có |
| ResNet-50 | 76.1 | 73.7 | 73.1 | 73.4 | 2.4 | Không |
| Inception-v3 | 78.1 | 75.7 | 75.2 | 75.5 | 2.4 | Không |

(Sai số chuẩn < 0.2.) Chỉ các trường hợp "matching" mới là các trường hợp mạng con ổn định ở $k=0$ (Figure 5): LeNet, ResNet-20 low/warmup, VGG-16 low/warmup. Còn lại (ResNet-20/VGG-16 chuẩn, ResNet-50, Inception-v3) vừa không ổn định vừa không matching; mạng con IMP không hơn mạng con cắt ngẫu nhiên hay reinit.

### 4.3 Rewinding (Figures 6–7)

- Mạng con IMP của ResNet-20 và VGG-16 chuẩn **trở nên ổn định ở iteration 500 (0.8% huấn luyện) và 1000 (1.6%)**; của ResNet-50 và Inception-v3 ở **epoch 5 (5.5%) và 6 (3.5%)**. Trong mọi trường hợp, mạng con ổn định *sớm hơn* mạng đầy đủ (ResNet-50: epoch 5 so với 18; Inception-v3: epoch 6 so với 28).
- Các mạng con vốn không matching ở $k=0$ trở thành matching khi quay về $k$ sau đó, và điểm quay lại **trùng khớp** với điểm mạng con trở nên ổn định. Kết luận (tại độ thưa cực đoan): *mạng con IMP khớp độ chính xác khi và chỉ khi nó ổn định*.
- Train error cũng giảm khi rewind (ResNet-20, VGG-16 đạt 0% train error), gợi ý mạng con ổn định tối ưu tốt hơn.
- Mạng con cắt ngẫu nhiên hay reinit **không ổn định và không matching ở mọi bước quay lại** (LeNet là ngoại lệ về độ ổn định).

### 4.4 Các mức thưa khác (Figure 9, ResNet-20 và VGG-16)

- **Dải I** (còn >80.0% trọng số với ResNet-20, >16.8% với VGG-16): đủ quá tham số hóa nên ngay cả cắt ngẫu nhiên cũng khớp ("thưa tầm thường").
- **Dải II** (80.0%–13.4% ResNet-20; 16.8%–1.2% VGG-16): chỉ mạng con IMP khớp. Trong một phần dải (51.2%–13.4% và 6.9%–1.5%) bước mà mạng con thành stable và thành matching xấp xỉ trùng nhau.
- **Dải III** (<13.4% và <1.2%): cực thưa đến mức ngay cả IMP không khớp ở bước quay lại nào; lỗi vẫn giảm khi mạng con trở nên ổn định.
- Khi độ thưa tăng, bước mà IMP trở nên stable giảm, đi ngang rồi tăng; bước của cắt ngẫu nhiên chỉ tăng đến khi không bao giờ stable.

## 5. Phân tích / ablation

- **Loại trừ giải thích "chỉ do train ngắn":** instability không đổi khi train lại $T$ bước với lịch lr reset (Figure 4).
- **Hai giai đoạn huấn luyện:** pha bất ổn (SGD noise đẩy vào các cực tiểu không nối tuyến tính) rồi pha ổn định; phù hợp với việc Hessian sớm tách thành vài trị riêng lớn và "bulk" (Gur-Ari 2018) và với lợi ích của warmup lr (Goyal 2017).
- **Hyperparameter "low"/"warmup" của bài 08** vừa làm IMP matching vừa làm mạng con ổn định ở $k=0$: bằng chứng thêm cho mối liên hệ giữa ổn định và độ chính xác.
- **Rewinding dùng trong thực tế:** theo Discussion, Morcos et al. 2019 cho thấy vé có rewinding chuyển được giữa các tác vụ ảnh; Renda et al. 2020 cho thấy IMP + rewinding đạt sparsity SOTA, bằng hoặc hơn fine-tune với lr thấp.
- Phụ lục G (không đọc kỹ ở đây) so sánh các cách đo khác ngoài nội suy tuyến tính.

## 6. Giới hạn

**Do tác giả nêu:**
- Kết quả là *thực nghiệm*, không có chứng minh lý thuyết; mối quan hệ nhân quả giữa stable và matching chỉ là tương quan ("matching khi stable").
- Matching ở độ thưa cực đoan chỉ trong điều kiện bước $k>0$; không còn là khởi tạo ngẫu nhiên, nên không còn "winning ticket" theo nghĩa gốc, và không tiết kiệm được chi phí train nếu cần mạng đầy đủ đến bước $k$.
- Dải III: ngay cả khi stable, IMP không matching, tức stable không phải điều kiện đủ.
- Chi phí tính toán cao: mức thưa chỉ được khảo sát toàn diện với ResNet-20 và VGG-16.

**Do chúng tôi nhận xét:**
- Chỉ phân loại ảnh; không có mô hình ngôn ngữ, transformer, hay mô hình tạo sinh.
- Cắt là thưa không cấu trúc; kết luận về stability không đảm bảo chuyển sang cắt có cấu trúc.
- Đo stability dựa trên *lỗi* phân loại; với bài toán sinh chuỗi (OCR/VLM) chưa có định nghĩa tương đương (nội suy trọng số tuyến tính vẫn áp dụng được nhưng metric cần chọn).
- Ngưỡng 2% và việc dùng 9 mẫu là quy ước thực nghiệm.

## 7. Liên hệ với các bài khác

- **08 Lottery Ticket:** bài này giải thích vì sao LTH gốc thất bại ở mạng sâu/lr cao (mạng chưa ổn định ở $k=0$) và sửa bằng rewind.
- **11 Rethinking:** Liu et al. cho rằng reinit và train từ đầu cho kết quả ngang cắt; bài 09 giải thích mâu thuẫn bề ngoài: ở độ thưa tầm thường (dải I), đúng như vậy; khác biệt chỉ xuất hiện ở dải II.
- **10 SNIP:** cắt ngay tại khởi tạo: nếu các mạng thưa chưa ổn định lúc $k=0$ thì chọn mặt nạ tại $k=0$ có thể thiệt thòi; SNIP đối lập trực tiếp với kết luận "cần train một chút trước khi chọn mặt nạ".
- **15 LTH for BERT:** dùng rewinding cho BERT khi vé ở khởi tạo không hoạt động.
- **16 Movement pruning, 03 Zhu & Gupta:** cắt trong lúc train, khác hướng.
- **43 Pruning vs scratch:** cùng chủ đề "cắt + phục hồi so với train lại".

## 8. Ứng dụng cho dự án (suy luận)

**Liên quan gián tiếp, mang tính khái niệm.** Dự án cắt có cấu trúc một mô hình đã huấn luyện xong rồi phục hồi bằng chưng cất; không chọn mặt nạ lúc khởi tạo.

- **Rewinding không áp dụng thẳng:** ta không có checkpoint "bước $k$ sớm" của LightOnOCR (nếu không phải bản của tác giả). Nhưng ý tưởng "phục hồi từ điểm quay về" có thể thay bằng: sau khi cắt, khởi tạo tham số bị đụng tới từ checkpoint gốc (đó là "rewind về trạng thái cuối").
- **Công cụ đo mới: nội suy tuyến tính giữa hai lần phục hồi.** Chạy hai lần chưng cất với thứ tự dữ liệu khác nhau từ cùng một mô hình đã cắt, nội suy tuyến tính trọng số và đo loss: nếu có rào cản, quá trình phục hồi nhạy với nhiễu (instability cao); nếu không, có thể trung bình trọng số (model soup), và quan trọng hơn, ta biết kết quả ổn định. Có thể đối chiếu: cắt neuron MLP có instability cao hơn cắt layer không? Đó có thể là một cách (giả thuyết) giải thích phục hồi kém của cắt neuron.
- **Phù hợp D_lin / D_res (giả thuyết):** nếu cắt layer chỉ để lại phần dư gần tuyến tính thì hai lần phục hồi có thể cùng một thung lũng; cắt neuron để lại phần dư phi tuyến có thể dẫn tới nhiều thung lũng. Cần thí nghiệm dưới.
- **Warmup/lr khi phục hồi:** bài này tiếp tục củng cố nguyên tắc dùng lr nhỏ/warmup ở giai đoạn đầu sau khi đổi kiến trúc (pha bất ổn xảy ra đầu huấn luyện).
- **Ít liên quan:** thuật toán IMP và việc chọn mặt nạ thưa không cấu trúc.

## 9. Thí nghiệm gợi ý

1. **Instability của phục hồi sau cắt:** với mô hình đã cắt (cắt layer vs cắt neuron MLP cùng số tham số), chạy 2 phục hồi bằng chưng cất với 2 seed dữ liệu, nội suy tuyến tính trọng số ở 11 điểm $\alpha$, vẽ loss KL/CER theo $\alpha$; so rào cản hai kiểu cắt.
2. **Điểm bắt đầu phục hồi:** thay vì phục hồi ngay từ trạng thái sau cắt, thử "warm start" bằng vài trăm bước lr thấp rồi mới lr đầy đủ, đo instability theo số bước warmup (mô phỏng ý "stable sau $k$ bước").
3. **Model soup sau cắt:** trung bình trọng số của 2–3 lần phục hồi ổn định; so với mô hình đơn (chỉ làm khi thí nghiệm 1 cho rào cản ≈ 0).

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| Instability analysis | Train hai bản sao với nhiễu SGD khác nhau và so sánh kết quả |
| $E_\alpha=E(\alpha W_1+(1-\alpha)W_2)$ | Lỗi tại điểm nội suy tuyến tính |
| instability $=E_{\sup}-\bar E$ | Chiều cao rào cản lỗi |
| Stable | instability $<2\%$ (linearly mode connected) |
| Linear mode connectivity | Hai cực tiểu nối bằng đoạn thẳng không có rào cản |
| IMP với rewinding | Train đến $W_T$, cắt trọng số nhỏ nhất, quay về $W_k$ (không phải $W_0$) |
| Matching | Mạng con đạt độ chính xác của mạng đầy đủ (sai số < 0.2 điểm trong Figure 9) |
| Sparsity range I/II/III | Cắt ngẫu nhiên khớp / chỉ IMP khớp / không ai khớp |
| Kết luận chính | IMP subnetwork matching $\Leftrightarrow$ stable (ở độ thưa cực đoan khảo sát) |

## 11. Kết luận ngắn

Bài này biến câu hỏi "LTH đúng không" thành câu hỏi có thể đo: mạng con có ổn định với nhiễu SGD không. Mạng con IMP chỉ khớp độ chính xác khi chúng ổn định; ở mạng lớn điều đó xảy ra sau 0.8–5.5% đầu huấn luyện, nên phải *rewind* về bước sớm thay vì khởi tạo. Với dự án, công cụ đáng giữ lại là phép nội suy tuyến tính để kiểm tra độ nhạy của quá trình phục hồi sau cắt.

Link gốc: https://arxiv.org/abs/1912.05671
