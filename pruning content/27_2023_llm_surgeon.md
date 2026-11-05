# The LLM Surgeon (2023)

**Tác giả:** Tycho F.A. van der Ouderaa, Markus Nagel, Mart van Baalen, Yuki M. Asano, Tijmen Blankevoort (Imperial College London; Qualcomm AI Research; QUVA Lab, University of Amsterdam).  
**Venue / arXiv:** arXiv:2312.17244 (28/12/2023, bản v2 03/2024). Bản text không ghi tên hội nghị.  
**Nguồn:** https://arxiv.org/abs/2312.17244  
**Chủ đề:** Pruning bậc hai kiểu OBS/OBD cho LLM với xấp xỉ曲率 Kronecker-factored (KFAC) của Fisher có tính cả gradient; hỗ trợ structured (hàng/cột), semi-structured (N:M), unstructured; phân bổ thưa toàn cục theo ngưỡng; cập nhật trọng số có tương quan; cắt nhiều "shot".  
**Vị trí trong lộ trình:** Đọc sau 01 (OBD/OBS), 22 (OBC), 23 (SparseGPT), 25 (ZipLM) và 26 (LLM-Pruner). Bài này là nỗ lực "làm OBS đúng nghĩa" cho LLM: dùng loss thật (có gradient) thay vì chỉ lỗi tái tạo từng layer. Bài kế: 28 (FLAP), 29 (SliceGPT, được trích như concurrent work).

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Số liệu lấy từ paper. Phần liên hệ D_lin / D_res và dự án là suy luận của chúng ta.

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

OBD/OBS (1989, 1992; bài 01) viết pruning thành bài toán tối ưu có ràng buộc: xấp xỉ loss bằng khai triển Taylor bậc hai quanh trọng số đã huấn luyện, rồi chọn trọng số xoá và cập nhật các trọng số còn lại để loss tăng ít nhất. Nút thắt là **Hessian** có kích thước (số tham số)$^2$, không thể lưu cho mạng thật. Các hướng giải:

- **EigenDamage** (Wang 2019): phân rã Kronecker của Hessian chia khối đường chéo.
- **OBC, SparseGPT** (bài 22, 23): chỉ dùng độ cong của lỗi tái tạo bình phương đầu ra *của từng layer*, **bỏ qua gradient** liên hệ chi phí cục bộ với loss đích. Theo tác giả, xấp xỉ loss như vậy kém chính xác, và các phương pháp này không mở rộng dễ sang structured.

Một vài khái niệm cần có:
- **Fisher information matrix** (ma trận thông tin Fisher): kỳ vọng tích ngoài gradient của log-likelihood; với mô hình huấn luyện bằng negative log-likelihood, Fisher chính bằng Hessian (theo paper) và luôn nửa xác định dương.
- **KFAC** (Kronecker-factored approximate curvature): xấp xỉ khối Fisher của một layer thành tích Kronecker $G\otimes A$ của hai ma trận nhỏ: $A$ từ activation đầu vào, $G$ từ gradient đầu ra. Nghịch đảo dễ: $(G\otimes A)^{-1}=G^{-1}\otimes A^{-1}$.
- **Laplace approximation:** xấp xỉ hậu nghiệm bằng Gaussian quanh nghiệm, với hiệp phương sai là Hessian nghịch đảo; paper dùng góc nhìn này để giải thích tại sao OBS là "tối đa hoá likelihood dưới ràng buộc".

## 2. Ý tưởng chính

Hãy coi loss quanh mô hình đã huấn luyện là một cái bát hình parabol (xấp xỉ Gaussian) với độ cong ước lượng bằng KFAC. Xoá một hàng hoặc cột của ma trận trọng số là một ràng buộc; trong cái bát đó tính được ngay (1) **chi phí** (loss tăng bao nhiêu nếu xoá) và (2) **cập nhật tối ưu** cho mọi trọng số còn lại. LLM Surgeon áp dụng đúng công thức này lên LLM, bổ sung ba thứ: ngưỡng chi phí **toàn cục** để mỗi layer tự bị cắt nhiều hay ít tuỳ độ nhạy; cập nhật trọng số **tính cả tương quan giữa nhiều hàng/cột bị xoá cùng lúc**; cắt theo **nhiều shot** (cắt một ít, cập nhật độ cong, cắt tiếp). Có thể đổi thêm thời gian tính để lấy chất lượng cao hơn.

## 3. Phương pháp chi tiết

### 3.1 Bài toán có ràng buộc

Tham số $\theta\in\mathbb R^P$, loss $\mathcal L(\theta)=-\log p(\theta|D)$. Khai triển bậc hai quanh $\theta^*$ (giả sử gradient $\approx0$ ban đầu):

$$-\log q(\theta)\approx-\log p(D|\theta^*)-(\theta-\theta^*)^\top\nabla\mathcal L(\theta^*)-\tfrac12(\theta-\theta^*)^\top H(\theta-\theta^*).$$

Với $H=F$ (Fisher), xoá tập chỉ số $K$ là bài toán:

$$\min_{\Delta\theta}\tfrac12\Delta\theta^\top F\Delta\theta\quad\text{s.t.}\quad e_k^\top\Delta\theta+e_k^\top\theta=0,\ \forall k\in K.$$

Giải bằng nhân tử Lagrange, với $E_K=[e_1\dots e_K]$ là ma trận chọn các tham số bị xoá:

$$\boxed{\mathcal L=\tfrac12(E_K\theta)^\top\big(E_KF^{-1}E_K^\top\big)^{-1}(E_K\theta)}\qquad\boxed{\Delta\theta=-F^{-1}E_K^\top\big(E_KF^{-1}E_K^\top\big)^{-1}E_K\theta}$$

(Eq. 7, 8). Với Fisher xấp xỉ KFAC theo từng layer: $F_l\approx G_l\otimes A_l$, $G_l=\frac1{\sqrt N}\sum_n g_{l,n}g_{l,n}^\top$ (gradient theo đầu ra $y=Wa$, kích thước $R\times R$), $A_l=\frac1{\sqrt N}\sum_n a_{l,n}a_{l,n}^\top$ (activation đầu vào, $C\times C$).

### 3.2 Chi phí xoá một phần tử, một hàng, một cột

Giả sử chi phí của các phần tử/hàng/cột là độc lập khi xếp hạng. Thay $E$ thích hợp vào Eq. 7 (Eq. 10):

$$L_k=\frac12\frac{(\theta_k)^2}{[G^{-1}\otimes A^{-1}]_{kk}},\qquad L_r=\frac12\frac{\theta_r^\top A\,\theta_r}{[G^{-1}]_{rr}},\qquad L_c=\frac12\frac{\theta_c^\top G\,\theta_c}{[A^{-1}]_{cc}}$$

Ở đây $\theta_r\in\mathbb R^C$ là hàng $r$ của $W$ và $\theta_c\in\mathbb R^R$ là cột $c$. Đọc công thức: xoá hàng $r$ (một output) làm đầu ra đó mất đi, mức hại là năng lượng $\theta_r^\top A\theta_r$ của đầu ra, chia cho $[G^{-1}]_{rr}$, đại lượng nhỏ khi output $r$ **có thể được bù bởi các output khác** trong không gian gradient; tương tự cột $c$ (một input) có chi phí $\theta_c^\top G\theta_c$ chia cho $[A^{-1}]_{cc}$, nhỏ khi kênh đầu vào đó gần như là tổ hợp tuyến tính của kênh khác. $L_k$ trùng OBS; $L_r$, $L_c$ gần với "Structured Brain Surgeon" của EigenDamage nhưng viết cho hàng/cột.

### 3.3 Ngưỡng toàn cục và phân bổ động

Thay vì cắt đều mỗi layer (như SparseGPT), chọn **ngưỡng $\tau$ chung** sao cho phần trọng số bị xoá bằng $1-\alpha$ ($\alpha$ là kích thước đích, tỉ lệ còn lại). Với unstructured: sắp xếp mọi $L_k$ trên toàn mạng. Với N:M: lấy tổng $M$ chi phí thấp nhất trong mỗi khối. Với structured: sắp xếp $L_r,L_c$ có trọng số theo số phần tử mỗi hàng/cột (hàng dài tốn nhiều tham số). Các layer nhạy bị cắt ít, layer bền bị cắt nhiều.

### 3.4 Cập nhật trọng số có tương quan

Cộng các cập nhật độc lập cho từng phần tử là gần đúng. Paper tính **cập nhật chung** cho toàn bộ tập bị xoá:

- **Tổng quát:** cần nghịch đảo ma trận $K\times K$ ($K$ = số tham số bị xoá cùng lúc); với giới hạn $m$ tham số tương quan trong một nhóm để kiểm soát chi phí.
- **Structured:** nhờ cấu trúc Kronecker, chỉ phải nghịch đảo ma trận $R'\times R'$ (khi xoá $R'$ hàng) hoặc $C'\times C'$ (khi xoá $C'$ cột), thay vì $R'C\times R'C$. Hai trường hợp đặc biệt khi xoá *một* hàng $r$ hoặc *một* cột $c$ (Eq. 13), chỉ cần phép chia vô hướng:

$$\text{xoá hàng }r:\ \Delta\theta=-\frac{G^{-1}e_r\otimes\theta_r}{[G^{-1}]_{rr}},\qquad \text{xoá cột }c:\ \Delta\theta=-\frac{\theta_c\otimes A^{-1}e_c}{[A^{-1}]_{cc}}.$$

Ở dạng ma trận, xoá cột $c$: $\Delta W=-\theta_c\,(A^{-1}e_c)^\top/[A^{-1}]_{cc}$ (ta viết lại từ Eq. 13 cho dễ hình dung): các cột còn lại được cộng thêm một lượng tỉ lệ cột $c$, trọng số theo hàng $c$ của $A^{-1}$, giống bước bù của OBS/SparseGPT/ZipLM. Bản tổng quát cho nhiều hàng/cột có ở Eq. 12 (ta không chép lại vì bản text của công thức bị lệch thứ tự).

### 3.5 Nhiều shot

Mô hình bậc hai chỉ đúng cục bộ; bước nhảy lớn làm xấp xỉ sai. Do đó cắt trong $T$ shot theo lịch tuyến tính $\alpha_t$, giữa các shot tính lại độ cong (và cập nhật trọng số). Thực nghiệm: nhiều shot tốt hơn đơn điệu, sparsity cao cần nhiều shot hơn. Cấu hình: 40 shot cho mục tiêu 0.5 (structured), báo cáo các mức trung gian: $T=8$ cho 0.9, 16 cho 0.8, 24 cho 0.7, 32 cho 0.6; N:M và unstructured dùng $T=5$.

### 3.6 Hiệu chỉnh bậc nhất bằng LoRA xen kẽ

Mô hình có thể không ở cực tiểu, hoặc loss dùng khi nén khác loss huấn luyện, hoặc đã cắt nhiều shot, nên gradient $\ne0$. Giữa các shot, tuỳ chọn thêm cập nhật LoRA ($W+UV$) và hấp thụ vào $W$ (nên tổng các cập nhật rank thấp có rank cao hơn từng cái).

```text
LLM_Surgeon_structured(theta0, alpha_target, D, T):
    for t in 1..T:
        # alpha_t: lịch tuyến tính từ 1 xuống alpha_target
        (G, A) = KFAC factors per layer from D (forward + backward)   # activation, output-gradient
        L_r, L_c = row/column costs (Eq.10)
        tau      = global threshold s.t. fraction (1 - alpha_t) of weights is cut
        remove rows/cols with cost <= tau
        Delta    = correlated update for removed rows/cols (Eq.12)
        theta   += Delta
        (optional) low-rank (LoRA) first-order update, absorbed into theta
    return theta
```

## 4. Thí nghiệm và kết quả chính

**Mô hình:** OPT 125m, 1.3b, 2.7b, 6.7b và Llama-v2 7B (13B ở phụ lục). **Dữ liệu nén:** 128 chuỗi 2048 token từ train WikiText-2; đánh giá test PPL WikiText-2 (cũng thử C4 và tiếng Pháp/Đức/Ý ở phụ lục). **Baseline:** Magnitude; L-OBD (chỉ activation, đường chéo $I\otimes A$); K-OBD (cùng KFAC nhưng độc lập hoàn toàn, **không cập nhật** trọng số còn lại); SparseGPT (unstructured/N:M).

**Structured, test PPL WikiText-2 (Table 1; "size" = % tham số còn lại):**

| Phương pháp | Size | OPT-125m | OPT-1.3b | OPT-6.7b | Llama-v2 7B |
|---|---|---|---|---|---|
| Dense | 100% | 27.65 | 14.62 | 10.86 | 5.12 |
| Magnitude | 90% | 767.2 | 894.4 | 3464 | 36746 |
| L-OBD | 90% | 33.3 | 20.76 | 27.20 | 14259 |
| K-OBD (không update) | 90% | 27.97 | 14.68 | 10.53 | 5.48 |
| K-OBD | 80% | 29.89 | 15.63 | 11.28 | 9.14 |
| K-OBD | 70% | 36.54 | 18.29 | 13.03 | 15.43 |
| K-OBD | 50% | 75.95 | 37.68 | 25.54 | 46.64 |
| LLM Surgeon (tương quan trong hàng/cột) | 80% | 29.37 | 15.27 | 11.22 | 7.29 |
| LLM Surgeon (tương quan trong hàng/cột) | 50% | 51.48 | 23.81 | 15.38 | 25.62 |
| LLM Surgeon (tương quan đầy đủ) | 90% | 28.01 | 14.70 | 10.77 | 5.25 |
| LLM Surgeon (tương quan đầy đủ) | 80% | 28.73 | 15.12 | 11.02 | 6.18 |
| LLM Surgeon (tương quan đầy đủ) | 70% | 31.82 | 16.24 | 11.64 | 7.83 |
| LLM Surgeon (tương quan đầy đủ) | 60% | 38.47 | 18.45 | 12.58 | 10.39 |
| LLM Surgeon (tương quan đầy đủ) | 50% | 49.78 | 22.95 | 14.90 | 15.38 |

Paper kết luận: nhiều tương quan hơn thì tốt hơn, nhất là trên Llama-v2; cắt tới 20 đến 30% hàng/cột với suy giảm nhỏ; **cắt 50% không tốt hơn một mô hình nhỏ cùng cỡ** (tác giả viết vậy). Hình 1 cho thấy nội suy giữa các cỡ mô hình OPT pretrained.

**Llama-v2 13B structured (Table 7 PPL; Table 8 zero-shot trung bình 10 tác vụ):** dense PPL 4.547 và avg 74.07%; LLM Surgeon 90%: 4.692 / 74.28%; 80%: 5.286 / 72.12%; 70%: 6.207 / 67.51%; 60%: 7.245 / 62.48%; 50%: 9.428 / 57.59%. K-OBD cùng các mức: 4.908, 6.294, 10.08, 13.06, 16.06 (PPL) và avg 74.00, 71.11, 60.76, 54.37, 51.50. Tác giả: cắt 13B 20% mà giảm dưới 2% trên downstream.

**Semi-structured 2:4 (Table 3; PPL):** OPT-125m / 1.3b / 2.7b / 6.7b: LLM Surgeon 44.64 / 25.10 / 14.64 / 12.10 so với SparseGPT 45.51 / 29.44 / 14.92 / 13.01.

**Unstructured (Table 4):** Llama-v2 7B: LLM Surgeon (nhiều shot, tương quan đầy đủ) 5.13 (90%), 5.20 (80%), 5.36 (70%), 5.66 (60%), 6.08 (50%) so với SparseGPT 5.49, 5.58, 5.71, 5.94, 6.51. Cũng thắng SparseGPT trên OPT.

**Chi phí tính (phụ lục H, Table 11/12):** Llama-v2 7B structured: LLM Surgeon 17 giờ 08 phút trên 4 H100 80GB (K-OBD 16 giờ 58 phút); 13B: 1 ngày 9 giờ 26 phút trên 8 H100 80GB. Unstructured Llama-v2 7B: LLM Surgeon 2 ngày 8 giờ 16 phút (4 H100) so với SparseGPT dưới 5 phút trên 1 GPU 32GB.

## 5. Phân tích / ablation

- **Tương quan trong cập nhật (Table 1):** từ K-OBD (không cập nhật) đến cập nhật trong hàng/cột đến cập nhật đầy đủ, PPL giảm đều; đối với Llama-v2 7B ở 50%: 46.64, 25.62, 15.38. Nghĩa là **phần bù (cập nhật trọng số) đóng góp nhiều** khi cắt structured mạnh.
- **Gradient (G) quan trọng:** K-OBD (có $G$) tốt hơn L-OBD (chỉ $A$) nhiều (ví dụ 90% Llama-v2 7B: 5.48 so với 14259), và LLM Surgeon (structured) vượt SparseGPT-style "chỉ activation" ở các thiết lập khác. Paper nhấn mạnh việc concurrent work SliceGPT "bỏ qua thông tin gradient, nên kém hơn" (nhận định của tác giả; paper không so trực tiếp bằng số liệu).
- **Số shot (Table 9, OPT-1.3b structured):** 90%: 6/8/10 shot cho PPL 14.70 / 14.70 / 14.72; 80%: 12/16/20 shot cho 15.14 / 15.12 / 15.08; shot nhiều hơn tốt hơn một chút (bảng đầy đủ có thêm các mức thấp hơn).
- **LoRA xen kẽ (Table 2, OPT-125m structured):** có ích với mô hình nhỏ nhất (ví dụ 50%: 49.78 xuống 36.50) nhưng **không ích với mô hình lớn**; giả thuyết của tác giả: mô hình lớn dễ overfit vài batch WikiText-2.
- **Phân bổ sparsity học được (Fig. 4, OPT-125m):** cắt nhiều hơn ở layer đầu và ít hơn ở layer giữa; cắt nhiều FC hơn attention, nhưng độ chênh nhỏ hơn các phương pháp khác.
- **Downstream:** hiệu năng zero-shot tương quan tốt với PPL và LLM Surgeon hơn K-OBD (Table 5, 6 ở phụ lục; bản text của Table 5 không nêu rõ mô hình nên ta chỉ dựa vào Table 8 cho 13B).

## 6. Giới hạn

**Do tác giả nêu:**
- Chi phí tính toán cao: unstructured mất hơn 2 ngày trên 4 H100 (SparseGPT dưới 5 phút); họ lập luận nén chỉ làm một lần, nhưng thừa nhận đây là đánh đổi tính toán lấy chất lượng, và khuyến nghị dùng phương pháp này cho structured.
- Giả thiết gradient bằng 0 (cực tiểu) có thể không đúng; LoRA xen kẽ là giải pháp tuỳ chọn và chỉ ích cho mô hình nhỏ.
- Cắt structured 50% không hơn mô hình nhỏ cùng cỡ.

**Do ta nhận xét:**
- Thí nghiệm chủ yếu trên WikiText-2 (cả nén lẫn đánh giá): dùng cùng phân phối cho nén và test có thể làm số đẹp hơn; phụ lục có thử C4 và ngôn ngữ khác, nhưng bản text mà ta đọc chỉ là bảng.
- Phép cắt hàng và cột được xử lý **trong từng ma trận**; bản text chính không nói rõ cách đảm bảo kích thước nhất quán giữa các ma trận nối nhau qua residual stream (ví dụ cột của ma trận sau phải khớp hàng của ma trận trước). Cần xem phụ lục A.3, A.4 hoặc mã nguồn trước khi dùng cho mục đích triển khai (tốc độ).
- Độ cong KFAC giả định độc lập giữa activation và gradient, và bỏ qua tương quan giữa các layer; chất lượng xấp xỉ với kiến trúc không phải Transformer chuẩn (GatedDeltaNet) chưa được kiểm tra.
- Cần backprop cho cả mô hình nên cần bộ nhớ lớn (paper dùng 4 đến 8 H100 80GB cho 7B/13B).
- Không đo tốc độ chạy thật của mô hình đã cắt.

## 7. Liên hệ với các bài khác

- **01 (OBD/OBS):** nền lý thuyết; paper viết lại bằng ngôn ngữ Laplace/Gaussian.
- **22, 23 (OBC, SparseGPT):** dùng độ cong của lỗi tái tạo layer; LLM Surgeon thêm $G$ (gradient) và cắt toàn cục. Bảng 3 và 4 so trực tiếp.
- **24 (Wanda):** nhận định của paper (phụ lục H) rằng các thuật toán xét ít tương quan, thậm chí bỏ gradient (Wanda, SparseGPT), rất nhanh nhưng không đủ linh hoạt cho structured.
- **25 (ZipLM):** cùng công thức bù OBS cho nhóm cột; ZipLM chỉ có độ cong cục bộ (không $G$), nhưng thêm latency table; LLM Surgeon có ngưỡng toàn cục theo chi phí (không theo ms).
- **26 (LLM-Pruner):** điểm bậc nhất bằng gradient trên 10 mẫu, không bù; LLM Surgeon cho cả điểm lẫn bù dựa trên Fisher.
- **29 (SliceGPT):** concurrent; cả hai xoá hàng/cột nhưng LLM Surgeon xoá trong cấu trúc có sẵn, SliceGPT xoay trước để xoá các chiều của residual stream.
- **28 (FLAP):** bù bằng bias thay vì cập nhật trọng số.

## 8. Ứng dụng cho dự án (suy luận)

- **Liên hệ D_lin / D_res.** Cập nhật $\Delta\theta$ ở Eq. 8 là nghiệm *tối ưu theo xấp xỉ bậc hai của loss thật* (không chỉ lỗi đầu ra layer) trong không gian trọng số còn lại. Do đó phần hại mà cập nhật chung sửa được gần với $D_{lin}$ tính theo loss; chi phí $L_r,L_c$ (Eq. 10) là **hại đã tính *trước* khi bù**, không phải sau bù. Điều này khác ZipLM: điểm chọn của ZipLM là sai số *sau bù* (gần $D_{res}$ cục bộ), còn chi phí của LLM Surgeon cho từng hàng/cột độc lập là sai số đơn lẻ trước khi gộp cập nhật. Nói cách khác, xếp hạng của LLM Surgeon theo $D_{lin}+D_{res}$ (một phần theo loss), rồi bù ($D_{lin}$ theo loss) làm giảm chi phí thực. Đây là diễn giải của ta; paper không dùng ngôn ngữ này.
- **Nút thắt phù hợp với quan sát của bạn về MLP.** Bảng 1 cho thấy cắt structured 20 đến 30% cần *cả* gradient lẫn bù tương quan; K-OBD (có curvature nhưng không bù) sụp ở Llama-v2 7B (46.64 PPL ở 50%, 9.14 ở 80%), trong khi bù đầy đủ còn 15.38 và 6.18. Điều này gợi ý rằng "cắt neuron MLP hồi phục kém" có thể bớt kém nếu có bước bù, nhưng cần thử trên mô hình của bạn.
- **Thực tế triển khai: nặng.** Backprop + KFAC cho mô hình 0.8B nhỏ hơn nhiều so với 7B/13B của paper, nên 1 GPU lớn là đủ (ước lượng của ta, không phải số của paper). Thích hợp để dùng *một lần* để chọn phân bổ sparsity toàn cục (ngưỡng $\tau$) cho từng layer, rồi dùng kết quả như một "bản đồ độ nhạy" thay cho đo thủ công (so Fig. 4). Mô hình có GatedDeltaNet cần tự cài lại việc thu thập $A,G$ cho các projection của nó.
- **Ngưỡng toàn cục là ý tưởng dùng được ngay** nếu bạn có điểm quan trọng chuẩn hoá được giữa ViT, GatedDeltaNet và full-attention; so với phân bổ theo latency của ZipLM (25), cần chọn một trong hai hoặc kết hợp (chi phí/ms).
- **Distillation:** bù OBS và distillation bổ sung nhau: bù cho khởi tạo tốt, distill từ thầy 4B làm phần còn lại (dự đoán của ta).

## 9. Thí nghiệm gợi ý

1. **Ước lượng $D_{lin}$ theo loss bằng KFAC trên một layer:** với một projection MLP của decoder, tính $L_c$ cho từng cột (neuron), đo thực tế loss tăng khi xoá có và không có cập nhật $\Delta W=-\theta_c(A^{-1}e_c)^\top/[A^{-1}]_{cc}$. Tỷ số hai đại lượng là ước lượng thực nghiệm cho "bao nhiêu hại là tuyến tính được".
2. **Bản đồ độ nhạy toàn cục:** chạy bước tính chi phí (không cập nhật) cho mọi hàng/cột của một mô hình nhỏ, sắp xếp toàn cục, và vẽ phần trăm bị cắt ở mỗi layer/loại ma trận để so với phân bổ đều (so Fig. 4).
3. **Số shot:** so 1 shot và 4 shot khi cắt neuron MLP xuống 0.3B, đo PPL/CER trước khi distill.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Bài toán | $\min\frac12\Delta\theta^\top F\Delta\theta$ s.t. $e_k^\top(\theta+\Delta\theta)=0$ |
| Chi phí tổng quát | $\mathcal L=\frac12(E\theta)^\top(EF^{-1}E^\top)^{-1}(E\theta)$ |
| Cập nhật tổng quát | $\Delta\theta=-F^{-1}E^\top(EF^{-1}E^\top)^{-1}E\theta$ |
| KFAC | $F_l\approx G_l\otimes A_l$; $G$ từ gradient đầu ra, $A$ từ activation đầu vào |
| Chi phí hàng / cột | $L_r=\frac12\theta_r^\top A\theta_r/[G^{-1}]_{rr}$; $L_c=\frac12\theta_c^\top G\theta_c/[A^{-1}]_{cc}$ |
| Ngưỡng toàn cục | $\tau$ chung, cắt hàng/cột có chi phí $\le\tau$ cho đến khi còn $\alpha$ |
| Multi-shot | $T$ shot, lịch $\alpha_t$ tuyến tính, tính lại độ cong mỗi shot |
| Con số chính | Llama-v2 7B structured: 5.25 (90%), 6.18 (80%), 7.83 (70%); 13B: PPL 5.286 ở 80%, avg 72.12 so với 74.07 gốc; thời gian 17h08 (7B) trên 4 H100 |

## 11. Kết luận ngắn

LLM Surgeon đưa OBS về đúng "tinh thần ban đầu" cho LLM bằng KFAC có gradient, ngưỡng toàn cục, cập nhật có tương quan và nhiều shot, cho kết quả structured thực sự dùng được ở mức cắt 20 đến 30% (và SOTA cho unstructured/2:4 lúc công bố). Đổi lại, nó tốn tính toán rất nhiều, và thông điệp quan trọng cho ta là: ở structured, **bù trọng số có tương quan và thông tin gradient đều đáng kể**; chúng giải thích nhiều chênh lệch giữa các phương pháp cắt rẻ tiền và đắt tiền.

Link gốc: https://arxiv.org/abs/2312.17244
