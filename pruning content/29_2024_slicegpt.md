# SliceGPT: Compress Large Language Models by Deleting Rows and Columns (2024)

**Tác giả:** Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari do Nascimento, Torsten Hoefler, James Hensman (ETH Zurich; Microsoft Research; Microsoft).  
**Venue / arXiv:** arXiv:2401.15024 (26/01/2024, bản v2 02/2024). Published as a conference paper at ICLR 2024.  
**Nguồn:** https://arxiv.org/abs/2401.15024  
**Chủ đề:** Cắt **chiều nhúng (embedding / residual-stream dimension)** của Transformer sau huấn luyện: xoay không gian bằng ma trận trực giao theo PCA (computational invariance) rồi xoá các hàng/cột tương ứng thành phần chính nhỏ, thu được ma trận dày nhỏ hơn.  
**Vị trí trong lộ trình:** Đọc sau 06 (ThiNet, cắt kênh rồi bù tuyến tính), 23 (SparseGPT), 26 (LLM-Pruner), 27 (LLM Surgeon). Bài này khác cả nhóm: thay vì chọn cấu trúc để xoá, nó **đổi hệ trục** để cấu trúc xoá được trở nên ít quan trọng. Đây là bài sát nhất với ý tưởng "bù tuyến tính tối ưu tại điểm cắt" (D_lin) của dự án. Bài kế: 30 (MaskLLM), 31 (ShortGPT).

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

Pruning sau huấn luyện (SparseGPT, Wanda; bài 23, 24) tạo ma trận **thưa** nhưng cần cấu trúc dữ liệu phụ và kernel đặc biệt; hướng 2:4 (bài 18) chỉ tăng tốc hạn chế và chất lượng giảm. Structured pruning kiểu LLM-Pruner (26) và LLM Surgeon (27) cần fine-tune hoặc nhiều tính toán. Paper nêu rằng baseline đơn giản "xoá cột/hàng có chuẩn nhỏ nhất" rất kém: WikiText-2 PPL "vọt lên hàng nghìn" chỉ sau khi xoá vài cột.

Các khái niệm cần có:
- **RMSNorm / LayerNorm:** chuẩn hoá từng hàng (từng token) của tín hiệu. RMSNorm chia hàng cho chuẩn L2 của nó, rồi nhân hệ số. LayerNorm = trừ trung bình hàng rồi chia độ lệch chuẩn.
- **Signal matrix $X$:** ma trận kích thước (độ dài chuỗi $N$) $\times$ (chiều nhúng $D$) chảy qua các block.
- **Residual stream:** đường skip cộng đầu ra từng block vào tín hiệu chính.
- **PCA:** chiếu dữ liệu lên các hướng có phương sai lớn nhất.
- **Ma trận trực giao $Q$:** $Q^\top Q=QQ^\top=I$, không đổi chuẩn vector.

## 2. Ý tưởng chính

Giữa các block của một Transformer dùng RMSNorm, ta có thể **xoay** tín hiệu bằng một ma trận trực giao $Q$ trước RMSNorm và xoay ngược bằng $Q^\top$ sau RMSNorm mà mô hình tính ra đúng như cũ (vì phép chuẩn hoá chỉ phụ thuộc chuẩn của hàng, mà phép xoay không đổi chuẩn). Điều này gọi là **computational invariance**. Chọn $Q$ là các thành phần chính (PCA) của tín hiệu giữa các block, thì sau khi xoay, năng lượng tập trung vào vài chiều đầu; các chiều cuối ít quan trọng nên **xoá được** bằng cách bỏ hàng của $W_{in}$ và cột của $W_{out}$ tương ứng. Kết quả là mọi ma trận nhỏ hơn và tín hiệu giữa các block cũng nhỏ hơn (giảm chiều nhúng), nên tăng tốc bằng kernel dày thông thường và cần ít GPU hơn. Không cần fine-tune (nhưng fine-tune nhẹ vẫn cải thiện).

## 3. Phương pháp chi tiết

### 3.1 Cấu trúc Transformer dưới dạng $\sigma(XW_{in}+b_{in})W_{out}+b_{out}$

Mỗi khối attention hoặc FFN đều viết được dạng trên: $W_{in}$ là ghép của $W_k,W_q,W_v$ (attention) hoặc $W_1$ (và gate) (FFN), $\sigma$ là phép phi tuyến theo điểm (hoặc multi-head attention), $W_{out}$ là ma trận đầu ra. Giữa các khối có RMSNorm và đường residual (Algorithm 1: $X\leftarrow\mathrm{Norm}_\ell(X+Z)$).

### 3.2 Computational invariance

RMSNorm áp dụng $x\leftarrow x/\|x\|$ cho từng hàng. Với $Q$ trực giao:

$$\boxed{\mathrm{RMSNorm}(XQ)\,Q^\top=\mathrm{RMSNorm}(X)}\qquad(2)$$

(vì $\|Qx\|=\|x\|$). **Định lý 1:** mạng sau đây tương đương mạng gốc nếu thay

$$\widetilde W_{embd}=W_{embd}Q,\quad \widetilde W^\ell_{in}=Q^\top W^\ell_{in},\quad \widetilde W^\ell_{out}=W^\ell_{out}Q,\quad \tilde b^\ell_{out}=Q^\top b^\ell_{out},\quad \widetilde W_{head}=Q^\top W_{head}$$

(với $b_{in}$, $b_{head}$ giữ nguyên). Tức là mọi thứ đọc từ residual stream thì nhận $Q^\top$, mọi thứ ghi vào thì nhân $Q$, và residual stream được "xoay" nhất quán từ embedding đến đầu LM.

**LayerNorm sang RMSNorm.** Mạng dùng LayerNorm (OPT, Phi-2) phải đổi trước: hấp thụ hệ số $\mathrm{diag}(\alpha)$ vào $W_{in}$ kế tiếp, và việc trừ trung bình (ma trận $M=I-\frac1D\mathbf 1\mathbf 1^\top$) vào $W_{out}$ của block trước; $W_{embd}$ cũng được trừ trung bình, $W_{head}$ được co theo hệ số cuối. Phép đổi này không đổi đầu ra.

### 3.3 Mỗi block một phép xoay, và đường skip

Tín hiệu ở các block không cùng hệ trục, nên PCA cho **một $Q_\ell$ riêng cho mỗi block**. Khi $Q_\ell$ khác nhau, đường skip (cộng đầu vào block vào đầu ra) bị lệch hệ trục, nên phải chèn một phép biến đổi tuyến tính vào đường skip:

$$\text{residual}:\ X\leftarrow X\,(Q_{\ell-1}^\top Q_\ell)$$

Ma trận $D\times D$ này **không tính trước được** vào trọng số khác và tạo ra một chi phí nhỏ (nhưng thật) $D\times D$; paper nói vẫn thấy tăng tốc tổng thể. Sau khi slice, nó thành ma trận $D_{small}\times D_{small}$ (xoá cả hàng lẫn cột).

### 3.4 Chọn $Q_\ell$ bằng PCA và cắt

Chạy một tập calibration qua mô hình (đã đổi sang RMSNorm). Gọi $X_{\ell,i}$ là đầu ra RMSNorm thứ $\ell$ của chuỗi $i$:

$$C_\ell=\sum_iX_{\ell,i}^\top X_{\ell,i},\qquad Q_\ell=\text{eigenvectors}(C_\ell)\ \text{xếp theo trị riêng giảm dần}.$$

Dùng đầu ra của mạng *đã biến đổi* để tính $Q$ của các block sau (làm tuần tự). PCA chuẩn: $Z=XQD$, $\widetilde X=ZD^\top Q^\top$, trong đó $D$ là ma trận xoá ($D_{small}$ cột của ma trận đơn vị $D\times D$). Phép tái tạo này **tối ưu L2**: $QD$ là ánh xạ tuyến tính (hạng $D_{small}$) cực tiểu hoá $\|X-\widetilde X\|^2$. Vì không bao giờ dựng $X$ đầy đủ, ta áp $D$ ngay vào các phép toán trước và sau: **xoá hàng của $W_{in}$ và cột của $W_{out}$, $W_{embd}$**, và cả hàng lẫn cột của $Q_{\ell-1}^\top Q_\ell$ trên đường skip.

```text
SliceGPT(model, calib, keep_ratio):
    model = layernorm_to_rmsnorm(model)          # hấp thụ diag(alpha), ma trận M
    X = embed(calib)
    for each block l = 0..L:
        Xn  = RMSNorm(X)
        C_l = sum_i Xn_i^T Xn_i                  # (D x D) hiệp phương sai
        Q_l = eigvecs(C_l)  sorted desc          # FP64 để tránh sai số
        # xoay:  W_in <- Q_l^T W_in ;  W_out_prev <- W_out_prev Q_l ;  skip: Q_{l-1}^T Q_l
        # cắt:   giữ D_small chiều đầu của Q_l
        X = forward_block_with_rotation(X, Q_l)  # dùng đầu ra đã xoay cho block kế
    slice rows of W_in, cols of W_out, W_embd; rows+cols of skip matrices
    # (tuỳ chọn) recovery fine-tuning bằng LoRA
```

**Ví dụ nhỏ tự dựng** (không phải của paper): $D=4$ và dữ liệu tín hiệu có trị riêng $[10,5,1,0.1]$. Xoay theo vector riêng rồi giữ 3 chiều đầu: năng lượng mất đi là $0.1/16.1\approx0.6\%$. Nếu cắt thẳng 1 trong 4 trục gốc (không xoay) mà năng lượng rải đều trên 4 trục (khoảng 25% mỗi trục) thì mất tới khoảng 25%. Chính phép xoay biến "mọi trục đều quan trọng" thành "vài trục rất quan trọng, vài trục gần như vô nghĩa".

**Slicing theo mức khác nhau giữa các layer.** Thay vì cắt đều, đặt phần phương sai muốn bỏ ở mỗi PCA (phụ lục A.4): xem mục 5.

## 4. Thí nghiệm và kết quả chính

**Thiết lập:** Hugging Face + PyTorch; $Q$ tính trên một H100 80GB (khoảng 3.5 giờ cho LLaMA-2 70B), **độ chính xác kép** cho PCA. **Mô hình:** OPT (125M đến 66B), LLaMA-2 (7B, 13B, 70B), Phi-2 (chỉ zero-shot). **Calibration:** WikiText-2 train hoặc Alpaca train; cỡ 1024 mẫu, độ dài 2048 cho kết quả chính. **Baseline:** SparseGPT 2:4 (vì là sơ đồ thưa duy nhất có tăng tốc); thử cả "xoá cột chuẩn nhỏ" nhưng PPL lên hàng nghìn nên bỏ. **Tỷ lệ cắt** tính trên **toàn bộ tham số kể cả embedding**.

**PPL WikiText-2 (Table 1):**

| Phương pháp | OPT-125M | OPT-6.7B | OPT-66B | LLaMA-2 7B | LLaMA-2 13B | LLaMA-2 70B |
|---|---|---|---|---|---|---|
| Dense | 27.64 | 10.85 | 9.33 | 5.47 | 4.88 | 3.32 |
| SparseGPT 2:4 | 45.07 | 13.00 | 10.22 | 8.69 | 7.07 | 4.98 |
| SliceGPT 10% | 29.34 | 10.92 | 9.43 | 5.89 | 5.21 | 3.69 |
| SliceGPT 20% | 34.26 | 11.48 | 9.57 | 6.64 | 5.81 | 4.25 |
| SliceGPT 25% | 37.74 | 11.90 | 9.68 | 7.24 | 6.30 | 4.60 |
| SliceGPT 30% | 43.98 | 12.51 | 9.85 | 8.12 | 6.99 | 5.05 |

Nhận xét của paper: SparseGPT 2:4 kém SliceGPT 25% trên mọi LLaMA-2; với OPT, SliceGPT 30% vượt 2:4 ở mọi cỡ trừ 2.7B; OPT nén tốt hơn LLaMA-2; mô hình càng lớn càng nén tốt hơn.

**Zero-shot (5 tác vụ: PIQA, WinoGrande, HellaSwag, ARC-e, ARC-c):** lệch dense ít khi mô hình lớn. Ở 25% slice: OPT-66B giữ 99%, LLaMA-2 70B giữ 96%, Phi-2 giữ 87% hiệu năng dense (phần Kết luận). **Recovery fine-tuning (RFT) bằng LoRA** (rank 32, alpha 10; khoảng 1k chuỗi WikiText-2 hoặc 5k chuỗi Alpaca): LLaMA-2 70B cắt 25% đạt 99% dense và Phi-2 90%; LLaMA-2 70B cắt 30% với RFT Alpaca: 74.3% so với 76.6% dense (mô hình khoảng 51.6B tham số); Phi-2 cắt 25% với RFT: 65.2% so với 72.2% dense (khoảng 2.2B tham số, giữ 90.3%). Không tìm được cấu hình RFT cải thiện OPT. RFT trên WikiText-2 yếu hơn Alpaca nhiều (tác giả đổ cho độ giống giữa Alpaca và các tác vụ).

**Tốc độ:**
- Throughput H100 80GB (độ dài 128, tăng batch tối đa): mô hình cắt 25% đạt tới **1.55×**; cắt 50% thì mô hình lớn nhất chỉ cần 1 GPU thay vì 2, throughput tăng **3.13×** và **1.87×** (tức 6.26× và 3.75× với số GPU cố định).
- Sinh 1 token, batch 1, độ dài 128 (Table 2): LLaMA-2 70B trên A100 40GB: dense 125 ms/4 GPU (500 GPUms), cắt 25%: 110 ms/3 GPU (330 GPUms, còn 66%); trên Quadro RTX6000 24GB: dense 252 ms/7 GPU (1764 GPUms), 25%: 215 ms/5 GPU (1075 GPUms, còn 64%). OPT 66B tương tự (A100: 456 giảm còn 306 GPUms). Tốc độ đơn lẻ nhanh hơn 16 đến 17% (RTX6000) và 11 đến 13% (A100).
- Thời gian nhân ma trận (phụ lục, Table 12): LLaMA-2 7B tổng 3.99 ms (dense), 2.70 ms (2:4, 1.48×), 2.99 ms (SliceGPT 25%, 1.33×), 2.06 ms (SliceGPT 50%, 1.94×); LLaMA-2 70B 25%: 1.32× bằng SparseGPT 2:4 nhưng PPL 4.60 so với 4.98.
- **Chi phí cắt** (Table 3, cắt 30% + RFT Alpaca): LLaMA-2 7B 0h44m + 0h23m = 1h07m (1 H100); 13B 1h08m + 0h44m = 1h52m; 70B 3h31m + 1h35m (4 H100) = 5h06m; Phi-2 0h49m + 1h59m = 2h48m (1 V100 32GB).

## 5. Phân tích / ablation

- **Độ chính xác số học của PCA (phụ lục A.2, Table 4):** dùng FP32 thay FP64 làm hỏng mô hình lớn: LLaMA-2 70B 25%: PPL 7.01 (FP32, calibration 128 mẫu) so với 4.89 (FP64, Table 5 cùng cỡ calibration); 7B 25%: 8.52 so với 7.56. Hệ quả thực hành: phải dùng độ chính xác kép.
- **Kích thước và độ dài calibration (A.3):** nên dùng ít nhất 128 mẫu; chuỗi dài hơn (tới 4096) cho PPL tốt hơn; mặc định 1024 mẫu và độ dài 2048. Giảm cỡ calibration (128 mẫu) làm PPL xấu đi trên mọi mô hình (Table 5: LLaMA-2 7B 25% 7.56 so với 7.24).
- **Phổ trị riêng (A.4, Fig. 8):** phổ của OPT 6.7B suy giảm nhanh hơn LLaMA-2 7B (LLaMA-2 có phân phối trị riêng "dẹt" hơn, ít thành phần chính vượt trội), giải thích vì sao OPT nén tốt hơn. Phổ giảm nhanh hơn ở các layer đầu, gợi ý cắt nhiều hơn ở layer đầu.
- **Cắt khác nhau theo layer (Table 6):** đặt phần phương sai cần bỏ ở mỗi PCA để tổng khoảng 25%. OPT cải thiện nhẹ (ví dụ 6.7B: 12.10 xuống 11.94 với 24.7% tổng; 13B: 11.04 xuống 10.76), còn LLaMA-2 thì **xấu đi** (7B: 6.84 lên 7.63; 13B 6.00 lên 6.17; 70B 4.44 lên 4.63). Vì vậy bản chính dùng cắt đều.
- **Chọn dữ liệu RFT:** Alpaca tốt hơn WikiText-2 rõ rệt.

## 6. Giới hạn

**Do tác giả nêu:**
- Với mô hình $\le$ 13B, **mô hình dense nhỏ hơn cùng số tham số tốt hơn** mô hình lớn bị cắt (tác giả nghĩ điều này sẽ không còn đúng lâu).
- Mô hình cắt có nhiều tham số hơn mô hình cắt bằng SparseGPT cùng độ nén nhưng cho phép batch lớn hơn và không có overhead cấu trúc thưa; kết hợp hai cách là hướng mở.
- Có thể có cách tính $Q$ tốt hơn PCA.

**Do ta nhận xét:**
- PCA tối ưu cho tái tạo **tín hiệu** $X$ (L2), không phải cho **loss**. Một thành phần chính nhỏ có thể vẫn quan trọng cho đầu ra; không có gradient trong tiêu chí.
- $Q_{\ell-1}^\top Q_\ell$ trên đường skip làm tăng số tham số và chi phí; "25% slicing" chưa tương đương 25% FLOPs (Table 12 cho 1.30 đến 1.33× nhân ma trận ở 25%, trong khi chi phí đầu cuối và overhead khác nhau).
- Kết quả tốt nhất cho OPT và mô hình lớn; với LLaMA-2 7B ở 30% PPL 8.12 so với 5.47: hơn 48% tăng, còn lớn.
- Cắt **chiều chung của residual stream**: cắt mọi layer cùng lúc; không cho kiểm soát cắt riêng MLP hay riêng attention.
- Không thử trên kiến trúc lai (SSM, linear-attention), mô hình thị giác hay đa phương thức.

## 7. Liên hệ với các bài khác

- **06 (ThiNet):** paper trích ThiNet là phương pháp gần nhất ("áp dụng phép tuyến tính giữa các layer"), nhưng ThiNet cần fine-tune và nhắm mô hình nhỏ hơn hàng bậc.
- **22, 23 (OBC, SparseGPT) và 27 (LLM Surgeon):** bù trọng số bằng nghịch đảo Hessian trong cùng ma trận; SliceGPT bù bằng cách chọn hệ trục. LLM Surgeon (concurrent) so sánh bằng cách nói SliceGPT bỏ qua gradient.
- **25 (ZipLM):** cùng cắt chiều trung gian của FFN và head nhưng giữ nguyên chiều residual; SliceGPT cắt đúng chiều residual mà ZipLM không đụng đến.
- **26 (LLM-Pruner):** nhóm "Channel" (chiều ẩn) của LLM-Pruner tệ; SliceGPT là cách làm đúng việc đó có bù.
- **28 (FLAP):** bù hằng số; SliceGPT bù tuyến tính đầy đủ qua $Q$.
- **18 (2:4):** SliceGPT dùng 2:4 SparseGPT làm baseline tốc độ.
- **31 đến 36 (ShortGPT, v.v.):** cắt theo chiều sâu (layer); SliceGPT cắt theo chiều rộng của residual. Đáng so khi tính ngân sách.
- **36 (ReplaceMe):** theo ta hiểu là thay layer bị bỏ bằng một ánh xạ tuyến tính; tinh thần "xấp xỉ tuyến tính tối ưu" tương tự (ta chưa đối chiếu bài 36 khi viết note này).

## 8. Ứng dụng cho dự án (suy luận)

**Liên hệ D_lin / D_res (mạnh nhất trong 7 bài).**
- PCA trên tín hiệu residual cho **ánh xạ tuyến tính hạng $D_{small}$ tối ưu theo L2** ($QD$ cực tiểu hoá $\|X-XQDD^\top Q^\top\|^2$). Đúng nghĩa đen đây là "ánh xạ tuyến tính tối ưu tại điểm cắt" cho điểm cắt là *chiều của residual stream*. Phần $\|X-\widetilde X\|^2$ còn lại (năng lượng các thành phần chính bị bỏ) là một **$D_{res}$ đo ở tín hiệu**, còn khả năng tái dựng bằng $QD$ là $D_{lin}$ (nếu ta định nghĩa $D_{lin}$ là phần tái tạo được bằng ánh xạ tuyến tính tối ưu tại điểm cắt). Phần năng lượng thấp bị bỏ không có nghĩa là loss không đổi, nên cần kiểm chứng bằng loss/CER.
- Ma trận $Q_{\ell-1}^\top Q_\ell$ trên đường skip là một **ánh xạ tuyến tính thực sự được thêm vào mô hình** như phần bù. Nếu cắt layer trong decoder lai thì cũng có thể cần chèn một ánh xạ như vậy (gợi ý ta rút ra từ cấu trúc của paper, không phải kết quả của paper).
- Quan sát "cắt MLP neuron phục hồi kém hơn cắt layer" có thể được so với cắt chiều residual: cả hai đều cắt một "chiều rộng", nhưng SliceGPT cho thấy cắt đúng sau khi xoay hệ trục thì chiều residual nén được khá nhiều (LLaMA-2 70B 25% gần dense). Mô hình 0.8B nhỏ hơn rất nhiều so với 70B và theo chính paper mô hình nhỏ chịu cắt kém hơn; nên đừng kỳ vọng số liệu của 70B với mô hình nhỏ.
- **Kiến trúc của dự án:** nếu mọi block (ViT 12 block; decoder hybrid GatedDeltaNet + full attention) đều có dạng "đọc residual bằng ma trận tuyến tính, ghi lại bằng ma trận tuyến tính, có RMSNorm/LayerNorm ở residual", thì computational invariance áp dụng được ở **mức luồng residual**; phần bên trong GatedDeltaNet (conv ngắn, chuẩn hoá theo head, cổng) nằm giữa $W_{in}$ và $W_{out}$ nên không bị ảnh hưởng. Cần kiểm tra thủ công: các norm bên trong block (nằm sau $W_{in}$) không cản trở; các norm có tham số ở residual phải hấp thụ được; ViT dùng LayerNorm cần đổi sang RMSNorm trước. Đây là suy luận, chưa được paper kiểm tra cho kiến trúc này.
- **Giới hạn phần cứng:** SliceGPT giảm D (nhỏ hơn cả bộ nhớ activation và KV/state), không đổi số layer; batch-1 latency phụ thuộc số lần launch kernel nhiều hơn số FLOPs, nên cần đo trước khi kỳ vọng (xem 25 về depth/width ở batch nhỏ).
- **Chi phí:** PCA double-precision trên mô hình 0.8B chỉ cỡ phút (suy đoán, từ 3.5 giờ cho 70B).

## 9. Thí nghiệm gợi ý

1. **Phổ residual:** thu tín hiệu residual (tại 3 đến 4 điểm giữa các block) của mô hình LightOnOCR trên dữ liệu OCR, vẽ phổ trị riêng (như Fig. 8). Dốc nhanh nghĩa là có chỗ cắt chiều residual; dẹt (như LLaMA-2) nghĩa là khó. Đây là thí nghiệm rẻ nhất để biết SliceGPT có hợp không.
2. **Sai số tín hiệu so với sai số loss:** cắt 10%, 20%, 30% chiều residual bằng PCA; vẽ phần năng lượng bị bỏ ($D_{res}$ tín hiệu) so với tăng loss/CER. Nếu hai đại lượng lệch nhau nhiều, PCA không phải tiêu chí tốt cho OCR.
3. **Bù ở điểm cắt layer:** thay vì cắt layer thuần, cắt layer rồi chèn ánh xạ tuyến tính tối ưu theo bình phương tối thiểu (ReplaceMe-style); so với bỏ layer thuần để ước lượng $D_{lin}$ ở mức layer.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Nội dung |
|---|---|
| Computational invariance | $\mathrm{RMSNorm}(XQ)Q^\top=\mathrm{RMSNorm}(X)$ với $Q$ trực giao |
| Biến đổi mạng | $\widetilde W_{embd}=W_{embd}Q$, $\widetilde W_{in}=Q^\top W_{in}$, $\widetilde W_{out}=W_{out}Q$, $\tilde b_{out}=Q^\top b_{out}$, $\widetilde W_{head}=Q^\top W_{head}$ |
| Skip khi $Q_\ell$ khác nhau | nhân thêm $Q_{\ell-1}^\top Q_\ell$ ($D\times D$, không tính trước được) |
| PCA | $C_\ell=\sum_iX_{\ell,i}^\top X_{\ell,i}$, $Q_\ell$ = vector riêng sắp theo trị riêng giảm dần |
| Slicing | $\widetilde X=XQDD^\top Q^\top$; xoá hàng $W_{in}$, cột $W_{out}$, $W_{embd}$, hàng+cột của ma trận skip |
| LayerNorm sang RMSNorm | hấp thụ $\mathrm{diag}(\alpha)$ vào $W_{in}$, $M=I-\frac1D\mathbf 1\mathbf 1^\top$ vào $W_{out}$ |
| Cấu hình | calibration 1024 mẫu x 2048 token, PCA FP64, RFT LoRA r=32 |
| Con số chính | LLaMA-2 70B 25%: PPL 4.60 (dense 3.32, SparseGPT 2:4 4.98); chi phí tính 66% (A100) / 64% (RTX6000); throughput 25% tới 1.55× |

## 11. Kết luận ngắn

SliceGPT biến "xoá chiều residual" thành khả thi nhờ tính bất biến tính toán: xoay theo PCA rồi xoá các thành phần nhỏ, kèm một ánh xạ tuyến tính $D_{small}\times D_{small}$ ở mỗi đường skip. Kết quả tốt với mô hình lớn (70B 25% gần dense, nhanh hơn, ít GPU hơn), kém hơn với mô hình nhỏ. Với dự án, giá trị chính là khung tư duy "bù tuyến tính tối ưu tại điểm cắt" ở dạng cụ thể, và phép đo phổ residual rẻ để dự đoán khả năng nén theo chiều rộng của luồng residual.

Link gốc: https://arxiv.org/abs/2401.15024
