# SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot (2023)

**Tác giả:** Elias Frantar, Dan Alistarh (IST Austria; Neural Magic)  
**Venue / arXiv:** arXiv:2301.00774 (v3, 22/3/2023); bản text không ghi tên venue (bài trình bày theo định dạng ICML). Code: github.com/IST-DASLab/sparsegpt.  
**Nguồn:** https://arxiv.org/abs/2301.00774  
**Chủ đề:** Pruning một lần (one-shot, không fine-tune) cho LLM cỡ 10-175B tham số; bài toán hồi quy thưa theo lớp; cập nhật kiểu OBS với **chia sẻ Hessian giữa các hàng**; chọn mask thích nghi theo khối; N:M; kết hợp lượng tử hoá.  
**Vị trí trong lộ trình:** Đọc ngay sau **22 (OBC)**, vì SparseGPT là "OBC mở rộng lên LLM bằng một xấp xỉ có kiểm soát". Cũng nên đọc cùng 18 (2:4) và 21 (survey). Tiếp theo: 24 (Wanda: bỏ hẳn bước cập nhật trọng số, chỉ dùng |w|·‖x‖), 27 (LLM Surgeon), 30 (MaskLLM).

> Đây là ghi chú học tập bằng tiếng Việt, không phải bản dịch. Phần "Ứng dụng cho dự án" là suy luận của chúng ta. Lưu ý quan trọng: SparseGPT cắt **trọng số lẻ (unstructured hoặc N:M)** của mô hình văn bản OPT/BLOOM; nó **không** giảm kích thước kiến trúc, nên quan hệ với dự án (cắt cấu trúc xuống 0.3B/0.1B) là gián tiếp.

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

Mô hình GPT 175B cần ít nhất 320 GB ở FP16 và ít nhất 5 GPU A100 80 GB để suy luận. Trước bài này, nén LLM gần như chỉ là **quantization** (LLM.int8, ZeroQuant, SmoothQuant, GPTQ). Pruning chính xác thì cần retrain dài, không khả thi ở quy mô này. Các phương pháp pruning post-training chính xác (AdaPrune, OBC, bài 22) chỉ chạy được trên mô hình cỡ vài trăm triệu tham số: AdaPrune đã mất vài giờ cho mô hình 1.3B, ngoại suy tuyến tính lên 175B sẽ cỡ vài tuần; OBC có độ phức tạp tăng như luỹ thừa 4 của kích thước ẩn. Magnitude pruning là baseline duy nhất chạy được ở quy mô lớn, nhưng theo bài báo mất chính xác rất nhanh trên LLM.

Nền tảng cần nhớ từ bài 22 (OBC): bài toán theo từng lớp

$$\arg\min_{M_\ell,\hat W_\ell}\ \|W_\ell X_\ell-(M_\ell\odot\hat W_\ell)X_\ell\|_2^2 \tag{1}$$

với $M_\ell$ là mask nhị phân. Bài toán này (chọn mask cùng cập nhật trọng số) là NP-hard, nên cần xấp xỉ.

Giải thích thuật ngữ:
- **One-shot / post-training pruning:** nén một lần trên mô hình đã train với ít dữ liệu hiệu chuẩn, không train lại.
- **Sparse regression (hồi quy thưa):** tìm vector trọng số ít thành phần khác 0 sao cho dự đoán gần nhất.
- **Cholesky decomposition:** phân tích $A=LL^\top$ với ma trận đối xứng xác định dương, ổn định số học.
- **Semi-structured N:M:** trong mỗi M trọng số liên tiếp có đúng N số 0 (xem 18).

## 2. Ý tưởng chính

Với mask cố định, cập nhật tối ưu trọng số mỗi hàng là hồi quy bình phương tối thiểu, nhưng mỗi hàng có mask khác nhau nên cần nghịch đảo một Hessian khác nhau cho mỗi hàng: quá đắt. SparseGPT **ép mọi hàng dùng chung một chuỗi các Hessian nghịch đảo** bằng cách xử lý các cột theo một thứ tự cố định (từ trái sang phải): khi đến cột $j$, mọi hàng đều chỉ cho phép cập nhật các trọng số ở bên phải $j$ (chưa xử lý), nên tập chỉ số "còn cập nhật được" giống nhau cho mọi hàng và chỉ phải tính một chuỗi nghịch đảo. Phần giá phải trả: mỗi lần xoá chỉ bù vào các trọng số ở bên phải thay vì tất cả trọng số còn lại (cập nhật "một phần", tối ưu trong tập con). Để mask không bị ép phải đều theo cột, mask được chọn thích nghi theo khối 128 cột dựa trên sai số OBS. Kết quả: độ phức tạp từ $O(d_{row}d_{col}^3)$ xuống $O(d_{col}^3+d_{row}d_{col}^2)$, cho phép cắt OPT-175B trong khoảng 4 giờ trên một A100.

## 3. Phương pháp chi tiết

### 3.1 Với mask cố định: nghiệm đóng và cách nhìn lặp

Cho mask hàng $M_i$, các trọng số giữ lại tối ưu (Phương trình 2):

$$w_{M_i}=(X_{M_i}X_{M_i}^\top)^{-1}X_{M_i}(w_{M_i}X_{M_i})^\top$$

(tức bình phương tối thiểu trên các đặc trưng đầu vào còn giữ). Cần $(H_{M_i})^{-1}$ riêng cho mỗi hàng $i$: $O(d_{row}d_{col}^3)$; với Transformer là luỹ thừa 4 của $d_{hidden}$ (cần nhanh hơn $d_{hidden}$ lần).

**Cách nhìn lặp (OBS):** cập nhật OBS khi xoá trọng số $m$ (như dẫn xuất ở bài 22) là

$$\delta_m=-\frac{w_m}{[H^{-1}]_{mm}}H^{-1}_{:,m},\qquad \varepsilon_m=\frac{w_m^2}{[H^{-1}]_{mm}}\tag{3}$$

(ở đây $H=XX^\top$; paper SparseGPT dùng $H$ không nhân 2, hệ số nhân không ảnh hưởng $w$ đã cập nhật). Vì loss một hàng chính xác bậc hai, OBS đúng tuyệt đối; xoá lần lượt $m_1,\dots,m_p$ bằng OBS cho cùng nghiệm với giải đóng với mask đầy đủ.

### 3.2 Cập nhật một phần và đồng bộ Hessian

Nếu thay vì cập nhật mọi trọng số còn lại $M$, chỉ cập nhật tập con $U\subseteq M$ (dùng $H_U$ thay $H_M$), OBS vẫn tối ưu **trong tập con đó**, không thêm sai số xấp xỉ riêng; chỉ phần bù có thể kém hơn vì ít trọng số để điều chỉnh.

Cố định thứ tự cột $j=1,\dots,d_{col}$ và định nghĩa

$$U_1=\{1,\dots,d_{col}\},\qquad U_{j+1}=U_j-\{j\}.$$

Chuỗi nghịch đảo dùng chung cho mọi hàng: $(H_{U_j})^{-1}=((XX^\top)_{U_j})^{-1}$, và mỗi bước cập nhật được tính bằng một bước khử Gauss (như Lemma 1 của bài 22):

$$\boxed{(H_{U_{j+1}})^{-1}=\Big(B-\frac{1}{[B]_{11}}B_{:,1}B_{1,:}\Big)_{2:,2:}},\quad B=(H_{U_j})^{-1}$$

Mỗi bước $O(d_{col}^2)$; cả chuỗi $O(d_{col}^3)$ (bằng chi phí một phép nghịch đảo thêm).

**Quy tắc xử lý:** lần lượt qua các cột $j$; ở cột $j$, với **mọi hàng $i$ có $j\notin M_i$** (trọng số $w_{ij}$ bị cắt) áp dụng OBS dùng $(H_{U_j})^{-1}$ để xoá $w_{ij}$ và bù vào các trọng số ở cột $>j$ của hàng đó; các trọng số đã xử lý (cột $<j$) bị "đóng băng" và không còn được cập nhật. Mỗi nghịch đảo $(H_{U_j})^{-1}$ chỉ tính một lần và dùng cho mọi hàng.

**Độ phức tạp:** (a) tính Hessian ban đầu: $\Theta(n\,d_{col}^2)$, với $n$ số mẫu hiệu chuẩn (paper thấy $n$ cỡ một bội nhỏ của $d_{col}$ là đủ); (b) chuỗi nghịch đảo: $O(d_{col}^3)$; (c) áp dụng cập nhật cho mọi hàng, mọi cột: $O(d_{col}\,d_{row}\,d_{col})$. Tổng:

$$\boxed{O(d_{col}^3+d_{row}\,d_{col}^2)}$$

Với Transformer là $O(d_{hidden}^3)$, tức nhanh hơn tái tạo chính xác đúng $d_{hidden}$ lần.

**Cách diễn giải "đóng băng" (weight freezing):** nén một trọng số nghĩa là cố định nó tại một giá trị (0 nếu cắt, giá trị hiện tại nếu giữ) rồi không bao giờ đổi nữa. Định nghĩa $\text{compress}(w_j)_i=0$ nếu $j\notin M_i$, ngược lại giữ $w_{ij}$. SparseGPT khi đó là một lược đồ tham lam đúng theo cột (giống GPTQ). Cách nhìn này cho phép **gộp cắt tỉa và lượng tử hoá** trong một lượt.

### 3.3 Chọn mask thích nghi theo khối (iterative blocking)

Nếu dùng magnitude để chọn mask trước (như AdaPrune), bỏ qua việc các cập nhật làm thay đổi đáng kể trọng số. Cách đơn giản là chọn $p\%$ trọng số dễ cắt nhất trong **mỗi cột** khi tới lượt, nhưng như vậy ép mật độ đều theo cột; không hợp với LLM, vốn có vài đặc trưng (cột) outlier rất nhạy. Giải pháp:

- Chọn mask cho **$B_s=128$ cột một lần**, theo sai số OBS $\varepsilon=w^2/[H^{-1}]^2_{cc}$ lấy từ đường chéo chuỗi Hessian (trong thuật toán dùng đường chéo của nhân tử Cholesky), giữ $(1-p)\%$ trọng số có $\varepsilon$ lớn nhất trong khối.
- Rồi thực hiện cập nhật cho khối đó, sau đó chọn mask cho khối kế tiếp (đã tính các cập nhật trước).

Như vậy cột nào cũng có thể thưa khác nhau trong khối nhưng sparsity đều ở mức khối. (Với một cột đơn lẻ, tiêu chí thu về magnitude vì $[H^{-1}]_{jj}$ hằng với mọi hàng.)

### 3.4 Mở rộng N:M

Chọn $B_s=m$ và trong mỗi nhóm $m$ cột liên tiếp của mỗi hàng, ép đúng $n$ số 0 bằng cách chọn $n$ trọng số có sai số (3) nhỏ nhất. Khối lớn hơn $m$ không có ích với N:M vì các số 0 không được phân bố bất đều giữa các nhóm cột kích thước $m$.

### 3.5 Pseudo-code đầy đủ (Algorithm 1, tóm tắt)

```
# Input: W (d_row x d_col), Hinv = (X X^T + lambda*I)^{-1}, sparsity p,
#        lazy-update block B, mask block Bs (=128)
M = ones(d_row, d_col)                 # mask nhị phân
E = zeros(d_row, B)                    # sai số của khối
Hinv = Cholesky(Hinv)^T                # nhân tử trên; hàng j chứa thông tin chuỗi (H_Uj)^-1
for i in 0, B, 2B, ...:                # khối lazy-batch
    for j in i .. i+B-1:
        if j mod Bs == 0:
            # chọn mask cho Bs cột kế: giữ (1-p)% trọng số có w^2 / Hinv[c,c]^2 lớn nhất
            M[:, j:j+Bs] = top_(1-p)% theo  W[:, j:j+Bs]^2 / diag(Hinv)^2
        E[:, j-i] = W[:, j] / Hinv[j, j]          # sai số cắt (chuẩn hoá)
        E[:, j-i] = (1 - M[:, j]) * E[:, j-i]     # chỉ giữ ở vị trí bị cắt (đóng băng phần còn lại)
        W[:, j:i+B] -= E[:, j-i] * Hinv[j, j:i+B] # bù vào các cột bên phải trong khối
    W[:, i+B:] -= E @ Hinv[i:i+B, i+B:]           # lazy: bù vào phần còn lại, một lần
W = W * M                                         # đặt trọng số bị cắt về 0
```

Chi tiết cài đặt kế thừa từ GPTQ: dùng **Cholesky** của $H^{-1}$ để lấy sẵn toàn bộ thông tin chuỗi nghịch đảo một cách ổn định số học, và **cập nhật lazy theo khối** $B$ để tăng tỉ lệ tính toán/bộ nhớ. (Ghi chú của chúng ta: việc hàng $j$ của nhân tử Cholesky chứa đúng hàng của $(H_{U_j})^{-1}$ là một tính chất đại số được GPTQ dùng; paper chỉ nêu lợi ích, không dẫn xuất lại.) Hessian được làm "dampening" bằng $\lambda I$ với $\lambda=1\%$ trung bình đường chéo.

### 3.6 Kết hợp lượng tử hoá

Vì cùng khung đóng băng theo cột như GPTQ, tất cả trọng số bị đóng băng được **lượng tử hoá luôn**, và sai số cần bù là (Phương trình 7):

$$E_{:,j-i}\leftarrow\frac{W_{:,j}-M_{:,j}\odot \text{quant}(W_{:,j})}{[H^{-1}]_{jj}}$$

(trong đó $M=0$ tại vị trí cắt nên sai số là chính giá trị trọng số; $M=1$ tại vị trí giữ nên sai số là sai số làm tròn.) Nhờ vậy các quyết định cắt và làm tròn ảnh hưởng lẫn nhau trong cùng một lượt, khác với OBC (cắt xong mới lượng tử hoá).

## 4. Thí nghiệm và kết quả chính

**Thiết lập:** OPT (125M đến 175B) và BLOOM-176B; mọi lớp linear được cắt đồng đều trừ embedding và head; một A100 80 GB; xử lý các lớp Transformer tuần tự; calibration = **128 đoạn 2048 token** lấy ngẫu nhiên từ shard đầu của C4 (zero-shot thực sự vì không thấy dữ liệu tác vụ); đánh giá perplexity trên raw-WikiText2, PTB, tập con C4, cộng zero-shot (LAMBADA, PIQA, ARC-easy/challenge, StoryCloze). Baseline: magnitude theo lớp; AdaPrune (chỉ đến ~1B).

**Perplexity raw-WikiText2 trên OPT (Bảng 1):**

| OPT, 50% | 125M | 350M | 1.3B |
|---|---|---|---|
| Dense | 27.66 | 22.00 | 14.62 |
| Magnitude | 193. | 97.80 | 1.7e4 |
| AdaPrune | 58.66 | 48.46 | 32.52 |
| SparseGPT | 36.85 | 31.58 | 17.46 |

| OPT | Sparsity | 2.7B | 6.7B | 13B | 30B | 66B | 175B |
|---|---|---|---|---|---|---|---|
| Dense | 0% | 12.47 | 10.86 | 10.13 | 9.56 | 9.34 | 8.35 |
| Magnitude | 50% | 265. | 969. | 1.2e4 | 168. | 4.2e3 | 4.3e4 |
| SparseGPT | 50% | 13.48 | 11.55 | 11.17 | 9.79 | 9.32 | 8.21 |
| SparseGPT | 4:8 | 14.98 | 12.56 | 11.77 | 10.30 | 9.65 | 8.45 |
| SparseGPT | 2:4 | 17.18 | 14.20 | 12.96 | 10.90 | 10.09 | 8.74 |

Ở 175B: perplexity tăng chỉ 0.11 (4:8) và 0.39 (2:4) so với dense (8.35 → 8.45, 8.74). Cắt 50% còn **thấp hơn dense một chút** (8.21 vs 8.35) nhưng paper nói đây là hiện tượng đặc thù tập dữ liệu: trên PTB và C4 thì không (PTB OPT-175B: dense 12.01, SparseGPT 50% 12.37; C4: 10.13 vs 10.36).

**Quy mô lớn:** OPT-175B, magnitude giữ chất lượng chỉ đến khoảng 10% sparsity rồi suy sụp; SparseGPT đạt 60% với tăng perplexity tương đương. BLOOM-176B dễ cho magnitude hơn (đến 30%). Ở 80% sparsity SparseGPT vẫn cho perplexity "hợp lý", magnitude sụp đổ (>100 perplexity) ở 40% (OPT) / 60% (BLOOM). Paper nhấn mạnh hơn 100 tỉ trọng số có thể bỏ. Thời gian: 175B trong khoảng 4 giờ (abstract: dưới 4.5 giờ), trong khi AdaPrune cần ≈1.3 giờ cho 350M và ≈4.3 giờ cho 1.3B.

**Zero-shot OPT-175B (Bảng 2):**

| Phương pháp | Lamb. | PIQA | ARC-e | ARC-c | Story. | Avg. |
|---|---|---|---|---|---|---|
| Dense | 75.59 | 81.07 | 71.04 | 43.94 | 79.82 | 70.29 |
| Magnitude 50% | 00.02 | 54.73 | 28.03 | 25.60 | 47.10 | 31.10 |
| SparseGPT 50% | 78.47 | 80.63 | 70.45 | 43.94 | 79.12 | 70.52 |
| SparseGPT 4:8 | 80.30 | 79.54 | 68.85 | 41.30 | 78.10 | 69.62 |
| SparseGPT 2:4 | 80.92 | 79.54 | 68.77 | 39.25 | 77.08 | 69.11 |

(LAMBADA cao hơn dense là nhiễu của đánh giá zero-shot, theo lời tác giả.)

**Cắt + lượng tử hoá (Hình 6):** 50% thưa + 4-bit có cùng bộ nhớ với 3-bit (lưu giá trị khác 0 + bitmask). OPT-175B: 8.29 (50% + 4-bit) vs 8.68 (GPTQ 3-bit); 2:4 + 4-bit: 8.55; 4:8 + 4-bit: 8.85; từ 2.7B trở lên cấu hình 50%+4bit chính xác hơn 3-bit. Phụ lục: 50% + 3-bit đạt 8.60 trên OPT-175B so với GPTQ 2.5-bit 8.94; 4:8 + 3-bit đạt 8.93.

**Tăng tốc (Phụ lục E):** CPU DeepSparse, OPT-2.7B, 1 batch 400 token: 40%/50%/60% thưa cho tăng tốc 1.57x / 1.82x / 2.16x. GPU 2:4 (CUTLASS so với cuBLAS dense, 2048 token, kích thước ma trận của OPT-175B): Q/K/V/Out 2.84 ms → 1.59 ms (1.79x); FC1 10.26 ms → 6.15 ms (1.67x); FC2 10.23 ms → 6.64 ms (1.54x).

## 5. Phân tích / ablation

- **Mô hình lớn dễ nén hơn** (hiện tượng chính): ở sparsity cố định, độ sụt giảm tương đối thu hẹp theo kích thước; tác giả suy đoán do dư tham số (overparametrization), chưa kiểm chứng.
- **Magnitude sụp đổ trên LLM** dù hiệu quả với CNN nhỏ, nhấn mạnh vai trò của bước bù trọng số.
- **Nhạy cảm theo tầng và theo loại lớp (partial 2:4, Hình 7):** nếu chỉ cắt 2/3 số lớp, **các lớp cuối nhạy hơn lớp đầu**; bỏ qua 1/3 lớp cuối cho độ chính xác tốt nhất. Hệ quả thực tế: một lượt SparseGPT cho phép tạo chuỗi mô hình thưa 2:4 một phần (1/2, 2/3, 3/4...) bằng cách ghép x lớp đầu đã cắt với các lớp sau còn dense. Ví dụ OPT-175B raw-WikiText2: dense 8.34, 1/2 8.22, 2/3 8.38, 3/4 8.49, 4/5 8.52, full 8.74 (Bảng 5).
- **Số mẫu hiệu chuẩn:** làm việc khá tốt với chỉ vài đoạn 2048 token, tốt hơn khi tăng nhưng đường cong phẳng nhanh; chọn 128 mẫu.
- **Dampening:** 0.001 đến 0.1 gần như tương đương; chỉ khi rất lớn mới xấu đi; chọn 1%.
- **Block size chọn mask:** blocksize 1 (từng cột) và rất lớn (4096-8192) đều kém rõ rệt; nhiều giá trị ở mức vài trăm hơi tốt hơn; chọn 128.
- **Độ nhạy theo seed lấy mẫu dữ liệu:** 13.52 ± 0.075 (5 lần), nên khá ổn định (OPT-2.7B, 50%).
- **Chất lượng xấp xỉ so với tái tạo chính xác (A.1, OPT-2.7B 50%):** sai số bình phương theo lớp của SparseGPT chỉ khoảng 20% tệ hơn tái tạo chính xác trung bình (một số ngoại lệ ở out-projection của attention các lớp đầu; ở các lớp FC2 cuối chỉ ~10%).

## 6. Giới hạn

**Tác giả nêu hoặc ngầm nêu:**
- Chỉ cắt trọng số (unstructured / N:M), mô hình lớn nhất vẫn có khoảng cách so với dense ở N:M với mô hình nhỏ ("accuracy loss relative to the dense baseline for smaller models").
- Tăng tốc thực chỉ "preliminary": CPU có DeepSparse, GPU 2:4 chỉ benchmark từng lớp; tăng tốc end-to-end chưa đo ("will likely be slightly lower").
- Giải thích "mô hình lớn dễ nén hơn" chỉ là suy đoán.
- Perplexity là chỉ số nhạy; zero-shot nhiễu.

**Chúng ta nhận xét:**
- Chỉ OPT và BLOOM (kiến trúc GPT cũ), chưa có LLaMA, mô hình lai hay VLM trong paper này.
- Kết quả đẹp nhất ở mô hình rất lớn (66B-175B). Ở mô hình cỡ 125M-1.3B (gần cỡ của dự án) mất chất lượng đáng kể: ví dụ OPT-125M 50%: 27.66 → 36.85; OPT-1.3B: 14.62 → 17.46. Mô hình nhỏ ít dư thừa hơn.
- Thứ tự cột cố định (giữ thứ tự gốc, không hoán vị) là lựa chọn tiện lợi; paper nói bất kỳ hoán vị nào cũng có thể dùng nhưng không khám phá.
- Mỗi lớp được cắt với đầu vào của mô hình đang được cắt tuần tự, nhưng mục tiêu tái tạo vẫn so với đầu ra dense của chính lớp đó, nên sai số lớp trước có thể tích luỹ.
- Calibration chỉ dùng C4 văn bản chung; chưa thử miền hẹp (ví dụ OCR/ảnh).

## 7. Liên hệ với các bài khác

- **01 (OBD/OBS) và 22 (OBC):** SparseGPT = OBS theo hàng (22) + chia sẻ Hessian qua thứ tự cột cố định + chọn mask khối + Cholesky lazy-batch của GPTQ. Công thức (3) giống hệt dẫn xuất ở 22.
- **18 (2:4):** SparseGPT cung cấp cách chọn mask 2:4 tốt hơn magnitude cho LLM không cần retrain; số tăng tốc 1.54-1.79x theo lớp khớp với chặn trên ~2x của bài 18.
- **21 (survey):** phân loại SparseGPT là unstructured và semi-structured pruning; Δ perplexity OPT-175B 50% = -0.14, 2:4 = +0.39 (bảng của survey, khớp với Bảng 1 ở đây).
- **24 (Wanda):** bỏ hẳn bước cập nhật trọng số và Hessian, chỉ dùng |w|·‖x‖; nhanh hơn nhưng không bù. So sánh trực tiếp nên có trong bài 24.
- **27 (LLM Surgeon), 25 (ZipLM):** mở rộng OBS sang nhiều dạng cấu trúc hơn (cắt cấu trúc, không chỉ trọng số lẻ).
- **29 (SliceGPT), 36 (ReplaceMe):** cũng tìm một ánh xạ tuyến tính tối ưu theo bình phương tối thiểu trên dữ liệu hiệu chuẩn để bù cho phần bị cắt.

## 8. Ứng dụng cho dự án (suy luận)

- **Liên quan gián tiếp.** SparseGPT không đạt mục tiêu giảm số tham số hay độ trễ batch-1 xuống 0.3B/0.1B; thưa 50% unstructured không tăng tốc trên GPU thông thường, và 2:4 chỉ có ý nghĩa nếu GPU đích có Sparse Tensor Core (xem 18). Với mô hình nhỏ như 0.8B, mất chất lượng ở 50% có thể đáng kể (xem cảnh báo ở Giới hạn).
- **Phần đáng học là phương pháp bù, gắn D_lin/D_res.** Cập nhật OBS là phép "sửa tuyến tính tối ưu cho các trọng số còn lại để khớp đầu ra dense", tức cùng loại thiệt hại mà một ánh xạ tuyến tính có thể xoá (D_lin của bạn). Phần sai số còn lại sau khi bù (ε trong công thức 3) tương ứng D_res. Gợi ý: khi cắt neuron MLP, thay vì chỉ cắt rồi distill, tính trước **hồi quy bình phương tối thiểu** để cập nhật $W_{down}$ (cùng tinh thần SparseGPT/OBC, áp cho cắt nhóm); đây là suy diễn chưa kiểm.
- **Nhận xét về "cắt MLP neuron hồi phục kém hơn cắt layer":** SparseGPT cho thấy trên LLM, bước bù trọng số quan trọng hơn nhiều so với tiêu chí chọn (magnitude sụp đổ, SparseGPT giữ được). Nếu cắt neuron của dự án đang dùng magnitude/norm không bù thì có thể đang thiếu đúng bước này; xem thêm bài 22 và 29.
- **Gợi ý về lớp nhạy:** phát hiện "các lớp cuối nhạy hơn" (Hình 7) là bằng chứng gián tiếp để tránh cắt đều mọi lớp; ở dự án nên đo độ nhạy từng khối trước khi phân bổ ngân sách cắt.
- **Calibration:** dùng dữ liệu đúng miền OCR (ảnh-văn bản tài liệu) khi tính $H=XX^\top$ cho decoder/ViT.
- **Phần nén cuối (tuỳ chọn):** sau khi đã cắt cấu trúc và distill, SparseGPT 2:4 (+ lượng tử hoá cùng lượt) có thể là bước nén phụ, nếu phần cứng hỗ trợ.

## 9. Thí nghiệm gợi ý

1. Với một khối decoder của dự án, chạy SparseGPT (repo của tác giả) ở 50% và 2:4 trên 128 mẫu OCR so với 128 mẫu văn bản chung (C4/Wikipedia); đo MSE đầu ra khối và CER sau khi cắt, để xem dữ liệu hiệu chuẩn ảnh hưởng thế nào.
2. Cắt 30% neuron của MLP ở 3 chế độ: (a) norm, không bù; (b) norm + bù bằng bình phương tối thiểu cho $W_{down}$; (c) (b) + distill ngắn. So khoảng cách tới dense để tách D_lin khỏi D_res.
3. Quét độ nhạy từng khối: áp 2:4 chỉ cho khối $k$ (cả decoder lẫn ViT) và đo CER, để vẽ đường nhạy cảm giống Hình 7 và dùng làm cơ sở phân bổ ngân sách cắt.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / nội dung |
|---|---|
| Bài toán theo lớp | $\min \|WX-(M\odot\hat W)X\|_2^2$ (NP-hard) |
| Nghiệm với mask cố định | $w_{M_i}=(X_{M_i}X_{M_i}^\top)^{-1}X_{M_i}(w_{M_i}X_{M_i})^\top$ |
| OBS một trọng số | $\delta_m=-\frac{w_m}{[H^{-1}]_{mm}}H^{-1}_{:,m}$, $\varepsilon_m=\frac{w_m^2}{[H^{-1}]_{mm}}$ |
| Chuỗi tập chỉ số | $U_1=\{1..d_{col}\}$, $U_{j+1}=U_j-\{j\}$ |
| Cập nhật nghịch đảo | $(H_{U_{j+1}})^{-1}=(B-\frac{1}{[B]_{11}}B_{:,1}B_{1,:})_{2:,2:}$ |
| Chi phí | $O(d_{col}^3+d_{row}d_{col}^2)$ so với $O(d_{row}d_{col}^3)$ của OBC |
| Chọn mask | Theo khối $B_s=128$ cột, tiêu chí $w^2/[H^{-1}]^2_{cc}$ |
| N:M | $B_s=m$, chọn $n$ trọng số sai số nhỏ nhất mỗi nhóm |
| Kết hợp lượng tử hoá | Sai số $=(W-M\odot\text{quant}(W))/[H^{-1}]_{jj}$ |
| Siêu tham số | 128 mẫu × 2048 token (C4), dampening 1%, $B_s=128$ |
| Kết quả nổi bật | OPT-175B: 50% → 8.21 (dense 8.35), 2:4 → 8.74, 4:8 → 8.45; thời gian ~4 giờ trên A100; 50%+4bit 8.29 vs GPTQ 3-bit 8.68 |

## 11. Kết luận ngắn

SparseGPT biến phương pháp OBS chính xác nhưng đắt của OBC thành thuật toán chạy được trên 175B tham số, bằng cách cho mọi hàng dùng chung một chuỗi Hessian nghịch đảo (cập nhật một phần về bên phải), chọn mask theo khối và dùng Cholesky/lazy-batch của GPTQ. Kết quả: 50-60% thưa với ít mất chất lượng ở mô hình rất lớn, và kết hợp tự nhiên với lượng tử hoá. Với dự án, giá trị nằm ở ý tưởng bù tuyến tính tối ưu sau khi cắt và việc đo độ nhạy theo tầng, hơn là ở dạng thưa unstructured.

Link gốc: https://arxiv.org/abs/2301.00774
