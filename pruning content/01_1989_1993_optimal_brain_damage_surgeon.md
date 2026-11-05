# Optimal Brain Damage (1989/1990) và Second Order Derivatives for Network Pruning: Optimal Brain Surgeon (1992/1993)

**Tác giả:** (OBD) Yann Le Cun, John S. Denker, Sara A. Solla — AT&T Bell Laboratories. (OBS) Babak Hassibi, David G. Stork — Ricoh California Research Center / Stanford.  
**Venue / arXiv:** Không có trên arXiv. OBD: NeurIPS (NIPS) 1989, in trong volume 2 (trang 598–605, xuất bản 1990). OBS: NeurIPS (NIPS) 1992, trang 164–171 (xuất bản 1993). Hai bài được tải về từ proceedings.neurips.cc và **đọc đầy đủ** để viết ghi chú này.  
**Nguồn:** https://proceedings.neurips.cc/paper_files/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html và https://proceedings.neurips.cc/paper_files/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html  
**Chủ đề:** Pruning dựa trên đạo hàm bậc hai (saliency theo Hessian); gốc rễ của mọi phương pháp "tiêu chí chính xác + bù trọng số".  
**Vị trí trong lộ trình:** Đọc ngay sau `00_preface`. Chưa cần bài nào trước. Bài này đặt nền cho: 02 Han 2015 (magnitude + retrain, phản ứng với "second-order đắt"), 07 Molchanov (Taylor), 22 OBC, 23 SparseGPT, 27 LLM Surgeon (hồi sinh OBS ở quy mô lớn).

> Tài liệu học tập bằng tiếng Việt, là ghi chú diễn giải, không phải bản dịch. Số liệu lấy từ hai bản PDF gốc. Phần "Ứng dụng cho dự án" là suy luận của chúng ta. Một số chỗ OCR của PDF bị lỗi (công thức); các công thức dưới đây được dựng lại theo logic của paper và đối chiếu với phần mô tả bằng lời.

## Mục lục

1. [Bối cảnh](#1-bối-cảnh)
2. [Ý tưởng chính](#2-ý-tưởng-chính)
3. [Phương pháp chi tiết: OBD](#3-phương-pháp-chi-tiết-obd)
4. [Phương pháp chi tiết: OBS](#4-phương-pháp-chi-tiết-obs)
5. [Thí nghiệm và kết quả chính](#5-thí-nghiệm-và-kết-quả-chính)
6. [Phân tích / ablation](#6-phân-tích--ablation)
7. [Giới hạn](#7-giới-hạn)
8. [Liên hệ với các bài khác](#8-liên-hệ-với-các-bài-khác)
9. [Ứng dụng cho dự án (suy luận)](#9-ứng-dụng-cho-dự-án-suy-luận)
10. [Thí nghiệm gợi ý](#10-thí-nghiệm-gợi-ý)
11. [Tóm tắt công thức / khái niệm](#11-tóm-tắt-công-thức--khái-niệm)
12. [Kết luận ngắn](#12-kết-luận-ngắn)

---

## 1. Bối cảnh

Cuối thập niên 1980, mạng nơ-ron nhỏ nhưng dữ liệu cũng ít, nên **overfitting** là vấn đề chính: mạng có quá nhiều trọng số so với số mẫu thì khái quát hóa kém. Giải pháp thống kê cổ điển là **cân bằng sai số huấn luyện với độ phức tạp mô hình** (AIC, MDL, VC-dimension...). Độ phức tạp tự nhiên cho mạng nơ-ron là *số tham số tự do khác 0*.

Vấn đề: nên xóa tham số nào trước? Hai cách làm trước đó mà hai bài đề cập:
- **Magnitude / weight decay không đều**: xóa trọng số nhỏ (Hanson & Pratt, Rumelhart, Chauvin...). OBD gọi đây là giả định "magnitude equals saliency" và nói cần đi xa hơn nó.
- **Gating / skeletonization** (Mozer & Smolensky): thêm hệ số cổng. Cần tinh chỉnh hệ số và làm chậm học.

**Saliency** (độ quan trọng) ở đây được định nghĩa rõ: *mức tăng của hàm mục tiêu khi xóa tham số đó*. Bài toán là tính nó mà không phải xóa thử từng tham số và đánh giá lại.

OBD (1989) cho đáp án bằng đạo hàm bậc hai đường chéo. OBS (1992) chỉ ra giả định đường chéo là sai trên các bài toán thật và đưa ra lời giải đầy đủ với Hessian nghịch đảo, đồng thời **tự điều chỉnh các trọng số còn lại**.

## 2. Ý tưởng chính

Xây một mô hình cục bộ bậc hai của hàm lỗi quanh nghiệm đã huấn luyện, rồi dự đoán bằng giải tích việc xóa một trọng số làm lỗi tăng bao nhiêu. OBD giả sử Hessian là ma trận **đường chéo** nên saliency $=\tfrac12 h_{kk}u_k^2$, xóa các tham số saliency thấp, rồi huấn luyện lại, lặp. OBS không giả sử gì về dạng Hessian: giải bài toán tối ưu có ràng buộc "trọng số $q$ bằng 0" bằng nhân tử Lagrange, thu được cả (i) saliency $w_q^2/(2[H^{-1}]_{qq})$ và (ii) công thức cập nhật mọi trọng số còn lại để bù, **không cần huấn luyện lại bằng gradient**. Khó khăn thực tế của OBS là tính $H^{-1}$, giải quyết bằng một công thức đệ quy duyệt dữ liệu một lần.

---

## 3. Phương pháp chi tiết: OBD

### 3.1 Khai triển Taylor và ba xấp xỉ

Nhiễu loạn $\delta U$ của vector tham số làm hàm mục tiêu $E$ thay đổi:

$$
\delta E=\sum_i g_i\,\delta u_i+\tfrac12\sum_i h_{ii}\,\delta u_i^2+\tfrac12\sum_{i\ne j}h_{ij}\,\delta u_i\delta u_j+O(\|\delta U\|^3),
$$

với $g_i=\partial E/\partial u_i$ và $h_{ij}=\partial^2E/\partial u_i\partial u_j$. Ma trận $H$ rất lớn (paper nêu $6.5\times10^6$ phần tử cho mạng 2600 tham số) nên cần ba xấp xỉ mà paper đặt tên:

| Xấp xỉ | Nội dung | Hệ quả |
|---|---|---|
| **Diagonal** | Bỏ các số hạng chéo $h_{ij}, i\neq j$ | $\delta E$ khi xóa nhiều tham số = tổng $\delta E$ khi xóa từng tham số |
| **Extremal** | Xóa sau khi đã hội tụ → ở cực tiểu địa phương | Bỏ số hạng bậc nhất $g_i\delta u_i$; mọi $h_{ii}\ge0$ |
| **Quadratic** | Hàm mục tiêu gần bậc hai | Bỏ số hạng bậc ba trở lên |

Kết quả:

$$
\boxed{\ \delta E\ \approx\ \tfrac12\sum_i h_{ii}\,\delta u_i^2\ }\qquad\Rightarrow\qquad \boxed{\ s_k=\tfrac12\,h_{kk}\,u_k^2\ }
$$

(xóa tham số $k$ tương ứng $\delta u_k=-u_k$).

### 3.2 Tính đường chéo Hessian bằng lan truyền ngược

Paper dùng hàm MSE; kết quả áp dụng cho mỗi mẫu rồi lấy trung bình. Với mạng có trạng thái $z_i=f(a_i)$, $a_i=\sum_j w_{ij}z_j$, và tham số có thể **chia sẻ** ($w_{ij}=u_k$ cho mọi $(i,j)\in V_k$):

$$
h_{kk}=\sum_{(i,j)\in V_k}\frac{\partial^2E}{\partial w_{ij}^2},\qquad \frac{\partial^2E}{\partial w_{ij}^2}=\frac{\partial^2E}{\partial a_i^2}\,z_j^2 .
$$

Đạo hàm bậc hai theo $a_i$ được lan truyền ngược từng tầng, với điều kiện biên ở tầng ra (công thức (8) trong paper):

$$
\frac{\partial^2E}{\partial a_i^2}=2f'(a_i)^2-2(d_i-z_i)f''(a_i).
$$

Chi phí tính đường chéo **cùng bậc với tính gradient**. Nếu bỏ số hạng thứ hai (chứa $f''$) ta được xấp xỉ Levenberg–Marquardt, đảm bảo ước lượng dương.

### 3.3 Công thức (recipe) của OBD

```
1. Chọn kiến trúc hợp lý
2. Huấn luyện đến khi có nghiệm tốt
3. Tính h_kk cho mọi tham số   (1 lượt lan truyền ngược bậc hai)
4. Tính saliency s_k = h_kk * u_k^2 / 2
5. Sắp xếp theo s_k, xóa (đặt = 0 và đóng băng) các tham số saliency thấp
6. Quay lại bước 2 (huấn luyện lại), lặp
```

Paper nêu các biến thể: giảm dần giá trị thay vì đặt 0, hoặc cho tham số đã xóa được thích nghi lại. "Đơn vị đếm" là *tham số tự do*, không phải kết nối, vì trong mạng ràng buộc một tham số có thể điều khiển nhiều kết nối.

---

## 4. Phương pháp chi tiết: OBS

### 4.1 Bài toán có ràng buộc và lời giải

Xuất phát từ cùng khai triển Taylor: ở cực tiểu, bỏ số hạng bậc nhất và bậc ba trở lên,
$\delta E=\tfrac12\,\delta w^\top H\,\delta w$. Xóa trọng số $w_q$ là ràng buộc $e_q^\top\delta w+w_q=0$ ($e_q$ là vector đơn vị). Bài toán (paper dùng nhãn Eq. 3):

$$
\min_q\Big\{\min_{\delta w}\ \tfrac12\,\delta w^\top H\,\delta w\ \ \text{s.t.}\ \ e_q^\top\delta w+w_q=0\Big\}.
$$

Lagrangian $L=\tfrac12\delta w^\top H\delta w+\lambda(e_q^\top\delta w+w_q)$. Đạo hàm theo $\delta w$ bằng 0 cho $\delta w=-\lambda H^{-1}e_q$; thế vào ràng buộc ra $\lambda=w_q/[H^{-1}]_{qq}$. Do đó:

$$
\boxed{\ \delta w=-\frac{w_q}{[H^{-1}]_{qq}}\,H^{-1}e_q,\qquad L_q=\frac{1}{2}\frac{w_q^2}{[H^{-1}]_{qq}}\ }
$$

$L_q$ là **saliency** của trọng số $q$. Paper nhấn mạnh: không cần $H$ hay $H^{-1}$ chéo; mọi trọng số khác được tính lại bằng $\delta w$; và định nghĩa của OBD là trường hợp đặc biệt khi $H$ chéo (khi đó $[H^{-1}]_{qq}=1/h_{qq}$). Cũng theo paper, magnitude pruning ứng với giả sử $H\propto I$ (đẳng hướng), OBD ứng với $H$ chéo, và FARM (Kung & Hu) giả sử $f$ tuyến tính, chỉ cập nhật trọng số tầng ra-ẩn.

```
OBS procedure
1. Huấn luyện mạng "đủ lớn" đến sai số cực tiểu
2. Tính H^{-1}
3. Tìm q có L_q = w_q^2 / (2 [H^{-1}]_qq) nhỏ nhất.
   Nếu L_q << E: xóa, sang bước 4; ngược lại sang bước 5
4. Cập nhật mọi trọng số bằng δw (Eq. 5). Quay lại bước 2
5. Không xóa thêm được nữa. (Có thể huấn luyện lại)
```

Hình 1 trong paper mô tả bất đẳng thức bậc hai $E(\text{mag})\gtrsim E(\text{OBD})\gtrsim E(\text{OBS})$: sau khi xóa (trước retrain), OBS có sai số thấp nhất vì ngoài việc chọn đúng trọng số còn bù các trọng số khác theo hướng cong thấp.

### 4.2 Tính $H^{-1}$ bằng đệ quy

Với mạng một đầu ra, $F(w,\text{in})$, sai số bình phương trung bình, bỏ số hạng $(t-o)$ gắn với đạo hàm bậc hai của $F$ (phần dư):

$$
H=\frac1P\sum_{k=1}^{P}X^{[k]}X^{[k]\top},\qquad X^{[k]}=\frac{\partial F(w,\text{in}^{[k]})}{\partial w}.
$$

Tức $H$ là **ma trận hiệp phương sai mẫu** của vector gradient $X$ (Eq. 12). Dựng dần: $H_{m+1}=H_m+\tfrac1P X^{[m+1]}X^{[m+1]\top}$, $H_0=\alpha I$. Áp dụng công thức nghịch đảo ma trận (Woodbury / Sherman–Morrison) cho từng số hạng:

$$
H_{m+1}^{-1}=H_m^{-1}-\frac{H_m^{-1}X^{[m+1]}X^{[m+1]\top}H_m^{-1}}{P+X^{[m+1]\top}H_m^{-1}X^{[m+1]}},\qquad H_0^{-1}=\alpha^{-1}I,\ H_P^{-1}=H^{-1}.
$$

Hằng $\alpha$ nhỏ (paper nêu khoảng $10^{-8}\le\alpha\le10^{-4}$) làm $H_0^{-1}$ có nghĩa; việc này tương đương nghịch đảo của $H+\alpha I$, tức một phạt $\alpha\|\delta w\|^2$ ngăn bước nhảy lớn và giúp bỏ số hạng bậc cao hợp lệ; paper nói phương pháp *không nhạy* với $\alpha$. Một lượt duyệt dữ liệu, chi phí $O(P\,n^2)$ với $n$ là số tham số. Trường hợp nhiều đầu ra thì có đệ quy theo cả hai chỉ số.

### 4.3 Vì sao được phép bỏ số hạng $(t-o)$

Paper đưa lý do tính toán và lý do chức năng:
- Tính toán: $H$ đầy đủ thường suy biến, đặc biệt trước khi cắt; xấp xỉ này bảo đảm không suy biến kỳ dị và giữ chi phí $O(Pn^2)$. Trong thống kê đây là cơ sở của *Fisher's method of scoring* (thay Hessian thật bằng kỳ vọng).
- Chức năng: nếu giữ số hạng $(t-o)$, mỗi bước cắt sẽ "đưa nhiễu trở lại" vì phạt cả các thành phần nhiễu. Paper lập luận rằng sau khi cắt ta muốn ở gần giá trị sai số mới, như thể có một tín hiệu dạy hiệu dụng $t^*$ sao cho $(t^*-o)\approx0$.

### 4.4 Ví dụ nhỏ (hai trọng số)

Giả sử $H=\begin{pmatrix}2&1.9\\1.9&2\end{pmatrix}$, $w=(1,\,1)$. Đường chéo bằng nhau nên OBD cho hai saliency bằng nhau ($\tfrac12\cdot2\cdot1=1$). Tính $H^{-1}=\frac1{2^2-1.9^2}\begin{pmatrix}2&-1.9\\-1.9&2\end{pmatrix}\approx\frac1{0.39}\begin{pmatrix}2&-1.9\\-1.9&2\end{pmatrix}$, nên $[H^{-1}]_{11}\approx5.13$ và $L_1=\dfrac{1}{2\cdot5.13}\approx0.097$ — nhỏ hơn rất nhiều so với OBD. Và $\delta w=-\frac{1}{5.13}H^{-1}e_1=(-1,\ +0.95)$: sau khi xóa $w_1$, trọng số $w_2$ tăng gần gấp đôi để bù, vì hai đặc trưng gần như đồng hướng. Đây là hình ảnh trực quan của "off-diagonal quan trọng": OBD đánh giá quá cao cái giá phải trả (nó không thấy khả năng bù).  *(Ví dụ số tự dựng cho minh họa, không có trong paper.)*

---

## 5. Thí nghiệm và kết quả chính

### 5.1 OBD (NIPS 1989)

- Mạng: nhận dạng chữ số viết tay zip code, ban đầu **kết nối thưa với 10^5 kết nối do 2578 tham số tự do điều khiển**; huấn luyện trên khoảng 9300 mẫu train và 3350 mẫu test (chữ số viết tay và chữ số in).
- Hình 1a: xóa theo saliency làm hàm mục tiêu tăng **ít hơn đáng kể** so với xóa theo độ lớn; xóa ngẫu nhiên tệ đến mức không vẽ cùng thang. Hình 1b: dự đoán (tổng saliency) khớp thực tế **đến khoảng 800 tham số bị xóa (~30%)**; sau đó đường cong tách ra vì các số hạng ngoài chéo và số hạng bậc cao trở nên quan trọng.
- Hình 2: sau khi **huấn luyện lại**, sai số train và test hầu như không đổi **khi xóa đến 1500 tham số (60%)**.
- Dùng như công cụ thiết kế: với biểu đồ saliency của mạng 10 000 tham số, xóa nguyên tầng áp chót giảm số tham số **một nửa**; MSE train tăng 10 lần, MSE tổng quát hóa chỉ tăng 50%, và **sai số phân loại 10 lớp trên test thực sự giảm**.
- Kết luận của bài: dùng OBD tương tác giảm số tham số **hệ số 4**, và xóa tự động thêm **hơn hệ số 2**; tốc độ mạng tăng đáng kể, độ chính xác nhận dạng tăng nhẹ. Tác giả nhấn mạnh xuất phát từ mạng hiện đại tốt (state-of-the-art), không phải mạng dở.

### 5.2 OBS (NIPS 1992)

| Thí nghiệm | Kết quả theo paper |
|---|---|
| XOR 2-2-1 (9 trọng số) | Chỉ OBS xóa đúng trọng số trong mọi trường hợp huấn luyện từ các khởi tạo khác nhau; magnitude và OBD đôi khi xóa sai và **huấn luyện lại không cứu được** (không đưa sai số về 0). OBS đạt sai số 0 không cần retrain. |
| MONK's 1 | BPWD (backprop + weight decay, Thrun et al.) 58 trọng số, OBS 14 trọng số, cùng 100%/100% |
| MONK's 2 | BPWD 39, OBS 15, cùng 100%/100% |
| MONK's 3 | BPWD 39, OBS 4, cùng 93.4% train / 97.2% test |
| NETtalk | Khởi đầu 5546 trọng số (test error 5259); OBS xuống 2438, retrain, cắt tiếp còn **1560 trọng số với test error 4701**. Sejnowski & Rosenberg dùng 18 000 trọng số |

Paper tóm tắt: OBS cần chỉ **24%, 38% và 10%** số trọng số của mạng BPWD cho MONK's 1, 2, 3 với cùng hiệu năng (bản tóm tắt ghi giảm 90%, 76%, 62%), *không retrain*. Với MONK's 1, mạng sau OBS đơn giản đến mức có thể khôi phục quy tắc logic sinh dữ liệu.

---

## 6. Phân tích / ablation

- **Hessian không chéo** (OBS): tác giả "ngạc nhiên" nhận thấy Hessian của **mọi bài toán họ thử, kể cả XOR và MONK's, đều rất không chéo**; nhiều phần tử ngoài chéo ngang bằng phần tử chéo. Đây là lời giải thích của họ vì sao OBD đôi khi xóa sai.
- **Magnitude có thể xóa sai trọng số mà retrain không sửa được** (Hình 1, 3, 5): ví dụ XOR, magnitude xóa $v_3$, OBD xóa $u_{22}$, chỉ OBS xóa $u_{23}$ — và chỉ lựa chọn đó cho phép mạng giải được XOR. Họ phản bác ý "mạng sẽ tự sắp xếp lại sau retrain", nhất là khi đã cắt nhiều.
- **Số hạng $(t-o)\to0$** (OBS): lý do chức năng/tính toán ở mục 4.3.
- **OBD: khớp dự đoán đến ~30%** rồi lệch (mục 5.1), cho thấy phạm vi hiệu lực của xấp xỉ chéo + bậc hai.
- **Phụ thuộc retrain**: OBD luôn kèm retrain; OBS "không đòi hỏi retrain (thường chậm) sau khi cắt một trọng số", dù tác giả vẫn nói có thể muốn retrain ở cuối.
- OBS còn có thể tổng quát: xóa nhiều trọng số cùng lúc, miễn trừ bias, chiếu lên mặt phẳng khác ($w_q=$ hằng số), và dùng cross-entropy/KL → ma trận thông tin Fisher thay Hessian (nêu nhưng chi tiết ở bài đang chuẩn bị của họ).

## 7. Giới hạn

**Do tác giả nêu:**
- OBD: tại sao dự đoán lệch sau ~30% (off-diagonal, bậc cao); đếm "số tham số tự do" chỉ là thước đo phức tạp *không hoàn hảo*.
- OBS: tính $H^{-1}$ cho mạng nghìn–triệu tham số trông "bất khả thi" nếu làm trực tiếp; họ dùng đệ quy. OBS không tự có tiêu chí dừng.
- Cả hai đều dựa vào giả thiết đã hội tụ về cực tiểu địa phương.

**Do chúng ta nhận xét (suy luận):**
- Quy mô: thử nghiệm trên mạng 10^3–10^4 tham số. Chi phí $O(Pn^2)$ bộ nhớ và thời gian không nhân được lên triệu hay tỷ tham số — đó là lý do 22–23 phải làm xấp xỉ theo từng hàng/tầng.
- Chỉ cắt *tham số lẻ* (unstructured); không bàn tốc độ trên phần cứng. Trong thời đại đó việc giảm kết nối còn "cải thiện tốc độ" ở mức nhận định.
- MSE và mạng nhỏ; chưa có BatchNorm, residual, attention; Hessian của mạng sâu hiện đại có tính chất khác (đa cực tiểu, phẳng, không hội tụ hẳn).
- So sánh với magnitude dựa trên một mạng; chưa có nhiều seed.
- Dự đoán sai khi loss không còn gần bậc hai (xóa trọng số lớn).

## 8. Liên hệ với các bài khác

- **02 Han 2015**: trích dẫn OBD/OBS, nói tiêu chí bậc hai chính xác hơn magnitude *nhưng cần tính toán bổ sung*, rồi chọn magnitude + retrain lặp. Đây là nước đi thực dụng thắng thế suốt nhiều năm.
- **03 Zhu & Gupta**: nhắc lại OBD/OBS trong related work; chọn magnitude + lịch cắt dần.
- **04 Li et al.**: trích OBD/OBS như tiền đề của "đánh giá trọng số theo saliency".
- **07 Molchanov 2019**: Taylor bậc nhất/bậc hai cho cấu trúc, nối trực tiếp với khai triển ở mục 3.1.
- **22 OBC → 23 SparseGPT → 27 LLM Surgeon**: hồi sinh OBS: xấp xỉ $H=XX^\top$ theo từng tầng (đây cũng là dạng Gauss–Newton bên trên) và cập nhật trọng số còn lại bằng công thức $\delta w$ gần như y hệt; 27 mở rộng bằng xấp xỉ Kronecker (K-FAC) cho cắt có cấu trúc.
- **24 Wanda**: có thể xem là OBD/OBS với Hessian chỉ giữ đường chéo $\propto\|X_j\|^2$ và bỏ bù trọng số (cách diễn giải của người soạn; hãy đọc Wanda để xác nhận).
- **08 Lottery Ticket, 11, 12**: nhìn lại câu hỏi "retrain sau khi xóa sai trọng số có cứu được không?" — OBS nói *không phải lúc nào cũng cứu được*.

## 9. Ứng dụng cho dự án (suy luận)

- **Khuôn mẫu tư duy quan trọng nhất**: chi phí cắt không chỉ là "mất trọng số này", mà là "mất trọng số này *sau khi các phần còn lại đã bù tối ưu*". Điều này tương ứng với tách $D_{lin}$ / $D_{res}$ của dự án: phần bù tuyến tính tối ưu của OBS (cập nhật $\delta w$) chính là loại "sửa" có thể xóa tổn hại $D_{lin}$ mà không cần gradient; phần còn lại là $D_{res}$ mà chỉ KD/fine-tune mới sửa được. (Đây là suy luận ghép khái niệm, chưa phải kết luận của paper.)
- **Quan sát "cắt MLP neuron phục hồi kém hơn cắt layer"** có thể được hiểu theo ngôn ngữ bài này: nếu các neuron MLP có Hessian/hiệp phương sai mạnh ngoài chéo (đặc trưng tương quan), tiêu chí chéo/magnitude chọn sai, còn cắt cả layer ít bị lỗi chọn sai nhưng lại mất cả khối. Cần kiểm chứng bằng thí nghiệm.
- Với mô hình 0.8B của dự án, OBS đầy đủ là không khả thi; hướng khả thi là phiên bản **theo từng tầng** (22–23, 27, 29).
- Việc cắt trực tiếp trên ViT 12 block và decoder GatedDeltaNet lai: các công thức ở đây áp dụng cho bất kỳ khối tuyến tính nào có dữ liệu hiệu chuẩn, nên áp dụng được cho cả hai miền (cần thử).

## 10. Thí nghiệm gợi ý

1. **Kiểm định giả thiết "Hessian không chéo" trên một tầng MLP nhỏ của mô hình**: lấy $X$ là các activation đầu vào tầng `down_proj` trên ~512 mẫu OCR, tính $H=XX^\top$, vẽ tỷ lệ $\|{\rm offdiag}\|_F/\|{\rm diag}\|_F$ và hạng hiệu dụng. Hệ số ngoài chéo lớn dự báo cắt theo magnitude/OBD sẽ kém.
2. **So sánh OBD-giống (saliency chéo) và OBS-giống (có bù)** cho việc cắt $k$ neuron MLP của một block: dùng $H^{-1}$ của $XX^\top+\lambda I$; đo lỗi tái tạo đầu ra của block và sau đó loss KD. Mục tiêu: đo xem bù tuyến tính (OBS) lấy lại bao nhiêu phần trăm tổn hại trước khi fine-tune.
3. **Kiểm tra vùng hiệu lực của dự đoán**: vẽ "tổng saliency dự đoán" so với tăng loss thực khi cắt 5%, 10%, 20%, 30% neuron, giống Hình 1b của OBD, để biết ngưỡng mà xấp xỉ bậc hai hỏng.

## 11. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / ý | Nguồn |
|---|---|---|
| Taylor bậc hai | $\delta E\approx g^\top\delta u+\tfrac12\delta u^\top H\delta u$ | OBD Eq. 1, OBS Eq. 1 |
| Saliency OBD | $s_k=\tfrac12h_{kk}u_k^2$ | OBD |
| Đường chéo Hessian | $h_{kk}=\sum_{(i,j)\in V_k}\frac{\partial^2E}{\partial a_i^2}z_j^2$ | OBD Eq. 5–6 |
| Điều kiện biên | $\frac{\partial^2E}{\partial a_i^2}=2f'^2-2(d_i-z_i)f''$ | OBD Eq. 8 |
| Saliency OBS | $L_q=\dfrac{w_q^2}{2[H^{-1}]_{qq}}$ | OBS Eq. 5 |
| Cập nhật OBS | $\delta w=-\dfrac{w_q}{[H^{-1}]_{qq}}H^{-1}e_q$ | OBS Eq. 5 |
| Hessian ≈ hiệp phương sai | $H=\frac1P\sum_kX^{[k]}X^{[k]\top}$ | OBS Eq. 12 |
| Đệ quy nghịch đảo | $H_{m+1}^{-1}=H_m^{-1}-\dfrac{H_m^{-1}XX^\top H_m^{-1}}{P+X^\top H_m^{-1}X}$ | OBS Eq. 17 |
| Magnitude / OBD / OBS | giả sử $H\propto I$ / $H$ chéo / $H$ tổng quát | OBS mục 7 |

## 12. Kết luận ngắn

OBD thay "to nhỏ = quan trọng" bằng "độ cong nhân bình phương trọng số"; OBS đi tiếp: dùng Hessian đầy đủ để vừa **chọn đúng** vừa **bù lại** các trọng số còn lại. Hai bài đưa ra đúng hai câu hỏi mà toàn bộ lĩnh vực vẫn lặp lại: *tiêu chí nào*, và *phục hồi bằng cách nào*. Điểm yếu là chi phí tính Hessian — thứ mà các bài sau (02 magnitude, 22–23 xấp xỉ theo tầng) đã khắc phục theo hai hướng khác nhau.

**Nguồn gốc:**
- OBD: https://proceedings.neurips.cc/paper_files/paper/1989/hash/6c9882bbac1c7093bd25041881277658-Abstract.html
- OBS: https://proceedings.neurips.cc/paper_files/paper/1992/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html
