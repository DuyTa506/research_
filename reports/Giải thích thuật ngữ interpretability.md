# Interpretability cho người mới: các thuật ngữ trong báo cáo pruning

Tài liệu này giải thích các thuật ngữ trong báo cáo *Diễn giải tổn thương do pruning*. Thứ tự đi từ trực giác đến công cụ đo, và mỗi khái niệm đều có ví dụ trên model OCR của ta (LightOnOCR-3-0.8B). Bạn không cần đọc paper nào để hiểu phần này.

---

## 0. Bức tranh tổng trong một đoạn

Hãy hình dung model là một **băng chuyền** (residual stream). Mỗi token có một vector, ví dụ dài 1024 số, chạy dọc qua các layer. Mỗi layer, gồm attention, GatedDeltaNet và MLP, **đọc** vector đó, tính một ít thông tin rồi **cộng thêm** vào vector. Model không ghi đè, chỉ cộng dồn.

Interpretability hỏi ba câu:
1. Trong vector đó có **những thông tin gì** (feature)?
2. **Thành phần nào** ghi hoặc đọc thông tin đó (neuron, head, layer)?
3. Nếu **bỏ hay thay** thành phần đó thì output đổi thế nào (đo nhân quả)?

Với pruning, ta hỏi đúng ba câu đó cho phần bị cắt: cắt đi thì **mất thông tin gì, ở đâu, và có dựng lại được không**.

---

## 1. Khái niệm nền

**Residual stream (dòng dư).** Vector chạy xuyên model như mô tả ở trên. Công thức mỗi block là `h ← h + Attn(h)`, rồi `h ← h + MLP(h)`. Vì các block chỉ *cộng* vào, nên bỏ một block chỉ làm mất một phần đóng góp, chứ không phá cả vector. Đây là lý do cắt layer thường không chết model.

**Feature (đặc trưng).** Một "khái niệm" model biểu diễn bên trong, ví dụ "đang ở trong ô bảng", "ký tự này là chữ Hán", "từ tiếp theo là chữ hoa". Giả thuyết phổ biến (*linear representation hypothesis*) cho rằng mỗi feature là **một hướng** trong không gian vector. Feature càng "bật" thì vector càng chiếu mạnh lên hướng đó.

**Direction / basis (hướng / hệ trục).** Một vector 1024 chiều có thể đọc theo các trục chuẩn (tọa độ 1, 2, 3…) hoặc theo bất kỳ hệ trục xoay nào khác. Câu hỏi quan trọng là trục nào *có ý nghĩa*. Mục 2 trả lời câu này.

**Activation (kích hoạt).** Giá trị của vector hoặc neuron khi model chạy trên một input cụ thể.

---

## 2. Neuron và feature: chìa khóa để hiểu vì sao cắt neuron khó phục hồi

**Neuron.** Trong MLP, `MLP(h) = W_out · σ(W_in · h)`. Mỗi neuron là một dòng của `W_in` cộng với hàm phi tuyến σ (GELU/SiLU). Cắt neuron (pruning width) là xóa một số dòng của `W_in` cùng các cột tương ứng của `W_out`.

**Monosemantic vs polysemantic.**
- *Monosemantic*: một neuron ứng với một khái niệm.
- *Polysemantic*: một neuron bật cho nhiều khái niệm không liên quan. Anthropic ghi nhận một neuron bật cho "trích dẫn học thuật, hội thoại tiếng Anh, HTTP request, chữ Hàn".

**Superposition (chồng chập).** Model cần biểu diễn **nhiều feature hơn số chiều** nó có. Nó làm được vì đa số feature **thưa**, tức hiếm khi bật cùng lúc. Model xếp các feature theo những hướng *gần vuông góc*, chứ không vuông góc hẳn. Hệ quả:
- Mỗi feature trải ra trên **nhiều neuron**, và mỗi neuron mang **mảnh của nhiều feature**. Trong *Towards Monosemanticity*, Anthropic tách 512 neuron thành hơn 4.000 feature.
- Vì các hướng không vuông góc hẳn, các feature **giao thoa (interference)** với nhau. Model dùng **ngưỡng phi tuyến** (bias âm cộng σ) để lọc nhiễu: feature yếu hơn ngưỡng sẽ bị tắt.

**Ví dụ đồ chơi.** Có 3 feature A, B, C nhưng chỉ có 2 neuron.
- Model đặt A = (1, 0), B = (−0.5, 0.87), C = (−0.5, −0.87), ba hướng cách nhau 120°.
- Vì A, B, C hiếm khi bật cùng lúc, ngưỡng phi tuyến vẫn tách được chúng.
- Giờ **cắt neuron thứ 2**. A còn nguyên, nhưng B và C đều sụp về (−0.5): không phân biệt được nữa, và cường độ còn dưới ngưỡng nên bị tắt.
- **Không ma trận tuyến tính nào ở đầu ra dựng lại được B và C**, vì thông tin phân biệt chúng đã mất *trước* hàm phi tuyến.

**Privileged basis (hệ trục ưu tiên).** Một hệ trục là "ưu tiên" khi các tọa độ của nó có ý nghĩa riêng, thường vì có phép toán tác động **riêng lẻ lên từng tọa độ**. Ví dụ σ áp lên từng neuron, hay Adam chuẩn hóa theo từng chiều.
- **Lớp ẩn của MLP là privileged.** Mỗi neuron là một trục thật vì σ áp lên nó. Cắt neuron là cắt *trong hệ trục mà feature đang chồng lên nhau*.
- **Residual stream về lý thuyết không privileged.** Xoay cả không gian thì model vẫn tương đương. Vì vậy khi bỏ một layer làm vector "lệch trục", **một phép xoay hoặc co giãn tuyến tính** có thể sửa được.
- Thực tế residual stream có một ít ưu tiên: vài chiều có giá trị cực lớn, gọi là **outlier features**, đo bằng **kurtosis** (độ "đuôi nặng" của phân phối). Anthropic nghi nguyên nhân là Adam.

> **Đây là cầu nối với D_lin/D_res của ta.** Cắt layer phần lớn làm lệch trục trong không gian không ưu tiên, nên map tuyến tính sửa được (D_lin lớn). Cắt neuron xóa mảnh feature trong không gian ưu tiên, trước hàm phi tuyến, nên map tuyến tính không sửa được (D_res lớn). Đây vẫn là *giả thuyết*, chưa ai đo trực tiếp.

---

## 3. D_lin và D_res: định nghĩa của ta, viết rõ

Tại vị trí cắt, gọi `h_p` là vector của parent và `h_s` là vector của model đã prune, cùng trên một tập ảnh trang.
- Tìm ma trận `W` sao cho `‖W·h_s − h_p‖²` nhỏ nhất. Đây là **least squares**, có lời giải đóng (closed-form), không cần train.
- **D_lin** là phần tổn thương (KL ở output so với parent) **biến mất** sau khi chèn `W`.
- **D_res** là phần **còn lại** sau khi chèn `W`.

Các thuật ngữ liên quan trong báo cáo:
- **Linear repair / linear patch.** Chèn `W` vào chỗ cắt. LinearPatch, ReplaceMe và Ghosted Layers đều là biến thể của ý này.
- **Symmetric vs anti-symmetric.** Mọi ma trận đều tách được thành `W = S + A`, trong đó `S` đối xứng (thuần co giãn theo các trục) và `A` phản đối xứng (thành phần "xoay"). Ghosted Layers thấy khoảng **47% độ lớn** của `W` tối ưu nằm ở phần xoay. LinearPatch ép `W` phải đối xứng, nên bỏ sót phần đó và chỉ gỡ được một phần gap.
- **Frobenius norm.** "Độ lớn" của một ma trận, bằng căn của tổng bình phương mọi phần tử.
- **Linearity score (Razzhigaev et al.).** Đo mức hai layer liên tiếp liên hệ với nhau bằng một phép biến đổi tuyến tính, sau khi chuẩn hóa (dùng **Procrustes**: tìm phép xoay khớp hai tập điểm tốt nhất). Điểm ≈ 0.99 nghĩa là gần như tuyến tính hoàn toàn.

---

## 4. Đo "giống parent đến đâu": cosine, CKA, SVCCA, stitching

Các thước đo này so biểu diễn của student với parent trên cùng tập input.

- **Cosine similarity.** Góc giữa hai vector, 1 là cùng hướng. Đơn giản, nhưng nhạy với phép xoay: student có thể chứa đủ thông tin mà chỉ bị xoay trục, cosine vẫn thấp.
- **CKA (Centered Kernel Alignment).** Thay vì so vector, CKA so *cấu trúc tương đồng*: nếu với parent ảnh 1 giống ảnh 2 hơn ảnh 3, thì với student có như vậy không. CKA **không đổi khi xoay hoặc co giãn đều**, cho giá trị từ 0 đến 1.
  - *Cạm bẫy*: CKA bị chi phối bởi các hướng có phương sai lớn. Một vài hướng nhỏ nhưng quyết định, như hướng phân biệt "ố" với "ô", có thể bị phá mà CKA vẫn cao. Shi et al. (2026) thấy CKA các layer sâu vẫn cao dù model đã sụp. **Vì vậy đừng dùng CKA làm thước đo tổn thương.**
- **SVCCA.** Rút gọn chiều bằng SVD, rồi tìm các cặp hướng tương quan mạnh nhất giữa hai model (CCA). Cùng họ với CKA.
- **Model stitching (khâu model).** Lấy nửa đầu model A nối vào nửa sau model B qua một layer tuyến tính học được. Nếu nối mà vẫn chạy tốt, hai biểu diễn "tương đương tuyến tính". Thực chất đây chính là phép đo D_lin của ta, nhìn từ góc khác. Bansal et al. (2021) cho thấy stitching bắt được khác biệt mà CKA bỏ sót.

---

## 5. Tách feature ra khỏi neuron: SAE và các biến thể

Vì neuron polysemantic, muốn biết model *biết gì* thì phải "gỡ chồng chập". Công cụ chính là **SAE**.

**SAE (Sparse Autoencoder, autoencoder thưa).**
- Lấy activation `x` (ví dụ residual stream ở layer 10, 1024 chiều). Huấn luyện một mạng nhỏ:
  - encoder `f = ReLU(W_enc·x + b)`, với `f` **rất rộng** (ví dụ 16.000 chiều) nhưng **thưa**: mỗi token chỉ vài chục phần tử khác 0;
  - decoder `x̂ = W_dec·f`, cố tái tạo lại `x`.
- Loss = sai số tái tạo + phạt độ thưa (L1, hoặc giữ đúng top-k phần tử).
- Mỗi cột của `W_dec` là **một feature**: một hướng có nghĩa, thường đơn nghĩa và dễ đặt tên.
- Các thông số hay gặp:
  - **dictionary size**: số feature, ví dụ 1M, 4M, 34M ở *Scaling Monosemanticity*;
  - **variance explained / reconstruction loss**: SAE giữ lại được bao nhiêu thông tin;
  - **dead features**: feature không bao giờ bật;
  - **feature splitting**: tăng dictionary thì một feature tách thành nhiều feature tinh hơn, ví dụ "chữ số" tách thành "chữ số trong bảng" và "chữ số trong ngày tháng".
- **Dùng cho ta.** Train SAE trên parent. Đưa activation của student qua *cùng* SAE đó và xem feature nào còn bật đúng chỗ. Làm lần nữa sau khi áp `W` của D_lin: feature hồi lại là tổn thương tuyến tính, feature vẫn mất là tổn thương thật.

**Transcoder.** Giống SAE nhưng **thay thế cả MLP**: đọc input của MLP và dự đoán output của MLP qua một lớp feature thưa. Nó cho thấy MLP *làm gì* (feature vào → feature ra), không chỉ *chứa gì*.

**Crosscoder.** Một từ điển feature dùng chung cho **nhiều layer** hoặc **nhiều model**. Mỗi feature có phần đọc/ghi riêng ở từng nơi.
- **Model diffing (so model).** Train crosscoder chung cho hai model, ví dụ parent và student, rồi phân loại feature: **chung** (cả hai có), **chỉ parent có** (student đã *mất*), **chỉ student có** (student *học mới* khi distill). Đây là công cụ trực tiếp nhất để trả lời "pruning làm mất feature nào".
- **DFC (Dedicated Feature Crosscoder).** Biến thể của Anthropic có vùng từ điển riêng cho từng model, so được cả hai model *khác kiến trúc*.
- **Stage-wise diffing.** So theo từng giai đoạn: parent → vừa cắt → sau distill. Nhạy hơn với thay đổi nhỏ.

---

## 6. Đo nhân quả: ablation và patching

Các thước đo ở mục 4 và 5 chỉ là *tương quan* ("có thông tin"). Muốn biết thành phần nào *gây ra* output thì phải can thiệp.

- **Ablation (xóa thử).** Tắt một thành phần (đặt về 0 hoặc về giá trị trung bình) rồi xem output tệ đi bao nhiêu. Zero-shot pruning damage của ta chính là một dạng ablation.
- **Activation patching (vá activation).**
  1. Chạy model trên input "sạch" và lưu activation.
  2. Chạy trên input "hỏng" (ví dụ đổi một chữ trên ảnh).
  3. Ở một vị trí duy nhất, **thay** activation hỏng bằng activation sạch.
  4. Output hồi lại bao nhiêu thì vị trí đó quan trọng bấy nhiêu.

  Thay vì "tắt", ta "trả lại giá trị đúng", nên định vị chính xác hơn ablation.
- **Cross-model patching (vá giữa hai model).** Cho ta, "sạch" là **parent** và "hỏng" là **student đã prune**. Chép activation của parent vào student tại layer k và đo KL hồi lại bao nhiêu. Nếu hai model khác số chiều thì cần một adapter, và map `W` của D_lin dùng làm adapter được luôn. Với GatedDeltaNet, ngoài activation còn có **recurrent state** (bộ nhớ trạng thái chạy dọc chuỗi), cũng cần thử vá riêng.
- **Path patching.** Chỉ vá *một đường đi* (A → B) thay vì cả activation, để biết thông tin chảy qua đường nào.
- **Attribution patching.** Patching thật tốn một lượt chạy cho mỗi vị trí. Attribution patching **xấp xỉ** kết quả bằng gradient (khai triển Taylor bậc 1), nên có điểm cho mọi vị trí chỉ với một forward và một backward. Rẻ nhưng kém chính xác khi thay đổi lớn.
- **Causal tracing (ROME).** Một dạng patching dùng để tìm *nơi lưu fact*, ví dụ "Paris là thủ đô của…" nằm ở MLP tầng giữa.
- **Cảnh báo từ báo cáo.** Vị trí patching hồi được nhiều nhất **chưa chắc** là nơi sửa hiệu quả nhất (2609.01587, quantization). Phải đo lợi ích sửa thực tế, ví dụ đặt hidden-state KD ở đó rồi xem KL giảm bao nhiêu.

**Circuit tracing / attribution graph (Anthropic, 2025).**
- Thay mọi MLP bằng transcoder để được một **replacement model** dễ đọc, rồi vẽ **đồ thị** feature nào góp vào feature nào, đến tận token output.
- Phần MLP mà transcoder không tái tạo được được giữ lại dưới dạng **error node**. Error node lớn nghĩa là phương pháp chưa giải thích được phần đó.
- Ý tưởng cho ta: so tổng "khối lượng" error node giữa parent và student để xem phần không giải thích được tăng lên ở đâu. Có thư viện mở `circuit-tracer`, hỗ trợ sẵn Qwen3 0.6B–14B.

**Logit lens / tuned lens.**
- **Logit lens**: lấy residual stream ở layer giữa, đưa thẳng qua lớp unembedding cuối để xem model "đang nghĩ" token gì ở độ sâu đó.
- **Tuned lens**: học thêm một map tuyến tính (affine) cho mỗi layer trước khi unembed, nên chính xác hơn.
- Cho ta: biết ở layer nào ký tự đầu ra đã "chốt". Báo cáo dẫn kết quả thông tin hình ảnh chuyển thành chữ ở khoảng 47% độ sâu.

---

## 7. Vai trò của từng loại thành phần

**MLP như "bộ nhớ key–value" (Geva 2021).**
- Dòng của `W_in` là các **key** (mẫu nhận diện, ví dụ "chuỗi trông như số điện thoại").
- Cột tương ứng của `W_out` là **value** (thông tin ghi vào residual stream khi key khớp).
- Cắt neuron là xóa cặp key–value, tức xóa *tri thức*.
- **Knowledge neurons** là các neuron gắn với một fact cụ thể. Ta không nhất thiết cần khái niệm này.

**Attention head như "bộ định tuyến / sao chép".** Attention quyết định *lấy thông tin từ token nào*.
- **Induction head**: thấy "A B … A" thì đoán "B" tiếp theo, tức là sao chép mẫu đã gặp.
- **Retrieval / copy head**: chép nguyên một token từ ngữ cảnh ra output. Chiếm dưới 5% số head nhưng thiếu chúng thì tìm kim đáy bể sụp.
- **OCR head**: head mà token đầu ra *nhìn đúng vào patch ảnh chứa chữ đang đọc*. Theo báo cáo, các head này trùng khoảng 73% với copy head. Đọc chữ chủ yếu là "chép từ ảnh".
- **Attention sink**: một vài token (token đầu, hoặc vài token ảnh) hút rất nhiều attention mà không mang nội dung, giống chỗ "đỗ" attention khi head không cần lấy gì. Cắt nhầm có thể làm hỏng nhiều head.

**GatedDeltaNet vs full attention.** GatedDeltaNet nén toàn bộ quá khứ vào một **recurrent state** kích thước cố định, nên chép chính xác một chi tiết xa kém hơn. Full attention nhìn lại được từng token. Bằng chứng từ các model hybrid cho thấy khả năng sao chép và truy xuất nằm chủ yếu ở các layer full-attention. Vì vậy báo cáo *suy luận* (chưa kiểm chứng) rằng mạch OCR của ta nằm trong khoảng 6 layer full-attention.

---

## 8. "Mất gì" khi nén: long tail

- **PIE / CIE (Hooker et al.).** *Pruning/Compression-Identified Exemplars* là các mẫu mà model nén và model gốc trả lời khác nhau. Accuracy toàn tập gần như không đổi, nhưng trên nhóm mẫu này accuracy sụp, ví dụ 76.75 → 39.81 ở 90% sparsity.
- **Long tail (đuôi dài).** Các trường hợp hiếm. Với OCR, đó là font lạ, chữ không phải Latin, ký hiệu toán, ô bảng phức tạp. Nén thường hại đuôi trước, trong khi số trung bình che mất điều đó.
- **Phản chứng.** Một paper 2026 thấy **feature SAE hiếm lại sống tốt hơn** dưới weight pruning. Câu "nén làm mất cái hiếm" chưa phải luật chung, ta phải tự đo.
- **Loss sensitivity.** Loss tăng bao nhiêu nếu bỏ một channel. "1% channel giữ 58.7% sensitivity" nghĩa là thiệt hại tập trung vào một nhóm rất nhỏ "siêu channel".
- **Wanda.** Tiêu chí prune phổ biến: |trọng số| × độ lớn activation đầu vào. Là baseline để so.

---

## 9. Dùng interpretability để prune hoặc phục hồi tốt hơn

- **LRP (Layer-wise Relevance Propagation) / "Pruning by Explaining".** Lan truyền ngược "độ liên quan" từ output về từng neuron, rồi cắt neuron ít liên quan. Mới có kết quả rõ trên CNN/ViT.
- **Edge pruning.** Học mask trên các *cạnh* giữa các thành phần để tìm **mạch (circuit)** nhỏ nhất vẫn làm được một việc. Dùng để *tìm hiểu*, không phải để nén.
- **Subspace-preserving pruning (SliceGPT…).** Xoay residual stream theo PCA rồi cắt các hướng ít phương sai. Dễ hỏng khi có outlier features (mục 2).
- **Feature-preserving pruning.** Chọn neuron để giữ sao cho giữ được nhiều feature SAE quan trọng nhất, thay vì chấm điểm từng neuron riêng lẻ. Đây là ý tưởng, chưa có paper đo số.
- **Hidden-state KD.** Bắt activation giữa của student khớp với parent, không chỉ khớp output. Interpretability giúp chọn *khớp ở layer nào*.
- **Causal distillation / IIT (interchange intervention training).** Train student sao cho *khi tráo activation giữa hai input* thì student phản ứng giống teacher. Tức là distill cả "cấu trúc nhân quả", không chỉ output.

---

## 10. Bảng tra nhanh

| Thuật ngữ | Một câu | Với model OCR của ta |
|---|---|---|
| Residual stream | Vector chạy xuyên model, mỗi layer cộng thêm vào | Nơi đặt map `W` của D_lin |
| Feature | Một khái niệm = một hướng trong vector | "đang ở ô bảng", "chữ Hán" |
| Polysemantic | Một neuron mang nhiều khái niệm | Lý do cắt neuron phá nhiều thứ một lúc |
| Superposition | Nhiều feature hơn số chiều, xếp gần vuông góc | Giải thích ứng viên cho D_res lớn khi cắt neuron |
| Privileged basis | Trục có nghĩa vì phép toán tác động theo từng tọa độ | Lớp ẩn MLP có; residual stream gần như không |
| Kurtosis / outlier | Vài chiều có giá trị cực lớn | Kiểm tra trước khi prune kiểu xoay |
| D_lin / D_res | Tổn thương sửa được / không sửa được bằng map tuyến tính | Phép đo mới của ta |
| Linearity score | Hai layer liên hệ tuyến tính đến đâu | ≈ 0.99, nên cắt layer dễ vá |
| CKA / SVCCA | So cấu trúc tương đồng, không đổi khi xoay | Không dùng làm thước đo tổn thương |
| Model stitching | Nối hai nửa model bằng layer tuyến tính | Cùng bản chất với D_lin |
| SAE | Autoencoder thưa tách activation thành feature | Train trên parent, kiểm tra feature trên student |
| Transcoder | SAE thay thế cả MLP | Thấy MLP biến feature nào thành feature nào |
| Crosscoder / model diffing | Từ điển chung cho hai model, tìm feature chung/riêng | "Student mất feature gì sau khi cắt?" |
| Ablation | Tắt một thành phần, xem output | Chính là zero-shot damage |
| Activation patching | Thay activation bằng giá trị "đúng", xem hồi bao nhiêu | Vá parent → student để tìm chỗ hỏng |
| Attribution patching | Xấp xỉ patching bằng gradient | Quét nhanh mọi vị trí |
| Attribution graph / error node | Đồ thị luồng feature; phần chưa giải thích được | So mức "không giải thích được" giữa parent và student |
| Logit / tuned lens | Đọc ra token từ layer giữa | Ký tự "chốt" ở độ sâu nào |
| Key–value memory | MLP: key nhận mẫu, value ghi thông tin | Cắt neuron là xóa tri thức |
| Retrieval / copy head | Head chép token từ ngữ cảnh | Gần trùng với OCR head |
| OCR head | Head nhìn đúng patch ảnh đang đọc | Phải bảo vệ khi prune |
| Attention sink | Token hút attention nhưng không mang nội dung | Đừng cắt nhầm |
| PIE / long tail | Mẫu hiếm mà model nén trả lời sai | Đo CER theo script / font hiếm |
| LRP | Chấm "độ liên quan" từng neuron từ output ngược lại | Một tiêu chí prune thay Wanda |
| IIT / causal distillation | Distill cả cấu trúc nhân quả | Một biến thể hidden-state KD |

---

## 11. Bốn phép đo đề xuất, đọc lại bằng ngôn ngữ thường

1. **Tách D_lin/D_res theo loại cắt và theo nhóm chữ hiếm.** Chèn `W` least squares vào chỗ cắt, xem KL giảm bao nhiêu, rồi tách kết quả theo Latin, chữ Hán, toán, bảng.
   - Nếu cắt layer chủ yếu là D_lin: khởi tạo student bằng `W` trước khi distill.
   - Nếu cắt neuron chủ yếu là D_res, nhất là trên chữ hiếm: dùng hidden-state KD và tăng dữ liệu hiếm.
2. **Feature nào của parent còn sống.** Train SAE (hoặc crosscoder) trên parent, đưa student qua, đếm feature mất trước và sau khi áp `W`. Kết quả này cho biết nên giữ neuron nào (giữ neuron mang nhiều feature quan trọng) hoặc thêm loss khớp feature.
3. **Vá parent → student theo từng layer.** Tìm layer mà "trả lại giá trị đúng" hồi KL nhiều nhất. Đặt hidden-state KD ở đó, rồi kiểm tra lại bằng lợi ích thật.
4. **Tìm mạch đọc chữ trên teacher.** Lần lượt tắt từng head và từng layer, đo CER (tỉ lệ lỗi ký tự). Head hoặc layer làm CER tăng mạnh là "vùng cấm cắt". Nếu ViT phải nhỏ lại thì cân nhắc tăng độ phân giải ảnh để bù.

Nếu cần, tôi có thể viết code mẫu (PyTorch) cho phép đo 1 hoặc 3. Hai phép đo này rẻ nhất và nối trực tiếp với giả thuyết D_lin/D_res.
