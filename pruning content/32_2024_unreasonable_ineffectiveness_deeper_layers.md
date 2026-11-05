# The Unreasonable Ineffectiveness of the Deeper Layers (2024)

**Tác giả:** Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, Daniel A. Roberts (Meta FAIR, UMD, Cisco, Zyphra, MIT, Sequoia Capital).  
**Venue / arXiv:** arXiv:2403.17887 (v2, 03/2025); "Published as a conference paper at ICLR 2025".  
**Nguồn:** https://arxiv.org/abs/2403.17887  
**Chủ đề:** Cắt một **khối liền** các layer của LLM dựa trên khoảng cách góc giữa biểu diễn, rồi "heal" (chữa) bằng QLoRA ít token.  
**Vị trí trong lộ trình:** Đọc trước: 31 ShortGPT (cùng ý tưởng, đo từng layer). Bài này thêm: (a) chọn *khối n layer liền kề* thay vì n layer rời; (b) bước healing ngắn; (c) phát hiện "QA benchmark bền nhưng loss next-token xuống dốc đều". Đọc tiếp: 33 Shortened LLaMA (so sánh cách retrain), 34 SLEB, 36 ReplaceMe, 37 (repair đổi layout tối ưu).

> Tài liệu học tập tiếng Việt, ghi chú diễn giải, không phải bản dịch paper. Hầu hết kết quả của paper nằm trong **hình** (không có bảng số trong text), nên phần số liệu dưới đây chủ yếu là ngưỡng/xu hướng mà tác giả viết ra bằng chữ. Ứng dụng cho dự án là suy luận của chúng ta.

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

- Pruning LLM sau pretraining thường cắt theo chiều rộng (head, neuron, kênh). Bài này tự hỏi một câu hỏi "khoa học" hơn: **tri thức được lưu ở đâu trong trọng số?** Nếu bỏ một layer mà benchmark hỏi-đáp không đổi thì layer đó không cần để lưu tri thức cho các câu hỏi ấy.
- **Healing**: sau khi cắt, có "mismatch" ở chỗ nối (layer sau nhận đầu vào khác kỳ vọng). Fine-tune nhẹ để vá chỗ nối này gọi là heal.
- **QLoRA**: lượng tử hoá mô hình xuống 4 bit và chỉ huấn luyện các adapter hạng thấp (LoRA, cộng thêm ma trận $BA$ nhỏ vào trọng số). Nhờ đó mỗi thí nghiệm chạy trên một GPU A100 40GB.
- **Residual stream**: $x^{(\ell+1)}=x^{(\ell)}+f(x^{(\ell)},\theta^{(\ell)})$; đầu ra cuối là tổng của embedding và đóng góp mọi layer.

## 2. Ý tưởng chính

Nếu các biểu diễn thay đổi chậm theo độ sâu ($x^{(\ell)}\approx x^{(\ell-1)}+\epsilon$) thì xoá một khối liên tiếp layer chỉ gây thay đổi nhỏ. Do đó đo **khoảng cách góc** giữa đầu vào layer $\ell$ và đầu vào layer $\ell+n$; chọn $\ell^*$ làm khoảng cách đó nhỏ nhất; xoá n layer từ $\ell^*$; tuỳ chọn heal bằng QLoRA. Kết quả chính: với Llama-2-70B bỏ được "khoảng một nửa" layer trước khi độ chính xác QA sụp xuống ngẫu nhiên. Tuy nhiên, loss next-token xấu đi đều đặn theo tỉ lệ cắt, và các tác vụ suy luận (GSM8K, HellaSwag) hỏng ngay khi cắt chút ít.

## 3. Phương pháp chi tiết

### 3.1 Trực giác (Eq. 1–5)

Nếu xoá layer $\ell-1$ thì layer $\ell$ nhận $x^{(\ell-1)}$ thay vì $x^{(\ell)}$:

$$x^{(\ell+1)}=x^{(\ell-1)}+f\big(x^{(\ell-1)},\theta^{(\ell)}\big).$$

Mismatch này nhỏ nếu biểu diễn hội tụ thành hàm biến thiên chậm. Để cắt n layer trước layer $\ell$ cần $x^{(\ell)}\approx x^{(\ell-n)}+\epsilon$. Ba giả thuyết paper kiểm chứng: (0) cắt được layer của residual net; (1) cắt layer sâu dễ hơn layer nông; (2) khối cắt thành công có đầu ra giống đầu vào.

### 3.2 Khoảng cách góc và chọn khối

$$\boxed{d\big(x^{(\ell)},x^{(\ell+n)}\big)=\frac1\pi\arccos\!\left(\frac{x^{(\ell)}_T\cdot x^{(\ell+n)}_T}{\lVert x^{(\ell)}_T\rVert\,\lVert x^{(\ell+n)}_T\rVert}\right)},\qquad \boxed{\ell^*(n)=\arg\min_\ell\, d\big(x^{(\ell)},x^{(\ell+n)}\big)}$$

$T$ là **token cuối** của chuỗi (vì causal mask nên chỉ embedding này phụ thuộc cả chuỗi). Cộng/trung bình trên nhiều mẫu (heat-map dùng 10k mẫu C4 validation). Paper nói rõ lựa chọn góc thay vì cosine "không quan trọng".

```
Input: model L layers, số layer cần cắt n, tập dữ liệu trung tính (C4)
1. for mỗi ℓ in 0..L-n:  d[ℓ] = mean_samples angular_distance(x[ℓ], x[ℓ+n])  # x[ℓ] = input của layer ℓ
2. ℓ* = argmin d
3. xoá layer ℓ* .. ℓ*+n-1 (nối input của layer ℓ* thẳng vào layer ℓ*+n)
4. (tuỳ chọn) heal: QLoRA trên C4
```

### 3.3 Heuristic đơn giản hơn

Cắt n layer sâu nhất **trừ layer cuối**: bỏ layer $L-n$ đến $L-1$ (không cần chạy model gốc trên GPU). Ý nghĩa: ablation cho việc tối ưu khối.

### 3.4 Healing (Appendix B)

QLoRA 4-bit; LoRA chỉ gắn vào các module **FFN**; dữ liệu C4; 5000 bước, batch toàn cục 16; cosine LR với 100 bước warmup; 164M token (model ≤7B, seq 2048) hoặc 328M token (≥13B, seq 4096). Bảng hyperparameter của paper: Llama-2-7B rank 2, LR 3e-4; Llama-2-13B rank 64; Llama-2-70B rank 8, LR 3e-5; Mistral-7B rank 4, LR 3e-6; Qwen và Phi-2 rank 64. Chọn rank theo quy trình: bắt đầu 64, nếu heal làm xấu hơn không heal thì quét rank.

Gợi ý trong footnote 8 (chưa thử): thêm loss phụ thầy-trò $\mathcal L_{aux}\sim\lVert x^{(\ell^*+n)}(\theta_0)-x^{(\ell^*)}(\theta)\rVert^2$ để ép đầu vào của layer sau khớp với mô hình gốc.

## 4. Thí nghiệm và kết quả chính

**Mô hình:** Llama-2 7B/13B/70B, Qwen-7B/14B, Mistral-7B, Phi-2 (2.7B). **Eval:** MMLU 5-shot, BoolQ 0-shot, loss next-token trên C4 validation (chuẩn hoá bằng $\log V$); sau đó CoT-MMLU, GSM8K, HellaSwag.

Các con số paper nêu bằng chữ (đọc từ đồ thị, không có bảng):

| Mô hình | Tỉ lệ layer cắt thì MMLU sụp về ngẫu nhiên (xấp xỉ) |
|---|---|
| Llama-2 family | khoảng 45%–55% |
| Mistral-7B | khoảng 35% |
| Phi-2 | khoảng 25% |
| Qwen family | khoảng 20% |

- Dạng đường cong: vùng phẳng (gần như không mất) rồi **chuyển pha** sang ngẫu nhiên (25% với MMLU 4 lựa chọn).
- Healing chỉ cải thiện **khiêm tốn** accuracy QA và đẩy điểm sụp sang phải một chút.
- **Loss next-token**: không heal thì cũng sụp gần chỗ QA sụp; heal xong thì loss gần mức chưa cắt và tăng chậm, gần tuyến tính theo tỉ lệ cắt, **liên tục** qua điểm mà QA sụp (đến ít nhất 80% với Llama-2-70B theo chú thích Fig. 1).
- Mô hình lớn/sâu hơn (Llama-2-13B, 70B) bền hơn; tác giả đưa 2 giả thuyết: mô hình nhỏ bị overtrain nên ít dư thừa hơn, hoặc mô hình sâu chịu mất nhiều layer tuyệt đối hơn.
- **Heuristic "cắt layer sâu nhất"**: không heal thì rất tệ; sau heal gần như ngang chiến lược similarity (similarity hơi tốt hơn trước điểm chuyển pha; heuristic có thể đẩy điểm chuyển pha hơi xa hơn). Loss hầu như chồng lên nhau, similarity nhỉnh hơn một chút.
- Kết luận của tác giả: mục đích của fine-tune sau cắt là **vá giao diện**, không phải học thêm tri thức.
- **Reasoning (Llama-2-70B, Fig. 6):** GSM8K và HellaSwag giảm ngay với bất kỳ mức cắt nào; CoT-MMLU có vùng phẳng giống MMLU. MMLU không CoT ~69% so với CoT-MMLU ~43% ở 0% cắt.

## 5. Phân tích / ablation

- **Chuẩn dữ liệu chọn khối:** thông thường dùng C4 (trung tính); có thể dùng dữ liệu đúng tác vụ đích (đề xuất, chưa đánh giá kỹ).
- **LoRA rank (Appendix C.3):** rank thấp hơn thường cải thiện nhẹ; hành vi định tính giống nhau giữa các rank. Có ablation về prompt và seed (Appendix C.1, C.2); text không đủ để trích số.
- **Layer cuối đặc biệt:** khối chứa layer cuối luôn có khoảng cách cực đại hoặc gần cực đại, nên không bao giờ cắt layer cuối. Ngoại lệ: Phi-2 và các khối lớn của Llama-2-7B cho thấy vài layer cuối cũng quan trọng.
- **Qwen "kỳ lạ":** heat-map có các "đảo" tương tự cao ở khối nông, tác giả cho rằng điều này giải thích vùng bền ngắn hơn.
- Giả thuyết tác giả: layer nông lưu/truy xuất tri thức; layer sâu dùng cho tính toán bậc cao (suy luận toán). Đây là **giả thuyết phù hợp dữ liệu**, chưa được chứng minh.

## 6. Giới hạn

**Do tác giả nêu:**
- Chưa biết chiến lược cắt tốt hơn hoặc cách heal tốt hơn; chưa giải thích vì sao heal xoá chuyển pha ở loss nhưng không ở QA.
- Eval mới có QA, một ít suy luận; cần eval toàn diện hơn; chưa rõ mô hình over-trained hay distilled có khó cắt hơn không.
- Dùng QLoRA vì giới hạn tính toán; rank chọn bằng quét hạn chế do ngân sách.

**Do chúng ta nhận thấy:**
- Chỉ tính khoảng cách ở **token cuối**; không rõ độ ổn định với token khác (vd. chuỗi OCR dài).
- Kết quả là đồ thị, không bảng; rất khó tái lập con số chính xác từ text.
- Không đo latency/throughput thật; "heal bằng 164M–328M token C4" khác nhiều so với hồi phục bằng distillation.
- Chỉ LLM Transformer thuần, không có hybrid linear-attention.

## 7. Liên hệ với các bài khác

- **31 ShortGPT:** cùng bản chất (so biểu diễn vào/ra), nhưng ShortGPT cắt layer rời theo BI không heal; bài này cắt khối liền và heal. Cả hai kết luận layer sâu dư.
- **33 Shortened LLaMA:** so sánh kỹ cách retrain (LoRA vs continued pretraining) sau depth pruning, bổ sung bài này.
- **34 SLEB:** chọn lặp từng block bằng perplexity thực, không đo góc.
- **36 ReplaceMe:** thay vì heal bằng LoRA toàn mô hình, tìm ánh xạ tuyến tính đóng ở chỗ nối (dạng đóng, không train).
- **37:** chỉ ra khi có repair tuyến tính thì layout tối ưu chuyển từ rời sang khối liền, nhất quán với lựa chọn "khối liền" ở đây.
- **39–40 Minitron:** đối chiếu depth pruning kết hợp distillation.

## 8. Ứng dụng cho dự án (suy luận)

- **Lựa chọn khối liền** chính là cách đơn giản nhất xác định layer để cắt ở decoder 0.8B của LightOnOCR (và có thể ở ViT 12 block): quét $\ell^*(n)$ cho n=1..k.
- Phát hiện quan trọng cho OCR: loss token tăng đều khi cắt, trong khi benchmark "trả lời một lựa chọn" bền. OCR là sinh chuỗi dài, nên đồng hành với loss/CER chứ không giống MMLU. Đề nghị đánh giá bằng CER/WER, không bằng accuracy kiểu trắc nghiệm.
- Heal bằng C4 không phù hợp: trong dự án nên heal bằng distillation từ teacher 4B trên dữ liệu OCR; footnote 8 của paper (khớp biểu diễn tại chỗ nối) gần với feature-matching distillation.
- Decoder có GatedDeltaNet + full-attention: góc giữa $x^{(\ell)}$ và $x^{(\ell+n)}$ vẫn tính được, nhưng khối cắt có thể phá cấu trúc xen kẽ; cần kiểm.
- Liên hệ D_lin/D_res: khoảng cách góc nhỏ ~ "khối gần đồng nhất", nhưng khối có thể khác đồng nhất mà vẫn gần tuyến tính; bài này không phân biệt (đó là chỗ bài 36 và 37 bổ sung).

## 9. Thí nghiệm gợi ý

1. Vẽ heat-map $d(x^{(\ell)},x^{(\ell+n)})$ cho decoder LightOnOCR, lấy trung bình trên mọi token (không chỉ token cuối) của chuỗi OCR; xem khối nào tối ưu.
2. Cắt khối $\ell^*$ vs khối sâu nhất (trừ layer cuối), rồi heal bằng feature-matching tại điểm nối (Eq. 8) với 1–2k bước; so CER.
3. Đo CER theo số layer cắt n để xem có "chuyển pha" như MMLU không, hay đi xuống mượt như loss C4.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| $d(x^{(\ell)},x^{(\ell+n)})$ | khoảng cách góc (chia $\pi$) giữa input layer $\ell$ và $\ell+n$ |
| $\ell^*(n)$ | điểm bắt đầu khối n layer có khoảng cách nhỏ nhất |
| Healing | QLoRA ngắn (FFN-only LoRA, C4) để vá chỗ nối |
| Simple heuristic | cắt n layer sâu nhất, trừ layer cuối |
| Chuyển pha | QA accuracy phẳng rồi rơi về ngẫu nhiên tại tỉ lệ cắt phụ thuộc mô hình |

## 11. Kết luận ngắn

Cắt một khối liền layer sâu rồi heal nhẹ cho QA khá bền, nhưng loss next-token và suy luận thì không. Bài này là bằng chứng sớm rằng thước đo đánh giá quyết định mức "cắt được bao nhiêu", điều cực kỳ liên quan khi đánh giá OCR bằng CER thay vì trắc nghiệm.

Link gốc: https://arxiv.org/abs/2403.17887
