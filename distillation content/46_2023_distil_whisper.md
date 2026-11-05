# Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling (2023)

**Tác giả:** Sanchit Gandhi, Patrick von Platen, Alexander M. Rush (Hugging Face).
**Venue / arXiv:** arXiv:2311.00430 (v1, 01/11/2023); bài không nêu venue.
**Nguồn:** https://arxiv.org/abs/2311.00430
**Chủ đề:** Distillation mô hình seq2seq nhận dạng tiếng nói (Whisper) bằng pseudo-label quy mô lớn, lọc theo WER, giữ nguyên encoder và cắt decoder còn 2 lớp; đánh giá robust (OOD, nhiễu, ảo giác trên audio dài); dùng làm assistant cho speculative decoding.
**Vị trí trong lộ trình:** Đọc trước: 15 (Sequence-level KD, pseudo-label), 16 (DistilBERT, khởi tạo bằng cách bỏ lớp + loss KL/CE), 32 (Speculative KD). Bài này bổ sung: một công thức hoàn chỉnh cho bài toán **"transcription dài, gần tất định"**, gần nhất với OCR trong lộ trình. Đọc tiếp: 47 (ShortOPD, phục hồi mô hình bị cắt bằng on-policy distillation).

> Đây là ghi chú học tập bằng tiếng Việt, không phải bản dịch. Số liệu lấy từ paper; phần "Ứng dụng cho dự án" là suy luận của chúng ta về LightOnOCR, không phải kết luận của tác giả.

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

**Whisper** là mô hình nhận dạng tiếng nói (ASR) dạng encoder-decoder (seq2seq transformer), 1.5B tham số ở bản large-v2, huấn luyện trên 680 000 giờ dữ liệu "giám sát yếu" cào từ web. Nó khái quát tốt sang nhiều miền nhưng nặng khi cần độ trễ thấp hoặc phần cứng hạn chế.

Distillation đã thành công cho BERT (DistilBERT, TinyBERT: 16, 18) và seq2seq tóm tắt (DistilBART của Shleifer & Rush). Với ASR thì kết quả cũ kém: distill Wav2Vec 2.0 làm WER tăng 6.9%, HuBERT tăng 7.0%, LAS tăng 2.2% (theo paper). Hai thuật ngữ:

- **WER (Word Error Rate):** số từ sai (thay + chèn + xoá) chia số từ tham chiếu. Tương tự CER ở mức ký tự cho OCR.
- **Pseudo-label (PL):** dùng đầu ra sinh của teacher thay cho nhãn thật để train student (sequence-level KD, xem 15).
- **OOD (out-of-distribution):** dữ liệu từ miền không có trong tập train; tốt trên OOD = "robust".
- **Ảo giác (hallucination) / lặp:** mô hình sinh text không có trong audio hoặc lặp một đoạn nhiều lần; xảy ra nhiều nhất ở audio dài có khoảng lặng.

Điểm khó: distilled model thường tốt trên dữ liệu cùng phân phối (ID) nhưng sụt mạnh trên OOD (Du et al. 2023, theo paper). Cần dữ liệu train đa dạng.

## 2. Ý tưởng chính

Giữ nguyên **toàn bộ encoder** của Whisper (đóng băng), chỉ giảm decoder từ 32 xuống **2 lớp** (khởi tạo từ lớp đầu và lớp cuối của teacher). Dùng teacher sinh pseudo-label cho ~22k giờ audio từ 9 bộ dữ liệu công khai (10 miền), rồi **lọc bỏ** mọi mẫu mà pseudo-label lệch quá ngưỡng WER 10% so với nhãn thật. Huấn luyện student với tổng có trọng số của KL trên logit (word-level) và cross-entropy trên pseudo-label (sequence-level). Kết quả: nhỏ hơn 51% tham số, nhanh 5.8x, WER OOD ngắn chỉ hơn 1 điểm, và ít ảo giác hơn teacher trên audio dài.

## 3. Phương pháp chi tiết

### 3.1 Kiến trúc và khởi tạo

Teacher: encoder $F_{\text{Enc}}$ ánh xạ $X_{1:T}\to H_{1:M}$ ($M\approx T/2$ do conv stem), decoder tự hồi quy dự đoán $P(y_i\mid y_{<i},H_{1:M})$. Student:

- **Encoder:** sao chép nguyên từ teacher và **đóng băng**; nên chỉ chạy forward 1 lần và backprop chỉ qua decoder.
- **Decoder:** "shrink and fine-tune" theo Shleifer & Rush: copy các lớp cách đều nhau của teacher; với 2 lớp thì lấy lớp 1 và lớp 32.

| Mô hình | Lớp enc | Lớp dec | Tham số (M) |
|---|---|---|---|
| distil-medium.en | 24 | 2 | 394 |
| distil-large-v2 | 32 | 2 | 756 |

### 3.2 Hàm mất mát

Pseudo-label: thay nhãn thật $y$ bằng $\hat y_{1:N'}$ do teacher sinh (greedy):

$$
L_{PL}=-\sum_{i=1}^{N'}\log P(\hat y_i\mid \hat y_{<i},H_{1:M}).
$$

(Paper viết công thức thiếu $\log$; chúng ta sửa theo dạng chuẩn.) KL trên toàn phân phối, mỗi vị trí:

$$
L_{KL}=\sum_{i=1}^{N'}\mathrm{KL}(Q_i\,\|\,P_i),
$$

với $Q$ là teacher, $P$ là student, nhiệt độ softmax **2.0** cho cả hai (Phụ lục B.1). Tổng:

$$
\boxed{L_{KD}=\alpha_{KL}L_{KL}+\alpha_{PL}L_{PL},\quad \alpha_{KL}=0.8,\ \alpha_{PL}=1.0}
$$

(theo Shleifer & Rush). Lý do chọn KL: cho thông tin trên mọi lớp và phương sai gradient thấp hơn CE (Hinton).

### 3.3 Lọc pseudo-label bằng ngưỡng WER

Chuẩn hoá cả nhãn thật và pseudo-label bằng bộ chuẩn hoá tiếng Anh của Whisper, tính WER giữa hai chuỗi, **loại mẫu nếu**

$$
\boxed{\mathrm{WER}(\mathrm{norm}(y),\mathrm{norm}(\hat y))>\lambda}
$$

Chọn $\lambda=10\%$ (tune trên tập validation). Lọc bù cho việc dùng greedy thay vì beam search khi gán nhãn (tác giả thấy hai cách khác biệt nhỏ nên chọn greedy cho nhanh).

### 3.4 Transcription dài: chunking

Whisper chỉ nhìn 30 giây. Thuật toán gốc dịch cửa sổ tuần tự theo timestamp, cần beam search và "temperature fallback". Paper dùng cách chunk (Patry 2022): cắt audio thành các đoạn có **chồng lấn nhỏ**, giải mã từng đoạn độc lập, rồi ghép tại vùng chồng bằng **chuỗi con chung dài nhất**. Chỉ cần greedy; vì các đoạn độc lập nên batch song song (nhanh tới 9x).

### 3.5 Speculative decoding

Assistant (Distil-Whisper) sinh vài token ứng viên; teacher kiểm chứng tất cả trong một lần forward; token chỉ được giữ nếu mọi token trước đó khớp dự đoán của teacher. Do đó đầu ra **giống hệt** teacher. Vì dùng chung encoder nên chỉ cần thêm decoder 2 lớp: +8% tham số.

### 3.6 Pseudo-code

```
# 1. Gán nhãn giả (một lần)
for (audio, y_true) in corpus_21k_hours:
    y_hat = teacher.generate(audio, greedy=True)
    if WER(norm(y_true), norm(y_hat)) <= 0.10:
        keep (audio, y_hat)

# 2. Khởi tạo student
student.encoder = copy(teacher.encoder); freeze(student.encoder)
student.decoder = [teacher.dec[0], teacher.dec[-1]]

# 3. Huấn luyện (80k bước, batch 256, 8 epoch)
for audio, y_hat in kept:
    H = student.encoder(audio)              # frozen, chạy forward 1 lần
    logits_s = student.decoder(y_hat, H)
    logits_t = teacher(audio, y_hat)        # teacher-forced trên pseudo-label
    L = 0.8 * KL(softmax(logits_t/2), softmax(logits_s/2)) \
      + 1.0 * CE(logits_s, y_hat)
```

Hyperparameter (Bảng 15): AdamW, $\beta=(0.9,0.999)$, weight decay 0.0, LR đỉnh 1e-4 với 500 bước warmup rồi giảm tuyến tính, max grad norm 1.0, bf16, xác suất giữ timestamp 50%. **Không dùng augmentation hay regularisation** vì chỉ 8 epoch và dữ liệu đa dạng.

## 4. Thí nghiệm và kết quả chính

**Dữ liệu train (Bảng 2), tổng 21 170 giờ, 9 bộ, 10 miền:** People's Speech 12 000 h, GigaSpeech 2 500, Common Voice 13 2 400, Fisher 1 960, LibriSpeech 960, VoxPopuli 540, TED-LIUM 450, SwitchBoard 260, AMI 100. Đánh giá: 11 tập ID + 4 tập OOD ngắn (CHiME-4, Earnings-22, FLEURS, SPGISpeech) + 4 tập OOD dài (Earnings-21, Earnings-22, Meanwhile, Rev16). Latency đo trên A100 40GB, fp16.

**Bảng 5: WER trung bình OOD và độ trễ tương đối so với large-v2** (ngắn: batch 1; dài: chunked, batch 16).

| Mô hình | Tham số (M) | Ngắn: độ trễ | Ngắn: WER | Dài: độ trễ | Dài: WER |
|---|---|---|---|---|---|
| tiny.en | 39 | 6.1 | 18.9 | 5.4 | 18.9 |
| small.en | 244 | 2.6 | 10.8 | 2.2 | 14.7 |
| medium.en | 769 | 1.4 | 9.5 | 1.3 | 12.3 |
| large-v2 | 1550 | 1.0 | 9.1 | 1.0 | 11.7 |
| distil-medium.en | 394 | 6.8 | 11.1 | 8.5 | 12.4 |
| **distil-large-v2** | 756 | 5.8 | 10.1 | 5.8 | **11.6** |

(Chú ý: "độ trễ tương đối" ở đây là hệ số tăng tốc, số lớn hơn là nhanh hơn.)

**Dài:** distil-large-v2 thấp hơn large-v2 0.1 WER. **Bảng 7:** large-v2 tuần tự 10.4 WER; large-v2 chunked 11.7 (nhanh 9.9x); distil-large-v2 chunked 11.6 (nhanh 57.5x so với large-v2 tuần tự).

**Ảo giác (Bảng 8, trung bình 4 tập dài):**

| Mô hình | 5-Dup. | IER | SER | DER | WER |
|---|---|---|---|---|---|
| wav2vec2-large-960h (CTC) | 7971 | 4.8 | 18.9 | 4.6 | 28.3 |
| large-v2 | 23792 | 3.3 | 3.9 | 4.5 | 11.7 |
| distil-large-v2 | 18503 | 2.1 | 5.3 | 4.2 | 11.6 |

(5-Dup. = số 5-gram từ lặp; IER/SER/DER = tỷ lệ chèn/thay/xoá.) Distil-large-v2 có ít lặp hơn 1.3x và IER thấp hơn 1.2 điểm, nhưng SER cao hơn 1.4 điểm. Mô hình CTC chỉ-encoder lặp rất ít (bất lợi chung của seq2seq).

**Nhiễu (Hình 2):** dưới nhiễu tăng dần, WER của Distil-Whisper suy giảm chậm hơn các mô hình train trên LibriSpeech và tiến gần Whisper, nhờ dùng chung encoder đóng băng. Ở SNR 40 dB nhiều mô hình tốt hơn. (Số chi tiết nằm trong hình, chúng ta không có.)

**Speculative decoding (Bảng 9, so với large-v2 không SD = 1.0):** large-v2 + tiny: 2.1x; large-v2 + distil-large-v2: **2.0x** (1669M tham số); medium.en + distil-medium.en: 3.3 (so với 1.4 của medium.en riêng); medium.en + tiny.en: 2.7.

**Effective robustness (Bảng 6, macro-trung bình gồm GigaSpeech):** large-v2 9.4, distil-large-v2 10.2 (+10.7% tương đối); riêng FLEURS 4.2 → 6.3, SPGISpeech 3.8 → 3.3 (cải thiện).

## 5. Phân tích / ablation

**Ngưỡng WER (Bảng 10, distil-large-v2 train 10 000 bước):**

| λ | Dữ liệu bị lọc % | WER ID | WER OOD | WER TB |
|---|---|---|---|---|
| 100 (không lọc) | 0.0 | 14.8 | 9.1 | 13.4 |
| 40 | 11.9 | 13.3 | 7.4 | 12.0 |
| 20 | 24.2 | 13.1 | 7.3 | 11.7 |
| 10 | 45.4 | 12.6 | 7.4 | 11.4 |
| 5 | 60.3 | 12.6 | 7.3 | 11.4 |

Lọc quá lỏng để lọt nhãn sai/ảo giác; lọc quá chặt chỉ giữ mẫu dễ và vứt nhiều dữ liệu. Chọn 10% là thoả hiệp; tác giả nói nếu train lâu hơn thì ngưỡng cao hơn có thể có lợi.

**Lượng dữ liệu (Bảng 11):** WER TB 16.4 (435 h) → 12.0 (3 483 h) → 11.6 (13 933 h) → 11.4 (21 770 h). OOD cải thiện nhanh tới 3 483 h, chậm tới 13 933 h (≈2% dữ liệu pre-train của Whisper), rồi không thêm khi dùng toàn bộ.

**Kích thước student (Bảng 12, 10 000 bước):**

| Enc | Dec | Tham số (M) | Tăng tốc | OOD WER |
|---|---|---|---|---|
| 32 | 32 (teacher) | 1543 | 1.0 | 5.3 |
| 32 | 16 | 1124 | 1.9 | 5.4 |
| 32 | 8 | 914 | 3.0 | 6.0 |
| 32 | 4 | 809 | 4.3 | 6.5 |
| 32 | 2 | 756 | 5.8 | 7.4 |
| 16 | 2 | 441 | 8.6 | 10.5 |

Giảm decoder xuống 16 lớp gần như không mất; xuống 2 lớp mất 2.1 điểm OOD. **Giảm encoder** xuống 16 lớp (giảm thêm 19% tham số) làm OOD WER xấu thêm 3.1 điểm tuyệt đối: encoder sâu rất quan trọng.

## 6. Giới hạn

**Do tác giả nêu:** nhược điểm chính là lặp/ảo giác vẫn còn ở mô hình seq2seq (so với CTC); cần nghiên cứu thêm; distil-medium.en kém hơn trên tập Meanwhile (một người nói, nhiều từ hiếm); SER tăng.

**Chúng ta nhận thấy:**
- Chỉ **tiếng Anh** (bản .en và bộ chuẩn hoá tiếng Anh); đa ngôn ngữ không được thử trong paper này.
- Ngưỡng WER dựa vào nhãn thật; ở tình huống không có nhãn (dữ liệu hoàn toàn chưa gán) bộ lọc này không chạy được.
- Lọc 45% dữ liệu có thể làm lệch về mẫu dễ; paper không phân tích loại mẫu khó bị loại.
- Ablation train chỉ 10 000 bước (2 epoch), chưa chắc phản ánh mô hình 80 000 bước. Paper không có ablation tách riêng KL và PL (không cho biết mỗi thành phần đóng góp bao nhiêu).
- Thuật toán chunking cho audio dài dùng chung cho teacher lẫn student, nên cải thiện ảo giác có thể một phần đến từ pipeline, và Bảng 7 cho thấy chunking làm teacher kém đi 1.3 WER so với tuần tự.
- Hai con số trung bình trong paper (Bảng 5 vs Bảng 6) khác nhau (10.1 vs 10.2 và 9.1 vs 9.4) vì Bảng 6 tính trên bộ tập khác (gồm GigaSpeech).

## 7. Liên hệ với các bài khác

- **15 (Sequence-level KD):** pseudo-label là chính nó; paper nhấn mạnh PL = sequence-level, KL = word-level, và kết hợp hai.
- **16 (DistilBERT):** cùng ý tưởng trọng số hoá KD + data loss, khởi tạo bằng lớp của teacher; Distil-Whisper kế thừa qua Shleifer & Rush.
- **32 (Speculative KD) / 29 (GKD):** dùng student làm drafter; ở đây student căn chỉnh phân phối với teacher (như DistilSpec).
- **45 (Sparse logit sampling):** bài đó nói cách *lưu* KL; Distil-Whisper tính KL online (teacher chạy trên pseudo-label mỗi bước).
- **47 (ShortOPD):** cùng gặp vấn đề lặp ở suffix; ShortOPD phát hiện và rút ngắn rollout lặp, còn Distil-Whisper giảm lặp một phần nhờ pseudo-label sạch + lọc.
- **36 (Distilling step-by-step) / 37 (Orca):** các cách dùng đầu ra teacher làm dữ liệu; so sánh với lọc theo nhãn thật ở đây.

## 8. Ứng dụng cho dự án (suy luận)

Đây là bài **gần nhất** với OCR trong lộ trình: ASR cũng là sinh chuỗi dài, gần tất định từ một encoder đa phương thức, đánh giá bằng WER/CER, bị lặp/ảo giác ở chuỗi dài.

1. **Quy trình pseudo-label + lọc theo CER.** Teacher 4B sinh transcript greedy cho các trang; so với nhãn thật (nếu có) và loại mẫu CER > λ (khởi điểm thử ~10% theo mô hình WER của paper, cần tune lại vì OCR có phân phối lỗi khác). Với trang không có nhãn thật, ta cần một tín hiệu thay thế (ví dụ đồng thuận giữa nhiều lần giải mã, hoặc log-prob thấp); đó là suy luận, paper không đề cập.
2. **Công thức loss làm điểm khởi đầu:** $0.8\cdot\mathrm{KL}_T + 1.0\cdot\mathrm{CE}$ trên pseudo-label, nhiệt độ 2.0. Chúng ta lưu top-20 logprob nên $\mathrm{KL}$ phải tính trên top-20 (+ bin đuôi, xem 45). Nhớ rằng nhiệt độ 2.0 làm phẳng phân phối; với teacher gần tất định, nhiệt độ cao giúp lộ "dark knowledge" ở vị trí mà top-1 ≈ 1.0, nhưng top-20 phải đủ chứa các ứng viên; hãy kiểm tra trên cache của bạn.
3. **Giữ encoder sâu, cắt phần khác.** Bảng 12: cắt encoder (32→16 lớp) làm OOD WER +3.1, còn cắt decoder 32→16 gần như miễn phí. Nếu phép tương tự đúng cho VLM (suy luận, cần kiểm chứng): ViT 12 khối là phần đáng bảo vệ, còn decoder hybrid GatedDeltaNet/attention là nơi cắt trước. Lưu ý Distil-Whisper dùng decoder rất mỏng (2 lớp) nhờ lời giải thường tất định từ encoder; decoder của LightOnOCR còn phải đọc layout dài, nên số lớp tối thiểu có thể khác.
4. **Khởi tạo bằng lớp cách đều + đóng băng encoder** là baseline dễ so sánh với cắt có cấu trúc dựa trên D_lin/D_res: ghi lại khoảng cách giữa "chọn lớp cách đều" và "chọn lớp theo D_res thấp nhất".
5. **Lặp ở chuỗi dài** là lỗi đặc trưng; theo dõi tương tự 5-gram duplicate và IER (chèn) cho OCR (số dòng/ký tự lặp). Nếu student sau phục hồi vẫn lặp, xem 47.
6. **Speculative decoding** có thể là cách rẻ để đạt độ trễ batch-1 mà vẫn giữ đầu ra giống teacher, nhưng cần teacher 4B trong bộ nhớ nên với GPU ≤ 8 GB thì không khả thi nếu chạy cả teacher; chỉ có ích khi student đóng vai drafter của mô hình lớn hơn. Ít liên quan cho triển khai đích của bạn.
7. **Lọc làm lệch về mẫu dễ:** đối với OCR, mẫu khó (bảng, chữ viết tay, scan xấu) có thể bị loại nhiều hơn; hãy theo dõi CER theo loại tài liệu sau lọc.

## 9. Thí nghiệm gợi ý

1. **Quét ngưỡng CER** (không lọc / 20% / 10% / 5%) với student đã cắt, cùng ngân sách bước; vẽ CER trên tập giữ lại theo loại tài liệu (sạch/khó) để thấy hiệu ứng lệch mẫu dễ.
2. **Tách KL và PL:** (CE-only trên pseudo-label), (KL-only top-20), (0.8 KL + 1.0 CE). So CER và tỷ lệ chuỗi lặp trên đầu ra dài.
3. **Cắt đối xứng:** cắt 50% decoder so với cắt ViT cùng tỷ lệ tham số, cùng dữ liệu; xem có lặp lại nhận xét "encoder quan trọng hơn" của Bảng 12 hay không.

## 10. Tóm tắt công thức / khái niệm

| Khái niệm | Công thức / giá trị | Ghi chú |
|---|---|---|
| Pseudo-label loss | $L_{PL}=-\sum\log P(\hat y_i\mid\hat y_{<i},H)$ | sequence-level KD |
| KL loss | $L_{KL}=\sum_i\mathrm{KL}(Q_i\|P_i)$, $T=2.0$ | word-level KD |
| Tổng | $0.8L_{KL}+1.0L_{PL}$ | theo Shleifer & Rush |
| Bộ lọc | bỏ nếu $\mathrm{WER}(y,\hat y)>\lambda$, $\lambda=10\%$ | lọc 45.4% dữ liệu |
| Student | enc copy+đóng băng, dec 2 lớp (lớp 1 và N) | 5.8x nhanh, −51% tham số |
| Chunked long-form | chồng lấn + LCS, greedy, batch | tới 9x nhanh |
| Speculative decoding | assistant đề xuất, teacher kiểm chứng | đầu ra không đổi; 2.0x |
| Dữ liệu | 21 170 h, 9 bộ; lợi ích OOD chậm lại sau ~13 933 h | 80k bước, batch 256 |

## 11. Kết luận ngắn

Distil-Whisper cho thấy một công thức đơn giản đủ mạnh cho seq2seq sinh chuỗi: đóng băng encoder, cắt decoder, pseudo-label từ teacher trên dữ liệu đa dạng, lọc bằng WER 10% và kết hợp KL + CE. Kết quả: nhỏ hơn 51%, nhanh 5.8x, sát teacher trên OOD ngắn (+1 WER), ngang hoặc tốt hơn trên audio dài do ít ảo giác chèn. Với dự án OCR, bài học chính là: giữ encoder, lọc pseudo-label theo nhãn thật, và theo dõi lặp/chèn ở chuỗi dài.

Nguồn: https://arxiv.org/abs/2311.00430
