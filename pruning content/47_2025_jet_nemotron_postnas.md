# Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search (2025)

**Tác giả:** Yuxian Gu, Qinghao Hu, Shang Yang, Haocheng Xi, Junyu Chen, Song Han, Han Cai (NVIDIA).  
**Venue / arXiv:** arXiv:2508.15884 (v3, 28/9/2025). Bài không nêu venue.  
**Nguồn:** https://arxiv.org/abs/2508.15884  
**Chủ đề:** **PostNAS**: bắt đầu từ mô hình full-attention đã huấn luyện (Qwen2.5), đóng băng MLP, rồi tìm kiến trúc attention hiệu quả; trọng tâm cho dự án là bước **chọn layer nào giữ full attention, layer nào đổi sang linear attention**.  
**Vị trí trong lộ trình:** Đọc trước: 42 Puzzle, 44 Nemotron-H (hybrid ~8% attention), 31-34 (layer importance). Bài này thêm: học vị trí full-attention bằng super-network + beam search, thay vì rải đều; và phát hiện các attention layer đóng góp rất không đều và cho các năng lực khác nhau. Liên quan trực tiếp tới decoder GatedDeltaNet + full attention của dự án.

> Ghi chú học tập bằng tiếng Việt, không phải bản dịch. Bài này **không phải bài pruning thuần**: nó là kiến trúc/NAS bắt đầu từ mô hình đã huấn luyện, nhưng phép "bỏ hầu hết full-attention layer và thay bằng layer rẻ" là một dạng nén có cấu trúc nên được đưa vào lộ trình. Phần "Ứng dụng cho dự án" là suy luận của chúng tôi.

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
10. [Tóm tắt khái niệm](#10-tóm-tắt-khái-niệm)
11. [Kết luận ngắn](#11-kết-luận-ngắn)

---

## 1. Bối cảnh

- **Full attention** có độ phức tạp $O(n^2)$ và KV cache lớn. **Linear attention** (RWKV7, RetNet, Mamba2, GLA, DeltaNet, Gated DeltaNet...) là $O(n)$, trạng thái nhỏ. Hybrid: giữ **vài** full-attention layer để giữ khả năng tra cứu/ MMLU, còn lại là linear.
- Hai vấn đề của hybrid trước đây: (1) độ chính xác vẫn thua các mô hình full-attention mạnh trên MMLU(-Pro), toán, retrieval, code, long-context; (2) **vị trí** của các full-attention layer thường chọn *đều đặn* (uniform) mà không có cơ sở.
- Huấn luyện kiến trúc mới từ đầu rất đắt, nên bài đề xuất khám phá kiến trúc *trên nền một mô hình đã huấn luyện*.
- Gloss: **NAS** (Neural Architecture Search) = tự động tìm kiến trúc. **Super-network once-for-all** = một mạng chứa nhiều đường đi (mỗi layer có thể là full hoặc linear); huấn luyện một lần rồi trích mạng con. **KV cache** = bộ nhớ key/value của các token cũ.

## 2. Ý tưởng chính

PostNAS bắt đầu từ mô hình full-attention đã huấn luyện (Qwen2.5-1.5B hoặc 3B), **đóng băng toàn bộ MLP**, và chỉ thay/tối ưu khối attention theo bốn bước thô-đến-mịn: (1) học vị trí giữ full-attention (và loại bỏ phần còn lại); (2) chọn khối linear attention tốt nhất (Gated DeltaNet); (3) thiết kế khối mới JetBlock (conv động); (4) tìm siêu tham số theo throughput thật. Với bước (1): huấn luyện super-network mà mỗi layer có thể là full hay linear, rồi dùng beam search để chọn $k$ layer full-attention theo một mục tiêu phụ thuộc tác vụ. Kết quả là chỉ cần **2 layer full-attention trên 28 layer** để đạt độ chính xác ngang Qwen3-1.7B với throughput sinh cao hơn tới 53.6x.

## 3. Phương pháp chi tiết

### 3.1 Học vị trí full-attention (Mục 2.2, phần trọng tâm)

1. Xây **once-for-all super-network**: mô hình pretrained, mỗi layer attention được bổ sung đường linear attention (GLA được dùng cho đơn giản và nhanh hơn một chút; kết quả ổn định khi đổi loại linear attention).
2. Huấn luyện: mỗi bước **lấy mẫu ngẫu nhiên** một đường đi (một tập layer là full, còn lại linear) → mạng con; huấn luyện bằng **feature distillation loss** (khớp đặc trưng với mô hình gốc).
3. Sau huấn luyện, **beam search** chọn vị trí tối ưu cho số full-attention layer cho trước (ví dụ 2). Mục tiêu theo tác vụ:
   - MMLU: chọn cấu hình có loss thấp nhất trên đáp án đúng (tức tối đa hoá $-\text{loss}$);
   - toán và retrieval: chọn cấu hình có accuracy cao nhất.
4. Hình 5(a) trực quan hoá bằng cách cho *từng* layer làm full-attention duy nhất (các layer khác là linear) và chấm điểm.

```text
PostNAS-Placement(pretrained model M, budget k, task T):
  S = M with each attention layer l augmented by linear path L_l
  freeze all MLP weights                       # chỉ học phần attention
  repeat many steps:
      path = random choice per layer in {full_l, linear_l}
      sub  = subnetwork(S, path)
      loss = feature_distill(sub, M)            # student = sub, teacher = M
      update(sub's trainable parts)
  # Tìm kiếm:
  best = beam_search(configs with exactly k full layers, objective = score_T)
  return best                                   # tập layer giữ full-attention
```

### 3.2 Các bước còn lại (tóm tắt)

- **Chọn khối linear** (Bảng 1, đo trên cùng khung): Gated DeltaNet đạt tổng thể tốt nhất, nhờ *data-dependent gating* (quyết định tập trung vào token hiện tại hay trạng thái lịch sử) và *delta rule* (cập nhật trạng thái bằng phần thông tin tăng thêm).
- **JetBlock:** thêm một "kernel generator" sinh **kernel conv nhân quả động** theo đầu vào, áp lên $V$; bỏ conv tĩnh trên $Q,K$. Reduction ratio 8, kích thước kernel 4.
- **Hardware-aware search:** lấy throughput sinh làm mục tiêu thay vì số tham số; giữ **kích thước cache cố định** và lưới tìm trên key dim, value dim, số head (Bảng 2).

## 4. Thí nghiệm và kết quả chính

**Thiết lập:** Jet-Nemotron-2B xây trên Qwen2.5-1.5B; Jet-Nemotron-4B trên Qwen2.5-3B. Huấn luyện hai giai đoạn: giai đoạn 1 đóng băng MLP + distillation, **50B token** (Nemotron-CC + Redstone-QA); giai đoạn 2 huấn luyện toàn mô hình với thêm dữ liệu toán/code, **350B token**. Throughput đo trên H100, context 64K (trừ khi ghi chú).

**Kiến trúc cuối (Bảng 9 trong phụ lục):**

| | Jet-Nemotron-2B | Jet-Nemotron-4B |
|---|---|---|
| Tổng số block | 28 | 36 |
| Full attention layers | No. 15, 20 | No. 18, 21, 22, 28, 33 |
| Sliding window attention (SWA) | No. 21, 22 | No. 17, 20, 23, 24, 26 |
| Hidden / MLP size | 1536 / 8960 | 2048 / 11008 |

Mọi layer còn lại là JetBlock. Lưu ý: Mục 3.1 của chính paper mô tả 4B với full-attention **18, 21, 33** và 7 SWA (6, 17, 20, 22, 23, 26, 28), không khớp với Bảng 9 ở trên. Chúng tôi không biết bảng nào đúng; xem [Giới hạn](#6-giới-hạn). Với 2B hai nguồn khớp nhau.

**Chọn khối linear (Bảng 1, accuracy MMLU / Math / Retrieval / Commonsense):**

| Khối | MMLU | Math | Retrieval | Common. |
|---|---|---|---|---|
| RetNet | 53.6 | 29.9 | 63.7 | 58.1 |
| Mamba2 | 51.5 | 26.0 | 68.9 | 57.5 |
| GLA | 55.8 | 31.2 | 66.6 | 58.5 |
| DeltaNet | 48.9 | 27.4 | 67.9 | 56.6 |
| Gated DeltaNet | 55.6 | 32.3 | 69.3 | 58.7 |
| JetBlock | 56.3 | 32.8 | 69.9 | 58.5 |
| + hardware-aware search | 58.1 | 34.9 | 70.4 | 59.5 |

(RWKV7 bị loại vì throughput huấn luyện thấp: 123 token/s so với ~250.)

**Mô hình cuối so với đối thủ (Bảng 3-8, trích):**

| Mô hình | Throughput (tok/s) | Cache (MB, 64K) | MMLU | MMLU-Pro | BBH | Math avg | Retrieval avg | Code avg |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | 241 | 1792 | 59.5 | 28.9 | 44.1 | 38.4 | 72.4 | 52.0 |
| Qwen3-1.7B-Base | 61 | 7168 | 60.3 | 37.8 | 54.2 | 42.3 | 76.1 | 58.9 |
| Jet-Nemotron-2B | 2885 | 154 | 60.8 | 39.0 | 58.3 | 49.6 | 74.2 | 59.5 |
| Jet-Nemotron-4B | 1271 | 258 | 65.2 | 44.2 | 65.0 | 51.3 | 76.2 | 63.5 |

Hybrid/linear khác (Mamba2-2.7B: MMLU 25.1; Zamba2-1.2B: 43.1; Hymba-1.5B: 49.7) thấp hơn nhiều. Long-context (LongBench, Bảng 8): Jet-Nemotron-2B trung bình 41.1 với chỉ hai full-attention layer, xấp xỉ Qwen2.5-1.5B 39.1 và Gemma3n-E2B 40.4; Jet-Nemotron-4B 43.9.

**Tốc độ:** so với Qwen3-1.7B-Base ở 256K: prefill 6.14x, decode 53.6x; ở 4K decode 15.6x. Cận trên lý thuyết là 56x vì 2 full-attention layer × 2 nhóm KV so với 28 layer × 8 nhóm KV. Trên phần cứng nhỏ (context 64K): Jetson Orin 55.00 vs 6.22 tok/s (8.84x), RTX 3090 684.01 vs 105.18 (6.50x) so với Qwen2.5-1.5B.

**Chi phí PostNAS cho 2B (Bảng 12):** bước học vị trí full-attention: 50B token, 0.8 ZFLOPs, 808 giờ H100; chọn khối linear 3120 giờ; JetBlock 624 giờ; hardware-aware search 5616 giờ; huấn luyện cuối giai đoạn 2 7536 giờ.

## 5. Phân tích / ablation

Bốn "key finding" của paper:

1. **Các attention layer không đóng góp ngang nhau.** Với MMLU chỉ hai layer thật sự quan trọng; với retrieval chỉ hai đến ba layer.
2. **Layer khác nhau phụ trách năng lực khác nhau.** Layer quan trọng cho MMLU không nhất thiết quan trọng cho retrieval.
3. **Toán phức tạp hơn**, nhưng hợp của các layer top cho MMLU và retrieval đã bao phủ phần lớn layer quan trọng cho toán.
4. **Kích thước KV cache quyết định throughput sinh**, không phải số tham số (decode bị giới hạn băng thông bộ nhớ): khi cache cố định, các cấu hình có số tham số khác nhau (1.62B đến 1.98B) cho throughput tương tự (~2950-2980 tok/s ở Bảng 2).

Thêm:
- **Học vị trí tốt hơn rải đều** (Hình 5b): PostNAS vượt uniform về accuracy MMLU; các con số trong hình bị mangle trong bản text nên chúng tôi không trích.
- **Dùng SWA cho tác vụ trắc nghiệm:** tác giả đặt retrieval hướng dẫn chọn full-attention và MMLU hướng dẫn chọn SWA; cho rằng tác vụ trắc nghiệm chủ yếu cần tính chất "khớp mẫu" của softmax và SWA giữ được điều đó (giải thích của tác giả).
- **Tách nguồn gốc dữ liệu (Bảng 14):** khi các baseline được huấn luyện tiếp trên cùng dữ liệu của Jet-Nemotron, Jet-Nemotron-2B vẫn hơn (MMLU 59.6 vs Qwen2.5-continual 56.7, Mamba2 41.0, RWKV7 49.8).
- **PostNAS tổng:** MMLU +5.3, Math +8.4, Retrieval +7.8, Commonsense +3.2 qua bốn bước (Hình 3).
- **So với Falcon-H1 (hybrid theo head):** Jet-Nemotron-2B tương đương Falcon-H1-1.5B-deep nhưng throughput 2885 vs 66 tok/s; paper giải thích do head-wise hybrid phải chạy tuần tự Mamba2 và attention trong cùng layer.

## 6. Giới hạn

*Paper nêu / ngụ ý:*
- Kiến trúc tìm được "có thể không tối ưu nếu huấn luyện từ đầu"; giá trị nằm ở dùng được ngay và làm testbed.
- Hiệu năng prefill ở context ngắn chỉ nhỉnh hơn một chút (1.14x ở 4K); cần kernel JetBlock tối ưu hơn.

*Chúng tôi nhận xét:*
- Vị trí full-attention được chọn theo tác vụ đại diện (retrieval, MMLU); nếu tác vụ đích khác (OCR/tài liệu) có thể cần chạy lại.
- Bảng 9 và Mục 3.1 mâu thuẫn về chỉ số layer của 4B (xem trên).
- Chi phí đáng kể: hàng nghìn giờ H100 cho mỗi bước tìm kiếm; train 350B token. Không phải "nhẹ" cho dự án nhỏ.
- Điều kiện thực nghiệm chỉ trên Qwen2.5 (decoder-only thuần ngôn ngữ), không có VLM.
- Không phải pruning thuần: mô hình cuối được huấn luyện thêm 400B token, không chỉ phục hồi nhẹ.

## 7. Liên hệ với các bài khác

- **44 Nemotron-H:** cũng thấy attention layer không đồng đều (một số attention layer ít quan trọng); ở Nemotron-H attention ~8% rải đều từ đầu, còn ở đây vị trí được học và hạ xuống 2/28 (~7%).
- **45 Minitron-SSM:** thấy attention layer đầu tiên ít quan trọng; bổ sung cho Finding 1.
- **42 Puzzle:** NAS theo khối có điều kiện phần cứng; PostNAS cũng dùng "mạng con từ một mô hình cha".
- **31 ShortGPT, 34 SLEB, 35 LaCo, 36 ReplaceMe:** đo độ quan trọng layer; ở đây thay layer bằng phiên bản rẻ thay vì bỏ hẳn.
- **38 Sheared LLaMA, 41 Flextron:** cũng huấn luyện mạng con từ mô hình lớn.
- **46 Mamba-Shedder:** cùng câu hỏi "tỉ lệ SSM/attention phù hợp?"; Mamba-Shedder là cắt, bài này là thay.

## 8. Ứng dụng cho dự án (suy luận)

- **Liên quan rất cao.** Decoder LightOnOCR-3 là hybrid GatedDeltaNet + full attention, và GatedDeltaNet chính là khối linear mà paper này chọn là tốt nhất trong các khối đã thử (Bảng 1). Bài này đưa ra một cách có hệ thống để quyết định **full-attention layer nào đáng giữ** khi nén xuống 0.3B/0.1B.
- **Cách dùng thực tế:** thay vì huấn luyện super-network (đắt), có thể làm phiên bản rẻ: với mô hình hybrid đã có, lần lượt thay *từng* full-attention layer bằng GatedDeltaNet (hoặc bỏ), đo loss OCR (và loss riêng cho tác vụ cần "tra cứu" như chép lại chuỗi dài), rồi giữ k layer tốt nhất. Đây là suy luận từ Hình 5a, không phải quy trình của paper.
- **Finding 2 cảnh báo:** layer quan trọng cho một năng lực có thể không quan trọng cho năng lực khác; hãy chấm theo tác vụ OCR thực (đọc chữ, bảng, công thức) thay vì perplexity chung.
- **Finding 4 liên quan trực tiếp tới batch-1 latency và ≤ 8 GB:** KV cache là yếu tố chính của throughput sinh ở context dài; với OCR, chuỗi đầu ra dài (trang tài liệu) thì giảm số full-attention layer (và số nhóm KV) có thể có lợi lớn hơn nhiều so với cắt neuron MLP. Chưa kiểm chứng với mô hình của bạn.
- **Nhìn D_lin / D_res:** việc thay full attention bằng một khối tuyến tính khác với cắt layer (không có $D_{lin}$ để hồi quy tuyến tính bù hết); đây là một nhánh thứ ba ngoài "cắt layer" và "cắt neuron", đáng ghi vào danh sách phương án.
- **Chi phí:** nhớ rằng paper dùng 50B + 350B token; dự án của bạn chỉ có distill từ teacher 4B, nên chỉ nên học ý tưởng, không sao chép quy trình.

## 9. Thí nghiệm gợi ý

1. "Heatmap quan trọng" kiểu Hình 5a: với mỗi full-attention layer trong decoder, đo loss khi thay riêng layer đó bằng GatedDeltaNet (có fine-tune ngắn hoặc không), cho từng nhóm tác vụ OCR; tìm layer nào thật sự cần.
2. Giữ k = 1, 2, 3, 4 full-attention layer tốt nhất theo bước 1 so với k rải đều, ở cùng ngân sách distill; báo accuracy OCR và latency batch-1.
3. Đo KV cache và latency theo độ dài chuỗi đầu ra khi giảm số full-attention layer, để kiểm chứng Finding 4 trên GPU 8 GB.

## 10. Tóm tắt khái niệm

| Khái niệm | Ý nghĩa |
|---|---|
| PostNAS | NAS trên mô hình pretrained, MLP đóng băng |
| Super-network once-for-all | Mỗi layer là full hoặc linear; lấy mẫu mạng con khi huấn luyện |
| Mục tiêu tìm kiếm | MMLU: $-\text{loss}$ đáp án đúng; toán/retrieval: accuracy |
| Beam search | Tìm vị trí k full-attention layer tốt nhất |
| Gated DeltaNet | Linear attention với gating phụ thuộc dữ liệu + delta rule |
| JetBlock | Gated DeltaNet + conv động trên V |
| SWA | Sliding-window attention (cửa sổ 1152 ở 2B, 2048 ở 4B) |
| Finding 4 | KV cache, không phải số tham số, quyết định throughput sinh |
| Kết quả 2B | 2 full-attention / 28 layer; 47x throughput so với Qwen3-1.7B (64K) |

## 11. Kết luận ngắn

Trong hybrid, không phải full-attention layer nào cũng đáng giữ, và vị trí quan trọng phụ thuộc năng lực mong muốn. Học vị trí bằng super-network + beam search vượt rải đều; chỉ 2/28 layer full-attention (cùng 2 SWA) là đủ để đạt độ chính xác ngang mô hình full-attention cùng cỡ. Gated DeltaNet là khối linear tốt nhất trong các khối thử. Với dự án, đây là hướng chọn full-attention layer cần giữ, kèm cảnh báo về chi phí và tính phụ thuộc tác vụ.

Link gốc: https://arxiv.org/abs/2508.15884
