# Học Mechanistic Interpretability qua Transformer Circuits Thread

Thư mục này chứa ghi chú học tập bằng tiếng Việt cho **toàn bộ 53 trang** trên [transformer-circuits.pub](https://transformer-circuits.pub/), lấy nội dung ngày 05/11/2026. Ghi chú được sắp theo **lộ trình cho người mới**. Thứ tự gần như theo thời gian, chỉ đảo vài chỗ để học nền tảng trước.

> Đây là ghi chú diễn giải, không phải bản dịch. Mỗi file có một mục "Ứng dụng giả thuyết cho pruning / distillation". Mục đó là suy luận của chúng ta để nối với dự án (LightOnOCR, D_lin/D_res), **không phải** kết luận của paper. Khi cần trích số liệu hay luận điểm, hãy đọc bản gốc theo link ở đầu mỗi file.

Nếu gặp thuật ngữ lạ, xem trước [`../reports/Giải thích thuật ngữ interpretability.md`](../reports/Giải%20thích%20thuật%20ngữ%20interpretability.md).

---

## Lộ trình

### Giai đoạn 0: Chuẩn bị
| # | File | Nội dung |
|---|---|---|
| 00 | `00_preface_toan_va_huong_dan.md` | Toán nền tảng (đại số tuyến tính, hình học nhiều chiều, KL/softmax, tính thưa, attribution), ký hiệu transformer, cách đọc paper interp, công cụ, lịch học, bài tự kiểm tra có đáp án. **Đọc trước tiên.** |

### Giai đoạn 1: Nền tảng
Học cách "nhìn" một transformer như hệ thống đọc/ghi vào residual stream, và hiểu circuit là gì.

| # | File | Bài gốc | Vì sao đọc |
|---|---|---|---|
| 01 | `01_2022_mech_interp_essay.md` | Mechanistic Interpretability, Variables, and the Importance of Interpretable Bases (2022) | Bài luận ngắn: interp là gì, vì sao cần một hệ trục "dễ hiểu". Đọc đầu tiên. |
| 02 | `02_2021_mathematical_framework.md` | A Mathematical Framework for Transformer Circuits (2021) | Residual stream, QK/OV, path expansion, composition. Toàn bộ phần sau dựa vào bài này. |
| 02b | `02b_2021_mathematical_framework_ghi_chu_ngan.md` | (ghi chú mẫu bạn đã gửi) | Bản tóm tắt ngắn của bài 02, dùng để ôn. |
| 03 | `03_2021_tools_exercises_videos.md` | Exercises, Garcon, Videos (2021) | Bài tập thực hành cho bài 02. |
| 04 | `04_2022_induction_heads.md` | In-context Learning and Induction Heads (2022) | Circuit cụ thể đầu tiên trên model thật: cơ chế "copy" (gần với OCR head). |

### Giai đoạn 2: Superposition (vì sao neuron khó hiểu, và khó cắt)
Đây là giai đoạn quan trọng nhất cho câu hỏi "vì sao cắt neuron khó phục hồi hơn cắt layer".

| # | File | Bài gốc | Vì sao đọc |
|---|---|---|---|
| 05 | `05_2022_toy_models_of_superposition.md` | Toy Models of Superposition (2022) | Model chứa nhiều feature hơn số chiều. Bài trung tâm. |
| 06 | `06_2023_privileged_basis.md` | Privileged Bases in the Transformer Residual Stream (2023) | Vì sao residual stream gần như "xoay tự do" mà vẫn có outlier. Liên quan D_lin. |
| 07 | `07_2022_softmax_linear_units.md` | Softmax Linear Units (2022) | Thử làm neuron đơn nghĩa bằng kiến trúc, và cái giá phải trả. |
| 08 | `08_2023_superposition_double_descent.md` | Superposition, Memorization, and Double Descent (2023) | Superposition của dữ liệu và ghi nhớ. |
| 09 | `09_2023_composition_superposition.md` | Distributed Representations: Composition & Superposition (2023) | Hai cách biểu diễn phân tán, phân biệt rõ ràng. |

### Giai đoạn 3: Dictionary learning (gỡ superposition bằng SAE)
Công cụ để biết model *biết gì*: tách activation thành feature.

| # | File | Bài gốc | Vì sao đọc |
|---|---|---|---|
| 10 | `10_2023_towards_monosemanticity.md` | Towards Monosemanticity (2023) | SAE đầu tiên của Anthropic, có hướng dẫn tự train. |
| 11 | `11_2023_interpretability_dreams.md` | Interpretability Dreams (2023) | Tầm nhìn: sau khi có feature thì làm gì. |
| 12 | `12_2024_scaling_monosemanticity.md` | Scaling Monosemanticity (2024) | SAE trên Claude 3 Sonnet: feature đa ngôn ngữ, đa phương thức, steering. |
| 13 | `13_2024_features_as_classifiers.md` | Using Dictionary Learning Features as Classifiers (2024) | Dùng feature như bộ phân loại: một ứng dụng thực tế. |

### Giai đoạn 4: So model và lần theo mạch (cross-layer, model diffing, attribution graph)
Nhóm công cụ dùng trực tiếp được cho "parent so với student đã prune".

| # | File | Bài gốc | Vì sao đọc |
|---|---|---|---|
| 14 | `14_2024_sparse_crosscoders.md` | Sparse Crosscoders (2024) | Feature xuyên layer, so feature giữa hai model. |
| 15 | `15_2024_stage_wise_model_diffing.md` | Stage-Wise Model Diffing (2024) | So model theo từng giai đoạn train. Áp được cho parent → cắt → distill. |
| 16 | `16_2025_crosscoder_diffing_insights.md` | Insights on Crosscoder Model Diffing (2025) | Cạm bẫy khi dùng crosscoder để diff. |
| 17 | `17_2025_circuit_tracing_methods.md` | Circuit Tracing: Revealing Computational Graphs (2025) | Transcoder, replacement model, error node, attribution graph. |
| 18 | `18_2024_qualitative_research_reflections.md` | Reflections on Qualitative Research (2024) | Phương pháp: làm nghiên cứu "định tính" với model thế nào cho chặt. |

### Giai đoạn 5: Ứng dụng và cơ chế cụ thể
Các phát hiện lớn dựa trên công cụ ở giai đoạn 4, cùng các hướng mới về attention và interference.

| # | File | Bài gốc | Vì sao đọc |
|---|---|---|---|
| 19 | `19_2025_biology_of_an_llm.md` | On the Biology of a Large Language Model (2025) | Một loạt case study bằng attribution graph. |
| 20 | `20_2025_tracing_attention_qk.md` | Tracing Attention Computation Through Feature Interactions (2025) | Giải thích QK (attention nhìn đâu) bằng feature. Liên quan OCR head. |
| 21 | `21_2025_progress_on_attention.md` | Progress on Attention (2025) | Cập nhật hướng nghiên cứu attention. |
| 22 | `22_2025_sparse_mixtures_linear_transforms.md` | Sparse Mixtures of Linear Transforms (2025) | Cách mới mô tả MLP. Liên quan trực tiếp D_lin/D_res. |
| 23 | `23_2025_toy_model_interference_weights.md` | A Toy Model of Interference Weights (2025) | Trọng số "nhiễu" sinh ra từ superposition. |
| 24 | `24_2026_interference_weights_tiny_lm.md` | Characterizing Interference Weights in a Tiny LM (2026) | Kiểm chứng bài 23 trên model ngôn ngữ thật. |
| 25 | `25_2025_toy_model_unfaithfulness.md` | A Toy Model of Mechanistic (Un)Faithfulness (2025) | Khi nào lời giải thích từ replacement model *sai*. |
| 26 | `26_2025_manifolds_counting_linebreaks.md` | When Models Manipulate Manifolds (2025) | Model đếm ký tự và xuống dòng: gần với bố cục văn bản trong OCR. |
| 27 | `27_2025_introspection.md` | Emergent Introspective Awareness in LLMs (2025) | Model có "biết" trạng thái bên trong của nó không. |
| 28 | `28_2026_emotion_concepts.md` | Emotion Concepts and their Function in an LLM (2026) | Tìm, kiểm chứng và can thiệp một nhóm khái niệm trừu tượng. |
| 29 | `29_2026_headvis.md` | HeadVis (2026) | Công cụ trực quan hóa attention head. |
| 30 | `30_2026_natural_language_autoencoders.md` | Natural Language Autoencoders (2026) | Giải thích activation bằng ngôn ngữ tự nhiên, không cần giám sát. |
| 31 | `31_2026_global_workspace.md` | Verbalizable Representations Form a Global Workspace (2026) | Thông tin dùng chung, "nói ra được", nằm ở đâu trong model. |

### Giai đoạn 6: Circuits Updates (đọc dần, tra khi cần)
Các ghi chú ngắn hằng tháng: ý tưởng dở dang, kết quả âm tính, mẹo kỹ thuật. Mỗi năm một file.

| # | File | Nội dung |
|---|---|---|
| 32 | `32_2023_circuits_updates.md` | Tháng 5 và 7/2023 |
| 33 | `33_2024_circuits_updates.md` | Tháng 1, 2, 3, 4, 6, 7, 8, 9/2024 |
| 34 | `34_2025_circuits_updates.md` | Tháng 1, 4, 7, 8, 9, 10, 11/2025 |
| 35 | `35_2026_circuits_updates.md` | Tháng 5, 6, 10/2026 |

---

## Cách đọc gợi ý

- **Đọc kỹ giai đoạn 1–3** theo đúng thứ tự. Mỗi bài dựa vào bài trước.
- **Giai đoạn 4–5** có thể chọn theo nhu cầu. Đọc 17 trước 19, và 23 trước 24.
- Mỗi file có mục **Thí nghiệm gợi ý**. Làm thử 1–2 thí nghiệm nhỏ trên một model nhỏ sẽ hiểu nhanh hơn chỉ đọc.

## Lối tắt cho câu hỏi nghiên cứu của dự án

| Câu hỏi | Đọc |
|---|---|
| Vì sao cắt neuron MLP khó phục hồi hơn cắt layer? | 05 → 06 → 10 → 22 → 23 |
| Làm sao biết student đã prune mất feature gì so với parent? | 10 → 14 → 15 → 16 |
| Làm sao tìm chỗ hỏng bên trong và chọn layer cho hidden-state KD? | 17 → 19 → 25 |
| Mạch đọc / sao chép chữ (OCR) hoạt động thế nào? | 02 → 04 → 20 → 26 |

## Nguồn

Toàn bộ 53 trang liên kết từ trang chủ transformer-circuits.pub đã được tải và trích văn bản. Bản quyền nội dung thuộc tác giả gốc (Anthropic). Ghi chú ở đây chỉ dùng cho mục đích học tập.
