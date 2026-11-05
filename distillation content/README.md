# Học Knowledge Distillation từ đầu: lộ trình 48 bài

Thư mục này chứa ghi chú học tập bằng tiếng Việt về **knowledge distillation (KD)**, từ ý tưởng gốc (2006–2015) đến KD cho LLM, VLM và phục hồi model đã prune (2026). Thứ tự được sắp cho **người chưa biết gì về domain**.

Mọi paper đều được đọc từ **full text trên arXiv** (PDF, tải ngày 05/11/2026). ID của từng bài đã được đối chiếu với tiêu đề.

> Đây là ghi chú diễn giải, không phải bản dịch. Mục "Ứng dụng cho dự án" trong mỗi file là suy luận của chúng ta cho recipe phục hồi LightOnOCR (CE + forward KL, top-20 logprob từ teacher 4B, output OCR dài và gần tất định), **không phải** kết luận của paper.

---

## Giai đoạn 0: Chuẩn bị
| # | File | Nội dung |
|---|---|---|
| 00 | `00_preface_toan_va_huong_dan.md` | KD là gì, phân loại, toán (temperature, hệ số T², forward/reverse KL, f-divergence, feature loss, exposure bias, estimator top-k), cạm bẫy, bảng thuật ngữ, câu tự kiểm tra. **Đọc trước tiên.** |

## Giai đoạn 1: Nền tảng
| # | File | Bài | Ý chính |
|---|---|---|---|
| 01 | `01_2014_do_deep_nets_need_to_be_deep.md` | Ba & Caruana | Model nông học theo logit của model sâu (gốc từ Model Compression 2006). |
| 02 | `02_2015_distilling_knowledge_hinton.md` | Hinton et al. | Soft target với temperature: định nghĩa KD kinh điển. |
| 03 | `03_2015_fitnets.md` | FitNets | Học theo **hidden state** (hint) của teacher. |
| 04 | `04_2017_attention_transfer.md` | Attention Transfer | Học theo bản đồ attention / kích hoạt. |
| 05 | `05_2021_kd_survey.md` | KD: A Survey | Bản đồ toàn cảnh KD. |

## Giai đoạn 2: Hiểu KD (vì sao, khi nào nó hoạt động)
| # | File | Bài | Ý chính |
|---|---|---|---|
| 06 | `06_2018_born_again_networks.md` | Born-Again Networks | Student cùng kích cỡ **vượt** teacher. |
| 07 | `07_2018_deep_mutual_learning.md` | Deep Mutual Learning | Các student học lẫn nhau, không cần teacher lớn. |
| 08 | `08_2020_teacher_assistant.md` | Teacher Assistant | Capacity gap: teacher quá lớn thì kém; dùng model trung gian. |
| 09 | `09_2019_efficacy_of_kd.md` | On the Efficacy of KD | Teacher tốt hơn không phải lúc nào cũng dạy tốt hơn; early stopping teacher. |
| 10 | `10_2021_does_kd_really_work.md` | Does KD Really Work? | Student thường **không khớp** được teacher; vấn đề tối ưu. |
| 11 | `11_2022_patient_consistent_teacher.md` | Patient and Consistent Teacher | Cùng augmentation cho teacher và student, train rất lâu. |

## Giai đoạn 3: Feature, relation và loss mới
| # | File | Bài | Ý chính |
|---|---|---|---|
| 12 | `12_2019_relational_kd.md` | Relational KD | Học quan hệ giữa các mẫu (khoảng cách, góc). |
| 13 | `13_2020_contrastive_representation_distillation.md` | CRD | KD bằng contrastive learning trên biểu diễn. |
| 14 | `14_2022_decoupled_kd.md` | Decoupled KD | Tách KL thành phần lớp đúng và phần các lớp khác. |

## Giai đoạn 4: KD cho NLP và chuỗi
| # | File | Bài | Ý chính |
|---|---|---|---|
| 15 | `15_2016_sequence_level_kd.md` | Sequence-Level KD | KD ở mức **chuỗi**: học theo output beam search của teacher. Trung tâm cho OCR. |
| 16 | `16_2019_distilbert.md` | DistilBERT | Khởi tạo từ teacher, bỏ nửa số layer, loss ba thành phần. |
| 17 | `17_2019_patient_kd_bert.md` | Patient KD | Học hidden state của nhiều layer trung gian. |
| 18 | `18_2020_tinybert.md` | TinyBERT | Ánh xạ layer; loss trên attention, hidden, embedding, prediction. |
| 19 | `19_2020_minilm.md` | MiniLM | Distill quan hệ self-attention của layer cuối. |
| 20 | `20_2022_causal_distillation.md` | Causal Distillation | Distill **cấu trúc nhân quả** (interchange intervention). Nối với interpretability. |

## Giai đoạn 5: KD cho vision transformer và VLM
| # | File | Bài | Ý chính |
|---|---|---|---|
| 21 | `21_2021_deit.md` | DeiT | Distillation token cho ViT; hard vs soft distillation. |
| 22 | `22_2021_dino.md` | DINO | Self-distillation không cần nhãn cho ViT. |
| 23 | `23_2024_am_radio.md` | AM-RADIO | Distill nhiều teacher vào một vision encoder. Liên quan việc nén ViT. |
| 24 | `24_2024_clip_kd.md` | CLIP-KD | Nghiên cứu thực nghiệm về distill CLIP. |
| 25 | `25_2024_llava_mod.md` | LLaVA-MoD | Làm LLaVA nhỏ lại bằng MoE và KD. |
| 26 | `26_2024_llava_kd.md` | LLaVA-KD | Framework distill MLLM (output và relation). |

## Giai đoạn 6: KD cho LLM: hàm loss và on-policy
| # | File | Bài | Ý chính |
|---|---|---|---|
| 27 | `27_2024_survey_kd_of_llms.md` | Survey on KD of LLMs | Bản đồ KD cho LLM. |
| 28 | `28_2023_minillm.md` | MiniLLM | Reverse KL, tối ưu bằng policy gradient. |
| 29 | `29_2024_gkd_on_policy.md` | GKD | On-policy distillation: student học trên output của chính nó. |
| 30 | `30_2024_distillm.md` | DistiLLM | Skew KL và dữ liệu student sinh ra một cách thích nghi. |
| 31 | `31_2024_rethinking_kl_llm_kd.md` | Rethinking KL in LLM KD | Forward hay reverse KL: thực tế khác lý thuyết. |
| 32 | `32_2025_speculative_kd.md` | Speculative KD | Xen kẽ token của student và teacher khi sinh. |
| 33 | `33_2025_taid.md` | TAID | Teacher "nội suy" thay đổi dần theo thời gian để giảm capacity gap. |
| 34 | `34_2025_distillm2.md` | DistiLLM-2 | Loss contrastive giữa output của teacher và student. |

## Giai đoạn 7: Distill bằng dữ liệu và distill reasoning
| # | File | Bài | Ý chính |
|---|---|---|---|
| 35 | `35_2023_self_instruct.md` | Self-Instruct | Model tự sinh dữ liệu instruction: KD ở mức dữ liệu. |
| 36 | `36_2023_distilling_step_by_step.md` | Distilling Step-by-Step | Distill cả rationale; model nhỏ vượt model lớn. |
| 37 | `37_2023_orca.md` | Orca | Học từ explanation trace của GPT-4. |
| 38 | `38_2025_deepseek_r1_distillation.md` | DeepSeek-R1 | Distill reasoning vào model dense nhỏ; so distill với RL. |
| 39 | `39_2025_small_models_strong_reasoners.md` | Small Models Struggle to Learn from Strong Reasoners | Model nhỏ học từ teacher nhỏ hơn lại tốt hơn. |

## Giai đoạn 8: Quy mô, teacher và thực hành
| # | File | Bài | Ý chính |
|---|---|---|---|
| 40 | `40_2024_gemma2_distillation.md` | Gemma 2 | KD lúc pretraining từ teacher lớn hơn. |
| 41 | `41_2023_law_of_capacity_gap.md` | Law of Capacity Gap | Kích thước teacher tối ưu tăng theo kích thước student. |
| 42 | `42_2025_distillation_scaling_laws.md` | Distillation Scaling Laws | Khi nào distill hơn supervised; khi nào teacher lớn gây hại. |
| 43 | `43_2025_miniplm.md` | MiniPLM | KD pretraining bằng cách teacher "định hình" lại dữ liệu offline. |
| 44 | `44_2025_qwen3_distillation.md` | Qwen3 | Strong-to-weak distillation: off-policy rồi on-policy; so chi phí với RL. |
| 45 | `45_2025_sparse_logit_sampling.md` | Sparse Logit Sampling | Top-k logit đã renormalise bị bias; lấy mẫu ngẫu nhiên thì không. Áp trực tiếp cho top-20 của ta. |
| 46 | `46_2023_distil_whisper.md` | Distil-Whisper | Analogue gần nhất cho OCR: pseudo-label, lọc theo WER, transcription dài. |
| 47 | `47_2026_shortopd.md` | ShortOPD | On-policy distillation để phục hồi LLM đã prune. |

---

## Cách đọc gợi ý
- **Người mới:** 00 → 01–04 → 06, 08, 10 → 15 → 18 → 28, 29 → 42.
- **Áp vào recipe của dự án:** 00 → 15 → 46 → 45 → 41, 42 → 29, 44 → 47 → 23 (nếu nén ViT).
- Mỗi file có mục **Thí nghiệm gợi ý**.

## Liên kết với các thư mục khác
- `../pruning content/`: cách cắt model trước khi distill.
- `../interp content/`: interpretability, để hiểu distill phục hồi được gì bên trong model.
- `../reports/Distillation phục hồi sau pruning.md`: báo cáo D1–D7 cho dự án.
