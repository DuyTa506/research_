# Học LLM, Attention, VLM và Inference Optimization từ đầu: lộ trình 58 bài

Thư mục này chứa ghi chú học tập bằng tiếng Việt, đi từ Transformer gốc (2017) đến các kiến trúc linear attention và hybrid (GatedDeltaNet, Kimi Linear), VLM, model OCR và các kỹ thuật tăng tốc inference. Thứ tự được sắp cho **người mới**. Mọi paper đều được đọc từ **full text trên arXiv**, tải ngày 05/11/2026, và ID đã đối chiếu với tiêu đề.

> Đây là ghi chú diễn giải, không phải bản dịch. Mục "Ứng dụng cho dự án" là suy luận của chúng ta cho LightOnOCR-3-0.8B (decoder hybrid GatedDeltaNet + full attention, ViT 12 block, output khoảng 1.500 token/trang, GPU ≤ 8 GB, batch 1). Những con số ghi "tự tính" là do chúng ta tính, không lấy từ paper.

---

## Giai đoạn 0: Chuẩn bị
| # | File | Nội dung |
|---|---|---|
| 00 | `00_preface_toan_va_huong_dan.md` | Token, attention với shape cụ thể, causal mask, O(n²), prefill và decode, công thức KV cache, đếm tham số và FLOPs, GPU (HBM/SRAM, roofline, vì sao decode batch-1 bị nghẽn băng thông bộ nhớ), định dạng số, linear attention/SSM, cơ bản về VLM, bảng thuật ngữ, câu tự kiểm tra. **Đọc trước tiên.** |

## Giai đoạn 1: Nền tảng LLM
| # | File | Bài | Ý chính |
|---|---|---|---|
| 01 | `01_2017_attention_is_all_you_need.md` | Attention Is All You Need | Kiến trúc Transformer gốc. |
| 02 | `02_2020_gpt3_few_shot_learners.md` | GPT-3 | Mở rộng quy mô; in-context learning. |
| 03 | `03_2020_scaling_laws_kaplan.md` | Scaling Laws (Kaplan) | Loss giảm theo luật lũy thừa của N, D và compute. |
| 04 | `04_2022_chinchilla.md` | Chinchilla | Phân bổ compute tối ưu: khoảng 20 token cho mỗi tham số. |
| 05 | `05_2023_llama.md` | LLaMA | Công thức "chuẩn": Pre-RMSNorm, SwiGLU, RoPE. |
| 06 | `06_2023_mistral_7b.md` | Mistral 7B | Sliding-window attention, rolling buffer cache, GQA. |
| 07 | `07_2022_instructgpt_rlhf.md` | InstructGPT | SFT → reward model → PPO (RLHF). |
| 08 | `08_2021_lora.md` | LoRA | Fine-tune bằng cập nhật hạng thấp. |

## Giai đoạn 2: Các thành phần của Transformer hiện đại
| # | File | Bài | Ý chính |
|---|---|---|---|
| 09 | `09_2020_pre_ln_transformer.md` | On Layer Normalization in the Transformer | Pre-LN và Post-LN; vì sao Pre-LN train ổn định. |
| 10 | `10_2019_rmsnorm.md` | RMSNorm | Chuẩn hóa chỉ dùng RMS. |
| 11 | `11_2020_glu_variants_swiglu.md` | GLU Variants | SwiGLU trong FFN. |
| 12 | `12_2021_rope_roformer.md` | RoPE | Vị trí mã hóa bằng phép quay; tính chất vị trí tương đối. |
| 13 | `13_2021_alibi.md` | ALiBi | Bias tuyến tính theo khoảng cách; ngoại suy độ dài. |

## Giai đoạn 3: Các loại attention (từ MHA đến linear, SSM và hybrid)
| # | File | Bài | Ý chính |
|---|---|---|---|
| 14 | `14_2019_multi_query_attention.md` | MQA | Một K/V cho mọi head; KV cache nhỏ. |
| 15 | `15_2023_gqa.md` | GQA | Nhóm head dùng chung K/V; uptrain từ checkpoint MHA. |
| 16 | `16_2024_mla_deepseek_v2.md` | MLA (DeepSeek-V2) | Nén KV xuống latent hạng thấp; decoupled RoPE. |
| 17 | `17_2020_longformer.md` | Longformer | Attention cửa sổ trượt kết hợp global. |
| 18 | `18_2020_linear_transformers_rnns.md` | Transformers are RNNs | Linear attention = RNN với trạng thái ma trận. |
| 19 | `19_2023_retnet.md` | RetNet | Retention: song song, hồi quy, theo chunk. |
| 20 | `20_2024_gated_linear_attention.md` | Gated Linear Attention | Cổng phụ thuộc dữ liệu; train theo chunk hiệu quả trên phần cứng. |
| 21 | `21_2023_mamba.md` | Mamba | Selective SSM. |
| 22 | `22_2024_mamba2_ssd.md` | Mamba-2 / SSD | Đối ngẫu giữa SSM và attention. |
| 23 | `23_2024_deltanet_parallel.md` | DeltaNet (song song hóa delta rule) | Cập nhật trạng thái kiểu delta rule; biểu diễn WY. |
| 24 | `24_2025_gated_deltanet.md` | **Gated DeltaNet** | Cổng kết hợp delta rule: **kiến trúc decoder của dự án**. |
| 25 | `25_2024_jamba.md` | Jamba | Hybrid Transformer-Mamba (+MoE). |
| 26 | `26_2025_kimi_linear.md` | Kimi Linear | Hybrid linear attention thế hệ mới. |

## Giai đoạn 4: Mixture of Experts
| # | File | Bài | Ý chính |
|---|---|---|---|
| 27 | `27_2021_switch_transformer_moe.md` | Switch Transformer | Routing top-1, cân bằng tải. |
| 28 | `28_2024_mixtral.md` | Mixtral | Sparse MoE top-2. |

## Giai đoạn 5: Vision encoder và VLM
| # | File | Bài | Ý chính |
|---|---|---|---|
| 29 | `29_2020_vit.md` | ViT | Ảnh chia thành patch rồi đưa vào Transformer. |
| 30 | `30_2021_clip.md` | CLIP | Contrastive ảnh-chữ; zero-shot. |
| 31 | `31_2023_siglip.md` | SigLIP | Loss sigmoid thay cho softmax. |
| 32 | `32_2022_flamingo.md` | Flamingo | Perceiver resampler, gated cross-attention. |
| 33 | `33_2023_blip2.md` | BLIP-2 | Q-Former nối encoder đóng băng với LLM. |
| 34 | `34_2023_llava.md` | LLaVA | Projector đơn giản, visual instruction tuning. |
| 35 | `35_2024_llava_15.md` | LLaVA-1.5 | MLP projector, công thức baseline mạnh. |
| 36 | `36_2024_internvl_15.md` | InternVL 1.5 | Chia tile động độ phân giải cao, pixel shuffle. |
| 37 | `37_2024_qwen2_vl.md` | Qwen2-VL | Độ phân giải động, M-RoPE. |
| 38 | `38_2025_smolvlm.md` | SmolVLM | Thiết kế VLM nhỏ: cân bằng encoder/LM, ngân sách token. |

## Giai đoạn 6: Document AI và OCR
| # | File | Bài | Ý chính |
|---|---|---|---|
| 39 | `39_2022_donut.md` | Donut | Hiểu tài liệu không cần OCR, end-to-end. |
| 40 | `40_2023_nougat.md` | Nougat | PDF học thuật thành markup; xử lý lặp lại và hallucination. |
| 41 | `41_2024_got_ocr2.md` | GOT-OCR2.0 | "OCR-2.0" thống nhất end-to-end. |
| 42 | `42_2025_olmocr.md` | olmOCR | Pipeline dữ liệu, document anchoring, benchmark. |
| 43 | `43_2026_lightonocr.md` | **LightOnOCR** | Thế hệ trước của model dự án: kiến trúc, dữ liệu, distill, cắt vocabulary. |

## Giai đoạn 7: Inference optimization
| # | File | Bài | Ý chính |
|---|---|---|---|
| 44 | `44_2022_flashattention.md` | FlashAttention | Tiling, online softmax, phân tích IO. |
| 45 | `45_2023_flashattention2.md` | FlashAttention-2 | Song song hóa và chia việc tốt hơn. |
| 46 | `46_2023_pagedattention_vllm.md` | PagedAttention (vLLM) | KV cache chia trang như bộ nhớ ảo. |
| 47 | `47_2024_sglang_radixattention.md` | SGLang | RadixAttention: dùng lại prefix. |
| 48 | `48_2023_speculative_decoding.md` | Speculative Decoding | Draft model nhỏ đoán trước, model lớn kiểm tra; phân phối output không đổi. |
| 49 | `49_2024_medusa.md` | Medusa | Nhiều decoding head, tree attention. |
| 50 | `50_2024_eagle.md` | EAGLE | Speculative ở mức feature. |
| 51 | `51_2022_llm_int8.md` | LLM.int8() | Outlier feature; nhân ma trận 8-bit. |
| 52 | `52_2023_smoothquant.md` | SmoothQuant | Chuyển độ khó quantization từ activation sang weight. |
| 53 | `53_2023_gptq.md` | GPTQ | Quantization dựa trên OBS (cùng họ với SparseGPT). |
| 54 | `54_2024_awq.md` | AWQ | Quantization weight có tính đến activation. |
| 55 | `55_2024_streamingllm_attention_sinks.md` | StreamingLLM | Attention sink; cache cửa sổ cộng sink. |
| 56 | `56_2023_h2o_kv_cache.md` | H2O | Loại bớt KV cache theo "heavy hitter". |
| 57 | `57_2024_fastv_visual_token_pruning.md` | FastV | Cắt visual token sau layer 2. |

---

## Cách đọc gợi ý
- **Người mới hoàn toàn:** 00 → 01 → 05 → 09–12 → 14, 15 → 29, 34, 35 → 44 → 48.
- **Hiểu decoder hybrid của dự án:** 00 → 01 → 14, 15 → 18 → 20 → 21 → 23 → **24** → 25 → 26.
- **Hiểu phần vision và OCR:** 29 → 31 → 35 → 36 → 37 → 38 → 39–43 → 57.
- **Tối ưu deploy batch-1 trên GPU ≤ 8 GB:** 00 (roofline, KV cache) → 44 → 48–50 → 51–54 → 55–57.

## Liên kết với các thư mục khác
- `../pruning content/` và `../distillation content/`: nén và phục hồi model.
- `../interp content/`: interpretability.
- `../reports/`: báo cáo tổng hợp cho dự án.
