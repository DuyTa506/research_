# Học Pruning từ đầu: lộ trình 50 bài

Thư mục này chứa ghi chú học tập bằng tiếng Việt về **pruning mạng neural**, từ các bài kinh điển năm 1989 đến các paper năm 2026. Thứ tự được sắp cho **người chưa biết gì về domain**: cơ bản trước, rồi chuyển dần sang LLM, model hybrid và VLM.

Mọi paper (trừ OBD/OBS) đều được đọc từ **full text trên arXiv** (PDF, tải ngày 05/11/2026). ID của từng bài đã được đối chiếu với tiêu đề.

> Đây là ghi chú diễn giải, không phải bản dịch. Mục "Ứng dụng cho dự án" trong mỗi file là suy luận của chúng ta cho LightOnOCR (cắt 0.8B xuống 0.3B/0.1B, D_lin/D_res), **không phải** kết luận của paper. Trước khi trích số liệu, hãy mở bản gốc theo link ở đầu mỗi file.

---

## Giai đoạn 0: Chuẩn bị
| # | File | Nội dung |
|---|---|---|
| 00 | `00_preface_toan_va_huong_dan.md` | Pruning là gì, phân loại (unstructured / N:M / structured), tiêu chí, phục hồi, toán (Taylor, Hessian, OBS, least squares), đánh giá latency, bảng thuật ngữ, câu tự kiểm tra. **Đọc trước tiên.** |

## Giai đoạn 1: Nền tảng kinh điển (CNN era)
| # | File | Bài | Ý chính |
|---|---|---|---|
| 01 | `01_1989_1993_optimal_brain_damage_surgeon.md` | Optimal Brain Damage / Optimal Brain Surgeon | Chọn trọng số để cắt bằng đạo hàm bậc 2 và **bù trừ** các trọng số còn lại. Gốc của SparseGPT. |
| 02 | `02_2015_learning_weights_and_connections.md` | Han et al. 2015 | Cắt theo độ lớn (magnitude), xen kẽ với retrain. |
| 03 | `03_2017_to_prune_or_not_to_prune.md` | Zhu & Gupta 2017 | Gradual pruning; model lớn đã thưa so với model nhỏ dày đặc. |
| 04 | `04_2016_pruning_filters_convnets.md` | Li et al. 2016 | Structured pruning: cắt cả filter để tăng tốc thật. |
| 05 | `05_2017_network_slimming.md` | Network Slimming | Dùng hệ số BatchNorm với phạt L1 làm tiêu chí cắt kênh. |
| 06 | `06_2017_thinet.md` | ThiNet | Chọn kênh theo **sai số tái tạo** của layer kế tiếp (least squares). |
| 07 | `07_2019_importance_estimation_taylor.md` | Molchanov et al. 2019 | Ước lượng importance bằng Taylor bậc 1 và 2. |

## Giai đoạn 2: Lý thuyết và tranh luận
| # | File | Bài | Ý chính |
|---|---|---|---|
| 08 | `08_2018_lottery_ticket_hypothesis.md` | Lottery Ticket Hypothesis | Mạng con "trúng thưởng" train lại được từ khởi tạo ban đầu. |
| 09 | `09_2019_linear_mode_connectivity_lth.md` | Linear Mode Connectivity & LTH | Khi nào ticket tồn tại; vai trò của độ ổn định với nhiễu SGD. |
| 10 | `10_2018_snip.md` | SNIP | Cắt ngay lúc khởi tạo, trước khi train. |
| 11 | `11_2018_rethinking_value_of_pruning.md` | Rethinking the Value of Pruning | Train lại từ đầu thường ngang fine-tune; giá trị nằm ở **kiến trúc** tìm được. |
| 12 | `12_2020_state_of_neural_network_pruning.md` | What is the State of Pruning? | Cách đọc paper pruning một cách phản biện; vấn đề baseline và benchmark. |

## Giai đoạn 3: Pruning transformer (BERT era)
| # | File | Bài | Ý chính |
|---|---|---|---|
| 13 | `13_2019_sixteen_heads.md` | Are Sixteen Heads Really Better than One? | Cắt được nhiều head mà gần như không mất gì. |
| 14 | `14_2019_voita_specialized_heads.md` | Voita et al. 2019 | Vài head chuyên biệt làm phần lớn việc; phần còn lại cắt được. |
| 15 | `15_2020_lottery_ticket_bert.md` | LTH for pre-trained BERT | Ticket trên model pretrain; mask có chuyển giao được không. |
| 16 | `16_2020_movement_pruning.md` | Movement Pruning | Cắt theo chiều trọng số **di chuyển** khi fine-tune, thay vì theo độ lớn. |
| 17 | `17_2022_cofi_structured_pruning.md` | CoFi | Structured pruning với mask đa độ mịn kết hợp distillation từng layer. |

## Giai đoạn 4: Phần cứng và công cụ
| # | File | Bài | Ý chính |
|---|---|---|---|
| 18 | `18_2021_nvidia_2to4_sparsity.md` | Accelerating Sparse DNNs (2:4) | Semi-structured sparsity được GPU hỗ trợ. |
| 19 | `19_2021_halp_latency_pruning.md` | HALP | Pruning theo **ngân sách latency** thay vì số tham số. |
| 20 | `20_2023_depgraph.md` | DepGraph | Xử lý phụ thuộc giữa các layer khi cắt structured cho kiến trúc bất kỳ. |

## Giai đoạn 5: LLM, pruning một lần (post-training)
| # | File | Bài | Ý chính |
|---|---|---|---|
| 21 | `21_2023_survey_llm_compression.md` | Survey on LLM Compression | Bản đồ toàn cảnh: pruning, quantization, distillation, low-rank. |
| 22 | `22_2022_optimal_brain_compression.md` | Optimal Brain Compression | OBS chính xác theo từng hàng, tái tạo layer. |
| 23 | `23_2023_sparsegpt.md` | SparseGPT | Cắt 50% LLM một lần, có bù trừ trọng số. |
| 24 | `24_2023_wanda.md` | Wanda | \|W\|·\|\|X\|\|, không bù trừ; đơn giản mà mạnh. |
| 25 | `25_2023_ziplm.md` | ZipLM | Structured, chọn theo sai số **sau khi bù trừ OBS**, có ràng buộc latency. Gần ý tưởng D_lin. |
| 26 | `26_2023_llm_pruner.md` | LLM-Pruner | Structured pruning LLM với nhóm phụ thuộc, phục hồi bằng LoRA. |
| 27 | `27_2023_llm_surgeon.md` | LLM Surgeon | Dùng curvature (Kronecker-factored) để cắt structured và cập nhật trọng số. |
| 28 | `28_2023_flap.md` | FLAP | Tiêu chí fluctuation, bù trừ bằng bias, không cần retrain. |
| 29 | `29_2024_slicegpt.md` | SliceGPT | Xoay residual stream bằng PCA rồi xóa hàng và cột. |
| 30 | `30_2024_maskllm.md` | MaskLLM | Học mask N:M bằng huấn luyện end-to-end. |

## Giai đoạn 6: Depth pruning (cắt layer)
| # | File | Bài | Ý chính |
|---|---|---|---|
| 31 | `31_2024_shortgpt.md` | ShortGPT | Block Influence: cắt các layer ít thay đổi residual stream. |
| 32 | `32_2024_unreasonable_ineffectiveness_deeper_layers.md` | Gromov et al. | Cắt khối layer sâu theo khoảng cách góc, "heal" bằng QLoRA. |
| 33 | `33_2024_shortened_llama.md` | Shortened LLaMA | So depth với width; LoRA và continued pretraining. |
| 34 | `34_2024_sleb.md` | SLEB | Loại block theo kiểm chứng dư thừa. |
| 35 | `35_2024_laco.md` | LaCo | "Gập" các layer vào nhau thay vì xóa. |
| 36 | `36_2025_replaceme.md` | ReplaceMe | Thay các block bị cắt bằng **một phép biến đổi tuyến tính**. Rất gần D_lin. |
| 37 | `37_2026_repair_changes_which_layers.md` | Repair Changes Which Layers to Remove | Khi có repair, tập layer tối ưu để cắt thay đổi. Rất gần hướng nghiên cứu của ta. |

## Giai đoạn 7: Prune và distill ở quy mô lớn
| # | File | Bài | Ý chính |
|---|---|---|---|
| 38 | `38_2023_sheared_llama.md` | Sheared LLaMA | Targeted structured pruning, dynamic batch loading. |
| 39 | `39_2024_minitron.md` | Minitron | Depth so với width; thứ hạng đảo sau retrain; công thức KD. |
| 40 | `40_2024_llama31_minitron.md` | Llama-3.1-Minitron | Teacher correction; width thắng depth sau distill. |
| 41 | `41_2024_flextron.md` | Flextron | Model "nhiều trong một", chọn cấu hình theo latency. |
| 42 | `42_2024_puzzle.md` | Puzzle | NAS dựa trên distillation, tối ưu cho inference trên phần cứng thật. |
| 43 | `43_2026_small_llms_pruning_vs_scratch.md` | Small LLMs: Pruning vs. Training from Scratch | Khi nào prune có lợi hơn train từ đầu. |

## Giai đoạn 8: Hybrid (SSM / linear attention) và VLM
| # | File | Bài | Ý chính |
|---|---|---|---|
| 44 | `44_2025_nemotron_h.md` | Nemotron-H | Kiến trúc hybrid Mamba-Transformer; nén 56B xuống 47B bằng MiniPuzzle. |
| 45 | `45_2025_minitron_ssm.md` | Minitron-SSM | Group-aware pruning cho Mamba head. |
| 46 | `46_2025_mamba_shedder.md` | Mamba-Shedder | Loại block, module SSM và kênh trong Mamba/hybrid. |
| 47 | `47_2025_jet_nemotron_postnas.md` | Jet-Nemotron | Nên giữ layer full-attention nào trong hybrid. Liên quan GatedDeltaNet. |
| 48 | `48_2024_ecoflap_vlm.md` | ECoFLaP | Pruning VLM: phân bổ sparsity từ thô đến mịn. |
| 49 | `49_2026_structural_pruning_vlms.md` | Structural Pruning of Large VLMs | Width so với layer trên VLM; phục hồi bằng SFT/KD; hiệu quả dữ liệu. |

---

## Cách đọc gợi ý
- **Người mới:** đọc 00 → giai đoạn 1 → 08, 11, 12 → 21, 22, 23, 24 → giai đoạn 6 → 39, 40.
- **Người cần áp vào dự án nhanh:** 00 → 22 → 25 → 36 → 37 → 39 → 40 → 47 → 49.
- Mỗi file có mục **Thí nghiệm gợi ý**. Thực hành trên một model nhỏ sẽ giúp hiểu nhanh hơn chỉ đọc.

## Liên kết với các thư mục khác
- `../distillation content/`: phục hồi model sau pruning bằng distillation.
- `../interp content/`: interpretability, để hiểu *bên trong* model mất gì khi bị cắt.
- `../reports/`: các báo cáo tổng hợp cho bài báo của dự án.
