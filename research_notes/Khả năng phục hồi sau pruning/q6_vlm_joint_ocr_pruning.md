# Q6: Joint pruning of the vision encoder and LLM decoder in VLMs, latency-budgeted pruning, and OCR/document-specific compression

Method note: arxiv.org, huggingface.co, openaccess.thecvf.com and tldr.takara.ai could not be fetched from this session (DNS blocked by network policy). Every finding below comes from web-search result snippets of primary pages (arXiv abs/pdf/html, CVF, PMLR, ICLR proceedings, Springer, GitHub, HF model cards), not from full-text reading. Numbers are as stated in those snippets. Items not confirmed by any search are marked "unverified". To re-check tables, allow arxiv.org and huggingface.co in the session's network settings.

## Which papers structurally prune BOTH the vision encoder and the LLM decoder of a VLM (beyond visual-token pruning)?

### Takeaway
Few papers do true joint encoder+decoder weight pruning. UPop (ICML 2023) and MULTIFLOW (CVPR 2024) prune whole BLIP/XVLM-style vision-language transformers, and ECoFLaP (ICLR 2024) prunes BLIP-2's vision encoder and LLM. All of these predate modern LLaVA-style decoder VLMs, and ECoFLaP/MULTIFLOW are mostly unstructured. The 2025-2026 structural-pruning studies of LLaVA-type MLLMs (Huang et al., GCPR 2025 → IJCV 2026) prune **only the language backbone**. The user-given ID 2604.24380 is correct. Joint "parameter + token" pruning (CoMP 2026, SlimVLM 2026) is the newest direction. None of these papers is OCR-specific.

### Cited Findings
**Huang et al.: verified, with two versions**
- Conference version: "Investigating Structural Pruning and Recovery Techniques for Compressing Multimodal Large Language Models: An Empirical Study". Authors: Yiran Huang, Lukas Thede, Massimiliano Mancini, Wenjia Xu, Zeynep Akata. arXiv **2507.20749** (28 Jul 2025), accepted at **GCPR 2025** (Oral per GitHub). It studies LLaVA-v1.5-7B and Bunny-v1.0-3B and prunes the **language backbone only**, not the vision encoder. It compares layerwise and widthwise pruning, with recovery by SFT and KD. — [arXiv 2507.20749](https://arxiv.org/abs/2507.20749v1); [GitHub VLMCompression](https://github.com/YiranHuangIrene/VLMCompression)
- 2507.20749 findings: widthwise pruning is more robust under low compute/data; SFT + hidden-state distillation gives the best recovery; recovery with ~5% of the original training data kept >95% of original performance (search-snippet summary; not re-checked against tables). — [arXiv 2507.20749](https://arxiv.org/abs/2507.20749v1)
- Journal version: "Structural Pruning of Large Vision Language Models: A Comprehensive Study on Pruning Dynamics, Recovery, and Data Efficiency", **IJCV 2026**, DOI 10.1007/s11263-026-02865-5, arXiv **2604.24380**. **The user's ID 2604.24380 matches this paper.** — [Springer IJCV](https://link.springer.com/article/10.1007/s11263-026-02865-5); [arXiv 2604.24380](https://arxiv.org/abs/2604.24380)
- 2604.24380 scope: widthwise and layerwise pruning of the LLM on **LLaVA-v1.5-7B, InternVL-Chat-4B (Mini-InternVL), Bunny-v1-3B**, at pruning ratios of 15%–60%, with recovery by SFT and KD. No Qwen models were found in the snippets. — [arXiv 2604.24380 pdf](https://arxiv.org/pdf/2604.24380)
- 2604.24380 results:
  - Without finetuning, widthwise pruning beats layerwise.
  - With recovery training, layerwise is slightly better at ≤30% and widthwise is better at higher ratios. Above ~45%, widthwise recovers better on Mini-InternVL and Bunny and roughly ties on LLaVA.
  - Layerwise pruning disproportionately hurts complex reasoning.
  - **Layerwise pruning gives lower inference latency than widthwise in all tested settings.**
  - Recommended regimes: widthwise only for extremely low-resource settings; layerwise + MM-projector finetuning for moderate compression (≤30%); widthwise + KD for high compression (≤45%).
  - At small compression ratios, finetuning only the multimodal projector suffices.
  — [arXiv 2604.24380](https://arxiv.org/pdf/2604.24380); [Springer](https://link.springer.com/article/10.1007/s11263-026-02865-5)

**ECoFLaP**
- "ECoFLaP: Efficient Coarse-to-Fine Layer-Wise Pruning for Vision-Language Models", Sung, Yoon, Bansal. **ICLR 2024**, arXiv **2310.02998** (ID verified).
- Method: global layer-wise sparsity ratios come from a zeroth-order approximation of global gradients (coarse step), followed by local layer-wise **unstructured** pruning in the style of Wanda/SparseGPT (fine step).
- Results: up to 5% relative improvement in average multimodal performance over SparseGPT/Wanda/global pruning, mainly at high sparsity. The method also applies to unimodal FlanT5, LLaMA and EVA-ViT.
— [ICLR 2024 poster](https://www.iclr.cc/virtual/2024/poster/18067); [ICLR proceedings pdf](https://proceedings.iclr.cc/paper_files/paper/2024/file/8a67127ed400dee7851c99469fe4b829-Paper-Conference.pdf); [ar5iv](https://ar5iv.labs.arxiv.org/html/2310.02998); [GitHub](https://github.com/ylsung/ECoFLaP)
- ECoFLaP numbers on BLIP-2 at 0.5 sparsity: macro-average 58.0 vs Wanda 57.2 vs SparseGPT 57.4 (secondary summary). Calibration uses 128 samples from CC3M/ImageNet/C4. The follow-up paper RESSA reproduces ECoFLaP at 50% **unstructured** sparsity with VQAv2 57.5, OK-VQA 36.2, GQA 42.1. — [Liner review](https://liner.com/review/ecoflap-efficient-coarsetofine-layerwise-pruning-for-visionlanguage-models); [RESSA arXiv 2404.02424](https://arxiv.org/html/2404.02424v1)
- In a **structured** 50% setting, ECoFLaP collapses to VQAv2 6.40, OK-VQA 7.76, GQA 4.25, as reported by competitor UKMP (AAAI 2025). This shows that unstructured VLM pruning methods do not transfer to structured pruning. — [UKMP, AAAI](https://ojs.aaai.org/index.php/AAAI/article/download/32923/35078)
- Related paper: "Rethinking Pruning for Vision-Language Models: Strategies for Effective Sparsity and Performance Restoration" (RESSA), arXiv **2404.02424**, studies how to allocate sparsity across the vision and language modules and repairs sparse VLMs with sparse cross-modality adaptation. Its venue (NAACL Findings 2025?) is unverified. — [arXiv 2404.02424](https://arxiv.org/abs/2404.02424v2)

**UPop and MULTIFLOW**
- UPop: "Unified and Progressive Pruning for Compressing Vision-Language Transformers", Dachuan Shi, Chaofan Tao et al. **ICML 2023** (PMLR v202), arXiv **2301.13741**.
  - Method: it searches multimodal subnets in a continuous space, so pruning ratios are **automatically allocated across modalities and structures** (vision, language, cross-modal). It then progressively searches and retrains.
  - Results: on NLVR2 at 2× compression, the progressive step adds +0.83/+0.81 (dev/test) over unified-only pruning. The paper notes that separate (per-modality) and unified pruning behave notably differently. The headline speedups were not retrieved.
  — [PMLR](https://proceedings.mlr.press/v202/shi23e.html); [arXiv html](https://arxiv.org/html/2301.13741)
- MULTIFLOW: "Shifting Towards Task-Agnostic Vision-Language Pruning", Farina, Mancini, Cunegatti, Liu, Iacca, Ricci. **CVPR 2024** pp. 16185–16195, arXiv **2404.05621**.
  - Method: it defines task-agnostic VLP (prune once, transfer to unknown tasks). It is gradient-free and scores each parameter by magnitude × information flow.
  - Setup: models are XVLM and BLIP; tasks are retrieval, captioning and VQA; three sparsity ratios.
  - Results: it outperforms combinatorial competitors in most cases, with a gap over TAMT that grows with sparsity. Per-sparsity tables were not retrieved.
  — [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Farina_MULTIFLOW_Shifting_Towards_Task-Agnostic_Vision-Language_Pruning_CVPR_2024_paper.html); [GitHub](https://github.com/FarinaMatteo/multiflow); [supplement](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Farina_MULTIFLOW_Shifting_Towards_CVPR_2024_supplemental.pdf)
- PORTA, "Prune Once: Retraining-Free Task-Agnostic Pruning for Vision-Language Models", Kang et al., arXiv **2608.06901** (2026; one citing paper calls it ECCV 2026, unverified). Method: activation-variation importance plus layer-wise ratios from output-feature variability. Models: CLIP, BLIP, **Qwen2-VL**. It reports competitive results at high sparsity with no retraining; no numbers were retrieved. — [arXiv 2608.06901](https://arxiv.org/abs/2608.06901)

**Joint parameter + token pruning (2025–2026)**
- CoMP, "Collaborative Multi-Mode Pruning for Vision-Language Models", arXiv **2604.02956** (3 Apr 2026). It jointly prunes parameters and visual tokens using a Collaborative Importance Metric and a staged strategy that picks a mode by estimated cost, under a fixed total FLOPs budget. Latency is reported on an RTX 4090. No numbers were retrieved, and the code is "coming soon". — [arXiv 2604.02956](https://arxiv.org/abs/2604.02956); [GitHub](https://github.com/Wuzimeng/CoMP)
- SlimVLM, "Sensitivity-aware Dynamic Structured Pruning with Adaptive Visual Token Selection", arXiv **2608.03580** (2026). It claims near-original accuracy at 20% pruning. A table fragment shows a 1.25× speedup at 20% pruning; its column headers were missing. — [arXiv 2608.03580](https://arxiv.org/html/2608.03580)
- EfficientLLaVA, "Generalizable Auto-Pruning for LVLMs", Liang et al., **CVPR 2025** pp. 9445–9454, arXiv **2503.15369**. It searches a pruning policy from only 64 samples: ScienceQA 83.05% with a **1.8× speedup** vs dense LLaVA-v1.5-7B. It prunes **only LLaMA MLP weights** and explicitly excludes the visual encoder and projector because pruning them has minimal impact. — [CVF pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Liang_EfficientLLaVA_Generalizable_Auto-Pruning_for_Large_Vision-language_Models_CVPR_2025_paper.pdf); [arXiv 2503.15369](https://arxiv.org/pdf/2503.15369)
- Other 2026 LVLM layer/structured pruning found but not read:
  - Topology-Aware Layer Pruning for LVLMs, arXiv **2604.16502**. It reports that ECoFLaP, Wanda and OWL degrade more than its method beyond ~60% sparsity.
  - Short-LVLM, arXiv **2507.23362** (layer pruning).
  - HiPP-Prune, arXiv **2603.06270**.
  - Information-theoretic adaptive structural pruning, arXiv **2511.19518**.
  — [2604.16502](https://arxiv.org/pdf/2604.16502); [2507.23362](https://arxiv.org/pdf/2507.23362); [2603.06270](https://arxiv.org/pdf/2603.06270); [2511.19518](https://arxiv.org/pdf/2511.19518)

**Distillation and small-model design (alternatives to pruning)**
- LLaVA-MoD, arXiv **2408.15881**, **ICLR 2025** per GitHub. It distills a large MLLM into a small MLLM whose LLM is turned into a **sparse MoE** (FFNs as experts, top-k gating). It uses mimic distillation (KL on outputs) followed by preference distillation. — [arXiv 2408.15881](https://arxiv.org/abs/2408.15881); [GitHub](https://github.com/shufangxun/llava-mod)
- TinyLLaVA, arXiv **2402.14289** (2024). This is a framework for small LMMs. TinyLLaVA-3.1B beats 7B LLaVA-1.5 and Qwen-VL overall, and data quality and training recipe matter as much as scale. The claims "SigLIP > CLIP" and "unfreezing the encoder helps smaller LMs" come only from a secondary review (unverified against the paper). — [arXiv 2402.14289](https://arxiv.org/pdf/2402.14289)
- Other distillation works found: LLaVA-KD (arXiv 2410.16236) and MoVE-KD (arXiv 2501.01709); details not read. — [LLaVA-KD](https://arxiv.org/pdf/2410.16236); [MoVE-KD](https://arxiv.org/pdf/2501.01709)
- SmolVLM, arXiv **2504.05299** (2025). Finding 1: compact VLMs benefit from **balanced encoder/LM allocation**. At LM 135M, a larger encoder *hurt* performance. At 360M, a larger encoder gave about +11.6% at the cost of about 66% more parameters, so the compact encoder was still preferred. At 1.7B, the large encoder added only about 10% parameters. Shipped configurations: SigLIP-B/16 (93M) for the 256M/500M models; SigLIP-SO400M (428M) for the 2.2B model. — [arXiv 2504.05299](https://arxiv.org/pdf/2504.05299)
- FastVLM (Apple), **CVPR 2025** pp. 19769–19780, arXiv **2412.13303**. It introduces FastViTHD, a hybrid conv-transformer encoder that outputs fewer tokens at high resolution.
  - In the LLaVA-1.5 setup: 3.2× better TTFT at similar accuracy.
  - The 0.5B variant vs LLaVA-OneVision-0.5B at 1152×1152: **85× faster TTFT** with a 3.4× smaller vision encoder.
  — [CVF pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Vasu_FastVLM_Efficient_Vision_Encoding_for_Vision_Language_Models_CVPR_2025_paper.pdf); [GitHub](https://github.com/apple/ml-fastvlm)
- Isomorphic Pruning (Fang, Ma, Mi, Wang), **ECCV 2024**, LNCS pp. 232–250. It applies to ViTs and CNNs only, **not VLMs**. It groups isomorphic structures and ranks them within each group. Pruning DeiT-Base gives a DeiT-Tiny-size model at 77.50% vs 74.52% for the original DeiT-Tiny. It is integrated in Torch-Pruning v1.4.1, which makes it a candidate for pruning the user's ViT. — [ECVA pdf](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04364.pdf); [GitHub](https://github.com/VainF/Isomorphic-Pruning)

**MLLM layer skipping (visual tokens in deep layers)**
- FitPrune, arXiv **2409.10197**: pruning visual tokens in deep layers costs little (about 3.2% for a 50% cut after layer 4; about 75% dropped at layer 28 with no loss). — [FitPrune](https://arxiv.org/pdf/2409.10197)
- Other work: HiDrop (arXiv 2602.23699), operator-level visual skipping (arXiv 2606.31903), DOP (arXiv 2507.02909), and a redundancy-lifecycle analysis (arXiv 2604.16462). — [HiDrop](https://arxiv.org/html/2602.23699v1); [2606.31903](https://arxiv.org/pdf/2606.31903)

### Inferences
- For the user, no published recipe structurally prunes a ViT and a hybrid GatedDeltaNet decoder together. The closest templates are:
  - UPop: learned, automatic allocation across modalities (BLIP-era).
  - Huang et al. IJCV 2026: LLM-only, with SFT + hidden-state KD recovery.
  - Isomorphic pruning / Torch-Pruning for the ViT part.
- Huang et al.'s latency finding (layerwise < widthwise latency) matters for batch-1 decode, where depth sets the per-token latency. Their accuracy finding (widthwise better at >45%) matters for the user's 0.8B→0.1B target (about 87% compression), which is far beyond any ratio in that study (60% maximum).

### Gaps
- Not found in any search: LLaVA-PruMerge (token-only; recalled as arXiv 2403.15388, unverified); "Pruning by Explaining revisited" for VLMs; MoE-LLaVA (recalled as arXiv 2401.15947, unverified); DeepSeek-VL2-Tiny (recalled as DeepSeek-VL2 arXiv 2412.10302, about 1.0B activated / 3.37B total, unverified).
- Per-sparsity tables for UPop, MULTIFLOW and PORTA were not retrieved. It is also unclear whether ECoFLaP's 0.5-sparsity BLIP-2 results prune EVA-ViT and FlanT5 jointly in the main table.

## Latency-budgeted (not parameter-budgeted) structured pruning

### Takeaway
HALP and ZipLM are the canonical latency-constrained structured pruners. Both rely on a measured latency lookup table for the target device. Puzzle and Flextron (NVIDIA) extend this to LLMs with throughput/latency-targeted NAS or elastic sub-networks. I found no paper that does latency-budgeted joint pruning of a VLM's vision encoder and decoder on a ≤8 GB GPU. CoMP uses a FLOPs budget, not a latency budget.

### Cited Findings
- **HALP**, "Hardware-Aware Latency Pruning", arXiv **2110.10811**. The later version, "Structural Pruning via Latency-Saliency Knapsack" (arXiv **2210.06659**), is **NeurIPS 2022**.
  - Method: global resource allocation that maximizes accuracy under a latency budget, using a **latency lookup table** plus a global saliency score (a knapsack problem).
  - Results: ResNet-50/-101 throughput ×1.60/×1.90 with +0.3/−0.2% top-1; SSD on VOC ×1.94 throughput at −0.56 mAP.
  - Critique from a competitor (Archtree, arXiv 2311.10549): HALP models often miss the latency target, and HALP keeps only a single candidate.
  — [arXiv 2110.10811](https://arxiv.org/abs/2110.10811); [2210.06659](https://arxiv.org/pdf/2210.06659); [Archtree](https://arxiv.org/pdf/2311.10549)
- **ZipLM**, "Inference-Aware Structured Pruning of Language Models", Kurtic et al., **NeurIPS 2023**, arXiv **2302.04089**.
  - Method: it uses a latency table for the target inference environment and has separate throughput and latency regimes. It works one-shot or with recovery.
  - Results: snippets state 3×–13× speedup at full accuracy recovery and 30×–50× at the largest compression in compound compression for edge inference. The exact settings were not confirmed.
  — [NeurIPS pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/ced46a50befedcb884ccf0cbe8c3ad23-Paper-Conference.pdf); [OpenReview](https://openreview.net/pdf?id=bPFFPueAxm)
- **Puzzle** (NVIDIA), "Distillation-Based NAS for Inference-Optimized LLMs", arXiv **2411.19146**, **ICML 2025** (PMLR v267).
  - Method: blockwise local distillation (BLD) over per-block alternatives, then mixed-integer programming under a user-given throughput constraint (tokens/s/GPU on H100, FP8).
  - Results: Nemotron-51B from Llama-3.1-70B-Instruct gives 2.17× throughput and fits on a single H100, keeping 98.4% of capabilities. The throughput constraint was about 5,500 tok/s. It keeps >96% accuracy at 16K context although it was trained on ≤8K.
  — [PMLR](https://proceedings.mlr.press/v267/bercovich25a.html); [arXiv](https://arxiv.org/html/2411.19146v1)
- **Flextron** (NVIDIA/UT Austin), "Many-in-One Flexible LLM", **ICML 2024 Oral**, arXiv 2406.10260 (ID not confirmed by search). It uses a nested elastic architecture that meets latency/accuracy targets at inference without finetuning, and it is input-adaptive (it routes tokens through sub-networks). — [project page](https://cairuisi.github.io/Flextron/)
- MDP, "Multidimensional Vision Model Pruning with Latency Constraint", arXiv **2504.02168**: linear latency models break for transformers, so it models latency jointly over all prunable dimensions. — [arXiv 2504.02168](https://arxiv.org/pdf/2504.02168)
- LightPrune (ICCV 2025 Workshop): latency-aware structured pruning on embedded devices, for CNNs only. — [CVF](https://openaccess.thecvf.com/content/ICCV2025W/EVW/papers/Belhadi_LightPrune_Latency-Aware_Structured_Pruning_for_Efficient_Deep_Inference_on_Embedded_ICCVW_2025_paper.pdf)
- Multi-objective structured pruning of LLMs for latency and size: arXiv **2607.22583** (Pareto and Bayesian-optimization layerwise sparsity under latency constraints; LLM only). — [arXiv 2607.22583](https://arxiv.org/pdf/2607.22583)
- FastVLM is the clearest latency-oriented VLM result. It optimizes **TTFT** (encoder plus prefill), not per-token decode. — [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Vasu_FastVLM_Efficient_Vision_Encoding_for_Vision_Language_Models_CVPR_2025_paper.pdf)

### Inferences
- For OCR with long outputs at batch 1, end-to-end latency ≈ T_ViT(resolution) + T_prefill(#image tokens) + N_out × T_decode/token. A HALP/ZipLM-style lookup table would need entries for both the per-token decode cost (dominated by depth and by memory bandwidth for weights and KV/state) and the ViT cost at the target resolution. Huang et al.'s finding that layer removal cuts latency more than width removal supports dropping decoder depth when the budget is latency, not parameters.
- LAPP: I found no source under that name, so it is unverified.

### Gaps
- No latency-budgeted structured pruning paper for VLMs on small (≤8 GB) GPUs was found. No paper treats GatedDeltaNet/linear-attention hybrid decoders under latency-aware pruning.

## OCR / document-specific compression, distillation, or pruning

### Takeaway
I found no published paper that applies structured pruning + KD to an end-to-end OCR VLM. Small OCR VLMs are built from small parts (SmolDocling/Granite-Docling 256–258M, PaddleOCR-VL 0.9B, MinerU2.5 1.2B, LightOnOCR 1B and 0.8B) and trained by distilling a teacher's transcriptions. LightOn's only "pruning" is **vocabulary pruning** of LightOnOCR-1B. Visual-token or KV compression for documents (DeepSeek-OCR, FastOCR) is the other active line of work.

### Cited Findings
**LightOn**
- LightOnOCR (1 and 2): arXiv **2601.14251**, "LightOnOCR: A 1B End-to-End Multilingual Vision-Language Model for State-of-the-Art OCR". v1 is from 20 Jan 2026 and v2 from 30 Jun 2026.
  - Architecture (LightOnOCR-1B): a Pixtral-based ViT encoder plus a Qwen3-based decoder.
  - Training data: transcriptions distilled from a strong VLM teacher.
  - LightOnOCR-2: same architecture with a new nvpdftex-based arXiv pipeline. Checkpoints are made by model merging (an OCR soup at α=0.4 and a bbox soup at α=0.1).
  — [arXiv 2601.14251](https://arxiv.org/abs/2601.14251v1); [HF blog LightOnOCR-1B](https://huggingface.co/blog/lightonai/lightonocr)
- Vocabulary pruning in LightOnOCR-1B: Qwen3's 151,936-token vocabulary was reduced to 51k/32k/16k.
  - 32k: **11.6% faster inference, 96% of base performance**.
  - 16k: **−13.8% parameters**, OlmOCR-Bench 75.4 vs 76.1.
  - Non-Latin scripts suffer: about 3× token inflation for Arabic and Chinese.
  - Release names: "LightOnOCR-0.9B-32k-1025" / "-16k-1025" in the paper; the HF card labels them as 1B variants, so naming is inconsistent.
  — [arXiv 2601.14251](https://arxiv.org/pdf/2601.14251); [HF blog](https://huggingface.co/blog/lightonai/lightonocr)
- Throughput: LightOnOCR-1B is quoted at **5.71 pages/s on a single H100** (about 493k pages/day, under $0.01 per 1k pages, per the HF card). A secondary source gives 5.55 and one review gives 15.7; the official figure is 5.71. — [HF card LightOnOCR-1B-1025](https://huggingface.co/lightonai/LightOnOCR-1B-1025)
- LightOnOCR-3 family: 0.8B and 4B models use the **Qwen3.5 VLM architecture**, while LightOnOCR-3-1B keeps the LightOnOCR-2 design.
  - The 0.8B model is reported at an average of 76.9 and an olmOCR score of 85.5 (the 4B averages 78.5). Its best olmOCR-Bench score uses 400 DPI with a 5 MP cap, **about 4.8k image tokens per page**.
  - A third-party ONNX repo describes 18 Gated-DeltaNet + 6 full-attention layers (unverified vs the official card).
  - I found no LightOn report of structured pruning or distilled sub-0.8B variants.
  — [HF LightOnOCR-3-0.8B](https://huggingface.co/lightonai/LightOnOCR-3-0.8B); [HF blog LightOnOCR-3](https://huggingface.co/blog/lightonai/lightonocr-3); [LightOn research page](https://www.lighton.ai/research/lightonocr-3); [ONNX repo (third-party)](https://huggingface.co/tomngdev/LightOnOCR-3-0.8B-ONNX)

**Other small document VLMs**
- Granite-Docling-258M (IBM): siglip2-base-patch16-512 encoder, pixel-shuffle projector (Idefics3-style) and a Granite 165M LM. It is the successor of SmolDocling-256M (arXiv 2503.11576; SmolDocling internals were not confirmed by search). — [HF card](https://huggingface.co/ibm-granite/granite-docling-258M)
- PaddleOCR-VL, arXiv **2510.14528**: a 0.9B vision encoder + projector + LM with a two-stage layout-then-recognition pipeline. The NaViT-style encoder and ERNIE-4.5-0.3B LM are unverified by search. — [arXiv 2510.14528](https://arxiv.org/html/2510.14528v2)
- MinerU2.5, arXiv **2509.22186**: 1.2B total, with a 675M NaViT (from Qwen2-VL) and a Qwen2-Instruct-0.5B LM. It is decoupled into global layout analysis and local crop recognition. The high-resolution vision side is thus larger than the LM. — [arXiv 2509.22186](https://arxiv.org/html/2509.22186v1)
- dots.ocr: a 1.7B LLM; one source gives about 3B total (1.2B vision encoder + Qwen2.5-1.5B). The size is inconsistent across secondary sources. — [HF dots.ocr](https://huggingface.co/dots-studio/dots.ocr); [E2E Networks blog](https://www.e2enetworks.com/blog/complete-guide-open-source-ocr-models-2025)
- olmOCR-2: Qwen2.5-VL-7B based (about 8B), arXiv **2510.19817**. It uses unit-test rewards (RL), not compression. — [arXiv 2510.19817](https://arxiv.org/pdf/2510.19817)
- GOT-OCR2.0: 580M. Nougat: 250M/350M (secondary listing gives 350M). Nanonets-OCR-s: about 3B (Qwen2.5-VL-3B based; secondary). — [aimadetools listing](https://www.aimadetools.com/blog/best-open-source-ocr-models-2026/); [Ollama Nanonets-OCR-s](https://ollama.com/yasserrmd/Nanonets-OCR-s)
- PP-OCRv6, arXiv **2606.13108**: "From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR Tasks". This is a non-VLM pipeline counterpoint; content not read. — [arXiv 2606.13108](https://arxiv.org/pdf/2606.13108)

**Visual-token / KV compression for documents**
- DeepSeek-OCR, "Contexts Optical Compression", arXiv **2510.18234** (2025). On Fox: 96%+ OCR precision at 9–10× text-to-vision-token compression, about 90% at 10–12×, and **about 60% at 20×**. Secondary sources report it beating GOT-OCR2.0 and MinerU2.0 with fewer than 800 vision tokens per page. The authors frame it as a feasibility study. — [arXiv 2510.18234](https://arxiv.org/pdf/2510.18234); [InfoQ](https://infoq.com/news/2025/10/deepseek-ocr)
- FastOCR, arXiv **2605.17447**: training-free **KV-cache pruning during decoding** for document parsing. On Qwen2.5-VL it keeps 98% of accuracy while attending to 5% of visual tokens per decode step, cutting attention latency 3.0×. — [arXiv 2605.17447](https://arxiv.org/html/2605.17447)

**Text-recognition models**
- TrOCR (arXiv 2109.10282) is a ViT encoder + autoregressive decoder. I found no pruning paper on TrOCR or on scene-text recognizers. DLoRA-TrOCR (arXiv 2404.12734) is PEFT, not pruning. DTrOCR (Fujitake, **WACV 2024**, decoder-only OCR) was found only as a citation, so its content is unverified. — [TrOCR](https://arxiv.org/abs/2109.10282); [DLoRA-TrOCR](https://arxiv.org/html/2404.12734v1)
- Not found: TextHawk, DocOwl token compression (DocOwl2 "High-resolution compressing", unverified), and Nougat/Donut compression papers.

### Inferences
- The user's work (structured pruning + KD of an OCR VLM down to 0.3B/0.1B) appears to be novel, judging by these searches. Vocabulary pruning (LightOn) is a complementary, almost free saving for small decoders, where the embedding and LM head are a large fraction of parameters. It costs non-Latin languages.
- FastOCR's decode-time KV pruning targets the same bottleneck the user identifies (long decode over thousands of image tokens). Note that LightOnOCR-3-0.8B uses about 4.8k image tokens per page at its best setting. For GatedDeltaNet layers the state is fixed-size, so only the 6 (unverified) full-attention layers carry a growing KV cache, and FastOCR-style KV pruning would apply only to those layers.

### Gaps
- No primary source on pruning or distillation of Donut, Nougat, GOT-OCR2.0, dots.ocr, PaddleOCR-VL or MinerU2.5 into smaller variants. The SmolDocling paper content (2503.11576) was not retrieved.

## What is known about relative sensitivity of vision encoder vs decoder to pruning in VLMs (numbers)?

### Takeaway
Direct matched-budget comparisons are scarce. The evidence suggests that (a) mild LLM pruning first breaks vision-language alignment, not language ability; (b) LLaVA-style works exclude the encoder because it is a small share of parameters; and (c) in small VLMs, encoder/LM balance matters, and oversizing the encoder hurts when the LM is tiny.

### Cited Findings
- Huang et al. (IJCV 2026), LLaVA-v1.5-7B: at **15% LLM compression, multimodal scores fall by about 9% while text-only scores fall by about 1%**, so mild pruning mainly disrupts alignment. Degradation accelerates beyond 30%. At **45% compression LLaVA retains 83.99%**, below the authors' 90% fidelity threshold. Finetuning only the MM projector suffices at small ratios. — [arXiv 2604.24380](https://arxiv.org/pdf/2604.24380); [Springer](https://link.springer.com/article/10.1007/s11263-026-02865-5)
- In LLaVA, the LLM has more than 20× the parameters of the vision encoder and projector combined, so it is the main pruning target (medical MLLM compression paper, arXiv 2507.21976). — [arXiv 2507.21976](https://arxiv.org/pdf/2507.21976)
- EfficientLLaVA excludes the visual encoder and projector because pruning them has minimal impact on overall cost (a statement about compute share, not sensitivity). — [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Liang_EfficientLLaVA_Generalizable_Auto-Pruning_for_Large_Vision-language_Models_CVPR_2025_paper.pdf)
- SmolVLM: at LM 135M a larger encoder reduced performance; at 360M it gave +11.6% for +66% parameters. This is evidence that tiny decoders cannot exploit a large encoder. — [arXiv 2504.05299](https://arxiv.org/pdf/2504.05299)
- MinerU2.5 (675M vision + 0.5B LM) and dots.ocr (about 1.2B vision + 1.5B LM) show that document VLMs, unlike LLaVA, often have vision encoders comparable to or larger than the LM. That makes encoder pruning relevant for OCR. — [MinerU2.5](https://arxiv.org/html/2509.22186v1); [E2E blog](https://www.e2enetworks.com/blog/complete-guide-open-source-ocr-models-2025)
- UPop reports that separate (per-modality) vs unified pruning behave notably differently on multimodal tasks, and it lets the search allocate ratios across modalities automatically. Per-modality ratio numbers were not retrieved. — [PMLR](https://proceedings.mlr.press/v202/shi23e.html)
- ECoFLaP: Wanda drops sharply at high sparsity on unimodal EVA-ViT (ImageNet) and FlanT5 (MMLU); ECoFLaP mitigates this. No direct ViT-vs-LM ratio numbers were retrieved. — [Liner review](https://liner.com/review/ecoflap-efficient-coarsetofine-layerwise-pruning-for-visionlanguage-models)

### Inferences
- For OCR, fine-grained glyph detail likely makes the ViT more sensitive than in general VQA. This is not measured in any source found. The user should measure per-component sensitivity (for example, prune the ViT alone vs the decoder alone at matched latency savings) rather than rely on LLaVA-era evidence.
- Because mild pruning hits alignment first, cheap projector or adapter re-tuning after each pruning step (Huang et al.) is a low-cost recovery step to include.

### Gaps
- No paper reports matched-budget sensitivity numbers for the vision encoder vs the LLM in a modern decoder VLM, and none does so for OCR. MULTIFLOW's and UPop's learned per-modality sparsity distributions (which would answer this indirectly) were not retrieved, because the full texts were inaccessible.
