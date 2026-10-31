# D6: Distillation for VLMs and OCR/document models (encoder features, visual tokens, long-sequence outputs)

**Verification legend and network caveat.** arxiv.org, huggingface.co, alphaxiv, openreview.net, CVF open access, Springer, Semantic Scholar, ResearchGate and lighton.ai could not be reached from this session (DNS failure or proxy 403). Only raw.githubusercontent.com worked. So:
- **VERIFIED (GitHub primary)** means I read the authors' own GitHub README or docs file in full.
- **NOT VERIFIED (snippet only)** means the claim comes from web-search snippets or summaries of the paper. I did not read the paper's full text. Numbers marked this way should be checked against the PDF before anyone quotes them.
- arxiv.org, huggingface.co and openreview.net would have to be allowed in the session's network settings to verify the paper-level numbers.

---

## Q1. VLM distillation methods: what losses are used and what numbers do they report?

### Takeaway
General-purpose VLM KD (LLaVA-MoD, LLaVA-KD, Align-KD, EM-KD, CompoDistill, MoVE-KD) mostly combines token-level KL on response logits with a modest auxiliary term: visual-token logits or relations, early-layer attention, or encoder features. Gains over plain SFT run from about +1 to +4.5 average points. The clearest pruning-specific result comes from Huang et al. (GCPR 2025 / IJCV 2026). For pruned MLLMs, SFT plus hidden-state (L2 on intermediate states) distillation gave the best recovery, retaining over 95% of performance with about 5% of the original data.

### Cited Findings

**LLaVA-MoD (arXiv 2408.15881, ICLR 2025)**
- Two-stage "progressive knowledge transfer". (1) Mimic distillation minimizes KL divergence between the student's and teacher's output distributions. (2) Preference distillation uses DPO, so the student learns to "outperform the teacher, especially in hallucination benchmarks." The student is a sparse MoE upcycled from a dense student. With 2B activated params it beats Qwen-VL-Chat-7B by 8.8% average while using 0.3% of the training data and 23% trainable params. — VERIFIED (GitHub primary): [LLaVA-MoD README](https://github.com/shufangxun/LLaVA-MoD)
- Training recipe in the repo docs. The teacher is a LLaVA-1.5-style model with Qwen-2-7B, and teacher and student share the same CLIP-ViT-L/336 encoder, so the encoder is not compressed. Dense-to-dense distillation uses 2.4M general caption and conversation samples. Dense-to-sparse distillation uses 1.4M multi-task samples (VQA, documents, science, OCR). Preference distillation uses 80K samples. The default config is `LOSS_TYPE='only_kd'` (options `kd_lm | only_kd`), so KD alone is used without the LM/CE term. `DISTILL_ALL_TOKENS=False` distills only response tokens. — VERIFIED (GitHub primary): [LLaVA-MoD TRAIN_EVAL.md](https://github.com/shufangxun/LLaVA-MoD/blob/main/docs/TRAIN_EVAL.md)
- Ablation (Table 6 in v2): KD (mimic distillation) averages 60.2 against 55.7 for SFT, with the largest gains on VizWiz, MME and MMBench. The text says "KD surpasses SFT on all benchmarks, achieving an 8.1% average gain." The paper also says cross-entropy is added to the KL loss for the dense-to-sparse stage. Preference distillation cut the Object HalBench response-level hallucination rate from 39.1 to 11.4. — NOT VERIFIED (snippet only): [arXiv 2408.15881v2](https://arxiv.org/html/2408.15881v2)

**LLaVA-KD (arXiv 2410.16236, ICCV 2025)**
- MDist (Multimodal Distillation) transfers the teacher's output distributions on both visual and linguistic tokens. RDist (Relation Distillation) transfers the teacher's modeling of relations among visual tokens. Training has three stages: Distilled Pre-Training (DPT), SFT, and Distilled Fine-Tuning (DFT). The architecture is unchanged, and students are SigLIP-so400m with Qwen1.5-0.5B/1.8B or Qwen2.5-0.5B/1.5B. — VERIFIED (GitHub primary): [LLaVA-KD README](https://github.com/fantasyele/llava-kd)
- Ablation: RDist alone beats MDist alone, and the combination is best. One table row reads about 57.9 average for both losses in DFT, against 55.6 for RDist-only and 55.1 for MDist-only. Full DPT-SFT-DFT gives about 58.0, and the authors say an extra DFT phase "yields comparable performance … but requires increased computational overhead." At 0.5B, LLaVA-KD averages about 55.2 against 53.7 for LLaVA-MoD (supplement Table A5). The slides claim about +1.1% over LLaVA-MoD using 1.2M samples against 5M. At 2B the lead is +0.4% over LLaVA-MoD and +1.4% over Imp-2B. The tables were partly garbled in the snippets. — NOT VERIFIED (snippet only): [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Cai_LLaVA-KD_A_Framework_of_Distilling_Multimodal_Large_Language_Models_ICCV_2025_paper.pdf), [supplement](https://openaccess.thecvf.com/content/ICCV2025/supplemental/Cai_LLaVA-KD_A_Framework_ICCV_2025_supplemental.pdf), [slides](https://iccv.thecvf.com/media/iccv-2025/Slides/227.pdf)
- Teacher-size ablation: "Larger teachers exhibit lower distillation effectiveness than the smaller ones." — NOT VERIFIED (snippet only, reported as from LLaVA-KD): [arXiv 2410.16236](https://arxiv.org/pdf/2410.16236)

**Align-KD (arXiv 2412.01282, CVPR 2025)**
- Teacher is MobileVLM V2 7B and student is MobileVLM V2 1.7B. The paper argues that cross-modal alignment happens mainly in the first LLM layer, so it distills only the text-query-to-vision attention of the first attention layer, plus a lightweight loss. It reports +2.0 average over 6 benchmarks under two training subsets. — NOT VERIFIED (snippet only): [arXiv 2412.01282](https://arxiv.org/pdf/2412.01282), [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Feng_Align-KD_Distilling_Cross-Modal_Alignment_Knowledge_for_Mobile_Vision-Language_Large_Model_CVPR_2025_paper.html)

**EM-KD (arXiv 2511.21106, AAAI 2026)**
- Targets students with fewer vision tokens than the teacher ("unbalanced vision tokens"). It computes the Manhattan distance between teacher and student *vision-token logits* and aligns tokens spatially with Hungarian matching. Vision Semantic Distillation uses reverse KL on the aligned vision-token logits over the vocabulary. Vision-Language Affinity Distillation is the second loss. Setup: Qwen2-0.5B LLM with SigLIP-base, evaluated on DocVQA and TextVQA among others. It claims to beat prior efficient MLLMs "by a large margin", but I did not obtain the numbers. — NOT VERIFIED (snippet only): [arXiv 2511.21106](https://arxiv.org/pdf/2511.21106), [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/39254)

**CompoDistill (arXiv 2510.12184, ICLR 2026)**
- Argues that standard MLLM KD fails to transfer visual perception because of student-teacher *visual attention misalignment*. It adds a Visual ATtention alignment (VAT) module and a Teacher Adapter Fetch (TAF) module, trained in three stages. Student is Qwen1.5-1.8B with SigLIP-so400m, and teacher is a Qwen1.5-4B LLaVA-style model. Fully replacing the student's attention with the teacher's was slightly worse than vanilla LLaVA-KD. I did not obtain the numbers. — NOT VERIFIED (snippet only): [arXiv 2510.12184](https://arxiv.org/pdf/2510.12184), [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10009041)

**Pruning + KD for MLLMs: Huang, Thede, Mancini, Xu, Akata (arXiv 2507.20749, GCPR 2025; extended as arXiv 2604.24380 / IJCV 2026)**
- They prune the *language backbone* of LLaVA-v1.5-7B and Bunny-v1.0-3B (the IJCV version covers three LVLM families, 3B–7B), layerwise or widthwise, then recover with SFT and/or KD. Findings:
  - Widthwise pruning holds up better in low-resource settings.
  - Tuning only the multimodal projector is enough at small compression.
  - **SFT plus hidden-state distillation gives the best recovery across pruning levels.**
  - Recovery reaches over 95% of original performance with about 5% of the original training data.
  - The IJCV version says KD "especially with SFT and an L2 loss on intermediate states" is most effective across compression ratios.
  - Recommended regimes: widthwise-only for extreme low resource; layerwise plus projector tuning for moderate compression; widthwise plus KD for high compression. The thresholds are about 40% in v1 and 30%/45% in the extended version.
  - NOT VERIFIED (snippet only): [arXiv 2507.20749](https://arxiv.org/abs/2507.20749), [arXiv 2604.24380](https://arxiv.org/pdf/2604.24380), [IJCV 2026](https://link.springer.com/article/10.1007/s11263-026-02865-5), [MCML listing](https://mcml.ai/publications/htm+25/)
- An earlier OpenReview version is titled "From Bulk to Budget: Best Practices To Compress Multimodal Large Language Models" (same authors). — NOT VERIFIED (snippet only): [OpenReview](https://openreview.net/forum?id=774F8gF0UO)

**Other VLM KD (seen only in search results)**
- VLsI (arXiv 2412.01822) does "verbalized layers-to-interactions" distillation from large to small VLMs. I have no details. — NOT VERIFIED (title only): [arXiv 2412.01822](https://arxiv.org/pdf/2412.01822)
- LLaVA-CKD (arXiv 2605.10641) does "bottom-up cascaded KD" for VLMs. I have no details. — NOT VERIFIED (title only): [arXiv 2605.10641](https://arxiv.org/pdf/2605.10641)
- Visual Embedding Distillation (arXiv 2412.09585) distills target vision-encoder features into the LLM hidden states through a Perceiver Resampler. — NOT VERIFIED (snippet only): [arXiv 2412.09585](https://arxiv.org/pdf/2412.09585)

### Inferences
- The VLM KD papers that report SFT vs KD ablations show KD beating plain SFT on the same data: +4.5 avg points for LLaVA-MoD, about +2 for Align-KD, and +1–2 for LLaVA-KD over LLaVA-MoD. The user's setup (top-20 logprobs on ~50% of pages, CE elsewhere) is close to LLaVA-MoD's `kd_lm` option and Distil-Whisper's PL+KL.
- Huang et al. is the closest published analogue to the user's plan (prune an existing VLM, then recover). Their result points toward adding an **intermediate hidden-state L2 term** on top of logit KD, especially at high compression. The user's targets (0.8B to 0.3B is about 62% compression, and 0.8B to 0.1B about 88%) go beyond the 30–45% "high compression" regime they studied. Extrapolating that far is risky.
- The LLaVA-KD claim that larger teachers distill less effectively hints at a capacity-gap problem for a 4B to 0.1B transfer. A staged or teacher-assistant route (4B, then 0.8B, then 0.3B, then 0.1B), or using the unpruned 0.8B as teacher for the hidden-state term, is worth an ablation. This is my inference, not a published OCR result.

### Gaps
- I did not obtain exact tables for LLaVA-KD, EM-KD, CompoDistill or Align-KD. None of them evaluates a pure page-OCR task.
- I found no published distillation details for DeepSeek-VL2, the small Qwen2-VL/Qwen2.5-VL models, PaliGemma 2, SmolVLM, TinyLLaVA or Imp. The TinyLLaVA Factory README has no mention of distillation or teachers ([TinyLLaVA Factory README](https://github.com/TinyLLaVA/TinyLLaVA_Factory)). The others were not checked in depth.
- MiniVLM and DistillVLM (2021), "VLM-KD", C2KD, VisionZip and Imp were not covered because of the tool budget.

---

## Q2. Vision-encoder / feature distillation (RADIO, InternViT-300M, CLIP-KD, MoVE-KD): does encoder-feature KD work, and does encoder capacity matter for text?

### Takeaway
Feature-level KD is the standard way to build *small* vision encoders for MLLMs. InternViT-300M was distilled from InternViT-6B with negative cosine similarity on the last-K-layer hidden states, and later refined with an NTP loss. RADIO uses multi-teacher feature distillation, and CLIP-KD finds plain MSE feature mimicry works best. Text reading, however, is very sensitive to encoder capacity and resolution. In RADIOv2.5, swapping ViT-H for ViT-B within the same LLaVA setup drops TextVQA without OCR tokens from 56.93 to 43.96 at 768px.

### Cited Findings
- **InternViT-300M (Mini-InternVL, arXiv 2410.16261):**
  - The student was initialized from CLIP-ViT-L-336px and aligned to InternViT-6B by minimizing **negative cosine similarity between hidden states of the last K transformer layers**. The result is InternViT-300M. — NOT VERIFIED (snippet only): [arXiv 2410.16261](https://arxiv.org/pdf/2410.16261)
  - The InternVL 2.5 report (arXiv 2412.05271) lists InternViT-300M-448px-Distill (0.3B) as trained with a cosine distillation loss. It was later incrementally pre-trained with **NTP loss** on a more diverse mix to give InternViT-300M-448px-V2.5. — NOT VERIFIED (snippet only): [arXiv 2412.05271](https://arxiv.org/pdf/2412.05271)
  - The HF model card names the teacher as InternViT-6B-448px-V1-5. — NOT VERIFIED (snippet only): [InternViT-300M-448px model card](https://huggingface.co/OpenGVLab/InternViT-300M-448px/blob/da8e588511e5f1ca24c78e08599cf3e42e44e95f/README.md?code=true)
- Mini-InternVL-Chat-2B-V1-5 (distilled 300M ViT with InternLM2-1.8B) scores DocVQA test 85.0 and OCRBench 654. Mini-InternVL-4B-1-5 scores DocVQA 87.7 and OCRBench 638. InternVL2-4B scores DocVQA 89.2 and OCRBench 788. These are model-card numbers, not a KD ablation. — NOT VERIFIED (snippet only): [InternVL2-4B model card](https://huggingface.co/OpenGVLab/InternVL2-4B/blob/144479bc70ecd24975d054d3d26975f4694c1193/README.md)
- Current InternVL2.5/3 small models (1B–8B) all use InternViT-300M-448px-V2_5, and the large ones use InternViT-6B. — VERIFIED (GitHub primary): [InternVL README](https://github.com/OpenGVLab/InternVL)
- **AM-RADIO (arXiv 2312.06709, CVPR 2024) / RADIO family:**
  - Distills CLIP variants, DINOv2 and SAM into one backbone. It beats its teachers on ImageNet zero-shot (+6.8%), kNN (+2.39%) and linear-probe segmentation (+3.8%), and improves LLaVA-1.5 by up to 1.5%. E-RADIO is 6–10x faster than CLIP and DINOv2.
  - Later versions use PHI-S distribution balancing for label-free multi-teacher distillation, and the newest teacher set is SigLIP2-g-384, DINOv3-7B and SAM3.
  - The README says the C-RADIOv2-VLM encoder powers Llama Nemotron Nano VL 8B, "#1 on OCR Bench v2" (a vendor claim).
  - VERIFIED (GitHub primary): [RADIO README](https://github.com/NVlabs/RADIO)
- **RADIOv2.5 tech report**, LLaVA-1.5 with Vicuna-7B, TextVQA *without* OCR tokens, which isolates how well the encoder itself reads text:
  - RADIOv2.5-B (ViT-B): 32.43 at 432px, 36.39 at 512px, 43.96 at 768px.
  - RADIOv2.5-L: 42.34, 46.50 and 53.95.
  - RADIOv2.5-H: 47.61, 51.99 and 56.93.
  - With OCR tokens the gap shrinks (B 56.93 vs H 62.39 at 768). GQA and POPE barely change across sizes; for example POPE is 87.54 for B and 87.36 for H at 768.
  - An earlier RADIOv2 "mode switching" problem (features changing abruptly around 704–720px input) made hi-res VLM results poor. It was fixed in v2.5.
  - VERIFIED (GitHub primary): [RADIOv2.5 tech report](https://github.com/NVlabs/RADIO/blob/main/RADIOv2.5_tech_report.md)
- **CLIP-KD (arXiv 2307.12732, CVPR 2024):**
  - "The simplest feature mimicry with MSE loss performs best." Interactive contrastive learning and relation-based distillation also help.
  - Distillation effectiveness correlates with how much teacher-student feature similarity a method achieves.
  - With a ViT-L/14 teacher on LAION-400M, students reach 57.5% (ViT-B/16) and 55.4% (ResNet-50) ImageNet zero-shot, +20.5 and +20.1 over no-KD. On CC3M+12M, MobileViT-S goes from 32.6% to 36.0%.
  - NOT VERIFIED (snippet only): [arXiv 2307.12732](https://arxiv.org/html/2307.12732v2)
- **MoVE-KD (arXiv 2501.01709, CVPR 2025):**
  - Distills multiple vision encoders into one student encoder, using LoRA experts plus attention-based (CLIP [CLS]-attention-weighted) feature distillation.
  - Results with LLaVA-1.5 Vicuna-7B: TextVQA 58.2 for the baseline, 58.3 for v1.0 and 59.6 for v1.1; MMB 64.3, 66.3 and 67.4.
  - With 13B: TextVQA 61.3 for the baseline, 59.7 for v1.0 and 61.1 for v1.1. Both versions are *below* the baseline on TextVQA.
  - Training: pre-training tunes the student encoder's MoLE, adapters and projector; fine-tuning updates everything except the teacher encoders.
  - VERIFIED (GitHub primary): [MoVE-KD README](https://github.com/hey-cjj/MoVE-KD)
- A separate paper reports "better teachers don't make better students" for CLIP-style teachers in VQA. — NOT VERIFIED (snippet only): [arXiv 2511.17886](https://arxiv.org/html/2511.17886v1)

### Inferences
- If the user prunes the 12-block ViT (by depth or width), the best-documented recipe is InternViT-style: cosine (or MSE, per CLIP-KD) alignment of the last-K-layer hidden states to the *unpruned* ViT, followed by end-to-end NTP/KD training through the LLM. The unpruned 0.8B model's ViT is the natural feature teacher. If the 4B teacher's ViT has a different width, a projector is needed (as in RADIO's adaptors and MoVE-KD's encoder adapters).
- The RADIOv2.5 B-vs-H TextVQA (no-OCR-tokens) gap of about 13 points at 768px, against near-zero gaps on GQA and POPE, suggests that OCR is the task most hurt by shrinking the encoder. ViT pruning should therefore be gentler than decoder pruning and checked on text-dense benchmarks, not general VQA. MoVE-KD's flat or negative TextVQA deltas point the same way: encoder-feature KD tuned for general VQA does not automatically help reading.
- Resolution interacts with encoder capacity. The smaller RADIO models gain much more from 432 to 768px. LightOnOCR-3-0.8B runs at 2048px, so any pruned ViT must be validated at the full serving resolution.

### Gaps
- I did not obtain the value of K, the step count or the data size for the InternViT-300M distillation. I found no ablation of "feature KD vs no feature KD" for the InternViT student on OCR benchmarks.
- I found no paper reporting OCR or document-benchmark recovery for a *structurally pruned* ViT inside an MLLM with feature KD. The searches surfaced only visual-*token* pruning work (EM-KD; "Recoverable Compression", AAAI; IDPruner) — NOT VERIFIED (snippet only): [Recoverable Compression, AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32229/34384).
- SigLIP distillation details (for example the SigLIP 2 small variants) were not checked.

---

## Q3. OCR/document-specific distillation: how are OCR VLMs trained from teachers?

### Takeaway
In the OCR VLMs reviewed here, published "distillation" is overwhelmingly **sequence-level** KD: teacher transcriptions are used as training labels. LightOnOCR-1 used Qwen2-VL-72B, LightOnOCR-2 used Qwen3-VL-235B-A22B, olmOCR used GPT-4o, PaddleOCR-VL used ERNIE-4.5-VL and Qwen2.5-VL refining PP-StructureV3 pseudo-labels, and DeepSeek-OCR used a GOT-OCR2.0 / PaddleOCR "model flywheel". None of these reports token-level logit KD. The newer step is RL with verifiable rewards (olmOCR 2 unit tests, LightOnOCR-2 RLVR), which notably reduces repetition loops.

### Cited Findings
- **LightOnOCR (arXiv 2601.14251; LightOnOCR-1B and LightOnOCR-2-1B):**
  - Both versions are "trained on large-scale document OCR corpora built primarily through distillation": a strong VLM teacher transcribes rendered PDF pages into ordered Markdown with LaTeX (and HTML tables in the 0126 mix).
  - LightOnOCR-1 used transcriptions from **Qwen2-VL-72B-Instruct**. LightOnOCR-2 switched to **Qwen3-VL-235B-A22B-Instruct** for better supervision quality and cleaner formatting, with a training mix about 2.5x larger (more scans, French, scientific).
  - LightOnOCR-2-1B adds RLVR, which "improves overall performance and reduces failures such as repetition loops".
  - Scores: 83.2 ± 0.9 on olmOCR-Bench, excluding headers and footers. The RLVR reward favored keeping headers and footers, which lowers the headers_footers score under the original metric. Throughput is 5.71 pages/s on one H100.
  - NOT VERIFIED (snippet only): [arXiv 2601.14251](https://arxiv.org/html/2601.14251v1), [LightOnOCR-mix-0126 dataset](https://huggingface.co/datasets/lightonai/LightOnOCR-mix-0126)
- **LightOnOCR-3:**
  - Released as 0.8B, 1B and 4B. 0.8B and 4B use Qwen3.5's VL architecture and 2048px pages; 1B keeps the LightOnOCR-2 design at 1540px. They support grounding mode (layout boxes). The recommended sampling is temperature 0.1, not greedy. — VERIFIED (GitHub primary): [LightOnOCR README](https://github.com/lightonai/LightOnOCR)
  - LightOn's page reports olmOCR-Bench 86.3 for 4B and 85.5 for 0.8B. I found nothing on whether LightOnOCR-3 itself used distillation or pruning. — NOT VERIFIED (snippet only): [LightOn research page](https://www.lighton.ai/research/lightonocr-3), [HF blog](https://huggingface.co/blog/lightonai/lightonocr-3)
- **olmOCR (arXiv 2502.18443, 2025):**
  - Fine-tuned Qwen2-VL-7B-Instruct on about 250–260K pages from over 100K PDFs labeled with **GPT-4o** (olmOCR-mix-0225). Converts 1M pages for about $176. — NOT VERIFIED (snippet only): [ICML 2025 listing](https://icml.cc/virtual/2025/48195), [Simon Willison summary](https://simonwillison.net/2025/Feb/26/olmocr/)
  - The repo changelog says olmOCR-Bench launched at 77.4 (May 2025), v0.2.1 scored +3 points (July 2025), and olmOCR-2 (Oct 2025) "boosts olmOCR-bench score by ~4 points using synthetic data and introduces RL training". — VERIFIED (GitHub primary): [olmocr README](https://github.com/allenai/olmocr)
  - Training data format: single-page PDF plus a markdown file with YAML front-matter (language, rotation, is_table, is_diagram). Released configs train Qwen2.5-VL for 1–2 epochs. — VERIFIED (GitHub primary): [olmocr train README](https://github.com/allenai/olmocr/blob/main/olmocr/train/README.md)
- **olmOCR 2 (arXiv 2510.19817):**
  - SFT of Qwen2.5-VL-7B-Instruct on olmOCR-mix-1025, then one epoch of GRPO (HF TRL) on olmOCR2-synthmix-1025: 2,186 synthetic pages rendered from clean HTML, with 30,381 binary unit tests derived from the HTML.
  - 28 completions are sampled per document. Auxiliary rewards cover clean termination and metadata.
  - Scores 82.4 ± 1.1 on olmOCR-Bench, which the report describes as "+14.2 points" over the initial release. A German news summary instead says "almost 4 points" (different baseline; it matches the repo's "~4 points" for the v0.4.0 step).
  - NOT VERIFIED (snippet only): [arXiv 2510.19817](https://arxiv.org/pdf/2510.19817), [Ai2 blog](https://allenai.org/blog/olmocr-2)
- **PaddleOCR-VL (arXiv 2510.14528):**
  - PaddleOCR-VL-0.9B is a NaViT-style dynamic-resolution encoder with ERNIE-4.5-0.3B. — VERIFIED (GitHub primary): [PaddleOCR README](https://github.com/PaddlePaddle/PaddleOCR)
  - The annotation pipeline uses PP-StructureV3 pseudo-labels as prompts for ERNIE-4.5-VL and Qwen2.5-VL to refine. Then come hallucination filtering and rule checks (n-gram, HTML validity, rendering success), plus synthetic data for hard cases (messy handwriting, dense or blurry text). The result is about 20M text samples, and over 30M samples in total.
  - NOT VERIFIED (snippet only): [arXiv 2510.14528](https://arxiv.org/pdf/2510.14528), [hyper.ai summary](https://hyper.ai/fr/papers/2510.14528)
- **DeepSeek-OCR (arXiv 2510.18234):**
  - About 30M PDF pages with coarse and fine annotations. Recognition data comes from a "model flywheel": GOT-OCR2.0 is trained on fitz-extracted patches, then used to label patches after layout processing (600K samples).
  - Scene-text images come from LAION and Wukong, labeled by PaddleOCR (10M each for Chinese and English).
  - NOT VERIFIED (snippet only): [arXiv 2510.18234](https://arxiv.org/pdf/2510.18234)
- **MinerU2.5 (arXiv 2509.22186):**
  - A decoupled 1.2B model: layout on downsampled pages, then recognition on native-resolution crops. Its "data engine" generates pretraining and SFT corpora; a third-party summary puts SFT at about 630K examples.
  - I found no confirmation of a larger-model teacher.
  - NOT VERIFIED (snippet only): [arXiv 2509.22186](https://arxiv.org/abs/2509.22186)
- **SmolDocling (arXiv 2503.11576):**
  - SmolVLM-256M (secondary source: SigLIP 93M plus SmolLM-2 135M), outputting DocTags.
  - The data augments existing document datasets with new annotations and new synthetic sets (code, charts). The vision encoder is frozen at first while the model aligns to DocTags.
  - I found no teacher-KD details.
  - NOT VERIFIED (snippet only): [arXiv 2503.11576](https://arxiv.org/pdf/2503.11576)
- **DocKD (arXiv 2410.03061, EMNLP 2024):**
  - An LLM teacher generates prompt-answer pairs for visual document understanding (VQA, entity extraction, classification) from OCR text, enriched with document knowledge (key-value pairs, layout, descriptions).
  - Students trained only on DocKD data are comparable to human-annotation-trained students in-domain and clearly better out-of-domain. This is sequence-level data distillation, not logit KD.
  - A blog names the teacher as Claude-2 (unconfirmed).
  - NOT VERIFIED (snippet only): [arXiv 2410.03061](https://arxiv.org/pdf/2410.03061)
- **DistilDoc (arXiv 2406.08226)** is KD for visually-rich document applications. I have no details. — NOT VERIFIED (title only): [arXiv 2406.08226](https://arxiv.org/pdf/2406.08226)
- **Handwriting / text-line OCR KD:**
  - HTR-JAND (arXiv 2412.18524) uses KL on temperature-softened logits plus an auxiliary feature loss for a compact handwriting-recognition student. — NOT VERIFIED (snippet only): [arXiv 2412.18524](https://arxiv.org/pdf/2412.18524)
  - ICFHR 2018 (MSRA) used teacher-student compression of CNN-DBLSTM (CTC) OCR and handwriting models; the student came close to teacher error rates. — NOT VERIFIED (slides snippet): [ICFHR 2018 slides](https://iapr.org/archives/icfhr2018/SlidesPosters/Slides-Paper22.pdf)

### Inferences
- The user's design (top-20 teacher logprobs on ~50% of pages, CE on teacher labels elsewhere) is *richer* than any published page-level OCR VLM pipeline found here, which all use only teacher text. There is no OCR-specific evidence on how much logit KD adds over sequence-level KD. The closest evidence is Distil-Whisper (Q4) and LLaVA-MoD (KD beats SFT by 4.5 avg points on general VQA).
- Every pipeline applies **label-quality filtering**: PaddleOCR-VL's hallucination and n-gram, HTML and render checks; Distil-Whisper's WER filter; LightOnOCR-2's teacher upgrade for "cleaner formatting". For 1,500-token pages, teacher hallucinations and repetitions would be copied by the student, so similar filters (n-gram repetition, HTML/LaTeX validity, agreement with the PDF text layer where one exists) are directly applicable.
- Both olmOCR 2 and LightOnOCR-2 apply RLVR *after* SFT/KD, and LightOnOCR-2 states it reduces repetition loops. Small pruned students are the models most prone to looping on long outputs, so a short RLVR or unit-test phase after KD recovery is a published, OCR-specific option.

### Gaps
- I could not verify LightOnOCR-1/2 hyperparameters, data size, or whether any logit-level KD was used (the paper says "distillation" in the sense of teacher-generated corpora).
- dots.ocr and GOT-OCR2.0 training-data provenance was not found (the READMEs say nothing about teachers). Granite-Docling and Nougat were not checked.
- I found no paper that distills a full-page OCR VLM with token-level KL.

---

## Q4. Long-sequence transcription KD analogue: Distil-Whisper (arXiv 2311.00430) and related

### Takeaway
Distil-Whisper is the most transferable recipe. It combines pseudo-label CE (sequence-level KD) with word-level KL (α_PL=1.0, α_KL=0.8) and filters pseudo-labels by WER against ground truth (10% threshold optimal for English). The student copies the teacher encoder and keeps only 2 maximally-spaced decoder layers. The student came within 1% WER on out-of-distribution data and slightly *beat* the teacher on long-form audio because it hallucinates less. A hidden-state MSE term was tried and dropped.

### Cited Findings
- **Headline results (repo):** distil-large-v3 has 756M params, 6.3x relative latency, short-form WER 9.7 and long-form WER 10.8. distil-large-v2 has 756M, 5.8x, 10.1 and 11.6. distil-medium.en has 394M, 6.8x, 11.1 and 12.4. distil-small.en has 166M, 5.6x, 12.1 and 12.8. The README describes the family as 6x faster, 49% smaller and within 1% WER on out-of-distribution sets; distil-small.en is within 4% WER of large-v3. OpenAI's Whisper Turbo "leverages the same principles". — VERIFIED (GitHub primary): [distil-whisper README](https://github.com/huggingface/distil-whisper)
- **Training recipe (repo docs):**
  - **Student init:** maximally spaced teacher layers are copied (per DistilBART); for example the 2 decoder layers come from teacher layers 1 and 32, and all 32 encoder layers are kept.
  - **Loss:** a weighted sum of CE and KL.
  - **`wer_threshold`:** samples where the normalized pseudo-label vs ground truth WER exceeds the threshold are discarded, to avoid "training the student model on pseudo-labels where Whisper hallucinated." "A WER threshold of 10% provides the optimal trade-off" for English; multilingual thresholds should follow the teacher's test WER.
  - **Greedy vs beam:** greedy pseudo-labels were "comparable" to beam, "likely because the WER filter compensates for the lower quality pseudo-labels."
  - **Audio packing:** concatenating audio to 30s improves sequential long-form WER and throughput.
  - **`condition_on_prev_probability` about 0.2** (OpenAI used 50%) is needed so the student works with the sequential long-form algorithm.
  - **`freeze_encoder`:** when the encoder is copied unchanged, the teacher's encoder hidden states are reused, saving memory (up to 2x batch size).
  - **Data scale:** 21k hours and 80k steps for the paper; more than 13k hours were needed for convergence on English; at least 1k hours and 50k steps recommended. A 15h demo (about 1.5h after filtering) ended at 76% WER.
  - **LR schedule:** linear for runs over 5k steps, constant-with-warmup for short runs.
  - VERIFIED (GitHub primary): [distil-whisper training README](https://github.com/huggingface/distil-whisper/blob/main/training/README.md)
- **Paper details:**
  - α_KL=0.8 and α_PL=1.0 (following Shleifer & Rush 2020).
  - Pseudo-labels give "a sequence-level distillation signal" and consistent formatting; KL is word-level.
  - An MSE hidden-state loss was tried and dropped, because it needs per-layer hidden states and PL+KL "trained faster and performed better" in the time budget.
  - Final: 5.8x faster, 51% fewer params, within 1% WER OOD, and on long-form 0.1% WER *better* than Whisper.
  - It serves as a speculative-decoding draft for about 2x speed-up with identical outputs.
  - Hallucination is measured with repeated 5-gram duplicates and insertion error rate on 4 OOD long-form sets.
  - NOT VERIFIED (snippet only): [arXiv 2311.00430](https://arxiv.org/pdf/2311.00430)
- Groq's blog states "1.3x fewer repeated 5-gram word duplicates and 2.1% reduction in insertion error rate vs Whisper large-v3" for distil-large-v3-en. This is a vendor source, absolute vs relative is unclear, and it is not confirmed in the paper. — NOT VERIFIED (vendor snippet): [Groq blog](https://groq.com/blog/distil-whisper-is-now-available-to-the-developer-community-on-groqcloud-for-faster-and-more-efficient-speech-recognition)

### Inferences
- Mapping to the user's setup:
  1. PL+KL with α_KL about 0.8 and α_PL 1.0 is a tested starting weight for transcription. It matches the user's mixed top-20-logprob and CE plan.
  2. Filtering teacher labels by agreement with an independent reference is the OCR analogue of the WER filter. The PDF text layer, or agreement between two teacher samples, could serve as that reference (normalized edit distance, per page).
  3. Distil-Whisper's large *long-form* gain came from *fewer hallucinations/repetitions*, which is exactly the failure mode of small OCR students on 1,500-token pages.
  4. Keeping the teacher's encoder frozen and shrinking only the decoder worked. By analogy, pruning the LM decoder hard while pruning the ViT lightly (or not at all) is the lower-risk path, consistent with Q2.
  5. Distil-Whisper dropped hidden-state MSE for cost reasons, while Huang et al. found hidden-state L2 helped pruned MLLMs. The evidence is mixed and depends on the setup; the deciding factor may be whether the student is *pruned from* the teacher (aligned representations), as in Huang et al. and the user's case, or a fresh smaller architecture.
- Distil-Whisper needed over 13k hours (very large data) to converge from a 2-layer-decoder init. For an 88%-compressed 0.1B OCR student, data and steps may need to be far larger than for the 0.3B target.

### Gaps
- I did not obtain the Distil-Whisper ablation tables (PL-only vs KL-only vs both; WER-threshold sweep values).
- I found no long-sequence KD paper for handwriting or page OCR with token-level KL on outputs over 1k tokens.

---

## Q5. Do VLM/OCR works report that hidden-state / encoder-feature KD helps when the vision encoder is compressed?

### Takeaway
There is only indirect evidence. The encoder was distilled feature-wise with cosine loss and then fine-tuned with NTP (InternViT-6B to 300M), and multi-teacher feature KD produced strong small encoders (RADIO, with B/L/H sizes). But I found **no published ablation** that prunes an MLLM's ViT and measures OCR or document recovery with vs without feature KD. Huang et al.'s hidden-state KD finding concerns the *LLM backbone*, not the encoder.

### Cited Findings
- InternViT-300M: negative cosine on last-K-layer hidden states from InternViT-6B, then NTP refinement (V2.5). It is used as the encoder for all small InternVL2.5/3 models. — NOT VERIFIED (snippet only): [arXiv 2410.16261](https://arxiv.org/pdf/2410.16261), [arXiv 2412.05271](https://arxiv.org/pdf/2412.05271); VERIFIED usage: [InternVL README](https://github.com/OpenGVLab/InternVL)
- RADIO: distilled ViT-B/L/H students all improve with resolution. The text-reading gap between B and H is large (TextVQA without OCR tokens at 768px: 43.96 vs 56.93), while the general-VQA gap is small. — VERIFIED (GitHub primary): [RADIOv2.5 tech report](https://github.com/NVlabs/RADIO/blob/main/RADIOv2.5_tech_report.md)
- MoVE-KD: attention-weighted feature KD into one encoder gave +0.1/+1.4 TextVQA at 7B and −1.6/−0.2 at 13B (v1.0/v1.1). — VERIFIED (GitHub primary): [MoVE-KD README](https://github.com/hey-cjj/MoVE-KD)
- CLIP-KD: MSE feature mimicry is the strongest single CLIP distillation loss. — NOT VERIFIED (snippet only): [arXiv 2307.12732](https://arxiv.org/html/2307.12732v2)
- Huang et al.: SFT plus hidden-state (L2 intermediate) distillation is best for recovering pruned MLLM *language backbones* (over 95% performance with 5% data). — NOT VERIFIED (snippet only): [arXiv 2507.20749](https://arxiv.org/abs/2507.20749), [IJCV 2026](https://link.springer.com/article/10.1007/s11263-026-02865-5)
- EM-KD distills *vision-token logits* (reverse KL after Hungarian matching) when the student has fewer visual tokens; evaluated on DocVQA and TextVQA among others. — NOT VERIFIED (snippet only): [arXiv 2511.21106](https://arxiv.org/pdf/2511.21106)

### Inferences
- For a pruned ViT, a reasonable approach is a two-stage recipe in the style of InternViT-300M:
  1. Feature-align the pruned ViT (and projector outputs) to the unpruned 0.8B ViT with cosine or MSE on the last K layers plus the projector output. This is cheap, needs no text labels, and can use any page images.
  2. Recover end-to-end with logit KD/CE through the decoder.
  Using the unpruned student-family ViT as the feature teacher avoids the dimension mismatch with the 4B teacher. This is an inference, not a tested OCR result.
- Because text reading degrades much faster than general VQA when the encoder shrinks (RADIO), evaluation after ViT pruning should use olmOCR-Bench, OmniDocBench or similar dense-text sets at full 2048px resolution.
- Visual-token logit distillation (EM-KD style) is an alternative if the user reduces visual tokens (for example lower resolution or token merging) instead of ViT weights.

### Gaps
- I found no paper reporting numbers for "ViT pruned inside an OCR VLM, with vs without feature KD". This is a genuine gap in the literature found here, not a retrieval failure I could confirm, because the paper texts could not be fetched.
- Exact Mini-InternVL ablations (with vs without the distilled encoder on DocVQA and OCRBench) were not obtained. The "90% performance with 5% params" title claim was not checked against tables.
