# Zero-shot pruning importance vs post-recovery quality; per-unit-type recoverability

Method note: arxiv.org, huggingface.co and api.semanticscholar.org could not be reached from this session (DNS/egress blocked). Everything below was checked through web-search snippets of the primary pages (arXiv abs/html/pdf, NeurIPS/ICLR/IJCAI proceedings, GitHub). Where the snippets came from a secondary summary (Medium, Liner, Lacuna, Emergent Mind, Latent Space), that is marked. No full tables were read, so a number is only as reliable as the snippet it came from. Items marked **unverified** need checking against the PDF before they are cited in the paper.

## Q1: Who has shown that zero-shot (pre-recovery) importance mis-ranks pruning choices after recovery/retraining/distillation, what did they compare, and did anyone explain why?

### Takeaway
A depth-vs-width ranking flip after recovery has been shown directly by NVIDIA Minitron (NeurIPS 2024). Depth-pruned candidates look better right after pruning, width-pruned ones win after a short retraining (about 200 steps), and Minitron therefore ranks candidates only after about 1.8B tokens of light retraining. Two VLM studies (Huang et al. 2025, GCPR 2025 / IJCV 2026) report a related, ratio-dependent flip: widthwise wins with no recovery, but with recovery layerwise wins at ≤30% compression and widthwise at ≥40–45%. In the CNN era, Laurent et al. 2020 measured a weak rank correlation between pruning-time loss and post-fine-tuning accuracy (Spearman ρ ≈ 0.20–0.27). Liu et al. 2019 and Mittal et al. 2018/2019 showed the criterion or inherited weights matter little after enough retraining. No source found gives a rigorous theory of why the ranking flips. The proposed mechanisms are:
- plasticity / architecture-not-weights (Liu 2019; Mittal);
- gradient or loss-landscape alignment with the parent as a proxy for trainability (TraceNAS 2026);
- importance rankings drifting with pruning ratio, with recovery repairing "form rather than knowledge" (2610.15574);
- explicit per-sublayer "tunability" (SPRINT, IJCAI 2025).

### Cited Findings

**Direct evidence of a ranking flip (LLM, depth vs width, distillation)**
- **Minitron: "Compact Language Models via Pruning and Knowledge Distillation"** (Muralidharan, Sreenivas, et al., NVIDIA; arXiv 2407.14679; NeurIPS 2024). The ID resolves correctly to this paper.
  - Snippet from the NeurIPS PDF: "pruning the width yields a better model than pruning the depth" for models ≤15B, "though depth-pruned models look better right after pruning. The superiority of width pruning only reveals itself after a short period of retraining." It also reports "a flip in the ranking prior to 200 steps of retraining." — [NeurIPS 2024 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/4822991365c962105b1b95b1107d30e5-Paper-Conference.pdf); [author PDF](https://www.jankautz.com/publications/Minitron_NeurIPS24.pdf)
  - Answer to the user's "LM loss right after pruning is not predictive?": yes, in practice. Candidate architectures that fit the budget get "lightweight retraining (~1.8B tokens)", which, in the authors' words, "stabilizes relative rankings and helps us find a more accurate candidate to train further." — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/4822991365c962105b1b95b1107d30e5-Paper-Conference.pdf); ~1.8B figure via [Medium summary](https://ritvik19.medium.com/papers-explained-208-minitron-e55ea374d9dd) (secondary)
  - Stated best practices: "Prefer width pruning over depth for the model scales we consider (≤15B)" and "Use (logit+intermediate state+embedding) distillation when depth is reduced significantly." — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/4822991365c962105b1b95b1107d30e5-Paper-Conference.pdf)
  - Unit types compared: depth (layers) vs width (MLP neurons, attention heads, embedding channels), on Nemotron-4 15B → 8B/4B.
  - **Unverified:** the exact figure and table numbers (loss values before and after retraining for the ranked candidates).
  - A 2026 re-evaluation (2606.14150 v3) states it reimplemented Minitron's search by pruning Llama-3.1-8B and scoring validation loss on the calibration set *without* retraining. That departs from Minitron's own retraining-based ranking, so watch for this when citing reproductions. — [2606.14150 v3 HTML](https://arxiv.org/html/2606.14150v3)
- **Llama-3.1-Minitron / "LLM Pruning and Distillation in Practice: The Minitron Approach"** (NVIDIA; arXiv 2408.11796; tech report 2024).
  - Width beats depth at equal parameters after distillation: Llama-3.1-Minitron-4B width vs depth gives MMLU ≈60.5 vs 58.7 and GSM8K ≈41.2 vs 16.8. Depth is faster (TensorRT-LLM ≈2.7× vs ≈1.8× over Llama-3.1-8B). Numbers via [Medium](https://ritvik19.medium.com/papers-explained-209-minitron-approach-in-practice-6b473f67328d) and [Liner](https://liner.com/review/llm-pruning-and-distillation-in-practice-minitron-approach) (secondary); speedup from [arXiv v3 HTML](https://arxiv.org/html/2408.11796v3). Primary-table check: **unverified**.
  - A second loss-vs-downstream mismatch inside depth pruning: LM validation loss and perplexity favour dropping non-contiguous layers, but downstream accuracy (Winogrande) favours one contiguous block of middle layers. The paper picks the contiguous block whose removal hurts Winogrande least. — [Lacuna summary](https://lacuna.tiptreesystems.com/work/llm-pruning-and-distillation-in-practice/wrk_bace53a2b062b2cf8ad9293c6117db2a) (secondary; **unverified** against the PDF)

**VLM evidence (closest to the user's setting): ratio-dependent depth/width flip once recovery is applied**
- **Huang, Thede, Mancini, Xu, Akata, "Investigating Structural Pruning and Recovery Techniques for Compressing Multimodal Large Language Models: An Empirical Study"** (arXiv 2507.20749; GCPR 2025, Springer LNCS).
  - Setup: layerwise vs widthwise pruning of the *language backbone* of MLLMs, recovered by SFT and/or KD.
  - Findings:
    - Widthwise is better when no recovery is possible.
    - With recovery, layerwise is slightly better below 30% compression and widthwise is better above 40%.
    - Fine-tuning only the projector is enough below 20% compression.
    - SFT + hidden-state distillation is best overall.
    - About 5% of the original data keeps over 95% of performance.
  - — [arXiv abs](https://arxiv.org/abs/2507.20749); [Springer](https://link.springer.com/chapter/10.1007/978-3-032-12840-9_21)
- **Huang et al., "Structural Pruning of Large Vision Language Models: A Comprehensive Study on Pruning Dynamics, Recovery, and Data Efficiency"** (arXiv 2604.24380; IJCV 2026, extended version of the GCPR 2025 oral).
  - Models: LLaVA-v1.5-7B, InternVL-Chat-4B, Bunny-v1-3B.
  - "Widthwise pruning consistently outperforms layerwise pruning when the model is not finetuned." With recovery, "layerwise pruning excels for smaller compression ratios (≤30%), while widthwise pruning performs better at higher ratios (≥45%)." Layerwise always has lower latency.
  - Scope: the vision encoder is **not** pruned (language backbone only).
  - This is a direct example of zero-shot ranking (width > depth) disagreeing with post-recovery ranking at low ratios (depth > width). It matches the direction of the user's observation that layers are more recoverable than MLP neurons.
  - — [arXiv HTML](https://arxiv.org/html/2604.24380); [IJCV](https://link.springer.com/article/10.1007/s11263-026-02865-5); [code](https://github.com/YiranHuangIrene/VLMCompression)

**Rank correlation between pre-recovery damage and post-recovery quality (quantified)**
- **Laurent, Ballas, George, Vincent, et al., "Revisiting Loss Modelling for Unstructured Pruning"** (arXiv 2006.12279; 2020; CNNs, unstructured).
  - Performance after fine-tuning "do[es] not correlate significantly with the gap" (the loss or quality gap at pruning time).
  - Spearman ρ = 0.27 (VGG11) and ρ = 0.20 (PreActResNet18).
  - This is the cleanest published number showing that a pre-recovery criterion barely predicts the post-recovery outcome.
  - — [arXiv PDF](https://arxiv.org/pdf/2006.12279) (via search snippet; **verify** exact wording and whether the "gap" is loss before vs after pruning)
- **"The Fragility and Dynamics of LLM Pruning-Recovery Steps"** (Salim et al.; arXiv 2610.15574; Oct 2026, preprint).
  - Varies stage order, pruning and recovery data, and objective alignment.
  - Reports Spearman rank correlation between importance scores before and after pruning, for attention heads and MLP channels, by pruning ratio and criterion. The ranking shifts more as the ratio grows.
  - Matching the recovery loss to the pruning loss gives no measurable benefit.
  - In-domain or mixed data gives the best pruning signal; instruction data gives the best recovery signal.
  - The authors suggest recovery mostly repairs "form" rather than "knowledge."
  - Correlation values: **unverified** (not shown in the snippets).
  - — [arXiv abs](https://arxiv.org/abs/2610.15574); [HTML](https://arxiv.org/html/2610.15574)
- **"Reassessing Layer Pruning in LLMs: New Insights and Methods"** (arXiv 2411.15558; 2024).
  - Fine-tuning only the LM head plus the last 1–3 layers recovers best after layer pruning.
  - A critique (Pith review) notes that its ranking of layer-selection metrics may depend on the recovery method, i.e. metric rankings may not carry over between recovery regimes.
  - — [alphaXiv](https://www.alphaxiv.org/abs/2411.15558); [Pith review](https://pith.science/paper/2411.15558) (secondary)

**CNN-era "criterion barely matters after fine-tuning" results (2018–2020)**
- **Liu, Sun, Zhou, Huang, Darrell, "Rethinking the Value of Network Pruning"** (arXiv 1810.05270; ICLR 2019).
  - "For all state-of-the-art structured pruning algorithms we examined, fine-tuning a pruned model only gives comparable or worse performance than training that model with randomly initialized weights."
  - "The pruned architecture itself, rather than a set of inherited 'important' weights, is what leads to the efficiency benefit." In other words, pruning works as architecture search.
  - — [arXiv](https://arxiv.org/abs/1810.05270v2); [ICLR](https://iclr.cc/virtual/2019/poster/877)
- **Mittal, Bhardwaj, Khapra, Ravindran, "Recovering from Random Pruning: On the Plasticity of Deep CNNs"** (arXiv 1801.10447; WACV 2018) and the journal version **"Studying the Plasticity in Deep Convolutional Neural Networks using Random Pruning"** (arXiv 1812.10240; Machine Vision and Applications 2019).
  - Randomly pruning 25–50% of filters in VGG-16 / ResNet-50, then fine-tuning, matches state-of-the-art importance-based pruning.
  - The authors attribute this to "the inherent plasticity of deep neural networks," not to the criterion.
  - Faster R-CNN: a 74% fps gain at the same accuracy.
  - — [arXiv 1801.10447](https://arxiv.org/pdf/1801.10447); [arXiv 1812.10240 (ar5iv)](https://ar5iv.arxiv.org/html/1812.10240)
- **Blalock, Gonzalez Ortiz, Frankle, Guttag, "What is the State of Neural Network Pruning?"** (arXiv 2003.03033; MLSys 2020).
  - Correction to the user's framing: Blalock et al. do **not** conclude that magnitude or zero-shot criteria are no better than random.
  - Their reliable findings are that magnitude pruning compresses substantially without accuracy loss, and that "many pruning methods outperform random pruning."
  - Their main point is the lack of standard benchmarks (meta-analysis of 81 papers; ShrinkBench).
  - — [arXiv](https://arxiv.org/abs/2003.03033); [MLSys](https://proceedings.mlsys.org/paper_files/paper/2020/hash/6c44dc73014d66ba49b28d483a8f8b0d-Abstract.html)

**Recovery budget washes out the initialization (LLM, 2026)**
- **Xu, Lu, Li, Zhu, Sun, Liu, "Small LLMs: Pruning vs. Training from Scratch"** (arXiv 2606.14150; June 2026; Princeton zlab). The ID resolves correctly.
  - Setup: Llama-3.1-8B parent; six pruning methods (structured and sparse); ratios 0.5–0.8.
  - Equal *training* tokens: pruned initialization beats random, but the advantage shrinks as the retraining budget and pruning ratio grow, nearly vanishing at the highest ratio.
  - Equal *total* tokens: scratch training matches or beats coarse structured pruning, while fine-grained sparse pruning keeps an edge.
  - Implication: the value of what zero-shot importance preserves decays with recovery budget, and decays faster for coarse units.
  - — [arXiv PDF](https://arxiv.org/pdf/2606.14150); [code](https://github.com/zlab-princeton/pruning-vs-scratch). Detail numbers come from hyper.ai summaries and are **unverified**.

**Proxies designed to predict post-recovery quality (implicit admission that zero-shot loss mis-ranks)**
- **TraceNAS: "Zero-shot LLM Pruning via Gradient Trace Correlation"** (Malettira, Nagaraj, Roy, Negi, Roy; arXiv 2602.02891; Feb 2026). The ID resolves correctly.
  - Training-free NAS over depth and width jointly.
  - It scores candidates with a scale-invariant correlation between the pruned model's and the parent's gradients on calibration data. The stated mechanism is that keeping "loss landscape alignment" with the parent makes a better starting point for post-pruning training.
  - About 8.5 GPU-hours on one GPU, roughly 10× fewer GPU-hours than training-aware search.
  - A mirror reports 62.81% average on 8 benchmarks after 20B recovery tokens, vs 62.24% for DarwinLM and 60.86% for Sheared-LLaMA (**unverified**).
  - — [arXiv](https://arxiv.org/abs/2602.02891); [HTML](https://arxiv.org/html/2602.02891)
- **Puzzle: "Distillation-Based NAS for Inference-Optimized LLMs"** (NVIDIA; arXiv 2411.19146; 2024/ICML 2025 per NVIDIA, **venue unverified**).
  - Scores each block variant by a "replace-1-block" metric (KL divergence, LM loss or downstream accuracy) after blockwise local distillation (BLD). Architecture quality is assumed additive: the sum of replace-1-block scores.
  - Global KD (GKD) is then needed to fix inter-block incompatibility, because BLD trains blocks on parent activations.
  - No reported correlation between pre-GKD summed scores and post-GKD quality was found in the snippets. The additivity assumption plus the need for GKD suggests these can diverge (inference).
  - — [arXiv HTML](https://arxiv.org/html/2411.19146); [Pith review](https://pith.science/paper/2411.19146)
- **LOOP: "Recovery-Aligned Iterative Layer Pruning via On-Policy Distillation"** (Xu, Shen, et al.; arXiv 2610.25679; Oct 2026).
  - Argues that decoupling layer selection from recovery "limits the use of recovery information to guide selection."
  - Alternates single-layer pruning with short on-policy distillation, and uses recovery signals to choose the next layer.
  - Beats one-shot baselines across scales and ratios; the ablation says the gain is not from OPD alone but from recovery-oriented selection.
  - — [arXiv](https://arxiv.org/abs/2610.25679)
- **ReCal: "Calibrating Structured Pruning for On-Policy Distillation Recovery"** (Jiang et al.; arXiv 2610.11332; Oct 2026).
  - Pruning damage that persists after offline KD limits OPD recovery. "Recovery-aware calibration" chooses the calibration before pruning, i.e. the pruning criterion is tuned for recoverability rather than zero-shot damage.
  - Models: Qwen3-8B and Qwen3-4B-Instruct; FFN-neuron pruning (~25% of parameters) with Wanda-sp / FLAP.
  - — [arXiv](https://arxiv.org/abs/2610.11332)
- **SPRINT: "Accurate Sublayer Pruning for Large Language Models by Exploiting Latency and Tunability Information"** (Park et al.; IJCAI 2025, pp. 8213–8221, DOI 10.24963/ijcai.2025/913). This is the "IJCAI 2025 sublayer pruning paper."
  - Iteratively removes attention or MLP sublayers, choosing by latency saved *and* "tunability" (how well the rest can be re-tuned), then quickly re-tunes.
  - Up to 23.88 percentage points higher zero-shot commonsense accuracy than prior pruning.
  - — [mlanthology](https://mlanthology.org/ijcai/2025/park2025ijcai-accurate); [IJCAI bibtex](https://www.ijcai.org/proceedings/2025/bibtex/913)

**Verification of the other user-supplied IDs**
- **2310.06694 = Sheared LLaMA** (Xia, Gao, Zeng, Chen; ICLR 2024).
  - Targeted structured pruning to a predefined shape (layers, heads, intermediate and hidden dims), plus dynamic batch loading.
  - LLaMA2-7B → 1.3B / 2.7B using 50B tokens (5% of OpenLLaMA's budget).
  - No snippet found of it comparing zero-shot vs post-training rankings across unit types. The target shape is fixed from existing model configs, not chosen by zero-shot loss.
  - — [arXiv PDF](https://arxiv.org/pdf/2310.06694); [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/160adf2dc118a920e7858484b92a37d8-Paper-Conference.pdf)
- **2510.14444 = "A Free Lunch in LLM Compression: Revisiting Retraining after Pruning"** (Wagner, Roux, Zimmer, Pokutta; Oct 2025, v2 Feb 2026).
  - About *local reconstruction granularity*: once the reconstructed window contains at least one nonlinear submodule, final quality barely depends on window size; per-matrix reconstruction is consistently worse. Tested up to 72B.
  - Not about ranking unit types.
  - — [arXiv](https://arxiv.org/abs/2510.14444); [code](https://github.com/ZIB-IOL/Free-Lunch-in-LLM-Compression)
- **2504.04342 = "Pruning Laws for Large Language Models"** (Sengupta, Chaudhary, Chakraborty; v1 titled "Compression Laws for Large Language Models").
  - Scaling relations from unpruned performance and pruning ratio to post-pruning performance. 10 LLMs (1.3B–30B) plus a 20B MoE; unstructured / width / depth; 8 tasks; under 7% extrapolation error.
  - Identifies "pruning thresholds beyond which recovery is not feasible."
  - Depth gives large speedups but is "less stable"; unstructured and width are steadier.
  - — [arXiv](https://arxiv.org/abs/2504.04342)
- **2607.13124 = "ShortOPD: Recovering Pruned LLMs with Short-to-Long On-Policy Distillation"** (Zhang et al.; Jul 2026; work in progress).
  - Block-Influence depth pruning of Qwen3-4B (~25% of parameters), recovered by on-policy distillation from the frozen parent with a short-to-long rollout schedule.
  - About 9× the unrecovered score, and 1.6–4.4× SFT / KD / SeqKD recovery.
  - Within 2 points of a fixed 8192-token horizon at a quarter of the time (8.5 vs 35.9 h) with 71% fewer rollout tokens.
  - Only one unit type (depth), so no ranking comparison.
  - — [arXiv](https://arxiv.org/abs/2607.13124)
- **2407.13331 = LIAR, "Reconstruct the Pruned Model without Any Retraining"** (Jul 2024).
  - Retraining-free linear-interpolation reconstruction of preserved weights.
  - BERT-base keeps 98% accuracy at 50% pruning; LLaMA-7B gets a "2.56× performance enhancement" at 50% within 1 minute.
  - About retraining-free recovery, not ranking flips.
  - — [arXiv PDF](https://arxiv.org/pdf/2407.13331)
- **2403.17887 = Gromov et al., "The Unreasonable Ineffectiveness of the Deeper Layers"** (ICLR 2025).
  - Removes the block of n consecutive layers with the smallest input/output angular distance, then optionally "heals" with QLoRA.
  - A simpler heuristic (drop from the penultimate layer inward) nearly matches the similarity-based choice *after* healing. This is a small instance of healing erasing differences between selection criteria.
  - Healing gains are described as modest; reasoning tasks are sensitive to any removal.
  - — [arXiv](https://arxiv.org/abs/2403.17887v1); [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/cbabc2f70de2dd09f491a8715ec3e80f-Abstract-Conference.html)
- **2402.02834 = Kim et al., "Shortened LLaMA: Depth Pruning for LLMs with Comparison of Retraining Methods."**
  - Depth pruning plus LoRA is comparable to or better than width pruning (FLAP, LLM-Pruner) at moderate ratios (up to ~35%).
  - Example: 35%-pruned LLaMA-7B averages 55.2 vs FLAP 52.7 and LLM-Pruner 55.5 (secondary summary, **unverified**).
  - At severe ratios (≥~45%, secondary), continued pretraining beats LoRA markedly.
  - Depth gives real speedups at small batch; width often does not.
  - — [arXiv v2 HTML](https://arxiv.org/html/2402.02834v2); [Emergent Mind](https://www.emergentmind.com/papers/2402.02834) (secondary)

### Inferences
- The flip direction differs between settings, so it is not universal:
  - Minitron (LLM, Nemotron-15B, KD with ~1.8B probe tokens then ~94B tokens): depth looks better zero-shot and width wins after retraining.
  - Huang et al. (VLM language backbone, SFT/KD): width looks better zero-shot and depth wins after recovery at ≤30%.
  - The user's result (layers about 10× more recoverable per unit of zero-shot damage than MLP neurons) matches Huang's low-ratio regime and the Shortened-LLaMA moderate-ratio result, and is opposite to Minitron's width preference. Likely factors are the compression ratio, the recovery budget and the distillation objective (Minitron needed intermediate-state distillation for depth). This is an inference across studies, not a measured result.
- Normalizing by zero-shot damage (post-recovery gain per unit of pre-recovery KL) does not appear in any source found. All prior work compares unit types at matched *parameter count* or *compression ratio*. That makes the user's "recoverability per unit of zero-shot damage" framing apparently novel (absence of evidence, not proof).
- Candidate mechanisms in the literature, none formalized as theory for the flip:
  1. Plasticity / architecture-not-weights (Liu 2019; Mittal): with enough retraining the specific inherited weights stop mattering, so zero-shot damage, which measures inherited-weight loss, becomes irrelevant.
  2. Trainability vs damage (TraceNAS; SPRINT "tunability"): what matters is how aligned the pruned model's gradients and loss landscape are with the parent's, not the instantaneous loss.
  3. Residual-stream redundancy: whole-layer removal in residual nets is partially an identity-like perturbation that later layers can absorb (Gromov's angular-distance argument), while removing MLP neurons removes stored features or knowledge across every layer. This is consistent with 2610.15574's "recovery repairs form, not knowledge." This is the report's synthesis, not a claim made by any single paper.
  4. Importance non-stationarity (2610.15574): unit importance rankings change as pruning proceeds and during recovery.

### Gaps
- Exact Minitron figure numbers for the ranking flip (which candidates, loss before and after ~1.8B tokens) could not be read; arxiv.org was blocked.
- The Spearman values in 2610.15574 and the full Laurent et al. 2020 table were not retrieved.
- No source found gives a formal theory or bound explaining *why* a zero-shot ranking flips after distillation across unit types.
- No ranking-flip study was found for hybrid linear-attention decoders (GatedDeltaNet/Mamba-style) or for OCR / document VLMs specifically.
- Puzzle's own evidence on whether replace-1-block scores track post-GKD quality was not found in the snippets.

## Q3: Has anyone measured "recoverability" per unit type (depth vs width vs heads vs vision encoder) as a reusable coefficient, or used it to allocate a pruning budget across unit types?

### Takeaway
No source found defines a reusable per-unit-type recoverability coefficient: post-recovery quality regained per unit of zero-shot damage, for layers vs MLP neurons vs heads vs ViT blocks. The closest work splits into three groups:
- Unit-level "tunability" or recovery signals used to *select* units: SPRINT (IJCAI 2025, per sublayer), LOOP (2026, per layer, via short OPD) and ReCal (2026, calibration for OPD-recoverability).
- Empirical depth-vs-width-with-recovery comparisons across ratios: Huang et al. VLM 2025/2026, Shortened LLaMA 2024, Minitron 2024, Pruning Laws 2025.
- Search methods whose proxy is a short recovery run or a trainability score: Minitron's ~1.8B-token retraining probe, TraceNAS, Puzzle's BLD.

None of these turns recoverability into a constant used to split a budget across unit types. Vision-encoder recoverability relative to the decoder is essentially unmeasured in this literature.

### Cited Findings
- **SPRINT** (IJCAI 2025, DOI 10.24963/ijcai.2025/913) is the closest to "recoverability as a selection score." It ranks each attention or MLP *sublayer* by latency gain and *tunability* (post-removal re-tunability) and removes them iteratively with fast re-tuning; it reports up to +23.88 percentage points zero-shot commonsense accuracy over baselines. It is a per-instance score inside one model, not a per-type constant. — [mlanthology](https://mlanthology.org/ijcai/2025/park2025ijcai-accurate)
- **LOOP** (arXiv 2610.25679) uses recovery signals from short on-policy distillation to pick the next layer to remove. It covers depth only, not cross-type allocation. — [arXiv](https://arxiv.org/abs/2610.25679)
- **ReCal** (arXiv 2610.11332) optimizes the pruning calibration for OPD recoverability, using FFN-neuron pruning on Qwen3-8B / 4B-Instruct. It covers a single unit type. — [arXiv](https://arxiv.org/abs/2610.11332)
- **Minitron** (2407.14679, NeurIPS 2024) allocates across depth, heads, MLP and embedding by enumerating candidate architectures and ranking them after ~1.8B tokens of lightweight retraining. Recoverability is measured empirically per candidate rather than as a per-type coefficient. — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/4822991365c962105b1b95b1107d30e5-Paper-Conference.pdf)
- **Huang et al. VLM studies** (2507.20749 GCPR 2025; 2604.24380 IJCV 2026) give per-type recovery behaviour as a function of ratio:
  - Layerwise recovers better at ≤30%, widthwise at ≥40–45%.
  - Projector-only tuning is enough below 20%.
  - 5% of the data recovers over 95% of performance.
  - Language backbone only; the **vision encoder is not pruned**.
  - — [2507.20749](https://arxiv.org/abs/2507.20749); [2604.24380](https://arxiv.org/html/2604.24380)
- **Pruning Laws** (2504.04342) fits per-strategy (unstructured / width / depth) scaling relations and identifies "pruning thresholds beyond which recovery is not feasible." This is the closest thing to a per-type recoverability *law*, but it predicts from unpruned performance and ratio, not from zero-shot damage, and it is not used to split a mixed budget. — [arXiv](https://arxiv.org/abs/2504.04342)
- **Shortened LLaMA** (2402.02834): depth plus LoRA recovers to parity with width methods up to ~35%; beyond ~45%, LoRA is not enough and continued pretraining is required. This is a per-type recovery-budget threshold, not a coefficient. — [arXiv v2 HTML](https://arxiv.org/html/2402.02834v2) (thresholds from a secondary summary, **unverified**)
- **Attention vs MLP recovery** (2026 search hit, apparently "High-Fidelity Pruning for Large Language Models," arXiv 2603.08083; attribution **unverified**): after fine-tuning, MLP-only pruning reaches 61.9% average at 20% vs 60.3% for attention+MLP pruning. MLP-only shows "larger improvement potential" while attention+MLP shows "more limited recovery." — [arXiv HTML 2603.08083](https://arxiv.org/html/2603.08083) (**unverified** which paper the snippet came from)
- **AMP: Attention Heads and MLP Pruning** (arXiv 2504.21174): concludes that attention and MLP pruning ratios "should be considered independently," since they saturate at different points. This is the per-type allocation idea without a recoverability constant. — [arXiv HTML](https://arxiv.org/html/2504.21174v2) (snippet attribution **unverified**)
- **Vision encoder:** MoPE-CLIP (CVPR 2024) prunes CLIP encoders with module-wise importance and prioritizes width before depth. No per-type recoverability coefficient was found. — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Lin_MoPE-CLIP_Structured_Pruning_for_Efficient_Vision-Language_Models_with_Module-wise_Pruning_CVPR_2024_paper.pdf) (details **unverified**)
- **Puzzle** (2411.19146) allocates heterogeneous per-layer budgets (attention variants, FFN widths, no-op blocks) by MIP over replace-1-block scores computed after blockwise local distillation. This bakes a small amount of local recovery into the score, the nearest NAS analogue to "recovery-aware allocation," but still per-block and pre-GKD. — [arXiv HTML](https://arxiv.org/html/2411.19146)
- **TraceNAS** (2602.02891) jointly allocates depth and width using a gradient-alignment proxy intended to predict post-training quality. No per-type constant is reported in the snippets. — [arXiv](https://arxiv.org/abs/2602.02891)
- **OverRep: "Train Overcomplete, Deploy Compact: Scaling Recovery Capacity for Structured LLM Pruning"** (arXiv 2609.06974; HF repo labels it EMNLP 2026, **unverified**). Treats recovery as capacity-limited reconstruction; retained performance at 25% pruning is about 90–95% on reasoning and 47–67% on generation (HF repo numbers). This concerns recovery capacity, not per-type allocation. — [arXiv PDF](https://arxiv.org/pdf/2609.06974); [HF](https://huggingface.co/SoongE/OverRep)

### Inferences
- Allocating a pruning budget across unit types by a measured recoverability coefficient appears to be an open gap:
  1. Measure the zero-shot KL per unit type.
  2. Measure the post-distillation gain per unit of that KL.
  3. Allocate by the marginal recoverable cost.
- The literature supports the ingredients:
  - type-dependent recovery (Huang; Shortened LLaMA; Minitron);
  - recovery-aware unit scores (SPRINT, LOOP, ReCal);
  - short-recovery probes (Minitron ~1.8B tokens; Puzzle BLD).
- The user's ~10× layer-vs-MLP-neuron recoverability ratio would be a new quantitative datum, especially for a hybrid GatedDeltaNet decoder plus ViT in OCR.
- The evidence suggests any such coefficient is ratio- and budget-dependent, not constant:
  - Huang's crossover at ~30–45%.
  - Shortened LLaMA needs CPT beyond ~45%.
  - 2606.14150: pruned-init advantage shrinks with budget.
  - The paper should present recoverability as a function of ratio and recovery tokens, or at least report the regime where it was measured.

### Gaps
- No source found measures vision-encoder (ViT) prunability or recoverability *relative to* the language decoder inside a VLM with recovery. Huang et al. prune only the LLM backbone; MoPE-CLIP covers CLIP alone.
- No source found studies recoverability in hybrid linear-attention (GatedDeltaNet / Mamba) decoders, i.e. linear-attention layers vs full-attention layers vs MLP.
- The exact numbers in SPRINT's tunability metric definition, and whether it transfers across models, were not retrievable (full text not accessed).
- The full texts of all 2026 preprints (LOOP, ReCal, 2610.15574, TraceNAS, 2606.14150) were not read because arxiv.org was blocked. Their quantitative claims above come from abstracts or snippets.
