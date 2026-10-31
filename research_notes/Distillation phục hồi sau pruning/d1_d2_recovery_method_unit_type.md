# D1/D2: Does the recovery method change which pruned candidate wins, and which distillation signal recovers which unit type?

Context for the report writer: the user prunes LightOnOCR-3-0.8B (hybrid GatedDeltaNet + full-attention decoder, 12-block ViT) to ~0.3B / ~0.1B. They recover with CE + forward-KL top-20 logit KD (0.5/0.5), with no hidden-state KD and no on-policy training. They see depth cuts recover about 10x better than MLP-width cuts per unit of zero-shot damage, which is the opposite of Minitron.

**How these notes were gathered:** arxiv.org, proceedings.neurips.cc, alphaxiv, bytez.com and the Helmholtz repository all returned ENOTFOUND from this session. Only github.com could be fetched. So almost every paper below is marked **NOT VERIFIED (snippet only)**: the finding comes from web-search snippets of the abstract, HTML or PDF, or from secondary summaries, not from reading the full text. Exact table numbers that I could not see are listed under Gaps rather than guessed. Some items are marked "prior knowledge, not re-checked this session"; treat them as leads only.

---

## D1: Does the recovery method change which pruned candidate wins?

### Takeaway
Yes, there is direct evidence that the recovery method changes which candidate wins.
- **Repair or recovery changes which layers should be removed:** 2610.30552 (exact search), 2610.11332 / 2610.25679 (on-policy recovery), and Puzzle's BLD→GKD.
- **No recovery vs. recovery flips depth vs. width,** in opposite directions in two studies:
  - Minitron: depth looks better zero-shot, width wins after KD.
  - Huang et al. on VLMs: width wins without recovery; with recovery, layerwise wins at small/moderate ratios and width wins at high ratios.

I found no paper that ranks depth vs. width under logit-only KD versus hidden-state KD and reports the rank change in a single table. That exact controlled comparison is a gap. The closest is Huang et al. (VLM, SFT vs. logit KD vs. hidden-state KD × layerwise vs. widthwise), but I could not read their tables.

### Cited Findings

**Minitron, "Compact Language Models via Pruning and Knowledge Distillation" (arXiv 2407.14679, NeurIPS 2024). NOT VERIFIED (snippet only)**
- Retraining objective: L_total = L_CLM + α·L_IS + L_logits. L_IS is an intermediate-state similarity term and L_logits is a KLD between teacher and student — [alphaXiv summary](https://www.alphaxiv.org/abs/2407.14679); [NeurIPS PDF (search snippet)](https://proceedings.neurips.cc/paper_files/paper/2024/file/4822991365c962105b1b95b1107d30e5-Paper-Conference.pdf)
- Best-practice list (as quoted by summaries):
  - "Prefer width pruning over depth for the model scales we consider (≤15B)."
  - "Retrain exclusively with distillation loss using KLD instead of conventional training."
  - "Use (logit+intermediate state+embedding) distillation when depth is reduced significantly"; logit-only distillation when depth is not reduced significantly.
  - Sources: [NeurIPS PDF snippet](https://proceedings.neurips.cc/paper_files/paper/2024/file/4822991365c962105b1b95b1107d30e5-Paper-Conference.pdf); [Medium summary](https://ritvik19.medium.com/papers-explained-208-minitron-e55ea374d9dd); [NVIDIA dev blog](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model)
- **Zero-shot vs. retrained rank flip.** A commentary on the paper says "right after pruning, depth-pruned models look better. The superiority of width pruning only reveals itself after a short period of retraining." — [secondary summary via search](https://www.emergentmind.com/papers/2407.14679); [NeurIPS PDF snippet](https://papers.nips.cc/paper_files/paper/2024/file/4822991365c962105b1b95b1107d30e5-Paper-Conference.pdf)
  - Note for the writer: this is the same kind of flip the user sees, but in the *opposite* direction to the user's result.
- Appendix comparison: MINITRON 8B Width reaches LM loss 2.049, versus ~2.155 and ~2.177 for depth-only variants at similar size — [search snippet of paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/4822991365c962105b1b95b1107d30e5-Paper-Conference.pdf)
- For some models (MINITRON 8B and 4B) the authors used "L_logits alone in such cases (see Table 18)". Table 18 is captioned "Ablation study for MINITRON 8B with and without the loss component L_is…". I could not retrieve its numbers — [NeurIPS PDF snippet](https://proceedings.neurips.cc/paper_files/paper/2024/file/4822991365c962105b1b95b1107d30e5-Paper-Conference.pdf)
- **What this means for the user's question:** Minitron's depth-vs-width comparison was not run under a matched loss.
  - Intermediate-state loss was used and recommended specifically when depth is cut heavily.
  - Logit-only was used when depth is not cut much (for example the width-pruned 8B).
  - I found no snippet showing that they ranked depth vs. width under both loss configurations.

**Llama-3.1-Minitron, "LLM Pruning and Distillation in Practice: The Minitron Approach" (arXiv 2408.11796, 2024). NOT VERIFIED (snippet only)**
- Setup: Llama-3.1-8B pruned to 4B in two ways.
  - Depth: 32→16 layers, removing one contiguous block chosen by the smallest Winogrande drop.
  - Width: hidden, MLP and heads.
- Results after distillation:
  - MMLU: width ≈60.5% vs. depth ≈58.7%.
  - GSM8K: width ≈41.2% vs. depth ≈16.8%.
  - Throughput: depth ≈2.7× the 8B teacher, width ≈1.8×.
  - Sources: [Liner review](https://liner.com/review/llm-pruning-and-distillation-in-practice-minitron-approach); [arXiv 2408.11796v4 snippet](https://arxiv.org/pdf/2408.11796v4); [alphaXiv](https://www.alphaxiv.org/abs/2408.11796)
- Width-pruned models start at lower loss and converge faster during distillation — [arXiv snippet](https://arxiv.org/pdf/2408.11796v4)
- Layer selection: removing a contiguous block beat removing individually "important" non-adjacent layers on downstream accuracy. LM loss / PPL suggested the opposite (non-contiguous looked best), so the proxy metric and downstream accuracy disagreed — [Lacuna summary](https://lacuna.tiptreesystems.com/work/llm-pruning-and-distillation-in-practice/wrk_bace53a2b062b2cf8ad9293c6117db2a); [Medium](https://ritvik19.medium.com/papers-explained-209-minitron-approach-in-practice-6b473f67328d)
- Teacher correction (fine-tuning the teacher on the distillation data) is part of the recipe — prior knowledge, not re-checked this session.

**Huang, Thede, Mancini, Xu, Akata: GCPR 2025 (arXiv 2507.20749) and IJCV 2026 extended version (arXiv 2604.24380). NOT VERIFIED (snippet only)**
- Setup: prunes the LLM backbone of LLaVA-v1.5-7B, InternVL-Chat-4B and Bunny-v1-3B.
  - Layerwise pruning is ShortGPT-style.
  - Widthwise pruning is LLM-Pruner-style (heads and MLP channels).
  - Recovery options compared: SFT, logit KD and hidden-state KD — [arXiv 2604.24380](https://arxiv.org/abs/2604.24380); [GitHub README (fetched)](https://github.com/YiranHuangIrene/VLMCompression)
- **Rank depends on recovery:**
  - Without recovery training, widthwise is better.
  - With recovery, layerwise is better at small ratios and widthwise at higher ratios.
  - Sources: [search summary of 2604.24380](https://arxiv.org/html/2604.24380); [author page](https://yiranhuangirene.github.io/)
- The crossover point differs between versions: about ≤30% in one, <40% in the GCPR summary, and 45% in another. Recommended regimes (IJCV):
  - (i) widthwise only, with no recovery, in extremely low-resource settings;
  - (ii) layerwise + MM-projector fine-tuning for moderate compression (≤30%);
  - (iii) widthwise + KD for high compression (≤45%).
  - Sources: [arXiv 2604.24380](https://arxiv.org/html/2604.24380); [Pith summary of 2507.20749](https://pith.science/paper/2507.20749)
- **Best recovery method:** SFT + hidden-state distillation (L2 on intermediate states) is "optimal across various pruning levels". Logit-only KD is reported to be "significantly worse at higher ratios than SFT alone when not combined with feature alignment". At 15% pruning, all methods are about equal to SFT — [search summary of 2604.24380](https://arxiv.org/html/2604.24380)
- Layerwise pruning "disproportionately degrades performance on complex reasoning tasks" relative to widthwise — [search summary](https://arxiv.org/html/2604.24380)
- Data efficiency: SFT + hidden-state KD keeps >95% of original performance using 5% of the recovery data — [author page](https://yiranhuangirene.github.io/); [arXiv 2604.24380](https://arxiv.org/abs/2604.24380)
- The GCPR version has a figure comparing KD loss variants (KL, RKL, L2 and combinations) on widthwise-pruned Bunny and LLaVA (Figure 3). I did not obtain the numbers — [Pith 2507.20749](https://pith.science/paper/2507.20749)
- The released code has a `-dist` (hidden-state distillation) template for Mini-InternVL. The README does not document which layers are matched or the loss weights. Logit KD is not mentioned in the README — [GitHub README (fetched)](https://github.com/YiranHuangIrene/VLMCompression)

**Shortened LLaMA (arXiv 2402.02834; Kim et al., Nota AI, 2024). NOT VERIFIED (snippet only)**
- Depth pruning plus LoRA is enough at moderate ratios. A 35% depth-pruned LLaMA-7B with LoRA averages 55.2%, vs. FLAP 52.7% and LLM-Pruner 55.5%. All width baselines were "identically retrained with LoRA" — [arXiv v1 HTML snippet](https://arxiv.org/html/2402.02834v1); [Nota AI](https://www.nota.ai/community/shortened-llm-a-simple-depth-pruning-for-large-language-models)
- At severe ratios, continued pretraining (CPT) "markedly outperforms LoRA". A secondary summary puts the threshold at roughly ≥45% — [Moonlight review](https://www.themoonlight.io/en/review/shortened-llama-depth-pruning-for-large-language-models-with-comparison-of-retraining-methods); [alphaXiv](https://www.alphaxiv.org/abs/2402.02834)
- Gap: width pruning was **not** also retrained with CPT, so this paper cannot say whether CPT changes the depth-vs-width ranking — [search summary](https://arxiv.org/html/2402.02834v1)

**"Repair Changes Which Layers to Remove: Exact Optima for Depth Pruning of Language Models" (arXiv 2610.30552, Su et al., Oct 2026). NOT VERIFIED (snippet only)**
- The authors exhaustively evaluated about 53,000 pruned models across four families, with and without a training-free linear repair.
- **The optimal layer set depends on whether repair is applied.**
  - Linear repair turns the optimum from a split layout into a contiguous block.
  - Without repair, removing two nearby layers costs more than the sum of their individual costs, so a contiguous block is near-worst without repair and near-best with it.
  - Choosing layers while ignoring repair can miss the repaired optimum by up to 189%.
  - A repair-aware greedy search, at SLEB cost, lowers perplexity by 20–66% vs. ReplaceMe on five models up to 27B.
  - Limit: exact optima were computed only up to 6 two-layer groups or 3 single layers.
  - Sources: [arXiv abs](https://arxiv.org/abs/2610.30552); [HTML](https://arxiv.org/html/2610.30552)

**ReCal: "Calibrating Structured Pruning for On-Policy Distillation Recovery" (arXiv 2610.11332, Jiang et al., Oct 2026). NOT VERIFIED (snippet only)**
- Method: uses forward KL between the unpruned teacher and a pruned probe to find teacher-supported predictions that pruning disrupted. It then reweights calibration statistics so that existing pruning criteria keep those predictions.
- Results: after on-policy distillation (OPD), gains of up to +16.7 pp on AIME and improvements in most code comparisons. Results are mixed on instruction following and on the 8B LLM-Pruner case.
- I.e., making the pruning choice *aware of the recovery method* changes and improves which units are kept.
- Sources: [arXiv abs](https://arxiv.org/abs/2610.11332); [HTML](https://arxiv.org/html/2610.11332)
- A later paper (2610.34336) says ReCal prunes Qwen3-8B FFN neurons. It reports that recalibrating on the dense model's own responses brings the pruned model to 25.7% and the recovered model to 48.5% — [arXiv 2610.34336](https://arxiv.org/html/2610.34336)

**LOOP: "Recovery-Aligned Iterative Layer Pruning via On-Policy Distillation" (arXiv 2610.25679, Xu et al., Oct 2026). NOT VERIFIED (snippet only)**
- Argues that separating layer selection from recovery "limits how recovery information can guide selection".
- Method: interleaves single-layer pruning with short OPD, and scores each layer by its reasoning contribution and its sensitivity to recovery, using on-policy trajectories and gradients. Includes a warm-up of recovery updates before the first prune.
- Ablation (Qwen3-4B-Instruct, 20% layers): iterative pruning and recovery-oriented selection each contribute, vs. one-shot pruning at a matched recovery budget. I did not get the headline numbers.
- Sources: [arXiv abs](https://arxiv.org/abs/2610.25679); [HTML](https://arxiv.org/html/2610.25679)

**ShortOPD (arXiv 2607.13124, 2026, "work in progress"). NOT VERIFIED (snippet only)**
- Setup: depth-pruned Qwen3-4B-Instruct (e.g., 4 of 36 layers by Block Influence; main experiments ~25% of parameters), with the frozen unpruned model as teacher.
- Short-to-long OPD:
  - scores about 9× the unrecovered model;
  - scores 1.6–4.4× standard recipes (SFT without KD, KD, SeqKD);
  - matches a fixed 8192-token horizon within 2 points with a quarter of the training time (8.5 vs. 35.9 h) and 71% fewer rollout tokens.
- Mechanism: pruned models fall into repetitive suffixes. ShortOPD detects teacher-confirmed repetition and treats the prefix before it as the rollout's effective length.
- Sources: [arXiv HTML v1](https://arxiv.org/html/2607.13124v1); [HF page](https://huggingface.co/papers/2607.13124)
- Only one pruning type (depth) was used, so this paper does not rank candidates under different recovery methods.

**"The Fragility and Dynamics of LLM Pruning-Recovery Steps" (arXiv 2610.15574, Salim et al., Oct 2026). NOT VERIFIED (snippet only)**
- Under structural pruning, instruction-following degrades first, before knowledge, reasoning or core LM.
- In-domain data gives the strongest pruning signal; instruction-style data gives the strongest recovery signal.
- Aligning the recovery loss with the pruning loss gave no measurable benefit. Pruning-first ordering helps only modestly.
- The paper also analyses how the importance ranking drifts before vs. after pruning.
- Sources: [arXiv abs](https://arxiv.org/abs/2610.15574); [HTML](https://arxiv.org/html/2610.15574)

**Gromov et al., "The Unreasonable Ineffectiveness of the Deeper Layers" (arXiv 2403.17887, ICLR 2025). NOT VERIFIED (snippet only)**
- Without healing, the simple "drop deepest layers" heuristic deteriorates quickly on Llama-2-70B. After QLoRA healing (4-bit), it is "quite comparable" to the similarity-based layer choice on MMLU, BoolQ and C4 loss.
  - This is another case of recovery erasing the zero-shot ranking difference between layer-selection methods.
- QA accuracy stays flat until about 40–50% of layers are removed and then collapses. Healing extends the pruning fraction at which the model falls to random guessing.
- Sources: [arXiv PDF snippet](https://arxiv.org/pdf/2403.17887); [IAIFI post](https://research.iaifi.org/posts/the-unreasonable-ineffectiveness-of-the-deeper-layers/); [Liner](https://liner.com/review/the-unreasonable-ineffectiveness-of-the-deeper-layers)

**Reassessing Layer Pruning in LLMs (arXiv 2411.15558, ICLR 2026). NOT VERIFIED (snippet only)**
- Setup: prunes the final 25% of layers of Llama-3.1-8B-Instruct.
- Recovery methods compared: LoRA, QLoRA, lm_head-only, and lm_head + last 1/2/3 layers.
- Fine-tuning lm_head + the last three remaining layers recovers better than LoRA, and also trains faster. On *unpruned* models, LoRA and partial-layer tuning perform about the same.
- The authors' explanation: after pruning, the distribution shift sits in the deepest remaining layers, so LoRA updates spread over all layers miss it.
- Sources: [arXiv HTML](https://arxiv.org/html/2411.15558); [ICLR 2026 PDF snippet](https://proceedings.iclr.cc/paper_files/paper/2026/file/40624bef9a7d22c0c780dfe9291f5843-Paper-Conference.pdf); [Liner](https://liner.com/review/reassessing-layer-pruning-in-llms-new-insights-and-methods)

**Self-Data Distillation (arXiv 2410.09982; NeurIPS 2024 Compression Workshop / MLSys 2025). NOT VERIFIED (snippet only)**
- Setup: Llama-3.1-8B-Instruct with 6 decoder blocks removed (32→26 layers, 8.03B→6.72B).
- Fine-tuning on teacher-generated (self-distilled) data keeps 91.2% of the original accuracy, vs. 81.7% with plain SFT on the original data. The advantage holds across all pruned models of Llama-3.1-8B and Mistral-7B.
- Version discrepancy: the workshop abstract says 32→24 layers and an 18.75% FLOPs reduction; arXiv and MLSys say 26 layers and 16.3%.
- Sources: [arXiv abs](https://arxiv.org/abs/2410.09982); [OpenReview PDF snippet](https://openreview.net/pdf?id=aUaltIxtol); [MLSys](https://proceedings.mlsys.org/paper_files/paper/2025/hash/af2d9fb5bcee19ef2dfa70d843520c97-Abstract-Conference.html)
- Only depth was tested, so this paper does not rank depth vs. width.

**Puzzle: "Distillation-Based NAS for Inference-Optimized LLMs" (arXiv 2411.19146, NVIDIA, 2024/25). NOT VERIFIED (snippet only)**
- Blockwise local distillation (BLD) trains replacement blocks; global KD (GKD) is then used "precisely to repair the mismatch from blockwise training, which doesn't account for cross-block compatibility" — [Pith review](https://pith.science/paper/2411.19146)
- Coupled BLD (trained on a subspace found by decoupled BLD) scored 73.98 vs. 73.10 for decoupled BLD. So *how* the candidate blocks are distilled changes which architecture the search picks and how well it does — [search snippets of arXiv HTML](https://arxiv.org/html/2411.19146)
- Table 1 ablates combinations of LM loss, block (hidden-activation) loss and logit KLD for GKD on Nemotron-51B for ~5B tokens; the final model used the best combination for 45B tokens. I could not retrieve the per-combination numbers — [arXiv HTML v3](https://arxiv.org/html/2411.19146v3)

**Small LLMs: Pruning vs. Training from Scratch (arXiv 2606.14150, Xu, Lu, …, Mingjie Sun, Zhuang Liu, 2026). NOT VERIFIED (snippet only)**
- Setup: Llama-3.1-8B pruned at ratios 0.5–0.8 with six methods.
- Token-matched comparison: pruned initialization beats random initialization, but the advantage shrinks as the training budget and the pruning ratio grow.
- If scratch training gets the parent's whole token budget, sparse pruning still wins, but scratch training can match structured pruning. It catches up at coarser granularities, not finer ones.
- Sources: [alphaXiv](https://www.alphaxiv.org/abs/2606.14150); [arXiv HTML](https://arxiv.org/html/2606.14150v1)

**Not covered in this session (no snippets retrieved): Sheared-LLaMA (2310.06694), LLM-Pruner (2305.11627), SlimGPT.**
- From prior knowledge, not re-checked: LLM-Pruner recovers with LoRA only, and Sheared-LLaMA uses continued pretraining with dynamic batch loading.
- To my knowledge, neither ranks pruning candidates under two recovery methods.
- Sheared-LLaMA PDF link from a search result: [arXiv 2310.06694](https://arxiv.org/pdf/2310.06694)

### Inferences
- **Direction of the flip.** The literature shows ranking flips from *no recovery → recovery* in both directions.
  - Minitron (LLM, KD, heavy retraining): depth looks better zero-shot, width wins after KD.
  - Huang (VLM backbone, light SFT/KD): width wins zero-shot; layerwise wins after recovery at small/moderate ratios.
  - The user's result (depth recovers ~10× better per unit of zero-shot damage) matches Huang's direction. Huang's setting is closer to the user's: a VLM, a light recovery budget, and a modest data/step count (3k–12.5k steps × 16 pages is small next to Minitron's ~94B tokens).
  - Minitron's width win appeared at large token budgets and is stated only for ≤15B models with ~50% cuts. Light-budget recovery seems to favour depth at moderate ratios.
- **The user's recovery method probably does not artificially favour depth.** The papers that study this suggest logit-only KD *under-serves* depth cuts relative to hidden-state KD:
  - Minitron recommends intermediate-state loss specifically when depth is cut heavily.
  - Huang found hidden-state L2 KD was best overall and logit-only KD was weak at high ratios.
  - So adding hidden-state KD would likely improve depth cuts at least as much as width cuts, which strengthens rather than reverses the user's depth preference. This is an inference; no paper ran the matched 2×2 (depth/width × logit/hidden) that would confirm it.
  - The bigger confounder is probably the *ratio regime*: Huang's crossover to width is at ~30–45% of LLM-backbone parameters, and both Minitron and Huang show width wins at aggressive ratios. At ~0.3B and ~0.1B from 0.8B (62–88% total reduction), the user is in the regime where both papers favour width after recovery.
- **The recovery type also changes which *layers* win, not only depth vs. width.**
  - Linear repair makes contiguous blocks optimal (2610.30552).
  - Healing makes "drop deepest layers" as good as similarity-based selection (Gromov).
  - Recovery-aware selection (LOOP, ReCal) beats recovery-agnostic selection.
  - Practical consequence: the user's zero-shot layer importance may not be the right ranking for KD-recovered candidates. A short-KD proxy ranking, as Minitron does with lightweight retraining for candidate selection, or a recovery-aware search is safer.
- **Fit of on-policy recovery to the user's setting.** ShortOPD reports that repetitive suffixes are the failure mode of depth-pruned models, and OPD fixes them far better than SFT/KD/SeqKD. With near-deterministic outputs of ~1,500 tokens per page, the user's regime is long-sequence generation where exposure bias and repetition loops could appear. On-policy checks are therefore relevant, but no OCR-specific evidence was found.

### Gaps
- No paper found that ranks depth vs. width (or different layer sets) under **logit-only KD vs. hidden-state KD** with a matched budget and reports the rank change. Huang et al. (2604.24380) is the closest. Their tables (SFT / logit KD / hidden KD × layerwise / widthwise × ratio) could not be read because arxiv.org, the Springer PDF and the Helmholtz mirror were unreachable.
- Minitron Table 18 (8B with and without L_is) and the depth-pruned intermediate-loss ablation numbers were not retrievable.
- Shortened LLaMA did not retrain width baselines with CPT, so it cannot tell whether CPT vs. LoRA changes the depth-vs-width ranking.
- LOOP's headline numbers, ReCal's per-setting numbers and 2610.15574's quantitative results were not retrieved.
- Sheared-LLaMA, LLM-Pruner, SlimGPT and "Distill-then-prune" were not checked this session.

---

## D2: Which distillation signal recovers which unit type (depth vs. width vs. heads)?

### Takeaway
Across sources, the consistent pattern is:
- **Depth cuts** benefit from feature-level signals, which work on the "boundary activation mismatch" left at the cut:
  - hidden-state / intermediate-state KD (Minitron; Huang L2);
  - linear repair at the boundary (ReplaceMe, Ghosted Layers, LinearPatch) followed by healing;
  - tuning the layers right after the cut (Reassessing Layer Pruning).
- **Width cuts** are recovered adequately by logit KD alone (Minitron, Nemotron-H / Nano 2 / Minitron-SSM).
- For hybrid SSM models, NVIDIA used logit-only forward-KL KD after pruning Mamba heads, FFN, embedding dimension and layers.
- Attention vs. MLP: MLP-only width pruning generally recovers better after fine-tuning than pruning attention. Whole attention *sublayers* can be dropped with little loss on some metrics, but the evidence conflicts.

Numeric evidence on attention transfer (MiniLM-style) for pruned decoder LLMs is scarce. The only recent example found is Eddy-VL (CKA on attention maps).

### Cited Findings

**Hidden-state / intermediate KD for depth**
- Minitron (2407.14679): "Use (logit+intermediate state+embedding) distillation when depth is reduced significantly"; logit-only otherwise. L_IS is added with weight α to CLM + logit KLD — [NVIDIA dev blog](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model); [alphaXiv](https://www.alphaxiv.org/abs/2407.14679). NOT VERIFIED (snippet only)
  - Which layers are matched and the exact loss form (MSE / normalized / cosine) were not visible in snippets; see Gaps.
- Huang et al. (2604.24380): "distillation with L2 on intermediate states delivers the most effective performance recovery after pruning across all compression ratios", best when combined with SFT. Logit-only KD is weaker at higher ratios — [search summary](https://arxiv.org/html/2604.24380). NOT VERIFIED (snippet only)
  - The summary does not say whether the benefit differs between layerwise and widthwise.
- Offline hidden-state distillation for pruned VLAs (arXiv 2609.19579, 2026). NOT VERIFIED (snippet only)
  - Removing 63% of the OpenVLA-OFT backbone drops LIBERO-Long success from 93.2% to 0.8%.
  - Training on a cache built from one teacher pass brings the 63%-reduced student to within 3.5 points of the teacher in about 8 GPU-hours.
  - Gain from hidden-state KD over the baseline recovery:
    - OpenVLA-OFT: +2.1 to +4.5 points at 63–87% reduction; no significant difference up to 45%.
    - CogACT: +9.4 to +22.1 points from 63% onward.
  - Width beats depth at matched compression on task success; depth gives lower latency.
  - A reviewer notes that the depth-pruning distillation layer mapping needs clarification.
  - Sources: [arXiv abs](https://arxiv.org/abs/2609.19579); [Pith](https://pith.science/paper/2609.19579)
  - **Pattern: hidden-state KD matters mainly at aggressive ratios.**
- Eddy-VL 1.9B (arXiv 2607.16316, 2026): concrete layer mapping after depth pruning. NOT VERIFIED (snippet only)
  - Setup: Qwen3-VL-Embedding-2B text decoder pruned 28→24 layers, chosen by adjacent-layer linear CKA.
  - Student:teacher layer pairs: 3:4, 21:23, 22:26, 23:27, 24:28.
  - Mid-layer loss: 1 − CKA on flattened attention maps (attention-transfer style). Final-layer loss: MSE + cosine with Matryoshka.
  - MMEB-V2: teacher 68.9, pruned 56.8, distilled 63.2 (~91.7% retained).
  - Sources: [arXiv HTML v1](https://arxiv.org/html/2607.16316v1); [HF page](https://huggingface.co/papers/2607.16316)

**Logit-only KD for width / hybrid (NVIDIA line)**
- Nemotron Nano 2 (arXiv 2508.14444, 2025): "we opt for logit-only distillation, minimizing the forward KL Divergence loss across the teacher and student probabilities", citing Minitron — [NVIDIA tech report PDF](https://research.nvidia.com/labs/adlr/files/NVIDIA-Nemotron-Nano-2-Technical-Report.pdf); [arXiv 2508.14444](https://arxiv.org/pdf/2508.14444). NOT VERIFIED (snippet only)
- Minitron-SSM (arXiv 2504.11409, NeurIPS 2025). NOT VERIFIED (snippet only)
  - Group-aware Mamba pruning (keeps SSM groups intact) plus FFN, embedding-dimension and layer pruning, then KD retraining; produced Nemotron-H 4B from 8B.
  - Pruning Mamba *heads* beat pruning Mamba head *channels*. The authors argue hybrids tolerate head pruning better than attention-only models.
  - Recovery is logit-based forward-KL KD, reported in the Nemotron-H / Nano 2 reports to beat ordinary fine-tuning.
  - Sources: [NeurIPS 2025 listing](https://mlanthology.org/neurips/2025/taghibakhshi2025neurips-efficient/); [arXiv abs](https://arxiv.org/abs/2504.11409); [Nemotron-H](https://arxiv.org/html/2504.03624v3)
  - Relevance: the closest analogue to the user's GatedDeltaNet hybrid. These snippets give no evidence of hidden-state KD being needed for linear-attention/SSM width cuts.
- Minitron: "Retrain exclusively with distillation loss using KLD instead of conventional training" (KD beats LM-loss retraining) — [Medium summary](https://ritvik19.medium.com/papers-explained-208-minitron-e55ea374d9dd). NOT VERIFIED (snippet only)

**Linear-repair initialization for depth (then optional healing)**
- ReplaceMe (arXiv 2505.02819, NeurIPS 2025). NOT VERIFIED (snippet only)
  - Replaces a contiguous block with one linear transform estimated by least squares on calibration activations, merged into an existing weight. Layers are chosen by the cosine distance of hidden states across the cut.
  - About 90% of performance retained at 25% pruning of LLaMA-2-7B with no healing; beats UIDL (Gromov-style) across models and ratios.
  - At extreme compression a healing phase is still required. "Partial healing" (training only the linear map) is much cheaper than full fine-tuning.
  - Sources: [arXiv HTML](https://arxiv.org/html/2505.02819); [Moonlight](https://www.themoonlight.io/en/review/replaceme-network-simplification-via-layer-pruning-and-linear-transformations)
- Ghosted Layers (arXiv 2605.15491, 2026; repo says NeurIPS 2026). NOT VERIFIED (snippet only)
  - Fits a closed-form linear operator at each pruning boundary (forward hook) to fix the "boundary activation mismatch"; works for contiguous and non-contiguous cuts.
  - Claims gains over Prune&Comp, ReplaceMe and LinearPatch; supports LLM-Streamline and ShortGPT selection.
  - Sources: [arXiv abs](https://arxiv.org/abs/2605.15491); [GitHub](https://github.com/daniel-eai/ghosted_layers_official_repository)
- Repair-aware selection (2610.30552): with linear repair, contiguous blocks become optimal; 20–66% lower PPL than ReplaceMe — [arXiv abs](https://arxiv.org/abs/2610.30552). NOT VERIFIED (snippet only)
- Reassessing Layer Pruning (2411.15558): fine-tuning the lm_head + last 3 remaining layers beats LoRA for depth-pruned models but not for unpruned ones — [arXiv HTML](https://arxiv.org/html/2411.15558). NOT VERIFIED (snippet only)
  - Implies that depth cuts create a localized mismatch best repaired by capacity placed near or after the cut.

**Blockwise local distillation (Puzzle)**
- BLD trains each replacement block (FFN width reduction, attention removal or GQA change, no-op) to mimic its parent block's output, then GKD over the whole model fixes cross-block incompatibility.
- Ablations: coupled vs. decoupled BLD 73.98 vs. 73.10; a no-op-only search space (depth-like) 75.40 vs. a full search space (width + depth) 78.39.
- GKD loss combinations (LM / block hidden / logit KLD) were ablated in Table 1; numbers not retrieved.
- Sources: [arXiv HTML](https://arxiv.org/html/2411.19146); [Pith](https://pith.science/paper/2411.19146). NOT VERIFIED (snippet only)
- From prior knowledge, not re-checked: Puzzle reports that adding LM loss to GKD hurt, and that cosine-similarity block loss + KLD was best. Mark as NOT VERIFIED; it could not be confirmed this session.

**Attention vs. MLP unit types**
- After fine-tuning, MLP-only pruning reached ~61.9% average at 20% pruning vs. 60.3% for joint attention + MLP pruning, with a larger gap at 30% — [arXiv 2603.08083](https://arxiv.org/html/2603.08083v1). NOT VERIFIED (snippet only)
- SDMPrune (arXiv 2506.11120): keeping attention intact and pruning only MLPs with self-distillation recovers better with minimal fine-tuning — [arXiv PDF](https://arxiv.org/pdf/2506.11120). NOT VERIFIED (snippet only)
- Data-free attention-sublayer pruning (arXiv 2512.20636): mid/late attention sublayers are far more redundant than MLPs by cosine similarity — [arXiv PDF](https://arxiv.org/pdf/2512.20636). This conflicts with the two items above; the difference is probably the metric and whether fine-tuning was applied. NOT VERIFIED (snippet only)
- Mamba heads vs. channels: heads are better to prune (Minitron-SSM, above).

**Classic layer-mapping KD (DistilBERT / TinyBERT / MiniLM), LaCo, BlockPruner, SLEB, FinerCut, LLM-Streamline, LinearPatch**
- No snippets retrieved this session.
- From prior knowledge, not re-checked:
  - TinyBERT uses uniform layer mapping with MSE on hidden states and attention matrices.
  - MiniLM transfers last-layer self-attention distributions and value relations.
  - SLEB, LaCo and BlockPruner are mainly evaluated training-free.
  - LLM-Streamline trains a lightweight replacement network for pruned layers.
- Mark all of these as NOT VERIFIED.

### Inferences
- **Mechanism.** Removing a block leaves the next surviving layer receiving out-of-distribution inputs; this is the "boundary activation mismatch" named in Ghosted Layers and implied by Reassessing Layer Pruning. Signals that target this directly work well for depth cuts:
  - hidden-state matching at or after the cut;
  - linear repair at the boundary;
  - tuning the layers next to the cut.
- Width cuts spread damage across all layers. They are not localized, so a global output-level signal (logit KD) is the natural fit, which is consistent with NVIDIA using logit-only KD for width and hybrid pruning.
- **For the user (inference):**
  1. The user's depth cuts already recover well with logit-only KD. Adding a cheap boundary linear repair (ReplaceMe or Ghosted-Layers style) as initialization, plus hidden-state KD at the layers just after each cut (e.g., cosine or L2, TinyBERT-style mapping), is the most evidence-backed upgrade for depth.
  2. For MLP-neuron cuts in the GatedDeltaNet/attention hybrid, the literature does not suggest hidden-state KD is necessary. Huang's VLM result (hidden-state L2 helped overall) is the only evidence that it might help width too.
  3. Because the user's KD signal is top-20 renormalized logits on only ~50% of pages, the logit signal is weaker than full-vocab KD on all data. A hidden-state signal from a same-family teacher would need a projection if the teacher is the 4B model rather than the unpruned 0.8B. Using the unpruned 0.8B as a hidden-state teacher, as in Minitron, Huang and Eddy-VL, avoids the dimension mismatch.
  4. Since several papers show feature-level KD matters most at aggressive ratios, the effect should be largest for the ~0.1B target.

### Gaps
- No paper reports a matched comparison of {logit KD, hidden-state KD, attention transfer} × {depth, MLP-width, head} with numbers. Evidence is pieced together across papers with different models, ratios and budgets.
- Minitron's exact L_IS formulation (which layers, normalized vs. raw MSE, the α value) and the numbers in its depth-vs-loss ablation were not retrievable (arXiv and NeurIPS proceedings blocked).
- Puzzle Table 1 (GKD loss combinations) numbers were not retrieved.
- LinearPatch, LLM-Streamline, LaCo, BlockPruner, SLEB, FinerCut, TinyBERT/MiniLM-style KD applied to pruned decoder LLMs, and "Distill-then-prune": no session evidence.
- No evidence was found on KD for pruned **GatedDeltaNet / linear-attention** layers specifically (only Mamba via Minitron-SSM), and none on ViT-block pruning recovery inside VLMs.
- No OCR-specific pruning-recovery study was found.
