# Q7–Q8: Transfer of pruning masks/importance across fine-tunes, and mechanistic explanations of which pruned units are recoverable

Method note (read first): in this session `arxiv.org`, `export.arxiv.org`, `huggingface.co` and `alphaxiv.org` could NOT be fetched directly (network allow-list / DNS failure). All "verified" claims below were checked against search-engine extracts of the primary paper (arXiv abstract/HTML, ACL Anthology, NeurIPS/ICLR proceedings pages), not full-text reads. Claims tagged **[unverified]** come from prior knowledge and were not re-checked this session; the report writer should not state their numbers as confirmed without that tag. Venue/ID of each paper is given; when a user-provided ID matched, that is noted.

## Q7: Do pruning masks / importance rankings transfer between fine-tunes (base ↔ instruct / domain-adapted) of the same architecture?

### Takeaway
No paper found tests the exact case of "a structured mask computed on the base model, applied to the instruct/domain fine-tune". The indirect evidence points one way. **Coarse importance (which layers matter) is largely set in pre-training and survives post-training.** **Fine-grained masks (weights/neurons) depend on task, domain and calibration data and transfer poorly**, except for masks found with the pre-training objective, which transfer broadly. For an OCR VLM, this means layer rankings from the base (or another fine-tune) are probably reusable, but neuron/channel importance should be recomputed on the OCR fine-tune with OCR calibration data.

### Cited Findings
**Lottery-ticket transfer (foundational)**
- Morcos, Yu, Paganini, Tian, "One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers", NeurIPS 2019, arXiv 1906.02773 (user's ID matches). Winning tickets transferred across natural-image datasets (Fashion-MNIST, SVHN, CIFAR-10/100, ImageNet, Places365), "often achieving performance close to that of winning tickets generated on the same dataset". Tickets "generated using larger datasets consistently transferred better" than tickets from smaller datasets (wording in arXiv v1). They also transfer across optimizers. — [arXiv 1906.02773](https://arxiv.org/abs/1906.02773v1); [NeurIPS poster](https://neurips.cc/virtual/2019/poster/13638)
- Morcos et al. caveat: a NeurIPS reviewer noted that generalization does "not seem that this is true consistently", and that comparisons with non-lottery pruning baselines were missing. — [NeurIPS 2019 reviews](https://papers.neurips.cc/paper_files/paper/2019/file/a4613e8d72a61b3b69b32d040f89ad81-Reviews.html)
- Chen, Frankle, Chang, Liu, Zhang, Wang, Carbin, "The Lottery Ticket Hypothesis for Pre-trained BERT Networks", NeurIPS 2020, arXiv 2007.12223 (user's ID matches). Verified claims:
  - Matching subnetworks exist at 40%–90% sparsity for a range of downstream tasks.
  - They exist at the pre-trained initialization, not only after some training.
  - "Subnetworks found on the masked language modeling task (the same task used to pre-train the model) transfer universally; those found on other tasks transfer in a limited fashion if at all."
  - Note: this is unstructured IMP (iterative magnitude pruning) on BERT, not structured pruning of a generative model. Per-GLUE-task transfer numbers were not retrieved (gap).
  - Sources: [ACM DL](https://dl.acm.org/doi/10.5555/3495724.3497052); [Semantic Scholar](https://www.semanticscholar.org/paper/The-Lottery-Ticket-Hypothesis-for-Pre-trained-BERT-Chen-Frankle/389036b1366b64579725457993c1f63a4f3370ba); [mlanthology](https://mlanthology.org/neurips/2020/chen2020neurips-lottery/)
- **[unverified]** Liang et al., "Super Tickets in Pre-Trained Language Models: From Model Compression to Improving Generalization", ACL 2021, arXiv 2105.12002. At low compression, a structured head/neuron subnetwork of BERT/MT-DNN, after fine-tuning, matches or beats the full model ("super tickets"). Task-specific super tickets were shared across tasks in multi-task learning. Not re-checked this session.
- **[unverified]** Prasanna, Rogers, Rumshisky, "When BERT Plays the Lottery, All Tickets Are Winning", EMNLP 2020, arXiv 2005.00561. Structured (head + MLP) "good" subnetworks for GLUE tasks vary across tasks and random seeds. The "bad" subnetworks (pruned the opposite way) can be re-fine-tuned to close to full performance. This bears on Q8 recoverability: in BERT, the choice of structured units mattered little once retraining was allowed. Not re-checked this session.

**Layer importance from base to post-trained LLMs (most direct evidence for depth rankings)**
- Nepal, Shrestha, Shrestha, Kim, Naghiyev, Shwartz-Ziv, Ross, "Layer Importance for Mathematical Reasoning is Forged in Pre-Training and Invariant after Post-Training", arXiv 2506.22638, NeurIPS 2025 MATH-AI workshop.
  - Method: layer-wise ablation on base models and their post-trained variants (instruction tuning, distillation, RLVR).
  - "Critical layers remain largely invariant across all four model variants"; post-training "tunes these layers rather than restructuring them".
  - Ablating the critical layers gives about an 80% drop in math accuracy. Factual recall has "more distributed layer importance", with much smaller drops from single-layer ablation.
  - Sources: [arXiv HTML](https://arxiv.org/html/2506.22638v2); [author page](https://www.ravid-shwartz-ziv.com/publication/layer-importance-math-reasoning/)

**Domain/task dependence of fine-grained masks in LLMs**
- Zhang et al. (D-Pruner), "Pruning as a Domain-specific LLM Extractor", Findings of NAACL 2024, arXiv 2405.06275.
  - Setup: LLaMA2-7B at 50% (unstructured) sparsity; overlap measured as the share of unpruned weights two masks have in common.
  - Masks from different domains "differed a lot, with shared elements as low as 35%".
  - Self-attention contributes more to domain specificity; "MLP modules store knowledge that is shared by different domains".
  - Sources: [ACL Anthology PDF](https://aclanthology.org/2024.findings-naacl.91.pdf); [arXiv](https://arxiv.org/pdf/2405.06275)
- Hou et al., "Instruction-Following Pruning for Large Language Models" (IFPruning), ICML 2025, arXiv 2501.02086.
  - Method: an input-dependent structured mask predictor.
  - Lower layers (especially layer 1) activate "very similar sub-networks for different inputs"; higher layers make more domain-specific selections.
  - Shows that a single static FFN-width mask is a compromise across tasks.
  - Sources: [arXiv](https://arxiv.org/abs/2501.02086); [ACM DL](https://dl.acm.org/doi/10.5555/3780338.3781275)
- MaskLLM (Fang et al., NeurIPS 2024): learnable 2:4 semi-structured masks. It reports transfer of sparsity patterns across tasks and uses a general-domain mask as a prior for learning task-specific masks. Masks carry reusable structure but are refined per task. The snippet did not say whether masks move between fine-tuned variants. — [NeurIPS 2024 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/0e9a05f5ce62284c91e4a33498899124-Paper-Conference.pdf)
- "Iterative Structured Pruning for LLMs with Multi-Domain Calibration" (EACL 2026 Industry, arXiv 2601.02674): "activation statistics vary across data domains, affecting pruning accuracy"; it uses hybrid multi-domain calibration. — [arXiv HTML](https://arxiv.org/html/2601.02674); [ACL Anthology](https://aclanthology.org/2026.eacl-industry.1.pdf)
- "Rethinking Layer Redundancy: Calibration Matters More Than Search in LLM Depth Pruning" (arXiv 2604.24938) argues that different calibration configurations induce different sets of removable layers. So even depth rankings are not fully calibration-invariant. — [arXiv HTML](https://arxiv.org/html/2604.24938v3)

**Calibration-data dependence (post-training pruning)**
- Williams & Aletras, "On the Impact of Calibration Data in Post-training Quantization and Pruning". Venue: ACL 2024 **main conference, Long Papers** (pp. 10100–10118, DOI 10.18653/v1/2024.acl-long.544), not Findings; arXiv 2311.09755.
  - Describes itself as the first empirical study of calibration data in LLM compression.
  - Recommends evaluating downstream performance across multiple calibration sets to measure sensitivity.
  - SparseGPT mostly outperformed Wanda in their experiments.
  - The exact variance tables and the "pruning more sensitive than quantization" claim were not retrieved (gap).
  - Sources: [ACL Anthology](https://aclanthology.org/2024.acl-long.544); [arXiv](https://arxiv.org/abs/2311.09755v2)
- Bandari et al., "Is C4 Dataset Optimal for Pruning? An Investigation of Calibration Data for LLM Pruning", arXiv 2410.07461 (venue **[unverified]**, believed EMNLP 2024). — [arXiv PDF](https://arxiv.org/pdf/2410.07461)
- Counterpoint on seed-level stability within one calibration source: SparseGPT reported 13.52 ± 0.075 perplexity over 5 calibration seeds at 50% sparsity. Wanda reported low variance across seeds. Same-distribution resampling is stable; changing the distribution is what matters. — [SparseGPT arXiv 2301.00774](https://arxiv.org/pdf/2301.00774); [Wanda arXiv 2306.11695](https://arxiv.org/pdf/2306.11695)

**Structured pruning + distillation pipelines and the base/instruct question**
- Sreenivas et al., "LLM Pruning and Distillation in Practice: The Minitron Approach", arXiv 2408.11796.
  - Prunes base Llama-3.1-8B to 4B and Mistral-NeMo-12B to 8B.
  - Adds "teacher correction": a light fine-tune of the teacher on the distillation dataset (reported as about 127B tokens in one summary vs. about 100B in another, so the token counts conflict across secondary sources), because the original pre-training data was unavailable.
  - Distillation uses logit-only forward KL (CE loss ignored), about 94B tokens.
  - Width beat depth for Llama-3.1-Minitron-4B: MMLU 60.5 vs 58.7; GSM8K about 41.2 vs 16.8. Depth gives about 2.7× throughput vs about 1.8× for width.
  - Width-pruned models start with higher LM loss right after pruning, but "this trend reversed after a short period of retraining".
  - Sources: [arXiv](https://arxiv.org/pdf/2408.11796); [arXiv HTML v2](https://arxiv.org/html/2408.11796v2)
  - **[unverified]** The instruct versions were produced by aligning the pruned base, not by transferring a mask to Llama-3.1-8B-Instruct.
- Muralidharan et al., "Compact Language Models via Pruning and Knowledge Distillation" (Minitron), NeurIPS 2024, arXiv 2407.14679. Recommends width over depth at ≤15B scale. Recommends logit + intermediate-state + embedding distillation when depth is reduced significantly. — [NeurIPS 2024 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/4822991365c962105b1b95b1107d30e5-Paper-Conference.pdf)
- Compresso (arXiv 2310.05015): learns binary structured masks with LoRA on the frozen LLM, plus "collaborative prompting". Masks are learned per model; no cross-fine-tune transfer test was found. — [arXiv](https://arxiv.org/pdf/2310.05015)
- **[unverified]** Sheared LLaMA (Xia et al., ICLR 2024, arXiv 2310.06694): targeted structured pruning to a fixed shape, then continued pre-training with dynamic batch loading. Pruned the base model; did not test mask transfer.
- LLM-Pruner's own Llama-3/3.1 recipe prunes Meta-Llama-3-8B-Instruct directly, computing importance on the instruct weights rather than transferring from the base. — [GitHub horseee/LLM-Pruner](https://github.com/horseee/LLM-Pruner)
- "Pruning for Protection" (arXiv 2401.10862) prunes chat models (LLaMA-2 Chat, Vicuna) per model; it does not reuse masks. — [arXiv](https://arxiv.org/pdf/2401.10862)
- Xu & Lu, "Small LLMs: Pruning vs. Training from Scratch", arXiv 2606.14150 (June 2026 preprint). Setup: Llama-3.1-8B pruned at ratios 0.5–0.8 with six methods.
  - Under equal tokens, the pruned-init advantage shrinks with pruning ratio; for depth pruning it is "nearly gone" at the most aggressive ratio, while width keeps a positive edge.
  - Loss gaps vs scratch shrink with training: 0.39 → 0.04 (depth) and 0.49 → 0.09 (width).
  - With the full pipeline token budget, scratch training matches or beats coarse structured pruning (Minitron-depth scratch 66.2% avg).
  - Finer-grained and sparse pruning keep a lasting advantage.
  - Search-snippet numbers only; the snippet metadata was partly inconsistent.
  - Source: [arXiv HTML](https://arxiv.org/html/2606.14150v1)

### Inferences
- **Layer/block rankings (depth importance) probably transfer base ↔ fine-tune.** Layer importance is "forged in pre-training" (2506.22638), and early-layer subnetworks are input-invariant (IFPruning). The OCR fine-tunes share the base's pre-training, so a layer-drop ranking computed on one fine-tune is a reasonable prior for the sibling fine-tune. Re-check it with a few hundred OCR calibration samples, since depth rankings still depend on calibration (2604.24938).
- **Neuron/channel rankings transfer poorly across tasks/domains.** Evidence: D-Pruner's 35% mask overlap, Chen 2020's "limited if at all" for task-found masks, IFPruning's domain-specific upper layers, and Williams & Aletras. An OCR-domain VLM shifts activation statistics a lot (visual tokens, layout/text-dense outputs), so FFN-width importance should be recomputed on the OCR fine-tune with OCR-like calibration (image + transcription), not borrowed from a text-only base or chat model.
- **Exception: masks found with the pre-training objective.** Chen 2020's MLM masks transfer "universally". A neuron ranking computed on the base model with generic LM data may be a robust starting point, but probably suboptimal compared with a task-specific ranking. This is untested for structured generative pruning.
- **Possible tension with Minitron.** Minitron found width pruning MORE recoverable than depth after distillation: higher initial loss, lower loss after short retraining. The user observes the opposite: neurons are about 10× less recoverable per unit of zero-shot KL. The difference could come from scale (8B vs 0.8B), importance estimator (activation-based vs other), architecture (hybrid GatedDeltaNet), distillation budget (about 94B tokens vs likely far less), or the normalization (recovery per unit initial damage vs absolute final quality). Worth stating explicitly in the report.

### Gaps
- No primary study found that computes a structured mask on base Llama/Qwen and applies it unchanged to its instruct or domain fine-tune, then measures the gap against a natively computed mask. This appears to be an open, directly testable question for the user's project.
- No mask-overlap (IoU/Jaccard) numbers found for structured FFN-neuron masks between base and instruct checkpoints of the same model.
- Not retrieved: Chen 2020's per-GLUE-task transfer matrix; Williams & Aletras's quantitative variance numbers; Bandari et al.'s results.
- "Elastic weight sharing" / Flextron-style nested models were not investigated (out of the tool budget).
- No VLM-specific (vision-token) mask-transfer study found.

## Q8: Mechanistic explanations, and is there work linking mechanisms to which pruned units are RECOVERABLE (not just zero-shot robust)?

### Takeaway
There is strong mechanistic literature on **zero-shot** robustness:
- residual "ensemble of shallow paths" and layer-deletion robustness;
- near-linear inter-layer maps;
- self-repair / the Hydra effect;
- FFN-as-key-value-memory and localized factual recall;
- knowledge-intensive abilities degrading first under width/weight pruning.

Very little work connects these mechanisms causally to **post-retraining recoverability**. The closest 2025–2026 evidence:
1. Layer-removal damage is largely a **boundary/representation misalignment**, fixable by a closed-form linear map. This is shown by ReplaceMe, LinearPatch, Ghosted Layers and "Your transformer is secretly linear", and by the finding that the optimal set of layers to drop changes once repair is accounted for (2610.30552).
2. Width/FFN pruning hits **fact recall** first (Jin et al. ICLR 2024; LLM-KICK). Fact recall is the capability that is most data-hungry to re-learn.

This supports the user's "misalignment vs. lost knowledge" framing. It is, however, an inference, not a demonstrated causal result. Minitron's width > depth after distillation is a counterexample.

### Cited Findings
**Residual nets as ensembles; layer-deletion robustness**
- **[unverified numbers]** Veit, Wilber, Belongie, "Residual Networks Behave Like Ensembles of Relatively Shallow Networks", NeurIPS 2016, arXiv 1605.06431. Deleting single residual blocks at test time barely hurts ImageNet/CIFAR ResNets, except at downsampling blocks; error increases smoothly with the number of blocks removed. Most gradient comes from short paths (for a 110-layer ResNet, effective paths of about 10–34 layers). Not re-fetched this session; cite with the tag.
- Lad, Gurnee, Tegmark (later version adds J. H. Lee), "The Remarkable Robustness of LLMs: Stages of Inference?", arXiv 2406.19384 (user's ID matches). ICML 2024 workshop; NeurIPS 2025.
  - Deleting or swapping adjacent layers at inference with no fine-tuning retains 72–95% of original top-1 prediction accuracy; deeper models are more robust; models are especially robust to dropping middle layers.
  - Proposes four stages: detokenization → feature engineering → prediction ensembling → residual sharpening (renamed "residual calibration" in the NeurIPS 2025 version).
  - Models: 8 models incl. Pythia, GPT-2, Phi.
  - Sources: [arXiv PDF v3](https://arxiv.org/pdf/2406.19384); [mlanthology NeurIPS 2025](https://mlanthology.org/neurips/2025/lad2025neurips-remarkable/); [mlanthology ICMLW 2024](https://mlanthology.org/icmlw/2024/lad2024icmlw-remarkable/)
- Gromov, Tirumala, Shapourian, Glorioso, Roberts, "The Unreasonable Ineffectiveness of the Deeper Layers", arXiv 2403.17887 (user's ID matches), ICLR 2025.
  - Pruning: remove the block of layers whose input/output representations are most similar.
  - Healing: a small QLoRA run on pretraining-style data (C4).
  - Llama-2-70B: "nearly half of the layers" can be pruned before a significant QA drop.
  - Healing gives a "modest but consistent" improvement. After healing, the simple "drop the deepest layers" heuristic nearly matches similarity-based selection on MMLU/BoolQ/C4 loss.
  - Recoverability relevance: healing mostly repairs the mismatch at the cut. QA accuracy still drops sharply past a threshold, which secondary reviewers read as knowledge loss that light healing does not restore (interpretive).
  - Sources: [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/cbabc2f70de2dd09f491a8715ec3e80f-Paper-Conference.pdf); [mlanthology](https://mlanthology.org/iclr/2025/gromov2025iclr-unreasonable/)
- **[unverified]** Sun, Pickett, Nain, Jones, "Transformer Layers as Painters", arXiv 2407.09298 (user's ID; believed AAAI 2025).
  - Middle layers share a representation space and are robust to skipping/reordering.
  - First and last layers are critical; parallel execution of middle layers degrades gracefully.
  - Math/reasoning-heavy tasks are more sensitive to layer order.
  - Not re-fetched.

**Inter-layer near-linearity and linear repair of depth pruning ("misalignment, not lost depth")**
- Razzhigaev et al., "Your Transformer is Secretly Linear", ACL 2024 (long, pp. 5376–5384), arXiv 2405.12250 (user's ID matches).
  - Embedding transforms between consecutive decoder layers have a generalized-Procrustes linearity score of about 0.99.
  - Linearity drops if the residual component is removed.
  - "Removing or linearly approximating some of the most linear blocks… does not affect significantly the loss or model performance."
  - A cosine-similarity regularizer that reduces linearity during pretraining improved small-model benchmarks.
  - Sources: [ACL Anthology](https://aclanthology.org/2024.acl-long.293); [arXiv HTML](https://arxiv.org/html/2405.12250v1)
- Shopkhoev et al. (MTS AI), "ReplaceMe: Network Simplification via Depth Pruning and Transformer Block Linearization", arXiv 2505.02819 (user's ID matches), listed as NeurIPS 2025.
  - Training-free: replaces a contiguous run of blocks with one linear map, estimated on a small calibration set and merged into the adjacent MLP (no extra parameters).
  - Blocks selected by cosine distance.
  - "Up to 25% pruning while retaining ~90%" of benchmark performance with no healing.
  - Sources: [arXiv PDF](https://arxiv.org/pdf/2505.02819); [arXiv HTML](https://arxiv.org/html/2505.02819)
- "A Simple Linear Patch Revives Layer-Pruned Large Language Models" (LinearPatch), arXiv 2505.24680: a channel-wise scaling patch at the pruning boundary. — [arXiv PDF](https://arxiv.org/pdf/2505.24680)
- Yun, Jo, Karimireddy, Lee, "Ghosted Layers: Unconstrained Activation Alignment for Recovering Layer-Pruned LLMs", arXiv 2605.15491 (reported accepted to NeurIPS 2026 per the lab site).
  - Frames the damage explicitly: "the next surviving layer receives a hidden state it was not trained on".
  - Inserts one closed-form optimal unconstrained linear operator per pruned region (contiguous or not).
  - Reports gains over constrained training-free methods (e.g. LinearPatch) at matched cost; author-reported only.
  - Sources: [arXiv](https://arxiv.org/abs/2605.15491); [arXiv HTML](https://arxiv.org/html/2605.15491v1)
- Su, Xiang, Zhang, Zhang, "Repair Changes Which Layers to Remove: Exact Optima for Depth Pruning of Language Models", arXiv 2610.30552 (preprint, 22 Oct 2026, not peer-reviewed). **The closest paper found to "recoverability-aware selection".**
  - Exhaustively evaluated about 53,000 pruned models over 4 families, with and without linear repair.
  - Without repair, removing nearby layers costs more than the sum of their individual costs (super-additive), so the optimum spreads removals out. With linear repair, the optimum becomes one contiguous block.
  - Selecting layers while ignoring the repair misses the repaired optimum by up to 189% (on their metric).
  - Their repair-aware greedy search cuts perplexity 20–66% vs ReplaceMe on 5 models up to 27B.
  - Fitting the map to the removed run's output, computed on the repaired network's own inputs, was about 5% better on Qwen3-1.7B than fitting to the intact network's activations.
  - Source: [arXiv](https://arxiv.org/abs/2610.30552); [arXiv HTML](https://arxiv.org/html/2610.30552)
- **[unverified]** LLM-Streamline (Chen et al., arXiv 2403.19135; later ICLR 2025 version believed): replaces a contiguous block of low-importance layers with a lightweight trained network (FFN/transformer layer) fitted to hidden states. No source retrieved this session.

**Self-repair (why zero-shot damage under-estimates or over-estimates true importance)**
- McGrath, Rahtz, Kramár, Mikulik, Legg, "The Hydra Effect: Emergent Self-repair in Language Model Computations", arXiv 2307.15771 (user's ID matches), Google DeepMind 2023.
  - Ablating one attention layer causes another layer to compensate.
  - Late MLP layers act as "erasure"/counterbalancing units that down-regulate the max-likelihood token.
  - Layers are "relatively loosely coupled".
  - The effect occurs even in models trained without dropout.
  - A secondary review reports the compensation regression slope is below 1 for later layers, so total effects are attenuated but nonzero.
  - Source: [arXiv](https://arxiv.org/abs/2307.15771)
- Rushing & Nanda, "Explorations of Self-Repair in Language Models", ICML 2024, arXiv 2402.15390.
  - Self-repair exists across model families and sizes when ablating individual heads on the full training distribution, but is imperfect and noisy.
  - Two mechanisms: changes in the final LayerNorm scaling factor (which can repair up to about 30% of the direct effect **[unverified number]**), and sparse sets of neurons implementing "Anti-Erasure".
  - Venue and the 30% figure are not re-verified this session; the two-mechanism description was confirmed via a Semantic Scholar snippet.
  - Source: [Semantic Scholar via search](https://www.semanticscholar.org/paper/c8fd279848e9c50eec85b852f22f7fffb08f97fc)
- Recent follow-ups surfaced but not read: "LLM Layers Immediately Correct Each Other" (arXiv 2609.07876) and "Conditional Co-Ablation: Recovering Self-Repair Backups in Transformer Circuits" (arXiv 2607.01940). — [2609.07876](https://arxiv.org/pdf/2609.07876); [2607.01940](https://arxiv.org/pdf/2607.01940)

**FFN as key–value memory; localized knowledge; superposition**
- **[unverified]** Geva, Schuster, Berant, Levy, "Transformer Feed-Forward Layers Are Key-Value Memories", EMNLP 2021, arXiv 2012.14913.
  - Each FFN key (first-layer row) correlates with human-interpretable input patterns: shallow patterns in lower layers, semantic ones in upper layers.
  - Each value (second-layer column) induces a distribution over the output vocabulary.
  - Layer output is a composition of memories refined via the residual.
- **[unverified]** Geva, Caciularu, Wang, Goldberg, "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space", EMNLP 2022, arXiv 2203.14680. FFN updates decompose into sub-updates that promote interpretable concepts in vocabulary space.
- **[unverified]** Dai et al., "Knowledge Neurons in Pretrained Transformers", ACL 2022, arXiv 2104.08696. Integrated-gradient attribution finds a small set of FFN neurons whose suppression or amplification modulates the expression of specific relational facts.
- **[unverified]** Meng, Bau, Andonian, Belinkov, "Locating and Editing Factual Associations in GPT" (ROME), NeurIPS 2022, arXiv 2202.05262. Causal tracing localizes factual recall to mid-layer MLPs at the last subject token; rank-one edits of those MLPs change facts.
- **[unverified]** Elhage et al., "Toy Models of Superposition", Transformer Circuits Thread 2022 (transformer-circuits.pub/2022/toy_model; arXiv 2209.10652).
  - Models represent more sparse features than dimensions by superposing them in non-orthogonal directions; neurons become polysemantic.
  - Implication for neuron pruning: removing one neuron partially deletes many features and adds interference to the rest, so a neuron's importance is not separable.
- Jin et al., "The Cost of Down-Scaling Language Models: Fact Recall Deteriorates before In-Context Learning", arXiv 2310.04680 (user's ID matches; the ICLR 2024 camera-ready is retitled "The Cost of Scaling Down Large Language Models: Reducing Model Size Affects Memory before In-context Learning").
  - Reducing size by more than 30% (via weight pruning OR training smaller dense models) "significantly decreases the ability to recall facts seen in pre-training".
  - A 60–70% reduction largely preserves in-context processing (long-context retrieval, learning functions from examples).
  - A secondary summary reports FFW layers matter more for fact recall than attention.
  - Sources: [OpenReview](https://openreview.net/forum?id=ldJXXxPE0L); [arXiv](https://arxiv.org/abs/2310.04680); [mlanthology ICLR 2024](https://mlanthology.org/iclr/2024/jin2024iclr-cost/)
- Jaiswal et al., "Compressing LLMs: The Truth is Rarely Pure and Never Simple" (LLM-KICK), arXiv 2310.01382 (user's ID matches), ICLR 2024 **[venue unverified]**.
  - Knowledge-intensive tasks show significant drops starting at 25–30% sparsity.
  - Pruned models at ≥50% sparsity remain robust for in-context retrieval and summarization.
  - Secondary-source numbers only.
  - Source: [Lacuna summary](https://lacuna.tiptreesystems.com/work/compressing-llms-the-truth-is-rarely-pure-5s8f)
- D-Pruner (above): "MLP modules store knowledge that is shared by different domains". — [ACL Anthology](https://aclanthology.org/2024.findings-naacl.91.pdf)
- Layer-importance invariance paper (above): factual recall shows "more distributed layer importance" than math. This is consistent with knowledge being spread across many FFNs rather than concentrated in a few layers. — [arXiv HTML](https://arxiv.org/html/2506.22638v2)

**Depth vs width after retraining (recoverability evidence, mixed)**
- Minitron 2408.11796: width starts worse but ends better after short retraining (see Q7). — [arXiv](https://arxiv.org/pdf/2408.11796)
- Xu & Lu 2606.14150: under matched tokens, width keeps its pruned-init advantage at high ratios while depth's nearly vanishes. Coarser structure is easier to match from scratch. — [arXiv HTML](https://arxiv.org/html/2606.14150v1)
- "Recovering Aggressively Pruned Vision-Language-Action Models with Offline Hidden-State Distillation" (arXiv 2609.19579):
  - Width pruning beat a CKA-guided depth baseline at every tested point, by 2.8–33.2 points.
  - Depth-pruned OpenVLA-OFT reached 0% success at all three points, vs 54.4% retained for width at 45%.
  - Their argument: "a removed layer eliminates every computation it performed, while a narrowed layer retains the highest-scoring heads and channels at every depth".
  - Source: [arXiv PDF](https://arxiv.org/pdf/2609.19579)
- **[unverified]** Kim et al., "Shortened LLaMA: Depth Pruning for LLMs with Comparison of Retraining Methods", arXiv 2402.02834. Depth pruning is competitive with width under memory-bound inference. Continued pre-training on a large corpus markedly outperforms LoRA for recovery at high pruning ratios. — [arXiv HTML](https://arxiv.org/html/2402.02834v2)
- "Distilled Circuits: A Mechanistic Study of Internal Restructuring in Knowledge Distillation" (arXiv 2505.10822) is a mechanistic study of how student circuits reorganize under KD. It surfaced in search but was not read. — [arXiv PDF](https://arxiv.org/pdf/2505.10822)

### Inferences
- **Supported mechanistic story for "layers are cheap to recover".** Inter-layer maps are near-linear (0.99 Procrustes) and middle layers share a representation space (Lad et al.; Painters [unverified]). Much of the zero-shot damage from dropping a layer is a distribution shift at the boundary that a linear map (ReplaceMe, Ghosted Layers) or a few distillation steps can absorb. Self-repair (Hydra; Rushing & Nanda) means downstream layers already partly compensate. The user's KL-per-unit metric would show high initial damage, but much of it is "misalignment" that a little training removes. This matches the "ensemble of shallow paths" view: the remaining paths still compute most features.
- **Supported story for "neurons are expensive to recover".** FFN neurons store key→value associations, especially facts (Geva 2021, Dai 2022, ROME). Under superposition, each neuron holds slices of many features. Removing neurons therefore deletes stored content spread across many features. Jin et al. and LLM-KICK show fact/knowledge recall is the first ability lost under width/weight reduction (about 25–30%), while ICL survives to 60–70%. Re-learning stored associations needs data covering those facts. A distillation set with limited coverage (OCR data) cannot restore knowledge that is absent from it. For OCR, the relevant "facts" may be glyph/vocabulary/language priors: character n-grams, rare scripts, LaTeX/table tokens.
- **Testable prediction distinguishing the two stories.** If the layer-vs-neuron gap is misalignment vs lost knowledge, then:
  - (a) fitting a ReplaceMe/Ghosted-style linear map should remove a large fraction of layer-drop KL zero-shot, but very little of neuron-drop KL;
  - (b) after distillation, the residual error of neuron-pruned models should concentrate on rare tokens/scripts (knowledge-like) rather than uniformly across tokens;
  - (c) neuron pruning recovery should improve markedly if distillation data covers the teacher's broad pre-training distribution, not just OCR.

  Also note 2610.30552: selection without repair-awareness can be badly suboptimal. Importance scores computed zero-shot (as in the user's KL damage) are a biased proxy for post-recovery importance.
- **Caveat: the opposite empirical finding exists.** Minitron (8B, about 94B KD tokens) and the VLA paper found width MORE recoverable than depth. The user's "about 10× less recoverable per unit zero-shot KL" normalizes by initial damage. Width pruning may cause less initial KL per removed parameter yet still end better in absolute terms. The report should keep "recoverability ratio" and "final quality at matched parameter/latency budget" separate.
- **Hybrid GatedDeltaNet decoder (no direct literature found).** Linear-attention/recurrent state layers may not share the near-linear residual-stream property measured on pure transformers. Whether layer-drop recoverability differs between DeltaNet layers and full-attention layers is untested in the sources found.

### Gaps
- No paper found that directly measures, for the same model, recovery-after-distillation per unit of zero-shot damage for neurons vs layers, nor one that causally ties recoverability to key-value memory content or superposition. The mechanistic link is inferential.
- Veit 2016, Geva 2021/2022, Dai 2022, ROME, Toy Models, Painters, Super Tickets, Prasanna 2020, Shortened LLaMA, LLM-Streamline: claims are from prior knowledge, not re-fetched (arXiv was blocked in this session).
- The exact Rushing & Nanda LayerNorm-repair percentage and the LLM-KICK primary numbers were not verified.
- No study found on self-repair or linear-repair behavior in hybrid linear-attention (GatedDeltaNet/Mamba-style) models or in VLM decoders with visual tokens.
- "Not all layers are equal" / "Do LLMs really need all layers" style papers were not separately retrieved (budget). Lad et al., Gromov, and 2506.22638 cover the same ground.
