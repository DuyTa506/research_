# Training-free predictors of post-recovery quality and post-compensation error as a predictor (Q2, Q4)

Method note: in this session arxiv.org, semanticscholar.org and similar hosts could not be fetched (DNS blocked). Every finding below comes from WebSearch result summaries of primary pages (arXiv abstract/HTML/PDF, CVF, NeurIPS proceedings, ACL Anthology, ICLR blog track). Items marked **(unverified)** come from background knowledge and were not confirmed against the source in this session. Before citing exact numbers in the paper, the report writer should open the PDFs.

## Q2: Is there a training-free or cheap quantity that predicts post-recovery (finetune/distillation) quality of a pruned model, how well, and across which unit types?

### Takeaway
Some cheap predictors exist and a few report strong rank correlations. None of them, as far as I could find, is validated across heterogeneous unit types (layers vs MLP neurons vs heads vs ViT blocks) or for VLMs or hybrid linear-attention decoders. The most relevant are:
- TraceNAS (2602.02891): gradient-trace correlation with the parent. Reports Spearman ρ=0.94 and Kendall τ=0.82 over about 70 pruned LLM candidates. It covers both depth and width candidates.
- P² Law (2411.10272, ACL 2025): a fitted scaling law for post-training loss.
- PRACTISE "recoverability" (CVPR 2023): CNN block dropping. The authors say it is "more related to accuracy after finetuning" than previous criteria, but I found no coefficient.

Classic zero-cost NAS proxies predict trained accuracy poorly and are often beaten by FLOPs or parameter count. Minitron reports that pre-retraining rankings are unstable and uses short distillation (about 1.8B tokens) to stabilize them. That is consistent with the user's observation that raw zero-shot damage does not rank models after distillation.

### Cited Findings

**LLM structured pruning: proxies for post-recovery quality**
- **TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation.** arXiv 2602.02891 (Feb 2026). Authors: Malettira, Nagaraj, Roy, Negi, K. Roy (Purdue).
  - What it is: a training-free NAS over joint depth and width structured pruning of LLMs (Llama and Qwen families).
  - The proxy scores a candidate by how closely its gradient "trace" (loss-landscape alignment) matches the pretrained parent's. The stated goal is to pick candidates that recover best after post-pruning training. The proxy is described as a scale-invariant "Sparsity-Weighted Pearson Correlation" of gradients.
  - Cost: about 8.5 h on one GPU, roughly 10x fewer GPU-hours than training-aware methods.
  - Sources: [arXiv abs](https://arxiv.org/abs/2602.02891); [arXiv HTML](https://arxiv.org/html/2602.02891v1).
- TraceNAS reported numbers:
  - Proxy vs downstream accuracy: Spearman ρ=0.94, Kendall τ=0.82.
  - Baselines: unweighted Pearson 0.77/0.62 and cosine similarity 0.91/0.79.
  - Computed over about 70 models with 3 seeds (per the alphaXiv summary).
  - A PDF snippet also mentions accuracy correlations of 0.864 and 0.902, a PPL correlation of 0.932, and 0.900 accuracy correlation at gradient-subspace rank r=2. The context of these is unclear (likely ablations; appendix Tables 6–7).
  - Sources: [arXiv HTML](https://arxiv.org/html/2602.02891v1); [alphaXiv summary](https://www.alphaxiv.org/de/abs/2602.02891); [PDF](https://www.arxiv.org/pdf/2602.02891).
  - Caveat: I could not confirm from the snippets whether the target is accuracy after the post-pruning training phase or one-shot accuracy. The abstract frames the proxy as predicting recovery after training. Verify which in the full text.
  - TraceNAS is gradient-based, not compensation-based. It mixes depth and width candidates, but I did not find a per-unit-type correlation breakdown.
- **P² Law: Scaling Law for Post-Training After Model Pruning.** Chen et al., arXiv 2411.10272, ACL 2025 (long, 2025.acl-long.283).
  - What it predicts: the post-training loss of a pruned LLM from four inputs: model size before pruning, number of post-training tokens, pruning rate, and loss before pruning.
  - Fit on Llama-3 and Qwen-2.5 pruned with several common methods (depth and width).
  - Reports that it extrapolates to larger datasets, larger models and higher pruning rates. It is a fitted law (it needs some post-training runs), not a zero-shot ranker.
  - Sources: [arXiv](https://arxiv.org/html/2411.10272v2); [ACL Anthology](https://preview.aclanthology.org/setup/2025.acl-long.283).
  - Exact R²/error values were not retrieved. Whether "loss before pruning" means the parent's loss (which is how the summary reads) or the pruned model's zero-shot loss: verify.
- **Pruning Laws for Large Language Models.** arXiv 2504.04342.
  - Fits laws of post-pruning performance vs pruning ratio.
  - Models: OPT 1.3B–30B and LLaMA-2 7B/13B, plus LLaMA-3.1-8B, LLaMA-3.2-3B and Phi-3-mini; an MoE depth sweep on GPT-OSS-20B with ShortGPT.
  - Methods: SparseGPT (unstructured), LaCo (depth), SliceGPT (width), plus SlimGPT, LLM-Pruner, SVD-LLM and PruneNet.
  - Tests in-distribution with rolling splits (20–90% ratios) and extrapolation in zero-shot and one-shot setups.
  - Source: [arXiv PDF](https://arxiv.org/pdf/2504.04342).
  - From the snippets this is a curve-fitting/extrapolation paper over pruning ratio per method. I found no evidence that it predicts post-recovery quality, or that it ranks unit types by recoverability. The user's description should be corrected accordingly. Do not confuse it with "The Neural Pruning Law Hypothesis" (2504.05349).
- **A Free Lunch in LLM Compression: Revisiting Retraining after Pruning.** Wagner, Roux, Zimmer, Pokutta; arXiv 2510.14444 (v1 16 Oct 2025, v2 6 Feb 2026).
  - Studies how to reconstruct weights after pruning, i.e. local reconstruction vs full retraining.
  - Main findings:
    - Reconstructing attention and MLP separately within each block is Pareto-optimal and "beats full retraining" at a fraction of the memory.
    - With proper reconstruction, simple criteria (Wanda) can beat more complex ones.
  - Source: [arXiv](https://arxiv.org/html/2510.14444v1).
  - It is not a predictor paper: it does not propose post-reconstruction error as a predictor of post-retraining quality. It is relevant to Q4 because it shows reconstruction granularity matters and can substitute for retraining.
- **Minitron: Compact Language Models via Pruning and Knowledge Distillation.** Muralidharan et al., arXiv 2407.14679, NeurIPS 2024.
  - Searched pruned candidates' pre-retraining rankings are unstable.
  - Their best-practices list includes "perform lightweight retraining to stabilize the rankings of searched pruned candidates", using about 1.8B tokens of distillation before full retraining of the best ones.
  - In one comparison, a method looked better before retraining but "all 3 candidates converge to the same loss value" after retraining.
  - They also note that layer-importance metrics (BI vs LM loss) are highly correlated with each other but do not give the most accurate pruned model on Winogrande.
  - Sources: [arXiv HTML](https://arxiv.org/html/2407.14679v1); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/4822991365c962105b1b95b1107d30e5-Paper-Conference.pdf).
  - I did not retrieve a Spearman number for pre- vs post-retraining rank agreement.
- **Puzzle: Distillation-Based NAS for Inference-Optimized LLMs.** arXiv 2411.19146, ICML 2025.
  - Builds a library of block variants (attention variants and FFN variants) trained by Blockwise Local Distillation (BLD) against the parent. Decoupled BLD needs (m+n)·l variants instead of m·n·l.
  - Scoring: "replace-1-block". Each variant at each location is swapped alone into the parent and a whole-model performance measure is computed.
  - Then a Mixed-Integer Program assembles the architecture under memory, throughput and latency constraints.
  - Sources: [arXiv PDF](https://arxiv.org/pdf/2411.19146); [GitHub NVlabs/puzzle](https://github.com/NVlabs/puzzle/); [ICML poster](https://icml.cc/virtual/2025/poster/45275).
  - The exact scoring metric (KL vs LM loss vs downstream accuracy) was not confirmed by the snippets (unverified; I recall KL divergence and LM loss being used). Puzzle scores blocks after local repair (BLD = a trained local compensation), measured at the whole-model output. It is conceptually close to "damage after local compensation, measured at the output", but the repair is trained rather than closed-form, and the paper does not report a correlation of this score with final post-distillation quality.
- **Rethinking Layer Redundancy: Calibration Matters More Than Search in LLM Depth Pruning.** arXiv 2604.24938 (2026).
  - Under perplexity-based depth pruning, calibration perplexity and downstream reasoning accuracy show negative or weak rank correlations. They correlate positively under a task-likelihood-margin calibration objective.
  - Evaluated without fine-tuning.
  - Source: [arXiv HTML](https://arxiv.org/html/2604.24938v3).
- **Pruning Large Language Models via Accuracy Predictor.** arXiv 2309.09507. Trains a learned accuracy predictor over pruned-architecture/accuracy pairs. It is not training-free. [arXiv PDF](https://arxiv.org/pdf/2309.09507)
- **Flextron.** Cai et al., ICML 2024, arXiv 2406.10260 **(unverified, not retrieved this session)**. An elastic nested LLM with routers and a learned latency/accuracy surrogate. I found no evidence of a training-free post-recovery predictor.

**CNN-era "recoverability" (closest prior art to the user's hypothesis in spirit)**
- **PRACTISE: Practical Network Acceleration with Tiny Sets.** Wang and Wu, CVPR 2023, arXiv 2202.07861; extended version arXiv 2303.00972 ("Hypothesis, Theory, and Algorithm").
  - Defines a **"recoverability"** metric for choosing which blocks to drop when only a tiny training set is available.
  - Conclusion: "compared with previous pruning criteria, recoverability is more related to the model's accuracy after finetuning". Also: "end-to-end finetuning with a tiny set mainly recovers the model's linear ability".
  - Result: +7% on average over prior methods on ImageNet-1k at 22% latency reduction (ResNet-34, MobileNetV2).
  - Sources: [CVF open access](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Practical_Network_Acceleration_With_Tiny_Sets_CVPR_2023_paper.html); [arXiv](https://arxiv.org/abs/2202.07861); [extended version](https://ar5iv.labs.arxiv.org/html/2303.00972).
  - No numeric correlation coefficient was found in the snippets. My recollection, **unverified**: recoverability is measured as the loss remaining after a cheap linear (1x1 conv "adaptor") repair is fitted on the tiny set. If that is right, this is the closest published instance of "damage remaining after local linear compensation predicts post-finetune accuracy". It is only for CNN residual blocks: no cross-unit-type comparison and no KL at the output of an LM. **Must be checked in the full text.**
- Counter-evidence that fine-tuning washes out criteria differences:
  - Mittal et al., "Studying the Plasticity in Deep CNNs using Random Pruning" (arXiv 1812.10240): given enough data and fine-tuning, the gap between criteria, including random pruning, closes. [arXiv PDF](https://arxiv.org/pdf/1812.10240)
  - arXiv 2410.15176: all methods show similar recovery after fine-tuning. [arXiv HTML](https://arxiv.org/html/2410.15176v1)

**Zero-cost NAS proxies (synflow, NASWOT, grad-norm, etc.)**
- **Zero-Cost Proxies for Lightweight NAS.** Abdelfattah et al., ICLR 2021, arXiv 2101.08134 **(paper not retrieved; numbers unverified)**.
  - Proposed pruning-at-init saliencies (synflow, snip, grasp, fisher, grad_norm, jacob_cov) as NAS proxies.
  - My recollection, **unverified**: synflow Spearman ≈0.74–0.76 on NAS-Bench-201 CIFAR-10.
  - A later paper (ZAPS) says Abdelfattah et al. recommend a Borda-style vote of three proxies. [ZAPS arXiv PDF](https://arxiv.org/pdf/2609.14184)
- "A Deeper Look at Zero-Cost Proxies for Lightweight NAS" (ICLR 2022 Blog Track):
  - **FLOPs (a trivial baseline) had the best average rank over all 12 tasks studied.**
  - "For some tasks the majority of ZC proxies have a negative correlation with model performance."
  - Synflow is highly correlated with FLOPs/params and "somewhat acts as a soft parameter counter".
  - Source: [ICLR blog](https://iclr-blog-track.github.io/2022/03/25/zero-cost-proxies/).
- Other evidence on proxies vs size baselines:
  - ParZC (arXiv 2402.02105): most pruning-based proxies (including synflow) "still fail to outperform the baseline" (params/FLOPs). [arXiv PDF](https://arxiv.org/pdf/2402.02105)
  - NEAR (arXiv 2408.08776): only some proxies (reg_swap, SynFlow, ZiCo, MeCoopt, NEAR) beat #params on average. [arXiv PDF](https://arxiv.org/pdf/2408.08776)
- **(unverified, not retrieved)**:
  - NASWOT: Mellor et al., ICML 2021, arXiv 2006.04647.
  - NAS-Bench-Suite-Zero: Krishnakumar et al., NeurIPS 2022 Datasets & Benchmarks, arXiv 2210.03230. Found that proxies largely encode params/FLOPs.
  - White et al., "How Powerful are Performance Predictors in NAS?" (arXiv 2104.01177, NeurIPS 2021). Systematic predictor comparison; listed in search results ([arXiv PDF](https://arxiv.org/pdf/2104.01177)), but numbers were not retrieved.
- These proxies target architectures trained **from scratch**, not pruned and then distilled children of a parent. Their known failure modes are size-confounding and task-dependence. They are not a good template for ranking unit types of a pretrained parent.

**NTK / Fisher / "trainability" metrics**
- I found no primary source this session that uses the NTK condition number or Fisher trace to predict post-distillation quality of *pruned pretrained LLMs/VLMs*. The NTK-based NAS literature (TE-NAS, NASI arXiv 2109.00817) targets from-scratch training ([NASI](https://arxiv.org/pdf/2109.00817)). See Gaps.

### Inferences
- The user's raw-damage finding is consistent with Minitron (rankings before retraining are unstable and converge after retraining) and with arXiv 2604.24938 (calibration PPL can even anti-correlate with accuracy). It is not a novel observation per se, but the cross-unit-type framing (layers vs neurons) appears novel.
- The closest published ideas to "residual after local compensation predicts recovery" are:
  1. PRACTISE recoverability: CNN blocks, tiny-set fitted repair, qualitative claim of better relation to finetuned accuracy.
  2. Puzzle replace-1-block scoring: after trained BLD repair, measured at the whole-model output, used for selection but not reported as a predictor of final quality.
  3. TraceNAS: a different mechanism (gradient alignment), but it is the only LLM paper with explicit ρ/τ for post-training quality.
- None of these:
  - uses closed-form OBS or least-squares repair plus output KL,
  - compares unit types (depth vs width) by the same metric with a correlation analysis, or
  - targets VLMs, ViT blocks or GatedDeltaNet layers.
- So the hypothesis appears unpublished in its specific form (based on the searches done; not exhaustive).
- A paper making this claim should report Spearman/Kendall against post-distillation quality and compare with TraceNAS-style gradient proxies and the raw-KL baseline. It should also control for parameter count, since zero-cost proxies are known to collapse into size counters.

### Gaps
- Exact PRACTISE recoverability definition and any correlation coefficient: full text not retrievable here (arxiv.org blocked). Highest-priority verification.
- TraceNAS: whether ρ=0.94 is against post-training (recovered) accuracy or one-shot accuracy, and whether a depth-vs-width breakdown exists.
- P² Law: fit-error numbers, and whether "loss before pruning" is the parent's loss.
- Puzzle: exact replace-1-block metric (KL vs LM loss).
- Abdelfattah 2021 exact Spearman values: not retrieved.
- No source found for NTK, Fisher or "distillability" predictors applied to pruned pretrained LMs or VLMs.
- No source found for any predictor validated on VLMs/OCR models, hybrid linear-attention decoders, or MoE expert pruning followed by distillation.

## Q4: Error compensation after pruning (OBS/OBC/SparseGPT/least-squares). Is post-compensation residual error used as a predictor of final quality after retraining/distillation, or to compare unit types?

### Takeaway
Compensation is used almost everywhere as a repair step. In OBS-family structured methods (ZipLM, OBC, SlimGPT, LLM Surgeon), the compensated error is also used as the **selection/saliency criterion**. ZipLM is the closest:
- It scores heads, FFN neurons and whole attention/FFN modules by OBS error *after optimal update of the remaining weights*.
- It builds per-layer databases of compensated pruned variants.
- It selects per-layer levels under a latency constraint with an SPDY-style search driven by calibration loss of the stitched model.

However, I found no paper that:
- (a) measures post-compensation damage as KL at the model output, and
- (b) reports its correlation with quality *after* retraining or distillation, or
- (c) uses it to explain why raw damage mis-ranks unit types.

The CNN line (ThiNet, He et al. 2017, NISP, Neuron Merging) also selects by reconstruction error but never validates it as a post-finetune predictor.

### Cited Findings

**OBS-family structured pruning (compensated error as the selection criterion)**
- **ZipLM: Inference-Aware Structured Pruning of Language Models.** Kurtic, Frantar, Alistarh; arXiv 2302.04089, NeurIPS 2023.
  - Extends Optimal Brain Surgeon from single weights to whole structures (sub-components of weight matrices). It updates the remaining weights to compensate for removed structures.
  - Runtime constraints enter via a per-hardware latency table and an adapted SPDY search.
  - Evaluated at 1.5x/2x/2.5x/3x speedup targets (throughput and latency regimes), including GPT-2.
  - Sources: [arXiv](https://arxiv.org/abs/2302.04089); [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/ced46a50befedcb884ccf0cbe8c3ad23-Paper-Conference.pdf); [OpenReview](https://openreview.net/pdf?id=bPFFPueAxm).
- ZipLM details from background knowledge **(unverified this session; check §3 of the paper)**:
  - Structures: attention heads, FFN intermediate neurons, and the option to drop an entire attention or FFN module.
  - Within each layer, the structure to remove next is chosen greedily by the **OBS structured saliency**, i.e. the layer-output reconstruction error after the optimal compensating update. This produces a database of compensated variants per layer at multiple sparsities.
  - Globally, SPDY-style search picks one level per layer under the latency budget. Candidate profiles are evaluated by the stitched model's loss on calibration data.
  - Pipeline: ZipLM is then used gradually with fine-tuning plus distillation (BERT) or one-shot followed by distillation (GPT-2).
- How close ZipLM is to the user's hypothesis:
  - Shared: (i) compensated error decides selection; (ii) heads, neurons and whole modules are all scored by the same compensated criterion, so there is an implicit cross-unit-type comparison; (iii) the global search uses a model-level calibration loss of compensated models.
  - Different: (i) ZipLM's per-structure score is local layer-output MSE, not output KL; (ii) the global objective is a one-shot calibration loss, not a predictor validated against post-distillation quality; (iii) no correlation analysis between compensated error and final distilled accuracy is reported (as far as I know); (iv) no layers-vs-neurons comparison to show raw damage fails while compensated damage succeeds.
- **Optimal Brain Compression (OBC).** Frantar & Alistarh; arXiv 2208.11580, NeurIPS 2022 **(not retrieved; unverified)**.
  - ExactOBS for layer-wise pruning and quantization with exact greedy OBS updates.
  - Supports N:M and block structures.
  - Builds a database of per-layer compressed versions and selects per-layer levels (via SPDY / DP over per-layer error) to meet a global FLOP or latency target, then can fine-tune.
  - Like ZipLM, the compensated error is used for selection, not as a validated predictor of post-retraining quality.
- **SparseGPT.** Frantar & Alistarh; arXiv 2301.00774, ICML 2023 (PMLR v202).
  - Approximate OBS row-wise reconstruction for unstructured or N:M pruning at scale.
  - Defines the layer-wise reconstruction objective and reports per-layer error vs exact reconstruction.
  - No predictor-of-recovery analysis was found.
  - Source: [PMLR](https://proceedings.mlr.press/v202/frantar23a/frantar23a.pdf).
- **Wanda.** Sun et al., arXiv 2306.11695, ICLR 2024 **(unverified this session)**. |W|·‖X‖ saliency without weight update; compensation-free.
- **SlimGPT.** Ling et al., arXiv 2412.18110, NeurIPS 2024 **(not retrieved; unverified)**.
  - OBS-based structured pruning of heads and FFN channels with "Batched Greedy Pruning" and "Incremental Pruning Ratio" (layer-wise ratios).
  - Uses OBS compensation for selection and update.
  - I am not aware of a post-recovery correlation analysis in it.
- **LLM Surgeon.** van der Ouderaa et al., arXiv 2312.17244, ICLR 2024 **(not retrieved; unverified)**.
  - Kronecker-factored curvature (K-FAC) of the global loss for structured (rows/columns) and unstructured removal, with correlated weight updates over multiple shots.
  - The compensated loss increase is the selection criterion; no post-retraining predictor analysis that I know of.
- **OSSCAR.** Meng et al., arXiv 2403.12983, ICML 2024 **(unverified)**. Combinatorial structured pruning of LLMs, minimizing layer-wise reconstruction error with local search.
- Additional methods **(all not retrieved; unverified)**:
  - Boža, "Fast and Optimal Weight Update for Pruned LLMs" (ADMM-based optimal update), arXiv 2401.02938 (ID unverified).
  - FLAP (An et al., AAAI 2024, arXiv 2312.11983): fluctuation-based structured pruning plus bias compensation.
  - SliceGPT (Ashkboos et al., ICLR 2024, arXiv 2401.15024): PCA rotation and deletion.
  - Bonsai (arXiv 2402.05406): forward-only perturbative structured pruning.
  - Thanos, OATS, LayerCollapse: IDs not verified.
  - All use reconstruction or compensation as repair or selection. None was found to report a correlation with post-retraining quality.
- **Free Lunch (arXiv 2510.14444).** Local per-block reconstruction (attention and MLP separately) can beat full retraining. It treats reconstruction as a replacement for recovery, not as a predictor. [arXiv](https://arxiv.org/html/2510.14444v1)
- **OPTIMA (arXiv 2512.13886, preprint).** An unstructured one-shot method. Reports that its improvement in layer-wise error tracks better downstream accuracy, with no fine-tuning. [arXiv PDF](https://www.arxiv.org/pdf/2512.13886)

**Depth pruning with linear or trained replacement (compensation for removed layers)**
- **ReplaceMe.** arXiv 2505.02819.
  - Training-free depth pruning. Removed contiguous transformer blocks are replaced by a single linear transform estimated on a small calibration set (least squares / cosine objective). The transform is merged into the remaining weights, so no parameters are added.
  - Results: about 25% pruning while retaining about 90% performance with no healing.
  - The cut location is chosen by **cosine distance of hidden states before vs after the cut**, which is a raw (uncompensated) criterion, not post-compensation error. The authors note that very high compression needs healing.
  - Sources: [arXiv HTML](https://arxiv.org/html/2505.02819); [GitHub](https://github.com/mts-ai/replaceme).
- **SHIFT-LLM.** arXiv 2608.25068 (2026). A local linear correction at each pruning site, estimated without gradient training; reports higher average zero-shot accuracy than ReplaceMe (L2). [arXiv PDF](https://arxiv.org/pdf/2608.25068)
- **LLM-Streamline** (Chen et al., arXiv 2403.19135, ICLR 2025) and **LinearPatch** (2025) **(not retrieved; unverified)**. LLM-Streamline replaces pruned contiguous layers with a lightweight network trained to mimic them; LinearPatch inserts a closed-form patch (Hadamard plus channel scaling) at the cut.
- **Puzzle BLD (2411.19146)** is a trained local compensation, scored at model output (see Q2).

**CNN-era compensation-aware selection (background, 2017–2020)**
- **ThiNet.** Luo, Wu, Lin; ICCV 2017, arXiv 1707.06342 **(unverified this session)**.
  - Selects channels of layer i greedily by the **reconstruction error of layer i+1's output** using the remaining channels, then rescales by least squares and fine-tunes.
  - Selection is by compensated reconstruction error. No correlation with post-finetune accuracy is reported (to my knowledge).
- **Channel Pruning for Accelerating Very Deep NNs.** He, Zhang, Sun; ICCV 2017, arXiv 1707.06168 **(unverified)**. LASSO channel selection plus least-squares reconstruction.
- **NISP.** Yu et al., CVPR 2018, arXiv 1711.05908 **(unverified)**. Propagates importance from the final-response layer. It is output-aware but compensation-free.
- **Neuron Merging.** Kim et al., NeurIPS 2020, arXiv 2010.13160 **(unverified)**. Compensates pruned neurons by merging them into similar remaining ones, data-free.
- **PRACTISE.** CVPR 2023, 2202.07861 (see Q2). The only paper found that explicitly argues its (repair-aware?) metric correlates with **post-finetune** accuracy better than prior criteria.

**Width vs depth, unit-type comparisons**
- **Sheared LLaMA** (Xia et al., arXiv 2310.06694, ICLR 2024) **(unverified)**. Learned structured masks (layers, heads, intermediate dims, hidden dims) to a target architecture, then continued pretraining with dynamic batch loading.
- **Minitron (2407.14679)** compared depth vs width pruning followed by distillation and found width pruning better at their scales.
- In both, unit types are compared only by post-retraining outcomes, not by a pre-retraining compensated-error predictor.

### Inferences
- In the OBS/OBC/ZipLM/SlimGPT/LLM-Surgeon line, the post-compensation error is used as a **selection criterion** for one-shot quality. It is never presented, as far as found, as a **predictor of post-retraining/distillation quality** with a reported correlation coefficient.
- ZipLM is the nearest neighbour: it applies one compensated criterion across heads, neurons and whole modules, and pipelines into distillation. What it lacks is (a) output-KL measurement, (b) a correlation study vs final distilled quality, and (c) the contrast with raw damage across unit types.
- PRACTISE is the nearest neighbour on the "recoverability predicts finetuned accuracy" claim, but only for CNN blocks with fitted adaptors. Its definition must be checked.
- The user's hypothesis therefore looks novel in combination:
  - closed-form OBS/least-squares compensation,
  - damage measured as output KL to the parent,
  - validated as a cross-unit-type predictor of post-distillation quality,
  - on a hybrid GatedDeltaNet/attention VLM with a ViT.
- Must-cite related work: ZipLM, OBC, PRACTISE, Puzzle, ReplaceMe, TraceNAS, Minitron, P² Law, Free Lunch.
- The "linear ability" finding in PRACTISE ("finetuning with a tiny set mainly recovers the model's linear ability") is a useful theoretical hook: it offers an explanation for why linear-compensated residual should predict recoverability.

### Gaps
- Could not open full texts (arxiv.org blocked). These must be confirmed in their papers:
  - ZipLM's exact selection algorithm (per-structure OBS saliency and how the SPDY objective is evaluated) and whether ZipLM reports any compensated-error vs final-accuracy correlation.
  - SlimGPT, OSSCAR, LLM Surgeon, Boža 2024, FLAP, Bonsai, Thanos, OATS, LayerCollapse, LLM-Streamline, LinearPatch: details and IDs.
- Did not find (and so cannot assert absence with certainty) any paper reporting Spearman/Kendall between post-reconstruction error and post-finetune accuracy. Further targeted queries recommended:
  - "recoverability" plus LLM,
  - "compensation-aware" importance,
  - citing papers of PRACTISE and ZipLM on Semantic Scholar.
- No information found on compensation-based selection for GatedDeltaNet/linear-attention layers, ViT blocks in VLMs, or MoE experts with recovery-correlation analysis.
