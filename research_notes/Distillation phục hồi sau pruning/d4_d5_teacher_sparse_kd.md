# D4/D5: Teacher choice for pruned students & sparse/partial KD signals

Context: LightOnOCR-3-0.8B pruned to ~0.3B (≈63%) / ~0.1B (≈87%). Current KD = stored top-20 logprobs from a 4B teacher, renormalised over the top-20, on ~50% of pages; the rest are CE on teacher labels. Loss = 0.5 CE + 0.5 forward KL.

**Verification status (read first):** arxiv.org, aclanthology.org, ai.meta.com and developer.nvidia.com all failed with ENOTFOUND in this session. That means **no paper below was read in full text**. Every item is **NOT VERIFIED (snippet only)**: it comes from web-search snippets of the primary PDF or abstract page, or from secondary summaries where marked. To confirm the numbers, allow arxiv.org (and ideally aclanthology.org, ai.meta.com, developer.nvidia.com) in the session's network settings.

Teacher/student ratios in the user's setup (arithmetic, not sourced): 4B teacher → 0.3B student ≈ 13×, 4B → 0.1B ≈ 40×; parent 0.8B → 0.3B ≈ 2.7×, 0.8B → 0.1B ≈ 8×.

## D4: Teacher choice for pruned students: parent vs larger same-family teacher vs ensemble/mix

### Takeaway
The evidence says a stronger teacher is not always better. A "capacity gap" (a U-shaped curve of student quality against teacher strength) is documented from CNNs (Cho & Hariharan 2019; Mirzadeh 2020) to LLM pre-training distillation (Zhang et al. ACL 2025; Busbridge et al. ICML 2025). Gemma 3 reports that a smaller teacher is better for **short training horizons** and a larger teacher wins only for long ones. Pruning-recovery budgets are short by design, and the 0.1B student would sit 40× below the 4B teacher. So for the 87% student, the most defensible default is the **parent (0.8B), possibly "teacher-corrected" (fine-tuned on the distillation data, as in Minitron)**, or a parent+4B mix. For the 0.3B student, the 4B teacher is less risky but still not proven better. This must be tested empirically: no source tests a pruned student at exactly these ratios on OCR.

### Cited Findings
**Capacity gap: classic background**
- Cho & Hariharan (ICCV 2019, "On the Efficacy of Knowledge Distillation"): "larger models do not often make better teachers", attributed to capacity mismatch, where small students cannot mimic large teachers. Mitigation: **early-stop the teacher's training**. A sequence of KD steps did not help. NOT VERIFIED (snippet only). — [mlanthology ICCV 2019](https://mlanthology.org/iccv/2019/cho2019iccv-efficacy/)
- Mirzadeh et al. (AAAI 2020, arXiv 1902.03393, "Improved KD via Teacher Assistant"): a student distilled from a larger and more accurate teacher can perform **worse** than the same student distilled from a smaller teacher. Fix: an intermediate-size "teacher assistant". Evaluated on CIFAR-10/100 and ImageNet with CNN/ResNet. NOT VERIFIED (snippet only). — [arXiv 1902.03393](https://arxiv.org/pdf/1902.03393); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/5963/5819)

**Capacity gap in LM distillation**
- Zhang et al., "Towards the Law of Capacity Gap in Distilling Language Models" (arXiv 2311.07052, ACL 2025 long, pp. 22504–22528): there is an optimal teacher along the teacher-scaling curve (the "curse of capacity gap"). The **optimal teacher scale follows an approximately linear scaling with student scale**, consistently across architectures and data scales. Extended to 7B-scale LLM distillation. I did **not** find a specific fitted ratio such as "2.5×" in snippets. NOT VERIFIED (snippet only). — [arXiv 2311.07052](https://arxiv.org/pdf/2311.07052); [ACL Anthology](https://preview.aclanthology.org/setup/2025.acl-long.1097)
- Busbridge et al., "Distillation Scaling Laws" (arXiv 2502.08606, Apple, ICML 2025). NOT VERIFIED (snippet only):
  - The law includes a capacity gap: "making a teacher too capable eventually reduces the student performance". Student cross-entropy is modelled as a function of teacher cross-entropy, via a **broken power law** with two regimes, plus power laws in student parameters and distillation tokens. — [arXiv PDF](https://arxiv.org/pdf/2502.08606)
  - Teacher size affects the student **only through the teacher's cross-entropy**. The gap is interpreted as a difference in relative **learning capacity** of teacher and student on a reference task, not a raw parameter ratio. The paper refines Zhang et al.'s roughly linear optimal-teacher relation into a U-shape. (Busbridge's own X thread plus a survey; secondary.) — [Busbridge X thread](https://x.com/danbusbridge/status/1890013666575282669); [On-policy distillation survey 2604.00626](https://arxiv.org/html/2604.00626v2)
  - Figure 1: teachers with lower CE generally give students with lower CE "until the capacity gap". A student can also outperform its teacher. — [arXiv PDF](https://arxiv.org/pdf/2502.08606)
  - Compute recipe: if a teacher **already exists** (or many students will be distilled), distillation beats supervised training up to a compute level that grows predictably with student size. If a teacher must also be trained for a single student, supervised training is preferred. — [arXiv abs v2](https://arxiv.org/abs/2502.08606v2)
- Li et al., "Small Models Struggle to Learn from Strong Reasoners" (arXiv 2502.12143, ACL Findings 2025, pp. 25366–25394). NOT VERIFIED (snippet only):
  - Small models (≤3B) do not consistently benefit from long CoT or from distillation by larger models. They do better on shorter CoT or with **smaller teachers** (the "learnability gap").
  - Fix: **Mix Distillation**. Mix-Long combines long and short CoT. Mix-Large combines outputs from large and small teachers.
  - Qwen2.5-3B-Instruct: >7 points on MATH/AIME/AMC with Mix-Large vs large-teacher-only data (v1 headline), and >8 points on MATH/AMC with Mix-Long vs long-CoT-only (later version). The numbers differ across versions.
  - The setting is reasoning SFT data, not logit KD. — [arXiv 2502.12143](https://arxiv.org/pdf/2502.12143); [ICML virtual page](https://icml.cc/virtual/2025/52467)
- Gemma 3 Technical Report (arXiv 2503.19786, 2025): trained a student with a small and a large teacher over different training horizons (Fig. 8, "small versus large teacher", relative perplexity difference). "For short training horizons, the smaller teacher is better, but the trend is reversed for longer training." I could not find the crossover token count in snippets. NOT VERIFIED (snippet only). — [Gemma 3 report](https://arxiv.org/pdf/2503.19786)

**What production prune+distill pipelines actually use as teacher**
- Minitron / Llama-3.1-Minitron ("LLM Pruning and Distillation in Practice: The Minitron Approach", arXiv 2408.11796, NVIDIA 2024). NOT VERIFIED (snippet only):
  - Teacher = the **unpruned parent**, first **"teacher-corrected"**: fine-tuned on the distillation dataset because the original pre-training data was unavailable. "Omitting teacher correction causes a data distribution mismatch, negatively impacting distillation."
  - The paper figure labels teacher correction as **127B tokens**. One secondary summary says ~100B.
  - Distillation = forward KL on logits only (LM CE loss ignored).
  - A secondary summary claims teacher correction cut validation loss by >6%. Unconfirmed.
  - Llama-3.1-Minitron-4B performs "favorably" vs its 8B teacher using **94B vs 15T tokens (150× fewer)**, and width pruning beats depth pruning.
  - Sources: [arXiv 2408.11796](https://arxiv.org/pdf/2408.11796); [NVIDIA blog](https://developer.nvidia.com/blog/how-to-prune-and-distill-llama-3-1-8b-to-an-nvidia-llama-3-1-minitron-4b-model); [secondary summary](https://hub.athina.ai/research-papers/llm-pruning-and-distillation-in-practice-the-minitron-approach)
  - Caveat: the evidence shows the pruned 4B is competitive with its 8B parent, **not** that it beats the parent across the board. A later paper (arXiv 2602.01997) argues large-scale distillation did not restore reasoning in the depth-pruned Minitron-4B checkpoint. — [arXiv 2602.01997](https://arxiv.org/pdf/2602.01997)
- Llama 3.2 1B/3B (Meta, Sept 2024): structured pruning from Llama 3.1 8B, then KD using **logits from Llama 3.1 8B and 70B as token-level targets** during pre-training, i.e. a parent plus larger-teacher ensemble. Only secondary descriptions were found. Meta's blog and model card could not be fetched (ai.meta.com blocked). NOT VERIFIED (secondary snippet only). — [Novita summary](https://blogs.novita.ai/llama-3-2-1b/); [Featherless listing quoting the card](https://featherless.ai/models/Rev124/llama-3-pruned)
- Gemma 2 (arXiv 2408.00118): the 2B and 9B models are trained by distilling from the per-token probability distribution of a larger teacher (loss = −Σ P_T log P_S). The teacher-size details were not visible in snippets. NOT VERIFIED (snippet only). — [Gemma 2 report](https://arxiv.org/pdf/2408.00118)

**Pruned student surpassing parent**
- I found no source in this session showing a pruned-and-distilled student strictly beating its unpruned parent across benchmarks. The closest are Busbridge's statement that a student "can outperform its teacher" (general distillation, not pruning) and Minitron-4B being "favorable" vs its 8B teacher on some benchmarks. — [Busbridge PDF](https://arxiv.org/pdf/2502.08606); [Minitron](https://arxiv.org/pdf/2408.11796)

### Inferences
- **0.1B student (≈87% compression):** a 40× teacher ratio sits deep in the regime where capacity-gap papers report degradation. Pruning recovery is a short training horizon, where Gemma 3 found the smaller teacher better. The parent (0.8B, 8×) is likely the safer primary teacher, and a two-hop "teacher assistant" chain (4B → 0.8B already exists → 0.3B → 0.1B) is the Mirzadeh-style option. Concretely: distil the 0.1B from the recovered 0.3B or from the 0.8B parent rather than from the 4B.
- **0.3B student (≈63%):** the ratio is 13× to the 4B vs 2.7× to the parent. Because Busbridge says the gap depends on teacher CE rather than size, and OCR transcription targets are highly deterministic (low entropy), a better 4B may help more here. A **mix** (Llama 3.2-style 8B+70B ensemble; Mix-Large) is the hedge.
- **Coverage asymmetry is a practical argument for the parent:** the parent is cheap enough to produce logits on **100%** of pages, online or cached. That removes the 50% coverage hole. The 4B currently gives KD on only half the data, so "4B teacher vs parent" is currently confounded with "50% vs 100% KD coverage".
- **Teacher correction applies directly:** the parent LightOnOCR-0.8B was trained on its own data mix. Briefly fine-tuning it on the exact recovery corpus (Minitron-style) before using it as teacher addresses the distribution-mismatch issue Minitron reports.
- Suggested ablation (cheap, decisive), each at matched tokens on 0.3B and 0.1B:
  - (a) 4B top-20 on 50% of pages plus CE on the rest (current setup);
  - (b) parent full-distribution KD on 100% of pages;
  - (c) teacher-corrected parent on 100%;
  - (d) parent on 100% plus 4B KL where available (weighted mix).

### Gaps
- None of the papers was read in full text (network block), so the exact fitted coefficients are missing: Busbridge's capacity-gap location and Zhang's linear ratio. The Gemma 3 Fig. 8 crossover token count and Minitron's teacher-correction loss numbers are also missing.
- No paper found tests capacity gap specifically for **pruned** students (initialised from the parent) rather than randomly initialised students. Inheriting weights from the parent plausibly makes the parent's distribution easier to match, but this is unverified.
- Meta's official Llama 3.2 wording (teacher mix ratio, which stage used 8B vs 70B logits, whether logits were top-k) could not be verified.
- No source found on capacity gap in VLM or OCR distillation specifically.
- The ensemble vs single teacher question for logit KD has no quantitative source beyond Llama 3.2 (no ablation reported) and Mix-Large (SFT data, not logits).

## D5: Sparse/partial KD: top-k stored logits, renormalisation vs residual mass, sampled logits, partial coverage

### Takeaway
Top-k caching with renormalisation is a **biased** estimator: it pushes non-top-k probabilities to zero and hurts calibration (Sparse Logit Sampling). Renormalisation alone does not remove the bias. Unbiased alternatives are importance-sampled logits (RS-KD; Gemma 3 samples **256 logits per token** weighted by teacher probability). The cheapest fix compatible with already stored top-20 is to keep the **residual mass as a tail/"other" bucket** (one extra scalar per position) and add a binary/tail KL term. Two 2026 papers (TAD, ICML 2026; a reward-gated OPD paper) report that dropping or neglecting the tail term hurts. For partial coverage, the closest evidence ("Rethinking Selective KD", 2026) suggests KD on a well-chosen subset can match full KD. No paper prescribes a CE/KD split for 50% coverage.

### Cited Findings
**Bias of top-k and the sampling fix**
- Sparse Logit Sampling (Anshumann et al., Samsung; arXiv 2503.16870, ACL 2025 long, pp. 18085–18108). NOT VERIFIED (snippet only):
  - Naive top-K caching gives **biased estimates** of the teacher distribution, which hurts performance and calibration.
  - Under KL with top-K targets, non-top-K tokens are pushed to probability 0. The bias appears **even without normalising** the top-K probabilities (§2.2.1), so renormalisation is not the fix.
  - The authors argue that "ghost token" (residual-mass) methods are either biased or give too little tail supervision.
  - Fix: **Random-Sampling KD**, which importance-samples tokens from the teacher, is unbiased, preserves the expected gradient, and stores much sparser logits.
  - Reported cost <10% overhead vs CE training, with performance competitive with full KD, at 300M–3B. I could not confirm the specific sample counts or tables from snippets.
  - The user already knows this paper; listed for completeness of numbers. — [arXiv 2503.16870](https://arxiv.org/pdf/2503.16870); [ACL](https://preview.aclanthology.org/setup/2025.acl-long.885)
- Gemma 3 (arXiv 2503.19786): "We sample **256 logits per token**, weighted by teacher probabilities. The teacher's target distribution is set to zero probability for non-sampled logits, and renormalized." The student is trained with CE against this sparse target. This is Google's production offline-sparse practice. NOT VERIFIED (snippet only). — [Gemma 3 report](https://arxiv.org/pdf/2503.19786); [secondary deep-dive](https://vizuara.substack.com/p/technical-deep-dive-into-gemma3-next)

**Tail / residual-mass treatments**
- "Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient LM Distillation" (TAD; Dasgupta, Cohn, Baldwin; arXiv 2602.20816, Feb 2026; ICML 2026 / PMLR v306 under the title "…Decoupled Distillation Produces Top Maths Students on an Academic Budget"). NOT VERIFIED (snippet only):
  - KL is decomposed into a top-K term plus a tail term, with the tail renormalised over the non-top-K tokens.
  - For a peaked teacher the top-K mass is high, so the tail contribution to the total KL is tiny. They up-weight the tail with a hyperparameter **β**.
  - Same compute profile as KL. Competitive in pre-training and supervised distillation, at "academic budget" scale.
  - Benchmark numbers were not visible.
  - **Note:** TAD needs tail probabilities (full teacher logits), not just stored top-k. — [arXiv html](https://arxiv.org/html/2602.20816v1); [ICML 2026 poster](https://icml.cc/virtual/2026/poster/62467); [PMLR](https://proceedings.mlr.press/v306/dasgupta26a.html)
- "Reward-Gated On-Policy Distillation" (arXiv 2607.04037, 2026): uses top-K reverse KL at the student's top-K positions and folds all remaining vocabulary into **one aggregate token**, with a binary KL on the residual masses. **Removing the residual bucket hurts performance**, even though its share of the KL is small. NOT VERIFIED (snippet only). — [arXiv 2607.04037](https://arxiv.org/pdf/2607.04037)
- Sparse-distillation probability-domain framework (arXiv 2601.03195, Jan 2026): treats top-k truncation as partial access and lists three ways to handle the missing mass:
  - tail modelling (a parametric form, e.g. Zipf);
  - renormalisation over visible tokens;
  - max-entropy completion ("entropy matching").

  NOT VERIFIED (snippet only). — [arXiv 2601.03195](https://arxiv.org/html/2601.03195v1)

**How large k needs to be; offline top-k in practice**
- "Efficient KD for LLMs: Offline Top-K Logits and a Fused Chunked KL Loss" (arXiv 2608.03796, Aug 2026; Multiverse Computing): caching the teacher's **top-100** probabilities matches online KD at "near-identical training loss". This drops the teacher from memory and runs ~**29% faster per step**. The renormalisation or residual choice and the k-ablation were not visible. NOT VERIFIED (snippet + secondary summaries). — [arXiv 2608.03796](https://arxiv.org/pdf/2608.03796); [opentrain summary](https://www.opentrain.ai/papers/efficient-knowledge-distillation-for-llms-offline-top-k-logits-and-a-fused-chunk--arxiv-2608.03796/); [Modelwire](https://themodelwire.com/article/offline-distillation-cuts-llm-compression-training-time-by-29-percent-01KZ7ZS632M21DE5MJK1M46GKE)
- TopKD (arXiv 2508.04539) is another top-k-focused KD method that appeared in search. Details not retrieved. — [arXiv 2508.04539](https://arxiv.org/html/2508.04539v1)

**Partial coverage / selective KD**
- "Rethinking Selective Knowledge Distillation" (arXiv 2602.01395, Feb 2026). NOT VERIFIED (snippet + search summary):
  - Studies KD on subsets of positions, vocabulary classes, or samples.
  - Student-entropy-based **sample selection cut wall time by 70%** on an 80M-token run.
  - **Class-sampled offline caching cut storage by 99.96%** while maintaining performance.
  - Selecting the **top 20% of positions by student entropy** scored slightly above full KD on average accuracy. — [arXiv 2602.01395](https://arxiv.org/pdf/2602.01395)
- MiniPLM (Gu et al., arXiv 2410.17215, ICLR 2025): **offline** teacher inference ("Difference Sampling") **reshapes the pre-training corpus** rather than storing logits.
  - Teacher Qwen1.5-1.8B; students 200M, 500M and 1.2B.
  - Beats no-KD pre-training and Vanilla KD, MiniLLM and SeqKD on 9 zero-shot tasks.
  - **2.4× lower data demand** (v1).
  - Under matched training FLOPs, other KD methods were similar to or worse than no-KD pre-training.

  Relevant as a "no stored logits" alternative for the uncovered 50%. NOT VERIFIED (snippet only). — [arXiv 2410.17215](https://arxiv.org/html/2410.17215v3); [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ea05e4fc0299c27648c9985266abad47-Abstract-Conference.html)
- Minitron (2408.11796) uses logit-only forward KL and drops CE (already known to user). Gemma 2/3 use teacher-distribution CE. No source found that ablates mixing KD-covered and CE-only examples in one run. — [Minitron](https://arxiv.org/pdf/2408.11796)

### Inferences
- **Measure before fixing.** The stored values are logprobs, so the **un-renormalised top-20 mass** (Σ exp(logprob) over the top-20) is computable per position. If, as is typical for OCR transcription, the mass is ≥0.99 on most tokens, the renormalisation bias is negligible (TAD's own argument: the tail matters little for peaked teachers) and k=20 is enough. The bias then concentrates on ambiguous tokens: rare glyphs, numbers, layout/markup tokens. Those are worth checking specifically.
- **Cheapest upgrade with no re-run of the teacher:** stop renormalising. Keep r = 1 − Σ top-20 p_T as an "other" bucket. Compute the student's other-mass as 1 − Σ_{top-20 of T} p_S. Use KL over 21 bins (top-20 + bucket). This is the residual-bucket formulation of 2607.04037, applied here to forward KL. Unlike renormalised top-20, it also penalises the student for putting mass outside the teacher's top-20.
- **If the teacher can be re-run** (the 4B or the parent), switch to sampled storage: Gemma 3-style 256 samples weighted by p_T, or RS-KD. These are unbiased in expectation. Storage at k=20 vs 256 differs by about 13×. A hybrid (top-~8 exact plus a few sampled tail tokens plus the residual scalar) is a reasonable middle ground, but **no source found evaluates that exact hybrid**.
- **For the uncovered 50%:**
  - (i) Fill it with parent logits, which is cheap and links to D4.
  - (ii) Keep CE on teacher labels, but consider up-weighting KL on covered pages so the per-token KD signal is not halved in expectation. Current effective weights are 0.5 CE + 0.25 KL averaged over the data.
  - (iii) Selective-KD results suggest that a high-entropy subset carries most of the KD value. If budget limits coverage, prioritise teacher logits on pages or tokens where the student is uncertain, rather than a random 50%.
- With a 0.5 CE weight on teacher labels (hard pseudo-labels from the same 4B teacher), the CE term duplicates the KL argmax on covered pages. On those pages, a higher KL weight (Minitron-style logit-only) may be preferable. This is a hypothesis to test; no source here directly tests it for OCR.

### Gaps
- None of the full texts was read. Missing items:
  - RS-KD tables (sample counts, ECE, speedup);
  - the TAD β values and gains;
  - the 2608.03796 k-ablation (whether k=20 vs 100 matters);
  - the 2607.04037 ablation magnitude.
- No source found that directly compares **renormalised top-k vs residual-bucket vs sampled** on the same model with numbers that I could see.
- No source found on the effect of KD **coverage fraction** (e.g. 50% vs 100% of examples) or on how to weight CE-only vs KD examples in the same batch.
- Llama 3.2's offline logit storage format (top-k? how many? sampled?) was not found. Gemma 3's "256 sampled" is the only production format confirmed in snippets.
- No sparse-KD evidence specific to VLM/OCR decoders, where target entropy is much lower than in general LM pre-training.
