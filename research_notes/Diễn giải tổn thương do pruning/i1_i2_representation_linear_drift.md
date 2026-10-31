# I1–I2: What pruned/compressed models lose internally, and whether post-pruning drift is linear

**Reading status.** I read **0 papers in full text**. arxiv.org, aclanthology.org, proceedings.neurips.cc, huggingface.co, ar5iv, web3.arxiv.org, cs.hse.ru and object.cloud.sdsc.edu were all unreachable from this session (ENOTFOUND or proxy 403). The GitHub API was also refused (repos not attached).

I did read in full, via raw.githubusercontent.com, the **official READMEs** of six repositories. Several contain the papers' own result tables:
1. Ghosted Layers (daniel-eai/ghosted_layers_official_repository): Table 2 and Table 3 reproduced, plus the theorem statement.
2. LinearPatch (chenxinrui-tsinghua/LinearPatch): Tables 1–4 reproduced.
3. LLM-Streamline (RUCKBReasoning/LLM-Streamline): results of the released models.
4. tuned-lens (EleutherAI/AlignmentResearch).
5. LLM-Microscope (AIRI-Institute; the code for "Your Transformer is Secretly Linear").
6. An unofficial ShortGPT/Gromov reimplementation (sramshetty/ShortGPT).

Numbers marked **[README]** come from those official tables. Every other item comes from search snippets or abstracts and is marked **NV** (not verified in full text).

Derived "fraction recovered" numbers are **my own arithmetic on README tables**, marked **[computed]**.

---

## I1: What do pruned/compressed models lose internally? (representation similarity, error accumulation, long-tail loss)

### Takeaway
Compression damage is concentrated, not uniform:
- **What is lost.** Top-line metrics stay flat while a small set of hard, rare or atypical examples (PIEs/CIEs), memorized facts and low-resource scripts or languages degrade first and most.
- **Where it is lost.** Error introduced in one layer propagates, and is often amplified, through later layers. Propagated error dominates freshly introduced error, and accumulation can be super-linear.
- **How it is measured.** Representation-similarity metrics (CKA) are a poor damage gauge. A 2026 ICML paper finds deep-layer CKA stays high even when layer-pruned LLMs have collapsed on the task.
- **Depth vs width.** Depth pruning mostly removes low-norm residual updates. Middle layers share a representation space, while the first and last layers are fragile. I found no paper that directly compares per-layer CKA/SVCCA to the parent for depth-pruned vs width-pruned versions of the same model.

### Cited Findings

**Long-tail / rare-feature loss (vision, classic)**
- **Hooker et al., "What Do Compressed Deep Neural Networks Forget?" (arXiv 1911.05248). NV**
  - Models with very different weight counts have comparable top-line metrics, but "diverge considerably in behavior on a narrow subset of the dataset". These are the Pruning Identified Exemplars (PIEs), where pruned and non-pruned predictions disagree. [Google Research](https://research.google/pubs/what-do-compressed-deep-neural-networks-forget/); [arXiv PDF](https://arxiv.org/pdf/1911.05248)
  - Compression "disproportionately impacts model performance on the underrepresented long-tail of the data distribution". PIEs over-index on atypical or noisy images. [arXiv PDF](https://arxiv.org/pdf/1911.05248)
  - **ImageNet numbers (snippets):**
    - At 90% sparsity, 10.27% of test images are PIEs.
    - Restricting evaluation to PIE images drops top-1 from 76.75% to 39.81%.
    - Averaged top-1 is 72.60 at 90% pruning vs 76.68 unpruned, so the full test set loses only about 4 points.
    - [arXiv PDF](https://arxiv.org/pdf/1911.05248)
- **Hooker, Moorosi, Clark, Bengio, Denton, "Characterising Bias in Compressed Models" (arXiv 2010.03058). NV**
  - Setup: ResNet-18 on CelebA (blond vs non-blond), pruned to 0.3–0.99 sparsity, with populations of 30 models.
  - Top-1 stays at about 94.4–94.8% from 0 to 70% pruning, while the number of Compression Identified Exemplars (CIEs) rises from 555 at 30% pruning to 990 at 70%.
  - "Pruning disproportionately impacts performance on underrepresented features" and "compression amplifies existing algorithmic bias".
  - [arXiv PDF](https://arxiv.org/pdf/2010.03058); [Google Research](https://research.google/pubs/characterising-bias-in-compressed-models/)

**Knowledge / fact-recall loss in LLM compression**
- **Jaiswal et al., LLM-KICK, "Compressing LLMs: The Truth is Rarely Pure and Never Simple" (arXiv 2310.01382, ICLR 2024). NV**
  - All pruning methods "suffer significant performance degradation, sometimes at trivial sparsity ratios (e.g., 25-30%)", and fail for N:M sparsity on knowledge-intensive tasks.
  - Quantization preserves knowledge-intensive factoid QA better than pruning.
  - Pruned LLMs at ≥50% sparsity remain "robust in-context retrieval and summarization systems".
  - Compression hurts long-context synthesis more, and compressed models produce less distinct content.
  - Perplexity fails to reveal these changes.
  - [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract-Conference.html); [GitHub](https://github.com/VITA-Group/llm-kick); [Apple ML](https://machinelearning.apple.com/research/compressing-llms)
- **Jin et al., "The Cost of Down-Scaling Language Models: Fact Recall Deteriorates before In-Context Learning" (arXiv 2310.04680; ICLR 2024 as "The Cost of Scaling Down LLMs: Reducing Model Size Affects Memory Before In-Context Learning"). NV**
  - Removing more than 30% of weights causes significant (>5% relative) accuracy loss on fact recall.
  - Even after pruning 60% of weights, the relative drop on ICL tasks stays below 5%.
  - The same dissociation holds for dense scaling, so it is not specific to pruning.
  - [arXiv PDF](https://arxiv.org/pdf/2310.04680); [ML Anthology](https://mlanthology.org/iclr/2024/jin2024iclr-cost/)
- **Gromov et al., "The Unreasonable Ineffectiveness of the Deeper Layers" (arXiv 2403.17887, ICLR 2025). NV**
  - About half of Llama-2-70B's layers can be removed before performance collapses. Collapse points: 45–55% for Llama-2, 35% for Mistral-7B, 25% for Phi-2.
  - MMLU/BoolQ stay flat until a sharp transition, while C4 next-token loss degrades smoothly.
  - The authors suggest "shallow layers may play a critical role in storing knowledge".
  - [arXiv HTML](https://arxiv.org/html/2403.17887v1); [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/cbabc2f70de2dd09f491a8715ec3e80f-Paper-Conference.pdf)

**Low-resource languages / scripts**
- **Marchisio et al., "How Does Quantization Affect Multilingual LLMs?" (arXiv 2407.03211, Findings of EMNLP 2024). NV**
  - Automatic benchmarks show a 1.7% average drop in Japanese, but human evaluation on realistic prompts shows a 16.0% drop. Automatic metrics "severely underestimate" the harm.
  - Non-Latin-script languages are hit hardest. Approximate automatic-metric drops:

    | Model | Latin-script drop | Non-Latin-script drop |
    |---|---|---|
    | 103B | 0.7% | 1.9% |
    | 8B | 3.0% | 3.7% |

  - Mathematical reasoning degrades fastest.
  - [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.935/); [Moonlight review](https://www.themoonlight.io/tw/review/how-does-quantization-affect-multilingual-llms)
- **M-Wanda (arXiv 2505.21171). NV.** Wanda degrades multilingual performance, "especially above 50% sparsity and for underrepresented languages". Language-aware activation statistics reduce perplexity across all languages. [arXiv PDF](https://arxiv.org/pdf/2505.21171)
- **Kurz et al., language-specific calibration for pruning (arXiv 2408.14398, TACL 2026). NV.**
  - Calibrating in the target language minimizes pruning error and best preserves perplexity, but not always downstream performance.
  - Pruning reorders which languages the model handles best.
  - [arXiv HTML](https://arxiv.org/html/2408.14398v1)
- **Rare tokens.** I found no study that measures pruning or quantization damage bucketed by token frequency in LLMs.
  - COMPACT (arXiv 2509.06836, IJCAI-26) deliberately prunes the rarest vocabulary entries on the premise that "rare tokens … contribute little to downstream performance". It also argues that "not all tokens are equally important". NV. [arXiv HTML](https://arxiv.org/html/2509.06836v3)

**Representation similarity (CKA / SVCCA / angular / cosine) after pruning**
- **Kornblith et al., "Similarity of Neural Network Representations Revisited" (arXiv 1905.00414, ICML 2019).** This is the CKA method paper.
  - As width increases, CKA between independently trained networks approaches 1, and early layers saturate faster than later ones.
  - In very deep networks, CKA shows representations of more than half the network are very similar to the last layer, which suggests redundant deep layers.
  - It concerns width-varied *training*, not pruning. NV. [arXiv PDF](https://arxiv.org/pdf/1905.00414)
- **Shi et al., "Understanding Performance Collapse in Layer-Pruned LLMs via Decision Representation Transitions" (arXiv 2605.07271, ICML 2026). NV.**
  - Models: LLaMA3-8B, LLaMA2-7B, Qwen3-4B.
  - Finds a sharp decision transition between a "Silent Phase", where the model cannot yet predict the answer, and a "Decisive Phase", where the answer emerges.
  - Pruning Decisive-Phase layers has minimal impact; pruning Silent-Phase layers "triggers immediate performance collapse".
  - Even at 50% pruning, where performance has collapsed, CKA shows deep-layer representations stay largely preserved. "High representation similarity does not necessarily imply effective decision discrimination."
  - [arXiv PDF](https://arxiv.org/pdf/2605.07271); [ICML 2026](https://icml.cc/virtual/2026/poster/62965)
- **"Effective Layer Pruning Through Similarity Metric Perspective" (arXiv 2405.17081). NV.** Removes the layer whose removal keeps CKA to the original network highest. Cost is |L| forwards plus CKA comparisons. [arXiv PDF](https://arxiv.org/pdf/2405.17081)
- **MPruner (arXiv 2408.13482). NV.** Uses CKA-based layer clustering to decide how many layers can be removed. [arXiv](https://arxiv.org/pdf/2408.13482)
- **Gromov angular distance (2403.17887). NV.** For each n, removes the n-layer block whose input and output have the smallest angular distance, then heals with QLoRA. The simple heuristic of dropping layers from the penultimate inward nearly matches it after healing. [arXiv HTML](https://arxiv.org/html/2403.17887v1)
- **ShortGPT, Block Influence (arXiv 2403.03853). NV.**
  - BI = 1 − cosine(input, output) of a layer; layers are removed in ascending BI order.
  - Removing the last 10 of 40 layers of LLaMA-2-13B drops MMLU from 55.0 to 52.2 (95% retention). A review reports about 91.6% average retention at about 24.6% of layers removed.
  - A Hugging Face community comment (informal) says middle-layer removal hurts multilingual generation, math and reasoning.
  - [arXiv HTML v2](https://arxiv.org/html/2403.03853v2); [Liner review](https://liner.com/review/shortgpt-layers-in-large-language-models-are-more-redundant-than)
  - The unofficial reimplementation reproduces the same 9 removed layers for Llama-2-7B ([27, 26, 25, 28, 24, 29, 23, 21, 22]) in a different order. [README] [GitHub](https://github.com/sramshetty/ShortGPT)
- **Sun et al., "Transformer Layers as Painters" (arXiv 2407.09298, AAAI 2025). NV.**
  - Layer-similarity analysis of Llama2-13B shows roughly three groups: first layers, a large middle block, and the last one or two layers. This suggests middle layers share a representation space.
  - Skipping several middle layers degrades gracefully. Reversing the middle-layer order is worse than randomizing it.
  - Per a blog summary, replacing middle layers with copies of a single layer's weights is worse than skipping them.
  - [arXiv HTML](https://arxiv.org/html/2407.09298v1); [Emergence AI blog](https://www.emergence.ai/blog/exploring-the-functional-roles-of-transformer-layers)
- **Lad, Gurnee, Tegmark (+Lee), "The Remarkable Robustness of LLMs: Stages of Inference?" (arXiv 2406.19384; ICML-W 2024, NeurIPS 2025). NV.**
  - Deleting or swapping adjacent layers without fine-tuning retains 72–95% of the original prediction accuracy, across eight models.
  - Deeper models are more robust.
  - Early and final layers are the most fragile; middle layers are robust.
  - Proposed stages: detokenization, feature engineering, prediction ensembling, residual sharpening.
  - [arXiv HTML](https://arxiv.org/html/2406.19384v3); [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/117558)
- **Bansal, Nakkiran, Barak, "Revisiting Model Stitching" (arXiv 2106.07682, NeurIPS 2021). NV.**
  - Model stitching "reveals aspects of representations that measures such as CKA cannot".
  - Typical SGD minima are "stitching-connected".
  - Representations learned with more data, greater width or longer training can be "plugged in" to weaker models.
  - [NeurIPS](https://papers.nips.cc/paper/2021/hash/01ded4259d101feb739b06c399e9cd9c-Abstract.html)
- **Rethinking Layer Redundancy (arXiv 2604.24938; ICML 2026 as "…Calibration Objectives and Search for Depth Pruning"). NV.**
  - Layer redundancy depends on the calibration objective. Different objectives yield qualitatively different removal patterns.
  - Under a fixed objective, different search algorithms converge to similar solutions.
  - Under perplexity-based pruning, calibration PPL and downstream reasoning rankings show negative or weak correlations. Under a task-likelihood-margin objective they correlate positively.
  - [arXiv PDF](https://arxiv.org/pdf/2604.24938); [ICML 2026](https://icml.cc/virtual/2026/75247)

**Error accumulation / propagation across layers**
- **SlimGPT (arXiv 2412.18110). NV.** Per-layer FFN output error vs the original LLaMA-7B "increases with model depth and accumulates at a rate exceeding linear progression" as the first layer's pruning ratio grows. Fix: a logarithmic, non-uniform layer-ratio schedule. [arXiv HTML](https://arxiv.org/html/2412.18110v1)
- **Týr-the-Pruner (arXiv 2503.09657). NV.** Modeling accumulated error matters. The snippet reports perplexity of about 208.92 with random accumulation vs 58.09 in the ideal case. [arXiv PDF](https://arxiv.org/pdf/2503.09657)
- **Other propagation-aware pruners. NV.**
  - LLM-BIP (arXiv 2412.06419) cites "rapid error accumulation issues across layers" in layer-wise methods. [arXiv HTML](https://arxiv.org/html/2412.06419v1)
  - The convex-optimization layer-wise pruner (arXiv 2408.03728) prunes sequentially within a decoder layer so later operators see already-perturbed inputs. [arXiv PDF](https://arxiv.org/pdf/2408.03728)
  - The on-device structured pruner (arXiv 2501.15255) takes each layer's inputs from the original model to avoid cumulative deviation. [arXiv PDF](https://arxiv.org/pdf/2501.15255)
- **LaCo, "Layer-wise Compensation for Pruned LLMs" (ACL 2026). NV, abstract preview only.** Attributes degradation to "layer-wise accumulation and amplification of local errors". It compensates in hidden states to "intercept the error propagation chain at its source". [ACL preview](https://preview.aclanthology.org/ingest-acl/2026.acl-long.1342/)
- **Quantization**
  - **"Quantization Degradation in LLMs: A Signal-Noise Perspective" (arXiv 2608.08188). NV.**
    - Splits accumulated error into propagated and freshly introduced error. Propagated error dominates at early, middle and late stages.
    - Smaller Qwen3 models amplify error through the middle layers, while larger models preserve or slightly reduce it.
    - [arXiv PDF](https://arxiv.org/pdf/2608.08188)
  - **"Why Do Some Inputs Break Low-Bit LLM Quantization?" (EMNLP 2025). NV.**
    - Errors are strongly correlated across 3–4-bit methods.
    - The full-precision residual-stream magnitude predicts future quantization error.
    - Large-error inputs depend on precise late-layer residual activations.
    - [ACL Anthology](https://aclanthology.org/2025.emnlp-main.168)
  - **QEP (arXiv 2504.09629) and ApiQ (arXiv 2402.05147). NV.**
    - Growing cross-layer error is a key limitation of layer-wise PTQ at low bit-widths.
    - The second layer's error contains cross terms from the first layer's quantization error.
    - [ApiQ PDF](https://arxiv.org/pdf/2402.05147)

### Inferences
- **What is destroyed: rare, hard, memorized information.** Long-tail examples (PIEs/CIEs), facts and non-Latin scripts degrade first, while frequent patterns and in-context "copy from input" skills survive.
  - For an OCR VLM, the analogues are likely rare glyphs, diacritics, non-Latin scripts, unusual layouts and low-quality scans. Bulk Latin text of the kind that dominates CER on standard benchmarks should survive longer.
  - Aggregate CER and perplexity on calibration-like data will under-report this damage, as Marchisio found (1.7% automatic vs 16.0% human).
  - This is extrapolation; I found no OCR-specific study.
- **Possible mechanism for the user's "layers recover ~10x better than MLP neurons".** This is my synthesis, not a published claim.
  - Width (MLP-neuron) cuts remove stored features and knowledge in every layer. Fact recall is the capability most sensitive to parameter count (Jin et al.; LLM-KICK), and the damage compounds super-linearly through depth (SlimGPT).
  - Depth cuts in residual nets remove low-norm residual updates, especially in the shared-space middle region (Painters, Gromov, Lad). The surviving layers can largely absorb these.
  - KD can restore "how to use" the remaining features, but cannot re-create information that is no longer encoded anywhere.
- **Do not use per-layer CKA to the parent as the damage metric.** 2605.07271 shows CKA stays high through collapse, and Bansal shows CKA misses differences that stitching reveals. Better options:
  - KL or CER bucketed by rarity (token or glyph frequency, script);
  - decision-margin metrics;
  - linear stitching or probing of each layer back to the parent.
- **Calibration objective shapes which layers look redundant (2604.24938).** OCR-specific calibration data, including rare scripts, should be used when computing D_lin/D_res. Otherwise the long tail is invisible to the criterion.

### Gaps
- I found no paper that directly reports per-layer CKA, SVCCA or PWCCA to the parent for depth-pruned vs width-pruned versions of the same LLM or VLM. The CKA results found are depth-only, or width-varied training.
- I found no frequency-bucketed (rare-token) analysis of pruning damage in LLMs, and nothing on OCR or rare glyphs.
- Exact tables of LLM-KICK, Jin et al. and Hooker (e.g., PIE vs non-PIE accuracy for pruned models) were not retrievable because the PDF hosts were blocked.
- LaCo (ACL 2026) and 2608.08188 were seen only as abstracts.

---

## I2: Is representation drift after pruning linear? How much does a linear map remove, and by unit type?

### Takeaway
For **layer (depth) removal**, there is now a large literature on linear repair at the cut:
- "Your Transformer is Secretly Linear", ReplaceMe, LinearPatch, Ghosted Layers, SHIFT-LLM, plus the tuned-lens and stitching precedents.
- Consecutive-layer maps are near-linear: Procrustes similarity ≈0.99. Much of that comes from the residual identity, so linearity falls without the residual.
- A training-free linear map at the cut recovers a sizeable but **minority-to-majority** share of the damage. From the LinearPatch tables: about 19–34% of the zero-shot QA gap and about 4–60% of the log-perplexity gap, depending on model and selection criterion. Ghosted Layers' unconstrained least-squares map recovers more: +9 to +19 accuracy points over unrepaired cuts on LLaMA-3-8B-class models.

I found **no paper that measures the fraction of drift, loss or KL removed by an optimal linear map broken down by unit type** (layer vs MLP neuron vs attention head).
- Width-side analogues (FLAP bias compensation, OBS/SparseGPT/SlimGPT-style weight updates, Olica's least-squares FFN calibration, LinDeps for CNN filters) report end accuracy only, not a D_lin/D_res split.
- The user's decomposition by unit type therefore appears novel.

### Cited Findings

**Linearity of layer-to-layer maps**
- **Razzhigaev et al., "Your Transformer is Secretly Linear" (arXiv 2405.12250; ACL 2024 Long, pp. 5376–5384). NV.**
  - Embedding transformations between consecutive decoder layers (GPT, LLaMA, OPT, BLOOM and others) have Procrustes-based linearity ≈0.99. The score is 1 − min_A‖X̃A − Ỹ‖² on normalized, centered embeddings.
  - "Linearity decreases when the residual component is removed due to a consistently low output norm of the transformer layer."
  - Average linearity of the main stream *decreases* during pretraining; fine-tuning *increases* linearity in all tested models.
  - "Removing or linearly approximating some of the most linear blocks … does not affect significantly the loss or model performance."
  - Their pruning removes the most linear layers sequentially, replaces them with linear approximations, and adds layer-wise MSE distillation. A review says the ARC-easy results on OPT-1.3B show performance maintained.
  - A cosine-similarity regularizer in pretraining *reduces* linearity and improves small models on Tiny Stories and SuperGLUE.
  - **Exact pruning table numbers not retrieved.**
  - Sources: [ACL Anthology](https://aclanthology.org/2024.acl-long.293); [arXiv HTML](https://arxiv.org/html/2405.12250v1); [Liner review](https://liner.com/review/your-transformer-is-secretly-linear)
  - The official repo (LLM-Microscope) exposes `procrustes_similarity` (the linearity score) and `procrustes_similarity_centered` ("linearity between X and Y − X", i.e., without the residual). [README] [GitHub](https://github.com/AIRI-Institute/LLM-Microscope)
- **Tuned lens (Belrose et al., arXiv 2303.08112).**
  - Replaces the last m layers with a per-layer *affine translator* trained to minimize KL to the final output distribution.
  - Translators are needed because "representations may be rotated, shifted, or stretched from layer to layer". [README] [GitHub](https://github.com/AlignmentResearch/tuned-lens)
  - The paper reports substantially lower perplexity than the logit lens on Pythia and GPT-NeoX-20B (exact values not retrieved). NV. [arXiv PDF](https://arxiv.org/pdf/2303.08112)
  - Implication: skipping layers with an affine map is lossy, since late layers add non-linearly decodable information. The size of the residual KL per skipped depth was not retrieved.
- **Model stitching (Bansal et al. 2021; Lenc & Vedaldi 2015 is the origin).** Low-capacity stitching layers (often linear or 1×1 conv) align representations of different networks with little loss. Same-architecture networks trained differently can be stitched "with no drop in performance". NV. [NeurIPS](https://papers.nips.cc/paper/2021/hash/01ded4259d101feb739b06c399e9cd9c-Abstract.html)

**Linear repair after layer removal: numbers**
- **ReplaceMe (Shopkhoev et al., arXiv 2505.02819, NeurIPS 2025). NV.**
  - Replaces a contiguous block with one linear transform estimated on a small calibration set, by least squares or a cosine objective, optionally with L1/L2 regularization. The map is merged into existing weights, adding no parameters.
  - "Up to 25% pruning while retaining approximately 90% of the original model's performance", with no healing. It beats other training-free methods and is competitive with retrained ones.
  - Per-table numbers were not retrieved.
  - In Ghosted Layers' Table A7 (garbled extraction, provisional), ReplaceMe averages about 49.00 (cosine variant) at 7/32 on LLaMA-3-8B with WikiText-2 calibration, vs 59.89 for Ghosted.
  - [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1c10d0c087c14689628124bbc8fa69f6-Abstract-Conference.html); [arXiv HTML v4](https://arxiv.org/html/2505.02819v4); [Ghosted PDF](https://arxiv.org/pdf/2605.15491)
- **LinearPatch (Chen et al., arXiv 2505.24680, NeurIPS 2025).**
  - Attributes layer-pruning damage to activation-magnitude mismatch at the boundary and massive outliers at particular tokens. The fix is a Hadamard transform plus channel-wise scaling, fused into one matrix.
  - Headline: on LLaMA-3-8B with 5/32 layers pruned it retains 94.15% (QA); this rises to 95.16% after fine-tuning the patch on 5K samples. NV. [NeurIPS](https://papers.nips.cc/paper_files/paper/2025/hash/e323ebf7c91a4ca061612ba4a2f2164d-Abstract-Conference.html)
  - **[README] tables, 7/32 layers pruned:**

    | Model / criterion | Dense | Pruned | +LinearPatch |
    |---|---|---|---|
    | LLaMA-2-7B, PPL avg (Wiki/C4/PTB) | 11.65 | — | — |
    | — SLEB | | 19.60 | 19.24 |
    | — ShortGPT | | 33.87 | 24.59 |
    | — Taylor+ | | 33.87 | 25.79 |
    | LLaMA-3-8B, PPL avg | 8.54 | — | — |
    | — SLEB | | 16.97 | 15.75 |
    | — ShortGPT | | 58.43 | 28.42 |
    | — LLM-Streamline (no training) | | 2839.30 | 85.10 |
    | — Taylor+ | | 2840.38 | 236.49 |
    | LLaMA-2-7B, 9-task QA avg | 69.98 | — | — |
    | — ShortGPT | | 60.59 | 62.42 |
    | LLaMA-3-8B, 9-task QA avg | 73.44 | — | — |
    | — ShortGPT | | 61.54 | 64.42 |
    | — LLM-Streamline (no training) | | 44.29 | 54.28 |

    [GitHub](https://github.com/chenxinrui-tsinghua/LinearPatch)
  - **[computed] Fraction of the damage removed by the LinearPatch linear map:**

    | Model / criterion | QA gap closed | log-PPL gap closed | raw-PPL gap closed |
    |---|---|---|---|
    | LLaMA-2-7B, ShortGPT | 19.5% | 30.0% | 41.8% |
    | LLaMA-2-7B, Taylor+ | — | 25.5% | — |
    | LLaMA-2-7B, SLEB | — | 3.6% | — |
    | LLaMA-3-8B, ShortGPT | 24.2% | 37.5% | 60.2% |
    | LLaMA-3-8B, LLM-Streamline | 34.3% | 60.4% | — |
    | LLaMA-3-8B, Taylor+ | — | 42.8% | — |
    | LLaMA-3-8B, SLEB | — | 10.9% | — |

    - Selection by SLEB, which already minimizes output damage, leaves little linear-removable drift.
    - Note: LinearPatch is a constrained (symmetric) operator, so these are **lower bounds** on what an optimal linear map would remove.
- **Ghosted Layers (arXiv 2605.15491; repo says NeurIPS 2026).**
  - Unconstrained closed-form least squares at the boundary: M* = X_pre⁺ Δ, where Δ is the removed block's summed residual contribution and W = I + M.
  - Theorem 4.1: LinearPatch's H D Hᵀ is symmetric, but the optimal M* "has a substantial anti-symmetric component (≈47% of its Frobenius norm)". In other words, the optimal correction contains a large rotation-like part. [README]
  - **Table 2 (9-task average, LLM-Streamline selection) [README]:**

    | Model | Layers cut | Pruned | LinearPatch-D | LinearPatch-R | Ghosted |
    |---|---|---|---|---|---|
    | LLaMA-3-8B | 7/32 | 40.70 | 48.65 | 50.55 | 60.10 |
    | LLaMA-3-8B | 11/32 | 44.24 | 51.40 | 51.37 | 53.66 |
    | LLaMA-3.1-8B | 7/32 | 42.60 | 50.46 | 53.27 | 60.01 |
    | LLaMA-3.1-8B | 11/32 | 44.68 | 52.39 | 52.39 | 53.92 |
    | DeepSeek-R1-Distill-8B | 7/32 | 48.01 | 51.68 | 53.59 | 57.80 |
    | DeepSeek-R1-Distill-8B | 11/32 | 46.52 | 50.40 | 50.37 | 52.11 |

  - **Table 3 (PPL avg, LLaMA-3.1-8B, 7 layers) [README]:**

    | Selection | LinearPatch-D | LinearPatch-R | Ghosted |
    |---|---|---|---|
    | LLM-Streamline | 127.04 | 66.06 | 27.81 |
    | ShortGPT | 32.71 | 37.31 | 20.82 |

  - Accuracy saturates at 32 calibration sequences. Ghosted also works for non-contiguous cuts, with one operator per removed layer.
  - The dense 9-task average is not in the README, so I cannot compute the fraction of the gap closed.
  - The linear map's benefit shrinks sharply from 7 to 11 removed layers: +19.4 vs +9.4 points on LLaMA-3-8B. Deeper cuts leave more non-linear residual.
  - [GitHub](https://github.com/daniel-eai/ghosted_layers_official_repository); [arXiv](https://arxiv.org/abs/2605.15491)
- **LLM-Streamline (Chen et al., arXiv 2403.19135, ICLR 2025).**
  - Trains a *non-linear* lightweight replacement (FFN, SwiGLU or a transformer layer). The authors found a transformer layer initialized from the first pruned layer works better than an FFN or SwiGLU.
  - Released models [README]:
    - Llama-3.1-8B, avg 64.0 → Llama-3.1-5.4B, 58.2 (two transformer layers as replacement, trained on 1.3B tokens);
    - Llama-2-7B, 59.7 → Llama-2-4.7B, 54.1 (one layer, 0.06B tokens).
  - "Removing layers without replacement significantly decreases model performance" (NV).
  - [GitHub](https://github.com/RUCKBReasoning/LLM-Streamline); [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4b00a351b41358965613c118e87dc28c-Abstract-Conference.html)
- **"Repair Changes Which Layers to Remove: Exact Optima for Depth Pruning" (arXiv 2610.30552, Oct 2026). NV, snippet from prior session.**
  - About 53,000 pruned models evaluated exhaustively across four families, with and without training-free linear repair.
  - Linear repair turns the optimal layer set from a split layout into a contiguous block.
  - Ignoring repair when choosing layers can miss the repaired optimum by up to 189%.
  - Repair-aware greedy search lowers perplexity by 20–66% vs ReplaceMe on five models up to 27B.
  - [arXiv abs](https://arxiv.org/abs/2610.30552)
- **SHIFT-LLM (arXiv 2608.25068). NV.** A training-free local linear correction at each pruning site, with higher average zero-shot accuracy than ReplaceMe (L2). [arXiv PDF](https://arxiv.org/pdf/2608.25068)
- **Painters (2407.09298). NV.** Replacing several middle layers with repeated copies of one middle layer is worse than simply skipping them. A "shared space" does not mean the layers compute the same function. [Emergence AI blog](https://www.emergence.ai/blog/exploring-the-functional-roles-of-transformer-layers)

**Width / neuron / head side: closest analogues (no D_lin fraction reported)**
- **FLAP.** Bias-only compensation after channel or head removal. A follow-up notes it "does not update the remaining weights". NV. [FASP arXiv PDF](https://arxiv.org/pdf/2501.09412)
- **Olica (ICML 2025, arXiv 2506.08436).**
  - Prunes attention via PCA on the combined attention matrix products.
  - Prunes FFN neurons and adds a "linear calibration" that reconstructs the residual error with low-rank matrices, via closed-form least squares, without retraining.
  - NV. [PMLR](https://proceedings.mlr.press/v267/he25m.html); [arXiv](https://arxiv.org/abs/2506.08436v1)
- **RCPU (OpenReview).** Direct least-squares refitting after structured pruning "tends to overfit to the limited calibration set, destructively modifying pretrained weights". NV. [OpenReview PDF](https://openreview.net/pdf/8677642b188f2df7a910fce9c19486855e996de0.pdf)
- **LinDeps (arXiv 2507.21573, ICPR 2026). CNNs.**
  - Pivoted QR on feature maps finds *linearly dependent* filters. These are removed and exactly compensated in the next layer's kernels, with "guaranteed performance preservation".
  - This is the width-side case where D_res = 0 by construction.
  - NV. [arXiv PDF](https://arxiv.org/pdf/2507.21573)
- **Layer Folding (arXiv 2106.09309).** Depth reduction by linearizing activations so adjacent layers can be folded. NV. [arXiv PDF](https://arxiv.org/pdf/2106.09309)

### Inferences
- **Answer to "is drift linear?" for depth.** Partly.
  - The *total* layer map is ≈0.99 linear mainly because the residual identity dominates and the layer's update has low norm (Razzhigaev). Removing a block therefore leaves a small perturbation, Δ, a meaningful share of which is linearly predictable from X_pre.
  - The optimal predictor is not just a magnitude rescale: about 47% of it is anti-symmetric, i.e., rotation-like (Ghosted).
  - In the README tables, the linear-removable share is typically 20–35% of QA damage and 30–60% of log-PPL damage for a constrained map, and larger for unconstrained least squares. It shrinks as more layers are removed.
- **Mapping onto the user's D_lin / D_res.**
  - For depth, published numbers imply D_lin ≈ 0.2–0.6 of zero-shot damage in 7–8B LLMs. D_res is still a large share, yet layer cuts recover well under KD.
  - This suggests that recoverability is not explained by D_lin alone. A plausible complementary explanation (inference) is that D_res for depth is "re-learnable routing", because the surviving layers still hold the features. D_res for width is "missing features": information no longer linearly *or* non-linearly present at the cut.
- **The width-side D_lin is formally what OBS/SparseGPT/SlimGPT-style compensation of `down_proj` computes.**
  - It is the linear reconstruction of the removed neurons' contribution from the remaining neurons' activations, for the same input.
  - Width cuts thus already get their "free" linear repair in Hessian-based pruners, and the remaining D_res is exactly the non-redundant features.
  - LinDeps is the limiting case: linearly dependent units have D_res = 0. The paper reports no per-unit-type split.
- **Possible novel contribution for the user.** A per-unit-type measurement has no published precedent that I found: measure the fraction of KL, or of CER increase, removed by the optimal linear map at the cut, for layer removal vs MLP neurons vs attention or GatedDeltaNet heads, at matched zero-shot damage. Ideally include a rarity breakdown connecting I1 and I2. A suggested protocol (inference):
  - closed-form Ghosted-style M* for layers;
  - OBS-style down-proj least squares for neurons;
  - o_proj least squares for heads;
  - report D_lin / (D_lin + D_res) in KL and in CER on common vs rare glyphs or scripts.
- **Expect selection to interact with repair (2610.30552).** Rankings of layers by raw damage will differ from rankings by post-linear-repair damage (D_res). D_res is the more relevant quantity if linear repair or KD follows.

### Gaps
- I found **no measurement of the fraction of drift or loss removed by a linear map broken down by unit type** (layer vs neuron vs head). Searches for linear compensation in width and head pruning returned methods (FLAP, Olica, SlimGPT, RCPU, COEC 2608.21142) that report only end accuracy.
- The exact pruning and linear-replacement tables of Razzhigaev (2405.12250), the ReplaceMe tables (LS vs cosine, UIDL baseline, ViT/CLIP results), tuned-lens perplexity values, and Lenc & Vedaldi (2015) details were not retrieved, because arxiv, ACL Anthology and NeurIPS proceedings were blocked.
- The dense 9-task average for Ghosted Layers' task set is missing, so I cannot compute its gap fraction.
- Relative representations (Moschella et al.) were not covered.
- I found no evidence on linear repair for **GatedDeltaNet / linear-attention** layers or for **ViT-block removal in VLMs**.
