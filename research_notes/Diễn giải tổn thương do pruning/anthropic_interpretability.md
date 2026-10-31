# Anthropic interpretability research and what it implies for structured-pruning damage (LightOnOCR-3-0.8B → 0.3B / 0.1B)

**Read-status statement (read this first)**
- **Full-text primary papers on transformer-circuits.pub read: 0.** The host `transformer-circuits.pub` does not resolve in this session (`getaddrinfo ENOTFOUND`; curl through the proxy also got `CONNECT 403`). `lesswrong.com`, `greaterwrong.com` and a third-party PDF mirror also failed with ENOTFOUND, and arxiv/alphaxiv/openreview were off-limits by instruction. To read the papers themselves, `transformer-circuits.pub` has to be allowed in the session's network settings.
- **Pages read in full (12), all short:** these are Anthropic's own anthropic.com summary or blog pages plus one GitHub README. None is the full paper.
  1. "Toy Models of Superposition" summary page.
  2. "Superposition, memorization, and double descent" summary page.
  3. "Privileged bases in the transformer residual stream" abstract page.
  4. "Softmax Linear Units" abstract page.
  5. "Decomposing language models into understandable components" (the blog post for Towards Monosemanticity).
  6. "Mapping the mind of a large language model" (the blog post for Scaling Monosemanticity).
  7. "Tracing the thoughts of a large language model" (the blog post for Circuit Tracing and Biology).
  8. "Open-sourcing circuit-tracing tools".
  9. "A 'diff' tool for AI" (the Dedicated Feature Crosscoder post).
  10. The anthropic.com stub pages for "Crosscoder model diffing" (2025) and for Circuits Updates (May 2023; April, June, July, August and Sept 2024). These are link-outs only, with no content.
  11. "Persona vectors" summary (partial).
  12. GitHub `safety-research/circuit-tracer` README.
  - The anthropic.com pages for "A Mathematical Framework" and "In-context Learning and Induction Heads" contain only a title and a link.
- **Label conventions:**
  - **[V-blog]**: verified on the anthropic.com or GitHub page I read.
  - **NV**: not verified in full text. These come from search-engine snippets or summaries of the paper, or from third-party write-ups. Treat NV numbers as needing a check against the paper.
  - **"INFERENCE (pruning)"**: my own reasoning. It is not an Anthropic claim.

---

## Q1. Residual stream linearity: why could cutting a layer mostly cause linear misalignment? (A Mathematical Framework for Transformer Circuits, 2021; In-context Learning and Induction Heads, 2022)

### Takeaway
In Anthropic's framework, the residual stream is a linear, additive communication channel. Every layer reads from it through a linear projection and writes back by addition. Deleting a block therefore removes one additive term. Downstream layers see a shifted input, and that shift can be partly predicted and corrected linearly. The exception is when the removed block took part in nonlinear compositions, such as attention-pattern (Q/K) composition in induction circuits. Those compositions cannot be recovered by a linear map.

### Cited Findings
- **Framework (Elhage et al. 2021), NV:**
  - The residual stream is the sum of the embedding and all earlier layer outputs.
  - Attention and MLP layers read from it through a linear projection and write back by adding a projection.
  - Commentators describe it as "a communication channel between all the layers" that does little processing itself.
  - The paper notes that a completely linear residual stream is unusual among architectures, and that this is what makes the path-by-path ("path expansion" / "virtual weights") decomposition tractable.
  - Source: [Transformer Circuits, framework](https://transformer-circuits.pub/2021/framework/index.html), via search snippets and the [Oxen.ai Arxiv Dives summary](https://ghost.oxen.ai/arxiv-dives-a-mathematical-framework-for-transformer-circuits/).
- **Framework, NV:** "Virtual weights" means that each layer reads with an arbitrary linear transform and writes with another. Composing these transforms (W_read of layer j times W_write of layer i) gives the effective connection between non-adjacent layers. Secondary sources also stress that residual bandwidth is in high demand. [Oxen.ai summary](https://ghost.oxen.ai/arxiv-dives-a-mathematical-framework-for-transformer-circuits/)
- **Induction heads (Olsson et al. 2022), NV:**
  - Induction heads form at the same point in training as "a sudden sharp increase in in-context learning ability, visible as a bump in the training loss".
  - The paper gives six complementary lines of evidence that induction heads may be the mechanistic source of general in-context learning.
  - One summary says this rests on 34 transformers and more than 50,000 head ablations.
  - The window is the only place in training where the loss is not convex.
  - Sources: [Transformer Circuits, induction heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html); [emergentmind summary](https://www.emergentmind.com/papers/2209.11895).
- **Induction heads, background known from the paper but NV in this session:** An induction circuit is a two-layer composition. A previous-token head in an earlier layer writes into the residual stream, and a later head uses that output in its key (K-composition). The softmax attention pattern of the second head therefore depends nonlinearly on what the first head wrote.

### Inferences
- **INFERENCE (pruning), why layer removal looks like D_lin:**
  - Removing block l deletes one additive term h_l = f_l(x_l) from x_{l+1} = x_l + h_l.
  - When h_l is largely predictable from x_l, the student's residual at the cut equals the parent's minus a linear function of the available state. A linear map at the cut then recovers most of it.
  - h_l is predictable from x_l when, for example, the block mostly rescales or rotates directions that already exist, or when adjacent layers write redundant, linearly correlated content.
  - The crosscoder finding that cross-layer structure is "redundant (linearly correlated)" (Q5) is direct Anthropic evidence for this redundancy.
- **INFERENCE (pruning), when layer removal is not D_lin:**
  - If the removed block carries one half of a composed circuit, the downstream damage passes through a softmax over attention patterns. Examples are a previous-token head feeding an induction or copy head, or the full-attention layers of the hybrid decoder that probably do long-range copying of recognised text.
  - A linear correction at the residual stream cannot rebuild a changed attention pattern unless it restores exactly the key or query input.
  - **Prediction:** D_res from layer removal should concentrate in the full-attention layers and in layers that host the K-composition sources of copy/induction heads. It should be smallest for GatedDeltaNet or MLP-dominated blocks whose outputs are well predicted linearly.
  - **Diagnostic:** after cutting a layer and applying the optimal linear map, compare the attention patterns of later heads, especially copy and induction heads on repeated tokens, between parent and student.
- **INFERENCE (pruning):** The Framework's exact linear path expansion holds only for attention-only models; MLPs break it. That is consistent with MLP damage being the less linear kind.

### Gaps
- I could not read the Framework's own statements about bandwidth, subspace "memory management" or path expansion, or the induction-heads ablation numbers and token counts. The 2.5–5B-token figure in the question was not found in any source.
- Anthropic has no work on GatedDeltaNet or linear-attention layers. Whether their state-update recurrence breaks the "additive residual write" picture is not covered.

---

## Q2. Privileged bases: is the neuron basis privileged in the MLP but not in the residual stream? (SoLU, 2022; Privileged Bases in the Transformer Residual Stream, 2023)

### Takeaway
Anthropic's theory says the MLP hidden layer has a privileged (neuron-aligned) basis because of its elementwise nonlinearity, while residual-stream coordinates should be arbitrary. In practice the residual stream does develop a privileged basis, with outlier dimensions. Anthropic provisionally blames Adam's per-dimension normalisers. Even in the privileged MLP basis, SoLU showed that making neurons more interpretable may just "hide" other features. Superposition persists whatever the basis.

### Cited Findings
- **Privileged Bases abstract (2023) [V-blog]:**
  - "Our mathematical theories of the Transformer architecture suggest that individual coordinates in the residual stream should have no special significance… Recent work has shown that this observation is false in practice."
  - "we… provisionally conclude that the per-dimension normalizers in the Adam optimizer are to blame for the effect."
  - "Layer normalization, and finite-precision floating-point calculations. We confidently rule these out as being the source of the observed basis-alignment."
  - Source: [Anthropic, privileged bases](https://www.anthropic.com/research/privileged-bases-in-the-transformer-residual-stream).
- **Privileged Bases, NV:**
  - Some residual coordinates have outlier values "up to 20× larger than any other coordinate".
  - Excess kurtosis is used as a statistical test for a privileged basis, with a fixed random rotation as the control.
  - The paper also trained a model with rotations applied at every read from and write to the residual stream; its result is not shown in the snippets.
  - Source: [Transformer Circuits, privileged basis](https://transformer-circuits.pub/2023/privileged-basis/index.html), via search snippets.
  - **Conflict:** one search summary said the paper "did not find a definitive cause". That contradicts the abstract's "provisionally conclude… Adam". The abstract is primary.
- **Community follow-up, NV:** In a 12M-parameter model, Adam-trained runs reached excess kurtosis above 100, while SGD-trained runs stayed near 0. [LessWrong post, "Adam Optimizer Causes Privileged Basis…"](https://www.lesswrong.com/posts/yrhu6MeFddnGRSLtQ/adam-optimizer-causes-privileged-basis-in-transformer-lm)
- **ACL 2025 paper, NV:** A Muon-type optimizer without element-wise scaling removes privileged bases and outliers; the authors call outliers "consequences of training strategies". [Outlier-Safe Pre-Training, ACL 2025](https://aclanthology.org/2025.acl-long.618.pdf)
- **SoLU (2022) [V-blog]:**
  - Replacing the MLP activation with a softmax linear unit "significantly increases the fraction of neurons in the MLP layers which seem to correspond to readily human-understandable concepts… as measured by randomized and blinded experiments", "at little to no cost to ML performance".
  - However, "we also discover some evidence that the superposition hypothesis is true and there is no free lunch: SoLU may be making some features more interpretable by 'hiding' others and thus making them even more deeply uninterpretable."
  - Source: [Anthropic, SoLU](https://www.anthropic.com/research/softmax-linear-units).
  - The exact fraction of interpretable neurons, SoLU versus baseline, was not found (NV).

### Inferences
- **INFERENCE (pruning), MLP neuron pruning acts in a privileged but wrong basis:**
  - The elementwise nonlinearity makes MLP neurons the only basis in which you can drop units without changing the function of the units you keep. That is why structured MLP pruning is done per neuron.
  - But features in that layer are not aligned to neurons (SoLU's "no free lunch"; Towards Monosemanticity, Q4).
  - So every neuron cut removes part of several features. The nonlinearity sits before the cut point, so no linear map applied to the MLP output can re-synthesise the lost nonlinear response. This is a mechanistic reason for MLP damage being mostly D_res.
- **INFERENCE (pruning), residual-stream (hidden-width) pruning:**
  - Apart from the Adam-induced outliers, residual coordinates are not privileged. Removing hidden dimensions coordinate by coordinate is therefore arbitrary.
  - Rotate first, then project onto the top principal subspace (in the spirit of SliceGPT, which is not Anthropic work). Within a residual-stream framework this kind of truncation is largely linear and correctable.
  - Never remove the outlier coordinates; they carry disproportionate norm.
- **INFERENCE (pruning), check before trusting rotations:** LightOnOCR was presumably Adam-trained, so measure per-coordinate kurtosis of the residual stream (the Anthropic test). This shows how far the residual basis is privileged before relying on rotation-based width pruning.

### Gaps
- I could not access the rotation-trained model results, the kurtosis numbers in the Anthropic paper, or SoLU's interpretability percentages.
- I found nothing Anthropic-specific on whether GatedDeltaNet state channels have a privileged basis.

---

## Q3. Superposition: why is removing neurons non-linear and many-to-many? (Toy Models of Superposition, 2022; Superposition, Memorization and Double Descent, 2023)

### Takeaway
When features are sparse, networks store more features than they have dimensions. The cost is interference, which the nonlinearity has to filter out. This creates two linked fragilities:
- Deleting a dimension or neuron removes a fraction of many features at once.
- It also upsets the learned balance that lets the nonlinearity suppress interference.
So the damage is non-linear and spread out.

### Cited Findings
- **Toy Models (Elhage et al. 2022) [V-blog]:** "small ReLU networks trained on synthetic data with sparse input features — to investigate how and when models represent more features than they have dimensions. We call this phenomenon superposition. When features are sparse, superposition allows compression beyond what a linear model would do, at the cost of 'interference' that requires nonlinear filtering." [Anthropic, toy models](https://www.anthropic.com/research/toy-models-of-superposition)
- **Toy Models, NV:**
  - Whether features are stored in superposition is governed by a phase change, which is first-order: the loss curves of the two geometries cross.
  - With importance I=1 and density 1−S=0.05, features form a regular pentagon; at higher sparsity it collapses to digons.
  - The geometries seen include antipodal pairs, triangles, pentagons, tetrahedra and square antiprisms.
  - Secondary sources say "dimensions per feature" clusters at 1/2 (antipodal pairs) and 2/5 (pentagons).
  - Sources: [Transformer Circuits, toy model](https://transformer-circuits.pub/2022/toy_model/index.html); [Tyler Crosse notes](https://www.tylercrosse.com/ideas/2026/circuits-superposition); [review, arXiv 2609.06862](https://arxiv.org/pdf/2609.06862).
- **Double descent (2023) [V-blog]:**
  - "Overfitting corresponds to storing data points, rather than features, in superposition."
  - "Depending on dataset size, our models fall into two different regimes: an overfitting regime (characterized by storing data points in superposition), and a generalizing regime (characterized by storing features in superposition)."
  - "We observe double descent as the model transitions between these regimes."
  - Source: [Anthropic, double descent](https://www.anthropic.com/research/superposition-memorization-and-double-descent).
- **Caveat from an independent review, NV:**
  - A rotation of mutually orthogonal feature directions can make neurons polysemantic while the directions stay orthogonal.
  - Polysemantic units also appear in models with spare capacity.
  - So neuron polysemanticity does not by itself prove superposition. [arXiv 2609.06862](https://arxiv.org/pdf/2609.06862)

### Inferences
- **INFERENCE (pruning), core mechanism for the ~10× recovery gap:**
  - In superposition, feature i is read as ReLU(w_i·x + b_i). The bias and the near-orthogonality of the w's are tuned so that interference from other active features stays under threshold.
  - Removing neuron or dimension k removes the k-th component from every w_i that uses it. That shrinks the signal for many features and raises their relative interference.
  - Features whose margin falls below the threshold drop out, or false activations rise. That is a thresholded, all-or-nothing loss per feature: non-linear damage, the D_res kind.
  - A linear map after the nonlinearity can rescale the surviving signal. It cannot recover features that the threshold has already zeroed, or separate interference that has been mixed in.
- **INFERENCE (pruning):**
  - The "importance × sparsity" trade-off predicts which features get dropped first. A small model under capacity pressure gives up the least important or least frequent features (Toy Models: low-importance features are not represented at all when capacity is short).
  - Expect a distilled 0.1B student to have abandoned rare-glyph, rare-script and rare-layout features first. Check this per class on OCR benchmarks (rare characters, mathematical notation, tables).
- **INFERENCE (pruning), distillation data size:**
  - The double-descent note suggests that with a small distillation set, a heavily pruned student may store data points (memorise) rather than features.
  - The number of distillation tokens and images relative to the student's capacity then matters. A double-descent-shaped curve of test loss against dataset size would be a warning sign.
- **INFERENCE (pruning), caveat to the user's hypothesis:** Polysemanticity is not proof of superposition (see the review above). To show superposition specifically, measure the number of SAE features per neuron at good reconstruction (Q4): many more features than neurons means superposition.

### Gaps
- I could not access the exact Toy Models figures: the phase diagrams and the interference-versus-sparsity plots.
- I found no Anthropic experiment that removes neurons or dimensions from a superposed toy model and measures feature loss. That would be a cheap, direct test the user could run.

---

## Q4. Features versus neurons: how many features, how distributed, how much is left unexplained? (Towards Monosemanticity, 2023; Scaling Monosemanticity, 2024)

### Takeaway
Dictionary learning (sparse autoencoders) breaks a 512-neuron MLP layer into more than 4,000 interpretable features, each a linear combination of neurons. At frontier scale, Claude 3 Sonnet yields millions of features that are multimodal and multilingual. Reconstructions leave a sizeable residual ("dark matter"). This is direct empirical support for "a neuron = a slice of many features", with a quantitative handle (features per neuron, variance explained) the user can measure on LightOnOCR.

### Cited Findings
- **Towards Monosemanticity (Bricken et al. 2023) [V-blog]:**
  - "a single neuron in a small language model is active in many unrelated contexts, including: academic citations, English dialogue, HTTP requests, and Korean text."
  - "These units, called features, correspond to patterns (linear combinations) of neuron activations."
  - "we decompose a layer with 512 neurons into more than 4000 features which separately represent things like DNA sequences, legal language, HTTP requests, Hebrew text, nutrition statements… Most of these model properties are invisible when looking at the activations of individual neurons in isolation."
  - Source: [Anthropic, decomposing language models](https://www.anthropic.com/research/decomposing-language-models-into-understandable-components).
- **Towards Monosemanticity [V-blog]:**
  - Blinded human raters and autointerpretability both score features above neurons.
  - "the features that are learned are largely universal between different models".
  - The number of features works as "a 'knob' for varying the resolution… a small set of features offers a coarse view… a large set… a more refined view". This is feature splitting.
  - Source: [Anthropic, decomposing language models](https://www.anthropic.com/research/decomposing-language-models-into-understandable-components).
- **Towards Monosemanticity, NV:**
  - The model is a one-layer transformer with a 512-neuron MLP.
  - The SAE was trained on MLP activations from 8B data points.
  - One search snippet gives dictionary sizes "from 512 to about 13,100". This conflicts with my recollection that runs go up to about 131k (A/5). The figure is unverified and possibly garbled.
  - A third-party blog says a roughly 16× (~15k) configuration was chosen.
  - Sources: [Transformer Circuits, monosemantic features](https://transformer-circuits.pub/2023/monosemantic-features/index.html), via search; [Galileo blog](https://galileo.ai/blog/anthropic-ai-interpretability-breakthrough).
- **Scaling Monosemanticity (Templeton et al. 2024) [V-blog]:**
  - "each concept is represented across many neurons, and each neuron is involved in representing many concepts."
  - "We successfully extracted millions of features from the middle layer of Claude 3.0 Sonnet."
  - "These features are multimodal and multilingual, responding to images of a given entity as well as its name or description in many languages." For example, the Golden Gate Bridge feature fires on text in several languages and "an image".
  - Feature distance was measured "based on which neurons appeared in their activation patterns".
  - Steering (clamping features) causally changes behaviour.
  - "finding a full set of features using our current techniques would be cost-prohibitive (the computation required… would vastly exceed the compute used to train the model in the first place)."
  - Source: [Anthropic, mapping the mind](https://www.anthropic.com/research/mapping-mind-language-model).
- **Scaling Monosemanticity, NV:**
  - SAEs of 1M, 4M and 34M features.
  - Reconstructions explain "at least 65% of the variance".
  - On average fewer than 300 features are active per token for all three SAEs.
  - Dead features (never active over 10^7 tokens) are about 2% (1M), 35% (4M) and 65% (34M).
  - Source: [Transformer Circuits, scaling monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/), via search summary.

### Inferences
- **INFERENCE (pruning), quantifying superposition in the user's own MLPs:**
  - Train a TopK or BatchTopK SAE (or a transcoder) on a few LightOnOCR decoder MLP layers. Measure how many features are needed, at fixed fraction-of-variance-explained or fixed spliced-in CE loss, relative to the neuron count, and how many neurons each feature's encoder/decoder loads on (decoder-weight participation ratio across neurons).
  - High features-per-neuron combined with high participation ratio means neuron-aligned removal must cut many features: the user's hypothesis made quantitative.
  - Then measure, per pruned neuron set, the fraction of each feature's decoder norm removed (‖W_dec[f, removed]‖² / ‖W_dec[f]‖²). Correlate that with per-feature activation loss in the student, and with D_res.
- **INFERENCE (pruning), "WHAT is destroyed":** Run the parent's SAE on the pruned student (before and after distillation) and on the same inputs. Per feature, record:
  - activation correlation;
  - firing-rate change;
  - dead in student (yes/no);
  - "new" activations (false positives from extra interference).
  Group the features by autointerp label (glyph or script features, layout or table features, language features, copy-related features) to name what was lost.
- **INFERENCE (pruning), fix:**
  - Use feature-aware neuron selection: choose the neurons to keep so as to maximise importance-weighted retained decoder norm of SAE features, rather than per-neuron saliency.
  - Or add a feature-matching distillation loss: the student's SAE-encoded activations should match the parent's, using the parent's SAE as a fixed, interpretable target basis.
- **INFERENCE (pruning):** The roughly 35% of unexplained variance at Sonnet scale (if the ≥65% figure is right) is a warning. SAE-based damage accounting will miss part of D_res. Report the SAE error term separately; see "dark matter" in Q7.

### Gaps
- I could not verify the exact Towards Monosemanticity dictionary sizes, the loss-recovered percentages, the feature-splitting statistics or the numerical universality correlations.
- I found no Anthropic work that applies SAEs before and after pruning or distillation.

---

## Q5. Model diffing: can parent and pruned/distilled student be diffed directly? (Sparse Crosscoders, 2024; Stage-wise Model Diffing, 2024; Insights on Crosscoder Model Diffing, 2025; Dedicated Feature Crosscoder, 2025)

### Takeaway
Crosscoders learn one shared sparse dictionary that reads from, and reconstructs, activations of several layers or several models. Features with large decoder norm in only one model are model-specific: "learned, or forgotten". This is close to a ready-made tool for diffing parent against student. Anthropic and others have since found artifacts (L1 sparsity creates false "exclusive" features) and fixes: BatchTopK, latent scaling, and dedicated exclusive-feature partitions. The fixes matter for a parent/student pair with different widths and depths.

### Cited Findings
- **Sparse Crosscoders (Lindsey et al. 2024), NV:**
  - At equal feature count, "crosscoders substantially outperform per-layer SAEs on eval loss. This result indicates that there is a significant degree of redundant (linearly correlated) structure across layers."
  - In model diffing, "model-specific features would indicate features learned, or forgotten, during finetuning".
  - A secondary summary adds that crosscoders give "long-term feature tracking", simplify circuit analysis by removing redundant features, and cost more compute.
  - Sources: [Transformer Circuits, crosscoders](https://transformer-circuits.pub/2024/crosscoders/index.html), via search snippets; [summary on X](https://x.com/TheAITimeline/status/1850237737154015607).
- **Stage-wise Model Diffing (Bricken et al. 2024), NV:**
  - The method fine-tunes an SAE along two paths through four stages:
    - S: base model, base data
    - D: base model, sleeper data
    - M: sleeper model, base data
    - F: sleeper model, sleeper data
  - It flags features that change in response to both the data change and the model change.
  - In the sleeper-agent experiments, the stage-wise diff "showed higher sensitivity" than a crosscoder diff. The crosscoder found sleeper features but "failed to separate them from many unrelated features".
  - Source: [Transformer Circuits, model diffing](https://transformer-circuits.pub/2024/model-diffing/), via search.
- **Stage-wise replication, NV:** Soligo et al. (MARS) replicated the key results on TinyStories-Instruct-33M, using single-model all-layer crosscoders. [Alignment Forum replication](https://www.alignmentforum.org/posts/hxxramAB82tjtpiQu/replication-crosscoder-based-stage-wise-model-diffing-2)
- **Insights on Crosscoder Model Diffing (2025):** Anthropic's stub page confirms it exists as "developing work… preliminary experiments" [V-blog]; its content is NV. [Anthropic, crosscoder model diffing](https://www.anthropic.com/research/crosscoder-model-diffing) → [Transformer Circuits, 2025 diffing update](https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html)
- **Related academic work, Minder et al. 2025, NV:** On Gemma-2-2B base versus chat:
  - With L1 crosscoders, most apparent "chat-only" latents were not actually chat-specific.
  - The authors introduce "latent scaling" to measure how model-specific a latent is.
  - A BatchTopK crosscoder "substantially mitigates these issues, finding more genuinely chat-specific and highly interpretable concepts".
  - Many features with high Δ_norm in L1 crosscoders are "uninterpretable or causally irrelevant".
  - Source: [arXiv 2504.02922](https://arxiv.org/abs/2504.02922), via search.
- **Dedicated Feature Crosscoder (Jiralerspong and Bricken, Anthropic Fellows, 2025) [V-blog]:**
  - The project extends model diffing "to its most challenging and general use case: comparing models with entirely different architectures".
  - A standard crosscoder "struggles to find words that are unique to one language… tries to force an imperfect translation". The DFC splits the dictionary into a shared section and two model-exclusive sections.
  - "A single diff can surface thousands of unique features… only a small fraction of these may correspond to meaningful behavioral risks… a high-recall screening tool."
  - Validation is by steering, suppressing or amplifying a feature.
  - Robustness: the CCP-alignment feature was "independently rediscovered five out of five times", and American Exceptionalism "four out of five".
  - Models: Llama-3.1-8B-Instruct, Qwen3-8B, GPT-OSS-20B, DeepSeek-R1-0528-Qwen3-8B.
  - Source: [Anthropic, diff tool](https://www.anthropic.com/research/diff-tool).
- **Cross-architecture diffing paper, NV:** A 2026 paper reports that standard crosscoders have a prior against model-exclusive features, because the objective rewards shared features. [arXiv 2602.11729](https://arxiv.org/pdf/2602.11729)

### Inferences
- **INFERENCE (pruning), most direct tool for "WHAT is destroyed":**
  - Train a BatchTopK crosscoder, or better a DFC, on paired activations of the parent and the pruned student on identical OCR inputs.
  - Use the parent's residual at layer l and the student's residual at the corresponding depth. For dropped layers, pick the student layer whose output the optimal linear map aligns to.
  - Parent-exclusive features are the destroyed features. Student-exclusive features are artifacts, or new features created by distillation.
  - Using a DFC avoids the standard crosscoder's bias toward forcing a "shared" match. That bias would under-count destroyed features, which is exactly the error to avoid.
- **INFERENCE (pruning), separating D_lin from D_res in feature space:**
  - Fit the user's optimal linear map from the student layer to the parent layer. Then run the parent's SAE on the linearly mapped student activations.
  - Features recovered after the map belong to D_lin. Features still missing belong to D_res.
  - Prediction under the user's hypothesis: after MLP-neuron pruning, many features stay missing even after the map. After layer pruning, most come back.
- **INFERENCE (pruning), stage-wise diffing maps onto the pipeline:**
  - The stages would be: parent; pruned before KD; pruned after KD; and optionally a parent fine-tuned on the KD data as a control. This separates features destroyed by the cut from features restored or re-learned by distillation.
  - Anthropic found the stage-wise variant more sensitive than a crosscoder diff for small, targeted changes.
- **INFERENCE (pruning), crosscoder redundancy explains layer-drop recoverability:** "redundant (linearly correlated) structure across layers" means neighbouring layers write overlapping, linearly related content. This is the Anthropic-side explanation for why a linear map at the cut recovers most layer-removal damage (D_lin), and it can be tested directly. If a cross-layer crosscoder shows that block l's features are also written, or linearly predictable, at l±1, block l should be cheap to drop.

### Gaps
- I could not access the crosscoder paper's quantitative tables (shared fraction, decoder-norm histograms) or the 2025 Anthropic diffing-update content.
- Nobody, Anthropic or otherwise, has applied crosscoders to a pruned-versus-parent pair, as far as I found.

---

## Q6. Attribution graphs and transcoders: where were the circuits lost? (Circuit Tracing and On the Biology of a Large Language Model, 2025; open-source circuit-tracer, 2025)

### Takeaway
Anthropic replaces MLPs with cross-layer transcoders (CLTs) to build a "replacement model". Attention patterns are frozen per prompt and error nodes absorb what the transcoder misses. Attribution graphs then trace feature-to-feature direct effects from input to output. The open-source `circuit-tracer` already supports Qwen3 0.6B–14B transcoders and arbitrary HF models through `nnsight`. Diffing attribution graphs of parent and student on the same OCR prompts is a plausible way to locate lost circuits.

### Cited Findings
- **Tracing the thoughts (2025) [V-blog]:**
  - "we extend our prior work locating interpretable concepts ('features')… to link those concepts together into computational 'circuits'".
  - The method is applied to Claude 3.5 Haiku on ten behaviours.
  - Limitations: "Even on short, simple prompts, our method only captures a fraction of the total computation performed by Claude, and the mechanisms we do see may have some artifacts based on our tools…" and "It currently takes a few hours of human effort to understand the circuits we see, even on prompts with only tens of words."
  - Source: [Anthropic, tracing thoughts](https://www.anthropic.com/research/tracing-thoughts-language-model).
- **Tracing the thoughts, universality across languages and scale [V-blog]:**
  - "Shared features exist across English, French, and Chinese."
  - "the shared circuitry increases with model scale, with Claude 3.5 Haiku sharing more than twice the proportion of its features between languages as compared to a smaller model."
  - Source: [Anthropic, tracing thoughts](https://www.anthropic.com/research/tracing-thoughts-language-model).
- **Tracing the thoughts, multi-step and default circuits [V-blog]:**
  - The two-hop "Dallas → Texas → Austin" intermediate features are visible.
  - There is a default "can't answer" circuit that "known entity" features inhibit.
  - Interventions on features cause predictable output changes.
  - Source: [Anthropic, tracing thoughts](https://www.anthropic.com/research/tracing-thoughts-language-model).
- **Methods, NV:**
  - A CLT has features divided into L layers. Each reads from the residual stream at one layer and contributes to all subsequent MLP outputs.
  - The replacement model swaps CLT features in for MLP neurons and freezes attention patterns.
  - Error nodes fill in discrepancies and "have no interpretable function".
  - They are "especially a problem for complicated prompts requiring many reasoning steps, or unusual / 'off-distribution' prompts".
  - One evaluation table mentions a "Cross-Layer Transcoder (10m)" entry. The 30M-feature Haiku CLT figure was not verified.
  - Sources: [Transformer Circuits, methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) and [Transformer Circuits, biology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html), via search; [learnmechinterp summary](https://learnmechinterp.com/topics/circuit-tracing/).
- **Open-source tools [V-blog]:**
  - The library "supports the generation of attribution graphs on popular open-weights models", with a Neuronpedia frontend.
  - It lets researchers "Test hypotheses by modifying feature values and observing how model outputs change".
  - It was built by Anthropic Fellows (Hanna, Piotrowski) with Decode Research.
  - Source: [Anthropic, open-source circuit tracing](https://www.anthropic.com/research/open-source-circuit-tracing).
- **circuit-tracer README [V-blog]:**
  - It computes "the direct effect that each non-zero transcoder feature, transcoder error node, and input token has on each other non-zero transcoder feature and output logit".
  - Available transcoders: Gemma-2-2B (PLTs, plus CLTs of 426K and 2.5M features), Llama-3.2-1B (PLT and 524k CLT), Qwen-3 PLTs for 0.6B, 1.7B, 4B, 8B and 14B, GPT-OSS-20B (131k CLT), and Gemma-3 270M–27B.
  - `backend='nnsight'` "supports most HuggingFace models" (experimental).
  - Defaults: `max_feature_nodes` 7500, `node_threshold` 0.8, `edge_threshold` 0.98.
  - Locally trained transcoders can be loaded.
  - Source: [GitHub safety-research/circuit-tracer](https://github.com/safety-research/circuit-tracer).

### Inferences
- **INFERENCE (pruning), localising lost circuits (WHERE):**
  - Train per-layer transcoders (PLTs) on the parent's decoder MLPs, or a CLT if compute allows.
  - Build attribution graphs on a fixed OCR probe set: line transcription, table cell, a rare-script token, and a long repeated-text copy case.
  - Use the parent's transcoder features to read the student's activations at the aligned depths; for dropped layers, read through the linear map.
  - Nodes and edges present in the parent but absent or weakened in the student mark the lost circuit pieces, together with their layers.
- **INFERENCE (pruning), error-node mass as a damage meter:** In the student's replacement model built with the parent's transcoders, the error-node share of attributions should grow most where D_res is large. Anthropic notes that error nodes dominate off-distribution, and a pruned student's activations are "off-distribution" relative to the parent's transcoders.
- **INFERENCE (pruning), frozen-attention caveat:** The replacement model freezes attention patterns, so attribution graphs do not explain changes in attention patterns. Damage from layer removal that passes through Q/K composition (Q1) will appear only indirectly. Complement graphs with direct attention-pattern diffs.
- **INFERENCE (pruning), practicality:**
  - The LightOnOCR decoder is reported as a hybrid of GatedDeltaNet and full attention. The Qwen3 transcoders will not apply directly because the weights differ.
  - Train your own PLTs. The `nnsight` backend is the route for a custom HF architecture, though GatedDeltaNet support is untested.
- **INFERENCE (pruning), universality hint:** Anthropic's finding that the larger model shares more than twice the proportion of features across languages implies that smaller models keep more language-specific (duplicated) features. Aggressive pruning toward 0.1B may push the model toward less shared, more duplicated representations. Check multilingual OCR separately.

### Gaps
- I could not verify the methods paper's quantitative fidelity metrics (replacement-model next-token match, normalised MSE, CLT versus PLT comparison, number of Haiku CLT features).
- There is no Anthropic work applying attribution graphs to compressed models, or to linear-attention or state-space layers.

---

## Q7. Circuits Updates (2023–2025): dark matter, SAE errors, multidimensional features, manifolds

### Takeaway
Anthropic's informal monthly updates repeatedly flag three things:
- SAEs leave "dark matter".
- SAE reconstruction errors lie largely along feature directions and are not random.
- Some representations may be multidimensional or manifold-like, though irreducible multidimensional features appear rare.
All three bear on how much of pruning damage an SAE-based audit can see.

### Cited Findings
- **April 2024, NV:** This update discusses Gurnee's "pathological" SAE reconstruction errors. Perturbing along the error direction has a larger effect than a random perturbation, which suggests "SAE reconstruction errors lie primarily along feature directions". [Transformer Circuits, April 2024 update](https://transformer-circuits.pub/2024/april-update/index.html), via search
- **June 2024, NV:** The "Multidimensional Features" section says "irreducible multidimensional features are rare"; most concepts may be built from one-dimensional features (with a caveat about the clustering method used). [Transformer Circuits, June 2024 update](https://transformer-circuits.pub/2024/june-update/index.html), via search
- **July 2024, NV:** "The Dark Matter of Neural Networks?" suggests dictionary learning may miss parts of Claude 3 Sonnet's knowledge. [Transformer Circuits, July 2024 update](https://transformer-circuits.pub/2024/july-update/index.html), via search
- **October 2025, NV:** "When Models Manipulate Manifolds: The Geometry of a Counting Task" is listed. A later citation dates it 2026, so the date is uncertain. [Transformer Circuits](https://transformer-circuits.pub/), via search
- **Stub pages only [V-blog]:** anthropic.com stub pages exist for the May 2023 and April, June, July, August and September 2024 updates; they contain no content. [Anthropic, Sept 2024 updates](https://www.anthropic.com/research/circuits-updates-sept-2024)
- **Related academic work (Engels, Riggs, Tegmark), NV:** "about half of the error vector itself and >90% of its norm — can be linearly predicted from the initial activation vector". [arXiv 2410.14670](https://arxiv.org/html/2410.14670.pdf)

### Inferences
- **INFERENCE (pruning), what a feature audit can and cannot see:**
  - SAE errors are partly structured (along feature directions) and partly linearly predictable (Engels).
  - When using SAE or crosscoder diffs, add an extra term: the change in SAE error norm between parent and student.
  - If the student's error grows substantially, damage is happening in directions the parent's SAE does not capture, so the feature audit under-counts D_res.
- **INFERENCE (pruning), manifold-shaped features:** Representations such as position-in-line, column index or counting (cf. the "counting task manifold" update) may be manifold-shaped. Neuron pruning could distort their geometry without "killing" any single SAE feature. For an OCR model these spatial and counting representations are plausible, and they would show up as degraded line or table structure rather than missing token types.

### Gaps
- None of the update texts were read; every claim above is NV. In particular, I could not confirm whether any update discusses features spread across many neurons with measured participation counts.

---

## Q8. Vision and multimodal features relevant to OCR

### Takeaway
Anthropic's direct multimodal finding is that Claude 3 Sonnet's middle-layer features respond to images and to text about the same entity. Anthropic has published no ViT-specific dictionary-learning or circuit work that I found. Third-party 2026 papers bring cross-layer transcoders and circuit tracing to ViTs and VLMs.

### Cited Findings
- **Scaling Monosemanticity [V-blog]:** features are "multimodal and multilingual, responding to images of a given entity as well as its name or description in many languages". The Golden Gate Bridge feature fires on text in several languages "and an image". [Anthropic, mapping the mind](https://www.anthropic.com/research/mapping-mind-language-model)
- **Chatzoudis et al., CVPRW 2026, NV:**
  - CLTs replacing ViT MLPs give "high reconstruction fidelity" while "preserving and even improving, in some cases, CLIP zero-shot classification accuracy".
  - Substitution is "close to the baseline for late-layer substitution, but degrade[s] when cascading across all layers and all tokens".
  - "the final representation is concentrated in a smaller set of dominant layer-wise terms whose removal degrades performance".
  - "CLS aggregates information across depth, anchored by a broad contribution from the earliest layer… Patch tokens do the opposite", depending mostly on the immediately preceding layer.
  - Sources: [arXiv 2604.13304](https://arxiv.org/pdf/2604.13304); [CVF open access](https://openaccess.thecvf.com/content/CVPR2026W/XAI4CV/html/Chatzoudis_Can_Cross-Layer_Transcoders_Replace_Vision_Transformer_Activations_An_Interpretable_Perspective_CVPRW_2026_paper.html).
- **Yang et al. 2026, NV:** The paper claims "the first framework for transparent circuit tracing in VLMs", using transcoders and attribution. It reports distinct visual-feature circuits for math reasoning and cross-modal association; code is at UIUC-MONET. [arXiv 2602.20330](https://arxiv.org/abs/2602.20330)
- **Transcoders trace visual grounding, NV:** Transcoder circuit tracing on Gemma-3-4B is used to classify hallucinated generations. [arXiv 2605.22902](https://www.alphaxiv.org/abs/2605.22902.md)

### Inferences
- **INFERENCE (pruning), ViT block removal:**
  - OCR VLMs feed patch tokens, not a CLS token, to the decoder.
  - If patch tokens depend mostly on the immediately preceding layer (Chatzoudis), removing a ViT block breaks a strong local dependency. Damage should be larger and less linear than in a CLS-pooled classifier.
  - That argues for preferring decoder-layer cuts over ViT-block cuts, or for cutting ViT blocks only where the CLT attribution shows small layer-wise terms.
- **INFERENCE (pruning), dominant layers:** "dominant layer-wise terms whose removal degrades performance" means a small set of ViT layers carry most of the output. Find them with CLT or attribution analysis, or simply with per-layer ablation plus a linear map, and protect them.
- **INFERENCE (pruning), multimodal features as fragile targets:** If text-in-image features in the decoder are multimodal (the same feature for an image of a glyph and for the token), they are high-value targets for a parent/student diff. Losing them would show up as visual-to-text grounding errors (hallucinated characters) rather than as language-model errors.

### Gaps
- I found no Anthropic paper on ViT or vision-encoder features or on OCR-specific features. The question's "features responding to text in images" was not found as an Anthropic claim; only the general "responds to images of the entity" claim is verified.

---

## Q9. Academic follow-ups that directly connect SAE features to compression, fine-tuning and universality

### Takeaway
Several 2026 preprints apply SAEs before and after pruning and quantization. They find feature damage that perplexity misses, frequency-dependent feature survival, and layer-dependent sensitivity. Nearly all study unstructured weight pruning (magnitude or Wanda), not structured neuron or layer pruning. Feature-transfer results (Kissane) suggest SAEs carry over between closely related weights. Distilled students are further away, so transfer must be checked.

### Cited Findings
- **"How Pruning Reshapes Features" (arXiv 2603.25325), NV:**
  - Rare, low-firing-rate SAE features survive pruning far better than frequent ones. Within-condition Spearman ρ=−1.0 in 11 of 17 conditions.
  - Pruning "performs implicit feature selection by preferentially destroying generic, high-frequency features".
  - Wanda preserves feature structure up to 3.7× better than magnitude pruning.
  - Source: [arXiv 2603.25325](https://arxiv.org/html/2603.25325v1).
- **"Feature survival under weight pruning" (Neurocomputing 2026), NV:**
  - Same firing-rate pattern.
  - Wanda preserves SAE feature geometry better than magnitude pruning.
  - "geometric feature survival does not predict causal importance".
  - Source: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925231226022241).
- **"When Pruning Meets Interpretability" (arXiv 2608.25941), NV:**
  - Middle layers are significantly more sensitive to pruning than early or late layers.
  - Magnitude pruning "induces large perturbation energy and significantly degrades SAE behavior".
  - The authors propose layer-wise sparsity allocation.
  - Source: [arXiv 2608.25941](https://arxiv.org/abs/2608.25941).
- **"Perplexity Can Miss SAE Feature Damage Under Quantization" (arXiv 2606.03002), NV:**
  - Perplexity can stay stable or improve while many SAE features degrade.
  - Quantization damages many of the same features as pruning at matched perplexity.
  - The change is graded, with no sharp precision cliff.
  - Source: [arXiv 2606.03002](https://arxiv.org/html/2606.03002v1).
- **Kissane et al. 2024, "SAEs (usually) transfer between base and chat models", NV:**
  - Splicing a base SAE into the chat model "achieves similar CE loss to the chat SAE on both Mistral-7B and Qwen 1.5 0.5B".
  - Transfer fails on Gemma v1 2B, which the authors attribute to unusually large weight differences and outlier-norm tokens such as BOS.
  - A base SAE fine-tuned on about 5M chat activations approaches a chat SAE trained from scratch.
  - Source: [Alignment Forum post](https://alignmentforum.org/posts/fmwk6qxrpW8d4jvbd/saes-usually-transfer-between-base-and-chat-models).
- **Lan et al., arXiv 2410.06981 (later titled "Quantifying Feature Space Universality Across LLMs via SAEs"), NV:**
  - SAE feature spaces across different LLMs are substantially similar under SVCCA and RSA, which the authors call "weakly universal".
  - Similarity is higher for semantically grouped subspaces, such as emotions.
  - Different SAEs do not consistently learn the same universal features.
  - Source: [arXiv 2410.06981](https://arxiv.org/pdf/2410.06981).
- **"On the creation of narrow AI" (arXiv 2505.15811), NV:**
  - Pruning neurons by their effect on one subtask's loss did not fully remove other skills.
  - Polysemantic neurons and superposition are given as the reason.
  - Retraining after pruning recovered performance at higher sparsity.
  - Source: [arXiv 2505.15811](https://arxiv.org/pdf/2505.15811).
- **No clean structured-pruning study found:** No published study cleanly isolates interference-driven damage from structured neuron removal. The search summary noted an inconclusive bachelor's thesis (FernUni Hagen) in which polysemanticity tended to decrease after pruning plus retraining. [FernUni thesis](https://www.fernuni-hagen.de/aig/docs/bachelor.plettscher.pdf)

### Inferences
- **INFERENCE (pruning), perplexity is a blunt instrument:** The quantization result implies that zero-shot loss or CER damage can understate feature damage. The user's "per unit of zero-shot damage" comparison could be complemented by per-unit feature damage, where MLP pruning might look even worse.
- **INFERENCE (pruning), high-frequency features at risk:** Pruning destroys generic, high-frequency features preferentially (2603.25325, unstructured). For OCR those are common-glyph and common-token features: the bulk of CER. Activation-aware criteria (Wanda-like, or feature-aware ones) should be preferred over magnitude criteria even for structured pruning. That is an extrapolation from unstructured pruning.
- **INFERENCE (pruning), transfer must be measured:** Kissane's failure mode (large weight differences, outlier-norm tokens) is likely to occur after heavy pruning plus distillation. Before using the parent's SAE on the student, check spliced-in CE or CER and reconstruction FVE on student activations. If transfer is poor, use a crosscoder or DFC (Q5), or fine-tune the parent's SAE on student activations, as Kissane did for chat models.
- **INFERENCE (pruning), research gap:** A controlled study is missing. It would take the same parent, then structured neuron pruning versus layer pruning at matched zero-shot damage, and report the per-feature survival rate after the optimal linear map (D_lin) and the SAE-error growth (D_res). That would be a novel test of the superposition explanation.

### Gaps
- None of these preprints were read in full; peer-review status was not checked.
- None, as far as I found, studies structured (neuron or layer) pruning, VLMs or OCR.
