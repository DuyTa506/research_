# I5 / I7: Causal localisation of pruning damage, and interpretability-guided pruning or recovery

**Read-depth disclosure (important).** I read **0 papers in full text** this session. arxiv.org, alphaxiv.org and openreview.net were blocked beforehand. During the session, proceedings.neurips.cc, proceedings.mlr.press, excv-workshop.github.io and iphome.hhi.de all failed with ENOTFOUND, so each must be allowed in the session's network settings before its PDF can be read. I did not try aclanthology.org, because the coordinator's note said to stop once hosts fail. I read in full **3 GitHub READMEs**: princeton-nlp/Edge-Pruning, mit-han-lab/duo-attention and erfanhatefi/Pruning-by-eXplaining-in-PyTorch. That last README has no numbers. Everything else comes from search-engine snippets and abstracts and is marked **NV** (not verified in full text). Numbers marked NV may differ from the camera-ready versions.

Project context used for inferences: LightOnOCR-3-0.8B (hybrid GatedDeltaNet + full-attention decoder, 12-block ViT), pruned to about 0.3B or 0.1B and recovered with logit-only KD; hidden-state KD is under consideration. Damage is split into D_lin (removable by an optimal linear map at the cut) and D_res (what remains).

---

## I5: Can damage be localised causally between parent and compressed student (activation, path or attribution patching, cross-model patching, causal or IIT distillation)? Has anyone used the localisation to pick where to apply hidden-state KD or repair?

### Takeaway
- **Methods exist; direct evidence is thin.** I found very little published work that patches parent activations into a pruned student layer by layer and then uses the result to choose which layers to distil. The closest pieces are the four below.
- **(a) Quantization, NV.** A NeurIPS 2026 study raises each layer to 8-bit in turn (a causal intervention per layer). It reports that the layers where activation patching localises the computation do **not** reliably predict where repair helps.
- **(b) Teacher-to-student layer patching, NV.** "Boomerang distillation" and a 2026 follow-up swap teacher blocks into a distilled student and choose which student layers to swap using per-layer logit-lens KL.
- **(c) Layer-pruning interface repair (LinearPatch, NV).** Most of the damage from layer pruning comes from a linear magnitude mismatch at the cut, which a fused linear map fixes. This is effectively your D_lin.
- **(d) Error propagation from the full-precision parent in PTQ (GPTAQ, QEP).** Each compressed layer is calibrated against the parent's exact outputs.
- **What does exist on KD layer choice:** heuristics such as Minitron's rule (add intermediate-state KD only when depth is cut substantially) and task-aware layer-wise KD (TED). These are not causal localisation.

### Cited Findings

**A. Patching method references** (standard methodology for running such a diagnostic)
- **Zhang & Nanda, "Towards Best Practices of Activation Patching", ICLR 2024 (arXiv 2309.16042), NV.**
  - The choice of corruption method and metric changes which components look important.
  - The paper recommends sliding-window patching (patching several adjacent layers together) over single-layer patching for localisation.
  - [ICLR 2024 proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/06a52a54c8ee03cd86771136bc91eb1f-Paper-Conference.pdf) (search snippet)
- **Denoising versus noising patching (Heimersheim & Nanda, 2404.15255; LessWrong version), NV.**
  - Denoising (clean into corrupt) tests whether the patched activations are *sufficient* to restore behaviour.
  - The guide recommends continuous metrics (logit difference, log-prob) over sharp discrete ones.
  - [LessWrong: How to use and interpret activation patching](https://www.lesswrong.com/posts/FhryNAFknqKAdDcYy/how-to-use-and-interpret-activation-patching) (snippet)
- **Cross-model patching, NV.**
  - A 2026 "Weight Patching" paper separates standard clean/corrupt patching from a cross-model variant: activations from one model are patched into another model's forward pass on the same input.
  - It frames the cross-model variant as a way to locate where behaviour-relevant information takes effect.
  - [arXiv 2604.13694](https://arxiv.org/pdf/2604.13694) (snippet)
- **Cross-model activation patching (CMAP), NV.**
  - Prakash et al. 2024 is cited as the origin of CMAP, which restores specific activations of one model to another model's counterparts.
  - Search snippets say quantization studies have adapted it: "in 4-bit quantized models, patching in FP16 'clean' activations at the last subject token restores the model's predictive accuracy."
  - I could not identify which paper that sentence came from, so treat it as **unattributed, NV**. [search result context](https://neurips.cc/virtual/2026/156934)
- **Other method references the user listed** (causal tracing 2202.05262, path patching 2211.00593, AtP* 2403.00745, IIT 2112.00826).
  - None was retrieved this session.
  - I have no new numbers for them. They are canonical methodology, not compression results.

**B. Studies that localise compression damage causally**
- **"The Structure of Quantization Damage in LLMs: Why the Next Bit Should Be Spent Globally" (Hu & Ramachandran), arXiv 2609.01587, NeurIPS 2026, NV.**
  - **Method.** The ground truth is a causal mixed-precision intervention: raise each layer to 8-bit in turn and measure the accuracy recovered, across **9 open-weight models in 4 architecture families**.
  - **Localisation.** Causal activation patching places the computation at the boundary layers (first and last MLP).
  - **Repair test.** Protecting that first/last MLP pair at 8-bit recovered anywhere from **about 55% of the gap (Mistral-7B) to a negative amount (Qwen2.5-0.5B)**.
  - **Weight statistics.** They fail to identify the damaged layer, except in the LLaMA family.
  - **Main claim.** Where the computation happens does not reliably predict where repair helps.
  - Sources: [arXiv PDF](https://arxiv.org/pdf/2609.01587); [NeurIPS 2026 page](https://neurips.cc/virtual/2026/156934) (snippets only)
- **Apple coreai-optimization docs (engineering practice, not a paper).**
  - Run the compressed and original models on the same inputs, compute per-layer activation SNR, and treat the lowest-SNR layers as most affected.
  - The docs caution that random inputs can rank layers differently from real data.
  - [Comparing Intermediate Activations](https://apple.github.io/coreai-optimization/utils/activation_comparison.html) (snippet)
- **"Perplexity Cost Understates What Activation Quantisation Breaks", arXiv 2609.23125.**
  - It appeared in search results on this topic; I could not retrieve its content (NV, no numbers).
  - [arXiv](https://arxiv.org/pdf/2609.23125)

**C. Patching teacher (parent) layers into a student: the closest analogue to the proposed diagnostic**
- **Boomerang distillation, arXiv 2510.05064, NV.**
  - Distil a small student from a teacher, then patch selected student layers with blocks of teacher layers. Each student layer is replaced by its corresponding block of teacher layers.
  - This yields intermediate-size models with zero additional training.
  - [arXiv 2510.05064](https://arxiv.org/pdf/2510.05064) (snippet)
- **"Understanding Layer Patching in Model Size Interpolation", arXiv 2607.08170 (2026), NV.**
  - Describes itself as the first systematic study of which student layers to select for patching.
  - Selection uses the logit lens: compare each student layer's output with the teacher's and measure per-layer KL.
  - This is a per-layer "which patch restores the output" signal.
  - [arXiv abs](https://arxiv.org/abs/2607.08170) (snippet; results not seen)

**D. Interface repair: localising damage at the cut (direct D_lin analogue)**
- **LinearPatch, "A Simple Linear Patch Revives Layer-Pruned Large Language Models", arXiv 2505.24680, NeurIPS 2025, NV.**
  - **Diagnosis.** Most degradation from layer pruning comes from mismatched activation magnitudes at the pruning interface.
  - **Fix.** A Hadamard transform (suppresses massive outliers at special tokens) combined with channel-wise scaling, fused into a single matrix at the interface.
  - **Result.** Pruning 5 of 32 layers from LLaMA-3-8B keeps **up to 94.15%** of the dense model's question-answering performance, versus **90.84%** for LLM-Streamline.
  - **Plus distillation.** Offline distillation on **5K unlabeled samples** raises retention to **95.16%** in about **30 min on one GPU**.
  - Sources: [arXiv HTML](https://arxiv.org/html/2505.24680v2); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/136221) (snippets)

**E. Error propagation from the parent in PTQ (parent-referenced, layer-local repair)**
- **GPTAQ (Li et al., arXiv 2504.02692, 2025), NV.**
  - Keeps GPTQ's second-order compensation but changes the target: each quantized layer's output is matched to the full-precision model's exact output.
  - This reduces error accumulated in earlier layers.
  - [emergentmind summary](https://www.emergentmind.com/topics/gptaq) (snippet)
- **QEP (Arai & Ichikawa, NeurIPS 2025), NV.**
  - Each layer's objective matches the original layer output given the full-precision input (from full-precision upstream weights), not the shared quantized input.
  - α runs from 0 (plain layer-wise PTQ) to 1 (full correction).
  - On Llama-2-7B, RTN shows near-exponential error growth within quantized blocks; QEP suppresses that growth and the growth in later full-precision blocks.
  - [NeurIPS 2025 slides](https://neurips.cc/media/neurips-2025/Slides/117304_MX9FETs.pdf) (snippet)
- **Independent check (OneComp, arXiv 2603.28845), NV.**
  - Naive layer-wise PTQ was insufficient for aggressive quantization.
  - QEP "consistently restored model quality across all settings".
  - [arXiv 2603.28845](https://arxiv.org/pdf/2603.28845) (snippet)
- **Critique (arXiv 2604.07955), NV.**
  - Argues GPTAQ's residual objective is exact only at the first quantization step within a layer, and proposes a compensation-aware term.
  - [alphaxiv abs](https://www.alphaxiv.org/abs/2604.07955) (snippet)

**F. Causal or interchange-intervention distillation**
- **Wu, Geiger et al., "Causal Distillation for Language Models", arXiv 2112.02505, NAACL 2022, NV.**
  - Adds a third objective (DIITO, based on interchange intervention training) so the student imitates the teacher's causal computation. Student and teacher logits are matched under aligned interchange interventions.
  - Reported over standard DistilBERT trained the same way:
    - lower WikiText-103 perplexity
    - **GLUE +1.77%** (best configuration)
    - **SQuAD v1.1 EM +2.46%**
    - **CoNLL-2003 F1 +0.38%**
  - Sources: [arXiv abs](https://arxiv.org/abs/2112.02505v1); [deepai summary](https://deepai.org/publication/causal-distillation-for-language-models) (snippets)
  - The alignment between student and teacher layers is fixed by hand. It is not chosen by localisation.

**G. How hidden-state KD layers are chosen in practice (heuristic, not causal)**
- **Minitron (arXiv 2407.14679, NeurIPS 2024; and arXiv 2408.11796), NV.**
  - Layer importance is scored by LM validation loss, Block Importance (BI) or downstream accuracy; the task metric worked better than LM loss.
  - The intermediate-state KD loss uses a shared learned linear map that lifts the student's hidden state to the teacher's width.
  - **Best practice:** use logit + intermediate-state + embedding KD when depth is reduced substantially; use logit-only KD when it is not.
  - Sources: [arXiv 2408.11796](https://arxiv.org/pdf/2408.11796); [NVlabs/Minitron](https://github.com/nvlabs/minitron) (snippets)
  - A secondary blog reports that mapping the student's last layers to the teacher's last layers worked best (e.g. teacher 30–31 → student 14–15). That claim is **unverified in the primary paper**. [Medium summary](https://ritvik19.medium.com/papers-explained-208-minitron-e55ea374d9dd)
- **Observability paper, arXiv 2605.11093 (2026), NV.**
  - Adding hidden-state matching to logit KD "consistently outperforms logits-only distillation by 2–3% across all student sizes", using forward layer matching.
  - [arXiv 2605.11093](https://arxiv.org/pdf/2605.11093) (snippet)
- **TED, task-aware layer-wise distillation (Liang et al., ICML 2023, PMLR v202).**
  - Relevant to layer-wise matching with task-aware filters.
  - The PDF host failed with ENOTFOUND, so I have **no numbers**. [PMLR PDF](https://proceedings.mlr.press/v202/liang23j/liang23j.pdf)

### Inferences
- **The D_lin map is the patching adapter.** Under width pruning, a parent activation (d_parent) cannot be patched directly into the student (d_student). The optimal linear map at each cut is exactly the adapter that cross-model patching needs.
  - A practical protocol: at each candidate site ℓ (each ViT block, each GatedDeltaNet / full-attention block, the projector), replace the student's residual stream with W_ℓ·h_parent,ℓ (denoising) and measure the recovery of token-level KL or CE on OCR outputs.
  - Use KL or CE rather than CER, which is too discrete as a patching metric (per Heimersheim & Nanda and the LessWrong guide).
  - Then reverse the direction: patch student activations into the parent (noising) to test necessity.
  - Use sliding windows of 2–3 blocks, following Zhang & Nanda.
- **Patching restores state; it does not show where to fix.** The quantization-damage paper (NV) found that causal localisation of computation did not predict where repair helps; the recovery gain varied from about 55% of the gap to negative across models.
  - Measure **repair benefit directly**: for each site, fit or train a local fix (linear map, or a hidden-state KD term on that layer only) and measure end-to-end recovery.
  - Do not assume that "restores the most when patched" equals "best place for hidden-state KD".
- **Boundary layers deserve attention.** The quantization-damage (NV) and LinearPatch (NV) results both point to boundary or interface sites.
  - For an OCR VLM, the candidate equivalents are the ViT output and projector into the decoder, the first and last decoder blocks, and each depth-pruning seam.
- **Hybrid recurrent layers may carry damage forward.** In a hybrid GatedDeltaNet decoder, patching the residual stream at a GatedDeltaNet layer does not reset that layer's recurrent state.
  - Damage carried in the recurrent state could persist after a residual patch.
  - Patching the state directly, or patching the per-token outputs of the GatedDeltaNet block, should be a separate condition. This is untested in the literature I found.
- **Minitron's rule predicts where hidden-state KD helps.** It favours hidden-state KD when depth is cut substantially.
  - If the 0.1B variant removes many decoder blocks, hidden-state KD at the seams is the most literature-supported choice.
  - For mostly width-pruned variants, logit-only KD may already be close to optimal (Minitron, NV).

### Gaps
- I found no paper that runs layer-by-layer patching from parent into a *pruned* (not quantized) student and then uses the ranking to choose hidden-state KD layers, with an ablation against uniform or last-layer matching.
- I found no work on patching between parents and students in hybrid linear-attention / SSM models.
- No exact numbers for Zhang & Nanda, AtP*, path patching or causal tracing in a compression setting; none were retrieved.
- I could not attribute the "FP16 activations restore 4-bit model at last subject token" claim to a specific paper.
- Per-model tables for 2609.01587, 2607.08170 and TED were inaccessible (arXiv and PMLR blocked).

---

## I7: Interpretability-guided pruning or recovery: measured gains over magnitude, Wanda or Taylor criteria

### Takeaway
- **Attribution-based criteria (LRP).** They beat gradient, Taylor and magnitude criteria in the original CNN transfer-learning study (Yeom et al. 2021, NV). The 2024 "revisited" paper reports higher ViT and CNN compression rates at high ImageNet accuracy, but I could not retrieve its tables.
- **Subspace-preserving methods (SliceGPT and successors).** These keep principal activation subspaces; SliceGPT retains about 96–99% of zero-shot performance at 25% removal on 70B-class models (NV; the two sources conflict).
- **Head-aware KV compression.** Retrieval-head-aware methods (RazorAttention, DuoAttention) protect a small set of functionally identified heads and report more than 70% KV reduction or 1.5–2.5× memory savings with minimal loss.
- **Circuit discovery by pruning (Edge Pruning).** It is an interpretability tool, not a compression method.
- **SAE-guided pruning.** I found no SAE-feature pruning criterion with measured gains. SAE work so far is diagnostic: Wanda preserves feature geometry better than magnitude pruning, but feature survival does not predict ablation impact.

### Cited Findings

**A. Pruning by explaining (LRP)**
- **Yeom et al., "Pruning by Explaining: A Novel Criterion for Deep Neural Network Pruning", Pattern Recognition 2021 (arXiv 1912.08881), NV.**
  - The gradient and Taylor-based measures "do not reach the performance of LRP-based pruning".
  - Magnitude-criterion performance is constant across reference-sample settings because it depends only on the learned weights.
  - The setting is CNNs and toy models in transfer learning. I could not retrieve exact accuracies; the HHI PDF host failed.
  - Sources: [arXiv HTML](https://arxiv.org/html/1912.08881v3) (snippet); [HHI PDF, unreachable](https://iphome.hhi.de/samek/pdf/YeoPR21.pdf)
- **Hatefi, Dreyer, Achtibat, Wiegand, Samek, Lapuschkin, "Pruning By Explaining Revisited: Optimizing Attribution Methods to Prune CNNs and Transformers", arXiv 2408.12568, ECCV 2024 eXCV workshop, NV.**
  - Optimises the attribution-method hyperparameters specifically for pruning.
  - Reports "higher model compression rates of large transformer- and convolutional architectures (VGG, ResNet, ViT) compared to previous works, while still attaining high performance on ImageNet classification".
  - Finds that "transformers have a higher degree of over-parameterization compared to convolutional neural networks".
  - ViT-B-16 pruning covers linear-layer neurons and attention heads. I could not retrieve the exact table values.
  - Sources: [README, read in full](https://github.com/erfanhatefi/Pruning-by-eXplaining-in-PyTorch); [arXiv HTML](https://arxiv.org/html/2408.12568v2); [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-031-92648-8_10)
- **Accuracy-aware extension (arXiv 2511.10861), NV.**
  - An extension of LRP-based pruning for CNNs that aims to prevent cascading accuracy degradation in data-scarce transfer learning.
  - [arXiv 2511.10861](https://arxiv.org/pdf/2511.10861) (title or snippet only)

**B. SAE / feature-preserving**
- **"Feature survival under weight pruning: Sparse autoencoder diagnostics for compressed language models", Neurocomputing 2026, NV.**
  - **Wanda preserves SAE feature geometry better than magnitude pruning.**
  - **Geometric feature survival does not predict the impact of local ablations.**
  - [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925231226022241) (snippet)
- **"How Pruning Reshapes Features: SAE Analysis of Weight-Pruned Language Models", arXiv 2603.25325, NV.**
  - Three model families, two pruning methods.
  - Rare features survive pruning better than frequent ones: the within-condition correlation between firing rate and survival is **ρ = −1.0 in 11 of 17 conditions**.
  - The authors read this as pruning acting as implicit feature selection.
  - [arXiv HTML](https://arxiv.org/html/2603.25325v1) (snippet)
- **FineScope, arXiv 2505.00624, NV.**
  - Uses an SAE to *select domain data*, which then guides structured pruning and self-distillation.
  - The SAE is used for data selection, not as a weight-level criterion.
  - [arXiv abs](https://arxiv.org/abs/2505.00624)
- **"When Pruning Meets Interpretability: Preserving SAE Robustness in LLMs", arXiv 2608.25941.**
  - Content not retrievable (NV, no claims).
  - [arXiv](https://arxiv.org/pdf/2608.25941)

**C. Subspace-preserving (rotate and slice / low-rank on activations)**
- **SliceGPT, arXiv 2401.15024, ICLR 2024, NV.**
  - Removes up to 25% of parameters (including embeddings) from LLaMA-2 70B, OPT 66B and Phi-2.
  - **Conflict on retained zero-shot performance:**
    - abstract: 99% / 99% / 90%
    - introduction snippet: 99% (OPT 66B) / 96% (LLaMA-2 70B) / 87% (Phi-2)
  - LLaMA-2 70B at 25% with WikiText2 calibration averages **69.75** across zero-shot tasks.
  - Independent work reports noticeable drops at 30–40% on LLaMA-2 13B.
  - Sources: [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/316648eb8b4ffb6010f531b07848c300-Paper-Conference.pdf); [arXiv](https://arxiv.org/pdf/2401.15024); [unixer.de copy](http://unixer.de/publications/img/2024_iclr_sliceGPT.pdf) (snippets)
- **ESPACE (NeurIPS 2024), SVD-LLM, FLAT-LLM.**
  - Searches returned nothing usable. **No numbers.**

**D. Circuit-aware pruning**
- **Edge Pruning (Bhaskar, Wettig, Friedman, Chen), arXiv 2406.16778, NeurIPS 2024 Spotlight.**
  - **README (read in full):**
    - tasks: IOI, Greater-Than, Gendered-Pronoun on GPT-2; Tracr; Boolean Expressions on CodeLlama
    - the threshold is set by binary search for exact sparsity
    - CodeLlama runs needed 32 GPUs across nodes
    - ACDC and EAP baselines use the original code
  - **Paper (snippets, NV):**
    - GPT-2 circuits use **"less than half the number of edges"** of previous methods while being equally faithful
    - competitive with ACDC on overlap with hand-found circuits
    - scales to **CodeLlama-13B, "100× the size"** of models typically handled
  - Sources: [GitHub README](https://github.com/princeton-nlp/Edge-Pruning); [arXiv abs](https://arxiv.org/abs/2406.16778)
  - It is a circuit-*discovery* method; no compression-efficiency gains against Wanda or magnitude are reported.

**E. Head importance (attention-head pruning)**
- **Voita et al., ACL 2019 (arXiv 1905.09418), NV.**
  - Stochastic gates with an L0 relaxation. On English–Russian WMT, **pruning 38 of 48 encoder heads costs only 0.15 BLEU**.
  - Functionally specialised heads (positional, syntactic, rare-word) are the last to be pruned.
  - Sources: [arXiv](https://arxiv.org/pdf/1905.09418); [ACL Anthology P19-1580](https://preview.aclanthology.org/setup/P19-1580)
- **Michel et al., NeurIPS 2019 (arXiv 1905.10650), NV.**
  - A large proportion of heads can be removed at test time without significant impact; some layers can be cut to a single head.
  - The figures of about 20% (WMT) and 40% (BERT) of heads removable before significant drops come from **secondary sources and a reviewer**; I could not verify them in the PDF.
  - Sources: [arXiv](https://arxiv.org/pdf/1905.10650v2); [NeurIPS reviews](https://papers.nips.cc/paper/2019/file/2c601ad9d2ff9bc8b282670cdd54f69f-Reviews.html)

**F. Retrieval-head-aware compression**
- **RazorAttention (Tang et al., arXiv 2407.15891, ICLR 2025), NV.**
  - Training-free. A full KV cache is kept only for retrieval heads; the other heads drop distant tokens, plus a compensation token.
  - Results:
    - **KV cache reduced by more than 70%** without noticeable impact
    - nearly lossless Needle-in-a-Haystack at **50% compression up to 1024K tokens**
    - described as the first training-free method with nearly lossless **3×** KV reduction
    - FlashAttention-compatible
  - Evaluated on Qwen, Llama-2, Llama-3 and Baichuan; per-model numbers were not retrieved.
  - Sources: [arXiv HTML](https://arxiv.org/html/2407.15891v1); [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/2a98af4fea6a24b73af7b588ca95f755-Paper-Conference.pdf) (snippets)
- **DuoAttention (MIT Han Lab), arXiv 2410.10819, ICLR 2025. README read in full.**
  - Retrieval heads keep a full KV cache; streaming heads keep only attention sinks and recent tokens. Heads are identified by a lightweight optimisation-based algorithm trained on synthetic data.
  - **Abstract figures:**

    | Metric (up to) | MHA | GQA |
    |---|---|---|
    | Memory reduction | 2.55× | 1.67× |
    | Decoding speedup | 2.18× | 1.50× |
    | Prefill speedup | 1.73× | 1.63× |

  - **The figure captions give slightly different memory numbers** (2.45× and 1.65×) at 25% retrieval heads for Llama-2-7B (MHA) and 50% for Llama-3-8B (GQA).
  - Needle-in-a-Haystack accuracy is "comparable" to full attention at those ratios.
  - With quantization, Llama-3-8B decodes **3.3M-token context on one A100**.
  - [GitHub README](https://github.com/mit-han-lab/duo-attention)

### Inferences
- **None of these sources gives a clean head-to-head** of an interpretability-derived criterion against Wanda or Taylor *for structured LLM/VLM pruning followed by KD recovery*. The LRP evidence is from CNNs and ViT classification, and the head-aware methods are KV-cache compression, not parameter pruning.
- **The transferable lesson is to protect a small set of causally identified components.** That means retrieval or specialised heads (Voita; RazorAttention / DuoAttention) and boundary or interface layers (Section I5).
  - For an OCR decoder, attention heads that copy from or align to visual tokens are plausible analogues of retrieval heads.
  - They could be found with a DuoAttention-style gate optimisation, or by patching head outputs, and then excluded from pruning. This is my inference and untested for OCR.
- **SAE feature preservation is weak as a pruning criterion today.** The Neurocomputing result says geometric survival does not predict ablation impact.
  - A causal criterion (patching or attribution against the parent) is better motivated than "keep SAE features intact".
- **SliceGPT-style rotation targets your D_lin.** It removes exactly the part of the damage a linear map can absorb. For your D_lin/D_res split, it corresponds to driving D_lin to zero at the slicing stage, so hidden-state KD effort should go to the sites where D_res is large.

### Gaps
- No retrieved numbers for:
  - ViT-B-16 LRP versus magnitude/Taylor (2408.12568 tables)
  - ESPACE, SVD-LLM and FLAT-LLM
  - knowledge-neuron-preserving pruning
  - probing-guided KD layer selection
  - Michel et al.'s exact BLEU deltas
- I found no study that uses AtP or attribution patching as a structured-pruning criterion with measured gains over Wanda.
- I found no interpretability-guided pruning study on VLMs or OCR models, or on hybrid linear-attention models.
- The SliceGPT LLaMA-2 70B retention figure conflicts between sources (99% vs 96%).
- Network blocks prevented full-text verification of nearly every paper; all numbers above are from abstracts or snippets except the DuoAttention and Edge-Pruning READMEs.
