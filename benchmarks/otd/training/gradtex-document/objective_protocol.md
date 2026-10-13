# Auxiliary document supervision: controlled pilot

Two new isolated ModernBERT-large LoRA arms use the existing seed-42 recipe, original checkpoint selection, BF16 forwards, and frozen evaluation/calibration separation. Original checkpoints and results are retained.

- `objective-document-control-v1`: identical control data/order and processed tokens; adds a two-class auxiliary document head on the mean nonwhitespace source hidden states. Stage 1 remains the original segment loss. Stage 2 is the original paper combined loss plus 0.1 document cross entropy. A paper document label is inferred from complete known character regions only: AI-involved if any region is AI; human if all regions are human. Unknown or incomplete provenance is ignored.
- `gradtex-document10-v1`: same new head and objective, with 10% of stage-2 input-token exposure replaced by audited whole GRADTEX documents. Document-only rows contribute only 0.1 document cross entropy. They carry no token, sentence, segment, or mixedness labels. The remaining paper combined loss is weighted by paper-row fraction to preserve the contribution per retained paper row. Stage 1 remains control data.

Both arms retain original combined *paper* selection loss, without the new document term, for checkpoint selection. Standard reported document scores remain mean token AI probabilities: the auxiliary head is a training experiment and is not deployed. This distinguishes auxiliary representation learning from switching the inference score. Native document labels never become token gold.

Whole mixed documents must fit 510 text tokens plus two special tokens. Cropping/truncation is prohibited because it could remove the AI contribution. Training donors must pass source/response overlap exclusions against original heldouts, MAGE official validation/test, and all frozen profiles plus paper selection/calibration. Source caps, near-equal human/AI donor token exposure, matched lengths, fixed draw count and input-token budget are preparation gates.

The added head initializes inside a saved/restored CPU RNG context, preserving original initialization order for all existing parameters and LoRA. Model forwards run only inside BF16 autocast. Loss calculations may use FP32.

Validation gates: syntax; BF16 synthetic-logit gradient check; GPU preflight tests finite adapter gradients/frozen-base gradients, BF16 outputs, trainable-state roundtrip, and a document-only real forward/backward with zero token/segment/mixed-head gradients and nonzero document-head/adapter gradients. Preflight writes the existing code/config/manifest hashes before training. No warm-start from preflight.

Scientific limits: any-AI document classification does not estimate AI fraction or locate edits. GRADTEX replacement simultaneously changes source distribution and supervision granularity. Compare the treatment to the new head control, and the new head control to the old paper control. A single seed with different observed false-positive rates does not establish improvement.
