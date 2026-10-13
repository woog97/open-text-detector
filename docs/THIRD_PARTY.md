# Third-party references and attribution

Pangram and Open Pangram are names of Pangram Labs. This independent private workspace is not affiliated with Pangram and does not claim to reproduce its proprietary models, accuracy, plagiarism index, or commercial integrations.

The requested model is [pangram/editlens_Llama-3.2-3B](https://huggingface.co/pangram/editlens_Llama-3.2-3B). Its model card specifies **CC BY-NC-SA 4.0**, English, and the Meta Llama 3.2 3B base. The adapter and base have separate gated access requirements.

`backend/open_text_detector_backend/providers/preprocess.py` adapts the preprocessing behavior published in [Pangram Labs/EditLens](https://github.com/pangramlabs/EditLens/blob/main/scripts/preprocess.py), adding original-text offset mappings. `providers/editlens.py` adapts its NormedLinear architecture and expected-bucket score from [train.py](https://github.com/pangramlabs/EditLens/blob/main/scripts/train.py) and [inference.py](https://github.com/pangramlabs/EditLens/blob/main/scripts/inference.py), adding lazy loading, CPU/MPS support, and chunking. The upstream work and these adaptations are subject to [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Other backend files are original workspace implementation; no commercial rights to Pangram's models are implied.

Reference: Katherine Thai, Bradley Emi, Elyas Masrour, Mohit Iyyer. *EditLens: Quantifying the Extent of AI Editing in Text*. ICLR 2026. [arXiv:2510.03154](https://arxiv.org/abs/2510.03154).

The interface was informed by the user's signed-in Pangram dashboard and the public [Pangram site](https://www.pangram.com/). No paid plan was purchased. No private account history or user documents were imported into this workspace.

Visual reference assets in `app/public/reference/` (headline, wordmark, fox and dashboard backgrounds) are copied from publicly served Pangram assets for this requested private replica. These assets remain Pangram’s property; their inclusion does not imply a license for commercial redistribution. Original sources: `pangram-public.s3.us-east-1.amazonaws.com/web/dashboard/headline-text.svg`, `/web/brand/Logo%2BDark+Wordmark.svg`, `/web/assets/dashboard/background-v2.avif`, `/web/assets/dashboard/background-mobile.avif`, and `www.pangram.com/_next/static/media/pangram-4-fox.941e5ba2.webp`.

The active merged Qwen checkpoint is [DarrenJiaImbue/editlens-qwen3-4b-merged-v3](https://huggingface.co/DarrenJiaImbue/editlens-qwen3-4b-merged-v3), CC-BY-NC-SA-4.0, revision `33716e3667e3514a92cd1e9a6f1511655f12ef86`. `providers/qwen.py` adapts the original EditLens NormedLinear head and this model card’s loading instructions. See `research/README.md` for dataset attribution and limitations.


MELD v5 uses the MIT-licensed checkpoint and reference scorer from
[anon-review-meld-2026/meld](https://huggingface.co/anon-review-meld-2026/meld), revision
`453acf594d48f8c55c3a38bde396f9178516d817`. `providers/meld_model.py` adapts that scorer
with strict loading and token evidence output. The localization and window stitching
are workspace additions. The downloaded model card is retained beside the checkpoint.
Attribution: MELD contributors (Chenjun Li, Cheng Wan, Johannes C. Paetzold).

MIT license terms for the adapted reference scorer:

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Laya

English weights: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya), revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, Apache-2.0.
Reviewed architecture and input format adapted from [NandhaKishorM/laya v0.3.20](https://github.com/NandhaKishorM/laya/tree/v0.3.20), developed by Convai Innovations. License retained at `docs/licenses/laya-Apache-2.0.txt`. Local modifications add targeted phrase inputs, bounded paragraph context, portable strict weight loading, and native MLX inference with marker-only final-layer queries. No upstream model Python files are executed. This is an experimental zero-shot baseline, not a validated AI-authorship detector.
