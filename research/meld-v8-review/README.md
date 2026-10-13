---
language: en
license: mit
inference: false
tags:
  - ai-text-detection
base_model: jhu-clsp/ettin-encoder-1b
---

# MELD

An AI-generated text detector for English. Everything needed to run it is in this repository.

```bash
pip install torch transformers safetensors
python meld.py "Paste the text to check here."
```

For each input, `meld.py` prints a score, `p_ai`, and whether the text is flagged at the shipped 1% false-positive threshold (`score_offsets` in `meld_config.json`). Compare scores against that threshold, or against scores of your own human-written reference texts, rather than against a fixed cut. Inputs should be at least about 100 words. The scoring head is custom, so `pipeline()` and `AutoModel` do not apply; use `meld.py` or the `Scorer` class inside it.
