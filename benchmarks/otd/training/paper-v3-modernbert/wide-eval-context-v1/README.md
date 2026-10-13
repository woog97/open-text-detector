# Paired human-context sensitivity check

Same 18,222 held-out human focal paragraphs as the primary wider suite, with their original adjacent paragraphs supplied. Only focal paragraph tokens/sentences contribute to span metrics; neighbors carry -100 metric masks. The clean novel subset contains 7,713 paragraphs.

This supplementary check was added after observing paragraph-isolation false positives. It uses the unchanged checkpoint and thresholds and is not an independent test population. See ../wide-eval-v1/README.md for results, safeguards, and the human training pools.
