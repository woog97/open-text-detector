# Verification

- All seven frozen profile files validated: hashes, unique IDs, text hashes and label intervals.
- Four focused validation tests passed, including corrupted code/data and invalid provenance labels.
- All three models completed an eight-row workflow smoke test on the existing A100 in BF16: ours, MELD v5 and MELD v8. Saved receipts are in `verification/`.
- Exact A100 dependency versions are in `requirements-a100.txt`; model and reference-tokenizer file hashes are in the per-model run locks.
- The Hugging Face package was downloaded back at its recorded commit, byte-verified, and all four configurations loaded through `datasets`. Revision and split counts are in `huggingface_upload.json`.

These checks validate packaging and execution, not a new full benchmark measurement. Existing benchmark scores remain unchanged. A full run of the new workflow profile has not yet been performed through this wrapper.
