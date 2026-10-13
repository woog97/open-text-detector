"""Reviewed EditLens architectures; never execute code supplied by a model repo."""
QWEN_ID = "DarrenJiaImbue/editlens-qwen3-4b-merged-v3"
QWEN_REVISION = "33716e3667e3514a92cd1e9a6f1511655f12ef86"
MELD_ID = "anon-review-meld-2026/meld"
MELD_REVISION = "453acf594d48f8c55c3a38bde396f9178516d817"
BASE_MODELS = {
    "pangram/editlens_Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
    "pangram/editlens_roberta-large": "FacebookAI/roberta-large",
    QWEN_ID: QWEN_ID,
}

LAYA_ID = "convaiinnovations/laya"
LAYA_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
