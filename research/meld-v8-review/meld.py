"""Score text with MELD.

    pip install torch transformers safetensors
    python meld.py "Paste the text to check here." ["another text" ...]
    python meld.py --file document.txt

Run it from the folder you downloaded this repository into, or pass
--model-dir. Long inputs are scored in consecutive windows and pooled once, so
a long document is scored the same way a short one is. Inputs are canonicalized
first (look-alike characters folded, invisible characters and markdown removed)
unless --raw is given.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import unicodedata
from functools import lru_cache

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

# ----------------------------------------------------------------- canonical form
_FENCE_LINE = re.compile(r"(?m)^[ \t]*```.*$")
_HEAD_MARK = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+")
_QUOTE_MARK = re.compile(r"(?m)^[ \t]{0,3}>[ \t]?")
_BULLET_MARK = re.compile(r"(?m)^[ \t]{0,3}(?:[-*+]|•|\U0001F539|✅|\U0001F4CC|➡️)[ \t]+")
_NUM_MARK = re.compile(r"(?m)^\d{1,3}[.)][ \t]+")
_TABLE_SEP = re.compile(r"(?m)^[ \t]{0,3}\|?[\s:|-]*-{2,}[\s:|-]*\|?[\s:|-]*$")
_EMPH = re.compile(r"(\*\*\*|\*\*|__)(?=\S)(.+?)(?<=\S)\1", re.S)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\[([^\]\n]*)\]\([^)\n]*\)")
_WS = re.compile(r"[ \t]+")
_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_ADDED = ("Overview", "Key points", "Section", "Point", "Detail", "Topic",
          "Let me know if you'd like more detail.")
_HOMOGLYPH_FOLD = {
    "а": "a", "А": "A", "Α": "A", "В": "B", "Β": "B",
    "е": "e", "Е": "E", "Ε": "E", "с": "c", "р": "p",
    "К": "K", "Κ": "K", "О": "O", "Ο": "O", "Р": "P",
    "Ρ": "P", "М": "M", "Μ": "M", "Н": "H", "Η": "H",
    "Т": "T", "Τ": "T", "Х": "X", "Χ": "X", "С": "C",
    "у": "y", "о": "o", "х": "x", "І": "I", "Ι": "I",
    "і": "i", "Ν": "N", "Ζ": "Z",
}
_INVISIBLE = dict.fromkeys((0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD, 0x180E), None)
_TYPO_FOLD = {
    0x2010: "-", 0x2011: "-", 0x2212: "-",
    0x00A0: " ", 0x202F: " ", 0x2007: " ", 0x2008: " ", 0x2009: " ", 0x200A: " ",
    0x2002: " ", 0x2003: " ", 0x3000: " ",
    0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"', 0x2032: "'", 0x2033: '"',
}
_SURFACE_FOLD = {**_INVISIBLE, **_TYPO_FOLD}
_CONFUSABLE_CHARS = frozenset(_HOMOGLYPH_FOLD)
_TOKEN = re.compile(r"\S+")


@lru_cache(maxsize=8192)
def _blocks_fold(ch: str) -> bool:
    if ch in _CONFUSABLE_CHARS or not ch.isalpha():
        return False
    try:
        return unicodedata.name(ch).startswith("CYRILLIC SMALL ")
    except ValueError:
        return False


def _fold_token(tok: str) -> str:
    if not any(c in _CONFUSABLE_CHARS for c in tok):
        return tok
    if any(_blocks_fold(c) for c in tok):
        return tok
    return "".join(_HOMOGLYPH_FOLD.get(c, c) for c in tok)


def fold_surface(text: str) -> str:
    """Delete invisible characters and fold mixed-script look-alikes. Idempotent."""
    t = str(text or "").translate(_SURFACE_FOLD)
    if not any(c in _CONFUSABLE_CHARS for c in t):
        return t
    return _TOKEN.sub(lambda m: _fold_token(m.group(0)), t)


def canonicalize(text: str) -> str:
    """Strip formatting, leaving a plain paragraph sequence. Idempotent."""
    t = fold_surface(text)
    t = _FENCE_LINE.sub("", t)
    t = _LINK.sub(r"\1", t)
    t = _INLINE_CODE.sub(r"\1", t)
    for _ in range(3):
        new = _EMPH.sub(lambda m: m.group(2), t)
        if new == t:
            break
        t = new
    out = []
    for line in t.split("\n"):
        if _TABLE_SEP.match(line):
            continue
        s = line.strip()
        if s.startswith("|") and len(_CELL_SPLIT.split(s)) >= 3:
            cells = [c.strip().replace(r"\|", "|") for c in _CELL_SPLIT.split(s.strip("|"))]
            cells = [c for c in cells if c and c not in _ADDED]
            line = " ".join(cells)
        for mark in (_HEAD_MARK, _QUOTE_MARK, _BULLET_MARK, _NUM_MARK):
            stripped = mark.sub("", line, count=1)
            if stripped != line:
                line = stripped
                break
        out.append(_WS.sub(" ", line).strip())
    kept = []
    for line in out:
        bare = re.sub(r"\s*\d+\s*$", "", line).strip().rstrip(":")
        if bare in _ADDED:
            continue
        kept.append(line)
    paras, cur = [], []
    for line in kept:
        if line:
            cur.append(line)
        elif cur:
            paras.append(" ".join(cur))
            cur = []
    if cur:
        paras.append(" ".join(cur))
    return "\n\n".join(p for p in paras if p.strip())


# ------------------------------------------------------------------------ model
class Meld(nn.Module):
    def __init__(self, model_dir: str):
        super().__init__()
        self.cfg = json.load(open(os.path.join(model_dir, "meld_config.json")))
        r, H = self.cfg["style_rank"], self.cfg["backbone_hidden_size"]
        self.backbone = AutoModel.from_config(
            AutoConfig.from_pretrained(model_dir), attn_implementation="sdpa")
        self.style_proj = nn.Linear(H, r, bias=False)
        self.style_ln = nn.LayerNorm(r)
        self.human_anchors = nn.Parameter(torch.zeros(self.cfg["n_human_anchors"], r))
        self.family_protos = nn.Parameter(torch.zeros(self.cfg["n_families"], r))
        self.family_bias = nn.Parameter(torch.zeros(self.cfg["n_families"]))
        self.log_tau = nn.Parameter(torch.zeros(()))
        self.op_protos = nn.Parameter(torch.zeros(self.cfg["n_ops"], r))
        self.op_bias = nn.Parameter(torch.zeros(self.cfg["n_ops"]))
        self.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
        self.eval()

    @staticmethod
    def _sqdist(u, p):
        return ((u * u).sum(-1, keepdim=True) - 2.0 * u @ p.t() + (p * p).sum(-1).view(1, 1, -1))

    @torch.no_grad()
    def token_scores(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()
        u = self.style_ln(self.style_proj(h))
        tau = self.log_tau.clamp(-4.0, 4.0).exp()
        logit_h = -tau * self._sqdist(u, self.human_anchors)
        logit_g = -tau * self._sqdist(u, self.family_protos) + self.family_bias.view(1, 1, -1)
        if self.cfg.get("null_reduction", "lse") == "max":
            human = logit_h.max(dim=-1, keepdim=True).values
        else:
            human = torch.logsumexp(logit_h, dim=-1, keepdim=True)
        s_tok = (logit_g - human).clamp(-30.0, 30.0)                         # (B, L, G)
        c_tok = self.cfg["tau_agg"] * torch.logsumexp(s_tok / self.cfg["tau_agg"], dim=-1)
        o_tok = -tau * self._sqdist(u, self.op_protos) + self.op_bias.view(1, 1, -1)
        return s_tok, c_tok, o_tok


def top_quantile_mean(x: torch.Tensor, rho: float) -> torch.Tensor:
    """Mean of the top ceil(rho * n) entries of x along dim 0."""
    k = max(1, math.ceil(x.shape[0] * rho))
    vals, _ = x.sort(dim=0, descending=True)
    return vals[:k].mean(dim=0)


class Scorer:
    def __init__(self, model_dir: str = ".", device: str | None = None, canonical: bool = True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = Meld(model_dir).to(self.device)
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.cfg = self.model.cfg
        self.canonical = canonical
        self.window = int(self.cfg["max_length"]) - 2
        self.cap = int(self.cfg.get("cap_tokens", 16384))
        self.rho = float(self.cfg["rho"])
        self.threshold = float(self.cfg["score_offsets"]["overall"]["fpr_0.01"])
        cls = self.tok.cls_token_id if self.tok.cls_token_id is not None else self.tok.bos_token_id
        sep = self.tok.sep_token_id if self.tok.sep_token_id is not None else self.tok.eos_token_id
        self.cls_id, self.sep_id = int(cls), int(sep)
        self.pad_id = int(self.tok.pad_token_id if self.tok.pad_token_id is not None else 0)

    @torch.no_grad()
    def score(self, text: str, batch_size: int = 4) -> dict:
        if self.canonical:
            text = canonicalize(text)
        ids = self.tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        truncated = len(ids) > self.cap
        ids = ids[: self.cap]
        if not ids:
            raise ValueError("no scoreable tokens in input")
        windows = [ids[i:i + self.window] for i in range(0, len(ids), self.window)]
        c_parts, s_parts, o_parts = [], [], []
        for b0 in range(0, len(windows), batch_size):
            batch = windows[b0:b0 + batch_size]
            width = max(len(w) for w in batch) + 2
            inp = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
            att = torch.zeros((len(batch), width), dtype=torch.long)
            for j, w in enumerate(batch):
                inp[j, 0] = self.cls_id
                inp[j, 1:1 + len(w)] = torch.tensor(w, dtype=torch.long)
                inp[j, 1 + len(w)] = self.sep_id
                att[j, : len(w) + 2] = 1
            s_tok, c_tok, o_tok = self.model.token_scores(inp.to(self.device), att.to(self.device))
            for j, w in enumerate(batch):
                sl = slice(1, 1 + len(w))                                    # drop CLS/SEP/pad
                c_parts.append(c_tok[j, sl].cpu())
                s_parts.append(s_tok[j, sl].cpu())
                o_parts.append(o_tok[j, sl].cpu())
        c_tok = torch.cat(c_parts)
        s_fam = top_quantile_mean(torch.cat(s_parts), self.rho)
        s_op = top_quantile_mean(torch.cat(o_parts), self.rho)
        s = float(top_quantile_mean(c_tok, self.rho))
        return {
            "score": s,
            "p_ai": 1.0 / (1.0 + math.exp(-s)),
            "flagged": s > self.threshold,
            "threshold_fpr_0.01": self.threshold,
            "family": self.cfg["families"][int(s_fam.argmax())],
            "task": self.cfg["ops"][int(s_op.argmax())],
            "n_tokens": int(c_tok.numel()),
            "truncated": truncated,
        }


def main():
    ap = argparse.ArgumentParser(description="Score text with MELD.")
    ap.add_argument("texts", nargs="*", help="texts to score")
    ap.add_argument("--file", action="append", default=[], help="text file to score (repeatable)")
    ap.add_argument("--model-dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--raw", action="store_true", help="score the text as given, without canonicalization")
    a = ap.parse_args()
    inputs = list(a.texts) + [open(f, encoding="utf-8").read() for f in a.file]
    if not inputs:
        ap.error("give at least one text or --file")
    scorer = Scorer(a.model_dir, canonical=not a.raw)
    for text in inputs:
        print(json.dumps(scorer.score(text)))


if __name__ == "__main__":
    sys.exit(main())
