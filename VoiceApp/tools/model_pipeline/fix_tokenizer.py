"""Repair a Whisper tokenizer whose special-token names disagree with generation_config.

BuzzASR/persian keeps leftover entries at ids 0-1608 that carry special names
(<|endoftext|> = 0 ...), while the model was really trained with the standard
Whisper layout shifted by 1608 (<|endoftext|> = 51865, <|startoftranscript|> = 51866,
<|fa|> = 51908, <|notimestamps|> = 51972, timestamps from 51973), as listed in
added_tokens.json and generation_config.json. The auto-generated tokenizer.json
mixes the two, so faster-whisper would build wrong prompts.

This rebuilds tokenizer.json from vocab.json + merges + added_tokens.json, renames
the leftover entries to <|unusedN|> and suppresses them. Model weights are unchanged.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

SPECIAL_RE = re.compile(r"^<\|.*\|>$")
CHECK = {"<|endoftext|>": "eos_token_id", "<|startoftranscript|>": "decoder_start_token_id",
         "<|notimestamps|>": "no_timestamps_token_id"}


def needs_fix(src: Path) -> bool:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(src / "tokenizer.json"))
    gen = json.loads((src / "generation_config.json").read_text(encoding="utf-8"))
    return any(gen.get(key) is not None and tok.token_to_id(name) != gen[key] for name, key in CHECK.items())


def fix(src: Path, dst: Path) -> dict:
    vocab = json.loads((src / "vocab.json").read_text(encoding="utf-8"))
    added = json.loads((src / "added_tokens.json").read_text(encoding="utf-8"))
    gen = json.loads((src / "generation_config.json").read_text(encoding="utf-8"))
    tok = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))      # keeps merges, normalizer ...

    leftover = sorted(i for t, i in vocab.items() if SPECIAL_RE.match(t))
    tok["model"]["vocab"] = {(f"<|unused{i}|>" if SPECIAL_RE.match(t) else t): i for t, i in vocab.items()}
    tok["added_tokens"] = [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
                            "normalized": False, "special": True}
                           for t, i in sorted(added.items(), key=lambda kv: kv[1])]
    tok["post_processor"] = None

    eot = added["<|endoftext|>"]
    space = vocab.get("Ġ")                         # "Ġ" = a lone space token
    gen["suppress_tokens"] = sorted(set(gen.get("suppress_tokens") or []) | set(leftover))
    gen["begin_suppress_tokens"] = [i for i in (space, eot) if i is not None]
    gen["lang_to_id"] = {t: i for t, i in added.items() if re.match(r"^<\|([a-z]{2,3}|haw|yue)\|>$", t)}

    dst.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "preprocessor_config.json"):
        shutil.copy(src / name, dst / name)
    for weights in src.glob("*.safetensors"):
        target = dst / weights.name
        try:
            os.symlink(weights.resolve(), target)       # 3 GB: link instead of copying
        except OSError:
            shutil.copy(weights, target)
    (dst / "generation_config.json").write_text(json.dumps(gen, indent=1), encoding="utf-8")
    (dst / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False), encoding="utf-8")
    (dst / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "WhisperTokenizer", "processor_class": "WhisperProcessor",
        "bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>", "add_prefix_space": False, "errors": "replace", "model_max_length": 448}),
        encoding="utf-8")
    info = {"eot": eot, "sot": added.get("<|startoftranscript|>"), "fa": added.get("<|fa|>"),
            "notimestamps": added.get("<|notimestamps|>"), "first_timestamp": added.get("<|0.00|>"),
            "leftover_entries_suppressed": len(leftover)}
    print("fixed tokenizer:", info)
    return info


def write_vocabulary(model_dir: Path, size: int) -> None:
    """CTranslate2 looks up special tokens by name in vocabulary.json: write it from the fixed tokenizer."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    names = [tok.id_to_token(i) or f"<|unused{i}|>" for i in range(size)]
    (model_dir / "vocabulary.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
