"""Renumber the tokens of a Whisper model to the standard Whisper layout.

faster-whisper assumes the original OpenAI token layout:
  * all text tokens have smaller ids than <|endoftext|>
  * the timestamp tokens <|0.00|> ... follow <|notimestamps|> directly
Some fine-tuned models (e.g. BuzzASR/persian) use a new tokenizer where
<|endoftext|> = 0 and <|notimestamps|> = 7. Their output would be empty in
faster-whisper. This script permutes the token embedding / output projection
rows and rewrites tokenizer.json and the config files so the model behaves
exactly the same but uses the standard layout. Needs torch + transformers.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

SPECIAL_RE = re.compile(r"^<\|.*\|>$")
TIMESTAMP_RE = re.compile(r"^<\|(\d+\.\d\d)\|>$")
ORDER_BEFORE_LANG = ["<|endoftext|>", "<|startoftranscript|>"]
ORDER_AFTER_LANG = ["<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nospeech|>",
                    "<|nocaptions|>", "<|notimestamps|>"]


def needs_remap(tokenizer_json: Path) -> bool:
    tok = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    ids = {t["content"]: t["id"] for t in tok.get("added_tokens", [])}
    eot, nots, ts0 = ids.get("<|endoftext|>"), ids.get("<|notimestamps|>"), ids.get("<|0.00|>")
    regular_max = max((i for t, i in tok["model"]["vocab"].items() if not SPECIAL_RE.match(t)), default=0)
    return eot is None or regular_max > eot or ts0 is None or nots is None or ts0 != nots + 1


def build_permutation(tok: dict, vocab_size: int) -> tuple[list[int], dict[int, int], list[int]]:
    """Returns perm (new id -> old id), old->new map, and the new ids of unused padding rows."""
    id_to_token: dict[int, str] = {}
    for token, i in tok["model"]["vocab"].items():
        id_to_token[i] = token
    for t in tok.get("added_tokens", []):
        id_to_token[t["id"]] = t["content"]
    regular = sorted(i for i, t in id_to_token.items() if not SPECIAL_RE.match(t))
    specials = {t: i for i, t in id_to_token.items() if SPECIAL_RE.match(t)}
    timestamps = sorted((float(TIMESTAMP_RE.match(t).group(1)), i) for t, i in specials.items() if TIMESTAMP_RE.match(t))
    named = set(ORDER_BEFORE_LANG + ORDER_AFTER_LANG)
    langs = sorted(i for t, i in specials.items() if t not in named and not TIMESTAMP_RE.match(t)
                   and re.match(r"^<\|[a-z]{2,3}\|>$|^<\|haw\|>$|^<\|yue\|>$", t))
    others = sorted(i for t, i in specials.items() if t not in named and not TIMESTAMP_RE.match(t) and i not in langs)
    perm = list(regular)
    perm += [specials[t] for t in ORDER_BEFORE_LANG if t in specials]
    perm += langs
    perm += [specials[t] for t in ORDER_AFTER_LANG if t in specials]
    perm += [i for _, i in timestamps]
    perm += others
    used = set(perm)
    unused_old = [i for i in range(vocab_size) if i not in used]
    first_unused_new = len(perm)
    perm += unused_old
    assert sorted(perm) == list(range(vocab_size)), "permutation is not complete"
    old_to_new = {old: new for new, old in enumerate(perm)}
    return perm, old_to_new, list(range(first_unused_new, vocab_size))


def _map(value, m):
    if isinstance(value, list):
        return [_map(v, m) for v in value]
    if isinstance(value, int) and not isinstance(value, bool):
        return m.get(value, value)
    return value


def canonical_ids(gen: dict) -> dict[str, int]:
    """Special-token ids the model authors configured for generation (authoritative)."""
    ids = {}
    for token, key in (("<|endoftext|>", "eos_token_id"), ("<|startoftranscript|>", "decoder_start_token_id"),
                       ("<|notimestamps|>", "no_timestamps_token_id"), ("<|startofprev|>", "prev_sot_token_id")):
        if isinstance(gen.get(key), int):
            ids[token] = gen[key]
    for task, i in (gen.get("task_to_id") or {}).items():
        ids[f"<|{task}|>"] = i
    for lang, i in (gen.get("lang_to_id") or {}).items():
        ids[lang] = i
    return ids


def resolve_duplicates(tok: dict, canonical: dict[str, int]) -> list[tuple[str, int]]:
    """A special token may appear twice (in the BPE vocab and as an added token) with
    different ids. Keep the id from generation_config; rename the other one."""
    renamed = []
    vocab = tok["model"]["vocab"]
    for token, i in list(vocab.items()):
        if token in canonical and i != canonical[token]:
            del vocab[token]
            vocab[f"<|dup{i}|>"] = i
            renamed.append((token, i))
    for t in tok.get("added_tokens", []):
        if t["content"] in canonical and t["id"] != canonical[t["content"]]:
            renamed.append((t["content"], t["id"]))
            t["content"] = f"<|dup{t['id']}|>"
    # The canonical token must exist somewhere; add it as an added token if it vanished.
    present = set(vocab) | {t["content"] for t in tok.get("added_tokens", [])}
    for token, i in canonical.items():
        if token not in present:
            tok.setdefault("added_tokens", []).append({"id": i, "content": token, "single_word": False,
                                                       "lstrip": False, "rstrip": False, "normalized": False,
                                                       "special": True})
    return renamed


def remap(src: Path, dst: Path) -> dict:
    import torch
    from transformers import WhisperForConditionalGeneration

    tok = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    gen_raw = json.loads((src / "generation_config.json").read_text(encoding="utf-8"))
    canonical = canonical_ids(gen_raw)
    print("ids from generation_config:", {k: v for k, v in canonical.items() if not re.match(r"^<\|[a-z]{2,3}\|>$", k)
                                          or k == "<|fa|>"})
    renamed = resolve_duplicates(tok, canonical)
    print("duplicate special tokens renamed:", renamed)
    vocab_size = config["vocab_size"]
    perm, m, unused_new = build_permutation(tok, vocab_size)
    dup_new = [m[i] for t, i in list(tok["model"]["vocab"].items()) + [(a["content"], a["id"]) for a in tok["added_tokens"]]
               if t.startswith("<|dup")]

    model = WhisperForConditionalGeneration.from_pretrained(str(src), torch_dtype=torch.float16)
    index = torch.tensor(perm, dtype=torch.long)
    embed = model.model.decoder.embed_tokens
    tied = model.proj_out.weight.data_ptr() == embed.weight.data_ptr()
    with torch.no_grad():
        embed.weight.copy_(embed.weight[index].clone())
        if not tied:
            model.proj_out.weight.copy_(model.proj_out.weight[index].clone())

    # tokenizer.json
    tok["model"]["vocab"] = {t: m[i] for t, i in tok["model"]["vocab"].items()}
    for t in tok.get("added_tokens", []):
        t["id"] = m[t["id"]]
    tok["post_processor"] = None          # only used by transformers for special-token templates

    # configs
    id_keys = ("bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id", "no_timestamps_token_id",
               "prev_sot_token_id", "suppress_tokens", "begin_suppress_tokens", "forced_decoder_ids")
    gen = model.generation_config
    for key in id_keys:
        for obj in (model.config, gen):
            if getattr(obj, key, None) is not None:
                if key == "forced_decoder_ids":
                    setattr(obj, key, [[pos, m.get(i, i)] for pos, i in getattr(obj, key)])
                else:
                    setattr(obj, key, _map(getattr(obj, key), m))
    for key in ("lang_to_id", "task_to_id"):
        if getattr(gen, key, None):
            setattr(gen, key, {k: m[v] for k, v in getattr(gen, key).items()})
    # Never generate the unused padding rows.
    gen.suppress_tokens = sorted(set(gen.suppress_tokens or []) | set(unused_new) | set(dup_new))
    model.config.suppress_tokens = gen.suppress_tokens

    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(dst), safe_serialization=True)
    (dst / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False), encoding="utf-8")
    for name in ("preprocessor_config.json",):
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    ids = {t["content"]: t["id"] for t in tok["added_tokens"]}
    info = {"eot": ids.get("<|endoftext|>"), "notimestamps": ids.get("<|notimestamps|>"),
            "first_timestamp": ids.get("<|0.00|>"), "fa": ids.get("<|fa|>"),
            "timestamps": sum(1 for t in ids if TIMESTAMP_RE.match(t)), "unused_rows": len(unused_new),
            "renamed_duplicates": len(renamed),
            "tied_embeddings": tied}
    print("remapped token layout:", info)
    return info
