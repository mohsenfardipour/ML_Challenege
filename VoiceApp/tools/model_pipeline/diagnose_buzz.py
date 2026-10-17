"""Find which token layout BuzzASR/persian was really trained with.

Runs the original model (transformers, float32, greedy) on the start of a Persian
test file with different decoder prompts and scores each output against the
reference text. Prints raw token ids so the layout can be read off.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from transformers import AutoProcessor, AutoTokenizer, WhisperForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.core.text_align import score  # noqa: E402

audio_file, ref_file = sys.argv[1], sys.argv[2]
src = Path(snapshot_download("BuzzASR/persian", allow_patterns=["*.json", "*.txt", "*.safetensors"]))
gen = json.loads((src / "generation_config.json").read_text())
print("generation_config:", {k: v for k, v in gen.items() if k not in ("suppress_tokens", "lang_to_id")})
tok = AutoTokenizer.from_pretrained(str(src), use_fast=True)
tok.save_pretrained("/tmp/buzztok")
fast = Tokenizer.from_file("/tmp/buzztok/tokenizer.json")
vocab = json.loads((src / "vocab.json").read_text())
added = json.loads((src / "added_tokens.json").read_text()) if (src / "added_tokens.json").exists() else {}
for t in ("<|endoftext|>", "<|startoftranscript|>", "<|fa|>", "<|transcribe|>", "<|notimestamps|>", "<|0.00|>"):
    print(f"{t:24s} vocab.json={vocab.get(t)} added_tokens.json={added.get(t)} fast={fast.token_to_id(t)} "
          f"slow={tok.convert_tokens_to_ids(t)}")
print("vocab.json size", len(vocab), "min/max id", min(vocab.values()), max(vocab.values()))
print("added_tokens.json size", len(added), "ids", min(added.values(), default=None), max(added.values(), default=None))
print("id->token samples:", {i: fast.id_to_token(i) for i in (0, 1, 7, 8, 107, 108, 1608, 1609, 1610, 5000, 51864, 51865, 51866, 51906, 51908, 51964)})

raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", audio_file, "-t", "20", "-ac", "1", "-ar", "16000",
                      "-f", "s16le", "-"], capture_output=True, check=True).stdout
audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
reference = Path(ref_file).read_text(encoding="utf-8").splitlines()[0]
processor = AutoProcessor.from_pretrained(str(src))
model = WhisperForConditionalGeneration.from_pretrained(str(src), torch_dtype=torch.float32).eval()
feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
enc = model.model.encoder(feats)

fa_vocab = vocab.get("<|fa|>")
variants = {
    "tokenizer ids": [fast.token_to_id("<|startoftranscript|>"), fast.token_to_id("<|fa|>"),
                      fast.token_to_id("<|transcribe|>"), fast.token_to_id("<|notimestamps|>")],
    "vocab.json ids": [vocab.get("<|startoftranscript|>"), fa_vocab, vocab.get("<|transcribe|>"),
                       vocab.get("<|notimestamps|>")],
    "generation_config ids": [gen["decoder_start_token_id"], gen["lang_to_id"]["<|fa|>"],
                              gen["task_to_id"]["transcribe"], gen["no_timestamps_token_id"]],
}
for name, prompt in variants.items():
    if None in prompt:
        print(name, "skipped (missing id)", prompt)
        continue
    with torch.no_grad():
        ids = torch.tensor([prompt])
        past, out = None, []
        for _ in range(120):
            o = model(encoder_outputs=enc, decoder_input_ids=ids if past is None else ids[:, -1:],
                      past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = int(o.logits[0, -1].argmax())
            out.append(nxt)
            if nxt in (gen["eos_token_id"], fast.token_to_id("<|endoftext|>")) and len(out) > 1:
                break
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    text = fast.decode(out, skip_special_tokens=True)
    print(f"== {name}: prompt {prompt}")
    print("   first ids:", out[:25], "last:", out[-3:])
    print("   text:", text[:300])
    print(f"   CER vs first reference sentence: {score(reference, text).cer:.1%}")
