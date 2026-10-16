"""Print what a Hugging Face Whisper repo contains (files, configs, tokenizer special tokens)."""

import json
import sys

from huggingface_hub import HfApi, hf_hub_download

SMALL = ("config.json", "generation_config.json", "tokenizer_config.json", "preprocessor_config.json",
         "special_tokens_map.json", "added_tokens.json", "README.md")
SPECIAL = ("<|startoftranscript|>", "<|fa|>", "<|transcribe|>", "<|notimestamps|>", "<|endoftext|>", "<|0.00|>",
           "<|nocaptions|>", "<|nospeech|>", "<|startofprev|>")


def inspect(repo: str) -> None:
    print("=" * 30, repo)
    info = HfApi().model_info(repo, files_metadata=True)
    for s in info.siblings:
        print(f"  {s.rfilename:60s} {(s.size or 0) / 1e6:10.1f} MB")
    names = {s.rfilename for s in info.siblings}
    for name in SMALL:
        if name in names:
            path = hf_hub_download(repo, name)
            text = open(path, encoding="utf-8").read()
            if name.endswith(".json"):
                data = json.loads(text)
                for k in ("added_tokens_decoder",):
                    if k in data:
                        data[k] = f"<{len(data[k])} entries>"
                if "lang_to_id" in data:
                    data["lang_to_id"] = f"<{len(data['lang_to_id'])} langs, fa={data['lang_to_id'].get('<|fa|>')}>"
                text = json.dumps(data, ensure_ascii=False, indent=1)[:3000]
            else:
                text = text[:2500]
            print(f"--- {name}\n{text}")
    if "tokenizer.json" in names:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        print("--- tokenizer.json vocab size (with added):", tok.get_vocab_size(with_added_tokens=True))
        for t in SPECIAL:
            print(f"  {t:25s} -> {tok.token_to_id(t)}")
        print("  encode 'سلام دنیا':", tok.encode("سلام دنیا", add_special_tokens=False).ids)
    if "vocab.json" in names:
        vocab = json.load(open(hf_hub_download(repo, "vocab.json"), encoding="utf-8"))
        print("--- vocab.json size:", len(vocab))


if __name__ == "__main__":
    for repo in sys.argv[1:]:
        try:
            inspect(repo)
        except Exception as exc:  # noqa: BLE001
            print("FAILED", repo, repr(exc))
