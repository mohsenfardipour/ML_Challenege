"""Download a Whisper model from Hugging Face and convert it for faster-whisper.

    python convert_model.py HF_REPO KEY OUT_DIR

Produces OUT_DIR/KEY/ (CTranslate2, int8 weights) containing model.bin,
config.json, vocabulary.json, tokenizer.json, preprocessor_config.json and
voiceapp_model.json, and OUT_DIR/KEY-ct2-int8.zip with KEY/ as the top folder.
Runs in CI (needs torch + transformers + ctranslate2); not part of the app.
"""

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

SPECIAL = ("<|endoftext|>", "<|startoftranscript|>", "<|fa|>", "<|transcribe|>", "<|translate|>",
           "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>", "<|0.00|>", "<|0.02|>")
MAX_ZIP_BYTES = 2_000_000_000


def main(repo: str, key: str, out_dir: str) -> int:
    out = Path(out_dir)
    work = Path(tempfile.mkdtemp(prefix="src-"))
    info = HfApi().model_info(repo)
    print(f"== {repo} @ {info.sha}")
    src = Path(snapshot_download(repo, local_dir=str(work / "src"), revision=info.sha,
                                 allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model"]))

    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    gen_path = src / "generation_config.json"
    gen = json.loads(gen_path.read_text(encoding="utf-8")) if gen_path.exists() else {}
    print("config: vocab_size", config.get("vocab_size"), "encoder_layers", config.get("encoder_layers"),
          "decoder_layers", config.get("decoder_layers"), "num_mel_bins", config.get("num_mel_bins"))
    print("generation_config keys:", sorted(gen))

    # ---- tokenizer.json (faster-whisper needs the "fast" tokenizer file)
    from transformers import AutoTokenizer
    if not (src / "tokenizer.json").exists():
        tok = AutoTokenizer.from_pretrained(str(src), use_fast=True)
        tmp = work / "tok"
        tok.save_pretrained(str(tmp))
        shutil.copy(tmp / "tokenizer.json", src / "tokenizer.json")
        print("created tokenizer.json with", type(tok).__name__)

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(src / "tokenizer.json"))
    ids = {t: tokenizer.token_to_id(t) for t in SPECIAL}
    for t, i in ids.items():
        print(f"  {t:24s} {i}")
    size = tokenizer.get_vocab_size(with_added_tokens=True)
    print("tokenizer size", size, "config vocab_size", config.get("vocab_size"))
    test = "مردم از روزگار باستان عناصر شیمیایی را می‌شناختند."
    enc = tokenizer.encode(test, add_special_tokens=False)
    print("encode test:", len(enc.ids), "tokens; decode ok:", tokenizer.decode(enc.ids) == test)
    for needed in ("<|startoftranscript|>", "<|fa|>", "<|transcribe|>", "<|notimestamps|>", "<|endoftext|>"):
        if ids[needed] is None:
            print(f"ERROR: tokenizer has no {needed}; faster-whisper cannot decode this model.")
            return 2
    timestamps_ok = ids["<|0.00|>"] is not None and ids["<|0.00|>"] == ids["<|notimestamps|>"] + 1
    print("timestamp tokens usable:", timestamps_ok)

    # ---- generation_config fixes CTranslate2 needs
    changed = False
    for k in ("suppress_tokens", "begin_suppress_tokens"):
        if gen and gen.get(k) is None:
            gen[k] = config.get(k) or []
            changed = True
    if changed:
        gen_path.write_text(json.dumps(gen, indent=1), encoding="utf-8")
        print("patched generation_config.json (missing suppress token lists)")

    # ---- convert
    import ctranslate2
    from ctranslate2.converters import TransformersConverter
    target = out / key
    if target.exists():
        shutil.rmtree(target)
    copy = [f for f in ("tokenizer.json", "preprocessor_config.json") if (src / f).exists()]
    TransformersConverter(str(src), copy_files=copy, load_as_float16=True,
                          low_cpu_mem_usage=True).convert(str(target), quantization="int8")
    meta = {
        "key": key,
        "source_repo": repo,
        "source_revision": info.sha,
        "license": getattr(info.card_data, "license", None),
        "quantization": "int8",
        "ctranslate2_version": ctranslate2.__version__,
        "without_timestamps": not timestamps_ok,
    }
    (target / "voiceapp_model.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    for f in sorted(target.iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size / 1e6:9.1f} MB")

    zip_path = out / f"{key}-ct2-int8.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as z:
        for f in sorted(target.iterdir()):
            z.write(f, f"{key}/{f.name}")
    print("zip:", zip_path, f"{zip_path.stat().st_size / 1e9:.2f} GB")
    if zip_path.stat().st_size > MAX_ZIP_BYTES:
        print("ERROR: zip is larger than 2 GB")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
