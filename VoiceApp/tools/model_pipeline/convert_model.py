"""Download a Whisper model from Hugging Face and convert it for faster-whisper.

    python convert_model.py HF_REPO KEY OUT_DIR [TEST_AUDIO]

Produces OUT_DIR/KEY/ (CTranslate2, int8 weights) containing model.bin,
config.json, vocabulary.json, tokenizer.json, preprocessor_config.json and
voiceapp_model.json, and OUT_DIR/KEY-ct2-int8.zip with KEY/ as the top folder.

If the model uses a non-standard token layout (BuzzASR/persian), the tokens are
renumbered first (remap_vocab.py). With TEST_AUDIO, the converted model's output
is compared with the original model running in transformers on the same audio.
Runs in CI (needs torch + transformers + ctranslate2 + faster-whisper); not part of the app.
"""

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from remap_vocab import needs_remap, remap  # noqa: E402

SPECIAL = ("<|endoftext|>", "<|startoftranscript|>", "<|fa|>", "<|transcribe|>", "<|translate|>",
           "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>", "<|0.00|>", "<|0.02|>")
MAX_ZIP_BYTES = 2_000_000_000


def main(repo: str, key: str, out_dir: str, test_audio: str | None = None) -> int:
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
    remap_info = None
    convert_src = src
    if needs_remap(src / "tokenizer.json"):
        print("Non-standard token layout -> renumbering tokens to the standard Whisper layout")
        convert_src = work / "remapped"
        remap_info = remap(src, convert_src)
        (convert_src / "tokenizer_config.json").write_text(json.dumps({
            "tokenizer_class": "WhisperTokenizer", "processor_class": "WhisperProcessor",
            "bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>",
            "pad_token": "<|endoftext|>", "add_prefix_space": False, "errors": "replace",
            "model_max_length": 448}), encoding="utf-8")
        tokenizer = Tokenizer.from_file(str(convert_src / "tokenizer.json"))
        ids = {t: tokenizer.token_to_id(t) for t in SPECIAL}
        print("new ids:", ids)
        assert tokenizer.decode(tokenizer.encode(test, add_special_tokens=False).ids) == test
        gen_path = convert_src / "generation_config.json"
        gen = json.loads(gen_path.read_text(encoding="utf-8"))
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
    copy = [f for f in ("tokenizer.json", "preprocessor_config.json") if (convert_src / f).exists()]
    TransformersConverter(str(convert_src), copy_files=copy, load_as_float16=True,
                          low_cpu_mem_usage=True).convert(str(target), quantization="int8")
    meta = {
        "key": key,
        "source_repo": repo,
        "source_revision": info.sha,
        "license": getattr(info.card_data, "license", None),
        "quantization": "int8",
        "ctranslate2_version": ctranslate2.__version__,
        "without_timestamps": not timestamps_ok,
        "token_layout_remapped": remap_info is not None,
    }
    if test_audio:
        meta["check_vs_transformers"] = verify(src, target, Path(test_audio))
    (target / "voiceapp_model.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    if test_audio and meta["check_vs_transformers"]["cer"] > 0.15:
        print("ERROR: the converted model does not match the original model's output")
        return 4
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


def verify(original: Path, converted: Path, audio_file: Path, seconds: float = 25.0) -> dict:
    """Transcribe the start of a test file with the original model (transformers, float32)
    and with the converted model (faster-whisper, int8); the texts should match closely."""
    import gc
    import subprocess

    import numpy as np
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    from app.core.text_align import score

    import json

    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(audio_file), "-t", str(seconds), "-ac", "1",
                          "-ar", "16000", "-f", "s16le", "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    processor = AutoProcessor.from_pretrained(str(original))
    model = WhisperForConditionalGeneration.from_pretrained(str(original), torch_dtype=torch.float32).eval()
    gen = json.loads((original / "generation_config.json").read_text(encoding="utf-8"))
    prompt = [gen["decoder_start_token_id"], gen["lang_to_id"]["<|fa|>"], gen["task_to_id"]["transcribe"],
              gen["no_timestamps_token_id"]]
    eot = gen["eos_token_id"]
    features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
    # Plain greedy decoding (no transformers generate(): it rejects some fine-tuned configs).
    with torch.no_grad():
        encoder_out = model.model.encoder(features)
        ids = torch.tensor([prompt])
        past, out = None, []
        for _ in range(220):
            o = model(encoder_outputs=encoder_out, decoder_input_ids=ids if past is None else ids[:, -1:],
                      past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = int(o.logits[0, -1].argmax())
            if nxt == eot:
                break
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
    reference = processor.tokenizer.decode(out, skip_special_tokens=True).strip()
    del model
    gc.collect()

    from faster_whisper import WhisperModel
    fw = WhisperModel(str(converted), device="cpu", compute_type="int8")
    segments, _ = fw.transcribe(audio, language="fa", beam_size=1, vad_filter=False, temperature=0.0,
                                condition_on_previous_text=False, without_timestamps=True)
    converted_text = " ".join(s.text.strip() for s in segments)
    cer = score(reference, converted_text).cer if reference else 1.0
    print("transformers (original):", reference)
    print("faster-whisper (converted):", converted_text)
    print(f"difference (CER): {cer:.1%}")
    return {"cer": round(cer, 4), "transformers": reference, "faster_whisper": converted_text}


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
