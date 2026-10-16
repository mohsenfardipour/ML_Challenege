"""Compare converted models on Persian FLEURS speech, with and without personalization.

    python evaluate.py MODELS_DIR TESTAUDIO_DIR REPORT.md [KEY ...]

TESTAUDIO_DIR must contain rec1.ogg/rec1.txt and rec2.ogg/rec2.txt: the same
sentences read by two different speakers (tools/get_persian_test_audio.py).

Personalization test (like a user would do it in the Personalize tab):
  * rec1 + its correct text is added as the personal sample
  * vocabulary and correction rules are learned from it; "safe" rule
    suggestions are approved automatically
  * "same sample": rec1 transcribed again with personalization (upper bound)
  * "new recording": rec2 (other speaker, same words) with personalization
"""

import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.core.personal import Personalization, apply_rules, build_hotwords, suggest_rules, suggest_vocabulary  # noqa: E402
from app.core.persian_text import clean_transcript  # noqa: E402
from app.core.stt_models import find_model_dir, get_model  # noqa: E402
from app.core.text_align import dropped_sentences, score  # noqa: E402
from app.core.transcriber import Transcriber  # noqa: E402

THREADS = 4


def run(models_dir: Path, audio_dir: Path, report: Path, keys: list[str]) -> int:
    rec = {n: (audio_dir / f"{n}.ogg", (audio_dir / f"{n}.txt").read_text(encoding="utf-8")) for n in ("rec1", "rec2")}
    sentences = {n: [s for s in ref.splitlines() if s.strip()] for n, (_, ref) in rec.items()}
    results = {}
    for key in keys:
        model_dir = find_model_dir(models_dir / key)
        if not model_dir:
            print(f"!! {key}: model folder missing, skipped")
            continue
        print(f"== {key}", flush=True)
        try:
            t = Transcriber(model_dir, THREADS)
        except Exception as exc:  # noqa: BLE001
            print(f"!! {key}: failed to load: {exc!r}")
            from app.core.stt_models import model_options
            results[key] = {"error": f"load failed: {exc!r}", "meta": model_options(model_dir)}
            continue
        from app.core.stt_models import model_options
        r = {"load_seconds": t.load_seconds, "runs": {}, "meta": model_options(model_dir)}

        def do(name, audio_key, personalization=None):
            res = t.transcribe(rec[audio_key][0], personalization)
            ref = rec[audio_key][1]
            sc = score(ref, res.text)
            dropped = dropped_sentences(sentences[audio_key], res.text)
            r["runs"][name] = {"wer": sc.wer, "cer": sc.cer, "speed_ratio": res.processing_seconds / res.audio_seconds,
                               "audio_seconds": res.audio_seconds, "dropped": dropped,
                               "sentences": len(sentences[audio_key]), "text": res.text}
            print(f"   {name:28s} WER {sc.wer:6.1%}  CER {sc.cer:6.1%}  speed {r['runs'][name]['speed_ratio']:.2f}  "
                  f"dropped {len(dropped)}/{len(sentences[audio_key])}", flush=True)
            return res

        a = do("rec1 plain", "rec1")
        b = do("rec2 plain", "rec2")
        # learn from rec1, as the Personalize tab does
        ref1 = rec["rec1"][1]
        vocab = suggest_vocabulary([(ref1, a.raw_text)])
        rules = suggest_rules([(ref1, a.raw_text)])
        for rule in rules:
            rule.approved = rule.safe
        hotwords = build_hotwords(vocab, t.count_tokens)
        r["personalization"] = {"vocabulary": vocab, "hotwords": hotwords,
                                "rules": [(x.wrong, x.right, x.approved) for x in rules]}
        rules_only = Personalization(None, [x for x in rules if x.approved])
        full = Personalization(hotwords, rules_only.rules)
        for name, base, ref in (("rec1 + rules (same sample)", a, ref1), ("rec2 + rules (new recording)", b, rec["rec2"][1])):
            text = clean_transcript(apply_rules(base.raw_text, rules_only.rules))
            sc = score(ref, text)
            r["runs"][name] = {"wer": sc.wer, "cer": sc.cer, "text": text}
            print(f"   {name:28s} WER {sc.wer:6.1%}  CER {sc.cer:6.1%}", flush=True)
        for name, audio_key in (("rec1 + vocab + rules (same)", "rec1"), ("rec2 + vocab + rules (new)", "rec2")):
            res = do(name, audio_key, full)
            sc = score(rec[audio_key][1], res.raw_text)          # hints only, before rules
            hint_name = name.replace("vocab + rules", "vocab only")
            r["runs"][hint_name] = {"wer": sc.wer, "cer": sc.cer, "text": res.raw_text,
                                    "dropped": dropped_sentences(sentences[audio_key], res.raw_text)}
            print(f"   {hint_name:28s} WER {sc.wer:6.1%}  CER {sc.cer:6.1%}", flush=True)
        results[key] = r
        del t

    (report.with_suffix(".json")).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    report.write_text(render(results, rec), encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    return 0 if any("runs" in v for v in results.values()) else 1


def render(results: dict, rec: dict) -> str:
    def pct(x):
        return f"{x:.1%}"
    lines = [
        "## Model comparison on Persian speech (FLEURS test set)", "",
        f"Test audio: {len(rec['rec1'][1].splitlines())} read sentences, ~{results_audio(results):.0f} s, "
        "converted to OGG/Opus like a voice message. "
        f"CPU: GitHub runner ({platform.machine()}), int8, cpu_threads={THREADS}. "
        "Lower WER/CER is better; speed ratio = processing time / audio length.", "",
        "| model | WER | CER | dropped sentences | speed ratio | load time |",
        "|---|---|---|---|---|---|",
    ]
    for key, r in results.items():
        label = get_model(key).label if key in {"whisper-large-fa-v1", "buzzasr-persian", "large-v3-turbo"} else key
        if "error" in r:
            lines.append(f"| {label} | failed: {r['error'][:80]} | | | | |")
            continue
        p = r["runs"]["rec1 plain"]
        lines.append(f"| {label} | {pct(p['wer'])} | {pct(p['cer'])} | {len(p['dropped'])}/{p['sentences']} | "
                     f"{p['speed_ratio']:.2f} | {r['load_seconds']:.0f} s |")
    lines += ["", "### Model sources", "", "| file | original model | license | revision |", "|---|---|---|---|"]
    for key, r in results.items():
        m = r.get("meta", {})
        lines.append(f"| {key}-ct2-int8.zip | https://huggingface.co/{m.get('source_repo', '?')} | "
                     f"{m.get('license') or 'see model page'} | {str(m.get('source_revision', ''))[:10]} |")
    lines += ["", "### Personalization (learned from speaker 1, tested on speaker 1 and on speaker 2)", "",
              "| model | test recording | plain | + rules | + vocabulary hints | + both |",
              "|---|---|---|---|---|---|"]
    for key, r in results.items():
        if "runs" not in r:
            continue
        x = r["runs"]
        for label, plain, rules, hints, both in (
                ("speaker 1 (same as learned)", "rec1 plain", "rec1 + rules (same sample)",
                 "rec1 + vocab only (same)", "rec1 + vocab + rules (same)"),
                ("speaker 2 (new recording)", "rec2 plain", "rec2 + rules (new recording)",
                 "rec2 + vocab only (new)", "rec2 + vocab + rules (new)")):
            lines.append(f"| {key} | {label} | {pct(x[plain]['wer'])} | {pct(x[rules]['wer'])} | "
                         f"{pct(x[hints]['wer'])} | {pct(x[both]['wer'])} |")
    lines += ["", "(WER values; the speaker-1 number is an upper bound because the rules were learned from that "
              "same recording.)", ""]
    return "\n".join(lines)


def results_audio(results: dict) -> float:
    for r in results.values():
        if "runs" in r:
            return r["runs"]["rec1 plain"]["audio_seconds"]
    return 0.0


if __name__ == "__main__":
    sys.exit(run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]))
