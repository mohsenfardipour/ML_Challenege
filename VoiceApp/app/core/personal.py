"""Personalization without training: your samples, vocabulary and correction rules.

Everything lives in D:\\VoiceApp\\personal:
    samples\\<id>\\audio.<ext>        copy of your voice file (the original is never touched)
    samples\\<id>\\transcript.txt     the correct text you typed
    samples\\<id>\\meta.json          source file name, date
    samples\\<id>\\model_<key>.txt    what a model produced (cached, used for suggestions)
    vocabulary.txt                   your special words, one per line, most important first
    rules.json                       correction rules  wrong -> right  (approved or not)

How it is used during transcription:
  * vocabulary  -> passed to faster-whisper as "hotwords" (a short text the model sees
                   before every 30-second window), limited to a token budget
  * rules       -> applied to the text afterwards, whole words only
"""

from __future__ import annotations

import json
import re
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from app.core.persian_text import ZWNJ, clean_transcript, normalize_chars
from app.core.text_align import changed_runs, words_model_missed

# Kept short on purpose: in tests a long hint (80 tokens) made the fine-tuned model skip
# whole sentences. faster-whisper's own limit is 223 tokens.
HOTWORDS_TOKEN_BUDGET = 40

# Very common Persian words. They are never suggested as vocabulary or as the
# "wrong" side of a rule, because a rule on them would damage normal sentences.
STOPWORDS = set("""
و در به از که این را با است برای آن یک تا می هم بر شد شده بود نیز اما یا اگر هر
ما من تو او شما آنها ایشان خود چه چون پس باید کرد کند کنند کنم کنی کرده کردن داشت دارد
دارم داری دارند داشتم بودم هست هستم هستی هستند نیست نیستم بی همه هیچ دیگر خیلی ولی
روی زیر پیش بعد قبل وقتی چند نه بله آره خب حالا الان امروز فردا دیروز اینجا آنجا
باشد باشه بشه شود شدن کن بکن گفت گفتم میگه گفته رو اون این‌ها آن‌ها اینو اونو چی کی
کجا چرا چطور چقدر همین همان دو سه چهار پنج ده صد هزار جا کار وقت روز سال بار
""".split())


@dataclass
class Sample:
    id: str
    folder: Path
    audio: Path
    transcript: str
    source_name: str
    added: str

    def model_output(self, model_key: str) -> str | None:
        path = self.folder / f"model_{model_key}.txt"
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def save_model_output(self, model_key: str, text: str) -> None:
        (self.folder / f"model_{model_key}.txt").write_text(text, encoding="utf-8")


@dataclass
class Rule:
    wrong: str
    right: str
    approved: bool = False
    count: int = 1                 # how many times it was seen in your samples
    safe: bool = True              # False: the "wrong" text also appears in your correct transcripts
    source: str = "suggested"      # "suggested" or "manual"


@dataclass
class Personalization:
    """What the transcriber needs: hotwords text and approved rules."""
    hotwords: str | None = None
    rules: list[Rule] = field(default_factory=list)

    def apply(self, text: str) -> str:
        return clean_transcript(apply_rules(text, self.rules))


# ---------------------------------------------------------------- rules
def _phrase_pattern(phrase: str) -> str:
    # Words inside a phrase may be separated by a space or a half-space.
    parts = [re.escape(p) for p in re.split(r"[ \u200c]+", phrase.strip()) if p]
    return r"[ \u200c]+".join(parts)


def apply_rules(text: str, rules: list[Rule]) -> str:
    """Replace whole words/phrases only (never inside a longer word). Approved rules only."""
    active = {normalize_chars(r.wrong).strip(): r.right for r in rules if r.approved and r.wrong.strip()}
    if not active:
        return text
    keys = sorted(active, key=len, reverse=True)          # longer phrases first
    lookup = {re.sub(r"[ \u200c]+", " ", k): v for k, v in active.items()}
    pattern = re.compile(r"(?<![\w\u200c])(" + "|".join(_phrase_pattern(k) for k in keys) + r")(?![\w\u200c])")
    return pattern.sub(lambda m: lookup.get(re.sub(r"[ \u200c]+", " ", m.group(1)), m.group(1)),
                       normalize_chars(text))


def suggest_rules(pairs: list[tuple[str, str]], existing: list[Rule] | None = None) -> list[Rule]:
    """pairs: (correct transcript, model transcript). Returns merged rule list.

    Existing rules keep their approval state. A suggestion is marked unsafe when
    its "wrong" side is a common word or appears in any of your correct
    transcripts (replacing it would then break correct text).
    """
    counts: Counter = Counter()
    for reference, hypothesis in pairs:
        for wrong, right in changed_runs(clean_transcript(reference), clean_transcript(hypothesis)):
            if _same_ignoring_spacing(wrong, right):
                continue      # only a space/half-space difference: handled by normalization
            counts[(wrong, right)] += 1
    references = " \n ".join(normalize_chars(r) for r, _ in pairs)
    by_wrong: dict[str, Rule] = {r.wrong: r for r in (existing or [])}
    for (wrong, right), n in counts.most_common():
        if wrong in by_wrong:
            rule = by_wrong[wrong]
            if rule.source == "suggested" and rule.right == right:
                rule.count = max(rule.count, n)
            continue
        safe = (wrong not in STOPWORDS and len(wrong.replace(" ", "")) >= 3
                and not re.search(r"(?<![\w\u200c])" + _phrase_pattern(wrong) + r"(?![\w\u200c])", references))
        by_wrong[wrong] = Rule(wrong, right, approved=False, count=n, safe=safe)
    return sorted(by_wrong.values(), key=lambda r: (not r.approved, not r.safe, -r.count, r.wrong))


def _same_ignoring_spacing(a: str, b: str) -> bool:
    return a.replace(" ", "").replace(ZWNJ, "") == b.replace(" ", "").replace(ZWNJ, "")


# ---------------------------------------------------------------- vocabulary
_WORD = re.compile(r"[\w\u200c]+")


def suggest_vocabulary(pairs: list[tuple[str, str | None]], limit: int = 60) -> list[str]:
    """Rank words from your correct transcripts: words the model got wrong first,
    then words that are frequent in your messages but not common Persian words."""
    freq: Counter = Counter()
    missed: Counter = Counter()
    for reference, hypothesis in pairs:
        ref = clean_transcript(reference)
        for w in _WORD.findall(ref):
            if len(w) >= 3 and w not in STOPWORDS and not w.isdigit():
                freq[w] += 1
        if hypothesis:
            for w in words_model_missed(ref, clean_transcript(hypothesis)):
                w = w.strip()
                if len(w) >= 3 and w not in STOPWORDS:
                    missed[w] += 1
    scores = {w: missed[w] * 3 + freq[w] for w in set(freq) | set(missed)}
    ranked = sorted(scores, key=lambda w: (-scores[w], -missed[w], w))
    # Words the model got wrong are the useful hints. A word it always got right
    # is only kept if it is frequent in your messages (3+ times).
    ranked = [w for w in ranked if missed[w] or (not any(h for _, h in pairs) and freq[w] >= 2) or freq[w] >= 3]
    return ranked[:limit]


def build_hotwords(words: list[str], count_tokens: Callable[[str], int] | None = None,
                   budget: int = HOTWORDS_TOKEN_BUDGET) -> str | None:
    """Join the most important words until the token budget is reached."""
    count_tokens = count_tokens or (lambda s: len(s) // 2 + 1)   # rough guess if no tokenizer
    chosen: list[str] = []
    for word in words:
        word = word.strip()
        if not word or word.startswith("#"):
            continue
        candidate = " ".join(chosen + [word])
        if count_tokens(candidate) > budget:
            break
        chosen.append(word)
    return " ".join(chosen) or None


# ---------------------------------------------------------------- storage
class PersonalStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.samples_dir = self.root / "samples"

    # samples
    def list_samples(self) -> list[Sample]:
        samples = []
        if not self.samples_dir.is_dir():
            return samples
        for folder in sorted(self.samples_dir.iterdir()):
            sample = self._load_sample(folder)
            if sample:
                samples.append(sample)
        return samples

    def _load_sample(self, folder: Path) -> Sample | None:
        try:
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
            audio = folder / meta["audio"]
            transcript = (folder / "transcript.txt").read_text(encoding="utf-8")
            return Sample(folder.name, folder, audio, transcript, meta.get("source_name", ""), meta.get("added", ""))
        except (OSError, ValueError, KeyError):
            return None

    def add_sample(self, audio_source: Path, transcript: str) -> Sample:
        audio_source = Path(audio_source)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        base = time.strftime("%Y%m%d-%H%M%S")
        sample_id, n = base, 1
        while (self.samples_dir / sample_id).exists():
            sample_id = f"{base}-{n}"
            n += 1
        folder = self.samples_dir / sample_id
        folder.mkdir()
        audio_name = "audio" + audio_source.suffix.lower()
        shutil.copy2(audio_source, folder / audio_name)          # copy: the original stays untouched
        (folder / "transcript.txt").write_text(transcript.strip() + "\n", encoding="utf-8")
        meta = {"audio": audio_name, "source_name": audio_source.name, "added": time.strftime("%Y-%m-%d %H:%M")}
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return self._load_sample(folder)

    def update_transcript(self, sample: Sample, transcript: str) -> None:
        (sample.folder / "transcript.txt").write_text(transcript.strip() + "\n", encoding="utf-8")
        sample.transcript = transcript

    def delete_sample(self, sample: Sample) -> None:
        if sample.folder.parent == self.samples_dir:      # safety: only delete inside samples/
            shutil.rmtree(sample.folder)

    # vocabulary
    @property
    def vocabulary_file(self) -> Path:
        return self.root / "vocabulary.txt"

    def load_vocabulary(self) -> list[str]:
        try:
            lines = self.vocabulary_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]

    def save_vocabulary(self, words: list[str]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        header = "# One word or name per line. Most important first. Lines starting with # are ignored.\n"
        self.vocabulary_file.write_text(header + "\n".join(w.strip() for w in words if w.strip()) + "\n",
                                        encoding="utf-8")

    # rules
    @property
    def rules_file(self) -> Path:
        return self.root / "rules.json"

    def load_rules(self) -> list[Rule]:
        try:
            raw = json.loads(self.rules_file.read_text(encoding="utf-8"))
            return [Rule(**{k: v for k, v in r.items() if k in Rule.__dataclass_fields__}) for r in raw]
        except (OSError, ValueError, TypeError):
            return []

    def save_rules(self, rules: list[Rule]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.rules_file.write_text(json.dumps([asdict(r) for r in rules], ensure_ascii=False, indent=1),
                                   encoding="utf-8")

    # combined
    def personalization(self, count_tokens: Callable[[str], int] | None = None) -> Personalization:
        return Personalization(build_hotwords(self.load_vocabulary(), count_tokens),
                               [r for r in self.load_rules() if r.approved])
