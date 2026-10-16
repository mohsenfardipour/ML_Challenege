"""Word alignment between a correct transcript and a model transcript.

Used for WER/CER scores, for finding sentences the model skipped, and for
suggesting correction rules (e.g. اناثر -> عناصر).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.persian_text import normalize_chars, normalize_for_scoring


@dataclass
class Op:
    kind: str          # "equal", "sub", "del" (missing in hypothesis), "ins" (extra in hypothesis)
    ref: str | None
    hyp: str | None
    ref_index: int | None = None


def align_words(ref: list[str], hyp: list[str]) -> list[Op]:
    """Levenshtein alignment with backtrace."""
    n, m = len(ref), len(hyp)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        row, prev = dist[i], dist[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + cost)
    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dist[i][j] == dist[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            kind = "equal" if ref[i - 1] == hyp[j - 1] else "sub"
            ops.append(Op(kind, ref[i - 1], hyp[j - 1], i - 1))
            i, j = i - 1, j - 1
        elif i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            ops.append(Op("del", ref[i - 1], None, i - 1))
            i -= 1
        else:
            ops.append(Op("ins", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


@dataclass
class Score:
    wer: float
    cer: float
    ref_words: int


def score(reference: str, hypothesis: str) -> Score:
    ref, hyp = normalize_for_scoring(reference), normalize_for_scoring(hypothesis)
    rw, hw = ref.split(), hyp.split()
    word_errors = sum(op.kind != "equal" for op in align_words(rw, hw))
    rc, hc = ref.replace(" ", ""), hyp.replace(" ", "")
    return Score(word_errors / max(1, len(rw)), _edit_distance(rc, hc) / max(1, len(rc)), len(rw))


def dropped_sentences(reference_sentences: list[str], hypothesis: str, threshold: float = 0.5) -> list[int]:
    """Indexes of reference sentences of which more than `threshold` of the words are missing."""
    ref_words, owner = [], []
    for idx, sentence in enumerate(reference_sentences):
        words = normalize_for_scoring(sentence).split()
        ref_words += words
        owner += [idx] * len(words)
    missing = [0] * len(reference_sentences)
    for op in align_words(ref_words, normalize_for_scoring(hypothesis).split()):
        if op.kind == "del":
            missing[owner[op.ref_index]] += 1
    totals = [owner.count(i) for i in range(len(reference_sentences))]
    return [i for i, (miss, tot) in enumerate(zip(missing, totals)) if tot and miss / tot > threshold]


def _words_for_rules(text: str) -> list[str]:
    # Keep half-spaces inside words, drop punctuation at word edges.
    words = []
    for w in normalize_chars(text).split():
        w = w.strip(".,،؛;:!?؟«»\"'()[]{}…-")
        if w:
            words.append(w)
    return words


def changed_runs(reference: str, hypothesis: str, max_words: int = 3) -> list[tuple[str, str]]:
    """(wrong phrase in hypothesis, correct phrase in reference) for each differing stretch."""
    ops = align_words(_words_for_rules(reference), _words_for_rules(hypothesis))
    runs, cur = [], []
    for op in ops + [Op("equal", None, None)]:
        if op.kind == "equal":
            if cur:
                runs.append(cur)
                cur = []
        else:
            cur.append(op)
    pairs = []
    for run in runs:
        wrong = [op.hyp for op in run if op.hyp]
        right = [op.ref for op in run if op.ref]
        if wrong and right and len(wrong) <= max_words and len(right) <= max_words:
            pairs.append((" ".join(wrong), " ".join(right)))
    return pairs


def words_model_missed(reference: str, hypothesis: str) -> list[str]:
    """Reference words that the model did not reproduce exactly."""
    ops = align_words(_words_for_rules(reference), _words_for_rules(hypothesis))
    return [op.ref for op in ops if op.kind in ("sub", "del") and op.ref]
