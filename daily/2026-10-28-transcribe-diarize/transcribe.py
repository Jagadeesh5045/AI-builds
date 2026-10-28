"""Speaker-attributed transcription pipeline (toy, stdlib only).

Inspired by NVIDIA Nemotron 3.5 Transcribe, released 28 October 2026:
an open-weight model that transcribes in 40 languages, attributes up to
eight speakers, translates 31 languages to/from English, and can be
prompted (for example, to return a speaker-labelled transcript or to
pull out decisions). This file does NOT run that model and does NOT
decode audio. It implements the pipeline around such a model with
synthetic stand-ins, so the diarization, scoring and prompting logic
can be read, tested and measured:

* ``cluster_speakers`` -- greedy speaker clustering over per-segment
  voice embeddings, in the spirit of a diarization front-end such as
  Sortformer. Segments arrive in time order; a segment joins the
  cluster whose centroid is most similar (cosine) if that similarity
  clears a threshold, otherwise it starts a new speaker, capped at
  eight speakers (the Transcribe limit reported for the real model).
* ``wer`` / ``cpwer`` -- word error rate, and concatenated
  minimum-permutation WER (cpWER), the speaker-attributed metric:
  speaker labels are arbitrary, so cpWER tries every mapping from
  hypothesis speakers to reference speakers and keeps the best.
* ``diarization_error_rate`` -- missed speech + false alarms +
  speaker confusion, over reference speech time, again under the best
  label mapping.
* ``apply_prompt_context`` -- promptable vocabulary correction: a
  meeting prompt supplies a glossary (product names, jargon); a
  recognised word within a small edit distance of a glossary term is
  corrected to the glossary spelling. This is the toy analogue of
  prompting a transcription model with domain context.
* ``extract_decisions`` -- the toy analogue of prompting for
  decisions: turns containing decision language are pulled out.

All metrics are defined on the synthetic segments in ``demo.py``.
Numbers printed by the demo are toy numbers, not Nemotron results.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field

MAX_SPEAKERS = 8
PUNCT = ".,;:!?()[]{}\"'"

@dataclass
class Segment:
    """One transcribed turn: time span, text, and a voice embedding."""

    start: float
    end: float
    text: str
    embedding: tuple = ()
    speaker: str = ""  # ground-truth label, when known (demo only)


def cosine(a, b):
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _mean(vectors):
    dim = len(vectors[0])
    return tuple(sum(v[i] for v in vectors) / len(vectors) for i in range(dim))


def cluster_speakers(segments, threshold=0.5, max_speakers=MAX_SPEAKERS):
    """Assign speaker labels in order of first appearance.

    Returns a list of labels such as ``SPEAKER_00`` aligned with
    ``segments``. A segment joins the most similar existing cluster
    when cosine similarity to that cluster centroid is at least
    ``threshold``; otherwise it opens a new cluster, unless the cap
    is reached, in which case it joins its nearest cluster. This
    cap behaviour mirrors a real diarizer's fixed speaker budget:
    an unexpected ninth voice is folded into the closest known voice
    rather than inventing a ninth label.
    """
    centroids = []  # list of (label, [member embeddings])
    labels = []
    for seg in segments:
        if not centroids:
            centroids.append(("SPEAKER_00", [seg.embedding]))
            labels.append("SPEAKER_00")
            continue
        sims = [(cosine(seg.embedding, _mean(members)), label)
                for label, members in centroids]
        best_sim, best_label = max(sims)
        if best_sim >= threshold or len(centroids) >= max_speakers:
            labels.append(best_label)
            for label, members in centroids:
                if label == best_label:
                    members.append(seg.embedding)
        else:
            label = "SPEAKER_%02d" % len(centroids)
            centroids.append((label, [seg.embedding]))
            labels.append(label)
    return labels


def _words(text):
    return [w.strip(PUNCT).lower() for w in text.split() if w.strip(PUNCT)]


def _edit_distance(ref, hyp):
    """Word-level Levenshtein distance (substitutions + ins + dels)."""
    prev = list(range(len(hyp) + 1))
    for i, rw in enumerate(ref, 1):
        cur = [i]
        for j, hw in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (0 if rw == hw else 1)))
        prev = cur
    return prev[-1]


def wer(ref_text, hyp_text):
    """Return (errors, reference_word_count, wer) for two strings."""
    ref, hyp = _words(ref_text), _words(hyp_text)
    errors = _edit_distance(ref, hyp)
    return errors, len(ref), (errors / len(ref) if ref else 0.0)


def _best_mapping(ref_keys, hyp_keys):
    """Yield mappings hyp_key -> ref_key covering every hyp key.

    If there are more hypothesis speakers than reference speakers,
    extra hypothesis speakers map to None (their words are ignored,
    matching the concatenated-permutation scoring convention used
    here); if fewer, some reference speakers get empty hypotheses.
    """
    if len(hyp_keys) <= len(ref_keys):
        for perm in itertools.permutations(ref_keys, len(hyp_keys)):
            yield dict(zip(hyp_keys, perm))
    else:
        for perm in itertools.permutations(hyp_keys, len(ref_keys)):
            mapping = dict(zip(perm, ref_keys))
            for extra in hyp_keys:
                mapping.setdefault(extra, None)
            yield mapping


def cpwer(ref_by_speaker, hyp_by_speaker):
    """Concatenated minimum-permutation WER over speaker texts.

    Returns (errors, reference_word_count, cpwer). Speaker labels
    are arbitrary, so every label mapping is tried and the lowest
    error total wins. This is why a system that transcribes every
    word correctly but attributes them all to one speaker still
    scores badly: words landing under the wrong speaker count as
    insertions there and deletions from their true speaker.
    """
    ref_keys = list(ref_by_speaker)
    hyp_keys = list(hyp_by_speaker)
    total_ref = sum(len(_words(t)) for t in ref_by_speaker.values())
    best = None
    for mapping in _best_mapping(ref_keys, hyp_keys):
        errors = 0
        for ref_key in ref_keys:
            hyp_key = next((h for h, r in mapping.items() if r == ref_key), None)
            hyp_text = hyp_by_speaker.get(hyp_key, "") if hyp_key else ""
            errors += _edit_distance(_words(ref_by_speaker[ref_key]),
                                     _words(hyp_text))
        if best is None or errors < best:
            best = errors
    return best, total_ref, (best / total_ref if total_ref else 0.0)


def _covering(segments, t):
    for seg in segments:
        if seg.start <= t < seg.end:
            return seg.speaker
    return None


def diarization_error_rate(ref_segments, hyp_segments):
    """Return (missed, false_alarm, confusion, total_ref_time, der).

    Time is scored on intervals cut at every segment boundary, so
    overlapping turns are scored by their midpoint speaker. The
    confusion term uses the label mapping that minimises confusion,
    because diarization labels are arbitrary up to permutation.
    """
    bounds = sorted({s.start for s in ref_segments} |
                    {s.end for s in ref_segments} |
                    {s.start for s in hyp_segments} |
                    {s.end for s in hyp_segments})
    ref_keys = sorted({s.speaker for s in ref_segments})
    hyp_keys = sorted({s.speaker for s in hyp_segments})
    total_ref = sum(s.end - s.start for s in ref_segments)
    missed = false_alarm = 0.0
    interval_speakers = []
    for t0, t1 in zip(bounds, bounds[1:]):
        mid = (t0 + t1) / 2.0
        r, h = _covering(ref_segments, mid), _covering(hyp_segments, mid)
        interval_speakers.append((t1 - t0, r, h))
        if r is not None and h is None:
            missed += t1 - t0
        elif r is None and h is not None:
            false_alarm += t1 - t0
    best_confusion = None
    for mapping in _best_mapping(ref_keys, hyp_keys):
        confusion = 0.0
        for dur, r, h in interval_speakers:
            if r is not None and h is not None and mapping.get(h) != r:
                confusion += dur
        if best_confusion is None or confusion < best_confusion:
            best_confusion = confusion
    confusion = best_confusion or 0.0
    der = (missed + false_alarm + confusion) / total_ref if total_ref else 0.0
    return missed, false_alarm, confusion, total_ref, der


def _char_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def apply_prompt_context(text, glossary):
    """Correct near-miss words to glossary spellings (prompt context).

    A word is corrected only when it is not already a glossary term,
    is at least five characters long, and sits within edit distance
    2 of exactly one glossary term (ties are left alone, so ambiguous
    jargon is never guessed). Case of the glossary term is kept.
    """
    lookup = {g.lower(): g for g in glossary}

    def fix(match):
        word = match.group(0)
        low = word.lower()
        if low in lookup:
            return lookup[low]
        candidates = [g for g in glossary
                      if len(word) >= 5 and _char_distance(low, g.lower()) <= 2]
        if len(candidates) == 1:
            return candidates[0]
        return word

    return re.sub(r"[A-Za-z]+", fix, text)


DECISION_MARKERS = ("decided", "agreed", "action item", "we will", "action:")


def extract_decisions(segments, labels):
    """Pull out turns that carry decision language, with speakers."""
    out = []
    for seg, label in zip(segments, labels):
        if any(m in seg.text.lower() for m in DECISION_MARKERS):
            out.append((label, seg.text))
    return out


def format_transcript(segments, labels):
    """Render ``[mm:ss-mm:ss] SPEAKER_XX: text`` lines."""
    lines = []
    for seg, label in zip(segments, labels):
        fmt = lambda s: "%02d:%02d" % (int(s) // 60, int(s) % 60)
        lines.append("[%s-%s] %s: %s" % (fmt(seg.start), fmt(seg.end),
                                         label, seg.text))
    return "\n".join(lines)
