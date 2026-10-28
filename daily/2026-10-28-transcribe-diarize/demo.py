"""Demo: an 8-turn synthetic meeting through the pipeline.

Three speakers with clearly separated voice embeddings take turns.
The raw recogniser output misspells two glossary terms ("kubernete",
"postgres"); a meeting prompt supplying the glossary fixes both.
Ground truth is known, so WER, cpWER and DER can be scored exactly.

Run:  python3 demo.py
"""

from transcribe import (Segment, apply_prompt_context, cluster_speakers,
                        cpwer, diarization_error_rate, extract_decisions,
                        format_transcript, wer)

# Voice anchors: each speaker lives near one axis, with small jitter.
A = (1.00, 0.10, 0.10)
B = (0.10, 1.00, 0.10)
C = (0.10, 0.10, 1.00)

# (start, end, ground-truth speaker, raw recogniser text)
RAW = [
    (0, 6, "SPEAKER_00", "welcome everyone we decided to ship the retrieval update on friday"),
    (6, 12, "SPEAKER_01", "the kubernete rollout needs a postgres check first"),
    (12, 18, "SPEAKER_02", "agreed the search index rebuild is the action item for today"),
    (18, 24, "SPEAKER_00", "we will freeze new features until the rollout is green"),
    (24, 30, "SPEAKER_01", "the database check takes ten minutes and then we deploy"),
    (30, 36, "SPEAKER_02", "i will verify the index counts after the deploy"),
    (36, 42, "SPEAKER_00", "good we decided the release window is friday morning"),
    (42, 48, "SPEAKER_01", "noted i will post the rollout summary afterwards"),
]
EMB = {"SPEAKER_00": A, "SPEAKER_01": B, "SPEAKER_02": C}
GLOSSARY = ["Kubernetes", "PostgreSQL"]


def build():
    segments = []
    for i, (start, end, speaker, text) in enumerate(RAW):
        # Deterministic jitter so no two embeddings are identical.
        base = EMB[speaker]
        jitter = (0.02 * ((i % 3) - 1), 0.01 * ((i % 2) - 0.5), 0.0)
        emb = tuple(b + j for b, j in zip(base, jitter))
        segments.append(Segment(start, end, text, emb, speaker))
    return segments


def by_speaker(segments, labels, texts):
    out = {}
    for seg, label, text in zip(segments, labels, texts):
        out[label] = (out.get(label, "") + " " + text).strip()
    return out


def main():
    segments = build()
    truth_labels = [s.speaker for s in segments]
    labels = cluster_speakers(segments)
    print("clustered labels:", labels)
    print("ground truth     :", truth_labels)
    print("clustering exact (up to the order speakers first appear):",
          labels == truth_labels)

    corrected = [apply_prompt_context(s.text, GLOSSARY) for s in segments]
    raw_text = " ".join(s.text for s in segments)
    clean_text = " ".join(corrected)
    truth_text = clean_text  # ground truth is the corrected wording
    errors, n, rate = wer(truth_text, raw_text)
    print("\nraw WER vs corrected truth: %d errors / %d words = %.4f"
          % (errors, n, rate))
    errors, n, rate = wer(truth_text, clean_text)
    print("prompt-corrected WER      : %d errors / %d words = %.4f"
          % (errors, n, rate))

    ref = by_speaker(segments, truth_labels, corrected)
    hyp_clustered = by_speaker(segments, labels, corrected)
    hyp_flat = by_speaker(segments, ["SPEAKER_00"] * len(segments), corrected)
    for name, hyp in (("clustered", hyp_clustered), ("all-one-speaker", hyp_flat)):
        errors, n, rate = cpwer(ref, hyp)
        print("cpWER %-16s: %d errors / %d words = %.4f" % (name, errors, n, rate))

    hyp_segments = [Segment(s.start, s.end, s.text, s.embedding, lab)
                    for s, lab in zip(segments, labels)]
    flat_segments = [Segment(s.start, s.end, s.text, s.embedding, "SPEAKER_00")
                     for s in segments]
    for name, hyp in (("clustered", hyp_segments), ("all-one-speaker", flat_segments)):
        missed, fa, conf, total, der = diarization_error_rate(segments, hyp)
        print("DER %-16s: missed %.1fs, false alarm %.1fs, confusion %.1fs "
              "/ %.1fs = %.4f" % (name, missed, fa, conf, total, der))

    print("\n--- speaker-labelled transcript (prompt-corrected) ---")
    corrected_segments = [Segment(s.start, s.end, t, s.embedding, s.speaker)
                          for s, t in zip(segments, corrected)]
    print(format_transcript(corrected_segments, labels))
    print("\n--- decisions pulled out by prompt ---")
    for label, text in extract_decisions(corrected_segments, labels):
        print("%s: %s" % (label, text))


if __name__ == "__main__":
    main()
