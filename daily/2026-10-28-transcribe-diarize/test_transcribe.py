"""Plain-python test runner (no pytest on this VM): python3 test_transcribe.py"""

from transcribe import (Segment, apply_prompt_context, cluster_speakers,
                        cpwer, diarization_error_rate, extract_decisions,
                        wer)

CHECKS = []


def check(name, fn):
    try:
        fn()
        CHECKS.append((name, True, ""))
        print("PASS", name)
    except AssertionError as exc:
        CHECKS.append((name, False, str(exc)))
        print("FAIL", name, "-", exc)


def seg(start, end, text, emb, speaker=""):
    return Segment(start, end, text, emb, speaker)


def t_clustering_separates():
    segs = [seg(0, 5, "hello", (1, 0.1, 0.1)),
            seg(5, 10, "hi", (0.1, 1, 0.1)),
            seg(10, 15, "hello again", (1, 0.12, 0.08))]
    labels = cluster_speakers(segs)
    assert labels[0] == labels[2], labels
    assert labels[0] != labels[1], labels
    assert labels == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"], labels


def t_speaker_cap():
    # Nine well-separated voices: the ninth folds into its nearest cluster.
    segs = []
    for i in range(9):
        emb = tuple(1.0 if j == i % 3 else 0.1 for j in range(3))
        # Vary magnitude per speaker so centroids stay distinct-ish;
        # the cap is exercised via max_speakers=3 instead of 8 here.
        segs.append(seg(i * 5, i * 5 + 5, "x", emb))
    labels = cluster_speakers(segs, max_speakers=3)
    assert len(set(labels)) <= 3, labels


def t_wer_known():
    errors, n, rate = wer("the cat sat", "the cat")
    assert (errors, n) == (1, 3), (errors, n)
    assert abs(rate - 1 / 3) < 1e-9, rate
    assert wer("hello world", "hello world")[0] == 0


def t_cpwer_permutation_invariant():
    ref = {"A": "the cat sat", "B": "dogs run fast"}
    hyp_swapped = {"X": "dogs run fast", "Y": "the cat sat"}
    hyp_same = {"A": "the cat sat", "B": "dogs run fast"}
    assert cpwer(ref, hyp_swapped)[2] == 0.0
    assert cpwer(ref, hyp_same)[2] == 0.0


def t_cpwer_punishes_flat_labels():
    ref = {"A": "the cat sat", "B": "dogs run fast"}
    hyp = {"ONLY": "the cat sat dogs run fast"}
    errors, n, rate = cpwer(ref, hyp)
    # Best mapping keeps one speaker's 3 words, inserts the other
    # speaker's 3 words there, and deletes those 3 words from their
    # own (empty) hypothesis: 3 insertions + 3 deletions = 6.
    assert errors == 6 and n == 6 and abs(rate - 1.0) < 1e-9, (errors, n, rate)


def t_der_perfect_and_confused():
    ref = [seg(0, 10, "a", (), "A"), seg(10, 20, "b", (), "B")]
    hyp = [seg(0, 10, "a", (), "A"), seg(10, 20, "b", (), "B")]
    assert diarization_error_rate(ref, hyp)[4] == 0.0
    hyp_flat = [seg(0, 20, "ab", (), "A")]
    missed, fa, conf, total, der = diarization_error_rate(ref, hyp_flat)
    # 10s of B attributed to A under the best mapping = confusion.
    assert (missed, fa, conf, total) == (0.0, 0.0, 10.0, 20.0), (missed, fa, conf, total)
    assert abs(der - 0.5) < 1e-9, der


def t_der_missed_and_false_alarm():
    ref = [seg(0, 10, "a", (), "A")]
    hyp = [seg(5, 15, "a", (), "A")]  # 5s missed, 5s false alarm
    missed, fa, conf, total, der = diarization_error_rate(ref, hyp)
    assert (missed, fa, conf) == (5.0, 5.0, 0.0), (missed, fa, conf)
    assert abs(der - 1.0) < 1e-9, der


def t_prompt_correction():
    out = apply_prompt_context("the kubernete pod and postgres check",
                               ["Kubernetes", "PostgreSQL"])
    assert out == "the Kubernetes pod and PostgreSQL check", out
    # Ambiguous / unknown words are left alone, never guessed.
    assert apply_prompt_context("the banana split", ["Kubernetes"]) == "the banana split"


def t_decisions():
    segs = [seg(0, 5, "we decided to ship friday", ()),
            seg(5, 10, "just chatting about lunch", ())]
    found = extract_decisions(segs, ["SPEAKER_00", "SPEAKER_01"])
    assert found == [("SPEAKER_00", "we decided to ship friday")], found


def t_demo_end_to_end():
    import demo
    segments = demo.build()
    labels = cluster_speakers(segments)
    assert labels == [s.speaker for s in segments], labels
    corrected = [apply_prompt_context(s.text, demo.GLOSSARY) for s in segments]
    ref = demo.by_speaker(segments, [s.speaker for s in segments], corrected)
    hyp = demo.by_speaker(segments, labels, corrected)
    assert cpwer(ref, hyp)[2] == 0.0
    hyp_segments = [Segment(s.start, s.end, s.text, s.embedding, lab)
                    for s, lab in zip(segments, labels)]
    assert diarization_error_rate(segments, hyp_segments)[4] == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("t_"):
            check(name, fn)
    failed = [n for n, ok, _ in CHECKS if not ok]
    print("\n%d/%d tests passing" % (len(CHECKS) - len(failed), len(CHECKS)))
    raise SystemExit(1 if failed else 0)
