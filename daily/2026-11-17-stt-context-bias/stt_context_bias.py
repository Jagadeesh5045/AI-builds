"""Context-biased speech-to-text decoding, in pure Python (stdlib only).

Trending context (17 Nov 2026): AssemblyAI released STT-3.7, STT-3.7-Realtime
and STT-3.7-Realtime-Agent. The release notes (as reported on the release
tracker, see README) headline 32 languages with automatic detection plus an
optional 99-language fallback, *context biasing* and *natural-language
formatting instructions*. Those two features are the subject of this demo.

What this file implements, from scratch and deterministically:

1.  A toy *lattice* for each utterance: every position offers a few candidate
    tokens, each with an acoustic log-score (higher is better). A plain
    decoder picks the acoustically best token at every position, which is
    exactly how a recogniser mangles rare domain terms ("pipeline pulse"
    becomes "pipe line plus", "systmone" becomes "system one").

2.  *Context biasing* as lattice rescoring: a Viterbi search over the lattice
    adds a bonus ``beta`` once for every completed bias phrase (a unigram,
    bigram or trigram from a caller-supplied list) that appears consecutively
    in the chosen path. This is a small, honest stand-in for the shallow
    fusion / phrase-boosting used by real biased decoders: the bias can only
    *re-rank* candidates the acoustics already proposed, it cannot invent
    tokens that are not in the lattice.

3.  A *bias-strength sweep*: too little bias leaves the domain terms wrong,
    too much bias starts flipping ordinary speech into domain terms (the
    control utterance below, "...power bike...", becomes "...power bi..."
    at high beta). The sweep makes that trade-off measurable in WER.

4.  *Formatting instructions*: a tiny interpreter for plain-English
    instructions such as "capitalise product names" or "use the pound sign
    for amounts in pounds". It applies canonical casing for known product /
    system names and rewrites "<amount> pounds" as "£<amount>".

Everything is seeded / fixed data, so the numbers in the README reproduce
exactly. No audio, no model, no network: the point is the decoding maths,
not a recogniser.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    token: str
    acoustic: float  # log-score, higher is better


@dataclass
class Utterance:
    name: str
    reference: list[str]
    lattice: list[list[Candidate]]  # one candidate list per position
    control: bool = False  # control = bias phrases are NOT actually said


def cand(token: str, acoustic: float) -> Candidate:
    return Candidate(token, acoustic)


# Bias phrases the caller wants the recogniser to prefer. Lower-case token
# tuples; lengths 1-3 are supported by the decoder state.
BIAS_PHRASES: list[tuple[str, ...]] = [
    ("pipeline", "pulse"),
    ("rag", "eval", "lab"),
    ("power", "bi"),
    ("data", "quality"),
    ("systmone",),
    ("duckdb",),
    ("quarantined",),
]

# Canonical casing applied by the formatter for known names.
CANONICAL_CASING = {
    "systmone": "SystmOne",
    "duckdb": "DuckDB",
    "rag": "RAG",
    "bi": "BI",
}


def build_corpus() -> list[Utterance]:
    """Eight fixed utterances. Margins are deliberate:

    - domain terms sit a small acoustic margin (0.15-0.30) below a common
      confusable, so a moderate phrase bonus flips them back;
    - the control utterance's confusable ("bi" for "bike") sits 0.95 below
      the true token, so only an excessive bonus flips it.
    """
    return [
        Utterance(
            "pipeline-report",
            ["pipeline", "pulse", "quarantined", "503", "rows"],
            [
                [cand("pipeline", -0.55), cand("pipe", -0.10)],
                [cand("pulse", -0.50), cand("plus", -0.30)],
                [cand("quarantined", -0.60), cand("quarantine", -0.35)],
                [cand("503", -0.20), cand("5", -0.90)],
                [cand("rows", -0.20), cand("rose", -0.50)],
            ],
        ),
        Utterance(
            "rag-lab",
            ["the", "rag", "eval", "lab", "passed", "every", "check"],
            [
                [cand("the", -0.10), cand("a", -0.80)],
                [cand("rag", -0.55), cand("rack", -0.35)],
                [cand("eval", -0.50), cand("evil", -0.32)],
                [cand("lab", -0.45), cand("lap", -0.30)],
                [cand("passed", -0.20), cand("past", -0.60)],
                [cand("every", -0.15), cand("very", -0.70)],
                [cand("check", -0.20), cand("czech", -0.75)],
            ],
        ),
        Utterance(
            "bi-dashboard",
            ["the", "power", "bi", "dashboard", "uses", "duckdb"],
            [
                [cand("the", -0.10), cand("a", -0.80)],
                [cand("power", -0.25), cand("our", -0.70)],
                [cand("bi", -0.55), cand("buy", -0.35)],
                [cand("dashboard", -0.30), cand("dash", -0.65)],
                [cand("uses", -0.20), cand("use", -0.55)],
                [cand("duckdb", -0.60), cand("duck", -0.38)],
            ],
        ),
        Utterance(
            "nhs-systems",
            ["systmone", "holds", "the", "clinic", "records"],
            [
                [cand("systmone", -0.65), cand("system", -0.30)],
                [cand("holds", -0.25), cand("holes", -0.70)],
                [cand("the", -0.10), cand("a", -0.80)],
                [cand("clinic", -0.30), cand("clinical", -0.62)],
                [cand("records", -0.25), cand("record", -0.55)],
            ],
        ),
        Utterance(
            "data-quality",
            ["data", "quality", "checks", "flagged", "the", "file"],
            [
                [cand("data", -0.20), cand("date", -0.55)],
                [cand("quality", -0.30), cand("qualify", -0.58)],
                [cand("checks", -0.25), cand("cheques", -0.60)],
                [cand("flagged", -0.30), cand("flag", -0.55)],
                [cand("the", -0.10), cand("a", -0.80)],
                [cand("file", -0.20), cand("files", -0.50)],
            ],
        ),
        Utterance(
            "invoice-call",
            ["the", "invoice", "was", "50", "pounds"],
            [
                [cand("the", -0.10), cand("a", -0.80)],
                [cand("invoice", -0.25), cand("in", -0.70)],
                [cand("was", -0.15), cand("is", -0.60)],
                [cand("50", -0.30), cand("15", -0.55)],
                [cand("pounds", -0.20), cand("pound", -0.50)],
            ],
        ),
        Utterance(
            "plain-speech",
            ["we", "walked", "to", "the", "market", "early"],
            [
                [cand("we", -0.15), cand("wee", -0.70)],
                [cand("walked", -0.20), cand("walk", -0.55)],
                [cand("to", -0.10), cand("two", -0.65)],
                [cand("the", -0.10), cand("a", -0.80)],
                [cand("market", -0.25), cand("mark", -0.60)],
                [cand("early", -0.20), cand("earlier", -0.55)],
            ],
        ),
        Utterance(
            "control-power-bike",
            ["she", "rode", "a", "power", "bike", "home"],
            [
                [cand("she", -0.15), cand("see", -0.70)],
                [cand("rode", -0.20), cand("road", -0.55)],
                [cand("a", -0.10), cand("the", -0.75)],
                [cand("power", -0.25), cand("our", -0.70)],
                [cand("bike", -0.20), cand("bi", -1.15)],
                [cand("home", -0.20), cand("holm", -0.65)],
            ],
            control=True,
        ),
    ]


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


def _dedupe(candidates: list[Candidate]) -> list[Candidate]:
    """Keep the best acoustic score per token within one position."""
    best: dict[str, float] = {}
    for c in candidates:
        if c.token not in best or c.acoustic > best[c.token]:
            best[c.token] = c.acoustic
    return [Candidate(t, s) for t, s in best.items()]


def phrases_ending_at(path_tail: tuple[str, ...], phrases: list[tuple[str, ...]]) -> int:
    """How many bias phrases end exactly at the end of ``path_tail``."""
    count = 0
    for phrase in phrases:
        n = len(phrase)
        if n <= len(path_tail) and tuple(path_tail[-n:]) == phrase:
            count += 1
    return count


def decode(utterance: Utterance, beta: float = 0.0,
           phrases: list[tuple[str, ...]] = BIAS_PHRASES) -> list[str]:
    """Viterbi-best token path through the lattice.

    Score = sum of acoustic scores + beta * (completed bias phrases).
    The DP state is the last two chosen tokens, which is enough to detect
    phrase completions for phrases of length <= 3.
    With beta = 0 this reduces to per-position acoustic argmax.
    """
    # state: (last_two_tokens_tuple) -> (score, path)
    states: dict[tuple[str, ...], tuple[float, list[str]]] = {(): (0.0, [])}
    for position in utterance.lattice:
        new_states: dict[tuple[str, ...], tuple[float, list[str]]] = {}
        for cand_ in _dedupe(position):
            for tail, (score, path) in states.items():
                new_path = path + [cand_.token]
                bonus = beta * phrases_ending_at(tuple(new_path[-3:]), phrases)
                new_score = score + cand_.acoustic + bonus
                new_tail = tuple(new_path[-2:])
                if new_tail not in new_states or new_score > new_states[new_tail][0]:
                    new_states[new_tail] = (new_score, new_path)
        states = new_states
    return max(states.values(), key=lambda item: item[0])[1]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    """Levenshtein WER at word level: (S + D + I) / len(reference)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    # dp over words
    prev = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        cur = [i] + [0] * len(hypothesis)
        for j, hyp_word in enumerate(hypothesis, start=1):
            cur[j] = min(
                prev[j] + 1,                       # deletion
                cur[j - 1] + 1,                    # insertion
                prev[j - 1] + (ref_word != hyp_word),  # substitution
            )
        prev = cur
    return prev[-1] / len(reference)


def corpus_wer(corpus: list[Utterance], beta: float) -> tuple[float, list[float]]:
    """Mean WER over the corpus at a given bias strength, plus per-utterance."""
    per = [word_error_rate(u.reference, decode(u, beta)) for u in corpus]
    return sum(per) / len(per), per


def sweep(corpus: list[Utterance],
          betas: list[float]) -> list[tuple[float, float]]:
    return [(b, corpus_wer(corpus, b)[0]) for b in betas]


# --------------------------------------------------------------------------
# Formatting instructions
# --------------------------------------------------------------------------


@dataclass
class Formatter:
    """Applies plain-English formatting instructions to decoded tokens.

    Supported instruction fragments (matched case-insensitively):
      - "capitalise product names" / "canonical casing" -> CANONICAL_CASING
      - "pound sign" / "£" -> "<amount> pounds" becomes "£<amount>"
      - "capitalise the first word" -> sentence-case the first token
    Unknown instructions are recorded in ``ignored`` rather than guessed at.
    """

    instructions: list[str]
    use_casing: bool = field(init=False, default=False)
    use_currency: bool = field(init=False, default=False)
    use_sentence_case: bool = field(init=False, default=False)
    ignored: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        for raw in self.instructions:
            text = raw.lower()
            matched = False
            if "capitalise product" in text or "canonical casing" in text:
                self.use_casing = True
                matched = True
            if "pound sign" in text or "£" in text:
                self.use_currency = True
                matched = True
            if "first word" in text or "sentence" in text:
                self.use_sentence_case = True
                matched = True
            if not matched:
                self.ignored.append(raw)

    def format(self, tokens: list[str]) -> str:
        out: list[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if (self.use_currency and i + 1 < len(tokens)
                    and tokens[i + 1] == "pounds"):
                out.append(f"£{token}")
                i += 2
                continue
            if self.use_casing and token in CANONICAL_CASING:
                token = CANONICAL_CASING[token]
            out.append(token)
            i += 1
        text = " ".join(out)
        if self.use_sentence_case and text:
            text = text[0].upper() + text[1:]
        return text


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------


def main() -> None:
    corpus = build_corpus()
    betas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

    print("Context-biased STT decoding (AssemblyAI STT-3.7 release, 17 Nov 2026)")
    print("=" * 72)
    print("\nBaseline (beta = 0, acoustics only):")
    base_wer, _ = corpus_wer(corpus, 0.0)
    for u in corpus:
        hyp = decode(u, 0.0)
        print(f"  {u.name:<20} ref: {' '.join(u.reference)}")
        print(f"  {'':<20} hyp: {' '.join(hyp)}  WER {word_error_rate(u.reference, hyp):.3f}")

    print(f"\nBaseline mean WER: {base_wer:.4f}")

    print("\nBias-strength sweep (phrase bonus beta per completed bias phrase):")
    results = sweep(corpus, betas)
    for beta, wer in results:
        print(f"  beta {beta:>4}: mean WER {wer:.4f}")
    best_beta, best_wer = min(results, key=lambda item: item[1])
    print(f"Best beta: {best_beta} (mean WER {best_wer:.4f})")

    print(f"\nBiased decoding at beta = {best_beta}:")
    for u in corpus:
        hyp = decode(u, best_beta)
        tag = " [control]" if u.control else ""
        print(f"  {u.name:<20} hyp: {' '.join(hyp)}  "
              f"WER {word_error_rate(u.reference, hyp):.3f}{tag}")

    control = [u for u in corpus if u.control][0]
    print("\nOver-biasing the control utterance:")
    for beta in (best_beta, 2.0):
        print(f"  beta {beta:>4}: {' '.join(decode(control, beta))}")

    fmt = Formatter([
        "Capitalise product names",
        "Use the pound sign for amounts in pounds",
        "Capitalise the first word of each sentence",
        "Make it sound friendlier",  # not a formatting rule: ignored, not guessed
    ])
    print("\nFormatting instructions applied:")
    for name in ("bi-dashboard", "nhs-systems", "invoice-call"):
        u = next(x for x in corpus if x.name == name)
        print(f"  {name:<20} -> {fmt.format(decode(u, best_beta))}")
    print(f"  Ignored instructions (not formatting rules): {fmt.ignored}")


if __name__ == "__main__":
    main()
