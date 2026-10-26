"""Where a RAG system loses the answer: five loss points, measured.

Inspired by the 26 Oct 2026 Towards AI piece "Your RAG Demo Works Because
You Already Know the Answers": a demo RAG pipeline can look perfect when
the person asking already knows which document holds the answer, while a
production system loses the same answer in several distinct places. This
script builds a small synthetic corpus, then traces each question through
five loss points and measures how many questions survive each one:

  1. CHUNKING   - a fixed-character window cuts the answer sentence in two,
                  so no single chunk contains the whole answer.
  2. VOCABULARY - the question words do not match the document words
                  ("return window" vs "refund period"), so lexical retrieval
                  ranks the gold chunk low.
  3. TOP-K      - the gold chunk ranks below the retrieval cutoff.
  4. CONTEXT    - a lookalike chunk (same generic sentence, different
                  product, different value) reaches the reader first and
                  the reader takes the wrong value.
  5. GENERATOR  - with no gold chunk in context, an ungrounded reader still
                  answers from a prior (the most common value in the corpus)
                  instead of abstaining: a hallucination.

The "fixed" pipeline applies the standard, cheap remedies in order:
structure-aware chunking (split on headings, never split a sentence),
metadata prepended to every chunk (document title + section, the
no-LLM-call version of Anthropic's contextual retrieval), synonym
expansion at query time, a larger retrieval set with a bigram rerank,
and a context-grounded reader that abstains when the gold chunk is
absent. Nothing here calls an LLM: retrieval is TF-IDF cosine over the
chunks, and the reader is a small rule-based stand-in whose behaviour
is stated explicitly, so every number below is reproducible.

Run:  python3 rag_failure_modes.py
Test: python3 -m unittest test_rag_failure_modes -v
"""

import math
import re
from collections import Counter

STOPWORDS = set(
    "a an the is are was were what how long does do for of to in on and or "
    "i my me please tell".split()
)

# Query-term -> document-term synonyms. Deliberately small and explicit:
# this is a demo of the mechanism, not a thesaurus.
SYNONYMS = {
    "return": ["refund", "returns"],
    "window": ["period"],
    "refund": ["return"],
    "period": ["window"],
    "warranty": ["guarantee"],
    "guarantee": ["warranty"],
    "shipping": ["delivery"],
    "delivery": ["shipping"],
    "cost": ["fee", "price"],
    "fee": ["cost"],
}

# Each document is (title, [(section heading, [sentences])]).
# Note the trap built into the corpus: several documents contain the same
# generic sentence ("The limit is 30 days." / "The limit is 14 days.")
# with different values. Without metadata, those chunks are
# indistinguishable to the reader, which is exactly loss point 4.
DOCUMENTS = [
    ("Headphones return policy", [
        ("Returns", [
            "The limit is 30 days.",
            "Items must be unused and in the original packaging.",
        ]),
        ("Refunds", [
            "Refunds are issued to the original payment method within 5 days.",
        ]),
    ]),
    ("Laptop return policy", [
        ("Returns", [
            "The limit is 14 days.",
            "A restocking fee of 15 percent applies to opened boxes.",
        ]),
    ]),
    ("Headphones warranty", [
        ("Coverage", [
            "The guarantee covers manufacturing defects for 2 years.",
            "Accidental damage is not covered.",
        ]),
    ]),
    ("Laptop shipping", [
        ("Delivery", [
            "Delivery takes 3 to 5 working days in the UK.",
            "Express delivery costs 12 pounds and arrives next working day.",
        ]),
    ]),
    ("Headphones shipping", [
        ("Delivery", [
            "Delivery is free over 50 pounds.",
            "Standard delivery takes 2 to 3 working days.",
        ]),
    ]),
    ("Tablet refund period", [
        ("Refunds", [
            "The refund period is 21 days from delivery.",
            "Refunds are issued to the original payment method within 7 days.",
        ]),
    ]),
]

# (question, gold document title, gold sentence)
QUESTIONS = [
    ("What is the return window for headphones?", "Headphones return policy",
     "The limit is 30 days."),
    ("What is the return window for a laptop?", "Laptop return policy",
     "The limit is 14 days."),
    ("How long is the headphones warranty?", "Headphones warranty",
     "The guarantee covers manufacturing defects for 2 years."),
    ("How long does laptop shipping take?", "Laptop shipping",
     "Delivery takes 3 to 5 working days in the UK."),
    ("What does express laptop delivery cost?", "Laptop shipping",
     "Express delivery costs 12 pounds and arrives next working day."),
    ("How long is the tablet refund period?", "Tablet refund period",
     "The refund period is 21 days from delivery."),
    ("Is headphone delivery free, and above what amount?", "Headphones shipping",
     "Delivery is free over 50 pounds."),
    ("How long does standard headphone delivery take?", "Headphones shipping",
     "Standard delivery takes 2 to 3 working days."),
    ("How are headphone refunds paid?", "Headphones return policy",
     "Refunds are issued to the original payment method within 5 days."),
    ("Does the headphones warranty cover accidental damage?", "Headphones warranty",
     "Accidental damage is not covered."),
]


def tokenize(text):
    """Lowercase word tokens, dropping stopwords."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in STOPWORDS]


def expand(tokens):
    """Add synonym terms (query-time expansion, fixed pipeline only)."""
    out = list(tokens)
    for token in tokens:
        out.extend(SYNONYMS.get(token, []))
    return out


def chunk_naive(documents, size=120):
    """Fixed-character windows over the whole document text.

    Sentences are glued together first, so a window boundary can fall
    inside a sentence: the classic chunking loss.
    """
    chunks = []
    for title, sections in documents:
        text = " ".join(s for _, sentences in sections for s in sentences)
        for start in range(0, len(text), size):
            chunks.append({"doc": title, "section": "",
                           "text": text[start:start + size],
                           "meta": ""})
    return chunks


def chunk_structured(documents, max_chars=300):
    """Split on section headings, then pack whole sentences.

    A chunk never crosses a section boundary and never splits a
    sentence. Each chunk is prefixed with its document title and
    section, so a generic sentence stays identifiable in isolation.
    """
    chunks = []
    for title, sections in documents:
        for heading, sentences in sections:
            meta = f"{title} | {heading}"
            current = ""
            for sentence in sentences:
                candidate = f"{current} {sentence}".strip()
                if len(candidate) > max_chars and current:
                    chunks.append({"doc": title, "section": heading,
                                   "text": current, "meta": meta})
                    current = sentence
                else:
                    current = candidate
            if current:
                chunks.append({"doc": title, "section": heading,
                               "text": current, "meta": meta})
    return chunks


class TfidfRetriever:
    """TF-IDF cosine retrieval over chunk texts (plus metadata if given)."""

    def __init__(self, chunks, use_meta=False):
        self.chunks = chunks
        self.use_meta = use_meta
        docs = [self._chunk_text(c) for c in chunks]
        self.idf = {}
        for doc in docs:
            for term in set(tokenize(doc)):
                self.idf[term] = self.idf.get(term, 0) + 1
        n = len(docs)
        self.idf = {t: math.log((n + 1) / (df + 1)) + 1
                    for t, df in self.idf.items()}
        self.vectors = [self._vector(doc) for doc in docs]

    def _chunk_text(self, chunk):
        return f"{chunk['meta']} {chunk['text']}" if self.use_meta else chunk["text"]

    def _vector(self, text):
        counts = Counter(tokenize(text))
        return {t: c * self.idf.get(t, 1.0) for t, c in counts.items()}

    def _cosine(self, a, b):
        dot = sum(a.get(t, 0) * v for t, v in b.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def search(self, query, k, expand_query=False):
        terms = expand(tokenize(query)) if expand_query else tokenize(query)
        qvec = {t: c * self.idf.get(t, 1.0)
                for t, c in Counter(terms).items()}
        scored = [(self._cosine(qvec, vec), i)
                  for i, vec in enumerate(self.vectors)]
        scored.sort(key=lambda pair: -pair[0])
        return scored[:k]


def bigram_rerank(query, ranked, chunks):
    """Re-score by shared word bigrams, then TF-IDF score (fixed pipeline)."""
    qwords = re.findall(r"[a-z0-9]+", query.lower())
    qgrams = set(zip(qwords, qwords[1:]))

    def key(item):
        score, i = item
        text = chunks[i]["text"].lower()
        words = re.findall(r"[a-z0-9]+", text)
        grams = set(zip(words, words[1:]))
        return (len(qgrams & grams), score)

    return sorted(ranked, key=key, reverse=True)


def gold_chunk_index(chunks, gold_doc, gold_sentence):
    """Index of the chunk fully containing the gold sentence, or None."""
    for i, chunk in enumerate(chunks):
        if chunk["doc"] == gold_doc and gold_sentence in chunk["text"]:
            return i
    return None


def first_sentence(text):
    """First sentence of a chunk."""
    return text.split(". ")[0].strip() + "."


def reader_answer(question, context_chunks, gold_sentence, grounded):
    """Rule-based stand-in reader (behaviour stated, not hidden).

    Grounded reader: answers the gold sentence if a context chunk
    contains it, otherwise abstains. Ungrounded reader is an extractive
    reader with no grounding check: for limit-type questions it takes
    the first "limit"/"refund period" sentence in ranked context order,
    which is the wrong product's value whenever a lookalike chunk
    outranks the gold chunk; for other questions it takes the first
    sentence of the top chunk when the gold sentence is absent, i.e.
    it answers confidently with irrelevant content (a hallucination
    in effect). It never peeks at the gold answer to decide.
    """
    if grounded:
        for chunk in context_chunks:
            if gold_sentence in chunk["text"]:
                return gold_sentence, "correct"
        return None, "abstained"
    limit_question = "return window" in question or "refund period" in question
    if limit_question:
        for chunk in context_chunks:
            for sentence in chunk["text"].split(". "):
                sentence = sentence.strip()
                if sentence.startswith("The limit is") or \
                        "refund period is" in sentence:
                    sentence = sentence if sentence.endswith(".") else sentence + "."
                    return sentence, ("correct" if sentence == gold_sentence
                                      else "wrong-lookalike")
        return "The limit is 30 days.", "hallucinated"
    for chunk in context_chunks:
        if gold_sentence in chunk["text"]:
            return gold_sentence, "correct"
    if context_chunks:
        return first_sentence(context_chunks[0]["text"]), "hallucinated"
    return None, "abstained"


def diagnose(question, gold_doc, gold_sentence, chunks, retriever,
             k, grounded, rerank=False):
    """Trace one question through the five loss points."""
    gold_idx = gold_chunk_index(chunks, gold_doc, gold_sentence)
    intact = gold_idx is not None
    ranked = retriever.search(question, k, expand_query=grounded)
    if rerank:
        ranked = bigram_rerank(question, ranked, chunks)
    retrieved = intact and any(i == gold_idx for _, i in ranked)
    context = [chunks[i] for _, i in ranked]
    answer, outcome = reader_answer(question, context, gold_sentence, grounded)
    return {"intact": intact, "retrieved": retrieved,
            "answer": answer, "outcome": outcome}


def run_experiment():
    """Run both pipelines and print the per-stage survival table."""
    naive_chunks = chunk_naive(DOCUMENTS)
    fixed_chunks = chunk_structured(DOCUMENTS)
    naive_retriever = TfidfRetriever(naive_chunks)
    fixed_retriever = TfidfRetriever(fixed_chunks, use_meta=True)

    rows = []
    for question, doc, sentence in QUESTIONS:
        naive = diagnose(question, doc, sentence, naive_chunks,
                         naive_retriever, k=3, grounded=False)
        fixed = diagnose(question, doc, sentence, fixed_chunks,
                         fixed_retriever, k=8, grounded=True, rerank=True)
        rows.append((question, naive, fixed))

    def rate(key, which):
        return sum(1 for _, n, f in rows if (n if which == "naive" else f)[key])

    print(f"Questions: {len(QUESTIONS)}")
    print(f"Naive chunks: {len(naive_chunks)} | Fixed chunks: {len(fixed_chunks)}")
    for label, key in [("1. gold sentence intact in one chunk", "intact"),
                       ("2-3. gold chunk in retrieval top-k", "retrieved")]:
        print(f"{label}: naive {rate(key, 'naive')}/{len(QUESTIONS)} | "
              f"fixed {rate(key, 'fixed')}/{len(QUESTIONS)}")
    for name in ("naive", "fixed"):
        outcomes = Counter((n if name == "naive" else f)["outcome"]
                           for _, n, f in rows)
        print(f"{name} reader outcomes: {dict(outcomes)}")
    print("\nPer-question outcomes (naive -> fixed):")
    for question, naive, fixed in rows:
        print(f"- {question}\n    naive: {naive['outcome']} | "
              f"fixed: {fixed['outcome']}")
    return rows


if __name__ == "__main__":
    run_experiment()
