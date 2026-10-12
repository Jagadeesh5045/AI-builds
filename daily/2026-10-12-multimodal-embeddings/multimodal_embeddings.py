"""Toy proxy demonstrating the ideas behind Google's EmbeddingGemma 2
(released 6 October 2026): one embedding model mapping text, code, images,
video and audio into a SINGLE shared vector space, plus Matryoshka-style
truncation so one 768-d vector can be cut to 512/256/128 dims at query time
(Google reports up to 6x less local vector storage).

THIS IS NOT EmbeddingGemma 2. The real model has 740M parameters and was
trained on paired multimodal data; it cannot run in this environment. This
stdlib-only proxy demonstrates the *mechanisms* on synthetic features:

* shared-space alignment: every modality is hashed into the same 768-d
  space from shared concept keywords, plus per-modality noise, so a text
  query can retrieve a matching image or audio clip by cosine similarity;
* truncation: a stored 768-d vector serves every truncation level without
  re-encoding, so retrieval can trade dims for storage.

Real-model recall/benchmark numbers are NOT claimed here; the benchmark
below measures the proxy on its own synthetic corpus.
"""

from __future__ import annotations

import hashlib
import math
import random
import re

FULL_DIMS = 768
TRUNC_LEVELS = (768, 512, 256, 128)
BYTES_PER_FLOAT = 4  # float32 storage, as in typical vector DBs

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*")


def _seed_from_text(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _gaussian_vector(seed: int, dim: int = FULL_DIMS) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in range(dim)]


def _norm(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n > 0 else list(vec)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def word_vector(word: str) -> list[float]:
    """Deterministic pseudo-embedding for one word (proxy for a token embed)."""
    return _gaussian_vector(_seed_from_text("word:" + word.lower()))


def _tokens(text: str) -> list[str]:
    raw = _TOKEN_RE.findall(text.lower())
    out: list[str] = []
    for tok in raw:
        out.extend(tok.split("_"))  # code identifiers share words with prose
    return [t for t in out if t]


class ToyMultimodalEmbedder:
    """Embeds text, code, image descriptors and audio descriptors into one
    shared 768-d unit-norm space. Alignment across modalities comes from
    shared concept keywords; each modality adds its own noise, the way real
    modality encoders inject modality-specific signal on top of shared
    semantics."""

    def __init__(self, noise: float = 1.2):
        self.noise = noise

    # -- internal helpers -------------------------------------------------
    def _word_mean(self, words: list[str]) -> list[float]:
        if not words:
            raise ValueError("nothing to embed: no words/descriptors given")
        vecs = [word_vector(w) for w in words]
        mean = [sum(col) / len(vecs) for col in zip(*vecs)]
        return _norm(mean)

    def _add_noise(self, base: list[float], key: str) -> list[float]:
        # per-input noise scaled so the noise vector has expected norm == self.noise
        rng = random.Random(_seed_from_text("noise:" + key))
        scale = self.noise / math.sqrt(FULL_DIMS)
        noisy = [b + scale * rng.gauss(0.0, 1.0) for b in base]
        return _norm(noisy)

    # -- modality encoders ------------------------------------------------
    # `anchor` words model the shared semantics that paired multimodal
    # training data gives the real model: every modality of one concept is
    # pulled toward the same region of the space. Without an anchor the
    # modalities only share whatever words happen to overlap.
    def embed_text(self, text: str, anchor: list[str] | None = None) -> list[float]:
        toks = _tokens(text) + [a.lower() for a in (anchor or [])]
        if not toks:
            raise ValueError("embed_text got empty text")
        return self._add_noise(self._word_mean(toks), "text:" + text)

    def embed_code(self, code: str, anchor: list[str] | None = None) -> list[float]:
        toks = _tokens(code) + [a.lower() for a in (anchor or [])]
        if not toks:
            raise ValueError("embed_code got empty code")
        return self._add_noise(self._word_mean(toks), "code:" + code)

    def _embed_descriptor(self, descriptor: dict, tag: str,
                          anchor: list[str] | None = None) -> list[float]:
        words: list[str] = []
        for key, value in descriptor.items():
            values = value if isinstance(value, list) else [value]
            for val in values:
                words.append(str(val))
                words.append(str(key))
        words += [a.lower() for a in (anchor or [])]
        if not words:
            raise ValueError(f"embed_{tag} got an empty descriptor")
        flat = "|".join(words)
        return self._add_noise(self._word_mean(words), f"{tag}:{flat}")

    def embed_image(self, descriptor: dict, anchor: list[str] | None = None) -> list[float]:
        """descriptor e.g. {"tags": ["sunset", "beach"], "colors": ["orange"]}"""
        return self._embed_descriptor(descriptor, "image", anchor)

    def embed_audio(self, descriptor: dict, anchor: list[str] | None = None) -> list[float]:
        """descriptor e.g. {"transcript": ["refund", "customer"], "tempo": "slow"}"""
        return self._embed_descriptor(descriptor, "audio", anchor)


def truncate(vec: list[float], dims: int) -> list[float]:
    """Matryoshka-style truncation: keep the first `dims` dims, renormalize.
    Because the prefix is nested, one stored vector serves every level."""
    if dims <= 0 or dims > len(vec):
        raise ValueError(f"cannot truncate {len(vec)}-d vector to {dims} dims")
    return _norm(vec[:dims])


def storage_bytes(n_vectors: int, dims: int, bytes_per: int = BYTES_PER_FLOAT) -> int:
    return n_vectors * dims * bytes_per


class Retriever:
    """Brute-force cosine retriever over a mixed-modality corpus."""

    def __init__(self):
        self._items: list[tuple[str, str, list[float]]] = []

    def add(self, doc_id: str, modality: str, vec: list[float]) -> None:
        self._items.append((doc_id, modality, vec))

    def search(self, query_vec: list[float], k: int = 5,
               dims: int = FULL_DIMS, modality: str | None = None):
        """Returns [(score, doc_id, modality)] sorted by cosine desc."""
        q = truncate(query_vec, dims)
        scored = []
        for doc_id, mod, vec in self._items:
            if modality is not None and mod != modality:
                continue
            scored.append((_dot(q, truncate(vec, dims)), doc_id, mod))
        scored.sort(key=lambda r: r[0], reverse=True)
        return scored[:k]

    def __len__(self):
        return len(self._items)


# -- demo corpus: 10 concepts x 4 modalities --------------------------------
# Each concept shares keywords across its text/code/image/audio items, the
# way paired multimodal training data shares semantics across modalities.
CONCEPTS = [
    {"name": "refund-policy",
     "keywords": ["refund", "policy", "money", "return", "customer"],
     "text": "Our refund policy lets a customer return an item and get their money back within thirty days.",
     "query": "How do I get my money back? What is the return policy for a customer?",
     "code": "def process_refund(order_id):\n    policy = load_refund_policy()\n    return issue_money_return(order_id, policy)",
     "image": {"tags": ["refund", "receipt", "money"], "colors": ["white", "green"]},
     "audio": {"transcript": ["refund", "customer", "money"], "tempo": "calm"}},
    {"name": "churn-prediction",
     "keywords": ["churn", "predict", "customer", "model", "retention"],
     "text": "The churn model predicts which customers are likely to leave so retention offers can target them.",
     "query": "Which customers will churn? I need to predict retention with a model.",
     "code": "def predict_churn(customer):\n    features = build_retention_features(customer)\n    return churn_model.predict(features)",
     "image": {"tags": ["churn", "chart", "customer"], "colors": ["red", "grey"]},
     "audio": {"transcript": ["churn", "predict", "retention"], "tempo": "brisk"}},
    {"name": "sunset-beach",
     "keywords": ["sunset", "beach", "photo", "ocean", "orange"],
     "text": "A photo of an orange sunset over the ocean at the beach.",
     "query": "Show me the beach photo with the orange sunset over the ocean.",
     "code": "def enhance_sunset(photo):\n    warm = boost_orange_tones(photo)\n    return crop_beach_horizon(warm)",
     "image": {"tags": ["sunset", "beach", "ocean"], "colors": ["orange", "blue"]},
     "audio": {"transcript": ["sunset", "beach", "waves"], "tempo": "slow"}},
    {"name": "pandas-groupby",
     "keywords": ["pandas", "groupby", "dataframe", "aggregate", "rows"],
     "text": "Use pandas groupby to group dataframe rows and aggregate each group.",
     "query": "How do I group rows in a pandas dataframe and aggregate them?",
     "code": "summary = df.groupby(\"region\").agg({\"sales\": \"sum\"})  # pandas groupby rows",
     "image": {"tags": ["pandas", "table", "groupby"], "colors": ["white", "blue"]},
     "audio": {"transcript": ["pandas", "groupby", "aggregate"], "tempo": "brisk"}},
    {"name": "invoice-ocr",
     "keywords": ["invoice", "ocr", "scan", "extract", "total"],
     "text": "The OCR pipeline scans each invoice and extracts the total amount.",
     "query": "Extract the total from a scanned invoice with OCR.",
     "code": "def extract_invoice_total(scan):\n    text = ocr_engine.read(scan)\n    return parse_total(text)  # invoice ocr extract",
     "image": {"tags": ["invoice", "scan", "ocr"], "colors": ["white", "black"]},
     "audio": {"transcript": ["invoice", "ocr", "total"], "tempo": "steady"}},
    {"name": "traffic-routing",
     "keywords": ["traffic", "route", "road", "congest", "navigation"],
     "text": "The navigation service reroutes traffic around congested roads.",
     "query": "Reroute me around traffic and congested roads, best route please.",
     "code": "def reroute(origin, dest):\n    roads = fetch_traffic()\n    return fastest_route(origin, dest, roads)  # navigation congest",
     "image": {"tags": ["traffic", "road", "route"], "colors": ["red", "yellow"]},
     "audio": {"transcript": ["traffic", "route", "navigation"], "tempo": "urgent"}},
    {"name": "spam-filter",
     "keywords": ["spam", "filter", "email", "classify", "inbox"],
     "text": "The spam filter classifies each email before it reaches your inbox.",
     "query": "Classify this email, is it spam? Filter my inbox.",
     "code": "def is_spam(email):\n    score = spam_filter.classify(email.text)\n    return score > 0.9  # inbox filter",
     "image": {"tags": ["spam", "email", "filter"], "colors": ["grey", "red"]},
     "audio": {"transcript": ["spam", "filter", "email"], "tempo": "brisk"}},
    {"name": "house-prices",
     "keywords": ["house", "price", "regression", "predict", "bedrooms"],
     "text": "A regression model predicts house prices from bedrooms and location.",
     "query": "Predict the price of a house with four bedrooms using regression.",
     "code": "def predict_house_price(bedrooms, area):\n    return regression.predict([[bedrooms, area]])  # house price",
     "image": {"tags": ["house", "price", "chart"], "colors": ["green", "white"]},
     "audio": {"transcript": ["house", "price", "predict"], "tempo": "steady"}},
    {"name": "playlist-recommend",
     "keywords": ["playlist", "recommend", "music", "song", "taste"],
     "text": "The recommender builds a playlist of songs matching your music taste.",
     "query": "Recommend songs for my playlist based on my music taste.",
     "code": "def recommend_playlist(user):\n    songs = rank_by_taste(user.history)\n    return build_playlist(songs)  # music recommend",
     "image": {"tags": ["playlist", "music", "song"], "colors": ["purple", "black"]},
     "audio": {"transcript": ["playlist", "recommend", "music"], "tempo": "upbeat"}},
    {"name": "fraud-detection",
     "keywords": ["fraud", "detect", "transaction", "alert", "suspicious"],
     "text": "The fraud detector flags suspicious transactions and raises an alert.",
     "query": "Detect fraud in this transaction, it looks suspicious, raise an alert.",
     "code": "def detect_fraud(transaction):\n    if fraud_model.suspicious(transaction):\n        raise_alert(transaction)  # fraud detect",
     "image": {"tags": ["fraud", "alert", "transaction"], "colors": ["red", "black"]},
     "audio": {"transcript": ["fraud", "detect", "alert"], "tempo": "urgent"}},
]


def build_corpus(embedder: ToyMultimodalEmbedder):
    """Returns (retriever, queries): corpus of 40 mixed-modality items and
    one held-out paraphrase query per concept. Targets for a query are the
    3 non-text items of the same concept (pure cross-modal evaluation)."""
    retriever = Retriever()
    queries = []
    for concept in CONCEPTS:
        name = concept["name"]
        anchor = concept["keywords"]
        retriever.add(f"{name}#text", "text",
                      embedder.embed_text(concept["text"], anchor=anchor))
        retriever.add(f"{name}#code", "code",
                      embedder.embed_code(concept["code"], anchor=anchor))
        retriever.add(f"{name}#image", "image",
                      embedder.embed_image(concept["image"], anchor=anchor))
        retriever.add(f"{name}#audio", "audio",
                      embedder.embed_audio(concept["audio"], anchor=anchor))
        queries.append((name, embedder.embed_text(concept["query"], anchor=anchor)))
    return retriever, queries


def evaluate(retriever: Retriever, queries, k: int = 5, dims: int = FULL_DIMS):
    """recall@k and MRR over the queries; targets = same-concept items in a
    different modality than the text query."""
    recalls, rrs = [], []
    for name, qvec in queries:
        targets = {f"{name}#code", f"{name}#image", f"{name}#audio"}
        # exclude the query concept's own text item from candidates
        cands = Retriever()
        for doc_id, mod, vec in retriever._items:
            if doc_id != f"{name}#text":
                cands.add(doc_id, mod, vec)
        hits = cands.search(qvec, k=k, dims=dims)
        hit_ids = [doc_id for _, doc_id, _ in hits]
        found = [i for i, doc_id in enumerate(hit_ids) if doc_id in targets]
        recalls.append(len(found) / len(targets))
        rrs.append(1.0 / (found[0] + 1) if found else 0.0)
    return {
        "dims": dims,
        "recall@%d" % k: sum(recalls) / len(recalls),
        "mrr": sum(rrs) / len(rrs),
        "n_queries": len(queries),
    }
