"""
agent_context.py

A Headroom-style compression layer for agentic AI workloads, implemented from
scratch in stdlib Python.

Trending context (Oct 2026): production agents hit context limits fast. A coding
agent that runs tests, reads logs and pulls docs can burn 100k tokens in three
turns; RAG pipelines dump whole chunks into the prompt; tool outputs return
verbose JSON. The open-source "Headroom" project claims 20% token
reduction for coding agents and 60-95% for JSON-heavy workflows by compressing
what the agent *reads* (tool outputs, logs, RAG chunks, files) without touching
the user query or the final answer. Anthropic's Haiku 5.5 launch (Oct 7) pushed
context compression and classification into the spotlight.

This module demonstrates the core idea with transparent, deterministic
heuristics instead of a learned compressor:

* LogCompressor   - keeps ERROR/FATAL/WARNING lines byte-for-byte, collapses
                    repeated INFO/DEBUG noise, truncates with a summary.
* JsonCompressor  - trims verbose tool-output JSON while preserving the keys
                    downstream parsers need (ids, status, errors, schema
                    fields). Never "compresses" by reformatting semantics.
* ChunkCompressor - sentence-level RAG chunk filtering: keeps sentences with
                    high query-term overlap and drops boilerplate filler.
* AgentContextCompressor - orchestrates per-type compression under a token
                    budget and reports before/after statistics.

Token counts here are estimated as ceil(chars / 4). In production you would use
tiktoken; the ratio is what matters for the demo.
"""

import json
import math
import re
from collections import OrderedDict


def estimate_tokens(text):
    """Rough token estimate: ceil(len(text) / 4). Labelled everywhere it is used."""
    if not text:
        return 0
    return math.ceil(len(text) / 4)


# ---------------------------------------------------------------------------
# Log compression
# ---------------------------------------------------------------------------

SEVERITY_KEEP = re.compile(
    r"\b(ERROR|FATAL|CRITICAL|EXCEPTION|TRACEBACK|FAILED|FAIL|WARN|WARNING)\b",
    re.IGNORECASE,
)
NOISE_PREFIX = re.compile(r"^(DEBUG|INFO)\b", re.IGNORECASE)


class LogCompressor:
    """Compresses agent-visible logs.

    Invariants (checked by tests):
    1. Every line matching SEVERITY_KEEP survives byte-for-byte.
    2. Consecutive duplicate lines collapse to '<line> ... (x N)'.
    3. If truncation is needed, the head and tail are kept; dropped middle
       lines are summarised as counts, never silently deleted.
    """

    def __init__(self, head_lines=5, tail_lines=15):
        self.head_lines = head_lines
        self.tail_lines = tail_lines

    def _collapse_repeats(self, lines):
        out = []
        i = 0
        while i < len(lines):
            j = i
            while j < len(lines) and lines[j] == lines[i]:
                j += 1
            count = j - i
            if count > 1:
                out.append("%s ... (repeated x%d)" % (lines[i], count))
            else:
                out.append(lines[i])
            i = j
        return out

    def compress(self, log_text):
        lines = log_text.splitlines()
        collapsed = self._collapse_repeats(lines)

        severe = [ln for ln in collapsed if SEVERITY_KEEP.search(ln)]
        noise = [ln for ln in collapsed if not SEVERITY_KEEP.search(ln)]

        kept_noise = noise
        dropped = 0
        if len(noise) > self.head_lines + self.tail_lines:
            kept_noise = (
                noise[: self.head_lines] + noise[-self.tail_lines :]
            )
            dropped = len(noise) - len(kept_noise)

        result = []
        result.extend(kept_noise[: self.head_lines])
        if dropped:
            result.append(
                "[... %d low-severity lines summarised away ...]" % dropped
            )
        result.extend(kept_noise[self.head_lines :])
        for ln in severe:
            if ln not in result:
                result.append(ln)
        return "\n".join(result)


# ---------------------------------------------------------------------------
# JSON tool-output compression
# ---------------------------------------------------------------------------

# Keys downstream code almost always needs. Everything here is preserved.
PRESERVE_KEYS = {
    "id", "ids", "status", "error", "errors", "message", "name", "type",
    "signature", "function", "tool", "action", "result", "success", "code",
    "path", "url", "timestamp", "count", "total",
}

# Verbose keys that are safe to drop from tool outputs shown to the agent.
DROP_KEYS = {
    "trace", "traceback", "verbose", "debug", "raw", "metadata", "headers",
    "config", "settings", "html", "body_html", "stack",
}


class JsonCompressor:
    """Compresses verbose JSON tool outputs.

    Strategy: drop known-noise keys, truncate long strings and long lists,
    keep every preserve-key and every structural key that remains. The output
    is still valid JSON with the same shape for kept keys, so downstream
    parsers keep working.
    """

    def __init__(self, max_str_chars=120, max_list_items=5):
        self.max_str_chars = max_str_chars
        self.max_list_items = max_list_items

    def _trim(self, value):
        if isinstance(value, str) and len(value) > self.max_str_chars:
            return value[: self.max_str_chars] + "... (truncated)"
        if isinstance(value, list):
            if len(value) > self.max_list_items:
                trimmed = [self._trim(v) for v in value[: self.max_list_items]]
                trimmed.append("... (%d more items)" % (len(value) - self.max_list_items))
                return trimmed
            return [self._trim(v) for v in value]
        if isinstance(value, dict):
            out = OrderedDict()
            for k, v in value.items():
                if k in DROP_KEYS:
                    continue
                out[k] = self._trim(v)
            return out
        return value

    def compress(self, payload):
        """Accepts a dict/list or a JSON string; returns a JSON string."""
        if isinstance(payload, str):
            payload = json.loads(payload)
        trimmed = self._trim(payload)
        return json.dumps(trimmed, indent=1, ensure_ascii=False)


# ---------------------------------------------------------------------------
# RAG chunk compression (sentence-level)
# ---------------------------------------------------------------------------

FILLER_PATTERNS = re.compile(
    r"\b(as (?:mentioned|noted|discussed) (?:above|earlier|previously)|"
    r"in (?:conclusion|summary)|it is (?:important|worth) (?:noting|to note)|"
    r"please note that|for more information,? (?:see|visit))\b",
    re.IGNORECASE,
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
WORD = re.compile(r"[a-z0-9]+")

# Common function words carry no signal for overlap scoring; filtering them
# stops queries like "why did the test fail" from matching junk sentences.
STOPWORDS = frozenset(
    "a an the and or of to in on for with by as at it its this that these those "
    "is are was were be been being have has had having do does did will would "
    "should could may might must shall can not no yes if then than so such "
    "from into over under about between through during each other some any "
    "all both few more most once only own same too very he she they we you i "
    "me him her us them my your his our their which who whom whose when where "
    "why what how got get".split()
)


class ChunkCompressor:
    """Sentence-level RAG context compression.

    Scores each sentence against the query by token overlap, demotes
    boilerplate/filler sentences, and keeps the top sentences under a token
    budget, preserving original order (order matters for coherence).
    """

    def __init__(self, keep_ratio=0.4, max_tokens=800):
        self.keep_ratio = keep_ratio
        self.max_tokens = max_tokens

    def _query_terms(self, query):
        return {w for w in WORD.findall(query.lower()) if w not in STOPWORDS}

    def compress(self, chunk, query):
        sentences = [s.strip() for s in SENTENCE_SPLIT.split(chunk) if s.strip()]
        if not sentences:
            return ""
        terms = self._query_terms(query)

        scored = []
        for idx, sent in enumerate(sentences):
            words = set(WORD.findall(sent.lower()))
            matched = words & terms
            overlap = len(matched) / max(len(terms), 1)
            density = len(matched) / max(len(words), 1)
            # Position bonus: first sentences usually carry the answer.
            position_bonus = 0.1 if idx < 2 else 0.0
            # Sentences touching several distinct query terms beat single hits.
            term_bonus = 0.05 * max(0, len(matched) - 1)
            filler_penalty = 0.5 if FILLER_PATTERNS.search(sent) else 0.0
            score = (
                0.5 * overlap
                + 0.4 * density
                + position_bonus
                + term_bonus
                - filler_penalty
            )
            scored.append((score, idx, sent))

        n_keep = max(1, int(len(sentences) * self.keep_ratio))
        top = sorted(scored, reverse=True)[:n_keep]
        top.sort(key=lambda t: t[1])  # restore original order

        picked = []
        budget = 0
        for _, _, sent in top:
            cost = estimate_tokens(sent)
            if budget + cost > self.max_tokens and picked:
                break
            picked.append(sent)
            budget += cost
        return " ".join(picked)


# ---------------------------------------------------------------------------
# Orchestrator: compress a whole agent context under a budget
# ---------------------------------------------------------------------------

# Illustrative input pricing: ~Haiku-class $1.00 / 1M input tokens (Oct 2026).
COST_PER_MTOK = 1.00


class AgentContextCompressor:
    """Compresses a list of context items and reports savings.

    Item format: {"kind": "log" | "json" | "chunk" | "text",
                  "content": str | dict, "query": optional str}

    Returns (compressed_items, stats) where stats carries token counts,
    compression ratio and an estimated cost saving.
    """

    def __init__(self, token_budget=4000):
        self.token_budget = token_budget
        self.log_c = LogCompressor()
        self.json_c = JsonCompressor()
        self.chunk_c = ChunkCompressor(max_tokens=token_budget)

    def _compress_one(self, item):
        kind = item.get("kind", "text")
        content = item.get("content", "")
        if kind == "log":
            return self.log_c.compress(str(content))
        if kind == "json":
            return self.json_c.compress(content)
        if kind == "chunk":
            return self.chunk_c.compress(str(content), item.get("query", ""))
        return str(content)

    def compress(self, items):
        compressed_items = []
        tokens_before = 0
        for item in items:
            raw = item.get("content", "")
            raw_text = raw if isinstance(raw, str) else json.dumps(raw)
            tokens_before += estimate_tokens(raw_text)
            out = self._compress_one(item)
            compressed_items.append({"kind": item.get("kind"), "content": out})

        tokens_after = sum(
            estimate_tokens(c["content"]) for c in compressed_items
        )
        saved = tokens_before - tokens_after
        stats = {
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "tokens_saved": saved,
            "compression_ratio": round(tokens_before / max(tokens_after, 1), 2),
            "reduction_pct": round(100.0 * saved / max(tokens_before, 1), 1),
            "est_cost_saved_per_1k_runs_usd": round(
                saved * COST_PER_MTOK / 1e6 * 1000, 4
            ),
            "within_budget": tokens_after <= self.token_budget,
        }
        return compressed_items, stats
