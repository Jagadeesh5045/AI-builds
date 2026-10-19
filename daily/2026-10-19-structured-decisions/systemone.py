"""System One-style structured decisions engine (standard library only).

Implements the request/response contract of vLLM v0.32's
``POST /v1/systemone`` endpoint (released 2026-10-18; PR #59299) against a
deterministic local scoring backend, so the contract can be exercised without
a GPU or a vLLM install.

Contract (verified against the ``vllm-sr/d3`` Hugging Face model card and the
LLM Gateway "System One: Typed Decisions" changelog):

  request  = {"model": str, "state": str | dict,
              "questions": {question_id: question}}
  question = {"type": "choice" | "noul" | "score", "instructions": str, ...}
    choice: {"criteria": {label: description}}          (1-255 options)
    noul:   {}                                          (yes/no question)
    score:  {"levels": [ordered labels],                 (2-10 levels)
             "criteria": {level: description} (optional)}
  response = {"object": "structured_decision", "id": str, "created": int,
              "model": str, "answers": {question_id: answer},
              "usage": {"input_tokens": int, "output_tokens": 0}}

  choice answer: {"type": "choice", "choice": label,
                  "probabilities": {label: p}, "confidence": p_chosen}
  noul answer:   {"type": "noul", "noul": p_yes,
                  "probabilities": {"yes": p, "no": 1 - p},
                  "confidence": max(p_yes, p_no)}
  score answer:  {"type": "score", "score": float, "legend": [...],
                  "probabilities": {label: p}, "confidence": max(p)}

The point of the real endpoint: plenty of model calls exist only to make a
decision (route this ticket, score this passage). Instead of streaming prose
back and parsing JSON, you name the questions and every answer comes back as
a value your code can branch on, with a probability attached -- and because
nothing is generated, ``output_tokens`` is 0.

HONEST-PROXY NOTE: a real System One decision model reads label logprobs off
the model weights in one constrained pass. This demo scores stemmed keyword
evidence between the ``state`` text and each option's criteria description,
then normalises with a softmax. It is deterministic and explainable, and it
is labelled a stand-in everywhere it matters (including this docstring).
"""

from __future__ import annotations

import json
import math
import re
import time
import uuid

VALID_TYPES = ("choice", "noul", "score")
MAX_CHOICE_OPTIONS = 255


class DecisionError(Exception):
    """A 4xx-style request problem; ``status`` is the HTTP code to return."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


_STOPWORDS = frozenset(
    "a an the and or of to in on for with is are was were be been being as at by "
    "from into over after before between it its this that these those what which "
    "who whom how do does did done have has had having will would can could shall "
    "should may might must i you he she we they them his her their our your my me "
    "him us if then than so such only just very too also any all each other more "
    "most some there here when where why while".split()
)
_NEGATIONS = frozenset({"no", "not", "never", "without", "cannot", "none", "neither", "nor"})


def _tokens(text) -> list:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _stem(token: str) -> str:
    """Tiny deterministic stemmer so write/writing, charge/charges match."""
    t = token
    if len(t) > 4:
        for suffix in ("ing", "ed", "es"):
            if t.endswith(suffix) and len(t) - len(suffix) >= 3:
                t = t[: -len(suffix)]
                break
        else:
            if t.endswith("s") and len(t) >= 5:
                t = t[:-1]
            elif t.endswith("e") and len(t) >= 5:
                t = t[:-1]
    return t


def _content_stems(text) -> list:
    return [_stem(t) for t in _tokens(text) if t not in _STOPWORDS]


def _overlap_count(state_stems: set, phrase: str) -> int:
    terms = set(_content_stems(phrase))
    return len(terms & state_stems)


def _softmax(logits: list) -> list:
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _ask_choice(state_stems: set, question: dict) -> dict:
    criteria = question.get("criteria")
    if not isinstance(criteria, dict) or not criteria:
        raise DecisionError("choice questions require a non-empty 'criteria' map")
    labels = list(criteria.keys())
    if any(not isinstance(label, str) or not label for label in labels):
        raise DecisionError("choice criteria labels must be non-empty strings")
    if len(labels) > MAX_CHOICE_OPTIONS:
        raise DecisionError(f"choice questions take at most {MAX_CHOICE_OPTIONS} options")
    logits = [1.0 + 2.0 * _overlap_count(state_stems, criteria[label]) for label in labels]
    probs = _softmax(logits)
    best = max(range(len(labels)), key=lambda i: probs[i])
    return {
        "type": "choice",
        "choice": labels[best],
        "probabilities": {label: p for label, p in zip(labels, probs)},
        "confidence": probs[best],
    }


def _ask_noul(state_text: str, state_stems: set, question: dict) -> dict:
    instructions = question.get("instructions", "")
    q_terms = set(_content_stems(instructions))
    yes_hits = len(q_terms & state_stems)
    neg_hits = sum(1 for t in _tokens(state_text) if t in _NEGATIONS)
    if "n't" in str(state_text).lower():
        neg_hits += 1
    p_yes = _softmax([1.0 + 2.0 * yes_hits, 1.0 + 1.5 * neg_hits])[0]
    p_no = 1.0 - p_yes
    return {
        "type": "noul",
        "noul": p_yes,
        "probabilities": {"yes": p_yes, "no": p_no},
        "confidence": max(p_yes, p_no),
    }


def _ask_score(state_stems: set, question: dict) -> dict:
    levels = question.get("levels")
    if not isinstance(levels, list) or not all(isinstance(l, str) and l for l in levels):
        raise DecisionError("score questions require 'levels' as a list of labels")
    if len(set(levels)) != len(levels):
        raise DecisionError("score 'levels' must be unique")
    if not 2 <= len(levels) <= 10:
        raise DecisionError("score questions take two to ten ordered levels")
    criteria = question.get("criteria") or {}
    if not isinstance(criteria, dict):
        raise DecisionError("score 'criteria' must be a map when present")
    logits = [1.0 + 2.0 * _overlap_count(state_stems, criteria.get(level, level)) for level in levels]
    probs = _softmax(logits)
    score = sum(i * p for i, p in enumerate(probs))  # may land between levels
    return {
        "type": "score",
        "score": score,
        "legend": list(levels),
        "probabilities": {label: p for label, p in zip(levels, probs)},
        "confidence": max(probs),
    }


def decide(request: dict) -> tuple:
    """Validate a System One request; return ``(status_code, response_dict)``."""
    if not isinstance(request, dict):
        raise DecisionError("request body must be a JSON object")
    if "instructions" in request:
        # Mirrors the real endpoint: the prompt is fixed, so a top-level
        # "instructions" field is rejected with 400.
        raise DecisionError("top-level 'instructions' is not accepted")
    model = request.get("model")
    state = request.get("state")
    questions = request.get("questions")
    if not isinstance(model, str) or not model:
        raise DecisionError("'model' is required")
    if not isinstance(state, (str, dict)) or not state:
        raise DecisionError("'state' is required")
    if not isinstance(questions, dict) or not questions:
        raise DecisionError("'questions' must be a non-empty object")

    state_text = state if isinstance(state, str) else json.dumps(state, sort_keys=True)
    state_stems = set(_content_stems(state_text))
    input_tokens = len(_tokens(state_text))

    answers = {}
    for qid, question in questions.items():
        if not isinstance(question, dict):
            raise DecisionError(f"question '{qid}' must be an object")
        qtype = question.get("type")
        if qtype not in VALID_TYPES:
            raise DecisionError(f"question '{qid}': unknown type '{qtype}'")
        instructions = question.get("instructions")
        if not isinstance(instructions, str) or not instructions:
            raise DecisionError(f"question '{qid}': 'instructions' is required")
        input_tokens += len(_tokens(instructions))
        input_tokens += len(_tokens(json.dumps(question.get("criteria", {}), sort_keys=True)))
        input_tokens += len(_tokens(json.dumps(question.get("levels", []))))
        if qtype == "choice":
            answers[qid] = _ask_choice(state_stems, question)
        elif qtype == "noul":
            answers[qid] = _ask_noul(state_text, state_stems, question)
        else:
            answers[qid] = _ask_score(state_stems, question)

    return 200, {
        "object": "structured_decision",
        "id": "sysone-" + uuid.uuid4().hex[:12],
        "created": int(time.time()),
        "model": model,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }
