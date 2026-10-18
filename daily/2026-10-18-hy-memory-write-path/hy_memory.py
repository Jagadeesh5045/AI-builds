"""Hy Memory 2.0-style memory write path, toy edition (stdlib only).

On 16 Oct 2026 Tencent Hunyuan released Hy Memory 2.0: an open-source stack of
(1) Hy-Memory-30B-A3B, a *dedicated* memory-writer model that executes the write
path inside external memory systems, (2) the Hy-Memory framework, and (3) the
MemHub local app (all Apache 2.0). The core thesis: do not treat all agent memory
as one undifferentiated chat log. Separate working memory from code memory (and
chat memory), and give the write path to a small specialist (~3B active params)
instead of a general LLM.

This module implements a toy version of that write path in pure Python:

  MemoryWriter.execute_write(turn) -> decides, per turn,
    * which SCENE the content belongs to (chat / working / code),
    * which chunks are worth persisting (salience scoring),
    * which older chunks a new chunk SUPERSEDES (tombstoned, not deleted),
    * for code: a version entry with a line diff plus the rationale for the
      change (Hy Memory's version-evolution diff-chain: *why*, not just *what*).

It is a heuristic stand-in, not the real model: Hy-Memory-30B-A3B is trained
with Memory On-Policy Distillation plus attributed reinforcement learning on
memory (per-chunk checks, rewards attributed across chunks, arXiv paper dated
15 Oct 2026). That training is far beyond a demo; the heuristics here only model
the *protocol* of the write path, and the README says so explicitly.

Baselines in this module:
  * RawLog          - append everything to one undifferentiated chat log.
  * FlatWriter      - a general-LLM stand-in: writes everything, one flat list,
                      no scene separation, no supersede handling.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Scene(str, Enum):
    CHAT = "chat"
    WORKING = "working"
    CODE = "code"


# ---------------------------------------------------------------------------
# Chunk model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """One persisted memory unit."""
    id: int
    scene: Scene
    text: str
    turn: int
    salience: float
    entity: Optional[str] = None       # normalised (entity, attribute) key...
    attribute: Optional[str] = None    # ...used for supersede detection
    value: Optional[str] = None
    superseded_by: Optional[int] = None
    code_version: Optional[int] = None
    code_diff: Optional[str] = None
    code_rationale: Optional[str] = None


# ---------------------------------------------------------------------------
# Heuristic "writer model": classification + salience + fact extraction
# ---------------------------------------------------------------------------

_FACT_SPECS = [
    # (attribute, keyword regex, value regex applied after the keyword)
    ("port", re.compile(r"\bport\b", re.I), re.compile(r"(\d{2,5})")),
    ("deadline", re.compile(r"\bdeadline\b", re.I),
     re.compile(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
                re.I)),
    ("eta", re.compile(r"\bETA\b"),
     re.compile(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
                re.I)),
    ("retries", re.compile(r"\bretries\b", re.I), re.compile(r"(\d+)")),
    ("timeout", re.compile(r"\btimeout\b", re.I), re.compile(r"(\d+)\s*s", re.I)),
    ("version", re.compile(r"\bversion\b", re.I), re.compile(r"(v?[\d.]+)")),
]

_DETERMINERS = ("the", "a", "an", "my", "our", "this", "that")


def _entity_before(text: str, keyword_idx: int, window: int = 3) -> str:
    """Noun phrase just before the attribute keyword, normalised.

    'The staging database port ...' and 'Update: the staging database
    port ...' both normalise to 'staging database', so a later value for
    the same attribute is recognised as the SAME fact and supersedes it.
    """
    words = re.findall(r"[a-zA-Z]+", text[:keyword_idx].lower())
    phrase = words[-window:]
    while phrase and phrase[0] in _DETERMINERS:
        phrase.pop(0)
    return " ".join(phrase) if phrase else "global"

_CODE_HINTS = ("def ", "class ", "import ", "```", "->", "return ", "for ",
               "while ", "if __name__")
_RATIONALE_HINTS = ("because", "why:", "rationale", "so that", "to avoid",
                    "fixes", "instead of", "change:")
_LESSON_HINTS = ("lesson", "always ", "never ", "rule of thumb", "from now on",
                 "note to self")
_CHAT_HINTS = ("hi", "hello", "thanks", "thank you", "bye", "how are you",
               "weather", "weekend")


def _classify_scene(text: str, role: str) -> Scene:
    low = text.lower()
    if role == "code" or sum(h in low for h in _CODE_HINTS) >= 2 or "```" in text:
        return Scene.CODE
    if any(h in low for h in _LESSON_HINTS) or "task" in low or "todo" in low:
        return Scene.WORKING
    if any(h in low for h in _CHAT_HINTS) and len(text.split()) < 12:
        return Scene.CHAT
    # Default: task-oriented statements go to working memory, chit-chat to chat.
    if re.search(r"\b(is|are|will|should|must|set|use|deploy|run)\b", low):
        return Scene.WORKING
    return Scene.CHAT


def _extract_fact(text: str) -> Optional[Tuple[str, str, str]]:
    """Return (entity, attribute, value) for the first recognised fact.

    Entity is the normalised noun phrase before the attribute keyword, so
    'The staging database port is 5432' and 'Update: the staging database
    port is 5433 now' resolve to the same (entity, attribute) pair. Toy
    heuristic; the real Hy-Memory-30B-A3B learns this mapping from data.
    """
    for attribute, kw_pattern, val_pattern in _FACT_SPECS:
        kw = kw_pattern.search(text)
        if not kw:
            continue
        val = val_pattern.search(text, kw.end())
        if not val:
            continue
        return _entity_before(text, kw.start()), attribute, val.group(1)
    return None


def _salience(text: str, role: str, scene: Scene) -> float:
    """Toy salience score in [0, 1]: would this chunk help answer a FUTURE
    question? Stands in for the attributed-RL reward signal."""
    low = text.lower()
    score = 0.25
    if _extract_fact(text):
        score += 0.35                      # checkable fact: high write value
    if any(h in low for h in _LESSON_HINTS):
        score += 0.30                      # reusable lesson
    if re.search(r"\d", text):
        score += 0.10                      # concrete numbers survive time
    if role == "user":
        score += 0.05
    if scene is Scene.CODE:
        score += 0.15
    if len(text.split()) < 4 and scene is Scene.CHAT:
        score -= 0.20                      # chit-chat is rarely worth writing
    return max(0.0, min(1.0, score))


def _extract_rationale(text: str) -> Optional[str]:
    low = text.lower()
    for hint in _RATIONALE_HINTS:
        i = low.find(hint)
        if i != -1:
            sentence = text[max(0, text.rfind(".", 0, i) + 1):]
            sentence = sentence.split(".")[0].strip()
            return sentence[:160] if sentence else None
    return None


# ---------------------------------------------------------------------------
# The write path
# ---------------------------------------------------------------------------

@dataclass
class WriteStats:
    chunks_written: int = 0
    chunks_skipped: int = 0
    supersedes: int = 0
    tokens_written: int = 0


class MemoryWriter:
    """Executes the write path into three separated scenes."""

    SALIENCE_THRESHOLD = 0.35

    def __init__(self) -> None:
        self.scenes: Dict[Scene, List[Chunk]] = {s: [] for s in Scene}
        self.stats = WriteStats()
        self._next_id = 1
        self._code_versions: List[str] = []

    # -- public -----------------------------------------------------------
    def execute_write(self, turn: int, role: str, text: str) -> List[Chunk]:
        """Write-path for one turn. Returns the chunks actually persisted."""
        scene = _classify_scene(text, role)
        if scene is Scene.CODE:
            return self._write_code(turn, text)
        chunks = self._split_sentences(text)
        written: List[Chunk] = []
        for sentence in chunks:
            sal = _salience(sentence, role, scene)
            if sal < self.SALIENCE_THRESHOLD:
                self.stats.chunks_skipped += 1
                continue
            fact = _extract_fact(sentence)
            chunk = Chunk(
                id=self._next_id, scene=scene, text=sentence.strip(),
                turn=turn, salience=round(sal, 2),
                entity=fact[0] if fact else None,
                attribute=fact[1] if fact else None,
                value=fact[2] if fact else None,
            )
            self._next_id += 1
            if fact:
                self._supersede(chunk)
            self.scenes[scene].append(chunk)
            self.stats.chunks_written += 1
            self.stats.tokens_written += len(sentence.split())
            written.append(chunk)
        return written

    # -- internals --------------------------------------------------------
    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in (s.strip() for s in parts) if p]

    def _supersede(self, new: Chunk) -> None:
        """Tombstone older live chunks with the same (entity, attribute)."""
        for scene_chunks in self.scenes.values():
            for old in scene_chunks:
                if (old.superseded_by is None and old.entity == new.entity
                        and old.attribute == new.attribute
                        and old.value != new.value):
                    old.superseded_by = new.id
                    self.stats.supersedes += 1

    def _write_code(self, turn: int, text: str) -> List[Chunk]:
        code = self._strip_fences(text)
        version = len(self._code_versions) + 1
        diff = ""
        if self._code_versions:
            prev = self._code_versions[-1].splitlines()
            diff = "\n".join(difflib.unified_diff(
                prev, code.splitlines(),
                fromfile=f"v{version - 1}", tofile=f"v{version}",
                lineterm=""))
        self._code_versions.append(code)
        chunk = Chunk(
            id=self._next_id, scene=Scene.CODE,
            text=f"code v{version} ({len(code.splitlines())} lines)",
            turn=turn, salience=0.9, code_version=version,
            code_diff=diff or "(initial version)",
            code_rationale=_extract_rationale(text) or "(no rationale given)",
        )
        self._next_id += 1
        self.scenes[Scene.CODE].append(chunk)
        self.stats.chunks_written += 1
        self.stats.tokens_written += len(code.split())
        return [chunk]

    @staticmethod
    def _strip_fences(text: str) -> str:
        lines = [ln for ln in text.splitlines()
                 if not ln.strip().startswith("```")]
        return "\n".join(lines).strip()

    # -- read path (query time) -------------------------------------------
    def retrieve(self, query: str, top_k: int = 4) -> List[Chunk]:
        """Scene-targeted retrieval: route the query to a scene first, then
        rank live (non-superseded) chunks by token overlap."""
        qlow = query.lower()
        if any(w in qlow for w in ("code", "function", "version", "change",
                                   "diff", "why")):
            scenes = [Scene.CODE, Scene.WORKING]
        elif any(w in qlow for w in ("lesson", "rule", "always", "never")):
            scenes = [Scene.WORKING, Scene.CHAT]
        else:
            scenes = [Scene.WORKING, Scene.CHAT, Scene.CODE]
        qterms = set(re.findall(r"[a-z0-9]+", qlow))
        scored: List[Tuple[float, Chunk]] = []
        for scene in scenes:
            for c in self.scenes[scene]:
                if c.superseded_by is not None:
                    continue
                cterms = set(re.findall(r"[a-z0-9]+", c.text.lower()))
                overlap = len(qterms & cterms)
                if overlap:
                    scored.append((overlap + c.salience, c))
        scored.sort(key=lambda t: -t[0])
        return [c for _, c in scored[:top_k]]

    def live_counts(self) -> Dict[str, int]:
        return {s.value: sum(1 for c in chunks if c.superseded_by is None)
                for s, chunks in self.scenes.items()}


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class RawLog:
    """Baseline 1: one undifferentiated chat log; grep retrieval."""

    def __init__(self) -> None:
        self.lines: List[str] = []

    def write(self, role: str, text: str) -> None:
        self.lines.append(f"{role}: {text}")

    def retrieve(self, query: str, top_k: int = 4) -> List[str]:
        qterms = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []
        for ln in self.lines:
            lterms = set(re.findall(r"[a-z0-9]+", ln.lower()))
            overlap = len(qterms & lterms)
            if overlap:
                scored.append((overlap, ln))
        scored.sort(key=lambda t: -t[0])
        return [ln for _, ln in scored[:top_k]]


class FlatWriter:
    """Baseline 2: general-LLM stand-in. Writes everything into one flat
    memory list: no scene separation, no supersede handling, no diff-chains."""

    def __init__(self) -> None:
        self.chunks: List[str] = []

    def write(self, text: str) -> None:
        self.chunks.append(text)

    def retrieve(self, query: str, top_k: int = 4) -> List[str]:
        qterms = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []
        for ch in self.chunks:
            cterms = set(re.findall(r"[a-z0-9]+", ch.lower()))
            overlap = len(qterms & cterms)
            if overlap:
                scored.append((overlap, ch))
        scored.sort(key=lambda t: -t[0])
        return [ch for _, ch in scored[:top_k]]
