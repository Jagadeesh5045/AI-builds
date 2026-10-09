"""Deterministic citation verifier.

Parses citations in `path/to/file.py:LINE` format from draft answer text,
opens the cited file, and checks that the cited line exists and that the
line's tokens overlap the claim text.

No model calls. No learned weights. The only heuristic is a documented
token-overlap threshold (OVERLAP_THRESHOLD). Output per citation is one of:
SUPPORTED, UNSUPPORTED, MALFORMED, each with a human-readable reason.
"""

import os
import re

# Strict citation pattern: some/path.py:123
CITATION_RE = re.compile(r"([A-Za-z0-9_\-./]+\.py):(\d+)")

# Loose pattern used to catch malformed attempts like "auth.py" (no line)
# or "auth.py:abc" (non-numeric line) so they can be labeled MALFORMED
# instead of silently ignored.
LOOSE_CITATION_RE = re.compile(r"([A-Za-z0-9_\-./]*\.py)(?::([A-Za-z0-9]+))?")

STOPWORDS = frozenset(
    "a an the and or of to in on for with by as is are was were be been "
    "it its this that these those we you they he she him her them his our "
    "your their from at into over after before between through during each "
    "other more most such no not only own same so than too very can will "
    "just don what which who whom how when where why does did do have has "
    "had having i me my us there here then also into out up down all any "
    "s t re ll ve".split()
)

# A citation is SUPPORTED when at least this fraction of the claim's
# content tokens appear on the cited line. Chosen so that a claim quoting
# or closely paraphrasing the line passes, while a claim about something
# else on a nearby line fails. Documented here, not tuned per fixture.
OVERLAP_THRESHOLD = 0.5


class MalformedCitation(ValueError):
    """Raised when a citation string does not match path.py:LINE."""


def tokenize(text):
    """Lowercase alphanumeric tokens with stopwords removed."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def parse_citation(raw):
    """Parse 'path/to/file.py:123' into (path, line_number).

    Raises MalformedCitation for anything else, including a bare
    'file.py' with no line or a non-numeric line suffix.
    """
    m = CITATION_RE.fullmatch(raw.strip())
    if not m:
        raise MalformedCitation("expected format path/to/file.py:LINE, got %r" % raw)
    return m.group(1), int(m.group(2))


def find_citations(text):
    """Return every citation-like substring in text, well-formed or not."""
    return [m.group(0) for m in LOOSE_CITATION_RE.finditer(text)]


def verify(claim, citation, root="."):
    """Verify one (claim, citation) pair.

    Returns a dict with keys: status (SUPPORTED/UNSUPPORTED/MALFORMED),
    reason (human readable), overlap (float or None), line (the cited
    line text or None).
    """
    try:
        path, line_no = parse_citation(citation)
    except MalformedCitation as e:
        return {
            "status": "MALFORMED",
            "reason": "malformed citation: %s" % e,
            "overlap": None,
            "line": None,
        }

    full_path = os.path.join(root, path)
    if not os.path.isfile(full_path):
        return {
            "status": "MALFORMED",
            "reason": "cited file does not exist: %s" % path,
            "overlap": None,
            "line": None,
        }

    with open(full_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    if line_no < 1 or line_no > len(lines):
        return {
            "status": "UNSUPPORTED",
            "reason": "line %d out of range (file has %d lines)" % (line_no, len(lines)),
            "overlap": None,
            "line": None,
        }

    line_text = lines[line_no - 1]
    claim_tokens = tokenize(claim)
    if not claim_tokens:
        return {
            "status": "UNSUPPORTED",
            "reason": "claim has no content tokens to match",
            "overlap": 0.0,
            "line": line_text,
        }

    line_tokens = set(tokenize(line_text))
    overlap = sum(1 for t in claim_tokens if t in line_tokens) / len(claim_tokens)

    if overlap >= OVERLAP_THRESHOLD:
        return {
            "status": "SUPPORTED",
            "reason": "line exists and token overlap %.2f >= %.2f"
            % (overlap, OVERLAP_THRESHOLD),
            "overlap": round(overlap, 3),
            "line": line_text,
        }
    return {
        "status": "UNSUPPORTED",
        "reason": "line exists but token overlap %.2f < %.2f"
        % (overlap, OVERLAP_THRESHOLD),
        "overlap": round(overlap, 3),
        "line": line_text,
    }
