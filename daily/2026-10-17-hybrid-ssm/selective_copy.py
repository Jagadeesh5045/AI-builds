"""Selective-copy task data generator.

The canonical task from the Mamba paper (Gu & Dao, 2023): a sequence is
filled with noise tokens plus a few (marker, needle) pairs at random
positions. Appended queries ask "what token followed marker m?" -- the model
must *selectively* remember the needles and ignore everything else. Attention
can solve it by learning the right retrieval pattern; a selective SSM can
solve it by gating noise out of its state.
"""

import numpy as np

VOCAB = 16            # tokens 0..15
N_DATA = 8            # data/needle tokens: 0..7
MARKERS = [8, 9, 10]  # marker tokens
N_PAIRS = 3           # (marker, needle) pairs per sequence


def make_batch(batch, seq_data_len, rng):
    """Returns (ids (B, L), query_pos (B, Q), targets (B, Q)).

    Sequence layout: [data region of length seq_data_len] + [Q query tokens].
    Query token i is the marker of pair i; target i is its needle.
    """
    Q = N_PAIRS
    L = seq_data_len + Q
    ids = np.zeros((batch, L), dtype=np.int64)
    targets = np.zeros((batch, Q), dtype=np.int64)
    for b in range(batch):
        data = rng.integers(0, N_DATA, size=seq_data_len)
        # non-adjacent pair slots: a needle must never be overwritten by the
        # next pair's marker (that would make the label unlearnable noise)
        while True:
            positions = rng.choice(seq_data_len - 1, size=N_PAIRS, replace=False)
            positions.sort()
            if np.all(np.diff(positions) > 1):
                break
        for i, (m, pos) in enumerate(zip(MARKERS, positions)):
            needle = int(rng.integers(0, N_DATA))
            data[pos] = m
            data[pos + 1] = needle
            ids[b, seq_data_len + i] = m
            targets[b, i] = needle
        ids[b, :seq_data_len] = data
    query_pos = np.arange(seq_data_len, L)[None, :].repeat(batch, axis=0)
    return ids, query_pos, targets
