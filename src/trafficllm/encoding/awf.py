"""AWF trace encoding for GPT-2.

AWF raw data: each trace is a 1-D array of 5000 packet directions (1=out, -1=in).

Upstream encoding (YasodGinige/TrafficLLM ``save_csv_AWF``):
  1. X = X + 1          → values become {0, 2} (or other non-negative ints)
  2. X = X.astype(str)   → element-wise string conversion
  3. Pair consecutive elements: ``''.join(x[j:j+2])`` for j in range(0, len(x), 2)
  4. Convert each pair from base-3 to decimal string: ``str(int(pair, 3))``
  5. Join pairs with spaces (GPT-2 mode).

Bug in upstream: the loop guard ``if j < len(x)-2`` causes the **last pair**
(j == len(x)-2) to be silently skipped. Reference replicates this bug;
corrected includes the last pair.

Both ``encode_awf_trace_reference`` and ``encode_awf_trace_corrected`` are
**label-free**: they take only the raw trace array. AWF upstream encoding does
not use labels (unlike DC), so reference and corrected differ only in whether
the last pair is kept.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _shift_and_pair(
    values: Sequence[float] | np.ndarray,
    *,
    drop_last_pair: bool,
) -> list[str]:
    """Shift by +1, convert to str, pair consecutively, decode base-3."""
    arr = np.asarray(values, dtype=np.int64) + 1
    strs = arr.astype(str)
    n = len(strs)
    pairs: list[str] = []
    for j in range(0, n, 2):
        if drop_last_pair and j >= n - 2:
            break
        pair_str = "".join(strs[j : j + 2])
        pairs.append(str(int(pair_str, 3)))
    return pairs


def encode_awf_trace_reference(raw_trace: Sequence[float] | np.ndarray) -> str:
    """Encode one AWF trace replicating the upstream bug (last pair dropped).

    Parameters
    ----------
    raw_trace : array-like of shape (5000,)
        Raw packet directions / sizes as stored in X_train.npy etc.

    Returns
    -------
    str
        Space-separated base-3-decoded pairs, last pair missing (upstream bug).
    """
    pairs = _shift_and_pair(raw_trace, drop_last_pair=True)
    if not pairs:
        raise ValueError("raw_trace too short to produce any pair")
    return " ".join(pairs)


def encode_awf_trace_corrected(raw_trace: Sequence[float] | np.ndarray) -> str:
    """Encode one AWF trace with corrected protocol (last pair kept).

    Parameters
    ----------
    raw_trace : array-like of shape (5000,)

    Returns
    -------
    str
        Space-separated base-3-decoded pairs, all pairs included.
    """
    pairs = _shift_and_pair(raw_trace, drop_last_pair=False)
    if not pairs:
        raise ValueError("raw_trace too short to produce any pair")
    return " ".join(pairs)
