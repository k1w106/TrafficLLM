"""USTC trace encoding for GPT-2.

A raw USTC trace is a space-separated sequence of 4-hex-character groups
(e.g. ``"e742 421d 1dc8 ..."``); consecutive groups overlap by one byte.

The upstream reference implementation (YasodGinige/TrafficLLM) feeds GPT-2 a
character-level string where every character is separated by a space::

    ' '.join(raw_trace.replace(" ", ""))

The encoder below is strictly label-free: it receives the raw trace only, so
the ground-truth class can never leak into the model input.
"""

from __future__ import annotations


def encode_trace_gpt2(raw_trace: str) -> str:
    """Encode one raw USTC trace into the character-level string fed to GPT-2.

    Parameters
    ----------
    raw_trace : str
        Raw trace exactly as stored in ``traces.npy``.

    Returns
    -------
    str
        The trace characters separated by single spaces.
    """
    if not isinstance(raw_trace, str):
        raise TypeError(f"raw_trace must be str, got {type(raw_trace).__name__}")

    compact = "".join(raw_trace.split())
    if not compact:
        raise ValueError("raw_trace is empty")

    return " ".join(compact)
