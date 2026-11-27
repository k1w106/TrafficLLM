"""Tests for AWF encoding (reference vs corrected)."""

from __future__ import annotations

import numpy as np
import pytest

from trafficllm.encoding.awf import (
    _shift_and_pair,
    encode_awf_trace_corrected,
    encode_awf_trace_reference,
)


def _make_trace(vals: list[int], length: int = 10) -> np.ndarray:
    """Pad/truncate vals to *length* with zeros."""
    out = np.zeros(length, dtype=np.int64)
    out[: len(vals)] = vals
    return out


class TestShiftAndPair:
    def test_basic_pairing(self):
        vals = np.array([1, -1, 1, 1], dtype=np.int64)
        # +1 → [2, 0, 2, 2]; str → ["2","0","2","2"]
        # pairs: "20" → int("20",3)=6, "22" → int("22",3)=8
        pairs = _shift_and_pair(vals, drop_last_pair=False)
        assert pairs == ["6", "8"]

    def test_drop_last_pair(self):
        vals = np.array([1, -1, 1, 1], dtype=np.int64)
        # 4 elements → last pair at j=2, len-2=2 → skipped
        pairs = _shift_and_pair(vals, drop_last_pair=True)
        assert pairs == ["6"]

    def test_drop_last_pair_six_elements(self):
        vals = np.array([1, -1, 1, 1, -1, -1], dtype=np.int64)
        # +1 → [2,0,2,2,0,0]; pairs: j=0 "20"=6, j=2 "22"=8, j=4 "00"=0
        # drop_last: j=4 >= 6-2=4 → skip
        pairs_ref = _shift_and_pair(vals, drop_last_pair=True)
        pairs_corr = _shift_and_pair(vals, drop_last_pair=False)
        assert pairs_ref == ["6", "8"]
        assert pairs_corr == ["6", "8", "0"]

    def test_odd_length(self):
        vals = np.array([1, -1, 1], dtype=np.int64)
        # +1 → [2,0,2]; pairs: j=0 "20"=6, j=2 "2" (single char)
        # int("2",3) = 2
        pairs = _shift_and_pair(vals, drop_last_pair=False)
        assert pairs == ["6", "2"]


class TestReferenceVsCorrected:
    def test_reference_drops_last_pair(self):
        trace = _make_trace([1, -1] * 5, length=10)
        ref = encode_awf_trace_reference(trace)
        corr = encode_awf_trace_corrected(trace)
        ref_tokens = ref.split()
        corr_tokens = corr.split()
        assert len(corr_tokens) == len(ref_tokens) + 1

    def test_reference_prefix_matches_corrected(self):
        trace = _make_trace([1, -1, 1, 1, -1, -1], length=6)
        ref = encode_awf_trace_reference(trace)
        corr = encode_awf_trace_corrected(trace)
        assert corr.startswith(ref)

    def test_label_free(self):
        """Encoding functions accept only raw_trace, no label parameter."""
        import inspect

        for fn in [encode_awf_trace_reference, encode_awf_trace_corrected]:
            sig = inspect.signature(fn)
            params = [
                p
                for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
            ]
            assert len(params) == 1, f"{fn.__name__} should take exactly 1 required arg"


class TestOnRealDataSample:
    """Verify encoding on a small slice of real AWF data if available."""

    @pytest.fixture
    def real_trace(self):
        path = "data/raw/trafficllm_reference/trafficllm_data/AWF/X_train.npy"
        try:
            X = np.load(path, mmap_mode="r")
        except FileNotFoundError:
            pytest.skip("AWF data not available")
        return np.array(X[0])

    def test_real_trace_encodes(self, real_trace):
        ref = encode_awf_trace_reference(real_trace)
        corr = encode_awf_trace_corrected(real_trace)
        assert isinstance(ref, str)
        assert isinstance(corr, str)
        ref_toks = ref.split()
        corr_toks = corr.split()
        # 5000 elements → 2500 pairs; reference drops last → 2499
        assert len(ref_toks) == 2499
        assert len(corr_toks) == 2500
        # all tokens should be valid decimal strings
        for t in corr_toks:
            int(t)

    def test_upstream_equivalence(self, real_trace):
        """Verify our reference matches upstream's try_to_hex logic."""
        x = (real_trace + 1).astype(int).astype(str)
        upstream_tokens = []
        for j in range(0, len(x), 2):
            l = "".join(x[j : j + 2])
            if j < len(x) - 2:
                if j != len(x) - 2:
                    upstream_tokens.append(str(int(l, 3)))
        upstream_text = " ".join(upstream_tokens)

        ref = encode_awf_trace_reference(real_trace)
        assert ref == upstream_text
