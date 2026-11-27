"""DF trace encoding for GPT-2.

DF raw data: each trace is a 1-D array of 5000 packet directions (like AWF).
Upstream uses the **same** encoding as AWF (``save_csv_AWF``): shift +1,
pair consecutive strings, convert from base-3. The last-pair-dropped bug
applies identically.

DF differs from AWF only in data loading (pickle, ``encoding='latin1'``)
and class splitting (first 60 of 95 classes are known in upstream).
"""

from __future__ import annotations

from trafficllm.encoding.awf import (
    encode_awf_trace_corrected as encode_df_trace_corrected,
    encode_awf_trace_reference as encode_df_trace_reference,
)

__all__ = ["encode_df_trace_reference", "encode_df_trace_corrected"]
