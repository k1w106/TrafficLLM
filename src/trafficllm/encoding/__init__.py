"""Label-free encoders turning raw traffic traces into model input."""

from trafficllm.encoding.ustc import encode_trace_gpt2

__all__ = ["encode_trace_gpt2"]
